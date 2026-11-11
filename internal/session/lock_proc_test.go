package session

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pulseaiclub/phi/internal/llm"
)

// Real multi-process coverage. The helper is the same test binary re-execed
// with a marker env; it owns one session and blocks on stdin so the parent can
// drive graceful release or a hard kill.
const (
	lockHelperEnv     = "PHI_SESSION_LOCK_HELPER"
	lockHelperFileEnv = "PHI_SESSION_LOCK_HELPER_FILE"
)

// TestLockHelperProcess is the subprocess entry point. Skipped in normal runs.
func TestLockHelperProcess(t *testing.T) {
	if os.Getenv(lockHelperEnv) != "1" {
		t.Skip("subprocess helper entry point")
	}
	path := os.Getenv(lockHelperFileEnv)
	m, err := OpenOwnedSession(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if _, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "helper turn"}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := m.SetModel("helper-model"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println("ready")
	_ = os.Stdout.Sync()
	// Hold ownership until the parent closes stdin (graceful) or kills us.
	_, _ = io.Copy(io.Discard, os.Stdin)
	_ = m.Close()
	os.Exit(0)
}

// helperSessionFile writes a pre-ownership-era fixture so the helper can open
// an existing session rather than create a randomly named one.
func helperSessionFile(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "2025-01-01T00-00-00_shared.jsonl")
	fixture := []byte(`{"type":"session","id":"shared","cwd":"/tmp"}` + "\n" +
		`{"type":"EntryMessage","id":"m1","parentID":null,"message":{"role":"user","content":"seed question"}}` + "\n")
	require.NoError(t, os.WriteFile(path, fixture, 0o600))
	return path
}

func startLockHelper(t *testing.T, sessionPath string) (*exec.Cmd, io.WriteCloser) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestLockHelperProcess$", "-test.v")
	cmd.Env = append(os.Environ(), lockHelperEnv+"=1", lockHelperFileEnv+"="+sessionPath)
	stdout, err := cmd.StdoutPipe()
	require.NoError(t, err)
	stdin, err := cmd.StdinPipe()
	require.NoError(t, err)
	// Inline stderr: the helper only writes there on failure, and sharing a
	// buffer with exec's copy goroutine would race the ready-failure path.
	cmd.Stderr = os.Stderr
	require.NoError(t, cmd.Start())
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})

	scanner := bufio.NewScanner(stdout)
	ready := false
	for scanner.Scan() {
		if scanner.Text() == "ready" {
			ready = true
			break
		}
	}
	require.True(t, ready, "helper never became ready")
	return cmd, stdin
}

func TestCrossProcessContentionAndGracefulRelease(t *testing.T) {
	path := helperSessionFile(t)
	cmd, stdin := startLockHelper(t, path)

	// While the helper is alive, a second writer is refused, with holder info.
	_, err := OpenOwnedSession(path)
	require.ErrorIs(t, err, ErrSessionLocked)
	assert.Contains(t, err.Error(), "held by pid")
	assert.Contains(t, err.Error(), fmt.Sprintf("pid %d", cmd.Process.Pid))

	// Read-only replay and listing must stay usable.
	view, err := OpenSession(path)
	require.NoError(t, err)
	assert.Equal(t, "helper-model", view.ModelName())
	assert.Contains(t, messageContents(view.BuildContext()), "assistant:helper turn")
	dir := filepath.Dir(path)
	list, err := ListSessions(dir)
	require.NoError(t, err)
	require.Len(t, list, 1, "lock sidecars must not show up as sessions")

	// Graceful exit releases the lock; the next process claims the session and
	// continues the same branch instead of starting a divergent one.
	require.NoError(t, stdin.Close())
	require.NoError(t, cmd.Wait())

	var owner *Manager
	require.Eventually(t, func() bool {
		owner, err = OpenOwnedSession(path)
		return err == nil
	}, 5*time.Second, 25*time.Millisecond)
	defer func() { _ = owner.Close() }()

	assert.Equal(t, "helper-model", owner.ModelName())
	_, err = owner.AppendCompaction(Compaction{Summary: "post-release compaction"})
	require.NoError(t, err)
	newID, err := owner.Append(llm.Message{Role: llm.RoleUser, Content: "after handoff"})
	require.NoError(t, err)
	require.Len(t, chainIDs(owner), len(owner.entries)-1, "no orphan branch across the handoff")

	replay, err := OpenSession(path)
	require.NoError(t, err)
	assert.Equal(t, newID, replay.LeafID())
	assert.Equal(t, chainIDs(owner), chainIDs(replay))
}

func TestCrossProcessOwnershipAfterHardCrash(t *testing.T) {
	path := helperSessionFile(t)
	cmd, _ := startLockHelper(t, path)

	_, err := OpenOwnedSession(path)
	require.ErrorIs(t, err, ErrSessionLocked)

	// SIGKILL: no graceful Close. The kernel must release the flock so the
	// stale sidecar never wedges the session.
	require.NoError(t, cmd.Process.Kill())
	_ = cmd.Wait()

	var owner *Manager
	require.Eventually(t, func() bool {
		owner, err = OpenOwnedSession(path)
		return err == nil
	}, 5*time.Second, 25*time.Millisecond)
	defer func() { _ = owner.Close() }()

	// The sidecar now names the new owner.
	data, err := os.ReadFile(lockPathFor(path))
	require.NoError(t, err)
	var holder lockHolder
	require.NoError(t, json.Unmarshal(bytes.TrimSpace(data), &holder))
	assert.Equal(t, os.Getpid(), holder.PID)

	// Helper history, model and branch are intact and stay linear under the
	// new owner's writes.
	assert.Equal(t, "helper-model", owner.ModelName())
	assert.Contains(t, messageContents(owner.BuildContext()), "assistant:helper turn")
	_, err = owner.Append(llm.Message{Role: llm.RoleAssistant, Content: "after crash"})
	require.NoError(t, err)
	_, err = owner.AppendCompaction(Compaction{Summary: "crash recovery"})
	require.NoError(t, err)
	require.Len(t, chainIDs(owner), len(owner.entries)-1)

	replay, err := OpenSession(path)
	require.NoError(t, err)
	assert.Equal(t, chainIDs(owner), chainIDs(replay))
}

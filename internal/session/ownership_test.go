package session

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pulseaiclub/phi/internal/llm"
)

func ownedSession(t *testing.T) *Manager {
	t.Helper()
	dir := t.TempDir()
	m, err := NewSessionManager(dir, WithSessionDir(dir), WithShouldFlush(true), WithModel("model-a"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// chainIDs walks leaf -> root and returns every entry id on the active branch.
func chainIDs(m *Manager) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []string
	var cur *string
	if m.leafID != nil {
		id := *m.leafID
		cur = &id
	}
	for cur != nil {
		entry, ok := m.byIDs[*cur]
		if !ok {
			return out
		}
		out = append(out, entry.GetID())
		cur = entry.GetParent()
	}
	return out
}

func TestOwnedSessionSingleWriter(t *testing.T) {
	m := ownedSession(t)
	_, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "first"})
	require.NoError(t, err)

	// A second owned opener against a live owner is refused.
	second, err := OpenOwnedSession(m.File())
	require.ErrorIs(t, err, ErrSessionLocked)
	assert.Nil(t, second)

	// Read-only replay stays available while the owner is alive.
	view, err := OpenSession(m.File())
	require.NoError(t, err)
	assert.Equal(t, "model-a", view.ModelName())

	// After the owner exits, a new process can claim the same session.
	require.NoError(t, m.Close())
	reopened, err := OpenOwnedSession(m.File())
	require.NoError(t, err)
	defer func() { _ = reopened.Close() }()
	assert.Equal(t, "model-a", reopened.ModelName())
}

func TestCloseReleasesOwnershipAndFailsClosedWrites(t *testing.T) {
	m := ownedSession(t)
	_, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "first"})
	require.NoError(t, err)
	path := m.File()
	before, err := os.ReadFile(path)
	require.NoError(t, err)

	require.NoError(t, m.Close())
	assert.NoFileExists(t, lockPathFor(path), "Close removes the lock sidecar")

	// Retired writers fail loudly instead of silently appending.
	_, err = m.Append(llm.Message{Role: llm.RoleUser, Content: "late"})
	assert.ErrorIs(t, err, ErrSessionClosed)
	assert.ErrorIs(t, m.SetModel("late-model"), ErrSessionClosed)
	_, err = m.AppendCompaction(Compaction{Summary: "late"})
	assert.ErrorIs(t, err, ErrSessionClosed)
	_, err = m.AppendBash("ls", "", 0, false)
	assert.ErrorIs(t, err, ErrSessionClosed)
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, before, after, "closed-manager writes must never reach disk")

	// Reads still work for replay.
	assert.NotEmpty(t, m.BuildContext())

	// Close is idempotent and ownership is reclaimable.
	require.NoError(t, m.Close())
	next, err := OpenOwnedSession(path)
	require.NoError(t, err)
	defer func() { _ = next.Close() }()
}

func TestDeleteSessionRefusesLiveOwner(t *testing.T) {
	dir := t.TempDir()
	m, err := NewSessionManager(dir, WithSessionDir(dir), WithShouldFlush(true))
	require.NoError(t, err)
	_, err = m.Append(llm.Message{Role: llm.RoleUser, Content: "hi"})
	require.NoError(t, err)
	_, err = m.Append(llm.Message{Role: llm.RoleAssistant, Content: "ok"})
	require.NoError(t, err)

	err = DeleteSession(dir, m.ID())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrSessionLocked), "delete must name the ownership conflict, got %v", err)
	_, statErr := os.Stat(m.File())
	require.NoError(t, statErr, "a refused delete must keep the session file")

	require.NoError(t, m.Close())
	require.NoError(t, DeleteSession(dir, m.ID()))
	assert.NoFileExists(t, m.File())
	assert.NoFileExists(t, lockPathFor(m.File()))
}

func TestInMemorySessionNeedsNoLock(t *testing.T) {
	m := NewManager(t.TempDir())
	assert.False(t, m.shouldFlush)
	_, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "ephemeral"})
	require.NoError(t, err)
	require.NoError(t, m.SetModel("model-x"))
	require.NoError(t, m.Close())
	_, err = m.Append(llm.Message{Role: llm.RoleUser, Content: "late"})
	assert.ErrorIs(t, err, ErrSessionClosed)
}

// TestSingleWriterKeepsOneLinearBranch reproduces the original divergence
// scenario: messages, model selections and compactions appended through the
// one owner must all stay reachable from the leaf, with no orphan branch.
func TestSingleWriterKeepsOneLinearBranch(t *testing.T) {
	m := ownedSession(t)
	appendID := func() string {
		id, err := m.Append(llm.Message{Role: llm.RoleUser, Content: "q"})
		require.NoError(t, err)
		return id
	}
	first := appendID()
	require.NoError(t, m.SetModel("model-b"))
	_, err := m.AppendCompaction(Compaction{Summary: "earlier work"})
	require.NoError(t, err)
	last, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "a"})
	require.NoError(t, err)

	// Every stored entry is reachable from the leaf: no silent side branch.
	// The session header anchors the file but is not part of the parent chain.
	chain := chainIDs(m)
	require.Len(t, chain, len(m.entries)-1, "all entries must sit on the leaf branch")
	assert.Equal(t, last, chain[0])
	assert.Contains(t, chain, first)

	// A read-only view after reopen sees the identical branch and model.
	view, err := OpenSession(m.File())
	require.NoError(t, err)
	assert.Equal(t, chainIDs(view), chain)
	assert.Equal(t, "model-b", view.ModelName())
	contents := messageContents(view.BuildContext())
	assert.Contains(t, contents, "compaction:earlier work")
	assert.Contains(t, contents, "assistant:a")
}

// TestLegacyFormatSessionTakesOwnership covers sessions written before the
// ownership era: lowercase "session" headers, no version field, no sidecar.
func TestLegacyFormatSessionTakesOwnership(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "2025-01-01T00-00-00_legacy.jsonl")
	legacy := []byte(
		`{"type":"session","id":"legacy","cwd":"/tmp"}` + "\n" +
			`{"type":"EntryMessage","id":"m1","parentID":null,"message":{"role":"user","content":"old question"}}` + "\n" +
			`{"type":"EntryMessage","id":"m2","parentID":"m1","message":{"role":"assistant","content":"old answer"}}` + "\n")
	require.NoError(t, os.WriteFile(path, legacy, 0o600))

	// Listing and replay keep working on legacy files without any sidecar.
	list, err := ListSessions(dir)
	require.NoError(t, err)
	require.Len(t, list, 1)
	view, err := OpenSession(path)
	require.NoError(t, err)
	assert.Equal(t, []string{"user:old question", "assistant:old answer"}, messageContents(view.BuildContext()))

	// Taking ownership upgrades the file in place without rewriting history.
	m, err := OpenOwnedSession(path)
	require.NoError(t, err)
	assert.Equal(t, "legacy", m.ID())
	require.NoError(t, m.SetModel("model-new"))
	_, err = m.AppendCompaction(Compaction{Summary: "legacy compaction"})
	require.NoError(t, err)
	newID, err := m.Append(llm.Message{Role: llm.RoleAssistant, Content: "new answer"})
	require.NoError(t, err)

	// While held, no second writer can join; replay still can.
	_, err = OpenOwnedSession(path)
	assert.ErrorIs(t, err, ErrSessionLocked)
	replay, err := OpenSession(path)
	require.NoError(t, err)
	assert.Equal(t, "model-new", replay.ModelName())
	assert.Equal(t, newID, replay.LeafID())
	assert.Equal(t, chainIDs(m), chainIDs(replay))

	require.NoError(t, m.Close())
	reopened, err := OpenOwnedSession(path)
	require.NoError(t, err)
	defer func() { _ = reopened.Close() }()
	assert.Equal(t, "model-new", reopened.ModelName())
	assert.Equal(t, newID, reopened.LeafID(), "model and compaction branch survive the ownership handoff")
}

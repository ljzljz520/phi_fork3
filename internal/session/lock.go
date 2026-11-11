package session

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

// Cross-process session ownership.
//
// Every persisted session has a sidecar lock file next to its JSONL
// (<name>.jsonl.lock). The manager that owns the session holds an exclusive
// kernel lock on that sidecar for its whole lifetime, so two processes can
// never append to the same session at once: a second opener fails with
// ErrSessionLocked instead of silently growing a second branch. The kernel
// releases the lock when the holder exits, even after SIGKILL or a crash, so
// ownership is always reclaimable. The JSON payload inside the sidecar is
// diagnostic only; the lock itself is authoritative.
var (
	// ErrSessionLocked means a live process already owns the session.
	ErrSessionLocked = errors.New("session: session is open in another process")
	// ErrSessionNotOwner means the manager was opened read-only (no ownership)
	// and was asked to mutate the session.
	ErrSessionNotOwner = errors.New("session: not the owner; open the session for writing before mutating it")
	// ErrSessionClosed means ownership was released and the manager can no
	// longer accept writes.
	ErrSessionClosed = errors.New("session: closed; ownership was released")
)

const lockFileSuffix = ".lock"

// lockHolder is the best-effort diagnostic payload stored inside a lock file.
type lockHolder struct {
	PID       int    `json:"pid"`
	StartedAt string `json:"startedAt"`
}

// sessionLock is an owned exclusive lock on one sidecar file.
type sessionLock struct {
	file *os.File
	path string
}

// lockPathFor returns the sidecar path guarding sessionFile.
func lockPathFor(sessionFile string) string {
	return sessionFile + lockFileSuffix
}

// acquireSessionLock claims exclusive ownership of the session behind
// sessionFile without blocking. A contended lock returns ErrSessionLocked.
func acquireSessionLock(sessionFile string) (*sessionLock, error) {
	path := lockPathFor(sessionFile)
	// #nosec G304 -- path is derived from a session id resolved inside the
	// canonical project session directory.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("session: open lock %s: %w", path, err)
	}
	if err := tryLockExclusive(f); err != nil {
		_ = f.Close()
		if errors.Is(err, errLockHeld) {
			return nil, heldLockError(path)
		}
		return nil, fmt.Errorf("session: lock %s: %w", path, err)
	}
	// Re-annotate on every acquisition. A stale payload from a crashed holder
	// must not outlive the new owner; the flock is what actually guards writes.
	payload, err := json.Marshal(lockHolder{PID: os.Getpid(), StartedAt: time.Now().Format(time.RFC3339)})
	if err == nil {
		_ = f.Truncate(0)
		_, _ = f.Write(append(payload, '\n'))
		_ = f.Sync()
	}
	return &sessionLock{file: f, path: path}, nil
}

// heldLockError reads the holder annotation, when present, so the failure
// names the process that currently owns the session.
func heldLockError(path string) error {
	holder := ""
	if data, err := os.ReadFile(path); err == nil {
		var h lockHolder
		if json.Unmarshal(data, &h) == nil && h.PID > 0 {
			holder = fmt.Sprintf(" (held by pid %d since %s)", h.PID, h.StartedAt)
		}
	}
	return fmt.Errorf("%w%s", ErrSessionLocked, holder)
}

// release drops the lock, closes the descriptor, and removes the sidecar.
// Removal is best effort: a crashed process can leave a sidecar behind, and
// the next holder reuses it instead of depending on the cleanup.
func (l *sessionLock) release() {
	if l == nil {
		return
	}
	unlockExclusive(l.file)
	_ = l.file.Close()
	_ = os.Remove(l.path)
}

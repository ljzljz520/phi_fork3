//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !windows

package session

import (
	"errors"
	"os"
)

// Platforms without a kernel lock primitive degrade to no-op ownership:
// single-writer guarantees are not available, but session persistence keeps
// working. The supported targets (linux, darwin, windows) fail closed.
var errLockHeld = errors.New("session: file locking is unsupported on this platform")

func tryLockExclusive(_ *os.File) error { return nil }

func unlockExclusive(_ *os.File) {}

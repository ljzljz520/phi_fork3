//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package session

import (
	"os"
	"syscall"
)

// errLockHeld is the non-blocking contention result. On Linux EAGAIN and
// EWOULDBLOCK share one errno value; flock documents either name.
var errLockHeld = syscall.EWOULDBLOCK

// tryLockExclusive takes a non-blocking exclusive flock. The lock is tied to
// the open file description and released automatically on process exit.
func tryLockExclusive(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}

func unlockExclusive(f *os.File) {
	_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}

package session

import (
	"os"
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")

	// errLockHeld is ERROR_LOCK_VIOLATION, returned by the immediate lock path.
	errLockHeld = syscall.Errno(33)
)

const (
	lockfileExclusiveLock   = 0x00000002
	lockfileFailImmediately = 0x00000001
	lockFileRegionBytes     = 1
)

// tryLockExclusive takes a non-blocking exclusive byte-range lock on the
// sidecar. Like flock, the OS releases it when the process exits.
func tryLockExclusive(f *os.File) error {
	var overlapped syscall.Overlapped
	result, _, err := procLockFileEx.Call(
		f.Fd(),
		lockfileExclusiveLock|lockfileFailImmediately,
		0,
		lockFileRegionBytes,
		0,
		uintptr(unsafe.Pointer(&overlapped)),
	)
	if result == 0 {
		return err
	}
	return nil
}

func unlockExclusive(f *os.File) {
	var overlapped syscall.Overlapped
	_, _, _ = procUnlockFileEx.Call(f.Fd(), 0, lockFileRegionBytes, 0, uintptr(unsafe.Pointer(&overlapped)))
}

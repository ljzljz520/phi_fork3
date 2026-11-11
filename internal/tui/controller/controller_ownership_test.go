package controller

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pulseaiclub/phi/internal/llm"
	"github.com/pulseaiclub/phi/internal/session"
)

// TestResumeLockedSessionKeepsOriginalActive reproduces a second terminal
// holding the target session: Resume must fail, leave the current session
// owned and writable, and succeed once the other owner exits.
func TestResumeLockedSessionKeepsOriginalActive(t *testing.T) {
	ctrl := forkTestController(t)
	originalID := ctrl.SessionID()

	// Another writer owns a session in the same directory.
	other, err := session.NewSessionManager(
		ctrl.SessionDir(),
		session.WithSessionDir(ctrl.SessionDir()),
		session.WithShouldFlush(true),
	)
	require.NoError(t, err)
	_, err = other.Append(llm.Message{Role: llm.RoleUser, Content: "locked elsewhere"})
	require.NoError(t, err)
	_, err = other.Append(llm.Message{Role: llm.RoleAssistant, Content: "busy"})
	require.NoError(t, err)

	_, err = ctrl.Resume(other.ID())
	require.ErrorIs(t, err, session.ErrSessionLocked)
	assert.Equal(t, originalID, ctrl.SessionID(), "a failed resume must not switch sessions")
	assert.Equal(t, originalID, ctrl.engine.Session().ID())

	// The original engine still owns its session and can keep writing.
	require.NoError(t, ctrl.engine.Session().Append(
		llm.Message{Role: llm.RoleUser, Content: "still here"},
	))

	// Once the other process exits, the same resume claims ownership.
	require.NoError(t, other.Close())
	_, err = ctrl.Resume(other.ID())
	require.NoError(t, err)
	assert.Equal(t, other.ID(), ctrl.SessionID())
	msgs := ctrl.engine.Session().BuildContext()
	require.NotEmpty(t, msgs)
	assert.Equal(t, "busy", msgs[len(msgs)-1].Content, "the locked session's branch resumes intact")
}

// TestNewSessionAfterLockedResumeStillWorks proves the controller is not left
// in a half-switched state after a denied resume.
func TestNewSessionAfterLockedResumeStillWorks(t *testing.T) {
	ctrl := forkTestController(t)
	originalID := ctrl.SessionID()

	other, err := session.NewSessionManager(
		ctrl.SessionDir(),
		session.WithSessionDir(ctrl.SessionDir()),
		session.WithShouldFlush(true),
	)
	require.NoError(t, err)
	_, err = other.Append(llm.Message{Role: llm.RoleUser, Content: "x"})
	require.NoError(t, err)
	_, err = other.Append(llm.Message{Role: llm.RoleAssistant, Content: "y"})
	require.NoError(t, err)

	_, err = ctrl.Resume(other.ID())
	require.ErrorIs(t, err, session.ErrSessionLocked)
	assert.Equal(t, originalID, ctrl.SessionID())

	require.NoError(t, ctrl.NewSession())
	assert.NotEqual(t, originalID, ctrl.SessionID())
	require.NoError(t, other.Close())
}

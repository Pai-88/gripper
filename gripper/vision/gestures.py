"""Gesture -> Event mapping with frame-debouncing.

Decoupled from the vision front-end so it is pure logic and unit-testable: feed
it ``(gesture_name, score)`` once per frame; it fires the gesture's name exactly
once after it has been held for ``hold_frames`` consecutive frames above
``min_score``, then latches until the gesture is released. This kills mid-motion
false triggers (build guide, "Mode-switch gestures and e-stop").
"""

from __future__ import annotations

from typing import Optional

from ..state_machine import Event

# MediaPipe Gesture Recognizer canned labels -> our state-machine events.
GESTURE_EVENTS: dict[str, Event] = {
    "Open_Palm": Event.STOP,        # highest priority — e-stop
    "Victory": Event.TOGGLE_AUTO,   # teleop <-> auto
    "Closed_Fist": Event.ARM,       # arm the autonomous grasp (preview)
    "Thumb_Up": Event.TRIGGER,      # commit the previewed grasp
}


def gesture_to_event(name: Optional[str]) -> Optional[Event]:
    return GESTURE_EVENTS.get(name) if name else None


class GestureDebouncer:
    """Edge-detect a held gesture. Returns the gesture name on the frame it
    crosses the hold threshold, then ``None`` until it is released/changed."""

    def __init__(self, hold_frames: int = 12, min_score: float = 0.7):
        self.hold_frames = hold_frames
        self.min_score = min_score
        self._current: Optional[str] = None
        self._count = 0
        self._fired = False

    def reset(self) -> None:
        self._current = None
        self._count = 0
        self._fired = False

    def update(self, name: Optional[str], score: float = 1.0) -> Optional[str]:
        if not name or score < self.min_score:
            self.reset()
            return None
        if name == self._current:
            self._count += 1
        else:
            self._current = name
            self._count = 1
            self._fired = False
        if self._count >= self.hold_frames and not self._fired:
            self._fired = True
            return name
        return None

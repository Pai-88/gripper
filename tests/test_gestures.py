"""Gesture debouncing: fire once after the hold window, latch, reset on release."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.vision.gestures import (  # noqa: E402
    GestureDebouncer, gesture_to_event, GESTURE_EVENTS,
)
from gripper.state_machine import Event  # noqa: E402


def test_fires_after_hold_window_once():
    d = GestureDebouncer(hold_frames=3, min_score=0.7)
    assert d.update("Victory", 0.9) is None   # 1
    assert d.update("Victory", 0.9) is None   # 2
    assert d.update("Victory", 0.9) == "Victory"  # 3 -> fire
    assert d.update("Victory", 0.9) is None   # latched
    assert d.update("Victory", 0.9) is None


def test_low_score_does_not_count():
    d = GestureDebouncer(hold_frames=2, min_score=0.7)
    assert d.update("Thumb_Up", 0.5) is None
    assert d.update("Thumb_Up", 0.9) is None
    assert d.update("Thumb_Up", 0.9) == "Thumb_Up"


def test_switching_gesture_resets_count():
    d = GestureDebouncer(hold_frames=2)
    d.update("Victory", 1.0)
    assert d.update("Open_Palm", 1.0) is None   # switched -> count restarts
    assert d.update("Open_Palm", 1.0) == "Open_Palm"


def test_release_then_retrigger():
    d = GestureDebouncer(hold_frames=2)
    d.update("Victory", 1.0)
    assert d.update("Victory", 1.0) == "Victory"
    assert d.update(None, 0.0) is None          # released
    d.update("Victory", 1.0)
    assert d.update("Victory", 1.0) == "Victory"  # can fire again


def test_gesture_to_event_mapping():
    assert gesture_to_event("Open_Palm") is Event.STOP
    assert gesture_to_event("Thumb_Up") is Event.TRIGGER
    assert gesture_to_event(None) is None
    assert gesture_to_event("Unknown") is None
    assert set(GESTURE_EVENTS) == {"Open_Palm", "Victory", "Closed_Fist", "Thumb_Up"}

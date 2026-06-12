"""Verify result_to_observation parses MediaPipe GestureRecognizer results
(duck-typed fakes) — no mediapipe/opencv needed."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.vision.teleop_hands import result_to_observation  # noqa: E402


class _Cat:
    def __init__(self, name, score=0.9):
        self.category_name = name
        self.score = score


class _LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Result:
    """Mimics mediapipe's GestureRecognizerResult shape."""
    def __init__(self, hands=True, gesture="Victory", score=0.91, handed="Right"):
        if hands:
            self.hand_landmarks = [[_LM(i * 0.01, i * 0.02, i * 0.001) for i in range(21)]]
            self.handedness = [[_Cat(handed)]]
            self.gestures = [[_Cat(gesture, score)]] if gesture is not None else [[]]
        else:
            self.hand_landmarks = []
            self.handedness = []
            self.gestures = []


def test_parses_landmarks_and_gesture():
    obs = result_to_observation(_Result(gesture="Victory", score=0.91), t_capture=12.5)
    assert len(obs["landmarks"]) == 21
    assert obs["landmarks"][2] == (0.02, 0.04, 0.002)
    assert obs["gesture"] == "Victory"
    assert abs(obs["gesture_score"] - 0.91) < 1e-9
    assert obs["handedness"] == "Right"
    assert obs["t_capture"] == 12.5


def test_null_gesture_becomes_none():
    obs = result_to_observation(_Result(gesture="None", score=0.4), t_capture=0.0)
    assert obs["gesture"] is None
    assert obs["gesture_score"] == 0.0


def test_empty_gesture_list_is_none():
    obs = result_to_observation(_Result(gesture=None), t_capture=0.0)
    assert obs["gesture"] is None


def test_handedness_passthrough():
    obs = result_to_observation(_Result(handed="Left"), t_capture=0.0)
    assert obs["handedness"] == "Left"


def test_landmarks_are_plain_tuples():
    obs = result_to_observation(_Result(), t_capture=1.0)
    # must be JSON-serialisable for SharedLatest (no numpy / mp objects)
    import json
    json.dumps(obs)
    assert all(isinstance(p, tuple) and len(p) == 3 for p in obs["landmarks"])

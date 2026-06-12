"""Every legal transition fires; every illegal one is rejected; STOP and RESET
behave as universal rules."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.state_machine import (  # noqa: E402
    State, Event, next_state, is_legal, IllegalTransition,
    STATE_CODE, CODE_STATE, guard,
)
from tests._util import assert_raises  # noqa: E402


def test_happy_path():
    s = State.IDLE
    s = next_state(s, Event.START);            assert s is State.TELEOP
    s = next_state(s, Event.TOGGLE_AUTO);      assert s is State.ARM_AUTO
    s = next_state(s, Event.TRIGGER);          assert s is State.AUTO_GRASP
    s = next_state(s, Event.GRASP_COMPLETE);   assert s is State.RETURN
    s = next_state(s, Event.RETURN_COMPLETE);  assert s is State.IDLE


def test_toggle_back_to_teleop():
    assert next_state(State.ARM_AUTO, Event.TOGGLE_AUTO) is State.TELEOP


def test_stop_from_every_state_goes_to_estop():
    for s in State:
        assert next_state(s, Event.STOP) is State.ESTOP
        assert is_legal(s, Event.STOP)


def test_reset_only_from_estop():
    assert next_state(State.ESTOP, Event.RESET) is State.IDLE
    assert_raises(IllegalTransition, next_state, State.TELEOP, Event.RESET)
    assert not is_legal(State.IDLE, Event.RESET)


def test_illegal_transition_raises():
    assert_raises(IllegalTransition, next_state, State.IDLE, Event.TRIGGER)
    assert_raises(IllegalTransition, next_state, State.RETURN, Event.START)


def test_illegal_transition_carries_context():
    try:
        next_state(State.IDLE, Event.GRASP_COMPLETE)
    except IllegalTransition as e:
        assert e.state is State.IDLE
        assert e.event is Event.GRASP_COMPLETE
    else:
        raise AssertionError("expected IllegalTransition")


def test_state_codes_are_bijective():
    for s in State:
        assert CODE_STATE[STATE_CODE[s]] is s


def test_guard_blocks_illegal():
    assert guard(State.IDLE, Event.TRIGGER) == "illegal"


def test_guard_blocks_mode_switch_when_not_near_home():
    # Legal transition, but unsafe because the arm isn't near home.
    assert guard(State.TELEOP, Event.TOGGLE_AUTO, near_home=False) == "unsafe_mode_switch"


def test_guard_allows_mode_switch_when_near_home():
    assert guard(State.TELEOP, Event.TOGGLE_AUTO, near_home=True) is None


def test_guard_never_blocks_estop():
    for s in State:
        assert guard(s, Event.STOP, near_home=False) is None

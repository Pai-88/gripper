"""The system state machine. Hand-rolled (a transition table) rather than a
library — ~80 lines is the better CV story and trivially testable.

States and the happy-path edges (see the build guide diagram in section 6):

    IDLE --START--> TELEOP
    TELEOP --TOGGLE_AUTO--> ARM_AUTO           (display grasp, no motion)
    ARM_AUTO --TOGGLE_AUTO--> TELEOP           (toggle back)
    ARM_AUTO --TRIGGER--> AUTO_GRASP           (supervisor confirms)
    AUTO_GRASP --GRASP_COMPLETE--> RETURN
    RETURN --RETURN_COMPLETE--> IDLE

Plus two universal rules:
    * ``STOP`` from ANY state -> ESTOP   (interrupt, always available)
    * ``RESET`` only from ESTOP -> IDLE  (deliberate 2 s hold-to-reset upstream)

Any event with no table entry is *illegal*: :func:`next_state` raises
:class:`IllegalTransition`, which the caller logs and ignores rather than crashing.
"""

from __future__ import annotations

from enum import Enum


class State(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"
    ARM_AUTO = "ARM_AUTO"
    AUTO_GRASP = "AUTO_GRASP"
    RETURN = "RETURN"
    ESTOP = "ESTOP"


class Event(Enum):
    START = "START"
    TOGGLE_AUTO = "TOGGLE_AUTO"
    ARM = "ARM"
    TRIGGER = "TRIGGER"
    GRASP_COMPLETE = "GRASP_COMPLETE"
    RETURN_COMPLETE = "RETURN_COMPLETE"
    STOP = "STOP"      # e-stop: gesture, button, watchdog, or fault
    RESET = "RESET"    # operator hold-to-reset out of ESTOP


class IllegalTransition(Exception):
    def __init__(self, state: State, event: Event):
        super().__init__(f"no transition for {event.value} in {state.value}")
        self.state = state
        self.event = event


# Explicit happy-path transitions. STOP (-> ESTOP from anywhere) and
# RESET (ESTOP -> IDLE) are handled in code below, not in this table.
_TABLE: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.START): State.TELEOP,
    (State.TELEOP, Event.TOGGLE_AUTO): State.ARM_AUTO,
    (State.ARM_AUTO, Event.TOGGLE_AUTO): State.TELEOP,
    (State.ARM_AUTO, Event.ARM): State.ARM_AUTO,  # idempotent re-arm / preview refresh
    (State.ARM_AUTO, Event.TRIGGER): State.AUTO_GRASP,
    (State.AUTO_GRASP, Event.GRASP_COMPLETE): State.RETURN,
    (State.RETURN, Event.RETURN_COMPLETE): State.IDLE,
}


def is_legal(state: State, event: Event) -> bool:
    if event is Event.STOP:
        return True
    if event is Event.RESET:
        return state is State.ESTOP
    return (state, event) in _TABLE


def next_state(state: State, event: Event) -> State:
    """Return the state after applying ``event``. Raises
    :class:`IllegalTransition` for any undefined (state, event) pair."""
    if event is Event.STOP:
        return State.ESTOP  # interrupt: reachable from every state
    if event is Event.RESET:
        if state is State.ESTOP:
            return State.IDLE
        raise IllegalTransition(state, event)
    try:
        return _TABLE[(state, event)]
    except KeyError:
        raise IllegalTransition(state, event) from None


# States from which it is safe to switch operating mode (slow / near home).
SAFE_FOR_MODE_SWITCH = frozenset({State.IDLE, State.TELEOP, State.ARM_AUTO})

# Events that may only fire from a safe state AND with the arm near home, so a
# clench can't drop you out of a live grasp (build guide, "Mode switching").
MODE_SWITCH_EVENTS = frozenset({Event.TOGGLE_AUTO})


def guard(state: State, event: Event, near_home: bool = True):
    """Return a rejection reason (str) if ``event`` should be blocked in
    ``state``, else ``None``. Layers a safety policy on top of :func:`is_legal`:
    e-stop is always allowed; mode switches need a safe, near-home state."""
    if not is_legal(state, event):
        return "illegal"
    if event in MODE_SWITCH_EVENTS and not (state in SAFE_FOR_MODE_SWITCH and near_home):
        return "unsafe_mode_switch"
    return None

# Integer codes used in the telemetry S frame; mirrored in protocol.h.
STATE_CODE: dict[State, int] = {s: i for i, s in enumerate(State)}
CODE_STATE: dict[int, State] = {i: s for s, i in STATE_CODE.items()}

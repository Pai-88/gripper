"""decode_flags maps the telemetry bitfield to labels (pure; no fastapi needed)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.telemetry.dashboard import decode_flags  # noqa: E402
from gripper import protocol as p  # noqa: E402


def test_no_flags():
    assert decode_flags(0) == []


def test_single_flag():
    assert decode_flags(p.FLAG_WATCHDOG) == ["WATCHDOG"]


def test_multiple_flags_in_bit_order():
    flags = p.FLAG_UNDERVOLT | p.FLAG_OVERTEMP
    assert decode_flags(flags) == ["UNDERVOLT", "OVERTEMP"]


def test_all_flags():
    allf = (p.FLAG_UNDERVOLT | p.FLAG_LIMIT_CLAMP | p.FLAG_WATCHDOG
            | p.FLAG_OVERTEMP | p.FLAG_OVERCURRENT)
    assert decode_flags(allf) == ["UNDERVOLT", "CLAMP", "WATCHDOG", "OVERTEMP", "OVERCURRENT"]

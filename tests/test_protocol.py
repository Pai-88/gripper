"""Pins the serial wire format and CRC. If this breaks, the firmware and host
have drifted apart — fix both ``protocol.py`` and ``protocol.h``."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper import protocol as p  # noqa: E402
from tests._util import assert_raises  # noqa: E402


def test_crc8_maxim_check_vector():
    # The canonical CRC-8/MAXIM check value for "123456789" is 0xA1.
    assert p.crc8_maxim(b"123456789") == 0xA1


def test_crc8_empty():
    assert p.crc8_maxim(b"") == 0x00


def test_joint_roundtrip():
    angles = (1234, 9000, 4500, 17999)
    frame = p.encode_joint(angles, grip_milli=750, seq=42)
    assert frame.endswith(b"\n")
    decoded = p.decode_line(frame.decode())
    assert decoded is not None
    assert decoded.kind == "J"
    assert decoded.angles_cd == angles
    assert decoded.grip == 750
    assert decoded.seq == 42


def test_joint_wrong_arity_rejected():
    assert_raises(ValueError, p.encode_joint, (1, 2, 3), 0, 0)


def test_bad_crc_rejected():
    frame = p.encode_joint((10, 20, 30, 40), 500, 7).decode()
    # Corrupt one payload digit; CRC must now fail -> None.
    bad = frame.replace("10", "11", 1)
    assert p.decode_line(bad) is None


def test_truncated_frame_rejected():
    assert p.decode_line("J,1,2,3") is None
    assert p.decode_line("") is None
    assert p.decode_line("garbage") is None


def test_seq_rollover():
    f0 = p.decode_line(p.encode_joint((0, 0, 0, 0), 0, 256).decode())
    assert f0.seq == 0  # 256 & 0xFF
    f1 = p.decode_line(p.encode_joint((0, 0, 0, 0), 0, 257).decode())
    assert f1.seq == 1


def test_control_frames():
    assert p.decode_line(p.encode_mode(p.MODE_AUTO).decode()).mode == p.MODE_AUTO
    assert p.decode_line(p.encode_estop().decode()).kind == "E"
    assert p.decode_line(p.encode_home().decode()).kind == "H"
    assert p.decode_line(p.encode_ping().decode()).kind == "P"


def test_telemetry_roundtrip():
    frame = p.encode_telemetry(
        state=3, angles_cd=(100, 200, 300, 400), grip_milli=500,
        vbat_mV=7380, flags=p.FLAG_LIMIT_CLAMP, seq_echo=99,
    )
    d = p.decode_line(frame.decode())
    assert d is not None and d.kind == "S"
    assert d.state == 3
    assert d.angles_cd == (100, 200, 300, 400)
    assert d.grip == 500
    assert d.vbat_mV == 7380
    assert d.flags == p.FLAG_LIMIT_CLAMP
    assert d.seq == 99

"""Serial wire protocol between the Pi 5 (host) and the ESP32 (servo MCU).

ASCII, newline-framed, 1 000 000 baud 8N1. This module is the single source of
truth — ``firmware/esp32_servo/protocol.h`` is hand-mirrored from it, and
``tests/test_protocol.py`` pins the byte format and CRC.

Host -> MCU
    ``J,<base>,<shoulder>,<elbow>,<wrist>,<grip>,<seq>,<crc>\\n``
        joint angles in CENTIDEGREES (int, 0..18000), wire order = JOINT_NAMES
        grip 0..1000 (0 = closed, 1000 = fully open)
        seq  0..255 rolling sequence number
        crc  CRC-8/MAXIM (Dallas 1-Wire) over the ASCII payload ``J,...,<seq>``
    ``M,<mode>\\n``   mode: 0 idle/limp, 1 teleop, 2 autograsp
    ``E\\n``          immediate e-stop (deliberately the simplest possible frame)
    ``H\\n``          go to home pose
    ``P\\n``          ping / keep-alive

MCU -> Host (telemetry, ~20 Hz)
    ``S,<state>,<base>,<shoulder>,<elbow>,<wrist>,<grip>,<vbat_mV>,<flags>,<seq_echo>,<crc>\\n``

Only ``J`` and ``S`` carry a CRC; the short control frames are validated by
exact match. A CRC mismatch or malformed frame decodes to ``None`` and must be
dropped by the receiver (the firmware watchdog covers a run of dropped frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import JOINT_COUNT

# --- scaling -----------------------------------------------------------------
ANGLE_MIN_CD = 0
ANGLE_MAX_CD = 18000  # 180.00 deg
GRIP_MIN = 0
GRIP_MAX = 1000

# --- modes (M frame) ---------------------------------------------------------
MODE_IDLE = 0
MODE_TELEOP = 1
MODE_AUTO = 2

# --- telemetry flag bits (S frame) -------------------------------------------
FLAG_UNDERVOLT = 1 << 0
FLAG_LIMIT_CLAMP = 1 << 1
FLAG_WATCHDOG = 1 << 2
FLAG_OVERTEMP = 1 << 3
FLAG_OVERCURRENT = 1 << 4


def crc8_maxim(data: bytes) -> int:
    """CRC-8/MAXIM (a.k.a. Dallas / 1-Wire). poly 0x31 reflected (0x8C),
    init 0x00, refin/refout true, xorout 0x00. check("123456789") == 0xA1.

    Implemented identically in ``protocol.h`` (``crc8_maxim``)."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if (crc & 1) else (crc >> 1)
    return crc & 0xFF


@dataclass
class Frame:
    """A decoded inbound frame. ``kind`` is one of J M E H P S."""

    kind: str
    angles_cd: tuple = ()
    grip: int = 0
    seq: int = 0
    mode: int = 0
    state: int = 0
    vbat_mV: int = 0
    flags: int = 0


# --- encode (host side) ------------------------------------------------------
def encode_joint(angles_cd: Sequence[int], grip_milli: int, seq: int) -> bytes:
    if len(angles_cd) != JOINT_COUNT:
        raise ValueError(f"expected {JOINT_COUNT} angles, got {len(angles_cd)}")
    body = "J," + ",".join(str(int(a)) for a in angles_cd)
    body += f",{int(grip_milli)},{int(seq) & 0xFF}"
    crc = crc8_maxim(body.encode("ascii"))
    return f"{body},{crc}\n".encode("ascii")


def encode_mode(mode: int) -> bytes:
    return f"M,{int(mode)}\n".encode("ascii")


def encode_estop() -> bytes:
    return b"E\n"


def encode_home() -> bytes:
    return b"H\n"


def encode_ping() -> bytes:
    return b"P\n"


def encode_telemetry(
    state: int,
    angles_cd: Sequence[int],
    grip_milli: int,
    vbat_mV: int,
    flags: int,
    seq_echo: int,
) -> bytes:
    """Reference encoder for the S frame (mirrors the firmware; used in tests)."""
    if len(angles_cd) != JOINT_COUNT:
        raise ValueError(f"expected {JOINT_COUNT} angles, got {len(angles_cd)}")
    body = "S," + str(int(state)) + "," + ",".join(str(int(a)) for a in angles_cd)
    body += f",{int(grip_milli)},{int(vbat_mV)},{int(flags)},{int(seq_echo) & 0xFF}"
    crc = crc8_maxim(body.encode("ascii"))
    return f"{body},{crc}\n".encode("ascii")


# --- decode (either side) ----------------------------------------------------
def _crc_ok(parts: list[str]) -> bool:
    body = ",".join(parts[:-1])
    try:
        return crc8_maxim(body.encode("ascii")) == int(parts[-1])
    except ValueError:
        return False


def decode_line(line: str) -> Optional[Frame]:
    """Parse one line. Returns a :class:`Frame` or ``None`` if malformed / bad CRC."""
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    kind = parts[0]

    if kind == "J":
        # J + 4 angles + grip + seq + crc = 4 + JOINT_COUNT fields
        if len(parts) != 4 + JOINT_COUNT or not _crc_ok(parts):
            return None
        try:
            nums = [int(p) for p in parts[1:-1]]
        except ValueError:
            return None
        return Frame("J", angles_cd=tuple(nums[:JOINT_COUNT]),
                     grip=nums[JOINT_COUNT], seq=nums[JOINT_COUNT + 1])

    if kind == "S":
        # S + state + 4 angles + grip + vbat + flags + seq + crc = 7 + JOINT_COUNT fields
        if len(parts) != 7 + JOINT_COUNT or not _crc_ok(parts):
            return None
        try:
            nums = [int(p) for p in parts[1:-1]]
        except ValueError:
            return None
        i = 1
        return Frame(
            "S",
            state=nums[0],
            angles_cd=tuple(nums[i:i + JOINT_COUNT]),
            grip=nums[i + JOINT_COUNT],
            vbat_mV=nums[i + JOINT_COUNT + 1],
            flags=nums[i + JOINT_COUNT + 2],
            seq=nums[i + JOINT_COUNT + 3],
        )

    if kind == "M" and len(parts) == 2:
        try:
            return Frame("M", mode=int(parts[1]))
        except ValueError:
            return None

    if kind in ("E", "H", "P") and len(parts) == 1:
        return Frame(kind)

    return None

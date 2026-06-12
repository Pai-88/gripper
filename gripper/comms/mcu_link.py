"""Async serial link to the ESP32. Runs at a fixed wire rate decoupled from
vision via a 1-slot mailbox (latest target wins).

Wire angle convention (the Pi<->MCU contract): each joint is sent as
**centidegrees offset from that joint's ``min_deg``**, so the value is always
unsigned and within 0..18000. The firmware (``limits.h``) holds the same
min/max and maps offset-centidegrees back to LX bus-servo units. Unit mapping
thus lives in exactly two mirrored places (build guide: "Units mapped once").
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional, TYPE_CHECKING

from .. import JOINT_NAMES
from ..messages import JointTargets, Telemetry
from ..state_machine import CODE_STATE
from .. import protocol as proto

if TYPE_CHECKING:  # JointCfg used only in annotations (lazy via __future__)
    from ..config import JointCfg

try:
    import serial_asyncio  # type: ignore
    _HAVE_SERIAL = True
except ImportError:  # keep the module importable on dev machines / CI
    _HAVE_SERIAL = False


def targets_to_wire(targets: JointTargets, joints: dict[str, JointCfg]
                    ) -> tuple[list[int], int]:
    """Convert degree targets -> (offset-centidegree ints, grip milli-units).

    This is the last layer before the serial bus, so it is hardened to never
    raise and never emit an out-of-range value: a non-finite (NaN/inf) joint
    fails safe to that joint's minimum, and a non-finite grip fails safe to
    closed. The firmware re-clamps anyway, but this keeps the send thread alive.
    """
    cd: list[int] = []
    for name in JOINT_NAMES:
        cfg = joints[name]
        ang = targets.__dict__[name]
        if not math.isfinite(ang):
            ang = cfg.min_deg  # fail-safe: park at joint min
        val = (ang - cfg.min_deg) * 100.0
        if cfg.invert:
            val = cfg.span_deg * 100.0 - val
        cd.append(int(max(proto.ANGLE_MIN_CD, min(proto.ANGLE_MAX_CD, round(val)))))
    grip = targets.grip if math.isfinite(targets.grip) else 0.0
    grip_milli = int(max(proto.GRIP_MIN, min(proto.GRIP_MAX, round(grip * 1000))))
    return cd, grip_milli


class MCULink:
    """Owns the serial transport, a send mailbox, and telemetry decode."""

    def __init__(self, port: str, baud: int, joints: dict[str, JointCfg]):
        self.port = port
        self.baud = baud
        self.joints = joints
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._seq = 0
        self._mailbox: Optional[bytes] = None
        self._control: Optional[bytes] = None  # E/M/H frame, flushed first
        self.latest_telemetry: Optional[Telemetry] = None

    async def connect(self) -> None:
        if not _HAVE_SERIAL:
            raise RuntimeError("pyserial-asyncio not installed; pip install '.[]'")
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self.port, baudrate=self.baud
        )

    # -- outbound ------------------------------------------------------------
    def queue_targets(self, targets: JointTargets) -> None:
        """Stage the latest target; the send loop drains it at a fixed rate."""
        cd, grip = targets_to_wire(targets, self.joints)
        self._seq = (self._seq + 1) & 0xFF
        self._mailbox = proto.encode_joint(cd, grip, self._seq)

    def queue_estop(self) -> None:
        """Stage an immediate e-stop frame (sync, callable from step_once)."""
        self._control = proto.encode_estop()

    def queue_control(self, frame: bytes) -> None:
        self._control = frame

    async def _write(self, data: bytes) -> None:
        if self._writer is None:
            raise RuntimeError("not connected")
        self._writer.write(data)
        await self._writer.drain()

    async def send_mode(self, mode: int) -> None:
        await self._write(proto.encode_mode(mode))

    async def send_estop(self) -> None:
        await self._write(proto.encode_estop())

    async def send_home(self) -> None:
        await self._write(proto.encode_home())

    async def send_loop(self, hz: int) -> None:
        """Drain the mailbox at a constant rate so wire timing is independent
        of vision jitter."""
        period = 1.0 / hz
        while True:
            if self._control is not None:        # e-stop / mode / home: flush first
                await self._write(self._control)
                self._control = None
            if self._mailbox is not None:
                await self._write(self._mailbox)
            await asyncio.sleep(period)

    # -- inbound -------------------------------------------------------------
    async def read_loop(self) -> None:
        if self._reader is None:
            raise RuntimeError("not connected")
        while True:
            raw = await self._reader.readline()
            if not raw:
                await asyncio.sleep(0.001)
                continue
            frame = proto.decode_line(raw.decode("ascii", "ignore"))
            if frame is not None and frame.kind == "S":
                state = CODE_STATE.get(frame.state)
                self.latest_telemetry = Telemetry(
                    state=state.value if state else "?",
                    joints=frame.angles_cd,
                    grip=frame.grip / 1000.0,
                    vbat_mV=frame.vbat_mV,
                    flags=frame.flags,
                    seq=frame.seq,
                )

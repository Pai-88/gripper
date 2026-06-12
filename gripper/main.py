"""Entry point: wires the whole stack together on the Pi 5.

Layout (build guide section 6):
  * two CPU-heavy VISION PROCESSES (teleop hands, grasp), gated by Events, each
    publishing into a SharedLatest channel;
  * a MAIN asyncio process running the state machine, controller, MCU comms,
    watchdog and logger.

Only one vision mode is active at a time (the Pi 5 can't run both models at
usable rates) — they are kept warm but gated. This file is the integration glue;
the load-bearing logic it calls (protocol, kinematics, limits, state machine,
gestures, mapping) is unit-tested.

    python -m gripper.main            # real run (needs cameras + ESP32 + deps)
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from . import JOINT_NAMES
from .control.controller import Controller
from .control.servoing import GraspSequencer, GraspParams
from .control.teleop_map import Calibration, OneEuroFilter, landmarks_to_targets
from .messages import JointTargets
from .metrics import LatencyTracker, build_snapshot
from .state_machine import State, Event, next_state, guard, IllegalTransition
from .vision.gestures import GestureDebouncer, gesture_to_event
from .safety.watchdog import Watchdog
from .telemetry.logger import get_logger

if TYPE_CHECKING:  # load_config imported lazily in main(); Config only annotates
    from .config import Config


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = get_logger()
        self.state = State.IDLE
        dt = 1.0 / cfg.serial.command_hz
        self.controller = Controller(cfg.joints, dt=dt)
        self.cal = Calibration()
        self.filters = {n: OneEuroFilter(**cfg.teleop.one_euro.model_dump())
                        for n in (*JOINT_NAMES, "grip")}
        self.debounce = GestureDebouncer(**cfg.teleop.gesture.model_dump())
        self.watchdog = Watchdog(cfg.watchdog.vision_stale_ms / 1000.0, self._trip)
        self._latency = LatencyTracker()
        self._estop_sent = False
        self._sequencer = None  # GraspSequencer, created on entering AUTO_GRASP
        # Vision gates (always exist); channels/link/processes created in
        # start_hardware() at runtime.
        self._hand_active = mp.Event()
        self._grasp_active = mp.Event()
        self.hand_shm = None
        self.grasp_shm = None
        self.link = None  # MCULink
        self._procs: list = []

    async def start_hardware(self) -> None:
        """Create the shared-memory channels, spawn the vision processes, and
        connect the serial link. Call before the control loop on a real run."""
        from .shared_state import SharedLatest
        from .vision import teleop_hands, grasp_vision
        from .comms.mcu_link import MCULink

        self.hand_shm = SharedLatest(capacity=8192)
        self.grasp_shm = SharedLatest(capacity=4096)
        tp, wr = self.cfg.cameras["teleop"], self.cfg.cameras["wrist"]
        self._procs = [
            mp.Process(target=teleop_hands.run, daemon=True,
                       args=(self.hand_shm.name, self._hand_active, tp.index,
                             tp.width, tp.height, self.cfg.teleop.model_path,
                             self.cfg.teleop.mirror)),
            mp.Process(target=grasp_vision.run, daemon=True,
                       args=(self.grasp_shm.name, self._grasp_active, wr.index,
                             wr.width, wr.height, self.cfg.grasp.detect_hz)),
        ]
        for p in self._procs:
            p.start()
        self.link = MCULink(self.cfg.serial.port, self.cfg.serial.baud, self.cfg.joints)
        await self.link.connect()
        self.log.log("hardware_up", procs=[p.pid for p in self._procs])

    # -- state machine -------------------------------------------------------
    def dispatch(self, event: Event) -> None:
        reason = guard(self.state, event, self.controller.near_home())
        if reason:
            self.log.log("event_rejected", state=self.state.value,
                         event=event.value, reason=reason)
            return
        try:
            new = next_state(self.state, event)
        except IllegalTransition:
            return
        if new is not self.state:
            self.log.log("transition", frm=self.state.value, to=new.value,
                         via=event.value)
            self.state = new
            self._on_enter(new)

    def _on_enter(self, state: State) -> None:
        # The hand camera stays active for the gesture channel (incl. the e-stop
        # gesture) in every running state; the grasp camera only when grasping.
        if state is not State.ESTOP:
            self._estop_sent = False
        if state is State.AUTO_GRASP:
            self._sequencer = GraspSequencer(
                GraspParams(commit_frames=self.cfg.grasp.commit_frames))
        if self.hand_shm is None:
            return  # vision not started (dry run)
        hand_states = (State.TELEOP, State.ARM_AUTO, State.AUTO_GRASP, State.RETURN)
        self._hand_active.set() if state in hand_states else self._hand_active.clear()
        self._grasp_active.set() if state in (State.ARM_AUTO, State.AUTO_GRASP) \
            else self._grasp_active.clear()

    def snapshot(self) -> dict:
        """Status dict for the dashboard websocket."""
        t = self.link.latest_telemetry if self.link else None
        return build_snapshot(
            self.state.value, self.controller.current,
            flags=(t.flags if t else 0), vbat_mV=(t.vbat_mV if t else 0),
            p50=self._latency.p50(), p95=self._latency.p95(),
            clamped=self.controller.clamped, seq=(t.seq if t else 0),
        )

    def _trip(self, reason: str) -> None:
        self.log.log("watchdog_trip", reason=reason)
        self.dispatch(Event.STOP)

    # -- control tick --------------------------------------------------------
    def _teleop_request(self, hand: dict) -> JointTargets:
        raw = landmarks_to_targets(hand["landmarks"], self.cal)
        return JointTargets(
            base=self.filters["base"](raw.base),
            shoulder=self.filters["shoulder"](raw.shoulder),
            elbow=self.filters["elbow"](raw.elbow),
            wrist=self.filters["wrist"](raw.wrist),
            grip=self.filters["grip"](raw.grip),
        )

    def _queue(self, cmd: JointTargets) -> None:
        if self.link:
            self.link.queue_targets(cmd)

    def step_once(self, hand: dict | None, grasp: dict | None, now: float) -> None:
        """One synchronous control tick (gestures + per-state actuation). Pulled
        out of the async loop so the full state behaviour is unit-testable."""
        # Gesture channel + watchdog feed run in every state that has a hand.
        if hand:
            self.watchdog.feed_vision(hand.get("t_capture"))
            self._latency.add((now - hand.get("t_capture", now)) * 1000.0)
            fired = self.debounce.update(hand.get("gesture"),
                                         hand.get("gesture_score", 0.0))
            if fired:
                mapped = gesture_to_event(fired)
                if mapped:
                    self.dispatch(mapped)

        if self.state is State.TELEOP and hand:
            self._queue(self.controller.step(self._teleop_request(hand)))
        elif self.state is State.ARM_AUTO:
            self._queue(self.controller.step(self.controller.current))  # preview: hold
        elif self.state is State.AUTO_GRASP and self._sequencer is not None:
            target = self._sequencer.update(grasp, self.controller.current)
            self._queue(self.controller.step(target))
            if self._sequencer.done():
                self.log.log("grasp_done", phase=self._sequencer.phase.value)
                self.dispatch(Event.GRASP_COMPLETE)
        elif self.state is State.RETURN:
            self._queue(self.controller.home())
            if self.controller.near_home():
                self.dispatch(Event.RETURN_COMPLETE)
        elif self.state is State.ESTOP and not self._estop_sent:
            if self.link:
                self.link.queue_estop()
            self._estop_sent = True

        self.watchdog.check(now)

    async def control_loop(self) -> None:
        period = 1.0 / self.cfg.serial.command_hz
        tick = 0
        while True:
            now = time.monotonic()
            hand = self.hand_shm.get() if self.hand_shm is not None else None
            grasp = self.grasp_shm.get() if self.grasp_shm is not None else None
            self.step_once(hand, grasp, now)
            tick += 1
            if tick % 20 == 0:  # ~decimated telemetry to the log
                self.log.log("tick", state=self.state.value,
                             lat_p50=self._latency.p50(), lat_p95=self._latency.p95(),
                             clamped=self.controller.clamped)
            await asyncio.sleep(period)


async def _amain(cfg: Config, dry: bool, dashboard_port: int | None = None) -> None:
    app = App(cfg)
    app.log.log("boot", dry_run=dry, config_joints=list(cfg.joints))
    if dry:
        # Smoke test: drive the state machine through its happy path, no hardware.
        for ev in (Event.START, Event.TOGGLE_AUTO, Event.TRIGGER,
                   Event.GRASP_COMPLETE, Event.RETURN_COMPLETE):
            app.dispatch(ev)
        app.log.log("dry_run_complete", final_state=app.state.value)
        return
    # Real run: bring up cameras + vision processes + serial, enter TELEOP, and
    # run the control loop alongside the MCU send/read loops (+ optional dashboard).
    await app.start_hardware()
    app.dispatch(Event.START)
    tasks = [app.control_loop(),
             app.link.send_loop(cfg.serial.command_hz),
             app.link.read_loop()]
    if dashboard_port:
        from .telemetry.dashboard import serve
        tasks.append(serve(app.snapshot, port=dashboard_port))
        app.log.log("dashboard_up", port=dashboard_port)
    try:
        await asyncio.gather(*tasks)
    finally:
        for p in app._procs:
            p.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="gripper control stack")
    parser.add_argument("--config", default=None, help="path to robot.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="exercise the state machine with no hardware")
    parser.add_argument("--dashboard", action="store_true",
                        help="serve the live telemetry dashboard")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    args = parser.parse_args()
    from .config import load_config  # lazy: pulls in pydantic only for a real run
    cfg = load_config(args.config)
    mp.set_start_method("spawn", force=True)
    asyncio.run(_amain(cfg, dry=args.dry_run,
                       dashboard_port=args.port if args.dashboard else None))


if __name__ == "__main__":
    main()

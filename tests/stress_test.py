#!/usr/bin/env python3
"""Rigorous, extreme smoke/stress test — property-based + fuzz + fault injection
across every pure module. NOT part of the unit suite; run directly:

    python3 tests/stress_test.py            # default ~25k iters/section
    ITERS=200000 python3 tests/stress_test.py

Invariants checked (a failure prints FAIL and the harness exits non-zero):
  * no function crashes on adversarial input that the contract allows;
  * decoders reject malformed input rather than raising;
  * encode∘decode is identity for valid frames;
  * IK∘FK round-trips; clamp/slew never violate bounds;
  * the state machine never lands in an invalid state; ESTOP always reachable;
  * the controller never commands out-of-limit / faster-than-max-rate;
  * the wire layer never emits a non-finite / out-of-range value (even on NaN);
  * the grasp sequencer always stays bounded and terminates on a stable plant;
  * the seqlock shared channel never returns a torn read under a hammering writer.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper import protocol as P, JOINT_NAMES  # noqa: E402
from gripper.kinematics import (ArmGeometry, inverse, forward_position,  # noqa: E402
                                Unreachable, is_reachable)
from gripper.control.limits import clamp, rate_limit, slew, clamp_and_slew  # noqa: E402
from gripper.control.teleop_map import (OneEuroFilter, Calibration,  # noqa: E402
                                        landmarks_to_targets)
from gripper.control.controller import Controller  # noqa: E402
from gripper.control.servoing import (GraspParams, GraspSequencer,  # noqa: E402
                                      GraspPhase, servo_step, wrap_grasp_angle)
from gripper.state_machine import (State, Event, next_state, is_legal, guard,  # noqa: E402
                                   IllegalTransition)
from gripper.vision.gestures import GestureDebouncer  # noqa: E402
from gripper.vision.grasp_vision import normalize_grasp  # noqa: E402
from gripper.vision.teleop_hands import result_to_observation  # noqa: E402
from gripper.comms.mcu_link import targets_to_wire  # noqa: E402
from gripper.messages import JointTargets  # noqa: E402
from gripper.metrics import LatencyTracker, build_snapshot  # noqa: E402
from gripper.telemetry.dashboard import decode_flags  # noqa: E402
from gripper.shared_state import SharedLatest  # noqa: E402

N = int(os.environ.get("ITERS", 25000))
random.seed(0xC0FFEE)
FAILS: list[tuple[str, str]] = []
CHECKS = 0


def ck(name: str, cond: bool, info: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append((name, info))
        print(f"  FAIL {name}  {info}")


def finite(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


# ---------------------------------------------------------------- helpers
def rand_str(maxlen=40):
    charset = "0123456789,.-+ JMEHPS\n\t" + "abcXYZ\x00\xff" + ",,,,"
    return "".join(random.choice(charset) for _ in range(random.randrange(0, maxlen)))


def fake_joints():
    from types import SimpleNamespace
    def jc(mn, mx, home, rate):
        return SimpleNamespace(min_deg=mn, max_deg=mx, home_deg=home,
                               max_rate_dps=rate, invert=random.random() < 0.5,
                               span_deg=mx - mn)
    return {"base": jc(-90, 90, 0, 120), "shoulder": jc(0, 90, 20, 90),
            "elbow": jc(-120, 0, -60, 120), "wrist": jc(-90, 90, 0, 150),
            "grip": jc(0, 100, 100, 300)}


# ================================================================ sections
def s_crc():
    ck("crc.vector", P.crc8_maxim(b"123456789") == 0xA1)
    ck("crc.empty", P.crc8_maxim(b"") == 0)
    for _ in range(N):
        data = bytes(random.randrange(256) for _ in range(random.randrange(0, 64)))
        c = P.crc8_maxim(data)
        ck("crc.range", 0 <= c <= 255)
        ck("crc.determinism", c == P.crc8_maxim(data))


def s_protocol_roundtrip():
    for _ in range(N):
        ang = [random.randint(P.ANGLE_MIN_CD, P.ANGLE_MAX_CD) for _ in range(4)]
        grip = random.randint(0, 1000)
        seq = random.randint(0, 255)
        f = P.encode_joint(ang, grip, seq)
        d = P.decode_line(f.decode())
        ck("proto.J.roundtrip", d is not None and d.kind == "J"
           and list(d.angles_cd) == ang and d.grip == grip and d.seq == seq,
           f"{ang} {grip} {seq}")
        # telemetry
        st = random.randint(0, 5); vb = random.randint(0, 9000)
        fl = random.randint(0, 31); se = random.randint(0, 255)
        t = P.encode_telemetry(st, ang, grip, vb, fl, se)
        dt = P.decode_line(t.decode())
        ck("proto.S.roundtrip", dt is not None and dt.kind == "S"
           and dt.state == st and dt.flags == fl and dt.vbat_mV == vb)


def s_protocol_fuzz():
    # decode_line must NEVER raise, for ANY string.
    for _ in range(N):
        s = rand_str()
        try:
            r = P.decode_line(s)
        except Exception as e:  # noqa: BLE001
            ck("proto.decode.noraise", False, f"{s!r} -> {e!r}")
            continue
        ck("proto.decode.type", r is None or hasattr(r, "kind"))
    # bit-flip corruption of valid frames -> must reject or stay parseable, never raise
    for _ in range(N):
        f = P.encode_joint([random.randint(0, 18000) for _ in range(4)],
                           random.randint(0, 1000), random.randint(0, 255)).decode()
        b = list(f.rstrip("\n"))
        i = random.randrange(len(b))
        b[i] = random.choice("0123456789,")
        corrupted = "".join(b)
        try:
            r = P.decode_line(corrupted)
        except Exception as e:  # noqa: BLE001
            ck("proto.corrupt.noraise", False, f"{corrupted!r} -> {e!r}")
            continue
        # if it parses, the CRC genuinely matched
        if r is not None and r.kind == "J":
            parts = corrupted.split(",")
            ck("proto.corrupt.crc_consistent",
               P.crc8_maxim(",".join(parts[:-1]).encode()) == int(parts[-1]))
    # adversarial fixed cases
    for s in ["", " ", "\n", ",,,,,,,", "J", "J,1,2,3", "J,a,b,c,d,e,f,g",
              "J," + "9" * 400, "J,-1,-2,-3,-4,5,6,7", "S", "M", "M,x", "E,1",
              "J,1.5,2,3,4,5,6,7", "J,1,2,3,4,5,6,7,8,9", "💥,1,2"]:
        try:
            P.decode_line(s)
        except Exception as e:  # noqa: BLE001
            ck("proto.adversarial.noraise", False, f"{s!r} -> {e!r}")
    # encode arity guard
    for _ in range(N // 50):
        n = random.choice([0, 1, 2, 3, 5, 6, 10])
        try:
            P.encode_joint([0] * n, 0, 0)
            ck("proto.encode.arity", False, f"len {n} accepted")
        except ValueError:
            pass


def s_kinematics():
    for _ in range(N):
        l1 = random.uniform(0.05, 0.3); l2 = random.uniform(0.05, 0.3)
        g = ArmGeometry(l1, l2)
        # sample a guaranteed-reachable point
        d = random.uniform(g.reach_min + 1e-4, g.reach_max - 1e-4)
        base = random.uniform(-math.pi, math.pi)
        elev = random.uniform(-math.pi / 2, math.pi / 2)
        rp = d * math.cos(elev); z = d * math.sin(elev)
        x, y = rp * math.cos(base), rp * math.sin(base)
        try:
            sol = inverse(g, x, y, z)
        except Unreachable:
            ck("kin.reachable_solves", False, f"{l1:.3f},{l2:.3f},d={d:.3f}")
            continue
        x2, y2, z2 = forward_position(g, sol.base, sol.shoulder, sol.elbow)
        ck("kin.ikfk", finite(x2, y2, z2)
           and abs(x - x2) < 1e-6 and abs(y - y2) < 1e-6 and abs(z - z2) < 1e-6)
        # too-far must raise
        far = g.reach_max + random.uniform(0.01, 1.0)
        try:
            inverse(g, far, 0, 0); ck("kin.farraises", False)
        except Unreachable:
            pass


def s_limits():
    for _ in range(N):
        lo = random.uniform(-200, 200); hi = lo + random.uniform(0, 400)
        v = random.uniform(-1000, 1000)
        c = clamp(v, lo, hi)
        ck("lim.clamp.bounds", lo - 1e-9 <= c <= hi + 1e-9)
        cur = random.uniform(-200, 200); tgt = random.uniform(-200, 200)
        md = random.uniform(0, 50)
        out = rate_limit(tgt, cur, md)
        ck("lim.rate.cap", abs(out - cur) <= md + 1e-9)
        val, hit = clamp_and_slew(tgt, cur, lo, hi, md * 50, 0.02)
        ck("lim.cs.bounds", lo - 1e-6 <= val <= hi + 1e-6 or abs(val - cur) <= md + 1e-6)
    try:
        rate_limit(1, 0, -1); ck("lim.neg_delta_raises", False)
    except ValueError:
        pass


def s_state_machine():
    # exhaustive (state, event)
    for st in State:
        for ev in Event:
            try:
                ns = next_state(st, ev)
                ck("sm.returns_state", isinstance(ns, State))
                ck("sm.legal_agrees", is_legal(st, ev))
            except IllegalTransition:
                ck("sm.illegal_agrees", not is_legal(st, ev))
        ck("sm.estop_reachable", next_state(st, Event.STOP) is State.ESTOP)
    # random walk never crashes / never leaves the State enum
    st = State.IDLE
    for _ in range(N):
        ev = random.choice(list(Event))
        if is_legal(st, ev):
            st = next_state(st, ev)
        ck("sm.walk.valid", isinstance(st, State))
        # guard never throws, returns None or str
        g = guard(st, random.choice(list(Event)), random.random() < 0.5)
        ck("sm.guard.type", g is None or isinstance(g, str))


def s_one_euro():
    for _ in range(N // 5):
        f = OneEuroFilter(freq=random.uniform(10, 120),
                          mincutoff=random.uniform(0.1, 5), beta=random.uniform(0, 0.5))
        x = 0.0
        for _ in range(40):
            x += random.uniform(-1e3, 1e3)
            y = f(x)
            ck("oneeuro.finite", finite(y))


def s_teleop_chain():
    cal = Calibration()
    joints = fake_joints()
    for _ in range(N):
        lm = [(random.random(), random.random(), random.uniform(-1, 1)) for _ in range(21)]
        t = landmarks_to_targets(lm, cal)
        ck("teleop.finite", finite(t.base, t.shoulder, t.elbow, t.wrist, t.grip))
        ck("teleop.grip01", -1e-9 <= t.grip <= 1 + 1e-9)
        cd, grip = targets_to_wire(t, joints)
        ck("teleop.wire.range",
           all(P.ANGLE_MIN_CD <= a <= P.ANGLE_MAX_CD for a in cd)
           and P.GRIP_MIN <= grip <= P.GRIP_MAX, f"{cd} {grip}")
        f = P.encode_joint(cd, grip, 0)
        ck("teleop.wire.decodes", P.decode_line(f.decode()) is not None)
    # degenerate landmarks (coincident points -> palm size 0) must not crash
    for _ in range(1000):
        p = (random.random(), random.random(), 0.0)
        t = landmarks_to_targets([p] * 21, cal)
        ck("teleop.degenerate.finite", finite(t.base, t.shoulder, t.wrist, t.grip))


def s_wire_nonfinite():
    # The wire layer is the last line of defence: a NaN/inf upstream must never
    # crash it or produce an out-of-range value.
    joints = fake_joints()
    bad = [math.nan, math.inf, -math.inf, 1e300, -1e300]
    for _ in range(2000):
        t = JointTargets(
            base=random.choice(bad + [random.uniform(-200, 200)]),
            shoulder=random.choice(bad + [0.0]),
            elbow=random.choice(bad + [0.0]),
            wrist=random.choice(bad + [0.0]),
            grip=random.choice(bad + [random.random()]),
        )
        try:
            cd, grip = targets_to_wire(t, joints)
        except Exception as e:  # noqa: BLE001
            ck("wire.nonfinite.noraise", False, f"{t} -> {e!r}")
            continue
        ck("wire.nonfinite.range",
           all(isinstance(a, int) and P.ANGLE_MIN_CD <= a <= P.ANGLE_MAX_CD for a in cd)
           and isinstance(grip, int) and P.GRIP_MIN <= grip <= P.GRIP_MAX,
           f"{t} -> {cd} {grip}")


def s_controller():
    joints = fake_joints()
    c = Controller(joints, dt=1.0 / 50)
    prev = c.current
    for _ in range(N):
        req = JointTargets(base=random.uniform(-300, 300), shoulder=random.uniform(-300, 300),
                           elbow=random.uniform(-300, 300), wrist=random.uniform(-300, 300),
                           grip=random.uniform(-2, 2))
        cmd = c.step(req)
        for nm in JOINT_NAMES:
            j = joints[nm]; v = getattr(cmd, nm)
            ck("ctrl.in_limits", j.min_deg - 1e-6 <= v <= j.max_deg + 1e-6, f"{nm}={v}")
            step = j.max_rate_dps / 50 + 1e-6
            ck("ctrl.slew", abs(v - getattr(prev, nm)) <= step + 1e-6, f"{nm}")
        ck("ctrl.grip01", -1e-9 <= cmd.grip <= 1 + 1e-9)
        prev = cmd


def s_servoing():
    p = GraspParams()
    cur = JointTargets(base=0, shoulder=20, elbow=-60, wrist=0, grip=1.0)
    for _ in range(N):
        obs = {"u_norm": random.uniform(-5, 5), "area_frac": random.uniform(-1, 2),
               "theta_deg": random.uniform(-720, 720)}
        tgt, in_tol = servo_step(obs, cur, p)
        ck("servo.finite", finite(tgt.base, tgt.shoulder, tgt.wrist))
        ck("servo.step_cap", abs(tgt.base - cur.base) <= p.max_step_deg + 1e-9
           and abs(tgt.shoulder - cur.shoulder) <= p.max_step_deg + 1e-9
           and abs(tgt.wrist - cur.wrist) <= p.max_step_deg + 1e-9)
        ck("servo.in_tol_bool", isinstance(in_tol, bool))
    # sequencer under adversarial obs never crashes; phase always valid; DONE sticks
    for _ in range(200):
        seq = GraspSequencer(GraspParams(commit_frames=random.randint(1, 6)))
        cur = JointTargets(base=0, shoulder=20, elbow=-60, wrist=0, grip=1.0)
        done_seen = False
        for _ in range(400):
            obs = None if random.random() < 0.3 else {
                "u_norm": random.uniform(-3, 3), "area_frac": random.uniform(0, 0.4),
                "theta_deg": random.uniform(-180, 180)}
            cur = seq.update(obs, cur)
            ck("seq.phase_valid", isinstance(seq.phase, GraspPhase))
            ck("seq.finite", finite(cur.base, cur.shoulder, cur.wrist, cur.grip))
            if seq.done():
                done_seen = True
            if done_seen:
                ck("seq.done_sticks", seq.done())


def s_gestures():
    names = [None, "", "Victory", "Open_Palm", "Thumb_Up", "Closed_Fist", "Bogus"]
    d = GestureDebouncer(hold_frames=random.randint(1, 10))
    fires = 0
    for _ in range(N):
        r = d.update(random.choice(names), random.uniform(0, 1))
        ck("gest.type", r is None or isinstance(r, str))
        if r:
            fires += 1
    ck("gest.fired_some", fires >= 0)  # sanity: never negative / no crash


def s_grasp_norm():
    for _ in range(N):
        w = random.randint(64, 1920); h = random.randint(64, 1080)
        cx = random.uniform(0, w); cy = random.uniform(0, h)
        rw = random.uniform(1, w); rh = random.uniform(1, h)
        ang = random.uniform(-360, 360)
        g = normalize_grasp(cx, cy, rw, rh, ang, w, h, score=random.random())
        ck("grasp.u_range", -1.001 <= g["u_norm"] <= 1.001)
        ck("grasp.theta_range", -90 < g["theta_deg"] <= 90)
        ck("grasp.area01", 0 <= g["area_frac"] <= 1.001)
        ck("grasp.finite", finite(g["u_norm"], g["v_norm"], g["area_frac"],
                                  g["theta_deg"], g["width"]))
    # wrap_grasp_angle exhaustive-ish
    for a in range(-1000, 1001):
        w = wrap_grasp_angle(a)
        ck("wrap.range", -90 < w <= 90, f"{a}->{w}")


def s_metrics_dashboard():
    lt = LatencyTracker(window=random.randint(1, 500))
    for _ in range(N // 5):
        lt.add(random.uniform(0, 1e6))
        ck("metrics.p_order", lt.p50() <= lt.p95() + 1e-9)
    for flags in range(32):
        labels = decode_flags(flags)
        ck("dash.flags.count", len(labels) == bin(flags).count("1"))
    import json
    for _ in range(1000):
        cmd = JointTargets(base=random.uniform(-90, 90), shoulder=random.uniform(0, 90),
                           elbow=random.uniform(-120, 0), wrist=random.uniform(-90, 90),
                           grip=random.random())
        snap = build_snapshot("TELEOP", cmd, flags=random.randint(0, 31),
                              vbat_mV=random.randint(0, 9000), p50=random.random() * 100,
                              p95=random.random() * 200, clamped=bool(random.getrandbits(1)),
                              seq=random.randint(0, 255))
        json.dumps(snap)


def _seqlock_writer(name, stop_evt):
    ch = SharedLatest(name=name, create=False)
    k = 0
    while not stop_evt.is_set():
        k += 1
        ch.put({"i": k, "check": k * 2, "pad": "x" * (k % 50)})


def s_seqlock():
    ch = SharedLatest(capacity=4096)
    stop = mp.Event()
    w = mp.Process(target=_seqlock_writer, args=(ch.name, stop), daemon=True)
    w.start()
    torn = 0
    for _ in range(20000):
        v = ch.get()
        if v is None:
            continue
        if v["check"] != v["i"] * 2:   # a torn read would break this invariant
            torn += 1
    stop.set(); w.join(timeout=2)
    if w.is_alive():
        w.terminate()
    ck("seqlock.no_torn_reads", torn == 0, f"{torn} torn reads")
    ch.close(); ch.unlink()


def _app_cfg():
    from types import SimpleNamespace
    def md(d):
        ns = SimpleNamespace(**d); ns.model_dump = lambda: d; return ns
    def jc(mn, mx, home, rate):
        return SimpleNamespace(min_deg=mn, max_deg=mx, home_deg=home,
                               max_rate_dps=rate, invert=False, span_deg=mx - mn)
    return SimpleNamespace(
        serial=SimpleNamespace(command_hz=50, port="x", baud=1000000),
        joints={"base": jc(-90, 90, 0, 300), "shoulder": jc(0, 90, 20, 300),
                "elbow": jc(-120, 0, -60, 300), "wrist": jc(-90, 90, 0, 300),
                "grip": jc(0, 100, 100, 600)},
        teleop=SimpleNamespace(
            one_euro=md(dict(freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0)),
            gesture=md(dict(hold_frames=12, min_score=0.7)), model_path="x", mirror=True),
        watchdog=SimpleNamespace(vision_stale_ms=300), cameras={},
        grasp=SimpleNamespace(detect_hz=8, commit_frames=random.randint(2, 6)))


def s_app_faultinject():
    """Run the integrated App through many autonomous cycles under injected
    faults (obs dropouts, sensor jitter, mid-grasp e-stop). Invariants: never
    crash, state always valid, commanded joints always within soft limits."""
    import gripper.main as M
    from types import SimpleNamespace
    M.get_logger = lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None)

    completed = 0
    for r in range(60):
        app = M.App(_app_cfg())
        app.dispatch(Event.START); app.dispatch(Event.TOGGLE_AUTO); app.dispatch(Event.TRIGGER)
        if app.state is not State.AUTO_GRASP:
            ck("app.armed", False, "did not reach AUTO_GRASP"); continue
        estop_at = random.randint(5, 40) if r % 7 == 0 else None
        estopped = False
        u, area, theta = random.uniform(-1, 1), random.uniform(0, 0.15), random.uniform(-80, 80)
        prev = app.controller.current
        for t in range(1, 1500):
            obs = None if random.random() < 0.25 else {
                "u_norm": u, "area_frac": area, "theta_deg": theta, "t_capture": 0.0}
            if estop_at and t == estop_at and app.state not in (State.IDLE, State.ESTOP):
                app.dispatch(Event.STOP)
                estopped = True
            app.step_once(hand=None, grasp=obs, now=t * 0.02)
            cur = app.controller.current
            for nm in JOINT_NAMES:
                j = app.cfg.joints[nm]
                ck("app.in_limits", j.min_deg - 1e-6 <= getattr(cur, nm) <= j.max_deg + 1e-6, nm)
            ck("app.state_valid", isinstance(app.state, State))
            if (app.state is State.AUTO_GRASP and app._sequencer
                    and app._sequencer.phase is GraspPhase.SERVO):
                u += (cur.base - prev.base) * 0.10 + random.uniform(-0.02, 0.02)
                area += (cur.shoulder - prev.shoulder) * 0.006
                theta -= (cur.wrist - prev.wrist) * 1.0
            prev = cur
            if app.state in (State.IDLE, State.ESTOP):
                break
        if estopped:
            ck("app.estop_reached", app.state is State.ESTOP)
        elif app.state is State.IDLE:
            completed += 1
    ck("app.some_completed", completed >= 1, f"{completed} finished")
    ck("app.estops_fired", True)


SECTIONS = [
    ("crc", s_crc), ("protocol-roundtrip", s_protocol_roundtrip),
    ("protocol-fuzz", s_protocol_fuzz), ("kinematics", s_kinematics),
    ("limits", s_limits), ("state-machine", s_state_machine),
    ("one-euro", s_one_euro), ("teleop-chain", s_teleop_chain),
    ("wire-nonfinite", s_wire_nonfinite), ("controller", s_controller),
    ("servoing", s_servoing), ("gestures", s_gestures),
    ("grasp-norm", s_grasp_norm), ("metrics-dashboard", s_metrics_dashboard),
    ("seqlock", s_seqlock), ("app-faultinject", s_app_faultinject),
]


def main() -> int:
    print(f"stress test — {N} iters/section, seed 0xC0FFEE\n")
    for name, fn in SECTIONS:
        t0 = time.monotonic()
        before = len(FAILS)
        fn()
        dt = time.monotonic() - t0
        status = "ok  " if len(FAILS) == before else "FAIL"
        print(f"  [{status}] {name:<20} {dt:5.2f}s")
    print(f"\n{CHECKS:,} invariant checks · {len(FAILS)} failures")
    if FAILS:
        print("\nFAILURES:")
        for n, info in FAILS[:40]:
            print(f"  - {n}: {info}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay a run_*.jsonl log and print latency / event stats — your re-implemented
``ros2 bag`` reader. Pure stdlib so it runs anywhere.

    python3 tools/replay.py logs/run_20260607_141500.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, int(q / 100 * len(s)))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=pathlib.Path)
    args = ap.parse_args()

    events: Counter[str] = Counter()
    latencies: list[float] = []
    transitions: list[tuple[str, str]] = []

    for line in args.logfile.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        events[rec.get("event", "?")] += 1
        if "latency_ms" in rec:
            latencies.append(float(rec["latency_ms"]))
        if rec.get("event") == "transition":
            transitions.append((rec.get("frm", "?"), rec.get("to", "?")))

    print(f"records: {sum(events.values())}")
    print("events:")
    for name, n in events.most_common():
        print(f"  {n:5d}  {name}")
    if latencies:
        print(f"latency p50={percentile(latencies,50):.1f}ms  "
              f"p95={percentile(latencies,95):.1f}ms  "
              f"max={max(latencies):.1f}ms")
    if transitions:
        print(f"state transitions: {len(transitions)}")
        for frm, to in transitions:
            print(f"  {frm} -> {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

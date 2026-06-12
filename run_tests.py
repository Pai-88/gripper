#!/usr/bin/env python3
"""Zero-dependency test runner — no pytest required.

Discovers ``tests/test_*.py``, runs every ``test_*`` function, and reports
pass/fail. ``pytest`` works too (and is the documented runner); this exists so
the pure-logic suite can be verified on a bare Pi with only the stdlib.

    python3 run_tests.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    passed = failed = 0
    failures: list[tuple[str, str, str]] = []

    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"tests.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001
            failed += 1
            failures.append((path.stem, "<import>", traceback.format_exc()))
            continue
        for name in sorted(vars(module)):
            if name.startswith("test_") and callable(getattr(module, name)):
                try:
                    getattr(module, name)()
                    passed += 1
                except Exception:  # noqa: BLE001
                    failed += 1
                    failures.append((path.stem, name, traceback.format_exc()))

    print(f"\n{passed} passed, {failed} failed")
    for mod, name, tb in failures:
        print(f"\n--- FAIL {mod}::{name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

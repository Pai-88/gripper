"""Tiny assert helpers so the suite runs under both ``pytest`` and the bundled
zero-dependency ``run_tests.py`` (no pytest import required in test modules)."""

from __future__ import annotations


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def assert_approx(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tol {tol}) {msg}")


def assert_raises(exc, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc.__name__}, nothing raised")

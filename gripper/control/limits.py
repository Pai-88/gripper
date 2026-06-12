"""Pure functions for clamping and rate-limiting joint commands.

These run on the Pi (primary defence) and are re-implemented independently on
the ESP32 (belt-and-braces) — never trust the host. See the build guide,
"Limits, watchdog, e-stop — defense in depth".
"""

from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    """Constrain ``value`` to ``[lo, hi]``. ``lo``/``hi`` may be passed in any order."""
    if lo > hi:
        lo, hi = hi, lo
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def rate_limit(target: float, current: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta`` (>= 0)."""
    if max_delta < 0:
        raise ValueError("max_delta must be >= 0")
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


def slew(target: float, current: float, max_rate_dps: float, dt: float) -> float:
    """Rate-limit expressed as a maximum angular rate (deg/s) over a timestep ``dt``."""
    return rate_limit(target, current, max_rate_dps * dt)


def clamp_and_slew(target: float, current: float, lo: float, hi: float,
                   max_rate_dps: float, dt: float) -> tuple[float, bool]:
    """Clamp to limits then slew. Returns ``(value, clamped)`` where ``clamped``
    flags that the raw target hit a soft limit (surfaced as FLAG_LIMIT_CLAMP)."""
    clamped_target = clamp(target, lo, hi)
    hit = clamped_target != target
    return slew(clamped_target, current, max_rate_dps, dt), hit

"""Pi-side watchdog: forces an ESTOP event if the active camera feed goes stale
or the MCU stops acking. This is the *soft* net; the hard net is the ESP32's own
200 ms firmware watchdog, which halts the servos even if this whole Python
process is dead.
"""

from __future__ import annotations

import time
from typing import Callable


class Watchdog:
    def __init__(self, vision_stale_s: float, on_trip: Callable[[str], None]):
        self.vision_stale_s = vision_stale_s
        self.on_trip = on_trip
        self._last_vision = time.monotonic()
        self._tripped = False

    def feed_vision(self, t: float | None = None) -> None:
        self._last_vision = t if t is not None else time.monotonic()
        self._tripped = False

    def check(self, now: float | None = None) -> bool:
        """Call each control tick. Returns True (and fires ``on_trip`` once) if
        the active feed has been stale longer than ``vision_stale_s``."""
        now = now if now is not None else time.monotonic()
        if not self._tripped and (now - self._last_vision) > self.vision_stale_s:
            self._tripped = True
            self.on_trip("vision_stale")
            return True
        return False

"""VL53L5CX 8x8 time-of-flight range — plunge gating for Mode 2 (build guide §5).

Honest scope: an 8x8 grid is a genuine *range* signal along the approach axis,
ideal for gating the open-loop plunge precisely (and for a coarse contact check).
It is NOT a dense depth image — upsampling 64 zones to a 300x300 GG-CNN depth
channel is a stand-in only, so RGB GG-CNN stays the primary path, not true RGB-D.

The decision maths (``center_distance_mm``, ``plunge_ready``) is pure and
unit-tested; the hardware reader lazy-imports the vendor driver so this module
imports anywhere.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

GRID = 8  # VL53L5CX in 8x8 mode = 64 zones


def _median(values: Sequence[float]) -> Optional[float]:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def center_distance_mm(grid, k: int = 2) -> Optional[float]:
    """Median of the central ``k``x``k`` zones of the 8x8 grid, in mm. Ignores
    non-positive readings (the sensor flags invalid zones as 0/negative); returns
    None if every central zone is invalid. Median, not mean, so one dead zone
    can't drag the range."""
    start = (GRID - k) // 2
    vals: List[float] = []
    for r in range(start, start + k):
        for c in range(start, start + k):
            d = float(grid[r][c])
            if d > 0:
                vals.append(d)
    return _median(vals)


def plunge_ready(grid, fire_mm: float, k: int = 2) -> bool:
    """True once the centre of the field is within ``fire_mm`` — the precise
    plunge trigger the open-loop tick count only approximates."""
    d = center_distance_mm(grid, k)
    return d is not None and d <= fire_mm


def upsample(grid, out_w: int, out_h: int):
    """Coarse nearest-neighbour upsample of the 8x8 grid to (out_h, out_w) as a
    stand-in depth image. Documented as coarse — see the module note."""
    import numpy as np

    a = np.asarray(grid, dtype="float32")
    rows = (np.arange(out_h) * GRID // out_h).clip(0, GRID - 1)
    cols = (np.arange(out_w) * GRID // out_w).clip(0, GRID - 1)
    return a[rows][:, cols]


class VL53L5CX:
    """Thin wrapper over the Pimoroni ``vl53l5cx_ctypes`` driver. Lazy-imports the
    driver and starts ranging in 8x8 mode."""

    def __init__(self, ranging_freq_hz: int = 15, address: int = 0x29):
        try:
            import vl53l5cx_ctypes as vl53
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"missing '{e.name}' — install the ToF driver on the Pi: "
                "pip install vl53l5cx-ctypes  (needs I2C enabled)")

        self._dev = vl53.VL53L5CX(i2c_addr=address)
        self._dev.set_resolution(GRID * GRID)          # 64 = 8x8
        self._dev.set_ranging_frequency_hz(ranging_freq_hz)
        self._dev.start_ranging()

    def read_grid(self):
        """Latest 8x8 distance grid (mm) as a numpy array. Blocks briefly until a
        frame is ready."""
        import time

        import numpy as np

        while not self._dev.data_ready():
            time.sleep(0.001)
        data = self._dev.get_data()
        return np.array(data.distance_mm, dtype="float32").reshape((GRID, GRID))

    def center_distance_mm(self, k: int = 2) -> Optional[float]:
        return center_distance_mm(self.read_grid(), k)

    def depth_image(self, out_w: int, out_h: int):
        return upsample(self.read_grid(), out_w, out_h)

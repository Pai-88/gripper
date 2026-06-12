"""Round-trip the seqlock shared-memory channel (pure stdlib, no extra deps)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.shared_state import SharedLatest  # noqa: E402
from tests._util import assert_raises  # noqa: E402


def test_put_get_roundtrip():
    ch = SharedLatest(capacity=1024)
    try:
        assert ch.get() is None  # nothing written yet
        ch.put({"landmarks": [[0.1, 0.2, 0.0]], "gesture": "Victory"})
        got = ch.get()
        assert got["gesture"] == "Victory"
        assert got["landmarks"][0][1] == 0.2
    finally:
        ch.close()
        ch.unlink()


def test_latest_wins():
    ch = SharedLatest(capacity=256)
    try:
        for i in range(5):
            ch.put({"i": i})
        assert ch.get()["i"] == 4
    finally:
        ch.close()
        ch.unlink()


def test_oversize_payload_rejected():
    ch = SharedLatest(capacity=16)
    try:
        assert_raises(ValueError, ch.put, {"x": "y" * 100})
    finally:
        ch.close()
        ch.unlink()


def test_reader_can_attach_by_name():
    writer = SharedLatest(capacity=256)
    try:
        writer.put({"hello": "world"})
        reader = SharedLatest(name=writer.name, create=False)
        try:
            assert reader.get()["hello"] == "world"
        finally:
            reader.close()
    finally:
        writer.close()
        writer.unlink()

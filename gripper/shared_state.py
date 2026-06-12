"""Cross-process "latest value" channels backed by ``multiprocessing.shared_memory``.

The vision processes are CPU-heavy and must not block the control loop, so they
publish their newest result into a fixed-size shared buffer guarded by a
**seqlock**: the writer bumps an odd sequence before writing and an even one
after; a reader retries if the sequence is odd or changed mid-read. This gives
a wait-free "newest wins" channel (build guide: "shared state = one writer per
field", seqlock).

Payloads are JSON-encoded into a fixed capacity for simplicity; swap to a packed
struct if you need the last few microseconds.
"""

from __future__ import annotations

import json
import struct
from multiprocessing import shared_memory
from typing import Any, Optional

_HEADER = struct.Struct("<II")  # (seq, length)


class SharedLatest:
    """Single-writer / single-reader latest-value channel."""

    def __init__(self, name: Optional[str] = None, capacity: int = 4096, create: bool = True):
        self.capacity = capacity
        size = _HEADER.size + capacity
        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=size, name=name)
            _HEADER.pack_into(self._shm.buf, 0, 0, 0)  # seq=0 (even), len=0
        else:
            self._shm = shared_memory.SharedMemory(name=name)
        self.name = self._shm.name

    def put(self, obj: Any) -> None:
        payload = json.dumps(obj).encode("utf-8")
        if len(payload) > self.capacity:
            raise ValueError(f"payload {len(payload)}B exceeds capacity {self.capacity}B")
        seq, _ = _HEADER.unpack_from(self._shm.buf, 0)
        _HEADER.pack_into(self._shm.buf, 0, seq + 1, 0)        # odd: write in progress
        self._shm.buf[_HEADER.size:_HEADER.size + len(payload)] = payload
        _HEADER.pack_into(self._shm.buf, 0, seq + 2, len(payload))  # even: done

    def get(self) -> Optional[Any]:
        for _ in range(8):  # bounded retries; writer is fast
            seq1, length = _HEADER.unpack_from(self._shm.buf, 0)
            if seq1 & 1 or length == 0:        # write in progress or never written
                continue
            data = bytes(self._shm.buf[_HEADER.size:_HEADER.size + length])
            seq2, _ = _HEADER.unpack_from(self._shm.buf, 0)
            if seq1 == seq2:
                return json.loads(data.decode("utf-8"))
        return None

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

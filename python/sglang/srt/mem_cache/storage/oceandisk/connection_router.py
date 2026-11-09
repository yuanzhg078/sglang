# SPDX-License-Identifier: Apache-2.0
"""GIDS DPU hash ring matching od/nds_utils.py."""

from bisect import bisect_left
from functools import lru_cache, partial
from pathlib import Path
from threading import Lock


class ConnectionRouter:
    def __init__(self, config_path, connection_indices):
        try:
            from xxhash3_py import xxh3_64_hexdigest
        except ImportError:
            from xxhash import xxh3_64_hexdigest

        self.groups = {}
        self.endpoints = {}
        index = 0
        for raw in Path(config_path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or not line.startswith("ip_config"):
                continue
            fields = [part.strip() for part in line.split("=", 1)[1].split(",")]
            if index in connection_indices:
                dpu = int(fields[5])
                self.groups.setdefault(dpu, []).append(index)
                self.endpoints[index] = (fields[1], int(fields[2]), dpu)
            index += 1
        self.ring = sorted(
            (int(xxh3_64_hexdigest(f"DPU-{dpu}-V-{vnode}".encode("utf-8")), 16), dpu)
            for dpu in self.groups
            for vnode in range(50)
        )
        self.coordinates = [coord for coord, _ in self.ring]
        self._find_ring_index = lru_cache(maxsize=65536)(
            partial(bisect_left, self.coordinates)
        )
        self.next_connection = [0] * len(self.ring)
        self.lock = Lock()

    def _ring_index(self, key):
        return self._find_ring_index(key % (1 << 64)) % len(self.ring)

    def dpu_for_key(self, key):
        return self.ring[self._ring_index(key)][1]

    def select(self, key):
        index = self._ring_index(key)
        connections = self.groups[self.ring[index][1]]
        with self.lock:
            connection = connections[self.next_connection[index] % len(connections)]
            self.next_connection[index] += 1
        return connection

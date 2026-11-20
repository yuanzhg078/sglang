"""Registered device memory and consumable tasks for the KvClient SDK."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

# kv::StatusCode values returned by _kvclient.
_TIMEOUT_CODES = {3, 0x10716}  # TIMEOUT, CQE_IO_TIMEOUT
_UNCONFIRMED_WAIT_CODES = {2, 10, 13}  # NOT_INITIALIZED, TASK_NOT_FOUND, IN_PROGRESS


@dataclass(frozen=True)
class KvClientRange:
    key: bytes
    addr: int
    size: int
    device_id: int
    remote_offset: int


class KvClientIOError(RuntimeError):
    def __init__(self, result, *, timed_out=False):
        super().__init__(f"KvClient task failed: {result}")
        self.result = result
        self.timed_out = timed_out


class KvClientStorage:
    def __init__(self, config_path, device_id, *, client=None):
        if client is None:
            import _kvclient

            client = _kvclient.Client(config_path, device_id)
        self.client = client
        self.registered_regions = []
        self.buffer_sources = {}
        self.active_tasks = {}
        self._waiting = set()
        self._lock = Lock()
        self._accepting = True
        self._closed = False
        self._unsafe_tasks = {}

    def register_buffers(self, buffers):
        allocations = {}
        for name, buffer in buffers:
            allocation = buffer.untyped_storage()
            region = (
                int(allocation.data_ptr()),
                int(allocation.nbytes()),
                buffer.device.index,
            )
            allocations.setdefault(region, []).append((name, buffer))
        registered = self.client.register_regions(list(allocations))
        self.registered_regions = registered
        self.buffer_sources = allocations
        if (
            len(registered) != len(allocations)
            or {tuple(item["region"]) for item in registered} != set(allocations)
            or any(not item["handle"] for item in registered)
        ):
            raise RuntimeError("KvClient returned incomplete memory registrations")

    def _build_entries(self, ranges):
        entries = []
        for item in ranges:
            if len(item.key) != 8:
                raise ValueError("KvClient keys must contain exactly 8 bytes")
            if (
                item.addr <= 0
                or item.size <= 0
                or item.remote_offset < 0
                or item.remote_offset + item.size > 1 << 32
                or item.remote_offset % 512
            ):
                raise ValueError("Invalid IO range or unaligned uint32 remote offset")
            for registered in self.registered_regions:
                base, size, device_id = registered["region"]
                if (
                    device_id == item.device_id
                    and base <= item.addr
                    and item.addr + item.size <= base + size
                ):
                    entries.append(
                        (
                            item.key,
                            item.addr,
                            item.size,
                            item.device_id,
                            registered["handle"],
                            item.remote_offset,
                        )
                    )
                    break
            else:
                raise ValueError(f"IO range is outside registered memory: {item}")
        return entries

    def _submit(self, method, payload, query=False):
        # Serialize acceptance, never hold this lock across a blocking Wait.
        with self._lock:
            if not self._accepting or self._unsafe_tasks:
                raise RuntimeError("KvClient storage is stopped or has unsafe IO")
            task_id = method(payload)
            if not task_id or task_id in self.active_tasks:
                raise RuntimeError(f"Invalid KvClient task ID: {task_id}")
            self.active_tasks[task_id] = (query, payload)
            return task_id

    def get_async(self, ranges):
        return self._submit(self.client.load_async, self._build_entries(ranges))

    def put_async(self, ranges):
        return self._submit(self.client.store_async, self._build_entries(ranges))

    def exists_many(self, keys):
        if not keys:
            return []
        task = self._submit(self.client.query_async, keys, query=True)
        try:
            result = self.wait(task)
        except KvClientIOError as error:
            if error.timed_out:
                return [False] * len(keys)
            raise
        exists = result.get("exists")
        if exists is None:
            prefix = result["prefix_hit_keys"]
            if not 0 <= prefix <= len(keys):
                raise KvClientIOError(result)
            return [i < prefix for i in range(len(keys))]
        if len(exists) != len(keys) or any(value not in (0, 1) for value in exists):
            raise KvClientIOError(result)
        return [bool(value) for value in exists]

    def wait(self, task_id, timeout_ms=0):
        with self._lock:
            if task_id in self._waiting:
                raise RuntimeError(f"Task {task_id} is already being consumed")
            query, payload = self.active_tasks[task_id]
            self._waiting.add(task_id)
        try:
            try:
                result = self.client.wait(task_id, timeout_ms)
            except BaseException:
                with self._lock:
                    self._unsafe_tasks[task_id] = (query, payload)
                raise
            wait_code = result["wait_status"]["code"]
            task_code = result["status"]["code"]
            entry_codes = [status["code"] for status in result["entry_status"]]
            timed_out = bool(
                _TIMEOUT_CODES.intersection((wait_code, task_code, *entry_codes))
            )
            # A transport timeout can be reported as PARTIAL_FAILED by SDK Wait.
            # Query only submits keys, so a timed-out query owns no device ranges.
            if wait_code in _UNCONFIRMED_WAIT_CODES or (timed_out and not query):
                with self._lock:
                    self._unsafe_tasks[task_id] = (query, payload)
                raise KvClientIOError(result, timed_out=timed_out)
            allowed = {0, 0x300, 0x301} if query else {0}
            if (
                timed_out
                or wait_code != 0
                or task_code != 0
                or len(entry_codes) != len(payload)
                or any(code not in allowed for code in entry_codes)
            ):
                raise KvClientIOError(result, timed_out=timed_out)
            return result
        finally:
            with self._lock:
                self.active_tasks.pop(task_id)
                self._waiting.remove(task_id)

    @property
    def has_unsafe_io(self):
        with self._lock:
            return bool(self._unsafe_tasks)

    @property
    def io_quiescent(self):
        with self._lock:
            return not self.active_tasks and not self._unsafe_tasks

    def close(self):
        if self._closed:
            return
        with self._lock:
            self._accepting = False
            tasks = list(self.active_tasks)
        error = None
        for task in tasks:
            try:
                self.wait(task)
            except KvClientIOError as exc:
                error = exc
        if not self.io_quiescent:
            raise RuntimeError(
                "KvClient IO is not quiescent; retain buffers and MRs"
            ) from error
        self.client.unregister_regions(
            [item["handle"] for item in self.registered_regions]
        )
        self.client.shutdown()
        self.registered_regions.clear()
        self.buffer_sources.clear()
        self._closed = True

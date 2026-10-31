# SPDX-License-Identifier: Apache-2.0
"""OceanDisk direct-GPU storage with a reserved HiCache interface."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import itertools
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.storage.oceandisk.connection_router import ConnectionRouter

if TYPE_CHECKING:
    import torch

MAX_GIDS_CONNECTIONS = 8  # od/gids_kernel.cu: MAX_CONN_NUM
MAX_GIDS_RANGES = 409600  # od/gids_kernel.cu: MAX_BLOCK_NUM
logger = logging.getLogger(__name__)


def trace_event(event, **details):
    if os.environ.get("SGLANG_OCEANDISK_TRACE") == "1":
        logger.info(
            "[OceanDisk trace] event=%s pid=%d %s",
            event,
            os.getpid(),
            json.dumps(details, default=str, ensure_ascii=True),
        )


def trace_batch(event, batch, index, total, native_module):
    if os.environ.get("SGLANG_OCEANDISK_TRACE") != "1":
        return
    row = batch[0]
    count = (len(row) - 1) // 6
    samples = {}
    for name, position in (("kv", 0), ("indexer", count // 2)):
        samples[name] = {
            field: row[1 + column * count + position]
            for column, field in enumerate(
                ("mr", "key", "local_offset", "remote_offset", "bytes", "connection")
            )
        }
    trace_event(
        event,
        batch=index + 1,
        batches=total,
        layers=len(batch),
        io_per_layer=count,
        first_task_id=row[0],
        native_source=getattr(native_module, "__file__", "unknown"),
        first_layer_samples=samples,
    )


def gids_config_path(config_dir, device_index: int, use_single_port=None) -> Path:
    """Match gids_kernel.cu: unset USE_SINGLE_PORT defaults to single-port."""
    single = use_single_port is None or use_single_port == "1"
    filename = "gids_config.txt" if single else f"gids_config_{device_index}.txt"
    return Path(config_dir) / filename


def read_connection_count(path: Path) -> int:
    """Fail before native init, which otherwise logs bad config and may succeed."""
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "ip_config" not in line:
            continue
        name, sep, value = line.partition("=")
        fields = [field.strip() for field in value.split(",")]
        if (
            not raw.startswith("ip_config")
            or len(raw.encode()) >= 256
            or not sep
            or name.strip() != "ip_config"
            or len(fields) != 6
        ):
            raise ValueError(f"Invalid GIDS ip_config at {path}:{number}.")
        try:
            port, gid, roce_port, qp_count = map(int, fields[2:])
        except ValueError as error:
            raise ValueError(
                f"Invalid GIDS connection numbers at {path}:{number}."
            ) from error
        if (
            not fields[0]
            or not fields[1]
            or len(value.split(",")[0].lstrip().encode()) >= 32
            or len(value.split(",")[1].lstrip().encode()) >= 32
            or not 0 < port <= 65535
            or not 0 <= gid < 256
            or roce_port < 0
            or qp_count < 0
        ):
            raise ValueError(f"Invalid GIDS connection fields at {path}:{number}.")
        count += 1
    if not 1 <= count <= MAX_GIDS_CONNECTIONS:
        raise ValueError(
            f"OceanDisk requires 1..{MAX_GIDS_CONNECTIONS} ip_config entries: {path}."
        )
    return count


@dataclass(frozen=True)
class OceanDiskRange:
    key: int
    ptr: int
    size: int
    storage_offset: int = 0


def encode_key(identity: tuple) -> int:
    """Encode an object identity as a positive int64."""
    data = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    return (
        int.from_bytes(hashlib.sha256(data).digest()[:8], "big") & ((1 << 63) - 1)
    ) or 1


def resolve_mr(regions: list[tuple[int, int]], ptr: int, size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("OceanDisk ranges must have positive length.")
    for mr_id, (base, length) in enumerate(regions):
        if base <= ptr and ptr + size <= base + length:
            return mr_id, ptr - base
    raise ValueError(f"OceanDisk range ptr={ptr:#x}, size={size} is not registered.")


def pack_io(
    task_id: int,
    ranges: list[OceanDiskRange],
    regions: list[tuple[int, int]],
    connection_indices: tuple[int, ...],
    *,
    resolve=None,
    select_connection=None,
) -> list[int]:
    """Column-major ABI used by od/cache_controller.py, with remote offsets."""
    half = len(ranges) // 2
    if (
        not ranges
        or len(ranges) % 2
        or [r.key for r in ranges[:half]] != [r.key for r in ranges[half:]]
    ):
        raise ValueError("OceanDisk requires matching KV and Indexer halves.")
    mr_offsets = [
        resolve(item.ptr, item.size)
        if resolve is not None
        else resolve_mr(regions, item.ptr, item.size)
        for item in ranges
    ]
    if any(item.storage_offset < 0 for item in ranges):
        raise ValueError("OceanDisk storage offsets must be nonnegative.")
    paired_connections = [
        select_connection(item.key)
        if select_connection is not None
        else connection_indices[item.key % len(connection_indices)]
        for item in ranges[:half]
    ]
    return [
        task_id,
        *[mr for mr, _ in mr_offsets],
        *[item.key for item in ranges],
        *[offset for _, offset in mr_offsets],
        *[item.storage_offset for item in ranges],
        *[item.size for item in ranges],
        *paired_connections,
        *paired_connections,
    ]


class OceanDiskStorage(HiCacheStorage):
    """Direct-buffer I/O with separate load/store streams; aligned ranges can overlap pages."""

    _registration_lock = threading.Lock()
    _initialized = False
    _registered_tensors: list = []

    def __init__(
        self,
        config: dict,
        device: torch.device,
        buffers: list[torch.Tensor],
    ):
        connections = config.get("connection_indices", [])
        if "connection_indices" in config and (
            not isinstance(connections, list)
            or not connections
            or any(
                type(index) is not int or not 0 <= index < MAX_GIDS_CONNECTIONS
                for index in connections
            )
            or len(set(connections)) != len(connections)
        ):
            raise ValueError(
                "OceanDisk connection_indices must be distinct integers in [0, 7]."
            )
        self.connection_indices = tuple(connections)
        connection_source = (
            "extra_config" if "connection_indices" in config else "gids_config_all"
        )
        self._closed = False
        self._debug_put_history = OrderedDict()
        self._put_io_lock = threading.Lock()
        self._task_ids = itertools.count(1)

        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The OceanDisk direct linker requires CUDA buffers.")
        buffers = list(buffers)
        self.regions = []
        for buffer in buffers:
            if buffer.device != self.device:
                raise ValueError(
                    "OceanDisk buffers must all be on the linker's CUDA device."
                )
            storage = buffer.untyped_storage()
            region = (int(storage.data_ptr()), int(storage.nbytes()))
            if region[1] and region not in self.regions:
                self.regions.append(region)
        if not self.regions:
            raise ValueError("OceanDisk has no GPU allocations to register.")

        library = config.get("op_extension") or os.environ.get("OPEXTENSION")
        if not library:
            raise ValueError(
                "Set OceanDisk op_extension or OPEXTENSION to libop_extension.so."
            )
        # LIBGIDS must be set before load_library: the library constructor dlopens it.
        for key, env in (("core_library", "LIBGIDS"), ("config_dir", "GIDS_DIR")):
            if config.get(key):
                os.environ[env] = os.fspath(config[key])
        if not os.environ.get("GIDS_DIR"):
            raise ValueError(
                "Set OceanDisk config_dir or GIDS_DIR before initialization."
            )
        # Native code concatenates the directory and filename.
        os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")
        if "use_single_port" in config:
            if type(config["use_single_port"]) is not bool:
                raise ValueError("OceanDisk use_single_port must be a boolean.")
            os.environ["USE_SINGLE_PORT"] = "1" if config["use_single_port"] else "0"
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        config_path = gids_config_path(
            os.environ["GIDS_DIR"], device_index, os.environ.get("USE_SINGLE_PORT")
        )
        connection_count = read_connection_count(config_path)
        if "connection_indices" not in config:
            self.connection_indices = tuple(range(connection_count))
        if any(index >= connection_count for index in self.connection_indices):
            raise ValueError(
                f"OceanDisk connection_indices refer outside {connection_count} connections "
                f"in {config_path}."
            )

        self.connection_router = ConnectionRouter(config_path, self.connection_indices)
        with self._registration_lock, torch.cuda.device(self.device):
            if type(self)._initialized:
                raise RuntimeError(
                    "OceanDisk process-global state is already initialized; restart the "
                    "process rather than reinitializing registered GPU memory."
                )
            torch.ops.load_library(library)
            self._ops = torch.ops.my_ops
            self._get_stream = torch.cuda.Stream(device=self.device)
            self._put_stream = torch.cuda.Stream(device=self.device)
            self._output = torch.zeros(2, dtype=torch.int64, device=self.device)
            addresses, lengths = zip(*self.regions)
            # Match the old controller's MR extension for direct padded I/O.
            # This extends registration only; it does not allocate trailing memory.
            lengths = [length + 512 for length in lengths]
            tensor = torch.tensor(
                [*addresses, *lengths], dtype=torch.int64, device=self.device
            )
            self._get_stream.wait_stream(torch.cuda.current_stream(self.device))
            # Even failed init may have registered memory: retain it and prohibit retry.
            type(self)._initialized = True
            type(self)._registered_tensors = list(buffers)
            with torch.cuda.stream(self._get_stream):
                try:
                    # Match od/nds_utils.py: stagger GIDS connection setup by GPU.
                    time.sleep(device_index * 2)
                    trace_event(
                        "NATIVE_INIT_CALL",
                        device=str(self.device),
                        config_path=str(config_path),
                        configured_connections=connection_count,
                    )
                    status = self._ops.kvcache_init(tensor, self._output)
                finally:
                    self._get_stream.synchronize()
            self._check_status(status, "init")
            self._put_stream.wait_stream(self._get_stream)
            logger.info(
                "[OceanDisk connections] device=%s config=%s configured_count=%d "
                "selected=%s source=%s routing=dht groups=%s endpoints=%s",
                self.device,
                config_path,
                connection_count,
                self.connection_indices,
                connection_source,
                self.connection_router.groups,
                self.connection_router.endpoints,
            )
            logger.info(
                "[OceanDisk streams] device=%s get=%s put=%s",
                self.device,
                self._get_stream.stream_id,
                self._put_stream.stream_id,
            )
            trace_event(
                "STORAGE_INIT",
                source=__file__,
                device=str(self.device),
                op_extension=os.fspath(library),
                config_path=str(config_path),
                connections=self.connection_indices,
                registered_regions=len(self.regions),
            )

        self._io_queue = Queue()
        self._io_thread = threading.Thread(
            target=self._io_worker, daemon=True, name="oceandisk-storage-io"
        )
        self._io_thread.start()

    def _run_io(self, operation, stream):
        self._ensure_open()
        completion = Future()
        self._io_queue.put((operation, stream, completion))
        return completion.result()

    def _io_worker(self):
        while True:
            task = self._io_queue.get()
            try:
                if task is None:
                    return
                operation, stream, completion = task
                # EXISTS stays direct and excludes PUT submissions only.
                io_lock = (
                    self._put_io_lock if stream is self._put_stream else nullcontext()
                )
                try:
                    with (
                        io_lock,
                        self._torch.cuda.device(self.device),
                        self._torch.cuda.stream(stream),
                    ):
                        self._ensure_open()
                        result = operation()
                    completion.set_result(result)
                except BaseException as error:
                    completion.set_exception(error)
            finally:
                self._io_queue.task_done()

    @staticmethod
    def _check_status(status, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"OceanDisk {operation} returned status {status!r}.")

    def exists_many(self, keys: list[int], *, timings=None) -> list[bool]:
        # Exclude new PUT submissions until the synchronous query returns.
        with self._torch.cuda.device(self.device):
            started = time.perf_counter() if timings is not None else 0.0
            with self._put_io_lock:
                if timings is not None:
                    timings["put_lock_ms"] = (time.perf_counter() - started) * 1000
                self._ensure_open()
                started = time.perf_counter() if timings is not None else 0.0
                self._put_stream.synchronize()
                if timings is not None:
                    timings["put_sync_ms"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter() if timings is not None else 0.0
                result = self._exists_many(keys)
                if timings is not None:
                    timings["exists_ms"] = (time.perf_counter() - started) * 1000
                return result

    def _exists_many(self, keys: list[int]) -> list[bool]:
        """Return a prefix-only bitmap from one native batch existence query.

        kvcache_exist returns the contiguous hit prefix in input order, not
        individual key statuses. Entries after the first miss remain False.
        """
        torch = self._torch
        # Metadata uses the querying thread's current stream.
        with torch.cuda.device(self.device):
            self._ensure_open()
            if not keys:
                return []
            connections = [self.connection_router.select(key) for key in keys]
            if os.environ.get("SGLANG_OCEANDISK_CACHE_DEBUG") == "1":
                logger.info(
                    "[OceanDisk connection usage] pid=%d device=%s keys=%d counts=%s",
                    os.getpid(),
                    self.device,
                    len(keys),
                    {
                        index: connections.count(index)
                        for index in self.connection_indices
                    },
                )
            metadata = torch.tensor(
                [next(self._task_ids), *keys, *connections],
                dtype=torch.int64,
                device=self.device,
            )
            prefix_hits = self._ops.kvcache_exist(metadata, self._output)
            if not isinstance(prefix_hits, int) or not 0 <= prefix_hits <= len(keys):
                raise RuntimeError(
                    f"OceanDisk batch exist returned invalid prefix length "
                    f"{prefix_hits!r} for {len(keys)} keys."
                )
        if os.environ.get("SGLANG_OCEANDISK_CACHE_DEBUG") == "1" and prefix_hits < len(
            keys
        ):
            key = keys[prefix_hits]
            conn = connections[prefix_hits]
            put_record = self._debug_put_history.get(key)
            try:
                probe = torch.tensor(
                    [next(self._task_ids), key, conn],
                    dtype=torch.int64,
                    device=self.device,
                )
                single_hit = self._ops.kvcache_exist(probe, self._output)
            except Exception as error:
                single_hit = repr(error)
            logger.info(
                "[OceanDisk miss probe] pid=%d device=%s key=%d page_index=%d "
                "query_conn=%d query_endpoint=%s single_hit=%s "
                "put_record=%s history_limit=100000 "
                "put_record_fields=(state,conn,endpoint,task,layers)",
                os.getpid(),
                self.device,
                key,
                prefix_hits,
                conn,
                self.connection_router.endpoints[conn],
                single_hit,
                put_record,
            )
        return [True] * prefix_hits + [False] * (len(keys) - prefix_hits)

    def _debug_record_put(self, descriptors, state):
        if os.environ.get("SGLANG_OCEANDISK_CACHE_DEBUG") != "1":
            return
        for batch in descriptors:
            row = batch[0]
            count = (len(row) - 1) // 6
            for i in range(count // 2):
                key = row[1 + count + i]
                conn = row[1 + 5 * count + i]
                self._debug_put_history[key] = (
                    state,
                    conn,
                    self.connection_router.endpoints[conn],
                    row[0],
                    len(batch),
                )
                self._debug_put_history.move_to_end(key)
                if len(self._debug_put_history) > 100000:
                    self._debug_put_history.popitem(last=False)

    def _pack_layer_batches(self, layers, max_ranges):
        if type(max_ranges) is not int or not 2 <= max_ranges <= MAX_GIDS_RANGES:
            raise ValueError(f"OceanDisk max_ranges must be in [2, {MAX_GIDS_RANGES}].")
        if not layers:
            return []
        count = len(layers[0])
        if count % 2 or any(len(layer) != count for layer in layers):
            raise ValueError("OceanDisk layers require equal, even IO counts.")
        # Index registrations once per batch instead of scanning all layer MRs
        # for every KV/Indexer range. Preserve first-match behavior for overlaps.
        ordered = sorted(
            (base, length, mr) for mr, (base, length) in enumerate(self.regions)
        )
        starts = [base for base, _, _ in ordered]
        disjoint = all(a + n <= b for (a, n, _), (b, _, _) in zip(ordered, ordered[1:]))

        def resolve(ptr, size):
            if disjoint and size > 0:
                index = bisect_right(starts, ptr) - 1
                if index >= 0:
                    base, length, mr = ordered[index]
                    if ptr + size <= base + length:
                        return mr, ptr - base
            return resolve_mr(self.regions, ptr, size)

        half = count // 2
        # Fix one connection per page for the whole operation, including both
        # KV/Indexer halves and every layer.
        selected_connections = {
            item.key: self.connection_router.select(item.key)
            for item in layers[0][:half]
        }
        batch_pages = max_ranges // 2
        batches = [
            [
                pack_io(
                    next(self._task_ids),
                    ranges[start : min(start + batch_pages, half)]
                    + ranges[half + start : half + min(start + batch_pages, half)],
                    self.regions,
                    self.connection_indices,
                    resolve=resolve,
                    select_connection=selected_connections.__getitem__,
                )
                for ranges in layers
            ]
            for start in range(0, half, batch_pages)
        ]
        # Resolve original ranges against actual allocations before rounding lengths.
        # Legacy behavior deliberately includes bytes beyond each logical page.
        for batch in batches:
            for row in batch:
                count = (len(row) - 1) // 6
                for index in range(count):
                    position = 1 + 4 * count + index
                    row[position] = (row[position] + 511) & ~511
        return batches

    def get_layers(
        self, layers, *, max_ranges=256, on_layer_ready=None, timings=None
    ) -> None:
        """Publish each layer before calling the next layer's native GET.

        Native GIDS builds may block inside GET. Passing all layers to one
        dispatch would then delay every readiness notification until the last
        layer finishes. Prepare metadata first, but dispatch one layer at a
        time so model compute can start while subsequent layers are loaded.
        """
        import od_utils

        started = time.perf_counter()
        native_times = []
        event_wait_ms = 0.0
        first_layer_ms = None
        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._get_stream),
        ):
            self._ensure_open()
            descriptors = self._pack_layer_batches(layers, max_ranges)
            packed = time.perf_counter()
            cpu_tensor_ms = upload_ms = 0.0
            submitted = []
            trace_event(
                "GET_BEGIN",
                layers=len(layers),
                batches=len(descriptors),
                mode="layerwise_dispatch",
            )
            try:
                for batch_index, batch in enumerate(descriptors):
                    step = time.perf_counter()
                    cpu = torch.tensor(batch, dtype=torch.int64, device="cpu")
                    cpu_tensor_ms += (time.perf_counter() - step) * 1000
                    step = time.perf_counter()
                    gpu = cpu.to(self.device)
                    upload_ms += (time.perf_counter() - step) * 1000
                    events = [torch.cuda.Event() for _ in layers]
                    for event in events:
                        event.record(self._get_stream)
                    submitted.append((cpu, gpu, events))
                    trace_batch(
                        "GET_METADATA_READY",
                        batch,
                        batch_index,
                        len(descriptors),
                        od_utils,
                    )
                for layer in range(len(layers)):
                    for cpu, gpu, events in submitted:
                        # Slice both metadata tensors and use local layer 0:
                        # the native wrapper indexes GPU rows and CPU task IDs
                        # differently when start_layer is nonzero.
                        step = time.perf_counter()
                        self._run_io(
                            lambda: od_utils.batch_dispatch_get(
                                gpu[layer : layer + 1],
                                gpu.shape[1],
                                cpu[layer : layer + 1],
                                self._output,
                                [events[layer].cuda_event],
                                self._get_stream.stream_id,
                                self.device.index,
                                0,
                                1,
                                True,
                                False,
                            ),
                            self._get_stream,
                        )
                        native_times.append((time.perf_counter() - step) * 1000)
                    # A layer is usable only after every I/O batch containing
                    # its paired KV/Indexer fragments has completed.
                    step = time.perf_counter()
                    for _, _, events in submitted:
                        events[layer].synchronize()
                    event_wait_ms += (time.perf_counter() - step) * 1000
                    if layer in (0, len(layers) - 1):
                        trace_event("GET_LAYER_READY", layer=layer)
                    if on_layer_ready is not None:
                        on_layer_ready(layer)
                    if layer == 0:
                        first_layer_ms = (time.perf_counter() - started) * 1000
                if timings is not None:
                    timings.update(
                        pack_ms=(packed - started) * 1000,
                        cpu_tensor_ms=cpu_tensor_ms,
                        upload_ms=upload_ms,
                        calls=len(native_times),
                        native_total_ms=sum(native_times),
                        native_max_ms=max(native_times, default=0.0),
                        layer_event_wait_ms=event_wait_ms,
                        storage_first_layer_ms=first_layer_ms,
                        storage_total_ms=(time.perf_counter() - started) * 1000,
                    )
                trace_event(
                    "GET_DONE",
                    layers=len(layers),
                    meaning="layer_events_finished",
                )
            except BaseException:
                self._get_stream.synchronize()
                raise

    def put_layers(self, layers, *, max_ranges=256) -> None:
        import od_utils

        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            self._ensure_open()
            descriptors = self._pack_layer_batches(layers, max_ranges)
            self._debug_record_put(descriptors, "submitted")
            submitted = []
            trace_event(
                "PUT_BEGIN",
                layers=len(layers),
                batches=len(descriptors),
            )
            try:
                for batch_index, batch in enumerate(descriptors):
                    cpu = torch.tensor(batch, dtype=torch.int64, device="cpu")
                    gpu = cpu.to(self.device)
                    lengths = torch.tensor(
                        [len(row) for row in batch], dtype=torch.int64, device="cpu"
                    )
                    indices = torch.tensor(
                        list(range(len(batch))), dtype=torch.int64, device="cpu"
                    )
                    # Materialize CUDA handles; the native wrapper records
                    # each event again after the corresponding layer PUT.
                    events = [torch.cuda.Event() for _ in layers]
                    for event in events:
                        event.record(self._put_stream)
                    submitted.append((cpu, gpu, lengths, indices, events))
                    trace_batch(
                        "NATIVE_PUT_CALL",
                        batch,
                        batch_index,
                        len(descriptors),
                        od_utils,
                    )
                    result = self._run_io(
                        lambda: od_utils.batch_dispatch_put(
                            gpu,
                            lengths,
                            indices,
                            cpu,
                            self._output,
                            [event.cuda_event for event in events],
                            self._put_stream.stream_id,
                            self.device.index,
                            0,
                            len(layers),
                            True,
                            False,
                        ),
                        self._put_stream,
                    )
                    trace_event(
                        "NATIVE_PUT_RETURN",
                        batch=batch_index + 1,
                        result=result,
                        per_io_status_checked=False,
                    )
                for _, _, _, _, events in submitted:
                    if events:
                        events[-1].synchronize()
                self._debug_record_put(descriptors, "completed")
                trace_event(
                    "PUT_DONE",
                    layers=len(layers),
                    meaning="cuda_event_finished",
                    per_io_status_checked=False,
                )
            except BaseException:
                self._debug_record_put(descriptors, "failed")
                self._put_stream.synchronize()
                raise

    def prepare_layerwise_put(self, layers, *, max_ranges=256, timings=None):
        """Prepare descriptors on the PUT stream and retain through completion."""
        torch = self._torch
        with torch.cuda.device(self.device), torch.cuda.stream(self._put_stream):
            self._ensure_open()
            pack_start = time.perf_counter()
            batches = self._pack_layer_batches(layers, max_ranges)
            pack_ms = (time.perf_counter() - pack_start) * 1000
            cpu_ms = upload_ms = views_ms = 0.0
            prepared = [[] for _ in layers]
            for batch in batches:
                step = time.perf_counter()
                cpu = torch.tensor(batch, dtype=torch.int64, device="cpu")
                cpu_ms += (time.perf_counter() - step) * 1000
                step = time.perf_counter()
                gpu = cpu.to(self.device)
                upload_ms += (time.perf_counter() - step) * 1000
                step = time.perf_counter()
                lengths = torch.tensor([cpu.shape[1]], dtype=torch.int64, device="cpu")
                indices = torch.tensor([0], dtype=torch.int64, device="cpu")
                for layer in range(len(batch)):
                    # Single-row views retain the full allocations through the
                    # session's completion and keep native layer indices local.
                    prepared[layer].append(
                        (
                            cpu[layer : layer + 1],
                            gpu[layer : layer + 1],
                            lengths,
                            indices,
                        )
                    )
                views_ms += (time.perf_counter() - step) * 1000
            if timings is not None:
                timings.update(
                    pack=pack_ms, cpu_tensor=cpu_ms, upload=upload_ms, views=views_ms
                )
            return prepared

    def submit_put_layer(self, layer, prepared, ready_event):
        import od_utils

        torch = self._torch
        call_times = []
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            self._ensure_open()
            # Wait before queuing PUT; do not hold up GET/EXISTS for compute.
            ready_event.synchronize()
            for cpu, gpu, lengths, indices in prepared:
                call_start = time.perf_counter()
                self._run_io(
                    lambda: od_utils.batch_dispatch_put(
                        gpu,
                        lengths,
                        indices,
                        cpu,
                        self._output,
                        [],
                        self._put_stream.stream_id,
                        self.device.index,
                        0,
                        1,
                        False,
                        False,
                    ),
                    self._put_stream,
                )
                call_ms = (time.perf_counter() - call_start) * 1000
                call_times.append(call_ms)
            trace_event("PUT_LAYER_SUBMITTED", layer=layer, batches=len(prepared))
        return call_times

    def record_save_event(self):
        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            event = torch.cuda.Event()
            event.record(self._put_stream)
            return event

    @staticmethod
    def _unsupported_hicache() -> NotImplementedError:
        return NotImplementedError(
            "OceanDiskStorage currently supports UnifiedCacheLinker range I/O only; "
            "HiCache host-cache integration is not implemented."
        )

    # Abstract HiCache methods remain unsupported by the UC adapter.
    def get(self, key, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def exists(self, key):
        raise self._unsupported_hicache()

    # Override base no-ops to reject unsupported host-cache operations.
    def batch_get_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def register_mem_pool_host(self, mem_pool_host):
        raise self._unsupported_hicache()

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        raise self._unsupported_hicache()

    def clear(self):
        raise self._unsupported_hicache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OceanDisk storage is closed.")

    def close(self) -> None:
        if self._closed:
            return
        # Linker stops and joins its producers before closing Storage.
        self._io_queue.put(None)
        self._io_thread.join()
        with (
            self._put_io_lock,
            self._torch.cuda.device(self.device),
        ):
            self._get_stream.synchronize()
            self._put_stream.synchronize()
            self._closed = True
        # GIDS exposes no deregistration operation; retain registered buffers.

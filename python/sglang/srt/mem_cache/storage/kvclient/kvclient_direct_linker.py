"""Direct DSA KV/Indexer transfers through KvClient, without a host tier."""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import Future
from queue import Empty, Full, Queue

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.kvclient.kvclient_storage import (
    KvClientRange,
    KvClientStorage,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()
OFFLOAD = "OFFLOAD"


def make_cache_key(page_key: str) -> bytes:
    return hashlib.sha256(page_key.encode()).digest()[:8]


class LayerWiseLoadCounter:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.producer_index = self.consumer_index = -1
        self.futures = {}

    def update_producer(self):
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index):
        self.consumer_index = index

    def complete(self, index, layer):
        self.futures[index][layer].set_result(None)

    def fail(self, index, error):
        for future in self.futures[index]:
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold):
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        futures[threshold].result()
        if threshold == self.num_layers - 1:
            self.futures.pop(index)

    def reset(self):
        self.producer_index = self.consumer_index = -1
        self.futures.clear()


class KvClientDirectLinker(UnifiedCacheLinker):
    layerwise_save_enabled = False

    def __init__(self, server_args, params, *, components, storage=None):
        if (
            params.mtp_draft_device_pools
            or getattr(server_args, "speculative_algorithm", None)
            or get_disagg().disaggregation_linker_notify
        ):
            raise ValueError("KvClient does not support MTP draft or PD notifications")
        parallel = get_parallel()
        if parallel.enable_dsa_cache_layer_split:
            raise ValueError("KvClient requires replicated DSA cache layers")
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        if set(self.pools) != {PoolName.KV, PoolName.INDEXER} or self.num_layers < 1:
            raise ValueError("KvClient requires DSA KV and Indexer pools")
        self._build_remote_layout()
        buffers = [
            (name, buffer)
            for name in (PoolName.KV, PoolName.INDEXER)
            for buffer in self.pools[name].get_hybrid_pool_buffer()
        ]
        self.device = buffers[0][1].device
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        tp_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
        replicated = self.pool_group.rank_replicated
        shared_cp_cache = (
            replicated
            and params.attn_cp_size > 1
            and parallel.enable_prefill_cp
            and parallel.cp_strategy == "interleave"
        )
        self.offload_owner = (not replicated or tp_rank == 0) and (
            not shared_cp_cache or params.attn_cp_rank == 0
        )
        config_path = get_memory().kvclient_config_path
        if storage is None and not config_path:
            raise ValueError("KvClient requires --kvclient-config-path")
        self.storage = storage or KvClientStorage(config_path, self.device.index)
        self.storage.register_buffers(buffers)
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        self.pending_loads = {}
        self.load_queue = Queue(maxsize=64)
        self.offload_queue = Queue(maxsize=64)
        self._pending_offload_count = 0
        self.completed_loads = Queue()
        self.offload_results = Queue()
        self._fatal_error = None
        self._accepting = True
        self._closed = False
        self.load_thread = threading.Thread(
            target=self.load_thread_func, daemon=True, name=f"kvclient-load-{tp_rank}"
        )
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"kvclient-offload-{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread.start()

    def _build_remote_layout(self):
        self._storage_offsets = {}
        remote = 0
        for name in (PoolName.KV, PoolName.INDEXER):
            entry = self.pools[name]
            if not entry.packed:
                raise ValueError("KvClient requires packed DSA pools")
            offsets = {}
            raw = 0
            for component in entry.buffer_meta:
                for _, _, size in component:
                    offsets[raw] = remote
                    raw += size
                    # Align the next remote offset without changing the IO length.
                    remote += (size + 511) & ~511
            self._storage_offsets[name] = offsets
            for layer in range(self.num_layers):
                meta = entry.get_prepared_layer_range_meta([0], layer)
                if meta is None or not meta[1][0]:
                    raise ValueError("Every DSA layer needs both physical pools")
        if remote > 1 << 32:
            raise ValueError("KvClient page layout exceeds uint32 offset space")

    def _check_error(self):
        if self._fatal_error is not None:
            raise RuntimeError("KvClient direct linker failed") from self._fatal_error

    def _check_accepting(self):
        self._check_error()
        if not self._accepting:
            raise RuntimeError("KvClient direct linker is stopped")

    def _prepare(self, transfers):
        keys_by_page = {}
        cpu_indices = {}
        prepared = []
        for transfer in transfers:
            keys = []
            for page_key in transfer.keys:
                if page_key not in keys_by_page:
                    keys_by_page[page_key] = make_cache_key(page_key)
                keys.append(keys_by_page[page_key])
            indices = transfer.host_indices
            source = id(indices)
            if source not in cpu_indices:
                cpu_indices[source] = indices.detach().to(
                    device="cpu", dtype=torch.int64
                )
            locations = self.pools[transfer.name].prepare_locations(cpu_indices[source])
            prepared.append((transfer.name, keys, locations))
        return prepared

    def _layer_ranges(self, prepared, layer):
        ranges = []
        for name, keys, locations in prepared:
            ptrs, sizes, offsets = self.pools[name].get_prepared_layer_range_meta(
                locations, layer
            )
            for key, page_ptrs, page_sizes, page_offsets in zip(
                keys, ptrs, sizes, offsets, strict=True
            ):
                ranges.extend(
                    KvClientRange(
                        key,
                        ptr,
                        size,
                        self.device.index,
                        self._storage_offsets[name][offset],
                    )
                    for ptr, size, offset in zip(
                        page_ptrs, page_sizes, page_offsets, strict=True
                    )
                )
        return ranges

    def lookup(self, rid, transfers):
        self._check_accepting()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        keys = next(item.keys for item in expanded if item.name == PoolName.KV)
        hits = self.storage.exists_many([make_cache_key(key) for key in keys])
        prefix = []
        for index, hit in enumerate(hits, 1):
            if not hit:
                break
            prefix.append(index)
        return prefix

    def load(self, rid, transfers):
        self._check_accepting()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Load already queued for {rid}")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid):
        # The Wrapper has already committed tree state; no safe rollback exists.
        return False

    def start_layer_wise_loading(self):
        self._check_accepting()
        if not self.pending_loads:
            return -1
        if self.load_queue.full():
            raise RuntimeError("KvClient load queue is full; pending loads retained")
        ready_event = device_module.Event()
        ready_event.record()
        index = self.layer_done_counter.update_producer()
        self.load_queue.put_nowait((index, self.pending_loads, ready_event))
        self.pending_loads = {}
        return index

    def load_thread_func(self):
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                index, pending, ready_event = task
                try:
                    self._check_error()
                    device_module.set_device(self.device)
                    ready_event.synchronize()
                    self.load_layer_wise(index, list(pending.values()))
                    self.completed_loads.put(list(pending))
                except Exception as error:
                    self._fatal_error = error
                    self.layer_done_counter.fail(index, error)
                    logger.exception("KvClient load failed")
            finally:
                self.load_queue.task_done()

    def load_layer_wise(self, counter_index, request_transfers):
        prepared = self._prepare(
            [transfer for transfers in request_transfers for transfer in transfers]
        )
        tasks = []
        consumed = set()
        try:
            # Submit every layer before waiting so IO overlaps computation.
            for layer in range(self.num_layers):
                tasks.append(
                    self.storage.get_async(self._layer_ranges(prepared, layer))
                )
            for layer, task_id in enumerate(tasks):
                consumed.add(task_id)
                self.storage.wait(task_id)
                self.layer_done_counter.complete(counter_index, layer)
        except Exception:
            # Partial submission still owns device memory until all tasks drain.
            for task_id in tasks:
                if task_id not in consumed:
                    try:
                        self.storage.wait(task_id)
                    except Exception:
                        logger.exception("KvClient load drain failed")
            raise

    def offload(self, transfers):
        self._check_accepting()
        # Ranks retire the same FIFO prefix. Base admission on that shared
        # progress, since worker dequeue timing differs across ranks.
        if self._pending_offload_count >= self.offload_queue.maxsize:
            return False
        transfers = self.pool_group.resolve_transfers(transfers)
        if not transfers:
            return False
        ready_event = device_module.Event()
        ready_event.record()
        try:
            self.offload_queue.put_nowait((OFFLOAD, transfers, ready_event))
        except Full:
            return False
        self._pending_offload_count += 1
        return True

    def offload_thread_func(self):
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                kind, transfers, ready_event = task
                try:
                    if kind != OFFLOAD:
                        raise ValueError(f"Unsupported KvClient offload task: {kind}")
                    self._check_error()
                    device_module.set_device(self.device)
                    ready_event.synchronize()
                    if self.offload_owner:
                        prepared = self._prepare(transfers)
                        ranges = [
                            item
                            for layer in range(self.num_layers)
                            for item in self._layer_ranges(prepared, layer)
                        ]
                        self.storage.wait(self.storage.put_async(ranges))
                    self.offload_results.put(True)
                except Exception as error:
                    if self.storage.has_unsafe_io:
                        self._fatal_error = error
                    else:
                        self.offload_results.put(False)
                    logger.exception("KvClient offload failed")
            finally:
                self.offload_queue.task_done()

    def num_completed_loads(self):
        self._check_error()
        return self.completed_loads.qsize()

    def pop_completed_load(self):
        self._check_error()
        return self.completed_loads.get_nowait()

    def num_completed_offloads(self):
        self._check_error()
        return self.offload_results.qsize()

    def pop_completed_offload(self):
        self._check_error()
        result = self.offload_results.get_nowait()
        self._pending_offload_count -= 1
        return result

    def _drain(self):
        self._accepting = False
        self.load_queue.join()
        self.offload_queue.join()
        if not self.storage.io_quiescent:
            raise RuntimeError("KvClient IO is not quiescent; retain tree protection")

    def reset(self):
        if self._closed:
            raise RuntimeError("Cannot reset a closed KvClient direct linker")
        self._drain()
        self.pending_loads.clear()
        self._pending_offload_count = 0
        for queue in (self.completed_loads, self.offload_results):
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break
        self.layer_done_counter.reset()
        self._fatal_error = None
        self._accepting = True

    def close(self):
        if not self._closed:
            self._drain()
            self.load_queue.put(None)
            self.offload_queue.put(None)
            self.load_thread.join()
            self.offload_thread.join()
            self._closed = True
        self.storage.close()

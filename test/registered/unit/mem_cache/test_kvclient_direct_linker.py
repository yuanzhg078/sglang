"""CPU contract tests; no SDK, torch or device is needed for the fake transport."""

import hashlib
import importlib.util
import logging
import sys
import threading
import unittest
from pathlib import Path
from queue import Queue
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "python/sglang/srt/mem_cache/storage/kvclient"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BACKEND / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


storage_module = load_module("_test_kvclient_storage", "kvclient_storage.py")
KvClientStorage = storage_module.KvClientStorage
KvClientRange = storage_module.KvClientRange
KvClientIOError = storage_module.KvClientIOError


class Event:
    def record(self):
        pass

    def synchronize(self):
        pass


class SlotTensor:
    def __init__(self, slots):
        self.slots = slots
        self.copies = []

    def detach(self):
        return self

    def to(self, *, device, dtype):
        self.copies.append((device, dtype))
        return list(self.slots)


def load_linker():
    names = {
        "torch": {
            "distributed": SimpleNamespace(is_available=lambda: False),
            "int64": object(),
        },
        "sglang.srt.mem_cache.hicache_storage": {
            "PoolName": SimpleNamespace(KV="kv", INDEXER="indexer")
        },
        "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler": {
            "resolve_hybrid_device_pool_group": lambda **kwargs: None
        },
        "sglang.srt.mem_cache.storage.kvclient.kvclient_storage": vars(storage_module),
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker": {
            "UnifiedCacheLinker": object
        },
        "sglang.srt.runtime_context": {
            "get_disagg": lambda: SimpleNamespace(disaggregation_linker_notify=False),
            "get_memory": lambda: SimpleNamespace(kvclient_config_path="fake.conf"),
            "get_parallel": lambda: SimpleNamespace(
                enable_dsa_cache_layer_split=False,
                enable_prefill_cp=False,
                cp_strategy=None,
            ),
        },
        "sglang.srt.utils": {
            "get_device_module": lambda: SimpleNamespace(
                Event=Event, set_device=lambda device: None
            )
        },
    }
    modules = {}
    for name, attrs in names.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        return load_module("_test_kvclient_linker", "kvclient_direct_linker.py")


linker_module = load_linker()


def result(count, code=0, wait_code=0, **kwargs):
    return dict(
        wait_status={"code": wait_code},
        status={"code": code},
        entry_status=[{"code": 0}] * count,
        **kwargs,
    )


class FakeClient:
    def __init__(self):
        self.calls = []
        self.tasks = {}
        self.objects = {}
        self.memory = {}
        self.next_id = 0
        self.override = None
        self.fail_submit = None

    def register_regions(self, regions):
        self.calls.append(("register", regions))
        return [
            {"region": region, "handle": i + 7, "token_id": i + 70}
            for i, region in enumerate(regions)
        ]

    def submit(self, kind, payload):
        if self.fail_submit == (kind, self.next_id + 1):
            raise RuntimeError("submission failed")
        self.next_id += 1
        self.calls.append((kind, payload))
        self.tasks[self.next_id] = (kind, payload)
        return self.next_id

    def query_async(self, keys):
        return self.submit("query", keys)

    def load_async(self, entries):
        return self.submit("get", entries)

    def store_async(self, entries):
        return self.submit("put", entries)

    def wait(self, task, timeout):
        self.calls.append(("wait", task))
        kind, payload = self.tasks.pop(task)
        if self.override:
            return self.override(len(payload))
        if kind == "query":
            return result(len(payload), exists=[key in self.objects for key in payload])
        for key, addr, size, device, handle, offset in payload:
            if kind == "put":
                self.objects.setdefault(key, {})[offset] = self.memory[addr]
            else:
                self.memory[addr] = self.objects[key][offset]
        return result(len(payload))

    def unregister_regions(self, handles):
        self.calls.append(("unregister", handles))

    def shutdown(self):
        self.calls.append(("shutdown", None))


def buffer(addr, size=8192):
    return SimpleNamespace(
        device=SimpleNamespace(index=0),
        untyped_storage=lambda: SimpleNamespace(
            data_ptr=lambda: addr, nbytes=lambda: size
        ),
    )


class Entry:
    packed = True
    page_size = 1

    def __init__(self, base, size):
        self.buffer_meta = [[(base, 1024, size), (base + 4096, 1024, size)]]

    def prepare_locations(self, indices):
        return indices

    def get_prepared_layer_range_meta(self, locations, layer):
        base, stride, size = self.buffer_meta[0][layer]
        return (
            [[base + row * stride] for row in locations],
            [[size]] * len(locations),
            [[layer * size]] * len(locations),
        )


class TestKvClient(unittest.TestCase):
    def make_linker(self):
        linker = linker_module.KvClientDirectLinker.__new__(
            linker_module.KvClientDirectLinker
        )
        linker.pools = {"kv": Entry(512, 1024), "indexer": Entry(16896, 512)}
        linker.num_layers = 2
        linker.device = SimpleNamespace(index=0)
        linker._build_remote_layout()
        client = FakeClient()
        linker.storage = KvClientStorage(None, 0, client=client)
        linker.storage.register_buffers(
            [("kv", buffer(512)), ("indexer", buffer(16896))]
        )
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers: transfers
        )
        linker.pending_loads = {}
        linker.load_queue = Queue(maxsize=1)
        linker.offload_queue = Queue(maxsize=1)
        linker._pending_offload_count = 0
        linker.completed_loads = Queue()
        linker.offload_results = Queue()
        linker.layer_done_counter = linker_module.LayerWiseLoadCounter(2)
        linker.offload_owner = True
        linker._fatal_error = None
        linker._accepting = True
        linker._closed = False
        return linker, client

    def transfers(self, row=0):
        indices = SlotTensor([row])
        return [
            SimpleNamespace(name=name, keys=["page"], host_indices=indices)
            for name in ("kv", "indexer")
        ]

    def run_worker(self, linker, kind):
        queue = getattr(linker, kind + "_queue")
        target = getattr(linker, kind + "_thread_func")
        worker = threading.Thread(target=target)
        worker.start()
        queue.join()
        queue.put(None)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())

    def test_key_layout_roundtrip_and_layer_submission_order(self):
        linker, client = self.make_linker()
        self.assertEqual(
            linker_module.make_cache_key("page"), hashlib.sha256(b"page").digest()[:8]
        )
        self.assertEqual(
            linker._storage_offsets,
            {"kv": {0: 0, 1024: 1024}, "indexer": {0: 2048, 512: 2560}},
        )
        for layer in range(2):
            for item in linker._layer_ranges(linker._prepare(self.transfers()), layer):
                client.memory[item.addr] = (
                    bytes([item.remote_offset // 512]) * item.size
                )
        self.assertTrue(linker.offload(self.transfers()))
        self.run_worker(linker, "offload")
        self.assertTrue(linker.pop_completed_offload())
        self.assertEqual(linker.lookup("request", self.transfers()), [1])
        self.assertTrue(linker.load("request", self.transfers(1)))
        self.assertTrue(linker.load("request2", self.transfers(2)))
        index = linker.start_layer_wise_loading()
        client.calls.clear()
        self.run_worker(linker, "load")
        self.assertEqual(
            [call[0] for call in client.calls], ["get", "get", "wait", "wait"]
        )
        linker.layer_done_counter.set_consumer(index)
        for layer in range(2):
            linker.layer_done_counter.wait_until(layer)
            for item in linker._layer_ranges(linker._prepare(self.transfers(1)), layer):
                self.assertEqual(
                    client.memory[item.addr],
                    bytes([item.remote_offset // 512]) * item.size,
                )
        self.assertEqual(linker.pop_completed_load(), ["request", "request2"])
        linker.storage.close()

    def test_prepare_reuses_page_keys_across_pools_and_requests(self):
        linker, _ = self.make_linker()
        transfers = self.transfers(0) + self.transfers(1)
        indices = SlotTensor([1, 2])
        for transfer in transfers[2:]:
            transfer.keys = ["another-page", "page"]
            transfer.host_indices = indices
        with patch.object(
            linker_module, "make_cache_key", wraps=linker_module.make_cache_key
        ) as encode:
            prepared = linker._prepare(transfers)
        self.assertEqual(
            [call.args for call in encode.call_args_list],
            [("page",), ("another-page",)],
        )
        page, another = (
            linker_module.make_cache_key(key) for key in ("page", "another-page")
        )
        self.assertEqual(
            prepared,
            [
                ("kv", [page], [0]),
                ("indexer", [page], [0]),
                ("kv", [another, page], [1, 2]),
                ("indexer", [another, page], [1, 2]),
            ],
        )

    def test_prepare_copies_each_shared_slot_tensor_once(self):
        linker, _ = self.make_linker()
        first, second = self.transfers(), self.transfers()
        for transfers in (first, second):
            slots = SlotTensor(range(128, 192))
            for transfer in transfers:
                transfer.host_indices = slots
        with (
            patch.object(
                linker.pools["kv"],
                "prepare_locations",
                side_effect=lambda rows: rows[::64],
            ) as kv,
            patch.object(
                linker.pools["indexer"],
                "prepare_locations",
                side_effect=lambda rows: [row // 64 for row in rows[::64]],
            ) as indexer,
        ):
            prepared = linker._prepare(first + second)

        expected_copy = [("cpu", linker_module.torch.int64)]
        self.assertEqual(first[0].host_indices.copies, expected_copy)
        self.assertEqual(second[0].host_indices.copies, expected_copy)
        for kv_call, indexer_call in zip(
            kv.call_args_list, indexer.call_args_list, strict=True
        ):
            self.assertIs(kv_call.args[0], indexer_call.args[0])
        self.assertIsNot(kv.call_args_list[0].args[0], kv.call_args_list[1].args[0])
        self.assertEqual(
            [locations for _, _, locations in prepared], [[128], [2], [128], [2]]
        )

    def test_glm_page_roundtrip_preserves_unaligned_indexer_size(self):
        linker, client = self.make_linker()
        linker.num_layers = 78
        buffers = []
        for name, base, size in (("kv", 512, 73728), ("indexer", 1 << 25, 8448)):
            linker.pools[name].buffer_meta = [
                [
                    (base + layer * 2 * size, size, size)
                    for layer in range(linker.num_layers)
                ]
            ]
            buffers.append((name, buffer(base, linker.num_layers * 2 * size)))
        linker.storage = KvClientStorage(None, 0, client=client)
        linker.storage.register_buffers(buffers)
        linker._build_remote_layout()
        self.assertEqual(linker._storage_offsets["indexer"][0], 5750784)

        source = linker._layer_ranges(linker._prepare(self.transfers(0)), 1)
        target = linker._layer_ranges(linker._prepare(self.transfers(1)), 1)
        for item, value in zip(source, (1, 2), strict=True):
            client.memory[item.addr] = bytes([value]) * item.size
        linker.storage.wait(linker.storage.put_async(source))
        linker.storage.wait(linker.storage.get_async(target))

        for operation, entries in client.calls:
            if operation in ("put", "get"):
                kv, indexer = entries
                self.assertEqual(kv[0], indexer[0])
                self.assertEqual((kv[2], kv[5]), (73728, 73728))
                self.assertEqual((indexer[2], indexer[5]), (8448, 5759488))
        self.assertEqual(client.memory[target[1].addr], bytes([2]) * 8448)

    def test_partial_submission_drains_and_never_completes_failed_load(self):
        linker, client = self.make_linker()
        client.fail_submit = ("get", 2)
        client.override = lambda count: result(count)
        linker.load("failed", self.transfers())
        index = linker.start_layer_wise_loading()
        with self.assertLogs(linker_module.logger, logging.ERROR):
            self.run_worker(linker, "load")
        self.assertEqual(client.calls[-1], ("wait", 1))
        self.assertTrue(linker.storage.io_quiescent)
        self.assertTrue(linker.completed_loads.empty())
        linker.layer_done_counter.set_consumer(index)
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            linker.layer_done_counter.wait_until(0)
        with self.assertRaisesRegex(RuntimeError, "linker failed"):
            linker.num_completed_loads()

    def test_timeout_consumed_once_and_retains_registrations(self):
        linker, client = self.make_linker()
        client.override = lambda count: result(count, code=3, wait_code=3)
        task = linker.storage.get_async(
            linker._layer_ranges(linker._prepare(self.transfers()), 0)
        )
        with self.assertRaises(KvClientIOError):
            linker.storage.wait(task)
        with self.assertRaises(KeyError):
            linker.storage.wait(task)
        with self.assertRaisesRegex(RuntimeError, "retain tree protection"):
            linker.reset()
        with self.assertRaisesRegex(RuntimeError, "retain buffers"):
            linker.storage.close()
        self.assertFalse(any(call[0] == "unregister" for call in client.calls))
        self.assertTrue(linker.storage.buffer_sources)

    def test_terminal_put_failure_reports_false_and_can_close(self):
        linker, client = self.make_linker()
        client.override = lambda count: result(count, code=7, wait_code=7)
        linker.offload(self.transfers())
        with self.assertLogs(linker_module.logger, logging.ERROR):
            self.run_worker(linker, "offload")
        self.assertFalse(linker.pop_completed_offload())
        self.assertTrue(linker.storage.io_quiescent)
        linker.storage.close()

    def test_put_transport_timeout_keeps_protection(self):
        linker, client = self.make_linker()

        def partial_timeout(count):
            output = result(count, code=5, wait_code=5)  # PARTIAL_FAILED
            output["entry_status"][0] = {"code": 0x10716}  # CQE_IO_TIMEOUT
            return output

        client.override = partial_timeout
        self.assertTrue(linker.offload(self.transfers()))
        with self.assertLogs(linker_module.logger, logging.ERROR):
            self.run_worker(linker, "offload")
        self.assertTrue(linker.offload_results.empty())
        with self.assertRaisesRegex(RuntimeError, "linker failed"):
            linker.num_completed_offloads()
        with self.assertRaisesRegex(RuntimeError, "retain tree protection"):
            linker.reset()

    def test_registration_dedup_coverage_alignment_and_entry_errors(self):
        client = FakeClient()
        storage = KvClientStorage(None, 0, client=client)
        shared = buffer(512)
        storage.register_buffers([("kv", shared), ("indexer", shared)])
        self.assertEqual(len(client.calls[0][1]), 1)
        self.assertEqual(len(storage.buffer_sources[(512, 8192, 0)]), 2)
        for addr, size, offset in (
            (1, 512, 0),
            (8192, 1024, 0),
            (512, 512, 256),
            (512, 512, 1 << 32),
        ):
            with self.assertRaises(ValueError):
                storage.get_async([KvClientRange(b"12345678", addr, size, 0, offset)])

        def partial(count):
            output = result(count)
            output["entry_status"][0] = {"code": 7}
            return output

        client.override = partial
        task = storage.get_async([KvClientRange(b"12345678", 512, 512, 0, 0)])
        with self.assertRaises(KvClientIOError):
            storage.wait(task)
        self.assertTrue(storage.io_quiescent)

    def test_prefix_stops_at_first_miss_and_nonowner_does_not_put(self):
        linker, client = self.make_linker()
        client.override = lambda count: result(count, exists=[True, False, True])
        transfers = [SimpleNamespace(name="kv", keys=["a", "b", "c"])]
        self.assertEqual(linker.lookup("rid", transfers), [1])
        linker.offload_owner = False
        linker.offload(self.transfers())
        self.run_worker(linker, "offload")
        self.assertTrue(linker.pop_completed_offload())
        self.assertFalse(any(call[0] == "put" for call in client.calls))

    def test_query_timeout_is_a_miss_and_does_not_stop_new_query(self):
        linker, client = self.make_linker()
        transfers = [SimpleNamespace(name="kv", keys=["a", "b"])]
        client.override = lambda count: result(count, code=3, wait_code=3)
        self.assertEqual(linker.lookup("rid", transfers), [])
        self.assertTrue(linker.storage.io_quiescent)
        client.override = lambda count: result(count, exists=[1, 1])
        self.assertEqual(linker.lookup("rid", transfers), [1, 2])

    def test_full_queue_preserves_pending_and_rejects_offload(self):
        linker, _client = self.make_linker()
        linker.load_queue.put("busy")
        linker.load("request", self.transfers())
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            linker.start_layer_wise_loading()
        self.assertIn("request", linker.pending_loads)
        self.assertEqual(linker.layer_done_counter.producer_index, -1)
        linker.offload_queue.put("busy")
        self.assertFalse(linker.offload(self.transfers()))

    def test_offload_admission_stays_in_sync_with_different_worker_progress(self):
        slow, _ = self.make_linker()
        fast, _ = self.make_linker()
        slow.offload_owner = fast.offload_owner = False
        transfers = self.transfers()
        self.assertTrue(slow.offload(transfers))
        self.assertTrue(fast.offload(transfers))
        self.run_worker(fast, "offload")
        self.assertTrue(fast.offload_queue.empty())
        self.assertFalse(fast.offload(transfers))
        self.assertFalse(slow.offload(transfers))
        self.run_worker(slow, "offload")
        for linker in (slow, fast):
            self.assertTrue(linker.pop_completed_offload())
            self.assertTrue(linker.offload(transfers))

    def test_layer_counter_releases_first_layer_before_second(self):
        counter = linker_module.LayerWiseLoadCounter(2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.complete(index, 0)
        counter.wait_until(0)
        self.assertFalse(counter.futures[index][1].done())
        counter.fail(index, RuntimeError("later layer failed"))
        with self.assertRaisesRegex(RuntimeError, "later layer failed"):
            counter.wait_until(1)

    def test_load_timeout_fails_counter_without_completion(self):
        linker, client = self.make_linker()
        client.override = lambda count: result(count, code=3, wait_code=3)
        linker.load("request", self.transfers())
        index = linker.start_layer_wise_loading()
        with self.assertLogs(linker_module.logger, logging.ERROR):
            self.run_worker(linker, "load")
        linker.layer_done_counter.set_consumer(index)
        with self.assertRaises(KvClientIOError):
            linker.layer_done_counter.wait_until(0)
        self.assertTrue(linker.completed_loads.empty())
        self.assertEqual(
            [call[1] for call in client.calls if call[0] == "wait"], [1, 2]
        )
        self.assertFalse(linker.storage.io_quiescent)

    def test_blocking_wait_does_not_block_other_submissions(self):
        linker, client = self.make_linker()
        entered, release = threading.Event(), threading.Event()

        def wait(task, timeout):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test wait timed out")
            return result(2)

        client.wait = wait
        ranges = linker._layer_ranges(linker._prepare(self.transfers()), 0)
        task = linker.storage.get_async(ranges)
        worker = threading.Thread(target=linker.storage.wait, args=(task,))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(RuntimeError, "already being consumed"):
                linker.storage.wait(task)
            self.assertEqual(linker.storage.put_async(ranges), 2)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())

    def test_constructor_selects_put_owner_and_drains_workers(self):
        entries = {"kv": Entry(512, 1024), "indexer": Entry(16896, 512)}
        for entry in entries.values():
            base = entry.buffer_meta[0][0][0]
            entry.get_hybrid_pool_buffer = lambda base=base: [buffer(base)]
        group = SimpleNamespace(entry_map=entries, num_layers=2, rank_replicated=True)
        params = SimpleNamespace(
            mtp_draft_device_pools=(),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            page_size=1,
            attn_tp_cache_group=None,
            tp_cache_group=None,
        )
        cases = (
            (1, None, 0, 0, True),
            (2, "interleave", 0, 0, True),
            (2, "interleave", 0, 1, False),
            (2, "interleave", 1, 0, False),
            (2, "zigzag", 0, 1, True),
        )
        for cp_size, strategy, tp_rank, cp_rank, owner in cases:
            with self.subTest(strategy=strategy, tp_rank=tp_rank, cp_rank=cp_rank):
                params.attn_cp_size = cp_size
                params.attn_cp_rank = cp_rank
                parallel = SimpleNamespace(
                    enable_dsa_cache_layer_split=False,
                    enable_prefill_cp=cp_size > 1,
                    cp_strategy=strategy,
                )
                distributed = SimpleNamespace(
                    is_available=lambda: True,
                    is_initialized=lambda: True,
                    get_rank=lambda group, rank=tp_rank: rank,
                )
                client = FakeClient()
                storage = KvClientStorage(None, 0, client=client)
                with (
                    patch.object(linker_module.torch, "distributed", distributed),
                    patch.object(linker_module, "get_parallel", return_value=parallel),
                    patch.object(
                        linker_module,
                        "resolve_hybrid_device_pool_group",
                        return_value=group,
                    ),
                ):
                    linker = linker_module.KvClientDirectLinker(
                        SimpleNamespace(), params, components=set(), storage=storage
                    )
                self.assertEqual(linker.offload_owner, owner)
                self.assertFalse(linker.layerwise_save_enabled)
                linker.reset()
                linker.close()
                linker.close()
                self.assertFalse(linker.load_thread.is_alive())
                self.assertFalse(linker.offload_thread.is_alive())
                with self.assertRaisesRegex(RuntimeError, "Cannot reset a closed"):
                    linker.reset()
                self.assertEqual(
                    [call[0] for call in client.calls],
                    ["register", "unregister", "shutdown"],
                )


if __name__ == "__main__":
    unittest.main()

"""Standalone GIDS PUT/GET byte round trip with configurable layers, pages and
pool sizes.

Set OPEXTENSION and GIDS_DIR for the native installation. Run with Python
in an environment containing CUDA PyTorch and od_utils. This writes fresh
test keys to the configured GIDS namespace.

Single GPU:
    python oceandisk_dispatch_roundtrip.py --layers 78 --pages 100 ...

Multi-GPU pure-read comparison (one process per GPU, same keys, aligned GET):
    python oceandisk_dispatch_roundtrip.py --gpus 0,1,2,3 --layers 78 ...
Worker 0 performs PUT; every worker then reads the same objects at the same
wall-clock instant so per-layer deltas can be compared across cards.
"""

import argparse
import hashlib
import os
import secrets
import subprocess
import sys
import tempfile
import time
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from statistics import median

import torch


def derive_keys(run_token, pool, count):
    """Deterministic per-run keys shared by every worker process."""
    keys, seen = [], set()
    index = 0
    while len(keys) < count:
        digest = hashlib.sha256(f"{run_token}:{pool}:{index}".encode()).digest()
        key = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        index += 1
        if key == 0 or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


CHUNK_BYTES = 64 << 20


def expected_chunk(seed, pool, chunk_index, count):
    """Deterministic payload bytes; writer fill and verification share this."""
    generator = torch.Generator().manual_seed(
        (seed * 1_000_003 + pool * 97_409 + chunk_index) & 0x7FFFFFFF
    )
    return torch.randint(0, 256, (count,), generator=generator, dtype=torch.uint8)


def fill_source(buffer, seed, pool, payload):
    offset = chunk_index = 0
    while offset < payload:
        count = min(CHUNK_BYTES, payload - offset)
        buffer[offset : offset + count].copy_(
            expected_chunk(seed, pool, chunk_index, count)
        )
        offset += count
        chunk_index += 1


def verify_region(region, seed, pool, payload, size, pages):
    offset = chunk_index = 0
    while offset < payload:
        count = min(CHUNK_BYTES, payload - offset)
        expected = expected_chunk(seed, pool, chunk_index, count)
        actual = region[offset : offset + count].cpu()
        mismatch = (actual != expected).nonzero(as_tuple=True)[0]
        if mismatch.numel():
            position = int(mismatch[0])
            absolute = offset + position
            layer, page = divmod(absolute // size, pages)
            raise RuntimeError(
                f"pool={pool} layer={layer} page={page} "
                f"byte={absolute % size}: expected={int(expected[position])} "
                f"actual={int(actual[position])}"
            )
        offset += count
        chunk_index += 1


def wait_for_file(path, timeout, processes=None):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {path}")
        for process in processes or ():
            if process.poll() is not None and process.returncode != 0:
                raise RuntimeError(
                    f"worker exited early with code {process.returncode}"
                )
        time.sleep(0.001)


def touch_file(path):
    Path(path).write_text(str(os.getpid()), encoding="utf-8")


def wait_for_go(sync_dir, worker_index, stagger_ms):
    """Wait until base + worker_index * stagger, using one shared base."""
    path = os.path.join(sync_dir, "go")
    wait_for_file(path, timeout=600)
    base = int(Path(path).read_text(encoding="utf-8").strip())
    target = base + int(round(worker_index * stagger_ms * 1e6))
    while True:
        remaining = (target - time.time_ns()) / 1e9
        if remaining <= 0:
            break
        if remaining > 0.002:
            time.sleep(remaining - 0.002)
    return base, target, time.time_ns()


def gids_config_path(config_dir, device_index, use_single_port):
    single = use_single_port is None or use_single_port == "1"
    name = "gids_config.txt" if single else f"gids_config_{device_index}.txt"
    return Path(config_dir) / name


def load_connection_selector(config_path, selected=None):
    """Mirror od/nds_utils.py: hash keys onto DPU vnodes, then round-robin
    across that DPU's connections. A key must go to the DPU that owns it."""
    try:
        from xxhash3_py import xxh3_64_hexdigest
    except ImportError:
        from xxhash import xxh3_64_hexdigest

    groups = {}
    conn_to_dpu = {}
    available = 0
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("ip_config"):
            continue
        fields = [part.strip() for part in line.split("=", 1)[1].split(",")]
        dpu = int(fields[5])
        conn_to_dpu[available] = dpu
        if selected is None or available in selected:
            groups.setdefault(dpu, []).append(available)
        available += 1
    if not groups:
        raise ValueError(f"No usable ip_config entries in {config_path}.")
    if selected is not None and any(index >= available for index in selected):
        raise ValueError(f"connection indices {selected} exceed {available} entries.")
    ring = sorted(
        (int(xxh3_64_hexdigest(f"DPU-{dpu}-V-{vnode}".encode("utf-8")), 16), dpu)
        for dpu in groups
        for vnode in range(50)
    )
    coordinates = [coordinate for coordinate, _ in ring]
    cursors = [0] * len(ring)

    def select(key):
        slot = bisect_left(coordinates, key % (1 << 64)) % len(ring)
        connections = groups[ring[slot][1]]
        connection = connections[cursors[slot] % len(connections)]
        cursors[slot] += 1
        return connection

    return select, available, sorted(groups), conn_to_dpu


def report_routing(label, rows):
    """Read the metadata tensor native consumes and print its conn column."""
    count = (rows.shape[1] - 1) // 6
    keys = rows[:, 1 + count : 1 + 2 * count]
    sizes = rows[:, 1 + 4 * count : 1 + 5 * count]
    conns = rows[:, 1 + 5 * count : 1 + 6 * count]
    per_layer = [tuple(row) for row in conns.tolist()]
    if len(set(per_layer)) != 1:
        raise RuntimeError(f"{label}: connection column differs across layers")
    per_conn_bytes = Counter()
    for _, conn, size in zip(
        keys[0].tolist(), conns[0].tolist(), sizes[0].tolist()
    ):
        per_conn_bytes[conn] += size
    pairs = list(zip(keys[0].tolist(), conns[0].tolist()))
    print(
        f"{label}: io_per_layer={count} "
        f"conns_per_layer={dict(sorted(Counter(conns[0].tolist()).items()))} "
        f"bytes_per_conn={dict(sorted(per_conn_bytes.items()))} "
        f"max_conn={max(conns[0].tolist(), default=-1)}"
    )
    print(
        f"{label}: key->conn sample="
        f"{pairs[:8]}{' ...' if len(pairs) > 8 else ''}"
    )


def dispatch(label, submit, events, stream, per_layer=False, quiet=False, start_wait=None):
    for event in events:
        event.record(stream)
    sync = start_wait() if start_wait is not None else None
    started = time.perf_counter()
    submit_ns = time.time_ns()
    submit([event.cuda_event for event in events])
    returned = time.perf_counter()
    marks = []
    for event in events:
        event.synchronize()
        marks.append((time.perf_counter() - returned) * 1000)
    finished = time.perf_counter()
    finished_ns = time.time_ns()
    metrics = {
        "call_ms": (returned - started) * 1000,
        "wait_ms": (finished - returned) * 1000,
        "total_ms": (finished - started) * 1000,
        "marks": marks,
        "submit_ns": submit_ns,
        "finished_ns": finished_ns,
    }
    if not quiet:
        print(
            f"{label}: call_ms={metrics['call_ms']:.3f} "
            f"event_wait_ms={metrics['wait_ms']:.3f} "
            f"total_ms={metrics['total_ms']:.3f} "
            f"submit_ns={submit_ns} finished_ns={finished_ns}"
        )
        if per_layer:
            # Events are recorded by native after each layer's I/O on the same
            # stream, so they complete in layer order and each sync is cumulative.
            previous = 0.0
            for layer, mark in enumerate(marks):
                print(
                    f"{label} layer={layer}: ready_ms={mark:.3f} "
                    f"delta_ms={mark - previous:.3f}"
                )
                previous = mark
        if sync is not None:
            base_ns, target_ns, arrival_ns = sync
            print(
                f"{label} sync: base_ns={base_ns} target_ns={target_ns} "
                f"arrival_skew_ms={(arrival_ns - target_ns) / 1e6:.3f} "
                f"submit_skew_ms={(submit_ns - target_ns) / 1e6:.3f}"
            )
    return metrics


def dispatch_split(
    label, submit_span, events, stream, layers, quiet=False, start_wait=None
):
    """Layer 0 alone, then the rest, then wait. Mirrors get_layers()."""
    for event in events:
        event.record(stream)
    sync = start_wait() if start_wait is not None else None
    started = time.perf_counter()
    first_submit_ns = time.time_ns()
    submit_span(0, 1, [events[0].cuda_event])
    first_returned = time.perf_counter()
    if layers > 1:
        submit_span(1, layers, [event.cuda_event for event in events[1:]])
    rest_returned = time.perf_counter()
    # Query happens after the second submit returns, so True only means
    # "already complete when queried", not "complete before that return".
    layer0_done_at_query = events[0].query()
    marks = []
    for event in events:
        event.synchronize()
        marks.append((time.perf_counter() - rest_returned) * 1000)
    finished = time.perf_counter()
    finished_ns = time.time_ns()
    metrics = {
        "layer0_call_ms": (first_returned - started) * 1000,
        "rest_call_ms": (rest_returned - first_returned) * 1000,
        "wait_ms": (finished - rest_returned) * 1000,
        "total_ms": (finished - started) * 1000,
        "marks": marks,
        "layer0_done_at_query": layer0_done_at_query,
        "submit_ns": first_submit_ns,
        "finished_ns": finished_ns,
    }
    if not quiet:
        print(
            f"{label}: layer0_call_ms={metrics['layer0_call_ms']:.3f} "
            f"rest_call_ms={metrics['rest_call_ms']:.3f} "
            f"event_wait_ms={metrics['wait_ms']:.3f} "
            f"total_ms={metrics['total_ms']:.3f} "
            f"submit_ns={first_submit_ns} finished_ns={finished_ns} "
            f"layer0_done_at_query={layer0_done_at_query}"
        )
        previous = 0.0
        for layer, mark in enumerate(marks):
            print(
                f"{label} layer={layer}: ready_ms={mark:.3f} "
                f"delta_ms={mark - previous:.3f}"
            )
            previous = mark
        if sync is not None:
            base_ns, target_ns, arrival_ns = sync
            print(
                f"{label} sync: base_ns={base_ns} target_ns={target_ns} "
                f"arrival_skew_ms={(arrival_ns - target_ns) / 1e6:.3f} "
                f"submit_skew_ms={(first_submit_ns - target_ns) / 1e6:.3f}"
            )
    return metrics


def report_get(runs, split, layers):
    """Print medians across repeated GET measurements."""
    totals = sorted(run["total_ms"] for run in runs)
    waits = median(run["wait_ms"] for run in runs)
    print(
        f"GET: runs={len(runs)} total_ms median={median(totals):.3f} "
        f"min={totals[0]:.3f} max={totals[-1]:.3f} "
        f"event_wait_ms median={waits:.3f}"
    )
    if split:
        print(
            f"GET: layer0_call_ms median="
            f"{median(run['layer0_call_ms'] for run in runs):.3f} "
            f"rest_call_ms median="
            f"{median(run['rest_call_ms'] for run in runs):.3f} "
            f"layer0_done_at_query={runs[-1]['layer0_done_at_query']}"
        )
    medians = [median(run["marks"][layer] for run in runs) for layer in range(layers)]
    previous = 0.0
    for layer, mark in enumerate(medians):
        print(
            f"GET layer={layer}: ready_ms={mark:.3f} "
            f"delta_ms={mark - previous:.3f}"
        )
        previous = mark


def strip_option(argv, names):
    result, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in names:
            skip = True
            continue
        if any(token.startswith(name + "=") for name in names):
            continue
        result.append(token)
    return result


def launch_multi_gpu(args):
    """One process per GPU. Worker 0 PUTs, all workers then GET in lockstep."""
    gpus = [int(part) for part in args.gpus.split(",") if part.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise SystemExit("--gpus must list distinct GPU indices, e.g. 0,1,2,3")
    if args.repeat != 1:
        raise SystemExit("multi-GPU mode runs one aligned GET; keep --repeat 1")
    sync_dir = args.sync_dir or tempfile.mkdtemp(prefix="oceandisk-multigpu-")
    os.makedirs(sync_dir, exist_ok=True)
    for name in os.listdir(sync_dir):
        if name in (
            "go", "go.tmp", "put_done", "exists_done", "writer_exists_done"
        ) or name.startswith("ready_"):
            os.remove(os.path.join(sync_dir, name))
    run_token = args.run_token or secrets.token_hex(8)
    base = strip_option(
        sys.argv[1:],
        {
            "--gpus",
            "--device",
            "--sync-dir",
            "--run-token",
            "--worker-index",
            "--workers",
        },
    )
    print(f"multi-gpu: gpus={gpus} sync_dir={sync_dir} run_token={run_token}")

    logs, processes = [], []
    exit_code, failure = 0, None
    try:
        for index, gpu in enumerate(gpus):
            log_path = os.path.join(sync_dir, f"gpu_{gpu}.log")
            handle = open(log_path, "w", encoding="utf-8")
            command = [
                sys.executable,
                "-u",
                os.path.abspath(__file__),
                *base,
                "--device",
                str(gpu),
                "--worker-index",
                str(index),
                "--workers",
                str(len(gpus)),
                "--sync-dir",
                sync_dir,
                "--run-token",
                run_token,
            ]
            processes.append(
                subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            )
            logs.append((gpu, log_path, handle))
        wait_for_file(os.path.join(sync_dir, "put_done"), 900, processes)
        for index in range(len(gpus)):
            wait_for_file(os.path.join(sync_dir, f"ready_{index}"), 900, processes)
        if not args.skip_get:
            target_ns = time.time_ns() + 200_000_000
            # Publish atomically: a plain open+write exposes an empty file first.
            go_path = os.path.join(sync_dir, "go")
            temp_path = go_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(str(target_ns))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, go_path)
            print(f"multi-gpu: all ready, GET base_ns={target_ns} (+200ms)")
        for process in processes:
            code = process.wait()
            if code:
                exit_code = code
    except BaseException as error:
        failure = error
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for _, _, handle in logs:
            handle.close()

    for gpu, log_path, _ in logs:
        print(f"===== gpu {gpu} ({log_path}) =====")
        print(Path(log_path).read_text(encoding="utf-8"))
    if failure is not None:
        print(f"multi-gpu failed: {failure!r}")
        return 1
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--conn",
        type=int,
        default=None,
        help="pin every I/O to one connection; default routes per key",
    )
    parser.add_argument(
        "--conns",
        type=str,
        default=None,
        help="comma-separated connection subset to route over; default all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--kv-bytes", type=int, default=4096)
    parser.add_argument("--index-bytes", type=int, default=4096)
    parser.add_argument("--separate-pool-keys", action="store_true")
    parser.add_argument(
        "--exists-device", type=int, help="GPU index that runs EXISTS after PUT"
    )
    parser.add_argument(
        "--exists-on-writer", action="store_true", help="also run EXISTS on the PUT GPU"
    )
    parser.add_argument("--skip-get", action="store_true", help="run PUT and EXISTS only")
    parser.add_argument(
        "--split-first-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "GET submits layer 0 alone, then the rest, like get_layers() "
            "(default: on; disable with --no-split-first-layer)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discarded GET iterations before measuring (default: 0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="measured GET iterations, reported as medians (default: 1)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="launch one process per GPU (e.g. 0,1,2,3); worker 0 PUTs, all GET",
    )
    parser.add_argument(
        "--get-stagger-ms",
        type=float,
        default=0.0,
        help=(
            "delay worker i's GET by i * N ms from one shared base "
            "(0 = all start together)"
        ),
    )
    parser.add_argument("--sync-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-token", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    layers, pages = args.layers, args.pages
    logical_sizes = (args.kv_bytes, args.index_bytes)
    if min(layers, pages, *logical_sizes) <= 0:
        parser.error("layers, pages and pool sizes must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    if args.skip_get and args.exists_device is None:
        parser.error("--skip-get requires --exists-device")
    if args.exists_on_writer and args.exists_device is None:
        parser.error("--exists-on-writer requires --exists-device")
    if args.gpus and args.worker_index is None:
        if args.exists_device is not None and args.exists_device not in [
            int(part) for part in args.gpus.split(",") if part.strip()
        ]:
            parser.error("--exists-device must be included in --gpus")
        return launch_multi_gpu(args)
    is_worker = args.worker_index is not None
    if not is_worker and args.exists_device not in (None, args.device):
        parser.error("--exists-device must match --device without --gpus")
    is_writer = not is_worker or args.worker_index == 0
    if is_worker and not args.sync_dir:
        parser.error("--sync-dir is set by the --gpus launcher")
    if args.get_stagger_ms < 0:
        parser.error("--get-stagger-ms must be >= 0")
    if not is_worker and args.get_stagger_ms:
        parser.error("--get-stagger-ms requires --gpus")
    sizes = tuple((size + 511) // 512 * 512 for size in logical_sizes)
    payloads = tuple(layers * pages * size for size in sizes)
    for name in ("OPEXTENSION", "GIDS_DIR"):
        if not os.environ.get(name):
            parser.error(f"set {name} for your GIDS installation")
    os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")
    if not os.path.isabs(os.environ["GIDS_DIR"]):
        parser.error(
            f"GIDS_DIR must be an absolute path, got {os.environ['GIDS_DIR']!r}"
        )

    config_path = gids_config_path(
        os.environ["GIDS_DIR"], args.device, os.environ.get("USE_SINGLE_PORT")
    )
    single_port_env = os.environ.get("USE_SINGLE_PORT")
    if not config_path.is_file():
        parser.error(
            f"GIDS config not found: {config_path} "
            f"(USE_SINGLE_PORT={single_port_env!r}; unset or '1' uses "
            f"gids_config.txt, otherwise gids_config_<device>.txt)"
        )
    selected = (
        [int(part) for part in args.conns.split(",") if part.strip()]
        if args.conns
        else None
    )
    if selected is not None and (not selected or len(set(selected)) != len(selected)):
        parser.error("--conns must list distinct connection indices")
    select_conn, available, dpu_ids, conn_to_dpu = load_connection_selector(
        config_path, selected
    )
    if args.conn is not None and not 0 <= args.conn < available:
        parser.error(f"--conn must be in [0, {available}) from {config_path}")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.ops.load_library(os.environ["OPEXTENSION"])
    import od_utils

    pure_reader = is_worker and not is_writer
    regions = 1 if pure_reader or args.skip_get else 2
    buffers = [
        torch.full(
            (512 if pure_reader and args.skip_get else regions * payload + 512,),
            0xA5,
            device=device,
            dtype=torch.uint8,
        )
        for payload in payloads
    ]
    if not pure_reader:
        for pool, buffer in enumerate(buffers):
            fill_source(buffer, args.seed, pool, payloads[pool])

    registration = torch.tensor(
        [b.data_ptr() for b in buffers] + [b.numel() for b in buffers],
        dtype=torch.int64,
        device=device,
    )
    output = torch.zeros(2, dtype=torch.int64, device=device)
    if is_worker:
        # Match the production linker: stagger GIDS connection setup by GPU.
        time.sleep(args.device * 2)
    status = torch.ops.my_ops.kvcache_init(registration, output)
    torch.cuda.synchronize(device)
    if status != 0:
        raise RuntimeError(f"kvcache_init failed: {status}")

    # Production keys are hashed page identities, uniformly spread over the
    # DPU ring. Every worker in multi-GPU mode derives the same key set from
    # one run token, so all of them read the same objects.
    run_token = args.run_token or secrets.token_hex(8)
    kv_keys = derive_keys(run_token, "kv", pages)
    index_keys = (
        derive_keys(run_token, "index", pages) if args.separate_pool_keys else kv_keys
    )
    keys = kv_keys + index_keys
    if args.conn is not None:
        conns = [args.conn] * len(keys)
    elif args.separate_pool_keys:
        conns = [select_conn(key) for key in keys]
    else:
        # Production fixes one connection per page for both KV/Indexer halves.
        conns = [select_conn(key) for key in keys[:pages]] * 2
    print(
        f"device={args.device} worker={args.worker_index}/{args.workers} "
        f"writer={is_writer} pure_reader={pure_reader} "
        f"layers={layers} pages={pages} "
        f"logical_pool_bytes={logical_sizes} aligned_pool_bytes={sizes} "
        f"bytes_per_direction={sum(payloads)} chunk_bytes={CHUNK_BYTES} "
        f"run_token={run_token} "
        f"kv_first_key={kv_keys[0]} index_first_key={index_keys[0]} "
        f"seed={args.seed} config={config_path} connections={available} "
        f"dpus={dpu_ids} routed={dict(sorted(Counter(conns).items()))}"
    )
    stream = torch.cuda.Stream(device=device)

    def metadata(destination, first_task):
        rows = []
        # Pure readers register only the destination region, so their GET
        # offsets start at 0; the writer keeps source at 0 and destination at
        # payload.
        for layer in range(layers):
            local_offsets = [
                (0 if not destination or pure_reader else payloads[pool])
                + (layer * pages + page) * size
                for pool, size in enumerate(sizes)
                for page in range(pages)
            ]
            rows.append(
                [first_task + layer]
                + [0] * pages + [1] * pages
                + keys
                + local_offsets
                + [layer * size for size in sizes for _ in range(pages)]
                + [size for size in sizes for _ in range(pages)]
                + conns
            )
        return torch.tensor(rows, dtype=torch.int64)

    with torch.cuda.stream(stream):
        if not args.skip_get:
            get_cpu = metadata(True, 1 + layers)
            report_routing("GET", get_cpu)
            get_gpu = get_cpu.to(device)
            width = get_cpu.shape[1]

        def submit_span(first, last, event_handles):
            # Local slices plus start_layer=0: native indexes CPU task IDs and
            # GPU rows locally, so a nonzero start layer would misalign them.
            od_utils.batch_dispatch_get(
                get_gpu[first:last],
                width,
                get_cpu[first:last],
                output,
                event_handles,
                stream.stream_id,
                args.device,
                0,
                last - first,
                True,
                False,
            )

        if is_writer:
            put_cpu = metadata(False, 1)
            report_routing("PUT", put_cpu)
            put_gpu = put_cpu.to(device)
            lengths = torch.full((layers,), put_cpu.shape[1], dtype=torch.int64)
            indices = torch.arange(layers, dtype=torch.int64)
            tail = (stream.stream_id, args.device, 0, layers, True, False)
            dispatch(
                "PUT",
                lambda events: od_utils.batch_dispatch_put(
                    put_gpu, lengths, indices, put_cpu, output, events, *tail
                ),
                [torch.cuda.Event() for _ in range(layers)],
                stream,
            )
        elif is_worker:
            print("PUT: skipped (pure reader; worker 0 is the writer)")
            if is_worker:
                torch.cuda.synchronize(device)
            time.sleep(10)
        if is_worker:
            touch_tag = os.path.join(args.sync_dir, "put_done")
            if is_writer:
                touch_file(touch_tag)
            else:
                wait_for_file(touch_tag, timeout=900)

        if args.exists_device is not None:
            is_query_device = args.device == args.exists_device
            if is_query_device or (args.exists_on_writer and is_writer):
                pools = (("kv", kv_keys, conns[:pages]),)
                if args.separate_pool_keys:
                    pools += (("index", index_keys, conns[pages:]),)
                task = 1 + 2 * layers

                def query_exists(query_keys, query_conns):
                    nonlocal task
                    query = torch.tensor(
                        [task, *query_keys, *query_conns],
                        dtype=torch.int64,
                        device=device,
                    )
                    task += 1
                    result = torch.ops.my_ops.kvcache_exist(query, output)
                    if not isinstance(result, int) or not 0 <= result <= len(query_keys):
                        raise RuntimeError(
                            f"kvcache_exist returned invalid prefix: {result!r}"
                        )
                    return result

                for pool, page_keys, page_conns in pools:
                    hits = query_exists(page_keys, page_conns)
                    message = (
                        f"EXISTS: device={args.device} pool={pool} "
                        f"hit_pages={hits}/{pages}"
                    )
                    if hits < pages:
                        last_key = page_keys[hits - 1] if hits else None
                        last_dpu = conn_to_dpu[page_conns[hits - 1]] if hits else None
                        message += (
                            f" first_miss_page={hits + 1} first_miss_key={page_keys[hits]}"
                            f" first_miss_dpu={conn_to_dpu[page_conns[hits]]}"
                            f" last_hit_key={last_key} last_hit_dpu={last_dpu}"
                        )
                    print(message, flush=True)
                    if hits < pages:
                        miss_key, miss_conn = page_keys[hits], page_conns[hits]
                        single_hit = query_exists([miss_key], [miss_conn])
                        retry_hits = query_exists(page_keys, page_conns)
                        print(
                            f"EXISTS RECHECK: device={args.device} pool={pool} "
                            f"key={miss_key} conn={miss_conn} "
                            f"dpu={conn_to_dpu[miss_conn]} single_hit={single_hit} "
                            f"retry_hit_pages={retry_hits}/{pages}",
                            flush=True,
                        )
                if is_worker and is_query_device:
                    touch_file(os.path.join(args.sync_dir, "exists_done"))
                if is_worker and args.exists_on_writer and is_writer:
                    touch_file(os.path.join(args.sync_dir, "writer_exists_done"))
            if is_worker:
                wait_for_file(os.path.join(args.sync_dir, "exists_done"), timeout=900)
                if args.exists_on_writer:
                    wait_for_file(
                        os.path.join(args.sync_dir, "writer_exists_done"), timeout=900
                    )

        if args.skip_get:
            if is_worker:
                touch_file(os.path.join(args.sync_dir, f"ready_{args.worker_index}"))
            return

        def run_get(quiet, start_wait=None):
            events = [torch.cuda.Event() for _ in range(layers)]
            if args.split_first_layer:
                return dispatch_split(
                    "GET",
                    submit_span,
                    events,
                    stream,
                    layers,
                    quiet=quiet,
                    start_wait=start_wait,
                )
            return dispatch(
                "GET",
                lambda event_handles: submit_span(0, layers, event_handles),
                events,
                stream,
                per_layer=True,
                quiet=quiet,
                start_wait=start_wait,
            )

        if is_worker:
            # Read-only workers join at the shared start point; worker 0 already
            # published put_done, so the data exists for every warmup read.
            for _ in range(args.warmup):
                run_get(quiet=True)
            touch_file(os.path.join(args.sync_dir, f"ready_{args.worker_index}"))
            run_get(
                quiet=False,
                start_wait=lambda: wait_for_go(
                    args.sync_dir, args.worker_index, args.get_stagger_ms
                ),
            )
        else:
            for _ in range(args.warmup):
                run_get(quiet=True)
            if args.repeat == 1:
                run_get(quiet=False)
            else:
                runs = [run_get(quiet=True) for _ in range(args.repeat)]
                report_get(runs, args.split_first_layer, layers)

    for pool, (buffer, size, payload) in enumerate(zip(buffers, sizes, payloads)):
        prefix = 0 if pure_reader else payload
        verify_region(
            buffer[prefix : prefix + payload], args.seed, pool, payload, size, pages
        )
    print(f"PASS: verified {sum(payloads)} bytes")


if __name__ == "__main__":
    sys.exit(main() or 0)

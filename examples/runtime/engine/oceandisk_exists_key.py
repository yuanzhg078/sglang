"""Query one key through the native GIDS EXISTS interface."""

import argparse
import os
import time

import torch

from oceandisk_dispatch_roundtrip import gids_config_path, load_connection_selector


# python examples/runtime/engine/oceandisk_exists_key.py \
#   --device 0 --dpu 11 --key 9129832293312717817

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=int, required=True)
    parser.add_argument("--dpu", type=int, required=True, help="DPU ID from ip_config")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--conn", type=int, help="connection index on the selected DPU")
    args = parser.parse_args()
    if not 0 < args.key < 2**63:
        parser.error("--key must be a positive int64")
    extension = os.environ.get("OPEXTENSION")
    config_dir = os.environ.get("GIDS_DIR")
    if not extension or not config_dir:
        parser.error("set OPEXTENSION and GIDS_DIR for your GIDS installation")
    if not os.path.isabs(config_dir):
        parser.error("GIDS_DIR must be an absolute path")
    os.environ["GIDS_DIR"] = os.path.join(config_dir, "")
    config = gids_config_path(
        config_dir, args.device, os.environ.get("USE_SINGLE_PORT")
    )
    if not config.is_file():
        parser.error(f"GIDS config not found: {config}")
    _, _, _, conn_to_dpu = load_connection_selector(config)
    candidates = [conn for conn, dpu in conn_to_dpu.items() if dpu == args.dpu]
    if not candidates:
        parser.error(f"DPU {args.dpu} has no connection in {config}")
    conn = candidates[0] if args.conn is None else args.conn
    if conn not in candidates:
        parser.error(f"--conn {conn} does not belong to DPU {args.dpu} in {config}")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.ops.load_library(extension)
    ops = torch.ops.my_ops
    # Keep both registered buffers alive until the query returns.
    buffers = [torch.empty(4096, dtype=torch.uint8, device=device) for _ in range(2)]
    registration = torch.tensor(
        [buf.data_ptr() for buf in buffers] + [buf.numel() for buf in buffers],
        dtype=torch.int64,
        device=device,
    )
    output = torch.zeros(2, dtype=torch.int64, device=device)
    status = ops.kvcache_init(registration, output)
    torch.cuda.synchronize(device)
    if status != 0:
        raise RuntimeError(f"kvcache_init failed: {status}")

    query = torch.tensor([1, args.key, conn], dtype=torch.int64, device=device)
    start = time.perf_counter()
    hit = ops.kvcache_exist(query, output)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if hit not in (0, 1):
        raise RuntimeError(f"kvcache_exist returned invalid result: {hit!r}")
    print(
        f"EXISTS: device={args.device} dpu={args.dpu} conn={conn} "
        f"key={args.key} hit={hit} native_call_ms={elapsed_ms:.3f} config={config}",
        flush=True,
    )


if __name__ == "__main__":
    main()

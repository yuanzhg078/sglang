#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults for your GIDS installation.
export OPEXTENSION="${OPEXTENSION:-/path/to/libop_extension.so}"
export GIDS_DIR="${GIDS_DIR:-/path/to/gids_config_dir}"
# Per-GPU configs gids_config_<N>.txt need 0; gids_config.txt needs 1.
export USE_SINGLE_PORT="${USE_SINGLE_PORT:-0}"

GPUS="${GPUS-0,1,2,3,4,5,6,7}"   # empty string -> single-GPU mode
DEVICE="${DEVICE:-0}"            # used only without --gpus
EXISTS_DEVICE="${EXISTS_DEVICE-}" # empty -> skip EXISTS; set 1 to query on GPU 1
EXISTS_ON_WRITER="${EXISTS_ON_WRITER:-0}" # set 1 to also query on the PUT GPU
SKIP_GET="${SKIP_GET:-0}"        # set 1 to run PUT and EXISTS only
CONN="${CONN-}"                  # empty -> route each key to its DPU
CONNS="${CONNS-}"                # empty -> all connections; e.g. 0,1,2
LAYERS="${LAYERS:-78}"
PAGES="${PAGES:-100}"
KV_BYTES="${KV_BYTES:-73728}"
INDEX_BYTES="${INDEX_BYTES:-8448}"
WARMUP="${WARMUP:-0}"
GET_STAGGER_MS="${GET_STAGGER_MS:-0}"   # per-worker delay: i * N ms

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

args=(--device "${DEVICE}")
if [[ -n "${CONN}" ]]; then
  args+=(--conn "${CONN}")
fi
if [[ -n "${CONNS}" ]]; then
  args+=(--conns "${CONNS}")
fi
if [[ -n "${GPUS}" ]]; then
  args+=(--gpus "${GPUS}")
fi
if [[ -n "${EXISTS_DEVICE}" ]]; then
  args+=(--exists-device "${EXISTS_DEVICE}")
fi
if [[ "${EXISTS_ON_WRITER}" == "1" ]]; then
  args+=(--exists-on-writer)
fi
if [[ "${SKIP_GET}" == "1" ]]; then
  args+=(--skip-get)
fi

exec "${PYTHON_BIN:-python3}" "$script_dir/oceandisk_dispatch_roundtrip.py" \
  "${args[@]}" \
  --layers "${LAYERS}" \
  --pages "${PAGES}" \
  --kv-bytes "${KV_BYTES}" \
  --index-bytes "${INDEX_BYTES}" \
  --warmup "${WARMUP}" \
  --get-stagger-ms "${GET_STAGGER_MS}" \
  "$@"

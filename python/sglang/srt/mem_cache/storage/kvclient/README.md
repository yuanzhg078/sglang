# KvClient direct linker

Supports DSA KV and Indexer pools, prefix lookup, layer-wise GET and ordinary
whole-page PUT. Layer-wise PUT, MTP draft, SWA and PD notifications are excluded.
The SDK/provider must support concurrent submissions and make completed GETs
visible to the device. A successful key query must mean that all fragments of
both pools are readable; deployments where partial writes expose keys need a
publication protocol before this backend can be enabled.

Build the extension in the target inference environment with the SDK libraries
and device dependencies installed:

```sh
python -m pip install pybind11
cmake -S python/sglang/srt/mem_cache/storage/kvclient/bindings/python -B build/kvclient-python \
  -DKVCLIENT_SDK_ROOT=/path/to/kv_semantics \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/kvclient-python -j
cmake --install build/kvclient-python
```

If the SDK library is outside the searched directories, set `KVCLIENT_LIBRARY`
and `KVCLIENT_INCLUDE_DIR` explicitly. Make `libkv_client.so`, `libkv_transport.so`
and their dependencies available through the runtime library path. This directory
can also be added to an SDK build with an existing `kv_client` CMake target.

Enable with `--enable-unified-cache-external-linker
--unified-cache-external-linker-backend kvclient --kvclient-config-path /path/to/client.conf`.
Use deployment namespaces to isolate model versions and parallel shards; page
keys are exactly `sha256(page_key.encode()).digest()[:8]`.

Load failures are fatal and do not publish successful request completion. A
timed-out or indeterminate native Wait consumes its task ID and retains buffers, registrations and
tree protection: reset/close refuse to release them until IO quiescence can be
established. Restart the worker after resolving the underlying IO failure.

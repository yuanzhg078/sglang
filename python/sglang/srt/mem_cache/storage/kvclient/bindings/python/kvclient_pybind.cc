#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "kv_client.h"

namespace py = pybind11;
using RegionInput = std::tuple<std::uint64_t, std::uint64_t, std::int32_t>;
using EntryInput = std::tuple<py::bytes, std::uint64_t, std::uint64_t,
                              std::int32_t, kv::MRHandle, std::uint32_t>;

namespace {
void RequireOK(const kv::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error("KvClient status " +
                                 std::to_string(static_cast<int>(status.code)) +
                                 ": " + status.message);
    }
}

py::dict StatusDict(const kv::Status& status) {
    py::dict result;
    result["code"] = static_cast<int>(status.code);
    result["message"] = status.message;
    return result;
}

kv::CacheKey Key(const py::bytes& value) {
    std::string bytes = value;
    if (bytes.size() != kv::kCacheKeySizeBytes) {
        throw py::value_error("CacheKey must contain exactly 8 bytes");
    }
    kv::CacheKey key;
    std::memcpy(key.data(), bytes.data(), bytes.size());
    return key;
}

kv::MemoryRegion Region(const RegionInput& input) {
    const auto [addr, size, device] = input;
    if (!addr || !size || device < 0 || addr > UINT64_MAX - size) {
        throw py::value_error("Invalid device memory region");
    }
    kv::MemoryRegion region;
    region.memoryType = kv::MemoryType::DEVICE;
    region.addr = addr;
    region.size = size;
    region.deviceId = device;
    return region;
}

std::vector<kv::KVBuffer> Entries(const std::vector<EntryInput>& inputs) {
    std::vector<kv::KVBuffer> entries;
    entries.reserve(inputs.size());
    for (const auto& [key, addr, size, device, handle, offset] : inputs) {
        if (!handle || offset % kv::kAlignmentBytes ||
            size > (1ULL << 32) - offset) {
            throw py::value_error("Invalid handle or unaligned/out-of-range remote offset");
        }
        kv::KVBuffer entry{};
        entry.key = Key(key);
        entry.buffer.region = Region({addr, size, device});
        entry.buffer.handle = handle;
        entry.offset = offset;
        entries.push_back(entry);
    }
    return entries;
}
}  // namespace

class PyClient {
public:
    PyClient(const std::string& path, int device) {
        if (device < 0) {
            throw py::value_error("device_id must be nonnegative");
        }
        py::gil_scoped_release release;
        kv::KvClientConfig config;
        RequireOK(kv::KvClientConfig::LoadFromFile(path, config));
        for (auto& transport : config.transportConfigs) {
            transport.deviceId = device;
        }
        client_ = kv::CreateKvClient();
        RequireOK(client_->Init(config));
    }

    ~PyClient() {
        // KvClientImpl's destructor drains tasks before destroying resources.
        // Release the GIL since another Python worker may be awaiting the SDK.
        if (PyGILState_Check()) {
            py::gil_scoped_release release;
            client_.reset();
        } else {
            client_.reset();
        }
    }

    py::list RegisterRegions(const std::vector<RegionInput>& inputs) {
        std::vector<kv::MemoryRegion> regions;
        for (const auto& input : inputs) {
            regions.push_back(Region(input));
        }
        std::vector<kv::RegisteredMemory> registered;
        kv::Status status;
        {
            py::gil_scoped_release release;
            status = client_->RegisterRegions(regions, registered);
        }
        RequireOK(status);
        py::list result;
        for (const auto& item : registered) {
            py::dict record;
            record["region"] = py::make_tuple(item.region.addr, item.region.size,
                                              item.region.deviceId);
            record["handle"] = item.handle;
            record["token_id"] = item.tokenId;
            result.append(record);
        }
        return result;
    }

    kv::TaskId QueryAsync(const std::vector<py::bytes>& inputs) {
        std::vector<kv::CacheKey> keys;
        for (const auto& key : inputs) {
            keys.push_back(Key(key));
        }
        kv::TaskId task = kv::kInvalidTaskId;
        {
            py::gil_scoped_release release;
            RequireOK(client_->QueryAsync(keys, task));
        }
        return task;
    }

    kv::TaskId LoadAsync(const std::vector<EntryInput>& inputs) {
        auto entries = Entries(inputs);
        kv::TaskId task = kv::kInvalidTaskId;
        {
            py::gil_scoped_release release;
            RequireOK(client_->LoadAsync(entries, task));
        }
        return task;
    }

    kv::TaskId StoreAsync(const std::vector<EntryInput>& inputs) {
        auto entries = Entries(inputs);
        kv::TaskId task = kv::kInvalidTaskId;
        {
            py::gil_scoped_release release;
            RequireOK(client_->StoreAsync(entries, task));
        }
        return task;
    }

    bool Check(kv::TaskId task) {
        py::gil_scoped_release release;
        return client_->Check(task);
    }

    py::dict Wait(kv::TaskId task, std::uint64_t timeout) {
        kv::TaskResult result;
        kv::Status status;
        {
            py::gil_scoped_release release;
            status = client_->Wait(task, timeout, result);
        }
        py::dict output;
        output["wait_status"] = StatusDict(status);
        output["status"] = StatusDict(result.status);
        py::list entries;
        for (const auto& entry : result.entryStatus) {
            entries.append(StatusDict(entry));
        }
        output["entry_status"] = entries;
        if (result.queryResult) {
            output["exists"] = result.queryResult->exists;
            output["prefix_hit_keys"] = result.queryResult->prefixHitKeys;
        }
        return output;
    }

    void UnregisterRegions(const std::vector<kv::MRHandle>& handles) {
        py::gil_scoped_release release;
        RequireOK(client_->UnregisterRegions(handles));
    }

    void Shutdown() {
        py::gil_scoped_release release;
        RequireOK(client_->Shutdown());
    }

private:
    std::unique_ptr<kv::KvClient> client_;
};

PYBIND11_MODULE(_kvclient, m) {
    py::class_<PyClient>(m, "Client")
        .def(py::init<const std::string&, int>())
        .def("register_regions", &PyClient::RegisterRegions)
        .def("query_async", &PyClient::QueryAsync)
        .def("load_async", &PyClient::LoadAsync)
        .def("store_async", &PyClient::StoreAsync)
        .def("check", &PyClient::Check)
        .def("wait", &PyClient::Wait, py::arg("task_id"), py::arg("timeout_ms") = 0)
        .def("unregister_regions", &PyClient::UnregisterRegions)
        .def("shutdown", &PyClient::Shutdown);
}

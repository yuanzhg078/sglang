# KvClientLinkerBackend 设计方案

状态：设计草案，尚未实现。本文根据当前工作区代码和本地 KvClient 头文件编写；涉及底层传输行为的未确认项在文末列出。

## 1. 目标与范围

新增 `KvClientLinkerBackend`，实现 SGLang 的 `UnifiedCacheLinker` 接口，通过 `kv::KvClient` 将设备 KV cache 直接接入外部存储。

命名约定：

- Python 类：`KvClientLinkerBackend`。
- Python 文件：`kv_client_linker_backend.py`。
- 启动参数后端名称：拟定为 `kvclient`。
- 底层客户端：`kv::KvClient`，不以其所在仓库名称命名适配类。

第一阶段支持 FULL KV、单一已验证的设备和模型布局，完成查找、写入、读回、失败处理和资源回收。SWA、多 pool、逐层 IO 重叠及 PD 通知后续扩展；未支持的组合在初始化时明确拒绝。

## 2. 依据与现有入口

底层接口文件：

```text
D:\a_storage\4_ucm_dev最新代码\unified-cache-management\kv_semantics\include\kv_client.h
D:\a_storage\4_ucm_dev最新代码\unified-cache-management\kv_semantics\include\kv_types.h
```

SGLang 相关文件（路径相对仓库根目录）：

| 文件 | 职责 |
| --- | --- |
| `python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py` | 抽象接口及树侧 wrapper |
| `python/sglang/srt/mem_cache/registry.py` | 根据后端名称构造 linker |
| `python/sglang/srt/arg_groups/fields/memory.py` | external linker 参数及 choices |
| `python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py` | 将设备 pool 和页索引解析为物理布局 |
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py` | 排队、工作线程、层计数器实现参考 |

接入位置是 external linker 分支，不需要通过 `StorageBackendFactory` 接入 HiCache host storage。当前 registry 将 HiCache 和 external linker 作为不同初始化分支，第一版沿用该行为。

## 3. 分层与职责

```text
UnifiedRadixCache
  └─ UnifiedCacheLinkerWrapper
       └─ KvClientLinkerBackend
            ├─ DevicePoolGroup：布局和物理范围解析
            ├─ KeyCodec：稳定对象标识
            ├─ Load/Offload 队列及完成队列
            ├─ LayerDoneCounter：设备数据可用性门控
            └─ KvClientAdapter（Python/C++ binding）
                 └─ kv::KvClient
```

树侧 wrapper 继续负责缓存树操作、跨 rank 恢复边界协调和节点生命周期。新后端负责物理 IO、设备就绪同步、任务聚合、错误传播及 MR 生命周期。

绑定层负责 C++ 类型转换、原生客户端所有权、任务状态和设备同步桥接。第一版不在 Python 中直接模拟 C++ 对象内存布局，也不通过 ctypes 调用 C++ 虚函数接口。

## 4. 建议文件结构

```text
python/sglang/srt/mem_cache/storage/kvclient/
    __init__.py
    kv_client_linker_backend.py  # UnifiedCacheLinker 实现
    kv_client_adapter.py         # 对原生扩展的薄封装
    key_codec.py                 # 对象标识编码与布局版本
```

原生扩展建议由 KvClient SDK 一侧独立构建、发布，SGLang 通过可选依赖延迟导入。扩展模块名、链接库名和构建参数在核对 SDK 构建产物后确定。现有 `patch/pynds_kv` 仅作为待检查的候选绑定，不能据其目录名认定已经导出了新 `KvClient` API。

## 5. 接口映射

| Linker 方法 | 实现行为 |
| --- | --- |
| 构造函数 | 解析配置，创建并初始化客户端，解析 pool，注册设备 allocation，启动工作线程 |
| `lookup(rid, transfers)` | 展开逻辑页对应的全部对象，QueryAsync + Wait，计算可恢复页边界集合 |
| `load(rid, transfers)` | 校验并保存请求，不提交读取；持有必要引用 |
| `start_layer_wise_loading()` | 将当前排队请求组成 batch，分配层计数器索引，交给工作线程提交读取 |
| `cancel_queued_load(rid)` | 仅删除尚未启动的请求；已经提交的请求不作取消成功承诺 |
| `num_completed_loads()` | 返回已发布的成功 load batch 数量 |
| `pop_completed_load()` | 消费一个成功 batch，返回其 rid 列表 |
| `offload(transfers)` | 校验、捕获设备就绪依赖、排队，返回是否接受任务 |
| `num_completed_offloads()` | 返回可消费的 offload 结果数量 |
| `pop_completed_offload()` | 消费一次逻辑 offload 的整体成功结果 |
| `reset()` | 停止接收、排空活动 IO、清理队列与计数器，成功后恢复接收 |
| `close()` | 停止接收、排空 IO、停止线程、注销 MR、Shutdown；幂等 |

`load()`、`offload()` 返回 True 只表示接受请求，最终完成另行通知。`reset()` 不删除远端缓存，`DeleteAsync()` 不属于基础接入必需路径。

## 6. 对象模型与 Key

一个逻辑页可能覆盖多个层、K/V 分量和辅助 pool。设计上先采用“每个独立物理分量一个对象”，使每次 IO 对应一个确定的连续设备范围，避免在尚未确认 offset 语义时依赖远端对象内的分段写入。

对象身份至少包括：

```text
schema_version
cache_namespace（模型权重/版本标识）
layout_fingerprint（dtype、page_size、形状、分片方式等）
pool_name
pp_rank / cp_rank / tp_shard（复制型 pool 使用统一约定）
logical_layer_id
component_id
page_storage_hash
```

模型路径本身不能充分区分权重版本。布局指纹必须基于稳定序列化规则生成，不能包含本地设备地址、进程号或 Python hash()。

`kv::CacheKey` 固定为 8 字节。拟使用版本化身份串的 SHA-256 前 8 字节；绑定直接传递 8 字节，避免有符号整数和端序转换歧义。该方案只能降低碰撞概率，不能检测或消除碰撞。生产使用前需明确接受该风险，或由底层增加更长 key / 完整身份校验能力；单纯再哈希不能解决这一限制。

第一版每个物理对象单次完整写入，拟使用 offset=0，但仍须确认其底层语义、对象长度及创建规则后实施。

## 7. 设备布局与内存注册

复用 `resolve_hybrid_device_pool_group()` 建立 pool 视图，通过 `resolve_transfers()` 校验 transfer，通过 pool 的页/层元数据方法生成地址和大小。不能假设不同模型的 KV 都具有相同连续布局。

注册流程：

1. 遍历 `get_hybrid_pool_buffer()`，取得各 Tensor 的底层 allocation 地址及容量。
2. 按 allocation 去重，构造带正确 deviceId、MemoryType 的 `MemoryRegion`。
3. 调用 `RegisterRegions()`，保存完整 `RegisteredMemory`，包括 region、handle 和 tokenId。
4. 每个 IO 范围必须完全落在对应注册区间内。
5. 使用经实现验证的规则构造 `KVBuffer.buffer.region` 和 handle；`mrKey` 默认交由 KvClient 解析。

持有原始 Tensor、注册记录及任务描述符，直到相关 IO 确定停止。注销前必须排空引用该 handle 的全部任务。

头文件声明设备 buffer 和 offset 需 512B 对齐。初始化及提交阶段检查实际范围；长度约束也需从实现确认。第一版对不满足直接 IO 要求的布局报错。不得通过向前/向后扩大用户 buffer 范围实现对齐，因为读入可能覆盖相邻 KV 页。后续 staging buffer 方案需单独处理分配、设备拷贝、同步和容量限制。

## 8. 查找与完整性

`lookup()` 的返回值是本 rank 所有可恢复前缀长度，单位为页。

第一版 FULL KV：对每个页查询所有必要层和分量，仅在全部对象存在时认为该页存在，再返回从第一页起的连续可恢复边界。例如完整页标记为 `[True, True, False, True]`，返回 `[1, 2]`。

`QueryResult.prefixHitKeys` 是对象序列的前缀结果，不能直接作为逻辑页数量。必须使用 `exists` 和逻辑页到对象的映射归并；查询失败或结果长度异常时不报告未经证实的命中。

后续扩展多 pool 时，ALL_PAGES 要求对应前缀中的所有必要对象存在；TRAILING_PAGES 要求在候选边界处所需窗口存在。因此返回集合可能稀疏，树侧再跨 rank 求交集。

KvClient 头文件没有保证多对象原子持久化。第一版依赖“对象存在意味着该对象已完整可读”的底层契约，同时查询全部必需对象，仅在全体写入成功后上报 offload 成功。如果底层可能让未完成对象被查询为存在，就必须增加发布标记或等效提交协议；只检查每个对象存在不足以保证正确性。

## 9. Load 生命周期与层门控

```text
QUEUED → SUBMITTED → IO_COMPLETE → COUNTER_READY → COMPLETED
                     └─ FAILED → 唤醒等待者并传播错误
```

第一版采用整批读取：后台线程提交 LoadAsync，等待所有必要对象读取成功，再依次完成该 batch 的全部层计数器，最后发布 rid 列表。该方式先保证正确性，暂不实现 IO 与逐层计算重叠。

后续可以按层拆任务，在该层全部 pool 和对象成功、设备写入可见后推进相应计数器。`layer_done_counter` 必须符合现有 pool 和 worker 的消费约定。

IO completion 是否已经包含设备侧可见性保证，需要底层明确。若还需设备 event 或 stream 等待，必须在推进计数器前完成。失败不能伪装成正常完成；等待 Future 应携带异常，同时后端保存 fatal error，在调度线程后续入口抛出，避免工作线程退出后主线程永久等待。

对于已提交任务，取消请求不意味着可以立即释放其目标页。后端不提供原生取消时，保持页和任务引用直至 IO 静默，并沿用 wrapper 的生命周期约束。

## 10. Offload 生命周期与设备同步

```text
ACCEPTED → WAIT_DEVICE_READY → SUBMITTED → IO_COMPLETE → SUCCESS / FAILURE
```

调用线程在正确的生产 stream 上捕获设备 event。可通过 `StoreAsync(..., prerequisite)` 在客户端 worker 中设置设备上下文并等待 event；若设备或绑定不支持该桥接，第一版由适配层工作线程等待 event 后再提交。

原生 prerequisite 优先使用 C++ 持有的设备 event，避免回调重新获取 Python GIL 导致死锁。回调需要的 event、stream 和设备资源一直存活到回调返回。

一次逻辑 offload 可能拆为多个 task，仅在所有必需对象成功后发布 True，否则发布 False。拒绝或失败时仍需遵守树侧完成消费约定，不能丢失已接受任务的结果。页在写任务完成前不得被复用或修改。

## 11. 异步结果、错误及超时

必须区分三层状态：

1. Async 方法返回值：是否成功提交并取得有效 TaskId。
2. Wait 返回值及 TaskResult.status：任务执行/等待整体状态。
3. entryStatus 或 queryResult：逐条对象结果。

只有满足对应操作成功条件的对象才能进入成功集合。查询的存在性以 queryResult.exists 为准，不把任意非零状态码当作成功；检查结果数量和提交数量是否一致。

任务登记表保存 TaskId、操作类型、所属逻辑 batch、对象映射、buffer/MR/event 引用和提交时间。每个 TaskId 只能被一个消费者 Wait 一次。

`Check()` 对未知 TaskId 也返回 True，因此只能检查本地登记表中的有效任务。`Wait()` 即使超时也消费任务，禁止将其用作可重复的短超时轮询。正常路径先 Check，再 Wait 取结果；Wait、Shutdown 等可能阻塞的绑定调用应释放 GIL。

第一版默认不对已提交的写入做自动重试。超时是否保证 IO 已停止、是否可通过 Shutdown 排空，必须确认；在无法证明静默时进入 fatal 状态，不复用相关 buffer，不注销 MR，不以普通 cache miss 掩盖潜在设备写入。查询提交失败可以保守返回 miss，但失效结果不能进入正命中缓存。

## 12. 并发、背压与关闭

建议 Python 维护有界 load/offload 队列，各有后台工作线程；查找在调用路径执行。共享客户端是否允许并发调用尚需核实；若不允许，改用统一原生调用入口调度，不能持有一个全局锁阻塞等待全部任务而饿死其他 IO。

队列容量、在途 task 数和字节数应受限，并与 KvClient 配置保持一致。排队未接受时返回 False；已经接受的任务必须产生结果或明确传播 fatal error。

关闭顺序：

```text
停止接受任务
→ 唤醒工作线程并处理已接受任务
→ 确认所有原生 IO 静默
→ join 工作线程
→ UnregisterRegions
→ Shutdown
→ 释放设备对象及客户端
```

如果超时任务只能依赖 Shutdown 才能静默，需根据底层契约调整注销顺序，不能机械执行上述流程。构造中途失败也要回滚已经获得的资源。`close()` 不依赖进程退出时的隐式析构顺序。

## 13. 配置与接入改动

拟新增专用配置入口 `--kvclient-config-path`，传给 `KvClientConfig::LoadFromFile()`；不直接借用名称带 hicache 的配置承载全部客户端参数。命名空间和超时/队列策略等适配层配置在实现阶段明确字段并校验。

需要修改：

- `memory.py`：为 external linker choices 增加 `kvclient`，声明配置入口。
- `registry.py`：延迟导入并构造 `KvClientLinkerBackend(server_args, params, components=...)`。
- 按仓库参数生成/字段顺序机制补齐新增字段，并检查相关配置校验。
- 运行说明：记录 SDK 安装、设备支持范围、客户端配置及已验证布局。

预期启动参数片段（尚未实现，不能直接使用）：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend kvclient
--kvclient-config-path /path/to/kvclient.conf
```

第一版拒绝 `disaggregation_linker_notify` 等尚未接入的扩展组合，避免使用未实现的 PD 语义。

## 14. 实施步骤与验收

### 阶段一：核实 SDK 契约及原生绑定

确认 offset、注册区间、对齐、线程安全、结果状态、设备可见性和超时排空语义。提供 Init、注册、Query/Load/Store、Check/Wait、Shutdown 的最小绑定，以实际设备缓冲完成单对象读写验证。

### 阶段二：FULL KV 最小闭环

实现稳定对象 key、设备 pool 解析、内存注册、查询和整批加载；接入 registry、启动参数、完成队列和层门控。设备不支持或布局不满足约束时在启动阶段给出错误。

验收：确定性填充 KV → offload → 等完成 → 清空或重新分配本地页 → lookup → load → 门控释放后逐字节比较。覆盖不同页位置、多个层、K/V 分量以及多 rank 分片隔离。

### 阶段三：异常与生命周期

使用 fake adapter 验证部分失败、结果乱序、查询缺失对象、提交拒绝、队列满、排队取消、reset/close、初始化回滚及后台错误唤醒。对超时中的设备任务做真实故障注入，证明完成通知及资源回收符合底层保证。

### 阶段四：扩展与性能

在正确性闭环稳定后增加逐层加载、多 pool/SWA 恢复集合、批次优化和必要的 staging 支持。性能观测包括 lookup 耗时、队列等待、IO 时延、在途字节、逐条失败及超时次数。

## 15. 实现前必须确认的问题

| 问题 | 对设计的影响 |
| --- | --- |
| KVBuffer.offset 的确切含义及对象创建/定长规则 | 决定对象拆分及远端偏移编码 |
| buffer.region 应描述 allocation 还是本次 IO 子区间 | 决定注册 handle 与地址/长度的构造方式 |
| 512B 约束具体作用于地址、长度和哪些 offset | 决定直接 IO 可支持的模型布局 |
| 哪些设备、MemoryType 和 provider 组合真正支持 direct IO | 决定首版平台及启动校验 |
| Query 存在是否保证完整数据已可读；并发覆盖语义如何 | 决定是否必须增加提交标记 |
| Wait 超时后 IO 是否仍会访问设备内存，如何排空 | 决定失败后能否安全恢复和释放资源 |
| 完成状态是否保证设备写入对后续计算可见 | 决定层计数器之前的同步操作 |
| KvClient 公共方法的并发调用保证 | 决定查找及工作线程调度方式 |
| QueryResult、entryStatus 的数量、顺序及成功码契约 | 决定逐对象结果归并规则 |
| 64 位 key 碰撞风险是否被接受 | 决定生产可用的 key/身份校验方案 |

上述问题未解决前，可以完成后端骨架与 fake adapter 测试，但不能宣称设备直连及异常回收已经正确。

# Unified Cache Linker 与 OceanDisk 关系总结

核心关系：**UnifiedCacheLinkerWrapper 把缓存树操作转成传输任务；OceanDiskDirectLinker 把传输任务转换成 OceanDisk 实际 I/O。**

## 1. 类与对象关系

下面区分持有关系和继承关系，不能把所有类理解成一条继承链。

```text
UnifiedRadixCache
  └─ 持有 UnifiedCacheLinkerWrapper（cache.linker）
       ├─ 引用 cache：操作树节点、请求、锁
       └─ 持有 cache_linker：接口类型为 UnifiedCacheLinker
            └─ 实际对象为 OceanDiskDirectLinker 或 OceanDiskPDLinker

继承关系：
UnifiedCacheLinker（ABC，传输接口）
  └─ OceanDiskDirectLinker（OceanDisk 通用传输实现）
       └─ OceanDiskPDLinker（增加 PD 交接能力）

OceanDiskDirectLinker
  ├─ 持有 DevicePoolGroup
  │    └─ 包含 DevicePoolEntry：GPU 物理池视图、地址和布局映射
  └─ 持有 OceanDiskStorage：GIDS I/O、内存注册及 connection 路由
```

接入缓存树时：

```python
# UnifiedRadixCache.init_cache_linker()
self.linker = UnifiedCacheLinkerWrapper(self, cache_linker)
```

因此：

```python
cache.linker                # 树侧 Wrapper
cache.linker.cache_linker   # OceanDiskDirectLinker / OceanDiskPDLinker 实例
```

## 2. 各层职责

| 层 | 负责什么 | 主要输入 / 输出 |
| --- | --- | --- |
| UnifiedRadixCache | 前缀缓存的树与调度入口，调用 Wrapper 钩子 | 请求、token、树节点 |
| UnifiedCacheLinkerWrapper | 组织命中查询、恢复、备份，维护树状态及 I/O 生命周期 | 树操作 → PoolTransfer；完成结果 → 树状态提交和解锁 |
| UnifiedCacheLinker | 定义传输契约，不提供具体后端实现 | lookup、load、offload、完成队列、生命周期方法 |
| OceanDiskDirectLinker | 将逻辑传输转换为物理池的逐层 I/O；管理队列、线程和完成计数 | PoolTransfer → object key、GPU pointer、size、remote offset |
| OceanDiskPDLinker | 复用 DirectLinker，增加 P/D namespace、后台加载与交接控制 | PD handoff job、READY、decode gate |
| DevicePoolGroup / Entry | 将逻辑池和 slot 映射到已有 GPU 物理 buffer | 逻辑 transfer → 物理池 transfer / 每层地址元数据 |
| OceanDiskStorage | 执行 EXISTS / GET / PUT，封装 GIDS 原生调用与连接路由 | key 和地址描述 → 实际存储操作 |

### Wrapper：树侧编排与生命周期

- 计算 GPU 未命中部分的 page hashes，调用 linker 查询外部缓存。
- 协调各 rank 的可恢复前缀边界。
- 通过 tree component 协调 GPU slot 分配、树插入和重复前缀处理。
- I/O 期间锁住树节点；逐层保存时 pin 住源页面，避免提前回收。
- 消费完成结果，提交外部存储状态并释放节点锁或页面 pin。

### DirectLinker：具体传输实现

- 通过 namespace 将 page hash 编码为 OceanDisk object key。
- 使用 DevicePoolGroup 展开逻辑池，生成每层物理 I/O 描述。
- 管理 load / offload 队列、后台 worker 和完成结果。
- 通过 layer_done_counter 暴露逐层加载就绪状态。
- 调用 Storage 执行 EXISTS、GET 和 PUT。

**树节点能否回收由 Wrapper 管理；数据如何传输由 DirectLinker 管理。**

## 3. 双方通过 PoolTransfer 衔接

PoolTransfer 可以理解为一份逻辑传输描述：

```text
逻辑池名称 + page keys + 对应的 GPU slot indices
```

Wrapper 确定哪些数据需要传输，DirectLinker 再利用池布局确定每层具体读写哪些地址。

DevicePoolGroup 可以将一个逻辑 KV transfer 展开到 KV、Indexer 等物理池。当前 OceanDisk DirectLinker 的布局校验要求物理池为 KV 和 Indexer，并要求 packed page 布局；通用接口支持范围不等于这个具体后端的支持范围。

其中部分中间字段沿用 `host_indices` 名称，但在此路径中用于保存转换后的索引，不能据此认为 KV 数据经过 CPU host tier。CPU 上处理索引或元数据，也不等于把 KV payload 搬到 CPU。

## 4. 外部缓存命中与加载流程

```text
UnifiedRadixCache.match_prefix()
  → Wrapper.match()
      → 计算 GPU 未命中尾部的 page hashes
      → 构造各 component 的 PoolTransfer
      → DirectLinker.lookup()
          → DevicePoolGroup.resolve_transfers()
          → Storage.exists_many()
      → 跨 rank 汇总可恢复边界
      → 保存请求的 hit_marker

UnifiedRadixCache.init_load_back()
  → Wrapper.load_back()
      → 协调分配 GPU slot、插入树、选择实际采用的页面
      → 锁住加载目标节点
      → DirectLinker.load()：排队

start_layer_wise_loading()
  → DirectLinker 生成逐层 GET 地址描述
  → Storage.get_layers()
  → 更新 layer_done_counter

树侧消费加载完成结果
  → Wrapper 释放目标节点锁
```

注意：`load()` 成功表示任务已经排队，不表示 KV 已经可读。当前实现会先发布树节点再启动实际读取，节点锁保护内存生命周期，逐层计数器控制各层数据的可用性。

外部命中结果借用 `host_hit_length` 等字段传递给调度逻辑，不代表实际经过 host cache。

## 5. 逐层保存与树侧 offload 的关系

当前 DirectLinker 默认启用 `layerwise_save`。此时，实际写数据和树侧确认持久化分为两个阶段。

### 阶段一：forward 过程中逐层写入

```text
forward 前
  → Wrapper.prepare_layerwise_save(batch)
      → 选择新完成的整页、计算 hash、取得 GPU slot
      → DirectLinker.begin_layerwise_save()
      → 返回 LayerwiseSaveSession
      → Wrapper pin 住源页面

每层 forward 完成
  → session.save_layer(layer)
  → 记录 CUDA ready event，进入 offload 队列
  → worker 调用 Storage.submit_put_layer()

session 结束
  → worker 等待写入完成，产生 completion receipt
  → 树侧协调完成结果并调用 session.finish()
  → 释放源页面 pin
```

逐层保存由模型 forward 的钩子驱动，Wrapper 负责规划和源页面生命周期，DirectLinker 的 session 与 worker 负责提交 I/O。

### 阶段二：树侧确认持久化

```text
树产生 BackupKV action
  → Wrapper.offload_nodes()
      → 构造节点的 PoolTransfer、锁住节点
      → DirectLinker.offload()
          → 检查对应页面的逐层保存 receipts
      → 树侧协调成功结果
      → 提交节点的外部存储状态并解锁
```

**启用逐层保存时，offload() 主要确认已有保存结果，不重新发起 PUT，也不负责失败后的 PUT 重试。**

关闭逐层保存时，offload worker 才会等待源数据 ready event，并通过 `put_layers()` 执行完整写入。

## 6. PDLinker 与 DirectLinker 的关系

`OceanDiskPDLinker` 继承 `OceanDiskDirectLinker`，由 `--disaggregation-linker-notify` 路径选择。它复用物理池映射、object key 编码和存储能力，再增加：

- P/D 可共享的 namespace。
- DecodeHandoffLoader：D 分配 slot 后提前后台加载缺失历史页。
- DecodeLoadGate：decode 活跃时限制存储操作的发起频率。
- handoff keys、EXISTS、GET 以及交接状态处理。

Wrapper 仍然存在，并通过 handoff 钩子参与 PD 保存规划；但 D 的 handoff 加载走专门的 job / loader 路径，不能把它等同于普通的 `Wrapper.match() → load_back()` 流程。

## 7. 源码位置

以下路径相对于仓库根目录：

| 内容 | 文件 |
| --- | --- |
| 缓存树接入、完成结果协调 | `python/sglang/srt/mem_cache/unified_radix_cache.py` |
| ABC、Wrapper、树侧加载与备份 | `python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py` |
| DirectLinker、逐层保存 session、I/O worker | `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_direct_linker.py` |
| PD 扩展 | `python/sglang/srt/mem_cache/storage/oceandisk/pd_linker.py` |
| PD 后台加载与 gate | `python/sglang/srt/mem_cache/storage/oceandisk/pd_decode_loader.py` |
| PD 加载任务与 receiver 放行控制 | `python/sglang/srt/disaggregation/decode_linker_mixin.py` |
| GPU 物理池视图与 transfer 展开 | `python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py` |
| GIDS 存储调用与连接路由 | `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_storage.py` |

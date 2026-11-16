# UnifiedCacheLinker：运行逻辑与新增后端指南

本文基于当前仓库代码，解释 UnifiedCacheLinker（下文简称 UC Linker）如何连接缓存树与外部存储，以及如何新增一个类似 OceanDiskDirectLinker 的后端。文中的 `MyDirectLinker`、`MyStorage` 和 `mybackend` 都是示例名称，不是已经存在的实现。

## 1. 先理解它解决什么问题

模型算过一段 token 后，KV 数据在 GPU 上。GPU 空间有限，因此希望把这些数据保存到外部存储；以后遇到相同前缀，就读回 GPU，减少重复计算。

UC Linker 路径连接的是：

```text
缓存树管理的 GPU KV 数据 ←→ 外部存储
```

这条路径没有中间的 CPU host cache tier。CPU 仍然可以处理索引、hash、队列和 I/O 描述；这与把 KV payload 缓存在 CPU 上是两回事。

这里有两个问题需要分别解决：

1. 哪段前缀命中了？哪些 GPU slots 可以用？节点什么时候可以回收？这些由缓存树和 Wrapper 协调。
2. 从存储的哪个对象读取、写到哪个 GPU 地址、什么时候真正完成？这些由具体后端 linker 和 storage 协调。

## 2. 名字相近，但不是同一个对象

先用最简单的话区分：

- **`UnifiedRadixCache`：管缓存。** 哪些前缀已缓存、放在哪里、什么时候可以回收。
- **`UnifiedCacheLinkerWrapper`：帮它组织外部存取。** 哪些数据要从 OceanDisk 等后端加载或保存，传输期间保护节点，完成后更新状态。

**Cache 是缓存管理者，Wrapper 是它对接外部存储的助手。** 不启用外部 linker 时，Cache 也能管理 GPU 前缀缓存；启用后才挂接 Wrapper，复用 Cache 的树和内存管理能力。Wrapper 没有自己的另一棵缓存树。

例如一次从 OD 恢复缓存，Wrapper 组织的流程是：

```text
分配 slots → 插入树 → 保护节点 → 调用 OD → 完成后解除保护
```

这些步骤不是 Wrapper 自己全部实现：它通过 component 和已有分配器协调 slots 分配，通过 Cache 操作树和节点锁，再调用 OD 后端发起传输，最后消费完成通知并解除保护。它把后端传输前后的树侧步骤串成完整流程，因此叫 Wrapper（包装层）。其中“调用 OD”先排队，再由后续启动方法执行读取，不表示调用返回时数据已可用。

| 名称 | 性质 | 主要职责 |
| --- | --- | --- |
| `UnifiedRadixCache` | 缓存树及其对调度器的入口 | 前缀匹配、树节点、GPU slot 生命周期、各 rank 协调 |
| `UnifiedCacheLinkerWrapper` | 树侧编排对象 | 把树侧动作变成传输任务，维护命中标记、在途任务及节点锁 |
| `UnifiedCacheLinker` | 抽象接口（ABC） | 规定后端必须提供的查询、加载、保存、完成通知和生命周期方法 |
| `OceanDiskDirectLinker` 等 | 实现上述接口的具体后端 | 组织对象 key、地址描述、逐层 I/O、后台任务和完成结果 |
| `DevicePoolGroup / DevicePoolEntry` | 公共 GPU 池适配工具 | 把逻辑池和 slot 转换为物理池及内存布局元数据 |
| `OceanDiskStorage` 等 | 具体存储适配对象 | 调用存储 SDK / 原生接口，执行存在性查询和实际读写 |

持有关系与继承关系分别是：

```text
持有关系：
UnifiedRadixCache
  └─ cache.linker = UnifiedCacheLinkerWrapper
       ├─ self.cache：引用上面的同一个 UnifiedRadixCache
       └─ self.cache_linker：具体后端，例如 OceanDiskDirectLinker
            ├─ pool_group：GPU 物理池视图
            └─ storage：存储操作适配对象

继承关系：
UnifiedCacheLinker
  ├─ OceanDiskDirectLinker
  │    └─ OceanDiskPDLinker
  ├─ MooncakeDirectLinker
  └─ UMBPDirectLinker
```

### 2.1 Wrapper 与 UnifiedCacheLinker 是什么关系？

**Wrapper 持有并调用一个实现了 `UnifiedCacheLinker` 契约的后端对象，两者没有继承关系。**

```text
UnifiedRadixCache
  └─ 持有 UnifiedCacheLinkerWrapper：树侧组织者
       └─ cache_linker 引用实际后端，例如 OceanDiskDirectLinker

OceanDiskDirectLinker
  └─ 继承 UnifiedCacheLinker：实现抽象接口要求的方法
```

| 类 | 负责什么 |
| --- | --- |
| `UnifiedCacheLinkerWrapper` | 协调树节点与 slots、保护节点、发起任务、消费完成结果 |
| `UnifiedCacheLinker` | 定义后端必须提供的方法，如 lookup、load、offload；本身是抽象类 |
| `OceanDiskDirectLinker` | 实现这些方法，完成 OD 的具体传输适配 |

Wrapper 构造函数中的类型标注与运行时对象需要分开看：

```python
class UnifiedCacheLinkerWrapper:
    def __init__(self, cache, cache_linker: UnifiedCacheLinker):
        self.cache = cache
        self.cache_linker = cache_linker
        # 类型契约：UnifiedCacheLinker
        # 实际对象：传入的 OceanDiskDirectLinker 或其他后端实例
```

这里没有额外创建一个 `UnifiedCacheLinker` 抽象类对象。Wrapper 调用 `self.cache_linker.load(rid, transfers)` 时，执行的是实际后端的方法：传入 OD 就执行 OD 的实现，传入 Mooncake 就执行 Mooncake 的实现。因此更换后端时，Wrapper 的树侧流程可以复用。

### 2.2 用三个对象理解这些名字

假设使用 OD 后端，下面的简化伪代码展示对象关系，省略实际构造参数；真实代码通过 `cache.init_cache_linker(od)` 创建并挂接 Wrapper：

```python
cache = UnifiedRadixCache(...)   # 缓存树对象
od = OceanDiskDirectLinker(...)  # OD 传输对象
wrapper = UnifiedCacheLinkerWrapper(cache, od)
cache.linker = wrapper

# Wrapper 构造函数内保存：
# self.cache = cache
# self.cache_linker = od
```

因此，以下表达式指向相同对象：

```python
cache.linker is wrapper               # True
cache.linker.cache_linker is od       # True
wrapper.cache is cache               # True
```

`cache.linker` 这个属性名指的是 Wrapper；它里面的 `cache_linker` 才是 OD 后端。不要把两者都简称为 linker 后就混成一个对象。

### 2.3 load_back() 与 load() 的区别

```text
cache.linker.load_back(req)
也就是 wrapper.load_back(req)：组织整个恢复流程
    ├─ 通过 cache 和 components 准备 GPU slots
    ├─ 通过 cache 插入/处理树节点，选出实际需要加载的页面
    ├─ 通过 cache 保护目标节点
    └─ 调用 od.load(rid, transfers)：把读取任务交给后端排队
```

所以，`cache.linker.load_back(req)` 是“让 Wrapper 组织恢复缓存”；`cache.linker.cache_linker.load(rid, transfers)` 是“让 OD 接收这份读取任务”。它们是两个不同对象的方法，前者内部会调用后者。OD 接受任务后，实际读取还要由后续的 `start_layer_wise_loading()` 启动。

### 2.4 引用 cache 和节点锁是什么意思

`self.cache = cache` 只是保存访问同一棵树的入口，没有复制缓存树：

```text
外面的 cache ──────────┐
                      ├──→ 同一个 UnifiedRadixCache 对象
Wrapper 的 self.cache ┘
```

Wrapper 通过 `self.cache` 修改节点，就是在修改外面那棵树。这样它可以在调用 OD 前准备树状态，在传输结束后更新状态和释放保护。

这里的节点锁可以理解成给节点挂上“正在使用，暂勿回收”的标记：

```text
读取前：inc_lock_ref() → 增加锁引用，保护节点关联的 GPU 内存
读取中：GPU 内存仍在接收数据，不能被回收后分配给别的缓存页
完成后：dec_lock_ref() → 解除本次保护；其他引用仍可能继续保护它
```

它主要是引用计数式的回收保护，不表示其他线程都不能操作缓存树。锁保护内存生命周期；数据是否已经可读，还要通过逐层完成计数判断，详见第 5 节。

## 3. 初始化：后端如何接到缓存树上

入口在 `python/sglang/srt/mem_cache/registry.py` 的 `_create_unified_radix_cache()`：

```text
创建 UnifiedRadixCache 和 tree components
  → 检查 enable_unified_cache_external_linker
  → 按 unified_cache_external_linker_backend 选择后端类
  → linker_cls(server_args, params, components=set(cache.components))
  → cache.init_cache_linker(后端实例)
  → 创建 UnifiedCacheLinkerWrapper(cache, 后端实例)
  → 将后端的 layer_done_counter 注册给 GPU KV pool 和 TP worker
```

### 3.1 ODLinker 在哪一步挂进去？

在创建缓存树的初始化阶段，按配置选择类、创建后端对象，再通过 `cache.init_cache_linker(od)` 挂接。以 OD 为例，下面是省略其他分支的等价示意代码：

```python
cache = UnifiedRadixCache(params)
linker_cls = OceanDiskDirectLinker  # backend 配置为 oceandisk
od = linker_cls(server_args, params, components=set(cache.components))
cache.init_cache_linker(od)
```

`init_cache_linker()` 内部执行：

```python
self.linker = UnifiedCacheLinkerWrapper(self, cache_linker)
```

于是对象关系成为：

```text
cache
  └─ linker → Wrapper
                 ├─ cache → 原来的 cache
                 └─ cache_linker → 刚创建的 od
```

所谓“注册 ODLinker”，在这里实际是创建对象并保存到 Wrapper，没有单独的 `register_odlinker()`。选择后端的分支位于 `mem_cache/registry.py`。

随后注册 `layer_done_counter` 是另一件事：把同一个 OD 对象提供的加载等待机制接给 GPU KV pool 和 TP worker，让模型使用某层 KV 前能等待该层 GET 完成；不是再次挂接一个 OD 对象，也不会因此启动 GET。

### 3.2 当前后端选择

当前配置名称与类的对应关系：

| CLI backend 值 | 类 |
| --- | --- |
| `mooncake` | `MooncakeDirectLinker` |
| `oceandisk` | `OceanDiskDirectLinker`，启用相应 PD notify 配置时选择 `OceanDiskPDLinker` |
| `mori` | `UMBPDirectLinker` |

当前 Wrapper 只接受 FULL、SWA tree components；物理池工具支持某种池，并不代表 Wrapper 已支持对应的树组件。后端自己的布局限制还可能更严格，例如当前 OD 对物理池和 packed 布局有专门校验。

## 4. 树与后端之间传递什么：PoolTransfer

可以把 `PoolTransfer` 理解成一张传输单：

```text
name            逻辑池，例如 KV 或 SWA
keys            要查询、加载或保存的缓存页 hash
device_indices  这些页对应的 GPU token slots
hit_policy      恢复这段前缀需要满足的页面命中规则
```

查询阶段还没有分配目标 GPU slots，因此不一定有 `device_indices`。加载、保存阶段才需要实际的位置。

举例：假设 `page_size = 4`，两页数据的传输单可能表示：

```text
keys = [A, B]
device_indices = [128,129,130,131, 256,257,258,259]

A 对应 slots 128～131
B 对应 slots 256～259
```

这里的 slot 是分配器使用的编号，不是 GPU 指针；页 hash 也还不一定是存储系统最终使用的完整 object key。

`PoolTransfer` 是其他缓存路径也使用的数据结构，包含 `host_indices` 等字段。当前 direct-linker 的 `DevicePoolGroup.resolve_transfers()` 会把转换后的索引放入 `host_indices`，这个历史字段名不表示数据经过 CPU host cache。

## 5. 一次外部命中与 GET 的完整流程

### 5.1 先查 GPU，再查外部存储

```text
UnifiedRadixCache.match_prefix()
  → 先取得 GPU 缓存树的匹配结果
  → Wrapper.match()
      → 只对 GPU 尚未命中的完整页生成 hash
      → 各 tree component 构造查询用 PoolTransfer
      → 后端 lookup(rid, transfers)
      → 汇总各 rank 都可恢复的前缀边界
      → 保存 hit_markers[rid]
      → 返回外部命中长度给调度逻辑
```

`lookup()` 只是查询，既不分配目标 GPU slots，也不执行 GET。

它返回的是**本 rank 可完整恢复的前缀页数集合**，例如 `[2, 4]`，而不是一个最大长度，也不是每页的命中布尔值。对于普通连续前缀，候选可能是 `[1, 2, 3]`；带 trailing-window 状态时，候选边界可能稀疏。

例如 rank 0 返回 `[2, 4]`，rank 1 返回 `[2, 3]`，共同可恢复边界只有 2 页。只取两个最大值的最小值会得到 3，但 rank 0 不能恢复 3 页，因此会出错。Wrapper 做的是集合交集后选择最大共同边界。

命中策略包括：

- `ALL_PAGES`：所需前缀的每一页都存在。
- `TRAILING_PAGES`：在候选前缀边界处，所需的尾部窗口页面存在。

所有需要的池都满足各自策略，这个边界才算可恢复。结果中的 `host_hit_length` 是复用调度字段，不代表经过 CPU host cache。

### 5.2 准备 GPU 位置并把读取排队

```text
UnifiedRadixCache.init_load_back()
  → Wrapper.load_back(req)
      → 消费 hit_markers[rid]
      → FULL component 构造 LOAD transfer 时调用 allocator 分配 slots
      → 协调各 component 的 PREPARE 阶段（FULL 此时直接返回 transfer）
      → 将前缀插入树，处理已经由其他请求插入的重复范围
      → COMMIT 阶段保留本次实际采用的页面和索引
      → 增加目标节点的锁引用
      → 后端 load(rid, transfers)
      → 在 pending_loads 里记录目标节点及解锁参数
```

重要细节：树插入和页面选择发生在实际读取之前。并发请求可能已经补齐部分数据，因此后端收到的加载任务可能只包含剩余页面，甚至只剩 SWA 等侧池，不能假设每次 `load()` 都包含完整 KV transfer。

#### “准备 GPU slots”具体是什么意思？分配器属于谁？

这里的 GPU KV 分配器是已有的内存管理设施，由 `cache.token_to_kv_pool_allocator` 等入口访问，**不是 UC Linker 专属的分配器，也不是 ODLinker 自己创建的一套分配逻辑**。正常模型计算也需要通过分配器获得存放 KV 的位置；UC Linker 加载外部缓存时，复用这套能力来准备目标位置。

可以分清三个概念：

| 概念 | 含义 |
| --- | --- |
| GPU KV pool | 已经创建、用于存放 KV 数据的 GPU buffers |
| GPU KV 分配器 | 管理池内哪些位置可用，分配和释放 slots |
| UC Linker 的加载流程 | 通过 cache 和 component 协调分配，把目标 slots 交给后端读取 |

例如外部存储命中 A、B 两页，每页 4 个 token，需要为它们准备 8 个 token slots。分配器可能分配出：

```text
A 页 → slots 128～131
B 页 → slots 256～259
```

Wrapper 和 component 将这些索引放入 `PoolTransfer.device_indices`，后端再根据 Entry 的布局元数据计算实际 GPU 地址并读取数据：

```text
已有 GPU KV 分配器：分配可用 slots
  → Wrapper / component：构造包含目标 slots 的加载任务
  → 后端 linker + Entry：计算对应 GPU 地址
  → Storage：把外部数据读入这些地址
```

“分配 slots”通常是在已有 GPU 内存池中分配使用权，不是每次 GET 都重新申请一块 CUDA 内存。空间不足时，缓存侧可能需要协调回收可淘汰的数据；后端不能自行挑选正在使用的 slots 覆盖。

因此，“通过 cache 准备 GPU slots”是流程简写，更准确的说法是：**Wrapper 通过 cache 和对应 component，协调已有 GPU KV 分配器，为要恢复的缓存页分配目标位置。** 插入树时若发现重复前缀，还会进一步调整实际采用的页面，最终传给后端的是筛选后的加载范围。

`load()` 返回 `True` 代表接受并排队，**不代表数据已可读**。Wrapper 对排队失败会释放本次节点锁并抛错，不能把它理解成自动完成了整棵树的回滚和重算降级。

#### OD 的 load(rid, transfers) 为什么接收 PoolTransfer？

`PoolTransfer` 是传输任务单，把“读哪些页、放到哪些 GPU slots”关联起来。`rid` 标识请求，不能独自描述要搬的数据；`transfers` 提供这份描述。

例如 `page_size=4`，一份 KV transfer 中 `keys=[A,B]`，`device_indices=[128,129,130,131,256,257,258,259]`，就表示 A 页加载到前四个 slots，B 页加载到后四个 slots。这里还没有算好的 GPU 地址，也没有携带 KV payload。参数是列表，因为一个请求可能需要多类逻辑池的任务单。

OD 实现如下：

```python
def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
    self._ensure_open()
    expanded = self.pool_group.resolve_transfers(
        transfers, allow_partial=True, allow_missing_kv=True
    )
    if not expanded:
        return False
    if rid in self.pending_loads:
        raise RuntimeError(f"OceanDisk load already queued for {rid!r}.")
    self.pending_loads[rid] = expanded
    return True
```

| 代码 | 含义 |
| --- | --- |
| `_ensure_open()` | 检查后端没有关闭 |
| `resolve_transfers(...)` | 根据初始化时构建的池关系，展开物理池任务并转换索引；不是分配 slots |
| `allow_partial=True` | 允许只提供部分逻辑来源池的任务 |
| `allow_missing_kv=True` | 不强制要求本次存在非空 KV 任务；树侧去重后可能只剩其他必要数据 |
| `if not expanded` | 没有可执行的物理池任务，返回未接受 |
| 重复 `rid` 检查 | 避免覆盖同一请求尚未启动的加载任务 |
| `pending_loads[rid] = expanded` | 保存任务描述，等待后续启动 |
| `return True` | 表示任务已接受，不代表已读完 |

当前 DSA 路径中，一份逻辑 KV transfer 会展开为 KV、Indexer 两份物理池任务。此时 slots 已由树侧 component 和分配器准备；各池怎样把索引解释为行号与地址，留给后续布局适配。

```text
PoolTransfer：页 A → 已分配的 slots
  → resolve_transfers：展开 KV / Indexer 物理池任务
  → pending_loads：保存任务单
  → start_layer_wise_loading：组织地址并启动 GET
```

#### load_back() 和启动加载分别在调度的什么时候？

它们是组建同一轮模型批次过程中的两个阶段：先逐个处理候选请求，收集加载任务；再统一启动已收集的加载任务。

```text
先做前缀匹配：Wrapper 查询外部命中并保存 hit marker

调度器组建这一轮批次：
  尝试加入请求 A
    → schedule_policy 判断 req.needs_host_load_back()
    → cache.init_load_back(...)
    → 若 Wrapper 有该请求的 hit marker，调用 Wrapper.load_back(A)
    → 分配 slots、处理树节点、保护节点
    → 后端 load(A)：暂存到 pending_loads 字典

  尝试加入请求 B
    → 同样处理，必要时暂存加载任务

  完成本轮请求选择，创建 new_batch
    → scheduler 调用 tree_cache.ready_to_load_host_cache()
    → Wrapper.start_layer_wise_loading()
    → 具体后端取出 pending_loads，开始这一批读取
    → 返回 counter index，保存为本批 hicache_consumer_index
    → new_batch.prepare_for_extend()，后续执行模型前向
```

所以 `load_back()` 不是前缀匹配一发现外部命中就立即执行，而是在调度准备接纳该请求时安排恢复。没有外部命中的请求也能进入模型批次，但不会因此产生 GET。

`pending_loads` 是按请求 ID 收集待启动任务的字典；`load_queue` 是给后台 worker 的工作队列，两者不是同一个东西。启动时通常把多个请求组成一个加载任务放入队列，不是每调用一次 `load()` 就立即入队。若没有待加载任务，OD 和 Mooncake 都返回 `-1`。

| 后端 | start_layer_wise_loading() 的实际启动方式 |
| --- | --- |
| Mooncake | 取出待加载字典，连同 counter index 和 GPU ready event 放入 load_queue，由 worker 发起读取 |
| OD | 取出待加载字典，构造地址并先提交第 0 层 GET，再将 counter index、该字典和后续 continuation 放入 load_queue |

worker 取出任务后继续执行；新的 `pending_loads` 空字典用于接收下一批。两者还有独立的 `offload_queue` 工作队列，以及 `completed_loads`、`offload_results` 结果队列。

源码入口：`managers/schedule_policy.py` 中 `needs_host_load_back()` 附近，`unified_radix_cache.py` 的 `init_load_back()`，以及 `managers/scheduler.py` 中 `ready_to_load_host_cache()` 调用处。host 命名是历史复用，不表示 external linker 必须经过 CPU host cache。

### 5.3 开始逐层读取，模型按层等待

```text
start_layer_wise_loading()
  → 后端把待加载请求组织成一个批次
  → 为该批次创建逐层完成计数，返回 consumer index
  → 后台执行 GET
      第 0 层所需数据全部就绪 → 发布第 0 层完成
      第 1 层所需数据全部就绪 → 发布第 1 层完成
      ...

模型计算到某层
  → 通过已注册的 layer_done_counter 等待该层数据
```

“提交了 GET”“原生调用返回”“GPU 已经能安全读取数据”可能是三个不同时间点。后端必须按实际可用性发布完成，不能仅凭提交成功就把该层标为 ready。

加载失败还必须传给逐层等待者，避免模型使用未初始化的 KV，或永久等待。

### 5.4 消费批次完成通知，释放节点保护

后端将完成的加载批次放入完成队列，`pop_completed_load()` 返回该批次的请求 ID 列表。

缓存树在 `check_hicache_events()` 中协调各 rank 的可消费完成数量，再由 Wrapper 根据请求 ID 找到节点并释放锁引用。

两个机制各有用途：

| 机制 | 保护什么 |
| --- | --- |
| 节点锁引用 | GPU 内存不会在 I/O 期间被回收重用 |
| 逐层完成计数 | 模型不会在该层数据就绪前读取它 |

“树上已经有节点”和“节点里的数据已经可读”不能画等号。

### 5.5 OD PC 实例：从 UnifiedRadixCache 到原生 GET 的完整调用链

下面用当前 OD 的 PC 加载路径串起以上对象。分配、排队、提交和可读是不同阶段。

```text
① 查询：此时尚未执行 GET
UnifiedRadixCache.match_prefix()
  → GPU 缓存树匹配
  → Wrapper.match()
      → 为 GPU 未命中的完整页生成 hash
      → component 构造查询用 PoolTransfer
      → OceanDiskDirectLinker.lookup()
          → Group.resolve_transfers() 展开物理池
          → 编码 object keys
          → OceanDiskStorage.exists_many()
              → 原生 EXISTS
      → 跨 rank 确认共同可恢复边界，保存 hit marker

② 准备恢复位置
UnifiedRadixCache.init_load_back()
  → Wrapper.load_back(req)
      → 消费 hit marker
      → FULL component.build_external_linker_transfer(LOAD, ...)
          → allocator.alloc(num_tokens)：实际分配 GPU slots
          → 填入 PoolTransfer.device_indices
      → 协调各 component 的 PREPARE
      → cache.insert() 插入树并处理重复前缀
      → COMMIT 筛选实际采用的页面和 slots
      → 增加目标节点锁引用，防止传输期间回收

③ 后端接受任务：此时仍未完成读取
Wrapper
  → OceanDiskDirectLinker.load(rid, transfers)
      → Group.resolve_transfers() 展开 KV / Indexer 任务
      → 存入 pending_loads，返回 True

④ 启动实际读取
UnifiedRadixCache.start_layer_wise_loading()
  → Wrapper.start_layer_wise_loading()
  → OceanDiskDirectLinker.start_layer_wise_loading()
      → 创建批次 counter index
      → _get_transfer_layers() 生成逐层 I/O 描述
          → Entry.prepare_locations()：slots 转为各池行号
          → 使用初始化时缓存的 Entry 布局向量化计算地址
          → 得到 key / GPU pointer / size / storage offset
      → OceanDiskStorage.get_layers(..., defer_remaining=True)
          → pointer 转注册内存 MR id 和区域内偏移
          → 选择 connection，打包原生 metadata
          → od_utils.batch_dispatch_get() → GIDS
          → 先提交第 0 层，返回后续工作 continuation
      → 将 continuation 交给 load worker，提交其余层

⑤ 模型逐层等待与完成清理
模型使用某层 KV 前
  → layer_done_counter.wait_until(layer)
      → 等待该层 events 发布，再同步 events
  → 该层 KV / Indexer 可安全使用

load worker 等待读取结束
  → completed_loads 写入请求 ID 批次
UnifiedRadixCache.check_hicache_events()
  → 跨 rank 协调可消费的完成数量
  → Wrapper.drain_loads()
      → 根据请求 ID 找节点，dec_lock_ref() 解除本次保护
```

以页 A 加载到 slots 128～131 为例，目标 slots 是 component 调用已有分配器取得的，**不是 DevicePoolGroup 分配的**。Group 展开物理池任务，Entry 布局用于把已分配的 slots 翻译成各池各层的地址。

当前 GET 的实际地址生成有向量化优化：初始化 `_validate_dsa_layout()` 从 Entry 缓存 `_get_pool_layouts`；每次 `_get_transfer_layers()` 使用行号和缓存布局计算 `base + row × stride`，不必逐页重新创建 Entry 或调用 assembler。

加载失败会向尚未完成的层 Future 传播异常；完成 ID 也用于释放保护，不能单凭收到 ID 推断数据加载成功。节点锁保证内存生命周期，逐层 events 保证模型读取时机。

## 6. 地址到底是谁算的

以 GET 为例：Wrapper 给出“把 A 页读到 slots 128～131”；后端还需要将它转换成实际 GPU 内存位置。

```text
逻辑 PoolTransfer
  → DevicePoolGroup.resolve_transfers()
      按池的来源关系展开为各物理池，并转换索引
  → DevicePoolEntry.prepare_locations()
      校验连续、对齐的页面，并转换为 buffer 行位置
  → DevicePoolEntry.get_prepared_layer_range_meta()
      找到指定层的 buffer，算出指针、长度、布局偏移
  → 后端构造最终 key 和存储 I/O 描述
  → Storage / SDK 执行 GET
```

`DevicePoolEntry` 引用已有 GPU tensor，不另复制一份 KV。它记录 tensor 的设备指针、行步长、页大小、component 组织方式及层映射。

例如某 buffer 按 token 行存放，目标页起始 slot 是 128，行步长是 S 字节，则起始地址是 `buffer_base + 128 × S`。如果 buffer 按页存放，先把 slot 转成页行号；真实计算应使用 Entry 的布局元数据，不能硬套一个公式到所有池。

一份逻辑 KV 任务可能展开为 KV 与 Indexer 等多个物理池：它们关联同一组逻辑页面，但各自有 GPU buffer 和布局。

Entry 给出的是公共布局信息。**后端仍需完成存储格式适配**：例如 OD 的 `_page_fragments()` 会进一步把布局偏移转换成考虑存储对齐的 offset，构造 `OceanDiskRange`。不能把公共 Entry 返回的偏移无条件当成任意新后端的最终磁盘偏移。

这套 Group / Entry 同时被 OD、Mooncake、UMBP 使用。抽象 `UnifiedCacheLinker` 本身并未强制后端使用这些类；它们是当前实现复用的公共工具。

### 6.1 为什么 SGLang 用 slot，而存储 I/O 还需要 GPU 地址？

SGLang 的 KV 分配器用 slot 索引管理存储位置，数据实际放在 GPU tensor 中。slot 是位置编号，GPU 指针是具体内存地址，两者用途不同。

类比数组访问 `buffer[128]`：128 是索引，要访问这行数据仍需结合 buffer 的起始地址和行步长计算地址。平时用 tensor 索引或给 attention kernel 传 slots，框架或 kernel 会在内部完成寻址。

同一个 slot 还可能在多个模型层各有一份 KV：

```text
slot 128
  ├─ 第 0 层：buffer_0 的对应行
  ├─ 第 1 层：buffer_1 的对应行
  └─ 第 2 层：buffer_2 的对应行
```

这些 buffer 的地址不同，因此只给 slot 128 不能确定一个唯一的 GPU 地址。还要知道具体池、层和存储布局。

`PoolTransfer.device_indices` 只提供 slots，**其中还没有算好的 GPU 指针**。Entry 保存 buffer 的 `data_ptr()`、行步长和映射关系，先将 slots 转成布局对应的行号，再用 `base + row * stride` 计算地址。

例如按 token 行存储时：

```text
第 0 层 buffer 起始地址 = 0x100000
每行步长               = 256 字节
目标页起始 slot        = 128
目标 GPU 地址          = 0x100000 + 128 × 256 = 0x108000
```

按页存储的 buffer 则先把 token slot 转为页行号，不能直接使用上面的 token 行公式。

OD 和当前 Mooncake 的范围读取接口接收 GPU 指针、字节数和对象偏移。它们不能仅凭 slot 编号确定读写位置，因此具体 linker 需要把索引翻译成这些 I/O 参数。这个转换只是计算位置和描述信息，不搬运 KV 数据，也不必然改变数据格式。

### 6.2 当前 OD 和 Mooncake 的 DSA 路径实际用哪种布局？

**当前 DSA 路径中，两种布局同时存在：KV 池按 token 行存储，Indexer 池按 page 行存储。OD 和 Mooncake 在这条路径上相同。**

| 后端（DSA 路径） | KV 池 | Indexer 池 |
| --- | --- | --- |
| `OceanDiskDirectLinker` | `rows_are_pages=False`，token 行 | `rows_are_pages=True`，page 行 |
| `MooncakeDirectLinker` | `rows_are_pages=False`，token 行 | `rows_are_pages=True`，page 行 |

原因是两个后端都调用公共的 `resolve_hybrid_device_pool_group()`；对于 DSA GPU pool，公共策略调用 `_build_dsa_device_pool_group()`，其中明确设置：

```python
# 简化摘录：省略其他构造参数
DevicePoolEntry(
    name=PoolName.KV,
    components=[...],  # kvcache.kv_buffer 及可能附带的 draft buffers
    rows_are_pages=False,
)
DevicePoolEntry(
    name=PoolName.INDEXER,
    components=[...],  # kvcache.index_k_with_scale_buffer 及 draft buffers
    rows_are_pages=True,
)
```

例如 `page_size=4`，同一页对应 slots `[128,129,130,131]`：

```text
KV 池：
  页起始行号 = 128
  GPU 起始地址 = 该层 KV buffer base + 128 × KV token 行步长
  这一页的数据覆盖连续 4 个 token 行

Indexer 池：
  页起始行号 = 128 ÷ 4 = 32
  GPU 起始地址 = 该层 Indexer buffer base + 32 × Indexer page 行步长
  这一行已包含一整页的 Indexer 数据
```

两个池的 base、行步长和页字节数各自独立；共享逻辑 slots 不表示共享同一个 GPU 地址。它们都按页传输，只是 GPU buffer 的行组织不同。

**token 行还是 page 行，由底层 GPU buffer 布局决定，不是 OD 或 Mooncake 的存储协议决定的。** 后端先适配已有 GPU 布局，再按自己的存储格式组织对象、对齐和 I/O。

这张表限定当前 DSA 路径，不能推广成“所有模型的 KV 都按 token 行”。例如同一 assembler 中的 DeepSeek V4 构建路径使用 page 行视图；公共池工具能够构建这种视图，也不代表当前 OD 的布局校验接受它。

核对位置：

- `linker_pool_assembler.py`：`_build_dsa_device_pool_group()` 中 KV / Indexer 的 `rows_are_pages` 设置。
- `oceandisk_direct_linker.py`：构造函数调用 `resolve_hybrid_device_pool_group()`。
- `mooncake_direct_linker.py`：构造函数调用同一个公共函数。

### 6.3 这是 UC Linker 的公共逻辑吗？Mooncake 也需要吗？

当前 UC Linker 路径中的多个后端复用了 Entry 的地址计算能力，但要区分接口约定、公共工具和后端调用：

| 层 | 是否负责算地址 |
| --- | --- |
| `UnifiedCacheLinker` 抽象接口 | 定义 `load()` 等传输契约，本身不实现地址计算 |
| 公共 `DevicePoolEntry` | 根据 slot、层号和 buffer 布局计算 GPU 指针、长度及布局偏移 |
| OD / Mooncake 具体后端 | 调用 Entry，并把结果适配为各自存储接口的参数 |

两条现有加载路径可以对照来看：

```text
OD：
PoolTransfer 的 slots
  → Entry.get_prepared_layer_range_meta()
  → GPU 指针、长度、布局偏移
  → OD _page_fragments() 适配为 OceanDiskRange
  → Storage 执行 GET

Mooncake：
PoolTransfer 的 slots
  → Entry.get_prepared_layer_range_meta()
  → GPU 指针、长度、布局偏移
  → batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
  → 执行读取
```

Mooncake 的实际调用可在 `mooncake_direct_linker.py` 的 `load_layer_wise()` 中看到。它也需要这一步，不是只有 OD 才做 slot 到地址的转换。

**slot 最终需要对应到实际内存位置，是读写 GPU 数据所必需的；使用 Entry 来完成转换，是当前多个后端复用的实现方式。** 新后端可以复用 Entry，也可以用自己的适配方式，只要正确读写上层指定的 slots，并满足传输和同步契约。

## 7. 一次保存与 PUT 的流程

通用的树侧保存流程是：

```text
缓存树产生 BackupKV action
  → Wrapper.offload_nodes()
      → tree core 构造节点的 PoolTransfer
      → 给源节点增加锁引用
      → 后端 offload(transfers)
      → 接受成功后记录 pending offload
  → 后端确保所有必要数据已成功保存
  → 按提交顺序提供完成结果
  → 树侧跨 rank 汇总成功状态
  → Wrapper 提交节点的外部存储状态，再释放锁
```

`offload()` 返回 `True` 也只是接受任务。只有完成结果为成功，树才可以确认外部副本有效；失败时不能发布“已保存”的状态。

offload 完成接口只返回布尔值，不携带任务 ID；Wrapper 按 pending 列表头部匹配。因此内部即使并行乱序完成，也必须按接受任务的顺序向 Wrapper 交付结果，且不能丢失或重复结果。

这里的原子持久化是接口期望的**逻辑整体成功**：一次任务涉及的必要池、页和层必须全部成功，才可确认。它不等于公共层已经替存储系统提供了事务或掉电恢复；新后端需要自行保证不完整对象不会被 lookup 当作完整命中，例如使用适合其存储系统的完整性发布机制。

## 8. 哪些是 OD 的扩展

### 8.1 逐层保存

OD 支持前向过程中逐层保存，涉及额外的 session、receipt、GPU ready event 和源页面 pin / lease。

```text
前向前准备 session、页面和 hash
  → 每层产出 KV 后触发保存
  → 后台 PUT 与前向重叠
  → 收集保存 receipt
  → 树侧 offload 检查对应 receipt 并确认结果
```

**OD 开启逐层保存时，树侧 `offload()` 主要确认已经发起的保存，不重新执行整页 PUT，也不会在逐层 PUT 失败后自动补做 PUT 重试。** 非逐层模式则有普通的传输任务路径。

这不是 `UnifiedCacheLinker` ABC 要求所有后端实现的能力。新增后端可以先实现普通 offload；若要支持逐层保存，还需要接入模型层回调、GPU 数据就绪同步、source pin、失败清理及图捕获/回放等流程。当前相关接线存在 OD 专用条件，不能只添加一个同名方法就认为已经接通。

### 8.2 PD 交接

`OceanDiskPDLinker` 在 DirectLinker 上增加 P/D namespace、后台加载及 handoff 配合。普通前缀缓存后端不需要因此实现 PD；如果目标包括 P/D 解耦交接，需要另外审查 `pd_handoff.py`、`pd_linker.py` 和 PD 参数校验。

## 9. 新增类似 ODLinker 的后端：需要做什么

### 9.1 先确定后端支持范围和对象格式

实现前先回答：

- 首版支持哪些模型、tree components、物理池和 GPU 布局？遇到其他布局应在初始化时明确报错。
- 对象按页保存、按层保存，还是一页对象内包含多层？是否支持范围读写？
- key 如何区分模型、数据格式、物理池以及需要区分的 TP / CP / PP 分片？slots 会复用，不能作为缓存内容身份。
- 副本布局下哪些 rank 写入、哪些 rank 查询或读取？不要直接照搬 OD 的特定 rank 规则。
- 部分页或部分层保存失败时，如何避免被误判为完整命中？
- SDK 是否支持 GPU 指针？是否需要注册内存？如果采用 staging，如何同步和管理 staging buffer？不能默认任意存储 SDK 都能直接读写 GPU。

这些是新后端的设计工作，公共 Wrapper 不会代为决定。

### 9.2 实现具体 linker 类及存储适配

建议新增：

```text
python/sglang/srt/mem_cache/storage/mybackend/
  my_direct_linker.py   # MyDirectLinker(UnifiedCacheLinker)
  my_store.py           # MyStorage，封装 SDK 和具体 I/O
```

文件拆分是建议；硬性要求是实现 `UnifiedCacheLinker` 的全部抽象方法，并提供兼容的 `layer_done_counter`。

| 接口 | 实现要求 |
| --- | --- |
| `lookup(rid, transfers) -> list[int]` | 返回本 rank 所有可完整恢复的前缀页数边界；满足各池 hit policy |
| `load(rid, transfers) -> bool` | 接受并排队，目标 slots 由上层提供；支持树侧筛选后的部分传输 |
| `start_layer_wise_loading() -> int` | 启动待加载批次并返回 counter consumer index；现有实现无待加载时返回 `-1` |
| `cancel_queued_load(rid) -> bool` | 只有确实安全取消才返回 True；见下文限制 |
| `num_completed_loads() -> int` | 当前可消费的完成批次数，不是请求个数 |
| `pop_completed_load() -> list[str]` | 消费最早完成批次，返回其请求 ID，用于树侧解锁 |
| `offload(transfers) -> bool` | 接受保存任务，保证后续有对应完成结果；不代表已持久化 |
| `num_completed_offloads() -> int` | 当前可消费的保存完成结果数 |
| `pop_completed_offload() -> bool` | 按接受顺序消费一个结果；必要数据全部保存成功才返回 True |
| `reset()` | 返回前停止/排空在途 I/O，确保不会再访问随后被释放的 GPU 页面，并重置状态 |
| `close()` | 返回前确保 I/O 静止，再释放线程、连接和注册内存等资源 |

逐层 counter 的兼容性应参考现有 `LayerWiseLoadCounter` 及消费者调用；仅放一个整数不够。至少需要衔接批次 consumer 选择、按层等待以及失败传播。

**取消限制：** Wrapper 在排队前已经发布树节点，当前 `release_request()` 没有完整的原子树回滚流程。Mooncake 的 `cancel_queued_load()` 因此直接返回 False。新后端不要仅从 Python 队列删除任务就返回 True，否则可能保留指向未加载数据的树节点；基础版本可保守返回 False，让既有任务完成并正常清理。

### 9.3 初始化物理池与运行资源

可以沿用当前后端的构造参数：

```python
def __init__(self, server_args, params, *, components):
    ...
```

典型准备步骤：

1. 从 `params.token_to_kv_pool_allocator.get_kvcache()` 获取已有 GPU KV pool。
2. 调用 `resolve_hybrid_device_pool_group(...)` 构建 pool group，保存 `entry_map` 和层数。
3. 校验当前后端支持的池、布局和层映射。
4. 解析连接、namespace、并行拓扑及写入 owner 配置。
5. 按 SDK 要求注册 GPU 内存；共享底层 allocation 的 tensor 要避免重复注册。
6. 创建逐层 counter、待加载集合、任务队列、完成队列及 worker。
7. 补齐初始化失败时的资源清理。

若现有物理池工具已经支持目标模型且符合存储格式，通常复用即可；只有需要新增 GPU 布局或池映射时，才扩展 `linker_pool_assembler.py` 及 `hybrid_pool_assembler.py` 中对应 strategy 的 `build_direct_linker_pool_group()`。

### 9.4 接入 backend 选择与配置

| 文件 | 需要做的改动 |
| --- | --- |
| `python/sglang/srt/mem_cache/registry.py` | 在后端选择中增加 `mybackend` 分支，延迟导入并选择 `MyDirectLinker` |
| `python/sglang/srt/arg_groups/fields/memory.py` | 在 `unified_cache_external_linker_backend` 的 choices 加入 `mybackend` |
| 新后端配置代码 | 定义连接地址、认证、容量、并发等所需参数及解析；如复用 extra config，明确支持的字段 |
| 依赖声明及运行文档 | 声明 SDK / 原生库、安装和运行要求，保持未启用后端时不强制加载 SDK |

完成上述接线后，预期选择方式为：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend mybackend
```

这里不是 `--hicache-storage-backend mybackend`。当前参数校验要求 external linker 与 `--enable-hierarchical-cache` 互斥，也不接受同时指定 `--hicache-storage-backend`。

仅支持 UC direct-linker 路径时，不需要为了选择后端再接一套 HiCache storage factory，也不需要默认实现整个 `HiCacheStorage` 接口。是否继承/复用现有 storage 类，取决于希望兼容的其他路径。

### 9.5 基础版本通常不需要改什么

只要沿用现有契约、支持已有组件和布局，通常可以复用：

- `UnifiedRadixCache` 的树操作、外部命中入口和完成事件轮询。
- `UnifiedCacheLinkerWrapper` 的 hash、跨 rank 命中边界协调、节点锁与状态提交。
- FULL / SWA component 的传输构造和加载阶段协作。
- 已有 Group / Entry 的物理池与地址映射。

只有需要新的树状态、组件类型或额外能力时，再扩展公共层。普通后端不应把存储 SDK、key 编码和连接选择塞进缓存树。

## 10. 必须验证的行为

下面是新增后端需要覆盖的验证建议，不是已经替 `mybackend` 执行过的测试。

| 场景 | 要证明的行为 |
| --- | --- |
| 多池命中、前缀中间缺页、尾部窗口 | lookup 只返回完整可恢复边界；任何必要池缺失都不会误命中 |
| 不同 rank 的候选集合不一致 | 树侧选择共同有效边界，不只是最小最大值 |
| 多页、多层、Indexer 等侧池 | GET/PUT 使用正确 key、GPU pointer、size 和存储 offset |
| 插入时出现重复前缀 | 只加载实际采用的页；侧池单独剩余时也能处理 |
| I/O 很慢、GPU 异步生产数据 | 源/目标页面在传输完成前不被复用；正确等待生产和消费依赖 |
| 某层 GET 失败 | 等待者收到错误，未完成层不会被发布 ready，批次清理不会遗漏 |
| 某池 PUT 失败、内部乱序完成 | 不确认部分成功；对 Wrapper 按提交顺序输出结果 |
| 请求结束、reset、close | 没有悬空 GPU 访问、重复完成通知、节点锁泄漏或无法结束的线程 |
| GPU slots 被重用、跨进程再次查询 | 相同内容仍能命中，旧 slot 编号不会污染对象身份 |

可参考的现有测试：

- `test/registered/unit/mem_cache/test_unified_cache_linker.py`：Wrapper 生命周期、节点锁、重复范围、跨 rank 完成提交。
- `test/registered/unit/mem_cache/test_linker_pool_assembler.py`：物理池展开和布局映射。
- `test/registered/unit/mem_cache/test_oceandisk_direct_linker.py`：OD 地址、逐层 I/O、失败处理和扩展能力。
- `test/registered/radix_cache/unified_radix_tree/linker/`：现有真实模型集成测试。

建议先用 fake storage 验证任务和生命周期，再用真实 GPU / 存储跑“生成并保存前缀 → 清除 GPU 驻留副本 → 再请求相同前缀”的集成场景，检查外部命中和模型输出。真实环境应与声明支持的模型、布局及并行拓扑一致。

## 11. 按这个顺序读源码

| 顺序 | 文件 | 重点 |
| --- | --- | --- |
| 1 | `python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py` | 先读 ABC，再看 Wrapper 的 match、load_back、offload 和清理 |
| 2 | `python/sglang/srt/mem_cache/unified_radix_cache.py` | init_cache_linker、match_prefix、init_load_back、check_hicache_events |
| 3 | `python/sglang/srt/mem_cache/registry.py` | 后端实例化与 counter 注册 |
| 4 | `python/sglang/srt/mem_cache/hicache_storage.py` | PoolTransfer、PoolName、PoolHitPolicy |
| 5 | `python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py` | 逻辑任务如何变成物理地址 |
| 6 | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py` | 观察普通后端如何履行接口；其 counter 和错误文案带 Mooncake 语义，复制前需审查 |
| 7 | `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_direct_linker.py` | 观察 OD 的范围 I/O 和逐层保存扩展 |

阅读 OD 细节时，可结合同目录层级的已有文档：

- [OceanDisk 与 Unified Cache Linker 关系](oceandisk_linker_summary.md)
- [OceanDisk PUT / GET / EXISTS 细节](oceandisk_put_get_exists_details.md)
- [OceanDisk PD 设计与流程](oceandisk_pd_design_and_flow.md)

## 12. 后端范例：OceanDiskDirectLinker 与 OceanDiskStorage 如何配合

前面的章节说明公共契约，这里用已有 OD 后端说明如何把它落到具体代码。两个主要文件是：

- `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_direct_linker.py`
- `python/sglang/srt/mem_cache/storage/oceandisk/oceandisk_storage.py`

**DirectLinker 把 UC Linker 的逻辑传输任务转换成 OceanDisk I/O；Storage 把这些 I/O 提交给 OceanDisk 原生接口。** 可以理解成 DirectLinker 组织任务，Storage 执行读写和底层同步。

### 12.1 两个文件各自负责什么

| 对比 | `OceanDiskDirectLinker` | `OceanDiskStorage` |
| --- | --- | --- |
| 面向谁 | 上层 Wrapper / `UnifiedCacheLinker` 契约 | OceanDisk SDK、GIDS 原生接口 |
| 接收什么 | `PoolTransfer`：页 hash、池名、GPU slots 等 | object key、GPU 地址、字节数、存储偏移等 I/O 描述 |
| 任务转换 | 展开物理池、结合 Entry 计算地址、编码 key、适配存储布局、组织逐层传输 | 把 I/O 描述打包成原生接口需要的 metadata，处理注册内存描述和 connection 路由 |
| 状态管理 | 待加载请求、后台任务、逐层完成计数、完成队列、保存 receipts | 原生调用状态、CUDA stream/event、I/O metadata 的准备与释放 |
| 生命周期 | 停止和排空传输任务，协调 worker 与 Storage 关闭 | 释放存储连接、注册内存及相关底层资源 |
| 与缓存树的关系 | 通过接口向 Wrapper 报告结果，不负责直接操作树节点 | 不管理缓存树和节点锁 |

Storage 不是直接接收 `req` 或树节点来决定“应该缓存什么”；DirectLinker 也不直接完成所有 SDK、内存注册和原生 metadata 操作。

### 12.2 用一次 GET 串起分工

假设 `page_size=4`，外部页 A 需要读回 slots 128～131。下面展示职责顺序，实际实现还会批处理多个请求、页和层。

```text
Wrapper：
  通过 cache / component 分配并选定目标 slots
  保护目标树节点
  构造 PoolTransfer(keys=[A], device_indices=[128,129,130,131])
       ↓ load(rid, transfers)

OceanDiskDirectLinker：
  接受加载任务，暂存到待加载集合
       ↓ 后续 start_layer_wise_loading()
  1. 通过 Group 把逻辑 KV 任务展开成 KV、Indexer 物理池任务
  2. 按各池布局得到各层 GPU 指针和大小
  3. 把页 hash 编码成 OceanDisk object key
  4. 适配存储对齐与 offset，组织逐层 I/O 描述
       ↓ storage.get_layers(...)

OceanDiskStorage：
  1. 按 connection 规则路由请求
  2. 将 GPU 地址适配为原生接口需要的注册内存描述
  3. 打包、准备并传递 I/O metadata
  4. 提交原生 GET，提供对应的完成事件
       ↓

OceanDiskDirectLinker 的逐层计数：
  接入完成事件；模型等待该层时同步相应事件
  加载批次结束后提供请求 ID 完成通知
       ↓

Wrapper / cache：
  协调并消费完成通知，释放目标节点的锁引用
```

DSA 路径的同一组 slots 在两个池中的寻址不同：KV 使用第 128 个 token 行，Indexer 使用第 32 个 page 行。它们的 GPU 地址由各自 buffer 决定，相关规则见第 6.2 节。

OD 的 `LayerWiseLoadCounter` 可以先收到可等待的 event；这不等于 I/O 已完成。模型消费侧的 `wait_until()` 会同步相应事件，之后才可使用该层数据。不能把“Future 已有结果”直接理解为“GPU 数据已就绪”。

### 12.3 为什么两层代码都涉及地址

这里有两次不同的适配：

```text
slot + 池 + 层 + GPU 布局
  → DirectLinker / Entry：确定实际 GPU 指针与数据大小

GPU 指针 + object key + 存储 offset
  → Storage：确定注册内存区域、区域内偏移、connection 和原生参数
```

例如 Entry 算出 GPU 地址 X，Storage 还要确定 X 属于哪个已注册内存区域，以及如何按原生 ABI 描述它。因此 Storage 处理地址并不表示它重复决定目标 slot，也不表示 slot 分配属于 Storage。

存储侧的范围 offset 也有分工：DirectLinker 结合页面格式和对齐规则构造读写范围，Storage 将范围转成原生接口能执行的描述。公共 Entry 的布局偏移不能绕过后端格式适配直接用于所有存储系统。

### 12.4 PUT 有普通保存和逐层保存两条路径

普通保存模式下：

```text
树产生 BackupKV action
  → Wrapper 构造 transfer、保护源节点
  → DirectLinker.offload() 接受任务
  → worker 组织各层 I/O，调用 Storage.put_layers()
  → Storage 执行 PUT 并等待所需完成条件
  → DirectLinker 按序输出保存结果
  → Wrapper / cache 汇总各 rank 结果，提交存储状态并解锁
```

开启 OD 逐层保存时，PUT 的发起时机提前到模型前向期间：

```text
前向前准备 LayerwiseSaveSession 和源页面保护
  → 每层产出 KV，记录 ready event 并触发保存
  → DirectLinker worker 调用 Storage.prepare_layerwise_put()
    / submit_put_layer() 等接口
  → Storage 等待 GPU 生产依赖并提交 PUT
  → 收集最终保存结果和 receipts
  → 后续树侧 offload() 核对对应 receipts
  → 提交存储状态并释放相关保护
```

开启逐层保存时，后续树侧 `offload()` 主要确认已有保存结果，**不重新执行一遍整页 PUT，也不自动重试失败的逐层 PUT**。DirectLinker 管 session、receipt 和任务生命周期；Storage 管 metadata、原生提交与事件。

### 12.5 对照源码时看哪些方法

| 文件 | 方法 / 类 | 阅读目的 |
| --- | --- | --- |
| `oceandisk_direct_linker.py` | `lookup()`、`_lookup_local_prefix()` | 逻辑查询如何转换为 `storage.exists_many()` |
| `oceandisk_direct_linker.py` | `load()`、`start_layer_wise_loading()`、`_load_worker()` | 接受任务、启动读取与处理后台加载 |
| `oceandisk_direct_linker.py` | `_page_fragments()`、`_get_transfer_layers()`、`_transfer_layers()` | 从逻辑页、slots 和布局组织物理 I/O |
| `oceandisk_direct_linker.py` | `LayerWiseLoadCounter` | 如何把 I/O event 接到模型逐层等待 |
| `oceandisk_direct_linker.py` | `offload()`、`_offload_worker()` | 普通保存与 receipt 确认如何处理 |
| `oceandisk_direct_linker.py` | `LayerwiseSaveSession`、`begin_layerwise_save()` | OD 逐层保存扩展的组织方式 |
| `oceandisk_storage.py` | `exists_many()`、`get_layers()`、`put_layers()` | 实际存在性查询、范围读写与原生提交 |
| `oceandisk_storage.py` | `_build_mr_index()`、`_pack_layer_batches()`、`_pack_get_matrix()` | 注册内存描述与原生 metadata 打包 |
| `oceandisk_storage.py` | `prepare_layerwise_put()`、`submit_put_layer()` | 逐层 PUT 的 metadata 与提交 |

### 12.6 新后端如何借鉴这个拆分

新增 `MyDirectLinker` 时，可以参照上述边界：

- **放进 MyDirectLinker**：履行 UC Linker 接口、处理 PoolTransfer、复用池映射、定义对象格式、组织逐层加载和完成通知。
- **放进 MyStorage**：初始化 SDK、注册内存、管理连接、适配原生参数、执行查询和读写、返回明确的完成或失败信息。
- **继续留在公共树侧**：分配与采用哪些 slots、树节点的发布与存储状态、节点锁、跨 rank 的结果提交。

先把普通查询、加载、保存及失败清理接通，再按需求增加逐层保存或 PD 交接。OD 的 class 名称、GPU 对齐要求、特定 rank 策略和原生 ABI 都需要按新存储系统重新审查，不能整体复制后仅替换 SDK 调用。

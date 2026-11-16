# 代码走读笔记

## 1. registry 如何创建 Cache、ODLinker 和 Wrapper

范围：OceanDisk PC 路径，暂不展开 PD 分支。

图里的 **registry 不是函数名，是 `registry.py` 文件**。具体执行初始化的是：

- 文件：`python/sglang/srt/mem_cache/registry.py`
- 函数：`_create_unified_radix_cache()`，当前位于第 150 行。

以下行号按本次走读时的工作区代码记录，代码变动后以函数名为准。

### 第一步：创建缓存树

第 191 行：

```python
cache = UnifiedRadixCache(params)
```

此时有了 Cache，但还没有挂接 OD 后端。

### 第二步：检查 external linker 开关和后端配置

第 200 行开始：

```python
elif get_memory().enable_unified_cache_external_linker:
    backend = get_memory().unified_cache_external_linker_backend
```

这里读取开关和后端名称，对应启动配置：

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend oceandisk
```

### 第三步：选择 OD 后端类

第 208 行开始，省略导入完整路径：

```python
elif backend == "oceandisk":
    from ...oceandisk_direct_linker import OceanDiskDirectLinker

    linker_cls = OceanDiskDirectLinker
```

**这里只是选定类，还没有创建 OD 对象。** PC 场景先跳过后面的 PD 选择分支。

### 第四步：先创建 OD，再挂接到 Cache

第 231 行：

```python
cache.init_cache_linker(
    linker_cls(server_args, params, components=set(cache.components))
)
```

这句按“先里面、后外面”执行，等价于：

```python
od = OceanDiskDirectLinker(
    server_args,
    params,
    components=set(cache.components),
)
cache.init_cache_linker(od)
```

先进入 `OceanDiskDirectLinker.__init__()`，创建 Group / Entry、Storage、counter 和 worker；构造完成后，才将 OD 实例传给 Cache。

随后进入 `python/sglang/srt/mem_cache/unified_radix_cache.py` 第 369 行的函数：

```python
def init_cache_linker(self, cache_linker: UnifiedCacheLinker) -> None:
    self.linker = UnifiedCacheLinkerWrapper(self, cache_linker)
```

**Wrapper 就在这里创建。** 它保存原 Cache 和传入的 OD 实例，最终关系是：

```text
cache
  └─ linker → Wrapper
                 ├─ cache → 原来的 cache
                 └─ cache_linker → od
```

所以：

```python
cache.linker                # Wrapper
cache.linker.cache_linker   # OD 实例
```

所谓“注册 ODLinker”，在这条路径中就是选择类、创建对象，再通过 Wrapper 保存引用，没有单独的 `register_odlinker()`。

### 第五步：接通模型侧的逐层加载等待

回到 `registry.py` 第 247 行，跳过 PC 不使用的 handoff 分支：

```python
counter = cache.linker.layer_done_counter
kvcache = params.token_to_kv_pool_allocator.get_kvcache()

kvcache.register_layer_transfer_counter(counter)
ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
```

Wrapper 的 `layer_done_counter` 属性返回 OD 后端的 counter。这几句把同一个 counter 交给 GPU KV pool 和 TP worker，后续模型才能在使用某层 KV 前等待该层 GET。

**注册 counter 与挂接 OD 对象是两件事。** 此处没有再次创建 OD，也没有直接执行 GET。

### 第六步：接通逐层保存钩子，返回 Cache

函数末尾检查逐层保存开关和模型支持条件：

```python
if not decode_linker_load and getattr(
    cache.linker.cache_linker, "layerwise_save_enabled", False
):
    model_bases = type(ctx.tp_worker.model_runner.model).__mro__
    if any(cls.__name__ == "DeepseekV2ForCausalLM" for cls in model_bases):
        ctx.tp_worker.layerwise_cache_linker = cache.linker
return cache
```

在当前 PC 路径，`decode_linker_load` 为 False。满足保存开关和模型条件时，将 Wrapper 交给 TP worker，供前向前准备逐层保存 session。

### 建议断点与跳转顺序

```text
registry.py：_create_unified_radix_cache()
  → 选择 linker_cls
  → 第 231 行：先进入 OceanDiskDirectLinker.__init__()
  → 返回后进入 UnifiedRadixCache.init_cache_linker()
      → UnifiedCacheLinkerWrapper.__init__()
  → 返回 registry：注册 counter
  → 接保存钩子，返回 cache
```

第 231 行是本段走读的关键交汇点：**先创建 OD，后创建 Wrapper 并挂到 Cache。**

相关说明：

- [OD PC 架构与流程](oceandisk_pc_architecture_walkthrough.md)
- [UnifiedCacheLinker 运行与扩展指南](unified_cache_linker_guide.md)

## 2. attention 取 KV 时，在哪里等待 OceanDisk GET？

本段沿当前工作区的 DSA / MLA KV 访问路径走读。attention 有多个执行分支，下面选 `dsa_backend.py` 中的一个实际取 buffer 位置举例；不是声称所有 attention 分支都执行同一行代码。

### 第一步：初始化时，共用同一个 counter 对象

`registry.py` 第 247 行开始：

```python
counter = cache.linker.layer_done_counter
kvcache = params.token_to_kv_pool_allocator.get_kvcache()
kvcache.register_layer_transfer_counter(counter)
ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
```

这不是复制两个计数器，而是保存同一个对象的引用。`memory_pool.py` 中 `KVCache.register_layer_transfer_counter()`（第 1917 行）和 `tp_worker.py` 中的注册方法（第 532 行）都是将参数保存为成员。

两处用途不同：TP worker 选择“等待哪批加载”，pool 在返回 buffer 前等待“这批的哪一层”。

### 第二步：启动加载时生成批次编号，模型前向前选择这个编号

OD 的 `start_layer_wise_loading()` 调用 `layer_done_counter.update_producer()`，创建本批逐层 Futures，返回 index。调度器将 index 保存到 `new_batch.hicache_consumer_index`（`scheduler.py` 第 4034 行附近）。这个 index 是加载批次编号，不是请求 ID，也不是层号。

随后进入 `tp_worker.py` 的 `forward_batch_generation()`，第 606 行在创建 ForwardBatch 前执行：

```python
self.set_hicache_consumer(batch.hicache_consumer_index)
```

同文件第 535 行：

```python
def set_hicache_consumer(self, consumer_index: int):
    if self.hicache_layer_transfer_counter is not None:
        self.hicache_layer_transfer_counter.set_consumer(consumer_index)
```

OD counter 的 `set_consumer()` 保存这个编号。假设编号是 7，就是告诉后续 pool 访问：“检查加载批次 7 的状态。”

### 第三步：attention 访问这一层 KV，调用 pool getter

`layers/attention/dsa_backend.py` 第 2236 行有实际调用：

```python
kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
```

返回的 `kv_cache` 将供后续 attention 分支使用。DSA pool 继承 MLA pool；对应的基础 getter 在 `memory_pool.py` 的 `MLATokenToKVPool.get_key_buffer()`，第 4416 行：

```python
def get_key_buffer(self, layer_id: int):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    if self.store_dtype != self.dtype:
        return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

    return self.kv_buffer[layer_id - self.start_layer]
```

关键顺序是 **先 wait，后 return buffer**。这里是 attention 准备读取 KV 时等待，不是算完 attention 再等。减去 `start_layer` 是将模型层号映射为当前 pool 的本地层索引。

### 第四步：OD counter 等待 Future，再同步 GET events

`storage/oceandisk/oceandisk_direct_linker.py` 中 `LayerWiseLoadCounter.wait_until()`，第 119 行，核心代码：

```python
index = self.consumer_index
futures = self.futures.get(index)
if futures is None or futures[threshold] is None:
    return

for event in futures[threshold].result():
    event.synchronize()
futures[threshold] = None
```

假设 consumer index 是 7、threshold 是 0，这里等的是“加载批次 7 的第 0 层”。

- `Future.result()` 等待该层的 GET events 被发布；失败时会传播异常。
- `event.synchronize()` 等待相应 I/O 完成；Future 已有结果不等于数据已可用。
- 成功后将这一层标为已等待，重复 getter 访问可以直接继续。

这些 events 来自 `start_layer_wise_loading()` 传给 Storage 的 `on_layer_submitted` 回调，回调调用 `counter.complete(index, layer, events)`。这一层包含配对的 KV / Indexer I/O，等待机制不是只等待一个随意的 buffer。

#### “counter 真正等待 GET 事件”是什么意思？

**程序在这里等 OceanDisk 把这一层的数据写进 GPU，写完以后 attention 才继续读取。** 将核心代码拆开看：

```python
events = futures[threshold].result()
for event in events:
    event.synchronize()
```

这里有两次等待，含义不同：

1. **`result()`：等拿到“完成信号”的对象。** GET 提交后，Storage 通过回调把对应 CUDA events 交给 counter。拿到 event，只表示有东西可以等，不代表数据已经读完。若 Future 已发布异常，这里会抛出异常。
2. **`event.synchronize()`：等这个信号真正完成。** 事件完成后，才能确认对应 GET 已完成，GPU 数据可以使用。

```text
提交 GET：安排把数据读到 GPU
    ↓
拿到 event：拿到这次读取的“完成凭证”，但可能还在读
    ↓
event.synchronize()：如果没读完，就在这里等
    ↓
读取完成：等待返回，attention 可以使用 KV
```

**counter 自己不负责读数据，它负责让使用数据的一方等到读取完成。** 如果 Future 已发布且 GET 早已完成，这里的等待就会很快返回。

### 第五步：getter 返回，attention 才继续使用 KV

```text
OD 启动加载，生成 index=7
  → 调度器将 index 保存到 batch
  → TP worker：counter.set_consumer(7)
  → attention：pool.get_key_buffer(第 N 层)
      → counter.wait_until(本地层号 N)
          → 等 Future 发布 events
          → 同步 GET events
      → 返回该层 GPU buffer
  → attention 继续读取 KV、执行计算
```

逐层等待允许模型在第 0 层可用后计算第 0 层，而后续层仍可继续加载；不要求先等完所有层。具体重叠程度由实际执行路径和 I/O 决定。

补充边界：不同 pool、attention 后端以及图执行路径还可能有其他等待入口。例如 Indexer buffer 访问会进入其自己的 cache 封装，不能把上面的 `get_key_buffer()` 说成所有 buffer 唯一的等待入口；走读某个运行配置时，需要继续跟踪该配置实际调用的 getter 和层钩子。

## 3. GET 全流程走读：加载线程与计算侧分别等什么

本节串起当前 OD PC 的正常 GET 路径。这里的“计算侧”指执行 Python 模型前向、准备 GPU 计算的 CPU 执行线程；“加载线程”指 OD 单独创建的 `_load_thread`。GPU 计算和原生 I/O 则是它们提交的设备工作。不同调度/overlap 模式可能改变前向的具体承载线程，但下面两个等待入口和职责不变。

### 3.1 先看全局时序

```text
调度侧准备请求
  → 匹配 GPU / 外部缓存
  → Wrapper.load_back()：分配 slots、插入树、保护节点
  → OD.load()：将物理池任务暂存到 pending_loads，尚未 GET

调度侧启动本批加载
  → OD.start_layer_wise_loading()
      → 生成 counter index
      → 根据本次 slots 构造各层 GPU 地址与 I/O 描述
      → Storage.get_layers(..., defer_remaining=True)
          → 提交第 0 层 GET，并发布第 0 层 events
          → 返回 finish 函数，作为 continuation
      → (index, pending, continuation) 放入 load_queue
      → 返回 index，关联到模型 batch

此后有两条执行线：

加载线程 _load_worker()                计算侧模型前向
  load_queue.get()                     TP worker：set_consumer(index)
    ↓                                    ↓
  continuation()                       第 0 层 pool getter
    → 提交第 1～末层 GET                   → 等第 0 层 events 完成
    → 发布各层 events                      → 返回 buffer，继续计算
    → 等末层 events 完成                第 1 层 pool getter
    → 完成本批收尾                         → 等第 1 层 events 完成
    ↓                                      → 返回 buffer，继续计算
  completed_loads 写入请求 IDs          后续层同理

调度侧后续检查完成队列
  → 跨 rank 协调 → Wrapper 减少节点锁引用
```

“两条执行线”表达可以独立推进，不承诺它们每一步同时发生。计算侧能否与后续层 I/O 重叠，取决于实际读取速度和执行路径。

### 3.2 从启动方法进入 Storage：先交出事件，再交出后续工作

#### 先找调用者：谁在什么时刻启动加载？

**调度器选好本轮请求、创建 `new_batch` 后，主动调用启动入口；此时尚未执行 `prepare_for_extend()` 和模型前向。** 不是后台自动触发，也不是等待 pending 任务攒够一个阈值。

先在 `managers/scheduler.py` 第 4034 行附近设置断点：

```python
if self.enable_hierarchical_cache or self.enable_unified_cache_external_linker:
    new_batch.hicache_consumer_index = (
        self.tree_cache.ready_to_load_host_cache()
    )

new_batch.prepare_for_extend()
```

调用链：

```text
Scheduler 执行到 ready_to_load_host_cache()
  → UnifiedRadixCache.ready_to_load_host_cache()
  → Wrapper.start_layer_wise_loading()
  → OD.start_layer_wise_loading()
  ← 返回 consumer index，保存到 new_batch
  → prepare_for_extend()
  → 后续模型前向
```

例如一轮选入 A、B、C：前面逐个准备请求时，A、B 有外部命中，各自通过 `load()` 将任务暂存到 `pending_loads`；C 没有外部命中，不产生 GET。创建这三个请求的模型 batch 后，调度器执行上述入口，统一启动 A、B 的读取。

若没有任何 pending 加载任务，启动方法返回 `-1`，不发起 GET。因此走读时先检查本批 `pending_loads` 是否有内容，再单步进入 Storage。**选请求时收集任务，批次选好后启动加载，模型前向时逐层等数据。**

#### 再进入后端和 Storage

打开 `oceandisk_direct_linker.py` 的 `start_layer_wise_loading()`（第 400 行）：

```python
continuation = self.storage.get_layers(
    layers,
    on_layer_submitted=lambda layer, events:
        self.layer_done_counter.complete(index, layer, events),
    defer_remaining=True,
)
self.load_queue.put((index, pending, continuation))
```

这里传入的是 `on_layer_submitted`，不是“已读完”的通知。Storage 提交指定层后，把可等待的 events 交给 counter；counter 用 Future 将 events 发布给计算侧。

进入 `oceandisk_storage.py` 的 `get_layers()`（第 521 行），看三个局部函数/步骤：

- `submit_span(first, last)`：调用 `od_utils.batch_dispatch_get()` 提交指定层范围，并调用上述回调发布各层 events。
- `submit_span(0, 1)`：当前调用先提交第 0 层。
- `return finish`：`defer_remaining=True` 时返回后续执行函数，并未在这里同步等完所有层。

这也解释了 continuation 是什么：**不是另一份数据，而是留给后台加载线程执行的后续工作函数。**

### 3.3 加载线程：等队列、提交后续 GET、等本批读取收尾

回到 `oceandisk_direct_linker.py` 的 `_load_worker()`（第 584 行）：

```python
task = self.load_queue.get()
index, pending, continuation = task
try:
    if continuation is not None:
        continuation()
except BaseException as error:
    self.layer_done_counter.fail(index, error)
finally:
    self.completed_loads.put(list(pending))
```

这里有两类等待：

1. `load_queue.get()`：没任务时等调度侧把批次入队，这是 CPU 工作队列等待。
2. `continuation()` 内：继续执行 Storage 的 `finish()`，提交第 1～末层，再同步所需完成事件，这是 I/O 收尾等待。

当前正常路径使用 `on_layer_submitted` 且没有 `on_layer_ready`，`finish()` 选择最后一层，并等待每个 I/O 描述批次对应的末层 event：

```python
# 正常 direct-wait 路径的等价简化，省略异常和其他 callback 模式
if len(layers) > 1:
    submit_span(1, len(layers))
last_layer = len(layers) - 1
for _, _, events in submitted:
    events[last_layer].synchronize()
```

这条路径利用 GET stream 的顺序来确认本批读取收尾；加载线程不需要把每个模型层再逐一当作计算依赖等待。仅有一层时，它等待的就是第 0 层。

随后 worker 将请求 IDs 放进完成队列，**不是在 worker 中直接解锁树节点**。出现异常时 Storage 会同步排空相应 stream，worker 将异常传给尚未完成的层 Future，并提供清理通知；完成 ID 本身不能作为成功证明。

### 3.4 计算侧：等本批当前层，不等整批全部层

前向前 TP worker 选定 consumer index；attention 调用 pool getter，进入 counter 的 `wait_until(layer)`，详见第 2 节。等待的核心是：

```python
events = futures[layer].result()  # 等加载侧发布当前层的 events
for event in events:
    event.synchronize()          # 在调用侧 CPU 线程等该层 I/O 完成
```

因此，计算侧不是等 worker 从 Python 发来“第 N 层已完成”的额外消息。**加载侧发布 event，计算侧直接等待 event；event 完成后，getter 返回 buffer，模型才继续提交依赖该 KV 的计算。**

如果第 0 层 events 已发布但 I/O 未结束，计算侧停在 `synchronize()`；若某个后续层还未发布 events，则先停在 `Future.result()`。若相应工作早已完成，两步都可以很快返回。

### 3.5 为什么两边都等，不是重复通知吗？

| 执行方与代码 | 等什么 | 等完后做什么 |
| --- | --- | --- |
| 加载线程 `load_queue.get()` | 一个待处理批次 | 提交后续层读取 |
| 计算侧 `Future.result()` | 本批当前层的事件对象或异常 | 得到可等待的事件 |
| 计算侧 `event.synchronize()` | 本批当前层的数据可读 | getter 返回，继续 attention |
| 加载线程 `finish()` 中同步末层 events | 本批在 GET stream 上的读取收尾 | 发布请求 IDs，允许树侧清理 |

同一个 CUDA event 可以被不同 CPU 线程等待，等待不会“消费掉”这个事件。两边即使等待同一末层 event，也是分别保护模型读取和任务生命周期。**计算侧不必等 worker 写 completed_loads 才使用已就绪的层；树侧也不能只因模型已经用过某层，就提前回收还在参与 I/O 的页面。**

完成队列只在调度侧后续轮询时消费：`UnifiedRadixCache.check_hicache_events()` → `Wrapper.drain_loads()` → `cache.dec_lock_ref()`。这里解除的是加载任务持有的那份保护，其他请求引用仍可继续保护页面。

### 3.6 建议断点顺序

1. `OceanDiskDirectLinker.load()`：确认 pending 字典里仍是 keys 和索引。
2. `start_layer_wise_loading()`：观察 counter index、layers 和 continuation。
3. `OceanDiskStorage.get_layers()` 的 `submit_span()`：观察 event 发布回调。
4. `_load_worker()`：确认当前执行线程已是后台加载线程。
5. `Storage.get_layers()` 内的 `finish()`：观察后续层提交和末层事件同步。
6. `TPWorker.forward_batch_generation()` / `set_hicache_consumer()`：观察 batch 使用的 index。
7. `MLATokenToKVPool.get_key_buffer()` → `LayerWiseLoadCounter.wait_until()`：观察计算侧实际等待的层与 events。
8. `UnifiedCacheLinkerWrapper.drain_loads()`：观察完成通知与节点解锁。

多线程单步或暂停会改变原来的重叠时序，因此用断点确认职责与对象关系，用日志/事件时间线观察实际并发程度。

## 4. 普通 PUT 走读：layerwise_save=False

范围仍是 OD PC。普通保存从缓存树的备份动作开始，不需要每层模型保存钩子。`layerwise_save` 是 OD extra config 中的开关，当前默认 True；走读本节需明确处于关闭模式。

### 4.1 树产生备份动作，Wrapper 保护源节点

`unified_radix_cache.py` 处理 `BackupKV` action 时调用 `self.linker.offload_nodes(action.node_ids)`（当前第 1288 行）。进入 `unified_cache_linker.py`：

```text
Wrapper.offload_nodes(node_ids)
  → tree_core.build_external_linker_offload_transfers(node_id)
      构造已计算 KV 的页 keys 和源 GPU slots
  → Wrapper._offload_node(node_id, transfers)
      → cache.inc_lock_ref(node_id)：保护 PUT 源数据
      → OD.offload(transfers)
      → 若接受，标记 pending 并记录 pending_offloads
```

这里使用的是已有数据的源 slots，不是像 GET 一样为目标新分配 slots。节点锁用于防止传输期间源内存被回收重用。未接受或提交抛异常时，Wrapper 释放本次锁引用。

### 4.2 OD.offload()：展开物理池任务并入队

`oceandisk_direct_linker.py` 的 `offload()`（第 610 行），owner 的普通模式逻辑可简化为：

```python
expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
event = self._ready_event()
self.offload_queue.put((expanded, event))
return True
```

`_ready_event()` 在当前生产侧 CUDA stream 上记录 event，用于确保此前生产的源数据已准备好。`offload()` 返回 True 只是接受，不表示 PUT 完成。

非写入 owner 不发原生 PUT，产生本地确认结果；最终还需结合实际写入者的跨 rank 结果，才能确认外部副本有效。

### 4.3 offload worker：先等源数据，再执行 PUT

进入同文件 `_offload_worker()`，普通模式分支：

```python
event.synchronize()                          # 等 GPU 源数据生产完成
self._run_transfers(transfers, write=True)  # 组织并执行 PUT
success = True
# 异常则记录失败，随后仍交付对应结果
self.offload_results.put(success)
```

沿 `_run_transfers()` 继续走：

```text
_transfer_layers(transfers)
  → Entry.prepare_locations()：源 slots 转各池行号
  → _page_fragments()：计算源 GPU pointer / size / 存储 offset
  → 各层排列配对 KV / Indexer 范围
  → Storage.put_layers(layers)
```

这里才结合本次源 slots 构造 PUT 地址描述。assembler 和 Group / Entry 已在初始化时创建，不需要重新构建。

### 4.4 Storage：原生提交并等待保存收尾

`oceandisk_storage.py` 的 `put_layers()`（第 611 行）：

```text
在 _put_stream 上处理
  → _pack_layer_batches()：生成 MR / key / offset / size / connection metadata
  → metadata 传到 GPU，准备各层事件
  → od_utils.batch_dispatch_put()：提交原生 PUT
  → 等待各描述批次的末层 event
  → 返回 worker
```

因此 worker 等待的顺序是：**先等生产侧 ready event，再等 PUT 完成事件。** 后者在 `put_layers()` 内执行。异常时同步 PUT stream 后传播，避免还有 I/O 访问页面就提前解除保护。

### 4.5 树侧确认保存结果，解锁

```text
worker：offload_results.put(success)
  → Cache.check_hicache_events() 跨 rank 协调完成数量和成功状态
  → Wrapper.commit_completed_offloads(successes)
      → tree_core.finish_external_linker_offload(..., success)
      → cache.dec_lock_ref(...) 解除源节点保护
```

失败时不能把节点发布为成功保存。结果接口只有 bool，没有任务 ID，因此 worker 对外必须按接受顺序提供结果，与 Wrapper 的 pending 列表对应。

## 5. 逐层 PUT 走读：layerwise_save=True

当前 OD 默认模式。真实 PUT 从模型前向的层钩子发起；树侧后续的 offload 是确认已有保存结果。这两段要分开读。

### 5.1 前向前：TP worker 请求 Wrapper 创建保存计划

`tp_worker.py` 第 624 行附近：

```python
layerwise_linker = getattr(self, "layerwise_cache_linker", None)
if layerwise_linker is not None and batch is not None and not is_verify:
    forward_batch.layerwise_kv_saver = layerwise_linker.prepare_layerwise_save(batch)
```

进入 `Wrapper.prepare_layerwise_save()`（`unified_cache_linker.py` 第 198 行）：

```text
检查 allocator、forward mode 等支持条件
  → 选择本次新产出的完整页，生成/复用页面 hashes
  → 从 req_to_token 映射取得源 slots，构造逻辑 KV transfers
  → OD.begin_layerwise_save(transfers)
  → 保存请求对 receipts 的引用
  → allocator.pin_pages_for_io(pages)
  → 将 unpin 回调绑定为 session.release_source
```

源 slots 已由模型计算流程准备；此处 pin 是暂时禁止它们被回收重用，并不是又分配一份 KV。guards 不满足或没有新完整页时可以不生成 session，不能假定每次前向一定有 PUT。

### 5.2 OD 创建 session、地址描述和 receipts

`OceanDiskDirectLinker.begin_layerwise_save()`（第 515 行）：

```text
编码本次页面 object keys，跳过已在保存的页面
  → owner：逐份展开逻辑 KV 为 KV / Indexer
  → owner：_transfer_layers() 构造各层源地址描述
  → 创建 LayerwiseSaveSession
```

`LayerwiseSaveSession.__init__()` 为这次前向创建 `completion = Future()`，将页 key 映射到这个 Future，作为保存 receipt。同一 session 的多页可共享一个完成结果。session 的 `completion` 与 GET counter 中“承载 events 的 Future”不同：**这里最终存的是保存成功/失败的 bool。**

非 owner 也参与 session 生命周期，但不准备实际 PUT 地址和原生提交。

### 5.3 模型每产出一层，向 offload_queue 提交层任务

`model_runner.py` 第 1785 行附近用 saver 作为上下文管理器包住 forward：

```python
saver = getattr(forward_batch, "layerwise_kv_saver", None)
with ctx_mgr, saver if saver is not None else contextlib.nullcontext():
    # 执行模型 forward
    ...
```

`deepseek_v2.py` 第 2465 行附近，在该层 attention 返回后调用：

```python
if graph_hooks is not None:
    graph_hooks.save_layer(self.layer_id)
elif saver is not None:
    saver.save_layer(self.layer_id)
```

普通执行进入 `LayerwiseSaveSession.save_layer()`：owner 记录当前计算 stream 的 ready event，然后将 `(session, layer, event)` 放进 `offload_queue`。图路径通过专门 hooks 接入；不能把 Python 普通调用等同于所有图回放细节。

**此时不是要求前向线程等 PUT 完成。** 它提交层任务后可以继续；后台 worker 等 ready event，以确保读取源 KV 时 GPU 已生产完这一层数据。

### 5.4 offload worker：等待每层生产完成，再提交该层 PUT

`_offload_worker()` 识别三元组层任务，调用：

```python
if session.prepared is None:
    session.prepared = self.storage.prepare_layerwise_put(session.layers)
    session.layers = None
self.storage.submit_put_layer(layer, session.prepared[layer], ready_event)
```

`prepare_layerwise_put()`（Storage 第 704 行）分配 metadata arena 区间、打包各层描述、上传 metadata，并保留区间直到 I/O 收尾。

`submit_put_layer()`（第 732 行）的核心顺序：

```python
ready_event.synchronize()  # 后台 CPU worker 等该层 GPU 数据生产完成
# 随后在 PUT stream 上提交该层原生写入
od_utils.batch_dispatch_put(...)
```

这个方法不逐层等待 PUT 全部完成；层任务按队列继续推进，最终由 session 收尾确认保存结果。若提交失败，标记 `session.put_failed`，后续仍需排空已提交工作。

### 5.5 forward 退出：入队 session 收尾，worker 等最终保存完成

`LayerwiseSaveSession.__exit__()` 将 session 本身放入同一个 FIFO 队列，排在本 session 各层任务之后；同时通过 callback 把 session 交给 Wrapper 的 pending 列表。若 forward 出错或没有提交齐层，标记 aborted。

worker 识别 session 收尾任务：

```python
task.finish_event = self.storage.record_save_event()
task.finish_event.synchronize()
# 确认结束后，释放 Storage metadata 区间
success = not task.aborted and not task.put_failed
task.completion.set_result(success)
```

owner 在 PUT stream 末尾记录事件并等待，确认前面提交的 PUT 收尾后才完成 session Future。准备/完成阶段异常也按失败处理。**这里等的是 PUT 完成事件，不是生产侧 ready event。** 非 owner 不做原生等待，仍报告本地 session 状态参与全局协调。

源页 pin 尚不能由 forward 退出直接释放。调度侧 `check_hicache_events()` 协调 session 完成数量和跨 rank 成功状态，再调用：

```text
Wrapper.commit_completed_layerwise_saves()
  → session.finish(success)
      → 清理在途页记录和相应 receipt 状态
      → session.release_source()
          → allocator.unpin_pages_for_io(lease)
```

forward 失败也必须等在途 I/O 排空后再释放源页，不能因模型异常就提前复用这些 slots。

### 5.6 树侧 BackupKV：确认 receipt，不再 PUT 一遍

之后树侧仍可产生 `BackupKV`，走 `Wrapper.offload_nodes()` → `_offload_node()` → `OD.offload()`，但 layerwise 模式的 owner 将 receipts 入队：

```python
payload = self.get_layerwise_receipts(kv.keys)
self.offload_queue.put((payload, None))
```

worker 在此分支核对：

```python
success = all(
    receipt is not None and receipt.result()
    for receipt in transfers
)
self.offload_results.put(success)
```

这次 `receipt.result()` 等/读取的是 session 保存的 bool。对应 session 的收尾任务应排在前面；不能误认为这里又在等一个 GET event。receipt 缺失或保存失败就确认失败，**不会自动执行整页 PUT 重试**。

随后走与普通保存相同的树侧确认：跨 rank 汇总 → `finish_external_linker_offload()` → 释放该次 offload 的节点锁引用。它与前面 session 的页 pin 是不同的保护，分别在各自生命周期结束时解除。

## 6. PUT 两种模式的等待与断点对照

| 模式与执行位置 | 等什么 | 目的 |
| --- | --- | --- |
| 普通：offload worker，`event.synchronize()` | 源数据生产 ready event | GPU 生产完才能从源 buffer 读取 |
| 普通：worker 内 `Storage.put_layers()` | 各描述批次末层 PUT events | 完成写入后才能报告保存结果 |
| 逐层：计算侧 `session.save_layer()` | 记录 ready event 并入队，不在此同步等 PUT | 让前向与后台保存有机会重叠 |
| 逐层：worker 内 `submit_put_layer()` | 当前层生产 ready event | 保证本层源数据可读取 |
| 逐层：worker 处理 session 收尾 | PUT stream 最终保存 event | 保证全部已提交 PUT 收尾 |
| 逐层：worker 处理树侧确认任务 | receipt Future 的 bool | 核对先前保存是否成功，不重新 PUT |
| 两者：调度侧完成提交 | 各 rank 的可消费结果及成功共识 | 提交树状态并解除对应保护 |

建议分两轮设置断点，避免混淆分支：

```text
普通模式：
树 BackupKV → Wrapper.offload_nodes / _offload_node
  → OD.offload → _offload_worker 普通分支
  → _run_transfers(write=True) → Storage.put_layers
  → Wrapper.commit_completed_offloads

逐层模式：
TP worker → Wrapper.prepare_layerwise_save → OD.begin_layerwise_save
  → Session.__enter__ → 模型 save_layer → Session.save_layer
  → worker 层任务 → prepare_layerwise_put / submit_put_layer
  → Session.__exit__ → worker session 收尾 → session completion
  → Wrapper.commit_completed_layerwise_saves → Session.finish → unpin

另跟树侧确认：
BackupKV → OD.offload → receipts → worker 确认结果
  → Wrapper.commit_completed_offloads → 解除节点锁
```

源码中的保护和回调会影响不同任务的具体交错顺序，不能把这张读图顺序当作所有情况下唯一的线程时间线。以上是代码静态走读，未执行真实 GPU / GIDS 集成验证。

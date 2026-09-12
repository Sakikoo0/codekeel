# Agent 控制循环初探

## 1. 概述

源码: `src/codekeel/agent/loop.py`

```text
Agent.__init__(dependencies)
        │
        └─ 创建 Agent 对象, 保存协议对象、策略与默认组件

run(task, repo_context)
        │
        ├─ 建立 system/user 初始消息
        ├─ status = RUNNING
        ├─ 重置本次 run 的 ID、计数边界和 pending 状态
        ├─ 创建 ToolContext
        ├─ 记录 started_at
        └─ _drive(new=True)
                 │
                 └─ while status is RUNNING
                            │
                            └─ step()
                                 │
                                 └─ _step()
                                      ├─ 处理已有审批决定
                                      ├─ 检查预算
                                      ├─ 准备模型上下文
                                      ├─ 再次检查预算
                                      ├─ model.complete()
                                      ├─ 累计 usage 并写入 assistant message
                                      └─ _parse_action(response)
                                           ├─ FinalAnswer
                                           │    └─ 完成或进入验证
                                           └─ ToolCall
                                                └─ _execute_action()
```

最外层的 `_drive()` 没有复杂调度算法. Agent 是否继续, 完全由一个清晰条件决定:

```python
while self.state.status is RunStatus.RUNNING:
    await self.step()
```

因此理解主循环时不要把所有方法看成同一级别:

- `run()` 创建一次运行
- `_drive()` 驱动生命周期
- `step()` 定义单步外壳
- `_step()` 执行一次决策
- `_parse_action()` 解释本次模型响应

## 2. `Agent.__init__()`: 装配长期依赖

源码: `src/codekeel/agent/loop.py`

构造函数接收两个必需依赖:

- `model: Model`: 生成下一次响应;
- `workspace: Workspace`: 工具访问仓库和执行命令的边界

其余依赖均为 keyword-only, 可分成四组:

| 组别           | 参数                                                                    | 作用                                         |
| -------------- | ----------------------------------------------------------------------- | -------------------------------------------- |
| 工具与上下文   | `tool_registry`、`tool_output_manager`、`context_manager`、`explorer`   | 决定模型能调用什么, 以及如何准备和收缩上下文 |
| 运行控制       | `system_prompt`、`budgets`、`clock`、`policy`、`verification_policy`    | 决定行为提示、资源边界、动作权限和完成条件   |
| 可恢复性与审计 | `event_store`、`checkpoint_store`、`workspace_metadata`、`run_metadata` | 保存发生的事实及可恢复状态                   |
| 辅助任务状态   | `plan`                                                                  | 提供初始公开计划                             |

### 2.1 默认组件体现 "小而可运行"

调用方只提供 Model 和 Workspace 时, 构造函数会补齐:

```text
default_tool_registry()
BudgetLimits()
MemoryEventStore()
ToolOutputManager()
DeterministicContextManager()
ActionPolicy()
RunMetadata()
```

这使 Agent core 可以在不配置外部持久化的情况下运行, 同时仍具备默认工具、有限 step/wall time、动作策略和确定性上下文管理. 默认事件存储在内存中, 说明“记录事件”是主循环固有行为, 但“落盘”不是构造一个 Agent 的必要条件.

这里大量使用 `x if x is not None else default`, 而不是简单写 `x or default`. 这样一个合法但布尔值为假的自定义对象不会被错误替换. `tool_registry` 是例外, 源码使用 `tool_registry or default_tool_registry()`; 因此自定义注册表的 truthiness 若被实现为 `False`, 会意外触发默认注册表. 这不是当前内建注册表的正常行为, 但属于接口层值得留意的小差异.

### 2.2 Explorer 注入不会直接修改调用方的注册表

配置 Explorer 时, 构造函数先按原 definitions 的顺序重新建立一个 `ToolRegistry`, 再注册 `DelegateExploreTool`. 这避免把委托工具直接塞进调用方交来的 registry, 使 Agent 自己的装配选择不会反向污染外部对象.

第一遍只需要知道结果: 有 Explorer 时, 主 Agent 多获得一个委托工具; Explorer 如何保持只读、如何共享预算, 留到对应阶段.

### 2.3 可变配置在边界上复制或验证

- `policy` 使用 `model_copy(deep=True)`, 避免 Agent 运行期间与调用方共享嵌套可变配置;
- 初始 `plan` 通过 `Plan.model_validate()` 规范化, 并同时保存为 `_initial_plan` 和当前 `plan`;
- verification policy 也经 `model_validate()` 进入 Agent;
- budgets 被封装进 `TerminationPolicy`, clock 一并注入.

持久化参数存在最小组合约束: 只要提供 `checkpoint_store`, 就必须同时显式提供 workspace metadata 和 event store. 其深层原因本阶段不展开; 从装配角度看, 它避免创建一个“声称可恢复、却缺少身份或事件证据”的 Agent.

### 3.4 构造完成仍是 `IDLE`

`self.state = AgentState()` 产生的默认状态是 `IDLE`. 此时尚无:

- task 消息;
- run ID;
- ToolContext;
- started time;
- event sequence.

因此 `__init__()` 只是配置 Agent 实例, 并不等于启动一次 run. 真正把这些运行级数据建立起来的是 `run()`.

## 3. `run()`: 把可复用 Agent 变成一次具体运行

源码: `src/codekeel/agent/loop.py`

`run(task, repo_context=None)` 的主要工作不是循环, 而是重置并初始化 run-scoped state.

### 3.1 初始消息只有 system 与 user

它先取 `self.system_prompt`. 若调用方传入 `repo_context`, 则把 `repo_context.render()` 追加到本次使用的局部 `system_prompt` 字符串, 随后建立:

```python
[
    Message(role="system", content=system_prompt),
    Message(role="user", content=task),
]
```

这里不会修改 `self.system_prompt`, 所以同一个 Agent 对象之后再次运行且没有 repo context 时, 不会继承上一次拼接的仓库快照.

随后创建全新的 `AgentState`, 并直接设置 `status=RUNNING`. 旧 run 的 messages、usage 和 counters 不会被沿用.

### 3.2 重置运行级身份和控制字段

每次 `run()` 都会:

- 清空 `pending_approval`;
- 把计划恢复为 `_initial_plan`;
- 生成新的 UUID hex `run_id`;
- 将事件 sequence、checkpoint revision 和 last event ID 归零;
- 把当前位置标记为 boundary;
- 记录单调时钟起点.

这说明 `Agent` 对象保存可复用装配, 但 `AgentState` 与 run ID 属于单次运行. 再次调用 `run()` 的语义是开启新 run, 不是延续上次对话; 延续已有 run 属于 `resume()` 的职责.

### 3.3 创建本次运行的 `ToolContext`

`run()` 把以下能力注入工具上下文:

```text
workspace
run_id
defer_output_limits=True
update_plan=self._update_plan
delegate_explore=self._delegate_explore
```

模型只能看到工具 definition, 不能创建这份 Context. Context 把工具调用绑定到本次 workspace 和 run, 并通过窄回调连接计划与 Explorer. 这里仍不需要理解回调内部; 只需确认运行级能力是在 `run()` 时产生, 而不是模型参数的一部分.

完成初始化后, `run()` 立即调用并返回 `_drive(new=True)` 的最终 `AgentState`.

## 4. `_drive()`: 生命周期驱动器

源码: `src/codekeel/agent/loop.py`

主体可以压缩为:

```python
async def _drive(self, *, new: bool) -> AgentState:
    if new:
        record_run_started()

    while self.state.status is RunStatus.RUNNING:
        await self.step()

    if self.state.status is not RunStatus.WAITING_FOR_APPROVAL:
        record_run_finished()

    return self.state
```

它本身不判断 "任务完成了吗", 也不决定下一项工具. 它只信任 `state.status`:

- 仍是 `RUNNING`: 继续下一步;
- 变成 `WAITING_FOR_APPROVAL`: 退出驱动并把状态交还外部, 不写普通 RunFinished;
- 变成完成、预算终止或验证失败: 结束循环并返回;
- 发生取消或异常: 设置 `CANCELLED` / `FAILED`, 进行相应记录后重新抛出.

最后一点很重要: `_drive()` 对异常不是 "吞掉后返回一个失败状态". 普通异常和取消都会继续抛给调用者, 因此调用 API 时必须同时考虑两种结果通道:

```text
正常返回 AgentState
或
抛出异常, 同时 Agent 内部状态已转为 FAILED/CANCELLED
```

事件存储或检查点存储自身出错时, 源码直接标记 `FAILED` 并抛出, 不再尝试依赖同一个故障通道记录自身失败.

## 5. `step()`: 一次可恢复边界的外壳

源码: `src/codekeel/agent/loop.py`

`step()` 不是 "直接调用一次模型" 的同义词, 它先验证两个前置条件:

1. 当前状态必须严格为 `RUNNING`
2. 本次 run 的 ToolContext 与 started time 必须已经存在

所以不能在刚构造的 `IDLE` Agent 上直接调用 `step()`, 也不能在完成或等待审批状态下把它当作通用推进按钮.

其结构是:

```text
检查 active run
  → 标记 step 正在进行
  → 保存边界状态
  → await _step()
  → 根据结果判断是否到达稳定边界
  → 再保存状态
```

## 6. `_step()`: 一次模型决策的主干

源码: `src/codekeel/agent/loop.py:360-412`

它大致分为六段.

### 6.1 优先处理已决定的 pending action

如果存在 `pending_approval`, 说明模型调用早已发生, 本次推进不应再请求模型. 源码要求该 pending action 已有明确批准或拒绝结论, 然后清空 pending 字段并把原 ToolCall 交给 `_execute_action()`.

这条分支随后立即返回. 也就是说, 恢复审批动作和生成新模型响应是互斥的单步路径.

### 6.2 在准备上下文前检查终止条件

没有 pending action 时, 首先调用:

```python
self.termination_policy.evaluate(self.state, started_at=self._started_at)
```

若返回状态, 就写入 `state.status` 并结束本 step. 预算是进入新一轮工作的门卫, 而不是等模型调用后才检查.

### 6.3 收集工具定义并准备请求上下文

`definitions = self.tool_registry.definitions()` 得到本次可发给模型的工具规格, 随后 `_prepare_context(definitions)` 生成真正请求模型的 messages. 准备过程可能超时, 也可能为了压缩上下文额外调用模型, 所以它不是简单的列表复制.

本阶段不展开其算法, 但必须注意紧接着出现第二次 termination check. 原因是 `_prepare_context()` 自身可能已经花掉 model call、tokens、cost 或 wall time；第一次检查通过并不自动授权之后的主模型请求.

### 6.4 先记次数, 再请求模型

第二次预算检查通过后:

```python
self.state.steps += 1
self.state.model_calls += 1
```

然后记录预算与模型请求, 最后通过带 deadline 的 `_complete_with_deadline()` 调用模型. 计数发生在外部调用之前, 因此它表达“已开始一次请求”, 即使等待模型时超时, 也不会把这次尝试从账本抹去.

### 6.5 响应回来后累计 usage, 并写入规范历史

成功得到 `ModelResponse` 后, 源码依次:

1. `state.add_usage(response.usage)`；
2. 记录响应与更新后的预算；
3. 把响应转换成 `Message(role="assistant", ...)` 加入 `state.messages`；
4. 调用 `_parse_action(response)` 解释下一步.

模型 provider 对象不会进入 Agent 历史, 保存的是统一 `Message` 和规范化的 `Usage`. 这与第一阶段建立的数据协议正好接上.

### 6.6 最终答案或工具调用二选一

```text
_parse_action(response)
        │
        ├─ FinalAnswer
        │    ├─ 没配置 verification → COMPLETED
        │    └─ 配置 verification   → _verify_final_intent()
        │
        └─ ToolCall
             └─ _execute_action()
```

工具执行完成后, `_step()` 自身不会递归调用模型. 控制返回 `step()`, 再返回 `_drive()`；若 status 仍为 `RUNNING`, while 循环才发起下一 step. 这个“回到驱动器再前进”的结构保持了调用栈和状态边界的线性.

## 7. `_parse_action()`: 把宽协议收窄为线性控制流

源码: `src/codekeel/agent/loop.py`

`ModelResponse` 的通用协议允许 `tool_calls` 是列表, 但 baseline Agent 只接受三种结果:

| 响应形状                              | 解析结果                |
| ------------------------------------- | ----------------------- |
| 没有 tool call, `content is not None` | `FinalAnswer(content)`  |
| 恰好一个 tool call                    | 返回该 `ToolCall`       |
| 没有 tool call, `content is None`     | 抛 `AgentProtocolError` |
| 两个及以上 tool calls                 | 抛 `AgentProtocolError` |

两个细节容易漏看:

- 最终文本只检查是否为 `None`, 空字符串 `""` 仍会被包装成 `FinalAnswer`；
- 若响应同时带 content 和恰好一个 tool call, tool call 分支优先, content 不会让它提前完成.

`_parse_action()` 不负责检查工具是否已注册、arguments 是否正确, 也不执行工具. 它只做控制流形状校验, 把 provider-independent 的宽响应压成:

```text
ToolCall | FinalAnswer
```

`FinalAnswer` 是冻结且带 slots 的小 dataclass, 仅包含 `content`. 它的作用不是建立另一套消息模型, 而是让 `_step()` 可以用类型明确区分“继续执行动作”和“模型表达结束”.

## 8. 为什么这是线性 Agent

线性 Agent 是一种沿单一演化状态轨迹执行的智能体: 它重复进行模型调用、工具执行和结果状态更新, 直到满足终止条件, 并且不会维护彼此独立的执行分支.

一次主模型响应最多只能带一个工具调用:

```python
if len(response.tool_calls) != 1:
    raise AgentProtocolError(...)
```

完整工具链因此只能是:

```text
model response #1
  → tool call A
  → tool result A 加入 history
  → model response #2
  → tool call B
  → tool result B 加入 history
  → model response #3
  → final answer
```

而不能是:

```text
一个 model response
  ├─ tool A ─┐
  ├─ tool B ─┼─ 并发执行
  └─ tool C ─┘
```

这项约束同时出现在默认 system prompt 和运行时 `_parse_action()` 中: prompt 引导模型遵守, 代码则把它变成不可绕过的协议边界. 只写进 prompt 不足以保证行为；只在代码拒绝又会浪费模型轮次. 两层配合分别承担引导和强制.

线性设计的主要收益是:

- 每次工具结果都能在下一次决策前进入历史；
- tool call 与 tool result 的配对简单；
- 预算计数和 status 迁移顺序明确；
- 审批只需绑定一个 pending action；
- 单步持久化边界更容易定义.

代价也很直接: 两个互不依赖的只读工具不能在同一模型响应中并发执行, 会多消耗模型轮次和延迟. 当前项目明显选择可审计、可恢复的确定性顺序, 而不是最大化工具吞吐.

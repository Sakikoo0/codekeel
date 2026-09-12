# 工具与 Workspace

## 1. 工具注册和分发

源码: `src/codekeel/tools/registry.py`

### 1.1 概述

直接相关的基础协议:

```text
Tool
├─ name
├─ definition() -> ToolDefinition
└─ execute(arguments, ToolContext) -> ToolResult

ToolCall
├─ id
├─ name
└─ arguments
```

注册表位于模型协议与具体工具实现之间: 它把 Tool 对象组织成一个确定的名称空间, 并把已经规范化的 `ToolCall` 路由到对应 Tool.

`ToolRegistry` 只有四个公开行为:

```text
ToolRegistry(tools)
    └─ 对输入 iterable 中每个 Tool 调用 register()

register(tool)
    ├─ 校验 tool.name == tool.definition().name
    ├─ 拒绝重复名称
    └─ 保存 name -> Tool

definitions()
    └─ 按注册顺序重新取得每个 ToolDefinition

get(name)
    ├─ 找到: 返回 Tool
    └─ 未找到: 抛 UnknownToolError

execute(call, context)
    ├─ get(call.name)
    └─ await tool.execute(dict(call.arguments), context)
```

它不是安全策略、schema validator、异常处理器或 Workspace. 它只解决两个问题: 

- 当前 runtime 有哪些名字唯一的工具
- 一个调用名称应分发给哪个工具对象. 

这个边界越小, Agent、Explorer 或其他调用方就越能在同一分发机制外组合不同的 policy、budget、deadline 和错误处理. 

### 1.2 内部表示: `dict[str, Tool]`

源码: `src/codekeel/tools/registry.py`

构造函数先创建空字典: 

```python
self._tools: dict[str, Tool] = {}
```

键是工具名称, 值是满足 `Tool` Protocol 的具体对象. 这个表示同时提供: 

- 按名称平均常数时间查找; 
- 天然检测重复键; 
- 保留插入顺序, 使 definitions 顺序确定. 

构造参数是 `Iterable[Tool]`, 不要求调用方先建立 list. 列表、元组和 generator 都能输入. 构造函数没有另写一套批量校验, 而是逐个调用公开的 `register()`: 

```python
for tool in tools:
    self.register(tool)
```

这保证 "构造时加入" 和 "构造后追加" 遵守完全相同的名称一致性与去重规则. 若输入中某个 Tool 不合法, 构造会当场抛出, 而不会返回一个看似成功的 registry. 

默认参数使用空 tuple `()`, 它是不可变对象, 不存在共享可变默认值问题. `ToolRegistry()` 因而可以合法创建一个空注册表; 空表的 `definitions()` 返回空列表, 任何 `get()` 或 `execute()` 都会因未知名称失败. 

### 1.3 `register()`: 建立名称空间不变量

源码: `src/codekeel/tools/registry.py`

注册顺序是: 

```text
调用 tool.definition()
        │
        ├─ definition.name != tool.name
        │      └─ ToolRegistryError
        │
        └─ 名称一致
               │
               ├─ name 已存在
               │      └─ DuplicateToolError
               │
               └─ 写入 _tools[name] = tool
```

只有两个校验都通过, 工具才会进入内部字典. 因此失败注册不会替换已有条目, 也不会把当前非法对象写入 registry. 

#### 为什么要核对两处名称

同一个工具存在两个名称来源: 

- `tool.name`: runtime 查找和分发使用; 
- `tool.definition().name`: 发送给模型, 模型据此生成 `ToolCall.name`. 

如果二者不同, 就会形成断裂: 

```text
模型看到 definition.name = "read"
            │
            └─ 返回 ToolCall(name="read")

runtime 却按 tool.name = "read_file" 注册
            │
            └─ 无法找到模型刚被告知可用的工具
```

`register()` 在装配期直接拒绝这种对象, 比等模型实际调用后才报 unknown tool 更早、更确定. 异常信息同时包含两边名称, 便于定位具体实现错误. 

这项检查只锁定名称. 源码没有比较 `tool.description` 与 `definition.description`, 也没有验证 `definition.parameters` 与 `execute()` 的真实参数要求一致. 后两项仍是 Tool 实现者需要维护的契约. 

#### 为什么禁止同名覆盖

若名称已存在, 源码抛出 `DuplicateToolError`: 

```python
if tool.name in self._tools:
    raise DuplicateToolError(...)
```

它没有采用常见的 `self._tools[name] = new_tool` 覆盖语义. 这一点对 Agent 很重要, 因为工具名是模型调用的唯一分发键. 静默覆盖会产生几个问题: 

- 模型看到的 schema 可能来自一个工具, 执行时对象却已被另一个替换; 
- 注册顺序或装配顺序会暗中改变安全行为; 
- 同名伪工具可能冒充预期内建工具; 
- 配置错误直到产生副作用时才暴露. 

显式失败让歧义在装配阶段结束. 若调用方确实要替换某个工具, 应明确创建新的 registry, 而不是依赖覆盖副作用. 源码中的 Agent 在注入 Explorer 工具时也采用 "从原工具重建新注册表, 再追加新工具" 的方式. 

`DuplicateToolError` 是 `ToolRegistryError` 的子类, 后者又继承 `ValueError`. 因此调用方既可以针对重复注册精确处理, 也可以把所有注册/分发输入错误作为同一错误族处理. 

#### 注册检查是时点检查, 不是永久冻结

`register()` 调用一次 `definition()` 并校验当时的名称, 随后保存 Tool 对象本身, 而不是保存 definition 快照. `Tool` Protocol 也没有要求对象冻结. 因此如果某个第三方 Tool 的 `name` 或 `definition()` 会随时间变化, 注册表不会自动阻止之后发生漂移. 

内建工具通常使用稳定字段并每次生成一致 definition, 所以正常路径没有这个问题. 但从注册表自身保证看, 应准确表述为: 

> 注册时确认对象名称与当次 definition 名称一致. 

而不是: 

> 注册表永久证明这个工具的所有元数据都不可变. 

当前实现选择信任 Tool 在注册后保持契约稳定, 没有增加缓存、冻结包装或每次执行时重新比对. 

### 1.4 `definitions()`: 稳定地构造模型可见工具列表

源码: `src/codekeel/tools/registry.py`

实现只有一行: 

```python
return [tool.definition() for tool in self._tools.values()]
```

Python 字典按插入顺序迭代, 因此结果顺序等于成功注册顺序. 它不按名称排序, 也不根据工具类别重新排列. 

例如: 

```text
register(B)
register(A)
register(C)

definitions() -> [B.definition(), A.definition(), C.definition()]
```

稳定顺序有三层实际价值: 

- 相同装配得到相同的模型请求形状; 
- 工具列表进入 checkpoint 或一致性比较时不会随机抖动; 
- 调试时能把模型看到的工具顺序追溯到注册顺序. 

`definitions()` 每次返回一个新的 list, 并重新调用每个工具的 `definition()`. 调用方修改返回列表本身不会改变 registry 的 `_tools` 字典; 但列表中的 `ToolDefinition` 是否稳定, 仍依赖工具每次正确实现 `definition()`. 

注册表也没有暴露内部 `_tools` 字典, 因此普通调用方不能直接绕过 `register()` 写入、删除或覆盖条目. 它提供的是一个很窄的读/追加接口: `register`、`definitions`、`get`、`execute`. 

### 1.5 `get()`: 查找失败必须显式化

源码: `src/codekeel/tools/registry.py`

`get(name)` 直接按字典键查找. 成功时返回原始 Tool 对象; 失败时把内部 `KeyError` 转换为: 

```python
UnknownToolError(f"Unsupported tool: {name}")
```

并通过 `raise ... from error` 保留异常链. 

这层转换很有意义: `KeyError` 只说明字典没有键, 而 `UnknownToolError` 明确说明错误发生在工具分发边界. 上层不需要知道 registry 内部用字典实现, 也不会把未知模型工具名误判成普通容器错误. 

`get()` 不做近似匹配、大小写归一化、别名或 fallback. 例如注册名是 `read_file`, 调用 `Read_File` 就是未知工具. 这让模型可见名称到执行对象的映射保持精确, 避免 "自动纠错" 把不明确调用路由到错误副作用. 

成功返回的是注册时保存的同一个对象, 不是副本. 工具若包含内部状态, 该状态会跨多次调用保留. 注册表本身不声明工具必须无状态或并发安全; 这些性质属于具体 Tool 与调用 runtime. 

### 1.6 `execute()`: 只查找并转交

源码: `src/codekeel/tools/registry.py`

完整实现是: 

```python
tool = self.get(call.name)
return await tool.execute(dict(call.arguments), context)
```

数据流如下: 

```text
ToolCall
├─ name ───────────────> get(name) ──> Tool
└─ arguments ──> dict(arguments) ───────┐
                                        ├─> Tool.execute(...)
ToolContext ────────────────────────────┘
                                                │
                                                └─> ToolResult
```

#### 为什么复制 `arguments`

`dict(call.arguments)` 创建一份新的顶层字典后才交给工具. 因此工具对参数字典执行增加、删除或替换键, 不会直接改变 `ToolCall.arguments` 的顶层映射. 

这是浅复制, 不是深复制. 若参数值包含嵌套 list 或 dict, 工具仍可能拿到共享的嵌套对象. 注册表只提供轻量的顶层隔离, 并没有承诺把整个调用对象变成深度不可变数据. 

#### 为什么必须 `await`

Tool Protocol 的 `execute()` 是异步方法, 因为真实工具最终可能等待 Workspace 文件操作、命令执行或委托任务. 注册表不关心底层 I/O 形式, 只保持同一个异步分发接口, 并把最终 `ToolResult` 原样返回. 

#### `execute()` 刻意不做的事情

注册表不会: 

- 根据 `ToolDefinition.parameters` 通用校验 arguments; 
- 判断动作风险或请求审批; 
- 检查 step/tool/wall-time 预算; 
- 生成 `ToolCalled`、`ToolCompleted` 或 `ToolFailed` 事件; 
- 捕获 `UnknownToolError` 或具体 Tool 抛出的异常; 
- 裁剪、落盘或重写 `ToolResult.content`; 
- 直接访问 Workspace; 
- 根据 `ToolResult.is_error` 修改 Agent 状态. 

这些行为分别属于具体工具、ActionPolicy、Agent 编排、输出管理或 Workspace. 注册表只完成: 

```text
精确查找 → 参数顶层复制 → 异步调用 → 原样返回/原样抛出
```

这种窄职责解释了为什么同一个 registry 能同时被主 Agent 和 Explorer 使用: 两者可以围绕相同分发器设置不同的工具集合、策略和错误处理. 

### 1.7 默认工具集与确定顺序

源码: `src/codekeel/tools/registry.py`

`default_tool_registry()` 每次创建新的具体工具和新的注册表: 

```python
ToolRegistry([ShellTool(), *filesystem_tools(), UpdatePlanTool()])
```

默认非只读文件配置下, 模型可见顺序为: 

```text
1. shell
2. read_file
3. write_file
4. edit_file
5. list_directory
6. find_files
7. search_files
8. update_plan
```

这个顺序直接来自列表展开顺序: Shell 在前, `filesystem_tools()` 返回的六个工具居中, 计划工具最后. 每次调用 factory 都新建 registry, 而不是返回一个进程级单例, 所以不同 Agent 的默认注册表不会因为向其中一个追加工具而共享 `_tools` 字典. 

默认集合表达的是 "常规主 Agent 可用能力" , 不是 ToolRegistry 强制要求的固定全集. 其他调用方可以构造子集: 源码中的 Explorer 使用只读文件工具加 Git 读取工具; 评测装配也可以重建 registry. 注册表只维护传入集合的不变量, 不知道什么叫 "主 Agent 默认能力" . 

### 1.8 从主 Agent 看注册表处于哪一层

不展开 `_execute_action()` 的内部策略, 只看调用边界: 

```text
Agent 准备模型请求
    └─ registry.definitions()
          └─ 把模型可见 schemas 发给 Model

Model 返回 ToolCall(name, arguments)
    │
    ├─ Agent 负责预算、policy、deadline、事件等编排
    │
    └─ registry.execute(call, ToolContext)
          ├─ get(call.name)
          └─ tool.execute(arguments, context)
```

同一个 registry 同时服务 "公开能力" 和 "执行能力" : 

- `definitions()` 决定模型被告知哪些工具; 
- `get()/execute()` 决定名称实际落到哪个对象. 

注册时的名称一致性正是连接这两侧的关键不变量. 缺少它, 模型可见 schema 与 runtime 路由就可能属于两套名称空间. 

需要注意, Agent 的某些路径会先 `get()` 再直接调用具体工具, 而普通执行路径通过 registry `execute()` 分发. 这不是注册表提供了两套不同映射: 两者最终都使用同一个 `_tools[name]`; 区别只在上层是否需要先检查具体工具身份或执行额外逻辑. 

## 2 Shell 工具: 从模型参数到命令结果

源码：`src/codekeel/tools/shell.py`

### 2.1 概述

完整数据流是：

```text
ToolCall.arguments
        │
        ▼
ShellTool.validate_arguments()
  ├─ 必须恰好包含 command
  ├─ command 必须是 str
  └─ command 不能全空白
        │
        ▼
ShellConfig.validate_command()
  ├─ NUL / OS 编码检查
  ├─ regex deny_patterns
  ├─ shlex.split()
  └─ allowed 或 denied executable 检查
        │
        ├─ ShellConfig.filtered_environment(os.environ)
        ▼
Workspace.execute(command, cwd, env, timeout, inherit_env=False)
        │
        ▼
CommandResult(stdout, stderr, exit_code, timed_out)
        │
        ▼
_format_result() + 可选 _truncate_tail()
        │
        ▼
ToolResult(content, is_error)
```

### 2.2 `ShellConfig`：冻结且规范化的配置

源码：`src/codekeel/tools/shell.py:51-109`

`ShellConfig` 是 `@dataclass(frozen=True, slots=True)`，字段分成四组：

| 类别           | 字段                | 默认值                 |
| -------------- | ------------------- | ---------------------- |
| 可执行文件规则 | `allowed_commands`  | `()`                   |
| 可执行文件规则 | `denied_commands`   | `None`，随后推导默认值 |
| 整条命令规则   | `deny_patterns`     | `()`                   |
| 执行资源       | `timeout`           | `30.0` 秒              |
| 输出资源       | `max_output_bytes`  | `50_000` 字节          |
| 环境变量       | `env_allowlist`     | `()`                   |
| 环境变量       | `env_deny_patterns` | 默认敏感变量模式       |
| 工作目录       | `cwd`               | `"."`                  |

冻结 dataclass 并不意味着可以跳过输入校验。`__post_init__()` 会验证字段并通过 `object.__setattr__()` 写回规范化值；完成构造后，调用方不能再就地修改配置。
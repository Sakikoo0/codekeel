# 核心协议与数据模型

## 1. 模型与消息协议

### 1.1 概述

`codekeel/model/base.py` 定义了模型边界两侧唯一认可的通用语言: Agent 只向 `Model` 传递 `Message` 和 `ToolDefinition`, `Model` 只返回 `ModelResponse`.

```text
Agent
  │  messages: list[Message]
  │  tools: list[ToolDefinition] | None
  ▼
Model.complete(...)
  │
  └─> ModelResponse
        ├─ content: str | None
        ├─ tool_calls: list[ToolCall]
        └─ usage: Usage
```

### 1.2 核心对象

#### `ToolDefinition`: 模型可选择什么

源码: `src/codekeel/models/base.py`

| 字段          | 类型             | 含义                                 |
| ------------- | ---------------- | ------------------------------------ |
| `name`        | `str`            | 工具名称; 空串和全空白名称会被拒绝   |
| `description` | `str`            | 给模型看的能力说明                   |
| `parameters`  | `dict[str, Any]` | 独立于具体大模型提供商的 JSON Schema |

它是给模型看的工具规格, 不是可执行工具本身. `Tool.definition()` 产生它, `Model.complete(..., tools=...)` 消费它; **工具实现和执行上下文不会进入模型协议 (有清晰的安全边界, 模型与运行环境解耦).**

#### `ToolCall`: 模型想执行什么

源码: `src/codekeel/models/base.py`

| 字段        | 类型             | 含义                            |
| ----------- | ---------------- | ------------------------------- |
| `id`        | `str`            | 本次调用的关联 ID, 至少一个字符 |
| `name`      | `str`            | 目标工具名, 至少一个字符        |
| `arguments` | `dict[str, Any]` | 已解析的参数对象; 默认空字典    |

#### `ToolResult`: 工具实际返回什么

源码: `src/codekeel/models/base.py`

| 字段       | 类型   | 含义                                   |
| ---------- | ------ | -------------------------------------- |
| `content`  | `str`  | 规范化后的工具输出                     |
| `is_error` | `bool` | 工具是否以可报告错误结束, 默认 `False` |

`is_error=True` 表示一次已经被正常捕获并可反馈给模型的失败, 不等价于 Python 异常. Agent 会据此写入 `ToolFailed` 或 `ToolCompleted` 事件; 进入对话历史时, 完整 `ToolResult` 会被 `model_dump_json()` 编码进 tool message 的 `content`, 从而保留错误标记.

#### `Message`: 统一表示四种对话消息

源码: `src/codekeel/models/base.py`

| 字段           | 类型             | 默认值   | 用途                     |
| -------------- | ---------------- | -------- | ------------------------ | ------------------------ | ---- | ----------------- |
| `role`         | `system          | user     | assistant                | tool`                    | 必填 | 消息来源/协议角色 |
| `content`      | `str             | None`    | `None`                   | 文本内容                 |
| `tool_calls`   | `list[ToolCall]` | 新空列表 | assistant 发起的工具调用 |
| `tool_call_id` | `str             | None`    | `None`                   | tool 结果所对应的调用 ID |

一个 `Message` 通过不同的字段组合去表达普通文本, assistant tool call 和 tool result 等多种消息, 例如:

```python
# 普通文本
Message(role="system", content="You are a coding agent.")
# 普通文本
Message(role="user", content="Fix the parser.")
# assistant 请求工具
Message(
    role="assistant",
    content=None,
    tool_calls=[ToolCall(id="call-1", name="shell", arguments={"command": "pytest"})],
)
# 工具执行结果
Message(
    role="tool",
    content='{"content":"1 passed","is_error":false}',
    tool_call_id="call-1",
)
```

没有建立庞大的消息继承树. 这是一种宽松容器, 即**当前 Message 主要检查字段类型, 不检查字段组合是否符合业务语义.** 因此有些对象虽然业务上不合理, 却可以通过 Pydantic 校验, 例如:

```python
# user 不应该发起工具调用
Message(
    role="user",
    content="hello",
    tool_calls=[ToolCall(...)],
)
```

有些规则不能由单个 Message 验证. 规则可以分成两类:

- 单消息规则: 只看一个对象就能判断的规则, 例如: user 不能带 tool_calls; system 不能带 tool_calls; assistant 不能带 tool_call_id; tool 必须带 tool_call_id; tool 不能带 tool_calls等. 这些完全可以放进 Message 的 validator.

- 跨消息规则: 必须查看整段历史. 例如判断下面的历史是否合法:

  ```python
  [
      Message(
          role="assistant",
          tool_calls=[ToolCall(id="call-1", ...)],
      ),
      Message(
          role="tool",
          tool_call_id="call-2",
          content="result",
      ),
  ]
  ```

  单独看两条消息, 它们都可能合法; 放在一起才发现 ID 对不上. 因此这类规则必须由历史级验证器处理, 例如当前 `_turns(messages)` 所承担的职责.

**在后续的实现中, 我将逐步转为完整的消息继承树.**

#### `Usage`: 一次或多次模型调用的资源值

源码: `src/codekeel/models/base.py`

| 字段            | 类型    | 约束                      |
| --------------- | ------- | ------------------------- |
| `input_tokens`  | `int`   | 默认 `0`, 必须 `>= 0`     |
| `output_tokens` | `int`   | 默认 `0`, 必须 `>= 0`     |
| `cost`          | `float` | 默认 `0.0`, 必须 `>= 0.0` |

#### `ModelResponse`: provider 响应的统一出口

源码: `src/codekeel/models/base.py`

| 字段         | 类型             | 说明                      |
| ------------ | ---------------- | ------------------------- |
| `content`    | `int`            | 可选文本, 默认值为 `None` |
| `tool_calls` | `list[ToolCall]` | 默认为空列表              |
| `Usage`      | `Usage`          | 默认构造 `Usage`          |

常见响应有两种:

```python
# 最终文本
ModelResponse(content="Fixed the parser.", usage=Usage(output_tokens=4))

# 请求工具
ModelResponse(
    tool_calls=[ToolCall(id="call-1", name="shell", arguments={"command": "pytest"})],
    usage=Usage(input_tokens=120, output_tokens=16),
)
```

#### `Model`: 核心唯一依赖的模型行为

源码: `src/codekeel/models/base.py`

```python
class Model(Protocol):
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse: ...
```

这是结构化 `Protocol`, 实现类无需继承它; 只要方法签名兼容即可被静态类型系统视为 `Model`. 协议只规定一次异步补全, 不规定模型名、重试、序列化、provider 配置或 SDK 响应类型. 这些属于装配层(adapter).

`tools=None` 与 `tools=[]` 在协议上是两个可区分的输入: 前者表示调用方未提供工具集合, 后者表示明确提供空集合. 具体 adapter 决定是否把字段发送给 provider.

### 1.3 `FakeModel` 与 `ScriptedModel` 如何驱动测试

源码: `src/codekeel/models/fake.py`

`ScriptedModel` 在构造时把任意 `Iterable[ModelResponse]` 装入 `deque`, 每次 `complete()` 从左侧弹出一个响应. 它不读取 `messages`, 也不解释 `tools`, 测试结果完全由脚本顺序决定:

```python
model = FakeModel([
    ModelResponse(
        tool_calls=[ToolCall(id="1", name="shell", arguments={"command": "pytest"})]
    ),
    ModelResponse(content="Done"),
])

first = await model.complete(messages)   # 工具调用
second = await model.complete(messages)  # 最终回答
```

`FakeModel` 只是 `ScriptedModel` 的无改动子类, 提供更符合测试语境的名字. 这两者的行为边界很小.

## 2. Workspace 返回值与协议

### 2.1 概述

**`src/codekeel/workspace/models.py` 定义跨 backend 通用的操作结果, `src/codekeel/workspace/base.py` 则定义 Agent runtime 可以要求 Workspace 完成哪些事情.** 工具依赖 `Workspace` 协议, 不需要知道实际对象是在宿主机工作, 还是通过其他隔离环境工作.

在 coding agent 语境里, 可以严谨地将 Workspace 定义为: **Workspace 是 Agent 面向任务环境的受控能力边界. 它封装 Agent 可以观察和改变的文件状态、命令执行环境及相关资源生命周期, 并把底层实现转换成稳定、与具体运行环境无关的接口.** Agent 只能通过系统授予它的能力改变环境.

它不是单纯的 "项目目录", 也不等同于 "Docker 容器". 可以把 Workspace 抽象为：

```
Workspace
  ├─ 状态范围: Agent 能看见或修改哪些文件
  ├─ 执行范围: 命令在哪里、以什么环境运行
  ├─ 隔离规则: 哪些路径和宿主资源不可访问
  ├─ 结果模型: 操作结果如何统一表示
  └─ 生命周期: 相关进程、容器或连接如何释放
```

从概念上看, 一个 Workspace 至少包含四方面:

1. 任务状态的命名空间

   Workspace 确定 Agent 操作的 "世界":
   - "src/app.py" 相对于哪个根目录
   - "." 表示什么
   - 哪些文件属于当前任务
   - 路径解析后是否仍然位于允许范围内

   因此 Workspace 不只是保存一个 root path, 还负责解释路径.

2. 副作用边界

   Agent 的读文件写文件和运行命令都会产生或观察真实副作用. Workspace 决定这些副作用落在哪里. Agent core 不应该绕过 Workspace 直接调用 `open()`, `subprocess` 或 Docker API. 否则 Workspace 就无法继续作为统一的安全与架构边界.

   > 函数的副作用: 返回值是函数明确声明的主要结果, 不叫副作用. 副作用的副表示, 它发生在正常输入输出关系之外. 例如:
   >
   > ```python
   > def save_note() -> None:
   >       with open("note.txt", "w") as file:
   >           file.write("hello")
   > ```
   >
   > 它虽然返回 None, 但执行后磁盘发生了变化:
   >
   > - 执行前: `note.txt` 不存在
   > - 执行后: `note.txt` 存在, 内容为 hello
   >
   > 这个文件变化就是副作用. 副作用的关键不是返回了什么, 而是**函数执行以后, 函数外部的世界是否发生了可观察变化.**

3. Backend 的抽象

   相同的 Workspace 概念可以有不同实现:
   - 本地目录

   - Docker 容器

   - 远程虚拟机

   - 临时云端沙箱

   - 测试用内存或 fake workspace

   这些实现的基础设施完全不同, 但上层 Agent 希望执行的动作仍然是 "读文件", "写文件", "运行命令". 所以 Workspace 是概念, LocalWorkspace, DockerWorkspace 等是这个概念的具体实现.

4. 一组能力, 而不是一份数据

   Workspace 对象更像一个受控能力句柄:

   ```
   workspace.read_file(...)
   workspace.write_file(...)
   workspace.execute(...)
   ```

   持有它就意味着获得了一组操作任务环境的能力. 这也是为什么它比裸路径更有意义. 传递一个 `Path` 只告诉调用方 "文件在哪里"; 传递一个 Workspace 则同时限制调用方 "可以如何访问这个环境".

**Workspace 与几个相邻概念需要严格区分**:

- Workspace 不等于 repository: 仓库是任务数据, Workspace 是访问和操作这些数据的边界.
- Workspace 不等于 sandbox: sandbox 强调隔离和安全; Workspace 可以由 sandbox 实现, 但本地 Workspace 未必具备强隔离.
- Workspace 不等于 tool: 工具面向模型描述某项能力, Workspace 则为工具提供底层文件和执行能力.
- Workspace 不等于 Agent state: Agent state 记录步骤、消息、用量和运行状态; Workspace 保存或暴露任务环境的外部状态.
- Workspace 不等于 working directory: 工作目录只是命令执行位置, 是 Workspace 管理的一部分.

### 2.2 三个返回值类型的共同设计

源码: `src/codekeel/workspace/models.py`

三个结果都是 `@dataclass(frozen=True, slots=True)`:

- `frozen=True` 阻止调用方就地改写字段, 降低执行结果在层间传递时被意外篡改的可能
- `slots=True` 固定实例属性集合, 减少实例开销, 也会阻止随手挂载未声明属性
- 它们是标准库 `dataclass`, 不是 `Pydantic` 模型, 不承担输入解析或运行时强校验

#### `CommandResult`: 命令结束状态与两条输出流

| 字段/属性     | 类型     | 含义                         |
| ------------- | -------- | ---------------------------- |
| `stdout`      | `str`    | 标准输出                     |
| `stderr`      | `str`    | 标准错误                     |
| `exit_code`   | `int`    | 进程退出码                   |
| `timed_out`   | `bool`   | 是否因超时结束, 默认 `False` |
| `output`      | 只读属性 | `stdout + stderr` 的兼容视图 |
| `return_code` | 只读属性 | `exit_code` 的兼容别名       |

`return_code` 与 `exit_code` 指向同一个值, 是字段名兼容层, 而不是第二套状态.

#### `FileResult`: 一次读写的内容结果

| 字段        | 类型   | 含义                                     |
| ----------- | ------ | ---------------------------------------- |
| `path`      | `str`  | 本次操作使用的逻辑路径                   |
| `content`   | `str`  | 读出的文本或成功写入的文本               |
| `is_binary` | `bool` | 读取结果是否被识别为二进制, 默认 `False` |

读和写共用一个小对象. 读文本时 `content` 是完整 UTF-8 内容; 写入成功时它回显写入内容.

`is_binary=True` 时当前实现返回空 `content`. 这让上层能拒绝把二进制数据当文本处理, 但无法区分"内容为空的二进制文件"和"二进制内容具体是什么".

#### `FileInfo`: 逻辑路径与规范路径之间的桥

| 字段             | 类型   | 默认值  | 含义                               |
| ---------------- | ------ | ------- | ---------------------------------- |
| `path`           | `str`  | 必填    | 调用者视角的规范化逻辑路径         |
| `canonical_path` | `str`  | 必填    | 解析后, 相对 workspace root 的路径 |
| `exists`         | `bool` | 必填    | 目标当前是否存在                   |
| `is_directory`   | `bool` | `False` | 存在的目标是否为目录               |
| `size`           | `int`  | `0`     | backend 报告的字节级元数据         |

这是本阶段最关键的对象. 假设 workspace 内 `alias.py` 是指向 `src/app.py` 的符号链接, 概念上可能得到:

```python
FileInfo(
    path="alias.py",
    canonical_path="src/app.py",
    exists=True,
    is_directory=False,
    size=1024,
)
```

`path` 回答 "用户用哪个 workspace 路径提出请求", `canonical_path` 回答 "这个路径解析后实际指向 workspace 内哪里". 文件工具正是先调用 `inspect_path()`, 再对 `canonical_path` 做访问授权; 如果只检查原始 `path`, 允许路径可能借助别名或符号链接落到受保护目标.

不存在的目标也会返回 `FileInfo`, 而不是由 `inspect_path()` 抛 `FileNotFoundError`. 其 `canonical_path` 仍可用于写入前授权和父目录安全判断. 这解释了 `write_file` 工具为何能先检查尚未创建的路径, 再执行创建.

需要注意三个默认值陷阱:

- `exists=False, is_directory=False` 不表示它是文件, 只表示不存在时没有目录事实;
- `size=0` 既可能是不存在, 也可能是真正的空文件, 判断存在性必须看 `exists`.
- 目录也会携带 backend 给出的 `size`, 不要把它解释为目录下文件内容的总大小.

### 2.3 Workspace 协议中的方法

#### `execute()`

```python
async def execute(
    command: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = 30.0,
    inherit_env: bool = True,
) -> CommandResult
```

- `command` 是要交给 shell 的字符串, 即要执行的 shell 命令
- `cwd` 是 current working directory, 即命令执行时的当前工作目录. `cwd=None` 表示使用 backend 默认工作目录; 非空路径仍必须受 workspace 边界约束. 例如 Workspace root 是 `/project`, 执行 `await workspace.execute("pytest", cwd="tests")` 逻辑上相当于在 `/project/tests` 中运行 `pytest`.
- `env` 表示传给命令的环境变量. `env` 使用只读形状的 `Mapping`, 允许普通 dict.
- `inherit_env` 这个参数决定是否保留执行环境原有的环境变量, 把 "在调用者环境上覆盖" 与 "从空环境开始" 显式区分. 安全敏感调用会传 `False` 和最小化 env.
- 返回非零退出码或超时是正常的 `CommandResult`, 不是必然抛异常; 而非法 cwd、无法启动等基础设施/输入失败可以抛异常.

#### `read_file()`

接收 workspace 相对逻辑路径, 返回完整 `FileResult`. 协议文字把它限定为 UTF-8 文本读取; 当前返回模型又允许用 `is_binary` 报告检测到的二进制文件.

#### `write_file()`

接收逻辑路径和完整字符串, 返回写入后的 `FileResult`. 这是全量写入/覆盖接口, 不表达 append、原子替换、权限位或 compare-and-swap. 父目录是否自动创建属于当前实现行为, 而不是方法签名中显式建模的选项.

#### `inspect_path()`

它把路径解析和元数据查询合成一次安全前置操作：

```text
用户提供的逻辑路径
        │
        ▼
解析且限制在 workspace 内
        │
        ▼
FileInfo(path, canonical_path, exists, is_directory, size)
        │
        ├─ canonical_path -> 授权/别名检查
        ├─ exists/type    -> 操作前置条件
        └─ size           -> 读取与搜索预算
```

它并非只为 "看看存不存在". 源码中的实际调用还用它完成: 受保护路径授权、拒绝符号链接别名、确认恢复时 workspace root 仍是 canonical 目录、读取前大小过滤, 以及输出 artifact 写入前的路径一致性检查.

#### `list_directory()`

返回 `list[FileInfo]`, 不是只有名字的字符串列表, 因此列表结果同样携带 canonical path、类型和大小. `recursive=False` 只列直接成员; `True` 请求递归成员. 协议返回具体 `list`, 说明调用方可以依赖结果已经物化, 而不是一次性迭代器.

此方法的职责是列出 contained entries; “隐藏文件是否展示”“允许哪些路径”“最多显示多少条”仍可由工具层继续过滤. 当前实现会跳过解析失败或逃逸 workspace 的条目, 所以列表可能不是底层目录枚举的逐项镜像.

#### `close()`

这是生命周期协议, 用来释放 Workspace 拥有的资源. 对无持久资源的实现可以是空操作; 拥有外部资源的实现则在这里清理. 调用方应把它当作需要 `await` 的正常生命周期步骤, 并允许实现把重复关闭做成安全操作; 后一点在当前有状态 backend 中被明确实现, 但未由类型签名强制.

## 3. Tool 协议

### 3.1 概述

`src/codekeel/tools/base.py` 没有实现任何具体工具, 而是固定了所有工具共同遵守的最小边界: 工具用 `definition()` 向模型公开能力, 用异步 `execute()` 接收已经被选中的参数和本次运行的上下文, 最后返回统一的 `ToolResult`.

```text
                  模型可见                         模型不可见
            ┌──────────────────┐           ┌─────────────────────┐
Tool ──────>│ ToolDefinition   │           │ ToolContext         │
definition()│ name/description │           │ workspace/run_id    │
            │ parameters       │           │ callbacks/运行选项   │
            └────────┬─────────┘           └──────────┬──────────┘
                     │                                │
                     ▼                                ▼
                  ToolCall ─────────────────────> execute()
                  name + arguments                      │
                                                        ▼
                                                   ToolResult
```

这张图中最重要的是左右边界: `ToolDefinition` 是给模型的能力说明, `ToolContext` 是 runtime 给工具的可信依赖.

- `Tool` 首先通过 `definition()` 向模型介绍自己, 一个具体工具既有真正的 Python 实现, 也有一份给模型看的使用说明. `ToolDefinition` 不是工具本身. 它不包含 `execute()`, 也不能访问文件系统, 只是一份能力目录.

- 模型根据 `ToolDefinition` 生成 `ToolCall`

- `ToolCall.arguments` 由模型控制, 因此仍是不可信输入

- `ToolContext` 中的回调也是受限能力. Agent 没有把 "自己" 整个交给工具, 而是只给工具开放几个特定操作入口, 即**窄回调**.

- `ToolContext` 由 runtime 创建, 而不是由模型提供

- `execute()` 汇合两侧信息

  ```python
  async def execute(
      arguments: dict[str, Any], # 模型侧
      context: ToolContext, # runtime 侧
  ) -> ToolResult:
  ```

- 最后统一返回 `ToolResult`

依赖方向则是:

```text
Agent ──> ToolRegistry ──> Tool Protocol ──> Workspace Protocol
                                  │
                                  ├─> ToolDefinition / ToolResult
                                  └─> Plan（仅用于 update_plan 回调类型）
```

### 3.2 `Tool`: 声明与执行的二段式协议

源码: `src/codekeel/tools/base.py`

```python
class Tool(Protocol):
    name: str
    description: str

    def definition(self) -> ToolDefinition: ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...
```

它和 `Model`, `Workspace` 一样是结构化 `Protocol`: 具体工具无需继承共同基类, 只要公开兼容的属性和方法即可. 协议没有使用 `@runtime_checkable`, 所以它主要用于类型检查和依赖约束, 而不是运行时通过 `isinstance()` 认证工具.

#### `name` 与 `description`

二者是工具对象自身的元数据. 它们同时通常会出现在 `definition()` 返回的 `ToolDefinition` 中, 但协议本身不会证明两处永远一致. 源码把一致性检查留给注册阶段: 注册表会调用 `definition()`, 若 `definition.name != tool.name` 就拒绝注册.

`description` 也存在于两处, 但当前注册逻辑不校验两者相等. 具体工具都用 `self.description` 构造定义, 因此正常实现不会分叉; 从协议强度看, 这仍是一项约定, 而非被 runtime 强制的不变量.

#### `definition()`: 模型侧契约

`definition()` 返回 `ToolDefinition(name, description, parameters)`, 回答的是:

- 工具叫什么
- 它适合做什么
- 参数对象应符合什么 JSON Schema

它必须是同步且无上下文参数的方法, 因为工具定义会在某次具体调用之前被收集并发送给模型. 返回值只描述能力, 不包含可执行函数、Workspace 或本次运行状态.

需要特别区分 "声明参数规则" 和 "执行参数校验". 例如 `ShellTool.definition()` 声明只接收必填字符串 `command`, 但真正执行前仍由 `validate_arguments()` 检查缺失字段、额外字段、类型和空白字符串. `Tool` 协议及 `ToolDefinition` 不会自动拿 JSON Schema 校验 `arguments`.

这意味着一个实现正确的工具需要保持两件事一致:

```text
definition().parameters 所宣称的输入 == execute() 实际接受并验证的输入
```

如果二者漂移, 模型可能生成 "schema 合法, 实现拒绝" 的参数, 或者调用未向模型公开的隐藏参数. 基础协议刻意保持很小, 没有引入统一 schema validator; 一致性由具体工具实现负责.

#### `execute()`: runtime 侧契约

`execute()` 接收两个对象:

- `arguments`: 模型的 `ToolCall.arguments`, 已经是 `dict[str, Any]`, 但不代表内容已经可信或符合 schema
- `context`: runtime 构造的 `ToolContext`, 提供工具真正执行所需的可信能力

方法是异步的, 因为 Workspace 操作和 Explorer 委托都可能等待 I/O. 它把不同底层结果归一化为 `ToolResult(content, is_error)`; 例如命令工具会把 `CommandResult` 格式化成文本, 文件工具会把读取或写入结果转成适合模型消费的内容.

provider 已把参数解析成字典, 而参数形状仍须由工具在副作用发生前验证. 当前注册表只复制参数字典并分发, 不承担通用 schema 校验.

### 3.3 `ToolContext`: 可信能力注入

源码: `src/codekeel/tools/base.py`

`ToolContext` 是 `@dataclass(frozen=True, slots=True)`: 字段在构造后不能重新绑定, 也不能动态增加属性. 它是每次运行的依赖容器, 而不是模型参数的一部分.

| 字段                  | 类型                                    | 默认值  | 作用                                     |
| --------------------- | --------------------------------------- | ------- | ---------------------------------------- | -------------------------------- |
| `workspace`           | `Workspace`                             | 必填    | 所有命令和文件副作用的唯一入口           |
| `run_id`              | `str`                                   | 必填    | 标识当前运行, 供运行级输出或审计关联使用 |
| `defer_output_limits` | `bool`                                  | `False` | 告知工具是否把输出限制推迟给上层统一处理 |
| `update_plan`         | `Callable[[Plan], None]                 | None`   | `None`                                   | 同步、整份替换计划的运行级回调   |
| `delegate_explore`    | `Callable[[str], Awaitable[ToolResult]] | None`   | `None`                                   | 异步委托只读探索并取回结果的回调 |

#### `workspace`

工具拿到的是 `Workspace` 协议, 而不是具体 backend. 文件工具通过它 `inspect / read / write / list`, Shell 工具通过它 `execute`. 这样工具层可以表达 "需要执行命令" 或 "需要读取文件", 却不能绕开 Workspace 自己选择宿主进程、容器或其他实现.

这里的架构约束不是 "工具绝不产生副作用", 而是 "工具的仓库副作用必须沿 Workspace 边界发生". 因此切换 Workspace 实现时, `Tool` 的签名和核心调度不需要改变.

#### `run_id`

`__post_init__()` 要求 `run_id` 必须是字符串且去除空白后非空.

#### `defer_output_limits`

它是一次调用的协作信号，不是自动生效的拦截器。当前 `ShellTool` 会读取它：

- `False`：工具自身按 `max_output_bytes` 截断；
- `True`：跳过这次工具内截断，由 Agent 的统一输出管理阶段处理。

Agent 创建主运行上下文时把它设为 `True`；Explorer 创建上下文时保留默认 `False`，之后还会对返回报告做自己的有界压缩。这避免主路径发生不必要的双重裁剪，同时让独立使用工具或较窄 runtime 时仍有本地输出上限。

但该字段本身不会强迫任意第三方工具遵守限制。它表达的是 runtime 与工具间的约定；真正的上限仍由读取该字段的工具或调用后的输出管理器落实。

#### `update_plan`

这是同步回调，因为计划更新只修改当前 runtime 拥有的内存状态并记录对应变化，不需要 Tool 自己取得 Agent 实例。`UpdatePlanTool` 先把不可信参数解析为受约束的 `Plan`，再调用 `context.update_plan(plan)`。

回调是可选的。未配置时，计划工具不会抛出属性错误，而是返回 `is_error=True` 的 `ToolResult`。因此同一个工具对象可以被装配到不同 runtime，能力是否真正可用由本次运行上下文决定。

#### `delegate_explore`

Explorer 委托可能进行多轮模型和工具调用，所以回调返回 `Awaitable[ToolResult]`，`DelegateExploreTool.execute()` 会等待它完成。它只暴露一个自包含任务字符串和最终归一化结果，不把父 Agent、子对话或 Explorer 实例直接交给工具。

它同样是可选能力；缺失时工具返回普通错误结果。主 Agent 注入运行级委托回调，而 Explorer 自身创建的 `ToolContext` 不注入该回调，由此在上下文层面切断递归委托能力。

### 3.4 四类边界错误

源码: `src/codekeel/tools/base.py`

```text
ValueError
└─ ToolRegistryError
   ├─ DuplicateToolError
   ├─ UnknownToolError
   └─ ToolArgumentsError
```

| 异常                 | 表示的边界问题                   | 当前产生位置                     |
| -------------------- | -------------------------------- | -------------------------------- |
| `ToolRegistryError`  | 工具注册或分发请求无效的共同基类 | 注册时名称与 definition 不一致   |
| `DuplicateToolError` | 新注册会覆盖已有同名工具         | `ToolRegistry.register()`        |
| `UnknownToolError`   | 调用名称未注册, 不能分发         | `ToolRegistry.get()`             |
| `ToolArgumentsError` | 调用参数不符合工具承诺的形状     | Shell/文件工具的参数检查辅助函数 |

全部继承 `ValueError`, 说明它们描述的是工具协议输入或装配不成立, 而不是 "命令退出码非零" 这类领域执行结果。

## 4. 运行状态与预算

### 4.1 概述

这一组源码把 "运行到哪里了" "已经花了多少资源" "现在是否必须停止" 拆成三个对象:

```text
BudgetLimits                  AgentState
宿主配置的不可变上限          运行中持续变化的事实
       │                         │
       └──────────┬──────────────┘
                  ▼
        TerminationPolicy.evaluate()
                  │
                  ├─ None: 尚未触及终止线
                  └─ RunStatus: 按固定优先级停止
```

- `BudgetLimits` 只描述上限, 不保存已用量
- `AgentState` 保存消息、生命周期状态、usage 和各类计数
- `TerminationPolicy` 不修改状态, 只根据前两者和单调时钟返回判定
- 调用方负责在动作前检查、在资源消耗发生时记账，并把返回的终止状态写回 `AgentState.status`

### 4.2 `BudgetLimits`: 宿主给一次 run 划定的边界

源码: `src/codekeel/runtime/budgets.py`

| 字段                |   默认值 | 约束对象                              |
| ------------------- | -------: | ------------------------------------- |
| `max_steps`         |    `500` | 主 Agent step 与 Explorer step 的总数 |
| `max_model_calls`   |   `None` | 所有计入账本的模型调用                |
| `max_tool_calls`    |   `None` | 主工具、Explorer 工具和验证命令的总数 |
| `max_wall_time`     | `3600.0` | 整次 run 的单调时钟耗时, 单位为秒     |
| `max_input_tokens`  |   `None` | 累计输入 token                        |
| `max_output_tokens` |   `None` | 累计输出 token                        |
| `max_cost`          |   `None` | 累计费用                              |

**`None` 的统一含义是关闭该单项限制**. 默认配置仍保留有限的 `max_steps=500` 和 `max_wall_time=3600.0`, 所以即使宿主没有配置 token、cost 或调用次数, 默认 Agent 也不会被设计成可以无限循环.

所有非空上限都必须严格大于零, 因而不能用 `0` 表示 "禁止使用". 彻底禁止某种能力应通过装配或策略表达, 不应伪装成一个已经耗尽的正数预算.

该模型配置了：

- `frozen=True`: 构造后不能就地调整限额; 恢复或审计时可把它当作稳定配置值
- `strict=True`: 拒绝依赖宽松的字符串到数字等隐式转换
- `allow_inf_nan=False`: 浮点 wall time 和 cost 不能使用 `NaN` 或正负无穷规避正常比较

### 4.3 `RunStatus`: 生命周期结果, 不只是成功与失败

源码: `src/codekeel/agent/state.py`

`RunStatus` 是字符串枚举, 既方便代码用枚举身份判断, 也能以稳定字符串进入 JSON. 十二个状态可按语义分成四组:

| 类别             | 状态                                                                                 | 含义                                    |
| ---------------- | ------------------------------------------------------------------------------------ | --------------------------------------- |
| 尚未/正在运行    | `IDLE`、`RUNNING`                                                                    | 默认未启动; 主循环可继续推进            |
| 暂停等待外部决定 | `WAITING_FOR_APPROVAL`                                                               | 有 pending action, 当前不能继续自主执行 |
| 正常结束         | `COMPLETED`                                                                          | 已满足当前运行的完成条件                |
| 受控终止         | `VERIFICATION_FAILED`、`MAX_STEPS`、`MAX_COST`、`MAX_TOKENS`、`TIMEOUT`、`CANCELLED` | 因验证、预算、期限或取消而停止          |
| 非预期失败       | `FAILED`                                                                             | 运行异常或基础设施故障导致失败          |

这些名字并不是完整的状态转换规则; `state.py` 只定义词汇, 真正的迁移由 Agent 编排. 尤其要避免三种误读:

1. `WAITING_FOR_APPROVAL` 不是终态成功或失败, 而是需要持久化后由外部决定才能继续的稳定边界.
2. `COMPLETED` 不必然意味着运行过 verification; 未配置验证策略时, 最终文本可以直接完成, 此时 `verification_passed` 仍为 `None`.
3. `MAX_STEPS` 是当前实现复用的 "调用类预算耗尽" 状态: 不仅 step 上限, `max_model_calls` 和 `max_tool_calls` 命中后也返回它. 要知道究竟是哪一项用尽, 不能只看状态名, 还要对照 limits 与 counters.

`CANCELLED` 与 `FAILED` 也不由 `TerminationPolicy.evaluate()` 返回. 前者来自外部取消, 后者来自异常路径; 终止策略只负责时间、费用、token 和次数限制.

### 4.4 `AgentState`: 可序列化的运行事实账本

源码: `src/codekeel/agent/state.py`

`AgentState` 是可变的 Pydantic 模型, 默认值代表一份尚未启动、没有历史、没有消耗的状态:

```python
AgentState(
    messages=[],
    status=RunStatus.IDLE,
    usage=Usage(),
    steps=0,
    model_calls=0,
    tool_calls=0,
    explorer_steps=0,
    explorer_tool_calls=0,
    verification_attempts=0,
    verification_commands=0,
    verification_passed=None,
)
```

`messages` 与 `status` 回答 "当前运行处于什么位置", 其余字段回答 "已经发生了多少工作". 所有数值计数都要求非负; 列表和 `Usage` 使用 `default_factory`, 不同状态实例不会共享可变默认对象.

#### 主 Agent 的三个计数

- `steps`: 主循环真正发起一次主模型响应前增加一次. 它不是 while 循环空转次数, 也不包含上下文总结模型调用.
- `model_calls`: 全局模型请求次数. 主响应、模型总结和 Explorer 模型响应都会进入这个总数, 因此通常有 `model_calls >= steps + explorer_steps`.
- `tool_calls`: 父 Agent 接收并开始处理的模型工具请求数. 计数发生在策略判定前, 所以被 policy 拒绝或转入审批的调用也算一次; 审批恢复后不会为同一动作重复计数.

这三个数字不能互相替代. 一轮主 step 可以返回最终文本而不调用工具, 也可以返回一个工具调用; 上下文总结则会增加 `model_calls`, 但不增加 `steps`.

#### Explorer 的两个计数

- `explorer_steps`: Explorer 发起的模型轮次;
- `explorer_tool_calls`: Explorer 准备处理的工具请求数.

Explorer 使用独立字段是为了保留工作来源, 但并不获得独立的总预算. 终止策略把 `steps + explorer_steps` 与 `max_steps` 比较, 把 Explorer 工具调用并入全局工具调用上限; Explorer 的模型请求也直接增加全局 `model_calls`.

这形成 "分栏记账, 合并限额" 的设计: 观察时能知道资源花在主任务还是探索上, 限制时又不能靠委托子代理绕开父预算.

#### 验证的三个字段

- `verification_attempts`: 启动过多少轮完整验证 suite
- `verification_commands`: 实际开始处理过多少条验证命令
- `verification_passed`: `None` 表示尚无已完成结论或当前 attempt 尚未形成结论, `False` 表示最近完成的验证失败, `True` 表示通过.

attempt 与 command 必须分开, 因为一次 attempt 可以包含多条命令, 也可能在 wall time 或工具总预算耗尽时中途停止. 验证重试会再次执行完整 suite, 所以两个数字不会天然相等.

验证次数不属于 `BudgetLimits`, 它由 verification 配置自己的 `max_verification_attempts` 约束; 但每一条验证命令仍与主工具和 Explorer 工具共享 `max_tool_calls`, 并共享整次 run 的 wall time. 验证不会增加 `steps`, 也不会增加 `tool_calls` 原字段, 而是通过 `verification_commands` 单独记录后在判定时合并.

#### `add_usage()` 为什么重新赋值

`add_usage()` 只有一行：

```python
self.usage = self.usage + usage
```

`Usage.__add__()` 逐项相加并创建新的 `Usage`, 所以累计不是原地修改三个字段. 这保持了 "旧 usage 值" 和 "新增 usage 值" 的边界, 也复用了 `Usage` 对非负 token/cost 的构造校验.

### 4.5 三类计数如何映射到两类共享预算

| 上限                | 实际已用量表达式                                                             | 不计入的相关工作                |
| ------------------- | ---------------------------------------------------------------------------- | ------------------------------- |
| `max_steps`         | `state.steps + state.explorer_steps`                                         | 总结模型调用、验证命令          |
| `max_model_calls`   | `state.model_calls`                                                          | 工具调用                        |
| `max_tool_calls`    | `state.tool_calls + state.verification_commands + state.explorer_tool_calls` | 模型调用                        |
| `max_input_tokens`  | `state.usage.input_tokens`                                                   | 无; 所有已记账模型响应共享      |
| `max_output_tokens` | `state.usage.output_tokens`                                                  | 无; 所有已记账模型响应共享      |
| `max_cost`          | `state.usage.cost`                                                           | 无; 所有已记账模型响应共享      |
| `max_wall_time`     | `clock() - started_at`                                                       | 没有子流程能获得独立 wall clock |

一个简化例子: 主 Agent 已进行 2 个模型 step, 其中 1 次调用工具; 期间做了 1 次总结; Explorer 做了 3 个模型 step 和 2 次工具调用; 验证执行了 2 条命令. 此时:

```text
steps budget usage      = 2 + 3 = 5
model-call budget usage = 2 + 1 + 3 = 6
tool-call budget usage  = 1 + 2 + 2 = 5
```

如果只看 `tool_calls == 1` 就判断工具预算仍很充裕, 会漏掉 Explorer 与 verification 的真实消耗. 相反, 展示分栏原始字段又能解释总数由谁贡献.

### 4.6 `TerminationPolicy`: 固定优先级的纯判定

源码: `src/codekeel/agent/termination.py`

它是 `@dataclass(frozen=True, slots=True)`, 包含两个依赖: 不可变的 `budgets` 与可注入的 `clock`. 默认时钟为 `time.monotonic`, 适合测量持续时间, 不受系统日期或时区调整影响; 测试或嵌入方也可注入确定性时钟.

`evaluate()` 按以下固定顺序返回第一个命中的状态:

```text
wall time
  → cost
  → input tokens
  → output tokens
  → steps（主 + Explorer）
  → model calls
  → tool calls（主 + verification + Explorer）
```

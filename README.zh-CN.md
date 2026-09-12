<p align="center">
  <img
    src="./assets/images/codekeel-icon.svg"
    alt="CodeKeel logo"
    width="160"
  />
</p>

<h1 align="center">CodeKeel</h1>

<p align="center">
  一个小巧、可检查、可扩展的编码智能体运行时。
</p>

---

[English](README.md) | [简体中文](README.zh-CN.md)

CodeKeel 将语言模型连接到代码仓库工具和有边界的工作区，并提供把简单的模型工具循环变成可靠编码智能体所需的运行能力：仓库上下文、类型化工具、预算、审批控制、上下文管理、持久化轨迹、断点恢复，以及基于验证的完成判定。

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- Docker，仅在使用 Docker 工作区时需要

### 从源码安装

```bash
git clone https://github.com/Sakikoo0/codekeel codekeel
cd codekeel
uv sync
uv run codekeel --version
```

在 CodeKeel 尚未安装为系统命令时，请在下文命令前加上 `uv run`，例如 `uv run codekeel run ...`。

### 配置模型

CodeKeel 接受 LiteLLM 风格的模型标识，并使用模型提供方的常规环境配置。例如：

```bash
export DEEPSEEK_API_KEY="..."
# export OPENAI_API_KEY="..."
# export ANTHROPIC_API_KEY="..."
```

随后使用对应的 provider/model 标识，例如 `deepseek/deepseek-v4-flash`。

### 运行任务

开始运行前，先创建代码仓库目录和可信状态目录：

```bash
mkdir -p ./demo-repo ./.codekeel-state

codekeel run \
  --repo ./demo-repo \
  --root ./.codekeel-state \
  --model deepseek/deepseek-v4-flash \
  --task "创建 README.md，包含标题、简短的项目介绍和使用方法。"
```

`--repo` 是 Agent 可以检查和修改的代码仓库。`--root` 是 CodeKeel 保存检查点和运行轨迹的可信宿主机目录。将两者分开，可以避免仓库工具把运行时状态当作普通项目文件处理。

`run` 会输出一条 JSON 摘要，包含运行 ID、状态、资源用量、持续时间、验证结果、轨迹路径，以及待审批操作 ID。

### 使用 Docker 工作区

```bash
codekeel run \
  --repo ./demo-repo \
  --root ./.codekeel-state \
  --workspace docker \
  --image python:3.12-slim \
  --model deepseek/deepseek-v4-flash \
  --task "修复失败的测试" \
  --verify "pytest"
```

Docker 工作区是一次性的，并且默认关闭网络。CLI 当前只能恢复已记录的本地工作区。

## CLI 概览

| 命令                                | 用途                         |
| ----------------------------------- | ---------------------------- |
| `codekeel run`                      | 启动新的非交互编码智能体任务 |
| `codekeel inspect RUN_ID`           | 以 JSON Lines 输出运行事件流 |
| `codekeel resume RUN_ID`            | 继续可恢复的本地运行         |
| `codekeel approve RUN_ID ACTION_ID` | 批准当前待处理的精确操作     |
| `codekeel reject RUN_ID ACTION_ID`  | 拒绝当前待处理的精确操作     |

使用 `codekeel COMMAND --help` 查看完整参数。

### 检查和恢复运行

```bash
codekeel inspect RUN_ID --root ./.codekeel-state
codekeel resume RUN_ID --root ./.codekeel-state
```

后续的检查、批准、拒绝和恢复命令必须使用与 `run` 相同的 `--root`。

### 批准或拒绝操作

默认的 `--approval risky` 模式会暂停高风险和未知操作。使用 `--approval always` 可以审核每一个原本允许的操作；使用 `--approval never` 可以非交互运行。所有模式下，硬性拒绝规则和工作区边界始终有效。

当运行返回 `waiting_for_approval` 时，先检查事件，再记录一次精确决定，然后恢复：

```bash
codekeel inspect RUN_ID --root ./.codekeel-state
codekeel approve RUN_ID ACTION_ID --root ./.codekeel-state
# 或：codekeel reject RUN_ID ACTION_ID --root ./.codekeel-state
codekeel resume RUN_ID --root ./.codekeel-state
```

批准或拒绝命令只记录决定；`resume` 才会继续执行。

## 文档

详细设计文档和说明请参考 [docs](docs)。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest
```

涉及 Docker 的改动需要在 Docker daemon 运行时执行：

```bash
uv run pytest -m docker
```

## 评估单个任务

### 前置条件

建议目录结构：

```text
workspace/
├── codekeel-codex/
└── SWE-bench/
```

以下命令假设两个仓库位于同一个父目录.

### 安装官方 SWE-bench evaluator

在公共父目录执行：

```bash
git clone https://github.com/swe-bench/SWE-bench
cd SWE-bench
uv venv
uv pip install -e .
git clone --depth 1 \
  https://github.com/SWE-bench/swe-bench-tasks.git \
  ./swe-bench-tasks
```

检查任务仓库：

```bash
uv run swebench dataset check ./swe-bench-tasks
```

预期输出：

```bash
./swe-bench-tasks looks well formed
```

### 安装 CodeKeel 依赖

```bash
cd ../codekeel-codex
uv sync --group swebench
```

### 配置模型 API Key

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
```

### 固定 SWE-bench 数据集 revision

查询当前 revision：

```bash
uv run python -c \
  "from huggingface_hub import HfApi; print(HfApi().dataset_info('SWE-bench/SWE-bench', revision='main').sha)"
```

为了保证评估可复现，后续命令使用固定 revision：

```bash
export SWEBENCH_REVISION="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"
```

### 查看可选择的任务

```bash
uv run python -c \
 "from datasets import load_dataset; d=load_dataset('SWE-bench/SWE-bench', revision='$SWEBENCH_REVISION', split='test');
print('\n'.join(d['instance_id'][:20]))"
```

选择一个任务：

```bash
export TASK_ID="astropy__astropy-12057"
```

### 转换 SWE-bench 任务

```bash
export DATASET_DIR="swebench/swebench-smoke-1"
uv run codekeel eval-convert-swebench \
  --instance-id "$TASK_ID" \
  --revision "$SWEBENCH_REVISION" \
  --output "$DATASET_DIR"
```

转换完成后，目录大致如下：

```text
swebench/swebench-smoke-1/
├── metadata.json
├── astropy__astropy-12057.yaml
└── repos/
└── astropy__astropy-12057/
```

### 运行 CodeKeel 评估

设置模型：

```bash
export CODEKEEL_MODEL="openrouter/x-ai/grok-4.6"
```

运行完整配置：

```bash
uv run codekeel eval \
  --dataset "$DATASET_DIR" \
  --model "$CODEKEEL_MODEL" \
  --config configs/full.yaml \
  --root swebench/results
```

完成后，CodeKeel 会生成类似下面的运行目录：

```text
swebench/results/<evaluation-id>/<run-id>/
├── results.jsonl
├── patches/
│ └── astropy__astropy-12057.diff
└── .agent/
```

记录包含 results.jsonl 和 patches/ 的目录：

```bash
export CODEKEEL_RUN_DIR="swebench/results/<evaluation-id>/<run-id>"
```

将占位符替换成这次实际生成的目录。

可以确认补丁是否存在：

```bash
ls "$CODEKEEL_RUN_DIR/patches/$TASK_ID.diff"
```

注意：CodeKeel 的 success 或 verification_result 只是本地评估结果，不等同于官方 SWE-bench 的 resolved。

### 生成官方 predictions.jsonl

设置提交给 evaluator 的模型标识。该字段只是公开标签，不能包含 API Key、endpoint 或账户信息。

```bash
export SUBMISSION_MODEL_NAME="codekeel/grok-4.6"
```

生成官方输入：

```bash
uv run codekeel eval-export-swebench \
  --dataset "$DATASET_DIR" \
  --run "$CODEKEEL_RUN_DIR" \
  --model-name "$SUBMISSION_MODEL_NAME" \
  --output "$CODEKEEL_RUN_DIR/predictions.jsonl"
```

检查文件：

```bash
wc -l "$CODEKEEL_RUN_DIR/predictions.jsonl"
head -n 1 "$CODEKEEL_RUN_DIR/predictions.jsonl"
```

单任务评估应生成一行 JSONL，包含：

```json
{
  "instance_id": "astropy\_\_astropy-12057",
  "model_name_or_path": "codekeel/deepseek-v4-flash",
  "model_patch": "<完整补丁文本>"
}
```

### 运行官方 SWE-bench 评估

因为当前位于 SWE-bench/，CodeKeel 生成的预测文件位于相邻仓库：

```bash
cd ../SWE-bench
export PREDICTIONS_FILE="../codekeel-codex/$CODEKEEL_RUN_DIR/predictions.jsonl"
```

执行单任务评估：

```bash
uv run swebench eval full \
  --predictions "$PREDICTIONS_FILE" \
  --instance "$TASK_ID" \
  --run-id "codekeel-${TASK_ID}-001" \
  --workers 1 \
  --task-repo ./swe-bench-tasks
```

可能时间较长，请耐心等待五到十分钟。

### 查看官方结果

查看评估摘要：

```bash
cat "logs/evaluation/codekeel-${TASK_ID}-001/results.json"
```

也可以重新生成报告，无需再次运行容器：

```bash
uv run swebench report "codekeel-${TASK_ID}-001" -d full
```

最终应明确记录以下状态之一：

- resolved：补丁通过官方要求的测试。
- unresolved：补丁成功评估，但没有满足测试要求。
- evaluator error：镜像、补丁应用、测试环境或 evaluator 出错。

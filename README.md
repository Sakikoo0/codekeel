<p align="center">
  <img
    src="./assets/images/codekeel-icon.svg"
    alt="CodeKeel logo"
    width="160"
  />
</p>

<h1 align="center">CodeKeel</h1>

<p align="center">
  A small, inspectable, and extensible runtime for coding agents.
</p>

---

[English](README.md) | [简体中文](README.zh-CN.md)

CodeKeel connects language models to repository tools and bounded workspaces. It adds the runtime services that turn a model/tool loop into a dependable coding agent: repository context, typed tools, budgets, approval controls, context management, durable traces, resumable runs, and verification-aware completion.

## Quick Start

### Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Docker, only required when using Docker workspaces

### Install from Source

```bash
git clone https://github.com/Sakikoo0/codekeel codekeel
cd codekeel
uv sync
uv run codekeel --version
```

If CodeKeel has not yet been installed as a system command, prefix the commands below with `uv run`, for example `uv run codekeel run ...`.

### Configure the Model

CodeKeel accepts LiteLLM-style model identifiers and uses the standard environment configuration for the corresponding model provider. For example:

```bash
export DEEPSEEK_API_KEY="..."
# export OPENAI_API_KEY="..."
# export ANTHROPIC_API_KEY="..."
```

Then use the corresponding provider/model identifier, such as `deepseek/deepseek-v4-flash`.

### Run a Task

Before starting a run, create the code repository directory and the trusted state directory:

```bash
mkdir -p ./demo-repo ./.codekeel-state

codekeel run \
  --repo ./demo-repo \
  --root ./.codekeel-state \
  --model deepseek/deepseek-v4-flash \
  --task "Create README.md with a title, a brief project introduction, and usage instructions."
```

`--repo` is the code repository that the Agent can inspect and modify. `--root` is the trusted host directory where CodeKeel stores checkpoints and run traces. Keeping them separate prevents repository tools from treating runtime state as ordinary project files.

`run` outputs a JSON summary containing the run ID, status, resource usage, duration, verification result, trace path, and pending approval action ID.

### Use a Docker Workspace

```bash
codekeel run \
  --repo ./demo-repo \
  --root ./.codekeel-state \
  --workspace docker \
  --image python:3.12-slim \
  --model deepseek/deepseek-v4-flash \
  --task "Fix the failing tests" \
  --verify "pytest"
```

Docker workspaces are ephemeral and have networking disabled by default. The CLI can currently only resume recorded local workspaces.

## CLI Overview

| Command                             | Purpose                                       |
| ----------------------------------- | --------------------------------------------- |
| `codekeel run`                      | Start a new non-interactive coding agent task |
| `codekeel inspect RUN_ID`           | Output the run event stream as JSON Lines     |
| `codekeel resume RUN_ID`            | Continue a resumable local run                |
| `codekeel approve RUN_ID ACTION_ID` | Approve the exact currently pending action    |
| `codekeel reject RUN_ID ACTION_ID`  | Reject the exact currently pending action     |

Use `codekeel COMMAND --help` to view all available options.

### Inspect and Resume a Run

```bash
codekeel inspect RUN_ID --root ./.codekeel-state
codekeel resume RUN_ID --root ./.codekeel-state
```

Subsequent inspect, approve, reject, and resume commands must use the same `--root` as the original `run`.

### Approve or Reject Actions

The default `--approval risky` mode pauses high-risk and unknown actions. Use `--approval always` to review every action that would otherwise be allowed, or `--approval never` for non-interactive execution. Hard-deny rules and workspace boundaries remain enforced in all modes.

When a run returns `waiting_for_approval`, inspect the events first, record an exact decision, and then resume:

```bash
codekeel inspect RUN_ID --root ./.codekeel-state
codekeel approve RUN_ID ACTION_ID --root ./.codekeel-state
# Or: codekeel reject RUN_ID ACTION_ID --root ./.codekeel-state
codekeel resume RUN_ID --root ./.codekeel-state
```

The approve and reject commands only record the decision; `resume` is what continues execution.

## Documentation

For detailed design documents and explanations, see [docs](docs).

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```

Changes involving Docker require the Docker daemon to be running when executing:

```bash
uv run pytest -m docker
```

## Evaluate a Single Task

[SWE-bench](https://github.com/SWE-bench/SWE-bench) is a benchmark for evaluating coding agents on real-world software engineering tasks derived from GitHub issues and repositories. It measures whether a model can generate code changes that resolve the target issue and pass the repository's tests.

<details>
<summary>Show details</summary>

### Prerequisites

Recommended directory structure:

```text
workspace/
├── codekeel-codex/
└── SWE-bench/
```

The commands below assume that the two repositories are located under the same parent directory.

### Install the Official SWE-bench Evaluator

Run the following from the shared parent directory:

```bash
git clone https://github.com/swe-bench/SWE-bench
cd SWE-bench
uv venv
uv pip install -e .
git clone --depth 1 \
  https://github.com/SWE-bench/swe-bench-tasks.git \
  ./swe-bench-tasks
```

Check the task repository:

```bash
uv run swebench dataset check ./swe-bench-tasks
```

Expected output:

```bash
./swe-bench-tasks looks well formed
```

### Install CodeKeel Dependencies

```bash
cd ../codekeel-codex
uv sync --group swebench
```

### Configure the Model API Key

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
```

### Pin the SWE-bench Dataset Revision

Query the current revision:

```bash
uv run python -c \
  "from huggingface_hub import HfApi; print(HfApi().dataset_info('SWE-bench/SWE-bench', revision='main').sha)"
```

To make the evaluation reproducible, use a fixed revision for subsequent commands:

```bash
export SWEBENCH_REVISION="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"
```

### View Available Tasks

```bash
uv run python -c \
 "from datasets import load_dataset; d=load_dataset('SWE-bench/SWE-bench', revision='$SWEBENCH_REVISION', split='test');
print('\n'.join(d['instance_id'][:20]))"
```

Select a task:

```bash
export TASK_ID="astropy__astropy-12057"
```

### Convert a SWE-bench Task

```bash
export DATASET_DIR="swebench/swebench-smoke-1"
uv run codekeel eval-convert-swebench \
  --instance-id "$TASK_ID" \
  --revision "$SWEBENCH_REVISION" \
  --output "$DATASET_DIR"
```

After conversion, the directory should look roughly like this:

```text
swebench/swebench-smoke-1/
├── metadata.json
├── astropy__astropy-12057.yaml
└── repos/
      └── astropy__astropy-12057/
```

### Run the CodeKeel Evaluation

Set the model:

```bash
export CODEKEEL_MODEL="openrouter/x-ai/grok-4.6"
```

Run with the full configuration:

```bash
uv run codekeel eval \
  --dataset "$DATASET_DIR" \
  --model "$CODEKEEL_MODEL" \
  --config configs/full.yaml \
  --root swebench/results
```

After completion, CodeKeel will generate a run directory similar to the following:

```text
swebench/results/<evaluation-id>/<run-id>/
├── results.jsonl
├── patches/
│ └── astropy__astropy-12057.diff
└── .agent/
```

Record the directory containing `results.jsonl` and `patches/`:

```bash
export CODEKEEL_RUN_DIR="swebench/results/<evaluation-id>/<run-id>"
```

Replace the placeholders with the actual directory generated by this run.

You can verify that the patch exists:

```bash
ls "$CODEKEEL_RUN_DIR/patches/$TASK_ID.diff"
```

Note: CodeKeel's `success` or `verification_result` only reflects the local evaluation result and is not equivalent to the official SWE-bench `resolved` result.

### Generate the Official predictions.jsonl

Set the model identifier to submit to the evaluator. This field is only a public label and must not contain API keys, endpoints, or account information.

```bash
export SUBMISSION_MODEL_NAME="codekeel/grok-4.6"
```

Generate the official input:

```bash
uv run codekeel eval-export-swebench \
  --dataset "$DATASET_DIR" \
  --run "$CODEKEEL_RUN_DIR" \
  --model-name "$SUBMISSION_MODEL_NAME" \
  --output "$CODEKEEL_RUN_DIR/predictions.jsonl"
```

Check the file:

```bash
wc -l "$CODEKEEL_RUN_DIR/predictions.jsonl"
head -n 1 "$CODEKEEL_RUN_DIR/predictions.jsonl"
```

A single-task evaluation should generate one JSONL line containing:

```json
{
  "instance_id": "astropy\_\_astropy-12057",
  "model_name_or_path": "codekeel/deepseek-v4-flash",
  "model_patch": "<full patch text>"
}
```

### Run the Official SWE-bench Evaluation

Since the current directory is `SWE-bench/`, the prediction file generated by CodeKeel is located in the neighboring repository:

```bash
cd ../SWE-bench
export PREDICTIONS_FILE="../codekeel-codex/$CODEKEEL_RUN_DIR/predictions.jsonl"
```

Run the single-task evaluation:

```bash
uv run swebench eval full \
  --predictions "$PREDICTIONS_FILE" \
  --instance "$TASK_ID" \
  --run-id "codekeel-${TASK_ID}-001" \
  --workers 1 \
  --task-repo ./swe-bench-tasks
```

This may take some time. Please allow five to ten minutes for it to complete.

### View the Official Results

View the evaluation summary:

```bash
cat "logs/evaluation/codekeel-${TASK_ID}-001/results.json"
```

You can also regenerate the report without rerunning the container:

```bash
uv run swebench report "codekeel-${TASK_ID}-001" -d full
```

The final result should clearly record one of the following statuses:

- resolved: The patch passes the tests required by the official evaluator.
- unresolved: The patch was evaluated successfully but did not satisfy the test requirements.
- evaluator error: An error occurred with the image, patch application, test environment, or evaluator.

</details>

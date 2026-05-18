<div align="center" id="verl-mint">
  <h1>verl-mint</h1>

  <p><em>Open MinT training runtime on veRL.</em></p>

  <p>
    <a href="https://github.com/MindLab-Research/mindlab-toolkit">MinT SDK</a> .
    <a href="https://github.com/MindLab-Research/mint-doc">Documentation</a> .
    <a href="https://github.com/MindLab-Research/mint-quickstart">Quickstart</a> .
    <a href="https://mint-console.macaron.xin/">MinT Console</a> .
    <a href="https://github.com/MindLab-Research/mint-cookbook">Cookbook</a> .
    <a href="https://github.com/verl-project/verl">veRL</a>
  </p>
</div>

## What verl-mint is

`verl-mint` is the commercial open-source edition of the [MinT](https://mint-console.macaron.xin/) training runtime. The commercial MinT SaaS gives users hosted training infrastructure through the MinT SDK. `verl-mint` exposes the same training mental model for teams that want to run MinT-style workloads on their own GPUs, their own Ray or veRL cluster, and their own artifact storage.

The server exposes the MinT training workflow at the API boundary, so clients can keep familiar session, checkpoint, sampler, and future semantics while moving execution onto the MinT and veRL stack.

The project is built in collaboration with veRL as the training execution foundation. MinT exposes a client-facing workflow for creating a training session, running SFT or RL updates, saving checkpoints, handing weights to samplers, collecting rollouts, and continuing from saved state.

## Support matrix

### Algorithms

| Algorithm | Status | Notes |
| --- | --- | --- |
| SFT | Available | LoRA SFT, checkpoint save/load, resume, sampler handoff |
| GRPO | Available | Group Relative Policy Optimization; group-relative advantages without a value critic |
| DPO | Planned | Preference optimization track |

### Models

| Model family | Size | Status | Notes |
| --- | --- | --- | --- |
| Qwen | 0.6B | Available | veRL trainer smoke path |
| Qwen | 4B | Available | veRL model-backed run |
| Qwen | 8B | Planned | Not yet validated |
| Qwen | 30B | Available | veRL/Ray run |

## Install

```bash
git clone https://github.com/verl-project/verl-mint.git
cd verl-mint
uv sync --extra smoke
```

The `smoke` extra installs the official MinT SDK from the MindLab repository:
`mindlab-toolkit @ git+https://github.com/MindLab-Research/mindlab-toolkit.git`.

Local SDK checkouts are useful for reading source, but smoke runs should use the installed `mindlab-toolkit` dependency rather than `PYTHONPATH`.

For cluster runs, `verl-mint` assumes the execution environment already exists; it does not provision Ray, GPUs, or worker images.

| Prerequisite | Requirement |
| --- | --- |
| Ray | `ray[default]>=2.46.0,<3` on the API node and every worker |
| veRL runtime | [verl-project/verl](https://github.com/verl-project/verl) importable in the Python environment used by Ray actors |
| Storage | a shared artifact root visible to the API node and workers; copy `.env.example` to `.env` and set `VERL_MINT_STORAGE_ROOT` plus `VERL_MINT_SHARED_STORAGE_ROOTS` |

Install the Ray client/runtime dependency on the API node with:

```bash
uv sync --extra ray
```

## Quick start

| Path | Entry point |
| --- | --- |
| Smoke server: contract | `scripts/smoke_fake_backend.py` |
| Smoke server: Qwen SFT diagnostic | `scripts/smoke_qwen_sft.py` |
| Smoke server: veRL trainer job | `scripts/smoke_verl_ppo_job.py` |
| Remote SDK client smoke | `scripts/client_verl_ppo_job_smoke.py --base-url http://<host>:8000` |

Run the contract smoke with a lightweight backend:

```bash
uv run --env-file /dev/null --extra smoke \
  python scripts/smoke_fake_backend.py
```

Run the local Qwen SFT diagnostic path. This checks API compatibility with a small local PyTorch backend; it is not the recommended model-backed training path:

```bash
uv run --env-file /dev/null --extra smoke \
  python scripts/smoke_qwen_sft.py
```

Run an SDK-driven veRL trainer job smoke:

```bash
export VERL_CONFIG_DIR=/workspace/verl/verl/trainer/config

uv run --env-file /dev/null --extra smoke --extra ray \
  python scripts/smoke_verl_ppo_job.py \
  --config-dir "$VERL_CONFIG_DIR" \
  --model-path Qwen/Qwen3-0.6B \
  --train-files /path/to/train.parquet \
  --val-files /path/to/val.parquet \
  --total-steps 1
```

All `client_*` scripts exercise an already-running service through the installed MinT SDK and require `--base-url`. Scripts that start a local smoke server and immediately run the matching SDK scenario live under `scripts/smoke_*.py`. Scripts that post directly to server routes are server/API diagnostics, not client contract tests, and should not be named or treated as client smoke tests.

## Simple cluster service

For the open-source runtime, keep deployment simple: start one HTTP service on a network-reachable cluster node, point the official SDK at it, run a smoke or example, and stop the service with `Ctrl-C` or SIGTERM. This is not a production deployment recipe.

Start the service with the veRL backend:

```bash
export VERL_CONFIG_DIR=/workspace/verl/verl/trainer/config
export VERL_MINT_STORAGE_ROOT=/shared/path/verl-mint

uv run --extra smoke --extra ray \
  verl-mint serve \
  --backend verl \
  --host 0.0.0.0 \
  --port 8000 \
  --storage-root "$VERL_MINT_STORAGE_ROOT" \
  --model-path Qwen/Qwen3-0.6B \
  --verl-config-dir "$VERL_CONFIG_DIR" \
  --override data.train_files=/shared/path/data/train.parquet \
  --override data.val_files=/shared/path/data/val.parquet \
  --override trainer.total_training_steps=1
```

From any non-Ray host that can reach the service:

```python
import mint

client = mint.ServiceClient(
    base_url="http://<cluster-node-ip>:8000",
    api_key="sk-local-smoke",
)
training = client.create_lora_training_client("Qwen/Qwen3-0.6B", rank=16)
print(training.get_info())
```

Or run a packaged SDK client smoke against that service:

```bash
uv run --env-file /dev/null --extra client \
  python scripts/client_verl_ppo_job_smoke.py \
  --base-url http://<cluster-node-ip>:8000 \
  --base-model Qwen/Qwen3-0.6B
```

## Ray cluster run

`verl-mint` can move model execution into Ray or veRL workers while keeping the FastAPI process as the control plane. Start from an existing Ray 2.x cluster using `ray[default]>=2.46.0,<3` where the API node and workers share the same Python dependencies and artifact root. For open-source use, the expected deployment shape is one service process on the Ray or veRL cluster plus clients on any network-reachable non-Ray host.

Use the service command above on the Ray or veRL cluster node, then run a `client_*` smoke from a network-reachable non-Ray host with `--base-url`. The `scripts/smoke_*.py` entries are local one-shot smoke servers; they are not the remote-client path.

## Architecture

`verl-mint` runs a FastAPI control plane that accepts MinT-style training requests, routes them through backend abstractions, and stores checkpoint artifacts through a configurable storage repository. Local diagnostics can use the lightweight fake and Qwen backends, but the recommended model-backed path uses veRL trainer execution while keeping the API process responsible for sessions, futures, checkpoint metadata, and sampler handoff.

## API surface

`verl-mint` exposes MinT-style routes for:

- model lifecycle: create, load from state, unload, inspect
- training: forward, forward-backward, optimizer step, train step
- RL: GRPO/reverse-KL, rollout collection, train-on-experience
- inference: sample and compute logprobs
- checkpoints: save state, load state, export weights for sampler
- futures: retrieve asynchronous training results

Checkpoint endpoints keep `save_state`/`load_state` aliases for clients that use state-oriented wording; the runtime stores and returns the same underlying model weights artifacts.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands, pull request expectations, and release versioning. Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).

## MinT ecosystem

| Repository | Role |
| --- | --- |
| [mindlab-toolkit](https://github.com/MindLab-Research/mindlab-toolkit) | Python SDK used by MinT users and examples |
| [MinT Console](https://mint-console.macaron.xin/) | SaaS product entry point |
| [mint-doc](https://github.com/MindLab-Research/mint-doc) | Product and API documentation |
| [mint-quickstart](https://github.com/MindLab-Research/mint-quickstart) | First-run tutorials and migration examples |
| [mint-cookbook](https://github.com/MindLab-Research/mint-cookbook) | Reproducible MinT training recipes |
| [verl](https://github.com/verl-project/verl) | veRL training execution foundation used by the open runtime |

## Research notes

- [MinT: Managed Infrastructure for Training and Serving Millions of LLMs](https://arxiv.org/abs/2605.13779)
- [How We Build Trillion Parameter Reasoning RL with 10% GPUs](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)

## Requirements

- Python `>=3.11`
- CUDA GPU for practical model training
- `torch`, `transformers`, `peft`, and `accelerate` for local diagnostics
- `ray[default]>=2.46.0,<3` for distributed execution
- shared storage for multi-node checkpoint save and load

## Citation

If you use `verl-mint`, please cite the MinT technical report:

```bibtex
@misc{mindlab2026mint,
  title = {MinT: Managed Infrastructure for Training and Serving Millions of LLMs},
  author = {{Mind Lab}},
  year = {2026},
  eprint = {2605.13779},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2605.13779},
  url = {https://arxiv.org/abs/2605.13779},
}
```

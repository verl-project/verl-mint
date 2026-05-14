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

`verl-mint` is the commercial open-source edition of the [MinT](https://mint-console.macaron.xin/) training runtime. The commercial MinT SaaS gives users hosted training infrastructure through the MinT SDK. `verl-mint` exposes the same training mental model for teams that want to run MinT-style SFT and RL workloads on their own GPUs, their own Ray cluster, and their own artifact storage.

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
| Qwen | 0.6B | Available | Local Qwen SFT/RL smoke path |
| Qwen | 4B | Available | Validated model-backed run |
| Qwen | 8B | Planned | Not yet validated |
| Qwen | 30B | Available | Validated distributed veRL/Ray run |

## Install

```bash
git clone https://github.com/verl-project/verl-mint.git
cd verl-mint
uv sync --extra qwen-sft
```

Install the MinT SDK from the MindLab repository:

```bash
uv pip install "git+https://github.com/MindLab-Research/mindlab-toolkit.git"
```

For local development with an SDK checkout:

```bash
export MINT_SDK_PATH=/path/to/mindlab-toolkit/src
export PYTHONPATH=$MINT_SDK_PATH:src
```

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
| Contract smoke | `scripts/official_client_fake_backend_smoke.py` |
| Qwen SFT | `scripts/official_client_qwen_sft_smoke.py` |
| GRPO | `scripts/official_client_qwen_grpo_smoke.py` |
| veRL cluster | `scripts/official_client_verl_cluster_smoke.py` |
| Distributed GRPO | `scripts/official_client_verl_grpo_cluster_smoke.py` |

Run the contract smoke with a lightweight backend:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_fake_backend_smoke.py
```

Run the local Qwen SFT path:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_qwen_sft_smoke.py
```

Run the GRPO path:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_qwen_grpo_smoke.py
```

## Ray cluster run

`verl-mint` can move model execution into Ray actors while keeping the FastAPI process as the control plane. Start from an existing Ray 2.x cluster using `ray[default]>=2.46.0,<3` where the API node and workers share the same Python dependencies and artifact root.

```bash
export VERL_MINT_STORAGE_ROOT=/shared/path/verl-mint
export VERL_MINT_SHARED_STORAGE_ROOTS=/shared/path

uv run --env-file .env env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_verl_cluster_smoke.py \
  --storage-root "$VERL_MINT_STORAGE_ROOT"
```

## Architecture

`verl-mint` runs a FastAPI control plane that accepts MinT-style training requests, routes them through backend abstractions, and stores checkpoint artifacts through a configurable storage repository. Local development can use the lightweight fake and Qwen SFT backends; distributed runs use Ray/veRL workers while keeping the API process responsible for sessions, futures, checkpoint metadata, and sampler handoff.

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
- CUDA GPU for practical local model training
- `torch`, `transformers`, `peft`, and `accelerate` for Qwen local training
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

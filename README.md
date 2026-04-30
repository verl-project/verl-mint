<div align="center" id="verl-mint">
  <h1>verl-mint</h1>

  <p><em>Tinker-compatible open MinT training runtime on veRL.</em></p>

  <p>
    <a href="https://github.com/MindLab-Research/mindlab-toolkit">MinT SDK</a> .
    <a href="https://github.com/MindLab-Research/mint-doc">Documentation</a> .
    <a href="https://github.com/MindLab-Research/mint-quickstart">Quickstart</a> .
    <a href="https://mint-console.macaron.xin/">MinT Console</a> .
    <a href="https://github.com/MindLab-Research/mint-cookbook">Cookbook</a> .
    <a href="https://github.com/MindLab-Research/verl">veRL</a> .
    <span>Paper: TBD</span>
  </p>
</div>

## What verl-mint is

`verl-mint` is the commercial open-source edition of the [MinT](https://mint-console.macaron.xin/) training runtime. The commercial MinT SaaS gives users hosted training infrastructure through the MinT SDK. `verl-mint` exposes the same training mental model for teams that want to run MinT-style SFT and RL workloads on their own GPUs, their own Ray cluster, and their own artifact storage.

The server is Tinker-compatible at the API boundary, so existing clients can keep the familiar session, checkpoint, sampler, and future semantics while moving execution onto the MinT and veRL stack.

The project is built in collaboration with veRL as the training execution foundation. MinT exposes a client-facing workflow for creating a training session, running SFT or RL updates, saving checkpoints, handing weights to samplers, collecting rollouts, and continuing from saved state.

## Support matrix

### Algorithms

| Algorithm | Status | Notes |
| --- | --- | --- |
| SFT | Available | LoRA SFT, checkpoint save/load, resume, sampler handoff |
| GRPO | Available | Group Relative Policy Optimization; group-relative advantages without a value critic |
| DPO | Planned | Preference optimization track |

### Models

| Model family | Size | Status |
| --- | --- | --- |
| Qwen | 0.6B | Available |
| Qwen | 4B | Planned |
| Qwen | 8B | Planned |
| Qwen | 30B | Planned |

## Install

```bash
git clone https://github.com/MindLab-Research/verl-mint-alpha.git
cd verl-mint-alpha
uv sync --extra qwen-sft --extra verl-ray
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

## Quick start

| Path | Entry point |
| --- | --- |
| Contract smoke | `scripts/official_client_fake_backend_smoke.py` |
| Qwen SFT | `scripts/official_client_qwen_sft_smoke.py` |
| GRPO | `scripts/official_client_qwen_grpo_smoke.py` |
| Ray cluster | `scripts/official_client_verl_ray_cluster_smoke.py` |
| Distributed GRPO | `scripts/official_client_verl_ray_grpo_cluster_smoke.py` |

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

`verl-mint` can move model execution into Ray actors while keeping the FastAPI process as the control plane. The cluster path expects a shared artifact root visible to the API node and workers.

```bash
export VERL_MINT_RAY_ADDRESS=ray://<ray-head-host>:10001
export VERL_MINT_STORAGE_ROOT=/shared/path/verl-mint

uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_verl_ray_cluster_smoke.py \
  --storage-root "$VERL_MINT_STORAGE_ROOT" \
  --ray-address "$VERL_MINT_RAY_ADDRESS"
```

## Architecture

Architecture diagram placeholder. Replace this block with the official architecture diagram before publication.

## API surface

`verl-mint` exposes MinT-style routes for:

- model lifecycle: create, load from state, unload, inspect
- training: forward, forward-backward, optimizer step, train step
- RL: GRPO/reverse-KL, rollout collection, train-on-experience
- inference: sample and compute logprobs
- checkpoints: save state, load state, export weights for sampler
- futures: retrieve asynchronous training results

## MinT ecosystem

| Repository | Role |
| --- | --- |
| [mindlab-toolkit](https://github.com/MindLab-Research/mindlab-toolkit) | Python SDK used by MinT users and examples |
| [MinT Console](https://mint-console.macaron.xin/) | SaaS product entry point |
| [mint-doc](https://github.com/MindLab-Research/mint-doc) | Product and API documentation |
| [mint-quickstart](https://github.com/MindLab-Research/mint-quickstart) | First-run tutorials and migration examples |
| [mint-cookbook](https://github.com/MindLab-Research/mint-cookbook) | Reproducible MinT training recipes |
| [verl](https://github.com/MindLab-Research/verl) | veRL training execution foundation used by the open runtime |

## Research notes

- [How We Build Trillion Parameter Reasoning RL with 10% GPUs](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)

## Requirements

- Python `>=3.11`
- CUDA GPU for practical local model training
- `torch`, `transformers`, `peft`, and `accelerate` for Qwen local training
- Ray for distributed execution
- shared storage for multi-node checkpoint save and load

## Citation

TBD. Public paper, project page, and BibTeX entries will be added here.

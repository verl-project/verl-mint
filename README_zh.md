<div align="center" id="verl-mint">
  <h1>verl-mint</h1>

  <p><em>基于 veRL 的开源 MinT 训练运行时.</em></p>

  <p>
    <a href="https://github.com/MindLab-Research/mindlab-toolkit">MinT SDK</a> .
    <a href="https://github.com/MindLab-Research/mint-doc">文档</a> .
    <a href="https://github.com/MindLab-Research/mint-quickstart">快速开始</a> .
    <a href="https://mint-console.macaron.xin/">MinT Console</a> .
    <a href="https://github.com/MindLab-Research/mint-cookbook">Cookbook</a> .
    <a href="https://github.com/verl-project/verl">veRL</a>
  </p>
</div>

## verl-mint 是什么

`verl-mint` 是 [MinT](https://mint-console.macaron.xin/) 训练运行时的商业化开源版本. 商业 MinT SaaS 通过 MinT SDK 向用户提供托管训练基础设施. `verl-mint` 面向希望在自有 GPU, 自有 Ray 集群和自有 artifact 存储上运行 MinT 风格 SFT 与 RL 任务的团队.

服务端在 API 边界暴露 MinT 训练工作流, 因此客户端可以保留熟悉的 session, checkpoint, sampler 和 future 语义, 同时把执行迁移到 MinT 和 veRL stack.

项目和 veRL 合作构建训练执行底座. MinT 提供面向客户端的训练工作流: 创建训练 session, 执行 SFT 或 RL 更新, 保存 checkpoint, 将权重交给 sampler, 收集 rollout, 并从已保存状态继续训练.

## 支持矩阵

### 算法

| 算法 | 状态 | 说明 |
| --- | --- | --- |
| SFT | Available | LoRA SFT, checkpoint 保存/加载, resume, sampler handoff |
| GRPO | Available | Group Relative Policy Optimization; 无 value critic 的 group-relative advantages |
| DPO | Planned | Preference optimization 路径 |

### 模型

| 模型族 | 尺寸 | 状态 | 说明 |
| --- | --- | --- | --- |
| Qwen | 0.6B | Available | 本地 Qwen SFT/RL smoke 路径 |
| Qwen | 4B | Available | 已通过 model-backed run 验证 |
| Qwen | 8B | Planned | 尚未验证 |
| Qwen | 30B | Available | 已通过分布式 veRL/Ray run 验证 |

## 安装

```bash
git clone https://github.com/verl-project/verl-mint.git
cd verl-mint
uv sync --extra qwen-sft
```

从 MindLab 仓库安装 MinT SDK:

```bash
uv pip install "git+https://github.com/MindLab-Research/mindlab-toolkit.git"
```

如果使用本地 SDK checkout 开发:

```bash
export MINT_SDK_PATH=/path/to/mindlab-toolkit/src
export PYTHONPATH=$MINT_SDK_PATH:src
```

集群运行时, `verl-mint` 假设执行环境已经存在; 它不负责启动或部署 Ray, GPU 节点或 worker 镜像.

| 前置条件 | 要求 |
| --- | --- |
| Ray | API 节点和所有 worker 使用 `ray[default]>=2.46.0,<3` |
| veRL runtime | Ray actor 使用的 Python 环境中可以 import [verl-project/verl](https://github.com/verl-project/verl) |
| 存储 | API 节点和 worker 都能访问同一个 artifact 根目录; 复制 `.env.example` 为 `.env`, 并设置 `VERL_MINT_STORAGE_ROOT` 和 `VERL_MINT_SHARED_STORAGE_ROOTS` |

API 节点可以这样安装 Ray client/runtime 依赖:

```bash
uv sync --extra ray
```

## 快速开始

| 路径 | 入口 |
| --- | --- |
| Contract smoke | `scripts/official_client_fake_backend_smoke.py` |
| Qwen SFT | `scripts/official_client_qwen_sft_smoke.py` |
| GRPO | `scripts/official_client_qwen_grpo_smoke.py` |
| veRL 集群 | `scripts/official_client_verl_cluster_smoke.py` |
| 分布式 GRPO | `scripts/official_client_verl_grpo_cluster_smoke.py` |

先运行轻量 backend 的 contract smoke:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_fake_backend_smoke.py
```

运行本地 Qwen SFT 路径:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_qwen_sft_smoke.py
```

运行 GRPO 路径:

```bash
uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_qwen_grpo_smoke.py
```

## Ray 集群运行

`verl-mint` 可以把模型执行移动到 Ray actor, 同时让 FastAPI 进程保留 control plane 职责. 请从已有的 Ray 2.x 集群开始, Ray 版本需 `ray[default]>=2.46.0,<3`, 并确保 API 节点和 worker 使用一致的 Python 依赖和 artifact 根目录.

```bash
export VERL_MINT_STORAGE_ROOT=/shared/path/verl-mint
export VERL_MINT_SHARED_STORAGE_ROOTS=/shared/path

uv run --env-file .env env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_verl_cluster_smoke.py \
  --storage-root "$VERL_MINT_STORAGE_ROOT"
```

## 架构

`verl-mint` 运行一个 FastAPI control plane, 接收 MinT 风格训练请求, 通过 backend abstraction 路由执行, 并通过可配置的 storage repository 管理 checkpoint artifact. 本地开发可以使用轻量 fake backend 和 Qwen SFT backend; 分布式运行时使用 Ray/veRL worker, API 进程负责 session, future, checkpoint metadata 和 sampler handoff.

## API 接口范围

`verl-mint` 暴露 MinT 风格接口:

- model lifecycle: create, load from state, unload, inspect
- training: forward, forward-backward, optimizer step, train step
- RL: GRPO/reverse-KL, rollout collection, train-on-experience
- inference: sample 和 compute logprobs
- checkpoints: save state, load state, export weights for sampler
- futures: retrieve asynchronous training results

Checkpoint 接口保留 `save_state`/`load_state` 作为兼容别名, 面向 state 命名的客户端仍可使用; runtime 底层保存和返回的是同一类模型权重 artifact.

## 贡献

开发环境, 测试命令, PR 要求和版本发布规则见 [CONTRIBUTING.md](CONTRIBUTING.md). 版本变更记录见 [CHANGELOG.md](CHANGELOG.md).

## MinT 生态

| 仓库 | 角色 |
| --- | --- |
| [mindlab-toolkit](https://github.com/MindLab-Research/mindlab-toolkit) | MinT 用户和 examples 使用的 Python SDK |
| [MinT Console](https://mint-console.macaron.xin/) | SaaS 产品入口 |
| [mint-doc](https://github.com/MindLab-Research/mint-doc) | 产品与 API 文档 |
| [mint-quickstart](https://github.com/MindLab-Research/mint-quickstart) | 第一次运行教程和迁移 examples |
| [mint-cookbook](https://github.com/MindLab-Research/mint-cookbook) | 可复现的 MinT 训练 recipes |
| [verl](https://github.com/verl-project/verl) | 开源 runtime 使用的 veRL 训练执行底座 |

## 研究文章

- [MinT: Managed Infrastructure for Training and Serving Millions of LLMs](https://arxiv.org/abs/2605.13779)
- [How We Build Trillion Parameter Reasoning RL with 10% GPUs](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)

## 要求

- Python `>=3.11`
- 本地模型训练需要 CUDA GPU 才有实用速度
- Qwen 本地训练需要 `torch`, `transformers`, `peft`, `accelerate`
- 分布式执行需要 `ray[default]>=2.46.0,<3`
- 多节点 checkpoint 保存和加载需要共享存储

## 引用

如果使用 `verl-mint`, 请引用 MinT Tech Report:

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

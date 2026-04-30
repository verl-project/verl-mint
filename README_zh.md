<div align="center" id="verl-mint">
  <h1>verl-mint</h1>

  <p><em>兼容 Tinker API 的开源 MinT 训练运行时, 基于 veRL.</em></p>

  <p>
    <a href="https://github.com/MindLab-Research/mindlab-toolkit">MinT SDK</a> .
    <a href="https://github.com/MindLab-Research/mint-doc">文档</a> .
    <a href="https://github.com/MindLab-Research/mint-quickstart">快速开始</a> .
    <a href="https://mint-console.macaron.xin/">MinT Console</a> .
    <a href="https://github.com/MindLab-Research/mint-cookbook">Cookbook</a> .
    <a href="https://github.com/MindLab-Research/verl">veRL</a> .
    <span>论文: TBD</span>
  </p>
</div>

## verl-mint 是什么

`verl-mint` 是 [MinT](https://mint-console.macaron.xin/) 训练运行时的商业化开源版本. 商业 MinT SaaS 通过 MinT SDK 向用户提供托管训练基础设施. `verl-mint` 面向希望在自有 GPU, 自有 Ray 集群和自有 artifact 存储上运行 MinT 风格 SFT 与 RL 任务的团队.

服务端在 API 边界兼容 Tinker, 因此已有客户端可以保留熟悉的 session, checkpoint, sampler 和 future 语义, 同时把执行迁移到 MinT 和 veRL stack.

项目和 veRL 合作构建训练执行底座. MinT 提供面向客户端的训练工作流: 创建训练 session, 执行 SFT 或 RL 更新, 保存 checkpoint, 将权重交给 sampler, 收集 rollout, 并从已保存状态继续训练.

## 支持矩阵

### 算法

| 算法 | 状态 | 说明 |
| --- | --- | --- |
| SFT | Available | LoRA SFT, checkpoint 保存/加载, resume, sampler handoff |
| GRPO | Available | Group Relative Policy Optimization; 无 value critic 的 group-relative advantages |
| DPO | Planned | Preference optimization 路径 |

### 模型

| 模型族 | 尺寸 | 状态 |
| --- | --- | --- |
| Qwen | 0.6B | Available |
| Qwen | 4B | Planned |
| Qwen | 8B | Planned |
| Qwen | 30B | Planned |

## 安装

```bash
git clone https://github.com/MindLab-Research/verl-mint-alpha.git
cd verl-mint-alpha
uv sync --extra qwen-sft --extra verl-ray
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

## 快速开始

| 路径 | 入口 |
| --- | --- |
| Contract smoke | `scripts/official_client_fake_backend_smoke.py` |
| Qwen SFT | `scripts/official_client_qwen_sft_smoke.py` |
| GRPO | `scripts/official_client_qwen_grpo_smoke.py` |
| Ray 集群 | `scripts/official_client_verl_ray_cluster_smoke.py` |
| 分布式 GRPO | `scripts/official_client_verl_ray_grpo_cluster_smoke.py` |

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

`verl-mint` 可以把模型执行移动到 Ray actor, 同时让 FastAPI 进程保留 control plane 职责. 集群路径要求 API 节点和 worker 都能访问同一个 artifact 根目录.

```bash
export VERL_MINT_RAY_ADDRESS=ray://<ray-head-host>:10001
export VERL_MINT_STORAGE_ROOT=/shared/path/verl-mint

uv run --env-file /dev/null env PYTHONPATH=$PYTHONPATH \
  python scripts/official_client_verl_ray_cluster_smoke.py \
  --storage-root "$VERL_MINT_STORAGE_ROOT" \
  --ray-address "$VERL_MINT_RAY_ADDRESS"
```

## 架构

架构图占位. 发布前将此处替换为正式架构图.

## API 接口范围

`verl-mint` 暴露 MinT 风格接口:

- model lifecycle: create, load from state, unload, inspect
- training: forward, forward-backward, optimizer step, train step
- RL: GRPO/reverse-KL, rollout collection, train-on-experience
- inference: sample 和 compute logprobs
- checkpoints: save state, load state, export weights for sampler
- futures: retrieve asynchronous training results

## MinT 生态

| 仓库 | 角色 |
| --- | --- |
| [mindlab-toolkit](https://github.com/MindLab-Research/mindlab-toolkit) | MinT 用户和 examples 使用的 Python SDK |
| [MinT Console](https://mint-console.macaron.xin/) | SaaS 产品入口 |
| [mint-doc](https://github.com/MindLab-Research/mint-doc) | 产品与 API 文档 |
| [mint-quickstart](https://github.com/MindLab-Research/mint-quickstart) | 第一次运行教程和迁移 examples |
| [mint-cookbook](https://github.com/MindLab-Research/mint-cookbook) | 可复现的 MinT 训练 recipes |
| [verl](https://github.com/MindLab-Research/verl) | 开源 runtime 使用的 veRL 训练执行底座 |

## 研究文章

- [How We Build Trillion Parameter Reasoning RL with 10% GPUs](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)

## 要求

- Python `>=3.11`
- 本地模型训练需要 CUDA GPU 才有实用速度
- Qwen 本地训练需要 `torch`, `transformers`, `peft`, `accelerate`
- 分布式执行需要 Ray
- 多节点 checkpoint 保存和加载需要共享存储

## 引用

TBD. 后续会在这里补充公开论文, 项目页和 BibTeX.

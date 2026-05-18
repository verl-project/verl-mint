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

`verl-mint` 是 [MinT](https://mint-console.macaron.xin/) 训练运行时的商业化开源版本. 商业 MinT SaaS 通过 MinT SDK 向用户提供托管训练基础设施. `verl-mint` 面向希望在自有 GPU, 自有 Ray 或 veRL 集群和自有 artifact 存储上运行 MinT 风格训练任务的团队.

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
| Qwen | 0.6B | Available | veRL trainer smoke 路径 |
| Qwen | 4B | Available | veRL model-backed run |
| Qwen | 8B | Planned | 尚未验证 |
| Qwen | 30B | Available | veRL/Ray run |

## 安装

```bash
git clone https://github.com/verl-project/verl-mint.git
cd verl-mint
uv sync --extra smoke
```

`smoke` extra 会从 MindLab 正式仓库安装 MinT SDK:
`mindlab-toolkit @ git+https://github.com/MindLab-Research/mindlab-toolkit.git`.

本地 SDK checkout 只用于阅读源码; smoke 运行应使用已安装的 `mindlab-toolkit` 依赖, 不通过 `PYTHONPATH` 注入本地 SDK.

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
| Smoke server: contract | `scripts/smoke_fake_backend.py` |
| Smoke server: Qwen SFT 诊断 | `scripts/smoke_qwen_sft.py` |
| Smoke server: veRL trainer job | `scripts/smoke_verl_ppo_job.py` |
| 远端 SDK client smoke | `scripts/client_verl_ppo_job_smoke.py --base-url http://<host>:8000` |

先运行轻量 backend 的 contract smoke:

```bash
uv run --env-file /dev/null --extra smoke \
  python scripts/smoke_fake_backend.py
```

运行本地 Qwen SFT 诊断路径. 这条路径用于检查小型本地 PyTorch backend 的 API 兼容性; 推荐的 model-backed 训练路径是 veRL trainer job:

```bash
uv run --env-file /dev/null --extra smoke \
  python scripts/smoke_qwen_sft.py
```

运行 SDK 驱动的 veRL trainer job smoke:

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

所有 `client_*` 脚本都必须通过已安装的 MinT SDK 访问一个已经启动的服务, 并要求传入 `--base-url`. 会启动本地 smoke server 并立刻运行对应 SDK 场景的脚本放在 `scripts/smoke_*.py`. 直接向服务端 route 发 HTTP 请求的脚本属于 server/API 诊断, 不是 client contract 测试, 不应命名或视作 client smoke.

## 简单集群服务

开源 runtime 的部署应保持简单: 在网络可达的集群节点上启动一个 HTTP 服务, 让正式 SDK 指向它, 运行 smoke 或 example, 然后用 `Ctrl-C` 或 SIGTERM 停止. 这不是生产部署方案.

用 veRL backend 启动服务:

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

任意网络可达且不在 Ray 集群里的 client host:

```python
import mint

client = mint.ServiceClient(
    base_url="http://<cluster-node-ip>:8000",
    api_key="sk-local-smoke",
)
training = client.create_lora_training_client("Qwen/Qwen3-0.6B", rank=16)
print(training.get_info())
```

也可以用打包好的 SDK client smoke 访问这个服务:

```bash
uv run --env-file /dev/null --extra client \
  python scripts/client_verl_ppo_job_smoke.py \
  --base-url http://<cluster-node-ip>:8000 \
  --base-model Qwen/Qwen3-0.6B
```

## Ray 集群运行

`verl-mint` 可以把模型执行移动到 Ray 或 veRL worker, 同时让 FastAPI 进程保留 control plane 职责. 请从已有的 Ray 2.x 集群开始, Ray 版本需 `ray[default]>=2.46.0,<3`, 并确保 API 节点和 worker 使用一致的 Python 依赖和 artifact 根目录. 开源用法的部署形态是: 在 Ray 或 veRL 集群上启动一个 service 进程, client 可以运行在任意网络可达且不属于 Ray 集群的主机上.

在 Ray 或 veRL 集群节点上使用上面的 service 命令启动服务, 然后从网络可达且不在 Ray 集群里的 client host 使用 `--base-url` 运行 `client_*` smoke. `scripts/smoke_*.py` 是本地一次性 smoke server 入口, 不是远端 client 路径.

## 架构

`verl-mint` 运行一个 FastAPI control plane, 接收 MinT 风格训练请求, 通过 backend abstraction 路由执行, 并通过可配置的 storage repository 管理 checkpoint artifact. 本地诊断可以使用轻量 fake backend 和 Qwen backend; 推荐的 model-backed 路径使用 veRL trainer 执行, API 进程负责 session, future, checkpoint metadata 和 sampler handoff.

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
- 模型训练需要 CUDA GPU 才有实用速度
- 本地诊断需要 `torch`, `transformers`, `peft`, `accelerate`
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

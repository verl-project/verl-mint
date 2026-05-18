# Contributing to verl-mint

Thanks for taking the time to contribute. This project is still alpha, so the most useful contributions are narrow, tested changes that make the runtime easier to install, run, and integrate with MinT or veRL.

## Development Setup

Use Python 3.11 or newer. The repository is managed with `uv`.

```bash
git clone https://github.com/verl-project/verl-mint.git
cd verl-mint
uv sync --extra dev
uv pip install --torch-backend cpu \
  "torch>=2.3,<3" \
  "transformers>=4.44,<6" \
  "peft>=0.12,<1" \
  "accelerate>=0.33,<2"
```

If downloads are slow, use a temporary package mirror on the install command. This does not modify `uv.lock`:

```bash
uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple --torch-backend cpu \
  "torch>=2.3,<3" \
  "transformers>=4.44,<6" \
  "peft>=0.12,<1" \
  "accelerate>=0.33,<2"
```

Install the MinT SDK when you need to run the client smoke scripts:

```bash
uv pip install "git+https://github.com/MindLab-Research/mindlab-toolkit.git"
```

The same temporary mirror flag can be used for SDK dependency downloads:

```bash
uv pip install --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  "git+https://github.com/MindLab-Research/mindlab-toolkit.git"
```

## Tests

Run the CPU-safe unit test suite before opening a pull request:

```bash
uv run --env-file /dev/null pytest -q
```

For model-backed or cluster changes, also run the smallest relevant smoke script and include the command in the PR description:

```bash
uv run --env-file /dev/null --extra smoke \
  python scripts/smoke_fake_backend.py
```

Qwen, veRL, and Ray paths may require GPU access, shared storage, or a configured cluster. Keep those checks separate from CPU-only unit tests. Cluster smoke scripts assume an existing Ray 2.x environment using `ray[default]>=2.46.0,<3` with matching versions on the API node and workers. The veRL backend intentionally does not expose a `verl` package extra because worker images and GPU dependencies are environment-specific.

## Pull Requests

Use small, reviewable PRs. A good PR includes:

- a clear description of the behavior change
- tests for changed runtime behavior
- README or docs updates for user-facing changes
- migration notes for API or checkpoint compatibility changes
- the exact test and smoke commands you ran

Prefer Conventional Commit-style titles, such as `feat: add sampler checkpoint metadata`, `fix: validate shared storage roots`, or `docs: clarify cluster smoke setup`.

## Versioning

The project follows PEP 440-compatible semantic versioning while it is in the `0.x` series:

- patch releases, such as `0.1.2`, are for bug fixes and documentation-only changes
- minor releases, such as `0.2.0`, are for new features or breaking alpha API changes
- release tags use the `vX.Y.Z` format and should match `project.version` in `pyproject.toml`

Every release should update `CHANGELOG.md` before tagging.

## Release Checklist

Maintainers should use this checklist when preparing a release:

```bash
export MINT_SDK_REF=<reviewed-commit-or-tag>
uv sync --extra dev
uv pip install --torch-backend cpu \
  "torch>=2.3,<3" \
  "transformers>=4.44,<6" \
  "peft>=0.12,<1" \
  "accelerate>=0.33,<2" \
  "git+https://github.com/MindLab-Research/mindlab-toolkit.git@${MINT_SDK_REF}"
uv run --env-file /dev/null pytest -q
uv build
```

Then update `project.version` in `pyproject.toml`, update `CHANGELOG.md`, create and push a `vX.Y.Z` tag, attach the built artifacts to a GitHub Release, and publish to the package index selected by the maintainers.

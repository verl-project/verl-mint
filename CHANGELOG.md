# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project uses PEP 440-compatible semantic versioning.

## Unreleased

## 0.1.2 - 2026-05-17

### Added

- Added one-shot `smoke_*` scripts for local server smoke runs and separate `client_*` scripts for SDK-only remote service checks.
- Documented the simple cluster service workflow: start `verl-mint serve` on a Ray or veRL cluster node, then access it from a network-reachable non-Ray client host.

### Changed

- Updated smoke and client scripts to use the official `mindlab-toolkit` SDK dependency for client behavior instead of server internals.
- Sanitized default artifact path segments so generated `mint://` checkpoint and sampler paths avoid local path separators.
- Clarified README deployment guidance for fake, Qwen diagnostic, and veRL trainer job paths.

## 0.1.1 - 2026-05-14

### Added

- Added public contribution guidance for setup, tests, pull requests, and release versioning.
- Added package metadata for project URLs, classifiers, maintainers, keywords, and optional development/test/server extras.
- Added a public pull request CI workflow for unit tests and package builds.
- Added the MinT technical report citation to the READMEs.

### Changed

- Corrected README setup and smoke-script references for the current repository layout.

## 0.1.0 - 2026-04-30

### Added

- Initial alpha runtime with MinT-style training, inference, checkpoint, future, and rollout routes.
- Added local Qwen SFT/RL backend paths and veRL backend integration scaffolding.
- Added unit tests and client smoke scripts for core runtime flows.

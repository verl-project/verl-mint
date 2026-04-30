from __future__ import annotations

import copy
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
import uvicorn


class ServerThread(threading.Thread):
    def __init__(self, *, app: Any, host: str, port: int) -> None:
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not start on {host}:{port} within {timeout_s}s")


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_s = float(os.environ.get("VERL_MINT_HTTP_TIMEOUT_S", "300"))
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed: {exc.code} {detail}") from exc


def retrieve_future(base_url: str, future_payload: dict[str, Any], *, v1: bool = False) -> dict[str, Any]:
    path = "/api/v1/retrieve_future" if v1 else "/retrieve_future"
    return post_json(base_url, path, future_payload)


def assign_rewards(batch_payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(batch_payload)
    samples = payload.get("samples", [])
    base_rewards = [1.0, 0.25, -0.25, -0.5]
    default_group = str(payload.get("group_id") or payload.get("rollout_group_id") or "grpo-group")
    for i, sample in enumerate(samples):
        reward = base_rewards[i] if i < len(base_rewards) else max(-1.0, 1.0 - 0.25 * i)
        sample["reward"] = float(reward)
        sample["group_id"] = str(sample.get("group_id") or default_group)
        sample.setdefault("sample_id", str(i))
        sample.pop("reference_logprobs", None)
    payload["algorithm"] = "grpo"
    return payload


def assign_ppo_rewards(batch_payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(batch_payload)
    samples = payload.get("samples", [])
    base_rewards = [1.0, 0.25, -0.25, -0.5]
    for i, sample in enumerate(samples):
        reward = float(base_rewards[i] if i < len(base_rewards) else max(-1.0, 1.0 - 0.25 * i))
        completion_tokens = list(sample.get("completion_tokens", []))
        n = max(1, len(completion_tokens))
        sample["reward"] = reward
        sample["group_id"] = f"ppo-group-{i}"
        sample.setdefault("sample_id", str(i))
        sample["weights"] = [1.0 for _ in range(n)]
        sample["old_values"] = [0.0 for _ in range(n)]
        sample["returns"] = [reward for _ in range(n)]
        sample["advantages"] = [reward for _ in range(n)]
    payload["algorithm"] = "ppo"
    return payload


def state_delta_l2(before_path: str, after_path: str, state_key: str) -> float:
    before = torch.load(Path(before_path), map_location="cpu")
    after = torch.load(Path(after_path), map_location="cpu")
    before_state = before.get(state_key, {})
    after_state = after.get(state_key, {})
    total = 0.0
    for key, before_tensor in before_state.items():
        after_tensor = after_state.get(key)
        if after_tensor is None:
            continue
        diff = after_tensor.float() - before_tensor.float()
        total += float(torch.sum(diff * diff).item())
    return total ** 0.5


def checkpoint_delta_l2(before_path: str, after_path: str) -> float:
    return state_delta_l2(before_path, after_path, "model_state")

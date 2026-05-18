from __future__ import annotations

import mint
from mint import types

SMOKE_CLIENT_TOKEN = "sk-local-smoke"


def _service_client(base_url: str) -> mint.ServiceClient:
    return mint.ServiceClient(base_url=base_url, api_key=SMOKE_CLIENT_TOKEN)


def _tensor(data: list[int] | list[float], dtype: str) -> types.TensorData:
    return types.TensorData(data=data, dtype=dtype, shape=[len(data)])


def _model_input(tokens: list[int]) -> types.ModelInput:
    return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)])


def tiny_sft_datum() -> types.Datum:
    return types.Datum(
        model_input=_model_input([10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": _tensor([11, 12, 13], "int64"),
            "loss_mask": _tensor([1.0, 1.0, 1.0], "float32"),
        },
    )


def _rl_datum(
    *,
    prompt: list[int],
    completion: list[int],
    reward: float,
    sample_id: int,
    old_values: list[float] | None = None,
    returns: list[float] | None = None,
) -> types.Datum:
    n = len(completion)
    loss_inputs = {
        "target_tokens": _tensor(completion, "int64"),
        "weights": _tensor([1.0 for _ in completion], "float32"),
        "reward": _tensor([reward], "float32"),
        "sample_id": _tensor([sample_id], "int64"),
        "reference_logprobs": _tensor([0.0 for _ in completion], "float32"),
        "old_logprobs": _tensor([0.0 for _ in completion], "float32"),
        "advantages": _tensor([reward for _ in completion], "float32"),
    }
    if old_values is not None:
        loss_inputs["old_values"] = _tensor(old_values, "float32")
    if returns is not None:
        loss_inputs["returns"] = _tensor(returns, "float32")
    elif old_values is not None:
        loss_inputs["returns"] = _tensor([reward for _ in range(n)], "float32")
    return types.Datum(model_input=_model_input(prompt), loss_fn_inputs=loss_inputs)


def _dpo_datum(
    *,
    pair_id: int,
    role: int,
    prompt: list[int],
    completion: list[int],
) -> types.Datum:
    return types.Datum(
        model_input=_model_input([*prompt, *completion]),
        loss_fn_inputs={
            "pair_id": _tensor([pair_id], "int64"),
            "role": _tensor([role], "int64"),
            "prompt_tokens": _tensor(prompt, "int64"),
            "completion_tokens": _tensor(completion, "int64"),
            "target_tokens": _tensor(completion, "int64"),
            "weights": _tensor([1.0 for _ in completion], "float32"),
            "reference_logprobs": _tensor([0.0 for _ in completion], "float32"),
        },
    )


def run_fake_contract_smoke(base_url: str) -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=16,
        user_metadata={"smoke": "sdk-client-client"},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    fb = train.forward_backward([tiny_sft_datum()], loss_fn="cross_entropy").result()
    print("fb_metrics", fb.metrics)

    opt = train.optim_step(types.AdamParams(learning_rate=3e-4)).result()
    print("opt_metrics", opt.metrics)

    saved = train.save_state("checkpoint-001").result()
    print("saved_path", saved.path)

    loaded = train.load_state(saved.path).result()
    print("loaded_path", loaded.path)

    resumed = svc.create_training_client_from_state(saved.path)
    print("resumed_model_id", resumed.get_info().model_id)

    resumed_opt = svc.create_training_client_from_state_with_optimizer(saved.path)
    print("resumed_opt_model_id", resumed_opt.get_info().model_id)

    sampler_saved = train.save_weights_for_sampler("sampler-001").result()
    print("sampler_saved_path", sampler_saved.path)

    sampling_client = train.save_weights_and_get_sampling_client()
    sample = sampling_client.sample(
        prompt=_model_input([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).result()
    print("sample_sequences", len(sample.sequences))


def run_qwen_sft_smoke(base_url: str, *, base_model: str, smoke_label: str = "sdk-client-qwen-sft") -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model=base_model,
        rank=16,
        user_metadata={"smoke": smoke_label},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    fb = train.forward_backward([tiny_sft_datum()], loss_fn="cross_entropy").result()
    print("fb_metrics", fb.metrics)

    opt = train.optim_step(types.AdamParams(learning_rate=3e-4)).result()
    print("opt_metrics", opt.metrics)

    saved = train.save_state("checkpoint-001").result()
    print("saved_path", saved.path)

    loaded = train.load_state(saved.path).result()
    print("loaded_path", loaded.path)

    loaded_with_opt = train.load_state_with_optimizer(saved.path).result()
    print("loaded_with_optimizer_path", loaded_with_opt.path)

    sampler_saved = train.save_weights_for_sampler("sampler-001").result()
    print("sampler_saved_path", sampler_saved.path)

    named_sampling_client = train.create_sampling_client(sampler_saved.path)
    named_sample = named_sampling_client.sample(
        prompt=_model_input([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).result()
    print("named_sampler_sequences", len(named_sample.sequences))

    handoff_sampling_client = train.save_weights_and_get_sampling_client()
    handoff_sample = handoff_sampling_client.sample(
        prompt=_model_input([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).result()
    print("handoff_sampler_sequences", len(handoff_sample.sequences))


def run_qwen_grpo_smoke(
    base_url: str,
    *,
    base_model: str,
    learning_rate: float,
    smoke_label: str = "sdk-client-qwen-grpo",
) -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model=base_model,
        rank=16,
        user_metadata={"smoke": smoke_label},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    data = [
        _rl_datum(prompt=[101, 102, 103], completion=[104, 105], reward=1.0, sample_id=0),
        _rl_datum(prompt=[101, 102, 103], completion=[106, 107], reward=-0.5, sample_id=1),
    ]
    fb = train.forward_backward(
        data,
        loss_fn="grpo",
        loss_fn_config={"kl_coef": 0.05, "temperature": 1.0},
    ).result()
    print("grpo_metrics", fb.metrics)

    opt = train.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
    print("optim_metrics", opt.metrics)

    saved = train.save_state("grpo-checkpoint").result()
    print("saved_path", saved.path)
    loaded = train.load_state_with_optimizer(saved.path).result()
    print("loaded_path", loaded.path)


def run_qwen_ppo_smoke(
    base_url: str,
    *,
    base_model: str,
    learning_rate: float,
    smoke_label: str = "sdk-client-qwen-ppo",
) -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model=base_model,
        rank=16,
        user_metadata={"smoke": smoke_label},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    data = [
        _rl_datum(
            prompt=[101, 102, 103],
            completion=[104, 105],
            reward=1.0,
            sample_id=0,
            old_values=[0.0, 0.0],
        ),
        _rl_datum(
            prompt=[101, 102, 103],
            completion=[106, 107],
            reward=-0.5,
            sample_id=1,
            old_values=[0.0, 0.0],
        ),
    ]
    fb = train.forward_backward(
        data,
        loss_fn="ppo",
        loss_fn_config={
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "value_clip": 0.2,
        },
    ).result()
    print("ppo_metrics", fb.metrics)

    opt = train.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
    print("optim_metrics", opt.metrics)

    saved = train.save_state("ppo-checkpoint").result()
    print("saved_path", saved.path)
    loaded = train.load_state_with_optimizer(saved.path).result()
    print("loaded_path", loaded.path)


def run_qwen_dpo_smoke(
    base_url: str,
    *,
    base_model: str,
    learning_rate: float,
    beta: float,
) -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model=base_model,
        rank=16,
        user_metadata={"smoke": "sdk-client-qwen-dpo"},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    prompt = [101, 102, 103]
    data = [
        _dpo_datum(pair_id=0, role=1, prompt=prompt, completion=[104, 105]),
        _dpo_datum(pair_id=0, role=0, prompt=prompt, completion=[106, 107]),
    ]
    fb = train.forward_backward(data, loss_fn="dpo", loss_fn_config={"beta": beta}).result()
    print("dpo_metrics", fb.metrics)

    opt = train.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
    print("optim_metrics", opt.metrics)

    saved = train.save_state("dpo-checkpoint").result()
    print("saved_path", saved.path)


def run_verl_ppo_job_smoke(
    base_url: str,
    *,
    base_model: str,
    learning_rate: float = 1e-6,
) -> None:
    svc = _service_client(base_url)
    train = svc.create_lora_training_client(
        base_model=base_model,
        rank=16,
        user_metadata={"smoke": "sdk-client-verl-ppo-job"},
    )
    info = train.get_info()
    print("model_id", info.model_id)
    print("model_name", info.model_name)

    data = [
        _rl_datum(
            prompt=[101, 102, 103],
            completion=[104, 105],
            reward=1.0,
            sample_id=0,
            old_values=[0.0, 0.0],
        )
    ]
    fb = train.forward_backward(
        data,
        loss_fn="ppo",
        loss_fn_config={
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "value_clip": 0.2,
        },
    ).result()
    print("verl_ppo_metrics", fb.metrics)

    # The veRL job runner owns optimizer scheduling inside the trainer job.
    print("verl_ppo_learning_rate", learning_rate)

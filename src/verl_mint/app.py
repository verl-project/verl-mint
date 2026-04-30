from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from verl_mint.schemas import (
    CapabilitiesResponse,
    CollectExperienceRequest,
    CreateModelFromStateRequest,
    CreateModelFromStateResponse,
    CreateModelRequest,
    CreateModelResponse,
    ForwardBackwardOutput,
    ForwardBackwardPPORequest,
    ForwardBackwardPPOResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    ForwardBackwardReverseKLRequest,
    ForwardBackwardReverseKLResponse,
    ForwardRequest,
    ActRequest,
    ActResponse,
    CheckpointArchiveResponse,
    CheckpointInfo,
    CheckpointUploadResponse,
    WeightsInfoRequest,
    WeightsInfoResponse,
    CheckpointsListResponse,
    ComputeLogprobsRequest,
    ComputeLogprobsResponse,
    CreateActionSessionRequest,
    CreateActionSessionResponse,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ForwardResponse,
    FutureRetrieveRequest,
    GetInfoRequest,
    GetInfoResponse,
    GetModelResponse,
    GetSamplerResponse,
    GetSessionResponse,
    InterpolateCheckpointsRequest,
    InterpolateCheckpointsResponse,
    ListSessionsResponse,
    LoadStateRequest,
    LoadStateResponse,
    LossFnOutput,
    ModelData,
    ModelsListResponse,
    ModelSummary,
    OptimStepRequest,
    OptimStepResponse,
    PPOItemOutput,
    ResetExpertBiasRequest,
    ResetExpertBiasResponse,
    SampledSequence,
    SampleRequest,
    SampleResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    ReverseKLItemOutput,
    RolloutSessionRequest,
    SaveStateRequest,
    TelemetryRequest,
    TelemetryResponse,
    SaveStateResponse,
    TensorData,
    TokenizerInfoResponse,
    UntypedAPIFuture,
    TrainOnExperienceRequest,
    TrainingRun,
    TrainingRunsResponse,
    TrainStepRequest,
    TrainStepResponse,
)
from verl_mint.contracts import BackendKind
from verl_mint.future_store import InMemoryFutureStore, PendingFutureError
from verl_mint.service import MintService


def _model_id(session_id: str, model_seq_id: int) -> str:
    return f"{session_id}:{model_seq_id}"


def _parse_model_seq_id(model_id: str) -> int | None:
    parts = model_id.split(":", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _prompt_text(prompt: str | object) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        chunks = prompt.get("chunks", [])
    else:
        chunks = getattr(prompt, "chunks", [])
    parts: list[str] = []
    for chunk in chunks:
        chunk_type = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
        if chunk_type == "encoded_text":
            toks = chunk.get("tokens") if isinstance(chunk, dict) else getattr(chunk, "tokens", [])
            parts.append(" ".join(str(tok) for tok in toks))
        else:
            parts.append(f"<{chunk_type or 'chunk'}>")
    return " | ".join(parts)


def _tensor_data(value: float = 0.0) -> TensorData:
    return TensorData(data=value, shape=[], dtype="float32")


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _generate_details(result):
    raw = dict(getattr(result, "raw", {}) or {})
    token_ids = list(getattr(result, "token_ids", ()) or raw.get("token_ids") or [])
    token_logprobs = list(getattr(result, "token_logprobs", ()) or raw.get("token_logprobs") or [])
    prompt_token_ids = list(getattr(result, "prompt_token_ids", ()) or raw.get("prompt_token_ids") or [])
    prompt_logprobs = list(getattr(result, "prompt_logprobs", ()) or raw.get("prompt_logprobs") or [])
    stop_reason = str(getattr(result, "stop_reason", None) or raw.get("stop_reason") or "length")
    return {
        "raw": raw,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "prompt_token_ids": prompt_token_ids,
        "prompt_logprobs": prompt_logprobs,
        "stop_reason": stop_reason,
    }


def _result_loss_value(result) -> float:
    raw_loss = result.outputs.get("loss") if hasattr(result, "outputs") else None
    if isinstance(raw_loss, (int, float)):
        return float(raw_loss)
    return float(result.state.step) if result.state else 0.0


def _loss_outputs(result, *, token_count: int = 1) -> list[LossFnOutput]:
    loss_value = _result_loss_value(result)
    logprobs = [-(loss_value)] * max(1, token_count)
    return [
        LossFnOutput(
            loss=_tensor_data(loss_value),
            logprobs=TensorData(data=logprobs, shape=[len(logprobs)], dtype="float32"),
            metadata=dict(result.outputs),
        )
    ]


def _token_stats(forward_backward_input: dict | None) -> tuple[float, float]:
    if not forward_backward_input:
        return 0.0, 0.0
    data = forward_backward_input.get("data", [])
    sample_count = float(len(data))
    token_count = 0.0
    for item in data:
        loss_inputs = item.get("loss_fn_inputs", {})
        target = loss_inputs.get("target_tokens")
        if isinstance(target, dict):
            shape = target.get("shape", [])
            if shape:
                token_count += float(shape[0])
                continue
        model_input = item.get("model_input", {})
        chunks = model_input.get("chunks", [])
        for chunk in chunks:
            chunk_type = chunk.get("type")
            if chunk_type == "encoded_text" or (chunk_type is None and "tokens" in chunk):
                token_count += float(len(chunk.get("tokens", [])))
    return sample_count, token_count


def _fb_metrics(result, *, forward_backward_input: dict | None = None) -> dict[str, float]:
    sample_count, token_count = _token_stats(forward_backward_input)
    metrics = {
        "loss:mean": _result_loss_value(result),
        "num_samples:sum": sample_count,
        "num_tokens:sum": token_count,
    }
    for k, v in result.outputs.items():
        if isinstance(v, (int, float)):
            metric_key = k if ":" in k else f"{k}:last"
            metrics[metric_key] = float(v)
    return metrics


def _fb_output(result, *, loss_fn: str | None = None, forward_backward_input: dict | None = None) -> ForwardBackwardOutput:
    return ForwardBackwardOutput(
        loss_fn_output_type=f"{loss_fn}_loss" if loss_fn else "loss",
        loss_fn_outputs=_loss_outputs(result, token_count=int(_token_stats(forward_backward_input)[1] or 1)),
        metrics=_fb_metrics(result, forward_backward_input=forward_backward_input),
    )


def _fb_output_v1(result, *, loss_fn: str | None = None, forward_backward_input: dict | None = None) -> dict:
    token_count = int(_token_stats(forward_backward_input)[1] or 1)
    loss_value = _result_loss_value(result)
    logprobs = [-(loss_value)] * max(1, token_count)
    return {
        "loss_fn_output_type": f"{loss_fn}_loss" if loss_fn else "loss",
        "loss_fn_outputs": [
            {
                "loss": TensorData(
                    data=[loss_value],
                    shape=[],
                    dtype="float32",
                ).model_dump(),
                "logprobs": TensorData(data=logprobs, shape=[len(logprobs)], dtype="float32").model_dump(),
            }
        ],
        "metrics": _fb_metrics(result, forward_backward_input=forward_backward_input),
    }


def _select_backend_id(service: MintService, kind: BackendKind, requested: str | None) -> str:
    if requested:
        return requested
    for spec in service.backends.list():
        if spec.kind == kind:
            return spec.backend_id
    raise HTTPException(status_code=400, detail=f"No {kind.value} backend registered")


def create_app(service: MintService | None = None) -> FastAPI:
    service = service or MintService()
    future_store = InMemoryFutureStore()
    app = FastAPI(title="verl-mint")
    app.state.mint_service = service
    app.state.future_store = future_store
    app.state.session_index = {}
    app.state.sampling_sessions = {}
    app.state.action_sessions = {}
    app.state.checkpoints = {}
    app.state.weights_info = {}

    def _future(payload, **future_kwargs) -> UntypedAPIFuture:
        return UntypedAPIFuture(request_id=future_store.create_resolved(payload, **future_kwargs))

    def _session_entry(session_id: str) -> dict:
        store = app.state.session_index
        if session_id not in store:
            store[session_id] = {"training_run_ids": [], "sampler_ids": [], "tags": [], "user_metadata": {}}
        return store[session_id]

    def _checkpoint_entry(model_id: str) -> list[dict]:
        store = app.state.checkpoints
        if model_id not in store:
            store[model_id] = []
        return store[model_id]

    def _has_uri_scheme(path: str | None) -> bool:
        return bool(path) and "://" in path

    def _public_uri_aliases(path: str) -> set[str]:
        aliases = {path}
        if path.startswith("mint://"):
            aliases.add("tinker://" + path[len("mint://") :])
        elif path.startswith("tinker://"):
            aliases.add("mint://" + path[len("tinker://") :])
        return aliases

    def _training_checkpoint_uri(model_id: str, requested_path: str | None, *, versioned: bool) -> str:
        if _has_uri_scheme(requested_path):
            return str(requested_path)
        if versioned:
            checkpoint_id = requested_path or "checkpoint.pt"
            return f"mint://{model_id}/weights/{checkpoint_id}"
        return requested_path or f"mint://checkpoints/{model_id}.pt"

    def _sampler_checkpoint_uri(model_id: str, requested_path: str | None, *, versioned: bool, sampling_session_seq_id: int | None = None) -> str:
        if _has_uri_scheme(requested_path):
            return str(requested_path)
        if versioned:
            seq = sampling_session_seq_id if sampling_session_seq_id is not None else 1
            checkpoint_id = requested_path or f"sampler-{seq}.pt"
            return f"mint://{model_id}/sampler_weights/{checkpoint_id}"
        return requested_path or f"mint://sampler/{model_id}.pt"

    def _sampling_session_id(session_id: str, seq_id: int) -> str:
        return f"{session_id}:sampler:{seq_id}"

    def _select_inference_backend_for_base_model(base_model: str | None) -> str:
        specs = [spec for spec in service.backends.list() if spec.kind == BackendKind.INFERENCE]
        if not specs:
            raise HTTPException(status_code=400, detail="No inference backend registered")
        if base_model:
            key = str(base_model).lower()
            matches = [
                spec.backend_id
                for spec in specs
                if key in spec.model_family.lower() or spec.model_family.lower() in key
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Ambiguous inference backend for base_model '{base_model}'",
                )
        if len(specs) == 1:
            return specs[0].backend_id
        raise HTTPException(
            status_code=400,
            detail="Multiple inference backends registered; base_model must match one model_family",
        )

    def _register_sampling_session(
        *,
        session_id: str,
        seq_id: int,
        base_model: str | None,
        model_path: str | None,
        inference_backend_id: str | None,
        lora_rank: int = 32,
    ) -> str:
        sampling_session_id = _sampling_session_id(session_id, seq_id)
        app.state.sampling_sessions[sampling_session_id] = {
            "session_id": session_id,
            "base_model": base_model or "unknown",
            "model_path": model_path,
            "inference_backend_id": inference_backend_id,
            "lora_rank": lora_rank,
        }
        session = _session_entry(session_id)
        if sampling_session_id not in session["sampler_ids"]:
            session["sampler_ids"].append(sampling_session_id)
        return sampling_session_id

    def _require_sampling_session(sampling_session_id: str) -> dict:
        sampler = app.state.sampling_sessions.get(sampling_session_id)
        if sampler is None:
            raise HTTPException(status_code=404, detail=f"Sampler '{sampling_session_id}' not found")
        return sampler

    def _require_sampler_artifact(sampler: dict) -> None:
        model_path = sampler.get("model_path")
        if not model_path:
            return
        try:
            service.storage.resolve_for_read(str(model_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Sampler artifact '{model_path}' not found") from exc

    def _model_checkpoint_info(model_id: str) -> dict[str, object] | None:
        try:
            view = service.training.get_model(model_id)
        except Exception:
            return None
        metadata = dict(view.metadata)
        lora_config = metadata.get("lora_config") or {}
        rank = lora_config.get("rank") if hasattr(lora_config, "get") else None
        return {
            "base_model": str(metadata.get("base_model") or view.backend_id),
            "is_lora": bool(lora_config),
            "lora_rank": rank,
            "train_unembed": lora_config.get("train_unembed") if hasattr(lora_config, "get") else None,
            "train_mlp": lora_config.get("train_mlp") if hasattr(lora_config, "get") else None,
            "train_attn": lora_config.get("train_attn") if hasattr(lora_config, "get") else None,
        }

    def _record_checkpoint(model_id: str, *, path: str, checkpoint_type: str) -> dict:
        entry = {
            "checkpoint_id": path.split("/")[-1],
            "checkpoint_type": checkpoint_type,
            "time": "1970-01-01T00:00:00Z",
            "mint_path": path,
            "created_at": "1970-01-01T00:00:00Z",
            "size_bytes": None,
            "public": False,
            "expires_at": None,
            "storage_tier": "local_fs",
            "mirror_status": None,
            "mirror_error": None,
        }
        _checkpoint_entry(model_id).append(entry)
        if checkpoint_type == "training":
            info = _model_checkpoint_info(model_id)
            if info is not None:
                for alias in _checkpoint_uri_aliases(model_id, entry):
                    app.state.weights_info[alias] = info
        return entry

    def _checkpoint_uri_aliases(model_id: str, entry: dict) -> set[str]:
        path = str(entry["mint_path"])
        aliases = _public_uri_aliases(path)
        if not _has_uri_scheme(path):
            checkpoint_kind = "weights" if entry["checkpoint_type"] == "training" else "sampler_weights"
            aliases.add(f"tinker://{model_id}/{checkpoint_kind}/{entry['checkpoint_id']}")
        return aliases

    def _checkpoint_public_path(model_id: str, entry: dict, *, versioned: bool) -> str:
        path = str(entry["mint_path"])
        if not versioned:
            return path
        if path.startswith("tinker://"):
            return path
        if path.startswith("mint://"):
            return "tinker://" + path[len("mint://") :]
        checkpoint_kind = "weights" if entry["checkpoint_type"] == "training" else "sampler_weights"
        return f"tinker://{model_id}/{checkpoint_kind}/{entry['checkpoint_id']}"

    def _checkpoint_payload(model_id: str, entry: dict, *, versioned: bool) -> dict:
        public_path = _checkpoint_public_path(model_id, entry, versioned=versioned)
        if versioned:
            return {
                "checkpoint_id": entry["checkpoint_id"],
                "checkpoint_type": entry["checkpoint_type"],
                "time": entry["time"],
                "tinker_path": public_path,
                "size_bytes": entry["size_bytes"],
                "public": entry["public"],
                "expires_at": entry["expires_at"],
            }
        return CheckpointInfo(
            checkpoint_id=entry["checkpoint_id"],
            checkpoint_type=entry["checkpoint_type"],
            time=entry["time"],
            mint_path=public_path,
            created_at=entry["created_at"],
            size_bytes=entry["size_bytes"],
            public=entry["public"],
            expires_at=entry["expires_at"],
            storage_tier=entry["storage_tier"],
            mirror_status=entry["mirror_status"],
            mirror_error=entry["mirror_error"],
        ).model_dump()

    def _latest_checkpoint(model_id: str, checkpoint_type: str, *, versioned: bool) -> dict | None:
        for entry in reversed(_checkpoint_entry(model_id)):
            if entry["checkpoint_type"] == checkpoint_type:
                return _checkpoint_payload(model_id, entry, versioned=versioned)
        return None

    def _stringify_user_metadata(metadata: dict) -> dict[str, str] | None:
        user_metadata = metadata.get("user_metadata")
        if not hasattr(user_metadata, "items"):
            return None
        return {str(key): str(value) for key, value in user_metadata.items()}

    def _training_run_payload(view, *, versioned: bool) -> dict:
        metadata = dict(view.metadata)
        lora_config = metadata.get("lora_config") or {}
        payload = {
            "training_run_id": view.session_id,
            "base_model": str(metadata.get("base_model") or view.backend_id),
            "model_owner": str(metadata.get("model_owner") or metadata.get("owner") or "local"),
            "is_lora": bool(lora_config),
            "corrupted": view.status == "error",
            "lora_rank": lora_config.get("rank") if hasattr(lora_config, "get") else None,
            "last_request_time": str(
                metadata.get("last_request_time")
                or metadata.get("updated_at")
                or metadata.get("created_at")
                or "1970-01-01T00:00:00Z"
            ),
            "last_checkpoint": _latest_checkpoint(view.session_id, "training", versioned=versioned),
            "last_sampler_checkpoint": _latest_checkpoint(view.session_id, "sampler", versioned=versioned),
            "user_metadata": _stringify_user_metadata(metadata),
        }
        if not versioned:
            payload["last_activity"] = 0.0
            payload["idle_for_s"] = 0.0
        return payload

    def _find_checkpoint(model_id: str, checkpoint_id: str) -> dict | None:
        for entry in _checkpoint_entry(model_id):
            if entry["checkpoint_id"] == checkpoint_id:
                return entry
        return None

    def _delete_checkpoint_entry(model_id: str, checkpoint_id: str) -> dict | None:
        entries = _checkpoint_entry(model_id)
        for i, entry in enumerate(entries):
            if entry["checkpoint_id"] == checkpoint_id:
                removed = entries.pop(i)
                for alias in _checkpoint_uri_aliases(model_id, removed):
                    app.state.weights_info.pop(alias, None)
                return removed
        return None

    @app.get("/api/v1/healthz")
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/api/v1/get_server_capabilities")
    @app.get("/get_server_capabilities")
    def get_server_capabilities():
        supported = []
        for spec in service.backends.list():
            if spec.kind.value == "inference":
                supported.append({"model_name": spec.model_family, "max_context_length": 32768})
        return {"supported_models": supported, "status": "ready", "gateway_errors": []}

    @app.get("/api/v1/server_info")
    @app.get("/server_info")
    def server_info():
        return {"server": "verl-mint", "storage_root": str(service.storage.root)}

    @app.post("/api/v1/create_session")
    @app.post("/create_session")
    def create_session(body: CreateSessionRequest):
        session_id = f"session_{len(app.state.session_index) + 1}"
        entry = _session_entry(session_id)
        entry["tags"] = list(body.tags)
        entry["user_metadata"] = dict(body.user_metadata)
        return CreateSessionResponse(session_id=session_id)

    @app.post("/api/v1/create_sampling_session")
    @app.post("/create_sampling_session")
    def create_sampling_session(body: CreateSamplingSessionRequest):
        seq = body.sampling_session_seq_id if body.sampling_session_seq_id is not None else len(app.state.sampling_sessions) + 1
        inference_backend_id = _select_inference_backend_for_base_model(body.base_model)
        sampling_session_id = _register_sampling_session(
            session_id=body.session_id,
            seq_id=seq,
            base_model=body.base_model,
            model_path=body.model_path,
            inference_backend_id=inference_backend_id,
            lora_rank=body.lora_rank,
        )
        return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

    @app.get("/api/v1/sessions/{session_id}")
    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        entry = _session_entry(session_id)
        return GetSessionResponse(
            training_run_ids=list(entry["training_run_ids"]),
            sampler_ids=list(entry["sampler_ids"]),
        )

    @app.get("/api/v1/sessions")
    @app.get("/sessions")
    def list_sessions():
        return ListSessionsResponse(sessions=list(app.state.session_index.keys()))

    @app.get("/api/v1/samplers/{sampler_id}")
    @app.get("/samplers/{sampler_id}")
    def get_sampler(sampler_id: str):
        sampler = app.state.sampling_sessions.get(sampler_id)
        if sampler is None:
            raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")
        return GetSamplerResponse(
            sampler_id=sampler_id,
            base_model=str(sampler.get("base_model") or "unknown"),
            model_path=sampler.get("model_path"),
        )

    @app.post("/api/v1/session_heartbeat")
    @app.post("/session_heartbeat")
    def session_heartbeat(body: SessionHeartbeatRequest):
        _session_entry(body.session_id)
        return SessionHeartbeatResponse()

    @app.post("/api/v1/telemetry")
    @app.post("/telemetry")
    def telemetry(body: TelemetryRequest):
        return TelemetryResponse()

    @app.post("/api/v1/asample")
    @app.post("/asample")
    def asample(body: SampleRequest):
        sampler_model_path = body.model_path
        backend_id = (
            _select_inference_backend_for_base_model(body.base_model)
            if body.base_model is not None
            else _select_backend_id(service, BackendKind.INFERENCE, None)
        )
        if body.sampling_session_id is not None:
            sampler = _require_sampling_session(body.sampling_session_id)
            _require_sampler_artifact(sampler)
            sampler_model_path = sampler.get("model_path") or sampler_model_path
            backend_id = str(sampler.get("inference_backend_id") or backend_id)
        prompt_text = _prompt_text(body.prompt)
        prompt_token_count = 0
        for chunk in body.prompt.chunks:
            if getattr(chunk, "type", None) == "encoded_text":
                prompt_token_count += len(chunk.tokens)
        max_new_tokens = max(1, int(body.sampling_params.max_tokens))
        sequences = []
        prompt_logprobs = None
        for _ in range(body.num_samples):
            sampling = body.sampling_params.model_dump()
            sampling["prompt_logprobs"] = bool(body.prompt_logprobs or body.include_prompt_logprobs)
            if sampler_model_path:
                sampling["model_path"] = sampler_model_path
            generated = service.inference.generate(
                backend_id=backend_id,
                prompt=prompt_text,
                sampling=sampling,
            )
            details = _generate_details(generated)
            generated_tokens = details["token_ids"]
            if not generated_tokens:
                generated_tokens = [int(tok) for tok in generated.text.split() if tok.lstrip("-").isdigit()]
            if not generated_tokens:
                generated_tokens = [len(generated.text)]
            generated_logprobs = details["token_logprobs"] or [-0.1 for _ in generated_tokens]
            if len(generated_tokens) > max_new_tokens:
                generated_tokens = generated_tokens[:max_new_tokens]
                generated_logprobs = generated_logprobs[:max_new_tokens]
            sequences.append(
                SampledSequence(
                    tokens=generated_tokens,
                    logprobs=generated_logprobs,
                    stop_reason=details["stop_reason"],
                )
            )
            if prompt_logprobs is None and (body.prompt_logprobs or body.include_prompt_logprobs):
                prompt_logprobs = details["prompt_logprobs"] or [None] + [-0.1 for _ in range(max(0, prompt_token_count - 1))]
        response = SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=None,
        )
        return _future(response.model_dump())

    @app.post("/api/v1/compute_logprobs")
    @app.post("/compute_logprobs")
    def compute_logprobs(body: ComputeLogprobsRequest):
        sampler = _require_sampling_session(body.sampling_session_id)
        _require_sampler_artifact(sampler)
        backend_id = str(
            sampler.get("inference_backend_id")
            or _select_backend_id(service, BackendKind.INFERENCE, None)
        )
        sampling = {"mode": "logprobs", "prompt_logprobs": True}
        model_path = sampler.get("model_path")
        if model_path:
            sampling["model_path"] = model_path
        generated = service.inference.generate(
            backend_id=backend_id,
            prompt=_prompt_text(body.sequence),
            sampling=sampling,
        )
        details = _generate_details(generated)
        logprobs = details["prompt_logprobs"]
        if not logprobs:
            token_count = 0
            for chunk in body.sequence.chunks:
                if getattr(chunk, "type", None) == "encoded_text":
                    token_count += len(chunk.tokens)
            base_logprob = -max(0.1, float(len(generated.text.split()) or 1))
            logprobs = [None] + ([base_logprob] * max(0, token_count - 1))
        response = ComputeLogprobsResponse(logprobs=logprobs)
        return _future(response.model_dump())

    @app.post("/api/v1/action_sessions")
    @app.post("/action_sessions")
    def create_action_session(body: CreateActionSessionRequest):
        seq = body.action_session_seq_id if body.action_session_seq_id is not None else len(app.state.action_sessions) + 1
        action_session_id = f"{body.session_id}:action:{seq}"
        app.state.action_sessions[action_session_id] = {
            "session_id": body.session_id,
            "base_model": body.base_model or "unknown",
            "model_path": body.model_path,
        }
        return CreateActionSessionResponse(action_session_id=action_session_id)

    @app.post("/api/v1/action_sessions/{action_session_id}/act")
    @app.post("/action_sessions/{action_session_id}/act")
    def act(action_session_id: str, body: ActRequest):
        if action_session_id not in app.state.action_sessions:
            raise HTTPException(status_code=404, detail=f"Action session '{action_session_id}' not found")
        response = ActResponse(actions=TensorData(data=[0.0], shape=[1], dtype="float32"), policy_timing={"latency_s": 0.0})
        return _future(response.model_dump())

    @app.delete("/api/v1/action_sessions/{action_session_id}")
    @app.delete("/action_sessions/{action_session_id}")
    def delete_action_session(action_session_id: str):
        app.state.action_sessions.pop(action_session_id, None)
        return {"action_session_id": action_session_id, "status": "deleted"}

    @app.post("/api/v1/checkpoints/interpolate")
    @app.post("/checkpoints/interpolate")
    def interpolate_checkpoints(body: InterpolateCheckpointsRequest):
        path = body.output_path or "mint://interpolated/checkpoint.pt"
        return _future(
            InterpolateCheckpointsResponse(
                path=path,
                source_paths=body.source_paths,
                coefficients=body.coefficients,
            ).model_dump()
        )

    @app.post("/create_model")
    def create_model(body: CreateModelRequest):
        model_id = _model_id(body.session_id, body.model_seq_id)
        backend_id = _select_backend_id(service, BackendKind.TRAINING, body.backend_id)
        view = service.training.create_model(
            model_id=model_id,
            backend_id=backend_id,
            batch_codec=body.batch_codec,
            metadata={
                "session_id": body.session_id,
                "base_model": body.base_model,
                "user_metadata": body.user_metadata,
                "lora_config": body.lora_config.model_dump() if body.lora_config else None,
                "rollout_correction_config": (
                    body.rollout_correction_config.model_dump()
                    if body.rollout_correction_config
                    else None
                ),
                "request_type": body.type,
            },
        )
        session = _session_entry(body.session_id)
        if view.session_id not in session["training_run_ids"]:
            session["training_run_ids"].append(view.session_id)
        session["tags"] = []
        session["user_metadata"] = body.user_metadata or {}
        payload = CreateModelResponse(
            request_id=f"create:{view.session_id}",
            model_id=view.session_id,
            backend=backend_id,
        )
        return _future(payload.model_dump())

    @app.post("/create_model_from_state")
    def create_model_from_state(body: CreateModelFromStateRequest):
        model_id = _model_id(body.session_id, body.model_seq_id)
        backend_id = _select_backend_id(service, BackendKind.TRAINING, body.backend_id)
        view = service.training.create_model_from_state(
            model_id=model_id,
            backend_id=backend_id,
            batch_codec=body.batch_codec,
            uri=body.state_path,
            include_optimizer=body.load_optimizer,
            metadata={
                "session_id": body.session_id,
                "base_model": body.base_model,
                "user_metadata": body.user_metadata,
                "lora_config": body.lora_config.model_dump() if body.lora_config else None,
                "rollout_correction_config": (
                    body.rollout_correction_config.model_dump()
                    if body.rollout_correction_config
                    else None
                ),
                "request_type": body.type,
            },
        )
        session = _session_entry(body.session_id)
        if view.session_id not in session["training_run_ids"]:
            session["training_run_ids"].append(view.session_id)
        session["user_metadata"] = body.user_metadata or {}
        payload = CreateModelFromStateResponse(
            request_id=f"create_from_state:{view.session_id}",
            model_id=view.session_id,
        )
        return _future(payload.model_dump())

    @app.post("/api/v1/create_model")
    def create_model_v1(body: CreateModelRequest):
        model_id = _model_id(body.session_id, body.model_seq_id)
        backend_id = _select_backend_id(service, BackendKind.TRAINING, body.backend_id)
        view = service.training.create_model(
            model_id=model_id,
            backend_id=backend_id,
            batch_codec=body.batch_codec,
            metadata={
                "session_id": body.session_id,
                "base_model": body.base_model,
                "user_metadata": body.user_metadata,
                "lora_config": body.lora_config.model_dump() if body.lora_config else None,
                "rollout_correction_config": (
                    body.rollout_correction_config.model_dump()
                    if body.rollout_correction_config
                    else None
                ),
                "request_type": body.type,
            },
        )
        session = _session_entry(body.session_id)
        if view.session_id not in session["training_run_ids"]:
            session["training_run_ids"].append(view.session_id)
        session["tags"] = []
        session["user_metadata"] = body.user_metadata or {}
        return _future({"model_id": view.session_id, "type": "create_model"})

    @app.post("/api/v1/create_model_from_state")
    def create_model_from_state_v1(body: CreateModelFromStateRequest):
        model_id = _model_id(body.session_id, body.model_seq_id)
        backend_id = _select_backend_id(service, BackendKind.TRAINING, body.backend_id)
        view = service.training.create_model_from_state(
            model_id=model_id,
            backend_id=backend_id,
            batch_codec=body.batch_codec,
            uri=body.state_path,
            include_optimizer=body.load_optimizer,
            metadata={
                "session_id": body.session_id,
                "base_model": body.base_model,
                "user_metadata": body.user_metadata,
                "lora_config": body.lora_config.model_dump() if body.lora_config else None,
                "rollout_correction_config": (
                    body.rollout_correction_config.model_dump()
                    if body.rollout_correction_config
                    else None
                ),
                "request_type": body.type,
            },
        )
        session = _session_entry(body.session_id)
        if view.session_id not in session["training_run_ids"]:
            session["training_run_ids"].append(view.session_id)
        session["user_metadata"] = body.user_metadata or {}
        return _future({"model_id": view.session_id, "type": "create_model_from_state"})

    @app.post("/api/v1/unload_model")
    @app.post("/unload_model")
    def unload_model_v1(body: dict):
        model_id = str(body.get("model_id") or "")
        if not model_id:
            raise HTTPException(status_code=422, detail="model_id is required")
        service.training.delete_model(model_id)
        return _future({"model_id": model_id, "type": "unload_model"})

    @app.post("/api/v1/forward")
    def forward_v1(body: ForwardRequest):
        payload = body.forward_input.model_dump()
        result = service.training.forward(
            body.model_id,
            batch_payload=payload,
            options={"seq_id": body.seq_id},
        )
        return _future(
            _fb_output_v1(
                result,
                loss_fn=body.forward_input.loss_fn,
                forward_backward_input=payload,
            )
        )

    @app.post("/api/v1/forward_backward")
    def forward_backward_v1(body: ForwardBackwardRequest):
        payload = body.forward_backward_input.model_dump()
        result = service.training.forward_backward(
            body.model_id,
            batch_payload=payload,
            options={"seq_id": body.seq_id},
        )
        return _future(
            _fb_output_v1(
                result,
                loss_fn=body.forward_backward_input.loss_fn,
                forward_backward_input=payload,
            )
        )

    @app.post("/api/v1/optim_step")
    def optim_step_v1(body: OptimStepRequest):
        result = service.training.optim_step(
            body.model_id,
            batch_payload={},
            options={"adam_params": body.adam_params.model_dump(), "seq_id": body.seq_id},
        )
        metrics = {"step": float(result.state.step) if result.state else 0.0}
        return _future(OptimStepResponse(metrics=metrics).model_dump(exclude_none=True))

    @app.post("/forward")
    def forward(body: ForwardRequest):
        payload = body.forward_input.model_dump()
        result = service.training.forward(
            body.model_id,
            batch_payload=payload,
            options={"seq_id": body.seq_id},
        )
        response = ForwardResponse(
            output=_fb_output(
                result,
                loss_fn=body.forward_input.loss_fn,
                forward_backward_input=payload,
            )
        )
        return _future(response.model_dump())

    @app.post("/forward_backward")
    def forward_backward(body: ForwardBackwardRequest):
        payload = body.forward_backward_input.model_dump()
        result = service.training.forward_backward(
            body.model_id,
            batch_payload=payload,
            options={"seq_id": body.seq_id},
        )
        response = ForwardBackwardResponse(
            output=_fb_output(
                result,
                loss_fn=body.forward_backward_input.loss_fn,
                forward_backward_input=payload,
            )
        )
        return _future(response.model_dump())

    @app.post("/api/v1/forward_backward_ppo")
    @app.post("/forward_backward_ppo")
    def forward_backward_ppo(body: ForwardBackwardPPORequest):
        result = service.training.forward_backward_ppo(
            body.model_id,
            batch_payload={
                "data": [item.model_dump() for item in body.data],
                "algorithm": "ppo",
                "clip_coef": body.clip_coef,
                "value_coef": body.value_coef,
                "entropy_coef": body.entropy_coef,
                "value_clip": body.value_clip,
            },
            options={
                "clip_coef": body.clip_coef,
                "value_coef": body.value_coef,
                "entropy_coef": body.entropy_coef,
                "value_clip": body.value_clip,
                "seq_id": body.seq_id,
            },
        )
        sequence_losses = list(result.outputs.get("sequence_losses", [])) if hasattr(result, "outputs") else []
        if sequence_losses:
            outputs = [PPOItemOutput(loss=_tensor_data(float(loss))) for loss in sequence_losses]
        else:
            outputs = [PPOItemOutput(loss=_tensor_data(_result_loss_value(result)))]
        metrics = {}
        for key in [
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "reward_mean",
            "adv_mean",
            "return_mean",
            "value_mean",
            "num_tokens",
        ]:
            value = result.outputs.get(key) if hasattr(result, "outputs") else None
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        if "loss" in metrics:
            metrics["loss:mean"] = metrics["loss"]
        elif result.state is not None:
            metrics["loss:mean"] = _result_loss_value(result)
        response = ForwardBackwardPPOResponse(outputs=outputs, metrics=metrics)
        return _future(response.model_dump())

    @app.post("/api/v1/forward_backward_reverse_kl")
    @app.post("/forward_backward_reverse_kl")
    def forward_backward_reverse_kl(body: ForwardBackwardReverseKLRequest):
        result = service.training.forward_backward_reverse_kl(
            body.model_id,
            batch_payload={
                "reference_model_path": body.reference_model_path,
                "data": [item.model_dump() for item in body.data],
                "algorithm": "grpo",
            },
            options={"temperature": body.temperature, "seq_id": body.seq_id},
        )
        sequence_losses = list(result.outputs.get("sequence_losses", [])) if hasattr(result, "outputs") else []
        if sequence_losses:
            outputs = [ReverseKLItemOutput(loss=_tensor_data(float(loss))) for loss in sequence_losses]
        else:
            outputs = [ReverseKLItemOutput(loss=_tensor_data(_result_loss_value(result)))]
        metrics = {}
        for key in ["loss", "policy_loss", "kl", "reward_mean", "adv_mean", "num_tokens"]:
            value = result.outputs.get(key) if hasattr(result, "outputs") else None
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        if "loss" in metrics:
            metrics["loss:mean"] = metrics["loss"]
        elif result.state is not None:
            metrics["loss:mean"] = _result_loss_value(result)
        response = ForwardBackwardReverseKLResponse(outputs=outputs, metrics=metrics)
        return _future(response.model_dump())

    @app.post("/optim_step")
    def optim_step(body: OptimStepRequest):
        result = service.training.optim_step(
            body.model_id,
            batch_payload={},
            options={"adam_params": body.adam_params.model_dump(), "seq_id": body.seq_id},
        )
        metrics = {"step": float(result.state.step) if result.state else 0.0}
        response = OptimStepResponse(metrics=metrics)
        return _future(response.model_dump())

    @app.post("/api/v1/train_step")
    @app.post("/train_step")
    def train_step(body: TrainStepRequest):
        payload = body.forward_backward_input.model_dump()
        result = service.training.train_step(
            body.model_id,
            batch_payload=payload,
            options={
                "adam_params": body.adam_params.model_dump() if body.adam_params else None,
                "seq_id": body.seq_id,
            },
        )
        fb_output = _fb_output(
            result,
            loss_fn=body.forward_backward_input.loss_fn,
            forward_backward_input=payload,
        )
        metrics = dict(fb_output.metrics)
        if body.adam_params is not None:
            metrics["learning_rate"] = float(body.adam_params.learning_rate)
        response = TrainStepResponse(output=fb_output, metrics=metrics)
        return _future(response.model_dump())

    @app.post("/api/v1/save_state")
    @app.post("/api/v1/save_weights")
    @app.post("/save_state")
    @app.post("/save_weights")
    def save_weights(body: SaveStateRequest, request: Request):
        public_uri = _training_checkpoint_uri(body.model_id, body.path, versioned=request.url.path.startswith("/api/v1/"))
        artifact = service.training.save_weights(
            model_id=body.model_id,
            uri=public_uri,
            include_optimizer=True,
            metadata={
                "ttl_seconds": body.ttl_seconds,
                "seq_id": body.seq_id,
                "type": body.type,
            },
        )
        _record_checkpoint(body.model_id, path=artifact.uri, checkpoint_type="training")
        response = SaveStateResponse(path=artifact.uri)
        return _future(response.model_dump())

    @app.post("/api/v1/load_state")
    @app.post("/api/v1/load_weights")
    @app.post("/load_state")
    @app.post("/load_weights")
    def load_weights(body: LoadStateRequest):
        state = service.training.load_weights(
            model_id=body.model_id,
            uri=body.path,
            include_optimizer=body.optimizer,
            metadata={"seq_id": body.seq_id, "type": body.type},
        )
        resolved_path = body.path
        if state.step >= 0:
            resolved_path = body.path
        response = LoadStateResponse(path=resolved_path)
        return _future(response.model_dump())

    @app.post("/api/v1/weights_info")
    @app.post("/weights_info")
    def weights_info(body: WeightsInfoRequest):
        info = app.state.weights_info.get(body.mint_path)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{body.mint_path}' not found")
        return WeightsInfoResponse(**info)

    @app.post("/api/v1/retrieve_future")
    @app.post("/retrieve_future")
    def retrieve_future(body: FutureRetrieveRequest):
        try:
            return future_store.retrieve(body.request_id, allow_metadata_only=body.allow_metadata_only)
        except PendingFutureError as exc:
            payload = {"queue_state": exc.queue_state}
            if exc.queue_state_reason is not None:
                payload["queue_state_reason"] = exc.queue_state_reason
            return JSONResponse(status_code=408, content=payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Future '{body.request_id}' not found") from exc

    @app.post("/api/v1/get_info")
    @app.post("/get_info")
    def get_info(body: GetInfoRequest):
        info = service.training.get_info(body.model_id)
        model = info["model"]
        metadata = dict(model.metadata)
        base_model = metadata.get("base_model")
        lora_config = metadata.get("lora_config") or {}
        model_data = ModelData(
            arch="transformer",
            model_name=str(base_model or model.backend_id),
            tokenizer_id=str(base_model or model.backend_id),
        )
        lora_rank = lora_config.get("rank") if hasattr(lora_config, "get") else None
        return GetInfoResponse(
            model_id=model.session_id,
            model_data=model_data,
            model_name=str(base_model or model.backend_id),
            is_lora=bool(lora_config),
            lora_rank=lora_rank,
        )

    @app.get("/api/v1/models")
    @app.get("/models")
    def list_models():
        models = []
        for view in service.training.list_models():
            metadata = dict(view.metadata)
            models.append(
                ModelSummary(
                    model_id=view.session_id,
                    session_id=metadata.get("session_id", view.session_id.split(":", 1)[0]),
                    model_seq_id=_parse_model_seq_id(view.session_id),
                    base_model=metadata.get("base_model"),
                    created_at=metadata.get("created_at"),
                    current_step=view.state.step if view.state else None,
                    is_active=view.status == "active",
                )
            )
        return ModelsListResponse(models=models)

    @app.get("/api/v1/models/{model_id}")
    @app.get("/models/{model_id}")
    def get_model(model_id: str):
        view = service.training.get_model(model_id)
        metadata = dict(view.metadata)
        return GetModelResponse(
            model_id=view.session_id,
            session_id=metadata.get("session_id", view.session_id.split(":", 1)[0]),
            model_seq_id=_parse_model_seq_id(view.session_id),
            base_model=metadata.get("base_model"),
            created_at=metadata.get("created_at"),
            current_step=view.state.step if view.state else None,
            is_active=view.status == "active",
            backend_id=view.backend_id,
            batch_codec=view.batch_codec,
            status=view.status,
            last_error=view.last_error,
        )

    @app.get("/api/v1/models/{model_id}/capabilities")
    @app.get("/models/{model_id}/capabilities")
    def get_capabilities(model_id: str):
        caps = service.training.get_capabilities(model_id)
        return CapabilitiesResponse(
            supports_forward=caps.supports_forward,
            supports_train_step=caps.supports_train_step,
            supports_reverse_kl=caps.supports_reverse_kl,
            supports_tokenizer_info=caps.supports_tokenizer_info,
            supports_reset_expert_bias=caps.supports_reset_expert_bias,
            supports_checkpoint_load=caps.supports_checkpoint_load,
            supports_checkpoint_save=caps.supports_checkpoint_save,
            extras=dict(caps.extras),
        )

    @app.get("/api/v1/models/{model_id}/tokenizer")
    @app.get("/models/{model_id}/tokenizer")
    def get_tokenizer(model_id: str):
        tok = service.training.get_tokenizer_info(model_id)
        return TokenizerInfoResponse(metadata=dict(tok.metadata))

    @app.delete("/api/v1/models/{model_id}")
    @app.delete("/models/{model_id}")
    def delete_model(model_id: str):
        service.training.delete_model(model_id)
        return {"ok": True}

    @app.post("/api/v1/reset_expert_bias")
    @app.post("/reset_expert_bias")
    def reset_expert_bias(body: ResetExpertBiasRequest):
        result = service.training.reset_expert_bias(body.model_id)
        modules_reset = 1 if result.outputs.get("reset") else 0
        status = "success" if modules_reset else "not_applicable"
        return ResetExpertBiasResponse(
            model_id=body.model_id,
            modules_reset=modules_reset,
            status=status,
        )

    @app.post("/api/v1/save_weights_for_sampler")
    def save_weights_for_sampler_v1(body: SaveWeightsForSamplerRequest):
        seq = body.sampling_session_seq_id if body.sampling_session_seq_id is not None else len(app.state.sampling_sessions) + 1
        path = _sampler_checkpoint_uri(
            body.model_id,
            body.path,
            versioned=True,
            sampling_session_seq_id=seq,
        )
        if body.path is not None:
            artifact = service.training.save_weights(
                model_id=body.model_id,
                uri=path,
                include_optimizer=False,
                metadata={"seq_id": body.seq_id, "type": body.type, "sampler": True},
            )
            _record_checkpoint(body.model_id, path=artifact.uri, checkpoint_type="sampler")
            payload = {"path": artifact.uri, "type": "save_weights_for_sampler"}
        else:
            artifact = service.training.save_weights(
                model_id=body.model_id,
                uri=path,
                include_optimizer=False,
                metadata={"seq_id": body.seq_id, "type": body.type, "sampler": True},
            )
            _record_checkpoint(body.model_id, path=artifact.uri, checkpoint_type="sampler")
            view = service.training.get_model(body.model_id)
            metadata = dict(view.metadata)
            session_id = str(metadata.get("session_id") or body.model_id.split(":", 1)[0])
            sampling_session_id = _register_sampling_session(
                session_id=session_id,
                seq_id=seq,
                base_model=metadata.get("base_model"),
                model_path=artifact.uri,
                inference_backend_id=_select_inference_backend_for_base_model(metadata.get("base_model")),
                lora_rank=(metadata.get("lora_config") or {}).get("rank", 32) if hasattr(metadata.get("lora_config") or {}, "get") else 32,
            )
            payload = {
                "path": None,
                "sampling_session_id": sampling_session_id,
                "type": "save_weights_for_sampler",
            }
        return _future(payload)

    @app.post("/save_weights_for_sampler")
    def save_weights_for_sampler(body: SaveWeightsForSamplerRequest):
        path = _sampler_checkpoint_uri(body.model_id, body.path, versioned=False, sampling_session_seq_id=body.sampling_session_seq_id)
        artifact = service.training.save_weights(
            model_id=body.model_id,
            uri=path,
            include_optimizer=False,
            metadata={"seq_id": body.seq_id, "type": body.type, "sampler": True},
        )
        _record_checkpoint(body.model_id, path=artifact.uri, checkpoint_type="sampler")

        view = service.training.get_model(body.model_id)
        metadata = dict(view.metadata)
        session_id = str(metadata.get("session_id") or body.model_id.split(":", 1)[0])
        seq = body.sampling_session_seq_id if body.sampling_session_seq_id is not None else len(app.state.sampling_sessions) + 1
        try:
            inference_backend_id = _select_inference_backend_for_base_model(metadata.get("base_model"))
        except HTTPException:
            inference_backend_id = None
        sampling_session_id = _register_sampling_session(
            session_id=session_id,
            seq_id=seq,
            base_model=metadata.get("base_model"),
            model_path=artifact.uri,
            inference_backend_id=inference_backend_id,
            lora_rank=(metadata.get("lora_config") or {}).get("rank", 32) if hasattr(metadata.get("lora_config") or {}, "get") else 32,
        )

        response = SaveWeightsForSamplerResponse(path=artifact.uri, sampling_session_id=sampling_session_id)
        return _future(response.model_dump())

    @app.get("/api/v1/training_runs")
    @app.get("/training_runs")
    def list_training_runs(request: Request):
        versioned = request.url.path.startswith("/api/v1/")
        offset = int(request.query_params.get("offset", 0))
        limit = int(request.query_params.get("limit", 20))
        all_runs = [
            _training_run_payload(view, versioned=versioned)
            for view in service.training.list_models()
        ]
        runs = all_runs[offset : offset + limit]
        if versioned:
            return {
                "training_runs": runs,
                "cursor": {"offset": offset, "limit": limit, "total_count": len(all_runs)},
            }
        return TrainingRunsResponse(training_runs=[TrainingRun(**run) for run in runs])

    @app.get("/api/v1/training_runs/{training_run_id}")
    @app.get("/training_runs/{training_run_id}")
    def get_training_run(training_run_id: str, request: Request):
        view = service.training.get_model(training_run_id)
        payload = _training_run_payload(view, versioned=request.url.path.startswith("/api/v1/"))
        if request.url.path.startswith("/api/v1/"):
            return payload
        return TrainingRun(**payload)

    @app.get("/api/v1/training_runs/{model_id}/checkpoints")
    @app.get("/training_runs/{model_id}/checkpoints")
    def list_checkpoints(model_id: str, request: Request):
        versioned = request.url.path.startswith("/api/v1/")
        offset = int(request.query_params.get("offset", 0))
        limit = int(request.query_params.get("limit", 100))
        all_entries = [_checkpoint_payload(model_id, entry, versioned=versioned) for entry in _checkpoint_entry(model_id)]
        entries = all_entries[offset : offset + limit]
        if versioned:
            return {
                "checkpoints": entries,
                "cursor": {"offset": offset, "limit": limit, "total_count": len(all_entries)},
            }
        return CheckpointsListResponse(model_id=model_id, checkpoints=[CheckpointInfo(**entry) for entry in entries])

    @app.get("/api/v1/checkpoints")
    @app.get("/checkpoints")
    def list_user_checkpoints(request: Request):
        offset = int(request.query_params.get("offset", 0))
        limit = int(request.query_params.get("limit", 100))
        all_entries = []
        for model_id, model_entries in app.state.checkpoints.items():
            all_entries.extend(_checkpoint_payload(model_id, entry, versioned=True) for entry in model_entries)
        entries = all_entries[offset : offset + limit]
        return {
            "checkpoints": entries,
            "cursor": {"offset": offset, "limit": limit, "total_count": len(all_entries)},
        }

    @app.get("/_artifacts/{model_id}/checkpoints/{checkpoint_id}", name="download_checkpoint_archive")
    def download_checkpoint_archive(model_id: str, checkpoint_id: str):
        entry = _find_checkpoint(model_id, checkpoint_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
        try:
            artifact_path = service.storage.resolve_for_read(str(entry["mint_path"]))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' artifact not found") from exc
        return FileResponse(artifact_path, media_type="application/gzip", filename=f"{checkpoint_id}.tar.gz")

    @app.get("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive")
    def archive_checkpoint_v1(model_id: str, checkpoint_id: str, request: Request):
        entry = _find_checkpoint(model_id, checkpoint_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        location = str(request.url_for("download_checkpoint_archive", model_id=model_id, checkpoint_id=checkpoint_id))
        return RedirectResponse(
            url=location,
            status_code=302,
            headers={"Expires": format_datetime(expires, usegmt=True)},
        )

    @app.get("/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive")
    def archive_checkpoint(model_id: str, checkpoint_id: str):
        entry = _find_checkpoint(model_id, checkpoint_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
        return CheckpointArchiveResponse(checkpoint_id=checkpoint_id, mint_path=str(entry["mint_path"]))

    @app.get("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
    @app.get("/training_runs/{model_id}/checkpoints/{checkpoint_id}")
    def get_checkpoint(model_id: str, checkpoint_id: str, request: Request):
        entry = _find_checkpoint(model_id, checkpoint_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
        payload = _checkpoint_payload(model_id, entry, versioned=request.url.path.startswith("/api/v1/"))
        if request.url.path.startswith("/api/v1/"):
            return payload
        return CheckpointInfo(**payload)

    @app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
    @app.delete("/training_runs/{model_id}/checkpoints/{checkpoint_id}")
    def delete_checkpoint(model_id: str, checkpoint_id: str, request: Request):
        removed = _delete_checkpoint_entry(model_id, checkpoint_id)
        if removed is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
        if request.url.path.startswith("/api/v1/"):
            return Response(status_code=204)
        return {"ok": True}

    @app.post("/api/v1/checkpoints/upload")
    @app.post("/checkpoints/upload")
    def upload_checkpoint():
        checkpoint_id = "uploaded-checkpoint"
        path = f"mint://uploads/{checkpoint_id}.tar.gz"
        return CheckpointUploadResponse(checkpoint_id=checkpoint_id, mint_path=path)

    @app.post("/vla/train_step")
    def vla_train_step(body: TrainStepRequest):
        payload = body.forward_backward_input.model_dump()
        result = service.training.train_step(
            body.model_id,
            batch_payload=payload,
            options={
                "adam_params": body.adam_params.model_dump() if body.adam_params else None,
                "seq_id": body.seq_id,
            },
        )
        fb_output = _fb_output(result, loss_fn=body.forward_backward_input.loss_fn, forward_backward_input=payload)
        response = TrainStepResponse(output=fb_output, metrics=dict(fb_output.metrics))
        return _future(response.model_dump())

    @app.post("/api/v1/rollout_sessions")
    @app.post("/rollout_sessions")
    def open_rollout_session(body: RolloutSessionRequest):
        return service.rollouts.open_session(**body.model_dump())

    @app.post("/api/v1/rollout_sessions/{rollout_session_id}/collect")
    @app.post("/rollout_sessions/{rollout_session_id}/collect")
    def collect_experience(rollout_session_id: str, body: CollectExperienceRequest):
        batch = service.rollouts.collect_experience(
            rollout_session_id,
            prompt=_prompt_text(body.prompt),
            batch_codec=body.batch_codec,
            sampling=body.sampling.model_dump() if body.sampling else {},
            reward=body.reward,
            policy_version=body.policy_version,
            metadata=body.metadata,
            num_samples=body.num_samples,
        )
        return {
            "rollout_session_id": batch.rollout_session_id,
            "training_session_id": batch.training_session_id,
            "batch_codec": batch.batch_codec,
            "batch_payload": _json_safe(batch.batch_payload),
            "policy_backend_id": batch.policy_backend_id,
            "policy_version": batch.policy_version,
            "metadata": _json_safe(batch.metadata),
        }

    @app.post("/api/v1/rollout_sessions/{rollout_session_id}/train")
    @app.post("/rollout_sessions/{rollout_session_id}/train")
    def train_on_experience(rollout_session_id: str, body: TrainOnExperienceRequest):
        result = service.rollouts.train_on_experience(
            rollout_session_id,
            batch_payload=body.batch_payload,
            extension_op=body.extension_op,
            options=body.options,
        )
        return {
            "state": {
                "step": result.state.step,
                "extras": _json_safe(result.state.extras),
            },
            "outputs": _json_safe(result.outputs),
        }

    @app.delete("/api/v1/rollout_sessions/{rollout_session_id}")
    @app.delete("/rollout_sessions/{rollout_session_id}")
    def close_rollout_session(rollout_session_id: str):
        service.rollouts.close_session(rollout_session_id)
        return {"ok": True}

    return app


app = create_app()

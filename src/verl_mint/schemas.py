from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MintBaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class EncodedTextChunk(MintBaseSchema):
    tokens: list[int]
    type: Literal["encoded_text"] = "encoded_text"


class ImageAssetPointerChunk(MintBaseSchema):
    format: Literal["png", "jpeg"]
    location: str
    expected_tokens: int | None = None
    type: Literal["image_asset_pointer"] = "image_asset_pointer"


class ImageChunk(MintBaseSchema):
    data: str
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = None
    type: Literal["image"] = "image"


# The formal client omits default `type` fields when dumping chunks, so parsing
# must accept structural unions instead of requiring an explicit discriminator.
ModelInputChunk = EncodedTextChunk | ImageAssetPointerChunk | ImageChunk


class ModelInput(MintBaseSchema):
    chunks: list[ModelInputChunk]


class TensorData(MintBaseSchema):
    data: list[int] | list[float] | float
    shape: list[int] | None = None
    dtype: str = "float32"


class SamplingParams(MintBaseSchema):
    max_tokens: int = 512
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    stop: list[str] | list[int] | str | None = None
    seed: int | None = None


class LoRAConfig(MintBaseSchema):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True


class RolloutCorrectionConfig(MintBaseSchema):
    rollout_is: Literal["token", "sequence"] | None = None
    rollout_is_threshold: str | float | None = None
    rollout_is_batch_normalize: bool | None = None
    rollout_rs: str | None = None
    rollout_rs_threshold: str | float | None = None
    bypass_mode: bool | None = None
    loss_type: Literal["ppo_clip", "reinforce"] | None = None


class AdamParams(MintBaseSchema):
    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class Datum(MintBaseSchema):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_input: ModelInput
    loss_fn_inputs: dict[str, Any]


class ForwardBackwardInput(MintBaseSchema):
    data: list[Datum]
    loss_fn: str
    loss_fn_config: dict[str, Any] | None = None


class PPODatum(MintBaseSchema):
    student_input: ModelInput
    target_tokens: TensorData
    weights: TensorData
    prompt_tokens: list[int] | None = None
    completion_tokens: list[int] | None = None
    old_logprobs: TensorData | None = None
    old_values: TensorData | None = None
    advantages: TensorData | None = None
    returns: TensorData | None = None
    reward: float | None = None
    group_id: str | None = None
    sample_id: str | None = None


class ReverseKLDatum(MintBaseSchema):
    student_input: ModelInput
    reference_input: ModelInput
    target_tokens: TensorData
    weights: TensorData
    prompt_tokens: list[int] | None = None
    completion_tokens: list[int] | None = None
    old_logprobs: TensorData | None = None
    reference_logprobs: TensorData | None = None
    old_values: TensorData | None = None
    advantages: TensorData | None = None
    returns: TensorData | None = None
    reward: float | None = None
    group_id: str | None = None
    sample_id: str | None = None


class VLAObservation(MintBaseSchema):
    model_input: ModelInput
    state: TensorData


class VLADatum(MintBaseSchema):
    observation: VLAObservation
    supervision: dict[str, TensorData]


class UntypedAPIFuture(MintBaseSchema):
    request_id: str


class FutureRetrieveRequest(MintBaseSchema):
    request_id: str
    model_id: str | None = None
    allow_metadata_only: bool = False


class WeightsInfoRequest(MintBaseSchema):
    mint_path: str


class CreateModelRequest(MintBaseSchema):
    session_id: str
    model_seq_id: int
    base_model: str
    backend_id: str | None = None
    batch_codec: str = "torch"
    user_metadata: dict[str, Any] | None = None
    lora_config: LoRAConfig | None = None
    rollout_correction_config: RolloutCorrectionConfig | None = None
    type: Literal["create_model"] = "create_model"


class CreateModelResponse(MintBaseSchema):
    request_id: str
    model_id: str
    type: Literal["create_model"] = "create_model"
    backend: str | None = None


class CreateModelFromStateRequest(MintBaseSchema):
    session_id: str
    model_seq_id: int
    base_model: str
    backend_id: str | None = None
    batch_codec: str = "torch"
    state_path: str
    lora_config: LoRAConfig | None = None
    rollout_correction_config: RolloutCorrectionConfig | None = None
    load_optimizer: bool = True
    user_metadata: dict[str, Any] | None = None
    type: Literal["create_model_from_state"] = "create_model_from_state"


class CreateModelFromStateResponse(MintBaseSchema):
    request_id: str
    model_id: str
    type: Literal["create_model_from_state"] = "create_model_from_state"


class LossFnOutput(MintBaseSchema):
    loss: TensorData | None = None
    logprobs: TensorData | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForwardBackwardOutput(MintBaseSchema):
    loss_fn_output_type: str = "loss"
    loss_fn_outputs: list[LossFnOutput]
    metrics: dict[str, float] = Field(default_factory=dict)


class ForwardBackwardRequest(MintBaseSchema):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardRequest(MintBaseSchema):
    forward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardResponse(MintBaseSchema):
    output: ForwardBackwardOutput
    type: Literal["forward"] = "forward"


class ForwardBackwardResponse(MintBaseSchema):
    output: ForwardBackwardOutput
    type: Literal["forward_backward"] = "forward_backward"


class OptimStepRequest(MintBaseSchema):
    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None
    type: Literal["optim_step"] = "optim_step"


class OptimStepResponse(MintBaseSchema):
    metrics: dict[str, float] | None = None
    type: Literal["optim_step"] = "optim_step"


class TrainStepRequest(MintBaseSchema):
    forward_backward_input: ForwardBackwardInput
    adam_params: AdamParams | None = None
    model_id: str
    seq_id: int | None = None
    type: Literal["train_step"] = "train_step"


class TrainStepResponse(MintBaseSchema):
    output: ForwardBackwardOutput
    metrics: dict[str, float] | None = None
    type: Literal["train_step"] = "train_step"


class ResetExpertBiasRequest(MintBaseSchema):
    model_id: str


class ResetExpertBiasResponse(MintBaseSchema):
    model_id: str
    modules_reset: int = 0
    status: Literal["success", "not_applicable"] = "success"


class ForwardBackwardPPORequest(MintBaseSchema):
    model_id: str
    data: list[PPODatum]
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    value_clip: float = 0.2
    seq_id: int | None = None
    type: Literal["mint_forward_backward_ppo"] = "mint_forward_backward_ppo"


class PPOItemOutput(MintBaseSchema):
    loss: TensorData


class ForwardBackwardPPOResponse(MintBaseSchema):
    outputs: list[PPOItemOutput]
    metrics: dict[str, float] = Field(default_factory=dict)
    type: Literal["mint_forward_backward_ppo"] = "mint_forward_backward_ppo"


class ForwardBackwardReverseKLRequest(MintBaseSchema):
    model_id: str
    reference_model_path: str
    data: list[ReverseKLDatum]
    temperature: float = 1.0
    seq_id: int | None = None
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


class ReverseKLItemOutput(MintBaseSchema):
    loss: TensorData


class ForwardBackwardReverseKLResponse(MintBaseSchema):
    outputs: list[ReverseKLItemOutput]
    metrics: dict[str, float] = Field(default_factory=dict)
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


class SaveStateRequest(MintBaseSchema):
    model_id: str
    path: str | None = None
    ttl_seconds: int | None = None
    seq_id: int | None = None
    type: Literal["save_weights"] = "save_weights"


class SaveStateResponse(MintBaseSchema):
    path: str
    type: Literal["save_weights"] = "save_weights"


class LoadStateRequest(MintBaseSchema):
    model_id: str
    path: str
    optimizer: bool = True
    seq_id: int | None = None
    type: Literal["load_weights"] = "load_weights"


class LoadStateResponse(MintBaseSchema):
    path: str
    type: Literal["load_weights"] = "load_weights"


class GetInfoRequest(MintBaseSchema):
    model_id: str
    type: Literal["get_info"] = "get_info"


class ModelData(MintBaseSchema):
    arch: str | None = None
    model_name: str | None = None
    tokenizer_id: str | None = None


class GetInfoResponse(MintBaseSchema):
    model_id: str
    model_data: ModelData
    model_name: str | None = None
    is_lora: bool | None = None
    lora_rank: int | None = None
    type: Literal["get_info"] = "get_info"


class ModelSummary(MintBaseSchema):
    model_id: str
    session_id: str
    model_seq_id: int | None = None
    base_model: str | None = None
    created_at: str | None = None
    current_step: int | None = None
    is_active: bool = True


class ModelsListResponse(MintBaseSchema):
    models: list[ModelSummary]


class GetModelResponse(ModelSummary):
    backend_id: str | None = None
    batch_codec: str | None = None
    status: str = "active"
    last_error: str | None = None


class CapabilitiesResponse(MintBaseSchema):
    supports_forward: bool = False
    supports_train_step: bool = True
    supports_reverse_kl: bool = False
    supports_tokenizer_info: bool = True
    supports_reset_expert_bias: bool = False
    supports_checkpoint_load: bool = True
    supports_checkpoint_save: bool = True
    extras: dict[str, Any] = Field(default_factory=dict)


class TokenizerInfoResponse(MintBaseSchema):
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutSessionRequest(MintBaseSchema):
    rollout_session_id: str
    inference_backend_id: str
    training_session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectExperienceRequest(MintBaseSchema):
    prompt: str | ModelInput
    batch_codec: str
    sampling: SamplingParams | None = None
    reward: float | None = None
    policy_version: str | None = None
    num_samples: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainOnExperienceRequest(MintBaseSchema):
    batch_payload: Any = Field(default_factory=dict)
    extension_op: str = "rl"
    options: dict[str, Any] = Field(default_factory=dict)


class SampledSequence(MintBaseSchema):
    tokens: list[int]
    logprobs: list[float] | None = None
    routed_experts: list[Any] | None = None
    stop_reason: Literal["length", "stop", "eos"] = "length"


class SampleRequest(MintBaseSchema):
    sampling_session_id: str | None = None
    model_id: str | None = None
    base_model: str | None = None
    model_path: str | None = None
    seq_id: int | None = None
    num_samples: int
    prompt: ModelInput
    sampling_params: SamplingParams
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0
    include_prompt_logprobs: bool = False


class SampleResponse(MintBaseSchema):
    sequences: list[SampledSequence]
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None
    type: Literal["sample"] = "sample"


class ComputeLogprobsRequest(MintBaseSchema):
    sampling_session_id: str
    seq_id: int
    sequence: ModelInput


class ComputeLogprobsResponse(MintBaseSchema):
    logprobs: list[float | None]
    type: Literal["compute_logprobs"] = "compute_logprobs"


class CreateSessionRequest(MintBaseSchema):
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    sdk_version: str = ""
    type: Literal["create_session"] = "create_session"


class CreateSessionResponse(MintBaseSchema):
    session_id: str
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None
    type: Literal["create_session"] = "create_session"


class CreateSamplingSessionRequest(MintBaseSchema):
    session_id: str
    sampling_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None
    lora_rank: int = 32


class CreateSamplingSessionResponse(MintBaseSchema):
    sampling_session_id: str


class GetSessionResponse(MintBaseSchema):
    training_run_ids: list[str]
    sampler_ids: list[str]


class ListSessionsResponse(MintBaseSchema):
    sessions: list[str]


class GetSamplerResponse(MintBaseSchema):
    sampler_id: str
    base_model: str
    model_path: str | None = None


class SessionHeartbeatRequest(MintBaseSchema):
    session_id: str
    type: Literal["session_heartbeat"] = "session_heartbeat"


class SessionHeartbeatResponse(MintBaseSchema):
    type: Literal["session_heartbeat"] = "session_heartbeat"


class TelemetryRequest(MintBaseSchema):
    events: list[dict[str, Any]] = Field(default_factory=list)
    platform: str = ""
    sdk_version: str = ""
    session_id: str = ""


class TelemetryResponse(MintBaseSchema):
    status: Literal["accepted"] = "accepted"


class SaveWeightsForSamplerRequest(MintBaseSchema):
    model_id: str
    path: str | None = None
    ttl_seconds: int | None = None
    seq_id: int | None = None
    sampling_session_seq_id: int | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class SaveWeightsForSamplerResponse(MintBaseSchema):
    path: str | None = None
    sampling_session_id: str | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class Cursor(MintBaseSchema):
    offset: int
    limit: int
    total_count: int


class TrainingRun(MintBaseSchema):
    training_run_id: str
    base_model: str
    model_owner: str = "local"
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: int | None = None
    last_request_time: str = "1970-01-01T00:00:00Z"
    last_activity: float | None = None
    idle_for_s: float | None = None
    last_checkpoint: Any | None = None
    last_sampler_checkpoint: Any | None = None
    user_metadata: dict[str, str] | None = None


class TrainingRunsResponse(MintBaseSchema):
    training_runs: list[TrainingRun]
    cursor: Cursor | None = None


class CheckpointInfo(MintBaseSchema):
    checkpoint_id: str
    checkpoint_type: Literal["training", "sampler"]
    time: str
    mint_path: str
    created_at: str | None = None
    size_bytes: int | None = None
    public: bool = False
    expires_at: str | None = None
    storage_tier: str | None = None
    mirror_status: str | None = None
    mirror_error: str | None = None


class WeightsInfoResponse(MintBaseSchema):
    base_model: str
    is_lora: bool
    lora_rank: int | None = None
    train_unembed: bool | None = None
    train_mlp: bool | None = None
    train_attn: bool | None = None


class CheckpointsListResponse(MintBaseSchema):
    model_id: str | None = None
    checkpoints: list[CheckpointInfo]


class CheckpointArchiveResponse(MintBaseSchema):
    checkpoint_id: str
    mint_path: str


class CheckpointUploadResponse(MintBaseSchema):
    checkpoint_id: str
    mint_path: str


class CreateActionSessionRequest(MintBaseSchema):
    session_id: str
    action_session_seq_id: int | None = None
    base_model: str | None = None
    model_path: str | None = None


class CreateActionSessionResponse(MintBaseSchema):
    action_session_id: str


class ActRequest(MintBaseSchema):
    action_session_id: str
    seq_id: int | None = None
    observation: ModelInput
    extra_inputs: dict[str, TensorData] = Field(default_factory=dict)
    temperature: float | None = None


class ActResponse(MintBaseSchema):
    actions: TensorData
    policy_timing: dict[str, float] | None = None
    type: Literal["act"] = "act"


class InterpolateCheckpointsRequest(MintBaseSchema):
    source_paths: list[str]
    coefficients: list[float]
    output_path: str | None = None
    output_checkpoint_type: Literal["sampler"] = "sampler"
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"


class InterpolateCheckpointsResponse(MintBaseSchema):
    path: str
    checkpoint_type: Literal["sampler"] = "sampler"
    source_paths: list[str]
    coefficients: list[float]
    has_rank_shards: bool = False
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"

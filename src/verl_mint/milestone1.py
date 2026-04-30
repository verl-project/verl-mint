from __future__ import annotations

MILESTONE1_BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

# Optional override when local weights are not stored under the Hugging Face id.
MILESTONE1_MODEL_PATH_ENV = "VERL_MINT_QWEN_MODEL_PATH"

MILESTONE1_RUNTIME_NOTES = {
    "python": ">=3.11",
    "cuda": "12.x (if GPU is used)",
    "torch": ">=2.3",
    "transformers": ">=4.44",
    "peft": ">=0.12",
    "accelerate": ">=0.33",
    "ray": "not required for milestone-1 local path",
}

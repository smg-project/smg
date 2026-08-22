"""Model specifications for E2E tests.

Each model spec defines:
- model: HuggingFace model path or local path
- tp: Tensor parallelism size (number of GPUs needed)
- features: List of features this model supports (for test filtering)
- sglang_args: Optional SGLang-specific CLI arguments
- vllm_args: Optional vLLM-specific CLI arguments
- trtllm_args: Optional TensorRT-LLM CLI arguments
"""

from __future__ import annotations

import json
import os

# Environment variable for local model paths (CI uses local copies for speed)
ROUTER_LOCAL_MODEL_PATH = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")
# Nightly benchmarks skip --enforce-eager for performance measurement
_is_nightly = os.environ.get("E2E_NIGHTLY") == "1"


def _resolve_model_path(hf_path: str) -> str:
    """Resolve model path, preferring local path if available."""
    if ROUTER_LOCAL_MODEL_PATH:
        local_path = os.path.join(ROUTER_LOCAL_MODEL_PATH, hf_path)
        if os.path.exists(local_path):
            return local_path
    return hf_path


MODEL_SPECS: dict[str, dict] = {
    # Primary chat model - used for most tests
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Small model for quick tests
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "tool_choice"],
    },
    # Function calling specialist
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-7B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
    },
    # Function calling specialist (larger, for Response API tests)
    "Qwen/Qwen2.5-14B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-14B-Instruct"),
        # 14B BF16 weights (~28GB) fit on one H100/80GB; tp=1 avoids paying
        # NCCL setup on every restart. Override via E2E_MODEL_TP_OVERRIDES.
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
        "sglang_args": ["--context-length=16384"],  # Faster startup, prevents memory issues
    },
    # Reasoning model
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model": _resolve_model_path("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "tp": 1,
        "features": ["chat", "streaming", "reasoning"],
    },
    # Qwen3 instruct (non-thinking variant) — emits the same
    # `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` format as
    # Qwen 2.5, so the gateway's ``qwen`` tool-call parser applies. Used by
    # ``TestToolChoiceQwen`` and ``TestMultiTurnToolCall``: a Qwen3 model is
    # required because the Qwen2 family is not in TokenSpeed's model registry.
    "Qwen/Qwen3-4B-Instruct-2507": {
        "model": _resolve_model_path("Qwen/Qwen3-4B-Instruct-2507"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "tool_choice"],
    },
    "Qwen/Qwen3-30B-A3B": {
        "model": _resolve_model_path("Qwen/Qwen3-30B-A3B"),
        "tp": 1,
        "features": ["chat", "streaming", "thinking", "reasoning"],
        "vllm_args": [] if _is_nightly else ["--enforce-eager"],
        "trtllm_extra_config": {"kv_cache_config": {"free_gpu_memory_fraction": 0.8}},
    },
    # Qwen3.6-27B — thinking model with XML tool calls (qwen3_5 arch). Staged for
    # the nightly BFCL A/B (scripts/bfcl); tp=2 fits the 27B on two GPUs.
    "Qwen/Qwen3.6-27B": {
        "model": _resolve_model_path("Qwen/Qwen3.6-27B"),
        "tp": 2,
        "features": [
            "chat",
            "streaming",
            "function_calling",
            "tool_choice",
            "thinking",
            "reasoning",
        ],
    },
    # Mistral for function calling
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model": _resolve_model_path("mistralai/Mistral-7B-Instruct-v0.3"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
        "sglang_args": ["--constrained-json-disable-any-whitespace"],
        "vllm_args": [
            "--structured-outputs-config",
            '{"disable_any_whitespace": true, "backend": "xgrammar"}',
        ],
    },
    # Embedding model (last-token pooling; upstream-tested under transformers v5)
    "Qwen/Qwen3-Embedding-0.6B": {
        "model": _resolve_model_path("Qwen/Qwen3-Embedding-0.6B"),
        "tp": 1,
        "features": ["embedding"],
    },
    # Realtime ASR (speech-to-text). Served with the realtime architecture so
    # vLLM exposes `ws /v1/realtime`; `TORCH_SDPA` avoids the bundled flash-attn
    # CUTE kernel that rejects some GPU archs (e.g. GB300 / sm_103).
    "Qwen/Qwen3-ASR-1.7B": {
        "model": _resolve_model_path("Qwen/Qwen3-ASR-1.7B"),
        "tp": 1,
        "features": ["realtime", "transcription", "audio"],
        "vllm_args": [
            "--hf-overrides",
            '{"architectures":["Qwen3ASRRealtimeGeneration"]}',
            "--mm-encoder-attn-backend",
            "TORCH_SDPA",
            "--enforce-eager",
        ],
    },
    # GPT-OSS models (Harmony)
    "openai/gpt-oss-20b": {
        "model": _resolve_model_path("openai/gpt-oss-20b"),
        # MXFP4-quantized MoE (~13GB weights) fits easily on one H100; tp=1
        # roughly halves worker startup vs tp=2. Override via E2E_MODEL_TP_OVERRIDES.
        "tp": 1,
        "features": ["chat", "streaming", "reasoning", "harmony"],
        "vllm_args": [
            "--structured-outputs-config",
            '{"enable_in_reasoning": true}',
        ],
    },
    "openai/gpt-oss-120b": {
        "model": _resolve_model_path("openai/gpt-oss-120b"),
        "tp": 4,
        "features": ["chat", "streaming", "reasoning", "harmony"],
        "startup_timeout": 600,
        "vllm_args": [
            "--structured-outputs-config",
            '{"enable_in_reasoning": true}',
        ],
    },
    # MiniMax M2 - nightly benchmarks
    "minimaxai/minimax-m2": {
        "model": _resolve_model_path("minimaxai/minimax-m2"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling", "reasoning"],
        "sglang_args": ["--trust-remote-code"],
        "vllm_args": ["--trust-remote-code"],
    },
    # Vision-language model for multimodal benchmarks (MMMU)
    "Qwen/Qwen3-VL-8B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen3-VL-8B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "multimodal"],
    },
    # TokenSpeed EPD multimodal model. Qwen3.5-9B is a vision-language model
    # (hybrid Gated DeltaNet + sparse MoE); BF16 ~18GB fits one 80GB H100 at
    # tp=1, so every EPD topology (1e1p1d/1e2p1d/2e1p1d/1e1p2d) runs on the
    # 4-GPU h100 runner, one worker per card. EPD is TokenSpeed-only: the encode
    # worker runs the vision tower; prefill/decode run the LM. FA3 is the H100
    # attention backend (trtllm fails on H100 for this model). Disaggregation
    # role flags (--disaggregation-mode etc.) are added per-role in worker.py.
    "Qwen/Qwen3.5-9B": {
        "model": _resolve_model_path("Qwen/Qwen3.5-9B"),
        "tp": 1,
        "features": ["chat", "streaming", "multimodal", "moe"],
        "startup_timeout": 600,
        "tokenspeed_args": [
            "--attention-backend",
            "fa3",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.8",
            # This model's hybrid-attention KV pool opts into tokenspeed's
            # paged-cache PD adjunct on prefill/decode roles, which rejects
            # layerwise transfer — yet layerwise defaults ON (interval=1), so
            # the engine dies at startup under its own defaults. Scoped here
            # rather than to every disaggregation worker: non-hybrid models
            # support layerwise and should keep exercising that default path.
            "--disaggregation-layerwise-interval",
            "0",
        ],
        # TokenSpeed-only (GDN + MoE arch won't load under sglang/vllm/trt), so
        # keep it out of the tier-wide pre-download; the EPD job fetches it by id.
        "skip_tier_download": True,
    },
    # Llama-4-Maverick (17B with 128 experts, FP8) - Nightly benchmarks
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "model": _resolve_model_path("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        "tp": 8,  # Tensor parallelism across 8 GPUs
        "features": ["chat", "streaming", "function_calling", "moe"],
        "sglang_args": [
            "--trust-remote-code",
            "--context-length=163840",  # 160K context length (SGLang)
            "--attention-backend=fa3",  # fa3 attention backend
            "--mem-fraction-static=0.82",  # 82% GPU memory for static allocation
        ],
        "vllm_args": [
            "--trust-remote-code",
            "--max-model-len=163840",  # 160K context length (vLLM)
            "--attention-backend=FLASHINFER",  # FLASHINFER attention backend
        ],
        "startup_timeout": 1200,  # Large MoE model may need extra download/load time
    },
    # Llama-4-Scout (17B with 16 experts) - Nightly benchmarks and Multimodal tests
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling", "multimodal", "moe"],
        "sglang_args": [
            "--context-length=196608",
            "--attention-backend=fa3",
            "--cuda-graph-max-bs=256",
            "--max-running-requests=300",
            "--mem-fraction-static=0.85",
            "--enable-multimodal",
        ],
        "vllm_args": [
            "--max-model-len=196608",
        ],
        "startup_timeout": 1200,  # Large MoE model may need extra download/load time
    },
    # Llama-3.3-70B - Nightly benchmarks
    "meta-llama/Llama-3.3-70B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.3-70B-Instruct"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling"],
        "sglang_args": [
            "--mem-fraction-static=0.9",
        ],
        "vllm_args": [
            "--max-model-len=131072",
            "--gpu-memory-utilization=0.9",
            "--enable-chunked-prefill",
        ],
    },
    # Llama-3.3-70B FP8 - Nightly benchmarks
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": {
        "model": _resolve_model_path("RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling"],
        "sglang_args": [
            "--mem-fraction-static=0.9",
        ],
        "vllm_args": [
            "--max-model-len=131072",
            "--gpu-memory-utilization=0.9",
            "--enable-chunked-prefill",
        ],
    },
    # MLX (Apple Silicon). Smallest Qwen3 with tool calling + thinking (~400 MB).
    "mlx-community/Qwen3-0.6B-4bit": {
        "model": _resolve_model_path("mlx-community/Qwen3-0.6B-4bit"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "reasoning", "thinking"],
    },
}


def get_models_with_feature(feature: str) -> list[str]:
    """Get list of model IDs that support a specific feature."""
    return [
        model_id for model_id, spec in MODEL_SPECS.items() if feature in spec.get("features", [])
    ]


def _parse_tp_overrides() -> dict | None:
    """Parse E2E_MODEL_TP_OVERRIDES env var once at import time."""
    raw = os.environ.get("E2E_MODEL_TP_OVERRIDES")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


_TP_OVERRIDES = _parse_tp_overrides()


def get_model_spec(model_id: str) -> dict:
    """Get spec for a specific model, raising KeyError if not found."""
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_SPECS.keys())}")
    spec = dict(MODEL_SPECS[model_id])
    if _TP_OVERRIDES is not None:
        override = _TP_OVERRIDES.get(model_id)
        if isinstance(override, int) and override > 0:
            spec["tp"] = override
    return spec


# Convenience groupings for test parametrization
CHAT_MODELS = get_models_with_feature("chat")
EMBEDDING_MODELS = get_models_with_feature("embedding")
REASONING_MODELS = get_models_with_feature("reasoning")
FUNCTION_CALLING_MODELS = get_models_with_feature("function_calling")


# =============================================================================
# Default model path constants (for backward compatibility with existing tests)
# =============================================================================

DEFAULT_MODEL_PATH = MODEL_SPECS["meta-llama/Llama-3.1-8B-Instruct"]["model"]
DEFAULT_SMALL_MODEL_PATH = MODEL_SPECS["meta-llama/Llama-3.2-1B-Instruct"]["model"]
DEFAULT_REASONING_MODEL_PATH = MODEL_SPECS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]["model"]
DEFAULT_ENABLE_THINKING_MODEL_PATH = MODEL_SPECS["Qwen/Qwen3-30B-A3B"]["model"]
DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["Qwen/Qwen2.5-7B-Instruct"]["model"]
DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["mistralai/Mistral-7B-Instruct-v0.3"][
    "model"
]
DEFAULT_GPT_OSS_MODEL_PATH = MODEL_SPECS["openai/gpt-oss-20b"]["model"]
DEFAULT_EMBEDDING_MODEL_PATH = MODEL_SPECS["Qwen/Qwen3-Embedding-0.6B"]["model"]


# =============================================================================
# Third-party model configurations (cloud APIs)
# =============================================================================

THIRD_PARTY_MODELS: dict[str, dict] = {
    "openai": {
        "description": "OpenAI API",
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API",
        "model": "grok-4.20-0309-non-reasoning",
        "api_key_env": "XAI_API_KEY",
    },
    "anthropic": {
        "description": "Anthropic API",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
}

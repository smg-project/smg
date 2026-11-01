"""Defaults scoped to the explicitly selected SGLang Rust media worker.

Resolve the effective engine dtype, rather than guessing from quantized weight
storage or a model name. Explicit overrides retain their original semantics.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)


def encoder_dtype(override: str | None, model_dtype: Callable[[], Any]) -> str:
    """Resolve the effective engine dtype unless an explicit supported override is set."""
    supported = {"float32", "bfloat16", "float16"}
    if override is not None:
        value = override.strip().lower()
        if value not in supported:
            raise ValueError("SMG_SGLANG_MM_DTYPE must be float32, bfloat16 or float16")
        return value
    # ModelConfig applies the model's dtype and any explicit engine --dtype.
    # Configuration/load errors must propagate, not silently change precision.
    value = str(model_dtype()).removeprefix("torch.").lower()
    if value in supported:
        return value
    logger.warning("Unsupported model media dtype %s; retaining float32 transport", value)
    return "float32"


def configure_threads(environ: MutableMapping[str, str] | None = None) -> None:
    """Set worker-local defaults before pipeline creation, preserving explicit settings."""
    # Set before creating the native pipeline/runtime or spawning scheduler
    # ranks. Do not change shared defaults for Router, Python or vLLM processes.
    environ = os.environ if environ is None else environ
    for name, value in {
        "SMG_SERVICER_WORKER_THREADS": "8",
        "SMG_MM_THREADS": "32",
        "SMG_MM_IMAGE_CONCURRENCY": "8",
    }.items():
        environ.setdefault(name, value)

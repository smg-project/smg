"""Launch inputs for the explicit AMD GPT-OSS 20B compatibility profile.

This keeps the original packed checkpoint and selects its checked BF16/Triton
fallback. The caller must separately install/verify the pinned SDK patch and
run its numeric preflight; these launch inputs establish neither numerical
parity nor performance. Other models, vendors and the default stay unchanged.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PROFILE_ENV = "E2E_SGLANG_GPTOSS_MXFP4_BF16"


def enabled(engine: str, model_id: str) -> bool:
    """Only the original 20B model in an explicitly opted-in AMD job."""
    if engine != "sglang" or model_id != MODEL:
        return False
    # Match the harness's vendor normalization and SMI fallback without importing
    # it: the numeric preflight loads this stdlib-only module from its file.
    vendor = os.environ.get("E2E_GPU_VENDOR", "").strip().lower()
    if not vendor:
        vendor = (
            "amd"
            if shutil.which("nvidia-smi") is None and shutil.which("amd-smi") is not None
            else "nvidia"
        )
    if vendor != "amd":
        return False
    value = os.environ.get(PROFILE_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{PROFILE_ENV} must be 0 or 1")
    return value == "1"


def prepare_launch(
    engine: str,
    model_id: str,
    model_path: str,
    spec: dict,
    extra_args: list[str] | None,
    extra_env: dict[str, str] | None = None,
) -> tuple[str, dict]:
    """Bind the selected profile's checkpoint, served id and fixed MoE flags."""
    if not enabled(engine, model_id):
        return model_path, spec
    configured = list(spec.get("sglang_args", [])) + list(extra_args or [])
    owned_flags = {
        "--moe-runner-backend",
        "--moe-a2a-backend",
        "--model-path",
        "--model",
        "--revision",
    }
    if any(
        (flag := arg.split("=", 1)[0]).startswith("--")
        and any(owned.startswith(flag) for owned in owned_flags)
        for arg in configured
    ):
        raise ValueError(
            "The fixed GPT-OSS fallback owns its checkpoint and generic Triton/none MoE flags"
        )
    named = False
    for index, arg in enumerate(configured):
        flag, equals, value = arg.partition("=")
        if (
            flag.startswith("--")
            and flag != "--served-model-name"
            and "--served-model-name".startswith(flag)
        ):
            raise ValueError("The fixed GPT-OSS fallback requires the full served model name flag")
        if flag == "--served-model-name":
            if not equals:
                value = configured[index + 1] if index + 1 < len(configured) else ""
            if value != MODEL:
                raise ValueError("The fixed GPT-OSS fallback owns its served model name")
            named = True
    # Worker applies extra_env over its inherited environment before launch.
    # Resolve the checkpoint against that same effective cache, with the Hub
    # cache taking precedence over HF_HOME in both environments.
    effective_env = {**os.environ, **(extra_env or {})}
    cache = Path(
        effective_env.get("HF_HUB_CACHE") or Path(effective_env.get("HF_HOME") or "/models") / "hub"
    )
    snapshot_path = cache / "models--openai--gpt-oss-20b" / "snapshots" / REVISION
    snapshot = str(snapshot_path)
    # An arbitrary local mapping does not establish the numeric probe's
    # checkpoint identity. This opt-in accepts only its fixed original path.
    if model_path not in (MODEL, snapshot):
        raise ValueError("The fixed GPT-OSS fallback requires its original checkpoint revision")
    if not (snapshot_path / "config.json").is_file():
        raise FileNotFoundError(
            f"The fixed GPT-OSS fallback requires {MODEL}@{REVISION}: "
            f"missing {snapshot_path / 'config.json'}. "
            "Download this exact revision into the worker's effective HF cache before launch."
        )
    model_path = snapshot
    args = list(spec.get("sglang_args", []))
    if not named:
        args.extend(["--served-model-name", MODEL])
    args.extend(["--moe-runner-backend", "triton", "--moe-a2a-backend", "none"])
    return model_path, {**spec, "sglang_args": args}


def apply_environment(
    engine: str, model_id: str, env: dict[str, str], extra_env: dict[str, str] | None
) -> None:
    """Keep the checked worker's environment paired with its launch flags."""
    if not enabled(engine, model_id):
        return
    required = {"SGLANG_USE_AITER": "0", "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK": "1"}
    if extra_env and any(
        key in extra_env and extra_env[key] != value for key, value in required.items()
    ):
        raise ValueError(
            "The fixed GPT-OSS fallback requires AITER disabled and its explicit opt-in"
        )
    env.update(required)

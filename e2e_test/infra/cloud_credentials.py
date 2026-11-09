"""Resolve explicit pod-local proxy credentials without reading provider keys."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

PROXY_ENV = "E2E_AI_KEY_PROXY"
PROXY_PLACEHOLDER_KEY = "smg-e2e-key-proxy-placeholder"
_PROVIDER_CONFIG = {
    "openai": ("https://api.openai.com", "OPENAI_API_KEY", "openai"),
    "xai": ("https://api.x.ai", "XAI_API_KEY", "openai"),
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY", "anthropic"),
}
_PROVIDER_KEY_ENVS = frozenset(config[1] for config in _PROVIDER_CONFIG.values())


@dataclass(frozen=True)
class CloudCredentials:
    worker_root: str
    api_key_env: str
    backend_type: str
    api_key: str = field(repr=False)
    is_proxy: bool
    # None preserves the SDK's normal default/env configuration in direct mode.
    sdk_base_url: str | None


def _proxy_root(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or "?" in value
        or "#" in value
        or parsed.port is None
        or not 0 < parsed.port < 65536
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{PROXY_ENV} must be an http loopback origin with an explicit port")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return f"http://{host}:{parsed.port}"


def resolve_cloud_credentials(
    provider: str,
    *,
    require_key: bool = True,
    environ: Mapping[str, str] | None = None,
) -> CloudCredentials:
    """Use a public placeholder only in explicit validated proxy mode.

    A configured invalid proxy fails before reading any key. Missing direct
    keys are returned empty only when the existing caller owns fail/skip policy.
    """
    if provider not in _PROVIDER_CONFIG:
        raise ValueError(f"Unsupported cloud backend: {provider}")
    env = os.environ if environ is None else environ
    root, key_env, backend_type = _PROVIDER_CONFIG[provider]
    proxy = env.get(PROXY_ENV, "")
    if proxy:
        worker_root = f"{_proxy_root(proxy)}/{provider}"
        sdk_base = worker_root if provider == "anthropic" else f"{worker_root}/v1"
        return CloudCredentials(
            worker_root, key_env, backend_type, PROXY_PLACEHOLDER_KEY, True, sdk_base
        )
    key = env.get(key_env, "")
    if require_key and not key:
        raise ValueError(f"{key_env} environment variable required")
    return CloudCredentials(root, key_env, backend_type, key, False, None)


def cloud_environment(source: Mapping[str, str], *, proxy: bool) -> dict[str, str]:
    """Copy child environment without even looking up provider keys in proxy mode."""
    return {name: source[name] for name in source if not proxy or name not in _PROVIDER_KEY_ENVS}

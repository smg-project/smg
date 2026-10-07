"""Credential redaction for TokenSpeed server args before they leave the engine.

Both servicer launchers (Python and Rust) serialize ``dataclasses.asdict(server_args)``
into ``GetServerInfo``. The in-engine RL control key (``rl_control_api_key``) and tokens
such as ``hf_token`` must never reach the gateway, so both run the serialized form through
:func:`redact_secrets`.
"""

from __future__ import annotations

from typing import Any

SECRET_FRAGMENTS = ("api_key", "secret", "password")


def is_secret_key(key: str) -> bool:
    """Whether a server-args key names a credential that must not leave the engine."""
    lowered = key.lower()
    return any(fragment in lowered for fragment in SECRET_FRAGMENTS) or lowered.endswith("_token")


def redact_secrets(value: Any) -> Any:
    """Drop credential-looking keys at every level of a JSON-shaped value.

    Runs on the serialized form (dicts and lists), so a secret inside a nested
    sub-config, or inside a list of them, goes too.
    """
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items() if not is_secret_key(str(k))}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value

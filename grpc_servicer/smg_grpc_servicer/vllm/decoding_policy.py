"""Engine-side declaration of the gateway's constrained-decoding policy."""

from __future__ import annotations

import argparse
import inspect
import os
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.constrained_decoding import CONSTRAINED_DECODING_POLICIES as POLICIES

POLICY_ENV = "SMG_VLLM_CONSTRAINED_DECODING"


def constrained_decoding_policy(environ: Mapping[str, str] | None = None) -> str:
    """Read the launcher's declaration; an absent declaration keeps discovery silent."""
    env = os.environ if environ is None else environ
    value = env.get(POLICY_ENV, "")
    if value and value not in POLICIES:
        raise ValueError(f"{POLICY_ENV} must be 'requested' or 'always', got {value!r}")
    return value


def add_decoding_argument(parser: argparse.ArgumentParser) -> None:
    """Add the policy declaration to a gRPC parser without replacing an existing flag."""
    if "--constrained-decoding" not in parser._option_string_actions:  # noqa: SLF001
        parser.add_argument(
            "--constrained-decoding",
            choices=POLICIES,
            default=None,
            help=(
                "Declare the constrained-decoding policy to SMG through gRPC discovery. "
                "'always' also constrains optional tool calls when the model has a grammar; "
                "'requested' applies constraints only when requested. "
                f"Unset uses ${POLICY_ENV}, otherwise SMG's default (requested)."
            ),
        )


def carry_decoding_policy(parsed: argparse.Namespace) -> str:
    """Carry a flag ahead of the environment to either servicer, before engine startup."""
    value = getattr(parsed, "constrained_decoding", None)
    if value is not None:
        if value not in POLICIES:
            raise ValueError(f"--constrained-decoding must be one of {POLICIES}, got {value!r}")
        os.environ[POLICY_ENV] = value
    return constrained_decoding_policy()


def binding_decoding_policy(server_type: Any, policy: str) -> str:
    """Keep old Rust bindings usable unless the requested policy needs the new field."""
    if not policy:
        return ""
    parameters = inspect.signature(server_type).parameters
    if "constrained_decoding" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return policy
    if policy == "always":
        raise ValueError(
            "The installed VllmGrpcServer cannot advertise constrained_decoding; "
            "update the smg Rust binding to use --constrained-decoding always"
        )
    return ""

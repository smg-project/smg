"""The CUDA-graph capture cap a gRPC launch derives from the admission step.

The engine captures a CUDA graph for each size of a list of token counts and
replays the smallest captured one that holds a step; a step above the largest
size runs eagerly, kernel by kernel from Python. Its default for that largest
size (``max_cudagraph_capture_size``) is twice the running window
(``--max-num-seqs``), at most 512 (1024 on one newer GPU platform): it
covers the decode-only steps and little else. With a window of 32 the cap is 64
tokens, and a step that admits a prefill chunk beside a dozen decodes is above
it and runs eagerly, every such step, for as long as the engine runs. An
engine whose eager step is expensive (hundreds of launches per rank: a
sparse-attention or MoE model) spends that time on the host, and its step
grows with the number of running requests instead of staying GPU-bound.

A gRPC launch derives the cap from what the scheduler can admit per step
instead, where the operator set none: ``--max-num-batched-tokens`` (unset:
unbounded, the engine's own step-size defaults are 2048 or more), at most a
ceiling of 512, the engine's own default ceiling, so the capture cost stays
within what the engine chooses for itself at a large window (51 sizes at 512
against 11 at 64; each size costs capture time at start and graph memory per
rank). The derived cap applies only where it raises the engine's default. An
operator's cap or size list (``--max-cudagraph-capture-size``,
``--cudagraph-capture-sizes``, or the same keys of ``--compilation-config``)
is kept as given, and the engine's default stays where it is the better one:
``--enforce-eager`` (no graphs), a ``cudagraph_mode`` of NONE, speculative
decoding (the engine's default adds the uniform-decode sizes an explicit cap
drops), or no ``--max-num-seqs`` (the engine's own window defaults reach the
ceiling already).

:data:`CAPTURE_CAP_ENV` is the knob: ``auto`` (the default), ``engine`` (the
opt-out: the engine's default, as before this derivation), or a positive
integer, the ceiling in place of 512, rounded down to the engine's capture
grid (1, 2, 4, multiples of 8 up to 248, multiples of 16 from 256).

The rule and the grid are the engine's (``vllm/config/vllm.py``,
``VllmConfig._set_cudagraph_sizes``, the same from 0.28 through 0.31): the
engine clamps any cap to ``max_num_batched_tokens``, builds the list as
``[1, 2, 4] + range(8, 256, 8) + range(256, cap + 1, 16)`` plus
``max_num_batched_tokens`` when it fits under the cap, and truncates a cap
that is not the list's last entry. A derived cap is the step size itself or
a grid value, so the engine takes it as it is.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

CAPTURE_CAP_ENV = "SMG_VLLM_CUDAGRAPH_CAPTURE_CAP"
AUTO = "auto"
ENGINE = "engine"
# The ceiling of the derived cap: the engine's own default ceiling (the 1024
# it uses on one newer GPU platform is a default this derivation never lowers,
# see `derive_capture_cap`).
DEFAULT_CEILING = 512
ENGINE_DEFAULT_CEILING = 512
# The engine's flags that set the cap outright, for the messages.
CAP_FLAGS = "--max-cudagraph-capture-size / --cudagraph-capture-sizes (or --compilation-config)"


def grid_floor(size: int) -> int:
    """The largest value of the engine's capture grid at or below ``size``
    (1, 2, 4, multiples of 8 up to 248, multiples of 16 from 256); 0 below 1."""
    if size < 1:
        return 0
    if size < 8:
        return 4 if size >= 4 else 2 if size >= 2 else 1
    if size < 256:
        return size - size % 8
    return size - (size - 256) % 16


def capture_sizes(cap: int) -> list[int]:
    """The engine's capture list for a cap on its grid (its default list
    rule, without the step size it appends when that fits): what a setting
    costs, in graphs to capture."""
    sizes = [i for i in (1, 2, 4) if i <= cap]
    sizes += list(range(8, min(cap + 1, 256), 8))
    sizes += list(range(256, cap + 1, 16))
    return sizes


@dataclasses.dataclass(frozen=True)
class CaptureCap:
    """What a launch does about the capture cap: the cap it sets (``None``:
    the engine's default stays), why, in a word and in a sentence."""

    cap: int | None
    source: str
    """``derived``, or why the engine's default stays: ``engine`` (the
    opt-out), ``operator`` (an explicit cap or list), ``eager`` (no graphs),
    ``speculative``, ``window`` (no ``--max-num-seqs``), ``default`` (the
    engine's default is not below the derived value)."""
    detail: str
    """The sentence the launch logs."""

    @property
    def derived(self) -> bool:
        return self.cap is not None


def ceiling_from_setting(setting: str | None) -> int | None:
    """The derivation's ceiling for a value of :data:`CAPTURE_CAP_ENV`:
    ``None`` for ``engine`` (the opt-out), :data:`DEFAULT_CEILING` for
    ``auto`` or unset, an integer rounded down to the grid; ``ValueError``
    names the variable and the value for anything else."""
    text = (setting or "").strip().lower()
    if text in ("", AUTO):
        return DEFAULT_CEILING
    if text == ENGINE:
        return None
    if text.isdigit() and int(text) >= 1:
        return grid_floor(int(text))
    raise ValueError(
        f"{CAPTURE_CAP_ENV}={setting!r}: expected {AUTO} (the default), {ENGINE} (the "
        f"engine's own capture cap) or a positive integer (the ceiling of the derived cap, "
        f"{DEFAULT_CEILING} by default)"
    )


def derive_capture_cap(
    *,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    explicit_cap: int | None = None,
    explicit_sizes: list[int] | None = None,
    enforce_eager: bool = False,
    cudagraph_mode: Any = None,
    speculative: bool = False,
    setting: str | None = None,
) -> CaptureCap:
    """The capture cap for a launch, from the scheduler's two sizes and the
    settings that make the engine's default the better one (the module's
    docstring has the table); ``ValueError`` for a ``setting`` the knob does
    not take."""
    ceiling = ceiling_from_setting(setting)
    if ceiling is None:
        return CaptureCap(
            None,
            ENGINE,
            f"the engine's own default ({CAPTURE_CAP_ENV}={ENGINE}: twice --max-num-seqs, at "
            f"most {ENGINE_DEFAULT_CEILING})",
        )
    if explicit_cap is not None or explicit_sizes:
        given = (
            f"max_cudagraph_capture_size={explicit_cap}"
            if explicit_cap is not None
            else f"cudagraph_capture_sizes={sorted(set(explicit_sizes or []))}"
        )
        return CaptureCap(None, "operator", f"the operator's {given} is kept")
    if enforce_eager:
        return CaptureCap(None, "eager", "no CUDA graphs (--enforce-eager)")
    if _is_none_mode(cudagraph_mode):
        return CaptureCap(None, "eager", "no CUDA graphs (cudagraph_mode NONE)")
    if speculative:
        return CaptureCap(
            None,
            "speculative",
            "the engine's own default under speculative decoding (it adds the uniform-decode "
            "sizes an explicit cap drops)",
        )
    if max_num_seqs is None:
        return CaptureCap(
            None,
            "window",
            "the engine's own default (--max-num-seqs unset: the engine's window defaults "
            f"reach a cap of {ENGINE_DEFAULT_CEILING} or more)",
        )
    engine_default = min(2 * max_num_seqs, ENGINE_DEFAULT_CEILING)
    cap = ceiling if max_num_batched_tokens is None else min(max_num_batched_tokens, ceiling)
    step = "unset" if max_num_batched_tokens is None else str(max_num_batched_tokens)
    if cap <= engine_default:
        return CaptureCap(
            None,
            "default",
            f"the engine's own default {engine_default} (twice --max-num-seqs {max_num_seqs}) "
            f"covers the admission step (--max-num-batched-tokens {step}, ceiling {ceiling})",
        )
    return CaptureCap(
        cap,
        "derived",
        f"{cap}, derived from the admission step (--max-num-batched-tokens {step}, ceiling "
        f"{ceiling}); the engine's own default was {engine_default} (twice --max-num-seqs "
        f"{max_num_seqs}). {CAPTURE_CAP_ENV}={ENGINE} keeps the engine's default, a number is "
        f"another ceiling, {CAP_FLAGS} set the cap outright",
    )


def _is_none_mode(mode: Any) -> bool:
    if mode is None:
        return False
    name = getattr(mode, "name", None)
    if name is not None:
        return str(name).upper() == "NONE"
    if isinstance(mode, int):
        return mode == 0
    return str(mode).strip().upper() == "NONE"


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _compilation_config(parsed: Any) -> Any:
    """The parsed ``--compilation-config``: the engine's config object, a
    dict, or ``None``; a JSON string is read, anything else is ``None``."""
    config = getattr(parsed, "compilation_config", None)
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            return None
    return config


def _field(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def launch_settings(parsed: Any, environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """The keyword arguments of :func:`derive_capture_cap`, read off a
    launcher's parsed namespace (the engine's flags: the scheduler's sizes,
    the capture settings at the top level and inside ``--compilation-config``,
    ``--enforce-eager``, ``--speculative-config``) and the environment."""
    config = _compilation_config(parsed)
    explicit_cap = getattr(parsed, "max_cudagraph_capture_size", None)
    if explicit_cap is None:
        explicit_cap = _field(config, "max_cudagraph_capture_size")
    explicit_sizes = getattr(parsed, "cudagraph_capture_sizes", None)
    if not explicit_sizes:
        explicit_sizes = _field(config, "cudagraph_capture_sizes")
    return {
        "max_num_seqs": _positive_int(getattr(parsed, "max_num_seqs", None)),
        "max_num_batched_tokens": _positive_int(getattr(parsed, "max_num_batched_tokens", None)),
        "explicit_cap": _positive_int(explicit_cap),
        "explicit_sizes": list(explicit_sizes) if explicit_sizes else None,
        "enforce_eager": bool(getattr(parsed, "enforce_eager", False)),
        "cudagraph_mode": _field(config, "cudagraph_mode"),
        "speculative": bool(getattr(parsed, "speculative_config", None)),
        "setting": environ.get(CAPTURE_CAP_ENV),
    }


def apply_capture_cap(parsed: Any, environ: Mapping[str, str] = os.environ) -> CaptureCap:
    """Derive the cap for a launcher's parsed namespace and put a derived one
    where the engine reads it: the top-level ``max_cudagraph_capture_size``
    the engine's own flag sets (the engine merges it into its compilation
    config), else a copy of the parsed ``--compilation-config`` (the parser's
    default object is shared by every parse of the process, so it is never
    written). Returns what was decided; ``ValueError`` for a knob value the
    derivation does not take."""
    cap = derive_capture_cap(**launch_settings(parsed, environ))
    if cap.cap is None:
        return cap
    if hasattr(parsed, "max_cudagraph_capture_size"):
        parsed.max_cudagraph_capture_size = cap.cap
        return cap
    config = _compilation_config(parsed)
    if isinstance(config, Mapping):
        parsed.compilation_config = {**config, "max_cudagraph_capture_size": cap.cap}
    elif config is not None and hasattr(config, "max_cudagraph_capture_size"):
        config = copy.deepcopy(config)
        config.max_cudagraph_capture_size = cap.cap
        parsed.compilation_config = config
    else:
        parsed.max_cudagraph_capture_size = cap.cap
    return cap


# The decisions this process logged: one line per distinct decision, however
# many parses the launcher runs (the CLI parses its subcommand's parser once,
# a stock launcher once; a second parse with the same flags says nothing new).
_logged: set[str] = set()


def log_capture_cap(cap: CaptureCap) -> bool:
    """One INFO line at start for what a launch does about the capture cap,
    through vLLM's handlers once vLLM configured them; returns whether this
    decision was new to the process."""
    if cap.detail in _logged:
        return False
    _logged.add(cap.detail)
    from smg_grpc_servicer.vllm import attach_vllm_logging

    attach_vllm_logging()
    logger.info("CUDA-graph capture cap: %s", cap.detail)
    return True

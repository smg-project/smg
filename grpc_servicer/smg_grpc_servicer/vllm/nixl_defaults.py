"""The launch defaults a gRPC launch gives vLLM's NIXL connector in a
prefill/decode role.

NIXL moves KV over UCX. On one host UCX moves it over CUDA IPC, and an
imported handle keeps the exporter's allocation alive after the exporter
process died; the connector keeps a mapping per remote engine for as long as
it remembers the engine, which by its own default is an hour of idleness
(``engine_ttl``). A prefill or decode engine that dies therefore leaves its
KV cache allocated on its GPU until every peer has forgotten it, and its
restart on that GPU fails the engine's startup memory check for that long.
Two settings bound the hold, and a gRPC launch applies them where the
operator set none:

- ``UCX_CUDA_IPC_CACHE=n`` in the environment the engine's processes inherit:
  UCX closes an imported handle with the transfer instead of caching it;
- ``engine_ttl`` of :data:`DEFAULT_ENGINE_TTL_SECS` in the connector's extra
  config: a peer idle that long is forgotten (the connector checks on its
  next request), and the next request to it runs the handshake again. The
  value is a trade between how long a dead peer's memory stays held and how
  often an idle pair re-runs its handshake: the connector's hour makes a
  restart wait an hour; this default makes it wait at most the restart
  ladder's first steps, and an idle pair re-handshakes at most twice a
  minute.

Both apply only to a launch whose ``--kv-transfer-config`` names
``NixlConnector`` in a prefill/decode role. A value the operator set is kept
as given: the variable with any value, the key with any value, ``0`` (the
connector's "never forget") included. A wrapper connector (``MultiConnector``)
is not read into: its children keep the engine's defaults. One INFO line says
what applies.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from collections.abc import Mapping, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)

NIXL_CONNECTOR = "NixlConnector"
# The roles of a disaggregated side; the connector refuses a connector without one.
PD_ROLES = ("kv_producer", "kv_consumer", "kv_both")
UCX_IPC_CACHE_ENV = "UCX_CUDA_IPC_CACHE"
UCX_IPC_CACHE_DEFAULT = "n"
ENGINE_TTL_KEY = "engine_ttl"
DEFAULT_ENGINE_TTL_SECS = 30.0

SOURCE_DEFAULT = "default"
SOURCE_ENVIRONMENT = "environment"
SOURCE_FLAG = "flag"


@dataclasses.dataclass(frozen=True)
class NixlDefaults:
    """What a NIXL prefill/decode launch runs with: each setting's value and
    where it came from (this module's default, or the operator's)."""

    role: str
    ipc_cache: str
    ipc_cache_source: str
    engine_ttl: Any
    engine_ttl_source: str

    @property
    def detail(self) -> str:
        return (
            f"NIXL {self.role}: {UCX_IPC_CACHE_ENV}={self.ipc_cache} ({self.ipc_cache_source}), "
            f"{ENGINE_TTL_KEY}={self.engine_ttl} ({self.engine_ttl_source})"
        )


def kv_transfer_config(parsed: Any) -> Any:
    """The parsed ``--kv-transfer-config``: a dict (the engine's parser reads
    the flag as JSON), the engine's config object, or ``None``; a JSON string
    is read, anything else is ``None``."""
    config = getattr(parsed, "kv_transfer_config", None)
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


def nixl_pd_role(parsed: Any) -> str | None:
    """The prefill/decode role of a launch on the NIXL connector, ``None`` for
    any other launch (another connector, a wrapper, no role, no config)."""
    config = kv_transfer_config(parsed)
    if config is None or _field(config, "kv_connector") != NIXL_CONNECTOR:
        return None
    role = _field(config, "kv_role")
    return role if isinstance(role, str) and role in PD_ROLES else None


def _extra_config(config: Any) -> Mapping[str, Any]:
    extra = _field(config, "kv_connector_extra_config")
    return extra if isinstance(extra, Mapping) else {}


def _with_engine_ttl(parsed: Any, ttl: float) -> None:
    """Put ``ttl`` under ``engine_ttl`` of the parsed config's extra config,
    in the shape the parse left it: a new dict for a dict, a JSON string for a
    string, a deep copy for the engine's object (the parser's default object
    is shared by every parse of the process, so it is never written)."""
    given = getattr(parsed, "kv_transfer_config", None)
    config = kv_transfer_config(parsed)
    extra = {**_extra_config(config), ENGINE_TTL_KEY: ttl}
    if isinstance(config, Mapping):
        rewritten: Any = {**config, "kv_connector_extra_config": extra}
        parsed.kv_transfer_config = json.dumps(rewritten) if isinstance(given, str) else rewritten
        return
    config = copy.deepcopy(config)
    config.kv_connector_extra_config = extra
    parsed.kv_transfer_config = config


def apply_nixl_defaults(
    parsed: Any, environ: MutableMapping[str, str] | None = None
) -> NixlDefaults | None:
    """Apply the defaults to a launcher's parsed namespace and the environment
    the engine's processes inherit, where the operator set none. Returns what
    the launch runs with, ``None`` for a launch that is not a NIXL
    prefill/decode side."""
    role = nixl_pd_role(parsed)
    if role is None:
        return None
    source = os.environ if environ is None else environ
    if source.get(UCX_IPC_CACHE_ENV):
        ipc_cache, ipc_source = source[UCX_IPC_CACHE_ENV], SOURCE_ENVIRONMENT
    else:
        source[UCX_IPC_CACHE_ENV] = UCX_IPC_CACHE_DEFAULT
        ipc_cache, ipc_source = UCX_IPC_CACHE_DEFAULT, SOURCE_DEFAULT
    extra = _extra_config(kv_transfer_config(parsed))
    if ENGINE_TTL_KEY in extra:
        ttl: Any = extra[ENGINE_TTL_KEY]
        ttl_source = SOURCE_FLAG
    else:
        _with_engine_ttl(parsed, DEFAULT_ENGINE_TTL_SECS)
        ttl, ttl_source = DEFAULT_ENGINE_TTL_SECS, SOURCE_DEFAULT
    return NixlDefaults(
        role=role,
        ipc_cache=ipc_cache,
        ipc_cache_source=ipc_source,
        engine_ttl=ttl,
        engine_ttl_source=ttl_source,
    )


# The decisions this process logged: one line per distinct decision, however
# many parses the launcher runs.
_logged: set[tuple[str, str, Any]] = set()


def log_nixl_defaults(defaults: NixlDefaults) -> bool:
    """One INFO line at start for what a NIXL prefill/decode launch runs
    with, through vLLM's handlers once vLLM configured them (held until then
    by :func:`~smg_grpc_servicer.vllm.attach_vllm_logging`); returns whether
    this decision was new to the process. The launcher parses its flags more
    than once; the second parse finds the first one's values (the variable
    now set, the key now present) and says nothing new."""
    key = (defaults.role, defaults.ipc_cache, defaults.engine_ttl)
    if key in _logged:
        return False
    _logged.add(key)
    from smg_grpc_servicer.vllm import attach_vllm_logging

    attach_vllm_logging()
    logger.info("KV transfer launch defaults: %s", defaults.detail)
    return True

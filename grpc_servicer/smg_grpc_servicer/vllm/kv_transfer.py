"""KV-transfer param passthrough between the vLLM engine proto and connector dicts."""

import json
import logging

from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.vllm.config_view import EngineConfigView

logger = logging.getLogger(__name__)

_MULTI_CONNECTOR = "MultiConnector"
_MOONCAKE_CONNECTOR = "MooncakeConnector"
# The connectors that hand a request's KV to the one partner the router names
# (a rendezvous or side channel per pair): the router tags both sides of such
# a pair, so the child that carries the handoff is the one it must see.
_P2P_CONNECTORS = frozenset({_MOONCAKE_CONNECTOR, "NixlConnector", "MoRIIOConnector"})
# The P2P connector whose handoff the gRPC router does not drive (its pipeline
# refuses a disaggregated request on a worker that names it): beside exactly
# one store child the store is the mover the router can serve, as it was when
# the wrapper passed through unprojected.
_MORIIO_CONNECTOR = "MoRIIOConnector"
# The connectors whose sides meet in a store the deployment shares, by exact
# name or family prefix: the router's store profile (no side tagged, no
# bootstrap port handed or probed) applies to a worker that names one of them.
# The router's table (`model_gateway/src/worker/pd_pairing.rs`,
# `STORE_CONNECTORS` / `STORE_CONNECTOR_FAMILIES`) is the reference; a name
# listed there and not here is reported as the wrapper, which the router
# handles as an unknown connector.
_STORE_CONNECTORS = frozenset({"MooncakeStoreConnector", "SharedStorageConnector"})
_STORE_CONNECTOR_FAMILIES = ("FlexKV", "HF3FS")


def pairing_fields(vllm_config: object) -> dict:
    """The KV-layout facts a PD peer must share, read off the engine config.

    Model dtype, KV cache dtype and block size are the config-level factors
    vLLM folds into its NIXL compatibility hash. The attention backend is the
    *requested* one (`--attention-backend`): the resolved backend is picked
    inside the worker process, so on the common auto path this field is
    absent and the router treats it as unknown rather than as a match. Every
    field degrades to its proto default when the config does not carry it.
    """
    view = EngineConfigView(vllm_config)
    fields: dict = {}
    if view.cache.cache_dtype is not None:
        fields["kv_cache_dtype"] = view.cache.cache_dtype
    if view.cache.block_size is not None:
        fields["block_size"] = view.cache.block_size
    if view.attention_backend is not None:
        fields["attention_backend"] = view.attention_backend
    if view.model_dtype is not None:
        fields["model_dtype"] = view.model_dtype
    return fields


def _is_store_connector(connector: str) -> bool:
    return connector in _STORE_CONNECTORS or connector.startswith(_STORE_CONNECTOR_FAMILIES)


def runs_mooncake(vllm_config: object) -> bool:
    """Whether a Mooncake connector runs in this engine at all.

    Read off the raw ``kv_transfer_config``, not the projection
    :func:`resolve_pd_connector` reports: the top-level connector, or any
    child a ``MultiConnector`` lists, at any depth (vLLM builds each child
    from its own dict, so a wrapper may list a wrapper). A wrapper hands
    every child the allocation a request gets and merges every child's
    finished transfers, so a Mooncake child acts under whatever name the
    projection reports (a second P2P child beside it, an empty child engine
    id, a malformed sibling all leave the projection at the wrapper).
    """
    config = EngineConfigView(vllm_config).kv_transfer.config
    if config is None:
        return False
    return _names_mooncake(
        getattr(config, "kv_connector", None),
        getattr(config, "kv_connector_extra_config", None),
    )


def _names_mooncake(connector: object, extra_config: object) -> bool:
    """Whether ``connector`` is Mooncake or any connector under it is (the
    ``connectors`` of a wrapper's extra config, each child a dict of the same
    shape); a malformed child counts as no connector."""
    if connector == _MOONCAKE_CONNECTOR:
        return True
    children = extra_config.get("connectors") if isinstance(extra_config, dict) else None
    if not isinstance(children, list):
        return False
    return any(
        isinstance(child, dict)
        and _names_mooncake(child.get("kv_connector"), child.get("kv_connector_extra_config"))
        for child in children
    )


def resolve_pd_connector(config: object) -> tuple[str, str]:
    """Resolve the connector and engine id that SMG should report for PD.

    A ``MultiConnector`` is projected onto the child that moves the KV between
    the prefill and the decode: its one P2P child (Mooncake, NIXL, MoRI-IO),
    which carries the pair's handoff whatever rides along (a store, an
    offloading tier, a connector this projection does not know); else its one
    store child, which the sides meet in when no P2P child is configured. A
    MoRI-IO child is the exception: the gRPC router refuses a disaggregated
    request on a worker that names it, so beside store children it steps
    aside, for the one store child when there is exactly one (the store moved
    the KV while the wrapper passed through unprojected) and for the wrapper
    when there are more. The child's engine id is reported, the parent's when
    the child names none, as the engine itself resolves it. Wrapper configs
    that are malformed, that name no such child, or that name two of a kind
    are returned unchanged so the router does not guess which transfer
    protocol to use.
    """
    connector = getattr(config, "kv_connector", None) or ""
    engine_id = getattr(config, "engine_id", None) or ""
    if connector != _MULTI_CONNECTOR:
        return connector, engine_id

    extra_config = getattr(config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, dict):
        return connector, engine_id
    children = extra_config.get("connectors")
    if not isinstance(children, list):
        return connector, engine_id

    p2p_children = []
    store_children = []
    for child in children:
        if not isinstance(child, dict):
            return connector, engine_id
        child_connector = child.get("kv_connector")
        if not isinstance(child_connector, str) or not child_connector:
            return connector, engine_id
        if child_connector in _P2P_CONNECTORS:
            p2p_children.append(child)
        elif _is_store_connector(child_connector):
            store_children.append(child)

    if len(p2p_children) == 1:
        mover = p2p_children[0]
        if mover["kv_connector"] == _MORIIO_CONNECTOR and store_children:
            if len(store_children) != 1:
                return connector, engine_id
            mover = store_children[0]
    elif not p2p_children and len(store_children) == 1:
        mover = store_children[0]
    else:
        return connector, engine_id

    mover_engine_id = mover.get("engine_id", engine_id)
    if not isinstance(mover_engine_id, str) or not mover_engine_id:
        return connector, engine_id
    return mover["kv_connector"], mover_engine_id


def params_from_request(
    request: vllm_engine_pb2.GenerateRequest,
) -> dict | None:
    """Extract KV-transfer params; JSON field preferred, legacy typed field as fallback.

    Raises:
        ValueError: If the JSON field is malformed or the legacy field is invalid.
    """
    if request.HasField("kv_transfer_params_json"):
        try:
            params = json.loads(request.kv_transfer_params_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid kv_transfer_params_json: {e}") from e
        if not isinstance(params, dict):
            raise ValueError("kv_transfer_params_json must be a JSON object")
        return params
    if request.HasField("kv_transfer_params"):
        remote_host = request.kv_transfer_params.remote_host
        remote_port = request.kv_transfer_params.remote_port
        if not remote_host or not (1 <= remote_port <= 65535):
            raise ValueError(
                "Invalid kv_transfer_params: remote_host must be set and remote_port must be in [1, 65535]."
            )
        return {"remote_host": remote_host, "remote_port": remote_port}
    return None


def params_to_response_fields(
    params: dict | None,
) -> tuple[vllm_engine_pb2.KvTransferParams | None, str | None]:
    """Map engine-returned params to (legacy typed message, JSON field) for GenerateComplete."""
    if not params:
        return None, None

    params_json = None
    try:
        params_json = json.dumps(params)
    except (TypeError, ValueError):
        logger.warning("Dropping non-JSON-serializable kv_transfer_params: %r", params)

    # Legacy mirror for old routers; built only when host/port are valid (Mooncake shape)
    legacy = None
    remote_host = params.get("remote_host", "")
    remote_port = params.get("remote_port", 0)
    if (
        isinstance(remote_host, str)
        and remote_host
        and isinstance(remote_port, int)
        and 1 <= remote_port <= 65535
    ):
        legacy = vllm_engine_pb2.KvTransferParams(
            remote_host=remote_host,
            remote_port=remote_port,
        )
    return legacy, params_json

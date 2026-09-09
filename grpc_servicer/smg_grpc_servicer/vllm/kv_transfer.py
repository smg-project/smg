"""KV-transfer param passthrough between the vLLM engine proto and connector dicts."""

import json
import logging

from smg_grpc_proto import vllm_engine_pb2

logger = logging.getLogger(__name__)

_MULTI_CONNECTOR = "MultiConnector"
_SUPPORTED_PD_CONNECTORS = frozenset({"MooncakeConnector", "NixlConnector"})


def pairing_fields(vllm_config: object) -> dict:
    """The KV-layout facts a PD peer must share, read off the engine config.

    Model dtype, KV cache dtype and block size are the config-level factors
    vLLM folds into its NIXL compatibility hash. The attention backend is the
    *requested* one (`--attention-backend`): the resolved backend is picked
    inside the worker process, so on the common auto path this field is
    absent and the router treats it as unknown rather than as a match. Every
    field degrades to its proto default when the config does not carry it.
    """
    fields: dict = {}
    cache = getattr(vllm_config, "cache_config", None)
    if cache is not None:
        dtype = getattr(cache, "cache_dtype", None)
        if dtype:
            fields["kv_cache_dtype"] = str(dtype)
        block_size = getattr(cache, "block_size", None)
        if isinstance(block_size, int) and block_size > 0:
            fields["block_size"] = block_size
    attention = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention, "backend", None) if attention is not None else None
    if backend is not None:
        fields["attention_backend"] = str(getattr(backend, "name", backend))
    model = getattr(vllm_config, "model_config", None)
    dtype = getattr(model, "dtype", None) if model is not None else None
    if dtype is not None:
        fields["model_dtype"] = str(dtype)
    return fields


def resolve_pd_connector(config: object) -> tuple[str, str]:
    """Resolve the connector and engine id that SMG should report for PD.

    MultiConnector is only projected when its config has exactly one supported
    PD child. Invalid or ambiguous wrapper configs are returned unchanged so the
    router does not guess which transfer protocol to use.
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

    pd_children = []
    for child in children:
        if not isinstance(child, dict):
            return connector, engine_id
        child_connector = child.get("kv_connector")
        if not isinstance(child_connector, str) or not child_connector:
            return connector, engine_id
        if child_connector in _SUPPORTED_PD_CONNECTORS:
            pd_children.append(child)

    if len(pd_children) != 1:
        return connector, engine_id

    child = pd_children[0]
    child_engine_id = child.get("engine_id", engine_id)
    if not isinstance(child_engine_id, str) or not child_engine_id:
        return connector, engine_id
    return child["kv_connector"], child_engine_id


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

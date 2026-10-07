"""TokenSpeed-specific KV-events config resolution.

The wire-format conversion + ZMQ streaming helpers live in the engine-neutral
``smg_grpc_servicer.kv_events`` module; this module only resolves whether (and
where) TokenSpeed is publishing ZMQ KV events.

TokenSpeed's ``--kv-events-config`` is a raw JSON string on ``ServerArgs``
(``server_args.kv_events_config``), unlike vLLM's already-parsed config object.
We parse it with stdlib ``json`` (not TokenSpeed's pydantic ``KVEventsConfig``)
so this resolver stays unit-testable without TokenSpeed installed; the defaults
below mirror ``tokenspeed.runtime.pd.kv_events.KVEventsConfig``.
"""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Defaults mirror tokenspeed.runtime.pd.kv_events.KVEventsConfig.
_DEFAULT_ENDPOINT = "tcp://*:5557"
_DEFAULT_TOPIC = ""


@dataclass(frozen=True)
class ResolvedKvEventsConfig:
    """The subset of KVEventsConfig the bridge needs to open a SUB socket, and
    the replay ROUTER endpoint (empty when the publisher runs none) the relay
    asks for gaps and for the batches published before its subscription joined."""

    endpoint: str
    topic: str
    replay_endpoint: str = ""


def resolve_kv_events_config(server_args: object) -> ResolvedKvEventsConfig | None:
    """Return the resolved ZMQ endpoint iff TokenSpeed KV events are enabled, else None.

    Reads ``server_args.kv_events_config`` (a JSON string). Returns ``None`` —
    leaving ``SubscribeKvEvents`` to report UNIMPLEMENTED — when events are
    disabled, the publisher is not ``zmq``, or the config is absent/malformed.
    """
    raw = getattr(server_args, "kv_events_config", None)
    if not raw:
        return None

    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.warning("Could not parse --kv-events-config %r: %s", raw, e)
        return None
    if not isinstance(cfg, dict):
        logger.warning("--kv-events-config did not decode to an object: %r", raw)
        return None

    if not cfg.get("enable_kv_cache_events", False):
        logger.info("TokenSpeed KV cache events not enabled; SubscribeKvEvents disabled")
        return None

    # publisher unset → "zmq" when enabled (matches EventPublisherFactory.create).
    publisher = cfg.get("publisher")
    if publisher is None:
        publisher = "zmq"
    if publisher != "zmq":
        logger.info(
            "TokenSpeed KV events publisher is %r, not 'zmq'; SubscribeKvEvents disabled",
            publisher,
        )
        return None

    endpoint = cfg.get("endpoint") or _DEFAULT_ENDPOINT
    topic = cfg.get("topic")
    if topic is None:
        topic = _DEFAULT_TOPIC
    # Guard against non-string values from arbitrary JSON; otherwise they fail
    # later (e.g. topic.encode()) as opaque INTERNAL errors at subscribe time.
    if not isinstance(endpoint, str) or not isinstance(topic, str):
        logger.warning(
            "TokenSpeed kv-events endpoint/topic must be strings (got endpoint=%r topic=%r); "
            "SubscribeKvEvents disabled",
            endpoint,
            topic,
        )
        return None
    replay_endpoint = cfg.get("replay_endpoint") or ""
    if not isinstance(replay_endpoint, str):
        logger.warning(
            "TokenSpeed kv-events replay_endpoint must be a string (got %r); replay disabled",
            replay_endpoint,
        )
        replay_endpoint = ""
    logger.info(
        "TokenSpeed KV events enabled: endpoint=%s replay_endpoint=%s",
        endpoint,
        replay_endpoint or "(none)",
    )
    return ResolvedKvEventsConfig(endpoint=endpoint, topic=topic, replay_endpoint=replay_endpoint)

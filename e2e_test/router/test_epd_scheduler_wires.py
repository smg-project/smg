"""The encode-prefill-decode gRPC path on the vLLM and the TokenSpeed wires of the mock, on the CPU row.

The gateway's vLLM form of encode-prefill-decode sends a request's media to
one encode worker first (the encode stage's push arm): the encoder answers a token-less
completion with the media identity and one encoder-cache handle per item, the
prefill side receives the identity and the handles, the decode side the
identity alone. The mock worker serves the encode role on this wire
(``--worker-type encode``): it advertises the producer side of the handoff
and names each media reference by a digest.

The TokenSpeed form sends the media to the encode worker's encoder service
(its ``Encode``) with one rendezvous room per media item, and hands the prefill
side those rooms with the encoder's bootstrap server in its
``encode_bootstrap_info``; the embedding would travel between the two engines
out of band. The mock worker serves the encode role on this wire too: its
``Encode`` accepts the media and the rooms, its prefill side checks that each
media item has a room, and nothing moves.

Each class starts one encode, one prefill and one decode mock worker on its
wire and drives a text turn (prefill and decode alone) and an image turn
(through the encoder) through the gateway, with the roles read from
``/workers`` and the encode side from the encode worker's own count of the
encode requests it served (its admin API).

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_epd_scheduler_wires.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest
from fixtures.mock_answers import class_workers
from infra.constants import WorkerType

# The mock engine reads this model's tokenizer and chat template; an image
# turn needs a template with an image placeholder, so the model is the
# multimodal chat suite's (its files are json alone on the CPU lane).
_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-epd-wires-{os.getpid()}"
_GATEWAY_ARGS = ["--prefill-policy", "round_robin", "--decode-policy", "round_robin"]

_ANSWER = "The capital of France is Paris."
_QUESTION = [{"role": "user", "content": "What is the capital of France?"}]
# A solid 32x32 red PNG: the mock encoder names it by a digest and fetches nothing.
_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/"
    "un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="
)
_IMAGE_QUESTION = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"}},
        ],
    }
]
_IMAGE_ANSWER = "The image is red."


def _registered_workers(gateway) -> list[dict]:
    resp = httpx.get(f"{gateway.base_url}/workers", timeout=5.0)
    assert resp.status_code == 200, f"GET /workers returned {resp.status_code}: {resp.text}"
    workers = resp.json().get("workers")
    assert workers, f"the gateway registered no worker: {resp.text}"
    return workers


def _by_role(gateway) -> dict[str, dict]:
    return {worker["worker_type"]: worker for worker in _registered_workers(gateway)}


def _encodes_served() -> int:
    """The encoder completions the class's encode worker served so far, read
    off its admin API (the gateway counts no request on the encode side)."""
    encoders = [w for w in class_workers() if w.worker_type == WorkerType.ENCODE]
    assert len(encoders) == 1, f"one encode worker expected: {encoders}"
    resp = httpx.get(f"{encoders[0].admin_url}/admin/ops/all", timeout=5.0)
    assert resp.status_code == 200, f"GET /admin/ops/all returned {resp.status_code}: {resp.text}"
    return sum(int(entry.get("encodes") or 0) for entry in resp.json().get("workers") or [])


class _EncodePrefillDecodeTurns:
    """The two turns every wire's class drives: a text turn that needs no
    encoder and an image turn that goes through the encode worker first."""

    def test_a_text_turn_runs_on_prefill_and_decode_alone(self, setup_backend, mock_answers):
        """A text turn needs no encoder: the scripted answer comes back with a
        stop and a usage, and the encode worker's count of encode requests stands."""
        _backend, model, client, _gateway = setup_backend
        encoded_before = _encodes_served()
        mock_answers.enqueue(_ANSWER)
        response = client.chat.completions.create(
            model=model, messages=_QUESTION, max_tokens=64, temperature=0
        )
        choice = response.choices[0]
        assert choice.message.content == _ANSWER
        assert choice.finish_reason == "stop"
        assert response.usage is not None and response.usage.completion_tokens > 0
        assert _encodes_served() == encoded_before

    def test_an_image_turn_runs_through_the_encoder(self, setup_backend, mock_answers):
        """An image turn goes to the encode worker first (its count of encode
        requests rises by one), then to the prefill and the decode side:
        the decode side's scripted answer comes back with a stop and a usage."""
        _backend, model, client, _gateway = setup_backend
        encoded_before = _encodes_served()
        mock_answers.enqueue(_IMAGE_ANSWER)
        response = client.chat.completions.create(
            model=model, messages=_IMAGE_QUESTION, max_tokens=64, temperature=0
        )
        choice = response.choices[0]
        assert choice.message.content == _IMAGE_ANSWER
        assert choice.finish_reason == "stop"
        assert response.usage is not None and response.usage.prompt_tokens > 0
        assert _encodes_served() == encoded_before + 1, (
            "the encode worker served the image turn's encode side"
        )


def _healthy_roles(gateway, runtime: str) -> dict[str, dict]:
    """The three roles registered under `runtime`, every worker healthy."""
    by_role = _by_role(gateway)
    assert set(by_role) == {"encode", "prefill", "decode"}, f"the roles registered: {by_role}"
    runtimes = {role: worker.get("runtime_type") for role, worker in by_role.items()}
    assert set(runtimes.values()) == {runtime}, runtimes
    unhealthy = [role for role, worker in by_role.items() if not worker.get("is_healthy")]
    assert not unhealthy, f"unhealthy workers after startup: {unhealthy}"
    return by_role


@pytest.mark.engine("mock")
@pytest.mark.gpu(3)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(extra_engine_args=["--grpc-wire", "vllm"])
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR / "vllm"), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("epd_grpc", (1, 1, 1)), id="1e1p1d")], indirect=True
)
class TestVllmWireEncodePrefillDecode(_EncodePrefillDecodeTurns):
    """The three roles on the vLLM wire of the mock, through the gateway."""

    def test_the_three_roles_register_under_this_wire(self, setup_backend):
        """The encode, the prefill and the decode worker register under the
        vLLM runtime, each with its role, both ends of the encoder-cache
        handoff labelled: the encoder as the producer, the prefill side as
        the consumer, on one connector."""
        _backend, _model, _client, gateway = setup_backend
        by_role = _healthy_roles(gateway, "vllm")
        labels = {role: worker.get("labels") or {} for role, worker in by_role.items()}
        assert labels["encode"].get("supports_ec_transfer") == "true", labels["encode"]
        assert labels["encode"].get("ec_role") == "ec_producer", labels["encode"]
        assert labels["prefill"].get("ec_role") == "ec_consumer", labels["prefill"]
        assert labels["encode"].get("ec_connector") == labels["prefill"].get("ec_connector")
        assert "ec_role" not in labels["decode"], labels["decode"]


@pytest.mark.engine("mock")
@pytest.mark.gpu(3)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(extra_engine_args=["--grpc-wire", "tokenspeed"])
@pytest.mark.gateway(
    log_level="debug", log_dir=str(_LOG_DIR / "tokenspeed"), extra_args=_GATEWAY_ARGS
)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("epd_grpc", (1, 1, 1)), id="1e1p1d")], indirect=True
)
class TestTokenSpeedWireEncodePrefillDecode(_EncodePrefillDecodeTurns):
    """The three roles on the TokenSpeed wire of the mock, through the gateway:
    the encode side is the encoder service's ``Encode`` with one rendezvous room
    per media item, which the prefill side receives in its bootstrap info."""

    def test_the_three_roles_register_under_this_wire(self, setup_backend):
        """The encode, the prefill and the decode worker register under the
        TokenSpeed runtime, each with its role. The handoff here is the
        rendezvous the gateway mints per media item, not an encoder-cache
        connector: the encode worker advertises the bootstrap port where the
        prefill side's engine would come for the embedding, and no
        encoder-cache labels."""
        _backend, _model, _client, gateway = setup_backend
        by_role = _healthy_roles(gateway, "tokenspeed")
        labels = {role: worker.get("labels") or {} for role, worker in by_role.items()}
        for role in ("encode", "prefill", "decode"):
            assert labels[role].get("disaggregation_mode") == role, labels[role]
        assert labels["encode"].get("disaggregation_bootstrap_port"), labels["encode"]
        assert "ec_role" not in labels["encode"], labels["encode"]
        assert "ec_role" not in labels["prefill"], labels["prefill"]

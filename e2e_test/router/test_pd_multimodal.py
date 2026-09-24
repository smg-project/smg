"""Cross-image KV isolation tests for vLLM PD disaggregation (gRPC mode).

Two same-resolution images behind an identical text prefix expand to
identical placeholder-token runs; the decode worker must still keep their
KV apart (per-image mm identity) and, for M-RoPE models, decode with
grid-aware positions from the relayed grid tensors.

Requirements: same as test_pd_mmlu (2 GPUs, 1 prefill + 1 decode). Runs
under both NIXL and Mooncake KV backends. Under ``E2E_MM_PROCESSING=worker``
the legs fetch and process the media themselves, and each class asserts the
path the gateway took: Qwen3-VL forwards references, Phi-3.5 stays on the
router path because its placeholder is not one a vLLM worker expands.

The language-model-only lanes run the decode pool with
``--language-model-only`` (no vision encoder, encoder-cache budget 0): the
grid-less model takes the fully stripped decode leg, while the M-RoPE model
cannot pair with such a pool and must be rejected with a clear 400.

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_multimodal.py -v
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import httpx
import pytest
from infra import assert_mm_processing, get_mm_processing
from infra.constants import MM_PROCESSING_WORKER
from infra.mm_processing import mm_processing_samples
from infra.pd_logs import assert_worker_logs_captured
from PIL import Image

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"

# One shared resolution so both images expand to identical placeholder-token
# runs — the collision precondition these tests probe.
_PROBE_SIZE = (448, 448)

_PROMPT = "Is the dog wrapped in a blanket? Answer with one word: yes or no."

# The servicer's request line for a pixel-less leg built from an identity.
_DECODE_FROM_IDENTITY = "preprocessed_mm=True, media_refs=0"


def _worker_path_count(gateway) -> float:
    """Requests the gateway has sent down the worker-side media path."""
    return sum(
        value for labels, value in mm_processing_samples(gateway) if labels["mode"] == "worker"
    )


def _fixture_image_data_url(name: str) -> str:
    buf = io.BytesIO()
    Image.open(_FIXTURES_DIR / name).convert("RGB").resize(_PROBE_SIZE).save(buf, format="JPEG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


_DOG_IMAGE = _fixture_image_data_url("dog.jpg")  # black lab, no blanket -> "no"
_PUG_IMAGE = _fixture_image_data_url("pug.jpg")  # pug in a blanket -> "yes"


def _blanket_messages(image_url: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


def _first_word(answer: str | None) -> str:
    words = (answer or "").lower().split()
    return words[0].strip(".,!?\"'") if words else ""


def _ask_blanket(client, model: str, image_url: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=_blanket_messages(image_url),
        temperature=0,
        max_tokens=16,
    )
    return _first_word(response.choices[0].message.content)


def _assert_no_cross_image_aliasing(client, model: str) -> None:
    # Seed the decode worker's prefix cache with the blanket-less dog's KV.
    first = _ask_blanket(client, model, _DOG_IMAGE)
    logger.info("PD multimodal seed answer (dog, no blanket): %s", first)
    assert first == "no", f"Expected 'no' for the seed image, got: {first}"

    # Same text prefix, same-resolution image, different pixels. Without
    # per-image decode-side block hashes this aliases onto the seed
    # request's cached KV and answers 'no'.
    second = _ask_blanket(client, model, _PUG_IMAGE)
    logger.info("PD multimodal probe answer (pug in blanket): %s", second)
    assert second == "yes", (
        f"Expected 'yes', got: {second} — a 'no' answer means the decode "
        "worker served this request from the other image's KV "
        "(prefix-cache contamination across images)"
    )

    # Same-image reuse must still answer correctly: per-image hashing is
    # deterministic, so this leg may hit the decode prefix cache but must
    # land on the right KV.
    third = _ask_blanket(client, model, _DOG_IMAGE)
    logger.info("PD multimodal reuse answer (dog, no blanket): %s", third)
    assert third == "no", f"Expected 'no' on same-image reuse, got: {third}"


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("microsoft/Phi-3.5-vision-instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMultimodalKvIsolation:
    """Grid-less (standard-RoPE) decode leg: isolation via mm cache_salt."""

    def test_different_images_same_prefix_do_not_alias(self, setup_backend):
        backend, model, client, gateway = setup_backend
        _assert_no_cross_image_aliasing(client, model)
        assert_mm_processing(gateway, worker_expandable=False)


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMultimodalMrope:
    """M-RoPE decode leg: relayed grid tensors give correct positions.

    Without the grids the decode worker mis-rotates every generated token
    and answers with degenerate output, so any correct answer implies the
    position relay works; the three-probe sequence also covers isolation.
    """

    def test_mrope_decode_answers_and_does_not_alias(self, setup_backend):
        backend, model, client, gateway = setup_backend
        _assert_no_cross_image_aliasing(client, model)
        assert_mm_processing(gateway, worker_expandable=True)

    def test_parallel_sampling_keeps_vision_on_decode(self, setup_backend):
        # n>1 skips the KV handoff, so the decode leg recomputes the prompt
        # locally and must carry the full multimodal payload to run the
        # vision encoder (a pixel-less leg would answer image-blind).
        backend, model, client, *_ = setup_backend
        response = client.chat.completions.create(
            model=model,
            messages=_blanket_messages(_PUG_IMAGE),
            # Near-greedy: vLLM rejects n>1 with temperature=0.
            temperature=0.1,
            max_tokens=16,
            n=2,
        )
        assert len(response.choices) == 2
        for choice in response.choices:
            answer = _first_word(choice.message.content)
            logger.info("PD multimodal n=2 answer (pug in blanket): %s", answer)
            assert answer == "yes", f"Expected 'yes' from each sample, got: {answer}"

    def test_worker_media_is_processed_once_on_the_prefill_leg(self, setup_backend):
        # With worker-side processing the prefill leg fetches and processes
        # the image and answers with its identity; the decode leg is served
        # from that identity, without references and without pixels.
        if get_mm_processing() != MM_PROCESSING_WORKER:
            pytest.skip("router-side lane: the legs receive preprocessed tensors")
        backend, model, client, gateway = setup_backend
        prefill, decode = gateway.prefill_workers[0], gateway.decode_workers[0]
        # A serving worker has logged its startup, so an empty log means the
        # output is not captured: a skip locally (SHOW_WORKER_LOGS=1), a
        # failure in CI, where the lane always writes log files.
        for leg, worker in (("prefill", prefill), ("decode", decode)):
            assert_worker_logs_captured(worker.read_log(), f"the {leg} leg's media placement")
        worker_before = _worker_path_count(gateway)
        prefill_before = prefill.read_log().count("media_refs=1")
        decode_refs_before = decode.read_log().count("media_refs=1")
        decode_identity_before = decode.read_log().count(_DECODE_FROM_IDENTITY)

        assert _ask_blanket(client, model, _PUG_IMAGE) == "yes"

        assert _worker_path_count(gateway) == worker_before + 1
        assert prefill.read_log().count("media_refs=1") == prefill_before + 1, (
            "the prefill leg processed the reference"
        )
        assert decode.read_log().count("media_refs=1") == decode_refs_before, (
            "the decode leg received no reference to process"
        )
        assert decode.read_log().count(_DECODE_FROM_IDENTITY) == decode_identity_before + 1, (
            "the decode leg was served from the prefill's media identity"
        )


def _post_blanket(gateway, model: str, image_url: str, **sampling) -> httpx.Response:
    return httpx.post(
        f"{gateway.base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": _blanket_messages(image_url),
            **sampling,
        },
        timeout=120.0,
    )


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("microsoft/Phi-3.5-vision-instruct")
@pytest.mark.e2e
@pytest.mark.parametrize(
    "setup_backend",
    [("pd_grpc", (1, 1, {"decode_args": ["--language-model-only"]}))],
    indirect=True,
)
class TestPDMultimodalLanguageModelOnlyDecode:
    """Language-model-only decode pool (the production P/D shape).

    The decode workers run with no vision encoder and an encoder-cache
    budget of 0; the gateway strips the decode leg down to the expanded
    token ids, the KV handoff and the per-image content hashes, so the
    decode engine sees a pure-text prompt and never schedules its encoder
    cache. Grid-less (standard-RoPE) model: isolation still comes from the
    hash-derived cache_salt.
    """

    def test_different_images_same_prefix_do_not_alias(self, setup_backend):
        backend, model, client, gateway = setup_backend
        _assert_no_cross_image_aliasing(client, model)
        # A language-model-only decode worker cannot expand media references,
        # so the media stay router-processed.
        assert_mm_processing(gateway, worker_expandable=False)

    def test_parallel_sampling_is_rejected_clearly(self, setup_backend):
        # n>1 has no KV handoff; a pixel-less decode worker cannot recompute
        # a multimodal prompt locally, so the gateway must say so in a 400
        # rather than fail deep in the engine.
        backend, model, client, gateway = setup_backend
        resp = _post_blanket(gateway, model, _PUG_IMAGE, temperature=0.1, max_tokens=16, n=2)
        logger.info("n>1 on language-model-only decode: %s %s", resp.status_code, resp.text[:300])
        assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text[:300]}"
        assert "language-model-only" in resp.text


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.parametrize(
    "setup_backend",
    [("pd_grpc", (1, 1, {"decode_args": ["--language-model-only"]}))],
    indirect=True,
)
class TestPDMultimodalMropeLanguageModelOnlyDecode:
    """M-RoPE models cannot pair with a language-model-only decode pool.

    The decode leg needs the grid tensors for positions; the pixel-less
    decode worker cannot accept them. The gateway rejects the pairing with a
    clear 400 instead of silently mis-rotating positions.
    """

    def test_mrope_request_is_rejected_clearly(self, setup_backend):
        backend, model, client, gateway = setup_backend
        resp = _post_blanket(gateway, model, _PUG_IMAGE, temperature=0, max_tokens=16)
        logger.info(
            "M-RoPE on language-model-only decode: %s %s", resp.status_code, resp.text[:300]
        )
        assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text[:300]}"
        assert "language-model-only" in resp.text

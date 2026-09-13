"""Cross-image KV isolation tests for vLLM PD disaggregation (gRPC mode).

Two same-resolution images behind an identical text prefix expand to
identical placeholder-token runs; the decode worker must still keep their
KV apart (per-image mm identity) and, for M-RoPE models, decode with
grid-aware positions from the relayed grid tensors.

Requirements: same as test_pd_mmlu (2 GPUs, 1 prefill + 1 decode). Runs
under both NIXL and Mooncake KV backends.

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_multimodal.py -v
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import pytest
from PIL import Image

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"

# One shared resolution so both images expand to identical placeholder-token
# runs — the collision precondition these tests probe.
_PROBE_SIZE = (448, 448)

_PROMPT = "Is the dog wrapped in a blanket? Answer with one word: yes or no."


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
        backend, model, client, *_ = setup_backend
        _assert_no_cross_image_aliasing(client, model)


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
        backend, model, client, *_ = setup_backend
        _assert_no_cross_image_aliasing(client, model)

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

"""Cross-image KV isolation test for vLLM PD disaggregation (gRPC mode).

Regression test for decode-side prefix-cache image contamination. The PD
router strips multimodal tensors from the decode leg, and an image region in
the expanded prompt is a run of one repeated placeholder token id — so two
different same-resolution images behind an identical text prefix produce
byte-identical decode-side token sequences. Unless the router carries the
per-image identity into the decode worker's block hashes (the mm cache_salt),
the second request takes a local prefix-cache hit onto the first request's KV
and is answered from the wrong image.

The probe images share one resolution on purpose: identical vision grids
expand to identical placeholder-token runs, which is the collision
precondition this test probes. Unlike the KV-transfer tests, output
correctness is a sufficient signal here — before the cache_salt fix the
second answer names the first image's color.

The model must be a standard-RoPE VLM: on an M-RoPE model (Qwen-VL family)
the tensor-stripped decode leg computes text-only positions that disagree
with the prefill's grid-aware positions, so PD decoding is degenerate with
or without the salt — a separate bug this test cannot see past.

Requirements: same as test_pd_mmlu (2 GPUs, 1 prefill + 1 decode). Runs
under both NIXL and Mooncake KV backends.

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_multimodal.py -v
"""

from __future__ import annotations

import base64
import io
import logging

import pytest
from PIL import Image

logger = logging.getLogger(__name__)

# Large enough that the color survives any processor resize; identical for
# every probe image so the placeholder-token runs are identical.
_PROBE_SIZE = (448, 448)

_PROMPT = "What is the dominant color of this image? Answer with one English word."


def _solid_image_data_url(color: tuple[int, int, int]) -> str:
    buf = io.BytesIO()
    Image.new("RGB", _PROBE_SIZE, color).save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


_RED_IMAGE = _solid_image_data_url((220, 20, 20))
_BLUE_IMAGE = _solid_image_data_url((20, 20, 220))


def _ask_color(client, model: str, image_url: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0,
        max_tokens=16,
    )
    return (response.choices[0].message.content or "").lower()


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("microsoft/Phi-3.5-vision-instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMultimodalKvIsolation:
    """Decode-side prefix-cache isolation between different images."""

    def test_different_images_same_prefix_do_not_alias(self, setup_backend):
        backend, model, client, *_ = setup_backend

        # Seed the decode worker's prefix cache with the red image's KV.
        first = _ask_color(client, model, _RED_IMAGE)
        logger.info("PD multimodal seed answer (red image): %s", first)
        assert "red" in first, f"Expected 'red' for the seed image, got: {first}"

        # Same text prefix, same-resolution image, different pixels. Without
        # per-image decode-side block hashes this aliases onto the red
        # request's cached KV and answers 'red'.
        second = _ask_color(client, model, _BLUE_IMAGE)
        logger.info("PD multimodal probe answer (blue image): %s", second)
        assert "blue" in second, (
            f"Expected 'blue', got: {second} — a 'red' answer means the decode "
            "worker served this request from the other image's KV "
            "(prefix-cache contamination across images)"
        )

        # Same-image reuse must still answer correctly: the salt is
        # deterministic per image content, so this leg may hit the decode
        # prefix cache but must land on the right KV.
        third = _ask_color(client, model, _RED_IMAGE)
        logger.info("PD multimodal reuse answer (red image): %s", third)
        assert "red" in third, f"Expected 'red' on same-image reuse, got: {third}"

"""EPD (Encode-Prefill-Decode) multimodal Chat Completions E2E tests.

Exercises TokenSpeed's EPD disaggregation on a vision-language model across four
worker-count topologies: the encode worker runs the vision tower, prefill/decode
run the LM, and the gateway stitches encode -> prefill -> decode.

The gateway runs EPD-only (``RoutingMode::EncodePrefillDecode``) — there is no
single-worker fallback path — so a *correct* answer about the image proves the
encode->prefill->decode pipeline ran for that request: the model cannot name the
color/animal unless the encoder's embeddings reached prefill+decode. As an extra
per-request check we assert the router logged a fresh EPD encode dispatch. Qwen3.5
is a thinking model, so the answer may arrive in ``reasoning_content`` rather than
``content`` — we accept either channel.

Usage:
    pytest e2e_test/chat_completions/test_epd_multimodal.py -v

AMD SGLang functional-equivalence opt-in (Python gRPC servicer):
    E2E_AMD_SGLANG_EPD_EQUIVALENT=1 E2E_ENGINE=sglang pytest <same module>
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import pytest
from infra.pd_logs import LOG_FLUSH_TIMEOUT_S, read_logs

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"  # Black labrador puppy (checked in)
PUG_IMAGE_PATH = FIXTURES_DIR / "pug.jpg"  # Pug in a blanket (checked in)

# Solid 32x32 RGB PNGs (from docs/guides/epd-ts-test.md). A solid color is the
# most deterministic vision check: the model can only name it if the encoder's
# pixels actually reached the LM through prefill+decode.
RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/"
    "un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="
)
BLUE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJklEQVR4nO3NsQkAAAjAsP7/"
    "tF7hIASyp5pjAoFAIBAIBAKB4EmwOkv8Lm7+zY4AAAAASUVORK5CYII="
)

_LOG_DIR = Path(tempfile.mkdtemp(prefix="smg-e2e-epd-"))
# Per-request, router-side proof that the gateway routed through the encode stage.
# The worker-side "EPD encode: accepted" INFO line does not survive TokenSpeed's
# logging reconfiguration; this router line does, and the gateway runs at
# log_level=debug (below).
EPD_DISPATCH_MARKER = "EPD encode dispatch issued"

_EPD_ENGINES = ("tokenspeed",)
if os.environ.get("E2E_AMD_SGLANG_EPD_EQUIVALENT") == "1":
    _EPD_ENGINES += ("sglang",)

# (encode, prefill, decode) worker counts. Every worker is tp=1, so 1e1p1d uses
# 3 GPUs and the rest use 4 — all fit the 4-GPU runner. Counts ride in the param
# because setup_backend is class-scoped and can't read per-param marks.
_EPD_TOPOLOGIES = [
    pytest.param(("epd_grpc", (1, 1, 1)), id="1e1p1d"),
    pytest.param(("epd_grpc", (1, 2, 1)), id="1e2p1d"),
    pytest.param(("epd_grpc", (2, 1, 1)), id="2e1p1d"),
    pytest.param(("epd_grpc", (1, 1, 2)), id="1e1p2d"),
]


def _file_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _b64_png_to_data_url(data_b64: str) -> str:
    return f"data:image/png;base64,{data_b64}"


def _epd_dispatch_count() -> int:
    """How many EPD encode dispatches the router has logged so far (cumulative)."""
    return read_logs(_LOG_DIR, "smg*").count(EPD_DISPATCH_MARKER)


@pytest.mark.engine(*_EPD_ENGINES)
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3.5-9B")
@pytest.mark.gateway(log_level="debug", policy="cache_aware", log_dir=str(_LOG_DIR))
@pytest.mark.parametrize("setup_backend", _EPD_TOPOLOGIES, indirect=True)
class TestEPDMultimodal:
    """Verify the image really flows encode -> prefill -> decode for each topology."""

    def _check_image(self, client, model, image_url, question, keywords):
        """Send one image request; assert a correct answer + a fresh EPD dispatch."""
        # Baseline BEFORE the request: the router log accumulates across topologies
        # and is never cleared, so assert THIS request added a new dispatch.
        dispatches_before = _epd_dispatch_count()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
        )

        # (1) The EPD pipeline produced output. A thinking model may put the answer
        # in reasoning_content instead of content — accept either. Dump the whole
        # message on failure so we can see where (if anywhere) the answer landed.
        msg = response.choices[0].message.model_dump()
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        text = f"{content}\n{reasoning}".strip()
        assert text, f"EPD pipeline returned empty content AND reasoning; message={msg}"

        # (2) The answer is correct about the image — only possible if the encoder's
        # embeddings reached prefill+decode (EPD-only gateway; no fallback path).
        assert any(k in text.lower() for k in keywords), (
            f"expected one of {keywords} in the answer, got: {text!r}"
        )

        # (3) The router logged a fresh EPD encode dispatch for THIS request.
        deadline = time.monotonic() + LOG_FLUSH_TIMEOUT_S
        while _epd_dispatch_count() <= dispatches_before and time.monotonic() < deadline:
            time.sleep(0.5)
        assert _epd_dispatch_count() > dispatches_before, (
            "router logged no new EPD encode dispatch for this request; the gateway "
            f"did not route through encode->prefill->decode (checked {_LOG_DIR}/smg*)"
        )
        logger.info("EPD OK (%s): %s", keywords[0], text[:120])

    def test_color_images(self, model, setup_backend):
        """Solid red then blue — the most deterministic encode->decode check."""
        _, _, client, *_ = setup_backend
        question = "What color is this image? Reply with just the color."
        self._check_image(client, model, _b64_png_to_data_url(RED_PNG_B64), question, ["red"])
        self._check_image(client, model, _b64_png_to_data_url(BLUE_PNG_B64), question, ["blue"])

    def test_animal_images(self, model, setup_backend):
        """Real photos — dog then pug."""
        _, _, client, *_ = setup_backend
        question = "What animal is in this image?"
        self._check_image(
            client, model, _file_to_data_url(DOG_IMAGE_PATH), question, ["dog", "puppy", "labrador"]
        )
        self._check_image(
            client, model, _file_to_data_url(PUG_IMAGE_PATH), question, ["pug", "dog", "puppy"]
        )

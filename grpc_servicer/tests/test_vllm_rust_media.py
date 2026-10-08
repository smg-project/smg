"""The Rust servicer's media bridge: Rust-driven callbacks over the Python
processors, with the processor, vLLM's input processor and its encoder faked."""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
pytest.importorskip("msgspec")  # the processor module pulls in the sidecar protocol
pytest.importorskip("numpy")  # lend_buffer's views
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_servicer.vllm.media_refs import MediaRefItem  # noqa: E402
from smg_grpc_servicer.vllm.mm_processor import MmProcessorUnavailable, MmSettings  # noqa: E402
from smg_grpc_servicer.vllm.rust_media import RustMediaBridge  # noqa: E402


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="bridge-loop", daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class FakeProcessor:
    name = "fake"
    schemes = "http,https,data"
    accepted_schemes = {"http", "https", "data"}
    max_inflight = 3

    def __init__(self, outcome=None, probe=True):
        self.outcome = outcome
        self.probe_result = probe
        self.calls = []

    async def probe(self):
        if isinstance(self.probe_result, Exception):
            raise self.probe_result
        return self.probe_result

    async def process(self, prompt_token_ids, prompt_text, items, arrival_time, *, request_id=""):
        self.calls.append(
            (request_id, list(prompt_token_ids), prompt_text, list(items), arrival_time)
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        # The renderer's engine input: expanded ids, hashes, placeholders.
        return {
            "type": "multimodal",
            "prompt_token_ids": [1, 2, 2, 2, 3],
            "mm_hashes": {"image": ["h0"]},
            "mm_placeholders": {"image": [SimpleNamespace(offset=1, length=3, is_embed=None)]},
            "mm_kwargs": {"image": []},
        }


class FakeInputProcessor:
    def __init__(self):
        self.calls = []

    def process_inputs(self, request_id, engine_input, params, supported_tasks, arrival_time=None):
        self.calls.append((request_id, engine_input, params, supported_tasks, arrival_time))
        return SimpleNamespace(
            prompt_token_ids=[1, 2, 2, 2, 3], mm_features=["one item"], cache_salt="salt"
        )


class FakeEncoder:
    def encode(self, obj):
        assert obj == ["one item"]
        return [b"primary", memoryview(b"aux-1"), bytearray(b"aux-2")]


def bridge(loop, processor=None, input_processor=None):
    return RustMediaBridge(
        processor or FakeProcessor(),
        input_processor or FakeInputProcessor(),
        FakeEncoder(),
        loop=loop,
        source="flag",
        sampling_params=lambda: "params",
    )


def submit(b, *, want_identity=False, items=(("image", "https://x/y.png"),), request_id="r1"):
    answers: queue.Queue = queue.Queue()
    b.submit(
        request_id,
        [1, 2, 3],
        "describe",
        list(items),
        123.5,
        want_identity,
        lambda kind, payload: answers.put((kind, payload)),
    )
    return answers.get(timeout=5)


def test_the_bridge_advertises_the_processor(loop):
    b = bridge(loop)
    assert (b.name, b.schemes, b.source, b.max_inflight) == ("fake", "http,https,data", "flag", 3)


def test_submit_relays_what_the_input_processor_produced(loop):
    processor, input_processor = FakeProcessor(), FakeInputProcessor()
    kind, payload = submit(bridge(loop, processor, input_processor))
    assert kind == "ok"
    prompt_token_ids, mm_features, aux_frames, cache_salt, identity = payload
    assert prompt_token_ids == [1, 2, 2, 2, 3]
    # The primary as bytes; the tensor frames lent as read-only numpy views.
    assert mm_features == b"primary"
    assert [bytes(view) for view in aux_frames] == [b"aux-1", b"aux-2"]
    for view in aux_frames:
        assert view.flags.c_contiguous and not view.flags.writeable and view.nbytes == 5
    assert cache_salt == "salt"
    assert identity is None
    # The processor saw the request as the Router sent it.
    request_id, ids, text, items, arrival = processor.calls[0]
    assert (request_id, ids, text, arrival) == ("r1", [1, 2, 3], "describe", 123.5)
    assert items == [MediaRefItem(modality="image", url="https://x/y.png")]
    # The input processor ran over the renderer's output, as a generate task.
    request_id, engine_input, params, tasks, arrival = input_processor.calls[0]
    assert (request_id, params, tasks, arrival) == ("r1", "params", ("generate",), 123.5)
    assert engine_input["mm_hashes"] == {"image": ["h0"]}


def test_a_pd_prefill_leg_gets_the_media_identity(loop):
    pytest.importorskip("torch")  # the identity module builds grid tensors with it
    kind, payload = submit(bridge(loop), want_identity=True)
    assert kind == "ok"
    identity = vllm_engine_pb2.MediaIdentity.FromString(payload[4])
    assert list(identity.prompt_token_ids) == [1, 2, 2, 2, 3]
    assert list(identity.mm_inputs.mm_hashes) == ["h0"]
    assert identity.mm_inputs.mm_placeholders[0].offset == 1


def test_failures_are_classified_like_the_python_servicers_statuses(loop):
    kind, message = submit(bridge(loop, FakeProcessor(MmProcessorUnavailable("sidecar_timeout"))))
    assert (kind, message) == ("unavailable", "sidecar_timeout")
    kind, message = submit(bridge(loop, FakeProcessor(ValueError("bad image"))))
    assert (kind, message) == ("invalid", "bad image")
    kind, message = submit(bridge(loop, FakeProcessor(RuntimeError("boom"))))
    assert (kind, message) == ("internal", "boom")
    # A scheme the processor does not accept is refused before it runs.
    processor = FakeProcessor()
    kind, message = submit(bridge(loop, processor), items=(("image", "ftp://x/y.png"),))
    assert kind == "invalid" and "ftp" in message
    assert processor.calls == []


def test_probe_reports_the_processors_answer_and_current_schemes(loop):
    for probe, expected in [(True, True), (False, False), (RuntimeError("down"), False)]:
        answers: queue.Queue = queue.Queue()
        processor = FakeProcessor(probe=probe)
        b = bridge(loop, processor)
        # A sidecar announces its schemes on probe; the bridge reports them live.
        processor.schemes = "http,https,data,file"
        b.probe(lambda serving, schemes: answers.put((serving, schemes)))
        assert answers.get(timeout=5) == (expected, "http,https,data,file")
        assert b.schemes == "http,https,data,file"


def test_start_warmup_uses_the_renderers_background_warmup(loop):
    class Renderer:
        started = 0

        def start_mm_warmup_in_background(self):
            self.started += 1

    class InProcess(FakeProcessor):
        name = "inprocess"

    def with_renderer(processor, renderer):
        return RustMediaBridge(
            processor,
            FakeInputProcessor(),
            FakeEncoder(),
            loop=loop,
            source="flag",
            sampling_params=lambda: "params",
            renderer=renderer,
        )

    renderer = Renderer()
    with_renderer(InProcess(), renderer).start_warmup()
    assert renderer.started == 1
    # The sidecar runs the processor: nothing in this process to warm up.
    renderer = Renderer()
    with_renderer(FakeProcessor(), renderer).start_warmup()
    assert renderer.started == 0
    # A renderer without the hook (an older vLLM) is left alone.
    with_renderer(InProcess(), object()).start_warmup()


def test_build_is_off_without_a_mode_or_a_multimodal_model(loop):
    config = SimpleNamespace(model_config=SimpleNamespace(is_multimodal_model=True))
    assert RustMediaBridge.build(config, MmSettings(processor="off"), loop) is None
    text_only = SimpleNamespace(model_config=SimpleNamespace(is_multimodal_model=False))
    assert RustMediaBridge.build(text_only, MmSettings(processor="inprocess"), loop) is None

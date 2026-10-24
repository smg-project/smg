"""The Rust servicer's encode dispatch bridge for a language-only SGLang
worker (``smg_grpc_servicer.sglang.rust_encode``): SGLang's own receiver and
encoder bootstrap server, faked here, driven as the Rust server drives them."""

from __future__ import annotations

import sys
import threading
import types
from enum import Enum, auto
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import rust_encode


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        monkeypatch.setattr(parent_module, child, module, raising=False)
    return module


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


class FakeBootstrapServer:
    instances: list = []

    def __init__(self, host, port, urls=None):
        self.host, self.port, self.urls = host, port, urls
        self.closed = False
        FakeBootstrapServer.instances.append(self)

    def close(self):
        self.closed = True


class FakeReceiver:
    """``MMReceiverBase.send_encode_request`` as SGLang 0.5.21/0.5.22 define
    it: the assignment onto the request, the URL snapshot, the dispatch thread
    that may report a failure through ``on_dispatch_error`` once the request
    is released, or the wait flag off when no encoder is registered."""

    def __init__(self, encode_urls, fail_with=None):
        self.encode_urls = encode_urls
        self.fail_with = fail_with
        self.calls = []
        self.threads = []

    def send_encode_request(self, obj, time_stats_json=None, on_dispatch_error=None):
        self.calls.append(obj)
        urls = list(self.encode_urls)
        if not obj.image_data or not urls:
            obj.need_wait_for_mm_inputs = False
            return None
        obj.need_wait_for_mm_inputs = True
        obj.num_items_assigned = {Modality.IMAGE: [len(obj.image_data), 0][: len(urls)]}
        obj.encoder_urls = urls
        ready = threading.Event()
        if self.fail_with is not None:

            def report():
                ready.wait()
                on_dispatch_error(self.fail_with)

            thread = threading.Thread(target=report, daemon=True)
            thread.start()
            self.threads.append(thread)
        return ready


@pytest.fixture
def sglang(monkeypatch):
    """The SGLang modules the bridge imports, faked: the receiver factory
    records what it was built with."""
    FakeBootstrapServer.instances = []
    built = {}

    def create_mm_receiver(server_args, encode_urls=None, **_kwargs):
        built["server_args"] = server_args
        built["encode_urls"] = encode_urls
        return built["receiver"]

    _install(
        monkeypatch,
        "sglang.srt.disaggregation.encoder.receiver",
        EncoderBootstrapServer=FakeBootstrapServer,
        create_mm_receiver=create_mm_receiver,
    )
    published = []
    _install(
        monkeypatch,
        "sglang.srt.runtime_context",
        publish=lambda args, role: published.append(role),
        publish_role=lambda: published[-1] if published else None,
    )

    class GenerateReqInput:
        def __init__(self, rid=None, image_data=None):
            self.rid = rid
            self.image_data = image_data
            self.need_wait_for_mm_inputs = None
            self.num_items_assigned = None
            self.encoder_urls = None

    _install(monkeypatch, "sglang.srt.managers.io_struct", GenerateReqInput=GenerateReqInput)
    built["published"] = published
    return built


def _args(**overrides):
    args = dict(
        language_only=True,
        encoder_transfer_backend="zmq_to_scheduler",
        encoder_urls=["grpc://10.0.0.8:30001", "grpc://10.0.0.9:30001"],
        encoder_bootstrap_port=8997,
        host="127.0.0.1",
        tp_size=1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_the_bridge_starts_sglangs_own_objects_after_the_runtime_context(sglang):
    """Start builds the bootstrap server on the launcher's host and bootstrap
    port over the shared URL list (pre-filled with ``--encoder-urls``) and the
    receiver over that same list, publishing the runtime context first when
    nothing has; a second start is a no-op; shutdown closes the server."""
    sglang["receiver"] = FakeReceiver(encode_urls=[])
    bridge = rust_encode.EncodeDispatchBridge(_args())
    assert not bridge.started
    bridge.start()
    bridge.start()
    assert bridge.started
    assert sglang["published"] == ["tokenizer"]
    (server,) = FakeBootstrapServer.instances
    assert (server.host, server.port) == ("127.0.0.1", 8997)
    assert server.urls is bridge.encoder_urls is sglang["encode_urls"]
    assert bridge.encoder_urls == ["grpc://10.0.0.8:30001", "grpc://10.0.0.9:30001"]
    bridge.shutdown()
    assert server.closed and not bridge.started


def test_the_bridge_does_not_republish_a_context_the_launch_published(sglang):
    sglang["published"].append("tokenizer")
    sglang["receiver"] = FakeReceiver(encode_urls=[])
    rust_encode.EncodeDispatchBridge(_args()).start()
    assert sglang["published"] == ["tokenizer"]


def test_dispatch_hands_rust_what_the_receiver_set_and_the_release(sglang):
    """One request: the receiver gets a ``GenerateReqInput`` of the rid and
    the image references, and the answer is what it set (the wait flag, the
    assignment keyed by the modality's enum value, the URL snapshot) plus the
    release of its dispatch thread; a failure the thread reports after the
    release reaches ``on_error`` as (message, code)."""
    sglang["receiver"] = FakeReceiver(
        encode_urls=["grpc://10.0.0.8:30001", "grpc://10.0.0.9:30001"],
        fail_with=SimpleNamespace(rid="r1", error_msg="encoder down", error_code=502),
    )
    bridge = rust_encode.EncodeDispatchBridge(_args())
    bridge.start()
    errors = []
    need_wait, assigned, urls, release = bridge.dispatch(
        "r1",
        ["data:image/png;base64,AA==", "https://example.test/b.png"],
        lambda message, code: errors.append((message, code)),
    )
    (request,) = sglang["receiver"].calls
    assert request.rid == "r1"
    assert request.image_data == ["data:image/png;base64,AA==", "https://example.test/b.png"]
    assert need_wait is True
    assert assigned == {1: [2, 0]}
    assert urls == ["grpc://10.0.0.8:30001", "grpc://10.0.0.9:30001"]
    assert urls is not bridge.encoder_urls, "a snapshot, not the live list"
    assert errors == [], "nothing is reported before the release"
    release()
    sglang["receiver"].threads[0].join(timeout=5)
    assert errors == [("encoder down", 502)]


def test_dispatch_releases_the_receivers_thread_when_the_answer_does_not_build(sglang):
    """A failure after ``send_encode_request`` (here an assignment keyed by a
    modality whose value is not a number) re-raises with the release run, so
    the receiver's thread, parked on it since the send, reports instead of
    waiting forever."""

    class Drifted(Enum):
        IMAGE = "image"

    class DriftedReceiver(FakeReceiver):
        def send_encode_request(self, obj, time_stats_json=None, on_dispatch_error=None):
            ready = super().send_encode_request(obj, time_stats_json, on_dispatch_error)
            obj.num_items_assigned = {Drifted.IMAGE: [len(obj.image_data)]}
            return ready

    sglang["receiver"] = DriftedReceiver(
        encode_urls=["grpc://10.0.0.8:30001"],
        fail_with=SimpleNamespace(rid="r3", error_msg="encoder down", error_code=502),
    )
    bridge = rust_encode.EncodeDispatchBridge(_args())
    bridge.start()
    errors = []
    with pytest.raises(ValueError):
        bridge.dispatch(
            "r3",
            ["https://example.test/a.png"],
            lambda message, code: errors.append((message, code)),
        )
    sglang["receiver"].threads[0].join(timeout=5)
    assert errors == [("encoder down", 502)], "the thread passed the release"


def test_dispatch_with_no_encoder_registered_leaves_the_wait_flag_off(sglang):
    """The receiver's own rule: no encoder, no dispatch, the flag off (its
    warning says why); nothing to release."""
    sglang["receiver"] = FakeReceiver(encode_urls=[])
    bridge = rust_encode.EncodeDispatchBridge(_args(encoder_urls=[]))
    bridge.start()
    assert bridge.dispatch("r2", ["https://example.test/a.png"], lambda *_: None) == (
        False,
        None,
        None,
        None,
    )


def test_dispatch_before_start_is_an_error(sglang):
    with pytest.raises(RuntimeError, match="not started"):
        rust_encode.EncodeDispatchBridge(_args()).dispatch("r", ["u"], lambda *_: None)


def test_the_transfer_backend_resolves_as_sglang_resolves_it(monkeypatch):
    """An explicit backend is taken as given; ``auto`` goes through SGLang's
    own resolver (model- and TP-aware) when it is importable, else to the
    backend that resolver picks for every model but one."""
    assert (
        rust_encode.resolve_encoder_transfer_backend(
            SimpleNamespace(encoder_transfer_backend="mooncake")
        )
        == "mooncake"
    )
    # An engine release without the resolver: the module is there, the name
    # is not (poisoning ``sys.modules["sglang"]`` instead would break the
    # real package's own imports where it is installed).
    _install(monkeypatch, "sglang.srt.server_args")
    assert (
        rust_encode.resolve_encoder_transfer_backend(
            SimpleNamespace(encoder_transfer_backend="auto")
        )
        == "zmq_to_scheduler"
    )
    seen = {}

    def resolve(backend, model_arch, tp_size):
        seen.update(backend=backend, model_arch=model_arch, tp_size=tp_size)
        return "zmq_to_tokenizer"

    _install(monkeypatch, "sglang.srt.server_args", resolve_encoder_transfer_backend=resolve)
    monkeypatch.setattr(
        "smg_grpc_servicer.sglang.rust._model_config",
        lambda args: SimpleNamespace(hf_config=SimpleNamespace(architectures=["KimiK3"])),
    )
    assert (
        rust_encode.resolve_encoder_transfer_backend(
            SimpleNamespace(encoder_transfer_backend="auto", tp_size=4)
        )
        == "zmq_to_tokenizer"
    )
    assert seen == {"backend": "auto", "model_arch": "KimiK3", "tp_size": 4}


def test_only_a_language_only_launch_on_a_scheduler_receive_backend_dispatches():
    assert rust_encode.dispatches_from_the_frontend(_args())
    assert rust_encode.dispatches_from_the_frontend(_args(encoder_transfer_backend="mooncake"))
    assert not rust_encode.dispatches_from_the_frontend(_args(language_only=False))
    assert not rust_encode.dispatches_from_the_frontend(
        _args(encoder_transfer_backend="zmq_to_tokenizer")
    )

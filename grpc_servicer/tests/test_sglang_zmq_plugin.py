"""The SGLang plugin's hooks, driven with fakes (no SGLang needed)."""

import dataclasses
from types import SimpleNamespace

import pytest

# The plugin's wire module imports pyzmq and msgspec (SGLang's own deps). The
# plain unit-test job installs msgspec but not pyzmq and skips this module;
# the SGLang ZMQ e2e lane runs it.
pytest.importorskip("msgspec")
pytest.importorskip("zmq")
from smg_grpc_servicer.sglang import zmq_plugin  # noqa: E402
from smg_grpc_servicer.sglang.zmq_msgpack import MsgpackRecvSocket, MsgpackSendSocket  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(zmq_plugin, "_state", zmq_plugin._State())
    monkeypatch.delenv(zmq_plugin.HANDSHAKE_ENV, raising=False)
    monkeypatch.delenv(zmq_plugin.ENGINE_INDEX_ENV, raising=False)
    monkeypatch.delenv(zmq_plugin.GRAMMAR_ATTRIBUTION_ENV, raising=False)


def test_register_is_inert_without_the_launcher_env():
    # No import of sglang, no hooks: a plain SGLang install stays untouched.
    assert zmq_plugin.active() is False
    assert zmq_plugin.attributes_grammar_aborts() is False
    zmq_plugin.register()


def test_grammar_attribution_registers_under_either_env(monkeypatch):
    # The handshake implies it (the ZMQ wire carries the name); the Python
    # servicer asks for it alone, for the finish dicts it reads.
    monkeypatch.setenv(zmq_plugin.HANDSHAKE_ENV, "tcp://127.0.0.1:1")
    assert zmq_plugin.attributes_grammar_aborts() is True
    monkeypatch.delenv(zmq_plugin.HANDSHAKE_ENV)
    monkeypatch.setenv(zmq_plugin.GRAMMAR_ATTRIBUTION_ENV, "1")
    assert zmq_plugin.active() is False
    assert zmq_plugin.attributes_grammar_aborts() is True
    monkeypatch.setenv(zmq_plugin.GRAMMAR_ATTRIBUTION_ENV, "0")
    assert zmq_plugin.attributes_grammar_aborts() is False


class _Finish:
    """A stand-in for the scheduler's ``FINISH_ABORT``: a message, a status
    and the ``err_type`` its finish dict carries."""

    def __init__(self, message="Aborted", status_code=None, err_type=None):
        self.message = message
        self.status_code = status_code
        self.err_type = err_type


class _Grammar:
    def __init__(self, rejects: bool):
        self.rejects = rejects
        self.finished = False

    def accept_token(self, token):
        if self.rejects:
            raise ValueError(f"Tokens not accepted: {token}")

    def is_terminated(self):
        return False


class _Req:
    def __init__(self, rid, grammar=None, to_finish=None):
        self.rid = rid
        self.grammar = grammar
        self.to_finish = to_finish

    def finished(self):
        return self.to_finish is not None


def _accept_as_the_scheduler_does(processor, req, tokens):
    """The shape of the scheduler's own ``_accept_grammar_tokens``: a
    rejection becomes a bare ``FINISH_ABORT()`` on the request."""
    try:
        req.grammar.accept_token(tokens)
    except ValueError:
        req.to_finish = _Finish()
    return []


def test_the_grammar_hook_names_only_the_finish_the_advance_set():
    # A rejection: the bare abort gains the name and the message.
    req = _Req("r1", _Grammar(rejects=True))
    zmq_plugin.attribute_grammar_abort(_accept_as_the_scheduler_does, object(), req, [7])
    assert req.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    assert req.to_finish.message == zmq_plugin.GRAMMAR_REJECTED_MESSAGE
    assert req.to_finish.status_code is None
    # An accepted token: nothing to name.
    req = _Req("r2", _Grammar(rejects=False))
    zmq_plugin.attribute_grammar_abort(_accept_as_the_scheduler_does, object(), req, [7])
    assert req.to_finish is None
    # A finish set before the advance (a length cap, an abort-all) is not the
    # grammar's, even when the advance leaves it in place.
    earlier = _Finish("Aborted")
    req = _Req("r3", _Grammar(rejects=False), to_finish=earlier)
    zmq_plugin.attribute_grammar_abort(_accept_as_the_scheduler_does, object(), req, [7])
    assert req.to_finish is earlier and earlier.err_type is None
    # The keyword form of the prefill path.
    req = _Req("r4", _Grammar(rejects=True))

    def apply_prefill(processor, *, req, next_token_id, already_advanced=False):
        return _accept_as_the_scheduler_does(processor, req, next_token_id)

    zmq_plugin.attribute_grammar_abort(apply_prefill, object(), req=req, next_token_id=9)
    assert req.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    # A path that names the token itself keeps its own words and its status.
    req = _Req("r5", _Grammar(rejects=True))

    def decode_side(processor, req, tokens):
        req.to_finish = _Finish("Grammar accept_token failed for req r5 with token 7", 500)

    zmq_plugin.attribute_grammar_abort(decode_side, object(), req, [7])
    assert req.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    assert req.to_finish.message == "Grammar accept_token failed for req r5 with token 7"
    assert req.to_finish.status_code == 500


def test_the_prebuilt_hook_names_the_grammar_requests_the_step_aborted():
    rejected = _Req("a", _Grammar(rejects=True))
    accepted = _Req("b", _Grammar(rejects=False))
    plain = _Req("c")
    batch = SimpleNamespace(reqs=[rejected, accepted, plain], has_grammar=True)

    def process_prebuilt(batch, future_map):
        for req in batch.reqs:
            if req.grammar is not None:
                try:
                    req.grammar.accept_token(1)
                except ValueError:
                    req.to_finish = _Finish("Grammar accept_token failed for req a", 500)
        # The step aborts a request without a grammar for its own reasons.
        batch.reqs[2].to_finish = _Finish("Aborted")
        return "prebuilt"

    assert zmq_plugin.attribute_prebuilt_grammar_aborts(process_prebuilt, batch, None) == "prebuilt"
    assert rejected.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    assert rejected.to_finish.status_code == 500
    assert accepted.to_finish is None
    assert plain.to_finish.err_type is None, "no grammar: not the grammar's abort"
    # A batch without a grammar is not walked.
    untouched = _Req("d")
    batch = SimpleNamespace(reqs=[untouched], has_grammar=False)

    def abort_all(batch, future_map):
        batch.reqs[0].to_finish = _Finish("Aborted")

    zmq_plugin.attribute_prebuilt_grammar_aborts(abort_all, batch, None)
    assert untouched.to_finish.err_type is None


def test_grammar_hooks_follow_the_registry_calling_convention(monkeypatch):
    """AROUND on the three grammar paths, against the real registry on
    stand-in classes: the positional, the keyword-only and the batch form."""
    pytest.importorskip("sglang")
    import sys
    import types

    from sglang.srt.plugins.hook_registry import HookRegistry

    class SchedulerBatchResultProcessor:
        def _accept_grammar_tokens(self, req, tokens):
            return _accept_as_the_scheduler_does(self, req, tokens)

        def _apply_prefill_grammar(self, *, req, next_token_id, already_advanced=False):
            if not already_advanced:
                _accept_as_the_scheduler_does(self, req, next_token_id)
            req.grammar.finished = req.finished()

    class ScheduleBatchDisaggregationDecodeMixin:
        def process_prebuilt(self, future_map):
            for req in self.reqs:
                if req.grammar is not None:
                    _accept_as_the_scheduler_does(self, req, 1)

    class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
        def __init__(self, reqs):
            self.reqs = reqs
            self.has_grammar = any(req.grammar for req in reqs)

    mod = types.ModuleType("smg_zmq_plugin_grammar_contract")
    mod.SchedulerBatchResultProcessor = SchedulerBatchResultProcessor
    mod.ScheduleBatchDisaggregationDecodeMixin = ScheduleBatchDisaggregationDecodeMixin
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(
        zmq_plugin, "_BATCH_RESULTS", f"{mod.__name__}.SchedulerBatchResultProcessor"
    )
    monkeypatch.setattr(
        zmq_plugin, "_DECODE_BATCH", f"{mod.__name__}.ScheduleBatchDisaggregationDecodeMixin"
    )
    monkeypatch.setenv(zmq_plugin.GRAMMAR_ATTRIBUTION_ENV, "1")
    HookRegistry.reset()
    try:
        zmq_plugin.register()
        HookRegistry.apply_hooks()
        processor = SchedulerBatchResultProcessor()
        spec = _Req("spec", _Grammar(rejects=True))
        assert processor._accept_grammar_tokens(spec, [1, 2]) == []
        assert spec.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
        prefill = _Req("prefill", _Grammar(rejects=True))
        processor._apply_prefill_grammar(req=prefill, next_token_id=3)
        assert prefill.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
        assert prefill.grammar.finished is True
        fine = _Req("fine", _Grammar(rejects=False))
        processor._apply_prefill_grammar(req=fine, next_token_id=3, already_advanced=True)
        assert fine.to_finish is None
        decode = _Req("decode", _Grammar(rejects=True))
        ScheduleBatch([decode, _Req("plain")]).process_prebuilt(None)
        assert decode.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    finally:
        HookRegistry.reset()


def test_this_releases_grammar_paths_are_named():
    """The hooks against the release's own result processor: its
    ``_accept_grammar_tokens`` and ``_apply_prefill_grammar`` set the bare
    ``FINISH_ABORT()`` when the grammar raises, and the hook names it. (The
    PD decode path needs a device batch; its hook is covered above.)"""
    pytest.importorskip("sglang")
    import pkgutil

    for target in (
        f"{zmq_plugin._BATCH_RESULTS}._accept_grammar_tokens",
        f"{zmq_plugin._BATCH_RESULTS}._apply_prefill_grammar",
        f"{zmq_plugin._DECODE_BATCH}.process_prebuilt",
    ):
        assert zmq_plugin._resolves(target), target
    assert not zmq_plugin._resolves(f"{zmq_plugin._BATCH_RESULTS}._no_such_grammar_path")
    assert not zmq_plugin._resolves("smg_no_such_module.NoClass.method")

    processor_cls = pkgutil.resolve_name(zmq_plugin._BATCH_RESULTS)
    processor = processor_cls.__new__(processor_cls)  # the paths read nothing of self
    spec = _Req("spec", _Grammar(rejects=True))
    retained = zmq_plugin.attribute_grammar_abort(
        processor_cls._accept_grammar_tokens, processor, spec, [11, 12]
    )
    assert retained == []
    assert type(spec.to_finish).__name__ == "FINISH_ABORT"
    assert spec.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    assert spec.to_finish.message == zmq_plugin.GRAMMAR_REJECTED_MESSAGE
    assert spec.to_finish.status_code is None
    assert spec.to_finish.to_json() == {
        "type": "abort",
        "message": zmq_plugin.GRAMMAR_REJECTED_MESSAGE,
        "status_code": None,
        "err_type": zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE,
    }
    prefill = _Req("prefill", _Grammar(rejects=True))
    zmq_plugin.attribute_grammar_abort(
        processor_cls._apply_prefill_grammar, processor, req=prefill, next_token_id=3
    )
    assert prefill.to_finish.err_type == zmq_plugin.GRAMMAR_REJECTED_ERR_TYPE
    accepted = _Req("accepted", _Grammar(rejects=False))
    zmq_plugin.attribute_grammar_abort(
        processor_cls._accept_grammar_tokens, processor, accepted, [11]
    )
    assert accepted.to_finish is None


def test_engine_index_is_validated(monkeypatch):
    monkeypatch.setenv(zmq_plugin.ENGINE_INDEX_ENV, "70000")
    with pytest.raises(ValueError, match="0..=65535"):
        zmq_plugin.engine_index()
    monkeypatch.setenv(zmq_plugin.ENGINE_INDEX_ENV, "3")
    assert zmq_plugin.engine_index() == 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Channels:
    recv_from_tokenizer: object
    recv_from_rpc: object
    send_to_tokenizer: object
    send_to_detokenizer: object


def _original_create(cls, *, is_rank_zero, **_kwargs):
    if is_rank_zero:
        raise AssertionError("the plugin must not open tokenizer-manager sockets on rank zero")
    return _Channels(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        send_to_tokenizer="null-sender",
        send_to_detokenizer="null-sender",
    )


def test_rank_zero_channels_carry_one_deferred_sender_and_other_ranks_are_untouched():
    other = zmq_plugin.create_channels(_original_create, object, is_rank_zero=False)
    assert other.send_to_tokenizer == "null-sender" and zmq_plugin._state.sender is None

    channels = zmq_plugin.create_channels(_original_create, object, is_rank_zero=True)
    sender = zmq_plugin._state.sender
    assert isinstance(sender, MsgpackSendSocket)
    assert channels.send_to_tokenizer is sender and channels.send_to_detokenizer is sender
    assert channels.recv_from_tokenizer is None and channels.recv_from_rpc is None
    assert zmq_plugin._state.context is not None


def test_idle_sleeper_waits_for_the_handshake_only_on_the_io_rank():
    scheduler = SimpleNamespace(idle_sleeper="unset")
    calls = []
    zmq_plugin.init_idle_sleeper(lambda s: calls.append(s), scheduler)
    assert calls == [scheduler] and scheduler.idle_sleeper == "unset"

    zmq_plugin._state.sender = MsgpackSendSocket()
    zmq_plugin.init_idle_sleeper(lambda s: calls.append("not called"), scheduler)
    assert scheduler.idle_sleeper is None and calls == [scheduler]


def test_pull_raw_reqs_drains_the_msgpack_source_and_defers_otherwise():
    class Source(MsgpackRecvSocket):
        def __init__(self):  # no socket needed to test dispatch
            self.drained = []

        def drain(self, max_recv):
            self.drained.append(max_recv)
            return ["req"]

    source = Source()
    receiver = SimpleNamespace(recv_from_tokenizer=source, max_recv_per_poll=8)
    assert zmq_plugin.pull_raw_reqs(lambda r: ["original"], receiver) == ["req"]
    assert source.drained == [8]
    receiver = SimpleNamespace(recv_from_tokenizer=None, max_recv_per_poll=8)
    assert zmq_plugin.pull_raw_reqs(lambda r: ["original"], receiver) == ["original"]


def test_run_event_loop_announces_the_rank_is_gone():
    sent = []
    sender = MsgpackSendSocket()
    sender.send_engine_dead = lambda: sent.append("dead")  # type: ignore[method-assign]
    zmq_plugin._state.sender = sender

    def loop(_scheduler):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        zmq_plugin.run_event_loop(loop, object())
    assert sent == ["dead"]


def test_split_args_keeps_the_scheduler_tokenizer_and_validates():
    from smg_grpc_servicer.sglang.headless import split_args

    ours, rest = split_args(
        ["--zmq-handshake-address", "tcp://127.0.0.1:21000", "--model-path", "m"]
    )
    assert ours.zmq_handshake_address == "tcp://127.0.0.1:21000" and ours.zmq_engine_index == 0
    # SGLang's grammar backend only exists when the scheduler keeps its
    # tokenizer, so the launcher must not force --skip-tokenizer-init.
    assert rest == ["--model-path", "m"]
    with pytest.raises(SystemExit):
        split_args(["--zmq-handshake-address", "ipc:///x", "--model-path", "m"])


def test_hooks_follow_the_registry_calling_convention(monkeypatch):
    """The hooks are written for the wrappers SGLang's HookRegistry installs
    (AROUND on a classmethod gets ``(original, cls, **kwargs)``, AROUND on a
    method ``(original, self, ...)``, AFTER ``(result, self)``); pin that
    against the real registry on stand-in classes."""
    pytest.importorskip("sglang")
    import sys
    import types

    from sglang.srt.plugins.hook_registry import HookRegistry

    class Source(MsgpackRecvSocket):
        def __init__(self):  # no socket: dispatch only
            pass

        def drain(self, max_recv):
            return [f"drained:{max_recv}"]

    class Channels:
        @classmethod
        def create(cls, *, is_rank_zero, **_kwargs):
            assert is_rank_zero is False, "the plugin asks for non-rank-zero channels"
            return _Channels(
                recv_from_tokenizer=None,
                recv_from_rpc=None,
                send_to_tokenizer="null",
                send_to_detokenizer="null",
            )

    class Receiver:
        def __init__(self, source):
            self.recv_from_tokenizer = source
            self.max_recv_per_poll = 4

        def _pull_raw_reqs(self):
            return ["original"]

    class Scheduler:
        def __init__(self):
            self.idle_sleeper = "unset"
            self.recv_from_tokenizer = None
            self.server_args = SimpleNamespace(sleep_on_idle=False)
            self.skip_tokenizer_init = True

        def init_idle_sleeper(self):
            self.idle_sleeper = "sglang"

        def maybe_init_rust_server(self):
            return "rust-result"

        def run_event_loop(self):
            return "loop-done"

    mod = types.ModuleType("smg_zmq_plugin_contract")
    mod.Channels, mod.Receiver, mod.Scheduler = Channels, Receiver, Scheduler
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(zmq_plugin, "_CHANNELS", f"{mod.__name__}.Channels")
    monkeypatch.setattr(zmq_plugin, "_RECEIVER", f"{mod.__name__}.Receiver")
    monkeypatch.setattr(zmq_plugin, "_SCHEDULER", f"{mod.__name__}.Scheduler")
    monkeypatch.setenv(zmq_plugin.HANDSHAKE_ENV, "tcp://127.0.0.1:1")
    monkeypatch.setenv(zmq_plugin.ENGINE_INDEX_ENV, "0")
    connected = []

    def fake_connect(scheduler, context, sender, address, index):
        connected.append((address, index, sender is zmq_plugin._state.sender))
        return Source()

    monkeypatch.setattr(zmq_plugin, "connect_scheduler", fake_connect)
    HookRegistry.reset()
    try:
        zmq_plugin.register()
        HookRegistry.apply_hooks()

        channels = Channels.create(is_rank_zero=True)
        assert isinstance(channels.send_to_tokenizer, MsgpackSendSocket)
        scheduler = Scheduler()
        scheduler.init_idle_sleeper()
        assert scheduler.idle_sleeper is None, "deferred until the handshake"
        assert scheduler.maybe_init_rust_server() == "rust-result", "AFTER keeps the result"
        assert isinstance(scheduler.recv_from_tokenizer, Source)
        assert connected == [("tcp://127.0.0.1:1", 0, True)]
        assert Receiver(scheduler.recv_from_tokenizer)._pull_raw_reqs() == ["drained:4"]
        assert Receiver(None)._pull_raw_reqs() == ["original"]
        assert scheduler.run_event_loop() == "loop-done"
    finally:
        HookRegistry.reset()
        if zmq_plugin._state.context is not None:
            zmq_plugin._state.context.term()


def test_signal_handler_interrupts_startup_and_only_flags_afterwards():
    import signal
    import threading

    from smg_grpc_servicer.sglang.headless import Interrupted, make_signal_handler

    stop, faulted, started = threading.Event(), threading.Event(), threading.Event()
    handler = make_signal_handler(stop, faulted, started)
    with pytest.raises(Interrupted) as raised:
        handler(signal.SIGTERM, None)
    assert raised.value.signum == signal.SIGTERM and stop.is_set() and not faulted.is_set()
    started.set()
    handler(signal.SIGQUIT, None)  # after startup: flag the fault, let the loop exit
    assert faulted.is_set()


def test_signal_handler_raises_only_on_the_first_signal():
    """A second signal during an interrupted startup (an impatient Ctrl-C, a
    second rank's SIGQUIT) must not unwind the cleanup already under way."""
    import signal
    import threading

    from smg_grpc_servicer.sglang.headless import Interrupted, make_signal_handler

    stop, faulted, started = threading.Event(), threading.Event(), threading.Event()
    handler = make_signal_handler(stop, faulted, started)
    with pytest.raises(Interrupted):
        handler(signal.SIGINT, None)
    handler(signal.SIGQUIT, None)
    assert stop.is_set() and faulted.is_set()

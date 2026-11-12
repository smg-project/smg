"""Token rows and sparse usage remain lossless across SGLang output layouts."""

from array import array
from types import SimpleNamespace

import msgspec
import pytest
from smg_grpc_servicer.sglang.compat import SchedulerOutputAdapter
from smg_grpc_servicer.sglang.zmq_msgpack import BatchTokenIDSlimOutput


class Batch(SimpleNamespace):
    def __getattr__(self, _name):
        return None


def batch(*, flat=True, sparse=True, **kwargs):
    """Build independent token and usage layouts, including an empty row."""
    out = Batch(rids=["a", "b", "c"], finished_reasons=None)
    if flat:
        out.output_ids_flat = array("q", [10, 20, 21, 22])
        out.output_ids_offsets = [0, 1, 1, 4]
    else:
        out.output_ids = [array("q", [10]), [], [20, 21, 22]]
    if sparse:
        out.request_stats = [
            SimpleNamespace(prompt_tokens=p, cached_tokens=c)
            for p, c in [(100, 20), (200, 30), (300, 40)]
        ]
    else:
        out.prompt_tokens, out.cached_tokens = [100, 200, 300], [20, 30, 40]
    vars(out).update(kwargs)
    return out


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_layouts_preserve_empty_and_speculative_rows_and_wire(flat, sparse):
    """All layout combinations preserve token rows, counts and encoded bytes."""
    out = batch(flat=flat, sparse=sparse)
    adapter = SchedulerOutputAdapter()
    rows, prompts, cached = adapter.read(out)
    assert rows == [[10], [], [20, 21, 22]]
    assert prompts == [100, 200, 300]
    assert cached == [20, 30, 40]
    result = BatchTokenIDSlimOutput.from_full(out, output_adapter=adapter)
    reference = BatchTokenIDSlimOutput.from_full(batch(flat=False, sparse=False))
    assert msgspec.msgpack.encode(result) == msgspec.msgpack.encode(reference)
    if not sparse:
        assert not adapter._counts
        assert prompts is out.prompt_tokens
        assert cached is out.cached_tokens
    if not flat:
        assert rows[2] is out.output_ids[2]


@pytest.mark.parametrize("offsets", [None, [], [0], [1, 2, 3, 4], [0, 3, 2, 4], [0, 1, 2, 5]])
def test_bad_flat_offsets_are_refused(offsets):
    """Reject corrupt offsets before slicing tokens into the wrong request."""
    with pytest.raises(ValueError, match="offsets"):
        SchedulerOutputAdapter().read(batch(output_ids_offsets=offsets))


def test_sparse_usage_survives_interleaving_retraction_and_cleans_up():
    """Retain counts by request ID and replace them after retraction."""
    adapter = SchedulerOutputAdapter()
    adapter.read(batch())
    # Stats are absent on ordinary steps, but restored after retraction.
    step = batch(
        rids=["b", "a"],
        output_ids_flat=[1, 2],
        output_ids_offsets=[0, 1, 2],
        request_stats=None,
    )
    assert adapter.read(step)[1:] == ([200, 100], [30, 20])
    step.request_stats = [None, SimpleNamespace(prompt_tokens=100, cached_tokens=15)]
    assert adapter.read(step)[1:] == ([200, 100], [30, 15])
    step.request_stats = None
    step.finished_reasons = [None, {"type": "stop"}]
    assert adapter.read(step)[1:] == ([200, 100], [30, 15])
    assert "a" not in adapter._counts
    adapter.forget("b")
    adapter.forget("c")
    assert not adapter._counts
    assert adapter.read(step)[1:] == ([0, 0], [0, 0])
    assert not adapter._counts


def test_selection_is_once_per_socket_and_state_is_not_shared():
    """Absent sparse updates do not change the selected schema or leak across sockets."""
    adapter, other = SchedulerOutputAdapter(), SchedulerOutputAdapter()
    adapter.read(batch())
    # A transient sparse None cannot switch to the legacy usage reader.
    empty_stats = batch(request_stats=None)
    assert adapter.read(empty_stats)[1:] == ([100, 200, 300], [20, 30, 40])
    assert other.read(empty_stats)[1:] == ([0, 0, 0], [0, 0, 0])
    assert adapter._rows == adapter._flat_rows
    assert adapter._usage == adapter._sparse_usage


def test_empty_batch_can_select_the_msgspec_schema():
    """Schema selection works even before the first request arrives."""

    class Output(msgspec.Struct):
        rids: list[str]
        output_ids_flat: list[int]
        output_ids_offsets: list[int]
        request_stats: list | None = None
        finished_reasons: list | None = None

    assert SchedulerOutputAdapter().read(Output([], [], [0])) == ([], [], [])


def test_invalid_legacy_rows_and_unknown_layout_fail_explicitly():
    """Reject unrecognized schemas and mismatched legacy row counts."""
    with pytest.raises(ValueError, match="row count"):
        SchedulerOutputAdapter().read(batch(flat=False, output_ids=[]))
    with pytest.raises(ValueError, match="unsupported"):
        SchedulerOutputAdapter().read(SimpleNamespace(rids=[]))


def test_modern_dense_headers_and_sparse_stats_can_coexist():
    """Sparse updates take precedence and dense native-sink headers remain readable."""
    adapter = SchedulerOutputAdapter()
    out = batch(request_stats=None, prompt_tokens=[100, 200, 300], cached_tokens=[20, 30, 40])
    assert adapter.read(out)[1:] == ([100, 200, 300], [20, 30, 40])
    out.request_stats = [None, SimpleNamespace(prompt_tokens=200, cached_tokens=15), None]
    assert adapter.read(out)[1:] == ([100, 200, 300], [20, 15, 40])
    out.request_stats = out.prompt_tokens = out.cached_tokens = None
    out.finished_reasons = [{"type": "stop"}] * 3
    assert adapter.read(out)[1:] == ([100, 200, 300], [20, 15, 40])
    assert not adapter._counts


def test_installed_engine_struct_uses_the_same_wire():
    """Exercise the installed engine structure and real socket dispatch when available."""
    import inspect

    io = pytest.importorskip("sglang.srt.managers.io_struct")
    fields = inspect.signature(io.BatchTokenIDOutput).parameters
    out = batch(flat="output_ids_flat" in fields, sparse="request_stats" in fields)
    if "request_stats" in fields:
        # Use real sparse stats with no dense headers on modern engines.
        stats_fields = inspect.signature(io.RequestOutputStats).parameters
        required = {
            name: None for name, p in stats_fields.items() if p.default is inspect.Parameter.empty
        }
        out.request_stats = [
            io.RequestOutputStats(**(required | vars(stats))) for stats in out.request_stats
        ]
    values = {name: None for name, p in fields.items() if p.default is inspect.Parameter.empty}
    values.update({name: value for name, value in vars(out).items() if name in fields})
    actual = io.BatchTokenIDOutput(**values)
    assert msgspec.msgpack.encode(
        BatchTokenIDSlimOutput.from_full(actual)
    ) == msgspec.msgpack.encode(BatchTokenIDSlimOutput.from_full(out))

    from smg_grpc_servicer.sglang.zmq_msgpack import MsgpackSendSocket

    class Socket:
        frames = []

        def send(self, frame, **_kwargs):
            self.frames.append(msgspec.msgpack.decode(frame))

    sender = MsgpackSendSocket()
    sender.socket = Socket()
    sender.send_output(actual)
    assert not sender.pending_aborts
    assert sender.socket.frames[-1] == msgspec.msgpack.decode(
        msgspec.msgpack.encode(BatchTokenIDSlimOutput.from_full(actual))
    )
    if "request_stats" in fields:
        assert sender.output_adapter._counts["a"] == (100, 20)
    sender.send_terminal_abort("a", "test abort")
    assert "a" not in sender.output_adapter._counts


@pytest.mark.parametrize("sparse", [False, True])
def test_embedding_usage_preserves_the_wire(sparse):
    """Embedding stats normalize to the same wire as legacy dense columns."""
    from smg_grpc_servicer.sglang.zmq_msgpack import BatchEmbeddingSlimOutput

    legacy = Batch(
        rids=["e"],
        embeddings=[[0.5]],
        prompt_tokens=[12],
        cached_tokens=[4],
        finished_reasons=[{"type": "stop"}],
    )
    modern = Batch(
        rids=["e"],
        embeddings=[[0.5]],
        finished_reasons=[{"type": "stop"}],
        request_stats=[SimpleNamespace(prompt_tokens=12, cached_tokens=4)],
    )
    actual = BatchEmbeddingSlimOutput.from_full(modern if sparse else legacy)
    assert msgspec.msgpack.encode(actual) == msgspec.msgpack.encode(
        BatchEmbeddingSlimOutput.from_full(legacy)
    )


def test_abort_releases_sparse_counts_even_when_the_socket_fails(monkeypatch):
    """A failed terminal write must not keep a cancelled request's counts."""
    from smg_grpc_servicer.sglang.zmq_msgpack import MsgpackSendSocket

    sender = MsgpackSendSocket()
    sender.output_adapter.read(batch())

    def fail(_output):
        """Model a disconnected wire during a terminal abort."""
        raise RuntimeError("disconnected")

    monkeypatch.setattr(sender, "_send_slim", fail)
    with pytest.raises(RuntimeError, match="disconnected"):
        sender.send_terminal_abort("a", "cancelled")
    assert "a" not in sender.output_adapter._counts


@pytest.mark.parametrize("rows", [None, []])
def test_slim_wire_still_rejects_absent_legacy_rows(rows):
    """Python's empty-row fallback must not weaken the Rust wire contract."""
    with pytest.raises(ValueError, match="row count"):
        BatchTokenIDSlimOutput.from_full(batch(flat=False, output_ids=rows))


@pytest.mark.parametrize("sparse", [False, True])
def test_counts_are_indexable_for_every_request(sparse):
    """Missing or short usage columns normalize to one count per request."""
    for values in [None, [], [9]]:
        out = batch(sparse=sparse)
        if sparse:
            out.request_stats = values and [SimpleNamespace(prompt_tokens=9, cached_tokens=9)]
        else:
            out.prompt_tokens = out.cached_tokens = values
        rows, prompts, cached = SchedulerOutputAdapter().read(out)
        assert len(rows) == len(prompts) == len(cached) == len(out.rids)
        assert prompts == cached == ([9, 0, 0] if values else [0, 0, 0])

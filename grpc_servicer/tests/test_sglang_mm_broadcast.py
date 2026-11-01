"""The source rank must open its SHM handle before collective materialization."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import mm_broadcast


class Tensor:
    is_cpu = True

    def numel(self):
        """Return a feature size above the SHM threshold."""
        return 100_000

    def element_size(self):
        """Model two-byte encoder elements."""
        return 2


@pytest.fixture
def pointers(monkeypatch):
    """Stub engine SHM handles and record whether the source rank opens them."""
    made = []

    class Pointer:
        def __init__(self, tensor):
            """Retain the source tensor and record handle creation."""
            self.tensor = tensor
            self.opened = False
            made.append(self)

        def __getstate__(self):
            """Expose the payload sent to peers by the test handle."""
            return {"tensor": self.tensor}

        def __setstate__(self, state):
            """Record opening the serialized handle on the source rank."""
            self.tensor = state["tensor"]
            self.opened = True

    module = ModuleType("sglang.srt.managers.mm_utils")
    module.ShmPointerMMData = Pointer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return made, module


def receiver(nnodes=1, tp_size=8):
    """Describe local TP topology and the required engine finalization capability."""
    return SimpleNamespace(
        server_args=SimpleNamespace(nnodes=nnodes, tp_size=tp_size),
        _finalize_shm_features=lambda reqs: None,
    )


def request():
    """Create a CPU feature request whose tensor exceeds the broadcast threshold."""
    item = SimpleNamespace(feature=Tensor())
    return SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=[item])), item


def test_source_handle_opened_and_same_request_broadcast(pointers):
    """Verify source handle opened and same request broadcast."""
    made, _ = pointers
    req, item = request()
    feature = item.feature

    def original(owner, reqs, *, local_reqs):
        """Check the source handle is already open when collective broadcasting starts."""
        assert item.feature.opened
        assert item.feature.tensor is feature
        assert local_reqs == ["abort"]
        return reqs

    assert mm_broadcast.broadcast_with_shm(original, receiver(), [req], local_reqs=["abort"]) == [
        req
    ]
    assert len(made) == 1


@pytest.mark.parametrize("nnodes,tp_size", [(2, 8), (1, 1)])
def test_no_shared_handles_for_remote_or_single_rank(nnodes, tp_size, pointers):
    """Verify no shared handles for remote or single rank."""
    made, _ = pointers
    req, item = request()
    original = item.feature
    mm_broadcast.broadcast_with_shm(lambda _, reqs: reqs, receiver(nnodes, tp_size), [req])
    assert item.feature is original
    assert not made


def test_full_shm_retains_inline_feature(pointers):
    """Verify full shm retains inline feature."""
    _, module = pointers

    def full(_):
        """Simulate a recoverable shared-memory allocation failure."""
        raise OSError(28, "No space left on device")

    module.ShmPointerMMData = full
    req, item = request()
    original = item.feature
    result = mm_broadcast.broadcast_with_shm(lambda _, reqs: reqs, receiver(), [req])
    assert result == [req]
    assert item.feature is original


def test_engine_shm_roundtrip_preserves_dtype_values_and_cleans_up():
    """Verify engine shm roundtrip preserves dtype values and cleans up."""
    pytest.importorskip("sglang")
    torch = pytest.importorskip("torch")
    from multiprocessing import shared_memory

    from sglang.srt.managers.mm_utils import ShmPointerMMData

    feature = torch.arange(100_000, dtype=torch.float32).to(torch.bfloat16)
    item = SimpleNamespace(feature=feature)
    req = SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=[item]))
    names = []

    def broadcast(_, reqs):
        """Materialize the source handle using the real engine lifecycle."""
        source = item.feature
        names.append(source.shm_name)
        peer = ShmPointerMMData.__new__(ShmPointerMMData)
        peer.__setstate__(source.__getstate__())
        try:
            local_values = source.materialize()
            remote_values = peer.materialize()
            assert torch.equal(local_values, feature)
            assert torch.equal(remote_values, feature)
            assert local_values.dtype == torch.bfloat16
            # Both ranks own writable storage; Linux mappings are private.
            local_values[0] = -1
            assert remote_values[0] == feature[0]
        finally:
            source.close_and_unlink()
            peer.close_and_unlink()
        return reqs

    mm_broadcast.broadcast_with_shm(broadcast, receiver(), [req])
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0])


@pytest.mark.parametrize(
    "owner", [SimpleNamespace(), SimpleNamespace(server_args=SimpleNamespace(nnodes=1, tp_size=8))]
)
def test_unknown_engine_capabilities_keep_inline_transport(owner, pointers):
    """Verify unknown engine capabilities keep inline transport."""
    made, _ = pointers
    req, item = request()
    feature = item.feature
    assert mm_broadcast.broadcast_with_shm(lambda _, reqs: reqs, owner, [req]) == [req]
    assert item.feature is feature
    assert not made


def test_missing_engine_pointer_keeps_inline_transport(pointers):
    """Verify missing engine pointer keeps inline transport."""
    made, module = pointers
    del module.ShmPointerMMData
    req, item = request()
    feature = item.feature
    assert mm_broadcast.broadcast_with_shm(lambda _, reqs: reqs, receiver(), [req]) == [req]
    assert item.feature is feature
    assert not made

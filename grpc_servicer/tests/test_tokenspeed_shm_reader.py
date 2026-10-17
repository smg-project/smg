"""The TokenSpeed servicer reads a ``/dev/shm`` tensor through the shared reader
(``mm_shm``), so a symlink planted under the segment's name is refused before a
byte is read, as it is for the other engine on that reader."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2, tokenspeed_scheduler_pb2  # noqa: E402
from smg_grpc_servicer import mm_shm  # noqa: E402

SHM = Path("/dev/shm")

pytestmark = pytest.mark.skipif(
    not (SHM.is_dir() and os.access(SHM, os.W_OK)), reason="no writable /dev/shm"
)


@pytest.fixture
def segment() -> str:
    name = f"smg-tokenspeed-shm-test-{os.getpid()}-{uuid.uuid4().hex}"
    yield name
    path = SHM / name
    if path.is_symlink() or path.is_file():
        path.unlink()


def test_a_symlinked_segment_is_refused_through_the_servicer(
    tokenspeed_servicer_mod, segment, tmp_path, monkeypatch
):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", True)
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(bytes(8))
    os.symlink(target, SHM / segment)
    tensor = tokenspeed_scheduler_pb2.TensorData(
        shm=common_pb2.ShmHandle(name=segment, offset=0, nbytes=8),
        dtype="float32",
        shape=[2],
    )

    servicer = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    with pytest.raises(ValueError, match="is a symlink"):
        servicer._tensor_from_proto(tensor)
    assert (SHM / segment).is_symlink()
    assert target.read_bytes() == bytes(8)


def test_a_plain_segment_reaches_the_tensor_decode(tokenspeed_servicer_mod, segment, monkeypatch):
    """The shared reader hands the segment's bytes on: the decode after it is
    the engine's (stubbed here), so the read is observed through a spy that
    calls the real reader."""
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    (SHM / segment).write_bytes(bytes(range(8)))
    seen = []
    real_reader = mm_shm.tensor_payload_bytes
    monkeypatch.setattr(
        tokenspeed_servicer_mod.mm_shm,
        "tensor_payload_bytes",
        lambda tensor_data: seen.append(real_reader(tensor_data)) or seen[-1],
    )
    tensor = tokenspeed_scheduler_pb2.TensorData(
        shm=common_pb2.ShmHandle(name=segment, offset=0, nbytes=8),
        dtype="float32",
        shape=[2],
    )

    servicer = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    try:
        servicer._tensor_from_proto(tensor)
    except (AttributeError, TypeError):
        pass  # the stubbed engine has no tensor constructor; the read already happened
    assert seen == [bytes(range(8))]


def test_the_servicer_keeps_no_reader_of_its_own(tokenspeed_servicer_mod):
    servicer = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    for name in ("_tensor_payload_bytes", "_tensor_payload_bytes_from_shm", "_validated_shm_name"):
        assert not hasattr(servicer, name), name
    assert not hasattr(tokenspeed_servicer_mod, "UNLINK_MM_SHM_AFTER_READ")
    assert tokenspeed_servicer_mod.mm_shm is mm_shm

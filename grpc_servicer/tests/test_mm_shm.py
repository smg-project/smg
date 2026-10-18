"""The shared ``/dev/shm`` tensor reader (``mm_shm``) reads a plain segment of its
own and refuses everything else: a symlink planted under the name, a directory, a
segment shorter than the handle says, a name that is not a bare file name."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from smg_grpc_servicer import mm_shm

SHM = Path("/dev/shm")

pytestmark = pytest.mark.skipif(
    not (SHM.is_dir() and os.access(SHM, os.W_OK)), reason="no writable /dev/shm"
)


class _Handle:
    """The fields of a ``ShmHandle`` the reader uses."""

    def __init__(self, name: str, nbytes: int, offset: int = 0) -> None:
        self.name = name
        self.nbytes = nbytes
        self.offset = offset


class _TensorData:
    """A ``TensorData`` with one payload set."""

    def __init__(self, kind: str | None, value) -> None:
        self._kind = kind
        if kind is not None:
            setattr(self, kind, value)

    def WhichOneof(self, _oneof: str) -> str:  # noqa: N802 (protobuf's spelling)
        return self._kind


@pytest.fixture
def segment() -> str:
    """A fresh segment name; whatever the test leaves under it is removed."""
    name = f"smg-mm-shm-test-{os.getpid()}-{uuid.uuid4().hex}"
    yield name
    path = SHM / name
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        path.rmdir()


def test_a_segment_is_read_at_its_offset_and_unlinked(segment, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", True)
    (SHM / segment).write_bytes(b"abcdefgh")

    assert mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=4, offset=2)) == b"cdef"
    assert not (SHM / segment).exists()


def test_the_segment_stays_when_unlinking_is_off(segment, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    (SHM / segment).write_bytes(b"abcdefgh")

    assert mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=8)) == b"abcdefgh"
    assert (SHM / segment).read_bytes() == b"abcdefgh"


def test_a_symlink_under_the_name_is_refused_and_its_target_left_unread(
    segment, tmp_path, monkeypatch
):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", True)
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(b"not for the worker")
    os.symlink(target, SHM / segment)

    with pytest.raises(ValueError, match="is a symlink"):
        mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=4))
    # Neither the link nor what it points at is touched.
    assert (SHM / segment).is_symlink()
    assert target.read_bytes() == b"not for the worker"


def test_a_directory_under_the_name_is_refused_by_name(segment, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", True)
    (SHM / segment).mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=4))
    assert (SHM / segment).is_dir()


def test_a_segment_shorter_than_the_handle_is_a_length_error(segment, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    (SHM / segment).write_bytes(b"abcd")

    with pytest.raises(ValueError, match="byte length mismatch"):
        mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=8))


def test_a_missing_segment_is_the_operating_systems_error(segment):
    with pytest.raises(FileNotFoundError):
        mm_shm.tensor_payload_bytes_from_shm(_Handle(segment, nbytes=4))


@pytest.mark.parametrize("name", ["", "/", "a/b", ".", "..", "a\x00b", "//"])
def test_a_name_that_is_not_a_bare_file_name_is_refused(name):
    with pytest.raises(ValueError, match="Invalid TensorData.shm name"):
        mm_shm.validated_shm_name(name)


def test_a_leading_slash_is_dropped_from_a_bare_name():
    assert mm_shm.validated_shm_name("/plain") == "plain"


def test_the_payload_dispatcher_reads_inline_bytes_and_a_segment(segment, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    assert mm_shm.tensor_payload_bytes(_TensorData("inline", b"\x01\x02")) == b"\x01\x02"

    (SHM / segment).write_bytes(b"\x03\x04")
    assert (
        mm_shm.tensor_payload_bytes(_TensorData("shm", _Handle(segment, nbytes=2))) == b"\x03\x04"
    )

    with pytest.raises(ValueError, match="not implemented"):
        mm_shm.tensor_payload_bytes(_TensorData("remote", object()))
    with pytest.raises(ValueError, match="payload is required"):
        mm_shm.tensor_payload_bytes(_TensorData(None, None))

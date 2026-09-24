"""The SGLang multimodal wire decodes through the shared tensor reader.

Run with: pytest grpc_servicer/tests/test_sglang_mm_tensors.py
"""

import pytest

pytest.importorskip("smg_grpc_proto")
torch = pytest.importorskip("torch")

from smg_grpc_proto import sglang_scheduler_pb2  # noqa: E402
from smg_grpc_servicer.tensor_wire import tensor_from_parts  # noqa: E402

SAMPLE = [0.0, 1.0, -2.5, 3.14159]


def _decode(td):
    """What the servicer does with a TensorData off this wire."""
    return tensor_from_parts(td.data, td.shape, td.dtype)


def _raw_bytes(tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())


def test_float32_pixel_values_decode_unchanged():
    want = torch.tensor(SAMPLE, dtype=torch.float32)
    td = sglang_scheduler_pb2.TensorData(
        dtype="float32", shape=[len(SAMPLE)], data=_raw_bytes(want)
    )

    got = _decode(td)

    assert got.dtype == torch.float32
    assert torch.equal(got, want)


def test_a_model_specific_int64_tensor_keeps_its_shape():
    want = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    td = sglang_scheduler_pb2.TensorData(dtype="int64", shape=[2, 3], data=_raw_bytes(want))

    got = _decode(td)

    assert tuple(got.shape) == (2, 3)
    assert torch.equal(got, want)


def test_an_unknown_dtype_is_refused_rather_than_guessed():
    # Read as float32 this would have produced a well-shaped tensor of
    # meaningless numbers, and the model would have answered from it.
    td = sglang_scheduler_pb2.TensorData(dtype="float64", shape=[1], data=b"\x00" * 8)

    with pytest.raises(ValueError, match="Unsupported proto tensor dtype"):
        _decode(td)


def test_bytes_read_at_the_wrong_width_are_refused_by_length():
    payload = _raw_bytes(torch.tensor(SAMPLE, dtype=torch.float32))
    td = sglang_scheduler_pb2.TensorData(dtype="int64", shape=[len(SAMPLE)], data=payload)

    with pytest.raises(ValueError, match="byte length mismatch"):
        _decode(td)


def test_the_dtypes_this_wire_names_are_all_accepted():
    # The proto documents these three for TensorData.dtype; a name it advertises
    # and the reader does not know turns every media request into a failure.
    for name in ("float32", "int64", "uint32"):
        width = torch.empty(0, dtype=getattr(torch, name)).element_size()
        td = sglang_scheduler_pb2.TensorData(dtype=name, shape=[2], data=b"\x00" * (2 * width))
        assert _decode(td).dtype == getattr(torch, name)

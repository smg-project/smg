"""Unit tests for multimodal tensor decoding (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_tensors.py
"""

import struct

import pytest

pytest.importorskip("smg_grpc_proto")
torch = pytest.importorskip("torch")

from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_servicer.vllm import mm_tensors  # noqa: E402

# Values chosen to exercise the parts of the range that distinguish the two
# half-width formats: float16's largest finite value, one far below its
# smallest normal, and fractions neither format holds exactly.
SAMPLE = [0.0, 1.0, -2.5, 3.14159, 0.1, -0.333333, 65504.0, 1e-7]


def _f32_to_bf16_bits(value: float) -> int:
    """The sender's rounding: round-to-nearest-even on the discarded half."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def _inline(dtype: str, shape: list[int], payload: bytes) -> vllm_engine_pb2.TensorData:
    return vllm_engine_pb2.TensorData(dtype=dtype, shape=shape, inline=payload)


def _raw_bytes(tensor) -> bytes:
    """Little-endian bytes of a tensor. Goes through a uint8 view rather than
    numpy, which has no bfloat16 and is not a dependency of this package."""
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())


def test_a_bfloat16_payload_decodes_to_the_values_the_sender_rounded():
    # Half the bytes of float32 for the same element count, and the sender's
    # own rounding must land on exactly what torch would have produced, so
    # moving the cast across the wire changes nothing the engine sees.
    payload = b"".join(struct.pack("<H", _f32_to_bf16_bits(v)) for v in SAMPLE)
    assert len(payload) == 2 * len(SAMPLE)

    got = mm_tensors.tensor_from_proto(_inline("bfloat16", [len(SAMPLE)], payload))

    want = torch.tensor(SAMPLE, dtype=torch.float32).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)


def test_a_float16_payload_round_trips():
    want = torch.tensor(SAMPLE, dtype=torch.float32).to(torch.float16)
    payload = _raw_bytes(want)
    assert len(payload) == 2 * len(SAMPLE)

    got = mm_tensors.tensor_from_proto(_inline("float16", [len(SAMPLE)], payload))

    assert got.dtype == torch.float16
    assert torch.equal(got, want)


def test_float32_still_decodes_unchanged():
    # The width every already-deployed sender uses; widening the accepted set
    # must not disturb it.
    want = torch.tensor(SAMPLE, dtype=torch.float32)
    payload = _raw_bytes(want)
    assert len(payload) == 4 * len(SAMPLE)

    got = mm_tensors.tensor_from_proto(_inline("float32", [len(SAMPLE)], payload))

    assert got.dtype == torch.float32
    assert torch.equal(got, want)


def test_a_multidimensional_payload_keeps_its_shape():
    want = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(torch.bfloat16)

    got = mm_tensors.tensor_from_proto(_inline("bfloat16", [2, 3, 4], _raw_bytes(want)))

    assert tuple(got.shape) == (2, 3, 4)
    assert torch.equal(got, want)


def test_bytes_read_at_the_wrong_width_are_refused_by_length():
    # A sender that labels float32 bytes as bfloat16 would otherwise yield twice
    # the elements. The count is what catches it, so the error names the width
    # it was read against rather than only the byte total.
    payload = _raw_bytes(torch.tensor(SAMPLE, dtype=torch.float32))

    with pytest.raises(ValueError, match="bfloat16"):
        mm_tensors.tensor_from_proto(_inline("bfloat16", [len(SAMPLE)], payload))


def test_a_truncated_payload_is_refused():
    payload = _raw_bytes(torch.tensor(SAMPLE, dtype=torch.float32).to(torch.bfloat16))

    with pytest.raises(ValueError, match="byte length mismatch"):
        mm_tensors.tensor_from_proto(_inline("bfloat16", [len(SAMPLE)], payload[:-2]))


def test_an_unknown_dtype_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="Unsupported proto tensor dtype"):
        mm_tensors.tensor_from_proto(_inline("float64", [1], b"\x00" * 8))


def test_the_accepted_names_match_the_wire_vocabulary():
    # The names here are the contract with the sender; a rename on either side
    # turns every media request into an unsupported-dtype failure.
    assert set(mm_tensors.PROTO_DTYPE_MAP) == {
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "uint32",
    }

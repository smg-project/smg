"""Decoding for tensors that arrive as raw bytes plus a shape and a dtype name.

Shared by the backends, which differ only in how the bytes reach them.
"""

import torch

# Proto dtype name -> torch dtype. The half-width float entries let a sender
# ship pixel data at the width the model already runs at, rather than widening
# it for the trip and having the engine narrow it again on arrival. A name that
# is not listed is refused rather than guessed at, since reading the bytes at
# the wrong width would silently produce a plausible tensor of wrong numbers.
PROTO_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "int64": torch.int64,
    "uint32": torch.uint32,
}


def tensor_from_parts(payload: bytes, shape, dtype_name: str) -> torch.Tensor:
    """Build a tensor from wire bytes, or refuse the payload."""
    torch_dtype = PROTO_DTYPE_MAP.get(dtype_name)
    if torch_dtype is None:
        raise ValueError(f"Unsupported proto tensor dtype: {dtype_name!r}")

    dims = list(shape)
    element_count = 1
    for dim in dims:
        element_count *= dim
    expected = element_count * torch_dtype.itemsize
    # Say which dtype the length was read against. A sender and a receiver that
    # disagree about the width produce a length mismatch and nothing else, so
    # naming the width here is what distinguishes it from a truncated payload.
    if len(payload) != expected:
        raise ValueError(
            f"TensorData byte length mismatch for dtype={dtype_name!r}, shape={dims}: "
            f"expected {expected}, got {len(payload)}"
        )

    # bytearray so the buffer is writable; torch warns on read-only input.
    return torch.frombuffer(bytearray(payload), dtype=torch_dtype).reshape(*dims)

"""Tensor decoding for multimodal wire payloads (engine-free, no vLLM required).

The router sends preprocessed media as raw little-endian bytes plus a shape and
a dtype name; this turns that back into a tensor.
"""

import torch

from smg_grpc_servicer import mm_shm
from smg_grpc_servicer.tensor_wire import PROTO_DTYPE_MAP, tensor_from_parts

__all__ = ["PROTO_DTYPE_MAP", "tensor_from_proto"]


def tensor_from_proto(td) -> torch.Tensor:
    """Deserialize a ``TensorData`` proto message into a ``torch.Tensor``."""
    return tensor_from_parts(mm_shm.tensor_payload_bytes(td), td.shape, td.dtype)

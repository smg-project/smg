"""Exercise the actual metadata handler without importing vLLM or torch."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.vllm import model_info

_SERVICER = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "servicer.py"


def _server_info_method(response_type):
    # Execute the real RPC body, as in test_sglang_kv_events, so removing its
    # response assignment fails this test even if server_facts stays correct.
    tree = ast.parse(_SERVICER.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmEngineServicer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "GetServerInfo"
    )
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "vllm_engine_pb2": SimpleNamespace(GetServerInfoResponse=response_type),
        "server_facts": model_info.server_facts,
        "mm_device_do_normalize": model_info.mm_device_do_normalize,
        "mm_item_limits": model_info.mm_item_limits,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(_SERVICER), "exec"), namespace)
    return namespace["GetServerInfo"]


def _response_type(legacy):
    """Build either wire schema regardless of the installed proto version."""
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(
        vllm_engine_pb2.DESCRIPTOR.serialized_pb
    )
    response = next(m for m in descriptor.message_type if m.name == "GetServerInfoResponse")
    field = next((f for f in response.field if f.name == "multimodal_encoder_dtype"), None)
    if legacy and field is not None:
        response.field.remove(field)
    elif not legacy and field is None:
        response.field.add(
            name="multimodal_encoder_dtype",
            number=22,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(common_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("vllm.grpc.engine.GetServerInfoResponse")
    )


@pytest.mark.parametrize("installed_legacy", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("torch.bfloat16", "bfloat16"),
        ("torch.float16", "float16"),
        ("torch.float32", "float32"),
        ("torch.float64", ""),
        (None, ""),
    ],
)
def test_get_server_info_reports_encoder_dtype_with_current_and_legacy_protos(
    monkeypatch, installed_legacy, legacy, dtype, expected
):
    """The handler works with both schemas, even when only old stubs are installed."""
    if installed_legacy:
        monkeypatch.setattr(vllm_engine_pb2, "DESCRIPTOR", _response_type(True).DESCRIPTOR.file)
    response_type = _response_type(legacy)
    model_config = SimpleNamespace(dtype=dtype, is_multimodal_model=False)
    config = SimpleNamespace(
        model_config=model_config, parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    servicer = SimpleNamespace(engine=SimpleNamespace(vllm_config=config), _mm_processor=None)
    info = asyncio.run(_server_info_method(response_type)(servicer, None, None))
    assert info.model_dtype == (dtype or "")
    if legacy:
        assert "multimodal_encoder_dtype" not in info.DESCRIPTOR.fields_by_name
    else:
        assert info.multimodal_encoder_dtype == expected

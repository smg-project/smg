"""The SGLang Python servicer gives a request's ``cache_salt`` and ``extra_key``
to the scheduler, and advertises the fields it reads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_sglang_server_info import _info, _OlderServerArgs, _servicer
from test_sglang_server_info import servicer_mod as _servicer_fixture

pytest.importorskip("smg_grpc_proto")
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # noqa: E402
from smg_grpc_proto.generated import sglang_scheduler_pb2 as pb  # noqa: E402

servicer_mod = _servicer_fixture


def _older_request_class():
    """``GenerateRequest`` of a proto package without the field."""
    file = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)
    (request,) = (message for message in file.message_type if message.name == "GenerateRequest")
    (field,) = (field for field in request.field if field.name == "cache_partition")
    request.field.remove(field)
    pool = descriptor_pool.DescriptorPool()
    for dependency in pb.DESCRIPTOR.dependencies:
        pool.AddSerializedFile(dependency.serialized_pb)
    pool.AddSerializedFile(file.SerializeToString())
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("sglang.grpc.scheduler.GenerateRequest")
    )


def _request(partition=None, **sides):
    return pb.GenerateRequest(
        request_id="r",
        tokenized=pb.TokenizedInput(input_ids=[1]),
        cache_partition=partition,
        **sides,
    )


@pytest.fixture
def convert(monkeypatch, servicer_mod):
    servicer = _servicer(servicer_mod, _OlderServerArgs())
    servicer.tokenizer = None
    monkeypatch.setattr(servicer_mod, "TokenizedGenerateReqInput", SimpleNamespace)
    monkeypatch.setattr(
        servicer,
        "_convert_sampling_params",
        lambda _: SimpleNamespace(normalize=lambda **_: None),
    )

    def forwarded(request):
        converted = servicer._convert_generate_request(request)
        return converted.cache_salt, converted.extra_key

    return forwarded


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (None, (None, None)),
        ({}, (None, None)),
        ({"cache_salt": "", "extra_key": ""}, (None, None)),
        ({"cache_salt": "salt"}, ("salt", None)),
        ({"cache_salt": "", "extra_key": "key"}, (None, "key")),
        ({"cache_salt": "salt", "extra_key": "key"}, ("salt", "key")),
    ],
)
def test_both_values_reach_the_scheduler_request(convert, partition, expected):
    """An empty string is unset. The value does not depend on the side: a
    request of a prefill/decode pair has the bootstrap fields."""
    if partition is not None:
        partition = pb.CachePartition(**partition)
    assert convert(_request(partition)) == expected
    bootstrap = pb.DisaggregatedParams(
        bootstrap_host="127.0.0.1", bootstrap_port=8998, bootstrap_room=1
    )
    assert convert(_request(partition, disaggregated_params=bootstrap)) == expected


def test_server_info_advertises_the_fields(servicer_mod):
    info = _info(_servicer(servicer_mod, _OlderServerArgs()))
    assert info.server_args["cache_partition_fields"] == "cache_salt,extra_key"


def test_an_older_proto_package_neither_reads_nor_advertises_the_fields(
    monkeypatch, servicer_mod, convert
):
    wire = _request(pb.CachePartition(cache_salt="salt", extra_key="key")).SerializeToString()
    older = _older_request_class()
    monkeypatch.setattr(servicer_mod.sglang_scheduler_pb2, "GenerateRequest", older)
    request = older.FromString(wire)
    assert request.request_id == "r"
    assert convert(request) == (None, None)
    info = _info(_servicer(servicer_mod, _OlderServerArgs()))
    assert "cache_partition_fields" not in info.server_args


def test_the_rust_servicers_launcher_does_not_advertise_the_fields():
    """The Rust servicer writes the key: its package has its own version."""
    from smg_grpc_servicer.sglang.rust import server_facts

    facts = server_facts(SimpleNamespace(enable_lora=False, lora_paths=None))
    assert "cache_partition_fields" not in json.loads(facts["server_args_json"])

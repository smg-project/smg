"""Run the real Generate handler with a recording engine, without importing vLLM."""

import ast
import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2 as pb
from smg_grpc_servicer.vllm.mm_keys import batches_missing_pixels, mm_batches
from smg_grpc_servicer.vllm.mm_salt import has_preprocessed_mm_payload


class _Aborted(Exception):
    """The fake gRPC context stops the RPC just as context.abort does."""


def _generate(**overrides):
    """Compile the actual handler with engine-independent dependencies."""
    path = Path(__file__).parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    module = ast.parse(path.read_text())
    helper = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "_valid_cpu_ec_handle"
    )
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "VllmEngineServicer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "Generate"
    )
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "json": json,
        "asyncio": asyncio,
        "build_media_identity": Mock(),
        "time": time,
        "logger": logging.getLogger(__name__),
        "grpc": grpc,
        "mm_batches": mm_batches,
        "has_preprocessed_mm_payload": has_preprocessed_mm_payload,
        "batches_missing_pixels": batches_missing_pixels,
        "params_from_request": lambda req: (
            json.loads(req.kv_transfer_params_json)
            if req.HasField("kv_transfer_params_json")
            else None
        ),
        "trace_headers": lambda _: {},
        "_prompt_length": lambda prompt: len(prompt["prompt_token_ids"]),
        "grpc_code_for": lambda error: (
            grpc.StatusCode.INVALID_ARGUMENT
            if isinstance(error, ValueError)
            else grpc.StatusCode.INTERNAL
        ),
        "MmProcessorUnavailable": type("MmProcessorUnavailable", (Exception,), {}),
        "VLLMClientError": type("VLLMClientError", (Exception,), {}),
        "GracefulHTTPError": type("GracefulHTTPError", (Exception,), {}),
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Generate"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hashes", [[], ["h"]])
@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize(
    "ec",
    [
        None,
        "{}",
        '{"ec_items":[]}',
        '{"ec_items":[{"mm_hash":"h"}]}',
        '{"ec_items":[{"mm_hash":"h","transfer_id":"t"}]}',
        '{"ec_items":[{"mm_hash":"other","transfer_id":"t"}]}',
        "{",
        "[]",
        "null",
    ],
)
@pytest.mark.parametrize("kv", [False, True])
async def test_grid_only_requires_a_handoff(hashes, consumer, ec, kv):
    """Reject incomplete EC inputs before processing tensors or submitting engine work."""
    request = pb.GenerateRequest(
        request_id="ec-test",
        tokenized=pb.TokenizedInput(input_ids=[1, 2]),
        mm_inputs=pb.MultimodalInputs(mm_hashes=hashes),
    )
    request.mm_inputs.model_specific_tensors["image_grid_thw"].shape.extend([1, 3])
    if ec is not None:
        request.ec_transfer_params_json = ec
    if kv:
        request.kv_transfer_params_json = '{"do_remote_prefill":true}'
    seen = []

    async def generate(**kwargs):
        """Record requests that passed the real handler's validation."""
        seen.append(kwargs)
        if False:
            yield

    async def abort(code, message):
        """End the RPC with an observable status."""
        raise _Aborted(code, message)

    build = Mock(return_value={"prompt_token_ids": [1, 2]})
    servicer = SimpleNamespace(
        engine=SimpleNamespace(
            generate=generate,
            vllm_config=SimpleNamespace(
                ec_transfer_config=SimpleNamespace(
                    is_ec_consumer=consumer,
                    is_ec_producer=not consumer,
                    ec_connector="ECMooncakeConnector",
                )
            ),
        ),
        _build_preprocessed_mm_inputs=build,
        _off_the_event_loop=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            extra_args=None, logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    valid_json = ec not in ("{", "[]", "null")
    allowed = valid_json and (
        kv
        if ec is None
        else consumer and bool(hashes) and ec == '{"ec_items":[{"mm_hash":"h","transfer_id":"t"}]}'
    )
    if allowed:
        assert [reply async for reply in _generate()(servicer, request, context)] == []
        assert len(seen) == 1
        build.assert_called_once()
        if ec is not None:
            assert seen[0]["sampling_params"].extra_args["ec_transfer_params"] == json.loads(ec)
    else:
        with pytest.raises(_Aborted) as exc:
            _ = [reply async for reply in _generate()(servicer, request, context)]
        assert exc.value.args[0] is grpc.StatusCode.INVALID_ARGUMENT
        assert not seen
        build.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize("connector", ["ECCPUConnector", "ECExampleConnector"])
@pytest.mark.parametrize(
    "handle",
    [
        {},
        {"metadata": {}},
        {"peer_host": "e"},
        {"peer_host": "e", "peer_port": 0},
        {"peer_host": "e", "peer_port": 9000},
    ],
)
async def test_encoder_only_validates_connector_specific_handles(consumer, connector, handle):
    """CPU cache misses fail before metadata escapes; other connector handles stay opaque."""
    identity = pb.MediaIdentity(
        prompt_token_ids=[1, 2], mm_inputs=pb.MultimodalInputs(mm_hashes=["h"])
    )
    request = pb.GenerateRequest(
        request_id="encoder",
        tokenized=pb.TokenizedInput(input_ids=[1, 2]),
        mm_inputs=pb.MultimodalInputs(mm_hashes=["h"]),
        ec_transfer_params_json="{}",
    )
    request.mm_inputs.pixel_values.shape.extend([1, 2])

    async def generate(**kwargs):
        """Return an encoder-only completion with the chosen connector handle."""
        yield SimpleNamespace(
            finished=True, outputs=[], prompt_token_ids=[1, 2], ec_transfer_params={"h": handle}
        )

    async def abort(code, message):
        """End the RPC with an observable status."""
        raise _Aborted(code, message)

    complete = Mock(return_value="complete")
    servicer = SimpleNamespace(
        engine=SimpleNamespace(
            generate=generate,
            vllm_config=SimpleNamespace(
                ec_transfer_config=SimpleNamespace(
                    is_ec_consumer=consumer, is_ec_producer=True, ec_connector=connector
                )
            ),
        ),
        _build_preprocessed_mm_inputs=lambda *args: {"prompt_token_ids": [1, 2]},
        _off_the_event_loop=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            extra_args=None, logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
        _complete_response=complete,
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    method = _generate(
        build_media_identity=lambda prompt: identity, _request_rank=lambda *args, **kwargs: 0
    )
    if connector == "ECCPUConnector" and not (handle.get("peer_host") and handle.get("peer_port")):
        with pytest.raises(_Aborted) as exc:
            _ = [reply async for reply in method(servicer, request, context)]
        assert exc.value.args[0] is grpc.StatusCode.INTERNAL
        complete.assert_not_called()
    else:
        assert [reply async for reply in method(servicer, request, context)] == ["complete"]
        assert complete.call_args.kwargs["media_identity"] == identity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [None, "enabled", "disabled", "adaptive", "", "{", "[]", "null", "true"]
)
async def test_reasoning_template_kwargs_reach_engine_or_fail_before_submission(mode):
    """The Python handler preserves parser state and rejects non-object kwargs."""
    request = pb.GenerateRequest(
        request_id="thinking-test", tokenized=pb.TokenizedInput(input_ids=[1, 2])
    )
    invalid = mode in ("", "{", "[]", "null", "true")
    if mode is not None:
        request.chat_template_kwargs_json = mode if invalid else json.dumps({"thinking_mode": mode})
    seen = []

    async def generate(**kwargs):
        """Record arguments accepted by the real handler without running a model."""
        seen.append(kwargs)
        if False:
            yield

    async def abort(code, message):
        """Make gRPC validation failures observable to the test."""
        raise _Aborted(code, message)

    servicer = SimpleNamespace(
        engine=SimpleNamespace(
            generate=generate,
            renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
            vllm_config=SimpleNamespace(ec_transfer_config=None),
        ),
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            extra_args=None, logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    if invalid:
        with pytest.raises(_Aborted) as exc:
            _ = [reply async for reply in _generate()(servicer, request, context)]
        assert exc.value.args[0] is grpc.StatusCode.INVALID_ARGUMENT
        assert not seen
    else:
        assert [reply async for reply in _generate()(servicer, request, context)] == []
        assert len(seen) == 1
        expected = {"chat_template_kwargs": {"thinking_mode": mode}} if mode else None
        assert seen[0].get("reasoning_parser_kwargs") == expected

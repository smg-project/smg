"""Run the real Generate handler with a recording engine, without importing vLLM."""

import ast
import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2 as pb
from smg_grpc_servicer import constrained_decoding
from smg_grpc_servicer.vllm.mm_keys import batches_missing_pixels, mm_batches
from smg_grpc_servicer.vllm.mm_salt import has_preprocessed_mm_payload, request_cache_salt


class _Aborted(Exception):
    """The fake gRPC context stops the RPC just as context.abort does."""


def _generate(**overrides):
    """Compile the actual handler with engine-independent dependencies."""
    path = Path(__file__).parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    module = ast.parse(path.read_text())
    helpers = [
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_valid_cpu_ec_handle", "_takes_reasoning_ended")
    ]
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
        "inspect": inspect,
        "json": json,
        "asyncio": asyncio,
        "build_media_identity": Mock(),
        "time": time,
        "logger": logging.getLogger(__name__),
        "grpc": grpc,
        "mm_batches": mm_batches,
        "has_preprocessed_mm_payload": has_preprocessed_mm_payload,
        "request_cache_salt": request_cache_salt,
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
    exec(
        compile(ast.Module(body=[*helpers, method], type_ignores=[]), str(path), "exec"), namespace
    )
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
        _structured_outputs_backend="auto",
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
        _structured_outputs_backend="auto",
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
        _structured_outputs_backend="auto",
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_token", "constrained", "engine_takes", "handed"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, True, False),
        (True, True, False, False),
    ],
)
async def test_grammar_from_first_token_reaches_engine_as_reasoning_ended(
    first_token, constrained, engine_takes, handed
):
    """A shaped grammar's start reaches an engine whose generate() takes it."""
    params = pb.SamplingParams(grammar_from_first_token=first_token)
    if constrained:
        params.json_schema = "{}"
    request = pb.GenerateRequest(
        request_id="start-test",
        tokenized=pb.TokenizedInput(input_ids=[1, 2]),
        sampling_params=params,
    )
    seen = []

    async def generate(*, reasoning_ended="unsaid", **kwargs):
        """Record the switch an engine that names the keyword is handed."""
        seen.append(reasoning_ended)
        if False:
            yield

    async def older_generate(**kwargs):
        """Record the switch an engine that does not name the keyword is handed."""
        seen.append(kwargs.get("reasoning_ended", "unsaid"))
        if False:
            yield

    async def abort(code, message):
        """Make gRPC validation failures observable to the test."""
        raise _Aborted(code, message)

    handler = _generate()
    engine = SimpleNamespace(
        generate=generate if engine_takes else older_generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
        vllm_config=SimpleNamespace(ec_transfer_config=None),
    )
    servicer = SimpleNamespace(
        engine=engine,
        # The servicer's own read, the one its constructor makes once.
        _takes_reasoning_ended=handler.__globals__["_takes_reasoning_ended"](engine),
        _structured_outputs_backend="auto",
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            extra_args=None, logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    assert [reply async for reply in handler(servicer, request, context)] == []
    assert seen == [True if handed else "unsaid"]


def _converter():
    """Compile the actual sampling-params conversion over plain records in
    place of the engine's parameter classes."""
    path = Path(__file__).parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    function = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_sampling_params_from_proto"
    )
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    namespace = {
        **vars(constrained_decoding),
        "SamplingParams": SimpleNamespace,
        "StructuredOutputsParams": dict,
        "RequestOutputKind": SimpleNamespace(DELTA="delta", FINAL_ONLY="final_only"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_sampling_params_from_proto"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "sent_as"),
    [
        ("guidance", "json"),
        ("outlines", "json"),
        ("lm-format-enforcer", "json"),
        ("xgrammar", "structural_tag"),
        ("auto", "structural_tag"),
    ],
)
async def test_an_answer_tag_reaches_another_backend_as_its_schema(backend, sent_as):
    """A structural tag is spelled for xgrammar: one that is a JSON answer's
    schema alone reaches an engine started with another backend as that
    schema, and stays a tag under xgrammar and the engine's default."""
    schema = '{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}'
    tag = '{"format":{"type":"json_schema","json_schema":' + schema + ',"max_whitespace_cnt":32}}'
    request = pb.GenerateRequest(
        request_id="tag-test",
        tokenized=pb.TokenizedInput(input_ids=[1, 2]),
        sampling_params=pb.SamplingParams(structural_tag=tag),
    )
    seen = []

    async def generate(**kwargs):
        """Record the constraint the engine is handed."""
        seen.append(kwargs["sampling_params"].structured_outputs)
        if False:
            yield

    async def abort(code, message):
        """Make gRPC validation failures observable to the test."""
        raise _Aborted(code, message)

    engine = SimpleNamespace(
        generate=generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
        vllm_config=SimpleNamespace(ec_transfer_config=None),
    )
    servicer = SimpleNamespace(
        engine=engine,
        _structured_outputs_backend=backend,
        _sampling_params_from_proto=_converter(),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    assert [reply async for reply in _generate()(servicer, request, context)] == []
    assert seen == [{sent_as: schema if sent_as == "json" else tag}]

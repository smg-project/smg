"""Selected-token scores survive the Python SGLang transport without an engine.

Only engine imports are replaced; request state, output handling, protobuf
serialization, and the positional msgpack codec are the production code.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

msgspec = pytest.importorskip("msgspec")
pytest.importorskip("zmq")

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang"


def _load(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manager_mod(monkeypatch):
    # Importing SGLang itself loads the engine runtime's dependencies. These unused
    # imports are the only stand-ins; no request-manager behavior is mocked.
    symbols = {
        "sglang.srt.disaggregation.utils": "DisaggregationMode FAKE_BOOTSTRAP_HOST",
        "sglang.srt.managers.io_struct": (
            "AbortReq BatchEmbeddingOutput BatchTokenIDOutput FlushCacheReqOutput "
            "GetInternalStateReqOutput HealthCheckOutput ProfileReqOutput "
            "TokenizedEmbeddingReqInput TokenizedGenerateReqInput FlushCacheReqInput "
            "ProfileReq ProfileReqType"
        ),
        "sglang.srt.managers.load_snapshot": "LoadSnapshot create_load_snapshot_reader",
        "sglang.srt.observability.req_time_stats": (
            "APIServerReqTimeStats calibrate_time_diff real_time"
        ),
        "sglang.srt.server_args": "PortArgs ServerArgs",
        "sglang.srt.utils": "get_or_create_event_loop kill_process_tree get_bool_env_var",
        "sglang.srt.utils.network": "get_zmq_socket",
        "sglang.utils": "get_exception_traceback",
        "sglang.srt.configs.model_config": "ModelConfig",
        "sglang.srt.disaggregation.kv_events": "KVEventsConfig",
        "sglang.srt.managers.schedule_batch": (
            "Modality MultimodalDataItem MultimodalProcessorOutput"
        ),
        "sglang.srt.sampling.sampling_params": "SamplingParams",
        "sglang.srt.utils.hf_transformers_utils": "get_tokenizer",
        "smg_grpc_servicer.tensor_wire": "tensor_from_parts",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent.startswith("sglang") and parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, SimpleNamespace if attr == "AbortReq" else object)
        monkeypatch.setitem(sys.modules, name, module)
    return _load(
        monkeypatch,
        "smg_grpc_servicer.sglang.request_manager",
        "request_manager.py",
    )


@pytest.fixture
def servicer(monkeypatch, manager_mod):
    pytest.importorskip("smg_grpc_proto")
    module = _load(monkeypatch, "smg_grpc_servicer.sglang.servicer", "servicer.py")
    return module.SGLangSchedulerServicer.__new__(module.SGLangSchedulerServicer)


def _manager_and_state(module, *, stream=False):
    manager = module.GrpcRequestManager.__new__(module.GrpcRequestManager)
    state = module.GrpcReqState(
        request_id="score",
        grpc_context=None,
        out_queue=asyncio.Queue(),
        finished=False,
        event=asyncio.Event(),
        obj=SimpleNamespace(
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=[42, 7],
            stream=stream,
        ),
        time_stats=SimpleNamespace(
            first_token_time=0.0,
            set_first_token_time=lambda: None,
            set_last_time=lambda: None,
            set_finished_time=lambda: None,
        ),
    )
    manager.rid_to_state = {"score": state}
    return manager, state


def _output(*, values, ids=None, tokens=None, finished=True, **overrides):
    return SimpleNamespace(
        **{
            "rids": ["score"],
            "output_ids": [tokens or []],
            "finished_reasons": [{"type": "length"} if finished else None],
            "prompt_tokens": [3],
            "completion_tokens": [len(tokens or [])],
            "cached_tokens": [0],
            "input_token_logprobs_val": None,
            "input_token_logprobs_idx": None,
            "input_top_logprobs_val": None,
            "input_top_logprobs_idx": None,
            "output_token_logprobs_val": None,
            "output_token_logprobs_idx": None,
            "output_top_logprobs_val": None,
            "output_top_logprobs_idx": None,
            "output_token_ids_logprobs_val": [values],
            "output_token_ids_logprobs_idx": [ids or [[42, 7] for _ in values]],
            **overrides,
        }
    )


@pytest.mark.parametrize("input_values", [None, [], [None], [[]]])
def test_prefill_only_selected_scores_do_not_require_sampled_or_input_logprobs(
    manager_mod, input_values
):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        await manager._handle_batch_output(
            _output(values=[[-0.5, -7.0]], input_token_logprobs_val=input_values)
        )
        output = await state.out_queue.get()

        assert output["token_ids"] == []
        assert output["meta_info"]["completion_tokens"] == 0
        assert output["output_logprobs"]["token_ids_logprobs_val"] == [[-0.5, -7.0]]
        assert output["output_logprobs"]["token_ids_logprobs_idx"] == [[42, 7]]

    asyncio.run(exercise())


def test_nonstreaming_completion_keeps_scores_from_earlier_batches(manager_mod):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        await manager._handle_batch_output(_output(values=[[-0.5, -7.0]], finished=False))
        await state.out_queue.get()
        await manager._handle_batch_output(_output(values=[]))
        final = await state.out_queue.get()

        assert final["output_logprobs"]["token_ids_logprobs_val"] == [[-0.5, -7.0]]
        assert final["output_logprobs"]["token_ids_logprobs_idx"] == [[42, 7]]

    asyncio.run(exercise())


def test_streaming_chunks_are_incremental_and_complete_has_all_selected_rows(manager_mod, servicer):
    async def exercise():
        manager, state = _manager_and_state(manager_mod, stream=True)
        await manager._handle_batch_output(_output(values=[[-0.5, -7.0]], finished=False))
        first = servicer._create_chunk_response("score", await state.out_queue.get()).chunk
        await manager._handle_batch_output(_output(values=[[-1.0, -6.0]]))
        output = await state.out_queue.get()
        last = servicer._create_chunk_response("score", output).chunk
        complete = servicer._create_completion_response("score", output).complete

        assert [list(row.values) for row in first.output_logprobs.token_ids_logprobs] == [
            [-0.5, -7.0]
        ]
        assert [list(row.values) for row in last.output_logprobs.token_ids_logprobs] == [
            [-1.0, -6.0]
        ]
        assert [list(row.values) for row in complete.output_logprobs.token_ids_logprobs] == [
            [-0.5, -7.0],
            [-1.0, -6.0],
        ]

    asyncio.run(exercise())


def test_prefill_score_proto_roundtrip_keeps_candidate_order_and_raw_values(manager_mod, servicer):
    from smg_grpc_proto import sglang_scheduler_pb2

    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        await manager._handle_batch_output(_output(values=[[-0.5, -30.0]]))
        response = servicer._create_completion_response("score", await state.out_queue.get())
        decoded = sglang_scheduler_pb2.GenerateResponse.FromString(response.SerializeToString())

        assert list(decoded.complete.output_ids) == []
        assert decoded.complete.prompt_tokens == 3
        assert decoded.complete.completion_tokens == 0
        scores = decoded.complete.output_logprobs
        assert list(scores.token_logprobs) == []
        assert list(scores.top_logprobs) == []
        assert len(scores.token_ids_logprobs) == 1
        assert list(scores.token_ids_logprobs[0].token_ids) == [42, 7]
        assert list(scores.token_ids_logprobs[0].values) == [-0.5, -30.0]

    asyncio.run(exercise())


def test_batched_scores_keep_each_requests_candidate_set(manager_mod, servicer):
    async def exercise():
        manager, first = _manager_and_state(manager_mod)
        _, second = _manager_and_state(manager_mod)
        second.request_id = "other"
        second.obj.token_ids_logprob = [90, 2, 11]
        manager.rid_to_state["other"] = second
        await manager._handle_batch_output(
            _output(
                values=[],
                rids=["other", "score"],
                output_ids=[[], []],
                finished_reasons=[{"type": "length"}, {"type": "length"}],
                prompt_tokens=[4, 3],
                completion_tokens=[0, 0],
                cached_tokens=[0, 0],
                output_token_ids_logprobs_val=[[[-2.0, -6.0, -4.0]], [[-0.5, -7.0]]],
                output_token_ids_logprobs_idx=[[[90, 2, 11]], [[42, 7]]],
            )
        )
        first_response = servicer._create_completion_response("score", await first.out_queue.get())
        second_response = servicer._create_completion_response(
            "other", await second.out_queue.get()
        )

        first_row = first_response.complete.output_logprobs.token_ids_logprobs[0]
        second_row = second_response.complete.output_logprobs.token_ids_logprobs[0]
        assert list(first_row.token_ids) == [42, 7]
        assert list(first_row.values) == [-0.5, -7.0]
        assert list(second_row.token_ids) == [90, 2, 11]
        assert list(second_row.values) == [-2.0, -6.0, -4.0]

    asyncio.run(exercise())


@pytest.mark.parametrize("malformed_first", [True, False])
@pytest.mark.parametrize(
    ("bad_values", "bad_ids"),
    [([None], [[42, 7]]), ([[-0.5, -7.0]], [None]), (7, [[42, 7]])],
)
def test_malformed_score_row_does_not_strand_valid_sibling(
    manager_mod, servicer, malformed_first, bad_values, bad_ids
):
    async def exercise():
        manager, malformed = _manager_and_state(manager_mod)
        _, valid = _manager_and_state(manager_mod)
        valid.request_id = "other"
        manager.rid_to_state["other"] = valid
        rows = [
            ("score", bad_values, bad_ids),
            ("other", [[-0.5, -7.0]], [[42, 7]]),
        ]
        if not malformed_first:
            rows.reverse()
        await manager._handle_batch_output(
            _output(
                values=[],
                rids=[rid for rid, _, _ in rows],
                output_ids=[[], []],
                finished_reasons=[{"type": "length"}, {"type": "length"}],
                prompt_tokens=[3, 3],
                completion_tokens=[0, 0],
                cached_tokens=[0, 0],
                output_token_ids_logprobs_val=[values for _, values, _ in rows],
                output_token_ids_logprobs_idx=[ids for _, _, ids in rows],
            )
        )

        failed = await asyncio.wait_for(malformed.out_queue.get(), timeout=1)
        succeeded = await asyncio.wait_for(valid.out_queue.get(), timeout=1)
        assert failed["finished"] and malformed.finished
        assert "selected-token logprobs" in failed["error"]
        assert failed["meta_info"]["finish_reason"]["status_code"] == 500
        assert malformed.output_token_ids_logprobs_val == []
        assert malformed.output_token_ids_logprobs_idx == []
        assert "error" not in succeeded
        assert succeeded["finished"] and valid.finished
        response = servicer._create_completion_response("other", succeeded).complete
        assert list(response.output_logprobs.token_ids_logprobs[0].values) == [-0.5, -7.0]
        assert list(response.output_logprobs.token_ids_logprobs[0].token_ids) == [42, 7]

    asyncio.run(exercise())


def test_ordinary_logprobs_survive_without_selected_columns(manager_mod, servicer):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        state.obj.token_ids_logprob = None
        state.obj.top_logprobs_num = 2
        state.obj.logprob_start_len = 0
        batch = _output(
            values=[],
            tokens=[9],
            input_token_logprobs_val=[[None, -1.0]],
            input_token_logprobs_idx=[[1, 2]],
            input_top_logprobs_val=[[[], [-1.0, -2.0]]],
            input_top_logprobs_idx=[[[], [2, 3]]],
            output_token_logprobs_val=[[-0.5]],
            output_token_logprobs_idx=[[9]],
            output_top_logprobs_val=[[[-0.5, -2.0]]],
            output_top_logprobs_idx=[[[9, 8]]],
        )
        del batch.output_token_ids_logprobs_val
        del batch.output_token_ids_logprobs_idx
        await manager._handle_batch_output(batch)
        response = servicer._create_completion_response(
            "score", await state.out_queue.get()
        ).complete

        assert list(response.output_ids) == [9]
        assert list(response.output_logprobs.token_logprobs) == [-0.5]
        assert list(response.output_logprobs.token_ids) == [9]
        assert list(response.output_logprobs.top_logprobs[0].values) == [-0.5, -2.0]
        assert list(response.output_logprobs.top_logprobs[0].token_ids) == [9, 8]
        assert list(response.output_logprobs.token_ids_logprobs) == []
        assert not response.input_logprobs.token_logprobs[0].HasField("value")
        assert response.input_logprobs.token_logprobs[1].value == -1.0
        assert list(response.input_logprobs.top_logprobs[1].token_ids) == [2, 3]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("values", "ids"),
    [
        ([[[-0.5, -7.0]], [[-1.0, -2.0]]], [[[42, 7]]]),
        ([[[-0.5, -7.0], [-1.0, -2.0]]], [[[42, 7]]]),
        ([[[-0.5]]], [[[42, 7]]]),
        (None, [[[42, 7]]]),
    ],
)
def test_malformed_selected_columns_finish_with_error_without_accumulating(
    manager_mod, values, ids
):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        sent = []

        async def send_to_scheduler(request):
            sent.append(request)

        manager._send_to_scheduler = send_to_scheduler
        await manager._handle_batch_output(
            _output(
                values=[],
                finished=False,
                output_token_ids_logprobs_val=values,
                output_token_ids_logprobs_idx=ids,
            )
        )
        response = await state.out_queue.get()

        assert "selected-token logprobs" in response["error"]
        assert response["finished"]
        assert response["meta_info"]["finish_reason"]["status_code"] == 500
        assert state.finished
        assert state.output_token_ids_logprobs_val == []
        assert state.output_token_ids_logprobs_idx == []
        assert [request.rid for request in sent] == ["score"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("values", "ids"),
    [([[-0.5, -7.0], [-1.0, -2.0]], [[42, 7]]), ([[-0.5]], [[42, 7]])],
)
def test_proto_converter_refuses_malformed_selected_rows(servicer, values, ids):
    with pytest.raises(ValueError, match="selected-token logprobs"):
        servicer._convert_output_logprobs_to_proto(
            {"token_ids_logprobs_val": values, "token_ids_logprobs_idx": ids}
        )


@pytest.fixture
def wire(monkeypatch):
    return _load(monkeypatch, "_selected_scores_wire", "zmq_msgpack.py")


def test_slim_wire_appends_prefill_score_columns_and_accepts_older_senders(wire):
    slim = wire.BatchTokenIDSlimOutput.from_full(_output(values=[[-0.5, -30.0]]))
    encoded = msgspec.msgpack.encode(slim)
    array = msgspec.msgpack.decode(encoded)

    assert len(array) == 26
    assert array[24:] == [[[[-0.5, -30.0]]], [[[42, 7]]]]
    decoded = msgspec.msgpack.decode(encoded, type=wire.BatchTokenIDSlimOutput)
    assert decoded.output_token_ids_logprobs_val == [[[-0.5, -30.0]]]
    assert decoded.output_token_ids_logprobs_idx == [[[42, 7]]]

    legacy = msgspec.msgpack.decode(
        msgspec.msgpack.encode(array[:24]), type=wire.BatchTokenIDSlimOutput
    )
    assert legacy.output_token_ids_logprobs_val is None
    assert legacy.output_token_ids_logprobs_idx is None


def test_slim_wire_accepts_engine_output_without_selected_columns(wire):
    output = _output(values=[])
    del output.output_token_ids_logprobs_val
    del output.output_token_ids_logprobs_idx
    slim = wire.BatchTokenIDSlimOutput.from_full(output)

    assert slim.output_token_ids_logprobs_val == [[]]
    assert slim.output_token_ids_logprobs_idx == [[]]

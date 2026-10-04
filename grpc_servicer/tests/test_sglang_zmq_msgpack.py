"""The headless msgpack-over-ZMQ transport for SGLang's scheduler: the wire
shapes SMG relies on, and a loopback through a fake SMG that binds the three
sockets. Needs an SGLang install (its io_struct and sampling params)."""

import array
import os
import struct
import tempfile
import threading
import time

import pytest

# The wire rides on pyzmq and msgspec and is pinned against SGLang's own
# structs. The plain unit-test job installs msgspec but neither pyzmq nor
# SGLang and skips this module; the SGLang ZMQ e2e lane runs it.
msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")
sglang_io = pytest.importorskip("sglang.srt.managers.io_struct")
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from smg_grpc_servicer.sglang import zmq_msgpack as wire  # noqa: E402

AbortReq = sglang_io.AbortReq
BatchTokenIDOutput = sglang_io.BatchTokenIDOutput
TokenizedGenerateReqInput = sglang_io.TokenizedGenerateReqInput
VOCAB = 1000


def _positional_request(rid="r1", input_ids=(1, 2, 3), n=1, stream=True, top_k=-1):
    """What SMG emits: the tagged positional prefix through ``stream`` with a
    nested positional SamplingParams and token ids as a plain list."""
    sampling = SamplingParams(max_new_tokens=8, temperature=0.0, top_k=top_k, n=n)
    sampling_arr = msgspec.msgpack.decode(msgspec.msgpack.encode(sampling))
    return [
        "TokenizedGenerateReqInput",
        rid,
        None,  # http_worker_ipc
        None,  # input_text
        list(input_ids),
        None,  # input_embeds
        None,  # mm_inputs
        None,  # token_type_ids
        sampling_arr,
        False,  # return_logprob
        -1,  # logprob_start_len
        0,  # top_logprobs_num
        None,  # token_ids_logprob
        stream,
    ]


def _wait(predicate, what, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


def test_request_decodes_from_the_frontend_prefix_with_list_ids():
    frames = [wire.REQ_TYPE_ADD, msgspec.msgpack.encode(_positional_request())]
    (req,) = wire.decode_request_frames(frames)
    assert isinstance(req, TokenizedGenerateReqInput)
    assert req.rid == "r1" and req.stream is True
    assert isinstance(req.input_ids, array.array) and list(req.input_ids) == [1, 2, 3]
    assert req.sampling_params.max_new_tokens == 8
    # Trailing fields SMG did not send keep their defaults.
    assert req.require_reasoning is False and req.routed_dp_rank is None


def test_abort_frame_fans_out_one_abort_per_rid():
    aborts = wire.decode_request_frames([wire.REQ_TYPE_ABORT, msgspec.msgpack.encode(["a", "b"])])
    assert [a.rid for a in aborts] == ["a", "b"]
    assert all(isinstance(a, AbortReq) for a in aborts)
    with pytest.raises(ValueError):
        wire.decode_request_frames([b"\x07", b""])


def _full_output():
    none2 = [None, None]
    # Fields this SGLang version has that we do not set stay None (the slim
    # output never reads them; required ones vary across versions).
    unset = {name: None for name in BatchTokenIDOutput.__struct_fields__}
    return BatchTokenIDOutput(
        **unset
        | dict(
            rids=["a", "b"],
            http_worker_ipcs=none2,
            finished_reasons=[None, {"type": "stop", "matched": 42}],
            decoded_texts=["", ""],
            decode_ids=[array.array("q"), array.array("q")],
            read_offsets=[0, 0],
            output_ids=[array.array("q", [10]), array.array("q", [20, 21])],
            skip_special_tokens=[True, True],
            spaces_between_special_tokens=[True, True],
            no_stop_trim=[False, False],
            prompt_tokens=[3, 4],
            reasoning_tokens=[0, 0],
            completion_tokens=[1, 2],
            cached_tokens=[0, 1],
            input_token_logprobs_val=none2,
            input_token_logprobs_idx=none2,
            output_token_logprobs_val=[None, [-0.5, -0.25]],
            output_token_logprobs_idx=[None, [20, 21]],
            input_top_logprobs_val=none2,
            input_top_logprobs_idx=none2,
            output_top_logprobs_val=[None, [[-0.5, -1.0]]],
            output_top_logprobs_idx=[None, [[21, 22]]],
            input_token_ids_logprobs_val=none2,
            input_token_ids_logprobs_idx=none2,
            output_token_ids_logprobs_val=none2,
            output_token_ids_logprobs_idx=none2,
            output_token_entropy_val=None,
            output_token_sampling_mask=None,
            output_hidden_states=None,
            routed_experts=None,
            indexer_topk=None,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    )


def test_slim_output_pins_its_positional_layout():
    slim = wire.BatchTokenIDSlimOutput.from_full(
        _full_output(),
        engine_index=1,
        num_running=2,
        num_waiting=3,
        kv_used_tokens=40,
        kv_total_tokens=400,
    )
    arr = msgspec.msgpack.decode(msgspec.msgpack.encode(slim))
    assert arr == [
        "BatchTokenIDSlimOutput",
        ["a", "b"],  # rids
        [[10], [20, 21]],  # output_ids as plain lists
        ["", "stop"],  # finished_reasons
        [None, None],  # finished_messages
        [None, 42],  # finished_matched
        [3, 4],  # prompt_tokens
        [1, 2],  # completion_tokens
        [0, 1],  # cached_tokens
        [[], [-0.5, -0.25]],  # output_token_logprobs_val
        [[], [20, 21]],  # output_token_logprobs_idx
        1,
        2,
        3,
        40,
        400,
        [None, None],  # finished_status: no abort
        [[], [[-0.5, -1.0]]],  # output_top_logprobs_val: b asked for 2
        [[], [[21, 22]]],  # output_top_logprobs_idx
    ]


class _FakeSmg:
    """Binds handshake ROUTER, input ROUTER and output PULL like the gateway."""

    def __init__(self, context, tmp):
        self.handshake = context.socket(zmq.ROUTER)
        self.handshake.bind("tcp://127.0.0.1:0")
        self.handshake_address = self.handshake.getsockopt_string(zmq.LAST_ENDPOINT)
        self.input = context.socket(zmq.ROUTER)
        self.input_address = f"ipc://{os.path.join(tmp, 'in.sock')}"
        self.input.bind(self.input_address)
        self.output = context.socket(zmq.PULL)
        self.output_address = f"ipc://{os.path.join(tmp, 'out.sock')}"
        self.output.bind(self.output_address)

    def accept(self):
        identity, hello = self.handshake.recv_multipart()
        assert msgspec.msgpack.decode(hello)["status"] == "HELLO"
        init = wire.WireHandshakeInitMessage(
            addresses=wire.WireHandshakeAddresses(
                inputs=[self.input_address], outputs=[self.output_address]
            )
        )
        self.handshake.send_multipart([identity, wire.encode(init)])
        identity2, ready = self.handshake.recv_multipart()
        assert identity2 == identity and msgspec.msgpack.decode(ready)["status"] == "READY"
        reg_identity, registration = self.input.recv_multipart()
        assert reg_identity == identity
        return identity, msgspec.msgpack.decode(registration)

    def send(self, identity, type_byte, payload):
        self.input.send_multipart([identity, type_byte, payload])

    def recv_output(self, timeout_ms=5000):
        assert self.output.poll(timeout_ms), "no output from the scheduler side"
        return self.output.recv()

    def close(self):
        for sock in (self.handshake, self.input, self.output):
            sock.close(linger=0)


def test_loopback_handshake_requests_outputs_and_rejections():
    context = zmq.Context()
    with tempfile.TemporaryDirectory() as tmp:
        smg = _FakeSmg(context, tmp)
        sender = wire.MsgpackSendSocket()
        ready = wire.WireEngineCoreReadyResponse(
            max_model_len=4096, num_gpu_blocks=100, block_size=64, vllm_version="sglang-test"
        )
        result = {}

        def engine_side():
            result["recv"] = wire.connect_msgpack_engine(
                context, smg.handshake_address, 3, ready, VOCAB, sender
            )

        thread = threading.Thread(target=engine_side)
        thread.start()
        identity, registration = smg.accept()
        thread.join(timeout=10)
        assert identity == struct.pack("<H", 3)
        assert registration["max_model_len"] == 4096 and registration["block_size"] == 64
        recv = result["recv"]
        assert sender.socket is not None and sender.engine_index == 3

        # A valid request reaches the scheduler, normalized and verified.
        smg.send(identity, wire.REQ_TYPE_ADD, msgspec.msgpack.encode(_positional_request()))
        reqs = []

        def drained():
            reqs.extend(recv.drain(16))
            return bool(reqs)

        _wait(drained, "the request never arrived")
        (req,) = reqs
        assert req.rid == "r1" and req.sampling_params.is_normalized
        assert req.sampling_params.top_k != -1  # greedy collapsed top_k away from the API sentinel

        # n > 1 is SMG's job: the request is answered, not dropped.
        smg.send(
            identity, wire.REQ_TYPE_ADD, msgspec.msgpack.encode(_positional_request(rid="r2", n=2))
        )

        def rejected():
            assert not recv.drain(16), "an invalid request reached the scheduler"
            return smg.output.poll(0) != 0

        _wait(rejected, "no rejection was sent")
        rejection = msgspec.msgpack.decode(smg.recv_output())
        assert rejection[0] == "BatchTokenIDSlimOutput" and rejection[1] == ["r2"]
        assert rejection[3] == ["abort"] and "n=2" in rejection[4][0]

        # A request the scheduler cannot decode is answered too, by its rid.
        broken = _positional_request(rid="r3")
        broken[4] = "not-token-ids"
        smg.send(identity, wire.REQ_TYPE_ADD, msgspec.msgpack.encode(broken))

        def answered():
            assert not recv.drain(16), "an undecodable request reached the scheduler"
            return smg.output.poll(0) != 0

        _wait(answered, "no answer for the undecodable request")
        answer = msgspec.msgpack.decode(smg.recv_output())
        assert answer[1] == ["r3"] and answer[3] == ["abort"] and "decode" in answer[4][0]

        # Aborts fan out per rid.
        smg.send(identity, wire.REQ_TYPE_ABORT, msgspec.msgpack.encode(["r1", "r9"]))
        aborts = []

        def aborted():
            aborts.extend(recv.drain(16))
            return bool(aborts)

        _wait(aborted, "the aborts never arrived")
        assert [a.rid for a in aborts] == ["r1", "r9"]

        # A scheduler-side AbortReq reply becomes the request's terminal output.
        sender.send_output(AbortReq(rid="r1", abort_message="queue full"))
        terminal = msgspec.msgpack.decode(smg.recv_output())
        assert terminal[1] == ["r1"] and terminal[3] == ["abort"] and terminal[4] == ["queue full"]

        # A full batch goes out slimmed, with the load tail from the probe.
        sender.attach(sender.socket, 3, lambda: _Load())
        sender.send_output(_full_output())
        batch = msgspec.msgpack.decode(smg.recv_output())
        assert batch[1] == ["a", "b"] and batch[11:16] == [3, 7, 11, 500, 5000]

        sender.send_engine_dead()
        assert smg.recv_output() == wire.ENGINE_CORE_DEAD
        recv.close()
        sender.close()
        smg.close()
    context.term()


class _Load:
    num_running_reqs = 7
    num_waiting_reqs = 11
    num_used_tokens = 500
    max_total_num_tokens = 5000

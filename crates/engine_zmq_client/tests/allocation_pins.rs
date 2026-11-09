//! What the decode of an output with logprobs, the decode of a step of plain
//! outputs on every wire and the encode of an add-request allocate, pinned at
//! today's counts, so the next change that adds a copy fails a test rather
//! than moving a bench inside its noise. A separate executable, so it can
//! declare the counting allocator the pin needs. `ZMQ_DECODE_COUNTED=<n>`
//! runs the decode `n` times and exits, for a counter-based reading
//! (`perf stat`).

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]
#![expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]

mod common;

use std::hint::black_box;

use bytes::Bytes;
use common::counting_alloc::{bytes, calls, Counting};
use engine_zmq_client::{
    codec::encode_msgpack,
    protocol::{
        sglang::{
            output::BatchTokenIDSlimOutput,
            request::{SglangRequest, TokenizedGenerateReqInput as SglangGenerate},
            SglangLayout, SglangProtocol,
        },
        tokenspeed::{
            output::BatchTokenIDOutSlim, request::TokenizedGenerateReqInput as TokenSpeedGenerate,
            TokenSpeedProtocol,
        },
        trtllm::{request::TrtllmGenerateReqInput, TrtllmProtocol},
        vllm::{
            logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
            output::{
                decode_engine_core_outputs, EngineCoreFinishReason, EngineCoreOutput,
                EngineCoreOutputs, RequestBatchOutputs,
            },
            request::EngineCoreRequest,
            VllmProtocol,
        },
        EngineProtocol,
    },
};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = calls();
    let out = work();
    (out, calls() - before)
}

/// One engine step for one request: one new token with its logprob and five
/// ranked candidates (the sampled token first, as the engine lists them).
fn one_token_five_candidates() -> Vec<Bytes> {
    let entries = (0..6)
        .map(|rank| TokenLogprob {
            token_id: 100 + rank,
            logprob: -0.25 * (rank as f32 + 1.0),
            rank: if rank == 0 { 7 } else { rank },
        })
        .collect();
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "req-1".into(),
            new_token_ids: vec![100],
            new_logprobs: Some(Logprobs {
                positions: vec![PositionLogprobs { entries }],
            }),
            finish_reason: Some(EngineCoreFinishReason::Length),
            ..Default::default()
        }],
        scheduler_stats: None,
        kv_transfer: None,
        timestamp: 0.0,
        finished_requests: None,
    });
    vec![Bytes::from(encode_msgpack(&outputs).expect("encode"))]
}

/// Decoding one output whose logprobs ride inline: the outputs vector, the
/// token ids, the boxed logprobs payload, the three arrays' shape vectors,
/// the three decoded columns, the positions and the entries. The request id
/// is a view of the frame, as are the arrays' raw views, and the arrays'
/// dtypes are borrowed spellings, so none of those allocates (the frame's
/// shared header is made by the first decode of a frame, here the warm one).
/// The three field names a failed decode would name are formatted on the
/// error path alone. Before the view, the request id was one more (thirteen).
#[test]
fn decoding_one_output_with_inline_logprobs_allocates_this_much() {
    let frames = one_token_five_candidates();
    let warm = decode_engine_core_outputs(&frames).expect("decode");
    let (decoded, allocations) = allocations_during(|| decode_engine_core_outputs(&frames));
    let decoded = decoded.expect("decode");
    assert_eq!(decoded, warm);
    let outputs = &decoded.as_request_batch().expect("a batch").outputs;
    assert_eq!(outputs[0].new_logprobs.as_ref().map(Logprobs::len), Some(1));
    eprintln!("allocation calls per decode: {allocations}");
    assert_eq!(allocations, 12, "allocation calls per decode");
}

/// `ZMQ_DECODE_COUNTED=<n>`: the decode `n` times, for a counter-based
/// reading of the instructions per decode (the difference of two counts).
#[test]
fn counted_decodes() {
    let Some(count) = std::env::var("ZMQ_DECODE_COUNTED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return;
    };
    let frames = one_token_five_candidates();
    for _ in 0..count {
        black_box(decode_engine_core_outputs(black_box(&frames)).expect("decode"));
    }
}

/// A request id of the length the gateway mints, numbered.
fn request_id(index: usize) -> String {
    format!("req-0123456789abcdef0123456789abcdef{index:06}")
}

/// One engine step of `outputs` plain outputs (one new token each, no
/// logprobs, none finished) on the vLLM wire.
fn plain_step(outputs: usize) -> Vec<Bytes> {
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: (0..outputs)
            .map(|index| EngineCoreOutput {
                request_id: request_id(index).into(),
                new_token_ids: vec![100 + index as u32],
                ..Default::default()
            })
            .collect(),
        scheduler_stats: None,
        kv_transfer: None,
        timestamp: 0.0,
        finished_requests: None,
    });
    vec![Bytes::from(encode_msgpack(&outputs).expect("encode"))]
}

/// One warm decode, then the counted one: the batch and the allocation calls
/// of one step's decode on wire `P`.
fn decode_allocations<P: EngineProtocol>(frames: &[Bytes], layout: &P::Layout) -> (usize, usize) {
    P::decode_batch(frames, layout).expect("decode");
    let (decoded, allocations) = allocations_during(|| P::decode_batch(frames, layout));
    (decoded.expect("decode").outputs.len(), allocations)
}

/// Decoding a step of plain outputs allocates one block per output (its
/// token ids) and two per step (the outputs vector and the batch): the
/// request id of every output is a view of the message's frame, not a string
/// of its own. One output reads three; thirty-two read thirty-four. Before
/// the view, every output's id was one allocation more (four and sixty-six).
#[test]
fn decoding_a_step_of_plain_outputs_allocates_one_block_per_output() {
    let mut rows = Vec::new();
    for (outputs, expected) in [(1, 3), (32, 34)] {
        let frames = plain_step(outputs);
        let (decoded, allocations) = decode_allocations::<VllmProtocol>(&frames, &());
        assert_eq!(decoded, outputs);
        rows.push((outputs, allocations, expected));
    }
    for (outputs, allocations, _) in &rows {
        eprintln!("allocation calls per decode of {outputs} plain outputs: {allocations}");
    }
    for (outputs, allocations, expected) in rows {
        assert_eq!(
            allocations, expected,
            "allocation calls per decode of {outputs} plain outputs"
        );
    }
}

/// The slim batches of the other two wires carry their request ids as a
/// column: a step of thirty-two plain outputs decodes its ids as views of the
/// frame too, so the column's vector is the one block the ids cost; the rest
/// is the batch's other columns (one vector each, a token vector per output).
/// Before the view, every id was one allocation more (thirty-two more per
/// step).
#[test]
fn decoding_a_slim_step_of_plain_outputs_allocates_no_string_per_id() {
    let outputs = 32;
    let (sglang_allocations, tokenspeed_allocations) = slim_decode_allocations(outputs, "");
    eprintln!(
        "allocation calls per decode of {outputs} plain outputs, slim batches: sglang {sglang_allocations}, tokenspeed {tokenspeed_allocations}"
    );
    assert_eq!(
        sglang_allocations, SGLANG_SLIM_STEP_ALLOCATIONS,
        "sglang slim step of {outputs}"
    );
    assert_eq!(
        tokenspeed_allocations, TOKENSPEED_SLIM_STEP_ALLOCATIONS,
        "tokenspeed slim step of {outputs}"
    );
}

/// A slim step whose thirty-two outputs all finish costs each wire the finish
/// reasons' strings on top of the plain step, and nothing for the finished
/// list: its ids are the outputs' own, shared. Before, every finish copied
/// its id once more (thirty-two more per step).
#[test]
fn decoding_a_slim_step_of_finished_outputs_copies_no_id() {
    let outputs = 32;
    let (sglang_allocations, tokenspeed_allocations) = slim_decode_allocations(outputs, "stop");
    eprintln!(
        "allocation calls per decode of {outputs} finished outputs, slim batches: sglang {sglang_allocations}, tokenspeed {tokenspeed_allocations}"
    );
    assert_eq!(
        sglang_allocations, SGLANG_SLIM_FINISHED_STEP_ALLOCATIONS,
        "sglang finished slim step of {outputs}"
    );
    assert_eq!(
        tokenspeed_allocations, TOKENSPEED_SLIM_FINISHED_STEP_ALLOCATIONS,
        "tokenspeed finished slim step of {outputs}"
    );
}

/// The two slim wires' step of `outputs` outputs (one new token each, no
/// logprobs), every output finished with `reason` when it is not empty: the
/// first wire's batch, then the second's, each as its one frame.
fn slim_steps(outputs: usize, reason: &str) -> (Vec<Bytes>, Vec<Bytes>) {
    let reasons: Vec<String> = (0..outputs).map(|_| reason.to_string()).collect();
    let zeros: Vec<u32> = vec![0; outputs];
    let no_logprobs: Vec<Vec<f64>> = vec![Vec::new(); outputs];
    let no_logprob_ids: Vec<Vec<u32>> = vec![Vec::new(); outputs];
    let sglang = BatchTokenIDSlimOutput {
        rids: (0..outputs).map(|index| request_id(index).into()).collect(),
        output_ids: (0..outputs).map(|index| vec![100 + index as u32]).collect(),
        finished_reasons: reasons.clone(),
        finished_messages: vec![None; outputs],
        finished_matched: vec![None; outputs],
        prompt_tokens: zeros.clone(),
        completion_tokens: zeros.clone(),
        cached_tokens: zeros.clone(),
        output_token_logprobs_val: no_logprobs.clone(),
        output_token_logprobs_idx: no_logprob_ids.clone(),
        ..Default::default()
    };
    let tokenspeed = BatchTokenIDOutSlim {
        rids: (0..outputs).map(|index| request_id(index).into()).collect(),
        output_ids: (0..outputs).map(|index| vec![100 + index as u32]).collect(),
        finished_reasons: reasons,
        prompt_tokens: zeros.clone(),
        completion_tokens: zeros.clone(),
        cached_tokens: zeros,
        output_token_logprobs_val: no_logprobs,
        output_token_logprobs_idx: no_logprob_ids,
        ..Default::default()
    };
    (
        vec![Bytes::from(encode_msgpack(&sglang).expect("encode"))],
        vec![Bytes::from(encode_msgpack(&tokenspeed).expect("encode"))],
    )
}

/// The allocation calls of one slim step's decode on each wire, the output
/// count checked on both.
fn slim_decode_allocations(outputs: usize, reason: &str) -> (usize, usize) {
    let (sglang, tokenspeed) = slim_steps(outputs, reason);
    let (decoded, sglang_allocations) =
        decode_allocations::<SglangProtocol>(&sglang, &SglangLayout::default());
    assert_eq!(decoded, outputs);
    let (decoded, tokenspeed_allocations) =
        decode_allocations::<TokenSpeedProtocol>(&tokenspeed, &Default::default());
    assert_eq!(decoded, outputs);
    (sglang_allocations, tokenspeed_allocations)
}

/// What a slim step of thirty-two plain outputs costs on each wire with the
/// ids as frame views: a token vector per output, one vector per filled
/// column (ten on the first wire, eight on the second) and three blocks of
/// the step's own. Before the view every id was one more (seventy-seven and
/// seventy-five).
const SGLANG_SLIM_STEP_ALLOCATIONS: usize = 45;
const TOKENSPEED_SLIM_STEP_ALLOCATIONS: usize = 43;

/// The same step with every output finished: the plain step plus one string
/// per finish reason and the finished list's one vector (45 + 32 + 1 and
/// 43 + 32 + 1). Before the shared ids, every finish copied its id into the
/// finished list once more and the list grew by doubling to thirty-two
/// (thirty-two and three calls more on each wire).
const SGLANG_SLIM_FINISHED_STEP_ALLOCATIONS: usize = 78;
const TOKENSPEED_SLIM_FINISHED_STEP_ALLOCATIONS: usize = 76;

/// The prompt of the encode pins: 8,192 token ids, a long prompt, spread over
/// the id range so the ids take their usual three bytes each.
const PINNED_PROMPT_TOKENS: u32 = 8192;

fn prompt_ids() -> Vec<u32> {
    (0..PINNED_PROMPT_TOKENS)
        .map(|position| 1_000 + (position * 7_919) % 150_000)
        .collect()
}

/// A request id of the length the gateway mints.
const PINNED_REQUEST_ID: &str = "req-0123456789abcdef0123456789abcdef012345";

/// One warm encode, then the counted one: the payload, the allocation calls
/// and the bytes requested by one add-request encode on wire `P`.
fn encode_allocations<P: EngineProtocol>(
    request: &P::Request,
    layout: &P::Layout,
) -> (Vec<u8>, usize, usize) {
    P::encode_add(request, layout).expect("encode");
    let bytes_before = bytes();
    let (encoded, allocations) = allocations_during(|| P::encode_add(request, layout));
    let bytes = bytes() - bytes_before;
    let (payload, aux_frames) = encoded.expect("encode");
    assert!(
        aux_frames.is_empty(),
        "a text request rides in its one frame"
    );
    (payload, allocations, bytes)
}

/// Encoding one add-request of 8,192 token ids allocates its payload buffer
/// once on every wire: the buffer is sized from the prompt before the write.
/// A buffer grown from empty costs one allocation per doubling instead
/// (twelve for this prompt) and asks the allocator for a third more bytes
/// than the payload. The bytes on the wire are the request's plain msgpack
/// encoding either way.
#[test]
fn encoding_an_add_request_allocates_once_on_every_wire() {
    let mut rows = Vec::new();
    let request = EngineCoreRequest {
        request_id: PINNED_REQUEST_ID.to_string(),
        prompt_token_ids: Some(prompt_ids()),
        sampling_params: Some(Default::default()),
        ..Default::default()
    };
    let (payload, allocations, bytes) = encode_allocations::<VllmProtocol>(&request, &());
    assert_eq!(payload, rmp_serde::to_vec_named(&request).expect("encode"));
    rows.push(("vllm", payload.len(), allocations, bytes));

    let request = SglangRequest::Generate(SglangGenerate {
        rid: PINNED_REQUEST_ID.to_string(),
        input_ids: prompt_ids(),
        ..Default::default()
    });
    for tensor_aux_frames in [true, false] {
        let layout = SglangLayout { tensor_aux_frames };
        let (payload, allocations, bytes) = encode_allocations::<SglangProtocol>(&request, &layout);
        assert_eq!(payload, rmp_serde::to_vec_named(&request).expect("encode"));
        let wire = if tensor_aux_frames {
            "sglang, aux-frame layout"
        } else {
            "sglang, inline layout"
        };
        rows.push((wire, payload.len(), allocations, bytes));
    }

    let request = TokenSpeedGenerate {
        rid: PINNED_REQUEST_ID.to_string(),
        input_ids: prompt_ids(),
        ..Default::default()
    };
    let (payload, allocations, bytes) =
        encode_allocations::<TokenSpeedProtocol>(&request, &Default::default());
    assert_eq!(payload, rmp_serde::to_vec_named(&request).expect("encode"));
    rows.push(("tokenspeed", payload.len(), allocations, bytes));

    let request = TrtllmGenerateReqInput {
        rid: PINNED_REQUEST_ID.to_string(),
        input_ids: prompt_ids(),
        ..Default::default()
    };
    let (payload, allocations, bytes) = encode_allocations::<TrtllmProtocol>(&request, &());
    assert_eq!(payload, rmp_serde::to_vec_named(&request).expect("encode"));
    rows.push(("trtllm", payload.len(), allocations, bytes));

    for (wire, payload_bytes, allocations, bytes) in &rows {
        eprintln!(
            "allocation calls per add-request encode, {wire}: {allocations} ({bytes} bytes requested for a {payload_bytes}-byte payload)"
        );
    }
    for (wire, _, allocations, _) in &rows {
        assert_eq!(
            *allocations, 1,
            "allocation calls per add-request encode, {wire}"
        );
    }
}

//! SGLang wire protocol.
//!
//! SGLang's scheduler speaks to its Python tokenizer manager in `msgspec`
//! structs declared `array_like=True`: each rides the wire as a positional
//! msgpack array whose element 0 is the class-name tag string, followed by
//! the fields in declaration order (`python/sglang/srt/managers/io_struct.py`).
//! Field order is the wire contract. The request here is the scheduler's own
//! [`request::TokenizedGenerateReqInput`]; the encoder emits the shortest valid prefix
//! (the required fields), since `msgspec` fills missing trailing fields from
//! their defaults, and the engine normalizes the nested
//! [`sampling::SamplingParams`] itself (sent as its API prefix for the same
//! reason). The plugin declares each struct's field order in its ready
//! response, and [`SglangProtocol::check_layouts`] compares the prefixes the
//! encoders fill against it once, at connect time: a release that moved a
//! field inside a prefix is named there, not refused request by request.
//!
//! The scheduler joins SMG's transport through the SGLang plugin that ships
//! in `smg_grpc_servicer.sglang` (no SGLang change): it dials the handshake,
//! decodes `[type_byte, payload]` requests (`ADD = 0x00`, `ABORT = 0x01`, the
//! abort payload a plain msgpack `list[str]` of request ids; an ADD may carry
//! one more frame, the request's metadata map, see [`request::RequestMeta`])
//! and answers with that plugin's [`output::BatchTokenIDSlimOutput`]: the
//! token columns a frontend that detokenizes itself needs, plus a
//! scheduler-load tail.
//!
//! The positional-array helpers (`next_field`, `expect_tag`, `drain_trailing`)
//! are shared with the TokenSpeed dialect, whose structs follow the same
//! `msgspec` convention; the slim-output decoder
//! ([`output::decode_slim_batch`]) with the TensorRT-LLM dialect, whose
//! engine-side shim answers with the same records.

pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;

use bytes::Bytes;

use crate::{
    codec::{encode_msgpack, OpaqueValue},
    error::Result,
    protocol::{
        handshake::EngineCoreReadyResponse,
        positional::check_prefix,
        sglang::{
            multimodal::{
                DATA_ITEM_WIRE_FIELDS, MULTIMODAL_DATA_ITEM_NAME, MULTIMODAL_PROCESSOR_OUTPUT_NAME,
                PROCESSOR_OUTPUT_WIRE_FIELDS,
            },
            output::{decode_slim_batch, SglangOutput},
            request::{
                RequestMeta, SglangRequest, SglangRequestType, EMBEDDING_WIRE_FIELDS,
                ENCODER_DISPATCH_ERROR_REQ_TAG, ENCODER_DISPATCH_ERROR_WIRE_FIELDS,
                GENERATE_WIRE_FIELDS, TOKENIZED_EMBEDDING_REQ_INPUT_TAG,
                TOKENIZED_GENERATE_REQ_INPUT_TAG,
            },
            sampling::{SAMPLING_PARAMS_NAME, WIRE_FIELDS as SAMPLING_WIRE_FIELDS},
        },
        EngineBatch, EngineProtocol,
    },
};

/// The SGLang engine protocol: drives [`SglangRequest`]s (generation and
/// embedding) and control calls over the shared ZMQ transport and decodes the
/// plugin's tagged slim outputs back.
pub struct SglangProtocol;

impl EngineProtocol for SglangProtocol {
    type Request = SglangRequest;
    type Output = SglangOutput;
    type Layout = ();

    fn add_frame() -> Bytes {
        SglangRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        SglangRequestType::Abort.to_frame()
    }

    fn prompt_tokens(request: &Self::Request) -> u64 {
        match request {
            SglangRequest::Generate(request) => request.input_ids.len() as u64,
            SglangRequest::Embed(request) => request.input_ids.len() as u64,
        }
    }

    fn reported_every_step(request: &Self::Request) -> bool {
        match request {
            // The scheduler's `stream_output_generation` sends a request
            // that does not stream every `DEFAULT_FORCE_STREAM_INTERVAL`
            // (50) decoded tokens and at its end, and a streaming one every
            // `stream_interval` tokens: the request's own, else the server's
            // `--stream-interval` (1 unless raised). Only a streaming request
            // at an interval of one is reported every step.
            SglangRequest::Generate(request) => {
                request.stream
                    && request
                        .sampling_params
                        .stream_interval
                        .is_none_or(|interval| interval <= 1)
            }
            // One forward, one output: nothing to report in between.
            SglangRequest::Embed(_) => true,
        }
    }

    fn request_id(request: &Self::Request) -> &str {
        request.rid()
    }

    fn data_parallel_rank(request: &Self::Request) -> Option<u32> {
        // Rank routing is by ZMQ identity: each scheduler rank dials in with
        // its rank as its engine index, so a request routed to a rank goes
        // out on that engine's socket, the dispatch the engine's own
        // data-parallel controller does for its frontend; the request's
        // `routed_dp_rank` slot carries the same value. Unrouted, the
        // connector picks a rank by load.
        match request {
            SglangRequest::Generate(request) => request.routed_dp_rank,
            SglangRequest::Embed(_) => None,
        }
    }

    fn validate(_request: &Self::Request) -> Result<()> {
        Ok(())
    }

    fn check_layouts(ready: &EngineCoreReadyResponse) -> Result<()> {
        // The plugin declares the scheduler's `__struct_fields__` of the three
        // structs this wire encodes (a tagged struct's tag, element 0, is not
        // a field); the prefixes are the slots the encoders fill.
        check_prefix(
            TOKENIZED_GENERATE_REQ_INPUT_TAG,
            &GENERATE_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        check_prefix(
            TOKENIZED_EMBEDDING_REQ_INPUT_TAG,
            &EMBEDDING_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        check_prefix(
            SAMPLING_PARAMS_NAME,
            &SAMPLING_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        // The multimodal payload's structs: the processor output's prefix through
        // `im_token_id`, the data item whole; a plugin that declares them has
        // them compared too.
        check_prefix(
            MULTIMODAL_PROCESSOR_OUTPUT_NAME,
            &PROCESSOR_OUTPUT_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        check_prefix(
            MULTIMODAL_DATA_ITEM_NAME,
            &DATA_ITEM_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        // The encode dispatch's failure notice, every slot of which is
        // written ([`crate::Client::notify`] carries it under the ADD frame).
        check_prefix(
            ENCODER_DISPATCH_ERROR_REQ_TAG,
            &ENCODER_DISPATCH_ERROR_WIRE_FIELDS,
            &ready.positional_layouts,
        )
    }

    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        let payload = encode_msgpack(request)?;
        // The request's metadata (the caller's trace context) rides as one
        // more frame: the positional array is the scheduler's own struct. Its
        // tensors (`mm_inputs`) ride inline, as SGLang's own encoder writes
        // them: the plugin's decoder resolves no auxiliary frames on this
        // wire. The day it does, their frames go behind the metadata frame,
        // which the plugin reads at a fixed position
        // (`codec::tensor::TensorFields`).
        let meta = match request {
            SglangRequest::Generate(request) => RequestMeta::for_request(request),
            SglangRequest::Embed(_) => None,
        };
        let aux_frames = match meta {
            Some(meta) => vec![Bytes::from(encode_msgpack(&meta)?)],
            None => Vec::new(),
        };
        Ok((payload, aux_frames))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        encode_msgpack(&[request_id])
    }

    fn encode_start_wave(_wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // No wave protocol: SGLang's ranks run independently.
        Ok(None)
    }

    fn encode_utility(
        call_id: i64,
        method: &str,
        args: &[OpaqueValue],
    ) -> Result<Option<(Bytes, Vec<u8>)>> {
        // `[engine, CONTROL, (call_id, method, args)]`: the plugin builds the
        // scheduler's own control request (`FlushCacheReqInput`,
        // `ProfileReq`) from it and answers under the call id.
        Ok(Some((
            SglangRequestType::Control.to_frame(),
            encode_msgpack(&(call_id, method, args))?,
        )))
    }

    fn decode_batch(frames: &[Bytes], _layout: &()) -> Result<EngineBatch<Self::Output>> {
        decode_slim_batch(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::decode_msgpack,
        protocol::{
            sglang::output::{
                BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput, ControlReplySlim, MatchedStop,
            },
            EngineLoad,
        },
    };

    fn slim_batch() -> BatchTokenIDSlimOutput {
        BatchTokenIDSlimOutput {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20, 21]],
            finished_reasons: vec![String::new(), "stop".into()],
            finished_messages: vec![None, None],
            finished_matched: vec![None, Some(MatchedStop::TokenId(42))],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 2],
            cached_tokens: vec![0, 1],
            output_token_logprobs_val: vec![vec![], vec![-0.5, -0.25]],
            output_token_logprobs_idx: vec![vec![], vec![20, 21]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 3,
            kv_used_tokens: 40,
            kv_total_tokens: 400,
            finished_status: vec![None, None],
            ..Default::default()
        }
    }

    /// Which requests the engine reports every step, for the silence count:
    /// a streaming generation at an interval of one (the request's own, or
    /// the server's default when unset) and an embedding; a generation that
    /// does not stream, or streams every so many tokens, is reported at
    /// intervals. Every prompt counts as pending prefill.
    #[test]
    fn a_streaming_generation_at_interval_one_or_an_embedding_is_reported_every_step() {
        use crate::protocol::sglang::{
            request::{TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
            sampling::SamplingParams,
        };
        let generate = |stream: bool, stream_interval: Option<u32>| {
            SglangRequest::Generate(TokenizedGenerateReqInput {
                input_ids: vec![1, 2, 3],
                stream,
                sampling_params: SamplingParams {
                    stream_interval,
                    ..SamplingParams::default()
                },
                ..TokenizedGenerateReqInput::default()
            })
        };
        assert!(SglangProtocol::reported_every_step(&generate(true, None)));
        assert!(SglangProtocol::reported_every_step(&generate(
            true,
            Some(1)
        )));
        assert!(!SglangProtocol::reported_every_step(&generate(
            true,
            Some(8)
        )));
        assert!(!SglangProtocol::reported_every_step(&generate(false, None)));
        assert!(!SglangProtocol::reported_every_step(&generate(
            false,
            Some(1)
        )));
        let embed = SglangRequest::Embed(TokenizedEmbeddingReqInput {
            input_ids: vec![1, 2],
            ..TokenizedEmbeddingReqInput::default()
        });
        assert!(SglangProtocol::reported_every_step(&embed));
        assert_eq!(SglangProtocol::prompt_tokens(&generate(false, None)), 3);
        assert_eq!(SglangProtocol::prompt_tokens(&embed), 2);
    }

    #[test]
    fn decode_batch_maps_outputs_finished_ids_and_load() {
        let frames = vec![Bytes::from(encode_msgpack(&slim_batch()).unwrap())];
        let decoded = SglangProtocol::decode_batch(&frames, &()).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.finished_request_ids, vec!["b".to_string()]);
        assert_eq!(decoded.engine_index, 1);
        assert_eq!(
            decoded.load,
            Some(EngineLoad {
                num_running: 2,
                num_waiting: 3,
                kv_cache_usage: 0.1,
            })
        );
        assert_eq!(
            decoded.outputs[1].matched_stop,
            Some(MatchedStop::TokenId(42))
        );
    }

    #[test]
    fn decode_batch_tolerates_aux_frames() {
        let frames = vec![
            Bytes::from(encode_msgpack(&slim_batch()).unwrap()),
            Bytes::from_static(b"aux"),
        ];
        let decoded = SglangProtocol::decode_batch(&frames, &()).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
    }

    #[test]
    fn decode_batch_reports_no_load_without_a_capacity() {
        let batch = BatchTokenIDSlimOutput {
            kv_total_tokens: 0,
            ..slim_batch()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        assert!(SglangProtocol::decode_batch(&frames, &())
            .unwrap()
            .load
            .is_none());
    }

    #[test]
    fn control_calls_ride_the_control_frame_as_a_positional_triple() {
        let (frame, payload) =
            SglangProtocol::encode_utility(7, "flush_cache", &[OpaqueValue::F64(1.5)])
                .unwrap()
                .expect("SGLang has control calls");
        assert_eq!(frame.as_ref(), b"\x02");
        let decoded: (i64, String, Vec<OpaqueValue>) = decode_msgpack(&payload).unwrap();
        assert_eq!(
            decoded,
            (7, "flush_cache".to_string(), vec![OpaqueValue::F64(1.5)])
        );
    }

    #[test]
    fn decode_batch_routes_control_replies_and_embeddings() {
        let reply = ControlReplySlim {
            call_id: 9,
            success: false,
            message: Some("busy".into()),
            engine_index: 1,
        };
        let frames = vec![Bytes::from(encode_msgpack(&reply).unwrap())];
        let batch = SglangProtocol::decode_batch(&frames, &()).unwrap();
        assert!(batch.outputs.is_empty() && batch.load.is_none());
        let utility = batch.utility.expect("a utility reply");
        assert_eq!(utility.call_id, 9);
        let map = utility.outcome.unwrap();
        assert_eq!(
            map.as_map().unwrap()[0],
            (OpaqueValue::from("success"), OpaqueValue::Boolean(false))
        );
        let embeddings = BatchEmbeddingSlimOutput {
            rids: vec!["e1".into()],
            embeddings: vec![vec![0.5]],
            prompt_tokens: vec![2],
            cached_tokens: vec![0],
            finished_reasons: vec!["stop".into()],
            finished_messages: vec![None],
            finished_status: vec![None],
            engine_index: 0,
            num_running: 1,
            num_waiting: 0,
            kv_used_tokens: 10,
            kv_total_tokens: 100,
        };
        let frames = vec![Bytes::from(encode_msgpack(&embeddings).unwrap())];
        let batch = SglangProtocol::decode_batch(&frames, &()).unwrap();
        assert_eq!(batch.finished_request_ids, vec!["e1".to_string()]);
        assert_eq!(batch.outputs[0].embedding, Some(vec![0.5]));
        assert!(batch.load.is_some());
    }

    /// The handshake check against the field order each recorded release
    /// declares, and against releases that moved a field inside a prefix.
    #[test]
    fn check_layouts_accepts_the_recorded_releases_and_names_a_moved_field() {
        use std::collections::BTreeMap;

        use crate::{
            mock_engine::default_ready_response,
            protocol::sglang::{request::recorded as request, sampling::recorded as sampling},
        };
        let names = |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        let ready = |params: &[&str], generate: &[&str], embed: &[&str]| EngineCoreReadyResponse {
            positional_layouts: BTreeMap::from([
                (SAMPLING_PARAMS_NAME.to_string(), names(params)),
                (
                    TOKENIZED_GENERATE_REQ_INPUT_TAG.to_string(),
                    names(generate),
                ),
                (TOKENIZED_EMBEDDING_REQ_INPUT_TAG.to_string(), names(embed)),
            ]),
            ..default_ready_response()
        };
        assert!(
            SglangProtocol::check_layouts(&default_ready_response()).is_ok(),
            "an engine that declares nothing"
        );
        assert!(SglangProtocol::check_layouts(&ready(
            &sampling::SAMPLING_FIELDS_0_5_21,
            &request::GENERATE_FIELDS_0_5_21,
            &request::EMBEDDING_FIELDS_0_5_21,
        ))
        .is_ok());
        assert!(SglangProtocol::check_layouts(&ready(
            &sampling::SAMPLING_FIELDS_0_5_22,
            &request::GENERATE_FIELDS_0_5_22,
            &request::EMBEDDING_FIELDS_0_5_22,
        ))
        .is_ok());
        // A release that put its new sampling field before `custom_params`.
        let mut moved = sampling::SAMPLING_FIELDS_0_5_22.to_vec();
        moved.swap(25, 26);
        let error = SglangProtocol::check_layouts(&ready(
            &moved,
            &request::GENERATE_FIELDS_0_5_22,
            &request::EMBEDDING_FIELDS_0_5_22,
        ))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(
                "`SamplingParams` field 25 is `custom_params` on this side and `watermark` on \
                 the engine's"
            ),
            "{error}"
        );
        // A release that inserted a request field before `lora_id`.
        let mut generate = request::GENERATE_FIELDS_0_5_22.to_vec();
        generate.insert(21, "session_rank");
        let error = SglangProtocol::check_layouts(&ready(
            &sampling::SAMPLING_FIELDS_0_5_22,
            &generate,
            &request::EMBEDDING_FIELDS_0_5_22,
        ))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(
                "`TokenizedGenerateReqInput` field 21 is `lora_id` on this side and \
                 `session_rank` on the engine's"
            ),
            "{error}"
        );
    }

    /// The multimodal structs at the handshake: the processor output is
    /// compared as the prefix this side sends (a field a release adds after
    /// `im_token_id` passes), the data item whole; a field moved or inserted
    /// inside what is sent is refused by name.
    #[test]
    fn check_layouts_reads_the_multimodal_structs_as_sent() {
        use std::collections::BTreeMap;

        use crate::{
            mock_engine::default_ready_response,
            protocol::sglang::multimodal::recorded as multimodal,
        };
        let names = |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        let ready = |output: &[&str], item: &[&str]| EngineCoreReadyResponse {
            positional_layouts: BTreeMap::from([
                (MULTIMODAL_PROCESSOR_OUTPUT_NAME.to_string(), names(output)),
                (MULTIMODAL_DATA_ITEM_NAME.to_string(), names(item)),
            ]),
            ..default_ready_response()
        };
        let output = multimodal::PROCESSOR_OUTPUT_FIELDS_0_5_22;
        let item = multimodal::DATA_ITEM_FIELDS_0_5_22;
        assert!(SglangProtocol::check_layouts(&ready(&output, &item)).is_ok());
        // A release that adds a token id after `im_end_id`: past the prefix.
        let mut appended = output.to_vec();
        appended.insert(6, "video_start_id");
        assert!(SglangProtocol::check_layouts(&ready(&appended, &item)).is_ok());
        // A release that put a field before `input_ids`: inside the prefix.
        let mut inserted = output.to_vec();
        inserted.insert(1, "input_embeds");
        let error = SglangProtocol::check_layouts(&ready(&inserted, &item))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "`MultimodalProcessorOutput` field 1 is `input_ids` on this side and \
                 `input_embeds` on the engine's"
            ),
            "{error}"
        );
        // A release that moved the item's feature.
        let mut moved = item.to_vec();
        moved.swap(5, 6);
        let error = SglangProtocol::check_layouts(&ready(&output, &moved))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "`MultimodalDataItem` field 5 is `feature` on this side and \
                 `precomputed_embeddings` on the engine's"
            ),
            "{error}"
        );
    }

    #[test]
    fn request_type_frames_and_abort_payload_match_the_plugin() {
        assert_eq!(SglangProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(SglangProtocol::abort_frame().as_ref(), b"\x01");
        let payload = SglangProtocol::encode_abort("req-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["req-1".to_string()]);
    }
}

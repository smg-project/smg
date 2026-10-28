//! TensorRT-LLM wire protocol.
//!
//! The engine joins SMG's transport through the shim that ships beside its
//! LLM API (`tensorrt_llm.grpc.smg.shim`; the same package installed as an
//! overlay works alike): the shim dials the handshake, declares the field
//! order of the structs it decodes (`positional_layouts`), decodes
//! `[type_byte, payload, images...]` requests (`ADD = 0x00`, `ABORT = 0x01`,
//! the abort payload a plain msgpack `list[str]` of request ids;
//! `CONTROL = 0x02` is reserved: the engine's gRPC contract has no call that
//! would send one) and answers with the slim records the SGLang plugin
//! declares ([`crate::protocol::sglang::output::BatchTokenIDSlimOutput`] and
//! [`crate::protocol::sglang::output::ControlReplySlim`]): one wire, one
//! output decoder, two engines.
//!
//! The request is this dialect's own, [`request::TrtllmGenerateReqInput`]: a
//! tagged positional array (`msgspec.Struct(tag=True, array_like=True)` on
//! the shim's side) whose sampling block carries the LLM API's own keyword
//! names, so the shim builds its `SamplingParams(**fields)` without a rename
//! table. Field order is the wire contract (append only); the shim declares
//! both layouts at the handshake and [`TrtllmProtocol::check_layouts`]
//! compares the prefixes the encoder fills against them once per connection.
//! Images ride as auxiliary frames behind the payload, one raw image each,
//! counted by the array's `image_frames`.

pub mod request;

use bytes::Bytes;

use crate::{
    codec::encode_msgpack,
    error::{Error, Result},
    protocol::{
        handshake::EngineCoreReadyResponse,
        positional::check_prefix,
        sglang::output::{decode_slim_batch, SglangOutput},
        trtllm::request::{
            TrtllmGenerateReqInput, TrtllmRequestType, GENERATE_REQ_TAG, GENERATE_WIRE_FIELDS,
            SAMPLING_PARAMS_TAG, SAMPLING_WIRE_FIELDS,
        },
        EngineBatch, EngineProtocol,
    },
};

/// The TensorRT-LLM engine protocol: drives [`TrtllmGenerateReqInput`]s over
/// the shared ZMQ transport and decodes the shim's slim outputs back.
pub struct TrtllmProtocol;

impl EngineProtocol for TrtllmProtocol {
    type Request = TrtllmGenerateReqInput;
    type Output = SglangOutput;
    type Layout = ();

    fn add_frame() -> Bytes {
        TrtllmRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        TrtllmRequestType::Abort.to_frame()
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.rid
    }

    fn prompt_tokens(request: &Self::Request) -> u64 {
        request.input_ids.len() as u64
    }

    fn reported_every_step(request: &Self::Request) -> bool {
        // The shim forwards every result the LLM API yields: a streaming
        // request's on every step that decodes it, the other's once, at its
        // end.
        request.streaming
    }

    fn data_parallel_rank(_request: &Self::Request) -> Option<u32> {
        // Rank routing is by ZMQ identity: each shim dials in with its own
        // engine index and the connector picks one by load.
        None
    }

    fn validate(request: &Self::Request) -> Result<()> {
        // The array announces how many image frames follow it; the frames
        // are the request's `images`. A count the frames do not match would
        // make the shim read another request's frames as this one's.
        if request.image_frames as usize != request.images.len() {
            return Err(Error::Encode {
                target_type: GENERATE_REQ_TAG,
                message: format!(
                    "image_frames is {} but {} image frame(s) are attached",
                    request.image_frames,
                    request.images.len()
                ),
            });
        }
        Ok(())
    }

    fn check_layouts(ready: &EngineCoreReadyResponse) -> Result<()> {
        // The shim declares the `__struct_fields__` of the two structs this
        // wire encodes (a tagged struct's tag, element 0, is not a field);
        // the prefixes are the slots the encoder fills.
        check_prefix(
            GENERATE_REQ_TAG,
            &GENERATE_WIRE_FIELDS,
            &ready.positional_layouts,
        )?;
        check_prefix(
            SAMPLING_PARAMS_TAG,
            &SAMPLING_WIRE_FIELDS,
            &ready.positional_layouts,
        )
    }

    fn encode_add(request: &Self::Request, _layout: &()) -> Result<(Vec<u8>, Vec<Bytes>)> {
        // The payload is the positional array; each image is one auxiliary
        // frame behind it, in the order the array counts them.
        Ok((encode_msgpack(request)?, request.images.clone()))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        encode_msgpack(&[request_id])
    }

    fn encode_start_wave(_wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // No wave protocol: the engine's ranks run independently.
        Ok(None)
    }

    fn decode_batch(frames: &[Bytes], _layout: &()) -> Result<EngineBatch<Self::Output>> {
        decode_slim_batch(frames)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        codec::decode_msgpack,
        mock_engine::default_ready_response,
        protocol::{
            sglang::output::{BatchTokenIDSlimOutput, ControlReplySlim, MatchedStop},
            trtllm::request::TrtllmSamplingParams,
            EngineLoad,
        },
    };

    fn request(streaming: bool, images: Vec<Bytes>) -> TrtllmGenerateReqInput {
        TrtllmGenerateReqInput {
            rid: "r-1".to_string(),
            input_ids: vec![1, 2, 3, 4],
            sampling: TrtllmSamplingParams {
                max_tokens: 4,
                ..TrtllmSamplingParams::default()
            },
            streaming,
            image_frames: images.len() as u32,
            images,
            ..TrtllmGenerateReqInput::default()
        }
    }

    /// Only a streaming request is reported every step; every prompt counts
    /// as pending prefill; no rank is pinned by the request.
    #[test]
    fn a_streaming_request_is_reported_every_step() {
        assert!(TrtllmProtocol::reported_every_step(&request(true, vec![])));
        assert!(!TrtllmProtocol::reported_every_step(&request(
            false,
            vec![]
        )));
        assert_eq!(TrtllmProtocol::prompt_tokens(&request(false, vec![])), 4);
        assert_eq!(TrtllmProtocol::request_id(&request(false, vec![])), "r-1");
        assert!(TrtllmProtocol::data_parallel_rank(&request(true, vec![])).is_none());
    }

    /// The payload is the array; the images are the frames behind it, as
    /// many as the array announces. A count the frames do not match is
    /// refused before anything is sent.
    #[test]
    fn images_ride_as_aux_frames_behind_the_payload() {
        let images = vec![Bytes::from_static(b"img-a"), Bytes::from_static(b"img-b")];
        let request = request(true, images.clone());
        TrtllmProtocol::validate(&request).unwrap();
        let (payload, aux) = TrtllmProtocol::encode_add(&request, &()).unwrap();
        assert_eq!(aux, images);
        let decoded: TrtllmGenerateReqInput = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded.image_frames, 2);
        assert!(decoded.images.is_empty(), "the frames are not in the array");

        let mismatched = TrtllmGenerateReqInput {
            image_frames: 1,
            ..request
        };
        let error = TrtllmProtocol::validate(&mismatched)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("image_frames is 1 but 2 image frame(s) are attached"),
            "{error}"
        );
    }

    #[test]
    fn request_type_frames_and_abort_payload_match_the_shim() {
        assert_eq!(TrtllmProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(TrtllmProtocol::abort_frame().as_ref(), b"\x01");
        let payload = TrtllmProtocol::encode_abort("r-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["r-1".to_string()]);
        assert!(TrtllmProtocol::encode_start_wave(1).unwrap().is_none());
        assert!(TrtllmProtocol::encode_utility(1, "flush_cache", &[])
            .unwrap()
            .is_none());
    }

    /// The shim answers with the slim records: the shared decoder maps the
    /// token batch to outputs, finished ids and the load tail, and a control
    /// reply to the batch's utility reply.
    #[test]
    fn decode_batch_reads_the_slim_records() {
        let batch = BatchTokenIDSlimOutput {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20, 21]],
            finished_reasons: vec![String::new(), "stop".into()],
            finished_messages: vec![None, None],
            finished_matched: vec![None, Some(MatchedStop::TokenId(21))],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 2],
            cached_tokens: vec![0, 0],
            output_token_logprobs_val: vec![vec![], vec![-0.5, -0.25]],
            output_token_logprobs_idx: vec![vec![], vec![20, 21]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 3,
            kv_used_tokens: 40,
            kv_total_tokens: 400,
            finished_status: vec![None, None],
            ..Default::default()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        let decoded = TrtllmProtocol::decode_batch(&frames, &()).unwrap();
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
            Some(MatchedStop::TokenId(21))
        );

        let reply = ControlReplySlim {
            call_id: 4,
            success: true,
            message: None,
            engine_index: 0,
        };
        let frames = vec![Bytes::from(encode_msgpack(&reply).unwrap())];
        let decoded = TrtllmProtocol::decode_batch(&frames, &()).unwrap();
        assert!(decoded.outputs.is_empty());
        assert_eq!(decoded.utility.expect("a utility reply").call_id, 4);
    }

    /// The handshake check: a shim that declares nothing passes; one that
    /// declares the layouts this side fills passes; a field that moved
    /// inside a prefix is named.
    #[test]
    fn check_layouts_accepts_the_declared_layouts_and_names_a_moved_field() {
        let names = |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        let ready = |generate: &[&str], sampling: &[&str]| EngineCoreReadyResponse {
            positional_layouts: BTreeMap::from([
                (GENERATE_REQ_TAG.to_string(), names(generate)),
                (SAMPLING_PARAMS_TAG.to_string(), names(sampling)),
            ]),
            ..default_ready_response()
        };
        assert!(TrtllmProtocol::check_layouts(&default_ready_response()).is_ok());
        assert!(TrtllmProtocol::check_layouts(&ready(
            &GENERATE_WIRE_FIELDS,
            &SAMPLING_WIRE_FIELDS
        ))
        .is_ok());
        let mut longer = SAMPLING_WIRE_FIELDS.to_vec();
        longer.push("a_field_the_shim_appended");
        assert!(
            TrtllmProtocol::check_layouts(&ready(&GENERATE_WIRE_FIELDS, &longer)).is_ok(),
            "an appended field is not in the prefix"
        );
        let mut moved = SAMPLING_WIRE_FIELDS.to_vec();
        moved.swap(1, 2);
        let error = TrtllmProtocol::check_layouts(&ready(&GENERATE_WIRE_FIELDS, &moved))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "`TrtllmSamplingParams` field 1 is `temperature` on this side and `top_p` on the \
                 engine's"
            ),
            "{error}"
        );
        let mut shorter = GENERATE_WIRE_FIELDS.to_vec();
        shorter.pop();
        let error = TrtllmProtocol::check_layouts(&ready(&shorter, &SAMPLING_WIRE_FIELDS))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("`TrtllmGenerateReqInput` has 9 fields on the engine's side"),
            "{error}"
        );
    }
}

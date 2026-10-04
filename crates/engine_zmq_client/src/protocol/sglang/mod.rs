//! SGLang wire protocol.
//!
//! SGLang's scheduler speaks to its Python tokenizer manager in `msgspec`
//! structs declared `array_like=True`: each rides the wire as a positional
//! msgpack array whose element 0 is the class-name tag string, followed by
//! the fields in declaration order (`python/sglang/srt/managers/io_struct.py`).
//! Field order is the wire contract — append only, never reorder. The request
//! here is the scheduler's own [`TokenizedGenerateReqInput`]; the encoder
//! emits the shortest valid prefix (the required fields), since `msgspec`
//! fills missing trailing fields from their defaults, and the engine
//! normalizes the nested [`sampling::SamplingParams`] itself.
//!
//! The scheduler joins SMG's transport through the SGLang plugin that ships
//! in `smg_grpc_servicer.sglang` (no SGLang change): it dials the handshake,
//! decodes `[type_byte, payload]` requests (`ADD = 0x00`, `ABORT = 0x01`, the
//! abort payload a plain msgpack `list[str]` of request ids) and answers with
//! that plugin's [`output::BatchTokenIDSlimOutput`]: the token columns a
//! frontend that detokenizes itself needs, plus a scheduler-load tail.
//!
//! The positional-array helpers (`next_field`, `expect_tag`, `drain_trailing`)
//! are shared with the TokenSpeed dialect, whose structs follow the same
//! `msgspec` convention.

pub mod output;
pub mod request;
pub mod sampling;

use bytes::Bytes;

use crate::{
    codec::{decode_msgpack, encode_msgpack, OpaqueValue},
    error::Result,
    protocol::{
        sglang::{
            output::{SglangOutput, SglangWireOutput},
            request::{SglangRequest, SglangRequestType},
        },
        EngineBatch, EngineLoad, EngineProtocol, UtilityReply,
    },
};

/// The SGLang engine protocol: drives [`SglangRequest`]s (generation and
/// embedding) and control calls over the shared ZMQ transport and decodes the
/// plugin's tagged slim outputs back.
pub struct SglangProtocol;

impl EngineProtocol for SglangProtocol {
    type Request = SglangRequest;
    type Output = SglangOutput;

    fn add_frame() -> Bytes {
        SglangRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        SglangRequestType::Abort.to_frame()
    }

    fn request_id(request: &Self::Request) -> &str {
        request.rid()
    }

    fn data_parallel_rank(_request: &Self::Request) -> Option<u32> {
        // Rank routing is by ZMQ identity: each attention-DP leader dials in
        // with its own engine index and the connector picks one by load. The
        // request's own `routed_dp_rank` slot stays unset.
        None
    }

    fn validate(_request: &Self::Request) -> Result<()> {
        Ok(())
    }

    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        Ok((encode_msgpack(request)?, Vec::new()))
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

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        if frames.len() > 1 {
            tracing::debug!(
                aux_frames = frames.len() - 1,
                "ignoring aux frames on an SGLang output (the slim batches have no tensor fields)"
            );
        }
        let payload = frames.first().map(AsRef::as_ref).unwrap_or_default();
        let (engine_index, load, outputs) = match decode_msgpack::<SglangWireOutput>(payload)? {
            SglangWireOutput::Tokens(batch) => (
                batch.engine_index,
                load_of(
                    batch.num_running,
                    batch.num_waiting,
                    batch.kv_used_tokens,
                    batch.kv_total_tokens,
                ),
                (*batch).into_outputs()?,
            ),
            SglangWireOutput::Embeddings(batch) => (
                batch.engine_index,
                load_of(
                    batch.num_running,
                    batch.num_waiting,
                    batch.kv_used_tokens,
                    batch.kv_total_tokens,
                ),
                batch.into_outputs()?,
            ),
            SglangWireOutput::ControlReply(reply) => {
                return Ok(EngineBatch {
                    engine_index: reply.engine_index,
                    utility: Some(UtilityReply {
                        call_id: reply.call_id,
                        outcome: Ok(reply.into_outcome()),
                    }),
                    ..EngineBatch::default()
                });
            }
        };
        let finished_request_ids = outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .map(|output| output.request_id.clone())
            .collect();
        Ok(EngineBatch {
            engine_index,
            outputs,
            finished_request_ids,
            load,
            wave: None,
            utility: None,
        })
    }
}

/// The load tail as a signal. A zero capacity marks a batch sent before the
/// load probe was wired (or a probe that failed): report no load rather than
/// an empty scheduler that least-loaded selection would trust.
fn load_of(
    num_running: u64,
    num_waiting: u64,
    kv_used_tokens: u64,
    kv_total_tokens: u64,
) -> Option<EngineLoad> {
    (kv_total_tokens > 0).then(|| EngineLoad {
        num_running,
        num_waiting,
        kv_cache_usage: kv_used_tokens as f64 / kv_total_tokens as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::sglang::output::{
        BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput, ControlReplySlim, MatchedStop,
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

    #[test]
    fn decode_batch_maps_outputs_finished_ids_and_load() {
        let frames = vec![Bytes::from(encode_msgpack(&slim_batch()).unwrap())];
        let decoded = SglangProtocol::decode_batch(&frames).unwrap();
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
        let decoded = SglangProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
    }

    #[test]
    fn decode_batch_reports_no_load_without_a_capacity() {
        let batch = BatchTokenIDSlimOutput {
            kv_total_tokens: 0,
            ..slim_batch()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        assert!(SglangProtocol::decode_batch(&frames)
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
        let batch = SglangProtocol::decode_batch(&frames).unwrap();
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
        let batch = SglangProtocol::decode_batch(&frames).unwrap();
        assert_eq!(batch.finished_request_ids, vec!["e1".to_string()]);
        assert_eq!(batch.outputs[0].embedding, Some(vec![0.5]));
        assert!(batch.load.is_some());
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

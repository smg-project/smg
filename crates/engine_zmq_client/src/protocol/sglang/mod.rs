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
    codec::{decode_msgpack, encode_msgpack},
    error::Result,
    protocol::{
        sglang::{
            output::{BatchTokenIDSlimOutput, SglangOutput},
            request::{SglangRequestType, TokenizedGenerateReqInput},
        },
        EngineBatch, EngineLoad, EngineProtocol,
    },
};

/// The SGLang engine protocol: drives [`TokenizedGenerateReqInput`] over the
/// shared ZMQ transport and decodes [`BatchTokenIDSlimOutput`] back.
pub struct SglangProtocol;

impl EngineProtocol for SglangProtocol {
    type Request = TokenizedGenerateReqInput;
    type Output = SglangOutput;

    fn add_frame() -> Bytes {
        SglangRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        SglangRequestType::Abort.to_frame()
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.rid
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

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        if frames.len() > 1 {
            tracing::debug!(
                aux_frames = frames.len() - 1,
                "ignoring aux frames on an SGLang output (the slim batch has no tensor fields)"
            );
        }
        let payload = frames.first().map(AsRef::as_ref).unwrap_or_default();
        let batch: BatchTokenIDSlimOutput = decode_msgpack(payload)?;
        let engine_index = batch.engine_index;
        // A zero capacity marks a batch sent before the load probe was wired
        // (or a probe that failed): report no load rather than an empty
        // scheduler that least-loaded selection would trust.
        let load = (batch.kv_total_tokens > 0).then(|| EngineLoad {
            num_running: batch.num_running,
            num_waiting: batch.num_waiting,
            kv_cache_usage: batch.kv_used_tokens as f64 / batch.kv_total_tokens as f64,
        });
        let outputs = batch.into_outputs()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::sglang::output::MatchedStop;

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
    fn request_type_frames_and_abort_payload_match_the_plugin() {
        assert_eq!(SglangProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(SglangProtocol::abort_frame().as_ref(), b"\x01");
        let payload = SglangProtocol::encode_abort("req-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["req-1".to_string()]);
    }
}

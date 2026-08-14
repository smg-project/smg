//! TokenSpeed wire protocol.
//!
//! Clean-room port of TokenSpeed's native engine-IPC messages
//! (`runtime/engine/io_struct.py`). Every data-plane message is a Python
//! `msgspec.Struct(tag=True, tag_field="_tag", kw_only=True, array_like=True)`:
//! it rides the wire as a positional msgpack array whose element 0 is the
//! class-name tag string, followed by the fields in declaration order. Field
//! order is the wire contract — append only, never reorder. Decoders here
//! validate the tag and skip trailing fields they do not model (TokenSpeed
//! appends fields over time); the encoder emits the shortest valid prefix,
//! since `msgspec` fills missing trailing fields from their defaults.
//!
//! Only the text-generation path is typed. Per TokenSpeed's `zmq_wire.py`, the
//! request-type frame (a single raw byte ahead of the payload, so the receiver
//! dispatches without decoding) is `ADD = 0x00` / `ABORT = 0x01`, and the
//! abort payload is a plain msgpack `list[str]` of request ids, not a tagged
//! struct.

pub mod output;
pub mod request;
pub mod sampling;

use bytes::Bytes;
use serde::de::{Deserialize, Error as _, IgnoredAny, SeqAccess};

use crate::{
    codec::{decode_msgpack, encode_msgpack},
    error::Result,
    protocol::{
        tokenspeed::{
            output::{BatchTokenIDOutSlim, TokenSpeedOutput},
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
        },
        EngineBatch, EngineLoad, EngineProtocol,
    },
};

/// Read the next positional element, failing loudly when the array is shorter
/// than the modeled prefix (every modeled field is required on decode).
pub(crate) fn next_field<'de, A, T>(
    seq: &mut A,
    name: &'static str,
) -> std::result::Result<T, A::Error>
where
    A: SeqAccess<'de>,
    T: Deserialize<'de>,
{
    seq.next_element::<T>()?
        .ok_or_else(|| A::Error::custom(format!("missing positional field `{name}`")))
}

/// Validate the msgspec tag string at element 0. A wrong tag means the payload
/// is a different message type — fail loudly instead of misreading fields.
pub(crate) fn expect_tag<'de, A>(
    seq: &mut A,
    expected: &'static str,
) -> std::result::Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    let tag: String = next_field(seq, "_tag")?;
    if tag != expected {
        return Err(A::Error::custom(format!(
            "wrong msgspec tag: expected `{expected}`, got `{tag}`"
        )));
    }
    Ok(())
}

/// Drain positional elements beyond the modeled prefix. TokenSpeed appends new
/// fields at the end of its structs, so unknown trailing elements are skipped
/// rather than treated as a decode error.
pub(crate) fn drain_trailing<'de, A>(seq: &mut A) -> std::result::Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

/// The TokenSpeed engine protocol: drives [`TokenizedGenerateReqInput`] over
/// the shared ZMQ transport and decodes [`BatchTokenIDOutSlim`] back.
pub struct TokenSpeedProtocol;

impl EngineProtocol for TokenSpeedProtocol {
    type Request = TokenizedGenerateReqInput;
    type Output = TokenSpeedOutput;

    fn add_frame() -> Bytes {
        TokenSpeedRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        TokenSpeedRequestType::Abort.to_frame()
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.rid
    }

    fn data_parallel_rank(_request: &Self::Request) -> Option<u32> {
        // The TokenSpeed generate request carries no DP-rank field and needs
        // none: rank routing is purely by ZMQ identity — the connector selects
        // a rank and sends on that rank's socket identity. `None` means "never
        // pinned", so every request goes through least-loaded selection.
        None
    }

    fn validate(_request: &Self::Request) -> Result<()> {
        // The tokenized text path has no fields this client cannot represent.
        Ok(())
    }

    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        // Text path carries no aux tensor frames.
        Ok((encode_msgpack(request)?, Vec::new()))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        // The abort payload is a positional msgpack array of request-id strings,
        // decoded by the engine's `decode_abort` (a `msgspec.msgpack.Decoder(
        // list[str])`) into abort requests. A single-element array aborts one rid.
        encode_msgpack(&[request_id.to_string()])
    }

    fn encode_start_wave(_wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // TokenSpeed has no wave protocol: its scheduler never pauses a group
        // of ranks, so there is nothing to wake.
        Ok(None)
    }

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        // Output messages are `[payload, aux...]`. The slim batch carries no
        // tensor fields today, so aux frames are unexpected but not fatal.
        if frames.len() > 1 {
            tracing::debug!(
                aux_frames = frames.len() - 1,
                "ignoring aux frames on a TokenSpeed output (BatchTokenIDOutSlim \
                 has no tensor fields)"
            );
        }
        let payload = frames.first().map(AsRef::as_ref).unwrap_or_default();
        let batch: BatchTokenIDOutSlim = decode_msgpack(payload)?;
        let engine_index = batch.engine_index;
        // `kv_total_pages == 0` marks a pre-piggyback sender (or a snapshot
        // the engine could not take): report no load rather than fabricating
        // an empty-scheduler signal that least-loaded selection would trust.
        let load = (batch.kv_total_pages > 0).then(|| EngineLoad {
            num_running: batch.num_running,
            num_waiting: batch.num_waiting,
            kv_cache_usage: batch.kv_active_pages as f64 / batch.kv_total_pages as f64,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slim_batch() -> BatchTokenIDOutSlim {
        BatchTokenIDOutSlim {
            rids: vec!["a".into(), "b".into()],
            output_ids: vec![vec![10], vec![20]],
            finished_reasons: vec![String::new(), "stop".into()],
            prompt_tokens: vec![3, 4],
            completion_tokens: vec![1, 1],
            cached_tokens: vec![0, 0],
            output_token_logprobs_val: vec![vec![], vec![]],
            output_token_logprobs_idx: vec![vec![], vec![]],
            engine_index: 1,
            num_running: 2,
            num_waiting: 5,
            kv_active_pages: 100,
            kv_total_pages: 400,
        }
    }

    #[test]
    fn decode_batch_maps_outputs_and_finished_ids() {
        let frames = vec![Bytes::from(encode_msgpack(&slim_batch()).unwrap())];
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.finished_request_ids, vec!["b".to_string()]);
        // The batch names its producing DP rank; the connector routes in-flight
        // release and scoring by it.
        assert_eq!(decoded.engine_index, 1);
        // The piggybacked snapshot surfaces as the engine-neutral load signal.
        assert_eq!(
            decoded.load,
            Some(EngineLoad {
                num_running: 2,
                num_waiting: 5,
                kv_cache_usage: 0.25,
            })
        );
    }

    #[test]
    fn decode_batch_reports_no_load_without_a_snapshot() {
        // kv_total_pages == 0 marks a pre-piggyback sender (or a snapshot the
        // engine could not take): no load, not a fabricated empty scheduler.
        let batch = BatchTokenIDOutSlim {
            kv_total_pages: 0,
            ..slim_batch()
        };
        let frames = vec![Bytes::from(encode_msgpack(&batch).unwrap())];
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
        assert!(decoded.load.is_none());
    }

    #[test]
    fn decode_batch_tolerates_aux_frames() {
        // Slim outputs are single-frame in practice, but the decoder must not
        // choke if the engine ever sends `[payload, aux...]`.
        let frames = vec![
            Bytes::from(encode_msgpack(&slim_batch()).unwrap()),
            Bytes::from_static(b"opaque-aux-frame"),
        ];
        let decoded = TokenSpeedProtocol::decode_batch(&frames).unwrap();
        assert_eq!(decoded.outputs.len(), 2);
    }

    #[test]
    fn request_type_frames_match_wire_contract() {
        assert_eq!(TokenSpeedProtocol::add_frame().as_ref(), b"\x00");
        assert_eq!(TokenSpeedProtocol::abort_frame().as_ref(), b"\x01");
    }

    #[test]
    fn encode_abort_frames_rid_as_positional_array() {
        // The abort payload is a positional msgpack array of request-id strings,
        // matching the engine's `decode_abort` (`Decoder(list[str])`).
        let payload = TokenSpeedProtocol::encode_abort("req-1").unwrap();
        let decoded: Vec<String> = decode_msgpack(&payload).unwrap();
        assert_eq!(decoded, vec!["req-1".to_string()]);
    }
}

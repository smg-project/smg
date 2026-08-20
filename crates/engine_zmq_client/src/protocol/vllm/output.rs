// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/output.rs.
//
// `utility_output` is carried as OpaqueValue (typed utility RPC deferred); the
// semantic classification into RequestBatch / Utility / DpControl is preserved.

use std::collections::BTreeSet;

use bytes::Bytes;
use serde::{Deserialize, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::{decode_msgpack, deserialize_tolerant_seq, OpaqueValue, TrailingTolerant},
    error::{Error, Result},
    protocol::vllm::{
        logprobs::{Logprobs, WireLogprobs},
        stats::{PrefillStats, SchedulerStats},
    },
};

/// The stop reason associated with a finished output. Python models this as
/// `stop_reason: int | str | None`; narrowed here into a tagged enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Reason a request finished. Mirrors Python `FinishReason` (integer-encoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreFinishReason {
    /// A stop string was emitted.
    Stop = 0,
    /// `max_tokens` or `max_model_len` was reached.
    Length = 1,
    /// The request was aborted by the client.
    Abort = 2,
    /// A retryable request-level internal error occurred.
    Error = 3,
    /// A repetitive token pattern was detected.
    Repetition = 4,
}

/// Event types emitted by engine-core for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// A timestamped engine-core event associated with one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCoreEvent {
    pub r#type: EngineCoreEventType,
    pub timestamp: f64,
}

/// Engine-core output for a single request. Mirrors Python `EngineCoreOutput`
/// (`array_like` — field order is the wire contract).
///
/// Logprobs are resolved out of their wire form (aux frames, raw views) by
/// [`decode_engine_core_outputs`], so this type only ever carries decoded
/// [`Logprobs`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions.
    pub new_logprobs: Option<Logprobs>,
    /// Decoded prompt logprobs for the scored prompt positions.
    pub new_prompt_logprobs_tensors: Option<Logprobs>,
    pub pooling_output: Option<OpaqueValue>,
    pub finish_reason: Option<EngineCoreFinishReason>,
    pub stop_reason: Option<StopReason>,
    pub events: Option<Vec<EngineCoreEvent>>,
    pub kv_transfer_params: Option<serde_json::Value>,
    pub ec_transfer_params: Option<serde_json::Value>,
    pub trace_headers: Option<OpaqueValue>,
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    pub prefill_stats: Option<PrefillStats>,
    pub routed_experts: Option<OpaqueValue>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    pub num_nans_in_logits: u32,
    /// Multimodal hashes the engine's receiver cache missed, so the frontend
    /// resends those inputs.
    pub mm_cache_miss_hashes: Option<Vec<String>>,
    /// Updated sampling mask (untyped for now).
    pub new_sampling_mask: Option<OpaqueValue>,
}

impl EngineCoreOutput {
    /// Whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

impl Serialize for EngineCoreOutput {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let new_logprobs = wire_logprobs(self.new_logprobs.as_ref())?;
        let new_prompt_logprobs_tensors = wire_logprobs(self.new_prompt_logprobs_tensors.as_ref())?;
        WireEngineCoreOutputRef {
            request_id: &self.request_id,
            new_token_ids: &self.new_token_ids,
            new_logprobs: new_logprobs.as_ref(),
            new_prompt_logprobs_tensors: new_prompt_logprobs_tensors.as_ref(),
            pooling_output: self.pooling_output.as_ref(),
            finish_reason: self.finish_reason,
            stop_reason: self.stop_reason.as_ref(),
            events: self.events.as_deref(),
            kv_transfer_params: self.kv_transfer_params.as_ref(),
            ec_transfer_params: self.ec_transfer_params.as_ref(),
            trace_headers: self.trace_headers.as_ref(),
            prefill_stats: self.prefill_stats.as_ref(),
            routed_experts: self.routed_experts.as_ref(),
            num_nans_in_logits: self.num_nans_in_logits,
            mm_cache_miss_hashes: self.mm_cache_miss_hashes.as_deref(),
            new_sampling_mask: self.new_sampling_mask.as_ref(),
        }
        .serialize(serializer)
    }
}

/// Encode decoded logprobs back into their wire arrays (send path only: the
/// mock engine and tests).
fn wire_logprobs<E: serde::ser::Error>(
    logprobs: Option<&Logprobs>,
) -> std::result::Result<Option<WireLogprobs>, E> {
    logprobs
        .map(WireLogprobs::from_direct)
        .transpose()
        .map_err(serde::ser::Error::custom)
}

/// Raw Python/msgpack per-request output. Logprobs arrive here in wire form
/// (inline raw views or aux-frame indices) and are resolved before the public
/// [`EngineCoreOutput`] is built.
#[derive(Debug, Clone, PartialEq, Deserialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutput {
    request_id: String,
    new_token_ids: Vec<u32>,
    #[serde(default)]
    new_logprobs: Option<Box<WireLogprobs>>,
    #[serde(default)]
    new_prompt_logprobs_tensors: Option<Box<WireLogprobs>>,
    #[serde(default)]
    pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    stop_reason: Option<StopReason>,
    #[serde(default)]
    events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    kv_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    ec_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    trace_headers: Option<OpaqueValue>,
    #[serde(default)]
    prefill_stats: Option<PrefillStats>,
    #[serde(default)]
    routed_experts: Option<OpaqueValue>,
    #[serde(default)]
    num_nans_in_logits: u32,
    /// Multimodal hashes the engine's receiver cache missed, so the frontend
    /// resends those inputs.
    #[serde(default)]
    mm_cache_miss_hashes: Option<Vec<String>>,
    /// Updated sampling mask (untyped for now).
    #[serde(default)]
    new_sampling_mask: Option<OpaqueValue>,
}

impl WireEngineCoreOutput {
    /// Resolve the wire-format logprobs using the aux frames.
    fn resolve(self, frames: &[Bytes]) -> Result<EngineCoreOutput> {
        Ok(EngineCoreOutput {
            request_id: self.request_id,
            new_token_ids: self.new_token_ids,
            new_logprobs: self
                .new_logprobs
                .map(|value| value.resolve(frames, "new_logprobs"))
                .transpose()?,
            new_prompt_logprobs_tensors: self
                .new_prompt_logprobs_tensors
                .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
                .transpose()?,
            pooling_output: self.pooling_output,
            finish_reason: self.finish_reason,
            stop_reason: self.stop_reason,
            events: self.events,
            kv_transfer_params: self.kv_transfer_params,
            ec_transfer_params: self.ec_transfer_params,
            trace_headers: self.trace_headers,
            prefill_stats: self.prefill_stats,
            routed_experts: self.routed_experts,
            num_nans_in_logits: self.num_nans_in_logits,
            mm_cache_miss_hashes: self.mm_cache_miss_hashes,
            new_sampling_mask: self.new_sampling_mask,
        })
    }
}

/// Borrowed send-side view of [`WireEngineCoreOutput`]; keeps the wire field
/// order without cloning the output being encoded.
#[derive(Serialize_tuple)]
struct WireEngineCoreOutputRef<'a> {
    request_id: &'a str,
    new_token_ids: &'a [u32],
    new_logprobs: Option<&'a WireLogprobs>,
    new_prompt_logprobs_tensors: Option<&'a WireLogprobs>,
    pooling_output: Option<&'a OpaqueValue>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<&'a StopReason>,
    events: Option<&'a [EngineCoreEvent]>,
    kv_transfer_params: Option<&'a serde_json::Value>,
    ec_transfer_params: Option<&'a serde_json::Value>,
    trace_headers: Option<&'a OpaqueValue>,
    prefill_stats: Option<&'a PrefillStats>,
    routed_experts: Option<&'a OpaqueValue>,
    num_nans_in_logits: u32,
    mm_cache_miss_hashes: Option<&'a [String]>,
    new_sampling_mask: Option<&'a OpaqueValue>,
}

/// Raw Python/msgpack engine-core output envelope. Mirrors Python
/// `EngineCoreOutputs` (`array_like`).
#[derive(Debug, Clone, PartialEq, Deserialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutputs {
    #[serde(default)]
    engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default, deserialize_with = "deserialize_tolerant_seq")]
    outputs: Vec<WireEngineCoreOutput>,
    #[serde(default)]
    scheduler_stats: Option<Box<SchedulerStats>>,
    #[serde(default)]
    timestamp: f64,
    /// Utility RPC result (untyped for now).
    #[serde(default)]
    utility_output: Option<OpaqueValue>,
    #[serde(default)]
    finished_requests: Option<BTreeSet<String>>,
    /// In DP mode, signals that the current wave finished and engines are paused.
    #[serde(default)]
    wave_complete: Option<u64>,
    /// In DP mode, signals that a request arrived for an old wave and the next
    /// wave needs to start in other engines.
    #[serde(default)]
    start_wave: Option<u64>,
}

/// Data-parallel control notifications multiplexed through `EngineCoreOutputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DpControlMessage {
    WaveComplete(u64),
    StartWave(u64),
}

/// A batch of per-request outputs plus the piggybacked scheduler stats.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RequestBatchOutputs {
    pub engine_index: u32,
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<Box<SchedulerStats>>,
    pub timestamp: f64,
    pub finished_requests: Option<BTreeSet<String>>,
}

/// A utility RPC result (untyped payload for now).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: Option<OpaqueValue>,
}

/// A DP wave-control notification.
#[derive(Debug, Clone, PartialEq)]
pub struct DpControlOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub control: DpControlMessage,
}

/// Semantic engine-core output families. Python uses one product-shaped wire
/// struct; the Rust protocol exposes the finite semantic families while keeping
/// the same msgpack shape.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineCoreOutputs {
    RequestBatch(RequestBatchOutputs),
    Utility(UtilityCallOutput),
    DpControl(DpControlOutput),
}

impl EngineCoreOutputs {
    /// The request batch, if this is a `RequestBatch` variant.
    pub fn as_request_batch(&self) -> Option<&RequestBatchOutputs> {
        match self {
            Self::RequestBatch(batch) => Some(batch),
            _ => None,
        }
    }

    /// Consume into the request batch, if this is a `RequestBatch` variant.
    pub fn into_request_batch(self) -> Option<RequestBatchOutputs> {
        match self {
            Self::RequestBatch(batch) => Some(batch),
            _ => None,
        }
    }
}

impl From<RequestBatchOutputs> for EngineCoreOutputs {
    fn from(outputs: RequestBatchOutputs) -> Self {
        Self::RequestBatch(outputs)
    }
}

impl From<UtilityCallOutput> for EngineCoreOutputs {
    fn from(output: UtilityCallOutput) -> Self {
        Self::Utility(output)
    }
}

impl From<DpControlOutput> for EngineCoreOutputs {
    fn from(output: DpControlOutput) -> Self {
        Self::DpControl(output)
    }
}

impl WireEngineCoreOutputs {
    /// Classify into the semantic enum, resolving per-request wire logprobs
    /// against the aux `frames`.
    fn into_semantic(self, frames: &[Bytes]) -> Result<EngineCoreOutputs> {
        let value = self;
        let has_request_payload = !value.outputs.is_empty()
            || value.scheduler_stats.is_some()
            || value.finished_requests.is_some();

        match (
            has_request_payload,
            &value.utility_output,
            &value.wave_complete,
            &value.start_wave,
        ) {
            (true, None, None, None) => Ok(RequestBatchOutputs {
                engine_index: value.engine_index,
                outputs: value
                    .outputs
                    .into_iter()
                    .map(|output| output.resolve(frames))
                    .collect::<Result<Vec<_>>>()?,
                scheduler_stats: value.scheduler_stats,
                timestamp: value.timestamp,
                finished_requests: value.finished_requests,
            }
            .into()),
            (false, Some(_), None, None) => Ok(UtilityCallOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                output: value.utility_output,
            }
            .into()),
            (false, None, Some(wave), None) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::WaveComplete(*wave),
            }
            .into()),
            (false, None, None, Some(wave)) => Ok(DpControlOutput {
                engine_index: value.engine_index,
                timestamp: value.timestamp,
                control: DpControlMessage::StartWave(*wave),
            }
            .into()),
            _ => Err(Error::Decode {
                target_type: "EngineCoreOutputs",
                message: "invalid wire shape".to_string(),
            }),
        }
    }
}

/// Borrowed send-side view of [`WireEngineCoreOutputs`]; keeps the wire field
/// order without cloning the batch being encoded.
#[derive(Serialize_tuple)]
struct WireEngineCoreOutputsRef<'a> {
    engine_index: u32,
    outputs: &'a [EngineCoreOutput],
    scheduler_stats: Option<&'a SchedulerStats>,
    timestamp: f64,
    utility_output: Option<&'a OpaqueValue>,
    finished_requests: Option<&'a BTreeSet<String>>,
    wave_complete: Option<u64>,
    start_wave: Option<u64>,
}

impl<'a> From<&'a EngineCoreOutputs> for WireEngineCoreOutputsRef<'a> {
    fn from(value: &'a EngineCoreOutputs) -> Self {
        let empty = Self {
            engine_index: 0,
            outputs: &[],
            scheduler_stats: None,
            timestamp: 0.0,
            utility_output: None,
            finished_requests: None,
            wave_complete: None,
            start_wave: None,
        };
        match value {
            EngineCoreOutputs::RequestBatch(batch) => Self {
                engine_index: batch.engine_index,
                outputs: &batch.outputs,
                scheduler_stats: batch.scheduler_stats.as_deref(),
                timestamp: batch.timestamp,
                finished_requests: batch.finished_requests.as_ref(),
                ..empty
            },
            EngineCoreOutputs::Utility(utility) => Self {
                engine_index: utility.engine_index,
                timestamp: utility.timestamp,
                utility_output: utility.output.as_ref(),
                ..empty
            },
            EngineCoreOutputs::DpControl(control) => {
                let (wave_complete, start_wave) = match control.control {
                    DpControlMessage::WaveComplete(wave) => (Some(wave), None),
                    DpControlMessage::StartWave(wave) => (None, Some(wave)),
                };
                Self {
                    engine_index: control.engine_index,
                    timestamp: control.timestamp,
                    wave_complete,
                    start_wave,
                    ..empty
                }
            }
        }
    }
}

impl Serialize for EngineCoreOutputs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WireEngineCoreOutputsRef::from(self).serialize(serializer)
    }
}

/// Decode one ordinary or multipart engine-core output message into the typed
/// public protocol shape. Frame 0 is the primary msgpack; `frames[1..]` are the
/// ordered aux tensor frames.
pub fn decode_engine_core_outputs(frames: &[Bytes]) -> Result<EngineCoreOutputs> {
    let first_frame = frames.first().ok_or_else(|| Error::ExtValueDecode {
        message: "missing output frame".to_string(),
    })?;

    // Decoded through `TrailingTolerant` so an envelope a newer engine grew
    // past this struct still yields the fields this client knows.
    decode_msgpack::<TrailingTolerant<WireEngineCoreOutputs>>(first_frame)?
        .0
        .into_semantic(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        codec::{
            decode_value, encode_msgpack,
            tensor::{WireArrayData, WireNdArray},
        },
        protocol::vllm::logprobs::{PositionLogprobs, TokenLogprob},
    };

    fn batch(outputs: Vec<EngineCoreOutput>) -> EngineCoreOutputs {
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            outputs,
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        })
    }

    fn encoded_frame(outputs: &EngineCoreOutputs) -> Vec<Bytes> {
        vec![Bytes::from(encode_msgpack(outputs).unwrap())]
    }

    #[test]
    fn engine_core_outputs_roundtrip_finished_fields() {
        let outputs = batch(vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![42],
            finish_reason: Some(EngineCoreFinishReason::Length),
            stop_reason: Some(StopReason::Text("stop".to_string())),
            ..Default::default()
        }]);

        let decoded = decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap();

        assert_eq!(decoded, outputs);
    }

    #[test]
    fn engine_core_outputs_roundtrip_logprobs() {
        let logprobs = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 5,
                        logprob: -0.25,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 6,
                        logprob: -1.5,
                        rank: 1,
                    },
                ],
            }],
        };
        let outputs = batch(vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![5],
            new_logprobs: Some(logprobs.clone()),
            ..Default::default()
        }]);

        let decoded = decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap();

        assert_eq!(
            decoded.as_request_batch().unwrap().outputs[0].new_logprobs,
            Some(logprobs)
        );
    }

    #[test]
    fn decode_resolves_logprobs_from_aux_frames() {
        // Aux frames carry the three logprob arrays; frame 0 is the primary.
        let token_ids = Bytes::from(
            [7_i64, 8]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let logprobs = Bytes::from(
            [-0.5_f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let ranks = Bytes::from(2_i64.to_le_bytes().to_vec());
        let aux = |index| WireNdArray {
            dtype: "<i8".to_string(),
            shape: vec![1, 2],
            data: WireArrayData::AuxIndex(index),
        };
        let wire = WireEngineCoreOutputs {
            outputs: vec![WireEngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                new_logprobs: Some(Box::new(WireLogprobs {
                    logprob_token_ids: aux(1),
                    logprobs: WireNdArray {
                        dtype: "<f4".to_string(),
                        shape: vec![1, 2],
                        data: WireArrayData::AuxIndex(2),
                    },
                    token_ranks: WireNdArray {
                        dtype: "<i8".to_string(),
                        shape: vec![1],
                        data: WireArrayData::AuxIndex(3),
                    },
                    cu_num_generated_tokens: None,
                })),
                ..Default::default()
            }],
            ..Default::default()
        };

        let frames = vec![Bytes::new(), token_ids, logprobs, ranks];
        let decoded = wire.into_semantic(&frames).unwrap();

        let entries = &decoded.as_request_batch().unwrap().outputs[0]
            .new_logprobs
            .as_ref()
            .unwrap()
            .positions[0]
            .entries;
        assert_eq!(entries[0].token_id, 7);
        assert_eq!(entries[0].rank, 2);
        assert_eq!(entries[1].logprob, -2.0);
    }

    #[test]
    fn classify_request_batch() {
        let wire = WireEngineCoreOutputs {
            outputs: vec![WireEngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };
        let classified = wire.into_semantic(&[]).unwrap();
        let batch = classified.as_request_batch().expect("request batch");
        assert_eq!(batch.outputs[0].new_token_ids, vec![7]);
        assert_eq!(
            batch.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }

    #[test]
    fn classify_utility() {
        let wire = WireEngineCoreOutputs {
            utility_output: Some(rmpv::Value::from(42u32)),
            ..Default::default()
        };
        let classified = wire.into_semantic(&[]).unwrap();
        assert!(matches!(classified, EngineCoreOutputs::Utility(_)));
    }

    #[test]
    fn classify_dp_control_start_wave() {
        let wire = WireEngineCoreOutputs {
            start_wave: Some(3),
            ..Default::default()
        };
        let classified = wire.into_semantic(&[]).unwrap();
        assert_eq!(
            classified,
            EngineCoreOutputs::DpControl(DpControlOutput {
                engine_index: 0,
                timestamp: 0.0,
                control: DpControlMessage::StartWave(3),
            })
        );
    }

    #[test]
    fn classify_rejects_mixed_shape() {
        let wire = WireEngineCoreOutputs {
            outputs: vec![WireEngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            utility_output: Some(rmpv::Value::from(1u32)),
            ..Default::default()
        };
        let error = wire.into_semantic(&[]).unwrap_err();
        assert!(error.to_string().contains("invalid wire shape"), "{error}");
    }

    /// The positional array a value encodes to.
    fn wire_array<T: Serialize + std::fmt::Debug>(value: &T) -> Vec<rmpv::Value> {
        match decode_value(&encode_msgpack(value).unwrap()).unwrap() {
            rmpv::Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        }
    }

    fn encode_value(value: &rmpv::Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, value).unwrap();
        bytes
    }

    #[test]
    fn decode_tolerates_wire_arrays_longer_than_the_struct() {
        // A newer engine appends fields to both the envelope and each output;
        // the elements this client knows must still decode.
        let mut output = wire_array(&EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![7],
            mm_cache_miss_hashes: Some(vec!["hash-1".to_string()]),
            ..Default::default()
        });
        output.push(rmpv::Value::from("appended by a newer engine"));

        let mut envelope = wire_array(&batch(Vec::new()));
        envelope[1] = rmpv::Value::Array(vec![rmpv::Value::Array(output)]);
        envelope.push(rmpv::Value::from(true));

        let frame = Bytes::from(encode_value(&rmpv::Value::Array(envelope)));
        let decoded = decode_engine_core_outputs(&[frame])
            .unwrap()
            .into_request_batch()
            .expect("request batch");
        assert_eq!(decoded.outputs[0].new_token_ids, vec![7]);
        assert_eq!(
            decoded.outputs[0].mm_cache_miss_hashes,
            Some(vec!["hash-1".to_string()])
        );
    }

    #[test]
    fn decode_engine_core_outputs_from_single_frame() {
        let outputs = batch(vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![1, 2, 3],
            ..Default::default()
        }]);
        let decoded = decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap();
        assert_eq!(
            decoded.as_request_batch().unwrap().outputs[0].new_token_ids,
            vec![1, 2, 3]
        );
    }
}

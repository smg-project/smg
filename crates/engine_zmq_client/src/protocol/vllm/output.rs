// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/output.rs.
//
// `utility_output` is carried as OpaqueValue (typed utility RPC deferred); the
// semantic classification into RequestBatch / Utility / DpControl is preserved.

use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::{decode_msgpack, OpaqueValue},
    error::{Error, Result},
    protocol::vllm::{
        logprobs::MaybeWireLogprobs,
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
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions.
    #[serde(default)]
    pub new_logprobs: Option<MaybeWireLogprobs>,
    /// Decoded prompt logprobs for the scored prompt positions.
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<MaybeWireLogprobs>,
    #[serde(default)]
    pub pooling_output: Option<OpaqueValue>,
    #[serde(default)]
    pub finish_reason: Option<EngineCoreFinishReason>,
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub events: Option<Vec<EngineCoreEvent>>,
    #[serde(default)]
    pub kv_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub ec_transfer_params: Option<serde_json::Value>,
    #[serde(default)]
    pub trace_headers: Option<OpaqueValue>,
    /// Breakdown of the scheduled prefill computation, set on the first output
    /// of a newly scheduled prefill and elided for subsequent decode outputs.
    #[serde(default)]
    pub prefill_stats: Option<PrefillStats>,
    #[serde(default)]
    pub routed_experts: Option<OpaqueValue>,
    /// Number of NaNs seen in logits. Values above zero indicate corruption.
    #[serde(default)]
    pub num_nans_in_logits: u32,
}

impl EngineCoreOutput {
    /// Whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Resolve wire-format logprobs in-place using the aux frames.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        self.new_logprobs = self
            .new_logprobs
            .take()
            .map(|value| value.resolve(frames, "new_logprobs"))
            .transpose()?;
        self.new_prompt_logprobs_tensors = self
            .new_prompt_logprobs_tensors
            .take()
            .map(|value| value.resolve(frames, "new_prompt_logprobs_tensors"))
            .transpose()?;
        Ok(())
    }
}

/// Raw Python/msgpack engine-core output envelope. Mirrors Python
/// `EngineCoreOutputs` (`array_like`).
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
struct WireEngineCoreOutputs {
    #[serde(default)]
    engine_index: u32,
    /// Outputs grouped for this client in the current engine tick.
    #[serde(default)]
    outputs: Vec<EngineCoreOutput>,
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

    /// Resolve wire-format fields in-place using the aux frames.
    fn resolve_in_place<Frame>(&mut self, frames: &[Frame]) -> Result<()>
    where
        Frame: AsRef<[u8]>,
    {
        if let Self::RequestBatch(batch) = self {
            for output in &mut batch.outputs {
                output.resolve_in_place(frames)?;
            }
        }
        Ok(())
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

/// Classify the raw wire message into the semantic Rust enum.
impl TryFrom<WireEngineCoreOutputs> for EngineCoreOutputs {
    type Error = Error;

    fn try_from(value: WireEngineCoreOutputs) -> Result<Self> {
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
                outputs: value.outputs,
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

impl From<EngineCoreOutputs> for WireEngineCoreOutputs {
    fn from(value: EngineCoreOutputs) -> Self {
        match value {
            EngineCoreOutputs::RequestBatch(batch) => Self {
                engine_index: batch.engine_index,
                outputs: batch.outputs,
                scheduler_stats: batch.scheduler_stats,
                timestamp: batch.timestamp,
                finished_requests: batch.finished_requests,
                ..Default::default()
            },
            EngineCoreOutputs::Utility(utility) => Self {
                engine_index: utility.engine_index,
                timestamp: utility.timestamp,
                utility_output: utility.output,
                ..Default::default()
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
                    ..Default::default()
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
        WireEngineCoreOutputs::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EngineCoreOutputs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireEngineCoreOutputs::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

/// Decode one ordinary or multipart engine-core output message into the typed
/// public protocol shape. Frame 0 is the primary msgpack; `frames[1..]` are the
/// ordered aux tensor frames.
pub fn decode_engine_core_outputs<Frame>(frames: &[Frame]) -> Result<EngineCoreOutputs>
where
    Frame: AsRef<[u8]>,
{
    let first_frame = frames.first().ok_or_else(|| Error::ExtValueDecode {
        message: "missing output frame".to_string(),
    })?;

    let mut outputs: EngineCoreOutputs = decode_msgpack(first_frame.as_ref())?;
    outputs.resolve_in_place(frames)?;
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_msgpack, encode_msgpack};

    #[test]
    fn engine_core_outputs_roundtrip_finished_fields() {
        let outputs = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![42],
                finish_reason: Some(EngineCoreFinishReason::Length),
                stop_reason: Some(StopReason::Text("stop".to_string())),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };

        let encoded = encode_msgpack(&outputs).unwrap();
        let decoded: WireEngineCoreOutputs = decode_msgpack(&encoded).unwrap();

        assert_eq!(decoded.outputs.len(), 1);
        assert_eq!(
            decoded.outputs[0].finish_reason,
            Some(EngineCoreFinishReason::Length)
        );
        assert_eq!(
            decoded.outputs[0].stop_reason,
            Some(StopReason::Text("stop".to_string()))
        );
        assert_eq!(
            decoded.finished_requests,
            Some(BTreeSet::from(["req-1".to_string()]))
        );
    }

    #[test]
    fn classify_request_batch() {
        let wire = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
            ..Default::default()
        };
        let classified = EngineCoreOutputs::try_from(wire).unwrap();
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
        let classified = EngineCoreOutputs::try_from(wire).unwrap();
        assert!(matches!(classified, EngineCoreOutputs::Utility(_)));
    }

    #[test]
    fn classify_dp_control_start_wave() {
        let wire = WireEngineCoreOutputs {
            start_wave: Some(3),
            ..Default::default()
        };
        let classified = EngineCoreOutputs::try_from(wire).unwrap();
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
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            utility_output: Some(rmpv::Value::from(1u32)),
            ..Default::default()
        };
        let error = EngineCoreOutputs::try_from(wire).unwrap_err();
        assert!(error.to_string().contains("invalid wire shape"), "{error}");
    }

    #[test]
    fn decode_engine_core_outputs_from_single_frame() {
        let wire = WireEngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![1, 2, 3],
                ..Default::default()
            }],
            ..Default::default()
        };
        let bytes = encode_msgpack(&wire).unwrap();
        let decoded = decode_engine_core_outputs(&[bytes]).unwrap();
        assert_eq!(
            decoded.as_request_batch().unwrap().outputs[0].new_token_ids,
            vec![1, 2, 3]
        );
    }
}

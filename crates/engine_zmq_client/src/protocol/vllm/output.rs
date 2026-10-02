// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/output.rs.
//
// The semantic classification into RequestBatch / Utility / DpControl is
// preserved; `utility_output` is typed (`UtilityOutput`) so replies route to
// their callers.

use std::collections::BTreeSet;

use bytes::Bytes;
use serde::{Deserialize, Serialize, Serializer};
use serde_default::DefaultFromSerde;
use serde_repr::{Deserialize_repr, Serialize_repr};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::{
        decode_msgpack, deserialize_tolerant_seq, tensor::WireTensor, OpaqueValue, TrailingTolerant,
    },
    error::{Error, Result},
    protocol::vllm::{
        logprobs::{Logprobs, WireLogprobs},
        pooling::PoolingOutput,
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
/// Logprobs and the pooling tensor are resolved out of their wire form (aux
/// frames, raw views) by [`decode_engine_core_outputs`], so this type only
/// ever carries decoded [`Logprobs`] and a resolved [`PoolingOutput`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    /// Decoded sample logprobs for the newly generated positions.
    pub new_logprobs: Option<Logprobs>,
    /// Decoded prompt logprobs for the scored prompt positions.
    pub new_prompt_logprobs_tensors: Option<Logprobs>,
    /// The pooled tensor of a pooling request, set on its finishing output.
    pub pooling_output: Option<PoolingOutput>,
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
    /// Per-request speculative-decoding counters, on the final output only
    /// (vLLM `--per-request-spec-decode-metrics`).
    pub spec_decode_metrics: Option<SpecDecodeMetrics>,
}

/// vLLM's `RequestSpecDecodeMetrics` (a dataclass, so a msgpack map): the
/// histogram of accepted draft tokens per verify step and the drafted total.
/// Informational, so the wire decodes it leniently (see
/// [`WireEngineCoreOutput`]): the per-step detail is optional, and a shape
/// this struct does not know drops the metrics rather than the batch.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecDecodeMetrics {
    pub num_spec_tokens: u32,
    /// Index `j` counts the verify steps that accepted `j` draft tokens.
    pub histogram: Vec<u64>,
    pub num_draft_tokens: u64,
    pub per_step_accepted: Option<Vec<u32>>,
    pub per_step_drafted: Option<Vec<u32>>,
}

/// Decode `spec_decode_metrics` without failing the frame: an explicit nil,
/// a renamed or retyped field in a newer vLLM, or any other mismatch yields
/// `None` for this output; nothing downstream needs the counters.
fn lenient_spec_decode_metrics<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<SpecDecodeMetrics>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<OpaqueValue>::deserialize(deserializer)?;
    Ok(value.and_then(|value| rmpv::ext::from_value(value).ok()))
}

impl SpecDecodeMetrics {
    /// Accepted draft tokens over the request, as the Python servicer sums
    /// them (`sum(j * n for j, n in enumerate(histogram))`).
    pub fn accepted_tokens(&self) -> u64 {
        self.histogram
            .iter()
            .enumerate()
            .map(|(accepted, steps)| accepted as u64 * steps)
            .sum()
    }
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
            pooling_output: self.pooling_output.as_ref().map(PoolingOutput::as_wire),
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
            spec_decode_metrics: self.spec_decode_metrics.as_ref(),
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
    pooling_output: Option<WireTensor>,
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
    #[serde(default, deserialize_with = "lenient_spec_decode_metrics")]
    spec_decode_metrics: Option<SpecDecodeMetrics>,
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
            pooling_output: self
                .pooling_output
                .map(|tensor| PoolingOutput::resolve(tensor, frames))
                .transpose()?,
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
            spec_decode_metrics: self.spec_decode_metrics,
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
    pooling_output: Option<&'a WireTensor>,
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
    spec_decode_metrics: Option<&'a SpecDecodeMetrics>,
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
    /// A utility RPC reply, tolerant of fields a newer engine appends.
    #[serde(default)]
    utility_output: Option<TrailingTolerant<UtilityOutput>>,
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

/// The value a utility method returned, in vLLM's `UtilityResult` wire form
/// `[type_info, value]`: `type_info` is `nil` unless the engine runs with
/// `VLLM_ALLOW_INSECURE_SERIALIZATION` (pickled custom types, not decoded
/// here).
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct UtilityResult {
    pub type_info: OpaqueValue,
    pub value: OpaqueValue,
}

/// An engine's answer to a [`UtilityCall`](super::request::UtilityCall).
/// Mirrors Python `UtilityOutput` (`array_like`): a set `failure_message`
/// means the method raised and `result` is `None`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct UtilityOutput {
    pub call_id: i64,
    #[serde(default)]
    pub failure_message: Option<String>,
    #[serde(default)]
    pub result: Option<UtilityResult>,
}

impl UtilityOutput {
    /// The reply an engine sends for `call_id` (the mock engine's side).
    pub fn from_outcome(call_id: i64, outcome: std::result::Result<OpaqueValue, String>) -> Self {
        match outcome {
            Ok(value) => Self {
                call_id,
                failure_message: None,
                result: Some(UtilityResult {
                    type_info: OpaqueValue::Nil,
                    value,
                }),
            },
            Err(message) => Self {
                call_id,
                failure_message: Some(message),
                result: None,
            },
        }
    }

    /// The call's outcome as vLLM's client resolves it: the failure message
    /// when set, else the returned value.
    pub fn into_outcome(self) -> std::result::Result<OpaqueValue, String> {
        match (self.failure_message, self.result) {
            (Some(message), _) => Err(message),
            (None, Some(result)) => Ok(result.value),
            (None, None) => {
                Err("utility reply carried neither a result nor a failure message".to_string())
            }
        }
    }
}

/// A utility RPC reply, multiplexed on the output wire like a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilityCallOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: UtilityOutput,
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
        let Self {
            engine_index,
            outputs,
            scheduler_stats,
            timestamp,
            utility_output,
            finished_requests,
            wave_complete,
            start_wave,
        } = self;
        let has_request_payload =
            !outputs.is_empty() || scheduler_stats.is_some() || finished_requests.is_some();

        match (
            has_request_payload,
            utility_output,
            wave_complete,
            start_wave,
        ) {
            (true, None, None, None) => Ok(RequestBatchOutputs {
                engine_index,
                outputs: outputs
                    .into_iter()
                    .map(|output| output.resolve(frames))
                    .collect::<Result<Vec<_>>>()?,
                scheduler_stats,
                timestamp,
                finished_requests,
            }
            .into()),
            (false, Some(TrailingTolerant(output)), None, None) => Ok(UtilityCallOutput {
                engine_index,
                timestamp,
                output,
            }
            .into()),
            (false, None, Some(wave), None) => Ok(DpControlOutput {
                engine_index,
                timestamp,
                control: DpControlMessage::WaveComplete(wave),
            }
            .into()),
            (false, None, None, Some(wave)) => Ok(DpControlOutput {
                engine_index,
                timestamp,
                control: DpControlMessage::StartWave(wave),
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
    utility_output: Option<&'a UtilityOutput>,
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
                utility_output: Some(&utility.output),
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
            decode_value, encode_msgpack, hex,
            tensor::{WireArrayData, WireNdArray},
            unhex,
        },
        protocol::vllm::logprobs::{PositionLogprobs, TokenLogprob},
    };

    /// vLLM's own `MsgpackEncoder` bytes (vLLM 0.30.1rc1) for
    /// `EngineCoreOutputs(outputs=[EngineCoreOutput(request_id="emb-1",
    /// new_token_ids=[], pooling_output=tensor([0.25, -1.5, 3.0]),
    /// finish_reason=STOP)], finished_requests={"emb-1"})`: an 18-field
    /// output (two past this struct) with the tensor inline as a raw-view ext.
    const VLLM_POOLING_OUTPUTS_INLINE: &str = "980091dc0012a5656d622d3190c0c093a7666c6f617433329103c70c030000803e0000c0bf0000404000c0c0c0c0c0c0c000c0c0c0c0c0cb415e4c4313f62b42c091a5656d622d31c0c0";
    /// The same for `emb-2` with a 96-float tensor: over vLLM's 256-byte
    /// inline threshold, so the primary frame carries aux index 1 instead.
    const VLLM_POOLING_OUTPUTS_AUX: &str = "980091dc0012a5656d622d3290c0c093a7666c6f6174333291600100c0c0c0c0c0c0c000c0c0c0c0c0cb415e4c4313f6c03dc091a5656d622d32c0c0";

    #[test]
    fn decode_vllm_pooling_output_inline() {
        let frame = Bytes::from(unhex(VLLM_POOLING_OUTPUTS_INLINE));
        let batch = decode_engine_core_outputs(&[frame])
            .unwrap()
            .into_request_batch()
            .expect("request batch");
        assert_eq!(
            batch.finished_requests,
            Some(BTreeSet::from(["emb-1".to_string()]))
        );
        let output = &batch.outputs[0];
        assert_eq!(output.request_id, "emb-1");
        assert!(output.new_token_ids.is_empty());
        // Pooling finishes as STOP on the tick that produced the output.
        assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Stop));
        let pooled = output.pooling_output.as_ref().expect("pooling output");
        assert_eq!(pooled.dtype(), "float32");
        assert_eq!(pooled.shape(), &[3]);
        assert_eq!(pooled.to_vector().unwrap(), vec![0.25, -1.5, 3.0]);
    }

    #[test]
    fn decode_vllm_pooling_output_from_aux_frame() {
        let values: Vec<f32> = (0..96).map(|i| i as f32 / 8.0).collect();
        let aux = Bytes::from(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let frames = vec![Bytes::from(unhex(VLLM_POOLING_OUTPUTS_AUX)), aux.clone()];
        let batch = decode_engine_core_outputs(&frames)
            .unwrap()
            .into_request_batch()
            .expect("request batch");
        let pooled = batch.outputs[0]
            .pooling_output
            .as_ref()
            .expect("pooling output");
        assert_eq!(pooled.shape(), &[96]);
        // Zero-copy: the resolved payload aliases the aux frame.
        assert_eq!(
            pooled.as_wire().data.as_raw_view().unwrap().as_ptr(),
            aux.as_ptr()
        );
        assert_eq!(pooled.to_vector().unwrap(), values);
    }

    #[test]
    fn pooling_output_roundtrips_through_the_mock_send_path() {
        let outputs = batch(vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            pooling_output: Some(PoolingOutput::new(
                WireTensor::from_f32(vec![2], vec![1.0, -2.0]).unwrap(),
            )),
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        }]);
        let decoded = decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap();
        assert_eq!(decoded, outputs);
    }

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

    /// `spec_decode_metrics` is informational: the per-step lists may be nil,
    /// unknown keys are ignored, and a retyped field drops the metrics for
    /// that output without failing the batch.
    #[test]
    fn spec_decode_metrics_decode_leniently() {
        use rmpv::Value;

        let metrics = SpecDecodeMetrics {
            num_spec_tokens: 2,
            histogram: vec![1, 0, 3],
            num_draft_tokens: 8,
            per_step_accepted: Some(vec![0, 2]),
            per_step_drafted: Some(vec![2, 2]),
        };
        let outputs = batch(vec![EngineCoreOutput {
            request_id: "req-1".to_string(),
            new_token_ids: vec![1],
            spec_decode_metrics: Some(metrics.clone()),
            ..Default::default()
        }]);
        let decoded = decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap();
        assert_eq!(
            decoded.as_request_batch().unwrap().outputs[0].spec_decode_metrics,
            Some(metrics)
        );

        // Rewrite the encoded metrics map (the output array's last element).
        type MapEdit = dyn Fn(&mut Vec<(Value, Value)>);
        let re_encode = |edit: &MapEdit| {
            let mut value = decode_value(&encoded_frame(&outputs)[0]).unwrap();
            let Value::Array(top) = &mut value else {
                panic!("array")
            };
            let Value::Array(items) = &mut top[1] else {
                panic!("outputs")
            };
            let Value::Array(fields) = &mut items[0] else {
                panic!("output")
            };
            let Value::Map(map) = fields.last_mut().unwrap() else {
                panic!("metrics map")
            };
            edit(map);
            vec![Bytes::from(encode_msgpack(&value).unwrap())]
        };
        let nil_steps = re_encode(&|map| {
            for (key, value) in map.iter_mut() {
                if key.as_str() == Some("per_step_accepted") {
                    *value = Value::Nil;
                }
            }
            map.push((Value::from("future_field"), Value::from("x")));
        });
        let decoded = decode_engine_core_outputs(&nil_steps).unwrap();
        let lenient = decoded.as_request_batch().unwrap().outputs[0]
            .spec_decode_metrics
            .clone()
            .expect("metrics kept");
        assert_eq!(lenient.histogram, vec![1, 0, 3]);
        assert_eq!(lenient.per_step_accepted, None);
        assert_eq!(lenient.per_step_drafted, Some(vec![2, 2]));

        let retyped = re_encode(&|map| {
            for (key, value) in map.iter_mut() {
                if key.as_str() == Some("histogram") {
                    *value = Value::from("not a list");
                }
            }
        });
        let decoded = decode_engine_core_outputs(&retyped).unwrap();
        let output = &decoded.as_request_batch().unwrap().outputs[0];
        assert_eq!(output.new_token_ids, vec![1]);
        assert_eq!(output.spec_decode_metrics, None);
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
        let output = UtilityOutput::from_outcome(42, Ok(rmpv::Value::from(true)));
        let wire = WireEngineCoreOutputs {
            engine_index: 1,
            utility_output: Some(TrailingTolerant(output.clone())),
            ..Default::default()
        };
        assert_eq!(
            wire.into_semantic(&[]).unwrap(),
            EngineCoreOutputs::Utility(UtilityCallOutput {
                engine_index: 1,
                timestamp: 0.0,
                output,
            })
        );
    }

    /// Golden replies from vLLM's own encoder (`vllm.v1.serial_utils.MsgpackEncoder`,
    /// vLLM 0.30.1rc1): `EngineCoreOutputs(engine_index=1, timestamp=1.5,
    /// utility_output=UtilityOutput(0x0123456789ABCDEF, result=UtilityResult(True)))`,
    /// then the failure and `UtilityResult(None)` shapes.
    #[test]
    fn decode_utility_replies_as_vllm_encodes_them() {
        let call_id = 0x0123_4567_89AB_CDEF;
        let ok = hex("980190c0cb3ff800000000000093cf0123456789abcdefc092c0c3c0c0c0");
        assert_eq!(
            decode_engine_core_outputs(&[Bytes::from(ok)]).unwrap(),
            EngineCoreOutputs::Utility(UtilityCallOutput {
                engine_index: 1,
                timestamp: 1.5,
                output: UtilityOutput {
                    call_id,
                    failure_message: None,
                    result: Some(UtilityResult {
                        type_info: rmpv::Value::Nil,
                        value: rmpv::Value::from(true),
                    }),
                },
            })
        );

        let failed = hex(
            "980090c0cb400400000000000093cf0123456789abcdefd92e43616c6c20746f2072657365745f70\
             72656669785f6361636865206d6574686f64206661696c65643a20626f6f6dc0c0c0c0",
        );
        let EngineCoreOutputs::Utility(reply) =
            decode_engine_core_outputs(&[Bytes::from(failed)]).unwrap()
        else {
            panic!("expected a utility reply");
        };
        assert_eq!(reply.output.call_id, call_id);
        assert_eq!(
            reply.output.into_outcome(),
            Err("Call to reset_prefix_cache method failed: boom".to_string())
        );

        let none = hex("980090c0cb400400000000000093cf0123456789abcdefc092c0c0c0c0c0");
        let EngineCoreOutputs::Utility(reply) =
            decode_engine_core_outputs(&[Bytes::from(none)]).unwrap()
        else {
            panic!("expected a utility reply");
        };
        assert_eq!(reply.output.into_outcome(), Ok(rmpv::Value::Nil));
    }

    /// Our encoding decodes back, including a reserved negative call id (the
    /// notices vLLM sends unprompted on `-1`/`-2` must not fail the message).
    #[test]
    fn utility_replies_roundtrip_including_negative_call_ids() {
        let notice = rmpv::Value::Array(vec![rmpv::Value::from(1), rmpv::Value::from(0)]);
        for output in [
            UtilityOutput::from_outcome(7, Ok(rmpv::Value::from(false))),
            UtilityOutput::from_outcome(-1, Ok(notice)),
            UtilityOutput::from_outcome(3, Err("Server shutting down".to_string())),
        ] {
            let outputs = EngineCoreOutputs::Utility(UtilityCallOutput {
                engine_index: 2,
                timestamp: 4.0,
                output,
            });
            assert_eq!(
                decode_engine_core_outputs(&encoded_frame(&outputs)).unwrap(),
                outputs
            );
        }
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
            utility_output: Some(TrailingTolerant(UtilityOutput::from_outcome(
                1,
                Ok(rmpv::Value::Nil),
            ))),
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

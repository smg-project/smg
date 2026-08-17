//! vLLM EngineCore wire protocol.
//!
//! Clean-room port of vLLM's Apache-2.0 `vllm-engine-core-client`
//! (vllm-project/vllm, `rust/src/engine-core-client/src/protocol`). Struct
//! shapes, field order, and `array_like` positional-tuple encoding are the wire
//! contract with Python `EngineCoreProc` — do not reorder fields.
//!
//! Text generation, structured outputs (guided decoding), and multimodal
//! features are typed fully. Pooling params and prompt embeds are carried as
//! [`crate::codec::OpaqueValue`] — they serialize as `nil` on supported paths.

pub mod logprobs;
pub mod lora;
pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;
pub mod stats;
pub mod structured_outputs;

use bytes::Bytes;

use crate::{
    codec::encode_msgpack,
    error::Result,
    protocol::{
        vllm::{
            output::{
                decode_engine_core_outputs, DpControlMessage, EngineCoreOutput, EngineCoreOutputs,
            },
            request::{EngineCoreRequest, EngineCoreRequestType},
            stats::SchedulerStats,
        },
        EngineBatch, EngineLoad, EngineOutput, EngineProtocol, WaveEvent,
    },
};

/// Map vLLM's piggybacked scheduler stats onto the engine-neutral load signal.
impl From<SchedulerStats> for EngineLoad {
    fn from(stats: SchedulerStats) -> Self {
        Self {
            num_running: stats.num_running_reqs,
            num_waiting: stats.num_waiting_reqs,
            kv_cache_usage: stats.kv_cache_usage,
        }
    }
}

impl EngineOutput for EngineCoreOutput {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn finished(&self) -> bool {
        EngineCoreOutput::finished(self)
    }
}

/// The vLLM EngineCore protocol: drives [`EngineCoreRequest`] over the shared ZMQ
/// transport and decodes [`EngineCoreOutputs`] back.
pub struct VllmProtocol;

impl EngineProtocol for VllmProtocol {
    type Request = EngineCoreRequest;
    type Output = EngineCoreOutput;

    fn add_frame() -> Bytes {
        EngineCoreRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        EngineCoreRequestType::Abort.to_frame()
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }

    fn data_parallel_rank(request: &Self::Request) -> Option<u32> {
        request.data_parallel_rank
    }

    fn validate(request: &Self::Request) -> Result<()> {
        request.validate()
    }

    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        // Text path carries no aux tensor frames.
        Ok((encode_msgpack(request)?, Vec::new()))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        encode_msgpack(&[request_id])
    }

    fn encode_start_wave(wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // Python decodes this with the generic msgpack decoder and unpacks it
        // positionally as `new_wave, exclude_eng_index`. The sentinel matches
        // no rank, so every rank adopts `wave`.
        const NO_EXCLUDED_RANK: u32 = u32::MAX;
        Ok(Some((
            EngineCoreRequestType::StartDpWave.to_frame(),
            encode_msgpack(&(wave, NO_EXCLUDED_RANK))?,
        )))
    }

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        // vLLM multiplexes request batches, utility RPCs, and DP control on one
        // wire struct; only request batches carry per-request outputs (utility
        // results surface as an empty batch the dispatcher ignores).
        match decode_engine_core_outputs(frames)? {
            EngineCoreOutputs::RequestBatch(batch) => Ok(EngineBatch {
                engine_index: batch.engine_index,
                outputs: batch.outputs,
                finished_request_ids: batch
                    .finished_requests
                    .map(|ids| ids.into_iter().collect())
                    .unwrap_or_default(),
                load: batch.scheduler_stats.map(|stats| EngineLoad::from(*stats)),
                wave: None,
            }),
            EngineCoreOutputs::DpControl(control) => Ok(EngineBatch {
                engine_index: control.engine_index,
                wave: Some(match control.control {
                    DpControlMessage::WaveComplete(wave) => WaveEvent::Complete(wave),
                    DpControlMessage::StartWave(wave) => WaveEvent::Start(wave),
                }),
                ..EngineBatch::default()
            }),
            EngineCoreOutputs::Utility(_) => Ok(EngineBatch::default()),
        }
    }
}

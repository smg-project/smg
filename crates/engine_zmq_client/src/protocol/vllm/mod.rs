//! vLLM EngineCore wire protocol.
//!
//! Clean-room port of vLLM's Apache-2.0 `vllm-engine-core-client`
//! (vllm-project/vllm, `rust/src/engine-core-client/src/protocol`). Struct
//! shapes, field order, and `array_like` positional-tuple encoding are the wire
//! contract with Python `EngineCoreProc` — do not reorder fields.
//!
//! Text generation, structured outputs (guided decoding), multimodal features
//! and pooling (embedding) requests are typed fully. Prompt embeds are carried
//! as [`crate::codec::OpaqueValue`] — they serialize as `nil` on supported paths.

pub mod logprobs;
pub mod lora;
pub mod multimodal;
pub mod output;
pub mod pooling;
pub mod request;
pub mod sampling;
pub mod stats;
pub mod structured_outputs;

use bytes::Bytes;

use crate::{
    codec::{encode_msgpack, tensor::WireArrayData, OpaqueValue},
    error::Result,
    protocol::{
        vllm::{
            output::{
                decode_engine_core_outputs, DpControlMessage, EngineCoreOutput, EngineCoreOutputs,
            },
            request::{EngineCoreRequest, EngineCoreRequestType, MmFeaturesPayload, UtilityCall},
            stats::SchedulerStats,
        },
        EngineBatch, EngineLoad, EngineOutput, EngineProtocol, UtilityReply, WaveEvent,
    },
};

/// Map vLLM's piggybacked scheduler stats onto the engine-neutral load signal.
impl From<SchedulerStats> for EngineLoad {
    /// Count both runnable and deferred queues, saturating the waiting total.
    fn from(stats: SchedulerStats) -> Self {
        Self {
            num_running: stats.num_running_reqs,
            num_waiting: stats
                .num_waiting_reqs
                .saturating_add(stats.num_skipped_waiting_reqs),
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

/// Keep small tensors inline where copying costs less than cloning request metadata.
const TENSOR_AUX_MIN_BYTES: usize = 32 * 1024;

impl EngineProtocol for VllmProtocol {
    type Request = EngineCoreRequest;
    type Output = EngineCoreOutput;

    fn add_frame() -> Bytes {
        EngineCoreRequestType::Add.to_frame()
    }

    fn abort_frame() -> Bytes {
        EngineCoreRequestType::Abort.to_frame()
    }

    fn prompt_tokens(request: &Self::Request) -> u64 {
        request
            .prompt_token_ids
            .as_ref()
            .map_or(0, |ids| ids.len() as u64)
    }

    fn reported_every_step(_request: &Self::Request) -> bool {
        // The engine core reports every request it stepped on every step;
        // streaming is the frontend's affair and not on the request.
        true
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

    /// Encode large typed tensors as shared auxiliary frames, preserving caller indices.
    fn encode_add(request: &Self::Request) -> Result<(Vec<u8>, Vec<Bytes>)> {
        let Some(MmFeaturesPayload::Typed(features)) = &request.mm_features else {
            // Raw features already reference the caller's aux frames. Preserve
            // their indices, and keep the text path allocation unchanged.
            return Ok((encode_msgpack(request)?, Vec::new()));
        };
        let mut has_large_tensor = false;
        let has_aux_indices = any_tensor_data(features, &mut |data| match data {
            WireArrayData::AuxIndex(_) => true,
            WireArrayData::RawView(bytes) => {
                has_large_tensor |= bytes.len() >= TENSOR_AUX_MIN_BYTES;
                false
            }
        });
        // Avoid cloning metadata for small requests. Existing indices address
        // caller-provided frames, so preserve the entire encoding in that case.
        if has_aux_indices || !has_large_tensor {
            return Ok((encode_msgpack(request)?, Vec::new()));
        }
        // Tensor storage is Bytes: cloning this metadata shares its buffers.
        let mut wire_request = request.clone();
        let Some(MmFeaturesPayload::Typed(features)) = &mut wire_request.mm_features else {
            return Ok((encode_msgpack(request)?, Vec::new()));
        };
        let mut frames = Vec::new();
        visit_tensor_data(features, &mut |data| {
            if let WireArrayData::RawView(bytes) = data {
                // Large tensors share their storage with separate wire frames.
                if bytes.len() >= TENSOR_AUX_MIN_BYTES {
                    frames.push(bytes.clone());
                    *data = WireArrayData::AuxIndex(frames.len());
                }
            }
        });
        Ok((encode_msgpack(&wire_request)?, frames))
    }

    fn encode_abort(request_id: &str) -> Result<Vec<u8>> {
        encode_msgpack(&[request_id])
    }

    fn encode_start_wave(wave: u64) -> Result<Option<(Bytes, Vec<u8>)>> {
        // Python decodes this with the generic msgpack decoder and unpacks it
        // positionally as `new_wave, exclude_eng_index`. `nil` excludes no
        // rank — the same encoding vLLM's own DP coordinator emits — so every
        // rank adopts `wave`.
        const NO_EXCLUDED_RANK: Option<u32> = None;
        Ok(Some((
            EngineCoreRequestType::StartDpWave.to_frame(),
            encode_msgpack(&(wave, NO_EXCLUDED_RANK))?,
        )))
    }

    fn encode_utility(
        call_id: i64,
        method: &str,
        args: &[OpaqueValue],
    ) -> Result<Option<(Bytes, Vec<u8>)>> {
        // Framed `[engine, UTILITY, (client_index, call_id, method, args)]`,
        // as vLLM's `AsyncMPClient._call_utility_async` sends it; this client
        // has one output socket, so `client_index` is 0.
        Ok(Some((
            EngineCoreRequestType::Utility.to_frame(),
            encode_msgpack(&UtilityCall {
                client_index: 0,
                call_id,
                method: method.to_string(),
                args: args.to_vec(),
            })?,
        )))
    }

    fn decode_batch(frames: &[Bytes]) -> Result<EngineBatch<Self::Output>> {
        // vLLM multiplexes request batches, utility replies, and DP control on
        // one wire struct; each lands in its own slot of the batch.
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
                utility: None,
            }),
            EngineCoreOutputs::DpControl(control) => Ok(EngineBatch {
                engine_index: control.engine_index,
                wave: Some(match control.control {
                    DpControlMessage::WaveComplete(wave) => WaveEvent::Complete(wave),
                    DpControlMessage::StartWave(wave) => WaveEvent::Start(wave),
                }),
                ..EngineBatch::default()
            }),
            EngineCoreOutputs::Utility(reply) => Ok(EngineBatch {
                engine_index: reply.engine_index,
                utility: Some(UtilityReply {
                    call_id: reply.output.call_id,
                    outcome: reply.output.into_outcome(),
                }),
                ..EngineBatch::default()
            }),
        }
    }
}

/// Inspect tensor storage without cloning request metadata or tensor buffers.
fn any_tensor_data(
    features: &multimodal::MmFeatures,
    predicate: &mut impl FnMut(&WireArrayData) -> bool,
) -> bool {
    /// Inspect nested values, stopping at the first matching tensor.
    fn any_value(
        value: &multimodal::MmKwargValue,
        predicate: &mut impl FnMut(&WireArrayData) -> bool,
    ) -> bool {
        match value {
            multimodal::MmKwargValue::Tensor(tensor) => predicate(&tensor.data),
            multimodal::MmKwargValue::List(values) => {
                values.iter().any(|value| any_value(value, predicate))
            }
            multimodal::MmKwargValue::Int(_) | multimodal::MmKwargValue::Float(_) => false,
        }
    }
    features.iter().any(|feature| {
        feature
            .data
            .iter()
            .flat_map(|item| item.values())
            .any(|field| {
                field
                    .data
                    .as_ref()
                    .is_some_and(|value| any_value(value, predicate))
            })
            || feature
                .mm_position
                .is_embed
                .as_ref()
                .is_some_and(|mask| predicate(&mask.data))
    })
}

/// Visit tensor storage in nested feature values and embedding masks.
fn visit_tensor_data(
    features: &mut multimodal::MmFeatures,
    visit: &mut impl FnMut(&mut WireArrayData),
) {
    /// Descend through nested keyword values without changing their structure.
    fn visit_value(
        value: &mut multimodal::MmKwargValue,
        visit: &mut impl FnMut(&mut WireArrayData),
    ) {
        match value {
            multimodal::MmKwargValue::Tensor(tensor) => visit(&mut tensor.data),
            multimodal::MmKwargValue::List(values) => {
                for value in values {
                    visit_value(value, visit);
                }
            }
            multimodal::MmKwargValue::Int(_) | multimodal::MmKwargValue::Float(_) => {}
        }
    }
    for feature in features {
        if let Some(item) = &mut feature.data {
            for field in item.values_mut() {
                if let Some(value) = &mut field.data {
                    visit_value(value, visit);
                }
            }
        }
        if let Some(mask) = &mut feature.mm_position.is_embed {
            visit(&mut mask.data);
        }
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::decode_value;

    /// Build nested large and small tensors plus an embedding mask.
    fn media_request() -> EngineCoreRequest {
        use std::collections::BTreeMap;

        use multimodal::{
            MmBatchedField, MmFeatureSpec, MmField, MmFieldElem, MmKwargValue, PlaceholderRange,
        };

        use crate::codec::tensor::WireTensor;

        let tensor = |n| WireTensor::from_raw("uint8", vec![n], vec![42; n]);
        EngineCoreRequest {
            request_id: "media".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            mm_features: Some(
                vec![MmFeatureSpec {
                    data: Some(BTreeMap::from([(
                        "pixels".into(),
                        MmFieldElem {
                            data: Some(MmKwargValue::List(vec![
                                MmKwargValue::Tensor(tensor(4 * TENSOR_AUX_MIN_BYTES)),
                                MmKwargValue::List(vec![
                                    MmKwargValue::Tensor(tensor(255)),
                                    MmKwargValue::Int(3),
                                ]),
                            ])),
                            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                        },
                    )])),
                    modality: "image".into(),
                    identifier: "image-1".into(),
                    mm_position: PlaceholderRange {
                        offset: 0,
                        length: TENSOR_AUX_MIN_BYTES,
                        is_embed: Some(WireTensor::from_raw(
                            "bool",
                            vec![TENSOR_AUX_MIN_BYTES],
                            vec![1; TENSOR_AUX_MIN_BYTES],
                        )),
                    },
                    mm_hash: None,
                }]
                .into(),
            ),
            ..Default::default()
        }
    }

    /// Preserve values and share large tensor allocations across encoded frames.
    #[test]
    fn typed_media_aux_frames_preserve_nested_tensors_and_masks_without_copying() {
        use crate::codec::{decode_msgpack, tensor::resolve_array_bytes};

        let request = media_request();
        let before = encode_msgpack(&request).unwrap();
        let (payload, aux) = VllmProtocol::encode_add(&request).unwrap();
        assert_eq!(
            aux.iter().map(Bytes::len).collect::<Vec<_>>(),
            vec![4 * TENSOR_AUX_MIN_BYTES, TENSOR_AUX_MIN_BYTES]
        );
        assert!(payload.len() < 1024);
        assert_eq!(before, encode_msgpack(&request).unwrap());
        let mut original = request.clone();
        let Some(MmFeaturesPayload::Typed(features)) = &mut original.mm_features else {
            panic!()
        };
        let mut pointers = Vec::new();
        visit_tensor_data(features, &mut |data| {
            if let WireArrayData::RawView(bytes) = data {
                if bytes.len() >= TENSOR_AUX_MIN_BYTES {
                    pointers.push(bytes.as_ptr());
                }
            }
        });
        assert_eq!(
            pointers,
            aux.iter().map(|bytes| bytes.as_ptr()).collect::<Vec<_>>()
        );
        let mut decoded: EngineCoreRequest = decode_msgpack(&payload).unwrap();
        let mut frames = vec![Bytes::from(payload)];
        frames.extend(aux);
        let Some(MmFeaturesPayload::Typed(features)) = &mut decoded.mm_features else {
            panic!()
        };
        visit_tensor_data(features, &mut |data| {
            *data =
                WireArrayData::RawView(resolve_array_bytes(data.clone(), "test", &frames).unwrap());
        });
        assert_eq!(decoded, request);
    }

    /// Requests containing only small tensors stay byte-identical and inline.
    #[test]
    fn small_tensors_stay_inline() {
        use crate::codec::tensor::WireTensor;

        for size in [0, 255, 256, 4096, TENSOR_AUX_MIN_BYTES - 1] {
            let mut request = media_request();
            let Some(MmFeaturesPayload::Typed(features)) = &mut request.mm_features else {
                panic!()
            };
            let tensor = || WireTensor::from_raw("uint8", vec![size], vec![42; size]);
            features[0].mm_position.is_embed =
                Some(WireTensor::from_raw("bool", vec![size], vec![1; size]));
            features[0].mm_position.length = size;
            for field in features[0].data.as_mut().unwrap().values_mut() {
                field.data = Some(multimodal::MmKwargValue::List(vec![
                    multimodal::MmKwargValue::Tensor(tensor()),
                ]));
            }
            let (payload, aux) = VllmProtocol::encode_add(&request).unwrap();
            assert!(aux.is_empty(), "size={size}");
            assert_eq!(payload, encode_msgpack(&request).unwrap());
        }
    }

    /// Keep externally supplied auxiliary frame addressing unchanged.
    #[test]
    fn existing_aux_indices_and_raw_features_are_not_rebased() {
        let mut request = media_request();
        let Some(MmFeaturesPayload::Typed(features)) = &mut request.mm_features else {
            panic!()
        };
        features[0].mm_position.is_embed.as_mut().unwrap().data = WireArrayData::AuxIndex(1);
        let (payload, frames) = VllmProtocol::encode_add(&request).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&request).unwrap());
        let raw = decode_value(&encode_msgpack(&request.mm_features).unwrap()).unwrap();
        request.mm_features = Some(MmFeaturesPayload::Raw(raw));
        let (payload, frames) = VllmProtocol::encode_add(&request).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&request).unwrap());
    }

    /// Keep text-only requests byte-for-byte identical.
    #[test]
    fn text_request_wire_is_unchanged() {
        let request = EngineCoreRequest::default();
        let (payload, frames) = VllmProtocol::encode_add(&request).unwrap();
        assert_eq!(payload, encode_msgpack(&request).unwrap());
        assert!(frames.is_empty());
    }

    /// Waiting load includes deferred work and saturates instead of wrapping.
    #[test]
    fn scheduler_load_includes_deferred_waiting() {
        for (runnable, deferred, total) in [
            (0, 4, 4),
            (3, 4, 7),
            (3, 0, 3),
            (0, 0, 0),
            (u64::MAX, 1, u64::MAX),
        ] {
            let load = EngineLoad::from(SchedulerStats {
                num_running_reqs: 2,
                num_waiting_reqs: runnable,
                num_skipped_waiting_reqs: deferred,
                kv_cache_usage: 0.5,
                ..Default::default()
            });
            assert_eq!(load.num_waiting, total);
            assert_eq!(load.num_running, 2);
            assert_eq!(load.kv_cache_usage, 0.5);
        }
    }

    /// The engine core reports every request it stepped on every step
    /// (streaming is the frontend's affair and not on the request), so every
    /// request counts toward the silence, and its prompt as pending prefill.
    #[test]
    fn every_request_is_reported_every_step() {
        let request = EngineCoreRequest {
            request_id: "r".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        assert!(VllmProtocol::reported_every_step(&request));
        assert_eq!(VllmProtocol::prompt_tokens(&request), 3);
    }

    #[test]
    fn start_wave_encodes_a_nil_excluded_rank() {
        let (frame, payload) = VllmProtocol::encode_start_wave(9).unwrap().unwrap();
        assert_eq!(frame.as_ref(), b"\x02");
        assert_eq!(
            decode_value(&payload).unwrap(),
            Value::Array(vec![Value::from(9), Value::Nil])
        );
    }
}

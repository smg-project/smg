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
use tracing::warn;

use crate::{
    codec::{
        add_request_capacity, encode_msgpack, tensor::encode_with_aux_frames, FrameStr, OpaqueValue,
    },
    error::Result,
    protocol::{
        handshake::EngineCoreReadyResponse,
        vllm::{
            output::{
                decode_engine_core_outputs, DpControlMessage, EngineCoreOutput, EngineCoreOutputs,
            },
            request::{EngineCoreRequest, EngineCoreRequestType, UtilityCall},
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
            ..Self::default()
        }
    }
}

impl EngineOutput for EngineCoreOutput {
    fn request_id(&self) -> &str {
        self.request_id.as_str()
    }

    fn own_request_id(&mut self, id: FrameStr) {
        self.request_id = id;
    }

    fn request_id_shared(&self) -> FrameStr {
        self.request_id.clone()
    }

    fn finished(&self) -> bool {
        EngineCoreOutput::finished(self)
    }
}

/// How the engine of a connection reads the prompt token ids of an
/// add-request, told by the release its ready response reports
/// ([`EngineCoreReadyResponse::vllm_version`]). The add-request of 0.31 to
/// 0.34 was read against the engine's source, where nothing else this client
/// writes differs; an older release, and a version that names none, keep the
/// list every engine has been sent so far. A release above 0.34 is sent the
/// array of 0.34, and warned about as its engine connects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VllmLayout {
    /// A list of integers: every release before 0.34.
    #[default]
    TokenList,
    /// An `int32` array in the engine's `(dtype, shape, data)` form: 0.34
    /// and every release above it.
    TokenArray,
}

impl VllmLayout {
    /// The newest release whose add-request this client was read against.
    const NEWEST_READ: (u32, u32) = (0, 34);

    /// The release `version` names by its first two parts between dots, each
    /// a number (`0.34.0rc1` names 0.34). `None` for a version that names no
    /// release (a source tree reports `dev`).
    fn release(version: &str) -> Option<(u32, u32)> {
        let mut parts = version.split('.').map(str::parse::<u32>);
        match (parts.next(), parts.next()) {
            (Some(Ok(major)), Some(Ok(minor))) => Some((major, minor)),
            _ => None,
        }
    }

    /// The layout of the release `version` names: the array from 0.34 on,
    /// the list before it. A version that names no release connects as it
    /// always has, and is sent the list.
    fn of_release(version: &str) -> Self {
        match Self::release(version) {
            Some(release) if release >= (0, 34) => Self::TokenArray,
            _ => Self::TokenList,
        }
    }
}

/// The vLLM EngineCore protocol: drives [`EngineCoreRequest`] over the shared ZMQ
/// transport and decodes [`EngineCoreOutputs`] back.
pub struct VllmProtocol;

impl EngineProtocol for VllmProtocol {
    type Request = EngineCoreRequest;
    type Output = EngineCoreOutput;
    type Layout = VllmLayout;

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

    /// Encode the request with its large typed tensors as auxiliary frames
    /// ([`encode_with_aux_frames`]); raw features keep the encoding their
    /// producer gave them, and the text path its one allocation: the payload's
    /// buffer is sized from the prompt before the write.
    fn encode_add(request: &Self::Request, layout: &VllmLayout) -> Result<(Vec<u8>, Vec<Bytes>)> {
        let prompt_tokens = request.prompt_token_ids.as_ref().map_or(0, Vec::len);
        encode_with_aux_frames(
            &request.on_wire(*layout),
            add_request_capacity(prompt_tokens),
        )
    }

    /// The layout of the engine's release. A release above the newest this
    /// client was read against is sent the add-request of that one and warned
    /// about here, once for each engine as it connects: the engine decodes an
    /// add-request outside any handler, so one it cannot decode ends its
    /// input thread and every later request with it.
    fn read_layout(ready: &EngineCoreReadyResponse) -> VllmLayout {
        let version = ready.vllm_version.as_str();
        if VllmLayout::release(version).is_some_and(|release| release > VllmLayout::NEWEST_READ) {
            let (major, minor) = VllmLayout::NEWEST_READ;
            warn!(
                vllm_version = version,
                "the engine's vLLM release is above {major}.{minor}, the newest whose \
                 add-request this client was read against, and is sent the add-request of \
                 {major}.{minor}: if the release changed it, the first request the engine \
                 cannot decode ends its input thread and every later request fails; serve it \
                 with a servicer release that knows it"
            );
        }
        VllmLayout::of_release(version)
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

    /// Decode the batch and retain its engine timestamp on each local output.
    fn decode_batch(frames: &[Bytes], _layout: &VllmLayout) -> Result<EngineBatch<Self::Output>> {
        // vLLM multiplexes request batches, utility replies, and DP control on
        // one wire struct; each lands in its own slot of the batch.
        match decode_engine_core_outputs(frames)? {
            EngineCoreOutputs::RequestBatch(batch) => Ok(EngineBatch {
                engine_index: batch.engine_index,
                outputs: batch
                    .outputs
                    .into_iter()
                    .map(|mut output| {
                        output.batch_timestamp = Some(batch.timestamp);
                        output
                    })
                    .collect(),
                finished_request_ids: batch
                    .finished_requests
                    .map(|ids| ids.into_iter().map(FrameStr::from).collect())
                    .unwrap_or_default(),
                load: batch.scheduler_stats.map(EngineLoad::from),
                kv_transfer: batch.kv_transfer,
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

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::{
        codec::{
            decode_value, hex,
            tensor::{TensorFields, WireArrayData, TENSOR_AUX_MIN_BYTES},
        },
        mock_engine::default_ready_response,
        protocol::vllm::request::MmFeaturesPayload,
    };

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
        let (payload, aux) = VllmProtocol::encode_add(&request, &VllmLayout::TokenList).unwrap();
        assert_eq!(
            aux.iter().map(Bytes::len).collect::<Vec<_>>(),
            vec![4 * TENSOR_AUX_MIN_BYTES, TENSOR_AUX_MIN_BYTES]
        );
        assert!(payload.len() < 1024);
        assert_eq!(before, encode_msgpack(&request).unwrap());
        let mut pointers = Vec::new();
        request.for_each_tensor(&mut |data| {
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
        decoded.for_each_tensor_mut(&mut |data| {
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
            let (payload, aux) =
                VllmProtocol::encode_add(&request, &VllmLayout::TokenList).unwrap();
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
        let (payload, frames) = VllmProtocol::encode_add(&request, &VllmLayout::TokenList).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&request).unwrap());
        let raw = decode_value(&encode_msgpack(&request.mm_features).unwrap()).unwrap();
        request.mm_features = Some(MmFeaturesPayload::Raw(raw));
        let (payload, frames) = VllmProtocol::encode_add(&request, &VllmLayout::TokenList).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&request).unwrap());
    }

    /// Keep text-only requests byte-for-byte identical.
    #[test]
    fn text_request_wire_is_unchanged() {
        let request = EngineCoreRequest::default();
        let (payload, frames) = VllmProtocol::encode_add(&request, &VllmLayout::TokenList).unwrap();
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

    fn ready(version: &str) -> EngineCoreReadyResponse {
        EngineCoreReadyResponse {
            vllm_version: version.to_string(),
            ..default_ready_response()
        }
    }

    /// The bytes the engine's own encoder writes for a three-token prompt on
    /// 0.34 (`MsgpackEncoder._encode_ndarray` over the `int32` array its
    /// frontend packs): the dtype, the shape, and the data inline as a raw
    /// view. Behind the array header and the empty request id.
    #[test]
    fn a_0_34_engine_is_sent_its_prompt_as_an_int32_array() {
        let request = EngineCoreRequest {
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let layout = VllmProtocol::read_layout(&ready("0.34.0"));
        let (payload, frames) = VllmProtocol::encode_add(&request, &layout).unwrap();
        let wire = hex("dc0015 a0 93 a33c6934 9103 c70c03 01000000 02000000 03000000");
        assert_eq!(payload[..wire.len()], wire);
        assert!(frames.is_empty());
    }

    /// A release above the newest this client was read against is not
    /// refused: its engine connects and is sent the bytes of 0.34.
    #[test]
    fn a_release_above_0_34_connects_and_is_sent_the_array_of_0_34() {
        let request = EngineCoreRequest {
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let as_0_34 = VllmProtocol::read_layout(&ready("0.34.0"));
        let (sent_0_34, _) = VllmProtocol::encode_add(&request, &as_0_34).unwrap();
        for version in ["0.35.0", "1.0.0"] {
            assert!(
                VllmProtocol::check_layouts(&ready(version)).is_ok(),
                "{version} connects"
            );
            let layout = VllmProtocol::read_layout(&ready(version));
            let (payload, frames) = VllmProtocol::encode_add(&request, &layout).unwrap();
            assert_eq!(payload, sent_0_34, "{version} is sent the bytes of 0.34");
            assert!(frames.is_empty(), "{version}: a short prompt goes inline");
        }
        // What each kind of version text is served.
        for (version, served) in [
            ("0.31.0", VllmLayout::TokenList),
            ("0.33.2", VllmLayout::TokenList),
            ("0.34.1", VllmLayout::TokenArray),
            ("0.34.0rc1", VllmLayout::TokenArray),
            ("0.34.0.dev12+gabc", VllmLayout::TokenArray),
            ("0.35.0", VllmLayout::TokenArray),
            ("1.0.0", VllmLayout::TokenArray),
            ("dev", VllmLayout::TokenList),
            ("tokenspeed-0.1.1", VllmLayout::TokenList),
            ("trtllm-1.4.0", VllmLayout::TokenList),
            ("sglang-0.5.21", VllmLayout::TokenList),
        ] {
            assert_eq!(
                VllmProtocol::read_layout(&ready(version)),
                served,
                "{version}"
            );
        }
    }
}

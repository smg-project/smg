//! The multimodal payloads of the wrapped requests: the per-wire multimodal
//! data structs, the tensors they carry (inline bytes or a shared-memory
//! segment), the tensor-to-proto conversions and the RDMA export. Moved out
//! of `proto_wrapper.rs` as is; the request wrappers build them from the
//! parent and the shared-memory module unlinks what they stage. The two
//! multimodal-router helpers are read through the parent's imports.

use std::{collections::HashMap, time::Instant};

use rand::RngExt;
use smg_grpc_client::{
    common_proto as common, sglang_proto as sglang, tokenspeed_proto as tokenspeed,
    trtllm_proto as trtllm, vllm_proto as vllm,
};
use smg_mm_rdma::RdmaExporter;

use super::{
    log_mm_timing_enabled, mm_rdma_exporter, shm::write_tokenspeed_shm, VLLM_MROPE_GRID_KEYS,
};

/// How a streaming response's per-token payloads (token ids, sampled
/// logprobs, token counts) relate across the responses of one stream.
///
/// This is a property of the response *shape*, not of the engine behind it: a
/// TokenSpeed worker reached over ZMQ emits vLLM-shaped responses and so
/// carries `Delta` semantics, while the same engine reached over gRPC carries
/// `Cumulative`. Response-side accumulation must key on this, never on the
/// engine variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkSemantics {
    /// Each chunk carries only what is new since the previous one, so the
    /// consumer accumulates; the terminal `Complete` repeats the totals.
    Delta,
    /// Each chunk carries the run's totals so far, so the consumer replaces;
    /// the terminal `Complete` is authoritative.
    Cumulative,
}

impl ChunkSemantics {
    /// Whether chunks must be accumulated rather than replaced.
    pub fn is_delta(self) -> bool {
        matches!(self, Self::Delta)
    }
}

/// Backend-neutral encode->prefill bootstrap info for one multimodal item.
///
/// Backend wrappers translate this into their own proto shape when supported.
#[derive(Clone, Debug)]
pub(crate) struct EncodeItemBootstrapInfo {
    pub item_index: u32,
    pub bootstrap_host: String,
    pub bootstrap_port: i32,
    pub bootstrap_room: i64,
}

// =====================
// Multimodal Data
// =====================

/// Backend-specific multimodal data produced by the assembly stage.
///
/// Each variant carries only the fields its backend needs:
/// - SGLang: preprocessed vision tensor + model-specific tensors + patch-only placeholders
/// - vLLM: preprocessed vision tensor + model-specific tensors + structural placeholders + hashes + field keys
/// - TRT-LLM: raw image bytes only (preprocessing handled server-side)
/// - TokenSpeed: encoder_input + model_specific_tensors + patch-only placeholders
#[derive(Debug)]
pub enum MultimodalData {
    Sglang(SglangMultimodalData),
    Vllm(VllmMultimodalData),
    Trtllm(TrtllmMultimodalData),
    TokenSpeed(TokenSpeedMultimodalData),
}

impl MultimodalData {
    /// Bytes of media the request body itself carries.
    pub fn inline_bytes(&self) -> usize {
        match self {
            Self::Sglang(data) => {
                data.image_data.iter().map(Vec::len).sum::<usize>()
                    + data.pixel_values.len()
                    + tensor_bytes(&data.model_specific_tensors)
            }
            Self::Vllm(data) => data.inline_bytes(),
            Self::Trtllm(data) => data.image_data.iter().map(Vec::len).sum(),
            Self::TokenSpeed(data) => data
                .items
                .iter()
                .map(|item| {
                    let encoder = match &item.encoder_input.storage {
                        TokenSpeedTensorStorage::Inline(bytes) => bytes.len(),
                        TokenSpeedTensorStorage::Shm(_) | TokenSpeedTensorStorage::Remote(_) => 0,
                    };
                    encoder + tensor_bytes(&item.model_specific_tensors)
                })
                .sum(),
        }
    }
}

fn tensor_bytes(tensors: &HashMap<String, TensorBytes>) -> usize {
    tensors.values().map(|tensor| tensor.data.len()).sum()
}

impl VllmMultimodalData {
    /// Bytes of this batch and of every batch travelling with it.
    pub fn inline_bytes(&self) -> usize {
        self.pixel_values.len()
            + tensor_bytes(&self.model_specific_tensors)
            + self
                .extra_batches
                .iter()
                .map(Self::inline_bytes)
                .sum::<usize>()
    }
}

/// SGLang multimodal data: preprocessed tensors with patch-only placeholders.
#[derive(Debug)]
pub struct SglangMultimodalData {
    pub image_data: Vec<Vec<u8>>,
    pub pixel_values: Vec<u8>,
    pub pixel_values_shape: Vec<u32>,
    /// The width `pixel_values` was written in, which the receiver reads it at.
    pub pixel_values_dtype: String,
    pub model_specific_tensors: HashMap<String, TensorBytes>,
    pub im_token_id: Option<u32>,
    /// Patch-only placeholder offsets aligned 1:1 with vision encoder output.
    pub mm_placeholders: Vec<(u32, u32)>,
}

/// vLLM multimodal data: preprocessed tensors with hashing and field layout metadata.
#[derive(Debug)]
pub struct VllmMultimodalData {
    pub pixel_values: Vec<u8>,
    pub pixel_values_shape: Vec<u32>,
    /// The width `pixel_values` was written in, which the receiver reads it at.
    pub pixel_values_dtype: String,
    pub model_specific_tensors: HashMap<String, TensorBytes>,
    pub im_token_id: Option<u32>,
    /// Full structural placeholder offsets (vLLM filters via is_embed mask).
    pub mm_placeholders: Vec<(u32, u32)>,
    pub mm_hashes: Vec<String>,
    pub batched_keys: Vec<String>,
    pub flat_keys: HashMap<String, String>,
    /// Tensor keys that should remain on CPU (`keep_on_cpu=True` in vLLM).
    pub keep_on_cpu_keys: Vec<String>,
    /// Wire key for the primary tensor when the model's forward does not take
    /// `pixel_values` (`None` keeps the default name).
    pub encoder_input_key: Option<String>,
    /// Input modality (image/video). Selects the video modality
    /// (`pixel_values_videos` / `video_grid_thw`) on the servicer side.
    pub modality: common::Modality,
    /// Resolved per-request SHM transport decision + size threshold (bytes),
    /// computed upstream (transport mode / worker locality / config). `into_proto`
    /// uses these to place each tensor inline or in /dev/shm without re-reading
    /// config or the environment.
    pub shm_enabled: bool,
    pub shm_min_bytes: usize,
    /// Whether `pixel_values` may use the RDMA lane. Gated on the worker being able
    /// to pull, so SMG never emits a `remote` payload a worker would reject.
    pub rdma_enabled: bool,
    /// The other modalities of a mixed request (a video batch next to an
    /// image batch), each assembled the same way; they travel in the
    /// request's `extra_mm_inputs`.
    pub extra_batches: Vec<VllmMultimodalData>,
}

/// TRT-LLM multimodal data: raw image bytes only.
#[derive(Debug)]
pub struct TrtllmMultimodalData {
    pub image_data: Vec<Vec<u8>>,
}

/// TokenSpeed multimodal data: preprocessed encoder input with patch-only placeholders.
#[derive(Debug)]
pub struct TokenSpeedMultimodalData {
    pub items: Vec<TokenSpeedMultimodalItem>,
    /// Resolved per-request decision: may large multimodal tensors use the SHM
    /// transport? Computed upstream from the transport mode and (for `auto`)
    /// worker locality, so `into_proto` does not re-read the environment.
    pub shm_enabled: bool,
    /// Resolved per-request SHM size threshold (bytes): tensors smaller than this
    /// stay inline even when `shm_enabled`. Computed upstream (worker override →
    /// router config → env → default) so `into_proto` does not re-read the env.
    pub shm_min_bytes: usize,
}

#[derive(Debug)]
pub struct TokenSpeedMultimodalItem {
    pub modality: TokenSpeedModality,
    pub encoder_input: TokenSpeedTensor,
    pub model_specific_tensors: HashMap<String, TensorBytes>,
    pub placeholder_token_id: Option<u32>,
    pub mm_placeholders: Vec<(u32, u32)>,
    pub content_hash: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenSpeedModality {
    Image,
    Audio,
    Video,
}

/// Raw tensor bytes with shape and dtype metadata.
#[derive(Debug, Clone)]
pub struct TensorBytes {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    pub dtype: String,
}

/// TokenSpeed tensor with an explicit payload transport. Inline, SHM, and
/// remote descriptors all converge here before becoming generated proto fields,
/// so stages do not need to mutate `TensorData` oneofs directly.
#[derive(Debug, Clone)]
pub struct TokenSpeedTensor {
    pub storage: TokenSpeedTensorStorage,
    pub shape: Vec<u32>,
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub enum TokenSpeedTensorStorage {
    Inline(Vec<u8>),
    Shm(common::ShmHandle),
    Remote(common::RemoteTensorHandle),
}

impl TokenSpeedTensor {
    pub fn inline(data: Vec<u8>, shape: Vec<u32>, dtype: String) -> Self {
        Self {
            storage: TokenSpeedTensorStorage::Inline(data),
            shape,
            dtype,
        }
    }

    pub fn shm(handle: common::ShmHandle, shape: Vec<u32>, dtype: String) -> Self {
        Self {
            storage: TokenSpeedTensorStorage::Shm(handle),
            shape,
            dtype,
        }
    }

    pub fn remote(handle: common::RemoteTensorHandle, shape: Vec<u32>, dtype: String) -> Self {
        Self {
            storage: TokenSpeedTensorStorage::Remote(handle),
            shape,
            dtype,
        }
    }

    pub fn nbytes(&self) -> usize {
        match &self.storage {
            TokenSpeedTensorStorage::Inline(data) => data.len(),
            TokenSpeedTensorStorage::Shm(handle) => handle.nbytes as usize,
            TokenSpeedTensorStorage::Remote(handle) => handle.nbytes as usize,
        }
    }
}

impl SglangMultimodalData {
    /// Convert to SGLang proto MultimodalInputs.
    pub fn into_proto(self) -> sglang::MultimodalInputs {
        let model_specific_tensors = self
            .model_specific_tensors
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    sglang::TensorData {
                        data: v.data.into(),
                        shape: v.shape,
                        dtype: v.dtype,
                    },
                )
            })
            .collect();

        let mm_placeholders = self
            .mm_placeholders
            .into_iter()
            .map(|(offset, length)| sglang::PlaceholderRange { offset, length })
            .collect();

        sglang::MultimodalInputs {
            image_urls: vec![],
            video_urls: vec![],
            audio_urls: vec![],
            image_data: self.image_data,
            video_data: vec![],
            audio_data: vec![],
            modalities: vec!["image".to_string()],
            pixel_values: Some(sglang::TensorData {
                data: self.pixel_values.into(),
                shape: self.pixel_values_shape,
                dtype: self.pixel_values_dtype,
            }),
            model_specific_tensors,
            im_token_id: self.im_token_id,
            mm_placeholders,
        }
    }
}

impl VllmMultimodalData {
    /// Convert to the request's `mm_inputs` and `extra_mm_inputs`.
    pub fn into_protos(mut self) -> (vllm::MultimodalInputs, Vec<vllm::MultimodalInputs>) {
        let extras = std::mem::take(&mut self.extra_batches)
            .into_iter()
            .map(Self::into_proto)
            .collect();
        (self.into_proto(), extras)
    }

    /// Convert one batch to a vLLM proto MultimodalInputs.
    pub(super) fn into_proto(self) -> vllm::MultimodalInputs {
        let shm_enabled = self.shm_enabled;
        let shm_min_bytes = self.shm_min_bytes;
        // RDMA only for pixel_values (see TokenSpeed item conversion).
        let pixel_rdma = if self.rdma_enabled {
            mm_rdma_exporter().map(|exporter| MmRdmaExport {
                exporter,
                slot_key: rand::rng().random_range(0..i64::MAX),
            })
        } else {
            None
        };
        let model_specific_tensors = self
            .model_specific_tensors
            .into_iter()
            .map(|(k, v)| {
                // Grid tensors are a few ints and must survive the PD decode
                // clone's inline-only retention: never route them via SHM.
                let via_shm = shm_enabled && !VLLM_MROPE_GRID_KEYS.contains(&k.as_str());
                let payload = vllm_tensor_payload(v.data, via_shm, shm_min_bytes, None);
                (
                    k,
                    vllm::TensorData {
                        shape: v.shape,
                        dtype: v.dtype,
                        payload: Some(payload),
                    },
                )
            })
            .collect();

        let mm_placeholders = self
            .mm_placeholders
            .into_iter()
            .map(|(offset, length)| vllm::PlaceholderRange { offset, length })
            .collect();

        vllm::MultimodalInputs {
            pixel_values: Some(vllm::TensorData {
                shape: self.pixel_values_shape,
                dtype: self.pixel_values_dtype,
                payload: Some(vllm_tensor_payload(
                    self.pixel_values,
                    shm_enabled,
                    shm_min_bytes,
                    pixel_rdma,
                )),
            }),
            model_specific_tensors,
            im_token_id: self.im_token_id,
            mm_placeholders,
            mm_hashes: self.mm_hashes,
            batched_keys: self.batched_keys,
            flat_keys: self.flat_keys,
            keep_on_cpu_keys: self.keep_on_cpu_keys,
            modality: self.modality as i32,
            encoder_input_key: self.encoder_input_key,
        }
    }
}

impl TrtllmMultimodalData {
    /// Convert to TRT-LLM proto MultimodalInput.
    pub fn into_proto(self) -> trtllm::MultimodalInput {
        trtllm::MultimodalInput {
            image_data: self.image_data,
        }
    }
}

impl TokenSpeedMultimodalData {
    /// Convert to TokenSpeed proto MultimodalInputs. `rdma_enabled` stages each
    /// inline encoder_input over RDMA with a random slot key (the general path); EPD
    /// passes `false` since it stages explicitly with `bootstrap_room` first.
    pub fn into_proto(self, rdma_enabled: bool) -> tokenspeed::MultimodalInputs {
        let shm_enabled = self.shm_enabled;
        let shm_min_bytes = self.shm_min_bytes;
        let rdma_exporter = if rdma_enabled {
            mm_rdma_exporter()
        } else {
            None
        };
        let items = self
            .items
            .into_iter()
            .map(|item| item.into_proto(shm_enabled, shm_min_bytes, rdma_exporter))
            .collect();
        tokenspeed::MultimodalInputs { items }
    }
}

impl TokenSpeedMultimodalItem {
    fn into_proto(
        self,
        shm_enabled: bool,
        shm_min_bytes: usize,
        rdma_exporter: Option<&RdmaExporter>,
    ) -> tokenspeed::MultimodalItem {
        let placeholders = self
            .mm_placeholders
            .into_iter()
            .map(|(offset, length)| tokenspeed::PlaceholderRange { offset, length })
            .collect::<Vec<_>>();

        // No RDMA for the small model-specific side tensors: keeps the fixed slot
        // pool for the pixel-heavy encoder_input.
        let model_specific_tensors = self
            .model_specific_tensors
            .into_iter()
            .map(|(k, v)| (k, tensor_bytes_to_tokenspeed(v, shm_enabled, shm_min_bytes)))
            .collect::<HashMap<_, _>>();

        let encoder_rdma = rdma_exporter.map(|exporter| MmRdmaExport {
            exporter,
            slot_key: rand::rng().random_range(0..i64::MAX),
        });
        let encoder_input = Some(tokenspeed_tensor_to_proto(
            self.encoder_input,
            shm_enabled,
            shm_min_bytes,
            encoder_rdma,
        ));

        tokenspeed::MultimodalItem {
            modality: match self.modality {
                TokenSpeedModality::Image => common::Modality::Image as i32,
                TokenSpeedModality::Audio => common::Modality::Audio as i32,
                TokenSpeedModality::Video => common::Modality::Video as i32,
            },
            content_hash: self.content_hash,
            encoder_input,
            model_specific_tensors,
            placeholders,
            placeholder_token_id: self.placeholder_token_id,
        }
    }
}

fn tokenspeed_tensor_to_proto(
    value: TokenSpeedTensor,
    shm_enabled: bool,
    shm_min_bytes: usize,
    rdma: Option<MmRdmaExport<'_>>,
) -> tokenspeed::TensorData {
    use crate::observability::metrics::Metrics;
    let TokenSpeedTensor {
        storage,
        shape,
        dtype,
    } = value;
    let payload = match storage {
        // Inline storage: RDMA/SHM/inline is resolved (and metered) here.
        TokenSpeedTensorStorage::Inline(data) => {
            tokenspeed_tensor_payload(data, shm_enabled, shm_min_bytes, rdma)
        }
        // Encoder input already written directly to SHM upstream — meter it here.
        TokenSpeedTensorStorage::Shm(handle) => {
            Metrics::record_mm_tensor("tokenspeed", "shm", handle.nbytes as usize);
            tokenspeed::tensor_data::Payload::Shm(handle)
        }
        TokenSpeedTensorStorage::Remote(handle) => {
            Metrics::record_mm_tensor("tokenspeed", "remote", handle.nbytes as usize);
            tokenspeed::tensor_data::Payload::Remote(handle)
        }
    };

    tokenspeed::TensorData {
        shape,
        dtype,
        payload: Some(payload),
    }
}

pub(super) fn tensor_bytes_to_tokenspeed(
    value: TensorBytes,
    shm_enabled: bool,
    shm_min_bytes: usize,
) -> tokenspeed::TensorData {
    let TensorBytes { data, shape, dtype } = value;

    tokenspeed::TensorData {
        shape,
        dtype,
        payload: Some(tokenspeed_tensor_payload(
            data,
            shm_enabled,
            shm_min_bytes,
            None,
        )),
    }
}

/// Transport decision for one multimodal tensor; each backend maps it onto its
/// own `TensorData` oneof.
enum MmTensorPayload {
    Inline(Vec<u8>),
    Shm(common::ShmHandle),
    Remote(common::RemoteTensorHandle),
}

/// A request to stage one tensor over RDMA. `slot_key` tags the descriptor: the
/// general path mints a random key, EPD uses the load-bearing `bootstrap_room`.
#[derive(Clone, Copy)]
struct MmRdmaExport<'a> {
    exporter: &'a RdmaExporter,
    slot_key: i64,
}

/// Stage `data` into the RDMA arena under `slot_key`; `Ok(handle)`, or the bytes
/// back on failure (empty tensors are never staged).
fn export_rdma_tensor(
    exporter: &RdmaExporter,
    slot_key: i64,
    data: Vec<u8>,
) -> Result<common::RemoteTensorHandle, Vec<u8>> {
    if data.is_empty() {
        return Err(data);
    }
    let nbytes = data.len() as u64;
    exporter
        .export(slot_key, data)
        .map(|descriptor| common::RemoteTensorHandle {
            transport: "nixl".to_string(),
            descriptor,
            nbytes,
        })
}

/// Stage an inline TokenSpeed encoder input over RDMA under an explicit `slot_key`
/// (EPD uses `bootstrap_room`). Non-inline tensors pass through untouched.
pub(crate) fn stage_tokenspeed_tensor_rdma(
    exporter: &RdmaExporter,
    slot_key: i64,
    tensor: TokenSpeedTensor,
) -> TokenSpeedTensor {
    let TokenSpeedTensor {
        storage,
        shape,
        dtype,
    } = tensor;
    let TokenSpeedTensorStorage::Inline(data) = storage else {
        return TokenSpeedTensor {
            storage,
            shape,
            dtype,
        };
    };
    match export_rdma_tensor(exporter, slot_key, data) {
        Ok(handle) => TokenSpeedTensor::remote(handle, shape, dtype),
        Err(data) => TokenSpeedTensor::inline(data, shape, dtype),
    }
}

fn resolve_mm_tensor_payload(
    data: Vec<u8>,
    shm_enabled: bool,
    min_bytes: usize,
    engine: &'static str,
    rdma: Option<MmRdmaExport<'_>>,
) -> MmTensorPayload {
    use crate::observability::metrics::Metrics;
    let log_timing = log_tokenspeed_mm_timing_enabled();

    // RDMA first; on export failure the bytes are handed back so we fall through to
    // SHM/inline rather than drop the payload.
    let data = match rdma {
        Some(rdma) => {
            let nbytes = data.len();
            match export_rdma_tensor(rdma.exporter, rdma.slot_key, data) {
                Ok(handle) => {
                    Metrics::record_mm_tensor(engine, "remote", nbytes);
                    return MmTensorPayload::Remote(handle);
                }
                Err(data) => data,
            }
        }
        None => data,
    };

    let nbytes = data.len();
    if !shm_enabled || nbytes < min_bytes {
        if log_timing {
            tracing::info!(
                engine,
                nbytes,
                min_bytes,
                "smg_mm_timing mm_tensor_payload_inline"
            );
        }
        Metrics::record_mm_tensor(engine, "inline", nbytes);
        return MmTensorPayload::Inline(data);
    }

    let started = Instant::now();
    match write_tokenspeed_shm(&data) {
        Ok(handle) => {
            if log_timing {
                tracing::info!(
                    engine,
                    nbytes,
                    elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                    "smg_mm_timing mm_shm_write"
                );
            }
            Metrics::record_mm_tensor(engine, "shm", nbytes);
            MmTensorPayload::Shm(handle)
        }
        Err(error) => {
            tracing::warn!(
                ?error,
                engine,
                nbytes,
                "Failed to write multimodal SHM tensor; falling back to inline"
            );
            Metrics::record_mm_shm_write_failure(engine);
            Metrics::record_mm_tensor(engine, "inline", nbytes);
            MmTensorPayload::Inline(data)
        }
    }
}

fn tokenspeed_tensor_payload(
    data: Vec<u8>,
    shm_enabled: bool,
    min_bytes: usize,
    rdma: Option<MmRdmaExport<'_>>,
) -> tokenspeed::tensor_data::Payload {
    match resolve_mm_tensor_payload(data, shm_enabled, min_bytes, "tokenspeed", rdma) {
        MmTensorPayload::Inline(data) => tokenspeed::tensor_data::Payload::Inline(data.into()),
        MmTensorPayload::Shm(handle) => tokenspeed::tensor_data::Payload::Shm(handle),
        MmTensorPayload::Remote(handle) => tokenspeed::tensor_data::Payload::Remote(handle),
    }
}

fn vllm_tensor_payload(
    data: Vec<u8>,
    shm_enabled: bool,
    min_bytes: usize,
    rdma: Option<MmRdmaExport<'_>>,
) -> vllm::tensor_data::Payload {
    match resolve_mm_tensor_payload(data, shm_enabled, min_bytes, "vllm", rdma) {
        MmTensorPayload::Inline(data) => vllm::tensor_data::Payload::Inline(data.into()),
        MmTensorPayload::Shm(handle) => vllm::tensor_data::Payload::Shm(handle),
        MmTensorPayload::Remote(handle) => vllm::tensor_data::Payload::Remote(handle),
    }
}

fn log_tokenspeed_mm_timing_enabled() -> bool {
    log_mm_timing_enabled()
}

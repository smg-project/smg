//! Protocol buffer type wrappers for the supported gRPC backends.
//!
//! This module provides unified enums that wrap proto types from each
//! supported backend, allowing the router to work with any backend
//! transparently.

use std::{
    collections::HashMap,
    fs::{read_dir, remove_file, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use memmap2::MmapOptions;
use rand::RngExt;
#[cfg(target_os = "linux")]
use rustix::fs::FallocateFlags;
use smg_grpc_client::{
    common_proto::{self as common},
    mlx_engine::AbortOnDropStream as MlxStream,
    mlx_proto::{self as mlx},
    sglang_proto::{self as sglang, generate_complete::MatchedStop as SglangMatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    tokenspeed_proto::{
        self as tokenspeed, generate_complete::MatchedStop as TokenSpeedMatchedStop,
    },
    tokenspeed_scheduler::AbortOnDropStream as TokenSpeedStream,
    trtllm_proto::{self as trtllm, generate_complete::MatchedStop as TrtllmMatchedStop},
    trtllm_service::AbortOnDropStream as TrtllmStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto::{self as vllm, generate_complete::MatchedStop as VllmMatchedStop},
};
use smg_mm_rdma::RdmaExporter;

use crate::routers::grpc::{multimodal::mm_rdma_exporter, zmq_client::ZmqGenerateStream};

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

/// SGLang multimodal data: preprocessed tensors with patch-only placeholders.
#[derive(Debug)]
pub struct SglangMultimodalData {
    pub image_data: Vec<Vec<u8>>,
    pub pixel_values: Vec<u8>,
    pub pixel_values_shape: Vec<u32>,
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
    pub model_specific_tensors: HashMap<String, TensorBytes>,
    pub im_token_id: Option<u32>,
    /// Full structural placeholder offsets (vLLM filters via is_embed mask).
    pub mm_placeholders: Vec<(u32, u32)>,
    pub mm_hashes: Vec<String>,
    pub batched_keys: Vec<String>,
    pub flat_keys: HashMap<String, String>,
    /// Tensor keys that should remain on CPU (`keep_on_cpu=True` in vLLM).
    pub keep_on_cpu_keys: Vec<String>,
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
                        data: v.data,
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
                data: self.pixel_values,
                shape: self.pixel_values_shape,
                dtype: "float32".to_string(),
            }),
            model_specific_tensors,
            im_token_id: self.im_token_id,
            mm_placeholders,
        }
    }
}

impl VllmMultimodalData {
    /// Convert to vLLM proto MultimodalInputs.
    pub fn into_proto(self) -> vllm::MultimodalInputs {
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
                (
                    k,
                    vllm::TensorData {
                        shape: v.shape,
                        dtype: v.dtype,
                        payload: Some(vllm_tensor_payload(
                            v.data,
                            shm_enabled,
                            shm_min_bytes,
                            None,
                        )),
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
                dtype: "float32".to_string(),
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

fn tensor_bytes_to_tokenspeed(
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
        MmTensorPayload::Inline(data) => tokenspeed::tensor_data::Payload::Inline(data),
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
        MmTensorPayload::Inline(data) => vllm::tensor_data::Payload::Inline(data),
        MmTensorPayload::Shm(handle) => vllm::tensor_data::Payload::Shm(handle),
        MmTensorPayload::Remote(handle) => vllm::tensor_data::Payload::Remote(handle),
    }
}

fn log_tokenspeed_mm_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SMG_LOG_MM_TIMING")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

static TOKENSPEED_SHM_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_tokenspeed_shm(data: &[u8]) -> std::io::Result<common::ShmHandle> {
    write_tokenspeed_shm_with(data.len(), |output| {
        output.copy_from_slice(data);
        Ok(())
    })
}

/// Whether SMG can actually create+write files under `/dev/shm`. Probed once;
/// when false the SHM transport cannot work, so `auto`/`shm` must stay inline.
pub fn mm_shm_dev_writable() -> bool {
    static WRITABLE: OnceLock<bool> = OnceLock::new();
    *WRITABLE.get_or_init(|| {
        let name = format!("smg-tokenspeed-probe-{}", process::id());
        let path = tokenspeed_shm_path(&name);
        // `create_new` (no clobber) + owner-only mode: /dev/shm is world-writable,
        // so plain create(truncate) is open to symlink/clobber attacks and the
        // file would otherwise inherit umask and be world-readable.
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let ok = opts
            .open(&path)
            .and_then(|mut file| file.write_all(b"x"))
            .is_ok();
        let _ = remove_file(&path);
        if !ok {
            tracing::warn!(
                path = %path.display(),
                "/dev/shm is not writable; TokenSpeed SHM tensor transport will fall back to inline"
            );
        }
        ok
    })
}

/// Best-effort, run-once sweep of `/dev/shm` for TokenSpeed payload files left
/// behind by a *previous* SMG process that crashed between writing a segment and
/// the consumer unlinking it. Files are named `smg-tokenspeed-<pid>-...`; we only
/// remove those whose producer pid is no longer alive (and never our own).
fn sweep_orphan_tokenspeed_shm_once() {
    static SWEEP: OnceLock<()> = OnceLock::new();
    SWEEP.get_or_init(|| {
        let dir = Path::new("/dev/shm");
        let Ok(entries) = read_dir(dir) else {
            return;
        };
        let my_pid = process::id();
        let mut removed = 0u32;
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                continue;
            };
            let Some(rest) = name.strip_prefix("smg-tokenspeed-") else {
                continue;
            };
            // pid is the first '-'-separated field after the prefix.
            let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<u32>().ok()) else {
                continue;
            };
            // Skip our own files and any still-live producer (pid recycling is a
            // safe miss: we just keep the file rather than risk deleting a live one).
            if pid == my_pid || Path::new(&format!("/proc/{pid}")).exists() {
                continue;
            }
            if remove_file(dir.join(name)).is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            tracing::warn!(
                count = removed,
                "Swept orphaned TokenSpeed SHM files from dead producer processes"
            );
        }
    });
}

// TODO: pack all of a request's tensors (encoder_input + model_specific) into
// ONE /dev/shm segment at running offsets instead of one file per tensor
// (ShmHandle.offset already exists, always 0 here). Needs consumer
// ShmTensorHandle offset support + a per-segment refcount so the segment is
// unlinked exactly once after all its tensors are consumed. Cleanliness / fewer
// files, not a measured speed win (tmpfs makes per-file syscalls negligible).
#[expect(
    unsafe_code,
    reason = "mapping a new, exclusively owned, fixed-length SHM file"
)]
pub fn write_tokenspeed_shm_with(
    nbytes: usize,
    write_fn: impl FnOnce(&mut [u8]) -> std::io::Result<()>,
) -> std::io::Result<common::ShmHandle> {
    sweep_orphan_tokenspeed_shm_once();
    let name = next_tokenspeed_shm_name();
    let path = tokenspeed_shm_path(&name);
    // create_new (no clobber) + owner-only mode in world-writable /dev/shm.
    let mut opts = OpenOptions::new();
    opts.read(true).write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(&path)?;
    let result = if nbytes == 0 {
        write_fn(&mut [])
    } else {
        reserve_tokenspeed_shm_file(&file, nbytes).and_then(|()| {
            // SAFETY: this process exclusively owns the newly created file,
            // reserves its full length before mapping, and does not expose or
            // truncate it until the callback and mapping have both dropped.
            let mut mapping = unsafe { MmapOptions::new().len(nbytes).map_mut(&file)? };
            write_fn(&mut mapping)
        })
    };
    if let Err(error) = result {
        let _ = remove_file(&path);
        return Err(error);
    }

    Ok(common::ShmHandle {
        name,
        offset: 0,
        nbytes: nbytes as u64,
        owner_id: format!("smg:{}", process::id()),
    })
}

#[cfg(target_os = "linux")]
fn reserve_tokenspeed_shm_file(file: &std::fs::File, nbytes: usize) -> std::io::Result<()> {
    match rustix::fs::fallocate(file, FallocateFlags::empty(), 0, nbytes as u64) {
        Ok(()) => Ok(()),
        Err(error)
            if error == rustix::io::Errno::OPNOTSUPP || error == rustix::io::Errno::NOSYS =>
        {
            file.set_len(nbytes as u64)
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(target_os = "linux"))]
fn reserve_tokenspeed_shm_file(file: &std::fs::File, nbytes: usize) -> std::io::Result<()> {
    file.set_len(nbytes as u64)
}

pub fn collect_tokenspeed_multimodal_inputs_shm_handles(
    inputs: &tokenspeed::MultimodalInputs,
) -> Vec<common::ShmHandle> {
    let mut handles = Vec::new();
    for item in &inputs.items {
        collect_optional_tokenspeed_tensor_shm_handles(item.encoder_input.as_ref(), &mut handles);
        for tensor in item.model_specific_tensors.values() {
            collect_tokenspeed_tensor_shm_handles(tensor, &mut handles);
        }
    }
    handles
}

pub fn collect_tokenspeed_generate_request_shm_handles(
    request: &tokenspeed::GenerateRequest,
) -> Vec<common::ShmHandle> {
    request
        .mm_inputs
        .as_ref()
        .map(collect_tokenspeed_multimodal_inputs_shm_handles)
        .unwrap_or_default()
}

pub fn cleanup_mm_shm_handles(handles: &[common::ShmHandle]) {
    for handle in handles {
        let Some(name) = validate_tokenspeed_shm_name_for_cleanup(&handle.name) else {
            tracing::warn!(
                name = %handle.name,
                "Skipping cleanup for invalid TokenSpeed SHM name"
            );
            continue;
        };
        let path = tokenspeed_shm_path(name);
        match remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    ?error,
                    path = %path.display(),
                    "Failed to cleanup TokenSpeed SHM file"
                );
            }
        }
    }
}

/// Build a TokenSpeed proto `GenerateRequest` from the already-converted
/// multimodal proto, unlinking any `/dev/shm` segments it references if `build`
/// fails — so a build error doesn't leak SHM files before the send-path cleanup
/// can run. Keeping this engine-specific SHM lifecycle in the protocol layer
/// lets the per-engine dispatch in `client.rs` stay a thin, neutral wrapper.
pub(crate) fn finish_tokenspeed_request(
    tokenspeed_mm: Option<tokenspeed::MultimodalInputs>,
    build: impl FnOnce(
        Option<tokenspeed::MultimodalInputs>,
    ) -> Result<tokenspeed::GenerateRequest, String>,
) -> Result<ProtoGenerateRequest, String> {
    let shm_handles = tokenspeed_mm
        .as_ref()
        .map(collect_tokenspeed_multimodal_inputs_shm_handles)
        .unwrap_or_default();
    match build(tokenspeed_mm) {
        Ok(req) => Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req))),
        Err(error) => {
            cleanup_mm_shm_handles(&shm_handles);
            Err(error)
        }
    }
}

/// Build a vLLM generate request, cleaning up any `/dev/shm` segments backing
/// `vllm_mm` if the build fails. `into_proto` may write SHM files before the
/// request is fully assembled (sampling/tool validation), so a build error must
/// unlink them or the worker — which never receives the request — leaks them.
pub(crate) fn finish_vllm_request(
    vllm_mm: Option<vllm::MultimodalInputs>,
    build: impl FnOnce(Option<vllm::MultimodalInputs>) -> Result<vllm::GenerateRequest, String>,
) -> Result<ProtoGenerateRequest, String> {
    let shm_handles = vllm_mm
        .as_ref()
        .map(collect_vllm_multimodal_inputs_shm_handles)
        .unwrap_or_default();
    match build(vllm_mm) {
        Ok(req) => Ok(ProtoGenerateRequest::Vllm(Box::new(req))),
        Err(error) => {
            cleanup_mm_shm_handles(&shm_handles);
            Err(error)
        }
    }
}

/// Unlink the `/dev/shm` segments backing the encoder inputs of intermediate
/// `items` (plus an optional just-built `pending` tensor that hasn't been pushed
/// yet). Used when multimodal assembly aborts partway: the successfully built
/// `TokenSpeedTensor::Shm` segments would otherwise be dropped without their
/// handles ever reaching the send-path cleanup hooks, leaking files until the
/// next process sweep. Only the encoder input uses SHM (model-specific tensors
/// stay inline). MUST run on the error path only — the success path keeps the
/// files alive for the worker and unlinks them after the RPC.
pub(crate) fn cleanup_tokenspeed_items_encoder_shm(
    items: &[TokenSpeedMultimodalItem],
    pending: Option<&TokenSpeedTensor>,
) {
    let mut handles = Vec::new();
    let mut push = |tensor: &TokenSpeedTensor| {
        if let TokenSpeedTensorStorage::Shm(handle) = &tensor.storage {
            handles.push(handle.clone());
        }
    };
    for item in items {
        push(&item.encoder_input);
    }
    if let Some(tensor) = pending {
        push(tensor);
    }
    if !handles.is_empty() {
        cleanup_mm_shm_handles(&handles);
    }
}

fn collect_optional_tokenspeed_tensor_shm_handles(
    tensor: Option<&tokenspeed::TensorData>,
    handles: &mut Vec<common::ShmHandle>,
) {
    let Some(tensor) = tensor else {
        return;
    };
    collect_tokenspeed_tensor_shm_handles(tensor, handles);
}

fn collect_tokenspeed_tensor_shm_handles(
    tensor: &tokenspeed::TensorData,
    handles: &mut Vec<common::ShmHandle>,
) {
    if let Some(tokenspeed::tensor_data::Payload::Shm(handle)) = &tensor.payload {
        handles.push(handle.clone());
    }
}

pub fn collect_vllm_multimodal_inputs_shm_handles(
    inputs: &vllm::MultimodalInputs,
) -> Vec<common::ShmHandle> {
    let mut handles = Vec::new();
    collect_optional_vllm_tensor_shm_handles(inputs.pixel_values.as_ref(), &mut handles);
    for tensor in inputs.model_specific_tensors.values() {
        collect_vllm_tensor_shm_handles(tensor, &mut handles);
    }
    handles
}

pub fn collect_vllm_generate_request_shm_handles(
    request: &vllm::GenerateRequest,
) -> Vec<common::ShmHandle> {
    request
        .mm_inputs
        .as_ref()
        .map(collect_vllm_multimodal_inputs_shm_handles)
        .unwrap_or_default()
}

fn collect_optional_vllm_tensor_shm_handles(
    tensor: Option<&vllm::TensorData>,
    handles: &mut Vec<common::ShmHandle>,
) {
    if let Some(tensor) = tensor {
        collect_vllm_tensor_shm_handles(tensor, handles);
    }
}

fn collect_vllm_tensor_shm_handles(
    tensor: &vllm::TensorData,
    handles: &mut Vec<common::ShmHandle>,
) {
    if let Some(vllm::tensor_data::Payload::Shm(handle)) = &tensor.payload {
        handles.push(handle.clone());
    }
}

/// Prefix for every `/dev/shm` payload this transport creates
/// (see [`next_tokenspeed_shm_name`]). Cleanup only ever unlinks names
/// carrying this prefix so it cannot remove unrelated `/dev/shm` entries.
const TOKENSPEED_SHM_NAME_PREFIX: &str = "smg-tokenspeed-";

fn validate_tokenspeed_shm_name_for_cleanup(name: &str) -> Option<&str> {
    let name = name.strip_prefix('/').unwrap_or(name);
    if name.is_empty() || name.contains('/') || name == "." || name == ".." || name.contains('\0') {
        return None;
    }
    // Only unlink names this transport created; never touch arbitrary
    // top-level /dev/shm entries.
    if !name.starts_with(TOKENSPEED_SHM_NAME_PREFIX) {
        return None;
    }
    Some(name)
}

fn next_tokenspeed_shm_name() -> String {
    let seq = TOKENSPEED_SHM_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!(
        "{}{}-{}-{}",
        TOKENSPEED_SHM_NAME_PREFIX,
        process::id(),
        nanos,
        seq
    )
}

fn tokenspeed_shm_path(name: &str) -> PathBuf {
    PathBuf::from("/dev/shm").join(name)
}

// =====================
// Unified Logprobs Types
// =====================

/// Unified output logprobs (backend-agnostic)
#[derive(Clone, Debug)]
pub struct ProtoOutputLogProbs {
    pub token_logprobs: Vec<f32>,
    pub token_ids: Vec<u32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Unified top logprobs per position
#[derive(Clone, Debug)]
pub struct ProtoTopLogProbs {
    pub values: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Unified input (prompt) logprobs
#[derive(Clone, Debug)]
pub struct ProtoInputLogProbs {
    pub token_logprobs: Vec<Option<f32>>, // First token is None
    pub token_ids: Vec<u32>,
    pub top_logprobs: Vec<ProtoTopLogProbs>,
}

/// Convert TRT-LLM TokenLogprob slice to unified ProtoOutputLogProbs.
fn convert_trtllm_output_logprobs(
    logprobs: &[trtllm::TokenLogprob],
) -> Option<ProtoOutputLogProbs> {
    if logprobs.is_empty() {
        return None;
    }
    Some(ProtoOutputLogProbs {
        token_logprobs: logprobs.iter().map(|lp| lp.logprob).collect(),
        token_ids: logprobs.iter().map(|lp| lp.token_id).collect(),
        top_logprobs: logprobs
            .iter()
            .map(|lp| ProtoTopLogProbs {
                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                token_ids: lp.top_logprobs.iter().map(|t| t.token_id).collect(),
            })
            .collect(),
    })
}

/// Helper macro to convert output logprobs from proto types to unified type.
/// Both SGLang and vLLM have identical OutputLogProbs structure.
/// Note: Cloning is necessary as we convert from borrowed proto types to owned unified types.
/// OOM risk is mitigated by capping top_logprobs at 20 in sampling params.
macro_rules! convert_output_logprobs {
    ($lp:expr) => {
        ProtoOutputLogProbs {
            token_logprobs: $lp.token_logprobs.clone(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Helper macro to convert input logprobs from proto types to unified type.
macro_rules! convert_input_logprobs {
    ($lp:expr) => {
        ProtoInputLogProbs {
            token_logprobs: $lp.token_logprobs.iter().map(|t| t.value).collect(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Unified ProtoRequest
#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
    Embed(ProtoEmbedRequest),
}

impl ProtoRequest {
    /// Get request ID from either variant
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
            Self::Embed(req) => req.request_id(),
        }
    }
}

/// Unified GenerateRequest that works with all backends
#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
    Trtllm(Box<trtllm::GenerateRequest>),
    Mlx(Box<mlx::GenerateRequest>),
    TokenSpeed(Box<tokenspeed::GenerateRequest>),
}

impl ProtoGenerateRequest {
    /// Append stop token ids to the request's sampling params (TRT-LLM keeps
    /// them on the request itself). Requests without sampling params are left
    /// unchanged, matching the per-engine injection this replaces.
    pub fn extend_stop_token_ids(&mut self, ids: &[u32]) {
        match self {
            Self::Sglang(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Vllm(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Mlx(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Trtllm(req) => req.stop_token_ids.extend_from_slice(ids),
        }
    }

    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            _ => panic!("Expected vLLM GenerateRequest"),
        }
    }

    /// Get mutable vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm_mut(&mut self) -> &mut vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            _ => panic!("Expected vLLM GenerateRequest"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateRequest {
        match self {
            Self::Trtllm(req) => req,
            _ => panic!("Expected TensorRT-LLM GenerateRequest"),
        }
    }

    /// Get mutable TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm_mut(&mut self) -> &mut trtllm::GenerateRequest {
        match self {
            Self::Trtllm(req) => req,
            _ => panic!("Expected TensorRT-LLM GenerateRequest"),
        }
    }

    /// Get TokenSpeed variant (panics if not TokenSpeed)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed(&self) -> &tokenspeed::GenerateRequest {
        match self {
            Self::TokenSpeed(req) => req,
            _ => panic!("Expected TokenSpeed GenerateRequest"),
        }
    }

    /// Get mutable TokenSpeed variant (panics if not TokenSpeed)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed_mut(&mut self) -> &mut tokenspeed::GenerateRequest {
        match self {
            Self::TokenSpeed(req) => req,
            _ => panic!("Expected TokenSpeed GenerateRequest"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// Sanitize sampling params for the prefill-only leg (vLLM PD mode).
    /// max_tokens=1 computes KV without generating; min_tokens is cleared so the
    /// engine accepts it; n=1 so the prefill returns a single kv_transfer_params dict.
    /// Stop criteria are cleared and EOS ignored so the leg always finishes
    /// length-capped — vLLM < 0.20 returns the NIXL handoff only for that status.
    pub fn sanitize_sampling_for_prefill(&mut self, max_tokens: u32) {
        match self {
            Self::Vllm(req) => {
                let params = req.sampling_params.get_or_insert_with(Default::default);
                params.max_tokens = Some(max_tokens);
                params.min_tokens = 0;
                params.n = 1;
                params.stop.clear();
                params.stop_token_ids.clear();
                params.ignore_eos = true;
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!(
                    "sanitize_sampling_for_prefill called on non-vLLM request, ignoring"
                );
            }
        }
    }

    /// Set stream mode on the request.
    pub fn set_stream(&mut self, stream: bool) {
        match self {
            Self::Vllm(req) => req.stream = stream,
            Self::Sglang(req) => req.stream = stream,
            Self::Trtllm(req) => req.streaming = stream,
            Self::Mlx(req) => req.stream = stream,
            Self::TokenSpeed(req) => req.stream = stream,
        }
    }

    /// Clone for a PD leg that carries no multimodal pixels (see
    /// `clear_mm_pixel_values`), without ever duplicating them: detach,
    /// clone, reattach to `self`.
    pub fn clone_without_mm_pixels(&mut self) -> Self {
        match self {
            Self::Sglang(req) => {
                let mm = req.mm_inputs.take();
                let clone = Self::Sglang(req.clone());
                req.mm_inputs = mm;
                clone
            }
            Self::Vllm(req) => {
                let mm = req.mm_inputs.take();
                let clone = Self::Vllm(req.clone());
                req.mm_inputs = mm;
                clone
            }
            Self::TokenSpeed(req) => {
                let detached: Vec<_> = req
                    .mm_inputs
                    .as_mut()
                    .map(|mm| {
                        mm.items
                            .iter_mut()
                            .map(|item| item.encoder_input.take())
                            .collect()
                    })
                    .unwrap_or_default();
                let clone = Self::TokenSpeed(req.clone());
                if let Some(mm) = req.mm_inputs.as_mut() {
                    for (item, encoder_input) in mm.items.iter_mut().zip(detached) {
                        item.encoder_input = encoder_input;
                    }
                }
                clone
            }
            Self::Trtllm(req) => Self::Trtllm(req.clone()),
            Self::Mlx(req) => Self::Mlx(req.clone()),
        }
    }

    /// Drop raw multimodal encoder tensors while keeping item metadata.
    ///
    /// Used by the EPD prefill leg: multimodal embeddings arrive from encode workers,
    /// but prefill still needs placeholders/model-specific metadata to slot them.
    pub fn clear_mm_pixel_values(&mut self) {
        match self {
            Self::Sglang(req) => req.mm_inputs = None,
            Self::Vllm(req) => req.mm_inputs = None,
            Self::TokenSpeed(req) => {
                if let Some(mm) = req.mm_inputs.as_mut() {
                    for item in &mut mm.items {
                        item.encoder_input = None;
                    }
                }
            }
            Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
            Self::Trtllm(req) => &req.request_id,
            Self::Mlx(req) => &req.request_id,
            Self::TokenSpeed(req) => &req.request_id,
        }
    }

    /// Set KV transfer parameters for Mooncake PD disaggregation (vLLM only).
    /// These parameters tell the decode worker where to fetch KV cache from the prefill worker.
    pub fn set_kv_transfer_params(&mut self, remote_host: String, remote_port: u32) {
        match self {
            Self::Vllm(req) => {
                req.kv_transfer_params = Some(vllm::KvTransferParams {
                    remote_host,
                    remote_port,
                });
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!("set_kv_transfer_params called on non-vLLM request, ignoring");
            }
        }
    }

    /// Pin the request to a data-parallel rank (engines without the field ignore it).
    pub fn set_data_parallel_rank(&mut self, rank: i32) {
        match self {
            Self::Vllm(req) => req.data_parallel_rank = Some(rank),
            Self::Sglang(req) => req.data_parallel_rank = rank,
            Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {}
        }
    }

    /// Number of parallel samples requested (vLLM only; 1 when unset).
    pub fn sampling_n(&self) -> u32 {
        match self {
            Self::Vllm(req) => req
                .sampling_params
                .as_ref()
                .map(|p| p.n)
                .filter(|&n| n > 0)
                .unwrap_or(1),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 1,
        }
    }

    /// Set opaque connector KV-transfer params as JSON (vLLM only).
    /// Passed verbatim to the engine (NIXL handoff, etc.).
    pub fn set_kv_transfer_params_json(&mut self, json: String) {
        match self {
            Self::Vllm(req) => req.kv_transfer_params_json = Some(json),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!("set_kv_transfer_params_json called on non-vLLM request, ignoring");
            }
        }
    }

    /// Set encode->prefill bootstrap info for backends that receive multimodal embeddings
    /// out-of-band from encode workers.
    pub(crate) fn set_encode_bootstrap_info(&mut self, items: Vec<EncodeItemBootstrapInfo>) {
        match self {
            Self::TokenSpeed(req) => {
                let items = items
                    .into_iter()
                    .map(|item| tokenspeed::EncodeItemBootstrapInfo {
                        item_index: item.item_index,
                        bootstrap_host: item.bootstrap_host,
                        bootstrap_port: item.bootstrap_port,
                        bootstrap_room: item.bootstrap_room,
                    })
                    .collect();
                req.encode_bootstrap_info = Some(tokenspeed::EncodeBootstrapInfo { items });
            }
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {
                tracing::warn!(
                    "set_encode_bootstrap_info called on a backend without encode bootstrap info, ignoring"
                );
            }
        }
    }

    /// Clear prefill-only encode bootstrap info from the decode-side request.
    pub(crate) fn clear_encode_bootstrap_info(&mut self) {
        match self {
            Self::TokenSpeed(req) => req.encode_bootstrap_info = None,
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Set the PD prefill->decode KV rendezvous params (TokenSpeed only).
    ///
    /// The gateway sends identical params to both the prefill and decode worker:
    /// the prefill hosts the Mooncake bootstrap server at (`bootstrap_host`,
    /// `bootstrap_port`) and the decode worker discovers it there, keyed by
    /// `bootstrap_room`.
    pub fn set_kv_bootstrap_info(
        &mut self,
        bootstrap_host: String,
        bootstrap_port: i32,
        bootstrap_room: i64,
    ) {
        match self {
            Self::TokenSpeed(req) => {
                req.kv_bootstrap_info = Some(tokenspeed::KvBootstrapInfo {
                    bootstrap_host,
                    bootstrap_port,
                    bootstrap_room,
                });
            }
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {
                tracing::warn!("set_kv_bootstrap_info called on non-TokenSpeed request, ignoring");
            }
        }
    }
}

/// Unified GenerateResponse from stream
pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(Box<vllm::GenerateResponse>),
    Trtllm(Box<trtllm::GenerateResponse>),
    Mlx(Box<mlx::GenerateResponse>),
    TokenSpeed(Box<tokenspeed::GenerateResponse>),
}

impl ProtoGenerateResponse {
    /// Get the response variant (chunk, complete, or error)
    ///
    /// Consumes self to avoid cloning large proto messages in hot streaming path
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Trtllm(resp) => match resp.response {
                Some(trtllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Trtllm(chunk))
                }
                Some(trtllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Trtllm(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Mlx(resp) => match resp.response {
                Some(mlx::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Mlx(chunk))
                }
                Some(mlx::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Mlx(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::TokenSpeed(resp) => match resp.response {
                Some(tokenspeed::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::TokenSpeed(chunk))
                }
                Some(tokenspeed::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::TokenSpeed(complete))
                }
                None => ProtoResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from GenerateResponse
pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    None,
}

/// Unified GenerateStreamChunk
#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
    Trtllm(trtllm::GenerateStreamChunk),
    Mlx(mlx::GenerateStreamChunk),
    TokenSpeed(tokenspeed::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            _ => panic!("Expected SGLang GenerateStreamChunk"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            _ => panic!("Expected vLLM GenerateStreamChunk"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateStreamChunk {
        match self {
            Self::Trtllm(chunk) => chunk,
            _ => panic!("Expected TensorRT-LLM GenerateStreamChunk"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is MLX
    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// How this chunk's payloads relate to the preceding ones — see
    /// [`ChunkSemantics`].
    pub fn chunk_semantics(&self) -> ChunkSemantics {
        match self {
            Self::Vllm(_) => ChunkSemantics::Delta,
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                ChunkSemantics::Cumulative
            }
        }
    }

    /// Get token IDs from chunk (common field)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
            Self::Trtllm(c) => &c.token_ids,
            Self::Mlx(c) => &c.token_ids,
            Self::TokenSpeed(c) => &c.token_ids,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
            Self::Mlx(c) => c.index,
            Self::TokenSpeed(c) => c.index,
        }
    }

    /// Get output logprobs.
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Vllm(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Trtllm(c) => convert_trtllm_output_logprobs(&c.logprobs),
            Self::Mlx(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
        }
    }

    /// Get input logprobs (SGLang and vLLM only - streaming chunks don't have prompt logprobs)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Vllm(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            // TRT-LLM, MLX, and TokenSpeed streaming chunks don't have input_logprobs
            Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get prompt tokens (cumulative)
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
            Self::Mlx(c) => c.prompt_tokens,
            Self::TokenSpeed(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
            Self::Mlx(c) => c.completion_tokens,
            Self::TokenSpeed(c) => c.completion_tokens,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
            Self::Mlx(c) => c.cached_tokens,
            Self::TokenSpeed(c) => c.cached_tokens,
        }
    }

    /// Get reasoning tokens (cumulative).
    pub fn reasoning_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.reasoning_tokens,
            Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 0,
        }
    }
}

/// Unified GenerateComplete response
#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    Vllm(vllm::GenerateComplete),
    Trtllm(trtllm::GenerateComplete),
    Mlx(mlx::GenerateComplete),
    TokenSpeed(tokenspeed::GenerateComplete),
}

impl ProtoGenerateComplete {
    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            _ => panic!("Expected vLLM GenerateComplete"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateComplete {
        match self {
            Self::Trtllm(complete) => complete,
            _ => panic!("Expected TensorRT-LLM GenerateComplete"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is MLX
    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// Chunk semantics of the stream this `Complete` terminates — see
    /// [`ChunkSemantics`]. `Delta` streams already accumulated their counts
    /// chunk by chunk; `Cumulative` streams report them here.
    pub fn chunk_semantics(&self) -> ChunkSemantics {
        match self {
            Self::Vllm(_) => ChunkSemantics::Delta,
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                ChunkSemantics::Cumulative
            }
        }
    }

    /// Get token IDs from either backend (output_ids in proto)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
            Self::Mlx(c) => &c.output_ids,
            Self::TokenSpeed(c) => &c.output_ids,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
            Self::Mlx(c) => c.prompt_tokens,
            Self::TokenSpeed(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
            Self::Mlx(c) => c.completion_tokens,
            Self::TokenSpeed(c) => c.completion_tokens,
        }
    }

    /// Get finish reason
    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
            Self::Trtllm(c) => &c.finish_reason,
            Self::Mlx(c) => &c.finish_reason,
            Self::TokenSpeed(c) => &c.finish_reason,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
            Self::Mlx(c) => c.index,
            Self::TokenSpeed(c) => c.index,
        }
    }

    /// Get matched stop as a JSON value
    ///
    /// Converts the backend-specific `oneof matched_stop` into a `serde_json::Value`:
    /// - MatchedTokenId → Number
    /// - MatchedStopStr → String
    /// - None → None
    pub fn matched_stop_json(&self) -> Option<serde_json::Value> {
        macro_rules! convert {
            ($oneof:expr, $token_id:path, $stop_str:path) => {
                $oneof.as_ref().map(|m| match m {
                    $token_id(id) => serde_json::Value::Number((*id).into()),
                    $stop_str(s) => serde_json::Value::String(s.clone()),
                })
            };
        }
        match self {
            Self::Sglang(c) => convert!(
                &c.matched_stop,
                SglangMatchedStop::MatchedTokenId,
                SglangMatchedStop::MatchedStopStr
            ),
            Self::Vllm(c) => convert!(
                &c.matched_stop,
                VllmMatchedStop::MatchedTokenId,
                VllmMatchedStop::MatchedStopStr
            ),
            Self::Trtllm(c) => convert!(
                &c.matched_stop,
                TrtllmMatchedStop::MatchedTokenId,
                TrtllmMatchedStop::MatchedStopStr
            ),
            Self::Mlx(c) => c
                .matched_stop_token_id
                .map(|id| serde_json::Value::Number(id.into())),
            Self::TokenSpeed(c) => convert!(
                &c.matched_stop,
                TokenSpeedMatchedStop::MatchedTokenId,
                TokenSpeedMatchedStop::MatchedStopStr
            ),
        }
    }

    /// Get output IDs (decode tokens only)
    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
            Self::Mlx(c) => &c.output_ids,
            Self::TokenSpeed(c) => &c.output_ids,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
            Self::Mlx(c) => c.cached_tokens,
            Self::TokenSpeed(c) => c.cached_tokens,
        }
    }

    /// Get reasoning tokens.
    pub fn reasoning_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.reasoning_tokens,
            Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 0,
        }
    }

    /// Get input/prompt logprobs (SGLang, vLLM, and TensorRT-LLM)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Vllm(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Trtllm(c) => {
                if c.prompt_logprobs.is_empty() {
                    None
                } else {
                    Some(ProtoInputLogProbs {
                        // First token has None logprob (no prior context)
                        token_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .enumerate()
                            .map(|(i, lp)| if i == 0 { None } else { Some(lp.logprob) })
                            .collect(),
                        token_ids: c.prompt_logprobs.iter().map(|lp| lp.token_id).collect(),
                        top_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .map(|lp| ProtoTopLogProbs {
                                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                                token_ids: lp.top_logprobs.iter().map(|t| t.token_id).collect(),
                            })
                            .collect(),
                    })
                }
            }
            // MLX and TokenSpeed do not have input_logprobs
            Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get output logprobs.
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Vllm(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Trtllm(c) => convert_trtllm_output_logprobs(&c.logprobs),
            Self::Mlx(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
        }
    }

    /// Get KV transfer parameters from prefill response (vLLM Mooncake PD only).
    /// Returns (remote_host, remote_port) if present.
    pub fn kv_transfer_params(&self) -> Option<(String, u32)> {
        match self {
            Self::Vllm(c) => c
                .kv_transfer_params
                .as_ref()
                .map(|params| (params.remote_host.clone(), params.remote_port)),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get opaque connector KV-transfer params JSON returned by the engine (vLLM only).
    pub fn kv_transfer_params_json(&self) -> Option<&str> {
        match self {
            Self::Vllm(c) => c
                .kv_transfer_params_json
                .as_deref()
                .filter(|s| !s.is_empty()),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }
}

/// Unified stream wrapper.
///
/// One variant per backend. Each yields its own native proto response shape;
/// the chunk / complete accessors above match on the corresponding
/// `ProtoGenerateStreamChunk` / `ProtoGenerateComplete` arm.
pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
    Trtllm(TrtllmStream),
    Mlx(MlxStream),
    TokenSpeed(TokenSpeedStream),
    /// ZMQ backend: a custom stream yielding vLLM-proto responses built from
    /// EngineCore outputs. Auto-aborts on drop, so `mark_completed` is a no-op.
    Zmq(ZmqGenerateStream),
}

impl ProtoStream {
    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        match self {
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
            Self::Trtllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Trtllm(Box::new(r)))),
            Self::Mlx(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Mlx(Box::new(r)))),
            Self::TokenSpeed(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::TokenSpeed(Box::new(r)))),
            // Every ZMQ engine (including TokenSpeed) emits vllm-shaped
            // responses: the adapter translates wire output into
            // `vllm::GenerateResponse`, so variant checks like
            // `is_tokenspeed()` on a response are unreliable for ZMQ-backed
            // streams. Key response-side engine logic on the worker's
            // `runtime_type()`, never on the response variant; key
            // accumulation on `chunk_semantics()`, which the vLLM shape
            // (delta chunks, cumulative `Complete`) defines for this lane.
            Self::Zmq(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
        }
    }

    /// Mark stream as completed (no abort needed)
    pub fn mark_completed(&mut self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
            Self::Trtllm(stream) => stream.mark_completed(),
            Self::Mlx(stream) => stream.mark_completed(),
            Self::TokenSpeed(stream) => stream.mark_completed(),
            Self::Zmq(stream) => stream.mark_completed(),
        }
    }

    /// Defer the abort-on-drop until the stream yields its first response.
    ///
    /// Used for the disaggregated decode leg: aborting a decode request while
    /// it is still receiving the KV handoff from its prefill peer can tear
    /// down the transfer mid-write; the first response proves the handoff
    /// completed, after which the (deferred) abort is safe. No-op for ZMQ
    /// streams — the ZMQ adapter owns its own lifecycle and the ZMQ lane does
    /// not serve disaggregated requests.
    #[must_use]
    pub fn defer_abort_until_first_item(self) -> Self {
        match self {
            Self::Sglang(stream) => Self::Sglang(stream.defer_abort_until_first_item()),
            Self::Vllm(stream) => Self::Vllm(stream.defer_abort_until_first_item()),
            Self::Trtllm(stream) => Self::Trtllm(stream.defer_abort_until_first_item()),
            Self::Mlx(stream) => Self::Mlx(stream.defer_abort_until_first_item()),
            Self::TokenSpeed(stream) => Self::TokenSpeed(stream.defer_abort_until_first_item()),
            Self::Zmq(stream) => Self::Zmq(stream),
        }
    }
}

/// Unified EmbedRequest that works with all backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
    Vllm(Box<vllm::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// Get SGLang variant
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang embed request"),
        }
    }

    /// Get mutable SGLang variant
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang embed request"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Clone the inner request (for passing to embed())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
        }
    }
}

/// Unified embed completion — both backends now use flat EmbedResponse
#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedResponse),
    Vllm(vllm::EmbedResponse),
}

impl ProtoEmbedComplete {
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(r) => &r.embedding,
            Self::Vllm(r) => &r.embedding,
        }
    }

    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(r) => r.prompt_tokens,
            Self::Vllm(r) => r.prompt_tokens,
        }
    }

    pub fn embedding_dim(&self) -> u32 {
        match self {
            Self::Sglang(r) => r.embedding_dim,
            Self::Vllm(r) => r.embedding_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;

    #[test]
    fn tokenspeed_image_into_proto_uses_itemized_payload() {
        let mut model_specific_tensors = HashMap::new();
        model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            TensorBytes {
                data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
            },
        );

        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::inline(
                    vec![42; 8],
                    vec![1, 2],
                    "float32".to_string(),
                ),
                model_specific_tensors,
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: false,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        assert_eq!(proto.items.len(), 1);
        let item = &proto.items[0];
        assert_eq!(item.modality, common::Modality::Image as i32);
        assert_eq!(item.placeholder_token_id, Some(151655));
        assert_eq!(item.placeholders[0].offset, 4);
        assert_eq!(item.placeholders[0].length, 2);
        assert_eq!(
            inline_tensor_data(item.encoder_input.as_ref().unwrap()),
            &[42; 8]
        );
        assert!(item.model_specific_tensors.contains_key("image_grid_thw"));
    }

    #[test]
    fn tokenspeed_video_into_proto_uses_itemized_payload() {
        let mut model_specific_tensors = HashMap::new();
        model_specific_tensors.insert(
            "video_grid_thw".to_string(),
            TensorBytes {
                data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
            },
        );

        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Video,
                encoder_input: TokenSpeedTensor::inline(
                    vec![42; 8],
                    vec![1, 2],
                    "float32".to_string(),
                ),
                model_specific_tensors,
                placeholder_token_id: Some(151656),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: false,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        assert_eq!(proto.items.len(), 1);
        let item = &proto.items[0];
        assert_eq!(item.modality, common::Modality::Video as i32);
        assert_eq!(item.placeholder_token_id, Some(151656));
        assert_eq!(item.placeholders[0].offset, 4);
        assert_eq!(item.placeholders[0].length, 2);
        assert_eq!(
            inline_tensor_data(item.encoder_input.as_ref().unwrap()),
            &[42; 8]
        );
        assert!(item.model_specific_tensors.contains_key("video_grid_thw"));
    }

    #[test]
    fn tokenspeed_tensor_data_uses_clean_payload_tags() {
        let tensor = tensor_bytes_to_tokenspeed(
            TensorBytes {
                data: vec![0xaa, 0xbb],
                shape: vec![2, 3],
                dtype: "uint32".to_string(),
            },
            false,
            0,
        );

        assert_eq!(
            tensor.encode_to_vec(),
            vec![
                0x0a, 0x02, 0x02, 0x03, // shape = 1, packed uint32 [2, 3]
                0x12, 0x06, b'u', b'i', b'n', b't', b'3', b'2', // dtype = 2
                0x1a, 0x02, 0xaa, 0xbb, // inline = 3
            ]
        );
    }

    #[test]
    fn tokenspeed_shm_encoder_input_into_proto_uses_shm_payload() {
        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::shm(
                    common::ShmHandle {
                        name: "smg-test-shm".to_string(),
                        offset: 0,
                        nbytes: 8,
                        owner_id: "smg:test".to_string(),
                    },
                    vec![1, 2],
                    "bfloat16".to_string(),
                ),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        let tensor = proto.items[0].encoder_input.as_ref().unwrap();
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.dtype, "bfloat16");
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Shm(handle)) => {
                assert_eq!(handle.name, "smg-test-shm");
                assert_eq!(handle.nbytes, 8);
            }
            _ => panic!("expected shm TensorData payload"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn tokenspeed_shm_writer_exposes_complete_mapped_payload() {
        let expected = [0x12, 0x34, 0x56, 0x78];
        let handle = write_tokenspeed_shm_with(expected.len(), |output| {
            output.copy_from_slice(&expected);
            Ok(())
        })
        .unwrap();
        let payload = std::fs::read(tokenspeed_shm_path(&handle.name));
        cleanup_mm_shm_handles(std::slice::from_ref(&handle));

        assert_eq!(payload.unwrap(), expected);
        assert!(!tokenspeed_shm_path(&handle.name).exists());
    }

    #[test]
    fn tokenspeed_remote_encoder_input_into_proto_uses_remote_payload() {
        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::remote(
                    common::RemoteTensorHandle {
                        transport: "nixl".to_string(),
                        descriptor: vec![1, 2, 3],
                        nbytes: 8,
                    },
                    vec![1, 2],
                    "bfloat16".to_string(),
                ),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        let tensor = proto.items[0].encoder_input.as_ref().unwrap();
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.dtype, "bfloat16");
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Remote(handle)) => {
                assert_eq!(handle.transport, "nixl");
                assert_eq!(handle.descriptor, vec![1, 2, 3]);
                assert_eq!(handle.nbytes, 8);
            }
            _ => panic!("expected remote TensorData payload"),
        }
    }

    fn inline_tensor_data(tensor: &tokenspeed::TensorData) -> &[u8] {
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Inline(data)) => data,
            _ => panic!("expected inline TensorData payload"),
        }
    }

    #[test]
    fn set_data_parallel_rank_per_engine() {
        let mut vllm_req = ProtoGenerateRequest::Vllm(Box::default());
        vllm_req.set_data_parallel_rank(2);
        assert!(matches!(
            &vllm_req,
            ProtoGenerateRequest::Vllm(req) if req.data_parallel_rank == Some(2)
        ));

        let mut sglang_req = ProtoGenerateRequest::Sglang(Box::default());
        sglang_req.set_data_parallel_rank(3);
        assert!(matches!(
            &sglang_req,
            ProtoGenerateRequest::Sglang(req) if req.data_parallel_rank == 3
        ));

        // Engines without the proto field ignore the pin
        let mut mlx_req = ProtoGenerateRequest::Mlx(Box::default());
        mlx_req.set_data_parallel_rank(1);
    }

    fn vllm_mm_data(modality: common::Modality) -> VllmMultimodalData {
        let is_video = modality == common::Modality::Video;
        VllmMultimodalData {
            pixel_values: vec![0u8; 16],
            pixel_values_shape: vec![1, 4],
            model_specific_tensors: HashMap::new(),
            im_token_id: Some(if is_video { 151656 } else { 151655 }),
            mm_placeholders: vec![(3, 4)],
            mm_hashes: vec!["h0".to_string()],
            batched_keys: vec![],
            flat_keys: HashMap::new(),
            keep_on_cpu_keys: vec![],
            modality,
            shm_enabled: false,
            shm_min_bytes: 0,
            rdma_enabled: false,
        }
    }

    #[test]
    fn vllm_modality_round_trips_into_proto() {
        // The video path must set the proto `modality` so the servicer routes to
        // vLLM's video modality; the image path must set image.
        let video = vllm_mm_data(common::Modality::Video).into_proto();
        assert_eq!(video.modality, common::Modality::Video as i32);
        assert_eq!(video.im_token_id, Some(151656));

        let image = vllm_mm_data(common::Modality::Image).into_proto();
        assert_eq!(image.modality, common::Modality::Image as i32);
    }
    #[test]
    fn extend_stop_token_ids_reaches_every_variant() {
        let ids = [7u32, 8];
        let mut req = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams::default()),
            ..Default::default()
        }));
        req.extend_stop_token_ids(&ids);
        match &req {
            ProtoGenerateRequest::Vllm(r) => {
                assert_eq!(r.sampling_params.as_ref().unwrap().stop_token_ids, ids);
            }
            _ => panic!("variant changed"),
        }

        // TRT-LLM keeps ids on the request itself.
        let mut req = ProtoGenerateRequest::Trtllm(Box::default());
        req.extend_stop_token_ids(&ids);
        match &req {
            ProtoGenerateRequest::Trtllm(r) => assert_eq!(r.stop_token_ids, ids),
            _ => panic!("variant changed"),
        }

        // Missing sampling params: untouched, no panic.
        let mut req = ProtoGenerateRequest::TokenSpeed(Box::default());
        req.extend_stop_token_ids(&ids);
    }

    #[test]
    fn chunk_semantics_follow_the_response_shape() {
        // The vLLM shape is the delta contract — the ZMQ lane emits it for
        // every engine it fronts, TokenSpeed included.
        assert!(
            ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk::default())
                .chunk_semantics()
                .is_delta()
        );
        assert!(
            ProtoGenerateComplete::Vllm(vllm::GenerateComplete::default())
                .chunk_semantics()
                .is_delta()
        );

        // Every other shape reports running totals.
        for chunk in [
            ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::Mlx(mlx::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk::default()),
        ] {
            assert_eq!(chunk.chunk_semantics(), ChunkSemantics::Cumulative);
        }
        for complete in [
            ProtoGenerateComplete::Sglang(sglang::GenerateComplete::default()),
            ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete::default()),
            ProtoGenerateComplete::Mlx(mlx::GenerateComplete::default()),
            ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete::default()),
        ] {
            assert_eq!(complete.chunk_semantics(), ChunkSemantics::Cumulative);
        }
    }
}

//! The shared-memory side of the multimodal payloads: the `/dev/shm`
//! segments a request's tensors ride in (write, reserve, name, probe and
//! sweep), the handles collected from a request so the segments are unlinked
//! once the engine has read them, and the request-finish paths that do the
//! unlinking. Moved out of `proto_wrapper.rs` as is; the timing wrapper and
//! the payload builder that reads it live in the multimodal module.

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
    time::{SystemTime, UNIX_EPOCH},
};

use memmap2::MmapOptions;
#[cfg(target_os = "linux")]
use rustix::fs::FallocateFlags;
use smg_grpc_client::{common_proto as common, tokenspeed_proto as tokenspeed, vllm_proto as vllm};

use super::{
    ProtoGenerateRequest, TokenSpeedMultimodalItem, TokenSpeedTensor, TokenSpeedTensorStorage,
    VLLM_MROPE_GRID_KEYS,
};

static TOKENSPEED_SHM_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn write_tokenspeed_shm(data: &[u8]) -> std::io::Result<common::ShmHandle> {
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
    vllm_mm: Option<(vllm::MultimodalInputs, Vec<vllm::MultimodalInputs>)>,
    build: impl FnOnce(Option<vllm::MultimodalInputs>) -> Result<vllm::GenerateRequest, String>,
) -> Result<ProtoGenerateRequest, String> {
    let (primary, extra) = match vllm_mm {
        Some((primary, extra)) => (Some(primary), extra),
        None => (None, Vec::new()),
    };
    let shm_handles: Vec<_> = primary
        .iter()
        .chain(&extra)
        .flat_map(collect_vllm_multimodal_inputs_shm_handles)
        .collect();
    match build(primary) {
        Ok(mut req) => {
            req.extra_mm_inputs = extra;
            Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
        }
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

/// The identity of a multimodal batch without its pixels, for the PD decode
/// side: hashes, placeholders and the M-RoPE grid tensors (a few ints per item,
/// inline payloads only, since an SHM segment is unreadable after the prefill
/// send). A hash-less batch carries no identity worth keeping.
pub(crate) fn vllm_mm_identity(mm: &vllm::MultimodalInputs) -> Option<vllm::MultimodalInputs> {
    if mm.mm_hashes.is_empty() {
        return None;
    }
    let inline_tensor = |key: &str| {
        mm.model_specific_tensors
            .get(key)
            .filter(|tensor| matches!(tensor.payload, Some(vllm::tensor_data::Payload::Inline(_))))
            .cloned()
    };
    let mut grid_tensors: HashMap<String, vllm::TensorData> = VLLM_MROPE_GRID_KEYS
        .iter()
        .filter_map(|key| inline_tensor(key).map(|tensor| (key.to_string(), tensor)))
        .collect();
    // A flat-classified grid key needs its sizes tensor to keep per-item
    // slicing on the decode side.
    let mut flat_keys = HashMap::new();
    for (key, sizes_key) in &mm.flat_keys {
        if !grid_tensors.contains_key(key) {
            continue;
        }
        match inline_tensor(sizes_key) {
            Some(sizes) => {
                grid_tensors.insert(sizes_key.clone(), sizes);
                flat_keys.insert(key.clone(), sizes_key.clone());
            }
            None => {
                tracing::warn!(
                    grid_key = %key,
                    sizes_key = %sizes_key,
                    "dropping grid tensor from the PD decode side: its sizes tensor is not inline"
                );
                grid_tensors.remove(key);
            }
        }
    }
    let retained = |keys: &[String]| {
        keys.iter()
            .filter(|key| grid_tensors.contains_key(*key))
            .cloned()
            .collect()
    };
    Some(vllm::MultimodalInputs {
        im_token_id: mm.im_token_id,
        mm_placeholders: mm.mm_placeholders.clone(),
        mm_hashes: mm.mm_hashes.clone(),
        modality: mm.modality,
        batched_keys: retained(&mm.batched_keys),
        keep_on_cpu_keys: retained(&mm.keep_on_cpu_keys),
        flat_keys,
        model_specific_tensors: grid_tensors,
        encoder_input_key: mm.encoder_input_key.clone(),
        ..Default::default()
    })
}

/// The content identity of a multimodal batch for a language-model-only PD
/// decode side: only the per-image hashes survive. The servicer folds them
/// into the request's `cache_salt`, so two requests whose text and expanded
/// placeholder-token runs match but whose images differ cannot alias in the
/// decode-side prefix cache. A hash-less batch carries no identity worth
/// keeping.
pub(super) fn vllm_mm_hash_identity(mm: &vllm::MultimodalInputs) -> Option<vllm::MultimodalInputs> {
    if mm.mm_hashes.is_empty() {
        return None;
    }
    Some(vllm::MultimodalInputs {
        mm_hashes: mm.mm_hashes.clone(),
        modality: mm.modality,
        ..Default::default()
    })
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
        .iter()
        .chain(&request.extra_mm_inputs)
        .flat_map(collect_vllm_multimodal_inputs_shm_handles)
        .collect()
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

pub(super) fn tokenspeed_shm_path(name: &str) -> PathBuf {
    PathBuf::from("/dev/shm").join(name)
}

//! Proto multimodal inputs → EngineCore `mm_features` for the direct-ZMQ path.
//!
//! The gRPC servicer converts the batched proto tensors into per-item engine
//! structures Python-side (`_build_preprocessed_mm_inputs` + the engine's
//! `from_hf_inputs` split). The ZMQ path bypasses that process, so the same
//! split happens here: batched keys index row `i`, flat keys slice by the
//! cumulative sizes tensor, everything else is shared (replicated per item).
//! Floating tensors are cast to the model dtype — the engine applies no cast
//! on this path.

use std::collections::{BTreeMap, HashMap, HashSet};

use bytes::Bytes;
use engine_zmq_client::{
    codec::{
        dtype::ModelDtype,
        tensor::{WireArrayData, WireTensor},
    },
    protocol::vllm::multimodal::{
        MmBatchedField, MmFeatureSpec, MmFeatures, MmField, MmFieldElem, MmFlatField, MmKwargValue,
        MmKwargsItem, MmSharedField, MmSlice, PlaceholderRange, SliceSpec,
    },
};
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};

/// A decoded (and dtype-cast) proto tensor ready for per-item slicing.
struct Decoded {
    dtype: String,
    shape: Vec<usize>,
    bytes: Bytes,
}

impl Decoded {
    fn elem_size(&self) -> Result<usize, String> {
        match self.dtype.as_str() {
            "bool" | "uint8" | "int8" => Ok(1),
            "float16" | "bfloat16" => Ok(2),
            "float32" | "uint32" | "int32" => Ok(4),
            "int64" | "float64" => Ok(8),
            other => Err(format!("unsupported multimodal tensor dtype {other:?}")),
        }
    }

    /// Bytes per index step along dim 0.
    fn row_nbytes(&self) -> Result<usize, String> {
        let inner: usize = self.shape.iter().skip(1).product();
        Ok(inner * self.elem_size()?)
    }

    /// Zero-copy view of rows `[start, stop)` along dim 0.
    fn slice_rows(&self, start: usize, stop: usize) -> Result<WireTensor, String> {
        let row = self.row_nbytes()?;
        let (lo, hi) = (start * row, stop * row);
        if hi > self.bytes.len() || start > stop {
            return Err(format!(
                "row slice {start}..{stop} out of bounds for tensor of {} bytes",
                self.bytes.len()
            ));
        }
        let mut shape = self.shape.clone();
        shape[0] = stop - start;
        Ok(WireTensor::from_raw_bytes(
            self.dtype.clone(),
            shape,
            self.bytes.slice(lo..hi),
        ))
    }

    fn whole(&self) -> WireTensor {
        WireTensor::from_raw_bytes(self.dtype.clone(), self.shape.clone(), self.bytes.clone())
    }

    /// Flattened values as widened i64 (sizes tensors are int64 or uint32).
    fn flat_i64(&self) -> Result<Vec<i64>, String> {
        match self.dtype.as_str() {
            "int64" => Ok((self.bytes.as_chunks::<8>().0.iter())
                .map(|c| i64::from_le_bytes(*c))
                .collect()),
            "uint32" => Ok((self.bytes.as_chunks::<4>().0.iter())
                .map(|c| i64::from(u32::from_le_bytes(*c)))
                .collect()),
            other => Err(format!("flat sizes tensor has unsupported dtype {other:?}")),
        }
    }
}

/// The raw bytes behind a `TensorData`, from whichever transport carries them:
/// inline, or a `/dev/shm` file the Router wrote (read once and unlinked, as
/// the Python servicer's `mm_shm` does). Remote handles are not implemented.
fn payload_bytes(
    name: &str,
    payload: Option<vllm::tensor_data::Payload>,
) -> Result<Vec<u8>, String> {
    match payload {
        Some(vllm::tensor_data::Payload::Inline(data)) => Ok(data),
        Some(vllm::tensor_data::Payload::Shm(handle)) => read_shm_payload(name, &handle),
        Some(vllm::tensor_data::Payload::Remote(_)) => Err(format!(
            "multimodal tensor {name:?}: TensorData.remote payload is not implemented yet"
        )),
        None => Err(format!("multimodal tensor {name:?} has no payload")),
    }
}

/// Whether a read `/dev/shm` tensor file is unlinked afterwards (the default;
/// `TOKENSPEED_UNLINK_MM_SHM_AFTER_READ=0` keeps it, for debugging).
#[cfg(unix)]
fn unlink_shm_after_read() -> bool {
    static UNLINK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *UNLINK.get_or_init(|| {
        !matches!(
            std::env::var("TOKENSPEED_UNLINK_MM_SHM_AFTER_READ")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no"
        )
    })
}

/// Reject path traversal, absolute and empty SHM names before opening.
fn validated_shm_name(name: &str) -> Result<&str, String> {
    let name = name.trim_start_matches('/');
    if name.is_empty() || name.contains('/') || name == "." || name == ".." || name.contains('\0') {
        return Err(format!("invalid TensorData.shm name {name:?}"));
    }
    Ok(name)
}

/// Read `nbytes` at `offset` from `/dev/shm/<name>`: the file must be a
/// regular file reached without following a symlink at the final component,
/// and is unlinked after the read unless disabled.
fn read_shm_payload(tensor: &str, handle: &common::ShmHandle) -> Result<Vec<u8>, String> {
    let name = validated_shm_name(&handle.name)?;
    read_shm_file(tensor, name, handle)
}

#[cfg(unix)]
fn read_shm_file(tensor: &str, name: &str, handle: &common::ShmHandle) -> Result<Vec<u8>, String> {
    use rustix::fs::{FileType, Mode, OFlags};

    let path = std::path::Path::new("/dev/shm").join(name);
    let nbytes = usize::try_from(handle.nbytes)
        .map_err(|_| format!("multimodal tensor {tensor:?}: shm nbytes out of range"))?;
    let fd = rustix::fs::open(
        &path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| format!("multimodal tensor {tensor:?}: cannot open shm {name:?}: {error}"))?;
    let read = (|| {
        let stat = rustix::fs::fstat(&fd)
            .map_err(|error| format!("multimodal tensor {tensor:?}: fstat {name:?}: {error}"))?;
        if !FileType::from_raw_mode(stat.st_mode).is_file() {
            return Err(format!(
                "multimodal tensor {tensor:?}: TensorData.shm is not a regular file: {name:?}"
            ));
        }
        // The range comes off the wire: checked against the file before the
        // buffer is sized by it (a bogus `nbytes` would otherwise abort the
        // process at allocation, not fail the request).
        let file_len = u64::try_from(stat.st_size).unwrap_or(0);
        if handle
            .offset
            .checked_add(handle.nbytes)
            .is_none_or(|end| end > file_len)
        {
            return Err(format!(
                "multimodal tensor {tensor:?}: TensorData.shm range {}+{} exceeds {name:?} \
                 ({file_len} bytes)",
                handle.offset, handle.nbytes
            ));
        }
        let mut data = vec![0u8; nbytes];
        let mut filled = 0usize;
        while filled < nbytes {
            let n = rustix::io::pread(&fd, &mut data[filled..], handle.offset + filled as u64)
                .map_err(|error| {
                    format!("multimodal tensor {tensor:?}: pread {name:?}: {error}")
                })?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled != nbytes {
            return Err(format!(
                "multimodal tensor {tensor:?}: TensorData.shm byte length mismatch for \
                 name={name:?}: expected {nbytes}, got {filled}"
            ));
        }
        Ok(data)
    })();
    drop(fd);
    if unlink_shm_after_read() {
        if let Err(error) = rustix::fs::unlink(&path) {
            if error != rustix::io::Errno::NOENT {
                tracing::warn!(%error, shm = name, "could not unlink a read multimodal shm file");
            }
        }
    }
    read
}

#[cfg(not(unix))]
fn read_shm_file(tensor: &str, name: &str, _handle: &common::ShmHandle) -> Result<Vec<u8>, String> {
    Err(format!(
        "multimodal tensor {tensor:?}: TensorData.shm {name:?} is not supported on this platform"
    ))
}

/// Little-endian float32 bytes from a floating payload of `dtype`, so every
/// floating tensor takes the one cast to the model dtype (the engine's own
/// frontend casts whatever dtype it is handed).
fn to_f32_bytes(name: &str, dtype: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    let elem = match dtype {
        "float16" | "bfloat16" => 2,
        "float64" => 8,
        other => {
            return Err(format!(
                "multimodal tensor {name:?} has unsupported floating dtype {other:?}"
            ))
        }
    };
    if !data.len().is_multiple_of(elem) {
        return Err(format!(
            "multimodal tensor {name:?} has {} bytes, not a multiple of its {dtype} element size",
            data.len()
        ));
    }
    let mut out = Vec::with_capacity(data.len() / elem * 4);
    for chunk in data.chunks_exact(elem) {
        let value = match dtype {
            "float16" => half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
            "bfloat16" => half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
            _ => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
        };
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

fn decode_tensor(
    name: &str,
    tensor: vllm::TensorData,
    model_dtype: ModelDtype,
) -> Result<Decoded, String> {
    let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
    let data = payload_bytes(name, tensor.payload)?;
    // Floating tensors are cast to the model dtype, mirroring the cast the
    // engine's own frontend applies; one already in that dtype is forwarded.
    let floating = matches!(
        tensor.dtype.as_str(),
        "float32" | "float16" | "bfloat16" | "float64"
    );
    if floating && tensor.dtype != model_dtype.as_str() {
        let f32_bytes = if tensor.dtype == "float32" {
            data
        } else {
            to_f32_bytes(name, &tensor.dtype, &data)?
        };
        let cast = WireTensor::from_f32_bytes_cast(model_dtype, shape.clone(), &f32_bytes)?;
        let WireArrayData::RawView(bytes) = cast.data else {
            return Err(format!("cast tensor {name:?} lost its raw view"));
        };
        return Ok(Decoded {
            dtype: cast.dtype,
            shape,
            bytes,
        });
    }
    let decoded = Decoded {
        dtype: tensor.dtype,
        shape,
        bytes: Bytes::from(data),
    };
    // Guard against a truncated or oversized inline payload: the engine would
    // otherwise reinterpret the raw buffer against the declared shape.
    let expected = decoded
        .shape
        .iter()
        .try_fold(decoded.elem_size()?, |acc, &d| acc.checked_mul(d));
    if expected != Some(decoded.bytes.len()) {
        return Err(format!(
            "multimodal tensor {name:?} has {} bytes, which does not match shape {:?} of dtype {:?}",
            decoded.bytes.len(),
            decoded.shape,
            decoded.dtype
        ));
    }
    Ok(decoded)
}

/// Rename generic keys for video inputs, mirroring the servicer's `mm_key`.
fn mm_key(key: &str, is_video: bool) -> String {
    if is_video && key == "pixel_values" {
        "pixel_values_videos".to_string()
    } else {
        key.to_string()
    }
}

/// Whether a batch carries tensors the preprocessed path can use: pixels, or
/// the grid-only form of a PD decode leg. A bare identity payload (hashes
/// only) is not, and rides `cache_salt` instead.
fn has_preprocessed_payload(mm: &vllm::MultimodalInputs) -> bool {
    mm.pixel_values.is_some() || !mm.model_specific_tensors.is_empty()
}

/// The cache salt a tensor-less identity payload folds its media hashes
/// into, as the Python servicer does (`mm_identity_cache_salt`): same-image
/// reuse still hits the decode prefix cache, different images behind the same
/// text no longer alias.
fn identity_cache_salt(batches: &[vllm::MultimodalInputs]) -> Option<String> {
    let hashes: Vec<&str> = batches
        .iter()
        .flat_map(|batch| batch.mm_hashes.iter().map(String::as_str))
        .collect();
    (!hashes.is_empty()).then(|| format!("mm:{}", hashes.join(",")))
}

/// What the request's multimodal batches (`mm_inputs` plus `extra_mm_inputs`)
/// become on the wire: per-item `mm_features` in prompt order, or, for a
/// tensor-less identity payload, the `cache_salt` carrying the hashes. The
/// rules are the Python servicer's: a pixel-less (grid-only) batch is a PD
/// decode leg and needs remote KV, or the engine would run the encoder with
/// nothing to encode.
pub(crate) fn translate_batches(
    batches: Vec<vllm::MultimodalInputs>,
    prompt_token_ids: &[u32],
    model_dtype: ModelDtype,
    has_kv_transfer: bool,
) -> Result<(Option<MmFeatures>, Option<String>), String> {
    if batches.is_empty() {
        return Ok((None, None));
    }
    if !batches.iter().any(has_preprocessed_payload) {
        return Ok((None, identity_cache_salt(&batches)));
    }
    if !has_kv_transfer
        && batches
            .iter()
            .any(|batch| has_preprocessed_payload(batch) && batch.pixel_values.is_none())
    {
        return Err(
            "multimodal payload carries grid tensors but no pixel_values and no \
                    kv_transfer_params; a pixel-less leg requires remote KV"
                .to_string(),
        );
    }
    let mut features: MmFeatures = Vec::new();
    for batch in batches {
        features.extend(build_mm_features(batch, prompt_token_ids, model_dtype)?);
    }
    features.sort_by_key(|feature| feature.mm_position.offset);
    Ok(((!features.is_empty()).then_some(features), None))
}

/// Build per-item `mm_features` from batched proto multimodal inputs.
pub(crate) fn build_mm_features(
    mm: vllm::MultimodalInputs,
    prompt_token_ids: &[u32],
    model_dtype: ModelDtype,
) -> Result<MmFeatures, String> {
    if !has_preprocessed_payload(&mm) {
        // A bare identity payload carries nothing the engine can attach; its
        // hashes ride `cache_salt` (see `translate_batches`).
        return Ok(Vec::new());
    }
    let num_items = mm.mm_placeholders.len();
    if num_items == 0 {
        // Tensors with nowhere to attach means malformed input — surface it
        // instead of silently building a text-only request.
        return Err("multimodal inputs carry tensors but no placeholders".to_string());
    }
    if mm.mm_hashes.len() != num_items {
        return Err(format!(
            "multimodal hash count {} does not match placeholder count {num_items}",
            mm.mm_hashes.len()
        ));
    }
    let is_video = mm.modality == common::Modality::Video as i32;
    let modality = if is_video { "video" } else { "image" };

    // Decode every tensor once, applying the video key rename. The primary
    // tensor travels in `pixel_values` but is named by the model's forward
    // kwarg (`encoder_input_key`, e.g. DeepSeek-V4.1's `patches`).
    let primary_key = mm
        .encoder_input_key
        .clone()
        .unwrap_or_else(|| "pixel_values".to_string());
    let mut tensors: BTreeMap<String, Decoded> = BTreeMap::new();
    if let Some(pixel_values) = mm.pixel_values {
        tensors.insert(
            mm_key(&primary_key, is_video),
            decode_tensor(&primary_key, pixel_values, model_dtype)?,
        );
    }
    for (key, tensor) in mm.model_specific_tensors {
        let decoded = decode_tensor(&key, tensor, model_dtype)?;
        tensors.insert(mm_key(&key, is_video), decoded);
    }

    let batched: HashSet<String> = mm
        .batched_keys
        .iter()
        .map(|k| mm_key(k, is_video))
        .collect();
    let flat: HashMap<String, String> = mm
        .flat_keys
        .iter()
        .map(|(k, v)| (mm_key(k, is_video), mm_key(v, is_video)))
        .collect();
    let keep_on_cpu: HashSet<String> = mm
        .keep_on_cpu_keys
        .iter()
        .map(|k| mm_key(k, is_video))
        .collect();

    // Split every kwarg into per-item elems.
    let mut items: Vec<MmKwargsItem> = vec![MmKwargsItem::new(); num_items];
    for (key, decoded) in &tensors {
        let on_cpu = keep_on_cpu.contains(key);
        if batched.contains(key) {
            if decoded.shape.first() != Some(&num_items) {
                return Err(format!(
                    "batched tensor {key:?} has leading dim {:?}, expected {num_items} items",
                    decoded.shape.first()
                ));
            }
            for (i, item) in items.iter_mut().enumerate() {
                let mut tensor = decoded.slice_rows(i, i + 1)?;
                tensor.shape.remove(0);
                item.insert(
                    key.clone(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(tensor)),
                        field: MmField::Batched(MmBatchedField {
                            keep_on_cpu: on_cpu,
                        }),
                    },
                );
            }
        } else if let Some(sizes_key) = flat.get(key) {
            let sizes = tensors
                .get(sizes_key)
                .ok_or_else(|| format!("flat sizes tensor {sizes_key:?} missing for {key:?}"))?
                .flat_i64()?;
            if sizes.len() != num_items {
                return Err(format!(
                    "flat sizes tensor {sizes_key:?} has {} entries, expected {num_items}",
                    sizes.len()
                ));
            }
            // Cumulative row offsets, and the full per-item slice list every
            // elem carries (the engine's flat field serializes all slices).
            let mut bounds = Vec::with_capacity(num_items + 1);
            let mut total = 0usize;
            bounds.push(total);
            for size in &sizes {
                let size = usize::try_from(*size)
                    .map_err(|_| format!("negative size in flat sizes tensor {sizes_key:?}"))?;
                total = total.checked_add(size).ok_or_else(|| {
                    format!("flat sizes tensor {sizes_key:?} sums past usize range")
                })?;
                bounds.push(total);
            }
            if decoded.shape.first() != Some(&total) {
                return Err(format!(
                    "flat tensor {key:?} has leading dim {:?}, expected {total} total rows",
                    decoded.shape.first(),
                ));
            }
            let slices: Vec<MmSlice> = bounds
                .windows(2)
                .map(|w| {
                    MmSlice::Slice(SliceSpec {
                        start: Some(w[0] as isize),
                        stop: Some(w[1] as isize),
                        step: None,
                    })
                })
                .collect();
            for (i, item) in items.iter_mut().enumerate() {
                item.insert(
                    key.clone(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(
                            decoded.slice_rows(bounds[i], bounds[i + 1])?,
                        )),
                        field: MmField::Flat(MmFlatField {
                            slices: slices.clone(),
                            dim: 0,
                            keep_on_cpu: on_cpu,
                        }),
                    },
                );
            }
        } else {
            // Shared: the full tensor replicated per item (the servicer's
            // fallback for keys in neither batched nor flat sets).
            for item in &mut items {
                item.insert(
                    key.clone(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(decoded.whole())),
                        field: MmField::Shared(MmSharedField {
                            batch_size: num_items,
                            keep_on_cpu: on_cpu,
                        }),
                    },
                );
            }
        }
    }

    // One feature per placeholder, in prompt-offset order.
    let mut features: MmFeatures = Vec::with_capacity(num_items);
    for ((placeholder, item), hash) in mm
        .mm_placeholders
        .iter()
        .zip(items)
        .zip(mm.mm_hashes.iter())
    {
        let offset = placeholder.offset as usize;
        let length = placeholder.length as usize;
        features.push(MmFeatureSpec {
            data: Some(item),
            modality: modality.to_string(),
            identifier: hash.clone(),
            mm_position: PlaceholderRange {
                offset,
                length,
                is_embed: is_embed_mask(prompt_token_ids, offset, length, mm.im_token_id)?,
            },
            mm_hash: Some(hash.clone()),
        });
    }
    features.sort_by_key(|f| f.mm_position.offset);
    Ok(features)
}

/// Boolean embed mask over a placeholder range: `true` where the prompt token
/// is the image token, excluding structural tokens (vision start/end markers)
/// from the embedding scatter. `None` when every position is an embed slot.
fn is_embed_mask(
    prompt_token_ids: &[u32],
    offset: usize,
    length: usize,
    im_token_id: Option<u32>,
) -> Result<Option<WireTensor>, String> {
    // Validate the range first — it must hold regardless of whether a mask is
    // needed, so an absent `im_token_id` can't skip the bounds check.
    let end = offset
        .checked_add(length)
        .filter(|&end| end <= prompt_token_ids.len())
        .ok_or_else(|| {
            format!(
                "placeholder range {offset}+{length} exceeds prompt of {} tokens",
                prompt_token_ids.len()
            )
        })?;
    let Some(im_token_id) = im_token_id else {
        return Ok(None);
    };
    let mask: Vec<bool> = prompt_token_ids[offset..end]
        .iter()
        .map(|&id| id == im_token_id)
        .collect();
    if mask.iter().all(|&m| m) {
        return Ok(None);
    }
    Ok(Some(WireTensor::from_bool(vec![length], mask)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inline_tensor(shape: Vec<u32>, dtype: &str, data: Vec<u8>) -> vllm::TensorData {
        vllm::TensorData {
            shape,
            dtype: dtype.to_string(),
            payload: Some(vllm::tensor_data::Payload::Inline(data)),
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn placeholders(ranges: &[(u32, u32)]) -> Vec<vllm::PlaceholderRange> {
        ranges
            .iter()
            .map(|&(offset, length)| vllm::PlaceholderRange { offset, length })
            .collect()
    }

    fn base_inputs() -> vllm::MultimodalInputs {
        vllm::MultimodalInputs {
            pixel_values: Some(inline_tensor(
                vec![2, 4],
                "float32",
                f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            )),
            model_specific_tensors: Default::default(),
            im_token_id: None,
            mm_placeholders: placeholders(&[(1, 3), (6, 3)]),
            mm_hashes: vec!["h0".to_string(), "h1".to_string()],
            batched_keys: vec!["pixel_values".to_string()],
            flat_keys: Default::default(),
            keep_on_cpu_keys: vec![],
            modality: common::Modality::Image as i32,
            encoder_input_key: None,
        }
    }

    /// DeepSeek-V4.1 names its primary tensor `patches`: the proto's
    /// `pixel_values` field lands under that key, sliced by the flat sizes
    /// tensor, and no `pixel_values` kwarg is emitted.
    #[test]
    fn encoder_input_key_renames_the_primary_tensor() {
        let mut mm = base_inputs();
        mm.encoder_input_key = Some("patches".to_string());
        mm.batched_keys = vec!["patches_per_image".to_string()];
        mm.flat_keys = HashMap::from([("patches".to_string(), "patches_per_image".to_string())]);
        mm.model_specific_tensors.insert(
            "patches_per_image".to_string(),
            inline_tensor(vec![2], "int64", i64_bytes(&[1, 1])),
        );
        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        assert_eq!(features.len(), 2);
        for feature in &features {
            let item = feature.data.as_ref().expect("item present");
            assert!(
                item.contains_key("patches"),
                "{:?}",
                item.keys().collect::<Vec<_>>()
            );
            assert!(!item.contains_key("pixel_values"));
            assert_eq!(tensor_of(&item["patches"]).shape, vec![1, 4]);
        }
    }

    /// Raw pixels from a worker-side pipeline arrive as `uint8` (the engine
    /// normalizes them on device) and are forwarded in that dtype, split per
    /// item like any batched tensor.
    #[test]
    fn uint8_pixels_pass_through_in_their_own_dtype() {
        let mut mm = base_inputs();
        mm.pixel_values = Some(inline_tensor(
            vec![2, 4],
            "uint8",
            vec![0, 1, 2, 3, 4, 5, 6, 7],
        ));
        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        assert_eq!(features.len(), 2);
        let item = features[1].data.as_ref().expect("item present");
        let tensor = tensor_of(&item["pixel_values"]);
        assert_eq!(tensor.dtype, "uint8");
        // Batched tensors are handed over per item with the batch dim removed.
        assert_eq!(tensor.shape, vec![4]);
        match &tensor.data {
            WireArrayData::RawView(bytes) => assert_eq!(bytes.as_ref(), &[4, 5, 6, 7]),
            other @ WireArrayData::AuxIndex(_) => panic!("expected a raw view, got {other:?}"),
        }
    }

    fn tensor_of(elem: &MmFieldElem) -> &WireTensor {
        match elem.data.as_ref().expect("data present") {
            MmKwargValue::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn batched_keys_split_per_row_and_cast_to_model_dtype() {
        let features =
            build_mm_features(base_inputs(), &[0; 9], ModelDtype::BFloat16).expect("built");
        assert_eq!(features.len(), 2);

        for (i, feature) in features.iter().enumerate() {
            assert_eq!(feature.modality, "image");
            assert_eq!(feature.identifier, format!("h{i}"));
            assert_eq!(feature.mm_hash.as_deref(), Some(format!("h{i}").as_str()));
            let item = feature.data.as_ref().expect("item present");
            let tensor = tensor_of(&item["pixel_values"]);
            // Row i of the [2, 4] float32 batch, cast to bfloat16.
            assert_eq!(tensor.dtype, "bfloat16");
            assert_eq!(tensor.shape, vec![4]);
            assert!(matches!(
                item["pixel_values"].field,
                MmField::Batched(MmBatchedField { keep_on_cpu: false })
            ));
        }
        assert_eq!(features[0].mm_position.offset, 1);
        assert_eq!(features[1].mm_position.offset, 6);
    }

    #[test]
    fn flat_keys_slice_by_cumulative_sizes() {
        let mut mm = base_inputs();
        mm.pixel_values = Some(inline_tensor(vec![5, 2], "float32", f32_bytes(&[0.0; 10])));
        mm.batched_keys = vec!["patches_per_image".to_string()];
        mm.flat_keys = [("pixel_values".to_string(), "patches_per_image".to_string())].into();
        mm.model_specific_tensors = [(
            "patches_per_image".to_string(),
            inline_tensor(vec![2], "int64", i64_bytes(&[2, 3])),
        )]
        .into();

        let features = build_mm_features(mm, &[0; 9], ModelDtype::Float32).expect("built");
        let item0 = features[0].data.as_ref().expect("item 0");
        let item1 = features[1].data.as_ref().expect("item 1");
        assert_eq!(tensor_of(&item0["pixel_values"]).shape, vec![2, 2]);
        assert_eq!(tensor_of(&item1["pixel_values"]).shape, vec![3, 2]);

        // Every elem carries the full per-item slice list.
        let expected_slices = vec![
            MmSlice::Slice(SliceSpec {
                start: Some(0),
                stop: Some(2),
                step: None,
            }),
            MmSlice::Slice(SliceSpec {
                start: Some(2),
                stop: Some(5),
                step: None,
            }),
        ];
        for item in [item0, item1] {
            let MmField::Flat(flat) = &item["pixel_values"].field else {
                panic!("expected flat field");
            };
            assert_eq!(flat.slices, expected_slices);
            assert_eq!(flat.dim, 0);
        }
    }

    #[test]
    fn unlisted_keys_are_shared_and_replicated() {
        let mut mm = base_inputs();
        mm.model_specific_tensors = [(
            "video_second_per_grid".to_string(),
            inline_tensor(vec![2], "int64", i64_bytes(&[1, 1])),
        )]
        .into();

        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        for feature in &features {
            let item = feature.data.as_ref().expect("item present");
            let elem = &item["video_second_per_grid"];
            assert_eq!(tensor_of(elem).shape, vec![2]);
            assert!(matches!(
                elem.field,
                MmField::Shared(MmSharedField {
                    batch_size: 2,
                    keep_on_cpu: false,
                })
            ));
        }
    }

    #[test]
    fn is_embed_masks_structural_tokens() {
        let mut mm = base_inputs();
        mm.im_token_id = Some(7);
        // Placeholder 0 covers tokens [7, 7, 5] (mixed); placeholder 1 covers
        // [7, 7, 7] (all image tokens).
        let prompt = [9, 7, 7, 5, 9, 9, 7, 7, 7];

        let features = build_mm_features(mm, &prompt, ModelDtype::BFloat16).expect("built");
        let mask = features[0]
            .mm_position
            .is_embed
            .as_ref()
            .expect("mixed range keeps a mask");
        assert_eq!(mask.dtype, "bool");
        assert_eq!(mask.shape, vec![3]);
        assert!(features[1].mm_position.is_embed.is_none());
    }

    #[test]
    fn video_renames_pixel_values() {
        let mut mm = base_inputs();
        mm.modality = common::Modality::Video as i32;

        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        let item = features[0].data.as_ref().expect("item present");
        assert!(item.contains_key("pixel_values_videos"));
        assert!(!item.contains_key("pixel_values"));
        assert_eq!(features[0].modality, "video");
    }

    #[test]
    fn rejects_hash_mismatch_remote_payloads_and_bad_shm_names() {
        let mut mm = base_inputs();
        mm.mm_hashes.pop();
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("hash mismatch");
        assert!(err.contains("hash count"), "{err}");

        let mut mm = base_inputs();
        mm.pixel_values = Some(vllm::TensorData {
            shape: vec![2, 4],
            dtype: "float32".to_string(),
            payload: Some(vllm::tensor_data::Payload::Remote(Default::default())),
        });
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("remote rejected");
        assert!(err.contains("not implemented"), "{err}");

        for bad in ["", "../etc/passwd", "a/b", ".", ".."] {
            let mut mm = base_inputs();
            mm.pixel_values = Some(vllm::TensorData {
                shape: vec![2, 4],
                dtype: "float32".to_string(),
                payload: Some(vllm::tensor_data::Payload::Shm(common::ShmHandle {
                    name: bad.to_string(),
                    offset: 0,
                    nbytes: 32,
                    owner_id: String::new(),
                })),
            });
            let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("bad name");
            assert!(
                err.contains("invalid TensorData.shm name"),
                "{bad:?}: {err}"
            );
        }
    }

    /// A `/dev/shm` payload is read at its offset, cast like an inline one,
    /// and unlinked once read (the Python servicer's `mm_shm` contract).
    #[test]
    #[cfg(target_os = "linux")]
    fn shm_payloads_are_read_at_offset_and_unlinked() {
        let name = format!(
            "smg-adapter-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        );
        let path = std::path::Path::new("/dev/shm").join(&name);
        let payload = f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let mut file_bytes = vec![0xAAu8; 8]; // an 8-byte prefix the offset skips
        file_bytes.extend_from_slice(&payload);
        std::fs::write(&path, &file_bytes).expect("write shm file");

        let mut mm = base_inputs();
        mm.pixel_values = Some(vllm::TensorData {
            shape: vec![2, 4],
            dtype: "float32".to_string(),
            payload: Some(vllm::tensor_data::Payload::Shm(common::ShmHandle {
                name: name.clone(),
                offset: 8,
                nbytes: payload.len() as u64,
                owner_id: "smg:test".to_string(),
            })),
        });
        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        assert_eq!(features.len(), 2);
        let item = features[1].data.as_ref().expect("item");
        let tensor = tensor_of(&item["pixel_values"]);
        // Batched: one row per item, the batch dim removed.
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.dtype.as_str(), "bfloat16");
        assert!(!path.exists(), "shm file is unlinked after the read");

        // A short file is a length mismatch, not a silently padded tensor.
        std::fs::write(&path, &file_bytes[..16]).expect("write short shm file");
        let mut mm = base_inputs();
        mm.pixel_values = Some(vllm::TensorData {
            shape: vec![2, 4],
            dtype: "float32".to_string(),
            payload: Some(vllm::tensor_data::Payload::Shm(common::ShmHandle {
                name: name.clone(),
                offset: 0,
                nbytes: payload.len() as u64,
                owner_id: String::new(),
            })),
        });
        let err = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect_err("short");
        assert!(err.contains("exceeds"), "{err}");
        assert!(!path.exists(), "a refused payload is still unlinked");

        // A range the wire claims but the file cannot hold is refused before
        // any buffer is sized by it.
        for (offset, nbytes) in [(0, 1u64 << 50), (u64::MAX - 1, 4)] {
            std::fs::write(&path, &file_bytes).expect("write shm file");
            let mut mm = base_inputs();
            mm.pixel_values = Some(vllm::TensorData {
                shape: vec![2, 4],
                dtype: "float32".to_string(),
                payload: Some(vllm::tensor_data::Payload::Shm(common::ShmHandle {
                    name: name.clone(),
                    offset,
                    nbytes,
                    owner_id: String::new(),
                })),
            });
            let err = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect_err("bogus");
            assert!(err.contains("exceeds"), "{err}");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Floating payloads of any dtype end up in the model dtype: one already
    /// there is forwarded, the others are converted through float32.
    #[test]
    fn half_precision_payloads_are_forwarded_or_cast() {
        let bf16_one = half::bf16::from_f32(1.0).to_le_bytes();
        let f16_two = half::f16::from_f32(2.0).to_le_bytes();
        let mut bf16_payload = Vec::new();
        for _ in 0..8 {
            bf16_payload.extend_from_slice(&bf16_one);
        }
        let mut f16_payload = Vec::new();
        for _ in 0..8 {
            f16_payload.extend_from_slice(&f16_two);
        }

        let mut mm = base_inputs();
        mm.pixel_values = Some(inline_tensor(vec![2, 4], "bfloat16", bf16_payload.clone()));
        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        let tensor = tensor_of(&features[0].data.as_ref().unwrap()["pixel_values"]);
        assert_eq!(tensor.dtype.as_str(), "bfloat16");
        let WireArrayData::RawView(bytes) = &tensor.data else {
            panic!("raw view");
        };
        assert_eq!(bytes.as_ref(), &bf16_payload[..8], "forwarded as is");

        let mut mm = base_inputs();
        mm.pixel_values = Some(inline_tensor(vec![2, 4], "float16", f16_payload));
        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        let tensor = tensor_of(&features[0].data.as_ref().unwrap()["pixel_values"]);
        assert_eq!(tensor.dtype.as_str(), "bfloat16");
        let WireArrayData::RawView(bytes) = &tensor.data else {
            panic!("raw view");
        };
        let expected: Vec<u8> = (0..4)
            .flat_map(|_| half::bf16::from_f32(2.0).to_le_bytes())
            .collect();
        assert_eq!(bytes.as_ref(), &expected[..], "converted through f32");
    }

    /// `translate_batches` is the request-level rule set: extra modality
    /// batches merge in prompt order, a tensor-less identity payload becomes a
    /// cache salt, and a grid-only (pixel-less) batch needs remote KV.
    #[test]
    fn translate_batches_merges_salts_and_gates_pixel_less_legs() {
        // Two batches: the second one's placeholders sit before the first's.
        let mut second = base_inputs();
        second.mm_placeholders = placeholders(&[(0, 1), (12, 1)]);
        second.pixel_values = Some(inline_tensor(vec![2, 4], "float32", f32_bytes(&[1.0; 8])));
        second.mm_hashes = vec!["h2".to_string(), "h3".to_string()];
        let (features, salt) = translate_batches(
            vec![base_inputs(), second],
            &[0; 13],
            ModelDtype::BFloat16,
            false,
        )
        .expect("translated");
        let features = features.expect("features");
        assert!(salt.is_none());
        assert_eq!(
            features
                .iter()
                .map(|f| f.mm_position.offset)
                .collect::<Vec<_>>(),
            vec![0, 1, 6, 12]
        );
        assert_eq!(
            features
                .iter()
                .map(|f| f.identifier.clone())
                .collect::<Vec<_>>(),
            vec!["h2", "h0", "h1", "h3"]
        );

        // Hashes only: nothing to attach, identity rides the cache salt.
        let identity = vllm::MultimodalInputs {
            mm_hashes: vec!["h0".to_string(), "h1".to_string()],
            ..Default::default()
        };
        let (features, salt) =
            translate_batches(vec![identity], &[], ModelDtype::BFloat16, false).expect("salted");
        assert!(features.is_none());
        assert_eq!(salt.as_deref(), Some("mm:h0,h1"));

        // Grid only (a PD decode leg): refused without remote KV, built with it.
        let grid_only = || {
            let mut mm = base_inputs();
            mm.pixel_values = None;
            mm.batched_keys = vec!["image_grid_thw".to_string()];
            mm.model_specific_tensors.insert(
                "image_grid_thw".to_string(),
                inline_tensor(vec![2, 3], "int64", i64_bytes(&[1, 2, 2, 1, 2, 2])),
            );
            mm
        };
        let err = translate_batches(vec![grid_only()], &[0; 9], ModelDtype::BFloat16, false)
            .expect_err("needs remote KV");
        assert!(err.contains("requires remote KV"), "{err}");
        let (features, _) =
            translate_batches(vec![grid_only()], &[0; 9], ModelDtype::BFloat16, true)
                .expect("decode leg");
        let features = features.expect("grid features");
        assert_eq!(features.len(), 2);
        let item = features[0].data.as_ref().expect("item");
        assert!(item.contains_key("image_grid_thw"));
        assert!(!item.contains_key("pixel_values"));
    }

    #[test]
    fn rejects_tensors_without_placeholders() {
        // A payload that carries tensors but no placeholders must not silently
        // degrade to a text-only request.
        let mut mm = base_inputs();
        mm.mm_placeholders = placeholders(&[]);
        mm.mm_hashes = vec![];
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16)
            .expect_err("tensors with no placeholders");
        assert!(err.contains("no placeholders"), "{err}");
    }

    #[test]
    fn rejects_tensor_byte_length_mismatch() {
        // int64 [2] needs 16 bytes; supply 8 so the buffer can't match the shape.
        let mut mm = base_inputs();
        mm.batched_keys = vec!["patches_per_image".to_string()];
        mm.model_specific_tensors = [(
            "patches_per_image".to_string(),
            inline_tensor(vec![2], "int64", i64_bytes(&[2])),
        )]
        .into();
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("truncated payload");
        assert!(err.contains("does not match shape"), "{err}");
    }

    #[test]
    fn shared_branch_preserves_keep_on_cpu() {
        let mut mm = base_inputs();
        mm.model_specific_tensors = [(
            "video_second_per_grid".to_string(),
            inline_tensor(vec![2], "int64", i64_bytes(&[1, 1])),
        )]
        .into();
        mm.keep_on_cpu_keys = vec!["video_second_per_grid".to_string()];

        let features = build_mm_features(mm, &[0; 9], ModelDtype::BFloat16).expect("built");
        let item = features[0].data.as_ref().expect("item present");
        assert!(matches!(
            item["video_second_per_grid"].field,
            MmField::Shared(MmSharedField {
                keep_on_cpu: true,
                ..
            })
        ));
    }

    #[test]
    fn validates_placeholder_range_without_im_token() {
        // With no im_token_id the range check must still run.
        let mut mm = base_inputs();
        mm.im_token_id = None;
        let prompt = [9, 7, 7, 5, 9]; // 5 tokens; placeholder 1 spans [6, 9)
        let err = build_mm_features(mm, &prompt, ModelDtype::BFloat16)
            .expect_err("out-of-range placeholder");
        assert!(err.contains("exceeds prompt"), "{err}");
    }
}

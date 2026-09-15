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
            "bool" => Ok(1),
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

fn decode_tensor(
    name: &str,
    tensor: vllm::TensorData,
    model_dtype: ModelDtype,
) -> Result<Decoded, String> {
    let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
    let data = match tensor.payload {
        Some(vllm::tensor_data::Payload::Inline(data)) => data,
        Some(_) => {
            return Err(format!(
                "multimodal tensor {name:?} uses a non-inline payload; the ZMQ wire carries \
                 tensors inline"
            ));
        }
        None => return Err(format!("multimodal tensor {name:?} has no payload")),
    };
    // Floating tensors arrive as float32 and are cast to the model dtype,
    // mirroring the cast the engine's own frontend applies.
    if tensor.dtype == "float32" {
        let cast = WireTensor::from_f32_bytes_cast(model_dtype, shape.clone(), &data)?;
        let WireArrayData::RawView(bytes) = cast.data else {
            return Err(format!("cast tensor {name:?} lost its raw view"));
        };
        return Ok(Decoded {
            dtype: cast.dtype,
            shape,
            bytes,
        });
    }
    // Non-float32 tensors are forwarded as-is (integer/bool kwargs like the
    // flat sizes or grid tensors). A floating dtype other than float32 would
    // reach the engine uncast — reject it rather than produce garbage.
    if matches!(tensor.dtype.as_str(), "float16" | "bfloat16" | "float64") {
        return Err(format!(
            "multimodal tensor {name:?} has floating dtype {:?}; the ZMQ path expects float32 \
             so it can cast to the model dtype",
            tensor.dtype
        ));
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

/// Build per-item `mm_features` from batched proto multimodal inputs.
pub(crate) fn build_mm_features(
    mm: vllm::MultimodalInputs,
    prompt_token_ids: &[u32],
    model_dtype: ModelDtype,
) -> Result<MmFeatures, String> {
    let num_items = mm.mm_placeholders.len();
    if num_items == 0 {
        // No placeholders is only valid for a genuinely empty payload. Tensors
        // or hashes with nowhere to attach means malformed input — surface it
        // instead of silently building a text-only request.
        if mm.pixel_values.is_some()
            || !mm.model_specific_tensors.is_empty()
            || !mm.mm_hashes.is_empty()
        {
            return Err(
                "multimodal inputs carry tensors or hashes but no placeholders".to_string(),
            );
        }
        return Ok(Vec::new());
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
    fn rejects_hash_mismatch_and_non_inline_payloads() {
        let mut mm = base_inputs();
        mm.mm_hashes.pop();
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("hash mismatch");
        assert!(err.contains("hash count"), "{err}");

        let mut mm = base_inputs();
        mm.pixel_values = Some(vllm::TensorData {
            shape: vec![2, 4],
            dtype: "float32".to_string(),
            payload: Some(vllm::tensor_data::Payload::Shm(Default::default())),
        });
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16).expect_err("shm rejected");
        assert!(err.contains("inline"), "{err}");
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
    fn rejects_non_float32_floating_dtype() {
        // A bf16 pixel tensor would reach the engine uncast — reject it.
        let mut mm = base_inputs();
        mm.pixel_values = Some(inline_tensor(vec![2, 4], "bfloat16", vec![0u8; 16]));
        let err = build_mm_features(mm, &[], ModelDtype::BFloat16)
            .expect_err("non-float32 floating dtype");
        assert!(err.contains("float32"), "{err}");
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

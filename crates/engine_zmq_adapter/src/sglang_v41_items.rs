//! DeepSeek-V4.1 image batches as SGLang's own processor lays them out.
//!
//! The Router sends a V4.1 image batch as one tensor of every image's
//! patches, with each image's patch grid (`vit_grid`), patch count
//! (`patches_per_image`) and token count (`types_per_image`) beside it: the
//! layout a vLLM engine splits by the batch's field layouts. SGLang's V4.1
//! model encodes one image per item and reads its grid as `n_vit_h` /
//! `n_vit_w` (its `multimodal/processors/deepseek_v41.py` and
//! `models/deepseek_v4.py`), so the batch is cut into those items here, by
//! the Python servicer's rule (`_deepseek_v41_items`). A batch whose counts
//! disagree is refused before the scheduler, where it would stop the engine.

use std::collections::BTreeMap;

use bytes::Bytes;
use engine_zmq_client::{
    codec::tensor::WireArrayData,
    protocol::sglang::multimodal::{SglangWireMmItem, SglangWireModality, SglangWireTensor},
};

const PREFIX: &str = "DeepSeek-V4.1 image batch:";

/// The grids the Router's DeepSeek-V4.1 preprocessor sends with an image batch.
const GRID_KEYS: [&str; 2] = ["vit_grid", "patches_per_image"];

/// Whether an image batch's model tensors are the V4.1 grids.
pub(crate) fn is_v41_batch(tensors: &BTreeMap<String, SglangWireTensor>) -> bool {
    GRID_KEYS.iter().all(|key| tensors.contains_key(*key))
}

/// One item per image: its rows of the feature (viewed, not copied), its
/// placeholder range, and its grid as the 0-d integers `n_vit_h` and
/// `n_vit_w`. The batch's hash named every image's bytes at once, so the items
/// carry none: the scheduler, or the opt-in hasher, hashes each.
pub(crate) fn deepseek_v41_items(
    feature: Option<SglangWireTensor>,
    tensors: &BTreeMap<String, SglangWireTensor>,
    offsets: &[(u64, u64)],
) -> Result<Vec<SglangWireMmItem>, String> {
    let feature = feature
        .ok_or_else(|| format!("{PREFIX} no pixel_values to cut its per-image items from"))?;
    let grid_values = ints(tensors, "vit_grid")?;
    let patches = ints(tensors, "patches_per_image")?;
    let tokens = tensors
        .contains_key("types_per_image")
        .then(|| ints(tensors, "types_per_image"))
        .transpose()?;
    let (grids, odd) = grid_values.as_chunks::<2>();
    if !odd.is_empty() {
        return Err(format!(
            "{PREFIX} vit_grid holds {} values, not (height, width) pairs",
            grid_values.len()
        ));
    }
    let images = grids.len();
    if images == 0 {
        return Err(format!("{PREFIX} no images"));
    }
    for (name, count) in [
        ("placeholders", offsets.len()),
        ("patch counts", patches.len()),
        ("token counts", tokens.as_ref().map_or(images, Vec::len)),
    ] {
        if count != images {
            return Err(format!("{PREFIX} {images} images but {count} {name}"));
        }
    }
    let (rows, row_dims) = match feature.shape.split_first() {
        Some((&rows, row_dims)) if !row_dims.is_empty() => (rows, row_dims),
        _ => {
            return Err(format!(
                "{PREFIX} pixel_values has shape {:?}, not rows of patches",
                feature.shape
            ))
        }
    };
    let total = patches
        .iter()
        .try_fold(0u64, |sum, &count| sum.checked_add(count));
    if total != Some(rows) {
        return Err(format!(
            "{PREFIX} {} patches but the pixel tensor has {rows} rows",
            total.map_or_else(|| "more than u64".to_string(), |sum| sum.to_string())
        ));
    }
    let bytes = feature
        .data
        .as_raw_view()
        .ok_or_else(|| format!("{PREFIX} pixel_values are not inline"))?;
    let row_bytes = match rows {
        0 => 0,
        rows => bytes.len() / rows as usize,
    };
    let mut items = Vec::with_capacity(images);
    let mut start = 0usize;
    for (index, ((&[height, width], &count), &(first, last))) in
        grids.iter().zip(&patches).zip(offsets).enumerate()
    {
        if height == 0 || width == 0 || height.checked_mul(width) != Some(count) {
            return Err(format!(
                "{PREFIX} image {index} has a {height}x{width} grid but {count} patches"
            ));
        }
        let span = last - first + 1;
        if let Some(tokens) = &tokens {
            if span != tokens[index] {
                return Err(format!(
                    "{PREFIX} image {index} fills {span} prompt positions but has {} tokens",
                    tokens[index]
                ));
            }
        }
        let end = start + count as usize * row_bytes;
        let shape = std::iter::once(count)
            .chain(row_dims.iter().copied())
            .collect();
        let pixels = SglangWireTensor::new(feature.dtype.clone(), shape, bytes.slice(start..end))?;
        start = end;
        items.push(SglangWireMmItem {
            modality: SglangWireModality::Image,
            hash: None,
            offsets: Some(vec![(first, last)]),
            feature: Some(pixels),
            model_specific_data: BTreeMap::from([
                ("n_vit_h".to_string(), scalar(height)),
                ("n_vit_w".to_string(), scalar(width)),
            ]),
        });
    }
    Ok(items)
}

/// An integer tensor's values, as the Router writes its grids (`int64`, or
/// `uint32`).
fn ints(tensors: &BTreeMap<String, SglangWireTensor>, name: &str) -> Result<Vec<u64>, String> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| format!("{PREFIX} no {name}"))?;
    let bytes = tensor
        .data
        .as_raw_view()
        .ok_or_else(|| format!("{PREFIX} {name} is not inline"))?;
    match tensor.dtype.as_str() {
        "int64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&value| {
                u64::try_from(i64::from_le_bytes(value))
                    .map_err(|_| format!("{PREFIX} {name} holds a negative value"))
            })
            .collect(),
        "uint32" => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&value| u64::from(u32::from_le_bytes(value)))
            .collect()),
        dtype => Err(format!(
            "{PREFIX} {name} is {dtype}, not an int64 or uint32 tensor"
        )),
    }
}

/// A 0-d `int64` tensor: the scheduler reads it with `int(item.n_vit_h)`, as
/// it reads the plain integer SGLang's own processor stores.
fn scalar(value: u64) -> SglangWireTensor {
    SglangWireTensor {
        dtype: "int64".to_string(),
        shape: Vec::new(),
        data: WireArrayData::RawView(Bytes::copy_from_slice(&(value as i64).to_le_bytes())),
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::sglang_proto;

    use crate::sglang::translate_sglang_multimodal;

    /// Two images: a 2x3 and a 1x2 patch grid, 6 + 2 patches of 3x2x2 floats,
    /// filling token spans of 5 and 4 positions in a 12-token prompt.
    const GRIDS: [(i64, i64); 2] = [(2, 3), (1, 2)];
    const PATCHES: [i64; 2] = [6, 2];
    const SPANS: [(u32, u32); 2] = [(1, 5), (7, 4)];
    const PROMPT_LEN: usize = 12;
    const ROW_FLOATS: usize = 3 * 2 * 2;

    fn tensor(dtype: &str, shape: Vec<u32>, data: Vec<u8>) -> sglang_proto::TensorData {
        sglang_proto::TensorData {
            data: data.into(),
            shape,
            dtype: dtype.to_string(),
        }
    }

    fn int64(values: &[i64], shape: Vec<u32>) -> sglang_proto::TensorData {
        tensor(
            "int64",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }

    /// A V4.1 image batch as the Router assembles it: every float of patch
    /// row `i` is `i`.
    fn v41_batch(
        grids: &[(i64, i64)],
        patches: &[i64],
        spans: &[(u32, u32)],
    ) -> sglang_proto::MultimodalInputs {
        let rows: i64 = PATCHES.iter().sum();
        let pixels: Vec<u8> = (0..rows)
            .flat_map(|row| std::iter::repeat_n(row as f32, ROW_FLOATS))
            .flat_map(f32::to_le_bytes)
            .collect();
        let flat_grids: Vec<i64> = grids.iter().flat_map(|&(h, w)| [h, w]).collect();
        let tokens: Vec<i64> = spans.iter().map(|&(_, n)| i64::from(n)).collect();
        let types = vec![0; tokens.iter().sum::<i64>() as usize];
        sglang_proto::MultimodalInputs {
            modalities: vec!["image".to_string()],
            pixel_values: Some(tensor("float32", vec![rows as u32, 3, 2, 2], pixels)),
            model_specific_tensors: [
                ("vit_grid", int64(&flat_grids, vec![grids.len() as u32, 2])),
                ("llm_grid", int64(&flat_grids, vec![grids.len() as u32, 2])),
                (
                    "patches_per_image",
                    int64(patches, vec![patches.len() as u32]),
                ),
                ("types_per_image", int64(&tokens, vec![tokens.len() as u32])),
                ("types", int64(&types, vec![types.len() as u32])),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
            im_token_id: Some(9),
            mm_placeholders: spans
                .iter()
                .map(|&(offset, length)| sglang_proto::PlaceholderRange { offset, length })
                .collect(),
            ..Default::default()
        }
    }

    fn scalar_of(item: &super::SglangWireMmItem, key: &str) -> i64 {
        let tensor = &item.model_specific_data[key];
        assert_eq!((tensor.dtype.as_str(), tensor.shape.len()), ("int64", 0));
        let bytes = tensor.data.as_raw_view().unwrap();
        i64::from_le_bytes(bytes.as_ref().try_into().unwrap())
    }

    /// Each image's patches, span and grid make one item SGLang's V4.1 model
    /// can encode; the feature bytes are views of the request's, not copies.
    #[test]
    fn each_image_is_its_own_item() {
        let batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        let pixels_ptr = batch.pixel_values.as_ref().unwrap().data.as_ptr();
        let output = translate_sglang_multimodal(batch, Vec::new(), PROMPT_LEN).unwrap();
        assert_eq!(output.im_token_id, Some(9));
        assert_eq!(output.mm_items.len(), 2);
        let row_bytes = ROW_FLOATS * 4;
        for (index, item) in output.mm_items.iter().enumerate() {
            let (first, length) = SPANS[index];
            let (first, length) = (u64::from(first), u64::from(length));
            assert_eq!(item.offsets, Some(vec![(first, first + length - 1)]));
            assert_eq!(item.hash, None);
            assert_eq!(scalar_of(item, "n_vit_h"), GRIDS[index].0);
            assert_eq!(scalar_of(item, "n_vit_w"), GRIDS[index].1);
            assert_eq!(item.model_specific_data.len(), 2);
            let feature = item.feature.as_ref().unwrap();
            assert_eq!(feature.dtype, "float32");
            assert_eq!(feature.shape, vec![PATCHES[index] as u64, 3, 2, 2]);
            let bytes = feature.data.as_raw_view().unwrap();
            let first_row = if index == 0 { 0 } else { PATCHES[0] as usize };
            assert_eq!(
                bytes.as_ptr(),
                pixels_ptr.wrapping_add(first_row * row_bytes)
            );
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .step_by(ROW_FLOATS)
                .map(|&value| f32::from_le_bytes(value))
                .collect();
            let expected: Vec<f32> = (first_row..first_row + PATCHES[index] as usize)
                .map(|row| row as f32)
                .collect();
            assert_eq!(values, expected);
        }
    }

    /// A batch whose grids, patch counts, token counts and placeholders
    /// disagree never reaches the scheduler.
    #[test]
    fn counts_that_disagree_are_refused() {
        let refused = |batch: sglang_proto::MultimodalInputs, needle: &str| {
            let error = translate_sglang_multimodal(batch, Vec::new(), PROMPT_LEN).unwrap_err();
            assert!(error.contains(needle), "{error:?} does not name {needle:?}");
        };
        refused(
            v41_batch(&GRIDS, &PATCHES, &SPANS[..1]),
            "2 images but 1 placeholders",
        );
        refused(
            v41_batch(&GRIDS, &[6], &SPANS),
            "2 images but 1 patch counts",
        );
        refused(
            v41_batch(&GRIDS, &[5, 2], &SPANS),
            "7 patches but the pixel tensor has 8 rows",
        );
        refused(
            v41_batch(&[(2, 3), (1, 3)], &PATCHES, &SPANS),
            "image 1 has a 1x3 grid but 2 patches",
        );
        refused(
            v41_batch(&[(2, 3), (-1, -2)], &PATCHES, &SPANS),
            "vit_grid holds a negative value",
        );
        refused(
            v41_batch(
                &[(2, 3), (1, 2), (0, 5)],
                &[6, 2, 0],
                &[(1, 5), (7, 4), (11, 1)],
            ),
            "image 2 has a 0x5 grid but 0 patches",
        );
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch
            .model_specific_tensors
            .insert("types_per_image".into(), int64(&[5, 3], vec![2]));
        refused(batch, "image 1 fills 4 prompt positions but has 3 tokens");
        let mut pixel_less = v41_batch(&GRIDS, &PATCHES, &SPANS);
        pixel_less.pixel_values = None;
        pixel_less.feature_hash = Some(7);
        refused(pixel_less, "no pixel_values");
        let mut flat_pixels = v41_batch(&GRIDS, &PATCHES, &SPANS);
        let floats: Vec<u8> = (0..8).flat_map(|row| (row as f32).to_le_bytes()).collect();
        flat_pixels.pixel_values = Some(tensor("float32", vec![8], floats));
        refused(
            flat_pixels,
            "pixel_values has shape [8], not rows of patches",
        );
    }

    /// Empty grids make no items. Refused alone, and refused, not a panic,
    /// as the extra batch after a video batch, where the caller reads the
    /// first item of each batch.
    #[test]
    fn a_batch_without_images_is_refused() {
        let empty = v41_batch(&[], &[], &[]);
        let error = translate_sglang_multimodal(empty.clone(), Vec::new(), PROMPT_LEN).unwrap_err();
        assert!(
            error.contains("DeepSeek-V4.1 image batch: no images"),
            "{error:?}"
        );
        let video = sglang_proto::MultimodalInputs {
            modalities: vec!["video".to_string()],
            im_token_id: Some(8),
            pixel_values: Some(tensor("float32", vec![2], vec![0; 8])),
            mm_placeholders: vec![sglang_proto::PlaceholderRange {
                offset: 1,
                length: 3,
            }],
            ..Default::default()
        };
        let error = translate_sglang_multimodal(video, vec![empty], PROMPT_LEN).unwrap_err();
        assert!(error.contains("no images"), "{error:?}");
    }

    /// The token counts are optional, as in the Python servicer; a batch hash
    /// is not passed on to items it no longer names.
    #[test]
    fn without_token_counts_or_with_a_batch_hash_the_items_are_the_same() {
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch.model_specific_tensors.remove("types_per_image");
        batch.feature_hash = Some(7);
        let output = translate_sglang_multimodal(batch, Vec::new(), PROMPT_LEN).unwrap();
        assert_eq!(output.mm_items.len(), 2);
        assert!(output.mm_items.iter().all(|item| item.hash.is_none()));
    }

    /// A batch without the V4.1 grids (a Qwen-style `image_grid_thw`) stays
    /// one item.
    #[test]
    fn other_families_keep_one_item_per_batch() {
        let mut batch = v41_batch(&GRIDS, &PATCHES, &SPANS);
        batch.model_specific_tensors = [(
            "image_grid_thw".to_string(),
            int64(&[1, 2, 2, 1, 2, 2], vec![2, 3]),
        )]
        .into_iter()
        .collect();
        let output = translate_sglang_multimodal(batch, Vec::new(), PROMPT_LEN).unwrap();
        assert_eq!(output.mm_items.len(), 1);
        assert_eq!(output.mm_items[0].offsets, Some(vec![(1, 5), (7, 10)]));
        assert!(output.mm_items[0]
            .model_specific_data
            .contains_key("image_grid_thw"));
    }
}

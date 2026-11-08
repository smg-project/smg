//! The pixel-less identity of a multimodal batch on this wire, for the side
//! that did not run the encoder: a PD decode side and an encoder-cache prefill
//! side read it off the preprocessing side's completion. One rule for the
//! gateway and the servicer.

use std::collections::HashMap;

use crate::vllm_proto::{tensor_data::Payload, MultimodalInputs, TensorData};

/// Per-item grid metadata the engine reads to compute M-RoPE positions
/// (`_get_mrope_input_positions`); a few ints per item. The video timing has
/// two spellings: `second_per_grid_ts` from the Qwen-VL processors,
/// `video_second_per_grid` from the Qwen-Omni family and the gateway's own
/// processors. The Python servicer's `GRID_KEYS` mirrors this list.
pub const MROPE_GRID_KEYS: [&str; 4] = [
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "video_second_per_grid",
];

/// The identity of a multimodal batch without its pixels: the hashes, the
/// placeholders, the image token id and the modality, and the M-RoPE grid
/// tensors with inline payloads (a shared-memory segment is unreadable once
/// its reader has unlinked it) with the size tensors of the flat ones; every
/// other tensor goes with the pixels and the key lists are trimmed to what
/// stays. A hash-less batch carries no identity worth keeping.
pub fn identity(mm: &MultimodalInputs) -> Option<MultimodalInputs> {
    if mm.mm_hashes.is_empty() {
        return None;
    }
    let inline_tensor = |key: &str| {
        mm.model_specific_tensors
            .get(key)
            .filter(|tensor| matches!(tensor.payload, Some(Payload::Inline(_))))
            .cloned()
    };
    let mut grid_tensors: HashMap<String, TensorData> = MROPE_GRID_KEYS
        .iter()
        .filter_map(|key| inline_tensor(key).map(|tensor| (key.to_string(), tensor)))
        .collect();
    // A flat-classified grid key needs its sizes tensor to keep per-item
    // slicing on the reading side.
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
                    "dropping a grid tensor from the media identity: its sizes tensor is not inline"
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
    Some(MultimodalInputs {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common_proto::ShmHandle;

    fn inline(shape: Vec<u32>, values: &[i64]) -> TensorData {
        TensorData {
            shape,
            dtype: "int64".to_string(),
            payload: Some(Payload::Inline(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into(),
            )),
        }
    }

    fn shared(shape: Vec<u32>) -> TensorData {
        TensorData {
            shape,
            dtype: "int64".to_string(),
            payload: Some(Payload::Shm(ShmHandle {
                name: "a-segment".to_string(),
                offset: 0,
                nbytes: 48,
                owner_id: "test".to_string(),
            })),
        }
    }

    fn batch() -> MultimodalInputs {
        MultimodalInputs {
            mm_hashes: vec!["h0".to_string(), "h1".to_string()],
            pixel_values: Some(inline(vec![2, 4], &[0; 8])),
            batched_keys: vec![
                "pixel_values".to_string(),
                "image_grid_thw".to_string(),
                "video_grid_thw".to_string(),
                "patches_per_image".to_string(),
            ],
            keep_on_cpu_keys: vec![
                "image_grid_thw".to_string(),
                "patches_per_image".to_string(),
            ],
            model_specific_tensors: [
                (
                    "image_grid_thw".to_string(),
                    inline(vec![2, 3], &[1, 2, 2, 1, 2, 2]),
                ),
                ("video_grid_thw".to_string(), shared(vec![2, 3])),
                ("patches_per_image".to_string(), inline(vec![2], &[4, 4])),
            ]
            .into(),
            ..Default::default()
        }
    }

    /// The inline grid stays with its key-list entries; the pixels, a grid in
    /// shared memory and a non-grid tensor go.
    #[test]
    fn keeps_the_inline_grids_alone() {
        let stripped = identity(&batch()).expect("an identity");
        let mut kept: Vec<&String> = stripped.model_specific_tensors.keys().collect();
        kept.sort();
        assert_eq!(kept, ["image_grid_thw"]);
        assert!(stripped.pixel_values.is_none());
        assert_eq!(stripped.batched_keys, vec!["image_grid_thw"]);
        assert_eq!(stripped.keep_on_cpu_keys, vec!["image_grid_thw"]);
        assert_eq!(stripped.mm_hashes, vec!["h0", "h1"]);
    }

    /// A flat grid rides with its inline sizes tensor; without an inline sizes
    /// tensor it is dropped rather than left unsliceable.
    #[test]
    fn a_flat_grid_needs_its_inline_sizes() {
        let mut mm = batch();
        mm.batched_keys = vec!["pixel_values".to_string()];
        mm.flat_keys = [("image_grid_thw".to_string(), "grid_sizes".to_string())].into();
        mm.model_specific_tensors
            .insert("grid_sizes".to_string(), inline(vec![2], &[1, 1]));
        let with_sizes = identity(&mm).expect("an identity");
        assert!(with_sizes.model_specific_tensors.contains_key("grid_sizes"));
        assert_eq!(
            with_sizes
                .flat_keys
                .get("image_grid_thw")
                .map(String::as_str),
            Some("grid_sizes")
        );

        mm.model_specific_tensors
            .insert("grid_sizes".to_string(), shared(vec![2]));
        let without_sizes = identity(&mm).expect("an identity");
        assert!(without_sizes.model_specific_tensors.is_empty());
        assert!(without_sizes.flat_keys.is_empty());
    }

    /// A hash-less batch has no identity worth keeping.
    #[test]
    fn a_hash_less_batch_has_no_identity() {
        let mut mm = batch();
        mm.mm_hashes.clear();
        assert!(identity(&mm).is_none());
    }
}

//! The vLLM wire's batch message under the rule.

use super::{Batch, Kept};
use crate::vllm_proto::{tensor_data::Payload, MultimodalInputs};

impl Batch for MultimodalInputs {
    fn hashed(&self) -> bool {
        !self.mm_hashes.is_empty()
    }

    /// A shared-memory segment is unreadable once its reader has unlinked
    /// it: the inline payloads alone ride.
    fn readable(&self, key: &str) -> bool {
        self.model_specific_tensors
            .get(key)
            .is_some_and(|tensor| matches!(tensor.payload, Some(Payload::Inline(_))))
    }

    fn sizes_key(&self, key: &str) -> Option<&str> {
        self.flat_keys.get(key).map(String::as_str)
    }

    fn without_pixels(&self, kept: &Kept) -> Self {
        let retained = |keys: &[String]| keys.iter().filter(|key| kept.has(key)).cloned().collect();
        Self {
            im_token_id: self.im_token_id,
            mm_placeholders: self.mm_placeholders.clone(),
            mm_hashes: self.mm_hashes.clone(),
            modality: self.modality,
            batched_keys: retained(&self.batched_keys),
            keep_on_cpu_keys: retained(&self.keep_on_cpu_keys),
            flat_keys: kept.flat_keys().clone(),
            model_specific_tensors: self
                .model_specific_tensors
                .iter()
                .filter(|(key, _)| kept.has(key))
                .map(|(key, tensor)| (key.clone(), tensor.clone()))
                .collect(),
            encoder_input_key: self.encoder_input_key.clone(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common_proto::{Modality, ShmHandle},
        media_identity::identity,
        vllm_proto::{PlaceholderRange, TensorData},
    };

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

    /// The fields the rule carries as they are survive the cut: the token
    /// id, the modality, the placeholders, the hashes and the encoder
    /// input's key.
    #[test]
    fn the_passthrough_fields_survive() {
        let mut mm = batch();
        mm.im_token_id = Some(151_655);
        mm.modality = Modality::Video as i32;
        mm.mm_placeholders = vec![
            PlaceholderRange {
                offset: 3,
                length: 8,
            },
            PlaceholderRange {
                offset: 20,
                length: 8,
            },
        ];
        mm.encoder_input_key = Some("patches".to_string());
        let stripped = identity(&mm).expect("an identity");
        assert_eq!(stripped.im_token_id, Some(151_655));
        assert_eq!(stripped.modality, Modality::Video as i32);
        assert_eq!(stripped.mm_placeholders, mm.mm_placeholders);
        assert_eq!(stripped.mm_hashes, mm.mm_hashes);
        assert_eq!(stripped.encoder_input_key.as_deref(), Some("patches"));
    }

    /// A hash-less batch has no identity worth keeping.
    #[test]
    fn a_hash_less_batch_has_no_identity() {
        let mut mm = batch();
        mm.mm_hashes.clear();
        assert!(identity(&mm).is_none());
    }
}

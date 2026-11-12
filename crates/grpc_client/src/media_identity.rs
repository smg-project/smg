//! The identity of a multimodal batch without its pixels, for the side of a
//! disaggregated request that did not run the preprocessing: a decode side
//! reads it off the prefill side's, an encoder-cache prefill side off the
//! producer's completion. One rule for the gateway and the servicers, over
//! each wire's batch message through its [`Batch`] view: the vLLM message
//! has one here, the SGLang message has one beside the wire's check in
//! [`sglang`]. The TokenSpeed wire carries no identity: its decode side takes
//! the gateway's clone of every item's metadata whole, the encoder input
//! alone gone.

use std::collections::{HashMap, HashSet};

pub mod sglang;
mod vllm;

/// Per-item grid metadata the engine reads to compute M-RoPE positions
/// (`_get_mrope_input_positions`); a few ints per item. The video timing has
/// two spellings: `second_per_grid_ts` from the Qwen-VL processors,
/// `video_second_per_grid` from the Qwen-Omni family and the gateway's own
/// processors. The Python servicer's `GRID_KEYS` mirrors this list.
pub const GRID_KEYS: [&str; 4] = [
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "video_second_per_grid",
];

/// One wire's batch message as the rule reads it, and as the rule writes the
/// identity back in the same message.
pub trait Batch: Sized {
    /// Whether the batch carries the hashes an identity is keyed by; one
    /// without them has no identity worth keeping.
    fn hashed(&self) -> bool;

    /// Whether the tensor under `key` is there with a payload a side without
    /// the pixels' storage can read.
    fn readable(&self, key: &str) -> bool;

    /// The sizes key of a flat tensor key, on a wire that slices a flat
    /// tensor per item by a sizes tensor.
    fn sizes_key(&self, key: &str) -> Option<&str>;

    /// The batch without its pixels: its hashes, placeholders, token id and
    /// modality as they are, its tensors and key lists cut to `kept`.
    fn without_pixels(&self, kept: &Kept) -> Self;
}

/// The tensors an identity keeps: their keys, and the flat keys among them
/// with their sizes keys.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Kept {
    tensors: HashSet<String>,
    flat_keys: HashMap<String, String>,
}

impl Kept {
    /// Whether the tensor under `key` rides with the identity.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains(key)
    }

    /// The flat keys that ride, each with its sizes key.
    pub fn flat_keys(&self) -> &HashMap<String, String> {
        &self.flat_keys
    }
}

/// The identity of a batch: the hashes, the placeholders, the token id and
/// the modality, and the M-RoPE grid tensors ([`GRID_KEYS`]) the reading side
/// can read without the pixels' storage, each flat one with its sizes
/// tensor; every other tensor goes with the pixels and the key lists are
/// trimmed to what stays. A hash-less batch carries no identity worth
/// keeping.
pub fn identity<B: Batch>(batch: &B) -> Option<B> {
    if !batch.hashed() {
        return None;
    }
    let mut kept = Kept::default();
    for key in GRID_KEYS {
        if !batch.readable(key) {
            continue;
        }
        match batch.sizes_key(key) {
            None => {
                kept.tensors.insert(key.to_string());
            }
            // A flat grid key needs its sizes tensor to keep per-item
            // slicing on the reading side.
            Some(sizes_key) if batch.readable(sizes_key) => {
                kept.tensors.insert(key.to_string());
                kept.tensors.insert(sizes_key.to_string());
                kept.flat_keys.insert(key.to_string(), sizes_key.to_string());
            }
            Some(sizes_key) => tracing::warn!(
                grid_key = key,
                sizes_key,
                "dropping a grid tensor from the media identity: its sizes tensor is not readable without the pixels"
            ),
        }
    }
    Some(batch.without_pixels(&kept))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wire's view for the rule alone: the tensors by key with whether
    /// each is readable, the flat keys, the hashes; the identity written
    /// back is what the rule kept.
    #[derive(Debug, Default, Clone)]
    struct View {
        hashed: bool,
        tensors: HashMap<&'static str, bool>,
        flat_keys: HashMap<&'static str, &'static str>,
        kept: Option<Kept>,
    }

    impl Batch for View {
        fn hashed(&self) -> bool {
            self.hashed
        }

        fn readable(&self, key: &str) -> bool {
            self.tensors.get(key).copied().unwrap_or(false)
        }

        fn sizes_key(&self, key: &str) -> Option<&str> {
            self.flat_keys.get(key).copied()
        }

        fn without_pixels(&self, kept: &Kept) -> Self {
            View {
                kept: Some(kept.clone()),
                ..Default::default()
            }
        }
    }

    fn view(tensors: &[(&'static str, bool)], flat_keys: &[(&'static str, &'static str)]) -> View {
        View {
            hashed: true,
            tensors: tensors.iter().copied().collect(),
            flat_keys: flat_keys.iter().copied().collect(),
            kept: None,
        }
    }

    fn kept_of(view: &View) -> Kept {
        identity(view)
            .expect("an identity")
            .kept
            .expect("the kept tensors")
    }

    fn keys(kept: &Kept) -> Vec<&str> {
        let mut keys: Vec<&str> = kept.tensors.iter().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// Of the tensors, the readable grids ride alone: an unreadable grid and
    /// a readable non-grid tensor go with the pixels.
    #[test]
    fn the_readable_grids_ride_alone() {
        let kept = kept_of(&view(
            &[
                ("image_grid_thw", true),
                ("video_grid_thw", false),
                ("patches_per_image", true),
            ],
            &[],
        ));
        assert_eq!(keys(&kept), ["image_grid_thw"]);
        assert!(kept.flat_keys().is_empty());
    }

    /// A flat grid rides with its readable sizes tensor, and the pair is the
    /// identity's flat key; without a readable sizes tensor neither rides.
    #[test]
    fn a_flat_grid_rides_with_its_readable_sizes() {
        let flat = [("image_grid_thw", "grid_sizes")];
        let kept = kept_of(&view(
            &[("image_grid_thw", true), ("grid_sizes", true)],
            &flat,
        ));
        assert_eq!(keys(&kept), ["grid_sizes", "image_grid_thw"]);
        assert_eq!(
            kept.flat_keys().get("image_grid_thw").map(String::as_str),
            Some("grid_sizes")
        );

        let kept = kept_of(&view(
            &[("image_grid_thw", true), ("grid_sizes", false)],
            &flat,
        ));
        assert!(keys(&kept).is_empty());
        assert!(kept.flat_keys().is_empty());
    }

    /// A hash-less batch has no identity worth keeping.
    #[test]
    fn a_hash_less_batch_has_no_identity() {
        let mut hash_less = view(&[("image_grid_thw", true)], &[]);
        hash_less.hashed = false;
        assert!(identity(&hash_less).is_none());
    }
}

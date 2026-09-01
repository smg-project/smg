use std::collections::BTreeMap;

/// Compute a blake3 hex-digest hash for a single image's raw bytes.
pub fn hash_image(raw_bytes: &[u8]) -> String {
    blake3::hash(raw_bytes).to_hex().to_string()
}

/// Compute a blake3 hex-digest hash for an image whose decoded pixels depend on
/// a per-request resolution cap.
///
/// The hash identifies the *decoded* image, not the encoded payload: the same
/// bytes under a different `max_long_side_pixel` preprocess to different
/// pixels, and both the gateway's pixel cache and the `mm_hashes` handed to the
/// backend key off this value. Folding the cap in keeps those caches from
/// serving one resolution tier's tensors for another.
pub fn hash_image_with_resolution_cap(
    raw_bytes: &[u8],
    max_long_side_pixel: Option<u32>,
) -> String {
    let Some(cap) = max_long_side_pixel else {
        // No cap: keep the plain byte hash so existing entries stay valid.
        return hash_image(raw_bytes);
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(raw_bytes);
    hasher.update(b"max_long_side_pixel=");
    hasher.update(&cap.to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

/// TODO(yechan): Decide whether video hashes should cover the full encoded
/// payload or a normalized representation of the sampled frames.
/// Compute a blake3 hex-digest hash for a single video's raw bytes.
pub fn hash_video(raw_bytes: &[u8]) -> String {
    blake3::hash(raw_bytes).to_hex().to_string()
}

/// Compute a blake3 hex-digest hash for a single audio payload's raw bytes.
pub fn hash_audio(raw_bytes: &[u8]) -> String {
    blake3::hash(raw_bytes).to_hex().to_string()
}

/// Compute per-image hashes keyed by modality.
///
/// Returns a `BTreeMap` of per-modality hash lists,
/// e.g. `{"image": ["abc123...", "def456..."]}`.
pub fn hash_images(raw_bytes: &[impl AsRef<[u8]>]) -> BTreeMap<String, Vec<String>> {
    let hashes: Vec<String> = raw_bytes.iter().map(|b| hash_image(b.as_ref())).collect();
    let mut map = BTreeMap::new();
    if !hashes.is_empty() {
        map.insert("image".to_string(), hashes);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let data = b"test image bytes";
        assert_eq!(hash_image(data), hash_image(data));
    }

    #[test]
    fn test_hash_different_inputs() {
        let a = b"image A";
        let b = b"image B";
        assert_ne!(hash_image(a), hash_image(b));
    }

    #[test]
    fn test_hash_images_empty() {
        let empty: Vec<Vec<u8>> = vec![];
        let result = hash_images(&empty);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hash_images_keyed_by_modality() {
        let images = vec![b"img1".to_vec(), b"img2".to_vec()];
        let result = hash_images(&images);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("image"));
        assert_eq!(result["image"].len(), 2);
    }

    #[test]
    fn test_hash_is_hex() {
        let hash = hash_image(b"test");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash.len(), 64); // blake3 produces 256-bit = 64 hex chars
    }
}

//! Namespaced content hashing.
//!
//! A block stored under a LoRA adapter or a cache salt is not reusable by a
//! plain prompt, and two salts are not reusable by each other. The engines
//! fold these into their own block hashes; this crate recomputes content
//! hashes from token ids, so it folds them into the XXH3 seed instead. The
//! mixing is fixed (seed = 1337 + xxh3(lora_name, 0) + xxh3(cache_salt, 1),
//! wrapping), so a corpus hashed elsewhere by the same rule stays comparable. Empty strings count as absent, as a client
//! sending `lora_name = ""` means the base model.

use xxhash_rust::xxh3::{xxh3_64_with_seed, Xxh3};

use crate::{ContentHash, XXH3_SEED};

/// The XXH3 seed for a cache namespace; [`XXH3_SEED`] when there is none.
pub fn namespace_seed(lora_name: Option<&str>, cache_salt: Option<&str>) -> u64 {
    let mut seed = XXH3_SEED;
    if let Some(name) = lora_name.filter(|name| !name.is_empty()) {
        seed = seed.wrapping_add(xxh3_64_with_seed(name.as_bytes(), 0));
    }
    if let Some(salt) = cache_salt.filter(|salt| !salt.is_empty()) {
        seed = seed.wrapping_add(xxh3_64_with_seed(salt.as_bytes(), 1));
    }
    seed
}

/// [`crate::compute_content_hash`] under an explicit seed.
pub fn content_hash_with_seed(token_ids: &[u32], seed: u64) -> ContentHash {
    use std::hash::Hasher;
    let mut hasher = Xxh3::with_seed(seed);
    for &token in token_ids {
        hasher.write(&token.to_le_bytes());
    }
    ContentHash(hasher.finish())
}

/// [`crate::compute_request_content_hashes`] under an explicit seed (see
/// [`namespace_seed`]): one hash per full block of `block_size` tokens, the
/// partial tail ignored.
pub fn request_content_hashes_with_seed(
    token_ids: &[u32],
    block_size: usize,
    seed: u64,
) -> Vec<ContentHash> {
    if block_size == 0 {
        return Vec::new();
    }
    token_ids
        .chunks_exact(block_size)
        .map(|block| content_hash_with_seed(block, seed))
        .collect()
}

/// [`crate::compute_request_content_hashes`] under a cache namespace.
pub fn namespaced_request_content_hashes(
    token_ids: &[u32],
    block_size: usize,
    lora_name: Option<&str>,
    cache_salt: Option<&str>,
) -> Vec<ContentHash> {
    request_content_hashes_with_seed(token_ids, block_size, namespace_seed(lora_name, cache_salt))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The content hash of one block stored under a cache namespace. Equal to
    /// [`crate::compute_content_hash`] when the namespace is empty.
    fn namespaced_content_hash(
        token_ids: &[u32],
        lora_name: Option<&str>,
        cache_salt: Option<&str>,
    ) -> ContentHash {
        content_hash_with_seed(token_ids, namespace_seed(lora_name, cache_salt))
    }

    use crate::{compute_content_hash, compute_request_content_hashes};

    const TOKENS: [u32; 6] = [11, 22, 33, 44, 55, 66];

    #[test]
    fn no_namespace_is_the_plain_content_hash() {
        assert_eq!(namespace_seed(None, None), XXH3_SEED);
        assert_eq!(namespace_seed(Some(""), Some("")), XXH3_SEED);
        assert_eq!(
            namespaced_content_hash(&TOKENS, None, None),
            compute_content_hash(&TOKENS)
        );
        assert_eq!(
            namespaced_request_content_hashes(&TOKENS, 4, Some(""), None),
            compute_request_content_hashes(&TOKENS, 4)
        );
    }

    #[test]
    fn lora_and_salt_each_change_the_hash_and_do_not_collide() {
        let plain = namespaced_content_hash(&TOKENS, None, None);
        let lora = namespaced_content_hash(&TOKENS, Some("adapter"), None);
        let salt = namespaced_content_hash(&TOKENS, None, Some("adapter"));
        let both = namespaced_content_hash(&TOKENS, Some("adapter"), Some("adapter"));
        let other_salt = namespaced_content_hash(&TOKENS, None, Some("other"));
        let distinct = [plain, lora, salt, both, other_salt];
        for (i, a) in distinct.iter().enumerate() {
            for b in &distinct[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn seed_mixing_matches_the_documented_formula() {
        let expected = XXH3_SEED
            .wrapping_add(xxh3_64_with_seed(b"adapter", 0))
            .wrapping_add(xxh3_64_with_seed(b"salty", 1));
        assert_eq!(namespace_seed(Some("adapter"), Some("salty")), expected);
    }

    #[test]
    fn request_hashes_follow_the_block_grid() {
        let hashes = namespaced_request_content_hashes(&TOKENS, 4, Some("adapter"), None);
        assert_eq!(hashes.len(), 1);
        assert_eq!(
            hashes[0],
            namespaced_content_hash(&TOKENS[..4], Some("adapter"), None)
        );
        assert!(namespaced_request_content_hashes(&TOKENS, 0, None, None).is_empty());
    }
}

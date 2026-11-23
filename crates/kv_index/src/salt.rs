//! Namespaced content hashing.
//!
//! A block stored under a LoRA adapter or a cache salt is not reusable by a
//! plain prompt, and two salts are not reusable by each other. The engines
//! fold these into their own block hashes; this crate recomputes content
//! hashes from token ids, so it folds them into the XXH3 seed instead. The
//! mixing is fixed (seed = 1337 + xxh3(lora_name, 0) + xxh3(cache_salt, 1),
//! wrapping), so a corpus hashed elsewhere by the same rule stays comparable. Empty strings count as absent, as a client
//! sending `lora_name = ""` means the base model.
//!
//! A host with no token ids hashes fixed windows of a request's text
//! ([`request_text_hashes_with_seed`]). Their seed adds xxh3("text", 2) to
//! the namespace's, so a window of text and a block of token ids over the
//! same bytes never share a hash.

use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::{block::hash_token_ids, ContentHash, XXH3_SEED};

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

/// [`crate::compute_content_hash`] under an explicit seed, in the same form:
/// one shot over the block's bytes. The streaming hasher derives a seeded
/// secret per instance before it sees a byte, which outweighs hashing a
/// 16-token block. Same digest either way (the tests pin that). The plain
/// hash and this one share one inlined body, so the plain hash's seed folds
/// as a constant.
pub fn content_hash_with_seed(token_ids: &[u32], seed: u64) -> ContentHash {
    ContentHash(hash_token_ids(token_ids, seed))
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

/// The seed a namespace's text windows hash under: the namespace's own
/// with the text kind folded in, as [`namespace_seed`] folds its parts.
fn text_seed(seed: u64) -> u64 {
    seed.wrapping_add(xxh3_64_with_seed(b"text", 2))
}

/// The text form of [`request_content_hashes_with_seed`], for a host with
/// no token ids: one hash per full window of `window` bytes of the text,
/// the partial tail ignored. A window is of the text's bytes as the request
/// carries them, so it may end inside a character. `seed` is the
/// namespace's (see [`namespace_seed`]).
pub fn request_text_hashes_with_seed(text: &[u8], window: usize, seed: u64) -> Vec<ContentHash> {
    if window == 0 {
        return Vec::new();
    }
    let seed = text_seed(seed);
    text.chunks_exact(window)
        .map(|bytes| ContentHash(xxh3_64_with_seed(bytes, seed)))
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
    use std::hash::Hasher;

    use xxhash_rust::xxh3::Xxh3;

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

    #[test]
    fn text_windows_follow_the_byte_grid_under_the_documented_seed() {
        let seed = namespace_seed(Some("adapter"), None);
        let window = |bytes: &[u8]| {
            ContentHash(xxh3_64_with_seed(
                bytes,
                seed.wrapping_add(xxh3_64_with_seed(b"text", 2)),
            ))
        };
        // Ten bytes are two windows of four: the tail of two is not hashed.
        assert_eq!(
            request_text_hashes_with_seed(b"abcdefghij", 4, seed),
            [window(b"abcd"), window(b"efgh")]
        );
        // A window ends where its bytes end, inside a character or not.
        let text = "abc\u{e9}";
        assert_eq!(text.len(), 5);
        assert_eq!(
            request_text_hashes_with_seed(text.as_bytes(), 4, seed),
            [window(&text.as_bytes()[..4])]
        );
        assert!(request_text_hashes_with_seed(b"abc", 4, seed).is_empty());
        assert!(request_text_hashes_with_seed(b"abcd", 0, seed).is_empty());
    }

    #[test]
    fn a_text_window_keeps_to_its_kind_and_its_namespace() {
        // One id whose little-endian bytes are the window's own.
        let id = u32::from_le_bytes(*b"abcd");
        let text = |seed| request_text_hashes_with_seed(b"abcd", 4, seed);
        assert_ne!(
            text(XXH3_SEED),
            [content_hash_with_seed(&[id], XXH3_SEED)],
            "a token block over the same bytes"
        );
        assert_ne!(
            text(XXH3_SEED),
            text(namespace_seed(None, Some("tenant"))),
            "another namespace"
        );
    }

    /// The streaming hasher's digest of the same ids: the rule a corpus
    /// hashed elsewhere follows, so the one-shot form must equal it under
    /// every seed.
    fn streaming_hash(token_ids: &[u32], seed: u64) -> ContentHash {
        let mut hasher = Xxh3::with_seed(seed);
        for &token in token_ids {
            hasher.write(&token.to_le_bytes());
        }
        ContentHash(hasher.finish())
    }

    #[test]
    fn a_seeded_hash_is_the_streaming_hashers_digest() {
        let tokens: Vec<u32> = (0..300).map(|i| i * 7919 + 3).collect();
        let namespace = namespace_seed(Some("adapter"), Some("salty"));
        for seed in [0, XXH3_SEED, namespace, u64::MAX] {
            // Each regime of the hash (up to 16 bytes, up to 128, up to 240
            // and beyond). On a target of another byte order, both sides of
            // every bound of its copy too (32, 64 and 256 tokens).
            for len in [0, 1, 16, 32, 33, 64, 65, 255, 256, 257, 300] {
                assert_eq!(
                    content_hash_with_seed(&tokens[..len], seed),
                    streaming_hash(&tokens[..len], seed),
                    "seed {seed}, {len} tokens"
                );
            }
        }
    }

    /// Six blocks' hashes as values, plain and under a namespace. Workers
    /// report their blocks under these hashes and stored snapshots hold
    /// them, so a change in how a block is hashed may not move one. The
    /// lengths take each regime of the hash: up to 16 bytes, up to 128, up
    /// to 240, and beyond that up to 1,200.
    #[test]
    fn block_hashes_are_the_recorded_values() {
        let tokens: Vec<u32> = (0..300).map(|i| i * 7919 + 3).collect();
        let namespace = namespace_seed(Some("adapter"), Some("salty"));
        let recorded: [(usize, u64, u64); 6] = [
            (1, 0x8020_bf96_9d17_68ca, 0xeb4c_1e87_7ab1_a673),
            (16, 0x5232_5333_90ea_4d1e, 0x24cf_7f3e_4dd1_08b1),
            (33, 0x6822_087f_c70d_046a, 0x1b6c_1dfb_43f0_3eb1),
            (64, 0x4baf_d877_ac9f_d0a1, 0xf325_60f0_3962_ee10),
            (256, 0x1bb3_97e2_fe40_51ec, 0x2db7_b5a4_7ed0_6f9a),
            (300, 0xc837_2a8c_c85e_f860, 0xf164_947b_61d5_fd41),
        ];
        for (len, plain, namespaced) in recorded {
            let block = &tokens[..len];
            assert_eq!(compute_content_hash(block).0, plain, "{len} tokens");
            assert_eq!(
                content_hash_with_seed(block, namespace).0,
                namespaced,
                "{len} tokens under a namespace"
            );
        }
    }
}

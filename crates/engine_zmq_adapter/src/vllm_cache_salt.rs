//! The `cache_salt` of a vLLM engine request. The Python servicer has the
//! same rule (`smg_grpc_servicer/vllm/mm_salt.py`): change them together.

use std::fmt::Write;

use sha2::{Digest, Sha256};

/// The reserved prefix.
const RESERVED_PREFIX: &str = "mm:";
/// The domain tag of the digest that follows the prefix.
const IDENTITY_DOMAIN: &[u8] = b"smg.vllm.mm-identity.v1\0";

/// The limits of vLLM's HTTP front end (`validate_cache_salt` of vLLM 0.31),
/// which this wire bypasses. The length is in characters, not in bytes.
const MAX_CHARS: usize = 128;
const FORBIDDEN_CHARS: [char; 4] = ['@', '/', '\\', '\0'];

/// The engine's salt for a request salt and the media hashes of a tensor-less
/// side. Without hashes it is the request salt. With hashes it is a digest of
/// both. Requests with different salts then share no blocks, and the length is
/// the same for any number of images. A request salt outside vLLM's limits, or
/// with the reserved prefix, is an error.
pub(crate) fn engine_cache_salt(
    request_salt: Option<String>,
    identity_hashes: &[String],
) -> Result<Option<String>, String> {
    if let Some(salt) = &request_salt {
        validate_cache_salt(salt)?;
    }
    if identity_hashes.is_empty() {
        return Ok(request_salt);
    }
    let mut digest = Sha256::new();
    digest.update(IDENTITY_DOMAIN);
    match &request_salt {
        None => digest.update([0]),
        Some(salt) => {
            digest.update([1]);
            write_len_prefixed(&mut digest, salt);
        }
    }
    for hash in identity_hashes {
        write_len_prefixed(&mut digest, hash);
    }
    let mut engine_salt = RESERVED_PREFIX.to_string();
    for byte in digest.finalize() {
        let _ = write!(engine_salt, "{byte:02x}");
    }
    Ok(Some(engine_salt))
}

/// Refuse a request salt outside vLLM's limits or with the reserved prefix.
/// The error texts do not contain the salt. A salt can be a secret.
pub fn validate_cache_salt(salt: &str) -> Result<(), String> {
    if salt.is_empty() {
        return Err("cache_salt must not be empty".to_string());
    }
    if salt.chars().count() > MAX_CHARS {
        return Err(format!("cache_salt must be at most {MAX_CHARS} characters"));
    }
    if salt.contains(FORBIDDEN_CHARS) {
        return Err(r"cache_salt must not contain '@', '/', '\' or NUL".to_string());
    }
    if salt.starts_with(RESERVED_PREFIX) {
        return Err("cache_salt must not start with the reserved prefix 'mm:'".to_string());
    }
    Ok(())
}

fn write_len_prefixed(digest: &mut Sha256, text: &str) {
    digest.update((text.len() as u64).to_le_bytes());
    digest.update(text);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn salt(request_salt: Option<&str>, hashes: &[&str]) -> Option<String> {
        let hashes: Vec<String> = hashes.iter().copied().map(String::from).collect();
        engine_cache_salt(request_salt.map(String::from), &hashes).expect("a valid salt")
    }

    /// `grpc_servicer/tests/test_vllm_mm_salt.py` has the same vectors.
    #[test]
    fn media_hashes_give_the_digest_of_the_salt_and_the_hashes() {
        for (request_salt, hashes, digest) in [
            (
                None,
                &["h1", "h2"][..],
                "c67efe1de6d4c5ef289ccdfd6953f085e29da935ac501db9cd8512e6a39f185a",
            ),
            (
                Some("tenant-a"),
                &["h1", "h2"],
                "662beced721917156bf4b76f866d2c307d82dd3bcfa95262ec7104fa2ba0fa3f",
            ),
            (
                Some("tenant-b"),
                &["h1", "h2"],
                "905c55f34a7ed6802bd075bbf00211abf77cab562139443433b6edb664a9ec54",
            ),
            (
                Some("tenant-a"),
                &["h1h2"],
                "54ce43702e479ec4a01a1298d1077d8907e957821dc5a80b97ceea5bbc25f978",
            ),
        ] {
            assert_eq!(salt(request_salt, hashes), Some(format!("mm:{digest}")));
        }
    }

    /// The limit counts characters, as vLLM does.
    #[test]
    fn the_longest_salt_stays_inside_the_limits_for_any_number_of_images() {
        let (longest, image) = ("é".repeat(128), "a".repeat(64));
        assert_eq!(salt(Some(&longest), &[]).as_ref(), Some(&longest));
        for images in 1..=8 {
            let salt = salt(Some(&longest), &vec![image.as_str(); images]).expect("a salt");
            assert_eq!(salt.len(), 67, "{images} images");
            assert!(!salt.contains(FORBIDDEN_CHARS), "{images} images");
        }
    }

    #[test]
    fn a_bad_request_salt_is_refused_without_its_value() {
        for (salt, reason) in [
            (String::new(), "empty"),
            ("é".repeat(129), "at most 128 characters"),
            ("secret@".to_string(), "not contain"),
            ("secret/".to_string(), "not contain"),
            (r"secret\".to_string(), "not contain"),
            ("secret\0".to_string(), "not contain"),
            ("mm:secret".to_string(), "reserved prefix 'mm:'"),
        ] {
            for hashes in [vec![], vec!["h1".to_string()]] {
                let error = engine_cache_salt(Some(salt.clone()), &hashes).expect_err(reason);
                assert!(error.starts_with("cache_salt must"), "{error}");
                assert!(error.contains(reason), "{error}");
                assert!(!error.contains("secret"), "{error}");
            }
        }
    }
}

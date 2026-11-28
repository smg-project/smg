//! The engines' own KV block hashes, reproduced for verification.
//!
//! The gateway indexes SMG content hashes, never these; but recomputing what
//! an engine published tells whether a worker hashes the way its peers do
//! (same algorithm, seed, page size, token layout). The relay's opt-in check
//! ([`crate::kv_wire::Normalizer::with_hash_check`]) counts mismatches and
//! never drops an event, so a worker with a different hash setup shows up as
//! a counter long before it shows up as a routing miss.
//!
//! SGLang (`python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp` and
//! `mem_cache/utils.py` at b1bbd74f28): per page, `SHA256(prior digest ||
//! page tokens as u32 little-endian)`; the first page of an unseeded chain
//! hashes the page bytes alone; a request's `cache_salt` seeds the chain with
//! `SHA256(b"sglang-cache-salt-v1\0" || salt)`; the prior of a child node is
//! the last page digest of its parent; the published integer is the first
//! eight digest bytes big-endian as a signed i64. Under Eagle-family bigram
//! hashing a page's words are `t, t+1` per position.
//!
//! vLLM (`vllm/v1/core/kv_cache_utils.py` and `vllm/utils/hashing.py` at
//! 0c16eee3f1) with `--prefix-caching-hash-algo sha256_cbor` and
//! `PYTHONHASHSEED` unset: `NONE = sha256(cbor("vllm-none-hash"))`, and
//! `H_i = sha256(cbor([bstr(H_{i-1} or NONE), [uint tokens...], null | extra
//! keys]))` where the extra keys are arrays in the order LoRA
//! `["lora", name, path]`, multimodal `["mm", identifier, offset]`,
//! `["cache_salt", salt]` (block 0 only) and `["prompt_embeds", bstr]`;
//! the published integer is the last eight digest bytes big-endian, unsigned
//! (carried here as the same 64 bits in an i64). The other algorithms
//! (`sha256` over pickle, `xxhash*`) are not reproducible outside the
//! engine's process and are not attempted. The LoRA adapter path is inside
//! the hash but not in the events, so a LoRA block cannot be verified from
//! its event alone.

use sha2::{Digest, Sha256};

/// A full SHA-256 digest, what the engines chain on.
pub type Digest32 = [u8; 32];

/// Which engine hash a worker publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineHash {
    /// SGLang's per-page SHA-256 chain.
    Sglang,
    /// vLLM's `sha256_cbor` with the default seed.
    VllmSha256Cbor,
}

impl EngineHash {
    /// The setting's spelling: `sglang` or `vllm-sha256-cbor` (underscores
    /// accepted), case-insensitive; `None` for anything else.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "sglang" => Some(Self::Sglang),
            "vllm-sha256-cbor" => Some(Self::VllmSha256Cbor),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::VllmSha256Cbor => "vllm-sha256-cbor",
        }
    }
}

// ---------------------------------------------------------------------------
// SGLang
// ---------------------------------------------------------------------------

/// The chain seed of a salted request.
pub fn sglang_salt_seed(cache_salt: &str) -> Digest32 {
    let mut hasher = Sha256::new();
    hasher.update(b"sglang-cache-salt-v1\0");
    hasher.update(cache_salt.as_bytes());
    hasher.finalize().into()
}

/// One page: the prior digest (if any) then the page's words, each as four
/// little-endian bytes. Under bigram hashing pass `t, t+1` per position.
pub fn sglang_page(prior: Option<&Digest32>, words: &[u32]) -> Digest32 {
    let mut hasher = Sha256::new();
    if let Some(prior) = prior {
        hasher.update(prior);
    }
    for word in words {
        hasher.update(word.to_le_bytes());
    }
    hasher.finalize().into()
}

/// The integer SGLang publishes for a digest.
pub fn sglang_event_int(digest: &Digest32) -> i64 {
    let mut top = [0u8; 8];
    top.copy_from_slice(&digest[..8]);
    i64::from_be_bytes(top)
}

/// Every full page of `tokens` chained from `prior`: `(digest, published
/// integer)` per page, as a request starting at `prior` would be stored.
pub fn sglang_chain(
    tokens: &[u32],
    page_size: usize,
    prior: Option<&Digest32>,
) -> Vec<(Digest32, i64)> {
    if page_size == 0 {
        return Vec::new();
    }
    let mut prior = prior.copied();
    tokens
        .chunks_exact(page_size)
        .map(|page| {
            let digest = sglang_page(prior.as_ref(), page);
            prior = Some(digest);
            (digest, sglang_event_int(&digest))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// vLLM sha256_cbor
// ---------------------------------------------------------------------------

/// One of the keys vLLM folds into a block hash, as the hash sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VllmExtraKey {
    Lora { name: String, path: String },
    Mm { identifier: String, offset: i64 },
    CacheSalt(String),
    PromptEmbeds(Vec<u8>),
}

const VLLM_NONE_HASH_SEED: &str = "vllm-none-hash";

/// `NONE_HASH`: the parent of a chain's first block.
pub fn vllm_none_hash() -> Digest32 {
    let mut hasher = Sha256::new();
    cbor::text(&mut hasher, VLLM_NONE_HASH_SEED);
    hasher.finalize().into()
}

/// One block: `sha256(cbor([parent, tokens, extra_keys]))`. The encoding is
/// fed to the hasher as it is produced, so a block costs the SHA-256 and no
/// buffer; the parent of a chain's first block is derived only for it.
pub fn vllm_block(
    parent: Option<&Digest32>,
    tokens: &[u32],
    extra_keys: Option<&[VllmExtraKey]>,
) -> Digest32 {
    let mut hasher = Sha256::new();
    cbor::array(&mut hasher, 3);
    match parent {
        Some(parent) => cbor::bytes(&mut hasher, parent),
        None => cbor::bytes(&mut hasher, &vllm_none_hash()),
    }
    cbor::array(&mut hasher, tokens.len());
    for &token in tokens {
        cbor::uint(&mut hasher, u64::from(token));
    }
    match extra_keys {
        None => cbor::null(&mut hasher),
        Some(keys) => {
            cbor::array(&mut hasher, keys.len());
            for key in keys {
                match key {
                    VllmExtraKey::Lora { name, path } => {
                        cbor::array(&mut hasher, 3);
                        cbor::text(&mut hasher, "lora");
                        cbor::text(&mut hasher, name);
                        cbor::text(&mut hasher, path);
                    }
                    VllmExtraKey::Mm { identifier, offset } => {
                        cbor::array(&mut hasher, 3);
                        cbor::text(&mut hasher, "mm");
                        cbor::text(&mut hasher, identifier);
                        cbor::int(&mut hasher, *offset);
                    }
                    VllmExtraKey::CacheSalt(salt) => {
                        cbor::array(&mut hasher, 2);
                        cbor::text(&mut hasher, "cache_salt");
                        cbor::text(&mut hasher, salt);
                    }
                    VllmExtraKey::PromptEmbeds(digest) => {
                        cbor::array(&mut hasher, 2);
                        cbor::text(&mut hasher, "prompt_embeds");
                        cbor::bytes(&mut hasher, digest);
                    }
                }
            }
        }
    }
    hasher.finalize().into()
}

/// The integer vLLM publishes for a digest (`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`):
/// the low 64 bits, carried as the same bit pattern in an i64.
pub fn vllm_event_int(digest: &Digest32) -> i64 {
    let mut low = [0u8; 8];
    low.copy_from_slice(&digest[24..]);
    i64::from_be_bytes(low)
}

/// Every full block of `tokens` chained from `parent` with no extra keys.
pub fn vllm_chain(
    tokens: &[u32],
    block_size: usize,
    parent: Option<&Digest32>,
) -> Vec<(Digest32, i64)> {
    if block_size == 0 {
        return Vec::new();
    }
    let mut parent = parent.copied();
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            let digest = vllm_block(parent.as_ref(), block, None);
            parent = Some(digest);
            (digest, vllm_event_int(&digest))
        })
        .collect()
}

/// The subset of canonical CBOR (RFC 8949 §4.2) that vLLM's hash input uses:
/// definite-length arrays, byte and text strings, integers and null. `cbor2`
/// with `canonical=True` produces exactly these bytes for those values. The
/// encoder writes into a `Sink`: a byte buffer, or a hasher that takes the
/// bytes as they come and never holds the whole encoding.
pub(crate) mod cbor {
    use sha2::{Digest, Sha256};

    /// Where the encoded bytes go.
    pub(crate) trait Sink {
        fn put(&mut self, bytes: &[u8]);
    }

    impl Sink for Vec<u8> {
        fn put(&mut self, bytes: &[u8]) {
            self.extend_from_slice(bytes);
        }
    }

    impl Sink for Sha256 {
        fn put(&mut self, bytes: &[u8]) {
            self.update(bytes);
        }
    }

    /// An item's head: the major type with its argument, one to nine bytes,
    /// handed to the sink in one piece.
    fn head(out: &mut impl Sink, major: u8, value: u64) {
        let major = major << 5;
        let mut bytes = [0u8; 9];
        let len = if value < 24 {
            bytes[0] = major | value as u8;
            1
        } else if value <= u64::from(u8::MAX) {
            bytes[0] = major | 24;
            bytes[1] = value as u8;
            2
        } else if value <= u64::from(u16::MAX) {
            bytes[0] = major | 25;
            bytes[1..3].copy_from_slice(&(value as u16).to_be_bytes());
            3
        } else if value <= u64::from(u32::MAX) {
            bytes[0] = major | 26;
            bytes[1..5].copy_from_slice(&(value as u32).to_be_bytes());
            5
        } else {
            bytes[0] = major | 27;
            bytes[1..9].copy_from_slice(&value.to_be_bytes());
            9
        };
        out.put(&bytes[..len]);
    }

    pub(crate) fn uint(out: &mut impl Sink, value: u64) {
        head(out, 0, value);
    }

    pub(crate) fn int(out: &mut impl Sink, value: i64) {
        if value >= 0 {
            head(out, 0, value as u64);
        } else {
            head(out, 1, !(value as u64));
        }
    }

    pub(crate) fn bytes(out: &mut impl Sink, value: &[u8]) {
        head(out, 2, value.len() as u64);
        out.put(value);
    }

    pub(crate) fn text(out: &mut impl Sink, value: &str) {
        head(out, 3, value.len() as u64);
        out.put(value.as_bytes());
    }

    pub(crate) fn array(out: &mut impl Sink, len: usize) {
        head(out, 4, len as u64);
    }

    pub(crate) fn null(out: &mut impl Sink) {
        out.put(&[0xf6]);
    }
}

#[cfg(test)]
#[path = "../tests/unit/engine_hash/tests.rs"]
mod tests;

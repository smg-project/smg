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
    let mut encoded = Vec::with_capacity(VLLM_NONE_HASH_SEED.len() + 1);
    cbor::text(&mut encoded, VLLM_NONE_HASH_SEED);
    Sha256::digest(&encoded).into()
}

/// One block: `sha256(cbor([parent, tokens, extra_keys]))`.
pub fn vllm_block(
    parent: Option<&Digest32>,
    tokens: &[u32],
    extra_keys: Option<&[VllmExtraKey]>,
) -> Digest32 {
    let none = vllm_none_hash();
    let parent = parent.unwrap_or(&none);
    let mut encoded = Vec::with_capacity(64 + tokens.len() * 3);
    cbor::array(&mut encoded, 3);
    cbor::bytes(&mut encoded, parent);
    cbor::array(&mut encoded, tokens.len());
    for &token in tokens {
        cbor::uint(&mut encoded, u64::from(token));
    }
    match extra_keys {
        None => cbor::null(&mut encoded),
        Some(keys) => {
            cbor::array(&mut encoded, keys.len());
            for key in keys {
                match key {
                    VllmExtraKey::Lora { name, path } => {
                        cbor::array(&mut encoded, 3);
                        cbor::text(&mut encoded, "lora");
                        cbor::text(&mut encoded, name);
                        cbor::text(&mut encoded, path);
                    }
                    VllmExtraKey::Mm { identifier, offset } => {
                        cbor::array(&mut encoded, 3);
                        cbor::text(&mut encoded, "mm");
                        cbor::text(&mut encoded, identifier);
                        cbor::int(&mut encoded, *offset);
                    }
                    VllmExtraKey::CacheSalt(salt) => {
                        cbor::array(&mut encoded, 2);
                        cbor::text(&mut encoded, "cache_salt");
                        cbor::text(&mut encoded, salt);
                    }
                    VllmExtraKey::PromptEmbeds(digest) => {
                        cbor::array(&mut encoded, 2);
                        cbor::text(&mut encoded, "prompt_embeds");
                        cbor::bytes(&mut encoded, digest);
                    }
                }
            }
        }
    }
    Sha256::digest(&encoded).into()
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
/// with `canonical=True` produces exactly these bytes for those values.
pub(crate) mod cbor {
    fn head(out: &mut Vec<u8>, major: u8, value: u64) {
        let major = major << 5;
        if value < 24 {
            out.push(major | value as u8);
        } else if value <= u64::from(u8::MAX) {
            out.push(major | 24);
            out.push(value as u8);
        } else if value <= u64::from(u16::MAX) {
            out.push(major | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        } else if value <= u64::from(u32::MAX) {
            out.push(major | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        } else {
            out.push(major | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }

    pub(crate) fn uint(out: &mut Vec<u8>, value: u64) {
        head(out, 0, value);
    }

    pub(crate) fn int(out: &mut Vec<u8>, value: i64) {
        if value >= 0 {
            head(out, 0, value as u64);
        } else {
            head(out, 1, !(value as u64));
        }
    }

    pub(crate) fn bytes(out: &mut Vec<u8>, value: &[u8]) {
        head(out, 2, value.len() as u64);
        out.extend_from_slice(value);
    }

    pub(crate) fn text(out: &mut Vec<u8>, value: &str) {
        head(out, 3, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    pub(crate) fn array(out: &mut Vec<u8>, len: usize) {
        head(out, 4, len as u64);
    }

    pub(crate) fn null(out: &mut Vec<u8>) {
        out.push(0xf6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_events::golden::bytes as hex;

    fn digest(hex_text: &str) -> Digest32 {
        hex(hex_text).try_into().expect("32 bytes")
    }

    // SGLang vectors published with the router's hash tests
    // (experimental/sgl-router/src/state/kv_events/hash.rs).
    #[test]
    fn sglang_reproduces_the_published_vectors() {
        let ints = |tokens: &[u32], page| -> Vec<i64> {
            sglang_chain(tokens, page, None)
                .into_iter()
                .map(|(_, int)| int)
                .collect()
        };
        assert_eq!(ints(&[1, 2, 3, 4], 4), vec![-3488128144981237669]);
        assert_eq!(
            ints(&[10, 20, 30, 40, 50, 60, 70, 80], 2),
            vec![
                978178666101069530,
                -895308556211281782,
                -8033692805846017938,
                835415944263129316
            ]
        );
        assert_eq!(
            sglang_event_int(&Sha256::digest(b"").into()),
            -2039914840885289964
        );
        assert!(ints(&[1, 2, 3], 4).is_empty(), "only full pages are hashed");
        assert!(sglang_chain(&[1, 2, 3, 4], 0, None).is_empty());
    }

    // Derived with hashlib from the algorithm above (vectors file in the
    // branch notes): salted and bigram chains.
    #[test]
    fn sglang_salted_and_bigram_chains() {
        let seed = sglang_salt_seed("tenant-a");
        assert_eq!(
            seed,
            digest("f5d0f785efe6042a4e4b0297a4d712917e1763c850835e93df58b373fd47fd2c")
        );
        let salted: Vec<i64> = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, Some(&seed))
            .into_iter()
            .map(|(_, int)| int)
            .collect();
        assert_eq!(salted, vec![3718046898735569995, 3308615664605479373]);
        let plain = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
        assert_eq!(
            plain[0].0,
            digest("cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72")
        );
        assert_eq!(
            plain[1].0,
            digest("4ebfa8a1f3c341517621838c6e1b9aa350307e3f00b3cbd1a07ef740f54396d6")
        );
        // Bigram page (1,2) (2,3) (3,4) (4,5): both words of every pair.
        let bigram = sglang_page(None, &[1, 2, 2, 3, 3, 4, 4, 5]);
        assert_eq!(sglang_event_int(&bigram), -638950109823820341);
    }

    // vLLM vectors: `hash_block_tokens` with `sha256_cbor` from the checkout
    // at 0c16eee3f1, run under cbor2 with PYTHONHASHSEED unset.
    #[test]
    fn vllm_reproduces_the_reference_run() {
        assert_eq!(
            vllm_none_hash(),
            digest("9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0")
        );
        let a = vllm_block(None, &[1, 2, 3, 4], None);
        assert_eq!(
            a,
            digest("58d0879dff3800f65f8c5fd449d73048c0f7699dd238a03b84b146b151d55111")
        );
        assert_eq!(vllm_event_int(&a), -8885242862429187823);
        assert_eq!(vllm_event_int(&a) as u64, 9561501211280363793);
        let b = vllm_block(Some(&a), &[5, 6, 7, 8], None);
        assert_eq!(
            b,
            digest("e1bc29393a2da9fd798f35ff04d215f846aee6df4b803ce3d43b57df56b58fb1")
        );
        assert_eq!(vllm_event_int(&b), -3153830497298837583);
        let chain = vllm_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
        assert_eq!(
            chain,
            vec![(a, vllm_event_int(&a)), (b, vllm_event_int(&b))]
        );

        let lora = VllmExtraKey::Lora {
            name: "adapter".into(),
            path: "/adapters/adapter".into(),
        };
        let c = vllm_block(
            None,
            &[1, 2, 3, 4],
            Some(&[
                lora.clone(),
                VllmExtraKey::Mm {
                    identifier: "mm-abc".into(),
                    offset: 0,
                },
                VllmExtraKey::CacheSalt("salt-1".into()),
                VllmExtraKey::PromptEmbeds((0u8..32).collect()),
            ]),
        );
        assert_eq!(
            c,
            digest("280bce663478b44320e68be593605214c94f7648ac34b4b94274fdcd0f2a8a04")
        );
        assert_eq!(vllm_event_int(&c), 4788731360966248964);
        let d = vllm_block(
            Some(&c),
            &[5, 6, 7, 8],
            Some(&[
                lora,
                VllmExtraKey::Mm {
                    identifier: "mm-abc".into(),
                    offset: -4,
                },
            ]),
        );
        assert_eq!(
            d,
            digest("33e7394e25b7be46c7bf0c090da02e903b4e9d9536b1c18eed4d6e57e7a3658a")
        );
        assert_eq!(vllm_event_int(&d), -1347299389686454902);

        let e = vllm_block(
            None,
            &[1, 2, 3, 4],
            Some(&[
                VllmExtraKey::Mm {
                    identifier: "mm-abc".into(),
                    offset: 0,
                },
                VllmExtraKey::CacheSalt("salt-1".into()),
                VllmExtraKey::PromptEmbeds((0u8..32).collect()),
            ]),
        );
        assert_eq!(e, digest(VECTOR_E));
        let f = vllm_block(
            Some(&e),
            &[5, 6, 7, 8],
            Some(&[VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: -4,
            }]),
        );
        assert_eq!(f, digest(VECTOR_F));
        let g = vllm_block(Some(&f), &[9, 10, 11, 12], None);
        assert_eq!(g, digest(VECTOR_G));
    }

    // The CBOR bytes `cbor2.dumps(value, canonical=True)` produced for the
    // same inputs.
    #[test]
    fn cbor_matches_cbor2_canonical_bytes() {
        let mut seed = Vec::new();
        cbor::text(&mut seed, VLLM_NONE_HASH_SEED);
        assert_eq!(seed, hex("6e766c6c6d2d6e6f6e652d68617368"));

        let mut ints = Vec::new();
        cbor::array(&mut ints, 14);
        for value in [
            0i64,
            23,
            24,
            255,
            256,
            65535,
            65536,
            4_294_967_295,
            4_294_967_296,
        ] {
            cbor::int(&mut ints, value);
        }
        for value in [-1i64, -24, -25, -256, -257] {
            cbor::int(&mut ints, value);
        }
        assert_eq!(ints, hex(CBOR_INTS));

        // `[NONE, [1, 2, 3, 4], null]` and block D's input.
        let mut a_input = Vec::new();
        cbor::array(&mut a_input, 3);
        cbor::bytes(&mut a_input, &vllm_none_hash());
        cbor::array(&mut a_input, 4);
        for token in 1..=4u64 {
            cbor::uint(&mut a_input, token);
        }
        cbor::null(&mut a_input);
        assert_eq!(
            a_input,
            hex("8358209bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef08401020304f6")
        );
    }

    #[test]
    fn setting_names_parse() {
        assert_eq!(EngineHash::parse("sglang"), Some(EngineHash::Sglang));
        assert_eq!(EngineHash::parse(" SGLang "), Some(EngineHash::Sglang));
        assert_eq!(
            EngineHash::parse("vllm_sha256_cbor"),
            Some(EngineHash::VllmSha256Cbor)
        );
        assert_eq!(
            EngineHash::parse("vllm-sha256-cbor"),
            Some(EngineHash::VllmSha256Cbor)
        );
        assert_eq!(EngineHash::parse("vllm"), None);
        assert_eq!(EngineHash::parse(""), None);
        assert_eq!(EngineHash::VllmSha256Cbor.as_str(), "vllm-sha256-cbor");
    }

    const VECTOR_E: &str = "0d7b0d8344b79ad5e183b117cacc04aeb415bfdfa968fd28f611bc6971f4abba";
    const VECTOR_F: &str = "b81a4631bec909b72bcd0081cc2d7c87bcce5652c48318f9615e4aa3370c1d69";
    const VECTOR_G: &str = "024cf74ecdb93c6049fe59a66321192aab04701a75ac3e86619395104df937c4";
    const CBOR_INTS: &str =
        "8e0017181818ff19010019ffff1a000100001affffffff1b00000001000000002037381838ff390100";
}

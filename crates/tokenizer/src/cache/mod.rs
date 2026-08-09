//! Tokenizer Caching Layer
//!
//! Provides a caching wrapper around any tokenizer implementation to speed up
//! repeated tokenization of the same strings (e.g., system prompts).
//!
//! # Architecture
//! - **L0 Cache**: Whole-string exact match (90% of wins)
//! - **L1 Cache**: Prefix matching at fixed boundaries (future work)
//!
//! # Usage
//! ```ignore
//! let tokenizer = Arc::new(HuggingFaceTokenizer::from_file("tokenizer.json")?);
//! let cached = Arc::new(CachedTokenizer::new(tokenizer, CacheConfig::default()));
//! let encoding = cached.encode("Hello world")?;
//! ```

mod fingerprint;
mod l0;
mod l1;

use std::sync::Arc;

use anyhow::Result;
pub use fingerprint::TokenizerFingerprint;
pub use l0::{CacheStats, L0Cache};
use l1::PrefixLookup;
pub use l1::{L1Cache, L1CacheStats};
use rayon::prelude::*;

use crate::{
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
    },
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
};

/// Configuration for the tokenizer cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable L0 (whole-string) cache
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache
    pub l0_max_entries: usize,
    /// Enable L1 (prefix) cache
    pub enable_l1: bool,
    /// Maximum memory for L1 cache in bytes
    pub l1_max_memory: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000, // ~22MB memory for typical prompts
            enable_l1: false,       // Opt-in for now
            l1_max_memory: 50 * 1024 * 1024, // 50MB
        }
    }
}

/// A caching wrapper around any tokenizer
pub struct CachedTokenizer {
    /// The underlying tokenizer
    inner: Arc<dyn Tokenizer>,
    /// L0 cache (whole-string exact match)
    l0: Option<L0Cache>,
    /// L1 cache (prefix matching at fixed boundaries)
    l1: Option<L1Cache>,
    /// Fingerprint for cache invalidation
    fingerprint: TokenizerFingerprint,
    /// Cached special token strings (extracted once at construction)
    special_token_strings: Vec<String>,
}

impl CachedTokenizer {
    /// Create a new cached tokenizer
    pub fn new(inner: Arc<dyn Tokenizer>, config: CacheConfig) -> Self {
        let fingerprint = TokenizerFingerprint::from_tokenizer(inner.as_ref());

        let l0 = if config.enable_l0 {
            Some(L0Cache::new(config.l0_max_entries))
        } else {
            None
        };

        let l1 = if config.enable_l1 {
            Some(L1Cache::new(config.l1_max_memory))
        } else {
            None
        };

        // Extract special tokens once at construction time
        let special_token_strings = Self::extract_special_token_strings(&inner);

        Self {
            inner,
            l0,
            l1,
            fingerprint,
            special_token_strings,
        }
    }

    /// Extract all special token strings from the tokenizer (called once at construction)
    fn extract_special_token_strings(tokenizer: &Arc<dyn Tokenizer>) -> Vec<String> {
        let special_tokens = tokenizer.get_special_tokens();
        let mut tokens = Vec::new();

        if let Some(ref token) = special_tokens.bos_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.eos_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.unk_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.sep_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.pad_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.cls_token {
            tokens.push(token.clone());
        }
        if let Some(ref token) = special_tokens.mask_token {
            tokens.push(token.clone());
        }

        tokens.extend(special_tokens.additional_special_tokens.iter().cloned());
        tokens
    }

    /// Get L0 cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.l0.as_ref().map(|cache| cache.stats())
    }

    /// Get L1 cache statistics
    pub fn l1_cache_stats(&self) -> Option<L1CacheStats> {
        self.l1.as_ref().map(|cache| cache.stats())
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Some(l0) = &self.l0 {
            l0.clear();
        }
        if let Some(l1) = &self.l1 {
            l1.clear();
        }
    }

    /// Get the fingerprint of the underlying tokenizer
    pub fn fingerprint(&self) -> &TokenizerFingerprint {
        &self.fingerprint
    }

    /// Get a reference to the inner (wrapped) tokenizer
    pub fn inner(&self) -> &Arc<dyn Tokenizer> {
        &self.inner
    }
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        // L0 cache lookup (exact match, keyed on input + add_special_tokens)
        if let Some(l0) = &self.l0 {
            if let Some(cached) = l0.get(input, add_special_tokens) {
                return Ok((*cached).clone());
            }
        }

        // L1 path (prefix match at special token boundaries): a hit splices the
        // shared cached prefix with a fresh suffix encode; a miss tokenizes the
        // input once, seeding every boundary entry along the way.
        if let Some(l1) = &self.l1 {
            let tokens: Vec<&str> = self
                .special_token_strings
                .iter()
                .map(|s| s.as_str())
                .collect();

            let encoding = match l1.lookup_with_seeds(input, &tokens, add_special_tokens) {
                PrefixLookup::Hit(prefix_tokens, prefix_len) if prefix_len < input.len() => {
                    let suffix = &input[prefix_len..];
                    // The cached prefix already carries any leading special tokens,
                    // so the suffix must never re-add them (it is never segment 0).
                    let suffix_encoding = self.inner.encode(suffix, false)?;
                    let suffix_tokens = suffix_encoding.token_ids();

                    // Splice with exact capacity: one allocation and a single copy
                    // of the shared prefix.
                    let mut merged_tokens =
                        Vec::with_capacity(prefix_tokens.len() + suffix_tokens.len());
                    merged_tokens.extend_from_slice(&prefix_tokens);
                    merged_tokens.extend_from_slice(suffix_tokens);
                    Encoding::Plain(merged_tokens)
                }
                // Defensive: boundaries always exclude input.len(), so a full-input
                // match cannot occur; if it ever did, the entry is keyed on the whole
                // input and the cached tokens ARE the full encoding.
                PrefixLookup::Hit(prefix_tokens, _) => Encoding::Plain(prefix_tokens.to_vec()),
                // No special token boundaries — nothing cacheable; single plain encode
                // (preserves the inner tokenizer's native encoding variant).
                PrefixLookup::Miss(seeds) if seeds.is_empty() => {
                    self.inner.encode(input, add_special_tokens)?
                }
                PrefixLookup::Miss(seeds) => {
                    match l1.populate_with_seeds(
                        input,
                        &seeds,
                        self.inner.as_ref(),
                        add_special_tokens,
                    ) {
                        Ok(encoding) => encoding,
                        // Seeding failed mid-segment (nothing was inserted) — fall
                        // back to the plain uncached encode, matching the previous
                        // behavior where boundary-insert errors were swallowed.
                        Err(_) => self.inner.encode(input, add_special_tokens)?,
                    }
                }
            };

            if let Some(l0) = &self.l0 {
                l0.insert(input.to_string(), add_special_tokens, encoding.clone());
            }

            return Ok(encoding);
        }

        // Full tokenization (no L1 configured), cached in L0 only
        let encoding = self.inner.encode(input, add_special_tokens)?;

        if let Some(l0) = &self.l0 {
            l0.insert(input.to_string(), add_special_tokens, encoding.clone());
        }

        Ok(encoding)
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        // Process each input in parallel, leveraging thread-safe caches
        // This maintains the parallelism from the underlying HuggingFaceTokenizer
        inputs
            .par_iter()
            .map(|&input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // Decoding is not cached (it's fast enough and rarely repeated)
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.inner.apply_chat_template(messages, params)
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.inner.chat_template_content_format()
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        self.inner.thinking_toggle()
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.inner.thinking_key_name()
    }

    fn template_reasoning_effort_enables_thinking(&self) -> bool {
        self.inner.template_reasoning_effort_enables_thinking()
    }

    fn think_in_prefill(&self) -> bool {
        self.inner.think_in_prefill()
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        self.inner.eos_token_ids()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::{mock::MockTokenizer, *};

    /// Tokenizer that prepends a sentinel BOS when `add_special_tokens` is set,
    /// so duplicated specials across a prefix/suffix merge are observable.
    struct BosTokenizer {
        special_tokens: SpecialTokens,
    }

    const BOS_ID: TokenIdType = 99;

    impl BosTokenizer {
        fn new() -> Self {
            Self {
                special_tokens: SpecialTokens {
                    bos_token: Some("<bos>".to_string()),
                    additional_special_tokens: vec![
                        "<|im_start|>".to_string(),
                        "<|im_end|>".to_string(),
                    ],
                    ..Default::default()
                },
            }
        }
    }

    impl Encoder for BosTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = Vec::new();
            if add_special_tokens {
                ids.push(BOS_ID);
            }
            ids.extend(input.bytes().map(TokenIdType::from));
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for BosTokenizer {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for BosTokenizer {
        fn vocab_size(&self) -> usize {
            256
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }
        fn token_to_id(&self, _token: &str) -> Option<TokenIdType> {
            None
        }
        fn id_to_token(&self, _id: TokenIdType) -> Option<String> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Wraps [`BosTokenizer`], counting `encode` invocations and the total bytes
    /// of text tokenized, so tests can assert how much tokenizer work each cache
    /// path actually performs.
    struct CountingTokenizer {
        inner: BosTokenizer,
        encode_calls: AtomicUsize,
        bytes_encoded: AtomicUsize,
    }

    impl CountingTokenizer {
        fn new() -> Self {
            Self {
                inner: BosTokenizer::new(),
                encode_calls: AtomicUsize::new(0),
                bytes_encoded: AtomicUsize::new(0),
            }
        }

        fn encode_calls(&self) -> usize {
            self.encode_calls.load(Ordering::Relaxed)
        }

        fn bytes_encoded(&self) -> usize {
            self.bytes_encoded.load(Ordering::Relaxed)
        }

        fn reset(&self) {
            self.encode_calls.store(0, Ordering::Relaxed);
            self.bytes_encoded.store(0, Ordering::Relaxed);
        }
    }

    impl Encoder for CountingTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            self.encode_calls.fetch_add(1, Ordering::Relaxed);
            self.bytes_encoded.fetch_add(input.len(), Ordering::Relaxed);
            self.inner.encode(input, add_special_tokens)
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for CountingTokenizer {
        fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl traits::Tokenizer for CountingTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }
        fn token_to_id(&self, _token: &str) -> Option<TokenIdType> {
            None
        }
        fn id_to_token(&self, _id: TokenIdType) -> Option<String> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// L1-only cache config used by the counting/equivalence tests (L0 disabled so
    /// every call exercises the L1 hit/miss machinery).
    fn l1_only_config() -> CacheConfig {
        CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
            enable_l1: true,
            l1_max_memory: 1024 * 1024,
        }
    }

    #[test]
    fn test_l1_miss_tokenizes_input_exactly_once() {
        let counting = Arc::new(CountingTokenizer::new());
        let cached = CachedTokenizer::new(counting.clone(), l1_only_config());

        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nquery<|im_end|>";

        // Cold miss: the fused path must tokenize each input byte exactly once.
        // (Previously: one full encode for the result + a re-encode of every
        // boundary prefix while seeding — roughly 2x the input.)
        let first = cached.encode(input, true).unwrap();
        assert_eq!(
            counting.bytes_encoded(),
            input.len(),
            "miss path must tokenize the input exactly once"
        );

        // Warm hit: only the suffix past the deepest cached boundary is tokenized.
        // The deepest boundary is the end of the last special token occurrence that
        // still leaves a suffix (the final <|im_end|> ends at input.len() and is
        // never a boundary).
        let deepest_boundary = input.rfind("<|im_start|>").unwrap() + "<|im_start|>".len();
        counting.reset();
        let second = cached.encode(input, true).unwrap();
        assert_eq!(
            counting.bytes_encoded(),
            input.len() - deepest_boundary,
            "hit path must tokenize only the suffix"
        );
        assert_eq!(
            counting.encode_calls(),
            1,
            "hit path performs a single suffix encode"
        );

        // Both paths return the same stream as a fresh uncached encode.
        let fresh = BosTokenizer::new().encode(input, true).unwrap();
        assert_eq!(first.token_ids(), fresh.token_ids());
        assert_eq!(second.token_ids(), fresh.token_ids());
    }

    #[test]
    fn test_l1_multi_turn_growth_matches_uncached() {
        // Append-only conversation: every turn's cached encode (miss on turn 0,
        // prefix hits afterwards) must equal a fresh uncached encode, under both
        // add_special_tokens keys.
        let mut conversation = String::from("<|im_start|>system\nYou are helpful.<|im_end|>");
        let mut turns = Vec::new();
        for i in 0..4 {
            conversation.push_str(&format!(
                "<|im_start|>user\nquestion {i}<|im_end|><|im_start|>assistant\nanswer {i}<|im_end|>"
            ));
            turns.push(conversation.clone());
        }

        for add_special_tokens in [false, true] {
            let cached = CachedTokenizer::new(Arc::new(BosTokenizer::new()), l1_only_config());
            let plain = BosTokenizer::new();

            for (i, turn) in turns.iter().enumerate() {
                let got = cached.encode(turn, add_special_tokens).unwrap();
                let want = plain.encode(turn, add_special_tokens).unwrap();
                assert_eq!(
                    got.token_ids(),
                    want.token_ids(),
                    "turn {i} (add_special_tokens={add_special_tokens}): cached encode must equal uncached"
                );
            }

            // Growth actually exercised the hit path (turn 0 misses, later turns hit).
            let stats = cached.l1_cache_stats().expect("L1 enabled");
            assert!(stats.hits >= 1, "expected L1 prefix hits across turns");
        }
    }

    #[test]
    fn test_l1_hit_does_not_duplicate_special_tokens() {
        let tokenizer = Arc::new(BosTokenizer::new());
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
            enable_l1: true,
            l1_max_memory: 1024 * 1024,
        };
        let cached = CachedTokenizer::new(tokenizer, config);

        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nquery<|im_end|>";

        // First call warms the L1 cache at special-token boundaries.
        let first = cached.encode(input, true).unwrap();
        // Second call hits the warm L1 cache and merges prefix + suffix.
        let second = cached.encode(input, true).unwrap();

        // The merged result must match a fresh, uncached encode: a single BOS at
        // the start, none duplicated mid-sequence and none dropped.
        let fresh = CachedTokenizer::new(
            Arc::new(BosTokenizer::new()),
            CacheConfig {
                enable_l0: false,
                l0_max_entries: 0,
                enable_l1: false,
                l1_max_memory: 0,
            },
        )
        .encode(input, true)
        .unwrap();

        assert_eq!(first.token_ids(), fresh.token_ids());
        assert_eq!(second.token_ids(), fresh.token_ids());
        assert_eq!(
            second.token_ids().iter().filter(|&&t| t == BOS_ID).count(),
            1,
            "exactly one BOS expected, got tokens: {:?}",
            second.token_ids()
        );
    }

    #[test]
    fn test_cache_hit() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let input = "Hello world";

        // First call - miss
        let result1 = cached.encode(input, false).unwrap();

        // Second call - hit
        let result2 = cached.encode(input, false).unwrap();

        // Results should be identical
        assert_eq!(result1.token_ids(), result2.token_ids());

        // Check cache stats
        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_disabled() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
            enable_l1: false,
            l1_max_memory: 0,
        };
        let cached = CachedTokenizer::new(tokenizer, config);

        let input = "Hello world";

        // Both calls should work even without cache
        let result1 = cached.encode(input, false).unwrap();
        let result2 = cached.encode(input, false).unwrap();

        assert_eq!(result1.token_ids(), result2.token_ids());

        // No cache stats available
        assert!(cached.cache_stats().is_none());
    }

    #[test]
    fn test_encode_batch() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let inputs = vec!["Hello", "world", "Hello"]; // "Hello" repeated

        let results = cached.encode_batch(&inputs, false).unwrap();

        assert_eq!(results.len(), 3);

        // With parallel execution, duplicate inputs may be processed simultaneously
        // and both see cache misses. Verify results are correct instead.
        assert_eq!(results[0].token_ids(), results[2].token_ids()); // Both "Hello" should match

        // After batch processing, cache should be populated
        // Subsequent calls should hit the cache
        let _ = cached.encode("Hello", false).unwrap();
        let stats = cached.cache_stats().unwrap();

        // Should have at least 1 hit from the call above (cache was populated by batch)
        assert!(
            stats.hits >= 1,
            "Expected at least 1 cache hit after batch processing"
        );
    }

    #[test]
    fn test_decoder_passthrough() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let tokens = vec![1, 2, 3];
        let decoded = cached.decode(&tokens, false).unwrap();

        // Should just pass through to inner tokenizer
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_trait_methods() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer.clone(), CacheConfig::default());

        // Should pass through to inner tokenizer
        assert_eq!(cached.vocab_size(), tokenizer.vocab_size());
        assert!(cached.token_to_id("Hello").is_some());
        assert!(cached.id_to_token(1).is_some());
    }
}

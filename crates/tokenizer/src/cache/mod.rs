//! Tokenizer Caching Layer
//!
//! Provides a caching wrapper around any tokenizer implementation to speed up
//! repeated tokenization of the same strings (e.g., system prompts).
//!
//! # Architecture
//! - **L0 Cache**: Whole-string exact match (90% of wins)
//! - **L1 Cache**: Prefix matching at special-token boundaries (opt-in)
//!
//! # Usage
//! ```ignore
//! let tokenizer = Arc::new(HuggingFaceTokenizer::from_file("tokenizer.json")?);
//! let cached = Arc::new(CachedTokenizer::new(tokenizer, CacheConfig::default()));
//! let encoding = cached.encode("Hello world")?;
//! ```

mod activity;
mod fingerprint;
mod l0;
mod l1;

use std::sync::Arc;

pub use activity::{cache_activity_stats, CacheActivityStats};
use anyhow::Result;
pub use fingerprint::TokenizerFingerprint;
pub use l0::{CacheStats, L0Cache};
pub use l1::{L1Cache, L1CacheStats};
use l1::{PrefixLookup, SpecialTokenBoundaries, SpecialTokenFrame};
use rayon::prelude::*;

use crate::{
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
    },
    traits::{
        ChatTemplateOutput, DecodeStep, Decoder, Encoder, Encoding, IncrementalDecoder,
        SpecialTokens, TokenIdType, Tokenizer,
    },
};

/// Configuration for the tokenizer cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable L0 (whole-string) cache
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache
    pub l0_max_entries: usize,
    /// Byte budget of the L0 cache (texts, ids and per-entry overhead); an
    /// input whose entry would exceed a quarter of it is not cached
    pub l0_max_memory: usize,
    /// Enable L1 (prefix) cache
    pub enable_l1: bool,
    /// Maximum memory for L1 cache in bytes
    pub l1_max_memory: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000,
            l0_max_memory: l0::DEFAULT_MAX_BYTES,
            enable_l1: false,                // Opt-in for now
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
    /// The special tokens L1 splits at, as one matcher built at construction.
    special_tokens: SpecialTokenBoundaries,
    /// What `add_special_tokens` adds around a text's ids, applied once around
    /// a spliced result; `None` with L1 on: such requests are not cached.
    frame: Option<SpecialTokenFrame>,
}

impl CachedTokenizer {
    /// Create a new cached tokenizer
    pub fn new(inner: Arc<dyn Tokenizer>, config: CacheConfig) -> Self {
        let fingerprint = TokenizerFingerprint::from_tokenizer(inner.as_ref());

        let l0 = if config.enable_l0 {
            Some(L0Cache::with_limits(
                config.l0_max_entries,
                config.l0_max_memory,
            ))
        } else {
            None
        };

        let l1 = if config.enable_l1 {
            Some(L1Cache::new(config.l1_max_memory))
        } else {
            None
        };

        // The special tokens the cache splits at: those matched on the raw
        // text (not after normalization) whose splice the tokenizer
        // reproduces, found once at construction; none without L1.
        let special_tokens = if config.enable_l1 {
            let after_normalization = inner.added_tokens_after_normalization();
            let mut candidates = Self::extract_special_token_strings(&inner);
            candidates.retain(|token| !after_normalization.contains(token));
            SpecialTokenBoundaries::for_encoder(
                &candidates,
                &inner.added_token_strings(),
                inner.as_ref(),
            )
        } else {
            SpecialTokenBoundaries::new::<&str>(&[])
        };
        // What `add_special_tokens` adds around a text's ids, read once with L1
        // on; when it cannot be, requests with the flag are not cached.
        let frame = if config.enable_l1 {
            let frame = SpecialTokenFrame::probe(inner.as_ref());
            if frame.is_none() {
                tracing::debug!(
                    "the prefix cache does not serve requests with add_special_tokens: what the flag adds is not one frame on every probe"
                );
            }
            frame
        } else {
            None
        };

        Self {
            inner,
            l0,
            l1,
            fingerprint,
            special_tokens,
            frame,
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
        // shared cached prefix with the suffix encoded after the boundary's
        // special token, as it is in the whole text; a miss tokenizes the input
        // once, seeding every boundary entry along the way. Where the splice
        // cannot be shown exact for this input (the boundary's token does not
        // come back as itself in front of the suffix), the plain encode serves
        // the request, counted as a fallback in the L1 stats; a prefix cached
        // before that point stays, its own segments having come back exact.
        // Segments are encoded without `add_special_tokens`; what the flag adds
        // (a BOS in front, an EOS after) goes around the whole result once.
        if let Some(l1) = &self.l1 {
            let inner = self.inner.as_ref();
            let plain = || inner.encode(input, add_special_tokens);
            let encoding = if add_special_tokens && self.frame.is_none() {
                // What the flag adds could not be read off the tokenizer: the
                // plain encode, uncached in L1.
                plain()?
            } else {
                let fallback = || {
                    l1.note_fallback();
                    plain()
                };
                // The frame the flag adds goes around the spliced ids: its leading
                // ids first in the result buffer (never in a cached entry), its
                // trailing ids last.
                let frame = self.frame.as_ref().filter(|_| add_special_tokens);
                let head = frame.map_or(0, |frame| frame.prepended().len());
                // The result buffer, sized once: the frame, the shared prefix and
                // at most one id per byte of the rest of the input.
                let buffer = |prefix: &[TokenIdType], from: usize| {
                    let tail = frame.map_or(0, |frame| frame.appended().len());
                    let mut tokens =
                        Vec::with_capacity(head + prefix.len() + (input.len() - from) + tail);
                    if let Some(frame) = frame {
                        tokens.extend_from_slice(frame.prepended());
                    }
                    tokens.extend_from_slice(prefix);
                    tokens
                };
                let finish = |mut tokens: Vec<TokenIdType>| {
                    if let Some(frame) = frame {
                        tokens.extend_from_slice(frame.appended());
                    }
                    Encoding::Plain(tokens)
                };
                match l1.lookup_with_seeds(input, &self.special_tokens, add_special_tokens) {
                    PrefixLookup::Hit(prefix_tokens, hit, Some(deepest)) => {
                        // Also cache this input's deepest boundary: a later input that extends
                        // it (the next turn of the same conversation) then matches there, not at
                        // the shorter prefix this one shares with other conversations.
                        let mut tokens = buffer(&prefix_tokens, hit.end);
                        let spliced = self
                            .special_tokens
                            .encode_segment_into(&hit, input, deepest.end, inner, &mut tokens)
                            .and_then(|()| {
                                l1.insert_prefix(deepest.digest, deepest.end, &tokens[head..]);
                                self.special_tokens.encode_segment_into(
                                    &deepest,
                                    input,
                                    input.len(),
                                    inner,
                                    &mut tokens,
                                )
                            });
                        match spliced {
                            Ok(()) => finish(tokens),
                            Err(_) => fallback()?,
                        }
                    }
                    PrefixLookup::Hit(prefix_tokens, hit, None) if hit.end < input.len() => {
                        // The suffix is encoded after the boundary's token (never as segment 0)
                        // straight into the result: one allocation, one copy of the prefix.
                        let mut tokens = buffer(&prefix_tokens, hit.end);
                        match self.special_tokens.encode_segment_into(
                            &hit,
                            input,
                            input.len(),
                            inner,
                            &mut tokens,
                        ) {
                            Ok(()) => finish(tokens),
                            Err(_) => fallback()?,
                        }
                    }
                    // Defensive: boundaries always exclude input.len(), so a full-input
                    // match cannot occur; if it ever did, the entry is keyed on the whole
                    // input and the cached tokens ARE the text's full encoding.
                    PrefixLookup::Hit(prefix_tokens, _, _) => {
                        finish(buffer(&prefix_tokens, input.len()))
                    }
                    // No special token boundaries — nothing cacheable; single plain encode
                    // (preserves the inner tokenizer's native encoding variant).
                    PrefixLookup::Miss(seeds) if seeds.is_empty() => plain()?,
                    PrefixLookup::Miss(seeds) => {
                        match l1.populate_with_seeds(
                            input,
                            &seeds,
                            inner,
                            &self.special_tokens,
                            add_special_tokens,
                            frame,
                        ) {
                            Ok(encoding) => encoding,
                            // A segment's splice is not exact for this input (nothing is
                            // stored before every segment came back exact): the plain
                            // uncached encode, as before.
                            Err(_) => fallback()?,
                        }
                    }
                }
            };

            if let Some(l0) = &self.l0 {
                l0.insert(
                    input.to_string(),
                    add_special_tokens,
                    Encoding::Plain(encoding.token_ids().to_vec()),
                );
            }

            return Ok(encoding);
        }

        // Full tokenization (no L1 configured), cached in L0 only
        let encoding = self.inner.encode(input, add_special_tokens)?;

        if let Some(l0) = &self.l0 {
            l0.insert(
                input.to_string(),
                add_special_tokens,
                Encoding::Plain(encoding.token_ids().to_vec()),
            );
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
    fn uses_byte_fallback(&self) -> bool {
        self.inner.uses_byte_fallback()
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // Decoding is not cached (it's fast enough and rarely repeated)
        self.inner.decode(token_ids, skip_special_tokens)
    }

    // Incremental decoding is the inner tokenizer's business too; without these
    // forwards a cached tokenizer would fall back to the generic double decode.
    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        self.inner
            .decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode_step_eager(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        self.inner
            .decode_step_eager(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }

    fn decode_step_reporting(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
        eager: bool,
    ) -> Result<DecodeStep> {
        self.inner.decode_step_reporting(
            token_id,
            ids,
            prefix,
            prefix_index,
            skip_special_tokens,
            eager,
        )
    }

    fn finish_decode(
        &self,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<DecodeStep> {
        self.inner
            .finish_decode(ids, prefix, prefix_index, skip_special_tokens)
    }

    fn incremental_decoder(
        &self,
        skip_special_tokens: bool,
    ) -> Option<Box<dyn IncrementalDecoder>> {
        self.inner.incremental_decoder(skip_special_tokens)
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

    fn apply_chat_template_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        // Forward rather than inherit the default: the default would render
        // through the inner tokenizer's flat template and report `FromText`,
        // silently dropping a deferred encode. A flat rendering still reaches
        // the L0/L1 caches because the caller encodes its text through
        // `self.encode`; a deferred encode never does, since two renderings
        // can share one text with different ids.
        self.inner
            .apply_chat_template_with_encoding(messages, params, assistant_prefix)
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

    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        self.inner.native_reasoning_effort_values()
    }

    fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        self.inner.native_reasoning_effort_off_values()
    }

    fn think_in_prefill(&self) -> bool {
        self.inner.think_in_prefill()
    }

    fn prefill_opens_think_block(&self) -> bool {
        self.inner.prefill_opens_think_block()
    }

    fn renderer_capabilities(&self) -> crate::traits::RendererCapabilities {
        self.inner.renderer_capabilities()
    }

    fn native_thinking_mode(
        &self,
        template_kwargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> Option<bool> {
        self.inner.native_thinking_mode(template_kwargs, thinking)
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        self.inner.eos_token_ids()
    }

    fn added_token_strings(&self) -> Vec<String> {
        self.inner.added_token_strings()
    }

    fn added_tokens_after_normalization(&self) -> Vec<String> {
        self.inner.added_tokens_after_normalization()
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

    /// The special tokens of the test encoders: atomic (one id each from
    /// 200), extracted leftmost-longest as a BPE tokenizer's added vocabulary
    /// does it; one id per byte otherwise.
    const TEST_SPECIALS: [&str; 3] = ["<bos>", "<|im_start|>", "<|im_end|>"];

    fn atomic_ids(input: &str, ids: &mut Vec<TokenIdType>) {
        let matcher = aho_corasick::AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(TEST_SPECIALS)
            .unwrap();
        let mut last = 0;
        for found in matcher.find_iter(input) {
            ids.extend(input[last..found.start()].bytes().map(TokenIdType::from));
            ids.push(200 + found.pattern().as_u32());
            last = found.end();
        }
        ids.extend(input[last..].bytes().map(TokenIdType::from));
    }

    /// The bytes of the special tokens an L1 encode re-reads: each boundary's
    /// token is encoded in front of the segment after it.
    fn boundary_token_bytes(input: &str) -> usize {
        let matcher = aho_corasick::AhoCorasickBuilder::new()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(TEST_SPECIALS)
            .unwrap();
        matcher
            .find_iter(input)
            .filter(|found| found.end() < input.len())
            .map(|found| found.len())
            .sum()
    }

    impl Encoder for BosTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = Vec::new();
            if add_special_tokens {
                ids.push(BOS_ID);
            }
            atomic_ids(input, &mut ids);
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
            l0_max_memory: usize::MAX,
            enable_l1: true,
            l1_max_memory: 1024 * 1024,
        }
    }

    #[test]
    fn test_l1_miss_tokenizes_input_exactly_once() {
        let counting = Arc::new(CountingTokenizer::new());
        let cached = CachedTokenizer::new(counting.clone(), l1_only_config());
        // The construction probed the special tokens; the request path starts here.
        counting.reset();

        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nquery<|im_end|>";

        // Cold miss: the fused path tokenizes each input byte exactly once,
        // plus each boundary's special token in front of the segment after it.
        // (Previously: one full encode for the result + a re-encode of every
        // boundary prefix while seeding — roughly 2x the input.)
        let first = cached.encode(input, true).unwrap();
        assert_eq!(
            counting.bytes_encoded(),
            input.len() + boundary_token_bytes(input),
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
            input.len() - deepest_boundary + "<|im_start|>".len(),
            "hit path must tokenize only the suffix, after the boundary's token"
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
    fn test_l1_hit_caches_the_deepest_boundary_for_the_next_turn() {
        // Two conversations share a system prompt. The second one's first turn hits the
        // first conversation's system-prompt boundary; its next turn must then match its
        // own previous turn, tokenizing only the bytes past that turn's deepest boundary.
        let counting = Arc::new(CountingTokenizer::new());
        let cached = CachedTokenizer::new(counting.clone(), l1_only_config());
        let system = "<|im_start|>system\nYou are helpful.<|im_end|>";
        let first = format!(
            "{system}<|im_start|>user\nfirst conversation<|im_end|><|im_start|>assistant\n"
        );
        cached.encode(&first, true).unwrap();

        let turn_one = format!(
            "{system}<|im_start|>user\nsecond conversation<|im_end|><|im_start|>assistant\n"
        );
        cached.encode(&turn_one, true).unwrap();
        let deepest = turn_one.rfind("<|im_start|>").unwrap() + "<|im_start|>".len();

        let turn_two = format!(
            "{turn_one}answer<|im_end|><|im_start|>user\nmore<|im_end|><|im_start|>assistant\n"
        );
        counting.reset();
        let encoded = cached.encode(&turn_two, true).unwrap();
        // The text past the previous turn's deepest boundary, encoded in two
        // parts (up to this turn's deepest boundary, then its tail), each after
        // the `<|im_start|>` that ends its boundary.
        assert_eq!(
            counting.bytes_encoded(),
            turn_two.len() - deepest + 2 * "<|im_start|>".len(),
            "the next turn must match its own previous turn, not the shared prefix"
        );
        let fresh = BosTokenizer::new().encode(&turn_two, true).unwrap();
        assert_eq!(encoded.token_ids(), fresh.token_ids());
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
            l0_max_memory: usize::MAX,
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
                l0_max_memory: usize::MAX,
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

    /// Tokenizer whose special tokens are extracted leftmost-longest, as a
    /// BPE tokenizer's added vocabulary does it: one id per special token, one
    /// per byte otherwise. Its short special token is a prefix of its long one.
    struct NestedSpecialsTokenizer {
        special_tokens: SpecialTokens,
        matcher: aho_corasick::AhoCorasick,
    }

    const SHORT_SPECIAL: &str = "<|a|>";
    const LONG_SPECIAL: &str = "<|a|>b|>";

    impl NestedSpecialsTokenizer {
        fn new() -> Self {
            let specials = [SHORT_SPECIAL, LONG_SPECIAL];
            Self {
                special_tokens: SpecialTokens {
                    additional_special_tokens: specials.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                },
                matcher: aho_corasick::AhoCorasickBuilder::new()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(specials)
                    .unwrap(),
            }
        }
    }

    impl Encoder for NestedSpecialsTokenizer {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids = Vec::new();
            let mut last = 0;
            for found in self.matcher.find_iter(input) {
                ids.extend(input[last..found.start()].bytes().map(TokenIdType::from));
                ids.push(1000 + found.pattern().as_u32());
                last = found.end();
            }
            ids.extend(input[last..].bytes().map(TokenIdType::from));
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for NestedSpecialsTokenizer {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for NestedSpecialsTokenizer {
        fn vocab_size(&self) -> usize {
            1002
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

    /// Tokenizer that extracts its added tokens in two passes, as the
    /// `tokenizers` crate does: the raw-pass special `<|s|>` first, then on
    /// the pieces left over the normalized `<|n|>`, which the configuration
    /// names as the EOS, and the longer normalized `a<|n|>b` (leftmost-longest
    /// among them).
    struct TwoPassTokenizer {
        special_tokens: SpecialTokens,
        second_pass: aho_corasick::AhoCorasick,
    }

    impl TwoPassTokenizer {
        fn new() -> Self {
            Self {
                special_tokens: SpecialTokens {
                    additional_special_tokens: vec!["<|s|>".to_owned()],
                    eos_token: Some("<|n|>".to_owned()),
                    ..Default::default()
                },
                second_pass: aho_corasick::AhoCorasickBuilder::new()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(["<|n|>", "a<|n|>b"])
                    .unwrap(),
            }
        }

        fn second_pass(&self, piece: &str, ids: &mut Vec<TokenIdType>) {
            let mut last = 0;
            for found in self.second_pass.find_iter(piece) {
                ids.extend(piece[last..found.start()].bytes().map(TokenIdType::from));
                ids.push(2000 + found.pattern().as_u32());
                last = found.end();
            }
            ids.extend(piece[last..].bytes().map(TokenIdType::from));
        }
    }

    impl Encoder for TwoPassTokenizer {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids = Vec::new();
            let mut last = 0;
            for (start, _) in input.match_indices("<|s|>") {
                self.second_pass(&input[last..start], &mut ids);
                ids.push(1000);
                last = start + "<|s|>".len();
            }
            self.second_pass(&input[last..], &mut ids);
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for TwoPassTokenizer {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for TwoPassTokenizer {
        fn vocab_size(&self) -> usize {
            2002
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
        fn added_tokens_after_normalization(&self) -> Vec<String> {
            vec!["<|n|>".to_owned(), "a<|n|>b".to_owned()]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn a_token_matched_after_normalization_is_no_boundary_whatever_the_configuration_calls_it() {
        // `<|n|>` is the configuration's EOS, so the cache takes it as a
        // candidate; it is extracted in the second pass, where the longer
        // `a<|n|>b` wins over it. A raw-text matcher cannot see that pass, so
        // `<|n|>` is no boundary (and the longer token, matched there too, is
        // no consuming pattern). Split at it, the cache would encode `a<|n|>`
        // as `a`, `<|n|>` and the rest after `<|n|>`, where the tokenizer has
        // one token.
        let config = CacheConfig {
            enable_l0: false,
            enable_l1: true,
            ..Default::default()
        };
        let plain = TwoPassTokenizer::new();
        let cached = CachedTokenizer::new(Arc::new(TwoPassTokenizer::new()), config);
        let text = "<|s|>xa<|n|>by<|s|>z";
        for input in [text, text] {
            assert_eq!(
                cached.encode(input, false).unwrap().token_ids(),
                plain.encode(input, false).unwrap().token_ids()
            );
        }
        let stats = cached.l1_cache_stats().unwrap();
        assert_eq!((stats.misses, stats.hits), (1, 1), "{stats:?}");
    }

    /// What [`FramesText`] puts after the text when `add_special_tokens` is set.
    #[derive(Clone, Copy)]
    enum Tail {
        /// An EOS: a frame the cache can apply around a result.
        Eos,
        /// The input's length as an id: no frame.
        TextLength,
    }

    /// Tokenizer whose post-processor frames a text when `add_special_tokens`
    /// is set: a BOS in front, `tail` after it; the chat specials atomic.
    struct FramesText {
        special_tokens: SpecialTokens,
        matcher: aho_corasick::AhoCorasick,
        tail: Tail,
    }

    const FRAME_BOS: TokenIdType = 1;
    const FRAME_EOS: TokenIdType = 2;

    impl FramesText {
        fn new(tail: Tail) -> Self {
            let specials = ["<|im_start|>", "<|im_end|>"];
            Self {
                special_tokens: SpecialTokens {
                    additional_special_tokens: specials.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                },
                matcher: aho_corasick::AhoCorasickBuilder::new()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(specials)
                    .unwrap(),
                tail,
            }
        }
    }

    impl Encoder for FramesText {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            let mut ids = Vec::new();
            if add_special_tokens {
                ids.push(FRAME_BOS);
            }
            let mut last = 0;
            for found in self.matcher.find_iter(input) {
                ids.extend(input[last..found.start()].bytes().map(TokenIdType::from));
                ids.push(1000 + found.pattern().as_u32());
                last = found.end();
            }
            ids.extend(input[last..].bytes().map(TokenIdType::from));
            if add_special_tokens {
                ids.push(match self.tail {
                    Tail::Eos => FRAME_EOS,
                    Tail::TextLength => {
                        TokenIdType::try_from(input.len()).unwrap_or(TokenIdType::MAX)
                    }
                });
            }
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for FramesText {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for FramesText {
        fn vocab_size(&self) -> usize {
            1002
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

    const FRAMED_TEXT: &str =
        "<|im_start|>system\nYou help.<|im_end|><|im_start|>user\nHi<|im_end|>";

    #[test]
    fn a_post_processor_that_appends_puts_its_token_at_the_end_of_the_whole_input() {
        let config = CacheConfig {
            enable_l0: false,
            enable_l1: true,
            ..Default::default()
        };
        let plain = FramesText::new(Tail::Eos);
        let cached = CachedTokenizer::new(Arc::new(FramesText::new(Tail::Eos)), config);
        let next_turn = format!("{FRAMED_TEXT}<|im_start|>assistant\nHello");
        // Per flag: a miss, a hit, a hit with a suffix (the deepest boundary
        // cached too), the hit there.
        for input in [
            FRAMED_TEXT,
            FRAMED_TEXT,
            next_turn.as_str(),
            next_turn.as_str(),
        ] {
            for add_special_tokens in [true, false] {
                assert_eq!(
                    cached
                        .encode(input, add_special_tokens)
                        .unwrap()
                        .token_ids(),
                    plain.encode(input, add_special_tokens).unwrap().token_ids(),
                    "{input:?} with add_special_tokens={add_special_tokens}"
                );
            }
        }
        let stats = cached.l1_cache_stats().unwrap();
        assert_eq!(
            (stats.misses, stats.hits, stats.fallbacks),
            (2, 6, 0),
            "both flag values are served from the cache: {stats:?}"
        );
    }

    #[test]
    fn a_flag_whose_additions_depend_on_the_text_is_served_by_the_plain_encode() {
        let config = CacheConfig {
            enable_l0: false,
            enable_l1: true,
            ..Default::default()
        };
        let plain = FramesText::new(Tail::TextLength);
        let cached = CachedTokenizer::new(Arc::new(FramesText::new(Tail::TextLength)), config);
        for _ in 0..2 {
            for add_special_tokens in [true, false] {
                assert_eq!(
                    cached
                        .encode(FRAMED_TEXT, add_special_tokens)
                        .unwrap()
                        .token_ids(),
                    plain
                        .encode(FRAMED_TEXT, add_special_tokens)
                        .unwrap()
                        .token_ids(),
                    "add_special_tokens={add_special_tokens}"
                );
            }
        }
        // Only the requests without the flag went through the cache.
        let stats = cached.l1_cache_stats().unwrap();
        assert_eq!((stats.misses, stats.hits), (1, 1), "{stats:?}");
    }

    /// Tokenizer with the special tokens `/think` and `<|im_end|>` and the
    /// longer, non-special added token `</think>`, all extracted whole.
    struct ThinkTokenizer {
        special_tokens: SpecialTokens,
        matcher: aho_corasick::AhoCorasick,
    }

    impl ThinkTokenizer {
        fn new() -> Self {
            Self {
                special_tokens: SpecialTokens {
                    additional_special_tokens: vec!["/think".to_owned(), "<|im_end|>".to_owned()],
                    ..Default::default()
                },
                matcher: aho_corasick::AhoCorasickBuilder::new()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(["/think", "<|im_end|>", "</think>"])
                    .unwrap(),
            }
        }
    }

    impl Encoder for ThinkTokenizer {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids = Vec::new();
            let mut last = 0;
            for found in self.matcher.find_iter(input) {
                ids.extend(input[last..found.start()].bytes().map(TokenIdType::from));
                ids.push(1000 + found.pattern().as_u32());
                last = found.end();
            }
            ids.extend(input[last..].bytes().map(TokenIdType::from));
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for ThinkTokenizer {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for ThinkTokenizer {
        fn vocab_size(&self) -> usize {
            1003
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
        fn added_token_strings(&self) -> Vec<String> {
            vec!["</think>".to_owned()]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn a_cached_tokenizer_passes_the_added_tokens_on_to_a_cache_around_it() {
        // The boundary matcher takes the added tokens from the tokenizer it
        // wraps; a cached tokenizer in between must hand them on, or the
        // `/think` inside `</think>` is a boundary for the cache around it.
        let config = CacheConfig {
            enable_l0: false,
            enable_l1: true,
            ..Default::default()
        };
        let plain = ThinkTokenizer::new();
        let inner = CachedTokenizer::new(Arc::new(ThinkTokenizer::new()), config.clone());
        let outer = CachedTokenizer::new(Arc::new(inner), config);
        assert_eq!(outer.added_token_strings(), vec!["</think>"]);
        for text in [
            "a</think>b /think c<|im_end|>d",
            "a</think>b /think c<|im_end|>d /think e",
        ] {
            assert_eq!(
                outer.encode(text, false).unwrap().token_ids(),
                plain.encode(text, false).unwrap().token_ids(),
                "{text}"
            );
        }
    }

    #[test]
    fn a_special_token_ending_inside_a_longer_one_is_no_split_point() {
        // A prefix cached after the short special token must not be spliced
        // into an input where the same bytes begin the long one: the tokenizer
        // reads them as a single token there, so the ids differ.
        let tokenizer = Arc::new(NestedSpecialsTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer.clone(), l1_only_config());
        let seen = format!("{SHORT_SPECIAL}first");
        assert_eq!(
            cached.encode(&seen, false).unwrap().token_ids(),
            tokenizer.encode(&seen, false).unwrap().token_ids()
        );

        let input = format!("{LONG_SPECIAL}second");
        let got = cached.encode(&input, false).unwrap();
        let want = tokenizer.encode(&input, false).unwrap();
        assert_eq!(
            got.token_ids(),
            want.token_ids(),
            "a hit on the short token's prefix splits the long token"
        );
    }

    /// Tokenizer whose first piece of an input gets a marker id, like a
    /// pre-tokenizer that prepends a word marker to the first piece only: the
    /// same text encodes differently at the start of an input and after a
    /// special token, so a suffix encoded on its own is wrong.
    struct PrependsToFirstPiece;

    const MARKER_ID: TokenIdType = 7;

    impl Encoder for PrependsToFirstPiece {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = Vec::new();
            atomic_ids(input, &mut ids);
            if ids.first().is_some_and(|&id| id < 200) {
                ids.insert(0, MARKER_ID);
            }
            Ok(Encoding::Plain(ids))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|i| self.encode(i, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for PrependsToFirstPiece {
        fn decode(&self, _token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
            Ok(String::new())
        }
    }

    impl traits::Tokenizer for PrependsToFirstPiece {
        fn vocab_size(&self) -> usize {
            256
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            static SPECIALS: std::sync::OnceLock<SpecialTokens> = std::sync::OnceLock::new();
            SPECIALS.get_or_init(|| SpecialTokens {
                additional_special_tokens: vec!["<|im_start|>".into(), "<|im_end|>".into()],
                ..Default::default()
            })
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

    #[test]
    fn segments_after_a_boundary_are_encoded_as_in_the_whole_text() {
        // A segment encoded on its own would get the first-piece marker the
        // whole text gives only to its very first piece; encoded after the
        // boundary's special token it does not. The miss (segment by segment)
        // and the hit (a cached prefix plus the suffix) both equal the plain
        // encode.
        let tokenizer = Arc::new(PrependsToFirstPiece);
        let cached = CachedTokenizer::new(tokenizer.clone(), l1_only_config());
        let first = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nq<|im_end|>";
        let second = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nq2<|im_end|>";
        for input in [first, second, "plain start<|im_end|>then more"] {
            for add_special_tokens in [false, true] {
                let got = cached.encode(input, add_special_tokens).unwrap();
                let want = tokenizer.encode(input, add_special_tokens).unwrap();
                assert_eq!(got.token_ids(), want.token_ids(), "{input:?}");
            }
        }
        assert!(
            cached.l1_cache_stats().unwrap().hits >= 1,
            "the second input hit the first's prefix"
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
            l0_max_memory: usize::MAX,
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

    #[test]
    fn forwards_deferred_chat_encodes_from_the_inner_tokenizer() {
        use crate::{
            chat_template::ChatTemplateParams,
            mock::MockTokenizer,
            traits::{PromptEncoding, Tokenizer as _},
        };

        let inner = MockTokenizer::new()
            .with_deferred_chat_ids(vec![7, 8, 9])
            .with_unbilled_prompt_tokens(3);
        let cached = CachedTokenizer::new(
            Arc::new(inner),
            CacheConfig {
                enable_l0: true,
                ..Default::default()
            },
        );
        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };

        let rendered = cached
            .apply_chat_template_with_encoding(&messages, params, Some("Sure"))
            .unwrap();
        assert!(
            rendered.text.ends_with("assistant: Sure"),
            "{}",
            rendered.text
        );
        assert_eq!(rendered.unbilled_prompt_tokens, 3);
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("the wrapper must hand the inner tokenizer's deferred encode through");
        };
        assert_eq!(job.run().unwrap().token_ids(), &[7, 8, 9]);

        // A flat inner tokenizer reports `FromText`, and the caller's
        // `encode(text)` still goes through the wrapper's caches.
        let flat = CachedTokenizer::new(
            Arc::new(MockTokenizer::new()),
            CacheConfig {
                enable_l0: true,
                ..Default::default()
            },
        );
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            ..Default::default()
        };
        let rendered = flat
            .apply_chat_template_with_encoding(&messages, params, None)
            .unwrap();
        assert!(matches!(rendered.encoding, PromptEncoding::FromText));
        let _ = flat.encode(&rendered.text, false).unwrap();
        let _ = flat.encode(&rendered.text, false).unwrap();
        assert_eq!(flat.cache_stats().map(|s| s.hits), Some(1));
    }
}

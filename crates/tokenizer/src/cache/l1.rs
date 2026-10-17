//! L1 Cache: Special-token boundary prefix cache
//!
//! Caches tokenization results at ALL special token boundaries.
//! Special tokens (like `<|im_start|>`, `<|im_end|>`) are atomic in BPE tokenizers (special: true, normalized: false),
//! making them the ONLY safe split points that guarantee correctness. A tokenizer extracts its added tokens in two
//! passes: the `normalized: false` ones from the raw text, then the `normalized: true` ones from the pieces that pass
//! leaves, after normalization. The cache lives in the first pass: its boundaries are raw-pass special tokens and the
//! tokens it matches so that no boundary falls inside them are raw-pass added tokens; a token of the second pass can
//! neither displace a raw-pass token nor be a boundary of a raw-text matcher.
//!
//! **Design**: Cache at every special token boundary (not at fixed granularity intervals)
//! - Simple: No granularity parameter, no search windows
//! - Efficient: Fewer cache entries (10 instead of 64 for typical 8KB prompt)
//! - Natural: Aligns with actual chat template structure
//!
//! Example:
//!
//! Template: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\n{query}<|im_end|>"
//!
//! Request 1: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>"
//! Request 2: "<|im_start|>system\nYou are helpful.<|im_end|><|im_start|>user\nHello!<|im_end|>"
//!
//! Cache points: After each "<|im_end|>" (atomic tokens, guaranteed safe)
//! Result: tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)

use std::{
    mem::size_of,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use anyhow::anyhow;
use blake3;
use dashmap::DashMap;

use super::activity::L1;
use crate::traits::{Encoder, Encoding, TokenIdType};

/// Hash type for cache keys
type Blake3Hash = [u8; 32];

/// A boundary of an input: the byte offset right after a special token, the
/// blake3 digest of `add_special_tokens || input[..end]`, and which special
/// token ends there (an index into [`SpecialTokenBoundaries`]). Computed once
/// per lookup and reused by the insert side, so a miss neither re-scans for
/// boundaries nor re-hashes prefixes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Seed {
    pub(super) end: usize,
    pub(super) digest: Blake3Hash,
    pub(super) special: usize,
}

/// The boundaries of an input, in order.
pub(super) type PrefixSeeds = Vec<Seed>;

/// Outcome of [`L1Cache::lookup_with_seeds`].
pub(super) enum PrefixLookup {
    /// Longest cached prefix: the shared token allocation plus the boundary
    /// it ends at (its byte length and the special token there). The tokens
    /// are returned as `Arc<[TokenIdType]>` so a hit does not copy the (large)
    /// cached prefix; the caller splices the suffix into an exact-capacity
    /// buffer. The last field is the input's deepest boundary when it lies
    /// beyond the match, so the caller can cache it.
    Hit(Arc<[TokenIdType]>, Seed, Option<Seed>),
    /// No cached prefix. Carries the seeds the failed search computed (empty
    /// when the input has no special-token boundaries) so the miss path can
    /// seed entries without recomputing them.
    Miss(PrefixSeeds),
}

/// Number of shards for concurrent access
const NUM_SHARDS: usize = 16;

/// The special tokens a cache may split at, as one leftmost-longest automaton
/// over the distinct strings: one pass over an input finds every boundary,
/// however many special tokens the vocabulary has (a few for ChatML, hundreds
/// where a vocabulary reserves them).
///
/// **ONLY special tokens** are split points: they are atomic (special: true,
/// normalized: false) in BPE vocabularies, which guarantees
/// `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)` after
/// one. No fallback to whitespace or punctuation: better not to cache than to
/// risk corruption. Leftmost-longest is how the tokenizer's own added-token
/// extraction decides between overlapping candidates, so a special token that
/// ends inside a longer one is not a boundary.
///
/// Common special tokens:
/// - ChatML: `<|im_start|>`, `<|im_end|>`
/// - Llama 3: `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>`
/// - GPT: `<|endoftext|>`
/// - Custom: `<|reserved_special_token_N|>`
pub(super) struct SpecialTokenBoundaries {
    /// `None` without special tokens: nothing is cacheable then. Its patterns
    /// are the special tokens first, then the other added tokens: a match of
    /// one of those consumes text (the tokenizer extracts it whole, so no
    /// special token inside it is a boundary) and is no boundary itself.
    automaton: Option<AhoCorasick>,
    /// The special tokens, in the automaton's pattern order.
    strings: Vec<String>,
    /// The id each special token encodes to on its own (one id each; see
    /// [`Self::for_encoder`]); empty when the matcher is for lookups only.
    ids: Vec<TokenIdType>,
}

/// Texts a special token is probed between at construction: letters, a
/// space-led word, digits, a non-ASCII character. A pre-tokenizer or an
/// added-token flag that treats the text after the token differently when
/// the token is not at the start of the input shows up on one of them.
const SPLICE_PROBES: [&str; 4] = ["ab", " x", "1 2", "\u{e9}t\u{e9}"];

impl SpecialTokenBoundaries {
    /// The matcher over `special_tokens`, for lookups: duplicates and empty
    /// strings are dropped. A cache that also encodes needs
    /// [`Self::for_encoder`].
    pub(super) fn new<S: AsRef<str>>(special_tokens: &[S]) -> Self {
        Self::build(Self::distinct(special_tokens), Vec::new(), Vec::new())
    }

    fn distinct<S: AsRef<str>>(tokens: &[S]) -> Vec<String> {
        let mut distinct: Vec<String> = tokens
            .iter()
            .map(|token| token.as_ref().to_owned())
            .filter(|token| !token.is_empty())
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        distinct
    }

    /// The matcher over the special tokens of `encoder` at which a splice is
    /// exact: a token keeps its place when it encodes to one id on its own and
    /// the encoder tokenizes `text + token + more` as `text + token` followed
    /// by `more` encoded after the token ([`Self::encode_segment_into`]) on every
    /// probe of [`SPLICE_PROBES`]. A token that fails (one the tokenizer
    /// extracts only as a whole word, one it normalizes, one it does not
    /// extract at all) is no boundary, so the cache never splits at it; the
    /// tokens that are left are logged at debug level with the ones dropped.
    /// `other_added_tokens` are the tokenizer's added tokens that are not
    /// special and are extracted in the same raw pass as the specials:
    /// matched so that no boundary falls inside one, never boundaries
    /// themselves (a token extracted after normalization is not among them:
    /// it cannot take a raw-pass token's place).
    pub(super) fn for_encoder<S: AsRef<str>, T: AsRef<str>, E: Encoder + ?Sized>(
        special_tokens: &[S],
        other_added_tokens: &[T],
        encoder: &E,
    ) -> Self {
        let specials = Self::distinct(special_tokens);
        let mut strings = Vec::with_capacity(specials.len());
        let mut ids = Vec::with_capacity(specials.len());
        let mut dropped = Vec::new();
        for token in specials {
            match Self::splices_exactly(&token, encoder) {
                Some(id) => {
                    strings.push(token);
                    ids.push(id);
                }
                None => dropped.push(token),
            }
        }
        if !dropped.is_empty() {
            tracing::debug!(
                kept = strings.len(),
                dropped = ?dropped,
                "special tokens the prefix cache does not split at: their splice is not the whole encode"
            );
        }
        // The tokens that can hide a boundary, the specials dropped above
        // among them: matched so that no boundary falls where the tokenizer
        // extracts a longer token whole. A token joins when it can take the
        // place of a token already in the matcher (a kept special, or one
        // that joined before it) in a leftmost-longest match, until nothing
        // joins: the rest can neither displace nor cut into a match of the
        // matcher's tokens and stay out, which keeps the automaton small.
        let mut pending: Vec<String> = Self::distinct(other_added_tokens)
            .into_iter()
            .chain(dropped)
            .filter(|token| !strings.contains(token))
            .collect();
        let mut included = strings.clone();
        let mut others = Vec::new();
        loop {
            let (hiding, rest): (Vec<String>, Vec<String>) = pending
                .into_iter()
                .partition(|token| Self::can_hide(token, &included));
            if hiding.is_empty() {
                break;
            }
            included.extend(hiding.iter().cloned());
            others.extend(hiding);
            pending = rest;
        }
        Self::build(strings, ids, others)
    }

    /// Whether `token`, extracted whole by the tokenizer, can take the place
    /// of one of the `included` tokens in a leftmost-longest match: it
    /// contains one, or it ends with the start of one (a match beginning
    /// earlier wins, so the token after it is text).
    fn can_hide(token: &str, included: &[String]) -> bool {
        included.iter().any(|special| {
            token.contains(special.as_str())
                || (1..token.len())
                    .filter(|&at| token.is_char_boundary(at))
                    .any(|at| special.starts_with(&token[at..]))
        })
    }

    fn build(strings: Vec<String>, ids: Vec<TokenIdType>, others: Vec<String>) -> Self {
        if strings.is_empty() {
            return Self {
                automaton: None,
                strings,
                ids,
            };
        }
        let automaton = match AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(strings.iter().chain(&others))
        {
            Ok(automaton) => Some(automaton),
            // Only a pattern set beyond the automaton's limits fails to build;
            // the cache then stays empty rather than wrong.
            Err(error) => {
                tracing::warn!(%error, "special-token boundary matcher not built; the prefix cache stays empty");
                None
            }
        };
        Self {
            automaton,
            strings,
            ids,
        }
    }

    /// The id `token` encodes to when the splice at it is exact on every
    /// probe, else `None`.
    fn splices_exactly<E: Encoder + ?Sized>(token: &str, encoder: &E) -> Option<TokenIdType> {
        let alone = encoder.encode(token, false).ok()?;
        let &[id] = alone.token_ids() else {
            return None;
        };
        for probe in SPLICE_PROBES {
            let whole = encoder
                .encode(&format!("{probe}{token}{probe}"), false)
                .ok()?;
            let head = encoder.encode(&format!("{probe}{token}"), false).ok()?;
            let mut spliced: Vec<TokenIdType> = head.token_ids().to_vec();
            Self::after_token_into(id, &format!("{token}{probe}"), encoder, &mut spliced).ok()?;
            if spliced != whole.token_ids() {
                return None;
            }
        }
        Some(id)
    }

    /// `text` starts with the special token `id`: encode it as a whole, so a
    /// pre-tokenizer that treats the first piece of an input specially sees
    /// what follows the token as a later piece, and append everything after
    /// that first id to `out`. An error when the token does not come back as
    /// that one id in front: the splice is not exact for this input.
    fn after_token_into<E: Encoder + ?Sized>(
        id: TokenIdType,
        text: &str,
        encoder: &E,
        out: &mut Vec<TokenIdType>,
    ) -> anyhow::Result<()> {
        let encoding = encoder.encode(text, false)?;
        match encoding.token_ids() {
            [first, rest @ ..] if *first == id => {
                out.extend_from_slice(rest);
                Ok(())
            }
            _ => Err(anyhow!(
                "the special token at the boundary does not encode as itself in front of the segment"
            )),
        }
    }

    /// Append to `out` the ids of `input[from.end..to]` as the tokenizer
    /// encodes that text in the whole input: the special token that ends at
    /// `from` is encoded in front of it (it sits right there in `input`, so
    /// nothing is copied) and its own id is dropped again. An error when the
    /// token does not come back as that one id in front: the splice is not
    /// exact for this input, and the caller falls back to the plain encode.
    pub(super) fn encode_segment_into<E: Encoder + ?Sized>(
        &self,
        from: &Seed,
        input: &str,
        to: usize,
        encoder: &E,
        out: &mut Vec<TokenIdType>,
    ) -> anyhow::Result<()> {
        let (Some(token), Some(&id)) = (self.strings.get(from.special), self.ids.get(from.special))
        else {
            return Err(anyhow!("a boundary without an encodable special token"));
        };
        let start = from
            .end
            .checked_sub(token.len())
            .filter(|&start| input.get(start..from.end) == Some(token.as_str()))
            .ok_or_else(|| anyhow!("the boundary does not end its special token"))?;
        Self::after_token_into(id, &input[start..to], encoder, out)
    }

    /// The positions right after each special token in `text`, in order,
    /// that leave a suffix to tokenize (never `text.len()`), each with the
    /// special token that ends there.
    pub(super) fn find(&self, text: &str) -> Vec<(usize, usize)> {
        let Some(automaton) = &self.automaton else {
            return Vec::new();
        };
        automaton
            .find_iter(text)
            .map(|found| (found.end(), found.pattern().as_usize()))
            .filter(|&(boundary, special)| special < self.strings.len() && boundary < text.len())
            .collect()
    }
}

/// A cached prefix entry
/// Uses Arc<[TokenIdType]> for zero-copy access to tokens
#[derive(Debug, Clone)]
struct CachedPrefix {
    /// The pre-computed token IDs for this prefix (Arc for zero-copy cloning)
    tokens: Arc<[TokenIdType]>,
    /// Last access timestamp (for LRU eviction)
    last_accessed: Arc<AtomicU64>,
    /// Size in bytes (for memory tracking during eviction)
    size_bytes: usize,
}

/// What `add_special_tokens` adds around a text's own ids: the ids the
/// tokenizer's post-processor puts in front of them (a BOS) and after them
/// (an EOS, a separator). The cache splices segments encoded without the flag
/// and adds the frame once around the whole result, so an appended token
/// lands at the end of the input and not after the first segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SpecialTokenFrame {
    prepended: Vec<TokenIdType>,
    appended: Vec<TokenIdType>,
}

impl SpecialTokenFrame {
    /// The frame `encoder` adds, read off every probe of [`SPLICE_PROBES`]
    /// encoded with and without the flag. `None` when the probes disagree
    /// (what the flag adds depends on the text) or a probe's own ids do not
    /// appear whole inside its flagged encode; the cache then does not serve
    /// requests with the flag.
    pub(super) fn probe<E: Encoder + ?Sized>(encoder: &E) -> Option<Self> {
        let mut frame: Option<Self> = None;
        for probe in SPLICE_PROBES {
            let with = encoder.encode(probe, true).ok()?;
            let without = encoder.encode(probe, false).ok()?;
            let (with, without) = (with.token_ids(), without.token_ids());
            if without.is_empty() {
                return None;
            }
            let at = (0..=with.len().checked_sub(without.len())?)
                .find(|&at| with[at..at + without.len()] == *without)?;
            let found = Self {
                prepended: with[..at].to_vec(),
                appended: with[at + without.len()..].to_vec(),
            };
            match &frame {
                None => frame = Some(found),
                Some(known) if *known == found => {}
                Some(_) => return None,
            }
        }
        frame
    }

    /// The ids the flag puts in front of the text's own.
    pub(super) fn prepended(&self) -> &[TokenIdType] {
        &self.prepended
    }

    /// The ids the flag puts after the text's own.
    pub(super) fn appended(&self) -> &[TokenIdType] {
        &self.appended
    }
}

/// L1 cache implementation with special-token-boundary prefix matching
pub struct L1Cache {
    /// Sharded maps for concurrent access
    /// Key: Blake3 hash of bytes[0..boundary]
    /// Value: Cached token IDs for that prefix
    shards: Vec<Arc<DashMap<Blake3Hash, CachedPrefix>>>,
    /// Serializes admission, eviction and clear; busy writers skip caching.
    admission: Mutex<()>,
    /// Maximum retained-entry charge (boundary bytes plus token bytes).
    /// Excludes map/Arc overhead and allocations still held by readers.
    max_memory: usize,
    /// Current memory usage estimate
    current_memory: AtomicU64,
    /// Cache hit counter
    hits: AtomicU64,
    /// Cache miss counter
    misses: AtomicU64,
    /// Lookups served by the plain encode after all: the boundary's special
    /// token did not come back as itself in front of a segment.
    fallbacks: AtomicU64,
    /// Monotonic counter for LRU timestamps
    access_counter: AtomicU64,
}

impl L1Cache {
    /// Create a new L1 cache with the specified memory limit
    pub fn new(max_memory: usize) -> Self {
        let shards = (0..NUM_SHARDS).map(|_| Arc::new(DashMap::new())).collect();

        Self {
            shards,
            admission: Mutex::new(()),
            max_memory,
            current_memory: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            fallbacks: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    /// Compute the seed list for `input`: the blake3 digest of
    /// `add_special_tokens || input[..boundary]` at every special token boundary,
    /// built with one incremental O(N) hashing pass.
    ///
    /// The two values of `add_special_tokens` seed distinct keys; the entries
    /// hold the text's own ids either way (the frame the flag adds is applied
    /// around a result, never stored).
    fn boundary_seeds(
        input: &str,
        special_tokens: &SpecialTokenBoundaries,
        add_special_tokens: bool,
    ) -> PrefixSeeds {
        let boundaries = special_tokens.find(input);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&[add_special_tokens as u8]);
        let mut seeds = Vec::with_capacity(boundaries.len());
        let mut last_pos = 0;
        let bytes = input.as_bytes();
        for &(boundary_pos, special) in &boundaries {
            hasher.update(&bytes[last_pos..boundary_pos]);
            // `finalize(&self)` borrows — no need to clone the hasher to keep updating it.
            seeds.push(Seed {
                end: boundary_pos,
                digest: *hasher.finalize().as_bytes(),
                special,
            });
            last_pos = boundary_pos;
        }
        seeds
    }

    /// Try to find the longest prefix match at special token boundaries
    /// Returns (cached_tokens, byte_offset) if found
    ///
    /// Uses pre-computed tokens cached during insertion.
    /// Returns the shared `Arc<[TokenIdType]>` directly — the caller splices suffix
    /// tokens into an exact-capacity buffer instead of cloning the cached prefix.
    pub fn longest_prefix_match(
        &self,
        input: &str,
        special_tokens: &[&str],
        add_special_tokens: bool,
    ) -> Option<(Arc<[TokenIdType]>, usize)> {
        let special_tokens = SpecialTokenBoundaries::new(special_tokens);
        match self.lookup_with_seeds(input, &special_tokens, add_special_tokens) {
            PrefixLookup::Hit(tokens, boundary, _) => Some((tokens, boundary.end)),
            PrefixLookup::Miss(_) => None,
        }
    }

    /// Like [`Self::longest_prefix_match`], but on a miss hands back the
    /// seeds computed during the failed search so the caller can
    /// seed entries via [`Self::populate_with_seeds`] without re-scanning the input
    /// for boundaries or re-hashing the prefixes. Takes the matcher the caller
    /// built once, so the request path builds nothing per call.
    pub(super) fn lookup_with_seeds(
        &self,
        input: &str,
        special_tokens: &SpecialTokenBoundaries,
        add_special_tokens: bool,
    ) -> PrefixLookup {
        let seeds = Self::boundary_seeds(input, special_tokens, add_special_tokens);

        if seeds.is_empty() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            L1.miss();
            return PrefixLookup::Miss(seeds);
        }

        // Search from the longest boundary to find the best match
        for &seed in seeds.iter().rev() {
            let shard_idx = seed.digest[0] as usize % NUM_SHARDS;

            if let Some(entry) = self.shards[shard_idx].get(&seed.digest) {
                // Update last accessed timestamp for LRU
                let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
                entry.last_accessed.store(timestamp, Ordering::Relaxed);

                self.hits.fetch_add(1, Ordering::Relaxed);
                L1.hit(seed.end);
                // Share the cached allocation instead of copying it on every hit.
                let deeper = seeds
                    .last()
                    .copied()
                    .filter(|deepest| deepest.end > seed.end);
                return PrefixLookup::Hit(Arc::clone(&entry.tokens), seed, deeper);
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        L1.miss();
        PrefixLookup::Miss(seeds)
    }

    /// Insert prefix entries at all special token boundaries (incremental hashing
    /// and tokenization, O(N)). The miss path of [`super::CachedTokenizer`] instead
    /// uses [`Self::populate_with_seeds`], which reuses this per-segment work to also
    /// return the full encoding.
    pub fn insert_at_boundaries<E: Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
        special_tokens: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<()> {
        let special_tokens =
            SpecialTokenBoundaries::for_encoder::<_, &str, _>(special_tokens, &[], tokenizer);
        let seeds = Self::boundary_seeds(input, &special_tokens, add_special_tokens);

        if seeds.is_empty() {
            return Ok(());
        }

        self.populate_boundaries(input, &seeds, tokenizer, &special_tokens, None)?;
        Ok(())
    }

    /// Miss-path encode: tokenize `input` exactly once, caching the cumulative prefix
    /// at every special token boundary, and return the assembled encoding, equal
    /// to an uncached `encode(input, add_special_tokens)`: the segments are
    /// encoded in context and the frame the flag adds goes around the whole.
    /// A request with the flag is served by the plain encode, uncached, when
    /// the frame cannot be read off the tokenizer ([`SpecialTokenFrame::probe`]).
    pub fn populate_and_encode<E: Encoder + ?Sized>(
        &self,
        input: &str,
        tokenizer: &E,
        special_tokens: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Encoding> {
        let frame = if add_special_tokens {
            match SpecialTokenFrame::probe(tokenizer) {
                Some(frame) => Some(frame),
                None => return tokenizer.encode(input, true),
            }
        } else {
            None
        };
        let special_tokens =
            SpecialTokenBoundaries::for_encoder::<_, &str, _>(special_tokens, &[], tokenizer);
        let seeds = Self::boundary_seeds(input, &special_tokens, add_special_tokens);
        self.populate_with_seeds(
            input,
            &seeds,
            tokenizer,
            &special_tokens,
            add_special_tokens,
            frame.as_ref(),
        )
    }

    /// Like [`Self::populate_and_encode`], but reuses the seed
    /// list a failed [`Self::lookup_with_seeds`] already computed, so the miss path
    /// neither re-scans for boundaries nor re-hashes the prefixes it just hashed.
    /// `frame` is what `add_special_tokens` adds around the ids when it is set
    /// (the caller's, read once at construction).
    pub(super) fn populate_with_seeds<E: Encoder + ?Sized>(
        &self,
        input: &str,
        seeds: &[Seed],
        tokenizer: &E,
        special_tokens: &SpecialTokenBoundaries,
        add_special_tokens: bool,
        frame: Option<&SpecialTokenFrame>,
    ) -> anyhow::Result<Encoding> {
        let Some(last) = seeds.last() else {
            // No special token boundaries — nothing cacheable; a single plain encode
            // (preserves the inner tokenizer's native encoding variant).
            return tokenizer.encode(input, add_special_tokens);
        };

        // Tokenize + cache every boundary prefix; `running` covers input[0..last
        // boundary], behind the frame's leading ids.
        let mut running =
            self.populate_boundaries(input, seeds, tokenizer, special_tokens, frame)?;

        // The trailing segment after the last boundary is not a cache key (boundaries
        // exclude input.len()); encoding it, after the special token that ends
        // the boundary, completes the full tokenization; the frame's trailing ids
        // come last.
        special_tokens.encode_segment_into(last, input, input.len(), tokenizer, &mut running)?;
        if let Some(frame) = frame {
            running.extend_from_slice(frame.appended());
        }
        Ok(Encoding::Plain(running))
    }

    /// Shared core of the insert and fused-miss paths: walk the precomputed
    /// `seeds`, encoding each segment once without the flag (the first as the
    /// start of the input, each later one after the special token that ends
    /// the boundary before it), cache the cumulative prefix at each boundary,
    /// and return the running token vector (covering `input[0..last boundary]`)
    /// behind `frame`'s leading ids when there is a frame; the entries never
    /// hold the frame. Nothing is stored before every segment came back exact.
    fn populate_boundaries<E: Encoder + ?Sized>(
        &self,
        input: &str,
        seeds: &[Seed],
        tokenizer: &E,
        special_tokens: &SpecialTokenBoundaries,
        frame: Option<&SpecialTokenFrame>,
    ) -> anyhow::Result<Vec<TokenIdType>> {
        let mut running_tokens: Vec<TokenIdType> =
            frame.map_or_else(Vec::new, |frame| frame.prepended().to_vec());
        let head = running_tokens.len();
        let mut last: Option<Seed> = None;
        let mut prefix_lengths = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            // The first segment is the start of the input; every later segment
            // is encoded after the special token that ends the boundary before
            // it, as it is in the whole text.
            match last {
                None => {
                    let segment = tokenizer.encode(&input[..seed.end], false)?;
                    running_tokens.extend_from_slice(segment.token_ids());
                }
                Some(before) => special_tokens.encode_segment_into(
                    &before,
                    input,
                    seed.end,
                    tokenizer,
                    &mut running_tokens,
                )?,
            }

            prefix_lengths.push(running_tokens.len());

            last = Some(seed);
        }

        // Keep only offsets until every segment is known to splice exactly.
        // Copy each admitted prefix directly from the completed running buffer.
        for (seed, end) in seeds.iter().zip(prefix_lengths) {
            self.insert_prefix(seed.digest, seed.end, &running_tokens[head..end]);
        }

        Ok(running_tokens)
    }

    /// A lookup the plain encode served after all (see
    /// [`SpecialTokenBoundaries::encode_segment_into`]).
    pub(super) fn note_fallback(&self) {
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Cache one cumulative prefix ending at `boundary_pos`, only if it fits.
    /// Admission never waits for another writer. Reserve space before copying
    /// tokens; lookups may keep evicted Arcs alive outside this retained budget.
    pub(super) fn insert_prefix(
        &self,
        hash_bytes: Blake3Hash,
        boundary_pos: usize,
        tokens: &[TokenIdType],
    ) {
        let Some(size_bytes) = tokens
            .len()
            .checked_mul(size_of::<TokenIdType>())
            .and_then(|bytes| boundary_pos.checked_add(bytes))
        else {
            return;
        };
        if self.max_memory == 0 || size_bytes > self.max_memory {
            return;
        }
        let Ok(_admission) = self.admission.try_lock() else {
            return;
        };
        let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;
        // An existing hash already represents the same prefix. Avoid copying
        // or evicting it merely because another miss encoded it concurrently.
        if let Some(cached) = self.shards[shard_idx].get(&hash_bytes) {
            cached.last_accessed.store(
                self.access_counter.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            return;
        }
        let current = self.current_memory.load(Ordering::Relaxed) as usize;
        let available = self.max_memory - current;
        if size_bytes > available {
            self.evict_lru(size_bytes - available);
        }
        let cached = CachedPrefix {
            tokens: tokens.into(),
            last_accessed: Arc::new(AtomicU64::new(self.access_counter.load(Ordering::Relaxed))),
            size_bytes,
        };
        self.shards[shard_idx].insert(hash_bytes, cached);
        self.current_memory
            .fetch_add(size_bytes as u64, Ordering::Relaxed);
    }

    /// Evict least recently used entries using approximate LRU via random sampling
    ///
    /// This uses an approximate LRU strategy that's much faster than true LRU:
    /// - Samples K random entries from the cache (K=32)
    /// - Evicts the oldest entry among the samples
    /// - Repeats until enough space is freed
    ///
    /// This provides O(samples) complexity instead of O(total_entries * log(total_entries)),
    /// avoiding latency spikes when eviction is triggered on large caches.
    ///
    /// The approximation is excellent in practice - sampling 32 entries from a large cache
    /// gives high probability of finding very old entries.
    /// Caller holds admission, so removals and accounting cannot race writers or clear.
    fn evict_lru(&self, space_needed: usize) {
        const SAMPLE_SIZE: usize = 32; // Number of entries to sample per eviction round
        let mut freed = 0usize;
        let mut iteration = 0usize;

        // Keep evicting until we have enough space
        while freed < space_needed {
            // Collect samples from shards
            let mut samples: Vec<(usize, Blake3Hash, u64, usize)> = Vec::with_capacity(SAMPLE_SIZE);

            // Sample entries across different shards
            for i in 0..SAMPLE_SIZE {
                // Distribute samples across shards using iteration and index for variety
                let shard_idx = (iteration * SAMPLE_SIZE + i) % NUM_SHARDS;

                // Get first entry from that shard (DashMap iteration order is arbitrary)
                if let Some(entry) = self.shards[shard_idx].iter().next() {
                    let hash = *entry.key();
                    let timestamp = entry.value().last_accessed.load(Ordering::Relaxed);
                    let size = entry.value().size_bytes;
                    samples.push((shard_idx, hash, timestamp, size));
                }
            }

            if samples.is_empty() {
                // Cache is empty, nothing to evict
                break;
            }

            // Find the oldest entry among samples
            if let Some((shard_idx, hash, _, _)) =
                samples.iter().min_by_key(|(_, _, ts, _)| ts).copied()
            {
                // Remove it
                if let Some((_, removed)) = self.shards[shard_idx].remove(&hash) {
                    L1.evict();
                    freed += removed.size_bytes;
                    self.current_memory
                        .fetch_sub(removed.size_bytes as u64, Ordering::Relaxed);
                }
            }

            iteration += 1;
        }
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get cache statistics
    pub fn stats(&self) -> L1CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        L1CacheStats {
            hits,
            misses,
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            entries: self.len(),
            memory_bytes: self.current_memory.load(Ordering::Relaxed) as usize,
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Clear retained entries and counters while excluding cache writers.
    pub fn clear(&self) {
        let _admission = self
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for shard in &self.shards {
            shard.clear();
        }
        self.current_memory.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.fallbacks.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct L1CacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Lookups the plain encode served after all (a boundary's special token
    /// did not come back as itself in front of a segment); counted among the
    /// hits or misses they started as.
    pub fallbacks: u64,
    pub entries: usize,
    /// Retained-entry charge, bounded by the configured limit. This estimate
    /// excludes container overhead and evicted tokens still held by readers.
    pub memory_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::{SpecialTokenBoundaries, SpecialTokenFrame};
    use crate::*;

    /// Oversized prefixes must be rejected without displacing useful entries.
    #[test]
    fn budget_rejects_oversized_prefix() {
        let cache = L1Cache::new(128);
        cache.insert_prefix([1; 32], 8, &[1; 8]);
        cache.insert_prefix([2; 32], 100, &[2; 100]);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().memory_bytes, 40);
    }

    /// A disabled cache must retain no entries, including empty prefixes.
    #[test]
    fn budget_zero_retains_nothing() {
        let cache = L1Cache::new(0);
        cache.insert_prefix([1; 32], 1, &[1]);
        cache.insert_prefix([2; 32], 0, &[]);
        assert!(cache.is_empty());
        assert_eq!(cache.stats().memory_bytes, 0);
    }

    /// Many cumulative prefixes must fit the budget without changing encoding.
    #[test]
    fn budget_many_boundaries_preserves_encoding() {
        let input = format!("{}tail", "abcdef<|im_end|>".repeat(15000));
        for flag in [false, true] {
            let cache = L1Cache::new(4096);
            let encoded = cache
                .populate_and_encode(&input, &BosTokenizer, &SPECIALS, flag)
                .unwrap();
            assert_eq!(
                encoded.token_ids(),
                BosTokenizer.encode(&input, flag).unwrap().token_ids()
            );
            assert!(cache.stats().memory_bytes <= 4096);
        }
    }

    /// Contending admissions skip allocation rather than waiting for the writer.
    #[test]
    fn budget_busy_writer_skips_insertion() {
        let cache = L1Cache::new(1024);
        let _guard = cache.admission.lock().unwrap();
        cache.insert_prefix([0; 32], 4, &[1; 4]);
        assert!(cache.is_empty());
    }

    /// Concurrent admissions and clears preserve the retained accounting.
    #[test]
    fn budget_concurrent_admission_and_clear() {
        use std::sync::{Arc, Barrier};
        let cache = Arc::new(L1Cache::new(4096));
        let barrier = Arc::new(Barrier::new(9));
        std::thread::scope(|scope| {
            for worker in 0..8u8 {
                let cache = &cache;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for i in 0..1000u32 {
                        let mut key = [worker; 32];
                        key[1..5].copy_from_slice(&i.to_le_bytes());
                        cache.insert_prefix(key, 128, &[1; 128]);
                        assert!(cache.stats().memory_bytes <= 4096);
                    }
                });
            }
            barrier.wait();
            for _ in 0..1000 {
                cache.clear();
            }
        });
        let retained: usize = cache
            .shards
            .iter()
            .flat_map(|s| s.iter().map(|e| e.size_bytes).collect::<Vec<_>>())
            .sum();
        assert_eq!(cache.stats().memory_bytes, retained);
        assert!(retained <= 4096);
        cache.clear();
        assert_eq!(cache.stats().memory_bytes, 0);
        assert!(cache.is_empty());
    }

    /// Eviction releases cache accounting, while a reader can still own its Arc.
    #[test]
    fn budget_evicted_reader_keeps_valid_tokens() {
        let cache = L1Cache::new(40);
        cache.insert_prefix([1; 32], 8, &[1; 8]);
        let held = cache.shards[1].get(&[1; 32]).unwrap().tokens.clone();
        cache.insert_prefix([2; 32], 8, &[2; 8]);
        assert_eq!(cache.stats().memory_bytes, 40);
        assert_eq!(&*held, &[1; 8]);
        cache.clear();
        assert_eq!(cache.stats().memory_bytes, 0);
        assert_eq!(&*held, &[1; 8]);
    }

    /// The special tokens of the test encoders below: atomic (one id each),
    /// extracted leftmost-longest as a BPE tokenizer's added vocabulary does
    /// it; `<|a|>` is a prefix of `<|a|>b|>` on purpose.
    const SPECIALS: [&str; 5] = [
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|a|>",
        "<|a|>b|>",
    ];

    /// Encoder with the atomic [`SPECIALS`] (ids 1000 and up) and one id per
    /// byte otherwise; a sentinel BOS when `add_special_tokens` is set, so the
    /// same text yields different tokens per flag.
    struct BosTokenizer;

    const BOS_ID: TokenIdType = 99;

    impl Encoder for BosTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            let matcher = aho_corasick::AhoCorasickBuilder::new()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(SPECIALS)
                .unwrap();
            let mut ids: Vec<TokenIdType> = Vec::new();
            if add_special_tokens {
                ids.push(BOS_ID);
            }
            let mut last = 0;
            for found in matcher.find_iter(input) {
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

    /// The byte offsets of a matcher's boundaries.
    fn ends(boundaries: Vec<(usize, usize)>) -> Vec<usize> {
        boundaries.into_iter().map(|(end, _)| end).collect()
    }

    #[test]
    fn test_basic_prefix_match() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        // Realistic ChatML template with special tokens
        let input1 = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nHello there! How are you doing today?<|im_end|>";

        // Insert at special token boundaries (re-tokenizes prefixes)
        cache
            .insert_at_boundaries(input1, &tokenizer, special_tokens, false)
            .unwrap();

        // Should have cached at special token boundaries
        assert!(!cache.is_empty());

        // Search with same prefix but different user query
        let input2 = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>";
        let result = cache.longest_prefix_match(input2, special_tokens, false);

        // Should find a match at the special token boundary (after system message)
        assert!(result.is_some());
        let (tokens, offset) = result.unwrap();
        assert!(offset > 0);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_short_input_with_boundaries() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        // Short input with special tokens
        let input = "<|im_start|>user\nHi<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();

        // Should cache at <|im_start|> boundary (has suffix left)
        assert!(!cache.is_empty());

        // Should find a match
        let result = cache.longest_prefix_match(input, special_tokens, false);
        assert!(result.is_some());
    }

    #[test]
    fn test_longest_match() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        // Create multi-turn conversation with multiple special token boundaries (~400 bytes)
        let input = "<|im_start|>system\nYou are a helpful AI assistant that provides detailed and accurate responses.<|im_end|><|im_start|>user\nHello there! How are you today? Can you help me understand how tokenization works in language models?<|im_end|><|im_start|>assistant\nI'm doing well, thank you! I'd be happy to explain tokenization. Tokenization is the process of breaking text into smaller units called tokens.<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();

        // Should have multiple entries at special token boundaries
        assert!(cache.len() >= 2); // At least 2 boundaries

        // Search with partial conversation - should match at a special token boundary
        let partial_input = "<|im_start|>system\nYou are a helpful AI assistant that provides detailed and accurate responses.<|im_end|><|im_start|>user\nHello there! How are you today? Can you help me understand how tokenization works in language models?<|im_end|>";
        let result = cache.longest_prefix_match(partial_input, special_tokens, false);

        // Should find a match at a special token boundary
        assert!(result.is_some());
        let (_, offset) = result.unwrap();
        assert!(offset > 0);
        assert!(offset <= partial_input.len());
    }

    #[test]
    fn test_stats() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        // ChatML input with special tokens
        let input = "<|im_start|>system\nYou are a helpful assistant that provides detailed answers.<|im_end|><|im_start|>user\nHello there! How are you today?<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();

        // Try to find match
        let _ = cache.longest_prefix_match(input, special_tokens, false);

        let stats = cache.stats();
        // Should have at least one hit (the longest special token boundary should match)
        assert!(stats.hits >= 1);
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[test]
    fn test_clear() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        // ChatML input with special tokens
        let input = "<|im_start|>system\nYou are a helpful assistant that provides clear and detailed responses.<|im_end|><|im_start|>user\nHello there!<|im_end|>";

        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_lru_eviction() {
        // Create a small cache (5KB) to trigger eviction
        let cache = L1Cache::new(5 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>", "<|eot_id|>"];
        let tokenizer = BosTokenizer;

        // Insert first conversation
        let input1 = "<|im_start|>system\nYou are a helpful assistant specialized in mathematics.<|im_end|><|im_start|>user\nCan you explain calculus to me?<|im_end|><|im_start|>assistant\nCertainly! Calculus is a branch of mathematics that studies continuous change.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input1, &tokenizer, special_tokens, false)
            .unwrap();

        // Access the first entry to update its timestamp
        let result = cache.longest_prefix_match(input1, special_tokens, false);
        assert!(result.is_some());

        // Insert second conversation
        let input2 = "<|im_start|>system\nYou are a helpful assistant specialized in physics.<|im_end|><|im_start|>user\nWhat is quantum mechanics?<|im_end|><|im_start|>assistant\nQuantum mechanics is the fundamental theory describing nature at atomic and subatomic scales.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input2, &tokenizer, special_tokens, false)
            .unwrap();

        // Access the second entry to make it more recent
        let result = cache.longest_prefix_match(input2, special_tokens, false);
        assert!(result.is_some());

        // Insert third conversation (should trigger eviction of oldest)
        let input3 = "<|im_start|>system\nYou are a helpful assistant specialized in chemistry.<|im_end|><|im_start|>user\nExplain the periodic table to me please.<|im_end|><|im_start|>assistant\nThe periodic table is a tabular arrangement of chemical elements organized by atomic number and electron configuration.<|im_end|><|eot_id|>";
        cache
            .insert_at_boundaries(input3, &tokenizer, special_tokens, false)
            .unwrap();

        // Verify cache didn't exceed max memory
        let stats = cache.stats();
        assert!(stats.memory_bytes <= 5 * 1024);

        // The most recently accessed entries should still be present
        let result = cache.longest_prefix_match(input3, special_tokens, false);
        assert!(result.is_some());
    }

    #[test]
    fn test_concurrent_access() {
        use std::{sync::Arc, thread};

        let cache = Arc::new(L1Cache::new(1024 * 1024));
        let special_tokens_owned: Vec<String> =
            vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];
        let special_tokens_arc = Arc::new(special_tokens_owned);

        let mut handles = vec![];

        // Spawn 10 threads that each insert different special-token-bounded strings
        // and query for prefix matches concurrently.
        for i in 0..10 {
            let cache_clone = cache.clone();
            let st_clone = special_tokens_arc.clone();
            handles.push(thread::spawn(move || {
                let tokenizer = BosTokenizer;
                let special_tokens: Vec<&str> = st_clone.iter().map(|s| s.as_str()).collect();

                // Each thread uses a unique user message to avoid hash collisions
                let input = format!(
                    "<|im_start|>system\nYou are assistant number {i}.<|im_end|>\
                     <|im_start|>user\nThread {i} says hello world test token.<|im_end|>"
                );

                // Insert prefix entries at boundaries
                cache_clone
                    .insert_at_boundaries(&input, &tokenizer, &special_tokens, false)
                    .unwrap();

                // Query for the same input - should find a prefix match
                let result = cache_clone.longest_prefix_match(&input, &special_tokens, false);
                assert!(
                    result.is_some(),
                    "Thread {i} expected a prefix match after insertion"
                );

                let (tokens, offset) = result.unwrap();
                assert!(
                    !tokens.is_empty(),
                    "Thread {i} expected non-empty cached tokens"
                );
                assert!(offset > 0, "Thread {i} expected positive byte offset");
                assert!(
                    offset <= input.len(),
                    "Thread {i}: offset {offset} exceeds input length {}",
                    input.len()
                );
            }));
        }

        // Wait for all threads to complete (no panics)
        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should contain entries from the concurrent inserts
        assert!(!cache.is_empty());

        // Memory tracking should be consistent (non-zero after inserts)
        let stats = cache.stats();
        assert!(
            stats.memory_bytes > 0,
            "Expected non-zero memory tracking after concurrent inserts"
        );
        assert!(
            stats.entries > 0,
            "Expected non-zero cache entries after concurrent inserts"
        );
        // Total hits should be at least 10 (one per thread)
        assert!(
            stats.hits >= 10,
            "Expected at least 10 cache hits, got {}",
            stats.hits
        );
    }

    #[test]
    fn test_add_special_tokens_separates_keys() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;
        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nq<|im_end|>";

        // Insert the same input under both flags.
        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, true)
            .unwrap();
        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();

        // Each flag has its own entry (the keys differ); both hold the text's
        // own ids, as the frame the flag adds goes around a result, never
        // into the cache.
        let (with_flag, _) = cache
            .longest_prefix_match(input, special_tokens, true)
            .expect("match for add_special_tokens=true");
        let (without_flag, _) = cache
            .longest_prefix_match(input, special_tokens, false)
            .expect("match for add_special_tokens=false");

        assert_eq!(with_flag, without_flag);
        assert_ne!(with_flag.first(), Some(&BOS_ID));
    }

    #[test]
    fn test_opposite_flag_does_not_collide() {
        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;
        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nq<|im_end|>";

        // Only the `true` flag is populated.
        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, true)
            .unwrap();

        // A lookup with the opposite flag must miss rather than return the other
        // flag's entry.
        assert!(cache
            .longest_prefix_match(input, special_tokens, false)
            .is_none());
    }

    /// The boundary scan before the one-pass matcher: every special string
    /// searched on its own over the whole text.
    fn boundaries_per_string(text: &str, special_tokens: &[&str]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        for &token in special_tokens {
            let mut start = 0;
            while let Some(pos) = text[start..].find(token) {
                let boundary = start + pos + token.len();
                if boundary < text.len() {
                    boundaries.push(boundary);
                }
                start = boundary;
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries
    }

    #[test]
    fn one_pass_boundaries_equal_the_per_string_scan_for_disjoint_special_tokens() {
        let special_tokens = &[
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
        ];
        let matcher = SpecialTokenBoundaries::new(special_tokens);
        let pieces = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
            "user\n",
            "assistant\n",
            "hello ",
            "<|im",
            "|>",
            "\n",
            "world",
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for round in 0..500 {
            let mut text = String::new();
            for _ in 0..(round % 23) {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                text.push_str(pieces[(state >> 33) as usize % pieces.len()]);
            }
            assert_eq!(
                ends(matcher.find(&text)),
                boundaries_per_string(&text, special_tokens),
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_special_token_ending_inside_a_longer_one_is_no_boundary() {
        let special_tokens = &["<|a|>", "<|a|>b|>"];
        let matcher = SpecialTokenBoundaries::new(special_tokens);
        // The tokenizer reads the long token whole: its only boundary is its end,
        // where the per-string scan also found the short token's end inside it.
        assert_eq!(ends(matcher.find("<|a|>b|>zzz")), vec![8]);
        assert_eq!(
            boundaries_per_string("<|a|>b|>zzz", special_tokens),
            vec![5, 8]
        );
        // On its own the short token is a boundary as before.
        assert_eq!(ends(matcher.find("<|a|>zzz")), vec![5]);
        // The end of the text is never a boundary; duplicates and empty strings
        // in the list change nothing.
        assert_eq!(ends(matcher.find("x<|a|>")), Vec::<usize>::new());
        assert_eq!(
            ends(SpecialTokenBoundaries::new(&["<|a|>", "<|a|>", ""]).find("<|a|>y")),
            vec![5]
        );
        assert_eq!(
            ends(SpecialTokenBoundaries::new::<&str>(&[]).find("<|a|>y")),
            Vec::<usize>::new()
        );
    }

    /// Encoder that extracts `<w>` only between non-letters (a whole-word
    /// special token) and `<|im_end|>` anywhere.
    struct WholeWordSpecial;

    impl Encoder for WholeWordSpecial {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = Vec::new();
            let bytes = input.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let rest = &bytes[i..];
                if rest.starts_with(b"<|im_end|>") {
                    ids.push(1001);
                    i += "<|im_end|>".len();
                } else if rest.starts_with(b"<w>")
                    && !(i > 0 && bytes[i - 1].is_ascii_alphabetic())
                    && !bytes.get(i + 3).is_some_and(|b| b.is_ascii_alphabetic())
                {
                    ids.push(1000);
                    i += 3;
                } else {
                    ids.push(TokenIdType::from(bytes[i]));
                    i += 1;
                }
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

    #[test]
    fn a_special_token_whose_splice_is_not_the_whole_encode_is_no_boundary() {
        // `<w>` between letters is text to the encoder but a token at the end
        // of a prefix, so a prefix cut after it would encode differently from
        // the whole text: the self-check drops it. `<|im_end|>` stays.
        let matcher = SpecialTokenBoundaries::for_encoder::<_, &str, _>(
            &["<w>", "<|im_end|>"],
            &[],
            &WholeWordSpecial,
        );
        assert_eq!(ends(matcher.find("ab<w>cd<|im_end|>ef")), vec![17]);
        assert_eq!(matcher.strings, vec!["<|im_end|>"]);
        // A matcher for lookups only keeps every string.
        assert_eq!(
            ends(SpecialTokenBoundaries::new(&["<w>", "<|im_end|>"]).find("ab<w>cd<|im_end|>ef")),
            vec![5, 17]
        );
    }

    /// Encoder with the special `/think` and the longer, non-special added
    /// token `</think>` that contains it, both atomic.
    struct ThinkTokens;

    impl Encoder for ThinkTokens {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let matcher = aho_corasick::AhoCorasickBuilder::new()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(["/think", "</think>", "<|im_end|>"])
                .unwrap();
            let mut ids: Vec<TokenIdType> = Vec::new();
            let mut last = 0;
            for found in matcher.find_iter(input) {
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

    /// Encoder like [`ThinkTokens`], but `</think>` is a special token the
    /// encoder extracts only at the end of its input (so the self-check
    /// drops it), while `/think` is extracted anywhere.
    struct ThinkTokensEndOnly;

    impl Encoder for ThinkTokensEndOnly {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = Vec::new();
            let (body, closed) = match input.strip_suffix("</think>") {
                Some(body) => (body, true),
                None => (input, false),
            };
            let mut last = 0;
            for (start, _) in body.match_indices("/think") {
                ids.extend(body[last..start].bytes().map(TokenIdType::from));
                ids.push(1000);
                last = start + "/think".len();
            }
            ids.extend(body[last..].bytes().map(TokenIdType::from));
            if closed {
                ids.push(1001);
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

    #[test]
    fn a_dropped_special_token_still_hides_the_boundaries_inside_it() {
        let matcher = SpecialTokenBoundaries::for_encoder::<_, &str, _>(
            &["/think", "</think>"],
            &[],
            &ThinkTokensEndOnly,
        );
        assert_eq!(
            matcher.strings,
            vec!["/think"],
            "the end-only token fails the self-check"
        );
        // `</think>` is no boundary itself, and the `/think` inside it is none
        // either: the dropped token still consumes the text it matches (one
        // boundary fewer where the tokenizer would split, never a wrong one).
        assert_eq!(ends(matcher.find("a</think>b /think c")), vec![17]);
    }

    #[test]
    fn an_added_token_ending_with_the_start_of_a_special_token_hides_it() {
        let matcher = SpecialTokenBoundaries::for_encoder(
            &["/think", "<|im_end|>"],
            &["b /t", "zz"],
            &ThinkTokens,
        );
        // Declared as an added token, `b /t` begins before the `/think` after
        // `b ` and wins the leftmost match: that `/think` is text. `zz` can
        // take no special token's place and stays out of the automaton.
        assert_eq!(ends(matcher.find("ab /think c<|im_end|>d")), vec![21]);
        assert_eq!(
            matcher
                .automaton
                .as_ref()
                .map(aho_corasick::AhoCorasick::patterns_len),
            Some(3)
        );
    }

    /// Encoder with the atomic special tokens `STOP` and `QQ` and the atomic
    /// added tokens `kSTO`, `mQQk`, `nm` and `Pp`, extracted leftmost-longest.
    struct ChainedTokens;

    impl Encoder for ChainedTokens {
        fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            let matcher = aho_corasick::AhoCorasickBuilder::new()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(["STOP", "QQ", "kSTO", "mQQk", "nm", "Pp"])
                .unwrap();
            let mut ids: Vec<TokenIdType> = Vec::new();
            let mut last = 0;
            for found in matcher.find_iter(input) {
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

    #[test]
    fn a_token_that_hides_a_hiding_token_joins_the_matcher_too() {
        let matcher = SpecialTokenBoundaries::for_encoder(
            &["STOP", "QQ"],
            &["kSTO", "mQQk", "nm", "Pp"],
            &ChainedTokens,
        );
        // `kSTO` and `mQQk` can take a special token's place; `nm` can take
        // `mQQk`'s. On `nmQQkSTOPping` the tokenizer extracts `nm`, `QQ`,
        // `kSTO` and `Pp`, never `STOP`: without `nm` the matcher would take
        // `mQQk` at 1 and then `STOP` at 5, a boundary the tokenizer has not.
        // `Pp` can take no included token's place and stays out.
        assert_eq!(ends(matcher.find("nmQQkSTOPping")), vec![4]);
        assert_eq!(
            matcher
                .automaton
                .as_ref()
                .map(aho_corasick::AhoCorasick::patterns_len),
            Some(5)
        );
    }

    #[test]
    fn a_special_token_inside_a_longer_added_token_is_no_boundary() {
        let matcher = SpecialTokenBoundaries::for_encoder(
            &["/think", "<|im_end|>"],
            &["</think>"],
            &ThinkTokens,
        );
        // `</think>` is extracted whole by the tokenizer: the `/think` inside
        // it is no boundary; a `/think` on its own still is.
        assert_eq!(
            ends(matcher.find("a</think>b /think c<|im_end|>d")),
            vec![17, 29]
        );
        let without = SpecialTokenBoundaries::for_encoder::<_, &str, _>(
            &["/think", "<|im_end|>"],
            &[],
            &ThinkTokens,
        );
        assert_eq!(
            ends(without.find("a</think>b /think c<|im_end|>d")),
            vec![8, 17, 29]
        );
    }

    #[test]
    fn test_hit_returns_shared_prefix_allocation() {
        use std::sync::Arc;

        let cache = L1Cache::new(1024 * 1024);
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;

        let input = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nWhat is 2+2?<|im_end|>";
        cache
            .insert_at_boundaries(input, &tokenizer, special_tokens, false)
            .unwrap();

        let (first, first_offset) = cache
            .longest_prefix_match(input, special_tokens, false)
            .expect("hit after insert");
        let (second, second_offset) = cache
            .longest_prefix_match(input, special_tokens, false)
            .expect("hit after insert");

        assert_eq!(first_offset, second_offset);
        // The hit path must hand back the cached allocation itself (shared
        // ownership) — not a fresh per-hit copy of the prefix tokens.
        assert!(
            Arc::ptr_eq(&first, &second),
            "longest_prefix_match must return the shared cached Arc, not a clone of its contents"
        );
    }

    /// Encoder with one id per byte and, when `add_special_tokens` is set,
    /// the input's length as a trailing id: what the flag adds depends on
    /// the text.
    struct LengthTail;

    impl Encoder for LengthTail {
        fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
            let mut ids: Vec<TokenIdType> = input.bytes().map(TokenIdType::from).collect();
            if add_special_tokens {
                ids.push(TokenIdType::try_from(input.len()).unwrap_or(TokenIdType::MAX));
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

    #[test]
    fn the_frame_probe_reads_what_the_flag_adds() {
        let frame = SpecialTokenFrame::probe(&BosTokenizer).expect("a constant frame");
        assert_eq!(frame.prepended(), &[BOS_ID]);
        assert_eq!(frame.appended(), &[] as &[TokenIdType]);
        assert_eq!(
            SpecialTokenFrame::probe(&LengthTail),
            None,
            "a text-dependent addition is no frame"
        );
    }

    #[test]
    fn test_populate_and_encode_matches_uncached_and_seeds_cache() {
        let special_tokens = &["<|im_start|>", "<|im_end|>"];
        let tokenizer = BosTokenizer;
        let input = "<|im_start|>system\nhi<|im_end|><|im_start|>user\nquery<|im_end|>";

        for add_special_tokens in [false, true] {
            let cache = L1Cache::new(1024 * 1024);

            // The fused miss path must return ids byte-exact to an uncached encode.
            let assembled = cache
                .populate_and_encode(input, &tokenizer, special_tokens, add_special_tokens)
                .unwrap();
            let plain = tokenizer.encode(input, add_special_tokens).unwrap();
            assert_eq!(
                assembled.token_ids(),
                plain.token_ids(),
                "fused miss encode must equal uncached encode (add_special_tokens={add_special_tokens})"
            );

            // It must also seed the cache, keyed on the same add_special_tokens flag.
            assert!(
                !cache.is_empty(),
                "miss path must populate boundary entries"
            );
            let (_, offset) = cache
                .longest_prefix_match(input, special_tokens, add_special_tokens)
                .expect("hit after populate");
            assert!(offset > 0);
            assert!(
                cache
                    .longest_prefix_match(input, special_tokens, !add_special_tokens)
                    .is_none(),
                "opposite flag must not collide with populated entries"
            );
        }
    }
}

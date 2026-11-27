//! The block hashes of a request's prompt: [`PromptHashes`] and its seeds.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use kv_index::{
    salt::{request_content_hashes_with_seed, request_text_hashes_with_seed},
    XXH3_SEED,
};

use crate::{
    document::spec::AppliedSpec,
    engine::view::{HashKind, RequestFacts, TEXT_BYTES_PER_TOKEN},
};

/// The two seeds of a prompt's hashes.
///
/// A request with no cache partition uses [`HashSeeds::PLAIN`]. A caller
/// that partitions its workers' caches (by a salt, an adapter or a key of
/// its own) derives both seeds from the request's partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashSeeds {
    /// The seed the token blocks are hashed under. It must be the seed the
    /// workers hash the blocks of this partition under, so the index of
    /// their cache events finds the request's blocks.
    pub event: u64,
    /// The partition's whole identity. It is stated as
    /// [`RequestFacts::namespace_seed`], so a record of sent prefixes and a
    /// hash ring keep two partitions apart. A text is hashed under it too:
    /// no worker hashes text, so this seed need match no other.
    pub partition: u64,
}

impl HashSeeds {
    /// The seeds of a request with no cache partition.
    pub const PLAIN: Self = Self {
        event: XXH3_SEED,
        partition: XXH3_SEED,
    };
}

/// The block hashes of one request's prompt, with what a decision reads
/// beside them. One value serves every decision of the request.
///
/// A stage that reads a cache reads the prompt as block hashes
/// ([`RequestFacts::block_hashes`]): one hash for each block of the prompt's
/// token ids, or for each window of its text where the caller holds no token
/// ids. One entry computes them: [`PromptHashes::under_document`] hashes a
/// prompt as a policy document says. Token ids win over a text. A block is
/// the model's block size of record, and the document's where the caller
/// has none. The document bounds how many blocks are hashed
/// ([`AppliedSpec::block_hashes_read`]).
///
/// The hashing itself takes a block size, a depth and the seeds and reads
/// no document. It is private: a window of text is sized by a rule of this
/// crate, which a caller with no document would have to repeat.
///
/// The caller names the seeds ([`HashSeeds`]). What separates one request's
/// cache from another's is the caller's own rule.
#[derive(Debug, Default)]
pub struct PromptHashes {
    blocks: Vec<u64>,
    /// What the blocks hash: token blocks, or windows of the text.
    kind: HashKind,
    /// The size of one block: in tokens, or in bytes for a window.
    block_size: u32,
    namespace_seed: u64,
    prompt_tokens: u32,
    /// The length of a prompt of text in characters, where the document
    /// reads a request's size: `0` otherwise, and for token ids.
    prompt_chars: u32,
    /// The length of a prompt of text in bytes: `0` for token ids.
    prompt_bytes: u32,
}

impl PromptHashes {
    /// The hashes of a prompt's token ids: one for each whole block of
    /// `block_size` tokens, `depth` blocks at most, under
    /// [`HashSeeds::event`]. A prompt that holds no whole block is one
    /// block.
    fn of_tokens(tokens: &[u32], block_size: u32, depth: usize, seeds: HashSeeds) -> Self {
        let size = block_size as usize;
        let end = tokens.len().min(depth.saturating_mul(size));
        // A prompt under one block is one block of what it has: the
        // classic's prefix policy keys a short prompt too.
        let blocks = request_content_hashes_with_seed(&tokens[..end], size.min(end), seeds.event)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        Self {
            blocks,
            kind: HashKind::Token,
            block_size,
            namespace_seed: seeds.partition,
            prompt_tokens: prompt_tokens_of(Some(tokens)),
            prompt_chars: 0,
            prompt_bytes: 0,
        }
    }

    /// The hashes of a prompt its caller holds as text: one for each whole
    /// window of `window` bytes, `depth` windows at most. The text is hashed
    /// as it stands, under [`HashSeeds::partition`]. No worker hashes text,
    /// so the event seed is not read. A prompt of text states no count of
    /// tokens. Its length in bytes is at hand and is always stated. Its
    /// length in characters is counted where `sized` says a stage of the
    /// document reads a request's size: the one pass over the text is made
    /// for such a document alone.
    fn of_text(text: &str, window: u32, depth: usize, sized: bool, seeds: HashSeeds) -> Self {
        let bytes = text.as_bytes();
        let size = window as usize;
        let end = bytes.len().min(depth.saturating_mul(size));
        let blocks = request_text_hashes_with_seed(&bytes[..end], size.min(end), seeds.partition)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        Self {
            blocks,
            kind: HashKind::Text,
            block_size: window,
            namespace_seed: seeds.partition,
            prompt_tokens: 0,
            prompt_chars: prompt_chars_of(None, Some(text), sized),
            prompt_bytes: prompt_bytes_of(None, Some(text)),
        }
    }

    /// The hashes of a prompt under the policy document `spec`: those of its
    /// token ids where the caller holds them, else those of its text, else
    /// none.
    ///
    /// Tokens are hashed by the model's block size. That is
    /// `block_size_of_record`, the size the model's workers reported, and
    /// the document's where the caller knows none, so a model whose workers
    /// publish no cache events is hashed too. A window of text is that size
    /// times [`TEXT_BYTES_PER_TOKEN`], so one number a model sizes both
    /// kinds. The document bounds the number of blocks hashed
    /// ([`AppliedSpec::block_hashes_read`]).
    pub fn under_document(
        spec: &AppliedSpec,
        block_size_of_record: Option<u32>,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> Self {
        Self::of_prompt(tokens, text, seeds, |kind| {
            block_size_and_depth(spec, block_size_of_record, kind)
        })
    }

    /// What [`Self::under_document`] states of a prompt's token ids beside
    /// its list, with no block hashed: for a request whose list is hashed
    /// where it is read ([`Self::whole`]).
    #[cfg(feature = "in-process")]
    pub(crate) fn unhashed(
        spec: &AppliedSpec,
        block_size_of_record: Option<u32>,
        tokens: &[u32],
        seeds: HashSeeds,
    ) -> Self {
        let (block_size, _, _) = block_size_and_depth(spec, block_size_of_record, HashKind::Token);
        Self::of_tokens(tokens, block_size, 0, seeds)
    }

    /// The hashes [`Self::under_document`] makes of `tokens`, the prompt
    /// this value was made of with no block hashed ([`Self::unhashed`]): by
    /// the block size it was made with, as deep as `spec` reads. `None`
    /// where `spec` reads no block: this value holds all a decision reads.
    #[cfg(feature = "in-process")]
    pub(crate) fn whole(
        &self,
        spec: &AppliedSpec,
        tokens: &[u32],
        seeds: HashSeeds,
    ) -> Option<Self> {
        let depth = spec.block_hashes_read(HashKind::Token, self.block_size);
        (depth > 0).then(|| Self::of_tokens(tokens, self.block_size, depth, seeds))
    }

    /// The hashes of a prompt by what its caller holds of it: its token ids
    /// win, else its text is hashed, else it has none. `read` is asked once,
    /// for the kind that is hashed: the size of one block of that kind and
    /// how many blocks are read, and whether the document reads a request's
    /// size ([`block_size_and_depth`]). A prompt with neither asks nothing,
    /// and keeps the seed of its partition.
    fn of_prompt(
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
        read: impl FnOnce(HashKind) -> (u32, usize, bool),
    ) -> Self {
        match (tokens, text) {
            (Some(tokens), _) => {
                let (block_size, depth, _) = read(HashKind::Token);
                Self::of_tokens(tokens, block_size, depth, seeds)
            }
            (None, Some(text)) => {
                let (window, depth, sized) = read(HashKind::Text);
                Self::of_text(text, window, depth, sized, seeds)
            }
            (None, None) => Self {
                namespace_seed: seeds.partition,
                ..Self::default()
            },
        }
    }

    /// Writes the hashes into a request's facts: the blocks, what they
    /// hash, the size of one and the partition's seed. The facts borrow the
    /// blocks. A prompt size the facts already hold stays: the hashing's own
    /// count is written only where they hold none, and so is each length of
    /// its text.
    ///
    /// Every decision of the request calls this from the caller's crate, so
    /// it is `#[inline]`: its body can be inlined across the crate boundary.
    #[inline]
    pub fn write_to<'a>(&'a self, facts: &mut RequestFacts<'a>) {
        facts.block_hashes = &self.blocks;
        facts.hash_kind = self.kind;
        facts.block_size = self.block_size;
        facts.namespace_seed = self.namespace_seed;
        facts.prompt_tokens = self.prompt_tokens_or(facts.prompt_tokens);
        if facts.prompt_chars == 0 {
            facts.prompt_chars = self.prompt_chars;
        }
        if facts.prompt_bytes == 0 {
            facts.prompt_bytes = self.prompt_bytes;
        }
    }

    /// A copy of `facts` with these hashes written over it
    /// ([`Self::write_to`]): what a decision reads of a request whose
    /// making left its list out ([`Self::unhashed`]), where it reads the
    /// list.
    #[cfg(feature = "in-process")]
    pub(crate) fn over<'a>(&'a self, facts: &RequestFacts<'a>) -> RequestFacts<'a> {
        let mut whole = RequestFacts { ..*facts };
        self.write_to(&mut whole);
        whole
    }

    /// The size in tokens of a prompt whose request states `stated`: the
    /// stated size, and the hashing's own count where it states none (`0`).
    /// What [`Self::write_to`] leaves in a request's facts, for a caller
    /// that states the size with no request made.
    #[inline]
    #[must_use]
    pub fn prompt_tokens_or(&self, stated: u32) -> u32 {
        if stated == 0 {
            self.prompt_tokens
        } else {
            stated
        }
    }
}

/// The size of one block of a model's prompt under `spec`, how many blocks
/// of it the document reads, and whether a stage of the document reads a
/// request's size ([`Needs::prompt_size`](crate::document::Needs::prompt_size)).
/// A token block is the model's block size: its size of record, and the
/// document's where it has none. A window of text is that many tokens in
/// bytes.
fn block_size_and_depth(
    spec: &AppliedSpec,
    block_size_of_record: Option<u32>,
    kind: HashKind,
) -> (u32, usize, bool) {
    let block_size = block_size_of_record.unwrap_or_else(|| spec.block_size());
    let size = match kind {
        HashKind::Token => block_size,
        HashKind::Text => block_size.saturating_mul(TEXT_BYTES_PER_TOKEN),
    };
    let sized = spec.needs().prompt_size;
    (size, spec.block_hashes_read(kind, size), sized)
}

fn prompt_token_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// The count of the token ids of a prompt its caller holds as `tokens`,
/// with no hash made: zero for a prompt it holds as text, or not at all.
/// [`PromptHashes`] keeps this count of it.
pub(crate) fn prompt_tokens_of(tokens: Option<&[u32]>) -> u32 {
    tokens.map_or(0, |tokens| prompt_token_count(tokens.len()))
}

/// The length in characters of a prompt its caller holds as `text` and not
/// as `tokens`, with no hash made, where `sized` says a stage of the
/// document reads a request's size; zero otherwise. Token ids win, as they
/// do where the prompt is hashed, and the text is passed over for such a
/// document alone. [`PromptHashes`] keeps this length of it.
pub(crate) fn prompt_chars_of(tokens: Option<&[u32]>, text: Option<&str>, sized: bool) -> u32 {
    match (tokens, text) {
        (None, Some(text)) if sized => prompt_token_count(text.chars().count()),
        _ => 0,
    }
}

/// The length in bytes of a prompt its caller holds as `text` and not as
/// `tokens`, with no hash made and no pass over the text; zero otherwise.
/// Token ids win, as they do where the prompt is hashed.
/// [`PromptHashes`] keeps this length of it.
pub(crate) fn prompt_bytes_of(tokens: Option<&[u32]>, text: Option<&str>) -> u32 {
    match (tokens, text) {
        (None, Some(text)) => prompt_token_count(text.len()),
        _ => 0,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/prompt_hashes/tests.rs"]
mod tests;

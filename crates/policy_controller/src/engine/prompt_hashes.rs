//! The block hashes of a request's prompt: [`PromptHashes`] and its seeds.

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
/// ids. Three entries compute them, each over the one before it:
///
/// - [`PromptHashes::of_tokens`] and [`PromptHashes::of_text`] hash by the
///   block size, the depth and the seeds they are handed. They read no
///   document and no store.
/// - [`PromptHashes::under_document`] hashes a prompt as a policy document
///   says. Token ids win over a text. A block is the model's block size of
///   record, and the document's where the caller has none. The document
///   bounds how many blocks are hashed ([`AppliedSpec::block_hashes_read`]).
/// - [`MemoryStore::prompt_hashes`](crate::store::MemoryStore::prompt_hashes)
///   does the same for a caller that decides over a store. The block size
///   of record and the document in force are the store's.
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
}

impl PromptHashes {
    /// The hashes of a prompt's token ids: one for each whole block of
    /// `block_size` tokens, `depth` blocks at most, under
    /// [`HashSeeds::event`]. A prompt that holds no whole block is one
    /// block.
    pub fn of_tokens(tokens: &[u32], block_size: u32, depth: usize, seeds: HashSeeds) -> Self {
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
            prompt_tokens: prompt_token_count(tokens.len()),
        }
    }

    /// The hashes of a prompt its caller holds as text: one for each whole
    /// window of `window` bytes, `depth` windows at most. The text is hashed
    /// as it stands, under [`HashSeeds::partition`]. No worker hashes text,
    /// so the event seed is not read.
    pub fn of_text(text: &str, window: u32, depth: usize, seeds: HashSeeds) -> Self {
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
            prompt_tokens: prompt_token_count(bytes.len() / TEXT_BYTES_PER_TOKEN as usize),
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

    /// The hashes of a prompt by what its caller holds of it: its token ids
    /// win, else its text is hashed, else it has none. `read` is asked once,
    /// for the kind that is hashed: the size of one block of that kind and
    /// how many blocks are read ([`block_size_and_depth`]). A prompt with
    /// neither asks nothing.
    pub(crate) fn of_prompt(
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
        read: impl FnOnce(HashKind) -> (u32, usize),
    ) -> Self {
        match (tokens, text) {
            (Some(tokens), _) => {
                let (block_size, depth) = read(HashKind::Token);
                Self::of_tokens(tokens, block_size, depth, seeds)
            }
            (None, Some(text)) => {
                let (window, depth) = read(HashKind::Text);
                Self::of_text(text, window, depth, seeds)
            }
            (None, None) => Self::default(),
        }
    }

    /// Writes the hashes into a request's facts: the blocks, what they
    /// hash, the size of one and the partition's seed. The facts borrow the
    /// blocks. A prompt size the facts already hold stays: the hashing's own
    /// count is written only where they hold none.
    ///
    /// Every decision of the request calls this from the caller's crate, so
    /// it is `#[inline]`: its body can be inlined across the crate boundary.
    #[inline]
    pub fn write_to<'a>(&'a self, facts: &mut RequestFacts<'a>) {
        facts.block_hashes = &self.blocks;
        facts.hash_kind = self.kind;
        facts.block_size = self.block_size;
        facts.namespace_seed = self.namespace_seed;
        if facts.prompt_tokens == 0 {
            facts.prompt_tokens = self.prompt_tokens;
        }
    }
}

/// The size of one block of a model's prompt under `spec`, and how many
/// blocks of it the document reads. A token block is the model's block size:
/// its size of record, and the document's where it has none. A window of
/// text is that many tokens in bytes.
pub(crate) fn block_size_and_depth(
    spec: &AppliedSpec,
    block_size_of_record: Option<u32>,
    kind: HashKind,
) -> (u32, usize) {
    let block_size = block_size_of_record.unwrap_or_else(|| spec.block_size());
    let size = match kind {
        HashKind::Token => block_size,
        HashKind::Text => block_size.saturating_mul(TEXT_BYTES_PER_TOKEN),
    };
    (size, spec.block_hashes_read(kind, size))
}

fn prompt_token_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use kv_index::{compute_request_content_hashes, salt::namespace_seed, XXH3_SEED};

    use super::{prompt_token_count, HashSeeds, PromptHashes};
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::view::{HashKind, RequestFacts, TEXT_BYTES_PER_TOKEN},
        store::{KvEventSink, MemoryStore},
    };

    const MODEL: &str = "model-a";
    const BLOCK_SIZE: u32 = 16;
    /// A text of two windows and a tail, at a block of sixteen tokens.
    const RECORDED_TEXT: &str = "A request's text is hashed by windows of its bytes, as deep as \
                                 the generation reads, and a tail shorter than one window gives \
                                 no hash at all.";
    const RECORDED_TOKENS: [u64; 2] = [0xea06_fc71_a5ee_6e88, 0x1fbf_39bd_106b_e0b5];
    const RECORDED_SCOPED_TOKENS: [u64; 2] = [0x3c6f_83b1_c88e_dc76, 0xcbfd_d2b6_595c_c9c6];
    const RECORDED_WINDOWS: [u64; 2] = [0xc67b_5989_9662_3ba3, 0x627b_9253_c3e5_d0f8];

    /// The cache-aware preset's document: it reads token blocks and text
    /// windows.
    fn document() -> AppliedSpec {
        let document = PolicySpec::preset("cache_aware").expect("the preset");
        AppliedSpec::validate(&document).expect("the preset validates")
    }

    /// A store under that document.
    fn store() -> MemoryStore {
        let store = MemoryStore::new();
        store.apply(document()).expect("the store takes the preset");
        store
    }

    /// The facts `hashes` writes into a request that holds none.
    fn written(hashes: &PromptHashes) -> RequestFacts<'_> {
        let mut facts = RequestFacts::default();
        hashes.write_to(&mut facts);
        facts
    }

    /// The five facts `hashes` writes, as one value to compare.
    fn five(hashes: &PromptHashes) -> (Vec<u64>, HashKind, u32, u64, u32) {
        let facts = written(hashes);
        (
            facts.block_hashes.to_vec(),
            facts.hash_kind,
            facts.block_size,
            facts.namespace_seed,
            facts.prompt_tokens,
        )
    }

    /// The block hashes of three prompts, as values of record: a plain
    /// token prompt, the same tokens under a partition's event seed, and a
    /// text prompt. The gateway's case reads the same values through its
    /// own cache partition.
    #[test]
    fn a_prompts_hashes_are_the_recorded_ones() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=40).collect();
        let plain = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        assert_eq!(written(&plain).block_hashes, RECORDED_TOKENS);
        let seeds = HashSeeds {
            event: namespace_seed(None, Some("partition")),
            partition: 7,
        };
        let scoped = store.prompt_hashes(MODEL, Some(&tokens), None, seeds);
        let facts = written(&scoped);
        assert_eq!(facts.block_hashes, RECORDED_SCOPED_TOKENS);
        assert_eq!(facts.namespace_seed, 7);
        let windows = store.prompt_hashes(MODEL, None, Some(RECORDED_TEXT), HashSeeds::PLAIN);
        let facts = written(&windows);
        assert_eq!(facts.block_hashes, RECORDED_WINDOWS);
        assert_eq!(facts.prompt_tokens, RECORDED_TEXT.len() as u32 / 4);
    }

    /// A model whose workers state no block size is hashed by the
    /// document's. A size of record wins once there is one, and one of
    /// zero states no hash.
    #[test]
    fn a_model_with_no_block_size_of_record_is_hashed_by_the_documents() {
        let store = store();
        let size = store.deciding(|world| world.spec().block_size());
        let tokens: Vec<u32> = (1..=2 * size).collect();
        for (recorded, blocks, block_size) in [
            (None, 2, size),
            (Some(0), 0, 0),
            (Some(2 * size), 1, 2 * size),
        ] {
            if let Some(size) = recorded {
                store.set_block_size(MODEL, size);
            }
            assert_eq!(store.block_size(MODEL), recorded);
            let hashes = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
            let facts = written(&hashes);
            assert_eq!(
                (facts.block_hashes.len(), facts.block_size),
                (blocks, block_size),
                "{recorded:?}"
            );
            assert_eq!(facts.prompt_tokens, tokens.len() as u32);
            assert_eq!(facts.namespace_seed, XXH3_SEED);
        }
    }

    #[test]
    fn the_plain_seeds_give_the_index_crates_plain_hashes() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=2 * BLOCK_SIZE).collect();
        let hashes = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let expected: Vec<_> = compute_request_content_hashes(&tokens, BLOCK_SIZE as usize)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let facts = written(&hashes);
        assert_eq!(facts.block_hashes, expected);
        assert_eq!(facts.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn a_tail_shorter_than_a_block_gives_no_hash() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens = [1; BLOCK_SIZE as usize + 1];
        let hashes = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let facts = written(&hashes);
        assert_eq!(facts.block_hashes.len(), 1);
        assert_eq!(facts.prompt_tokens, BLOCK_SIZE + 1);
    }

    /// A prompt that holds no whole block states one block, the hash of
    /// what it has, as the classic's prefix policy keys a short prompt. Two
    /// short prompts are two blocks, and an empty prompt states none.
    #[test]
    fn a_prompt_under_one_block_states_one_block_of_what_it_has() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens =
            |tokens: &[u32]| store.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let text = |text: &str| store.prompt_hashes(MODEL, None, Some(text), HashSeeds::PLAIN);
        let (sevens, eights) = ([7; BLOCK_SIZE as usize - 1], [8; 3]);
        for (short, again, other, block_size, prompt_tokens) in [
            (
                tokens(&sevens),
                tokens(&sevens),
                tokens(&eights),
                BLOCK_SIZE,
                BLOCK_SIZE - 1,
            ),
            (
                text("a prompt under one window"),
                text("a prompt under one window"),
                text("another"),
                BLOCK_SIZE * TEXT_BYTES_PER_TOKEN,
                6,
            ),
        ] {
            let facts = written(&short);
            assert_eq!(facts.block_hashes.len(), 1, "{short:?}");
            assert_eq!(facts.block_hashes, written(&again).block_hashes);
            assert_ne!(facts.block_hashes, written(&other).block_hashes);
            assert_eq!(
                (facts.block_size, facts.prompt_tokens),
                (block_size, prompt_tokens)
            );
        }
        for empty in [tokens(&[]), text("")] {
            assert!(written(&empty).block_hashes.is_empty(), "{empty:?}");
        }
    }

    /// The hashes write their five facts into a request that holds no
    /// prompt size. A size the request already holds stays.
    #[test]
    fn a_prompt_size_the_facts_hold_wins_over_the_hashings_count() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let seeds = HashSeeds {
            event: XXH3_SEED,
            partition: 9,
        };
        let hashes = store.prompt_hashes(MODEL, Some(&[3; 88]), None, seeds);
        let facts = written(&hashes);
        assert_eq!(
            (
                facts.hash_kind,
                facts.block_size,
                facts.namespace_seed,
                facts.prompt_tokens
            ),
            (HashKind::Token, BLOCK_SIZE, 9, 88)
        );
        assert!(!facts.block_hashes.is_empty());
        let mut sized = RequestFacts {
            prompt_tokens: 37,
            ..RequestFacts::default()
        };
        hashes.write_to(&mut sized);
        assert_eq!(sized.prompt_tokens, 37);
        assert_eq!(sized.block_hashes, facts.block_hashes);
    }

    /// No block is hashed past the depth the document reads, of token
    /// blocks and of text windows. A document that reads no hashes gets
    /// none, and the model's block size is still written.
    #[test]
    fn no_block_is_hashed_past_the_documents_depth() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
        let (blocks, windows) = store.deciding(|world| {
            let spec = world.spec();
            (
                spec.block_hashes_read(HashKind::Token, BLOCK_SIZE),
                spec.block_hashes_read(HashKind::Text, window),
            )
        });
        assert!(blocks > 0 && windows > 0);
        let tokens: Vec<u32> = (0..(blocks as u32 + 1) * BLOCK_SIZE).collect();
        let hashes = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let facts = written(&hashes);
        assert_eq!(facts.block_hashes.len(), blocks);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        let text = "a".repeat((windows + 1) * window as usize);
        let hashes = store.prompt_hashes(MODEL, None, Some(&text), HashSeeds::PLAIN);
        assert_eq!(written(&hashes).block_hashes.len(), windows);

        let mut document = PolicySpec::preset("random").expect("the preset");
        document.generation += 1;
        store
            .apply(AppliedSpec::validate(&document).expect("the generation"))
            .expect("the newer generation");
        let hashes = store.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let facts = written(&hashes);
        assert!(facts.block_hashes.is_empty());
        assert_eq!(facts.block_size, BLOCK_SIZE);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
    }

    /// A text writes its kind, the size of one window and the partition's
    /// seed, which it is hashed under. Another partition moves every window.
    /// The event seed moves none: no worker hashes text.
    #[test]
    fn a_text_is_hashed_under_the_partition_seed_alone() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
        let text = "a".repeat(2 * window as usize);
        let windows = |event: u64, partition: u64| {
            store.prompt_hashes(MODEL, None, Some(&text), HashSeeds { event, partition })
        };
        let keyed = windows(5, 11);
        let facts = written(&keyed);
        assert_eq!(
            (facts.hash_kind, facts.block_size, facts.namespace_seed),
            (HashKind::Text, window, 11)
        );
        assert_eq!(facts.block_hashes.len(), 2);
        assert_eq!(written(&windows(6, 11)).block_hashes, facts.block_hashes);
        let other = windows(5, 12);
        assert!(facts
            .block_hashes
            .iter()
            .zip(written(&other).block_hashes)
            .all(|(keyed, other)| keyed != other));
    }

    /// The hashing itself reads no document and no store: it hashes by the
    /// block size and the depth it is handed. The values are the recorded
    /// ones.
    #[test]
    fn the_hashing_alone_reads_the_block_size_and_the_depth_it_is_handed() {
        let tokens: Vec<u32> = (1..=40).collect();
        let whole = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, usize::MAX, HashSeeds::PLAIN);
        assert_eq!(written(&whole).block_hashes, RECORDED_TOKENS);
        let first = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, 1, HashSeeds::PLAIN);
        assert_eq!(written(&first).block_hashes, &RECORDED_TOKENS[..1]);
        let none = PromptHashes::of_tokens(&tokens, BLOCK_SIZE, 0, HashSeeds::PLAIN);
        assert_eq!(
            five(&none),
            (Vec::new(), HashKind::Token, BLOCK_SIZE, XXH3_SEED, 40)
        );
        let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
        let whole = PromptHashes::of_text(RECORDED_TEXT, window, usize::MAX, HashSeeds::PLAIN);
        assert_eq!(written(&whole).block_hashes, RECORDED_WINDOWS);
        let first = PromptHashes::of_text(RECORDED_TEXT, window, 1, HashSeeds::PLAIN);
        assert_eq!(written(&first).block_hashes, &RECORDED_WINDOWS[..1]);
    }

    /// A caller that holds a document and no store hashes a prompt as a
    /// store under that document does: by the block size it has on record
    /// for the model, and by the document's where it has none.
    #[test]
    fn a_document_alone_hashes_a_prompt_as_a_store_under_it_does() {
        let document = document();
        let tokens: Vec<u32> = (1..=200).collect();
        let text = "a".repeat(1_000);
        let seeds = HashSeeds {
            event: 3,
            partition: 7,
        };
        for of_record in [None, Some(BLOCK_SIZE)] {
            let store = store();
            if let Some(size) = of_record {
                store.set_block_size(MODEL, size);
            }
            for (tokens, text) in [(Some(tokens.as_slice()), None), (None, Some(text.as_str()))] {
                let alone = PromptHashes::under_document(&document, of_record, tokens, text, seeds);
                let stored = store.prompt_hashes(MODEL, tokens, text, seeds);
                assert_eq!(five(&alone), five(&stored), "{of_record:?}");
                assert!(!written(&alone).block_hashes.is_empty(), "{of_record:?}");
            }
        }
    }

    /// Token ids win over a text, a text is hashed where there are none, and
    /// a prompt with neither has no hashes and asks for no block size.
    #[test]
    fn token_ids_win_over_a_text_and_a_prompt_with_neither_has_no_hashes() {
        let document = document();
        let tokens: Vec<u32> = (1..=40).collect();
        let under = |tokens: Option<&[u32]>, text: Option<&str>| {
            let seeds = HashSeeds::PLAIN;
            PromptHashes::under_document(&document, Some(BLOCK_SIZE), tokens, text, seeds)
        };
        let both = under(Some(&tokens), Some(RECORDED_TEXT));
        assert_eq!(five(&both), five(&under(Some(&tokens), None)));
        assert_eq!(written(&both).block_hashes, RECORDED_TOKENS);
        let text = under(None, Some(RECORDED_TEXT));
        assert_eq!(written(&text).hash_kind, HashKind::Text);
        assert_eq!(written(&text).block_hashes, RECORDED_WINDOWS);
        assert_eq!(five(&under(None, None)), five(&PromptHashes::default()));
        let neither = PromptHashes::of_prompt(None, None, HashSeeds::PLAIN, |kind| {
            panic!("a prompt with neither asked for the size of a {kind:?} block")
        });
        assert_eq!(five(&neither), five(&PromptHashes::default()));
    }

    #[test]
    fn the_prompt_token_count_saturates_at_u32_max() {
        let past_max = (u32::MAX as usize).saturating_add(1);
        assert_eq!(prompt_token_count(past_max), u32::MAX);
    }
}

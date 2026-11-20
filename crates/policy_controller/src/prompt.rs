//! The block hashes a host states of a request's prompt.
//!
//! A stage that reads a cache reads a request's prompt as block hashes
//! ([`RequestFacts::block_hashes`]): hashes of the prompt's token blocks,
//! or of windows of its text where the host holds no token ids. A block is
//! the model's block size, and the generation in force bounds how many are
//! hashed. The computation is one type, [`Hashes`], for a host in process
//! and for a gateway that asks a controller under an address. The host
//! names the seeds ([`Seeds`]): what partitions a request's cache is its
//! own rule.
//!
//! [`RequestFacts::block_hashes`]: crate::RequestFacts::block_hashes

use kv_index::salt::{request_content_hashes_with_seed, request_text_hashes_with_seed};

use crate::{
    spec::AppliedSpec,
    store::{KvEventSink, MemoryStore},
    view::{HashKind, RequestFacts, TEXT_BYTES_PER_TOKEN},
};

/// The two seeds of a token prompt, each named for what it keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seeds {
    /// The seed the engines hash the blocks of the request's partition
    /// under, so the index of cache events finds the request's blocks. That
    /// seed folds no extra key.
    pub event: u64,
    /// The partition's whole identity. A prefix is named under it: the
    /// record of sent prefixes and the ring keep two partitions apart that
    /// differ by the extra key alone, as an engine's cache does.
    pub partition: u64,
}

/// A request's block hashes as the door reads them. The attempt owns the
/// buffer, and its asks share it.
#[derive(Debug, Default)]
pub struct Hashes {
    blocks: Vec<u64>,
    /// What the blocks hash: token blocks, or windows of the text.
    kind: HashKind,
    /// The size of one block: in tokens, or in bytes for a window.
    block_size: u32,
    namespace_seed: u64,
    prompt_tokens: u32,
}

impl Hashes {
    /// The hashes of a prompt's token ids for `model`, under the generation
    /// in force.
    ///
    /// Prompt tokens are hashed by the model's block size: its size of
    /// record, and the document's where it has none. So a model whose
    /// workers publish no cache events is hashed too. The generation bounds
    /// the number of full blocks hashed. A prompt that holds no whole block
    /// is one block.
    pub fn of_tokens(store: &MemoryStore, model: &str, tokens: &[u32], seeds: Seeds) -> Self {
        let (block_size, depth) = store.deciding(|world| {
            let spec = world.spec();
            let block_size = block_size_of(store, spec, model);
            let depth = spec.block_hashes_read(HashKind::Token, block_size);
            (block_size, depth)
        });
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

    /// The hashes of a prompt its router holds as text: one a window of the
    /// text's bytes, as deep as the generation reads. A window is the
    /// model's block size times [`TEXT_BYTES_PER_TOKEN`], so one number a
    /// model sizes both kinds. The text is hashed as it stands: its whole
    /// partition is in the seed ([`Seeds::partition`]), and no marker
    /// stands before it. No engine hashes text, so that seed need match
    /// none.
    pub fn of_text(store: &MemoryStore, model: &str, text: &str, partition_seed: u64) -> Self {
        let bytes = text.as_bytes();
        let (window, depth) = store.deciding(|world| {
            let spec = world.spec();
            let window = block_size_of(store, spec, model).saturating_mul(TEXT_BYTES_PER_TOKEN);
            (window, spec.block_hashes_read(HashKind::Text, window))
        });
        let size = window as usize;
        let end = bytes.len().min(depth.saturating_mul(size));
        let blocks = request_text_hashes_with_seed(&bytes[..end], size.min(end), partition_seed)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        Self {
            blocks,
            kind: HashKind::Text,
            block_size: window,
            namespace_seed: partition_seed,
            prompt_tokens: prompt_token_count(bytes.len() / TEXT_BYTES_PER_TOKEN as usize),
        }
    }

    /// State the hashes in a request's facts: the blocks, what they hash,
    /// the size of one and the partition's seed. The facts borrow the
    /// blocks. A prompt size the request already states wins over the
    /// count of the hashing.
    pub fn state<'a>(&'a self, facts: &mut RequestFacts<'a>) {
        facts.block_hashes = &self.blocks;
        facts.hash_kind = self.kind;
        facts.block_size = self.block_size;
        facts.namespace_seed = self.namespace_seed;
        if facts.prompt_tokens == 0 {
            facts.prompt_tokens = self.prompt_tokens;
        }
    }
}

/// The model's block size: its size of record, and the document's where it
/// has none.
pub(crate) fn block_size_of(store: &MemoryStore, spec: &AppliedSpec, model: &str) -> u32 {
    store.block_size(model).unwrap_or_else(|| spec.block_size())
}

fn prompt_token_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use kv_index::{compute_request_content_hashes, salt::namespace_seed, XXH3_SEED};

    use super::{prompt_token_count, Hashes, Seeds};
    use crate::{
        spec::{AppliedSpec, PolicySpec},
        store::{KvEventSink, MemoryStore},
        view::{HashKind, RequestFacts, TEXT_BYTES_PER_TOKEN},
    };

    const MODEL: &str = "model-a";
    const BLOCK_SIZE: u32 = 16;
    /// The seeds of a request with no partition.
    const PLAIN: Seeds = Seeds {
        event: XXH3_SEED,
        partition: XXH3_SEED,
    };
    /// A text of two windows and a tail, at a block of sixteen tokens.
    const RECORDED_TEXT: &str = "A request's text is hashed by windows of its bytes, as deep as \
                                 the generation reads, and a tail shorter than one window gives \
                                 no hash at all.";
    const RECORDED_TOKENS: [u64; 2] = [0xea06_fc71_a5ee_6e88, 0x1fbf_39bd_106b_e0b5];
    const RECORDED_SCOPED_TOKENS: [u64; 2] = [0x3c6f_83b1_c88e_dc76, 0xcbfd_d2b6_595c_c9c6];
    const RECORDED_WINDOWS: [u64; 2] = [0xc67b_5989_9662_3ba3, 0x627b_9253_c3e5_d0f8];

    /// A store under the cache-aware preset: its generation reads token
    /// blocks and text windows.
    fn store() -> MemoryStore {
        let store = MemoryStore::new();
        let document = PolicySpec::preset("cache_aware").expect("the preset");
        let applied = AppliedSpec::validate(&document).expect("the preset validates");
        store.apply(applied).expect("the store takes the preset");
        store
    }

    /// The facts `hashes` states of a request that states none.
    fn stated(hashes: &Hashes) -> RequestFacts<'_> {
        let mut facts = RequestFacts::default();
        hashes.state(&mut facts);
        facts
    }

    /// The block hashes of three prompts, as values of record: a plain
    /// token prompt, the same tokens under a partition's event seed, and a
    /// text prompt. The gateway's case reads the same values through its
    /// own namespace.
    #[test]
    fn a_prompts_hashes_are_the_recorded_ones() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=40).collect();
        let plain = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
        assert_eq!(stated(&plain).block_hashes, RECORDED_TOKENS);
        let seeds = Seeds {
            event: namespace_seed(None, Some("partition")),
            partition: 7,
        };
        let scoped = Hashes::of_tokens(&store, MODEL, &tokens, seeds);
        let facts = stated(&scoped);
        assert_eq!(facts.block_hashes, RECORDED_SCOPED_TOKENS);
        assert_eq!(facts.namespace_seed, 7);
        let windows = Hashes::of_text(&store, MODEL, RECORDED_TEXT, XXH3_SEED);
        let facts = stated(&windows);
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
            let hashes = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
            let facts = stated(&hashes);
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
    fn the_plain_seed_gives_the_index_crates_plain_hashes() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=2 * BLOCK_SIZE).collect();
        let hashes = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
        let expected: Vec<_> = compute_request_content_hashes(&tokens, BLOCK_SIZE as usize)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let facts = stated(&hashes);
        assert_eq!(facts.block_hashes, expected);
        assert_eq!(facts.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn a_tail_shorter_than_a_block_gives_no_hash() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens = [1; BLOCK_SIZE as usize + 1];
        let hashes = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
        let facts = stated(&hashes);
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
        let tokens = |tokens: &[u32]| Hashes::of_tokens(&store, MODEL, tokens, PLAIN);
        let text = |text: &str| Hashes::of_text(&store, MODEL, text, XXH3_SEED);
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
            let facts = stated(&short);
            assert_eq!(facts.block_hashes.len(), 1, "{short:?}");
            assert_eq!(facts.block_hashes, stated(&again).block_hashes);
            assert_ne!(facts.block_hashes, stated(&other).block_hashes);
            assert_eq!(
                (facts.block_size, facts.prompt_tokens),
                (block_size, prompt_tokens)
            );
        }
        for empty in [tokens(&[]), text("")] {
            assert!(stated(&empty).block_hashes.is_empty(), "{empty:?}");
        }
    }

    /// The hashes state their five facts in a request that states no prompt
    /// size. A size the request already states stands.
    #[test]
    fn a_stated_prompt_size_wins_over_the_hashes_count() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let seeds = Seeds {
            event: XXH3_SEED,
            partition: 9,
        };
        let hashes = Hashes::of_tokens(&store, MODEL, &[3; 88], seeds);
        let facts = stated(&hashes);
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
        hashes.state(&mut sized);
        assert_eq!(sized.prompt_tokens, 37);
        assert_eq!(sized.block_hashes, facts.block_hashes);
    }

    /// No block is hashed past the depth the generation reads, of token
    /// blocks and of text windows. A generation that reads no hashes states
    /// none, and still states the model's block size.
    #[test]
    fn no_block_is_hashed_past_the_generations_depth() {
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
        let hashes = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
        let facts = stated(&hashes);
        assert_eq!(facts.block_hashes.len(), blocks);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        let text = "a".repeat((windows + 1) * window as usize);
        let hashes = Hashes::of_text(&store, MODEL, &text, XXH3_SEED);
        assert_eq!(stated(&hashes).block_hashes.len(), windows);

        let mut document = PolicySpec::preset("random").expect("the preset");
        document.generation += 1;
        store
            .apply(AppliedSpec::validate(&document).expect("the generation"))
            .expect("the newer generation");
        let hashes = Hashes::of_tokens(&store, MODEL, &tokens, PLAIN);
        let facts = stated(&hashes);
        assert!(facts.block_hashes.is_empty());
        assert_eq!(facts.block_size, BLOCK_SIZE);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
    }

    /// A text states its kind, the size of one window and the seed it was
    /// hashed under. Another seed moves every window.
    #[test]
    fn a_text_states_its_kind_its_window_and_its_seed() {
        let store = store();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let window = BLOCK_SIZE * TEXT_BYTES_PER_TOKEN;
        let text = "a".repeat(2 * window as usize);
        let keyed = Hashes::of_text(&store, MODEL, &text, 11);
        let facts = stated(&keyed);
        assert_eq!(
            (facts.hash_kind, facts.block_size, facts.namespace_seed),
            (HashKind::Text, window, 11)
        );
        assert_eq!(facts.block_hashes.len(), 2);
        let plain = Hashes::of_text(&store, MODEL, &text, XXH3_SEED);
        let plain = stated(&plain);
        assert!(facts
            .block_hashes
            .iter()
            .zip(plain.block_hashes)
            .all(|(keyed, plain)| keyed != plain));
    }

    #[test]
    fn the_prompt_token_count_saturates_at_u32_max() {
        let past_max = (u32::MAX as usize).saturating_add(1);
        assert_eq!(prompt_token_count(past_max), u32::MAX);
    }
}

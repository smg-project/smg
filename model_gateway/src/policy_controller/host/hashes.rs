//! The prompt facts an attempt keeps for its asks to the controller.

use kv_index::{salt::request_content_hashes_with_seed, XXH3_SEED};
use smg_policy_controller::{
    store::{KvEventSink, MemoryStore},
    view::HashKind,
};

use crate::policies::CacheNamespace;

/// What an attempt states of its prompt, for the cache index.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Prompt<'a> {
    /// The prompt's token ids, where the router tokenized it.
    pub tokens: Option<&'a [u32]>,
    /// The request's cache partition, where it has one.
    pub namespace: Option<CacheNamespace>,
}

/// A request's block hashes as the door reads them. The attempt owns the
/// buffer, and its asks share it.
#[derive(Debug, Default)]
pub(crate) struct Hashes {
    pub(super) blocks: Vec<u64>,
    pub(super) block_size: u32,
    pub(super) namespace_seed: u64,
    pub(super) prompt_tokens: u32,
}

impl Hashes {
    /// The hashes of `prompt` for `model`, under the generation in force.
    ///
    /// Hashing prompt tokens needs a positive block size of record. The
    /// index can hold prehashed blocks before a size is recorded.
    /// The generation bounds the number of full blocks hashed.
    pub(crate) fn of(store: &MemoryStore, model: &str, prompt: &Prompt<'_>) -> Self {
        let Some(tokens) = prompt.tokens else {
            return Self::default();
        };
        let tokens = if prompt.namespace.is_some() {
            tokens
        } else {
            CacheNamespace::unpartitioned_tokens(tokens)
        };
        let block_size = store.block_size(model).unwrap_or(0);
        let size = block_size as usize;
        let depth = if block_size == 0 {
            0
        } else {
            store.deciding(|world| world.spec().block_hashes_read(HashKind::Token, block_size))
        };
        let namespace_seed = prompt
            .namespace
            .map_or(XXH3_SEED, CacheNamespace::event_seed);
        let end = tokens.len().min(depth.saturating_mul(size));
        let blocks = request_content_hashes_with_seed(&tokens[..end], size, namespace_seed)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        Self {
            blocks,
            block_size,
            namespace_seed,
            prompt_tokens: prompt_token_count(tokens.len()),
        }
    }
}

fn prompt_token_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use kv_index::{
        compute_request_content_hashes, salt::request_content_hashes_with_seed, XXH3_SEED,
    };
    use openai_protocol::common::CachePartition;
    use smg_policy_controller::{
        store::{Ask as ViewAsk, BatchOutcome, KvEventSink, Named, Naming, ViewBuffers},
        view::{HashKind, RequestFacts, Role},
        AppliedSpec, PolicySpec,
    };

    use super::{prompt_token_count, Hashes};
    use crate::{
        policies::CacheNamespace,
        policy_controller::host::{
            testing::{cache_bed, store_blocks, MODEL},
            tests::first,
            Ask, Prompt, GATEWAY,
        },
    };

    const BLOCK_SIZE: u32 = 16;

    fn plain_prompt(tokens: &[u32]) -> Prompt<'_> {
        Prompt {
            tokens: Some(tokens),
            namespace: None,
        }
    }

    #[test]
    fn a_namespace_hashes_under_its_own_seed() {
        let (_, store, holder) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let namespace = CacheNamespace::derive(&CachePartition {
            cache_salt: Some("partition"),
            ..CachePartition::default()
        })
        .expect("the namespace");
        let tokens: Vec<u32> = namespace
            .token_marker()
            .into_iter()
            .chain(1..=BLOCK_SIZE)
            .collect();
        let plain = Hashes::of(&store, MODEL, &plain_prompt(&tokens));
        let scoped = Hashes::of(
            &store,
            MODEL,
            &Prompt {
                namespace: Some(namespace),
                ..plain_prompt(&tokens)
            },
        );
        assert_ne!(plain.blocks, scoped.blocks);
        let scoped = Ask {
            hashes: Some(&scoped),
            ..first(b"scoped")
        };
        let facts = scoped.request(&[]).facts;

        let blocks =
            request_content_hashes_with_seed(&tokens, BLOCK_SIZE as usize, namespace.event_seed());
        assert_eq!(store_blocks(&store, holder, &blocks), BatchOutcome::Applied);
        store.deciding(|world| {
            let mut view = ViewBuffers::default();
            let ask = ViewAsk {
                gateway: GATEWAY,
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            let mut named = Named::default();
            let mut naming = Naming {
                gateway: GATEWAY,
                block_hashes: facts.block_hashes,
                seed: 0,
                cap: 0,
                wires: None,
                now_micros: built.now_micros,
            };
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert_eq!(named.holders.len(), 1);
            assert_eq!(view.candidates[named.holders[0].row as usize].id, holder);
            assert_eq!(named.holders[0].overlap_blocks, 1);
            naming.block_hashes = &plain.blocks;
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert!(named.holders.is_empty());
        });
        assert_eq!(facts.namespace_seed, namespace.event_seed());
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        assert_eq!(facts.hash_kind, HashKind::Token);
    }

    #[tokio::test]
    async fn no_block_is_hashed_past_the_generations_depth() {
        let (host, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let depth =
            store.deciding(|world| world.spec().block_hashes_read(HashKind::Token, BLOCK_SIZE));
        assert!(depth > 0);
        let tokens: Vec<u32> = (0..(depth as u32 + 1) * BLOCK_SIZE).collect();
        let prompt = plain_prompt(&tokens);
        let hashes = Hashes::of(&store, MODEL, &prompt);
        assert_eq!(hashes.blocks.len(), depth);
        assert_eq!(hashes.prompt_tokens, tokens.len() as u32);
        let ask = Ask {
            hashes: Some(&hashes),
            ..first(b"long")
        };
        let _asked = host
            .ask(&ask, &[], Instant::now())
            .await
            .expect("a decision");
        assert_eq!(host.door.hashes_cut(), 0);

        let mut document = PolicySpec::preset("random").expect("the preset");
        document.generation += 1;
        store
            .apply(AppliedSpec::validate(&document).expect("the generation"))
            .expect("the newer generation");
        let hashes = Hashes::of(&store, MODEL, &prompt);
        assert!(hashes.blocks.is_empty());
        assert_eq!(hashes.block_size, BLOCK_SIZE);
        assert_eq!(hashes.prompt_tokens, tokens.len() as u32);
        assert_eq!(hashes.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn a_request_with_no_tokens_states_no_hash() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let hashes = Hashes::of(&store, MODEL, &Prompt::default());
        assert!(hashes.blocks.is_empty());
        assert_eq!(
            (
                hashes.block_size,
                hashes.namespace_seed,
                hashes.prompt_tokens
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn a_model_with_no_block_size_of_record_states_no_hash() {
        let (_, store, _) = cache_bed();
        let tokens = [1; BLOCK_SIZE as usize];
        let prompt = plain_prompt(&tokens);
        for recorded in [None, Some(0)] {
            if let Some(size) = recorded {
                store.set_block_size(MODEL, size);
            }
            assert_eq!(store.block_size(MODEL), recorded);
            let hashes = Hashes::of(&store, MODEL, &prompt);
            assert!(hashes.blocks.is_empty());
            assert_eq!(hashes.block_size, 0);
            assert_eq!(hashes.prompt_tokens, BLOCK_SIZE);
            assert_eq!(hashes.namespace_seed, XXH3_SEED);
        }
    }

    #[test]
    fn the_plain_seed_gives_the_index_crates_plain_hashes() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=2 * BLOCK_SIZE).collect();
        let hashes = Hashes::of(&store, MODEL, &plain_prompt(&tokens));
        let expected: Vec<_> = compute_request_content_hashes(&tokens, BLOCK_SIZE as usize)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        assert_eq!(hashes.blocks, expected);
        assert_eq!(hashes.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn an_unpartitioned_request_drops_its_leading_marker_ids() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = [1 << 31, u32::MAX]
            .into_iter()
            .chain(1..=BLOCK_SIZE)
            .collect();
        let hashes = Hashes::of(&store, MODEL, &plain_prompt(&tokens));
        let expected = compute_request_content_hashes(&tokens[2..], BLOCK_SIZE as usize);
        assert_eq!(hashes.blocks, [expected[0].0]);
        assert_eq!(hashes.prompt_tokens, BLOCK_SIZE);
    }

    #[test]
    fn a_tail_shorter_than_a_block_gives_no_hash() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens = [1; BLOCK_SIZE as usize + 1];
        let hashes = Hashes::of(&store, MODEL, &plain_prompt(&tokens));
        assert_eq!(hashes.blocks.len(), 1);
        assert_eq!(hashes.prompt_tokens, BLOCK_SIZE + 1);
    }

    #[test]
    fn an_ask_with_no_hashes_states_the_facts_of_today() {
        let _bed = cache_bed();
        let facts = first(b"plain").request(&[]).facts;
        let defaults = RequestFacts::default();
        assert_eq!(facts.block_hashes, defaults.block_hashes);
        assert_eq!(facts.hash_kind, defaults.hash_kind);
        assert_eq!(facts.block_size, defaults.block_size);
        assert_eq!(facts.namespace_seed, defaults.namespace_seed);
        assert_eq!(facts.prompt_tokens, defaults.prompt_tokens);
    }

    #[test]
    fn the_prompt_token_count_saturates_at_u32_max() {
        let past_max = (u32::MAX as usize).saturating_add(1);
        assert_eq!(prompt_token_count(past_max), u32::MAX);
    }
}

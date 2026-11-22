//! The prompt facts an attempt keeps for its asks to the controller.

use smg_policy_controller::engine::{HashSeeds, PromptHashes};

use super::Prompt;
use crate::policies::CacheNamespace;

/// The hashes of `prompt` as `hash` makes them of what the prompt holds:
/// its token ids where the router tokenized it, else its text, under the
/// seeds. The policy engine hands the hashing of the generation in force
/// ([`DecisionEngine::prompt_hashes`](smg_policy_controller::engine::DecisionEngine::prompt_hashes)).
///
/// The seeds are the cache partition's. A request with no partition is
/// hashed under the plain seeds, less the leading ids that carry a
/// partition's marker.
pub(super) fn hashed(
    prompt: &Prompt<'_>,
    hash: impl FnOnce(Option<&[u32]>, Option<&str>, HashSeeds) -> PromptHashes,
) -> PromptHashes {
    let seeds = prompt
        .namespace
        .map_or(HashSeeds::PLAIN, |namespace| HashSeeds {
            // The blocks are hashed as the engines hash theirs, so the
            // index of cache events finds them. That seed folds no extra
            // key.
            event: namespace.event_seed(),
            // A prefix is named under the partition's whole identity.
            // The record of sent prefixes and the ring keep two
            // partitions apart that differ by the extra key alone, as an
            // engine's cache does.
            partition: namespace.partition_seed(),
        });
    let tokens = match prompt.namespace {
        Some(_) => prompt.tokens,
        None => prompt.tokens.map(CacheNamespace::unpartitioned_tokens),
    };
    hash(tokens, prompt.text, seeds)
}

/// A prompt's hashes for a model under the generation in force in a store,
/// for the cases of the policy engine.
#[cfg(test)]
pub(super) trait StoreHashes {
    /// The prompt's hashes for `model` under the generation in force in
    /// `store`.
    fn hashes(
        &self,
        store: &smg_policy_controller::store::MemoryStore,
        model: &str,
    ) -> PromptHashes;
}

#[cfg(test)]
impl StoreHashes for Prompt<'_> {
    fn hashes(
        &self,
        store: &smg_policy_controller::store::MemoryStore,
        model: &str,
    ) -> PromptHashes {
        hashed(self, |tokens, text, seeds| {
            store.prompt_hashes(model, tokens, text, seeds)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use kv_index::{
        compute_request_content_hashes,
        salt::{request_content_hashes_with_seed, request_text_hashes_with_seed},
        XXH3_SEED,
    };
    use openai_protocol::common::CachePartition;
    use smg_policy_controller::{
        document::{AppliedSpec, PolicySpec},
        engine::{HashKind, Identity, PromptHashes, RequestFacts, Role, TEXT_BYTES_PER_TOKEN},
        store::{
            AffinityQuery, AffinityRowsBuffers, BatchOutcome, KvEventSink, ViewBuffers,
            ViewQuery as ViewAsk,
        },
    };

    use crate::{
        policies::namespace_of,
        policy_controller::engine::{
            hashes::StoreHashes as _,
            testing::{cache_bed, record_bed, store_blocks, MODEL},
            tests::first,
            Ask, Prompt, GATEWAY,
        },
    };

    const BLOCK_SIZE: u32 = 16;
    /// A text of two windows and a tail, at a block of sixteen tokens.
    const RECORDED_TEXT: &str = "A request's text is hashed by windows of its bytes, as deep as \
                                 the generation reads, and a tail shorter than one window gives \
                                 no hash at all.";
    const RECORDED_TOKENS: [u64; 2] = [0xea06_fc71_a5ee_6e88, 0x1fbf_39bd_106b_e0b5];
    const RECORDED_SCOPED_TOKENS: [u64; 2] = [0x3c6f_83b1_c88e_dc76, 0xcbfd_d2b6_595c_c9c6];
    const RECORDED_WINDOWS: [u64; 2] = [0xc67b_5989_9662_3ba3, 0x627b_9253_c3e5_d0f8];

    fn plain_prompt(tokens: &[u32]) -> Prompt<'_> {
        Prompt {
            tokens: Some(tokens),
            ..Prompt::default()
        }
    }

    /// The facts `hashes` writes into a request that holds none.
    fn written(hashes: &PromptHashes) -> RequestFacts<'_> {
        let mut facts = RequestFacts::default();
        hashes.write_to(&mut facts);
        facts
    }

    #[test]
    fn a_namespace_hashes_under_its_own_seed() {
        let (_, store, holder) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let namespace = namespace_of(&CachePartition {
            cache_salt: Some("partition"),
            ..CachePartition::default()
        })
        .expect("the namespace");
        let tokens: Vec<u32> = (1..=BLOCK_SIZE).collect();
        let plain = plain_prompt(&tokens).hashes(&store, MODEL);
        let scoped = Prompt {
            namespace: Some(namespace),
            ..plain_prompt(&tokens)
        }
        .hashes(&store, MODEL);
        assert_ne!(written(&plain).block_hashes, written(&scoped).block_hashes);
        let scoped = Ask {
            hashes: Some(&scoped),
            ..first(b"scoped")
        };
        let facts = scoped.request(Identity::NONE, 0, &[]).facts;

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
                signal_cells: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            let mut named = AffinityRowsBuffers::default();
            let mut naming = AffinityQuery {
                gateway: GATEWAY,
                block_hashes: facts.block_hashes,
                namespace_seed: facts.namespace_seed,
                seed: 0,
                cap: 0,
                wires: None,
                now_micros: built.now_micros,
            };
            world.fill_affinity_rows(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert_eq!(named.holders.len(), 1);
            assert_eq!(view.candidates[named.holders[0].row as usize].id, holder);
            assert_eq!(named.holders[0].overlap_blocks, 1);
            naming.block_hashes = written(&plain).block_hashes;
            world.fill_affinity_rows(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert!(named.holders.is_empty());
        });
        assert_eq!(facts.namespace_seed, namespace.partition_seed());
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        assert_eq!(facts.hash_kind, HashKind::Token);
    }

    #[tokio::test]
    async fn no_block_is_hashed_past_the_generations_depth() {
        let (policy_engine, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let depth =
            store.deciding(|world| world.spec().block_hashes_read(HashKind::Token, BLOCK_SIZE));
        assert!(depth > 0);
        let tokens: Vec<u32> = (0..(depth as u32 + 1) * BLOCK_SIZE).collect();
        let prompt = plain_prompt(&tokens);
        let hashes = prompt.hashes(&store, MODEL);
        let facts = written(&hashes);
        assert_eq!(facts.block_hashes.len(), depth);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        let ask = Ask {
            hashes: Some(&hashes),
            ..first(b"long")
        };
        let _asked = policy_engine
            .ask(&ask, &[], Instant::now())
            .await
            .expect("a decision");
        let door = policy_engine
            .door
            .followed()
            .expect("the engine over the store");
        assert_eq!(door.hashes_cut(), 0);

        let mut document = PolicySpec::preset("random").expect("the preset");
        document.generation += 1;
        store
            .apply(AppliedSpec::validate(&document).expect("the generation"))
            .expect("the newer generation");
        let hashes = prompt.hashes(&store, MODEL);
        let facts = written(&hashes);
        assert!(facts.block_hashes.is_empty());
        assert_eq!(facts.block_size, BLOCK_SIZE);
        assert_eq!(facts.prompt_tokens, tokens.len() as u32);
        assert_eq!(facts.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn a_text_prompt_is_hashed_by_windows_of_bytes() {
        // A generation whose prefix stage reads the record of sent
        // prefixes: it declares text hashes, so the policy engine hashes them.
        let (_, store) = record_bed();
        store.set_block_size(MODEL, 32);
        let window = 32 * TEXT_BYTES_PER_TOKEN;
        let depth = store.deciding(|world| world.spec().block_hashes_read(HashKind::Text, window));
        assert!(depth > 0, "the generation reads text windows");
        // Two windows past the depth, and a tail of three bytes.
        let text: String = ('a'..='z')
            .cycle()
            .take((depth + 2) * window as usize + 3)
            .collect();
        let prompt = Prompt {
            text: Some(&text),
            ..Prompt::default()
        };
        let hashes = prompt.hashes(&store, MODEL);
        let read = &text.as_bytes()[..depth * window as usize];
        let windows = request_text_hashes_with_seed(read, window as usize, XXH3_SEED);
        assert_eq!(
            written(&hashes).block_hashes,
            windows.iter().map(|hash| hash.0).collect::<Vec<_>>()
        );
        let facts = Ask {
            hashes: Some(&hashes),
            ..first(b"text")
        }
        .request(Identity::NONE, 0, &[])
        .facts;
        assert_eq!(
            (facts.hash_kind, facts.block_size, facts.prompt_tokens),
            (HashKind::Text, window, text.len() as u32 / 4)
        );
        // Token ids win where a prompt states both.
        let tokens: Vec<u32> = (1..=64).collect();
        let both = Prompt {
            tokens: Some(&tokens),
            ..prompt
        };
        assert_eq!(
            written(&both.hashes(&store, MODEL)).hash_kind,
            HashKind::Token
        );
        // A partition is in the text's seed whole. An extra key alone moves
        // every window, where it leaves the seed of the engines' events.
        let keyed = namespace_of(&CachePartition {
            extra_key: Some("k"),
            ..CachePartition::default()
        })
        .expect("the namespace");
        assert_eq!(keyed.event_seed(), XXH3_SEED);
        let scoped = Prompt {
            namespace: Some(keyed),
            ..prompt
        }
        .hashes(&store, MODEL);
        let (scoped, plain) = (written(&scoped), written(&hashes));
        assert_eq!(scoped.block_hashes.len(), plain.block_hashes.len());
        assert!(scoped
            .block_hashes
            .iter()
            .zip(plain.block_hashes)
            .all(|(a, b)| a != b));
        assert_eq!(scoped.namespace_seed, keyed.partition_seed());
    }

    #[test]
    fn a_token_prompt_under_another_extra_key_is_named_apart() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=BLOCK_SIZE * 2).collect();
        let under = |extra_key| {
            let namespace = namespace_of(&CachePartition {
                extra_key: Some(extra_key),
                ..CachePartition::default()
            })
            .expect("the namespace");
            let prompt = Prompt {
                namespace: Some(namespace),
                ..plain_prompt(&tokens)
            };
            prompt.hashes(&store, MODEL)
        };
        let (a, b) = (under("a"), under("b"));
        let (a, b) = (written(&a), written(&b));
        // The engines' events fold no extra key: the blocks are one, and
        // the index of cache events finds them under either partition.
        assert!(!a.block_hashes.is_empty());
        assert_eq!(a.block_hashes, b.block_hashes);
        // The prefix is named under each partition's own seed, so the
        // record of sent prefixes and the ring keep the two apart.
        assert_ne!(a.namespace_seed, b.namespace_seed);
        assert_ne!(a.namespace_seed, XXH3_SEED);
    }

    #[test]
    fn a_request_with_no_tokens_states_no_hash() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let hashes = Prompt::default().hashes(&store, MODEL);
        let facts = written(&hashes);
        assert!(facts.block_hashes.is_empty());
        assert_eq!(
            (facts.block_size, facts.namespace_seed, facts.prompt_tokens),
            (0, 0, 0)
        );
    }

    #[test]
    fn an_unpartitioned_request_drops_its_leading_marker_ids() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = [1 << 31, u32::MAX]
            .into_iter()
            .chain(1..=BLOCK_SIZE)
            .collect();
        let hashes = plain_prompt(&tokens).hashes(&store, MODEL);
        let facts = written(&hashes);
        let expected = compute_request_content_hashes(&tokens[2..], BLOCK_SIZE as usize);
        assert_eq!(facts.block_hashes, [expected[0].0]);
        assert_eq!(facts.prompt_tokens, BLOCK_SIZE);
    }

    #[test]
    fn an_ask_with_no_hashes_states_the_facts_of_today() {
        let _bed = cache_bed();
        let facts = first(b"plain").request(Identity::NONE, 0, &[]).facts;
        let defaults = RequestFacts::default();
        assert_eq!(facts.block_hashes, defaults.block_hashes);
        assert_eq!(facts.hash_kind, defaults.hash_kind);
        assert_eq!(facts.block_size, defaults.block_size);
        assert_eq!(facts.namespace_seed, defaults.namespace_seed);
        assert_eq!(facts.prompt_tokens, defaults.prompt_tokens);
    }

    /// The block hashes of three prompts, as values of record: a plain
    /// token prompt, the same tokens under a partition's event seed, and a
    /// text prompt. The policy crate's case reads the same values.
    #[test]
    fn a_prompts_hashes_are_the_recorded_ones() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=40).collect();
        let plain = plain_prompt(&tokens).hashes(&store, MODEL);
        assert_eq!(written(&plain).block_hashes, RECORDED_TOKENS);
        let namespace = namespace_of(&CachePartition {
            cache_salt: Some("partition"),
            ..CachePartition::default()
        })
        .expect("the namespace");
        let scoped = Prompt {
            namespace: Some(namespace),
            ..plain_prompt(&tokens)
        }
        .hashes(&store, MODEL);
        let facts = written(&scoped);
        assert_eq!(facts.block_hashes, RECORDED_SCOPED_TOKENS);
        assert_eq!(facts.namespace_seed, namespace.partition_seed());
        let text = Prompt {
            text: Some(RECORDED_TEXT),
            ..Prompt::default()
        };
        let windows = text.hashes(&store, MODEL);
        let facts = written(&windows);
        assert_eq!(facts.block_hashes, RECORDED_WINDOWS);
        assert_eq!(facts.prompt_tokens, RECORDED_TEXT.len() as u32 / 4);
    }
}

//! The prompt facts an attempt keeps for its asks to the controller.

use kv_index::{
    salt::{request_content_hashes_with_seed, request_text_hashes_with_seed},
    XXH3_SEED,
};
use smg_policy_controller::{
    store::{KvEventSink, MemoryStore},
    view::{HashKind, TEXT_BYTES_PER_TOKEN},
    AppliedSpec,
};

use crate::policies::CacheNamespace;

/// What an attempt states of its prompt, for the cache index.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Prompt<'a> {
    /// The prompt's token ids, where the router tokenized it.
    pub tokens: Option<&'a [u32]>,
    /// The request's text as its route reads it for routing, where the
    /// router holds no token ids. Token ids win where both are stated.
    pub text: Option<&'a str>,
    /// The request's cache partition, where it has one.
    pub namespace: Option<CacheNamespace>,
}

impl<'a> Prompt<'a> {
    /// A prompt of `tokens` under the request's cache partition, with no
    /// text. A stage that states a text names it over this one, and a stage
    /// that states none names no field of the prompt.
    pub(crate) const fn of(tokens: Option<&'a [u32]>, namespace: Option<CacheNamespace>) -> Self {
        Self {
            tokens,
            text: None,
            namespace,
        }
    }
}

/// A request's block hashes as the door reads them. The attempt owns the
/// buffer, and its asks share it.
#[derive(Debug, Default)]
pub(crate) struct Hashes {
    pub(super) blocks: Vec<u64>,
    /// What the blocks hash: token blocks, or windows of the text.
    pub(super) kind: HashKind,
    /// The size of one block: in tokens, or in bytes for a window.
    pub(super) block_size: u32,
    pub(super) namespace_seed: u64,
    pub(super) prompt_tokens: u32,
}

impl Hashes {
    /// The hashes of `prompt` for `model`, under the generation in force.
    ///
    /// Prompt tokens are hashed by the model's block size
    /// ([`block_size_of`]), so a model whose workers publish no cache
    /// events is hashed too. The generation bounds the number of full
    /// blocks hashed.
    pub(crate) fn of(store: &MemoryStore, model: &str, prompt: &Prompt<'_>) -> Self {
        let Some(tokens) = prompt.tokens else {
            // No engine hashes text: its seed is the whole partition's.
            let seed = prompt
                .namespace
                .map_or(XXH3_SEED, CacheNamespace::partition_seed);
            return prompt.text.map_or_else(Self::default, |text| {
                Self::of_text(store, model, text, seed)
            });
        };
        // The blocks are hashed as the engines hash theirs, so the index of
        // cache events finds them. That seed folds no extra key.
        let event_seed = prompt
            .namespace
            .map_or(XXH3_SEED, CacheNamespace::event_seed);
        let tokens = if prompt.namespace.is_some() {
            tokens
        } else {
            CacheNamespace::unpartitioned_tokens(tokens)
        };
        let (block_size, depth) = store.deciding(|world| {
            let spec = world.spec();
            let block_size = block_size_of(store, spec, model);
            let depth = spec.block_hashes_read(HashKind::Token, block_size);
            (block_size, depth)
        });
        let size = block_size as usize;
        let end = tokens.len().min(depth.saturating_mul(size));
        let blocks = request_content_hashes_with_seed(&tokens[..end], size, event_seed)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        // A prefix is named under the partition's whole identity. The
        // record of sent prefixes and the ring keep two partitions apart
        // that differ by the extra key alone, as an engine's cache does.
        let namespace_seed = prompt
            .namespace
            .map_or(XXH3_SEED, CacheNamespace::partition_seed);
        Self {
            blocks,
            kind: HashKind::Token,
            block_size,
            namespace_seed,
            prompt_tokens: prompt_token_count(tokens.len()),
        }
    }

    /// The hashes of a prompt its router holds as text: one a window of the
    /// text's bytes, as deep as the generation reads. A window is the
    /// model's block size times [`TEXT_BYTES_PER_TOKEN`], so one number a
    /// model sizes both kinds. The text is hashed as it stands: its whole
    /// partition is in the seed ([`CacheNamespace::partition_seed`]), and no
    /// marker stands before it.
    fn of_text(store: &MemoryStore, model: &str, text: &str, namespace_seed: u64) -> Self {
        let bytes = text.as_bytes();
        let (window, depth) = store.deciding(|world| {
            let spec = world.spec();
            let window = block_size_of(store, spec, model).saturating_mul(TEXT_BYTES_PER_TOKEN);
            (window, spec.block_hashes_read(HashKind::Text, window))
        });
        let size = window as usize;
        let end = bytes.len().min(depth.saturating_mul(size));
        let blocks = request_text_hashes_with_seed(&bytes[..end], size, namespace_seed)
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        Self {
            blocks,
            kind: HashKind::Text,
            block_size: window,
            namespace_seed,
            prompt_tokens: prompt_token_count(bytes.len() / TEXT_BYTES_PER_TOKEN as usize),
        }
    }
}

/// The model's block size: its size of record, and the document's where it
/// has none.
fn block_size_of(store: &MemoryStore, spec: &AppliedSpec, model: &str) -> u32 {
    store.block_size(model).unwrap_or_else(|| spec.block_size())
}

fn prompt_token_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
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
        store::{Ask as ViewAsk, BatchOutcome, KvEventSink, Named, Naming, ViewBuffers},
        view::{HashKind, Identity, RequestFacts, Role, TEXT_BYTES_PER_TOKEN},
        AppliedSpec, PolicySpec,
    };

    use super::{prompt_token_count, Hashes};
    use crate::{
        policies::CacheNamespace,
        policy_controller::host::{
            testing::{cache_bed, record_bed, store_blocks, MODEL},
            tests::first,
            Ask, Prompt, GATEWAY,
        },
    };

    const BLOCK_SIZE: u32 = 16;

    fn plain_prompt(tokens: &[u32]) -> Prompt<'_> {
        Prompt {
            tokens: Some(tokens),
            ..Prompt::default()
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
        let tokens: Vec<u32> = (1..=BLOCK_SIZE).collect();
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
            let mut named = Named::default();
            let mut naming = Naming {
                gateway: GATEWAY,
                block_hashes: facts.block_hashes,
                namespace_seed: facts.namespace_seed,
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
        assert_eq!(facts.namespace_seed, namespace.partition_seed());
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
    fn a_text_prompt_is_hashed_by_windows_of_bytes() {
        // A generation whose prefix stage reads the record of sent
        // prefixes: it declares text hashes, so the host hashes them.
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
        let hashes = Hashes::of(&store, MODEL, &prompt);
        let read = &text.as_bytes()[..depth * window as usize];
        let windows = request_text_hashes_with_seed(read, window as usize, XXH3_SEED);
        assert_eq!(
            hashes.blocks,
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
        assert_eq!(Hashes::of(&store, MODEL, &both).kind, HashKind::Token);
        // A partition is in the text's seed whole. An extra key alone moves
        // every window, where it leaves the seed of the engines' events.
        let keyed = CacheNamespace::derive(&CachePartition {
            extra_key: Some("k"),
            ..CachePartition::default()
        })
        .expect("the namespace");
        assert_eq!(keyed.event_seed(), XXH3_SEED);
        let scoped = Hashes::of(
            &store,
            MODEL,
            &Prompt {
                namespace: Some(keyed),
                ..prompt
            },
        );
        assert_eq!(scoped.blocks.len(), hashes.blocks.len());
        assert!(scoped
            .blocks
            .iter()
            .zip(&hashes.blocks)
            .all(|(a, b)| a != b));
        assert_eq!(scoped.namespace_seed, keyed.partition_seed());
    }

    #[test]
    fn a_token_prompt_under_another_extra_key_is_named_apart() {
        let (_, store, _) = cache_bed();
        store.set_block_size(MODEL, BLOCK_SIZE);
        let tokens: Vec<u32> = (1..=BLOCK_SIZE * 2).collect();
        let under = |extra_key| {
            let namespace = CacheNamespace::derive(&CachePartition {
                extra_key: Some(extra_key),
                ..CachePartition::default()
            })
            .expect("the namespace");
            let prompt = Prompt {
                namespace: Some(namespace),
                ..plain_prompt(&tokens)
            };
            Hashes::of(&store, MODEL, &prompt)
        };
        let (a, b) = (under("a"), under("b"));
        // The engines' events fold no extra key: the blocks are one, and
        // the index of cache events finds them under either partition.
        assert!(!a.blocks.is_empty());
        assert_eq!(a.blocks, b.blocks);
        // The prefix is named under each partition's own seed, so the
        // record of sent prefixes and the ring keep the two apart.
        assert_ne!(a.namespace_seed, b.namespace_seed);
        assert_ne!(a.namespace_seed, XXH3_SEED);
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

    /// A model whose workers state no block size is hashed by the
    /// document's. A size of record wins once there is one, and one of
    /// zero states no hash.
    #[test]
    fn a_model_with_no_block_size_of_record_is_hashed_by_the_documents() {
        let (_, store, _) = cache_bed();
        let size = store.deciding(|world| world.spec().block_size());
        let tokens: Vec<u32> = (1..=2 * size).collect();
        let prompt = plain_prompt(&tokens);
        for (recorded, blocks, stated) in [
            (None, 2, size),
            (Some(0), 0, 0),
            (Some(2 * size), 1, 2 * size),
        ] {
            if let Some(size) = recorded {
                store.set_block_size(MODEL, size);
            }
            assert_eq!(store.block_size(MODEL), recorded);
            let hashes = Hashes::of(&store, MODEL, &prompt);
            assert_eq!(
                (hashes.blocks.len(), hashes.block_size),
                (blocks, stated),
                "{recorded:?}"
            );
            assert_eq!(hashes.prompt_tokens, tokens.len() as u32);
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
        let facts = first(b"plain").request(Identity::NONE, 0, &[]).facts;
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

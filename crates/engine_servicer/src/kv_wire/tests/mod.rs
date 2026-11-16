mod shapes;

use serde_json::{json, Value};

use super::*;

/// A publisher batch from JSON: objects become msgpack maps (the engines'
/// tagged-map events), arrays become arrays (the batch envelope and the
/// legacy event layout).
fn batch_from(value: Value) -> WireBatch {
    let bytes = rmp_serde::to_vec_named(&value).expect("encodes");
    rmp_serde::from_slice(&bytes).expect("decodes")
}

fn normalize_all(batches: Vec<Value>) -> (Vec<common::KvEventBatch>, Normalizer) {
    let mut normalizer = Normalizer::new();
    let mut event_id = 0;
    let out = batches
        .into_iter()
        .enumerate()
        .map(|(seq, value)| normalize(&mut normalizer, value, seq as u64, &mut event_id))
        .collect();
    (out, normalizer)
}

fn normalize(
    normalizer: &mut Normalizer,
    value: Value,
    seq: u64,
    event_id: &mut u64,
) -> common::KvEventBatch {
    normalizer.normalize_batch(batch_from(value), seq, event_id)
}

fn one(events: Vec<Value>) -> Value {
    json!([1700000000.5, events, 0])
}

fn store(hashes: &[i64], parent: Option<i64>, tokens: &[u32]) -> Value {
    json!({
        "type": "BlockStored",
        "block_hashes": hashes,
        "parent_block_hash": parent,
        "token_ids": tokens,
        "block_size": 4,
        "lora_id": null,
        "medium": "GPU",
        "lora_name": null,
        "group_idx": 0,
        "kv_cache_spec_kind": "full_attention",
    })
}

fn remove(hashes: &[i64]) -> Value {
    json!({"type": "BlockRemoved", "block_hashes": hashes, "medium": "GPU", "group_idx": 0})
}

fn with(mut value: Value, key: &str, item: Value) -> Value {
    value[key] = item;
    value
}

fn stored(event: &common::KvCacheEvent) -> &common::KvBlocksStored {
    match event.data {
        Some(kv_cache_event::Data::Stored(ref stored)) => stored,
        ref other => panic!("not a store: {other:?}"),
    }
}

fn removed(event: &common::KvCacheEvent) -> &common::KvBlocksRemoved {
    match event.data {
        Some(kv_cache_event::Data::Removed(ref removed)) => removed,
        ref other => panic!("not a removal: {other:?}"),
    }
}

#[test]
fn hash_identity_folds_like_vllm() {
    let mut digest = [0u8; 32];
    digest[23] = 0xaa;
    digest[24..].copy_from_slice(&0x8000_0000_0000_0001u64.to_be_bytes());
    assert_eq!(low64_big_endian(&digest), 0x8000_0000_0000_0001);
    assert_eq!(low64_big_endian(&[0, 0, 0, 0, 0, 0, 0, 7]), 7);

    let (batches, _) = normalize_all(vec![one(vec![store(
        &[i64::MIN + 1, -3],
        None,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    )])]);
    let blocks = &stored(&batches[0].events[0]).blocks;
    assert_eq!(
        blocks[0].block_hash,
        i64::MIN + 1,
        "a u64 above i64::MAX keeps its bits"
    );
    assert_eq!(
        blocks[1].block_hash, -3,
        "SGLang's signed form passes through"
    );
}

#[test]
fn unknown_and_malformed_events_cost_themselves_not_the_batch() {
    let (batches, normalizer) = normalize_all(vec![one(vec![
        json!({"type": "BlockMigrated", "block_hashes": [1], "destination": "peer"}),
        json!({"type": "BlockStored", "block_hashes": "nope", "token_ids": [1], "block_size": 1}),
        json!({"block_hashes": [1]}),
        json!(["BlockStored", "nope"]),
        remove(&[9]),
    ])]);
    assert_eq!(batches[0].events.len(), 1);
    assert_eq!(removed(&batches[0].events[0]).block_hashes, vec![9]);
    assert_eq!(
        batches[0].events[0].event_id, 5,
        "ids advance for dropped events too"
    );
    let counts = normalizer.counts();
    assert_eq!(counts.dropped(DropReason::UnknownType), 1);
    assert_eq!(counts.dropped(DropReason::Malformed), 3);
    assert_eq!(counts.forwarded_removed, 1);
}

#[test]
fn residency_agent_events_are_dropped() {
    let (batches, normalizer) = normalize_all(vec![one(vec![
        with(store(&[1], None, &[1, 2, 3, 4]), "ownership", json!("kvcr")),
        with(remove(&[1]), "ownership", json!("KVCR")),
        json!({"type": "AllBlocksCleared", "ownership": "kvcr"}),
        json!({"type": "AllBlocksCleared"}),
    ])]);
    assert_eq!(batches[0].events.len(), 1);
    assert!(matches!(
        batches[0].events[0].data,
        Some(kv_cache_event::Data::Cleared(_))
    ));
    assert_eq!(
        normalizer
            .counts()
            .dropped(DropReason::UnsupportedOwnership),
        3
    );
    assert_eq!(normalizer.counts().forwarded_cleared, 1);
}

#[test]
fn remote_and_unknown_localities_are_dropped() {
    let (batches, normalizer) = normalize_all(vec![one(vec![
        with(
            store(&[1], None, &[1, 2, 3, 4]),
            "locality",
            json!("REMOTE"),
        ),
        with(remove(&[1]), "locality", json!("elsewhere")),
        with(store(&[2], None, &[1, 2, 3, 4]), "locality", json!("local")),
    ])]);
    assert_eq!(batches[0].events.len(), 1);
    assert_eq!(
        stored(&batches[0].events[0]).locality,
        Some(KvCacheLocality::Local as i32)
    );
    assert_eq!(normalizer.counts().dropped(DropReason::NonLocalLocality), 2);
}

#[test]
fn media_map_to_tiers_and_cache_levels() {
    let table = [
        (None, KvCacheTier::Device, None),
        (Some("GPU"), KvCacheTier::Device, None),
        (Some("device"), KvCacheTier::Device, None),
        (Some("CPU"), KvCacheTier::Host, Some(1)),
        (Some("CPU_PINNED"), KvCacheTier::Host, Some(1)),
        (Some("CPU_TIER1"), KvCacheTier::Host, Some(1)),
        (Some("CPU_TIER2"), KvCacheTier::Disk, Some(2)),
        (Some("DISK"), KvCacheTier::Disk, Some(2)),
        (Some("NVME"), KvCacheTier::Disk, Some(2)),
        (Some("STORAGE"), KvCacheTier::Disk, Some(2)),
        (Some("EXTERNAL"), KvCacheTier::External, Some(3)),
        (Some("NETWORK"), KvCacheTier::External, Some(3)),
        (Some("REMOTE"), KvCacheTier::External, Some(3)),
        (Some("SHARED"), KvCacheTier::External, Some(3)),
    ];
    for (medium, tier, level) in table {
        assert_eq!(tier_of(medium), Some(tier), "{medium:?}");
        assert_eq!(cache_level_of(tier), level, "{medium:?}");
    }
    assert_eq!(tier_of(Some("MARS")), None);

    let (batches, normalizer) = normalize_all(vec![one(vec![
        with(
            store(&[1], None, &[1, 2, 3, 4]),
            "medium",
            json!("CPU_PINNED"),
        ),
        with(remove(&[1]), "medium", json!("STORAGE")),
        with(store(&[2], None, &[1, 2, 3, 4]), "medium", json!("MARS")),
        with(store(&[3], None, &[1, 2, 3, 4]), "medium", Value::Null),
    ])]);
    let events = &batches[0].events;
    assert_eq!(events.len(), 3);
    let host = stored(&events[0]);
    assert_eq!(host.tier, Some(KvCacheTier::Host as i32));
    assert_eq!(host.medium.as_deref(), Some("CPU_PINNED"));
    assert_eq!(host.blocks[0].cache_level, Some(1));
    let disk = removed(&events[1]);
    assert_eq!(disk.tier, Some(KvCacheTier::Disk as i32));
    assert_eq!(disk.cache_level, Some(2));
    let device = stored(&events[2]);
    assert_eq!(device.tier, Some(KvCacheTier::Device as i32));
    assert_eq!(device.medium, None);
    assert_eq!(device.blocks[0].cache_level, None);
    assert_eq!(normalizer.counts().dropped(DropReason::UnknownMedium), 1);
}

/// On a rank with a main-attention group, every other group's event is
/// forwarded on its own lane: a store with its hashes, parent and group
/// fields and no tokens, a removal as it came; the groups are remembered
/// for the events that omit the kind.
#[test]
fn non_main_groups_are_forwarded_on_their_own_lane_and_remembered() {
    let sliding = |hash: i64| {
        let event = with(store(&[hash], None, &[1, 2, 3, 4]), "group_idx", json!(1));
        let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
        with(event, "kv_cache_spec_sliding_window", json!(128))
    };
    let (batches, normalizer) = normalize_all(vec![
        one(vec![
            sliding(1),
            store(&[2], None, &[1, 2, 3, 4]),
            with(
                with(
                    store(&[3], None, &[1, 2, 3, 4]),
                    "kv_cache_spec_kind",
                    json!("mla_attention"),
                ),
                "group_idx",
                json!(2),
            ),
            with(
                with(
                    store(&[4], None, &[1, 2, 3, 4]),
                    "kv_cache_spec_kind",
                    json!("sink_full_attention"),
                ),
                "group_idx",
                json!(3),
            ),
            with(
                with(
                    store(&[5], None, &[1, 2, 3, 4]),
                    "kv_cache_spec_kind",
                    json!("mamba"),
                ),
                "group_idx",
                json!(4),
            ),
        ]),
        one(vec![
            // Removals carry no kind: the learned groups decide.
            with(remove(&[1]), "group_idx", json!(1)),
            with(remove(&[2]), "group_idx", json!(0)),
            // An unlearned group without a kind counts as main.
            with(remove(&[7]), "group_idx", json!(9)),
            // A kind-less store on a learned non-main group follows the group.
            with(
                with(store(&[8], None, &[1, 2, 3, 4]), "group_idx", json!(4)),
                "kv_cache_spec_kind",
                Value::Null,
            ),
        ]),
    ]);
    assert_eq!(batches[0].events.len(), 5, "{:?}", batches[0].events);
    let window = stored(&batches[0].events[0]);
    assert_eq!(window.group_idx, Some(1));
    assert_eq!(window.kv_cache_spec_kind.as_deref(), Some("sliding_window"));
    assert_eq!(window.kv_cache_spec_sliding_window, Some(128));
    assert_eq!(window.blocks.len(), 1);
    assert_eq!(window.blocks[0].block_hash, 1);
    assert!(
        window.blocks[0].token_ids.is_empty(),
        "no tokens on a group lane"
    );
    assert_eq!(window.blocks[0].block_size, 4);
    assert_eq!(
        stored(&batches[0].events[2]).kv_cache_spec_kind.as_deref(),
        Some("mla_attention")
    );
    let state = stored(&batches[0].events[4]);
    assert_eq!(state.group_idx, Some(4));
    assert_eq!(state.kv_cache_spec_kind.as_deref(), Some("mamba"));
    assert!(state.blocks[0].token_ids.is_empty());
    assert_eq!(batches[1].events.len(), 4, "{:?}", batches[1].events);
    let gone = removed(&batches[1].events[0]);
    assert_eq!(
        (gone.block_hashes.clone(), gone.group_idx),
        (vec![1], Some(1))
    );
    assert_eq!(removed(&batches[1].events[1]).block_hashes, vec![2]);
    assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![7]);
    let later = stored(&batches[1].events[3]);
    assert_eq!(later.group_idx, Some(4));
    assert!(
        later.blocks[0].token_ids.is_empty(),
        "the learned group decides"
    );
    let counts = normalizer.counts();
    assert_eq!((counts.group_stores, counts.group_removals), (3, 1));
    assert_eq!((counts.forwarded_stored, counts.forwarded_removed), (6, 3));
    assert!(counts.dropped.is_empty(), "{:?}", counts.dropped);
}

/// An engine that resets its prefix cache and stores the next prompt in
/// one step publishes the clear and the stores in one batch, the window
/// groups' stores ahead of the main group's. The clear empties the block
/// records but not the groups, so the window stores go on their group lane
/// as on every other step (not as a window-only rank's main signal) and
/// the main group's store is forwarded whole; the records cleared, the
/// prompt's blocks are new, not duplicates.
#[test]
fn a_clear_in_the_batch_keeps_the_learned_groups() {
    let window = |hashes: &[i64], tokens: &[u32]| {
        let event = with(store(hashes, None, tokens), "group_idx", json!(1));
        let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
        with(event, "kv_cache_spec_sliding_window", json!(8))
    };
    let prompt: Vec<u32> = (1..=12).collect();
    let (batches, normalizer) = normalize_all(vec![
        // The groups are learned on the first step.
        one(vec![
            window(&[1], &[1, 2, 3, 4]),
            store(&[1], None, &[1, 2, 3, 4]),
        ]),
        // The reset and the next prompt in one step: the window group
        // names the prompt's last two blocks, the main group all three.
        one(vec![
            json!({"type": "AllBlocksCleared"}),
            window(&[2, 3], &prompt),
            store(&[1, 2, 3], None, &prompt),
        ]),
        // The step after is judged the same way.
        one(vec![
            window(&[4], &[13, 14, 15, 16]),
            store(&[4], Some(3), &[13, 14, 15, 16]),
        ]),
    ]);
    assert_eq!(batches[0].events.len(), 2);
    assert_eq!(batches[1].events.len(), 3, "{:?}", batches[1].events);
    assert!(matches!(
        batches[1].events[0].data,
        Some(kv_cache_event::Data::Cleared(_))
    ));
    let window_store = stored(&batches[1].events[1]);
    assert_eq!(window_store.group_idx, Some(1));
    assert_eq!(
        window_store
            .blocks
            .iter()
            .map(|block| (block.block_hash, block.token_ids.len()))
            .collect::<Vec<_>>(),
        vec![(2, 0), (3, 0)],
        "the window group's blocks, hashes only"
    );
    let prompt_store = stored(&batches[1].events[2]);
    assert_eq!(
        prompt_store
            .blocks
            .iter()
            .map(|block| block.block_hash)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(prompt_store.group_idx, Some(0));
    assert_eq!(batches[2].events.len(), 2);
    let counts = normalizer.counts();
    assert_eq!(counts.group_stores, 3);
    assert_eq!(counts.window_only_stores, 0);
    assert_eq!(counts.forwarded_stored, 6);
    assert_eq!(
        counts.duplicate_stores, 0,
        "block 1 was stored again after the clear: a new block, not a resend"
    );
}

/// The relay's record of the reset step, as a snapshot hands it to a
/// subscriber that joins afterwards: one of the window group's blocks as a
/// store of its own lane first, the whole prompt as one run under the main
/// group, then the window group's other block. (Recorded as copies of the
/// main lane, the blocks the window named would be left out of the main
/// group's run: a fresh subscriber would score the prompt's first blocks
/// only.)
#[test]
fn a_snapshot_after_a_reset_step_holds_the_prompt_under_the_main_group() {
    use crate::kv_state::{LiveState, SnapshotChunks};

    let window = |hashes: &[i64], tokens: &[u32]| {
        let event = with(store(hashes, None, tokens), "group_idx", json!(1));
        let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
        with(event, "kv_cache_spec_sliding_window", json!(8))
    };
    let prompt: Vec<u32> = (1..=24).collect();
    let (batches, _) = normalize_all(vec![
        one(vec![
            window(&[9], &[1, 2, 3, 4]),
            store(&[9], None, &[1, 2, 3, 4]),
        ]),
        one(vec![
            json!({"type": "AllBlocksCleared"}),
            window(&[5, 6], &prompt),
            store(&[1, 2, 3, 4, 5, 6], None, &prompt),
        ]),
    ]);
    let mut state = LiveState::new();
    for batch in &batches {
        state.apply(batch);
    }
    let chunks: Vec<common::KvEventBatch> =
        SnapshotChunks::at(state.snapshot(), 1, 2.0, 0).collect();
    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert_eq!(chunk.snapshot.as_ref().map(|marker| marker.blocks), Some(8));
    assert!(matches!(
        chunk.events[0].data,
        Some(kv_cache_event::Data::Cleared(_))
    ));
    assert_eq!(chunk.events.len(), 4, "{:?}", chunk.events);
    let run = stored(&chunk.events[2]);
    assert_eq!(run.parent_block_hash, None);
    assert_eq!(run.group_idx, Some(0));
    assert_eq!(run.kv_cache_spec_kind.as_deref(), Some("full_attention"));
    assert_eq!(
        run.blocks
            .iter()
            .map(|block| block.block_hash)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6]
    );
    let window_blocks: Vec<(i64, usize)> = [1usize, 3]
        .into_iter()
        .flat_map(|at| {
            let window_run = stored(&chunk.events[at]);
            assert_eq!(window_run.group_idx, Some(1));
            assert_eq!(
                window_run.kv_cache_spec_kind.as_deref(),
                Some("sliding_window")
            );
            assert_eq!(window_run.kv_cache_spec_sliding_window, Some(8));
            window_run
                .blocks
                .iter()
                .map(|block| (block.block_hash, block.token_ids.len()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(window_blocks, vec![(5, 0), (6, 0)]);
}

#[test]
fn placeholders_unaligned_and_self_referencing_stores_are_dropped() {
    let (batches, normalizer) = normalize_all(vec![one(vec![
        // vLLM's CPU offload placeholder: a chunk key, no tokens, block_size 0.
        with(
            with(store(&[1], None, &[]), "block_size", json!(0)),
            "medium",
            json!("CPU"),
        ),
        store(&[], None, &[1, 2, 3, 4]),
        store(&[2], None, &[1, 2, 3, 4, 5, 6]),
        with(store(&[3], None, &[1, 2, 3, 4]), "block_size", json!(0)),
        store(&[4], Some(4), &[1, 2, 3, 4]),
        store(&[5, 5], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
        store(&[6], None, &[1, 2, 3, 4]),
    ])]);
    assert_eq!(batches[0].events.len(), 1);
    assert_eq!(stored(&batches[0].events[0]).blocks[0].block_hash, 6);
    let counts = normalizer.counts();
    assert_eq!(counts.dropped(DropReason::Placeholder), 2);
    assert_eq!(counts.dropped(DropReason::UnalignedBlocks), 2);
    assert_eq!(counts.dropped(DropReason::SelfReferencingHashes), 2);
}

#[test]
fn bigram_pages_fold_to_their_tokens() {
    let (batches, normalizer) = normalize_all(vec![one(vec![with(
        store(&[1], None, &[]),
        "token_ids",
        json!([[1, 2], [2, 3], [3, 4], [4, 5]]),
    )])]);
    let block = &stored(&batches[0].events[0]).blocks[0];
    assert_eq!(block.token_ids, vec![1, 2, 3, 4]);
    assert_eq!(block.block_size, 4);
    assert_eq!(normalizer.counts().bigram_stores, 1);
    assert_eq!(normalizer.counts().forwarded_stored, 1);
}

#[test]
fn stores_and_removals_are_forwarded_one_for_one() {
    let (batches, normalizer) = normalize_all(vec![
        one(vec![
            store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            store(&[2, 3], Some(1), &[5, 6, 7, 8, 9, 10, 11, 12]),
        ]),
        one(vec![
            remove(&[1]),
            remove(&[1]),
            remove(&[1, 2]),
            store(&[1], None, &[1, 2, 3, 4]),
        ]),
    ]);
    assert_eq!(batches[0].events.len(), 3);
    assert_eq!(batches[1].events.len(), 4);
    for event in &batches[1].events[..2] {
        assert_eq!(removed(event).block_hashes, vec![1]);
    }
    assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![1, 2]);
    let counts = normalizer.counts();
    assert_eq!(counts.forwarded_stored, 4);
    assert_eq!(counts.forwarded_removed, 3);
    assert_eq!(
        counts.duplicate_stores, 1,
        "only the exact resend; a re-store after removal is new"
    );
    assert!(counts.dropped.is_empty());
}

#[test]
fn namespaces_come_from_the_event_its_extra_keys_or_its_parent() {
    // (A prompt-embeddings digest is msgpack bin, which JSON cannot
    // express; the generated fixtures cover it.)
    let lora = |value: Value| with(value, "lora_name", json!("adapter"));
    let (batches, _) = normalize_all(vec![
        one(vec![
            // vLLM: the salt rides in block 0's extra keys, after the LoRA
            // name and the multimodal (identifier, offset) pairs.
            with(
                lora(store(&[1], None, &[1, 2, 3, 4])),
                "extra_keys",
                json!([["adapter", ["mm-abc", 0], "salt-1"]]),
            ),
            with(
                lora(store(&[2], Some(1), &[5, 6, 7, 8])),
                "extra_keys",
                json!([["adapter"]]),
            ),
            store(&[3], Some(2), &[9, 10, 11, 12]),
            // SGLang: the salt is a field; empty strings count as absent.
            with(
                store(&[4], None, &[1, 2, 3, 4]),
                "cache_salt",
                json!("tenant-a"),
            ),
            with(
                with(store(&[5], None, &[1, 2, 3, 4]), "cache_salt", json!("")),
                "lora_name",
                json!(""),
            ),
        ]),
        // Another rank does not inherit from this one.
        json!([1700000001.0, [store(&[6], Some(3), &[13, 14, 15, 16])], 1]),
    ]);
    let events = &batches[0].events;
    let first = stored(&events[0]);
    assert_eq!(first.lora_name.as_deref(), Some("adapter"));
    assert_eq!(first.cache_salt.as_deref(), Some("salt-1"));
    let keys: Vec<_> = first.blocks[0]
        .extra_keys
        .iter()
        .map(|key| key.key.clone().expect("a key"))
        .collect();
    assert_eq!(
        keys,
        vec![
            kv_block_extra_key::Key::Text("adapter".into()),
            kv_block_extra_key::Key::Multimodal(common::KvMultimodalKey {
                identifier: "mm-abc".into(),
                offset: 0,
            }),
            kv_block_extra_key::Key::Text("salt-1".into()),
        ]
    );
    let child = stored(&events[1]);
    assert_eq!(child.lora_name.as_deref(), Some("adapter"));
    assert_eq!(
        child.cache_salt.as_deref(),
        Some("salt-1"),
        "inherited from block 0"
    );
    let grandchild = stored(&events[2]);
    assert_eq!(grandchild.lora_name.as_deref(), Some("adapter"));
    assert_eq!(grandchild.cache_salt.as_deref(), Some("salt-1"));
    assert_eq!(stored(&events[3]).cache_salt.as_deref(), Some("tenant-a"));
    assert_eq!(stored(&events[3]).lora_name, None);
    assert_eq!(stored(&events[4]).cache_salt, None);
    assert_eq!(stored(&events[4]).lora_name, None);
    let other_rank = stored(&batches[1].events[0]);
    assert_eq!(other_rank.lora_name, None);
    assert_eq!(other_rank.cache_salt, None);
}

#[test]
fn a_clear_resets_its_rank_only() {
    let (batches, normalizer) = normalize_all(vec![
        one(vec![store(&[1], None, &[1, 2, 3, 4])]),
        json!([1700000001.0, [store(&[1], None, &[1, 2, 3, 4])], 1]),
        one(vec![
            json!({"type": "AllBlocksCleared"}),
            store(&[1], None, &[1, 2, 3, 4]),
        ]),
        json!([1700000003.0, [store(&[1], None, &[1, 2, 3, 4])], 1]),
    ]);
    assert!(matches!(
        batches[2].events[0].data,
        Some(kv_cache_event::Data::Cleared(_))
    ));
    assert_eq!(batches[2].events.len(), 2);
    assert_eq!(normalizer.counts().forwarded_cleared, 1);
    assert_eq!(
        normalizer.counts().duplicate_stores,
        1,
        "rank 1 kept its seen set"
    );
    assert_eq!(batches[3].dp_rank, Some(1));
}

#[test]
fn array_and_map_layouts_decode_alike() {
    let map = one(vec![
        with(
            with(
                store(&[1, 2], Some(7), &[1, 2, 3, 4, 5, 6, 7, 8]),
                "session_id",
                json!("req-1"),
            ),
            "lora_name",
            json!("adapter"),
        ),
        with(remove(&[1]), "locality", json!("LOCAL")),
        json!({"type": "AllBlocksCleared"}),
    ]);
    let array = json!([
        1700000000.5,
        [
            [
                "BlockStored",
                [1, 2],
                7,
                [1, 2, 3, 4, 5, 6, 7, 8],
                4,
                null,
                "GPU",
                "adapter",
                null,
                0,
                "full_attention",
                null,
                null,
                null,
                "req-1"
            ],
            ["BlockRemoved", [1], "GPU", 0, "LOCAL"],
            ["AllBlocksCleared"],
        ],
        0
    ]);
    let (from_map, _) = normalize_all(vec![map]);
    let (from_array, _) = normalize_all(vec![array]);
    assert_eq!(from_map, from_array);
    let first = stored(&from_map[0].events[0]);
    assert_eq!(first.session_id.as_deref(), Some("req-1"));
    assert_eq!(first.lora_name.as_deref(), Some("adapter"));
    assert_eq!(first.parent_block_hash, Some(7));
    assert_eq!(first.group_idx, Some(0));
    assert_eq!(
        removed(&from_map[0].events[1]).locality,
        Some(KvCacheLocality::Local as i32)
    );
}

#[test]
fn a_group_mask_after_the_layouts_last_slot_is_forwarded() {
    // A publisher that reports a boundary once for every group appends the
    // mask after the reference layout's last slot: the sixteenth of a store,
    // the seventh of a removal. It rides through to the gateway beside
    // group_idx, which names the lowest group the event is about; a map
    // layout carries it under its name.
    let array = json!([
        1700000000.5,
        [
            [
                "BlockStored",
                [1, 2],
                7,
                [1, 2, 3, 4, 5, 6, 7, 8],
                4,
                null,
                "GPU",
                null,
                null,
                0,
                "full_attention",
                null,
                null,
                null,
                null,
                3
            ],
            ["BlockRemoved", [1], "GPU", 0, null, null, 2],
        ],
        0
    ]);
    let map = one(vec![
        with(
            with(
                store(&[1, 2], Some(7), &[1, 2, 3, 4, 5, 6, 7, 8]),
                "group_idx",
                json!(0),
            ),
            "group_mask",
            json!(3),
        ),
        with(
            with(remove(&[1]), "group_idx", json!(0)),
            "group_mask",
            json!(2),
        ),
    ]);
    let (from_array, _) = normalize_all(vec![array]);
    let (from_map, _) = normalize_all(vec![map]);
    assert_eq!(from_array, from_map);
    let first = stored(&from_array[0].events[0]);
    assert_eq!(first.group_idx, Some(0));
    assert_eq!(first.group_mask, Some(3));
    assert_eq!(removed(&from_array[0].events[1]).group_mask, Some(2));

    // A publisher that reports each group on its own carries no mask.
    let (plain, _) = normalize_all(vec![one(vec![store(&[1], None, &[1, 2, 3, 4])])]);
    assert_eq!(stored(&plain[0].events[0]).group_mask, None);
}

#[test]
fn legacy_arrays_keep_their_trailing_slots_in_order() {
    // ownership is the removal's sixth slot and must gate the event even
    // when locality (the fifth) is set.
    let (batches, normalizer) = normalize_all(vec![json!([
        1700000000.5,
        [
            ["BlockRemoved", [1], "STORAGE", 0, "LOCAL", "kvcr"],
            ["BlockRemoved", [2], "STORAGE", 0, "REMOTE"],
            ["BlockRemoved", [3], "GPU"],
        ],
        0
    ])]);
    assert_eq!(batches[0].events.len(), 1);
    assert_eq!(removed(&batches[0].events[0]).block_hashes, vec![3]);
    assert_eq!(
        normalizer
            .counts()
            .dropped(DropReason::UnsupportedOwnership),
        1
    );
    assert_eq!(normalizer.counts().dropped(DropReason::NonLocalLocality), 1);
}

#[test]
fn hash_check_verifies_sglang_chains_and_counts_mismatches() {
    use crate::engine_hash::{sglang_chain, sglang_salt_seed};

    let chain = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
    let (first, second) = (chain[0].1, chain[1].1);
    assert_eq!(first, -3488128144981237669);
    let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
    let mut event_id = 0;
    let batch = normalize(
        &mut normalizer,
        one(vec![
            store(&[first], None, &[1, 2, 3, 4]),
            store(&[second], Some(first), &[5, 6, 7, 8]),
            // Tampered: still forwarded, counted.
            store(&[second + 1], Some(first), &[5, 6, 7, 8]),
            // A parent this stream never saw: nothing to chain on.
            store(&[99], Some(12345), &[9, 10, 11, 12]),
        ]),
        1,
        &mut event_id,
    );
    assert_eq!(batch.events.len(), 4, "a mismatch never drops");
    let counts = normalizer.counts();
    assert_eq!(
        (
            counts.hash_checked,
            counts.hash_mismatch,
            counts.hash_unverifiable
        ),
        (3, 1, 1)
    );

    // A salted request seeds its chain; a coalesced two-page store
    // chains page to page.
    let seed = sglang_salt_seed("tenant-a");
    let salted = sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, Some(&seed));
    let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
    normalize(
        &mut normalizer,
        one(vec![with(
            store(&[salted[0].1, salted[1].1], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            "cache_salt",
            json!("tenant-a"),
        )]),
        1,
        &mut 0,
    );
    let counts = normalizer.counts();
    assert_eq!((counts.hash_checked, counts.hash_mismatch), (2, 0));

    // An Eagle bigram page hashes both words of every pair.
    let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
    normalize(
        &mut normalizer,
        one(vec![with(
            store(&[-638950109823820341], None, &[]),
            "token_ids",
            json!([[1, 2], [2, 3], [3, 4], [4, 5]]),
        )]),
        1,
        &mut 0,
    );
    let counts = normalizer.counts();
    assert_eq!((counts.hash_checked, counts.hash_mismatch), (1, 0));
}

#[test]
fn hash_check_verifies_vllm_sha256_cbor_chains() {
    use crate::engine_hash::{vllm_block, vllm_event_int};

    // Vectors A and B from the reference run.
    let (a, b) = (-8885242862429187823i64, -3153830497298837583i64);
    let mut normalizer = Normalizer::with_hash_check(EngineHash::VllmSha256Cbor);
    let batch = normalize(
        &mut normalizer,
        one(vec![
            store(&[a, b], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            // A LoRA block: the adapter path is not in the event.
            with(
                store(&[7], Some(b), &[9, 10, 11, 12]),
                "lora_name",
                json!("adapter"),
            ),
            // Unaligned: dropped before the check runs.
            store(&[8], Some(b), &[9, 10, 11, 12, 13]),
        ]),
        1,
        &mut 0,
    );
    assert_eq!(batch.events.len(), 2);
    let counts = normalizer.counts();
    assert_eq!(
        (
            counts.hash_checked,
            counts.hash_mismatch,
            counts.hash_unverifiable
        ),
        (2, 0, 1)
    );

    // Multimodal, salt and prompt-embeddings keys rebuild the tagged hash
    // input (vectors E, F, G); the events carry the bytes JSON cannot.
    let embeds: Vec<u8> = (0..32).collect();
    let e = vllm_block(
        None,
        &[1, 2, 3, 4],
        Some(&[
            VllmExtraKey::Mm {
                identifier: "mm-abc".into(),
                offset: 0,
            },
            VllmExtraKey::CacheSalt("salt-1".into()),
            VllmExtraKey::PromptEmbeds(embeds.clone()),
        ]),
    );
    let f = vllm_block(
        Some(&e),
        &[5, 6, 7, 8],
        Some(&[VllmExtraKey::Mm {
            identifier: "mm-abc".into(),
            offset: -4,
        }]),
    );
    let g = vllm_block(Some(&f), &[9, 10, 11, 12], None);
    let stored = |hashes: Vec<i64>, parent: Option<i64>, tokens: Vec<u32>, keys| {
        WireEvent::BlockStored(WireStored {
            block_hashes: hashes.into_iter().map(BlockHash).collect(),
            parent_block_hash: parent.map(BlockHash),
            token_ids: WireTokens::Ids(tokens),
            block_size: 4,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            extra_keys: keys,
            tail: EventTail {
                medium: Some("GPU".into()),
                group_idx: Some(0),
                kv_cache_spec_kind: Some("full_attention".into()),
                ..EventTail::default()
            },
        })
    };
    let mut normalizer = Normalizer::with_hash_check(EngineHash::VllmSha256Cbor);
    let events = [
        stored(
            vec![vllm_event_int(&e)],
            None,
            vec![1, 2, 3, 4],
            Some(vec![Some(vec![
                ExtraKey::Multimodal {
                    identifier: "mm-abc".into(),
                    offset: 0,
                },
                ExtraKey::Text("salt-1".into()),
                ExtraKey::Blob(embeds),
            ])]),
        ),
        stored(
            vec![vllm_event_int(&f)],
            Some(vllm_event_int(&e)),
            vec![5, 6, 7, 8],
            Some(vec![Some(vec![ExtraKey::Multimodal {
                identifier: "mm-abc".into(),
                offset: -4,
            }])]),
        ),
        stored(
            vec![vllm_event_int(&g)],
            Some(vllm_event_int(&f)),
            vec![9, 10, 11, 12],
            None,
        ),
        // A key shape the hash input cannot be rebuilt from.
        stored(
            vec![5],
            Some(vllm_event_int(&g)),
            vec![13, 14, 15, 16],
            Some(vec![Some(vec![ExtraKey::Number(3)])]),
        ),
    ];
    for (index, event) in events.into_iter().enumerate() {
        assert!(normalizer
            .normalize(event, Some(0), index as u64 + 1)
            .is_ok());
    }
    let counts = normalizer.counts();
    assert_eq!(
        (
            counts.hash_checked,
            counts.hash_mismatch,
            counts.hash_unverifiable
        ),
        (3, 0, 1)
    );
    assert_eq!(stored_salt(&normalizer), Some("salt-1".to_string()));
}

fn stored_salt(normalizer: &Normalizer) -> Option<String> {
    normalizer
        .ranks
        .get(&0)
        .and_then(|rank| rank.tiers.get(&(KvCacheTier::Device as i32)))
        .and_then(|tier| tier.values().find_map(|record| record.namespace.clone()))
        .and_then(|namespace| namespace.cache_salt)
}

#[test]
fn hash_check_is_off_unless_asked() {
    assert_eq!(Normalizer::new().hash_check(), None);
    assert_eq!(
        Normalizer::with_hash_check(EngineHash::Sglang).hash_check(),
        Some(EngineHash::Sglang)
    );
    // Without the check, a wrong hash is nobody's business here.
    let (_, normalizer) = normalize_all(vec![one(vec![store(&[1], None, &[1, 2, 3, 4])])]);
    assert_eq!(normalizer.counts().hash_checked, 0);
}

/// A sliding-window group's store: the window's blocks against the whole
/// span the step computed.
fn window(hashes: &[i64], parent: Option<i64>, tokens: &[u32]) -> Value {
    let event = with(store(hashes, parent, tokens), "group_idx", json!(1));
    let event = with(event, "kv_cache_spec_kind", json!("sliding_window"));
    with(event, "kv_cache_spec_sliding_window", json!(128))
}

fn remove_in(hashes: &[i64], group: u32) -> Value {
    with(remove(hashes), "group_idx", json!(group))
}

/// A publisher that reports a boundary once for every group puts the group
/// mask on a window group's store too. The group lane forwards it as the
/// main lane does.
#[test]
fn a_group_lanes_store_carries_its_group_mask() {
    let (batches, _) = normalize_all(vec![one(vec![
        with(window(&[1], None, &[1, 2, 3, 4]), "group_mask", json!(2)),
        with(store(&[2], None, &[1, 2, 3, 4]), "group_mask", json!(3)),
    ])]);
    assert_eq!(batches[0].events.len(), 2, "{:?}", batches[0].events);
    let on_the_group_lane = stored(&batches[0].events[0]);
    assert_eq!(on_the_group_lane.group_idx, Some(1));
    assert_eq!(on_the_group_lane.group_mask, Some(2));
    assert!(
        on_the_group_lane.blocks[0].token_ids.is_empty(),
        "a group lane's store"
    );
    assert_eq!(stored(&batches[0].events[1]).group_mask, Some(3));
}

#[test]
fn two_groups_with_identical_hashes_keep_their_lanes_apart() {
    // vLLM's hybrid manager publishes every block in both groups under
    // the same hash, the window group's store first in the step and its
    // removals as the window moves on. The main group's events go through
    // whole; the window group's go through on their own lane, hashes only,
    // and never touch the relay's record of the main lane: the main group's
    // second store is still the one duplicate, and the window group's
    // removal takes no copy off the record.
    let (batches, normalizer) = normalize_all(vec![
        one(vec![
            window(&[2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
            store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
        ]),
        one(vec![
            remove_in(&[2], 1),
            store(&[1, 2], None, &[1, 2, 3, 4, 5, 6, 7, 8]), // the second copy
            remove_in(&[1], 0),
            remove_in(&[1], 0),
        ]),
    ]);
    assert_eq!(batches[0].events.len(), 2);
    let window_store = stored(&batches[0].events[0]);
    assert_eq!(window_store.group_idx, Some(1));
    assert_eq!(window_store.blocks.len(), 1);
    assert_eq!(window_store.blocks[0].block_hash, 2);
    assert!(window_store.blocks[0].token_ids.is_empty());
    let full = stored(&batches[0].events[1]);
    assert_eq!(
        full.blocks.iter().map(|b| b.block_hash).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(full.blocks[1].token_ids, vec![5, 6, 7, 8]);
    assert_eq!(full.group_idx, Some(0));
    assert_eq!(batches[1].events.len(), 4);
    let window_gone = removed(&batches[1].events[0]);
    assert_eq!(
        (window_gone.block_hashes.clone(), window_gone.group_idx),
        (vec![2], Some(1))
    );
    assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![1]);
    assert_eq!(removed(&batches[1].events[3]).block_hashes, vec![1]);
    let counts = normalizer.counts();
    assert_eq!((counts.group_stores, counts.group_removals), (1, 1));
    assert_eq!((counts.forwarded_stored, counts.forwarded_removed), (3, 3));
    assert_eq!(
        (
            counts.duplicate_stores,
            counts.window_only_stores,
            counts.tail_aligned_stores
        ),
        (1, 0, 0)
    );
    assert!(counts.dropped.is_empty(), "{:?}", counts.dropped);
}

#[test]
fn a_window_groups_hashes_take_the_tail_of_its_tokens() {
    // A rank with only a sliding-window group: a store spans the whole
    // range the step computed and names the window's last blocks, so the
    // hashes get the tail of the tokens, and the group rides along.
    let span: Vec<u32> = (1..=12).collect();
    let (batches, normalizer) = normalize_all(vec![one(vec![
        window(&[5], None, &span),
        window(&[6, 7], Some(5), &span),
        // A span that is not whole blocks is unreadable.
        window(&[8], Some(7), &(1..=13).collect::<Vec<u32>>()),
        // The window moved on without caching anything new.
        window(&[], Some(7), &(1..=16).collect::<Vec<u32>>()),
    ])]);
    assert_eq!(batches[0].events.len(), 2);
    let first = stored(&batches[0].events[0]);
    assert_eq!(first.blocks[0].token_ids, vec![9, 10, 11, 12]);
    assert_eq!(first.group_idx, Some(1));
    assert_eq!(first.kv_cache_spec_kind.as_deref(), Some("sliding_window"));
    assert_eq!(first.kv_cache_spec_sliding_window, Some(128));
    let second = stored(&batches[0].events[1]);
    assert_eq!(
        second
            .blocks
            .iter()
            .map(|b| b.token_ids.clone())
            .collect::<Vec<_>>(),
        vec![vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
    );
    assert_eq!(second.parent_block_hash, Some(5));
    let counts = normalizer.counts();
    assert_eq!(
        (counts.window_only_stores, counts.tail_aligned_stores),
        (2, 2)
    );
    assert_eq!(counts.dropped(DropReason::UnalignedBlocks), 1);
    assert_eq!(counts.dropped(DropReason::Placeholder), 1);
}

#[test]
fn a_window_only_rank_is_forwarded_until_a_main_group_appears() {
    let (batches, normalizer) = normalize_all(vec![
        one(vec![
            window(&[1], None, &[1, 2, 3, 4]),
            remove_in(&[1], 1),
            window(&[2], None, &[1, 2, 3, 4, 5, 6, 7, 8]),
        ]),
        one(vec![
            store(&[3], None, &[1, 2, 3, 4]), // a main-attention group shows up
            window(&[4], Some(3), &[1, 2, 3, 4, 5, 6, 7, 8]), // from here: the group lane
            remove_in(&[2], 1),               // forwarded while window-only: leaves the record too
            remove_in(&[9], 1),               // a group removal, forwarded as it came
            remove_in(&[2, 9], 1),
        ]),
    ]);
    assert_eq!(batches[0].events.len(), 3);
    let signal = stored(&batches[0].events[0]);
    assert_eq!(
        signal.blocks[0].token_ids,
        vec![1, 2, 3, 4],
        "the rank's main signal"
    );
    assert_eq!(batches[1].events.len(), 5, "{:?}", batches[1].events);
    let window_store = stored(&batches[1].events[1]);
    assert_eq!(window_store.group_idx, Some(1));
    assert_eq!(window_store.parent_block_hash, Some(3));
    assert!(
        window_store.blocks[0].token_ids.is_empty(),
        "once the rank has a main group the window group is a group lane"
    );
    assert_eq!(removed(&batches[1].events[2]).block_hashes, vec![2]);
    assert_eq!(removed(&batches[1].events[3]).block_hashes, vec![9]);
    assert_eq!(removed(&batches[1].events[4]).block_hashes, vec![2, 9]);
    let counts = normalizer.counts();
    assert_eq!(
        (counts.window_only_stores, counts.tail_aligned_stores),
        (2, 1)
    );
    assert_eq!((counts.group_stores, counts.group_removals), (1, 3));
    assert!(counts.dropped.is_empty(), "{:?}", counts.dropped);
}

#[test]
fn one_copy_removals_pass_through_uncollapsed() {
    // vLLM keeps up to two physical blocks per hash and publishes a
    // removal when one copy goes; the gateway counts copies, so both
    // stores and both removals must reach it as they are.
    let (batches, normalizer) = normalize_all(vec![one(vec![
        store(&[1], None, &[1, 2, 3, 4]),
        store(&[1], None, &[1, 2, 3, 4]),
        remove(&[1]),
        remove(&[1]),
    ])]);
    assert_eq!(batches[0].events.len(), 4);
    let counts = normalizer.counts();
    assert_eq!(
        (
            counts.forwarded_stored,
            counts.forwarded_removed,
            counts.duplicate_stores
        ),
        (2, 2, 1)
    );
}

/// The engine keeps up to two physical copies of a block and removes
/// them one at a time; the check's parent memory counts the copies the
/// way the gateway does, so a child stored after the first copy went
/// still chains on its parent's digest, and only after the last copy is
/// the parent gone and the child unverifiable. Both removals are
/// forwarded either way.
#[test]
fn the_hash_checks_parent_memory_counts_physical_copies() {
    let root = engine_hash::sglang_chain(&[1, 2, 3, 4], 4, None)[0].1;
    let child_b = engine_hash::sglang_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None)[1].1;
    let child_c = engine_hash::sglang_chain(&[1, 2, 3, 4, 13, 14, 15, 16], 4, None)[1].1;
    let mut normalizer = Normalizer::with_hash_check(EngineHash::Sglang);
    let mut event_id = 0;
    normalize(
        &mut normalizer,
        one(vec![
            store(&[root], None, &[1, 2, 3, 4]),
            store(&[root], None, &[1, 2, 3, 4]), // the second physical copy
            remove(&[root]),                     // one copy goes
            store(&[child_b], Some(root), &[5, 6, 7, 8]),
        ]),
        1,
        &mut event_id,
    );
    let counts = normalizer.counts();
    assert_eq!(
        (
            counts.hash_checked,
            counts.hash_mismatch,
            counts.hash_unverifiable,
            counts.duplicate_stores
        ),
        (3, 0, 0, 1)
    );
    normalize(
        &mut normalizer,
        one(vec![
            remove(&[root]), // the last copy
            store(&[child_c], Some(root), &[13, 14, 15, 16]),
        ]),
        2,
        &mut event_id,
    );
    let counts = normalizer.counts();
    assert_eq!((counts.hash_checked, counts.hash_unverifiable), (3, 1));
    assert_eq!((counts.forwarded_stored, counts.forwarded_removed), (4, 2));
}

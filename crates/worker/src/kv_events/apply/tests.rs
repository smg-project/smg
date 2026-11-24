use kv_index::{
    compute_content_hash, compute_request_content_hashes, salt::namespaced_request_content_hashes,
    ChainBlockMap, ChainIndex, ContentHash, GroupViews, OverlapScores, WorkerIdExhausted,
    XXH3_SEED,
};

use super::*;
use crate::common_proto::KvCacheCleared;

/// A chain index and the cache group views beside it.
struct KvIndex {
    chain: ChainIndex,
    groups: GroupViews,
}

impl KvIndex {
    fn chain() -> Self {
        Self {
            chain: ChainIndex::with_max_workers(1024),
            groups: GroupViews::new(),
        }
    }

    fn intern_worker(&self, url: &str) -> Result<u32, WorkerIdExhausted> {
        self.chain.intern_worker(url)
    }

    fn current_size(&self) -> usize {
        self.chain.current_size()
    }

    fn find_matches(&self, hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        self.chain.find_matches(hashes, early_exit)
    }

    fn groups(&self) -> &GroupViews {
        &self.groups
    }
}

impl BlockIndex for KvIndex {
    type Held = ChainBlockMap;

    fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut ChainBlockMap,
    ) -> Result<(), ApplyError> {
        self.chain.apply_stored(worker, blocks, parent, held)
    }

    fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut ChainBlockMap) {
        self.chain.apply_removed(worker, hashes, held);
    }

    fn apply_cleared(&self, worker: u32, held: &mut ChainBlockMap) {
        self.chain.apply_cleared(worker, held);
    }

    fn groups(&self, worker: u32) -> Arc<WorkerGroups> {
        self.groups.get_or_create(worker)
    }
}

impl HeldBlocks for ChainBlockMap {
    fn contains_key(&self, hash: SequenceHash) -> bool {
        Self::contains_key(self, hash)
    }

    fn len(&self) -> usize {
        Self::len(self)
    }

    fn hashes(&self) -> Vec<SequenceHash> {
        self.iter().map(|(hash, _)| hash).collect()
    }
}

#[test]
fn a_proto_block_becomes_its_engine_and_content_hashes() {
    let proto = block(42, &[1, 2, 3, 4]);
    let stored = convert_kv_block(&proto, XXH3_SEED);
    assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
    assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
}

#[test]
fn a_negative_engine_hash_keeps_its_bits() {
    let proto = block(-1, &[10, 20]);
    let stored = convert_kv_block(&proto, XXH3_SEED);
    assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
}

#[test]
fn a_block_without_tokens_hashes_the_empty_content() {
    let proto = block(100, &[]);
    let stored = convert_kv_block(&proto, XXH3_SEED);
    assert_eq!(stored.seq_hash, SequenceHash::from(100i64));
    assert_eq!(stored.content_hash, compute_content_hash(&[]));
}

#[test]
fn a_root_store_indexes_its_blocks() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let stored = KvBlocksStored {
        blocks: vec![block(1, &[10, 20, 30, 40]), block(2, &[50, 60, 70, 80])],
        parent_block_hash: None,
        ..Default::default()
    };

    apply_stored(&stored, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2);
}

#[test]
fn a_chained_store_extends_its_parent() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let stored1 = KvBlocksStored {
        blocks: vec![block(1, &[10, 20, 30, 40])],
        parent_block_hash: None,
        ..Default::default()
    };
    apply_stored(&stored1, w1, &indexer, &mut wb);

    let stored2 = KvBlocksStored {
        blocks: vec![block(2, &[50, 60, 70, 80])],
        parent_block_hash: Some(1),
        ..Default::default()
    };
    apply_stored(&stored2, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2);
}

#[test]
fn a_store_with_an_unknown_parent_starts_a_new_chain() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://new-worker:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    // Pass parent_block_hash for an untracked worker — should fallback to no parent.
    let stored = KvBlocksStored {
        blocks: vec![block(1, &[10, 20, 30, 40])],
        parent_block_hash: Some(999),
        ..Default::default()
    };
    apply_stored(&stored, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
}

#[test]
fn a_removal_takes_the_block_out() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let stored = KvBlocksStored {
        blocks: vec![block(1, &[10, 20, 30, 40]), block(2, &[50, 60, 70, 80])],
        parent_block_hash: None,
        ..Default::default()
    };
    apply_stored(&stored, w1, &indexer, &mut wb);

    let removed = KvBlocksRemoved {
        block_hashes: vec![2],
        cache_level: None,
        ..Default::default()
    };
    apply_removed(&removed, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
}

#[test]
fn a_clear_empties_the_worker() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let stored = KvBlocksStored {
        blocks: vec![block(1, &[10, 20, 30, 40])],
        parent_block_hash: None,
        ..Default::default()
    };
    apply_stored(&stored, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);

    apply_cleared(w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
}

#[test]
fn apply_event_dispatches_a_store() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let event = KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: vec![block(42, &[1, 2, 3, 4])],
            parent_block_hash: None,
            ..Default::default()
        })),
    };

    apply_event(&event, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
}

#[test]
fn apply_event_dispatches_a_removal() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let stored_event = KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: vec![block(1, &[1, 2, 3, 4])],
            parent_block_hash: None,
            ..Default::default()
        })),
    };
    apply_event(&stored_event, w1, &indexer, &mut wb);

    let removed_event = KvCacheEvent {
        event_id: 2,
        data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
            block_hashes: vec![1],
            cache_level: None,
            ..Default::default()
        })),
    };
    apply_event(&removed_event, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
}

#[test]
fn apply_event_dispatches_a_clear() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    apply_event(
        &KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![block(1, &[1, 2, 3, 4])],
                parent_block_hash: None,
                ..Default::default()
            })),
        },
        w1,
        &indexer,
        &mut wb,
    );

    // Clear
    apply_event(
        &KvCacheEvent {
            event_id: 2,
            data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
        },
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 0);
}

#[test]
fn an_event_without_data_is_ignored() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let event = KvCacheEvent {
        event_id: 1,
        data: None,
    };
    apply_event(&event, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
}

const TOKENS: [u32; 4] = [10, 20, 30, 40];

/// A block of these tokens at this hash, as wide as its tokens, the rest default.
fn block(hash: i64, tokens: &[u32]) -> KvBlock {
    KvBlock {
        block_hash: hash,
        token_ids: tokens.to_vec(),
        block_size: tokens.len() as i32,
        ..Default::default()
    }
}

fn stored_event(hash: i64, tokens: &[u32]) -> KvBlocksStored {
    KvBlocksStored {
        blocks: vec![block(hash, tokens)],
        ..Default::default()
    }
}

fn removed_event(hash: i64) -> KvBlocksRemoved {
    KvBlocksRemoved {
        block_hashes: vec![hash],
        ..Default::default()
    }
}

fn routable(indexer: &KvIndex, worker: u32, hashes: &[ContentHash]) -> bool {
    indexer
        .find_matches(hashes, false)
        .scores
        .get(&worker)
        .is_some_and(|&depth| depth > 0)
}

#[test]
fn salted_stores_match_only_their_namespace() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut stored = stored_event(1, &TOKENS);
    stored.lora_name = Some("adapter".to_string());
    stored.cache_salt = Some("tenant-a".to_string());
    apply_stored(&stored, w1, &indexer, &mut wb);

    let same = namespaced_request_content_hashes(&TOKENS, 4, Some("adapter"), Some("tenant-a"));
    assert!(routable(&indexer, w1, &same));
    let plain = compute_request_content_hashes(&TOKENS, 4);
    assert!(!routable(&indexer, w1, &plain));
    let lora_only = namespaced_request_content_hashes(&TOKENS, 4, Some("adapter"), None);
    assert!(!routable(&indexer, w1, &lora_only));
}

#[test]
fn device_removal_keeps_a_block_still_on_the_host() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    let mut on_host = stored_event(1, &TOKENS);
    on_host.tier = Some(KvCacheTier::Host as i32);
    apply_stored(&on_host, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);

    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(
        routable(&indexer, w1, &hashes),
        "the host copy keeps the block routable"
    );

    let mut from_host = removed_event(1);
    from_host.tier = Some(KvCacheTier::Host as i32);
    apply_removed(&from_host, w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &hashes));
    assert_eq!(indexer.current_size(), 0);
    assert!(wb.copies.is_empty());
}

#[test]
fn cache_level_stands_in_for_the_tier() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    let mut on_host = stored_event(1, &TOKENS);
    on_host.blocks[0].cache_level = Some(1);
    apply_stored(&on_host, w1, &indexer, &mut wb);
    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);

    let mut from_host = removed_event(1);
    from_host.cache_level = Some(1);
    apply_removed(&from_host, w1, &indexer, &mut wb);
    assert!(
        routable(&indexer, w1, &hashes),
        "the device copy keeps the block routable"
    );

    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &hashes));
}

#[test]
fn host_removal_without_a_host_copy_evicts_nothing() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    let mut from_host = removed_event(1);
    from_host.tier = Some(KvCacheTier::Host as i32);
    apply_removed(&from_host, w1, &indexer, &mut wb);
    assert!(routable(&indexer, w1, &hashes));
    assert_eq!(wb.counters.unknown_copy, 1);
}

#[test]
fn disk_and_external_tiers_are_counted_not_indexed() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let mut on_disk = stored_event(1, &TOKENS);
    on_disk.tier = Some(KvCacheTier::Disk as i32);
    apply_stored(&on_disk, w1, &indexer, &mut wb);
    let mut external = stored_event(2, &TOKENS);
    external.blocks[0].cache_level = Some(3);
    apply_stored(&external, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    let mut from_disk = removed_event(1);
    from_disk.tier = Some(KvCacheTier::Disk as i32);
    apply_removed(&from_disk, w1, &indexer, &mut wb);
    assert_eq!(
        indexer.current_size(),
        1,
        "a disk removal does not touch the device copy"
    );
    assert_eq!(wb.counters.untracked_tier, 3);
}

fn store_event(stored: KvBlocksStored) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(stored)),
    }
}

fn remove_event(removed: KvBlocksRemoved) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Removed(removed)),
    }
}

/// A store of `group` of `kind` naming `hashes` without tokens, the way
/// the relay forwards a group other than main attention.
fn group_store(
    group: u32,
    kind: Option<&str>,
    window: Option<u32>,
    hashes: &[i64],
) -> KvCacheEvent {
    store_event(KvBlocksStored {
        blocks: hashes
            .iter()
            .map(|&hash| KvBlock {
                block_hash: hash,
                token_ids: Vec::new(),
                block_size: 4,
                lora_id: None,
                cache_level: None,
                ..Default::default()
            })
            .collect(),
        parent_block_hash: None,
        group_idx: Some(group),
        kv_cache_spec_kind: kind.map(str::to_string),
        kv_cache_spec_sliding_window: window,
        ..Default::default()
    })
}

/// A kind-less event of a group (a removal never names its kind) is the
/// group's even when it comes earlier in the batch than the group's first
/// store, and even for a group past the view's bit limit: the main block
/// of the same hash stays indexed. A main store without tokens indexes
/// nothing.
#[test]
fn a_kind_less_group_event_never_reaches_the_main_index() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut gone = removed_event(1);
    gone.group_idx = Some(1);
    apply_events(
        &[
            store_event(stored_event(1, &TOKENS)),
            remove_event(gone),
            group_store(1, Some("sliding_window"), Some(5), &[1]),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 1, "the main block stays");
    assert_eq!(wb.counters.group_removals, 1);
    let view = indexer.groups().get(w1).expect("the view");
    assert_eq!(view.retained_blocks(), 1);

    // A group past the sixteenth bit: shown, never placed, and its
    // kind-less removal is still not the main group's.
    let mut second = stored_event(2, &[50, 60, 70, 80]);
    second.parent_block_hash = Some(1);
    apply_events(
        &[
            group_store(17, Some("mamba"), None, &[2]),
            store_event(second),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 2);
    let mut gone = removed_event(2);
    gone.group_idx = Some(17);
    apply_event(&remove_event(gone), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2, "the main block stays");
    assert_eq!(wb.counters.group_removals, 2);
    assert!(
        view.groups().iter().all(|group| group.index != 17),
        "a group without a bit is shown, never placed, never in the rule"
    );

    // A main store without tokens is counted, not indexed.
    let mut bare = stored_event(3, &[]);
    bare.blocks[0].token_ids.clear();
    apply_event(&store_event(bare), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2);
    assert_eq!(wb.counters.tokenless_stores, 1);
}

#[test]
fn remote_and_residency_agent_events_are_skipped() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();

    let mut remote = stored_event(1, &TOKENS);
    remote.locality = Some(KvCacheLocality::Remote as i32);
    apply_event(&store_event(remote), w1, &indexer, &mut wb);
    let mut agent = stored_event(1, &TOKENS);
    agent.ownership = Some("kvcr".to_string());
    apply_event(&store_event(agent), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    let cleared = KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Cleared(KvCacheCleared {
            ownership: Some("kvcr".to_string()),
        })),
    };
    apply_event(&cleared, w1, &indexer, &mut wb);
    assert_eq!(
        indexer.current_size(),
        1,
        "an agent's clear leaves the engine's blocks"
    );
    assert_eq!(wb.counters.remote, 1);
    assert_eq!(wb.counters.foreign_owner, 2);
}

#[test]
fn clearing_forgets_copies_and_residency() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    let mut on_host = stored_event(1, &TOKENS);
    on_host.tier = Some(KvCacheTier::Host as i32);
    apply_stored(&on_host, w1, &indexer, &mut wb);
    assert!(!wb.copies.is_empty());

    apply_cleared(w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
    assert!(wb.copies.is_empty());

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(
        !routable(&indexer, w1, &hashes),
        "no stale host bit survives a clear"
    );
}

#[test]
fn two_copies_need_two_removals() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    // An engine recomputes the last block of an exact resend into a second
    // physical copy with the same hash and removes the copies one by one.
    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    assert!(wb.copies.is_empty(), "a single device copy costs no entry");
    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    assert_eq!(wb.counters.duplicate_copies, 1);

    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(
        routable(&indexer, w1, &hashes),
        "the other copy is still cached"
    );
    assert_eq!(indexer.current_size(), 1);

    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &hashes));
    assert_eq!(indexer.current_size(), 0);
    assert!(wb.copies.is_empty());
}

#[test]
fn device_and_host_copies_are_counted_per_tier() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);
    let mut on_host = stored_event(1, &TOKENS);
    on_host.tier = Some(KvCacheTier::Host as i32);
    let mut from_host = removed_event(1);
    from_host.tier = Some(KvCacheTier::Host as i32);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_stored(&on_host, w1, &indexer, &mut wb);
    assert_eq!(
        wb.copies[&SequenceHash::from(1i64)],
        Copies { device: 2, host: 1 }
    );

    // A host removal consumes the host copy only.
    apply_removed(&from_host, w1, &indexer, &mut wb);
    assert!(routable(&indexer, w1, &hashes));
    // A second host removal has nothing to take and evicts nothing.
    apply_removed(&from_host, w1, &indexer, &mut wb);
    assert!(routable(&indexer, w1, &hashes));
    assert_eq!(wb.counters.unknown_copy, 1);

    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(routable(&indexer, w1, &hashes), "one device copy left");
    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &hashes));
}

#[test]
fn clearing_forgets_copy_counts() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_cleared(w1, &indexer, &mut wb);
    assert!(wb.copies.is_empty());

    apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(
        !routable(&indexer, w1, &hashes),
        "no count survives a clear"
    );
}

#[test]
fn copy_counts_are_capped() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let hashes = compute_request_content_hashes(&TOKENS, 4);

    // An engine's full report mode re-announces a hit chain on every
    // lookup without removals; the count stops at the cap.
    for _ in 0..20 {
        apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
    }
    assert_eq!(wb.copies[&SequenceHash::from(1i64)].device, COPIES_CAP);
    assert_eq!(wb.counters.capped_copies, 20 - u64::from(COPIES_CAP));

    for _ in 1..COPIES_CAP {
        apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    }
    assert!(routable(&indexer, w1, &hashes));
    apply_removed(&removed_event(1), w1, &indexer, &mut wb);
    assert!(
        !routable(&indexer, w1, &hashes),
        "the cap bounds the extra removals"
    );
}

fn stored_width(hash: i64, width: i32, tokens: &[u32], parent: Option<i64>) -> KvBlocksStored {
    KvBlocksStored {
        blocks: vec![KvBlock {
            block_hash: hash,
            token_ids: tokens.to_vec(),
            block_size: width,
            ..Default::default()
        }],
        parent_block_hash: parent,
        ..Default::default()
    }
}

/// A hybrid engine registers the prompt's tail inside its last block as a
/// prefix entry at its hash granularity: a one-block store narrower than
/// the block of record whose parent is an intermediate hash it never
/// publishes. It is counted and left out of the index, never a
/// parent-less store or a root chain of its own.
#[test]
fn a_sub_block_store_is_counted_and_not_indexed() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let first: Vec<u32> = (0..32).collect();
    apply_stored(&stored_width(1, 32, &first, None), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);

    // The tail entry: four tokens under a parent the chain never named.
    apply_stored(
        &stored_width(77, 4, &[1, 2, 3, 4], Some(555)),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        indexer.current_size(),
        1,
        "a sub-block entry is not indexed"
    );
    assert_eq!(wb.counters.sub_block_stores, 1);
    assert_eq!(wb.counters.sub_block_blocks, 1);
    assert_eq!(wb.counters.parentless_stores, 0, "not a parent-less store");
    assert_eq!(wb.counters.parentless_blocks, 0);

    // The chain goes on from the full block as if the entry never came.
    let second: Vec<u32> = (32..64).collect();
    apply_stored(
        &stored_width(2, 32, &second, Some(1)),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 2);
    assert_eq!(wb.counters.parentless_stores, 0);

    // The engine's eventual removal of the entry finds nothing to take out.
    apply_removed(
        &KvBlocksRemoved {
            block_hashes: vec![77],
            ..Default::default()
        },
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 2);
    assert_eq!(wb.counters.sub_block_stores, 1);
}

/// The block of record is the widest block the worker stored so far: a
/// narrower store before any wider one is indexed as it comes, a wider
/// store raises the record, and a width of zero says nothing either way.
#[test]
fn the_block_of_record_is_the_widest_block_stored() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_stored(
        &stored_width(1, 4, &[1, 2, 3, 4], None),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 1, "nothing to compare against yet");
    apply_stored(&stored_width(2, 0, &[5, 6], None), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2, "an unknown width is indexed");
    let full: Vec<u32> = (10..42).collect();
    apply_stored(&stored_width(3, 32, &full, None), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 3, "a wider block raises the record");
    apply_stored(
        &stored_width(4, 4, &[7, 8, 9, 10], Some(3)),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        indexer.current_size(),
        3,
        "narrower than the record: left out"
    );
    apply_stored(&stored_width(5, 0, &[11], Some(3)), w1, &indexer, &mut wb);
    assert_eq!(
        indexer.current_size(),
        4,
        "an unknown width is still indexed"
    );
    assert_eq!(wb.counters.sub_block_stores, 1);
    assert_eq!(wb.counters.sub_block_blocks, 1);
    assert_eq!(wb.counters.parentless_stores, 0);
}

/// A clear is the worker's new cache incarnation (its publisher restarted
/// or resynced): the block of record goes with the blocks, so a worker
/// that comes back with a narrower block indexes its stores as they come
/// instead of counting every one as a sub-block store.
#[test]
fn a_clear_resets_the_block_of_record() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let full: Vec<u32> = (10..42).collect();
    apply_stored(&stored_width(1, 32, &full, None), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
    apply_cleared(w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
    assert_eq!(
        wb.record_before_clear, 32,
        "kept for the first store's log line"
    );
    apply_stored(
        &stored_width(2, 4, &[1, 2, 3, 4], None),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        indexer.current_size(),
        1,
        "the restarted worker's narrower block is indexed, not left out"
    );
    assert_eq!(wb.counters.sub_block_stores, 0);
    assert_eq!(wb.counters.sub_block_blocks, 0);
    assert_eq!(wb.block_of_record, 4);
    assert_eq!(
        wb.record_before_clear, 0,
        "the first store after the clear settles it"
    );
}

/// A store of a window group that carries tokens, the way the relay
/// forwards a rank without a main-attention group.
fn window_only_store(group: u32, hash: i64, parent: Option<i64>, tokens: &[u32]) -> KvCacheEvent {
    store_event(KvBlocksStored {
        blocks: vec![block(hash, tokens)],
        parent_block_hash: parent,
        group_idx: Some(group),
        kv_cache_spec_kind: Some("sliding_window".to_string()),
        kv_cache_spec_sliding_window: Some(64),
        ..Default::default()
    })
}

/// A kind-less removal naming its group, as the engines publish them.
fn removal_in(group: u32, hash: i64) -> KvCacheEvent {
    let mut removed = removed_event(hash);
    removed.group_idx = Some(group);
    remove_event(removed)
}

fn overlap(indexer: &KvIndex, worker: u32, hashes: &[ContentHash]) -> u32 {
    indexer
        .find_matches(hashes, false)
        .scores
        .get(&worker)
        .copied()
        .unwrap_or(0)
}

const SECOND: [u32; 4] = [50, 60, 70, 80];
const THIRD: [u32; 4] = [90, 91, 92, 93];

/// A worker whose attention is all sliding window has no main group:
/// its window stores carry tokens and are the prefix index's, as a plain
/// worker's; no view is made for it; a kind-less removal of the group
/// leaves the index.
#[test]
fn a_window_only_workers_stores_and_removals_are_the_prefix_indexs() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let both: Vec<u32> = TOKENS.iter().chain(SECOND.iter()).copied().collect();
    let hashes = compute_request_content_hashes(&both, 4);
    let one = compute_request_content_hashes(&TOKENS, 4);

    apply_events(
        &[
            window_only_store(0, 1, None, &TOKENS),
            window_only_store(0, 2, Some(1), &SECOND),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(
        wb.groups.is_none(),
        "no view for a worker without a main group"
    );
    assert_eq!(indexer.current_size(), 2);
    let full = overlap(&indexer, w1, &hashes);
    assert!(full > 0 && full > overlap(&indexer, w1, &one));
    assert_eq!(
        (wb.counters.window_only_stores, wb.counters.group_stores),
        (2, 0)
    );

    apply_events(&[removal_in(0, 2)], w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
    assert_eq!(overlap(&indexer, w1, &hashes), overlap(&indexer, w1, &one));
    assert_eq!(
        (wb.counters.window_only_removals, wb.counters.group_removals),
        (1, 0)
    );
}

/// A store the index skips (a residency agent's, a remote tier's) shows
/// no group in a batch: a foreign main store ahead of a window-only
/// worker's store leaves that store in the prefix index, and a foreign
/// window store shows no group.
#[test]
fn a_foreign_store_in_a_batch_shows_no_group() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut agent_main = stored_event(7, &THIRD);
    agent_main.kv_cache_spec_kind = Some("full_attention".to_string());
    agent_main.group_idx = Some(1);
    agent_main.ownership = Some("kvcr".to_string());
    let mut remote_window = window_only_store(2, 8, None, &SECOND);
    if let Some(kv_cache_event::Data::Stored(stored)) = &mut remote_window.data {
        stored.locality = Some(KvCacheLocality::Remote as i32);
    }

    apply_events(
        &[
            store_event(agent_main),
            window_only_store(0, 1, None, &TOKENS),
            remote_window,
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(
        !wb.shown.main_shown,
        "a foreign main store shows no main group"
    );
    assert!(wb.groups.is_none(), "a foreign window store shows no group");
    assert_eq!(
        indexer.current_size(),
        1,
        "the window store is the prefix index's"
    );
    assert_eq!(
        (
            wb.counters.window_only_stores,
            wb.counters.group_stores,
            wb.counters.foreign_owner,
            wb.counters.remote,
        ),
        (1, 0, 1, 1)
    );

    // The group pass's own case: a worker whose main group has shown
    // gets a foreign window store, which would show that group and give
    // the worker a view; skipped, the worker keeps none.
    let w2 = indexer.intern_worker("http://w2:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut remote_window = window_only_store(2, 9, None, &THIRD);
    if let Some(kv_cache_event::Data::Stored(stored)) = &mut remote_window.data {
        stored.locality = Some(KvCacheLocality::Remote as i32);
    }
    apply_events(
        &[store_event(stored_event(1, &TOKENS)), remote_window],
        w2,
        &indexer,
        &mut wb,
    );
    assert!(wb.shown.main_shown, "the main store shows the main group");
    assert!(
        wb.groups.is_none(),
        "a foreign window store after the main group shows no group"
    );
    assert_eq!((wb.counters.group_stores, wb.counters.remote), (0, 1));
}

/// A relay snapshot replays the stores after a clear in one batch: the
/// batch reads as the stream did, and a removal after it still leaves
/// the index.
#[test]
fn a_window_only_workers_snapshot_replay_reads_as_its_stream() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let both: Vec<u32> = TOKENS.iter().chain(SECOND.iter()).copied().collect();
    let hashes = compute_request_content_hashes(&both, 4);
    apply_events(
        &[window_only_store(0, 1, None, &TOKENS)],
        w1,
        &indexer,
        &mut wb,
    );
    let cleared = KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
    };
    apply_events(
        &[
            cleared,
            window_only_store(0, 1, None, &TOKENS),
            window_only_store(0, 2, Some(1), &SECOND),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(wb.groups.is_none());
    assert_eq!(indexer.current_size(), 2);
    assert!(routable(&indexer, w1, &hashes));
    assert_eq!(wb.counters.window_only_stores, 3);

    apply_events(&[removal_in(0, 1), removal_in(0, 2)], w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
    assert!(!routable(&indexer, w1, &hashes));
    assert_eq!(wb.counters.window_only_removals, 2);
}

/// A worker whose first batches carried only a window group's stores
/// (its main signal) and then shows a main group: the index lets the
/// signal's blocks go and the main group's stores rebuild it, the window
/// group moves to the view, and its removals are the view's from then
/// on, never the main group's blocks'.
#[test]
fn a_main_group_shown_later_moves_the_window_group_to_the_view() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let one = compute_request_content_hashes(&TOKENS, 4);
    apply_events(
        &[window_only_store(0, 1, None, &TOKENS)],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(routable(&indexer, w1, &one));

    // The main lane's batch of the same block: the signal's copy goes,
    // the main group's comes, one copy counted, the window group known
    // to the view.
    let mut main = stored_event(1, &TOKENS);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    assert!(
        wb.groups.is_some(),
        "the main group makes the worker hybrid"
    );
    assert!(routable(&indexer, w1, &one));
    assert_eq!(indexer.current_size(), 1);
    assert_eq!(wb.counters.main_signal_dropped, 1);
    assert!(
        wb.copies.is_empty(),
        "the main group's copy is the only one"
    );

    // The window group's later store is the view's, and its removals
    // are the view's: the main group's blocks stand until the main
    // group lets them go.
    apply_events(
        &[group_store(0, Some("sliding_window"), Some(64), &[1])],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(wb.counters.group_stores, 1);
    let mut later = stored_event(4, &[90, 91, 92, 93]);
    later.group_idx = Some(1);
    later.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(later)], w1, &indexer, &mut wb);
    apply_events(&[removal_in(0, 1), removal_in(0, 4)], w1, &indexer, &mut wb);
    assert_eq!(
        indexer.current_size(),
        2,
        "the window group's removals leave the index alone"
    );
    assert_eq!(
        (wb.counters.window_only_removals, wb.counters.group_removals),
        (0, 2)
    );
    apply_events(&[removal_in(1, 1)], w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &one), "the main group let it go");
    assert_eq!(indexer.current_size(), 1);
}

/// Two groups gave a worker its main signal: whose block is whose is
/// unknown at the restart, so the view stays incomplete and the overlap
/// stands until a clear, counted as a late view is.
#[test]
fn two_signal_groups_leave_the_restarted_view_incomplete() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[
            window_only_store(0, 1, None, &TOKENS),
            window_only_store(2, 2, None, &SECOND),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    let mut main = stored_event(1, &TOKENS);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    let view = wb.groups.clone().expect("the restart makes the view");
    assert!(!view.is_complete(), "whose block is whose is unknown");
    assert_eq!(wb.counters.main_signal_dropped, 2);
    assert_eq!(
        wb.counters.group_view_late, 1,
        "on the record, as a late view"
    );
    assert_eq!(view.retained_blocks(), 0);
    assert!(wb.shown.knows_group(0) && wb.shown.knows_group(2));
}

/// The restart's wait is bounded. Hashes-only stores of the window group
/// before any main group fill the worker's view with blocks waiting for
/// a name; when the main group shows, the signal's own block finds the
/// wait full and is counted dropped, and the worker's counter reads the
/// view's count after the restart.
#[test]
fn a_signal_block_the_restarts_full_wait_refuses_counts_as_dropped() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[window_only_store(0, 1, None, &TOKENS)],
        w1,
        &indexer,
        &mut wb,
    );

    // Blocks the view has no name for wait in it, up to its bound: fill
    // the wait in batches until a batch adds nothing.
    const BATCH: i64 = 1 << 16;
    let waiting_store = |from: i64| {
        let hashes: Vec<i64> = (from..from + BATCH).collect();
        group_store(0, Some("sliding_window"), Some(64), &hashes)
    };
    apply_events(&[waiting_store(1_000)], w1, &indexer, &mut wb);
    let view = wb
        .groups
        .clone()
        .expect("the hashes-only store gives a view");
    let mut from = 1_000 + BATCH;
    let mut filled = false;
    for _ in 0..64 {
        let waiting = view.pending_blocks();
        apply_events(&[waiting_store(from)], w1, &indexer, &mut wb);
        from += BATCH;
        if view.pending_blocks() == waiting {
            filled = true;
            break;
        }
    }
    assert!(filled, "the wait has a bound");
    let full = view.pending_blocks();
    let dropped = view.dropped_blocks();
    assert!(dropped > 0, "the last batch found the wait full");
    assert_eq!(wb.counters.group_dropped, dropped);

    // The main group shows: the signal's block leaves the index and
    // asks the full wait for a place; refused, it is counted dropped,
    // and the counter reads the view.
    let mut main = stored_event(2, &SECOND);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    assert_eq!(wb.counters.main_signal_dropped, 1);
    assert!(view.is_complete(), "the restart made the view complete");
    assert_eq!(view.pending_blocks(), full, "the wait is still full");
    assert_eq!(
        view.dropped_blocks(),
        dropped + 1,
        "the signal's block found the wait full"
    );
    assert_eq!(
        wb.counters.group_dropped,
        view.dropped_blocks(),
        "the counter reads the view after the restart"
    );
}

/// Two groups gave a worker its main signal and their blocks left the
/// index before the main group showed: the restart finds no block under
/// the view, so the view is complete and no late view is counted.
#[test]
fn a_restart_over_no_blocks_counts_no_late_view() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[
            window_only_store(0, 1, None, &TOKENS),
            window_only_store(2, 2, None, &SECOND),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    apply_events(&[removal_in(0, 1), removal_in(2, 2)], w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0, "the signals' blocks left");
    assert_eq!(wb.counters.window_only_removals, 2);

    let mut main = stored_event(1, &TOKENS);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    let view = wb.groups.clone().expect("the restart makes the view");
    assert!(view.is_complete(), "nothing stood under the view");
    assert_eq!(wb.counters.main_signal_dropped, 0);
    assert_eq!(
        wb.counters.group_view_late, 0,
        "no late view over no blocks"
    );
    assert!(wb.shown.knows_group(0) && wb.shown.knows_group(2));
    assert_eq!(indexer.current_size(), 1, "the main group's block is in");
}

/// A window-only worker's second copy of a block (an engine's resend)
/// needs a second removal, as a main group's does.
#[test]
fn a_window_only_workers_second_copy_needs_a_second_removal() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let one = compute_request_content_hashes(&TOKENS, 4);
    apply_events(
        &[
            window_only_store(0, 1, None, &TOKENS),
            window_only_store(0, 1, None, &TOKENS),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(wb.counters.duplicate_copies, 1);
    apply_events(&[removal_in(0, 1)], w1, &indexer, &mut wb);
    assert!(routable(&indexer, w1, &one), "one copy remains");
    apply_events(&[removal_in(0, 1)], w1, &indexer, &mut wb);
    assert!(!routable(&indexer, w1, &one));
    assert_eq!(indexer.current_size(), 0);
    assert!(wb.copies.is_empty());
    assert_eq!(wb.counters.window_only_removals, 2);
}

/// A hybrid worker whose window stores carry tokens (an engine that
/// publishes every group's blocks with their tokens): the main group's
/// store in the same batch keeps them the view's.
#[test]
fn a_hybrid_workers_tokenful_window_store_stays_in_the_view() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut main = stored_event(1, &TOKENS);
    main.group_idx = Some(0);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(
        &[window_only_store(1, 1, None, &TOKENS), store_event(main)],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(wb.groups.is_some());
    assert_eq!(indexer.current_size(), 1);
    assert_eq!(
        (wb.counters.window_only_stores, wb.counters.group_stores),
        (0, 1)
    );
}

/// A store's width is its first block's, and a width of zero or less names
/// none.
#[test]
fn a_stores_width_is_its_first_blocks() {
    let store = |widths: &[i32]| KvBlocksStored {
        blocks: widths
            .iter()
            .map(|&block_size| KvBlock {
                block_size,
                ..KvBlock::default()
            })
            .collect(),
        ..KvBlocksStored::default()
    };
    let rows: [(&[i32], Option<u32>); 5] = [
        (&[], None),
        (&[0], None),
        (&[-16], None),
        (&[16], Some(16)),
        (&[16, 32], Some(16)),
    ];
    for (widths, width) in rows {
        assert_eq!(store_width(&store(widths)), width, "{widths:?}");
    }
}

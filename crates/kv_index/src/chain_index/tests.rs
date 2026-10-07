use super::{arena::*, slab::*, *};
use crate::reference::{request_prefix_hashes, ReferenceIndexer};

fn content(stream: u64, position: usize) -> ContentHash {
    crate::compute_content_hash(&[stream as u32, (stream >> 32) as u32, position as u32])
}

fn blocks_of(contents: &[ContentHash]) -> Vec<StoredBlock> {
    contents
        .iter()
        .zip(request_prefix_hashes(contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}

fn scores(index: &ChainIndex, query: &[ContentHash]) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = index
        .find_matches(query, false)
        .scores
        .into_iter()
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn array_classes_round_up_and_back() {
    for capacity in [1usize, 8, 9, 16, 100, 128, 129, 256, 257, 1000, 4096, 5000] {
        let class = array_class(capacity);
        assert!(class_capacity(class) >= capacity, "capacity {capacity}");
        assert_eq!(array_class(class_capacity(class)), class);
    }
    assert_eq!(table_class(2), 0);
    assert_eq!(table_class(4), 1);
    assert_eq!(table_class(1024), 9);
}

#[test]
fn store_lookup_and_divergence() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..10).map(|p| content(1, p)).collect();
    index
        .apply_stored(w, &blocks_of(&held), None, &mut map)
        .expect("store");
    assert_eq!(scores(&index, &held), vec![(w, 10)]);
    assert_eq!(scores(&index, &held[..4]), vec![(w, 4)]);
    let mut diverged = held.clone();
    diverged[5] = content(2, 0);
    assert_eq!(scores(&index, &diverged), vec![(w, 5)]);
    let mut extended = held.clone();
    extended.push(content(3, 0));
    assert_eq!(scores(&index, &extended), vec![(w, 10)]);
    assert_eq!(scores(&index, &[content(9, 0)]), vec![]);
    assert_eq!(index.current_size(), 10);
    assert_eq!(index.entry_count(), 10);
}

#[test]
fn two_workers_share_a_prefix_and_split_at_the_fork() {
    let index = ChainIndex::with_max_workers(8);
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let base: Vec<ContentHash> = (0..6).map(|p| content(1, p)).collect();
    let mut fork = base[..3].to_vec();
    fork.extend((0..4).map(|p| content(2, p)));
    index
        .apply_stored(a, &blocks_of(&base), None, &mut ma)
        .expect("store a");
    index
        .apply_stored(b, &blocks_of(&fork), None, &mut mb)
        .expect("store b");
    assert_eq!(scores(&index, &base), vec![(a, 6), (b, 3)]);
    assert_eq!(scores(&index, &fork), vec![(a, 3), (b, 7)]);
    let mut early: Vec<(u32, u32)> = index.find_matches(&base, true).scores.into_iter().collect();
    early.sort_unstable();
    assert_eq!(early, vec![(a, 1), (b, 1)]);
    let mut reference = ReferenceIndexer::new();
    reference
        .apply_stored(a, &blocks_of(&base), None)
        .expect("ref a");
    reference
        .apply_stored(b, &blocks_of(&fork), None)
        .expect("ref b");
    assert_eq!(index.debug_blocks(), reference.blocks());
    assert_eq!(index.entry_count(), 10);
}

#[test]
fn a_worker_holds_both_sides_of_its_own_divergence() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let base: Vec<ContentHash> = (0..6).map(|p| content(1, p)).collect();
    let mut fork = base[..3].to_vec();
    fork.extend((0..2).map(|p| content(2, p)));
    let blocks = blocks_of(&base);
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("base");
    let fork_blocks = blocks_of(&fork);
    index
        .apply_stored(w, &fork_blocks[3..], Some(blocks[2].seq_hash), &mut map)
        .expect("fork");
    assert_eq!(scores(&index, &base), vec![(w, 6)]);
    assert_eq!(scores(&index, &fork), vec![(w, 5)]);
    assert_eq!(index.current_size(), 8);
}

#[test]
fn a_hole_stops_the_match_at_the_hole() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..8).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("store");
    index.apply_removed(w, &[blocks[3].seq_hash], &mut map);
    assert_eq!(scores(&index, &held), vec![(w, 3)]);
    assert_eq!(index.current_size(), 7);
    // Re-storing the missing block after its parent heals the hole.
    index
        .apply_stored(w, &blocks[3..4], Some(blocks[2].seq_hash), &mut map)
        .expect("heal");
    assert_eq!(scores(&index, &held), vec![(w, 8)]);
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(w, &blocks, None).expect("ref");
    assert_eq!(index.debug_blocks(), reference.blocks());
}

#[test]
fn a_hole_in_a_shared_run_affects_only_the_evicting_worker() {
    let index = ChainIndex::with_max_workers(8);
    let v = index.intern_worker("v").expect("id");
    let w = index.intern_worker("w").expect("id");
    let (mut mv, mut mw) = (ChainBlockMap::default(), ChainBlockMap::default());
    let held: Vec<ContentHash> = (0..10).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index.apply_stored(v, &blocks, None, &mut mv).expect("v");
    index.apply_stored(w, &blocks, None, &mut mw).expect("w");
    index.apply_removed(w, &[blocks[5].seq_hash], &mut mw);
    assert_eq!(scores(&index, &held), vec![(v, 10), (w, 5)]);
    index.apply_removed(v, &[blocks[7].seq_hash, blocks[8].seq_hash], &mut mv);
    assert_eq!(scores(&index, &held), vec![(v, 7), (w, 5)]);
    index
        .apply_stored(w, &blocks[5..6], Some(blocks[4].seq_hash), &mut mw)
        .expect("heal");
    assert_eq!(scores(&index, &held), vec![(v, 7), (w, 10)]);
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(v, &blocks, None).expect("ref v");
    reference.apply_stored(w, &blocks, None).expect("ref w");
    reference.apply_removed(v, &[blocks[7].seq_hash, blocks[8].seq_hash]);
    assert_eq!(index.debug_blocks(), reference.blocks());
}

#[test]
fn tail_removal_truncates_and_a_clear_empties() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..8).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("store");
    let tail: Vec<SequenceHash> = blocks[5..].iter().map(|b| b.seq_hash).collect();
    index.apply_removed(w, &tail, &mut map);
    assert_eq!(scores(&index, &held), vec![(w, 5)]);
    assert_eq!(map.len(), 5);
    assert_eq!(index.entry_count(), 5);
    index.apply_cleared(w, &mut map);
    assert!(map.is_empty());
    assert_eq!(scores(&index, &held), vec![]);
    assert_eq!(index.current_size(), 0);
    assert_eq!(index.entry_count(), 0);
    assert!(index.debug_blocks().is_empty());
    let stats = index.stats();
    assert_eq!(stats.runs_live, 0);
    assert_eq!(stats.runs_free, 1, "the dead run waits for reuse");
    assert_eq!(
        stats.arena_free_bytes,
        stats.arena_bytes - 8,
        "every array and table is back in a free list"
    );
}

#[test]
fn dead_runs_and_arrays_are_reused() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    for round in 0..200u64 {
        let held: Vec<ContentHash> = (0..12).map(|p| content(10 + round, p)).collect();
        let blocks = blocks_of(&held);
        index
            .apply_stored(w, &blocks, None, &mut map)
            .expect("store");
        assert_eq!(scores(&index, &held), vec![(w, 12)]);
        let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.seq_hash).collect();
        index.apply_removed(w, &hashes, &mut map);
        assert_eq!(scores(&index, &held), vec![]);
    }
    let stats = index.stats();
    assert!(
        stats.runs_allocated <= 3,
        "runs were not recycled: {stats:?}"
    );
    assert!(
        stats.arena_bytes < 4096,
        "arena words were not recycled: {stats:?}"
    );
    assert_eq!(index.current_size(), 0);
}

#[test]
fn appends_reuse_the_array_until_another_worker_joins() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let v = index.intern_worker("v").expect("id");
    let (mut mw, mut mv) = (ChainBlockMap::default(), ChainBlockMap::default());
    let held: Vec<ContentHash> = (0..40).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index
        .apply_stored(w, &blocks[..4], None, &mut mw)
        .expect("first");
    for step in 1..10 {
        let from = step * 4;
        index
            .apply_stored(
                w,
                &blocks[from..from + 4],
                Some(blocks[from - 1].seq_hash),
                &mut mw,
            )
            .expect("extend");
    }
    assert_eq!(
        index.stats().runs_live,
        1,
        "decode extensions stay in one run"
    );
    assert_eq!(scores(&index, &held), vec![(w, 40)]);
    index
        .apply_stored(v, &blocks[..20], None, &mut mv)
        .expect("join");
    assert_eq!(scores(&index, &held), vec![(w, 40), (v, 20)]);
    assert_eq!(
        index.stats().runs_live,
        1,
        "a prefix holder joins as a partial holder, no split"
    );
    let more: Vec<ContentHash> = (0..3).map(|p| content(2, p)).collect();
    let mut long = held.clone();
    long.extend(more);
    let long_blocks = blocks_of(&long);
    index
        .apply_stored(w, &long_blocks[40..], Some(blocks[39].seq_hash), &mut mw)
        .expect("extend after join");
    assert_eq!(scores(&index, &long), vec![(w, 43), (v, 20)]);
    assert_eq!(index.stats().runs_live, 1, "the run is still w's own leaf");
}

#[test]
fn many_children_grow_the_table_and_stay_findable() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let prompt: Vec<ContentHash> = (0..3).map(|p| content(1, p)).collect();
    index
        .apply_stored(w, &blocks_of(&prompt), None, &mut map)
        .expect("prompt");
    let anchor = blocks_of(&prompt)[2].seq_hash;
    let mut chains = Vec::new();
    for branch in 0..300u64 {
        let mut chain = prompt.clone();
        chain.extend((0..2).map(|p| content(100 + branch, p)));
        index
            .apply_stored(w, &blocks_of(&chain)[3..], Some(anchor), &mut map)
            .expect("branch");
        chains.push(chain);
    }
    for chain in &chains {
        assert_eq!(scores(&index, chain), vec![(w, 5)]);
    }
    let mut unknown = prompt.clone();
    unknown.push(content(999, 0));
    assert_eq!(scores(&index, &unknown), vec![(w, 3)]);
    // Unlinking every other branch tombstones its slot; the rest stay findable.
    for chain in chains.iter().step_by(2) {
        let hashes: Vec<SequenceHash> = blocks_of(chain)[3..].iter().map(|b| b.seq_hash).collect();
        index.apply_removed(w, &hashes, &mut map);
    }
    for (branch, chain) in chains.iter().enumerate() {
        let expected = if branch % 2 == 0 { 3 } else { 5 };
        assert_eq!(
            scores(&index, chain),
            vec![(w, expected)],
            "branch {branch}"
        );
    }
    let mut reference = ReferenceIndexer::new();
    reference
        .apply_stored(w, &blocks_of(&prompt), None)
        .expect("ref prompt");
    for chain in chains.iter().skip(1).step_by(2) {
        reference
            .apply_stored(w, &blocks_of(chain)[3..], Some(anchor))
            .expect("ref branch");
    }
    assert_eq!(index.debug_blocks(), reference.blocks());
}

#[test]
fn tail_evictions_and_regrowth_do_not_split() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let v = index.intern_worker("v").expect("id");
    let (mut mw, mut mv) = (ChainBlockMap::default(), ChainBlockMap::default());
    let held: Vec<ContentHash> = (0..40).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index.apply_stored(w, &blocks, None, &mut mw).expect("w");
    index.apply_stored(v, &blocks, None, &mut mv).expect("v");
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(w, &blocks, None).expect("ref w");
    reference.apply_stored(v, &blocks, None).expect("ref v");
    // v evicts its tail twice: a cutoff, not a split.
    for keep in [30usize, 12] {
        let gone: Vec<SequenceHash> = blocks[keep..].iter().map(|b| b.seq_hash).collect();
        index.apply_removed(v, &gone, &mut mv);
        reference.apply_removed(v, &gone);
        assert_eq!(scores(&index, &held), vec![(w, 40), (v, keep as u32)]);
        assert_eq!(index.stats().runs_live, 1, "tail eviction to {keep}");
        assert_eq!(index.worker_block_count(v), keep);
        assert_eq!(index.entry_count(), 40);
    }
    // A decode extends v back to the end of the run: full holder again, still one run.
    index
        .apply_stored(v, &blocks[12..], Some(blocks[11].seq_hash), &mut mv)
        .expect("regrow");
    reference
        .apply_stored(v, &blocks[12..], Some(blocks[11].seq_hash))
        .expect("ref regrow");
    assert_eq!(scores(&index, &held), vec![(w, 40), (v, 40)]);
    assert_eq!(index.stats().runs_live, 1);
    assert_eq!(index.debug_blocks(), reference.blocks());
    // w evicts everything, v keeps a prefix: the run survives with one partial holder.
    let all: Vec<SequenceHash> = blocks.iter().map(|b| b.seq_hash).collect();
    index.apply_removed(w, &all, &mut mw);
    reference.apply_removed(w, &all);
    index.apply_removed(v, &all[25..], &mut mv);
    reference.apply_removed(v, &all[25..]);
    assert_eq!(scores(&index, &held), vec![(v, 25)]);
    assert_eq!(index.entry_count(), 25);
    assert_eq!(index.debug_blocks(), reference.blocks());
    let walked = index.score_into(&held, |c| c.0, false, |_, _| {});
    assert_eq!(walked, 1);
}

#[test]
fn a_staircase_of_prefix_holders_is_a_few_runs() {
    let index = ChainIndex::with_max_workers(128);
    let held: Vec<ContentHash> = (0..64).map(|p| content(3, p)).collect();
    let blocks = blocks_of(&held);
    let mut maps: Vec<ChainBlockMap> = (0..64).map(|_| ChainBlockMap::default()).collect();
    let mut reference = ReferenceIndexer::new();
    for step in 0..64usize {
        assert_eq!(index.intern_worker(&format!("w{step}")), Ok(step as u32));
    }
    // The longest holder stores first (a full prefill); every shorter prefix then joins as a
    // partial holder, the way evictions and cache hits shape a shared prompt.
    for step in (0..64usize).rev() {
        index
            .apply_stored(step as u32, &blocks[..=step], None, &mut maps[step])
            .expect("store");
        reference
            .apply_stored(step as u32, &blocks[..=step], None)
            .expect("ref");
    }
    // A divergence never splits, but more than `PARTIAL_CAP` prefix holders on one run do,
    // at their median cutoff: 64 prefixes of one chain end up in a few runs with the longer
    // holders whole on the prefixes, never in one run per holder.
    let runs = index.stats().runs_live;
    assert!(
        (2..=2 * 64 / PARTIAL_CAP).contains(&runs),
        "64 prefixes of one chain under the prefix-holder cap of {PARTIAL_CAP}: {runs} runs"
    );
    assert!(index.stats().splits_by_prefix_holders >= 1);
    let expected: Vec<(u32, u32)> = (0..64u32).map(|w| (w, w + 1)).collect();
    assert_eq!(scores(&index, &held), expected);
    assert_eq!(index.debug_blocks(), reference.blocks());
    let walked = index.score_into(&held, |c| c.0, false, |_, _| {});
    assert_eq!(walked, runs, "the walk visits each run of the chain once");
    let mut early: Vec<(u32, u32)> = index.find_matches(&held, true).scores.into_iter().collect();
    early.sort_unstable();
    assert_eq!(early, (0..64u32).map(|w| (w, 1)).collect::<Vec<_>>());
    // A hole in the longest holder's prefix still splits, and only for it.
    index.apply_removed(63, &[blocks[40].seq_hash], &mut maps[63]);
    reference.apply_removed(63, &[blocks[40].seq_hash]);
    assert_eq!(scores(&index, &held)[63], (63, 40));
    assert!(index.stats().runs_live >= runs);
    assert_eq!(index.debug_blocks(), reference.blocks());
}

/// A lock-free reader may still hold a table id whose words were recycled and rewritten by
/// the time it reads them; the version check discards the read, so the read itself must only
/// return garbage, never panic. Rewrites the headers a recycled slot could carry: a child
/// table that became a forwards table of another size, and a partial table whose used count
/// outgrew its slots.
#[test]
fn stale_table_headers_are_read_without_panicking() {
    let arena = WordArena::new();
    let child_table = arena.alloc_table(MIN_TABLE_SLOTS);
    arena.table_put(child_table, 0xabcd, 7, 1);
    let partial_table = arena.alloc_partials(MIN_TABLE_SLOTS);
    arena.partial_put(partial_table, 3, 5);
    let forwards_table = arena.alloc_forwards(MIN_TABLE_SLOTS);
    arena.forwards_push(forwards_table, 4, 9, 2);
    // Garbage of every shape a recycled header might show: zero, not a power of two, huge.
    for garbage in [
        0u64,
        3,
        u64::MAX,
        (u64::MAX << 32) | 5,
        (7u64 << 32) | (1 << 31),
    ] {
        for table in [child_table, partial_table, forwards_table] {
            arena.word(table).store(garbage, Ordering::Relaxed);
            let _ = arena.table_find(table, 0xabcd);
            let _ = arena.forwards_find(table, 4);
            let _ = arena.partial_find(table, 3);
            let _ = arena.partial_max(table);
            let (_, used) = arena.partials_shape(table);
            let _ = arena.words(table + 2, used).len();
        }
    }
    // A whole walk over an index keeps working after the headers it reads are rewritten.
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..12).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("store");
    assert_eq!(scores(&index, &held), vec![(w, 12)]);
}

/// The gateway's parent-missing fallback: a worker holds b0..b3, evicts b2 and b3, the
/// engine extends after b3 but the parent is unknown, so b4 and b5 are stored without a
/// parent and land at positions 0 and 1 under the hashes of positions 4 and 5; when the
/// engine then announces the whole chain, each hash is held at one place only, the old
/// memberships are released, the counts read six blocks, a query for the mislaid pair
/// scores nothing, and a worker interned later into the freed id inherits nothing.
#[test]
fn a_hash_stored_again_at_another_position_releases_its_old_place() {
    let index = ChainIndex::with_max_workers(8);
    let mut reference = ReferenceIndexer::new();
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let chain: Vec<ContentHash> = (0..6).map(|p| content(23, p)).collect();
    let blocks = blocks_of(&chain);
    index
        .apply_stored(w, &blocks[..4], None, &mut map)
        .expect("b0..b3");
    reference.apply_stored(w, &blocks[..4], None).expect("ref");
    let evicted = [blocks[2].seq_hash, blocks[3].seq_hash];
    index.apply_removed(w, &evicted, &mut map);
    reference.apply_removed(w, &evicted);
    // The fallback: b4 and b5 with no parent, so at positions 0 and 1.
    index
        .apply_stored(w, &blocks[4..], None, &mut map)
        .expect("fallback");
    reference
        .apply_stored(w, &blocks[4..], None)
        .expect("ref fallback");
    assert_eq!(scores(&index, &chain[4..]), vec![(w, 2)]);
    assert_eq!(index.worker_block_count(w), 4);
    // The engine announces the chain whole.
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("whole");
    reference.apply_stored(w, &blocks, None).expect("ref whole");
    assert_eq!(index.worker_block_count(w), 6);
    assert_eq!(index.current_size(), 6);
    assert_eq!(index.entry_count(), 6);
    assert_eq!(index.stats().moved_hashes, 2);
    assert_eq!(scores(&index, &chain), vec![(w, 6)]);
    assert_eq!(scores(&index, &chain[4..]), vec![]);
    assert_eq!(index.debug_blocks(), reference.blocks());
    assert_eq!(reference.find_matches(&chain[4..]).len(), 0);
    // The id is freed and reused: nothing is inherited.
    index.remove_worker(w, map);
    assert!(index.is_empty());
    let v = index.intern_worker("v").expect("id");
    assert_eq!(v, w);
    assert_eq!(scores(&index, &chain), vec![]);
    assert_eq!(index.worker_block_count(v), 0);
}

/// A worker that stores its chain again beside another worker's fork off it re-stores every
/// block at the same place: nothing moves, nothing is released.
#[test]
fn a_re_store_beside_a_fork_moves_nothing() {
    let index = ChainIndex::with_max_workers(8);
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let chain: Vec<ContentHash> = (0..30).map(|p| content(24, p)).collect();
    let blocks = blocks_of(&chain);
    index.apply_stored(a, &blocks, None, &mut ma).expect("a");
    let mut fork = chain[..12].to_vec();
    fork.extend((12..20).map(|p| content(25, p)));
    index
        .apply_stored(b, &blocks_of(&fork), None, &mut mb)
        .expect("b");
    // a re-stores the whole chain and then a tail after its own parent.
    index
        .apply_stored(a, &blocks, None, &mut ma)
        .expect("a again");
    index
        .apply_stored(a, &blocks[20..], Some(blocks[19].seq_hash), &mut ma)
        .expect("a tail");
    assert_eq!(index.stats().moved_hashes, 0);
    assert_eq!(index.worker_block_count(a), 30);
    assert_eq!(scores(&index, &chain), vec![(a, 30), (b, 12)]);
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(a, &blocks, None).expect("ref a");
    reference
        .apply_stored(b, &blocks_of(&fork), None)
        .expect("ref b");
    assert_eq!(index.debug_blocks(), reference.blocks());
}

/// A split by another lane can land between a store's placement, recorded under the run's
/// lock, and its lane-map write. The write then meets a recorded place that forwards to the
/// suffix, exactly as the map's old entry for the block does: not a move, nothing released.
/// Driven through the write phase directly, with the placement recorded before the split
/// (a hole another holder opens; a divergence hangs a child and splits nothing).
#[test]
fn a_placement_recorded_before_a_split_is_not_a_move() {
    let index = ChainIndex::with_max_workers(8);
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let chain: Vec<ContentHash> = (0..8).map(|p| content(26, p)).collect();
    let blocks = blocks_of(&chain);
    index.apply_stored(a, &blocks, None, &mut ma).expect("a");
    index.apply_stored(b, &blocks, None, &mut mb).expect("b");
    let first = ma.get(blocks[0].seq_hash).expect("mapped");
    // b drops block 1: a hole, the run is split at 2. a's map keeps its entries in the
    // original run's coordinates, forwarded to the suffix that holds blocks 2..8 now.
    index.apply_removed(b, &[blocks[1].seq_hash], &mut mb);
    let (suffix, _) = index
        .resolve(BlockRef {
            run: first.run,
            offset: 2,
        })
        .expect("forwarded");
    assert_ne!(suffix.run, first.run);
    assert_eq!(suffix.offset, 0);
    // What a's re-store of the chain records at this point: blocks 0..2 in the prefix,
    // blocks 2..8 in the suffix.
    let pending = vec![
        Placed {
            run: first.run,
            offset: 0,
            start: 0,
            count: 2,
            conflicts: 0,
        },
        Placed {
            run: suffix.run,
            offset: 0,
            start: 2,
            count: 6,
            conflicts: 0,
        },
    ];
    // Before the write lands, b drops block 4: the suffix is split at 3.
    index.apply_removed(b, &[blocks[4].seq_hash], &mut mb);
    assert_ne!(
        index
            .resolve(BlockRef {
                run: suffix.run,
                offset: 3,
            })
            .expect("forwarded again")
            .0
            .run,
        suffix.run
    );
    index.write_placements(a, &blocks, pending, &mut ma);
    assert_eq!(index.stats().moved_hashes, 0);
    assert_eq!(index.worker_block_count(a), 8);
    assert_eq!(ma.len(), 8);
    assert_eq!(scores(&index, &chain), vec![(a, 8), (b, 1)]);
    // The entries resolve to places that credit a: a store under block 6 finds its parent.
    index
        .apply_stored(a, &blocks[7..], Some(blocks[6].seq_hash), &mut ma)
        .expect("a tail");
    assert_eq!(index.worker_block_count(a), 8);
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(a, &blocks, None).expect("ref a");
    reference.apply_stored(b, &blocks, None).expect("ref b");
    reference.apply_removed(b, &[blocks[1].seq_hash]);
    reference.apply_removed(b, &[blocks[4].seq_hash]);
    assert_eq!(index.debug_blocks(), reference.blocks());
}

/// An engine that names one position by two engine hashes (the same content under the same
/// parent: a twin) has the second filed onto the first by content, and the lane map carries
/// both names for one held position. Removing the first name takes the position with it;
/// a store under the second then points past what the worker holds, which cuts the run at
/// the parent and joins what lies beyond. When nothing lies beyond (nobody else holds the
/// tail, no child hangs there) there is nothing to join: the run ends at the cut and the
/// blocks go after it.
#[test]
fn a_store_under_a_twin_whose_first_name_was_removed_does_not_panic() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let chain: Vec<ContentHash> = (0..8).map(|p| content(29, p)).collect();
    let blocks = blocks_of(&chain);
    index
        .apply_stored(w, &blocks, None, &mut map)
        .expect("chain");
    // Twins of positions 4..8: the same content under the same parent, other engine hashes.
    let twins: Vec<StoredBlock> = blocks[4..]
        .iter()
        .enumerate()
        .map(|(i, b)| StoredBlock {
            seq_hash: SequenceHash(b.seq_hash.0 ^ (0x5151 << (i + 8))),
            content_hash: b.content_hash,
        })
        .collect();
    index
        .apply_stored(w, &twins, Some(blocks[3].seq_hash), &mut map)
        .expect("twins");
    assert_eq!(index.stats().engine_conflicts, 4);
    assert_eq!(index.worker_block_count(w), 8);
    assert_eq!(map.len(), 12);
    // The first names of positions 4..8 go: the positions go with them.
    let first_names: Vec<SequenceHash> = blocks[4..].iter().map(|b| b.seq_hash).collect();
    index.apply_removed(w, &first_names, &mut map);
    assert_eq!(index.worker_block_count(w), 4);
    assert_eq!(scores(&index, &chain), vec![(w, 4)]);
    // A store under the twin at position 5: the parent entry points past the holding.
    let tail = blocks_of(&[
        chain[0],
        chain[1],
        chain[2],
        chain[3],
        chain[4],
        chain[5],
        content(30, 6),
    ]);
    let stored = index.apply_stored(w, &tail[6..], Some(twins[1].seq_hash), &mut map);
    assert!(stored.is_ok(), "{stored:?}");
    assert_eq!(index.worker_block_count(w), 5);
    // Positions 0..4 and the new block at 6 are held; positions 4 and 5 are not (the
    // documented twin gap: one engine hash per held position).
    assert_eq!(scores(&index, &chain[..4]), vec![(w, 4)]);
    let query: Vec<ContentHash> = tail.iter().map(|b| b.content_hash).collect();
    assert_eq!(scores(&index, &query), vec![(w, 4)]);
    assert!(index
        .debug_blocks()
        .iter()
        .any(|b| b.0 == w && b.1 == 6 && b.2 == content(30, 6)));
    // Everything the worker holds is still reachable and consistent.
    assert_eq!(index.debug_blocks().len(), 5);
}
/// `is_empty` follows the blocks: false from the first store, true again once every block
/// is removed, cleared or taken with its worker.
#[test]
fn emptiness_follows_the_blocks() {
    let index = ChainIndex::with_max_workers(8);
    assert!(index.is_empty());
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let chain: Vec<ContentHash> = (0..12).map(|p| content(21, p)).collect();
    let blocks = blocks_of(&chain);
    index.apply_stored(a, &blocks, None, &mut ma).expect("a");
    assert!(!index.is_empty());
    index
        .apply_stored(b, &blocks[..6], None, &mut mb)
        .expect("b");
    let hashes: Vec<SequenceHash> = blocks.iter().map(|block| block.seq_hash).collect();
    index.apply_removed(a, &hashes, &mut ma);
    assert!(!index.is_empty(), "b still holds a prefix");
    index.apply_cleared(b, &mut mb);
    assert!(index.is_empty());
    index
        .apply_stored(a, &blocks[..3], None, &mut ma)
        .expect("a again");
    assert!(!index.is_empty());
    index.remove_worker(a, ma);
    assert!(index.is_empty());
}

/// A content array recycled from a larger class gets an engine-hash twin of at least that
/// capacity, so an in-place append (which claims room from the content header and writes
/// both arrays) never runs past the twin into the words behind it.
#[test]
fn the_engine_twin_is_as_large_as_a_recycled_content_array() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut mw = ChainBlockMap::default();
    // One freed array two classes above what 140 blocks ask for: the content array takes
    // it, the twin must not come out smaller.
    let big = index.arena.alloc_array(&[0; 200], 200);
    index.arena.array_release(big);
    let chain: Vec<ContentHash> = (0..200).map(|p| content(9, p)).collect();
    let blocks = blocks_of(&chain);
    index
        .apply_stored(w, &blocks[..140], None, &mut mw)
        .expect("first store");
    let run_id = mw.get(blocks[0].seq_hash).expect("mapped").run;
    let run = index.slab.run(run_id);
    let (block, engine) = (
        run.block.load(Ordering::Relaxed),
        run.engine.load(Ordering::Relaxed),
    );
    assert_eq!(block, big, "the recycled array was taken");
    assert!(
        index.arena.array_capacity(engine) >= index.arena.array_capacity(block),
        "twin {} words, content {}",
        index.arena.array_capacity(engine),
        index.arena.array_capacity(block)
    );
    // Words bumped right behind the twin: an overflowing append would land here.
    let guard = index.arena.alloc_array(&[7; 8], 8);
    index
        .apply_stored(w, &blocks[140..], Some(blocks[139].seq_hash), &mut mw)
        .expect("append");
    assert_eq!(run.len(), 200, "appended in place");
    assert_eq!(
        run.block.load(Ordering::Relaxed),
        block,
        "same content array"
    );
    for (slot, word) in index.arena.words(guard, 8).iter().enumerate() {
        assert_eq!(
            word.load(Ordering::Relaxed),
            7,
            "guard word {slot} overwritten"
        );
    }
    for block in &blocks {
        assert!(index.is_held(&mw, block.seq_hash), "{:?}", block.seq_hash);
    }
}

/// A run is matched by its engine chain: a store that agrees to the end of the window is
/// taken whole on one compare, one that diverges inside is split exactly where the bisection
/// lands, one that stops short is a prefix, one that runs past the end extends; the
/// reference agrees on every block and no counter moves.
#[test]
fn stores_match_a_run_by_its_engine_chain() {
    let index = ChainIndex::with_max_workers(8);
    let mut reference = ReferenceIndexer::new();
    let chain: Vec<ContentHash> = (0..100).map(|p| content(5, p)).collect();
    let mut forked = chain[..57].to_vec();
    forked.extend((57..100).map(|p| content(6, p)));
    let short = chain[..30].to_vec();
    let longer: Vec<ContentHash> = (0..140).map(|p| content(5, p)).collect();
    let mut early_fork = chain[..1].to_vec();
    early_fork.extend((1..40).map(|p| content(7, p)));
    let mut maps = Vec::new();
    for (name, contents) in [
        ("whole", &chain),
        ("forked", &forked),
        ("short", &short),
        ("longer", &longer),
        ("early", &early_fork),
    ] {
        let worker = index.intern_worker(name).expect("id");
        let mut map = ChainBlockMap::default();
        index
            .apply_stored(worker, &blocks_of(contents), None, &mut map)
            .expect("store");
        reference
            .apply_stored(worker, &blocks_of(contents), None)
            .expect("reference store");
        maps.push((worker, map));
    }
    for query in [&chain, &forked, &short, &longer, &early_fork] {
        let expected: Vec<(u32, u32)> = reference.find_matches(query).into_iter().collect();
        assert_eq!(scores(&index, query), expected);
    }
    assert_eq!(index.debug_blocks(), reference.blocks());
    // The run of the first 57 blocks is shared by four workers and the walk after the
    // divergence took the fork's own run: a decode extension of the fork appends to it.
    let (fork_worker, fork_map) = &mut maps[1];
    let more: Vec<ContentHash> = (100..110).map(|p| content(6, p)).collect();
    let mut extended = forked.clone();
    extended.extend(more.iter().copied());
    let blocks = blocks_of(&extended);
    index
        .apply_stored(
            *fork_worker,
            &blocks[100..],
            Some(blocks[99].seq_hash),
            fork_map,
        )
        .expect("extend");
    reference
        .apply_stored(*fork_worker, &blocks[100..], Some(blocks[99].seq_hash))
        .expect("reference extend");
    let expected: Vec<(u32, u32)> = reference.find_matches(&extended).into_iter().collect();
    assert_eq!(scores(&index, &extended), expected);
    let stats = index.stats();
    assert_eq!(stats.engine_conflicts, 0);
    assert_eq!(stats.landing_mismatches, 0);
}

/// A worker whose engine names the same content by other hashes cannot match by engine
/// hash; the walk sees the content continue past the mismatch, compares block by block and
/// counts the conflicts, and the worker's own hashes key its lane map: lookups, an extension
/// after its own parent hash and removals by its hashes all stay exact.
#[test]
fn a_worker_with_other_engine_hashes_matches_by_content() {
    let index = ChainIndex::with_max_workers(8);
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let chain: Vec<ContentHash> = (0..50).map(|p| content(8, p)).collect();
    let blocks = blocks_of(&chain);
    let other: Vec<StoredBlock> = blocks
        .iter()
        .map(|block| StoredBlock {
            seq_hash: SequenceHash(block.seq_hash.0 ^ 0x5bd1_e995),
            content_hash: block.content_hash,
        })
        .collect();
    index
        .apply_stored(a, &blocks[..40], None, &mut ma)
        .expect("a");
    index
        .apply_stored(b, &other[..40], None, &mut mb)
        .expect("b");
    assert_eq!(scores(&index, &chain), vec![(a, 40), (b, 40)]);
    assert_eq!(index.stats().engine_conflicts, 40);
    index
        .apply_stored(b, &other[40..], Some(other[39].seq_hash), &mut mb)
        .expect("b extends");
    assert_eq!(scores(&index, &chain), vec![(a, 40), (b, 50)]);
    let tail: Vec<SequenceHash> = other[20..].iter().map(|block| block.seq_hash).collect();
    index.apply_removed(b, &tail, &mut mb);
    assert_eq!(scores(&index, &chain), vec![(a, 40), (b, 20)]);
    for block in &other[20..] {
        assert!(!index.is_held(&mb, block.seq_hash));
    }
    for block in &other[..20] {
        assert!(index.is_held(&mb, block.seq_hash));
    }
    assert_eq!(index.stats().landing_mismatches, 0);
}

/// A store that carries the engine hashes the index holds for other content (an engine
/// whose hashes do not follow its content, which the relay's hash check refuses) is counted
/// and placed by its content: the match ends where the content does.
#[test]
fn other_content_under_known_engine_hashes_is_counted_and_placed_by_content() {
    let index = ChainIndex::with_max_workers(8);
    let a = index.intern_worker("a").expect("id");
    let b = index.intern_worker("b").expect("id");
    let (mut ma, mut mb) = (ChainBlockMap::default(), ChainBlockMap::default());
    let chain: Vec<ContentHash> = (0..20).map(|p| content(9, p)).collect();
    let blocks = blocks_of(&chain);
    let mut other = chain[..10].to_vec();
    other.extend((10..20).map(|p| content(10, p)));
    let impostor: Vec<StoredBlock> = blocks_of(&other)
        .into_iter()
        .zip(&blocks)
        .map(|(block, known)| StoredBlock {
            seq_hash: known.seq_hash,
            content_hash: block.content_hash,
        })
        .collect();
    index.apply_stored(a, &blocks, None, &mut ma).expect("a");
    index.apply_stored(b, &impostor, None, &mut mb).expect("b");
    assert_eq!(index.stats().landing_mismatches, 1);
    assert_eq!(scores(&index, &chain), vec![(a, 20), (b, 10)]);
    assert_eq!(scores(&index, &other), vec![(a, 10), (b, 20)]);
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(a, &blocks, None).expect("ref a");
    reference.apply_stored(b, &impostor, None).expect("ref b");
    assert_eq!(index.debug_blocks(), reference.blocks());
    let hashes: Vec<SequenceHash> = impostor.iter().map(|block| block.seq_hash).collect();
    index.apply_removed(b, &hashes, &mut mb);
    assert_eq!(scores(&index, &other), vec![(a, 10)]);
    assert!(mb.is_empty());
}

/// The race the release harness caught: a split of the parent between a store walk's plan
/// and its lock-free claim must make the insert give up, not link the child after the new
/// end. Replayed deterministically: plan (take the version), split, then try the insert.
#[test]
fn a_child_insert_planned_before_a_split_gives_up() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut mw = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..10).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    index.apply_stored(w, &blocks, None, &mut mw).expect("w");
    let parent = mw.get(blocks[9].seq_hash).expect("mapped").run;
    let (_, planned) = index.slab.run(parent).snapshot();
    // A divergence inside the run leaves its end where it is (a fork hangs off its offset), and
    // neither does a tail eviction (the cutoff drops, the array stays for regrowth); a hole
    // does: w drops blocks 3 and 4, and the run is cut at 5.
    let hole: Vec<SequenceHash> = blocks[3..5].iter().map(|block| block.seq_hash).collect();
    index.apply_removed(w, &hole, &mut mw);
    assert_eq!(index.slab.run(parent).len(), 5);
    // The child prepared for blocks after the old end must not be linked after the new one.
    let contents = [content(3, 0).0, content(3, 1).0];
    let block = index.arena.alloc_array(&contents, capacity_for(2));
    let engine = index
        .arena
        .alloc_array(&[30, 31], index.arena.array_capacity(block));
    let child = index.slab.alloc(
        0,
        parent,
        Window {
            block,
            base: 0,
            engine,
            len: 2,
            children: NONE,
            partials: NONE,
            forwards: NONE,
        },
    );
    set(index.slab.coverage(child), w);
    assert!(matches!(
        index.insert_child(parent, child, 10, contents[0], planned),
        Claim::Changed
    ));
    let mut freed = Vec::new();
    index.discard_run(child, w, &mut freed);
    index.recycle(&mut freed);
    // The real path restarts from the parent block and lands the blocks after block 9, in
    // the suffix the hole split off; the lookup still stops at the hole.
    let mut longer = held.clone();
    longer.extend([content(3, 0), content(3, 1)]);
    let longer_blocks = blocks_of(&longer);
    index
        .apply_stored(w, &longer_blocks[10..], Some(blocks[9].seq_hash), &mut mw)
        .expect("extend");
    let mut reference = ReferenceIndexer::new();
    reference.apply_stored(w, &blocks, None).expect("ref w");
    reference.apply_removed(w, &hole);
    reference
        .apply_stored(w, &longer_blocks[10..], Some(blocks[9].seq_hash))
        .expect("ref extend");
    assert_eq!(index.debug_blocks(), reference.blocks());
    assert_eq!(scores(&index, &longer), vec![(w, 3)]);
}

/// Interning and releasing worker slots from many threads at once: an intern holds a name-map
/// shard and then the registry, a release must not hold the registry while it takes a shard,
/// or the two deadlock (this test hung within seconds before the order was fixed).
#[test]
fn worker_slots_churn_from_many_threads_without_deadlock() {
    let index = ChainIndex::with_max_workers(64);
    std::thread::scope(|scope| {
        for thread in 0..8u32 {
            let index = &index;
            scope.spawn(move || {
                for round in 0..2_000u32 {
                    let name = format!("t{thread}-r{}", round % 5);
                    let id = index.intern_worker(&name).expect("slot");
                    let mut map = ChainBlockMap::default();
                    let held: Vec<ContentHash> =
                        (0..3).map(|p| content(u64::from(id) + 1, p)).collect();
                    index
                        .apply_stored(id, &blocks_of(&held), None, &mut map)
                        .expect("store");
                    index.remove_worker(id, map);
                }
            });
        }
    });
    assert_eq!(index.current_size(), 0);
    assert_eq!(
        index.intern_worker("after"),
        Ok(index.intern_worker("after").expect("slot"))
    );
}

#[test]
fn parent_errors_match_the_positional_indexer() {
    let index = ChainIndex::with_max_workers(8);
    let w = index.intern_worker("w").expect("id");
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..3).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    assert!(matches!(
        index.apply_stored(w, &blocks[1..], Some(blocks[0].seq_hash), &mut map),
        Err(ApplyError::WorkerNotTracked)
    ));
    index
        .apply_stored(w, &blocks[..1], None, &mut map)
        .expect("store");
    assert!(matches!(
        index.apply_stored(w, &blocks[2..], Some(blocks[1].seq_hash), &mut map),
        Err(ApplyError::ParentBlockNotFound)
    ));
}

#[test]
fn worker_slots_are_bounded_and_reused_after_removal() {
    let index = ChainIndex::with_max_workers(2);
    assert_eq!(index.intern_worker("a"), Ok(0));
    assert_eq!(index.intern_worker("b"), Ok(1));
    assert_eq!(index.intern_worker("a"), Ok(0));
    assert_eq!(index.intern_worker("c"), Err(WorkerIdExhausted));
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..4).map(|p| content(1, p)).collect();
    index
        .apply_stored(0, &blocks_of(&held), None, &mut map)
        .expect("store");
    index.remove_worker(0, map);
    assert_eq!(index.worker_id("a"), None);
    assert_eq!(
        index.intern_worker("c"),
        Ok(0),
        "the freed slot is handed out again"
    );
    assert_eq!(
        scores(&index, &held),
        vec![],
        "nothing of the old holder survives"
    );
    assert_eq!(index.intern_worker("a"), Err(WorkerIdExhausted));
}

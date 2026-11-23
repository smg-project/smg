use kv_index::{compute_request_content_hashes, ContentHash, GroupCap, WorkerGroups};
use smg_policy_controller::engine::CacheIndex;
use smg_worker::{
    common_proto::{kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent},
    kv_events::apply::{apply_cleared, apply_event, apply_events, WorkerIndexState},
};

use super::KvIndex;

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

/// The index's overlap for `hashes` on `worker`, after the cache groups'
/// rule, with the holders the rule shrank.
fn capped(
    indexer: &KvIndex,
    worker: u32,
    hashes: &[ContentHash],
    block_size: usize,
) -> (Option<u32>, Vec<GroupCap>) {
    let mut overlap = indexer.find_matches(hashes, false);
    let caps = indexer.cap_by_groups(hashes, block_size, &mut overlap);
    (overlap.scores.get(&worker).copied(), caps)
}

/// A hybrid worker: the window group's store precedes the main store of
/// the same blocks in a batch and is placed through it; a kind-less event
/// on the learned group follows the group; the main group's chain is the
/// index's, the window group's blocks the view's.
#[test]
fn non_main_attention_groups_are_kept_in_the_worker_view() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut full = stored_event(1, &TOKENS);
    full.group_idx = Some(0);
    full.kv_cache_spec_kind = Some("full_attention".to_string());
    let mut second = stored_event(2, &[50, 60, 70, 80]);
    second.parent_block_hash = Some(1);
    let batch = [
        group_store(1, Some("sliding_window"), Some(5), &[1, 2]),
        store_event(full),
        store_event(second),
    ];
    apply_events(&batch, w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2, "the main chain alone is indexed");
    assert_eq!(indexer.groups().len(), 1);
    let view = indexer.groups().get(w1).expect("the view");
    assert_eq!(view.retained_blocks(), 2, "both window blocks placed");
    assert_eq!(
        (
            wb.counters.group_stores,
            wb.counters.group_blocks,
            wb.counters.group_unplaced
        ),
        (1, 2, 0)
    );

    let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));

    // A kind-less removal on the learned group takes the block out of the
    // view, not the index: a window of 5 tokens over 4-token blocks needs
    // one block ending at the hit; block 2 gone, the hit ends at block 1.
    let mut gone = removed_event(2);
    gone.group_idx = Some(1);
    apply_event(&remove_event(gone), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 2);
    assert_eq!(view.retained_blocks(), 1);
    assert_eq!(wb.counters.group_removals, 1);
    let (depth, caps) = capped(&indexer, w1, &hashes, 4);
    assert_eq!(depth, Some(1));
    assert_eq!(
        caps,
        vec![GroupCap {
            worker: w1,
            uncapped: 2,
            capped: 1,
            group: 1
        }]
    );
    // The policy engine's lookup lowers the holder by the same rule.
    let request: Vec<u64> = hashes.iter().map(|hash| hash.0).collect();
    let mut kept = Vec::new();
    indexer.deepest_into(&request, 4, &[1 << w1], 16, &mut |_| 0, &mut kept);
    assert_eq!(kept, [(w1, 1)]);

    // A kind-less store on the learned group follows the group.
    let mut later = stored_event(2, &[]);
    later.group_idx = Some(1);
    later.blocks[0].token_ids.clear();
    apply_event(&store_event(later), w1, &indexer, &mut wb);
    assert_eq!(view.retained_blocks(), 2);
    assert_eq!(capped(&indexer, w1, &hashes, 4).0, Some(2));

    // A group store naming blocks the index does not hold is counted, not placed.
    apply_event(&group_store(1, None, None, &[77]), w1, &indexer, &mut wb);
    assert_eq!(
        (wb.counters.group_blocks, wb.counters.group_unplaced),
        (3, 1)
    );

    // The main group's removal keeps the window block's name while the
    // window retains it; a clear empties the view with the index.
    let mut main_gone = removed_event(2);
    main_gone.group_idx = Some(0);
    apply_event(&remove_event(main_gone), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 1);
    assert_eq!(view.named_blocks(), 2);
    apply_cleared(w1, &indexer, &mut wb);
    assert_eq!((view.retained_blocks(), view.named_blocks()), (0, 0));
    assert!(view.knows(1));

    // The worker's removal drops its view.
    indexer.remove_worker(w1, wb.blocks);
    assert!(indexer.groups().is_empty());
}

/// The main store first in the batch (an engine that publishes the main
/// group first), or in a batch of its own after the window group's store
/// (a relay snapshot replays the group lanes first): the window's blocks
/// are placed either way, in the batch that names them.
#[test]
fn a_group_store_is_placed_whichever_side_of_the_main_store_it_comes() {
    let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
    let mut second = stored_event(2, &[50, 60, 70, 80]);
    second.parent_block_hash = Some(1);

    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[
            store_event(stored_event(1, &TOKENS)),
            store_event(second.clone()),
            group_store(1, Some("sliding_window"), Some(5), &[1, 2]),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    let view = indexer.groups().get(w1).expect("the view");
    assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
    assert_eq!(
        (wb.counters.group_blocks, wb.counters.group_unplaced),
        (2, 0)
    );
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));

    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[group_store(1, Some("sliding_window"), Some(5), &[1, 2])],
        w1,
        &indexer,
        &mut wb,
    );
    let view = indexer.groups().get(w1).expect("the view");
    assert_eq!((view.retained_blocks(), view.pending_blocks()), (0, 2));
    assert_eq!(
        (wb.counters.group_blocks, wb.counters.group_unplaced),
        (0, 2),
        "counted as they wait"
    );
    apply_events(
        &[store_event(stored_event(1, &TOKENS))],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        capped(&indexer, w1, &hashes, 4),
        (Some(1), Vec::new()),
        "one block named: the leading run"
    );
    apply_events(&[store_event(second)], w1, &indexer, &mut wb);
    assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
    assert_eq!(
        (wb.counters.group_blocks, wb.counters.group_unplaced),
        (2, 2),
        "placed as the main stores came"
    );
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));
}

/// A group a window store names before any main store of its blocks
/// places nothing when the main store never comes (another rank's cache,
/// a sub-block entry): the rule then leaves the worker's overlap standing
/// rather than take it away on a group it cannot read.
#[test]
fn a_group_that_never_placed_a_block_leaves_the_overlap_standing() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[
            group_store(1, Some("sliding_window"), Some(5), &[9]),
            store_event(stored_event(1, &TOKENS)),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        (wb.counters.group_blocks, wb.counters.group_unplaced),
        (0, 1)
    );
    let hashes = compute_request_content_hashes(&TOKENS, 4);
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));

    // An unknown kind places its blocks and is left out of the rule too.
    apply_event(
        &group_store(2, Some("chunked_local"), None, &[1]),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));
    // A state group that retains nothing on the path shrinks it to nothing: the holder leaves.
    apply_event(
        &group_store(3, Some("mamba"), None, &[1]),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(capped(&indexer, w1, &hashes, 4).0, Some(1));
    let mut gone = removed_event(1);
    gone.group_idx = Some(3);
    apply_event(&remove_event(gone), w1, &indexer, &mut wb);
    let (depth, caps) = capped(&indexer, w1, &hashes, 4);
    assert_eq!(depth, None, "capped to nothing: no longer a holder");
    assert_eq!(caps[0].capped, 0);
    assert_eq!(
        indexer.current_size(),
        1,
        "the index still holds the main block"
    );
}

/// A worker whose first batch carried main stores alone gets its view
/// later, over blocks the view never named: the rule leaves its overlap
/// standing (never a wrong name, never a shrunk hit on a path it cannot
/// read) until a clear, after which the view names every block as it is
/// indexed and reads again.
#[test]
fn a_view_created_after_the_first_main_stores_caps_nothing_until_a_clear() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    apply_events(
        &[store_event(stored_event(1, &TOKENS))],
        w1,
        &indexer,
        &mut wb,
    );
    assert!(indexer.groups().is_empty());
    let mut second = stored_event(2, &[50, 60, 70, 80]);
    second.parent_block_hash = Some(1);
    apply_events(
        &[
            store_event(second),
            group_store(1, Some("sliding_window"), Some(5), &[2]),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    let view = indexer.groups().get(w1).expect("the view");
    assert!(!view.is_complete());
    assert_eq!(
        view.named_blocks(),
        0,
        "block 2 hangs on a block the view never named"
    );
    assert_eq!(
        (wb.counters.group_unnamed, wb.counters.group_view_late),
        (1, 1)
    );
    let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
    assert_eq!(
        capped(&indexer, w1, &hashes, 4),
        (Some(2), Vec::new()),
        "the index's overlap stands"
    );

    apply_cleared(w1, &indexer, &mut wb);
    assert!(view.is_complete());
    let mut second = stored_event(2, &[50, 60, 70, 80]);
    second.parent_block_hash = Some(1);
    apply_events(
        &[
            store_event(stored_event(1, &TOKENS)),
            store_event(second),
            group_store(1, Some("sliding_window"), Some(5), &[1]),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(view.named_blocks(), 2);
    let (depth, caps) = capped(&indexer, w1, &hashes, 4);
    assert_eq!(
        depth,
        Some(1),
        "the window's one block ends the hit at block 1"
    );
    assert_eq!(caps.len(), 1);
}

/// A store of a kind other than main attention without a group index
/// names nothing the view can keep; a plain worker never gets a view.
#[test]
fn a_nameless_group_store_is_counted_and_a_plain_worker_has_no_view() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let mut nameless = stored_event(1, &TOKENS);
    nameless.kv_cache_spec_kind = Some("sliding_window".to_string());
    apply_event(&store_event(nameless), w1, &indexer, &mut wb);
    assert_eq!(indexer.current_size(), 0);
    assert_eq!(wb.counters.group_nameless, 1);
    apply_event(
        &store_event(stored_event(1, &TOKENS)),
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(indexer.current_size(), 1);
    assert!(indexer.groups().is_empty(), "no view without a group");
    let hashes = compute_request_content_hashes(&TOKENS, 4);
    assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));
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

const SECOND: [u32; 4] = [50, 60, 70, 80];
const THIRD: [u32; 4] = [90, 91, 92, 93];

/// A hashes-only store of the window group before any main group gives
/// the worker a view late, over the signal's blocks, incomplete; the main
/// group's store restarts the worker: the view names its blocks from an
/// empty index, complete, and the window group's memberships stay for
/// the names to come, the retained ones and the signal's own, so a path
/// the engine serves keeps its credit.
#[test]
fn a_view_created_late_is_complete_after_the_restart() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let one = compute_request_content_hashes(&TOKENS, 4);
    let two = compute_request_content_hashes(&SECOND, 4);
    let three = compute_request_content_hashes(&THIRD, 4);
    apply_events(
        &[window_only_store(0, 1, None, &TOKENS)],
        w1,
        &indexer,
        &mut wb,
    );
    apply_events(
        &[group_store(0, Some("sliding_window"), Some(64), &[1])],
        w1,
        &indexer,
        &mut wb,
    );
    let view = wb
        .groups
        .clone()
        .expect("the hashes-only store gives a view");
    assert!(!view.is_complete(), "created over the signal's block");
    assert_eq!(wb.counters.group_view_late, 1);

    // A second signal block, named by the view and retained by the
    // window group through it; a third with no hashes-only store of its
    // own, named only.
    apply_events(
        &[window_only_store(0, 2, None, &SECOND)],
        w1,
        &indexer,
        &mut wb,
    );
    apply_events(
        &[group_store(0, Some("sliding_window"), Some(64), &[2])],
        w1,
        &indexer,
        &mut wb,
    );
    apply_events(
        &[window_only_store(0, 3, None, &THIRD)],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!((view.named_blocks(), view.retained_blocks()), (2, 1));

    let mut main = stored_event(2, &SECOND);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    assert!(view.is_complete(), "the restart cleared the names");
    assert_eq!(wb.counters.main_signal_dropped, 3);
    assert!(!routable(&indexer, w1, &one), "the signal's block went");
    assert!(routable(&indexer, w1, &two));
    assert_eq!(indexer.current_size(), 1);
    assert_eq!(view.named_blocks(), 1, "the main group's block is named");
    assert_eq!(
        view.retained_blocks(),
        1,
        "the window group's membership came back with the name"
    );
    assert_eq!(
        capped(&indexer, w1, &two, 4).0,
        Some(1),
        "the path the engine serves keeps its credit"
    );

    // The third block, the signal's own: the main group's store names it
    // and the window group's membership is there.
    let mut third = stored_event(3, &THIRD);
    third.group_idx = Some(1);
    third.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(third)], w1, &indexer, &mut wb);
    assert_eq!((view.named_blocks(), view.retained_blocks()), (2, 2));
    assert_eq!(capped(&indexer, w1, &three, 4).0, Some(1));
}

/// The signal's own blocks are the window group's: a worker whose window
/// stores were its main signal shows a main group, and after the restart
/// the view holds the window group's membership for every block the
/// index let go, so a path the engine serves keeps its credit once the
/// group can be read.
#[test]
fn the_signals_blocks_keep_the_window_groups_membership_across_the_restart() {
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
    assert!(wb.groups.is_none());

    let mut main = stored_event(1, &TOKENS);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    let view = wb.groups.clone().expect("the restart makes the view");
    assert!(view.is_complete());
    assert_eq!(
        view.retained_blocks(),
        1,
        "the signal's block waited and was placed as the window group's"
    );

    // The window group can be read once a store of its own names its
    // kind.
    let mut second = stored_event(2, &SECOND);
    second.group_idx = Some(1);
    second.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(
        &[
            group_store(0, Some("sliding_window"), Some(64), &[2]),
            store_event(second),
        ],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(view.retained_blocks(), 2);
    assert_eq!(
        capped(&indexer, w1, &one, 4).0,
        Some(1),
        "the path the engine serves keeps its credit"
    );
}

/// The restart keeps every membership the view holds, retained or waiting,
/// not only the signal group's. A state group's hashes-only store of the
/// signal's first block (the view made late over it) waits for a name; its
/// store of a second signal block, one the view names, is retained. The
/// restart moves the retained membership back to waiting beside the
/// waiting one, and the main group's names place both, each beside the
/// signal group's seeded bit. A clear in place of the move drops both; the
/// move dropped alone loses the retained one.
#[test]
fn another_groups_memberships_survive_the_restart() {
    let indexer = KvIndex::chain();
    let w1 = indexer.intern_worker("http://w1:8000").unwrap();
    let mut wb = WorkerIndexState::default();
    let one = compute_request_content_hashes(&TOKENS, 4);
    let two = compute_request_content_hashes(&SECOND, 4);
    let placed_by = |view: &WorkerGroups, index: u32| {
        view.groups()
            .iter()
            .find(|group| group.index == index)
            .map_or(0, |group| group.placed)
    };
    apply_events(
        &[window_only_store(0, 1, None, &TOKENS)],
        w1,
        &indexer,
        &mut wb,
    );
    // The state group's store of the signal's first block: the view is
    // made late over the block, which it cannot name, so the membership
    // waits under the state group's bit.
    apply_events(
        &[group_store(2, Some("mamba"), None, &[1])],
        w1,
        &indexer,
        &mut wb,
    );
    let view = wb
        .groups
        .clone()
        .expect("the hashes-only store gives a view");
    assert!(!view.is_complete(), "created over the signal's block");
    // A second signal block, named by the view, and the state group's
    // store of it: retained.
    apply_events(
        &[window_only_store(0, 2, None, &SECOND)],
        w1,
        &indexer,
        &mut wb,
    );
    apply_events(
        &[group_store(2, Some("mamba"), None, &[2])],
        w1,
        &indexer,
        &mut wb,
    );
    assert_eq!(
        (
            view.named_blocks(),
            view.retained_blocks(),
            view.pending_blocks()
        ),
        (1, 1, 1),
        "block 2 named and retained, block 1 waiting"
    );
    assert_eq!(placed_by(&view, 2), 1);

    // The restart: the main group names block 2 anew. The retained
    // membership moved back to waiting and is placed with the name, the
    // waiting one keeps waiting; the signal group's bit is seeded beside.
    let mut main = stored_event(2, &SECOND);
    main.group_idx = Some(1);
    main.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(main)], w1, &indexer, &mut wb);
    assert!(view.is_complete(), "the restart cleared the names");
    assert_eq!(wb.counters.main_signal_dropped, 2);
    assert_eq!(
        (
            view.named_blocks(),
            view.retained_blocks(),
            view.pending_blocks()
        ),
        (1, 1, 1),
        "block 2 placed again, block 1 still waiting"
    );
    assert_eq!(
        placed_by(&view, 2),
        2,
        "the retained membership came back with the name (placed twice: before and after)"
    );
    assert_eq!(
        placed_by(&view, 0),
        1,
        "the signal group's seeded bit placed"
    );
    assert_eq!(
        capped(&indexer, w1, &two, 4).0,
        Some(1),
        "the state group reads block 2: the path keeps its credit"
    );

    // The main group names block 1: the waiting membership is placed too.
    let mut first = stored_event(1, &TOKENS);
    first.group_idx = Some(1);
    first.kv_cache_spec_kind = Some("full_attention".to_string());
    apply_events(&[store_event(first)], w1, &indexer, &mut wb);
    assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
    assert_eq!((placed_by(&view, 2), placed_by(&view, 0)), (3, 2));
    assert_eq!(capped(&indexer, w1, &one, 4).0, Some(1));
}

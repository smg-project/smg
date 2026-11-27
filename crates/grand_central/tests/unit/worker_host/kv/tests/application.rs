//! A batch's application: what its stores, removals and clears write, what
//! the row's stream showed of its groups, and a batch the cursor skips. The
//! port's other calls: a resync's reason, a cursor's reading, a
//! reconnection and a removal, as each reaches the tables.

use super::*;

fn assert_store_is_skipped(event: KvBlocksStored) {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 0);
}

/// A main store that names a namespace, an adapter and a tenant's salt,
/// is indexed with its chain as any other, over two batches. The hashes
/// under the namespace's seed are the apply rules' own.
#[test]
fn a_main_store_of_a_namespace_is_indexed_with_its_chain() {
    let f = Fixture::new();
    let parent = block(-1, &[1, 2]);
    let parent_hash = parent.block_hash;
    let mut event = KvBlocksStored {
        blocks: vec![parent],
        group_idx: Some(0),
        kv_cache_spec_kind: Some("full_attention".into()),
        lora_name: Some("adapter".into()),
        cache_salt: Some("tenant".into()),
        ..KvBlocksStored::default()
    };
    f.applied(1, &batch([Data::Stored(event.clone())]));
    event.parent_block_hash = Some(parent_hash);
    event.blocks = vec![block(2, &[3, 4]), block(3, &[5, 6])];
    f.applied(2, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 3);
}

#[test]
fn a_foreign_owners_store_is_passed_over() {
    assert_store_is_skipped(KvBlocksStored {
        ownership: Some("KvCr".into()),
        ..stored(1)
    });
}

#[test]
fn a_remote_store_is_passed_over() {
    assert_store_is_skipped(KvBlocksStored {
        locality: Some(KvCacheLocality::Remote as i32),
        ..stored(1)
    });
}

#[test]
fn a_store_on_an_unindexed_tier_is_passed_over() {
    for (tier, level) in [
        (Some(KvCacheTier::Disk as i32), Some(0)),
        (Some(KvCacheTier::External as i32), None),
        (None, Some(2)),
        (Some(KvCacheTier::Unspecified as i32), Some(3)),
        (Some(99), Some(2)),
    ] {
        let mut event = stored(1);
        event.tier = tier;
        event.blocks[0].cache_level = level;
        assert_store_is_skipped(event);
    }
}

#[test]
fn a_store_uses_the_named_tier_then_the_first_blocks_cache_level() {
    for (tier, level) in [
        (Some(KvCacheTier::Device as i32), Some(2)),
        (Some(KvCacheTier::Host as i32), Some(2)),
        (None, Some(1)),
        (Some(KvCacheTier::Unspecified as i32), Some(1)),
        (Some(99), Some(1)),
    ] {
        let f = Fixture::new();
        let mut event = stored(1);
        event.tier = tier;
        event.blocks[0].cache_level = level;
        event.blocks.push(KvBlock {
            cache_level: Some(2),
            ..block(2, &[3, 4])
        });
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 2, "tier {tier:?}, cache level {level:?}");
    }
}

#[test]
fn each_main_attention_kind_is_indexed_with_a_group() {
    for kind in ["full_attention", "mla_attention", "sink_full_attention"] {
        let f = Fixture::new();
        let event = KvBlocksStored {
            group_idx: Some(1),
            kv_cache_spec_kind: Some(kind.into()),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1, "{kind}");
    }
}

#[test]
fn a_kindless_store_of_a_group_not_shown_as_another_kind_is_the_main_groups() {
    let f = Fixture::new();
    let event = KvBlocksStored {
        group_idx: Some(0),
        ..stored(1)
    };
    f.applied(1, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 1);
}

#[test]
fn a_kindless_store_of_a_group_shown_as_another_kind_is_passed_over() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 1, false)),
        ]),
    );
    let event = KvBlocksStored {
        group_idx: Some(1),
        ..stored(2)
    };
    f.applied(2, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 1);
}

#[test]
fn a_window_group_with_tokens_is_the_main_signal_until_a_main_group_shows() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(window_store(&[1, 2], 1, true))]));
    assert_eq!(f.blocks(), 2);
    // Its kind-less removals follow it to the index.
    f.applied(2, &batch([Data::Removed(group_removal(&[2], 1))]));
    assert_eq!(f.blocks(), 1);
    // A main group shows: the signal's blocks go, the main group's stand.
    f.applied(3, &batch([Data::Stored(main_store(7, 0))]));
    assert_eq!(f.blocks(), 1);
    // The window group is a group of another kind from here.
    f.applied(
        4,
        &batch([
            Data::Stored(window_store(&[8], 1, true)),
            Data::Removed(group_removal(&[7], 1)),
        ]),
    );
    assert_eq!(f.blocks(), 1);
}

/// A removal that names a group without its kind is the group's where
/// the group was shown, even ahead of the group's first store in the
/// batch, and even for a group no view could place: the main block of
/// the same hash stays indexed. A main store without tokens indexes
/// nothing.
#[test]
fn a_kind_less_group_event_never_reaches_the_main_index() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([
            Data::Stored(stored(1)),
            Data::Removed(group_removal(&[1], 1)),
            Data::Stored(window_store(&[1], 1, false)),
        ]),
    );
    assert_eq!(f.blocks(), 1, "the main block stays");

    let mut hidden = window_store(&[2], 17, false);
    hidden.kv_cache_spec_kind = Some("mamba".into());
    let second = KvBlocksStored {
        parent_block_hash: Some(1),
        ..stored(2)
    };
    f.applied(2, &batch([Data::Stored(hidden), Data::Stored(second)]));
    assert_eq!(f.blocks(), 2);
    f.applied(3, &batch([Data::Removed(group_removal(&[2], 17))]));
    assert_eq!(f.blocks(), 2, "the main block stays");

    let bare = KvBlocksStored {
        blocks: vec![block(3, &[])],
        ..KvBlocksStored::default()
    };
    f.applied(4, &batch([Data::Stored(bare)]));
    assert_eq!(f.blocks(), 2);
}

/// A worker whose attention is all sliding window has no main group:
/// its window stores carry tokens and are the prefix index's, as a plain
/// worker's, and a kind-less removal of the group leaves the index.
#[test]
fn a_window_only_workers_stores_and_removals_are_the_prefix_indexs() {
    let f = Fixture::new();
    let second = KvBlocksStored {
        parent_block_hash: Some(1),
        ..window_store(&[2], 0, true)
    };
    f.applied(
        1,
        &batch([
            Data::Stored(window_store(&[1], 0, true)),
            Data::Stored(second),
        ]),
    );
    assert_eq!(f.blocks(), 2);
    f.applied(2, &batch([Data::Removed(group_removal(&[2], 0))]));
    assert_eq!(f.blocks(), 1);
}

/// A store the index skips (a residency agent's, a remote tier's) shows
/// no group in a batch: a foreign main store ahead of a window-only
/// worker's store leaves that store in the prefix index, and a foreign
/// window store shows no group.
#[test]
fn a_foreign_store_in_a_batch_shows_no_group() {
    let f = Fixture::new();
    let agent_main = KvBlocksStored {
        ownership: Some("kvcr".into()),
        ..main_store(7, 1)
    };
    let remote_window = KvBlocksStored {
        locality: Some(KvCacheLocality::Remote as i32),
        ..window_store(&[8], 2, true)
    };
    f.applied(
        1,
        &batch([
            Data::Stored(agent_main),
            Data::Stored(window_store(&[1], 0, true)),
            Data::Stored(remote_window),
        ]),
    );
    assert_eq!(f.blocks(), 1, "the window store is the prefix index's");

    // The group pass's own case: a worker whose main group has shown
    // gets a foreign window store, which would show that group. Skipped,
    // the group stays unshown, and its kind-less removal is the main
    // group's.
    let f = Fixture::new();
    let remote_window = KvBlocksStored {
        locality: Some(KvCacheLocality::Remote as i32),
        ..window_store(&[9], 2, true)
    };
    f.applied(
        1,
        &batch([Data::Stored(stored(1)), Data::Stored(remote_window)]),
    );
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Removed(group_removal(&[1], 2))]));
    assert_eq!(f.blocks(), 0);
}

/// A relay snapshot replays the stores after a clear in one batch: the
/// batch reads as the stream did, and a removal after it still leaves
/// the index.
#[test]
fn a_window_only_workers_snapshot_replay_reads_as_its_stream() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
    let second = KvBlocksStored {
        parent_block_hash: Some(1),
        ..window_store(&[2], 0, true)
    };
    f.applied(
        2,
        &batch([
            Data::Cleared(KvCacheCleared::default()),
            Data::Stored(window_store(&[1], 0, true)),
            Data::Stored(second),
        ]),
    );
    assert_eq!(f.blocks(), 2);
    f.applied(
        3,
        &batch([
            Data::Removed(group_removal(&[1], 0)),
            Data::Removed(group_removal(&[2], 0)),
        ]),
    );
    assert_eq!(f.blocks(), 0);
}

/// A worker whose first batches carried only a window group's stores
/// (its main signal) and then shows a main group: the index lets the
/// signal's blocks go and the main group's stores rebuild it, and the
/// window group's removals are never the main group's blocks' from then
/// on.
#[test]
fn a_main_group_shown_later_moves_the_window_group_to_the_view() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
    assert_eq!(f.blocks(), 1);

    // The main lane's batch of the same block: the signal's copy goes,
    // the main group's comes.
    f.applied(2, &batch([Data::Stored(main_store(1, 1))]));
    assert_eq!(f.blocks(), 1);

    // The window group's later store and its removals leave the index
    // alone: the main group's blocks stand until the main group lets
    // them go.
    f.applied(3, &batch([Data::Stored(window_store(&[1], 0, false))]));
    f.applied(4, &batch([Data::Stored(main_store(4, 1))]));
    f.applied(
        5,
        &batch([
            Data::Removed(group_removal(&[1], 0)),
            Data::Removed(group_removal(&[4], 0)),
        ]),
    );
    assert_eq!(
        f.blocks(),
        2,
        "the window group's removals leave the index alone"
    );
    f.applied(6, &batch([Data::Removed(group_removal(&[1], 1))]));
    assert_eq!(f.blocks(), 1, "the main group let it go");
}

/// A hybrid worker whose window stores carry tokens: the main group's
/// store in the same batch keeps them out of the prefix index.
#[test]
fn a_hybrid_workers_tokenful_window_store_stays_in_the_view() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([
            Data::Stored(window_store(&[1], 1, true)),
            Data::Stored(main_store(1, 0)),
        ]),
    );
    assert_eq!(f.blocks(), 1);
    // The window group's kind-less removal is the group's.
    f.applied(2, &batch([Data::Removed(group_removal(&[1], 1))]));
    assert_eq!(f.blocks(), 1);
}

/// A store that names a kind other than main attention and no group is
/// not applied, and a plain store after it is indexed.
#[test]
fn a_nameless_group_store_is_counted_and_a_plain_worker_has_no_view() {
    let f = Fixture::new();
    let nameless = KvBlocksStored {
        kv_cache_spec_kind: Some("sliding_window".into()),
        ..stored(1)
    };
    f.applied(1, &batch([Data::Stored(nameless)]));
    assert_eq!(f.blocks(), 0);
    f.applied(2, &batch([Data::Stored(stored(1))]));
    assert_eq!(f.blocks(), 1);
}

#[test]
fn a_batchs_stores_of_other_groups_are_decoded_last() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 1, false)),
        ]),
    );
    // Group 1 is a group of another kind, so its kind-less store waits
    // for the batch's other events. The main store behind it takes the
    // group back, and the store is the main group's when it is decoded.
    let kindless = KvBlocksStored {
        group_idx: Some(1),
        ..stored(2)
    };
    f.applied(
        2,
        &batch([Data::Stored(kindless), Data::Stored(main_store(3, 1))]),
    );
    assert_eq!(f.blocks(), 3);
}

#[test]
fn a_late_batch_that_shows_a_main_group_is_no_restarted_publisher() {
    let f = Fixture::new();
    f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
    f.applied(5, &batch([Data::Stored(window_store(&[2], 0, true))]));
    // A main store at a sequence the row holds would restart the worker.
    // Its clear is not the engine's own: the cursor skips the batch.
    let late = batch([Data::Stored(main_store(7, 1))]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts(4), &late),
        BatchOutcome::Skipped
    );
    assert_eq!(f.blocks(), 2);
    // The window group is the worker's main signal still.
    f.applied(6, &batch([Data::Stored(window_store(&[3], 0, true))]));
    assert_eq!(f.blocks(), 3);
}

/// A clear owned by a name the header does not count, at a sequence the
/// row holds: the batch is late, and the cursor skips it as any other.
#[test]
fn a_late_batch_with_an_owned_clear_is_skipped_as_any_late_batch() {
    let f = Fixture::new();
    f.applied(9, &batch([Data::Stored(stored(1))]));
    let second = KvBlocksStored {
        parent_block_hash: Some(1),
        ..stored(2)
    };
    f.applied(10, &batch([Data::Stored(second)]));
    let late = batch([
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(3, 0)),
    ]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts(9), &late),
        BatchOutcome::Skipped
    );
    assert_eq!(f.blocks(), 2);
}

/// The first batch of a new connection under the cursor is a restarted
/// publisher by the cursor's own rule. Its owned clear still wipes what
/// stood ahead of it in the batch.
#[test]
fn an_owned_clear_in_a_late_batch_the_store_takes_wipes_what_stood_ahead_of_it() {
    let f = Fixture::new();
    f.applied(9, &batch([Data::Stored(stored(1))]));
    f.sink.reconnected(f.row);
    let restarted = batch([
        Data::Stored(main_store(5, 0)),
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(6, 0)),
    ]);
    f.applied(3, &restarted);
    assert_eq!(f.blocks(), 1);
    // The block that stands is the one behind the clear.
    f.applied(4, &batch([Data::Removed(group_removal(&[6], 0))]));
    assert_eq!(f.blocks(), 0);
}

/// Engines number batches from 0, and the row's cursor reads 0 behind
/// batch 0. A batch 0 that arrives again is late there as at any other
/// sequence: its owned clear empties nothing.
#[test]
fn a_repeated_first_batch_with_an_owned_clear_is_skipped() {
    let f = Fixture::new();
    f.applied(0, &batch([Data::Stored(stored(1))]));
    let late = batch([
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(3, 0)),
    ]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
        BatchOutcome::Skipped
    );
    // The first batch's block stands: its removal empties the rank.
    assert_eq!(f.blocks(), 1);
    f.applied(1, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0);
}

/// A main store in a batch 0 that arrives again would restart the
/// worker by the adapter's own clear. Behind batch 0 it is late as
/// well, and the cursor skips it.
#[test]
fn a_repeated_first_batch_that_shows_a_main_group_is_no_restarted_publisher() {
    let f = Fixture::new();
    f.applied(0, &batch([Data::Stored(window_store(&[1], 0, true))]));
    let late = batch([Data::Stored(main_store(7, 1))]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
        BatchOutcome::Skipped
    );
    assert_eq!(f.blocks(), 1);
    // The window group is the worker's main signal still.
    f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
    assert_eq!(f.blocks(), 2);
}

/// A cursor reads 0 behind a resync too, and the resync emptied the
/// rank. A batch 0 there is the stream's first: the cursor takes it, and
/// what it holds behind its last clear stands.
#[test]
fn a_first_batch_behind_a_resync_stands_from_its_last_clear() {
    let f = Fixture::new();
    f.applied(4, &batch([Data::Stored(stored(1))]));
    f.sink.resync(f.row, None, ResyncReason::DataLoss);
    assert_eq!(f.blocks(), 0);
    let first = batch([
        Data::Stored(main_store(5, 0)),
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(6, 0)),
    ]);
    f.applied(0, &first);
    assert_eq!(f.blocks(), 1);
    f.applied(1, &batch([Data::Removed(group_removal(&[6], 0))]));
    assert_eq!(f.blocks(), 0);
}

/// A main group shown in a batch 0 behind a resync: the adapter's own
/// clear is read as late there, and the resync emptied the rank already.
#[test]
fn a_first_batch_behind_a_resync_that_shows_a_main_group_stands() {
    let f = Fixture::new();
    f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
    f.sink.resync(f.row, None, ResyncReason::DataLoss);
    f.applied(0, &batch([Data::Stored(main_store(7, 1))]));
    assert_eq!(f.blocks(), 1);
    // The window group is a group of another kind now: its store is not
    // the index's.
    f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
    assert_eq!(f.blocks(), 1);
    // The block that stands is the main group's.
    f.applied(2, &batch([Data::Removed(group_removal(&[7], 1))]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_batch_the_store_does_not_take_shows_nothing() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
    // A batch at a sequence the cursor holds: its store shows no group.
    let late = batch([Data::Stored(window_store(&[1], 0, false))]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts(1), &late),
        BatchOutcome::Skipped
    );
    f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_worker_that_left_takes_what_its_stream_showed_with_it() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([
            Data::Stored(main_store(1, 0)),
            Data::Stored(window_store(&[1], 1, false)),
        ]),
    );
    f.sink.removed(f.row);
    // The worker's next feed opens on the same row. Nothing is shown of
    // group 1 now: its kind-less store is the main group's.
    open_feed(&f.sink, f.row);
    let event = KvBlocksStored {
        group_idx: Some(1),
        ..stored(2)
    };
    f.applied(1, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 1);
}

#[test]
fn a_store_with_a_block_without_tokens_is_passed_over() {
    let mut event = stored(1);
    event.blocks.push(block(2, &[]));
    assert_store_is_skipped(event);
}

#[test]
fn a_store_without_blocks_is_passed_over() {
    let f = Fixture::new();
    let empty = batch([Data::Stored(KvBlocksStored::default())]);
    f.applied(1, &empty);
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_removal_takes_the_blocks_out() {
    let f = Fixture::new();
    let mut event = stored(-1);
    event.blocks.push(block(2, &[3, 4]));
    f.applied(1, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 2);
    // The rules count a block's copies a tier. A removal on the tier
    // that holds no copy of the blocks takes none out.
    let from_host = KvBlocksRemoved {
        tier: Some(KvCacheTier::Host as i32),
        cache_level: Some(2),
        ..removed(&[-1, 2])
    };
    f.applied(2, &batch([Data::Removed(from_host)]));
    assert_eq!(f.blocks(), 2);
    f.applied(3, &batch([Data::Removed(removed(&[-1, 2]))]));
    assert_eq!(f.blocks(), 0);
}

/// A block stored on the device and on the host is one block of the
/// index with two copies. It leaves with its last copy.
#[test]
fn a_block_leaves_the_index_with_its_last_copy() {
    let f = Fixture::new();
    let on_host = KvBlocksStored {
        tier: Some(KvCacheTier::Host as i32),
        ..stored(1)
    };
    f.applied(1, &batch([Data::Stored(stored(1)), Data::Stored(on_host)]));
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 1, "the host still holds a copy");
    let from_host = KvBlocksRemoved {
        tier: Some(KvCacheTier::Host as i32),
        ..removed(&[1])
    };
    f.applied(3, &batch([Data::Removed(from_host)]));
    assert_eq!(f.blocks(), 0);
}

/// Two copies on one tier need two removals.
#[test]
fn two_copies_on_a_tier_need_two_removals() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
    );
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 1, "the other copy is still cached");
    f.applied(3, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0);
}

/// The tables empty a rank in their own turn, here at a resync. The
/// row's next batch reads another value of its lane and forgets the
/// copies it counted: a block stored once behind the resync leaves at
/// its one removal.
#[test]
fn a_rank_the_store_emptied_forgets_the_copies_it_counted() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
    );
    f.tables.resync(f.row, Some(RANK), ResyncReason::DataLoss);
    assert_eq!(f.blocks(), 0);
    f.applied(5, &batch([Data::Stored(stored(1))]));
    assert_eq!(f.blocks(), 1);
    f.applied(6, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0, "one copy stood behind the resync");
}

/// A store narrower than the row's block of record is left out: an
/// engine's prefix entry at its hash granularity, which no request
/// chunked at the block matches.
#[test]
fn a_store_narrower_than_the_block_of_record_is_left_out() {
    let f = Fixture::new();
    f.applied(1, &wide(1, 32));
    assert_eq!(f.blocks(), 1);
    f.applied(2, &wide(77, 4));
    assert_eq!(f.blocks(), 1);
    // A wider store raises the record, and what was indexed stays.
    f.applied(3, &wide(2, 64));
    assert_eq!(f.blocks(), 2);
    f.applied(4, &wide(3, 32));
    assert_eq!(f.blocks(), 2);
}

/// The tables empty a rank in their own turn, and the row's next batch
/// forgets its block of record: an engine that comes back with a
/// narrower block has its stores indexed as they come.
#[test]
fn a_rank_the_store_emptied_forgets_its_block_of_record() {
    let f = Fixture::new();
    f.applied(1, &wide(1, 32));
    f.tables.resync(f.row, Some(RANK), ResyncReason::DataLoss);
    f.applied(5, &wide(2, 16));
    assert_eq!(f.blocks(), 1, "the narrower block is the row's block now");
}

/// The row's view in the tables' index, read by a lent batch with no
/// event at `sequence`.
fn view(f: &Fixture, sequence: u64) -> Arc<WorkerGroups> {
    let mut view = None;
    let outcome = f
        .tables
        .admit_through(f.row, RANK, facts(sequence), &mut |lane| {
            view = Some(lane.groups());
        });
    assert_eq!(outcome, BatchOutcome::Applied);
    view.expect("the body ran")
}

/// A main store and a window group's store of the same block, as a
/// hybrid engine writes them.
fn hybrid() -> KvEventBatch {
    batch([
        Data::Stored(main_store(1, 0)),
        Data::Stored(window_store(&[1], 1, false)),
    ])
}

/// A store of a cache group other than main attention feeds the row's
/// view in the tables' index, where a decision can read it.
#[test]
fn a_group_store_feeds_the_rows_view_in_the_stores_index() {
    let f = Fixture::new();
    f.applied(1, &hybrid());
    assert_eq!(f.blocks(), 1);
    let view = view(&f, 2);
    assert!(view.knows(1), "the window group is the view's");
    assert_eq!(view.named_blocks(), 1, "the main block is named in it");
}

/// A row's lane is made again under the row's id, and the index holds
/// another view for the row. The row's next batch reads another value
/// of its lane and takes that view: what its group stores name is in
/// the tables' index, not in the view of the lane before.
#[test]
fn a_lane_made_again_gives_the_rows_state_the_indexs_new_view() {
    let f = Fixture::new();
    f.applied(1, &hybrid());
    let old = view(&f, 2);
    // The row stops asking for its events, and asks again: each fact
    // is stated to the tables as a holder states it.
    let facts = f.store.get(f.row).expect("the row").facts;
    let quiet = WorkerFacts {
        signals: Signals::default(),
        ..facts.clone()
    };
    assert_eq!(f.store.upsert(quiet).id(), f.row);
    f.state(f.row);
    assert_eq!(f.store.upsert(facts).id(), f.row);
    f.state(f.row);
    f.applied(1, &hybrid());
    let new = view(&f, 2);
    assert!(!Arc::ptr_eq(&old, &new));
    assert!(new.knows(1), "the new view is the one the row writes");
    assert_eq!(new.named_blocks(), 1);
}

#[test]
fn a_removal_of_the_main_group_is_taken_under_its_group_number() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
    f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_foreign_remote_or_unindexed_removal_is_passed_over() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(stored(1))]));
    let options = [
        (Some("KvCr"), None, None, None),
        (None, Some(KvCacheLocality::Remote as i32), None, None),
        (None, None, Some(KvCacheTier::Disk as i32), Some(0)),
        (None, None, None, Some(2)),
    ];
    for (sequence, (owner, locality, tier, cache_level)) in (2..).zip(options) {
        let removal = KvBlocksRemoved {
            ownership: owner.map(str::to_owned),
            locality,
            tier,
            cache_level,
            ..removed(&[1])
        };
        f.applied(sequence, &batch([Data::Removed(removal)]));
        assert_eq!(f.blocks(), 1, "removal at sequence {sequence}");
    }
}

#[test]
fn a_clear_empties_the_rank() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(stored(1))]));
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Cleared(KvCacheCleared::default())]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_foreign_owners_clear_is_passed_over() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(stored(1))]));
    f.applied(
        2,
        &batch([Data::Cleared(KvCacheCleared {
            ownership: Some("KvCr".into()),
        })]),
    );
    assert_eq!(f.blocks(), 1);
}

#[test]
fn an_event_with_no_data_gives_none() {
    let f = Fixture::new();
    let missing = KvEventBatch {
        events: vec![KvCacheEvent::default()],
        ..KvEventBatch::default()
    };
    f.applied(1, &missing);
    assert_eq!(f.blocks(), 0);
}

#[test]
fn the_order_of_a_batch_of_three_event_kinds_is_kept() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(stored(1))]));
    let ordered = batch([
        Data::Removed(removed(&[1])),
        Data::Cleared(KvCacheCleared::default()),
        Data::Stored(stored(2)),
    ]);
    // The store behind the clear stands: the three were applied in the
    // batch's order.
    f.applied(2, &ordered);
    assert_eq!(f.blocks(), 1);
}

#[test]
fn the_facts_and_the_outcome_cross_as_they_are() {
    let f = Fixture::new();
    f.applied(21, &batch([Data::Stored(stored(1))]));
    let empty = batch([]);
    f.applied(22, &empty);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts(26), &empty),
        BatchOutcome::Gap {
            expected: 23,
            received: 26
        }
    );
    assert_eq!(
        f.sink.admit(f.row, RANK, facts(22), &empty),
        BatchOutcome::Skipped
    );

    let clear = BatchFacts {
        clears: true,
        ..facts(20)
    };
    assert_eq!(
        f.sink.admit(f.row, RANK, clear, &empty),
        BatchOutcome::Applied
    );
    assert_eq!(f.blocks(), 0);
    for (sequence, unknown_before) in [(70, true), (71, false)] {
        let chunk = BatchFacts {
            snapshot: Some(SnapshotChunk { unknown_before }),
            ..facts(sequence)
        };
        assert_eq!(
            f.sink.admit(f.row, RANK, chunk, &empty),
            BatchOutcome::Applied
        );
        assert_eq!(
            f.tables.cursor(f.row, RANK),
            Some(KvCursor {
                applied: sequence,
                degraded: unknown_before,
                replay_pending: false
            })
        );
    }
}

#[test]
fn each_reason_reaches_the_store_under_its_own_name() {
    let f = Fixture::new();
    let store = batch([Data::Stored(stored(1))]);
    for reason in [
        ResyncReason::OutOfRange,
        ResyncReason::DataLoss,
        ResyncReason::SnapshotAbandoned,
        ResyncReason::PublisherRestart,
        ResyncReason::GapCleared,
        ResyncReason::Snapshot,
    ] {
        f.sink.resync(f.row, Some(RANK), reason);
        assert_eq!(f.tables.resyncs(reason), 1, "{reason:?}");

        f.applied(7, &store);
        f.sink.resync(f.row, Some(0), reason);
        assert_eq!(f.blocks(), 1);
        f.sink.resync(f.row, Some(RANK), reason);
        // An abandoned snapshot keeps the block it showed until the
        // next reason empties the rank. Every other reason empties it.
        let kept = usize::from(reason == ResyncReason::SnapshotAbandoned);
        assert_eq!(f.blocks(), kept, "{reason:?}");
    }

    f.applied(7, &store);
    f.sink.resync(f.row, None, ResyncReason::OutOfRange);
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_cursor_reads_as_the_stores() {
    let f = Fixture::new();
    assert_eq!(f.sink.cursor(f.row, RANK), None);
    f.tables.admit_through(
        f.row,
        RANK,
        BatchFacts {
            sequence: 41,
            clears: false,
            snapshot: Some(SnapshotChunk {
                unknown_before: true,
            }),
        },
        &mut |_| {},
    );
    f.tables.admit_through(
        f.row,
        RANK,
        BatchFacts {
            sequence: 43,
            ..Default::default()
        },
        &mut |_| {},
    );
    let expected = f.tables.cursor(f.row, RANK).expect("the tables' cursor");
    assert!(expected.replay_pending && expected.degraded);
    assert_eq!(f.sink.cursor(f.row, RANK), Some(expected));
    assert_eq!(f.sink.cursor(f.row, 0), None);
}

#[test]
fn a_reconnection_is_reported_to_the_store() {
    let f = Fixture::new();
    f.applied(7, &batch([Data::Stored(stored(1))]));
    f.sink.reconnected(f.row);
    f.applied(6, &batch([]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_worker_reported_gone_loses_its_blocks_and_cursor() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(stored(1))]));
    assert_eq!(f.blocks(), 1);
    f.sink.removed(f.row);
    assert_eq!(f.blocks(), 0);
    assert_eq!(f.tables.cursor(f.row, RANK), None);
}

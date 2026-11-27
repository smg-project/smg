use std::time::Instant;

use smg_grpc_client::common_proto::{KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent};

use super::*;

fn block(hash: i64) -> KvBlock {
    KvBlock {
        block_hash: hash,
        token_ids: (0..4u32)
            .map(|i| (hash as u32).wrapping_mul(16).wrapping_add(i))
            .collect(),
        block_size: 4,
        ..Default::default()
    }
}

fn stored(parent: Option<i64>, hashes: &[i64], tier: KvCacheTier) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: hashes.iter().map(|&hash| block(hash)).collect(),
            parent_block_hash: parent,
            tier: Some(tier as i32),
            medium: Some("GPU".to_string()),
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".to_string()),
            ..Default::default()
        })),
    }
}

fn removed(hashes: &[i64], tier: KvCacheTier) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
            block_hashes: hashes.to_vec(),
            tier: Some(tier as i32),
            ..Default::default()
        })),
    }
}

fn cleared() -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Cleared(
            common::KvCacheCleared::default(),
        )),
    }
}

fn batch(seq: u64, rank: Option<i32>, events: Vec<KvCacheEvent>) -> common::KvEventBatch {
    common::KvEventBatch {
        sequence_number: seq,
        timestamp: 1.0,
        events,
        dp_rank: rank,
        snapshot: None,
        load: None,
    }
}

/// `(rank, tier, hash, parent)` per block of every stored event, in
/// emission order.
fn emitted(chunks: &[common::KvEventBatch]) -> Vec<(Option<i32>, i32, i64, Option<i64>)> {
    let mut out = Vec::new();
    for chunk in chunks {
        for event in &chunk.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                let mut parent = stored.parent_block_hash;
                for block in &stored.blocks {
                    out.push((
                        chunk.dp_rank,
                        stored.tier.unwrap(),
                        block.block_hash,
                        parent,
                    ));
                    parent = Some(block.block_hash);
                }
            }
        }
    }
    out
}

fn chunks_of(state: &LiveState, through: u64) -> Vec<common::KvEventBatch> {
    SnapshotChunks::at(state.snapshot(), through, 2.0, 0).collect()
}

/// A store's tail as the replay hands it out: `(hashes, group_mask)` per
/// stored event of the chunks, in emission order.
fn replayed_masks(chunks: &[common::KvEventBatch]) -> Vec<(Vec<i64>, Option<u64>)> {
    let mut out = Vec::new();
    for chunk in chunks {
        for event in &chunk.events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                out.push((
                    stored.blocks.iter().map(|block| block.block_hash).collect(),
                    stored.group_mask,
                ));
            }
        }
    }
    out
}

#[test]
fn a_snapshot_replays_each_stores_group_mask() {
    let device = KvCacheTier::Device;
    let masked = |parent: Option<i64>, hashes: &[i64], mask: u64| {
        let mut event = stored(parent, hashes, device);
        if let Some(kv_cache_event::Data::Stored(stored)) = &mut event.data {
            stored.group_mask = Some(mask);
        }
        event
    };
    let mut state = LiveState::new();
    // A chain whose boundaries three groups hold, one the first group
    // alone, then a publisher that names no mask at all.
    state.apply(&batch(1, Some(0), vec![masked(None, &[1, 2], 0b111)]));
    state.apply(&batch(2, Some(0), vec![masked(Some(2), &[3], 0b001)]));
    state.apply(&batch(3, Some(0), vec![stored(Some(3), &[4], device)]));
    let replayed = replayed_masks(&chunks_of(&state, 3));
    assert_eq!(
        replayed,
        vec![
            (vec![1, 2], Some(0b111)),
            (vec![3], Some(0b001)),
            (vec![4], None),
        ],
        "the replay keeps every store's mask and splits a run where the mask changes"
    );
}

#[test]
fn stores_add_copies_removals_take_them_and_a_clear_empties_the_rank() {
    let device = KvCacheTier::Device;
    let mut state = LiveState::new();
    state.apply(&batch(1, Some(0), vec![stored(None, &[1, 2], device)]));
    state.apply(&batch(2, Some(0), vec![stored(Some(2), &[3], device)]));
    state.apply(&batch(3, Some(1), vec![stored(None, &[1], device)]));
    assert_eq!((state.entries(), state.blocks(), state.ranks()), (4, 4, 2));
    // A second physical copy of 2 (vLLM's resend) counts without a new entry.
    state.apply(&batch(4, Some(0), vec![stored(Some(1), &[2], device)]));
    assert_eq!((state.entries(), state.blocks()), (4, 5));
    assert_eq!(state.copies(Some(0), device, 2), 2);
    // One removal per copy; the block stays until the last one.
    state.apply(&batch(5, Some(0), vec![removed(&[2], device)]));
    assert_eq!(state.copies(Some(0), device, 2), 1);
    state.apply(&batch(6, Some(0), vec![removed(&[2, 99], device)]));
    assert_eq!(state.copies(Some(0), device, 2), 0);
    assert_eq!(state.counts().removed_unknown, 1);
    assert_eq!((state.entries(), state.blocks()), (3, 3));
    // The host tier is a copy of its own.
    state.apply(&batch(
        7,
        Some(0),
        vec![stored(None, &[1], KvCacheTier::Host)],
    ));
    assert_eq!(state.copies(Some(0), KvCacheTier::Host, 1), 1);
    assert_eq!(state.copies(Some(0), device, 1), 1);
    state.apply(&batch(8, Some(0), vec![removed(&[1], device)]));
    assert_eq!(state.copies(Some(0), device, 1), 0);
    assert_eq!(state.copies(Some(0), KvCacheTier::Host, 1), 1);
    // Clearing rank 0 leaves rank 1 alone.
    state.apply(&batch(9, Some(0), vec![cleared()]));
    assert_eq!((state.entries(), state.blocks(), state.ranks()), (1, 1, 1));
    assert_eq!(state.copies(Some(1), device, 1), 1);
    assert_eq!(state.counts().cleared, 1);
    state.clear();
    assert_eq!((state.entries(), state.blocks(), state.ranks()), (0, 0, 0));
}

#[test]
fn copies_are_capped_like_the_gateway_counts_them() {
    let mut state = LiveState::new();
    for seq in 0..20 {
        state.apply(&batch(
            seq,
            None,
            vec![stored(None, &[7], KvCacheTier::Device)],
        ));
    }
    assert_eq!(state.copies(None, KvCacheTier::Device, 7), COPIES_CAP);
    assert_eq!(state.blocks(), u64::from(COPIES_CAP));
    assert_eq!(state.counts().capped, 20 - u64::from(COPIES_CAP));
    let chunks = chunks_of(&state, 19);
    assert_eq!(emitted(&chunks).len(), COPIES_CAP as usize);
}

/// A store of a cache group other than main attention, as the relay
/// forwards it: the group's hashes and fields, no tokens.
fn group_stored(group: u32, kind: &str, window: Option<u32>, hashes: &[i64]) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: hashes
                .iter()
                .map(|&hash| KvBlock {
                    block_hash: hash,
                    block_size: 4,
                    ..Default::default()
                })
                .collect(),
            parent_block_hash: None,
            tier: Some(KvCacheTier::Device as i32),
            medium: Some("GPU".to_string()),
            group_idx: Some(group),
            kv_cache_spec_kind: Some(kind.to_string()),
            kv_cache_spec_sliding_window: window,
            ..Default::default()
        })),
    }
}

fn group_removed(group: u32, hashes: &[i64]) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
            block_hashes: hashes.to_vec(),
            tier: Some(KvCacheTier::Device as i32),
            group_idx: Some(group),
            ..Default::default()
        })),
    }
}

/// A hybrid model's window and state groups publish their blocks under
/// the main group's hashes: each group's block is an entry on the
/// group's own lane, not a second copy of the main one, its removal
/// takes only its own entry, and a snapshot replays one group-lane
/// entry first, the main lane, and the other group lanes after it, each
/// as stores of their own.
#[test]
fn group_lanes_are_kept_apart_from_the_main_lane_and_one_entry_leads_it() {
    let device = KvCacheTier::Device;
    let mut state = LiveState::new();
    state.apply(&batch(
        1,
        Some(0),
        vec![
            group_stored(1, "sliding_window", Some(8), &[2, 3]),
            group_stored(2, "mamba", None, &[3]),
            stored(None, &[1, 2, 3], device),
        ],
    ));
    assert_eq!((state.entries(), state.blocks()), (6, 6));
    assert_eq!(state.copies(Some(0), device, 3), 1, "one main-lane copy");
    assert_eq!(state.group_copies(Some(0), device, 1, 3), 1);
    assert_eq!(state.group_copies(Some(0), device, 2, 3), 1);
    assert_eq!(
        state.group_copies(Some(0), device, 0, 3),
        0,
        "a main-attention group's blocks are on the main lane"
    );
    // The window moves on: its removal names its group and leaves the
    // main lane's copy and the state group's entry alone.
    state.apply(&batch(2, Some(0), vec![group_removed(1, &[2])]));
    assert_eq!(state.copies(Some(0), device, 2), 1);
    assert_eq!(state.group_copies(Some(0), device, 1, 2), 0);
    assert_eq!(state.counts().removed_unknown, 0);
    let chunks = chunks_of(&state, 2);
    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert_eq!(chunk.snapshot.as_ref().map(|marker| marker.blocks), Some(5));
    let stores: Vec<&KvBlocksStored> = chunk
        .events
        .iter()
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some(stored),
            _ => None,
        })
        .collect();
    assert_eq!(stores.len(), 3, "{stores:#?}");
    assert_eq!(stores[1].group_idx, Some(0), "the main lane comes second");
    assert_eq!(
        stores[1]
            .blocks
            .iter()
            .map(|block| block.block_hash)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(stores[1].blocks[0].token_ids, block(1).token_ids);
    type LaneRow<'a> = (Option<u32>, Option<&'a str>, Option<u32>, Vec<(i64, usize)>);
    let lanes: Vec<LaneRow> = [&stores[0], &stores[2]]
        .into_iter()
        .map(|stored| {
            (
                stored.group_idx,
                stored.kv_cache_spec_kind.as_deref(),
                stored.kv_cache_spec_sliding_window,
                stored
                    .blocks
                    .iter()
                    .map(|block| (block.block_hash, block.token_ids.len()))
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        lanes,
        vec![
            (Some(1), Some("sliding_window"), Some(8), vec![(3, 0)]),
            (Some(2), Some("mamba"), None, vec![(3, 0)]),
        ],
        "one group entry leads and the other follows, hashes only, the removed block gone"
    );
    // The main lane's removal leaves the group lanes' entries alone.
    state.apply(&batch(3, Some(0), vec![removed(&[3], device)]));
    assert_eq!(state.copies(Some(0), device, 3), 0);
    assert_eq!(state.group_copies(Some(0), device, 2, 3), 1);
    // A clear empties every lane and keeps the groups: the next store of
    // group 2 without a kind still lands on its lane.
    state.apply(&batch(4, Some(0), vec![cleared()]));
    assert_eq!((state.entries(), state.blocks()), (0, 0));
    let mut kindless = group_stored(2, "mamba", None, &[9]);
    if let Some(kv_cache_event::Data::Stored(stored)) = &mut kindless.data {
        stored.kv_cache_spec_kind = None;
    }
    state.apply(&batch(5, Some(0), vec![kindless]));
    assert_eq!(state.group_copies(Some(0), device, 2, 9), 1);
    assert_eq!(state.copies(Some(0), device, 9), 0);
}

/// Two ranks of one engine, the first holding main blocks alone: the
/// snapshot puts the rank with a group entry first, so a gateway that
/// pools the ranks in one view has the view before any main lane.
#[test]
fn a_rank_with_a_group_entry_leads_a_snapshot_of_several_ranks() {
    let device = KvCacheTier::Device;
    let mut state = LiveState::new();
    state.apply(&batch(1, Some(0), vec![stored(None, &[1, 2], device)]));
    state.apply(&batch(
        1,
        Some(1),
        vec![
            group_stored(1, "sliding_window", Some(8), &[4]),
            stored(None, &[3, 4], device),
        ],
    ));
    let chunks = chunks_of(&state, 1);
    let stores: Vec<(Option<u32>, Vec<i64>)> = chunks
        .iter()
        .flat_map(|chunk| chunk.events.iter())
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some((
                stored.group_idx,
                stored.blocks.iter().map(|block| block.block_hash).collect(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        stores,
        vec![
            (Some(1), vec![4]),
            (Some(0), vec![3, 4]),
            (Some(0), vec![1, 2])
        ],
        "the group entry of rank 1 leads, then its main lane, then rank 0"
    );
    assert_eq!(
        chunks
            .iter()
            .filter(|chunk| chunk.dp_rank == Some(1))
            .count()
            + chunks
                .iter()
                .filter(|chunk| chunk.dp_rank == Some(0))
                .count(),
        chunks.len(),
        "every chunk carries its own rank"
    );
}

#[test]
fn a_snapshot_lists_the_live_set_parents_first_with_the_store_fields() {
    let device = KvCacheTier::Device;
    let mut state = LiveState::new();
    // A child stored before its (re-stored) parent must still follow it.
    state.apply(&batch(1, Some(0), vec![stored(None, &[1, 2, 3], device)]));
    state.apply(&batch(2, Some(0), vec![removed(&[1], device)]));
    state.apply(&batch(3, Some(0), vec![stored(Some(3), &[4], device)]));
    state.apply(&batch(4, Some(0), vec![stored(None, &[1], device)]));
    state.apply(&batch(5, Some(0), vec![stored(Some(4), &[5], device)]));
    let chunks = chunks_of(&state, 5);
    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert_eq!(chunk.sequence_number, 5);
    assert_eq!(chunk.dp_rank, Some(0));
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 5,
            unknown_before: 0,
        })
    );
    assert!(
        matches!(chunk.events[0].data, Some(kv_cache_event::Data::Cleared(_))),
        "the clear comes first"
    );
    let order = emitted(&chunks);
    assert_eq!(order.len(), 5);
    let position = |hash: i64| order.iter().position(|entry| entry.2 == hash).unwrap();
    for (_, _, hash, parent) in &order {
        if let Some(parent) = parent {
            assert!(
                position(*parent) < position(*hash),
                "{parent} before {hash}: {order:?}"
            );
        }
    }
    // Block 1's chain was broken by the removal: 2 keeps naming 1 as its
    // parent, 1 is live again, so 1 comes first and the run 2, 3 follows.
    assert_eq!(order[0].2, 1);
    assert_eq!(
        &order[1..3],
        &[(Some(0), 1, 2, Some(1)), (Some(0), 1, 3, Some(2))]
    );
    let stores: Vec<&KvBlocksStored> = chunk
        .events
        .iter()
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some(stored),
            _ => None,
        })
        .collect();
    // Every block chains from the one emitted before it under one tail:
    // the five come as one run, the way an engine stores a prompt.
    assert_eq!(stores.len(), 1, "{stores:#?}");
    let first = &stores[0];
    assert_eq!(first.parent_block_hash, None);
    assert_eq!(first.blocks.len(), 5);
    assert_eq!(first.medium.as_deref(), Some("GPU"));
    assert_eq!(first.kv_cache_spec_kind.as_deref(), Some("full_attention"));
    assert_eq!(first.group_idx, Some(0));
    assert_eq!(first.blocks[0].token_ids, block(1).token_ids);
    assert_eq!(first.blocks[0].block_size, 4);
}

#[test]
fn chunks_are_cut_per_rank_and_stamped_up_to_the_cut() {
    let device = KvCacheTier::Device;
    let mut state = LiveState::new();
    let hashes: Vec<i64> = (1..=CHUNK_BLOCKS as i64 + 10).collect();
    state.apply(&batch(1, Some(0), vec![stored(None, &hashes, device)]));
    state.apply(&batch(2, Some(1), vec![stored(None, &[1, 2], device)]));
    let chunks = chunks_of(&state, 40);
    assert_eq!(chunks.len(), 3, "rank 0 takes two chunks, rank 1 one");
    // Each rank's chunks end at its own cut (both ranks stand at 40 here).
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| (chunk.sequence_number, chunk.dp_rank))
            .collect::<Vec<_>>(),
        vec![(39, Some(0)), (40, Some(0)), (40, Some(1))]
    );
    for (index, chunk) in chunks.iter().enumerate() {
        let marker = chunk.snapshot.as_ref().unwrap();
        assert_eq!((marker.index as usize, marker.count), (index, 3));
        assert_eq!(marker.blocks, CHUNK_BLOCKS as u64 + 12);
        assert_eq!(
            matches!(chunk.events[0].data, Some(kv_cache_event::Data::Cleared(_))),
            index == 0
        );
    }
    // The split run continues from the first chunk's last block.
    let second = emitted(&chunks[1..2]);
    assert_eq!(
        second[0],
        (
            Some(0),
            1,
            CHUNK_BLOCKS as i64 + 1,
            Some(CHUNK_BLOCKS as i64)
        )
    );
    assert_eq!(emitted(&chunks).len(), CHUNK_BLOCKS + 12);
}

#[test]
fn few_sequences_behind_a_large_state_mean_larger_chunks() {
    let mut state = LiveState::new();
    let hashes: Vec<i64> = (1..=4 * CHUNK_BLOCKS as i64).collect();
    state.apply(&batch(
        0,
        None,
        vec![stored(None, &hashes, KvCacheTier::Device)],
    ));
    state.apply(&batch(
        1,
        None,
        vec![stored(None, &[-1], KvCacheTier::Device)],
    ));
    // Two sequences exist (0 and 1): at most two chunks, stamped 0 and 1.
    let chunks = chunks_of(&state, 1);
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.sequence_number)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(emitted(&chunks).len(), 4 * CHUNK_BLOCKS + 1);
}

/// A block of `tokens` token ids, spread over a vocabulary so they
/// encode as an engine's would.
fn wide_block(hash: i64, tokens: usize) -> KvBlock {
    KvBlock {
        block_hash: hash,
        token_ids: (0..tokens as u32)
            .map(|i| {
                (hash as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(i.wrapping_mul(7_919))
                    % 150_000
            })
            .collect(),
        block_size: tokens as i32,
        ..Default::default()
    }
}

fn stored_wide(parent: Option<i64>, hashes: &[i64], tokens: usize) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 1,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: hashes
                .iter()
                .map(|&hash| wide_block(hash, tokens))
                .collect(),
            parent_block_hash: parent,
            tier: Some(KvCacheTier::Device as i32),
            medium: Some("GPU".to_string()),
            ..Default::default()
        })),
    }
}

/// The block entries of a chunk's stores, as they are framed on the wire.
fn entry_bytes(chunk: &common::KvEventBatch) -> usize {
    chunk
        .events
        .iter()
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some(stored),
            _ => None,
        })
        .flat_map(|stored| &stored.blocks)
        .map(|block| {
            let len = block.encoded_len();
            1 + varint_len(len as u64) + len
        })
        .sum()
}

#[test]
fn varint_lengths_are_protobufs() {
    for (value, len) in [
        (0, 1),
        (127, 1),
        (128, 2),
        (16_383, 2),
        (16_384, 3),
        (u64::MAX, 10),
    ] {
        assert_eq!(varint_len(value), len, "{value}");
    }
}

#[test]
fn large_blocks_are_cut_by_bytes_and_reassemble() {
    const BLOCKS: i64 = 3_000;
    const TOKENS: usize = 1_152;
    let mut state = LiveState::new();
    // Thirty prompts of a hundred blocks each, as an engine stores them.
    for prompt in 0..BLOCKS / 100 {
        let hashes: Vec<i64> = (prompt * 100 + 1..=prompt * 100 + 100).collect();
        state.apply(&batch(
            prompt as u64 + 1,
            Some(0),
            vec![stored_wide(None, &hashes, TOKENS)],
        ));
    }
    assert_eq!(state.blocks(), BLOCKS as u64);
    let chunks = chunks_of(&state, 10_000);

    // The block bound alone would make two chunks of about 4.8 MB; the
    // bytes make more, each under the cap plus its framing.
    let total: usize = chunks.iter().map(entry_bytes).sum();
    assert!(chunks.len() > 2, "{} chunks", chunks.len());
    assert!(
        chunks.len() <= total.div_ceil(CHUNK_BYTES) + 1,
        "{} chunks for {total} bytes",
        chunks.len()
    );
    for (index, chunk) in chunks.iter().enumerate() {
        assert!(
            entry_bytes(chunk) <= CHUNK_BYTES,
            "chunk {index} holds {} bytes of entries",
            entry_bytes(chunk)
        );
        assert!(
            chunk.encoded_len() <= CHUNK_BYTES + 64 * 1024,
            "chunk {index} encodes to {} bytes",
            chunk.encoded_len()
        );
        let marker = chunk.snapshot.as_ref().unwrap();
        assert_eq!(
            (marker.index as usize, marker.count as usize),
            (index, chunks.len())
        );
        assert_eq!(marker.blocks, BLOCKS as u64);
        assert_eq!(
            chunk.sequence_number,
            10_000 + 1 - chunks.len() as u64 + index as u64
        );
        assert_eq!(
            matches!(chunk.events[0].data, Some(kv_cache_event::Data::Cleared(_))),
            index == 0
        );
    }

    // Every block comes back once, under its parent, with its tokens.
    let order = emitted(&chunks);
    assert_eq!(order.len(), BLOCKS as usize);
    let mut hashes: Vec<i64> = order.iter().map(|entry| entry.2).collect();
    hashes.sort_unstable();
    assert_eq!(hashes, (1..=BLOCKS).collect::<Vec<_>>());
    for (_, _, hash, parent) in &order {
        let expected = if (hash - 1) % 100 == 0 {
            None
        } else {
            Some(hash - 1)
        };
        assert_eq!(*parent, expected, "parent of {hash}");
    }
    let tokens: HashMap<i64, Vec<u32>> = chunks
        .iter()
        .flat_map(|chunk| &chunk.events)
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some(stored),
            _ => None,
        })
        .flat_map(|stored| &stored.blocks)
        .map(|block| (block.block_hash, block.token_ids.clone()))
        .collect();
    for hash in 1..=BLOCKS {
        assert_eq!(
            tokens[&hash],
            wide_block(hash, TOKENS).token_ids,
            "tokens of {hash}"
        );
    }
}

#[test]
fn the_byte_bound_yields_to_the_stamping_rule_like_the_block_bound() {
    let mut state = LiveState::new();
    let hashes: Vec<i64> = (1..=3_000).collect();
    state.apply(&batch(0, None, vec![stored_wide(None, &hashes, 1_152)]));
    state.apply(&batch(1, None, vec![stored_wide(None, &[-1], 1_152)]));
    // Two sequences exist (0 and 1): at most two chunks, stamped 0 and 1,
    // however many bytes they take.
    let chunks = chunks_of(&state, 1);
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.sequence_number)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(chunks.iter().any(|chunk| entry_bytes(chunk) > CHUNK_BYTES));
    assert_eq!(emitted(&chunks).len(), 3_001);
}

#[test]
fn an_empty_state_snapshots_as_one_clear() {
    let state = LiveState::new();
    let chunks = chunks_of(&state, 12);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].sequence_number, 12);
    assert_eq!(chunks[0].events.len(), 1);
    assert_eq!(
        chunks[0].snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 0,
            unknown_before: 0,
        })
    );
    assert_eq!(chunks[0].dp_rank, None);
    // A record that starts late says so on every chunk.
    let late: Vec<common::KvEventBatch> =
        SnapshotChunks::at(state.snapshot(), 12, 2.0, 7).collect();
    assert_eq!(
        late[0].snapshot.as_ref().map(|chunk| chunk.unknown_before),
        Some(7)
    );
}

/// A relay whose rank 1 relayed (and freed) everything while rank 0 has
/// not spoken yet cuts an empty state at rank 1's cursor: the clear chunk
/// names rank 1, so the stamp moves rank 1's cursor at the gateway and
/// rank 0's first live batch is taken as its first.
#[test]
fn an_empty_states_clear_chunk_names_the_first_cuts_rank() {
    let cuts = BTreeMap::from([(
        Some(1),
        RankCut {
            through: 7,
            unknown_before: 2,
        },
    )]);
    let chunks: Vec<common::KvEventBatch> =
        SnapshotChunks::new(LiveState::new().snapshot(), &cuts, 2.0).collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!((chunks[0].dp_rank, chunks[0].sequence_number), (Some(1), 7));
    assert_eq!(
        chunks[0]
            .snapshot
            .as_ref()
            .map(|chunk| chunk.unknown_before),
        Some(2)
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured pass is this test's report; read it with --nocapture"
)]
fn a_snapshot_of_a_large_state_is_one_brief_pass() {
    // A 676k-block pool (a large worker's) in chains of 64.
    let mut state = LiveState::new();
    let mut hash = 1i64;
    let started = Instant::now();
    for seq in 0..(676_144 / 64) {
        let hashes: Vec<i64> = (hash..hash + 64).collect();
        hash += 64;
        state.apply(&batch(
            seq,
            Some(0),
            vec![stored(None, &hashes, KvCacheTier::Device)],
        ));
    }
    let built = started.elapsed();
    assert_eq!(state.blocks(), 676_096);
    let started = Instant::now();
    let snapshot = state.snapshot();
    let collected = started.elapsed();
    let started = Instant::now();
    let mut chunks = SnapshotChunks::at(snapshot, 20_000, 3.0, 0);
    let ordered = started.elapsed();
    let started = Instant::now();
    let mut blocks = 0;
    let count = chunks.chunk_count();
    while let Some(chunk) = chunks.next_chunk() {
        blocks += emitted(std::slice::from_ref(&chunk)).len();
    }
    let encoded = started.elapsed();
    eprintln!(
        "676k snapshot: state built in {built:?}, collect {collected:?}, order {ordered:?}, {count} chunks built in {encoded:?}"
    );
    assert_eq!(blocks, 676_096);
    assert_eq!(count as usize, 676_096_usize.div_ceil(CHUNK_BLOCKS));
    // Brief by the clock, or by this host's own pace: a loaded host
    // stretches the build above and the collect alike, so a collect past
    // the second is slow by its own work only when it also outlasts the
    // build of the state it walks.
    assert!(
        collected.as_millis() < 1_000 || collected < built,
        "collecting under the lock took {collected:?}; building the state took {built:?}"
    );
}

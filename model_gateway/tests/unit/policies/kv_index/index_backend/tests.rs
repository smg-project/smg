use kv_index::compute_content_hash;

use super::*;

fn chain(contents: &[&[u32]]) -> Vec<StoredBlock> {
    let hashes: Vec<ContentHash> = contents.iter().map(|c| compute_content_hash(c)).collect();
    hashes
        .iter()
        .zip(request_prefix_hashes(&hashes))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}

fn backends() -> Vec<KvIndex> {
    vec![KvIndex::chain(), KvIndex::reference()]
}

#[test]
fn every_backend_scores_stores_removals_and_clears_the_same_way() {
    let blocks = chain(&[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 10, 11, 12]]);
    let request: Vec<ContentHash> = blocks.iter().map(|b| b.content_hash).collect();
    for index in backends() {
        let name = index.name();
        assert!(index.is_empty(), "{name}: empty at start");
        let w1 = index.intern_worker("grpc://w1").unwrap();
        let w2 = index.intern_worker("grpc://w2").unwrap();
        assert_eq!(index.worker_id("grpc://w2"), Some(w2), "{name}");
        let mut held1 = WorkerBlocks::default();
        let mut held2 = WorkerBlocks::default();
        index.apply_stored(w1, &blocks, None, &mut held1).unwrap();
        index
            .apply_stored(w2, &blocks[..2], None, &mut held2)
            .unwrap();
        assert!(!index.is_empty(), "{name}: not empty after stores");
        assert!(held1.contains_key(blocks[2].seq_hash), "{name}");
        assert!(!held2.contains_key(blocks[2].seq_hash), "{name}");
        assert_eq!(held1.len(), 3, "{name}");
        assert_eq!(index.worker_block_count(w1), 3, "{name}");
        assert_eq!(index.worker_block_count(w2), 2, "{name}");
        assert_eq!(index.current_size(), 5, "{name}");

        let scores = index.find_matches(&request, false).scores;
        assert_eq!(scores.get(&w1), Some(&3), "{name}: w1 holds the chain");
        assert_eq!(scores.get(&w2), Some(&2), "{name}: w2 holds two blocks");
        let early = index.find_matches(&request, true).scores;
        assert_eq!(early.get(&w1), Some(&1), "{name}: early exit scores 1");
        assert_eq!(early.get(&w2), Some(&1), "{name}: early exit scores 1");

        // A middle removal ends w1's match at the hole; w2 is untouched.
        index.apply_removed(w1, &[blocks[1].seq_hash], &mut held1);
        let scores = index.find_matches(&request, false).scores;
        assert_eq!(scores.get(&w1), Some(&1), "{name}: hole ends the match");
        assert_eq!(scores.get(&w2), Some(&2), "{name}");
        assert_eq!(index.worker_block_count(w1), 2, "{name}");

        // A store after the parent heals the hole.
        index
            .apply_stored(w1, &blocks[1..2], Some(blocks[0].seq_hash), &mut held1)
            .unwrap();
        let scores = index.find_matches(&request, false).scores;
        assert_eq!(scores.get(&w1), Some(&3), "{name}: healed");

        // An unknown parent is reported, as the monitor's fallback expects.
        assert!(
            matches!(
                index.apply_stored(w2, &blocks[2..], Some(SequenceHash(0xdead)), &mut held2),
                Err(ApplyError::ParentBlockNotFound)
            ),
            "{name}"
        );
        let mut fresh = WorkerBlocks::default();
        let w3 = index.intern_worker("grpc://w3").unwrap();
        assert!(
            matches!(
                index.apply_stored(w3, &blocks[1..], Some(blocks[0].seq_hash), &mut fresh),
                Err(ApplyError::WorkerNotTracked)
            ),
            "{name}"
        );

        index.apply_cleared(w1, &mut held1);
        assert!(held1.is_empty(), "{name}: cleared state is empty");
        assert_eq!(index.worker_block_count(w1), 0, "{name}");
        assert!(
            !index.find_matches(&request, false).scores.contains_key(&w1),
            "{name}: cleared worker scores nothing"
        );
        index.remove_worker(w2, held2);
        assert_eq!(index.worker_block_count(w2), 0, "{name}");
        assert!(index.is_empty(), "{name}: empty again");
        assert_eq!(index.current_size(), 0, "{name}");
        assert!(index.debug_blocks().is_empty(), "{name}");
    }
}

#[test]
fn the_chain_index_hands_back_removed_worker_ids_and_knows_when_it_is_empty() {
    let index = KvIndex::chain();
    let first = index.intern_worker("grpc://a").unwrap();
    let mut held = WorkerBlocks::default();
    let blocks = chain(&[&[1, 2, 3, 4]]);
    index.apply_stored(first, &blocks, None, &mut held).unwrap();
    index.remove_worker(first, held);
    assert_eq!(index.worker_id("grpc://a"), None);
    assert!(index.is_empty());
    // The freed id is reused; the emptiness check still covers it.
    let again = index.intern_worker("grpc://b").unwrap();
    assert_eq!(again, first);
    let mut held = WorkerBlocks::default();
    index.apply_stored(again, &blocks, None, &mut held).unwrap();
    assert!(!index.is_empty());
    assert_eq!(index.chain_stats().map(|s| s.blocks_live), Some(1));
}

#[test]
fn the_index_names_itself() {
    assert_eq!(KvIndex::chain().name(), "chain");
    assert!(KvIndex::chain().chain_stats().is_some());
    assert!(KvIndex::reference().chain_stats().is_none());
    assert_eq!(
        format!("{:?}", KvIndex::chain()),
        "KvIndex { kind: \"chain\", blocks: 0 }"
    );
}

/// A mask that names `ids`, a bit an id.
fn naming(ids: &[u32]) -> Vec<u64> {
    let mut mask = vec![0u64; 1];
    for &id in ids {
        let word = (id / 64) as usize;
        if mask.len() <= word {
            mask.resize(word + 1, 0);
        }
        mask[word] |= 1 << (id % 64);
    }
    mask
}

/// Grand Central's lookup: the holders of a request among the
/// workers its mask names, the deepest first under a cap, and the depth
/// the cap cuts drawn from. Both backends answer alike.
#[test]
fn the_engines_lookup_keeps_the_deepest_holders_its_mask_names() {
    let blocks = chain(&[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 10, 11, 12]]);
    let request: Vec<u64> = blocks.iter().map(|b| b.content_hash.0).collect();
    for index in backends() {
        let name = index.name();
        let ids: Vec<u32> = (0..4)
            .map(|at| index.intern_worker(&format!("grpc://w{at}")).unwrap())
            .collect();
        // Three blocks, two, two and one.
        for (&id, depth) in ids.iter().zip([3, 2, 2, 1]) {
            let mut held = WorkerBlocks::default();
            index
                .apply_stored(id, &blocks[..depth], None, &mut held)
                .unwrap();
        }
        assert_eq!(index.worker_number("grpc://w2"), Some(ids[2]), "{name}");
        assert_eq!(index.worker_number("grpc://nobody"), None, "{name}");
        assert_eq!(index.worker_blocks(ids[0]), 3, "{name}");
        let lookup = |within: &[u32], cap: usize, drawn: u32| {
            let mut kept = vec![(u32::MAX, 0)];
            index.deepest_into(&request, 4, &naming(within), cap, &mut |_| drawn, &mut kept);
            kept.sort_unstable();
            kept
        };
        let whole = [(ids[0], 3), (ids[1], 2), (ids[2], 2), (ids[3], 1)];
        assert_eq!(lookup(&ids, 16, 0), whole, "{name}");
        // The deepest fits, and the cap cuts the two of the next depth:
        // one of them stands, by the draw.
        let cut: Vec<_> = (0..2).map(|drawn| lookup(&ids, 2, drawn)).collect();
        for kept in &cut {
            assert_eq!(kept.len(), 2, "{name}");
            assert!(kept.contains(&whole[0]), "{name}: {kept:?}");
            assert!(
                kept.iter().all(|&(_, depth)| depth >= 2),
                "{name}: {kept:?}"
            );
        }
        assert_ne!(cut[0], cut[1], "{name}: the draw chooses");
        assert_eq!(lookup(&ids, 1, 0), [whole[0]], "{name}");
        // A worker the mask does not name is no holder, whatever it holds.
        assert_eq!(lookup(&ids[1..], 16, 0), whole[1..], "{name}");
        assert!(lookup(&ids, 0, 0).is_empty(), "{name}");
        assert!(lookup(&[], 16, 0).is_empty(), "{name}");
    }
}

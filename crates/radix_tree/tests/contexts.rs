use radix_tree::{Config, ContextError, OverlapScratch, RadixTree};

#[test]
fn contexts_share_paths_and_match_only_complete_endpoints() {
    let mut tree = RadixTree::new(Config::default());
    let parent = tree.learn_context(None, &[1, 2]).unwrap();
    let tail = tree.learn_context(Some(parent), &[3, 4]).unwrap();
    let fork = tree.learn_context(Some(parent), &[8, 9]).unwrap();
    assert_eq!(tree.learn_context(None, &[1, 2, 3, 4]), Ok(tail));
    let middle = tree.learn_context(Some(parent), &[3]).unwrap();
    assert_eq!(tree.retained_contents(), 6);

    let mut scratch = OverlapScratch::default();
    let mut matched = Vec::new();
    tree.matching_contexts(&[1, 2, 3, 4, 5], &mut scratch, &mut matched);
    assert_eq!(matched, [parent, middle, tail]);
    tree.matching_contexts(&[1, 2, 8], &mut scratch, &mut matched);
    assert_eq!(matched, [parent]);
    tree.matching_contexts(&[1, 2, 8, 9], &mut scratch, &mut matched);
    assert_eq!(matched, [parent, fork]);
    tree.matching_contexts(&[1, 2, 3, 99], &mut scratch, &mut matched);
    assert_eq!(matched, [parent, middle]);
    tree.matching_contexts(&[], &mut scratch, &mut matched);
    assert!(matched.is_empty());
    tree.matching_contexts(&[99], &mut scratch, &mut matched);
    assert!(matched.is_empty());
    assert_eq!(tree.stats().holder_blocks, 0);
    tree.audit().unwrap();
}

#[test]
fn historical_parent_survives_point_eviction_and_unrelated_chain_reuse() {
    let mut tree = RadixTree::new(Config::default());
    let worker = tree.create_holder("worker");
    tree.store(worker, None, &[(10, 1), (20, 2)]).unwrap();
    let parent = tree.learn_context(None, &[1, 2]).unwrap();
    tree.remove(worker, &[10, 20]);
    tree.store(worker, None, &[(30, 7)]).unwrap();
    tree.clear(worker);
    assert_eq!(tree.retained_contents(), 2);
    tree.store(worker, None, &[(40, 8)]).unwrap();
    tree.retire_holder(worker);
    let tail = tree.learn_context(Some(parent), &[3, 4]).unwrap();

    let mut scratch = OverlapScratch::default();
    let mut matched = Vec::new();
    tree.matching_contexts(&[1, 2, 3, 4], &mut scratch, &mut matched);
    assert_eq!(matched, [parent, tail]);
    let mut overlap = Vec::new();
    tree.overlap(&[1, 2, 3, 4], &mut scratch, &mut overlap);
    assert!(overlap.is_empty());
    assert_eq!(tree.retained_contents(), 4);
    tree.audit().unwrap();
}

#[test]
fn released_contexts_share_the_existing_whole_chain_lifecycle() {
    let mut tree = RadixTree::new(Config::default());
    let parent = tree.learn_context(None, &[1, 2]).unwrap();
    let tail = tree.learn_context(Some(parent), &[3, 4]).unwrap();
    let fork = tree.learn_context(Some(parent), &[8, 9]).unwrap();
    assert!(tree.release_context(parent));
    assert!(tree.release_context(tail));
    assert_eq!(tree.retained_contents(), 6); // Existing GC keeps the parent chain's tail.
    assert_eq!(tree.drain_retired_contexts().count(), 0);
    assert!(tree.retain_context(parent));
    assert!(tree.release_context(fork));
    assert_eq!(tree.drain_retired_contexts().collect::<Vec<_>>(), [fork]);
    assert_eq!(tree.retained_contents(), 4);
    assert!(tree.release_context(parent));
    let retired = tree.drain_retired_contexts().collect::<Vec<_>>();
    assert_eq!(retired.len(), 2);
    assert!(retired.contains(&parent) && retired.contains(&tail));
    assert_eq!(tree.retained_contents(), 0);

    // Slot reuse cannot turn an old handle into an unrelated prefix.
    tree.learn_context(None, &[7, 8]).unwrap();
    assert!(!tree.retain_context(parent));
    assert_eq!(
        tree.learn_context(Some(parent), &[3]),
        Err(ContextError::InvalidParent)
    );
    tree.audit().unwrap();
}

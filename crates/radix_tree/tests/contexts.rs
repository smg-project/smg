use radix_tree::{Config, ContextError, FlatTree, HolderId, OverlapScratch, RadixTree};

#[test]
fn context_descriptions_follow_forks_and_released_ancestors() {
    let mut tree = RadixTree::new(Config::default());
    let root = tree.learn_context(None, &[1, 2]).unwrap();
    let tail = tree.learn_context(Some(root), &[3, 4]).unwrap();
    let fork = tree.learn_context(Some(root), &[8, 9]).unwrap();
    let middle = tree.learn_context(Some(root), &[3]).unwrap();
    assert_eq!(tree.context_parent(tail), Some(middle));
    assert_eq!(tree.context_parent(fork), Some(root));
    assert_eq!(tree.context_parent(root), None);
    assert_eq!(tree.context_contents(tail), Some(vec![1, 2, 3, 4]));
    assert_eq!(tree.context_suffix(tail, 2), Some(vec![3, 4]));
    assert_eq!(tree.context_suffix(fork, 3), Some(vec![2, 8, 9]));
    assert_eq!(tree.context_suffix(fork, 0), Some(Vec::new()));
    assert_eq!(tree.context_suffix(fork, 5), None);
    assert_eq!(
        tree.context_contents_since(tail, Some(middle)),
        Some(vec![4])
    );
    assert_eq!(
        tree.context_contents_since(fork, Some(root)),
        Some(vec![8, 9])
    );
    assert_eq!(tree.context_contents_since(fork, Some(middle)), None);
    assert_eq!(tree.context_contents_since(root, Some(tail)), None);
    assert!(tree.release_context(root));
    assert_eq!(tree.context_contents(fork), Some(vec![1, 2, 8, 9]));
    for context in [middle, tail, fork] {
        tree.release_context(context);
    }
    assert_eq!(tree.context_contents(root), None);
    assert_eq!(tree.context_contents(fork), None);
    tree.audit().unwrap();
}

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

#[test]
fn context_lifecycle_preserves_holder_membership() {
    fn check(tree: &RadixTree, flat: &FlatTree, holders: &[HolderId]) {
        tree.audit().unwrap();
        flat.audit().unwrap();
        assert_eq!(tree.stats().holders, flat.stats().holders);
        assert_eq!(tree.stats().holder_blocks, flat.stats().holder_blocks);
        assert_eq!(tree.stats().distinct_entries, flat.stats().distinct_entries);
        for &holder in holders {
            assert_eq!(tree.holder_blocks(holder), flat.holder_blocks(holder));
            assert_eq!(
                tree.enumerate(holder).collect::<Vec<_>>(),
                flat.enumerate(holder).collect::<Vec<_>>()
            );
        }
        let mut scratch = OverlapScratch::default();
        let mut actual = Vec::new();
        let mut expected = Vec::new();
        for query in [&[1, 2, 3, 4][..], &[1, 2, 8, 9], &[1, 2, 3, 99]] {
            tree.overlap(query, &mut scratch, &mut actual);
            flat.overlap(query, &mut scratch, &mut expected);
            actual.sort_unstable_by_key(|row| row.holder.parts());
            expected.sort_unstable_by_key(|row| row.holder.parts());
            assert_eq!(actual, expected);
        }
    }
    let mut tree = RadixTree::new(Config::default());
    let mut flat = FlatTree::new(Config::default());
    let holders: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|name| {
            let holder = tree.create_holder(name);
            assert_eq!(holder, flat.create_holder(name));
            holder
        })
        .collect();
    let parent = tree.learn_context(None, &[1, 2]).unwrap();
    let tail = tree.learn_context(Some(parent), &[3, 4]).unwrap();
    let fork = tree.learn_context(Some(parent), &[8, 9]).unwrap();
    check(&tree, &flat, &holders); // Learned history alone has no membership.

    let first = [(10, 1), (20, 2), (30, 3), (40, 4)];
    assert_eq!(
        tree.store(holders[0], None, &first),
        flat.store(holders[0], None, &first)
    );
    check(&tree, &flat, &holders);
    let second = [(50, 1), (60, 2), (70, 8), (80, 9)];
    assert_eq!(
        tree.store(holders[1], None, &second),
        flat.store(holders[1], None, &second)
    );
    check(&tree, &flat, &holders);
    assert!(tree.release_context(parent));
    check(&tree, &flat, &holders);
    assert!(tree.retain_context(parent));
    check(&tree, &flat, &holders);
    assert!(tree.release_context(parent));
    check(&tree, &flat, &holders);
    assert_eq!(
        tree.remove(holders[0], &[20, 40]),
        flat.remove(holders[0], &[20, 40])
    );
    check(&tree, &flat, &holders);
    tree.clear(holders[0]);
    flat.clear(holders[0]);
    check(&tree, &flat, &holders);
    assert!(tree.release_context(tail));
    check(&tree, &flat, &holders);
    tree.retire_holder(holders[1]);
    flat.retire_holder(holders[1]);
    check(&tree, &flat, &holders); // Fork history keeps ancestry, not residency.
    assert!(tree.release_context(fork));
    let retired = tree.drain_retired_contexts().collect::<Vec<_>>();
    assert_eq!(retired.len(), 3);
    assert!(retired.contains(&parent) && retired.contains(&tail) && retired.contains(&fork));
    assert_eq!(tree.retained_contents(), 0);
    check(&tree, &flat, &holders);

    // Drain caller identity records before identical-path recreation. The new
    // canonical handle may equal an old one.
    let recreated = tree.learn_context(None, &[1, 2, 3, 4]).unwrap();
    let mut matched = Vec::new();
    tree.matching_contexts(&[1, 2, 3, 4], &mut OverlapScratch::default(), &mut matched);
    assert_eq!(matched, [recreated]);
    check(&tree, &flat, &holders);
    assert!(tree.release_context(recreated));
    assert!(tree
        .drain_retired_contexts()
        .any(|context| context == recreated));
    check(&tree, &flat, &holders);
}

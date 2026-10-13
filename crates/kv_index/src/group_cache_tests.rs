use std::collections::BTreeSet;

use serde_json::Value;

use super::*;

fn query(cache: &GroupCache, tokens: &[u32]) -> Option<usize> {
    cache.reusable_tokens(&GroupRequest::new(tokens))
}

fn fixtures() -> Value {
    serde_json::from_str(include_str!("../tests/fixtures/vllm-group-cache.json")).unwrap()
}

fn number(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}

fn key(hex: &str) -> CacheKey {
    assert!(hex.len().is_multiple_of(2));
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn opaque(end: usize) -> CacheKey {
    (end as u64).to_be_bytes().to_vec()
}

fn tokens(end: usize) -> Vec<u32> {
    (1..=end as u32).collect()
}

fn store(
    group_id: u32,
    kind: &str,
    window: usize,
    block_size: usize,
    keys: Vec<CacheKey>,
    parent: Option<CacheKey>,
    tokens: Vec<u32>,
) -> GroupEvent {
    GroupEvent::Store {
        group_id,
        kind: kind.to_owned(),
        window,
        keys,
        parent,
        tokens,
        block_size,
        tokens_matchable: true,
        skipped_tokens: None,
        skipped_token_parent_hash: None,
        copy_counts: Vec::new(),
    }
}

fn root_report(
    group_id: u32,
    kind: &str,
    window: usize,
    span: usize,
    ends: &[usize],
    through: usize,
) -> GroupEvent {
    store(
        group_id,
        kind,
        window,
        span,
        ends.iter().map(|&end| opaque(end)).collect(),
        None,
        tokens(through),
    )
}

// Numbers retain the original semantic scenarios' batch labels; admission is external.
fn apply(cache: &mut GroupCache, _batch: u64, events: &[GroupEvent]) {
    cache.apply_events(events).unwrap();
}

fn child_report(group_id: u32, kind: &str, window: usize, start: usize, end: usize) -> GroupEvent {
    store(
        group_id,
        kind,
        window,
        end - start,
        vec![opaque(end)],
        (start > 0).then(|| opaque(start)),
        ((start + 1) as u32..=end as u32).collect(),
    )
}

fn remove(group_id: u32, ends: &[usize]) -> GroupEvent {
    GroupEvent::Remove {
        copy_counts: Vec::new(),
        group_id,
        keys: ends.iter().map(|&end| opaque(end)).collect(),
    }
}

// The snapshot adapter below explicitly seeds complete prefix history. Compute
// expectations from its declared cached endpoints using per-token coverage,
// independently of the SUT's hashes, interval sweep and event reconstruction.
fn fixture_expected(case: &Value) -> Option<usize> {
    let request_len = number(&case["request_tokens"]);
    let unit = number(&case["hash_block_size"]);
    let mut candidates = BTreeSet::from([0]);
    let mut requirements = Vec::new();
    for group in case["groups"].as_array().unwrap() {
        let kind = group["kind"].as_str().unwrap();
        let window = group["window"].as_u64().unwrap_or(0) as usize;
        if !matches!(kind, "full" | "swa" | "mamba") || (kind == "swa" && window == 0) {
            return None;
        }
        let physical = number(&group["block_size"]);
        let mut covered = vec![false; request_len];
        let mut checkpoints = BTreeSet::new();
        for end in group["cached"].as_array().unwrap().iter().map(number) {
            let span = if end.is_multiple_of(physical) {
                physical
            } else {
                unit
            };
            if span > end || end > request_len {
                continue;
            }
            covered[end - span..end].fill(true);
            checkpoints.insert(end);
            if end < request_len {
                candidates.insert(end);
            }
        }
        requirements.push((kind, window, covered, checkpoints));
    }
    if requirements.is_empty() {
        return None;
    }
    candidates.into_iter().rev().find(|&end| {
        end == 0
            || requirements
                .iter()
                .all(|(kind, window, covered, checkpoints)| {
                    if *kind == "mamba" {
                        checkpoints.contains(&end)
                    } else {
                        let start = if *kind == "swa" {
                            end.saturating_sub(window.saturating_sub(1).max(1))
                        } else {
                            0
                        };
                        covered[start..end].iter().all(|&present| present)
                    }
                })
    })
}

fn fixture_key(case: &Value, end: usize) -> CacheKey {
    let unit = number(&case["hash_block_size"]);
    assert_eq!(end % unit, 0);
    if let Some(hash) = case["request_hashes"]
        .as_array()
        .unwrap()
        .get(end / unit - 1)
    {
        key(hash.as_str().unwrap())
    } else {
        // The fixture exports no native hash beyond its request. Preserve such
        // history with a clearly synthetic opaque identity, not a forged digest.
        let mut synthetic = b"fixture-history:".to_vec();
        synthetic.extend_from_slice(&(end as u64).to_be_bytes());
        synthetic
    }
}

// Old snapshots contain privileged endpoints. Make that context explicit in a
// synthetic received history: learn dense context, remove that membership, then
// report the desired entries. No endpoints, native alignment or spec equality
// are supplied directly to the SUT. This is not an original engine event trace.
fn fixture_events(case: &Value) -> Vec<GroupEvent> {
    let unit = number(&case["hash_block_size"]);
    let groups = case["groups"].as_array().unwrap();
    let through = groups
        .iter()
        .flat_map(|g| g["cached"].as_array().unwrap())
        .map(number)
        .max()
        .unwrap_or(0)
        .max(number(&case["request_tokens"]) / unit * unit);
    let context_keys: Vec<_> = (unit..=through)
        .step_by(unit)
        .map(|end| fixture_key(case, end))
        .collect();
    let mut events = Vec::new();
    for (id, group) in groups.iter().enumerate() {
        let kind = match group["kind"].as_str().unwrap() {
            "full" => "full_attention",
            "swa" => "sliding_window",
            other => other,
        };
        let window = group["window"].as_u64().unwrap_or(0) as usize;
        events.push(store(
            id as u32,
            kind,
            window,
            unit,
            context_keys.clone(),
            None,
            tokens(through),
        ));
        events.push(GroupEvent::Remove {
            copy_counts: Vec::new(),
            group_id: id as u32,
            keys: context_keys.clone(),
        });
        let physical = number(&group["block_size"]);
        for cached in group["cached"].as_array().unwrap() {
            let end = number(cached);
            let span = if end.is_multiple_of(physical) {
                physical
            } else {
                unit
            };
            let start = end - span;
            events.push(store(
                id as u32,
                kind,
                window,
                span,
                vec![fixture_key(case, end)],
                (start > 0).then(|| fixture_key(case, start)),
                ((start + 1) as u32..=end as u32).collect(),
            ));
        }
    }
    events
}

fn classification(case: &Value) -> &'static str {
    let groups = case["groups"].as_array().unwrap();
    if groups
        .iter()
        .any(|g| g["cacheable"] == false || g["kind"] == "noncacheable")
    {
        "unavailable prefix-cacheability and global alignment"
    } else if case["partial_hash_hits"] == true {
        "unavailable partial gate, physical extent, or exact spec equality"
    } else if groups
        .iter()
        .map(|g| number(&g["block_size"]))
        .collect::<BTreeSet<_>>()
        .len()
        > 1
    {
        "unavailable native global alignment or partial gate"
    } else {
        "native expectation retained with explicit observed context"
    }
}

#[test]
fn received_state_fixture_scenarios_with_explicit_native_subset() {
    let data = fixtures();
    let (mut total, mut native_subset) = (0, 0);
    for collection in [
        "fixtures",
        "randomized_fixtures",
        "boundary_lookup_fixtures",
    ] {
        for case in data[collection].as_array().unwrap() {
            let events = fixture_events(case);
            let request = tokens(number(&case["request_tokens"]));
            let expected = fixture_expected(case);
            let mut cache = GroupCache::default();
            apply(&mut cache, 0, &events);
            assert_eq!(
                query(&cache, &request),
                expected,
                "{}: {}",
                case["name"],
                classification(case)
            );
            if classification(case) == "native expectation retained with explicit observed context"
            {
                assert_eq!(
                    expected,
                    Some(number(&case["expected_hit_tokens"])),
                    "native subset: {}",
                    case["name"]
                );
                native_subset += 1;
            }
            total += 1;
        }
    }
    assert_eq!((total, native_subset), (159, 63));
}

#[test]
fn newly_observed_group_joins_without_a_roster_or_contiguous_group_ids() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        17,
        &[
            root_report(3, "full_attention", 0, 2, &[2, 4, 6, 8], 8),
            root_report(9, "full_attention", 0, 2, &[2, 4, 6, 8], 8),
        ],
    );
    assert_eq!(query(&cache, &tokens(11)), Some(8));
    apply(&mut cache, 18, &[root_report(27, "mamba", 0, 2, &[6], 6)]);
    assert_eq!(query(&cache, &tokens(11)), Some(6));
    apply(&mut cache, 19, &[remove(27, &[6])]);
    assert_eq!(query(&cache, &tokens(11)), Some(0));
}

#[test]
fn sparse_joint_resume_is_24_even_though_independent_maxima_are_32_and_40() {
    let mut cache = GroupCache::default();
    let all: Vec<_> = (4..=40).step_by(4).collect();
    apply(
        &mut cache,
        0,
        &[
            root_report(0, "full_attention", 0, 4, &all, 40),
            remove(0, &[36, 40]),
        ],
    );
    assert_eq!(query(&cache, &tokens(41)), Some(32));
    apply(
        &mut cache,
        1,
        &[root_report(1, "mamba", 0, 4, &[24, 40], 40)],
    );
    assert_eq!(query(&cache, &tokens(41)), Some(24));
}

#[test]
fn sparse_hashes_are_not_ordinal_zipped_and_cross_group_context_resolves_them() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(1, "mamba", 0, 2, &[4, 8, 10, 12], 12)],
    );
    assert_eq!(query(&cache, &tokens(9)), Some(0));
    apply(
        &mut cache,
        1,
        &[root_report(
            0,
            "full_attention",
            0,
            2,
            &[2, 4, 6, 8, 10, 12],
            12,
        )],
    );
    assert_eq!(query(&cache, &tokens(7)), Some(4));
    assert_eq!(query(&cache, &tokens(9)), Some(8));
    assert_eq!(query(&cache, &tokens(13)), Some(12));
}

#[test]
fn wholly_evicted_history_requires_a_new_anchor() {
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[child_report(0, "mamba", 0, 2, 4)]);
    assert_eq!(query(&cache, &[3, 4, 5]), Some(0));
    assert_eq!(query(&cache, &tokens(5)), Some(0));
    apply(&mut cache, 1, &[root_report(0, "mamba", 0, 2, &[2], 2)]);
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    apply(&mut cache, 2, &[remove(0, &[2, 4])]);
    assert_eq!(query(&cache, &tokens(7)), Some(0));
    assert_eq!(cache.index.retained_contents(), 0);
    assert!(cache.positions.is_empty());
    assert!(cache.by_prefix.is_empty());
    apply(&mut cache, 3, &[child_report(0, "mamba", 0, 4, 6)]);
    assert_eq!(query(&cache, &tokens(7)), Some(0));
    assert_eq!(query(&cache, &[5, 6, 7]), Some(0));
    apply(&mut cache, 4, &[root_report(0, "mamba", 0, 4, &[4], 4)]);
    assert_eq!(query(&cache, &tokens(7)), Some(6));
}

#[test]
fn pending_ancestry_lives_only_while_a_report_needs_it() {
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[child_report(0, "mamba", 0, 2, 4)]);
    apply(&mut cache, 1, &[child_report(0, "mamba", 0, 4, 6)]);
    apply(&mut cache, 2, &[remove(0, &[4])]);
    assert_eq!(cache.pending.len(), 2);
    apply(&mut cache, 3, &[root_report(0, "mamba", 0, 2, &[2], 2)]);
    assert_eq!(query(&cache, &tokens(7)), Some(6));
    assert!(cache.pending.is_empty());
    assert!(cache.parents_by_child.is_empty());
    apply(&mut cache, 4, &[remove(0, &[2, 6])]);
    assert_eq!(cache.index.retained_contents(), 0);
    assert!(cache.positions.is_empty());

    // The same dependency chain can disappear before its anchor arrives.
    apply(&mut cache, 5, &[child_report(0, "mamba", 0, 2, 4)]);
    apply(&mut cache, 6, &[child_report(0, "mamba", 0, 4, 6)]);
    apply(&mut cache, 7, &[remove(0, &[4, 6])]);
    assert!(cache.pending.is_empty());
    assert!(cache.parents_by_child.is_empty());
    assert_eq!(query(&cache, &tokens(7)), Some(0));
}

#[test]
fn one_batch_keeps_ancestry_until_new_reports_are_applied() {
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[root_report(0, "mamba", 0, 2, &[2], 2)]);
    apply(
        &mut cache,
        1,
        &[remove(0, &[2]), child_report(0, "mamba", 0, 2, 4)],
    );
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    assert!(cache.pending.is_empty());
}

#[test]
fn aliases_and_returning_sparse_reports_keep_their_shared_context() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[
            root_report(0, "mamba", 0, 2, &[2], 2),
            store(0, "mamba", 0, 2, vec![opaque(99)], None, tokens(2)),
        ],
    );
    apply(&mut cache, 1, &[remove(0, &[2])]);
    assert_eq!(query(&cache, &tokens(3)), Some(2));
    apply(&mut cache, 2, &[child_report(0, "mamba", 0, 2, 4)]);
    apply(&mut cache, 3, &[remove(0, &[99])]);
    // Endpoint 2 has been released, but its live descendant still holds the path.
    let mut sparse = root_report(0, "mamba", 0, 2, &[2], 2);
    if let GroupEvent::Store { tokens, .. } = &mut sparse {
        tokens.clear();
    }
    apply(&mut cache, 4, &[sparse, remove(0, &[4])]);
    assert_eq!(query(&cache, &tokens(5)), Some(2));
    apply(&mut cache, 5, &[remove(0, &[2])]);
    assert_eq!(cache.index.retained_contents(), 0);
    assert!(cache.positions.is_empty());
    cache.index.audit().unwrap();
}

#[test]
fn repeated_reports_are_idempotent_and_one_remove_withdraws_the_key() {
    let mut cache = GroupCache::default();
    let report = root_report(0, "full_attention", 0, 2, &[2, 4], 4);
    apply(&mut cache, 0, std::slice::from_ref(&report));
    apply(&mut cache, 0, std::slice::from_ref(&report));
    apply(&mut cache, 1, &[report]);
    apply(&mut cache, 2, &[remove(0, &[4])]);
    assert_eq!(query(&cache, &tokens(5)), Some(2));
    apply(&mut cache, 3, &[remove(0, &[4])]);
    assert_eq!(query(&cache, &tokens(5)), Some(2));
    apply(&mut cache, 4, &[child_report(0, "full_attention", 0, 2, 4)]);
    assert_eq!(query(&cache, &tokens(5)), Some(4));
}

#[test]
fn clear_keeps_groups_and_explicit_invalidation_starts_new_observations() {
    let mut cache = GroupCache::default();
    let a = root_report(0, "full_attention", 0, 2, &[2, 4], 4);
    let b = root_report(1, "mamba", 0, 2, &[4], 4);
    apply(&mut cache, 0, &[a.clone(), b.clone()]);
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    apply(&mut cache, 1, &[GroupEvent::Clear, a.clone()]);
    assert_eq!(query(&cache, &tokens(5)), Some(0));
    apply(&mut cache, 2, &[b]);
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    apply(&mut cache, 4, std::slice::from_ref(&a));
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    cache.invalidate();
    assert_eq!(query(&cache, &tokens(5)), None);
    apply(&mut cache, 31, &[a]);
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    apply(
        &mut cache,
        32,
        &[child_report(0, "full_attention", 0, 8, 10)],
    );
    assert!(!cache.pending.is_empty());
    apply(&mut cache, 33, &[GroupEvent::Clear]);
    assert_eq!(query(&cache, &tokens(5)), Some(0));
    assert_eq!(cache.index.retained_contents(), 0);
    assert!(cache.positions.is_empty());
    assert!(cache.by_prefix.is_empty());
    assert!(cache.pending.is_empty());
    assert!(cache.parents_by_child.is_empty());
}

#[test]
fn partial_report_span_is_not_physical_group_size_or_global_alignment() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "full_attention", 0, 8, &[8], 8)],
    );
    apply(&mut cache, 1, &[child_report(1, "mamba", 0, 4, 6)]);
    assert_eq!(query(&cache, &tokens(9)), Some(0));
    apply(
        &mut cache,
        2,
        &[root_report(0, "full_attention", 0, 4, &[4], 4)],
    );
    assert_eq!(query(&cache, &tokens(9)), Some(6));
    apply(&mut cache, 3, &[root_report(1, "mamba", 0, 8, &[8], 8)]);
    assert_eq!(query(&cache, &tokens(9)), Some(8));
}

#[test]
fn raw_key_width_and_leading_bytes_are_identity() {
    let keys = [vec![7], vec![0, 7], vec![7; 8], {
        let mut wide = vec![0; 24];
        wide.extend_from_slice(&[7; 8]);
        wide
    }];
    let mut cache = GroupCache::default();
    for (i, key) in keys.iter().enumerate() {
        let end = (i + 1) * 2;
        apply(
            &mut cache,
            i as u64,
            &[store(
                0,
                "mamba",
                0,
                end,
                vec![key.clone()],
                None,
                tokens(end),
            )],
        );
    }
    assert_eq!(query(&cache, &tokens(9)), Some(8));
    for (sequence, i) in (0..keys.len()).rev().enumerate() {
        apply(
            &mut cache,
            (sequence + 4) as u64,
            &[GroupEvent::Remove {
                copy_counts: Vec::new(),
                group_id: 0,
                keys: vec![keys[i].clone()],
            }],
        );
        assert_eq!(query(&cache, &tokens(9)), Some(i * 2));
    }
}

#[test]
fn removing_one_group_preserves_other_groups_using_the_same_key() {
    let mut first = GroupCache::default();
    apply(
        &mut first,
        0,
        &[
            root_report(0, "full_attention", 0, 2, &[2], 2),
            root_report(1, "mamba", 0, 2, &[2], 2),
        ],
    );
    apply(&mut first, 1, &[remove(0, &[2])]);
    assert_eq!(query(&first, &tokens(3)), Some(0));
    apply(
        &mut first,
        2,
        &[root_report(0, "full_attention", 0, 2, &[2], 2)],
    );
    assert_eq!(query(&first, &tokens(3)), Some(2));
}

#[test]
fn matching_tail_content_does_not_match_a_different_prefix() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "mamba", 0, 2, &[2, 4], 4), remove(0, &[2])],
    );
    assert_eq!(query(&cache, &tokens(5)), Some(4));
    assert_eq!(query(&cache, &[9, 9, 3, 4, 5]), Some(0));
}

#[test]
fn only_reported_endpoints_below_request_length_are_candidates() {
    for kind in ["full_attention", "mla_attention"] {
        let mut cache = GroupCache::default();
        apply(&mut cache, 0, &[root_report(0, kind, 0, 4, &[4, 8], 8)]);
        assert_eq!(query(&cache, &tokens(10)), Some(8));
        assert_eq!(query(&cache, &tokens(8)), Some(4));
        assert_eq!(query(&cache, &tokens(7)), Some(4));
        assert_eq!(query(&cache, &[]), Some(0));
    }
}

#[test]
fn unsupported_or_missing_group_metadata_is_unavailable() {
    for kind in ["", "unknown", "chunked_local_attention"] {
        let mut cache = GroupCache::default();
        apply(&mut cache, 0, &[root_report(0, kind, 0, 2, &[2], 2)]);
        assert_eq!(query(&cache, &tokens(3)), None);
    }
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[remove(7, &[2])]);
    assert_eq!(query(&cache, &tokens(3)), None);
    apply(&mut cache, 1, &[root_report(7, "mamba", 0, 2, &[2], 2)]);
    assert_eq!(query(&cache, &tokens(3)), Some(2));
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "sliding_window", 0, 2, &[2], 2)],
    );
    assert_eq!(query(&cache, &tokens(3)), None);
}

#[test]
fn unmatchable_payload_keeps_the_required_group_unknown() {
    let mut report = root_report(0, "mamba", 0, 2, &[2], 2);
    if let GroupEvent::Store {
        tokens_matchable, ..
    } = &mut report
    {
        *tokens_matchable = false;
    }
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[report]);
    assert_eq!(query(&cache, &tokens(3)), None);
    assert!(cache.groups[&0].reported.contains_key(&opaque(2)));
}

#[test]
fn malformed_or_conflicting_identity_discards_evidence() {
    for invalid in [
        GroupEvent::Invalid,
        store(0, "full_attention", 0, 0, vec![opaque(2)], None, tokens(2)),
        store(0, "full_attention", 0, 2, vec![Vec::new()], None, tokens(2)),
    ] {
        let mut cache = GroupCache::default();
        let observes_group = !matches!(invalid, GroupEvent::Invalid);
        assert!(cache.apply_events(&[invalid]).is_err());
        assert_eq!(query(&cache, &tokens(3)), observes_group.then_some(0));
    }
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "full_attention", 0, 2, &[2], 2)],
    );
    assert!(cache
        .apply_events(&[store(
            0,
            "full_attention",
            0,
            2,
            vec![opaque(2)],
            None,
            vec![9, 9]
        )])
        .is_err());
    assert_eq!(query(&cache, &tokens(3)), Some(0));
    apply(
        &mut cache,
        2,
        &[root_report(0, "full_attention", 0, 2, &[2], 2)],
    );
    assert!(cache
        .apply_events(&[root_report(0, "mamba", 0, 2, &[2], 2)])
        .is_err());
    assert_eq!(query(&cache, &tokens(3)), Some(0));
}

#[test]
fn same_known_endpoint_accepts_full_and_partial_reports_with_unknown_parents() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[
            root_report(0, "full_attention", 0, 16, &[16], 16),
            child_report(1, "sliding_window", 9, 8, 16),
            child_report(2, "mamba", 0, 14, 16),
        ],
    );
    assert_eq!(query(&cache, &tokens(17)), Some(16));
}

#[test]
fn either_unknown_parent_can_resolve_the_same_checkpoint() {
    for first_parent in [8, 14] {
        let mut cache = GroupCache::default();
        apply(
            &mut cache,
            0,
            &[
                child_report(0, "mamba", 0, 8, 16),
                child_report(1, "mamba", 0, 14, 16),
            ],
        );
        assert_eq!(query(&cache, &tokens(17)), Some(0));
        apply(
            &mut cache,
            1,
            &[child_report(0, "mamba", 0, 0, first_parent)],
        );
        assert_eq!(query(&cache, &tokens(17)), Some(16));

        assert!(cache.pending.is_empty());
        assert!(cache.parents_by_child.is_empty());

        let other_parent = if first_parent == 8 { 14 } else { 8 };
        apply(
            &mut cache,
            2,
            &[child_report(0, "mamba", 0, 0, other_parent)],
        );
        assert_eq!(query(&cache, &tokens(17)), Some(16));
    }
}

#[test]
fn reported_span_longer_than_resolved_prefix_cannot_supply_coverage() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[
            root_report(0, "mamba", 0, 2, &[2], 2),
            root_report(1, "full_attention", 0, 8, &[2], 2),
        ],
    );
    assert_eq!(query(&cache, &tokens(3)), Some(0));
}

#[test]
fn later_full_report_can_bridge_an_earlier_gap() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[
            root_report(0, "mamba", 0, 2, &[2, 4, 6, 8], 8),
            child_report(1, "full_attention", 0, 2, 4),
            child_report(1, "full_attention", 0, 0, 8),
        ],
    );
    assert_eq!(query(&cache, &tokens(9)), Some(8));
}

#[test]
fn multiple_million_token_prefixes_keep_affinity() {
    let mut cache = GroupCache::default();
    let length = 1_048_576;
    let block_size = 16;
    let blocks = length / block_size;
    // Two independent prompts on one worker exceed both former policy limits:
    // 1.6M retained tokens and 100K native identities/live memberships.
    for branch in 0..2 {
        apply(
            &mut cache,
            branch as u64,
            &[store(
                0,
                "full_attention",
                0,
                block_size,
                (1..=blocks)
                    .map(|block| opaque(branch * blocks + block))
                    .collect(),
                None,
                vec![branch as u32 + 1; length],
            )],
        );
    }
    for token in [1, 2] {
        assert_eq!(query(&cache, &vec![token; length + 1]), Some(length));
    }
}

fn copies(mut event: GroupEvent, counts: &[u32]) -> GroupEvent {
    match &mut event {
        GroupEvent::Store { copy_counts, .. } | GroupEvent::Remove { copy_counts, .. } => {
            *copy_counts = counts.to_vec();
        }
        _ => panic!("copy counts need a block event"),
    }
    event
}

fn replay(cache: &GroupCache) -> GroupCache {
    let mut replayed = GroupCache::default();
    replayed.apply_events(&cache.snapshot_events()).unwrap();
    replayed
}

#[test]
fn skipped_ancestry_is_complete_root_history_without_residency() {
    let mut report = child_report(0, "mamba", 0, 6, 8);
    if let GroupEvent::Store { skipped_tokens, .. } = &mut report {
        *skipped_tokens = Some(tokens(6));
    }
    let mut cache = GroupCache::default();
    cache.apply_events(&[report]).unwrap();
    assert_eq!(query(&cache, &tokens(9)), Some(8));
    assert_eq!(query(&cache, &[7, 8, 9]), Some(0));
    assert_eq!(cache.groups[&0].reported.len(), 1);
    assert_eq!(query(&replay(&cache), &tokens(9)), Some(8));

    // Learning skipped FA history cannot claim that the skipped blocks are cached.
    cache
        .apply_events(&[child_report(1, "full_attention", 0, 6, 8)])
        .unwrap();
    assert_eq!(query(&cache, &tokens(9)), Some(0));
}

#[test]
fn skipped_ancestry_must_agree_with_a_known_parent() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[root_report(0, "mamba", 0, 2, &[2], 2)])
        .unwrap();
    let mut report = child_report(0, "mamba", 0, 2, 4);
    if let GroupEvent::Store { skipped_tokens, .. } = &mut report {
        *skipped_tokens = Some(vec![9, 9]);
    }
    assert!(cache.apply_events(&[report]).is_err());
    assert!(cache.has_groups());
    assert_eq!(query(&cache, &tokens(5)), Some(0));
}

#[test]
fn authoritative_copy_counts_and_repeated_store_preserve_last_copy() {
    let mut cache = GroupCache::default();
    let report = root_report(0, "mamba", 0, 2, &[2], 2);
    cache
        .apply_events(&[copies(report.clone(), &[2]), report.clone()])
        .unwrap();
    assert_eq!(cache.groups[&0].reported[&opaque(2)].copies, 2);
    cache
        .apply_events(&[copies(remove(0, &[2]), &[1])])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(2));
    cache.apply_events(&[copies(report.clone(), &[3])]).unwrap();
    let mut replayed = replay(&cache);
    assert_eq!(replayed.groups[&0].reported[&opaque(2)].copies, 3);
    replayed.apply_events(&[remove(0, &[2])]).unwrap();
    assert_eq!(replayed.groups[&0].reported[&opaque(2)].copies, 2);
    cache
        .apply_events(&[copies(remove(0, &[2]), &[0])])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(0));
    assert_eq!(cache.index.retained_contents(), 0);

    cache.apply_events(&[copies(report.clone(), &[0])]).unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(0));
    assert!(cache.groups[&0].reported.is_empty());
    // A zero-copy context report must not withdraw an already resident key.
    cache
        .apply_events(&[report.clone(), copies(report, &[0])])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(2));
}

#[test]
fn copy_count_lengths_are_validated_without_forgetting_required_groups() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[root_report(0, "full_attention", 0, 2, &[2], 2)])
        .unwrap();
    for event in [
        copies(root_report(1, "mamba", 0, 2, &[2], 2), &[1, 1]),
        copies(remove(0, &[2]), &[0, 0]),
    ] {
        assert!(cache.apply_events(&[event]).is_err());
        assert!(cache.has_groups());
        assert_eq!(query(&cache, &tokens(3)), Some(0));
        assert_eq!(cache.groups.len(), 2);
    }
}

#[test]
fn snapshot_preserves_empty_and_unknown_groups() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            root_report(0, "full_attention", 0, 2, &[2, 4], 4),
            root_report(1, "mamba", 0, 2, &[4], 4),
            remove(1, &[4]),
        ])
        .unwrap();
    let replayed = replay(&cache);
    assert_eq!(replayed.groups.len(), 2);
    assert!(replayed.groups[&1].reported.is_empty());
    assert_eq!(query(&replayed, &tokens(5)), Some(0));
    cache.apply_events(&[remove(9, &[])]).unwrap();
    assert_eq!(query(&replay(&cache), &tokens(5)), None);
}

#[test]
fn snapshot_preserves_pending_descriptions_and_released_ancestry() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            child_report(0, "mamba", 0, 2, 4),
            child_report(0, "mamba", 0, 4, 6),
            remove(0, &[4]),
        ])
        .unwrap();
    let mut replayed = replay(&cache);
    assert_eq!(query(&replayed, &tokens(7)), Some(0));
    replayed
        .apply_events(&[root_report(0, "mamba", 0, 2, &[2], 2), remove(0, &[2])])
        .unwrap();
    assert_eq!(query(&replayed, &tokens(7)), Some(6));
    let mut replayed = replay(&replayed);
    assert_eq!(query(&replayed, &tokens(7)), Some(6));
    assert!(!replayed.groups[&0].reported.contains_key(&opaque(2)));
    assert!(!replayed.groups[&0].reported.contains_key(&opaque(4)));
    replayed.apply_events(&[remove(0, &[6])]).unwrap();
    assert_eq!(replayed.index.retained_contents(), 0);
    assert!(replayed.positions.is_empty());
    replayed.index.audit().unwrap();
}

#[test]
fn snapshot_dense_chain_serializes_history_once() {
    let mut cache = GroupCache::default();
    let ends: Vec<_> = (2..=2048).step_by(2).collect();
    cache
        .apply_events(&[root_report(0, "full_attention", 0, 2, &ends, 2048)])
        .unwrap();
    let snapshot = cache.snapshot_events();
    let serialized_tokens: usize = snapshot
        .iter()
        .map(|event| match event {
            GroupEvent::Store {
                tokens,
                skipped_tokens,
                ..
            } => tokens.len() + skipped_tokens.as_ref().map_or(0, Vec::len),
            _ => 0,
        })
        .sum();
    assert_eq!(serialized_tokens, 2048);
    assert_eq!(query(&replay(&cache), &tokens(2049)), Some(2048));
}

#[test]
fn unmatchable_group_cannot_recover_from_other_groups_or_omitted_namespace() {
    let mut unmatchable = root_report(1, "mamba", 0, 2, &[2], 2);
    if let GroupEvent::Store {
        tokens_matchable, ..
    } = &mut unmatchable
    {
        *tokens_matchable = false;
    }
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            root_report(0, "full_attention", 0, 2, &[2], 2),
            unmatchable,
            root_report(1, "mamba", 0, 2, &[2], 2),
        ])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), None);
    assert_eq!(query(&replay(&cache), &tokens(3)), None);
    cache.clear_evidence();
    cache
        .apply_events(&[
            root_report(0, "full_attention", 0, 2, &[2], 2),
            root_report(1, "mamba", 0, 2, &[2], 2),
        ])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), None);
    cache.invalidate();
    cache
        .apply_events(&[root_report(0, "full_attention", 0, 2, &[2], 2)])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(2));
}

#[test]
fn invalid_and_discontinuity_keep_all_observed_required_groups() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            root_report(0, "full_attention", 0, 2, &[2], 2),
            root_report(1, "mamba", 0, 2, &[2], 2),
        ])
        .unwrap();
    assert!(cache.apply_events(&[GroupEvent::Invalid]).is_err());
    cache
        .apply_events(&[root_report(0, "full_attention", 0, 2, &[2], 2)])
        .unwrap();
    assert_eq!(query(&cache, &tokens(3)), Some(0));
    cache.clear_evidence();
    assert!(cache.has_groups());
    assert!(cache.positions.is_empty());
}

#[test]
fn sink_full_attention_and_same_kind_groups_use_joint_coverage() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            root_report(0, "sink_full_attention", 0, 2, &[2, 4], 4),
            root_report(1, "full_attention", 0, 2, &[2, 4], 4),
            remove(1, &[4]),
        ])
        .unwrap();
    assert_eq!(query(&cache, &tokens(5)), Some(2));
    assert_eq!(query(&replay(&cache), &tokens(5)), Some(2));
}

#[test]
fn known_hash_still_validates_tail_tokens_when_its_report_parent_is_unknown() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[root_report(0, "full_attention", 0, 4, &[4], 4)])
        .unwrap();
    let mut report = child_report(1, "mamba", 0, 2, 4);
    if let GroupEvent::Store { tokens, .. } = &mut report {
        *tokens = vec![9, 9];
    }
    assert!(cache.apply_events(&[report]).is_err());
    assert_eq!(query(&cache, &tokens(5)), Some(0));
}

#[test]
fn split_snapshot_keeps_zero_copy_ancestry_until_membership_arrives() {
    let mut source = GroupCache::default();
    let ends: Vec<_> = (2..=32).step_by(2).collect();
    source
        .apply_events(&[
            root_report(0, "full_attention", 0, 2, &ends, 32),
            root_report(1, "mamba", 0, 2, &[24, 32], 32),
            remove(0, &[26, 28, 30, 32]),
        ])
        .unwrap();
    let snapshot = source.snapshot_events();
    let resident_start = snapshot
        .iter()
        .position(|event| {
            matches!(event,
                GroupEvent::Store { copy_counts, .. } if copy_counts.iter().any(|&count| count > 0)
            )
        })
        .unwrap();
    let mut restored = GroupCache::default();
    restored.begin_snapshot();
    restored.apply_events(&[GroupEvent::Clear]).unwrap();
    // Deliberately split every zero-copy description into a separate chunk.
    for event in &snapshot[..resident_start] {
        restored.apply_events(std::slice::from_ref(event)).unwrap();
    }
    assert!(restored.index.retained_contents() > 0);
    for event in &snapshot[resident_start..] {
        restored.apply_events(std::slice::from_ref(event)).unwrap();
    }
    restored.finish_snapshot();
    assert_eq!(query(&restored, &tokens(33)), query(&source, &tokens(33)));
    assert_eq!(query(&restored, &tokens(33)), Some(24));
    restored.index.audit().unwrap();
}

#[test]
fn abandoned_or_malformed_snapshot_drops_scratch_and_can_restart() {
    let mut source = GroupCache::default();
    source
        .apply_events(&[root_report(0, "mamba", 0, 2, &[2, 4], 4)])
        .unwrap();
    let snapshot = source.snapshot_events();
    let description = snapshot.iter().find(|event| matches!(event,
        GroupEvent::Store { keys, copy_counts, .. } if !keys.is_empty() && copy_counts == &[0]
    )).unwrap();
    for malformed in [false, true] {
        let mut restored = GroupCache::default();
        restored.begin_snapshot();
        restored
            .apply_events(std::slice::from_ref(description))
            .unwrap();
        assert!(restored.index.retained_contents() > 0);
        if malformed {
            assert!(restored.apply_events(&[GroupEvent::Invalid]).is_err());
        } else {
            restored.clear_evidence();
        }
        assert!(restored.snapshot_changes.is_none());
        assert_eq!(restored.index.retained_contents(), 0);
        assert!(restored.positions.is_empty());
        restored.begin_snapshot();
        restored.apply_events(&[GroupEvent::Clear]).unwrap();
        for event in &snapshot {
            restored.apply_events(std::slice::from_ref(event)).unwrap();
        }
        restored.finish_snapshot();
        assert_eq!(query(&restored, &tokens(5)), Some(4));
        restored.apply_events(&[GroupEvent::Clear]).unwrap();
        assert_eq!(restored.index.retained_contents(), 0);
    }
}

fn anchored(mut event: GroupEvent, anchor: usize, skipped: Vec<u32>) -> GroupEvent {
    if let GroupEvent::Store {
        skipped_tokens,
        skipped_token_parent_hash,
        ..
    } = &mut event
    {
        *skipped_tokens = Some(skipped);
        *skipped_token_parent_hash = Some(opaque(anchor));
    } else {
        panic!("anchor needs a store");
    }
    event
}

fn split_replay(cache: &GroupCache) -> GroupCache {
    let mut restored = GroupCache::default();
    restored.begin_snapshot();
    for event in cache.snapshot_events() {
        restored.apply_events(&[event]).unwrap();
    }
    restored.finish_snapshot();
    restored
}

#[test]
fn spaced_checkpoints_append_delta_from_registered_ancestor_only() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[root_report(0, "mamba", 0, 4, &[4], 4)])
        .unwrap();
    let mut token_payload = 4;
    for end in (8..=256).step_by(4) {
        let report = anchored(
            child_report(0, "mamba", 0, end - 2, end),
            end - 4,
            ((end - 3) as u32..=(end - 2) as u32).collect(),
        );
        if let GroupEvent::Store {
            parent,
            tokens,
            skipped_tokens,
            ..
        } = &report
        {
            assert_eq!(parent.as_ref(), Some(&opaque(end - 2)));
            token_payload += tokens.len() + skipped_tokens.as_ref().unwrap().len();
        }
        cache.apply_events(&[report]).unwrap();
        cache.apply_events(&[remove(0, &[end - 4])]).unwrap();
        assert_eq!(query(&cache, &tokens(end + 1)), Some(end));
    }
    assert_eq!(token_payload, 256);
    assert_eq!(cache.groups[&0].reported[&opaque(256)].span, 2);
    assert_eq!(query(&split_replay(&cache), &tokens(257)), Some(256));
    cache.apply_events(&[remove(0, &[256])]).unwrap();
    assert_eq!(cache.index.retained_contents(), 0);
    cache.index.audit().unwrap();
}

#[test]
fn missing_delta_anchor_waits_for_native_identity_across_split_replay() {
    let report = copies(
        anchored(child_report(0, "mamba", 0, 6, 8), 4, vec![5, 6]),
        &[2],
    );
    let mut source = GroupCache::default();
    source.apply_events(&[report]).unwrap();
    assert_eq!(query(&source, &tokens(9)), None);
    assert_eq!(query(&source, &[5, 6, 7, 8, 9]), None);
    assert_eq!(source.pending.len(), 1);
    let mut restored = split_replay(&source);
    assert_eq!(query(&restored, &tokens(9)), None);
    let anchored_description = restored.snapshot_events().into_iter().find(|event| matches!(event,
        GroupEvent::Store { skipped_token_parent_hash: Some(_), tokens, .. } if !tokens.is_empty()
    )).unwrap();
    if let GroupEvent::Store {
        parent,
        skipped_tokens,
        skipped_token_parent_hash,
        ..
    } = anchored_description
    {
        assert_eq!(parent, Some(opaque(6)));
        assert_eq!(skipped_tokens, Some(vec![5, 6]));
        assert_eq!(skipped_token_parent_hash, Some(opaque(4)));
    }
    restored
        .apply_events(&[copies(root_report(0, "mamba", 0, 4, &[4], 4), &[0])])
        .unwrap();
    assert_eq!(query(&restored, &tokens(9)), Some(8));
    assert!(!restored.groups[&0].reported.contains_key(&opaque(4)));
    assert!(restored.pending.is_empty());
    restored
        .apply_events(&[copies(remove(0, &[8]), &[1])])
        .unwrap();
    assert_eq!(query(&split_replay(&restored), &tokens(9)), Some(8));
    restored
        .apply_events(&[copies(remove(0, &[8]), &[0])])
        .unwrap();
    assert_eq!(restored.index.retained_contents(), 0);
}

#[test]
fn batch_anchor_cleanup_preserves_unresolved_requirements_and_replay() {
    for batched in [false, true] {
        let mut cache = GroupCache::default();
        cache
            .apply_events(&[
                anchored(child_report(0, "mamba", 0, 6, 8), 4, vec![5, 6]),
                anchored(child_report(0, "mamba", 0, 14, 16), 12, vec![13, 14]),
            ])
            .unwrap();
        let resolve_first = [
            copies(root_report(0, "mamba", 0, 4, &[4], 4), &[0]),
            copies(
                store(0, "mamba", 0, 2, vec![opaque(99)], None, vec![31, 32]),
                &[0],
            ),
        ];
        if batched {
            cache.apply_events(&resolve_first).unwrap();
        } else {
            for event in resolve_first {
                cache.apply_events(&[event]).unwrap();
            }
        }
        assert!(!cache.groups[&0]
            .anchor_requirements
            .contains_key(&opaque(8)));
        assert_eq!(
            cache.groups[&0].anchor_requirements[&opaque(16)],
            opaque(12)
        );
        assert_eq!(query(&cache, &tokens(17)), None);
        let mut restored = split_replay(&cache);
        assert_eq!(query(&restored, &tokens(17)), None);
        restored
            .apply_events(&[copies(root_report(0, "mamba", 0, 12, &[12], 12), &[0])])
            .unwrap();
        assert!(restored.groups[&0].anchor_requirements.is_empty());
        assert_eq!(query(&restored, &tokens(17)), Some(16));
        assert_eq!(query(&restored, &tokens(9)), Some(8));
        restored.apply_events(&[remove(0, &[8, 16])]).unwrap();
        assert_eq!(restored.index.retained_contents(), 0);
        restored.index.audit().unwrap();
    }
}

#[test]
fn another_group_known_endpoint_does_not_replace_a_missing_delta_anchor() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[
            root_report(0, "full_attention", 0, 4, &[4, 8], 8),
            anchored(child_report(1, "mamba", 0, 6, 8), 99, vec![5, 6]),
        ])
        .unwrap();
    assert_eq!(query(&cache, &tokens(9)), None);
    assert!(!cache.pending.is_empty());
    let mut restored = split_replay(&cache);
    assert_eq!(query(&restored, &tokens(9)), None);
    restored
        .apply_events(&[copies(
            store(1, "mamba", 0, 4, vec![opaque(99)], None, tokens(4)),
            &[0],
        )])
        .unwrap();
    assert_eq!(query(&restored, &tokens(9)), Some(8));
    assert!(restored.pending.is_empty());
}

#[test]
fn removed_pending_delta_cannot_revive_when_its_anchor_arrives() {
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[anchored(child_report(0, "mamba", 0, 6, 8), 4, vec![5, 6])])
        .unwrap();
    cache.apply_events(&[remove(0, &[8])]).unwrap();
    assert!(cache.pending.is_empty());
    assert!(cache.parents_by_child.is_empty());
    cache
        .apply_events(&[root_report(0, "mamba", 0, 4, &[4], 4)])
        .unwrap();
    assert_eq!(query(&cache, &tokens(9)), Some(4));
    assert!(!cache.groups[&0].reported.contains_key(&opaque(8)));
}

#[test]
fn malformed_anchor_and_known_parent_delta_conflict_clear_evidence() {
    for empty_identity in [false, true] {
        let mut report = child_report(0, "mamba", 0, 6, 8);
        if let GroupEvent::Store {
            skipped_tokens,
            skipped_token_parent_hash,
            ..
        } = &mut report
        {
            *skipped_token_parent_hash = Some(if empty_identity {
                Vec::new()
            } else {
                opaque(4)
            });
            *skipped_tokens = empty_identity.then(|| vec![5, 6]);
        }
        let mut cache = GroupCache::default();
        assert!(cache.apply_events(&[report]).is_err());
        assert_eq!(query(&cache, &tokens(9)), Some(0));
    }
    let mut cache = GroupCache::default();
    cache
        .apply_events(&[root_report(0, "mamba", 0, 2, &[2, 4, 6], 6)])
        .unwrap();
    assert!(cache
        .apply_events(&[anchored(child_report(0, "mamba", 0, 6, 8), 4, vec![9, 9])])
        .is_err());
    assert_eq!(query(&cache, &tokens(9)), Some(0));
}

use std::collections::BTreeSet;

use serde_json::Value;

use super::*;

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

fn apply(cache: &mut GroupCache, sequence: u64, events: &[GroupEvent]) {
    cache.apply_batch(sequence, events).unwrap();
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
                cache.reusable_tokens(&request),
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
fn nine_native_hash_vectors_are_opaque_identities_not_recomputed_hashes() {
    let data = fixtures();
    let vectors = data["hash_vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 9);
    for vector in vectors {
        let block: Vec<u32> = serde_json::from_value(vector["token_ids"].clone()).unwrap();
        let first = key(vector["hash"].as_str().unwrap());
        let second = key(vector["next_hash"].as_str().unwrap());
        let mut request = [block.clone(), block.clone()].concat();
        request.push(42);
        let mut cache = GroupCache::default();
        apply(
            &mut cache,
            0,
            &[
                store(
                    0,
                    "full_attention",
                    0,
                    block.len(),
                    vec![first.clone()],
                    None,
                    block.clone(),
                ),
                store(
                    0,
                    "full_attention",
                    0,
                    block.len(),
                    vec![second],
                    Some(first.clone()),
                    block.clone(),
                ),
            ],
        );
        assert_eq!(cache.reusable_tokens(&request), Some(2 * block.len()));
        apply(
            &mut cache,
            1,
            &[GroupEvent::Remove {
                group_id: 0,
                keys: vec![first],
            }],
        );
        assert_eq!(cache.reusable_tokens(&request), Some(0));
    }
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
    assert_eq!(cache.reusable_tokens(&tokens(11)), Some(8));
    apply(&mut cache, 18, &[root_report(27, "mamba", 0, 2, &[6], 6)]);
    assert_eq!(cache.reusable_tokens(&tokens(11)), Some(6));
    apply(&mut cache, 19, &[remove(27, &[6])]);
    assert_eq!(cache.reusable_tokens(&tokens(11)), Some(0));
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
    assert_eq!(cache.reusable_tokens(&tokens(41)), Some(32));
    apply(
        &mut cache,
        1,
        &[root_report(1, "mamba", 0, 4, &[24, 40], 40)],
    );
    assert_eq!(cache.reusable_tokens(&tokens(41)), Some(24));
    assert_ne!(cache.reusable_tokens(&tokens(41)), Some(32));
}

#[test]
fn sparse_hashes_are_not_ordinal_zipped_and_cross_group_context_resolves_them() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(1, "mamba", 0, 2, &[4, 8, 10, 12], 12)],
    );
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(0));
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
    assert_eq!(cache.reusable_tokens(&tokens(7)), Some(4));
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(8));
    assert_eq!(cache.reusable_tokens(&tokens(13)), Some(12));
}

#[test]
fn unknown_parent_is_pending_and_eviction_keeps_prefix_context() {
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[child_report(0, "mamba", 0, 2, 4)]);
    assert_eq!(cache.reusable_tokens(&[3, 4, 5]), Some(0));
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(0));
    apply(&mut cache, 1, &[root_report(0, "mamba", 0, 2, &[2], 2)]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    apply(&mut cache, 2, &[remove(0, &[2, 4])]);
    assert_eq!(cache.reusable_tokens(&tokens(7)), Some(0));
    apply(&mut cache, 3, &[child_report(0, "mamba", 0, 4, 6)]);
    assert_eq!(cache.reusable_tokens(&tokens(7)), Some(6));
}

#[test]
fn repeated_reports_are_idempotent_and_one_remove_withdraws_the_key() {
    let mut cache = GroupCache::default();
    let report = root_report(0, "full_attention", 0, 2, &[2, 4], 4);
    apply(&mut cache, 0, std::slice::from_ref(&report));
    apply(&mut cache, 0, &[GroupEvent::Invalid]);
    apply(&mut cache, 1, &[report]);
    apply(&mut cache, 2, &[remove(0, &[4])]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(2));
    apply(&mut cache, 3, &[remove(0, &[4])]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(2));
    apply(&mut cache, 4, &[child_report(0, "full_attention", 0, 2, 4)]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
}

#[test]
fn clear_keeps_groups_but_gap_and_disconnect_start_new_observations() {
    let mut cache = GroupCache::default();
    let a = root_report(0, "full_attention", 0, 2, &[2, 4], 4);
    let b = root_report(1, "mamba", 0, 2, &[4], 4);
    apply(&mut cache, 0, &[a.clone(), b.clone()]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    apply(&mut cache, 1, &[GroupEvent::Clear, a.clone()]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(0));
    apply(&mut cache, 2, &[b]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    apply(&mut cache, 4, std::slice::from_ref(&a));
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    apply(&mut cache, 3, &[GroupEvent::Invalid]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    cache.invalidate();
    assert_eq!(cache.reusable_tokens(&tokens(5)), None);
    apply(&mut cache, 31, &[a]);
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
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
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(0));
    apply(
        &mut cache,
        2,
        &[root_report(0, "full_attention", 0, 4, &[4], 4)],
    );
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(6));
    apply(&mut cache, 3, &[root_report(1, "mamba", 0, 8, &[8], 8)]);
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(8));
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
    assert_eq!(cache.reusable_tokens(&tokens(9)), Some(8));
    for (sequence, i) in (0..keys.len()).rev().enumerate() {
        apply(
            &mut cache,
            (sequence + 4) as u64,
            &[GroupEvent::Remove {
                group_id: 0,
                keys: vec![keys[i].clone()],
            }],
        );
        assert_eq!(cache.reusable_tokens(&tokens(9)), Some(i * 2));
    }
}

#[test]
fn group_and_worker_identity_are_independent() {
    let mut first = GroupCache::default();
    let second = GroupCache::default();
    apply(
        &mut first,
        0,
        &[
            root_report(0, "full_attention", 0, 2, &[2], 2),
            root_report(1, "mamba", 0, 2, &[2], 2),
        ],
    );
    apply(&mut first, 1, &[remove(0, &[2])]);
    assert_eq!(first.reusable_tokens(&tokens(3)), Some(0));
    assert_eq!(second.reusable_tokens(&tokens(3)), None);
    assert!(first.groups[&1].reported.contains_key(&opaque(2)));
}

#[test]
fn matching_tail_content_does_not_match_a_different_prefix() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "mamba", 0, 2, &[2, 4], 4), remove(0, &[2])],
    );
    assert_eq!(cache.reusable_tokens(&tokens(5)), Some(4));
    assert_eq!(cache.reusable_tokens(&[9, 9, 3, 4, 5]), Some(0));
}

#[test]
fn only_reported_endpoints_below_request_length_are_candidates() {
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "full_attention", 0, 4, &[4, 8], 8)],
    );
    assert_eq!(cache.reusable_tokens(&tokens(10)), Some(8));
    assert_eq!(cache.reusable_tokens(&tokens(8)), Some(4));
    assert_eq!(cache.reusable_tokens(&tokens(7)), Some(4));
    assert_eq!(cache.reusable_tokens(&[]), Some(0));
}

#[test]
fn unsupported_or_missing_group_metadata_is_unavailable() {
    for kind in ["", "unknown", "mla_attention", "chunked_local_attention"] {
        let mut cache = GroupCache::default();
        apply(&mut cache, 0, &[root_report(0, kind, 0, 2, &[2], 2)]);
        assert_eq!(cache.reusable_tokens(&tokens(3)), None);
    }
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[remove(7, &[2])]);
    assert_eq!(cache.reusable_tokens(&tokens(3)), None);
    apply(&mut cache, 1, &[root_report(7, "mamba", 0, 2, &[2], 2)]);
    assert_eq!(cache.reusable_tokens(&tokens(3)), Some(2));
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "sliding_window", 0, 2, &[2], 2)],
    );
    assert_eq!(cache.reusable_tokens(&tokens(3)), None);
}

#[test]
fn unmatchable_payload_does_not_invent_prefix_context() {
    let mut report = root_report(0, "mamba", 0, 2, &[2], 2);
    if let GroupEvent::Store {
        tokens_matchable, ..
    } = &mut report
    {
        *tokens_matchable = false;
    }
    let mut cache = GroupCache::default();
    apply(&mut cache, 0, &[report]);
    assert_eq!(cache.reusable_tokens(&tokens(3)), Some(0));
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
        assert!(cache.apply_batch(0, &[invalid]).is_err());
        assert_eq!(cache.reusable_tokens(&tokens(3)), None);
    }
    let mut cache = GroupCache::default();
    apply(
        &mut cache,
        0,
        &[root_report(0, "full_attention", 0, 2, &[2], 2)],
    );
    assert!(cache
        .apply_batch(
            1,
            &[store(
                0,
                "full_attention",
                0,
                2,
                vec![opaque(2)],
                None,
                vec![9, 9]
            )]
        )
        .is_err());
    assert_eq!(cache.reusable_tokens(&tokens(3)), None);
    apply(
        &mut cache,
        2,
        &[root_report(0, "full_attention", 0, 2, &[2], 2)],
    );
    assert!(cache
        .apply_batch(3, &[root_report(0, "mamba", 0, 2, &[2], 2)])
        .is_err());
    assert_eq!(cache.reusable_tokens(&tokens(3)), None);
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
    assert_eq!(cache.reusable_tokens(&tokens(17)), Some(16));
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
        assert_eq!(cache.reusable_tokens(&tokens(17)), Some(0));
        apply(
            &mut cache,
            1,
            &[child_report(0, "mamba", 0, 0, first_parent)],
        );
        assert_eq!(cache.reusable_tokens(&tokens(17)), Some(16));

        let other_parent = if first_parent == 8 { 14 } else { 8 };
        apply(
            &mut cache,
            2,
            &[child_report(0, "mamba", 0, 0, other_parent)],
        );
        assert_eq!(cache.reusable_tokens(&tokens(17)), Some(16));
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
    assert_eq!(cache.reusable_tokens(&tokens(3)), Some(0));
}

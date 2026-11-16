use openai_protocol::worker::WorkerStatus;

use super::*;
use crate::{
    policies::{hash_ring::HashRing, test_support::make_workers},
    worker::WorkerLoadGuard,
};

#[test]
fn test_prefix_hash_consistent_routing() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    // Same tokens should always route to same worker
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };

    let (first_result, _, _) = policy.select_worker_impl(&workers, &info);
    let first_idx = first_result.unwrap();

    // Verify consistency
    for _ in 0..10 {
        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(first_idx));
    }
}

/// The ring lookup routes on `is_healthy_and_eligible()`, so the absolute
/// overload veto must move a key off its ring owner and back on recovery.
#[test]
fn overloaded_worker_is_skipped_by_the_ring_lookup() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let tokens: Vec<u32> = vec![11, 22, 33, 44, 55, 66, 77, 88];
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let owner = policy.select_worker_impl(&workers, &info).0.unwrap();

    workers[owner].set_overloaded(true);
    let respilled = policy.select_worker_impl(&workers, &info).0.unwrap();
    assert_ne!(respilled, owner, "a vetoed ring owner must not be selected");

    workers[owner].set_overloaded(false);
    assert_eq!(
        policy.select_worker_impl(&workers, &info).0,
        Some(owner),
        "recovery re-admits the ring owner"
    );
}

#[test]
fn test_different_prefixes_distribute() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let mut distribution = std::collections::HashMap::new();

    // Different token sequences should distribute across workers
    for i in 0..100 {
        let tokens: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        *distribution.entry(result.unwrap()).or_insert(0) += 1;
    }

    assert!(
        distribution.len() > 1,
        "Should distribute across workers, got {distribution:?}",
    );
}

#[test]
fn test_shared_prefix_routes_same() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        prefix_token_count: 5, // Only look at first 5 tokens
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    // Two sequences with same first 5 tokens should route to same worker
    let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
    let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 999, 888, 777];

    let info1 = SelectWorkerInfo {
        tokens: Some(&tokens1),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let info2 = SelectWorkerInfo {
        tokens: Some(&tokens2),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };

    let (result1, _, _) = policy.select_worker_impl(&workers, &info1);
    let (result2, _, _) = policy.select_worker_impl(&workers, &info2);

    assert_eq!(result1, result2, "Same prefix should route to same worker");
}

#[test]
fn test_untokenized_request_routes_consistently() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    // Chat and completions requests reach the policy with text only.
    let info = SelectWorkerInfo {
        tokens: None,
        request_text: Some("summarize the quarterly earnings report"),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (first_result, branch, _) = policy.select_worker_impl(&workers, &info);
    assert!(first_result.is_some(), "text alone must be routable");
    assert_eq!(branch, Branch::RingHit);

    for _ in 0..10 {
        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, first_result);
    }
}

#[test]
fn test_shared_text_prefix_routes_same() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        prefix_token_count: 2, // 8 characters of text
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let info1 = SelectWorkerInfo {
        request_text: Some("system: you are a helpful assistant"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let info2 = SelectWorkerInfo {
        request_text: Some("system: you are a terse assistant"),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result1, _, _) = policy.select_worker_impl(&workers, &info1);
    let (result2, _, _) = policy.select_worker_impl(&workers, &info2);

    assert_eq!(
        result1, result2,
        "text past the prefix budget must not change the worker"
    );
}

#[test]
fn test_text_prefix_budget_respects_char_boundaries() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        prefix_token_count: 1, // 4 characters, mid-way through the emoji run
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let info = SelectWorkerInfo {
        request_text: Some("🌊🌊🌊🌊🌊🌊 tide report"),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result, branch, _) = policy.select_worker_impl(&workers, &info);
    assert!(result.is_some());
    assert_eq!(branch, Branch::RingHit);
}

#[test]
fn test_tokens_take_priority_over_text() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let tokens: Vec<u32> = vec![7, 8, 9];
    let tokens_only = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let tokens_and_text = SelectWorkerInfo {
        tokens: Some(&tokens),
        request_text: Some("text that must not influence the ring key"),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result1, _, _) = policy.select_worker_impl(&workers, &tokens_only);
    let (result2, _, _) = policy.select_worker_impl(&workers, &tokens_and_text);

    assert_eq!(result1, result2);
}

#[test]
fn test_routing_key_hint_overrides_token_and_text_keying() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let key_only = SelectWorkerInfo {
        routing_key: Some("session-42"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let (key_only_result, branch, _) = policy.select_worker_impl(&workers, &key_only);
    assert!(key_only_result.is_some(), "a key alone must be routable");
    assert_eq!(branch, Branch::RingHit);

    // Same key with entirely different tokens and text keeps the worker.
    for tokens in [vec![1u32, 2, 3], vec![900u32, 901, 902, 903]] {
        let info = SelectWorkerInfo {
            routing_key: Some("session-42"),
            tokens: Some(&tokens),
            request_text: Some("unrelated prompt text"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, key_only_result, "the routing key must win");
    }
}

#[test]
fn test_routing_key_hint_consistent_and_distributes() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let info = SelectWorkerInfo {
        routing_key: Some("sticky-session"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let (first, _, _) = policy.select_worker_impl(&workers, &info);
    for _ in 0..10 {
        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, first);
    }

    let mut distribution = std::collections::HashMap::new();
    for i in 0..100 {
        let key = format!("session-{i}");
        let info = SelectWorkerInfo {
            routing_key: Some(&key),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, _, _) = policy.select_worker_impl(&workers, &info);
        *distribution.entry(result.unwrap()).or_insert(0) += 1;
    }
    assert!(
        distribution.len() > 1,
        "distinct keys must not pile onto one worker, got {distribution:?}"
    );
}

#[test]
fn test_no_routing_key_falls_back_to_load() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));
    let _busy = WorkerLoadGuard::new(workers[0].clone(), None);

    // Empty tokens, empty text
    let tokens: Vec<u32> = vec![];
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        request_text: Some(""),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };

    let (result, branch, _) = policy.select_worker_impl(&workers, &info);
    assert_eq!(result, Some(1), "a keyless request still has to be served");
    assert_eq!(branch, Branch::NoRoutingKey);

    // Neither field set
    let info_empty = SelectWorkerInfo {
        tokens: None,
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result2, branch2, _) = policy.select_worker_impl(&workers, &info_empty);
    assert_eq!(result2, Some(1));
    assert_eq!(branch2, Branch::NoRoutingKey);
}

#[test]
fn test_no_healthy_workers() {
    let policy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000"]);
    workers[0].set_status(WorkerStatus::NotReady);

    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));
    let tokens: Vec<u32> = vec![1, 2, 3];
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result, branch, _) = policy.select_worker_impl(&workers, &info);
    assert_eq!(result, None);
    assert_eq!(branch, Branch::NoHealthyWorkers);
}

#[test]
fn test_load_ok_calculation() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        load_factor: 1.25,
        balance_abs_threshold: 0,
        ..Default::default()
    });

    // Total load 100, 4 workers -> avg 25.25, threshold 31.5625
    assert!(policy.load_ok(30, 100, 4));
    assert!(!policy.load_ok(35, 100, 4));

    // Edge cases
    assert!(policy.load_ok(0, 0, 4)); // No load = OK
    assert!(policy.load_ok(100, 0, 0)); // No workers = OK (shouldn't happen)
}

#[test]
fn test_absolute_margin_absorbs_small_count_noise() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        load_factor: 1.25,
        balance_abs_threshold: 10,
        ..Default::default()
    });

    // Average 10.25 across 4 workers: the relative margin alone flags 13,
    // but 13 is within 10 requests of average so it stays acceptable.
    assert!(policy.load_ok(13, 40, 4));
    assert!(!policy.load_ok(21, 40, 4));
}

#[test]
fn test_relative_margin_still_binds_at_high_load() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        load_factor: 1.25,
        balance_abs_threshold: 10,
        ..Default::default()
    });

    // Average 200.25: the relative margin (250.3) now exceeds the absolute
    // one (210.25), so it is the binding constraint.
    assert!(policy.load_ok(240, 800, 4));
    assert!(!policy.load_ok(260, 800, 4));
}

#[test]
fn test_absolute_margin_defaults_on() {
    assert_eq!(PrefixHashConfig::default().balance_abs_threshold, 10);
}

#[test]
fn test_policy_name() {
    let policy = PrefixHashPolicy::with_defaults();
    assert_eq!(policy.name(), "prefix_hash");
}

#[test]
fn test_boundary_hashing_selects_deepest_applicable() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![16, 64],
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    // 70 and 100 tokens sharing the first 64: both key at boundary 64,
    // so the divergent tails cannot split them.
    let tokens1: Vec<u32> = (0..70).collect();
    let tokens2: Vec<u32> = (0..64).chain(9000..9036).collect();
    let info1 = SelectWorkerInfo {
        tokens: Some(&tokens1),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let info2 = SelectWorkerInfo {
        tokens: Some(&tokens2),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };

    let (result1, branch1, level1) = policy.select_worker_impl(&workers, &info1);
    let (result2, _, level2) = policy.select_worker_impl(&workers, &info2);
    assert_eq!(result1, result2, "shared boundary head must co-hash");
    assert_eq!(branch1, Branch::RingHit);
    assert_eq!((level1, level2), (64, 64));

    // 63 tokens fall short of the deeper boundary and key at 16.
    let tokens3: Vec<u32> = (0..63).collect();
    let info3 = SelectWorkerInfo {
        tokens: Some(&tokens3),
        hash_ring: Some(ring),
        ..Default::default()
    };
    let (result3, _, level3) = policy.select_worker_impl(&workers, &info3);
    assert!(result3.is_some());
    assert_eq!(level3, 16);
}

#[test]
fn test_below_smallest_boundary_keeps_fixed_depth_keying() {
    let boundaries = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![16, 64],
        ..Default::default()
    });
    let legacy = PrefixHashPolicy::with_defaults();
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let tokens: Vec<u32> = (0..15).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result, branch, level) = boundaries.select_worker_impl(&workers, &info);
    let (legacy_result, _, _) = legacy.select_worker_impl(&workers, &info);
    assert_eq!(result, legacy_result);
    assert_eq!(branch, Branch::RingHit);
    assert_eq!(level, 0);
}

#[test]
fn test_short_turn_and_long_followup_co_hash_at_shared_boundary() {
    // A short first turn and its much longer follow-up share only the
    // head; no single fixed depth can co-hash both, the boundary does.
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![2048, 32768],
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let head: Vec<u32> = (0..2048).map(|i| i * 7 % 50000).collect();
    let turn1: Vec<u32> = head.iter().copied().chain(200_000..201_000).collect();
    let turn2: Vec<u32> = head.iter().copied().chain(500_000..514_952).collect();
    assert_eq!(turn1.len(), 3048);
    assert_eq!(turn2.len(), 17_000);

    let info1 = SelectWorkerInfo {
        tokens: Some(&turn1),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let info2 = SelectWorkerInfo {
        tokens: Some(&turn2),
        hash_ring: Some(ring),
        ..Default::default()
    };

    let (result1, branch1, level1) = policy.select_worker_impl(&workers, &info1);
    let (result2, branch2, level2) = policy.select_worker_impl(&workers, &info2);
    assert_eq!(result1, result2, "turns sharing the head must co-hash");
    assert_eq!((branch1, branch2), (Branch::RingHit, Branch::RingHit));
    assert_eq!((level1, level2), (2048, 2048));
}

#[test]
fn test_overloaded_deep_target_retries_shallower_boundary() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        load_factor: 1.0,
        balance_abs_threshold: 0,
        cache_boundaries: vec![16, 32],
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    // Find a stream whose level-32 and level-16 ring targets differ.
    let mut tokens: Vec<u32> = Vec::new();
    let mut deep_idx = usize::MAX;
    let mut shallow_idx = usize::MAX;
    for seed in 0..1000u32 {
        tokens = (seed..seed + 32).collect();
        let full = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let head = SelectWorkerInfo {
            tokens: Some(&tokens[..16]),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        deep_idx = policy.select_worker_impl(&workers, &full).0.unwrap();
        shallow_idx = policy.select_worker_impl(&workers, &head).0.unwrap();
        if deep_idx != shallow_idx {
            break;
        }
    }
    assert_ne!(deep_idx, shallow_idx, "no split stream found");

    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring),
        ..Default::default()
    };

    // Overloaded level-32 target: retry lands on the level-16 target.
    let _deep_busy = [
        WorkerLoadGuard::new(workers[deep_idx].clone(), None),
        WorkerLoadGuard::new(workers[deep_idx].clone(), None),
    ];
    let (result, branch, level) = policy.select_worker_impl(&workers, &info);
    assert_eq!(result, Some(shallow_idx));
    assert_eq!(branch, Branch::RingHit);
    assert_eq!(level, 16);

    // Both targets overloaded: the load-balance walk takes over.
    let _shallow_busy = [
        WorkerLoadGuard::new(workers[shallow_idx].clone(), None),
        WorkerLoadGuard::new(workers[shallow_idx].clone(), None),
    ];
    let (result, branch, level) = policy.select_worker_impl(&workers, &info);
    let spare = (0..3)
        .find(|i| *i != deep_idx && *i != shallow_idx)
        .unwrap();
    assert_eq!(result, Some(spare));
    assert_eq!(branch, Branch::LoadBalanceWalk);
    assert_eq!(level, 0);
}

#[test]
fn test_text_boundary_co_hashing() {
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![4], // 16 characters of text
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let info1 = SelectWorkerInfo {
        request_text: Some("shared-16-chars! then a question about tides"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let info2 = SelectWorkerInfo {
        request_text: Some("shared-16-chars! and an unrelated follow-up about currents"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let (result1, _, level1) = policy.select_worker_impl(&workers, &info1);
    let (result2, _, level2) = policy.select_worker_impl(&workers, &info2);
    assert_eq!(result1, result2, "shared scaled head must co-hash");
    assert_eq!((level1, level2), (4, 4));

    // Shorter than the scaled boundary: fixed-depth keying.
    let short = SelectWorkerInfo {
        request_text: Some("hi there"),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let (short_result, short_branch, short_level) = policy.select_worker_impl(&workers, &short);
    assert!(short_result.is_some());
    assert_eq!(short_branch, Branch::RingHit);
    assert_eq!(short_level, 0);

    // Scaled boundary landing mid-way through a multibyte run stays on
    // a character boundary.
    let emoji = SelectWorkerInfo {
        request_text: Some("🌊🌊🌊🌊🌊🌊 tide report"),
        hash_ring: Some(ring),
        ..Default::default()
    };
    let (emoji_result, emoji_branch, emoji_level) = policy.select_worker_impl(&workers, &emoji);
    assert!(emoji_result.is_some());
    assert_eq!(emoji_branch, Branch::RingHit);
    assert_eq!(emoji_level, 4);
}

#[test]
fn test_unset_boundaries_key_at_fixed_depth() {
    // Empty boundaries: the ring key is exactly the fixed-depth hash.
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        prefix_token_count: 5,
        ..Default::default()
    });
    let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

    let tokens: Vec<u32> = (0..40).collect();
    let info = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: Some(ring.clone()),
        ..Default::default()
    };
    let (result, branch, level) = policy.select_worker_impl(&workers, &info);
    assert_eq!(branch, Branch::RingHit);
    assert_eq!(level, 0);

    let hash = xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&tokens[..5]));
    let key = format!("{hash:016x}");
    let expected_url = ring.find_healthy_url(&key, |_| true).unwrap();
    assert_eq!(workers[result.unwrap()].url(), expected_url);
}

/// Sixteen workers on a ring with boundaries at 64 and 256 tokens and
/// thirty-two prompts of 70 to 400 tokens: every prompt lands where it
/// did when a per-request URL map translated the ring's answer and the
/// ring key was a formatted string, with every worker eligible and with
/// three of them not ready. The placement is the ring's; neither the
/// translation nor the key's spelling may move it.
#[test]
fn the_recorded_prompt_set_lands_where_the_ring_placed_it() {
    const RECORDED_ALL_ELIGIBLE: [usize; 32] = [
        15, 6, 10, 15, 14, 8, 3, 4, 4, 9, 9, 7, 11, 7, 7, 10, 5, 11, 10, 6, 0, 6, 1, 6, 3, 1, 15,
        11, 3, 11, 7, 3,
    ];
    const RECORDED_THREE_NOT_READY: [usize; 32] = [
        15, 6, 10, 15, 14, 8, 8, 4, 4, 9, 9, 9, 12, 1, 1, 10, 5, 5, 10, 6, 0, 6, 1, 6, 15, 1, 15,
        15, 9, 4, 0, 9,
    ];
    let policy = PrefixHashPolicy::new(PrefixHashConfig {
        cache_boundaries: vec![64, 256],
        ..Default::default()
    });
    let urls: Vec<String> = (0..16).map(|i| format!("http://w{i:02}:8000")).collect();
    let workers = make_workers(&urls.iter().map(String::as_str).collect::<Vec<_>>());
    let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));
    let prompts: Vec<Vec<u32>> = (0..32u32)
        .map(|p| {
            let len = 70 + (p * 37) % 331;
            (0..len)
                .map(|i| (p / 4) * 1_000 + i * (1 + p % 3))
                .collect()
        })
        .collect();
    let placements = |workers: &[Arc<dyn Worker>]| -> Vec<usize> {
        prompts
            .iter()
            .map(|tokens| {
                let info = SelectWorkerInfo {
                    tokens: Some(tokens),
                    hash_ring: Some(ring.clone()),
                    ..Default::default()
                };
                let (result, branch, _) = policy.select_worker_impl(workers, &info);
                assert_eq!(branch, Branch::RingHit, "prompt of {} tokens", tokens.len());
                result.expect("an eligible owner exists")
            })
            .collect()
    };
    assert_eq!(
        placements(&workers),
        RECORDED_ALL_ELIGIBLE,
        "with every worker eligible the prompts placed {:?}",
        placements(&workers)
    );
    for not_ready in [3, 7, 11] {
        workers[not_ready].set_status(WorkerStatus::NotReady);
    }
    let moved = placements(&workers);
    assert!(
        !moved.contains(&3) && !moved.contains(&7) && !moved.contains(&11),
        "a not-ready worker owns a prompt: {moved:?}"
    );
    assert_eq!(
        moved, RECORDED_THREE_NOT_READY,
        "with three workers not ready the prompts placed {moved:?}"
    );
}

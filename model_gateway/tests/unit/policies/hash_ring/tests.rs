use super::*;

/// The engine's walk from where a key stands asks the workers a lookup
/// of that key asks, in the lookup's order, and ends at the worker
/// taken. An empty ring and an unfilled one answer a position and ask
/// nobody.
#[test]
fn the_lent_walk_from_a_keys_position_is_the_lookups_own() {
    let urls: Vec<String> = (0..40).map(|i| format!("http://w{i}:8000")).collect();
    let ring = HashRing::new(&urls);
    for key in ["routing-key", "conversation-7", ""] {
        let lookup = std::cell::RefCell::new(Vec::new());
        let found = ring.find_healthy_url(key, |url| {
            lookup.borrow_mut().push(url.to_owned());
            false
        });
        assert_eq!(found, None);
        let mut walk = Vec::new();
        LentRing::walk(&ring, LentRing::position(&ring, key), &mut |url| {
            walk.push(url.to_owned());
            false
        });
        assert_eq!(walk, lookup.into_inner(), "{key}");
        assert_eq!(walk.len(), 40);
    }
    let mut asked = 0;
    LentRing::walk(&ring, 0, &mut |_| {
        asked += 1;
        asked == 3
    });
    assert_eq!(asked, 3);
    let empty = HashRing::new(std::iter::empty::<&str>());
    LentRing::walk(&empty, 7, &mut |_| panic!("nobody to ask"));
    assert_eq!(
        LentRing::position(&Unfilled, "k"),
        HashRing::hash_position("k")
    );
    LentRing::walk(&Unfilled, 7, &mut |_| panic!("nobody to ask"));
}

/// The engine's walk by ids names the workers of its walk by URLs, each
/// by its URL's id and in that order: on a ring built whole, and on one
/// reconciled from it with URLs gone, kept and added.
#[test]
fn the_lent_walk_by_ids_names_the_walks_workers_by_their_urls_ids() {
    let urls: Vec<String> = (0..40).map(|i| format!("http://w{i}:8000")).collect();
    let built = HashRing::new(&urls);
    let kept = urls[3..].iter().map(String::as_str);
    let reconciled = built.updated(kept.chain(["http://new:8000"]));
    for ring in [&built, &reconciled] {
        for key in ["routing-key", "conversation-7", ""] {
            let position = LentRing::position(ring, key);
            let mut by_url = Vec::new();
            LentRing::walk(ring, position, &mut |url| {
                by_url.push(WorkerId::from_address(url));
                false
            });
            let mut by_id = Vec::new();
            LentRing::walk_ids(ring, position, &mut |id| {
                by_id.push(id);
                false
            });
            assert_eq!(by_id, by_url, "{key}");
            assert_eq!(by_id.len(), ring.worker_count());
        }
    }
}

fn assert_same_selection(a: &HashRing, b: &HashRing, keys: usize) {
    for i in 0..keys {
        let key = format!("key-{i}");
        assert_eq!(
            a.find_healthy_url(&key, |_| true),
            b.find_healthy_url(&key, |_| true),
            "selection diverged for {key}"
        );
    }
}

#[test]
fn empty_ring_returns_none() {
    let ring = HashRing::new(std::iter::empty::<&str>());
    assert!(ring.is_empty());
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.worker_count(), 0);
    assert_eq!(ring.find_healthy_url("any-key", |_| true), None);
}

#[test]
fn len_scales_with_virtual_nodes() {
    let ring = HashRing::new(["http://a", "http://b", "http://c"]);
    assert!(!ring.is_empty());
    assert_eq!(ring.len(), 3 * VIRTUAL_NODES_PER_WORKER);
    assert_eq!(ring.worker_count(), 3);
}

#[test]
fn find_healthy_url_is_deterministic() {
    let ring = HashRing::new(["http://a", "http://b", "http://c"]);
    let first = ring.find_healthy_url("routing-key", |_| true).unwrap();
    for _ in 0..10 {
        assert_eq!(ring.find_healthy_url("routing-key", |_| true), Some(first));
    }
}

#[test]
fn find_healthy_url_skips_unhealthy() {
    let ring = HashRing::new(["http://a", "http://b", "http://c"]);
    let picked = ring.find_healthy_url("routing-key", |url| url != "http://a");
    assert!(matches!(picked, Some("http://b") | Some("http://c")));
}

#[test]
fn find_healthy_url_returns_none_when_all_unhealthy() {
    let ring = HashRing::new(["http://a", "http://b"]);
    assert_eq!(ring.find_healthy_url("k", |_| false), None);
}

/// Every worker is asked once while a lookup walks the whole ring, on a
/// fleet the stack marks cover and on one past them.
#[test]
fn find_healthy_url_asks_each_worker_once_on_small_and_large_fleets() {
    for workers in [
        3,
        64 * INLINE_VISITED_WORDS,
        64 * INLINE_VISITED_WORDS + 1,
        700,
    ] {
        let urls: Vec<String> = (0..workers).map(|i| format!("http://w{i}:8000")).collect();
        let ring = HashRing::new(&urls);
        let asked = std::cell::RefCell::new(HashMap::<String, usize>::new());
        assert_eq!(
            ring.find_healthy_url("routing-key", |url| {
                *asked.borrow_mut().entry(url.to_string()).or_insert(0) += 1;
                false
            }),
            None
        );
        let asked = asked.into_inner();
        assert_eq!(
            asked.len(),
            workers,
            "{workers} workers: every worker asked"
        );
        assert!(
            asked.values().all(|&times| times == 1),
            "{workers} workers: each asked once, got {asked:?}"
        );
    }
}

#[test]
fn accepts_owned_string_iterators() {
    let urls = vec!["http://a".to_string(), "http://b".to_string()];
    let ring = HashRing::new(urls);
    assert_eq!(ring.worker_count(), 2);
}

#[test]
fn updated_with_addition_matches_full_rebuild() {
    let base = HashRing::new(["http://a", "http://b"]);
    let incremental = base.updated(["http://a", "http://b", "http://c"]);
    let rebuilt = HashRing::new(["http://a", "http://b", "http://c"]);

    assert_eq!(incremental.worker_count(), 3);
    assert_eq!(incremental.len(), rebuilt.len());
    assert_same_selection(&incremental, &rebuilt, 500);
}

#[test]
fn updated_with_removal_matches_full_rebuild() {
    let base = HashRing::new(["http://a", "http://b", "http://c"]);
    let incremental = base.updated(["http://a", "http://c"]);
    let rebuilt = HashRing::new(["http://a", "http://c"]);

    assert_eq!(incremental.worker_count(), 2);
    assert_eq!(incremental.len(), rebuilt.len());
    assert_same_selection(&incremental, &rebuilt, 500);
}

#[test]
fn updated_with_same_membership_is_a_cheap_clone() {
    let base = HashRing::new(["http://a", "http://b"]);
    let same = base.updated(["http://b", "http://a"]);
    assert!(Arc::ptr_eq(&base.entries, &same.entries));
    assert!(Arc::ptr_eq(&base.urls, &same.urls));
}

#[test]
fn updated_survives_a_registration_wave() {
    // Grow one URL at a time — the registration pattern — and verify the
    // final ring is indistinguishable from a single full build.
    let urls: Vec<String> = (0..40).map(|i| format!("http://w{i}:8000")).collect();
    let mut ring = HashRing::new(std::iter::empty::<&str>());
    for i in 0..urls.len() {
        ring = ring.updated(&urls[..=i]);
    }
    let rebuilt = HashRing::new(&urls);
    assert_eq!(ring.len(), rebuilt.len());
    assert_eq!(ring.worker_count(), rebuilt.worker_count());
    assert_same_selection(&ring, &rebuilt, 1000);
}

#[test]
fn updated_to_empty_clears_the_ring() {
    let base = HashRing::new(["http://a"]);
    let cleared = base.updated(std::iter::empty::<&str>());
    assert!(cleared.is_empty());
    assert_eq!(cleared.worker_count(), 0);
}

#[test]
fn a_ring_is_over_its_own_urls_in_its_own_order() {
    let ring = HashRing::new(["http://a", "http://b", "http://c"]);
    assert!(ring.is_over(["http://a", "http://b", "http://c"]));
    assert!(!ring.is_over(["http://a", "http://b"]), "one short");
    assert!(
        !ring.is_over(["http://a", "http://b", "http://c", "http://d"]),
        "one more"
    );
    assert!(
        !ring.is_over(["http://b", "http://a", "http://c"]),
        "another order"
    );
    assert!(
        !ring.is_over(["http://a", "http://a", "http://b", "http://c"]),
        "a repeat"
    );

    // Retained URLs keep their place and a new one goes behind them, as
    // a membership list does when a worker leaves and another joins.
    let moved = ring.updated(["http://a", "http://c", "http://d"]);
    assert!(moved.is_over(["http://a", "http://c", "http://d"]));
}

#[test]
fn updated_dedupes_repeated_urls() {
    let base = HashRing::new(["http://a"]);
    let ring = base.updated(["http://a", "http://b", "http://b"]);
    assert_eq!(ring.worker_count(), 2);
    assert_eq!(ring.len(), 2 * VIRTUAL_NODES_PER_WORKER);
}

#[test]
fn new_dedupes_repeated_urls() {
    // A repeated URL must not carry extra virtual-node weight.
    let deduped = HashRing::new(["http://a", "http://a"]);
    let single = HashRing::new(["http://a"]);
    assert_eq!(deduped.worker_count(), 1);
    assert_eq!(deduped.len(), VIRTUAL_NODES_PER_WORKER);
    assert_eq!(deduped.len(), single.len());
    assert_same_selection(&deduped, &single, 100);
}

#[test]
fn duplicate_inputs_build_the_same_ring_through_both_paths() {
    let duplicated = ["http://a", "http://b", "http://b"];
    let rebuilt = HashRing::new(duplicated);
    let incremental = HashRing::new(["http://a"]).updated(duplicated);

    assert_eq!(rebuilt.worker_count(), incremental.worker_count());
    assert_eq!(rebuilt.len(), incremental.len());
    assert_same_selection(&rebuilt, &incremental, 500);
}

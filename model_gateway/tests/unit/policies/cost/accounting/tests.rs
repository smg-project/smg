use std::sync::mpsc;

use super::*;

#[test]
fn key_positions_are_powers_of_two_then_full_length() {
    assert_eq!(key_positions(0).collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(key_positions(1).collect::<Vec<_>>(), vec![0]);
    assert_eq!(key_positions(2).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(key_positions(5).collect::<Vec<_>>(), vec![0, 1, 3, 4]);
    assert_eq!(key_positions(8).collect::<Vec<_>>(), vec![0, 1, 3, 7]);
    assert_eq!(key_positions(9).collect::<Vec<_>>(), vec![0, 1, 3, 7, 8]);
}

#[test]
fn bookings_sum_until_released_or_expired() {
    let acc = OptimisticAccounting::new(Duration::from_millis(50));
    acc.record_dispatch("w1", 100, &[]);
    acc.record_dispatch("w1", 200, &[]);
    assert_eq!(acc.pending_prefill_tokens("w1"), 300);
    acc.release("w1");
    assert_eq!(acc.pending_prefill_tokens("w1"), 200);
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(acc.pending_prefill_tokens("w1"), 0);
}

#[test]
fn reconcile_releases_bookings_beyond_the_live_count_oldest_first() {
    let acc = OptimisticAccounting::new(Duration::from_secs(5));
    acc.record_dispatch("w1", 100, &[]);
    acc.record_dispatch("w1", 200, &[]);
    acc.record_dispatch("w1", 400, &[]);
    assert_eq!(acc.reconcile("w1", 3), 0, "nothing beyond the live count");
    assert_eq!(acc.reconcile("w1", 1), 2, "two completions never arrived");
    assert_eq!(
        acc.pending_prefill_tokens("w1"),
        400,
        "the newest booking is the one still in flight"
    );
    assert_eq!(acc.reconcile("w1", 0), 1);
    assert_eq!(acc.pending_prefill_tokens("w1"), 0);
    assert_eq!(
        acc.reconcile("w1", 0),
        0,
        "a worker with no bookings releases nothing"
    );
}

#[test]
fn predicted_overlap_matches_shared_prefixes_at_keyed_depths() {
    let acc = OptimisticAccounting::new(Duration::from_secs(5));
    let prefix: Vec<u64> = (1..=10).collect();
    acc.record_dispatch("w1", 0, &prefix);

    // A longer sibling sharing all ten blocks matches the full-length key.
    let longer: Vec<u64> = (1..=12).collect();
    let found = acc.predicted_overlaps(&longer);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].0, "w1");
    assert_eq!(
        found[0].1, 8.0,
        "deepest keyed depth below 12 shared with 10 is 8"
    );

    // The identical prompt gets the full ten blocks.
    let found = acc.predicted_overlaps(&prefix);
    assert_eq!(found[0].1, 10.0);

    // A prompt sharing only three blocks matches the depth-2 key.
    let short: Vec<u64> = vec![1, 2, 3, 99, 98];
    let found = acc.predicted_overlaps(&short);
    assert_eq!(found[0].1, 2.0);

    // Nothing shared, nothing predicted.
    assert!(acc.predicted_overlaps(&[70, 80, 90]).is_empty());

    // Another worker gets the same prefix: both are reported.
    acc.record_dispatch("w2", 0, &prefix);
    let mut urls: Vec<String> = acc
        .predicted_overlaps(&prefix)
        .into_iter()
        .map(|(url, _)| url.to_string())
        .collect();
    urls.sort();
    assert_eq!(urls, vec!["w1", "w2"]);

    acc.forget_worker("w1");
    let found = acc.predicted_overlaps(&prefix);
    assert_eq!(found.len(), 1);
    assert_eq!(&*found[0].0, "w2");
}

/// More live keys than the bound under one long TTL: the eviction walk
/// must drop the oldest and return, not cycle a live head through the
/// queue for ever under the lock.
#[test]
fn more_live_keys_than_the_bound_evict_the_oldest_and_return() {
    let acc = Arc::new(OptimisticAccounting::new(Duration::from_secs(3600)));
    let extra = 100u64;
    let total = MAX_PREDICTED_KEYS as u64 + extra;
    let (done_tx, done_rx) = mpsc::channel();
    let worker = Arc::clone(&acc);
    std::thread::spawn(move || {
        for key in 1..=total {
            worker.record_dispatch("w1", 0, &[key]);
        }
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(60))
        .expect("record_dispatch returns with more live keys than the bound");
    let state = acc.state.lock();
    assert!(state.predicted_order.len() <= MAX_PREDICTED_KEYS);
    assert_eq!(state.predicted.len(), state.predicted_order.len());
    assert!(
        state.predicted.contains_key(&total),
        "the newest keys are the ones kept"
    );
    assert!(
        !state.predicted.contains_key(&1),
        "the oldest keys are the ones evicted"
    );
}

#[test]
fn predictions_expire() {
    let acc = OptimisticAccounting::new(Duration::from_millis(30));
    acc.record_dispatch("w1", 0, &[1, 2, 3, 4]);
    assert_eq!(acc.predicted_overlaps(&[1, 2, 3, 4]).len(), 1);
    std::thread::sleep(Duration::from_millis(40));
    assert!(acc.predicted_overlaps(&[1, 2, 3, 4]).is_empty());
}

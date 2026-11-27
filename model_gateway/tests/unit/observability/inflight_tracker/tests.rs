use std::time::Duration;

use super::*;
use crate::observability::metrics::test_support::render_with_recorder;

impl InFlightRequestTracker {
    fn insert_with_time(&self, request_id: u64, start_time: Instant) {
        self.requests.insert(request_id, start_time);
    }
}

#[test]
fn test_track_and_drop() {
    let tracker = InFlightRequestTracker::new();
    {
        let _guard1 = tracker.track();
        let _guard2 = tracker.track();
        assert_eq!(tracker.len(), 2);
    }
    assert_eq!(tracker.len(), 0);
}

#[test]
fn test_guard_auto_deregister() {
    let tracker = InFlightRequestTracker::new();
    let guard = tracker.track();
    assert_eq!(tracker.len(), 1);
    drop(guard);
    assert_eq!(tracker.len(), 0);
}

#[test]
fn test_request_age_tracking() {
    let tracker = InFlightRequestTracker::new();
    let _guard = tracker.track();
    std::thread::sleep(Duration::from_millis(100));

    let entry = tracker.requests.iter().next().unwrap();
    let age = entry.value().elapsed();
    assert!(age >= Duration::from_millis(100));
}

#[test]
fn test_collect_ages_empty() {
    let tracker = InFlightRequestTracker::new();
    let ages = tracker.collect_ages();
    assert!(ages.is_empty());
}

#[test]
fn test_collect_ages() {
    let tracker = InFlightRequestTracker::new();
    let now = Instant::now();

    tracker.insert_with_time(1, now);
    tracker.insert_with_time(2, now - Duration::from_secs(45));
    tracker.insert_with_time(3, now - Duration::from_secs(100));

    let ages = tracker.collect_ages();
    assert_eq!(ages.len(), 3);
    // Ages should be approximately 0, 45, 100 (order may vary due to DashMap)
    let mut sorted_ages = ages.clone();
    sorted_ages.sort_unstable();
    assert!(sorted_ages[0] <= 1); // ~0s
    assert!((44..=46).contains(&sorted_ages[1])); // ~45s
    assert!((99..=101).contains(&sorted_ages[2])); // ~100s
}

#[test]
fn test_concurrent_tracking() {
    use std::thread;

    let tracker = InFlightRequestTracker::new();
    let mut handles = vec![];

    for _ in 0..10 {
        let t = tracker.clone();
        handles.push(thread::spawn(move || {
            (0..100).map(|_| t.track()).collect::<Vec<_>>()
        }));
    }

    let all_guards: Vec<_> = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();

    assert_eq!(tracker.len(), 1000);
    drop(all_guards);
    assert_eq!(tracker.len(), 0);
}

#[test]
fn test_unique_ids() {
    let tracker = InFlightRequestTracker::new();
    let g1 = tracker.track();
    let g2 = tracker.track();
    let g3 = tracker.track();

    assert_ne!(g1.request_id, g2.request_id);
    assert_ne!(g2.request_id, g3.request_id);
    assert_eq!(tracker.len(), 3);
    assert_ne!(
        tracker.snapshot().instance_id,
        InFlightRequestTracker::new().snapshot().instance_id
    );
}

/// What two releases that interleave their loads and sets leave behind:
/// the gauge above the count, set through the recorder as the edge sets
/// it. One tick of the sampler reads the count and sets the gauge from
/// it.
#[test]
fn a_sampler_tick_sets_the_active_gauge_from_the_count_after_a_stale_set() {
    let tracker = InFlightRequestTracker::new();
    let _guards = [tracker.track(), tracker.track()];
    assert_eq!(tracker.len(), 2);
    let rendered = render_with_recorder(|| {
        Metrics::set_http_connections_active(3);
        tracker.sample_and_record();
    });
    assert!(
        rendered
            .lines()
            .any(|line| line == "smg_http_connections_active 2"),
        "{rendered}"
    );
}

#[test]
fn test_draining_flag() {
    let tracker = InFlightRequestTracker::new();
    assert!(!tracker.is_draining());
    tracker.begin_drain();
    assert!(tracker.is_draining());
    // Idempotent
    tracker.begin_drain();
    assert!(tracker.is_draining());
}

#[tokio::test]
async fn test_wait_for_drain_already_empty() {
    let tracker = InFlightRequestTracker::new();
    tracker.begin_drain();
    let drained = tracker.wait_for_drain(Duration::from_secs(1)).await;
    assert!(drained);
}

#[tokio::test]
async fn test_wait_for_drain_completes_when_guards_drop() {
    let tracker = InFlightRequestTracker::new();
    let guard1 = tracker.track();
    let guard2 = tracker.track();
    assert_eq!(tracker.len(), 2);

    tracker.begin_drain();

    let tracker_clone = tracker.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "test needs a concurrent task to exercise drain notification"
    )]
    let drain_handle =
        tokio::spawn(async move { tracker_clone.wait_for_drain(Duration::from_secs(5)).await });

    // Give the drain task a moment to start waiting
    tokio::time::sleep(Duration::from_millis(10)).await;

    drop(guard1);
    assert_eq!(tracker.len(), 1);

    drop(guard2);
    assert_eq!(tracker.len(), 0);

    let drained = drain_handle.await.unwrap();
    assert!(drained);
}

#[tokio::test]
async fn test_wait_for_drain_times_out() {
    let tracker = InFlightRequestTracker::new();
    let _guard = tracker.track();

    tracker.begin_drain();

    let drained = tracker.wait_for_drain(Duration::from_millis(50)).await;
    assert!(!drained);
    assert_eq!(tracker.len(), 1);
}

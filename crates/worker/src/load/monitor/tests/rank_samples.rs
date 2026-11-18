use super::*;

fn setup() -> (Arc<WorkerMonitor>, Arc<dyn Worker>) {
    let (registry, monitor) = build_monitor();
    let worker = ready_worker("grpc://rank-samples:9000", "llama-3");
    registry.register(Arc::clone(&worker)).unwrap();
    (monitor, worker)
}

fn published_sample(monitor: &WorkerMonitor, worker: &dyn Worker) -> Instant {
    monitor.flush_pushed();
    monitor
        .load_state
        .snapshot()
        .report(worker.url())
        .unwrap()
        .1
}

#[tokio::test]
async fn a_mixed_rank_sample_advances_only_after_its_oldest_row_is_replaced() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    let t1 = t0 + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), t1);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        t0 - PUSHED_ONE_WAY_MARGIN
    );

    let t2 = t1 + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(3), t2);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        t1 - PUSHED_ONE_WAY_MARGIN
    );

    let t3 = t2 + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 1, &pushed(4), t3);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        t2 - PUSHED_ONE_WAY_MARGIN
    );
}

#[tokio::test]
async fn sample_age_orders_accounting_while_receipt_orders_stream_freshness() {
    let (monitor, worker) = setup();
    let t0 = Instant::now() + Duration::from_secs(60);
    let t1 = t0 + Duration::from_secs(1);
    let aged = PushedRecord {
        age: Duration::from_secs(20),
        ..pushed(2)
    };
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &aged, t1);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        t1 - Duration::from_secs(20) - PUSHED_ONE_WAY_MARGIN,
        "the last receipt can carry the oldest engine sample",
    );
    assert_eq!(
        monitor.poll_mode(worker.url(), t1, Duration::from_secs(10)),
        PollMode::Suppressed,
        "sample age cannot make an active stream look quiet",
    );
}

#[tokio::test]
async fn replacing_a_quiet_row_uses_the_poll_sample_and_preserves_stream_freshness() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    let tick = t0 + Duration::from_secs(10);
    let pushed_at = tick + Duration::from_secs(1);
    let polled_sample = pushed_at + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), t0);
    monitor.apply_pushed_load(&worker, 0, &pushed(3), pushed_at);
    let poll = WorkerLoadResponse {
        dp_rank_count: 2,
        sampled_at: Some(polled_sample),
        loads: (0..2)
            .map(|dp_rank| SchedulerLoadSnapshot {
                dp_rank,
                num_running_reqs: 4,
                max_total_num_tokens: 1_000,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let (carried, sampled_at) = monitor.apply_polled_load(worker.url(), &poll, tick);
    assert_eq!(carried.len(), 1);
    assert_eq!(carried[0].dp_rank, 0);
    assert_eq!(sampled_at, Some(pushed_at - PUSHED_ONE_WAY_MARGIN));

    // A later push of rank 0 carries the quiet rank's POLL row, so its
    // boundary must now be the poll sample, not rank 1's original push.
    let next = polled_sample + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(5), next);
    assert_eq!(published_sample(&monitor, worker.as_ref()), polled_sample);
    assert_eq!(
        monitor.poll_mode(worker.url(), next, Duration::from_secs(10)),
        PollMode::Fallback,
        "a poll replaces a sample but is not the quiet stream's receipt",
    );
}

#[tokio::test]
async fn a_partial_polls_carried_boundary_excludes_rows_it_does_not_publish() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    let tick = t0 + Duration::from_secs(10);
    let fresh = tick + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), fresh);
    let poll = WorkerLoadResponse {
        // Partial and without absolute capacity: it cannot retract rank 0
        // from the file, but its publish does not contain that rank either.
        dp_rank_count: 2,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 1,
            ..Default::default()
        }],
        sampled_at: Some(fresh + Duration::from_secs(1)),
        ..Default::default()
    };
    let (carried, sampled_at) = monitor.apply_polled_load(worker.url(), &poll, tick);
    assert_eq!(carried.len(), 1);
    assert_eq!(carried[0].dp_rank, 1);
    assert_eq!(sampled_at, Some(fresh - PUSHED_ONE_WAY_MARGIN));
    assert_eq!(
        monitor
            .pushed_reports
            .lock()
            .get(worker.url())
            .unwrap()
            .sampled_at,
        Some(t0 - PUSHED_ONE_WAY_MARGIN),
        "the file still carries rank 0, unlike this poll's output",
    );
}

#[tokio::test]
async fn an_authoritative_poll_removes_an_omitted_ranks_sample_boundary() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    let tick = t0 + Duration::from_secs(10);
    let fresh = tick + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), fresh);
    let poll = WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 1,
            max_total_num_tokens: 1_000,
            ..Default::default()
        }],
        sampled_at: Some(fresh + Duration::from_secs(1)),
        ..Default::default()
    };
    let (carried, sampled_at) = monitor.apply_polled_load(worker.url(), &poll, tick);
    assert_eq!(carried.len(), 1);
    assert_eq!(sampled_at, Some(fresh - PUSHED_ONE_WAY_MARGIN));
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        fresh - PUSHED_ONE_WAY_MARGIN
    );
    assert_eq!(monitor.pushed_at.lock()[worker.url()].len(), 1);

    let next = fresh + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 1, &pushed(3), next);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        next - PUSHED_ONE_WAY_MARGIN
    );
}

#[tokio::test]
async fn an_empty_poll_keeps_only_post_poll_push_times_and_can_clear_the_boundary() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    let tick = t0 + Duration::from_secs(10);
    let fresh = tick + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), fresh);
    monitor.retract_pushed_load(worker.url(), None, tick);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        fresh - PUSHED_ONE_WAY_MARGIN
    );
    assert_eq!(monitor.pushed_at.lock()[worker.url()].len(), 1);

    monitor.retract_pushed_load(worker.url(), None, fresh + Duration::from_secs(1));
    assert!(monitor.pushed_at.lock()[worker.url()].is_empty());
    let reports = monitor.pushed_reports.lock();
    let report = reports.get(worker.url()).unwrap();
    assert!(report.loads.is_empty());
    assert_eq!(report.sampled_at, None);
}

#[tokio::test]
async fn evicting_a_worker_clears_both_rank_receipts_and_samples() {
    let (monitor, worker) = setup();
    let t0 = Instant::now();
    monitor.apply_pushed_load(&worker, 0, &pushed(1), t0);
    monitor.apply_pushed_load(&worker, 1, &pushed(2), t0);
    monitor.evict_worker_loads(&worker);
    assert!(!monitor.pushed_at.lock().contains_key(worker.url()));
    assert!(!monitor.pushed_reports.lock().contains_key(worker.url()));

    let next = t0 + Duration::from_secs(1);
    monitor.apply_pushed_load(&worker, 1, &pushed(3), next);
    assert_eq!(
        published_sample(&monitor, worker.as_ref()),
        next - PUSHED_ONE_WAY_MARGIN
    );
    assert_eq!(
        monitor
            .load_state
            .snapshot()
            .get(worker.url())
            .unwrap()
            .loads
            .len(),
        1
    );
}

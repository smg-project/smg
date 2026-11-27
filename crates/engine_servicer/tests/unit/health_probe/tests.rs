use std::{
    future::{pending, ready},
    sync::{Arc, Mutex},
};

use super::*;

fn never() -> Option<Instant> {
    None
}

/// The probe's own output is the answer, at once.
#[tokio::test(start_paused = true)]
async fn the_probes_output_passes_at_once() {
    let before = Instant::now();
    let verdict = run(DEFAULT_TIMEOUT, never, ready(Ok(()))).await;
    assert_eq!(verdict, Verdict::Passed);
    assert_eq!(before.elapsed(), Duration::ZERO, "no step waited");
}

/// A probe that fails with nothing pushed since it started is answered
/// at its failure, not at the deadline.
#[tokio::test(start_paused = true)]
async fn a_failed_probe_with_no_push_fails_at_once() {
    let verdict = run(
        DEFAULT_TIMEOUT,
        never,
        ready(Err(Status::unavailable("refused"))),
    )
    .await;
    assert_eq!(verdict, Verdict::Failed);
}

/// A push after the start is read before the probe's own verdict: a
/// scheduler that refuses the probe but speaks is alive.
#[tokio::test(start_paused = true)]
async fn a_push_since_the_start_beats_a_failed_probe() {
    let stamp = Arc::new(Mutex::new(None));
    let reader = Arc::clone(&stamp);
    let pushed = async move {
        *stamp.lock().expect("stamp") = Some(Instant::now() + Duration::from_millis(1));
        Err(Status::unavailable("refused"))
    };
    let verdict = run(
        DEFAULT_TIMEOUT,
        move || *reader.lock().expect("stamp"),
        pushed,
    )
    .await;
    assert_eq!(verdict, Verdict::Passed);
}

/// A push while the probe waits its turn is read on the next step; a
/// stamp from before the start is not one.
#[tokio::test(start_paused = true)]
async fn a_push_while_the_probe_waits_passes_on_the_next_step() {
    let before = Instant::now();
    let stale = Instant::now() - Duration::from_secs(1);
    let reads = Mutex::new(0_u32);
    // Stale for three steps, then a push one millisecond after whatever
    // start the wait took.
    let last_output_at = || {
        let mut reads = reads.lock().expect("reads");
        *reads += 1;
        Some(if *reads <= 3 {
            stale
        } else {
            Instant::now() + Duration::from_millis(1)
        })
    };
    let verdict = run(DEFAULT_TIMEOUT, last_output_at, pending()).await;
    assert_eq!(verdict, Verdict::Passed);
    assert_eq!(
        before.elapsed(),
        STEP * 4,
        "three stale reads, the fourth a push"
    );
}

/// A stamp within the millisecond the wait started in reads as since
/// the start (the connector floors its stamps to the millisecond); one a
/// whole millisecond before it does not.
#[tokio::test(start_paused = true)]
async fn a_stamp_rounded_down_into_the_starting_millisecond_counts() {
    let floored = Instant::now() - Duration::from_micros(500);
    assert_eq!(
        run(
            DEFAULT_TIMEOUT,
            || Some(floored),
            ready(Err(Status::unavailable("refused")))
        )
        .await,
        Verdict::Passed
    );
    let stale = Instant::now() - STAMP_RESOLUTION;
    assert_eq!(
        run(
            DEFAULT_TIMEOUT,
            || Some(stale),
            ready(Err(Status::unavailable("refused")))
        )
        .await,
        Verdict::Failed
    );
}

/// Neither a push nor an answer within the bound: unhealthy at the bound.
#[tokio::test(start_paused = true)]
async fn neither_within_the_bound_times_out() {
    let before = Instant::now();
    let verdict = run(Duration::from_secs(3), never, pending()).await;
    assert_eq!(verdict, Verdict::TimedOut);
    assert_eq!(before.elapsed(), Duration::from_secs(3));
}

#[test]
fn a_role_other_than_null_is_disaggregated() {
    assert!(!disaggregated_role(""));
    assert!(!disaggregated_role(r#"{"tp_size":2}"#));
    assert!(!disaggregated_role(r#"{"disaggregation_mode":"null"}"#));
    for role in ["prefill", "decode", "encode"] {
        assert!(disaggregated_role(&format!(
            r#"{{"disaggregation_mode":"{role}"}}"#
        )));
    }
}

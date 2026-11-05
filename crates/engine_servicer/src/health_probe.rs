//! The deep `HealthCheck` of a wire whose Python servicer probes the
//! scheduler: one request of a single token through the wire's own request
//! path, judged the way those servicers judge it. The wire's module builds
//! the request, submits it as a client's (admitted by the registry, placed
//! and translated as any request, its engine-side request aborted when the
//! future drops) and keeps its own words for the three answers; this module
//! holds the wait, so every wire answers the same case the same way.

use std::{future::Future, time::Duration};

use tokio::time::{Instant, MissedTickBehavior};
use tonic::Status;

/// Seconds the probe waits with neither a push nor an answer before the
/// RPC answers unhealthy, unless the wire's knob says otherwise.
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// How often the wait reads the engine's last push while the probe is out:
/// a push, or a probe that fails at once, is answered at most this late.
pub(crate) const STEP: Duration = Duration::from_millis(500);
/// The resolution of the connector's output stamp (`last_output_at`).
const STAMP_RESOLUTION: Duration = Duration::from_millis(1);

/// What `HealthCheck` says of a disaggregated role, which the probe skips:
/// a prefill or decode worker pairs each request with a peer and an encode
/// worker serves no generate, so a lone one-token request would exercise
/// the rendezvous, not the scheduler.
pub(crate) const SHALLOW_MESSAGE: &str =
    "Health check passed (disaggregation worker; shallow probe)";

/// The wait's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// The engine pushed something since the probe started, or the probe's
    /// own output came.
    Passed,
    /// The probe failed on its own (refused at the submit, its stream
    /// raised or ended without an output) and nothing was pushed since it
    /// started: answered the moment it failed, not at the deadline.
    Failed,
    /// Neither within `timeout`.
    TimedOut,
}

impl Verdict {
    pub(crate) fn healthy(self) -> bool {
        self == Self::Passed
    }
}

/// The probe's bound from the wire's knob (whole seconds), [`DEFAULT_TIMEOUT`]
/// when unset.
pub(crate) fn timeout_from_env(knob: &str) -> Duration {
    let default = usize::try_from(DEFAULT_TIMEOUT.as_secs()).unwrap_or(usize::MAX);
    Duration::from_secs(
        u64::try_from(crate::kv_events::env_usize(knob, default)).unwrap_or(u64::MAX),
    )
}

/// Whether the launch's server args name a disaggregation role (any
/// `disaggregation_mode` other than the engines' `null`).
pub(crate) fn disaggregated_role(server_args_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(server_args_json)
        .ok()
        .is_some_and(|args| {
            args.get("disaggregation_mode")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|mode| mode != "null")
        })
}

/// Drive `probe`, the wire's one-token request resolved at its first output
/// (`Ok`) or its failure (`Err`), for at most `timeout` from now. Every
/// [`STEP`], and when the probe resolves, `last_output_at` is read: an
/// output the engine sent after the wait started means it is alive, however
/// busy (the probe waits its turn at a busy scheduler, and the probe's own
/// output arrives as one), and it is read before the probe's own verdict, as
/// the Python servicers read their receive stamp.
pub(crate) async fn run<F>(
    timeout: Duration,
    last_output_at: impl Fn() -> Option<Instant>,
    probe: F,
) -> Verdict
where
    F: Future<Output = Result<(), Status>>,
{
    let started = Instant::now();
    // The connector stamps an output to the millisecond, rounded down: a
    // stamp within the millisecond the wait started in is read as since
    // the start (a batch routed right after the probe's submit lands there).
    let pushed_since_start = || last_output_at().is_some_and(|at| at + STAMP_RESOLUTION > started);
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    tokio::pin!(probe);
    let mut steps = tokio::time::interval_at(Instant::now() + STEP, STEP);
    steps.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            outcome = &mut probe => {
                return if pushed_since_start() || outcome.is_ok() {
                    Verdict::Passed
                } else {
                    Verdict::Failed
                };
            }
            _ = steps.tick() => {
                if pushed_since_start() {
                    return Verdict::Passed;
                }
            }
            () = &mut deadline => return Verdict::TimedOut,
        }
    }
}

#[cfg(test)]
mod tests {
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
}

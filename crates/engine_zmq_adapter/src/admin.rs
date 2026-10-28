//! The rules every caller of the scheduler's admin control calls applies
//! before the adapter sends them: a cache flush's reply wait, the profiler's
//! reply wait and the profile id a start is stamped with. The Python
//! servicers set them (their `max(30, timeout_s + 10)`, their
//! `PROFILE_COMM_TIMEOUT`, their wall-clock profile id); the Rust servicer
//! and a frontend's direct lane read them here, once, so the two lanes answer
//! the same request the same way.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use smg_grpc_client::common_proto;

use crate::sglang::SglangProfileStart;

/// How long a cache flush's reply may take: the scheduler itself waits up to
/// `timeout_s` for an idle moment (a zero flushes at once or refuses), and the
/// reply wait is that plus slack, the Python servicer's `max(30, timeout_s +
/// 10)`; a wait the clock cannot hold is unbounded.
pub fn flush_reply_wait(timeout_s: f32) -> Duration {
    Duration::try_from_secs_f32(timeout_s.max(20.0) + 10.0).unwrap_or(Duration::MAX)
}

/// How long a profiler start or stop may take to answer: the Python
/// servicer's `PROFILE_COMM_TIMEOUT`.
pub const PROFILE_WAIT: Duration = Duration::from_secs(600);

/// The profile id a start is stamped with, as the Python servicer stamps it:
/// the wall-clock start time in seconds, six decimals.
pub fn profile_id() -> String {
    format!(
        "{:.6}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    )
}

impl SglangProfileStart {
    /// The scheduler's start for a `StartProfile` request: its options as
    /// given, stamped with [`profile_id`] now.
    pub fn from_request(request: common_proto::StartProfileRequest) -> Self {
        Self {
            output_dir: request.output_dir,
            start_step: request.start_step.map(i64::from),
            num_steps: request.num_steps.map(i64::from),
            activities: request.activities,
            with_stack: request.with_stack,
            record_shapes: request.record_shapes,
            profile_by_stage: request.profile_by_stage,
            profile_id: profile_id(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flush reply wait is the scheduler's own wait plus slack, never
    /// under 30 s, and unbounded for a wait the clock cannot hold.
    #[test]
    fn the_flush_reply_wait_is_the_schedulers_wait_plus_slack() {
        assert_eq!(flush_reply_wait(0.0), Duration::from_secs(30));
        assert_eq!(flush_reply_wait(20.0), Duration::from_secs(30));
        assert_eq!(flush_reply_wait(45.0), Duration::from_secs(55));
        assert_eq!(flush_reply_wait(f32::MAX), Duration::MAX);
    }

    /// A profile id is the wall-clock start time in seconds with six
    /// decimals, and a start takes the request's options with one.
    #[test]
    fn a_profile_start_carries_the_requests_options_and_a_wall_clock_id() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let start = SglangProfileStart::from_request(common_proto::StartProfileRequest {
            output_dir: Some("/traces".to_string()),
            start_step: Some(3),
            num_steps: Some(8),
            activities: vec!["CPU".to_string(), "GPU".to_string()],
            with_stack: Some(true),
            record_shapes: None,
            profile_by_stage: true,
        });
        let (seconds, decimals) = start.profile_id.split_once('.').expect("seconds.decimals");
        assert_eq!(decimals.len(), 6, "{}", start.profile_id);
        assert!(
            seconds.bytes().all(|b| b.is_ascii_digit()),
            "{}",
            start.profile_id
        );
        let stamped: f64 = start.profile_id.parse().unwrap();
        assert!(
            stamped >= before - 1e-6 && stamped <= before + 60.0,
            "{stamped} vs {before}"
        );
        assert_eq!(start.output_dir.as_deref(), Some("/traces"));
        assert_eq!(start.start_step, Some(3));
        assert_eq!(start.num_steps, Some(8));
        assert_eq!(start.activities, ["CPU", "GPU"]);
        assert_eq!(start.with_stack, Some(true));
        assert_eq!(start.record_shapes, None);
        assert!(start.profile_by_stage);
        assert_eq!(PROFILE_WAIT, Duration::from_secs(600));
    }
}

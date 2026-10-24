//! What the admin RPCs check before an engine is asked anything, and the
//! RPCs themselves where the engine's wire carries them: `FlushCache`,
//! `StartProfile` and `StopProfile` are the scheduler's own control requests
//! (SGLang's `FlushCacheReqInput` and `ProfileReq` through the plugin's
//! control call; a TokenSpeed scheduler of the control-wire revision takes
//! the same call), carried to every rank and aggregated as the Python
//! servicers aggregate their per-rank results. A wire without control calls
//! answers UNIMPLEMENTED naming the gap (the adapter's word).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use engine_zmq_adapter::SglangProfileStart;
use smg_grpc_client::common_proto as common;
use tonic::{Code, Request, Status};
use tracing::warn;

use crate::servicer::Shared;

/// `FlushCache`'s `timeout_s`, as the Python servicer accepts it: finite and
/// non-negative (a zero flushes at once or refuses), anything else refused
/// as INVALID_ARGUMENT.
pub(crate) fn flush_timeout(timeout_s: f32) -> Result<f32, Status> {
    if !timeout_s.is_finite() || timeout_s < 0.0 {
        return Err(Status::invalid_argument(
            "timeout_s must be finite and non-negative",
        ));
    }
    Ok(timeout_s)
}

/// The Python servicer's `PROFILE_COMM_TIMEOUT`.
const PROFILE_WAIT: Duration = Duration::from_secs(600);

pub(crate) async fn flush_cache(
    shared: &Shared,
    request: Request<common::FlushCacheRequest>,
) -> Result<common::FlushCacheResponse, Status> {
    let timeout_s = flush_timeout(request.get_ref().timeout_s)?;
    let engine = shared.engine()?;
    // The scheduler itself waits up to `timeout_s` for an idle moment (a
    // zero flushes at once or refuses); the reply wait is that plus slack,
    // the Python servicer's `max(30, timeout_s + 10)`.
    let wait = Duration::try_from_secs_f32(timeout_s.max(20.0) + 10.0).unwrap_or(Duration::MAX);
    match engine.flush_cache_control(timeout_s, wait).await {
        Ok((success, message)) => Ok(common::FlushCacheResponse { success, message }),
        // A wire without control calls: the adapter's own refusal, as is.
        Err(status) if status.code() == Code::Unimplemented => Err(status),
        Err(status) if status.code() == Code::DeadlineExceeded => {
            Err(Status::deadline_exceeded(format!(
                "Flush cache timed out after {}s. The scheduler may be unresponsive or under \
                 heavy load.",
                wait.as_secs()
            )))
        }
        Err(status) => {
            warn!(error = status.message(), "FlushCache failed");
            Err(Status::internal(format!(
                "Flush cache failed: {}",
                status.message()
            )))
        }
    }
}

pub(crate) async fn start_profile(
    shared: &Shared,
    request: common::StartProfileRequest,
) -> Result<common::ProfileResponse, Status> {
    let engine = shared.engine()?;
    let start = SglangProfileStart {
        output_dir: request.output_dir,
        start_step: request.start_step.map(i64::from),
        num_steps: request.num_steps.map(i64::from),
        activities: request.activities,
        with_stack: request.with_stack,
        record_shapes: request.record_shapes,
        profile_by_stage: request.profile_by_stage,
        profile_id: profile_id(),
    };
    profile(
        engine.profile_control(Some(start), PROFILE_WAIT).await,
        "Start profiling",
    )
}

pub(crate) async fn stop_profile(shared: &Shared) -> Result<common::ProfileResponse, Status> {
    let engine = shared.engine()?;
    profile(
        engine.profile_control(None, PROFILE_WAIT).await,
        "Stop profiling",
    )
}

fn profile(
    result: Result<(bool, String), Status>,
    op_name: &str,
) -> Result<common::ProfileResponse, Status> {
    match result {
        Ok((success, message)) => Ok(common::ProfileResponse { success, message }),
        Err(status) if status.code() == Code::Unimplemented => Err(status),
        Err(status) if status.code() == Code::DeadlineExceeded => Err(Status::deadline_exceeded(
            format!("{op_name} timed out after {}s", PROFILE_WAIT.as_secs()),
        )),
        Err(status) => {
            warn!(error = status.message(), "{op_name} failed");
            Err(Status::internal(format!(
                "{op_name} failed: {}",
                status.message()
            )))
        }
    }
}

/// The profile id the Python servicer stamps on a start: the wall-clock
/// start time in seconds.
fn profile_id() -> String {
    format!(
        "{:.6}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs_f64())
            .unwrap_or(0.0)
    )
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn a_flush_timeout_is_finite_and_non_negative() {
        assert_eq!(flush_timeout(0.0).unwrap(), 0.0);
        assert_eq!(flush_timeout(2.5).unwrap(), 2.5);
        for refused in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let status = flush_timeout(refused).expect_err("refused");
            assert_eq!(status.code(), Code::InvalidArgument, "{refused}");
            assert_eq!(
                status.message(),
                "timeout_s must be finite and non-negative"
            );
        }
    }
}

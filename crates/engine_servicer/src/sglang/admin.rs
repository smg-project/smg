//! `FlushCache`, `StartProfile` and `StopProfile`: the scheduler's own
//! control requests (`FlushCacheReqInput`, `ProfileReq`), carried to every
//! rank by the plugin's control call and aggregated as the Python servicer
//! aggregates its per-rank communicator results.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use engine_zmq_adapter::SglangProfileStart;
use smg_grpc_client::common_proto as common;
use tonic::{Code, Request, Status};
use tracing::warn;

use super::State;

/// The Python servicer's `PROFILE_COMM_TIMEOUT`.
const PROFILE_WAIT: Duration = Duration::from_secs(600);

pub(super) async fn flush_cache(
    state: &State,
    request: Request<common::FlushCacheRequest>,
) -> Result<common::FlushCacheResponse, Status> {
    let timeout_s = request.get_ref().timeout_s;
    if !timeout_s.is_finite() || timeout_s < 0.0 {
        return Err(Status::invalid_argument(
            "timeout_s must be finite and non-negative",
        ));
    }
    let engine = state.engine()?;
    // The scheduler itself waits up to `timeout_s` for an idle moment (a
    // zero flushes at once or refuses); the reply wait is that plus slack,
    // the Python servicer's `max(30, timeout_s + 10)`.
    let wait = Duration::try_from_secs_f32(timeout_s.max(20.0) + 10.0).unwrap_or(Duration::MAX);
    match engine.flush_cache_sglang(timeout_s, wait).await {
        Ok((success, message)) => Ok(common::FlushCacheResponse { success, message }),
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

pub(super) async fn start_profile(
    state: &State,
    request: common::StartProfileRequest,
) -> Result<common::ProfileResponse, Status> {
    let engine = state.engine()?;
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
        engine.profile_sglang(Some(start), PROFILE_WAIT).await,
        "Start profiling",
    )
}

pub(super) async fn stop_profile(state: &State) -> Result<common::ProfileResponse, Status> {
    let engine = state.engine()?;
    profile(
        engine.profile_sglang(None, PROFILE_WAIT).await,
        "Stop profiling",
    )
}

fn profile(
    result: Result<(bool, String), Status>,
    op_name: &str,
) -> Result<common::ProfileResponse, Status> {
    match result {
        Ok((success, message)) => Ok(common::ProfileResponse { success, message }),
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

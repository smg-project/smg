//! `FlushCache`: vLLM's `reset_prefix_cache` on every engine rank, under the
//! Python servicer's `admin.flush_cache` contract. `timeout_s == 0` is one
//! immediate attempt; a positive value retries until the deadline, since the
//! engine refuses while requests still hold KV blocks.
//!
//! `StartProfile` and `StopProfile`: vLLM's `profile(is_start)` on every
//! engine rank, the engine's own torch profiler as its launch configured it
//! (the trace directory and the profiler's options come from the engine's
//! configuration, not from the request), bounded by the client's deadline
//! under [`PROFILE_WAIT`].

use std::time::{Duration, Instant};

use smg_grpc_client::common_proto as common;
use tonic::{metadata::MetadataMap, Code, Request, Status};
use tracing::warn;

use super::State;
use crate::{admin::flush_timeout, rpc::grpc_timeout};

/// Bounds the single attempt of `timeout_s == 0`, so an engine that never
/// answers cannot hang the RPC.
const RESET_RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// Pause between attempts while in-flight requests drain.
const RETRY_INTERVAL: Duration = Duration::from_millis(100);
/// The most a profiler start or stop may take, the Python servicers'
/// `PROFILE_TIMEOUT`; the client's gRPC deadline caps it when shorter.
const PROFILE_WAIT: Duration = Duration::from_secs(600);

pub(super) async fn flush_cache(
    state: &State,
    request: Request<common::FlushCacheRequest>,
) -> Result<common::FlushCacheResponse, Status> {
    let timeout_s = flush_timeout(request.get_ref().timeout_s)?;
    let engine = state.shared.engine()?;
    let started = Instant::now();
    let immediate = timeout_s == 0.0;
    // A deadline too far out to represent is no deadline at all.
    let deadline = (!immediate)
        .then(|| Duration::try_from_secs_f32(timeout_s).ok())
        .flatten()
        .and_then(|wait| started.checked_add(wait));
    // The client's gRPC deadline caps the wait, as `context.time_remaining()`
    // does on the Python servicer.
    let rpc_deadline = grpc_timeout(request.metadata()).and_then(|wait| started.checked_add(wait));
    let remaining = |deadline: Option<Instant>| {
        deadline.map_or(Duration::MAX, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        })
    };

    loop {
        let budget = if immediate {
            RESET_RPC_TIMEOUT
        } else {
            remaining(deadline)
        }
        .min(remaining(rpc_deadline));
        if budget.is_zero() {
            return Err(timed_out());
        }
        match engine.reset_prefix_cache(budget).await {
            Ok(true) => return Ok(response(true, "Local KV prefix cache flushed successfully")),
            Ok(false) if immediate => {
                return Ok(response(
                    false,
                    "KV prefix cache reset refused; requests may be in flight",
                ))
            }
            Ok(false) => tokio::time::sleep(RETRY_INTERVAL.min(remaining(deadline))).await,
            Err(status) if status.code() == Code::DeadlineExceeded => return Err(timed_out()),
            Err(status) => {
                warn!(error = status.message(), "FlushCache failed");
                return Err(Status::internal(format!(
                    "Flush cache failed: {}",
                    status.message()
                )));
            }
        }
    }
}

pub(super) async fn start_profile(
    state: &State,
    request: Request<common::StartProfileRequest>,
) -> Result<common::ProfileResponse, Status> {
    refuse_launch_time_options(request.get_ref())?;
    let wait = profile_wait(request.metadata());
    let engine = state.shared.engine()?;
    profile(
        engine.profile_vllm(true, wait).await,
        "Start profiling",
        wait,
    )
}

pub(super) async fn stop_profile(
    state: &State,
    request: Request<common::StopProfileRequest>,
) -> Result<common::ProfileResponse, Status> {
    let wait = profile_wait(request.metadata());
    let engine = state.shared.engine()?;
    profile(
        engine.profile_vllm(false, wait).await,
        "Stop profiling",
        wait,
    )
}

/// The engine takes its profiler's directory and options at its launch, not
/// per request: a start that sets any of them is refused, so an operator is
/// never told a trace went where, or how, it did not.
fn refuse_launch_time_options(request: &common::StartProfileRequest) -> Result<(), Status> {
    let set: Vec<&str> = [
        (request.output_dir.is_some(), "output_dir"),
        (request.start_step.is_some(), "start_step"),
        (request.num_steps.is_some(), "num_steps"),
        (!request.activities.is_empty(), "activities"),
        (request.with_stack.is_some(), "with_stack"),
        (request.record_shapes.is_some(), "record_shapes"),
        (request.profile_by_stage, "profile_by_stage"),
    ]
    .into_iter()
    .filter_map(|(set, name)| set.then_some(name))
    .collect();
    if set.is_empty() {
        return Ok(());
    }
    Err(Status::invalid_argument(format!(
        "StartProfile options {} cannot be set per request on this engine: its profiler's \
         directory and options are its launch configuration; send a start with no options",
        set.join(", ")
    )))
}

/// [`PROFILE_WAIT`], or the client's gRPC deadline when that is shorter.
fn profile_wait(metadata: &MetadataMap) -> Duration {
    grpc_timeout(metadata).map_or(PROFILE_WAIT, |deadline| deadline.min(PROFILE_WAIT))
}

fn profile(
    result: Result<(), Status>,
    op_name: &str,
    wait: Duration,
) -> Result<common::ProfileResponse, Status> {
    match result {
        Ok(()) => Ok(common::ProfileResponse {
            success: true,
            message: format!("{op_name} succeeded"),
        }),
        Err(status) if status.code() == Code::DeadlineExceeded => Err(Status::deadline_exceeded(
            format!("{op_name} timed out after {}s", wait.as_secs_f64()),
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

fn response(success: bool, message: &str) -> common::FlushCacheResponse {
    common::FlushCacheResponse {
        success,
        message: message.to_string(),
    }
}

fn timed_out() -> Status {
    Status::deadline_exceeded(
        "Flush cache timed out; the engine may still complete an issued reset",
    )
}

//! `FlushCache`: vLLM's `reset_prefix_cache` on every engine rank, under the
//! Python servicer's `admin.flush_cache` contract. `timeout_s == 0` is one
//! immediate attempt; a positive value retries until the deadline, since the
//! engine refuses while requests still hold KV blocks.

use std::time::{Duration, Instant};

use smg_grpc_client::common_proto as common;
use tonic::{metadata::MetadataMap, Code, Request, Status};
use tracing::warn;

use super::State;

/// Bounds the single attempt of `timeout_s == 0`, so an engine that never
/// answers cannot hang the RPC.
const RESET_RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// Pause between attempts while in-flight requests drain.
const RETRY_INTERVAL: Duration = Duration::from_millis(100);

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

/// The client's `grpc-timeout` header (`<1-8 digits><H|M|S|m|u|n>`), if any.
fn grpc_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let value = metadata.get("grpc-timeout")?.to_str().ok()?;
    let unit = value.chars().last()?;
    let digits = &value[..value.len() - unit.len_utf8()];
    if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let amount: u64 = digits.parse().ok()?;
    Some(match unit {
        'H' => Duration::from_secs(amount * 3600),
        'M' => Duration::from_secs(amount * 60),
        'S' => Duration::from_secs(amount),
        'm' => Duration::from_millis(amount),
        'u' => Duration::from_micros(amount),
        'n' => Duration::from_nanos(amount),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use tonic::metadata::MetadataValue;

    use super::*;

    #[test]
    fn grpc_timeout_follows_the_header_grammar() {
        let parse = |value: &str| {
            let mut metadata = MetadataMap::new();
            metadata.insert("grpc-timeout", MetadataValue::try_from(value).unwrap());
            grpc_timeout(&metadata)
        };
        assert_eq!(parse("2H"), Some(Duration::from_secs(7200)));
        assert_eq!(parse("3M"), Some(Duration::from_secs(180)));
        assert_eq!(parse("30S"), Some(Duration::from_secs(30)));
        assert_eq!(parse("250m"), Some(Duration::from_millis(250)));
        assert_eq!(parse("7u"), Some(Duration::from_micros(7)));
        assert_eq!(parse("9n"), Some(Duration::from_nanos(9)));
        for malformed in ["", "S", "5", "123456789S", "+5S", "5x"] {
            assert_eq!(parse(malformed), None, "{malformed:?}");
        }
        assert_eq!(grpc_timeout(&MetadataMap::new()), None);
    }
}

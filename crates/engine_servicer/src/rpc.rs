//! The RPC bodies every wire's service shares. A wire's gRPC service trait is
//! its own (tonic generates one per proto service), so each wire keeps its
//! impl block and its proto types; what the four blocks had in common, word
//! for word, is written here once and called from a one-line body: the
//! answer of a `Generate` that did or did not start, the tokenizer bundle,
//! and the KV-event subscription with its records-only mode.

use std::{sync::Arc, time::Duration};

use smg_grpc_client::common_proto as common;
use tonic::{metadata::MetadataMap, Request, Response, Status};

use crate::{kv_events::RankLoads, load_stream, servicer::WireState, tokenizer_bundle, BoxStream};

/// The `Generate` RPC's answer from its handler's result: the response stream,
/// or the refusal, counted as a request that never started
/// (`request_not_started`) before it goes back to the caller.
pub(crate) fn generated<S: WireState, T>(
    state: &Arc<S>,
    result: Result<BoxStream<T>, Status>,
) -> Result<Response<BoxStream<T>>, Status> {
    result
        .inspect_err(|status| state.shared().metrics.request_not_started(status))
        .map(Response::new)
}

/// `GetTokenizer`: the servicer's tokenizer directory as the bundle stream
/// ([`tokenizer_bundle::get_tokenizer`]), refused without one.
pub(crate) async fn get_tokenizer<S: WireState>(
    state: &Arc<S>,
) -> Result<Response<BoxStream<common::GetTokenizerChunk>>, Status> {
    tokenizer_bundle::get_tokenizer(state.shared().tokenizer_dir.clone())
        .await
        .map(Response::new)
}

/// `SubscribeKvEvents`: the relay's event stream, or UNIMPLEMENTED with
/// `disabled`, the wire's reason, when the engine's publisher was not
/// configured. The load records alone (`loads_only`) are the servicer's own
/// figures, served with or without a relay: an engine whose events are off
/// streams its load all the same.
pub(crate) fn subscribe_kv_events<S: WireState + RankLoads>(
    state: &Arc<S>,
    request: Request<common::SubscribeKvEventsRequest>,
    disabled: &str,
) -> Result<Response<BoxStream<common::KvEventBatch>>, Status> {
    let request = request.into_inner();
    if request.loads_only {
        return Ok(Response::new(load_stream::load_records_of(state)));
    }
    let Some(relay) = &state.shared().kv_relay else {
        return Err(Status::unimplemented(disabled));
    };
    relay.subscribe(request).map(Response::new)
}

/// The client's `grpc-timeout` header (`<1-8 digits><H|M|S|m|u|n>`), if any.
pub(crate) fn grpc_timeout(metadata: &MetadataMap) -> Option<Duration> {
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

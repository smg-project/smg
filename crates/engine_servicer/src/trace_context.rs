//! The W3C trace context the Router sends in a call's gRPC metadata
//! (`traceparent`, `tracestate`), read once per request for the engine's
//! tracer: each servicer hands it to its engine the way the engine's own
//! frontend does, so the engine's spans for the request join the Router's
//! trace.

use std::collections::BTreeMap;

use tonic::metadata::MetadataMap;

/// The trace context among the call's metadata, keyed by header name;
/// `None` when the call carries none (the engine then starts a trace of its
/// own, or none).
pub(crate) fn trace_headers(metadata: &MetadataMap) -> Option<BTreeMap<String, String>> {
    let headers: BTreeMap<String, String> = ["traceparent", "tracestate"]
        .into_iter()
        .filter_map(|name| {
            let value = metadata.get(name)?.to_str().ok()?;
            Some((name.to_string(), value.to_string()))
        })
        .collect();
    (!headers.is_empty()).then_some(headers)
}

#[cfg(test)]
#[path = "../tests/unit/trace_context/tests.rs"]
mod tests;

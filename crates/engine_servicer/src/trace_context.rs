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
mod tests {
    use super::*;

    #[test]
    fn the_two_context_headers_are_read_and_nothing_else() {
        let mut metadata = MetadataMap::new();
        metadata.insert("user-agent", "grpc-rust/0.1".parse().unwrap());
        assert_eq!(trace_headers(&metadata), None);
        metadata.insert("traceparent", "00-aa-bb-01".parse().unwrap());
        metadata.insert("tracestate", "vendor=a".parse().unwrap());
        assert_eq!(
            trace_headers(&metadata),
            Some(BTreeMap::from([
                ("traceparent".to_string(), "00-aa-bb-01".to_string()),
                ("tracestate".to_string(), "vendor=a".to_string()),
            ]))
        );
    }
}

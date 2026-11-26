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

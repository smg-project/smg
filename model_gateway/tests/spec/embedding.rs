use openai_protocol::{
    classify::ClassifyRequest, common::GenerationRequest, embedding::EmbeddingRequest,
};
use serde_json::{from_str, json, to_string};

#[test]
fn test_embedding_request_serialization_string_input() {
    let req = EmbeddingRequest {
        model: "test-emb".to_string(),
        input: json!("hello"),
        encoding_format: Some("float".to_string()),
        user: Some("user-1".to_string()),
        dimensions: Some(128),
        rid: Some("rid-123".to_string()),
        other: Default::default(),
    };

    let serialized = to_string(&req).unwrap();
    let deserialized: EmbeddingRequest = from_str(&serialized).unwrap();

    assert_eq!(deserialized.model, req.model);
    assert_eq!(deserialized.input, req.input);
    assert_eq!(deserialized.encoding_format, req.encoding_format);
    assert_eq!(deserialized.user, req.user);
    assert_eq!(deserialized.dimensions, req.dimensions);
    assert_eq!(deserialized.rid, req.rid);
}

#[test]
fn test_embedding_request_serialization_array_input() {
    let req = EmbeddingRequest {
        model: "test-emb".to_string(),
        input: json!(["a", "b", "c"]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
        other: Default::default(),
    };

    let serialized = to_string(&req).unwrap();
    let de: EmbeddingRequest = from_str(&serialized).unwrap();
    assert_eq!(de.model, req.model);
    assert_eq!(de.input, req.input);
}

#[test]
fn test_embedding_generation_request_trait_string() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!("hello"),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
        other: Default::default(),
    };
    assert!(!req.is_stream());
    assert_eq!(req.get_model(), Some("emb-model"));
    assert_eq!(req.extract_text_for_routing(), "hello");
}

#[test]
fn test_embedding_generation_request_trait_array() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!(["hello", "world"]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
        other: Default::default(),
    };
    assert_eq!(req.extract_text_for_routing(), "hello world");
}

#[test]
fn test_embedding_generation_request_trait_non_text() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!({"tokens": [1, 2, 3]}),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
        other: Default::default(),
    };
    assert_eq!(req.extract_text_for_routing(), "");
}

#[test]
fn test_embedding_generation_request_trait_mixed_array_ignores_nested() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!(["a", ["b", "c"], 123, {"k": "v"}]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
        other: Default::default(),
    };
    // Only top-level string elements are extracted
    assert_eq!(req.extract_text_for_routing(), "a");
}

/// A key the type does not model rides in `other` and is written back, so the
/// relay forwards the body as the client wrote it; the typed fields stay out of
/// the map.
#[test]
fn test_embedding_request_keeps_the_fields_it_does_not_model() {
    let sent = json!({
        "model": "m",
        "input": "hello",
        "dimensions": 4,
        "truncate_prompt_tokens": 10,
        "add_special_tokens": false,
        "normalize": false,
        "extra_body": {"k": 1}
    });
    let req: EmbeddingRequest = serde_json::from_value(sent.clone()).unwrap();
    assert_eq!(req.dimensions, Some(4));
    assert_eq!(req.other.len(), 4, "{:?}", req.other);
    assert_eq!(req.other["truncate_prompt_tokens"], json!(10));
    assert_eq!(serde_json::to_value(&req).unwrap(), sent);
}

/// The classify request keeps its unmodelled fields the same way.
#[test]
fn test_classify_request_keeps_the_fields_it_does_not_model() {
    let sent =
        json!({"model": "m", "input": ["a", "b"], "priority": 3, "truncate_prompt_tokens": 5});
    let req: ClassifyRequest = serde_json::from_value(sent.clone()).unwrap();
    assert_eq!(req.priority, Some(3));
    assert_eq!(req.other.len(), 1, "{:?}", req.other);
    assert_eq!(serde_json::to_value(&req).unwrap(), sent);
}

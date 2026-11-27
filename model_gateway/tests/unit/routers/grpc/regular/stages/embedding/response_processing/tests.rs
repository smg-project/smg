use openai_protocol::embedding::EmbeddingRequest;
use serde_json::json;
use smg_grpc_client::vllm_proto as vllm;

use super::{super::request_building::asked_for_base64, *};

fn request(value: serde_json::Value) -> EmbeddingRequest {
    serde_json::from_value(value).expect("an embeddings request")
}

fn answer(embedding: Vec<f32>, prompt_tokens: u32) -> ProtoEmbedComplete {
    ProtoEmbedComplete::Vllm(vllm::EmbedResponse {
        embedding_dim: embedding.len() as u32,
        embedding,
        prompt_tokens,
    })
}

/// A batch's answers come out as one list, each vector at its input's
/// index, the usage summed over the inputs.
#[test]
fn each_input_gets_its_vector_at_its_index_and_the_usage_sums() {
    let response = embedding_response(
        "m".to_owned(),
        &[
            answer(vec![0.1, 0.2], 3),
            answer(vec![0.3, 0.4], 5),
            answer(vec![0.5, 0.6], 1),
        ],
        false,
    );
    assert_eq!(response.object, "list");
    assert_eq!(response.model, "m");
    let indexed: Vec<(u32, Embedding)> = response
        .data
        .iter()
        .map(|item| (item.index, item.embedding.clone()))
        .collect();
    assert_eq!(
        indexed,
        vec![
            (0, Embedding::Floats(vec![0.1, 0.2])),
            (1, Embedding::Floats(vec![0.3, 0.4])),
            (2, Embedding::Floats(vec![0.5, 0.6]))
        ]
    );
    assert!(response.data.iter().all(|item| item.object == "embedding"));
    assert_eq!(response.usage.prompt_tokens, 9);
    assert_eq!(response.usage.total_tokens, 9);
    assert_eq!(response.usage.completion_tokens, 0);
}

/// One input is one vector at index 0, as it always was.
#[test]
fn a_single_input_is_one_vector_at_index_zero() {
    let response = embedding_response("m".to_owned(), &[answer(vec![1.0], 4)], false);
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].index, 0);
    assert_eq!(response.data[0].embedding, Embedding::Floats(vec![1.0]));
    assert_eq!(response.usage.prompt_tokens, 4);
}

/// A request that asked for `encoding_format: "base64"` gets each vector
/// as the base64 of its little-endian f32 bytes (four values of 0.5 are
/// `AAAAPwAAAD8AAAA/AAAAPw==`), at its index, the usage as before.
#[test]
fn a_base64_request_gets_the_little_endian_bytes_encoded() {
    let asked = request(json!({"model": "m", "input": ["a", "b"], "encoding_format": "base64"}));
    let response = embedding_response(
        "m".to_owned(),
        &[
            answer(vec![0.5, 0.5, 0.5, 0.5], 3),
            answer(vec![1.0, -2.0], 2),
        ],
        asked_for_base64(&asked),
    );
    assert_eq!(
        response.data[0].embedding,
        Embedding::Base64("AAAAPwAAAD8AAAA/AAAAPw==".to_string())
    );
    assert_eq!(
        response.data[1].embedding,
        Embedding::Base64(
            BASE64_STANDARD.encode([1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat())
        )
    );
    assert_eq!(response.data[1].index, 1);
    assert_eq!(response.usage.prompt_tokens, 5);
}

/// `float`, no format, and a value the public API does not name all keep
/// the float array, byte for byte as before.
#[test]
fn a_float_absent_or_unknown_format_keeps_the_float_array() {
    for body in [
        json!({"model": "m", "input": "a", "encoding_format": "float"}),
        json!({"model": "m", "input": "a"}),
        json!({"model": "m", "input": "a", "encoding_format": "hex"}),
    ] {
        let response = embedding_response(
            "m".to_owned(),
            &[answer(vec![0.5, 0.25], 1)],
            asked_for_base64(&request(body.clone())),
        );
        assert_eq!(
            response.data[0].embedding,
            Embedding::Floats(vec![0.5, 0.25]),
            "{body}"
        );
    }
}

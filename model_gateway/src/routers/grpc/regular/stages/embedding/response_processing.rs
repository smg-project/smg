//! Response processing stage for embedding requests: one vector per input,
//! in the request's order and in the encoding it asked for, under the
//! request's summed usage.

use async_trait::async_trait;
use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    common::UsageInfo,
    embedding::{Embedding, EmbeddingObject, EmbeddingResponse},
};
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::{wrong_response_spec, DispatchContext, ExecutionResult, FinalResponse},
        proto_wrapper::ProtoEmbedComplete,
        spec::{EmbeddingResponseSpec, ResponseSpec},
    },
};

/// Response processing stage for embedding requests
pub(crate) struct EmbeddingResponseProcessingStage;

impl EmbeddingResponseProcessingStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProcessStage for EmbeddingResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Embedding(EmbeddingResponseSpec { base64 }) = spec else {
            return Err(wrong_response_spec(
                "EmbeddingResponseProcessingStage::process",
            ));
        };

        // Extract execution result
        let execution_result = ctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::process",
                "Execution result missing"
            );
            error::internal_error("execution_result_missing", "Execution result missing")
        })?;

        // Expect Embedding result variant
        let ExecutionResult::Embedding { responses } = execution_result else {
            error!(
                function = "EmbeddingResponseProcessingStage::process",
                "Invalid execution result: expected Embedding"
            );
            return Err(error::internal_error(
                "invalid_execution_result",
                "Expected Embedding result",
            ));
        };

        let dispatch = ctx.dispatch.take().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::convert_response",
                "Dispatch metadata missing in context"
            );
            error::internal_error("dispatch_missing", "Dispatch metadata missing")
        })?;

        // Store in context for pipeline to extract; the encoding is the spec's:
        // the engine's wire carries the floats whatever the request said, so
        // the gateway encodes here.
        ctx.response.final_response = Some(FinalResponse::Embedding(embedding_response(
            dispatch.model,
            &responses,
            base64,
        )));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingResponseProcessing"
    }
}

/// The public API's `base64` encoding of a vector: the base64 of its
/// little-endian f32 bytes, four per value, in order.
fn base64_of(vector: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

/// The response to the request: the engine's answers as the list of
/// vectors, `data[i]` the vector of input `i`, each the float array or, when
/// `base64`, the base64 of its little-endian bytes; the usage their prompt
/// tokens summed (an embedding has no completion tokens).
fn embedding_response(
    model: String,
    responses: &[ProtoEmbedComplete],
    base64: bool,
) -> EmbeddingResponse {
    let mut prompt_tokens = 0u32;
    let data = responses
        .iter()
        .enumerate()
        .map(|(index, proto)| {
            prompt_tokens = prompt_tokens.saturating_add(proto.prompt_tokens());
            let embedding = if base64 {
                Embedding::Base64(base64_of(proto.embedding()))
            } else {
                Embedding::Floats(proto.embedding().to_vec())
            };
            EmbeddingObject {
                object: "embedding".to_string(),
                embedding,
                index: index as u32,
            }
        })
        .collect();
    EmbeddingResponse {
        object: "list".to_string(),
        data,
        model,
        usage: UsageInfo {
            prompt_tokens,
            total_tokens: prompt_tokens,
            completion_tokens: 0,
            prompt_tokens_details: None,
            reasoning_tokens: None,
        },
    }
}

#[cfg(test)]
mod tests {
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
        let asked =
            request(json!({"model": "m", "input": ["a", "b"], "encoding_format": "base64"}));
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
}

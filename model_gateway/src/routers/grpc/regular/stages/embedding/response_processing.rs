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
#[path = "../../../../../../tests/unit/routers/grpc/regular/stages/embedding/response_processing/tests.rs"]
mod tests;

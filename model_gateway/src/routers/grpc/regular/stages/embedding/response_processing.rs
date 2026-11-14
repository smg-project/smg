//! Response processing stage for embedding requests: one vector per input,
//! in the request's order, under the request's summed usage.

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::{
    common::UsageInfo,
    embedding::{EmbeddingObject, EmbeddingResponse},
};
use tracing::error;

use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::{DispatchContext, ExecutionResult, FinalResponse},
        proto_wrapper::ProtoEmbedComplete,
        spec::ResponseSpec,
    },
};

/// Response processing stage for embedding requests
pub(crate) struct EmbeddingResponseProcessingStage;

impl EmbeddingResponseProcessingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingResponseProcessingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ProcessStage for EmbeddingResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        if !matches!(spec, ResponseSpec::Embedding) {
            error!(
                function = "EmbeddingResponseProcessingStage::process",
                "Wrong response spec"
            );
            return Err(error::internal_error(
                "wrong_response_spec",
                "Wrong response spec",
            ));
        }

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

        let dispatch = ctx.dispatch.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingResponseProcessingStage::convert_response",
                "Dispatch metadata missing in context"
            );
            error::internal_error("dispatch_missing", "Dispatch metadata missing")
        })?;

        // Store in context for pipeline to extract
        ctx.response.final_response = Some(FinalResponse::Embedding(embedding_response(
            dispatch.model.clone(),
            &responses,
        )));

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingResponseProcessing"
    }
}

/// The response to the request: the engine's answers as the list of
/// vectors, `data[i]` the vector of input `i`, the usage their prompt tokens
/// summed (an embedding has no completion tokens).
fn embedding_response(model: String, responses: &[ProtoEmbedComplete]) -> EmbeddingResponse {
    let mut prompt_tokens = 0u32;
    let data = responses
        .iter()
        .enumerate()
        .map(|(index, proto)| {
            prompt_tokens = prompt_tokens.saturating_add(proto.prompt_tokens());
            EmbeddingObject {
                object: "embedding".to_string(),
                embedding: proto.embedding().to_vec(),
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
    use smg_grpc_client::vllm_proto as vllm;

    use super::*;

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
        );
        assert_eq!(response.object, "list");
        assert_eq!(response.model, "m");
        let indexed: Vec<(u32, Vec<f32>)> = response
            .data
            .iter()
            .map(|item| (item.index, item.embedding.clone()))
            .collect();
        assert_eq!(
            indexed,
            vec![
                (0, vec![0.1, 0.2]),
                (1, vec![0.3, 0.4]),
                (2, vec![0.5, 0.6])
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
        let response = embedding_response("m".to_owned(), &[answer(vec![1.0], 4)]);
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].index, 0);
        assert_eq!(response.data[0].embedding, vec![1.0]);
        assert_eq!(response.usage.prompt_tokens, 4);
    }
}

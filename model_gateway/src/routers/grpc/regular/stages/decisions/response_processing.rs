use std::collections::HashMap;

use async_trait::async_trait;
use axum::response::Response;
use futures::future::try_join_all;

use super::scoring::finish_decisions;
use crate::routers::{
    error,
    grpc::{
        common::stages::ProcessStage,
        context::{DispatchContext, ExecutionResult, FinalResponse},
        proto_wrapper::{ProtoGenerateComplete, ProtoResponseVariant},
        spec::{DecisionQuestionSpec, ResponseSpec},
        utils::tonic_ext::TonicStatusExt,
    },
};

pub(crate) struct DecisionsResponseProcessingStage;

fn invalid(message: impl Into<String>) -> Response {
    error::bad_gateway("invalid_decisions_response", message)
}

/// Each question is one prefill-only stream. Keep abort-on-drop armed until
/// its complete result has passed validation; sibling streams drop on failure.
async fn collect_score(
    result: ExecutionResult,
    question: &DecisionQuestionSpec,
) -> Result<(Vec<f64>, u64, u64), Response> {
    let labels = &question.label_ids;
    let ExecutionResult::Single { mut stream } = result else {
        return Err(invalid("Expected a single scoring stream per question"));
    };
    let mut complete = None;
    while let Some(response) = stream.next().await {
        let response = response.map_err(|status| {
            status.to_http_error(
                "worker_stream_failed",
                format!("Decisions scoring failed: {}", status.message()),
            )
        })?;
        match response.into_response() {
            ProtoResponseVariant::Complete(value) => {
                if complete.replace(value).is_some() {
                    return Err(invalid("Multiple scoring results for one question"));
                }
            }
            ProtoResponseVariant::Chunk(chunk) => {
                if !chunk.token_ids().is_empty() || chunk.completion_tokens() != 0 {
                    return Err(invalid("Decisions scoring must not generate tokens"));
                }
            }
            ProtoResponseVariant::None => return Err(invalid("Empty scoring response")),
        }
    }
    let Some(ProtoGenerateComplete::Sglang(complete)) = complete else {
        return Err(invalid("Missing SGLang scoring result"));
    };
    if !complete.output_ids.is_empty()
        || complete.completion_tokens != 0
        || complete.reasoning_tokens != 0
    {
        return Err(invalid(
            "Decisions scoring must not generate output or reasoning tokens",
        ));
    }
    if complete.index != 0 || !matches!(complete.finish_reason.as_str(), "stop" | "length") {
        return Err(invalid("Invalid scoring completion status or index"));
    }
    // The scheduler can truncate below the advertised context limit when KV
    // capacity is smaller. Never return a decision scored on a shortened prompt.
    if complete.prompt_tokens as usize != question.input_tokens {
        return Err(invalid(format!("Scoring backend reported {} prompt tokens; expected {}. The prompt may have been truncated", complete.prompt_tokens, question.input_tokens)));
    }
    if complete.cached_tokens > complete.prompt_tokens {
        return Err(invalid(
            "Cached token usage exceeds the decision prompt size",
        ));
    }
    let rows = complete
        .output_logprobs
        .ok_or_else(|| invalid("Missing selected-token logprobs; upgrade the SGLang gRPC servicer to support Decisions"))?
        .token_ids_logprobs;
    let [row] = rows.as_slice() else {
        return Err(invalid("Expected one selected-token logprob row; upgrade the SGLang gRPC servicer if scores are absent"));
    };
    if row.values.len() != labels.len() || row.token_ids.len() != labels.len() {
        return Err(invalid(
            "Selected-token scores do not match the decision candidates",
        ));
    }
    let mut scores = HashMap::with_capacity(labels.len());
    for (&id, &value) in row.token_ids.iter().zip(&row.values) {
        if scores.insert(id, f64::from(value)).is_some() {
            return Err(invalid("Duplicate selected-token score"));
        }
    }
    let ordered = labels
        .iter()
        .map(|id| {
            scores
                .remove(id)
                .ok_or_else(|| invalid("Missing decision candidate score"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stream.mark_completed();
    Ok((
        ordered,
        u64::from(complete.prompt_tokens),
        u64::from(complete.cached_tokens),
    ))
}

#[async_trait]
impl ProcessStage for DecisionsResponseProcessingStage {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response> {
        let ResponseSpec::Decisions(spec) = spec else {
            return Err(error::internal_error(
                "wrong_response_spec",
                "Expected Decisions response spec",
            ));
        };
        let Some(ExecutionResult::Batch { results }) = ctx.response.execution_result.take() else {
            return Err(invalid("Missing decision scoring batch"));
        };
        if results.len() != spec.questions.len() {
            return Err(invalid("Scoring result count does not match the questions"));
        }
        let results = try_join_all(
            results
                .into_iter()
                .zip(&spec.questions)
                .map(|(result, question)| collect_score(result, question)),
        )
        .await?;
        let mut input_tokens = 0u64;
        let mut cached_tokens = 0u64;
        let mut logprobs = Vec::with_capacity(results.len());
        for (scores, tokens, cached) in results {
            input_tokens = input_tokens
                .checked_add(tokens)
                .ok_or_else(|| invalid("Scoring token usage overflow"))?;
            cached_tokens = cached_tokens
                .checked_add(cached)
                .ok_or_else(|| invalid("Scoring cache usage overflow"))?;
            logprobs.push(scores);
        }
        let mut response =
            finish_decisions(&spec.scoring, &ctx.dispatch_model, &logprobs, input_tokens)
                .map_err(invalid)?;
        response.usage.input_tokens_details.cached_tokens = cached_tokens;
        ctx.response.final_response = Some(FinalResponse::Decisions(response));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "DecisionsResponseProcessing"
    }
}

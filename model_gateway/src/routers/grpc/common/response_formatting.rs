//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use std::collections::HashMap;

use openai_protocol::common::Usage;

use crate::routers::grpc::proto_wrapper::{ProtoGenerateComplete, ProtoGenerateStreamChunk};

/// Build usage information from collected gRPC responses
///
/// Sums completion_tokens (and reasoning_tokens, each choice's own
/// independent reasoning trace) across all responses, since each is a
/// distinct generation. `prompt_tokens` and `cached_tokens` take the max
/// instead: chat/generate requests have exactly one prompt shared by every
/// `n>1` choice -- cached_tokens is a property of that same prompt (how much
/// of it hit the KV cache), not of the individual completion -- and each
/// response reports that same full length, so summing would charge it once
/// per choice instead of once per request (Completion's own batched usage
/// computation already does this correctly per-prompt; mirrored here).
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses
        .iter()
        .map(|r| r.prompt_tokens())
        .max()
        .unwrap_or(0);
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens()).sum();
    let total_cached_tokens: u32 = responses
        .iter()
        .map(|r| r.cached_tokens())
        .max()
        .unwrap_or(0);
    let total_reasoning_tokens: u32 = responses.iter().map(|r| r.reasoning_tokens()).sum();

    Usage::from_counts(total_prompt_tokens, total_completion_tokens)
        .with_cached_tokens(total_cached_tokens)
        .with_reasoning_tokens(total_reasoning_tokens)
}

/// Tracks per-index completion token counts across streaming chunks.
///
/// Handles the vLLM vs SGLang difference:
/// - vLLM sends delta token counts per chunk (must accumulate)
/// - SGLang sends cumulative counts in the Complete message
pub(crate) struct CompletionTokenTracker {
    tokens: HashMap<u32, u32>,
}

impl CompletionTokenTracker {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    /// Record tokens from a streaming chunk.
    /// For vLLM, accumulates the chunk's token count.
    /// For SGLang/TRT-LLM, this is a no-op (they report in Complete).
    pub fn record_chunk(&mut self, chunk: &ProtoGenerateStreamChunk) {
        if chunk.is_vllm() {
            *self.tokens.entry(chunk.index()).or_insert(0) += chunk.token_ids().len() as u32;
        }
    }

    /// Record the final count from a Complete message.
    /// For vLLM, preserves the accumulated count.
    /// For SGLang/TRT-LLM, uses the cumulative value from Complete.
    pub fn record_complete(&mut self, complete: &ProtoGenerateComplete) {
        let index = complete.index();
        if complete.is_vllm() {
            // Keep accumulated count; ensure entry exists
            self.tokens.entry(index).or_insert(0);
        } else {
            self.tokens.insert(index, complete.completion_tokens());
        }
    }

    /// Get total completion tokens across all indices
    pub fn total(&self) -> u32 {
        self.tokens.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::tokenspeed_proto as tokenspeed;

    use super::*;

    fn complete(
        prompt_tokens: u32,
        cached_tokens: u32,
        completion_tokens: u32,
    ) -> ProtoGenerateComplete {
        ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn build_usage_charges_the_shared_prompt_once_for_n_greater_than_1() {
        let responses = vec![complete(10, 8, 4), complete(10, 8, 6)];
        let usage = build_usage(&responses);

        assert_eq!(
            usage.prompt_tokens, 10,
            "n>1 choices share one prompt -- must not be multiplied per choice"
        );
        assert_eq!(
            usage.prompt_tokens_details.as_ref().map(|d| d.cached_tokens),
            Some(8),
            "cached_tokens is a property of the shared prompt too -- must not be multiplied per choice"
        );
        assert_eq!(
            usage.completion_tokens, 10,
            "each choice generates its own completion -- must be summed"
        );
    }
}

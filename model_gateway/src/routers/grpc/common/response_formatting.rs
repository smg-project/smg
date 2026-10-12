//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use std::collections::HashMap;

use openai_protocol::common::Usage;

use crate::routers::grpc::proto_wrapper::{ProtoGenerateComplete, ProtoGenerateStreamChunk};

/// The token counters of one request, aggregated over its choices the way
/// the engines report them. The `n>1` choices of a chat/generate request
/// share exactly one prompt, and `cached_tokens` is a property of that same
/// prompt (how much of it hit the KV cache), so `prompt` and `cached` take
/// the max over the choices: summing would charge the prompt once per choice
/// instead of once per request. Each choice is its own generation, so
/// `completion`, `reasoning` (each choice's own reasoning trace) and the
/// speculative counts sum. (Completion's own batched usage computation does
/// this per prompt and sums across prompts.)
///
/// Every path that aggregates the engine's frames itself builds its usage
/// from these counters, so the streamed and the non-streamed answer of a
/// request carry one and the same usage object.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenCounts {
    pub prompt: u32,
    pub completion: u32,
    pub cached: u32,
    pub reasoning: u32,
    pub spec_accepted: u32,
    pub spec_drafted: u32,
}

impl TokenCounts {
    /// The counters of the `Complete` frames of one request.
    pub(crate) fn of(responses: &[ProtoGenerateComplete]) -> Self {
        Self {
            prompt: responses
                .iter()
                .map(|r| r.prompt_tokens())
                .max()
                .unwrap_or(0),
            completion: responses.iter().map(|r| r.completion_tokens()).sum(),
            cached: responses
                .iter()
                .map(|r| r.cached_tokens())
                .max()
                .unwrap_or(0),
            reasoning: responses.iter().map(|r| r.reasoning_tokens()).sum(),
            spec_accepted: responses.iter().map(|r| r.spec_accepted_tokens()).sum(),
            spec_drafted: responses.iter().map(|r| r.spec_draft_tokens()).sum(),
        }
    }

    /// The usage object the counters make: the counts with the cached,
    /// reasoning and speculative details the engine reported. The usage of
    /// a legacy completion, streamed or not, is this object.
    pub(crate) fn usage(self) -> Usage {
        Usage::from_counts(self.prompt, self.completion)
            .with_cached_tokens(self.cached)
            .with_reasoning_tokens(self.reasoning)
            .with_speculative_tokens(self.spec_accepted, self.spec_drafted)
    }
}

/// Build usage information from collected gRPC responses (the
/// [`TokenCounts`] aggregation of their counters).
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    TokenCounts::of(responses).usage()
}

/// The usage of a chat completion from `engine_usage`, the usage the
/// engine's counters make: the rendered prompt tokens the provider does not
/// bill are dropped, the reasoning the gateway's own parser split off is
/// accounted (`parsed_reasoning`, see `with_parsed_reasoning_tokens`), and
/// every detail counter is present as the OpenAI API sends it on every chat
/// completion, zero where nothing was counted. The streamed answer's usage
/// chunk and the non-streamed answer both build their usage here, so a
/// request's two usage objects are the same.
pub(crate) fn chat_usage(
    engine_usage: Usage,
    parsed_reasoning: Option<u32>,
    unbilled_prompt_tokens: u32,
) -> Usage {
    with_parsed_reasoning_tokens(
        engine_usage.with_unbilled_prompt_tokens(unbilled_prompt_tokens),
        parsed_reasoning,
    )
    .with_complete_details()
}

/// Account for the reasoning the gateway's own reasoning parser split off:
/// `parsed` is the number of output tokens it routed to `reasoning_content`
/// for the request, `None` when it did not run. Like an engine with a
/// reasoning parser, the details are then always reported, zero included; an
/// engine that counts reasoning itself is not overruled, whatever its count.
pub(crate) fn with_parsed_reasoning_tokens(mut usage: Usage, parsed: Option<u32>) -> Usage {
    if let Some(parsed) = parsed {
        let details = usage.completion_tokens_details.get_or_insert_default();
        details.reasoning_tokens.get_or_insert(parsed);
    }
    usage
}

/// The version to report on a generate response: what the engine stamped on
/// this very response (the proto accessors already treat an empty string and
/// the engine's `"default"` placeholder as unset) beats the dispatch-time
/// label (kept as registered, so an empty one counts as unset here), which
/// beats the historical `"default"`.
///
/// Borrowed on purpose: the SSE call sites run once per chunk and serialize
/// the value straight into the event, so no per-chunk allocation.
pub(crate) fn effective_weight_version<'a>(
    reported: Option<&'a str>,
    dispatch: Option<&'a str>,
) -> &'a str {
    reported
        .or(dispatch)
        .filter(|v| !v.is_empty())
        .unwrap_or("default")
}

/// Tracks per-index completion token counts across streaming chunks.
///
/// Handles the two chunk conventions (`ChunkSemantics`):
/// - delta streams send per-chunk token counts (must accumulate)
/// - cumulative streams send the count in the Complete message, and the
///   running count in every chunk before it
pub(crate) struct CompletionTokenTracker {
    tokens: HashMap<u32, u32>,
    /// The running count a cumulative stream reported in its latest chunk:
    /// how far it had come, for a stream that ends before its Complete.
    reported: HashMap<u32, u32>,
}

impl CompletionTokenTracker {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            reported: HashMap::new(),
        }
    }

    /// Record tokens from a streaming chunk.
    /// Delta streams accumulate the chunk's token count; cumulative streams
    /// report it in Complete instead, so a chunk of theirs only notes the
    /// running count it carries.
    pub fn record_chunk(&mut self, chunk: &ProtoGenerateStreamChunk) {
        if chunk.chunk_semantics().is_delta() {
            *self.tokens.entry(chunk.index()).or_insert(0) += chunk.token_ids().len() as u32;
        } else {
            self.reported
                .insert(chunk.index(), chunk.completion_tokens());
        }
    }

    /// Record the final count from a Complete message.
    /// Delta streams keep the count accumulated from their chunks; cumulative
    /// streams take the value from Complete.
    pub fn record_complete(&mut self, complete: &ProtoGenerateComplete) {
        let index = complete.index();
        if complete.chunk_semantics().is_delta() {
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

    /// Completion tokens delivered so far across all indices: the count
    /// [`total`](Self::total) knows, else the running count a cumulative
    /// stream last reported. How far a stream that ends before its Complete
    /// had come.
    pub fn delivered(&self) -> u32 {
        let mut per_index = self.reported.clone();
        for (index, tokens) in &self.tokens {
            let known = per_index.entry(*index).or_insert(0);
            *known = (*known).max(*tokens);
        }
        per_index.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::{tokenspeed_proto as tokenspeed, vllm_proto as vllm};

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

    #[test]
    fn parsed_reasoning_tokens_are_reported_once_the_parser_ran() {
        let reasoning_tokens = |usage: Usage| {
            usage
                .completion_tokens_details
                .and_then(|d| d.reasoning_tokens)
        };
        let usage = build_usage(&[complete(10, 8, 4)]);
        assert_eq!(reasoning_tokens(usage.clone()), None);
        assert_eq!(
            reasoning_tokens(with_parsed_reasoning_tokens(usage.clone(), None)),
            None,
            "no reasoning parser: nothing to report"
        );
        assert_eq!(
            reasoning_tokens(with_parsed_reasoning_tokens(usage.clone(), Some(0))),
            Some(0),
            "the parser ran and found no reasoning: zero, not absent"
        );
        assert_eq!(
            reasoning_tokens(with_parsed_reasoning_tokens(usage, Some(7))),
            Some(7)
        );
        let engine_counted = Usage::from_counts(10, 4).with_reasoning_tokens(5);
        assert_eq!(
            reasoning_tokens(with_parsed_reasoning_tokens(engine_counted, Some(3))),
            Some(5),
            "an engine that counts reasoning itself is not overruled"
        );
        let engine_counted_fewer = Usage::from_counts(10, 4).with_reasoning_tokens(3);
        assert_eq!(
            reasoning_tokens(with_parsed_reasoning_tokens(engine_counted_fewer, Some(5))),
            Some(3),
            "nor when its count is lower than the parser's"
        );
    }

    #[test]
    fn token_counts_usage_carries_the_cached_reasoning_and_speculative_details() {
        let usage = TokenCounts {
            prompt: 10,
            completion: 5,
            cached: 4,
            reasoning: 3,
            spec_accepted: 2,
            spec_drafted: 3,
        }
        .usage();

        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(4)
        );
        let details = usage
            .completion_tokens_details
            .expect("the reasoning and speculative details");
        assert_eq!(details.reasoning_tokens, Some(3));
        assert_eq!(details.accepted_prediction_tokens, Some(2));
        assert_eq!(details.rejected_prediction_tokens, Some(1));
    }

    #[test]
    fn chat_usage_has_every_detail_counter_and_no_unbilled_prompt_tokens() {
        let usage = serde_json::to_value(chat_usage(build_usage(&[complete(10, 8, 4)]), None, 3))
            .expect("a JSON usage");
        assert_eq!(
            usage,
            serde_json::json!({
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 7, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            })
        );
        let parsed = chat_usage(build_usage(&[complete(10, 8, 4)]), Some(2), 0);
        assert_eq!(
            parsed
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens),
            Some(2)
        );
    }

    #[test]
    fn build_usage_then_unbilled_charges_the_stub_once_for_n_greater_than_1() {
        let usage =
            build_usage(&[complete(10, 10, 4), complete(10, 10, 6)]).with_unbilled_prompt_tokens(3);
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            Some(7)
        );
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn completion_token_tracker_follows_chunk_semantics() {
        // Delta stream (the vLLM shape, which the ZMQ lane also emits for
        // TokenSpeed workers): the chunks carry the counts.
        let mut tracker = CompletionTokenTracker::new();
        tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
            token_ids: vec![1, 2, 3],
            ..Default::default()
        }));
        tracker.record_complete(&ProtoGenerateComplete::Vllm(Box::new(
            vllm::GenerateComplete {
                completion_tokens: 99,
                ..Default::default()
            },
        )));
        assert_eq!(
            tracker.total(),
            3,
            "delta stream keeps the accumulated count"
        );

        // Cumulative stream: chunks are not accumulated, Complete is the count.
        let mut tracker = CompletionTokenTracker::new();
        tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
            tokenspeed::GenerateStreamChunk {
                token_ids: vec![1, 2, 3],
                ..Default::default()
            },
        ));
        tracker.record_complete(&complete(0, 0, 7));
        assert_eq!(
            tracker.total(),
            7,
            "cumulative stream reports it in Complete"
        );
    }

    #[test]
    fn delivered_tokens_follow_a_stream_that_ends_before_complete() {
        // Cumulative stream: a chunk carries the running count; the total is
        // unknown until Complete, the delivered count is not.
        let mut tracker = CompletionTokenTracker::new();
        tracker.record_chunk(&ProtoGenerateStreamChunk::TokenSpeed(
            tokenspeed::GenerateStreamChunk {
                token_ids: vec![5],
                completion_tokens: 41,
                ..Default::default()
            },
        ));
        assert_eq!(tracker.total(), 0, "no authoritative count before Complete");
        assert_eq!(
            tracker.delivered(),
            41,
            "the running count the engine reported"
        );
        tracker.record_complete(&complete(0, 0, 42));
        assert_eq!((tracker.total(), tracker.delivered()), (42, 42));

        // Delta stream: the accumulated chunks are both.
        let mut tracker = CompletionTokenTracker::new();
        tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk {
            token_ids: vec![1, 2, 3],
            ..Default::default()
        }));
        assert_eq!((tracker.total(), tracker.delivered()), (3, 3));
    }

    #[test]
    fn engine_reported_version_beats_dispatch_which_beats_default() {
        assert_eq!(effective_weight_version(Some("v7"), Some("v3")), "v7");
        assert_eq!(effective_weight_version(None, Some("v3")), "v3");
        assert_eq!(effective_weight_version(None, None), "default");
        assert_eq!(
            effective_weight_version(None, Some("")),
            "default",
            "an empty registration label is unset"
        );
    }
}

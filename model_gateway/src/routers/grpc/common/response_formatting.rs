//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use openai_protocol::{common::Usage, generate::GenerateResponse};

use crate::{
    rate_limit::UsageSettlement,
    routers::grpc::proto_wrapper::{ProtoGenerateComplete, ProtoGenerateStreamChunk},
};

/// The token counters of one request, aggregated over its choices the way
/// the engines report them. The `n>1` choices of a chat/generate request
/// share exactly one prompt, and `cached_tokens` is a property of that same
/// prompt (how much of it hit the KV cache), so `prompt` and `cached` take
/// the max over the choices: summing would charge the prompt once per choice
/// instead of once per request. Each choice is its own generation, so
/// `completion`, `reasoning` (each choice's own reasoning trace) and the
/// accepted counts sum; drafted counts sum only if all choices supply
/// consistent positive proposals (otherwise unavailable). (Completion's own batched usage computation does
/// this per prompt and sums across prompts.)
///
/// Every path that aggregates the engine's frames itself builds its usage
/// from these counters, so the streamed and the non-streamed answer of a
/// request carry one and the same usage object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenCounts {
    pub prompt: u32,
    pub completion: u32,
    pub cached: u32,
    pub reasoning: u32,
    pub spec_accepted: u32,
    pub spec_drafted: Option<u32>,
}

impl Default for TokenCounts {
    fn default() -> Self {
        Self {
            prompt: 0,
            completion: 0,
            cached: 0,
            reasoning: 0,
            spec_accepted: 0,
            spec_drafted: Some(0),
        }
    }
}

/// Sum proposals only when every choice supplied a consistent positive count.
/// Scalar proto zeros have no presence: all-zero legacy usage retains schema
/// defaults, but a zero mixed with measured speculation makes rejections unknown.
pub(crate) fn known_spec_drafted(counts: impl Iterator<Item = (u32, u32)>) -> Option<u32> {
    let mut total = 0;
    let mut has_unknown = false;
    for (accepted, drafted) in counts {
        if drafted < accepted {
            return None;
        }
        has_unknown |= drafted == 0;
        total += drafted;
    }
    if has_unknown && total > 0 {
        None
    } else {
        Some(total)
    }
}

/// Every requested choice must have its own terminal frame. Arrival count
/// alone cannot distinguish a missing choice from a repeated terminal index.
/// Some legacy workers send only index zero for n>1; those frames cannot prove
/// complete membership either, so their rejections remain unavailable.
pub(crate) fn all_choices_completed(
    responses: &[ProtoGenerateComplete],
    expected_choices: u32,
) -> bool {
    responses.len() >= expected_choices as usize
        && (0..expected_choices).all(|index| responses.iter().any(|r| r.index() == index))
}

impl TokenCounts {
    /// Collected counters retain partial answers, but only certify rejections
    /// once every choice requested by the client supplied terminal counters.
    pub(crate) fn of_collected(responses: &[ProtoGenerateComplete], expected_choices: u32) -> Self {
        let mut counts = Self::of(responses);
        if !all_choices_completed(responses, expected_choices) {
            counts.spec_drafted = None;
        }
        counts
    }

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
            spec_drafted: known_spec_drafted(
                responses
                    .iter()
                    .map(|r| (r.spec_accepted_tokens(), r.spec_draft_tokens())),
            ),
        }
    }

    /// The counters of the generate answers of one request, one per choice,
    /// as [`Self::of`] takes them from the frames: the shared prompt's count
    /// and cache hit once, the greatest any choice reports (the choice that
    /// prefilled the shared prefix may report fewer hits than its siblings),
    /// the choices' own completion and reasoning counts summed. A generate
    /// answer carries no speculative counts.
    pub(crate) fn of_generated(responses: &[GenerateResponse]) -> Self {
        Self {
            prompt: responses
                .iter()
                .map(|r| r.meta_info.prompt_tokens)
                .max()
                .unwrap_or(0),
            completion: responses
                .iter()
                .map(|r| r.meta_info.completion_tokens)
                .sum(),
            cached: responses
                .iter()
                .map(|r| r.meta_info.cached_tokens)
                .max()
                .unwrap_or(0),
            reasoning: responses
                .iter()
                .map(|r| r.meta_info.reasoning_tokens.unwrap_or(0))
                .sum(),
            spec_accepted: 0,
            spec_drafted: Some(0),
        }
    }

    /// The engine's reasoning count as the usage and the settlement take it:
    /// no higher than the completion count. A speculative step that lands
    /// past the token limit can be counted whole as reasoning while the
    /// completion count stops at the limit, and a part above the whole is
    /// no count a client can meter by. Each choice's count arrives here
    /// already within its own completion count: the frame's where it is
    /// read, the ids a generate stream delivered where that stream keeps
    /// it. This bound is the guard on the request's sum.
    fn reasoning_within_completion(self) -> u32 {
        self.reasoning.min(self.completion)
    }

    /// The usage object the counters make: the counts with the cached,
    /// reasoning and speculative details the engine reported, the reasoning
    /// count as `reasoning_within_completion` reads it. The usage of
    /// a legacy completion, streamed or not, is this object.
    pub(crate) fn usage(self) -> Usage {
        let mut usage = Usage::from_counts(self.prompt, self.completion)
            .with_cached_tokens(self.cached)
            .with_reasoning_tokens(self.reasoning_within_completion())
            .with_speculative_tokens(self.spec_accepted, self.spec_drafted.unwrap_or(0));
        if self.spec_drafted.is_none() {
            // Keep aggregate availability separate from the old proto's zero sentinel.
            // Explicit accepted zero must also preserve unknown rejected in chat usage.
            let details = usage.completion_tokens_details.get_or_insert_default();
            details.accepted_prediction_tokens = Some(self.spec_accepted);
            details.rejected_prediction_tokens = None;
        }
        usage
    }

    /// The settlement of the request's reservation: the engine's counts as
    /// they are, except a reasoning count above the completion count
    /// (`reasoning_within_completion`), before the client-facing usage
    /// drops unbilled prompt tokens or substitutes the gateway parser's
    /// reasoning split.
    pub(crate) fn settlement(self) -> UsageSettlement {
        UsageSettlement {
            actual_input_tokens: self.prompt,
            completion_tokens: self.completion,
            cached_input_tokens: self.cached,
            reasoning_tokens: self.reasoning_within_completion(),
        }
    }
}

/// The usage of a chat completion from `engine_usage`, the usage the
/// engine's counters make: the rendered prompt tokens the provider does not
/// bill are dropped, the reasoning the gateway's own parser split off is
/// accounted (`parsed_reasoning`, see `with_parsed_reasoning_tokens`), the
/// image tokens the gateway's own expansion wrote into the prompt are
/// reported (`image_tokens`, `None` for a request without one), and default
/// counters are present, except unavailable speculative rejections remain
/// absent even when accepted is zero. The streamed answer's usage chunk and
/// the non-streamed answer both build their usage here, so a request's two
/// usage objects are the same.
pub(crate) fn chat_usage(
    engine_usage: Usage,
    parsed_reasoning: Option<u32>,
    unbilled_prompt_tokens: u32,
    image_tokens: Option<u32>,
) -> Usage {
    with_parsed_reasoning_tokens(
        engine_usage.with_unbilled_prompt_tokens(unbilled_prompt_tokens),
        parsed_reasoning,
    )
    .with_image_tokens(image_tokens)
    .with_complete_details()
}

/// Account for the reasoning the gateway's own reasoning parser split off:
/// `parsed` is the number of output tokens it routed to `reasoning_content`
/// for the request, `None` when it did not run. Like an engine with a
/// reasoning parser, the details are then always reported, zero included; an
/// engine that counts reasoning itself is not overruled: its count stands as
/// [`TokenCounts::usage`] reports it. The parser's count is inserted under
/// the same bound, no higher than the completion count.
pub(crate) fn with_parsed_reasoning_tokens(mut usage: Usage, parsed: Option<u32>) -> Usage {
    if let Some(parsed) = parsed {
        let whole = usage.completion_tokens;
        let details = usage.completion_tokens_details.get_or_insert_default();
        details
            .reasoning_tokens
            .get_or_insert_with(|| parsed.min(whole));
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
    /// One entry per choice index, grown as the indices come (the engines
    /// send them dense from zero): the count known for the choice, and the
    /// running count a cumulative stream reported in its latest chunk (how
    /// far it had come, for a stream that ends before its Complete).
    choices: Vec<TrackedChoice>,
}

#[derive(Clone, Copy, Default)]
struct TrackedChoice {
    tokens: u32,
    reported: u32,
}

/// Choice indices past this many are a malformed engine message, not a
/// vector to grow: their counts are dropped.
const MAX_TRACKED_CHOICES: usize = 4096;

impl CompletionTokenTracker {
    pub fn new() -> Self {
        Self {
            choices: Vec::new(),
        }
    }

    /// The entry of choice `index`, made when first seen; `None` for an
    /// index past [`MAX_TRACKED_CHOICES`].
    fn choice(&mut self, index: u32) -> Option<&mut TrackedChoice> {
        let index = index as usize;
        if index >= MAX_TRACKED_CHOICES {
            return None;
        }
        if self.choices.len() <= index {
            self.choices.resize(index + 1, TrackedChoice::default());
        }
        self.choices.get_mut(index)
    }

    /// Record tokens from a streaming chunk.
    /// Delta streams accumulate the chunk's token count; cumulative streams
    /// report it in Complete instead, so a chunk of theirs only notes the
    /// running count it carries.
    pub fn record_chunk(&mut self, chunk: &ProtoGenerateStreamChunk) {
        let Some(choice) = self.choice(chunk.index()) else {
            return;
        };
        if chunk.chunk_semantics().is_delta() {
            choice.tokens += chunk.token_ids().len() as u32;
        } else {
            choice.reported = chunk.completion_tokens();
        }
    }

    /// Record the final count from a Complete message.
    /// Delta streams keep the count accumulated from their chunks; cumulative
    /// streams take the value from Complete.
    pub fn record_complete(&mut self, complete: &ProtoGenerateComplete) {
        let Some(choice) = self.choice(complete.index()) else {
            return;
        };
        if !complete.chunk_semantics().is_delta() {
            choice.tokens = complete.completion_tokens();
        }
    }

    /// Settle a choice the gateway's own stop ended, as its Complete would
    /// have: a delta stream keeps the count it accumulated, a cumulative
    /// stream takes the running count its latest chunk reported. The choice
    /// is counted through the step in which the stop matched, as the engine's
    /// own server counts a stopped request. An engine that ends the choice on
    /// that same step still sends its Complete, recorded as usual (the same
    /// step, in the engine's own count); the Complete of a choice the engine
    /// generated past the stop carries the tail's count, and the stream does
    /// not pass it on.
    pub fn record_local_stop(&mut self, index: u32) {
        if let Some(choice) = self.choice(index) {
            choice.tokens = choice.tokens.max(choice.reported);
        }
    }

    /// Get total completion tokens across all indices
    pub fn total(&self) -> u32 {
        self.choices.iter().map(|choice| choice.tokens).sum()
    }

    /// Completion tokens delivered so far across all indices: the count
    /// [`total`](Self::total) knows, else the running count a cumulative
    /// stream last reported. How far a stream that ends before its Complete
    /// had come.
    pub fn delivered(&self) -> u32 {
        self.choices
            .iter()
            .map(|choice| choice.tokens.max(choice.reported))
            .sum()
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/common/response_formatting/tests.rs"]
mod tests;

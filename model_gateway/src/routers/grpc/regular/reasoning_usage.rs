//! Final reasoning usage for vLLM's token-only protocol. SGLang already reports
//! its own count; Harmony has its own channel parser. Neither uses this counter.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Weak},
};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::Usage;
use reasoning_parser::ParserFactory;

use crate::routers::grpc::{
    proto_wrapper::{ProtoGenerateComplete, ProtoGenerateStreamChunk},
    spec::ChatResponseSpec,
    utils,
};

#[cfg(test)]
pub(super) mod tests;

/// Only tokenizer-dependent marker validation is cached. Request thinking state
/// and parser selection are resolved afresh, including on cache hits.
struct Markers {
    start: u32,
    end: u32,
    eos: Vec<u32>,
}

impl Markers {
    fn resolve(tokenizer: &dyn Tokenizer, start: &str, end: &str) -> Option<Arc<Self>> {
        let marker = |text: &str| {
            let id = tokenizer.token_to_id(text)?;
            let encoded = tokenizer.encode(text, false).ok()?;
            (encoded.token_ids() == [id] && tokenizer.decode(&[id], false).ok()? == text)
                .then_some(id)
        };
        let start = marker(start)?;
        let end = marker(end)?;
        if start == end {
            return None;
        }
        let mut eos = tokenizer.eos_token_ids().to_vec();
        if let Some(id) = tokenizer
            .get_special_tokens()
            .eos_token
            .as_deref()
            .and_then(|text| tokenizer.token_to_id(text))
        {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
        Some(Arc::new(Self { start, end, eos }))
    }
}

struct MarkerCache {
    tokenizer: Weak<dyn Tokenizer>,
    start: String,
    end: String,
    markers: Option<Arc<Markers>>,
}

thread_local! {
    // Bounded to the last tokenizer/marker pair per executor thread. A Weak
    // keeps its allocation identity valid without retaining the tokenizer.
    static MARKERS: RefCell<Option<MarkerCache>> = const { RefCell::new(None) };
}

/// Resolved once per request, independent of the text parser's mutable state.
pub(super) struct Counter {
    markers: Arc<Markers>,
    starts_in_reasoning: bool,
}

impl Counter {
    pub(super) fn new(
        factory: &ParserFactory,
        name: Option<&str>,
        model: &str,
        tokenizer: &Arc<dyn Tokenizer>,
        request: &ChatResponseSpec,
    ) -> Option<Self> {
        // Continuations do not carry the actual assistant-prefix parser state.
        if request.continues_final_assistant {
            return None;
        }
        let parser = utils::create_reasoning_parser(factory, name, model)?;
        let (start, end) = parser.token_counting_markers()?;
        let markers = MARKERS.with(|cache| {
            let hit = cache
                .borrow()
                .as_ref()
                .filter(|entry| {
                    std::ptr::eq(entry.tokenizer.as_ptr(), Arc::as_ptr(tokenizer))
                        && entry.start == start
                        && entry.end == end
                })
                .map(|entry| entry.markers.clone());
            if let Some(markers) = hit {
                return markers;
            }
            // Do not hold the RefCell borrow across tokenizer calls.
            let markers = Markers::resolve(tokenizer.as_ref(), start, end);
            *cache.borrow_mut() = Some(MarkerCache {
                tokenizer: Arc::downgrade(tokenizer),
                start: start.to_owned(),
                end: end.to_owned(),
                markers: markers.clone(),
            });
            markers
        })?;
        Some(Self {
            markers,
            starts_in_reasoning: parser.is_in_reasoning()
                || request.reasoning_starts_in_prefill(tokenizer.as_ref()),
        })
    }

    fn count(&self, ids: &[u32]) -> Option<u32> {
        let mut state = self.initial_state();
        self.feed(&mut state, ids)?;
        Some(state.reasoning)
    }

    fn initial_state(&self) -> StreamCount {
        StreamCount {
            inside: Some(self.starts_in_reasoning),
            completion: 0,
            reasoning: 0,
        }
    }

    fn feed(&self, state: &mut StreamCount, ids: &[u32]) -> Option<()> {
        state.completion = state
            .completion
            .checked_add(u32::try_from(ids.len()).ok()?)?;
        // None means the first reasoning span has closed; never reopen it.
        let Some(mut inside) = state.inside else {
            return Some(());
        };
        for &id in ids {
            if id == self.markers.end {
                // A stray closer before the span is model-specific framing.
                // Leave it to the existing parser instead of claiming an exact zero.
                if !inside {
                    return None;
                }
                state.inside = None;
                return Some(());
            }
            if id == self.markers.start {
                // A start marker generated after a prompt-opened span is not
                // another nesting level (the legacy parser strips it too).
                inside = true;
            } else if inside && !self.markers.eos.contains(&id) {
                state.reasoning = state.reasoning.checked_add(1)?;
            }
        }
        state.inside = Some(inside);
        Some(())
    }
}

struct StreamCount {
    inside: Option<bool>,
    completion: u32,
    reasoning: u32,
}

#[derive(Default)]
pub(super) struct VllmReasoningUsage {
    // Outer None: not initialized. Inner None: counting is unsupported.
    counter: Option<Option<Counter>>,
    // Constant space per streamed choice; None means accounting is ambiguous or overflowed.
    streams: HashMap<u32, Option<StreamCount>>,
    // (completion, reasoning) from each authoritative Complete.
    choices: HashMap<u32, Option<(u32, u32)>>,
}

impl VllmReasoningUsage {
    pub(super) fn record_chunk(
        &mut self,
        chunk: &ProtoGenerateStreamChunk,
        init: impl FnOnce() -> Option<Counter>,
    ) {
        let ProtoGenerateStreamChunk::Vllm(chunk) = chunk else {
            return;
        };
        if chunk.token_ids.is_empty() {
            return;
        }
        let Some(counter) = self.counter.get_or_insert_with(init) else {
            return;
        };
        let state = self
            .streams
            .entry(chunk.index)
            .or_insert_with(|| Some(counter.initial_state()));
        if let Some(count) = state {
            if counter.feed(count, &chunk.token_ids).is_none() {
                *state = None;
            }
        }
    }

    pub(super) fn record(
        &mut self,
        complete: &ProtoGenerateComplete,
        init: impl FnOnce() -> Option<Counter>,
    ) {
        let ProtoGenerateComplete::Vllm(complete) = complete else {
            return;
        };
        let counter = self.counter.get_or_insert_with(init);
        let count = counter.as_ref().and_then(|counter| {
            if !matches!(complete.finish_reason.as_str(), "stop" | "length") {
                return None;
            }
            // Python DELTA completions repeat the last chunk (or are empty),
            // while Rust returns all IDs. Both have already sent their chunks.
            if let Some(state) = self.streams.get(&complete.index) {
                return state
                    .as_ref()
                    .map(|state| (state.completion, state.reasoning));
            }
            // Non-streaming requires a complete token sequence.
            if complete.output_ids.len() != complete.completion_tokens as usize {
                return None;
            }
            counter
                .count(&complete.output_ids)
                .map(|reasoning| (complete.completion_tokens, reasoning))
        });
        self.choices.insert(complete.index, count);
    }

    /// Available only after every choice completed and the completion total
    /// agrees with usage. Callers may skip parser accounting only then.
    pub(super) fn exact_total(&self, completion_tokens: u32, expected_choices: u32) -> Option<u32> {
        if self.choices.is_empty() || self.choices.len() != expected_choices as usize {
            return None;
        }
        let (completion, reasoning) =
            self.choices
                .values()
                .try_fold((0u32, 0u32), |(completion, reasoning), count| {
                    let (c, r) = (*count)?;
                    Some((completion.checked_add(c)?, reasoning.checked_add(r)?))
                })?;
        (completion == completion_tokens).then_some(reasoning)
    }

    pub(super) fn apply(&self, usage: &mut Usage, expected_choices: u32) {
        // Do not mix full-output reasoning with a local-stop-trimmed count.
        if let Some(total) = self.exact_total(usage.completion_tokens, expected_choices) {
            usage
                .completion_tokens_details
                .get_or_insert_default()
                .reasoning_tokens = Some(total);
        }
    }
}

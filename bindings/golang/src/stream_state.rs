//! Stream state tracking for per-index data

use std::collections::HashMap;

use llm_tokenizer::{stream::DecodeStream, StopSequenceDecoder};

/// Per-index state for streaming responses
#[derive(Default)]
pub struct StreamIndexState {
    pub stop_decoder: Option<StopSequenceDecoder>,
    pub decode_stream: Option<DecodeStream>,
    pub has_tool_calls: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Manager for per-index stream state
#[derive(Default)]
pub struct StreamStateManager {
    states: HashMap<u32, StreamIndexState>,
}

impl StreamStateManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create state for an index
    pub fn get_or_create(&mut self, index: u32) -> &mut StreamIndexState {
        self.states.entry(index).or_default()
    }

    /// Remove and return state for an index
    pub fn remove(&mut self, index: u32) -> Option<StreamIndexState> {
        self.states.remove(&index)
    }
}

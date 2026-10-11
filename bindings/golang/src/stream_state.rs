//! Stream state tracking for per-index data

use std::collections::HashMap;

use llm_tokenizer::{stream::DecodeStream, StopSequenceDecoder};

/// Per-index state for streaming responses
pub struct StreamIndexState {
    pub stop_decoder: Option<StopSequenceDecoder>,
    pub decode_stream: Option<DecodeStream>,
    pub has_tool_calls: bool,
    pub is_first_chunk: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl Default for StreamIndexState {
    fn default() -> Self {
        Self {
            stop_decoder: None,
            decode_stream: None,
            has_tool_calls: false,
            is_first_chunk: true, // New streams start with first chunk
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }
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

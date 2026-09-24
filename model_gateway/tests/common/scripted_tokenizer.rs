use std::sync::atomic::{AtomicBool, Ordering};

use llm_tokenizer::{
    chat_template::ChatTemplateParams, traits::Tokenizer, Decoder, Encoder, Encoding,
    MockTokenizer, SpecialTokens,
};

/// Decode the canned worker's generated tokens starting at 100 as chosen chunks.
pub struct ScriptedTokenizer {
    base: MockTokenizer,
    chunks: Vec<String>,
    final_answer: Option<(usize, String)>,
    use_final_answer: AtomicBool,
}

impl ScriptedTokenizer {
    pub fn new(output: &str) -> Self {
        Self::from_chunks(vec![output.to_string()])
    }

    /// Keep model chunks separate so a streamed batch exercises every tool call.
    pub fn from_chunks(chunks: Vec<String>) -> Self {
        Self {
            base: MockTokenizer::new(),
            chunks,
            final_answer: None,
            use_final_answer: AtomicBool::new(false),
        }
    }

    /// The canned worker reuses token ids on each request. Select its decoded
    /// answer from the actual tool-result history sent back by the router.
    pub fn with_final_answer(mut self, tool_results: usize, answer: &str) -> Self {
        self.final_answer = Some((tool_results, answer.to_string()));
        self
    }
}

impl Encoder for ScriptedTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> anyhow::Result<Encoding> {
        self.base.encode(text, add_special)
    }
    fn encode_batch(&self, texts: &[&str], add_special: bool) -> anyhow::Result<Vec<Encoding>> {
        self.base.encode_batch(texts, add_special)
    }
}

impl Decoder for ScriptedTokenizer {
    fn decode(&self, ids: &[u32], _skip_special: bool) -> anyhow::Result<String> {
        if self.use_final_answer.load(Ordering::Relaxed) {
            return Ok(self
                .final_answer
                .as_ref()
                .filter(|_| ids.contains(&100))
                .map(|(_, answer)| answer.clone())
                .unwrap_or_default());
        }
        Ok(ids
            .iter()
            .filter_map(|id| {
                id.checked_sub(100)
                    .and_then(|index| self.chunks.get(index as usize))
            })
            .cloned()
            .collect())
    }
}

impl Tokenizer for ScriptedTokenizer {
    fn vocab_size(&self) -> usize {
        self.base.vocab_size() + self.chunks.len()
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        self.base.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.chunks
            .iter()
            .position(|chunk| chunk == token)
            .map(|index| 100 + index as u32)
            .or_else(|| self.base.token_to_id(token))
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        id.checked_sub(100)
            .and_then(|index| self.chunks.get(index as usize))
            .cloned()
            .or_else(|| self.base.id_to_token(id))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> anyhow::Result<String> {
        if let Some((limit, _)) = &self.final_answer {
            let results = messages
                .iter()
                .filter(|message| message["role"] == "tool")
                .count();
            self.use_final_answer
                .store(results >= *limit, Ordering::Relaxed);
        }
        self.base.apply_chat_template(messages, params)
    }
}

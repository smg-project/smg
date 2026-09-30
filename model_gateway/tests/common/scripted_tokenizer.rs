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
    /// Chunk ids that decoding with `skip_special` drops.
    special_ids: Vec<u32>,
    eos_ids: Vec<u32>,
    response_template: Option<serde_json::Value>,
    /// Appended to the rendered prompt.
    prompt_tail: String,
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
            special_ids: Vec::new(),
            eos_ids: Vec::new(),
            response_template: None,
            prompt_tail: String::new(),
        }
    }

    /// The canned worker reuses token ids on each request. Select its decoded
    /// answer from the actual tool-result history sent back by the router.
    pub fn with_final_answer(mut self, tool_results: usize, answer: &str) -> Self {
        self.final_answer = Some((tool_results, answer.to_string()));
        self
    }

    /// Treat chunks that look like `<|...|>` as special tokens, and the
    /// chunks equal to `eos` as EOS tokens.
    pub fn with_special_tokens(mut self, eos: &str) -> Self {
        self.special_ids = self.ids_where(|chunk| chunk.starts_with("<|") && chunk.ends_with("|>"));
        self.eos_ids = self.ids_where(|chunk| chunk == eos);
        self
    }

    pub fn with_response_template(mut self, template: serde_json::Value) -> Self {
        self.response_template = Some(template);
        self
    }

    /// End every rendered prompt with `tail`.
    pub fn with_prompt_tail(mut self, tail: &str) -> Self {
        self.prompt_tail = tail.to_string();
        self
    }

    fn ids_where(&self, pick: impl Fn(&str) -> bool) -> Vec<u32> {
        (0..self.chunks.len())
            .filter(|&index| pick(&self.chunks[index]))
            .map(|index| 100 + index as u32)
            .collect()
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
    fn decode(&self, ids: &[u32], skip_special: bool) -> anyhow::Result<String> {
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
            .filter(|id| !(skip_special && self.special_ids.contains(id)))
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
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }
    fn response_template(&self) -> Option<&serde_json::Value> {
        self.response_template.as_ref()
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
        let prompt = self.base.apply_chat_template(messages, params)?;
        Ok(prompt + &self.prompt_tail)
    }
}

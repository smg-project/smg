use llm_tokenizer::{
    chat_template::ChatTemplateParams, traits::Tokenizer, Decoder, Encoder, Encoding,
    MockTokenizer, SpecialTokens,
};

/// Decode the canned worker's generated token 100 as a chosen model output.
pub struct ScriptedTokenizer {
    base: MockTokenizer,
    output: String,
}

impl ScriptedTokenizer {
    pub fn new(output: &str) -> Self {
        Self {
            base: MockTokenizer::new(),
            output: output.to_string(),
        }
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
        Ok(if ids.contains(&100) {
            self.output.clone()
        } else {
            String::new()
        })
    }
}

impl Tokenizer for ScriptedTokenizer {
    fn vocab_size(&self) -> usize {
        self.base.vocab_size() + 1
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        self.base.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        if token == self.output {
            Some(100)
        } else {
            self.base.token_to_id(token)
        }
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        if id == 100 {
            Some(self.output.clone())
        } else {
            self.base.id_to_token(id)
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> anyhow::Result<String> {
        self.base.apply_chat_template(messages, params)
    }
}

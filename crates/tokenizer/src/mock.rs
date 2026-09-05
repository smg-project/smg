//! Mock tokenizer implementation for testing

use std::{collections::HashMap, sync::Arc};

use anyhow::Result;

use crate::{
    chat_template::ChatTemplateParams,
    traits::{
        ChatTemplateOutput, Decoder, EncodeJob, Encoder, Encoding, PromptEncoding, SpecialTokens,
        Tokenizer as TokenizerTrait,
    },
};

/// Mock tokenizer for testing purposes
pub struct MockTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    special_tokens: SpecialTokens,
    /// When set, `apply_chat_template_with_encoding` hands back these ids as
    /// a deferred encode, standing in for a renderer that encodes itself.
    deferred_chat_ids: Option<Vec<u32>>,
    /// Runs inside the deferred job, on whatever thread the caller runs it.
    deferred_chat_probe: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for MockTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MockTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add some basic tokens
        let tokens = vec![
            ("Hello", 1),
            ("world", 2),
            ("test", 3),
            ("token", 4),
            (" ", 5),
            (".", 6),
            ("<eos>", 999),
            ("<bos>", 1000),
            ("<|im_start|>", 1001),
            ("<|im_end|>", 1002),
            ("<|eot_id|>", 1003),
            ("system", 7),
            ("user", 8),
            ("assistant", 9),
        ];

        for (token, id) in tokens {
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }

        let special_tokens = SpecialTokens {
            bos_token: Some("<bos>".to_string()),
            eos_token: Some("<eos>".to_string()),
            unk_token: Some("<unk>".to_string()),
            sep_token: None,
            pad_token: None,
            cls_token: None,
            mask_token: None,
            additional_special_tokens: vec![],
        };

        Self {
            vocab,
            reverse_vocab,
            special_tokens,
            deferred_chat_ids: None,
            deferred_chat_probe: None,
        }
    }

    /// Make `apply_chat_template_with_encoding` return `ids` as a deferred
    /// encode, the way a renderer whose ids are not a function of its text
    /// does. The rendered text is unchanged.
    pub fn with_deferred_chat_ids(mut self, ids: Vec<u32>) -> Self {
        self.deferred_chat_ids = Some(ids);
        self
    }

    /// Run `probe` inside the deferred job, so a test can observe where the
    /// caller ran it.
    pub fn with_deferred_chat_probe(mut self, probe: impl Fn() + Send + Sync + 'static) -> Self {
        self.deferred_chat_probe = Some(Arc::new(probe));
        self
    }
}

impl Encoder for MockTokenizer {
    fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
        // Simple word-based tokenization using the vocab
        // Split by whitespace and look up each word (decoder adds spaces back)
        let tokens: Vec<u32> = input
            .split_whitespace()
            .filter_map(|word| self.vocab.get(word).copied())
            .collect();

        Ok(Encoding::Plain(tokens))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for MockTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|id| {
                self.reverse_vocab.get(id).and_then(|token| {
                    if skip_special_tokens && (token == "<eos>" || token == "<bos>") {
                        None
                    } else {
                        Some(token.clone())
                    }
                })
            })
            .collect();

        Ok(tokens.join(" "))
    }
}

impl TokenizerTrait for MockTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }

    fn eos_token_ids(&self) -> &[u32] {
        // `<eos>` in the mock vocab.
        &[999]
    }

    /// One `role: content` line per message, plus an `assistant:` tail when a
    /// generation prompt is requested.
    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        let mut text = String::new();
        for message in messages {
            let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            text.push_str(role);
            text.push_str(": ");
            text.push_str(content);
            text.push('\n');
        }
        if params.add_generation_prompt {
            text.push_str("assistant: ");
        }
        Ok(text)
    }

    fn apply_chat_template_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        let mut text = self.apply_chat_template(messages, params)?;
        if let Some(prefix) = assistant_prefix {
            text.push_str(prefix);
        }
        let encoding = match &self.deferred_chat_ids {
            Some(ids) => {
                let ids = ids.clone();
                let probe = self.deferred_chat_probe.clone();
                PromptEncoding::Deferred(EncodeJob::new(move || {
                    if let Some(probe) = &probe {
                        probe();
                    }
                    Ok(Encoding::Plain(ids))
                }))
            }
            None => PromptEncoding::FromText,
        };
        Ok(ChatTemplateOutput { text, encoding })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

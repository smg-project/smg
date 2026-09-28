//! Compiled tokenizer response templates shared across requests.

use std::sync::{Arc, Weak};

use llm_tokenizer::traits::Tokenizer;
use parking_lot::Mutex;
use response_template_parser::{ParserConfig, ResponseTemplateError, ResponseTemplateParser};

/// Compiled response templates keyed by tokenizer instance and model name.
///
/// Compiling decodes the template and builds its regexes once; requests clone
/// the compiled parser and own only their streaming state. Entries hold a
/// weak tokenizer reference, so a replaced tokenizer compiles afresh and the
/// stale entry is pruned on the next insert.
#[derive(Default)]
pub(crate) struct ResponseTemplateCache {
    entries: Mutex<Vec<CachedResponseTemplate>>,
}

struct CachedResponseTemplate {
    tokenizer: Weak<dyn Tokenizer>,
    model: String,
    parser: ResponseTemplateParser,
}

impl ResponseTemplateCache {
    /// The compiled template declared by `tokenizer`, if it declares one.
    /// `model` labels parser errors.
    pub(crate) fn get_or_compile(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        model: &str,
    ) -> Result<Option<ResponseTemplateParser>, ResponseTemplateError> {
        let Some(template) = tokenizer.response_template() else {
            return Ok(None);
        };
        let mut entries = self.entries.lock();
        // A held weak reference keeps its allocation alive, so an address
        // match with a live tokenizer is that same tokenizer.
        if let Some(entry) = entries.iter().find(|entry| {
            entry.model == model
                && std::ptr::addr_eq(entry.tokenizer.as_ptr(), Arc::as_ptr(tokenizer))
        }) {
            return Ok(Some(entry.parser.clone()));
        }
        let parser = ResponseTemplateParser::from_json(model, template, ParserConfig::default())?;
        entries.retain(|entry| entry.tokenizer.strong_count() > 0);
        entries.push(CachedResponseTemplate {
            tokenizer: Arc::downgrade(tokenizer),
            model: model.to_owned(),
            parser: parser.clone(),
        });
        Ok(Some(parser))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::{
        traits::{Decoder, Encoder, Encoding, SpecialTokens},
        MockTokenizer,
    };
    use serde_json::{json, Value};

    use super::*;

    struct TemplateTokenizer {
        inner: MockTokenizer,
        template: Option<Value>,
    }

    impl Encoder for TemplateTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            self.inner.encode(input, add_special_tokens)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            self.inner.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for TemplateTokenizer {
        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for TemplateTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn response_template(&self) -> Option<&Value> {
            self.template.as_ref()
        }
    }

    fn tokenizer(template: Option<Value>) -> Arc<dyn Tokenizer> {
        Arc::new(TemplateTokenizer {
            inner: MockTokenizer::new(),
            template,
        })
    }

    fn template() -> Value {
        json!({
            "start_anchor_pattern": "<s>",
            "fields": {
                "thinking": {"open_pattern": "<think>", "close": "</think>", "content": "text"},
                "content": {"open_pattern": "<answer>", "close": "</answer>", "content": "text"},
                "tool_calls": {
                    "open_pattern": "<tool name=\"(?P<name>[^\"]+)\">",
                    "close": "</tool>",
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern": "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>",
                        "value_parser": {"name": "text"}
                    },
                    "repeats": true,
                    "transform": {"name": "{name}", "arguments": "{content}"}
                }
            }
        })
    }

    #[test]
    fn compiles_once_per_tokenizer_and_model() {
        let cache = ResponseTemplateCache::default();
        assert!(cache
            .get_or_compile(&tokenizer(None), "model")
            .unwrap()
            .is_none());

        let first = tokenizer(Some(template()));
        let parser = cache.get_or_compile(&first, "model").unwrap().unwrap();
        assert_eq!(
            parser.close_literals().collect::<Vec<_>>(),
            ["</answer>", "</think>", "</tool>"]
        );
        assert!(cache.get_or_compile(&first, "model").unwrap().is_some());
        assert_eq!(cache.len(), 1);
        assert!(cache.get_or_compile(&first, "alias").unwrap().is_some());
        assert_eq!(cache.len(), 2);

        // A replacement tokenizer compiles again; the dropped one is pruned.
        drop(first);
        let second = tokenizer(Some(template()));
        assert!(cache.get_or_compile(&second, "model").unwrap().is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalid_templates_are_not_cached() {
        let cache = ResponseTemplateCache::default();
        let invalid = tokenizer(Some(json!({"fields": {}})));
        for _ in 0..2 {
            assert!(matches!(
                cache.get_or_compile(&invalid, "model"),
                Err(ResponseTemplateError::InvalidTemplate { .. })
            ));
        }
        assert_eq!(cache.len(), 0);
    }
}

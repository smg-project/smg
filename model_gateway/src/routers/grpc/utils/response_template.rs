//! Parser selection from a tokenizer-declared response template.

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use parking_lot::RwLock;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg_response_template::CompiledTemplate;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::warn;

/// The tokenizer a result was computed for, and the result.
type Entry = (Weak<dyn Tokenizer>, Option<String>);

/// Registers the parsers for the response template of each model's tokenizer.
pub(crate) struct ResponseTemplateParsers {
    tokenizers: Arc<TokenizerRegistry>,
    reasoning: ReasoningParserFactory,
    tools: ToolParserFactory,
    cache: RwLock<HashMap<String, Entry>>,
}

impl ResponseTemplateParsers {
    pub(crate) fn new(
        tokenizers: Arc<TokenizerRegistry>,
        reasoning: ReasoningParserFactory,
        tools: ToolParserFactory,
    ) -> Self {
        let cache = RwLock::default();
        Self {
            tokenizers,
            reasoning,
            tools,
            cache,
        }
    }

    /// Name of the reasoning and tool parsers for `model`'s response template,
    /// registered on first use. `None` without a supported template; an
    /// unsupported one is logged once per tokenizer.
    pub(crate) fn parser_name(&self, model: &str) -> Option<String> {
        let tokenizer = self.tokenizers.get(model)?;
        let current = Arc::downgrade(&tokenizer);
        if let Some((cached, name)) = self.cache.read().get(model) {
            if cached.ptr_eq(&current) {
                return name.clone();
            }
        }
        let name = tokenizer.response_template().and_then(|raw| {
            let template = CompiledTemplate::from_value(raw)
                .inspect_err(|error| warn!(model, %error, "ignoring response_template"))
                .ok()?;
            let template = Arc::new(template);
            self.reasoning.register_response_template(template.clone());
            Some(self.tools.register_response_template(template))
        });
        let entry = (current, name.clone());
        self.cache.write().insert(model.to_string(), entry);
        name
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::MockTokenizer;
    use openai_protocol::model_card::ModelCard;
    use serde_json::{json, Value};

    use super::*;
    use crate::{
        routers::grpc::utils::ParserResolver,
        worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
    };

    fn template(tool_close: &str) -> Value {
        json!({
            "start_anchor_pattern": "<\\|bot\\|>",
            "fields": {
                "thinking": {"open_pattern": "<\\|think\\|>", "close": "<|done|>", "content": "text"},
                "content": {"open_pattern": "<\\|say\\|>", "close": "<|done|>", "content": "text"},
                "tool_calls": {
                    "open_pattern": "<\\|tool\\|>(?P<name>\\w+)<\\|args\\|>",
                    "close": tool_close,
                    "repeats": true,
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern": "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>",
                        "value_parser": {"name": "text"}
                    },
                    "transform": {"name": "{name}", "arguments": "{content}"}
                }
            }
        })
    }

    async fn load(registry: &TokenizerRegistry, model: &str, template: Option<Value>) {
        let tokenizer = match template {
            Some(template) => MockTokenizer::new().with_response_template(template),
            None => MockTokenizer::new(),
        };
        registry.remove(model);
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
        registry
            .load(model, model, "test", || async move { Ok(tokenizer) })
            .await
            .unwrap();
    }

    fn parsers(registry: &Arc<TokenizerRegistry>) -> ResponseTemplateParsers {
        let reasoning = ReasoningParserFactory::new();
        ResponseTemplateParsers::new(registry.clone(), reasoning, ToolParserFactory::new())
    }

    #[tokio::test]
    async fn registers_both_parsers_once_per_tokenizer() {
        let registry = Arc::new(TokenizerRegistry::new());
        load(&registry, "m", Some(template("<|done|>"))).await;
        let parsers = parsers(&registry);

        let name = parsers.parser_name("m").unwrap();
        let expected = CompiledTemplate::from_value(&template("<|done|>")).unwrap();
        assert_eq!(name, expected.parser_name());
        assert!(parsers.reasoning.registry().has_parser(&name));
        assert!(parsers.tools.registry().has_parser(&name));
        assert_eq!(parsers.parser_name("m"), Some(name.clone()));

        // A replaced tokenizer is looked at again; its new template gets a
        // new name, so no parser built for the old one is reused.
        load(&registry, "m", Some(template("<|eos|>"))).await;
        let replaced = parsers.parser_name("m").unwrap();
        assert_ne!(replaced, name);
        load(&registry, "m", None).await;
        assert_eq!(parsers.parser_name("m"), None);
    }

    #[tokio::test]
    async fn unsupported_templates_and_unknown_models_resolve_to_none() {
        let registry = Arc::new(TokenizerRegistry::new());
        load(&registry, "bad", Some(json!({"fields": {}}))).await;
        let parsers = parsers(&registry);
        assert_eq!(parsers.parser_name("bad"), None);
        assert_eq!(parsers.parser_name("unknown"), None);
        // Only models with a tokenizer are cached.
        assert_eq!(parsers.cache.read().len(), 1);
    }

    #[tokio::test]
    async fn template_ranks_between_card_overrides_and_configured_names() {
        let registry = Arc::new(TokenizerRegistry::new());
        load(&registry, "m", Some(template("<|done|>"))).await;
        load(&registry, "plain", None).await;
        let parsers = Arc::new(parsers(&registry));
        let name = parsers.parser_name("m");

        let workers = Arc::new(WorkerRegistry::new());
        let card = ModelCard::new("m").with_tool_parser("json");
        let worker = BasicWorkerBuilder::new("http://w1:8000")
            .model(card)
            .worker_type(WorkerType::Regular)
            .build();
        workers.register(Arc::new(worker));
        let resolver = ParserResolver::new(
            workers,
            Some("qwen".to_string()),
            Some("passthrough".to_string()),
            Some(parsers),
        );
        // The card replaces only its own side; the template beats the flags.
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("json"));
        assert_eq!(resolver.reasoning_parser("m"), name);
        assert_eq!(resolver.tool_parser("plain").as_deref(), Some("qwen"));
        assert_eq!(
            resolver.reasoning_parser("plain").as_deref(),
            Some("passthrough")
        );
    }
}

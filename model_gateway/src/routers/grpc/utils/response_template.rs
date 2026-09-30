//! Parsers selected from the response template of a model's tokenizer.

use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Weak},
};

use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::common::Tool;
use parking_lot::RwLock;
use reasoning_parser::{ParserFactory as ReasoningParserFactory, TemplateReasoningParser};
use serde_json::Value;
use smg_response_template::{
    adapter::{self, ResponseParserState},
    load_response_template, ResponseTemplate,
};
use tool_parser::{ParserFactory as ToolParserFactory, TemplateToolParser};
use tracing::warn;

/// The tokenizer an entry was computed for, and its parser name and template.
type Entry = (Weak<dyn Tokenizer>, Option<(String, ResponseTemplate)>);

/// Loads the response template of each model's tokenizer once and registers
/// the reasoning and tool parsers for it.
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
        Self {
            tokenizers,
            reasoning,
            tools,
            cache: RwLock::default(),
        }
    }

    /// The parser name and the template of `model`'s response template, or
    /// `None` without one the parsers can use (logged once per tokenizer).
    /// The name follows the template, so a changed template gets new parsers.
    pub(crate) fn get(&self, model: &str) -> Option<(String, ResponseTemplate)> {
        let tokenizer = self.tokenizers.get(model)?;
        let current = Arc::downgrade(&tokenizer);
        if let Some((cached, entry)) = self.cache.read().get(model) {
            if cached.ptr_eq(&current) {
                return entry.clone();
            }
        }
        let entry = tokenizer.response_template().and_then(|raw| {
            let template = if has_float(raw) {
                Err("a float, which serde_json can read differently from Python".to_owned())
            } else {
                load_response_template(raw).map_err(|error| error.to_string())
            }
            .and_then(|t| adapter::check(&t).map(|()| t).map_err(|e| e.to_string()))
            .inspect_err(|reason| warn!(model, reason, "not using the response_template"))
            .ok()?;
            let mut hasher = DefaultHasher::new();
            raw.to_string().hash(&mut hasher);
            let name = format!("response_template_{:016x}", hasher.finish());
            if !self.reasoning.registry().has_parser(&name) {
                let t = template.clone();
                let parser = move || Box::new(TemplateReasoningParser::new(t.clone())) as _;
                self.reasoning.registry().register_parser(&name, parser);
            }
            if !self.tools.registry().has_parser(&name) {
                let t = template.clone();
                let parser = move || Box::new(TemplateToolParser::new(t.clone())) as _;
                self.tools.registry().register_parser(&name, parser);
            }
            Some((name, template))
        });
        let cached = (current, entry.clone());
        self.cache.write().insert(model.to_string(), cached);
        entry
    }
}

/// Whether a template read from `tokenizer_config.json` holds a float: serde_json
/// rounds some float literals differently from Python's `json.loads` and reads
/// integers beyond u64 as floats, and a float can reach the output or decide how
/// the template parses (`strip`, `version`, ...).
///
/// A follow-up can drop this refusal: keep the template's raw text from
/// `tokenizer_config.json` (a `serde_json::value::RawValue`) and read it with
/// the crate's port of `json.loads`, which parses numbers as Python does.
fn has_float(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.is_f64(),
        Value::Array(items) => items.iter().any(has_float),
        Value::Object(map) => map.values().any(has_float),
        _ => false,
    }
}

/// What the response-template parsers of one request start from: the
/// rendered prompt after the template's last start anchor, and the tools.
#[derive(Clone)]
pub(crate) struct ResponseParserSpec {
    template: ResponseTemplate,
    prompt_tail: Arc<str>,
    tools: Arc<[Value]>,
    continuation: bool,
}

impl ResponseParserSpec {
    /// `continuation`: the prompt ends inside the assistant message.
    pub(crate) fn new<'a>(
        template: ResponseTemplate,
        prompt: &str,
        tools: impl IntoIterator<Item = &'a Tool>,
        continuation: bool,
    ) -> Self {
        let tools = tools
            .into_iter()
            .filter_map(|t| serde_json::to_value(t).ok());
        Self {
            prompt_tail: template.truncate_past_last_anchor(prompt).into(),
            template,
            tools: tools.collect(),
            continuation,
        }
    }

    /// The state one generated choice's reasoning and tool parsers share.
    pub(crate) fn new_state(&self) -> ResponseParserState {
        let tools = &self.tools;
        ResponseParserState::new(&self.template, &self.prompt_tail, tools, self.continuation)
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::MockTokenizer;
    use openai_protocol::model_card::ModelCard;
    use serde_json::json;
    use smg_response_template::LoadError;

    use super::*;
    use crate::{
        routers::grpc::utils::ParserResolver,
        worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
    };

    fn template(close: &str) -> Value {
        json!({
            "start_anchor": "<|assistant|>",
            "fields": {
                "thinking": {"open": "<think>", "close": close},
                "content": {}
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
        load(&registry, "m", Some(template("</think>"))).await;
        let parsers = parsers(&registry);

        let (name, _) = parsers.get("m").unwrap();
        assert!(parsers.reasoning.registry().has_parser(&name));
        assert!(parsers.tools.registry().has_parser(&name));
        assert_eq!(parsers.get("m").unwrap().0, name);

        // A replaced tokenizer is looked at again; a changed template gets
        // new parsers, so no pooled parser of the old one serves it.
        load(&registry, "m", Some(template("</thought>"))).await;
        assert_ne!(parsers.get("m").unwrap().0, name);
        load(&registry, "m", None).await;
        assert!(parsers.get("m").is_none());
        assert!(parsers.get("unknown").is_none());
        let mut with_float = template("</think>");
        with_float["fields"]["content"]["content_args"] = json!({"strip": 0.5});
        load(&registry, "m", Some(with_float)).await;
        assert!(parsers.get("m").is_none());
    }

    /// A template transformers rejects, one this crate refuses, and one the
    /// parsers cannot use leave the parsers to the existing selection.
    #[tokio::test]
    async fn refused_templates_fall_back() {
        let invalid = json!({"start_anchor": "<a>", "fields": {"content": {"content": "yaml"}}});
        let lookahead = json!({"start_anchor": "<a>", "fields": {"x": {"open_pattern": "(?=x)"}}});
        let content_only = json!({"start_anchor": "<a>", "fields": {"content": {}}});
        assert!(matches!(
            load_response_template(&invalid),
            Err(LoadError::Invalid { .. })
        ));
        assert!(matches!(
            load_response_template(&lookahead),
            Err(LoadError::Unsupported { .. })
        ));
        assert!(load_response_template(&content_only).is_ok_and(|t| adapter::check(&t).is_err()));
        let registry = Arc::new(TokenizerRegistry::new());
        let parsers = parsers(&registry);
        for raw in [invalid, lookahead, content_only] {
            load(&registry, "m", Some(raw)).await;
            assert!(parsers.get("m").is_none());
        }
    }

    #[tokio::test]
    async fn a_template_needs_both_sides_free_of_explicit_choices() {
        let registry = Arc::new(TokenizerRegistry::new());
        load(&registry, "m", Some(template("</think>"))).await;
        load(&registry, "carded", Some(template("</think>"))).await;
        let parsers = Arc::new(parsers(&registry));
        let name = parsers.get("m").map(|(name, _)| name);

        let workers = Arc::new(WorkerRegistry::new());
        let card = ModelCard::new("carded").with_tool_parser("json");
        let worker = BasicWorkerBuilder::new("http://w1:8000")
            .model(card)
            .worker_type(WorkerType::Regular)
            .build();
        workers.register(Arc::new(worker));

        let resolver = ParserResolver::new(workers.clone(), None, None, Some(parsers.clone()));
        assert_eq!(resolver.reasoning_parser("m"), name);
        assert_eq!(resolver.tool_parser("m"), name);
        assert!(resolver.response_template("m").is_some());
        // A card override on one side turns the template off for both.
        assert_eq!(resolver.tool_parser("carded").as_deref(), Some("json"));
        assert_eq!(resolver.reasoning_parser("carded"), None);
        assert!(resolver.response_template("carded").is_none());

        // So does a configured name.
        let resolver = ParserResolver::new(workers, None, Some("qwen3".into()), Some(parsers));
        assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("qwen3"));
        assert_eq!(resolver.tool_parser("m"), None);
        assert!(resolver.response_template("m").is_none());
    }
}

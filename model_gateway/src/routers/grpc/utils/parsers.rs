//! Reasoning and tool parser helpers.

use std::sync::Arc;

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::{chat::thinking_from_reasoning_effort, model_card::ModelCard};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ReasoningParser};
use serde_json::Value;
use tool_parser::{
    ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
};
use tracing::warn;

use crate::worker::WorkerRegistry;

/// Per-request parser-name resolution.
///
/// Precedence: the model's `ModelCard` override (`tool_parser` /
/// `reasoning_parser`, populated from worker labels or an explicit
/// `WorkerSpec` card) → the process-wide configured name
/// (`--tool-call-parser` / `--reasoning-parser`) → `None`, which lets the
/// factory helpers fall back to their name-based auto-detection, unchanged.
///
/// Lookups borrow straight from worker metadata (no card clones); only the
/// resolved name is cloned.
#[derive(Clone)]
pub(crate) struct ParserResolver {
    /// `None` disables card lookups (parser-free endpoints, tests).
    worker_registry: Option<Arc<WorkerRegistry>>,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
    /// Emit the tool-call suppression constraint when tools are present but
    /// `tool_choice` is `"none"` (RouterConfig `tool_choice_none_ban`).
    tool_choice_none_ban: bool,
}

impl ParserResolver {
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            worker_registry: Some(worker_registry),
            configured_tool_parser,
            configured_reasoning_parser,
            tool_choice_none_ban: false,
        }
    }

    /// Resolver that never consults model cards and carries no configured
    /// names — preserves the parser-free endpoints' behavior.
    pub(crate) fn disabled() -> Self {
        Self {
            worker_registry: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            tool_choice_none_ban: false,
        }
    }

    #[must_use]
    pub(crate) fn with_tool_choice_none_ban(mut self, enabled: bool) -> Self {
        self.tool_choice_none_ban = enabled;
        self
    }

    /// Whether the `tool_choice: "none"` suppression constraint is enabled.
    pub(crate) fn tool_choice_none_ban(&self) -> bool {
        self.tool_choice_none_ban
    }

    /// Effective tool-parser name for `model`, if any.
    pub(crate) fn tool_parser(&self, model: &str) -> Option<String> {
        self.card_parser(model, |card| card.tool_parser.as_ref())
            .or_else(|| self.configured_tool_parser.clone())
    }

    /// Effective reasoning-parser name for `model`, if any.
    pub(crate) fn reasoning_parser(&self, model: &str) -> Option<String> {
        self.card_parser(model, |card| card.reasoning_parser.as_ref())
            .or_else(|| self.configured_reasoning_parser.clone())
    }

    fn card_parser(
        &self,
        model: &str,
        pick: impl Fn(&ModelCard) -> Option<&String>,
    ) -> Option<String> {
        let registry = self.worker_registry.as_ref()?;
        // Cards built by the label pipeline agree across workers of one model;
        // if they don't (mixed labels, e.g. mid rolling-upgrade), pick the
        // lexicographically smallest so resolution is deterministic rather
        // than registry-iteration-order dependent. Registration logs a
        // warning for the conflict; here it's debug (per-request hot path).
        let mut chosen: Option<String> = None;
        let mut conflict = false;
        for worker in registry.get_by_model(model).iter() {
            let Some(name) = worker.metadata().spec.models.find(model).and_then(&pick) else {
                continue;
            };
            match &chosen {
                None => chosen = Some(name.clone()),
                Some(existing) if existing != name => {
                    conflict = true;
                    if name < existing {
                        chosen = Some(name.clone());
                    }
                }
                Some(_) => {}
            }
        }
        if conflict {
            tracing::debug!(
                model,
                chosen = chosen.as_deref(),
                "Workers for this model declare conflicting parser overrides; \
                 using the lexicographically smallest"
            );
        }
        chosen
    }
}

/// Determine if thinking is effectively ON based on the template's thinking
/// toggle and the user's request.
///
/// `user_thinking`: `Some(true)` = user enabled thinking, `Some(false)` = user
/// disabled it, `None` = not specified (use template default).
pub fn should_mark_reasoning_started(
    user_thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
) -> bool {
    match tokenizer.thinking_toggle() {
        ThinkingToggle::None => false,
        ThinkingToggle::DefaultOn => user_thinking != Some(false),
        ThinkingToggle::DefaultOff => user_thinking == Some(true),
    }
}

/// Extract the user's thinking preference from chat_template_kwargs.
///
/// Only checks the key that the template actually uses (e.g. `enable_thinking`
/// for Qwen3, `thinking` for Kimi-K2.5). This prevents mismatches where the
/// user passes the wrong key name and the template ignores it.
pub(crate) fn extract_thinking_from_kwargs(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    let kwargs = kwargs?;
    match tokenizer.thinking_key_name() {
        Some(ThinkingKeyName::EnableThinking) => kwargs.get("enable_thinking"),
        Some(ThinkingKeyName::Thinking) => kwargs.get("thinking"),
        None => None,
    }
    .and_then(|v| v.as_bool())
}

/// Report `Some(true)` when the renderer will enter thinking mode because of
/// a native reasoning-effort value, so the reasoning parser is armed
/// consistently with the rendered prompt. Mirrors the template-kwargs merge:
/// an explicit kwargs entry wins over the top-level `reasoning_effort` field.
fn extract_template_effort_thinking(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    let native_values = tokenizer.native_reasoning_effort_values();
    if native_values.is_empty() {
        return None;
    }
    let effort = kwargs
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(Value::as_str)
        .or(reasoning_effort)?;
    native_values.contains(&effort).then_some(true)
}

/// Precedence for the effective thinking preference: an explicit template
/// toggle always wins, then a native template effort for renderers that
/// support it, then the protocol-level OpenAI `reasoning_effort` mapping
/// ([`thinking_from_reasoning_effort`]).
fn resolve_thinking_pref(
    explicit: Option<bool>,
    template_effort: Option<bool>,
    reasoning_effort: Option<&str>,
) -> Option<bool> {
    explicit
        .or(template_effort)
        .or_else(|| thinking_from_reasoning_effort(reasoning_effort))
}

/// Resolve the user's effective thinking preference.
pub fn resolve_user_thinking(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    resolve_thinking_pref(
        extract_thinking_from_kwargs(kwargs, tokenizer),
        extract_template_effort_thinking(kwargs, reasoning_effort, tokenizer),
        reasoning_effort,
    )
}

/// Check if a reasoning parser is available for the given model
pub(crate) fn check_reasoning_parser_availability(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        reasoning_parser_factory.registry().has_parser(parser_name)
    } else {
        reasoning_parser_factory
            .registry()
            .has_parser_for_model(model)
    }
}

/// Check if a tool parser is available for the given model
pub(crate) fn check_tool_parser_availability(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        tool_parser_factory.registry().has_parser(parser_name)
    } else {
        tool_parser_factory.registry().has_parser_for_model(model)
    }
}

/// Create a fresh reasoning parser instance.
///
/// Used for both streaming (state isolation across chunks) and non-streaming
/// (avoids serializing on the shared pooled parser mutex).
pub(crate) fn create_reasoning_parser(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ReasoningParser>> {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        reasoning_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.registry().create_for_model(model)
            })
    } else {
        // Auto-detect based on model
        reasoning_parser_factory.registry().create_for_model(model)
    }
}

/// Whether the selected reasoning parser needs tokenizer special tokens to be
/// preserved in decoded output.
pub(crate) fn reasoning_parser_requires_special_tokens(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    create_reasoning_parser(reasoning_parser_factory, configured_parser, model).is_some_and(
        |parser| {
            let parser_ref: &dyn ReasoningParser = parser.as_ref();
            parser_ref.requires_special_tokens()
        },
    )
}

/// Get the appropriate tool parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
/// Get a pooled tool parser (for non-streaming where state doesn't matter)
pub(crate) fn get_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> ToolPooledParser {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        tool_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.get_pooled(model)
            })
    } else {
        // Auto-detect based on model
        tool_parser_factory.get_pooled(model)
    }
}

/// Create a fresh tool parser instance (for streaming where state isolation is needed)
pub(crate) fn create_tool_parser(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ToolParser>> {
    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        tool_parser_factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.registry().create_for_model(model)
            })
    } else {
        // Auto-detect based on model
        tool_parser_factory.registry().create_for_model(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_thinking_pref_precedence() {
        // Explicit toggle > native template effort > reasoning_effort mapping.
        assert_eq!(
            resolve_thinking_pref(Some(false), Some(true), Some("high")),
            Some(false)
        );
        assert_eq!(
            resolve_thinking_pref(None, Some(true), Some("none")),
            Some(true)
        );
        assert_eq!(resolve_thinking_pref(None, None, Some("none")), Some(false));
        assert_eq!(resolve_thinking_pref(None, None, Some("high")), None);
        assert_eq!(resolve_thinking_pref(None, None, None), None);
    }

    #[test]
    fn template_effort_thinking_covers_kwargs_and_top_level_field() {
        use llm_tokenizer::traits::{Encoder, Encoding};
        struct T(llm_tokenizer::MockTokenizer);
        impl Encoder for T {
            fn encode(&self, i: &str, s: bool) -> anyhow::Result<Encoding> {
                self.0.encode(i, s)
            }
            fn encode_batch(&self, i: &[&str], s: bool) -> anyhow::Result<Vec<Encoding>> {
                self.0.encode_batch(i, s)
            }
        }
        impl llm_tokenizer::traits::Decoder for T {
            fn decode(&self, ids: &[u32], s: bool) -> anyhow::Result<String> {
                self.0.decode(ids, s)
            }
        }
        impl Tokenizer for T {
            fn vocab_size(&self) -> usize {
                self.0.vocab_size()
            }
            fn get_special_tokens(&self) -> &llm_tokenizer::traits::SpecialTokens {
                self.0.get_special_tokens()
            }
            fn token_to_id(&self, t: &str) -> Option<u32> {
                self.0.token_to_id(t)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
                &["low", "high", "max"]
            }
        }
        let tok = T(llm_tokenizer::MockTokenizer::new());

        // Top-level field arms thinking when the renderer would interpret it.
        assert_eq!(
            extract_template_effort_thinking(None, Some("high"), &tok),
            Some(true)
        );
        // Unrecognized values never arm (the renderer ignores them too).
        assert_eq!(
            extract_template_effort_thinking(None, Some("medium"), &tok),
            None
        );
        // A kwargs entry wins over the top-level field, matching the merge.
        let kwargs = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            Value::String("max".to_string()),
        )]);
        assert_eq!(
            extract_template_effort_thinking(Some(&kwargs), Some("medium"), &tok),
            Some(true)
        );
        // Renderers without native efforts never arm.
        assert_eq!(
            extract_template_effort_thinking(
                Some(&kwargs),
                Some("high"),
                &llm_tokenizer::MockTokenizer::new()
            ),
            None
        );
    }

    #[test]
    fn create_reasoning_parser_returns_independent_instances() {
        let factory = ReasoningParserFactory::new();

        // qwen3 starts with in_reasoning=false (explicit <think> required).
        let mut a =
            create_reasoning_parser(&factory, None, "qwen3").expect("qwen3 has a reasoning parser");
        let mut b =
            create_reasoning_parser(&factory, None, "qwen3").expect("qwen3 has a reasoning parser");

        // Each call returns an independent instance: state mutated on one parser
        // must not leak into the other (the shared pooled parser the non-streaming
        // path used to take would have violated this).
        a.mark_reasoning_started();
        assert!(a.is_in_reasoning());
        assert!(!b.is_in_reasoning());

        // The untouched instance still parses a full document correctly.
        let rb = b
            .detect_and_parse_reasoning("<think>reasoning</think>answer")
            .unwrap();
        assert_eq!(rb.normal_text, "answer");
        assert_eq!(rb.reasoning_text, "reasoning");
    }

    #[test]
    fn create_reasoning_parser_honors_configured_parser() {
        let factory = ReasoningParserFactory::new();

        let parser = create_reasoning_parser(&factory, Some("qwen3"), "unknown-model")
            .expect("configured qwen3 parser exists");
        assert_eq!(parser.model_type(), "qwen3");
    }

    #[test]
    fn inkling_parser_requires_special_tokens() {
        let factory = ReasoningParserFactory::new();

        assert!(reasoning_parser_requires_special_tokens(
            &factory,
            Some("inkling"),
            "served-model"
        ));
        assert!(!reasoning_parser_requires_special_tokens(
            &factory,
            Some("qwen3"),
            "served-model"
        ));
    }
}

#[cfg(test)]
mod parser_resolver_tests {
    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType};

    fn registry_with_card(card: ModelCard) -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("http://w1:8000")
            .model(card)
            .worker_type(WorkerType::Regular)
            .build();
        registry.register(Arc::new(worker));
        registry
    }

    #[test]
    fn card_override_wins_over_configured() {
        let registry = registry_with_card(
            ModelCard::new("m")
                .with_tool_parser("json")
                .with_reasoning_parser("basic"),
        );
        let resolver = ParserResolver::new(
            registry,
            Some("mistral".to_string()),
            Some("deepseek_r1".to_string()),
        );
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("json"));
        assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("basic"));
    }

    #[test]
    fn falls_back_to_configured_without_card_override() {
        let registry = registry_with_card(ModelCard::new("m"));
        let resolver = ParserResolver::new(
            registry,
            Some("mistral".to_string()),
            Some("deepseek_r1".to_string()),
        );
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("mistral"));
        assert_eq!(
            resolver.reasoning_parser("m").as_deref(),
            Some("deepseek_r1")
        );
        // Unknown model: no card, same configured fallback.
        assert_eq!(resolver.tool_parser("other").as_deref(), Some("mistral"));
    }

    #[test]
    fn no_override_and_no_configured_resolves_none() {
        let registry = registry_with_card(ModelCard::new("m"));
        let resolver = ParserResolver::new(registry, None, None);
        assert_eq!(resolver.tool_parser("m"), None);
        assert_eq!(resolver.reasoning_parser("m"), None);
    }

    #[test]
    fn disabled_resolver_never_resolves() {
        let resolver = ParserResolver::disabled();
        assert_eq!(resolver.tool_parser("m"), None);
        assert_eq!(resolver.reasoning_parser("m"), None);
    }

    #[test]
    fn tool_choice_none_ban_defaults_off_and_round_trips() {
        let registry = registry_with_card(ModelCard::new("m"));
        let resolver = ParserResolver::new(registry, None, None);
        assert!(!resolver.tool_choice_none_ban());
        assert!(!ParserResolver::disabled().tool_choice_none_ban());
        let enabled = resolver.with_tool_choice_none_ban(true);
        assert!(enabled.tool_choice_none_ban());
    }

    #[test]
    fn conflicting_overrides_resolve_deterministically() {
        // Two same-model workers with different overrides: resolution must
        // not depend on registration/iteration order — the lexicographically
        // smallest name wins either way.
        for (first, second) in [("zebra", "alpha"), ("alpha", "zebra")] {
            let registry = Arc::new(WorkerRegistry::new());
            for (i, name) in [first, second].iter().enumerate() {
                let worker = BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                    .model(ModelCard::new("m").with_tool_parser(*name))
                    .worker_type(WorkerType::Regular)
                    .build();
                registry.register(Arc::new(worker));
            }
            let resolver = ParserResolver::new(registry, None, None);
            assert_eq!(resolver.tool_parser("m").as_deref(), Some("alpha"));
        }
    }
}

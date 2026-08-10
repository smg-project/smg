//! Reasoning and tool parser helpers.

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::chat::thinking_from_reasoning_effort;
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ReasoningParser};
use serde_json::Value;
use tool_parser::{
    ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
};
use tracing::warn;

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

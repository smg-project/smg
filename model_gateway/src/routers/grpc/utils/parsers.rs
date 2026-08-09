//! Reasoning and tool parser helpers.

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::chat::ChatCompletionRequest;
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

/// Resolve the user's effective thinking preference: an explicit template
/// kwarg wins, then the protocol-level preference
/// ([`ChatCompletionRequest::thinking_preference`] — DeepSeek's official
/// `thinking` field, then the OpenAI `reasoning_effort` compatibility mapping).
pub fn resolve_user_thinking(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    extract_thinking_from_kwargs(request.chat_template_kwargs.as_ref(), tokenizer)
        .or_else(|| request.thinking_preference())
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
    use llm_tokenizer::{
        mock::MockTokenizer,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType},
    };
    use serde_json::json;

    use super::*;

    /// A mock tokenizer whose template uses the `thinking` toggle key, like the
    /// DeepSeek V4 renderer.
    struct ThinkingKeyTokenizer(MockTokenizer);

    impl Encoder for ThinkingKeyTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            self.0.encode(input, add_special_tokens)
        }
        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            self.0.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for ThinkingKeyTokenizer {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            skip_special_tokens: bool,
        ) -> anyhow::Result<String> {
            self.0.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for ThinkingKeyTokenizer {
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            self.0.get_special_tokens()
        }
        fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
            self.0.token_to_id(token)
        }
        fn id_to_token(&self, id: TokenIdType) -> Option<String> {
            self.0.id_to_token(id)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn thinking_toggle(&self) -> ThinkingToggle {
            ThinkingToggle::DefaultOn
        }
        fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
            Some(ThinkingKeyName::Thinking)
        }
    }

    fn request_with(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hello"}],
        });
        request.as_object_mut().unwrap().extend(
            extra
                .as_object()
                .expect("extra request fields must be an object")
                .clone(),
        );
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn resolve_user_thinking_template_kwarg_wins() {
        let tokenizer = ThinkingKeyTokenizer(MockTokenizer::new());

        // The template's own toggle key beats the protocol `thinking` field
        // and the reasoning_effort mapping.
        let request = request_with(json!({
            "chat_template_kwargs": {"thinking": true},
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        }));
        assert_eq!(resolve_user_thinking(&request, &tokenizer), Some(true));

        let request = request_with(json!({
            "chat_template_kwargs": {"thinking": false},
            "thinking": {"type": "enabled"},
        }));
        assert_eq!(resolve_user_thinking(&request, &tokenizer), Some(false));

        // A kwarg under a key the template does not use is ignored.
        let request = request_with(json!({
            "chat_template_kwargs": {"enable_thinking": false},
            "thinking": {"type": "enabled"},
        }));
        assert_eq!(resolve_user_thinking(&request, &tokenizer), Some(true));
    }

    #[test]
    fn resolve_user_thinking_falls_back_to_protocol_preference() {
        let tokenizer = ThinkingKeyTokenizer(MockTokenizer::new());

        // DeepSeek's official `thinking` field beats reasoning_effort.
        let request = request_with(json!({
            "thinking": {"type": "enabled"},
            "reasoning_effort": "none",
        }));
        assert_eq!(resolve_user_thinking(&request, &tokenizer), Some(true));

        let request = request_with(json!({
            "thinking": {"type": "disabled"},
            "reasoning_effort": "high",
        }));
        assert_eq!(resolve_user_thinking(&request, &tokenizer), Some(false));

        // reasoning_effort none/minimal map to off; levels express no opinion.
        for (effort, expected) in [
            ("none", Some(false)),
            ("minimal", Some(false)),
            ("high", None),
        ] {
            let request = request_with(json!({"reasoning_effort": effort}));
            assert_eq!(
                resolve_user_thinking(&request, &tokenizer),
                expected,
                "{effort}"
            );
        }

        // No signal at all -> no preference; the template default decides.
        assert_eq!(
            resolve_user_thinking(&request_with(json!({})), &tokenizer),
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

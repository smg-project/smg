//! Reasoning and tool parser helpers.

use std::sync::Arc;

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{thinking_from_reasoning_effort, ChatCompletionRequest},
    model_card::ModelCard,
    profile::ProviderProfile,
    worker::ConstrainedDecoding,
};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ReasoningParser};
use serde_json::Value;
use tool_parser::{
    GrammarStart, ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
};
use tracing::warn;

use crate::{
    routers::grpc::context::WorkerSelection,
    worker::{Worker, WorkerRegistry, WorkerType},
};

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
        }
    }

    /// Resolver that never consults model cards and carries no configured
    /// names — preserves the parser-free endpoints' behavior.
    pub(crate) fn disabled() -> Self {
        Self {
            worker_registry: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
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

    /// Both names for `model`, as [`Self::tool_parser`] and [`Self::reasoning_parser`] resolve
    /// them: Symphony reads them as a pair (`regular::symphony::reads`).
    pub(crate) fn names(&self, model: &str) -> Names {
        Names {
            tool: self.tool_parser(model),
            reasoning: self.reasoning_parser(model),
        }
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

/// The parser names set for one model, resolved by [`ParserResolver`]: the tool-call parser's
/// and the reasoning parser's, each `None` when nothing names one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Names {
    pub tool: Option<String>,
    pub reasoning: Option<String>,
}

/// Determine if thinking is effectively ON based on the template's thinking
/// toggle and the user's request.
///
/// `user_thinking`: `Some(true)` = user enabled thinking, `Some(false)` = user
/// disabled it, `None` = not specified (use template default).
///
/// A template without a switch cannot turn thinking off, whatever the request
/// asks: when its generation prompt opens the think block (GLM-5.3), the
/// completion starts inside it and the parser must start there too, or the
/// reasoning streams as content.
pub fn should_mark_reasoning_started(
    user_thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
) -> bool {
    match tokenizer.thinking_toggle() {
        ThinkingToggle::None => tokenizer.prefill_opens_think_block(),
        ThinkingToggle::DefaultOn => user_thinking != Some(false),
        ThinkingToggle::DefaultOff => user_thinking == Some(true),
    }
}

/// Extract the user's thinking preference from chat_template_kwargs.
///
/// Only checks the key that the template actually uses (e.g. `enable_thinking`
/// for Qwen3, `thinking` for Kimi-K2.5), plus vLLM's `enable_thinking` alias
/// for renderers that declare it (`RendererCapabilities::enable_thinking_alias`).
/// This prevents mismatches where the user passes a key name the template
/// ignores. A template keyed on `reasoning_effort` is read from the value it
/// sees: the request-level field written over a kwargs entry, as the render's
/// merge does.
pub(crate) fn extract_thinking_from_kwargs(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    match tokenizer.thinking_key_name() {
        Some(ThinkingKeyName::EnableThinking) => {
            kwargs?.get("enable_thinking").and_then(Value::as_bool)
        }
        // Renderers that honour vLLM's `enable_thinking` alias (DeepSeek-V4.1)
        // read it too; `thinking` wins when both are present (the renderer
        // rejects a disagreeing pair before anything is dispatched).
        Some(ThinkingKeyName::Thinking) => {
            let kwargs = kwargs?;
            kwargs.get("thinking").and_then(Value::as_bool).or_else(|| {
                tokenizer
                    .renderer_capabilities()
                    .enable_thinking_alias
                    .then(|| kwargs.get("enable_thinking").and_then(Value::as_bool))
                    .flatten()
            })
        }
        // The template's own on/off words for `reasoning_effort`, read from
        // the value the template renders (`reasoning_effort_as_rendered`).
        Some(ThinkingKeyName::ReasoningEffort) => {
            let effort = reasoning_effort_as_rendered(kwargs, reasoning_effort)?;
            native_effort_thinking(effort, tokenizer)
        }
        // Tri-state string toggle: adaptive adds no reasoning prefix.
        Some(ThinkingKeyName::ThinkingMode) => {
            match kwargs?.get("thinking_mode").and_then(Value::as_str) {
                Some("enabled") => Some(true),
                Some("disabled") => Some(false),
                _ => None,
            }
        }
        None => None,
    }
}

/// The thinking preference implied by `reasoning_effort` for a renderer that
/// reads the kwarg natively, so the reasoning parser is armed consistently
/// with the rendered prompt: `Some(true)` for one of its on words, `Some(false)`
/// for one of its off words (which short-circuits the generic
/// `reasoning_effort` fallback in `resolve_thinking_pref`), `None` otherwise.
/// Mirrors the template-kwargs merge: the request-level `reasoning_effort`
/// is written over an explicit kwargs entry.
fn extract_template_effort_thinking(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    if tokenizer.native_reasoning_effort_values().is_empty()
        && tokenizer.native_reasoning_effort_off_values().is_empty()
    {
        return None;
    }
    let effort = reasoning_effort_as_rendered(kwargs, reasoning_effort)?;
    native_effort_thinking(effort, tokenizer)
}

/// The `reasoning_effort` the template sees: the request-level field
/// (`thinking.effort`, else the top-level field), written over a
/// `chat_template_kwargs` entry as the engine's own server renders it, else
/// the entry (`build_chat_template_kwargs` in `chat_utils` merges them the
/// same way).
fn reasoning_effort_as_rendered<'a>(
    kwargs: Option<&'a std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&'a str>,
) -> Option<&'a str> {
    reasoning_effort.or_else(|| {
        kwargs
            .and_then(|k| k.get("reasoning_effort"))
            .and_then(Value::as_str)
    })
}

/// What a `reasoning_effort` value means to this renderer, wherever it
/// arrives (kwargs or top-level; the gateway forwards both to the template).
/// A renderer that declares its own off words (Hy4's `no_think`) is read by
/// those alone: an off word disarms the parser and every other value renders
/// the template's default, so the protocol switch must not apply. One that
/// declares none is switched off by the protocol's `"none"`/`"minimal"`,
/// which the DeepSeek renderers render as chat mode, and armed by an on word.
fn native_effort_thinking(effort: &str, tokenizer: &dyn Tokenizer) -> Option<bool> {
    let off_values = tokenizer.native_reasoning_effort_off_values();
    let on_values = tokenizer.native_reasoning_effort_values();
    if off_values.is_empty() {
        if thinking_from_reasoning_effort(Some(effort)) == Some(false) {
            return Some(false);
        }
        return on_values.contains(&effort).then_some(true);
    }
    if off_values.contains(&effort) {
        return Some(false);
    }
    Some(
        on_values.contains(&effort)
            || matches!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOn),
    )
}

/// Precedence for the effective thinking preference: an explicit template
/// toggle always wins, then a native template effort for renderers that
/// support it, then the typed `thinking.type` toggle, then the provider
/// profile's reading of `reasoning_effort`
/// ([`ProviderProfile::thinking_from_reasoning_effort`]: the OpenAI mapping
/// for every profile but MiniMax, which no effort value switches off).
/// The typed rank matches where K3, V3.2 and V4.1 read `params.thinking`;
/// V4 ranks it above its native effort, so a typed `disabled` plus a native
/// effort disagrees there.
fn resolve_thinking_pref(
    explicit: Option<bool>,
    template_effort: Option<bool>,
    typed: Option<bool>,
    effort_thinking: Option<bool>,
) -> Option<bool> {
    explicit.or(template_effort).or(typed).or(effort_thinking)
}

/// Whether the reasoning parser must start in reasoning mode, i.e. whether
/// the rendered prompt ends inside `<think>`.
///
/// The effective thinking preference decides, with one exception: a renderer
/// that continues a trailing assistant message natively
/// (`continue_final_message`, see `RendererCapabilities`) renders that
/// message past its `</think>`, so the completion starts in content mode and
/// the parser must not be armed. `continues_final_assistant` says whether
/// the prompt does that. Once the prompt is rendered it is the prompt's
/// `continued_final_message`, not the request's ask: a template that cannot
/// continue the message appends its text after the generation prompt, and
/// the completion then starts wherever that prompt leaves it.
pub fn reasoning_starts_in_prefill(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    thinking: Option<bool>,
    continues_final_assistant: bool,
    tokenizer: &dyn Tokenizer,
    provider: ProviderProfile,
) -> bool {
    if continues_final_assistant
        && tokenizer
            .renderer_capabilities()
            .native_assistant_continuation
    {
        return false;
    }
    should_mark_reasoning_started(
        resolve_user_thinking(kwargs, reasoning_effort, thinking, tokenizer, provider),
        tokenizer,
    )
}

/// [`reasoning_starts_in_prefill`] for a chat request, without its rendered
/// prompt (the Go binding): a trailing assistant message is taken to be
/// continued whenever the request asks for it. The gateway decides from the
/// rendered prompt with [`chat_reasoning_starts_in_rendered_prefill`].
pub fn chat_reasoning_starts_in_prefill(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> bool {
    chat_reasoning_starts_in_rendered_prefill(
        request,
        request.continues_final_assistant(),
        tokenizer,
    )
}

/// [`reasoning_starts_in_prefill`] for a chat request whose prompt is
/// rendered: `continued_final_message` is the rendered prompt's.
pub fn chat_reasoning_starts_in_rendered_prefill(
    request: &ChatCompletionRequest,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    reasoning_starts_in_prefill(
        request.chat_template_kwargs.as_ref(),
        request.effective_reasoning_effort(),
        request.effective_thinking_toggle(),
        continued_final_message,
        tokenizer,
        ProviderProfile::for_model(&request.model),
    )
}

/// [`reasoning_starts_in_prefill`] for a Messages API request: the
/// `thinking` block is the user's preference (`enabled`/`adaptive` on,
/// `disabled` off, absent → the template's default), and
/// `continued_final_message` is the rendered prompt's (a trailing assistant
/// message with text and no tool call is continued where the template can).
/// `output_config.effort` is the `reasoning_effort` the template renders, as
/// a Chat request's effective effort is.
pub fn messages_reasoning_starts_in_prefill(
    request: &openai_protocol::messages::CreateMessageRequest,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    reasoning_starts_in_prefill(
        None,
        request.reasoning_effort(),
        request.thinking_toggle(),
        continued_final_message,
        tokenizer,
        ProviderProfile::for_model(&request.model),
    )
}

/// Where the selected worker's engine starts a request's grammar on a prompt
/// that ends inside the model's thinking block ([`WorkerDecoding::start`]).
pub fn grammar_start_for(workers: Option<&WorkerSelection>) -> GrammarStart {
    WorkerDecoding::of(workers).start
}

/// The selected worker's constrained decoding; under PD the decode worker's,
/// whose engine generates the tokens the grammar constrains. Request building
/// decides a request's constraint on it (`regular::symphony::worker_constraint`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerDecoding {
    /// Whether the engine takes a grammar on every turn or only when the
    /// request asks, from the worker's spec or its `constrained_decoding`
    /// label ([`ConstrainedDecoding::of`]); `requested` without a selection.
    pub policy: ConstrainedDecoding,
    /// Where the engine starts a grammar on a prompt that ends inside the
    /// model's thinking block, from the worker's `constrained_decoding_start`
    /// label (the servicers advertise it off the engine's config); unknown
    /// without a selection or a label.
    pub start: GrammarStart,
}

impl WorkerDecoding {
    pub fn of(workers: Option<&WorkerSelection>) -> Self {
        match workers {
            Some(WorkerSelection::Single { worker }) => Self::of_worker(worker.as_ref()),
            Some(WorkerSelection::Disaggregated { decode, .. }) => Self::of_worker(decode.as_ref()),
            None => Self {
                policy: ConstrainedDecoding::Requested,
                start: GrammarStart::Unknown,
            },
        }
    }

    fn of_worker(worker: &dyn Worker) -> Self {
        let spec = &worker.metadata().spec;
        Self {
            policy: ConstrainedDecoding::of(spec),
            start: GrammarStart::from_label(
                spec.labels.get(GrammarStart::LABEL).map(String::as_str),
            ),
        }
    }

    /// The whole turn's grammar applies: the engine takes a grammar on every
    /// turn and starts it at the first token.
    pub fn takes_the_whole_turn(self) -> bool {
        self.policy == ConstrainedDecoding::Always && self.start == GrammarStart::FirstToken
    }

    /// Whether `worker` decides a request's constraint as this one does: the
    /// same policy, and the whole turn's grammar on both or on neither. A plan
    /// whose constraint the policy chose holds a retry to such workers, since
    /// the decision reads the request, which is gone once the plan is built;
    /// the attempt stamp shapes a tag for the start of the worker the retry
    /// takes (`helpers::ToolConstraintStamp`). A prefill worker generates no
    /// token a grammar constrains.
    pub fn decides_alike(self, worker: &dyn Worker) -> bool {
        if *worker.worker_type() == WorkerType::Prefill {
            return true;
        }
        let other = Self::of_worker(worker);
        other.policy == self.policy && other.takes_the_whole_turn() == self.takes_the_whole_turn()
    }
}

/// The request's tool constraint, shaped for the worker that will run it. A
/// forced call's structural tag built on a thinking prompt opens with free
/// text that fits either kind of engine (`ParserRegistry::generate_tool_constraint`);
/// once the worker is known it takes the one shape its engine needs
/// (`ParserRegistry::structural_tag_for_grammar_start`): the bare calls for an
/// engine that starts the grammar after its reasoning parser's gate, so the
/// call is forced right there as the engine's own server does, the thought
/// block and then the calls for an engine that starts it at the first token.
/// A worker that does not say, a JSON-schema constraint and a tag without the
/// prefix pass through unchanged.
pub fn tool_constraints_for_worker(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    tool_constraints: Option<(String, String)>,
    workers: Option<&WorkerSelection>,
) -> Result<Option<(String, String)>, String> {
    let Some((kind, value)) = tool_constraints else {
        return Ok(None);
    };
    if kind != "structural_tag" {
        return Ok(Some((kind, value)));
    }
    let shaped = tool_parser_factory
        .registry()
        .structural_tag_for_grammar_start(configured_parser, &value, grammar_start_for(workers))?;
    Ok(Some((kind, shaped.unwrap_or(value))))
}

/// Resolve the user's effective thinking preference: the answer of a
/// renderer that resolves the mode by its own rule, else the generic
/// precedence of [`resolve_thinking_pref`]. `provider` reads the protocol's
/// `reasoning_effort` by its contract, both halves of it: the effort's own
/// reading, and whether an explicit effort outranks a typed `disabled`
/// toggle, applied here to `thinking` so a caller passing the raw
/// `thinking.type` arms the parser the way the prompt was rendered
/// (`ChatCompletionRequest::effective_thinking_toggle` applies the same
/// rule, so an already-effective toggle passes through unchanged).
pub fn resolve_user_thinking(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
    provider: ProviderProfile,
) -> Option<bool> {
    let thinking = thinking.filter(|&on| {
        on || reasoning_effort.is_none() || !provider.effort_outranks_thinking_disabled()
    });
    let effort_thinking = provider.thinking_from_reasoning_effort(reasoning_effort);
    if let Some(mode) = rendered_thinking_mode(
        kwargs,
        reasoning_effort,
        thinking.or(effort_thinking),
        tokenizer,
    ) {
        return Some(mode);
    }
    resolve_thinking_pref(
        extract_thinking_from_kwargs(kwargs, reasoning_effort, tokenizer),
        extract_template_effort_thinking(kwargs, reasoning_effort, tokenizer),
        thinking,
        effort_thinking,
    )
}

/// The mode a renderer with its own thinking rule renders for the request,
/// asked with the inputs the render receives (`build_chat_template_kwargs`
/// and `resolve_template_thinking` in `chat_utils` build the same for the
/// render): the request's template kwargs with the effective `reasoning_effort`
/// written over their entry under that key, whatever its type (the render's
/// merge writes the request-level field over any kwargs entry), the kwargs as
/// they are when the request carries no field, and `template_thinking`: the
/// typed toggle or the profile's reading of the effort, as the render
/// resolves it. `None` for every other renderer.
fn rendered_thinking_mode(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    template_thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    match reasoning_effort {
        Some(effort) => {
            let mut as_rendered = kwargs.cloned().unwrap_or_default();
            as_rendered.insert(
                "reasoning_effort".to_string(),
                Value::String(effort.to_string()),
            );
            tokenizer.native_thinking_mode(Some(&as_rendered), template_thinking)
        }
        None => tokenizer.native_thinking_mode(kwargs, template_thinking),
    }
}

/// Check if a reasoning parser is available for the given model
pub(crate) fn check_reasoning_parser_availability(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    create_reasoning_parser(reasoning_parser_factory, configured_parser, model).is_some()
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

/// Native Gemma4 channels derive their initial state from the rendered prompt,
/// using the same explicit, automatic or fallback parser selection as responses.
pub(crate) fn rendered_prompt_reasoning_in_prefill(
    reasoning_parser_factory: &ReasoningParserFactory,
    configured_parser: Option<&str>,
    model: &str,
    prompt: &str,
) -> Option<bool> {
    create_reasoning_parser(reasoning_parser_factory, configured_parser, model)
        .filter(|parser| parser.model_type() == "gemma4")
        .map(|_| reasoning_parser::parsers::Gemma4Parser::prompt_ends_in_reasoning(prompt))
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
    use std::sync::Arc;

    use llm_multimodal::MediaPartOrder;
    use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
    use tool_parser::ToolConstraint;

    use super::*;
    use crate::{
        routers::grpc::utils::{chat_utils, message_utils},
        worker::{BasicWorkerBuilder, WorkerType},
    };

    fn worker(labels: &[(&str, &str)]) -> WorkerSelection {
        let mut builder =
            BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        WorkerSelection::Single {
            worker: Arc::new(builder.build()),
        }
    }

    fn forced_call_on_a_thinking_prompt(factory: &ToolParserFactory) -> (String, String) {
        let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        }]))
        .unwrap();
        let required = ToolChoice::Value(ToolChoiceValue::Required);
        match factory
            .registry()
            .generate_tool_constraint(Some("glm47_moe"), &tools, &required, true)
            .unwrap()
        {
            Some(ToolConstraint::StructuralTag(tag)) => ("structural_tag".to_string(), tag),
            other => panic!("expected a structural tag, got {other:?}"),
        }
    }

    /// The worker's constrained-decoding policy comes off its spec first, then
    /// its `constrained_decoding` label; the decode worker's under PD; a
    /// missing or unreadable value, and no selection, are `requested`.
    #[test]
    fn constrained_decoding_follows_the_spec_then_the_label_of_the_decode_worker() {
        assert_eq!(
            WorkerDecoding::of(None).policy,
            ConstrainedDecoding::Requested
        );
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[("constrained_decoding", "always")]))).policy,
            ConstrainedDecoding::Always
        );
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[("constrained_decoding", "sometimes")]))).policy,
            ConstrainedDecoding::Requested
        );
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[]))).policy,
            ConstrainedDecoding::Requested
        );
        let spec_wins = BasicWorkerBuilder::new("grpc://engine:50051")
            .worker_type(WorkerType::Regular)
            .label("constrained_decoding", "requested")
            .constrained_decoding(ConstrainedDecoding::Always)
            .build();
        assert_eq!(
            WorkerDecoding::of(Some(&WorkerSelection::Single {
                worker: Arc::new(spec_wins),
            }))
            .policy,
            ConstrainedDecoding::Always
        );
        let prefill = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:50051")
                .worker_type(WorkerType::Prefill)
                .label("constrained_decoding", "always")
                .build(),
        );
        let decode = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:50051")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(
            WorkerDecoding::of(Some(&WorkerSelection::Disaggregated {
                encode_assignments: None,
                prefill,
                decode,
                runtime_type: openai_protocol::worker::RuntimeType::TokenSpeed,
            }))
            .policy,
            ConstrainedDecoding::Requested
        );
    }

    /// A retry takes only workers that decide a request's constraint as the
    /// first attempt's did: under `requested` whatever the start, under
    /// `always` the whole turn's grammar on both or on neither; and any
    /// prefill worker, which generates no token a grammar constrains.
    #[test]
    fn a_retry_takes_only_workers_that_decide_the_constraint_alike() {
        let always = ("constrained_decoding", "always");
        let labels: [&[(&str, &str)]; 5] = [
            &[],
            &[("constrained_decoding_start", "first_token")],
            &[always, ("constrained_decoding_start", "first_token")],
            &[always, ("constrained_decoding_start", "after_reasoning")],
            &[always],
        ];
        // The workers alike: both `requested` ones; the whole turn's alone;
        // the two `always` ones that start the grammar later or do not say.
        let alike = [0, 0, 1, 2, 2];
        let engine = |labels: &[(&str, &str)]| match worker(labels) {
            WorkerSelection::Single { worker } => worker,
            WorkerSelection::Disaggregated { .. } => panic!("expected one worker"),
        };
        let prefill = BasicWorkerBuilder::new("grpc://prefill:50051")
            .worker_type(WorkerType::Prefill)
            .build();
        for (first, first_labels) in labels.iter().enumerate() {
            let decided_on = WorkerDecoding::of(Some(&worker(first_labels)));
            for (other, other_labels) in labels.iter().enumerate() {
                assert_eq!(
                    decided_on.decides_alike(engine(other_labels).as_ref()),
                    alike[first] == alike[other],
                    "decided on {first_labels:?}, retried on {other_labels:?}"
                );
            }
            assert!(decided_on.decides_alike(&prefill));
        }
    }

    /// The worker's `constrained_decoding_start` label names where its engine
    /// starts the grammar; no selection, no label or another value is unknown.
    #[test]
    fn grammar_start_comes_from_the_workers_label() {
        assert_eq!(grammar_start_for(None), GrammarStart::Unknown);
        assert_eq!(grammar_start_for(Some(&worker(&[]))), GrammarStart::Unknown);
        assert_eq!(
            grammar_start_for(Some(&worker(&[(
                "constrained_decoding_start",
                "after_reasoning"
            )]))),
            GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_for(Some(&worker(&[(
                "constrained_decoding_start",
                "first_token"
            )]))),
            GrammarStart::FirstToken
        );
        assert_eq!(
            grammar_start_for(Some(&worker(&[("constrained_decoding_start", "later")]))),
            GrammarStart::Unknown
        );
    }

    /// A forced call's tag on a thinking prompt takes the shape the selected
    /// worker's engine needs: the bare calls behind a reasoning parser's gate,
    /// the thought block then the calls from the first token, and the
    /// free-text prefix it was built with where the worker does not say.
    #[test]
    fn tool_constraints_take_the_shape_the_workers_engine_needs() {
        let factory = ToolParserFactory::new();
        let parser = Some("glm47_moe");
        let built = forced_call_on_a_thinking_prompt(&factory);
        let built_tag: Value = serde_json::from_str(&built.1).unwrap();
        assert_eq!(built_tag["format"]["type"], "sequence");

        let shape = |labels: &[(&str, &str)]| -> Value {
            let (kind, value) = tool_constraints_for_worker(
                &factory,
                parser,
                Some(built.clone()),
                Some(&worker(labels)),
            )
            .unwrap()
            .expect("the constraint stays");
            assert_eq!(kind, "structural_tag");
            serde_json::from_str(&value).unwrap()
        };
        let deferred = shape(&[("constrained_decoding_start", "after_reasoning")]);
        assert_eq!(
            deferred["format"]["type"], "triggered_tags",
            "nothing precedes the calls: {deferred}"
        );
        assert_eq!(deferred["format"]["at_least_one"], true);
        let first_token = shape(&[("constrained_decoding_start", "first_token")]);
        assert_eq!(first_token["format"]["type"], "sequence");
        assert_eq!(first_token["format"]["elements"][0]["type"], "tag");
        assert_eq!(first_token["format"]["elements"][0]["end"], "</think>");
        assert_eq!(
            first_token["format"]["elements"][1],
            built_tag["format"]["elements"][1]
        );
        assert_eq!(
            shape(&[]),
            built_tag,
            "a worker that does not say keeps the shape that fits both"
        );

        // Nothing else is touched: no selection, a JSON-schema constraint, no constraint.
        assert_eq!(
            tool_constraints_for_worker(&factory, parser, Some(built.clone()), None).unwrap(),
            Some(built.clone())
        );
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert_eq!(
            tool_constraints_for_worker(
                &factory,
                parser,
                Some(schema.clone()),
                Some(&worker(&[(
                    "constrained_decoding_start",
                    "after_reasoning"
                )]))
            )
            .unwrap(),
            Some(schema)
        );
        assert_eq!(
            tool_constraints_for_worker(&factory, parser, None, Some(&worker(&[]))).unwrap(),
            None
        );
    }

    #[test]
    fn template_without_a_switch_starts_in_reasoning_when_its_prompt_opens_the_think_block() {
        use llm_tokenizer::MockTokenizer;
        // GLM-5.3: no thinking switch, the generation prompt ends in `<think>`.
        // No request field can switch thinking off, so none may disarm the parser.
        let always_thinks = MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::None)
            .with_think_in_prefill(true);
        for user_thinking in [None, Some(true), Some(false)] {
            assert!(
                should_mark_reasoning_started(user_thinking, &always_thinks),
                "{user_thinking:?}"
            );
        }
        // A non-thinking template without a switch writes an empty, closed
        // think block: the tag is in the prefill, nothing is open.
        let closed_block = MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::None)
            .with_think_in_prefill(true)
            .with_prefill_opens_think_block(false);
        for user_thinking in [None, Some(true), Some(false)] {
            assert!(
                !should_mark_reasoning_started(user_thinking, &closed_block),
                "{user_thinking:?}"
            );
        }
        // A template without a switch that leaves the think block to the model
        // starts in content mode, as before.
        let never_prefills = MockTokenizer::new().with_thinking_toggle(ThinkingToggle::None);
        for user_thinking in [None, Some(true), Some(false)] {
            assert!(!should_mark_reasoning_started(
                user_thinking,
                &never_prefills
            ));
        }
    }

    #[test]
    fn resolve_thinking_pref_precedence() {
        // Explicit toggle > native template effort > typed toggle > the profile's
        // reading of reasoning_effort (here the OpenAI one: none off, high none).
        let effort =
            |value: &str| ProviderProfile::Generic.thinking_from_reasoning_effort(Some(value));
        assert_eq!(
            resolve_thinking_pref(Some(false), Some(true), Some(true), effort("high")),
            Some(false)
        );
        assert_eq!(
            resolve_thinking_pref(None, Some(true), Some(false), effort("none")),
            Some(true)
        );
        assert_eq!(
            resolve_thinking_pref(None, None, Some(false), effort("high")),
            Some(false)
        );
        assert_eq!(
            resolve_thinking_pref(None, None, Some(true), effort("none")),
            Some(true)
        );
        assert_eq!(
            resolve_thinking_pref(None, None, None, effort("none")),
            Some(false)
        );
        assert_eq!(
            resolve_thinking_pref(None, None, None, effort("high")),
            None
        );
        assert_eq!(resolve_thinking_pref(None, None, None, None), None);
    }

    /// A kwargs `reasoning_effort` of `"none"` renders chat mode for native
    /// renderers, so it must disarm the parser too — even when the top-level
    /// field names a native level (the kwargs entry wins in the merge).
    #[test]
    fn kwargs_none_disarms_like_the_renderer() {
        let tok = T(llm_tokenizer::MockTokenizer::new());
        let none_kw = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            Value::String("none".to_string()),
        )]);
        assert_eq!(
            extract_template_effort_thinking(Some(&none_kw), None, &tok),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(Some(&none_kw), None, None, &tok, ProviderProfile::Generic),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
            Some(false)
        );
        // The request-level field is written over the kwargs entry, as the
        // engine renders it: a top-level `high` arms against a kwargs `none`.
        assert_eq!(
            extract_template_effort_thinking(Some(&none_kw), Some("high"), &tok),
            Some(true)
        );
        assert_eq!(
            resolve_user_thinking(
                Some(&none_kw),
                Some("high"),
                None,
                &tok,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        // `minimal` is the switch's other spelling and disarms the same way.
        let minimal_kw = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            Value::String("minimal".to_string()),
        )]);
        assert_eq!(
            extract_template_effort_thinking(Some(&minimal_kw), None, &tok),
            Some(false)
        );
    }

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
        fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
            Some(ThinkingKeyName::ThinkingMode)
        }
    }

    /// A tokenizer shaped like the DeepSeek-V4.1 renderer: `thinking` key,
    /// native effort names, thinking on by default, and every renderer
    /// capability declared.
    fn v41_like() -> llm_tokenizer::MockTokenizer {
        llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOn)
            .with_thinking_key_name(ThinkingKeyName::Thinking)
            .with_native_reasoning_effort_values(&["low", "high", "xhigh", "max"])
            .with_renderer_capabilities(llm_tokenizer::traits::RendererCapabilities {
                media_handling: llm_tokenizer::traits::MediaHandling::Preprocess,
                enable_thinking_alias: true,
                native_assistant_continuation: true,
                raw_tool_call_arguments: true,
                declares_tools_under_tool_choice_none: false,
                leading_system_only: false,
            })
    }

    /// The gateway arms the reasoning parser exactly the way the V4.1
    /// renderer picks the mode: `thinking` or its `enable_thinking` alias
    /// first, then the effective `reasoning_effort` (`"none"` off, a native
    /// name on), then the OpenAI mapping of the top-level field.
    #[test]
    fn v41_alias_and_kwargs_none_arm_like_the_renderer() {
        let tok = v41_like();
        let alias_off =
            std::collections::HashMap::from([("enable_thinking".to_string(), Value::Bool(false))]);
        assert_eq!(
            extract_thinking_from_kwargs(Some(&alias_off), None, &tok),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(
                Some(&alias_off),
                Some("high"),
                None,
                &tok,
                ProviderProfile::Generic
            ),
            Some(false)
        );
        // `thinking` wins when both keys are present.
        let both = std::collections::HashMap::from([
            ("thinking".to_string(), Value::Bool(true)),
            ("enable_thinking".to_string(), Value::Bool(false)),
        ]);
        assert_eq!(
            extract_thinking_from_kwargs(Some(&both), None, &tok),
            Some(true)
        );
        // Tokenizers that do not declare the alias keep reading their own key only.
        let other = T(llm_tokenizer::MockTokenizer::new());
        assert_eq!(
            extract_thinking_from_kwargs(Some(&alias_off), None, &other),
            None
        );

        // A kwargs `"none"` disarms; a top-level native level is written over
        // it, as the engine renders the request-level field.
        let none_kw = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            Value::String("none".to_string()),
        )]);
        assert_eq!(
            extract_template_effort_thinking(Some(&none_kw), None, &tok),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(Some(&none_kw), None, None, &tok, ProviderProfile::Generic),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(
                Some(&none_kw),
                Some("high"),
                None,
                &tok,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        // Native names arm; a top-level "none" disarms; an explicit toggle beats "none".
        assert_eq!(
            resolve_user_thinking(None, Some("xhigh"), None, &tok, ProviderProfile::Generic),
            Some(true)
        );
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
            Some(false)
        );
        let explicit_on = std::collections::HashMap::from([
            ("thinking".to_string(), Value::Bool(true)),
            (
                "reasoning_effort".to_string(),
                Value::String("none".to_string()),
            ),
        ]);
        assert_eq!(
            resolve_user_thinking(
                Some(&explicit_on),
                None,
                None,
                &tok,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        assert!(should_mark_reasoning_started(
            resolve_user_thinking(
                Some(&explicit_on),
                None,
                None,
                &tok,
                ProviderProfile::Generic
            ),
            &tok
        ));

        // A native continuation renders the trailing assistant message past
        // its `</think>`, so the parser is not armed for that request even
        // though thinking is on; any other trailing role arms as usual.
        let request = |continue_final: bool, last_role: &str| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": last_role, "content": "a"}
                ],
                "continue_final_message": continue_final,
            }))
            .expect("chat request")
        };
        assert!(chat_reasoning_starts_in_prefill(
            &request(false, "assistant"),
            &tok
        ));
        assert!(!chat_reasoning_starts_in_prefill(
            &request(true, "assistant"),
            &tok
        ));
        assert!(chat_reasoning_starts_in_prefill(
            &request(true, "user"),
            &tok
        ));
        // The request-level `high` is written over the kwargs `none`.
        assert!(should_mark_reasoning_started(
            resolve_user_thinking(
                Some(&none_kw),
                Some("high"),
                None,
                &tok,
                ProviderProfile::Generic
            ),
            &tok
        ));
    }

    /// A Messages prefill (trailing assistant text) continued natively is
    /// rendered past the turn's reasoning, so the parser is not armed; a
    /// trailing tool call, with or without text, opens a new turn and a
    /// popped prefill follows the generation prompt, and both arm as before.
    #[test]
    fn messages_prefill_continued_natively_does_not_arm() {
        let request = |last: Value| -> openai_protocol::messages::CreateMessageRequest {
            serde_json::from_value(serde_json::json!({
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "q"}, last]
            }))
            .expect("messages request")
        };
        let prefill = request(serde_json::json!({"role": "assistant", "content": "a"}));
        let tool_use = request(serde_json::json!({"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
        ]}));
        let text_and_tool_use = request(serde_json::json!({"role": "assistant", "content": [
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
        ]}));
        let thinking_on =
            || llm_tokenizer::MockTokenizer::new().with_thinking_toggle(ThinkingToggle::DefaultOn);
        let native =
            thinking_on().with_renderer_capabilities(llm_tokenizer::traits::RendererCapabilities {
                native_assistant_continuation: true,
                ..Default::default()
            });
        // Armed from the rendered prompt, as request building arms it.
        let armed = |request: &openai_protocol::messages::CreateMessageRequest,
                     tokenizer: &dyn Tokenizer| {
            let (processed, _) = message_utils::process_messages(
                request,
                tokenizer,
                None,
                None,
                MediaPartOrder::MediaFirst,
            )
            .expect("rendered");
            messages_reasoning_starts_in_prefill(
                request,
                processed.continued_final_message,
                tokenizer,
            )
        };

        assert!(!armed(&prefill, &native));
        assert!(armed(&tool_use, &native));
        assert!(armed(&text_and_tool_use, &native));
        assert!(armed(&prefill, &thinking_on()));
    }

    /// A generation prompt that opens `<think>` (thinking on by default) and
    /// no rendered assistant turn: a continued final message cannot be cut
    /// from the template's own rendering, so its text follows that prompt.
    const THINKING_PROMPT_NO_ASSISTANT_TURNS: &str = r"
{%- for m in messages -%}
{%- if m.role != 'assistant' -%}{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking is not defined or enable_thinking -%}{{- '<think>\n' -}}{%- endif -%}
{%- endif -%}";

    /// The prompt both APIs render for a `Let me` prefill on that template.
    const PREFILL_AFTER_THINKING_PROMPT: &str =
        "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nLet me";

    fn thinking_prompt_tokenizer() -> llm_tokenizer::TiktokenTokenizer {
        let mut tokenizer =
            llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
                .expect("tokenizer");
        tokenizer
            .set_chat_template(THINKING_PROMPT_NO_ASSISTANT_TURNS.to_string())
            .expect("template");
        tokenizer
    }

    /// The reasoning and the answer a fresh qwen3 parser, armed or not, finds
    /// in a completion that closes its reasoning and then answers.
    fn reasoning_and_answer(armed: bool) -> (String, String) {
        let factory = ReasoningParserFactory::new();
        let mut parser =
            create_reasoning_parser(&factory, Some("qwen3"), "m").expect("qwen3 parser");
        if armed {
            parser.mark_reasoning_started();
        }
        let parsed = parser
            .detect_and_parse_reasoning(" check.</think>Done")
            .expect("parsed");
        (parsed.reasoning_text, parsed.normal_text)
    }

    /// Where the template cannot continue a Messages prefill, the prompt ends
    /// inside the `<think>` its generation prompt opens. The parser is armed
    /// from the rendered prompt, not from the request asking to continue, so
    /// the completion's reasoning is parsed as reasoning.
    #[test]
    fn messages_continuation_after_a_thinking_generation_prompt_arms() {
        let tokenizer = thinking_prompt_tokenizer();
        let request: openai_protocol::messages::CreateMessageRequest =
            serde_json::from_value(serde_json::json!({
                "model": "m",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "Let me"}
                ]
            }))
            .expect("messages request");
        let (processed, _) = message_utils::process_messages(
            &request,
            &tokenizer,
            None,
            None,
            MediaPartOrder::MediaFirst,
        )
        .expect("rendered");
        assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
        assert!(!processed.continued_final_message);
        let armed = messages_reasoning_starts_in_prefill(
            &request,
            processed.continued_final_message,
            &tokenizer,
        );
        assert_eq!(
            reasoning_and_answer(armed),
            (" check.".to_string(), "Done".to_string())
        );
    }

    /// The same for a chat `continue_final_message`.
    #[test]
    fn chat_continuation_after_a_thinking_generation_prompt_arms() {
        let tokenizer = thinking_prompt_tokenizer();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "Let me"}
            ],
            "continue_final_message": true
        }))
        .expect("chat request");
        let (processed, _) = chat_utils::process_chat_messages_with_placeholders(
            &request,
            &tokenizer,
            None,
            MediaPartOrder::MediaFirst,
        )
        .expect("rendered");
        assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
        assert!(!processed.continued_final_message);
        let armed = chat_reasoning_starts_in_rendered_prefill(
            &request,
            processed.continued_final_message,
            &tokenizer,
        );
        assert_eq!(
            reasoning_and_answer(armed),
            (" check.".to_string(), "Done".to_string())
        );
    }

    /// Typed toggle: below an explicit kwargs toggle and a native effort, above the OpenAI mapping.
    #[test]
    fn typed_thinking_toggle_ranks_between_kwargs_and_effort() {
        let k3 = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOn)
            .with_thinking_key_name(ThinkingKeyName::Thinking);

        // KVV `thinking:{type:"disabled"}`: chat mode, parser not armed.
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("max"),
                Some(false),
                &k3,
                ProviderProfile::Generic
            ),
            Some(false)
        );
        assert!(!should_mark_reasoning_started(
            resolve_user_thinking(
                None,
                Some("max"),
                Some(false),
                &k3,
                ProviderProfile::Generic
            ),
            &k3
        ));
        // Absent `thinking`, K3 stays thinking-on by default.
        assert!(should_mark_reasoning_started(
            resolve_user_thinking(None, None, None, &k3, ProviderProfile::Generic),
            &k3
        ));
        // An explicit kwargs toggle outranks the typed one.
        let kw_on = std::collections::HashMap::from([("thinking".to_string(), Value::Bool(true))]);
        assert_eq!(
            resolve_user_thinking(
                Some(&kw_on),
                None,
                Some(false),
                &k3,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        // The typed toggle outranks the OpenAI mapping; absent it, the mapping applies.
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("none"),
                Some(true),
                &k3,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &k3, ProviderProfile::Generic),
            Some(false)
        );

        // A native template effort outranks the typed toggle (DeepSeek-V4.1).
        let v41 = v41_like();
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("high"),
                Some(false),
                &v41,
                ProviderProfile::Generic
            ),
            Some(true)
        );
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("medium"),
                Some(false),
                &v41,
                ProviderProfile::Generic
            ),
            Some(false)
        );

        // End to end through the chat request: `thinking.effort` is the effective effort.
        let request = |thinking: Value| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "q"}],
                "thinking": thinking,
                "reasoning_effort": "none",
            }))
            .expect("chat request")
        };
        assert!(chat_reasoning_starts_in_prefill(
            &request(serde_json::json!({"type": "enabled"})),
            &k3
        ));
        assert!(!chat_reasoning_starts_in_prefill(
            &request(serde_json::json!({"type": "disabled"})),
            &k3
        ));
        assert!(chat_reasoning_starts_in_prefill(
            &request(serde_json::json!({"effort": "high"})),
            &v41
        ));
    }

    /// A K3 directory with a byte-complete stub vocabulary (every byte one
    /// token, no merges) and the real renderer: prompts render and encode
    /// without the checkpoint's files.
    fn k3_stub() -> (tempfile::TempDir, llm_tokenizer::TiktokenTokenizer) {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let dir = tempfile::TempDir::new().unwrap();
        let vocabulary: String = (0..=255u8)
            .map(|byte| format!("{} {byte}\n", BASE64.encode([byte])))
            .collect();
        std::fs::write(dir.path().join("tiktoken.model"), vocabulary).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures": ["KimiK3ForConditionalGeneration"]}"#,
        )
        .unwrap();
        let tokenizer = llm_tokenizer::TiktokenTokenizer::from_dir(dir.path()).unwrap();
        (dir, tokenizer)
    }

    /// The K3 parser is armed from the decision the prompt was rendered
    /// with, whatever the request's mix of typed toggle, top-level effort and
    /// kwargs: the renderer's own rule decides both (an effort word other
    /// than `none` leaves the mode to the typed toggle, `none` switches it
    /// off, a kwargs `minimal` keeps the default on).
    #[test]
    fn k3_parser_is_armed_the_way_the_prompt_was_rendered() {
        use crate::routers::grpc::utils::process_chat_messages;

        let (_dir, tokenizer) = k3_stub();
        let request = |fields: Value| -> ChatCompletionRequest {
            let mut body = serde_json::json!({
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": "q"}],
            });
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            serde_json::from_value(body).expect("chat request")
        };
        let armed_like_the_prompt = |fields: Value| -> bool {
            let request = request(fields.clone());
            let prompt = process_chat_messages(&request, &tokenizer, None)
                .expect("render")
                .text;
            let opens_think = prompt.ends_with("<|open|>think<|sep|>");
            assert!(
                opens_think || prompt.ends_with("<|open|>response<|sep|>"),
                "{prompt}"
            );
            assert_eq!(
                chat_reasoning_starts_in_prefill(&request, &tokenizer),
                opens_think,
                "{fields}: prompt {prompt:?}"
            );
            opens_think
        };
        // Typed disabled plus an effort word: chat mode on both sides.
        assert!(!armed_like_the_prompt(serde_json::json!({
            "thinking": {"type": "disabled"}, "reasoning_effort": "high"
        })));
        // A kwargs `minimal` is not the renderer's off word: thinking stays on.
        assert!(armed_like_the_prompt(serde_json::json!({
            "chat_template_kwargs": {"reasoning_effort": "minimal"}
        })));
        // `none` switches thinking off, even against a typed enabled.
        assert!(!armed_like_the_prompt(serde_json::json!({
            "chat_template_kwargs": {"reasoning_effort": "none"}, "thinking": {"type": "enabled"}
        })));
        assert!(!armed_like_the_prompt(
            serde_json::json!({"reasoning_effort": "none"})
        ));
        // The top-level `minimal` reaches the renderer as the protocol's off toggle.
        assert!(!armed_like_the_prompt(
            serde_json::json!({"reasoning_effort": "minimal"})
        ));
        assert!(armed_like_the_prompt(
            serde_json::json!({"reasoning_effort": "high"})
        ));
        assert!(!armed_like_the_prompt(serde_json::json!({
            "chat_template_kwargs": {"thinking": false, "reasoning_effort": "high"}
        })));
        // The request-level effort is written over a kwargs entry of any type
        // (the render's merge does the same): the top-level `none` replaces
        // the null entry, so the response channel opens on both sides.
        assert!(!armed_like_the_prompt(serde_json::json!({
            "thinking": {"type": "enabled"}, "reasoning_effort": "none",
            "chat_template_kwargs": {"reasoning_effort": null}
        })));
        assert!(armed_like_the_prompt(serde_json::json!({})));
    }

    #[test]
    fn template_effort_thinking_covers_kwargs_and_top_level_field() {
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
        // The top-level field wins over a kwargs entry, matching the merge.
        let kwargs = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            Value::String("max".to_string()),
        )]);
        assert_eq!(
            extract_template_effort_thinking(Some(&kwargs), Some("medium"), &tok),
            None
        );
        assert_eq!(
            extract_template_effort_thinking(Some(&kwargs), None, &tok),
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

    /// The MiniMax profile reads `reasoning_effort` by its contract: no value
    /// switches thinking off, so the parser's arming follows the template's
    /// default as the render does; the OpenAI reading stays for every other
    /// profile.
    #[test]
    fn minimax_effort_never_switches_the_arming_off() {
        // A tokenizer like the MiniMax M3 template's: the tri-state key, off by
        // default, and no native `reasoning_effort` words of its own.
        let tok = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOff)
            .with_thinking_key_name(ThinkingKeyName::ThinkingMode);
        for effort in ["none", "minimal", "low", "high"] {
            assert_eq!(
                resolve_user_thinking(None, Some(effort), None, &tok, ProviderProfile::Minimax),
                None,
                "MiniMax {effort}"
            );
        }
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Kimi),
            Some(false)
        );
        // The typed toggle, as the profile read it, still ranks above the effort.
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("none"),
                Some(true),
                &tok,
                ProviderProfile::Minimax
            ),
            Some(true)
        );
        // The profile's other half applies here too: a raw `disabled` next to
        // an explicit effort is demoted under MiniMax, honoured elsewhere.
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("high"),
                Some(false),
                &tok,
                ProviderProfile::Minimax
            ),
            None
        );
        assert_eq!(
            resolve_user_thinking(None, None, Some(false), &tok, ProviderProfile::Minimax),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("high"),
                Some(false),
                &tok,
                ProviderProfile::Generic
            ),
            Some(false)
        );
    }

    #[test]
    fn thinking_mode_kwargs_map_to_tristate_pref() {
        let tok = T(llm_tokenizer::MockTokenizer::new());
        let kw = |v: Value| std::collections::HashMap::from([("thinking_mode".to_string(), v)]);

        assert_eq!(
            extract_thinking_from_kwargs(
                Some(&kw(Value::String("enabled".to_string()))),
                None,
                &tok
            ),
            Some(true)
        );
        assert_eq!(
            extract_thinking_from_kwargs(
                Some(&kw(Value::String("disabled".to_string()))),
                None,
                &tok
            ),
            Some(false)
        );
        assert_eq!(
            extract_thinking_from_kwargs(
                Some(&kw(Value::String("adaptive".to_string()))),
                None,
                &tok
            ),
            None
        );
        // Non-string values are not a preference for the tri-state key.
        assert_eq!(
            extract_thinking_from_kwargs(Some(&kw(Value::Bool(true))), None, &tok),
            None
        );
        assert_eq!(extract_thinking_from_kwargs(None, None, &tok), None);
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
        assert!(check_reasoning_parser_availability(
            &factory,
            Some("qwen3"),
            "unknown-model",
        ));
        for configured in [None, Some("unknown-parser")] {
            assert!(!check_reasoning_parser_availability(
                &factory,
                configured,
                "unknown-model",
            ));
        }
    }

    #[test]
    fn gemma4_rendered_prompt_state_follows_actual_parser_selection() {
        let factory = ReasoningParserFactory::new();
        for configured in [Some("gemma4"), None, Some("unknown-parser")] {
            assert!(check_reasoning_parser_availability(
                &factory,
                configured,
                "google/gemma-4-serve",
            ));
            for (prompt, expected) in [
                ("<|im_start|>model\n", false),
                ("<|im_start|>model<|channel>thought\n", true),
                ("<|im_start|>model<|channel>thought\nplan<channel|>", false),
            ] {
                assert_eq!(
                    rendered_prompt_reasoning_in_prefill(
                        &factory,
                        configured,
                        "google/gemma-4-serve",
                        prompt,
                    ),
                    Some(expected),
                    "configured={configured:?}, prompt={prompt:?}"
                );
            }
        }
        // A valid override wins over the model; other models keep their defaults.
        for (configured, model) in [(Some("qwen3"), "gemma-4"), (None, "qwen3")] {
            assert_eq!(
                rendered_prompt_reasoning_in_prefill(
                    &factory,
                    configured,
                    model,
                    "<|channel>thought\n",
                ),
                None
            );
        }
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

#[cfg(test)]
mod hy_v4_tests {
    use super::*;
    #[test]
    fn hy4_effort_arms_parser_like_template() {
        let tok = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOn)
            .with_thinking_key_name(ThinkingKeyName::ReasoningEffort)
            .with_native_reasoning_effort_values(&["high"])
            .with_native_reasoning_effort_off_values(&["no_think"]);
        assert!(should_mark_reasoning_started(None, &tok));
        assert_eq!(
            extract_template_effort_thinking(None, Some("no_think"), &tok),
            Some(false)
        );
        let kwargs = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            serde_json::json!("high"),
        )]);
        assert_eq!(
            extract_thinking_from_kwargs(Some(&kwargs), None, &tok),
            Some(true)
        );
        assert_eq!(
            extract_template_effort_thinking(Some(&kwargs), None, &tok),
            Some(true)
        );
        // The request-level off word is written over the kwargs entry.
        assert_eq!(
            extract_template_effort_thinking(Some(&kwargs), Some("no_think"), &tok),
            Some(false)
        );
        // The parser follows the value the template sees: the request-level
        // off word over the kwargs `high` disarms it, as the prompt renders
        // thinking off; the kwargs entry alone arms it.
        assert_eq!(
            extract_thinking_from_kwargs(Some(&kwargs), Some("no_think"), &tok),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(
                Some(&kwargs),
                Some("no_think"),
                None,
                &tok,
                ProviderProfile::Generic
            ),
            Some(false)
        );
        assert_eq!(
            resolve_user_thinking(Some(&kwargs), None, None, &tok, ProviderProfile::Generic),
            Some(true)
        );
        // The template renders its default for a value it does not know, so
        // the protocol's `none` must not disarm the parser here.
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
            Some(true)
        );
        let unknown = std::collections::HashMap::from([(
            "reasoning_effort".to_string(),
            serde_json::json!("none"),
        )]);
        assert_eq!(
            resolve_user_thinking(Some(&unknown), None, None, &tok, ProviderProfile::Generic),
            Some(true)
        );
    }
}

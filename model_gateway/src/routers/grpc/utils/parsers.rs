//! Reasoning and tool parser helpers.

use std::sync::Arc;

use llm_tokenizer::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::Tokenizer,
};
/// The structural-tag vocabulary label lives beside `ConstrainedDecoding` in the
/// protocol crate; it is re-exported here, where the routers read it.
pub use openai_protocol::worker::TagVocabulary;
use openai_protocol::{
    chat::{thinking_from_reasoning_effort, ChatCompletionRequest},
    model_card::ModelCard,
    profile::ProviderProfile,
    worker::{ConstrainedDecoding, GrammarStart},
};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ReasoningParser};
use serde_json::Value;
use smg_worker::identity::basic::{Worker, WorkerType};
use tool_parser::{
    ParserFactory as ToolParserFactory, PooledParser as ToolPooledParser, ToolParser,
};
use tracing::warn;

use crate::routers::{common::snapshot::WorkerRegistry, grpc::context::WorkerSelection};

/// Per-request parser-name resolution.
///
/// For the legacy parsers, each kind's name is the model's `ModelCard`
/// override (`tool_parser` / `reasoning_parser`, populated from worker labels
/// or an explicit `WorkerSpec` card) → the process-wide configured name
/// (`--tool-call-parser` / `--reasoning-parser`), when a legacy parser of the
/// kind has it ([`Self::new`] asks the two registries once: under `--parser
/// symphony` a flag may name a Symphony table or the response template, which
/// no legacy parser reads) → `None`, which lets the factory helpers fall back
/// to their name-based auto-detection, unchanged. Symphony takes the two
/// sources apart (`names_of`), the flags as set, with its registry
/// between them (`regular::symphony::names::reads`).
///
/// Lookups borrow straight from worker metadata (no card clones); only the
/// resolved name is cloned.
#[derive(Clone)]
pub struct ParserResolver {
    /// `None` disables card lookups (parser-free endpoints, tests).
    worker_registry: Option<Arc<WorkerRegistry>>,
    /// The gateway's `--tool-call-parser` and `--reasoning-parser`.
    flags: Names,
    /// The flags as the legacy parsers take them: each kind's when a legacy
    /// parser of the kind has the name ([`Self::new`]); the flags themselves
    /// for the tests' `with_flags`.
    legacy_flags: Names,
}

impl ParserResolver {
    /// The gateway's resolver: `worker_registry` for the cards, the configured
    /// names, and the two legacy registries, asked once which of the
    /// configured names they have. Under `--parser symphony` a flag may name a
    /// Symphony table or the response template, which select for Symphony and
    /// are nothing to the legacy parsers; handed such a name (by a request
    /// Symphony leaves to them: a worker labeled `passthrough` alone, or the
    /// `X-SMG-Parser: legacy` header), the legacy tool side would not parse at
    /// all, since its availability check asks the registry for the exact name
    /// and never falls back to the model (the calls come back as content), and
    /// the reasoning side would warn on every request and detect from the
    /// model anyway. So such a name is dropped here, once, and the legacy
    /// parsers' own detection reads the model. Under `--parser legacy` the
    /// startup check refuses such a flag, so the flags pass whole.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
        tools: &ToolParserFactory,
        reasoning: &ReasoningParserFactory,
    ) -> Self {
        let flags = Names {
            tool: configured_tool_parser,
            reasoning: configured_reasoning_parser,
        };
        let legacy_flags = Names {
            tool: flags.tool.clone().filter(|name| tools.has_parser(name)),
            reasoning: flags
                .reasoning
                .clone()
                .filter(|name| reasoning.registry().has_parser(name)),
        };
        Self {
            worker_registry: Some(worker_registry),
            flags,
            legacy_flags,
        }
    }

    /// A resolver whose configured names pass to the legacy parsers whole, the
    /// registries unasked: for tests, whose names are the legacy parsers' own,
    /// or which assert Symphony's view of the flags ([`Self::names`]).
    #[cfg(test)]
    pub(crate) fn with_flags(
        worker_registry: Arc<WorkerRegistry>,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let flags = Names {
            tool: configured_tool_parser,
            reasoning: configured_reasoning_parser,
        };
        Self {
            worker_registry: Some(worker_registry),
            legacy_flags: flags.clone(),
            flags,
        }
    }

    /// Resolver that never consults model cards and carries no configured
    /// names — preserves the parser-free endpoints' behavior.
    pub(crate) fn disabled() -> Self {
        Self {
            worker_registry: None,
            flags: Names::default(),
            legacy_flags: Names::default(),
        }
    }

    /// Effective tool-parser name for `model`, if any.
    pub fn tool_parser(&self, model: &str) -> Option<String> {
        self.card_name(model, ParserKind::Tool)
            .or_else(|| self.legacy_flags.tool.clone())
    }

    /// Effective reasoning-parser name for `model`, if any.
    pub fn reasoning_parser(&self, model: &str) -> Option<String> {
        self.card_name(model, ParserKind::Reasoning)
            .or_else(|| self.legacy_flags.reasoning.clone())
    }

    /// The tool-call parser's name as the legacy parsers take it for a request
    /// whose names are `names`: the labels', else the flags' when a legacy
    /// parser has it ([`Self::tool_parser`] without another card lookup).
    pub(crate) fn legacy_tool<'a>(&'a self, names: &'a ModelNames<'_>) -> Option<&'a str> {
        names
            .labels
            .tool
            .as_deref()
            .or(self.legacy_flags.tool.as_deref())
    }

    /// [`Self::names_of`] over this resolver's own registry walk, for the
    /// tests: a request's readers take the names its preparation resolved
    /// ([`Self::names_for`]).
    #[cfg(test)]
    pub(crate) fn names(&self, model: &str) -> ModelNames<'_> {
        ModelNames {
            labels: self.card_names(model),
            flags: &self.flags,
        }
    }

    /// The parser names set for a model, apart by who set them: its cards'
    /// (its workers' labels), `cards` as [`Self::card_names_among`] read them
    /// among the model's workers, which a resolver without a registry does
    /// not read; and the gateway's flags as set. A request's preparation
    /// hands in the card names the registry's membership snapshot keeps of
    /// that walk, so no request walks the workers for them. Symphony reads
    /// the two in that order with its registry between
    /// (`regular::symphony::names::reads`); the legacy parsers take each
    /// kind's first, from the flags only a name they read
    /// ([`Self::tool_parser`], [`Self::reasoning_parser`],
    /// [`Self::legacy_tool`]).
    pub(crate) fn names_of(&self, cards: Names) -> ModelNames<'_> {
        ModelNames {
            labels: if self.worker_registry.is_some() {
                cards
            } else {
                Names::default()
            },
            flags: &self.flags,
        }
    }

    /// [`Self::names_of`] for a request whose preparation resolved the card names once
    /// (`ProcessingState::parser_names`): `resolved` as they were then, with the flags, and no
    /// second walk over the model's workers; `None` resolves them now, over the registry's walk.
    pub(crate) fn names_for(&self, resolved: Option<&Names>, model: &str) -> ModelNames<'_> {
        ModelNames {
            labels: resolved.map_or_else(|| self.card_names(model), Names::clone),
            flags: &self.flags,
        }
    }

    /// [`Self::tool_parser`] for a request whose card names were resolved once
    /// ([`Self::names_for`]).
    pub(crate) fn tool_parser_for(&self, resolved: Option<&Names>, model: &str) -> Option<String> {
        match resolved {
            Some(names) => names
                .tool
                .clone()
                .or_else(|| self.legacy_flags.tool.clone()),
            None => self.tool_parser(model),
        }
    }

    /// [`Self::reasoning_parser`] for a request whose card names were resolved once
    /// ([`Self::names_for`]).
    pub(crate) fn reasoning_parser_for(
        &self,
        resolved: Option<&Names>,
        model: &str,
    ) -> Option<String> {
        match resolved {
            Some(names) => names
                .reasoning
                .clone()
                .or_else(|| self.legacy_flags.reasoning.clone()),
            None => self.reasoning_parser(model),
        }
    }

    /// The names `model`'s own cards set (worker labels), without the process-wide fallback:
    /// what the workers say of the model, each `None` when no card names one.
    fn card_names(&self, model: &str) -> Names {
        self.worker_registry
            .as_ref()
            .map_or_else(Names::default, |registry| {
                Self::card_names_among(&registry.get_by_model(model), model)
            })
    }

    /// One kind's name among `model`'s cards, for a reader that asks for one
    /// kind: the same walk and the same conflict rule as [`Self::card_names`],
    /// cloning only the name asked for.
    fn card_name(&self, model: &str, kind: ParserKind) -> Option<String> {
        self.worker_registry
            .as_ref()
            .and_then(|registry| Self::card_name_among(&registry.get_by_model(model), model, kind))
    }

    /// Both names `model`'s cards set among `workers`, in one walk. Cards
    /// built by the label pipeline agree across the workers of one model; where
    /// they do not (mixed labels, e.g. mid rolling-upgrade) each kind takes
    /// the lexicographically smallest name, so the choice does not depend on
    /// the registry's iteration order. Registration logs the conflict as a
    /// warning; here it is debug.
    pub(crate) fn card_names_among(workers: &[Arc<dyn Worker>], model: &str) -> Names {
        let mut tool = CardPick::default();
        let mut reasoning = CardPick::default();
        for worker in workers {
            let Some(card) = worker.metadata().spec.models.find(model) else {
                continue;
            };
            tool.offer(ParserKind::Tool.of(card));
            reasoning.offer(ParserKind::Reasoning.of(card));
        }
        Names {
            tool: tool.chosen(model, ParserKind::Tool.label()),
            reasoning: reasoning.chosen(model, ParserKind::Reasoning.label()),
        }
    }

    /// [`Self::card_names_among`] for one kind: the one pick over the same
    /// walk, so a single reader clones nothing it did not ask for.
    fn card_name_among(
        workers: &[Arc<dyn Worker>],
        model: &str,
        kind: ParserKind,
    ) -> Option<String> {
        let mut pick = CardPick::default();
        for worker in workers {
            let Some(card) = worker.metadata().spec.models.find(model) else {
                continue;
            };
            pick.offer(kind.of(card));
        }
        pick.chosen(model, kind.label())
    }
}

/// Which of a card's two parser names a reader asks for.
#[derive(Clone, Copy)]
enum ParserKind {
    Tool,
    Reasoning,
}

impl ParserKind {
    /// The card's name of this kind, borrowed.
    fn of(self, card: &ModelCard) -> Option<&String> {
        match self {
            Self::Tool => card.tool_parser.as_ref(),
            Self::Reasoning => card.reasoning_parser.as_ref(),
        }
    }

    /// The kind's word in the conflict log.
    fn label(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Reasoning => "reasoning",
        }
    }
}

/// One parser kind's choice over a model's cards: the name the cards agree
/// on, or the lexicographically smallest where they conflict.
#[derive(Default)]
struct CardPick<'a> {
    chosen: Option<&'a String>,
    conflict: bool,
}

impl<'a> CardPick<'a> {
    fn offer(&mut self, name: Option<&'a String>) {
        let Some(name) = name else {
            return;
        };
        match self.chosen {
            None => self.chosen = Some(name),
            Some(existing) if existing != name => {
                self.conflict = true;
                if name < existing {
                    self.chosen = Some(name);
                }
            }
            Some(_) => {}
        }
    }

    fn chosen(self, model: &str, kind: &'static str) -> Option<String> {
        if self.conflict {
            tracing::debug!(
                model,
                kind,
                chosen = self.chosen.map(String::as_str),
                "Workers for this model declare conflicting parser overrides; \
                 using the lexicographically smallest"
            );
        }
        self.chosen.cloned()
    }
}

/// The parser names one source sets for a model: the tool-call parser's and the reasoning
/// parser's, each `None` when nothing names one. A source is a model's card (its workers' labels,
/// or the operator's card for it) or the gateway's `--tool-call-parser` and `--reasoning-parser`
/// flags, which every model shares; [`ModelNames`] holds a model's two.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Names {
    pub tool: Option<String>,
    pub reasoning: Option<String>,
}

/// No names at all.
static NO_NAMES: Names = Names {
    tool: None,
    reasoning: None,
};

/// The parser names set for one model, apart by who set them ([`ParserResolver::names_of`]): the
/// names of its card, which its workers' labels (or the operator's card) set for the one model,
/// and the gateway's flags, which every model shares. Symphony reads the two in that order ahead
/// of its registry (`regular::symphony::names::reads`): the card's names when they select
/// anything, else the flags when they select anything (a reasoning name alone that selects a table
/// excepted, `regular::symphony::names::flags_read_first`), else the registry's table for a model it
/// knows; the legacy parsers
/// take each kind's first name, from the flags only one a legacy parser of the kind has
/// ([`ParserResolver::tool_parser`], [`ParserResolver::legacy_tool`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModelNames<'a> {
    pub labels: Names,
    pub flags: &'a Names,
}

impl Default for ModelNames<'_> {
    fn default() -> Self {
        Self {
            labels: Names::default(),
            flags: &NO_NAMES,
        }
    }
}

#[cfg(test)]
impl ModelNames<'_> {
    /// The names a model's workers label it with, and no flags.
    pub(crate) fn labeled(labels: Names) -> ModelNames<'static> {
        ModelNames {
            labels,
            flags: &NO_NAMES,
        }
    }

    /// The gateway's flags, for a model whose workers label nothing.
    pub(crate) fn flagged(flags: &Names) -> ModelNames<'_> {
        ModelNames {
            labels: Names::default(),
            flags,
        }
    }
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
        // Renderers that honour vLLM's `enable_thinking` alias (DeepSeek-V3.2,
        // V4 and V4.1) read it too; `thinking` wins when both are present, the
        // order their shims render in (V3.2 and V4 resolve a disagreeing pair
        // silently that way; V4.1 rejects it before anything is dispatched).
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
/// is written over an explicit kwargs entry, as sent
/// ([`reasoning_effort_as_rendered`] names where the MiniMax render differs).
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

/// The `reasoning_effort` a renderer with its own effort words sees: the
/// request-level field (`thinking.effort`, else the top-level field), written
/// over a `chat_template_kwargs` entry as the engine's own server renders it,
/// else the entry. The render's merge (`build_chat_template_kwargs` in
/// `chat_utils`) writes the same under every profile but MiniMax, where it
/// writes the profile's reading of the field instead; no MiniMax template
/// declares effort words, so the arming reads the same there.
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
/// for every profile but MiniMax, where `none` alone switches off).
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

/// `reasoning_starts_in_prefill` for a chat request, without its rendered
/// prompt (the Go binding): a trailing assistant message is taken to be
/// continued whenever the request asks for it, and the generation header to
/// be rendered unless the request asks for none. The gateway decides from the
/// rendered prompt with `chat_reasoning_starts_in_rendered_prefill`.
pub fn chat_reasoning_starts_in_prefill(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> bool {
    let continues_final_assistant = request.continues_final_assistant();
    chat_reasoning_starts_in_rendered_prefill(
        request,
        request.adds_generation_prompt() || continues_final_assistant,
        continues_final_assistant,
        tokenizer,
    )
}

/// `reasoning_starts_in_prefill` for a chat request whose prompt is
/// rendered: `generation_prompt` and `continued_final_message` are the
/// rendered prompt's (`ProcessedMessages`), not the request's. A renderer
/// with an encoder of its own writes the generation header whatever the
/// request asked, as the engines' own encoders do, and its prompt arms as
/// before; a prompt rendered without the header (`add_generation_prompt:
/// false`, honoured by the renderer, and no continued message) opens no
/// `<think>`: the completion starts in content mode after the closed last
/// message.
pub fn chat_reasoning_starts_in_rendered_prefill(
    request: &ChatCompletionRequest,
    generation_prompt: bool,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    chat_reasoning_starts_in_rendered_prefill_for(
        request,
        ProviderProfile::for_model(&request.model),
        generation_prompt,
        continued_final_message,
        tokenizer,
    )
}

/// [`chat_reasoning_starts_in_rendered_prefill`] with the model's provider
/// profile already derived: the preparation stage derives it once per request
/// and the later reads take that copy.
pub fn chat_reasoning_starts_in_rendered_prefill_for(
    request: &ChatCompletionRequest,
    provider: ProviderProfile,
    generation_prompt: bool,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    if !generation_prompt && !continued_final_message {
        return false;
    }
    reasoning_starts_in_prefill(
        request.chat_template_kwargs.as_ref(),
        request.effective_reasoning_effort(),
        request.thinking_toggle(),
        continued_final_message,
        tokenizer,
        provider,
    )
}

/// [`messages_reasoning_starts_in_prefill_for`] with the model's provider
/// profile derived here, as the tests ask it.
#[cfg(test)]
pub fn messages_reasoning_starts_in_prefill(
    request: &openai_protocol::messages::CreateMessageRequest,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    messages_reasoning_starts_in_prefill_for(
        request,
        ProviderProfile::for_model(&request.model),
        continued_final_message,
        tokenizer,
    )
}

/// [`reasoning_starts_in_prefill`] for a Messages API request, with the
/// model's provider profile already derived (the preparation stage derives it
/// once per request): the `thinking` block is the user's preference
/// (`enabled`/`adaptive` on, `disabled` off, absent → the template's default),
/// and `continued_final_message` is the rendered prompt's (a trailing
/// assistant message with text and no tool call is continued where the
/// template can). `output_config.effort` is the `reasoning_effort` the
/// template renders, as a Chat request's effective effort is. The tests ask
/// through `messages_reasoning_starts_in_prefill`, which derives the profile
/// first.
pub fn messages_reasoning_starts_in_prefill_for(
    request: &openai_protocol::messages::CreateMessageRequest,
    provider: ProviderProfile,
    continued_final_message: bool,
    tokenizer: &dyn Tokenizer,
) -> bool {
    reasoning_starts_in_prefill(
        None,
        request.reasoning_effort(),
        request.thinking_toggle(),
        continued_final_message,
        tokenizer,
        provider,
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
    /// The vocabulary of structural-tag nodes the engine's grammar compiler
    /// takes, from the worker's `structural_tag_vocabulary` label; the
    /// current one without a selection or a label.
    pub vocabulary: TagVocabulary,
}

impl WorkerDecoding {
    pub fn of(workers: Option<&WorkerSelection>) -> Self {
        match workers {
            Some(WorkerSelection::Single { worker }) => Self::of_worker(&**worker),
            Some(WorkerSelection::Disaggregated { decode, .. }) => Self::of_worker(&**decode),
            None => Self {
                policy: ConstrainedDecoding::Requested,
                start: GrammarStart::Unknown,
                vocabulary: TagVocabulary::Current,
            },
        }
    }

    /// The worker's labels as its metadata parsed them at construction
    /// (`WorkerMetadata::decoding`): no lookup on a request path.
    fn of_worker(worker: &dyn Worker) -> Self {
        let labels = worker.metadata().decoding;
        Self {
            policy: labels.policy,
            start: labels.start,
            vocabulary: labels.vocabulary,
        }
    }

    /// The whole turn's grammar applies: the engine takes a grammar on every
    /// turn and starts it at the first token.
    pub fn takes_the_whole_turn(self) -> bool {
        self.policy == ConstrainedDecoding::Always && self.start == GrammarStart::FirstToken
    }

    /// Whether `worker` decides a request's constraint as this one does: the
    /// same policy, the whole turn's grammar on both or on neither, and, where
    /// the whole turn's applies, the same vocabulary of structural-tag nodes.
    /// A plan whose constraint the policy chose holds a retry to such workers,
    /// since the decision reads the request, which is gone once the plan is
    /// built; the attempt stamp shapes a tag for the start of the worker the
    /// retry takes (`helpers::ToolConstraintStamp`), and the whole turn's tag,
    /// spelled once for the first worker's compiler, for its start only. A
    /// prefill worker generates no token a grammar constrains.
    pub fn decides_alike(self, worker: &dyn Worker) -> bool {
        if *worker.worker_type() == WorkerType::Prefill {
            return true;
        }
        let other = Self::of_worker(worker);
        other.policy == self.policy
            && other.takes_the_whole_turn() == self.takes_the_whole_turn()
            && (!self.takes_the_whole_turn() || other.vocabulary == self.vocabulary)
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
/// prefix pass through unchanged. The flag beside the constraint says the
/// shaping led the tag with the thought block for an engine that starts the
/// grammar at the first token: the engine's request then says the grammar
/// starts there whatever reasoning parser the engine runs
/// (`ProtoGenerateRequest::set_grammar_from_first_token`); a constraint that
/// passed through unchanged starts where the engine does.
pub fn tool_constraints_for_worker(
    tool_parser_factory: &ToolParserFactory,
    configured_parser: Option<&str>,
    tool_constraints: Option<(String, String)>,
    workers: Option<&WorkerSelection>,
) -> Result<(Option<(String, String)>, bool), String> {
    let Some((kind, value)) = tool_constraints else {
        return Ok((None, false));
    };
    if kind != "structural_tag" {
        return Ok((Some((kind, value)), false));
    }
    let start = grammar_start_for(workers);
    let shaped = tool_parser_factory
        .registry()
        .structural_tag_for_grammar_start(configured_parser, &value, start)?;
    let first_token = shaped.is_some() && start == GrammarStart::FirstToken;
    Ok((Some((kind, shaped.unwrap_or(value))), first_token))
}

/// Resolve the user's effective thinking preference: the answer of a
/// renderer that resolves the mode by its own rule, else the generic
/// precedence of [`resolve_thinking_pref`]. `provider` reads the protocol's
/// `reasoning_effort` by its contract; the typed `thinking.type` toggle
/// ranks above that reading under every profile (a renderer's native
/// template effort still ranks above the toggle, see
/// [`resolve_thinking_pref`]), so under a renderer without native efforts an
/// explicit `disabled` next to an effort disarms the parser the way the
/// prompt was rendered (`ChatCompletionRequest::template_thinking` resolves
/// the render the same way).
pub fn resolve_user_thinking(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
    provider: ProviderProfile,
) -> Option<bool> {
    let effort_thinking = provider.thinking_from_reasoning_effort(reasoning_effort);
    if let Some(mode) = rendered_thinking_mode(
        kwargs,
        reasoning_effort,
        thinking.or(effort_thinking),
        tokenizer,
        provider,
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
/// render): the request's template kwargs with the `reasoning_effort` entry
/// as the request's provider profile hands it to the template
/// (`ProviderProfile::template_reasoning_effort`, written over any entry
/// under that key whatever its type, the entry removed when the profile hands
/// none, given the thinking preference the render resolves: the template's
/// own switch among the kwargs where the template reads that key, else
/// `template_thinking`), the kwargs as they
/// are when the request carries no field, and `template_thinking`: the typed
/// toggle or the profile's reading of the effort. `None` for every other
/// renderer.
fn rendered_thinking_mode(
    kwargs: Option<&std::collections::HashMap<String, Value>>,
    reasoning_effort: Option<&str>,
    template_thinking: Option<bool>,
    tokenizer: &dyn Tokenizer,
    provider: ProviderProfile,
) -> Option<bool> {
    if reasoning_effort.is_none() {
        return tokenizer.native_thinking_mode(kwargs, template_thinking);
    }
    let mut as_rendered = kwargs.cloned().unwrap_or_default();
    // The template's own switch counts where the template reads that key, as
    // the render gates it.
    let explicit = matches!(
        tokenizer.thinking_key_name(),
        Some(ThinkingKeyName::ThinkingMode)
    )
    .then(|| provider.thinking_from_template_kwargs(kwargs))
    .flatten();
    let thinking = explicit.or(template_thinking);
    match provider.template_reasoning_effort(reasoning_effort, thinking) {
        Some(effort) => {
            as_rendered.insert(
                "reasoning_effort".to_string(),
                Value::String(effort.to_string()),
            );
        }
        None => {
            as_rendered.remove("reasoning_effort");
        }
    }
    tokenizer.native_thinking_mode(Some(&as_rendered), template_thinking)
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
#[path = "../../../../tests/unit/routers/grpc/utils/parsers/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/parsers/parser_resolver_tests.rs"]
mod parser_resolver_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/parsers/hy_v4_tests.rs"]
mod hy_v4_tests;

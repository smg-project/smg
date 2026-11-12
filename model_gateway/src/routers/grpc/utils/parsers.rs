//! Reasoning and tool parser helpers: the parser names set for a model, apart by who set them,
//! the worker-side decoding facts the constraint reads, and the thinking switch.

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
use serde_json::Value;

use crate::{
    routers::grpc::context::WorkerSelection,
    worker::{Worker, WorkerRegistry, WorkerType},
};

/// Per-request parser-name resolution: the names set for a model, apart by who set them, the
/// model's cards (`tool_parser` / `reasoning_parser`, populated from worker labels or an explicit
/// `WorkerSpec` card) and the gateway's `--tool-call-parser` and `--reasoning-parser`. Symphony
/// reads them in that order with its registry between (`regular::symphony::names::reads`).
///
/// Lookups borrow straight from worker metadata (no card clones); only the resolved name is
/// cloned.
#[derive(Clone)]
pub struct ParserResolver {
    /// `None` disables card lookups (parser-free endpoints, tests).
    worker_registry: Option<Arc<WorkerRegistry>>,
    /// The gateway's `--tool-call-parser` and `--reasoning-parser`.
    flags: Names,
}

impl ParserResolver {
    /// The gateway's resolver: `worker_registry` for the cards, and the configured names.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            worker_registry: Some(worker_registry),
            flags: Names {
                tool: configured_tool_parser,
                reasoning: configured_reasoning_parser,
            },
        }
    }

    /// A resolver over no registry and no flags, for the tests' components: it reads no card
    /// and names nothing, so every model is read as the registry decides.
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self {
            worker_registry: None,
            flags: Names::default(),
        }
    }

    /// `names_among` over this resolver's own registry walk, for the
    /// tests: a request's readers take the names its preparation resolved
    /// ([`Self::names_for`]) or read them among the workers the stage holds.
    #[cfg(test)]
    pub(crate) fn names(&self, model: &str) -> ModelNames<'_> {
        ModelNames {
            labels: self.card_names(model),
            flags: &self.flags,
        }
    }

    /// The parser names set for `model`, apart by who set them: its cards'
    /// (its workers' labels), read among `workers`, the model's workers a
    /// stage already holds, with no registry walk of this resolver's own; and
    /// the gateway's flags as set. Symphony reads them in that order with its
    /// registry between (`regular::symphony::names::reads`).
    pub(crate) fn names_among(&self, workers: &[Arc<dyn Worker>], model: &str) -> ModelNames<'_> {
        ModelNames {
            labels: if self.worker_registry.is_some() {
                Self::card_names_among(workers, model)
            } else {
                Names::default()
            },
            flags: &self.flags,
        }
    }

    /// `names_among` for a request whose preparation resolved the card names once
    /// (`ProcessingState::parser_names`): `resolved` as they were then, with the flags, and no
    /// second walk over the model's workers; `None` resolves them now, over the registry's walk.
    pub fn names_for(&self, resolved: Option<&Names>, model: &str) -> ModelNames<'_> {
        ModelNames {
            labels: resolved.map_or_else(|| self.card_names(model), Names::clone),
            flags: &self.flags,
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

    /// Both names `model`'s cards set among `workers`, in one walk. Cards
    /// built by the label pipeline agree across the workers of one model; where
    /// they do not (mixed labels, e.g. mid rolling-upgrade) each kind takes
    /// the lexicographically smallest name, so the choice does not depend on
    /// the registry's iteration order. Registration logs the conflict as a
    /// warning; here it is debug (the per-request hot path).
    fn card_names_among(workers: &[Arc<dyn Worker>], model: &str) -> Names {
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
}

/// The names the cards of `card`'s model set once the worker that carries `card` has joined
/// `workers`, the model's registered workers: what a request of the model then reads, since it
/// reads the names among every worker of the model, whatever its wire
/// ([`ParserResolver::names_among`], by the same pick). For the registration step, which speaks
/// before the worker is in the registry.
pub(crate) fn card_names_with(workers: &[Arc<dyn Worker>], card: &ModelCard) -> Names {
    let mut tool = CardPick::default();
    let mut reasoning = CardPick::default();
    let registered = workers
        .iter()
        .filter_map(|worker| worker.metadata().spec.models.find(&card.id));
    for card in registered.chain(std::iter::once(card)) {
        tool.offer(ParserKind::Tool.of(card));
        reasoning.offer(ParserKind::Reasoning.of(card));
    }
    Names {
        tool: tool.chosen(&card.id, ParserKind::Tool.label()),
        reasoning: reasoning.chosen(&card.id, ParserKind::Reasoning.label()),
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
pub struct Names {
    pub tool: Option<String>,
    pub reasoning: Option<String>,
}

/// No names at all.
static NO_NAMES: Names = Names {
    tool: None,
    reasoning: None,
};

/// The parser names set for one model, apart by who set them (`ParserResolver::names_among`): the
/// names of its card, which its workers' labels (or the operator's card) set for the one model,
/// and the gateway's flags, which every model shares. Symphony reads the two in that order with
/// its registry between them (`regular::symphony::names::reads`): the card's names when they select
/// anything, else the registry's table for a model it knows, else the flags, else the `plain`
/// table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelNames<'a> {
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
            Some(WorkerSelection::Single { worker }) => Self::of_worker(worker.as_ref()),
            Some(WorkerSelection::Disaggregated { decode, .. }) => Self::of_worker(decode.as_ref()),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::MediaPartOrder;

    use super::*;
    use crate::{
        routers::grpc::{
            testing::worker,
            utils::{chat_utils, message_utils},
        },
        worker::{BasicWorkerBuilder, WorkerType},
    };

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
    /// first attempt's did: under `requested` whatever the start or the
    /// vocabulary, under `always` the whole turn's grammar on both or on
    /// neither, and where it is on both, in the same vocabulary (its tag is
    /// spelled once, for the first worker's compiler); and any prefill
    /// worker, which generates no token a grammar constrains.
    #[test]
    fn a_retry_takes_only_workers_that_decide_the_constraint_alike() {
        let always = ("constrained_decoding", "always");
        let first_token = ("constrained_decoding_start", "first_token");
        let older = ("structural_tag_vocabulary", "older");
        let labels: [&[(&str, &str)]; 8] = [
            &[],
            &[first_token],
            &[always, first_token],
            &[always, ("constrained_decoding_start", "after_reasoning")],
            &[always],
            &[always, first_token, older],
            &[older],
            &[always, older],
        ];
        // The workers alike: the `requested` ones, whatever their vocabulary;
        // the whole turn's in the current vocabulary; the `always` ones that
        // start the grammar later or do not say, whatever their vocabulary;
        // the whole turn's in the older vocabulary, alone.
        let alike = [0, 0, 1, 2, 2, 3, 0, 2];
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

    /// The worker's `structural_tag_vocabulary` label names the vocabulary of
    /// structural-tag nodes its grammar compiler takes: the older one on
    /// `older`, the current one otherwise, the label's absence and any other
    /// word included; under PD the decode worker's label.
    #[test]
    fn the_structural_tag_vocabulary_follows_the_label_of_the_decode_worker() {
        assert_eq!(WorkerDecoding::of(None).vocabulary, TagVocabulary::Current);
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[]))).vocabulary,
            TagVocabulary::Current
        );
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[("structural_tag_vocabulary", "older")]))).vocabulary,
            TagVocabulary::Older
        );
        assert_eq!(
            WorkerDecoding::of(Some(&worker(&[("structural_tag_vocabulary", "current")])))
                .vocabulary,
            TagVocabulary::Current
        );
        let prefill = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:50051")
                .worker_type(WorkerType::Prefill)
                .label("structural_tag_vocabulary", "older")
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
            .vocabulary,
            TagVocabulary::Current
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
        // Without a generation header nothing opens a `<think>`: a request
        // with `add_generation_prompt: false` and no continued message is not
        // armed; a continued message keeps the rule above.
        let closed = |continue_final: bool, last_role: &str| -> ChatCompletionRequest {
            let mut body = serde_json::to_value(request(continue_final, last_role)).expect("value");
            body["add_generation_prompt"] = Value::Bool(false);
            serde_json::from_value(body).expect("chat request")
        };
        assert!(!chat_reasoning_starts_in_prefill(
            &closed(false, "user"),
            &tok
        ));
        assert!(!chat_reasoning_starts_in_prefill(
            &closed(false, "assistant"),
            &tok
        ));
        assert!(!chat_reasoning_starts_in_prefill(
            &closed(true, "assistant"),
            &tok
        ));
        // The gateway decides from the rendered prompt, not the request: a
        // renderer with an encoder of its own writes the header whatever the
        // request asked (the DeepSeek ports, as the engine's own encoders
        // do), so its prompt arms as before; one that honoured the request
        // rendered no header and does not; a continued message keeps the
        // native rule.
        assert!(chat_reasoning_starts_in_rendered_prefill(
            &closed(false, "user"),
            true,
            false,
            &tok
        ));
        assert!(!chat_reasoning_starts_in_rendered_prefill(
            &closed(false, "user"),
            false,
            false,
            &tok
        ));
        assert!(!chat_reasoning_starts_in_rendered_prefill(
            &closed(true, "assistant"),
            false,
            true,
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
                ProviderProfile::for_model(&request.model),
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

    /// Where the template cannot continue a Messages prefill, the prompt ends
    /// inside the `<think>` its generation prompt opens. The reader is armed
    /// from the rendered prompt, not from the request asking to continue, so
    /// the completion's reasoning is read as reasoning.
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
            ProviderProfile::for_model(&request.model),
        )
        .expect("rendered");
        assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
        assert!(!processed.continued_final_message);
        let armed = messages_reasoning_starts_in_prefill(
            &request,
            processed.continued_final_message,
            &tokenizer,
        );
        assert!(armed, "the prompt's open thought arms the reader");
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
            ProviderProfile::for_model(&request.model),
        )
        .expect("rendered");
        assert_eq!(processed.text, PREFILL_AFTER_THINKING_PROMPT);
        assert!(!processed.continued_final_message);
        let armed = chat_reasoning_starts_in_rendered_prefill(
            &request,
            processed.generation_prompt,
            processed.continued_final_message,
            &tokenizer,
        );
        assert!(armed, "the prompt's open thought arms the reader");
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

    /// The MiniMax profile reads `reasoning_effort` by its contract: `none`
    /// switches thinking off, so the parser is not armed, as the render
    /// writes the template's off mode; `minimal` is the lowest level and a
    /// level leaves the template's default. The OpenAI reading stays for
    /// every other profile.
    #[test]
    fn minimax_effort_none_disarms_the_parser_as_everywhere() {
        // A tokenizer like the MiniMax M3 template's: the tri-state key, off by
        // default, and no native `reasoning_effort` words of its own.
        let tok = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOff)
            .with_thinking_key_name(ThinkingKeyName::ThinkingMode);
        assert_eq!(
            resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Minimax),
            Some(false),
            "MiniMax none"
        );
        for effort in ["minimal", "low", "high", "max"] {
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
        // The typed toggle still ranks above the profile's reading of the effort.
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
        // A typed `disabled` next to an explicit effort is honoured under
        // MiniMax as everywhere else: the parser is not armed.
        assert_eq!(
            resolve_user_thinking(
                None,
                Some("high"),
                Some(false),
                &tok,
                ProviderProfile::Minimax
            ),
            Some(false)
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

    /// An explicit `thinking.type: disabled` wins over `reasoning_effort` at
    /// any level, through the request: the prompt renders thinking off and
    /// the parser is not armed. The effort alone leaves the template's
    /// default; `enabled` arms. (A template with the tri-state toggle, off by
    /// default, under the MiniMax profile.)
    #[test]
    fn thinking_disabled_wins_over_reasoning_effort() {
        let tok = llm_tokenizer::MockTokenizer::new()
            .with_thinking_toggle(ThinkingToggle::DefaultOff)
            .with_thinking_key_name(ThinkingKeyName::ThinkingMode);
        let request = |fields: Value| -> ChatCompletionRequest {
            let mut value = serde_json::json!({
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "Hi"}],
            });
            value
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().cloned().unwrap_or_default());
            serde_json::from_value(value).expect("chat request")
        };
        for (fields, rendered, armed) in [
            (
                serde_json::json!({"thinking": {"type": "disabled"}}),
                Some(false),
                false,
            ),
            (
                serde_json::json!({"reasoning_effort": "high", "thinking": {"type": "disabled"}}),
                Some(false),
                false,
            ),
            (
                serde_json::json!({"reasoning_effort": "low", "thinking": {"type": "disabled"}}),
                Some(false),
                false,
            ),
            (
                serde_json::json!({"thinking": {"type": "disabled", "effort": "high"}}),
                Some(false),
                false,
            ),
            (serde_json::json!({"reasoning_effort": "high"}), None, false),
            (
                serde_json::json!({"thinking": {"type": "enabled"}}),
                Some(true),
                true,
            ),
        ] {
            let request = request(fields.clone());
            assert_eq!(request.template_thinking(), rendered, "{fields}");
            assert_eq!(
                chat_reasoning_starts_in_prefill(&request, &tok),
                armed,
                "{fields}"
            );
        }
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
}

#[cfg(test)]
mod parser_resolver_tests {
    use openai_protocol::model_card::ModelCard;

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

    /// A request's card names, resolved once in preparation, are what the later reads take: the
    /// names as resolved then with the flags beside them, and no second walk over the model's
    /// workers (a card registered after the resolution does not move them); a read with nothing
    /// resolved walks the cards as before.
    #[test]
    fn names_resolved_once_are_read_in_place_of_the_cards() {
        let registry = registry_with_card(ModelCard::new("m").with_tool_parser("json"));
        let resolver = ParserResolver::new(Arc::clone(&registry), None, Some("qwen3".to_string()));
        let resolved = resolver.names("m").labels;
        assert_eq!(resolved.tool.as_deref(), Some("json"));
        let later = BasicWorkerBuilder::new("http://w2:8000")
            .model(ModelCard::new("m").with_tool_parser("alpha"))
            .worker_type(WorkerType::Regular)
            .build();
        registry.register(Arc::new(later));
        let read = resolver.names_for(Some(&resolved), "m");
        assert_eq!(read.labels, resolved);
        assert_eq!(read.flags, &resolver.flags);
        assert_eq!(
            resolver.names_for(None, "m").labels.tool.as_deref(),
            Some("alpha")
        );
    }

    /// The names a model's cards set are its labels, each kind on its own, and the flags stand
    /// beside them as given, for a model with a card and for one without; a resolver over no
    /// registry reads no card.
    #[test]
    fn the_cards_names_are_the_labels_and_the_flags_stand_beside_them() {
        let registry = registry_with_card(
            ModelCard::new("m")
                .with_tool_parser("json")
                .with_reasoning_parser("basic"),
        );
        let resolver = ParserResolver::new(
            registry,
            Some("mistral".to_string()),
            Some("deepseek".to_string()),
        );
        let names = resolver.names("m");
        assert_eq!(
            names.labels,
            Names {
                tool: Some("json".to_string()),
                reasoning: Some("basic".to_string()),
            }
        );
        assert_eq!(
            names.flags,
            &Names {
                tool: Some("mistral".to_string()),
                reasoning: Some("deepseek".to_string()),
            }
        );
        // No card for the model: no labels, the same flags.
        let other = resolver.names("other");
        assert_eq!(other.labels, Names::default());
        assert_eq!(other.flags, names.flags);
        // A card that sets one kind leaves the other unset.
        let one = ParserResolver::new(registry_with_card(ModelCard::new("m")), None, None);
        assert_eq!(one.names("m"), ModelNames::default());
        assert_eq!(ParserResolver::disabled().names("m"), ModelNames::default());
    }

    /// The reading stage holds the model's workers already: the names read among them are the
    /// names a registry walk resolves, each kind chosen on its own in the one pass (the smallest
    /// where the cards disagree, a card for another model skipped), and a disabled resolver
    /// reads none among them either.
    #[test]
    fn the_names_among_a_models_workers_are_the_registry_walks() {
        let registry = Arc::new(WorkerRegistry::new());
        let cards = [
            ModelCard::new("m")
                .with_tool_parser("zebra")
                .with_reasoning_parser("qwen3"),
            ModelCard::new("m").with_tool_parser("alpha"),
            ModelCard::new("other")
                .with_tool_parser("aardvark")
                .with_reasoning_parser("basic"),
        ];
        for (i, card) in cards.into_iter().enumerate() {
            let worker = BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                .model(card)
                .worker_type(WorkerType::Regular)
                .build();
            registry.register(Arc::new(worker));
        }
        let resolver = ParserResolver::new(Arc::clone(&registry), None, Some("flag".to_string()));
        let workers = registry.get_by_model("m");
        let among = resolver.names_among(&workers, "m");
        assert_eq!(among, resolver.names("m"));
        assert_eq!(
            among.labels,
            Names {
                tool: Some("alpha".to_string()),
                reasoning: Some("qwen3".to_string()),
            }
        );
        assert_eq!(among.flags, &resolver.flags);
        assert_eq!(
            resolver.names_among(&workers, "other").labels,
            Names::default()
        );
        assert_eq!(
            ParserResolver::disabled().names_among(&workers, "m").labels,
            Names::default()
        );
        // The slice is the stage's snapshot: a worker registered after it was
        // taken moves a registry walk's names, not the names read among it.
        let later = BasicWorkerBuilder::new("http://w3:8000")
            .model(ModelCard::new("m").with_tool_parser("aardvark"))
            .worker_type(WorkerType::Regular)
            .build();
        registry.register(Arc::new(later));
        assert_eq!(
            resolver.names_among(&workers, "m").labels.tool.as_deref(),
            Some("alpha")
        );
        assert_eq!(resolver.names("m").labels.tool.as_deref(), Some("aardvark"));
    }

    #[test]
    fn conflicting_overrides_resolve_deterministically() {
        // Two same-model workers with different overrides: resolution must
        // not depend on registration/iteration order: the lexicographically
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
            assert_eq!(resolver.names("m").labels.tool.as_deref(), Some("alpha"));
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

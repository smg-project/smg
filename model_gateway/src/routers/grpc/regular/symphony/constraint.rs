//! Constraint derivation: what a request asks of the table (`Asks`), the structural tags for a
//! forced call, a JSON answer or the calls `auto` allows, the whole turn's grammar, and the
//! decision on the selected worker.

use std::{borrow::Cow, fmt, sync::Arc};

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue, ToolReference},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
    profile::ProviderProfile,
    worker::{ConstrainedDecoding, GrammarStart as WorkerGrammarStart},
};
use serde_json::{json, Value};
use smg_worker::identity::basic::{Worker, WorkerType};
use symphony::{
    grammar::{GrammarStart, Options},
    Format, Grammar, GrammarBackend, OutputAnswer, OutputChoice, Spelling, Turn, Unspelled,
};

#[cfg(test)]
use super::names::reading;
use super::{
    names::{reads, table_reading, ByAlias, Reading},
    template::Templates,
    SymphonyError,
};
use crate::{
    config::ParserChoice,
    routers::grpc::{
        common::stages::helpers::TagForStart,
        spec::{scans_for_tool_calls_for, ChatResponseSpec},
        utils::{ModelNames, Names, TagVocabulary, WorkerDecoding},
    },
};

/// The schema a JSON `response_format` asks the answer to follow, borrowed from the request (owned
/// only for `json_object`'s synthesized object): any object for `json_object`;
/// none for a structural tag, a grammar of the request's own that no table shapes
/// (`Asks::own_structural_tag`).
fn answer_schema(response_format: Option<&ResponseFormat>) -> Option<Cow<'_, Value>> {
    match response_format? {
        ResponseFormat::JsonObject => Some(Cow::Owned(json!({"type": "object"}))),
        ResponseFormat::JsonSchema { json_schema } => Some(Cow::Borrowed(&json_schema.schema)),
        ResponseFormat::Text | ResponseFormat::StructuralTag { .. } => None,
    }
}

/// `grammar` spelled for the worker's grammar backend ([`Grammar::spelled`]): xgrammar's
/// structural tag, written straight from the tree, in the `vocabulary` the worker's grammar
/// compiler takes (spelled without the newer nodes for a worker that advertises the older one,
/// [`TagVocabulary`]); a JSON schema alone rides as the schema constraint the wires carry. Every
/// worker spells for xgrammar today. A backend with no spelling of the grammar is [`Unspelled`],
/// and what that counts as follows what the request asked and whose table reads the output: a tag
/// the table does not have under `auto` ([`auto_tag`]) and for a registry table's forced call or
/// answer ([`unspelled_tag`]: the older registry decides, as before), a refusal by the backend's
/// name for the response template's forced call or answer ([`unspelled_tag`]: that table alone
/// writes them) and for the [`whole_turn`].
fn structural_tag(
    grammar: Grammar,
    vocabulary: TagVocabulary,
) -> Result<ToolConstraint, Unspelled> {
    grammar
        .spelled(GrammarBackend::Xgrammar, vocabulary)
        .map(|spelling| match spelling {
            Spelling::StructuralTag(text) => ToolConstraint::StructuralTag(text),
            Spelling::JsonSchema(text) => ToolConstraint::JsonSchema(text),
        })
}

/// What a tag the worker's grammar backend has no spelling of counts as, for `model`'s `asked`:
/// no tag, which a registry table leaves to the older registry as it leaves a tag it does not
/// have ([`tagged`]), and which the response template's table (`compiled`) refuses by the
/// backend's name, since its output has no other syntax the older path could constrain.
fn unspelled_tag(
    unspelled: Unspelled,
    compiled: bool,
    model: &str,
    asked: &str,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    if compiled {
        Err(SymphonyError::Constraint(format!(
            "model '{model}' reads its output with its response template, whose structural tag \
             for {asked} the {} grammar backend has no spelling of",
            unspelled.backend
        )))
    } else {
        Ok(None)
    }
}

/// The constraint `grammar` gives, written straight from the tree as the request carries it, in
/// the `vocabulary` the worker's grammar compiler takes ([`structural_tag`]), for the response
/// template's table too (`compiled`, [`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)): a registry
/// table with no tag leaves the older registry to decide, as before; the template's table refuses
/// instead, since its output has no other syntax the older path could constrain and the table
/// still read, and so does a prompt that `continues` the assistant's message, since the tag
/// writes a turn from its start. A tag the worker's grammar backend has no spelling of counts
/// as no tag ([`unspelled_tag`]).
pub(super) fn tagged(
    grammar: Option<Grammar>,
    vocabulary: TagVocabulary,
    compiled: bool,
    model: &str,
    asked: &str,
    continues: bool,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    match grammar {
        Some(_) if compiled && continues => Err(SymphonyError::Constraint(format!(
            "model '{model}' constrains {asked} from the start of a new assistant message, as its \
             response template writes one, so it cannot continue the assistant's message"
        ))),
        Some(grammar) => match structural_tag(grammar, vocabulary) {
            Ok(constraint) => Ok(Some(constraint)),
            Err(unspelled) => unspelled_tag(unspelled, compiled, model, asked),
        },
        None if compiled => Err(SymphonyError::Constraint(format!(
            "model '{model}' reads its output with its response template, whose table has no \
             structural tag for {asked}"
        ))),
        None => Ok(None),
    }
}

/// The tag of a forced call to one of `tools`, at most one when `one_call`, with
/// `reasoning_open` when the prompt leaves the model inside its thought and `thought_closed` when
/// it closed the thought itself and wrote the template's text after it, in the `vocabulary` the
/// worker's grammar compiler takes ([`tagged`]). A registry table's tag that opens with free
/// text, the prefix an open thought takes on workers that name no one start for the grammar
/// ([`opens_with_free_text`]), comes marked ([`Prepared::OpensWithFreeText`]): text that owes
/// nothing at its end does not hold the model to the call, so preparation gives the call's
/// JSON schema in the tag's place where that schema can be built ([`Prepared::held_by`]).
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engine leave the model, \
              the vocabulary its compiler takes"
)]
fn calls_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    tools: &[Tool],
    one_call: bool,
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
    continues: bool,
) -> PreparedConstraint {
    let grammar = format.grammar_for(
        tools,
        Options {
            at_least_one: true,
            reasoning_open,
            thought_closed,
            one_call,
            start,
        },
    );
    // The response template's table keeps its tag unmarked: it has no other syntax the older
    // registry could constrain and the table still read ([`tagged`]).
    let free = !compiled && grammar.as_ref().is_some_and(opens_with_free_text);
    let first_token = start == GrammarStart::FirstToken;
    let tag = tagged(
        grammar,
        vocabulary,
        compiled,
        model,
        "a forced tool choice",
        continues,
    )?;
    Ok(match tag {
        Some(constraint) if free => Prepared::OpensWithFreeText {
            constraint,
            first_token,
        },
        Some(constraint) => Prepared::Constraint {
            constraint,
            first_token,
            table: true,
        },
        None => Prepared::None,
    })
}

/// Whether `grammar` opens with free text, which owes nothing at its end: a tag that begins so
/// does not hold the model to what follows it.
fn opens_with_free_text(grammar: &Grammar) -> bool {
    match grammar {
        Grammar::AnyText { .. } => true,
        Grammar::Sequence(elements) => elements.first().is_some_and(opens_with_free_text),
        _ => false,
    }
}

/// Whether `schema`, the JSON schema preparation builds for a forced call to `tools`, keeps what
/// every reference in the tools' parameters points at: each is a local one (`#` and a JSON
/// pointer) whose target in `schema` is its target in the tool's own parameters. Preparation
/// writes a named function's parameters as they are, and for a list of tools one schema that
/// nests each tool's parameters and lifts their `$defs` to its root, where a reference to
/// anything else (an entry of `definitions`, the tool's own root) points nowhere or at another
/// value. A schema that loses a target is not sent in a tag's place ([`Prepared::held_by`]).
/// A constraint of another kind, which preparation's `forced_call_schema` never gives, is
/// taken as it is.
pub(crate) fn keeps_references(schema: &ToolConstraint, tools: &[Tool]) -> bool {
    let ToolConstraint::JsonSchema(schema) = schema else {
        return true;
    };
    let mut references = tools
        .iter()
        .map(|tool| tool.function.parameters_schema())
        .flat_map(|parameters| {
            references_in(parameters)
                .into_iter()
                .map(move |reference| (parameters, reference))
        })
        .peekable();
    if references.peek().is_none() {
        return true;
    }
    let Ok(schema) = serde_json::from_str::<Value>(schema) else {
        return false;
    };
    references.all(|(parameters, reference)| {
        reference.strip_prefix('#').is_some_and(|pointer| {
            parameters
                .pointer(pointer)
                .is_some_and(|target| schema.pointer(pointer) == Some(target))
        })
    })
}

/// The text of every reference written in `schema`: each `$ref`, `$dynamicRef` and
/// `$recursiveRef` that stands where a schema does. The values of `const`, `enum`, `default`
/// and `examples` are data, and a member named like a reference there is none. The members of
/// `properties`, `patternProperties`, `$defs`, `definitions`, `dependentSchemas` and
/// `dependencies` are schemas under names of their own, so a property named `default` is read.
/// Any other keyword is followed, so one this does not know cannot hide a reference.
fn references_in(schema: &Value) -> Vec<&str> {
    let mut found = Vec::new();
    // Each value still to read, with whether its members are schemas under names of their own.
    let mut pending = vec![(schema, false)];
    while let Some((value, named)) = pending.pop() {
        match value {
            Value::Object(members) if named => {
                pending.extend(members.values().map(|member| (member, false)));
            }
            Value::Object(members) => {
                for (key, member) in members {
                    match (key.as_str(), member) {
                        ("$ref" | "$dynamicRef" | "$recursiveRef", Value::String(reference)) => {
                            found.push(reference.as_str());
                        }
                        ("const" | "enum" | "default" | "examples", _) => {}
                        (
                            "properties" | "patternProperties" | "$defs" | "definitions"
                            | "dependentSchemas" | "dependencies",
                            _,
                        ) => pending.push((member, true)),
                        _ => pending.push((member, false)),
                    }
                }
            }
            Value::Array(items) => pending.extend(items.iter().map(|item| (item, false))),
            _ => {}
        }
    }
    found
}

/// The tag of a JSON answer following `schema`, with `reasoning_open` when the prompt leaves the
/// model inside its thought ([`Format::answer_grammar`]): the answer written where the table
/// reads one, which the engine's own schema constraint is not for a table whose answer is a
/// message (the bare JSON lands outside every message and the table drops it), nor with the
/// thought open for an engine that runs the grammar from the first token (the JSON lands inside
/// the thought), nor for a table that admits a thought before its answer. A registry table whose
/// tag is the schema alone, the engine's own constraint, leaves it to the engine as before, and
/// so does one with no tag, or a prompt that `continues` the assistant's message, since the tag
/// writes a turn from its start; the template's table takes its tag either way and refuses what
/// it cannot constrain ([`tagged`]). `vocabulary` is the one the worker's grammar compiler takes.
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engine leave the model, \
              the vocabulary its compiler takes"
)]
fn answer_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    schema: &Arc<Value>,
    reasoning_open: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
    continues: bool,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    let grammar = format.answer_grammar_for(schema, reasoning_open, start);
    if !compiled && (continues || matches!(grammar, Some(Grammar::JsonSchema { .. }))) {
        return Ok(None);
    }
    tagged(
        grammar,
        vocabulary,
        compiled,
        model,
        "a JSON answer",
        continues,
    )
}

/// Where a worker's engine starts a request's grammar on a prompt that leaves the model inside
/// its thought, as Symphony's tags take it: the worker's `constrained_decoding_start` label
/// ([`WorkerDecoding::start`]).
pub(crate) fn grammar_start_from(start: WorkerGrammarStart) -> GrammarStart {
    match start {
        WorkerGrammarStart::AfterReasoning => GrammarStart::AfterReasoning,
        WorkerGrammarStart::FirstToken => GrammarStart::FirstToken,
        WorkerGrammarStart::Unknown => GrammarStart::Unknown,
    }
}

/// Where the model's `workers` start a grammar, before one is selected: the start every one of
/// them names in its `constrained_decoding_start` label (parsed once at its construction,
/// [`WorkerMetadata::decoding`](smg_worker::identity::metadata::WorkerMetadata::decoding): this walk runs on every
/// request and looks nothing up), a prefill or encode worker left out
/// since it generates no token a grammar constrains (the roles that do are allow-listed, so a
/// role added later is left out until it is named); `Either` when one that starts at the first
/// token stands beside one that does not, since no one tag fits both (the answer keeps the
/// prefix a forced call takes: [`Format::answer_grammar_for`]); unknown when none says, and
/// when workers that start after their reasoning parser stand beside ones that do not say,
/// since the unknown start's tags fit both (a forced call's tag, or the call's JSON schema
/// where the table's tag would open with free text; an answer's schema alone).
pub(crate) fn grammar_start_of(workers: &[Arc<dyn Worker>]) -> GrammarStart {
    let mut starts = workers
        .iter()
        .filter(|worker| generates(worker.as_ref()))
        .map(|worker| worker.metadata().decoding.start);
    let Some(first) = starts.next() else {
        return GrammarStart::Unknown;
    };
    let starts: Vec<_> = std::iter::once(first).chain(starts).collect();
    if starts.iter().all(|start| *start == first) {
        grammar_start_from(first)
    } else if starts.contains(&WorkerGrammarStart::FirstToken) {
        // An engine that starts at the first token beside one that does not: no one tag fits both.
        GrammarStart::Either
    } else {
        // After the reasoning parser beside not saying: the unknown start's tags fit both.
        GrammarStart::Unknown
    }
}

/// `worker` generates tokens a grammar constrains: a prefill or encode worker does not (the roles
/// that do are allow-listed, so a role added later is left out until it is named).
fn generates(worker: &dyn Worker) -> bool {
    matches!(
        worker.worker_type(),
        WorkerType::Regular | WorkerType::Decode
    )
}

/// The vocabulary of structural-tag nodes the model's `workers` take together, before one is
/// selected, for a tag every one of them takes as it is (a forced call's, [`constraint_for`]):
/// the older one when any worker that generates tokens ([`generates`]) advertises it in its
/// `structural_tag_vocabulary` label (read as parsed at its construction, like the start above),
/// since every compiler takes that spelling and the worker
/// that needs it may be the one selected; the current one otherwise, the tag then as the table
/// built it ([`TagVocabulary`]). The tags derived for the selected worker take its own label
/// ([`WorkerDecoding::vocabulary`]).
pub(crate) fn vocabulary_of(workers: &[Arc<dyn Worker>]) -> TagVocabulary {
    let older = workers
        .iter()
        .filter(|worker| generates(worker.as_ref()))
        .any(|worker| worker.metadata().decoding.vocabulary == TagVocabulary::Older);
    if older {
        TagVocabulary::Older
    } else {
        TagVocabulary::Current
    }
}

/// What a request asks of the table, as its API spells it: a chat request's own fields, or a
/// Messages request's with whether the prompt continues the assistant's message beside it, which
/// its stage decides (from the request in preparation, from the rendered prompt for the `auto`
/// tag). Each rule the seam reads off a request is one method here, in both spellings; the
/// functions that take an `Asks` know neither API.
#[derive(Clone, Copy)]
pub(crate) enum Asks<'a> {
    Chat(&'a ChatCompletionRequest),
    Messages {
        request: &'a CreateMessageRequest,
        /// The prompt continues the assistant's message.
        continues: bool,
        /// The request declared a `tools` field (an empty list being a declaration), as its
        /// response spec recorded it: the route takes the tools out of a request it still owns,
        /// so the request's own field reads `None` past the route.
        has_tools: bool,
    },
}

impl<'a> Asks<'a> {
    /// A Messages request's ask over a request that still carries its tools, as the tests build
    /// theirs: the declaration is the request's own field.
    #[cfg(test)]
    pub(crate) fn for_messages(request: &'a CreateMessageRequest, continues: bool) -> Self {
        Self::Messages {
            request,
            continues,
            has_tools: request.tools.is_some(),
        }
    }

    pub(crate) fn model(self) -> &'a str {
        match self {
            Self::Chat(request) => &request.model,
            Self::Messages { request, .. } => &request.model,
        }
    }

    /// The prompt continues the assistant's message.
    pub(super) fn continues(self) -> bool {
        match self {
            Self::Chat(request) => request.continues_final_assistant(),
            Self::Messages { continues, .. } => continues,
        }
    }

    /// The tool calls the conversation's history already holds.
    pub(super) fn history_tool_calls_count(self) -> usize {
        match self {
            Self::Chat(request) => request.history_tool_calls_count(),
            Self::Messages { request, .. } => request.history_tool_calls_count(),
        }
    }

    /// The request carries a `regex` or `ebnf` constraint, which applies outside a template's
    /// format; a Messages request carries neither.
    fn constrains_outside_the_format(self) -> bool {
        match self {
            Self::Chat(request) => request.regex.is_some() || request.ebnf.is_some(),
            Self::Messages { .. } => false,
        }
    }

    /// The request's `response_format` is a structural tag: a grammar of the request's own over
    /// the sections it names, with the text around them free, which the engine takes whole and as
    /// it arrived (its request builder sends it, `response_format_constraint`). No table shapes an
    /// answer's tag, an `auto` tag or a turn's grammar over it; a template's format stands around
    /// it, so a compiled table does not refuse it as it refuses a `regex` or an `ebnf`. A Messages
    /// request names none.
    pub(super) fn own_structural_tag(self) -> bool {
        match self {
            Self::Chat(request) => matches!(
                request.response_format,
                Some(ResponseFormat::StructuralTag { .. })
            ),
            Self::Messages { .. } => false,
        }
    }

    /// The JSON schema the request's answer must fit: a chat request's `response_format`'s, a
    /// Messages request's `output_config.format`'s, borrowed from the request; a caller that keeps it
    /// takes the one copy ownership needs.
    pub(super) fn answer_schema(self) -> Option<Cow<'a, Value>> {
        match self {
            Self::Chat(request) => answer_schema(request.response_format.as_ref()),
            Self::Messages { request, .. } => request.output_schema().map(Cow::Borrowed),
        }
    }

    /// [`Self::answer_schema`] for the engine's own JSON-schema constraint where nothing else
    /// constrains the answer (no table reads the response, or the table gives the selected
    /// worker no tag for the schema, which a registry table leaves to the engine): a Messages
    /// request's, unless its choice forces a call to one of `tools` ([`Self::forced_call`]): a
    /// turn forced into a call has no text answer to format, whether or not the call has a
    /// constraint of its own (a legacy parser may give a forced call none and let the model
    /// write its native call). A chat request's reaches the engine through its request builder
    /// (`response_format_constraint`), so none is named here.
    pub(crate) fn untabled_answer_schema(self, tools: &[Tool]) -> Option<Cow<'a, Value>> {
        match self {
            Self::Chat(_) => None,
            Self::Messages { .. } if self.forced_call(tools).is_some() => None,
            Self::Messages { request, .. } => request.output_schema().map(Cow::Borrowed),
        }
    }

    /// Whether the request's choice forces a call to one of `tools`, and then whether one call at
    /// most: a chat `tool_choice` that forces a call ([`ToolChoice::forces_call`]; validation holds
    /// a named function to the declared ones), one call for a named function or
    /// `parallel_tool_calls: false`; a Messages `tool_choice` of `any`, or one naming a tool among
    /// `tools` (validation lets it name a tool of another kind, which the custom tools the model
    /// sees never hold: the list narrowed by the choice at preparation and the declared one at the
    /// decision then agree that nothing is forced), one call for the named tool or
    /// `disable_parallel_tool_use`.
    fn forced_call(self, tools: &[Tool]) -> Option<bool> {
        if tools.is_empty() {
            return None;
        }
        match self {
            Self::Chat(request) => {
                let choice = request.tool_choice.as_ref()?;
                choice.forces_call().then(|| {
                    matches!(choice, ToolChoice::Function { .. })
                        || request.disables_parallel_tool_calls()
                })
            }
            Self::Messages { request, .. } => match &request.tool_choice {
                Some(MessagesToolChoice::Tool { name, .. }) => tools
                    .iter()
                    .any(|tool| tool.function.name == *name)
                    .then_some(true),
                Some(choice @ MessagesToolChoice::Any { .. }) => {
                    Some(choice.disables_parallel_tool_use())
                }
                _ => None,
            },
        }
    }

    /// Whether the calls the request allows under `auto` may be tagged over `tools`, and then
    /// whether one call at most. Nothing where the request has no callable tool, forbids the
    /// calls (`tool_choice: none`), or continues the assistant's message (the tag writes a turn
    /// from its start). A chat request that carries a constraint of its own, which the tag would
    /// displace (a JSON or structural-tag `response_format`, a `regex`, an `ebnf`), takes none; a
    /// Messages choice that forces a call takes none, its tag being [`constraint_for`]'s.
    fn auto_calls(self, tools: &[Tool]) -> Option<bool> {
        if tools.is_empty() || self.continues() {
            return None;
        }
        match self {
            Self::Chat(request) => {
                let forbids = matches!(
                    request.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                );
                let own = self.answer_schema().is_some()
                    || self.constrains_outside_the_format()
                    || self.own_structural_tag();
                (!forbids && !own).then(|| request.disables_parallel_tool_calls())
            }
            Self::Messages { request, .. } => match &request.tool_choice {
                None => Some(false),
                Some(MessagesToolChoice::Auto {
                    disable_parallel_tool_use,
                }) => Some(*disable_parallel_tool_use == Some(true)),
                Some(
                    MessagesToolChoice::None
                    | MessagesToolChoice::Any { .. }
                    | MessagesToolChoice::Tool { .. },
                ) => None,
            },
        }
    }

    /// The request's `tool_choice` as a chat one, which a Messages choice converts to.
    pub(crate) fn choice(self) -> Option<Cow<'a, ToolChoice>> {
        match self {
            Self::Chat(request) => request.tool_choice.as_ref().map(Cow::Borrowed),
            Self::Messages { request, .. } => request
                .tool_choice
                .as_ref()
                .map(|choice| Cow::Owned(ToolChoice::from(choice))),
        }
    }

    /// Whether the request lets the model make calls in parallel.
    fn parallel(self) -> bool {
        match self {
            Self::Chat(request) => !request.disables_parallel_tool_calls(),
            Self::Messages { request, .. } => !request.disables_parallel_tool_use(),
        }
    }

    /// The answer the request's turn ends in, `schema` being [`Self::answer_schema`]'s: the
    /// schema, else a chat request's `regex` or `ebnf` (the protocol refuses more than one).
    fn answer<'s>(self, schema: Option<&'s Value>) -> Option<OutputAnswer<'s>>
    where
        'a: 's,
    {
        let answer = schema.map(OutputAnswer::Json);
        match self {
            Self::Chat(request) => answer
                .or_else(|| request.regex.as_deref().map(OutputAnswer::Regex))
                .or_else(|| request.ebnf.as_deref().map(OutputAnswer::Ebnf)),
            Self::Messages { .. } => answer,
        }
    }

    /// The table reads no calls from the output: the legacy path would run no tool parser for the
    /// request (a chat response it would not scan for calls, no tools declared anywhere, for a
    /// provider that reads calls only with them: [`scans_for_tool_calls_for`]; a Messages request
    /// that declares no tools, an empty list being a declaration). A `tool_choice` of `none` is
    /// not this: the table keeps its calls and the reader delivers none of them
    /// ([`Asks::forbids_for`]). `provider` is the chat model's profile as the request's
    /// preparation derived it once.
    pub(super) fn no_calls_for(self, provider: ProviderProfile) -> bool {
        match self {
            Self::Chat(request) => !scans_for_tool_calls_for(request, provider),
            Self::Messages { has_tools, .. } => !has_tools,
        }
    }

    /// The calls the request's `tool_choice` forbids among those its response would be scanned
    /// for (over declared tools, or over a provider that reads calls without any): every call
    /// under `none`; under a named function, an allowed list or `required`, a call to a function
    /// outside the tools the choice leaves callable ([`callable`]); none under `auto`. The table
    /// keeps its calls, so a call the model writes anyway is read whole, and the reader drops
    /// it, as the engine's own server reads a call under `none` and withholds it. Under `none`
    /// the prompt declares the tools as under `auto` (`utils::request_as_rendered`), so the
    /// model may well write one. `provider` is the chat model's profile as the request's
    /// preparation derived it once.
    pub(super) fn forbids_for(self, provider: ProviderProfile) -> Forbids {
        let choice = self.choice();
        let forbids = match choice.as_deref() {
            None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => return Forbids::Nothing,
            Some(ToolChoice::Value(ToolChoiceValue::None)) => Forbids::Every,
            Some(ToolChoice::Value(ToolChoiceValue::Required)) => Forbids::Undeclared,
            Some(ToolChoice::Function { function, .. }) => {
                Forbids::AllBut([function.name.clone()].into())
            }
            Some(ToolChoice::AllowedTools { tools, .. }) => Forbids::AllBut(
                tools
                    .iter()
                    .filter_map(ToolReference::function_name)
                    .map(str::to_owned)
                    .collect(),
            ),
        };
        let scanned = match self {
            Self::Chat(request) => scans_for_tool_calls_for(request, provider),
            Self::Messages { has_tools, .. } => has_tools,
        };
        if scanned {
            forbids
        } else {
            Forbids::Nothing
        }
    }

    /// The table reads no thought from the output: a chat request that keeps the reasoning in the
    /// content (`separate_reasoning: false`), as the legacy path runs no reasoning parser then; a
    /// Messages request's thinking blocks are always its own.
    pub(crate) fn no_reasoning(self) -> bool {
        match self {
            Self::Chat(request) => !request.separate_reasoning,
            Self::Messages { .. } => false,
        }
    }

    /// The request asked for generation past the model's own end of output (a chat request's
    /// `ignore_eos`; a Messages request cannot ask it): the engine keeps sampling after the
    /// end-of-turn token, and the gateway's stop decoder does not stop on it either
    /// (`utils::create_stop_decoder`), so the output the table reads carries an unconstrained
    /// tail after the model's answer. A marker of the table in that tail is the tail's doing,
    /// not a frame the table failed to read, and the tally counts every such marker of the
    /// output apart, the answer's own included
    /// ([`Assignment::past_the_end`](super::reader::Assignment::past_the_end)).
    pub(super) fn past_the_end(self) -> bool {
        match self {
            Self::Chat(request) => request.ignore_eos,
            Self::Messages { .. } => false,
        }
    }

    /// The request holds the answer to JSON: a chat request's `json_object` or `json_schema`
    /// response format (the response spec's `json_mode`, the one rule both paths read through
    /// [`ChatResponseSpec::json_mode_of`]; a structural tag is free text around its tags), a
    /// Messages request's `output_config.format`. The table's engine then reads the answer's
    /// quote state and keeps a marker the model spells inside one of the answer's strings as
    /// the answer's text, as the family's own parsers do under their JSON mode
    /// ([`Assignment::json_answer`](super::reader::Assignment::json_answer)).
    pub(super) fn json_answer(self) -> bool {
        match self {
            Self::Chat(request) => ChatResponseSpec::json_mode_of(request),
            Self::Messages { request, .. } => request.output_schema().is_some(),
        }
    }
}

/// The tools the request's `choice` leaves callable among `declared`: none for a `tool_choice` of
/// `none`, the ones a named function or `allowed_tools` narrows to ([`ToolChoice::narrow_tools`]),
/// all of them otherwise.
pub(super) fn callable<'t>(choice: Option<&ToolChoice>, declared: &'t [Tool]) -> Cow<'t, [Tool]> {
    match choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => Cow::Borrowed(&[]),
        Some(choice) => choice
            .narrow_tools(declared)
            .map_or(Cow::Borrowed(declared), Cow::Owned),
        None => Cow::Borrowed(declared),
    }
}

/// The calls a request's `tool_choice` forbids among those the table reads
/// ([`Asks::forbids_for`]), which the reader drops: a call is forbidden when its function is
/// outside the tools the choice leaves callable ([`callable`]). `auto` narrows nothing and
/// forbids nothing: a call to a function the request did not declare goes out under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Forbids {
    /// No call (`auto`, or no choice).
    Nothing,
    /// Every call (`none`).
    Every,
    /// A call to a function the request did not declare (`required`: every declared tool is
    /// callable).
    Undeclared,
    /// A call to any function but these, the ones the choice names (a named function, an allowed
    /// list), and a call to one of these the request did not declare.
    AllBut(Arc<[String]>),
}

impl Forbids {
    /// Whether the choice forbids a call to `function`, `declared` saying whether the request
    /// declares that function.
    pub(super) fn call_to(&self, function: &str, declared: bool) -> bool {
        match self {
            Self::Nothing => false,
            Self::Every => true,
            Self::Undeclared => !declared,
            Self::AllBut(named) => !declared || !named.iter().any(|name| name == function),
        }
    }

    /// The choice as a reader of the `declared` tools asks it: with no tool declared none is
    /// callable, so a choice that forbids some calls forbids them all.
    pub(super) fn over(&self, declared: &[Tool]) -> Self {
        match self {
            Self::Undeclared | Self::AllBut(_) if declared.is_empty() => Self::Every,
            forbids => forbids.clone(),
        }
    }
}

/// [`constraint_for`] over the table decision read here ([`reading`]), as the tests ask it.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engines leave the model"
)]
pub(crate) fn constraint(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    tools: &[Tool],
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
    templates: &Templates,
) -> PreparedConstraint {
    constraint_for(
        &reading(parser, names, asks.model(), templates)?,
        asks,
        tools,
        reasoning_open,
        thought_closed,
        start,
        vocabulary,
    )
}

/// What the table gives a request's forced or required call to one of `tools`, or a chat request's
/// JSON `response_format`, with `reasoning_open` when the prompt leaves the model inside its
/// thought, over the table decision already read for the request (`read`, which preparation reads
/// once for the replay, the constraint and the decoder's special tokens,
/// `regular::stages::reading::prepare`): the call's structural tag, built for `start`, where the
/// model's workers start the grammar together ([`calls_tag`], [`grammar_start_of`]), in the
/// `vocabulary` they take together ([`vocabulary_of`]); the answer's as it derives for the selected
/// worker's own start and vocabulary ([`Prepared::Answer`], [`AnswerTag`]), since a pool whose
/// workers start the grammar at different places has no one answer tag that fits them all.
/// [`Prepared::None`] when no table reads the response ([`reads`]), the request asks neither, or
/// the table has no tag for the call; the older registry then decides as before. A call's tag
/// that opens with free text comes marked ([`Prepared::OpensWithFreeText`]). An error refuses
/// the request: the response template's table cannot constrain the call it is asked (the answer's
/// refusal comes with the derivation), or a chat request's `regex` or `ebnf` constraint would
/// apply outside its format. `thought_closed` says the prompt closed the thought itself and wrote
/// the template's text after it, which the calls' tag then owes nothing of; where the template
/// writes nothing after the close, the thought is over and a JSON answer is asked as after it
/// ([`AnswerTag::for_start`]). The tests ask through `constraint`, which reads the decision first.
pub(crate) fn constraint_for(
    read: &Reading,
    asks: Asks<'_>,
    tools: &[Tool],
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
) -> PreparedConstraint {
    let Reading::Table(format, compiled) = read else {
        return Ok(Prepared::None);
    };
    let (format, compiled) = (Arc::clone(format), *compiled);
    if compiled && asks.constrains_outside_the_format() {
        return Err(SymphonyError::Constraint(format!(
            "model '{}' writes its output in the format its response template declares, and a \
             regex or ebnf constraint would apply outside it",
            asks.model()
        )));
    }
    let continues = asks.continues();
    if let Some(one_call) = asks.forced_call(tools) {
        return calls_tag(
            &format,
            compiled,
            asks.model(),
            tools,
            one_call,
            reasoning_open,
            thought_closed,
            start,
            vocabulary,
            continues,
        );
    }
    Ok(match asks.answer_schema() {
        Some(schema) => {
            // Past the thought where the prompt closed it in the model's place. The layout read
            // below is the first arm of `Format::thought_closed` (a template that writes text
            // after the close), which keeps its tag: the rule is for its other arm alone, the
            // table that leaves the thought to the model, and an arm added there is weighed here.
            let past_the_thought = thought_closed
                && !matches!(
                    format.layout().and_then(|layout| layout.thought.as_ref()),
                    Some(thought) if !thought.after.is_empty()
                );
            Prepared::Answer(Arc::new(AnswerTag {
                format,
                compiled,
                model: asks.model().to_string(),
                schema: Arc::new(schema.into_owned()),
                reasoning_open,
                past_the_thought,
                continues,
            }))
        }
        None => Prepared::None,
    })
}

/// The tag of the calls a request's model may make under `auto`, for a worker whose engine takes
/// a grammar on every turn ([`ConstrainedDecoding::Always`]): the table's grammar for the request's
/// callable `tools` with no call required, at most one when parallel calls are off, and
/// `reasoning_open` when the prompt leaves the model inside its thought. The request asked for
/// nothing, so nothing is refused: `Ok(None)` when no table reads the response ([`reads`]), the
/// request takes no tag for its calls ([`Asks::auto_calls`]), the table has none for them, or
/// the worker's grammar backend has no spelling of it ([`auto_tag`]). A thought and an answer
/// alone are not constrained today. `thought_closed` as for [`constraint_for`].
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engine leave the model"
)]
pub(crate) fn auto_constraint(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    tools: &[Tool],
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    let Some(format) = reads(parser, names, asks.model(), templates)? else {
        return Ok(None);
    };
    Ok(asks.auto_calls(tools).and_then(|one_call| {
        auto_tag(
            &format,
            tools,
            reasoning_open,
            thought_closed,
            start,
            vocabulary,
            one_call,
        )
    }))
}

/// The tag of the calls `format` lets a model make under `auto`: no call required, one at most
/// when `one_call`, in the `vocabulary` the worker's grammar compiler takes; `None` when the
/// table has no tag for them or the worker's grammar backend has no spelling of it
/// ([`Unspelled`]): the request asked for no constraint, so none is refused.
pub(super) fn auto_tag(
    format: &Format,
    tools: &[Tool],
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
    vocabulary: TagVocabulary,
    one_call: bool,
) -> Option<ToolConstraint> {
    format
        .grammar_for(
            tools,
            Options {
                at_least_one: false,
                reasoning_open,
                thought_closed,
                one_call,
                start,
            },
        )
        .and_then(|grammar| structural_tag(grammar, vocabulary).ok())
}

/// A constraint an engine decodes under: a JSON schema, or a structural tag. The request builders
/// take it as its wire pair ([`Self::to_tuple`]).
#[derive(Debug, Clone)]
pub(crate) enum ToolConstraint {
    /// A JSON schema the engine holds the output to, from the first token or once the model's
    /// thought is over, by where the worker starts the grammar. The kind does not pick the
    /// reader: a table reads a schema it built, and a forced call's schema it did not build
    /// where it reads the response; the generic JSON parser reads one only on the legacy path
    /// (`regular::stages::reading::Prepared::json_schema`, `regular::symphony::assign_for`).
    JsonSchema(String),
    /// A structural tag: the answer carries the model's own framing, read by the model's parser.
    StructuralTag(String),
}

impl ToolConstraint {
    /// The kind and the text, as the request builders take them.
    pub(crate) fn to_tuple(&self) -> (String, String) {
        match self {
            Self::JsonSchema(schema) => ("json_schema".to_string(), schema.clone()),
            Self::StructuralTag(tag) => ("structural_tag".to_string(), tag.clone()),
        }
    }

    pub(crate) fn is_json_schema(&self) -> bool {
        matches!(self, Self::JsonSchema(_))
    }
}

impl From<tool_parser::ToolConstraint> for ToolConstraint {
    /// The older registry's constraint, kind for kind.
    fn from(constraint: tool_parser::ToolConstraint) -> Self {
        match constraint {
            tool_parser::ToolConstraint::JsonSchema(schema) => Self::JsonSchema(schema),
            tool_parser::ToolConstraint::StructuralTag(tag) => Self::StructuralTag(tag),
        }
    }
}

/// The constraint preparation finds for what a request asks, before its worker is known
/// ([`constraint_for`]): a forced call's tag, the table's or the older registry's, or the call's
/// JSON schema where the table's tag opens with free text; a JSON answer's tag as it derives for
/// a worker's start; or why the table cannot constrain the request. Request building decides on
/// the selected worker ([`worker_constraint`]).
pub(crate) type PreparedConstraint = Result<Prepared, SymphonyError>;

/// What preparation found for a request's constraint, before its worker is known.
#[derive(Debug)]
pub(crate) enum Prepared {
    /// Nothing: the request asks for no constraint the table or the older registry gives.
    None,
    /// A constraint every worker takes as it is: a forced call's tag, the table's built for the
    /// start the model's workers name together ([`grammar_start_of`]) or the older registry's,
    /// which request building shapes for the worker (`utils::tool_constraints_for_worker`), or
    /// the call's JSON schema where the table's tag opens with free text ([`Self::held_by`]).
    Constraint {
        constraint: ToolConstraint,
        /// The table built it for an engine that starts the grammar at the first token (the
        /// thought, closed, ahead of the calls), the start the model's workers name together.
        /// One no table built is shaped for the worker at request building, which says then
        /// whether it led the tag; here it is `false` ([`WorkerConstraint::first_token`]).
        first_token: bool,
        /// The table built it (its tag for the forced call, `calls_tag`), so the table reads
        /// the response with its own markers whatever the constraint's kind; `false` for the
        /// older registry's and the forced call's JSON schema, whose JSON the table reads with its
        /// forced call, and for the request's own schema where no table reads
        /// ([`WorkerConstraint::table`]).
        table: bool,
    },
    /// A forced call's tag a registry table built that opens with free text ([`calls_tag`]),
    /// which does not hold the model to the call. Preparation gives the call's JSON schema
    /// in its place, or keeps the tag as the table's own where that schema cannot be built
    /// ([`Self::held_by`]).
    OpensWithFreeText {
        constraint: ToolConstraint,
        /// As [`Self::Constraint`]'s.
        first_token: bool,
    },
    /// A JSON answer's tag, which derives for the selected worker's own start as the `auto` tag
    /// does ([`AnswerTag`]): the schema alone behind a reasoning parser's gate or where the engine
    /// does not say, the closed thought then the schema from the first token. A pool whose
    /// workers start the grammar at different places so gives each worker its own shape, where
    /// one tag for the whole pool fit one kind and left the other the free-text prefix, which
    /// says nothing of where the schema begins.
    Answer(Arc<AnswerTag>),
}

impl Prepared {
    /// What holds a forced call where the table's tag opens with free text
    /// ([`Self::OpensWithFreeText`]): the constraint `schema` gives, the forced call's JSON
    /// schema, which holds the model to the call wherever the engine starts the grammar; or the
    /// table's own tag where `schema` gives none, so that a request that schema cannot spell is
    /// served, not refused. Anything else stays as found, and `schema` is not asked.
    pub(crate) fn held_by(self, schema: impl FnOnce() -> Option<ToolConstraint>) -> Self {
        match self {
            Self::OpensWithFreeText {
                constraint,
                first_token,
            } => match schema() {
                Some(schema) => Self::from(Some(schema)),
                None => Self::Constraint {
                    constraint,
                    first_token,
                    table: true,
                },
            },
            found => found,
        }
    }

    /// The constraint a worker whose engine starts the grammar at `start` and whose compiler
    /// takes `vocabulary` takes: a JSON answer's tag derived for it, what was found as it is
    /// otherwise, a marked tag too. The tests read what preparation found through it; the
    /// decision ([`worker_constraint`]) reads each variant itself.
    #[cfg(test)]
    pub(crate) fn for_start(
        self,
        start: GrammarStart,
        vocabulary: TagVocabulary,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        match self {
            Self::None => Ok(None),
            Self::Constraint { constraint, .. } | Self::OpensWithFreeText { constraint, .. } => {
                Ok(Some(constraint))
            }
            Self::Answer(answer) => answer.for_start(start, vocabulary),
        }
    }
}

impl From<Option<ToolConstraint>> for Prepared {
    /// The older registry's constraint, as it built it (or a forced call's JSON schema in a
    /// tag's place, or the request's own schema where no table reads): request building shapes
    /// it for the worker and says whether it led the tag; no table built it.
    fn from(constraint: Option<ToolConstraint>) -> Self {
        constraint.map_or(Self::None, |constraint| Self::Constraint {
            constraint,
            first_token: false,
            table: false,
        })
    }
}

/// The table's tag for a JSON answer, as it derives for where a worker's engine starts the grammar
/// on a prompt that leaves the thought open ([`answer_tag`]): preparation builds it before the
/// worker is known, request building derives the tag for the worker it selected
/// ([`worker_constraint`]), and the attempt stamp for the worker each retry takes, whose start may
/// differ (`helpers::ToolConstraintStamp`). It owns what [`answer_tag`] reads, since the request
/// is gone from the pipeline once the plan is built: the table (shared), the request's schema, and
/// where the prompt leaves the model. Where the tag is the schema alone, the request's own schema
/// is the engine's constraint ([`TagForStart::schema`], serialized only when a retry's worker takes
/// it), and the worker a retry takes gets the one in the other's place.
pub(crate) struct AnswerTag {
    format: Arc<Format>,
    /// The table is the response template's, which takes its tag either way and refuses what it
    /// cannot constrain ([`tagged`]).
    compiled: bool,
    model: String,
    /// The request's schema, the one copy, shared with every grammar the tag builds.
    schema: Arc<Value>,
    /// The prompt leaves the model inside its thought.
    reasoning_open: bool,
    /// The prompt closed the thought in the model's place and its template writes nothing after
    /// the close: the thought is over where the prompt ends.
    past_the_thought: bool,
    /// The prompt continues the assistant's message.
    continues: bool,
}

impl AnswerTag {
    /// The tag for an engine whose grammar starts at `start`, in the `vocabulary` its compiler
    /// takes; `None` where the answer is the schema alone, the engine's own constraint, or the
    /// table has no tag for it; an error where the response template's table cannot constrain
    /// the answer.
    pub(crate) fn for_start(
        &self,
        start: GrammarStart,
        vocabulary: TagVocabulary,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        // A prompt past the thought leaves the answer to start where the prompt ends, wherever
        // the engine starts its grammar: asked as after the thought, the table gives the answer
        // alone, with no thought and no second close before it.
        let (reasoning_open, start) = if self.past_the_thought {
            (true, GrammarStart::AfterReasoning)
        } else {
            (self.reasoning_open, start)
        };
        answer_tag(
            &self.format,
            self.compiled,
            &self.model,
            &self.schema,
            reasoning_open,
            start,
            vocabulary,
            self.continues,
        )
    }
}

impl TagForStart for AnswerTag {
    fn tag_for(
        &self,
        start: WorkerGrammarStart,
        vocabulary: TagVocabulary,
    ) -> Result<Option<String>, String> {
        self.for_start(grammar_start_from(start), vocabulary)
            .map(tag_text)
            .map_err(String::from)
    }

    fn schema(&self) -> Option<&Value> {
        Some(self.schema.as_ref())
    }
}

impl fmt::Debug for AnswerTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnswerTag")
            .field("table", &self.format.name())
            .field("compiled", &self.compiled)
            .field("model", &self.model)
            .field("reasoning_open", &self.reasoning_open)
            .field("past_the_thought", &self.past_the_thought)
            .field("continues", &self.continues)
            .finish_non_exhaustive()
    }
}

/// The text of a table's structural tag; the schema kind, a JSON answer spelled as the schema
/// alone ([`structural_tag`]), is `None` here, since the attempt stamp writes the answer's own
/// schema in a tag's place ([`TagForStart::schema`]).
fn tag_text(constraint: Option<ToolConstraint>) -> Option<String> {
    match constraint? {
        ToolConstraint::StructuralTag(tag) => Some(tag),
        ToolConstraint::JsonSchema(_) => None,
    }
}

/// What constrains a request's output on the worker selected for it.
pub(crate) struct WorkerConstraint {
    pub constraint: Option<ToolConstraint>,
    /// The constraint is the whole turn's grammar, which holds the request's own format: the
    /// engine's request carries none beside it.
    pub whole_turn: bool,
    /// The worker's policy chose the constraint (the whole turn's grammar or the `auto` tag), not
    /// the request: a worker of another policy would not send it.
    pub by_policy: bool,
    /// What derives the constraint for a worker's grammar start, where the table shapes it by the
    /// start: the `auto` tag the policy chose ([`AutoTag::for_start`]) or a JSON answer's
    /// ([`AnswerTag::for_start`]). The attempt stamp gives the worker a retry takes the shape its
    /// own start needs (`helpers::ToolConstraintStamp`).
    pub table_tag: Option<Arc<dyn TagForStart>>,
    /// The constraint is shaped for an engine that starts the grammar at the first generated
    /// token: the whole turn's grammar, a tag derived for [`GrammarStart::FirstToken`] (the
    /// thought, closed, ahead of the answer or the calls), or the table's forced-call tag built
    /// for a pool that starts there ([`Prepared::Constraint`]). The engine's request says so
    /// where its dialect has a switch for the gate behind a reasoning parser, so an engine that
    /// gates its grammars there starts this one at once; a constraint shaped for another start,
    /// or none, starts where the engine does. Never a constraint no table built: request
    /// building shapes that one for the worker and says whether it led the tag.
    pub first_token: bool,
    /// The table built the constraint (the whole turn's grammar, the `auto` tag, a JSON
    /// answer's tag, a forced call's), so the table reads the response whatever the
    /// constraint's kind, which a backend's spelling may change; `false` for a forced call's
    /// constraint no table built (the older registry's, or the call's JSON schema in a tag's
    /// place), which the table reads with its forced call, and where nothing constrains.
    /// Symphony's assignment reads one no table built by its kind alone
    /// (`regular::symphony::assign_for`).
    pub table: bool,
}

/// The table's `auto` tag for a request, derived for where a worker's engine starts the grammar on
/// a prompt that leaves the thought open: request building shapes it for the worker it selected,
/// and the attempt stamp for the worker each retry takes, whose start may differ (the retry filter
/// compares the policy, not the start: [`WorkerDecoding::decides_alike`]). It owns what
/// [`auto_constraint`] reads, since the request is gone from the pipeline once the plan is built.
/// It goes with the plan only when the first worker's start got a tag: where the table gives that
/// start none, the plan carries no tag and no stamp, and a retry to a start the table would give
/// one is not covered (no registry table gives one of two starts a retry may cross between the tag
/// and the other none; the whole turn keeps the first token apart).
pub(crate) struct AutoTag {
    pub parser: ParserChoice,
    /// The model's parser names, both sources owned ([`ModelNames`] borrows the flags from the
    /// resolver, which the stamp outlives).
    pub labels: Names,
    pub flags: Names,
    /// The served models an alias reads, as the context of the names read them
    /// ([`ModelNames`]); `None` where it read none.
    pub by_alias: Option<Arc<[ByAlias]>>,
    pub request: AutoRequest,
    /// The prompt leaves the model inside its thought.
    pub reasoning_open: bool,
    /// The prompt closed the thought itself and wrote the template's text after it, which the tag
    /// owes nothing of ([`Format::thought_closed`]).
    pub thought_closed: bool,
    pub templates: Arc<Templates>,
}

/// The request an [`AutoTag`] derives for, and the tools its tag is over.
pub(crate) enum AutoRequest {
    /// A chat request's, over the tools its `tool_choice` leaves callable.
    Chat(Arc<ChatCompletionRequest>),
    /// A Messages request's, over `tools` as the template declares them.
    Messages {
        request: Arc<CreateMessageRequest>,
        /// The tools as the template declares them: the spec's own slice, shared, unless
        /// `tool_choice` narrows it (then the narrowed list, owned here).
        tools: Arc<[Tool]>,
        /// The prompt continues the assistant's message.
        continues: bool,
        /// The request declared tools, as its response spec recorded ([`Asks::Messages`]).
        has_tools: bool,
    },
}

impl AutoTag {
    /// The tag for an engine whose grammar starts at `start`, in the `vocabulary` its compiler
    /// takes; `None` where the table gives that engine none.
    pub(crate) fn for_start(
        &self,
        start: GrammarStart,
        vocabulary: TagVocabulary,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        let (asks, tools) = match &self.request {
            AutoRequest::Chat(request) => (Asks::Chat(request), request.callable_tools()),
            AutoRequest::Messages {
                request,
                tools,
                continues,
                has_tools,
            } => (
                Asks::Messages {
                    request,
                    continues: *continues,
                    has_tools: *has_tools,
                },
                Cow::Borrowed(&tools[..]),
            ),
        };
        auto_constraint(
            self.parser,
            &ModelNames {
                labels: self.labels.clone(),
                flags: &self.flags,
                by_alias: self.by_alias.as_deref().unwrap_or_default(),
            },
            asks,
            &tools,
            self.reasoning_open,
            self.thought_closed,
            start,
            vocabulary,
            &self.templates,
        )
    }

    /// The model the tag is for.
    fn model(&self) -> &str {
        match &self.request {
            AutoRequest::Chat(request) => &request.model,
            AutoRequest::Messages { request, .. } => &request.model,
        }
    }
}

impl TagForStart for AutoTag {
    fn tag_for(
        &self,
        start: WorkerGrammarStart,
        vocabulary: TagVocabulary,
    ) -> Result<Option<String>, String> {
        self.for_start(grammar_start_from(start), vocabulary)
            .map(tag_text)
            .map_err(String::from)
    }

    /// The `auto` tag stands in for nothing: a worker the table gives none, or whose grammar
    /// backend spells none, goes unconstrained.
    fn schema(&self) -> Option<&Value> {
        None
    }
}

impl fmt::Debug for AutoTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tag = f.debug_struct("AutoTag");
        tag.field("parser", &self.parser)
            .field("model", &self.model())
            .field("reasoning_open", &self.reasoning_open)
            .field("thought_closed", &self.thought_closed);
        if let AutoRequest::Messages {
            tools, continues, ..
        } = &self.request
        {
            tag.field("tools", &tools.len())
                .field("continues", continues);
        }
        tag.finish_non_exhaustive()
    }
}

/// What constrains a request's output on the worker selected for it, whose engine takes a grammar
/// on every turn or only when the request asks, starts it where its label says and takes the
/// tags in the vocabulary its label says (`decoding`, under PD the decode worker's). The one
/// place a worker's policy decides:
///
/// - `requested`, the default: what preparation found (`prepared`), its refusal included; a JSON
///   answer's tag derived for the worker's start and vocabulary ([`AnswerTag::for_start`]).
/// - `always`, on an engine that starts the grammar at the first token: the whole turn's grammar
///   (`turn`, [`turn_constraint`]) where it starts; its refusal refuses the request.
/// - `always` otherwise, or where the whole turn's grammar does not start: what preparation found,
///   and when that is nothing, the table's tag for the calls `auto` allows (`auto`, the request's
///   [`AutoTag`], derived for the worker's start).
///
/// When the policy chose the constraint, a retry re-selects only workers that decide alike
/// ([`WorkerDecoding::decides_alike`]); the `auto` tag and a JSON answer's go with the plan as
/// they derive, for the retry's worker to take in the shape its own start needs.
pub(crate) fn worker_constraint(
    decoding: WorkerDecoding,
    prepared: PreparedConstraint,
    turn: impl FnOnce() -> Result<Option<ToolConstraint>, SymphonyError>,
    auto: impl FnOnce() -> AutoTag,
) -> Result<WorkerConstraint, SymphonyError> {
    let always = decoding.policy == ConstrainedDecoding::Always;
    if decoding.takes_the_whole_turn() {
        if let Some(constraint) = turn()? {
            return Ok(WorkerConstraint {
                constraint: Some(constraint),
                whole_turn: true,
                by_policy: true,
                table_tag: None,
                first_token: true,
                table: true,
            });
        }
    }
    let start = grammar_start_from(decoding.start);
    let vocabulary = decoding.vocabulary;
    // A tag derived here is shaped for the selected worker's start; a prepared constraint says
    // for itself whether the table built it for the first token, and one no table built is
    // shaped and spoken for at request building (`utils::tool_constraints_for_worker`).
    let derived_for_first_token =
        |tag: &Option<ToolConstraint>| tag.is_some() && start == GrammarStart::FirstToken;
    let (constraint, by_policy, table_tag, first_token, table) = match prepared? {
        Prepared::None if always => {
            let auto = auto();
            let tag = auto.for_start(start, vocabulary)?;
            let by_policy = tag.is_some();
            let first_token = derived_for_first_token(&tag);
            let table_tag = by_policy.then(|| -> Arc<dyn TagForStart> { Arc::new(auto) });
            (tag, by_policy, table_tag, first_token, by_policy)
        }
        Prepared::Answer(answer) => {
            let tag = answer.for_start(start, vocabulary)?;
            let first_token = derived_for_first_token(&tag);
            let table = tag.is_some();
            (
                tag,
                false,
                Some(answer as Arc<dyn TagForStart>),
                first_token,
                table,
            )
        }
        Prepared::Constraint {
            constraint,
            first_token,
            table,
        } => (Some(constraint), false, None, first_token, table),
        // Preparation resolves a marked tag ([`Prepared::held_by`]); one nothing resolved is
        // the table's tag as it was built.
        Prepared::OpensWithFreeText {
            constraint,
            first_token,
        } => (Some(constraint), false, None, first_token, true),
        Prepared::None => (None, false, None, false, false),
    };
    Ok(WorkerConstraint {
        constraint,
        whole_turn: false,
        by_policy,
        table_tag,
        first_token,
        table,
    })
}

/// The whole turn's grammar for a request ([`Format::turn_grammar`]), where the response
/// template's table reads the response ([`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)); `declared` is the tools the template
/// sees (a chat request's effective tool list, a Messages request's chat tools) and `prompt` the
/// rendered prompt; `vocabulary` the one the worker's grammar compiler takes. `Ok(None)` for any
/// other table, and where the turn's grammar does not start; an error refuses what it cannot
/// hold.
pub(crate) fn turn_constraint(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    declared: &[Tool],
    prompt: &str,
    vocabulary: TagVocabulary,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    match table_reading(parser, names, asks.model(), templates)? {
        Some((format, true)) => turn_of(&format, asks, declared, prompt, vocabulary),
        _ => Ok(None),
    }
}

/// [`turn_constraint`] over `format`: the request's tools and choice, its parallel use, and the
/// answer's format, a chat request's `response_format`, `regex` or `ebnf` (the protocol refuses
/// more than one) or a Messages request's `output_config.format`. `Ok(None)` for a structural-tag
/// `response_format`: no turn's grammar holds a tag of the request's own, so none starts and the
/// tag goes to the engine as it arrived.
pub(super) fn turn_of(
    format: &Format,
    asks: Asks<'_>,
    declared: &[Tool],
    prompt: &str,
    vocabulary: TagVocabulary,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    if asks.own_structural_tag() {
        return Ok(None);
    }
    let choice = asks.choice();
    let callable = callable(choice.as_deref(), declared);
    let schema = asks.answer_schema();
    whole_turn(
        format,
        &Turn {
            prompt,
            continued: asks.continues(),
            declared,
            callable: &callable,
            choice: output_choice(choice.as_deref()),
            parallel: asks.parallel(),
            answer: asks.answer(schema.as_deref()),
        },
        vocabulary,
    )
}

/// How a tool choice shapes a turn: one call for a named function, one or more where the choice
/// forces a call, an answer or calls otherwise.
fn output_choice(choice: Option<&ToolChoice>) -> OutputChoice {
    match choice {
        Some(ToolChoice::Function { .. }) => OutputChoice::One,
        Some(choice) if choice.forces_call() => OutputChoice::Required,
        _ => OutputChoice::Auto,
    }
}

/// `format`'s grammar for `turn` as the engine's structural tag, in the `vocabulary` the
/// worker's grammar compiler takes; refused by the backend's name when that backend has no
/// spelling of it ([`Unspelled`]): the engine takes the whole turn, so there is no going without.
pub(super) fn whole_turn(
    format: &Format,
    turn: &Turn<'_>,
    vocabulary: TagVocabulary,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    format
        .turn_grammar(turn)
        .map_err(SymphonyError::Constraint)?
        .map(|grammar| {
            structural_tag(grammar, vocabulary)
                .map_err(|unspelled| SymphonyError::Constraint(unspelled.to_string()))
        })
        .transpose()
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/symphony/constraint/tests.rs"]
mod tests;

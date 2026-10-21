//! Constraint derivation: what a request asks of the table (`Asks`), the structural tags for a
//! forced call, a JSON answer or the calls `auto` allows, the whole turn's grammar, and the
//! decision on the selected worker.

use std::{borrow::Cow, fmt, sync::Arc};

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
    worker::ConstrainedDecoding,
};
use serde_json::{json, Value};
use symphony::{
    grammar::{GrammarStart, Options},
    Format, Grammar, OutputAnswer, OutputChoice, Turn,
};
use tool_parser::{GrammarStart as WorkerGrammarStart, ToolConstraint};

use super::{
    names::{reads, table_reading},
    template::Templates,
    SymphonyError,
};
use crate::{
    config::ParserChoice,
    routers::grpc::{
        common::stages::helpers::TagForStart,
        spec::scans_for_tool_calls,
        utils::{ModelNames, Names, WorkerDecoding},
    },
    worker::{Worker, WorkerType},
};

/// The schema a JSON `response_format` asks the answer to follow: any object for `json_object`.
fn answer_schema(response_format: Option<&ResponseFormat>) -> Option<Value> {
    match response_format? {
        ResponseFormat::JsonObject => Some(json!({"type": "object"})),
        ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        ResponseFormat::Text => None,
    }
}

/// The constraint `grammar` gives, written straight from the tree as the request carries it
/// ([`Grammar::payload_text`], the bytes of `payload().to_string()` without the JSON tree in
/// between), for the response template's table too (`compiled`, [`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)): a registry
/// table with no tag leaves the older registry to decide, as before; the template's table refuses
/// instead, since its output has no other syntax the older path could constrain and the table
/// still read, and so does a prompt that `continues` the assistant's message, since the tag
/// writes a turn from its start.
pub(super) fn tagged(
    grammar: Option<Grammar>,
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
        Some(grammar) => Ok(Some(ToolConstraint::StructuralTag(grammar.payload_text()))),
        None if compiled => Err(SymphonyError::Constraint(format!(
            "model '{model}' reads its output with its response template, whose table has no \
             structural tag for {asked}"
        ))),
        None => Ok(None),
    }
}

/// The tag of a forced call to one of `tools`, at most one when `one_call`, with
/// `reasoning_open` when the prompt leaves the model inside its thought and `thought_closed` when
/// it closed the thought itself and wrote the template's text after it ([`tagged`]).
#[expect(
    clippy::too_many_arguments,
    reason = "the table, the request's ask, where the prompt and the engine leave the model"
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
    continues: bool,
) -> Result<Option<ToolConstraint>, SymphonyError> {
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
    tagged(grammar, compiled, model, "a forced tool choice", continues)
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
/// it cannot constrain ([`tagged`]).
fn answer_tag(
    format: &Format,
    compiled: bool,
    model: &str,
    schema: &Value,
    reasoning_open: bool,
    start: GrammarStart,
    continues: bool,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    let grammar = format.answer_grammar_for(schema, reasoning_open, start);
    if !compiled && (continues || matches!(grammar, Some(Grammar::JsonSchema { .. }))) {
        return Ok(None);
    }
    tagged(grammar, compiled, model, "a JSON answer", continues)
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
/// them names in its `constrained_decoding_start` label, a prefill or encode worker left out
/// since it generates no token a grammar constrains (the roles that do are allow-listed, so a
/// role added later is left out until it is named); `Either` when one that starts at the first
/// token stands beside one that does not, since no one tag fits both (the answer keeps the
/// prefix a forced call takes: [`Format::answer_grammar_for`]); unknown when none says, and
/// when workers that start after their reasoning parser stand beside ones that do not say,
/// since the unknown start's tags fit both (a forced call's prefix, an answer's schema alone).
pub(crate) fn grammar_start_of(workers: &[Arc<dyn Worker>]) -> GrammarStart {
    let mut starts = workers
        .iter()
        .filter(|worker| {
            matches!(
                worker.worker_type(),
                WorkerType::Regular | WorkerType::Decode
            )
        })
        .map(|worker| {
            WorkerGrammarStart::from_label(
                worker
                    .metadata()
                    .spec
                    .labels
                    .get(WorkerGrammarStart::LABEL)
                    .map(String::as_str),
            )
        });
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
    },
}

impl<'a> Asks<'a> {
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

    /// The request carries a `regex` or `ebnf` constraint, which applies outside a template's
    /// format; a Messages request carries neither.
    fn constrains_outside_the_format(self) -> bool {
        match self {
            Self::Chat(request) => request.regex.is_some() || request.ebnf.is_some(),
            Self::Messages { .. } => false,
        }
    }

    /// The JSON schema the request's answer must fit; a Messages request asks none (an
    /// `output_format` is refused before this).
    pub(super) fn answer_schema(self) -> Option<Value> {
        match self {
            Self::Chat(request) => answer_schema(request.response_format.as_ref()),
            Self::Messages { .. } => None,
        }
    }

    /// Whether the request's choice forces a call to one of `tools`, and then whether one call at
    /// most: a chat `tool_choice` that forces a call ([`ToolChoice::forces_call`]), one call for a
    /// named function or `parallel_tool_calls: false`; a Messages `tool_choice` of `any` or a
    /// named tool, one call for the named tool or `disable_parallel_tool_use`.
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
                Some(MessagesToolChoice::Tool { .. }) => Some(true),
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
    /// displace (a JSON `response_format`, a `regex`, an `ebnf`), takes none; a Messages choice
    /// that forces a call takes none, its tag being [`constraint`]'s.
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
                let own = self.answer_schema().is_some() || self.constrains_outside_the_format();
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
    /// request (a `tool_choice` of `none`; a chat response it would not scan for calls, no tools
    /// declared anywhere, for a provider that reads calls only with them: [`scans_for_tool_calls`];
    /// a Messages request that declares no tools, an empty list being a declaration).
    pub(super) fn no_calls(self) -> bool {
        match self {
            Self::Chat(request) => {
                matches!(
                    request.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                ) || !scans_for_tool_calls(request)
            }
            Self::Messages { request, .. } => {
                matches!(request.tool_choice, Some(MessagesToolChoice::None))
                    || request.tools.is_none()
            }
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

/// What the table gives a request's forced or required call to one of `tools`, or a chat request's
/// JSON `response_format`, with `reasoning_open` when the prompt leaves the model inside its
/// thought: the call's structural tag, built for `start`, where the model's workers start the
/// grammar together ([`calls_tag`], [`grammar_start_of`]); the answer's as it derives for the
/// selected worker's own start ([`Prepared::Answer`], [`AnswerTag`]), since a pool whose workers
/// start the grammar at different places has no one answer tag that fits them all.
/// [`Prepared::None`] when no table reads the response ([`reads`]), the request asks neither, or
/// the table has no tag for the call; the older registry then decides as before. An error refuses
/// the request: the response template's table cannot constrain the call it is asked (the answer's
/// refusal comes with the derivation), or a chat request's `regex` or `ebnf` constraint would
/// apply outside its format. `thought_closed` says the prompt closed the thought itself and wrote
/// the template's text after it, which the calls' tag then owes nothing of.
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
    templates: &Templates,
) -> PreparedConstraint {
    let Some((format, compiled)) = table_reading(parser, names, asks.model(), templates)? else {
        return Ok(Prepared::None);
    };
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
            continues,
        )
        .map(Prepared::from);
    }
    Ok(match asks.answer_schema() {
        Some(schema) => Prepared::Answer(Arc::new(AnswerTag {
            format,
            compiled,
            model: asks.model().to_string(),
            schema,
            reasoning_open,
            continues,
        })),
        None => Prepared::None,
    })
}

/// The tag of the calls a request's model may make under `auto`, for a worker whose engine takes
/// a grammar on every turn ([`ConstrainedDecoding::Always`]): the table's grammar for the request's
/// callable `tools` with no call required, at most one when parallel calls are off, and
/// `reasoning_open` when the prompt leaves the model inside its thought. The request asked for
/// nothing, so nothing is refused: `Ok(None)` when no table reads the response ([`reads`]), the
/// request takes no tag for its calls ([`Asks::auto_calls`]), or the table has none for them. A
/// thought and an answer alone are not constrained today. `thought_closed` as for [`constraint`].
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
            one_call,
        )
    }))
}

/// The tag of the calls `format` lets a model make under `auto`: no call required, one at most
/// when `one_call`; `None` when the table has no tag for them.
pub(super) fn auto_tag(
    format: &Format,
    tools: &[Tool],
    reasoning_open: bool,
    thought_closed: bool,
    start: GrammarStart,
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
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload_text()))
}

/// The constraint preparation finds for what a request asks, before its worker is known
/// ([`constraint`]): a forced call's tag, the table's or the older registry's; a JSON answer's tag
/// as it derives for a worker's start; or why the table cannot constrain the request. Request
/// building decides on the selected worker ([`worker_constraint`]).
pub(crate) type PreparedConstraint = Result<Prepared, SymphonyError>;

/// What preparation found for a request's constraint, before its worker is known.
#[derive(Debug)]
pub(crate) enum Prepared {
    /// Nothing: the request asks for no constraint the table or the older registry gives.
    None,
    /// A constraint every worker takes as it is: a forced call's tag, the table's built for the
    /// start the model's workers name together ([`grammar_start_of`]) or the older registry's,
    /// which request building shapes for the worker (`utils::tool_constraints_for_worker`).
    Constraint(ToolConstraint),
    /// A JSON answer's tag, which derives for the selected worker's own start as the `auto` tag
    /// does ([`AnswerTag`]): the schema alone behind a reasoning parser's gate or where the engine
    /// does not say, the closed thought then the schema from the first token. A pool whose
    /// workers start the grammar at different places so gives each worker its own shape, where
    /// one tag for the whole pool fit one kind and left the other the free-text prefix, which
    /// says nothing of where the schema begins.
    Answer(Arc<AnswerTag>),
}

impl Prepared {
    /// The constraint a worker whose engine starts the grammar at `start` takes: a JSON answer's
    /// tag derived for it, what was found as it is otherwise.
    pub(crate) fn for_start(
        self,
        start: GrammarStart,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        match self {
            Self::None => Ok(None),
            Self::Constraint(constraint) => Ok(Some(constraint)),
            Self::Answer(answer) => answer.for_start(start),
        }
    }
}

impl From<Option<ToolConstraint>> for Prepared {
    fn from(constraint: Option<ToolConstraint>) -> Self {
        constraint.map_or(Self::None, Self::Constraint)
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
    schema: Value,
    /// The prompt leaves the model inside its thought.
    reasoning_open: bool,
    /// The prompt continues the assistant's message.
    continues: bool,
}

impl AnswerTag {
    /// The tag for an engine whose grammar starts at `start`; `None` where the answer is the
    /// schema alone, the engine's own constraint, or the table has no tag for it; an error where
    /// the response template's table cannot constrain the answer.
    pub(crate) fn for_start(
        &self,
        start: GrammarStart,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        answer_tag(
            &self.format,
            self.compiled,
            &self.model,
            &self.schema,
            self.reasoning_open,
            start,
            self.continues,
        )
    }
}

impl TagForStart for AnswerTag {
    fn tag_for(&self, start: WorkerGrammarStart) -> Result<Option<String>, String> {
        self.for_start(grammar_start_from(start))
            .map(tag_text)
            .map_err(String::from)
    }

    fn schema(&self) -> Option<&Value> {
        Some(&self.schema)
    }
}

impl fmt::Debug for AnswerTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnswerTag")
            .field("table", &self.format.name())
            .field("compiled", &self.compiled)
            .field("model", &self.model)
            .field("reasoning_open", &self.reasoning_open)
            .field("continues", &self.continues)
            .finish_non_exhaustive()
    }
}

/// The text of a table's structural tag; the schema kind is not built by a table ([`tagged`]).
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
    },
}

impl AutoTag {
    /// The tag for an engine whose grammar starts at `start`; `None` where the table gives that
    /// engine none.
    pub(crate) fn for_start(
        &self,
        start: GrammarStart,
    ) -> Result<Option<ToolConstraint>, SymphonyError> {
        let (asks, tools) = match &self.request {
            AutoRequest::Chat(request) => (Asks::Chat(request), request.callable_tools()),
            AutoRequest::Messages {
                request,
                tools,
                continues,
            } => (
                Asks::Messages {
                    request,
                    continues: *continues,
                },
                Cow::Borrowed(&tools[..]),
            ),
        };
        auto_constraint(
            self.parser,
            &ModelNames {
                labels: self.labels.clone(),
                flags: &self.flags,
            },
            asks,
            &tools,
            self.reasoning_open,
            self.thought_closed,
            start,
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
    fn tag_for(&self, start: WorkerGrammarStart) -> Result<Option<String>, String> {
        self.for_start(grammar_start_from(start))
            .map(tag_text)
            .map_err(String::from)
    }

    /// The `auto` tag stands in for nothing: a worker the table gives none goes unconstrained.
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
/// on every turn or only when the request asks and starts it where its label says (`decoding`,
/// under PD the decode worker's). The one place a worker's policy decides:
///
/// - `requested`, the default: what preparation found (`prepared`), its refusal included; a JSON
///   answer's tag derived for the worker's start ([`Prepared::for_start`]).
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
            });
        }
    }
    let start = grammar_start_from(decoding.start);
    let (constraint, by_policy, table_tag) = match prepared? {
        Prepared::None if always => {
            let auto = auto();
            let tag = auto.for_start(start)?;
            let by_policy = tag.is_some();
            let table_tag = by_policy.then(|| -> Arc<dyn TagForStart> { Arc::new(auto) });
            (tag, by_policy, table_tag)
        }
        Prepared::Answer(answer) => {
            let tag = answer.for_start(start)?;
            (tag, false, Some(answer as Arc<dyn TagForStart>))
        }
        prepared => (prepared.for_start(start)?, false, None),
    };
    Ok(WorkerConstraint {
        constraint,
        whole_turn: false,
        by_policy,
        table_tag,
    })
}

/// The whole turn's grammar for a request ([`Format::turn_grammar`]), where the response
/// template's table reads the response ([`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)); `declared` is the tools the template
/// sees (a chat request's effective tool list, a Messages request's chat tools) and `prompt` the
/// rendered prompt. `Ok(None)` for any other table, and where the turn's grammar does not start;
/// an error refuses what it cannot hold.
pub(crate) fn turn_constraint(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    declared: &[Tool],
    prompt: &str,
    templates: &Templates,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    match table_reading(parser, names, asks.model(), templates)? {
        Some((format, true)) => turn_of(&format, asks, declared, prompt),
        _ => Ok(None),
    }
}

/// [`turn_constraint`] over `format`: the request's tools and choice, its parallel use, and a chat
/// request's `response_format`, `regex` or `ebnf` as the answer's (the protocol refuses more than
/// one; a Messages request carries no answer format).
pub(super) fn turn_of(
    format: &Format,
    asks: Asks<'_>,
    declared: &[Tool],
    prompt: &str,
) -> Result<Option<ToolConstraint>, SymphonyError> {
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
            answer: asks.answer(schema.as_ref()),
        },
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

/// `format`'s grammar for `turn` as the engine's structural tag.
pub(super) fn whole_turn(
    format: &Format,
    turn: &Turn<'_>,
) -> Result<Option<ToolConstraint>, SymphonyError> {
    Ok(format
        .turn_grammar(turn)
        .map_err(SymphonyError::Constraint)?
        .map(|grammar| ToolConstraint::StructuralTag(grammar.payload_text())))
}

#[cfg(test)]
mod tests {
    use tool_parser::GrammarStart;

    use super::*;
    use crate::routers::grpc::regular::symphony::testing::{assign, constraint, *};

    /// Where a worker's engine starts the grammar shapes what precedes a table's answer or its
    /// calls when the prompt leaves the thought open: nothing after the reasoning parser's gate
    /// (a JSON answer then needs no tag; the request's own schema goes to the engine as it is),
    /// the thought closed first from the first token, the schema alone (the request's own, no
    /// tag) when the worker does not say, and for the calls the free-text prefix when the
    /// model's workers name different starts. The answer's tag derives for the selected worker,
    /// whatever start the pool names.
    #[test]
    fn the_workers_grammar_start_shapes_what_precedes_a_tables_answer_and_calls() {
        let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
        let asked = request(json!({"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": {"type": "object"}},
        }}));
        // The pool names either kind (a first-token worker beside another): the answer's tag is
        // not built for it, but for the worker each attempt selects.
        let prepared = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(&asked),
            &[],
            true,
            false,
            symphony::GrammarStart::Either,
            &templates(),
        );
        let Ok(Prepared::Answer(answer)) = prepared else {
            panic!("the answer's tag, derived per worker, not {prepared:?}");
        };
        let tag_of = |start| {
            answer
                .for_start(start)
                .expect("no hf")
                .map(|constraint| match constraint {
                    ToolConstraint::StructuralTag(tag) => {
                        serde_json::from_str::<Value>(&tag).expect("JSON")
                    }
                    ToolConstraint::JsonSchema(schema) => {
                        panic!("a structural tag, not the JSON schema {schema}")
                    }
                })
        };
        // Where the worker does not say, the answer takes no prefix either: free text before a
        // schema constrains nothing, so the schema stands alone and goes to the engine as the
        // request's own, as it does after the gate.
        assert_eq!(
            tag_of(symphony::GrammarStart::Unknown),
            None,
            "the schema stands alone where the worker does not say"
        );
        assert_eq!(
            tag_of(symphony::GrammarStart::AfterReasoning),
            None,
            "after the gate the schema stands alone, and needs no tag"
        );
        let first = tag_of(symphony::GrammarStart::FirstToken).expect("a tag");
        assert_eq!(first["format"]["elements"][0]["type"], "tag");
        assert_eq!(first["format"]["elements"][0]["begin"], "");
        assert_eq!(first["format"]["elements"][0]["end"], "</think>");
        assert_eq!(first["format"]["elements"][1]["type"], "json_schema");
        let forced = request(json!({"tool_choice": "required"}));
        let call_of = |start| {
            let Ok(Prepared::Constraint(ToolConstraint::StructuralTag(tag))) = super::constraint(
                ParserChoice::Symphony,
                &deepseek,
                Asks::Chat(&forced),
                &tools(),
                true,
                false,
                start,
                &templates(),
            ) else {
                panic!("a forced tag");
            };
            serde_json::from_str::<Value>(&tag).expect("JSON")
        };
        assert_eq!(
            call_of(symphony::GrammarStart::Unknown)["format"]["elements"][0]["type"],
            "any_text"
        );
        let bare = call_of(symphony::GrammarStart::AfterReasoning);
        assert_eq!(
            bare["format"]["elements"][0]["type"], "const_string",
            "the block's opener comes first at the gate: {bare}"
        );
        let first = call_of(symphony::GrammarStart::FirstToken);
        assert_eq!(first["format"]["elements"][0]["end"], "</think>");
        assert_eq!(first["format"]["elements"][1], bare["format"]);
    }

    /// A forced or required call on a table whose calls are given as data with JSON arguments
    /// takes the table's tag and the table reads the answer: the request is not handed to the
    /// older path. The older path's JSON-schema constraint still leaves the table out, as before.
    #[test]
    fn a_forced_call_on_the_kimi_k2_table_takes_its_tag_and_the_table_reads_the_answer() {
        let kimi = names(Some("kimi_k2"), Some("kimi_k2"));
        let prompt =
            "<|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|>";
        let named = request(json!({
            "model": "moonshotai/Kimi-K2-Thinking",
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }));
        let required = request(json!({
            "model": "moonshotai/Kimi-K2-Thinking",
            "tool_choice": "required",
        }));
        for (asked, one_call) in [(&named, true), (&required, false)] {
            let Ok(Prepared::Constraint(ToolConstraint::StructuralTag(tag))) = super::constraint(
                ParserChoice::Symphony,
                &kimi,
                Asks::Chat(asked),
                &tools(),
                false,
                false,
                symphony::GrammarStart::AfterReasoning,
                &templates(),
            ) else {
                panic!("the table's tag, not the older registry's schema");
            };
            let format = serde_json::from_str::<Value>(&tag).expect("JSON");
            let format = &format["format"];
            assert_eq!(format["type"], "triggered_tags");
            assert_eq!(
                format["triggers"],
                json!(["<|tool_calls_section_begin|>", "</think>"])
            );
            assert_eq!(format["at_least_one"], true);
            assert_eq!(
                format["stop_after_first"],
                if one_call { json!(true) } else { Value::Null }
            );
            let call = if one_call {
                &format["tags"][0]["content"]["elements"][0]
            } else {
                &format["tags"][0]["content"]["content"]["elements"][0]
            };
            assert_eq!(
                call["elements"][0]["value"],
                "<|tool_call_begin|>functions.lookup:"
            );
            assert_eq!(format["tags"][0]["end"], "<|tool_calls_section_end|>");
            // The constraint is a structural tag, so the table is assigned to the response.
            let structural = ("structural_tag".to_string(), tag);
            let assigned = assign(
                ParserChoice::Symphony,
                &kimi,
                asked,
                Some(&structural),
                prompt,
            )
            .expect("assigned");
            assert_eq!(assigned.format.name(), "kimi_k2");
        }
        // The older path's forced call (a JSON schema) is still read by the older parsers.
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(ParserChoice::Symphony, &kimi, &named, Some(&schema), prompt).is_none());
    }

    /// A forced or required call on a table whose calls are named arguments inside a block takes
    /// the table's tag and the table reads the answer: the request is not handed to the older
    /// path. The older path's JSON-schema constraint still leaves the table out, as before.
    #[test]
    fn a_forced_call_on_the_deepseek_v3_1_table_takes_its_tag_and_the_table_reads_the_answer() {
        let deepseek = names(Some("deepseek31"), Some("deepseek_v31"));
        let prompt = "<｜User｜>hi<｜Assistant｜></think>";
        let named = request(json!({
            "model": "deepseek-ai/DeepSeek-V3.1",
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }));
        let required = request(json!({
            "model": "deepseek-ai/DeepSeek-V3.1",
            "tool_choice": "required",
        }));
        for (asked, one_call) in [(&named, true), (&required, false)] {
            let Ok(Prepared::Constraint(ToolConstraint::StructuralTag(tag))) = super::constraint(
                ParserChoice::Symphony,
                &deepseek,
                Asks::Chat(asked),
                &tools(),
                false,
                false,
                symphony::GrammarStart::AfterReasoning,
                &templates(),
            ) else {
                panic!("the table's tag, not the older registry's schema");
            };
            let format = serde_json::from_str::<Value>(&tag).expect("JSON");
            let format = &format["format"];
            assert_eq!(format["type"], "triggered_tags");
            assert_eq!(
                format["triggers"],
                json!(["<｜tool▁calls▁begin｜>", "</think>"])
            );
            assert_eq!(format["at_least_one"], true);
            assert_eq!(
                format["stop_after_first"],
                if one_call { json!(true) } else { Value::Null }
            );
            let call = if one_call {
                &format["tags"][0]["content"]["elements"][0]
            } else {
                &format["tags"][0]["content"]["content"]["elements"][0]
            };
            assert_eq!(
                call["elements"][0]["value"],
                "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>"
            );
            assert_eq!(call["elements"][2]["value"], "<｜tool▁call▁end｜>");
            assert_eq!(format["tags"][0]["end"], "<｜tool▁calls▁end｜>");
            // The constraint is a structural tag, so the table is assigned to the response.
            let structural = ("structural_tag".to_string(), tag);
            let assigned = assign(
                ParserChoice::Symphony,
                &deepseek,
                asked,
                Some(&structural),
                prompt,
            )
            .expect("assigned");
            assert_eq!(assigned.format.name(), "deepseek_v3.1");
        }
        // The older path's forced call (a JSON schema) is still read by the older parsers.
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &deepseek,
            &named,
            Some(&schema),
            prompt
        )
        .is_none());
    }

    /// The model's workers name the start when they all carry the same label; none or a missing
    /// label leaves it unknown, as does a pool of workers that start after their reasoning parser
    /// beside ones that do not say, and a first-token worker beside another kind names either
    /// kind. A worker's own label maps one to one.
    #[test]
    fn the_models_workers_name_the_grammar_start_when_they_agree() {
        let worker = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(grammar_start_of(&[]), symphony::GrammarStart::Unknown);
        assert_eq!(
            grammar_start_of(&[worker(None)]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[
                worker(Some("after_reasoning")),
                worker(Some("after_reasoning"))
            ]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("first_token"))]),
            symphony::GrammarStart::FirstToken
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), worker(Some("first_token"))]),
            symphony::GrammarStart::Either
        );
        // After the reasoning parser beside not saying: the unknown start's tags fit both (a
        // forced call's prefix, an answer's schema alone), so the pool needs no prefix.
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), worker(None)]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("first_token")), worker(None)]),
            symphony::GrammarStart::Either
        );
        assert_eq!(
            grammar_start_of(&[
                worker(Some("after_reasoning")),
                worker(None),
                worker(Some("first_token"))
            ]),
            symphony::GrammarStart::Either
        );
        // A prefill worker generates no token a grammar constrains: its label does not count.
        let prefill = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://prefill:50051").worker_type(WorkerType::Prefill);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(
            grammar_start_of(&[
                worker(Some("after_reasoning")),
                prefill(Some("first_token"))
            ]),
            symphony::GrammarStart::AfterReasoning
        );
        // An encode worker (EPD) generates none either: its label does not count.
        let encode = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://encode:50051").worker_type(WorkerType::Encode);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), encode(Some("first_token"))]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[encode(Some("first_token"))]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_of(&[worker(Some("after_reasoning")), prefill(None)]),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_of(&[prefill(Some("first_token"))]),
            symphony::GrammarStart::Unknown
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::AfterReasoning),
            symphony::GrammarStart::AfterReasoning
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::FirstToken),
            symphony::GrammarStart::FirstToken
        );
        assert_eq!(
            grammar_start_from(WorkerGrammarStart::Unknown),
            symphony::GrammarStart::Unknown
        );
    }

    /// A prompt the template leaves without a turn opener (a system message rendered in place
    /// after the user turn) starts the model inside its thought on the DeepSeek V4 table, and the
    /// grammar built for the request reads it as the engine does, whatever the thinking switch
    /// said: the answer tag of a JSON `response_format` takes the closed thought first where the
    /// engine starts the grammar at the first token, and is the engine's own schema where it does
    /// not say or starts after its reasoning parser's gate, as on a prompt that opened the
    /// thought. A prompt that opened and closed the thought (thinking off) leaves the model in
    /// the content, and the legacy choice asks the table nothing.
    #[test]
    fn a_turn_the_prompt_leaves_unopened_is_inside_the_thought_for_the_grammar_too() {
        let model = "deepseek-ai/DeepSeek-V4-Flash";
        let unopened = concat!(
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>Be brief.",
            "<\u{ff5c}User\u{ff5c}>Fix the failing test.# Environment\n - Platform: linux"
        );
        let thinking = "<\u{ff5c}User\u{ff5c}>q<\u{ff5c}Assistant\u{ff5c}><think>";
        let closed = "<\u{ff5c}User\u{ff5c}>q<\u{ff5c}Assistant\u{ff5c}></think>";
        let open = |parser, prompt| {
            reasoning_open(parser, &names(None, None), model, &templates(), prompt)
        };
        let is_open = |parser, prompt| open(parser, prompt).map(|thought| thought.open);
        assert_eq!(is_open(ParserChoice::Symphony, unopened), Some(true));
        assert_eq!(is_open(ParserChoice::Symphony, thinking), Some(true));
        assert_eq!(is_open(ParserChoice::Symphony, closed), Some(false));
        assert_eq!(is_open(ParserChoice::Legacy, unopened), None);
        // The JSON answer's tag on the unopened prompt is the thinking prompt's: the closed
        // thought first where the engine starts at the first token, and otherwise the schema
        // alone, the engine's own constraint and no tag; with the thought closed the schema alone
        // on every start.
        let request = request(json!({
            "model": model, "response_format": {"type": "json_object"}
        }));
        let answer = |prompt, start| {
            super::constraint(
                ParserChoice::Symphony,
                &names(None, None),
                Asks::Chat(&request),
                &[],
                is_open(ParserChoice::Symphony, prompt) == Some(true),
                false,
                symphony::GrammarStart::Unknown,
                &templates(),
            )
            .and_then(|prepared| prepared.for_start(start))
        };
        let first = tag_of(answer(unopened, symphony::GrammarStart::FirstToken));
        assert_eq!(
            first,
            tag_of(answer(thinking, symphony::GrammarStart::FirstToken))
        );
        assert_eq!(first["format"]["elements"][0]["end"], "</think>", "{first}");
        for start in [
            symphony::GrammarStart::Unknown,
            symphony::GrammarStart::AfterReasoning,
        ] {
            assert!(matches!(answer(unopened, start), Ok(None)), "{start:?}");
            assert!(matches!(answer(thinking, start), Ok(None)), "{start:?}");
            assert!(matches!(answer(closed, start), Ok(None)), "{start:?}");
        }
        assert!(matches!(
            answer(closed, symphony::GrammarStart::FirstToken),
            Ok(None)
        ));
    }

    #[test]
    fn the_legacy_choice_assigns_nothing_and_asks_for_no_constraint() {
        assert!(assign(
            ParserChoice::Legacy,
            &ModelNames::default(),
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let required = request(json!({"tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Legacy,
            &ModelNames::default(),
            &required,
            &tools(),
            false
        )
        .is_none());
    }

    #[test]
    fn a_forced_call_takes_the_tables_structural_tag_and_auto_takes_none() {
        let required = request(json!({"tool_choice": "required"}));
        let Some(ToolConstraint::StructuralTag(tag)) = constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &required,
            &tools(),
            false,
        ) else {
            panic!("a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert!(tag["format"].is_object(), "{tag}");
        let auto = request(json!({"tool_choice": "auto"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &auto,
            &tools(),
            false
        )
        .is_none());
        let unknown = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &unknown,
            &tools(),
            false
        )
        .is_none());
    }

    /// A JSON `response_format` takes the table's answer tag whatever names select the table,
    /// unless the tag is the schema alone, the engine's own constraint, which the engine keeps as
    /// before. A table whose answer is written inside a message (Muse Spark's, Kimi K3's,
    /// Inkling's) takes its tag: the engine's schema writes bare JSON from the turn's own state,
    /// which such a table drops as framing, so the content came back empty; where the thought is
    /// a message of its own (Inkling's), the tag admits it first and leaves the way into the
    /// answer to the engine unless it says it starts at the first token. A table whose output
    /// starts in content takes the tag with the thought's close before the schema when the prompt
    /// leaves the thought open, and with the thought the model may write first where its table
    /// admits one (MiniMax M3's); a table with no answer state and a prompt that continues the
    /// assistant's message keep the engine's schema.
    #[test]
    fn a_json_response_format_takes_the_tables_answer_tag_unless_it_is_the_schema_alone() {
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "population": {"type": "integer"}
            },
            "required": ["city", "country", "population"],
            "additionalProperties": false
        });
        let muse = names(Some("muse_spark"), Some("muse_spark"));
        let any_object = json!({"model": "muse-spark", "response_format": {"type": "json_object"}});
        let answer = |names: &ModelNames<'_>, extra: Value, reasoning_open: bool| {
            super::constraint(
                ParserChoice::Symphony,
                names,
                Asks::Chat(&request(extra)),
                &[],
                reasoning_open,
                false,
                symphony::GrammarStart::Unknown,
                &templates(),
            )
            .and_then(|prepared| prepared.for_start(symphony::GrammarStart::Unknown))
        };
        // The named table, for a model the registry does not know: `json_object` asks for any
        // object, written in the message to the user.
        let tag = tag_of(answer(&muse, any_object.clone(), false));
        assert_eq!(tag["format"]["type"], "tag", "{tag}");
        assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
        assert_eq!(
            tag["format"]["content"],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{tag}"
        );
        // `json_schema` asks for the request's schema, strict or not.
        for strict in [true, false] {
            let tag = tag_of(answer(
                &muse,
                json!({
                    "model": "muse-spark",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "capital", "schema": schema, "strict": strict}
                    }
                }),
                false,
            ));
            assert_eq!(tag["format"]["begin"], " to=user<|message|>", "{tag}");
            assert_eq!(
                tag["format"]["content"]["json_schema"], schema,
                "strict {strict}: {tag}"
            );
        }
        // With the prompt inside the thought, the tag closes it before the message.
        let open = tag_of(answer(&muse, any_object.clone(), true));
        assert_ne!(open, tag, "{open}");
        let begin = open["format"]["begin"].as_str().unwrap_or_default();
        assert!(
            begin.ends_with(" to=user<|message|>") && begin.len() > " to=user<|message|>".len(),
            "{open}"
        );
        // The registry's tables whose answer is a message take theirs too: Kimi K3's is the
        // answer's tag alone.
        let kimi = tag_of(answer(
            &ModelNames::default(),
            json!({"model": "moonshotai/Kimi-K3", "response_format": {"type": "json_object"}}),
            false,
        ));
        assert_eq!(kimi["format"]["type"], "tag", "{kimi}");
        assert_eq!(kimi["format"]["content"]["type"], "json_schema", "{kimi}");
        // Inkling's thought is a message of its own: the tag admits it first, closed and the
        // next message opened with the text marker owed after it, or the marker alone, or
        // neither where the engine does not say it starts at the first token (an engine that
        // starts after its reasoning parser has consumed the model's own), then the schema and
        // the message's close.
        let inkling = tag_of(answer(
            &ModelNames::default(),
            json!({"model": "thinkingmachines/Inkling", "response_format": {"type": "json_object"}}),
            false,
        ));
        assert_eq!(inkling["format"]["type"], "sequence", "{inkling}");
        let elements = inkling["format"]["elements"]
            .as_array()
            .expect("a sequence");
        assert_eq!(elements.len(), 3, "{inkling}");
        assert_eq!(elements[0]["type"], "optional", "{inkling}");
        let ways = &elements[0]["content"];
        assert_eq!(ways["type"], "or", "{inkling}");
        let opener = json!({"type": "const_string", "value": "<|content_text|>"});
        assert_eq!(
            ways["elements"][0]["elements"][0]["begin"], "<|content_thinking|>",
            "{inkling}"
        );
        assert_eq!(ways["elements"][0]["elements"][1], opener, "{inkling}");
        assert_eq!(ways["elements"][1], opener, "{inkling}");
        assert_eq!(
            elements[1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{inkling}"
        );
        assert_eq!(
            elements[2],
            json!({"type": "const_string", "value": "<|end_message|>"}),
            "{inkling}"
        );
        // A table whose output starts in content (the registry's Qwen3 table) has the schema
        // alone for its tag with the thought closed: the engine's own constraint, which stands.
        let content_first = json!({"response_format": {"type": "json_object"}});
        assert!(answer(&ModelNames::default(), content_first.clone(), false)
            .expect("read")
            .is_none());
        // With the prompt inside the thought and the engine silent on where it starts the
        // grammar, the schema stands alone as well: a free-text prefix would let the answer run
        // on unconstrained, and the engine's own constraint applies where the engine starts.
        assert!(answer(&ModelNames::default(), content_first, true)
            .expect("read")
            .is_none());
        // A table that admits a thought before its answer (MiniMax M3's) keeps that tag with the
        // thought closed: the thought the model may write first, then the schema.
        let thought_first = tag_of(answer(
            &ModelNames::default(),
            json!({"model": "MiniMaxAI/MiniMax-M3", "response_format": {"type": "json_object"}}),
            false,
        ));
        assert_eq!(
            thought_first["format"]["type"], "sequence",
            "{thought_first}"
        );
        assert_eq!(
            thought_first["format"]["elements"][0]["type"], "optional",
            "{thought_first}"
        );
        assert_eq!(
            thought_first["format"]["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}}),
            "{thought_first}"
        );
        // No answer state (xLAM's list), a continued assistant message, or no JSON asked: the
        // engine's schema, or nothing.
        assert!(answer(
            &names(Some("xlam"), None),
            json!({"response_format": {"type": "json_object"}}),
            false
        )
        .expect("read")
        .is_none());
        let continued = json!({
            "model": "muse-spark",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{"}],
            "continue_final_message": true, "add_generation_prompt": false,
        });
        assert!(request(continued.clone()).continues_final_assistant());
        assert!(answer(&muse, continued, false).expect("read").is_none());
        assert!(answer(
            &muse,
            json!({"model": "muse-spark", "response_format": {"type": "text"}}),
            false
        )
        .expect("read")
        .is_none());
    }

    /// For a worker on `always`, the calls a request may make under `auto` take the table's tag
    /// with no call required. Nothing is refused: a request with no tools, `tool_choice: none`,
    /// a constraint of its own (a JSON `response_format`, a `regex`), or a prompt that continues
    /// the assistant's message takes none, and so does a forced Messages choice.
    #[test]
    fn an_always_worker_takes_the_tables_tag_for_the_calls_auto_allows() {
        let auto = request(json!({"tool_choice": "auto"}));
        let tag = auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::Chat(&auto),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates(),
        )
        .expect("no hf");
        let Some(ToolConstraint::StructuralTag(tag)) = tag else {
            panic!("a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert!(tag["format"].is_object(), "{tag}");
        let Some(ToolConstraint::StructuralTag(forced)) = constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({"tool_choice": "required"})),
            &tools(),
            false,
        ) else {
            panic!("a forced tag");
        };
        let forced: Value = serde_json::from_str(&forced).expect("JSON");
        assert_ne!(tag, forced, "the auto tag requires no call");
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::Chat(&auto),
            &[],
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates()
        )
        .expect("no hf")
        .is_none());
        assert!(auto_constraint(
            ParserChoice::Legacy,
            &ModelNames::default(),
            Asks::Chat(&auto),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates()
        )
        .expect("legacy")
        .is_none());
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::Chat(&request(json!({"tool_choice": "none"}))),
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates()
        )
        .expect("no hf")
        .is_none());
        for own in [
            json!({"tool_choice": "auto", "response_format": {"type": "json_object"}}),
            json!({"tool_choice": "auto", "regex": "[a-z]+"}),
            json!({
                "tool_choice": "auto",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "I was saying"}
                ],
                "continue_final_message": true, "add_generation_prompt": false
            }),
        ] {
            assert!(
                auto_constraint(
                    ParserChoice::Symphony,
                    &ModelNames::default(),
                    Asks::Chat(&request(own.clone())),
                    &tools(),
                    false,
                    false,
                    symphony::GrammarStart::Unknown,
                    &templates()
                )
                .expect("no hf")
                .is_none(),
                "{own}"
            );
        }
        let messages_auto = message_request(json!({
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }));
        let tools: Vec<Tool> = serde_json::from_value(json!([{
            "type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }]))
        .unwrap();
        assert!(matches!(
            auto_constraint(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Messages {
                    request: &messages_auto,
                    continues: false
                },
                &tools,
                false,
                false,
                symphony::GrammarStart::Unknown,
                &templates()
            )
            .expect("no hf"),
            Some(ToolConstraint::StructuralTag(_))
        ));
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::Messages {
                request: &messages_auto,
                continues: true
            },
            &tools,
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates()
        )
        .expect("no hf")
        .is_none());
        let messages_any = message_request(json!({
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        }));
        assert!(auto_constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::Messages {
                request: &messages_any,
                continues: false
            },
            &tools,
            false,
            false,
            symphony::GrammarStart::Unknown,
            &templates()
        )
        .expect("no hf")
        .is_none());
    }

    #[tokio::test]
    async fn the_templates_table_gives_a_forced_call_and_a_json_answer_their_tags_or_refuses() {
        let model = "Qwen/Qwen3-8B";
        let with_layout =
            template::fixtures::templates_with(model, template::fixtures::qwen3_5_tokenizer())
                .await;
        let without_layout =
            template::fixtures::templates_with(model, template::fixtures::unrendered_tokenizer())
                .await;
        let hf = names(Some("hf"), Some("hf"));
        let constrain = |templates: &Templates, extra: Value| {
            super::constraint(
                ParserChoice::Symphony,
                &hf,
                Asks::Chat(&request(extra)),
                &tools(),
                false,
                false,
                symphony::GrammarStart::Unknown,
                templates,
            )
            .and_then(|prepared| prepared.for_start(symphony::GrammarStart::Unknown))
        };
        // A forced call, one call for a named function, and a JSON answer take the table's tags.
        let required = tag_of(constrain(&with_layout, json!({"tool_choice": "required"})));
        assert!(required["format"].is_object(), "{required}");
        let named = tag_of(constrain(
            &with_layout,
            json!({"tool_choice": {"type": "function", "function": {"name": "lookup"}}}),
        ));
        assert_ne!(named, required, "one call ends where the turn does");
        let answer = tag_of(constrain(
            &with_layout,
            json!({"response_format": {"type": "json_object"}}),
        ));
        assert!(answer["format"].is_object(), "{answer}");
        // Nothing asked, nothing constrained.
        assert!(constrain(&with_layout, json!({"tool_choice": "auto"}))
            .expect("read")
            .is_none());
        // Without a layout the table has no tag for a call: the template's table refuses, where
        // a registry table would leave the older registry to decide. Its content opens with no
        // marker, so a JSON answer is written from the first byte, layout or not.
        let why =
            constrain(&without_layout, json!({"tool_choice": "required"})).expect_err("no tag");
        assert!(
            matches!(
                &why,
                SymphonyError::Constraint(text)
                    if text.contains("no structural tag for a forced tool choice")
            ),
            "{why}"
        );
        let answer = tag_of(constrain(
            &without_layout,
            json!({"response_format": {"type": "json_object"}}),
        ));
        assert_eq!(answer["format"]["type"], "json_schema", "{answer}");
        let plain = names(None, None);
        assert!(
            super::constraint(
                ParserChoice::Symphony,
                &plain,
                Asks::Chat(&request(json!({"response_format": {"type": "json_object"}}))),
                &tools(),
                false,
                false,
                symphony::GrammarStart::Unknown,
                &templates(),
            )
            .and_then(|prepared| prepared.for_start(symphony::GrammarStart::Unknown))
            .expect("read")
            .is_none(),
            "the Qwen3 table's answer tag is the schema alone, the engine's own constraint, so the \
             older registry's JSON schema stands"
        );
        // A regex or ebnf constraint would apply outside the template's format.
        let why = constrain(
            &with_layout,
            json!({"tool_choice": "required", "regex": "a+"}),
        )
        .expect_err("refused");
        assert!(
            matches!(&why, SymphonyError::Constraint(text) if text.contains("regex or ebnf")),
            "{why}"
        );
        // The tag writes a turn from its start: a continued assistant message is refused.
        let continued = json!({
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "I"}],
            "continue_final_message": true, "add_generation_prompt": false,
        });
        assert!(
            request(continued.clone()).continues_final_assistant(),
            "the request continues"
        );
        let why = constrain(&with_layout, continued).expect_err("refused");
        assert!(
            matches!(
                &why,
                SymphonyError::Constraint(text)
                    if text.contains("cannot continue the assistant's message")
            ),
            "{why}"
        );
    }

    #[tokio::test]
    async fn messages_are_read_by_the_same_tables_and_their_tool_choice_forces_calls() {
        let symphony = ParserChoice::Symphony;
        let plain = ModelNames::default();
        let read = reads(symphony, &plain, "Qwen/Qwen3-8B", &templates())
            .expect("read")
            .expect("the registry's table");
        assert_eq!(read.name(), "qwen3");
        assert!(
            reads(ParserChoice::Legacy, &plain, "Qwen/Qwen3-8B", &templates())
                .expect("read")
                .is_none()
        );
        // A tool_choice of none takes the table without its calls.
        let no_calls = super::super::assign(
            symphony,
            &plain,
            Asks::Messages {
                request: &message_request(json!({"tool_choice": {"type": "none"}})),
                continues: false,
            },
            None,
            PROMPT,
            &templates(),
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(no_calls.format.name(), "qwen3");
        assert_eq!(no_calls.format.call_syntax(), None);
        let constrain = |extra: Value| {
            super::constraint(
                symphony,
                &plain,
                Asks::Messages {
                    request: &message_request(extra),
                    continues: false,
                },
                &tools(),
                false,
                false,
                symphony::GrammarStart::Unknown,
                &templates(),
            )
            .and_then(|prepared| prepared.for_start(symphony::GrammarStart::Unknown))
        };
        let any = tag_of(constrain(json!({"tool_choice": {"type": "any"}})));
        let one = tag_of(constrain(
            json!({"tool_choice": {"type": "any", "disable_parallel_tool_use": true}}),
        ));
        let tool = tag_of(constrain(
            json!({"tool_choice": {"type": "tool", "name": "lookup"}}),
        ));
        assert_ne!(any, one, "one call ends where the turn does");
        assert_eq!(tool, one, "a named tool is one call");
        assert!(constrain(json!({"tool_choice": {"type": "auto"}}))
            .expect("read")
            .is_none());
        assert!(constrain(json!({})).expect("read").is_none());
        let assigned = super::super::assign(
            symphony,
            &plain,
            Asks::Messages {
                request: &message_request(json!({})),
                continues: false,
            },
            None,
            PROMPT,
            &templates(),
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(assigned.format.name(), "qwen3");
        assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
        // The template's table reads messages too, and refuses what it cannot tag.
        let hf = names(Some("hf"), None);
        let unrendered = template::fixtures::templates_with(
            "Qwen/Qwen3-8B",
            template::fixtures::unrendered_tokenizer(),
        )
        .await;
        let why = super::constraint(
            symphony,
            &hf,
            Asks::Messages {
                request: &message_request(json!({"tool_choice": {"type": "any"}})),
                continues: false,
            },
            &tools(),
            false,
            false,
            symphony::GrammarStart::Unknown,
            &unrendered,
        )
        .expect_err("no tag");
        assert!(
            matches!(&why, SymphonyError::Constraint(text) if text.contains("no structural tag")),
            "{why}"
        );
    }

    #[test]
    fn the_selected_workers_policy_decides_once() {
        use ConstrainedDecoding::{Always, Requested};
        use GrammarStart::{AfterReasoning, FirstToken, Unknown};

        let tag = |text: &str| ToolConstraint::StructuralTag(text.to_string());
        let refusal = |why: &str| SymphonyError::Constraint(why.to_string());
        // The request's `auto` tag: DeepSeek V4's table over one tool, the thought left open, so
        // the tag takes a shape for each start.
        let templates = Arc::new(templates());
        let auto_tag = || AutoTag {
            parser: ParserChoice::Symphony,
            labels: pair(Some("deepseek_v4"), Some("deepseek_v4")),
            flags: Names::default(),
            request: AutoRequest::Chat(Arc::new(request(json!({"tools": tools()})))),
            reasoning_open: true,
            thought_closed: false,
            templates: Arc::clone(&templates),
        };
        // What one decision gives (the tag's text, whether it is the whole turn's, whether the
        // policy chose it, and whether the `auto` tag goes with the plan), and which of the whole
        // turn's grammar and the `auto` tag it built.
        let decide = |policy,
                      start,
                      prepared: PreparedConstraint,
                      turn: Result<Option<ToolConstraint>, SymphonyError>| {
            let built = std::cell::RefCell::new(Vec::new());
            let decided = worker_constraint(
                WorkerDecoding { policy, start },
                prepared,
                || {
                    built.borrow_mut().push("turn");
                    turn
                },
                || {
                    built.borrow_mut().push("auto");
                    auto_tag()
                },
            )
            .map(|decided| {
                let text = decided.constraint.map(|constraint| constraint.to_tuple().1);
                (
                    text,
                    decided.whole_turn,
                    decided.by_policy,
                    decided.table_tag.is_some(),
                )
            });
            (decided, built.into_inner())
        };
        // What preparation found is the request's choice.
        let found = |text: &str| Ok((Some(text.to_string()), false, false, false));
        // The `auto` tag is the policy's, in the shape the worker's start takes, and goes with the
        // plan for a retry's worker.
        let auto = |start| {
            let text = auto_tag()
                .for_start(grammar_start_from(start))
                .expect("a table")
                .map(|constraint| constraint.to_tuple().1);
            assert!(text.is_some(), "{start:?}");
            Ok((text, false, true, true))
        };

        // `requested`: what preparation found, its refusal included; nothing else is built.
        for start in [FirstToken, AfterReasoning, Unknown] {
            assert_eq!(
                decide(
                    Requested,
                    start,
                    Ok(Prepared::Constraint(tag("prepared"))),
                    Ok(Some(tag("turn")))
                ),
                (found("prepared"), vec![])
            );
            assert_eq!(
                decide(
                    Requested,
                    start,
                    Err(refusal("no tag")),
                    Ok(Some(tag("turn")))
                ),
                (Err(SymphonyError::Constraint("no tag".to_string())), vec![])
            );
            assert_eq!(
                decide(Requested, start, Ok(Prepared::None), Ok(Some(tag("turn")))),
                (Ok((None, false, false, false)), vec![])
            );
        }
        // `always` from the first token: the whole turn's grammar stands over what preparation
        // found or refused, and its refusal refuses the request.
        assert_eq!(
            decide(
                Always,
                FirstToken,
                Err(refusal("no tag")),
                Ok(Some(tag("turn")))
            ),
            (
                Ok((Some("turn".to_string()), true, true, false)),
                vec!["turn"]
            )
        );
        assert_eq!(
            decide(
                Always,
                FirstToken,
                Ok(Prepared::None),
                Err(refusal("cannot hold"))
            ),
            (
                Err(SymphonyError::Constraint("cannot hold".to_string())),
                vec!["turn"]
            )
        );
        // Where the whole turn's grammar does not start, and on an engine that starts the grammar
        // later or does not say: what preparation found or refused, else the `auto` tag.
        assert_eq!(
            decide(
                Always,
                FirstToken,
                Ok(Prepared::Constraint(tag("prepared"))),
                Ok(None)
            ),
            (found("prepared"), vec!["turn"])
        );
        assert_eq!(
            decide(Always, FirstToken, Err(refusal("no tag")), Ok(None)),
            (
                Err(SymphonyError::Constraint("no tag".to_string())),
                vec!["turn"]
            )
        );
        assert_eq!(
            decide(Always, FirstToken, Ok(Prepared::None), Ok(None)),
            (auto(FirstToken), vec!["turn", "auto"])
        );
        for start in [AfterReasoning, Unknown] {
            assert_eq!(
                decide(Always, start, Err(refusal("no tag")), Ok(Some(tag("turn")))),
                (Err(SymphonyError::Constraint("no tag".to_string())), vec![])
            );
            assert_eq!(
                decide(
                    Always,
                    start,
                    Ok(Prepared::Constraint(tag("prepared"))),
                    Ok(Some(tag("turn")))
                ),
                (found("prepared"), vec![])
            );
            assert_eq!(
                decide(Always, start, Ok(Prepared::None), Ok(Some(tag("turn")))),
                (auto(start), vec!["auto"])
            );
        }
        // Where the table gives no `auto` tag (no tool to call), the policy chose nothing.
        let nothing = worker_constraint(
            WorkerDecoding {
                policy: Always,
                start: AfterReasoning,
            },
            Ok(Prepared::None),
            || Ok(None),
            || AutoTag {
                parser: ParserChoice::Symphony,
                labels: pair(Some("deepseek_v4"), Some("deepseek_v4")),
                flags: Names::default(),
                request: AutoRequest::Chat(Arc::new(request(json!({})))),
                reasoning_open: true,
                thought_closed: false,
                templates: Arc::clone(&templates),
            },
        )
        .unwrap();
        assert!(nothing.constraint.is_none() && !nothing.by_policy && nothing.table_tag.is_none());
        assert_eq!(
            decide(Requested, Unknown, Ok(Prepared::None), Ok(None)).0,
            Ok((None, false, false, false))
        );
    }

    /// A JSON answer's tag derives for the selected worker's own start, not for the pool's: from
    /// one mixed pool (a first-token worker beside one that does not say, which names either
    /// kind), a worker that starts after its reasoning parser or does not say gets the schema
    /// alone, the engine's own constraint, and the first-token worker the closed thought then the
    /// schema. The derivation goes with the plan, and the worker a retry takes gets its own shape
    /// from it, the request's own schema standing in where the table gives it no tag. A forced
    /// call's tag keeps the pool's start, and its prefix for a pool of either kind.
    #[test]
    fn a_json_answers_tag_derives_for_the_selected_worker_and_the_worker_a_retry_takes() {
        use GrammarStart::{AfterReasoning, FirstToken, Unknown};
        let worker = |start: Option<&str>| -> Arc<dyn Worker> {
            let mut builder =
                BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
            if let Some(start) = start {
                builder = builder.label(WorkerGrammarStart::LABEL, start);
            }
            Arc::new(builder.build())
        };
        let pool = [worker(Some("first_token")), worker(None)];
        assert_eq!(grammar_start_of(&pool), symphony::GrammarStart::Either);
        let deepseek = names(Some("deepseek_v4"), Some("deepseek_v4"));
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let asked = request(json!({"response_format": {
            "type": "json_schema", "json_schema": {"name": "city", "schema": schema},
        }}));
        let prepared = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(&asked),
            &[],
            true,
            false,
            grammar_start_of(&pool),
            &templates(),
        );
        let Ok(Prepared::Answer(answer)) = prepared else {
            panic!("the answer's tag, derived per worker, not {prepared:?}");
        };
        // Decided on the selected worker: each kind its own shape.
        let decided = |start| {
            worker_constraint(
                WorkerDecoding {
                    policy: ConstrainedDecoding::Requested,
                    start,
                },
                Ok(Prepared::Answer(Arc::clone(&answer))),
                || panic!("requested: the whole turn is not asked"),
                || panic!("requested: the auto tag is not asked"),
            )
            .expect("decided")
        };
        for start in [AfterReasoning, Unknown] {
            let decided = decided(start);
            assert!(
                decided.constraint.is_none(),
                "{start:?}: the schema alone, the engine's own constraint, not {:?}",
                decided.constraint
            );
            assert!(!decided.by_policy && !decided.whole_turn, "{start:?}");
            let stamp = decided
                .table_tag
                .expect("the derivation goes with the plan");
            assert_eq!(stamp.tag_for(start).expect("a table"), None, "{start:?}");
            assert_eq!(stamp.schema(), Some(&schema), "{start:?}");
        }
        let first = decided(FirstToken);
        let Some(ToolConstraint::StructuralTag(text)) = first.constraint else {
            panic!("the closed thought then the schema: {:?}", first.constraint);
        };
        let tag: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(tag["format"]["elements"][0]["end"], "</think>", "{tag}");
        assert_eq!(tag["format"]["elements"][1]["type"], "json_schema", "{tag}");
        assert_eq!(tag["format"]["elements"][1]["json_schema"], schema, "{tag}");
        assert!(!first.by_policy && !first.whole_turn);
        // The worker a retry takes, of the other kind, and back.
        let stamp = first.table_tag.expect("the derivation goes with the plan");
        assert_eq!(stamp.tag_for(AfterReasoning).expect("a table"), None);
        assert_eq!(stamp.tag_for(Unknown).expect("a table"), None);
        assert_eq!(
            stamp.tag_for(FirstToken).expect("a table").as_deref(),
            Some(text.as_str())
        );
        assert_eq!(stamp.schema(), Some(&schema));
        // The forced call's tag keeps the pool's start: the prefix that fits either kind.
        let forced = request(json!({"tool_choice": "required"}));
        let Ok(Prepared::Constraint(ToolConstraint::StructuralTag(call))) = super::constraint(
            ParserChoice::Symphony,
            &deepseek,
            Asks::Chat(&forced),
            &tools(),
            true,
            false,
            grammar_start_of(&pool),
            &templates(),
        ) else {
            panic!("a forced tag");
        };
        let call: Value = serde_json::from_str(&call).expect("JSON");
        assert_eq!(call["format"]["elements"][0]["type"], "any_text", "{call}");
    }

    /// The request's `auto` tag derives again for where another worker's engine starts the
    /// grammar, as a retry needs it: the free-text prefix before the calls for an engine that does
    /// not say, the bare block for one that starts after its reasoning parser's gate, the closed
    /// thought then the block for one that starts at the first token; each the tag request
    /// building gives a worker of that start. For a chat and a Messages request alike.
    #[test]
    fn the_auto_tag_derives_again_for_the_start_of_the_worker_a_retry_takes() {
        use WorkerGrammarStart::{AfterReasoning, FirstToken, Unknown};
        let templates = Arc::new(templates());
        let names = names(Some("deepseek_v4"), Some("deepseek_v4"));
        let chat = request(json!({"tools": tools()}));
        let messages = message_request(json!({
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }));
        let tags = [
            AutoTag {
                parser: ParserChoice::Symphony,
                labels: names.labels.clone(),
                flags: Names::default(),
                request: AutoRequest::Chat(Arc::new(chat.clone())),
                reasoning_open: true,
                thought_closed: false,
                templates: Arc::clone(&templates),
            },
            AutoTag {
                parser: ParserChoice::Symphony,
                labels: names.labels.clone(),
                flags: Names::default(),
                request: AutoRequest::Messages {
                    request: Arc::new(messages.clone()),
                    tools: tools().into(),
                    continues: false,
                },
                reasoning_open: true,
                thought_closed: false,
                templates: Arc::clone(&templates),
            },
        ];
        for auto_tag in &tags {
            let built = |start| {
                let constraint = match &auto_tag.request {
                    AutoRequest::Chat(_) => auto_constraint(
                        ParserChoice::Symphony,
                        &names,
                        Asks::Chat(&chat),
                        &chat.callable_tools(),
                        true,
                        false,
                        start,
                        &templates,
                    ),
                    AutoRequest::Messages { .. } => auto_constraint(
                        ParserChoice::Symphony,
                        &names,
                        Asks::Messages {
                            request: &messages,
                            continues: false,
                        },
                        &tools(),
                        true,
                        false,
                        start,
                        &templates,
                    ),
                };
                tag_of(constraint)
            };
            let derived = |start| {
                let tag = TagForStart::tag_for(auto_tag, start)
                    .expect("a table")
                    .expect("a tag");
                serde_json::from_str::<Value>(&tag).expect("a tag as JSON")
            };
            let unknown = derived(Unknown);
            assert_eq!(
                unknown,
                built(symphony::GrammarStart::Unknown),
                "{auto_tag:?}"
            );
            assert_eq!(unknown["format"]["elements"][0]["type"], "any_text");
            let bare = derived(AfterReasoning);
            assert_eq!(bare, built(symphony::GrammarStart::AfterReasoning));
            assert_ne!(bare["format"]["elements"][0]["type"], "any_text", "{bare}");
            let first = derived(FirstToken);
            assert_eq!(first, built(symphony::GrammarStart::FirstToken));
            assert_eq!(first["format"]["elements"][0]["end"], "</think>", "{first}");
            assert_eq!(first["format"]["elements"][1], bare["format"]);
        }
        // A tag the table gives no engine (no tool to call) derives as none for every start.
        let none = AutoTag {
            parser: ParserChoice::Symphony,
            labels: names.labels,
            flags: Names::default(),
            request: AutoRequest::Chat(Arc::new(request(json!({})))),
            reasoning_open: true,
            thought_closed: false,
            templates,
        };
        for start in [Unknown, AfterReasoning, FirstToken] {
            assert_eq!(TagForStart::tag_for(&none, start), Ok(None));
        }
        assert!(format!("{none:?}").contains("Qwen/Qwen3-8B"));
    }

    #[test]
    fn the_whole_turn_holds_the_answer_and_the_calls() {
        let free = whole(json!({}), None);
        assert_eq!(
            after_thought(&free)["elements"][1],
            json!({"type": "any_text", "excludes": ["<|done|>", "<|stop|>"]})
        );
        assert_eq!(
            after_thought(&whole(json!({"tools": weather()}), None))["type"],
            "or"
        );
        let required = whole(json!({"tools": weather(), "tool_choice": "required"}), None);
        assert_eq!(after_thought(&required)["type"], "sequence");
        let json_answer = whole(json!({"response_format": {"type": "json_object"}}), None);
        assert_eq!(
            after_thought(&json_answer)["elements"][1],
            json!({"type": "json_schema", "json_schema": {"type": "object"}})
        );
        let regex = whole(json!({"regex": "(yes|no)"}), None);
        assert_eq!(
            after_thought(&regex)["elements"][1],
            json!({"type": "regex", "pattern": "(yes|no)"})
        );
        let ebnf = whole(json!({"ebnf": "root ::= \"y\""}), None);
        assert_eq!(after_thought(&ebnf)["elements"][1]["type"], "grammar");
        // A forced call writes no answer, so it leaves the answer's format out, Lark's included.
        assert_eq!(
            turn(
                json!({"tools": weather(), "tool_choice": "required", "ebnf": "start: \"y\""}),
                None
            ),
            Ok(Some(required))
        );
    }

    #[test]
    fn the_prompt_says_where_the_turn_starts() {
        let thinking = whole(json!({}), Some("<|via|>notes<|body|>Let me"));
        assert_eq!(thinking["elements"][0]["begin"], "");
        let after = whole(
            json!({}),
            Some("<|via|>notes<|body|>Plan.<|stop|><|head|>assistant"),
        );
        assert_eq!(after["elements"][0]["value"], "<|via|>reply<|body|>");

        // Inside an answer: the rest of a free one; a forced call or an answer's format starts
        // no grammar there, and the table's tags decide (`worker_constraint`).
        let answering = Some("<|via|>reply<|body|>The answer");
        assert_eq!(
            whole(json!({}), answering)["elements"][0]["type"],
            "any_text"
        );
        for extra in [
            json!({"tools": weather(), "tool_choice": "required"}),
            json!({"tools": weather(), "tool_choice": {"type": "allowed_tools",
                "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}}),
            json!({"response_format": {"type": "json_object"}}),
        ] {
            assert_eq!(turn(extra, answering), Ok(None));
        }
        // Past text no field holds, no grammar starts.
        assert_eq!(turn(json!({"tools": weather()}), Some("Hello")), Ok(None));
        // Inside a call: the rest of it is free.
        let calling = " to=get_weather<|via|>notes <|as|>xml<|body|><arg name=\"city\">\nPa";
        assert_eq!(
            whole(json!({"tools": weather()}), Some(calling))["elements"][0]["type"],
            "any_text"
        );
    }

    #[test]
    fn the_whole_turn_refuses_what_it_cannot_hold() {
        let refused = |extra: Value| turn(extra, None).is_err();
        let served = |extra: Value| turn(extra, None).expect("held").is_some();
        let named = |name: &str| {
            json!([{"type": "function", "function": {"name": name, "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}}}}])
        };
        // A name a call cannot carry, whatever the choice.
        for choice in [json!("auto"), json!("none"), json!("required")] {
            assert!(refused(
                json!({"tools": named("get weather"), "tool_choice": choice})
            ));
        }
        let key = json!([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"say \"hi\"": {"type": "string"}}}}}]);
        assert!(refused(json!({"tools": key})));

        // A strict tool whose calls the grammar would hold more loosely, while it is callable.
        let tuple = |strict: bool| {
            json!([{"type": "function", "function": {"name": "f", "strict": strict,
                "parameters": {"type": "object", "properties": {"p": {"type": "tuple"}},
                    "required": ["p"]}}}])
        };
        assert!(refused(json!({"tools": tuple(true)})));
        assert!(served(json!({"tools": tuple(false)})));
        assert!(served(json!({"tools": tuple(true), "tool_choice": "none"})));

        // Under `auto`, a strict tool or one call at most beside the request's own grammar; and
        // an ebnf in Lark's syntax.
        let strict = json!([{"type": "function", "function": {"name": "f", "strict": true,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        for extra in [
            json!({"tools": strict, "regex": "(a|b)"}),
            json!({"tools": strict, "ebnf": "root ::= \"y\""}),
            json!({"tools": weather(), "parallel_tool_calls": false, "regex": "(a|b)"}),
            json!({"ebnf": "start: \"y\""}),
            json!({"tools": weather(), "ebnf": "start: \"y\""}),
        ] {
            assert!(refused(extra.clone()), "{extra}");
        }
        for extra in [
            json!({"tools": strict}),
            json!({"tools": strict, "response_format": {"type": "json_object"}}),
            json!({"tools": strict, "tool_choice": "required", "regex": "(a|b)"}),
            json!({"tools": weather(), "regex": "(a|b)"}),
            json!({"tools": weather(), "parallel_tool_calls": false}),
            json!({"regex": "(a|b)", "parallel_tool_calls": false}),
        ] {
            assert!(served(extra.clone()), "{extra}");
        }
        let why = turn(json!({"ebnf": "start: \"y\""}), None).expect_err("refused");
        assert!(why.contains("Lark"), "{why}");
    }

    #[test]
    fn over_its_limits_the_whole_turn_leaves_the_tags_to_decide() {
        let properties: serde_json::Map<String, Value> = (0
            ..=symphony::OutputSpelling::MAX_TOOL_PARAMETERS)
            .map(|i| (format!("p{i}"), json!({"type": "string"})))
            .collect();
        let wide = json!([{"type": "function", "function": {"name": "wide",
            "parameters": {"type": "object", "properties": properties}}}]);
        assert_eq!(turn(json!({"tools": wide}), None), Ok(None));

        let huge = "x".repeat(symphony::OutputSpelling::MAX_TAG_BYTES);
        let large = json!([{"type": "function", "function": {"name": "large",
        "parameters": {"type": "object", "properties": {
            "p": {"type": "string", "enum": [huge]}
        }}}}]);
        assert_eq!(turn(json!({"tools": large}), None), Ok(None));
    }

    #[test]
    fn a_messages_choice_takes_the_turn_its_chat_twin_takes() {
        let tools = json!([{"name": "get_weather", "input_schema": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
        let prompt = "user<|head|>assistant";
        for (choice, chat) in [
            (json!({"type": "auto"}), json!({})),
            (json!({"type": "any"}), json!({"tool_choice": "required"})),
            (
                json!({"type": "any", "disable_parallel_tool_use": true}),
                json!({"tool_choice": "required", "parallel_tool_calls": false}),
            ),
            (
                json!({"type": "tool", "name": "get_weather"}),
                json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
            ),
            (json!({"type": "none"}), json!({"tool_choice": "none"})),
        ] {
            let messages = message_request(json!({"tools": tools, "tool_choice": choice}));
            let mut chat = chat;
            chat["tools"] = weather();
            let tag = |constraint: Result<Option<ToolConstraint>, SymphonyError>| {
                tag_of(constraint)["format"].clone()
            };
            let chat = request(chat);
            assert_eq!(
                tag(turn_of(
                    &turn_table(),
                    Asks::Messages {
                        request: &messages,
                        continues: false
                    },
                    &messages.chat_tools(),
                    prompt,
                )),
                tag(turn_of(
                    &turn_table(),
                    Asks::Chat(&chat),
                    &chat.effective_tool_list(),
                    prompt,
                )),
                "{choice}"
            );
        }
    }

    #[test]
    fn only_the_response_templates_table_spells_a_whole_turn() {
        // Any other name, and the legacy parsers, leave the turn to the table's tags without
        // looking for a template.
        let request = request(json!({"tools": weather(), "tool_choice": "required"}));
        for (parser, names) in [
            (ParserChoice::Symphony, names(Some("qwen"), None)),
            (ParserChoice::Symphony, names(None, None)),
            (ParserChoice::Legacy, names(Some("hf"), None)),
        ] {
            assert!(matches!(
                turn_constraint(
                    parser,
                    &names,
                    Asks::Chat(&request),
                    &request.effective_tool_list(),
                    PROMPT,
                    &templates()
                ),
                Ok(None)
            ));
        }
        // The template's names ask for the model's own template, and a model without one is
        // refused, as the table's tags refuse it.
        let why = turn_constraint(
            ParserChoice::Symphony,
            &names(Some("hf"), None),
            Asks::Chat(&request),
            &request.effective_tool_list(),
            PROMPT,
            &templates(),
        )
        .expect_err("refused");
        assert_eq!(
            why,
            SymphonyError::Template(
                "no tokenizer is registered for model 'Qwen/Qwen3-8B'".to_string()
            )
        );
    }
}

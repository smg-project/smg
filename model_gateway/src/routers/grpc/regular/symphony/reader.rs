//! The reader: Symphony's assignment to a response and the reader the drivers run over the
//! stop decoder's output, one per choice.

use std::sync::{atomic::AtomicBool, Arc, LazyLock};

use llm_tokenizer::stop::StopSequenceDecoder;
use openai_protocol::{
    common::{Tool, ToolChoice},
    profile::ProviderProfile,
};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use symphony::{
    grammar::GrammarStart, Declared, DropReason, Engine, EngineFinish, Event, Events, ForcedCall,
    Format, Input, Parser as _, Text, TokenSpan,
};
use tracing::{debug, Span};

use super::{
    constraint::{Asks, Forbids},
    names::Reading,
    telemetry::{self, Tally, Who},
    SymphonyError,
};
#[cfg(test)]
use super::{names::reading, template::Templates};
use crate::routers::grpc::utils::{gives_way, isolation, CallIdFamily};
#[cfg(test)]
use crate::{config::ParserChoice, routers::grpc::utils::ModelNames};

/// Symphony's assignment to a response: the table that reads it, as the request's asks make it
/// ([`assign_for`]), and the part of the rendered prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    /// The table, shared with every reader of the response and every request it reads for.
    pub format: Arc<Format>,
    pub prompt: String,
    /// The calls the request's `tool_choice` forbids among those the table reads
    /// ([`Asks::forbids_for`]): every call under `none`, a call outside the callable tools under
    /// a named function, an allowed list or `required`. The reader drops such a call, as the
    /// engine's own server reads a call under `none` and withholds it.
    pub forbids_calls: Forbids,
    /// The request asked for generation past the model's own end of output (`ignore_eos`,
    /// [`Asks::past_the_end`]): the output carries an unconstrained tail after the answer, which
    /// the reader reads and delivers like the rest, and a marker of the table kept as text in it
    /// is no failure of the table's reading ([`Tally::new`]).
    pub past_the_end: bool,
    /// The function the request forces (`tool_choice` naming it) where the output's JSON is its
    /// arguments alone under the gateway's JSON-schema constraint: the table reads them as the
    /// forced call ([`forced`]), and the reader names that one call so ([`assign_for`]).
    pub forcing: Option<String>,
    /// The first byte of the forced JSON (`{` for a named function's arguments, `[` for a
    /// required call's list) where the output's own first bytes decide the reader's start
    /// ([`ForcedStart::ByTheFirstBytes`]): the reader stands after the prompt's replay unless
    /// the output opens as that JSON does, which makes it the call from there, with nothing of
    /// the prompt replayed. `None` where the start is known when the assignment is made.
    pub opener: Option<u8>,
    /// The model's family and the count of calls in the conversation's history, where the
    /// gateway's JSON-schema constraint forces the call and the family numbers its ids by the
    /// conversation ([`CallIdFamily`]). The forced JSON has no header for the table to take an id
    /// from, so the reader gives each of its calls the family's id, as the older forced arm does.
    /// So it does for every call of a table whose own name gives way ([`gives_way`]: today
    /// `kimi`): it reads in the place of the legacy parsers, which numbered the family's calls.
    /// `None` for every other request: its ids are the table's.
    pub forced_ids: Option<(CallIdFamily, usize)>,
    /// The request holds the answer to JSON (a `json_object` or `json_schema` response format, a
    /// Messages request's output format, [`Asks::json_answer`]): the reader's engine reads it
    /// under its JSON-answer option ([`Engine::reads_json_answer`]), so a marker the model spells
    /// inside one of the answer's strings is the answer's text, not a frame of the table.
    pub json_answer: bool,
    /// The request's span, current where the assignment is made ([`assignment`]), for the record
    /// a reading that was not clean leaves on it ([`Reader::end`], [`Reader::failed`]). A streamed
    /// response's reading runs outside the span (the response body drives it), so the span
    /// travels with the assignment rather than being looked up where the output ends.
    pub span: Span,
    /// Set by the first reading of the request that marks the span ([`Tally::mark`],
    /// [`telemetry::mark_failed`]): one reading stands on the span, the first that was not clean,
    /// and a later choice's reading or a later failure of the same request leaves it.
    pub marked: Arc<AtomicBool>,
}

impl Assignment {
    /// The `table` label of the format, for the counters and the lines.
    pub(crate) fn table(&self) -> &'static str {
        telemetry::table_label(self.format.name())
    }
}

/// The assignment of `format` to a response, with the part of `prompt` the engine replays,
/// under the span current here: the request's.
pub(super) fn assignment(
    format: Arc<Format>,
    model: &str,
    prompt: &str,
    forbids_calls: Forbids,
    past_the_end: bool,
    forcing: Option<String>,
    forced_ids: Option<(CallIdFamily, usize)>,
) -> Assignment {
    debug!(model, table = format.name(), "Symphony reads the response");
    let prompt = format.replay(prompt).to_string();
    Assignment {
        format,
        prompt,
        forbids_calls,
        past_the_end,
        forcing,
        opener: None,
        forced_ids,
        json_answer: false,
        span: Span::current(),
        marked: Arc::default(),
    }
}

/// The table as the request's asks make it: without its calls when the legacy path would run no
/// tool parser (the response is not scanned for calls:
/// [`scans_for_tool_calls_for`](crate::routers::grpc::spec::scans_for_tool_calls_for)), so a table that reads calls from the output's first byte (`json`,
/// `xlam`) is content from it; without its thought when the request keeps the reasoning in the
/// content, as the legacy path runs no reasoning parser then. A variant is a copy of the shared
/// table; a request that asks for neither reads with the shared table itself. A `tool_choice` of
/// `none` asks for no variant: the table keeps its calls and the reader drops them
/// ([`Assignment::forbids_calls`]).
pub(super) fn asked(format: Arc<Format>, no_calls: bool, no_reasoning: bool) -> Arc<Format> {
    if !no_calls && !no_reasoning {
        return format;
    }
    variant(
        format,
        Ask {
            calls: if no_calls { Calls::None } else { Calls::Read },
            no_reasoning,
        },
    )
}

/// The table as a request reads it whose call the gateway's JSON-schema constraint forces: the
/// output is the model's thought, where the engine lets it think first, and then the JSON alone,
/// so the table keeps its thought and reads everything else as the call, written as `written`
/// says ([`Format::with_forced_call`]); without its thought when the request keeps the reasoning
/// in the content, which then holds the thought's text ahead of the call. Kept per table as
/// [`asked`]'s variants are.
pub(super) fn forced(format: Arc<Format>, written: ForcedCall, no_reasoning: bool) -> Arc<Format> {
    variant(
        format,
        Ask {
            calls: Calls::Forced(written),
            no_reasoning,
        },
    )
}

/// What a request asks of the table's calls: read as the table reads them, none
/// ([`Format::without_calls`]), or the one the JSON-schema constraint forces
/// ([`Format::with_forced_call`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Calls {
    Read,
    None,
    Forced(ForcedCall),
}

/// What a request asks of the table that reads it: the key of a kept variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Ask {
    calls: Calls,
    no_reasoning: bool,
}

/// The variant of `format` that `ask` makes, kept per shared table.
fn variant(format: Arc<Format>, ask: Ask) -> Arc<Format> {
    // A table nobody else holds (one built for this request) is made over in
    // place, as before; a shared table's variant is kept, so the next request
    // that asks the same of it reads with the one copy.
    if Arc::strong_count(&format) == 1 {
        return Arc::new(variant_of(Arc::unwrap_or_clone(format), ask));
    }
    let key = (Arc::as_ptr(&format) as usize, ask);
    if let Some(kept) = ASKED_VARIANTS.read().get(&key) {
        return Arc::clone(&kept.variant);
    }
    let variant = Arc::new(variant_of((*format).clone(), ask));
    let mut kept = ASKED_VARIANTS.write();
    // Another request may have kept the same variant between the read and
    // the write: theirs is the one every reader shares from here on.
    if let Some(kept) = kept.get(&key) {
        return Arc::clone(&kept.variant);
    }
    if kept.len() >= ASKED_VARIANTS_KEPT {
        kept.clear();
    }
    kept.insert(
        key,
        AskedVariant {
            table: format,
            variant: Arc::clone(&variant),
        },
    );
    variant
}

/// `format` as `ask` makes it: without its calls or with the forced one, then without its
/// thought.
fn variant_of(mut format: Format, ask: Ask) -> Format {
    format = match ask.calls {
        Calls::Read => format,
        Calls::None => format.without_calls(),
        Calls::Forced(written) => format.with_forced_call(written),
    };
    if ask.no_reasoning {
        format = format.without_reasoning();
    }
    format
}

/// The variants [`asked`] and [`forced`] made of the shared tables, by the
/// table's address and the ask. A chat request without tools asks for the
/// table without its calls, which was rebuilt from the shared table on every
/// such request (a clone and the walks of `without_calls`: 19 allocation calls
/// per request, 2026-10-21).
static ASKED_VARIANTS: LazyLock<RwLock<AskedVariants>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));

/// The kept variants by the shared table's address and the ask (the calls, no
/// reasoning); the key is the process's own, hashed with the Fx hasher as the
/// gateway's other address-keyed maps are.
type AskedVariants = FxHashMap<(usize, Ask), AskedVariant>;

/// How many variants are kept before the map starts over: the registry's
/// groups, the named tables and the templates' tables come to a few hundred,
/// each with up to three asks and, for a call the JSON schema forces, four
/// more; a gateway serving the tables of the models it runs with every ask
/// stays under this, and a stream of tables built per worker cannot grow the
/// map without bound.
const ASKED_VARIANTS_KEPT: usize = 1024;

/// One kept variant: the shared table it was made of (held, so the address
/// the entry is keyed by stays its own for as long as the entry lives) and
/// the variant.
struct AskedVariant {
    #[expect(
        dead_code,
        reason = "held so the address the entry is keyed by cannot be reused"
    )]
    table: Arc<Format>,
    variant: Arc<Format>,
}

/// [`assign_for`] with the table decision read here ([`reading`]), the provider profile derived
/// from the model and workers that do not say where they start a grammar, as the tests ask it.
#[cfg(test)]
pub(crate) fn assign(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, SymphonyError> {
    Ok(assign_for(
        reading(parser, names, asks.model(), templates)?,
        asks,
        constraint,
        prompt,
        ProviderProfile::for_model(asks.model()),
        || GrammarStart::Unknown,
    ))
}

/// Symphony's assignment to a request, if a table reads the response (`read`, the table decision:
/// preparation reads it once per request, [`reading`](super::names::reading), and request building
/// passes it, `ProcessingState::reading`). The table goes without its calls where the legacy path
/// would run no tool parser, without its thought where it would run no reasoning parser
/// ([`Asks::no_calls_for`], [`Asks::no_reasoning`], [`asked`]). A call the conversation
/// established without a tool inventory comes back as a call under either parser.
///
/// `constraint` is one no table built (`WorkerConstraint::table`): the older registry's where
/// the table has no tag, or the forced call's JSON schema where the table's tag opens with free
/// text; the table's own is never passed, whatever kind its backend's spelling gave it.
///
/// Under a JSON-schema constraint (the forced call the older registry writes where the table has
/// no tag for it) the output has none of the table's call markers: it is the model's thought,
/// where the engine holds the output to the schema only once the thought is over, and then the
/// JSON alone. The table reads it with its forced call ([`forced`]): the thought as the table
/// reads a thought, then the arguments of the function the request names
/// ([`Assignment::forcing`]) or the list of calls a required call writes. A schema with no call
/// forced is an answer, which the table reads without its calls. The older reasoning parser is
/// not asked. The forced call's id is the family's where the model's family numbers its ids
/// ([`Assignment::forced_ids`]), the table's otherwise.
///
/// The reader of a forced call starts where the grammar starts ([`ForcedStart`]): the schema
/// holds the output from the first token on workers that start a grammar there, so the JSON
/// comes alone whatever the prompt opened, and nothing of the prompt is replayed; on workers
/// that start it once the thought is over the prompt's replay stands; where the workers do not
/// say, the output's first bytes decide ([`Assignment::opener`]). One forced call has no opener
/// to name there: a named function's arguments the schema does not hold to an object
/// ([`holds_to_an_object`]), with the reasoning kept in the content. That JSON may open with any
/// byte, and read after the replay it would be content and no call, so it starts in the call;
/// a thought written first is the call's text there. With the reasoning separated the opener
/// is named whatever the schema: an output that opens otherwise is read where the replay
/// leaves the reader, and JSON alone inside a thought the prompt opened is that thought.
/// `pool_start` says which start, for the model's workers together
/// ([`grammar_start_of`](super::constraint::grammar_start_of)); it is asked for a forced call
/// alone, so no other request pays for the answer.
///
/// `None` where the legacy parsers read the response, which the response's header then says; a
/// request Symphony was configured for is counted once here, by the reason
/// ([`telemetry::record_fallback`]): request building asks once per request, and nothing else
/// decides the reader. `provider` is the model's provider profile as the preparation stage derived
/// it once per request; request building passes it. The tests ask through `assign`, which reads
/// the decision and derives the profile first.
pub(crate) fn assign_for(
    read: Reading,
    asks: Asks<'_>,
    constraint: Option<&(String, String)>,
    prompt: &str,
    provider: ProviderProfile,
    pool_start: impl FnOnce() -> GrammarStart,
) -> Option<Assignment> {
    let format = match read {
        Reading::Table(format, _) => format,
        Reading::Legacy => return None,
        Reading::Fallback(why) => {
            telemetry::record_fallback(why);
            return None;
        }
    };
    let schema_alone = constraint.is_some_and(|(kind, _)| kind == "json_schema");
    let (format, forcing, is_forced) = if schema_alone {
        match asks.choice().as_deref() {
            Some(ToolChoice::Function { function, .. }) => (
                forced(format, ForcedCall::Arguments, asks.no_reasoning()),
                Some(function.name.clone()),
                true,
            ),
            Some(choice) if choice.forces_call() => (
                forced(format, ForcedCall::List, asks.no_reasoning()),
                None,
                true,
            ),
            _ => (asked(format, true, asks.no_reasoning()), None, false),
        }
    } else {
        (
            asked(format, asks.no_calls_for(provider), asks.no_reasoning()),
            None,
            false,
        )
    };
    // The forced JSON has no header to read an id from, and a table whose own name gives way
    // (`GIVING_WAY`: today `kimi`) leaves its ids to the gateway: a family that numbers its ids
    // by the conversation numbers these too, as the older parsers do. A table read through
    // another such name (`command`, through `cohere_cmd`) mints its own.
    let forced_ids = (is_forced || gives_way(format.name()))
        .then(|| CallIdFamily::of(asks.model()))
        .flatten()
        .map(|family| (family, asks.history_tool_calls_count()));
    // Where the reader of the forced call starts: in the call, after the prompt's replay, or
    // where the output's first bytes say, the JSON's own opener standing for the call.
    let (prompt, opener) = match is_forced.then(|| ForcedStart::of(pool_start())) {
        Some(ForcedStart::InTheCall) => ("", None),
        Some(ForcedStart::ByTheFirstBytes) => match &forcing {
            // The required call's list opens with its bracket: the schema's root is the
            // gateway's own.
            None => (prompt, Some(b'[')),
            // Kept in the content, arguments that may open with any byte: no byte tells that
            // JSON from a thought, and read as a thought it is content and no call.
            Some(_)
                if asks.no_reasoning()
                    && !constraint.is_some_and(|(_, schema)| holds_to_an_object(schema)) =>
            {
                ("", None)
            }
            Some(_) => (prompt, Some(b'{')),
        },
        Some(ForcedStart::AfterTheReplay) | None => (prompt, None),
    };
    Some(Assignment {
        opener,
        json_answer: asks.json_answer(),
        ..assignment(
            format,
            asks.model(),
            prompt,
            asks.forbids_for(provider),
            asks.past_the_end(),
            forcing,
            forced_ids,
        )
    })
}

/// Where the reader of a forced call under the JSON schema starts, by where the model's workers
/// start a grammar together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForcedStart {
    /// At the first token the engine holds the output to the schema before the model writes
    /// anything, so the JSON comes alone whatever the prompt opened: the reader starts in the
    /// call, with nothing of the prompt replayed.
    InTheCall,
    /// After the thought, the thought and its close come first, and the prompt's replay reads
    /// them: reasoning, or content where the request keeps it there.
    AfterTheReplay,
    /// Workers that do not say, or that start at both places, leave the shape to the output: the
    /// JSON alone, or the thought, its close and the JSON. The reader stands after the prompt's
    /// replay, and an output that opens as the JSON does (its opener, then a byte the JSON can
    /// write after it, whitespace aside) is the call from the opener, with nothing of the prompt
    /// replayed ([`Reader::decide`]). What this start reads wrong, which no reader can tell
    /// apart while the workers do not say where they start a grammar: a thought that itself
    /// opens as the JSON does, and a thought behind a run of whitespace that fills the output's
    /// deciding bytes ([`DECIDING_BYTES`], the opener counted where one came), are read as the
    /// call; where the reasoning is separated, a named call's JSON that opens with another byte
    /// than the brace is read as the thought ([`assign_for`]).
    ByTheFirstBytes,
}

impl ForcedStart {
    /// The start for workers that start a grammar at `start` together.
    fn of(start: GrammarStart) -> Self {
        match start {
            GrammarStart::FirstToken => Self::InTheCall,
            GrammarStart::AfterReasoning => Self::AfterTheReplay,
            GrammarStart::Unknown | GrammarStart::Either => Self::ByTheFirstBytes,
        }
    }
}

/// Whether `schema`, the text of the JSON schema the engine holds the output to, says at its
/// root that the output is an object (`"type": "object"`), so that its first byte is a brace. A
/// root that names no type, or several, does not say so: the empty schema of a function declared
/// without parameters, a `$ref` or a union at the root, whatever they resolve to. The root's
/// `type` alone is kept, borrowed from the text: the text is read once, its other members
/// skipped, and no tree is built (the skip keeps its stack of open brackets, the one
/// allocation).
fn holds_to_an_object(schema: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Root<'a> {
        #[serde(borrow, rename = "type")]
        kind: Option<&'a str>,
    }
    serde_json::from_str::<Root<'_>>(schema).is_ok_and(|root| root.kind == Some("object"))
}

/// Whether `byte` can follow `opener` in the JSON it opens, whitespace aside: a member's key or
/// the closing brace after an object's brace, a call's brace or the closing bracket after a
/// list's bracket.
fn follows_the_opener(opener: u8, byte: u8) -> bool {
    matches!((opener, byte), (b'{', b'"' | b'}') | (b'[', b'{' | b']'))
}

/// The engine's finish reason as Symphony takes it.
pub(crate) fn engine_finish(reason: &str) -> EngineFinish {
    match reason {
        "stop" => EngineFinish::Stop,
        "length" => EngineFinish::Length,
        "abort" => EngineFinish::Abort,
        other => EngineFinish::Other(other.to_string()),
    }
}

/// What the `Finish` event among `events` says.
pub(crate) struct Finished {
    /// The calls made in the output.
    pub tool_calls: u32,
    /// The engine tokens spent on reasoning, markers excluded.
    pub reasoning_tokens: u32,
}

/// The `Finish` event among `events`, if the output ended.
pub(crate) fn finished(events: &[Event]) -> Option<Finished> {
    events.iter().find_map(|event| match event {
        Event::Finish {
            tool_calls,
            reasoning_tokens,
            ..
        } => Some(Finished {
            tool_calls: *tool_calls,
            reasoning_tokens: *reasoning_tokens,
        }),
        _ => None,
    })
}

/// The delta the reader feeds the engine for one engine chunk: the text the stop decoder released,
/// the span of every token in it and the tokens read. The step's events borrow its text, so it
/// lives with the caller, next to the events, and a stream fills the same one every step.
#[derive(Debug, Default)]
pub(crate) struct Delta {
    text: String,
    spans: Vec<TokenSpan>,
    tokens: Vec<u32>,
}

impl Delta {
    fn clear(&mut self) {
        self.text.clear();
        self.spans.clear();
        self.tokens.clear();
    }

    fn input(&self) -> Input<'_> {
        Input::Delta {
            token_ids: &self.tokens,
            text: &self.text,
            spans: &self.spans,
        }
    }

    /// Adds `other`'s text and tokens after this one's, each span at its place in the joined
    /// text.
    fn append(&mut self, other: &Self) {
        let at = self.text.len();
        self.text.push_str(&other.text);
        self.spans.extend(other.spans.iter().map(|span| TokenSpan {
            start: span.start + at,
            end: span.end + at,
            ..*span
        }));
        self.tokens.extend_from_slice(&other.tokens);
    }
}

/// How many of an output's first bytes decide a start the output decides ([`Undecided`]), and so
/// the most text the reader holds before it starts: the newline and the indentation a
/// pretty-printed JSON writes after its opener fit under it.
const DECIDING_BYTES: usize = 64;

/// A forced call's start that the output's first bytes decide ([`Assignment::opener`]): what the
/// reader keeps until they arrive.
struct Undecided {
    /// The part of the prompt the engine replays, unless the output starts in the call.
    prompt: Box<str>,
    /// The first byte of the forced JSON.
    opener: u8,
    /// Whether the output's first byte that is not whitespace came and was the opener: the next
    /// such byte decides. Kept so each step scans its own delta alone.
    opened: bool,
    /// What the output wrote before its start was decided, with its tokens' spans: the
    /// whitespace it began with, then the opener and the whitespace after it. Its text stays
    /// short of [`DECIDING_BYTES`]: the step that fills them decides. A token whose bytes the
    /// decoder kept back adds its span and no text. Read once the start is decided, by the
    /// output's first bytes or by its end.
    held: Delta,
}

/// One choice's reader: the engine over the choice's output, fed what the stop decoder releases
/// with the span of every token in it. The events of a step go into the caller's [`Events`] and
/// borrow the step's [`Delta`] text; the caller reads them in place and recycles the buffer
/// (`docs/engine.md`, the shape of a step's events). What it reads is counted as it goes
/// ([`telemetry`]) and recorded when the output ends.
pub(crate) struct Reader {
    engine: Engine,
    /// The `table` label of the format, for the counters and the line.
    table: &'static str,
    /// What the output said so far that was not content, reasoning or a call.
    tally: Tally,
    /// The last token the decoder saw: text it releases later belongs to it.
    last_token: Option<u32>,
    /// The bytes of the output read so far.
    read: usize,
    ended: bool,
    /// Drop the events of every call after the first.
    first_call_only: bool,
    /// The calls the request's `tool_choice` forbids ([`Assignment::forbids_calls`]), dropped as
    /// the output makes them: the table read them whole, and the engine's own server delivers
    /// nothing of a call under `none`.
    forbidden: ForbiddenCalls,
    /// Give every call the family's id by its place in the conversation
    /// ([`Assignment::forced_ids`]).
    forced_ids: Option<(CallIdFamily, usize)>,
    /// Whether the output showed the client anything yet: text, reasoning or a call.
    shown: bool,
    /// The whitespace the table dropped before the output showed anything: the template's
    /// spacing when text follows it, the model's whole answer when nothing does, which is then
    /// released as one content event before the `Finish` ([`Self::end`]).
    unshown: String,
    /// Whether the output made a call: a call that started. Nothing is held past one: what the
    /// table refuses beside a call is debris, as the table says.
    called: bool,
    /// The bytes of the call blocks the table refused since the output's last text, while it
    /// made no call (`Dropped { MalformedCall }`: a block that names no call, Python or prose
    /// where the name goes), with the marker that led into the first of them and what the table
    /// dropped between two of them (its whitespace, the next marker), in the output's order: the
    /// output's own slice from the first block's opener to the last refused byte
    /// ([`Self::hold_refused`]). Debris when text or a call follows, which stays dropped as the
    /// table says; the tail of the model's answer when nothing else comes after it (its whole
    /// answer, or what it wrote after its text), which is then released as one content event
    /// before the `Finish` ([`Self::end`]).
    refused: String,
    /// What the table dropped since the last event that was not a drop, while the output made no
    /// call, run by run with its place in the output: the marker that leads into a refused block
    /// and the bytes between two refused blocks go out with the blocks
    /// ([`Self::hold_refused`]); the rest is the template's.
    framing: Vec<Framing>,
    /// The bytes of the runs `framing` holds, one after another in one buffer, kept for the
    /// response: a run is a range of it, and a run that follows the last one grows the last.
    framing_text: String,
    /// The text the test trap springs on, when this process runs with one
    /// ([`isolation::test_trap`]): a step whose delta carries it panics the reader, so the
    /// request's isolation is driven end to end. `None` in every other process: one branch per
    /// step.
    trap: Option<&'static str>,
    /// The request's span ([`Assignment::span`]): an output that ends other than clean leaves its
    /// reading on it ([`Tally::mark`]).
    span: Span,
    /// The request's mark ([`Assignment::marked`]): set by the first reading that is not clean.
    marked: Arc<AtomicBool>,
    /// The start the output's first bytes decide, until they do ([`Self::decide`]). `None` for
    /// every reader whose start the assignment knew, and from the deciding byte on: one branch
    /// per step.
    undecided: Option<Box<Undecided>>,
}

impl Reader {
    /// A reader for one choice, seeded with the assignment's prompt; the events the prompt's
    /// replay gives (a thought the prompt opens) go into `out`, before the first delta's. `tools`
    /// are the request's as its response spec holds them: every choice's reader shares them,
    /// none copies a schema. Where the output's first bytes decide the start
    /// ([`Assignment::opener`]) nothing is replayed here: the start's events come with the step
    /// that decides it.
    pub(crate) fn new(
        assignment: &Assignment,
        tools: Arc<[Tool]>,
        out: &mut Events<'_>,
    ) -> Result<Self, SymphonyError> {
        let forbids = assignment.forbids_calls.over(&tools);
        let declared = match &assignment.forcing {
            Some(name) => Declared::shared(tools).forcing(name.clone()),
            None => Declared::shared(tools),
        };
        let engine = Engine::shared(Arc::clone(&assignment.format), declared);
        let mut reader = Self {
            table: assignment.table(),
            engine: if assignment.json_answer {
                engine.reads_json_answer()
            } else {
                engine
            },
            tally: Tally::new(assignment.past_the_end),
            last_token: None,
            read: 0,
            ended: false,
            first_call_only: false,
            forbidden: ForbiddenCalls {
                any: forbids != Forbids::Nothing,
                forbids,
                dropping: None,
                dropped: 0,
            },
            forced_ids: assignment.forced_ids,
            shown: false,
            unshown: String::new(),
            called: false,
            refused: String::new(),
            framing: Vec::new(),
            framing_text: String::new(),
            trap: isolation::test_trap(),
            span: assignment.span.clone(),
            marked: Arc::clone(&assignment.marked),
            undecided: assignment.opener.map(|opener| {
                Box::new(Undecided {
                    prompt: assignment.prompt.as_str().into(),
                    opener,
                    opened: false,
                    held: Delta::default(),
                })
            }),
        };
        if reader.undecided.is_none() {
            reader.start(&assignment.prompt, None, out)?;
        }
        Ok(reader)
    }

    /// Starts the reader after `prompt`'s replay (an empty one leaves it at the table's first
    /// state), then reads what the output `held` before its start was decided. The events borrow
    /// nothing of the caller's (a thought the prompt opens or closes carries no text of the
    /// prompt, and what was held is the reader's own), so they move from a buffer of their
    /// own into `out` one by one, whatever delta that one borrows; nothing is collected on the
    /// way.
    fn start(
        &mut self,
        prompt: &str,
        held: Option<&Delta>,
        out: &mut Events<'_>,
    ) -> Result<(), SymphonyError> {
        let mut early = Events::new();
        self.feed(
            Input::Prompt {
                token_ids: &[],
                text: prompt,
            },
            &mut early,
        )?;
        // Held text, or held tokens whose bytes the decoder kept back: both are the output's.
        if let Some(held) = held.filter(|held| !held.text.is_empty() || !held.spans.is_empty()) {
            self.feed(held.input(), &mut early)?;
        }
        for event in early.consume() {
            out.push(event.into_owned());
        }
        Ok(())
    }

    /// Decides a start the output's first bytes decide ([`Undecided`]), by the bytes among the
    /// output's first [`DECIDING_BYTES`] that are not whitespace, wherever the steps cut them.
    /// The first such byte picks: any byte but the JSON's opener starts the reader where the
    /// prompt's replay leaves it, at once, so a thought waits for nothing. After the opener the
    /// next such byte can only refute: one the JSON can write there ([`follows_the_opener`])
    /// starts the reader in the call, with nothing of the prompt replayed, and any other is a
    /// thought that began with the opener's byte, read after the replay. Whitespace is the
    /// JSON's until a byte says otherwise: deciding bytes that run out on it, before the opener
    /// or after it, start the reader in the call, so the hold ends there however long the model
    /// writes whitespace. It is ASCII's whitespace, the form feed among it, which JSON does not
    /// take: no grammar writes that byte, so none comes where the schema holds the output. The
    /// start's events and those of what was held before it go into `out` ahead of the step's
    /// own, and the caller reads `delta` (`true`). A delta that decides nothing (whitespace
    /// alone, or the opener with nothing after it, short of the deciding bytes) is held with
    /// its tokens' spans, and the caller reads nothing (`false`).
    fn decide(&mut self, delta: &Delta, out: &mut Events<'_>) -> Result<bool, SymphonyError> {
        let Some(mut undecided) = self.undecided.take() else {
            return Ok(true);
        };
        // The deciding bytes this delta holds: what was held is short of them.
        let left = DECIDING_BYTES.saturating_sub(undecided.held.text.len());
        let deciding = &delta.text.as_bytes()[..delta.text.len().min(left)];
        let mut bytes = deciding.iter().filter(|byte| !byte.is_ascii_whitespace());
        let mut next = bytes.next();
        if !undecided.opened && next == Some(&undecided.opener) {
            undecided.opened = true;
            next = bytes.next();
        }
        let prompt = match next {
            Some(&byte) if undecided.opened && follows_the_opener(undecided.opener, byte) => "",
            Some(_) => &*undecided.prompt,
            None if deciding.len() < left => {
                undecided.held.append(delta);
                self.undecided = Some(undecided);
                return Ok(false);
            }
            None => "",
        };
        self.start(prompt, Some(&undecided.held), out)?;
        Ok(true)
    }

    /// Under `parallel_tool_calls: false` or `disable_parallel_tool_use`, the
    /// events of every call after the first are dropped: the table's grammar
    /// limits a forced call, the reader limits a free one.
    pub(crate) fn first_call_only(mut self, on: bool) -> Self {
        self.first_call_only = on;
        self
    }

    /// For a stream: a call starts at its name ([`Engine::starts_calls_at_name`]), so the
    /// client reads which function is called before the arguments' key arrives. A whole
    /// answer is read without it: an output cut behind a name holds no call there. Under a
    /// choice that drops a call (`none`, a named function, an allowed list, and a `required`
    /// list, which drops a call to a function the request does not declare) the engine stays
    /// as built: a call that ends before its arguments is no call there and its bytes stay
    /// the answer's content, where a call dropped at its name would take them with it.
    pub(crate) fn starts_calls_at_name(mut self) -> Self {
        if self.forbidden.forbids == Forbids::Nothing {
            self.engine = self.engine.starts_calls_at_name();
        }
        self
    }

    /// Reads `token_ids` through `decoder` as one delta: the text the decoder releases, with each
    /// token's span in it, zero-width for a token whose bytes it holds back, into `delta`, and
    /// the events into `out`, where they borrow the text. Says whether the decoder matched a
    /// stop, after which the rest of the tokens are not read.
    pub(crate) fn read<'d>(
        &mut self,
        decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
        delta: &'d mut Delta,
        out: &mut Events<'d>,
    ) -> Result<bool, SymphonyError> {
        delta.clear();
        // The step's tokens, reserved before the first push: a whole output read in one step
        // (the request that does not stream) grew both vectors by doubling from empty, eight
        // regrows and 3.3 KiB moved for an answer of forty tokens. A step of one token skips
        // the reserve: its push grows an empty vector the same way, and a stream's delta keeps
        // its capacity from the step before, so the check found the room there on every step.
        if token_ids.len() > 1 {
            delta.spans.reserve(token_ids.len());
            delta.tokens.reserve(token_ids.len());
        }
        let mut stopped = false;
        for &token_id in token_ids {
            let start = delta.text.len();
            stopped = decoder
                .process_token_into(token_id, &mut delta.text)
                .map_err(|e| {
                    self.failed(format!(
                        "Stop decoder failed to process token {token_id}: {e}"
                    ))
                })?
                .stopped();
            delta.spans.push(TokenSpan {
                token_id,
                start,
                end: delta.text.len(),
                continued: false,
            });
            delta.tokens.push(token_id);
            self.last_token = Some(token_id);
            if stopped {
                break;
            }
        }
        self.read += delta.text.len();
        if let Some(marker) = self.trap {
            if delta.text.contains(marker) {
                isolation::spring_test_trap(marker);
            }
        }
        if self.undecided.is_some() && !self.decide(delta, out)? {
            return Ok(stopped);
        }
        self.feed(delta.input(), out)?;
        Ok(stopped)
    }

    /// Reads `text` the decoder released after its last token, which held it back as a possible
    /// stop: bytes of that token. The text moves into `delta`, whose text the events borrow.
    pub(crate) fn tail<'d>(
        &mut self,
        text: String,
        delta: &'d mut Delta,
        out: &mut Events<'d>,
    ) -> Result<(), SymphonyError> {
        if text.is_empty() {
            return Ok(());
        }
        delta.clear();
        delta
            .spans
            .extend(self.last_token.map(|token_id| TokenSpan {
                token_id,
                start: 0,
                end: text.len(),
                continued: true,
            }));
        delta.text = text;
        self.read += delta.text.len();
        if self.undecided.is_some() && !self.decide(delta, out)? {
            return Ok(());
        }
        self.feed(delta.input(), out)
    }

    /// Ends the output with the engine's finish, once: the events, the last of them the
    /// `Finish`, go into `out`; nothing when the output already ended. An output that showed
    /// nothing (no text, no reasoning, no call) and dropped whitespace releases that whitespace
    /// as one content event right before the `Finish`: it is the whole answer, not the template's
    /// spacing around text that never came. The chat stream carries it as the engine's own server
    /// streams it; a whole response's adapter applies its own rule for whitespace-only content
    /// (the chat message, the Messages blocks and the Responses items leave it out). The markers
    /// and separators dropped with it stay dropped: they are the template's. An output that made
    /// no call and ended in call blocks the table refused (a call written as Python where the
    /// name goes, which the keyed syntax never reads as content), after its thought or after its
    /// text, releases the same way the output's own slice from the first of those blocks' opener
    /// to the last refused byte: the blocks' bytes as the table refused them, the marker that led
    /// into the first, and what the table dropped between two blocks (its whitespace, the next
    /// marker), in the order the model wrote them, after the text that came before, so the
    /// client reads what the model wrote, as its own server shows it, rather than a text that
    /// ends where the call should be; the whitespace before the first opener and after the last
    /// refused byte stays the template's, and the table's rule keeps debris out of an answer
    /// that went on past the block: a block that text or a call follows stays dropped. The
    /// output is recorded here, under `who`: the request span's mark when it was not clean
    /// ([`Tally::mark`]), then its counters and its line, written inside the span
    /// ([`Tally::report`]); an output of whitespace alone counts as `empty`, one with refused
    /// blocks as `malformed`, whether or not bytes were released, and the line says how many
    /// were.
    pub(crate) fn end(
        &mut self,
        finish: EngineFinish,
        who: Who<'_>,
        out: &mut Events<'_>,
    ) -> Result<(), SymphonyError> {
        if self.ended {
            return Ok(());
        }
        self.ended = true;
        // An output that ended before its bytes decided its start (whitespace alone, or the
        // opener and nothing after it) stands where the prompt's replay leaves it.
        if let Some(undecided) = self.undecided.take() {
            self.start(&undecided.prompt, Some(&undecided.held), out)?;
        }
        self.feed(Input::End { finish }, out)?;
        if self.tally.is_empty() && !self.unshown.is_empty() {
            let released = std::mem::take(&mut self.unshown);
            self.tally.released(released.len());
            release_before_the_finish(released, out);
        } else if !self.refused.is_empty() {
            let released = std::mem::take(&mut self.refused);
            self.tally.released(released.len());
            release_before_the_finish(released, out);
        }
        self.tally
            .mark(&self.span, &self.marked, self.table, who.choice);
        // The line is written inside the request's span, after the mark: a streamed response's
        // reading runs outside the span, and only from within does the line carry the span's
        // request id and the fields just marked.
        self.span.in_scope(|| self.tally.report(self.table, who));
        Ok(())
    }

    /// Whether the output ended.
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    /// The error of an output the reader could not read whole, which the request fails with:
    /// the decoder refused a token or the text it held, or the table refused the output
    /// ([`Self::feed`]). The output is recorded as failed for the table, left on the request's
    /// span with the bytes read so far, and is over: its end counts nothing more.
    pub(crate) fn failed(&mut self, why: String) -> SymphonyError {
        self.ended = true;
        telemetry::record_failed(self.table);
        telemetry::mark_failed(&self.span, &self.marked, self.table, self.read);
        SymphonyError::Read(why)
    }

    /// Feeds `input` to the engine, its events into `out` after the ones already there: the
    /// reader's one way in. A call the step started takes its family's id first, where the
    /// assignment gives one ([`Assignment::forced_ids`]). The events this step added are tallied,
    /// the calls the request's choice forbids are dropped ([`ForbiddenCalls`]), a call kept
    /// after a dropped one takes its family's id by the place it is returned in, and the rest are
    /// limited to the first call where the request asks. An output the engine refuses is
    /// recorded as failed for the table, which the error names.
    fn feed<'d>(&mut self, input: Input<'d>, out: &mut Events<'d>) -> Result<(), SymphonyError> {
        let before = out.len();
        let last = out.as_slice().last().map_or(0, telemetry::bytes_of);
        self.engine.feed(input, out).map_err(|e| {
            self.failed(format!(
                "Symphony failed to read the output with the {} table: {e}",
                self.table
            ))
        })?;
        // Text of one kind pushed right after text of that kind joins it (`Events::push`): bytes
        // that grew an event already tallied are counted without an event of their own.
        if let Some(grown) = before.checked_sub(1).map(|at| &out.as_slice()[at]) {
            self.tally.grew(telemetry::bytes_of(grown) - last);
        }
        let fresh = &mut out.as_mut_slice()[before..];
        if let Some((family, history)) = self.forced_ids {
            for event in fresh.iter_mut() {
                if let Event::ToolCallStart {
                    index, id, name, ..
                } = event
                {
                    *id = family.id(name, *index as usize, history);
                }
            }
        }
        // A step that adds no event ends here: nothing below acts on it.
        if fresh.is_empty() {
            return Ok(());
        }
        // Where the step's first event starts in the output: the held runs below are placed by it.
        let start = self.tally.at();
        // Under a choice that forbids calls the step is tallied where they are dropped
        // ([`ForbiddenCalls::drop_from`]).
        if !self.forbidden.any {
            for event in fresh.iter() {
                self.tally.see(event, self.first_call_only);
            }
        }
        // While the output makes no call, the bytes of the call blocks the table refuses are
        // kept for the end ([`Self::end`]), with the marker that led into the first and what the
        // table dropped between two ([`Self::hold_refused`]); text after them makes them debris
        // and the hold starts over behind it, a call makes them debris for good. What the table
        // dropped before a thought is the template's.
        if !self.called {
            let mut at = start;
            for event in fresh.iter() {
                let bytes = telemetry::bytes_of(event);
                match event {
                    Event::Dropped {
                        text,
                        why: DropReason::MalformedCall,
                    } => self.hold_refused(&text.text),
                    Event::Dropped { text, why } => {
                        // A marker of the table leads into a block; the whitespace a call syntax
                        // drops as wrapping (the newline before a block's first tag) is not one.
                        let marker =
                            matches!(why, DropReason::Wrapper) && !text.text.trim().is_empty();
                        self.hold_framing(&text.text, at, marker);
                    }
                    Event::Content(text) | Event::Malformed { text, .. }
                        if !text.text.is_empty() =>
                    {
                        self.drop_held();
                    }
                    Event::ToolCallStart { .. } => {
                        self.called = true;
                        self.drop_held();
                        break;
                    }
                    Event::Reasoning(_) | Event::ReasoningStart | Event::ReasoningEnd => {
                        self.clear_framing();
                    }
                    Event::Content(_)
                    | Event::Malformed { .. }
                    | Event::ToolCallArguments { .. }
                    | Event::ToolCallEnd { .. }
                    | Event::Finish { .. } => {}
                }
                at += bytes;
            }
        }
        // Until the output shows something, the whitespace it drops is kept for the end
        // ([`Self::end`]); the first text, reasoning or call makes it the template's for good.
        if !self.shown {
            for event in fresh.iter() {
                match event {
                    Event::Dropped {
                        text,
                        why: DropReason::Whitespace,
                    } => {
                        hold_text(&mut self.unshown, &text.text);
                    }
                    Event::Content(_)
                    | Event::Reasoning(_)
                    | Event::Malformed { .. }
                    | Event::ToolCallStart { .. }
                    | Event::ToolCallArguments { .. }
                    | Event::ToolCallEnd { .. } => {
                        self.shown = true;
                        self.unshown = String::new();
                        break;
                    }
                    Event::Dropped { .. }
                    | Event::ReasoningStart
                    | Event::ReasoningEnd
                    | Event::Finish { .. } => {}
                }
            }
        }
        if self.forbidden.any {
            let (tally, first_call_only) = (&mut self.tally, self.first_call_only);
            self.forbidden.drop_from(fresh, |event, forbidden| {
                if forbidden {
                    tally.see_forbidden(event);
                } else {
                    tally.see(event, first_call_only);
                }
            });
            // A call kept after a dropped one moved down: where the family numbers its ids by
            // the conversation, its id is the one of the place it is returned in. Until a call
            // is dropped every call keeps the id it took at the step's start.
            if self.forbidden.dropped > 0 {
                if let Some((family, history)) = self.forced_ids {
                    for event in fresh.iter_mut() {
                        if let Event::ToolCallStart {
                            index, id, name, ..
                        } = event
                        {
                            *id = family.id(name, *index as usize, history);
                        }
                    }
                }
            }
        }
        if self.first_call_only {
            symphony::event::first_call_only(fresh);
        }
        Ok(())
    }

    /// `text` the table dropped at `at` in the output, a `marker` of the table
    /// (`Dropped { Wrapper }` with a byte that is not whitespace) or not: held after the runs
    /// dropped since the last event that was not a drop, joined to the last when neither is a
    /// marker and it follows it, so a run of whitespace is one entry however it arrived.
    fn hold_framing(&mut self, text: &str, at: usize, marker: bool) {
        let start = self.framing_text.len();
        hold_text(&mut self.framing_text, text);
        if self.framing.capacity() == 0 {
            self.framing.reserve(HELD_RUNS_FLOOR);
        }
        match self.framing.last_mut() {
            Some(last) if !marker && !last.marker && last.at + last.text.len() == at => {
                last.text.end = self.framing_text.len();
            }
            _ => self.framing.push(Framing {
                text: start..self.framing_text.len(),
                at,
                marker,
            }),
        }
    }

    /// `text`, bytes of a call block the table refused: held for the end after the block's
    /// opener when it is the first block, which is the last marker the table dropped before it
    /// (what it dropped before the marker is the template's), or after everything the table
    /// dropped since the last refused byte when a block is held already (the whitespace between
    /// two blocks, the next block's opener): the output's own slice, in its order, from the
    /// first opener to the last refused byte.
    fn hold_refused(&mut self, text: &str) {
        let from = if self.refused.is_empty() {
            self.framing
                .iter()
                .rposition(|run| run.marker)
                .unwrap_or(self.framing.len())
        } else {
            0
        };
        for run in self.framing.drain(from..) {
            hold_text(&mut self.refused, &self.framing_text[run.text]);
        }
        self.clear_framing();
        hold_text(&mut self.refused, text);
    }

    /// The held runs go, their one buffer kept for the next.
    fn clear_framing(&mut self) {
        self.framing.clear();
        self.framing_text.clear();
    }

    /// The output went on past what was held, with text or a call: the held runs are debris
    /// beside it, as the table says. After text the hold starts over, for the blocks that may end
    /// the answer; after a call nothing more is held (`called`).
    fn drop_held(&mut self) {
        self.refused.clear();
        self.clear_framing();
    }
}

/// The calls of one output that the request's `tool_choice` forbids ([`Forbids`]), dropped as the
/// reader meets them: every call under `none`; under a named function, an allowed list or
/// `required`, each call to a function outside the tools the choice leaves callable; none under
/// `auto`.
struct ForbiddenCalls {
    /// The choice forbids a call, some or every one: the one test a step pays under `auto`, which
    /// forbids none and whose steps the reader tallies itself ([`Reader::feed`]).
    any: bool,
    forbids: Forbids,
    /// The engine's index of the call being dropped, from its start to its end.
    dropping: Option<u32>,
    /// How many calls were dropped so far, by which a call kept after them moves down.
    dropped: u32,
}

impl ForbiddenCalls {
    /// For a step under a choice that forbids calls ([`Self::any`]; a step under `auto` the reader
    /// tallies itself): tells `seen` of each of the step's `fresh` events, with whether the choice
    /// forbids it, then rewrites a forbidden call's events as dropped text
    /// (`event::forbidden_calls`): never a call or content. A call is asked once, at its start
    /// ([`Forbids::call_to`]): by its function's name and by the engine's word on whether the
    /// request declares it, the `Malformed { UnknownTool }` event of no bytes right before the
    /// start, which goes with the call it announces. The call's later events follow that answer
    /// by their index, so no name is compared on a delta, and text the table could not read
    /// inside a dropped call (the bytes behind the arguments of an object the output never
    /// closed) is dropped with it. A table reads one call at a time, so a call kept after a
    /// dropped one moves down by the count dropped before it, and the calls delivered are
    /// numbered from zero; the finish counts the calls kept, and with none kept it is forbidden
    /// too: a `tool_calls` finish is the engine's `stop` again.
    fn drop_from(&mut self, fresh: &mut [Event<'_>], mut seen: impl FnMut(&Event<'_>, bool)) {
        match self.forbids {
            Forbids::Nothing => {}
            Forbids::Every | Forbids::Undeclared | Forbids::AllBut(_) => {
                let mut undeclared = false;
                for event in fresh {
                    let announced = std::mem::take(&mut undeclared);
                    let forbidden = match event {
                        // The mark of a call to a function the request did not declare, which no
                        // choice leaves callable.
                        Event::Malformed {
                            why: symphony::MalformedReason::UnknownTool,
                            ..
                        } => {
                            undeclared = true;
                            true
                        }
                        Event::ToolCallStart { index, name, .. } => {
                            let forbidden = self.forbids.call_to(name, !announced);
                            if forbidden {
                                self.dropping = Some(*index);
                                self.dropped += 1;
                            } else {
                                *index -= self.dropped;
                            }
                            forbidden
                        }
                        // Text the table could not read inside the call being dropped is that
                        // call's: it goes with it, as dropped text.
                        Event::Malformed { text, .. } if self.dropping.is_some() => {
                            *event = Event::Dropped {
                                text: std::mem::take(text),
                                why: DropReason::ForbiddenCall,
                            };
                            false
                        }
                        Event::ToolCallArguments { index, .. } => {
                            let forbidden = self.dropping == Some(*index);
                            if !forbidden {
                                *index -= self.dropped;
                            }
                            forbidden
                        }
                        Event::ToolCallEnd { index, .. } => {
                            let forbidden = self.dropping == Some(*index);
                            if forbidden {
                                self.dropping = None;
                            } else {
                                *index -= self.dropped;
                            }
                            forbidden
                        }
                        Event::Finish { tool_calls, .. } => {
                            *tool_calls -= self.dropped;
                            self.dropped > 0 && *tool_calls == 0
                        }
                        _ => false,
                    };
                    seen(&*event, forbidden);
                    if forbidden {
                        symphony::event::forbidden_calls(std::slice::from_mut(event));
                    }
                }
            }
        }
    }
}

/// The capacity a held buffer takes on its first push: the runs a turn of a few calls drops (its
/// markers, the whitespace between its blocks) and a refused block fit without a regrow, where the
/// buffers grew by doubling from empty (the tool-call path's allocation census: eight regrows and
/// 3.3 KiB moved per request). Nothing is reserved before the first push, so an output that holds
/// nothing allocates nothing here.
const HELD_BYTES_FLOOR: usize = 256;
/// The runs `framing` holds before its first regrow.
const HELD_RUNS_FLOOR: usize = 8;

/// `text` onto `held`, the buffer reserved to the floor on its first push.
fn hold_text(held: &mut String, text: &str) {
    if held.capacity() == 0 {
        held.reserve(HELD_BYTES_FLOOR.max(text.len()));
    }
    held.push_str(text);
}

/// A run the table dropped, held until what follows says whose it is: a refused block's opener
/// and the bytes between two refused blocks go out with the blocks; the rest is the template's.
struct Framing {
    /// The run's bytes, in the reader's one buffer of held runs (`framing_text`).
    text: std::ops::Range<usize>,
    /// Where the run starts in the output.
    at: usize,
    /// Whether the run is a marker of the table (`Dropped { Wrapper }` with a byte that is not
    /// whitespace), as a block's opener is; the whitespace a call syntax drops as wrapping, as
    /// the newline before a block's first tag, is not one.
    marker: bool,
}

/// `released` goes into `out` as one content event right before the `Finish`, which stays the
/// last event: it makes way for the content and follows it.
fn release_before_the_finish(released: String, out: &mut Events<'_>) {
    let content = Event::Content(Text::uncounted(released));
    match out.as_mut_slice().last_mut() {
        Some(last) if matches!(last, Event::Finish { .. }) => {
            let finish = std::mem::replace(last, content);
            out.push(finish);
        }
        _ => out.push(content),
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/symphony/reader/tests.rs"]
mod tests;

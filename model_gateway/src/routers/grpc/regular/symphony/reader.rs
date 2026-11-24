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
mod tests {
    use super::*;

    #[test]
    fn a_shared_tables_asked_variant_is_kept_and_read_with_by_the_next_request() {
        let shared = Table::Qwen3.shared();
        let first = asked(Arc::clone(&shared), true, false);
        let second = asked(Arc::clone(&shared), true, false);
        assert!(
            Arc::ptr_eq(&first, &second),
            "the same ask of the same shared table reads with the one kept variant"
        );
        assert_eq!(
            *first,
            (*shared).clone().without_calls(),
            "the kept variant is the table without its calls"
        );
        let other = asked(Arc::clone(&shared), false, true);
        assert!(
            !Arc::ptr_eq(&first, &other),
            "another ask is another variant"
        );
        assert_eq!(*other, (*shared).clone().without_reasoning());
        assert!(
            Arc::ptr_eq(&asked(Arc::clone(&shared), false, false), &shared),
            "a request that asks for nothing reads with the shared table itself"
        );
        let named = forced(Arc::clone(&shared), ForcedCall::Arguments, false);
        assert!(
            Arc::ptr_eq(
                &named,
                &forced(Arc::clone(&shared), ForcedCall::Arguments, false)
            ),
            "a forced call's variant is kept like the others"
        );
        assert_eq!(
            *named,
            (*shared).clone().with_forced_call(ForcedCall::Arguments)
        );
        let listed = forced(Arc::clone(&shared), ForcedCall::List, true);
        assert_eq!(
            *listed,
            (*shared)
                .clone()
                .with_forced_call(ForcedCall::List)
                .without_reasoning(),
            "the list a required call writes, the thought kept in the content"
        );
        assert!(!Arc::ptr_eq(&named, &listed) && !Arc::ptr_eq(&named, &first));
        let unique = Arc::new((*shared).clone());
        let weak = Arc::downgrade(&unique);
        let made_over = asked(unique, true, true);
        assert_eq!(
            *made_over,
            (*shared).clone().without_calls().without_reasoning(),
            "a table nobody else holds is made over in place"
        );
        assert!(
            weak.upgrade().is_none(),
            "a table nobody else holds is not kept: nothing pins it"
        );
    }
    use symphony::formats::Table;

    use crate::routers::grpc::{
        regular::symphony::testing::{assign, constraint, *},
        utils::parse_json_schema_response,
    };

    /// A reader over `format` after `prompt`, fed `pieces` one by one, ended with `finish`.
    fn read_pieces(
        format: Format,
        prompt: &str,
        pieces: &[&str],
        finish: EngineFinish,
    ) -> Vec<Event<'static>> {
        let assignment = Assignment {
            format: Arc::new(format),
            prompt: prompt.to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        // A step's events borrow its delta, as a stream's do: each piece is a step of its own,
        // its events taken owned before the next.
        let mut all: Vec<Event<'static>> = Vec::new();
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
        all.extend(events.into_owned().drain());
        for piece in pieces {
            let mut delta = Delta::default();
            let mut events = Events::new();
            reader
                .tail((*piece).to_string(), &mut delta, &mut events)
                .expect("read");
            all.extend(events.into_owned().drain());
        }
        let mut events = Events::new();
        reader.end(finish, who(), &mut events).expect("ended");
        all.extend(events.into_owned().drain());
        all
    }

    fn newlines(count: usize) -> Vec<&'static str> {
        vec!["\n"; count]
    }

    /// The content events and whether the last event is the `Finish`.
    fn content_events(events: &[Event<'_>]) -> (Vec<String>, bool) {
        let contents = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.to_string()),
                _ => None,
            })
            .collect();
        let finish_last = matches!(events.last(), Some(Event::Finish { .. }));
        (contents, finish_last)
    }

    #[test]
    fn an_output_of_whitespace_alone_goes_out_as_one_content_event_before_the_finish() {
        use symphony::{
            formats::{minimax_m3, qwen3},
            tagged::{Placement, Spelling},
            CallSyntax,
        };
        // The recorded case: the thought closed at once, then nothing but newlines to the
        // forced length. The marker is the template's and stays dropped; the newlines are the
        // answer and go out as one content event, with the `Finish` still last.
        let pieces: Vec<&str> = std::iter::once("</mm:think>")
            .chain(newlines(300))
            .collect();
        let events = read_pieces(minimax_m3(), "]~b]ai\n", &pieces, EngineFinish::Length);
        let (contents, finish_last) = content_events(&events);
        assert_eq!(contents, vec!["\n".repeat(300)], "{events:?}");
        assert!(finish_last, "{events:?}");
        assert!(events
            .iter()
            .all(|event| !matches!(event, Event::Reasoning(_))));
        // The same over a table with tagged calls, whose template opens the thought in the
        // prompt: the output closes it at once, then nothing but newlines.
        let tagged = qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine));
        let pieces: Vec<&str> = std::iter::once("</think>").chain(newlines(300)).collect();
        let events = read_pieces(
            tagged,
            "<|im_start|>assistant\n<think>\n",
            &pieces,
            EngineFinish::Length,
        );
        let (contents, finish_last) = content_events(&events);
        assert_eq!(contents, vec!["\n".repeat(300)], "{events:?}");
        assert!(finish_last, "{events:?}");
        // Whitespace alone from the first byte, under a stop.
        let events = read_pieces(minimax_m3(), "]~b]ai\n", &newlines(3), EngineFinish::Stop);
        assert_eq!(content_events(&events).0, vec!["\n\n\n"]);
    }

    /// The same output as a whole chat message: the released whitespace folds into the message,
    /// whose adapter leaves whitespace-only content out, as the Messages and Responses adapters
    /// do; the output is still counted as `empty`, once.
    #[test]
    fn a_whole_chat_message_of_whitespace_alone_has_no_content_and_counts_as_empty() {
        use symphony::{adapt::chat::message, formats::minimax_m3};
        let scrape = rendered(|| {
            let events = read_pieces(minimax_m3(), "]~b]ai\n", &newlines(3), EngineFinish::Stop);
            assert_eq!(content_events(&events).0, vec!["\n\n\n"]);
            let choice = message(0, &events);
            assert_eq!(choice.message.content, None, "{events:?}");
            assert!(choice.message.tool_calls.is_none());
            assert!(choice.message.reasoning_content.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        });
        let outputs = line_with(
            &scrape,
            "smg_symphony_outputs_total",
            &["table=\"minimax_m3\"", "outcome=\"empty\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
    }

    #[test]
    fn whitespace_before_text_or_a_call_is_the_templates_and_goes_out_once_at_most() {
        use symphony::formats::minimax_m3;
        // Text after the lead: the table releases the lead as content before it; the reader
        // delivers nothing of its own.
        let pieces = ["</mm:think>", "\n", "\n", "Hello"];
        let events = read_pieces(minimax_m3(), "]~b]ai\n", &pieces, EngineFinish::Stop);
        assert_eq!(content_of(&events), "\n\nHello");
        assert_eq!(content_events(&events).0.len(), 1, "{events:?}");
        // A call after the lead: the lead is the template's spacing and no content goes out.
        let pieces = [
            "</mm:think>\n]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">",
            "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
        ];
        let events = read_pieces(minimax_m3(), "]~b]ai\n", &pieces, EngineFinish::Stop);
        assert_eq!(content_of(&events), "");
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
        // The template's separator alone is the template's: nothing goes out, and the output
        // is still empty.
        let events = read_pieces(
            minimax_m3(),
            "]~b]ai\n",
            &["]<]minimax[>["; 5],
            EngineFinish::Length,
        );
        assert_eq!(content_of(&events), "");
        // Whitespace inside a thought is reasoning as it arrives, so nothing is held for the end.
        let pieces: Vec<&str> = std::iter::once("<mm:think>").chain(newlines(4)).collect();
        let events = read_pieces(minimax_m3(), "]~b]ai\n", &pieces, EngineFinish::Length);
        assert_eq!(content_of(&events), "");
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "\n\n\n\n");
    }

    /// The generation prompt's tail of a table whose template opens the thought: the turn
    /// opened, and its thought.
    const GLM_PROMPT: &str = "<|assistant|><think>";

    /// An output that answers with nothing but a call written as Python inside the block's
    /// markers, after its thought (the recorded shape): the table refuses the block, never
    /// content, so the client would read an empty answer under a finish. The reader releases the
    /// block as one content event before the `Finish`, as the model wrote it: the marker that led
    /// into it, its text and the marker that ended it; the thought stays reasoning; the whole
    /// chat message carries the same content. The same when the stream's end cuts the block. The
    /// output counts as malformed under its reasons, with the bytes released on its line.
    #[test]
    fn an_answer_of_a_refused_call_block_alone_goes_out_as_its_bytes_before_the_finish() {
        use symphony::{adapt::chat::message, formats::glm};
        let scrape = rendered(|| {
            let pieces = [
                "plan</think>",
                "<tool_call>",
                "get_stock_info(company=",
                "\"Apple\")",
                "</tool_call>",
            ];
            let events = read_pieces(glm(), GLM_PROMPT, &pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert_eq!(
                contents,
                vec!["<tool_call>get_stock_info(company=\"Apple\")</tool_call>"],
                "{events:?}"
            );
            assert!(finish_last, "{events:?}");
            assert!(events
                .iter()
                .all(|event| !matches!(event, Event::ToolCallStart { .. })));
            let choice = message(0, &events);
            assert_eq!(
                choice.message.content.as_deref(),
                Some("<tool_call>get_stock_info(company=\"Apple\")</tool_call>")
            );
            assert!(choice.message.tool_calls.is_none());
            assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
            // The stream's end cuts the second of two such blocks: everything from the first
            // opener goes out, the `Finish` still last.
            let pieces = [
                "</think><tool_call>searchHotels(location=\"San Diego\")",
                "<tool_call>searchHotels(location=\"San",
                " Diego\")",
            ];
            let events = read_pieces(glm(), GLM_PROMPT, &pieces, EngineFinish::Length);
            let (contents, finish_last) = content_events(&events);
            assert_eq!(
                contents,
                vec!["<tool_call>searchHotels(location=\"San Diego\")<tool_call>searchHotels(location=\"San Diego\")"],
                "{events:?}"
            );
            assert!(finish_last, "{events:?}");
        });
        let outputs = line_with(
            &scrape,
            "smg_symphony_outputs_total",
            &["table=\"glm\"", "outcome=\"malformed\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 2"), "{outputs}");
        let reason = line_with(
            &scrape,
            "smg_symphony_malformed_total",
            &["table=\"glm\"", "reason=\"text_where_the_name_should_be\""],
        )
        .expect("the reason counter");
        assert!(reason.ends_with(" 2"), "{reason}");
        assert!(
            line_with(
                &scrape,
                "smg_symphony_outputs_total",
                &["table=\"glm\"", "outcome=\"empty\""]
            )
            .is_none(),
            "{scrape}"
        );
    }

    /// A call whose elements nest past the bound the JSON reader applies to every syntax
    /// (`PartialJson::DEFAULT_MAX_DEPTH` containers, the arguments object counted) still reaches
    /// the client as a call, and as no whole one: the member that crossed the bound is refused,
    /// something the model put in the call that the client will not get, so no close is made up
    /// for the call (the rule of every tagged syntax): its arguments are the containers the
    /// reader took, open, which a strict reader refuses, and the output counts as malformed
    /// under `invalid_arguments`, once.
    #[test]
    fn a_call_nested_past_the_bound_reaches_the_client_as_no_whole_call_and_counts_as_malformed() {
        use symphony::{adapt::chat::message, formats::minimax_m3, json::PartialJson};
        let levels = PartialJson::DEFAULT_MAX_DEPTH + 1;
        let body = format!("{}1{}", "<d>".repeat(levels), "</d>".repeat(levels))
            .replace('<', "]<]minimax[>[<");
        let output = format!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">{body}\
             ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let scrape = rendered(|| {
            let events = read_pieces(minimax_m3(), "]~b]ai\n", &[&output], EngineFinish::Stop);
            let choice = message(0, &events);
            let calls = choice.message.tool_calls.expect("the call");
            assert_eq!(calls.len(), 1, "{events:?}");
            assert_eq!(calls[0].function.name, "lookup");
            let arguments = calls[0].function.arguments.as_deref().expect("arguments");
            // The arguments object and the containers below it, as many as the reader takes,
            // all open: no close is made up for a call the syntax refused bytes in, and the
            // member that would have crossed the bound is unwritten.
            assert_eq!(
                arguments,
                format!(
                    "{{{}",
                    "\"d\": {".repeat(PartialJson::DEFAULT_MAX_DEPTH - 1)
                )
            );
            assert!(
                serde_json::from_str::<Value>(arguments).is_err(),
                "{arguments} reads as a whole value"
            );
            assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        });
        let outputs = line_with(
            &scrape,
            "smg_symphony_outputs_total",
            &["table=\"minimax_m3\"", "outcome=\"malformed\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        let reason = line_with(
            &scrape,
            "smg_symphony_malformed_total",
            &["table=\"minimax_m3\"", "reason=\"invalid_arguments\""],
        )
        .expect("the reason counter");
        assert!(reason.ends_with(" 1"), "{reason}");
    }

    /// Two closed blocks after the thought, with the template's newline between them: what goes
    /// out is the output's own slice from the first opener to the last refused byte, the newline
    /// and the second opener in their places, as one content event over one range, however the
    /// stream cut it; the whole chat message carries the same text. A newline after the last
    /// block stays the template's. A thought between two blocks stays reasoning: the blocks go
    /// out around it, each from its opener.
    #[test]
    fn two_refused_blocks_go_out_as_the_outputs_own_slice() {
        use symphony::{adapt::chat::message, formats::glm};
        let output = "</think><tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
        let released = "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
        let cuts: [&[&str]; 3] = [
            &[output],
            &[
                "</think>",
                "<tool_call>",
                "a(x=1)",
                "</tool_call>",
                "\n",
                "<tool_",
                "call>b(y=2)",
                "</tool_call>",
            ],
            &[
                "</think><tool_call>a(x=1)</tool_call>\n",
                "<tool_call>b(y=2)</tool_call>",
            ],
        ];
        for pieces in cuts {
            let events = read_pieces(glm(), GLM_PROMPT, pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert_eq!(contents, vec![released], "{pieces:?}: {events:?}");
            assert!(finish_last, "{events:?}");
            let choice = message(0, &events);
            assert_eq!(choice.message.content.as_deref(), Some(released));
            assert!(choice.message.tool_calls.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        }
        let events = read_pieces(glm(), GLM_PROMPT, &[output, "\n"], EngineFinish::Stop);
        assert_eq!(content_events(&events).0, vec![released], "{events:?}");
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &["</think><tool_call>a(x=1)</tool_call><think>more</think><tool_call>b(y=2)</tool_call>"],
            EngineFinish::Stop,
        );
        assert_eq!(
            content_events(&events).0,
            vec!["<tool_call>a(x=1)</tool_call><tool_call>b(y=2)</tool_call>"],
            "{events:?}"
        );
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "more");
    }

    /// A block that names no call, with the template's newline after its opener and between its
    /// tags: the call syntax drops the whitespace it carried as wrapping, bytes of no marker, so
    /// the opener stays the last marker before the block and the slice released begins at it,
    /// the newlines in their places, at every cut; the range is the one slice.
    #[test]
    fn a_nameless_block_goes_out_from_its_opener_with_its_newlines() {
        use symphony::formats::glm;
        let output = concat!(
            "plan</think><tool_call>\n<arg_key>city</arg_key>\n",
            "<arg_value>Paris</arg_value>\n</tool_call>"
        );
        let released = concat!(
            "<tool_call>\n<arg_key>city</arg_key>\n",
            "<arg_value>Paris</arg_value>\n</tool_call>"
        );
        let cuts: [&[&str]; 3] = [
            &[output],
            &[
                "plan</think><tool_call>",
                "\n",
                "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
            ],
            &[
                "plan</think>",
                "<tool_call>\n<arg_key>",
                "city</arg_key>\n",
                "<arg_value>Paris</arg_value>",
                "\n</tool_call>",
            ],
        ];
        for pieces in cuts {
            let events = read_pieces(glm(), GLM_PROMPT, pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert_eq!(contents, vec![released], "{pieces:?}: {events:?}");
            assert!(finish_last, "{events:?}");
        }
    }

    /// The generation prompt's tail of a table whose template leaves the thought to the model:
    /// the turn opened, nothing more.
    const GLM4_PROMPT: &str = "<|assistant|>";

    /// A table whose template writes a newline between the thought's close and the first call
    /// (`</think>\n<tool_call>`) holds the newline as the template's spacing and drops it when
    /// the block opens: the slice released begins at the opener, so the newline before it stays
    /// the template's, as a newline after the block, before the stop, does, and the one before
    /// the thought the model opens; the released range begins at the opener too. The whole chat
    /// message carries the same content. Each output counts as malformed, once, never as empty.
    #[test]
    fn the_templates_newlines_around_a_refused_block_stay_the_templates() {
        use symphony::{
            adapt::chat::message,
            formats::{glm4, ling},
        };
        let released = "<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
        let scrape = rendered(|| {
            let pieces = [
                "\n<think>plan</think>",
                "\n",
                "<tool_call>",
                "get_stock_info(company=",
                "\"Apple\")",
                "</tool_call>",
            ];
            let events = read_pieces(glm4(), GLM4_PROMPT, &pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert_eq!(contents, vec![released], "{events:?}");
            assert!(finish_last, "{events:?}");
            let choice = message(0, &events);
            assert_eq!(choice.message.content.as_deref(), Some(released));
            assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
            assert!(choice.message.tool_calls.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
            // The same syntax under a prompt that opens the thought, with a newline after the
            // block, before the stop.
            let pieces = [
                "plan</think>\n",
                "<tool_call>get_weather(city=\"Paris\")</tool_call>",
                "\n",
            ];
            let events = read_pieces(ling(), "<think>\n", &pieces, EngineFinish::Stop);
            assert_eq!(
                content_events(&events).0,
                vec!["<tool_call>get_weather(city=\"Paris\")</tool_call>"],
                "{events:?}"
            );
        });
        for table in ["glm4", "ling"] {
            let label = format!("table=\"{table}\"");
            let outputs = line_with(
                &scrape,
                "smg_symphony_outputs_total",
                &[&label, "outcome=\"malformed\""],
            )
            .expect("the outputs counter");
            assert!(outputs.ends_with(" 1"), "{outputs}");
            assert!(
                line_with(
                    &scrape,
                    "smg_symphony_outputs_total",
                    &[&label, "outcome=\"empty\""]
                )
                .is_none(),
                "{scrape}"
            );
        }
    }

    /// A call written as Python inside the block's markers after the answer's text (the recorded
    /// shape: a sentence, then the block): the table keeps the text as content and refuses the
    /// block, and nothing comes after it, so the block is the tail of the answer. The reader
    /// releases it, from its opener, as one content event after the text, right before the
    /// `Finish`, as the model wrote it; the whole chat message carries the text and the block, one
    /// after the other; the thought stays reasoning. The same at every cut, when the stream's end
    /// cuts the block, with two blocks and the template's newline between them after the text,
    /// and on other keyed tables, whose templates write a newline between the thought's close and
    /// the text. Each output counts as malformed under its reason, never as empty, with the bytes
    /// released on its line.
    #[test]
    fn a_refused_block_after_the_text_goes_out_after_it_before_the_finish() {
        use symphony::{
            adapt::chat::message,
            formats::{glm, glm4, ling},
        };
        let text = "I'll place both orders for you.";
        let block = "<tool_call>whole_foods_order(location=\"downtown\")</tool_call>";
        let answer = format!("{text}{block}");
        let scrape = rendered(|| {
            let cuts: [&[&str]; 3] = [
                &["plan</think>I'll place both orders for you.<tool_call>whole_foods_order(location=\"downtown\")</tool_call>"],
                &[
                    "plan</think>",
                    "I'll place both ",
                    "orders for you.",
                    "<tool_call>",
                    "whole_foods_order(location=",
                    "\"downtown\")",
                    "</tool_call>",
                ],
                &[
                    "plan</think>I'll place both orders for you.<tool_",
                    "call>whole_foods_order(location=\"downtown\")</tool_call>",
                ],
            ];
            for pieces in cuts {
                let events = read_pieces(glm(), GLM_PROMPT, pieces, EngineFinish::Stop);
                let (contents, finish_last) = content_events(&events);
                assert_eq!(contents.concat(), answer, "{pieces:?}: {events:?}");
                assert_eq!(
                    contents.last().map(String::as_str),
                    Some(block),
                    "{pieces:?}: {events:?}"
                );
                assert!(finish_last, "{events:?}");
                assert!(events
                    .iter()
                    .all(|event| !matches!(event, Event::ToolCallStart { .. })));
                let choice = message(0, &events);
                assert_eq!(choice.message.content.as_deref(), Some(answer.as_str()));
                assert!(choice.message.tool_calls.is_none());
                assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
                assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
            }
            // The stream's end cuts the block: what came of it goes out after the text.
            let events = read_pieces(
                glm(),
                GLM_PROMPT,
                &[
                    "plan</think>Let me look:<tool_call>Read(file_path=\"x",
                    "yz",
                ],
                EngineFinish::Length,
            );
            assert_eq!(
                content_events(&events).0,
                vec!["Let me look:", "<tool_call>Read(file_path=\"xyz"],
                "{events:?}"
            );
            // Two blocks after the text, the template's newline between them: the slice from the
            // first opener to the last refused byte.
            let events = read_pieces(
                glm(),
                GLM_PROMPT,
                &[
                    "</think>Both:",
                    "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>",
                ],
                EngineFinish::Stop,
            );
            assert_eq!(
                content_events(&events).0,
                vec![
                    "Both:",
                    "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>"
                ],
                "{events:?}"
            );
            // A block held before the text, then the text, then a block: the first is debris
            // beside the text (the table's rule), the hold starts over behind the text, and the
            // second goes out alone, from its opener.
            let events = read_pieces(
                glm(),
                GLM_PROMPT,
                &[
                    "</think><tool_call>a(x=1)</tool_call>",
                    "Sorry.",
                    "<tool_call>b(y=2)</tool_call>",
                ],
                EngineFinish::Stop,
            );
            assert_eq!(
                content_events(&events).0,
                vec!["Sorry.", "<tool_call>b(y=2)</tool_call>"],
                "{events:?}"
            );
            // Other keyed tables, whose templates write a newline after the thought's close.
            let block = "<tool_call>get_weather(city=\"Paris\")</tool_call>";
            let pieces = ["\n<think>plan</think>\nI'll check.", block];
            let events = read_pieces(glm4(), GLM4_PROMPT, &pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert!(
                contents.concat().ends_with(&format!("I'll check.{block}")),
                "{events:?}"
            );
            assert_eq!(
                contents.last().map(String::as_str),
                Some(block),
                "{events:?}"
            );
            assert!(finish_last, "{events:?}");
            let pieces = ["plan</think>\nI'll check.", block, "\n"];
            let events = read_pieces(ling(), "<think>\n", &pieces, EngineFinish::Stop);
            let (contents, finish_last) = content_events(&events);
            assert!(
                contents.concat().ends_with(&format!("I'll check.{block}")),
                "{events:?}"
            );
            assert_eq!(
                contents.last().map(String::as_str),
                Some(block),
                "{events:?}"
            );
            assert!(finish_last, "{events:?}");
        });
        for (table, count) in [("glm", " 6"), ("glm4", " 1"), ("ling", " 1")] {
            let label = format!("table=\"{table}\"");
            let outputs = line_with(
                &scrape,
                "smg_symphony_outputs_total",
                &[&label, "outcome=\"malformed\""],
            )
            .expect("the outputs counter");
            assert!(outputs.ends_with(count), "{outputs}");
            assert!(
                line_with(
                    &scrape,
                    "smg_symphony_outputs_total",
                    &[&label, "outcome=\"empty\""]
                )
                .is_none(),
                "{scrape}"
            );
        }
    }

    /// A refused block that text or a call follows is debris, as the table says: nothing of it
    /// goes out when text comes after the block, whether or not text came before it, when a
    /// well-formed call follows it, before or after text, or when the block follows a call; a
    /// thought alone does not make the block the answer. Two blocks and what stands between them
    /// the same.
    #[test]
    fn a_refused_block_that_text_or_a_call_follows_stays_dropped() {
        use symphony::formats::glm;
        // What was held is let go when the answer comes after the block.
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &[
                "</think><tool_call>Read(file_path=\"x\")</tool_call>",
                "Sorry, no tool for that.",
            ],
            EngineFinish::Stop,
        );
        assert_eq!(
            content_events(&events).0,
            vec!["Sorry, no tool for that."],
            "{events:?}"
        );
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &[
                "</think><tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>",
                "Sorry, no tool for that.",
            ],
            EngineFinish::Stop,
        );
        assert_eq!(
            content_events(&events).0,
            vec!["Sorry, no tool for that."],
            "{events:?}"
        );
        // Text on both sides of the block: the text goes out, the block stays dropped.
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &[
                "</think>Let me look:<tool_call>Read(file_path=\"x\")</tool_call>",
                "Sorry, no tool for that.",
            ],
            EngineFinish::Stop,
        );
        assert_eq!(
            content_events(&events).0,
            vec!["Let me look:", "Sorry, no tool for that."],
            "{events:?}"
        );
        // A well-formed call after a block of prose: the call, and no content at all.
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &["</think><tool_call>modify text with tool call<tool_call>lookup<arg_key>q</arg_key><arg_value>1</arg_value></tool_call>"],
            EngineFinish::Stop,
        );
        assert_eq!(content_of(&events), "", "{events:?}");
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { name, .. } if name == "lookup")));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // A well-formed call after the text and the block: the text and the call, nothing of
        // the block.
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &["</think>Let me look:<tool_call>Read(file_path=\"x\")</tool_call><tool_call>lookup<arg_key>q</arg_key><arg_value>1</arg_value></tool_call>"],
            EngineFinish::Stop,
        );
        assert_eq!(content_of(&events), "Let me look:", "{events:?}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // A block after a call: debris beside the call.
        let events = read_pieces(
            glm(),
            GLM_PROMPT,
            &[
                "</think><tool_call>lookup<arg_key>q</arg_key><arg_value>1</arg_value></tool_call>",
                "<tool_call>Read(file_path=\"x\")</tool_call>",
            ],
            EngineFinish::Stop,
        );
        assert_eq!(content_of(&events), "", "{events:?}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_request_without_tools_is_read_by_every_table_and_its_calls_are_content() {
        // `json` and `xlam` read a bare JSON answer as a call: without tools the request asked
        // for none, so the table goes without its calls and the answer is the text it was, as
        // the legacy parsers leave it when they run no tool parser; a table with call markers
        // goes the same way.
        let json = names(Some("json"), None);
        let without = request(json!({}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &without, None, PROMPT).expect("assigned");
        assert_eq!(assigned.format.name(), "json");
        assert_eq!(assigned.format.call_syntax(), None);
        let declared = serde_json::to_value(tools()).expect("tools");
        let with = request(json!({"tools": declared}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &with, None, PROMPT).expect("assigned");
        assert!(assigned.format.call_syntax().is_some());
        let xlam = request(json!({"model": "Salesforce/Llama-xLAM-2-8b-fc-r"}));
        let by_registry = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &xlam,
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(by_registry.format.name(), "xlam");
        assert_eq!(by_registry.format.call_syntax(), None);
        for named in ["inkling", "llama_nemotron"] {
            let assigned = assign(
                ParserChoice::Symphony,
                &names(Some(named), None),
                &without,
                None,
                PROMPT,
            )
            .expect("assigned");
            assert_eq!(assigned.format.name(), named);
            assert_eq!(assigned.format.call_syntax(), None);
        }
    }

    #[test]
    fn explicit_names_win_over_the_registry() {
        // A table named for a model the registry does not know reads it; a legacy name reads
        // with its alias, or with no alias leaves a model the registry knows to the legacy
        // parsers; a table named for a model the registry reads with another wins, and replays
        // from its own opener.
        let unknown = request(json!({"model": "unknown"}));
        let ling = names(Some("ling"), Some("ling"));
        let assigned = assign(ParserChoice::Symphony, &ling, &unknown, None, PROMPT);
        assert_eq!(assigned.expect("assigned").format.name(), "ling");
        let aliased = names(Some("qwen"), Some("qwen3"));
        let assigned = assign(
            ParserChoice::Symphony,
            &aliased,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let legacy = names(Some("pythonic"), Some("deepseek_r1"));
        assert!(assign(
            ParserChoice::Symphony,
            &legacy,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        // A reasoning name alone that selects no table leaves the choice to the registry: a
        // known model takes its table and the table's tag, an unknown one the legacy parsers.
        let required = request(json!({"tool_choice": "required"}));
        let reasoning_only = names(None, Some("deepseek_r1"));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &required,
            &tools(),
            false
        )
        .is_some());
        let assigned = assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let unknown_required = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown_required,
            &tools(),
            false
        )
        .is_none());
        assert!(assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown,
            None,
            PROMPT
        )
        .is_none());
        // `passthrough` alone is the choice to leave the thought as written: the legacy
        // parsers keep a known model too.
        let passthrough_only = names(None, Some("passthrough"));
        assert!(assign(
            ParserChoice::Symphony,
            &passthrough_only,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let kimi = names(Some("kimi_k3"), None);
        let assigned = assign(
            ParserChoice::Symphony,
            &kimi,
            &request(json!({})),
            None,
            PROMPT,
        );
        let assigned = assigned.expect("assigned");
        assert_eq!(assigned.format.name(), "kimi_k3");
        assert_eq!(
            assigned.prompt, PROMPT,
            "no Kimi opener in a Qwen prompt: the whole replays"
        );
        // The registry's own table named for its model keeps the model's turn opener.
        let exaone = request(json!({"model": "LGAI-EXAONE/K-EXAONE-236B-A23B"}));
        let prompt = "<|user|>hi<|endofturn|><|assistant|>\n<think>\n";
        let qwen3 = names(Some("qwen3"), Some("qwen3"));
        let by_name = assign(ParserChoice::Symphony, &qwen3, &exaone, None, prompt);
        let by_registry = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &exaone,
            None,
            prompt,
        );
        assert_eq!(
            by_name.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        assert_eq!(
            by_registry.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        // Under legacy the names change nothing.
        assert!(assign(ParserChoice::Legacy, &ling, &unknown, None, PROMPT).is_none());
    }

    /// The table Symphony reads `request`'s model with, under the default names and no templates.
    fn table_of(request: &ChatCompletionRequest) -> Arc<Format> {
        reads(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request.model,
            &templates(),
        )
        .expect("read")
        .expect("a table")
    }

    /// The table assigned to `request`'s response, under the default names and no templates.
    fn assigned_format(request: &ChatCompletionRequest) -> Arc<Format> {
        assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            request,
            None,
            PROMPT,
        )
        .expect("assigned")
        .format
    }

    #[test]
    fn the_requests_asks_take_a_variant_of_the_table_and_do_not_switch_the_reader() {
        // A request that keeps the thought in the content, or asks for no calls, is read by the
        // same table without its thought or its calls; the constraint is the full table's.
        let declared = serde_json::to_value(tools()).expect("tools");
        let kept = request(
            json!({"tools": declared, "tool_choice": "required", "separate_reasoning": false}),
        );
        assert_eq!(
            *assigned_format(&kept),
            Arc::unwrap_or_clone(table_of(&kept)).without_reasoning()
        );
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &kept,
            &tools(),
            false
        )
        .is_some());
        let no_calls = request(json!({"tool_choice": "none"}));
        assert_eq!(
            *assigned_format(&no_calls),
            Arc::unwrap_or_clone(table_of(&no_calls)).without_calls()
        );
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &no_calls,
            &tools(),
            false
        )
        .is_none());
        let neither = request(json!({"separate_reasoning": false}));
        assert_eq!(
            *assigned_format(&neither),
            Arc::unwrap_or_clone(table_of(&neither))
                .without_calls()
                .without_reasoning()
        );
        let both = request(json!({"tools": declared}));
        assert_eq!(assigned_format(&both), table_of(&both));
    }

    #[test]
    fn the_calls_stay_wherever_the_legacy_path_runs_a_tool_parser() {
        // The legacy path scans a response for calls when the request declares tools, an empty
        // list included, or when the model's provider reads calls without any; the table keeps
        // its calls exactly then, so a call the conversation established comes back as a call
        // under either parser, and a `tool_choice` of `none` keeps them for the reader to drop.
        let retry = request(json!({"model": "MiniMax-M3", "messages": retry_messages()}));
        assert!(scans_for_tool_calls_for(
            &retry,
            ProviderProfile::for_model(&retry.model)
        ));
        assert_eq!(
            assigned_format(&retry),
            table_of(&retry),
            "the provider reads calls without tools"
        );
        let refused = request(json!({
            "model": "MiniMax-M3", "messages": retry_messages(), "tool_choice": "none"
        }));
        let forbidden = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &refused,
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(
            forbidden.format,
            table_of(&refused),
            "none keeps the calls the provider reads; the reader drops them"
        );
        assert_eq!(forbidden.forbids_calls, Forbids::Every);
        let declared_none = request(json!({"tools": []}));
        assert!(scans_for_tool_calls_for(
            &declared_none,
            ProviderProfile::for_model(&declared_none.model)
        ));
        assert_eq!(
            assigned_format(&declared_none),
            table_of(&declared_none),
            "an empty list is a declaration"
        );
        let undeclared = request(json!({"messages": retry_messages()}));
        assert!(!scans_for_tool_calls_for(
            &undeclared,
            ProviderProfile::for_model(&undeclared.model)
        ));
        assert_eq!(
            *assigned_format(&undeclared),
            Arc::unwrap_or_clone(table_of(&undeclared)).without_calls(),
            "a provider that reads calls only with tools"
        );
        // A Messages request: declared tools, an empty list included, keep the calls; none
        // declared drops them, as the legacy path runs its tool parser on `tools` alone.
        let messages = |extra: Value| {
            super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::for_messages(&message_request(extra), false),
                None,
                PROMPT,
                &templates(),
            )
            .expect("read")
            .expect("assigned")
            .format
        };
        assert!(messages(json!({"tools": []})).call_syntax().is_some());
        assert_eq!(messages(json!({})).call_syntax(), None);
    }

    /// A chat request's `json_object` or `json_schema` response format, and a Messages request's
    /// output format, reach the assignment as `json_answer`; a text format and a request that
    /// asks for none read as before.
    #[test]
    fn a_request_held_to_a_json_answer_assigns_a_reader_that_knows_it() {
        let chat = |extra: Value| {
            assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &request(extra),
                None,
                PROMPT,
            )
            .expect("assigned")
            .json_answer
        };
        assert!(chat(json!({"response_format": {"type": "json_object"}})));
        assert!(chat(json!({"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": {"type": "object"}},
        }})));
        assert!(!chat(json!({"response_format": {"type": "text"}})));
        assert!(!chat(json!({})));
        let messages = |extra: Value| {
            super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::for_messages(&message_request(extra), false),
                None,
                PROMPT,
                &templates(),
            )
            .expect("read")
            .expect("assigned")
            .json_answer
        };
        assert!(messages(json!({"output_config": {
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        }})));
        assert!(!messages(json!({})));
    }

    /// A chat request's `ignore_eos` reaches the assignment as `past_the_end`, so the reader's
    /// tally knows the output carries an unconstrained tail; a request without it, and a
    /// Messages request, which cannot ask it, read as before.
    #[test]
    fn a_request_that_ignores_the_end_assigns_a_reader_that_knows_it() {
        let forced = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({"ignore_eos": true})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert!(forced.past_the_end);
        assert_eq!(forced.format.name(), "qwen3");
        let plain = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert!(!plain.past_the_end);
        let messages = super::assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            Asks::for_messages(&message_request(json!({})), false),
            None,
            PROMPT,
            &templates(),
        )
        .expect("read")
        .expect("assigned");
        assert!(!messages.past_the_end);
    }

    #[test]
    fn a_known_model_is_assigned_its_table_and_the_prompts_replay() {
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(assignment.format.name(), "qwen3");
        assert_eq!(assignment.prompt, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn an_unknown_model_is_not_assigned_and_a_json_schema_constraint_is() {
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({"model": "unknown"})),
            None,
            PROMPT
        )
        .is_none());
        // A schema with no call forced is an answer: the table reads it without its calls.
        let schema = ("json_schema".to_string(), "{}".to_string());
        let answer = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            Some(&schema),
            PROMPT,
        )
        .expect("the table reads an answer under the schema");
        assert_eq!(
            *answer.format,
            (*table_of(&request(json!({})))).clone().without_calls()
        );
        assert_eq!(answer.forcing, None);
        let tag = ("structural_tag".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            Some(&tag),
            PROMPT
        )
        .is_some());
    }

    /// The two calls a JSON schema forces, as the request asks and as the model writes them
    /// after a thought the prompt opened: the function the request names, whose JSON is the
    /// arguments alone, and a required call, whose JSON is the list.
    fn forced_under_the_schema() -> [(Value, Option<&'static str>, ForcedCall, &'static str); 2] {
        [
            (
                json!({"type": "function", "function": {"name": "lookup"}}),
                Some("lookup"),
                ForcedCall::Arguments,
                r#"plan</think>{"q": 1}"#,
            ),
            (
                json!("required"),
                None,
                ForcedCall::List,
                r#"plan</think>[{"name": "lookup", "parameters": {"q": 1}}]"#,
            ),
        ]
    }

    /// The reasoning text among `events`.
    fn thought_of(events: &[Event<'_>]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// A call the gateway's JSON schema forces, on an engine that holds the output to the schema
    /// once the model's thought is over: the output thinks, closes the thought and then writes
    /// the JSON. The model's table reads it with its forced call, so the thought is reasoning and
    /// the call holds the JSON alone, for the function the request names and for a required
    /// call's list. No legacy parser reads it, and the output counts under the model's table.
    #[test]
    fn a_forced_call_under_the_schema_keeps_the_thought_the_models_table_reads() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        for (choice, forcing, _, output) in forced_under_the_schema() {
            let forced = request(json!({
                "tools": declared.clone(), "tool_choice": choice, "separate_reasoning": true,
            }));
            let assignment = assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &forced,
                Some(&schema),
                PROMPT,
            )
            .expect("the model's table reads the forced call");
            assert_eq!(assignment.table(), "qwen3");
            assert_eq!(assignment.forcing.as_deref(), forcing);
            let mut replayed = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
            let events = read_tail(&mut reader, output);
            assert_eq!(thought_of(&events), "plan", "{events:?}");
            assert_eq!(content_of(&events), "", "{events:?}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{events:?}"
            );
            assert_eq!(
                calls_of(&events),
                vec![("lookup".to_string(), json!({"q": 1}))],
                "{events:?}"
            );
        }
    }

    /// The JSON-schema pair the older registry writes for `forced`'s call over `tools`, the ones
    /// its choice calls, as request building passes it: a named function's parameters, or the
    /// list of a required call.
    fn schema_of(forced: &ChatCompletionRequest, tools: &[Tool]) -> (String, String) {
        let schema = tool_parser::ParserFactory::new()
            .registry()
            .generate_tool_constraint(
                None,
                tools,
                forced.tool_choice.as_ref().expect("a choice"),
                false,
            )
            .expect("a constraint")
            .expect("the older registry's schema")
            .to_tuple();
        assert_eq!(schema.0, "json_schema");
        schema
    }

    /// What the reader of `forced`'s call under `schema` reads of an output in `pieces`, each a
    /// step of its own, after the tests' prompt, for workers that start a grammar at `start`
    /// together: the thought, the content and the calls.
    fn forced_reading(
        forced: &ChatCompletionRequest,
        schema: &(String, String),
        tools: &[Tool],
        start: GrammarStart,
        pieces: &[&str],
    ) -> (String, String, Vec<(String, Value)>) {
        let events = forced_events(forced, schema, tools, start, pieces);
        (thought_of(&events), content_of(&events), calls_of(&events))
    }

    /// The events of that reading, the prompt's replay first.
    fn forced_events(
        forced: &ChatCompletionRequest,
        schema: &(String, String),
        tools: &[Tool],
        start: GrammarStart,
        pieces: &[&str],
    ) -> Vec<Event<'static>> {
        let assignment = assign_for(
            Reading::Table(Table::Qwen3.shared(), false),
            Asks::Chat(forced),
            Some(schema),
            PROMPT,
            ProviderProfile::for_model(&forced.model),
            || start,
        )
        .expect("the model's table reads the forced call");
        let mut replayed = Events::new();
        let mut reader =
            Reader::new(&assignment, tools.to_vec().into(), &mut replayed).expect("a reader");
        let mut events = replayed.into_owned().drain();
        let (last, earlier) = pieces.split_last().expect("a piece");
        for piece in earlier {
            let mut delta = Delta::default();
            let mut step = Events::new();
            reader
                .tail((*piece).to_string(), &mut delta, &mut step)
                .expect("read");
            events.extend(step.into_owned().drain());
        }
        events.extend(read_tail(&mut reader, last));
        events
    }

    /// The forced request of the tests' `lookup` tool under `choice`, the reasoning separated or
    /// kept in the content, with the schema pair it is sent with.
    fn forced_lookup(
        choice: &Value,
        separate_reasoning: bool,
    ) -> (ChatCompletionRequest, (String, String)) {
        let forced = request(json!({
            "tools": serde_json::to_value(tools()).expect("tools"), "tool_choice": choice,
            "separate_reasoning": separate_reasoning,
        }));
        let schema = schema_of(&forced, &tools());
        (forced, schema)
    }

    /// The same request on workers that do not say where they start a grammar: the output is the
    /// JSON alone, where the engine held it to the schema from the first token, or the thought,
    /// its close and the JSON, and its first bytes say which. The assignment keeps the prompt's
    /// replay and names the JSON's opener, and the reader replays nothing until the output
    /// decides: the opener and then a byte the JSON writes after it make the output the call
    /// from the opener, and any other output is read where the replay leaves the reader, the
    /// thought as reasoning, or as content where the request keeps it there, and then the call.
    #[test]
    fn a_forced_call_on_workers_that_do_not_say_starts_where_the_outputs_first_bytes_say() {
        let call = vec![("lookup".to_string(), json!({"q": 1}))];
        for (choice, forcing, _, thought_first) in forced_under_the_schema() {
            let (_, json_alone) = thought_first
                .split_once("</think>")
                .expect("a thought first");
            // The same JSON with whitespace after its opener, as a grammar lets a model write it.
            let spaced = format!("{}\n {}", &json_alone[..1], &json_alone[1..]);
            for separate_reasoning in [true, false] {
                let (forced, schema) = forced_lookup(&choice, separate_reasoning);
                let assignment = assign(
                    ParserChoice::Symphony,
                    &ModelNames::default(),
                    &forced,
                    Some(&schema),
                    PROMPT,
                )
                .expect("the model's table reads the forced call");
                assert_eq!(assignment.prompt, "<|im_start|>assistant\n<think>\n");
                let opener = if forcing.is_some() { b'{' } else { b'[' };
                assert_eq!(assignment.opener, Some(opener));
                let mut replayed = Events::new();
                Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
                assert!(
                    replayed.as_slice().is_empty(),
                    "nothing is replayed before the output decides"
                );
                let read = |start: GrammarStart, pieces: &[&str]| {
                    forced_reading(&forced, &schema, &tools(), start, pieces)
                };
                for (output, thought, known) in [
                    (json_alone, "", GrammarStart::FirstToken),
                    (spaced.as_str(), "", GrammarStart::FirstToken),
                    (thought_first, "plan", GrammarStart::AfterReasoning),
                ] {
                    // Whitespace decides nothing, and neither does the opener alone: the byte
                    // after them does, and what came before it is read with the rest as the
                    // start it names reads it, whether it came with that byte or in steps of
                    // its own before it.
                    let led = format!("\n {output}");
                    let (first, rest) = output.split_at(1);
                    for pieces in [
                        vec![output],
                        vec![led.as_str()],
                        vec!["\n ", output],
                        vec!["\n", " ", output],
                        vec![first, rest],
                        vec!["\n", first, rest],
                    ] {
                        let case = format!("separated {separate_reasoning}, {pieces:?}");
                        let said = read(GrammarStart::Unknown, &pieces);
                        assert_eq!(
                            said,
                            read(known, &pieces),
                            "{case}: the reading of the start the first bytes name"
                        );
                        let (reasoning, content) = if separate_reasoning {
                            (thought, "")
                        } else {
                            ("", thought)
                        };
                        assert_eq!(
                            (said.0.trim(), said.1.trim()),
                            (reasoning, content),
                            "{case}"
                        );
                        assert_eq!(said.2, call, "{case}");
                    }
                }
                // The opener, the whitespace after it and the rest, each a step of its own: the
                // two held steps are read with the third, which decides.
                let pieces = [&spaced[..1], &spaced[1..3], &spaced[3..]];
                let said = read(GrammarStart::Unknown, &pieces);
                assert_eq!(said, read(GrammarStart::FirstToken, &pieces), "{pieces:?}");
                assert_eq!(said.2, call, "{pieces:?}");
            }
        }
    }

    /// A thought that begins with the JSON's own opener, on workers that do not say: the byte
    /// after the opener is none the JSON writes there, so the output is read where the prompt's
    /// replay leaves the reader, as on workers that start a grammar once the thought is over:
    /// the thought, its first byte included, and then the call. That byte tells it whether it
    /// comes in the opener's step or in a later one, behind whitespace or not, and a second
    /// opener is such a byte, whatever follows it: the opener is looked for once.
    #[test]
    fn a_thought_that_begins_with_the_jsons_opener_is_read_after_the_prompts_replay() {
        let call = vec![("lookup".to_string(), json!({"q": 1}))];
        for (choice, forcing, _, thought_first) in forced_under_the_schema() {
            let (_, json_alone) = thought_first
                .split_once("</think>")
                .expect("a thought first");
            let thoughts: [&str; 6] = if forcing.is_some() {
                [
                    "{thinking} plan",
                    "{ plan",
                    "{\n1. plan",
                    "{{ plan",
                    "{[ plan",
                    r#"{{"draft": 1}} plan"#,
                ]
            } else {
                [
                    "[analysis] plan",
                    "[ plan",
                    "[\n1. plan",
                    "[[ plan",
                    "[\" plan",
                    r#"[[{"draft": 1}]] plan"#,
                ]
            };
            for separate_reasoning in [true, false] {
                let (forced, schema) = forced_lookup(&choice, separate_reasoning);
                for thought in thoughts {
                    let output = format!("{thought}</think>{json_alone}");
                    for cut in 0..4 {
                        let pieces = [&output[..cut], &output[cut..]];
                        let case = format!("separated {separate_reasoning}, {pieces:?}");
                        let said = forced_reading(
                            &forced,
                            &schema,
                            &tools(),
                            GrammarStart::Unknown,
                            &pieces,
                        );
                        assert_eq!(
                            said,
                            forced_reading(
                                &forced,
                                &schema,
                                &tools(),
                                GrammarStart::AfterReasoning,
                                &pieces
                            ),
                            "{case}: read as after the thought"
                        );
                        let (reasoning, content) = if separate_reasoning {
                            (thought, "")
                        } else {
                            ("", thought)
                        };
                        assert_eq!(
                            (said.0.trim(), said.1.trim()),
                            (reasoning, content),
                            "{case}"
                        );
                        assert_eq!(said.2, call, "{case}");
                    }
                }
            }
        }
    }

    /// The JSON alone on workers that do not say: after its opener comes a byte the JSON writes
    /// there, a member's key or the closing brace after an object's brace, a call's brace or the
    /// closing bracket after a list's bracket, behind whitespace or not. Every such output is
    /// read as on workers that hold the output to the schema from the first token, cut anywhere
    /// into two steps: none of it is taken for the thought the prompt opened.
    #[test]
    fn the_json_alone_starts_in_the_call_whatever_the_json_writes_after_its_opener() {
        for (choice, forcing, _, _) in forced_under_the_schema() {
            let outputs: [&str; 5] = if forcing.is_some() {
                [r#"{"q": 1}"#, "{ \"q\": 1}", "{}", "{ }", "{\n}"]
            } else {
                [
                    r#"[{"name": "lookup", "parameters": {"q": 1}}]"#,
                    "[ {\"name\": \"lookup\", \"parameters\": {\"q\": 1}}]",
                    "[]",
                    "[ ]",
                    "[\n]",
                ]
            };
            for separate_reasoning in [true, false] {
                let (forced, schema) = forced_lookup(&choice, separate_reasoning);
                for output in outputs {
                    for cut in 0..=output.len() {
                        let pieces = [&output[..cut], &output[cut..]];
                        let case = format!("separated {separate_reasoning}, {pieces:?}");
                        let said = forced_reading(
                            &forced,
                            &schema,
                            &tools(),
                            GrammarStart::Unknown,
                            &pieces,
                        );
                        assert_eq!(
                            said,
                            forced_reading(
                                &forced,
                                &schema,
                                &tools(),
                                GrammarStart::FirstToken,
                                &pieces
                            ),
                            "{case}: read as from the first token"
                        );
                        assert_eq!(said.0, "", "{case}: no thought");
                    }
                }
            }
        }
    }

    /// A named function whose arguments the schema does not hold to an object, on workers that
    /// do not say: its JSON alone may open with any byte, so no opener tells it from a thought.
    /// With the reasoning kept in the content the reader starts in the call, as on workers that
    /// hold the output from the first token: read after the replay, that JSON would be content
    /// and no call. With the reasoning separated the opener stays named, and a JSON that opens
    /// otherwise is read after the replay, as the thought the prompt opened. A schema whose root
    /// says `object` names the opener on both sides.
    #[test]
    fn a_named_call_not_held_to_an_object_starts_in_the_call_where_the_thought_is_kept() {
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let object = json!({"type": "object", "properties": {"q": {"type": "integer"}}});
        for (parameters, held_to_an_object, written) in [
            (Some(object), true, vec![json!({"q": 1}), json!({})]),
            // A function declared without parameters is sent the empty schema: any value.
            (None, false, vec![json!({}), json!("Paris"), json!(null)]),
            (Some(json!({"type": "string"})), false, vec![json!("Paris")]),
            (
                Some(json!({"type": "integer"})),
                false,
                vec![json!(7), json!(-7)],
            ),
            (
                Some(json!({"type": "array", "items": {"type": "string"}})),
                false,
                vec![json!(["Paris"]), json!([])],
            ),
            (
                Some(json!({"type": ["object", "null"]})),
                false,
                vec![json!(null), json!({"q": 1})],
            ),
            (
                Some(json!({"properties": {"q": {"type": "integer"}}})),
                false,
                vec![json!(true), json!({"q": 1})],
            ),
            (
                Some(json!({"anyOf": [{"type": "object"}, {"type": "string"}]})),
                false,
                vec![json!("Paris"), json!({})],
            ),
        ] {
            let function = match &parameters {
                Some(parameters) => json!({"name": "lookup", "parameters": parameters}),
                None => json!({"name": "lookup"}),
            };
            let declared = json!([{"type": "function", "function": function}]);
            let tools: Vec<Tool> = serde_json::from_value(declared.clone()).expect("the tool");
            for separate_reasoning in [true, false] {
                let forced = request(json!({
                    "tools": declared.clone(), "tool_choice": named.clone(),
                    "separate_reasoning": separate_reasoning,
                }));
                let schema = schema_of(&forced, &tools);
                let case = format!("{}, separated {separate_reasoning}", schema.1);
                let assignment = assign(
                    ParserChoice::Symphony,
                    &ModelNames::default(),
                    &forced,
                    Some(&schema),
                    PROMPT,
                )
                .expect("the model's table reads the forced call");
                if separate_reasoning || held_to_an_object {
                    assert_eq!(assignment.opener, Some(b'{'), "{case}");
                    assert_eq!(
                        assignment.prompt, "<|im_start|>assistant\n<think>\n",
                        "{case}"
                    );
                } else {
                    assert_eq!(assignment.opener, None, "{case}");
                    assert_eq!(
                        assignment.prompt, "",
                        "{case}: the reader starts in the call"
                    );
                }
                for arguments in &written {
                    let output = arguments.to_string();
                    let said =
                        forced_reading(&forced, &schema, &tools, GrammarStart::Unknown, &[&output]);
                    if separate_reasoning && !output.starts_with('{') {
                        // The stated limit: no byte tells this JSON from a thought, and it is
                        // read as on workers that start a grammar once the thought is over.
                        assert_eq!(
                            said,
                            forced_reading(
                                &forced,
                                &schema,
                                &tools,
                                GrammarStart::AfterReasoning,
                                &[&output]
                            ),
                            "{case}: {output}"
                        );
                        assert!(said.2.is_empty(), "{case}: {output}");
                    } else {
                        let call = vec![("lookup".to_string(), arguments.clone())];
                        assert_eq!(
                            said,
                            (String::new(), String::new(), call),
                            "{case}: {output}"
                        );
                    }
                }
            }
        }
    }

    /// The root of a schema's text holds the output to an object where it says so, and nowhere
    /// else: a root that names no type, a list of types or another type, a root that is no
    /// object of keywords, and a text that is no JSON all leave the first byte open.
    #[test]
    fn a_schemas_root_holds_the_output_to_an_object_only_where_its_type_says_object() {
        for schema in [
            r#"{"type":"object"}"#,
            r#"{"type":"object","properties":{"type":{"type":"string"}}}"#,
            r#"{"properties":{"q":{}},"type":"object","required":["q"]}"#,
            r#" { "$defs" : {"a": {"type": "string"}}, "type" : "object" } "#,
        ] {
            assert!(holds_to_an_object(schema), "{schema}");
        }
        for schema in [
            "{}",
            r#"{"type":"string"}"#,
            r#"{"type":"array","minItems":1}"#,
            r#"{"type":["object","null"]}"#,
            r#"{"properties":{"type":{"type":"object"}}}"#,
            r##"{"$ref":"#/$defs/a","$defs":{"a":{"type":"object"}}}"##,
            r#"{"anyOf":[{"type":"object"},{"type":"string"}]}"#,
            r#"{"type":null}"#,
            "true",
            r#""object""#,
            r#"[{"type":"object"}]"#,
            "",
            r#"{"type":"object""#,
        ] {
            assert!(!holds_to_an_object(schema), "{schema}");
        }
    }

    /// The steps held before the start is decided keep their tokens, each span at its place in
    /// the joined text: the opener's token and a token of whitespace, read as two steps, are
    /// held as two spans one after the other, and the step that decides feeds them to the engine
    /// ahead of its own, so the call holds every byte.
    #[test]
    fn the_steps_held_before_the_start_is_decided_keep_each_tokens_span_at_its_place() {
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let (forced, schema) = forced_lookup(&named, true);
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &forced,
            Some(&schema),
            PROMPT,
        )
        .expect("the model's table reads the forced call");
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(Refusing::default());
        let mut decoder = StopSequenceDecoder::new(
            tokenizer,
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        for held in ['{', ' '] {
            let mut delta = Delta::default();
            let mut step = Events::new();
            reader
                .read(&mut decoder, &[u32::from(held)], &mut delta, &mut step)
                .expect("read");
            assert!(step.as_slice().is_empty(), "{held:?} decides nothing");
        }
        let held = &reader.undecided.as_ref().expect("no start yet").held;
        assert_eq!(held.text, "{ ");
        assert_eq!(held.tokens, [u32::from('{'), u32::from(' ')]);
        let span = |token: char, start: usize| TokenSpan {
            token_id: u32::from(token),
            start,
            end: start + 1,
            continued: false,
        };
        assert_eq!(held.spans, [span('{', 0), span(' ', 1)]);
        let rest: Vec<u32> = r#""q": 1}"#.chars().map(u32::from).collect();
        let mut delta = Delta::default();
        let mut events = Events::new();
        reader
            .read(&mut decoder, &rest, &mut delta, &mut events)
            .expect("read");
        assert!(reader.undecided.is_none(), "the third step decides");
        reader
            .end(EngineFinish::Stop, who(), &mut events)
            .expect("ended");
        let events = events.into_owned().drain();
        assert_eq!(
            calls_of(&events),
            vec![("lookup".to_string(), json!({"q": 1}))],
            "{events:?}"
        );
        assert_eq!(thought_of(&events), "", "{events:?}");
    }

    /// An output that never decides its start, on workers that do not say: whitespace alone,
    /// nothing, or the opener with nothing after it, as an output cut there by its length. Its
    /// end starts the reader where the prompt's replay leaves it, and the output ends as it does
    /// for workers that start a grammar once the thought is over.
    #[test]
    fn a_forced_call_whose_output_never_decides_ends_after_the_prompts_replay() {
        for (choice, forcing, _, _) in forced_under_the_schema() {
            let (forced, schema) = forced_lookup(&choice, true);
            let read = |start: GrammarStart, output: &str| {
                let assignment = assign_for(
                    Reading::Table(Table::Qwen3.shared(), false),
                    Asks::Chat(&forced),
                    Some(&schema),
                    PROMPT,
                    ProviderProfile::for_model(&forced.model),
                    || start,
                )
                .expect("the model's table reads the forced call");
                let mut replayed = Events::new();
                let mut reader =
                    Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
                let mut events = replayed.into_owned().drain();
                events.extend(read_tail(&mut reader, output));
                format!("{events:?}")
            };
            let opener = if forcing.is_some() { "{" } else { "[" };
            for output in [
                String::new(),
                " \n".to_string(),
                opener.to_string(),
                format!("\n{opener}"),
                format!("{opener} \n"),
            ] {
                assert_eq!(
                    read(GrammarStart::Unknown, &output),
                    read(GrammarStart::AfterReasoning, &output),
                    "{output:?}"
                );
            }
        }
    }

    /// The output's first [`DECIDING_BYTES`] decide, wherever the steps cut them, on workers
    /// that do not say. Whitespace is the JSON's until a byte says otherwise: a run of it that
    /// fills those bytes, ahead of the opener or behind it, starts the reader in the call, so
    /// the JSON behind it is the call, and a thought behind it is read as workers that hold the
    /// output to the schema from the first token read it, the stated limit. One byte short of
    /// them the next byte still decides: the JSON there is the call and a thought is read after
    /// the prompt's replay.
    #[test]
    fn whitespace_that_fills_the_outputs_deciding_bytes_starts_the_reader_in_the_call() {
        // One call of the tool whose arguments, as written, are the JSON the model meant.
        let is_the_call = |calls: &[(String, String)]| {
            matches!(calls, [(name, arguments)] if name == "lookup"
                && serde_json::from_str::<Value>(arguments).ok() == Some(json!({"q": 1})))
        };
        for (choice, _, _, thought_first) in forced_under_the_schema() {
            let (_, json_alone) = thought_first
                .split_once("</think>")
                .expect("a thought first");
            let (opener, rest) = json_alone.split_at(1);
            let filled = " ".repeat(DECIDING_BYTES);
            let short = &filled[1..];
            // Behind the opener, itself the first of the deciding bytes.
            let (opened_filled, opened_short) = (&filled[1..], &filled[2..]);
            for separate_reasoning in [true, false] {
                let (forced, schema) = forced_lookup(&choice, separate_reasoning);
                let read = |start: GrammarStart, pieces: &[&str]| {
                    let events = forced_events(&forced, &schema, &tools(), start, pieces);
                    (
                        thought_of(&events),
                        content_of(&events),
                        calls_written(&events),
                    )
                };
                for (output, known, whole_call) in [
                    (
                        format!("{filled}{json_alone}"),
                        GrammarStart::FirstToken,
                        true,
                    ),
                    (
                        format!("{short}{json_alone}"),
                        GrammarStart::FirstToken,
                        true,
                    ),
                    (
                        format!("{opener}{opened_filled}{rest}"),
                        GrammarStart::FirstToken,
                        true,
                    ),
                    (
                        format!("{opener}{opened_short}{rest}"),
                        GrammarStart::FirstToken,
                        true,
                    ),
                    (
                        format!("{short}{thought_first}"),
                        GrammarStart::AfterReasoning,
                        true,
                    ),
                    (
                        format!("{opener}{opened_short}{thought_first}"),
                        GrammarStart::AfterReasoning,
                        true,
                    ),
                    // The stated limit: the thought behind whitespace that fills them.
                    (
                        format!("{filled}{thought_first}"),
                        GrammarStart::FirstToken,
                        false,
                    ),
                    (
                        format!("{opener}{opened_filled}{thought_first}"),
                        GrammarStart::FirstToken,
                        false,
                    ),
                ] {
                    let case = format!("separated {separate_reasoning}, {known:?}, {output:?}");
                    let said = read(known, &[output.as_str()]);
                    assert_eq!(is_the_call(&said.2), whole_call, "{case}: {said:?}");
                    let bytes: Vec<&str> = (0..output.len()).map(|at| &output[at..=at]).collect();
                    let (before, after) = output.split_at(DECIDING_BYTES);
                    let (short_of, from) = output.split_at(DECIDING_BYTES - 1);
                    let splits: [&[&str]; 4] = [
                        &[output.as_str()],
                        &bytes,
                        &[before, after],
                        &[short_of, from],
                    ];
                    for pieces in splits {
                        assert_eq!(
                            read(GrammarStart::Unknown, pieces),
                            said,
                            "{case}, in {} steps",
                            pieces.len()
                        );
                    }
                }
            }
        }
    }

    /// What the reader keeps while the output decides nothing stays short of the deciding
    /// bytes, however long the model writes whitespace, ahead of the opener or behind it: a
    /// byte a step, each held, until the byte that fills them starts the reader, after which
    /// nothing is held. A step that brings more than them at once is not held at all.
    #[test]
    fn the_reader_holds_less_than_the_deciding_bytes_before_it_starts() {
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let (forced, schema) = forced_lookup(&named, true);
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &forced,
            Some(&schema),
            PROMPT,
        )
        .expect("the model's table reads the forced call");
        let step = |reader: &mut Reader, text: &str| {
            let mut delta = Delta::default();
            let mut events = Events::new();
            reader
                .tail(text.to_string(), &mut delta, &mut events)
                .expect("read");
            events.as_slice().len()
        };
        for led in ["", "{"] {
            let mut replayed = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
            step(&mut reader, led);
            for held in led.len() + 1..DECIDING_BYTES {
                assert_eq!(
                    step(&mut reader, " "),
                    0,
                    "{led:?}, byte {held} decides nothing"
                );
                let kept = &reader.undecided.as_ref().expect("no start yet").held;
                assert_eq!(kept.text.len(), held, "{led:?}");
            }
            for _ in 0..4 * DECIDING_BYTES {
                step(&mut reader, " ");
                assert!(
                    reader.undecided.is_none(),
                    "{led:?}: nothing is held past them"
                );
            }
        }
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        step(&mut reader, &" ".repeat(4 * DECIDING_BYTES));
        assert!(reader.undecided.is_none(), "one long step");
    }

    /// The steps of `tokens` read through a decoder that stops at `stops`, then the output's
    /// end, on workers that start a grammar at `start` together: whether each step stopped, and
    /// the events.
    fn read_stopping(
        forced: &ChatCompletionRequest,
        schema: &(String, String),
        start: GrammarStart,
        stops: llm_tokenizer::StopSequenceConfig,
        steps: &[&[u32]],
    ) -> (Vec<bool>, Vec<Event<'static>>) {
        let assignment = assign_for(
            Reading::Table(Table::Qwen3.shared(), false),
            Asks::Chat(forced),
            Some(schema),
            PROMPT,
            ProviderProfile::for_model(&forced.model),
            || start,
        )
        .expect("the model's table reads the forced call");
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(Refusing::default());
        let mut decoder = StopSequenceDecoder::new(tokenizer, stops, false);
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        let mut events = replayed.into_owned().drain();
        let mut stopped = Vec::new();
        for tokens in steps {
            let mut delta = Delta::default();
            let mut out = Events::new();
            stopped.push(
                reader
                    .read(&mut decoder, tokens, &mut delta, &mut out)
                    .expect("read"),
            );
            events.extend(out.into_owned().drain());
        }
        let mut out = Events::new();
        reader
            .end(EngineFinish::Stop, who(), &mut out)
            .expect("ended");
        events.extend(out.into_owned().drain());
        (stopped, events)
    }

    /// A token whose bytes the stop decoder keeps back while the start is undecided (they may
    /// begin a stop sequence) is held as its span alone, and the step that decides feeds it to
    /// the engine ahead of its own: the output reads as on workers whose start is known, token
    /// for token.
    #[test]
    fn a_held_token_whose_bytes_the_decoder_kept_back_is_read_with_the_step_that_decides() {
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let (forced, schema) = forced_lookup(&named, true);
        let tokens = |text: &str| text.chars().map(u32::from).collect::<Vec<u32>>();
        let (kept_back, rest) = (tokens("p"), tokens(r#"lan</think>{"q": 1}"#));
        let read = |start: GrammarStart| {
            let stops = llm_tokenizer::StopSequenceConfig::default().with_stop_sequence("pq");
            read_stopping(&forced, &schema, start, stops, &[&kept_back, &rest])
        };
        // The thought, the calls and the tokens the engine counted for the thought.
        let said = |events: &[Event<'_>]| {
            let counted = finished(events).expect("a finish").reasoning_tokens;
            (thought_of(events), calls_of(events), counted)
        };
        let (stopped, events) = read(GrammarStart::Unknown);
        assert_eq!(stopped, [false, false]);
        let call = vec![("lookup".to_string(), json!({"q": 1}))];
        assert_eq!(
            said(&events),
            ("plan".to_string(), call, 4),
            "the held token counts with the thought's other three: {events:?}"
        );
        let (_, known) = read(GrammarStart::AfterReasoning);
        assert_eq!(said(&events), said(&known));
    }

    /// A stop the decoder meets while the start is undecided ends the output there: the step
    /// says so, as a step read after the start does, and the end reads what was held, as on
    /// workers that start a grammar once the thought is over.
    #[test]
    fn a_stop_met_before_the_start_is_decided_ends_the_output() {
        const STOP: u32 = 0x1f;
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let (forced, schema) = forced_lookup(&named, true);
        let read = |start: GrammarStart| {
            let stops = llm_tokenizer::StopSequenceConfig::default().with_stop_token(STOP);
            read_stopping(
                &forced,
                &schema,
                start,
                stops,
                &[&[u32::from(' ')], &[STOP, u32::from('x')]],
            )
        };
        let (stopped, events) = read(GrammarStart::Unknown);
        assert_eq!(
            stopped,
            [false, true],
            "the held step says the decoder stopped"
        );
        let (known_stopped, known) = read(GrammarStart::AfterReasoning);
        assert_eq!(stopped, known_stopped);
        assert_eq!(format!("{events:?}"), format!("{known:?}"));
    }

    /// A schema with no call forced is an answer: the table reads it without its calls, so the
    /// thought the prompt opened is reasoning and the JSON after its close is content, a call's
    /// marker inside it included.
    #[test]
    fn an_answer_under_the_schema_is_the_thought_and_then_content_with_no_call() {
        let schema = ("json_schema".to_string(), "{}".to_string());
        let answer = request(json!({
            "tools": serde_json::to_value(tools()).expect("tools"), "separate_reasoning": true,
        }));
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &answer,
            Some(&schema),
            PROMPT,
        )
        .expect("the model's table reads the answer");
        assert_eq!(assignment.forcing, None);
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        let events = read_tail(&mut reader, r#"plan</think>{"a": "<tool_call>"}"#);
        assert_eq!(thought_of(&events), "plan", "{events:?}");
        assert_eq!(content_of(&events), r#"{"a": "<tool_call>"}"#, "{events:?}");
        assert!(calls_of(&events).is_empty(), "{events:?}");
    }

    /// Every table of the registry, the call forced by name and as required: the assignment
    /// under the schema is the table's own forced variant, named by the table, with the forced
    /// function's name where the request names one. The crate reads a thought and then the JSON
    /// with that variant on every table, so the two together are every table under both choices
    /// (its case `every_table_reads_a_forced_call_after_each_way_into_its_thought`).
    #[test]
    fn every_table_is_assigned_its_forced_variant_under_the_schema() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        for table in Table::ALL {
            for (choice, forcing, written, _) in forced_under_the_schema() {
                let forced = request(json!({
                    "tools": declared.clone(), "tool_choice": choice, "separate_reasoning": true,
                }));
                let assignment = assign_for(
                    Reading::Table(table.shared(), false),
                    Asks::Chat(&forced),
                    Some(&schema),
                    PROMPT,
                    ProviderProfile::for_model(&forced.model),
                    || GrammarStart::Unknown,
                )
                .expect("a table reads");
                assert_eq!(
                    *assignment.format,
                    table.format().with_forced_call(written),
                    "{}",
                    table.name()
                );
                assert_eq!(assignment.forcing.as_deref(), forcing, "{}", table.name());
                assert_eq!(
                    assignment.table(),
                    telemetry::table_label(table.format().name()),
                    "{}",
                    table.name()
                );
            }
        }
    }

    /// The ids of the calls among `events`.
    fn ids_of(events: &[Event<'_>]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect()
    }

    /// Under a choice that forbids some calls a family that numbers its ids by the conversation
    /// numbers the calls returned: a call kept after a dropped one takes the id of the place it
    /// is returned in, so the first call of the next turn, counted on from the one call the
    /// history then holds, takes a different id.
    #[test]
    fn a_call_kept_after_a_dropped_one_takes_the_familys_id_of_its_place() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        let list = r#"[{"name": "undeclared", "parameters": {"q": 1}}, {"name": "lookup", "parameters": {"q": 2}}]"#;
        let first = json!([{"role": "user", "content": "hi"}]);
        for (model, table, ids) in [
            (
                "moonshotai/Kimi-K2-Instruct",
                Table::KimiK2,
                ["functions.lookup:0", "functions.lookup:1"],
            ),
            (
                "moonshotai/Kimi-K3",
                Table::KimiK3,
                ["lookup_0", "lookup_1"],
            ),
        ] {
            // The second turn's history holds the one call the first turn returned.
            let second = json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": ids[0], "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": ids[0], "content": "1"},
            ]);
            for (history, id) in [(first.clone(), ids[0]), (second, ids[1])] {
                let forced = request(json!({
                    "model": model, "messages": history,
                    "tools": declared.clone(), "tool_choice": "required",
                }));
                let assignment = assign_for(
                    Reading::Table(table.shared(), false),
                    Asks::Chat(&forced),
                    Some(&schema),
                    "",
                    ProviderProfile::for_model(&forced.model),
                    || GrammarStart::Unknown,
                )
                .expect("the table reads the forced call");
                let mut replayed = Events::new();
                let mut reader =
                    Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
                let events = read_tail(&mut reader, list);
                assert_eq!(ids_of(&events), [id], "{model}: {events:?}");
            }
        }
    }

    /// Two calls kept after a dropped one take the ids of the first two places, where the family
    /// numbers its ids by the conversation.
    #[test]
    fn two_calls_kept_after_a_dropped_one_take_the_familys_ids_of_their_places() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        let list = r#"[{"name": "undeclared", "parameters": {"q": 1}}, {"name": "lookup", "parameters": {"q": 2}}, {"name": "lookup", "parameters": {"q": 3}}]"#;
        for (model, table, ids) in [
            (
                "moonshotai/Kimi-K2-Instruct",
                Table::KimiK2,
                ["functions.lookup:0", "functions.lookup:1"],
            ),
            (
                "moonshotai/Kimi-K3",
                Table::KimiK3,
                ["lookup_0", "lookup_1"],
            ),
        ] {
            let forced = request(json!({
                "model": model, "messages": [{"role": "user", "content": "hi"}],
                "tools": declared.clone(), "tool_choice": "required",
            }));
            let assignment = assign_for(
                Reading::Table(table.shared(), false),
                Asks::Chat(&forced),
                Some(&schema),
                "",
                ProviderProfile::for_model(&forced.model),
                || GrammarStart::Unknown,
            )
            .expect("the table reads the forced call");
            let mut replayed = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
            let events = read_tail(&mut reader, list);
            assert_eq!(ids_of(&events), ids, "{model}: {events:?}");
        }
    }

    /// The `kimi` table reads in the place of the legacy parsers, which numbered a Kimi model's
    /// calls by the conversation: each call takes the family's id, counted on from the call the
    /// history holds, where the model wrote the first number again.
    #[test]
    fn the_calls_of_a_table_that_gives_way_take_the_familys_id() {
        let asked = request(json!({
            "model": "acme/kimi-tuned",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "1"},
            ],
            "tools": serde_json::to_value(tools()).expect("tools"),
        }));
        let assignment = assign_for(
            Reading::Table(Table::Kimi.shared(), false),
            Asks::Chat(&asked),
            None,
            "",
            ProviderProfile::for_model(&asked.model),
            || GrammarStart::Unknown,
        )
        .expect("the table reads the call");
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        let events = read_tail(
            &mut reader,
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|>{\"q\": 1}<|tool_call_end|><|tool_calls_section_end|>",
        );
        assert_eq!(ids_of(&events), ["functions.lookup:1"], "{events:?}");
    }

    /// A forced call's JSON has no header for the table to take an id from. On a family that
    /// numbers its ids by the conversation the reader gives the family's id, as the older forced
    /// arm does: the K2 line's `functions.{name}:{n}` and the K3 line's `{name}_{n}`, counted on
    /// from the calls the history holds, for the function the request names and for a required
    /// call's list. Every other model keeps the id the table mints.
    #[test]
    fn a_forced_call_takes_the_familys_id_where_the_family_numbers_its_ids() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        // Two calls in the history: the forced calls are the conversation's third and fourth.
        let history = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ]);
        let named = json!({"type": "function", "function": {"name": "lookup"}});
        let list = r#"[{"name": "lookup", "parameters": {"q": 1}}, {"name": "lookup", "parameters": {"q": 2}}]"#;
        for (model, table, ids) in [
            (
                "moonshotai/Kimi-K2-Instruct",
                Table::KimiK2,
                ["functions.lookup:2", "functions.lookup:3"],
            ),
            (
                "moonshotai/Kimi-K3",
                Table::KimiK3,
                ["lookup_2", "lookup_3"],
            ),
        ] {
            for (choice, output, expected) in [
                (named.clone(), r#"{"q": 1}"#, &ids[..1]),
                (json!("required"), list, &ids[..]),
            ] {
                let forced = request(json!({
                    "model": model, "messages": history.clone(),
                    "tools": declared.clone(), "tool_choice": choice,
                }));
                let assignment = assign_for(
                    Reading::Table(table.shared(), false),
                    Asks::Chat(&forced),
                    Some(&schema),
                    "",
                    ProviderProfile::for_model(&forced.model),
                    || GrammarStart::Unknown,
                )
                .expect("the table reads the forced call");
                let mut replayed = Events::new();
                let mut reader =
                    Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
                let events = read_tail(&mut reader, output);
                assert_eq!(ids_of(&events), expected, "{model}: {events:?}");
            }
        }
        // A model whose ids are minted: the table's id stands.
        let minted = request(json!({
            "messages": history, "tools": declared, "tool_choice": named,
        }));
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &minted,
            Some(&schema),
            PROMPT,
        )
        .expect("the table reads the forced call");
        assert_eq!(assignment.forced_ids, None);
        let mut replayed = Events::new();
        let mut reader = Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
        let ids = ids_of(&read_tail(&mut reader, r#"plan</think>{"q": 1}"#));
        assert!(
            matches!(ids.as_slice(), [id] if id.starts_with("call_")),
            "{ids:?}"
        );
    }

    /// The events of `output` read as two deltas cut at `cut`, then ended by a stop, owned.
    fn read_cut(reader: &mut Reader, output: &str, cut: usize) -> Vec<Event<'static>> {
        let mut all = Vec::new();
        for part in [&output[..cut], &output[cut..]] {
            let mut delta = Delta::default();
            let mut events = Events::new();
            reader
                .tail(part.to_string(), &mut delta, &mut events)
                .expect("read");
            all.extend(events.into_owned().drain());
        }
        let mut events = Events::new();
        reader
            .end(EngineFinish::Stop, who(), &mut events)
            .expect("ended");
        all.extend(events.into_owned().drain());
        all
    }

    /// The JSON a schema forces reads through every table as the older arm reads it: the same
    /// calls by name and by JSON value and no content, for the function the request names and
    /// for a required call's list, whole and cut anywhere into two deltas. The output is the
    /// JSON alone after a prompt that opens no thought, which every table reads from its
    /// arguments state: a table without a thought has no other, and a table with one was
    /// written none.
    #[test]
    fn every_table_reads_the_json_a_schema_forces_as_the_older_arm_does() {
        let declared = serde_json::to_value(tools()).expect("tools");
        let schema = ("json_schema".to_string(), "{}".to_string());
        for (choice, output) in [
            (
                json!({"type": "function", "function": {"name": "lookup"}}),
                r#"{"q": "Paris", "n": [1, {"deep": null}]}"#,
            ),
            (
                json!("required"),
                r#"[{"name": "lookup", "parameters": {"q": "Paris"}}, {"name": "lookup", "parameters": {}}]"#,
            ),
        ] {
            let forced = request(json!({"tools": declared.clone(), "tool_choice": choice}));
            let (older, rest) =
                parse_json_schema_response(output, forced.tool_choice.as_ref(), &forced.model, 0);
            assert_eq!(rest, "");
            let older: Vec<(String, Value)> = older
                .expect("the older arm reads the calls")
                .into_iter()
                .map(|call| {
                    let arguments = call.function.arguments.expect("arguments");
                    let arguments = serde_json::from_str(&arguments).expect("arguments as JSON");
                    (call.function.name, arguments)
                })
                .collect();
            for table in Table::ALL {
                let assignment = assign_for(
                    Reading::Table(table.shared(), false),
                    Asks::Chat(&forced),
                    Some(&schema),
                    "",
                    ProviderProfile::for_model(&forced.model),
                    || GrammarStart::Unknown,
                )
                .expect("a table reads");
                for cut in output
                    .char_indices()
                    .map(|(at, _)| at)
                    .chain([output.len()])
                {
                    let mut replayed = Events::new();
                    let mut reader =
                        Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
                    let events = read_cut(&mut reader, output, cut);
                    let case = format!("{} cut {cut}", table.name());
                    assert_eq!(calls_of(&events), older, "{case}: {events:?}");
                    assert_eq!(content_of(&events), "", "{case}: {events:?}");
                    assert_eq!(thought_of(&events), "", "{case}: {events:?}");
                }
            }
        }
    }

    #[test]
    fn the_engines_finish_reasons_map_and_the_rest_carry_through() {
        assert_eq!(engine_finish("stop"), EngineFinish::Stop);
        assert_eq!(engine_finish("length"), EngineFinish::Length);
        assert_eq!(engine_finish("abort"), EngineFinish::Abort);
        assert_eq!(
            engine_finish("failed"),
            EngineFinish::Other("failed".to_string())
        );
    }

    /// The assignment's `json_answer` reaches the reader's engine: a JSON answer that spells the
    /// table's message close inside a string comes back whole under it, with no malformed run,
    /// where the table alone reads the spelling as the close.
    #[test]
    fn a_json_answers_quoted_marker_spelling_is_the_answers_text_under_the_assignment() {
        let json = "{\"note\": \"the close is <|end_message|> here\", \"n\": 1}";
        let output = format!("<|content_text|>{json}<|end_message|>");
        let read = |json_answer: bool| {
            let assignment = Assignment {
                format: Arc::new(symphony::formats::inkling().without_calls()),
                prompt: "<|message_model|>".to_string(),
                forbids_calls: Forbids::Nothing,
                past_the_end: false,
                forcing: None,
                opener: None,
                forced_ids: None,
                json_answer,
                span: Span::none(),
                marked: Arc::default(),
            };
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
            let mut delta = Delta::default();
            reader
                .tail(output.clone(), &mut delta, &mut events)
                .expect("read");
            reader
                .end(EngineFinish::Stop, who(), &mut events)
                .expect("ended");
            let events = events.as_slice();
            let malformed = events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. }));
            (content_of(events), malformed)
        };
        assert_eq!(read(true), (json.to_string(), false));
        assert_ne!(
            read(false).0,
            json,
            "the table alone reads the spelling as the close"
        );
    }

    #[test]
    fn a_reader_gives_a_json_answer_bare_when_the_model_said_its_kind_again() {
        // The table whose messages say their kind after their opener, as a request without tools
        // reads it, over the two outputs an engine that starts a JSON answer's grammar after its
        // own reasoning parser lets the model write: the text marker twice, or a second thought
        // opened inside the answer's message; and over the answer's marker written inside the
        // thought, which starts the answer there, as the family's own parsers read it. The
        // client sees the bare JSON as content, the thoughts as reasoning, and the reader reads
        // no malformed run.
        let assignment = Assignment {
            format: Arc::new(symphony::formats::inkling().without_calls()),
            prompt: "<|message_model|>".to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let json = "{\"city\": \"Oslo\", \"country\": \"Norway\"}";
        let thought = "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>";
        for (output, reasoning) in [
            (
                format!("{thought}<|content_text|>{json}<|end_message|>"),
                "plan",
            ),
            (
                format!(
                    "{thought}<|content_thinking|>more<|end_message|><|message_model|>\
                     <|content_text|>{json}<|end_message|>"
                ),
                "planmore",
            ),
            (
                format!("<|content_thinking|>plan<|content_text|>{json}<|end_message|>"),
                "plan",
            ),
        ] {
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
            let mut delta = Delta::default();
            reader
                .tail(output.clone(), &mut delta, &mut events)
                .expect("read");
            reader
                .end(EngineFinish::Stop, who(), &mut events)
                .expect("ended");
            let events = events.as_slice();
            assert_eq!(content_of(events), json, "{events:?}");
            let thoughts: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Reasoning(text) => Some(&*text.text),
                    _ => None,
                })
                .collect();
            assert_eq!(thoughts, reasoning, "{events:?}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{events:?}"
            );
            let Some(Event::Finish { tool_calls, .. }) = events.last() else {
                panic!("a finish");
            };
            assert_eq!(*tool_calls, 0);
        }
    }

    #[test]
    fn a_string_value_a_calls_closing_marker_finds_open_is_delivered_whole_under_tool_calls() {
        // The keyed syntax as a live stream carries it: the model skips the value's closing tag,
        // writes the call's closing marker and then its stop token, which the engine hides. The
        // message carries one call whose arguments parse, the value every byte the model wrote,
        // under `tool_calls`; the output is counted malformed under the marker's reason.
        use symphony::{adapt::chat::message, formats::glm4};
        let tools: Vec<Tool> = serde_json::from_value(json!([{
            "type": "function", "function": {
                "name": "answer_string",
                "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}}
            }
        }]))
        .expect("tools");
        let assignment = Assignment {
            format: Arc::new(glm4()),
            prompt: GLM4_PROMPT.to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let pieces = [
            "\n<think></think>\n<tool_call>answer_string\n<arg_key>answer</arg_key>\n<arg_value>",
            "Logistic regression is a statistical method for a binary outcome",
            ".",
            "<|end_of_box|>",
            "\n",
            "</tool_call>",
        ];
        let mut all: Vec<Event<'static>> = Vec::new();
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, tools.into(), &mut events).expect("a reader");
        all.extend(events.into_owned().drain());
        for piece in pieces {
            let mut delta = Delta::default();
            let mut events = Events::new();
            reader
                .tail(piece.to_string(), &mut delta, &mut events)
                .expect("read");
            all.extend(events.into_owned().drain());
        }
        let mut events = Events::new();
        reader
            .end(EngineFinish::Stop, who(), &mut events)
            .expect("ended");
        all.extend(events.into_owned().drain());
        let choice = message(0, &all);
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        let calls = choice.message.tool_calls.expect("one call");
        assert_eq!(calls.len(), 1);
        let arguments: Value =
            serde_json::from_str(calls[0].function.arguments.as_deref().expect("arguments"))
                .expect("whole JSON");
        assert_eq!(
            arguments["answer"],
            "Logistic regression is a statistical method for a binary outcome.<|end_of_box|>\n"
        );
        assert!(all.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: symphony::MalformedReason::OpenAtTheMarker,
                ..
            }
        )));
        assert_eq!(
            telemetry::reason_label(symphony::MalformedReason::OpenAtTheMarker),
            "open_at_the_marker"
        );
    }

    #[test]
    fn a_reader_ends_once() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
        assert!(!reader.ended());
        reader
            .end(EngineFinish::Stop, who(), &mut events)
            .expect("ended");
        assert!(finished(events.as_slice()).is_some());
        assert!(reader.ended());
        let mut again = Events::new();
        reader
            .end(EngineFinish::Stop, who(), &mut again)
            .expect("ended");
        assert!(again.is_empty());
    }

    #[test]
    fn an_assignment_carries_the_span_current_where_it_is_made_and_a_failed_output_marks_it() {
        let lines = json_lines(|| {
            let span = request_span();
            assert!(span.id().is_some(), "the subscriber takes the span");
            let assignment = span.in_scope(|| {
                assignment(
                    Arc::new(qwen3(CallSyntax::Json)),
                    "Qwen/Qwen3-8B",
                    PROMPT,
                    Forbids::Nothing,
                    false,
                    None,
                    None,
                )
            });
            assert_eq!(assignment.span.id(), span.id());
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(Refusing::default());
            let mut decoder = StopSequenceDecoder::new(
                tokenizer,
                llm_tokenizer::StopSequenceConfig::default(),
                false,
            );
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            // One byte read whole, then the token no character carries: the output fails after
            // one byte, outside the span, as a stream's reading runs.
            let mut delta = Delta::default();
            reader
                .read(&mut decoder, &[u32::from('H')], &mut delta, &mut events)
                .expect("a byte");
            let mut failing = Delta::default();
            let mut after = Events::new();
            reader
                .read(&mut decoder, &[Refusing::REFUSED], &mut failing, &mut after)
                .expect_err("the token no character carries");
            let _entered = span.enter();
            tracing::info!(target: "smg::response", "finished processing request");
        });
        let span = &lines.last().expect("the access log line")["span"];
        assert_eq!(span["name"], "http_request");
        assert_eq!(span["symphony.outcome"], "failed");
        assert_eq!(span["symphony.table"], "qwen3");
        assert_eq!(span["symphony.bytes"], 1);
        assert!(span.get("symphony.finish").is_none(), "{span}");
    }

    #[test]
    fn one_assignment_marks_the_span_once_and_writes_its_line_inside_the_span() {
        // The streamed order: the access line goes out first, then two choices' readers built
        // from one assignment end outside the span, the first with an unterminated call and the
        // second with nothing. The first output's line is written inside the request span with
        // the fields just marked, and the second reading leaves the first's mark. The table is
        // one no other test of this binary reads, so its line buckets hold both lines, and each
        // line is found by its own fields: a suppressed line fails the test instead of passing it.
        let lines = json_lines(|| {
            let span = request_span();
            let assignment = span.in_scope(|| {
                assignment(
                    Arc::new(symphony::formats::qwen2_5()),
                    "Qwen/Qwen2.5-7B-Instruct",
                    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                    Forbids::Nothing,
                    false,
                    None,
                    None,
                )
            });
            span.in_scope(
                || tracing::info!(target: "smg::response", "finished processing request"),
            );
            let mut events = Events::new();
            let mut unterminated =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            let mut delta = Delta::default();
            unterminated
                .tail(
                    "Hi <tool_call>\n{\"name\": \"get_weather\"".to_string(),
                    &mut delta,
                    &mut events,
                )
                .expect("read");
            unterminated
                .end(EngineFinish::Length, who(), &mut events)
                .expect("ended");
            let mut events = Events::new();
            let mut nothing =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            nothing
                .end(EngineFinish::Stop, Who { choice: 1, ..who() }, &mut events)
                .expect("ended");
        });
        let position = |field: &str, value: &str| {
            lines
                .iter()
                .position(|line| line[field] == value)
                .unwrap_or_else(|| panic!("no line with {field} {value}: {lines:#?}"))
        };
        let access = position("message", "finished processing request");
        let first = position("outcome", "malformed");
        let second = position("outcome", "empty");
        assert!(access < first && first < second, "{lines:#?}");
        let span = &lines[access]["span"];
        assert_eq!(span["name"], "http_request");
        assert!(span.get("symphony.outcome").is_none(), "{span}");
        let line = &lines[first];
        assert_eq!(line["choice"], 0);
        assert_eq!(line["span"]["name"], "http_request");
        assert_eq!(line["span"]["symphony.outcome"], "malformed");
        assert_eq!(line["span"]["symphony.reason"], "unterminated_region");
        assert_eq!(line["span"]["symphony.choice"], 0);
        let line = &lines[second];
        assert_eq!(line["choice"], 1);
        assert_eq!(line["span"]["symphony.outcome"], "malformed", "{line}");
        assert_eq!(line["span"]["symphony.choice"], 0, "{line}");
    }

    #[test]
    fn a_token_the_decoder_refuses_fails_the_output_and_counts_it_once() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(Refusing::default());
        let mut decoder = StopSequenceDecoder::new(
            tokenizer,
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let scrape = rendered(|| {
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            let mut delta = Delta::default();
            let tokens = [u32::from('H'), Refusing::REFUSED];
            let error = reader
                .read(&mut decoder, &tokens, &mut delta, &mut events)
                .expect_err("the token no character carries");
            assert!(
                matches!(&error, SymphonyError::Read(why)
                    if why.contains("Stop decoder failed to process token 55296")),
                "{error}"
            );
            // The output is over: its end counts nothing more.
            assert!(reader.ended());
            let mut again = Events::new();
            reader
                .end(EngineFinish::Stop, who(), &mut again)
                .expect("ended");
            assert!(again.is_empty());
        });
        let outputs = line_with(
            &scrape,
            "smg_symphony_outputs_total",
            &["table=\"qwen3\"", "outcome=\"failed\""],
        )
        .expect("the failed count");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert_eq!(
            scrape.matches("smg_symphony_outputs_total{").count(),
            1,
            "{scrape}"
        );
    }

    #[tokio::test]
    async fn hf_reads_with_the_response_templates_table_and_refuses_a_model_without_one() {
        let templates = template::fixtures::templates_with(
            "Qwen/Qwen3-8B",
            template::fixtures::qwen3_5_tokenizer(),
        )
        .await;
        let hf = names(Some("hf"), Some(" HF "));
        let read = |names: &ModelNames<'_>, request: &ChatCompletionRequest| {
            reads(ParserChoice::Symphony, names, &request.model, &templates)
        };
        // The template's table reads the model the registry knows with another table, named
        // on both sides or on either alone.
        for names in [&hf, &names(Some("hf"), None), &names(None, Some("hf"))] {
            let format = read(names, &request(json!({})))
                .expect("read")
                .expect("a table");
            assert_eq!(format.name(), "response_template");
        }
        assert!(
            reads(ParserChoice::Legacy, &hf, "Qwen/Qwen3-8B", &templates)
                .expect("read")
                .is_none()
        );
        // What the request asks of the channels takes a variant of the templates' table.
        let kept = request(json!({"separate_reasoning": false}));
        let assigned = super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&kept),
            None,
            PROMPT,
            &templates,
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(
            *assigned.format,
            Arc::unwrap_or_clone(read(&hf, &kept).expect("read").expect("a table"))
                .without_calls()
                .without_reasoning()
        );
        // A model with no table of its own is refused with the reason, in both decisions.
        let unknown = request(json!({"model": "unknown"}));
        assert_eq!(
            read(&hf, &unknown).expect_err("refused"),
            SymphonyError::Template("no tokenizer is registered for model 'unknown'".to_string())
        );
        assert!(super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&unknown),
            None,
            PROMPT,
            &templates
        )
        .is_err());
        let assigned = super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&request(json!({}))),
            None,
            PROMPT,
            &templates,
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(assigned.format.name(), "response_template");
        assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
    }

    /// Limited to one call, the reader keeps the first call's events and
    /// counts one call; unlimited, both calls come through.
    #[test]
    fn a_reader_limited_to_one_call_drops_the_calls_after_the_first() {
        const TWO_CALLS: &str = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>\
                                 <tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2}}\n</tool_call>";
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let calls = |first_only: bool| {
            let mut events = Events::new();
            let mut reader = Reader::new(&assignment, tools().into(), &mut events)
                .expect("a reader")
                .first_call_only(first_only);
            let events = read_tail(&mut reader, TWO_CALLS);
            let starts = events
                .iter()
                .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                .count();
            let Some(Event::Finish { tool_calls, .. }) = events.last() else {
                panic!("no finish event");
            };
            (starts, *tool_calls)
        };
        assert_eq!(calls(false), (2, 2));
        assert_eq!(calls(true), (1, 1));
    }

    /// The unary reads feed the output and then the decoder's last text into one buffer, where
    /// content at the output's end and the flushed text join one event: the tally counts the
    /// joined bytes, so a malformed run the end reports after them stands at the byte it stands
    /// at when the output comes in one step.
    #[test]
    fn the_tally_counts_the_bytes_a_later_step_merged_into_an_earlier_event() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let first_run_at = |steps: &[&str]| -> (Option<usize>, usize) {
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            let mut deltas: Vec<Delta> = steps.iter().map(|_| Delta::default()).collect();
            for (text, delta) in steps.iter().zip(deltas.iter_mut()) {
                reader
                    .tail((*text).to_string(), delta, &mut events)
                    .expect("read");
            }
            reader
                .end(EngineFinish::Stop, who(), &mut events)
                .expect("ended");
            // Where the first malformed event stands, by the events' own bytes.
            let before = events
                .as_slice()
                .iter()
                .take_while(|event| !matches!(event, Event::Malformed { .. }))
                .map(telemetry::bytes_of)
                .sum();
            (reader.tally.first_run().map(|first| first.at), before)
        };
        let (content, rest) = (
            "Hello",
            " there<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": ",
        );
        let whole = format!("{content}{rest}");
        let (in_one_step, by_the_events) = first_run_at(&[&whole]);
        assert_eq!(in_one_step, Some(by_the_events), "the one-step reading");
        assert!(
            by_the_events > content.len(),
            "the cut call stands after the content"
        );
        assert_eq!(
            first_run_at(&[content, rest]),
            (in_one_step, by_the_events),
            "the two-step reading, its second step's content joined to the first's"
        );
    }

    #[test]
    fn a_bare_answer_the_json_table_reads_as_a_call_comes_back_as_its_text() {
        let assignment = Assignment {
            format: Arc::new(symphony::formats::json().without_calls()),
            prompt: String::new(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
        let answer = "{\"answer\": 42}";
        let events = read_tail(&mut reader, answer);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(text, answer);
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
    }

    /// `tool_choice: none` over declared tools: the table keeps its calls, the call the model
    /// writes anyway is read whole, and the reader drops it, as the engine's own server reads
    /// and withholds such a call; the thought and the text are what they are under `auto`, the
    /// finish is the engine's `stop`, and the call is counted per table as forbidden.
    #[test]
    fn a_call_the_request_forbade_is_dropped_whole_and_counted() {
        use symphony::{formats::qwen3, CallSyntax, FinishReason};
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
            forbids_calls: Forbids::Every,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        };
        let mut events = Vec::new();
        let scrape = rendered(|| {
            let mut out = Events::new();
            let mut reader = Reader::new(&assignment, tools().into(), &mut out).expect("a reader");
            events = read_tail(&mut reader, QWEN3_OUTPUT);
        });
        assert!(
            events.iter().all(|event| !matches!(
                event,
                Event::ToolCallStart { .. }
                    | Event::ToolCallArguments { .. }
                    | Event::ToolCallEnd { .. }
            )),
            "no call is delivered: {events:?}"
        );
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        assert_eq!(content_of(&events).trim(), "Hello");
        let dropped: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::ForbiddenCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert!(
            dropped.contains("lookup"),
            "the call's bytes are dropped text, not content: {dropped:?}"
        );
        assert!(
            matches!(
                events.last(),
                Some(Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 0,
                    ..
                })
            ),
            "{events:?}"
        );
        let forbidden = line_with(&scrape, telemetry::CALLS_FORBIDDEN, &["table=\"qwen3\""])
            .expect("the forbidden counter");
        assert!(forbidden.ends_with(" 1"), "{forbidden}");
        // The same output under `auto` is a call: the ask, not the table, decides.
        let allowed = Assignment {
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
            ..assignment
        };
        let mut out = Events::new();
        let mut reader = Reader::new(&allowed, tools().into(), &mut out).expect("a reader");
        let events = read_tail(&mut reader, QWEN3_OUTPUT);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { name, .. } if name == "lookup")));
    }

    #[test]
    fn a_reader_over_a_tables_variant_gives_the_calls_or_the_thought_as_content() {
        // The table itself: the thought apart, the call a call.
        let events = read_with(qwen3(CallSyntax::Json));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert_eq!(content_of(&events).trim(), "Hello");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 1);
        // Without its calls: the call and its markers are content, the finish counts no call.
        let events = read_with(qwen3(CallSyntax::Json).without_calls());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        let content = content_of(&events);
        assert!(
            content
                .trim_start()
                .starts_with("Hello <tool_call>\n{\"name\": \"lookup\""),
            "{content:?}"
        );
        assert!(content.ends_with("</tool_call>"), "{content:?}");
        assert!(!content.contains("<think>"), "{content:?}");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
        // Without its thought: the thought and its markers are content, the call stays a call.
        let events = read_with(qwen3(CallSyntax::Json).without_reasoning());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Reasoning(_) | Event::ReasoningStart)));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let content = content_of(&events);
        assert!(content.starts_with("<think>plan</think>"), "{content:?}");
        assert!(content.contains("Hello"), "{content:?}");
        assert!(!content.contains("<tool_call>"), "{content:?}");
        let Some(Event::Finish {
            reasoning_tokens, ..
        }) = events.last()
        else {
            panic!("a finish");
        };
        assert_eq!(*reasoning_tokens, 0);
    }

    /// A request Symphony was configured for that the legacy parsers read is counted once, under
    /// the reason the seam names and the table `none`; a table's assignment, under a structural
    /// tag or the older registry's JSON schema alike, and the gateway's own choice of the legacy
    /// parsers, count nothing.
    #[test]
    fn a_request_the_legacy_parsers_read_under_symphony_is_counted_once_by_reason() {
        let unknown = request(json!({"model": "acme/ministral-3-14b-instruct-w4a16"}));
        let known = request(json!({}));
        let counted = |names: &ModelNames<'_>,
                       request: &ChatCompletionRequest,
                       constraint: Option<&(String, String)>| {
            let scrape = rendered(|| {
                assert!(
                    assign(ParserChoice::Symphony, names, request, constraint, PROMPT).is_none()
                );
            });
            let lines: Vec<&str> = scrape
                .lines()
                .filter(|line| line.starts_with(telemetry::FALLBACK))
                .collect();
            assert_eq!(lines.len(), 1, "{scrape}");
            assert!(lines[0].ends_with(" 1"), "{}", lines[0]);
            lines[0].to_string()
        };
        let labelled = |line: &str, reason: &str, table: &str| {
            assert!(line.contains(&format!("reason=\"{reason}\"")), "{line}");
            assert!(line.contains(&format!("table=\"{table}\"")), "{line}");
        };
        labelled(
            &counted(&ModelNames::default(), &unknown, None),
            "no_table",
            "none",
        );
        labelled(
            &counted(&names(None, Some("deepseek_r1")), &unknown, None),
            "reasoning_name_only",
            "none",
        );
        labelled(
            &counted(&names(Some("pythonic"), Some("deepseek_r1")), &known, None),
            "legacy_tool_name",
            "none",
        );
        labelled(
            &counted(&names(None, Some("passthrough")), &known, None),
            "passthrough",
            "none",
        );

        // The shape a table without a tag for a forced call hands over: the seam's constraint is
        // nothing, the older registry writes its JSON schema for the named tool, and request
        // building passes it here. The table reads the output with its forced call, so nothing
        // is counted: no legacy parser reads it.
        let forced = request(json!({
            "tools": serde_json::to_value(tools()).expect("tools"),
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }));
        let schema = tool_parser::ParserFactory::new()
            .registry()
            .generate_tool_constraint(
                None,
                &tools(),
                forced.tool_choice.as_ref().expect("a choice"),
                false,
            )
            .expect("a constraint")
            .expect("the older registry's schema")
            .to_tuple();
        assert_eq!(schema.0, "json_schema");
        let scrape = rendered(|| {
            let assigned = assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &forced,
                Some(&schema),
                PROMPT,
            )
            .expect("the table reads the forced call");
            assert_eq!(assigned.table(), "qwen3");
            assert_eq!(assigned.forcing.as_deref(), Some("lookup"));
        });
        assert!(!scrape.contains(telemetry::FALLBACK), "{scrape}");
        // A message the same way.
        let message = message_request(json!({}));
        let scrape = rendered(|| {
            let assigned = super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::for_messages(&message, false),
                Some(&schema),
                PROMPT,
                &templates(),
            )
            .expect("no hf");
            assert!(assigned.is_some());
        });
        assert!(!scrape.contains(telemetry::FALLBACK), "{scrape}");

        // Nothing counted where a table reads, under a structural tag too, nor under `legacy`.
        let tag = ("structural_tag".to_string(), "{}".to_string());
        let scrape = rendered(|| {
            let default = ModelNames::default();
            assert!(assign(ParserChoice::Symphony, &default, &known, None, PROMPT).is_some());
            assert!(assign(
                ParserChoice::Symphony,
                &default,
                &forced,
                Some(&tag),
                PROMPT
            )
            .is_some());
            assert!(assign(ParserChoice::Legacy, &default, &unknown, None, PROMPT).is_none());
            assert!(assign(
                ParserChoice::Legacy,
                &default,
                &known,
                Some(&schema),
                PROMPT
            )
            .is_none());
        });
        assert!(!scrape.contains(telemetry::FALLBACK), "{scrape}");
    }

    /// The events of `output` read whole by `format` after `prompt`, with the request's one
    /// declared tool, under a recorder of their own: the events and the scrape.
    fn read_scraped(format: Format, prompt: &str, output: &str) -> (Vec<Event<'static>>, String) {
        let mut events = Vec::new();
        let scrape = rendered(|| {
            let assignment = Assignment {
                format: Arc::new(format),
                prompt: prompt.to_string(),
                forbids_calls: Forbids::Nothing,
                past_the_end: false,
                forcing: None,
                opener: None,
                forced_ids: None,
                json_answer: false,
                span: Span::none(),
                marked: Arc::default(),
            };
            let mut replayed = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut replayed).expect("a reader");
            events.extend(replayed.into_owned().drain());
            events.extend(read_tail(&mut reader, output));
        });
        (events, scrape)
    }

    /// Where the `UnknownTool` events stand among `events`, each with the name of the call that
    /// starts right after it.
    fn undeclared_calls(events: &[Event<'_>]) -> Vec<(usize, String)> {
        events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                matches!(
                    event,
                    Event::Malformed {
                        why: symphony::MalformedReason::UnknownTool,
                        ..
                    }
                )
            })
            .map(|(at, event)| {
                assert_eq!(telemetry::bytes_of(event), 0, "an event of no bytes");
                match &events[at + 1] {
                    Event::ToolCallStart { name, .. } => (at, name.clone()),
                    other => panic!("a call start right after the event, not {other:?}"),
                }
            })
            .collect()
    }

    /// The calls among `events`: each name with its arguments joined, as the events spell them.
    fn calls_written(events: &[Event<'_>]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
                Event::ToolCallArguments { index, json, .. } => {
                    calls[*index as usize].1.push_str(json);
                }
                _ => {}
            }
        }
        calls
    }

    /// The calls among `events`: each name with its arguments joined.
    fn calls_of(events: &[Event<'_>]) -> Vec<(String, Value)> {
        calls_written(events)
            .into_iter()
            .map(|(name, arguments)| {
                let arguments = serde_json::from_str(&arguments).expect("arguments as JSON");
                (name, arguments)
            })
            .collect()
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_call_to_a_function_the_request_did_not_declare_goes_out_as_written_and_is_counted() {
        // The request declares `lookup`; the model calls `lookup_v2`. The engine announces it
        // with its event of no bytes right before the call's start, and the call's events are
        // the ones a declared call gets, the name as written. The output counts once under
        // `unknown_tool`, the call under its reason, and one line names the table and the
        // function; nothing on the wire says so.
        let output = QWEN3_OUTPUT.replace("\"lookup\"", "\"lookup_v2\"");
        let (events, scrape) =
            read_scraped(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", &output);
        let undeclared = undeclared_calls(&events);
        assert_eq!(undeclared.len(), 1, "{events:?}");
        assert_eq!(undeclared[0].1, "lookup_v2");
        assert_eq!(
            calls_of(&events),
            vec![("lookup_v2".to_string(), json!({"q": 1}))]
        );
        assert_eq!(content_of(&events).trim(), "Hello");
        assert!(
            matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
            "{events:?}"
        );
        let outputs = line_with(
            &scrape,
            telemetry::OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"unknown_tool\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        let runs = line_with(
            &scrape,
            telemetry::MALFORMED,
            &["table=\"qwen3\"", "reason=\"unknown_tool\""],
        )
        .expect("the reason counter");
        assert!(runs.ends_with(" 1"), "{runs}");
        assert!(logs_contain(
            "Symphony read a call to a function the request did not declare"
        ));
        assert!(logs_contain("table=\"qwen3\""));
        assert!(logs_contain("tool=\"lookup_v2\""));
        assert!(logs_contain("undeclared_calls=1"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_declared_and_an_undeclared_call_in_one_output_announce_the_undeclared_one_alone() {
        // Two keyed blocks back to back, the second to a function the request did not declare:
        // one event, right before the second call's start; both calls go out, in order, with
        // their arguments; the output counts once, with two calls and one undeclared.
        let output = concat!(
            "plan</think>Sure.",
            "<tool_call>lookup<arg_key>q</arg_key><arg_value>1</arg_value></tool_call>",
            "<tool_call>lookup_v2<arg_key>q</arg_key><arg_value>2</arg_value></tool_call>"
        );
        let (events, scrape) =
            read_scraped(symphony::formats::glm(), "<|assistant|>\n<think>", output);
        let undeclared = undeclared_calls(&events);
        assert_eq!(undeclared.len(), 1, "{events:?}");
        assert_eq!(undeclared[0].1, "lookup_v2");
        assert!(
            matches!(
                &events[undeclared[0].0 + 1],
                Event::ToolCallStart { index: 1, .. }
            ),
            "{events:?}"
        );
        assert_eq!(
            calls_of(&events),
            vec![
                ("lookup".to_string(), json!({"q": 1})),
                ("lookup_v2".to_string(), json!({"q": 2})),
            ]
        );
        assert!(
            matches!(events.last(), Some(Event::Finish { tool_calls: 2, .. })),
            "{events:?}"
        );
        let outputs = line_with(
            &scrape,
            telemetry::OUTPUTS,
            &["table=\"glm\"", "outcome=\"unknown_tool\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert!(logs_contain("table=\"glm\""));
        assert!(logs_contain("tool=\"lookup_v2\""));
        assert!(logs_contain("undeclared_calls=1"));
        assert!(logs_contain("tool_calls=2"));
    }

    #[test]
    fn a_forced_call_reads_as_before_the_name_being_the_requests() {
        // `tool_choice` naming a function: the table reads the output under its structural tag,
        // whose grammar holds the name to the declared one; the reader adds no rule of its own,
        // so the forced call reads clean, as it did.
        let declared = serde_json::to_value(tools()).expect("tools");
        let forced = request(json!({
            "tools": declared,
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }));
        let tag = ("structural_tag".to_string(), "{}".to_string());
        let assigned = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &forced,
            Some(&tag),
            "<|im_start|>assistant\n",
        )
        .expect("assigned");
        assert_eq!(assigned.format.name(), "qwen3");
        let (events, scrape) = read_scraped(
            Arc::unwrap_or_clone(assigned.format),
            "<|im_start|>assistant\n",
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>",
        );
        assert!(undeclared_calls(&events).is_empty(), "{events:?}");
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{events:?}"
        );
        assert_eq!(
            calls_of(&events),
            vec![("lookup".to_string(), json!({"q": 1}))]
        );
        let outputs = line_with(
            &scrape,
            telemetry::OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"clean\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
    }
}

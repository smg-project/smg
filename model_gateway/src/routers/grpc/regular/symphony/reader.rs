//! The reader: Symphony's assignment to a response and the reader the drivers run over the
//! stop decoder's output, one per choice.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use llm_tokenizer::stop::StopSequenceDecoder;
use openai_protocol::{common::Tool, profile::ProviderProfile};
use parking_lot::RwLock;
use symphony::{
    Declared, DropReason, Engine, EngineFinish, Event, Events, Format, Input, Parser as _, Text,
    TokenSpan,
};
use tracing::debug;

use super::{
    constraint::Asks,
    names::{Fallback, Reading},
    telemetry::{self, Tally, Who},
    SymphonyError,
};
#[cfg(test)]
use super::{names::reading, template::Templates};
use crate::routers::grpc::utils::isolation;
#[cfg(test)]
use crate::{config::ParserChoice, routers::grpc::utils::ModelNames};

/// Symphony's assignment to a response: the table that reads it, as the request's asks make it
/// ([`assign_for`]), and the part of the rendered prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    /// The table, shared with every reader of the response and every request it reads for.
    pub format: Arc<Format>,
    pub prompt: String,
    /// The request forbade the calls the table reads (`tool_choice: none`,
    /// [`Asks::forbids_calls_for`]): the reader drops every call, as the engine's own server reads
    /// such a call and withholds it.
    pub forbids_calls: bool,
    /// The request asked for generation past the model's own end of output (`ignore_eos`,
    /// [`Asks::past_the_end`]): the output carries an unconstrained tail after the answer, which
    /// the reader reads and delivers like the rest, and a marker of the table kept as text in it
    /// is no failure of the table's reading ([`Tally::new`]).
    pub past_the_end: bool,
}

impl Assignment {
    /// The `table` label of the format, for the counters and the lines.
    pub(crate) fn table(&self) -> &'static str {
        telemetry::table_label(self.format.name())
    }
}

/// The assignment of `format` to a response, with the part of `prompt` the engine replays.
pub(super) fn assignment(
    format: Arc<Format>,
    model: &str,
    prompt: &str,
    forbids_calls: bool,
    past_the_end: bool,
) -> Assignment {
    debug!(model, table = format.name(), "Symphony reads the response");
    let prompt = format.replay(prompt).to_string();
    Assignment {
        format,
        prompt,
        forbids_calls,
        past_the_end,
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
    // A table nobody else holds (one built for this request) is made over in
    // place, as before; a shared table's variant is kept, so the next request
    // that asks the same of it reads with the one copy.
    if Arc::strong_count(&format) == 1 {
        return Arc::new(variant_of(
            Arc::unwrap_or_clone(format),
            no_calls,
            no_reasoning,
        ));
    }
    let key = (Arc::as_ptr(&format) as usize, no_calls, no_reasoning);
    if let Some(kept) = ASKED_VARIANTS.read().get(&key) {
        return Arc::clone(&kept.variant);
    }
    let variant = Arc::new(variant_of((*format).clone(), no_calls, no_reasoning));
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

/// `format` as [`asked`] makes it: without its calls, its thought, or both.
fn variant_of(mut format: Format, no_calls: bool, no_reasoning: bool) -> Format {
    if no_calls {
        format = format.without_calls();
    }
    if no_reasoning {
        format = format.without_reasoning();
    }
    format
}

/// The variants [`asked`] made of the shared tables, by the table's address
/// and the ask. A chat request without tools asks for the table without its
/// calls, which was rebuilt from the shared table on every such request (a
/// clone and the walks of `without_calls`: 19 allocation calls per request,
/// 2026-10-21).
static ASKED_VARIANTS: LazyLock<RwLock<AskedVariants>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// The kept variants by the shared table's address and the ask (no calls, no
/// reasoning).
type AskedVariants = HashMap<(usize, bool, bool), AskedVariant>;

/// How many variants are kept before the map starts over: the registry's
/// groups, the named tables and the templates' tables come to a few hundred,
/// each with up to three asks; a gateway serving every one of them with every
/// ask stays under this, and a stream of tables built per worker cannot grow
/// the map without bound.
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

/// [`assign_for`] with the table decision read here ([`reading`]) and the provider profile derived
/// from the model, as the tests ask it.
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
    ))
}

/// Symphony's assignment to a request, if a table reads the response (`read`, the table decision:
/// preparation reads it once per request, [`reading`](super::names::reading), and request building
/// passes it, `ProcessingState::reading`) and the constraint is not a JSON schema: the older
/// path's forced call, whose output has no markers for a table to read, which the older registry
/// writes where the table has no tag. The table goes without its calls where the legacy path would
/// run no tool parser, without its thought where it would run no reasoning parser
/// ([`Asks::no_calls_for`], [`Asks::no_reasoning`], [`asked`]). A call the conversation
/// established without a tool inventory comes back as a call under either parser. `None` where
/// the legacy parsers read the response, which the response's header then says; a request
/// Symphony was configured for is counted once here, by the reason
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
) -> Option<Assignment> {
    let format = match read {
        Reading::Table(format, _) => format,
        Reading::Legacy => return None,
        Reading::Fallback(why) => {
            telemetry::record_fallback(why, None);
            return None;
        }
    };
    if constraint.is_some_and(|(kind, _)| kind == "json_schema") {
        telemetry::record_fallback(
            Fallback::ForcedCallWithoutTag,
            Some(telemetry::table_label(format.name())),
        );
        return None;
    }
    let format = asked(format, asks.no_calls_for(provider), asks.no_reasoning());
    Some(assignment(
        format,
        asks.model(),
        prompt,
        asks.forbids_calls_for(provider),
        asks.past_the_end(),
    ))
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
    /// Drop the events of every call: the request forbade them (`tool_choice: none`), the table
    /// read them whole, and the engine's own server delivers nothing of such a call
    /// ([`Assignment::forbids_calls`]).
    forbids_calls: bool,
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
}

impl Reader {
    /// A reader for one choice, seeded with the assignment's prompt; the events the prompt's
    /// replay gives (a thought the prompt opens) go into `out`, before the first delta's. `tools`
    /// are the request's as its response spec holds them: every choice's reader shares them,
    /// none copies a schema.
    pub(crate) fn new(
        assignment: &Assignment,
        tools: Arc<[Tool]>,
        out: &mut Events<'_>,
    ) -> Result<Self, SymphonyError> {
        let mut reader = Self {
            table: assignment.table(),
            engine: Engine::shared(Arc::clone(&assignment.format), Declared::shared(tools)),
            tally: Tally::new(assignment.past_the_end),
            last_token: None,
            read: 0,
            ended: false,
            first_call_only: false,
            forbids_calls: assignment.forbids_calls,
            shown: false,
            unshown: String::new(),
            called: false,
            refused: String::new(),
            framing: Vec::new(),
            framing_text: String::new(),
            trap: isolation::test_trap(),
        };
        // The prompt's events carry no text of the prompt (a thought it opens or closes), so they
        // move from the replay's own buffer into the caller's one by one, whatever delta that one
        // borrows; nothing is collected on the way.
        let mut replayed = Events::new();
        reader.feed(
            Input::Prompt {
                token_ids: &[],
                text: &assignment.prompt,
            },
            &mut replayed,
        )?;
        for event in replayed.consume() {
            out.push(event.into_owned());
        }
        Ok(reader)
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
    /// output is recorded here, under `who`: its counters, and the line when it was not clean
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
        self.tally.report(self.table, who);
        Ok(())
    }

    /// Whether the output ended.
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    /// The error of an output the reader could not read whole, which the request fails with:
    /// the decoder refused a token or the text it held, or the table refused the output
    /// ([`Self::feed`]). The output is recorded as failed for the table and is over: its end
    /// counts nothing more.
    pub(crate) fn failed(&mut self, why: String) -> SymphonyError {
        self.ended = true;
        telemetry::record_failed(self.table);
        SymphonyError::Read(why)
    }

    /// Feeds `input` to the engine, its events into `out` after the ones already there: the
    /// reader's one way in. The events this step added are tallied, and limited to the first
    /// call where the request asks. An output the engine refuses is recorded as failed for the
    /// table, which the error names.
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
        // Where the step's first event starts in the output: the held runs below are placed by it.
        let start = self.tally.at();
        for event in fresh.iter() {
            if self.forbids_calls {
                self.tally.see_forbidden(event);
            } else {
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
                        self.unshown.push_str(&text.text);
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
        if self.forbids_calls {
            symphony::event::forbidden_calls(fresh);
        } else if self.first_call_only {
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
        self.framing_text.push_str(text);
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
            self.refused.push_str(&self.framing_text[run.text]);
        }
        self.clear_framing();
        self.refused.push_str(text);
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
mod tests;

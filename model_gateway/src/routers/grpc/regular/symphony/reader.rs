//! The reader: Symphony's assignment to a response and the reader the drivers run over the
//! stop decoder's output, one per choice.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use llm_tokenizer::stop::StopSequenceDecoder;
use openai_protocol::common::Tool;
use parking_lot::RwLock;
use symphony::{
    Declared, DropReason, Engine, EngineFinish, Event, Events, Format, Input, Parser as _, Text,
    TokenSpan,
};
use tracing::debug;

use super::{
    constraint::Asks,
    names::{reading, Fallback, Reading},
    telemetry::{self, Tally, Who},
    template::Templates,
    SymphonyError,
};
use crate::{
    config::ParserChoice,
    routers::grpc::utils::{isolation, ModelNames},
};

/// Symphony's assignment to a response: the table that reads it, as the request's asks make it
/// ([`assign`]), and the part of the rendered prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    /// The table, shared with every reader of the response and every request it reads for.
    pub format: Arc<Format>,
    pub prompt: String,
    /// The request forbade the calls the table reads (`tool_choice: none`,
    /// [`Asks::forbids_calls`]): the reader drops every call, as the engine's own server reads
    /// such a call and withholds it.
    pub forbids_calls: bool,
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
) -> Assignment {
    debug!(model, table = format.name(), "Symphony reads the response");
    let prompt = format.replay(prompt).to_string();
    Assignment {
        format,
        prompt,
        forbids_calls,
    }
}

/// The table as the request's asks make it: without its calls when the legacy path would run no
/// tool parser (the response is not scanned for calls:
/// [`scans_for_tool_calls`](crate::routers::grpc::spec::scans_for_tool_calls)), so a table that reads calls from the output's first byte (`json`,
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

/// Symphony's assignment to a request, if a table reads the response ([`reading`]) and the
/// constraint is not a JSON schema: the older path's forced call, whose output has no markers for
/// a table to read, which the older registry writes where the table has no tag. The table goes
/// without its calls where the legacy path would run no tool parser, without its thought where it
/// would run no reasoning parser ([`Asks::no_calls`], [`Asks::no_reasoning`], [`asked`]). A call
/// the conversation established without a tool inventory comes back as a call under either
/// parser. `None` where the legacy parsers read the response, which the response's header then
/// says; a request Symphony was configured for is counted once here, by the reason
/// ([`telemetry::record_fallback`]): request building asks once per request, and nothing else
/// decides the reader.
pub(crate) fn assign(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, SymphonyError> {
    let format = match reading(parser, names, asks.model(), templates)? {
        Reading::Table(format, _) => format,
        Reading::Legacy => return Ok(None),
        Reading::Fallback(why) => {
            telemetry::record_fallback(why, None);
            return Ok(None);
        }
    };
    if constraint.is_some_and(|(kind, _)| kind == "json_schema") {
        telemetry::record_fallback(
            Fallback::ForcedCallWithoutTag,
            Some(telemetry::table_label(format.name())),
        );
        return Ok(None);
    }
    let format = asked(format, asks.no_calls(), asks.no_reasoning());
    Ok(Some(assignment(
        format,
        asks.model(),
        prompt,
        asks.forbids_calls(),
    )))
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
            tally: Tally::default(),
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
            trap: isolation::test_trap(),
        };
        // The prompt's events carry no text of the prompt (a thought it opens or closes), so they
        // move from the replay's own buffer into the caller's, whatever delta that one borrows.
        let mut replayed = Events::new();
        reader.feed(
            Input::Prompt {
                token_ids: &[],
                text: &assignment.prompt,
            },
            &mut replayed,
        )?;
        for event in replayed.into_owned() {
            out.push(event);
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
                        self.framing.clear();
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
        match self.framing.last_mut() {
            Some(last) if !marker && !last.marker && last.at + last.text.len() == at => {
                last.text.push_str(text);
            }
            _ => self.framing.push(Framing {
                text: text.to_string(),
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
            self.refused.push_str(&run.text);
        }
        self.framing.clear();
        self.refused.push_str(text);
    }

    /// The output went on past what was held, with text or a call: the held runs are debris
    /// beside it, as the table says. After text the hold starts over, for the blocks that may end
    /// the answer; after a call nothing more is held (`called`).
    fn drop_held(&mut self) {
        self.refused.clear();
        self.framing.clear();
    }
}

/// A run the table dropped, held until what follows says whose it is: a refused block's opener
/// and the bytes between two refused blocks go out with the blocks; the rest is the template's.
struct Framing {
    text: String,
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
        let shared = symphony::formats::Table::Qwen3.shared();
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
    use crate::routers::grpc::regular::symphony::testing::{assign, constraint, *};

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
            forbids_calls: false,
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
    /// the client as a call: its arguments close whole without the member that crossed the
    /// bound, so a strict reader takes them, and the output counts as malformed under
    /// `invalid_arguments`, once.
    #[test]
    fn a_call_nested_past_the_bound_reaches_the_client_whole_and_counts_as_malformed() {
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
            let arguments: Value =
                serde_json::from_str(calls[0].function.arguments.as_deref().expect("arguments"))
                    .expect("arguments a strict reader takes");
            // The arguments object and the containers below it, as many as the reader takes:
            // the deepest closes empty, the member that would have crossed the bound unwritten.
            let mut depth = 0;
            let mut value = &arguments;
            while let Some(inner) = value.get("d") {
                depth += 1;
                value = inner;
            }
            assert_eq!(depth, PartialJson::DEFAULT_MAX_DEPTH - 1, "{arguments}");
            assert_eq!(value, &serde_json::json!({}), "{arguments}");
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

    #[test]
    fn the_requests_asks_take_a_variant_of_the_table_and_do_not_switch_the_reader() {
        // A request that keeps the thought in the content, or asks for no calls, is read by the
        // same table without its thought or its calls; the constraint is the full table's.
        let table = |request: &ChatCompletionRequest| {
            reads(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &request.model,
                &templates(),
            )
            .expect("read")
            .expect("a table")
        };
        let assigned = |request: &ChatCompletionRequest| {
            assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                request,
                None,
                PROMPT,
            )
            .expect("assigned")
            .format
        };
        let declared = serde_json::to_value(tools()).expect("tools");
        let kept = request(
            json!({"tools": declared, "tool_choice": "required", "separate_reasoning": false}),
        );
        assert_eq!(
            *assigned(&kept),
            Arc::unwrap_or_clone(table(&kept)).without_reasoning()
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
            *assigned(&no_calls),
            Arc::unwrap_or_clone(table(&no_calls)).without_calls()
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
            *assigned(&neither),
            Arc::unwrap_or_clone(table(&neither))
                .without_calls()
                .without_reasoning()
        );
        let both = request(json!({"tools": declared}));
        assert_eq!(assigned(&both), table(&both));
    }

    #[test]
    fn the_calls_stay_wherever_the_legacy_path_runs_a_tool_parser() {
        // The legacy path scans a response for calls when the request declares tools, an empty
        // list included, or when the model's provider reads calls without any; the table keeps
        // its calls exactly then, so a call the conversation established comes back as a call
        // under either parser, and a `tool_choice` of `none` keeps them for the reader to drop.
        let table = |request: &ChatCompletionRequest| {
            reads(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &request.model,
                &templates(),
            )
            .expect("read")
            .expect("a table")
        };
        let assigned = |request: &ChatCompletionRequest| {
            assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                request,
                None,
                PROMPT,
            )
            .expect("assigned")
            .format
        };
        let retry = request(json!({"model": "MiniMax-M3", "messages": retry_messages()}));
        assert!(scans_for_tool_calls(&retry));
        assert_eq!(
            assigned(&retry),
            table(&retry),
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
            table(&refused),
            "none keeps the calls the provider reads; the reader drops them"
        );
        assert!(forbidden.forbids_calls);
        let declared_none = request(json!({"tools": []}));
        assert!(scans_for_tool_calls(&declared_none));
        assert_eq!(
            assigned(&declared_none),
            table(&declared_none),
            "an empty list is a declaration"
        );
        let undeclared = request(json!({"messages": retry_messages()}));
        assert!(!scans_for_tool_calls(&undeclared));
        assert_eq!(
            *assigned(&undeclared),
            Arc::unwrap_or_clone(table(&undeclared)).without_calls(),
            "a provider that reads calls only with tools"
        );
        // A Messages request: declared tools, an empty list included, keep the calls; none
        // declared drops them, as the legacy path runs its tool parser on `tools` alone.
        let messages = |extra: Value| {
            super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Messages {
                    request: &message_request(extra),
                    continues: false,
                },
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
    fn an_unknown_model_and_a_json_schema_constraint_are_not_assigned() {
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({"model": "unknown"})),
            None,
            PROMPT
        )
        .is_none());
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            Some(&schema),
            PROMPT
        )
        .is_none());
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
            forbids_calls: false,
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
            forbids_calls: false,
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
            forbids_calls: false,
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
    fn a_token_the_decoder_refuses_fails_the_output_and_counts_it_once() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
            forbids_calls: false,
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
            forbids_calls: false,
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
            forbids_calls: false,
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
            forbids_calls: false,
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
            forbids_calls: true,
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
            forbids_calls: false,
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
    /// the reason the seam names and the table that had no tag for it, or `none`; a table's
    /// assignment, and the gateway's own choice of the legacy parsers, count nothing.
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
        // nothing, the older registry writes its JSON schema for the named tool, request building
        // passes it here, and the table that reads the model is the one counted.
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
        labelled(
            &counted(&ModelNames::default(), &forced, Some(&schema)),
            "forced_call_without_tag",
            "qwen3",
        );
        // A message the same way.
        let message = message_request(json!({}));
        let scrape = rendered(|| {
            let assigned = super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Messages {
                    request: &message,
                    continues: false,
                },
                Some(&schema),
                PROMPT,
                &templates(),
            )
            .expect("no hf");
            assert!(assigned.is_none());
        });
        labelled(
            line_with(&scrape, telemetry::FALLBACK, &[]).expect("counted"),
            "forced_call_without_tag",
            "qwen3",
        );

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
                forbids_calls: false,
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

    /// The calls among `events`: each name with its arguments joined.
    fn calls_of(events: &[Event<'_>]) -> Vec<(String, Value)> {
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

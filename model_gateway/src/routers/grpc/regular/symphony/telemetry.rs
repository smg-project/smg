//! What the reader records about an output: the counters per table, and one sampled line per
//! output with a malformed run, written under the request span (the log line carries the span's
//! request id; the trace exporter's own filter does not take this target).
//!
//! The classes, from the crate's events (`symphony::Event`):
//!
//! - `failed`: output the table could not read at all: the engine refused (`ParseError`); the
//!   request answers with an error;
//! - `panicked`: output whose reading panicked in the gateway (the reader, or the step that
//!   carried its events), caught at the request's boundary
//!   (`routers::grpc::utils::isolation`): the request answers with an error under
//!   `reader_panicked`, and the line that names the request is the isolation's own, at ERROR,
//!   with the panic's message sanitized of the bytes it quotes; a defect of the reader, never
//!   the output's doing, so it is told apart from `failed`;
//! - `malformed`: a call region the table read but gave up on, or a marker of the table where no
//!   row leads, kept as content text (`Event::Malformed`), by its reason; the bytes reach the
//!   client as content, or, where the syntax's block is never prose, are dropped behind an event of
//!   no bytes (`Dropped { MalformedCall }`) and sized here as that event's; nothing is lost in the
//!   events, and nothing says so to the client. An output that made no call and ended in such
//!   dropped blocks (a call written as Python where the name goes, after the thought or after
//!   the answer's text) is the one case where the reader releases dropped bytes as content
//!   before the finish (`Reader::end`): the output's own slice from the first of those blocks'
//!   opener to the last refused byte, so a client reads what the model wrote instead of an empty
//!   answer or a text that ends where the call should be; the output still counts as
//!   `malformed`, and its line says how many bytes went out;
//! - `dropped`: bytes the table treats as the template's wrapping (`Event::Dropped`): correct,
//!   expected on every output, not counted here;
//! - `empty`: an output with no text, no reasoning and no call: only bytes the table drops
//!   (whitespace before any text, the template's markers and separators) or no bytes at all.
//!   The reader releases the dropped whitespace of such an output as one content event before
//!   the finish (`Reader::end`): the chat stream carries it as the answer, as the engine's own
//!   server streams it, and a whole response's adapter applies its own rule for whitespace-only
//!   content; the output is counted as its own outcome either way, and written like a malformed
//!   output, on a bucket of its own, with the engine's finish, the dropped bytes by class and the
//!   bytes released, so a model that answers with whitespace under a forced length, or a stream
//!   that ended with nothing, is seen;
//! - `unknown_tool`: an output the table read whole in which a call names a function the request
//!   did not declare (the engine's `UnknownTool` event of no bytes right before the call's
//!   start): the call goes out as the model named it, nothing says so to the client, and the
//!   output is counted once under this outcome rather than as `malformed`, so the malformed rate
//!   stays the table's reading and this one the model's naming; each such call still counts
//!   under `reason="unknown_tool"`. An output with a refused run as well is `malformed`, with
//!   `unknown_tool` among its reasons. The line such an output writes, from a bucket of its own,
//!   names the first undeclared function: the one place a byte of model output is logged, an
//!   identifier where a declared name was expected, cut to [`NAME_SHOWN`] bytes with its whole
//!   length beside it, so an operator tells a shortened name from a misspelt one;
//! - `forgiven`: a repair that keeps the answer's shape. Of those, the calls after the first
//!   dropped under `parallel_tool_calls: false` are counted (`suppressed_calls`); the refined
//!   finish, the empty arguments of a call without an `arguments` member and the chunking rule's
//!   stated exception are the crate's contract and are not.
//!
//! And, before any output, what did not reach a table at all:
//!
//! - `fallback`: a request Symphony was configured for (the gateway's flag, or the request's
//!   header) that the legacy parsers read instead of a table, by the reason the seam names
//!   ([`Fallback`]): no table for the model, a reasoning name alone, a legacy tool-call name,
//!   `passthrough`, or a table without a tag for the request's forced call; counted once per
//!   request where the assignment is decided (`assign`), under the table that had no tag, or
//!   `none`. The response's `x-smg-parser` header says `legacy` for the same request; the
//!   counter says how often, and why, across a deployment.
//!
//! And, after an output was read, what the response could not carry:
//!
//! - `refused`: a whole Messages response whose `tool_use` block cannot carry the arguments the
//!   model wrote (JSON no `input` value holds): the request answers with an error, the output
//!   keeps whatever count the read gave it (`clean`; `unknown_tool` or `malformed` when an
//!   undeclared call or a malformed run came beside the refused call), and the refusal is counted
//!   once by its reason under the table ([`record_refused`]): this series is the fold's, a subset
//!   of no outcome, so a sum over the outcomes still reads the table's reading.
//!
//! Cost: a clean output records one counter increment through a key built once per table (the
//! `metrics` macros build a key, with a `Vec` of labels, on every call), no allocation. A
//! malformed one adds a histogram observation, the per-reason counters, and, when the table's
//! token bucket allows it, one `warn` line: the only allocations. One counted `unknown_tool` adds
//! the per-reason counters, the first undeclared name cut to [`NAME_SHOWN`] bytes (one small
//! allocation, made when the call starts) and, from a bucket of its own, one `warn` line; no
//! histogram observation, the table having read it whole. A run's text is model output and is
//! never logged, at any level: `at` and `bytes` locate it in the response for a reader with
//! access to it; the undeclared name on the `unknown_tool` line is the one exception, an
//! identifier where a declared name was expected. Suppressed calls are counted and write no line.
//! A request the legacy parsers read under Symphony records one counter increment, through a key
//! built once per reason and table, and nothing where a table reads. A refused message records
//! one, through the table's key built once.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        OnceLock,
    },
    time::Instant,
};

use metrics::{Key, Label, Level, Metadata};
use parking_lot::Mutex;
use symphony::{formats::Table, DropReason, Event, FinishReason, MalformedReason};
use tracing::warn;

use super::names::Fallback;

/// Outputs read to their end, by `table` and `outcome` (`clean`, `malformed`, `failed`,
/// `empty`, `unknown_tool`, `panicked`).
pub(crate) const OUTPUTS: &str = "smg_symphony_outputs_total";
/// Malformed runs, by `table` and `reason`.
pub(crate) const MALFORMED: &str = "smg_symphony_malformed_total";
/// The bytes of the malformed runs of one output, by `table`; observed for an output counted
/// `malformed`.
pub(crate) const MALFORMED_BYTES: &str = "smg_symphony_malformed_bytes";
/// Calls after the first dropped under `parallel_tool_calls: false`, by `table`.
pub(crate) const CALLS_SUPPRESSED: &str = "smg_symphony_calls_suppressed_total";
/// Requests Symphony was configured for that the legacy parsers read, by `reason` and `table`
/// ([`record_fallback`]).
pub(crate) const FALLBACK: &str = "smg_symphony_fallback_total";
/// Whole Messages responses refused after the output was read (counted under its own outcome),
/// by `table` and `reason` ([`record_refused`]): `tool_use_input`, a `tool_use` block that cannot
/// carry the arguments the model wrote.
pub(crate) const REFUSED: &str = "smg_symphony_refused_total";
/// The buckets of [`MALFORMED_BYTES`]: a marker's worth to a whole answer.
pub(crate) const MALFORMED_BYTES_BUCKETS: [f64; 9] = [
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];

/// The format a response template compiles into carries this name (no registry table).
pub(crate) const TEMPLATE: &str = "response_template";
/// The `table` label of a format that is neither a registry table's nor the template's: a format
/// built by hand.
const OTHER_TABLE: &str = "other";
/// The `table` label of a request no table reads ([`FALLBACK`]).
const NO_TABLE: &str = "none";

/// The `reason` labels, in the order of [`Tally::reasons`]: every reason the crate names
/// (`MalformedReason`, in its order, each under the crate's own spelling of it), so a dashboard
/// tells a stray tag from a call written in the wrong syntax without the sampled line, and
/// `other` for a reason this build does not know (the crate's enum is non-exhaustive).
const REASONS: [&str; 31] = [
    "unterminated_region",
    "invalid_arguments",
    "unknown_tool",
    "text_between_tags",
    "tag_out_of_place",
    "tag_cut_short",
    "tag_tail",
    "empty_name",
    "call_without_name",
    "call_without_function_tag",
    "second_function_tag",
    "closed_before_the_name",
    "text_where_the_name_should_be",
    "text_between_calls",
    "text_where_an_argument_should_begin",
    "text_in_place_of_the_list",
    "text_after_the_arguments",
    "text_after_the_list",
    "name_without_arguments",
    "name_without_a_call",
    "key_written_twice",
    "key_without_a_value",
    "key_cut_short",
    "value_cut_short",
    "empty_key",
    "block_without_a_complete_call",
    "no_forced_function",
    "marker_out_of_place",
    "open_at_the_stop",
    "open_at_the_marker",
    "other",
];
/// The place of `other` in [`REASONS`]: a reason the match below does not know.
const OTHER_REASON: usize = REASONS.len() - 1;
/// The `outcome` labels, in the order of [`Series::outputs`].
const OUTCOMES: [&str; 6] = [
    "clean",
    "malformed",
    "failed",
    "empty",
    "unknown_tool",
    "panicked",
];
/// The `reason` labels of [`FALLBACK`], in the order of [`fallback_index`]: the four decisions
/// that leave a model to the legacy parsers, and the table that has no tag for a forced call.
const FALLBACK_REASONS: [&str; 5] = [
    "no_table",
    "reasoning_name_only",
    "legacy_tool_name",
    "passthrough",
    "forced_call_without_tag",
];
/// The `reason` label of [`REFUSED`]: the one refusal the Messages fold names.
const REFUSED_TOOL_USE_INPUT: &str = "tool_use_input";
const CLEAN: usize = 0;
const MALFORMED_OUTCOME: usize = 1;
const FAILED: usize = 2;
const EMPTY: usize = 3;
const UNKNOWN_TOOL: usize = 4;
const PANICKED: usize = 5;
/// The place of `unknown_tool` in [`REASONS`]: the runs that are calls to functions the request
/// did not declare, which decide between the `malformed` and the `unknown_tool` outcome.
const UNKNOWN_TOOL_REASON: usize = 2;
/// The bytes of an undeclared function's name the `unknown_tool` line shows at most.
const NAME_SHOWN: usize = 64;

/// The `table` label of the format named `name`: the registry table's own name, the response
/// template's, or `other`. Static, so a record allocates nothing for it.
pub(crate) fn table_label(name: &str) -> &'static str {
    Table::ALL
        .iter()
        .map(|table| table.name())
        .find(|table| *table == name)
        .unwrap_or(if name == TEMPLATE {
            TEMPLATE
        } else {
            OTHER_TABLE
        })
}

/// The `reason` label of a malformed run, and its place in [`REASONS`]. The arms follow the
/// crate's enum; a variant this build does not know lands on `other`.
fn reason_index(why: MalformedReason) -> usize {
    match why {
        MalformedReason::UnterminatedRegion => 0,
        MalformedReason::InvalidArguments => 1,
        MalformedReason::UnknownTool => 2,
        MalformedReason::TextBetweenTags => 3,
        MalformedReason::TagOutOfPlace => 4,
        MalformedReason::TagCutShort => 5,
        MalformedReason::TagTail => 6,
        MalformedReason::EmptyName => 7,
        MalformedReason::CallWithoutName => 8,
        MalformedReason::CallWithoutFunctionTag => 9,
        MalformedReason::SecondFunctionTag => 10,
        MalformedReason::ClosedBeforeTheName => 11,
        MalformedReason::TextWhereTheNameShouldBe => 12,
        MalformedReason::TextBetweenCalls => 13,
        MalformedReason::TextWhereAnArgumentShouldBegin => 14,
        MalformedReason::TextInPlaceOfTheList => 15,
        MalformedReason::TextAfterTheArguments => 16,
        MalformedReason::TextAfterTheList => 17,
        MalformedReason::NameWithoutArguments => 18,
        MalformedReason::NameWithoutACall => 19,
        MalformedReason::KeyWrittenTwice => 20,
        MalformedReason::KeyWithoutAValue => 21,
        MalformedReason::KeyCutShort => 22,
        MalformedReason::ValueCutShort => 23,
        MalformedReason::EmptyKey => 24,
        MalformedReason::BlockWithoutACompleteCall => 25,
        MalformedReason::NoForcedFunction => 26,
        MalformedReason::MarkerOutOfPlace => 27,
        MalformedReason::OpenAtTheStop => 28,
        MalformedReason::OpenAtTheMarker => 29,
        _ => OTHER_REASON,
    }
}

/// The `reason` label of a malformed run.
pub(crate) fn reason_label(why: MalformedReason) -> &'static str {
    REASONS[reason_index(why)]
}

/// The engine's finish as the line spells it, the public API's words.
fn finish_label(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Abort => "abort",
        FinishReason::Other(_) => "other",
    }
}

/// The place of a fallback's `reason` label in [`FALLBACK_REASONS`].
fn fallback_index(why: Fallback) -> usize {
    match why {
        Fallback::NoTable => 0,
        Fallback::ReasoningNameOnly => 1,
        Fallback::LegacyToolName => 2,
        Fallback::Passthrough => 3,
        Fallback::ForcedCallWithoutTag => FORCED_CALL_WITHOUT_TAG,
    }
}

/// The place of `forced_call_without_tag` in [`FALLBACK_REASONS`]: the one fallback counted under
/// a table of its own ([`record_fallback`]); the other four count under `none`.
const FORCED_CALL_WITHOUT_TAG: usize = 4;

/// The bytes an event accounts for in the output.
pub(super) fn bytes_of(event: &Event) -> usize {
    match event {
        Event::Content(text) | Event::Reasoning(text) => text.text.len(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.len(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.len(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => 0,
    }
}

/// `name` cut to [`NAME_SHOWN`] bytes on a character boundary: what the `unknown_tool` line shows
/// of an undeclared function's name.
fn shown_name(name: &str) -> Box<str> {
    let mut end = name.len().min(NAME_SHOWN);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    name[..end].into()
}

/// Whose output a reader reports on, for the line a non-clean output writes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Who<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub choice: u32,
}

/// The first malformed run of an output the table refused or kept as content, as the malformed
/// line shows it. The event of no bytes before a call to a function the request did not declare
/// is not one: the `unknown_tool` line locates that call ([`Undeclared`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FirstRun {
    /// Its `reason` label.
    pub reason: &'static str,
    /// The crate's sentence for the reason (`MalformedReason::sentence`).
    pub why: &'static str,
    /// Where it starts in the output, in bytes.
    pub at: usize,
    /// Its length in bytes: the event's own, and the bytes a syntax refused right after it.
    pub bytes: usize,
}

/// The first call of an output to a function the request did not declare, as the `unknown_tool`
/// line shows it.
#[derive(Debug)]
struct Undeclared {
    /// The function's name, cut to [`NAME_SHOWN`] bytes.
    name: Box<str>,
    /// The name's whole length in bytes.
    bytes: usize,
    /// Where the call starts in the output, in bytes.
    at: usize,
}

/// What the reader has seen of one output so far, counted as the events are drained.
#[derive(Debug, Default)]
pub(crate) struct Tally {
    /// The bytes of the events seen so far: where the next event's bytes start.
    at: usize,
    /// Malformed runs by reason, in the order of [`REASONS`]; allocated by the first run, so a
    /// clean output carries no counters and the tally stays small.
    reasons: Option<Box<[u32; REASONS.len()]>>,
    malformed_bytes: u64,
    first: Option<FirstRun>,
    /// The event before the next is the first run's: refused bytes that follow it are its size.
    announced: bool,
    /// The next call to start names a function the request did not declare: its `UnknownTool`
    /// event came right before it, and the first such call is the `unknown_tool` line's.
    naming: bool,
    /// The first call to a function the request did not declare, as the `unknown_tool` line
    /// shows it; allocated only for an output with one.
    undeclared: Option<Undeclared>,
    suppressed_calls: u32,
    tool_calls: u32,
    /// The bytes the client was shown: content, reasoning, malformed text kept as content, and
    /// the source of the call events.
    shown: u64,
    /// The calls the output began, suppressed or not.
    calls_begun: u32,
    /// The bytes dropped as whitespace, and as anything else the table drops.
    dropped_space: u64,
    dropped_wrapping: u64,
    /// The engine tokens of the dropped bytes, where the events carried their counts.
    dropped_tokens: u32,
    /// The engine's finish, as the `Finish` event says it.
    finish: &'static str,
    /// The bytes the reader released as content at the end of the output: the dropped whitespace
    /// of an empty output, whether it reaches the client being the surface's rule (the chat
    /// stream carries it), or the refused call blocks an output that made no call ended in,
    /// from the first of them's opener to the last refused byte, which every surface carries.
    released: u64,
}

impl Tally {
    /// Accounts for `event`, the next of the output; `first_call_only` says whether the reader
    /// drops the calls after the first, which are then counted as suppressed.
    pub(crate) fn see(&mut self, event: &Event, first_call_only: bool) {
        let announced = std::mem::take(&mut self.announced);
        let naming = std::mem::take(&mut self.naming);
        match event {
            Event::Content(text) | Event::Reasoning(text) => {
                self.shown += text.text.len() as u64;
            }
            Event::ToolCallArguments { source, .. } | Event::ToolCallEnd { source, .. } => {
                self.shown += source.text.len() as u64;
            }
            Event::Malformed { text, why } => {
                self.shown += text.text.len() as u64;
                self.reasons
                    .get_or_insert_with(|| Box::new([0; REASONS.len()]))[reason_index(*why)] += 1;
                self.malformed_bytes += text.text.len() as u64;
                // The event of no bytes before a call to a function the request did not declare:
                // the call starts next, and the first such call is the `unknown_tool` line's. The
                // malformed line's first run is one the table refused or kept as content, never
                // this event, so its `reason`, `at` and `bytes` locate debris, not a call read
                // whole.
                if matches!(why, MalformedReason::UnknownTool) {
                    self.naming = true;
                } else if self.first.is_none() {
                    self.first = Some(FirstRun {
                        reason: reason_label(*why),
                        why: why.sentence(),
                        at: self.at,
                        bytes: text.text.len(),
                    });
                    self.announced = true;
                }
            }
            // The bytes of a call block a syntax refused (the keyed one, the attributed one inside
            // an invoke): they are the malformed
            // run's, announced by the event of no bytes right before them, so the histogram and
            // the line still say how much of the answer was debris. At a cut call's end the bytes
            // held come before the event (`tagged::end_cut_call`): they count, and the run keeps
            // the size its event carried.
            Event::Dropped {
                text,
                why: DropReason::MalformedCall,
            } => {
                self.malformed_bytes += text.text.len() as u64;
                if announced {
                    if let Some(first) = &mut self.first {
                        first.bytes += text.text.len();
                    }
                }
            }
            Event::Dropped { text, why } => {
                match why {
                    DropReason::Whitespace => self.dropped_space += text.text.len() as u64,
                    _ => self.dropped_wrapping += text.text.len() as u64,
                }
                self.dropped_tokens += text.tokens.unwrap_or(0);
            }
            Event::ToolCallStart {
                index,
                name,
                source,
                ..
            } => {
                self.calls_begun += 1;
                self.shown += source.text.len() as u64;
                if first_call_only && *index > 0 {
                    self.suppressed_calls += 1;
                }
                if naming && self.undeclared.is_none() {
                    self.undeclared = Some(Undeclared {
                        name: shown_name(name),
                        bytes: name.len(),
                        at: self.at,
                    });
                }
            }
            Event::Finish {
                reason, tool_calls, ..
            } => {
                self.tool_calls = *tool_calls;
                self.finish = finish_label(reason);
            }
            Event::ReasoningStart | Event::ReasoningEnd => {}
        }
        self.at += bytes_of(event);
    }

    /// `bytes` more of the output went out in an event seen already: text pushed right after
    /// text of its kind joins the last event rather than making one, and the next event starts
    /// after it.
    pub(crate) fn grew(&mut self, bytes: usize) {
        self.at += bytes;
        self.shown += bytes as u64;
    }

    /// Where the next event's bytes start in the output: the bytes of the events seen so far.
    pub(super) fn at(&self) -> usize {
        self.at
    }

    /// Whether the output showed nothing: no text, no call begun, and no malformed run (whose
    /// text is content); only dropped bytes, or none at all.
    pub(super) fn is_empty(&self) -> bool {
        self.shown == 0 && self.calls_begun == 0 && self.malformed() == 0
    }

    /// `bytes` the table dropped were released as content at the end of the output: whitespace
    /// of an empty output, or the refused call blocks one that made no call ended in, from the
    /// first of them's opener to the last refused byte.
    pub(super) fn released(&mut self, bytes: usize) {
        self.released += bytes as u64;
    }

    /// The first malformed run, for the reader's tests.
    #[cfg(test)]
    pub(super) fn first_run(&self) -> Option<&FirstRun> {
        self.first.as_ref()
    }

    /// Malformed runs by reason, in the order of [`REASONS`]: none for a clean output.
    fn reasons(&self) -> &[u32] {
        self.reasons.as_deref().map_or(&[], |reasons| &reasons[..])
    }

    /// Malformed runs seen.
    fn malformed(&self) -> u32 {
        self.reasons().iter().sum()
    }

    /// The calls to functions the request did not declare among them (`unknown_tool`).
    fn undeclared_calls(&self) -> u32 {
        self.reasons()
            .get(UNKNOWN_TOOL_REASON)
            .copied()
            .unwrap_or(0)
    }

    /// The distinct reason labels seen, in the order of [`REASONS`], joined with a comma.
    fn reasons_seen(&self) -> String {
        let mut seen = String::new();
        for (count, label) in self.reasons().iter().zip(REASONS) {
            if *count > 0 {
                if !seen.is_empty() {
                    seen.push(',');
                }
                seen.push_str(label);
            }
        }
        seen
    }

    /// Writes the line of an output counted `unknown_tool`, when the table's bucket for those
    /// lines allows one: the first undeclared function's name, cut to [`NAME_SHOWN`] bytes with
    /// its whole length beside it, where its call starts, and how many such calls the output
    /// made.
    fn write_unknown_tool(
        &self,
        series: &Series,
        table: &'static str,
        who: Who<'_>,
        undeclared: u32,
    ) {
        let Some(suppressed_lines) = series.unknown_tool_lines.allow() else {
            return;
        };
        let (tool, tool_bytes, at) = match &self.undeclared {
            Some(call) => (&*call.name, call.bytes, call.at),
            None => ("", 0, 0),
        };
        warn!(
            target: "smg::symphony",
            request_id = who.request_id,
            choice = who.choice,
            model = who.model,
            table,
            outcome = OUTCOMES[UNKNOWN_TOOL],
            finish = self.finish,
            undeclared_calls = undeclared,
            tool,
            tool_bytes,
            at,
            tool_calls = self.tool_calls,
            suppressed_calls = self.suppressed_calls,
            suppressed_lines,
            "Symphony read a call to a function the request did not declare; the call went out as the model named it"
        );
    }

    /// Records the output under `table`, read to its end: the counters, and the line when the
    /// output has a malformed run, a call to a function the request did not declare, or showed
    /// nothing ([`Self::is_empty`]), and the table's bucket for that kind of line allows one (the
    /// empty lines and the undeclared-call lines have buckets of their own, so a flow of either
    /// suppresses no malformed line). An output whose malformed runs are all such calls is
    /// counted `unknown_tool`, not `malformed`: the table read it whole. Suppressed calls are
    /// counted and not written: the request asked for them to be dropped.
    pub(crate) fn report(&self, table: &'static str, who: Who<'_>) {
        let series = series(table);
        let malformed = self.malformed();
        let undeclared = self.undeclared_calls();
        let empty = self.is_empty();
        let outcome = if malformed > undeclared {
            MALFORMED_OUTCOME
        } else if undeclared > 0 {
            UNKNOWN_TOOL
        } else if empty {
            EMPTY
        } else {
            CLEAN
        };
        increment(&series.outputs[outcome], 1);
        if empty {
            let Some(suppressed_lines) = series.empty_lines.allow() else {
                return;
            };
            warn!(
                target: "smg::symphony",
                request_id = who.request_id,
                choice = who.choice,
                model = who.model,
                table,
                outcome = OUTCOMES[EMPTY],
                finish = self.finish,
                bytes = self.at,
                whitespace_bytes = self.dropped_space,
                wrapper_bytes = self.dropped_wrapping,
                dropped_tokens = self.dropped_tokens,
                released_bytes = self.released,
                suppressed_lines,
                "Symphony read an output with no text, reasoning or call: whitespace and framing alone, or nothing"
            );
            return;
        }
        if malformed == 0 && self.suppressed_calls == 0 {
            return;
        }
        for (count, key) in self.reasons().iter().zip(&series.malformed) {
            if *count > 0 {
                increment(key, u64::from(*count));
            }
        }
        if outcome == MALFORMED_OUTCOME {
            observe(&series.malformed_bytes, self.malformed_bytes as f64);
        }
        if self.suppressed_calls > 0 {
            increment(&series.calls_suppressed, u64::from(self.suppressed_calls));
        }
        if malformed == 0 {
            return;
        }
        if outcome == UNKNOWN_TOOL {
            self.write_unknown_tool(series, table, who, undeclared);
            return;
        }
        let Some(suppressed_lines) = series.sampler.allow() else {
            return;
        };
        let outcome = OUTCOMES[outcome];
        let (reason, why, at, bytes) = match &self.first {
            Some(first) => (first.reason, first.why, first.at, first.bytes),
            None => ("", "", 0, 0),
        };
        warn!(
            target: "smg::symphony",
            request_id = who.request_id,
            choice = who.choice,
            model = who.model,
            table,
            outcome,
            malformed,
            malformed_bytes = self.malformed_bytes,
            reason,
            why,
            at,
            bytes,
            reasons = %self.reasons_seen(),
            tool_calls = self.tool_calls,
            suppressed_calls = self.suppressed_calls,
            released_bytes = self.released,
            suppressed_lines,
            "Symphony did not read the whole output as content, reasoning and calls"
        );
        // The run's text is model output and may carry the user's data: it is not logged, at
        // any level; `at` and `bytes` locate it in the response for a reader with access to it.
    }
}

/// Records an output the reader could not read whole, which the request fails with: the table
/// refused it (`ParseError`), or the decoder refused a token or the text it held (the reader's
/// `failed`).
pub(crate) fn record_failed(table: &'static str) {
    increment(&series(table).outputs[FAILED], 1);
}

/// Records a whole Messages response refused after its output was read under `table`: its
/// `tool_use` block cannot carry the arguments the model wrote. The output's own count stands,
/// whatever outcome the read gave it; this one is the fold's, by its reason.
pub(crate) fn record_refused(table: &'static str) {
    increment(&series(table).refused, 1);
}

/// Records an output whose reading panicked under `table`, caught at the request's boundary
/// (`routers::grpc::utils::isolation`): the request fails with it. The line is the isolation's.
pub(crate) fn record_panicked(table: &'static str) {
    increment(&series(table).outputs[PANICKED], 1);
}

/// Records a request Symphony was configured for that the legacy parsers read, for `why`: under
/// `table` where a table reads the model and had no tag for the request's forced call, under
/// `none` where no table reads it. One increment through a key built once, no allocation.
pub(crate) fn record_fallback(why: Fallback, table: Option<&'static str>) {
    increment(
        &series(table.unwrap_or(NO_TABLE)).fallback[fallback_index(why)],
        1,
    );
}

/// Publishes the series of the table labelled `table` at zero, so a dashboard or an alert finds
/// `failed`, `malformed` and the rest before the first such output instead of no series at all:
/// the outcomes, the malformed reasons, the suppressed calls, the refusal, the fallback of a forced
/// call the table has no tag for, and the empty distribution of the malformed bytes. Called where
/// the table that reads a model becomes known, at the model's first registration
/// (`workflow::steps::local::create_worker`); a table published again sets nothing.
pub(crate) fn init_table_series(table: &'static str) {
    let series = series(table);
    for key in series.outputs.iter().chain(series.malformed.iter()).chain([
        &series.calls_suppressed,
        &series.refused,
        &series.fallback[FORCED_CALL_WITHOUT_TAG],
    ]) {
        publish(key);
    }
    // A histogram has no zero to set: registering it publishes the family with an empty
    // distribution.
    let _ = metrics::with_recorder(|recorder| {
        recorder.register_histogram(&series.malformed_bytes, &METADATA)
    });
}

/// Publishes at zero the series of the requests no table reads ([`FALLBACK`] under `none`): the
/// four decisions that leave a model to the legacy parsers are known before any worker registers.
/// Called once the recorder is installed, under `--parser symphony`.
pub(crate) fn init_startup_series() {
    let series = series(NO_TABLE);
    for (index, key) in series.fallback.iter().enumerate() {
        if index != FORCED_CALL_WITHOUT_TAG {
            publish(key);
        }
    }
}

/// The counter under `key`, at zero if it did not exist: `absolute(0)` never lowers a counter.
fn publish(key: &Key) {
    metrics::with_recorder(|recorder| recorder.register_counter(key, &METADATA)).absolute(0);
}

/// The metadata of every series here, as the `metrics` macros would set it.
static METADATA: Metadata<'static> =
    Metadata::new(module_path!(), Level::INFO, Some(module_path!()));

/// One more on the counter under `key`: the registry's lookup, no key built.
fn increment(key: &Key, by: u64) {
    metrics::with_recorder(|recorder| recorder.register_counter(key, &METADATA)).increment(by);
}

/// One observation on the histogram under `key`.
fn observe(key: &Key, value: f64) {
    metrics::with_recorder(|recorder| recorder.register_histogram(key, &METADATA)).record(value);
}

/// The keys of one table's series, built once, and the bucket for its lines.
struct Series {
    /// [`OUTPUTS`] by outcome, in the order of [`OUTCOMES`].
    outputs: [Key; OUTCOMES.len()],
    /// [`MALFORMED`] by reason, in the order of [`REASONS`].
    malformed: [Key; REASONS.len()],
    malformed_bytes: Key,
    calls_suppressed: Key,
    /// [`FALLBACK`] by reason, in the order of [`FALLBACK_REASONS`]: the series of the table
    /// that had no tag for a forced call, or of `none`, under which the other reasons count.
    fallback: [Key; 5],
    /// [`REFUSED`] under its one reason.
    refused: Key,
    /// The bucket of the lines on malformed outputs.
    sampler: Sampler,
    /// The bucket of the lines on empty outputs: their own, so a steady flow of empty outputs (a
    /// probe with `max_tokens: 1`, a keepalive that finishes with no token) suppresses no
    /// malformed line and counts on none.
    empty_lines: Sampler,
    /// The bucket of the lines on outputs with a call to a function the request did not declare:
    /// their own, so a model that names its functions freely suppresses no malformed line and
    /// counts on none.
    unknown_tool_lines: Sampler,
}

impl Series {
    fn new(table: &'static str) -> Self {
        let key = |name: &'static str, labels: &[(&'static str, &'static str)]| {
            let mut all = vec![Label::from_static_parts("table", table)];
            all.extend(
                labels
                    .iter()
                    .map(|(key, value)| Label::from_static_parts(key, value)),
            );
            Key::from_parts(name, all)
        };
        Self {
            outputs: OUTCOMES.map(|outcome| key(OUTPUTS, &[("outcome", outcome)])),
            malformed: REASONS.map(|reason| key(MALFORMED, &[("reason", reason)])),
            malformed_bytes: key(MALFORMED_BYTES, &[]),
            calls_suppressed: key(CALLS_SUPPRESSED, &[]),
            fallback: FALLBACK_REASONS.map(|reason| key(FALLBACK, &[("reason", reason)])),
            refused: key(REFUSED, &[("reason", REFUSED_TOOL_USE_INPUT)]),
            sampler: Sampler::new(),
            empty_lines: Sampler::new(),
            unknown_tool_lines: Sampler::new(),
        }
    }
}

/// Every table's series: the registry's tables in their order, then the template's format,
/// then the rest ([`OTHER_TABLE`]), then the series of a request no table reads ([`NO_TABLE`]).
static SERIES: OnceLock<Vec<Series>> = OnceLock::new();

/// The series of the table labelled `table`.
fn series(table: &'static str) -> &'static Series {
    let all = SERIES.get_or_init(|| {
        Table::ALL
            .iter()
            .map(|table| table.name())
            .chain([TEMPLATE, OTHER_TABLE, NO_TABLE])
            .map(Series::new)
            .collect()
    });
    let index = Table::ALL
        .iter()
        .position(|candidate| candidate.name() == table)
        .unwrap_or_else(|| match table {
            TEMPLATE => Table::ALL.len(),
            NO_TABLE => Table::ALL.len() + 2,
            _ => Table::ALL.len() + 1,
        });
    &all[index]
}

/// A token bucket for one table's lines: a burst, then a steady rate; the lines it refuses are
/// counted and reported on the next line it allows, so the count on the wire stays honest.
#[derive(Debug)]
pub(crate) struct Sampler {
    burst: f64,
    refill_per_second: f64,
    /// The tokens left and when they were last refilled.
    state: Mutex<(f64, Option<Instant>)>,
    suppressed: AtomicU32,
}

impl Sampler {
    /// Lines a table may write at once before the rate applies.
    const BURST: f64 = 20.0;
    /// Lines per second a table may write steadily.
    const REFILL_PER_SECOND: f64 = 2.0;

    fn new() -> Self {
        Self::with(Self::BURST, Self::REFILL_PER_SECOND)
    }

    /// A bucket of `burst` tokens refilled at `refill_per_second`.
    pub(crate) fn with(burst: f64, refill_per_second: f64) -> Self {
        Self {
            burst,
            refill_per_second,
            state: Mutex::new((burst, None)),
            suppressed: AtomicU32::new(0),
        }
    }

    /// Whether a line may be written now: the lines refused since the last one written, when it
    /// may; `None` when it may not.
    pub(crate) fn allow(&self) -> Option<u32> {
        let mut state = self.state.lock();
        let now = Instant::now();
        let (tokens, last) = &mut *state;
        if let Some(last) = last {
            *tokens = (*tokens
                + now.saturating_duration_since(*last).as_secs_f64() * self.refill_per_second)
                .min(self.burst);
        }
        *last = Some(now);
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            drop(state);
            Some(self.suppressed.swap(0, Ordering::Relaxed))
        } else {
            drop(state);
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use symphony::{DropReason, FinishReason, Text};

    use super::*;
    use crate::routers::grpc::regular::symphony::testing::{line_with, rendered};

    fn malformed(text: &str, why: MalformedReason) -> Event<'static> {
        Event::Malformed {
            text: Text::uncounted(text.to_owned()),
            why,
        }
    }

    fn call_start(index: u32, source: &str) -> Event<'static> {
        call_named(index, "lookup", source)
    }

    fn call_named(index: u32, name: &str, source: &str) -> Event<'static> {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.to_owned(),
            source: Text::uncounted(source.to_owned()),
        }
    }

    fn finish(tool_calls: u32) -> Event<'static> {
        Event::Finish {
            reason: FinishReason::Stop,
            tool_calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn every_registry_table_labels_itself_and_the_rest_is_named() {
        for table in Table::ALL {
            assert_eq!(table_label(table.name()), table.name());
        }
        assert_eq!(table_label("response_template"), "response_template");
        assert_eq!(table_label("markers"), "other");
        assert_eq!(table_label(""), "other");
    }

    #[test]
    fn a_tally_counts_the_runs_their_bytes_and_where_the_first_one_starts() {
        let mut tally = Tally::default();
        let events = [
            Event::Content(Text::uncounted("Hello ")),
            Event::Dropped {
                text: Text::uncounted("<tool_call>"),
                why: DropReason::Wrapper,
            },
            malformed("\nnot a call\n", MalformedReason::InvalidArguments),
            Event::Dropped {
                text: Text::uncounted("</tool_call>"),
                why: DropReason::Wrapper,
            },
            malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
            malformed("x", MalformedReason::CallWithoutName),
            finish(0),
        ];
        for event in &events {
            tally.see(event, false);
        }
        assert_eq!(tally.malformed(), 3);
        assert_eq!(tally.malformed_bytes, 12 + 12 + 1);
        assert_eq!(
            tally.first,
            Some(FirstRun {
                reason: "invalid_arguments",
                why: "arguments that did not parse under the call's syntax",
                at: "Hello <tool_call>".len(),
                bytes: 12,
            })
        );
        assert_eq!(
            tally.reasons_seen(),
            "unterminated_region,invalid_arguments,call_without_name"
        );
        assert_eq!(tally.suppressed_calls, 0);
    }

    #[test]
    fn refused_bytes_size_the_run_that_announced_them() {
        // The refuse order: the event of no bytes, then the bytes it refuses; a second refused
        // run's bytes count, but are not the first run's.
        let refused = |text: &str| Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::MalformedCall,
        };
        let mut tally = Tally::default();
        for event in [
            Event::Content(Text::uncounted("Hi")),
            malformed("", MalformedReason::TagCutShort),
            refused("<arg_key>format=2D"),
            malformed("", MalformedReason::TagOutOfPlace),
            refused("</arg_value>"),
            finish(1),
        ] {
            tally.see(&event, false);
        }
        assert_eq!(tally.malformed(), 2);
        assert_eq!(tally.malformed_bytes, 18 + 12);
        assert_eq!(
            tally.first,
            Some(FirstRun {
                reason: "tag_cut_short",
                why: "a tag that another tag or the call's end cut short",
                at: 2,
                bytes: 18,
            })
        );
        // The cut order: the bytes held at the cut, then the word before the call's end: the
        // bytes count, and the run keeps the size its event carried.
        let mut tally = Tally::default();
        for event in [
            refused("<"),
            malformed("", MalformedReason::UnterminatedRegion),
            Event::ToolCallEnd {
                index: 0,
                source: Text::uncounted("</tool_call>"),
            },
            finish(1),
        ] {
            tally.see(&event, false);
        }
        assert_eq!(tally.malformed_bytes, 1);
        assert_eq!(
            tally.first,
            Some(FirstRun {
                reason: "unterminated_region",
                why: "the stream ended inside a region that never closed",
                at: 1,
                bytes: 0,
            })
        );
        // The template's wrapping after a run is not the run's.
        let mut tally = Tally::default();
        for event in [
            malformed("x", MalformedReason::CallWithoutName),
            Event::Dropped {
                text: Text::uncounted("\n"),
                why: DropReason::Wrapper,
            },
            finish(0),
        ] {
            tally.see(&event, false);
        }
        assert_eq!(tally.malformed_bytes, 1);
        assert_eq!(tally.first.as_ref().map(|first| first.bytes), Some(1));
    }

    #[test]
    fn a_botched_keyed_block_is_sized_by_its_refused_bytes() {
        // Through the keyed table itself: a call whose key tag never opened. The assembler refuses
        // the tags and the value behind events of no bytes; the tally counts the runs and sizes
        // them by the refused bytes, the first by the bytes right after it.
        use symphony::{Declared, Engine, EngineFinish, Events, Input, Parser};
        let output =
            "</think>At once:<tool_call>f</arg_key><arg_value>1975</arg_value></tool_call>";
        let mut engine = Engine::new(Table::Glm.format(), Declared::default());
        let mut out = Events::new();
        engine
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|assistant|><think>",
                },
                &mut out,
            )
            .expect("the prompt");
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("the delta");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("the end");
        let mut tally = Tally::default();
        for event in out.as_slice() {
            tally.see(event, false);
        }
        assert_eq!(tally.malformed(), 4, "{:?}", out.as_slice());
        assert_eq!(
            tally.malformed_bytes,
            "</arg_key><arg_value>1975</arg_value>".len() as u64
        );
        assert_eq!(
            tally.first,
            Some(FirstRun {
                reason: "tag_out_of_place",
                why: "a tag where the call's syntax has none",
                at: "</think>At once:<tool_call>f".len(),
                bytes: "</arg_key>".len(),
            })
        );
        assert_eq!(tally.tool_calls, 1);
    }

    #[test]
    fn the_first_run_of_a_syntax_reason_carries_its_own_label_and_sentence() {
        let mut tally = Tally::default();
        tally.see(
            &malformed("<|end|>", MalformedReason::NameWithoutACall),
            false,
        );
        let first = tally.first.as_ref().expect("a first run");
        assert_eq!(first.reason, "name_without_a_call");
        assert_eq!(first.why, "a name without a call");
        assert_eq!(first.at, 0);
    }

    #[test]
    fn every_reason_the_crate_names_has_a_label_of_its_own_spelled_as_the_crate_spells_it() {
        // The crate's own list of its reasons (`MalformedReason::ALL`, written with the enum by
        // one macro list) drives this: the label is the crate's serde spelling of the reason, so a
        // line, a trace and the counter agree on the name; no two reasons share a label, none
        // lands on `other`, and a reason added to the crate fails this test until it has a label.
        let mut labels = std::collections::HashSet::new();
        for (index, why) in MalformedReason::ALL.iter().copied().enumerate() {
            let label = reason_label(why);
            let spelled = serde_json::to_value(why).expect("a reason serializes");
            assert_eq!(Some(label), spelled.as_str(), "{why:?}");
            assert_eq!(reason_index(why), index, "{why:?}");
            assert_ne!(label, "other", "{why:?}");
            assert!(labels.insert(label), "{why:?} shares its label");
        }
        assert_eq!(labels.len() + 1, REASONS.len());
        assert_eq!(REASONS[OTHER_REASON], "other");
    }

    #[test]
    fn a_clean_tally_allocates_no_counters_and_a_botched_one_counts_each_reason_by_name() {
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("Hi")), false);
        tally.see(&finish(0), false);
        assert!(tally.reasons.is_none());
        assert_eq!(tally.malformed(), 0);
        assert_eq!(tally.reasons_seen(), "");
        let mut tally = Tally::default();
        for event in [
            malformed("", MalformedReason::TextWhereTheNameShouldBe),
            malformed("", MalformedReason::TagOutOfPlace),
            malformed("", MalformedReason::TextBetweenTags),
            malformed("", MalformedReason::TagOutOfPlace),
            malformed("", MalformedReason::CallWithoutName),
            finish(0),
        ] {
            tally.see(&event, false);
        }
        assert_eq!(tally.malformed(), 5);
        assert_eq!(
            tally.reasons_seen(),
            "text_between_tags,tag_out_of_place,call_without_name,text_where_the_name_should_be"
        );
        let reasons = tally.reasons();
        assert_eq!(reasons[reason_index(MalformedReason::TagOutOfPlace)], 2);
        assert_eq!(reasons[reason_index(MalformedReason::TextBetweenTags)], 1);
    }

    #[test]
    fn a_kind_said_again_inside_an_answer_counts_the_output_as_clean() {
        // Through the table whose messages say their kind after their opener, as a request
        // without tools reads it: a JSON answer whose text marker the model wrote twice, one
        // with a second thought opened inside the answer's message, as an engine that starts the
        // grammar after its own reasoning parser lets the model write them, and one whose text
        // marker the model wrote inside its thought. The markers are framing, the thoughts are
        // reasoning, the JSON is the content, and the tally sees no malformed run: the output
        // counts as clean, under no reason.
        use symphony::{formats::Table, Declared, Engine, EngineFinish, Events, Input, Parser};
        let json = "{\"city\": \"Oslo\"}";
        for output in [
            format!(
                "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>\
                 <|content_text|>{json}<|end_message|>"
            ),
            format!(
                "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>\
                 <|content_thinking|>more<|end_message|><|message_model|><|content_text|>{json}\
                 <|end_message|>"
            ),
            format!("<|content_thinking|>plan<|content_text|>{json}<|end_message|>"),
        ] {
            let mut engine =
                Engine::new(Table::Inkling.format().without_calls(), Declared::default());
            let mut out = Events::new();
            engine
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: "<|message_model|>",
                    },
                    &mut out,
                )
                .expect("the prompt");
            engine
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: &output,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("the delta");
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .expect("the end");
            let mut tally = Tally::default();
            let mut content = String::new();
            for event in out.as_slice() {
                tally.see(event, false);
                if let Event::Content(text) = event {
                    content.push_str(&text.text);
                }
            }
            assert_eq!(content, json, "{:?}", out.as_slice());
            assert_eq!(tally.malformed(), 0, "{:?}", out.as_slice());
            assert_eq!(tally.malformed_bytes, 0);
            assert_eq!(tally.reasons_seen(), "");
            assert_eq!(tally.first, None);
        }
    }

    #[test]
    fn a_marker_out_of_place_has_a_label_of_its_own_and_counts_the_output_as_malformed() {
        // The crate says so with an event of no bytes before the marker's text: the run counts,
        // the first run stands where the marker does, and its bytes are none.
        let mut tally = Tally::default();
        tally.see(&Event::Content(Text::uncounted("<|text|>")), false);
        tally.see(&malformed("", MalformedReason::MarkerOutOfPlace), false);
        tally.see(&Event::Content(Text::uncounted("<|text|>{}")), false);
        assert_eq!(tally.malformed(), 1);
        assert_eq!(tally.malformed_bytes, 0);
        assert_eq!(tally.reasons_seen(), "marker_out_of_place");
        assert_eq!(
            tally.first,
            Some(FirstRun {
                reason: "marker_out_of_place",
                why: "a marker of the table where no row leads, kept as text",
                at: "<|text|>".len(),
                bytes: 0,
            })
        );
        assert_eq!(
            reason_label(MalformedReason::MarkerOutOfPlace),
            "marker_out_of_place"
        );
    }

    #[test]
    fn calls_after_the_first_count_as_suppressed_only_when_the_reader_drops_them() {
        let events = [
            call_start(0, "<tool_call>"),
            call_start(1, "<tool_call>"),
            call_start(2, "<tool_call>"),
            finish(3),
        ];
        let mut dropping = Tally::default();
        let mut keeping = Tally::default();
        for event in &events {
            dropping.see(event, true);
            keeping.see(event, false);
        }
        assert_eq!(dropping.suppressed_calls, 2);
        assert_eq!(keeping.suppressed_calls, 0);
        assert_eq!(dropping.tool_calls, 3);
        assert_eq!(dropping.at, 33);
    }

    #[test]
    fn a_bucket_allows_its_burst_then_counts_what_it_refuses() {
        let sampler = Sampler::with(2.0, 0.0);
        assert_eq!(sampler.allow(), Some(0));
        assert_eq!(sampler.allow(), Some(0));
        assert_eq!(sampler.allow(), None);
        assert_eq!(sampler.allow(), None);
        assert_eq!(sampler.allow(), None);
        // Refilled by hand: the next line reports the three refused.
        sampler.state.lock().0 = 1.0;
        assert_eq!(sampler.allow(), Some(3));
        assert_eq!(sampler.allow(), None);
    }

    #[test]
    fn a_bucket_refills_with_time() {
        let sampler = Sampler::with(1.0, 1.0e9);
        assert_eq!(sampler.allow(), Some(0));
        // A nanosecond refills a token at this rate.
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert_eq!(sampler.allow(), Some(0));
    }

    #[test]
    fn every_table_has_series_and_a_bucket_of_its_own() {
        let mut distinct = std::collections::HashSet::new();
        for table in Table::ALL {
            let own = series(table.name());
            assert_eq!(own.outputs[CLEAN].labels().count(), 2);
            assert!(own.outputs[CLEAN]
                .labels()
                .any(|label| label.key() == "table" && label.value() == table.name()));
            distinct.insert(own as *const Series);
        }
        distinct.insert(series("response_template") as *const Series);
        distinct.insert(series("other") as *const Series);
        distinct.insert(series("markers") as *const Series);
        assert_eq!(distinct.len(), Table::ALL.len() + 2);
    }

    #[test]
    fn a_clean_output_records_one_counter_and_no_line() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&Event::Content(Text::uncounted("Hello")), false);
            tally.see(&finish(0), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-1",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let line = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
            .expect("the outputs counter");
        assert!(line.ends_with(" 1"), "{line}");
        assert!(!scrape.contains(MALFORMED), "{scrape}");
        assert!(!scrape.contains(MALFORMED_BYTES), "{scrape}");
        assert!(!scrape.contains(CALLS_SUPPRESSED), "{scrape}");
    }

    #[test]
    fn a_malformed_output_records_the_runs_by_reason_and_their_bytes() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(
                &malformed("\nnot a call\n", MalformedReason::InvalidArguments),
                false,
            );
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&malformed("{", MalformedReason::UnterminatedRegion), false);
            tally.see(&finish(0), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-2",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"malformed\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        let invalid = line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", "reason=\"invalid_arguments\""],
        )
        .expect("the invalid_arguments counter");
        assert!(invalid.ends_with(" 1"), "{invalid}");
        let unterminated = line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", "reason=\"unterminated_region\""],
        )
        .expect("the unterminated_region counter");
        assert!(unterminated.ends_with(" 2"), "{unterminated}");
        let bytes_sum = line_with(
            &scrape,
            &format!("{MALFORMED_BYTES}_sum"),
            &["table=\"qwen3\""],
        )
        .expect("the bytes histogram");
        assert!(bytes_sum.ends_with(" 25"), "{bytes_sum}");
        let bytes_count = line_with(
            &scrape,
            &format!("{MALFORMED_BYTES}_count"),
            &["table=\"qwen3\""],
        )
        .expect("the bytes histogram count");
        assert!(bytes_count.ends_with(" 1"), "{bytes_count}");
    }

    #[test]
    fn suppressed_calls_are_counted_on_an_otherwise_clean_output() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&call_start(0, "<tool_call>"), true);
            tally.see(&call_start(1, "<tool_call>"), true);
            tally.see(&finish(2), true);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-3",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
            .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        let suppressed = line_with(&scrape, CALLS_SUPPRESSED, &["table=\"qwen3\""])
            .expect("the suppressed counter");
        assert!(suppressed.ends_with(" 1"), "{suppressed}");
    }

    #[test]
    fn a_refused_output_is_counted_as_failed() {
        let scrape = rendered(|| record_failed("qwen3"));
        let line = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"failed\""])
            .expect("the outputs counter");
        assert!(line.ends_with(" 1"), "{line}");
    }

    #[test]
    fn a_refused_message_is_counted_by_its_reason_and_not_as_an_output() {
        let scrape = rendered(|| record_refused("qwen3"));
        let line = line_with(
            &scrape,
            REFUSED,
            &["table=\"qwen3\"", "reason=\"tool_use_input\""],
        )
        .expect("the refused counter");
        assert!(line.ends_with(" 1"), "{line}");
        assert!(!scrape.contains(OUTPUTS), "{scrape}");
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_malformed_output_writes_one_line_with_the_first_run() {
        let _scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&Event::Content(Text::uncounted("Hello ")), false);
            tally.see(
                &malformed("\nnot a call\n", MalformedReason::InvalidArguments),
                false,
            );
            tally.see(&finish(0), false);
            tally.report(
                "plain",
                Who {
                    request_id: "req-line",
                    model: "the-model",
                    choice: 1,
                },
            );
        });
        assert!(logs_contain(
            "Symphony did not read the whole output as content, reasoning and calls"
        ));
        assert!(logs_contain("request_id=\"req-line\""));
        assert!(logs_contain("table=\"plain\""));
        assert!(logs_contain("reason=\"invalid_arguments\""));
        assert!(logs_contain("at=6"));
        assert!(logs_contain("choice=1"));
        // The run's text is model output: no line at any level carries it.
        assert!(!logs_contain("not a call"));
        assert!(!logs_contain("excerpt"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_clean_output_writes_no_line() {
        let _scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&Event::Content(Text::uncounted("Hello")), false);
            tally.see(&finish(0), false);
            tally.report(
                "plain",
                Who {
                    request_id: "req-quiet",
                    model: "the-model",
                    choice: 0,
                },
            );
        });
        assert!(!logs_contain("req-quiet"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn suppressed_calls_write_no_line() {
        let _scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&call_start(0, "<tool_call>"), true);
            tally.see(&call_start(1, "<tool_call>"), true);
            tally.see(&finish(2), true);
            tally.report(
                "plain",
                Who {
                    request_id: "req-dropped",
                    model: "the-model",
                    choice: 0,
                },
            );
        });
        assert!(!logs_contain("req-dropped"));
    }

    fn dropped(text: &str, tokens: u32, why: DropReason) -> Event<'static> {
        Event::Dropped {
            text: Text::new(text.to_owned(), tokens),
            why,
        }
    }

    fn finished(reason: FinishReason) -> Event<'static> {
        Event::Finish {
            reason,
            tool_calls: 0,
            reasoning_tokens: 0,
        }
    }

    #[tracing_test::traced_test]
    #[test]
    fn an_output_of_dropped_bytes_alone_is_empty_and_writes_one_line() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&dropped("</think>", 1, DropReason::Wrapper), false);
            tally.see(&dropped("\n\n\n", 3, DropReason::Whitespace), false);
            tally.see(&finished(FinishReason::Length), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-empty",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
            .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert!(
            line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
                .is_none_or(|line| line.ends_with(" 0"))
        );
        assert!(!scrape.contains(MALFORMED), "{scrape}");
        assert!(logs_contain(
            "Symphony read an output with no text, reasoning or call: whitespace and framing alone, or nothing"
        ));
        assert!(logs_contain("req-empty"));
        assert!(logs_contain("released_bytes=0"));
        assert!(logs_contain("finish=\"length\""));
        assert!(logs_contain("bytes=11"));
        assert!(logs_contain("whitespace_bytes=3"));
        assert!(logs_contain("wrapper_bytes=8"));
        assert!(logs_contain("dropped_tokens=4"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn an_output_of_no_bytes_is_empty_with_the_engines_finish() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-nothing",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
            .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert!(logs_contain("req-nothing"));
        assert!(logs_contain("finish=\"stop\""));
        assert!(logs_contain("bytes=0"));
    }

    /// The empty lines have a bucket of their own: a flow of empty outputs past the burst
    /// suppresses no malformed line, whose `suppressed_lines` keeps counting malformed lines alone.
    #[tracing_test::traced_test]
    #[test]
    fn empty_lines_take_from_a_bucket_of_their_own() {
        let empties: u32 = 40;
        rendered(|| {
            for choice in 0..empties {
                let mut tally = Tally::default();
                tally.see(&finished(FinishReason::Stop), false);
                tally.report(
                    "olmo3",
                    Who {
                        request_id: "req-empty-flow",
                        model: "m",
                        choice,
                    },
                );
            }
            let mut tally = Tally::default();
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "olmo3",
                Who {
                    request_id: "req-malformed-after-empties",
                    model: "m",
                    choice: 0,
                },
            );
        });
        logs_assert(|lines: &[&str]| {
            let empty_lines = lines
                .iter()
                .filter(|line| line.contains("req-empty-flow"))
                .count();
            if empty_lines >= empties as usize {
                return Err(format!(
                    "no empty line was suppressed: {empty_lines} written"
                ));
            }
            match lines
                .iter()
                .find(|line| line.contains("req-malformed-after-empties"))
            {
                Some(line) if line.contains("suppressed_lines=0") => Ok(()),
                Some(line) => Err(format!(
                    "the malformed line counts suppressed empty lines: {line}"
                )),
                None => Err("the empty flow suppressed the malformed line".to_string()),
            }
        });
    }

    #[tracing_test::traced_test]
    #[test]
    fn whitespace_before_text_or_a_call_is_not_an_empty_output() {
        let scrape = rendered(|| {
            let mut text = Tally::default();
            text.see(&dropped("\n", 1, DropReason::Whitespace), false);
            text.see(&Event::Content(Text::new("Hi", 1)), false);
            text.see(&finished(FinishReason::Stop), false);
            text.report(
                "qwen3",
                Who {
                    request_id: "req-text",
                    model: "m",
                    choice: 0,
                },
            );
            let mut call = Tally::default();
            call.see(&dropped("\n", 1, DropReason::Whitespace), false);
            call.see(&call_start(0, "<tool_call>"), false);
            call.see(&finish(1), false);
            call.report(
                "qwen3",
                Who {
                    request_id: "req-call",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"clean\""])
            .expect("the outputs counter");
        assert!(outputs.ends_with(" 2"), "{outputs}");
        assert!(
            line_with(&scrape, OUTPUTS, &["table=\"qwen3\"", "outcome=\"empty\""])
                .is_none_or(|line| line.ends_with(" 0"))
        );
        assert!(!logs_contain("req-text"));
        assert!(!logs_contain("req-call"));
    }

    #[test]
    fn a_request_the_legacy_parsers_read_is_counted_by_reason_under_the_table_or_none() {
        let scrape = rendered(|| {
            record_fallback(Fallback::NoTable, None);
            record_fallback(Fallback::NoTable, None);
            record_fallback(Fallback::Passthrough, None);
            record_fallback(Fallback::ForcedCallWithoutTag, Some("qwen3"));
        });
        let none = line_with(
            &scrape,
            FALLBACK,
            &["reason=\"no_table\"", "table=\"none\""],
        )
        .expect("the no_table counter");
        assert!(none.ends_with(" 2"), "{none}");
        let passthrough = line_with(
            &scrape,
            FALLBACK,
            &["reason=\"passthrough\"", "table=\"none\""],
        )
        .expect("the passthrough counter");
        assert!(passthrough.ends_with(" 1"), "{passthrough}");
        let forced = line_with(
            &scrape,
            FALLBACK,
            &["reason=\"forced_call_without_tag\"", "table=\"qwen3\""],
        )
        .expect("the forced call counter");
        assert!(forced.ends_with(" 1"), "{forced}");
        assert_eq!(
            scrape
                .lines()
                .filter(|line| line.starts_with(FALLBACK))
                .count(),
            3,
            "{scrape}"
        );
        assert!(!scrape.contains(OUTPUTS), "{scrape}");
        // Every reason has a label of its own, in the spelling the dashboards read.
        let every = [
            (Fallback::NoTable, "no_table"),
            (Fallback::ReasoningNameOnly, "reasoning_name_only"),
            (Fallback::LegacyToolName, "legacy_tool_name"),
            (Fallback::Passthrough, "passthrough"),
            (Fallback::ForcedCallWithoutTag, "forced_call_without_tag"),
        ];
        for (why, label) in every {
            assert_eq!(FALLBACK_REASONS[fallback_index(why)], label);
        }
        // `none` has a series of its own, apart from the template's and the hand-built formats'.
        assert!(!std::ptr::eq(series(NO_TABLE), series(OTHER_TABLE)));
        assert!(!std::ptr::eq(series(NO_TABLE), series(TEMPLATE)));
        assert_eq!(series(NO_TABLE).fallback.len(), FALLBACK_REASONS.len());
    }

    /// The events of a call to `name`, as the engine gives them for a function the request did
    /// not declare: its event of no bytes, then the call's start.
    fn undeclared_call(index: u32, name: &str) -> [Event<'static>; 2] {
        [
            malformed("", MalformedReason::UnknownTool),
            call_named(index, name, &format!("<tool_call>{name}")),
        ]
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_call_to_an_undeclared_function_is_an_outcome_of_its_own_and_the_line_names_it() {
        assert_eq!(REASONS[UNKNOWN_TOOL_REASON], "unknown_tool");
        assert_eq!(
            reason_index(MalformedReason::UnknownTool),
            UNKNOWN_TOOL_REASON
        );
        assert_eq!(OUTCOMES[UNKNOWN_TOOL], "unknown_tool");
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            tally.see(&Event::Content(Text::uncounted("Sure.")), false);
            for event in undeclared_call(0, "math_pow") {
                tally.see(&event, false);
            }
            tally.see(&finish(1), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-undeclared",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"unknown_tool\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert!(
            line_with(
                &scrape,
                OUTPUTS,
                &["table=\"qwen3\"", "outcome=\"malformed\""]
            )
            .is_none(),
            "{scrape}"
        );
        let runs = line_with(
            &scrape,
            MALFORMED,
            &["table=\"qwen3\"", "reason=\"unknown_tool\""],
        )
        .expect("the reason counter");
        assert!(runs.ends_with(" 1"), "{runs}");
        assert!(
            !scrape.contains(MALFORMED_BYTES),
            "no refused bytes to observe: {scrape}"
        );
        assert!(logs_contain(
            "Symphony read a call to a function the request did not declare; the call went out as the model named it"
        ));
        assert!(logs_contain("outcome=\"unknown_tool\""));
        assert!(logs_contain("table=\"qwen3\""));
        assert!(logs_contain("tool=\"math_pow\""));
        assert!(logs_contain("tool_bytes=8"));
        assert!(logs_contain("undeclared_calls=1"));
        assert!(logs_contain("at=5"));
        assert!(logs_contain("tool_calls=1"));
        assert!(logs_contain("request_id=\"req-undeclared\""));
        assert!(!logs_contain("did not read the whole output"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn every_undeclared_call_is_a_run_and_the_output_counts_once_under_the_first_name() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            for event in undeclared_call(0, "lookup_v2") {
                tally.see(&event, false);
            }
            tally.see(&call_start(1, "<tool_call>lookup"), false);
            for event in undeclared_call(2, "look_up") {
                tally.see(&event, false);
            }
            tally.see(&finish(3), false);
            tally.report(
                "glm",
                Who {
                    request_id: "req-undeclared-twice",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(
            &scrape,
            OUTPUTS,
            &["table=\"glm\"", "outcome=\"unknown_tool\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        let runs = line_with(
            &scrape,
            MALFORMED,
            &["table=\"glm\"", "reason=\"unknown_tool\""],
        )
        .expect("the reason counter");
        assert!(runs.ends_with(" 2"), "{runs}");
        assert!(logs_contain("undeclared_calls=2"));
        assert!(logs_contain("tool=\"lookup_v2\""));
        assert!(!logs_contain("tool=\"look_up\""));
        assert!(logs_contain("tool_calls=3"));
        assert!(logs_contain("at=0"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn an_undeclared_call_beside_a_refused_run_leaves_the_output_malformed() {
        let scrape = rendered(|| {
            let mut tally = Tally::default();
            for event in undeclared_call(0, "lookup_v2") {
                tally.see(&event, false);
            }
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finish(1), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-undeclared-and-refused",
                    model: "m",
                    choice: 0,
                },
            );
        });
        let outputs = line_with(
            &scrape,
            OUTPUTS,
            &["table=\"qwen3\"", "outcome=\"malformed\""],
        )
        .expect("the outputs counter");
        assert!(outputs.ends_with(" 1"), "{outputs}");
        assert!(
            line_with(
                &scrape,
                OUTPUTS,
                &["table=\"qwen3\"", "outcome=\"unknown_tool\""]
            )
            .is_none(),
            "{scrape}"
        );
        for reason in ["unknown_tool", "unterminated_region"] {
            let runs = line_with(
                &scrape,
                MALFORMED,
                &["table=\"qwen3\"", &format!("reason=\"{reason}\"")],
            )
            .expect("the reason counter");
            assert!(runs.ends_with(" 1"), "{runs}");
        }
        let count = line_with(&scrape, &format!("{MALFORMED_BYTES}_count"), &[])
            .expect("the refused bytes observed");
        assert!(count.ends_with(" 1"), "{count}");
        assert!(logs_contain(
            "Symphony did not read the whole output as content, reasoning and calls"
        ));
        // The line's first run is the refused one, after the undeclared call's 20 bytes; the
        // event of no bytes before that call is not a run the line locates.
        assert!(logs_contain("reason=\"unterminated_region\""));
        assert!(logs_contain("at=20 bytes=12"));
        assert!(!logs_contain("reason=\"unknown_tool\""));
        assert!(logs_contain("reasons=unterminated_region,unknown_tool"));
        assert!(!logs_contain(
            "Symphony read a call to a function the request did not declare"
        ));
    }

    #[tracing_test::traced_test]
    #[test]
    fn a_long_undeclared_name_is_cut_on_the_line_with_its_whole_length_beside_it() {
        // Forty two-byte characters: the cut falls on a character boundary, below the bound.
        let name = "ü".repeat(40);
        rendered(|| {
            let mut tally = Tally::default();
            for event in undeclared_call(0, &name) {
                tally.see(&event, false);
            }
            tally.see(&finish(1), false);
            tally.report(
                "qwen3",
                Who {
                    request_id: "req-undeclared-long",
                    model: "m",
                    choice: 0,
                },
            );
        });
        assert!(logs_contain(&format!("tool=\"{}\"", "ü".repeat(32))));
        assert!(logs_contain("tool_bytes=80"));
    }

    #[tracing_test::traced_test]
    #[test]
    fn unknown_tool_lines_take_from_a_bucket_of_their_own() {
        let flow: u32 = 40;
        rendered(|| {
            for choice in 0..flow {
                let mut tally = Tally::default();
                for event in undeclared_call(0, "lookup_v2") {
                    tally.see(&event, false);
                }
                tally.see(&finish(1), false);
                tally.report(
                    "hy4",
                    Who {
                        request_id: "req-undeclared-flow",
                        model: "m",
                        choice,
                    },
                );
            }
            let mut tally = Tally::default();
            tally.see(
                &malformed("<tool_call>{", MalformedReason::UnterminatedRegion),
                false,
            );
            tally.see(&finished(FinishReason::Stop), false);
            tally.report(
                "hy4",
                Who {
                    request_id: "req-malformed-after-undeclared",
                    model: "m",
                    choice: 0,
                },
            );
        });
        logs_assert(|lines: &[&str]| {
            let written = lines
                .iter()
                .filter(|line| line.contains("req-undeclared-flow"))
                .count();
            if written >= flow as usize {
                return Err(format!(
                    "no undeclared-call line was suppressed: {written} written"
                ));
            }
            match lines
                .iter()
                .find(|line| line.contains("req-malformed-after-undeclared"))
            {
                Some(line) if line.contains("suppressed_lines=0") => Ok(()),
                Some(line) => Err(format!(
                    "the malformed line counts suppressed undeclared-call lines: {line}"
                )),
                None => Err("the undeclared-call flow suppressed the malformed line".to_string()),
            }
        });
    }

    /// A table's failure classes exist at zero from the moment the table is published, so an
    /// alert on `failed` or `malformed` reads 0 and not "no data" until the first such output,
    /// and the first output counts on the published series.
    #[test]
    fn a_published_table_has_every_failure_class_at_zero_and_counts_on_it() {
        let scrape = rendered(|| {
            init_table_series("qwen3");
            record_failed("qwen3");
        });
        for outcome in OUTCOMES {
            let line = line_with(
                &scrape,
                OUTPUTS,
                &["table=\"qwen3\"", &format!("outcome=\"{outcome}\"")],
            )
            .expect(outcome);
            let expected = if outcome == "failed" { " 1" } else { " 0" };
            assert!(line.ends_with(expected), "{line}");
        }
        let reasons = scrape
            .lines()
            .filter(|line| line.starts_with(MALFORMED) && line.contains("table=\"qwen3\""))
            .count();
        assert_eq!(reasons, REASONS.len(), "{scrape}");
        for reason in REASONS {
            let line = line_with(
                &scrape,
                MALFORMED,
                &["table=\"qwen3\"", &format!("reason=\"{reason}\"")],
            )
            .expect(reason);
            assert!(line.ends_with(" 0"), "{line}");
        }
        let zero = |name: &str, labels: &[&str]| {
            line_with(&scrape, name, labels).is_some_and(|line| line.ends_with(" 0"))
        };
        assert!(zero(CALLS_SUPPRESSED, &["table=\"qwen3\""]), "{scrape}");
        assert!(
            zero(REFUSED, &["table=\"qwen3\"", "reason=\"tool_use_input\""]),
            "{scrape}"
        );
        assert!(
            zero(
                FALLBACK,
                &["table=\"qwen3\"", "reason=\"forced_call_without_tag\""]
            ),
            "{scrape}"
        );
        // The other fallback reasons count under `none`, never under a table.
        for reason in &FALLBACK_REASONS[..FORCED_CALL_WITHOUT_TAG] {
            assert!(
                line_with(
                    &scrape,
                    FALLBACK,
                    &["table=\"qwen3\"", &format!("reason=\"{reason}\"")]
                )
                .is_none(),
                "{reason}"
            );
        }
        // The histogram family is published with an empty distribution.
        assert!(
            zero(&format!("{MALFORMED_BYTES}_count"), &["table=\"qwen3\""]),
            "{scrape}"
        );
        // Nothing of another table.
        assert!(!scrape.contains("table=\"mistral\""), "{scrape}");
    }

    /// The requests no table reads are counted under `none` by four reasons known at start-up:
    /// their series exist at zero before any worker registers, and not the fifth, a table's own.
    #[test]
    fn the_startup_series_are_the_four_fallbacks_of_a_request_no_table_reads() {
        let scrape = rendered(init_startup_series);
        for reason in &FALLBACK_REASONS[..FORCED_CALL_WITHOUT_TAG] {
            let line = line_with(
                &scrape,
                FALLBACK,
                &["table=\"none\"", &format!("reason=\"{reason}\"")],
            )
            .expect(reason);
            assert!(line.ends_with(" 0"), "{line}");
        }
        assert!(
            line_with(&scrape, FALLBACK, &["reason=\"forced_call_without_tag\""]).is_none(),
            "{scrape}"
        );
        assert!(!scrape.contains(OUTPUTS), "{scrape}");
    }
}

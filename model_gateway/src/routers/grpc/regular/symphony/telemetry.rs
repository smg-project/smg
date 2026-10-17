//! What the reader records about an output: the counters per table, and one sampled line per
//! output with a malformed run, written under the request span (the log line carries the span's
//! request id; the trace exporter's own filter does not take this target).
//!
//! The classes, from the crate's events (`symphony::Event`):
//!
//! - `failed`: output the table could not read at all: the engine refused (`ParseError`); the
//!   request answers with an error;
//! - `malformed`: a call region the table read but gave up on (`Event::Malformed`), by its
//!   reason; the bytes reach the client as content, so nothing is lost and nothing says so;
//! - `dropped`: bytes the table treats as the template's wrapping (`Event::Dropped`): correct,
//!   expected on every output, not counted here;
//! - `forgiven`: a repair that keeps the answer's shape. Of those, the calls after the first
//!   dropped under `parallel_tool_calls: false` are counted (`suppressed_calls`); the refined
//!   finish, the empty arguments of a call without an `arguments` member and the chunking rule's
//!   stated exception are the crate's contract and are not.
//!
//! Cost: a clean output records one counter increment through a key built once per table (the
//! `metrics` macros build a key, with a `Vec` of labels, on every call), no allocation. A
//! malformed one adds a histogram observation, the per-reason counters, and, when the table's
//! token bucket allows it, one `warn` line: the only allocations. The run's text is model output
//! and is never logged, at any level: `at` and `bytes` locate it in the response for a reader with
//! access to it. Suppressed calls are counted and write no line.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        OnceLock,
    },
    time::Instant,
};

use axum::http::HeaderMap;
use metrics::{Key, Label, Level, Metadata};
use openai_protocol::smg::{ParseReport, ParseRun};
use parking_lot::Mutex;
use symphony::{formats::Table, DropReason, Event, MalformedReason};
use tracing::warn;

use crate::config::ParseFailures;

/// The request header that asks for the report: `on`, `off` or `detail`.
pub(crate) const REPORT_HEADER: &str = "x-smg-parse-report";
/// The request header that asks for strictness: `on` or `off`. An output that is not clean then
/// finishes with [`PARSE_FAILURE`] and carries its report whatever the report mode.
pub(crate) const STRICT_HEADER: &str = "x-smg-parse-strict";
/// The finish reason (and the Messages stop reason) of an output that is not clean under
/// strictness.
pub(crate) const PARSE_FAILURE: &str = "parse_failure";

/// Whether the request asks for strictness: its `X-SMG-Parse-Strict` header says `on`, and the
/// policy is not `silent` (which ignores the header). The error names a header value that is
/// neither `on` nor `off`, for the 400 response.
pub(crate) fn strict_mode(
    policy: ParseFailures,
    headers: Option<&HeaderMap>,
) -> Result<bool, String> {
    if policy == ParseFailures::Silent {
        return Ok(false);
    }
    match headers.and_then(|headers| headers.get(STRICT_HEADER)) {
        None => Ok(false),
        Some(value) => {
            let spelled = String::from_utf8_lossy(value.as_bytes());
            match spelled.trim().to_ascii_lowercase().as_str() {
                "on" => Ok(true),
                "off" => Ok(false),
                _ => Err(format!(
                    "{STRICT_HEADER} must be on or off, not {spelled:?}"
                )),
            }
        }
    }
}

/// What a response says about how its output was read: the `smg.parse` object next to its
/// usage (on a streamed choice's finish chunk; on a Messages answer's `message` and
/// `message_delta`). Decided once per request from the gateway's policy and the request's
/// `X-SMG-Parse-Report` header ([`report_mode`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Report {
    /// Nothing.
    #[default]
    Off,
    /// The report when the output is not clean.
    OnFailure,
    /// The report on every output, a clean one saying so.
    Always,
    /// The report on every output, with the runs the table dropped as wrapping too.
    Detail,
}

impl Report {
    /// Whether a report is wanted for an output that is clean (`not_clean` false) or not.
    pub(crate) fn wanted(self, not_clean: bool) -> bool {
        match self {
            Self::Off => false,
            Self::OnFailure => not_clean,
            Self::Always | Self::Detail => true,
        }
    }

    /// The mode under strictness: an output that is not clean is reported whatever the request
    /// asked, so `Off` becomes `OnFailure`; the others stay.
    pub(crate) fn at_least_on_failure(self) -> Self {
        match self {
            Self::Off => Self::OnFailure,
            other => other,
        }
    }

    /// Whether the runs are kept for the report at all.
    fn keeps_runs(self) -> bool {
        self != Self::Off
    }
}

/// The report mode of a request under `policy`: `surface` reports every output that is not
/// clean unless the header says `off`, and a clean one too when it says `on` or `detail`; `log`
/// reports when the header says `on` or `detail`; `silent` never, whatever the header. The
/// error names a header value that is none of the three, for the 400 response.
pub(crate) fn report_mode(
    policy: ParseFailures,
    headers: Option<&HeaderMap>,
) -> Result<Report, String> {
    if policy == ParseFailures::Silent {
        return Ok(Report::Off);
    }
    let asked = match headers.and_then(|headers| headers.get(REPORT_HEADER)) {
        None => None,
        Some(value) => {
            let spelled = String::from_utf8_lossy(value.as_bytes());
            match spelled.trim().to_ascii_lowercase().as_str() {
                "on" => Some(Report::Always),
                "detail" => Some(Report::Detail),
                "off" => Some(Report::Off),
                _ => {
                    return Err(format!(
                        "{REPORT_HEADER} must be on, off or detail, not {spelled:?}"
                    ))
                }
            }
        }
    };
    Ok(match (policy, asked) {
        (_, Some(asked)) => asked,
        (ParseFailures::Surface, None) => Report::OnFailure,
        (ParseFailures::Log | ParseFailures::Silent, None) => Report::Off,
    })
}

/// Outputs read to their end, by `table` and `outcome` (`clean`, `malformed`, `failed`).
pub(crate) const OUTPUTS: &str = "smg_symphony_outputs_total";
/// Malformed runs, by `table` and `reason`.
pub(crate) const MALFORMED: &str = "smg_symphony_malformed_total";
/// The bytes of the malformed runs of one output, by `table`; observed for an output with one.
pub(crate) const MALFORMED_BYTES: &str = "smg_symphony_malformed_bytes";
/// Calls after the first dropped under `parallel_tool_calls: false`, by `table`.
pub(crate) const CALLS_SUPPRESSED: &str = "smg_symphony_calls_suppressed_total";
/// The buckets of [`MALFORMED_BYTES`]: a marker's worth to a whole answer.
pub(crate) const MALFORMED_BYTES_BUCKETS: [f64; 9] = [
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
];

/// The format a response template compiles into carries this name (no registry table).
const TEMPLATE: &str = "response_template";
/// The `table` label of a format that is neither a registry table's nor the template's: a format
/// built by hand.
const OTHER_TABLE: &str = "other";

/// The `reason` labels, in the order of [`Tally::reasons`]: the three reasons that say what the
/// model got wrong, and `other` for the crate's syntax-level reasons (text where the call's syntax
/// has none, a tag out of place, a key without a value, ...), each spelled by the crate's sentence
/// on the sampled line.
const REASONS: [&str; 4] = [
    "unterminated_region",
    "invalid_arguments",
    "unknown_tool",
    "other",
];
/// The `outcome` labels, in the order of [`Series::outputs`].
const OUTCOMES: [&str; 3] = ["clean", "malformed", "failed"];
const CLEAN: usize = 0;
const MALFORMED_OUTCOME: usize = 1;
const FAILED: usize = 2;

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

/// The `reason` label of a malformed run, and its place in [`REASONS`].
fn reason_index(why: MalformedReason) -> usize {
    match why {
        MalformedReason::UnterminatedRegion => 0,
        MalformedReason::InvalidArguments => 1,
        MalformedReason::UnknownTool => 2,
        _ => 3,
    }
}

/// The `reason` label of a malformed run.
pub(crate) fn reason_label(why: MalformedReason) -> &'static str {
    REASONS[reason_index(why)]
}

/// The crate's sentence for a reason whose label says only `other`; the three typed labels spell
/// their reason themselves.
fn other_words(why: MalformedReason) -> Option<String> {
    (reason_index(why) == REASONS.len() - 1).then(|| why.sentence().to_string())
}

/// The bytes of a run the report quotes ([`ParseRun::excerpt`]): the client that asked for the
/// report gets this much of each run it names; no log line carries any of the output.
const EXCERPT_BYTES: usize = 120;

/// The first [`EXCERPT_BYTES`] of `text`, cut at a character boundary.
fn excerpt(text: &str) -> &str {
    if text.len() <= EXCERPT_BYTES {
        return text;
    }
    let mut end = EXCERPT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The bytes an event accounts for in the output.
fn bytes_of(event: &Event) -> usize {
    match event {
        Event::Content(text) | Event::Reasoning(text) => text.text.len(),
        Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.len(),
        Event::ToolCallStart { source, .. }
        | Event::ToolCallArguments { source, .. }
        | Event::ToolCallEnd { source, .. } => source.text.len(),
        Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => 0,
    }
}

/// Whose output a reader reports on, for the line a non-clean output writes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Who<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub choice: u32,
}

/// The first malformed run of an output, as the line shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FirstRun {
    /// Its `reason` label.
    pub reason: &'static str,
    /// The crate's sentence for the reason (`MalformedReason::sentence`).
    pub why: &'static str,
    /// Where it starts in the output, in bytes.
    pub at: usize,
    /// Its length in bytes.
    pub bytes: usize,
}

/// What the reader has seen of one output so far, counted as the events are drained.
#[derive(Debug, Default)]
pub(crate) struct Tally {
    /// The bytes of the events seen so far: where the next event's bytes start.
    at: usize,
    /// Malformed runs by reason, in the order of [`REASONS`].
    reasons: [u32; 4],
    malformed_bytes: u64,
    first: Option<FirstRun>,
    suppressed_calls: u32,
    tool_calls: u32,
    /// What the response reports, and the runs kept for it: every malformed run, and under
    /// `Detail` every dropped one, in the output's order. Nothing is kept under `Off`.
    report: Report,
    /// `--parse-failures silent`: the counters alone, no line and no excerpt event.
    silent: bool,
    malformed_runs: Vec<ParseRun>,
    dropped_runs: Vec<ParseRun>,
}

/// The `reason` of a dropped run in the report.
fn drop_label(why: DropReason) -> &'static str {
    match why {
        DropReason::ControlToken => "control_token",
        DropReason::Wrapper => "wrapper",
        DropReason::Whitespace => "whitespace",
    }
}

impl Tally {
    /// A tally that keeps what `report` needs.
    pub(crate) fn reporting(report: Report) -> Self {
        Self {
            report,
            ..Self::default()
        }
    }

    /// Under `--parse-failures silent`: the counters alone, no line and no excerpt event.
    pub(crate) fn silenced(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    /// Accounts for `event`, the next of the output; `first_call_only` says whether the reader
    /// drops the calls after the first, which are then counted as suppressed.
    pub(crate) fn see(&mut self, event: &Event, first_call_only: bool) {
        match event {
            Event::Malformed { text, why } => {
                self.reasons[reason_index(*why)] += 1;
                self.malformed_bytes += text.text.len() as u64;
                if self.first.is_none() {
                    self.first = Some(FirstRun {
                        reason: reason_label(*why),
                        why: why.sentence(),
                        at: self.at,
                        bytes: text.text.len(),
                    });
                }
                if self.report.keeps_runs() {
                    self.malformed_runs.push(ParseRun {
                        choice: 0,
                        reason: reason_label(*why).to_string(),
                        why: other_words(*why),
                        at: self.at as u64,
                        bytes: text.text.len() as u64,
                        excerpt: excerpt(&text.text).to_string(),
                    });
                }
            }
            Event::Dropped { text, why } if self.report == Report::Detail => {
                self.dropped_runs.push(ParseRun {
                    choice: 0,
                    reason: drop_label(*why).to_string(),
                    why: None,
                    at: self.at as u64,
                    bytes: text.text.len() as u64,
                    excerpt: excerpt(&text.text).to_string(),
                });
            }
            Event::ToolCallStart { index, .. } if first_call_only && *index > 0 => {
                self.suppressed_calls += 1;
            }
            Event::Finish { tool_calls, .. } => self.tool_calls = *tool_calls,
            _ => {}
        }
        self.at += bytes_of(event);
    }

    /// Whether the table gave up on a run of the output.
    pub(crate) fn has_malformed(&self) -> bool {
        self.malformed() > 0
    }

    /// Whether the output was not clean: a malformed run or a suppressed call.
    fn not_clean(&self) -> bool {
        self.malformed() > 0 || self.suppressed_calls > 0
    }

    /// The report of the output read under `table` as the choice at `choice`, when the mode
    /// wants one for it.
    pub(crate) fn parse_report(&self, table: &'static str, choice: u32) -> Option<ParseReport> {
        if !self.report.wanted(self.not_clean()) {
            return None;
        }
        let numbered = |runs: &[ParseRun]| {
            runs.iter()
                .cloned()
                .map(|run| ParseRun { choice, ..run })
                .collect()
        };
        Some(ParseReport {
            parser: "symphony".to_string(),
            table: Some(table.to_string()),
            outcome: Some(
                if self.malformed() > 0 {
                    "malformed"
                } else {
                    "clean"
                }
                .to_string(),
            ),
            malformed: numbered(&self.malformed_runs),
            dropped: numbered(&self.dropped_runs),
            suppressed_calls: self.suppressed_calls,
        })
    }

    /// Malformed runs seen.
    fn malformed(&self) -> u32 {
        self.reasons.iter().sum()
    }

    /// The distinct reason labels seen, in a fixed order, joined with a comma.
    fn reasons_seen(&self) -> String {
        let mut seen = String::new();
        for (count, label) in self.reasons.iter().zip(REASONS) {
            if *count > 0 {
                if !seen.is_empty() {
                    seen.push(',');
                }
                seen.push_str(label);
            }
        }
        seen
    }

    /// Records the output under `table`, read to its end: the counters, and the line when the
    /// output has a malformed run and the table's bucket allows one. Suppressed calls are counted
    /// and not written: the request asked for them to be dropped.
    pub(crate) fn report(&self, table: &'static str, who: Who<'_>) {
        let series = series(table);
        let malformed = self.malformed();
        let outcome = if malformed > 0 {
            MALFORMED_OUTCOME
        } else {
            CLEAN
        };
        increment(&series.outputs[outcome], 1);
        if malformed == 0 && self.suppressed_calls == 0 {
            return;
        }
        for (count, key) in self.reasons.iter().zip(&series.malformed) {
            if *count > 0 {
                increment(key, u64::from(*count));
            }
        }
        if malformed > 0 {
            observe(&series.malformed_bytes, self.malformed_bytes as f64);
        }
        if self.suppressed_calls > 0 {
            increment(&series.calls_suppressed, u64::from(self.suppressed_calls));
        }
        if malformed == 0 || self.silent {
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
            suppressed_lines,
            "Symphony did not read the whole output as content, reasoning and calls"
        );
        // The run's text is model output and may carry the user's data: it is not logged, at
        // any level; `at` and `bytes` locate it in the response for a reader with access to it.
    }
}

/// Records an output the table refused (`ParseError`): the request fails with it.
pub(crate) fn record_failed(table: &'static str) {
    increment(&series(table).outputs[FAILED], 1);
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
    outputs: [Key; 3],
    /// [`MALFORMED`] by reason, in the order of [`REASONS`].
    malformed: [Key; 4],
    malformed_bytes: Key,
    calls_suppressed: Key,
    sampler: Sampler,
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
            sampler: Sampler::new(),
        }
    }
}

/// Every table's series: the registry's tables in their order, then the template's format,
/// then the rest ([`OTHER_TABLE`]).
static SERIES: OnceLock<Vec<Series>> = OnceLock::new();

/// The series of the table labelled `table`.
fn series(table: &'static str) -> &'static Series {
    let all = SERIES.get_or_init(|| {
        Table::ALL
            .iter()
            .map(|table| table.name())
            .chain([TEMPLATE, OTHER_TABLE])
            .map(Series::new)
            .collect()
    });
    let index = Table::ALL
        .iter()
        .position(|candidate| candidate.name() == table)
        .unwrap_or_else(|| {
            if table == TEMPLATE {
                Table::ALL.len()
            } else {
                Table::ALL.len() + 1
            }
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
    use metrics_exporter_prometheus::PrometheusBuilder;
    use symphony::{DropReason, FinishReason, Text};

    use super::*;

    fn malformed(text: &str, why: MalformedReason) -> Event {
        Event::Malformed {
            text: Text::uncounted(text),
            why,
        }
    }

    fn call_start(index: u32, source: &str) -> Event {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: "lookup".into(),
            source: Text::uncounted(source),
        }
    }

    fn finish(tool_calls: u32) -> Event {
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
    fn the_reports_excerpt_is_bounded_at_a_character_boundary() {
        assert_eq!(excerpt("short"), "short");
        let exact = "x".repeat(EXCERPT_BYTES);
        assert_eq!(excerpt(&exact), exact);
        // Every character is three bytes; 120 is not a multiple of three.
        let wide = "日".repeat(60);
        let cut = excerpt(&wide);
        assert!(
            cut.len() <= EXCERPT_BYTES && cut.len() > EXCERPT_BYTES - 3,
            "{}",
            cut.len()
        );
        assert!(cut.chars().all(|c| c == '日'));
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
            "unterminated_region,invalid_arguments,other"
        );
        assert_eq!(tally.suppressed_calls, 0);
    }

    #[test]
    fn the_first_run_of_another_reason_carries_its_sentence() {
        let mut tally = Tally::default();
        tally.see(
            &malformed("<|end|>", MalformedReason::NameWithoutACall),
            false,
        );
        let first = tally.first.as_ref().expect("a first run");
        assert_eq!(first.reason, "other");
        assert_eq!(first.why, "a name without a call");
        assert_eq!(first.at, 0);
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

    /// The rendered scrape after `record` ran under a recorder of its own.
    fn rendered(record: impl FnOnce()) -> String {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, record);
        handle.render()
    }

    fn line_with<'a>(scrape: &'a str, name: &str, labels: &[&str]) -> Option<&'a str> {
        scrape
            .lines()
            .find(|line| line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
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

    fn headers(value: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(value) = value {
            headers.insert(REPORT_HEADER, value.parse().expect("a header value"));
        }
        headers
    }

    #[test]
    fn the_report_mode_follows_the_policy_and_the_header() {
        use ParseFailures::{Log, Silent, Surface};
        let mode = |policy, header| report_mode(policy, Some(&headers(header))).expect("a mode");
        assert_eq!(mode(Log, None), Report::Off);
        assert_eq!(mode(Log, Some("on")), Report::Always);
        assert_eq!(mode(Log, Some("detail")), Report::Detail);
        assert_eq!(mode(Log, Some("off")), Report::Off);
        assert_eq!(mode(Surface, None), Report::OnFailure);
        assert_eq!(mode(Surface, Some("on")), Report::Always);
        assert_eq!(mode(Surface, Some("Detail")), Report::Detail);
        assert_eq!(mode(Surface, Some("off")), Report::Off);
        assert_eq!(mode(Silent, None), Report::Off);
        assert_eq!(mode(Silent, Some("on")), Report::Off);
        assert_eq!(mode(Silent, Some("detail")), Report::Off);
        assert_eq!(report_mode(Log, None).expect("a mode"), Report::Off);
        let why = report_mode(Log, Some(&headers(Some("yes")))).expect_err("refused");
        assert_eq!(
            why,
            "x-smg-parse-report must be on, off or detail, not \"yes\""
        );
        // The header value is not read under `silent`, so a bad one is not an error there.
        assert_eq!(mode(Silent, Some("yes")), Report::Off);
    }

    fn strict_headers(value: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(value) = value {
            headers.insert(STRICT_HEADER, value.parse().expect("a header value"));
        }
        headers
    }

    #[test]
    fn strictness_follows_the_header_unless_the_policy_is_silent() {
        use ParseFailures::{Log, Silent, Surface};
        let strict = |policy, header| {
            strict_mode(policy, Some(&strict_headers(header))).expect("a decision")
        };
        assert!(!strict(Log, None));
        assert!(strict(Log, Some("on")));
        assert!(strict(Log, Some(" ON ")));
        assert!(!strict(Log, Some("off")));
        assert!(strict(Surface, Some("on")));
        assert!(!strict(Silent, Some("on")));
        assert!(!strict_mode(Log, None).expect("a decision"));
        let why = strict_mode(Log, Some(&strict_headers(Some("yes")))).expect_err("refused");
        assert_eq!(why, "x-smg-parse-strict must be on or off, not \"yes\"");
        // Not read under `silent`, so a bad value is not an error there.
        assert!(!strict(Silent, Some("yes")));
    }

    #[test]
    fn strictness_reports_a_failure_whatever_the_mode_asked() {
        assert_eq!(Report::Off.at_least_on_failure(), Report::OnFailure);
        assert_eq!(Report::OnFailure.at_least_on_failure(), Report::OnFailure);
        assert_eq!(Report::Always.at_least_on_failure(), Report::Always);
        assert_eq!(Report::Detail.at_least_on_failure(), Report::Detail);
    }

    #[test]
    fn a_report_is_wanted_as_its_mode_says() {
        assert!(!Report::Off.wanted(true));
        assert!(!Report::OnFailure.wanted(false));
        assert!(Report::OnFailure.wanted(true));
        assert!(Report::Always.wanted(false));
        assert!(Report::Detail.wanted(false));
    }

    fn seen(report: Report) -> Tally {
        let mut tally = Tally::reporting(report);
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
            malformed("x", MalformedReason::NameWithoutACall),
            call_start(0, "<tool_call>"),
            call_start(1, "<tool_call>"),
            finish(2),
        ];
        for event in &events {
            tally.see(event, true);
        }
        tally
    }

    #[test]
    fn the_report_lists_the_malformed_runs_with_their_choice_and_place() {
        let report = seen(Report::Always)
            .parse_report("qwen3", 2)
            .expect("a report");
        assert_eq!(report.parser, "symphony");
        assert_eq!(report.table.as_deref(), Some("qwen3"));
        assert_eq!(report.outcome.as_deref(), Some("malformed"));
        assert_eq!(report.suppressed_calls, 1);
        assert!(report.dropped.is_empty());
        assert_eq!(report.malformed.len(), 2);
        let first = &report.malformed[0];
        assert_eq!(
            (first.choice, first.reason.as_str(), first.at, first.bytes),
            (2, "invalid_arguments", 17, 12)
        );
        assert_eq!(first.why, None);
        assert_eq!(first.excerpt, "\nnot a call\n");
        let second = &report.malformed[1];
        assert_eq!(second.reason, "other");
        assert_eq!(second.why.as_deref(), Some("a name without a call"));
        assert_eq!(second.at, 17 + 12 + "</tool_call>".len() as u64);
    }

    #[test]
    fn detail_adds_the_dropped_runs_in_order() {
        let report = seen(Report::Detail)
            .parse_report("qwen3", 0)
            .expect("a report");
        assert_eq!(report.malformed.len(), 2);
        let dropped: Vec<(&str, u64)> = report
            .dropped
            .iter()
            .map(|run| (run.reason.as_str(), run.at))
            .collect();
        assert_eq!(dropped, vec![("wrapper", 6), ("wrapper", 29)]);
    }

    #[test]
    fn off_keeps_no_runs_and_gives_no_report() {
        let tally = seen(Report::Off);
        assert!(tally.malformed_runs.is_empty() && tally.dropped_runs.is_empty());
        assert_eq!(tally.parse_report("qwen3", 0), None);
        // The counters still see everything.
        assert_eq!(tally.malformed(), 2);
    }

    #[test]
    fn on_failure_reports_only_an_output_that_is_not_clean() {
        let mut clean = Tally::reporting(Report::OnFailure);
        clean.see(&Event::Content(Text::uncounted("Hello")), false);
        clean.see(&finish(0), false);
        assert_eq!(clean.parse_report("qwen3", 0), None);
        let mut suppressed = Tally::reporting(Report::OnFailure);
        suppressed.see(&call_start(0, "<tool_call>"), true);
        suppressed.see(&call_start(1, "<tool_call>"), true);
        suppressed.see(&finish(2), true);
        let report = suppressed.parse_report("qwen3", 0).expect("a report");
        assert_eq!(report.outcome.as_deref(), Some("clean"));
        assert_eq!(report.suppressed_calls, 1);
        let mut always = Tally::reporting(Report::Always);
        always.see(&Event::Content(Text::uncounted("Hello")), false);
        always.see(&finish(0), false);
        let report = always.parse_report("plain", 0).expect("a report");
        assert_eq!(report.outcome.as_deref(), Some("clean"));
        assert!(report.malformed.is_empty());
    }

    #[tracing_test::traced_test]
    #[test]
    fn under_silent_a_malformed_output_is_counted_and_writes_nothing() {
        let scrape = rendered(|| {
            let mut tally = Tally::reporting(Report::Off).silenced(true);
            tally.see(
                &malformed("\nnot a call\n", MalformedReason::InvalidArguments),
                false,
            );
            tally.see(&finish(0), false);
            tally.report(
                "plain",
                Who {
                    request_id: "req-silent",
                    model: "the-model",
                    choice: 0,
                },
            );
        });
        assert!(
            line_with(
                &scrape,
                OUTPUTS,
                &["table=\"plain\"", "outcome=\"malformed\""]
            )
            .is_some(),
            "{scrape}"
        );
        assert!(
            line_with(&scrape, MALFORMED, &["table=\"plain\""]).is_some(),
            "{scrape}"
        );
        assert!(!logs_contain("req-silent"));
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
}

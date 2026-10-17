//! What the reader records about an output: the counters per table, and one sampled line per
//! output that is not clean, written under the request span so a tracing exporter keeps it as a
//! span event as well.
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
//! token bucket allows it, one `warn` line, with the first run's bounded excerpt on a `debug` event
//! beside it: the only allocations. Suppressed calls are counted and write no line.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        OnceLock,
    },
    time::Instant,
};

use metrics::{Key, Label, Level, Metadata};
use parking_lot::Mutex;
use symphony::{formats::Table, Event, MalformedReason};
use tracing::{debug, warn};

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
/// How many bytes of the first malformed run the line shows.
pub(crate) const EXCERPT_BYTES: usize = 120;

/// The `reason` labels, in the order of [`Tally::reasons`].
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
fn reason_index(why: &MalformedReason) -> usize {
    match why {
        MalformedReason::UnterminatedRegion => 0,
        MalformedReason::InvalidArguments => 1,
        MalformedReason::UnknownTool => 2,
        MalformedReason::Other(_) => 3,
    }
}

/// The `reason` label of a malformed run.
pub(crate) fn reason_label(why: &MalformedReason) -> &'static str {
    REASONS[reason_index(why)]
}

/// The first [`EXCERPT_BYTES`] of `text`, cut at a character boundary.
pub(crate) fn excerpt(text: &str) -> &str {
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
    /// The crate's reason: the label, or the text of `Other`.
    pub why: String,
    /// Where it starts in the output, in bytes.
    pub at: usize,
    /// Its length in bytes.
    pub bytes: usize,
    /// Its first bytes ([`EXCERPT_BYTES`]).
    pub excerpt: String,
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
}

impl Tally {
    /// Accounts for `event`, the next of the output; `first_call_only` says whether the reader
    /// drops the calls after the first, which are then counted as suppressed.
    pub(crate) fn see(&mut self, event: &Event, first_call_only: bool) {
        match event {
            Event::Malformed { text, why } => {
                self.reasons[reason_index(why)] += 1;
                self.malformed_bytes += text.text.len() as u64;
                if self.first.is_none() {
                    self.first = Some(FirstRun {
                        reason: reason_label(why),
                        why: match why {
                            MalformedReason::Other(text) => text.to_string(),
                            typed => reason_label(typed).to_string(),
                        },
                        at: self.at,
                        bytes: text.text.len(),
                        excerpt: excerpt(&text.text).to_string(),
                    });
                }
            }
            Event::ToolCallStart { index, .. } if first_call_only && *index > 0 => {
                self.suppressed_calls += 1;
            }
            Event::Finish { tool_calls, .. } => self.tool_calls = *tool_calls,
            _ => {}
        }
        self.at += bytes_of(event);
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
        if malformed == 0 {
            return;
        }
        let Some(suppressed_lines) = series.sampler.allow() else {
            return;
        };
        let outcome = OUTCOMES[outcome];
        let (reason, why, at, bytes, excerpt) = match &self.first {
            Some(first) => (
                first.reason,
                first.why.as_str(),
                first.at,
                first.bytes,
                first.excerpt.as_str(),
            ),
            None => ("", "", 0, 0, ""),
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
        // The run's text is model output: a debug event beside the line, not the line itself,
        // which the default level and the trace exporters see.
        debug!(
            target: "smg::symphony",
            request_id = who.request_id,
            choice = who.choice,
            table,
            reason,
            at,
            excerpt,
            "the first malformed run's text"
        );
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
    fn the_excerpt_is_bounded_at_a_character_boundary() {
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
            malformed(
                "x",
                MalformedReason::Other("a tool call without a name".into()),
            ),
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
                why: "invalid_arguments".into(),
                at: "Hello <tool_call>".len(),
                bytes: 12,
                excerpt: "\nnot a call\n".into(),
            })
        );
        assert_eq!(
            tally.reasons_seen(),
            "unterminated_region,invalid_arguments,other"
        );
        assert_eq!(tally.suppressed_calls, 0);
    }

    #[test]
    fn the_first_run_of_another_reason_carries_its_text() {
        let mut tally = Tally::default();
        tally.see(
            &malformed(
                "<|end|>",
                MalformedReason::Other("a name without a call".into()),
            ),
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
        logs_assert(|lines: &[&str]| {
            let line = lines
                .iter()
                .find(|line| line.contains("Symphony did not read the whole output"))
                .ok_or("no warn line")?;
            if line.contains("excerpt=") {
                return Err("the excerpt is on the warn line".to_string());
            }
            let event = lines
                .iter()
                .find(|line| line.contains("excerpt=\"\\nnot a call\\n\""))
                .ok_or("no excerpt event")?;
            if !event.contains("DEBUG") {
                return Err(format!("the excerpt is not a debug event: {event}"));
            }
            Ok(())
        });
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

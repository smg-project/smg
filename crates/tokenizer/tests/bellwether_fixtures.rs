//! The tokenizer against the bellwether reference fixtures, in both
//! directions: text to ids, and ids to text one token at a time.
//!
//! bellwether (smg-project/bellwether) records, for each model at a pinned
//! Hugging Face revision, what the checkpoint's own chat template and
//! tokenizer produce. This test reads the tokenizer's part of two fixture
//! kinds:
//!
//! - A `render` line carries a rendered prompt and its token ids, the flat
//!   encode of the text without added special tokens. Encoding the text the
//!   way the gateway encodes a rendered prompt must give those ids.
//! - A `parse` line carries an output's token ids and the text each id
//!   contributes under the reference's incremental decode (tokenizers'
//!   `DecodeStream`, special tokens kept): an empty piece for a token that
//!   does not complete a character, the whole character on the token that
//!   completes it. Feeding the ids one at a time through [`Sequence`], the
//!   incremental decoder behind the gateway's stop-sequence decoder, must give
//!   the same piece for every id, and the pieces joined must give the output
//!   text.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the tree `bellwether
//! unpack` writes (`fixtures-plain/` unless `--out` names another), where
//! every fixture set is plain JSON Lines beside each model's `manifest.toml`
//! and `sets.toml`; without it the test prints a skip notice and passes.
//! The tree the lists below are judged against is the one CI unpacks,
//! bellwether at the commit `.github/versions/bellwether.ref` pins: a
//! case's reference moves between bellwether commits, so an entry taken
//! from another checkout's tree fails the run there as stale, or leaves a
//! difference unlisted.
//! bellwether's own `fixtures/` directory keeps a benchmark set as
//! `<set>.jsonl.zst` in Git LFS, so a compressed set or a Git LFS pointer under
//! the root fails the run and says to unpack first. Each case is compared as
//! its line is read, and the cases read from each set must number what the
//! model's `sets.toml` lists for it; a set the table lists that the tree lacks,
//! or a set file the table does not list, fails the run too, so it cannot pass
//! on part of the fixtures.
//!
//! The run prints one line per model: for each direction the cases compared,
//! matching and differing, how many of the differing are listed and
//! unlisted, and the model's stale entries of the list; under it the first
//! unlisted differences of the model, up to [`SHOWN`]; then one count of the
//! models without sets, and the totals. The per-case report, each set read
//! with its count and each difference with its description and its reason
//! when it is listed, followed by the summary and the failures, goes to a
//! file: `bellwether_fixtures.txt` under `BELLWETHER_REPORT_DIR`, or under
//! the build's `target/tmp` when the variable is not set, and the run says
//! where. `BELLWETHER_VERBOSE=1` prints the per-case report on stdout too,
//! as the run did before it went to a file.
//!
//! Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`: `tokenizer_config.json`
//! and the vocabulary, `tokenizer.json` or, for checkpoints that ship a
//! tiktoken vocabulary instead (Kimi), `tiktoken.model`, plus `config.json`
//! when the checkpoint has one. That download needs the TLS backend a
//! workspace build enables: under `cargo test -p llm-tokenizer` alone the
//! crate's `reqwest` dev-dependency has none, so with a cold cache run it
//! from the workspace or fill the Hugging Face cache first. A model whose
//! tokenizer does not load is listed when the run fails at the end, after
//! every other model has been compared.
//!
//! A difference is a finding, not something to hide: every known one is listed
//! in [`KNOWN_DIFFERENCES`] with its reason and where it is tracked, by id or
//! by a prefix when a cause covers a model or a set wholesale; the run fails
//! on any other, on a listed case that starts matching, and on a listed case
//! that the loaded fixtures no longer contain, so the list cannot rot.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
    ops::Bound,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer, MockTokenizer, Sequence};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

use crate::common::hub_fetch::{
    answered, cached_download, fetch, rate_limit_wait, RATE_LIMITED_TRIES, RATE_LIMIT_WINDOW,
    TRANSPORT_ERROR_RETRIES, TRANSPORT_ERROR_WAIT,
};

mod common;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";
/// Set to anything but empty or `0`, the per-case report is printed on stdout
/// as well.
const VERBOSE_ENV: &str = "BELLWETHER_VERBOSE";
/// The directory the per-case report is written to; the build's `target/tmp`
/// when it is not set.
const REPORT_DIR_ENV: &str = "BELLWETHER_REPORT_DIR";
/// The per-case report's file name under that directory.
const REPORT_FILE: &str = "bellwether_fixtures.txt";
/// How many unlisted differences of one model stdout shows, and how many
/// lines of a failure; the rest are in the per-case report.
const SHOWN: usize = 20;
/// The fixture kinds this test reads, each a directory of sets beside a
/// model's manifest.
const KINDS: [&str; 2] = ["render", "parse"];
/// The set bellwether records by hand for every model, whose cases carry
/// names of their own (`no-generation-prompt`) where a benchmark set's are
/// numbered (`gsm8k-17`): the tree holds it for every model, so a case name's
/// entry is stale when it is not among the cases.
const HAND_WRITTEN_SET: &str = "common";
/// How a file Git LFS has not fetched begins: the pointer stands where the
/// content would be.
const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// Cases known to differ from the reference, each with the reason and where
/// it is tracked: a fixture id, `<slug>/render/<name>` or `<slug>/parse/<name>`;
/// a prefix ending in `*`, which stands for every case whose id begins with
/// what is before it (`<slug>/render/*` for a model's whole render side,
/// `<slug>/render/mgsm-th-*` for one of its sets); or a suffix beginning with
/// `*`, which stands for the case of that name of every model
/// (`*/parse/content-with-marker-in-code-fence`); or a set by its file,
/// `<slug>/<kind>/<set>.jsonl`, for a benchmark set whole (its case ids need
/// not begin with its name). A prefix is for a cause that covers a model or a
/// family of sets wholesale, a reference
/// recorded with another pre-tokenizer for instance, a suffix for one that
/// covers a hand-written case across the models, and each is held to the rule
/// an id is: the run fails when no loaded case under it
/// differs any more (a differing case counts for the entry of its own id, else
/// for the one entry that covers it and for none when several do, so an entry
/// does not outlive its cause on the cases another entry names), and when
/// the tree holds its set and no loaded case is
/// under it; the entries of a benchmark set the tree omits (`bellwether unpack
/// --sample-sets` writes every Nth) are left alone, an id of one of its cases
/// among them.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[];

/// Models whose tokenizer is known not to load, with the reason and where it
/// is tracked, so the run compares the others: an unlisted model that does
/// not load fails the run, and so does a listed one whose tokenizer loads
/// now, so the list cannot rot.
const KNOWN_UNLOADED: &[(&str, &str)] = &[
    (
        "c4ai-command-a-03-2025",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "c4ai-command-r-v01",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "command-a-reasoning-08-2025",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "functiongemma-270m-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-1.1-2b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-2-2b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3-12b-it-qat-q4-0-unquantized",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3-1b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3-1b-it-qat-q4-0-unquantized",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3-270m-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3-4b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "gemma-3n-e2b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-3.2-11b-vision-instruct",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-3.2-1b-instruct",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-4-maverick-17b-128e-instruct",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-4-scout-17b-16e",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-4-scout-17b-16e-instruct",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-guard-3-1b",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "llama-guard-4-12b",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "medgemma-27b-it",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
    (
        "skywork-or1-32b-preview",
        "a gated repository: the job runs without a Hugging Face token, so the tokenizer \
         files behind the gate are not fetched (bellwether recorded the group with its \
         own secret)",
    ),
];

/// The reason `known` lists for the case `id` of the set `set`: the entry
/// that is the id itself, else the first entry that covers it; none when the
/// case is not listed.
fn known_reason<'a>(known: &BTreeMap<&'a str, &'a str>, id: &str, set: &str) -> Option<&'a str> {
    known.get(id).copied().or_else(|| {
        known
            .iter()
            .find(|(entry, _)| Listed::of(entry).covers(id, set))
            .map(|(_, reason)| *reason)
    })
}

/// The entry of `known` a differing case `id` of the set `set` counts for
/// when the list is held to its rule: the entry that is the id itself, else
/// the one entry that covers it; none when the case is not listed, and none
/// when two or more entries cover it, since such a case differs for either
/// entry's cause and must keep neither alive (a model whose whole render side
/// is listed has its `no-generation-prompt` case under the case-name entry as
/// well: the header the gateway appends and the model's own cause both stand
/// in it, so it says nothing about which of the two causes is gone).
fn counting_entry<'a>(known: &BTreeMap<&'a str, &'a str>, id: &str, set: &str) -> Option<&'a str> {
    if let Some((entry, _)) = known.get_key_value(id) {
        return Some(*entry);
    }
    let mut covering = known
        .keys()
        .copied()
        .filter(|entry| Listed::of(entry).covers(id, set));
    let first = covering.next()?;
    covering.next().is_none().then_some(first)
}

/// What an entry of the list stands for: one fixture id; a prefix, the part
/// before its trailing `*`, for a cause that covers a model or a family of
/// sets wholesale; a set, named by its file `<slug>/<kind>/<set>.jsonl`, for a
/// cause that covers a benchmark set, whose case ids need not begin with its
/// name; or a suffix, the part after its leading `*`, for a cause that covers
/// one case of every model (`*/render/no-generation-prompt`).
#[derive(Clone, Copy)]
enum Listed<'e> {
    Id(&'e str),
    Prefix(&'e str),
    Set(&'e str),
    Suffix(&'e str),
}

impl<'e> Listed<'e> {
    fn of(entry: &'e str) -> Self {
        if let Some(set) = entry.strip_suffix(".jsonl") {
            Self::Set(set)
        } else if let Some(prefix) = entry.strip_suffix('*') {
            Self::Prefix(prefix)
        } else if let Some(suffix) = entry.strip_prefix('*') {
            Self::Suffix(suffix)
        } else {
            Self::Id(entry)
        }
    }

    /// Whether the entry lists the case `id` of the set `set`
    /// (`<slug>/<kind>/<set>`).
    fn covers(self, id: &str, set: &str) -> bool {
        match self {
            Self::Id(entry) => id == entry,
            Self::Prefix(prefix) => id.starts_with(prefix),
            Self::Set(entry) => set == entry,
            Self::Suffix(suffix) => id.ends_with(suffix),
        }
    }
}

/// `fixtures/<slug>/manifest.toml`: the model and the revision its fixtures
/// were recorded at.
#[derive(Deserialize)]
struct Manifest {
    model: String,
    revision: String,
}

/// The fields this test reads from one line of
/// `fixtures/<slug>/render/<set>.jsonl`, bellwether's `case.schema.json`.
#[derive(Deserialize)]
struct RenderCase {
    id: String,
    kind: String,
    model: String,
    reference: RenderReference,
}

#[derive(Deserialize)]
struct RenderReference {
    input_ids: Vec<u32>,
    text: String,
    #[serde(default)]
    provenance: Value,
}

/// The fields this test reads from one line of
/// `fixtures/<slug>/parse/<set>.jsonl`, bellwether's `case.schema.json`.
#[derive(Deserialize)]
struct ParseCase {
    id: String,
    kind: String,
    model: String,
    output_ids: Vec<u32>,
    output_pieces: Vec<String>,
    reference: ParseReference,
}

#[derive(Deserialize)]
struct ParseReference {
    text: String,
    #[serde(default)]
    provenance: Value,
}

/// One `[<kind>.<set>]` table of `<slug>/sets.toml`, which `bellwether record`
/// writes beside the sets and `bellwether unpack` copies along: how many cases
/// the set holds. Its other fields (`form`, `rejected`, `plain_bytes`,
/// `plain_sha256`) are not read here.
#[derive(Deserialize)]
struct SetTable {
    cases: usize,
}

/// What one direction compared for one model.
#[derive(Default)]
struct Tally {
    loaded: bool,
    cases: usize,
    ids: usize,
    differ: usize,
    /// How many of the differing cases [`KNOWN_DIFFERENCES`] lists.
    listed: usize,
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.loaded {
            return f.write_str("no fixtures");
        }
        write!(
            f,
            "{} cases ({} ids) compared, {} match, {} differ, {} listed, {} unlisted",
            self.cases,
            self.ids,
            self.cases - self.differ,
            self.differ,
            self.listed,
            self.differ - self.listed
        )
    }
}

/// The first unlisted differences of a model, as stdout shows them.
#[derive(Default)]
struct Shown {
    lines: String,
    unlisted: usize,
}

impl Shown {
    /// Count one unlisted difference, and keep its line while fewer than
    /// [`SHOWN`] are kept.
    fn unlisted(&mut self, line: &str) {
        self.unlisted += 1;
        if self.unlisted <= SHOWN {
            self.lines.push_str(line);
        }
    }

    /// The lines kept, and how many more the per-case report has.
    fn text(mut self, slug: &str) -> String {
        if self.unlisted > SHOWN {
            self.lines.push_str(&format!(
                "  ... and {} more unlisted differences of {slug}, every one in the per-case \
                 report\n",
                self.unlisted - SHOWN
            ));
        }
        self.lines
    }
}

/// The run's stdout, through `print!`, which the test harness captures as it
/// does every test's output unless it runs with `--nocapture`.
struct Stdout;

impl Write for Stdout {
    #[expect(
        clippy::print_stdout,
        reason = "the per-model lines are test diagnostic output"
    )]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        print!("{}", String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

/// Where a run writes: the per-case report to `report`, and to `stdout` as
/// well when the run is verbose; else `stdout` gets one line per model, the
/// first unlisted differences of each, and the totals.
struct Output<'a> {
    stdout: &'a mut dyn Write,
    report: &'a mut dyn Write,
    verbose: bool,
}

impl Output<'_> {
    /// Write what comparing one model gave.
    fn model(&mut self, compared: &Compared) -> io::Result<()> {
        self.report.write_all(compared.lines.as_bytes())?;
        if self.verbose {
            return self.stdout.write_all(compared.lines.as_bytes());
        }
        if compared.has_sets {
            writeln!(self.stdout, "{}", compared.summary)?;
            self.stdout.write_all(compared.shown.as_bytes())?;
        }
        Ok(())
    }

    /// Write the summary, with the line of every model, and the totals.
    fn finish(
        &mut self,
        summaries: &[String],
        without_sets: usize,
        totals: &str,
    ) -> io::Result<()> {
        let mut summary = String::from("summary:\n");
        for line in summaries {
            summary.push_str(&format!("  {line}\n"));
        }
        summary.push_str(totals);
        summary.push('\n');
        self.report.write_all(summary.as_bytes())?;
        if self.verbose {
            return self.stdout.write_all(summary.as_bytes());
        }
        if without_sets > 0 {
            writeln!(
                self.stdout,
                "{without_sets} models without render or parse sets"
            )?;
        }
        writeln!(self.stdout, "{totals}")
    }
}

/// What a run compared, and what it could not.
#[derive(Default)]
struct Report {
    /// `<slug>/<kind>` of each kind whose sets were read.
    loaded_dirs: BTreeSet<String>,
    /// `<slug>/<kind>/<set>` of each set read.
    sets_read: BTreeSet<String>,
    /// The set of each case that differs, by id.
    set_of_difference: BTreeMap<String, String>,
    /// The id of every case compared.
    seen: BTreeSet<String>,
    /// Why each case that differs from the reference differs, by id.
    differences: BTreeMap<String, String>,
    /// Each model whose tokenizer did not load, and why.
    unloaded: Vec<(String, String)>,
    /// Each set whose cases read differ from what its model's `sets.toml`
    /// lists: `<slug>/<kind>/<set>`, the cases read (none when the tree has no
    /// file for the set) and the cases listed (none when the table does not
    /// list it).
    mismatches: Vec<(String, Option<usize>, Option<usize>)>,
    /// The first unlisted differences, as stdout shows them; of one model's
    /// report, not merged.
    shown: Shown,
}

/// One case compared: its id, its set (`<slug>/<kind>/<set>`) and how many
/// ids it carries.
struct CaseRead<'s> {
    id: String,
    set: &'s str,
    ids: usize,
}

impl Report {
    /// Count one compared case into `tally`, and keep its difference if it
    /// has one, written to `lines` as it is found, with its reason when it is
    /// listed; an unlisted one is counted into the shown ones too.
    fn record(
        &mut self,
        case: CaseRead<'_>,
        outcome: Result<(), String>,
        tally: &mut Tally,
        known: &BTreeMap<&str, &str>,
        lines: &mut String,
    ) {
        let CaseRead { id, set, ids } = case;
        assert!(
            self.seen.insert(id.clone()),
            "{id}: recorded more than once"
        );
        tally.cases += 1;
        tally.ids += ids;
        if let Err(why) = outcome {
            tally.differ += 1;
            let line = format!("  differs {id}: {why}\n");
            lines.push_str(&line);
            match known_reason(known, &id, set) {
                Some(reason) => {
                    tally.listed += 1;
                    lines.push_str(&format!("          known: {reason}\n"));
                }
                None => self.shown.unlisted(&line),
            }
            self.set_of_difference.insert(id.clone(), set.to_string());
            self.differences.insert(id, why);
        }
    }

    /// The set of a differing case, `<slug>/<kind>/<set>`.
    fn set_of(&self, id: &str) -> &str {
        self.set_of_difference.get(id).map_or("", String::as_str)
    }

    /// Take in what one model's comparison reported.
    fn merge(&mut self, other: Report) {
        self.loaded_dirs.extend(other.loaded_dirs);
        self.sets_read.extend(other.sets_read);
        self.set_of_difference.extend(other.set_of_difference);
        for id in other.seen {
            assert!(
                self.seen.insert(id.clone()),
                "{id}: recorded more than once"
            );
        }
        self.differences.extend(other.differences);
        self.unloaded.extend(other.unloaded);
        self.mismatches.extend(other.mismatches);
    }

    /// What fails the run; empty when it passes.
    fn failures(
        &self,
        root: &Path,
        known: &BTreeMap<&str, &str>,
        known_unloaded: &BTreeMap<&str, &str>,
    ) -> Vec<String> {
        let mut failures = Vec::new();
        let unlisted: Vec<String> = self
            .unloaded
            .iter()
            .filter(|(slug, _)| !known_unloaded.contains_key(slug.as_str()))
            .map(|(slug, why)| format!("{slug}: {why}"))
            .collect();
        if !unlisted.is_empty() {
            failures.push(format!(
                "models not compared, because their tokenizer did not load:\n{}",
                unlisted.join("\n")
            ));
        }
        let loads_now: Vec<&str> = known_unloaded
            .keys()
            .copied()
            .filter(|slug| {
                self.loaded_dirs.iter().any(|dir| {
                    dir.strip_prefix(*slug)
                        .is_some_and(|rest| rest.starts_with('/'))
                })
            })
            .collect();
        if !loads_now.is_empty() {
            failures.push(format!(
                "listed in KNOWN_UNLOADED but their tokenizer loads now; remove: {}",
                loads_now.join(", ")
            ));
        }
        if !self.mismatches.is_empty() {
            let sets: Vec<String> = self
                .mismatches
                .iter()
                .map(|(set, read, listed)| format!("{set}: {}", counts(*read, *listed)))
                .collect();
            failures.push(format!(
                "sets whose cases read are not the cases sets.toml lists:\n{}",
                sets.join("\n")
            ));
        }
        if self.seen.is_empty() && self.unloaded.is_empty() {
            failures.push(format!(
                "no render or parse case under {}: every <slug>/render and <slug>/parse \
                 directory is missing or empty",
                root.display()
            ));
        }
        let unexpected: Vec<(&str, &str)> = self
            .differences
            .iter()
            .filter(|(id, _)| known_reason(known, id, self.set_of(id)).is_none())
            .map(|(id, why)| (id.as_str(), why.as_str()))
            .collect();
        if !unexpected.is_empty() {
            let mut per_model: BTreeMap<&str, usize> = BTreeMap::new();
            for (id, _) in &unexpected {
                *per_model
                    .entry(id.split('/').next().unwrap_or_default())
                    .or_default() += 1;
            }
            let models: Vec<String> = per_model
                .iter()
                .map(|(slug, count)| format!("{slug} {count}"))
                .collect();
            let cases: Vec<String> = unexpected
                .iter()
                .map(|(id, why)| format!("{id}: {why}"))
                .collect();
            failures.push(format!(
                "differences not listed in KNOWN_DIFFERENCES: {} ({}):\n{}",
                unexpected.len(),
                models.join(", "),
                cases.join("\n")
            ));
        }
        let healed = self.healed(known);
        if !healed.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but matching the reference now; remove: {}",
                healed.join(", ")
            ));
        }
        let gone = self.gone(known);
        if !gone.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but no longer among the loaded fixtures; remove or \
                 rename: {}",
                gone.join(", ")
            ));
        }
        failures
    }

    /// Whether a compared case is one `listed` covers: the id itself, one
    /// under its prefix, one of its set, or one that ends with its suffix.
    fn seen_listed(&self, listed: Listed<'_>) -> bool {
        match listed {
            Listed::Id(id) => self.seen.contains(id),
            Listed::Set(set) => self.sets_read.contains(set),
            Listed::Prefix(prefix) => self
                .seen
                .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
                .next()
                .is_some_and(|id| id.starts_with(prefix)),
            Listed::Suffix(suffix) => self.seen.iter().any(|id| id.ends_with(suffix)),
        }
    }

    /// The entries of `known` whose compared cases all match the reference
    /// now: the entries no differing case counts for ([`counting_entry`]: a
    /// case with an entry of its own id counts for that entry, otherwise for
    /// the one entry that covers it and for none when several do), so that a
    /// prefix heals once every case under it that no other entry names
    /// matches, and a case-name entry heals once its case matches on every
    /// model that has no whole-side entry, whatever the whole-side models'
    /// cases do; an entry is live while one differing case counts for it.
    fn healed<'k>(&self, known: &BTreeMap<&'k str, &'k str>) -> Vec<&'k str> {
        let live: BTreeSet<&str> = self
            .differences
            .keys()
            .filter_map(|id| counting_entry(known, id, self.set_of(id)))
            .collect();
        known
            .keys()
            .copied()
            .filter(|entry| self.seen_listed(Listed::of(entry)) && !live.contains(entry))
            .collect()
    }

    /// The entries of `known` whose set the tree holds that are no longer
    /// among its cases.
    fn gone<'k>(&self, known: &BTreeMap<&'k str, &'k str>) -> Vec<&'k str> {
        known
            .keys()
            .copied()
            .filter(|entry| {
                let listed = Listed::of(entry);
                self.holds(listed) && !self.seen_listed(listed)
            })
            .collect()
    }

    /// Whether the tree read holds what `listed` stands for, so that an entry
    /// with no case under it is stale: for a case name, any loaded model; for
    /// a set, the set read; for an id or a prefix, its model loaded and its
    /// set read: the set whose name the id or the prefix begins with, any set
    /// under a prefix of set names, and the hand-written set for an id of a
    /// hand-written case's shape. A benchmark case is `<set>-<n>` and a
    /// hand-written case a name, so an id that no read set prefixes is a
    /// hand-written case unless it ends in `-<n>`: a benchmark set the tree
    /// omits, the unpack's sample being every Nth, leaves its entries alone,
    /// an id of one of its cases among them, while an id of a hand-written
    /// case the tree no longer has (renamed, dropped, or a typo in the entry)
    /// is held to the hand-written set and read as gone.
    fn holds(&self, listed: Listed<'_>) -> bool {
        let entry = match listed {
            Listed::Suffix(_) => return !self.loaded_dirs.is_empty(),
            Listed::Set(set) => return self.sets_read.contains(set),
            Listed::Id(entry) | Listed::Prefix(entry) => entry,
        };
        // `<slug>/<kind>/` and what follows it.
        let (dir, rest) = entry
            .match_indices('/')
            .nth(1)
            .map_or((entry, ""), |(at, _)| entry.split_at(at + 1));
        let mut sets = self
            .sets_read
            .range::<str, _>((Bound::Included(dir), Bound::Unbounded))
            .map_while(|set| set.strip_prefix(dir));
        match listed {
            Listed::Prefix(_) => sets.any(|set| set.starts_with(rest) || rest.starts_with(set)),
            _ => {
                sets.any(|set| rest.starts_with(set))
                    || (!is_benchmark_case(rest)
                        && self.sets_read.contains(&format!("{dir}{HAND_WRITTEN_SET}")))
            }
        }
    }
}

/// Whether `name`, the part of a case id after `<slug>/<kind>/`, is of a
/// benchmark case's shape, `<set>-<n>`; a hand-written case's name does not
/// end in a number.
fn is_benchmark_case(name: &str) -> bool {
    name.rsplit_once('-')
        .is_some_and(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report's path are test diagnostic output"
)]
fn encode_and_incremental_decode_match_the_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; point it at the tree `bellwether unpack` writes"
        );
        return;
    };
    let manifests = read_manifests(&root).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !manifests.is_empty(),
        "no <slug>/manifest.toml under {}",
        root.display()
    );
    let packed = packed_sets(&root, &manifests).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        packed.is_empty(),
        "{FIXTURES_ENV}={} is not the tree `bellwether unpack` writes: {} fixture sets under it \
         are compressed or Git LFS pointers (the first is {}), and this test reads plain JSON \
         Lines only. Run `bellwether unpack --fixtures <bellwether checkout>/fixtures --out \
         <dir>` and point {FIXTURES_ENV} at <dir>",
        root.display(),
        packed.len(),
        packed[0].display()
    );

    let known: BTreeMap<&str, &str> = KNOWN_DIFFERENCES.iter().copied().collect();
    for id in known.keys() {
        let mut parts = id.splitn(3, '/');
        let (slug, kind, name) = (parts.next(), parts.next(), parts.next());
        assert!(
            slug.is_some_and(|s| !s.is_empty())
                && matches!(kind, Some("render" | "parse"))
                && name.is_some_and(|n| {
                    !n.is_empty()
                        && !n.contains('/')
                        && n.find('*').is_none_or(|at| at == n.len() - 1)
                }),
            "KNOWN_DIFFERENCES entry {id} is not <slug>/render/<name> or <slug>/parse/<name>, with \
             `*` only at the end, as a prefix"
        );
    }
    let known_unloaded: BTreeMap<&str, &str> = KNOWN_UNLOADED.iter().copied().collect();
    let verbose = verbose();
    let (path, mut file) = report_file().unwrap_or_else(|e| panic!("{e}"));
    let mut stdout = Stdout;
    let report = compare(
        &root,
        &manifests,
        &known,
        load_tokenizer,
        &mut Output {
            stdout: &mut stdout,
            report: &mut file,
            verbose,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let failures = report.failures(&root, &known, &known_unloaded);
    for failure in &failures {
        writeln!(file, "{failure}")
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    }
    file.flush()
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    println!("per-case report: {}", path.display());
    let message = if verbose {
        failures.join("\n")
    } else {
        let capped: Vec<String> = failures
            .iter()
            .map(|failure| capped(failure, SHOWN, &path))
            .collect();
        capped.join("\n")
    };
    assert!(failures.is_empty(), "{message}");
}

/// Whether [`VERBOSE_ENV`] asks for the per-case report on stdout: set to
/// anything but empty or `0`.
fn verbose() -> bool {
    std::env::var(VERBOSE_ENV).is_ok_and(|value| !value.is_empty() && value != "0")
}

/// The per-case report's file, [`REPORT_FILE`] under [`REPORT_DIR_ENV`] or
/// the build's `target/tmp`, created with its directory.
fn report_file() -> Result<(PathBuf, BufWriter<fs::File>), String> {
    let dir = std::env::var_os(REPORT_DIR_ENV)
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join(REPORT_FILE);
    let file =
        fs::File::create(&path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    Ok((path, BufWriter::new(file)))
}

/// `failure` with at most `shown` lines after its first; the rest are
/// counted, and the reader sent to the per-case report at `report`.
fn capped(failure: &str, shown: usize, report: &Path) -> String {
    let lines: Vec<&str> = failure.lines().collect();
    if lines.len() <= shown + 1 {
        return failure.to_string();
    }
    format!(
        "{}\n... and {} more lines, every one in {}",
        lines[..=shown].join("\n"),
        lines.len() - shown - 1,
        report.display()
    )
}

/// Compare each model's render and parse sets under `root` with the tokenizer
/// `load` gives for the model, each case as its line is read, and the cases
/// read from each set with the model's `sets.toml`. The models are compared
/// on as many threads as the run has CPUs; what each gave is written to `out`
/// in the manifests' order, each model as soon as it and the ones before it
/// are done, so the report reads as a serial run's. A model whose tokenizer
/// does not load is kept in the report and the run goes on.
fn compare(
    root: &Path,
    manifests: &[(String, Manifest)],
    known: &BTreeMap<&str, &str>,
    load: impl Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String> + Sync,
    out: &mut Output<'_>,
) -> Result<Report, String> {
    let mut report = Report::default();
    let mut summaries = Vec::with_capacity(manifests.len());
    let mut without_sets = 0;
    let mut not_written = None;
    in_order(
        manifests,
        |slug, manifest| compare_model(root, slug, manifest, known, &load),
        |compared| {
            if let Err(e) = out.model(&compared) {
                not_written.get_or_insert(e);
            }
            if !compared.has_sets {
                without_sets += 1;
            }
            report.merge(compared.report);
            summaries.push(compared.summary);
        },
    )?;
    if let Some(e) = not_written {
        return Err(format!("cannot write the report: {e}"));
    }
    let totals = format!(
        "{} cases compared, {} match, {} differ",
        report.seen.len(),
        report.seen.len() - report.differences.len(),
        report.differences.len()
    );
    out.finish(&summaries, without_sets, &totals)
        .map_err(|e| format!("cannot write the report: {e}"))?;
    Ok(report)
}

/// What comparing one model gave: the per-case lines a serial run would have
/// printed for it, its line of the summary, the first unlisted differences
/// stdout shows, whether it had sets to compare, and its part of the report.
struct Compared {
    lines: String,
    summary: String,
    shown: String,
    has_sets: bool,
    report: Report,
}

/// Compare one model's sets, as [`compare`] describes.
fn compare_model<F>(
    root: &Path,
    slug: &str,
    manifest: &Manifest,
    known: &BTreeMap<&str, &str>,
    load: &F,
) -> Result<Compared, String>
where
    F: Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String>,
{
    let mut report = Report::default();
    let mut lines = String::new();
    let model_dir = root.join(slug);
    let sets = set_files(&model_dir)?;
    let listed = read_set_table(&model_dir)?;
    if sets.is_empty() {
        // Nothing to compare; a set the table lists is missing.
        let listed = listed.unwrap_or_default();
        let mismatches = set_mismatches(slug, &BTreeMap::new(), &listed);
        report.mismatches.extend(mismatches);
        return Ok(Compared {
            lines,
            summary: format!("{slug}: no render or parse sets"),
            shown: String::new(),
            has_sets: false,
            report,
        });
    }
    let Some(listed) = listed else {
        return Err(format!(
            "{slug}: no sets.toml beside its manifest.toml to check its sets against; \
             {FIXTURES_ENV} must point at the tree `bellwether unpack` writes, which carries \
             each model's sets.toml"
        ));
    };
    let (tok, from) = match load(slug, manifest) {
        Ok(loaded) => loaded,
        Err(why) => {
            report.unloaded.push((slug.to_string(), why));
            return Ok(Compared {
                lines,
                summary: format!("{slug}: not compared, its tokenizer did not load"),
                shown: String::new(),
                has_sets: true,
                report,
            });
        }
    };
    lines.push_str(&format!(
        "{slug}: {} at {} from {from}\n",
        manifest.model, manifest.revision
    ));
    let mut encode = Tally::default();
    let mut decode = Tally::default();
    let mut read = BTreeMap::new();
    for (kind, set, path) in &sets {
        report.loaded_dirs.insert(format!("{slug}/{kind}"));
        let set_path = format!("{slug}/{set}");
        report.sets_read.insert(set_path.clone());
        let mut tokenizers = None;
        let cases = if *kind == "render" {
            encode.loaded = true;
            for_each_case(path, |case: RenderCase| {
                assert_eq!(case.kind, "render", "{}: not a render case", case.id);
                assert_eq!(
                    case.model, manifest.model,
                    "{}: model differs from the manifest",
                    case.id
                );
                tokenizers.get_or_insert_with(|| recorded_with(&case.reference.provenance));
                let outcome = check_encode(tok.as_ref(), &case.reference);
                let ids = case.reference.input_ids.len();
                let read = CaseRead {
                    id: case.id,
                    set: &set_path,
                    ids,
                };
                report.record(read, outcome, &mut encode, known, &mut lines);
            })?
        } else {
            decode.loaded = true;
            for_each_case(path, |case: ParseCase| {
                assert_eq!(case.kind, "parse", "{}: not a parse case", case.id);
                assert_eq!(
                    case.model, manifest.model,
                    "{}: model differs from the manifest",
                    case.id
                );
                tokenizers.get_or_insert_with(|| recorded_with(&case.reference.provenance));
                let outcome = check_decode(&tok, &case);
                let ids = case.output_ids.len();
                let read = CaseRead {
                    id: case.id,
                    set: &set_path,
                    ids,
                };
                report.record(read, outcome, &mut decode, known, &mut lines);
            })?
        };
        let tokenizers = tokenizers.map_or(String::new(), |version| {
            format!("; recorded with tokenizers {version}")
        });
        let counted = counts(Some(cases), listed.get(set).copied());
        lines.push_str(&format!("  {set}: {counted}{tokenizers}\n"));
        read.insert(set.clone(), cases);
    }
    report
        .mismatches
        .extend(set_mismatches(slug, &read, &listed));
    // The model's own stale entries, judged with the whole list so that a case
    // counts as it does on the merged report; the entries that span models
    // (`*/...`) are judged once, there.
    let own_prefix = format!("{slug}/");
    let stale = report
        .healed(known)
        .into_iter()
        .chain(report.gone(known))
        .filter(|entry| entry.starts_with(&own_prefix))
        .count();
    Ok(Compared {
        lines,
        summary: format!("{slug}: encode: {encode}; incremental decode: {decode}; {stale} stale"),
        shown: std::mem::take(&mut report.shown).text(slug),
        has_sets: true,
        report,
    })
}

/// Run `each` on every model, on as many threads as the run has CPUs, and
/// hand what it gives to `then` in the models' order, each as soon as it and
/// the ones before it are done. An error from `each` ends the run once the
/// models in flight are done: the thread that met it raises a flag, as does
/// one that unwinds from a panic, and no thread takes a model once it is up.
fn in_order<T: Send>(
    manifests: &[(String, Manifest)],
    each: impl Fn(&str, &Manifest) -> Result<T, String> + Sync,
    then: impl FnMut(T),
) -> Result<(), String> {
    let threads = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(manifests.len())
        .max(1);
    in_order_on(
        threads,
        manifests,
        |(slug, manifest)| each(slug, manifest),
        then,
    )
}

/// [`in_order`] on `threads` threads, over any items.
fn in_order_on<I: Sync, T: Send>(
    threads: usize,
    items: &[I],
    each: impl Fn(&I) -> Result<T, String> + Sync,
    mut then: impl FnMut(T),
) -> Result<(), String> {
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, stop, each) = (&next, &stop, &each);
            scope.spawn(move || {
                let _raised_on_panic = StopOnPanic(stop);
                while !stop.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    let outcome = each(item);
                    if outcome.is_err() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    if tx.send((index, outcome)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut done = 0;
        for (index, outcome) in rx {
            pending.insert(index, outcome);
            while let Some(outcome) = pending.remove(&done) {
                then(outcome?);
                done += 1;
            }
        }
        Ok(())
    })
}

/// Raises the flag it holds when its thread unwinds from a panic, so that the
/// other threads of [`in_order_on`] take no further item.
struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

/// Encode the reference text as the gateway encodes a rendered prompt, without
/// adding special tokens, and compare the ids with the reference.
fn check_encode(tok: &dyn Tokenizer, reference: &RenderReference) -> Result<(), String> {
    let encoding = tok
        .encode(&reference.text, false)
        .map_err(|e| format!("encode failed: {e}"))?;
    let got = encoding.token_ids();
    let want = reference.input_ids.as_slice();
    if got == want {
        return Ok(());
    }
    let at = first_difference(got, want);
    Err(format!(
        "ids part at index {at} ({} encoded, {} in the reference): encoded {}, reference {}",
        got.len(),
        want.len(),
        ids_around(tok, got, at),
        ids_around(tok, want, at)
    ))
}

/// Feed the output ids one at a time through a fresh [`Sequence`], the
/// incremental decoder the gateway's stop-sequence decoder streams through,
/// keeping special tokens as the reference did; compare the piece each id
/// gives (empty when it emits nothing) with the recorded one, then the pieces
/// joined with the recorded output text.
fn check_decode(tok: &Arc<dyn Tokenizer>, case: &ParseCase) -> Result<(), String> {
    if case.output_pieces.len() != case.output_ids.len() {
        return Err(format!(
            "the fixture records {} pieces for {} ids",
            case.output_pieces.len(),
            case.output_ids.len()
        ));
    }
    let mut sequence = Sequence::new_with_options(Arc::clone(tok), false);
    let mut pieces = Vec::with_capacity(case.output_ids.len());
    for (index, &id) in case.output_ids.iter().enumerate() {
        let piece = sequence
            .append_token(id)
            .map_err(|e| format!("decoding id {id} at index {index} failed: {e}"))?;
        pieces.push(piece);
    }
    let mut parts = Vec::new();
    if pieces != case.output_pieces {
        let at = first_difference(&pieces, &case.output_pieces);
        parts.push(format!(
            "pieces part at index {at} (id {}): decoded {:?}, recorded {:?}",
            case.output_ids[at], pieces[at], case.output_pieces[at]
        ));
    }
    let text = pieces.concat();
    if text != case.reference.text {
        let at = first_difference(text.as_bytes(), case.reference.text.as_bytes());
        parts.push(format!(
            "the joined pieces part from the output text at byte {at}: decoded {:?}, recorded {:?}",
            window(&text, at),
            window(&case.reference.text, at)
        ));
    }
    if parts.is_empty() {
        Ok(())
    } else {
        Err(parts.join("; "))
    }
}

/// The first index where the two sequences differ, or the shorter length when
/// one is a prefix of the other.
fn first_difference<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| a.len().min(b.len()))
}

/// The ids from two before `at` to three after it, with their decoded text.
fn ids_around(tok: &dyn Tokenizer, ids: &[u32], at: usize) -> String {
    let around = &ids[at.saturating_sub(2)..(at + 3).min(ids.len())];
    let text = tok
        .decode(around, false)
        .unwrap_or_else(|e| format!("<decode failed: {e}>"));
    format!("{around:?} {text:?}")
}

/// Up to 40 bytes before `at` and 60 after, cut on character boundaries.
fn window(text: &str, at: usize) -> &str {
    let mut start = at.saturating_sub(40);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (at + 60).min(text.len());
    while !text.is_char_boundary(end) {
        end += 1;
    }
    &text[start..end]
}

/// The tokenizers version a fixture set was recorded with, from the provenance
/// of its first line.
fn recorded_with(provenance: &Value) -> String {
    provenance
        .get("tokenizers")
        .and_then(Value::as_str)
        .unwrap_or("an unrecorded version")
        .to_string()
}

fn read_manifests(root: &Path) -> Result<Vec<(String, Manifest)>, String> {
    let entries = fs::read_dir(root).map_err(|e| format!("cannot read {}: {e}", root.display()))?;
    let mut manifests = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", root.display()))?;
        let path = entry.path().join("manifest.toml");
        if !path.is_file() {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let manifest: Manifest =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        manifests.push((entry.file_name().to_string_lossy().into_owned(), manifest));
    }
    manifests.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(manifests)
}

/// The set files under `root` that are not plain JSON Lines: a
/// `<set>.jsonl.zst`, the form bellwether keeps a benchmark set in, and a set
/// file holding the pointer a clone has until Git LFS fetches the set.
/// bellwether's `fixtures/` directory has them; the tree `bellwether unpack`
/// writes has none.
fn packed_sets(root: &Path, manifests: &[(String, Manifest)]) -> Result<Vec<PathBuf>, String> {
    let mut packed = Vec::new();
    for (slug, _) in manifests {
        for dir in entries(&root.join(slug))? {
            if !dir.is_dir() {
                continue;
            }
            for path in entries(&dir)? {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.ends_with(".jsonl.zst")
                    || (name.ends_with(".jsonl") && starts_with_lfs_pointer(&path)?)
                {
                    packed.push(path);
                }
            }
        }
    }
    Ok(packed)
}

/// Whether the file begins with a Git LFS pointer.
fn starts_with_lfs_pointer(path: &Path) -> Result<bool, String> {
    let mut start = Vec::with_capacity(LFS_POINTER.len());
    fs::File::open(path)
        .and_then(|file| file.take(LFS_POINTER.len() as u64).read_to_end(&mut start))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(start == LFS_POINTER)
}

/// The model's sets of the kinds this test reads, kind by kind and in name
/// order: the kind, `<kind>/<set>`, and the set's file.
fn set_files(model_dir: &Path) -> Result<Vec<(&'static str, String, PathBuf)>, String> {
    let mut sets = Vec::new();
    for kind in KINDS {
        let dir = model_dir.join(kind);
        if !dir.is_dir() {
            continue;
        }
        for path in entries(&dir)? {
            let name = path.file_name().and_then(|name| name.to_str());
            if let Some(set) = name.and_then(|name| name.strip_suffix(".jsonl")) {
                let set = format!("{kind}/{set}");
                sets.push((kind, set, path));
            }
        }
    }
    Ok(sets)
}

/// The cases `<slug>/sets.toml` lists for each set of the kinds this test
/// reads, by `<kind>/<set>`; none when the model has no `sets.toml`.
fn read_set_table(model_dir: &Path) -> Result<Option<BTreeMap<String, usize>>, String> {
    let path = model_dir.join("sets.toml");
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let tables: BTreeMap<String, BTreeMap<String, SetTable>> =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let listed = tables
        .into_iter()
        .filter(|(kind, _)| KINDS.contains(&kind.as_str()))
        .flat_map(|(kind, sets)| {
            sets.into_iter()
                .map(move |(set, table)| (format!("{kind}/{set}"), table.cases))
        })
        .collect();
    Ok(Some(listed))
}

/// Each of the model's sets whose cases read differ from the cases its
/// `sets.toml` lists: `<slug>/<kind>/<set>`, the cases read (none when the
/// tree has no file for the set) and the cases listed (none when the table
/// does not list it).
fn set_mismatches(
    slug: &str,
    read: &BTreeMap<String, usize>,
    listed: &BTreeMap<String, usize>,
) -> Vec<(String, Option<usize>, Option<usize>)> {
    let sets: BTreeSet<&String> = read.keys().chain(listed.keys()).collect();
    sets.into_iter()
        .map(|set| (set, read.get(set).copied(), listed.get(set).copied()))
        .filter(|(_, read, listed)| read != listed)
        .map(|(set, read, listed)| (format!("{slug}/{set}"), read, listed))
        .collect()
}

/// A set's cases read and the cases its `sets.toml` lists, as the report
/// gives them.
fn counts(read: Option<usize>, listed: Option<usize>) -> String {
    let read = read.map_or_else(|| "no file".to_string(), |n| format!("{n} read"));
    let listed = listed.map_or_else(
        || "not in sets.toml".to_string(),
        |n| format!("sets.toml lists {n}"),
    );
    format!("{read}, {listed}")
}

/// The paths in `dir`, sorted.
fn entries(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = fs::read_dir(dir)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    paths.sort();
    Ok(paths)
}

/// Read a set a line at a time, handing each case to `each` as its line is
/// read; the number of cases read.
fn for_each_case<T: DeserializeOwned>(
    path: &Path,
    mut each: impl FnMut(T),
) -> Result<usize, String> {
    let file = fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut cases = 0;
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let case = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), number + 1))?;
        each(case);
        cases += 1;
    }
    Ok(cases)
}

/// The model's tokenizer, from the files [`tokenizer_dir`] finds, and where
/// they are.
fn load_tokenizer(slug: &str, manifest: &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String> {
    let dir = tokenizer_dir(&manifest.model, &manifest.revision, slug)?;
    let dir = dir
        .to_str()
        .ok_or_else(|| format!("tokenizer path {} is not UTF-8", dir.display()))?
        .to_string();
    let tok =
        create_tokenizer(&dir).map_err(|e| format!("the tokenizer at {dir} does not load: {e}"))?;
    Ok((tok, dir))
}

/// The vocabulary files a checkpoint may ship, in the order they are tried:
/// a `tokenizers` file; a tiktoken file for the models that have no
/// `tokenizer.json` (Kimi); the `vocab.json` + `merges.txt` pair of the
/// models that ship neither (Qwen3-Omni), which counts only whole.
const VOCABULARIES: [&[&str]; 3] = [
    &["tokenizer.json"],
    &["tiktoken.model"],
    &["vocab.json", "merges.txt"],
];

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download of the
/// files the tokenizer loads: `tokenizer_config.json`, the first of
/// [`VOCABULARIES`] the checkpoint serves whole, and `config.json` when it
/// has one (the renderer detection reads it). A download is written beside
/// its name and renamed into place, so an interrupted write never leaves a
/// short file the next run would trust.
fn tokenizer_dir(model: &str, revision: &str, slug: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = hf_cache_snapshot(model, revision) {
        return Ok(snapshot);
    }
    let dir = PathBuf::from(CACHE_DIR).join(slug).join(revision);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let client = common::http_client();
    download(
        &client,
        HUB,
        model,
        revision,
        &dir,
        "tokenizer_config.json",
        100,
    )?
    .ok_or_else(|| format!("{model} at {revision} serves no tokenizer_config.json"))?;
    download(&client, HUB, model, revision, &dir, "config.json", 2)?;
    let mut vocabulary = None;
    for files in VOCABULARIES {
        let mut whole = true;
        for file in files {
            whole &= download(&client, HUB, model, revision, &dir, file, 100_000)?.is_some();
        }
        if whole {
            vocabulary = Some(files);
            break;
        }
    }
    vocabulary.ok_or_else(|| format!("{model} at {revision} serves none of {}", vocabularies()))?;
    Ok(dir)
}

/// [`VOCABULARIES`] for a message: `tokenizer.json, tiktoken.model, vocab.json + merges.txt`.
fn vocabularies() -> String {
    VOCABULARIES
        .iter()
        .map(|files| files.join(" + "))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The hub the checkpoints' files are fetched from, at the manifest's revision.
const HUB: &str = "https://huggingface.co";

/// Fetches `file` from `hub` into `dir` unless it is there already; `Ok(None)` when the
/// checkpoint does not serve it (HTTP 404, remembered beside the name as
/// `<file>.absent`, since the revision is a commit and the answer final), an
/// error for any other failure or a file shorter than `min_bytes`.
fn download(
    client: &reqwest::blocking::Client,
    hub: &str,
    model: &str,
    revision: &str,
    dir: &Path,
    file: &str,
    min_bytes: usize,
) -> Result<Option<PathBuf>, String> {
    let path = dir.join(file);
    if path.is_file() {
        return Ok(Some(path));
    }
    let absent = dir.join(format!("{file}.absent"));
    if absent.is_file() {
        return Ok(None);
    }
    let url = format!("{hub}/{model}/resolve/{revision}/{file}");
    let answer = fetch(client, &url).map_err(|e| format!("GET {url}: {e}"))?;
    if answer.status == reqwest::StatusCode::NOT_FOUND {
        fs::write(&absent, b"").map_err(|e| format!("cannot write {}: {e}", absent.display()))?;
        return Ok(None);
    }
    if !answer.status.is_success() {
        return Err(format!("GET {url}: HTTP {}", answer.status));
    }
    let bytes = answer.body;
    if bytes.len() < min_bytes {
        return Err(format!(
            "{url}: {} bytes, expected at least {min_bytes}",
            bytes.len()
        ));
    }
    // Named for this process: two test binaries may fetch the same file into one
    // shared directory at the same moment, and each renames its own whole file
    // into place.
    let part = dir.join(format!("{file}.{}.part", std::process::id()));
    fs::write(&part, &bytes).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
    fs::rename(&part, &path)
        .map_err(|e| format!("cannot rename {} into place: {e}", part.display()))?;
    Ok(Some(path))
}

/// `<hub cache>/models--<org>--<name>/snapshots/<revision>`, the layout
/// `huggingface_hub` keeps, under `HF_HUB_CACHE`, `HF_HOME/hub` or the default
/// `~/.cache/huggingface/hub`.
fn hf_cache_snapshot(model: &str, revision: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
        })?;
    snapshot_under(&hub, model, revision)
}

/// The snapshot of `model` at `revision` under the hub cache `hub`, when it
/// holds `tokenizer_config.json` and one of [`VOCABULARIES`] whole.
fn snapshot_under(hub: &Path, model: &str, revision: &str) -> Option<PathBuf> {
    let dir = hub
        .join(format!("models--{}", model.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    (dir.join("tokenizer_config.json").is_file()
        && VOCABULARIES
            .iter()
            .any(|files| files.iter().all(|file| dir.join(file).is_file())))
    .then_some(dir)
}

// The run's own checks, on fixture trees the tests below write.

/// A fixture tree in a temporary directory: each file's path under the root
/// and its contents.
fn tree(files: &[(&str, &str)]) -> io::Result<tempfile::TempDir> {
    let root = tempfile::tempdir()?;
    for (path, contents) in files {
        let path = root.path().join(path);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, contents)?;
    }
    Ok(root)
}

fn manifest(model: &str) -> String {
    format!("model = \"{model}\"\nrevision = \"0123456789abcdef0123456789abcdef01234567\"\n")
}

/// A `sets.toml` as `bellwether record` writes it, listing each
/// `<kind>/<set>` with its number of cases.
fn sets_toml(sets: &[(&str, usize)]) -> String {
    let mut text = String::new();
    for (set, cases) in sets {
        let (kind, name) = set.split_once('/').unwrap_or((set, ""));
        let sha256 = "0".repeat(64);
        text += &format!(
            "[{kind}.{name}]\nform = \"zstd\"\ncases = {cases}\nrejected = 0\nplain_bytes = 100\n\
             plain_sha256 = \"{sha256}\"\n\n"
        );
    }
    text
}

/// One render line whose text the mock tokenizer encodes to its ids.
fn render_line(id: &str, model: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"kind\":\"render\",\"model\":\"{model}\",\
         \"reference\":{{\"input_ids\":[1],\"text\":\"Hello\"}}}}\n"
    )
}

/// One render line whose ids are not what the mock tokenizer encodes its text
/// to.
fn differing_line(id: &str, model: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"kind\":\"render\",\"model\":\"{model}\",\
         \"reference\":{{\"input_ids\":[2],\"text\":\"Hello\"}}}}\n"
    )
}

/// One parse line whose ids the mock tokenizer decodes to its pieces.
fn parse_line(id: &str, model: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"kind\":\"parse\",\"model\":\"{model}\",\"output_ids\":[1],\
         \"output_pieces\":[\"Hello\"],\"reference\":{{\"text\":\"Hello\"}}}}\n"
    )
}

/// The tokenizer crate's mock tokenizer, and where a loader would say it
/// came from.
fn mock() -> (Arc<dyn Tokenizer>, String) {
    (
        Arc::new(MockTokenizer::new()),
        "the mock tokenizer".to_string(),
    )
}

/// [`compare`] writing to buffers nobody reads, for the tests that read the
/// report it returns.
fn compare_quietly(
    root: &Path,
    manifests: &[(String, Manifest)],
    known: &BTreeMap<&str, &str>,
    load: impl Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String> + Sync,
) -> Result<Report, String> {
    let (mut stdout, mut report) = (Vec::new(), Vec::new());
    compare(
        root,
        manifests,
        known,
        load,
        &mut Output {
            stdout: &mut stdout,
            report: &mut report,
            verbose: false,
        },
    )
}

#[test]
fn a_prefix_entry_heals_when_only_cases_listed_by_their_own_id_still_differ() {
    let model = "org/m";
    let listed = BTreeMap::from([
        ("m/render/*", "recorded with another pipeline"),
        ("m/render/common-0", "the template drops the role"),
    ]);
    // common-2 differs under the prefix with no entry of its own: the prefix
    // is live, and nothing fails the run.
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        ("m/sets.toml", &sets_toml(&[("render/common", 3)])),
        (
            "m/render/common.jsonl",
            &(differing_line("m/render/common-0", model)
                + &render_line("m/render/common-1", model)
                + &differing_line("m/render/common-2", model)),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // Once common-2 matches, the one difference left under the prefix is the
    // by-id entry's own: the prefix has healed and must go, the by-id entry
    // stays.
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        ("m/sets.toml", &sets_toml(&[("render/common", 3)])),
        (
            "m/render/common.jsonl",
            &(differing_line("m/render/common-0", model)
                + &render_line("m/render/common-1", model)
                + &render_line("m/render/common-2", model)),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 1, "{:#?}", report.differences);
    let failures = report.failures(root.path(), &listed, &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("matching the reference now")
            && failures[0].ends_with("remove: m/render/*"),
        "{failures:#?}"
    );
}

/// Two models for the tests of entries that share a case: "m", whose whole
/// render side an entry covers, with common-0 differing and common-2 as
/// `m_common_2` says; "n", with no entry of its own, with common-0 as
/// `n_common_0` says. The case-name entry `*/render/common-0` names common-0
/// of both.
fn two_models(
    m_common_2: fn(&str, &str) -> String,
    n_common_0: fn(&str, &str) -> String,
) -> io::Result<tempfile::TempDir> {
    tree(&[
        ("m/manifest.toml", &manifest("org/m")),
        ("m/sets.toml", &sets_toml(&[("render/common", 3)])),
        (
            "m/render/common.jsonl",
            &(differing_line("m/render/common-0", "org/m")
                + &render_line("m/render/common-1", "org/m")
                + &m_common_2("m/render/common-2", "org/m")),
        ),
        ("n/manifest.toml", &manifest("org/n")),
        ("n/sets.toml", &sets_toml(&[("render/common", 2)])),
        (
            "n/render/common.jsonl",
            &(n_common_0("n/render/common-0", "org/n")
                + &render_line("n/render/common-1", "org/n")),
        ),
    ])
}

#[test]
fn a_prefix_entry_heals_when_only_the_cases_a_suffix_entry_names_still_differ() {
    let listed = BTreeMap::from([
        (
            "*/render/common-0",
            "a marker token decoded with a leading space",
        ),
        ("m/render/*", "recorded with another pipeline"),
    ]);
    // m's common-2 and n's common-0 differ: each entry has a case of its own
    // (m's common-0, which both cover, counts for neither), nothing fails.
    let root = two_models(differing_line, differing_line).unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 3, "{:#?}", report.differences);
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // m's common-2 matches: the one difference left under the prefix is a
    // case the suffix entry names too, so the prefix has healed and must go,
    // while the suffix lives on n's case.
    let root = two_models(render_line, differing_line).unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    let failures = report.failures(root.path(), &listed, &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("matching the reference now")
            && failures[0].ends_with("remove: m/render/*"),
        "{failures:#?}"
    );
}

#[test]
fn a_suffix_entry_heals_when_its_only_differing_case_sits_under_a_whole_side_prefix() {
    let listed = BTreeMap::from([
        (
            "*/render/common-0",
            "a marker token decoded with a leading space",
        ),
        ("m/render/*", "recorded with another pipeline"),
    ]);
    // n's common-0 matches: the suffix entry's one difference left is m's
    // common-0, which the prefix covers too, so the suffix has healed and
    // must go, while the prefix lives on m's common-2.
    let root = two_models(differing_line, render_line).unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    let failures = report.failures(root.path(), &listed, &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("matching the reference now")
            && failures[0].ends_with("remove: */render/common-0"),
        "{failures:#?}"
    );
}

#[test]
fn compressed_sets_and_lfs_pointers_are_found_under_the_root() {
    let pointer = "version https://git-lfs.github.com/spec/v1\n\
                   oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n\
                   size 12345\n";
    let root = tree(&[
        ("m/manifest.toml", &manifest("org/m")),
        ("m/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("m/render/common.jsonl", &render_line("m/render/a", "org/m")),
        ("m/render/bfcl-simple.jsonl.zst", "a zstd frame"),
        ("m/parse/bfcl-live.jsonl.zst", pointer),
        ("m/parse/bfcl-multiple.jsonl", pointer),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let found = packed_sets(root.path(), &manifests).unwrap();
    let under = |path: &str| root.path().join(path);
    assert_eq!(
        found,
        [
            under("m/parse/bfcl-live.jsonl.zst"),
            under("m/parse/bfcl-multiple.jsonl"),
            under("m/render/bfcl-simple.jsonl.zst"),
        ]
    );
}

#[test]
fn the_cases_read_must_be_the_cases_sets_toml_lists() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[
                ("parse/common", 1),
                ("render/bfcl-simple", 3),
                ("render/common", 2),
                ("tokenize/common", 4),
            ]),
        ),
        ("m/parse/common.jsonl", &parse_line("m/parse/a", model)),
        ("m/render/common.jsonl", &render_line("m/render/a", model)),
        ("m/render/extra.jsonl", &render_line("m/render/b", model)),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report =
        compare_quietly(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock())).unwrap();
    assert_eq!(
        report.mismatches,
        [
            ("m/render/bfcl-simple".to_string(), None, Some(3)),
            ("m/render/common".to_string(), Some(1), Some(2)),
            ("m/render/extra".to_string(), Some(1), None),
        ]
    );
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    for set in ["m/render/bfcl-simple", "m/render/common", "m/render/extra"] {
        assert!(failures[0].contains(set), "{set} is not in {failures:#?}");
    }
}

#[test]
fn sets_without_a_sets_toml_stop_the_run() {
    let root = tree(&[
        ("m/manifest.toml", &manifest("org/m")),
        ("m/render/common.jsonl", &render_line("m/render/a", "org/m")),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let error = compare_quietly(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock()))
        .err()
        .expect("sets with no sets.toml to check them against should stop the run");
    assert!(error.contains("sets.toml"), "{error}");
}

#[test]
fn a_cache_snapshot_with_any_whole_vocabulary_is_the_tokenizer_dir() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let snapshot = |model: &str, files: &[&str]| -> Vec<(String, String)> {
        files
            .iter()
            .map(|file| {
                (
                    format!(
                        "models--{}/snapshots/{revision}/{file}",
                        model.replace('/', "--")
                    ),
                    String::from("{}"),
                )
            })
            .collect()
    };
    let files = [
        snapshot("org/json", &["tokenizer_config.json", "tokenizer.json"]),
        snapshot("org/tiktoken", &["tokenizer_config.json", "tiktoken.model"]),
        snapshot(
            "org/pair",
            &["tokenizer_config.json", "vocab.json", "merges.txt"],
        ),
        snapshot("org/half-pair", &["tokenizer_config.json", "vocab.json"]),
        snapshot("org/no-config", &["vocab.json", "merges.txt"]),
    ]
    .concat();
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_str()))
        .collect();
    let hub = tree(&files).unwrap();
    for (model, found) in [
        ("org/json", true),
        ("org/tiktoken", true),
        ("org/pair", true),
        ("org/half-pair", false),
        ("org/no-config", false),
        ("org/absent", false),
    ] {
        let snapshot = snapshot_under(hub.path(), model, revision);
        assert_eq!(snapshot.is_some(), found, "{model}: {snapshot:?}");
    }
}

#[test]
fn a_model_whose_tokenizer_does_not_load_fails_the_run_after_the_others() {
    let root = tree(&[
        ("a/manifest.toml", &manifest("org/a")),
        ("a/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("a/render/common.jsonl", &render_line("a/render/x", "org/a")),
        ("b/manifest.toml", &manifest("org/b")),
        ("b/sets.toml", &sets_toml(&[("render/common", 1)])),
        ("b/render/common.jsonl", &render_line("b/render/x", "org/b")),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report = compare_quietly(
        root.path(),
        &manifests,
        &BTreeMap::new(),
        |slug, _| match slug {
            "a" => Err("no tokenizer.json".to_string()),
            _ => Ok(mock()),
        },
    )
    .unwrap();
    assert_eq!(
        report.unloaded,
        [("a".to_string(), "no tokenizer.json".to_string())]
    );
    assert_eq!(report.seen, BTreeSet::from(["b/render/x".to_string()]));
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("a: no tokenizer.json"),
        "{failures:#?}"
    );
    // Listed as known not to load, "a" no longer fails the run; a listed model
    // whose tokenizer loads must leave the list.
    let listed = BTreeMap::from([("a", "ships neither tokenizer.json nor tiktoken.model")]);
    assert_eq!(
        report.failures(root.path(), &BTreeMap::new(), &listed),
        Vec::<String>::new()
    );
    let stale = BTreeMap::from([("a", "does not load"), ("b", "does not load")]);
    let failures = report.failures(root.path(), &BTreeMap::new(), &stale);
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("loads now") && failures[0].ends_with("remove: b"),
        "{failures:#?}"
    );
}

#[test]
fn a_prefix_entry_covers_the_cases_under_it_and_must_cover_one() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[("render/common", 1), ("render/mgsm-th", 2)]),
        ),
        (
            "m/render/common.jsonl",
            &render_line("m/render/common-1", model),
        ),
        (
            "m/render/mgsm-th.jsonl",
            &(differing_line("m/render/mgsm-th-1", model)
                + &differing_line("m/render/mgsm-th-2", model)),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let listed = BTreeMap::from([("m/render/mgsm-th-*", "recorded with another pre-tokenizer")]);
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    // The prefix covers both differing cases, so nothing fails the run.
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // Without it the two are unlisted.
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("m/render/mgsm-th-1") && failures[0].contains("m/render/mgsm-th-2"),
        "{failures:#?}"
    );
    // A prefix under which every loaded case matches must go, and so must one
    // under a set the tree holds that no loaded case begins with; one under a
    // set the tree omits is left alone, since the tree holds every Nth
    // benchmark set.
    let stale = BTreeMap::from([
        ("m/render/common-*", "matches now"),
        ("m/render/mgsm-bn-*", "a set the tree omits"),
        ("m/render/mgsm-th-9*", "no such case"),
        ("m/render/mgsm-th-*", "recorded with another pre-tokenizer"),
    ]);
    let failures = report.failures(root.path(), &stale, &BTreeMap::new());
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures.iter().any(|failure| {
            failure.contains("matching the reference now")
                && failure.contains("m/render/common-*")
                && !failure.contains("mgsm")
        }),
        "{failures:#?}"
    );
    assert!(
        failures.iter().any(|failure| {
            failure.contains("no longer among the loaded fixtures")
                && failure.contains("m/render/mgsm-th-9*")
                && !failure.contains("mgsm-bn")
                && !failure.contains("mgsm-th-*")
        }),
        "{failures:#?}"
    );
}

#[test]
fn an_id_entry_of_an_omitted_benchmark_case_is_left_alone_and_of_a_lost_hand_written_one_is_not() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[("render/bfcl-multi-turn-base", 1), ("render/common", 1)]),
        ),
        (
            "m/render/bfcl-multi-turn-base.jsonl",
            &render_line("m/render/bfcl-multi-turn-base-0", model),
        ),
        (
            "m/render/common.jsonl",
            &render_line("m/render/text-developer-role", model),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let report =
        compare_quietly(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock())).unwrap();
    // A hand-written case that matches now, a case of a set the tree holds
    // that is not among its cases, and a hand-written name the tree's set
    // lacks (renamed, dropped, or a typo) are stale; a case of a set the tree
    // omits is left alone, since the tree holds every Nth benchmark set.
    let stale = BTreeMap::from([
        ("m/render/bfcl-multi-turn-base-9", "no such case"),
        ("m/render/text-developer-role", "matches now"),
        ("m/render/text-develoepr-role", "a typo"),
        ("m/render/continue-final-message", "no such case"),
        ("m/render/gsm8k-17", "a case of a set the tree omits"),
    ]);
    // One line names the healed entry, one the gone entries.
    let failures = report.failures(root.path(), &stale, &BTreeMap::new());
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures.iter().all(|failure| !failure.contains("gsm8k")),
        "{failures:#?}"
    );
    assert!(
        failures.iter().any(|failure| {
            failure.contains("matching the reference now")
                && failure.contains("m/render/text-developer-role")
        }),
        "{failures:#?}"
    );
    for gone in [
        "m/render/bfcl-multi-turn-base-9",
        "m/render/text-develoepr-role",
        "m/render/continue-final-message",
    ] {
        assert!(
            failures.iter().any(|failure| {
                failure.contains("no longer among the loaded fixtures") && failure.contains(gone)
            }),
            "{gone}: {failures:#?}"
        );
    }
}

#[test]
fn a_set_entry_covers_the_set_whole_and_must_cover_a_difference() {
    let model = "org/m";
    let root = tree(&[
        ("m/manifest.toml", &manifest(model)),
        (
            "m/sets.toml",
            &sets_toml(&[("render/common", 1), ("render/glaive-v2-05", 2)]),
        ),
        (
            "m/render/common.jsonl",
            &render_line("m/render/common-1", model),
        ),
        // The set's ids carry the sample's numbers, not the set's name.
        (
            "m/render/glaive-v2-05.jsonl",
            &(differing_line("m/render/glaive-v2-8174-3", model)
                + &differing_line("m/render/glaive-v2-8194-1", model)),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let listed = BTreeMap::from([(
        "m/render/glaive-v2-05.jsonl",
        "recorded with another pre-tokenizer",
    )]);
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    // The set entry covers both differing cases, so nothing fails the run.
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // Without it the two are unlisted.
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("m/render/glaive-v2-8174-3")
            && failures[0].contains("m/render/glaive-v2-8194-1"),
        "{failures:#?}"
    );
    // A set entry whose every case matches must go; one for a set the tree
    // omits is left alone.
    let stale = BTreeMap::from([
        ("m/render/common.jsonl", "matches now"),
        (
            "m/render/glaive-v2-05.jsonl",
            "recorded with another pre-tokenizer",
        ),
        ("m/render/gsm8k.jsonl", "a set the tree omits"),
    ]);
    let failures = report.failures(root.path(), &stale, &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("matching the reference now")
            && failures[0].contains("m/render/common.jsonl")
            && !failures[0].contains("glaive")
            && !failures[0].contains("gsm8k"),
        "{failures:#?}"
    );
}

#[test]
fn a_suffix_entry_covers_the_case_of_every_model_and_must_cover_one() {
    let (m, n) = ("org/m", "org/n");
    let root = tree(&[
        ("m/manifest.toml", &manifest(m)),
        ("m/sets.toml", &sets_toml(&[("render/common", 2)])),
        (
            "m/render/common.jsonl",
            &(render_line("m/render/common-1", m) + &differing_line("m/render/common-2", m)),
        ),
        ("n/manifest.toml", &manifest(n)),
        ("n/sets.toml", &sets_toml(&[("render/common", 2)])),
        (
            "n/render/common.jsonl",
            &(render_line("n/render/common-1", n) + &differing_line("n/render/common-2", n)),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let listed = BTreeMap::from([("*/render/common-2", "recorded with another pre-tokenizer")]);
    let report = compare_quietly(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
    assert_eq!(report.differences.len(), 2, "{:#?}", report.differences);
    // The suffix covers the case of both models, so nothing fails the run.
    assert_eq!(
        report.failures(root.path(), &listed, &BTreeMap::new()),
        Vec::<String>::new()
    );
    // Without it the two are unlisted.
    let failures = report.failures(root.path(), &BTreeMap::new(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("m/render/common-2") && failures[0].contains("n/render/common-2"),
        "{failures:#?}"
    );
    // A suffix whose every loaded case matches must go, and so must one that
    // no loaded case ends with.
    let stale = BTreeMap::from([
        ("*/render/common-1", "matches now"),
        ("*/render/common-2", "recorded with another pre-tokenizer"),
        ("*/render/common-3", "no such case"),
    ]);
    let failures = report.failures(root.path(), &stale, &BTreeMap::new());
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures.iter().any(|failure| {
            failure.contains("matching the reference now")
                && failure.contains("*/render/common-1")
                && !failure.contains("common-2")
        }),
        "{failures:#?}"
    );
    assert!(
        failures.iter().any(|failure| {
            failure.contains("no longer among the loaded fixtures")
                && failure.contains("*/render/common-3")
                && !failure.contains("common-2")
        }),
        "{failures:#?}"
    );
}

#[test]
fn stdout_has_one_line_per_model_and_the_report_every_case() {
    // "m" has one listed difference and SHOWN + 1 unlisted ones, "n" no sets,
    // "o" one matching case that the list says differs.
    let (m, o) = ("org/m", "org/o");
    let mut gsm8k = String::new();
    for i in 0..=SHOWN {
        gsm8k += &differing_line(&format!("m/render/gsm8k-{i}"), m);
    }
    let root = tree(&[
        ("m/manifest.toml", &manifest(m)),
        (
            "m/sets.toml",
            &sets_toml(&[("render/common", 1), ("render/gsm8k", SHOWN + 1)]),
        ),
        (
            "m/render/common.jsonl",
            &differing_line("m/render/common-0", m),
        ),
        ("m/render/gsm8k.jsonl", &gsm8k),
        ("n/manifest.toml", &manifest("org/n")),
        ("o/manifest.toml", &manifest(o)),
        ("o/sets.toml", &sets_toml(&[("render/common", 1)])),
        (
            "o/render/common.jsonl",
            &render_line("o/render/common-0", o),
        ),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let listed = BTreeMap::from([
        ("m/render/common-0", "the reference ids are not the text's"),
        ("o/render/common-0", "matches now"),
    ]);
    let (mut stdout, mut report) = (Vec::new(), Vec::new());
    compare(
        root.path(),
        &manifests,
        &listed,
        |_, _| Ok(mock()),
        &mut Output {
            stdout: &mut stdout,
            report: &mut report,
            verbose: false,
        },
    )
    .unwrap();
    let (stdout, report) = (
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(report).unwrap(),
    );
    // stdout: m's line and its first SHOWN unlisted differences, o's line, the
    // count of the models without sets, the totals; no listed difference, no
    // set read.
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines[0],
        format!(
            "m: encode: {} cases ({} ids) compared, 0 match, {} differ, 1 listed, {} unlisted; \
             incremental decode: no fixtures; 0 stale",
            SHOWN + 2,
            SHOWN + 2,
            SHOWN + 2,
            SHOWN + 1
        ),
        "{stdout}"
    );
    assert!(
        lines[1..=SHOWN]
            .iter()
            .all(|line| line.starts_with("  differs m/render/gsm8k-")),
        "{stdout}"
    );
    assert_eq!(
        lines[SHOWN + 1..].join("\n"),
        format!(
            "  ... and 1 more unlisted differences of m, every one in the per-case report\n\
             o: encode: 1 cases (1 ids) compared, 1 match, 0 differ, 0 listed, 0 unlisted; \
             incremental decode: no fixtures; 1 stale\n\
             1 models without render or parse sets\n\
             {} cases compared, 1 match, {} differ",
            SHOWN + 3,
            SHOWN + 2
        ),
        "{stdout}"
    );
    assert!(
        !stdout.contains("known:") && !stdout.contains("common"),
        "{stdout}"
    );
    // The report: every difference, with its reason when listed, each set
    // read, the summary with every model, the totals.
    assert!(
        report.contains("  differs m/render/common-0: ")
            && report.contains("\n          known: the reference ids are not the text's\n"),
        "{report}"
    );
    assert_eq!(
        report.matches("  differs m/render/gsm8k-").count(),
        SHOWN + 1,
        "{report}"
    );
    assert!(
        report.contains("\n  render/common: 1 read, sets.toml lists 1")
            && report.contains("\nsummary:\n  m: ")
            && report.contains("\n  n: no render or parse sets\n")
            && report.ends_with(&format!(
                "\n{} cases compared, 1 match, {} differ\n",
                SHOWN + 3,
                SHOWN + 2
            )),
        "{report}"
    );
    // Verbose, stdout gets the report too.
    let (mut stdout, mut again) = (Vec::new(), Vec::new());
    compare(
        root.path(),
        &manifests,
        &listed,
        |_, _| Ok(mock()),
        &mut Output {
            stdout: &mut stdout,
            report: &mut again,
            verbose: true,
        },
    )
    .unwrap();
    assert_eq!(stdout, again);
    assert_eq!(String::from_utf8(again).unwrap(), report);
}

#[test]
fn a_failure_shows_its_first_lines_and_counts_the_rest() {
    let report = Path::new("reports/tokenizer.txt");
    assert_eq!(capped("header\na\nb", 2, report), "header\na\nb");
    assert_eq!(
        capped("header\na\nb\nc\nd", 2, report),
        "header\na\nb\n... and 2 more lines, every one in reports/tokenizer.txt"
    );
}

#[test]
fn the_fan_out_hands_the_items_over_in_their_order_however_they_finish() {
    // Item 0 waits until item 1 is done, so on two threads item 1 finishes
    // first and items 2 and 3 may follow it, all before item 0; `then` still
    // sees them in the items' order.
    let item_1_done = AtomicBool::new(false);
    let items = [0, 1, 2, 3];
    let mut handed = Vec::new();
    let ran = in_order_on(
        2,
        &items,
        |&item| {
            if item == 0 {
                let deadline = Instant::now() + Duration::from_secs(10);
                while !item_1_done.load(Ordering::Relaxed) {
                    assert!(Instant::now() < deadline, "item 1 never finished");
                    std::thread::yield_now();
                }
            }
            if item == 1 {
                item_1_done.store(true, Ordering::Relaxed);
            }
            Ok(item * 10)
        },
        |outcome| handed.push(outcome),
    );
    assert_eq!(ran, Ok(()));
    assert_eq!(handed, [0, 10, 20, 30]);
}

#[test]
fn an_error_in_the_fan_out_leaves_the_other_threads_their_item_in_flight_and_no_more() {
    // Item 0 fails at once; every other item waits for that failure and then
    // a while longer, so the flag is up before any other thread looks for its
    // next item: on two threads, the other thread finishes the one item it
    // had and takes none of the six left, and the error reaches the caller.
    let item_0_failed = AtomicBool::new(false);
    let started = AtomicUsize::new(0);
    let items: Vec<usize> = (0..8).collect();
    let ran = in_order_on(
        2,
        &items,
        |&item| {
            if item == 0 {
                item_0_failed.store(true, Ordering::Relaxed);
                return Err("item 0".to_string());
            }
            started.fetch_add(1, Ordering::Relaxed);
            let deadline = Instant::now() + Duration::from_secs(10);
            while !item_0_failed.load(Ordering::Relaxed) {
                assert!(Instant::now() < deadline, "item 0 never failed");
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(200));
            Ok(item)
        },
        |_| {},
    );
    assert_eq!(ran, Err("item 0".to_string()));
    let started = started.load(Ordering::Relaxed);
    assert!(
        started <= 1,
        "{started} items started beside the one that failed; the other thread had one in flight"
    );
}

/// What the loopback server does with one connection once it has read the request: answers
/// with the head and the body `ok`; closes it before any answer, as a connection the hub
/// dropped does (the client reads `error sending request`); or announces a long body and cuts
/// it short (the client reads `error decoding response body`).
#[derive(Clone, Copy)]
enum Reply {
    Head(&'static str),
    NoAnswer,
    CutBody,
}

/// A loopback server for the loaders' tests: it answers its connections with `replies` in order
/// (each a response head whole with the body `ok`, or a connection broken below HTTP), then
/// stops, so a request past the last answer is refused; the count is the requests read, bumped
/// before the answer goes out. A failure on the server's side ends it, and the test reads it in
/// the count or in a refused connection.
fn loopback(replies: &'static [Reply]) -> io::Result<(String, Arc<AtomicUsize>)> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let url = format!("http://{}", listener.local_addr()?);
    let served = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&served);
    std::thread::spawn(move || {
        for reply in replies {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") && matches!(stream.read(&mut byte), Ok(1)) {
                request.push(byte[0]);
            }
            count.fetch_add(1, Ordering::SeqCst);
            let answer = match reply {
                Reply::Head(head) => {
                    format!("{head}\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                }
                Reply::NoAnswer => continue,
                Reply::CutBody => {
                    "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nConnection: close\r\n\r\nok"
                        .to_string()
                }
            };
            if stream.write_all(answer.as_bytes()).is_err() {
                return;
            }
        }
    });
    Ok((url, served))
}

/// A GET the hub rate-limits is sent again after the wait the answer asks for, and the answer
/// past the limit is the one returned; after the last try, the limit itself is.
#[test]
fn a_rate_limited_get_is_sent_again_after_the_retry_after_wait() {
    const LIMITED: Reply = Reply::Head("HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0");
    const LIMITED_EVERY_TIME: &[Reply] = &[LIMITED; RATE_LIMITED_TRIES as usize];
    let client = common::http_client();
    let (url, served) = loopback(&[LIMITED, Reply::Head("HTTP/1.1 200 OK")]).unwrap();
    let answer = fetch(
        &client,
        &format!("{url}/org/m/resolve/r/tokenizer_config.json"),
    )
    .unwrap();
    assert_eq!(answer.status, reqwest::StatusCode::OK);
    assert_eq!(answer.body, b"ok");
    assert_eq!(served.load(Ordering::SeqCst), 2);
    let (url, served) = loopback(LIMITED_EVERY_TIME).unwrap();
    let answer = fetch(
        &client,
        &format!("{url}/org/m/resolve/r/tokenizer_config.json"),
    )
    .unwrap();
    assert_eq!(answer.status, reqwest::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(served.load(Ordering::SeqCst), RATE_LIMITED_TRIES as usize);
}

/// The wait between two tries: `Retry-After` in seconds first, else the seconds left in the
/// hub's window from the `ratelimit` header plus one (whole seconds: a try at the boundary is
/// limited again), else ten seconds; never more than one window (an HTTP-date `Retry-After` is
/// not a number and falls through).
#[test]
fn the_wait_after_a_rate_limit_is_what_the_answer_says_and_a_window_at_most() {
    let headers = |pairs: &[(&'static str, &str)]| {
        let mut map = reqwest::header::HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().unwrap());
        }
        map
    };
    assert_eq!(rate_limit_wait(&headers(&[])), Duration::from_secs(10));
    assert_eq!(
        rate_limit_wait(&headers(&[("retry-after", "7")])),
        Duration::from_secs(7)
    );
    assert_eq!(
        rate_limit_wait(&headers(&[("ratelimit", "\"resolvers\";r=0;t=240")])),
        Duration::from_secs(241)
    );
    assert_eq!(
        rate_limit_wait(&headers(&[
            ("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT"),
            ("ratelimit", "\"resolvers\";r=0;t=30"),
        ])),
        Duration::from_secs(31)
    );
    assert_eq!(
        rate_limit_wait(&headers(&[
            ("retry-after", "7"),
            ("ratelimit", "\"resolvers\";r=0;t=240")
        ])),
        Duration::from_secs(7)
    );
    assert_eq!(
        rate_limit_wait(&headers(&[("ratelimit", "\"resolvers\";r=0;t=1000")])),
        RATE_LIMIT_WINDOW
    );
}

/// A cached download the hub rate-limits once lands on the second try, and a call after it
/// reads the cache; a download that failed under the download lock (limited at every try)
/// leaves no file and no poisoned lock behind, so the test after it downloads instead of
/// failing on the poison.
#[test]
fn a_cached_download_outlives_a_rate_limit_and_a_failed_download_before_it() {
    const LIMITED: Reply = Reply::Head("HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0");
    const LIMITED_EVERY_TIME: &[Reply] = &[LIMITED; RATE_LIMITED_TRIES as usize];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    let (url, served) = loopback(LIMITED_EVERY_TIME).unwrap();
    let failed = std::thread::spawn({
        let path = path.clone();
        move || cached_download(&common::http_client(), &url, &path, 2)
    })
    .join();
    assert!(
        failed.is_err(),
        "limited at every try, the download must fail"
    );
    assert_eq!(served.load(Ordering::SeqCst), RATE_LIMITED_TRIES as usize);
    assert!(!path.exists(), "a failed download leaves no file");
    let (url, served) = loopback(&[LIMITED, Reply::Head("HTTP/1.1 200 OK")]).unwrap();
    cached_download(&common::http_client(), &url, &path, 2);
    assert_eq!(fs::read(&path).unwrap(), b"ok");
    assert_eq!(served.load(Ordering::SeqCst), 2);
    cached_download(&common::http_client(), &url, &path, 2);
    assert_eq!(
        served.load(Ordering::SeqCst),
        2,
        "a cached file is not fetched again"
    );
}

/// A file the checkpoint does not serve is asked for once: the answer is remembered beside its
/// name, and the next run reads it without a request.
#[test]
fn a_file_the_checkpoint_does_not_serve_is_asked_for_once() {
    let client = common::http_client();
    let (url, served) = loopback(&[Reply::Head("HTTP/1.1 404 Not Found")]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for _ in 0..2 {
        let got = download(&client, &url, "org/m", "r", dir.path(), "config.json", 2).unwrap();
        assert!(got.is_none());
    }
    assert!(dir.path().join("config.json.absent").is_file());
    assert_eq!(served.load(Ordering::SeqCst), 1);
}

/// A GET whose body breaks below HTTP is sent again after a short wait and the file lands whole
/// on the second try; a whole first answer costs one request and no wait.
#[test]
#[expect(
    clippy::print_stdout,
    reason = "the requests counted and the times measured are test diagnostic output"
)]
fn a_download_whose_body_breaks_is_sent_again_and_lands_on_the_second_try() {
    let client = common::http_client();
    let dir = tempfile::tempdir().unwrap();
    let (url, served) = loopback(&[Reply::CutBody, Reply::Head("HTTP/1.1 200 OK")]).unwrap();
    let started = Instant::now();
    let path = download(&client, &url, "org/m", "r", dir.path(), "tokenizer.json", 2)
        .unwrap()
        .expect("the file lands on the second try");
    let elapsed = started.elapsed();
    assert_eq!(fs::read(&path).unwrap(), b"ok");
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert!(
        elapsed >= TRANSPORT_ERROR_WAIT,
        "the wait before the second try was {elapsed:?}"
    );
    assert!(
        !dir.path()
            .join(format!("tokenizer.json.{}.part", std::process::id()))
            .exists(),
        "no part is left behind"
    );
    println!("a body broken once: 2 requests, the file landed after {elapsed:?}");
    let (url, served) = loopback(&[Reply::Head("HTTP/1.1 200 OK")]).unwrap();
    let started = Instant::now();
    download(&client, &url, "org/m", "r", dir.path(), "config.json", 2)
        .unwrap()
        .expect("the file lands on the first try");
    let elapsed = started.elapsed();
    assert_eq!(
        served.load(Ordering::SeqCst),
        1,
        "a whole first answer is one request"
    );
    println!("a whole first answer: 1 request, the file landed after {elapsed:?}");
}

/// `answered` is what decides whether a loader fails or skips: a body the hub cut under the read
/// on every try is an answered error; a hub that never answers is not; and a try that fails to
/// connect after answered tries does not turn them into a skip, since `fetch` keeps the first
/// answered error.
#[test]
fn a_broken_body_reads_as_answered_and_no_answer_does_not() {
    const CUT_EVERY_TIME: &[Reply] = &[
        Reply::CutBody,
        Reply::CutBody,
        Reply::CutBody,
        Reply::CutBody,
    ];
    const NO_ANSWER_EVER: &[Reply] = &[
        Reply::NoAnswer,
        Reply::NoAnswer,
        Reply::NoAnswer,
        Reply::NoAnswer,
    ];
    const ANSWERED_THEN_NOT: &[Reply] = &[
        Reply::CutBody,
        Reply::NoAnswer,
        Reply::NoAnswer,
        Reply::NoAnswer,
    ];
    assert_eq!(CUT_EVERY_TIME.len(), TRANSPORT_ERROR_RETRIES as usize + 1);
    let client = common::http_client();
    let fetch_error = |replies: &'static [Reply]| {
        let (url, served) = loopback(replies).unwrap();
        let error = fetch(&client, &format!("{url}/org/m/resolve/r/tokenizer.json")).unwrap_err();
        assert_eq!(served.load(Ordering::SeqCst), replies.len());
        error
    };
    assert!(
        answered(&fetch_error(CUT_EVERY_TIME)),
        "a body cut under the read is an answer"
    );
    assert!(
        !answered(&fetch_error(NO_ANSWER_EVER)),
        "no connection is no answer"
    );
    assert!(
        answered(&fetch_error(ANSWERED_THEN_NOT)),
        "the first answered error is kept past later tries that never connected"
    );
}

/// A GET that breaks below HTTP at every try (the connection closed before any answer, then the
/// body cut short) fails after the bounded retries with the waits honoured, no file and no part
/// left behind: the loader reports the model as not loaded instead of hanging.
#[test]
#[expect(
    clippy::print_stdout,
    reason = "the requests counted and the times measured are test diagnostic output"
)]
fn a_download_that_breaks_at_every_try_fails_after_the_bounded_retries() {
    const BROKEN_EVERY_TIME: &[Reply] = &[
        Reply::NoAnswer,
        Reply::CutBody,
        Reply::CutBody,
        Reply::CutBody,
    ];
    assert_eq!(
        BROKEN_EVERY_TIME.len(),
        TRANSPORT_ERROR_RETRIES as usize + 1
    );
    let client = common::http_client();
    let dir = tempfile::tempdir().unwrap();
    let (url, served) = loopback(BROKEN_EVERY_TIME).unwrap();
    let started = Instant::now();
    let error = download(&client, &url, "org/m", "r", dir.path(), "tokenizer.json", 2).unwrap_err();
    let elapsed = started.elapsed();
    assert!(
        error.starts_with(&format!("GET {url}/org/m/resolve/r/tokenizer.json: ")),
        "{error}"
    );
    assert_eq!(
        served.load(Ordering::SeqCst),
        TRANSPORT_ERROR_RETRIES as usize + 1
    );
    let waits = TRANSPORT_ERROR_WAIT * (2u32.pow(TRANSPORT_ERROR_RETRIES) - 1);
    assert!(
        elapsed >= waits,
        "the waits of {waits:?} in all were honoured: {elapsed:?}"
    );
    assert!(!dir.path().join("tokenizer.json").exists());
    assert!(!dir
        .path()
        .join(format!("tokenizer.json.{}.part", std::process::id()))
        .exists());
    println!(
        "broken at every try: {} requests, failed after {elapsed:?}: {error}",
        served.load(Ordering::SeqCst)
    );
}

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
//! bellwether's own `fixtures/` directory keeps a benchmark set as
//! `<set>.jsonl.zst` in Git LFS, so a compressed set or a Git LFS pointer under
//! the root fails the run and says to unpack first. Each case is compared as
//! its line is read, and the cases read from each set must number what the
//! model's `sets.toml` lists for it; a set the table lists that the tree lacks,
//! or a set file the table does not list, fails the run too, so it cannot pass
//! on part of the fixtures. The run prints each set with its count, each
//! difference, and a tally per model.
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
    io::{BufRead, BufReader, Read},
    ops::Bound,
    path::{Path, PathBuf},
    sync::Arc,
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer, MockTokenizer, Sequence};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";
/// The fixture kinds this test reads, each a directory of sets beside a
/// model's manifest.
const KINDS: [&str; 2] = ["render", "parse"];
/// How a file Git LFS has not fetched begins: the pointer stands where the
/// content would be.
const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// Cases known to differ from the reference, each with the reason and where
/// it is tracked: a fixture id, `<slug>/render/<name>` or `<slug>/parse/<name>`,
/// or a prefix ending in `*`, which stands for every case whose id begins
/// with what is before it (`<slug>/render/*` for a model's whole render side,
/// `<slug>/render/mgsm-th-*` for one of its sets). A prefix is for a cause
/// that covers a model or a set wholesale, a reference recorded with another
/// pre-tokenizer for instance, and is held to the rule an id is: the run
/// fails when no loaded case under it differs any more, and when no loaded
/// case is under it at all.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[
    (
        "qwen-agentworld-35b-a3b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen-drive-1.0-4b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen3.5-27b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen3.5-2b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen3.5-9b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen3.8-2.4t-a95b/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "qwen3.8-flash-next/render/*",
        "the reference's ids follow transformers' Qwen2Tokenizer pattern, which splits \
         combining marks from their letters where tokenizer.json keeps them \
         (smg-project/smg-lab#56); re-recorded with the file's encoder in \
         smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "phi-4-multimodal-instruct/render/*",
        "the reference's ids follow transformers' GPT2Tokenizer pattern, which splits \
         contractions, digit runs and punctuation as GPT-2 does where tokenizer.json carries \
         an o200k-style regex (smg-project/smg-lab#102); re-recorded with the file's encoder \
         in smg-project/bellwether#104, the pin bump that follows removes this entry",
    ),
    (
        "tinyllama-1.1b-chat-v1.0/render/*",
        "the reference's ids follow transformers' LlamaTokenizer with legacy=false, which \
         prepends the dummy space once per input where tokenizer.json's normalizer prepends \
         it per segment, so no dummy space follows a special token (smg-project/smg-lab#104); \
         re-recorded with the file's encoder in smg-project/bellwether#104, the pin bump that \
         follows removes this entry",
    ),
];

/// Models whose tokenizer is known not to load, with the reason and where it
/// is tracked, so the run compares the others: an unlisted model that does
/// not load fails the run, and so does a listed one whose tokenizer loads
/// now, so the list cannot rot.
const KNOWN_UNLOADED: &[(&str, &str)] = &[];

/// The reason `known` lists for `id`: the entry that is the id itself, else
/// the first prefix entry the id begins with; none when the id is not listed.
fn known_reason<'a>(known: &BTreeMap<&'a str, &'a str>, id: &str) -> Option<&'a str> {
    known.get(id).copied().or_else(|| {
        known
            .iter()
            .find(|(entry, _)| prefix_of(entry).is_some_and(|prefix| id.starts_with(prefix)))
            .map(|(_, reason)| *reason)
    })
}

/// What a prefix entry stands for: the part before its trailing `*`; none for
/// an entry that is one fixture id.
fn prefix_of(entry: &str) -> Option<&str> {
    entry.strip_suffix('*')
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
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.loaded {
            return f.write_str("no fixtures");
        }
        write!(
            f,
            "{} cases ({} ids) compared, {} match, {} differ",
            self.cases,
            self.ids,
            self.cases - self.differ,
            self.differ
        )
    }
}

/// What a run compared, and what it could not.
#[derive(Default)]
struct Report {
    /// `<slug>/<kind>` of each kind whose sets were read.
    loaded_dirs: BTreeSet<String>,
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
}

impl Report {
    /// Count one compared case into `tally`, and keep and print its
    /// difference if it has one.
    #[expect(
        clippy::print_stdout,
        reason = "a difference is test diagnostic output, printed as it is found"
    )]
    fn record(
        &mut self,
        id: String,
        outcome: Result<(), String>,
        ids: usize,
        tally: &mut Tally,
        known: &BTreeMap<&str, &str>,
    ) {
        assert!(
            self.seen.insert(id.clone()),
            "{id}: recorded more than once"
        );
        tally.cases += 1;
        tally.ids += ids;
        if let Err(why) = outcome {
            tally.differ += 1;
            let listed = known_reason(known, &id).map_or(String::new(), |reason| {
                format!("\n          known: {reason}")
            });
            println!("  differs {id}: {why}{listed}");
            self.differences.insert(id, why);
        }
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
        let unexpected: Vec<String> = self
            .differences
            .iter()
            .filter(|(id, _)| known_reason(known, id).is_none())
            .map(|(id, why)| format!("{id}: {why}"))
            .collect();
        if !unexpected.is_empty() {
            failures.push(format!(
                "differences not listed in KNOWN_DIFFERENCES:\n{}",
                unexpected.join("\n")
            ));
        }
        let healed: Vec<&str> = known
            .keys()
            .copied()
            .filter(|entry| match prefix_of(entry) {
                Some(prefix) => self.seen_under(prefix) && !self.differs_under(prefix),
                None => self.seen.contains(*entry) && !self.differences.contains_key(*entry),
            })
            .collect();
        if !healed.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but matching the reference now; remove: {}",
                healed.join(", ")
            ));
        }
        let gone: Vec<&str> = known
            .keys()
            .copied()
            .filter(|entry| {
                let dir = entry.rsplit_once('/').map_or("", |(dir, _)| dir);
                self.loaded_dirs.contains(dir)
                    && match prefix_of(entry) {
                        Some(prefix) => !self.seen_under(prefix),
                        None => !self.seen.contains(*entry),
                    }
            })
            .collect();
        if !gone.is_empty() {
            failures.push(format!(
                "listed in KNOWN_DIFFERENCES but no longer among the loaded fixtures; remove or \
                 rename: {}",
                gone.join(", ")
            ));
        }
        failures
    }

    /// Whether a compared case's id begins with `prefix`.
    fn seen_under(&self, prefix: &str) -> bool {
        self.seen
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .is_some_and(|id| id.starts_with(prefix))
    }

    /// Whether a differing case's id begins with `prefix`.
    fn differs_under(&self, prefix: &str) -> bool {
        self.differences
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .is_some_and(|(id, _)| id.starts_with(prefix))
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
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
    let report =
        compare(&root, &manifests, &known, load_tokenizer).unwrap_or_else(|e| panic!("{e}"));
    let failures = report.failures(&root, &known, &known_unloaded);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Compare each model's render and parse sets under `root` with the tokenizer
/// `load` gives for the model, each case as its line is read, and the cases
/// read from each set with the model's `sets.toml`. A model whose tokenizer
/// does not load is kept in the report and the run goes on. Prints each set
/// read, each difference, and a tally per model.
#[expect(
    clippy::print_stdout,
    reason = "the per-set and per-model report is test diagnostic output"
)]
fn compare(
    root: &Path,
    manifests: &[(String, Manifest)],
    known: &BTreeMap<&str, &str>,
    load: impl Fn(&str, &Manifest) -> Result<(Arc<dyn Tokenizer>, String), String>,
) -> Result<Report, String> {
    let mut report = Report::default();
    let mut summaries = Vec::new();
    for (slug, manifest) in manifests {
        let model_dir = root.join(slug);
        let sets = set_files(&model_dir)?;
        let listed = read_set_table(&model_dir)?;
        if sets.is_empty() {
            // Nothing to compare; a set the table lists is missing.
            let listed = listed.unwrap_or_default();
            let mismatches = set_mismatches(slug, &BTreeMap::new(), &listed);
            report.mismatches.extend(mismatches);
            summaries.push(format!("{slug}: no render or parse sets"));
            continue;
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
                summaries.push(format!("{slug}: not compared, its tokenizer did not load"));
                report.unloaded.push((slug.clone(), why));
                continue;
            }
        };
        println!(
            "{slug}: {} at {} from {from}",
            manifest.model, manifest.revision
        );
        let mut encode = Tally::default();
        let mut decode = Tally::default();
        let mut read = BTreeMap::new();
        for (kind, set, path) in &sets {
            report.loaded_dirs.insert(format!("{slug}/{kind}"));
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
                    report.record(case.id, outcome, ids, &mut encode, known);
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
                    report.record(case.id, outcome, ids, &mut decode, known);
                })?
            };
            let tokenizers = tokenizers.map_or(String::new(), |version| {
                format!("; recorded with tokenizers {version}")
            });
            let counted = counts(Some(cases), listed.get(set).copied());
            println!("  {set}: {counted}{tokenizers}");
            read.insert(set.clone(), cases);
        }
        report
            .mismatches
            .extend(set_mismatches(slug, &read, &listed));
        summaries.push(format!(
            "{slug}: encode: {encode}; incremental decode: {decode}"
        ));
    }
    println!("summary:");
    for summary in &summaries {
        println!("  {summary}");
    }
    println!(
        "{} cases compared, {} match, {} differ",
        report.seen.len(),
        report.seen.len() - report.differences.len(),
        report.differences.len()
    );
    Ok(report)
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
    let client = reqwest::blocking::Client::new();
    download(&client, model, revision, &dir, "tokenizer_config.json", 100)?
        .ok_or_else(|| format!("{model} at {revision} serves no tokenizer_config.json"))?;
    download(&client, model, revision, &dir, "config.json", 2)?;
    let mut vocabulary = None;
    for files in VOCABULARIES {
        let mut whole = true;
        for file in files {
            whole &= download(&client, model, revision, &dir, file, 100_000)?.is_some();
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

/// Fetches `file` into `dir` unless it is there already; `Ok(None)` when the
/// checkpoint does not serve it (HTTP 404), an error for any other failure
/// or a file shorter than `min_bytes`.
fn download(
    client: &reqwest::blocking::Client,
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
    let url = format!("https://huggingface.co/{model}/resolve/{revision}/{file}");
    let response = client
        .get(&url)
        .send()
        .map_err(|e| format!("GET {url}: {e}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(format!("GET {url}: HTTP {}", response.status()));
    }
    let bytes = response.bytes().map_err(|e| format!("GET {url}: {e}"))?;
    if bytes.len() < min_bytes {
        return Err(format!(
            "{url}: {} bytes, expected at least {min_bytes}",
            bytes.len()
        ));
    }
    let part = dir.join(format!("{file}.part"));
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
fn tree(files: &[(&str, &str)]) -> std::io::Result<tempfile::TempDir> {
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
    let report = compare(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock())).unwrap();
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
    let error = compare(root.path(), &manifests, &BTreeMap::new(), |_, _| Ok(mock()))
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
    let report = compare(
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
    let report = compare(root.path(), &manifests, &listed, |_, _| Ok(mock())).unwrap();
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
    // that no loaded case begins with.
    let stale = BTreeMap::from([
        ("m/render/common-*", "matches now"),
        ("m/render/mgsm-bn-*", "no such set"),
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
                && failure.contains("m/render/mgsm-bn-*")
                && !failure.contains("mgsm-th")
        }),
        "{failures:#?}"
    );
}

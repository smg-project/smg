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
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory
//! of a bellwether checkout; without it the test prints a skip notice and
//! passes. Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`. That download needs the
//! TLS backend a workspace build enables: under `cargo test -p llm-tokenizer`
//! alone the crate's `reqwest` dev-dependency has none, so with a cold cache
//! run it from the workspace or fill the Hugging Face cache first.
//!
//! A difference is a finding, not something to hide: every known one is listed
//! in [`KNOWN_DIFFERENCES`] with its reason and where it is tracked, the run
//! fails on any other, on a listed case that starts matching, and on a listed
//! case that the loaded fixtures no longer contain, so the list cannot rot.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer, Sequence};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";

/// Cases known to differ from the reference, by fixture id, each with the
/// reason and where it is tracked.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[];

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

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn encode_and_incremental_decode_match_the_reference() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let manifests = read_manifests(&root).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        !manifests.is_empty(),
        "no fixtures/<slug>/manifest.toml under {}",
        root.display()
    );

    let known: BTreeMap<&str, &str> = KNOWN_DIFFERENCES.iter().copied().collect();
    for id in known.keys() {
        let mut parts = id.splitn(3, '/');
        let (slug, kind, name) = (parts.next(), parts.next(), parts.next());
        assert!(
            slug.is_some_and(|s| !s.is_empty())
                && matches!(kind, Some("render" | "parse"))
                && name.is_some_and(|n| !n.is_empty() && !n.contains('/')),
            "KNOWN_DIFFERENCES entry {id} is not <slug>/render/<name> or <slug>/parse/<name>"
        );
    }
    let mut loaded_dirs = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut differences = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut record = |id: String, outcome: Result<(), String>, ids: usize, tally: &mut Tally| {
        assert!(seen.insert(id.clone()), "{id}: recorded more than once");
        tally.cases += 1;
        tally.ids += ids;
        match outcome {
            Ok(()) => println!("  match   {id}"),
            Err(why) => {
                tally.differ += 1;
                let listed = known.get(id.as_str()).map_or(String::new(), |reason| {
                    format!("\n          known: {reason}")
                });
                println!("  differs {id}: {why}{listed}");
                differences.insert(id, why);
            }
        }
    };
    for (slug, manifest) in &manifests {
        let render_dir = root.join(slug).join("render");
        let parse_dir = root.join(slug).join("parse");
        if !render_dir.is_dir() && !parse_dir.is_dir() {
            summaries.push(format!("{slug}: no render or parse directory"));
            continue;
        }
        let dir = tokenizer_dir(&manifest.model, &manifest.revision, slug)
            .unwrap_or_else(|e| panic!("{slug}: {e}"));
        let dir_str = dir
            .to_str()
            .unwrap_or_else(|| panic!("{slug}: tokenizer path {} is not UTF-8", dir.display()));
        let tok = create_tokenizer(dir_str)
            .unwrap_or_else(|e| panic!("{slug}: the tokenizer at {dir_str} should load: {e}"));
        println!(
            "{slug}: {} at {} from {dir_str}",
            manifest.model, manifest.revision
        );

        let mut encode = Tally::default();
        if render_dir.is_dir() {
            loaded_dirs.insert(format!("{slug}/render"));
            encode.loaded = true;
            let cases: Vec<RenderCase> =
                read_cases(&render_dir).unwrap_or_else(|e| panic!("{slug}: {e}"));
            println!(
                "  encode: {} render cases recorded with tokenizers {}",
                cases.len(),
                recorded_with(cases.first().map(|c| &c.reference.provenance))
            );
            for case in cases {
                assert_eq!(case.kind, "render", "{}: not a render case", case.id);
                assert_eq!(
                    case.model, manifest.model,
                    "{}: model differs from the manifest",
                    case.id
                );
                let outcome = check_encode(tok.as_ref(), &case.reference);
                record(
                    case.id,
                    outcome,
                    case.reference.input_ids.len(),
                    &mut encode,
                );
            }
        }

        let mut decode = Tally::default();
        if parse_dir.is_dir() {
            loaded_dirs.insert(format!("{slug}/parse"));
            decode.loaded = true;
            let cases: Vec<ParseCase> =
                read_cases(&parse_dir).unwrap_or_else(|e| panic!("{slug}: {e}"));
            println!(
                "  incremental decode: {} parse cases recorded with tokenizers {}",
                cases.len(),
                recorded_with(cases.first().map(|c| &c.reference.provenance))
            );
            for case in cases {
                assert_eq!(case.kind, "parse", "{}: not a parse case", case.id);
                assert_eq!(
                    case.model, manifest.model,
                    "{}: model differs from the manifest",
                    case.id
                );
                let outcome = check_decode(&tok, &case);
                record(case.id, outcome, case.output_ids.len(), &mut decode);
            }
        }
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
        seen.len(),
        seen.len() - differences.len(),
        differences.len()
    );
    assert!(
        !seen.is_empty(),
        "no render or parse case under {}: every fixtures/<slug>/render and fixtures/<slug>/parse \
         directory is missing or empty",
        root.display()
    );

    let unexpected: Vec<String> = differences
        .iter()
        .filter(|(id, _)| !known.contains_key(id.as_str()))
        .map(|(id, why)| format!("{id}: {why}"))
        .collect();
    assert!(
        unexpected.is_empty(),
        "differences not listed in KNOWN_DIFFERENCES:\n{}",
        unexpected.join("\n")
    );
    let healed: Vec<&str> = known
        .keys()
        .copied()
        .filter(|id| seen.contains(*id) && !differences.contains_key(*id))
        .collect();
    assert!(
        healed.is_empty(),
        "listed in KNOWN_DIFFERENCES but matching the reference now; remove: {}",
        healed.join(", ")
    );
    let gone: Vec<&str> = known
        .keys()
        .copied()
        .filter(|id| {
            let dir = id.rsplit_once('/').map_or("", |(dir, _)| dir);
            loaded_dirs.contains(dir) && !seen.contains(*id)
        })
        .collect();
    assert!(
        gone.is_empty(),
        "listed in KNOWN_DIFFERENCES but no longer among the loaded fixtures; remove or rename: {}",
        gone.join(", ")
    );
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
fn recorded_with(provenance: Option<&Value>) -> &str {
    provenance
        .and_then(|p| p.get("tokenizers"))
        .and_then(Value::as_str)
        .unwrap_or("an unrecorded version")
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

fn read_cases<T: DeserializeOwned>(dir: &Path) -> Result<Vec<T>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "jsonl") {
            files.push(path);
        }
    }
    files.sort();
    let mut cases = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            cases.push(case);
        }
    }
    Ok(cases)
}

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download of the two
/// files the tokenizer loads, `tokenizer.json` and `tokenizer_config.json`.
/// A download is written beside its name and renamed into place, so an
/// interrupted write never leaves a short file the next run would trust.
fn tokenizer_dir(model: &str, revision: &str, slug: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = hf_cache_snapshot(model, revision) {
        return Ok(snapshot);
    }
    let dir = PathBuf::from(CACHE_DIR).join(slug).join(revision);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let client = reqwest::blocking::Client::new();
    for (file, min_bytes) in [
        ("tokenizer.json", 100_000usize),
        ("tokenizer_config.json", 100),
    ] {
        let path = dir.join(file);
        if path.is_file() {
            continue;
        }
        let url = format!("https://huggingface.co/{model}/resolve/{revision}/{file}");
        let response = client
            .get(&url)
            .send()
            .map_err(|e| format!("GET {url}: {e}"))?;
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
    }
    Ok(dir)
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
    let dir = hub
        .join(format!("models--{}", model.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    (dir.join("tokenizer.json").is_file() && dir.join("tokenizer_config.json").is_file())
        .then_some(dir)
}

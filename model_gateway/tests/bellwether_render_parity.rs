//! Render parity with the bellwether reference fixtures, through the gateway's
//! chat request path.
//!
//! bellwether (smg-project/bellwether) records, for each model at a pinned
//! Hugging Face revision, what the checkpoint's own chat template renders for
//! a corpus of chat requests: the prompt text and its token ids. This test
//! turns each recorded request into the `ChatCompletionRequest` the HTTP layer
//! would hand on (deserialized, normalized and validated as `ValidatedJson`
//! does, tools narrowed by `tool_choice` as the chat preparation stage does),
//! renders it with `process_chat_messages`, the entry point the gateway and the
//! bindings share (content format, typed tools, tool-call arguments parsed for
//! renderers that do not take them raw, template kwargs, the thinking toggle,
//! `continue_final_message`), encodes the text as the gateway's tokenize step
//! does, and compares text and ids with the reference byte for byte.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the tree `bellwether
//! unpack` writes (`fixtures-plain/` unless `--out` names another), where
//! every fixture set is plain JSON Lines beside each model's `manifest.toml`
//! and `sets.toml`; without it the test prints a skip notice and passes.
//! bellwether's own `fixtures/` directory keeps a benchmark set as
//! `<set>.jsonl.zst` in Git LFS, so a compressed set or a Git LFS pointer under
//! the root fails the run and says to unpack first. Each case is compared as
//! its line is read, and the cases read from each render set must number what
//! the model's `sets.toml` lists for it; a render set the table lists that the
//! tree lacks, or a render set file the table does not list, fails the run
//! too, so it cannot pass on part of the fixtures. The run prints each set
//! with its count, each difference, and a tally per model.
//!
//! Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`. A model whose tokenizer
//! does not load is listed when the run fails at the end, after every other
//! model has been compared.
//!
//! A difference is a finding, not something to hide: every known one is listed
//! in [`KNOWN_DIFFERENCES`] with its reason and where it is tracked, the run
//! fails on any other, on a listed case that starts matching, and on a listed
//! case that the loaded fixtures no longer contain, so the list cannot rot.
//!
//! What the test cannot see: the public entry point returns the rendered text
//! and not the deferred encode a segment-aware renderer prepares, so for such
//! a renderer the ids compared here are a flat encode of the text, which the
//! gateway itself warns may not reproduce its own ids. The two models recorded
//! so far render to text that is encoded flat.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer, MockTokenizer};
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;
use smg::routers::grpc::utils::process_chat_messages;
use validator::Validate;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";
/// The fixture kinds this test reads, each a directory of sets beside a
/// model's manifest.
const KINDS: [&str; 1] = ["render"];
/// How a file Git LFS has not fetched begins: the pointer stands where the
/// content would be.
const LFS_POINTER: &[u8] = b"version https://git-lfs.github.com/spec/v1";

/// Cases known to differ from the reference, by fixture id, each with the
/// reason and where it is tracked.
const KNOWN_DIFFERENCES: &[(&str, &str)] = &[
    (
        "qwen3-8b/render/continue-final-message",
        "the gateway renders continue_final_message by popping the assistant turn and appending its \
         text after the generation header, which drops the empty think block the template writes \
         before a continued turn (smg-project/smg#2779)",
    ),
    (
        "deepseek-r1/render/continue-final-message",
        "the gateway's generation header adds `<think>\\n` before the continued text; the template \
         writes none for a continued turn (smg-project/smg#2779)",
    ),
    (
        "qwen3-8b/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
    (
        "deepseek-r1/render/no-generation-prompt",
        "add_generation_prompt is not a field of SMG's chat request; the header is always appended \
         (smg-project/smg#2780)",
    ),
    (
        "deepseek-r1/render/tools-history-single-call",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-parallel-calls",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-results-reordered",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "deepseek-r1/render/tools-history-content-and-call",
        "the gateway parses tool-call arguments into objects before rendering and the R1 template \
         concatenates them as text, so the render fails (smg-project/smg#2783)",
    ),
    (
        "qwen3-8b/render/tools-call-arguments-object",
        "SMG's request schema types tool-call arguments as a string, as the API and the engines do; \
         the Qwen3 template accepts an object (smg-project/bellwether#12, needs:simo)",
    ),
];

/// `fixtures/<slug>/manifest.toml`: the model and the revision its fixtures
/// were recorded at.
#[derive(Deserialize)]
struct Manifest {
    model: String,
    revision: String,
}

/// One line of `fixtures/<slug>/render/<set>.jsonl`, bellwether's
/// `case.schema.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    id: String,
    kind: String,
    model: String,
    request: Value,
    reference: Reference,
    #[serde(default)]
    witnesses: Option<Value>,
}

#[derive(Deserialize)]
struct Reference {
    source: String,
    input_ids: Vec<u32>,
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

struct Rendered {
    text: String,
    ids: Vec<u32>,
}

/// What one model compared.
#[derive(Default)]
struct Tally {
    cases: usize,
    differ: usize,
    witnessed: usize,
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} cases compared, {} match, {} differ, {} carry engine witnesses",
            self.cases,
            self.cases - self.differ,
            self.differ,
            self.witnessed
        )
    }
}

/// What a run compared, and what it could not.
#[derive(Default)]
struct Report {
    /// Each model whose render sets were read.
    loaded_slugs: BTreeSet<String>,
    /// The id of every case compared.
    seen: BTreeSet<String>,
    /// Why each case that differs from the reference differs, by id.
    differences: BTreeMap<String, String>,
    /// How many of the cases compared carry engine witnesses.
    witnessed: usize,
    /// Each model whose tokenizer did not load, and why.
    unloaded: Vec<(String, String)>,
    /// Each render set whose cases read differ from what its model's
    /// `sets.toml` lists: `<slug>/render/<set>`, the cases read (none when the
    /// tree has no file for the set) and the cases listed (none when the table
    /// does not list it).
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
        fixture: &Fixture,
        outcome: Result<(), String>,
        tally: &mut Tally,
        known: &BTreeMap<&str, &str>,
    ) {
        self.seen.insert(fixture.id.clone());
        tally.cases += 1;
        if fixture.witnesses.is_some() {
            tally.witnessed += 1;
            self.witnessed += 1;
        }
        if let Err(why) = outcome {
            tally.differ += 1;
            let listed = known
                .get(fixture.id.as_str())
                .map_or(String::new(), |reason| {
                    format!("\n          known: {reason}")
                });
            println!(
                "  differs {} (reference {}): {why}{listed}",
                fixture.id, fixture.reference.source
            );
            self.differences.insert(fixture.id.clone(), why);
        }
    }

    /// What fails the run; empty when it passes.
    fn failures(&self, root: &Path, known: &BTreeMap<&str, &str>) -> Vec<String> {
        let mut failures = Vec::new();
        if !self.unloaded.is_empty() {
            let models: Vec<String> = self
                .unloaded
                .iter()
                .map(|(slug, why)| format!("{slug}: {why}"))
                .collect();
            failures.push(format!(
                "models not compared, because their tokenizer did not load:\n{}",
                models.join("\n")
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
                "no render case under {}: every <slug>/render directory is missing or empty",
                root.display()
            ));
        }
        let unexpected: Vec<String> = self
            .differences
            .iter()
            .filter(|(id, _)| !known.contains_key(id.as_str()))
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
            .filter(|id| self.seen.contains(*id) && !self.differences.contains_key(*id))
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
            .filter(|id| {
                let slug = id.split('/').next().unwrap_or_default();
                self.loaded_slugs.contains(slug) && !self.seen.contains(*id)
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
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice is test diagnostic output"
)]
fn render_fixtures_match_the_reference_byte_for_byte() {
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
    let report =
        compare(&root, &manifests, &known, load_tokenizer).unwrap_or_else(|e| panic!("{e}"));
    let failures = report.failures(&root, &known);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Render each model's render sets under `root` through the chat request path
/// with the tokenizer `load` gives for the model, each case as its line is
/// read, and compare the cases read from each set with the model's
/// `sets.toml`. A model whose tokenizer does not load is kept in the report
/// and the run goes on. Prints each set read, each difference, and a tally
/// per model.
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
            summaries.push(format!("{slug}: no render sets"));
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
        report.loaded_slugs.insert(slug.clone());
        println!(
            "{slug}: {} at {} from {from}",
            manifest.model, manifest.revision
        );
        let mut tally = Tally::default();
        let mut read = BTreeMap::new();
        for (_, set, path) in &sets {
            let mut transformers = None;
            let cases = for_each_case(path, |fixture: Fixture| {
                assert_eq!(fixture.kind, "render", "{}: not a render case", fixture.id);
                assert_eq!(
                    fixture.model, manifest.model,
                    "{}: model differs from the manifest",
                    fixture.id
                );
                transformers.get_or_insert_with(|| recorded_with(&fixture.reference.provenance));
                let outcome = match render(tok.as_ref(), &manifest.model, &fixture.request) {
                    Err(e) => Err(e),
                    Ok(got)
                        if got.text == fixture.reference.text
                            && got.ids == fixture.reference.input_ids =>
                    {
                        Ok(())
                    }
                    Ok(got) => Err(describe(&got, &fixture.reference)),
                };
                report.record(&fixture, outcome, &mut tally, known);
            })?;
            let transformers = transformers.map_or(String::new(), |version| {
                format!("; recorded with transformers {version}")
            });
            let counted = counts(Some(cases), listed.get(set).copied());
            println!("  {set}: {counted}{transformers}");
            read.insert(set.clone(), cases);
        }
        report
            .mismatches
            .extend(set_mismatches(slug, &read, &listed));
        summaries.push(format!("{slug}: {tally}"));
    }
    println!("summary:");
    for summary in &summaries {
        println!("  {summary}");
    }
    println!(
        "{} cases match, {} differ, {} carry engine witnesses",
        report.seen.len() - report.differences.len(),
        report.differences.len(),
        report.witnessed
    );
    Ok(report)
}

/// Hand a corpus request to the gateway's own request processing. The request
/// becomes the `ChatCompletionRequest` the HTTP layer would hand on: the
/// manifest's model added, keys SMG does not know ignored the way the gateway
/// ignores them, then normalized and validated as `ValidatedJson` does with
/// every chat request (the provider profile's rewrites, deprecated-field
/// migration, the `tool_choice` default; a request the gateway would answer
/// with 400 is a difference, not a rendering), then the tools narrowed by
/// `tool_choice` as the chat preparation stage does before rendering.
/// `process_chat_messages` then renders it exactly as the gateway does before
/// tokenizing, and the ids are the flat encode of that text, which is the
/// gateway's tokenize step for a renderer that returns text to encode.
fn render(tok: &dyn Tokenizer, model: &str, request: &Value) -> Result<Rendered, String> {
    let mut body = request.clone();
    let object = body.as_object_mut().ok_or("the request is not an object")?;
    object.insert("model".to_string(), Value::String(model.to_string()));
    let mut request: ChatCompletionRequest = serde_json::from_value(body)
        .map_err(|e| format!("the gateway does not accept this request: {e}"))?;
    request.normalize();
    request
        .validate()
        .map_err(|e| format!("the gateway rejects this request with 400: {e}"))?;
    // `filter_chat_request_by_tool_choice`, crate-private, applies this rule.
    let narrowed = match (&request.tools, &request.tool_choice) {
        (Some(tools), Some(choice)) => choice.narrow_tools(tools),
        _ => None,
    };
    if let Some(tools) = narrowed {
        request.tools = Some(tools);
    }
    let processed = process_chat_messages(&request, tok, None)?;
    let ids = tok
        .encode(&processed.text, false)
        .map_err(|e| format!("encode failed: {e}"))?
        .token_ids()
        .to_vec();
    Ok(Rendered {
        text: processed.text,
        ids,
    })
}

/// Where the rendering and the reference part: the first differing token and
/// the text around the first differing byte.
fn describe(got: &Rendered, want: &Reference) -> String {
    let id_at = got
        .ids
        .iter()
        .zip(&want.input_ids)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.ids.len().min(want.input_ids.len()));
    let byte_at = got
        .text
        .bytes()
        .zip(want.text.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.text.len().min(want.text.len()));
    format!(
        "ids part at index {id_at} ({} rendered, {} in the reference); text parts at byte {byte_at}: rendered {:?}, reference {:?}",
        got.ids.len(),
        want.input_ids.len(),
        window(&got.text, byte_at),
        window(&want.text, byte_at)
    )
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

/// The transformers version a fixture set was recorded with, from the
/// provenance of its first line.
fn recorded_with(provenance: &Value) -> String {
    provenance
        .get("transformers")
        .and_then(Value::as_str)
        .unwrap_or("an unrecorded version")
        .to_string()
}

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download of the files
/// the tokenizer and the renderer read: `tokenizer.json`,
/// `tokenizer_config.json`, the separate chat template files a checkpoint may
/// ship instead of a template inside the config, and `config.json` for
/// renderer detection. Only the first two must exist. A download is written
/// beside its name and renamed into place, so an interrupted write never
/// leaves a short file the next run would trust.
fn tokenizer_dir(model: &str, revision: &str, slug: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = hf_cache_snapshot(model, revision) {
        return Ok(snapshot);
    }
    let dir = PathBuf::from(CACHE_DIR).join(slug).join(revision);
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let client = reqwest::blocking::Client::new();
    for (file, min_bytes, required) in [
        ("tokenizer.json", 100_000usize, true),
        ("tokenizer_config.json", 100, true),
        ("chat_template.jinja", 1, false),
        ("chat_template.json", 1, false),
        ("config.json", 50, false),
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
        if response.status() == reqwest::StatusCode::NOT_FOUND && !required {
            continue;
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
    }
    Ok(dir)
}

/// `<hub cache>/models--<org>--<name>/snapshots/<revision>`, the layout
/// `huggingface_hub` keeps, under `HF_HUB_CACHE`, `HF_HOME/hub` or the default
/// `~/.cache/huggingface/hub`. A snapshot holds every file the checkpoint
/// ships, so a separate chat template file is there when one exists.
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

/// One render line whose request the mock tokenizer renders and encodes to
/// its reference.
fn render_line(id: &str, model: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"kind\":\"render\",\"model\":\"{model}\",\
         \"request\":{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}},\
         \"reference\":{{\"source\":\"hf-template\",\"input_ids\":[1],\
         \"text\":\"user: Hello\\nassistant: \"}}}}\n"
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
        ("m/render/bfcl-live.jsonl.zst", pointer),
        ("m/parse/bfcl-multiple.jsonl", pointer),
    ])
    .unwrap();
    let manifests = read_manifests(root.path()).unwrap();
    let found = packed_sets(root.path(), &manifests).unwrap();
    let under = |path: &str| root.path().join(path);
    assert_eq!(
        found,
        [
            under("m/parse/bfcl-multiple.jsonl"),
            under("m/render/bfcl-live.jsonl.zst"),
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
            ]),
        ),
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
    let failures = report.failures(root.path(), &BTreeMap::new());
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
    let failures = report.failures(root.path(), &BTreeMap::new());
    assert_eq!(failures.len(), 1, "{failures:#?}");
    assert!(
        failures[0].contains("a: no tokenizer.json"),
        "{failures:#?}"
    );
}

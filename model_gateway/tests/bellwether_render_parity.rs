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
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory
//! of a bellwether checkout; without it the test prints a skip notice and
//! passes. Tokenizer files come from the Hugging Face cache snapshot at the
//! manifest's revision when it is there, else from a one-time download into
//! `.tokenizer_cache/bellwether/<slug>/<revision>/`.
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
    fs,
    path::{Path, PathBuf},
};

use llm_tokenizer::{create_tokenizer, traits::Tokenizer};
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde::Deserialize;
use serde_json::Value;
use smg::routers::grpc::utils::process_chat_messages;
use validator::Validate;

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const CACHE_DIR: &str = ".tokenizer_cache/bellwether";

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
        "qwen3-8b/render/text-empty-user",
        "the gateway's request validation rejects an empty message content with 400; the template \
         renders it (smg-project/bellwether#13, needs:simo)",
    ),
    (
        "deepseek-r1/render/text-empty-user",
        "the gateway's request validation rejects an empty message content with 400; the template \
         renders it (smg-project/bellwether#13, needs:simo)",
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

struct Rendered {
    text: String,
    ids: Vec<u32>,
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notice and the per-case report are test diagnostic output"
)]
fn render_fixtures_match_the_reference_byte_for_byte() {
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
    let mut loaded_slugs = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut differences = BTreeMap::new();
    let mut matched = 0usize;
    let mut with_witnesses = 0usize;
    for (slug, manifest) in &manifests {
        let render_dir = root.join(slug).join("render");
        if !render_dir.is_dir() {
            continue;
        }
        loaded_slugs.insert(slug.clone());
        let dir = tokenizer_dir(&manifest.model, &manifest.revision, slug)
            .unwrap_or_else(|e| panic!("{slug}: {e}"));
        let dir_str = dir
            .to_str()
            .unwrap_or_else(|| panic!("{slug}: tokenizer path {} is not UTF-8", dir.display()));
        let tok = create_tokenizer(dir_str)
            .unwrap_or_else(|e| panic!("{slug}: the tokenizer at {dir_str} should load: {e}"));
        let fixtures = read_fixtures(&render_dir).unwrap_or_else(|e| panic!("{slug}: {e}"));
        let transformers = fixtures
            .first()
            .and_then(|f| f.reference.provenance.get("transformers"))
            .and_then(Value::as_str)
            .unwrap_or("an unrecorded version");
        println!(
            "{slug}: {} at {} from {dir_str}; {} cases recorded with transformers {transformers}",
            manifest.model,
            manifest.revision,
            fixtures.len()
        );
        for fixture in fixtures {
            assert_eq!(fixture.kind, "render", "{}: not a render case", fixture.id);
            assert_eq!(
                fixture.model, manifest.model,
                "{}: model differs from the manifest",
                fixture.id
            );
            if fixture.witnesses.is_some() {
                with_witnesses += 1;
            }
            seen.insert(fixture.id.clone());
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
            match outcome {
                Ok(()) => {
                    matched += 1;
                    println!("  match   {}", fixture.id);
                }
                Err(why) => {
                    let listed = known
                        .get(fixture.id.as_str())
                        .map_or(String::new(), |reason| {
                            format!("\n          known: {reason}")
                        });
                    println!(
                        "  differs {} (reference {}): {why}{listed}",
                        fixture.id, fixture.reference.source
                    );
                    differences.insert(fixture.id, why);
                }
            }
        }
    }
    println!(
        "{matched} cases match, {} differ, {with_witnesses} carry engine witnesses",
        differences.len()
    );
    assert!(
        !seen.is_empty(),
        "no render case under {}: every fixtures/<slug>/render directory is missing or empty",
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
            let slug = id.split('/').next().unwrap_or_default();
            loaded_slugs.contains(slug) && !seen.contains(*id)
        })
        .collect();
    assert!(
        gone.is_empty(),
        "listed in KNOWN_DIFFERENCES but no longer among the loaded fixtures; remove or rename: {}",
        gone.join(", ")
    );
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

fn read_fixtures(dir: &Path) -> Result<Vec<Fixture>, String> {
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
    let mut fixtures = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture: Fixture = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", file.display(), number + 1))?;
            fixtures.push(fixture);
        }
    }
    Ok(fixtures)
}

/// The checkpoint's tokenizer files at the manifest's revision: the Hugging
/// Face cache snapshot when it is there, else a one-time download of the files
/// the tokenizer and the renderer read: `tokenizer.json`,
/// `tokenizer_config.json`, the separate chat template files a checkpoint may
/// ship instead of a template inside the config, and `config.json` for
/// renderer detection. Only the first two must exist.
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
        fs::write(&path, &bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
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

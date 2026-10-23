//! The metric catalogue keeps its promises: every `smg_*` family the gateway
//! describes (the `# HELP` text a scrape shows) is recorded by some code path,
//! and every recorder on `Metrics` has a caller. A family described and never
//! recorded reads as a live signal on `/metrics` and never moves; a recorder
//! nobody calls is the same promise one layer down.
//!
//! Both tests read Rust sources with line comments removed. The first reads
//! the whole workspace (this crate, the library crates, the clients, the
//! bindings and the examples), so a family recorded through a wrapper type or
//! by a library crate counts as recorded, not only one recorded through the
//! `counter!`, `gauge!` and `histogram!` macros. The second reads only the
//! code that can reach this crate's `Metrics` (this crate, the bindings and
//! the examples): the library crates sit below it and have same-named
//! recorders of their own, which must not count as callers.

#![expect(clippy::expect_used, reason = "test failures should panic")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use regex::Regex;

const METRICS_SOURCE: &str = "model_gateway/src/observability/metrics.rs";
/// Where a described family may be recorded: anywhere in the workspace.
const SOURCE_ROOTS: &[&str] = &["model_gateway", "crates", "clients", "bindings", "examples"];
/// Where a caller of this crate's `Metrics` may live: the crates that depend
/// on it, not the ones it depends on.
const CALLER_ROOTS: &[&str] = &["model_gateway", "bindings", "examples"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the gateway crate lives one level under the workspace root")
        .to_path_buf()
}

/// Every `.rs` file under `dir`, build directories skipped.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if name != "target" && !name.to_string_lossy().starts_with('.') {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The source text with `//` line comments cut off.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|line| line.split("//").next().unwrap_or_default())
        .fold(String::new(), |mut acc, line| {
            acc.push_str(line);
            acc.push('\n');
            acc
        })
}

/// Every Rust source under `roots` as (path, comment-free text).
fn sources_under(roots: &[&str]) -> Vec<(PathBuf, String)> {
    let root = repo_root();
    let mut files = Vec::new();
    for sub in roots {
        rust_files(&root.join(sub), &mut files);
    }
    assert!(
        files.len() > 100,
        "the source walk found {} files under {}; the roots moved?",
        files.len(),
        root.display()
    );
    files
        .into_iter()
        .map(|path| {
            let text = fs::read_to_string(&path).unwrap_or_default();
            (path, code_only(&text))
        })
        .collect()
}

fn metrics_source(sources: &[(PathBuf, String)]) -> &str {
    let root = repo_root().join(METRICS_SOURCE);
    sources
        .iter()
        .find(|(path, _)| *path == root)
        .map(|(_, text)| text.as_str())
        .expect("the metrics module is among the sources")
}

/// Every described family (`describe_*!("smg_…", …)`) is recorded somewhere:
/// its name occurs in the sources outside its own description.
#[test]
fn every_described_family_is_recorded() {
    let sources = sources_under(SOURCE_ROOTS);
    let metrics = metrics_source(&sources);
    let describe = Regex::new(r#"describe_(?:counter|gauge|histogram)!\(\s*"(smg_[a-z0-9_]+)""#)
        .expect("a valid pattern");
    let families: Vec<&str> = describe
        .captures_iter(metrics)
        .map(|c| c.get(1).expect("the family group").as_str())
        .collect();
    assert!(
        families.len() > 50,
        "only {} described families found",
        families.len()
    );

    let mut unrecorded = BTreeMap::new();
    for family in families {
        let quoted = format!("\"{family}\"");
        let described = describe
            .captures_iter(metrics)
            .filter(|c| c.get(1).is_some_and(|m| m.as_str() == family))
            .count();
        let mentions: usize = sources
            .iter()
            .map(|(_, text)| text.matches(quoted.as_str()).count())
            .sum();
        if mentions <= described {
            unrecorded.insert(family.to_owned(), mentions);
        }
    }
    assert!(
        unrecorded.is_empty(),
        "families described and never recorded (describe them where they are recorded, or drop the description): {unrecorded:?}"
    );
}

/// Every public function of the metrics module (the recorders, the gauges'
/// setters, the exporter's setup) is referenced by name somewhere outside
/// its own definition, in code that can reach it. A wrapper that only
/// forwards to a same-named recorder of another `*Metrics` type does not
/// count as its own caller.
#[test]
fn every_recorder_has_a_caller() {
    let sources = sources_under(CALLER_ROOTS);
    let metrics = metrics_source(&sources);
    let definition = Regex::new(r"(?m)^\s*pub fn ([A-Za-z0-9_]+)\s*[<(]").expect("a valid pattern");
    let recorders: Vec<&str> = definition
        .captures_iter(metrics)
        .map(|c| c.get(1).expect("the name group").as_str())
        .collect();
    assert!(
        recorders.len() > 50,
        "only {} public functions found",
        recorders.len()
    );

    let mut uncalled = Vec::new();
    for name in recorders {
        let word = Regex::new(&format!(r"\b{}\b", regex::escape(name))).expect("a valid pattern");
        let definitions = Regex::new(&format!(r"\bfn {}\b", regex::escape(name)))
            .expect("a valid pattern")
            .find_iter(metrics)
            .count();
        let forwards = Regex::new(&format!(r"[A-Za-z0-9_]Metrics::{}\b", regex::escape(name)))
            .expect("a valid pattern")
            .find_iter(metrics)
            .count();
        let mentions: usize = sources
            .iter()
            .map(|(_, text)| word.find_iter(text).count())
            .sum();
        if mentions <= definitions + forwards {
            uncalled.push(name.to_owned());
        }
    }
    assert!(
        uncalled.is_empty(),
        "public metrics functions with no caller in the code that can reach them: {uncalled:?}"
    );
}

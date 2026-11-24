//! The README's two tables are held to the crate: the feature table to
//! `Cargo.toml`, the module map to `src/lib.rs`, each both ways. A feature or
//! a module that one side alone names fails its case, and so does a row whose
//! features differ. The page's Rust blocks are doc tests of the library: the
//! last item of `src/lib.rs` includes the page.
//!
//! The manifest is read through `cargo metadata`, so the cases read the same
//! in the checkout and in a published copy of the package.

mod common;

use std::collections::BTreeSet;

use common::{cargo_metadata, this_package, Outcome};
use serde_json::{json, Value};

const README: &str = include_str!("../README.md");
const LIBRARY: &str = include_str!("../src/lib.rs");

/// How the header row of the feature table starts, and the module map's.
const FEATURE_TABLE: &str = "| Feature |";
const MODULE_MAP: &str = "| Module |";

/// The feature table's name for what a build gets with no feature asked.
const NO_FEATURE: &str = "none";

/// What a table says, one sentence a fact, so that a difference reads as the
/// sentence one side is missing.
type Facts = BTreeSet<String>;

/// Two tables that say the same.
const NO_DIFFERENCE: [Vec<String>; 2] = [Vec::new(), Vec::new()];

/// What the page alone says, then what its source alone says.
fn differences(page: &Facts, source: &Facts) -> [Vec<String>; 2] {
    [
        page.difference(source).cloned().collect(),
        source.difference(page).cloned().collect(),
    ]
}

/// The cells of each row of the page's table whose header row starts with
/// `header`.
fn rows<'a>(page: &'a str, header: &str) -> Vec<Vec<&'a str>> {
    page.lines()
        .skip_while(|line| !line.starts_with(header))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .map(|line| {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            cells[1..cells.len() - 1].to_vec()
        })
        .collect()
}

/// The names a cell spells as code.
fn code_spans(cell: &str) -> impl Iterator<Item = &str> {
    cell.split('`').skip(1).step_by(2)
}

/// The feature table as the page has it: its first, third and fourth columns.
fn features_on_the_page(page: &str) -> Facts {
    let mut facts = Facts::new();
    for cells in rows(page, FEATURE_TABLE) {
        let cell = |column: usize| cells.get(column).copied().unwrap_or_default();
        let feature = cell(0);
        facts.insert(format!("{feature} is a row"));
        facts.extend(code_spans(cell(2)).map(|on| format!("{feature} turns on `{on}`")));
        facts.extend(code_spans(cell(3)).map(|krate| format!("{feature} pulls in `{krate}`")));
    }
    facts
}

/// The feature table as the manifest has it. A feature's `dep:` entries are
/// the crates it pulls in, and its other entries the features it turns on.
/// With no feature asked a build gets what `default` names and the crates
/// that are not optional.
fn features_in_the_manifest(package: &Value) -> Facts {
    let mut facts = Facts::from([format!("{NO_FEATURE} is a row")]);
    for (name, entries) in package["features"].as_object().into_iter().flatten() {
        let feature = if name == "default" {
            NO_FEATURE.to_owned()
        } else {
            format!("`{name}`")
        };
        facts.insert(format!("{feature} is a row"));
        let entries = entries.as_array().into_iter().flatten();
        for entry in entries.filter_map(Value::as_str) {
            facts.insert(match entry.strip_prefix("dep:") {
                Some(krate) => format!("{feature} pulls in `{krate}`"),
                None => format!("{feature} turns on `{entry}`"),
            });
        }
    }
    for dependency in package["dependencies"].as_array().into_iter().flatten() {
        if dependency["kind"].is_null() && dependency["optional"] == false {
            let krate = dependency["name"].as_str().unwrap_or_default();
            facts.insert(format!("{NO_FEATURE} pulls in `{krate}`"));
        }
    }
    facts
}

/// The module map as the page has it: its first and second columns.
fn modules_on_the_page(page: &str) -> Facts {
    let mut facts = Facts::new();
    for cells in rows(page, MODULE_MAP) {
        let cell = |column: usize| cells.get(column).copied().unwrap_or_default();
        let module = cell(0);
        facts.insert(format!("{module} is a row"));
        facts.extend(code_spans(cell(1)).map(|feature| format!("{module} needs `{feature}`")));
    }
    facts
}

/// The module map as the library has it: every `pub mod` of `lib.rs`, with
/// the features the `cfg` on the line above it names.
fn modules_in_the_library(library: &str) -> Facts {
    let mut facts = Facts::new();
    let mut above = "";
    for line in library.lines() {
        if let Some(rest) = line.strip_prefix("pub mod ") {
            let module = rest.trim_end_matches([';', '{', ' ']);
            facts.insert(format!("`{module}` is a row"));
            let features = above.split("feature = \"").skip(1);
            for feature in features.filter_map(|rest| rest.split('"').next()) {
                facts.insert(format!("`{module}` needs `{feature}`"));
            }
        }
        above = if line.starts_with("#[cfg(") { line } else { "" };
    }
    facts
}

#[test]
fn the_feature_table_says_what_the_manifest_says() -> Outcome<()> {
    let metadata = cargo_metadata()?;
    let manifest = features_in_the_manifest(this_package(&metadata)?);
    assert_eq!(
        differences(&features_on_the_page(README), &manifest),
        NO_DIFFERENCE,
        "what the README's feature table alone says, then what Cargo.toml alone says"
    );
    Ok(())
}

#[test]
fn the_module_map_says_what_the_library_declares() {
    assert_eq!(
        differences(
            &modules_on_the_page(README),
            &modules_in_the_library(LIBRARY)
        ),
        NO_DIFFERENCE,
        "what the README's module map alone says, then what src/lib.rs alone says"
    );
}

/// The comparison on a table and a manifest made in code: a feature, a
/// feature it turns on or a crate it pulls in that one side alone names is
/// reported on that side.
#[test]
fn a_feature_one_side_alone_names_is_reported_on_that_side() {
    let page = "\
| Feature | What it adds | Turns on | Pulls in | Who needs it |
|---|---|---|---|---|
| none | The types. | | `serde` | Every dependent. |
| `metrics` | The labels. | | | A writer. |
| `runtime` | The runtime. | `metrics` | `tokio` | A host. |
";
    let manifest = json!({
        "features": { "default": [], "metrics": [], "runtime": ["metrics", "dep:tokio"] },
        "dependencies": [
            { "name": "serde", "kind": null, "optional": false },
            { "name": "tokio", "kind": null, "optional": true },
            { "name": "axum", "kind": "dev", "optional": false }
        ]
    });
    let on_the_page = features_on_the_page(page);
    let in_the_manifest = features_in_the_manifest(&manifest);
    assert_eq!(differences(&on_the_page, &in_the_manifest), NO_DIFFERENCE);

    let stale = features_on_the_page(&format!(
        "{page}| `proto` | Gone. | | `prost` | Nobody. |\n"
    ));
    assert_eq!(
        differences(&stale, &in_the_manifest),
        [
            vec![
                "`proto` is a row".to_owned(),
                "`proto` pulls in `prost`".to_owned()
            ],
            vec![]
        ],
        "a row the manifest has no feature for is the page's alone"
    );

    let mut grown = manifest;
    grown["features"]["proto"] = json!([]);
    grown["features"]["runtime"] = json!(["metrics", "dep:tokio", "dep:tonic"]);
    assert_eq!(
        differences(&on_the_page, &features_in_the_manifest(&grown)),
        [
            vec![],
            vec![
                "`proto` is a row".to_owned(),
                "`runtime` pulls in `tonic`".to_owned()
            ]
        ],
        "a feature and a crate the page does not name are the manifest's alone"
    );
}

/// The comparison on a map and a library made in code: a private module and
/// a module inside another are no rows, and a module or a feature that one
/// side alone names is reported on that side.
#[test]
fn a_module_one_side_alone_names_is_reported_on_that_side() {
    let library = "\
pub mod worker;
#[cfg(feature = \"runtime\")]
pub mod registry;
#[cfg(all(feature = \"runtime\", any(test, feature = \"test-util\")))]
pub mod test_support;
#[cfg(feature = \"proto\")]
mod load_record;
#[cfg(feature = \"proto\")]
pub mod common_proto {
    pub mod inner {}
}
";
    let page = "\
| Module | Feature | What it holds |
|---|---|---|
| `worker` | | The wire types. |
| `registry` | `runtime` | The registry. |
| `test_support` | `runtime`, `test-util` | The test support. |
| `common_proto` | `proto` | The messages. |
";
    let on_the_page = modules_on_the_page(page);
    let in_the_library = modules_in_the_library(library);
    assert_eq!(differences(&on_the_page, &in_the_library), NO_DIFFERENCE);

    let stale = modules_on_the_page(&format!("{page}| `load_record` | `proto` | Private. |\n"));
    assert_eq!(
        differences(&stale, &in_the_library),
        [
            vec![
                "`load_record` is a row".to_owned(),
                "`load_record` needs `proto`".to_owned()
            ],
            vec![]
        ],
        "a row the library declares no public module for is the page's alone"
    );

    let grown = modules_in_the_library(&format!(
        "{library}#[cfg(feature = \"axum\")]\npub mod admin;\n"
    ));
    assert_eq!(
        differences(&on_the_page, &grown),
        [
            vec![],
            vec![
                "`admin` is a row".to_owned(),
                "`admin` needs `axum`".to_owned()
            ]
        ],
        "a module the page does not name is the library's alone"
    );
}

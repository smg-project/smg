//! The step-2 acceptance corpus through the parser and the validator. The
//! corpus is the directory `tests/corpus`: one Markdown item a routing
//! configuration of the gateway, as its title, the dialect it is written
//! in, its input, and under `## Expected spec` one fenced YAML document that
//! says the same as a spec. The directory holds the corpus's items that
//! state a spec document, under their numbers in the corpus, so the numbers
//! have gaps. Every expected document must parse (the schema covers every
//! field the corpus writes) and validate, or be refused only because this
//! build lacks a stage, a plan, a matcher or a binding rule; any other
//! refusal, and an item with no document, is a corpus or a validator
//! defect. An item refused for a form this build lacks is named in
//! [`NOT_IN_BUILD`], and the test fails once it validates. The per-item
//! verdicts are printed as the corpus report.

use std::{
    fs,
    path::{Path, PathBuf},
};

use smg_policy_controller::document::{PolicySpec, SpecError};

/// The fenced YAML document under the item's `## Expected spec` heading.
fn expected_document(item: &str) -> Option<&str> {
    let section = &item[item.find("## Expected spec")?..];
    let fence = section.find("```yaml")?;
    let body = &section[fence + "```yaml".len()..];
    let end = body.find("```")?;
    Some(body[..end].trim_start_matches('\n'))
}

/// The items whose document names a form this build lacks, each with the
/// validator's words for it: the first such form, where a document lacks
/// several. The test fails when a listed item validates, so an entry leaves
/// with the change that builds the last form its document lacks, and when
/// an item is refused as not in this build in words this list does not give
/// it.
const NOT_IN_BUILD: &[(&str, &str)] = &[(
    // The document also names the prefix-affinity accounting, its
    // uncached credit and the best-with-temperature picker: the change
    // that builds the decay rewrites these words, and the entry leaves
    // with the last of the four.
    "28-cli-cache-aware-hash-index-full.md",
    "the prefix-affinity decay",
)];

/// The corpus's size: an item that leaves the directory or arrives in it,
/// or takes an extension the walk does not read, is a defect, told after
/// the report of the items that stand.
const ITEMS: usize = 28;

enum Verdict {
    NoDocument,
    Accepted(String),
    NotInBuild(String, &'static str),
    Refused(String),
}

fn verdict(item: &str) -> Verdict {
    let Some(document) = expected_document(item) else {
        return Verdict::NoDocument;
    };
    let spec: PolicySpec = match serde_yaml::from_str(document) {
        Ok(spec) => spec,
        Err(err) => return Verdict::Refused(format!("parse: {err}")),
    };
    match spec.validate() {
        Ok(engine) => {
            let needs = engine.needs();
            Verdict::Accepted(format!(
                "profiles={} needs(loads={} events={} headers={})",
                engine.profiles().len(),
                needs.loads,
                needs.events,
                needs.headers
            ))
        }
        Err(SpecError::NotInBuild { path, what }) => Verdict::NotInBuild(path, what),
        Err(err) => Verdict::Refused(format!("validate: {err}")),
    }
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the per-item verdicts are the corpus report"
)]
fn the_step_2_corpus_parses_and_every_item_names_its_verdict() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let mut items: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("the corpus directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    items.sort();

    let (mut documents, mut accepted, mut not_in_build) = (0usize, 0usize, 0usize);
    let mut defects = Vec::new();
    println!("| item | verdict |");
    println!("|---|---|");
    for path in &items {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = fs::read_to_string(path).expect("an item is readable");
        let line = match verdict(&text) {
            Verdict::NoDocument => {
                defects.push(format!("{name}: no expected document"));
                "NO EXPECTED DOCUMENT".to_string()
            }
            Verdict::Accepted(summary) => {
                documents += 1;
                accepted += 1;
                if NOT_IN_BUILD.iter().any(|(item, _)| *item == name) {
                    defects.push(format!(
                        "{name}: listed as not in this build, and it validates"
                    ));
                }
                format!("ACCEPTED {summary}")
            }
            Verdict::NotInBuild(path, what) => {
                documents += 1;
                not_in_build += 1;
                if !NOT_IN_BUILD.contains(&(name.as_str(), what)) {
                    defects.push(format!("{name}: not listed as not in this build: {what}"));
                }
                format!("WELL-FORMED, not in this build: {what} ({path})")
            }
            Verdict::Refused(why) => {
                documents += 1;
                defects.push(format!("{name}: {why}"));
                format!("REFUSED {why}")
            }
        };
        println!("| {name} | {line} |");
    }
    for (item, _) in NOT_IN_BUILD {
        let listed = |path: &PathBuf| path.file_name().is_some_and(|name| name == *item);
        if !items.iter().any(listed) {
            defects.push(format!(
                "{item}: listed as not in this build, and no such item"
            ));
        }
    }
    if items.len() != ITEMS {
        defects.push(format!(
            "{} items under {}, and the corpus holds {ITEMS}",
            items.len(),
            dir.display()
        ));
    }
    println!();
    println!(
        "items {}, with a document {documents}, accepted {accepted}, well-formed but not in this build {not_in_build}, defects {}",
        items.len(),
        defects.len()
    );
    assert!(defects.is_empty(), "corpus defects: {defects:#?}");
}

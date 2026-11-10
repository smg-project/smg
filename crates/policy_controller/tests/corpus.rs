//! The step-2 acceptance corpus through the parser and the validator. The
//! corpus lives outside this repository; `SMG_POLICY_SPEC_CORPUS` names a
//! directory of its items (Markdown, each with an `## Expected spec` section
//! holding one fenced YAML document), and the test is skipped when it is
//! unset. Every expected document must parse (the schema covers every field
//! the corpus writes) and validate, or be refused only because this build
//! lacks a stage, a plan, a matcher or a binding rule; any other refusal is
//! a corpus or a validator defect. The per-item verdicts are printed as the
//! corpus report.

use std::{fs, path::PathBuf};

use smg_policy_controller::{PolicySpec, SpecError};

/// The fenced YAML document under the item's `## Expected spec` heading.
fn expected_document(item: &str) -> Option<&str> {
    let section = &item[item.find("## Expected spec")?..];
    let fence = section.find("```yaml")?;
    let body = &section[fence + "```yaml".len()..];
    let end = body.find("```")?;
    Some(body[..end].trim_start_matches('\n'))
}

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

/// Ignored by default because the corpus lives in the test-data repository,
/// not here: `cargo test --test corpus -- --ignored` with
/// `SMG_POLICY_SPEC_CORPUS` naming its directory runs it, and it fails rather
/// than passes when the variable is unset.
#[test]
#[ignore = "needs SMG_POLICY_SPEC_CORPUS, the corpus directory of the test-data repository"]
#[expect(
    clippy::print_stdout,
    reason = "the per-item verdicts are the corpus report"
)]
fn the_step_2_corpus_parses_and_every_item_names_its_verdict() {
    let dir = std::env::var_os("SMG_POLICY_SPEC_CORPUS")
        .expect("SMG_POLICY_SPEC_CORPUS names the corpus directory");
    let mut items: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("the corpus directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    items.sort();
    assert!(
        !items.is_empty(),
        "no items under {}",
        dir.to_string_lossy()
    );

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
                "no expected document (a rule, report or values item)".to_string()
            }
            Verdict::Accepted(summary) => {
                documents += 1;
                accepted += 1;
                format!("ACCEPTED {summary}")
            }
            Verdict::NotInBuild(path, what) => {
                documents += 1;
                not_in_build += 1;
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
    println!();
    println!(
        "items {}, with a document {documents}, accepted {accepted}, well-formed but not in this build {not_in_build}, refused {}",
        items.len(),
        defects.len()
    );
    assert!(defects.is_empty(), "corpus items refused: {defects:#?}");
}

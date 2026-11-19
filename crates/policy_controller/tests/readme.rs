//! The README, checked against the crate, so the page says what this build
//! holds. Four things are read off `README.md`:
//!
//! - every fenced `yaml` block is a whole policy document: it parses, and
//!   the stock catalog validates it (a fragment is fenced as text);
//! - a `yaml` block under the mark `<!-- refused -->` is a document the
//!   validator refuses, and the fenced text after it ends with the
//!   validator's words;
//! - the three stage tables, each between its `begin` and `end` marks, name
//!   the catalog's filters, scorers and pickers: every name, no other, in
//!   the catalog's order;
//! - a `yaml` block under `<!-- preset: <name> -->` is the crate's preset of
//!   that name but for its `generation` line, which the page leaves out, and
//!   the page shows every preset.

use std::{
    fs,
    path::{Path, PathBuf},
};

use smg_policy_controller::{Catalog, PolicySpec, PRESETS};

/// The mark on the line before a document the validator refuses.
const REFUSED: &str = "<!-- refused -->";

/// The mark on the line before a preset's document, up to the preset's name.
const PRESET: &str = "<!-- preset: ";

/// The documents the page shows running: a walk that reads fewer has lost a
/// fence, and a document the page gains raises the count.
const DOCUMENTS: usize = 15;

/// The documents the page shows refused.
const REFUSALS: usize = 2;

fn readme() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md")
}

/// One fenced block of the page.
struct Block<'a> {
    /// The fence's language: `yaml`, `text`, `rust` or `sh`.
    language: &'a str,
    /// The last line with text before the fence: a mark, or prose.
    before: &'a str,
    body: String,
}

/// The page's fenced blocks, in its order.
fn blocks(page: &str) -> Vec<Block<'_>> {
    let mut blocks = Vec::new();
    let mut before = "";
    let mut open: Option<Block<'_>> = None;
    for line in page.lines() {
        match open.take() {
            Some(block) if line == "```" => blocks.push(block),
            Some(mut block) => {
                block.body.push_str(line);
                block.body.push('\n');
                open = Some(block);
            }
            None => match line.strip_prefix("```") {
                Some(language) => {
                    open = Some(Block {
                        language,
                        before,
                        body: String::new(),
                    });
                }
                None if line.trim().is_empty() => {}
                None => before = line,
            },
        }
    }
    blocks
}

/// The words a document is refused in: the validator's, or the parser's
/// where the text is no document. `None` for a document that runs.
fn refusal(document: &str) -> Option<String> {
    match serde_yaml::from_str::<PolicySpec>(document) {
        Ok(spec) => spec.validate().err().map(|error| error.to_string()),
        Err(error) => Some(error.to_string()),
    }
}

/// The names a stage table of the page lists, in the table's order: the
/// first cell of each row between the kind's marks, where it is a code span.
fn listed<'a>(page: &'a str, kind: &str) -> Vec<&'a str> {
    let begin = format!("<!-- {kind}: begin -->");
    let end = format!("<!-- {kind}: end -->");
    page.lines()
        .skip_while(|line| *line != begin)
        .take_while(|line| *line != end)
        .filter_map(|row| row.strip_prefix("| `")?.split('`').next())
        .collect()
}

#[test]
fn every_document_of_the_page_runs_over_the_stock_catalog() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let blocks = blocks(&page);
    let documents: Vec<&Block<'_>> = blocks
        .iter()
        .filter(|block| block.language == "yaml" && block.before != REFUSED)
        .collect();
    let refused: Vec<String> = documents
        .iter()
        .filter_map(|block| refusal(&block.body).map(|words| format!("{words}\n{}", block.body)))
        .collect();
    assert!(
        refused.is_empty(),
        "the README shows as running documents the validator refuses:\n{}",
        refused.join("\n")
    );
    assert_eq!(
        documents.len(),
        DOCUMENTS,
        "the README shows {DOCUMENTS} documents that run: a fence was lost, or a new document raises the count"
    );
}

#[test]
fn a_refused_document_is_refused_in_the_words_the_page_shows() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let blocks = blocks(&page);
    let mut shown = 0;
    for (at, block) in blocks.iter().enumerate() {
        if block.language != "yaml" || block.before != REFUSED {
            continue;
        }
        shown += 1;
        let words = refusal(&block.body);
        assert!(
            words.is_some(),
            "the README shows as refused a document that runs:\n{}",
            block.body
        );
        let text = blocks
            .get(at + 1)
            .filter(|next| next.language == "text")
            .map(|next| next.body.trim_end());
        assert!(
            text.zip(words.as_deref())
                .is_some_and(|(text, words)| text.ends_with(words)),
            "the text after a refused document ends with the validator's words {words:?}, and the README shows {text:?}"
        );
    }
    assert_eq!(
        shown, REFUSALS,
        "the README shows {REFUSALS} refused documents: a mark was lost, or a new one raises the count"
    );
}

#[test]
fn the_stage_tables_name_the_catalogs_stages_and_no_other() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let catalog = Catalog::builtin();
    let kinds: [(&str, Vec<&str>); 3] = [
        ("filters", catalog.filters().collect()),
        ("scorers", catalog.scorers().collect()),
        ("pickers", catalog.pickers().collect()),
    ];
    for (kind, held) in kinds {
        let listed = listed(&page, kind);
        let missing: Vec<&str> = held
            .iter()
            .copied()
            .filter(|name| !listed.contains(name))
            .collect();
        let unknown: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|name| !held.contains(name))
            .collect();
        assert!(
            missing.is_empty() && unknown.is_empty(),
            "the README's {kind} table: the catalog holds {missing:?} and the table does not; \
             the table lists {unknown:?} and the catalog does not"
        );
        assert_eq!(
            listed, held,
            "the README's {kind} table lists the catalog's {kind} once each, in the catalog's order"
        );
    }
}

#[test]
fn each_preset_document_is_the_crates_preset() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let blocks = blocks(&page);
    let shown: Vec<(&str, &Block<'_>)> = blocks
        .iter()
        .filter(|block| block.language == "yaml")
        .filter_map(|block| {
            let name = block.before.strip_prefix(PRESET)?.strip_suffix(" -->")?;
            Some((name, block))
        })
        .collect();
    let names: Vec<&str> = shown.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names, PRESETS,
        "the README shows every preset of the crate once, in the crate's order"
    );
    for (name, block) in shown {
        let preset = PolicySpec::preset(name).expect("a preset of the crate");
        let mut document: PolicySpec =
            serde_yaml::from_str(&block.body).expect("a preset's document parses");
        // A file that writes the line raises it at every change, so the page
        // leaves it out: the documents are the presets but for that line.
        assert_eq!(
            document.generation, 0,
            "the README's `{name}` document writes no generation line"
        );
        document.generation = preset.generation;
        assert_eq!(
            document, preset,
            "the README's `{name}` document is the crate's preset"
        );
    }
}

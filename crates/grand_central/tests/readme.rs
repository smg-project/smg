//! The crate's two pages, checked against the crate, so each says what this
//! build holds. `README.md` is the first page, short enough to read whole.
//! `REFERENCE.md` holds the detail. Six things are read:
//!
//! - every fenced `yaml` block of either page is a whole policy document: it
//!   parses, and the stock catalog validates it (a fragment is fenced as
//!   text);
//! - a `yaml` block of the reference under the mark `<!-- refused -->` is a
//!   document the validator refuses, and the fenced text after it ends with
//!   the validator's words;
//! - the three stage tables of either page, each between its `begin` and
//!   `end` marks, name the catalog's filters, scorers and pickers: every
//!   name, no other, in the catalog's order;
//! - a `yaml` block of the reference under `<!-- preset: <name> -->` is the
//!   crate's preset of that name but for its `generation` line, which the
//!   page leaves out, and the reference shows every preset. The README's
//!   preset table names every preset too, in the crate's order;
//! - the reference names, each in a code span, every flag of the standalone
//!   binary, every call of the controller's contract and every series the
//!   controller exports: each list is read off the files that declare it.
//!   Each is read the other way too: a row of the reference's flag tables,
//!   of its call table or of its metrics table names none that is gone;
//! - the README's prose stays inside the bounds a person reads with ease:
//!   no sentence over 30 words, no paragraph over five sentences or 100
//!   words, no list item over 50 words, and none of the filler words. Each
//!   fenced block of the README is closed: behind an open one the walk
//!   would read the page's prose as code.

use std::{
    fs,
    path::{Path, PathBuf},
};

use smg_grand_central::{
    document::{PolicySpec, PRESETS},
    stages::StageRegistry,
};

/// The mark on the line before a document the validator refuses.
const REFUSED: &str = "<!-- refused -->";

/// The mark on the line before a preset's document, up to the preset's name.
const PRESET: &str = "<!-- preset: ";

/// The documents the README shows running: a walk that reads fewer has lost
/// a fence, and a document the page gains raises the count.
const README_DOCUMENTS: usize = 2;

/// The documents the reference shows running, counted as the README's are.
const DOCUMENTS: usize = 16;

/// The documents the reference shows refused.
const REFUSALS: usize = 2;

/// The flags the standalone binary declares, at the least: a walk that
/// finds fewer has lost the shape of a source file, and a change that
/// takes a flag away lowers the count.
const FLAGS: usize = 55;

/// The flags the argument parser adds by itself: a row of the page names
/// them, and no line of the command line's declaration does.
const PARSERS_OWN: [&str; 2] = ["--help", "--version"];

/// The calls the controller's contract holds, at the least.
const CALLS: usize = 15;

/// The series the standalone controller exports, at the least.
const SERIES: usize = 22;

/// The most words a sentence of the README holds. Nothing but a sentence's
/// end splits one, so a chain joined by semicolons is one sentence.
const SENTENCE_WORDS: usize = 30;

/// The most sentences a paragraph of the README holds.
const PARAGRAPH_SENTENCES: usize = 5;

/// The most words a paragraph of the README holds.
const PARAGRAPH_WORDS: usize = 100;

/// The most words a list item of the README holds.
const ITEM_WORDS: usize = 50;

/// Words that fill a sentence and say nothing: the README holds none.
const FILLER: [&str; 13] = [
    "robust",
    "seamless",
    "seamlessly",
    "leverage",
    "leverages",
    "leveraging",
    "comprehensive",
    "delve",
    "utilize",
    "utilizes",
    "utilizing",
    "cutting-edge",
    "simply",
];

/// The first page: what the crate is, in a text a person reads whole.
fn readme() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md")
}

/// The page of detail behind the first one.
fn reference() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("REFERENCE.md")
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
    for (name, path, shown) in [
        ("README", readme(), README_DOCUMENTS),
        ("reference", reference(), DOCUMENTS),
    ] {
        let page = fs::read_to_string(path).expect("the page is readable");
        let blocks = blocks(&page);
        let documents: Vec<&Block<'_>> = blocks
            .iter()
            .filter(|block| block.language == "yaml" && block.before != REFUSED)
            .collect();
        let refused: Vec<String> = documents
            .iter()
            .filter_map(|block| {
                refusal(&block.body).map(|words| format!("{words}\n{}", block.body))
            })
            .collect();
        assert!(
            refused.is_empty(),
            "the {name} shows as running documents the validator refuses:\n{}",
            refused.join("\n")
        );
        assert_eq!(
            documents.len(),
            shown,
            "the {name} shows {shown} documents that run: a fence was lost, or a new document raises the count"
        );
    }
}

#[test]
fn a_refused_document_is_refused_in_the_words_the_page_shows() {
    let page = fs::read_to_string(reference()).expect("the reference is readable");
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
            "the reference shows as refused a document that runs:\n{}",
            block.body
        );
        let text = blocks
            .get(at + 1)
            .filter(|next| next.language == "text")
            .map(|next| next.body.trim_end());
        assert!(
            text.zip(words.as_deref())
                .is_some_and(|(text, words)| text.ends_with(words)),
            "the text after a refused document ends with the validator's words {words:?}, and the reference shows {text:?}"
        );
    }
    assert_eq!(
        shown, REFUSALS,
        "the reference shows {REFUSALS} refused documents: a mark was lost, or a new one raises the count"
    );
}

#[test]
fn the_stage_tables_name_the_catalogs_stages_and_no_other() {
    let catalog = StageRegistry::builtin();
    let kinds: [(&str, Vec<&str>); 3] = [
        ("filters", catalog.filters().collect()),
        ("scorers", catalog.scorers().collect()),
        ("pickers", catalog.pickers().collect()),
    ];
    // The README's rows are short and the reference's hold each stage's
    // rules: both list the same stages.
    for (name, path) in [("README", readme()), ("reference", reference())] {
        let page = fs::read_to_string(path).expect("the page is readable");
        for (kind, held) in &kinds {
            let listed = listed(&page, kind);
            let missing: Vec<&str> = held
                .iter()
                .copied()
                .filter(|stage| !listed.contains(stage))
                .collect();
            let unknown: Vec<&str> = listed
                .iter()
                .copied()
                .filter(|stage| !held.contains(stage))
                .collect();
            assert!(
                missing.is_empty() && unknown.is_empty(),
                "the {name}'s {kind} table: the catalog holds {missing:?} and the table does not; \
                 the table lists {unknown:?} and the catalog does not"
            );
            assert_eq!(
                &listed, held,
                "the {name}'s {kind} table lists the catalog's {kind} once each, in the catalog's order"
            );
        }
    }
}

#[test]
fn each_preset_document_is_the_crates_preset() {
    let page = fs::read_to_string(reference()).expect("the reference is readable");
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
        "the reference shows every preset of the crate once, in the crate's order"
    );
    for (name, block) in shown {
        let preset = PolicySpec::preset(name).expect("a preset of the crate");
        let mut document: PolicySpec =
            serde_yaml::from_str(&block.body).expect("a preset's document parses");
        // A file that writes the line raises it at every change, so the page
        // leaves it out: the documents are the presets but for that line.
        assert_eq!(
            document.generation, 0,
            "the reference's `{name}` document writes no generation line"
        );
        document.generation = preset.generation;
        assert_eq!(
            document, preset,
            "the reference's `{name}` document is the crate's preset"
        );
    }
    // The README shows no preset's document: a table names each in a row.
    let first = fs::read_to_string(readme()).expect("the README is readable");
    assert_eq!(
        table(&first, "| Preset |", 1),
        PRESETS,
        "the README's preset table names every preset of the crate once, in the crate's order"
    );
}

/// The field a line of the command line's declaration names, where it
/// names one: `load_monitor_interval: u64,` names `load_monitor_interval`.
fn field(line: &str) -> Option<&str> {
    let (name, _) = line.split_once(": ")?;
    (!name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')).then_some(name)
}

/// The standalone binary's flags, off the declaration of its command line
/// and of the discovery build's flags: the field under an `arg` attribute
/// is one flag, spelled with dashes.
fn flags() -> Vec<String> {
    [
        include_str!("../src/service/controller/run.rs"),
        include_str!("../src/service/controller/discovery/flags.rs"),
    ]
    .into_iter()
    .flat_map(|declaration| {
        let mut under_arg = false;
        declaration.lines().map(str::trim).filter_map(move |line| {
            if line.starts_with("#[arg(") {
                under_arg = true;
                return None;
            }
            let name = field(line).filter(|_| under_arg)?;
            under_arg = false;
            Some(format!("--{}", name.replace('_', "-")))
        })
    })
    .collect()
}

/// The calls of the controller's contract: each `rpc` of its service.
fn calls() -> Vec<&'static str> {
    include_str!("../contracts/grand_central.proto")
        .lines()
        .filter_map(|line| line.trim().strip_prefix("rpc ")?.split('(').next())
        .collect()
}

/// The series the standalone controller exports: each name its metrics
/// module declares.
fn series() -> Vec<&'static str> {
    include_str!("../src/service/controller/metrics.rs")
        .split("&str =")
        .skip(1)
        .filter_map(|rest| rest.trim_start().strip_prefix('"')?.split('"').next())
        .collect()
}

/// The declared names the page holds in no code span.
fn left_out<'a>(page: &str, declared: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    declared
        .into_iter()
        .filter(|name| !page.contains(&format!("`{name}`")))
        .collect()
}

/// The code spans of one cell of a table's row, the first cell being 1.
fn spans(row: &str, cell: usize) -> impl Iterator<Item = &str> {
    row.split('|')
        .nth(cell)
        .into_iter()
        .flat_map(|cell| cell.split('`').skip(1).step_by(2))
}

/// The names the page's table under the heading row `heading` holds: the
/// code spans of one cell of each of its rows.
fn table<'a>(page: &'a str, heading: &str, cell: usize) -> Vec<&'a str> {
    page.lines()
        .skip_while(|line| !line.starts_with(heading))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .flat_map(|row| spans(row, cell))
        .collect()
}

#[test]
fn the_page_names_every_flag_of_the_standalone_binary() {
    let page = fs::read_to_string(reference()).expect("the reference is readable");
    let flags = flags();
    assert!(
        flags.len() >= FLAGS,
        "the command line declares {} flags by this walk",
        flags.len()
    );
    let left_out = left_out(&page, flags.iter().map(String::as_str));
    assert!(
        left_out.is_empty(),
        "the reference's flag table leaves out {left_out:?}"
    );
    // The other way: a row of the flag tables names no flag that is gone.
    let named: Vec<&str> = page
        .lines()
        .filter(|line| line.starts_with("| `--"))
        .flat_map(|row| spans(row, 1))
        .collect();
    assert!(
        named.len() >= FLAGS,
        "the reference's flag rows name {} flags by this walk",
        named.len()
    );
    let gone: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !PARSERS_OWN.contains(name) && !flags.iter().any(|flag| flag == name))
        .collect();
    assert!(
        gone.is_empty(),
        "the reference's flag table names {gone:?}, which the command line does not declare"
    );
}

#[test]
fn the_page_names_every_call_of_the_contract() {
    let page = fs::read_to_string(reference()).expect("the reference is readable");
    let calls = calls();
    assert!(
        calls.len() >= CALLS,
        "the contract holds {} calls by this walk",
        calls.len()
    );
    let left_out = left_out(&page, calls.iter().copied());
    assert!(
        left_out.is_empty(),
        "the reference's call table leaves out {left_out:?}"
    );
    // The other way: the table of calls names no call that is gone.
    let named = table(&page, "| Scope | Call |", 2);
    assert!(
        named.len() >= CALLS,
        "the reference's call table names {} calls by this walk",
        named.len()
    );
    let gone: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !calls.contains(name))
        .collect();
    assert!(
        gone.is_empty(),
        "the reference's call table names {gone:?}, which the contract does not hold"
    );
}

#[test]
fn the_page_names_every_series_the_controller_exports() {
    let page = fs::read_to_string(reference()).expect("the reference is readable");
    let series = series();
    assert!(
        series.len() >= SERIES,
        "the metrics module declares {} series by this walk",
        series.len()
    );
    let left_out = left_out(&page, series.iter().copied());
    assert!(
        left_out.is_empty(),
        "the reference's metrics table leaves out {left_out:?}"
    );
    // The other way: the table of series names none that is gone.
    let named = table(&page, "| Metric | What it measures |", 1);
    assert!(
        named.len() >= SERIES,
        "the reference's metrics table names {} series by this walk",
        named.len()
    );
    let gone: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !series.contains(name))
        .collect();
    assert!(
        gone.is_empty(),
        "the reference's metrics table names {gone:?}, which the metrics module does not declare"
    );
}

/// One paragraph or one list item of a page's prose.
struct Prose {
    /// The line of the page it starts on, the first line being 1.
    line: usize,
    /// Whether it is a list item. A paragraph otherwise.
    item: bool,
    text: String,
}

/// The text of a list item where `line` starts one: what stands behind a
/// dash, a star or a number and its full stop.
fn item(line: &str) -> Option<&str> {
    let numbered = line.trim_start_matches(|c: char| c.is_ascii_digit());
    if numbered.len() < line.len() {
        return numbered.strip_prefix(". ");
    }
    line.strip_prefix("- ").or_else(|| line.strip_prefix("* "))
}

/// The page's prose: its paragraphs and its list items, in its order. A
/// fenced block, a table's row, a heading and a mark are no prose. A list
/// item is read by itself, with the lines that continue it.
fn prose(page: &str) -> Vec<Prose> {
    let mut read: Vec<Prose> = Vec::new();
    let mut open: Option<Prose> = None;
    let mut fenced = false;
    for (at, line) in page.lines().enumerate() {
        let line = line.trim();
        if line.starts_with("```") {
            fenced = !fenced;
        }
        let no_prose = fenced
            || line.is_empty()
            || line.starts_with("```")
            || line.starts_with('#')
            || line.starts_with('|')
            || line.starts_with("<!--");
        if no_prose {
            read.extend(open.take());
        } else if let Some(text) = item(line) {
            read.extend(open.replace(Prose {
                line: at + 1,
                item: true,
                text: text.to_owned(),
            }));
        } else if let Some(prose) = &mut open {
            prose.text.push(' ');
            prose.text.push_str(line);
        } else {
            open = Some(Prose {
                line: at + 1,
                item: false,
                text: line.to_owned(),
            });
        }
    }
    read.extend(open);
    read
}

/// A text as its reader counts it: a code span is one word, a link reads
/// as its text alone, and the marks of emphasis are dropped.
fn plain(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut rest = text.chars().peekable();
    while let Some(c) = rest.next() {
        match c {
            '`' => {
                rest.by_ref().find(|c| *c == '`');
                plain.push_str(" CODE ");
            }
            ']' if rest.peek() == Some(&'(') => {
                rest.by_ref().find(|c| *c == ')');
            }
            '[' | ']' | '*' | '_' => {}
            c => plain.push(c),
        }
    }
    plain
}

/// Whether `word` counts as a word: it holds a letter or a digit.
fn counts(word: &str) -> bool {
    word.contains(|c: char| c.is_ascii_alphanumeric())
}

/// The words of each sentence of a plain text. A sentence ends at a full
/// stop, a question mark or an exclamation mark, where a capital or a digit
/// starts the next word. Nothing else ends one. A lone word is a label and
/// no sentence: the bold word that opens a list item is one.
fn sentences(plain: &str) -> Vec<usize> {
    let mut sentences = Vec::new();
    let (mut tokens, mut words) = (0, 0);
    let mut rest = plain.split_whitespace().peekable();
    while let Some(token) = rest.next() {
        tokens += 1;
        words += usize::from(counts(token));
        let ends = token
            .trim_end_matches(['"', '\'', ')'])
            .ends_with(['.', '!', '?']);
        let begins = rest.peek().is_none_or(|next| {
            next.trim_start_matches(['"', '\'', '('])
                .starts_with(|c: char| c.is_ascii_uppercase() || c.is_ascii_digit())
        });
        if rest.peek().is_none() || (ends && begins) {
            if tokens > 1 {
                sentences.push(words);
            }
            (tokens, words) = (0, 0);
        }
    }
    sentences
}

/// The filler words a plain text holds, each as a whole word.
fn fillers(plain: &str) -> Vec<&'static str> {
    let lowered = plain.to_ascii_lowercase();
    let held: Vec<&str> = lowered
        .split(|c: char| !c.is_ascii_alphabetic() && c != '-')
        .collect();
    FILLER
        .into_iter()
        .filter(|filler| held.contains(filler))
        .collect()
}

#[test]
fn the_readmes_prose_stays_inside_the_bounds_a_person_reads_with_ease() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    // A fence line opens a block and the next one closes it. Behind a block
    // left open the walk below reads nothing, to the page's end.
    let open = page
        .lines()
        .filter(|line| line.trim().starts_with("```"))
        .fold(false, |open, _| !open);
    assert!(
        !open,
        "the README leaves a fenced block open: the prose behind it is read as code, by no bound"
    );
    let mut over: Vec<String> = Vec::new();
    for Prose { line, item, text } in prose(&page) {
        let plain = plain(&text);
        let sentences = sentences(&plain);
        let words = plain.split_whitespace().filter(|word| counts(word)).count();
        for long in sentences.iter().filter(|words| **words > SENTENCE_WORDS) {
            over.push(format!(
                "line {line}: a sentence of {long} words (at most {SENTENCE_WORDS})"
            ));
        }
        if item && words > ITEM_WORDS {
            over.push(format!(
                "line {line}: a list item of {words} words (at most {ITEM_WORDS})"
            ));
        }
        if !item && (words > PARAGRAPH_WORDS || sentences.len() > PARAGRAPH_SENTENCES) {
            over.push(format!(
                "line {line}: a paragraph of {words} words in {} sentences (at most {PARAGRAPH_WORDS} and {PARAGRAPH_SENTENCES})",
                sentences.len()
            ));
        }
        for filler in fillers(&plain) {
            over.push(format!("line {line}: the filler word `{filler}`"));
        }
    }
    assert!(
        over.is_empty(),
        "the README is the page a person reads first, and its prose is over its bounds:\n{}",
        over.join("\n")
    );
}

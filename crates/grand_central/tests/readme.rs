//! The crate's pages, checked against the crate, so each says what this
//! build holds. `README.md` is the first page, short enough to read whole.
//! `docs/policy-spec.md` is the policy spec's page, cut out of the reference
//! and held to the README's bounds. `docs/stages.md` is the stages' page and
//! `docs/service.md` the service's page, each cut out and held the same way.
//! `REFERENCE.md` holds the rest of the detail. Nine things are read:
//!
//! - every fenced `yaml` block of a page is a whole policy document: it
//!   parses, and the stock catalog validates it (a fragment is fenced as
//!   text);
//! - a `yaml` block of the reference under the mark `<!-- refused -->` is a
//!   document the validator refuses, and the fenced text after it ends with
//!   the validator's words;
//! - the three stage tables of the README and of the stages' page, each
//!   between its `begin` and `end` marks, name the catalog's filters,
//!   scorers and pickers: every name, no other, in the catalog's order;
//! - a `yaml` block of the reference under `<!-- preset: <name> -->` is the
//!   crate's preset of that name but for its `generation` line, which the
//!   page leaves out, and the reference shows every preset. The README's
//!   preset table names every preset too, in the crate's order;
//! - the service's page names, each in a row of its tables, every flag of
//!   the standalone binary and every call of the controller's contract, and
//!   the reference names, each in a code span, every series the controller
//!   exports: each list is read off the files that declare it. Each is read
//!   the other way too: a row of the service page's flag tables or of its
//!   call table, or of the reference's metrics table, names none that is
//!   gone;
//! - the README and the three pages cut out stay inside the bounds a person
//!   reads with ease: no sentence over 30 words, no paragraph over five
//!   sentences or 100 words, no list item over 50 words, no table cell over
//!   30 words, no heading over 30 words, and none of the filler words in a
//!   paragraph, an item, a cell or a heading. Each fenced block names its
//!   language and is closed by a
//!   bare fence: behind an open one the walk would read the page's prose
//!   as code;
//! - the walk that holds those bounds refuses a short text over each of
//!   them and passes one that stands at a bound, and the reading of a
//!   page's blocks, headings and links is held to short texts the same
//!   way;
//! - every link of the five pages to a path of the checkout resolves, and
//!   a link that names a heading names one its page holds;
//! - each command the README shows names what the checkout holds: this
//!   package, features and an example of its manifest, a unit test of the
//!   library, and flags the standalone binary's command line
//!   declares.

use std::{
    fs, mem,
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
const DOCUMENTS: usize = 12;

/// The documents the policy spec's page shows running, counted the same way.
const PAGE_DOCUMENTS: usize = 5;

/// The documents the stages' page shows running, counted the same way.
const STAGES_DOCUMENTS: usize = 1;

/// The documents the service's page shows running: none. Its fences are
/// commands and Rust, and one named `yaml` there would be read as a document.
const SERVICE_DOCUMENTS: usize = 0;

/// The commands the README shows, counted as its documents are.
const README_COMMANDS: usize = 4;

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

/// The most words a sentence of a short page holds: the README and the three
/// pages cut out of the reference are the short pages. Nothing but a
/// sentence's end splits one, so a chain joined by semicolons is one
/// sentence.
const SENTENCE_WORDS: usize = 30;

/// The most sentences a paragraph of a short page holds.
const PARAGRAPH_SENTENCES: usize = 5;

/// The most words a paragraph of a short page holds.
const PARAGRAPH_WORDS: usize = 100;

/// The most words a list item of a short page holds.
const ITEM_WORDS: usize = 50;

/// The most words a table cell of a short page holds.
const CELL_WORDS: usize = 30;

/// Words that fill a sentence and say nothing: a short page holds none.
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

/// The policy spec's page: its fields, its bindings and its plans.
fn policy_spec() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/policy-spec.md")
}

/// The stages' page: each stock filter, scorer and picker with its rules.
fn stages() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/stages.md")
}

/// The service's page: the standalone binary's flags, its worker discovery
/// and the calls it serves.
fn service() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/service.md")
}

/// One fenced block of the page.
struct Block<'a> {
    /// The fence's language: `yaml`, `text`, `rust` or `sh`.
    language: &'a str,
    /// The last line with text before the fence: a mark, or prose.
    before: &'a str,
    body: String,
}

/// How many spaces and tabs `line` begins with.
fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

/// The page's fenced blocks, in its order. A fence line is read as every
/// walk of a page reads one ([`fence`]), so a block that stands indented,
/// as under a list item, is a block too: its lines lose the indentation of
/// the fence that opens it.
fn blocks(page: &str) -> Vec<Block<'_>> {
    let mut blocks = Vec::new();
    let mut before = "";
    // The block that stands open, and where its opening fence begins.
    let mut open: Option<(Block<'_>, usize)> = None;
    for line in page.lines() {
        match (open.take(), fence(line)) {
            (Some((block, _)), Some("")) => blocks.push(block),
            // A fence that names a language inside a block is the block's
            // text here. On a short page the walk of the fences refuses it.
            // The reference has no such walk: a lost fence shows there
            // where a document stops running or a count of documents moves.
            (Some((mut block, margin)), _) => {
                block.body.push_str(&line[indent(line).min(margin)..]);
                block.body.push('\n');
                open = Some((block, margin));
            }
            (None, Some(language)) => {
                let block = Block {
                    language,
                    before,
                    body: String::new(),
                };
                open = Some((block, indent(line)));
            }
            (None, None) if line.trim().is_empty() => {}
            (None, None) => before = line.trim(),
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
        ("policy spec's page", policy_spec(), PAGE_DOCUMENTS),
        ("stages' page", stages(), STAGES_DOCUMENTS),
        ("service's page", service(), SERVICE_DOCUMENTS),
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
    // The reading itself, on a short text: a block under a list item is a
    // block, with the mark above it, and its lines lose the fence's
    // indentation. A fence that stands indented closes a block too.
    let page = "- An item:\n\n  <!-- refused -->\n  ```yaml\n  profiles:\n    default: {pick: random}\n  ```\n\nA command:\n```sh\ncargo test\n  ```\nBehind them.\n";
    let blocks = blocks(page);
    let read: Vec<(&str, &str, &str)> = blocks
        .iter()
        .map(|block| (block.language, block.before, block.body.as_str()))
        .collect();
    assert_eq!(
        read,
        [
            ("yaml", REFUSED, "profiles:\n  default: {pick: random}\n"),
            ("sh", "A command:", "cargo test\n"),
        ],
        "an indented block"
    );
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
    // The README's rows are short and the stages' page holds each stage's
    // rules: both list the same stages.
    for (name, path) in [("README", readme()), ("stages' page", stages())] {
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

/// The rows of a page's flag tables: each line outside a fenced block that
/// begins a row with a flag's name.
fn flag_rows(page: &str) -> Vec<&str> {
    let mut fenced = false;
    page.lines()
        .filter(|line| {
            if fence(line).is_some() {
                fenced = !fenced;
                return false;
            }
            !fenced && line.starts_with("| `--")
        })
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
    let page = fs::read_to_string(service()).expect("the service's page is readable");
    let flags = flags();
    assert!(
        flags.len() >= FLAGS,
        "the command line declares {} flags by this walk",
        flags.len()
    );
    // A flag's row defines it: its name in a sentence alone defines none,
    // and neither does a row inside a fenced block. The two flags the
    // parser adds have their row too.
    let named: Vec<&str> = flag_rows(&page)
        .into_iter()
        .flat_map(|row| spans(row, 1))
        .collect();
    let left_out: Vec<&str> = flags
        .iter()
        .map(String::as_str)
        .chain(PARSERS_OWN)
        .filter(|flag| !named.contains(flag))
        .collect();
    assert!(
        left_out.is_empty(),
        "the service page's flag tables leave out {left_out:?}"
    );
    // The other way: a row of the flag tables names no flag that is gone.
    assert!(
        named.len() >= FLAGS,
        "the service page's flag rows name {} flags by this walk",
        named.len()
    );
    let gone: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !PARSERS_OWN.contains(name) && !flags.iter().any(|flag| flag == name))
        .collect();
    assert!(
        gone.is_empty(),
        "the service page's flag tables name {gone:?}, which the command line does not declare"
    );
}

#[test]
fn the_page_names_every_call_of_the_contract() {
    let page = fs::read_to_string(service()).expect("the service's page is readable");
    let calls = calls();
    assert!(
        calls.len() >= CALLS,
        "the contract holds {} calls by this walk",
        calls.len()
    );
    // A call's row defines it: its name in a sentence alone defines none.
    let named = table(&page, "| Scope | Call |", 2);
    let left_out: Vec<&str> = calls
        .iter()
        .copied()
        .filter(|call| !named.contains(call))
        .collect();
    assert!(
        left_out.is_empty(),
        "the service page's call table leaves out {left_out:?}"
    );
    // The other way: the table of calls names no call that is gone.
    assert!(
        named.len() >= CALLS,
        "the service page's call table names {} calls by this walk",
        named.len()
    );
    let gone: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !calls.contains(name))
        .collect();
    assert!(
        gone.is_empty(),
        "the service page's call table names {gone:?}, which the contract does not hold"
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

/// The language a fence line names, where `line` is one: what stands behind
/// its three backticks, nothing on the line that closes a block. Each walk
/// of a page's text reads a fence line by this.
fn fence(line: &str) -> Option<&str> {
    line.trim().strip_prefix("```").map(str::trim)
}

/// The page's prose: its paragraphs and its list items, in its order. A
/// fenced block, a table's row, a heading and a mark are no prose. A list
/// item is read by itself, with the lines that continue it.
fn prose(page: &str) -> Vec<Prose> {
    let mut read: Vec<Prose> = Vec::new();
    let mut open: Option<Prose> = None;
    let mut fenced = false;
    for (at, line) in page.lines().enumerate() {
        let fence = fence(line).is_some();
        let line = line.trim();
        if fence {
            fenced = !fenced;
        }
        let no_prose = fenced
            || line.is_empty()
            || fence
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

/// One row of a table as its cells, the bars at its two ends apart. A bar
/// behind a backslash is the cell's own.
fn row_cells(row: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut held = String::new();
    for part in row.trim().trim_matches('|').split('|') {
        held.push_str(part);
        if part.ends_with('\\') {
            held.push('|');
        } else {
            cells.push(mem::take(&mut held).trim().to_owned());
        }
    }
    cells
}

/// Whether `row` is the row of dashes under a table's heading row.
fn rule_row(row: &str) -> bool {
    row.chars().all(|c| matches!(c, '|' | ':' | '-' | ' '))
}

/// The faults of a page's fences, each with its line. The fence that opens a
/// block names its language and the one that closes it is bare. So a lost
/// fence shows where a count of the fence lines would not: two lost fences
/// leave an even count. Behind a block left open the walks of the prose and
/// of the tables read nothing, to the page's end.
fn fence_faults(page: &str) -> Vec<String> {
    let mut faults = Vec::new();
    let mut fenced = false;
    for (at, line) in page.lines().enumerate() {
        let Some(language) = fence(line) else {
            continue;
        };
        let line = at + 1;
        match (fenced, language.is_empty()) {
            // The fence that closed the block above is lost, and this line
            // opens the next block.
            (true, false) => faults.push(format!(
                "line {line}: a fenced block opens inside another: the fence that closes the one above is lost"
            )),
            (false, true) => {
                faults.push(format!("line {line}: a fenced block names no language"));
                fenced = true;
            }
            (false, false) => fenced = true,
            (true, true) => fenced = false,
        }
    }
    if fenced {
        faults.push("a fenced block is left open: the prose behind it is read as code".to_owned());
    }
    faults
}

/// What a page holds over the bounds of a short page, each with its line:
/// its sentences, its paragraphs and its list items, the cells of its
/// tables, the filler words of each and of its headings, and the faults of
/// its fences.
fn over_bounds(page: &str) -> Vec<String> {
    let mut over = fence_faults(page);
    for Prose { line, item, text } in prose(page) {
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
    let mut fenced = false;
    for (at, row) in page.lines().enumerate() {
        let line = at + 1;
        if fence(row).is_some() {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let row = row.trim();
        if row.starts_with('#') {
            let plain = plain(row);
            let words = plain.split_whitespace().filter(|word| counts(word)).count();
            if words > SENTENCE_WORDS {
                over.push(format!(
                    "line {line}: a heading of {words} words (at most {SENTENCE_WORDS})"
                ));
            }
            for filler in fillers(&plain) {
                over.push(format!(
                    "line {line}: the filler word `{filler}` in a heading"
                ));
            }
        } else if row.starts_with('|') && !rule_row(row) {
            for cell in row_cells(row) {
                let plain = plain(&cell);
                let words = plain.split_whitespace().filter(|word| counts(word)).count();
                if words > CELL_WORDS {
                    over.push(format!(
                        "line {line}: a table cell of {words} words (at most {CELL_WORDS})"
                    ));
                }
                for filler in fillers(&plain) {
                    over.push(format!(
                        "line {line}: the filler word `{filler}` in a table cell"
                    ));
                }
            }
        }
    }
    over
}

#[test]
fn the_readmes_prose_stays_inside_the_bounds_a_person_reads_with_ease() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let over = over_bounds(&page);
    assert!(
        over.is_empty(),
        "the README is the page a person reads first, and it is over its bounds:\n{}",
        over.join("\n")
    );
}

#[test]
fn the_policy_spec_pages_prose_stays_inside_the_same_bounds() {
    let page = fs::read_to_string(policy_spec()).expect("the policy spec's page is readable");
    let over = over_bounds(&page);
    assert!(
        over.is_empty(),
        "the policy spec's page is held to the README's bounds, and it is over them:\n{}",
        over.join("\n")
    );
}

#[test]
fn the_stages_pages_prose_stays_inside_the_same_bounds() {
    let page = fs::read_to_string(stages()).expect("the stages' page is readable");
    let over = over_bounds(&page);
    assert!(
        over.is_empty(),
        "the stages' page is held to the README's bounds, and it is over them:\n{}",
        over.join("\n")
    );
}

#[test]
fn the_service_pages_prose_stays_inside_the_same_bounds() {
    let page = fs::read_to_string(service()).expect("the service's page is readable");
    let over = over_bounds(&page);
    assert!(
        over.is_empty(),
        "the service's page is held to the README's bounds, and it is over them:\n{}",
        over.join("\n")
    );
}

/// A sentence of `words` words: a capital first and a full stop last.
fn sentence(words: usize) -> String {
    let mut sentence = String::from("One");
    for _ in 1..words {
        sentence.push_str(" word");
    }
    sentence.push('.');
    sentence
}

/// `count` sentences of `words` words each, a space between two.
fn sentences_of(count: usize, words: usize) -> String {
    vec![sentence(words); count].join(" ")
}

/// The walk has inputs of its own that must fail: a text over each bound is
/// refused in one line, and a text that stands at a bound is passed. A bound
/// that moved, or a walk that reads past a kind of text, turns this red
/// whatever the pages hold.
#[test]
fn the_bounds_refuse_a_text_over_each_of_them() {
    let chain = format!(
        "{}; {}; {}.",
        sentence(11).trim_end_matches('.'),
        sentence(10).trim_end_matches('.').to_lowercase(),
        sentence(10).trim_end_matches('.').to_lowercase()
    );
    let texts: [(&str, String, Option<&str>); 16] = [
        ("a sentence at the bound", sentence(30), None),
        (
            "a sentence over the bound",
            sentence(31),
            Some("a sentence of 31 words"),
        ),
        (
            "a chain joined by semicolons",
            chain,
            Some("a sentence of 31 words"),
        ),
        ("five sentences", sentences_of(5, 2), None),
        ("six sentences", sentences_of(6, 2), Some("in 6 sentences")),
        ("a paragraph at the bound", sentences_of(4, 25), None),
        (
            "a paragraph over the bound",
            sentences_of(4, 26),
            Some("a paragraph of 104 words"),
        ),
        (
            "a list item over the bound",
            format!("- {}", sentences_of(3, 17)),
            Some("a list item of 51 words"),
        ),
        (
            "a filler word",
            "This page simply works.".to_owned(),
            Some("the filler word `simply`"),
        ),
        (
            "a table cell at the bound",
            format!("| A name | {} |", sentence(30)),
            None,
        ),
        (
            "a table cell over the bound",
            format!("| A name | {} |", sentence(31)),
            Some("a table cell of 31 words"),
        ),
        (
            "a heading over the bound",
            format!("## {}", sentence(31).trim_end_matches('.')),
            Some("a heading of 31 words"),
        ),
        (
            "a filler word in a heading",
            "## A seamless heading".to_owned(),
            Some("the filler word `seamless` in a heading"),
        ),
        (
            "a fenced block left open",
            format!("```sh\ncargo build\n\n{}", sentence(31)),
            Some("a fenced block is left open"),
        ),
        (
            "a block opened inside another",
            "```sh\ncargo build\n\n```sh\ncargo test\n```".to_owned(),
            Some("line 4: a fenced block opens inside another"),
        ),
        (
            "a block that names no language",
            "```\ncargo build\n```".to_owned(),
            Some("line 1: a fenced block names no language"),
        ),
    ];
    for (what, text, said) in texts {
        let over = over_bounds(&text);
        match said {
            None => assert!(
                over.is_empty(),
                "{what}: the walk refuses a text inside the bounds: {over:?}"
            ),
            Some(said) => assert!(
                over.len() == 1 && over.iter().all(|line| line.contains(said)),
                "{what}: the walk says {over:?}, and one line that holds {said:?} is due"
            ),
        }
    }
    // Two lost fences leave an even count of fence lines. The second block's
    // fence still shows the loss of the first one's, and the page's end the
    // loss of its own.
    let two_lost = over_bounds("```sh\ncargo build\n\n```sh\ncargo test\n");
    assert!(
        two_lost.len() == 2
            && two_lost[0].contains("line 4: a fenced block opens inside another")
            && two_lost[1].contains("a fenced block is left open"),
        "two lost fences: the walk says {two_lost:?}"
    );
    assert_eq!(
        row_cells("| `on\\|off` | a bar of its own |"),
        ["`on\\|off`", "a bar of its own"],
        "a bar behind a backslash is the cell's own"
    );
}

/// The anchors of a page's headings, as the forge makes them: the heading
/// in lower case, its marks dropped, a dash for each space, and a number
/// behind a name that stands again.
fn anchors(page: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut anchors: Vec<String> = Vec::new();
    let mut fenced = false;
    for line in page.lines() {
        if fence(line).is_some() {
            fenced = !fenced;
            continue;
        }
        if fenced || !line.starts_with('#') {
            continue;
        }
        let name: String = line
            .trim_start_matches('#')
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_'))
            .map(|c| if c == ' ' { '-' } else { c })
            .collect();
        let again = names.iter().filter(|held| **held == name).count();
        anchors.push(if again == 0 {
            name.clone()
        } else {
            format!("{name}-{again}")
        });
        names.push(name);
    }
    anchors
}

/// The targets of a page's links, each with its line: what stands between
/// `](` and the next `)`, outside the fenced blocks and the code spans.
fn links(page: &str) -> Vec<(usize, String)> {
    let mut links = Vec::new();
    let mut fenced = false;
    for (at, line) in page.lines().enumerate() {
        if fence(line).is_some() {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        // A code span is no text: every second part of a line cut at its
        // backticks.
        let text = line.split('`').step_by(2).collect::<Vec<_>>().join(" ");
        links.extend(
            text.split("](")
                .skip(1)
                .filter_map(|rest| rest.split(')').next())
                .map(|target| (at + 1, target.to_owned())),
        );
    }
    links
}

/// The links of the page at `path` that lead nowhere, each with its line: a
/// path the checkout does not hold, or a heading its page does not hold. A
/// link to another site is not followed.
fn dead_links(path: &Path) -> Vec<String> {
    let Ok(page) = fs::read_to_string(path) else {
        return vec![format!("{}: the page is not readable", path.display())];
    };
    let mut dead = Vec::new();
    for (line, target) in links(&page) {
        if target.contains("://") || target.starts_with("mailto:") {
            continue;
        }
        let (file, heading) = match target.split_once('#') {
            Some((file, heading)) => (file, Some(heading)),
            None => (target.as_str(), None),
        };
        let to = match path.parent() {
            Some(folder) if !file.is_empty() => folder.join(file),
            _ => path.to_path_buf(),
        };
        if !to.exists() {
            dead.push(format!(
                "line {line}: {target}: the checkout holds no such path"
            ));
            continue;
        }
        let Some(heading) = heading else { continue };
        if to.extension().is_none_or(|extension| extension != "md") {
            continue;
        }
        let held = fs::read_to_string(&to)
            .map(|page| anchors(&page))
            .unwrap_or_default();
        if !held.iter().any(|anchor| anchor == heading) {
            dead.push(format!(
                "line {line}: {target}: the page holds no such heading"
            ));
        }
    }
    dead
}

#[test]
fn every_link_of_a_page_resolves_heading_and_all() {
    for (name, path) in [
        ("README", readme()),
        ("policy spec's page", policy_spec()),
        ("stages' page", stages()),
        ("service's page", service()),
        ("reference", reference()),
    ] {
        let dead = dead_links(&path);
        assert!(
            dead.is_empty(),
            "the {name} holds links that lead nowhere:\n{}",
            dead.join("\n")
        );
    }
    // The reading itself, on short texts: a heading's anchor and a link's
    // target are what the forge makes of them.
    assert_eq!(
        anchors("# One\n## Two words\n```\n# no heading\n```\n## Two words\n### 9.3 The `options` **[partly]**"),
        ["one", "two-words", "two-words-1", "93-the-options-partly"],
        "a heading's anchor"
    );
    assert_eq!(
        links("See [one](a.md#b) and `[code](c.md)`.\n```\n[fenced](d.md)\n```\n[two](#e)"),
        [(1, "a.md#b".to_owned()), (5, "#e".to_owned())],
        "a link's target"
    );
}

/// The commands a page shows: the lines of its `sh` blocks, a line that ends
/// with a backslash joined to the next.
fn commands(page: &str) -> Vec<String> {
    blocks(page)
        .iter()
        .filter(|block| block.language == "sh")
        .flat_map(|block| {
            let joined = block.body.replace("\\\n", " ");
            let lines: Vec<String> = joined.lines().map(str::to_owned).collect();
            lines
        })
        .filter(|line| !line.trim().is_empty())
        .collect()
}

/// Whether a file under `folder` holds `text`: its files and those of its
/// folders are read.
fn held_under(folder: &Path, text: &str) -> bool {
    let Ok(entries) = fs::read_dir(folder) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if path.is_dir() {
            held_under(&path, text)
        } else {
            fs::read_to_string(&path).is_ok_and(|file| file.contains(text))
        }
    })
}

/// What a command names that the checkout does not hold. A cargo command
/// names this package, features and an example of its manifest, and a unit
/// test of the library. A command of the standalone binary names flags
/// its command line declares. A command of another kind is a fault too: the
/// case reads it once it is taught to.
fn command_faults(command: &str) -> Vec<String> {
    let manifest = include_str!("../Cargo.toml");
    let package = env!("CARGO_PKG_NAME");
    let words: Vec<&str> = command.split_whitespace().collect();
    let value = |flag: &str| {
        let at = words.iter().position(|word| *word == flag)?;
        words.get(at + 1).copied()
    };
    let mut faults = Vec::new();
    let first = words.first().copied().unwrap_or_default();
    if first == "cargo" {
        if value("-p") != Some(package) {
            faults.push(format!(
                "`{command}`: it names another package than {package}"
            ));
        }
        for feature in value("--features")
            .into_iter()
            .flat_map(|list| list.split(','))
        {
            if !manifest
                .lines()
                .any(|line| line.starts_with(&format!("{feature} = [")))
            {
                faults.push(format!(
                    "`{command}`: the manifest declares no feature {feature}"
                ));
            }
        }
        if let Some(example) = value("--example") {
            if !manifest.contains(&format!("[[example]]\nname = \"{example}\"")) {
                faults.push(format!(
                    "`{command}`: the manifest declares no example {example}"
                ));
            }
        }
        // A test command's last word is the test's name, where a flag
        // that takes a value does not stand before it.
        let takes_value = ["-p", "--features", "--example", "--test"];
        let name = match words.as_slice() {
            [_, "test", .., before, last]
                if !last.starts_with('-') && !takes_value.contains(before) =>
            {
                Some(*last)
            }
            _ => None,
        };
        if let Some(name) = name {
            // The library's unit tests stand under `tests/unit`.
            let unit_tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/unit");
            if !held_under(&unit_tests, &format!("fn {name}(")) {
                faults.push(format!(
                    "`{command}`: the library's unit tests hold no test {name}"
                ));
            }
        }
    } else if first.rsplit('/').next() == Some(package) {
        if !manifest.contains(&format!("[[bin]]\nname = \"{package}\"")) {
            faults.push(format!(
                "`{command}`: the manifest declares no binary {package}"
            ));
        }
        let declared = flags();
        for flag in words.iter().filter(|word| word.starts_with("--")) {
            if !declared.iter().any(|held| held == flag) {
                faults.push(format!(
                    "`{command}`: the command line declares no flag {flag}"
                ));
            }
        }
    } else {
        faults.push(format!("`{command}`: a command this case does not read"));
    }
    faults
}

#[test]
fn every_command_of_the_readme_names_what_the_checkout_holds() {
    let page = fs::read_to_string(readme()).expect("the README is readable");
    let commands = commands(&page);
    assert_eq!(
        commands.len(),
        README_COMMANDS,
        "the README shows {README_COMMANDS} commands: a fence was lost, or a new command raises the count"
    );
    let faults: Vec<String> = commands
        .iter()
        .flat_map(|command| command_faults(command))
        .collect();
    assert!(
        faults.is_empty(),
        "the README shows commands that name what the checkout does not hold:\n{}",
        faults.join("\n")
    );
    // The reading itself, on short commands: each names one thing that is
    // gone, and is refused for it in one line.
    for (command, said) in [
        (
            "cargo test -p smg-grand-central --locked --lib a_test_that_stands_nowhere",
            "no test a_test_that_stands_nowhere",
        ),
        (
            "cargo run -p smg-grand-central --features no-such-feature --example in_process",
            "no feature no-such-feature",
        ),
        (
            "cargo run -p smg-grand-central --features in-process --example no_such_example",
            "no example no_such_example",
        ),
        (
            "cargo build --release --locked -p another-package",
            "another package",
        ),
        (
            "target/release/smg-grand-central --no-such-flag 1",
            "no flag --no-such-flag",
        ),
        ("make all", "a command this case does not read"),
    ] {
        let faults = command_faults(command);
        assert!(
            faults.len() == 1 && faults[0].contains(said),
            "`{command}`: the reading says {faults:?}, and one line that holds {said:?} is due"
        );
    }
}

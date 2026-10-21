//! The shape matcher against Python's `regex` module, the dialect response templates are written
//! in.
//!
//! bellwether's `bellwether patterns` records, with `regex` and `DOTALL` as transformers compiles
//! a template's patterns, what `match` says about every pattern a call shape runs in the public
//! response templates (the header after a call's marker, the text before its closing marker, a
//! parameter's tag with its value's ends), and about a few patterns on abstract tags for what no
//! public template writes: whether each text matches, where the match ends, what `name`, `key` and
//! `value` capture, and which end closed a value. For a written text it also records the answer for
//! every prefix taken as the whole text. Each pattern is lowered here by hand into a
//! [`Pattern`] and its [`End`]s, next to the source it is lowered from, which must be the source
//! the fixture recorded. The recordings are `patterns/regex.jsonl`, `patterns/search.jsonl` and
//! `patterns/classes.json` of the tree `BELLWETHER_FIXTURES` names, the one `bellwether unpack`
//! writes from the commit CI pins; without them each test prints why it skips and passes.
//!
//! Every text is fed at every chunking (whole, cut once at every place for a short one, a
//! character at a time, thirty seeded plans), and the matcher's answer must be the recorded one.
//! A verdict the matcher gives before the end of a text must be the one it gives at the end, and
//! a verdict it gives at the edge of a prefix must be the recorded answer both for the prefix and
//! for the whole text: a decision is taken only when no continuation can change it. The classes
//! are built from the module's own `\w`, `\s` and `\d` tables, also recorded, and checked against
//! the module's answer for a set of probe characters.
//!
//! The script also records where a region closes: for each trailer, the text before a call's
//! closing marker, the start `search` gives for the trailer followed by the marker at the end of a
//! text. [`Suffix`] must give that start at every chunking, and, fed a written text a character
//! at a time, the start recorded for each prefix as it goes.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one reads the patterns' directory only"
)]
mod common;

use std::{collections::BTreeMap, fs, path::Path};

use serde::Deserialize;
use symphony::shape::{Atom, Class, End, Matcher, Pattern, Slot, Suffix, Verdict};

/// One pattern as the fixture recorded it.
#[derive(Deserialize)]
struct Recorded {
    id: String,
    source: String,
    before: Option<String>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    /// The answer for the whole text.
    #[serde(rename = "match")]
    answer: Option<Answer>,
    /// For a written text, the answer for each prefix of it, by its length in characters.
    #[serde(default)]
    prefixes: Option<Vec<Option<Answer>>>,
}

/// A trailer as the fixture recorded it: the marker after it, and where the region closes.
#[derive(Deserialize)]
struct Searched {
    id: String,
    source: String,
    terminal: String,
    before: Option<String>,
    cases: Vec<Search>,
}

#[derive(Deserialize)]
struct Search {
    text: String,
    /// Where the trailer starts when the marker comes at the end of the text.
    start: Option<usize>,
    /// For a written text, the start for each prefix of it, by its length in characters.
    #[serde(default)]
    prefixes: Option<Vec<Option<usize>>>,
}

/// A match, in bytes of the text: where it ends, what it captured, which end closed it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
struct Answer {
    end: usize,
    name: Option<[usize; 2]>,
    key: Option<[usize; 2]>,
    value: Option<[usize; 2]>,
    ending: Option<usize>,
}

/// The module's tables, recorded with it.
#[derive(Deserialize)]
struct Tables {
    regex: String,
    python: String,
    word: Vec<String>,
    space: Vec<String>,
    digit: Vec<String>,
    probes: String,
    members: BTreeMap<String, String>,
}

/// The classes the patterns use, built from the tables.
struct Classes {
    word: Class,
    space: Class,
    not_space: Class,
    any: Class,
}

#[expect(
    clippy::panic,
    reason = "a fixture that cannot be read ends the test with its path"
)]
fn fixture(patterns: &Path, name: &str) -> String {
    let path = patterns.join(name);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

#[expect(
    clippy::expect_used,
    reason = "tables that cannot be read end the test"
)]
fn tables(patterns: &Path) -> Tables {
    let tables: Tables =
        serde_json::from_str(&fixture(patterns, "classes.json")).expect("classes.json");
    // The module's version is the recording's to state (bellwether pins it); the Python it ran
    // under is the one transformers' templates are written for.
    assert_eq!(tables.python.as_str(), "3.12");
    tables
}

/// A table's ranges, each written as its first and last character.
#[expect(
    clippy::panic,
    reason = "a range written otherwise ends the test with the range"
)]
fn ranges(table: &[String]) -> Vec<(char, char)> {
    table
        .iter()
        .map(|range| {
            let mut chars = range.chars();
            match (chars.next(), chars.next(), chars.next()) {
                (Some(first), Some(last), None) => (first, last),
                _ => panic!("a range is two characters: {range:?}"),
            }
        })
        .collect()
}

impl Classes {
    fn new(tables: &Tables) -> Self {
        Self {
            word: Class::new(ranges(&tables.word), false),
            space: Class::new(ranges(&tables.space), false),
            not_space: Class::new(ranges(&tables.space), true),
            any: Class::any(),
        }
    }

    /// The class a pattern's source spells.
    #[expect(
        clippy::panic,
        reason = "a class the test has no spelling for ends it with the class"
    )]
    fn named(&self, tables: &Tables, source: &str) -> Class {
        match source {
            "\\w" => self.word.clone(),
            "\\W" => Class::new(ranges(&tables.word), true),
            "\\s" => self.space.clone(),
            "\\S" => self.not_space.clone(),
            "\\d" => Class::new(ranges(&tables.digit), false),
            "\\D" => Class::new(ranges(&tables.digit), true),
            "." => self.any.clone(),
            "[^>]" => none_of(">"),
            "[^\"]" => none_of("\""),
            "[^>\\n]" | "[^\\n>]" => none_of(">\n"),
            "[^<]" => none_of("<"),
            other => panic!("no class for {other:?}"),
        }
    }
}

fn none_of(chars: &str) -> Class {
    Class::new(chars.chars().map(|c| (c, c)), true)
}

fn lit(text: &str) -> Atom {
    Atom::Lit(text.into())
}

fn greedy(class: &Class, min: u16, max: Option<u16>) -> Atom {
    Atom::Run {
        class: class.clone(),
        min,
        max,
        greedy: true,
        capture: None,
    }
}

fn lazy(class: &Class, min: u16, max: Option<u16>) -> Atom {
    Atom::Run {
        class: class.clone(),
        min,
        max,
        greedy: false,
        capture: None,
    }
}

#[expect(
    clippy::panic,
    reason = "capturing anything but a run is a mistake in the test's lowering"
)]
fn captured(slot: Slot, atom: Atom) -> Atom {
    match atom {
        Atom::Run {
            class,
            min,
            max,
            greedy,
            capture: None,
        } => Atom::Run {
            class,
            min,
            max,
            greedy,
            capture: Some(slot),
        },
        other => panic!("only a run is captured: {other:?}"),
    }
}

fn pattern(atoms: impl IntoIterator<Item = Atom>) -> Pattern {
    Pattern(atoms.into_iter().collect())
}

fn end(consume: impl IntoIterator<Item = Atom>, ahead: impl IntoIterator<Item = Atom>) -> End {
    End {
        consume: pattern(consume),
        ahead: pattern(ahead),
    }
}

/// A pattern lowered by hand: the source it is lowered from, its atoms, and its value's ends.
struct Lowered {
    source: &'static str,
    pattern: Pattern,
    ends: Vec<End>,
}

fn lowered(source: &'static str, atoms: impl IntoIterator<Item = Atom>) -> Lowered {
    Lowered {
        source,
        pattern: pattern(atoms),
        ends: Vec::new(),
    }
}

fn lowered_with_ends(
    source: &'static str,
    atoms: impl IntoIterator<Item = Atom>,
    ends: impl IntoIterator<Item = End>,
) -> Lowered {
    Lowered {
        source,
        pattern: pattern(atoms),
        ends: ends.into_iter().collect(),
    }
}

/// Muse Glimmer's parameter tag, the longest source here.
const MUSE_PARAMETER: &str =
    r#"<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>"#;

/// Every recorded pattern, lowered as a compiler lowers it: consecutive characters are one
/// literal, `(?:LIT)?` is optional, a class or a repeated class is a run (`\n?` one of at most one
/// character), a named group around a run captures it, and a value's ends are the alternatives
/// after its `.*?`, each split at its lookahead.
#[expect(
    clippy::panic,
    reason = "a recorded pattern with no lowering ends the test with its id"
)]
fn lower(id: &str, c: &Classes) -> Lowered {
    let (w, s, any) = (&c.word, &c.space, &c.any);
    match id {
        "gemma4/header" => lowered(r"(?P<name>\w+)", [captured(Slot::Name, greedy(w, 1, None))]),
        "muse-glimmer/header" => lowered(
            r#"\b[^>]*?\bname="(?P<name>[^"]+)">"#,
            [
                Atom::Boundary,
                lazy(&none_of(">"), 0, None),
                Atom::Boundary,
                lit("name=\""),
                captured(Slot::Name, greedy(&none_of("\""), 1, None)),
                lit("\">"),
            ],
        ),
        "muse-glimmer/parameter" => lowered_with_ends(
            MUSE_PARAMETER,
            [
                lit("<atem:parameter"),
                Atom::Boundary,
                lazy(&none_of(">"), 0, None),
                Atom::Boundary,
                lit("name=\""),
                captured(Slot::Key, greedy(&none_of("\""), 1, None)),
                lit("\""),
                lazy(&none_of(">"), 0, None),
                lit(">"),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [end([lit("</atem:parameter>")], [])],
        ),
        "serve-qwen3_5/header" => lowered(
            r"\s*<function=(?P<name>[^>\n]+)>",
            [
                greedy(s, 0, None),
                lit("<function="),
                captured(Slot::Name, greedy(&none_of(">\n"), 1, None)),
                lit(">"),
            ],
        ),
        "serve-qwen3_5/trailer" => {
            lowered(r"</function>\s*", [lit("</function>"), greedy(s, 0, None)])
        }
        "serve-qwen3_5/close" => lowered(
            r"</function>\s*</tool_call>",
            [lit("</function>"), greedy(s, 0, None), lit("</tool_call>")],
        ),
        "serve-qwen3_5/parameter" => lowered_with_ends(
            r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
            [
                lit("<parameter="),
                captured(Slot::Key, greedy(&none_of(">\n"), 1, None)),
                lit(">"),
                greedy(s, 0, None),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [end([greedy(s, 0, None), lit("</parameter>")], [])],
        ),
        "hf-qwen3/header" => lowered(
            r"\s*<function=(?P<name>\w+)>",
            [
                greedy(s, 0, None),
                lit("<function="),
                captured(Slot::Name, greedy(w, 1, None)),
                lit(">"),
            ],
        ),
        "hf-qwen3/parameter" => lowered_with_ends(
            r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
            [
                lit("<parameter="),
                captured(Slot::Key, greedy(w, 1, None)),
                lit(">"),
                greedy(s, 0, None),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [end([greedy(s, 0, None), lit("</parameter>")], [])],
        ),
        "hf-kv-lines/header" => lowered(
            r"\s*<function=(?P<name>\w+)>\n",
            [
                greedy(s, 0, None),
                lit("<function="),
                captured(Slot::Name, greedy(w, 1, None)),
                lit(">\n"),
            ],
        ),
        "trl-qwen3_5/header" => lowered(
            r"\s*<function=(?P<name>[^\n>]+)>",
            [
                greedy(s, 0, None),
                lit("<function="),
                captured(Slot::Name, greedy(&none_of("\n>"), 1, None)),
                lit(">"),
            ],
        ),
        "hf-gpt-oss/header" => lowered(
            r"(?P<name>\w+).*?<\|message\|>",
            [
                captured(Slot::Name, greedy(w, 1, None)),
                lazy(any, 0, None),
                lit("<|message|>"),
            ],
        ),
        "trl-gptoss/header" => lowered(
            r"(?P<name>\S+?)<\|channel\|>commentary\s+json<\|message\|>",
            [
                captured(Slot::Name, lazy(&c.not_space, 1, None)),
                lit("<|channel|>commentary"),
                greedy(s, 1, None),
                lit("json<|message|>"),
            ],
        ),
        "trl-llama3/header" => lowered(
            r#"\s*"(?P<name>[^"]+)",\s*"parameters":\s*"#,
            [
                greedy(s, 0, None),
                lit("\""),
                captured(Slot::Name, greedy(&none_of("\""), 1, None)),
                lit("\","),
                greedy(s, 0, None),
                lit("\"parameters\":"),
                greedy(s, 0, None),
            ],
        ),
        "trl-llama3/close" => lowered(
            r"\}<\|eot_id\|>\s*",
            [lit("}<|eot_id|>"), greedy(s, 0, None)],
        ),
        "trl-glm4moe/header" => lowered(
            r"(?P<name>\S+)\n?",
            [
                captured(Slot::Name, greedy(&c.not_space, 1, None)),
                greedy(&Class::new([('\n', '\n')], false), 0, Some(1)),
            ],
        ),
        "trl-glm4moe/parameter" => lowered_with_ends(
            r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)</arg_value>",
            [
                lit("<arg_key>"),
                captured(Slot::Key, greedy(&none_of("<"), 1, None)),
                lit("</arg_key>"),
                greedy(s, 0, None),
                lit("\n<arg_value>"),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [end([lit("</arg_value>")], [])],
        ),
        "abstract/optional-and-bounded" => lowered(
            r"(?:<a>)?(?P<name>\w{2,4})y",
            [
                Atom::Maybe("<a>".into()),
                captured(Slot::Name, greedy(w, 2, Some(4))),
                lit("y"),
            ],
        ),
        "abstract/lazy-bounded" => lowered(
            r"(?P<name>\w{2,4}?)y",
            [captured(Slot::Name, lazy(w, 2, Some(4))), lit("y")],
        ),
        "abstract/two-optionals" => lowered(
            r"(?:ab)?(?:a)?b",
            [Atom::Maybe("ab".into()), Atom::Maybe("a".into()), lit("b")],
        ),
        "abstract/boundaries" => lowered(
            r"\b(?P<name>\w+)\b\s*=",
            [
                Atom::Boundary,
                captured(Slot::Name, greedy(w, 1, None)),
                Atom::Boundary,
                greedy(s, 0, None),
                lit("="),
            ],
        ),
        "abstract/filler" => lowered(r"[^<]*?<a>", [lazy(&none_of("<"), 0, None), lit("<a>")]),
        "abstract/lookahead-end" => lowered_with_ends(
            r"<k=(?P<key>\w+)>(?P<value>.*?)(?:</k>|(?=<k=))",
            [
                lit("<k="),
                captured(Slot::Key, greedy(w, 1, None)),
                lit(">"),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [end([lit("</k>")], []), end([], [lit("<k=")])],
        ),
        "trl-llama3/trailer" => lowered(r"\}", [lit("}")]),
        "abstract/word-then-boundary" => lowered(r"\w+\b", [greedy(w, 1, None), Atom::Boundary]),
        "abstract/boundary-before-a-word" => {
            lowered(r"\s*\bz", [greedy(s, 0, None), Atom::Boundary, lit("z")])
        }
        "abstract/lazy-tag" => lowered(
            r"<[^>]*?>\s*",
            [
                lit("<"),
                lazy(&none_of(">"), 0, None),
                lit(">"),
                greedy(s, 0, None),
            ],
        ),
        "abstract/lookahead-end-trimmed" => lowered_with_ends(
            r"<k=(?P<key>\w+)>\s*(?P<value>.*?)(?:\s*</k>|\s*(?=<k=))",
            [
                lit("<k="),
                captured(Slot::Key, greedy(w, 1, None)),
                lit(">"),
                greedy(s, 0, None),
                captured(Slot::Value, lazy(any, 0, None)),
            ],
            [
                end([greedy(s, 0, None), lit("</k>")], []),
                end([greedy(s, 0, None)], [lit("<k=")]),
            ],
        ),
        other => panic!("no lowering for {other}"),
    }
}

/// The ways to cut `text`: whole, once at every place for a text of at most 64 bytes, a character
/// at a time, and thirty seeded plans of one to eight characters a piece.
fn plans(text: &str) -> Vec<Vec<usize>> {
    let starts: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    let mut plans = vec![Vec::new(), starts.iter().skip(1).copied().collect()];
    if text.len() <= 64 {
        plans.extend(starts.iter().skip(1).map(|&cut| vec![cut]));
    }
    for seed in 1..=30u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cuts = Vec::new();
        let mut at = 0;
        while at < starts.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            at += 1 + (state >> 33) as usize % 8;
            if at < starts.len() {
                cuts.push(starts[at]);
            }
        }
        plans.push(cuts);
    }
    plans
}

/// The matcher's verdict in the fixture's terms; `None` for no match.
#[expect(clippy::panic, reason = "a finished matcher always has an answer")]
fn answer(verdict: &Verdict) -> Option<Answer> {
    match verdict {
        Verdict::Matched(found) => {
            let span = |slot| found.span(slot).map(|span| [span.start, span.end]);
            Some(Answer {
                end: found.end(),
                name: span(Slot::Name),
                key: span(Slot::Key),
                value: span(Slot::Value),
                ending: found.ending(),
            })
        }
        Verdict::NoMatch => None,
        Verdict::Pending => panic!("a finished matcher has an answer"),
    }
}

/// Feed `text` cut at `cuts`, then finish: the answer. A verdict given on the way must stay.
fn replay(
    lowered: &Lowered,
    word: &Class,
    before: Option<char>,
    text: &str,
    cuts: &[usize],
) -> Option<Answer> {
    let mut matcher = Matcher::new(&lowered.pattern, &lowered.ends, word, before).hold_cap(None);
    let mut early: Option<Verdict> = None;
    let mut from = 0;
    for &cut in cuts.iter().chain([&text.len()]) {
        let said = matcher.feed(&text[from..cut]);
        from = cut;
        match &early {
            Some(early) => assert_eq!(&said, early, "a verdict stays"),
            None if said != Verdict::Pending => early = Some(said),
            None => {}
        }
    }
    let last = matcher.finish();
    if let Some(early) = early {
        assert_eq!(last, early, "a verdict stays at the end");
    }
    answer(&last)
}

#[test]
fn every_class_holds_what_the_regex_module_says_it_holds() {
    let Some(patterns) = common::patterns() else {
        return;
    };
    let tables = tables(&patterns);
    let classes = Classes::new(&tables);
    let probes: Vec<char> = tables.probes.chars().collect();
    for (source, members) in &tables.members {
        let class = classes.named(&tables, source);
        assert_eq!(members.len(), probes.len(), "{source}");
        for (&c, member) in probes.iter().zip(members.chars()) {
            assert_eq!(class.contains(c), member == '1', "{source} and {c:?}");
        }
    }
    // The tables read by search are the tables as written, at every character there is.
    for table in [&tables.word, &tables.space, &tables.digit] {
        let written = ranges(table);
        let class = Class::new(written.iter().copied(), false);
        let mut next = written.iter().peekable();
        for c in (0..=u32::from(char::MAX)).filter_map(char::from_u32) {
            while next.peek().is_some_and(|&&(_, last)| last < c) {
                next.next();
            }
            let inside = next.peek().is_some_and(|&&(first, _)| first <= c);
            assert_eq!(class.contains(c), inside, "{c:?}");
        }
    }
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the count of what was checked is test diagnostic output"
)]
fn the_matcher_answers_as_the_regex_module_does_at_every_chunking() {
    let Some(dir) = common::patterns() else {
        return;
    };
    let tables = tables(&dir);
    let classes = Classes::new(&tables);
    let mut patterns = 0;
    let mut texts = 0;
    let mut replays = 0;
    for line in fixture(&dir, "regex.jsonl").lines() {
        let recorded: Recorded = serde_json::from_str(line).expect("a recorded pattern");
        let id = recorded.id.as_str();
        let lowered = lower(id, &classes);
        assert_eq!(
            recorded.source, lowered.source,
            "{id}: lowered from the recorded source"
        );
        assert_eq!(lowered.pattern.check(), Ok(()), "{id}");
        for end in &lowered.ends {
            assert_eq!(end.check(), Ok(()), "{id}");
        }
        let before = one_before(id, recorded.before.as_deref());
        patterns += 1;
        for case in &recorded.cases {
            let text = &*case.text;
            texts += 1;
            for cuts in plans(text) {
                let got = replay(&lowered, &classes.word, before, text, &cuts);
                assert_eq!(got, case.answer, "{id}: {text:?} cut at {cuts:?}");
                replays += 1;
            }
            let Some(prefixes) = &case.prefixes else {
                continue;
            };
            let ends: Vec<usize> = text
                .char_indices()
                .map(|(at, _)| at)
                .chain([text.len()])
                .collect();
            assert_eq!(prefixes.len(), ends.len(), "{id}: {text:?}");
            for (&upto, expected) in ends.iter().zip(prefixes) {
                let prefix = &text[..upto];
                let whole = replay(&lowered, &classes.word, before, prefix, &[]);
                assert_eq!(&whole, expected, "{id}: {prefix:?} as the whole text");
                let each = plans(prefix).swap_remove(1);
                let apart = replay(&lowered, &classes.word, before, prefix, &each);
                assert_eq!(&apart, expected, "{id}: {prefix:?} a character at a time");
                let mut matcher =
                    Matcher::new(&lowered.pattern, &lowered.ends, &classes.word, before)
                        .hold_cap(None);
                let said = matcher.feed(prefix);
                if said != Verdict::Pending {
                    let decided = answer(&said);
                    assert_eq!(
                        &decided, expected,
                        "{id}: decided at {prefix:?}, then the end came"
                    );
                    assert_eq!(
                        decided, case.answer,
                        "{id}: decided at {prefix:?}, then {text:?} came"
                    );
                }
                replays += 3;
            }
        }
    }
    println!(
        "{patterns} patterns, {texts} texts, {replays} replays: every answer the module's (regex {})",
        tables.regex
    );
    assert_eq!((patterns, texts), (24, 1_790), "the fixture as recorded");
}

/// The one character before a text, as the fixture writes it.
#[expect(
    clippy::expect_used,
    reason = "a fixture with no character where one is written ends the test"
)]
fn one_before(id: &str, before: Option<&str>) -> Option<char> {
    before.map(|before| {
        let mut chars = before.chars();
        let c = chars.next().expect("one character before");
        assert_eq!(chars.next(), None, "{id}: one character before");
        c
    })
}

/// Feed `text` cut at `cuts` to a search, then ask where a match ending there starts.
fn search(
    lowered: &Lowered,
    word: &Class,
    before: Option<char>,
    text: &str,
    cuts: &[usize],
    next: Option<char>,
) -> Option<usize> {
    let mut search = Suffix::new(&lowered.pattern, word, before).hold_cap(None);
    let mut from = 0;
    for &cut in cuts.iter().chain([&text.len()]) {
        search.feed(&text[from..cut]);
        from = cut;
    }
    search.start(next)
}

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the count of what was checked is test diagnostic output"
)]
fn the_search_finds_where_the_regex_module_closes_a_region_at_every_chunking() {
    let Some(dir) = common::patterns() else {
        return;
    };
    let tables = tables(&dir);
    let classes = Classes::new(&tables);
    let mut trailers = 0;
    let mut texts = 0;
    let mut asked = 0;
    for line in fixture(&dir, "search.jsonl").lines() {
        let recorded: Searched = serde_json::from_str(line).expect("a recorded trailer");
        let id = recorded.id.as_str();
        let lowered = lower(id, &classes);
        assert_eq!(
            recorded.source, lowered.source,
            "{id}: lowered from the recorded source"
        );
        assert_eq!(lowered.pattern.check(), Ok(()), "{id}");
        let before = one_before(id, recorded.before.as_deref());
        let next = recorded.terminal.chars().next();
        trailers += 1;
        for case in &recorded.cases {
            let text = &*case.text;
            texts += 1;
            for cuts in plans(text) {
                let got = search(&lowered, &classes.word, before, text, &cuts, next);
                assert_eq!(got, case.start, "{id}: {text:?} cut at {cuts:?}");
                asked += 1;
            }
            let Some(prefixes) = &case.prefixes else {
                continue;
            };
            let mut suffix = Suffix::new(&lowered.pattern, &classes.word, before).hold_cap(None);
            let mut chars = text.char_indices();
            for expected in prefixes {
                let read = chars.next();
                assert_eq!(&suffix.start(next), expected, "{id}: {text:?} as it grows");
                if let Some((at, c)) = read {
                    suffix.feed(&text[at..at + c.len_utf8()]);
                }
                asked += 1;
            }
        }
    }
    println!(
        "{trailers} trailers, {texts} texts, {asked} answers: every start the module's (regex {})",
        tables.regex
    );
    assert_eq!(trailers, 5, "the fixture as recorded");
}

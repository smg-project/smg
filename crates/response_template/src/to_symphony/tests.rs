//! Unit tests of the lowering: the classes it resolves, the pieces each construct lowers into,
//! and its answers against Python's `regex` module, through symphony's matcher.
//!
//! symphony's `tests/fixtures/shape/` records, with `regex` 2026.9.10, the module's `\w`, `\s` and
//! `\d` tables, and what `match` gives for every pattern a call shape runs in the public response
//! templates (the header after a call's marker, the text before its closing marker, and a
//! parameter's tag with the ways its value ends) at every prefix of a set of texts. These tests
//! read those recordings: the classes the lowering resolves are the module's, and each recorded
//! pattern, lowered here from its source, answers as the module does.

use std::{fs, path::PathBuf};

use serde::Deserialize;
use symphony::shape::{Atom, Class, End, Matcher, Slot, Verdict};

use super::lower::{
    anchor_literals, chars, lower, lower_tag, pattern, split_close, split_open, Branches, Captures,
    Chars, EndItems, Item, Tag,
};
use crate::{pyre::Regex, to_symphony::refusal::Construct};

/// A file of symphony's shape recordings.
fn shape_fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../symphony/tests/fixtures/shape")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[derive(Deserialize)]
struct Tables {
    regex: String,
    word: Vec<String>,
    space: Vec<String>,
    digit: Vec<String>,
}

/// A recorded table: ranges, each written as its first and last character.
fn recorded_ranges(table: &[String]) -> Vec<(char, char)> {
    table
        .iter()
        .map(|range| {
            let mut chars = range.chars();
            let (Some(first), Some(last), None) = (chars.next(), chars.next(), chars.next()) else {
                panic!("a range is two characters: {range:?}");
            };
            (first, last)
        })
        .collect()
}

fn escape(source: &str) -> Chars {
    let regex = Regex::parse(source).unwrap();
    match &regex.node {
        crate::pyre::Node::One(set) => chars(set).unwrap(),
        other => panic!("{source}: {other:?}"),
    }
}

#[test]
fn the_classes_are_the_regex_modules_own() {
    let tables: Tables = serde_json::from_str(&shape_fixture("classes.json")).unwrap();
    assert_eq!(tables.regex, "2026.9.10");
    for (source, table) in [
        ("\\w", &tables.word),
        ("\\s", &tables.space),
        ("\\d", &tables.digit),
    ] {
        let resolved = escape(source);
        let recorded = Chars::new(recorded_ranges(table), false);
        for c in (0..=u32::from(char::MAX)).filter_map(char::from_u32) {
            assert_eq!(resolved.contains(c), recorded.contains(c), "{source} {c:?}");
        }
    }
    // A class with an escape and its complement is any character, also negated, as the module
    // reads it.
    for source in ["[\\s\\S]", "[^\\d\\D]", "."] {
        let any = escape(source);
        assert!(
            ['\0', 'a', '\n', char::MAX]
                .iter()
                .all(|&c| any.contains(c)),
            "{source}"
        );
    }
    let negated = escape("[^\\s>]");
    assert!(!negated.contains(' ') && !negated.contains('>') && negated.contains('a'));
    let mixed = escape("[a-c\\d_]");
    assert!(
        mixed.contains('b') && mixed.contains('٣') && mixed.contains('_') && !mixed.contains('d')
    );
}

/// One recorded pattern: its source, the character before the text it is run over, and its
/// cases.
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
    #[serde(rename = "match")]
    answer: Option<Answer>,
    #[serde(default)]
    prefixes: Option<Vec<Option<Answer>>>,
}

#[derive(Debug, Default, Deserialize, PartialEq, Eq)]
struct Answer {
    end: usize,
    name: Option<[usize; 2]>,
    key: Option<[usize; 2]>,
    value: Option<[usize; 2]>,
    ending: Option<usize>,
}

/// The groups a header's pattern captures, by name.
fn captures(regex: &Regex) -> Captures {
    Captures(
        regex
            .names
            .iter()
            .filter_map(|(name, number)| {
                let slot = match name.as_str() {
                    "name" => Slot::Name,
                    "key" => Slot::Key,
                    "value" => Slot::Value,
                    _ => return None,
                };
                Some((*number, slot))
            })
            .collect(),
    )
}

fn answer(
    text: &str,
    (pattern, ends): (&symphony::shape::Pattern, &[End]),
    word: &Class,
    before: Option<char>,
) -> Option<Answer> {
    let mut matcher = Matcher::new(pattern, ends, word, before).hold_cap(None);
    let _ = matcher.feed(text);
    match matcher.finish() {
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

/// The ends of a lowered tag as the matcher runs them.
fn ends(tag: &Tag) -> Vec<End> {
    tag.ends
        .iter()
        .map(|end| End {
            consume: pattern(&end.consume),
            ahead: pattern(&end.ahead),
        })
        .collect()
}

/// A recorded pattern lowered from its source: a header or a trailer as one pattern, a parameter's
/// tag with the ways its value ends.
fn lowered_recording(recorded: &Recorded) -> (symphony::shape::Pattern, Vec<End>) {
    let regex = Regex::parse(&recorded.source).unwrap();
    let captures = captures(&regex);
    let value = regex
        .names
        .iter()
        .find(|(name, _)| name == "value")
        .map(|(_, number)| *number);
    if let Some(value) = value {
        let (tag, refused) = lower_tag(&regex, &captures, value);
        assert_eq!(refused, [], "{}", recorded.id);
        let tag = tag.unwrap();
        return (pattern(&tag.tag), ends(&tag));
    }
    let branches =
        lower(&regex, &captures).unwrap_or_else(|refused| panic!("{}: {refused:?}", recorded.id));
    let [branch] = branches.as_slice() else {
        panic!("{}: {} alternatives", recorded.id, branches.len());
    };
    (pattern(branch), Vec::new())
}

/// Every pattern of a public template's call that symphony's recordings hold (the header after a
/// call's marker, the text before its closing marker, a parameter's tag with the ways its value
/// ends), and the abstract patterns beside them, lowered here from the recorded source, answers as
/// the `regex` module does: for each text and each of its prefixes, whether it matches, where the
/// match ends, what it captures, and which of a value's ends closed it.
#[test]
fn lowered_call_patterns_answer_as_the_regex_module_does() {
    let word = escape("\\w").class();
    let (mut lowered, mut tags) = (0, 0);
    let mut answers = 0;
    for line in shape_fixture("regex.jsonl").lines() {
        let recorded: Recorded = serde_json::from_str(line).unwrap();
        let (pattern, ends) = lowered_recording(&recorded);
        assert_eq!(pattern.check(), Ok(()), "{}", recorded.id);
        for end in &ends {
            assert_eq!(end.check(), Ok(()), "{}", recorded.id);
        }
        let before = recorded.before.as_deref().and_then(|b| b.chars().next());
        lowered += 1;
        tags += usize::from(!ends.is_empty());
        for case in &recorded.cases {
            let text = case.text.as_str();
            assert_eq!(
                answer(text, (&pattern, &ends), &word, before),
                case.answer,
                "{}: {text:?}",
                recorded.id
            );
            answers += 1;
            let Some(prefixes) = &case.prefixes else {
                continue;
            };
            let upto: Vec<usize> = text
                .char_indices()
                .map(|(at, _)| at)
                .chain([text.len()])
                .collect();
            for (&upto, expected) in upto.iter().zip(prefixes) {
                assert_eq!(
                    &answer(&text[..upto], (&pattern, &ends), &word, before),
                    expected,
                    "{}: {:?} as the whole text",
                    recorded.id,
                    &text[..upto]
                );
                answers += 1;
            }
        }
    }
    assert!(lowered >= 24, "{lowered} patterns lowered");
    assert!(tags >= 6, "{tags} tags lowered");
    assert!(answers > 1000, "{answers} answers");
}

/// A tag pattern lowers into the tag through its value and the ways the value ends, in the order
/// the module tries them, each split at its lookahead; what a tag cannot be is refused by name.
#[test]
fn a_tag_lowers_through_its_value_and_its_ends_split_at_their_lookahead() {
    let tag = |source: &str| {
        let regex = Regex::parse(source).unwrap();
        let captures = captures(&regex);
        let value = regex
            .names
            .iter()
            .find(|(name, _)| name == "value")
            .map(|(_, number)| *number)
            .unwrap();
        let (tag, refused) = lower_tag(&regex, &captures, value);
        let refused: Vec<Construct> = refused.into_iter().map(|r| r.0).collect();
        (tag, refused)
    };
    let lowered = |source: &str| {
        let (tag, refused) = tag(source);
        assert_eq!(refused, [], "{source}");
        tag.unwrap()
    };
    let space = escape("\\s");
    let slack = Item::Run {
        chars: space,
        min: 0,
        max: None,
        greedy: true,
        capture: None,
    };
    let trimmed = lowered(r"<k=(?P<key>\w+)>\s*(?P<value>.*?)\s*(?:</k>|(?=<k=|<m>))");
    assert_eq!(trimmed.tag.len(), 5);
    assert_eq!(trimmed.tag[0], lit("<k="));
    assert!(matches!(
        trimmed.tag[4],
        Item::Run {
            capture: Some(Slot::Value),
            greedy: false,
            ..
        }
    ));
    assert_eq!(
        trimmed.ends,
        [
            EndItems {
                consume: vec![slack.clone(), lit("</k>")],
                ahead: Vec::new(),
            },
            EndItems {
                consume: vec![slack.clone()],
                ahead: vec![lit("<k=")],
            },
            EndItems {
                consume: vec![slack],
                ahead: vec![lit("<m>")],
            },
        ],
        "a lookahead's alternatives are ends of their own, after what the match takes"
    );
    // Groups that capture nothing around the value flatten; literals merge across them.
    let grouped = lowered(r"(?:<k (?P<key>\w+)>)(?:(?P<value>.*?))(?:</k>)\n");
    assert_eq!(grouped.tag[0], lit("<k "));
    assert_eq!(
        grouped.ends,
        [EndItems {
            consume: vec![lit("</k>\n")],
            ahead: Vec::new(),
        }]
    );
    // A value with nothing after it has an empty end, which no stream can run.
    let open = lowered(r"<k=(?P<key>\w+)>(?P<value>.*?)");
    assert_eq!(open.ends, [EndItems::default()]);
    assert!(ends(&open)[0].check().is_err());
    // A way the lowering refuses is left out of the ends; the others stay.
    let (dollar, refused) = tag(r"<k=(?P<key>\w+)>(?P<value>.*?)(?:</k>|(?=<k=|$))");
    assert_eq!(refused, [Construct::EndOfText]);
    assert_eq!(dollar.unwrap().ends.len(), 2);
    let refused = |source: &str| -> Vec<Construct> {
        let (_, refused) = tag(source);
        assert!(!refused.is_empty(), "{source} lowered");
        refused
    };
    for (source, construct) in [
        (
            r"<k=(?P<key>\w+)>(?:a(?P<value>.*?)|b)</k>",
            Construct::TagPattern,
        ),
        (
            r"(?:<k=|<j=)(?P<key>\w+)>(?P<value>.*?)</k>",
            Construct::TagPattern,
        ),
        (
            r"<k=(?P<key>\w+)>(?P<value>.*?)(?=<k=)(?=<k)",
            Construct::LookAhead,
        ),
        (
            r"<k=(?P<key>\w+)>(?P<value>.*?)(?=<k=)x",
            Construct::LookAhead,
        ),
        (
            r"<k=(?P<key>\w+)(?=>)>(?P<value>.*?)</k>",
            Construct::LookAhead,
        ),
        (
            r"<k=(?P<key>\w+)>(?P<value>.*?)(?:</k>|(?!<k=))",
            Construct::NegativeLookAround,
        ),
        (
            r"(?i)<k=(?P<key>\w+)>(?P<value>.*?)</k>",
            Construct::InlineFlags,
        ),
    ] {
        assert!(
            refused(source).contains(&construct),
            "{source}: {:?}",
            refused(source)
        );
    }
}

fn lit(text: &str) -> Item {
    Item::Lit(text.to_owned())
}

fn lowered(source: &str, names: &[(&str, Slot)]) -> Branches {
    let regex = Regex::parse(source).unwrap();
    let captures = Captures(
        regex
            .names
            .iter()
            .filter_map(|(name, number)| {
                names
                    .iter()
                    .find(|(wanted, _)| wanted == name)
                    .map(|(_, slot)| (*number, *slot))
            })
            .collect(),
    );
    lower(&regex, &captures).unwrap_or_else(|refused| panic!("{source}: {refused:?}"))
}

fn refused(source: &str) -> Vec<Construct> {
    let regex = Regex::parse(source).unwrap();
    match lower(&regex, &Captures::default()) {
        Ok(branches) => panic!("{source} lowered into {branches:?}"),
        Err(refused) => refused.into_iter().map(|r| r.0).collect(),
    }
}

#[test]
fn a_pattern_lowers_node_by_node() {
    let branches = lowered(r"\s*<function=(?P<name>[^>\n]+)>", &[("name", Slot::Name)]);
    let [branch] = branches.as_slice() else {
        panic!("{branches:?}");
    };
    assert_eq!(branch.len(), 4);
    assert!(branch[0].is_slack());
    assert_eq!(branch[1], lit("<function="));
    assert!(matches!(
        &branch[2],
        Item::Run { min: 1, max: None, greedy: true, capture: Some(Slot::Name), chars }
            if !chars.contains('>') && !chars.contains('\n') && chars.contains('f')
    ));
    assert_eq!(branch[3], lit(">"));
    // Escaped markers merge into one literal; a group nobody reads is flattened.
    assert_eq!(
        lowered(r"<\|channel\|>(?:analysis) <\|message\|>", &[]),
        [vec![lit("<|channel|>analysis <|message|>")]]
    );
    assert_eq!(
        lowered(r"a(?P<other>b)c", &[]),
        [vec![lit("abc")]],
        "a named group the transform does not read"
    );
    // An optional literal, and an optional alternation spelled out greedily.
    assert_eq!(
        lowered(r"<a>(?:\n\n)?", &[]),
        [vec![lit("<a>"), Item::Maybe("\n\n".to_owned())]]
    );
    assert_eq!(
        lowered(r"<a>(?:<b>|<c>)?", &[]),
        [vec![lit("<a><b>")], vec![lit("<a><c>")], vec![lit("<a>")]]
    );
    // An alternation in the middle spells each way through it, in the order the module tries
    // them.
    assert_eq!(
        lowered(r"x(?:1|2)y(?:3|4)", &[]),
        [
            vec![lit("x1y3")],
            vec![lit("x1y4")],
            vec![lit("x2y3")],
            vec![lit("x2y4")],
        ]
    );
    // Counted and lazy runs, and a boundary.
    let branches = lowered(r"\b\w{2,5}?.*?", &[]);
    assert!(matches!(
        branches[0].as_slice(),
        [
            Item::Boundary,
            Item::Run {
                min: 2,
                max: Some(5),
                greedy: false,
                ..
            },
            Item::Run {
                min: 0,
                max: None,
                greedy: false,
                ..
            },
        ]
    ));
}

#[test]
fn every_construct_no_table_reads_is_named() {
    for (source, construct) in [
        (r"(?i)<a>", Construct::InlineFlags),
        (r"(a)\1", Construct::Backreference),
        (r"(?<=a)b", Construct::LookBehind),
        (r"a(?!b)", Construct::NegativeLookAround),
        (r"a(?=b)", Construct::LookAhead),
        (r"^a", Construct::StartOfText),
        (r"a|$", Construct::EndOfText),
        (r"a\B", Construct::NotWordBoundary),
        (r"(?:ab)+", Construct::NestedRepeat),
        (r"(?:a(?:b)*)*", Construct::NestedRepeat),
        (r"(?:ab)??", Construct::NestedRepeat),
        (r"a*+", Construct::Possessive),
        (r"a{1,70000}", Construct::LongRepeat),
        (r"[[:alpha:]]", Construct::Property),
        (r"(?>ab)", Construct::OtherRegex),
        (
            r"(?:1|2)(?:1|2)(?:1|2)(?:1|2)(?:1|2)(?:1|2)",
            Construct::Alternation,
        ),
    ] {
        assert!(
            refused(source).contains(&construct),
            "{source}: {:?}",
            refused(source)
        );
    }
    // Every construct of a pattern is named, not only the first.
    assert_eq!(
        refused(r"(a)\1(?<=b)c\B"),
        [
            Construct::Backreference,
            Construct::LookBehind,
            Construct::NotWordBoundary
        ]
    );
    // A group the transform reads must be one run.
    let regex = Regex::parse(r"(?P<name>a|b)").unwrap();
    let names = Captures(vec![(1, Slot::Name)]);
    let refused: Vec<Construct> = lower(&regex, &names)
        .unwrap_err()
        .into_iter()
        .map(|r| r.0)
        .collect();
    assert_eq!(refused, [Construct::Capture]);
}

#[test]
fn an_anchor_is_a_finite_set_of_literals_in_the_order_the_module_tries_them() {
    let anchors = |source: &str| anchor_literals(&lowered(source, &[]));
    assert_eq!(
        anchors(r"<\|im_start\|>assistant\n<think>(?:\n|</think>)?").unwrap(),
        [
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think></think>",
            "<|im_start|>assistant\n<think>",
        ]
    );
    assert_eq!(
        anchors(r"(?:<\|turn>model\n|<tool_response\|>)").unwrap(),
        ["<|turn>model\n", "<tool_response|>"]
    );
    assert_eq!(
        anchors(r"<a>(?:b)?(?:c)?").unwrap(),
        ["<a>bc", "<a>b", "<a>c", "<a>"]
    );
    for source in [r"<a>\s*", r"<a>\w+", r"\b<a>"] {
        assert_eq!(
            anchors(source).unwrap_err().0,
            Construct::AnchorNotFinite,
            "{source}"
        );
    }
    assert_eq!(
        anchors(r"<x>(?:a)?(?:b)?(?:c)?(?:d)?(?:e)?(?:f)?")
            .unwrap_err()
            .0,
        Construct::AnchorNotFinite,
        "more than 32 literals"
    );
}

#[test]
fn a_delimiter_splits_at_its_marker() {
    let open = split_open(
        &lowered(
            r"\s*<tool_call>\s*<function=(?P<name>\w+)>",
            &[("name", Slot::Name)],
        )[0],
    )
    .unwrap();
    assert_eq!(open.slack.len(), 1);
    assert_eq!(open.marker, "<tool_call>");
    assert_eq!(open.header.len(), 4);
    let close = split_close(&lowered(r"</function>\s*</tool_call>\s*", &[])[0]).unwrap();
    assert_eq!(close.marker, "</tool_call>");
    assert_eq!(close.trailer.len(), 2);
    assert_eq!(close.slack.len(), 1);
    assert!(close.slack.iter().all(Item::is_slack));
    // A marker is a literal: a filler before an open's, or a run after a close's, is none.
    assert_eq!(
        split_open(&lowered(r"[^<]*<a>", &[])[0]).unwrap_err().0,
        Construct::NoLiteral
    );
    assert_eq!(
        split_close(&lowered(r"</a>\w+", &[])[0]).unwrap_err().0,
        Construct::NoLiteral
    );
    // Whitespace that may be absent is slack; a run that must take one character is not, nor is
    // a run of a class that holds more than whitespace.
    for (source, slack) in [
        (r"\s*", true),
        (r"\n?", true),
        (r"[ \t]*", true),
        (r"(?:\n\n)?", true),
        (r"\s+", false),
        (r"[\s,]*", false),
        (r"\s*?", false),
    ] {
        let branch = &lowered(source, &[])[0];
        assert_eq!(branch.iter().all(Item::is_slack), slack, "{source}");
    }
    // Python's `str.strip` whitespace, which `\s` of the module leaves out, is slack too.
    assert!(Item::Run {
        chars: Chars::new(vec![('\u{1c}', '\u{1f}')], false),
        min: 0,
        max: None,
        greedy: true,
        capture: None,
    }
    .is_slack());
    assert_eq!(pattern(&[lit("a")]).0.as_ref(), [Atom::Lit("a".into())]);
}

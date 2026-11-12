use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::{
    format::Whitespace,
    formats::Table,
    grammar::{
        spelling::tests::{every_grammar, tools},
        Bound, GrammarStart, Options, Tag,
    },
};

fn text(value: &str) -> Grammar {
    Grammar::ConstString(value.to_string())
}

fn any(excludes: &[&str]) -> Grammar {
    Grammar::AnyText {
        excludes: excludes.iter().map(|e| e.to_string()).collect(),
    }
}

fn tag(begin: &str, content: Grammar, end: Bound) -> Tag {
    Tag {
        begin: begin.to_string(),
        content: Box::new(content),
        end,
    }
}

fn ends(text: &str) -> Bound {
    Bound::Text(text.to_string())
}

fn schema(schema: Value, excludes: &[&str], whitespace: Whitespace) -> Grammar {
    Grammar::JsonSchema {
        schema: Arc::new(schema),
        style: None,
        excludes: excludes.iter().map(|e| e.to_string()).collect(),
        whitespace,
    }
}

fn tokens(pairs: &[(&str, u32)], eos: Option<u32>) -> AddedTokens {
    AddedTokens::new(pairs.iter().map(|(text, id)| (text.to_string(), *id)), eos)
}

/// The spellings below are the reference texts of the design's lowering, each read through the
/// backend's own compiler on the recorded outputs of the tables they stand for.
const PLAIN_TAG: &str = r#"%llguidance {}
start: ("<a>" h1 "")
l2[lazy]: T1 "</a>"
h1: l2
T1: /(?s:.*)/ & ~/(?s:.*)(?:<\/a>)(?s:.*)/
"#;

const TRIGGERED_CALL: &str = r#"%llguidance {}
start: ((t6)* x1)
x1: T1 | T1 <[151658,151667-151668]> x1
l3[lazy]: T1 "<tool_call>"
k4: T1 <tool_call>
h2: l3 | k4 | T1 <[151658,151667-151668]> h2
f5: (<tool_call> | "<tool_call>") "\n" ("{\"name\": \"get_weather\", \"arguments\": " %json {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]} "}") "\n" (</tool_call> | "</tool_call>")
t6: h2 "\n" ("{\"name\": \"get_weather\", \"arguments\": " %json {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]} "}") "\n" (</tool_call> | "</tool_call>")
T1: /(?s:.*)/
"#;

const SEPARATED_CALLS: &str = r#"%llguidance {}
start: (((("<think>" h1) | "</think>"))? (s5 ((<|sep|> | "<|sep|>") s5)*) x6)
l2[lazy]: T1 "</think>"
h1: l2 | T1 <[7-9]> h1
k4: T2 <|b|>
h3: k4 | T3 <|sep|> h3
s5: (((<|a|> | "<|a|>") "call" (<|sep|> | "<|sep|>") ("name=" /[a-z_\/]+/ (((<|b|> | "<|b|>") h3 ""))* %json {"x-guidance":{"whitespace_pattern":"[\\x20\\x0A\\x0D\\x09]{0,4}"}} %json {"x-guidance":{"whitespace_pattern":"[\\x20\\x0A\\x0D\\x09]{0,16}"},"type":"object","properties":{"n":{"type":"integer"}}}) ((<|a|> | "<|a|>") "end" (<|sep|> | "<|sep|>") | (<|im_end|> | "<|im_end|>"))))
x6: T4 | T4 <[7-9]> x6
T1: /(?s:.*)/ & ~/(?s:.*)(?:<\/think>)(?s:.*)/
T2: /(?s:.*)/ & ~/(?s:.*)(?:<\|b\|>|<\|a\|>)(?s:.*)/
T3: /(?s:.*)/ & ~/(?s:.*)(?:<\|b\|>|<\|a\|>)(?s:.*)/ & ~/(?s:.*)(?:<\|a\|>end)/
T4: /(?s:.*)/
"#;
const COMPOSITE_TRIGGER: &str = r#"%llguidance {}
start: ((t12) | x1)
x1: T1 | T1 <|open|> x2 | T1 <|sep|> x1
x2: T2 | T2 <|open|> x2 | T3 <|sep|> x1
l5[lazy]: T1 "<|open|>tools<|sep|>"
k6: T1 <|open|> "tools" (<|sep|> | "<|sep|>")
h3: l5 | k6 | T1 <|open|> h4 | T1 <|sep|> h3
l7[lazy]: T2 "<|open|>tools<|sep|>"
k8: T2 <|open|> "tools" (<|sep|> | "<|sep|>")
h4: l7 | k8 | T2 <|open|> h4 | T3 <|sep|> h3
l10[lazy]: T4 "\"}"
h9: l10 | T5 <|sep|> h9
f11: (<|open|> | "<|open|>") "tools" (<|sep|> | "<|sep|>") ("{\"name\":\"" h9 "") (<|open|> | "<|open|>") "end" (<|sep|> | "<|sep|>")
t12: h3 ("{\"name\":\"" h9 "") (<|open|> | "<|open|>") "end" (<|sep|> | "<|sep|>")
T1: /(?s:.*)/
T2: /(?s:.*)/ & ~/(?:tools<\|sep\|>)(?s:.*)/
T3: /(?s:.*)/ & ~/(?:tools<\|sep\|>)(?s:.*)/ & ~/(?:tools)/
T4: /(?s:.*)/ & ~/(?s:.*)(?:"|<\|open\|>)(?s:.*)/
T5: /(?s:.*)/ & ~/(?s:.*)(?:"|<\|open\|>)(?s:.*)/ & ~/(?s:.*)(?:<\|open\|>end)/
"#;

/// A token node after free text: the text ends at the token alone, with no byte spelling of it.
const TOKEN_AFTER_TEXT: &str = r#"%llguidance {}
start: ("[" h1 "" "]")
k2: T1 <X>
h1: k2
T1: /(?s:.*)/
"#;

/// A word around a token node: the bytes before the token end the text's lexeme, the token is
/// required, the bytes after it stay bytes.
const TOKEN_INSIDE_WORD: &str = r#"%llguidance {}
start: (h1 ())
k2: T2 <X> "b"
h1: k2
T1: /(?s:.*)/
T2: /(?s:.*)a/
"#;

/// `<X>` begins the forbidden word `<X>b` and ends `a<X>`: every state of the text names it in
/// one transition, after a text not ending with `a` and into a text not starting with `b`.
const PREFIX_AND_SUFFIX: &str = r#"%llguidance {}
start: ("[" h1 "")
l3[lazy]: T1 "]"
h1: l3 | T2 <X> h2
l4[lazy]: T3 "]"
h2: l4 | T4 <X> h2
T1: /(?s:.*)/ & ~/(?s:.*)(?:a<X>|<X>b)(?s:.*)/
T2: /(?s:.*)/ & ~/(?s:.*)(?:a<X>|<X>b)(?s:.*)/ & ~/(?s:.*)(?:a)/
T3: /(?s:.*)/ & ~/(?s:.*)(?:a<X>|<X>b)(?s:.*)/ & ~/(?:b)(?s:.*)/
T4: /(?s:.*)/ & ~/(?s:.*)(?:a<X>|<X>b)(?s:.*)/ & ~/(?:b)(?s:.*)/ & ~/(?s:.*)(?:a)/
"#;

/// `<|x|>` stands inside `a<|x|>b` between bytes, so the text leaves it out; `<|x|>q<|y|>` is
/// then read as bytes up to `<|y|>`, which follows a text not ending with them, in the start
/// state: the left-out token names no state.
const LEFT_OUT_TOKEN: &str = r#"%llguidance {}
start: ("[" h1 "")
l2[lazy]: T1 "]"
h1: l2 | T2 <|y|> h1
T1: /(?s:.*)/ & ~/(?s:.*)(?:a<\|x\|>b|<\|x\|>q<\|y\|>)(?s:.*)/
T2: /(?s:.*)/ & ~/(?s:.*)(?:a<\|x\|>b|<\|x\|>q<\|y\|>)(?s:.*)/ & ~/(?s:.*)(?:<\|x\|>q)/
"#;

#[test]
fn free_text_before_a_closing_word_ends_at_the_word_through_a_lazy_head() {
    let grammar = Grammar::Sequence(vec![text("<a>"), any(&["</a>"]), text("</a>")]);
    let lark = grammar
        .to_lark(&AddedTokens::default())
        .expect("a tag around free text is spelled");
    assert_eq!(lark.text, PLAIN_TAG);
    assert!(lark.notes.is_empty(), "{:?}", lark.notes);
    well_formed(&lark.text).expect("the reference text reads");
}

#[test]
fn a_triggered_call_names_the_tokens_its_free_text_may_hold_and_the_trigger_token() {
    let grammar = Grammar::TriggeredTags {
        triggers: vec!["<tool_call>".to_string()],
        tags: vec![tag(
            "<tool_call>\n",
            Grammar::Sequence(vec![
                text("{\"name\": \"get_weather\", \"arguments\": "),
                schema(
                    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
                    &["</tool_call>"],
                    Whitespace::Free,
                ),
                text("}"),
            ]),
            ends("\n</tool_call>"),
        )],
        at_least_one: false,
        stop_after_first: false,
    };
    let vocabulary = tokens(
        &[
            ("<tool_call>", 151657),
            ("</tool_call>", 151658),
            ("<think>", 151667),
            ("</think>", 151668),
            ("<|im_end|>", 151645),
        ],
        Some(151645),
    );
    let lark = grammar
        .to_lark(&vocabulary)
        .expect("a triggered call is spelled");
    assert_eq!(lark.text, TRIGGERED_CALL);
    // the schema's excluded words are a loosening the spelling names
    assert_eq!(lark.notes.len(), 1, "{:?}", lark.notes);
    assert!(lark.notes[0].contains("excluded words"), "{:?}", lark.notes);
}

#[test]
fn separated_calls_carry_pieces_bounds_choices_repeats_tokens_and_regexes() {
    let grammar = Grammar::Sequence(vec![
        Grammar::Optional(Box::new(Grammar::Or(vec![
            Grammar::Tag(tag("<think>", any(&["</think>"]), ends("</think>"))),
            text("</think>"),
        ]))),
        Grammar::TagsWithSeparator {
            tags: vec![tag(
                "<|a|>call<|sep|>",
                Grammar::Sequence(vec![
                    text("name="),
                    Grammar::Regex("[a-z_/]+".to_string()),
                    Grammar::Star(Box::new(Grammar::Sequence(vec![
                        text("<|b|>"),
                        any(&["<|b|>", "<|a|>"]),
                        Grammar::Token("<|b|>".to_string()),
                    ]))),
                    schema(json!(true), &[], Whitespace::Bounded(4)),
                    schema(
                        json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
                        &[],
                        Whitespace::Bounded(16),
                    ),
                ]),
                Bound::Texts(vec![
                    "<|a|>end<|sep|>".to_string(),
                    "<|im_end|>".to_string(),
                ]),
            )],
            separator: "<|sep|>".to_string(),
            at_least_one: true,
            stop_after_first: false,
        },
        any(&[]),
    ]);
    let vocabulary = tokens(
        &[
            ("<|a|>", 7),
            ("<|b|>", 8),
            ("<|sep|>", 9),
            ("<|im_end|>", 2),
        ],
        Some(2),
    );
    let lark = grammar
        .to_lark(&vocabulary)
        .expect("separated calls are spelled");
    assert_eq!(lark.text, SEPARATED_CALLS);
    assert!(lark.notes.is_empty(), "{:?}", lark.notes);
}

#[test]
fn a_trigger_spelled_across_tokens_constrains_the_text_around_each_piece() {
    let grammar = Grammar::TriggeredTags {
        triggers: vec!["<|open|>tools<|sep|>".to_string()],
        tags: vec![tag(
            "<|open|>tools<|sep|>",
            Grammar::Sequence(vec![
                text("{\"name\":\""),
                any(&["\"", "<|open|>"]),
                text("\"}"),
            ]),
            ends("<|open|>end<|sep|>"),
        )],
        at_least_one: false,
        stop_after_first: true,
    };
    let vocabulary = tokens(
        &[("<|open|>", 11), ("<|sep|>", 12), ("<|im_end|>", 2)],
        Some(2),
    );
    let lark = grammar
        .to_lark(&vocabulary)
        .expect("a composite trigger is spelled");
    assert_eq!(lark.text, COMPOSITE_TRIGGER);
}

#[test]
fn a_token_is_referred_to_by_name_or_by_id() {
    let vocabulary = tokens(&[("<x>", 5), ("[CALL]", 6), ("<x>", 7), ("", 8)], None);
    assert_eq!(vocabulary.len(), 2);
    assert_eq!(
        vocabulary.id("<x>"),
        Some(5),
        "a repeated text keeps its first id"
    );
    assert_eq!(vocabulary.id(""), None, "an empty text is no token");
    let by_name = Grammar::Token("<x>".to_string())
        .to_lark(&vocabulary)
        .expect("spelled");
    assert_eq!(by_name.text, "%llguidance {}\nstart: <x>\n");
    let by_id = Grammar::Token("[CALL]".to_string())
        .to_lark(&vocabulary)
        .expect("spelled");
    assert_eq!(by_id.text, "%llguidance {}\nstart: <[6]>\n");
    let word = Grammar::ConstString("a[CALL]b".to_string())
        .to_lark(&vocabulary)
        .expect("spelled");
    assert_eq!(
        word.text,
        "%llguidance {}\nstart: \"a\" (<[6]> | \"[CALL]\") \"b\"\n"
    );
}

#[test]
fn what_the_lark_cannot_carry_is_refused_by_name() {
    let refused = |grammar: Grammar| {
        grammar
            .to_lark(&tokens(&[("<x>", 5)], None))
            .expect_err("refused")
            .to_string()
    };
    let style = refused(Grammar::JsonSchema {
        schema: Arc::new(json!({"type": "object"})),
        style: Some("glm_xml".to_string()),
        excludes: Vec::new(),
        whitespace: Whitespace::Free,
    });
    assert!(style.contains("glm_xml"), "{style}");
    let nothing = refused(schema(json!(false), &[], Whitespace::Free));
    assert!(nothing.contains("accepts nothing"), "{nothing}");
    let ebnf = refused(Grammar::Ebnf("root ::= \"y\"".to_string()));
    assert!(ebnf.contains("EBNF"), "{ebnf}");
    let missing = refused(Grammar::Token("<y>".to_string()));
    assert!(missing.contains("<y>"), "{missing}");
    let inside = refused(Grammar::Sequence(vec![
        text("<a>"),
        Grammar::Or(vec![Grammar::Token("<y>".to_string()), text("b")]),
    ]));
    assert!(inside.contains("<y>"), "{inside}");
}

#[test]
fn a_token_node_after_free_text_is_reached_by_the_token_alone() {
    let vocabulary = tokens(&[("<X>", 5)], None);
    let lark = Grammar::Sequence(vec![
        text("["),
        any(&[]),
        Grammar::Token("<X>".to_string()),
        text("]"),
    ])
    .to_lark(&vocabulary)
    .expect("a token after free text is spelled");
    assert_eq!(lark.text, TOKEN_AFTER_TEXT);
    well_formed(&lark.text).expect("reads");
    // bytes around the token node stay bytes; the token alone stands between them
    let lark = Grammar::Sequence(vec![
        any(&[]),
        Grammar::Sequence(vec![
            text("a"),
            Grammar::Token("<X>".to_string()),
            text("b"),
        ]),
    ])
    .to_lark(&vocabulary)
    .expect("a word around a token node is spelled");
    assert_eq!(lark.text, TOKEN_INSIDE_WORD);
    well_formed(&lark.text).expect("reads");
    // a vocabulary without the token refuses the grammar, as it refuses the lone token
    for vocabulary in [AddedTokens::default(), tokens(&[("<Y>", 5)], None)] {
        let missing = Grammar::Sequence(vec![any(&[]), Grammar::Token("<X>".to_string())])
            .to_lark(&vocabulary)
            .expect_err("a token the vocabulary lacks is refused");
        assert!(missing.to_string().contains("<X>"), "{missing}");
    }
}

#[test]
fn a_token_that_begins_one_forbidden_word_and_ends_another_has_one_transition_per_state() {
    let lark = Grammar::Sequence(vec![text("["), any(&["a<X>", "<X>b"]), text("]")])
        .to_lark(&tokens(&[("<X>", 5)], None))
        .expect("spelled");
    assert_eq!(lark.text, PREFIX_AND_SUFFIX);
    well_formed(&lark.text).expect("reads");
    // every state of the text names the token once, in a transition that holds both sides
    for rule in lark.text.lines().filter(|line| line.starts_with('h')) {
        assert_eq!(rule.matches(" <X> ").count(), 1, "{rule}");
    }
}

#[test]
fn a_token_the_text_leaves_out_names_no_state() {
    let lark = Grammar::Sequence(vec![text("["), any(&["a<|x|>b", "<|x|>q<|y|>"]), text("]")])
        .to_lark(&tokens(&[("<|x|>", 7), ("<|y|>", 8)], None))
        .expect("spelled");
    assert_eq!(lark.text, LEFT_OUT_TOKEN);
    well_formed(&lark.text).expect("reads");
    assert!(!lark.text.contains("<|x|>"), "{}", lark.text);
}

#[test]
fn a_slash_in_a_regex_is_escaped_once() {
    let spelled = |pattern: &str| {
        Grammar::Regex(pattern.to_string())
            .to_lark(&AddedTokens::default())
            .expect("a regex is spelled")
            .text
    };
    assert_eq!(
        spelled(r"^https?:\/\/[a-z.]+$"),
        "%llguidance {}\nstart: /^https?:\\/\\/[a-z.]+$/\n"
    );
    assert_eq!(
        spelled(r"a/b\\/c"),
        "%llguidance {}\nstart: /a\\/b\\\\\\/c/\n"
    );
    assert_eq!(spelled(""), "%llguidance {}\nstart: \"\"\n");
}

#[test]
fn the_note_on_a_schema_s_excluded_words_is_given_once() {
    let call = |value: Value| {
        tag(
            "<t>",
            schema(value, &["</t>"], Whitespace::Free),
            ends("</t>"),
        )
    };
    let grammar = Grammar::TriggeredTags {
        triggers: vec!["<t>".to_string()],
        tags: vec![
            call(json!({"type": "object"})),
            call(json!({"type": "string"})),
        ],
        at_least_one: false,
        stop_after_first: false,
    };
    let lark = grammar
        .to_lark(&tokens(&[("<t>", 3), ("</t>", 4)], None))
        .expect("spelled");
    well_formed(&lark.text).expect("reads");
    assert_eq!(lark.notes.len(), 1, "{:?}", lark.notes);
}

#[test]
fn the_structural_reading_accepts_the_references_and_refuses_broken_texts() {
    for text in [
        PLAIN_TAG,
        TRIGGERED_CALL,
        SEPARATED_CALLS,
        COMPOSITE_TRIGGER,
    ] {
        well_formed(text).unwrap_or_else(|why| panic!("{why}: {text}"));
    }
    let broken = [
        ("%llguidance {}\nstart: x1\n", "x1 is named and not defined"),
        ("%llguidance {}\nstart: (\"a\"\n", "does not close"),
        ("%llguidance {}\nstart: \"a\")\n", "closes nothing"),
        ("%llguidance {}\nstart: /a\n", "regex does not close"),
        ("%llguidance {}\nstart: \"a\n", "literal does not close"),
        ("%llguidance {}\nstart: <[1-2\n", "does not close"),
        ("%llguidance {}\nstart: %json {\"a\":\n", "does not parse"),
        (
            "%llguidance {}\nstart: \"a\"\nstart: \"b\"\n",
            "more than once",
        ),
        ("%llguidance {}\nx1: \"a\"\n", "no start rule"),
        ("start: \"a\"\n", "header"),
    ];
    for (text, why) in broken {
        let read = well_formed(text).expect_err(text);
        assert!(read.contains(why), "{text:?}: {read}");
    }
}

#[test]
fn every_registry_grammar_is_spelled_or_names_its_schema_style() {
    let grammars = every_grammar();
    let mut spelled = 0;
    let mut refused = 0;
    for (name, grammar) in &grammars {
        match grammar.to_lark(&AddedTokens::default()) {
            Ok(lark) => {
                well_formed(&lark.text).unwrap_or_else(|why| panic!("{name}: {why}"));
                spelled += 1;
            }
            Err(why) => {
                assert!(why.to_string().contains("style"), "{name}: {why}");
                assert!(
                    grammar.payload_text().contains("\"style\":\"glm_xml\""),
                    "{name}: refused without the style in the tag: {why}"
                );
                refused += 1;
            }
        }
    }
    assert!(
        spelled > grammars.len() * 9 / 10,
        "{spelled} spelled, {refused} refused of {}",
        grammars.len()
    );
    assert!(refused > 0, "the XML schema styles are refused by name");
}

/// The tables' grammars with the markers spelled as tokens ([`crate::Format::marker_tokens`]),
/// in a vocabulary that has every terminal as an added token: the spelling the gateway sends a
/// worker whose vocabulary has them.
#[test]
fn every_tokened_table_grammar_is_spelled_in_its_vocabulary() {
    let tools = tools();
    let answer = Arc::new(json!({"type": "object", "properties": {"answer": {"type": "string"}}}));
    let mut spelled = 0;
    for table in Table::ALL {
        let shared = table.shared();
        let terminals: Vec<String> = shared.terminal_texts().map(str::to_string).collect();
        let vocabulary =
            AddedTokens::new(terminals.iter().cloned().zip(200_000u32..), Some(199_999));
        let format = (*shared).clone().marker_tokens(terminals.iter());
        let mut grammars = Vec::new();
        for (at_least_one, one_call) in [(true, false), (false, false)] {
            for reasoning_open in [false, true] {
                let options = Options {
                    at_least_one,
                    reasoning_open,
                    thought_closed: false,
                    one_call,
                    start: GrammarStart::Unknown,
                };
                grammars.extend(format.grammar_for(&tools, options));
            }
        }
        grammars.extend(format.answer_grammar_for(&answer, false, GrammarStart::Unknown));
        for grammar in grammars {
            match grammar.to_lark(&vocabulary) {
                Ok(lark) => {
                    well_formed(&lark.text).unwrap_or_else(|why| panic!("{table:?}: {why}"));
                    spelled += 1;
                }
                Err(why) => assert!(why.to_string().contains("style"), "{table:?}: {why}"),
            }
        }
    }
    assert!(spelled > Table::ALL.len() * 3, "{spelled} spelled");
}

/// A reading of a Lark text's structure, backend-independent, for the sweeps: the header, then
/// one definition per line, `name: body` for a rule (`name[lazy]: body` for a lazy one) or
/// `T<n>: body` for a terminal; every rule and terminal a body names is defined exactly once;
/// parentheses balance outside literals and regexes; every literal, regex, token reference and
/// `%json` value closes; `start` is defined.
fn well_formed(text: &str) -> Result<(), String> {
    let mut lines = text.lines();
    if lines.next() != Some("%llguidance {}") {
        return Err("the header is missing".to_string());
    }
    let mut defined: BTreeMap<&str, usize> = BTreeMap::new();
    let mut named: BTreeSet<&str> = BTreeSet::new();
    for line in lines {
        let (head, body) = line
            .split_once(": ")
            .ok_or_else(|| format!("no definition on the line {line:?}"))?;
        let name = match head.split_once('[') {
            Some((name, "lazy]")) => name,
            Some(_) => return Err(format!("an unknown option on {head:?}")),
            None => head,
        };
        if !is_name(name) {
            return Err(format!("not a rule or terminal name: {head:?}"));
        }
        *defined.entry(name).or_default() += 1;
        read_body(body, &mut named).map_err(|why| format!("{name}: {why} in {body:?}"))?;
    }
    if let Some((name, _)) = defined.iter().find(|(_, count)| **count > 1) {
        return Err(format!("{name} is defined more than once"));
    }
    if !defined.contains_key("start") {
        return Err("no start rule".to_string());
    }
    if let Some(name) = named.iter().find(|name| !defined.contains_key(*name)) {
        return Err(format!("{name} is named and not defined"));
    }
    Ok(())
}

/// A rule name (`[a-z][a-z0-9_]*`) or a terminal name (`T<n>`).
fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some('T') => name.len() > 1 && chars.all(|c| c.is_ascii_digit()),
        Some(c) if c.is_ascii_lowercase() => {
            chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        }
        _ => false,
    }
}

/// The names a body refers to collected into `named`; an error names what does not close or
/// balance.
fn read_body<'a>(body: &'a str, named: &mut BTreeSet<&'a str>) -> Result<(), String> {
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'|' | b'?' | b'*' | b'+' | b'&' | b'~' => i += 1,
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.checked_sub(1).ok_or("a parenthesis closes nothing")?;
                i += 1;
            }
            b'"' => i = closing(bytes, i + 1, b'"').ok_or("a literal does not close")? + 1,
            b'/' => i = closing(bytes, i + 1, b'/').ok_or("a regex does not close")? + 1,
            b'<' => {
                let end = body[i..]
                    .find('>')
                    .ok_or("a token reference does not close")?;
                if body[i..].starts_with("<[") && !body[i..i + end].ends_with(']') {
                    return Err("a token range does not close".to_string());
                }
                i += end + 1;
            }
            b'%' => {
                let rest = body[i..]
                    .strip_prefix("%json ")
                    .ok_or("an unknown directive")?;
                let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
                match values.next() {
                    Some(Ok(_)) => i += "%json ".len() + values.byte_offset(),
                    _ => return Err("a %json value does not parse".to_string()),
                }
            }
            c if c.is_ascii_lowercase() || c == b'T' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let name = &body[start..i];
                if !is_name(name) {
                    return Err(format!("not a name: {name:?}"));
                }
                named.insert(name);
            }
            other => return Err(format!("an unexpected character {:?}", other as char)),
        }
    }
    if depth != 0 {
        return Err("a parenthesis does not close".to_string());
    }
    Ok(())
}

/// The index of the `end` byte that closes a literal or regex opened before `from`, backslash
/// escapes skipped.
fn closing(bytes: &[u8], from: usize, end: u8) -> Option<usize> {
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            c if c == end => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// The end-of-sequence token (`<|im_end|>`) heads a forbidden word that goes on with bytes and
/// another token: it never stands in free text, so the word is read as bytes across it (the lead
/// `<|im_end|>\n` is held out before `<|sep|>` in every state) and it names no state.
const EOS_WORD: &str = r#"%llguidance {}
start: ("[" h1 "")
l2[lazy]: T1 "]"
h1: l2 | T2 <|sep|> h1
T1: /(?s:.*)/ & ~/(?s:.*)(?:<\|im_end\|>\n<\|sep\|>)(?s:.*)/
T2: /(?s:.*)/ & ~/(?s:.*)(?:<\|im_end\|>\n<\|sep\|>)(?s:.*)/ & ~/(?s:.*)(?:<\|im_end\|>\n)/
"#;

#[test]
fn the_end_of_sequence_token_never_stands_in_free_text() {
    let lark = Grammar::Sequence(vec![text("["), any(&["<|im_end|>\n<|sep|>"]), text("]")])
        .to_lark(&tokens(&[("<|sep|>", 12), ("<|im_end|>", 2)], Some(2)))
        .expect("spelled");
    assert_eq!(lark.text, EOS_WORD);
    well_formed(&lark.text).expect("reads");
    assert!(
        !lark.text.contains(" <|im_end|> "),
        "the eos token names no state and no transition: {}",
        lark.text
    );
}

/// Free text that may not hold `a`, then the word `a<X>` with a token node: the head's lead lexeme
/// holds `a` out of the text BEFORE the lead only (`& ~/(?s:.*)(?:a)(?s:.*)a/`), never out of the
/// lead itself; the empty text followed by `a` and the token is valid, and a text ending in `a`
/// before the lead is not.
const LEAD_WITH_EXCLUDE: &str = r#"%llguidance {}
start: ("[" h1 () "]")
k2: T2 <X>
h1: k2 | T3 <X> h1
T1: /(?s:.*)/ & ~/(?s:.*)(?:a)(?s:.*)/
T2: /(?s:.*)a/ & ~/(?s:.*)(?:a)(?s:.*)a/
T3: /(?s:.*)/ & ~/(?s:.*)(?:a)(?s:.*)/ & ~/(?s:.*)(?:a)/
"#;

#[test]
fn an_excluded_word_is_held_out_of_the_text_before_a_lead_not_of_the_lead() {
    let lark = Grammar::Sequence(vec![
        text("["),
        any(&["a"]),
        Grammar::Sequence(vec![text("a"), Grammar::Token("<X>".to_string())]),
        text("]"),
    ])
    .to_lark(&tokens(&[("<X>", 5)], None))
    .expect("spelled");
    assert_eq!(lark.text, LEAD_WITH_EXCLUDE);
    well_formed(&lark.text).expect("reads");
    // the lead lexeme is satisfiable: `a` alone (the empty text) matches it
    let lead = lark
        .text
        .lines()
        .find(|line| line.starts_with("T2:"))
        .expect("the lead lexeme");
    assert!(lead.ends_with("(?s:.*)a/"), "{lead}");
}

/// Free text that may not hold `<X>ab` (a word a token begins), then the word `b<Y>` (a byte lead
/// and a token): in the state after `<X>` the head's lead lexeme binds its start ban to the text
/// BEFORE the lead (`& ~/(?:ab)(?s:.*)b/`), as its exclusion ban does, so the text `a` followed by
/// the word is taken by the token path as it is by the byte path; the text `ab` is not.
const LEAD_WITH_START_BAN: &str = r#"%llguidance {}
start: (h1 "")
l3[lazy]: T1 "b<Y>"
k4: T2 <Y>
h1: l3 | k4 | T1 <X> h2 | T3 <Y> h1
l5[lazy]: T4 "b<Y>"
k6: T5 <Y>
h2: l5 | k6 | T4 <X> h2 | T6 <Y> h1
T1: /(?s:.*)/ & ~/(?s:.*)(?:<X>ab)(?s:.*)/
T2: /(?s:.*)b/ & ~/(?s:.*)(?:<X>ab)(?s:.*)b/
T3: /(?s:.*)/ & ~/(?s:.*)(?:<X>ab)(?s:.*)/ & ~/(?s:.*)(?:b)/
T4: /(?s:.*)/ & ~/(?s:.*)(?:<X>ab)(?s:.*)/ & ~/(?:ab)(?s:.*)/
T5: /(?s:.*)b/ & ~/(?s:.*)(?:<X>ab)(?s:.*)b/ & ~/(?:ab)(?s:.*)b/
T6: /(?s:.*)/ & ~/(?s:.*)(?:<X>ab)(?s:.*)/ & ~/(?:ab)(?s:.*)/ & ~/(?s:.*)(?:b)/
"#;

#[test]
fn a_start_ban_binds_the_text_before_a_lead_not_the_lead() {
    let lark = Grammar::Sequence(vec![any(&["<X>ab"]), text("b<Y>")])
        .to_lark(&tokens(&[("<X>", 5), ("<Y>", 6)], None))
        .expect("spelled");
    assert_eq!(lark.text, LEAD_WITH_START_BAN);
    well_formed(&lark.text).expect("reads");
    // the token path's lead lexeme in the state after <X> ends its start ban with the lead
    let lead = lark
        .text
        .lines()
        .find(|line| line.starts_with("T5:"))
        .expect("the lead lexeme of the <X> state");
    assert!(lead.ends_with("~/(?:ab)(?s:.*)b/"), "{lead}");
}

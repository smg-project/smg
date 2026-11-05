//! The grammar spelled in the older structural-tag vocabulary: without `optional`, `plus` and
//! `star`, which xgrammar reads from 0.1.33 on and an older grammar compiler refuses at compile
//! time ("Format type not recognized"), so the request fails on every forced call. The spelling
//! has the same language, written with the nodes every version reads:
//!
//! - `optional(X)` is `or(X, "")`; an optional list that may repeat (`Y (S Y)*`, below) is the
//!   list with `at_least_one` off;
//! - `star(X)` and `plus(X)` are a `tags_with_separator` with an empty separator, of `X` itself
//!   when it is a tag or a choice of tags, else of a tag with an empty begin and end around it;
//! - `Y star(S Y)` and `Y star(Y)`, the shapes the formats write for a separated list, are one
//!   `tags_with_separator` with `S` as the separator and `at_least_one` on.
//!
//! Two rules of the older analyzer bind where those lists may stand: a list, like free text, is
//! unlimited, and an unlimited element must end its sequence; so a sequence whose list is followed
//! by constant text folds that text into the end of a tag around the list, and the constants
//! before it into the tag's begin. A tag with an empty begin and end around a repeated item becomes
//! the nearest tag of the item's texts, whose stop strings the engines read from it, so the
//! enclosing tag's end joins those texts' excluded strings (it stopped them before). And the older
//! matcher cannot build the text of a tag whose excluded strings name the tag's own end (equal to
//! it, or extending it); where the text is the last thing before the end, those excludes are
//! redundant (the end closes the text before it could hold them) and are left out. Nothing else
//! changes: the tags, their begins and ends, the triggers, the JSON schemas, the regular
//! expressions and the single-token markers are written as derived.

use super::{Bound, Grammar, Tag};

impl Grammar {
    /// The grammar in the older vocabulary (the module documentation): the same language without
    /// `optional`, `plus` and `star`.
    #[must_use]
    pub fn in_older_vocabulary(self) -> Grammar {
        older(self, &[], true)
    }
}

/// `grammar` rewritten, under a tag whose ends are `ends` (none at the root); `tail` while the
/// node is the last thing before that end.
fn older(grammar: Grammar, ends: &[String], tail: bool) -> Grammar {
    match grammar {
        Grammar::AnyText { excludes } if tail && !ends.is_empty() => Grammar::AnyText {
            excludes: excludes
                .into_iter()
                .filter(|exclude| {
                    !ends
                        .iter()
                        .any(|end| !end.is_empty() && exclude.starts_with(end.as_str()))
                })
                .collect(),
        },
        Grammar::Tag(tag) => Grammar::Tag(in_tag(tag)),
        Grammar::TriggeredTags {
            triggers,
            tags,
            at_least_one,
            stop_after_first,
        } => Grammar::TriggeredTags {
            triggers,
            tags: tags.into_iter().map(in_tag).collect(),
            at_least_one,
            stop_after_first,
        },
        Grammar::TagsWithSeparator {
            tags,
            separator,
            at_least_one,
            stop_after_first,
        } => Grammar::TagsWithSeparator {
            tags: tags.into_iter().map(in_tag).collect(),
            separator,
            at_least_one,
            stop_after_first,
        },
        Grammar::Optional(content) => optional(older(*content, ends, false)),
        Grammar::Star(content) => list(older(*content, ends, false), String::new(), false, ends),
        Grammar::Plus(content) => list(older(*content, ends, false), String::new(), true, ends),
        Grammar::Or(elements) => Grammar::Or(
            elements
                .into_iter()
                .map(|element| older(element, ends, tail))
                .collect(),
        ),
        Grammar::Sequence(elements) => sequence(elements, ends, tail),
        leaf @ (Grammar::ConstString(_)
        | Grammar::AnyText { .. }
        | Grammar::Regex(_)
        | Grammar::JsonSchema { .. }
        | Grammar::Ebnf(_)
        | Grammar::Token(_)) => leaf,
    }
}

/// A tag with its content rewritten under its own ends.
fn in_tag(tag: Tag) -> Tag {
    let ends = match &tag.end {
        Bound::Text(text) => vec![text.clone()],
        Bound::Texts(texts) => texts.clone(),
    };
    Tag {
        content: Box::new(older(*tag.content, &ends, true)),
        ..tag
    }
}

/// `content` or nothing: a list that may repeat with `at_least_one` off; a choice with the
/// empty text otherwise.
fn optional(content: Grammar) -> Grammar {
    match content {
        Grammar::TagsWithSeparator {
            tags,
            separator,
            at_least_one: true,
            stop_after_first: false,
        } => Grammar::TagsWithSeparator {
            tags,
            separator,
            at_least_one: false,
            stop_after_first: false,
        },
        content => Grammar::Or(vec![content, Grammar::ConstString(String::new())]),
    }
}

/// `item` any number of times (at least once with `at_least_one`), `separator` between repeats:
/// a list of the item's tags, or of a tag with an empty begin and end around it. That tag becomes
/// the nearest tag of the item's texts, whose stop strings the engines read from it, so the
/// enclosing tag's `ends` go into the excluded strings of every text not under a tag of its own:
/// the same language, since the enclosing end stopped those texts before.
fn list(item: Grammar, separator: String, at_least_one: bool, ends: &[String]) -> Grammar {
    let tags = tags_of(&item).unwrap_or_else(|| {
        vec![Tag::bounded(
            "",
            with_ends(item, ends),
            Bound::Text(String::new()),
        )]
    });
    Grammar::TagsWithSeparator {
        tags,
        separator,
        at_least_one,
        stop_after_first: false,
    }
}

/// `grammar` with `ends` among the excluded strings of every text not under a tag of its own.
fn with_ends(grammar: Grammar, ends: &[String]) -> Grammar {
    match grammar {
        Grammar::AnyText { mut excludes } => {
            for end in ends.iter().filter(|end| !end.is_empty()) {
                if !excludes.contains(end) {
                    excludes.push(end.clone());
                }
            }
            Grammar::AnyText { excludes }
        }
        Grammar::Sequence(elements) => Grammar::Sequence(
            elements
                .into_iter()
                .map(|element| with_ends(element, ends))
                .collect(),
        ),
        Grammar::Or(elements) => Grammar::Or(
            elements
                .into_iter()
                .map(|element| with_ends(element, ends))
                .collect(),
        ),
        other => other,
    }
}

/// The tags `grammar` is, when it is one tag or a choice of tags.
fn tags_of(grammar: &Grammar) -> Option<Vec<Tag>> {
    match grammar {
        Grammar::Tag(tag) => Some(vec![tag.clone()]),
        Grammar::Or(elements) => elements
            .iter()
            .map(|element| match element {
                Grammar::Tag(tag) => Some(tag.clone()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// A sequence's elements rewritten: a repeated item folded with its repeats into one list, then
/// an unlimited element that does not end the sequence put under a tag whose end is the constant
/// text after it.
fn sequence(elements: Vec<Grammar>, ends: &[String], tail: bool) -> Grammar {
    let last = elements.len().saturating_sub(1);
    let mut out: Vec<Grammar> = Vec::with_capacity(elements.len());
    let mut elements = elements.into_iter().peekable();
    let mut at = 0;
    while let Some(element) = elements.next() {
        let repeat = match elements.peek() {
            // Y star(Y): the item, then any number of it.
            Some(Grammar::Star(content)) if **content == element => Some(String::new()),
            // Y star(S Y): the item, then any number of the separator and the item.
            Some(Grammar::Star(content)) => match content.as_ref() {
                Grammar::Sequence(pair) => match pair.as_slice() {
                    [Grammar::ConstString(separator), item] if *item == element => {
                        Some(separator.clone())
                    }
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        if let Some(separator) = repeat {
            elements.next();
            out.push(list(older(element, ends, false), separator, true, ends));
            at += 2;
            continue;
        }
        out.push(older(element, ends, tail && at == last));
        at += 1;
    }
    if let Some(at) = out[..out.len().saturating_sub(1)]
        .iter()
        .position(unlimited)
    {
        if let Some(end) = constant_text(&out[at + 1..]) {
            let before = constant_text(&out[..at]);
            let content = out.remove(at);
            let tag = |begin: String| Grammar::Tag(Tag::bounded(begin, content, Bound::Text(end)));
            return match before {
                Some(begin) if at > 0 => tag(begin),
                _ => {
                    out.truncate(at);
                    out.push(tag(String::new()));
                    one_or_all(out)
                }
            };
        }
    }
    one_or_all(out)
}

/// The one element of `elements` as it is, or the elements as a sequence.
fn one_or_all(mut elements: Vec<Grammar>) -> Grammar {
    match elements.len() {
        1 => elements.pop().unwrap_or(Grammar::Sequence(Vec::new())),
        _ => Grammar::Sequence(elements),
    }
}

/// The text of `elements` when every one is a constant string, nested sequences included.
fn constant_text(elements: &[Grammar]) -> Option<String> {
    let mut text = String::new();
    for element in elements {
        match element {
            Grammar::ConstString(value) => text.push_str(value),
            Grammar::Sequence(inner) => text.push_str(&constant_text(inner)?),
            _ => return None,
        }
    }
    Some(text)
}

/// Whether the older analyzer counts `grammar` as unlimited: free text without excluded strings,
/// a list of tags, a sequence whose last element is, a choice with one; not a tag, a constant, a
/// JSON value, a regular expression, a grammar or a token, nor free text or triggered tags held
/// by their excluded strings.
fn unlimited(grammar: &Grammar) -> bool {
    match grammar {
        Grammar::AnyText { excludes } => excludes.is_empty(),
        Grammar::TriggeredTags { .. } | Grammar::TagsWithSeparator { .. } => true,
        Grammar::Star(_) | Grammar::Plus(_) => true,
        Grammar::Optional(content) => unlimited(content),
        Grammar::Sequence(elements) => elements.last().is_some_and(unlimited),
        Grammar::Or(elements) => elements.iter().any(unlimited),
        Grammar::ConstString(_)
        | Grammar::Regex(_)
        | Grammar::JsonSchema { .. }
        | Grammar::Ebnf(_)
        | Grammar::Token(_)
        | Grammar::Tag(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json, Value};

    use super::{
        super::{GrammarStart, Options, OutputChoice, Turn},
        *,
    };
    use crate::formats::Table;

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        }
    }

    fn tools() -> Vec<Tool> {
        vec![
            tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"],
                }),
            ),
            tool("ping", json!({"type": "object", "properties": {}})),
        ]
    }

    fn text(value: &str) -> Grammar {
        Grammar::ConstString(value.to_string())
    }

    fn free(excludes: &[&str]) -> Grammar {
        Grammar::AnyText {
            excludes: excludes.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn tag(begin: &str, content: Grammar, end: &str) -> Tag {
        Tag::new(begin, content, end)
    }

    /// The node types written in `value`.
    fn types(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                if let Some(Value::String(t)) = map.get("type") {
                    out.push(t.clone());
                }
                map.values().for_each(|v| types(v, out));
            }
            Value::Array(items) => items.iter().for_each(|v| types(v, out)),
            _ => {}
        }
    }

    /// Every grammar a registry table writes: the forced and `auto` tags in every configuration
    /// the gateway asks for, the answer grammars, and the whole-turn grammars.
    fn every_registry_grammar() -> Vec<(String, Grammar)> {
        let tools = tools();
        let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
        let mut all = Vec::new();
        for table in Table::ALL {
            let format = table.format();
            for (at_least_one, one_call) in
                [(true, true), (true, false), (false, false), (false, true)]
            {
                for reasoning_open in [false, true] {
                    for start in [
                        GrammarStart::Unknown,
                        GrammarStart::AfterReasoning,
                        GrammarStart::FirstToken,
                        GrammarStart::Either,
                    ] {
                        let options = Options {
                            at_least_one,
                            reasoning_open,
                            thought_closed: false,
                            one_call,
                            start,
                        };
                        for passed in [&tools[..1], &tools[1..], &tools[..]] {
                            if let Some(grammar) = format.grammar_for(passed, options) {
                                all.push((format!("{} {options:?}", table.name()), grammar));
                            }
                        }
                        if let Some(grammar) =
                            format.answer_grammar_for(&schema, reasoning_open, start)
                        {
                            all.push((
                                format!("{} answer {reasoning_open} {start:?}", table.name()),
                                grammar,
                            ));
                        }
                    }
                }
            }
            for choice in [
                OutputChoice::One,
                OutputChoice::Required,
                OutputChoice::Auto,
            ] {
                for parallel in [true, false] {
                    let turn = Turn {
                        prompt: "",
                        continued: false,
                        declared: &tools,
                        callable: &tools,
                        choice,
                        parallel,
                        answer: None,
                    };
                    if let Ok(Some(grammar)) = format.turn_grammar(&turn) {
                        all.push((
                            format!("{} turn {choice:?} {parallel}", table.name()),
                            grammar,
                        ));
                    }
                }
            }
        }
        all
    }

    #[test]
    fn every_registry_grammar_in_the_older_vocabulary_has_none_of_the_newer_nodes_and_passes_the_rule(
    ) {
        let all = every_registry_grammar();
        assert!(all.len() > 1000, "{} grammars", all.len());
        let mut rewritten = 0;
        for (name, grammar) in all {
            let older = grammar.clone().in_older_vocabulary();
            let mut found = Vec::new();
            types(&older.to_json(), &mut found);
            let newer: Vec<&String> = found
                .iter()
                .filter(|t| matches!(t.as_str(), "optional" | "plus" | "star"))
                .collect();
            assert!(newer.is_empty(), "{name}: {newer:?} left in {older:?}");
            older.check().unwrap_or_else(|why| {
                panic!("{name}: the engines refuse the older spelling: {why}")
            });
            // The spelling is a fixed point: nothing to rewrite the second time.
            assert_eq!(older.clone().in_older_vocabulary(), older, "{name}");
            let mut before = Vec::new();
            types(&grammar.to_json(), &mut before);
            if before
                .iter()
                .any(|t| matches!(t.as_str(), "optional" | "plus" | "star"))
            {
                rewritten += 1;
            } else {
                // Without the nodes, every node stays; only an exclude an end makes redundant goes.
                assert_eq!(
                    found, before,
                    "{name}: a grammar without the nodes keeps its nodes"
                );
            }
        }
        assert!(
            rewritten > 100,
            "{rewritten} grammars carried the newer nodes"
        );
    }

    #[test]
    fn an_optional_limited_content_is_a_choice_with_the_empty_text() {
        let grammar = Grammar::Sequence(vec![
            text("a"),
            Grammar::Optional(Box::new(text("b"))),
            text("c"),
        ]);
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::Sequence(vec![
                text("a"),
                Grammar::Or(vec![text("b"), text("")]),
                text("c"),
            ])
        );
    }

    #[test]
    fn a_repeated_tag_is_a_list_of_that_tag() {
        // The item's own text loses the exclude its end makes redundant.
        let item = tag("<p>", free(&["</p>", "<p>"]), "</p>");
        let grammar = Grammar::Tag(tag(
            "<call>",
            Grammar::Star(Box::new(Grammar::Tag(item))),
            "</call>",
        ));
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::Tag(tag(
                "<call>",
                Grammar::TagsWithSeparator {
                    tags: vec![tag("<p>", free(&["<p>"]), "</p>")],
                    separator: String::new(),
                    at_least_one: false,
                    stop_after_first: false,
                },
                "</call>",
            ))
        );
    }

    #[test]
    fn one_or_more_of_a_choice_of_tags_is_a_list_of_those_tags() {
        let a = tag("<a>", text("1"), "</a>");
        let b = tag("<b>", text("2"), "</b>");
        let grammar = Grammar::Plus(Box::new(Grammar::Or(vec![
            Grammar::Tag(a.clone()),
            Grammar::Tag(b.clone()),
        ])));
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::TagsWithSeparator {
                tags: vec![a, b],
                separator: String::new(),
                at_least_one: true,
                stop_after_first: false,
            }
        );
    }

    #[test]
    fn a_separated_list_between_brackets_is_a_tag_around_one_list() {
        // `[ A (, A)* ]`, as the JSON list writes it: the brackets become the tag, since the
        // older analyzer lets a list stand only last in its sequence.
        let item = Grammar::Sequence(vec![text("{"), text("}")]);
        let grammar = Grammar::Sequence(vec![
            text("["),
            item.clone(),
            Grammar::Star(Box::new(Grammar::Sequence(vec![text(", "), item.clone()]))),
            text("]"),
        ]);
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::Tag(tag(
                "[",
                Grammar::TagsWithSeparator {
                    tags: vec![tag("", item, "")],
                    separator: ", ".to_string(),
                    at_least_one: true,
                    stop_after_first: false,
                },
                "]",
            ))
        );
    }

    #[test]
    fn an_optional_separated_list_is_the_list_that_may_be_empty() {
        let item = Grammar::Sequence(vec![text("k:"), text("v")]);
        let grammar = Grammar::Tag(tag(
            "{",
            Grammar::Optional(Box::new(Grammar::Sequence(vec![
                item.clone(),
                Grammar::Star(Box::new(Grammar::Sequence(vec![text(","), item.clone()]))),
            ]))),
            "}",
        ));
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::Tag(tag(
                "{",
                Grammar::TagsWithSeparator {
                    tags: vec![tag("", item, "")],
                    separator: ",".to_string(),
                    at_least_one: false,
                    stop_after_first: false,
                },
                "}",
            ))
        );
    }

    #[test]
    fn a_list_not_last_in_its_sequence_takes_the_text_after_it_as_a_tag_end() {
        let item = Grammar::Tag(tag("<i>", text("1"), "</i>"));
        let grammar = Grammar::Sequence(vec![
            Grammar::Regex("[a-z]+".to_string()),
            Grammar::Plus(Box::new(item.clone())),
            text("</list>"),
            text("\n"),
        ]);
        let Grammar::Tag(i) = item else {
            panic!("a tag")
        };
        assert_eq!(
            grammar.in_older_vocabulary(),
            Grammar::Sequence(vec![
                Grammar::Regex("[a-z]+".to_string()),
                Grammar::Tag(tag(
                    "",
                    Grammar::TagsWithSeparator {
                        tags: vec![i],
                        separator: String::new(),
                        at_least_one: true,
                        stop_after_first: false,
                    },
                    "</list>\n",
                )),
            ])
        );
    }

    #[test]
    fn a_wrapped_item_carries_the_enclosing_end_into_its_texts() {
        // The tag around the repeated item becomes the nearest tag of the item's texts, so the
        // enclosing end, which stopped them, joins their excluded strings; a text under a tag of
        // the item's own keeps its list.
        let item = Grammar::Sequence(vec![
            text("<k>"),
            free(&["</k>"]),
            text("</k>"),
            Grammar::Tag(tag("<v>", free(&["</v>"]), "</v>")),
        ]);
        let grammar = Grammar::Tag(tag("<call>", Grammar::Star(Box::new(item)), "</call>"));
        let Grammar::Tag(older) = grammar.in_older_vocabulary() else {
            panic!("a tag")
        };
        let Grammar::TagsWithSeparator { tags, .. } = *older.content else {
            panic!("a list")
        };
        assert_eq!(
            *tags[0].content,
            Grammar::Sequence(vec![
                text("<k>"),
                free(&["</k>", "</call>"]),
                text("</k>"),
                // (the nested tag's text loses the exclude its own end makes redundant)
                Grammar::Tag(tag("<v>", free(&[]), "</v>")),
            ])
        );
    }

    #[test]
    fn a_text_last_before_its_tag_end_drops_the_excludes_the_end_makes_redundant() {
        // The thought: its text may not hold the close that ends it, which the end already
        // guarantees; a text with more after it keeps the exclude, since the end does not stop it
        // there; an exclude the end merely begins stays (it is not redundant).
        let thought = Grammar::Tag(tag(
            "<think>",
            free(&["<think>", "</think>", "</think>more", "<tool>"]),
            "</think>",
        ));
        assert_eq!(
            thought.in_older_vocabulary(),
            Grammar::Tag(tag("<think>", free(&["<think>", "<tool>"]), "</think>"))
        );
        let value = Grammar::Tag(tag(
            "<p>",
            Grammar::Sequence(vec![free(&["</p>", "<p>"]), text(")")]),
            "</p>",
        ));
        assert_eq!(value.clone().in_older_vocabulary(), value);
        let prefix = Grammar::Tag(tag("<p>", free(&["</p"]), "</p>\n"));
        assert_eq!(prefix.clone().in_older_vocabulary(), prefix);
    }

    #[test]
    fn tokens_triggers_and_leaves_are_left_as_derived() {
        let grammar = Grammar::TriggeredTags {
            triggers: vec!["<t>".to_string()],
            tags: vec![tag(
                "<t>",
                Grammar::Sequence(vec![
                    Grammar::Token("<|x|>".to_string()),
                    Grammar::JsonSchema {
                        schema: json!({"type": "object"}),
                        style: None,
                        excludes: vec!["</t>".to_string()],
                        whitespace: crate::Whitespace::Free,
                    },
                    Grammar::Ebnf("root ::= \"a\"".to_string()),
                ]),
                "</t>",
            )],
            at_least_one: true,
            stop_after_first: true,
        };
        assert_eq!(grammar.clone().in_older_vocabulary(), grammar);
    }
}

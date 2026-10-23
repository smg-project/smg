//! Layouts learned from the renderings of chat templates written for these tests, on abstract
//! tags: a table whose text outside every field is dropped, a thought between `<t>` and `</t>`, an
//! answer after `<a>` that the stop token `<s>` ends, and calls opened by `<c` and a header that
//! names the tool, closed by `</c>`, each parameter `<p KEY>VALUE</p>`; and a table whose last call
//! the stream's stop closes, each call `[c] NAME` and its parameters `<k KEY>VALUE</k>`, a value
//! that may also end where the next parameter begins, a call before another closed by `[e]`.

use std::sync::Arc;

use symphony::{
    shape::{
        intern, Arguments, Atom, CallLayout, CallShape, Class, Dialect, End, Layout, LayoutError,
        Message, NameSource, Nested, Pattern, Segment, Slot, Spellings, ValueParser, Words,
        Written,
    },
    CallSyntax, Emits, Format,
};

use super::*;
use crate::to_symphony::Transformers;

fn word() -> Class {
    Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
}

fn none_of(chars: &str) -> Class {
    Class::new(chars.chars().map(|c| (c, c)), true)
}

fn run(class: Class, min: u16, greedy: bool, capture: Option<Slot>) -> Atom {
    Atom::Run {
        class,
        min,
        max: None,
        greedy,
        capture,
    }
}

fn lit(text: &str) -> Atom {
    Atom::Lit(text.into())
}

/// The calls of the first table: after `<c`, attributes skipped up to `n="NAME">`; each parameter
/// `<p KEY>` and its value up to `</p>`.
fn tagged() -> CallShape {
    CallShape {
        header: Pattern(
            [
                Atom::Boundary,
                run(none_of(">"), 0, false, None),
                Atom::Boundary,
                lit("n=\""),
                run(none_of("\""), 1, true, Some(Slot::Name)),
                lit("\">"),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    lit("<p "),
                    run(none_of(">"), 1, true, Some(Slot::Key)),
                    lit(">"),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern([lit("</p>")].into()),
                ahead: Pattern::default(),
            }]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    }
}

/// The first table: the text outside every field dropped, a thought, an answer the stop token
/// ends, and the calls; the turn after `<as>`. Its values are typed as transformers types them,
/// as a compiled table's are.
fn framed(shape: CallShape) -> Format {
    Format::new("framed")
        .terminal("open", "<c")
        .terminal("close", "</c>")
        .terminal("think_open", "<t>")
        .terminal("think_close", "</t>")
        .terminal("answer_open", "<a>")
        .terminal("stop", "<s>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "open", "call")
        .transition("call", "close", "outside")
        .transition("outside", "think_open", "reasoning")
        .transition("reasoning", "think_close", "outside")
        .transition("outside", "answer_open", "content")
        .transition("content", "stop", "outside")
        .calls(CallSyntax::Shaped(intern(shape)))
        .framing("outside")
        .turn_anchor("<as>")
        .stop_tokens(["<s>"])
        .reads_values_with(Arc::new(Transformers))
}

/// The second table: a call is `[c]` and the tool's name after a space; each parameter `<k KEY>`
/// and its value up to `</k>`, or up to a newline before the next parameter's tag; a call before
/// another is closed by `[e]`, and the stream's stop (`[s]`) closes the last.
fn stopped() -> Format {
    let shape = CallShape {
        header: Pattern([lit(" "), run(word(), 1, true, Some(Slot::Name))].into()),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    lit("<k "),
                    run(none_of(">"), 1, true, Some(Slot::Key)),
                    lit(">"),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [
                End {
                    consume: Pattern([lit("</k>")].into()),
                    ahead: Pattern::default(),
                },
                End {
                    consume: Pattern([lit("\n")].into()),
                    ahead: Pattern([lit("<k ")].into()),
                },
            ]
            .into(),
            values: ValueParser::Raw,
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: 8192,
    };
    Format::new("stopped")
        .terminal("open", "[c]")
        .terminal("next", "[e]")
        .terminal("answer_open", "[a]")
        .terminal("stop", "[s]")
        .state("outside", Emits::Content)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "open", "call")
        .transition("call", "next", "outside")
        .transition("call", "stop", "outside")
        .transition("outside", "answer_open", "content")
        .transition("content", "stop", "outside")
        .calls(CallSyntax::Shaped(intern(shape)))
        .framing("outside")
        .closes_at_stop("call")
        .turn_anchor("[as]")
        .stop_tokens(["[s]"])
        .reads_values_with(Arc::new(Transformers))
}

/// How a chat template written for these tests renders the probe on the first table; each test
/// changes what it is about.
#[derive(Clone)]
struct Style {
    /// The field the template writes a thought from, if any.
    thought: Option<ThoughtField>,
    /// Whether the generation prompt opens the thought.
    opens_thought: bool,
    /// Whether a turn writes its first call alone.
    one_call: bool,
    /// What a call is written after, for a tool.
    lead: fn(&str) -> String,
    /// What the second call's header writes before `n=`.
    second_header: &'static str,
    /// How `true` and `null` are written.
    words: (&'static str, &'static str),
    /// How a parameter after the first is written, for its key and value.
    later: fn(&str, &str) -> String,
    /// Whether the parameters are written in the order of their keys.
    sorted: bool,
    /// The cases the template raises on.
    raises: fn(Case) -> bool,
    /// Whether a turn whose message has no thought writes an empty one.
    empty_thought: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            thought: Some(ThoughtField::ReasoningContent),
            opens_thought: false,
            one_call: false,
            lead: |name| format!("@{name}\n"),
            second_header: " ",
            words: ("true", "null"),
            later: |key, value| format!("<p {key}>{value}</p>"),
            sorted: false,
            raises: |_| false,
            empty_thought: false,
        }
    }
}

const PROMPT: &str = "<us>Call the probe tools.<as>";

impl Style {
    /// The rendering of `case`.
    fn render(&self, case: Case) -> Option<String> {
        if (self.raises)(case) {
            return None;
        }
        let prompt = if self.opens_thought {
            format!("{PROMPT}<t>")
        } else {
            PROMPT.to_owned()
        };
        let (field, calls) = match case {
            Case::Prompt => return Some(prompt),
            Case::Calls => (None, Some(2)),
            Case::OneCall => (None, Some(1)),
            Case::Answer => (None, None),
            Case::ThoughtCalls(field) => (Some(field), Some(2)),
            Case::ThoughtOneCall(field) => (Some(field), Some(1)),
            Case::ThoughtAnswer(field) => (Some(field), None),
        };
        let thinks = field.is_some() && field == self.thought;
        let mut turn = String::new();
        if self.opens_thought {
            turn.push_str(if thinks { "Probe thought." } else { "" });
            turn.push_str("</t>\n");
        } else if thinks {
            turn.push_str("<t>Probe thought.</t>\n");
        } else if self.empty_thought {
            turn.push_str("<t></t>\n");
        }
        match calls {
            Some(count) => {
                let count = if self.one_call { 1 } else { count };
                let (true_word, null_word) = self.words;
                let alpha = [
                    ("s", "sigma"),
                    ("i", "42"),
                    ("n", "2.5"),
                    ("b", true_word),
                    ("a", "[1, 2]"),
                    ("o", "{\"m\": \"mu\"}"),
                    ("z", null_word),
                ];
                let calls: [(&str, Vec<(&str, &str)>); 2] = [
                    ("fn_alpha", alpha.to_vec()),
                    ("fn_beta", vec![("q", "kappa")]),
                ];
                for (at, (name, mut params)) in calls.into_iter().take(count).enumerate() {
                    if self.sorted {
                        params.sort_unstable();
                    }
                    if at > 0 {
                        turn.push('\n');
                    }
                    turn.push_str(&(self.lead)(name));
                    let space = if at > 0 { self.second_header } else { " " };
                    turn.push_str(&format!("<c{space}n=\"{name}\">\n"));
                    for (index, (key, value)) in params.iter().enumerate() {
                        if index == 0 {
                            turn.push_str(&format!("<p {key}>{value}</p>"));
                        } else {
                            turn.push('\n');
                            turn.push_str(&(self.later)(key, value));
                        }
                    }
                    turn.push_str("\n</c>");
                }
            }
            None => turn.push_str(" <a>{\"probe\": 1}"),
        }
        turn.push_str("<s>\n");
        Some(format!("{prompt}{turn}"))
    }

    fn learn(&self) -> Result<Layout, NoLayout> {
        self.learned().map(|learned| learned.layout)
    }

    fn learned(&self) -> Result<Learned, NoLayout> {
        learn(&framed(tagged()), &["<s>"], |case| self.render(case))
    }
}

fn written(segments: impl IntoIterator<Item = Segment>) -> Written {
    Written(segments.into_iter().collect())
}

fn text(text: &str) -> Segment {
    Segment::Lit(text.into())
}

/// The layout the default style renders.
fn framed_layout() -> Layout {
    Layout {
        call: CallLayout {
            lead: written([text("@"), Segment::Slot(Slot::Name), text("\n")]),
            header: written([text(" n=\""), Segment::Slot(Slot::Name), text("\">")]),
            before: "\n".into(),
            parameter: written([
                text("<p "),
                Segment::Slot(Slot::Key),
                text(">"),
                Segment::Slot(Slot::Value),
                text("</p>"),
            ]),
            between: "\n".into(),
            after: "\n".into(),
            trailer: "".into(),
            close: Some("</c>".into()),
        },
        between_calls: Some("\n".into()),
        after_calls: "".into(),
        values: Spellings::default(),
        thought: Some(Message {
            before: "".into(),
            after: "\n".into(),
            stops: false,
            close: Some("</t>".into()),
        }),
        answer: Some(Message {
            before: " ".into(),
            after: "".into(),
            stops: true,
            close: None,
        }),
    }
}

#[test]
fn the_probes_cases_are_named_and_hold_their_messages() {
    let names: Vec<String> = Case::ALL.iter().map(|case| case.name()).collect();
    assert_eq!(names[..4], ["prompt", "calls", "one_call", "answer"]);
    assert_eq!(names[4], "calls+reasoning_content");
    assert_eq!(names[12], "answer+thinking");
    assert!(Case::Prompt.generation_prompt());
    assert!(Case::ALL[1..].iter().all(|case| !case.generation_prompt()));
    let turn = &Case::ThoughtOneCall(ThoughtField::Thinking).messages()[1];
    assert_eq!(turn["thinking"], json!("Probe thought."));
    assert_eq!(turn["tool_calls"][0]["function"]["name"], json!("fn_alpha"));
    assert_eq!(turn["tool_calls"].as_array().map(Vec::len), Some(1));
    let tools: Vec<Tool> = serde_json::from_value(tools()).expect("the probe's tools");
    assert_eq!(tools.len(), 2);
    assert_eq!(Case::Answer.messages()[1]["content"], json!(ANSWER));
}

#[test]
fn a_layout_is_learned_from_the_probes_renderings() {
    let learned = Style::default().learned().expect("a layout");
    assert!(!learned.thinks_first);
    let layout = learned.layout;
    assert_eq!(layout, framed_layout());
    let format = framed(tagged()).with_layout(layout);
    let tools: Vec<Tool> = serde_json::from_value(tools()).expect("the probe's tools");
    assert!(format.grammar(&tools, true, false).is_some());
    assert!(format
        .answer_grammar(&json!({"type": "object"}), false)
        .is_some());
    assert_eq!(
        summary(&framed_layout()),
        "lead \"@{name}\\n\" header \" n=\\\"{name}\\\">\" before \"\\n\" parameter \
         \"<p {key}>{value}</p>\" between \"\\n\" after \"\\n\" trailer \"\" close Some(\"</c>\"); \
         between calls Some(\"\\n\"); after calls \"\"; values Json Json; thought before \"\" \
         close \"</t>\" after \"\\n\"; answer before \" \" after \"\" stops"
    );
}

#[test]
fn a_template_that_writes_one_call_in_a_turn_has_a_layout_of_one_call() {
    let one = Style {
        one_call: true,
        ..Style::default()
    };
    let layout = one.learn().expect("a layout");
    assert_eq!(layout.between_calls, None);
    assert_eq!(layout.call, framed_layout().call);
    // A turn with the calls that has no rendering is read with one call too.
    let raised = learn(&framed(tagged()), &["<s>"], |case| match case {
        Case::Calls | Case::ThoughtCalls(_) => None,
        case => Style::default().render(case),
    })
    .expect("a layout");
    assert_eq!(raised.layout.between_calls, None);
    // A turn with one call that writes what the turn with the calls does not is no layout.
    let uneven = learn(&framed(tagged()), &["<s>"], |case| {
        let rendered = Style::default().render(case)?;
        Some(match case {
            Case::OneCall => rendered.replace("</c><s>", "</c>\n<s>"),
            _ => rendered,
        })
    });
    assert_eq!(uneven, Err(NoLayout::Uneven("turn with one call")));
}

#[test]
fn the_thought_is_learned_from_the_field_the_template_writes_it_from() {
    for field in ThoughtField::ALL {
        let style = Style {
            thought: Some(field),
            ..Style::default()
        };
        assert_eq!(
            style.learn().expect("a layout"),
            framed_layout(),
            "{field:?}"
        );
    }
    let unwritten = Style {
        thought: None,
        ..Style::default()
    };
    let layout = unwritten.learn().expect("a layout");
    assert_eq!(layout.thought, None, "no turn writes a thought");
    assert_eq!(layout.call, framed_layout().call);
}

#[test]
fn where_the_prompt_opens_the_thought_the_text_after_its_close_is_the_thoughts() {
    let open = Style {
        opens_thought: true,
        lead: |_| String::new(),
        ..Style::default()
    };
    let layout = open.learn().expect("a layout");
    assert_eq!(layout.call.lead, Written::default());
    assert_eq!(
        layout.thought,
        Some(Message {
            before: "".into(),
            after: "\n".into(),
            stops: false,
            close: Some("</t>".into()),
        })
    );
    assert_eq!(layout.answer, framed_layout().answer);
    // A lead that names the tool cannot be told from the thought's close there: the text the
    // layout would write between two calls, or after the thought, names one.
    let named = Style {
        opens_thought: true,
        ..Style::default()
    };
    assert_eq!(
        named.learn(),
        Err(NoLayout::Named("text between two calls"))
    );
    let one = Style {
        opens_thought: true,
        one_call: true,
        ..Style::default()
    };
    assert_eq!(one.learn(), Err(NoLayout::Named("text around the thought")));
}

#[test]
fn a_template_that_requires_the_thought_is_read_from_the_turns_that_write_it() {
    // The template raises on an assistant turn without a thought.
    let required = Style {
        raises: |case| matches!(case, Case::Calls | Case::OneCall | Case::Answer),
        lead: |_| String::new(),
        ..Style::default()
    };
    let learned = required.learned().expect("a layout");
    let mut expected = framed_layout();
    expected.call.lead = Written::default();
    assert_eq!(learned.layout, expected);
    assert!(learned.thinks_first, "no turn without a thought");
    // Read from the thought's field the template writes: the others render no thought.
    let thinking = Style {
        thought: Some(ThoughtField::Thinking),
        ..required.clone()
    };
    assert_eq!(thinking.learn().expect("a layout"), expected);
    // A lead that names the tool cannot be told from the thought's close.
    let named = Style {
        lead: |name| format!("@{name}\n"),
        ..required
    };
    assert_eq!(
        named.learn(),
        Err(NoLayout::Named("text between two calls"))
    );
}

#[test]
fn a_template_that_writes_a_thought_before_every_call_and_answer_thinks_first() {
    let empty = Style {
        empty_thought: true,
        lead: |_| String::new(),
        ..Style::default()
    };
    let learned = empty.learned().expect("a layout");
    let mut expected = framed_layout();
    expected.call.lead = Written::default();
    assert_eq!(learned.layout, expected);
    assert!(learned.thinks_first);
    let format = learned.apply(framed(tagged()));
    assert_eq!(format.layout(), Some(&expected));
    // A thought the prompt opened is the prompt's, not the turn's: no thinking first.
    let open = Style {
        opens_thought: true,
        lead: |_| String::new(),
        ..Style::default()
    };
    assert!(!open.learned().expect("a layout").thinks_first);
}

#[test]
fn python_words_are_learned() {
    let python = Style {
        words: ("True", "None"),
        ..Style::default()
    };
    let layout = python.learn().expect("a layout");
    assert_eq!(
        layout.values,
        Spellings {
            words: Words::Python,
            nested: Nested::Json,
        }
    );
    let other = Style {
        words: ("1", "None"),
        ..Style::default()
    };
    assert_eq!(other.learn(), Err(NoLayout::Spelling("boolean or null")));
}

#[test]
fn calls_written_unevenly_or_out_of_order_have_no_layout() {
    let header = Style {
        second_header: "  ",
        ..Style::default()
    };
    assert_eq!(header.learn(), Err(NoLayout::Uneven("header")));
    let parameter = Style {
        later: |key, value| format!("<p {key}>{value}</p >"),
        ..Style::default()
    };
    assert!(matches!(parameter.learn(), Err(NoLayout::ReadBack { .. })));
    let spaced = Style {
        later: |key, value| match key {
            "n" => format!(" <p {key}>{value}</p>"),
            _ => format!("<p {key}>{value}</p>"),
        },
        ..Style::default()
    };
    assert_eq!(
        spaced.learn(),
        Err(NoLayout::Uneven("text between two parameters"))
    );
    let sorted = Style {
        sorted: true,
        ..Style::default()
    };
    assert_eq!(sorted.learn(), Err(NoLayout::Order));
    // Two places for the name in the lead: the layout does not fit the table.
    let twice = Style {
        lead: |name| format!("@{name} {name}\n"),
        ..Style::default()
    };
    assert_eq!(
        twice.learn(),
        Err(NoLayout::Unfit(LayoutError::Places("lead")))
    );
}

#[test]
fn renderings_that_do_not_read_back_have_no_layout() {
    // The calls written with a header the response template does not read: no calls.
    let other = learn(&framed(tagged()), &["<s>"], |case| {
        Some(Style::default().render(case)?.replace(" n=\"", " id=\""))
    });
    assert!(
        matches!(&other, Err(NoLayout::ReadBack { case: Case::Calls, read }) if read.contains("\"calls\":[]")),
        "{other:?}"
    );
    // Turns that do not follow the prompt's rendering.
    let prompt = learn(&framed(tagged()), &["<s>"], |case| match case {
        Case::Prompt => Some(format!("{PROMPT}\n")),
        case => Style::default().render(case),
    });
    assert_eq!(prompt, Err(NoLayout::Prompt(Case::Calls)));
    // A template that raises on the answer without the thought has its answer read from the turn
    // with the thought; one that raises on both has no layout.
    let thought_only = learn(&framed(tagged()), &["<s>"], |case| match case {
        Case::Answer => None,
        case => Style::default().render(case),
    });
    assert_eq!(
        thought_only.map(|learned| learned.layout),
        Ok(framed_layout())
    );
    let unrendered = learn(&framed(tagged()), &["<s>"], |case| match case {
        Case::Answer | Case::ThoughtAnswer(_) => None,
        case => Style::default().render(case),
    });
    let field = ThoughtField::ReasoningContent;
    assert_eq!(
        unrendered,
        Err(NoLayout::Unrendered(Case::ThoughtAnswer(field)))
    );
    let unthought = learn(&framed(tagged()), &["<s>"], |case| match case {
        Case::Answer => None,
        case => Style {
            thought: None,
            ..Style::default()
        }
        .render(case),
    });
    assert_eq!(unthought, Err(NoLayout::Unrendered(Case::Answer)));
    let no_prompt = learn(&framed(tagged()), &["<s>"], |_| None);
    assert_eq!(no_prompt, Err(NoLayout::Unrendered(Case::Prompt)));
}

#[test]
fn a_table_without_tagged_parameters_given_as_data_has_no_layout() {
    let json = CallShape {
        arguments: Arguments::Json(Dialect::default()),
        ..tagged()
    };
    let render = |case| Style::default().render(case);
    assert_eq!(learn(&framed(json), &["<s>"], render), Err(NoLayout::Calls));
    let plain = Format::new("plain")
        .terminal("open", "<c")
        .state("outside", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "open", "call")
        .calls(CallSyntax::Json);
    assert_eq!(learn(&plain, &[], render), Err(NoLayout::Calls));
    assert_eq!(
        learn(&Format::new("none"), &[], render),
        Err(NoLayout::Calls)
    );
    let codes: Vec<&str> = [
        NoLayout::Calls,
        NoLayout::Table(String::new()),
        NoLayout::Unrendered(Case::Prompt),
        NoLayout::Prompt(Case::Calls),
        NoLayout::Uneven("header"),
        NoLayout::Named("lead"),
        NoLayout::Order,
        NoLayout::Spelling("number"),
        NoLayout::Unexplained(Case::Calls),
    ]
    .iter()
    .map(NoLayout::code)
    .collect();
    assert_eq!(
        codes,
        [
            "Calls",
            "Table",
            "Unrendered",
            "Prompt",
            "Uneven",
            "Named",
            "Order",
            "Spelling",
            "Unexplained"
        ]
    );
}

/// The first table's terminals with the marker `<n>` and the answer's close `</a>` too, and the
/// rows `rows` adds for the thought and the answer around the calls' rows.
fn varied(rows: impl FnOnce(Format) -> Format) -> Format {
    let format = Format::new("varied")
        .terminal("open", "<c")
        .terminal("close", "</c>")
        .terminal("think_open", "<t>")
        .terminal("think_close", "</t>")
        .terminal("next", "<n>")
        .terminal("answer_open", "<a>")
        .terminal("answer_close", "</a>")
        .terminal("stop", "<s>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("gap", Emits::Content)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "open", "call")
        .transition("call", "close", "outside");
    rows(format)
        .calls(CallSyntax::Shaped(intern(tagged())))
        .framing("outside")
        .framing("gap")
        .turn_anchor("<as>")
        .stop_tokens(["<s>"])
        .reads_values_with(Arc::new(Transformers))
}

/// A thought as the first table has it, and an answer the stop ends or `</a>` closes, its text
/// stripped.
fn closed_answer() -> Format {
    varied(|format| {
        format
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "answer_close", "outside")
            .transition("content", "stop", "outside")
            .trims("content")
    })
}

/// The default style's rendering of `case`, with `from` written as `to` where `which` says.
fn edited(
    which: fn(Case) -> bool,
    from: &'static str,
    to: &'static str,
) -> impl Fn(Case) -> Option<String> {
    move |case| {
        let rendered = Style::default().render(case)?;
        Some(if which(case) {
            rendered.replace(from, to)
        } else {
            rendered
        })
    }
}

/// A thought the template may close two ways, `<n>` or `</t>`, the row on `<n>` first in the
/// table, and an answer the stop ends.
fn closed_two_ways() -> Format {
    varied(|format| {
        format
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "next", "outside")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "stop", "outside")
    })
}

#[test]
fn a_thought_closed_one_of_two_ways_is_learned_with_the_close_the_template_renders() {
    let render = |case| Style::default().render(case);
    let learned = learn(&closed_two_ways(), &["<s>"], render).expect("a layout");
    assert_eq!(learned.layout, framed_layout());
    let thought = learned.layout.thought.as_ref().expect("a thought");
    assert_eq!(thought.close.as_deref(), Some("</t>"));
    // The tags close the thought as the template does, not with the table's first row.
    let format = learned.apply(closed_two_ways());
    let begin = format
        .answer_grammar(&json!({"type": "object"}), true)
        .map(|grammar| grammar.payload()["format"]["begin"].clone());
    assert_eq!(begin, Some(json!("</t>\n <a>")));
    // A template that closes the thought another way before an answer than before the calls has
    // no layout: a tag could not write both.
    let other = edited(
        |case| matches!(case, Case::ThoughtAnswer(_)),
        "Probe thought.</t>",
        "Probe thought.<n>",
    );
    assert_eq!(
        learn(&closed_two_ways(), &["<s>"], other).map(|learned| learned.layout),
        Err(NoLayout::Uneven("thought"))
    );
}

#[test]
fn a_thought_the_prompt_opens_and_a_turn_closes_another_way_has_no_layout() {
    // The prompt opens the thought, and every turn closes it with `</t>`, of its two closes.
    let open = Style {
        opens_thought: true,
        lead: |_| String::new(),
        ..Style::default()
    };
    let learned = learn(&closed_two_ways(), &["<s>"], |case| open.render(case)).expect("a layout");
    let thought = learned.layout.thought.as_ref().expect("a thought");
    assert_eq!(thought.close.as_deref(), Some("</t>"));
    // The turns in `cases` close it with `<n>` instead: the turns with the calls, against the
    // turn with the calls and the probe's thought, or the turn with the answer alone, against
    // the turns with the calls. A tag could not write both.
    let otherwise = |cases: fn(Case) -> bool| {
        let open = open.clone();
        learn(&closed_two_ways(), &["<s>"], move |case| {
            let rendered = open.render(case)?;
            Some(if cases(case) {
                rendered.replacen("</t>", "<n>", 1)
            } else {
                rendered
            })
        })
        .map(|learned| learned.layout)
    };
    assert_eq!(
        otherwise(|case| matches!(case, Case::Calls | Case::OneCall)),
        Err(NoLayout::Uneven("thought"))
    );
    assert_eq!(
        otherwise(|case| case == Case::Answer),
        Err(NoLayout::Uneven("thought"))
    );
}

#[test]
fn an_answers_text_stands_right_between_the_terminals_of_its_region() {
    let closed = edited(|_| true, "{\"probe\": 1}<s>", "{\"probe\": 1}</a><s>");
    let learned = learn(&closed_answer(), &["<s>"], closed).expect("a layout");
    let mut expected = framed_layout();
    expected.answer = Some(Message {
        before: " ".into(),
        after: "".into(),
        stops: false,
        close: Some("</a>".into()),
    });
    assert_eq!(learned.layout, expected);
    // A newline the stripped answer drops after its opener, or before its close, has no place in
    // a layout, and the tag would not write it.
    for (from, to) in [
        ("<a>{\"probe\": 1}<s>", "<a>\n{\"probe\": 1}</a><s>"),
        ("{\"probe\": 1}<s>", "{\"probe\": 1}\n</a><s>"),
    ] {
        let render = edited(|_| true, from, to);
        assert_eq!(
            learn(&closed_answer(), &["<s>"], render),
            Err(NoLayout::Unexplained(Case::Answer)),
            "{to:?}"
        );
    }
}

#[test]
fn the_text_the_tags_write_where_a_way_in_passes_rows_holds_no_marker() {
    // The way into the answer is two rows, `<n>` and `<a>`: the tag writes `<n>` itself, so the
    // text the layout writes before the answer's opener cannot hold it.
    let two_rows = varied(|format| {
        format
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "outside")
            .transition("outside", "next", "gap")
            .transition("gap", "answer_open", "content")
            .transition("content", "stop", "outside")
    });
    let render = edited(|_| true, " <a>", "<n> <a>");
    assert_eq!(
        learn(&two_rows, &["<s>"], render),
        Err(NoLayout::Unexplained(Case::Answer))
    );
    // The thought closes into a framing state that `<n>` leaves: the tags write `<n>` on their way
    // out of the thought, so the text the layout writes after the thought cannot hold it.
    let gap = varied(|format| {
        format
            .transition("outside", "think_open", "reasoning")
            .transition("reasoning", "think_close", "gap")
            .transition("gap", "next", "outside")
            .transition("outside", "answer_open", "content")
            .transition("content", "stop", "outside")
    });
    let render = edited(|_| true, "</t>\n", "</t>\n<n>");
    assert_eq!(
        learn(&gap, &["<s>"], render),
        Err(NoLayout::Unexplained(Case::ThoughtCalls(
            ThoughtField::ReasoningContent
        )))
    );
}

#[test]
fn a_thought_the_template_writes_itself_comes_before_the_calls_and_the_answer_alike() {
    let own = Style {
        empty_thought: true,
        lead: |_| String::new(),
        ..Style::default()
    };
    let none = Style {
        lead: |_| String::new(),
        ..Style::default()
    };
    // An empty thought before the calls but not before the answer, and the other way round: a
    // tag that writes the thought first, or one that does not, misses one of them.
    for (calls, answer) in [(own.clone(), none.clone()), (none, own.clone())] {
        let render = move |case: Case| match case {
            Case::Answer => answer.render(case),
            case => calls.render(case),
        };
        assert_eq!(
            learn(&framed(tagged()), &["<s>"], render),
            Err(NoLayout::Uneven("thought"))
        );
    }
    // An empty thought before the calls, and no answer without the probe's thought: every turn
    // the template renders writes a thought first.
    let render = move |case: Case| match case {
        Case::Answer => None,
        case => own.render(case),
    };
    let learned = learn(&framed(tagged()), &["<s>"], render).expect("a layout");
    assert!(learned.thinks_first);
}

#[test]
fn the_turns_with_the_probes_thought_write_the_calls_and_the_answer_as_the_others() {
    let calls = |case: Case| matches!(case, Case::ThoughtCalls(_));
    let header = edited(calls, "<c n=\"fn_beta\">", "<c  n=\"fn_beta\">");
    assert_eq!(
        learn(&framed(tagged()), &["<s>"], header),
        Err(NoLayout::Uneven("turn with the thought"))
    );
    let answer = |case: Case| matches!(case, Case::ThoughtAnswer(_));
    let closed = |case: Case| {
        let rendered = edited(|_| true, "{\"probe\": 1}<s>", "{\"probe\": 1}</a><s>")(case)?;
        Some(match case {
            Case::ThoughtAnswer(_) => rendered.replace("</a><s>", "</a>\n<s>"),
            _ => rendered,
        })
    };
    assert_eq!(
        learn(&closed_answer(), &["<s>"], closed),
        Err(NoLayout::Uneven("answer"))
    );
    // An answer the turn with the thought writes otherwise reads back otherwise where the table
    // keeps the text as written.
    let unstripped = edited(answer, "{\"probe\": 1}<s>", "{\"probe\": 1}\n<s>");
    assert!(matches!(
        learn(&framed(tagged()), &["<s>"], unstripped),
        Err(NoLayout::ReadBack {
            case: Case::ThoughtAnswer(ThoughtField::ReasoningContent),
            ..
        })
    ));
}

/// The second table's chat template: each parameter's value up to `</k>`, or, with `ahead`, a
/// value before another up to a newline before the next tag.
fn render_stopped(case: Case, ahead: bool) -> Option<String> {
    let prompt = "[us]Call the probe tools.[as]";
    let params = |params: &[(&str, &str)]| -> String {
        let mut out = String::new();
        for (at, (key, value)) in params.iter().enumerate() {
            let last = at + 1 == params.len();
            if ahead && !last {
                out.push_str(&format!("<k {key}>{value}\n"));
            } else {
                out.push_str(&format!("<k {key}>{value}</k>"));
                if !last {
                    out.push('\n');
                }
            }
        }
        out
    };
    let alpha = format!(
        "[c] fn_alpha\n{}",
        params(&[
            ("s", "sigma"),
            ("i", "42"),
            ("n", "2.5"),
            ("b", "true"),
            ("a", "[1, 2]"),
            ("o", "{\"m\": \"mu\"}"),
            ("z", "null"),
        ])
    );
    let beta = format!("[c] fn_beta\n{}", params(&[("q", "kappa")]));
    let turn = match case {
        Case::Prompt => return Some(prompt.to_owned()),
        Case::Calls => format!("{alpha}[e]{beta}[s]"),
        Case::OneCall => format!("{alpha}[s]"),
        Case::Answer => "[a]{\"probe\": 1}[s]".to_owned(),
        Case::ThoughtCalls(_) | Case::ThoughtOneCall(_) | Case::ThoughtAnswer(_) => return None,
    };
    Some(format!("{prompt}{turn}"))
}

#[test]
fn a_call_the_stop_closes_has_no_close_and_a_terminal_between_two_calls() {
    let learned =
        learn(&stopped(), &["[s]"], |case| render_stopped(case, false)).expect("a layout");
    assert!(!learned.thinks_first);
    let layout = learned.layout;
    assert_eq!(
        layout,
        Layout {
            call: CallLayout {
                lead: Written::default(),
                header: written([text(" "), Segment::Slot(Slot::Name)]),
                before: "\n".into(),
                parameter: written([
                    text("<k "),
                    Segment::Slot(Slot::Key),
                    text(">"),
                    Segment::Slot(Slot::Value),
                    text("</k>"),
                ]),
                between: "\n".into(),
                after: "".into(),
                trailer: "".into(),
                close: None,
            },
            between_calls: Some("[e]".into()),
            after_calls: "".into(),
            values: Spellings::default(),
            thought: None,
            answer: Some(Message {
                before: "".into(),
                after: "".into(),
                stops: true,
                close: None,
            }),
        }
    );
    // A value that ends where the next parameter begins is written otherwise than one that ends
    // at its tag: the parameters are not written alike.
    let ahead = learn(&stopped(), &["[s]"], |case| render_stopped(case, true));
    assert_eq!(ahead, Err(NoLayout::Uneven("parameter")));
}

#[test]
fn a_table_that_fails_its_own_check_has_no_layout() {
    // A turn that starts in a thought, with no row that opens it.
    let invalid = Format::new("invalid")
        .terminal("open", "<c")
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("reasoning", "open", "call")
        .calls(CallSyntax::Shaped(intern(tagged())));
    let result = learn(&invalid, &[], |case| Style::default().render(case));
    assert!(matches!(result, Err(NoLayout::Table(_))), "{result:?}");
}

use serde_json::{json, Map, Value};

use super::*;
use crate::shape::{intern, Pattern};

const OPENER: &str = "<|head|>assistant";
const THINK: &str = "<|via|>notes<|body|>";
const REPLY: &str = "<|via|>reply<|body|>";
const HEADER: &str = "<|via|>tool<|body|>";

/// A neutral table of the shape a response-template compiler gives a template whose calls are
/// addressed to a tool: a framing state outside every field, the thought, the answer and the calls
/// each entered from it and left back to it, and the turn's anchor. The thought has three closes,
/// the answer three, the calls two; `<|done|>` and `<eos>` are stops, the calls' last close among
/// them.
fn table(next_key_closes: bool) -> Format {
    table_stopping_at(next_key_closes, &["<|done|>", "<eos>"])
}

/// [`table`] with `stops` the tokens the engine stops on.
fn table_stopping_at(next_key_closes: bool, stops: &[&str]) -> Format {
    let thought_closes = [
        "<|stop|><|head|>assistant",
        "<|halt|><|head|>assistant",
        OPENER,
    ];
    let mut format = Format::new("neutral")
        .terminal(THINK, THINK)
        .terminal("<|stop|><|head|>assistant", "<|stop|><|head|>assistant")
        .terminal("<|halt|><|head|>assistant", "<|halt|><|head|>assistant")
        .terminal(OPENER, OPENER)
        .terminal(REPLY, REPLY)
        .terminal("<|done|>", "<|done|>")
        .terminal("<|stop|>", "<|stop|>")
        .terminal("<eos>", "<eos>")
        .terminal(" to=", " to=")
        .state("outside", Emits::Content)
        .state("thinking", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("tool_calls", Emits::Arguments)
        .transition("outside", THINK, "thinking");
    for close in thought_closes {
        format = format.transition("thinking", close, "outside");
    }
    format
        .transition("outside", REPLY, "content")
        .transition("content", "<|done|>", "outside")
        .transition("content", "<|stop|>", "outside")
        .transition("content", "<eos>", "outside")
        .transition("outside", " to=", "tool_calls")
        .transition("tool_calls", "<|stop|><|head|>assistant", "outside")
        .transition("tool_calls", "<eos>", "outside")
        .transition("content", "<|stop|><|head|>assistant", "outside")
        .calls(CallSyntax::Shaped(intern(shape(next_key_closes))))
        .closes_at_stop("tool_calls")
        .framing("outside")
        .turn_anchor(OPENER)
        .stop_tokens(stops.iter().copied())
}

fn lit(text: &str) -> Atom {
    Atom::Lit(text.into())
}

/// The neutral table's call: ` to=`, then the tool's name and the header's close; each parameter
/// the key's tag, a newline and the value, ended by a newline and `</arg>` or, where the next key
/// closes a value, by a newline before the next key.
fn shape(next_key_closes: bool) -> CallShape {
    let run = |class: Class, min: u16, greedy: bool, slot: Slot| Atom::Run {
        class,
        min,
        max: None,
        greedy,
        capture: Some(slot),
    };
    let name = Class::new([('\t', '\r'), (' ', ' '), ('<', '<')], true);
    let mut ends = vec![End {
        consume: Pattern([lit("\n</arg>")].into()),
        ahead: Pattern::default(),
    }];
    if next_key_closes {
        ends.push(End {
            consume: Pattern([lit("\n")].into()),
            ahead: Pattern([lit("<arg name=\"")].into()),
        });
    }
    CallShape {
        header: Pattern([run(name, 1, true, Slot::Name), lit(HEADER)].into()),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    lit("<arg name=\""),
                    run(Class::new([('"', '"')], true), 1, true, Slot::Key),
                    lit("\">\n"),
                    run(Class::any(), 0, false, Slot::Value),
                ]
                .into(),
            ),
            ends: ends.into(),
            values: ValueParser::Text { strip: false },
        },
        trailer: Pattern::default(),
        word: Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
        hold_cap: 8192,
    }
}

fn spelling() -> TurnSpelling {
    table(true).turn_spelling().unwrap()
}

fn tool(name: &str, parameters: Value) -> Tool {
    serde_json::from_value(json!({"type": "function", "function": {
        "name": name, "parameters": parameters
    }}))
    .unwrap()
}

fn weather() -> Tool {
    tool(
        "get_weather",
        json!({"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"type": "string", "enum": ["c", "f"]}
        }, "required": ["city"]}),
    )
}

fn turn<'a>(
    position: TurnPosition,
    tools: &'a [Tool],
    choice: TurnChoice,
    answer: Option<TurnAnswer<'a>>,
) -> Turn<'a> {
    Turn {
        position,
        declared: tools,
        callable: tools,
        choice,
        parallel: true,
        answer,
    }
}

fn json_of(grammar: Option<Grammar>) -> Value {
    grammar.expect("a grammar").to_json()
}

#[test]
fn the_rows_give_every_text_of_a_turn() {
    let spelling = spelling();
    let thought = spelling.thought.as_ref().unwrap();
    assert_eq!(thought.open, THINK);
    // Every close of the thought, longest first, the template's order among equals.
    assert_eq!(
        thought.ends,
        [
            "<|stop|><|head|>assistant",
            "<|halt|><|head|>assistant",
            OPENER
        ]
    );
    assert_eq!(thought.first, "[^<]");
    let answer = spelling.answer.as_ref().unwrap();
    assert_eq!(answer.open, REPLY);
    // A close that holds another (the one a row derives through the turn's opener) adds nothing.
    assert_eq!(answer.closes, ["<|done|>", "<|stop|>", "<eos>"]);
    // The stop no call ends on ends the answer.
    assert_eq!(answer.end.as_deref(), Some("<|done|>"));
    assert_eq!(spelling.call_lead, " to=");
    assert_eq!(spelling.separator, "<|stop|><|head|>assistant");
    assert_eq!(spelling.call_end, "<eos>");
    assert_eq!(spelling.call_terminators, ["<|stop|>", "<eos>"]);
    assert_eq!(spelling.stops, ["<|done|>", "<eos>"]);
}

#[test]
fn a_table_whose_calls_are_not_addressed_has_no_turn() {
    assert!(crate::formats::qwen3(CallSyntax::Json)
        .turn_spelling()
        .is_err());
    assert_eq!(
        table_stopping_at(true, &["<|done|>"])
            .turn_spelling()
            .unwrap_err(),
        "no row out of a call is on a stop token"
    );
    // A value the template strips, or a trailer before the call's close, is not read as raw text
    // between the tags.
    let mut stripped = shape(true);
    if let Arguments::Params { values, .. } = &mut stripped.arguments {
        *values = ValueParser::Text { strip: true };
    }
    let mut trailed = shape(true);
    trailed.trailer = Pattern([lit("</call>")].into());
    for other in [stripped, trailed] {
        let format = table(true).calls(CallSyntax::Shaped(intern(other)));
        assert!(format.turn_spelling().is_err());
    }
}

#[test]
fn tools_the_call_cannot_carry_are_refused() {
    let spelling = spelling();
    assert!(spelling.check_tools(&[weather()]).is_ok());
    for (name, parameters) in [
        ("two words", json!({"type": "object"})),
        ("a<b", json!({"type": "object"})),
        ("x</arg>", json!({"type": "object"})),
        ("f", json!({"type": "string"})),
        ("f", json!({"type": "object", "required": ["missing"]})),
        ("f", json!({"type": "object", "properties": {"a\"b": {}}})),
        (
            "f",
            json!({"type": "object", "properties": {"e": {"enum": []}}}),
        ),
        (
            "f",
            json!({"type": "object", "properties": {"e": {"enum": ["x<eos>"]}}}),
        ),
    ] {
        assert!(
            spelling
                .check_tools(&[tool(name, parameters.clone())])
                .is_err(),
            "{name} {parameters}"
        );
    }
    let wide: Map<String, Value> = (0..=TurnSpelling::MAX_TOOL_PARAMETERS)
        .map(|i| (format!("p{i}"), json!({"type": "string"})))
        .collect();
    let wide = tool("wide", json!({"type": "object", "properties": wide}));
    assert!(TurnSpelling::check_parameter_counts(&[weather()]).is_ok());
    assert!(TurnSpelling::check_parameter_counts(&[wide]).is_err());
}

#[test]
fn the_prompt_says_where_the_turn_starts() {
    let format = table(true);
    let at = |prompt: &str, continued: bool| format.turn_position(prompt, continued);
    let user = "<|head|>user<|body|>hi<|stop|>";
    assert_eq!(at(&format!("{user}{OPENER}"), false), TurnPosition::Start);
    assert_eq!(
        at(&format!("{user}{OPENER}{THINK}Let me"), true),
        TurnPosition::InThought
    );
    assert_eq!(
        at(&format!("{user}{OPENER}{THINK}Plan.<|stop|>"), true),
        TurnPosition::InThought,
        "a close of the answer closes no thought"
    );
    assert_eq!(
        at(&format!("{user}{OPENER}{REPLY}The answer"), true),
        TurnPosition::InAnswer
    );
    for tail in ["Hello", "<|via|>rep", " to=f<|via|>tool<|body|>x"] {
        assert_eq!(
            at(&format!("{user}{OPENER}{tail}"), true),
            TurnPosition::Elsewhere,
            "{tail}"
        );
    }
    // A continued message whose thought the prompt closed is past its thought; a new turn after
    // a history turn's thought is not.
    let closed = format!("{user}{OPENER}{THINK}Plan.<|stop|><|head|>assistant");
    assert_eq!(at(&closed, true), TurnPosition::AfterThought);
    assert_eq!(at(&closed, false), TurnPosition::Start);
    let after_answer = format!("{user}{OPENER}{REPLY}Hi<|stop|><|head|>assistant");
    assert_eq!(at(&after_answer, true), TurnPosition::Start);
    assert_eq!(at("no opener at all", false), TurnPosition::Elsewhere);
}

#[test]
fn a_turn_without_tools_is_the_thought_then_an_answer() {
    let spelling = spelling();
    let grammar =
        json_of(spelling.grammar(&turn(TurnPosition::Start, &[], TurnChoice::Auto, None)));
    assert_eq!(
        grammar,
        json!({"type": "sequence", "elements": [
            {"type": "tag", "begin": THINK,
             "content": {"type": "sequence", "elements": [
                 {"type": "regex", "pattern": "[^<]"},
                 {"type": "any_text", "excludes": ["<|done|>", "<eos>"]},
             ]},
             "end": ["<|stop|><|head|>assistant", "<|halt|><|head|>assistant", OPENER]},
            {"type": "sequence", "elements": [
                {"type": "const_string", "value": REPLY},
                {"type": "any_text", "excludes": ["<|done|>", "<|stop|>", "<eos>"]},
                {"type": "optional", "content": {"type": "const_string", "value": "<|done|>"}},
            ]},
        ]})
    );
    // Inside a thought the prompt opened, the thought goes on and may close at once.
    let inside =
        json_of(spelling.grammar(&turn(TurnPosition::InThought, &[], TurnChoice::Auto, None)));
    assert_eq!(inside["elements"][0]["begin"], "");
    assert_eq!(inside["elements"][0]["content"]["type"], "any_text");
    // Past a thought the prompt closed, no second one.
    let after = json_of(spelling.grammar(&turn(
        TurnPosition::AfterThought,
        &[],
        TurnChoice::Auto,
        None,
    )));
    assert_eq!(after, grammar["elements"][1]);
    // Inside an answer: the rest of a free answer, and nothing for a format or a forced call.
    let rest =
        json_of(spelling.grammar(&turn(TurnPosition::InAnswer, &[], TurnChoice::Auto, None)));
    assert_eq!(
        rest["elements"][0]["excludes"],
        json!(["<|done|>", "<|stop|>", "<eos>"])
    );
    let schema = json!({"type": "object"});
    let json_answer = Some(TurnAnswer::Json(&schema));
    assert!(spelling
        .grammar(&turn(
            TurnPosition::InAnswer,
            &[],
            TurnChoice::Auto,
            json_answer
        ))
        .is_none());
    assert!(spelling
        .grammar(&turn(TurnPosition::Elsewhere, &[], TurnChoice::Auto, None))
        .is_none());
    // A call asked for and nothing to call: no grammar.
    assert!(spelling
        .grammar(&turn(TurnPosition::Start, &[], TurnChoice::Required, None))
        .is_none());
}

#[test]
fn an_answer_format_is_the_answers_body() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let body = |answer: TurnAnswer<'_>| {
        json_of(spelling.grammar(&turn(
            TurnPosition::Start,
            &[],
            TurnChoice::Auto,
            Some(answer),
        )))["elements"][1]["elements"][1]
            .clone()
    };
    // A JSON answer's strings hold none of the answer's closes.
    assert_eq!(
        body(TurnAnswer::Json(&schema)),
        json!({"type": "json_schema", "json_schema": schema,
               "excludes": ["<|done|>", "<|stop|>", "<eos>"]})
    );
    assert_eq!(
        body(TurnAnswer::Regex("(yes|no)")),
        json!({"type": "regex", "pattern": "(yes|no)"})
    );
    assert_eq!(
        body(TurnAnswer::Ebnf("root ::= \"y\"")),
        json!({"type": "grammar", "grammar": "root ::= \"y\""})
    );
    assert!(TurnAnswer::Ebnf("root ::= x").composes());
    assert!(!TurnAnswer::Ebnf("start: \"y\"").composes());
}

#[test]
fn the_choice_shapes_the_output() {
    let spelling = spelling();
    let tools = [weather(), tool("get_time", json!({"type": "object"}))];
    let output = |choice: TurnChoice, parallel: bool| {
        let mut asked = turn(TurnPosition::Start, &tools, choice, None);
        asked.parallel = parallel;
        json_of(spelling.grammar(&asked))["elements"][1].clone()
    };
    // auto: calls or the answer; the calls may name an undeclared tool too.
    let auto = output(TurnChoice::Auto, true);
    assert_eq!(auto["type"], "or");
    let calls = &auto["elements"][0]["elements"];
    assert_eq!(
        calls[0]["elements"].as_array().unwrap().len(),
        3,
        "two declared, one other"
    );
    assert_eq!(
        calls[0]["elements"][0]["elements"][0]["value"],
        format!(" to=get_weather{HEADER}")
    );
    assert_eq!(
        calls[0]["elements"][2]["elements"][0]["elements"][1]["type"],
        "regex"
    );
    assert_eq!(calls[1]["type"], "star");
    assert_eq!(
        calls[1]["content"]["elements"][0]["value"],
        "<|stop|><|head|>assistant"
    );
    assert_eq!(calls[2], json!({"type": "const_string", "value": "<eos>"}));
    assert_eq!(auto["elements"][1]["elements"][0]["value"], REPLY);
    // required: calls to the declared tools only.
    let required = output(TurnChoice::Required, true);
    assert_eq!(required["type"], "sequence");
    assert_eq!(
        required["elements"][0]["elements"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    // one call, and parallel calls off: no repeat.
    for single in [
        output(TurnChoice::One, true),
        output(TurnChoice::Required, false),
    ] {
        let elements = single["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(
            elements[1],
            json!({"type": "const_string", "value": "<eos>"})
        );
    }
}

/// The names a regex matches whole.
fn matches(pattern: &str, name: &str) -> bool {
    regex::Regex::new(&format!("^(?:{pattern})$"))
        .unwrap()
        .is_match(name)
}

#[test]
fn an_undeclared_call_names_anything_but_a_declared_tool() {
    let pattern = other_names(&["get", "get_weather", "a.b"]);
    for name in ["get", "get_weather", "a.b"] {
        assert!(!matches(&pattern, name), "{name}");
    }
    for name in [
        "ge",
        "gett",
        "get_weathe",
        "get_weather_v2",
        "x",
        "a.bc",
        "aXb",
        "a",
    ] {
        assert!(matches(&pattern, name), "{name}");
    }
    for name in ["", "two words", "a<b"] {
        assert!(!matches(&pattern, name), "{name:?}");
    }
}

/// The tags of a call's arguments, by their begins, in the order a list writes them.
fn begins(grammar: &Value, out: &mut Vec<String>) {
    match grammar {
        Value::Object(map) => {
            if map.get("type") == Some(&json!("tag")) {
                out.push(map["begin"].as_str().unwrap().to_string());
            }
            for value in map.values() {
                begins(value, out);
            }
        }
        Value::Array(values) => values.iter().for_each(|value| begins(value, out)),
        _ => {}
    }
}

#[test]
fn arguments_take_the_schemas_order_or_the_names() {
    let spelling = spelling();
    // Declared out of name order: two lists, the schema's and the names'.
    let schema = json!({"type": "object", "properties": {
        "zeta": {"type": "string"}, "alpha": {"type": "integer"}
    }, "required": ["zeta"]});
    let arguments = spelling.arguments(&schema).to_json();
    assert_eq!(arguments["elements"][1]["value"], "<arg name=\"");
    let lists = arguments["elements"][2]["elements"].as_array().unwrap();
    assert_eq!(lists.len(), 2);
    let mut first = Vec::new();
    begins(&lists[0], &mut first);
    let mut second = Vec::new();
    begins(&lists[1], &mut second);
    assert_eq!(first[0], "zeta\">\n");
    assert_eq!(second[0], "alpha\">\n");
    // Sorted already: one list.
    let sorted = json!({"type": "object", "properties": {"a": {}, "b": {}}});
    let arguments = spelling.arguments(&sorted).to_json();
    assert_eq!(arguments["type"], "optional", "none required");
    // No property and no other key: no argument at all.
    let none = json!({"type": "object", "properties": {}});
    assert_eq!(
        spelling.arguments(&none),
        Grammar::ConstString(String::new())
    );
}

#[test]
fn a_value_is_raw_text_a_constant_or_json() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {
        "s": {"type": "string"},
        "n": {"type": "integer"},
        "e": {"enum": ["ok", "bad<|stop|>x", 3]},
        "p": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "q": {"type": "string", "pattern": "^(?!x).+$"},
        "t": {"type": "tuple"}
    }, "required": ["s", "n", "e", "p", "q", "t"]});
    let closed = spelling.parameters(&schema, Syntax::Closed);
    let value = |key: &str| {
        let parameter = closed
            .declared
            .iter()
            .find(|parameter| parameter.key == key)
            .unwrap();
        parameter.grammar.to_json()["elements"][1]["content"].clone()
    };
    let raw = json!(["<|stop|>", "<eos>", "</arg>"]);
    assert_eq!(value("s"), json!({"type": "any_text", "excludes": raw}));
    assert_eq!(
        value("n"),
        json!({"type": "json_schema", "json_schema": {"type": "integer"},
               "excludes": ["<|stop|>", "<eos>"]})
    );
    // A string constant holding a word a value cannot hold is left out; others are themselves.
    let e = value("e");
    assert_eq!(
        e["elements"][0],
        json!({"type": "const_string", "value": "ok"})
    );
    assert_eq!(e["elements"][1]["json_schema"], json!({"const": 3}));
    assert_eq!(
        value("p"),
        json!({"type": "regex", "pattern": "^[A-Z]{3}$"})
    );
    // A pattern xgrammar does not compile leaves the value free text.
    assert_eq!(value("q"), json!({"type": "any_text", "excludes": raw}));
    // A type name no JSON type is takes any value.
    assert_eq!(value("t")["elements"].as_array().unwrap().len(), 6);
    // Where the next key closes a value, a raw value holds no next key either.
    let joined = spelling.raw_excludes(Syntax::Joined);
    assert_eq!(joined.last().unwrap(), "\n<arg name=\"");
}

#[test]
fn a_json_value_also_takes_its_keys_by_name() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {
        "when": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}},
        "sorted": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    }});
    let closed = spelling.parameters(&schema, Syntax::Closed);
    let when = closed.declared[0].grammar.to_json()["elements"][1]["content"].clone();
    assert_eq!(when["type"], "or");
    let keys = |schema: &Value| {
        schema["json_schema"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(&when["elements"][0]), ["start", "end"]);
    assert_eq!(keys(&when["elements"][1]), ["end", "start"]);
    let sorted = closed.declared[1].grammar.to_json()["elements"][1]["content"].clone();
    assert_eq!(
        sorted["type"], "json_schema",
        "one form when the schema lists them by name"
    );
}

#[test]
fn free_keys_and_closed_values_where_the_next_key_closes_none() {
    // A schema that declares nothing takes free keys; without the next key closing a value each
    // parameter carries its own opening and close, after an optional newline.
    let closed_table = table(false).turn_spelling().unwrap();
    let arguments = closed_table.arguments(&json!({})).to_json();
    assert_eq!(arguments["elements"][0]["type"], "star");
    let free = &arguments["elements"][0]["content"]["elements"][1];
    assert_eq!(free["elements"][0]["value"], "<arg name=\"");
    assert_eq!(
        free["elements"][1],
        json!({"type": "regex", "pattern": FREE_KEY})
    );
    assert_eq!(free["elements"][4]["value"], "\n</arg>");
    // Where it does, a free parameter but the last ends at the next key.
    let lenient = spelling().arguments(&json!({})).to_json();
    assert_eq!(lenient["type"], "optional");
    let list = &lenient["content"]["elements"][2]["elements"];
    assert_eq!(list[0]["type"], "star");
    assert_eq!(list[0]["content"]["elements"][4]["type"], "or");
}

#[test]
fn halving_keeps_a_long_optional_list_small() {
    // Twelve optional parameters: each value is written O(log n) times, not O(n).
    let properties: Map<String, Value> = (0..12)
        .map(|i| (format!("p{i:02}"), json!({"type": "string"})))
        .collect();
    let schema = json!({"type": "object", "properties": properties});
    let text = spelling().arguments(&schema).to_json().to_string();
    let count = text.matches("p05\\\">").count();
    assert!(count <= 2 * (1 + 4) * 2, "{count}");
    assert!(count >= 2);
}

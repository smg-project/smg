use serde_json::{json, Map, Value};

use super::*;
use crate::shape::{intern, Pattern};

const OPENER: &str = "<|head|>assistant";
const THINK: &str = "<|via|>notes<|body|>";
const REPLY: &str = "<|via|>reply<|body|>";
const HEADER: &str = "<|via|>tool<|body|>";

/// A neutral table with addressed calls and distinct thought, answer and argument states.
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

/// An addressed call with raw tagged parameters.
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

fn spelling() -> OutputSpelling {
    table(true).output_spelling().unwrap()
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

fn output<'a>(
    position: OutputPosition,
    tools: &'a [Tool],
    choice: OutputChoice,
    answer: Option<OutputAnswer<'a>>,
) -> Output<'a> {
    Output {
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
fn the_rows_give_every_text_of_the_output() {
    let spelling = spelling();
    let thought = spelling.thought.as_ref().unwrap();
    assert_eq!(thought.open, THINK);
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
    assert_eq!(answer.closes, ["<|done|>", "<|stop|>"]);
    assert_eq!(answer.end.as_deref(), Some("<|done|>"));
    assert_eq!(spelling.call_lead, " to=");
    assert_eq!(spelling.separator, "<|stop|><|head|>assistant");
    assert_eq!(spelling.call_end, "<eos>");
    assert_eq!(spelling.call_terminators, ["<|stop|>", "<eos>"]);
    assert_eq!(spelling.stops, ["<|done|>", "<eos>"]);
}

#[test]
fn a_table_whose_calls_are_not_addressed_has_no_output_spelling() {
    assert!(crate::formats::qwen3(CallSyntax::Json)
        .output_spelling()
        .is_err());
    assert_eq!(
        table_stopping_at(true, &["<|done|>"])
            .output_spelling()
            .unwrap_err(),
        "no row out of a call is on a stop token"
    );
    let mut stripped = shape(true);
    if let Arguments::Params { values, .. } = &mut stripped.arguments {
        *values = ValueParser::Text { strip: true };
    }
    let mut trailed = shape(true);
    trailed.trailer = Pattern([lit("</call>")].into());
    for other in [stripped, trailed] {
        let format = table(true).calls(CallSyntax::Shaped(intern(other)));
        assert!(format.output_spelling().is_err());
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
    let wide: Map<String, Value> = (0..=OutputSpelling::MAX_TOOL_PARAMETERS)
        .map(|i| (format!("p{i}"), json!({"type": "string"})))
        .collect();
    let wide = tool("wide", json!({"type": "object", "properties": wide}));
    assert!(OutputSpelling::check_parameter_counts(&[weather()]).is_ok());
    assert!(OutputSpelling::check_parameter_counts(&[wide]).is_err());
}

#[test]
fn names_a_call_cannot_carry_are_refused_for_any_tool() {
    let spelling = spelling();
    let unsupported = json!({"properties": {"mode": {"enum": []}}, "required": ["other"]});
    assert!(spelling.check_names(&[tool("f", unsupported)]).is_ok());
    let object = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let error = spelling
        .check_names(&[tool("has space", object)])
        .unwrap_err();
    assert!(error.contains("cannot be read back"), "{error}");
    let parameters = json!({"type": "object", "properties": {"say \"hi\"": {}}});
    let error = spelling.check_names(&[tool("f", parameters)]).unwrap_err();
    assert!(error.contains("parameter name"), "{error}");
}

#[test]
fn the_prompt_says_where_the_output_starts() {
    let format = table(true);
    let at = |prompt: &str, continued: bool| format.output_position(prompt, continued);
    let user = "<|head|>user<|body|>hi<|stop|>";
    let closed = format!("{user}{OPENER}{THINK}Plan.<|stop|><|head|>assistant");
    let cases = [
        (format!("{user}{OPENER}"), false, OutputPosition::Start),
        (
            format!("{user}{OPENER}{THINK}Let me"),
            true,
            OutputPosition::InThought,
        ),
        (closed, true, OutputPosition::AfterThought),
        (
            format!("{user}{OPENER}{REPLY}The answer"),
            true,
            OutputPosition::InAnswer,
        ),
        (
            format!("{user}{OPENER} to=f{HEADER}"),
            true,
            OutputPosition::InCall,
        ),
        (
            format!("{user}{OPENER}Hello"),
            true,
            OutputPosition::Elsewhere,
        ),
    ];
    for (prompt, continued, expected) in cases {
        assert_eq!(at(&prompt, continued), expected, "{prompt}");
    }
}

#[test]
fn an_output_grammar_without_tools_is_the_thought_then_an_answer() {
    let spelling = spelling();
    let at = |position, choice, answer| spelling.grammar(&output(position, &[], choice, answer));
    let grammar = json_of(at(OutputPosition::Start, OutputChoice::Auto, None));
    assert_eq!(grammar["elements"][0]["begin"], THINK);
    assert_eq!(
        grammar["elements"][0]["content"]["elements"][0]["pattern"],
        "[^<]"
    );
    assert_eq!(grammar["elements"][1]["elements"][0]["value"], REPLY);
    assert_eq!(
        grammar["elements"][1]["elements"][1]["excludes"],
        json!(["<|done|>", "<|stop|>"])
    );
    assert_eq!(grammar["elements"][1]["elements"][2]["type"], "optional");

    let inside = json_of(at(OutputPosition::InThought, OutputChoice::Auto, None));
    assert_eq!(inside["elements"][0]["begin"], "");
    assert_eq!(inside["elements"][0]["content"]["type"], "any_text");
    let after = json_of(at(OutputPosition::AfterThought, OutputChoice::Auto, None));
    assert_eq!(after, grammar["elements"][1]);

    let rest = json_of(at(OutputPosition::InAnswer, OutputChoice::Required, None));
    assert_eq!(
        rest["elements"][0]["excludes"],
        json!(["<|done|>", "<|stop|>"])
    );
    let schema = json!({"type": "object"});
    assert!(at(
        OutputPosition::InAnswer,
        OutputChoice::Auto,
        Some(OutputAnswer::Json(&schema))
    )
    .is_none());
    assert!(at(OutputPosition::Elsewhere, OutputChoice::Auto, None).is_none());
    assert!(at(OutputPosition::Start, OutputChoice::Required, None).is_none());
}

#[test]
fn an_answer_format_is_the_answers_body() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let body = |answer: OutputAnswer<'_>| {
        json_of(spelling.grammar(&output(
            OutputPosition::Start,
            &[],
            OutputChoice::Auto,
            Some(answer),
        )))["elements"][1]["elements"][1]
            .clone()
    };
    // A JSON answer is the schema's JSON alone.
    assert_eq!(
        body(OutputAnswer::Json(&schema)),
        json!({"type": "json_schema", "json_schema": schema})
    );
    assert_eq!(
        body(OutputAnswer::Regex("(yes|no)")),
        json!({"type": "regex", "pattern": "(yes|no)"})
    );
    assert_eq!(
        body(OutputAnswer::Ebnf("root ::= \"y\"")),
        json!({"type": "grammar", "grammar": "root ::= \"y\""})
    );
    assert!(OutputAnswer::Ebnf("root ::= x").composes());
    assert!(!OutputAnswer::Ebnf("start: \"y\"").composes());
}

#[test]
fn the_choice_shapes_the_output() {
    let spelling = spelling();
    let tools = [weather(), tool("get_time", json!({"type": "object"}))];
    let output = |choice: OutputChoice, parallel: bool| {
        let mut asked = output(OutputPosition::Start, &tools, choice, None);
        asked.parallel = parallel;
        json_of(spelling.grammar(&asked))["elements"][1].clone()
    };
    let auto = output(OutputChoice::Auto, true);
    assert_eq!(auto["type"], "or");
    let calls = &auto["elements"][0]["elements"];
    assert_eq!(calls[0]["elements"].as_array().unwrap().len(), 3);
    assert_eq!(
        calls[0]["elements"][0]["elements"][0]["value"],
        format!(" to=get_weather{HEADER}")
    );
    assert_eq!(calls[1]["type"], "star");
    let end = json!({"type": "optional", "content": {"type": "const_string", "value": "<eos>"}});
    assert_eq!(calls[2], end);
    assert_eq!(auto["elements"][1]["elements"][0]["value"], REPLY);
    let required = output(OutputChoice::Required, true);
    assert_eq!(required["type"], "sequence");
    assert_eq!(
        required["elements"][0]["elements"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for single in [
        output(OutputChoice::One, true),
        output(OutputChoice::Required, false),
    ] {
        let elements = single["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[1], end);
    }
}

#[test]
fn inside_a_call_its_rest_is_free_and_later_calls_are_held() {
    let spelling = spelling();
    let tools = [weather(), tool("get_time", json!({"type": "object"}))];
    let rest = |choice: OutputChoice, parallel: bool| {
        let mut asked = output(OutputPosition::InCall, &tools, choice, None);
        asked.parallel = parallel;
        json_of(spelling.grammar(&asked))
    };
    let free = json!({"type": "any_text", "excludes": ["<|stop|><|head|>assistant"]});
    let end = json!({"type": "optional", "content": {"type": "const_string", "value": "<eos>"}});
    let auto = rest(OutputChoice::Auto, true);
    let elements = auto["elements"].as_array().unwrap();
    assert_eq!(elements.len(), 3);
    assert_eq!(elements[0], free);
    assert_eq!(elements[1]["type"], "star");
    assert_eq!(elements[2], end);
    for single in [
        rest(OutputChoice::One, true),
        rest(OutputChoice::Required, false),
    ] {
        assert_eq!(single, json!({"type": "sequence", "elements": [free, end]}));
    }
    assert_eq!(
        json_of(spelling.grammar(&output(
            OutputPosition::InCall,
            &[],
            OutputChoice::Auto,
            None
        ))),
        json!({"type": "sequence", "elements": [free, end]})
    );
}

#[test]
fn auto_calls_include_one_undeclared_name_grammar() {
    let tools = [weather(), tool("get_time", json!({"type": "object"}))];
    let grammar = json_of(spelling().grammar(&output(
        OutputPosition::Start,
        &tools,
        OutputChoice::Auto,
        None,
    )));
    let calls = &grammar["elements"][1]["elements"][0]["elements"][0]["elements"];
    assert_eq!(calls.as_array().unwrap().len(), tools.len() + 1);
    assert_eq!(
        calls[tools.len()]["elements"][0]["elements"][1]["type"],
        "grammar"
    );
}

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
    let schema = json!({"type": "object", "properties": {
        "zeta": {"type": "string"}, "alpha": {"type": "integer"}
    }, "required": ["zeta"]});
    let arguments = spelling.arguments(&schema).to_json();
    let lists = arguments["elements"][2]["elements"].as_array().unwrap();
    assert_eq!(lists.len(), 2);
    let mut first = Vec::new();
    begins(&lists[0], &mut first);
    let mut second = Vec::new();
    begins(&lists[1], &mut second);
    assert_eq!(first[0], "zeta\">\n");
    assert_eq!(second[0], "alpha\">\n");
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
    assert_eq!(value("q"), json!({"type": "any_text", "excludes": raw}));
    assert_eq!(value("t")["elements"].as_array().unwrap().len(), 6);
    let joined = spelling.raw_excludes(Syntax::Joined);
    assert_eq!(joined[3], "\n<arg name=\"");
}

#[test]
fn a_raw_value_keeps_out_only_the_calls_own_words() {
    let spelling = table(true)
        .marker_tokens([
            "<arg name=\"",
            "<|head|>",
            "<|via|>",
            "<|body|>",
            "<|done|>",
            "</arg>",
        ])
        .output_spelling()
        .unwrap();
    let schema = json!({"type": "object", "properties": {
        "s": {"type": "string"},
        "e": {"enum": ["ok", "a<|via|>b"]}
    }, "required": ["s", "e"]});
    let joined = ["<|stop|>", "<eos>", "</arg>", "\n<arg name=\""];
    for syntax in [Syntax::Joined, Syntax::Last] {
        assert_eq!(spelling.raw_excludes(syntax), joined, "{syntax:?}");
    }
    assert_eq!(spelling.raw_excludes(Syntax::Closed), joined[..3]);
    let parameters = spelling.parameters(&schema, Syntax::Closed);
    let e = parameters.declared[1].grammar.to_json()["elements"][1]["content"].clone();
    assert_eq!(
        e["elements"][1],
        json!({"type": "const_string", "value": "a<|via|>b"})
    );
}

#[test]
fn a_json_value_takes_its_keys_in_the_schemas_order() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {
        "when": {"type": "object", "properties": {
            "start": {"type": "string"}, "end": {"type": "string"}
        }}
    }});
    let closed = spelling.parameters(&schema, Syntax::Closed);
    let when = closed.declared[0].grammar.to_json()["elements"][1]["content"].clone();
    assert_eq!(when["type"], "json_schema", "one form");
    let keys: Vec<&String> = when["json_schema"]["properties"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(keys, ["start", "end"]);
}

#[test]
fn free_keys_and_closed_values_where_the_next_key_closes_none() {
    let closed_table = table(false).output_spelling().unwrap();
    let arguments = closed_table.arguments(&json!({})).to_json();
    assert_eq!(arguments["elements"][0]["type"], "star");
    let free = &arguments["elements"][0]["content"]["elements"][1];
    assert_eq!(free["elements"][0]["value"], "<arg name=\"");
    assert_eq!(
        free["elements"][1],
        json!({"type": "regex", "pattern": FREE_KEY})
    );
    assert_eq!(free["elements"][4]["value"], "\n</arg>");
    let lenient = spelling().arguments(&json!({})).to_json();
    assert_eq!(lenient["type"], "optional");
    let list = &lenient["content"]["elements"][2]["elements"];
    assert_eq!(list[0]["type"], "star");
    assert_eq!(list[0]["content"]["elements"][4]["type"], "or");
}

#[test]
fn the_length_is_the_written_tags_without_building_it() {
    let tools = [
        weather(),
        tool(
            "plan",
            json!({"type": "object", "properties": {
                "place": {"$ref": "#/$defs/Spot"},
                "level": {"enum": [1, "two"]}
            }, "required": ["place"], "additionalProperties": {"type": "number"},
            "$defs": {"Spot": {"type": "object", "properties": {"lon": {"type": "number"}}}}}),
        ),
    ];
    let schema = json!({"type": "object", "properties": {"z": {"type": "integer"}}});
    let cases = [
        output(OutputPosition::Start, &tools, OutputChoice::Auto, None),
        output(OutputPosition::InCall, &tools, OutputChoice::Required, None),
        output(
            OutputPosition::Start,
            &[],
            OutputChoice::Auto,
            Some(OutputAnswer::Json(&schema)),
        ),
    ];
    for next_key_closes in [true, false] {
        let spelling = table(next_key_closes).output_spelling().unwrap();
        for asked in &cases {
            let grammar = spelling.grammar(asked).unwrap();
            let text = grammar.payload_text();
            assert_eq!(spelling.payload_len(asked), Some(text.len()), "{asked:?}");
            assert_eq!(text, grammar.payload().to_string());
        }
    }
}

fn documents(grammar: &Value, out: &mut Vec<Value>) {
    match grammar {
        Value::Object(map) => {
            if map.get("type") == Some(&json!("json_schema")) {
                out.push(map["json_schema"].clone());
                return;
            }
            map.values().for_each(|value| documents(value, out));
        }
        Value::Array(values) => values.iter().for_each(|value| documents(value, out)),
        _ => {}
    }
}

#[test]
fn a_json_value_carries_only_the_definitions_it_reaches() {
    let schema = json!({"type": "object", "properties": {
        "via": {"type": "object", "properties": {"x": {"$ref": "#/$defs/A"}}}
    }, "required": ["via"],
    "$defs": {
        "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
        "B": {"type": "string"},
        "Unused": {"type": "boolean"}
    }});
    let parameters = spelling().parameters(&schema, Syntax::Closed);
    let mut found = Vec::new();
    documents(&parameters.declared[0].grammar.to_json(), &mut found);
    assert_eq!(
        found[0]["$defs"],
        json!({
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "string"}
        })
    );
}

#[test]
fn annotations_are_left_out_of_json_values() {
    let spelling = spelling();
    let schema = json!({"type": "object", "properties": {
        "spec": {"type": "object", "title": "Spec", "description": "what to do",
                 "examples": [{"description": "kept"}],
                 "properties": {
                     "description": {"type": "string", "description": "a property named so"},
                     "n": {"type": "integer", "examples": [1], "minimum": 0}
                 },
                 "default": {"n": 1}}
    }, "required": ["spec"]});
    let closed = spelling.parameters(&schema, Syntax::Closed);
    let mut found = Vec::new();
    documents(&closed.declared[0].grammar.to_json(), &mut found);
    assert_eq!(
        found,
        [json!({"type": "object", "properties": {
            "description": {"type": "string"},
            "n": {"type": "integer", "minimum": 0}
        }, "default": {"n": 1}})]
    );
}

#[test]
fn a_strict_tool_whose_schema_the_calls_hold_loosely_is_told() {
    let spelling = spelling();
    let strict = |parameters: Value| -> Tool {
        serde_json::from_value(json!({"type": "function", "function": {
            "name": "f", "strict": true, "parameters": parameters
        }}))
        .unwrap()
    };
    let unenforced = |tool: &Tool, position: OutputPosition, choice: OutputChoice| {
        let tools = std::slice::from_ref(tool);
        spelling.unenforced(&output(position, tools, choice, None))
    };
    let at_start = |tool: &Tool| unenforced(tool, OutputPosition::Start, OutputChoice::Auto);
    let object = |properties: Value| json!({"type": "object", "properties": properties});
    for (parameters, why) in [
        (
            object(json!({"n": {"type": "object", "properties": {
                "q": {"type": "string", "pattern": "\\p{L}"}
            }}})),
            "a pattern xgrammar",
        ),
        (
            object(json!({"t": {"type": "tuple"}})),
            "parameter 't': a type name",
        ),
        (
            object(json!({"a": {"allOf": [{"type": "string"}, {"maxLength": 3}]}})),
            "an allOf",
        ),
        (
            object(json!({"r": {"$ref": "#/$defs/Missing"}})),
            "a $ref that does not resolve",
        ),
        (
            object(json!({"m": {"minimum": 3}})),
            "keywords without a type",
        ),
        (
            object(json!({"d": {"type": "string", "format": "date"}})),
            "the format of parameter 'd'",
        ),
        (
            object(json!({"d": {"type": ["string", "null"], "maxLength": 9}})),
            "the maxLength",
        ),
        (
            json!({"type": "object", "properties": {}, "required": ["x"]}),
            "required parameter 'x'",
        ),
        (
            json!({"type": "object", "properties": {"e": {"enum": []}}, "required": ["e"]}),
            "required parameter 'e', which no value",
        ),
        (
            json!({"type": "object", "properties": {"e": {"const": "a</arg>"}}, "required": ["e"]}),
            "required parameter 'e', which no value",
        ),
        (json!({"type": "string"}), "not an object"),
        (
            json!({"anyOf": [{"type": "object"}, {"type": "object"}]}),
            "its parameters' anyOf",
        ),
        (
            json!({"type": "object", "additionalProperties": {"type": "tuple"}}),
            "a free parameter",
        ),
    ] {
        let tool = strict(parameters.clone());
        let told = at_start(&tool).unwrap_or_else(|| panic!("{parameters}"));
        assert!(told.starts_with("tool 'f' is strict"), "{told}");
        assert!(told.contains(why), "{parameters}: {told}");
    }
    let held = strict(json!({"type": "object", "properties": {
        "s": {"type": "string"}, "p": {"type": "string", "pattern": "^[a-z]+$"},
        "e": {"enum": ["a", "b"]}, "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "j": {"type": "array", "items": {"type": "string"}}, "free": {}
    }, "required": ["s", "e"], "additionalProperties": false}));
    assert_eq!(at_start(&held), None);
    assert_eq!(at_start(&strict(json!({}))), None);
    let loose = strict(object(json!({"t": {"type": "tuple"}})));
    let mut lax = loose.clone();
    lax.function.strict = None;
    assert_eq!(at_start(&lax), None);
    for position in [OutputPosition::InAnswer, OutputPosition::Elsewhere] {
        assert_eq!(unenforced(&loose, position, OutputChoice::Auto), None);
    }
    assert_eq!(
        unenforced(&loose, OutputPosition::InCall, OutputChoice::One),
        None
    );
    assert!(unenforced(&loose, OutputPosition::InCall, OutputChoice::Auto).is_some());
    assert!(unenforced(&loose, OutputPosition::InThought, OutputChoice::Required).is_some());
}

/// What a request asks of [`Format::turn_grammar`]: `tools` declared and callable under `choice`,
/// with `answer`, the prompt ending at `prompt`.
fn turn<'a>(
    tools: &'a [Tool],
    choice: OutputChoice,
    answer: Option<OutputAnswer<'a>>,
    prompt: &'a str,
) -> Turn<'a> {
    Turn {
        prompt,
        continued: false,
        declared: tools,
        callable: tools,
        choice,
        parallel: true,
        answer,
    }
}

#[test]
fn the_turns_grammar_is_the_spellings_where_the_prompt_leaves_the_model() {
    let format = table(true);
    let tools = [weather()];
    let start = format!("user{OPENER}");
    let built = |choice, answer| {
        json_of(spelling().grammar(&output(OutputPosition::Start, &tools, choice, answer)))
    };
    let auto = format.turn_grammar(&turn(&tools, OutputChoice::Auto, None, &start));
    assert_eq!(json_of(auto.unwrap()), built(OutputChoice::Auto, None));
    // A forced call writes no answer: it leaves the answer's format out, Lark's included.
    let lark = OutputAnswer::Ebnf("start: \"y\"");
    let forced = format.turn_grammar(&turn(&tools, OutputChoice::Required, Some(lark), &start));
    assert_eq!(
        json_of(forced.unwrap()),
        built(OutputChoice::Required, None)
    );
    // An answer in Lark's syntax is refused; one in xgrammar's is held.
    let why = format
        .turn_grammar(&turn(&[], OutputChoice::Auto, Some(lark), &start))
        .unwrap_err();
    assert!(why.contains("Lark"), "{why}");
    let xgrammar = OutputAnswer::Ebnf("root ::= \"y\"");
    let held = format.turn_grammar(&turn(&[], OutputChoice::Auto, Some(xgrammar), &start));
    assert!(held.unwrap().is_some());
    // Inside an answer a forced call starts no grammar, and the table's tags decide.
    let answering = format!("user{OPENER}{REPLY}The answer");
    let inside = format.turn_grammar(&turn(&tools, OutputChoice::Required, None, &answering));
    assert_eq!(inside, Ok(None));
    // A table whose calls spell no turn has no grammar for one.
    let unaddressed = crate::formats::qwen3(CallSyntax::Json);
    let none = unaddressed.turn_grammar(&turn(&tools, OutputChoice::Auto, None, &start));
    assert_eq!(none, Ok(None));
}

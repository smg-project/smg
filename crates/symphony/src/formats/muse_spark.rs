//! Muse Spark: a turn of messages, each addressed to a recipient. The template opens the turn
//! with `<|start|>assistant` and writes every message as ` to=RECIPIENT<|message|>`, its body,
//! and `<|eom|>` (another message follows) or `<|eot|>` (the turn ends): the thought is the
//! message to `self`, the answer the message to `user`, an update before or between calls the
//! message to `commentary`, and a call the message to the tool, whose body is one
//! `<atem:function_calls>` block holding `<atem:invoke name="…">` with one
//! `<atem:parameter name="…">` tag per argument, a string as it is and every other value as JSON.
//! Parallel calls are consecutive messages. As the checkpoint's template renders it for
//! `meta-models/Muse-Spark-Open-816B-A42B`:
//!
//! ```text
//!  to=self<|message|>The user asks about Paris.<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>
//! <atem:invoke name="get_weather">
//! <atem:parameter name="city">Paris</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eot|>
//! ```
//!
//! The checkpoint's `tokenizer_config.json` declares the same reading as a response template,
//! and the table spells it: the text between the messages (the headers, the block around a call,
//! `<|start|>assistant`) is framing the template drops, a call is a call given as data
//! ([`CallSyntax::Shaped`]) whose header names the tool, and the layout the tags write is what
//! the chat template renders.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    shape::{
        self, Arguments, Atom, CallLayout, CallShape, Class, Dialect, End, Layout, Message,
        NameSource, Pattern, Segment, Slot, Spellings, ValueParser, Written,
    },
};

/// The most bytes a call's header holds undecided, the matcher's own default.
const HOLD_CAP_BYTES: u32 = 8192;

/// The Muse Spark table.
pub fn muse_spark() -> Format {
    Format::new("muse_spark")
        .terminal("thought", "to=self<|message|>")
        .terminal("answer", "to=user<|message|>")
        .terminal("commentary", "to=commentary<|message|>")
        .terminal("eom", "<|eom|>")
        .terminal("eot", "<|eot|>")
        .terminal("call_open", "<atem:invoke")
        .terminal("call_close", "</atem:invoke>")
        .state("message", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("message", "thought", "reasoning")
        .transition("message", "answer", "content")
        .transition("message", "commentary", "content")
        .transition("reasoning", "eom", "message")
        .transition("reasoning", "eot", "message")
        .transition("content", "eom", "message")
        .transition("content", "eot", "message")
        .transition("message", "call_open", "call")
        .transition("call", "call_close", "message")
        .transition("call", "eom", "message")
        .transition("call", "eot", "message")
        .calls(CallSyntax::Shaped(shape::intern(call_shape())))
        .framing("message")
        .closes_at_stop("call")
        .turn_anchor("<|start|>assistant")
        .stop_tokens(["<|end_of_text|>", "<|eot|>"])
        .with_layout(layout())
}

/// A call as the template's reader finds it: after `<atem:invoke`, a header that skips other
/// attributes and captures the tool's name (`\b[^>]*?\bname="(?P<name>[^"]+)">`); the parameters
/// as `finditer` finds `<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)`, each
/// value up to `</atem:parameter>`, read as JSON where it is JSON and as text otherwise; nothing
/// before the invoke's close.
fn call_shape() -> CallShape {
    let not_gt = || none_of(">");
    CallShape {
        header: Pattern(
            [
                Atom::Boundary,
                run(not_gt(), 0, false, None),
                Atom::Boundary,
                Atom::Lit("name=\"".into()),
                run(none_of("\""), 1, true, Some(Slot::Name)),
                Atom::Lit("\">".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Params {
            tag: Pattern(
                [
                    Atom::Lit("<atem:parameter".into()),
                    Atom::Boundary,
                    run(not_gt(), 0, false, None),
                    Atom::Boundary,
                    Atom::Lit("name=\"".into()),
                    run(none_of("\""), 1, true, Some(Slot::Key)),
                    Atom::Lit("\"".into()),
                    run(not_gt(), 0, false, None),
                    Atom::Lit(">".into()),
                    run(Class::any(), 0, false, Some(Slot::Value)),
                ]
                .into(),
            ),
            ends: [End {
                consume: Pattern([Atom::Lit("</atem:parameter>".into())].into()),
                ahead: Pattern::default(),
            }]
            .into(),
            values: ValueParser::Json {
                dialect: Dialect::default(),
                allow_non_json: true,
                strip: true,
            },
        },
        trailer: Pattern::default(),
        word: word(),
        hold_cap: HOLD_CAP_BYTES,
    }
}

/// How the chat template writes a turn: a call's message as ` to=NAME<|message|>`, the block's
/// opener and a newline, the invoke with its name, each parameter on a line of its own, the
/// invoke's close, then the block's close; `<|eom|>` and the next message's `<|start|>assistant`
/// between two calls; the thought after a space, closed by `<|eom|>` and followed by the next
/// message's opener; the answer after a space, ended by the turn's stop.
fn layout() -> Layout {
    let written = |segments: &[Segment]| Written(segments.into());
    let piece = |text: &str| Segment::Lit(text.into());
    Layout {
        call: CallLayout {
            lead: written(&[
                piece(" to="),
                Segment::Slot(Slot::Name),
                piece("<|message|><atem:function_calls>\n"),
            ]),
            header: written(&[piece(" name=\""), Segment::Slot(Slot::Name), piece("\">")]),
            before: "\n".into(),
            parameter: written(&[
                piece("<atem:parameter name=\""),
                Segment::Slot(Slot::Key),
                piece("\">"),
                Segment::Slot(Slot::Value),
                piece("</atem:parameter>"),
            ]),
            between: "\n".into(),
            after: "\n".into(),
            trailer: "".into(),
            close: Some("</atem:invoke>".into()),
        },
        between_calls: Some("\n</atem:function_calls><|eom|><|start|>assistant".into()),
        after_calls: "\n</atem:function_calls>".into(),
        values: Spellings::default(),
        thought: Some(Message {
            before: " ".into(),
            after: "<|start|>assistant".into(),
            stops: false,
            close: Some("<|eom|>".into()),
        }),
        answer: Some(Message {
            before: " ".into(),
            after: "".into(),
            stops: true,
            close: None,
        }),
    }
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

fn none_of(chars: &str) -> Class {
    Class::new(chars.chars().map(|c| (c, c)), true)
}

/// The characters `\b` reads as a word's in the header and the parameter tags: the ASCII word
/// characters, which is what stands on either side of the boundaries there (`<atem:invoke` and
/// `name` on one side, a space, a quote or `>` on the other).
fn word() -> Class {
    Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json as value, Value};

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        grammar::{GrammarStart, Options},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

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
                value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "options": {"type": "object"},
                }, "required": ["city"]}),
            ),
            tool(
                "lookup",
                value!({"type": "object", "properties": {
                    "q": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "integer"}},
                }}),
            ),
        ]
    }

    fn declared() -> Declared {
        Declared::of(&tools())
    }

    /// The generation prompt's tail: the user's message and the turn's opener.
    const PROMPT: &str = "<|start|>user<|message|>Weather in Paris?<|eot|><|start|>assistant";

    /// A turn as the template renders it: the thought, an update, and two calls, each in its own
    /// message, the last one ended by the stop the decoder hides.
    const OUTPUT: &str = concat!(
        " to=self<|message|>The user asks about Paris.<|eom|>",
        "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"get_weather\">\n",
        "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
        "<atem:parameter name=\"days\">3</atem:parameter>\n",
        "</atem:invoke>\n</atem:function_calls><|eom|>",
        "<|start|>assistant to=lookup<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"lookup\">\n",
        "<atem:parameter name=\"q\">Paris</atem:parameter>\n",
        "<atem:parameter name=\"ids\">[1, 2]</atem:parameter>\n",
        "</atem:invoke>\n</atem:function_calls>"
    );

    fn run_with(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Engine::new(muse_spark(), declared());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        for piece in pieces {
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.drain()
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        run_with(prompt, pieces, EngineFinish::Stop)
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    /// The calls as (name, arguments) pairs, the arguments as the JSON value their fragments make.
    fn calls(events: &[Event]) -> Vec<(String, Value)> {
        let mut found: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { index, name, .. } => {
                    assert_eq!(*index as usize, found.len(), "{events:?}");
                    found.push((name.clone(), String::new()));
                }
                Event::ToolCallArguments { index, json, .. } => {
                    found[*index as usize].1.push_str(json);
                }
                _ => {}
            }
        }
        found
            .into_iter()
            .map(|(name, arguments)| {
                let value = serde_json::from_str(&arguments)
                    .unwrap_or_else(|e| panic!("{name}: arguments {arguments:?}: {e}"));
                (name, value)
            })
            .collect()
    }

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn dropped_bytes(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_calls_finished(events: &[Event]) -> u32 {
        match events.last() {
            Some(Event::Finish { tool_calls, .. }) => *tool_calls,
            other => panic!("no finish: {other:?}"),
        }
    }

    #[test]
    fn a_rendered_turn_gives_its_thought_its_update_and_its_calls_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "Checking.");
        assert_eq!(
            calls(&events),
            [
                (
                    "get_weather".to_string(),
                    value!({"city": "Paris", "days": 3})
                ),
                ("lookup".to_string(), value!({"q": "Paris", "ids": [1, 2]})),
            ]
        );
        assert_eq!(tool_calls_finished(&events), 2);
        // The headers, the block around each call and the next message's opener are the
        // template's framing, dropped; nothing is malformed.
        assert_eq!(malformed(&events), Vec::<&str>::new());
        let framing = dropped_bytes(&events);
        for piece in [
            " to=self<|message|>",
            "<|eom|>",
            "<|start|>assistant to=commentary<|message|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
            "\n</atem:function_calls>",
        ] {
            assert!(framing.contains(piece), "{piece:?} in {framing:?}");
        }
        assert_eq!(
            events[..4],
            [
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Wrapper
                },
                Event::Dropped {
                    text: Text::uncounted("to=self<|message|>"),
                    why: DropReason::Wrapper
                },
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.")),
            ]
        );
    }

    #[test]
    fn a_thought_and_an_answer_are_the_messages_bytes_exactly() {
        let output = concat!(
            " to=self<|message|>Plan: 16 - 3 - 4 = 9 eggs.\nShe makes $18.<|eom|>",
            "<|start|>assistant to=user<|message|>She makes $18 a day."
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            reasoning(&events),
            "Plan: 16 - 3 - 4 = 9 eggs.\nShe makes $18."
        );
        assert_eq!(content(&events), "She makes $18 a day.");
        assert_eq!(tool_calls_finished(&events), 0);
        assert!(events.contains(&Event::ReasoningEnd));
    }

    #[test]
    fn an_update_and_the_answer_are_both_content_in_order() {
        let output = concat!(
            " to=self<|message|>private<|eom|>",
            "<|start|>assistant to=commentary<|message|>working<|eom|>",
            "<|start|>assistant to=user<|message|>final answer<|eot|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "private");
        assert_eq!(content(&events), "workingfinal answer");
        assert_eq!(tool_calls_finished(&events), 0);
    }

    #[test]
    fn call_markup_quoted_in_the_answer_or_the_thought_is_text() {
        let quoted = concat!(
            "Example: <atem:function_calls>\n<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n",
            "</atem:function_calls>"
        );
        let answer = format!(" to=user<|message|>{quoted}");
        let events = run(PROMPT, &[&answer]);
        assert_eq!(bytes(&events), answer);
        assert_eq!(content(&events), quoted);
        assert_eq!(tool_calls_finished(&events), 0);

        let thought =
            format!(" to=self<|message|>{quoted}<|eom|><|start|>assistant to=user<|message|>No.");
        let events = run(PROMPT, &[&thought]);
        assert_eq!(bytes(&events), thought);
        assert_eq!(reasoning(&events), quoted);
        assert_eq!(content(&events), "No.");
        assert_eq!(tool_calls_finished(&events), 0);
    }

    #[test]
    fn values_are_typed_by_the_tool_and_read_as_json_otherwise() {
        // A declared string keeps its text, whatever it looks like; the other declared types
        // take the JSON the template writes.
        let output = concat!(
            " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">42</atem:parameter>\n",
            "<atem:parameter name=\"days\">3</atem:parameter>\n",
            "<atem:parameter name=\"metric\">true</atem:parameter>\n",
            "<atem:parameter name=\"options\">{\"a\": 1}</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                value!({"city": "42", "days": 3, "metric": true, "options": {"a": 1}})
            )]
        );
        // A tool the request did not declare: JSON where the text is JSON, text otherwise.
        let output = concat!(
            " to=other<|message|><atem:function_calls>\n<atem:invoke name=\"other\">\n",
            "<atem:parameter name=\"p\">plain words</atem:parameter>\n",
            "<atem:parameter name=\"l\">[1, 2]</atem:parameter>\n",
            "<atem:parameter name=\"z\">null</atem:parameter>\n",
            "<atem:parameter name=\"n\">2.5</atem:parameter>\n",
            "<atem:parameter name=\"s\">\"quoted\"</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(
                "other".to_string(),
                value!({"p": "plain words", "l": [1, 2], "z": null, "n": 2.5, "s": "quoted"})
            )]
        );
    }

    #[test]
    fn a_header_with_other_attributes_names_the_tool_and_one_without_a_name_opens_no_call() {
        let output = concat!(
            " to=lookup<|message|><atem:function_calls>\n",
            "<atem:invoke id=\"1\" name=\"lookup\">\n",
            "<atem:parameter type=\"string\" name=\"q\">x y</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("lookup".to_string(), value!({"q": "x y"}))]
        );

        // No name: the invoke is framing, and the turn goes on to the answer.
        let output = "<atem:invoke id=\"1\">text</atem:invoke> to=user<|message|>Hi";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Hi");
        assert_eq!(tool_calls_finished(&events), 0);
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run(PROMPT, &[OUTPUT]);
        let whole_calls = calls(&whole);
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(PROMPT, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(
                reasoning(&events),
                "The user asks about Paris.",
                "cut at {cut}"
            );
            assert_eq!(content(&events), "Checking.", "cut at {cut}");
            assert_eq!(calls(&events), whole_calls, "cut at {cut}");
            assert_eq!(tool_calls_finished(&events), 2, "cut at {cut}");
        }
    }

    #[test]
    fn a_call_cut_by_the_end_of_its_message_or_by_the_stop_closes_with_what_arrived() {
        // `<|eom|>` before the invoke's close: the call closes there, and the turn goes on.
        let output = concat!(
            " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n<|eom|>",
            "<|start|>assistant to=user<|message|>Done"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), value!({"city": "Paris"}))]
        );
        assert_eq!(content(&events), "Done");
        assert_eq!(tool_calls_finished(&events), 1);

        // The model's stop inside the call: the stop row closes it.
        let output = concat!(
            " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), value!({"city": "Paris"}))]
        );
        assert_eq!(tool_calls_finished(&events), 1);

        // Cut for length: still one call, every byte accounted for.
        let events = run_with(PROMPT, &[output], EngineFinish::Length);
        assert_eq!(bytes(&events), output);
        assert_eq!(tool_calls_finished(&events), 1);
    }

    #[test]
    fn text_outside_the_messages_is_framing_and_dropped() {
        let output = " junk to=user<|message|>Hi";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Hi");
        assert_eq!(malformed(&events), Vec::<&str>::new());
        assert!(dropped_bytes(&events).contains(" junk "));

        // An output with no message at all is dropped whole: nothing reaches the client.
        let events = run(PROMPT, &["unframed text"]);
        assert_eq!(bytes(&events), "unframed text");
        assert_eq!(content(&events), "");
        assert_eq!(reasoning(&events), "");
        assert_eq!(dropped_bytes(&events), "unframed text");
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        // An earlier turn's call and a user's quoted markers move nothing: the replay starts
        // after the last `<|start|>assistant`.
        let prompt = concat!(
            "<|start|>user<|message|>q<|eot|>",
            "<|start|>assistant to=self<|message|>t<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris",
            "</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>",
            "<|start|>tool get_weather<|message|><tool_output name=\"get_weather\">\n",
            "Sunny <atem:invoke name=\"x\"> to=self<|message|>\n</tool_output><|eot|>",
            "<|start|>assistant"
        );
        let output = " to=self<|message|>Plan.<|eom|><|start|>assistant to=user<|message|>Sunny.";
        let events = run(prompt, &[output]);
        assert_eq!(reasoning(&events), "Plan.");
        assert_eq!(content(&events), "Sunny.");
        assert_eq!(tool_calls_finished(&events), 0);

        // A prompt that opens the thought leaves the output inside it.
        let prompt = format!("{PROMPT} to=self<|message|>");
        assert!(muse_spark().reasoning_open(&prompt));
        let events = run(
            &prompt,
            &["Plan.<|eom|><|start|>assistant to=user<|message|>Hi"],
        );
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning(&events), "Plan.");
        assert_eq!(content(&events), "Hi");

        // A prompt that continues the answer leaves the output in content.
        let prompt = format!("{PROMPT} to=user<|message|>The capital");
        assert!(!muse_spark().reasoning_open(&prompt));
        let events = run(&prompt, &[" is Paris."]);
        assert_eq!(content(&events), " is Paris.");
        assert!(!events.contains(&Event::ReasoningStart));
    }

    /// The required-call tag's payload for the test's tools.
    fn required_payload(reasoning_open: bool) -> Value {
        muse_spark()
            .grammar_for(
                &tools(),
                Options {
                    at_least_one: true,
                    reasoning_open,
                    one_call: false,
                    start: GrammarStart::Unknown,
                },
            )
            .expect("a tag")
            .payload()
    }

    #[test]
    fn a_forced_call_is_written_as_the_template_writes_a_calls_message() {
        let payload = required_payload(false);
        assert_eq!(
            payload["format"]["elements"][0],
            value!({"type": "or", "elements": [
                {"type": "const_string", "value": " to=get_weather<|message|><atem:function_calls>\n"},
                {"type": "const_string", "value": " to=lookup<|message|><atem:function_calls>\n"},
            ]})
        );
        let calls = &payload["format"]["elements"][1];
        assert_eq!(calls["type"], "triggered_tags");
        assert_eq!(calls["triggers"], value!(["<atem:invoke"]));
        assert_eq!(calls["at_least_one"], true);
        assert_eq!(
            calls["tags"][0]["begin"],
            "<atem:invoke name=\"get_weather\">\n"
        );
        assert_eq!(calls["tags"][0]["end"], "</atem:invoke>");
        assert_eq!(calls["tags"][1]["begin"], "<atem:invoke name=\"lookup\">\n");
        let city = calls["tags"][0]["content"].to_string();
        assert!(
            city.contains("<atem:parameter name=\\\"city\\\">"),
            "the city parameter's tag: {city}"
        );
        // A named function, and parallel calls off, hold one call.
        let one = muse_spark()
            .grammar_for(
                &tools(),
                Options {
                    at_least_one: true,
                    reasoning_open: false,
                    one_call: true,
                    start: GrammarStart::Unknown,
                },
            )
            .expect("a tag")
            .payload();
        assert_eq!(one["format"]["elements"][1]["stop_after_first"], true);
        // With the thought left open by the prompt, free text leads the call.
        let open = required_payload(true);
        assert_eq!(open["format"]["elements"][0]["type"], "any_text");
    }

    #[test]
    fn a_json_answer_is_the_answers_message_with_the_schema_in_it() {
        let grammar = muse_spark()
            .answer_grammar(&value!({"type": "object"}), false)
            .expect("an answer tag");
        let payload = grammar.payload();
        assert_eq!(payload["format"]["type"], "tag");
        assert_eq!(payload["format"]["begin"], " to=user<|message|>");
        assert_eq!(payload["format"]["end"], "");
        assert_eq!(payload["format"]["content"]["type"], "json_schema");
    }

    #[test]
    fn the_table_validates_and_its_layout_fits() {
        let format = muse_spark();
        format.validate().expect("a sound table");
        format
            .clone()
            .without_calls()
            .validate()
            .expect("a sound table without calls");
        format
            .clone()
            .without_reasoning()
            .validate()
            .expect("a sound table without the thought");
        assert_eq!(layout().check(&format), Ok(()));
        // Of the model's two stop tokens only `<|eot|>` is a terminal; `<|eom|>` is not a stop.
        assert!(format.is_stop_token("<|eot|>"));
        assert!(!format.is_stop_token("<|end_of_text|>"));
        assert!(!format.is_stop_token("<|eom|>"));
        assert_eq!(
            format.replay("<|start|>user<|message|>q<|eot|><|start|>assistant to=self<|message|>x"),
            "<|start|>assistant to=self<|message|>x"
        );
    }
}

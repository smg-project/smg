//! Kimi K2: the thought between `<think>` and `</think>`, and the calls of one turn as one block,
//! `<|tool_calls_section_begin|>` to `<|tool_calls_section_end|>`, holding one
//! `<|tool_call_begin|>` per call: the call's id, `functions.NAME:N` (the tool's name after
//! `functions.`, the call's place in the turn after the colon), `<|tool_call_argument_begin|>`,
//! the arguments object alone and `<|tool_call_end|>`, with nothing between the calls. Content
//! comes before the block. As the line's templates render a turn, after the generation prompt's
//! `<|im_assistant|>assistant<|im_middle|>`:
//!
//! ```text
//! Let me check.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>
//! ```
//!
//! One table reads the line, and the prompt's replay decides where the output starts: Kimi K2
//! Instruct (and Instruct-0905) write no thought; K2 Thinking writes its own `<think>`; K2.5, K2.6
//! and K2.7-Code have the prompt open the thought (`<think>`), or close it at once
//! (`<think></think>`) when the request turns thinking off. A call is given as data
//! ([`CallSyntax::Shaped`]): its header after `<|tool_call_begin|>` is the id, `functions.`
//! (which the gateway's older parser lets a model leave out), the name, `:`, the index and
//! `<|tool_call_argument_begin|>`, with the whitespace the vendor's reader allows around the id;
//! an id the header does not match opens no call. The call carries the id as the model wrote it
//! ([`CallId::Header`]): the line's templates render a turn's ids back into the prompt
//! (`<|tool_call_begin|>ID…` and `## Return of ID`), so a call keeps the id the model will see
//! again and a tool result pairs with it; the gateway numbers no K2 call. The arguments are one JSON value, closed by
//! `<|tool_call_end|>` right after it, so a closing marker inside a string argument ends the call
//! there, as the engine reads every table (bellwether #16), and the block's close ends a call
//! still open with what arrived; a second `<|tool_call_begin|>` inside an open call is the
//! call's text, since a shaped call's opener may open none and the engine cannot send it back
//! into a state that holds calls. Every marker is one token of the vocabulary. bellwether holds the
//! line's manifests and no recorded set yet: the table stands on the tests below, which hold a
//! hand-picked set of the turns the six templates render for the shapes a tool-calling conversation
//! takes (an answer, a call, two calls, content before a call, a thought and a call, the turn after
//! a tool result, a call with thinking off, arguments that nest and hold text outside ASCII, a
//! brace inside a string value), read whole and at every cut, and on the id contract; the six slugs
//! are listed in the corpus harness, which replays their recorded sets from the fixture pin once
//! bellwether has them. The gateway's older `kimik2` parser reads the same spelling, with the
//! `kimi_thinking` and `kimi_k25` thought parsers; one departure from `kimi_thinking`, which read
//! everything before `</think>` as the thought whether or not `<think>` came first: this table
//! starts K2 Thinking in content, so an output that closes a thought it never opened is content as
//! written, the marker included (the templates always write the opener; the tests pin it).
//!
//! The structural tag of a forced or required call is written from the table's layout
//! ([`layout`], [`Format::grammar_for`]), which spells a call as the templates render one: the
//! block's opener, `<|tool_call_begin|>functions.NAME:`, the call's number as a run of digits
//! (the model counts its calls; no request knows the count ahead), `<|tool_call_argument_begin|>`,
//! the tool's schema as the one JSON value, `<|tool_call_end|>`, and the block's close; from the
//! turn's own state or after the thought's close, which the templates follow with nothing. Where
//! the engine starts its grammar after its reasoning parser has seen `</think>`, the tag is the
//! block at once there, so a thinking model keeps its thought and the forced call follows it in
//! the model's own syntax, which the table reads back (the tests below hold the recorded forced
//! answers of a live cell in that shape, whole and at every cut). Without the tag the gateway
//! hands such a request to its older path, whose JSON-schema constraint makes the model write the
//! bare arguments after the thought, which the table reads as content (pinned below too).
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallId, CallSyntax, Emits, Format},
    shape::{
        self, Arguments, Atom, CallLayout, CallShape, Class, Dialect, Layout, Message, NameSource,
        Pattern, Segment, Slot, Spellings, Written,
    },
};

/// The most bytes a call's id holds undecided, the matcher's own default.
const HOLD_CAP_BYTES: u32 = 8192;

/// The Kimi K2 table.
pub fn kimi_k2() -> Format {
    Format::new("kimi_k2")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<|tool_calls_section_begin|>")
        .terminal("calls_close", "<|tool_calls_section_end|>")
        .terminal("call_open", "<|tool_call_begin|>")
        .terminal("call_close", "<|tool_call_end|>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("calls", "call_open", "call")
        .transition("call", "call_close", "calls")
        .transition("call", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Shaped(shape::intern(call_shape())))
        .call_id(CallId::Header)
        .opens_turn("<|im_assistant|>assistant<|im_middle|>")
        .with_layout(layout())
}

/// How the line's templates write a turn where the table's rows leave the text open, as the
/// module documentation renders one: a call's id as `functions.NAME:N` (the tool's name after
/// `functions.`, the call's number after the colon) and `<|tool_call_argument_begin|>`, the
/// arguments one JSON value the call's close follows at once, nothing of the template's between
/// two calls or after the last (the block's markers are the table's own), and the thought closed
/// by `</think>` with nothing after it before the content or the block. The structural tags are
/// written from it; parsing does not read it.
fn layout() -> Layout {
    let piece = |text: &str| Segment::Lit(text.into());
    Layout {
        call: CallLayout {
            lead: Written::default(),
            header: Written(
                [
                    piece("functions."),
                    Segment::Slot(Slot::Name),
                    piece(":"),
                    Segment::Index,
                    piece("<|tool_call_argument_begin|>"),
                ]
                .into(),
            ),
            before: "".into(),
            parameter: Written::default(),
            between: "".into(),
            after: "".into(),
            trailer: "".into(),
            close: Some("<|tool_call_end|>".into()),
        },
        between_calls: Some("".into()),
        after_calls: "".into(),
        values: Spellings::default(),
        thought: Some(Message {
            before: "".into(),
            after: "".into(),
            stops: false,
            close: Some("</think>".into()),
        }),
        answer: None,
    }
}

/// A run of whitespace, as long as it is: the `\s*` the vendor's reader allows on either side
/// of the id.
fn spaces() -> Atom {
    Atom::Run {
        class: Class::new(
            [(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
            false,
        ),
        min: 0,
        max: None,
        greedy: true,
        capture: None,
    }
}

/// A call as the id spells it after `<|tool_call_begin|>`: `functions.`, which the gateway's
/// older parser also reads without, the tool's name (letters, digits, `_`, `-` and `.`), `:`,
/// the call's index, and `<|tool_call_argument_begin|>`; then one JSON value, which
/// `<|tool_call_end|>` closes with nothing between.
fn call_shape() -> CallShape {
    CallShape {
        header: Pattern(
            [
                spaces(),
                Atom::Maybe("functions.".into()),
                Atom::Run {
                    class: Class::new(
                        [('-', '.'), ('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')],
                        false,
                    ),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Lit(":".into()),
                Atom::Run {
                    class: Class::new([('0', '9')], false),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: None,
                },
                spaces(),
                Atom::Lit("<|tool_call_argument_begin|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word: Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
        hold_cap: HOLD_CAP_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json as value, Value};

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, FinishReason},
        grammar::{GrammarStart, Options},
        input::EngineFinish,
        testing::{bytes, calls, content, drive, drive_to, reasoning},
        Declared,
    };

    /// The generation prompt's tail as K2 Instruct and K2 Thinking end it: the user's message and
    /// the turn's opener.
    const PROMPT: &str =
        "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|><|im_assistant|>assistant<|im_middle|>";
    /// The tail as K2.5, K2.6 and K2.7-Code end it: the thought opened.
    const PROMPT_THOUGHT_OPEN: &str = concat!(
        "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|><|im_assistant|>assistant<|im_middle|>",
        "<think>"
    );
    /// The tail as K2.5 and K2.6 end it when the request turns thinking off: the thought closed
    /// at once.
    const PROMPT_THOUGHT_CLOSED: &str = concat!(
        "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|><|im_assistant|>assistant<|im_middle|>",
        "<think></think>"
    );
    /// One call, as the templates render it.
    const CALL: &str = concat!(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>"
    );
    /// A thought, an answer and two calls, as K2.5's template renders them after the prompt's
    /// `<think>`.
    const TURN: &str = concat!(
        "Paris and Rome.</think>Checking both.<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>",
        "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Rome\", \"days\": 3}<|tool_call_end|>",
        "<|tool_calls_section_end|>"
    );
    /// The tail of the generation prompt after a tool result, as the templates render it: the
    /// result names the call it answers by the id the model gave.
    const PROMPT_AFTER_RESULT: &str = concat!(
        "<|im_system|>tool<|im_middle|>## Return of functions.get_weather:0\n{\"temperature\": 18}",
        "<|im_end|><|im_assistant|>assistant<|im_middle|>"
    );
    /// The same tail as K2.5, K2.6 and K2.7-Code end it: the thought opened.
    const PROMPT_AFTER_RESULT_THOUGHT_OPEN: &str = concat!(
        "<|im_system|>tool<|im_middle|>## Return of functions.get_weather:0\n{\"temperature\": 18}",
        "<|im_end|><|im_assistant|>assistant<|im_middle|><think>"
    );
    /// The thought and a call after a tool result, as K2.5's template renders them after the
    /// prompt's `<think>`.
    const THOUGHT_CALL_AFTER_RESULT: &str = concat!(
        "Rome too.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>"
    );
    /// A call whose arguments nest and hold text outside ASCII, as the templates write them
    /// (`tojson` with `ensure_ascii` off, the escapes JSON's own).
    const CALL_NESTED: &str = concat!(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>",
        "{\"title\": \"計画 🌍\", \"body\": \"a \\\"q\\\" \\\\ b\\nc\", \"tags\": [\"a\", \"b\"], ",
        "\"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}<|tool_call_end|><|tool_calls_section_end|>"
    );
    /// A call with a brace inside a string value.
    const CALL_BRACE_IN_STRING: &str = concat!(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>",
        "{\"title\": \"Braces\", \"body\": \"a } b { c\"}<|tool_call_end|><|tool_calls_section_end|>"
    );

    fn run_with(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event<'static>> {
        drive_to(
            &mut Engine::new(kimi_k2(), Declared::default()),
            prompt,
            pieces,
            finish,
        )
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(kimi_k2(), Declared::default()),
            prompt,
            pieces,
        )
    }

    /// The calls' ids, in order.
    fn ids<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn starts<'a>(events: &'a [Event<'_>]) -> Vec<(u32, &'a str, &'a str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart {
                    index,
                    name,
                    source,
                    ..
                } => Some((*index, name.as_str(), &*source.text)),
                _ => None,
            })
            .collect()
    }

    fn ends<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The bytes the table dropped as wrapping, in order.
    fn framing(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The engine's finish: its reason as the engine gave it, and the calls counted.
    fn finish(events: &[Event]) -> (FinishReason, u32) {
        events
            .iter()
            .find_map(|event| match event {
                Event::Finish {
                    reason, tool_calls, ..
                } => Some((reason.clone(), *tool_calls)),
                _ => None,
            })
            .expect("a finish")
    }

    #[test]
    fn a_rendered_turn_gives_its_content_its_call_and_every_byte() {
        let output = format!("Let me check.{CALL}");
        let events = run(PROMPT, &[&output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(reasoning(&events), "");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        // The block's markers and the call's opener are wrapping; the id is the start's source,
        // the closing marker the end's.
        assert_eq!(
            framing(&events),
            "<|tool_calls_section_begin|><|tool_call_begin|><|tool_calls_section_end|>"
        );
        assert_eq!(
            starts(&events),
            [(
                0,
                "get_weather",
                "functions.get_weather:0<|tool_call_argument_begin|>"
            )]
        );
        assert_eq!(ends(&events), ["<|tool_call_end|>"]);
        assert_eq!(malformed(&events), "");
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        // K2.5, K2.6 and K2.7-Code: the prompt opens the thought, and the output starts inside it.
        let table = kimi_k2();
        assert!(table.reasoning_open(PROMPT_THOUGHT_OPEN));
        assert!(!table.reasoning_open(PROMPT_THOUGHT_CLOSED));
        assert!(!table.reasoning_open(PROMPT));
        let output = format!("Need the weather.</think>{CALL}");
        let events = run(PROMPT_THOUGHT_OPEN, &[&output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Need the weather.");
        assert_eq!(content(&events), "");
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(malformed(&events), "");
        // An empty thought: the template's rendering of an answer in the history.
        let events = run(PROMPT_THOUGHT_OPEN, &["</think>It is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "It is sunny.");
        // Thinking off: the prompt closed the thought, and the output is content from its start.
        let events = run(PROMPT_THOUGHT_CLOSED, &["It is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "It is sunny.");
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        // K2 Thinking: the prompt ends at the opener, and the model writes its own thought.
        let events = run(PROMPT, &["<think>Plan.</think>It is sunny."]);
        assert_eq!(reasoning(&events), "Plan.");
        assert_eq!(content(&events), "It is sunny.");
        // K2 Thinking closing a thought it never opened: the output starts in content, where
        // the table has no row for `</think>`, so the turn is content as written, the marker
        // included. The older `kimi_thinking` parser read everything before the marker as the
        // thought; the table departs from it there, and this pins the departure.
        let events = run(PROMPT, &["Plan.</think>It is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "Plan.</think>It is sunny.");
        // K2 Instruct: no thought at all.
        let events = run(PROMPT, &["It is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "It is sunny.");
    }

    #[test]
    fn every_cut_reads_the_same_turn() {
        let whole = run(PROMPT_THOUGHT_OPEN, &[TURN]);
        assert_eq!(bytes(&whole), TURN);
        assert_eq!(reasoning(&whole), "Paris and Rome.");
        assert_eq!(content(&whole), "Checking both.");
        assert_eq!(
            calls(&whole),
            [
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Paris\"}".to_string()
                ),
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Rome\", \"days\": 3}".to_string()
                ),
            ]
        );
        assert_eq!(finish(&whole), (FinishReason::Stop, 2));
        for (at, _) in TURN.char_indices().skip(1) {
            let (head, rest) = TURN.split_at(at);
            let events = run(PROMPT_THOUGHT_OPEN, &[head, rest]);
            assert_eq!(bytes(&events), TURN, "cut at {at}");
            assert_eq!(reasoning(&events), reasoning(&whole), "cut at {at}");
            assert_eq!(content(&events), content(&whole), "cut at {at}");
            assert_eq!(calls(&events), calls(&whole), "cut at {at}");
            assert_eq!(framing(&events), framing(&whole), "cut at {at}");
            assert_eq!(malformed(&events), "", "cut at {at}");
            assert_eq!(finish(&events), finish(&whole), "cut at {at}");
        }
    }

    /// The shapes the templates render that the turns above do not hold, each read whole and at
    /// every cut: a call with thinking off (the prompt closed the thought), the thought and a
    /// call after a tool result where the prompt opens the thought, arguments that nest and hold
    /// text outside ASCII (a cut falls between its characters, never inside one), and a brace
    /// inside a string value, which closes nothing.
    #[test]
    fn every_cut_reads_the_same_shape() {
        let shapes: [(&str, &str, &str, u32); 4] = [
            (PROMPT_THOUGHT_CLOSED, CALL, "", 1),
            (
                PROMPT_AFTER_RESULT_THOUGHT_OPEN,
                THOUGHT_CALL_AFTER_RESULT,
                "Rome too.",
                1,
            ),
            (PROMPT, CALL_NESTED, "", 1),
            (PROMPT, CALL_BRACE_IN_STRING, "", 1),
        ];
        for (prompt, turn, thought, count) in shapes {
            let whole = run(prompt, &[turn]);
            assert_eq!(bytes(&whole), turn);
            assert_eq!(reasoning(&whole), thought, "{turn:?}");
            assert_eq!(content(&whole), "", "{turn:?}");
            assert_eq!(calls(&whole).len() as u32, count, "{turn:?}");
            assert_eq!(malformed(&whole), "", "{turn:?}");
            assert_eq!(finish(&whole), (FinishReason::Stop, count), "{turn:?}");
            for (at, _) in turn.char_indices().skip(1) {
                let (head, rest) = turn.split_at(at);
                let events = run(prompt, &[head, rest]);
                assert_eq!(bytes(&events), turn, "{turn:?} cut at {at}");
                assert_eq!(reasoning(&events), thought, "{turn:?} cut at {at}");
                assert_eq!(content(&events), "", "{turn:?} cut at {at}");
                assert_eq!(calls(&events), calls(&whole), "{turn:?} cut at {at}");
                assert_eq!(ids(&events), ids(&whole), "{turn:?} cut at {at}");
                assert_eq!(framing(&events), framing(&whole), "{turn:?} cut at {at}");
                assert_eq!(malformed(&events), "", "{turn:?} cut at {at}");
                assert_eq!(finish(&events), finish(&whole), "{turn:?} cut at {at}");
            }
        }
        // What each call carries: the thinking-off call's arguments and the id as written; the
        // call after the result named by the id the template renders for a new turn; the nested
        // arguments as the value the template was given; the brace kept inside its string.
        let events = run(PROMPT_THOUGHT_CLOSED, &[CALL]);
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(ids(&events), ["functions.get_weather:0"]);
        let events = run(
            PROMPT_AFTER_RESULT_THOUGHT_OPEN,
            &[THOUGHT_CALL_AFTER_RESULT],
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Rome\"}".to_string()
            )]
        );
        assert_eq!(ids(&events), ["functions.get_weather:0"]);
        let events = run(PROMPT, &[CALL_NESTED]);
        let (name, arguments) = &calls(&events)[0];
        assert_eq!(name, "save_note");
        let value: Value = serde_json::from_str(arguments).expect("JSON arguments");
        assert_eq!(
            value,
            serde_json::json!({
                "title": "計画 🌍",
                "body": "a \"q\" \\ b\nc",
                "tags": ["a", "b"],
                "meta": {"nested": {"deep": [1, 2.5, true, null]}},
            })
        );
        let events = run(PROMPT, &[CALL_BRACE_IN_STRING]);
        assert_eq!(
            calls(&events),
            [(
                "save_note".to_string(),
                "{\"title\": \"Braces\", \"body\": \"a } b { c\"}".to_string()
            )]
        );
    }

    #[test]
    fn a_call_carries_the_id_the_model_wrote() {
        // One call and two: each start carries the id of its header, not one minted for it.
        assert_eq!(ids(&run(PROMPT, &[CALL])), ["functions.get_weather:0"]);
        assert_eq!(
            ids(&run(PROMPT_THOUGHT_OPEN, &[TURN])),
            ["functions.get_weather:0", "functions.get_weather:1"]
        );
        // The turn after a tool result: the model numbers on from the call the result answered,
        // and the id it writes is the one carried, whole and character by character.
        let output = concat!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:1",
            "<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>"
        );
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        for events in [
            run(PROMPT_AFTER_RESULT, &[output]),
            run(PROMPT_AFTER_RESULT, &pieces),
        ] {
            assert_eq!(ids(&events), ["functions.get_weather:1"]);
            assert_eq!(
                calls(&events),
                [(
                    "get_weather".to_string(),
                    "{\"city\": \"Rome\"}".to_string()
                )]
            );
            assert_eq!(finish(&events), (FinishReason::Stop, 1));
        }
        // The older parser's spelling without `functions.` is carried as written too.
        let bare = "<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        assert_eq!(ids(&run(PROMPT, &[bare])), ["get_weather:0"]);
        // Two calls the model gave one id: the second takes the minted id, so the response's ids
        // stay distinct and a tool result still names one call.
        let repeated = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|>",
            "<|tool_calls_section_end|>"
        );
        let events = run(PROMPT, &[repeated]);
        let given = ids(&events);
        assert_eq!(given.len(), 2);
        assert_eq!(given[0], "functions.get_weather:0");
        assert!(crate::ids::is_minted(given[1]), "{given:?}");
        assert_eq!(finish(&events), (FinishReason::Stop, 2));
    }

    #[test]
    fn two_calls_take_their_own_indexes_and_the_blocks_close_ends_an_open_call() {
        // Two ways a call ends: its own closing marker and the block's close, a token the model
        // cannot write as text, which ends the call with what arrived; the end carries either.
        let output = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\"a\": true}<|tool_call_end|>",
            "<|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>{\"b\": \"x",
            "<|tool_calls_section_end|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [
                ("f".to_string(), "{\"a\": true}".to_string()),
                ("g".to_string(), "{\"b\": \"x".to_string()),
            ]
        );
        let indexes: Vec<u32> = starts(&events).iter().map(|start| start.0).collect();
        assert_eq!(indexes, [0, 1]);
        assert_eq!(
            ends(&events),
            ["<|tool_call_end|>", "<|tool_calls_section_end|>"]
        );
        assert_eq!(content(&events), "");
        assert_eq!(finish(&events), (FinishReason::Stop, 2));
    }

    #[test]
    fn an_id_the_header_does_not_match_opens_no_call() {
        // No index: the opener and the id are the block's text, not a call.
        let output = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather<|tool_call_argument_begin|>{}<|tool_call_end|>",
            "<|tool_calls_section_end|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(calls(&events), Vec::<(String, String)>::new());
        assert_eq!(
            malformed(&events),
            "<|tool_call_begin|>functions.get_weather<|tool_call_argument_begin|>{}<|tool_call_end|>"
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        // The spellings the gateway's older parser also reads: the id without `functions.`, and
        // whitespace around the id.
        for output in [
            "<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
            "<|tool_calls_section_begin|><|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
        ] {
            let events = run(PROMPT, &[output]);
            assert_eq!(bytes(&events), output, "{output:?}");
            assert_eq!(
                calls(&events),
                [("get_weather".to_string(), "{}".to_string())],
                "{output:?}"
            );
            assert_eq!(malformed(&events), "", "{output:?}");
            assert_eq!(finish(&events), (FinishReason::Stop, 1), "{output:?}");
        }
    }

    #[test]
    fn a_closing_marker_inside_a_string_argument_ends_the_call_there() {
        // The engine's rule for every table (bellwether #16): a terminal inside a string in a
        // call's arguments is a terminal; the marker is one token the model cannot write as text.
        let output = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.save_note:0<|tool_call_argument_begin|>",
            "{\"body\": \"a <|tool_call_end|> b\"}<|tool_call_end|>",
            "<|tool_calls_section_end|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("save_note".to_string(), "{\"body\": \"a ".to_string())]
        );
        assert_eq!(ends(&events), ["<|tool_call_end|>"]);
        // What followed the marker is the block's text: its whitespace dropped, the rest malformed.
        assert_eq!(malformed(&events), "b\"}<|tool_call_end|>");
        assert_eq!(content(&events), "");
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn a_call_the_end_cuts_is_a_call_with_what_arrived() {
        let piece = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par";
        let events = run(PROMPT, &[piece]);
        assert_eq!(bytes(&events), piece);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events).1, 1);
        let events = run_with(PROMPT, &[piece], EngineFinish::Length);
        assert_eq!(finish(&events).0, FinishReason::Length);
        // Cut inside the id: no call opened, and every byte accounted for.
        let piece = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_wea";
        let events = run_with(PROMPT, &[piece], EngineFinish::Length);
        assert_eq!(bytes(&events), piece);
        assert_eq!(calls(&events), Vec::<(String, String)>::new());
        assert_eq!(finish(&events), (FinishReason::Length, 0));
    }

    #[test]
    fn the_table_validates() {
        let table = kimi_k2();
        assert_eq!(table.validate(), Ok(()));
        assert_eq!(table.name(), "kimi_k2");
    }

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

    /// The tool of the recorded named case: a forecast with three required parameters.
    fn forecast() -> Tool {
        tool(
            "detailed_weather_forecast",
            value!({"type": "object", "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
                "details": {"type": "array", "items": {"type": "string"}},
            }, "required": ["location", "days", "details"]}),
        )
    }

    /// The tool of the recorded required case: one required parameter, one optional.
    fn artworks() -> Tool {
        tool(
            "metropolitan_museum_get_top_artworks",
            value!({"type": "object", "properties": {
                "number": {"type": "integer"},
                "sort_by": {"type": "string", "enum": ["popularity", "chronological", "alphabetical"]},
            }, "required": ["number"]}),
        )
    }

    /// The payload of the table's tag for `tools` as `options` ask.
    fn payload(tools: &[Tool], options: Options) -> Option<Value> {
        kimi_k2()
            .grammar_for(tools, options)
            .map(|grammar| grammar.payload())
    }

    fn forced(reasoning_open: bool, one_call: bool, start: GrammarStart) -> Options {
        Options {
            at_least_one: true,
            reasoning_open,
            thought_closed: false,
            one_call,
            start,
        }
    }

    /// One call to `tool` as the tag writes it: the call's opener and the id up to the number,
    /// the number as digits, the arguments' opener, the tool's schema, the call's close.
    fn written_call(tool: &Tool) -> Value {
        value!({"type": "sequence", "elements": [
            {"type": "const_string", "value": format!("<|tool_call_begin|>functions.{}:", tool.function.name)},
            {"type": "sequence", "elements": [
                {"type": "regex", "pattern": "[0-9]+"},
                {"type": "const_string", "value": "<|tool_call_argument_begin|>"},
                {"type": "json_schema", "json_schema": tool.function.parameters},
            ]},
            {"type": "const_string", "value": "<|tool_call_end|>"},
        ]})
    }

    #[test]
    fn the_table_validates_and_its_layout_fits() {
        let format = kimi_k2();
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
        assert_eq!(format.layout(), Some(&layout()));
    }

    #[test]
    fn a_forced_call_is_the_block_holding_the_id_the_number_and_the_tools_schema() {
        // A named function on a K2 Thinking prompt (the model in content, its thought its own),
        // for an engine that starts the grammar after its reasoning parser has seen the thought's
        // close: the block at once, entered from the turn's own state or right after `</think>`,
        // holding the one call, closed by the block's close, and nothing after it.
        let got = payload(
            &[forecast()],
            forced(false, true, GrammarStart::AfterReasoning),
        )
        .expect("a tag");
        let one_call = value!({"type": "or", "elements": [written_call(&forecast())]});
        let tag = |begin: &str| {
            value!({
                "type": "tag",
                "begin": begin,
                "content": one_call,
                "end": "<|tool_calls_section_end|>",
            })
        };
        assert_eq!(
            got,
            value!({"format": {
                "type": "triggered_tags",
                "triggers": ["<|tool_calls_section_begin|>", "</think>"],
                "tags": [
                    tag("<|tool_calls_section_begin|>"),
                    tag("</think><|tool_calls_section_begin|>"),
                ],
                "at_least_one": true,
                "stop_after_first": true,
            }})
        );
        // `required` with parallel calls allowed: one or more calls to any of the tools, the
        // block closed after the last, and more may follow.
        let got = payload(
            &[forecast(), artworks()],
            forced(false, false, GrammarStart::AfterReasoning),
        )
        .expect("a tag");
        let format = &got["format"];
        assert_eq!(format["at_least_one"], true);
        assert_eq!(format["stop_after_first"], Value::Null);
        assert_eq!(
            format["tags"][0]["content"],
            value!({"type": "plus", "content": {"type": "or", "elements": [
                written_call(&forecast()),
                written_call(&artworks()),
            ]}})
        );
        // The engine's start changes nothing on a prompt that opens no thought: the model writes
        // its own `<think>`, which the table's rows leave to it, and the block follows the close
        // where the engine gates the grammar; an engine that starts at the first token gets the
        // block at once, as every table without a thought-first row does.
        assert_eq!(
            payload(&[forecast()], forced(false, true, GrammarStart::FirstToken)),
            payload(&[forecast()], forced(false, true, GrammarStart::Unknown))
        );
        assert_eq!(
            payload(&[forecast()], forced(false, true, GrammarStart::FirstToken)),
            payload(
                &[forecast()],
                forced(false, true, GrammarStart::AfterReasoning)
            )
        );
        // No call owed (`auto`): the same tags, free text before them.
        let auto = payload(
            &[forecast()],
            Options {
                at_least_one: false,
                reasoning_open: false,
                thought_closed: false,
                one_call: false,
                start: GrammarStart::Unknown,
            },
        )
        .expect("a tag");
        assert_eq!(auto["format"]["at_least_one"], false);
        assert_eq!(auto["format"]["tags"], got["format"]["tags"].as_array().map(|tags| {
            tags.iter().map(|tag| {
                let mut tag = tag.clone();
                tag["content"] = value!({"type": "plus", "content": {"type": "or", "elements": [written_call(&forecast())]}});
                tag
            }).collect::<Value>()
        }).expect("tags"));
    }

    #[test]
    fn with_the_prompts_thought_open_the_tag_follows_the_close_as_the_engine_starts() {
        // K2.5's prompt ends with `<think>`. After the gate of the engine's reasoning parser the
        // block comes at once, since the template writes nothing between `</think>` and the
        // block; from the first token the thought is written first, closed by its marker; an
        // engine that does not say gets the free-text prefix.
        let gated = payload(
            &[forecast()],
            forced(true, true, GrammarStart::AfterReasoning),
        )
        .expect("a tag");
        assert_eq!(
            gated,
            payload(
                &[forecast()],
                forced(false, true, GrammarStart::AfterReasoning)
            )
            .expect("a tag")
        );
        let first =
            payload(&[forecast()], forced(true, true, GrammarStart::FirstToken)).expect("a tag");
        assert_eq!(first["format"]["type"], "sequence");
        assert_eq!(first["format"]["elements"][0]["type"], "tag");
        assert_eq!(first["format"]["elements"][0]["begin"], "");
        assert_eq!(first["format"]["elements"][0]["end"], "</think>");
        assert_eq!(first["format"]["elements"][1], gated["format"]);
        let unknown =
            payload(&[forecast()], forced(true, true, GrammarStart::Unknown)).expect("a tag");
        assert_eq!(unknown["format"]["elements"][0]["type"], "any_text");
        assert_eq!(unknown["format"]["elements"][1], gated["format"]);
        // A tool with no name has no call; no tool, no tag.
        assert!(payload(&[], forced(false, true, GrammarStart::Unknown)).is_none());
        assert!(payload(
            &[tool("", value!({"type": "object"}))],
            forced(false, true, GrammarStart::Unknown)
        )
        .is_none());
    }

    /// The thought of the recorded named case (a live Kimi K2 Thinking cell: `tool_choice`
    /// naming `detailed_weather_forecast`), as the model wrote it between its markers.
    const FORCED_NAMED_THOUGHT: &str = "The user is asking for a weather forecast for London, United Kingdom for the next 3 days. They want:\n1. High and low temperatures\n2. Humidity\n3. Precipitation\n\nI need to use the detailed_weather_forecast function with:\n- location: \"London, United Kingdom\" (or just \"London\")\n- days: 3\n- details: [\"high_low_temperature\", \"humidity\", \"precipitation\"]\n\nLet me make this function call.";
    /// Its arguments, as the model wrote them under the older path's JSON-schema constraint
    /// (the bare object after the thought) and as it writes them inside the tag's call.
    const FORCED_NAMED_ARGUMENTS: &str = "{\"location\": \"London, United Kingdom\", \"days\": 3, \"details\": [\"high_low_temperature\", \"humidity\", \"precipitation\"]}";
    /// The thought of the recorded required case (`tool_choice: required`, one tool).
    const FORCED_REQUIRED_THOUGHT: &str = "The user wants to get the top 5 popular artworks at the Metropolitan Museum of Art, sorted by popularity. This matches exactly with the function I have available. I need to call the function with:\n- number: 5\n- sort_by: \"popularity\" (which is the default, but I'll specify it as requested)\n\nLet me make this function call.";
    /// The call list the older path's constraint had the model write after the thought.
    const FORCED_REQUIRED_LIST: &str = "[{\"name\": \"metropolitan_museum_get_top_artworks\", \"parameters\": {\"number\": 5, \"sort_by\": \"popularity\"}}]";
    const FORCED_REQUIRED_ARGUMENTS: &str = "{\"number\": 5, \"sort_by\": \"popularity\"}";

    /// The call as the tag writes it after the thought's close: the block holding one call with
    /// the first number, the arguments as recorded.
    fn written(name: &str, arguments: &str) -> String {
        format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0\
             <|tool_call_argument_begin|>{arguments}<|tool_call_end|><|tool_calls_section_end|>"
        )
    }

    /// What the table said of an output: the calls, the finish, the content and the reasoning.
    #[derive(Debug, PartialEq)]
    struct Said {
        calls: Vec<(String, String)>,
        finish: (FinishReason, u32),
        content: String,
        reasoning: String,
    }

    fn said(events: &[Event]) -> Said {
        Said {
            calls: calls(events),
            finish: finish(events),
            content: content(events),
            reasoning: reasoning(events),
        }
    }

    fn read(tools: &[Tool], pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(kimi_k2(), Declared::of(tools)),
            PROMPT,
            pieces,
        )
    }

    /// `output` read whole, one character at a time and at every two-way cut says the same.
    fn the_same_at_every_cut(tools: &[Tool], output: &str) -> Said {
        let whole = said(&read(tools, &[output]));
        let chars: Vec<String> = output.chars().map(String::from).collect();
        let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            said(&read(tools, &pieces)),
            whole,
            "one character at a time"
        );
        for cut in 1..output.len() {
            if !output.is_char_boundary(cut) {
                continue;
            }
            let events = read(tools, &[&output[..cut], &output[cut..]]);
            assert_eq!(said(&events), whole, "cut at {cut}");
        }
        whole
    }

    #[test]
    fn the_recorded_forced_answers_in_the_tags_shape_read_as_the_thought_and_the_call() {
        // The named case and the required case of the live cell, each as the tag writes it after
        // the thought's close: one call with the recorded arguments, the thought as reasoning, no
        // content, `stop` with one call counted; the same in a stream of any shape.
        let named = format!(
            "<think>{FORCED_NAMED_THOUGHT}</think>{}",
            written("detailed_weather_forecast", FORCED_NAMED_ARGUMENTS)
        );
        assert_eq!(
            the_same_at_every_cut(&[forecast()], &named),
            Said {
                calls: vec![(
                    "detailed_weather_forecast".to_string(),
                    FORCED_NAMED_ARGUMENTS.to_string()
                )],
                finish: (FinishReason::Stop, 1),
                content: String::new(),
                reasoning: FORCED_NAMED_THOUGHT.to_string(),
            }
        );
        let required = format!(
            "<think>{FORCED_REQUIRED_THOUGHT}</think>{}",
            written(
                "metropolitan_museum_get_top_artworks",
                FORCED_REQUIRED_ARGUMENTS
            )
        );
        assert_eq!(
            the_same_at_every_cut(&[artworks()], &required),
            Said {
                calls: vec![(
                    "metropolitan_museum_get_top_artworks".to_string(),
                    FORCED_REQUIRED_ARGUMENTS.to_string()
                )],
                finish: (FinishReason::Stop, 1),
                content: String::new(),
                reasoning: FORCED_REQUIRED_THOUGHT.to_string(),
            }
        );
        // The id is the model's, the first call's number.
        let events = read(&[forecast()], &[&named]);
        assert_eq!(ids(&events), ["functions.detailed_weather_forecast:0"]);
        assert_eq!(malformed(&events), "");
    }

    #[test]
    fn the_older_paths_forced_answers_read_as_the_thought_and_json_content() {
        // What the gateway's older path had the model write when the table had no tag (the
        // recorded bytes of the live cell): the thought, then the bare arguments object for a
        // named function, or the list of calls for `required`. The table reads them as what they
        // are in its syntax, a thought and content, with no call: the tag is what makes a forced
        // call a call here. Pinned whole and at every cut.
        let named = format!("<think>{FORCED_NAMED_THOUGHT}</think>{FORCED_NAMED_ARGUMENTS}");
        assert_eq!(
            the_same_at_every_cut(&[forecast()], &named),
            Said {
                calls: Vec::new(),
                finish: (FinishReason::Stop, 0),
                content: FORCED_NAMED_ARGUMENTS.to_string(),
                reasoning: FORCED_NAMED_THOUGHT.to_string(),
            }
        );
        let required = format!("<think>{FORCED_REQUIRED_THOUGHT}</think>{FORCED_REQUIRED_LIST}");
        assert_eq!(
            the_same_at_every_cut(&[artworks()], &required),
            Said {
                calls: Vec::new(),
                finish: (FinishReason::Stop, 0),
                content: FORCED_REQUIRED_LIST.to_string(),
                reasoning: FORCED_REQUIRED_THOUGHT.to_string(),
            }
        );
    }
}

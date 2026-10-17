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
//! into a state that holds calls. Every marker is one token of the vocabulary. bellwether holds
//! the line's manifests and no recorded set yet: the table stands on the turns the six templates
//! render, held in the tests below; the recorded sets replay from the fixture pin once bellwether
//! has them. The gateway's older `kimik2` parser reads the same spelling, with the
//! `kimi_thinking` and `kimi_k25` thought parsers; one departure from `kimi_thinking`, which read
//! everything before `</think>` as the thought whether or not `<think>` came first: this table
//! starts K2 Thinking in content, so an output that closes a thought it never opened is content
//! as written, the marker included (the templates always write the opener; the tests pin it).
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallId, CallSyntax, Emits, Format},
    shape::{self, Arguments, Atom, CallShape, Class, Dialect, NameSource, Pattern, Slot},
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
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, FinishReason},
        input::EngineFinish,
        tagged::Declared,
        testing::{bytes, calls, content, drive, drive_to, reasoning},
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

    fn run_with(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        drive_to(
            &mut Engine::new(kimi_k2(), Declared::default()),
            prompt,
            pieces,
            finish,
        )
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        drive(
            &mut Engine::new(kimi_k2(), Declared::default()),
            prompt,
            pieces,
        )
    }

    /// The calls' ids, in order.
    fn ids(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn starts(events: &[Event]) -> Vec<(u32, &str, &str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart {
                    index,
                    name,
                    source,
                    ..
                } => Some((*index, name.as_str(), source.text.as_str())),
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
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
                } => Some(text.text.as_str()),
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

    /// The tail of the generation prompt after a tool result, as the templates render it: the
    /// result names the call it answers by the id the model gave.
    const PROMPT_AFTER_RESULT: &str = concat!(
        "<|im_system|>tool<|im_middle|>## Return of functions.get_weather:0\n{\"temperature\": 18}",
        "<|im_end|><|im_assistant|>assistant<|im_middle|>"
    );

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
}

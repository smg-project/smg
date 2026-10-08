//! Kimi K3: a turn of tagged regions. The template writes every region between an `<|open|>` tag
//! and a `<|close|>` tag, each a special token followed by the region's word and the `<|sep|>`
//! token: the thought between `<|open|>think<|sep|>` and `<|close|>think<|sep|>`, the answer
//! between `<|open|>response<|sep|>` and `<|close|>response<|sep|>`, the calls of one turn between
//! `<|open|>tools<|sep|>` and `<|close|>tools<|sep|>`, each call between
//! `<|open|>call tool="…" index="N"<|sep|>` and `<|close|>call<|sep|>`, and the whole turn closed
//! by `<|close|>message<|sep|>`. As recorded for `moonshotai/Kimi-K3`.
//!
//! What this table says beyond the DeepSeek one:
//!
//! - **The turn itself is a state.** Between the regions there is nothing: the template writes
//!   the regions back to back, and the end-of-turn tag after the last. `message` is that
//!   wrapping, so a byte there is `Dropped { Wrapper }` when it is whitespace and `Malformed`
//!   otherwise, and it is the state the turn opener leaves the engine in. Content is only what the
//!   `response` region holds, every byte of it; the recorded answers match it exactly.
//! - **Two states for the calls**, as for DSML: `tools` is the wrapping between calls, and `call`
//!   is one call's arguments, read by the XTML assembler ([`tagged::xtml`](crate::tagged::xtml)).
//!   The call tag's opening is the terminal that enters `call`, so the assembler starts inside the
//!   function's name; the call's closing tag leaves it, and the call's end carries its bytes. A
//!   second call before the first closed ends the first (`call + call_open = call`), and the
//!   block's close ends a call still open (`call + tools_close = message`).
//! - **The end-of-turn tag ends whatever is open.** `<|close|>message<|sep|>` is a token the model
//!   cannot write by accident, so from any state it ends the region and returns to `message`: a
//!   thought or an answer cut by it is closed, a call is closed into its object, and the tag
//!   itself is dropped as wrapping. The same holds for a region's opening inside the thought
//!   (`reasoning + response_open = content`, `reasoning + tools_open = tools`) and for a tools
//!   block opened inside the answer (`content + tools_open = tools`): the region ended, however
//!   the model left it, since the tags are tokens the model cannot write as text.
//! - **The prompt opens the thought.** The template ends the generation prompt with
//!   `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`, so an empty thought writes
//!   `<|close|>think<|sep|>` at once, and the engine's prompt replay puts the output inside the
//!   thought first. The turn opener is the message tag with the assistant's role, so an earlier
//!   turn's regions move nothing.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::xtml,
};

/// The Kimi K3 table.
pub fn kimi_k3() -> Format {
    Format::new("kimi_k3")
        .terminal("think_open", "<|open|>think<|sep|>")
        .terminal("think_close", "<|close|>think<|sep|>")
        .terminal("response_open", "<|open|>response<|sep|>")
        .terminal("response_close", "<|close|>response<|sep|>")
        .terminal("tools_open", "<|open|>tools<|sep|>")
        .terminal("tools_close", "<|close|>tools<|sep|>")
        .terminal("call_open", "<|open|>call tool=\"")
        .terminal("call_close", xtml::CALL_CLOSE)
        .terminal("message_close", "<|close|>message<|sep|>")
        .state("message", Emits::Wrapper)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("tools", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .transition("message", "think_open", "reasoning")
        .transition("reasoning", "think_close", "message")
        .transition("reasoning", "response_open", "content")
        .transition("reasoning", "tools_open", "tools")
        .transition("reasoning", "message_close", "message")
        .transition("message", "response_open", "content")
        .transition("content", "response_close", "message")
        .transition("content", "message_close", "message")
        .transition("content", "tools_open", "tools")
        .transition("message", "tools_open", "tools")
        .transition("tools", "call_open", "call")
        .transition("call", "call_close", "tools")
        .transition("call", "call_open", "call")
        .transition("call", "tools_close", "message")
        .transition("call", "message_close", "message")
        .transition("tools", "tools_close", "message")
        .transition("tools", "message_close", "message")
        .transition("message", "message_close", "message")
        .calls(CallSyntax::Xtml)
        .opens_turn("<|open|>message role=\"assistant\"<|sep|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, MalformedReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// The generation prompt's tail: the assistant's message opened, and its thought.
    const PROMPT: &str = "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";

    /// A recorded output: an empty thought, an empty answer, one call with a string and an object.
    const OUTPUT: &str = concat!(
        "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>",
        "<|open|>tools<|sep|><|open|>call tool=\"ChaDri_change_drink\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"drink_id\" type=\"string\"<|sep|>latte<|close|>argument<|sep|>",
        "<|open|>argument key=\"new_preferences\" type=\"object\"<|sep|>",
        "{\"size\": \"large\", \"temperature\": \"hot\"}<|close|>argument<|sep|>",
        "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(kimi_k3(), Declared::default());
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
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.drain()
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

    fn arguments_of(events: &[Event], call: u32) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
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

    fn ends(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_recorded_output_gives_its_call_with_its_types_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..7],
            [
                Event::ReasoningStart,
                dropped("<|close|>think<|sep|>"),
                Event::ReasoningEnd,
                dropped("<|open|>response<|sep|>"),
                dropped("<|close|>response<|sep|>"),
                dropped("<|open|>tools<|sep|>"),
                dropped("<|open|>call tool=\""),
            ]
        );
        assert_eq!(
            events[7],
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "ChaDri_change_drink".into(),
                source: Text::uncounted("ChaDri_change_drink\" index=\"1\"<|sep|>"),
            }
        );
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temperature": "hot"}}"#
        );
        assert_eq!(ends(&events), [xtml::CALL_CLOSE]);
        assert_eq!(content(&events), "");
        assert!(events.contains(&dropped("<|close|>tools<|sep|>")));
        assert!(events.contains(&dropped("<|close|>message<|sep|>")));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_thought_and_an_answer_are_the_regions_bytes_exactly() {
        let output = concat!(
            "Janet sells 16 - 3 - 4 = 9 eggs a day.\nShe makes $18.<|close|>think<|sep|>",
            "<|open|>response<|sep|>18<|close|>response<|sep|><|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..3],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted(
                    "Janet sells 16 - 3 - 4 = 9 eggs a day.\nShe makes $18."
                )),
                dropped("<|close|>think<|sep|>"),
            ]
        );
        assert_eq!(events[3], Event::ReasoningEnd);
        assert_eq!(content(&events), "18");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn a_string_value_streams_as_it_arrives_and_a_typed_value_is_written_at_its_close() {
        let pieces = [
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>",
            "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>Par",
            "is<|close|>argument<|sep|><|open|>argument key=\"n\" type=\"number\"<|sep|>1",
            "2<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|><|close|>message<|sep|>",
        ];
        let events = run(PROMPT, &pieces);
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            ["{\"city\": \"", "Par", "is", "\"", ", \"n\": 12", "}"]
        );
        assert_eq!(bytes(&events), pieces.concat());
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run(PROMPT, &[OUTPUT]);
        let whole_arguments = arguments_of(&whole, 0);
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(PROMPT, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(arguments_of(&events, 0), whole_arguments, "cut at {cut}");
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn two_calls_in_one_block_take_their_own_indexes_and_only_the_call_tag_is_an_end() {
        // Three ways a call ends: its own closing tag, which the end carries; the next call's
        // opening and the block's close, which the engine drops as the region's.
        let output = concat!(
            "<|close|>think<|sep|><|open|>tools<|sep|>",
            "<|open|>call tool=\"f\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>",
            "<|close|>call<|sep|>",
            "<|open|>call tool=\"g\" index=\"2\"<|sep|>",
            "<|open|>call tool=\"h\" index=\"3\"<|sep|>",
            "<|open|>argument key=\"b\" type=\"string\"<|sep|>x",
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        let starts: Vec<(u32, &str)> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, name, .. } => Some((*index, name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, [(0, "f"), (1, "g"), (2, "h")]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": true}"#);
        assert_eq!(arguments_of(&events, 1), "{}");
        // The block's close cut `h` inside a string: the string and the object are closed.
        assert_eq!(arguments_of(&events, 2), r#"{"b": "x"}"#);
        assert_eq!(ends(&events), [xtml::CALL_CLOSE, "", ""]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 3, .. })
        ));
    }

    #[test]
    fn the_end_of_turn_tag_closes_whatever_is_open() {
        // A thought, an answer and a call each cut by `<|close|>message<|sep|>`: the region ends,
        // and the tag is wrapping.
        let thought = "A thought.<|close|>message<|sep|>";
        let events = run(PROMPT, &[thought]);
        assert_eq!(bytes(&events), thought);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("A thought.")),
                dropped("<|close|>message<|sep|>"),
                Event::ReasoningEnd,
            ]
        );

        let answer = "<|close|>think<|sep|><|open|>response<|sep|>Hi<|close|>message<|sep|>";
        let events = run(PROMPT, &[answer]);
        assert_eq!(bytes(&events), answer);
        assert_eq!(content(&events), "Hi");

        let call = concat!(
            "<|close|>think<|sep|><|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"null\"<|sep|>null<|close|>argument<|sep|>",
            "<|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[call]);
        assert_eq!(bytes(&events), call);
        assert_eq!(arguments_of(&events, 0), r#"{"a": null}"#);
        assert_eq!(ends(&events), [""]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_region_opened_inside_the_thought_ends_it() {
        let output = concat!(
            "Let me call.<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Let me call.")),
                dropped("<|open|>tools<|sep|>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(arguments_of(&events, 0), "{}");

        // The answer's opening ends the thought the same way, at every cut.
        let answer =
            "A plan.<|open|>response<|sep|>Hi<|close|>response<|sep|><|close|>message<|sep|>";
        for cut in 0..answer.len() {
            let pieces: Vec<&str> = if cut == 0 {
                vec![answer]
            } else {
                vec![&answer[..cut], &answer[cut..]]
            };
            let events = run(PROMPT, &pieces);
            assert_eq!(bytes(&events), answer, "cut at {cut}");
            assert_eq!(
                events[..4],
                [
                    Event::ReasoningStart,
                    Event::Reasoning(Text::uncounted("A plan.")),
                    dropped("<|open|>response<|sep|>"),
                    Event::ReasoningEnd,
                ],
                "cut at {cut}"
            );
            assert_eq!(content(&events), "Hi", "cut at {cut}");
        }
    }

    #[test]
    fn a_tools_block_opened_inside_the_answer_ends_it() {
        let output = concat!(
            "<|close|>think<|sep|><|open|>response<|sep|>Hi<|open|>tools<|sep|>",
            "<|open|>call tool=\"f\" index=\"1\"<|sep|><|close|>call<|sep|><|close|>tools<|sep|>",
            "<|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Hi");
        assert_eq!(arguments_of(&events, 0), "{}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn text_between_the_regions_is_reported_and_whitespace_there_is_wrapping() {
        let output = concat!(
            "<|close|>think<|sep|>\nHello<|open|>response<|sep|>Hi<|close|>response<|sep|>",
            "<|close|>message<|sep|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Hi");
        assert!(events.contains(&dropped("\n")));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::Other(_) } if text.text == "Hello"
        )));
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_the_call_with_its_arguments_cut() {
        let events = run(
            PROMPT,
            &[concat!(
                "<|close|>think<|sep|><|open|>tools<|sep|>",
                "<|open|>call tool=\"f\" index=\"1\"<|sep|>",
                "<|open|>argument key=\"x\" type=\"number\"<|sep|>1"
            )],
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 0, source } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn an_earlier_turns_regions_in_the_prompt_move_nothing() {
        // The replay starts at the last assistant message tag, so an earlier turn's call block,
        // closed or not, and a user's quoted tag are not entered.
        for earlier in [
            concat!(
                "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
                "<|close|>call<|sep|>",
                "<|close|>tools<|sep|>"
            ),
            "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"1\"<|sep|>",
        ] {
            let prompt = format!(
                "<|open|>message role=\"user\"<|sep|>q<|close|>message<|sep|>\
                 <|open|>message role=\"assistant\"<|sep|><|close|>think<|sep|>{earlier}\
                 <|close|>message<|sep|><|open|>message role=\"user\"<|sep|>\
                 Why <|open|>tools<|sep|>?<|close|>message<|sep|>{PROMPT}"
            );
            let events = run(&prompt, &["A thought.<|close|>think<|sep|>"]);
            assert_eq!(events[0], Event::ReasoningStart, "{earlier:?}");
            assert_eq!(
                events[1],
                Event::Reasoning(Text::uncounted("A thought.")),
                "{earlier:?}"
            );
        }
    }

    #[test]
    fn a_call_that_named_nothing_is_reported_however_it_ended() {
        for (output, ending) in [
            (
                concat!(
                    "<|close|>think<|sep|><|open|>tools<|sep|><|open|>call tool=\"",
                    "<|close|>tools<|sep|>"
                ),
                "<|close|>tools<|sep|>",
            ),
            (
                concat!(
                    "<|close|>think<|sep|><|open|>tools<|sep|><|open|>call tool=\"",
                    "<|open|>call tool=\"g\" index=\"2\"<|sep|><|close|>call<|sep|>",
                    "<|close|>tools<|sep|>"
                ),
                "<|open|>call tool=\"",
            ),
            (
                "<|close|>think<|sep|><|open|>tools<|sep|><|open|>call tool=\"<|close|>call<|sep|>",
                "<|close|>call<|sep|>",
            ),
        ] {
            let events = run(PROMPT, &[output]);
            assert_eq!(bytes(&events), output, "{ending}");
            assert!(
                events.iter().any(|event| matches!(
                    event,
                    Event::Malformed { text, .. } if text.text == ending
                )),
                "{ending}: {events:?}"
            );
        }
    }
}

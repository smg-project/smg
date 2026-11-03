//! Gemma 4: reasoning between `<|channel>thought\n` and `<channel|>`, each call between
//! `<|tool_call>` and `<tool_call|>` in Gemma's own spelling (`call:NAME{key:value,…}`, a string
//! between `<|"|>` tokens, a null `null`: [`gemma`](crate::gemma)), everything else content.
//! Recorded as `gemma-4-e4b-it` (google/gemma-4-E4B-it), 54,416 parse cases, 15,669 of them with
//! calls:
//!
//! ```text
//! <|channel>thought
//! The user wants two songs.
//! <channel|><|tool_call>call:play{artist:<|"|>Taylor Swift<|"|>,n:20}<tool_call|>
//! ```
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::{
    format::{CallSyntax, Emits, Format},
    gemma::Spelling,
};

/// The Gemma 4 table.
pub fn gemma4() -> Format {
    Format::new("gemma4")
        .terminal("think_open", "<|channel>thought\n")
        .terminal("think_close", "<channel|>")
        .terminal("call_open", "<|tool_call>")
        .terminal("call_close", "<tool_call|>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("reasoning", "call_open", "call")
        .transition("call", "call_close", "content")
        .calls(CallSyntax::Gemma(Spelling::GEMMA4))
        .opens_turn("<|turn>model")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{Event, Events, FinishReason, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{
            arguments_of, bytes, content, drive, drive_to, reasoning, the_same_at_every_cut,
        },
        Declared,
    };

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "play".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "artist": {"type": "string"},
                    "n": {"type": "integer"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail: the turn opened and nothing more.
    const PROMPT: &str = "<|turn>model\n";

    /// An output as the template renders a turn: the thought, two calls with nothing between
    /// them, then the content.
    const OUTPUT: &str = concat!(
        "<|channel>thought\nThe user wants two songs.\n<channel|>",
        "<|tool_call>call:play{artist:<|\"|>Taylor Swift<|\"|>,n:20}<tool_call|>",
        "<|tool_call>call:play{artist:<|\"|>Maroon 5<|\"|>,n:15}<tool_call|>Two songs, then."
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut Engine::new(gemma4(), declared()), prompt, pieces)
    }

    fn malformed(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.to_string()),
                _ => None,
            })
            .collect()
    }

    /// Prose after the call's object, before its closing marker: the chat message's content and
    /// the stream's deltas are the model's words with the spaces between them, whole, at every
    /// cut and one character at a time alike; the call keeps its arguments.
    #[test]
    fn text_after_the_object_reaches_the_client_as_the_model_wrote_it() {
        let output = "<|tool_call>call:play{artist:<|\"|>Maroon 5<|\"|>} please note this and that <tool_call|>";
        let prose = "please note this and that";
        let shown = |events: &[Event]| chat::message(0, events).message.content;
        let streamed = |events: &[Event]| -> String {
            chat::deltas(0, events)
                .into_iter()
                .filter_map(|choice| choice.delta.content)
                .collect()
        };
        let whole = run(PROMPT, &[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(arguments_of(&whole, 0), r#"{"artist": "Maroon 5"}"#);
        assert_eq!(
            malformed(&whole),
            ["please", " note", " this", " and", " that"]
        );
        assert_eq!(shown(&whole).as_deref(), Some(prose));
        assert_eq!(streamed(&whole), prose);
        for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
            let events = run(PROMPT, &[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(shown(&events).as_deref(), Some(prose), "cut at {cut}");
            assert_eq!(streamed(&events), prose, "cut at {cut}");
        }
        let pieces: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let events = run(PROMPT, &pieces);
        assert_eq!(shown(&events).as_deref(), Some(prose));
        assert_eq!(streamed(&events), prose);
    }

    #[test]
    fn a_rendered_turn_gives_its_thought_two_calls_and_its_content_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "The user wants two songs.\n");
        assert_eq!(content(&events), "Two songs, then.");
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Taylor Swift", "n": 20}"#
        );
        assert_eq!(
            arguments_of(&events, 1),
            r#"{"artist": "Maroon 5", "n": 15}"#
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["play", "play"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 1, source } if source.text == "<tool_call|>"
        )));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_call_cut_across_pieces_streams_its_string_and_reads_the_same() {
        let whole = run(PROMPT, &[OUTPUT]);
        let boundaries: Vec<usize> = OUTPUT.char_indices().map(|(at, _)| at).collect();
        for &at in boundaries.iter().skip(1).step_by(7) {
            let events = run(PROMPT, &[&OUTPUT[..at], &OUTPUT[at..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {at}");
            assert_eq!(
                arguments_of(&events, 0),
                arguments_of(&whole, 0),
                "cut at {at}"
            );
            assert_eq!(
                arguments_of(&events, 1),
                arguments_of(&whole, 1),
                "cut at {at}"
            );
            assert_eq!(content(&events), content(&whole), "cut at {at}");
            assert_eq!(reasoning(&events), reasoning(&whole), "cut at {at}");
        }
    }

    #[test]
    fn a_string_with_newlines_quotes_and_a_split_token_streams_as_json_text() {
        let pieces = [
            "<|tool_call>call:play{artist:<|\"",
            "|>Say \"hi\"\n",
            "twice<|\"|>}<tool_call|>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Say \"hi\"\ntwice"}"#
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_stream_cut_inside_a_string_leaves_it_open_says_so_and_ends_the_call() {
        let mut parser = Engine::new(gemma4(), declared());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: PROMPT,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "<|tool_call>call:play{artist:<|\"|>Tay",
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Length,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        // Nothing is made up for the string the end cut: the fragments stand, and the one word
        // of no bytes before the call's end says that the arguments are no whole value.
        assert_eq!(arguments_of(&events, 0), r#"{"artist": "Tay"#);
        let ended = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty()))
            .expect("the call ends");
        assert!(matches!(
            &events[ended - 1],
            Event::Malformed { text, .. } if text.text.is_empty()
        ));
    }

    /// A call the token limit cut, the shape of a tool request whose `max_tokens` is below the
    /// call's length: inside a bare member, inside a nested object, inside a string and right
    /// after a member's comma. The call's arguments are the prefix as the model wrote it, the cut
    /// member's key and bytes included and no brace the model never wrote; nothing of the call
    /// reaches a chat client as content; the one word before the call's end says that the
    /// arguments are no whole value; and the finish is the engine's `length`, whole, a character
    /// at a time and at every two-way cut.
    #[test]
    fn a_call_the_token_limit_cut_keeps_its_bytes_as_its_arguments_and_nothing_as_content() {
        for (output, arguments) in [
            (
                "<|tool_call>call:play{artist:<|\"|>Taylor<|\"|>,n:1",
                r#"{"artist": "Taylor", "n": 1"#,
            ),
            (
                "<|tool_call>call:play{artist:<|\"|>Taylor<|\"|>,prefs:{size:<|\"|>lar",
                r#"{"artist": "Taylor", "prefs": {"size": "lar"#,
            ),
            (
                "<|tool_call>call:play{artist:<|\"|>Tay",
                r#"{"artist": "Tay"#,
            ),
            (
                "<|tool_call>call:play{artist:<|\"|>Taylor<|\"|>,",
                r#"{"artist": "Taylor""#,
            ),
        ] {
            let read = |pieces: &[&str]| {
                drive_to(
                    &mut Engine::new(gemma4(), declared()),
                    PROMPT,
                    pieces,
                    EngineFinish::Length,
                )
            };
            let said = the_same_at_every_cut(read, output);
            assert_eq!(
                said.calls,
                vec![("play".to_string(), arguments.to_string())],
                "{output:?}"
            );
            assert_eq!(said.finish, Some((FinishReason::Length, 1)), "{output:?}");
            assert_eq!(said.content, "", "{output:?}");
            assert!(
                serde_json::from_str::<serde_json::Value>(arguments).is_err(),
                "{output:?}: the arguments stay cut"
            );
            let events = read(&[output]);
            assert_eq!(bytes(&events), output, "{output:?}");
            assert!(
                malformed(&events).iter().all(String::is_empty),
                "nothing of the call is content: {events:?}"
            );
            assert!(
                matches!(
                    &events[events.len() - 3..],
                    [
                        Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                        Event::ToolCallEnd { source, .. },
                        Event::Finish { .. },
                    ] if text.text.is_empty() && source.text.is_empty()
                ),
                "{output:?}: {events:?}"
            );
        }
    }

    /// The model's own end of output inside a call, before its closing marker: the table has no
    /// stop row, so the engine ends the call as a cut ends it (the braces and tokens are the
    /// model's own), and the call stays open, where it used to be closed with a brace the model
    /// never wrote; whole, a character at a time and at every two-way cut.
    #[test]
    fn a_call_the_model_leaves_open_at_its_own_end_stays_open_too() {
        let output = "<|tool_call>call:play{artist:<|\"|>Taylor<|\"|>,n:1";
        let said = the_same_at_every_cut(|pieces| run(PROMPT, pieces), output);
        assert_eq!(
            said.calls,
            vec![(
                "play".to_string(),
                r#"{"artist": "Taylor", "n": 1"#.to_string()
            )]
        );
        assert!(matches!(said.finish, Some((_, 1))), "{said:?}");
        assert_eq!(said.content, "");
        let events = run(PROMPT, &[output]);
        assert!(
            malformed(&events).iter().all(String::is_empty),
            "nothing of the call is content: {events:?}"
        );
    }

    #[test]
    fn the_table_validates() {
        gemma4().validate().expect("a valid table");
    }
}

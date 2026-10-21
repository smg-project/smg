//! MiniCPM5: reasoning between `<think>` and `</think>`, each call between `<function name="…">`
//! and `</function>` as `<param name="…">` tags whose key is the attribute and whose value the
//! request's tools type, everything else content, and no block around the calls. Recorded as
//! `minicpm5-2b` (openbmb/MiniCPM5-2B), 54,181 parse cases, 15,432 of them with calls:
//!
//! ```text
//! <think>
//! The user wants two songs.
//! </think>
//!
//! <function name="play"><param name="artist">Swift</param><param name="n">20</param></function>
//! <function name="play"><param name="artist">Maroon 5</param><param name="n">15</param></function>
//! ```
//!
//! The table carries the one part of its chat template's layout the tags write: after the
//! thought's close the template writes two newlines before the content or the first call
//! (`</think>\n\n<function name="…">`), so a forced call written where the close left the engine
//! begins with them, as the template writes it, and the call's marker is not the grammar's first
//! text ([`Format::grammar_for`]). A call's opener inside the thought opens the call (the row out
//! of `reasoning` on it), so a call an engine started there is read as the call.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed,
};

/// The MiniCPM5 table.
pub fn minicpm5() -> Format {
    Format::new("minicpm5")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("function_open", "<function name=\"")
        .terminal("function_close", attributed::Tags::MINICPM5.invoke_close)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("function", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "function_open", "function")
        .transition("reasoning", "function_open", "function")
        .transition("function", "function_close", "content")
        .calls(CallSyntax::Attributed(attributed::Tags::MINICPM5))
        .opens_turn("<|im_start|>assistant")
        .with_layout(super::thought_layout("</think>", "\n\n"))
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, FinishReason},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{
            a_forced_tag_begins_with_the_templates_text, arguments_of, bytes, content, drive,
            reasoning, the_same_at_every_cut, Said,
        },
        Declared,
    };

    /// The two recorded tools, as a request lists them.
    fn tools() -> Vec<Tool> {
        let tool = |name: &str, parameters: serde_json::Value| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        };
        vec![
            tool(
                "spotify_play",
                value!({"type": "object", "properties": {
                    "artist": {"type": "string"},
                    "duration": {"type": "integer"},
                    "shuffle": {"type": "boolean"},
                    "tags": {"type": "array"},
                }}),
            ),
            tool(
                "submit_patch",
                value!({"type": "object", "properties": {
                    "patch": {"type": "string"},
                }}),
            ),
        ]
    }

    fn declared() -> Declared {
        Declared::of(&tools())
    }

    /// The generation prompt's tail when the request leaves thinking to the model.
    const PROMPT: &str = "<|im_start|>assistant\n";

    /// An output as the template renders a turn: the thought, content, then two calls on lines
    /// of their own, nothing between a call's tags.
    const OUTPUT: &str = concat!(
        "<think>\nThe user wants two songs.\n</think>\n\nTwo songs, then.\n",
        "<function name=\"spotify_play\"><param name=\"artist\">Taylor Swift</param>",
        "<param name=\"duration\">20</param></function>\n",
        "<function name=\"spotify_play\"><param name=\"artist\">Maroon 5</param>",
        "<param name=\"duration\">15</param></function>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut Engine::new(minicpm5(), declared()), prompt, pieces)
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

    #[test]
    fn a_rendered_turn_gives_its_thought_its_content_and_two_calls_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "\nThe user wants two songs.\n");
        // The newlines around the thought and the one after the content are the model's; the
        // one between the calls is the template's spacing, dropped.
        assert_eq!(content(&events), "\n\nTwo songs, then.\n");
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Taylor Swift", "duration": 20}"#
        );
        assert_eq!(
            arguments_of(&events, 1),
            r#"{"artist": "Maroon 5", "duration": 15}"#
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["spotify_play", "spotify_play"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 1, source } if source.text == "</function>"
        )));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_cdata_value_is_the_text_inside_the_section_however_it_is_cut() {
        let call = concat!(
            "<function name=\"submit_patch\"><param name=\"patch\"><![CDATA[--- a/x\n+++ b/x\n",
            "+if a < b && c:]]></param></function>"
        );
        let whole = run(PROMPT, &[call]);
        assert_eq!(bytes(&whole), call);
        assert_eq!(
            arguments_of(&whole, 0),
            r#"{"patch": "--- a/x\n+++ b/x\n+if a < b && c:"}"#
        );
        assert!(malformed(&whole).is_empty());
        // Cut inside the opener, inside the text, between the closer's bytes.
        let pieces: Vec<&str> = {
            let a = call.find("CDATA").expect("opener");
            let b = call.find("+++").expect("text");
            let c = call.find("]]>").expect("closer") + 1;
            vec![&call[..a], &call[a..b], &call[b..c], &call[c..]]
        };
        let cut = run(PROMPT, &pieces);
        assert_eq!(bytes(&cut), call);
        assert_eq!(arguments_of(&cut, 0), arguments_of(&whole, 0));
        assert!(malformed(&cut).is_empty());
    }

    #[test]
    fn a_boolean_and_a_list_are_typed_by_the_tool_as_python_prints_them() {
        let call = concat!(
            "<function name=\"spotify_play\"><param name=\"shuffle\">True</param>",
            "<param name=\"tags\">['pop', 'live']</param></function>"
        );
        let events = run(PROMPT, &[call]);
        assert_eq!(bytes(&events), call);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"shuffle": true, "tags": ["pop", "live"]}"#
        );
    }

    #[test]
    fn a_call_right_after_the_thought_and_a_string_cut_across_pieces_stream() {
        let pieces = [
            "<think>\n\n</think>\n\n<function name=\"spotify_pl",
            "ay\"><param name=\"artist\">Tay",
            "lor Swift</param></function>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        // An empty thought is the newline after `<think>` and the one before `</think>`; the two
        // between the thought and the call are the template's spacing, not content.
        assert_eq!(reasoning(&events), "\n\n");
        assert_eq!(content(&events), "");
        assert_eq!(arguments_of(&events, 0), r#"{"artist": "Taylor Swift"}"#);
        // The string streams as its pieces arrive: three fragments and the closing quote.
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            ["{\"artist\": \"", "Tay", "lor Swift", "\"", "}"]
        );
    }

    #[test]
    fn a_stream_cut_inside_a_call_closes_nothing_and_still_ends_the_call() {
        let mut parser = Engine::new(minicpm5(), declared());
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
                    text: "<function name=\"spotify_play\"><param name=\"artist\">Tay",
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
        assert_eq!(arguments_of(&events, 0), "{\"artist\": \"Tay");
        assert!(events.iter().any(
            |event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty())
        ));
    }

    #[test]
    fn the_table_validates() {
        minicpm5().validate().expect("a valid table");
    }

    /// One call as the tag writes it after the thought's close, and as the template renders one.
    const FORCED_CALL: &str =
        "<function name=\"spotify_play\"><param name=\"artist\">Taylor Swift</param></function>";

    #[test]
    fn a_prompt_with_thinking_off_has_written_the_templates_two_newlines_itself() {
        // A request with `enable_thinking` off closes an empty thought in the prompt and writes
        // the two newlines after it: the table reads that from the prompt, and a forced call
        // written from there is the calls at once on every start, the newlines not written a
        // second time; the prompt that leaves the thought to the model has written nothing.
        use crate::testing::a_forced_tag_after_a_prompt_that_wrote_the_templates_text;
        a_forced_tag_after_a_prompt_that_wrote_the_templates_text(
            &minicpm5(),
            &tools(),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            PROMPT,
            "<function name=\"",
        );
    }

    #[test]
    fn a_forced_call_after_the_thoughts_close_begins_with_the_templates_two_newlines() {
        // The template writes two newlines after the thought's close before the content or the
        // first call, and the table carries that part of its layout: where the grammar starts
        // after the close, the calls' tag begins with them, so the call's marker is not the
        // grammar's first text; with the thought closed, at the first token or where the engine
        // does not say, the calls at once, as before.
        a_forced_tag_begins_with_the_templates_text(
            &minicpm5(),
            &tools(),
            "</think>",
            "\n\n",
            "<function name=\"",
        );
    }

    #[test]
    fn a_forced_call_written_after_the_thoughts_close_reads_as_the_call_after_the_thought() {
        // The tag's shape on an engine whose grammar starts after the thought's close: the
        // thought, its close, the template's two newlines and the call. The thought is reasoning,
        // the newlines the template's spacing, the call the call; whole, one character at a time
        // and at every cut.
        let thought = "\nThe user wants a song; the call follows.\n";
        let output = format!("<think>{thought}</think>\n\n{FORCED_CALL}");
        assert_eq!(
            the_same_at_every_cut(|pieces| run(PROMPT, pieces), &output),
            Said {
                calls: vec![(
                    "spotify_play".to_string(),
                    r#"{"artist": "Taylor Swift"}"#.to_string()
                )],
                finish: Some((FinishReason::Stop, 1)),
                content: String::new(),
                reasoning: thought.to_string(),
            }
        );
    }

    #[test]
    fn a_call_opener_inside_the_thought_opens_the_call() {
        // An engine whose reasoning gate took a call's opener the model wrote inside its thought
        // for the thought's end, and started a forced call there, wrote the call inside the
        // thought, the output ending there. The table's row out of the thought on the opener reads
        // the call wherever it stands: one call, the thought the text before it, no content. The
        // tag no longer lets such an engine start there (its first text after the close is the
        // template's newlines, not the opener); the reading is pinned here, whole, one character
        // at a time and at every cut.
        let thought = "\nThe user wants a song: ";
        let output = format!("<think>{thought}{FORCED_CALL}");
        assert_eq!(
            the_same_at_every_cut(|pieces| run(PROMPT, pieces), &output),
            Said {
                calls: vec![(
                    "spotify_play".to_string(),
                    r#"{"artist": "Taylor Swift"}"#.to_string()
                )],
                finish: Some((FinishReason::Stop, 1)),
                content: String::new(),
                reasoning: thought.to_string(),
            }
        );
    }
}

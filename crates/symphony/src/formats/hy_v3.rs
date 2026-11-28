//! Hy V3: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<tool_calls>` and `</tool_calls>`, each call between `<tool_call>` and `</tool_call>` as its
//! name, `<tool_sep>` and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the request's
//! tools, with the template's newline after the separator, after each `</arg_key>` and after each
//! `</arg_value>`, and the block's own before each call and before its close, as K2-Horizon writes
//! its block (`keyed::Tags::HY_V3`). Everything else is content. The prompt writes the thought's
//! markers by the request's `reasoning_effort`: `<think></think>` for `no_think`, the template's
//! default, so the output starts in content, and `<think>` alone for `low` and `high`, so it
//! starts inside the thought (`enable_thinking` moves nothing). A turn opens with
//! `<｜hy_Assistant｜>`, so the prompt's replay starts there. Recorded as `hy-memory-30b-a3b`
//! (tencent/Hy-Memory-30B-A3B, whose config names the HYV3 architecture). Hy4 writes the same
//! grammar with every marker suffixed and no separator after the name ([`hy4()`](super::hy4())).
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The Hy V3 table.
pub fn hy_v3() -> Format {
    Format::new("hy_v3")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<tool_calls>")
        .terminal("calls_close", "</tool_calls>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("calls", "call_open", "call")
        .transition("call", "call_close", "calls")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Keyed(keyed::Tags::HY_V3))
        .opens_turn("<｜hy_Assistant｜>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Text},
        testing::{
            arguments_of, bytes, calls, content, drive, dropped, names, reasoning,
            the_same_at_every_cut, weather_declared,
        },
    };

    /// The generation prompt's tail under the template's default (`reasoning_effort`
    /// `no_think`): the turn opened, and its thought closed at once.
    const PROMPT: &str = "<｜hy_Assistant｜><think></think>";

    /// The tail under `reasoning_effort` `low` or `high`: the turn opened, and its thought open.
    const THINKING: &str = "<｜hy_Assistant｜><think>";

    /// An output as the template renders a turn: content, then one call with a string and an
    /// integer in a block, the separator after the name, each tag on its own line.
    const OUTPUT: &str = concat!(
        "Let me check.<tool_calls>\n<tool_call>get_weather<tool_sep>\n",
        "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n",
        "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n",
        "</tool_call>\n</tool_calls>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(hy_v3(), weather_declared()),
            prompt,
            pieces,
        )
    }

    #[test]
    fn a_rendered_turn_gives_its_content_and_its_call_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        // The prompt closed the thought, so the output starts in content and has no thought.
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ReasoningStart)));
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "Let me check.");
        // The block's opener and its newline are wrapping, then the call's opener; the separator
        // after the name is the call's text, not the name's.
        assert_eq!(
            events[1..4],
            [
                dropped("<tool_calls>"),
                dropped("\n"),
                dropped("<tool_call>")
            ]
        );
        assert_eq!(names(&events), ["get_weather"]);
        // The newlines between the tags are the template's, taken as wrappers, not as values; the
        // integer is typed by the tool.
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_prompt_that_opens_the_thought_starts_the_output_inside_it() {
        // `reasoning_effort` `low` or `high`: the template writes `<think>` alone; the model
        // writes the thought, closes it, and answers.
        let output = "The user asks about Paris.</think>It is cold there.";
        let events = run(THINKING, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.")),
                dropped("</think>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "It is cold there.");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn two_calls_in_one_block_after_the_thought_and_the_content() {
        let output = concat!(
            "Two cities.</think>Here are both.<tool_calls>\n",
            "<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n",
            "<arg_value>Paris</arg_value>\n</tool_call>\n",
            "<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n",
            "<arg_value>Rome</arg_value>\n</tool_call>\n</tool_calls>"
        );
        let events = run(THINKING, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Two cities.");
        assert_eq!(content(&events), "Here are both.");
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"city": "Rome"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_call_without_arguments_is_its_name_and_the_separator() {
        // As the recorded zero-argument call reads: `<tool_call>NAME<tool_sep>\n</tool_call>`.
        let output = "<tool_calls>\n<tool_call>get_weather<tool_sep>\n</tool_call>\n</tool_calls>";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{}".to_string())]
        );
        assert_eq!(content(&events), "");
    }

    #[test]
    fn a_call_marker_in_the_content_is_content() {
        // The table enters a call from its block alone: the syntax quoted in a code fence
        // (bellwether's probe) is content, as the reference reads it.
        let output = "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \
                      \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), output);
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn the_prompt_is_replayed_from_the_turn_opener() {
        // A block marker quoted in the user's turn, then the generation prompt with its closed
        // thought: the replay starts at the last turn opener, so the output is content.
        let prompt =
            "<｜hy_User｜>Why did you print <tool_calls> there?<｜hy_Assistant｜><think></think>";
        let output = "It opens the turn's block of calls.";
        let events = run(prompt, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), output);
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let output = format!("The user asks.</think>{OUTPUT}");
        let said = the_same_at_every_cut(|pieces| run(THINKING, pieces), &output);
        assert_eq!(said.calls.len(), 1);
    }
}

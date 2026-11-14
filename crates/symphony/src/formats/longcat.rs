//! LongCat: reasoning between `<longcat_think>` and `</longcat_think>`, each call between
//! `<longcat_tool_call>` and `</longcat_tool_call>` as its name and keyed arguments
//! (`<longcat_arg_key>`, `<longcat_arg_value>`), typed by the request's tools, with the template's
//! newline after the name, after each `</longcat_arg_key>` and after each `</longcat_arg_value>`,
//! as GLM 4.5 and 4.6 write theirs under the plain tags ([`glm4()`](super::glm4());
//! `keyed::Tags::LONGCAT`), and the template's newline after each call's close. Everything else
//! is content. The template writes the thinking switch into the user turn (`/think_on` when the
//! request's `enable_thinking` is true, `/think_off` otherwise, its default; `reasoning_effort`
//! moves nothing) and the assistant header after it: `<longcat_think>` alone when thinking is on,
//! so the output starts inside the thought, and `<longcat_think>\n\n</longcat_think>\n` otherwise,
//! so it starts in content. A turn opens with `<longcat_assistant>`, so the prompt's replay starts
//! there. The template writes a call right after the content's last byte. Recorded as
//! `longcat-2.5` (meituan-longcat/LongCat-2.5).
//!
//! The table carries the one part of its chat template's layout the tags write: after the
//! thought's close the template writes a newline before the content or the first call, so a
//! forced call written where the model's own close left the engine begins on its own line, as
//! GLM 4.6's does ([`Format::grammar_for`]); the default prompt closed the thought itself and
//! wrote that newline (`</longcat_think>\n`), so there the tag owes none of it and begins with the
//! call's marker on every start ([`Format::thought_closed`]), where GLM 4.6's bare close leaves
//! the newline to the tag.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The LongCat table.
pub fn longcat() -> Format {
    Format::new("longcat")
        .terminal("think_open", "<longcat_think>")
        .terminal("think_close", "</longcat_think>")
        .terminal("call_open", "<longcat_tool_call>")
        .terminal("call_close", "</longcat_tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .transition("call", "call_open", "call")
        .calls(CallSyntax::Keyed(keyed::Tags::LONGCAT))
        .opens_turn("<longcat_assistant>")
        .with_layout(super::thought_layout("</longcat_think>", "\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Text},
        testing::{
            a_forced_tag_after_a_prompt_that_wrote_the_templates_text,
            a_forced_tag_begins_with_the_templates_text, arguments_of, bytes, calls, content,
            drive, dropped, names, reasoning, the_same_at_every_cut, weather_declared,
            weather_tools,
        },
    };

    /// The generation prompt's tail under the template's default (`/think_off` in the user turn):
    /// the turn opened, an empty thought closed, and the template's newline after the close.
    const PROMPT: &str = "<longcat_assistant><longcat_think>\n\n</longcat_think>\n";

    /// The tail when the request turns thinking on (`/think_on`): the turn opened, and its
    /// thought open.
    const THINKING: &str = "<longcat_assistant><longcat_think>";

    /// An output as the template renders a turn: content, then one call with a string and an
    /// integer right after it, each tag on its own line, and the newline after the close that
    /// ends the turn.
    const OUTPUT: &str = concat!(
        "Let me check.<longcat_tool_call>get_weather\n",
        "<longcat_arg_key>city</longcat_arg_key>\n",
        "<longcat_arg_value>Paris</longcat_arg_value>\n",
        "<longcat_arg_key>days</longcat_arg_key>\n",
        "<longcat_arg_value>3</longcat_arg_value>\n",
        "</longcat_tool_call>\n"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(longcat(), weather_declared()),
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
        // The call follows the content's last byte; its opener is a wrapper.
        assert_eq!(events[1], dropped("<longcat_tool_call>"));
        assert_eq!(names(&events), ["get_weather"]);
        // The newlines between the tags are the template's, taken as wrappers, not as values; the
        // integer is typed by the tool.
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);
        // The newline after the close is the template's spacing: a stay of whitespace alone,
        // dropped when the stream ends, never content.
        assert!(events.contains(&Event::Dropped {
            text: Text::uncounted("\n"),
            why: DropReason::Whitespace,
        }));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_prompt_that_opens_the_thought_starts_the_output_inside_it() {
        // `/think_on`: the template writes `<longcat_think>` alone; the model writes the thought,
        // closes it, and answers.
        let output = "The user asks about Paris.</longcat_think>It is cold there.";
        let events = run(THINKING, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.")),
                dropped("</longcat_think>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "It is cold there.");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn two_calls_after_the_thought_and_the_content_with_the_templates_newline_between() {
        let output = concat!(
            "Two cities.</longcat_think>Here are both.<longcat_tool_call>get_weather\n",
            "<longcat_arg_key>city</longcat_arg_key>\n",
            "<longcat_arg_value>Paris</longcat_arg_value>\n",
            "</longcat_tool_call>\n<longcat_tool_call>get_weather\n",
            "<longcat_arg_key>city</longcat_arg_key>\n",
            "<longcat_arg_value>Rome</longcat_arg_value>\n",
            "</longcat_tool_call>\n"
        );
        let events = run(THINKING, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Two cities.");
        // The newline between the calls and the one after the last are the template's spacing,
        // dropped; the content is the words before the first call.
        assert_eq!(content(&events), "Here are both.");
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"city": "Rome"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_call_without_arguments_is_its_name_and_the_newline() {
        // As the recorded zero-argument call reads, then the template's newline:
        // `<longcat_tool_call>NAME\n</longcat_tool_call>`.
        let output = "<longcat_tool_call>get_weather\n</longcat_tool_call>\n";
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
        // The table enters a call at its own marker alone: the plain syntax quoted in a code fence
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
        // A call marker quoted in the user's turn, then the generation prompt with its closed
        // thought: the replay starts at the last turn opener, so the output is content.
        let prompt = concat!(
            "<longcat_user>Why did you print <longcat_tool_call> there? /think_off ",
            "<longcat_assistant><longcat_think>\n\n</longcat_think>\n"
        );
        let output = "It opens a call.";
        let events = run(prompt, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), output);
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let output = format!("The user asks.</longcat_think>{OUTPUT}");
        let said = the_same_at_every_cut(|pieces| run(THINKING, pieces), &output);
        assert_eq!(said.calls.len(), 1);
    }

    #[test]
    fn a_prompt_with_thinking_off_has_written_the_templates_newline_itself() {
        // `/think_off`, the default: the prompt closes an empty thought and writes the newline
        // after it, so the table reads the thought as closed and a forced call written from there
        // is the calls at once on every start, the newline not written a second time; the prompt
        // that opens the thought (`/think_on`) has written nothing after a close.
        a_forced_tag_after_a_prompt_that_wrote_the_templates_text(
            &longcat(),
            &weather_tools(),
            PROMPT,
            THINKING,
            "<longcat_tool_call>",
        );
    }

    #[test]
    fn a_forced_call_after_the_thoughts_close_begins_with_the_templates_newline() {
        // The template writes one newline after the thought's close before the content or the
        // first call (`</longcat_think>\n<longcat_tool_call>` in the recorded history turns), and
        // the table carries that part of its layout: where the grammar starts after the model's
        // own close, the calls' tag begins with it; with the thought closed, at the first token or
        // where the engine does not say, the calls at once.
        a_forced_tag_begins_with_the_templates_text(
            &longcat(),
            &weather_tools(),
            "</longcat_think>",
            "\n",
            "<longcat_tool_call>",
        );
    }

    #[test]
    fn the_table_validates() {
        longcat().validate().expect("a sound table");
    }
}

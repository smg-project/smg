//! Inkling: a turn of messages, each opened by `<|message_model|>` and closed by `<|end_message|>`,
//! and what a message is comes after its opener: `<|content_thinking|>` makes it the thought,
//! `<|content_text|>` the answer, and a tool's name followed by `<|content_invoke_tool_json|>` a
//! call, written as one JSON object `{"name": …, "args": {…}}`. The template writes an empty
//! answer (`<|content_text|><|end_message|>`) before the calls when the turn has none, and every
//! message of a turn in its own `<|message_model|>`. The generation prompt is `<|message_model|>`
//! alone, whatever the request says: the template has no thinking switch (a system message carries
//! the `reasoning_effort`), so the output starts where a message's kind is said, and the model
//! writes the thought or not. Everything the markers do not frame is the message's text. Recorded
//! as `inkling` (thinkingmachines/Inkling).
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The marker between a call's name and its object.
pub const INVOKE_JSON: &str = "<|content_invoke_tool_json|>";

/// The Inkling table.
pub fn inkling() -> Format {
    Format::new("inkling")
        .terminal("thinking", "<|content_thinking|>")
        .terminal("text", "<|content_text|>")
        .terminal("end_message", "<|end_message|>")
        .terminal("message", "<|message_model|>")
        .ignores("end_sampling", "<|content_model_end_sampling|>")
        .state("header", Emits::Arguments)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("ended", Emits::Wrapper)
        .transition("header", "thinking", "reasoning")
        .transition("header", "text", "content")
        .transition("header", "end_message", "ended")
        .transition("reasoning", "end_message", "ended")
        .transition("content", "end_message", "ended")
        .transition("ended", "message", "header")
        .calls(CallSyntax::NamedJson(INVOKE_JSON))
        .opens_turn("<|message_model|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// The generation prompt's tail: the turn's first message opened, and nothing more.
    const PROMPT: &str = "<|message_model|>";

    /// A recorded output: the thought, an empty answer, one call.
    const OUTPUT: &str = concat!(
        "<|content_thinking|>The user asks about Paris.<|end_message|><|message_model|>",
        "<|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>",
        "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(inkling(), Declared::default());
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

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn names(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.clone()),
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

    fn malformed_count(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, Event::Malformed { .. }))
            .count()
    }

    #[test]
    fn a_recorded_turn_gives_its_thought_its_empty_answer_and_its_call_with_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..4],
            [
                dropped("<|content_thinking|>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.")),
                dropped("<|end_message|>"),
            ]
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "");
        assert_eq!(names(&events), ["get_weather"]);
        assert_eq!(arguments_of(&events, 0), "{\"city\":\"Paris\"}");
        assert_eq!(malformed_count(&events), 0);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn an_answer_alone_is_content_between_its_markers() {
        let output = "<|content_text|>Hello!<|end_message|>";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..3],
            [
                dropped("<|content_text|>"),
                Event::Content(Text::uncounted("Hello!")),
                dropped("<|end_message|>"),
            ]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn an_answer_then_two_calls_each_in_its_own_message() {
        let output = concat!(
            "<|content_text|>Let me check.<|end_message|><|message_model|>get_weather",
            "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}",
            "<|end_message|><|message_model|>get_stock_price<|content_invoke_tool_json|>",
            "{\"name\":\"get_stock_price\",\"args\":{\"ticker\":\"AAPL\"}}<|end_message|>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(names(&events), ["get_weather", "get_stock_price"]);
        assert_eq!(arguments_of(&events, 0), "{\"city\":\"Paris\"}");
        assert_eq!(arguments_of(&events, 1), "{\"ticker\":\"AAPL\"}");
        // The names before the markers are the template's framing.
        assert!(events.contains(&dropped("get_weather<|content_invoke_tool_json|>")));
        assert_eq!(malformed_count(&events), 0);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_call_cut_across_pieces_streams_its_arguments_as_they_arrive() {
        let pieces = [
            "<|content_text|><|end_message|><|message_mo",
            "del|>get_weather<|content_invoke_",
            "tool_json|>{\"name\":\"get_weather\",\"args\":{\"ci",
            "ty\":\"Paris\"}}<|end_message|>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(names(&events), ["get_weather"]);
        assert_eq!(arguments_of(&events, 0), "{\"city\":\"Paris\"}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_the_call_with_what_arrived() {
        let output = "<|content_text|><|end_message|><|message_model|>get_weather\
                      <|content_invoke_tool_json|>\
                      {\"name\":\"get_weather\",\"args\":{\"city\":\"Pa";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(names(&events), ["get_weather"]);
        assert_eq!(arguments_of(&events, 0), "{\"city\":\"Pa");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_turns_end_token_is_dropped_wherever_it_stands() {
        let output = "<|content_text|>Hello!<|end_message|><|content_model_end_sampling|>";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Hello!");
        assert!(events.contains(&dropped("<|content_model_end_sampling|>")));
        assert_eq!(malformed_count(&events), 0);
    }

    #[test]
    fn the_table_validates() {
        inkling().validate().expect("a sound table");
    }
}

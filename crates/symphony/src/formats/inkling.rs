//! Inkling: a turn of messages, each opened by `<|message_model|>` and closed by `<|end_message|>`,
//! and what a message is comes after its opener: `<|content_thinking|>` makes it the thought,
//! `<|content_text|>` the answer, and a tool's name followed by `<|content_invoke_tool_json|>` a
//! call, written as one JSON object `{"name": …, "args": {…}}`. The template writes an empty
//! answer (`<|content_text|><|end_message|>`) before the calls when the turn has none, and every
//! message of a turn in its own `<|message_model|>`. The generation prompt is `<|message_model|>`
//! alone, whatever the request says: the template has no thinking switch (a system message carries
//! the `reasoning_effort`), so the output starts where a message's kind is said, and the model
//! writes the thought or not. Everything the markers do not frame is the message's text. Recorded
//! as `inkling` (thinkingmachines/Inkling); Inkling-Small and Inkling-NVFP4 ship the same template
//! byte for byte and read with it.
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
        event::{DropReason, Event, Text},
        testing::{arguments_of, bytes, content, drive, names, reasoning},
        Declared,
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
        drive(
            &mut Engine::new(inkling(), Declared::default()),
            prompt,
            pieces,
        )
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

    /// An output recorded from a request that declared no tools, as the gateway reads it then
    /// (`Format::without_calls`): the thought in its message, the answer in the next.
    const RECORDED_THOUGHT_AND_ANSWER: &str = concat!(
        "<|content_thinking|>We need solve math problem. Shop sells pens at 3 for 2 dollars. ",
        "So price per pen = 2/3 dollars. 18 pens cost 18 * 2/3 = 12 dollars. Alternatively ",
        "18/3=6 groups, each 2 dollars => 12. Answer: 12.<|end_message|>",
        "<|message_model|><|content_text|>12<|end_message|>"
    );

    /// The thought of [`RECORDED_THOUGHT_AND_ANSWER`].
    const RECORDED_THOUGHT: &str = concat!(
        "We need solve math problem. Shop sells pens at 3 for 2 dollars. So price per pen = 2/3 ",
        "dollars. 18 pens cost 18 * 2/3 = 12 dollars. Alternatively 18/3=6 groups, each 2 ",
        "dollars => 12. Answer: 12."
    );

    /// Another recorded output of a request without tools, with a longer answer.
    const RECORDED_REWRITE: &str = concat!(
        "<|content_thinking|>The user wants to rewrite \"The meeting was postponed by the ",
        "manager\" in the active voice.\n\nOriginal: The meeting was postponed by the manager.\n",
        "- Subject: The meeting\n- Verb: was postponed (passive)\n- Agent: by the manager\n\n",
        "Active voice: The manager postponed the meeting.\n\nThis is straightforward. I should ",
        "provide the rewritten sentence clearly.<|end_message|><|message_model|>",
        "<|content_text|>The manager postponed the meeting.<|end_message|>"
    );

    /// A recorded output of a request without tools whose answer a JSON object was asked for:
    /// the engine's grammar wrote the text marker again after the model's own.
    const RECORDED_JSON_ANSWER: &str = concat!(
        "<|content_thinking|>The user asks: \"Give the facts about Tokyo.\" The system ",
        "instruction says: \"Answer with a JSON object with the keys city, country and ",
        "population.\"\n\nSo I need to provide facts about Tokyo, but specifically format the ",
        "answer as a JSON object with keys: city, country, population.\n\nI should include the ",
        "relevant values:\n- city: Tokyo\n- country: Japan\n- population: Approximately 13.9 ",
        "million (for Tokyo proper) or 37 million (for Greater Tokyo Area). Usually \"Tokyo\" ",
        "as a city refers to the prefecture/capital, but the city proper is about 14 million. ",
        "I should provide a clear figure. Let's use ~13.96 million (as of recent estimates) or ",
        "just \"13.9 million\". To be safe and accurate, I can specify the metropolitan area, ",
        "but the key is just population. Let's provide \"13.9 million\" for the city proper, ",
        "or mention both? The instruction is strict: JSON object with those keys. So just ",
        "those three.\n\nLet's construct:\n{\n  \"city\": \"Tokyo\",\n  \"country\": ",
        "\"Japan\",\n  \"population\": \"13.9 million\"\n}\n\nThis satisfies. I will not add ",
        "extra text outside JSON? The instruction says \"Answer with a JSON object...\". ",
        "Usually that means the response should be just the JSON. I will provide just the ",
        "JSON.<|end_message|><|message_model|><|content_text|><|content_text|>{\"city\": ",
        "\"Tokyo\", \"country\": \"Japan\", \"population\": \"13.9 million\"}<|end_message|>",
    );

    /// One turn through the table as a request without tools reads it.
    fn run_without_tools(pieces: &[&str]) -> Vec<Event> {
        drive(
            &mut Engine::new(inkling().without_calls(), Declared::default()),
            PROMPT,
            pieces,
        )
    }

    /// `text` one character at a time.
    fn characters(text: &str) -> Vec<String> {
        text.chars().map(String::from).collect()
    }

    #[test]
    fn without_tools_a_recorded_thought_and_answer_keep_their_framing_out_of_the_content() {
        let events = run_without_tools(&[RECORDED_THOUGHT_AND_ANSWER]);
        assert_eq!(bytes(&events), RECORDED_THOUGHT_AND_ANSWER);
        assert_eq!(reasoning(&events), RECORDED_THOUGHT);
        assert_eq!(content(&events), "12");
        assert_eq!(names(&events), [""; 0]);
        assert_eq!(malformed_count(&events), 0);
        assert!(
            events.contains(&dropped("<|message_model|>"))
                && events.contains(&dropped("<|content_text|>")),
            "{events:?}"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn without_tools_the_recorded_outputs_read_the_same_one_character_at_a_time() {
        for (output, thought_tail, answer) in [
            (RECORDED_THOUGHT_AND_ANSWER, "Answer: 12.", "12"),
            (
                RECORDED_REWRITE,
                "provide the rewritten sentence clearly.",
                "The manager postponed the meeting.",
            ),
        ] {
            let pieces = characters(output);
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let events = run_without_tools(&pieces);
            assert_eq!(bytes(&events), output);
            assert!(reasoning(&events).ends_with(thought_tail), "{events:?}");
            assert_eq!(content(&events), answer);
            assert_eq!(malformed_count(&events), 0);
        }
    }

    #[test]
    fn without_tools_a_recorded_json_answer_carries_no_message_framing() {
        // The second text marker stands inside the answer, where the table has no row for it,
        // and is the answer's text as any marker out of place is; the message's framing is not.
        for pieces in [
            vec![RECORDED_JSON_ANSWER.to_string()],
            characters(RECORDED_JSON_ANSWER),
        ] {
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let events = run_without_tools(&pieces);
            assert_eq!(bytes(&events), RECORDED_JSON_ANSWER);
            assert!(reasoning(&events).ends_with("I will provide just the JSON."));
            let content = content(&events);
            assert!(
                !content.contains("<|end_message|>") && !content.contains("<|message_model|>"),
                "{content}"
            );
            assert!(
                content.ends_with("\"population\": \"13.9 million\"}"),
                "{content}"
            );
            assert_eq!(malformed_count(&events), 0);
        }
    }

    #[test]
    fn without_tools_a_call_the_model_writes_anyway_is_content_between_dropped_framing() {
        let events = run_without_tools(&[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(
            content(&events),
            "get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
        );
        assert_eq!(names(&events), [""; 0]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn the_table_validates() {
        inkling().validate().expect("a sound table");
    }
}

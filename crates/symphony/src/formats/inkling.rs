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
        // and is the answer's text as any marker out of place is, said by one `Malformed` event
        // of no bytes before it; the message's framing is not.
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
            assert_eq!(malformed_count(&events), 1);
            assert!(
                events.iter().any(|event| matches!(
                    event,
                    Event::Malformed { text, why: crate::event::MalformedReason::MarkerOutOfPlace }
                        if text.text.is_empty()
                )),
                "{events:?}"
            );
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

/// The JSON answer of a request without tools, from the outputs a live cell recorded: what the
/// table reads back of the answer tag's output, before and after the tag left the text marker to
/// the engine ([`Format::answer_grammar`]).
#[cfg(test)]
mod answer_tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, MalformedReason},
        testing::{bytes, content, drive, reasoning},
        Declared,
    };

    /// The generation prompt's tail: the turn's first message opened, and nothing more.
    const PROMPT: &str = "<|message_model|>";

    /// The thought of a recorded JSON-schema answer (a bug-report triage), as the model wrote it.
    const TRIAGE_THOUGHT: &str = concat!(
        "The user wants me to triage a bug report: \"A typo in the footer: ",
        "Copyrigth.\"\n\nTriage means assessing severity, priority, category, and action ", // codespell:ignore copyrigth
        "needed.\n\nLet's analyze:\n- Issue: Typo in footer text \"Copyrigth\" instead of ", // codespell:ignore copyrigth
        "\"Copyright\"\n- Impact: Very low. Cosmetic/text error. Doesn't break functionality.\n- ",
        "Severity: Trivial/Minor (cosmetic)\n- Priority: Low\n- Category: UI/Text/Copy\n- ",
        "Action: Fix typo, likely quick win.\n\nI should provide a structured triage response ",
        "with:\n- Severity/Impact\n- Priority\n- Category/Component\n- Notes/Recommendation\n- ",
        "Maybe suggested fix\n\nKeep it concise and professional."
    );

    /// The answer the schema shaped, as recorded.
    const TRIAGE_JSON: &str = concat!(
        "{\"severity\": \"low\", \"summary\": \"Typo in footer: 'Copyrigth' should be ", // codespell:ignore copyrigth
        "'Copyright'\", \"needs_followup\": true, \"owner\": \"frontend/content\"}"
    );

    /// The thought of a recorded `json_object` answer (facts about a city), as the model wrote it.
    const CITY_THOUGHT: &str = concat!(
        "The user asks: \"Give the facts about Nairobi.\" The system instruction says: \"Answer ",
        "with a JSON object with the keys city, country and population.\"\n\nSo I need to ",
        "provide facts about Nairobi, but specifically format the answer as a JSON object with ",
        "keys: city, country, population.\n\nI should include the relevant values:\n- city: ",
        "Nairobi\n- country: Kenya\n- population: approximately 4.4 million (city proper) or 5.3 ",
        "million (metro). I should provide a common figure, maybe ~4.4 million (2023 estimate) ",
        "or ~5.3 million for the greater area. Let's use a standard figure like 4,397,073 (2023 ",
        "census) or approximate 4.4 million. To be safe, I can put \"4.4 million\" or the exact ",
        "number. The instruction just says keys city, country and population. It doesn't specify ",
        "exact format for population value, but JSON should have values. I can use a number or ",
        "string. Let's use a number with approximate: 4397073 or \"approximately 4.4 million\". ",
        "Since it's JSON, maybe string is safer for approximate, but number is fine. Let's ",
        "provide \"4.4 million\" as string, or 4397073. Actually, the user asked \"Give the ",
        "facts about Nairobi.\" The constraint is to answer with a JSON object with those keys. ",
        "I should probably include only those keys? The instruction says \"Answer with a JSON ",
        "object with the keys city, country and population.\" It implies the answer should be ",
        "that JSON object. I will provide exactly that.\n\nLet's choose:\n{\n  \"city\": ",
        "\"Nairobi\",\n  \"country\": \"Kenya\",\n  \"population\": 4397073\n}\n\nOr maybe ",
        "include more context? The instruction is strict: \"Answer with a JSON object with the ",
        "keys city, country and population.\" So I will output just the JSON.\n\nWait, is there ",
        "any hidden instruction? The system message is just that. The user asks for facts. I ",
        "must comply with the format.\n\nI'll provide:\n{\n  \"city\": \"Nairobi\",\n  ",
        "\"country\": \"Kenya\",\n  \"population\": \"4.4 million (approx.)\"\n}\n\nBut to be ",
        "precise, let's use the 2019 census figure: 4,397,073. Actually 2023 might be different. ",
        "Let's use \"approximately 4.4 million\". As a JSON value, string is okay.\n\nI'll ",
        "output:\n{\n  \"city\": \"Nairobi\",\n  \"country\": \"Kenya\",\n  \"population\": ",
        "4397073\n}\n\nThis is clean."
    );

    /// The object the request asked for, as recorded.
    const CITY_JSON: &str =
        "{\n  \"city\": \"Nairobi\",\n  \"country\": \"Kenya\",\n  \"population\": 4397073\n}";

    /// A recorded output: the thought's message, then the answer's with the text marker written
    /// twice, as the answer tag had it before it left the marker to the engine: the engine
    /// started the grammar after the model's own marker, and the tag's `begin` wrote it again.
    fn recorded(thought: &str, json: &str) -> String {
        format!(
            "<|content_thinking|>{thought}<|end_message|><|message_model|><|content_text|>\
             <|content_text|>{json}<|end_message|>"
        )
    }

    /// `output` whole, and one character at a time.
    fn chunkings(output: &str) -> [Vec<String>; 2] {
        [
            vec![output.to_string()],
            output.chars().map(String::from).collect(),
        ]
    }

    fn run(format: Format, pieces: &[String]) -> Vec<Event> {
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        drive(
            &mut Engine::new(format, Declared::default()),
            PROMPT,
            &pieces,
        )
    }

    /// The `Malformed` events, as (reason, bytes) pairs.
    fn malformed(events: &[Event]) -> Vec<(MalformedReason, &str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((*why, text.text.as_str())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_recorded_answer_whose_text_marker_was_written_twice_keeps_its_bytes_and_says_so() {
        // The second marker stands inside the answer, where no row leads: it is the answer's
        // text, as any marker out of place is, and the event of no bytes before it counts the
        // frame the table did not read; the framing around it is read and dropped as always.
        for (thought, json) in [(TRIAGE_THOUGHT, TRIAGE_JSON), (CITY_THOUGHT, CITY_JSON)] {
            let output = recorded(thought, json);
            for pieces in chunkings(&output) {
                let events = run(inkling(), &pieces);
                assert_eq!(bytes(&events), output);
                assert_eq!(reasoning(&events), thought);
                assert_eq!(content(&events), format!("<|content_text|>{json}"));
                assert_eq!(
                    malformed(&events),
                    [(MalformedReason::MarkerOutOfPlace, "")],
                    "{events:?}"
                );
                let at = events
                    .iter()
                    .position(|event| matches!(event, Event::Malformed { .. }))
                    .expect("the event");
                assert!(
                    matches!(&events[at + 1], Event::Content(text) if text.text.starts_with("<|content_text|>")),
                    "{events:?}"
                );
                assert!(matches!(
                    events.last(),
                    Some(Event::Finish { tool_calls: 0, .. })
                ));
            }
        }
    }

    #[test]
    fn without_tools_the_second_text_marker_is_said_the_same() {
        // The variant a request without tools reads with keeps the bytes and says so too.
        let output = recorded(TRIAGE_THOUGHT, TRIAGE_JSON);
        for pieces in chunkings(&output) {
            let events = run(inkling().without_calls(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(reasoning(&events), TRIAGE_THOUGHT);
            let content = content(&events);
            assert!(
                content.contains(&format!("<|content_text|>{TRIAGE_JSON}")),
                "{content}"
            );
            assert_eq!(
                malformed(&events),
                [(MalformedReason::MarkerOutOfPlace, "")],
                "{events:?}"
            );
        }
    }

    #[test]
    fn a_second_text_marker_after_whitespace_is_said_right_before_it() {
        // The whitespace the answer's stay holds before its first text is the text's once the
        // marker follows: it comes out first, then the event of no bytes, then the marker, whole
        // and one character at a time; so the event stands where the marker does.
        let output = "<|content_text|>\n<|content_text|>{}<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(content(&events), "\n<|content_text|>{}");
            assert_eq!(
                malformed(&events),
                [(MalformedReason::MarkerOutOfPlace, "")]
            );
            let at = events
                .iter()
                .position(|event| matches!(event, Event::Malformed { .. }))
                .expect("the event");
            assert!(
                matches!(&events[at - 1], Event::Content(text) if text.text == "\n"),
                "{events:?}"
            );
            assert!(
                matches!(&events[at + 1], Event::Content(text) if text.text.starts_with("<|content_text|>")),
                "{events:?}"
            );
        }
    }

    #[test]
    fn without_tools_a_marker_in_the_header_is_no_marker_out_of_place() {
        // The header is a state the no-tools variant made content: a marker with no row out of
        // it is its text and adds no count of its own; the table feeds it to the call's header,
        // which says what it is.
        let output = "<|message_model|><|content_text|>Hi<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling().without_calls(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(malformed(&events), []);
            assert_eq!(
                malformed(&run(inkling(), &pieces)),
                [(MalformedReason::NameWithoutACall, "<|message_model|>")],
                "the table feeds the marker to the call's header"
            );
        }
    }

    #[test]
    fn the_answer_the_tag_leaves_the_model_reads_as_the_bare_answer() {
        // What the tag admits once the marker is the engine's to write or to have consumed: the
        // thought first or not, one text marker, the JSON, the message's close; each reads as the
        // bare answer, whole and one character at a time.
        let thought =
            format!("<|content_thinking|>{TRIAGE_THOUGHT}<|end_message|><|message_model|>");
        for (output, json) in [
            (
                format!("{thought}<|content_text|>{TRIAGE_JSON}<|end_message|>"),
                TRIAGE_JSON,
            ),
            (
                format!("<|content_text|>{CITY_JSON}<|end_message|>"),
                CITY_JSON,
            ),
        ] {
            for pieces in chunkings(&output) {
                let events = run(inkling(), &pieces);
                assert_eq!(bytes(&events), output);
                assert_eq!(content(&events), json);
                assert_eq!(malformed(&events), []);
                assert!(matches!(
                    events.last(),
                    Some(Event::Finish { tool_calls: 0, .. })
                ));
            }
        }
    }
}

//! Inkling: a turn of messages, each opened by `<|message_model|>` and closed by `<|end_message|>`,
//! and what a message is comes after its opener: `<|content_thinking|>` makes it the thought,
//! `<|content_text|>` the answer, and a tool's name followed by `<|content_invoke_tool_json|>` a
//! call, written as one JSON object `{"name": …, "args": {…}}`. The template writes an empty
//! answer (`<|content_text|><|end_message|>`) before the calls when the turn has none, and every
//! message of a turn in its own `<|message_model|>`. The generation prompt is `<|message_model|>`
//! alone, whatever the request says: the template has no thinking switch (a system message carries
//! the `reasoning_effort`), so the output starts where a message's kind is said, and the model
//! writes the thought or not. Everything the markers do not frame is the message's text. A kind
//! said again inside a message's body is the kind, as the family's own parsers read it (the
//! blocks are self-describing and may repeat in any order): `<|content_text|>` in the answer
//! starts the answer's text over and in the thought starts the answer, `<|content_thinking|>` in
//! the answer or in the thought opens a thought; none is the body's text, and none closes the
//! message, which `<|end_message|>` closes wherever its kind was said again: the tag derivation
//! tells a kind said again from a close (`Format::is_kind_switch`), so the answer tag and a
//! forced call's tag end the thought at the message's close the model writes, and the thought's
//! text under them still writes no kind marker. An engine that holds a JSON answer's grammar back
//! until its own reasoning parser has seen the model's `<|content_text|>` lets the model write
//! both there ([`Format::answer_grammar`]: the tag admits the thought and the marker before the
//! schema for an engine that starts at the first token, and cannot tell the two starts apart).
//! Recorded as `inkling` (thinkingmachines/Inkling); Inkling-Small and Inkling-NVFP4 ship the
//! same template byte for byte and read with it.
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
        // The kind said again inside a body: the answer's marker starts its text over or, in the
        // thought, starts the answer; the thought's opens a thought from either body. Framing, as
        // from the header; the message closes where it did (the tag derivation reads these rows
        // as no close: `Format::is_kind_switch`).
        .transition("content", "text", "content")
        .transition("content", "thinking", "reasoning")
        .transition("reasoning", "thinking", "reasoning")
        .transition("reasoning", "text", "content")
        .calls(CallSyntax::NamedJson(INVOKE_JSON))
        .opens_turn("<|message_model|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Text},
        testing::{arguments_of, bytes, content, drive, dropped, names, reasoning},
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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(inkling(), Declared::default()),
            prompt,
            pieces,
        )
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
    fn run_without_tools(pieces: &[&str]) -> Vec<Event<'static>> {
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
        // The second text marker stands inside the answer and says its kind again: framing,
        // dropped as the message's is, and the answer is the bare JSON.
        for pieces in [
            vec![RECORDED_JSON_ANSWER.to_string()],
            characters(RECORDED_JSON_ANSWER),
        ] {
            let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let events = run_without_tools(&pieces);
            assert_eq!(bytes(&events), RECORDED_JSON_ANSWER);
            assert!(reasoning(&events).ends_with("I will provide just the JSON."));
            assert_eq!(
                content(&events),
                "{\"city\": \"Tokyo\", \"country\": \"Japan\", \"population\": \"13.9 million\"}"
            );
            assert_eq!(malformed_count(&events), 0);
            assert_eq!(
                events
                    .iter()
                    .filter(|event| **event == dropped("<|content_text|>"))
                    .count(),
                2,
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

/// The JSON answer of a request without tools, from the outputs live cells recorded: what the
/// table reads back of the answer tag's output ([`Format::answer_grammar`]) on an engine that
/// starts the grammar at the first token, and on one that starts it after its own reasoning
/// parser has seen the model's `<|content_text|>`, where the tag's optional thought and marker
/// are open to the model once more and it writes the kind again.
#[cfg(test)]
mod answer_tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, MalformedReason, Text},
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

    /// A recorded output of a request without tools, on an engine that started the grammar after
    /// its reasoning parser had seen the model's text marker: the first thought, as the model
    /// wrote it before the grammar started.
    const FIRST_THOUGHT: &str = concat!(
        "The user wants me to triage a bug report: \"A typo in the footer: Copyrigth.\"\n\nTriage ", // codespell:ignore copyrigth
        "means assessing severity/priority and categorizing the issue. I should consider:\n- What ",
        "is it? A typo (\"Copyrigth\" instead of \"Copyright\") in the footer.\n- Impact: Very low. ", // codespell:ignore copyrigth
        "Cosmetic/minor. Doesn't affect functionality.\n- Likelihood: Easy fix.\n- Priority: Low/P3 ",
        "or similar.\n- Category: UI/UX, typo, copy error.\n\nI should provide a structured triage ",
        "response: Severity (Low/Minor), Priority (Low), Type (Cosmetic/Typo), Action (Fix in next ",
        "maintenance/release, quick win).\n\nKeep it concise and practical.",
    );

    /// The second thought of that output: the tag's optional thought, which the model wrote where
    /// the grammar started, inside the answer's message.
    const SECOND_THOUGHT: &str = concat!(
        "\nThe user asks to \"triage\" a very brief bug report: \"A typo in the footer: Copyrigth.\" ", // codespell:ignore copyrigth
        "I need to assess severity, priority, category, and recommended action.\n\nKey points:\n- ",
        "Issue: Typo (\"Copyrigth\" missing the 't' in \"Copyright\").\n- Location: Footer.\n- Impact: ", // codespell:ignore copyrigth
        "Purely cosmetic / copy error. No functional impact, no security risk, no data loss, minimal ",
        "user confusion.\n- Fix effort: Trivial (single character/string change).\n- Triage classification: ",
        "Low severity, Low priority (or \"trivial/quick win\"). Could be batched with other copy ",
        "fixes.\n\nStructure of answer:\n1. Quick verdict / summary.\n2. Severity / Priority.\n3. ",
        "Category / Tags.\n4. Rationale (why low).\n5. Recommended action (fix, batch, etc.).\n6. ",
        "Maybe ask for clarification if needed (e.g., which site/page, but not strictly necessary).\n",
        "\nI should be direct and use standard triage terms (e.g., P4/Low, Cosmetic, UI Copy).\n\n",
        "Let's draft:\n- Severity: Low / Minor\n- Priority: Low (P3/P4)\n- Type: Cosmetic / Copy ",
        "/ UI\n- Status: Ready for fix / Quick win\n- Notes: One-character typo in footer. No user ",
        "workflow blocked. Suggest fixing in next content pass or as a micro-fix. If this is a legal/copyright ",
        "notice, slightly more important for legal accuracy, but still cosmetic. I can mention that ",
        "if it's a legal notice, it might warrant slightly faster fix for compliance, but generally ",
        "still low.\n\nKeep it short.\n",
    );

    /// The answer the schema shaped after the second thought, as recorded.
    const SECOND_THOUGHT_JSON: &str = concat!(
        "{\"severity\": \"low\", \"summary\": \"Cosmetic typo in footer text ('Copyrigth' → 'Copyright'). ", // codespell:ignore copyrigth
        "No functional impact.\", \"needs_followup\": false, \"owner\": \"UI/Content team\"}",
    );

    /// A recorded output: the thought's message, then the answer's with the text marker written
    /// twice. An engine that started the grammar after the model's own marker had it forced by
    /// the tag's `begin` before the tag left the marker to the engine, and since then may have
    /// the model write the optional marker again where the grammar starts.
    fn recorded(thought: &str, json: &str) -> String {
        format!(
            "<|content_thinking|>{thought}<|end_message|><|message_model|><|content_text|>\
             <|content_text|>{json}<|end_message|>"
        )
    }

    /// A recorded output: the thought's message, then the answer's opened and, where an engine
    /// that starts the grammar after its reasoning parser hands over, the tag's optional thought
    /// written as a second one, closed and the next message opened, with or without the optional
    /// text marker, then the JSON and the message's close.
    fn recorded_with_a_second_thought(
        thought: &str,
        second: &str,
        opener: bool,
        json: &str,
    ) -> String {
        let opener = if opener { "<|content_text|>" } else { "" };
        format!(
            "<|content_thinking|>{thought}<|end_message|><|message_model|><|content_text|>\
             <|content_thinking|>{second}<|end_message|><|message_model|>{opener}{json}\
             <|end_message|>"
        )
    }

    /// The `Dropped { Wrapper }` events among `events` carrying `text`.
    fn dropped_count(events: &[Event], text: &str) -> usize {
        let dropped = Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        };
        events.iter().filter(|event| **event == dropped).count()
    }

    /// `output` whole, and one character at a time.
    fn chunkings(output: &str) -> [Vec<String>; 2] {
        [
            vec![output.to_string()],
            output.chars().map(String::from).collect(),
        ]
    }

    /// `output` in two pieces, cut at every character boundary.
    fn cuts(output: &str) -> Vec<Vec<String>> {
        output
            .char_indices()
            .skip(1)
            .map(|(at, _)| vec![output[..at].to_string(), output[at..].to_string()])
            .collect()
    }

    fn run(format: Format, pieces: &[String]) -> Vec<Event<'static>> {
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        drive(
            &mut Engine::new(format, Declared::default()),
            PROMPT,
            &pieces,
        )
    }

    /// The `Malformed` events, as (reason, bytes) pairs.
    fn malformed<'a>(events: &'a [Event<'_>]) -> Vec<(MalformedReason, &'a str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((*why, &*text.text)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_text_marker_written_twice_says_the_answers_kind_again_and_the_answer_reads_bare() {
        // The second marker stands inside the answer and starts its text over: framing, dropped
        // as the first is, with no malformed run; the answer is the bare JSON, whole and one
        // character at a time, with the table and with the variant a request without tools reads
        // with.
        for (thought, json) in [(TRIAGE_THOUGHT, TRIAGE_JSON), (CITY_THOUGHT, CITY_JSON)] {
            let output = recorded(thought, json);
            for format in [inkling(), inkling().without_calls()] {
                for pieces in chunkings(&output) {
                    let events = run(format.clone(), &pieces);
                    assert_eq!(bytes(&events), output);
                    assert_eq!(reasoning(&events), thought);
                    assert_eq!(content(&events), json);
                    assert_eq!(malformed(&events), [], "{events:?}");
                    assert_eq!(dropped_count(&events, "<|content_text|>"), 2, "{events:?}");
                    assert!(matches!(
                        events.last(),
                        Some(Event::Finish { tool_calls: 0, .. })
                    ));
                }
            }
        }
    }

    #[test]
    fn a_second_thought_written_in_the_answer_is_a_thought_and_the_answer_reads_bare() {
        // The thought's opener inside the answer opens a thought: its text is reasoning after the
        // first thought's, its close and the next message's opener are framing, and so is the
        // text marker the tag owes after the thought; the answer is the bare JSON, whole and one
        // character at a time, with the table and with its no-tools variant. Without that
        // marker, which the tag no longer admits, the JSON stands in the next message's header:
        // the no-tools variant, whose header is content, still reads it as the answer; the table
        // reads a call's name without a call, as the message says no kind.
        for opener in [true, false] {
            let output = recorded_with_a_second_thought(
                FIRST_THOUGHT,
                SECOND_THOUGHT,
                opener,
                SECOND_THOUGHT_JSON,
            );
            for (format, with_calls) in [(inkling(), true), (inkling().without_calls(), false)] {
                for pieces in chunkings(&output) {
                    let events = run(format.clone(), &pieces);
                    assert_eq!(bytes(&events), output);
                    assert_eq!(
                        reasoning(&events),
                        format!("{FIRST_THOUGHT}{SECOND_THOUGHT}")
                    );
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| matches!(event, Event::ReasoningStart))
                            .count(),
                        2,
                        "two thoughts, each opened: {events:?}"
                    );
                    assert_eq!(dropped_count(&events, "<|content_thinking|>"), 2);
                    assert_eq!(
                        dropped_count(&events, "<|content_text|>"),
                        if opener { 2 } else { 1 }
                    );
                    if opener || !with_calls {
                        assert_eq!(content(&events), SECOND_THOUGHT_JSON);
                        assert_eq!(malformed(&events), [], "{events:?}");
                    } else {
                        assert_eq!(content(&events), "");
                        assert_eq!(
                            malformed(&events),
                            [(MalformedReason::NameWithoutACall, SECOND_THOUGHT_JSON)],
                            "{events:?}"
                        );
                    }
                    assert!(matches!(
                        events.last(),
                        Some(Event::Finish { tool_calls: 0, .. })
                    ));
                }
            }
        }
    }

    #[test]
    fn whitespace_before_a_text_marker_written_again_is_the_templates() {
        // The whitespace the answer's stay holds before its first text ends with the stay the
        // marker ends: the template's spacing, dropped, whole and one character at a time; the
        // answer's text starts over after the marker.
        let output = "<|content_text|>\n<|content_text|>{}<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(content(&events), "{}");
            assert_eq!(malformed(&events), []);
            assert!(
                events.contains(&Event::Dropped {
                    text: Text::uncounted("\n"),
                    why: DropReason::Whitespace,
                }),
                "{events:?}"
            );
        }
    }

    #[test]
    fn a_kind_said_again_after_text_moves_the_body_as_the_familys_parsers_do() {
        // The kind markers move between the bodies wherever they stand, as the family's own
        // parsers read them: a thought opened inside the answer's text, a thought's opener
        // inside a thought, the answer's marker inside a thought; none is the body's text. The
        // message's opener inside a body has no row and stays out of place, counted as before.
        let output = "<|content_text|>Hi <|content_thinking|>more<|end_message|><|message_model|>\
                      <|content_text|>there<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(content(&events), "Hi there");
            assert_eq!(reasoning(&events), "more");
            assert_eq!(malformed(&events), []);
        }
        let output = "<|content_thinking|>one <|content_thinking|>two<|end_message|>\
                      <|message_model|><|content_text|>Hi<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(reasoning(&events), "one two");
            assert_eq!(content(&events), "Hi");
            assert_eq!(malformed(&events), []);
            assert_eq!(dropped_count(&events, "<|content_thinking|>"), 2);
        }
        let output = "<|content_thinking|>plan<|content_text|>{\"a\":1}<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(reasoning(&events), "plan");
            assert_eq!(content(&events), "{\"a\":1}");
            assert_eq!(malformed(&events), []);
            assert_eq!(dropped_count(&events, "<|content_text|>"), 1);
        }
        let output = "<|content_text|>Hi <|message_model|>there<|end_message|>";
        for pieces in chunkings(output) {
            let events = run(inkling(), &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(content(&events), "Hi <|message_model|>there");
            assert_eq!(
                malformed(&events),
                [(MalformedReason::MarkerOutOfPlace, "")],
                "{events:?}"
            );
        }
    }

    #[test]
    fn the_answers_marker_inside_a_thought_starts_the_answer_as_the_familys_parsers_read_it() {
        // The family's parsers switch to the answer at `<|content_text|>` wherever it stands, the
        // thought included: the marker is framing, dropped, the text before it is the thought's,
        // the text after it the answer's, the message closes where it did, and nothing is
        // malformed; whole, one character at a time, and cut in two at every character, with the
        // table and with its no-tools variant. Before the row, such an answer was the thought's
        // text, with empty content and no malformed run.
        for (thought, json) in [("plan", "{\"a\":1}"), (TRIAGE_THOUGHT, TRIAGE_JSON)] {
            let output =
                format!("<|content_thinking|>{thought}<|content_text|>{json}<|end_message|>");
            for format in [inkling(), inkling().without_calls()] {
                for pieces in chunkings(&output).into_iter().chain(cuts(&output)) {
                    let events = run(format.clone(), &pieces);
                    assert_eq!(bytes(&events), output, "{pieces:?}");
                    assert_eq!(reasoning(&events), thought, "{pieces:?}");
                    assert_eq!(content(&events), json, "{pieces:?}");
                    assert_eq!(malformed(&events), [], "{pieces:?}: {events:?}");
                    assert_eq!(dropped_count(&events, "<|content_text|>"), 1, "{pieces:?}");
                    assert_eq!(dropped_count(&events, "<|content_thinking|>"), 1);
                    assert_eq!(dropped_count(&events, "<|end_message|>"), 1);
                    assert!(matches!(
                        events.last(),
                        Some(Event::Finish { tool_calls: 0, .. })
                    ));
                }
            }
        }
    }

    #[test]
    fn the_kinds_may_repeat_in_any_order_inside_one_message() {
        // Thought, answer, thought, answer, each kind said where it starts, in one message: the
        // thoughts are one reasoning in two spans, the answers one content, every marker is
        // framing, the message's close ends it, and nothing is malformed; whole and one
        // character at a time, with the table and with its no-tools variant.
        let output = "<|content_thinking|>one<|content_text|>two<|content_thinking|>three\
                      <|content_text|>four<|end_message|>";
        for format in [inkling(), inkling().without_calls()] {
            for pieces in chunkings(output) {
                let events = run(format.clone(), &pieces);
                assert_eq!(bytes(&events), output);
                assert_eq!(reasoning(&events), "onethree");
                assert_eq!(content(&events), "twofour");
                assert_eq!(malformed(&events), [], "{events:?}");
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| matches!(event, Event::ReasoningStart))
                        .count(),
                    2,
                    "{events:?}"
                );
                assert_eq!(dropped_count(&events, "<|content_thinking|>"), 2);
                assert_eq!(dropped_count(&events, "<|content_text|>"), 2);
                assert!(matches!(
                    events.last(),
                    Some(Event::Finish { tool_calls: 0, .. })
                ));
            }
        }
    }

    #[test]
    fn without_the_thought_the_kind_markers_stay_dropped() {
        // The variant for a request that keeps the reasoning in the content shows the thought's
        // text as content and none of the four markers, with and without tools: the kind markers
        // frame the messages as much as a call, and their rows between the bodies change nothing
        // of that.
        let recorded = recorded_with_a_second_thought(
            FIRST_THOUGHT,
            SECOND_THOUGHT,
            true,
            SECOND_THOUGHT_JSON,
        );
        let answer_in_the_thought =
            "<|content_thinking|>plan<|content_text|>{\"a\":1}<|end_message|>";
        for (output, text) in [
            (
                recorded.as_str(),
                format!("{FIRST_THOUGHT}{SECOND_THOUGHT}{SECOND_THOUGHT_JSON}"),
            ),
            (answer_in_the_thought, "plan{\"a\":1}".to_string()),
        ] {
            for format in [
                inkling().without_reasoning(),
                inkling().without_calls().without_reasoning(),
            ] {
                for pieces in chunkings(output) {
                    let events = run(format.clone(), &pieces);
                    assert_eq!(bytes(&events), output);
                    assert_eq!(reasoning(&events), "");
                    assert_eq!(content(&events), text);
                    assert_eq!(malformed(&events), [], "{events:?}");
                }
            }
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

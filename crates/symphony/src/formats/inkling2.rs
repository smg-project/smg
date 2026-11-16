//! Inkling 2: Inkling's messages, each opened by `<|message_model|>`, its kind said after the
//! opener (`<|content_thinking|>` the thought, `<|content_text|>` the answer, a tool's name and
//! `<|content_invoke_tool_json|>` a call written as one JSON object) and closed by
//! `<|end_message|>`, from a template that opens no turn for the model: it writes the message
//! headers and closes as tokens and no generation opener, so a prompt ends with the last message's
//! close, whatever the request says, and the model writes `<|message_model|>` itself. The table is
//! Inkling's rows with the unopened row for that start: a prompt that holds no `<|message_model|>`,
//! or whose last one is followed by `<|end_message|>` (an assistant message in the history), starts
//! the output between the messages, from where the opener leads into the header; a prompt that
//! continues an assistant message replays from its opener as Inkling's does. The opener said
//! again inside a body is framing, as a kind said again is, and the kind marker after it is the
//! kind: the answer tag's optional way in admits them after an engine that consumed them.
//! Recorded as `inkling-2-small` (thinkingmachines/Inkling-2-Small); the MXFP8 build ships the
//! same template and reads with it.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::format::Format;

/// The Inkling 2 table: Inkling's rows, and the turn the model opens itself.
pub fn inkling2() -> Format {
    super::inkling::rows(Format::new("inkling2"))
        .unopened_turn_starts_in("ended", "<|end_message|>")
        // The opener said again inside a body is framing, as a kind said again is, and the kind
        // marker after it is the kind: the answer tag's optional way in (a start other than the
        // first token cannot say whether the engine consumed the opener and the kind marker)
        // admits them after an engine that has them consumed, and Inkling's rows read `message`
        // out of `ended` alone, so without these rows a second opener was the answer's text.
        .transition("content", "message", "content")
        .transition("reasoning", "message", "reasoning")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::*;
    use crate::{
        engine::Engine,
        event::Event,
        formats::inkling,
        grammar::GrammarStart,
        ids::CallIds,
        testing::{
            arguments_of, bytes, content, drive, dropped, names, reasoning, said, weather_tools,
        },
        Declared, ForcedCall, OutputPosition,
    };

    /// A prompt that opens no turn: the user's message closed, as the template leaves it.
    const UNOPENED: &str = "<|end_message|>";

    /// A recorded output of the group: the model opens its turn, thinks, answers with nothing and
    /// calls a tool, each in a message of its own.
    const OUTPUT: &str = concat!(
        "<|message_model|><|content_thinking|>The user asks about Paris.<|end_message|>",
        "<|message_model|><|content_text|><|end_message|>",
        "<|message_model|>get_weather<|content_invoke_tool_json|>",
        "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>"
    );

    /// An engine whose ids count from the one nonce, so that the events of two engines compare.
    fn engine(format: Format) -> Engine {
        let ids = CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex");
        Engine::with_call_ids(Arc::new(format), Declared::default(), ids)
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut engine(inkling2()), prompt, pieces)
    }

    fn run_inkling(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut engine(inkling()), prompt, pieces)
    }

    fn malformed_count(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, Event::Malformed { .. }))
            .count()
    }

    #[test]
    fn the_table_validates_and_is_inklings_rows() {
        let format = inkling2();
        assert!(format.validate().is_ok());
        assert_eq!(format.name(), "inkling2");
        assert_eq!(format.turn_opener(), Some("<|message_model|>"));
        let terminals: Vec<&str> = format.terminal_texts().collect();
        let inkling = inkling();
        let inklings: Vec<&str> = inkling.terminal_texts().collect();
        assert_eq!(terminals, inklings);
    }

    #[test]
    fn a_turn_the_model_opens_reads_whole_after_a_prompt_that_opens_none() {
        let events = run(UNOPENED, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(events[0], dropped("<|message_model|>"));
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
    fn the_turn_reads_the_same_at_every_cut() {
        let whole = said(&run(UNOPENED, &[OUTPUT]));
        for at in 1..OUTPUT.len() {
            let cut = said(&run(UNOPENED, &[&OUTPUT[..at], &OUTPUT[at..]]));
            assert_eq!(cut, whole, "cut at {at}");
        }
    }

    #[test]
    fn a_prompt_without_the_opener_or_with_its_last_message_closed_starts_between_the_messages() {
        let whole = said(&run(UNOPENED, &[OUTPUT]));
        // The user's text alone, with no marker of the table's.
        assert_eq!(said(&run("Weather in Paris?", &[OUTPUT])), whole);
        // An assistant message in the history, closed, and the user's after it.
        let history = concat!(
            "<|message_model|><|content_text|>Earlier.<|end_message|>",
            "Weather in Paris?<|end_message|>"
        );
        assert_eq!(said(&run(history, &[OUTPUT])), whole);
        // A prompt that ends at the opener (Inkling's generation prompt) opens the turn: the
        // output is read from the header, every byte still in one event.
        assert_eq!(bytes(&run("<|message_model|>", &[OUTPUT])), OUTPUT);
    }

    #[test]
    fn a_continued_message_and_an_empty_prompt_read_as_inklings_do() {
        // The template's continuation leaves the answer open: the replay from the opener finds
        // the stay begun, as Inkling's does.
        let continued = "<|end_message|><|message_model|><|content_text|>Partly";
        let tail = [" cloudy.<|end_message|>"];
        assert_eq!(run(continued, &tail), run_inkling(continued, &tail));
        assert_eq!(content(&run(continued, &tail)), " cloudy.");
        // No prompt at all (the contract's and the robustness tests' form) starts in the header,
        // as every table starts without one: the shapes after the opener read as Inkling's.
        let after_the_opener = &OUTPUT["<|message_model|>".len()..];
        assert_eq!(
            run("", &[after_the_opener]),
            run_inkling("", &[after_the_opener])
        );
    }

    #[test]
    fn a_continued_thought_goes_on_past_an_opener_said_again_inside_it() {
        // The last opener stands inside the thought the prompt left open, where no row reads it as
        // an opener: the replay starts at the message's own, so the output goes on as the thought
        // under both tables.
        let prompt = "<|end_message|><|message_model|><|content_thinking|>Par<|message_model|>is";
        let tail = [" a city.<|end_message|><|message_model|><|content_text|>Yes.<|end_message|>"];
        for (table, events) in [
            ("inkling", run_inkling(prompt, &tail)),
            ("inkling2", run(prompt, &tail)),
        ] {
            assert_eq!(reasoning(&events), " a city.", "{table}: {events:?}");
            assert_eq!(content(&events), "Yes.", "{table}: {events:?}");
            assert!(names(&events).is_empty(), "{table}: {events:?}");
            assert_eq!(malformed_count(&events), 0, "{table}: {events:?}");
        }
        // Under a forced call the table keeps the opener's row and the thought's close: the
        // continued thought replays from the same opener.
        for format in [inkling(), inkling2()] {
            let forced = format.with_forced_call(ForcedCall::Arguments);
            assert_eq!(forced.replay(prompt), &prompt["<|end_message|>".len()..]);
        }
    }

    #[test]
    fn a_continued_answer_goes_on_past_an_opener_said_again_inside_it() {
        let prompt = "<|end_message|><|message_model|><|content_text|>Par<|message_model|>is";
        let tail = [" a city.<|end_message|>"];
        for (table, events) in [
            ("inkling", run_inkling(prompt, &tail)),
            ("inkling2", run(prompt, &tail)),
        ] {
            assert_eq!(content(&events), " a city.", "{table}: {events:?}");
            assert_eq!(malformed_count(&events), 0, "{table}: {events:?}");
        }
    }

    #[test]
    fn a_prompt_that_ends_at_an_opener_said_again_inside_a_thought_is_inside_it() {
        // The output grammar's position reads the turn's start, as the replay does.
        let prompt = "<|end_message|><|message_model|><|content_thinking|>Par<|message_model|>";
        for format in [inkling(), inkling2()] {
            assert_eq!(
                format.output_position(prompt, true),
                OutputPosition::InThought,
                "{}",
                format.name()
            );
        }
    }

    #[test]
    fn a_prompt_past_a_thought_with_an_opener_said_again_inside_it_is_past_the_thought() {
        // A prompt that ends at an opener is past its thought when the message before that opener
        // was one, and that message too is read from its own opener.
        let prompt = concat!(
            "<|end_message|><|message_model|><|content_thinking|>Pl<|message_model|>an",
            "<|end_message|><|message_model|>"
        );
        for format in [inkling(), inkling2()] {
            assert_eq!(
                format.output_position(prompt, true),
                OutputPosition::AfterThought,
                "{}",
                format.name()
            );
        }
    }

    #[test]
    fn a_quoted_spelling_in_a_json_answer_is_the_answers_text_under_the_option() {
        let answer = concat!(
            "<|message_model|><|content_text|>",
            "{\"note\":\"<|content_thinking|>not a thought<|end_message|>\"}<|end_message|>"
        );
        let mut engine = Engine::new(inkling2(), Declared::default()).reads_json_answer();
        let events = drive(&mut engine, UNOPENED, &[answer]);
        assert_eq!(
            content(&events),
            "{\"note\":\"<|content_thinking|>not a thought<|end_message|>\"}"
        );
        assert_eq!(reasoning(&events), "");
        assert!(names(&events).is_empty());
        assert_eq!(malformed_count(&events), 0);
    }

    fn answer_schema() -> Arc<Value> {
        Arc::new(json!({"type": "object", "properties": {"city": {"type": "string"}}}))
    }

    fn answer_elements(format: &Format, start: GrammarStart) -> Vec<Value> {
        let payload = format
            .answer_grammar_for(&answer_schema(), false, start)
            .expect("an answer tag")
            .payload();
        payload["format"]["elements"]
            .as_array()
            .cloned()
            .expect("a sequence")
    }

    #[test]
    fn the_answer_tag_writes_the_opener_before_the_thought_and_the_marker() {
        let lead = json!({"type": "const_string", "value": "<|message_model|>"});
        // From the first token the opener is owed outright, then Inkling's tag: the thought the
        // model may write first, the text marker, the schema and the message's close.
        let first = answer_elements(&inkling2(), GrammarStart::FirstToken);
        let inklings = answer_elements(&inkling(), GrammarStart::FirstToken);
        assert_eq!(first.len(), inklings.len() + 1);
        assert_eq!(first[0], lead);
        assert_eq!(first[1..], inklings[..]);
        // Where the engine does not say, starts after its reasoning parser, or may be of either
        // kind, the opener stands inside the optional way in, which that parser may have consumed
        // with the marker.
        for (i, start) in [
            GrammarStart::Unknown,
            GrammarStart::AfterReasoning,
            GrammarStart::Either,
        ]
        .into_iter()
        .enumerate()
        {
            let ours = answer_elements(&inkling2(), start);
            let theirs = answer_elements(&inkling(), start);
            assert_eq!(ours.len(), theirs.len(), "start {i}");
            assert_eq!(ours[0]["type"], "optional", "start {i}");
            let way = json!({
                "type": "sequence",
                "elements": [lead.clone(), theirs[0]["content"].clone()],
            });
            assert_eq!(ours[0]["content"], way, "start {i}");
            assert_eq!(ours[1..], theirs[1..], "start {i}");
        }
        // A thought the prompt left open reaches the answer only through the next message's
        // header, which reads calls: no answer tag, as for Inkling.
        assert!(inkling2().answer_grammar(&answer_schema(), true).is_none());
    }

    #[test]
    fn the_answer_tags_optional_way_in_reads_as_framing_where_the_engine_consumed_it() {
        // An engine that gates the answer's grammar on its reasoning parser has consumed the
        // opener and the kind marker when the tag starts, and the tag's optional way in (the
        // engine's start cannot say) still admits them: the model may write the opener again from
        // inside the answer, with or without a thought first, and, outside the tag's constraint,
        // again inside the thought. The opener said again inside a body is framing and the kind
        // marker after it the kind, so the answer is the JSON alone, whole and at every cut, with
        // and without the JSON-answer option.
        let consumed = "<|end_message|><|message_model|><|content_text|>";
        let answer = "{\"city\":\"Paris\"}";
        let plain = format!("<|message_model|><|content_text|>{answer}<|end_message|>");
        let thought = format!(
            "<|message_model|><|content_thinking|>Paris.<|end_message|>\
             <|message_model|><|content_text|>{answer}<|end_message|>"
        );
        let inside = format!(
            "<|message_model|><|content_thinking|>Par<|message_model|>is.<|end_message|>\
             <|message_model|><|content_text|>{answer}<|end_message|>"
        );
        for (output, thinks) in [
            (plain.as_str(), ""),
            (thought.as_str(), "Paris."),
            (inside.as_str(), "Paris."),
        ] {
            for option in [false, true] {
                let whole = std::iter::once(vec![output]);
                let cuts = (1..output.len()).map(|at| vec![&output[..at], &output[at..]]);
                for pieces in whole.chain(cuts) {
                    let mut engine = engine(inkling2());
                    if option {
                        engine = engine.reads_json_answer();
                    }
                    let events = drive(&mut engine, consumed, &pieces);
                    let case = format!("{} piece(s), option {option}", pieces.len());
                    assert_eq!(content(&events), answer, "{case}");
                    assert_eq!(reasoning(&events), thinks, "{case}");
                    assert_eq!(malformed_count(&events), 0, "{case}");
                    assert_eq!(bytes(&events), output, "{case}");
                    assert!(names(&events).is_empty(), "{case}");
                }
            }
        }
    }

    #[test]
    fn the_forced_calls_tag_is_inklings() {
        let tools = weather_tools();
        for (at_least_one, reasoning_open) in [(true, false), (false, false), (true, true)] {
            assert_eq!(
                inkling2()
                    .grammar(&tools, at_least_one, reasoning_open)
                    .map(|grammar| grammar.payload()),
                inkling()
                    .grammar(&tools, at_least_one, reasoning_open)
                    .map(|grammar| grammar.payload())
            );
        }
    }
}

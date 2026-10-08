//! Plain: a template that writes the message's content and nothing else. No thought, no calls, no
//! marker: the output is the content, byte for byte. As recorded for
//! `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and `microsoft/Phi-4-mini-instruct`, whose parse sets carry
//! neither a reasoning nor a call (bellwether refuses a parse case whose template drops either),
//! only content.
//!
//! The table has one state, content, and no terminal, so every byte of the output is content and
//! the prompt replay moves nothing: it is the table for a model the gateway otherwise reads
//! through no parser at all, with the same events, the same byte accounting and the same finish
//! as every other table. A model whose template has markers for a thought or for calls takes its
//! own table; this one is for the templates that have none.

use crate::format::{Emits, Format};

/// The plain table.
pub fn plain() -> Format {
    Format::new("plain").state("content", Emits::Content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(plain(), Declared::default());
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

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_output_is_the_content_byte_for_byte() {
        let output = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n#### 18";
        let events = run("<|assistant|>\n", &[output]);
        assert_eq!(events[0], Event::Content(Text::uncounted(output)));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn another_formats_markers_are_text_here() {
        // A template without a thought or calls has no marker to read: what another family would
        // take as a thought or a call block is this model's prose.
        let output = "<think>a plan</think>Sure.<tool_call>{\"name\": \"f\"}</tool_call>";
        let events = run("", &[output]);
        assert_eq!(content(&events), output);
        assert!(events
            .iter()
            .all(|event| matches!(event, Event::Content(_) | Event::Finish { .. })));
    }

    #[test]
    fn every_chunking_says_the_same() {
        let output = "You're welcome! Let me know if there's anything else.";
        let whole = content(&run("", &[output]));
        for cut in 1..output.len() {
            let events = run("", &[&output[..cut], &output[cut..]]);
            assert_eq!(content(&events), whole, "cut at {cut}");
        }
        let by_char: Vec<String> = output.chars().map(String::from).collect();
        let pieces: Vec<&str> = by_char.iter().map(String::as_str).collect();
        assert_eq!(content(&run("", &pieces)), whole);
    }

    #[test]
    fn an_empty_output_is_no_content_and_a_stop() {
        let events = run("<|assistant|>\n", &[]);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::Finish { tool_calls: 0, .. }));
    }

    #[test]
    fn an_earlier_turn_in_the_prompt_moves_nothing() {
        let prompt = concat!(
            "<|user|>\nIs 97 prime?</s>\n<|assistant|>\nYes.</s>\n",
            "<|user|>\nAnd 91?</s>\n<|assistant|>\n"
        );
        let events = run(prompt, &["No: 7 × 13."]);
        assert_eq!(events[0], Event::Content(Text::uncounted("No: 7 × 13.")));
    }
}

//! Command: the turn in parts, each between its own pair of markers, as Command A's chat template
//! renders it: the thought between `<|START_THINKING|>` and `<|END_THINKING|>`, then the calls as
//! one JSON list between `<|START_ACTION|>` and `<|END_ACTION|>`, or the answer between
//! `<|START_RESPONSE|>` and `<|END_RESPONSE|>`. Each object of the list is a call's `tool_name`
//! and its `parameters`, behind a `tool_call_id` the template counts from zero, and one call is a
//! list of one:
//!
//! ```text
//! <|START_ACTION|>[
//!     {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}}
//! ]<|END_ACTION|>
//! ```
//!
//! The format of the gateway's older `cohere` and `cohere_cmd` parsers. The turn starts in the
//! content state and the answer's markers move nothing, so an answer with no marker around it,
//! as Command R v01 writes its own, reads as the marked one does; `<|START_TEXT|>` and
//! `<|END_TEXT|>` move nothing either and are dropped from the answer, as the older parser
//! dropped them. Between the action markers one object alone is one call, as the older parser
//! read it beside the list, and text in place of the calls is malformed, as it is between Llama
//! Nemotron's markers. Recorded as `command-a-reasoning-08-2025` and `c4ai-command-r-v01`.
//!
//! Notes on this table, on where it reads otherwise than the older `cohere` parser, and on what
//! it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The Command table.
pub fn command() -> Format {
    Format::new("command")
        .terminal("think_open", "<|START_THINKING|>")
        .terminal("think_close", "<|END_THINKING|>")
        .terminal("calls_open", "<|START_ACTION|>")
        .terminal("calls_close", "<|END_ACTION|>")
        .terminal("answer_open", "<|START_RESPONSE|>")
        .terminal("answer_close", "<|END_RESPONSE|>")
        .terminal("text_open", "<|START_TEXT|>")
        .terminal("text_close", "<|END_TEXT|>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("calls", "calls_close", "content")
        .transition("content", "answer_open", "content")
        .transition("content", "answer_close", "content")
        .transition("content", "text_open", "content")
        .transition("content", "text_close", "content")
        .calls(CallSyntax::JsonAlone(None))
        .opens_turn("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        testing::{calls, content, drive, reasoning, the_same_at_every_cut},
        Declared,
    };

    #[test]
    fn each_part_reads_from_between_its_own_markers() {
        let run = |output: &str| {
            drive(
                &mut Engine::new(command(), Declared::default()),
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                &[output],
            )
        };
        // command-a-reasoning-08-2025/parse/calls-parallel-two, behind a thought
        let events = run(
            "<|START_THINKING|>Two lookups.<|END_THINKING|><|START_ACTION|>[\n    \
             {\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \"parameters\": \
             {\"city\": \"Paris\"}},\n    {\"tool_call_id\": \"1\", \"tool_name\": \
             \"get_stock_price\", \"parameters\": {\"ticker\": \"AAPL\"}}\n]<|END_ACTION|>",
        );
        assert_eq!(reasoning(&events), "Two lookups.");
        assert_eq!(
            calls(&events),
            [
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Paris\"}".to_string()
                ),
                (
                    "get_stock_price".to_string(),
                    "{\"ticker\": \"AAPL\"}".to_string()
                ),
            ]
        );
        assert_eq!(content(&events), "");
        // content-thinking-off, and c4ai-command-r-v01's, which has no marker
        for output in ["<|START_RESPONSE|>Hello!<|END_RESPONSE|>", "Hello!"] {
            assert_eq!(content(&run(output)), "Hello!", "{output}");
        }
    }

    #[test]
    fn without_its_thought_the_answer_reads_bare_and_the_call_still_reads() {
        // The answer's markers frame the answer, not the thought: where the thought's own are
        // shown they stay dropped, as the legacy tool parser strips them; with no tool declared
        // the call's bytes follow the bare answer as content.
        let call = "<|START_ACTION|>[{\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \
                    \"parameters\": {\"city\": \"Paris\"}}]<|END_ACTION|>";
        let output = format!(
            "<|START_THINKING|>Hm.<|END_THINKING|><|START_RESPONSE|>Checking.<|END_RESPONSE|>{call}"
        );
        let bare = "<|START_THINKING|>Hm.<|END_THINKING|>Checking.";
        let whole = format!("{bare}{call}");
        for (table, read, count) in [
            (command(), bare, 1),
            (command().without_calls(), whole.as_str(), 0),
        ] {
            let events = drive(
                &mut Engine::new(table.without_reasoning(), Declared::default()),
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                &[&output],
            );
            assert_eq!(content(&events), read);
            assert_eq!(reasoning(&events), "");
            assert_eq!(calls(&events).len(), count);
        }
    }

    #[test]
    fn a_lone_object_between_the_action_markers_is_a_call_and_the_text_markers_drop() {
        // A unit case of the gateway's older `cohere` parser, which read the same call and kept
        // the two line breaks between the parts.
        let said = the_same_at_every_cut(
            |pieces| {
                drive(
                    &mut Engine::new(command(), Declared::default()),
                    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                    pieces,
                )
            },
            "<|START_TEXT|>Some intro<|END_TEXT|>\n<|START_ACTION|>{\"tool_name\": \"test\", \
             \"parameters\": {}}<|END_ACTION|>\n<|START_TEXT|>Conclusion<|END_TEXT|>",
        );
        assert_eq!(said.content, "Some introConclusion");
        assert_eq!(said.calls, [("test".to_string(), "{}".to_string())]);
    }
}

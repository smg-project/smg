//! Llama Nemotron: the calls as one JSON list between `<TOOLCALL>` and `</TOOLCALL>`,
//! `<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</TOOLCALL>`, content
//! before the list; the thought between `<think>` and `</think>`, which the model writes itself.
//! The templates of Llama-3.3-Nemotron-Super-49B-v1.5, Llama-3.1-Nemotron-Nano-8B-v1 and the DLER
//! merge of the Nano write this. The Super template has the thought: `/no_think` in the system
//! prompt makes it close one in the prompt (`<think>\n\n</think>\n\n`), the request's
//! `enable_thinking` changes nothing, and an assistant message's content is rendered from its
//! `</think>` on. The Nano and DLER templates have no thought. Nemotron Ultra keeps Llama 3.1's
//! own template, a bare `{"name": …, "parameters": …}` object, and Nemotron 3 writes
//! Qwen3-Coder's tags: other syntaxes, other tables. Recorded as
//! `llama-3-3-nemotron-super-49b-v1-5`, `llama-3.1-nemotron-nano-8b-v1` and
//! `dler-llama-nemotron-8b-merge-research`.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The Llama Nemotron table.
pub fn llama_nemotron() -> Format {
    Format::new("llama_nemotron")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<TOOLCALL>")
        .terminal("call_close", "</TOOLCALL>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .calls(CallSyntax::JsonList)
        .opens_turn("<|start_header_id|>assistant<|end_header_id|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// The generation prompt's tail: the assistant header, and nothing more.
    const PROMPT: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(llama_nemotron(), Declared::default());
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
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
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

    fn calls(events: &[Event]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
                Event::ToolCallArguments { json, .. } => {
                    calls.last_mut().expect("a call started").1.push_str(json);
                }
                _ => {}
            }
        }
        calls
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

    fn dropped(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The engine's finish: its reason as the engine gave it (the adapters refine it to
    /// `tool_calls`), and the calls counted.
    fn finish(events: &[Event]) -> (FinishReason, u32) {
        events
            .iter()
            .find_map(|event| match event {
                Event::Finish {
                    reason, tool_calls, ..
                } => Some((reason.clone(), *tool_calls)),
                _ => None,
            })
            .expect("a finish")
    }

    #[test]
    fn the_list_between_the_markers_gives_each_object_as_a_call() {
        // llama-3-3-nemotron-super-49b-v1-5/parse/call-single-thinking-off, calls-parallel-three
        let events = run(
            PROMPT,
            "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]\
             </TOOLCALL>",
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(content(&events), "");
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
        assert_eq!(
            dropped(&events),
            ["<TOOLCALL>", "[", "]", "</TOOLCALL>"],
            "the markers and the brackets are the template's"
        );
        let events = run(
            PROMPT,
            "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
             {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}, \
             {\"name\": \"get_stock_price\", \"arguments\": {\"ticker\": \"MSFT\"}}]</TOOLCALL>",
        );
        assert_eq!(
            calls(&events),
            [
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Paris\"}".to_string()
                ),
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Rome\"}".to_string()
                ),
                (
                    "get_stock_price".to_string(),
                    "{\"ticker\": \"MSFT\"}".to_string()
                ),
            ]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 3));
        assert!(dropped(&events).contains(&", "));
    }

    #[test]
    fn content_before_the_list_and_the_thought_before_both() {
        // call-with-content; the Super template's thought, which the model writes itself
        let events = run(
            PROMPT,
            "Let me check.<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": \
             {\"city\": \"Paris\", \"unit\": \"celsius\"}}]</TOOLCALL>",
        );
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\", \"unit\": \"celsius\"}".to_string()
            )]
        );
        let events = run(
            PROMPT,
            "<think>The user asks about Paris.</think><TOOLCALL>[{\"name\": \"get_weather\", \
             \"arguments\": {\"city\": \"Paris\"}}]</TOOLCALL>",
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "");
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn empty_arguments_and_nested_ones_arrive_as_the_model_wrote_them() {
        // call-empty-arguments, call-nested-arguments
        let events = run(
            PROMPT,
            "<TOOLCALL>[{\"name\": \"get_stock_price\", \"arguments\": {}}]</TOOLCALL>",
        );
        assert_eq!(
            calls(&events),
            [("get_stock_price".to_string(), "{}".to_string())]
        );
        let nested = "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" \
                      and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": \
                      {\"nested\": {\"deep\": [1, 2.5, true, null]}}}";
        let events = run(
            PROMPT,
            &format!("<TOOLCALL>[{{\"name\": \"save_note\", \"arguments\": {nested}}}]</TOOLCALL>"),
        );
        assert_eq!(
            calls(&events),
            [("save_note".to_string(), nested.to_string())]
        );
    }

    #[test]
    fn content_without_calls_keeps_the_fence_and_a_thought_ends_at_its_first_close() {
        // content-thinking-off, content-with-marker-in-code-fence: `<tool_call>` is not this
        // table's, so the fence stays content; `</think>` is, so the reasoning probe differs here
        // as under every table that reads it (the fixture test lists it).
        let events = run(PROMPT, "Hello!");
        assert_eq!(content(&events), "Hello!");
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, fence);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
        let events = run(
            PROMPT,
            "<think>I could write </think> here but it is just text... no.</think>Paris is sunny.",
        );
        assert_eq!(reasoning(&events), "I could write ");
        assert_eq!(
            content(&events),
            " here but it is just text... no.</think>Paris is sunny."
        );
    }

    #[test]
    fn text_in_place_of_the_list_is_malformed_and_the_block_still_closes() {
        // Between the markers prose has no place: the template writes the list and nothing else.
        let events = run(PROMPT, "<TOOLCALL>not a list</TOOLCALL>Done.");
        assert_eq!(malformed(&events), ["not a list"]);
        assert_eq!(content(&events), "Done.");
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let events = run(
            PROMPT,
            "<TOOLCALL>[{\"name\": \"get_stock_price\", \"arguments\": {}}] tail</TOOLCALL>",
        );
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(malformed(&events), ["tail"]);
        assert_eq!(
            dropped(&events),
            ["<TOOLCALL>", "[", "]", " ", "</TOOLCALL>"]
        );
    }

    #[test]
    fn the_prompt_after_a_tool_result_or_with_the_thought_closed_leaves_the_output_in_content() {
        // After a tool result the Super template writes a tool turn and the assistant header: the
        // replay starts at that header and finds nothing. The Nano template writes the result as a
        // user turn, the same for the replay.
        let prompt = concat!(
            "<|start_header_id|>user<|end_header_id|>\n\nWeather in Paris?<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
            "</TOOLCALL><|eot_id|><|start_header_id|>tool<|end_header_id|>\n\n",
            "<TOOL_RESPONSE>[{\"temperature\": 18}]</TOOL_RESPONSE><|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(llama_nemotron().replay(prompt), PROMPT);
        let events = run(prompt, "It is 18 degrees.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // `/no_think` in the system prompt: the Super template closes the thought in the prompt,
        // and the output starts in content.
        let prompt = format!("{PROMPT}<think>\n\n</think>\n\n");
        let events = run(&prompt, "Paris is sunny.");
        assert_eq!(content(&events), "Paris is sunny.");
        assert_eq!(reasoning(&events), "");
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ReasoningStart)));
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = llama_nemotron();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "llama_nemotron");
        assert_eq!(
            format.turn_opener(),
            Some("<|start_header_id|>assistant<|end_header_id|>")
        );
    }
}

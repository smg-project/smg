//! Mistral: the thought between `[THINK]` and `[/THINK]`, which the model writes itself (the
//! generation prompt ends at `[/INST]` whatever the request says), and each call as `[TOOL_CALLS]`,
//! the tool's name, `[ARGS]` and the arguments object alone, one call after another with nothing
//! between or after them: `[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}`. Content comes before
//! the calls. The templates of Devstral 2, Ministral 3, Mistral Medium 3.5 and Mistral Small 4
//! write this (the v13 tokenizer's spelling; the older `[TOOL_CALLS][{…}]` list of Mistral 7B
//! v0.3 is another syntax, the gateway's older `mistral` parser's). Recorded as
//! `ministral-3-14b-reasoning-2512`, `mistral-medium-3.5-128b`, `mistral-small-4-119b-2603` and
//! the instruct groups `devstral-2-123b-instruct-2512`, `ministral-3-3b-instruct-2512`,
//! `ministral-3-3b-instruct-2512-bf16`, `ministral-3-8b-instruct-2512-bf16` and
//! `ministral-3-14b-instruct-2512-bf16`, whose templates have no thought.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The marker between a call's name and its arguments.
pub const ARGS: &str = "[ARGS]";

/// The Mistral table.
pub fn mistral() -> Format {
    Format::new("mistral")
        .terminal("think_open", "[THINK]")
        .terminal("think_close", "[/THINK]")
        .terminal("call_open", "[TOOL_CALLS]")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_open", "call")
        .calls(CallSyntax::NamedArguments(ARGS))
        .opens_turn("[/INST]")
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

    /// The generation prompt's tail: the user turn's close, and nothing more.
    const PROMPT: &str = "[/INST]";

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(mistral(), Declared::default());
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
    fn the_thought_then_a_call_as_the_reasoning_template_writes_them() {
        // ministral-3-14b-reasoning-2512/parse/call-single-thinking-on
        let events = run(
            PROMPT,
            "[THINK]The user asks about Paris.[/THINK][TOOL_CALLS]get_weather[ARGS]\
             {\"city\": \"Paris\"}",
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        assert_eq!(content(&events), "");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "[TOOL_CALLS]"
        )));
    }

    #[test]
    fn calls_follow_one_another_from_their_own_openers_with_content_before_them() {
        // ministral-3-14b-reasoning-2512/parse/calls-parallel-three, call-with-content
        let events = run(
            PROMPT,
            "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}[TOOL_CALLS]get_weather[ARGS]\
             {\"city\": \"Rome\"}[TOOL_CALLS]get_stock_price[ARGS]{\"ticker\": \"MSFT\"}",
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
        let events = run(
            PROMPT,
            "Let me check.[TOOL_CALLS]get_weather[ARGS]\
             {\"city\": \"Paris\", \"unit\": \"celsius\"}",
        );
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\", \"unit\": \"celsius\"}".to_string()
            )]
        );
    }

    #[test]
    fn empty_arguments_and_nested_ones_arrive_as_the_model_wrote_them() {
        // call-empty-arguments, call-nested-arguments
        let events = run(PROMPT, "[TOOL_CALLS]get_stock_price[ARGS]{}");
        assert_eq!(
            calls(&events),
            [("get_stock_price".to_string(), "{}".to_string())]
        );
        let nested = "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" \
                      and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": \
                      {\"nested\": {\"deep\": [1, 2.5, true, null]}}}";
        let events = run(PROMPT, &format!("[TOOL_CALLS]save_note[ARGS]{nested}"));
        assert_eq!(
            calls(&events),
            [("save_note".to_string(), nested.to_string())]
        );
    }

    #[test]
    fn content_and_a_thought_without_calls_keep_the_other_families_markers_as_text() {
        // content-thinking-on, reasoning-then-content, and the two probes: neither `<tool_call>`
        // nor `</think>` is this table's, so the fence stays content and the marker reasoning text.
        let events = run(PROMPT, "Hello!");
        assert_eq!(content(&events), "Hello!");
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let events = run(
            PROMPT,
            "[THINK]97 is not divisible by 2, 3, 5 or 7.[/THINK]Yes, 97 is prime.",
        );
        assert_eq!(reasoning(&events), "97 is not divisible by 2, 3, 5 or 7.");
        assert_eq!(content(&events), "Yes, 97 is prime.");
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, fence);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
        let events = run(
            PROMPT,
            "[THINK]I could write </think> here but it is just text... no, I will answer.[/THINK]\
             Paris is sunny.",
        );
        assert_eq!(
            reasoning(&events),
            "I could write </think> here but it is just text... no, I will answer."
        );
        assert_eq!(content(&events), "Paris is sunny.");
    }

    #[test]
    fn the_prompt_after_a_tool_result_leaves_the_output_in_content() {
        // After a tool result the prompt ends in `[/TOOL_RESULTS]`, and its last `[/INST]` stands
        // before the previous turn's calls: the replay walks into the call state, which the prompt
        // never enters, so the output starts in content, where the template puts it.
        let prompt = "[INST]Weather in Paris?[/INST][TOOL_CALLS]get_weather[ARGS]{\"city\": \
                      \"Paris\"}</s>[TOOL_RESULTS]{\"temperature\": 18}[/TOOL_RESULTS]";
        assert_eq!(
            mistral().replay(prompt),
            &prompt[prompt.find("[/INST]").expect("the user turn's close")..]
        );
        let events = run(prompt, "It is 18 degrees.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = mistral();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "mistral");
        assert_eq!(format.turn_opener(), Some("[/INST]"));
    }
}

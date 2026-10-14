//! FunctionGemma: each call between `<start_function_call>` and `<end_function_call>` in Gemma's
//! spelling with the template's own tokens, `call:NAME{key:<escape>text<escape>,…}`, a null as
//! `None` ([`Spelling::FUNCTION_GEMMA`]); the calls one after another with nothing between them,
//! content before them, no thought: the generation prompt is `<start_of_turn>model\n` whatever
//! the request says, and the template renders no `reasoning_content`. Recorded as
//! `functiongemma-270m-it` (google/functiongemma-270m-it):
//!
//! ```text
//! <start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>
//! ```
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    gemma::Spelling,
};

/// The FunctionGemma table.
pub fn function_gemma() -> Format {
    Format::new("function_gemma")
        .terminal("call_open", "<start_function_call>")
        .terminal("call_close", "<end_function_call>")
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .calls(CallSyntax::Gemma(Spelling::FUNCTION_GEMMA))
        .opens_turn("<start_of_turn>model")
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

    /// The generation prompt's tail: the model's turn opened, and nothing more.
    const PROMPT: &str = "<start_of_turn>model\n";

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(function_gemma(), Declared::default());
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
    fn a_call_as_the_template_writes_it_and_two_one_after_another() {
        // functiongemma-270m-it/parse/call-single-thinking-off, calls-parallel-two
        let events = run(
            PROMPT,
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>",
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
            ["<start_function_call>"],
            "the opener is the template's; the closer is the call's end"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { source, .. } if source.text == "<end_function_call>"
        )));
        let events = run(
            PROMPT,
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>\
             <start_function_call>call:get_stock_price{ticker:<escape>AAPL<escape>}\
             <end_function_call>",
        );
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
        assert_eq!(finish(&events), (FinishReason::Stop, 2));
    }

    #[test]
    fn content_before_the_call_and_empty_arguments() {
        // call-with-content, call-empty-arguments
        let events = run(
            PROMPT,
            "Let me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>,\
             unit:<escape>celsius<escape>}<end_function_call>",
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
            "<start_function_call>call:get_stock_price{}<end_function_call>",
        );
        assert_eq!(
            calls(&events),
            [("get_stock_price".to_string(), "{}".to_string())]
        );
    }

    #[test]
    fn nested_arguments_in_the_templates_spelling_come_back_as_json() {
        // call-nested-arguments: the keys sorted by the template, `None` for the null, the
        // string's newline, quotes and backslash as they are between the tokens.
        let events = run(
            PROMPT,
            "<start_function_call>call:save_note{body:<escape>Line one\nLine two with \"quotes\" \
             and a backslash \\<escape>,meta:{nested:{deep:[1,2.5,true,None]}},tags:[<escape>a\
             <escape>,<escape>b<escape>],title:<escape>Plan<escape>}<end_function_call>",
        );
        assert_eq!(
            calls(&events),
            [(
                "save_note".to_string(),
                "{\"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \
                 \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}, \"tags\": [\"a\", \
                 \"b\"], \"title\": \"Plan\"}"
                    .to_string()
            )]
        );
    }

    #[test]
    fn content_without_calls_keeps_the_fence_as_text() {
        // content-thinking-off, content-with-marker-in-code-fence: `<tool_call>` is not this
        // table's marker, so the fence stays content.
        let events = run(PROMPT, "Hello!");
        assert_eq!(content(&events), "Hello!");
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, fence);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn the_prompt_replays_from_the_last_model_turn() {
        // A previous turn's call stands before the last `<start_of_turn>model`, so the replay
        // never sees it and the output starts in content.
        let prompt = "<start_of_turn>user\nWeather in Paris?<end_of_turn>\n<start_of_turn>model\n\
                      <start_function_call>call:get_weather{city:<escape>Paris<escape>}\
                      <end_function_call><end_of_turn>\n<start_of_turn>user\nAnd Rome?\
                      <end_of_turn>\n<start_of_turn>model\n";
        assert_eq!(function_gemma().replay(prompt), PROMPT);
        let events = run(prompt, "Rome is sunny.");
        assert_eq!(content(&events), "Rome is sunny.");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = function_gemma();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "function_gemma");
        assert_eq!(format.turn_opener(), Some("<start_of_turn>model"));
    }
}

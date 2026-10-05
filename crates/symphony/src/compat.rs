//! A bridge from the two existing parser crates to [`Parser`].
//!
//! SMG parses model output with two separate crates today: a reasoning parser that splits
//! reasoning from content, and a tool parser that finds tool calls in the content. [`Combined`]
//! drives such a pair as one Symphony parser, in the order the gateway drives them, so every
//! format SMG serves today can be replayed through Symphony before any format is ported, and so
//! the gateway can switch to [`Parser`] without waiting for the ports.
//!
//! The pair gives this bridge less than Symphony promises, and the tracker records both limits as
//! compromises to remove with the old crates:
//!
//! - neither old trait sees token ids, so every `Text::tokens` is `None` and
//!   `Event::Finish::reasoning_tokens` is zero for want of a count;
//! - neither reports the bytes a tool call was written as, so the `source` of tool-call events is
//!   empty and the conservation property cannot be checked through this bridge.
//!
//! One deliberate difference from the gateway: the gateway skips the tool parser for text produced
//! in a chunk that ends inside reasoning. This bridge sends all content to the tool parser, because
//! the reasoning parser has already said that text is content.

use futures::executor::block_on;
use openai_protocol::common::Tool;
use reasoning_parser::ReasoningParser;
use tool_parser::{types::ToolCallItem, ToolParser};

use crate::{
    event::{Event, Events, FinishReason, Text},
    input::{EngineFinish, Input},
    parser::{ParseError, Parser},
};

/// The old reasoning parser and tool parser of one model, driven as one [`Parser`].
pub struct Combined {
    reasoning: Option<Box<dyn ReasoningParser>>,
    tool: Option<Box<dyn ToolParser>>,
    tools: Vec<Tool>,
    starts_in_reasoning: bool,
    reasoning_open: bool,
    calls_started: Vec<u32>,
    stage: Stage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Fresh,
    Streaming,
    Ended,
}

/// A parser name the old factories do not know.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnknownParser {
    /// No reasoning parser is registered under this name.
    #[error("unknown reasoning parser {0:?}")]
    Reasoning(String),
    /// No tool parser is registered under this name.
    #[error("unknown tool parser {0:?}")]
    Tool(String),
}

impl Combined {
    /// Drive `reasoning` and `tool` as one parser. Either may be absent: without a reasoning parser
    /// all text is content; without a tool parser no tool calls are found.
    pub fn new(
        reasoning: Option<Box<dyn ReasoningParser>>,
        tool: Option<Box<dyn ToolParser>>,
        tools: Vec<Tool>,
    ) -> Self {
        Self {
            reasoning,
            tool,
            tools,
            starts_in_reasoning: false,
            reasoning_open: false,
            calls_started: Vec::new(),
            stage: Stage::Fresh,
        }
    }

    /// Build the pair from the names the old factories register, as the gateway's flags name them.
    pub fn by_name(
        reasoning: Option<&str>,
        tool: Option<&str>,
        tools: Vec<Tool>,
    ) -> Result<Self, UnknownParser> {
        let reasoning = match reasoning {
            Some(name) => Some(
                reasoning_parser::ParserFactory::new()
                    .registry()
                    .create_parser(name)
                    .ok_or_else(|| UnknownParser::Reasoning(name.to_string()))?,
            ),
            None => None,
        };
        let tool = match tool {
            Some(name) => Some(
                tool_parser::ParserFactory::new()
                    .registry()
                    .create_parser(name)
                    .ok_or_else(|| UnknownParser::Tool(name.to_string()))?,
            ),
            None => None,
        };
        Ok(Self::new(reasoning, tool, tools))
    }

    /// Tell the reasoning parser that the prompt already opened the reasoning block, the way the
    /// gateway does when the rendered prompt ends inside it. Takes effect when the prompt is fed.
    pub fn starting_in_reasoning(mut self) -> Self {
        self.starts_in_reasoning = true;
        self
    }

    /// Split a chunk into reasoning events and the content that remains.
    fn split_reasoning(&mut self, text: &str, out: &mut Events) -> Result<String, ParseError> {
        let Some(reasoning) = self.reasoning.as_mut() else {
            return Ok(text.to_string());
        };
        let split = reasoning
            .parse_reasoning_streaming_incremental(text)
            .map_err(|e| ParseError::Internal(e.to_string()))?;
        let still_open = reasoning.is_in_reasoning();
        self.emit_reasoning(&split.reasoning_text, still_open, out);
        Ok(split.normal_text)
    }

    fn emit_reasoning(&mut self, text: &str, still_open: bool, out: &mut Events) {
        if !text.is_empty() {
            if !self.reasoning_open {
                out.push(Event::ReasoningStart);
                self.reasoning_open = true;
            }
            out.push_reasoning(Text::uncounted(text));
        }
        if self.reasoning_open && !still_open {
            out.push(Event::ReasoningEnd);
            self.reasoning_open = false;
        }
    }

    /// Send content through the tool parser, emitting the content and calls it finds.
    fn parse_tools(&mut self, content: &str, out: &mut Events) -> Result<(), ParseError> {
        if content.is_empty() {
            return Ok(());
        }
        let Some(tool) = self.tool.as_mut() else {
            out.push_content(Text::uncounted(content));
            return Ok(());
        };
        let found = block_on(tool.parse_incremental(content, &self.tools))
            .map_err(|e| ParseError::Internal(e.to_string()))?;
        out.push_content(Text::uncounted(found.normal_text));
        self.emit_calls(found.calls, out);
        Ok(())
    }

    fn emit_calls(&mut self, items: Vec<ToolCallItem>, out: &mut Events) {
        for item in items {
            let index = item.tool_index as u32;
            if let Some(name) = item.name {
                if !self.calls_started.contains(&index) {
                    self.calls_started.push(index);
                    out.push(Event::ToolCallStart {
                        index,
                        id: format!("call_{index}"),
                        name,
                        source: Text::default(),
                    });
                }
            }
            if !item.parameters.is_empty() {
                out.push(Event::ToolCallArguments {
                    index,
                    json: item.parameters,
                    source: Text::default(),
                });
            }
        }
    }

    fn finish(&mut self, finish: EngineFinish, out: &mut Events) -> Result<(), ParseError> {
        if let Some(reasoning) = self.reasoning.as_mut() {
            let held = reasoning
                .flush()
                .map_err(|e| ParseError::Internal(e.to_string()))?;
            let still_open = reasoning.is_in_reasoning();
            self.emit_reasoning(&held.reasoning_text, still_open, out);
            self.parse_tools(&held.normal_text, out)?;
        }
        if let Some(tool) = self.tool.as_mut() {
            out.push_content(Text::uncounted(tool.take_unstreamed_normal_text()));
            if let Some(items) = tool.get_unstreamed_tool_args() {
                self.emit_calls(items, out);
            }
        }
        if self.reasoning_open {
            out.push(Event::ReasoningEnd);
            self.reasoning_open = false;
        }
        for index in &self.calls_started {
            out.push(Event::ToolCallEnd {
                index: *index,
                source: Text::default(),
            });
        }
        let tool_calls = self.calls_started.len() as u32;
        let reason = match finish {
            EngineFinish::Stop if tool_calls > 0 => FinishReason::ToolCalls,
            EngineFinish::Stop => FinishReason::Stop,
            EngineFinish::Length => FinishReason::Length,
            EngineFinish::Abort => FinishReason::Abort,
            EngineFinish::Other(other) => FinishReason::Other(other),
        };
        out.push(Event::Finish {
            reason,
            tool_calls,
            reasoning_tokens: 0,
        });
        Ok(())
    }
}

impl Parser for Combined {
    fn feed(&mut self, input: Input<'_>, out: &mut Events) -> Result<(), ParseError> {
        match input {
            Input::Prompt { .. } => {
                if self.stage != Stage::Fresh {
                    return Err(ParseError::Lifecycle(
                        "prompt after output began".to_string(),
                    ));
                }
                if self.starts_in_reasoning {
                    if let Some(reasoning) = self.reasoning.as_mut() {
                        reasoning.mark_reasoning_started();
                    }
                }
                self.stage = Stage::Streaming;
                Ok(())
            }
            Input::Delta { text, .. } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("delta after end".to_string()));
                }
                self.stage = Stage::Streaming;
                let content = self.split_reasoning(text, out)?;
                self.parse_tools(&content, out)
            }
            Input::End { finish } => {
                if self.stage == Stage::Ended {
                    return Err(ParseError::Lifecycle("end after end".to_string()));
                }
                self.stage = Stage::Ended;
                self.finish(finish, out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::TokenSpan;

    fn weather_tool() -> Tool {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }))
        .expect("a valid tool definition")
    }

    fn delta<'a>(text: &'a str) -> Input<'a> {
        Input::Delta {
            token_ids: &[],
            text,
            spans: &[],
        }
    }

    fn run(parser: &mut Combined, chunks: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut out = Events::new();
        for chunk in chunks {
            parser.feed(delta(chunk), &mut out).expect("feed succeeds");
        }
        parser
            .feed(Input::End { finish }, &mut out)
            .expect("end succeeds");
        out.drain()
    }

    fn arguments_of(events: &[Event], call: u32) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn content_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Content(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn reasoning_then_a_json_tool_call_becomes_the_expected_events() {
        let mut parser = Combined::by_name(Some("deepseek_r1"), Some("json"), vec![weather_tool()])
            .expect("known parsers");
        let events = run(
            &mut parser,
            &[r#"<think>plan</think>{"name": "get_weather", "arguments": {"city": "Paris"}}"#],
            EngineFinish::Stop,
        );
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(events[1], Event::Reasoning(Text::uncounted("plan")));
        assert_eq!(events[2], Event::ReasoningEnd);
        assert_eq!(
            events[3],
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
                source: Text::default(),
            }
        );
        assert_eq!(arguments_of(&events, 0), r#"{"city":"Paris"}"#);
        assert_eq!(
            events[events.len() - 2],
            Event::ToolCallEnd {
                index: 0,
                source: Text::default()
            }
        );
        assert_eq!(
            events[events.len() - 1],
            Event::Finish {
                reason: FinishReason::ToolCalls,
                tool_calls: 1,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(content_of(&events), "");
    }

    #[test]
    fn content_without_a_tool_parser_stays_content() {
        let mut parser =
            Combined::by_name(Some("deepseek_r1"), None, vec![]).expect("known parser");
        let events = run(
            &mut parser,
            &["<think>a</think>hello"],
            EngineFinish::Length,
        );
        assert_eq!(
            events,
            vec![
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("a")),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("hello")),
                Event::Finish {
                    reason: FinishReason::Length,
                    tool_calls: 0,
                    reasoning_tokens: 0,
                },
            ]
        );
    }

    #[test]
    fn reasoning_left_open_is_closed_at_the_end() {
        let mut parser =
            Combined::by_name(Some("deepseek_r1"), None, vec![]).expect("known parser");
        let events = run(&mut parser, &["<think>still thinking"], EngineFinish::Stop);
        assert_eq!(
            events,
            vec![
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("still thinking")),
                Event::ReasoningEnd,
                Event::Finish {
                    reason: FinishReason::Stop,
                    tool_calls: 0,
                    reasoning_tokens: 0,
                },
            ]
        );
    }

    // The old `json` parser is not chunking-invariant when prose precedes the JSON: in one chunk
    // `Sure.{...}` is all content, character by character `Sure.` is content and the JSON becomes
    // a call. That is a finding for bellwether's fixtures, not for this bridge, which reproduces
    // the old parsers as they are. This test uses a case the old parsers handle consistently.
    #[test]
    fn one_character_chunks_give_the_same_text_and_arguments_as_one_chunk() {
        let output =
            r#"<think>plan</think>{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let mut whole = Combined::by_name(Some("deepseek_r1"), Some("json"), vec![weather_tool()])
            .expect("known parsers");
        let whole_events = run(&mut whole, &[output], EngineFinish::Stop);
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let mut chunked =
            Combined::by_name(Some("deepseek_r1"), Some("json"), vec![weather_tool()])
                .expect("known parsers");
        let chunked_events = run(&mut chunked, &refs, EngineFinish::Stop);
        assert_eq!(content_of(&chunked_events), content_of(&whole_events));
        assert_eq!(
            arguments_of(&chunked_events, 0),
            arguments_of(&whole_events, 0)
        );
        assert_eq!(chunked_events.last(), whole_events.last());
    }

    #[test]
    fn the_engine_finish_reason_is_kept_when_no_call_was_made() {
        let mut parser =
            Combined::by_name(None, Some("json"), vec![weather_tool()]).expect("known parser");
        let events = run(
            &mut parser,
            &["plain answer"],
            EngineFinish::Other("content_filter".into()),
        );
        assert_eq!(
            events,
            vec![
                Event::Content(Text::uncounted("plain answer")),
                Event::Finish {
                    reason: FinishReason::Other("content_filter".into()),
                    tool_calls: 0,
                    reasoning_tokens: 0,
                },
            ]
        );
    }

    #[test]
    fn inputs_out_of_order_are_lifecycle_errors() {
        let mut parser = Combined::new(None, None, vec![]);
        let mut out = Events::new();
        parser.feed(delta("a"), &mut out).expect("delta");
        assert!(matches!(
            parser.feed(
                Input::Prompt {
                    token_ids: &[],
                    text: ""
                },
                &mut out
            ),
            Err(ParseError::Lifecycle(_))
        ));
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        assert!(matches!(
            parser.feed(delta("b"), &mut out),
            Err(ParseError::Lifecycle(_))
        ));
        assert!(matches!(
            parser.feed(
                Input::End {
                    finish: EngineFinish::Stop
                },
                &mut out
            ),
            Err(ParseError::Lifecycle(_))
        ));
        let spans: [TokenSpan; 0] = [];
        assert!(TokenSpan::partitions("", &spans));
    }

    #[test]
    fn unknown_parser_names_are_reported() {
        assert_eq!(
            Combined::by_name(Some("nope"), None, vec![]).err(),
            Some(UnknownParser::Reasoning("nope".into()))
        );
        assert_eq!(
            Combined::by_name(None, Some("nope"), vec![]).err(),
            Some(UnknownParser::Tool("nope".into()))
        );
    }
}

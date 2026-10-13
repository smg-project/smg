//! Symphony reading a chat response: the opt-in path that parses a model's output with one
//! [`Engine`] per choice in place of the tool and reasoning parser crates.
//!
//! The switch is the gateway's `--parser` flag (`RouterConfig::parser`, `legacy` by default),
//! which a request's `X-SMG-Parser` header overrides for that one request ([`choice`]). With
//! `symphony` on, the parser names set for the model decide ([`reads`]): the `tool_parser` and
//! `reasoning_parser` of the model's card (worker labels), else the `--tool-call-parser` and
//! `--reasoning-parser` flags, resolved as the legacy path resolves them (`ParserResolver`).
//! Names that are one Symphony table's (`symphony::formats::Table::name`; both the same, or one
//! alone) read every request for the model with that table, whatever the model is; a legacy
//! parser's name keeps the legacy parsers, which is the way back for one model when its table
//! misbehaves; no name at all takes the registry's table for the model
//! (`symphony::formats::for_model`), and the legacy parsers with their own detection when it has
//! none. The names are checked at startup and at a worker's registration ([`check_names`]).
//! Every request read with `legacy` keeps the older parsers. The response names the parser that
//! read it in `x-smg-parser`, legacy when the choice was Symphony but no table took the request,
//! so a client comparing the two can tell which one read each answer.
//!
//! Request building assigns Symphony to a request ([`assign`]) and stores the [`Assignment`] in
//! the response spec: the table, and the part of the rendered prompt the engine replays to find
//! the state the generation starts in. Preparation asks the table for a forced or required call's
//! structural tag ([`constraint`]) before the older registry. Both decide with one predicate,
//! [`reads`], so the output a table's tag constrains is read by that table and never by the
//! older parsers, which expect another syntax. The streaming and non-streaming
//! drivers run one [`Reader`] per choice over the stop decoder's output and render its events
//! with `symphony::adapt::chat`, so the client sees the deltas and the message it saw before.

use axum::http::HeaderMap;
use llm_tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice, ToolChoiceValue},
};
use symphony::{
    formats::{for_model, table_for, Table},
    Declared, Engine, EngineFinish, Event, Events, Format, Input, Parser as _, TokenSpan,
};
use tool_parser::ToolConstraint;
use tracing::debug;

use crate::{config::ParserChoice, routers::grpc::utils::Names};

/// The request header that names the parser for one request, `legacy` or `symphony`, over the
/// gateway's `--parser` flag.
pub(crate) const HEADER: &str = "x-smg-parser";

/// The parser that reads a request's response: the one its `X-SMG-Parser` header names, else
/// `default`, the gateway's `--parser` flag. The error names a header value that is neither
/// `legacy` nor `symphony`, for the 400 response.
pub(crate) fn choice(
    default: ParserChoice,
    headers: Option<&HeaderMap>,
) -> Result<ParserChoice, String> {
    let Some(value) = headers.and_then(|headers| headers.get(HEADER)) else {
        return Ok(default);
    };
    let spelled = String::from_utf8_lossy(value.as_bytes());
    ParserChoice::parse(&spelled)
        .ok_or_else(|| format!("{HEADER} must be legacy or symphony, not {spelled:?}"))
}

/// The table the parser names set for a model select, when they do: every name set is one
/// table's ([`Table::named`]), the same one. A legacy parser's name, or two different tables,
/// select none; so do no names at all.
pub(crate) fn named_table(names: &Names) -> Option<Table> {
    let tables: Vec<Option<Table>> = [&names.tool, &names.reasoning]
        .into_iter()
        .flatten()
        .map(|name| Table::named(name))
        .collect();
    let first = *tables.first()?;
    if tables.iter().any(|table| *table != first) {
        return None;
    }
    first
}

/// The table that reads `request`'s response, if one does: `parser` is Symphony, the request
/// asks nothing the older path answers differently (the response separates reasoning, and
/// `tool_choice` is not `none`), and a table is there to read it: the one the model's parser
/// `names` select ([`named_table`]), whatever the model, or the registry's for the model when no
/// name is set. A legacy parser's name leaves the response to the legacy parsers. Preparation
/// and request building both decide with this one predicate.
pub(crate) fn reads(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
) -> Option<Format> {
    if parser != ParserChoice::Symphony
        || !request.separate_reasoning
        || matches!(
            request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
    {
        return None;
    }
    if names.tool.is_none() && names.reasoning.is_none() {
        return for_model(&request.model);
    }
    let table = named_table(names)?;
    // The registry's table comes with the model's own turn opener; a table named for a model
    // the registry reads with another, or does not know, replays from the table's own.
    match for_model(&request.model) {
        Some(format) if table_for(&request.model) == Some(table) => Some(format),
        _ => Some(table.format()),
    }
}

/// Whether the parser names set for a model (the flags, or one worker's labels) are names the
/// gateway reads with under `parser`: a legacy parser's, or, with Symphony on, a table's.
/// `legacy_tool` and `legacy_reasoning` say whether the legacy registries have each name. A
/// table no legacy parser shares binds the other name to the same table or to nothing: the
/// legacy parsers read what a table declines (a response without separate reasoning, a
/// `tool_choice` of `none`) by the names they are given, and have no parser of that name. The
/// error says what is wrong with which name, for the startup or registration failure.
pub(crate) fn check_names(
    parser: ParserChoice,
    names: &Names,
    legacy_tool: bool,
    legacy_reasoning: bool,
) -> Result<(), String> {
    let symphony = parser == ParserChoice::Symphony;
    let kinds = [
        ("tool-call", &names.tool, legacy_tool, &names.reasoning),
        ("reasoning", &names.reasoning, legacy_reasoning, &names.tool),
    ];
    for (kind, name, legacy, other) in kinds {
        let Some(name) = name else {
            continue;
        };
        if legacy {
            continue;
        }
        match (symphony, Table::named(name)) {
            (true, Some(table)) => {
                if let Some(other) = other {
                    if Table::named(other) != Some(table) {
                        return Err(format!(
                            "{kind} parser '{name}' is a Symphony table's name and no legacy \
                             parser's, so the other parser must name the same table or \
                             nothing, not '{other}'"
                        ));
                    }
                }
            }
            (true, None) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': neither a legacy parser nor a Symphony table"
                ));
            }
            (false, Some(_)) => {
                return Err(format!(
                    "unknown {kind} parser '{name}': a Symphony table's name, which --parser \
                     symphony reads with"
                ));
            }
            (false, None) => return Err(format!("unknown {kind} parser '{name}'")),
        }
    }
    Ok(())
}

/// Whether `tool_choice` leaves the model no choice but a call: a named function, `required`,
/// or allowed tools in `required` mode.
fn forces_call(tool_choice: &ToolChoice) -> bool {
    match tool_choice {
        ToolChoice::Function { .. } => true,
        ToolChoice::Value(ToolChoiceValue::Required) => true,
        ToolChoice::AllowedTools { mode, .. } => mode == "required",
        ToolChoice::Value(_) => false,
    }
}

/// The structural tag the table gives a forced or required call to one of `tools`, with
/// `reasoning_open` when the prompt leaves the model inside its thought. `None` when no table
/// reads the response ([`reads`]), the choice forces no call, or the table has no grammar for its
/// call syntax yet; the older registry then decides as before.
pub(crate) fn constraint(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    tools: &[Tool],
    reasoning_open: bool,
) -> Option<ToolConstraint> {
    let format = reads(parser, names, request)?;
    if !request.tool_choice.as_ref().is_some_and(forces_call) {
        return None;
    }
    let grammar = format.grammar(tools, true, reasoning_open)?;
    Some(ToolConstraint::StructuralTag(grammar.payload().to_string()))
}

/// Symphony's assignment to a response: the table that reads it, and the part of the rendered
/// prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    pub format: Format,
    pub prompt: String,
}

/// Symphony's assignment to `request`, if a table reads the response ([`reads`]) and the
/// constraint, when there is one, is not a JSON schema: the older path's forced call, whose
/// output has no markers for a table to read, which the older registry writes where the table
/// has no grammar for its syntax yet.
pub(crate) fn assign(
    parser: ParserChoice,
    names: &Names,
    request: &ChatCompletionRequest,
    constraint: Option<&(String, String)>,
    prompt: &str,
) -> Option<Assignment> {
    let format = reads(parser, names, request)?;
    if constraint.is_some_and(|(kind, _)| kind == "json_schema") {
        return None;
    }
    debug!(
        model = %request.model,
        table = format.name(),
        "Symphony reads the response"
    );
    let prompt = format.replay(prompt).to_string();
    Some(Assignment { format, prompt })
}

/// The engine's finish reason as Symphony takes it.
pub(crate) fn engine_finish(reason: &str) -> EngineFinish {
    match reason {
        "stop" => EngineFinish::Stop,
        "length" => EngineFinish::Length,
        "abort" => EngineFinish::Abort,
        other => EngineFinish::Other(other.to_string()),
    }
}

/// What the `Finish` event among `events` says.
pub(crate) struct Finished {
    /// The calls made in the output.
    pub tool_calls: u32,
    /// The engine tokens spent on reasoning, markers excluded.
    pub reasoning_tokens: u32,
}

/// The `Finish` event among `events`, if the output ended.
pub(crate) fn finished(events: &[Event]) -> Option<Finished> {
    events.iter().find_map(|event| match event {
        Event::Finish {
            tool_calls,
            reasoning_tokens,
            ..
        } => Some(Finished {
            tool_calls: *tool_calls,
            reasoning_tokens: *reasoning_tokens,
        }),
        _ => None,
    })
}

/// The byte ranges of the output that are the message's content, from the events' sources in
/// order (every byte of the output lands in one event), adjacent runs joined: what the logprobs
/// filter keeps, as the older path keeps the tokens of `message.content` alone. Malformed text
/// is content, as `adapt::chat::message` writes it.
pub(crate) fn content_spans(events: &[Event]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    for event in events {
        let (bytes, content) = match event {
            Event::Content(text) | Event::Malformed { text, .. } => (text.text.len(), true),
            Event::Reasoning(text) | Event::Dropped { text, .. } => (text.text.len(), false),
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => (source.text.len(), false),
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => (0, false),
        };
        if content && bytes > 0 {
            match spans.last_mut() {
                Some((_, end)) if *end == at => *end = at + bytes,
                _ => spans.push((at, at + bytes)),
            }
        }
        at += bytes;
    }
    spans
}

/// The engine tokens the reasoning text among `events` cost.
pub(crate) fn reasoning_text_tokens(events: &[Event]) -> u32 {
    events
        .iter()
        .map(|event| match event {
            Event::Reasoning(text) => text.tokens.unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// One choice's reader: the engine over the choice's output, fed what the stop decoder releases
/// with the span of every token in it.
pub(crate) struct Reader {
    engine: Engine,
    events: Events,
    /// The last token the decoder saw: text it releases later belongs to it.
    last_token: Option<u32>,
    /// Where each token's text ends in the output read so far, in order; the text the decoder
    /// releases after a token extends that token's end, as the older path's `token_ends` do.
    ends: Vec<usize>,
    /// The bytes of the output read so far.
    read: usize,
    ended: bool,
}

impl Reader {
    /// A reader for one choice, seeded with the assignment's prompt.
    pub(crate) fn new(assignment: &Assignment, tools: &[Tool]) -> Result<Self, String> {
        let mut reader = Self {
            engine: Engine::new(assignment.format.clone(), Declared::of(tools)),
            events: Events::new(),
            last_token: None,
            ends: Vec::new(),
            read: 0,
            ended: false,
        };
        reader.feed(Input::Prompt {
            token_ids: &[],
            text: &assignment.prompt,
        })?;
        Ok(reader)
    }

    /// Reads `token_ids` through `decoder` as one delta: the text the decoder releases, with each
    /// token's span in it, zero-width for a token whose bytes it holds back. Gives the events,
    /// and whether the decoder matched a stop, after which the rest of the tokens are not read.
    pub(crate) fn read(
        &mut self,
        decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> Result<(Vec<Event>, bool), String> {
        let mut text = String::new();
        let mut spans = Vec::with_capacity(token_ids.len());
        let mut read = Vec::with_capacity(token_ids.len());
        let mut stopped = false;
        for &token_id in token_ids {
            let output = decoder
                .process_token(token_id)
                .map_err(|e| format!("Stop decoder failed to process token {token_id}: {e}"))?;
            let start = text.len();
            match output {
                SequenceDecoderOutput::Text(released) => text.push_str(&released),
                SequenceDecoderOutput::StoppedWithText(released) => {
                    text.push_str(&released);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => {}
            }
            spans.push(TokenSpan {
                token_id,
                start,
                end: text.len(),
                continued: false,
            });
            self.ends.push(self.read + text.len());
            read.push(token_id);
            self.last_token = Some(token_id);
            if stopped {
                break;
            }
        }
        self.read += text.len();
        self.feed(Input::Delta {
            token_ids: &read,
            text: &text,
            spans: &spans,
        })?;
        Ok((self.events.drain(), stopped))
    }

    /// Where each token's text ends in the output read so far, for the logprobs filter the older
    /// path applies: a token belongs to the content when its text does.
    pub(crate) fn token_ends(&self) -> &[usize] {
        &self.ends
    }

    /// Reads `text` the decoder released after its last token, which held it back as a possible
    /// stop: bytes of that token.
    pub(crate) fn tail(&mut self, text: &str) -> Result<Vec<Event>, String> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let spans: Vec<TokenSpan> = self
            .last_token
            .map(|token_id| TokenSpan {
                token_id,
                start: 0,
                end: text.len(),
                continued: true,
            })
            .into_iter()
            .collect();
        self.read += text.len();
        if let Some(last) = self.ends.last_mut() {
            *last = self.read;
        }
        self.feed(Input::Delta {
            token_ids: &[],
            text,
            spans: &spans,
        })?;
        Ok(self.events.drain())
    }

    /// Ends the output with the engine's finish, once: the events, the last of them the
    /// `Finish`, or nothing when the output already ended.
    pub(crate) fn end(&mut self, finish: EngineFinish) -> Result<Vec<Event>, String> {
        if self.ended {
            return Ok(Vec::new());
        }
        self.ended = true;
        self.feed(Input::End { finish })?;
        Ok(self.events.drain())
    }

    /// Whether the output ended.
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    fn feed(&mut self, input: Input<'_>) -> Result<(), String> {
        self.engine
            .feed(input, &mut self.events)
            .map_err(|e| format!("Symphony failed to read the output: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use symphony::{formats::qwen3, CallSyntax};

    use super::*;

    fn request(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({"model": "Qwen/Qwen3-8B", "messages": [], "stream": true});
        for (key, value) in extra.as_object().into_iter().flatten() {
            request[key] = value.clone();
        }
        serde_json::from_value(request).expect("chat request")
    }

    fn tools() -> Vec<Tool> {
        serde_json::from_value(json!([{
            "type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .expect("tools")
    }

    const PROMPT: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";

    fn names(tool: Option<&str>, reasoning: Option<&str>) -> Names {
        Names {
            tool: tool.map(str::to_string),
            reasoning: reasoning.map(str::to_string),
        }
    }

    #[test]
    fn the_names_select_a_table_when_they_are_one_tables_and_nothing_otherwise() {
        assert_eq!(named_table(&Names::default()), None);
        assert_eq!(
            named_table(&names(Some("kimi_k3"), Some("kimi_k3"))),
            Some(Table::KimiK3)
        );
        assert_eq!(named_table(&names(None, Some("Ling"))), Some(Table::Ling));
        assert_eq!(
            named_table(&names(Some("qwen3_tagged"), None)),
            Some(Table::Qwen3Tagged)
        );
        // A legacy parser's name, alone or beside a table's, and two tables select none.
        assert_eq!(named_table(&names(Some("qwen"), None)), None);
        assert_eq!(named_table(&names(Some("qwen"), Some("qwen3"))), None);
        assert_eq!(named_table(&names(Some("kimi_k3"), Some("qwen3"))), None);
    }

    #[test]
    fn explicit_names_win_over_the_registry() {
        // A table named for a model the registry does not know reads it; a legacy name leaves a
        // model the registry knows to the legacy parsers; a table named for a model the registry
        // reads with another wins, and replays from its own opener.
        let unknown = request(json!({"model": "unknown"}));
        let ling = names(Some("ling"), Some("ling"));
        let assigned = assign(ParserChoice::Symphony, &ling, &unknown, None, PROMPT);
        assert_eq!(assigned.expect("assigned").format.name(), "ling");
        let legacy = names(Some("qwen"), Some("qwen3"));
        assert!(assign(
            ParserChoice::Symphony,
            &legacy,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let required = request(json!({"tool_choice": "required"}));
        let reasoning_only = names(None, Some("deepseek_r1"));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &required,
            &tools(),
            false
        )
        .is_none());
        let kimi = names(Some("kimi_k3"), None);
        let assigned = assign(
            ParserChoice::Symphony,
            &kimi,
            &request(json!({})),
            None,
            PROMPT,
        );
        let assigned = assigned.expect("assigned");
        assert_eq!(assigned.format.name(), "kimi_k3");
        assert_eq!(
            assigned.prompt, PROMPT,
            "no Kimi opener in a Qwen prompt: the whole replays"
        );
        // The registry's own table named for its model keeps the model's turn opener.
        let exaone = request(json!({"model": "LGAI-EXAONE/K-EXAONE-236B-A23B"}));
        let prompt = "<|user|>hi<|endofturn|><|assistant|>\n<think>\n";
        let qwen3 = names(Some("qwen3"), Some("qwen3"));
        let by_name = assign(ParserChoice::Symphony, &qwen3, &exaone, None, prompt);
        let by_registry = assign(
            ParserChoice::Symphony,
            &Names::default(),
            &exaone,
            None,
            prompt,
        );
        assert_eq!(
            by_name.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        assert_eq!(
            by_registry.expect("assigned").prompt,
            "<|assistant|>\n<think>\n"
        );
        // Under legacy the names change nothing.
        assert!(assign(ParserChoice::Legacy, &ling, &unknown, None, PROMPT).is_none());
    }

    #[test]
    fn names_are_checked_against_both_registries_by_the_parser_in_force() {
        let legacy = ParserChoice::Legacy;
        let symphony = ParserChoice::Symphony;
        assert_eq!(check_names(legacy, &Names::default(), true, true), Ok(()));
        assert_eq!(
            check_names(legacy, &names(Some("qwen"), Some("qwen3")), true, true),
            Ok(())
        );
        let why = check_names(legacy, &names(Some("nonsense"), None), false, true);
        assert_eq!(
            why.expect_err("unknown"),
            "unknown tool-call parser 'nonsense'"
        );
        let why = check_names(legacy, &names(None, Some("ling")), true, false);
        assert_eq!(
            why.expect_err("a table under legacy"),
            "unknown reasoning parser 'ling': a Symphony table's name, which --parser symphony \
             reads with"
        );
        // Under Symphony a table's name passes alone or doubled, and binds the other name.
        assert_eq!(
            check_names(symphony, &names(Some("ling"), None), false, true),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(Some("ling"), Some("LING")), false, false),
            Ok(())
        );
        assert_eq!(
            check_names(symphony, &names(Some("kimi_k3"), Some("qwen3")), true, true),
            Ok(())
        );
        let why = check_names(symphony, &names(Some("ling"), Some("qwen3")), false, true);
        assert_eq!(
            why.expect_err("bound to the table"),
            "tool-call parser 'ling' is a Symphony table's name and no legacy parser's, so the \
             other parser must name the same table or nothing, not 'qwen3'"
        );
        let why = check_names(symphony, &names(None, Some("nonsense")), true, false);
        assert_eq!(
            why.expect_err("unknown"),
            "unknown reasoning parser 'nonsense': neither a legacy parser nor a Symphony table"
        );
    }

    #[test]
    fn the_header_names_the_parser_over_the_flag_and_an_unknown_name_is_refused() {
        assert_eq!(choice(ParserChoice::Legacy, None), Ok(ParserChoice::Legacy));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&HeaderMap::new())),
            Ok(ParserChoice::Symphony)
        );
        let mut headers = HeaderMap::new();
        headers.insert(HEADER, "symphony".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Legacy, Some(&headers)),
            Ok(ParserChoice::Symphony)
        );
        headers.insert(HEADER, " Legacy ".parse().expect("a header value"));
        assert_eq!(
            choice(ParserChoice::Symphony, Some(&headers)),
            Ok(ParserChoice::Legacy)
        );
        headers.insert(HEADER, "fast".parse().expect("a header value"));
        let why = choice(ParserChoice::Legacy, Some(&headers)).expect_err("refused");
        assert_eq!(why, "x-smg-parser must be legacy or symphony, not \"fast\"");
    }

    #[test]
    fn the_legacy_choice_assigns_nothing_and_asks_for_no_constraint() {
        assert!(assign(
            ParserChoice::Legacy,
            &Names::default(),
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        let required = request(json!({"tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Legacy,
            &Names::default(),
            &required,
            &tools(),
            false
        )
        .is_none());
    }

    #[test]
    fn a_request_no_table_reads_takes_no_constraint_from_one() {
        // The older parsers read this response (it keeps the thought in the content), so a tag
        // in the table's syntax would hand them a call they cannot read: the older registry
        // decides, as it does for the whole request.
        let unread = request(json!({"tool_choice": "required", "separate_reasoning": false}));
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &unread,
            None,
            PROMPT
        )
        .is_none());
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &unread,
            &tools(),
            false
        )
        .is_none());
        let read = request(json!({"tool_choice": "required"}));
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &read,
            None,
            PROMPT
        )
        .is_some());
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &read,
            &tools(),
            false
        )
        .is_some());
    }

    #[test]
    fn a_known_model_is_assigned_its_table_and_the_prompts_replay() {
        let assignment = assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(assignment.format.name(), "qwen3");
        assert_eq!(assignment.prompt, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn an_unknown_model_and_requests_the_older_path_answers_differently_are_not_assigned() {
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"model": "unknown"})),
            None,
            PROMPT
        )
        .is_none());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"separate_reasoning": false})),
            None,
            PROMPT
        )
        .is_none());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({"tool_choice": "none"})),
            None,
            PROMPT
        )
        .is_none());
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            Some(&schema),
            PROMPT
        )
        .is_none());
        let tag = ("structural_tag".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &Names::default(),
            &request(json!({})),
            Some(&tag),
            PROMPT
        )
        .is_some());
    }

    #[test]
    fn a_forced_call_takes_the_tables_structural_tag_and_auto_takes_none() {
        let required = request(json!({"tool_choice": "required"}));
        let Some(ToolConstraint::StructuralTag(tag)) = constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &required,
            &tools(),
            false,
        ) else {
            panic!("a structural tag");
        };
        let tag: Value = serde_json::from_str(&tag).expect("JSON");
        assert!(tag["format"].is_object(), "{tag}");
        let auto = request(json!({"tool_choice": "auto"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &auto,
            &tools(),
            false
        )
        .is_none());
        let unknown = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &Names::default(),
            &unknown,
            &tools(),
            false
        )
        .is_none());
    }

    #[test]
    fn the_engines_finish_reasons_map_and_the_rest_carry_through() {
        assert_eq!(engine_finish("stop"), EngineFinish::Stop);
        assert_eq!(engine_finish("length"), EngineFinish::Length);
        assert_eq!(engine_finish("abort"), EngineFinish::Abort);
        assert_eq!(
            engine_finish("failed"),
            EngineFinish::Other("failed".to_string())
        );
    }

    #[test]
    fn a_reader_ends_once() {
        let assignment = Assignment {
            format: qwen3(CallSyntax::Json),
            prompt: String::new(),
        };
        let mut reader = Reader::new(&assignment, &[]).expect("a reader");
        assert!(!reader.ended());
        let events = reader.end(EngineFinish::Stop).expect("ended");
        assert!(finished(&events).is_some());
        assert!(reader.ended());
        assert!(reader.end(EngineFinish::Stop).expect("ended").is_empty());
    }
}

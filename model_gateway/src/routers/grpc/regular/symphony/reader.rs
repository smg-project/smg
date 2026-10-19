//! The reader: Symphony's assignment to a response and the reader the drivers run over the
//! stop decoder's output, one per choice.

use std::sync::Arc;

use llm_tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use openai_protocol::common::Tool;
use symphony::{
    Declared, Engine, EngineFinish, Event, Events, Format, Input, Parser as _, TokenSpan,
};
use tracing::debug;

use super::{
    constraint::Asks,
    names::reads,
    telemetry::{self, Tally, Who},
    template::Templates,
    SymphonyError,
};
use crate::{config::ParserChoice, routers::grpc::utils::ModelNames};

/// Symphony's assignment to a response: the table that reads it, as the request's asks make it
/// ([`assign`]), and the part of the rendered prompt the engine replays.
#[derive(Clone, Debug)]
pub(crate) struct Assignment {
    /// The table, shared with every reader of the response and every request it reads for.
    pub format: Arc<Format>,
    pub prompt: String,
}

/// The assignment of `format` to a response, with the part of `prompt` the engine replays. Not
/// made when the constraint is a JSON schema: the older path's forced call, whose output has no
/// markers for a table to read, which the older registry writes where the table has no tag.
pub(super) fn assignment(
    format: Option<Arc<Format>>,
    model: &str,
    constraint: Option<&(String, String)>,
    prompt: &str,
) -> Option<Assignment> {
    let format = format?;
    if constraint.is_some_and(|(kind, _)| kind == "json_schema") {
        return None;
    }
    debug!(model, table = format.name(), "Symphony reads the response");
    let prompt = format.replay(prompt).to_string();
    Some(Assignment { format, prompt })
}

/// The table as the request's asks make it: without its calls when the legacy path would run no
/// tool parser (the request asks for none, or its response is not scanned for calls:
/// [`scans_for_tool_calls`](crate::routers::grpc::spec::scans_for_tool_calls)), so a table that reads calls from the output's first byte (`json`,
/// `xlam`) is content from it; without its thought when the request keeps the reasoning in the
/// content, as the legacy path runs no reasoning parser then. A variant is a copy of the shared
/// table; a request that asks for neither reads with the shared table itself.
pub(super) fn asked(format: Arc<Format>, no_calls: bool, no_reasoning: bool) -> Arc<Format> {
    if !no_calls && !no_reasoning {
        return format;
    }
    let mut format = Arc::unwrap_or_clone(format);
    if no_calls {
        format = format.without_calls();
    }
    if no_reasoning {
        format = format.without_reasoning();
    }
    Arc::new(format)
}

/// Symphony's assignment to a request, if a table reads the response ([`reads`]) and the
/// constraint is not a JSON schema ([`assignment`]): the table without its calls where the legacy
/// path would run no tool parser, without its thought where it would run no reasoning parser
/// ([`Asks::no_calls`], [`Asks::no_reasoning`], [`asked`]). A call the conversation established
/// without a tool inventory comes back as a call under either parser.
pub(crate) fn assign(
    parser: ParserChoice,
    names: &ModelNames<'_>,
    asks: Asks<'_>,
    constraint: Option<&(String, String)>,
    prompt: &str,
    templates: &Templates,
) -> Result<Option<Assignment>, SymphonyError> {
    let format = reads(parser, names, asks.model(), templates)?;
    let format = format.map(|format| asked(format, asks.no_calls(), asks.no_reasoning()));
    Ok(assignment(format, asks.model(), constraint, prompt))
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

/// The delta the reader feeds the engine for one engine chunk: the text the stop decoder released,
/// the span of every token in it and the tokens read. The step's events borrow its text, so it
/// lives with the caller, next to the events, and a stream fills the same one every step.
#[derive(Debug, Default)]
pub(crate) struct Delta {
    text: String,
    spans: Vec<TokenSpan>,
    tokens: Vec<u32>,
}

impl Delta {
    fn clear(&mut self) {
        self.text.clear();
        self.spans.clear();
        self.tokens.clear();
    }

    fn input(&self) -> Input<'_> {
        Input::Delta {
            token_ids: &self.tokens,
            text: &self.text,
            spans: &self.spans,
        }
    }
}

/// One choice's reader: the engine over the choice's output, fed what the stop decoder releases
/// with the span of every token in it. The events of a step go into the caller's [`Events`] and
/// borrow the step's [`Delta`] text; the caller reads them in place and recycles the buffer
/// (`docs/engine.md`, the shape of a step's events). What it reads is counted as it goes
/// ([`telemetry`]) and recorded when the output ends.
pub(crate) struct Reader {
    engine: Engine,
    /// The `table` label of the format, for the counters and the line.
    table: &'static str,
    /// What the output said so far that was not content, reasoning or a call.
    tally: Tally,
    /// The last token the decoder saw: text it releases later belongs to it.
    last_token: Option<u32>,
    /// Where each token's text ends in the output read so far, in order; the text the decoder
    /// releases after a token extends that token's end, as the older path's `token_ends` do.
    ends: Vec<usize>,
    /// The bytes of the output read so far.
    read: usize,
    ended: bool,
    /// Drop the events of every call after the first.
    first_call_only: bool,
}

impl Reader {
    /// A reader for one choice, seeded with the assignment's prompt; the events the prompt's
    /// replay gives (a thought the prompt opens) go into `out`, before the first delta's. `tools`
    /// are the request's as its response spec holds them: every choice's reader shares them,
    /// none copies a schema.
    pub(crate) fn new(
        assignment: &Assignment,
        tools: Arc<[Tool]>,
        out: &mut Events<'_>,
    ) -> Result<Self, SymphonyError> {
        let mut reader = Self {
            table: telemetry::table_label(assignment.format.name()),
            engine: Engine::shared(Arc::clone(&assignment.format), Declared::shared(tools)),
            tally: Tally::default(),
            last_token: None,
            ends: Vec::new(),
            read: 0,
            ended: false,
            first_call_only: false,
        };
        // The prompt's events carry no text of the prompt (a thought it opens or closes), so they
        // move from the replay's own buffer into the caller's, whatever delta that one borrows.
        let mut replayed = Events::new();
        reader.feed(
            Input::Prompt {
                token_ids: &[],
                text: &assignment.prompt,
            },
            &mut replayed,
        )?;
        for event in replayed.into_owned() {
            out.push(event);
        }
        Ok(reader)
    }

    /// Under `parallel_tool_calls: false` or `disable_parallel_tool_use`, the
    /// events of every call after the first are dropped: the table's grammar
    /// limits a forced call, the reader limits a free one.
    pub(crate) fn first_call_only(mut self, on: bool) -> Self {
        self.first_call_only = on;
        self
    }

    /// Reads `token_ids` through `decoder` as one delta: the text the decoder releases, with each
    /// token's span in it, zero-width for a token whose bytes it holds back, into `delta`, and
    /// the events into `out`, where they borrow the text. Says whether the decoder matched a
    /// stop, after which the rest of the tokens are not read.
    pub(crate) fn read<'d>(
        &mut self,
        decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
        delta: &'d mut Delta,
        out: &mut Events<'d>,
    ) -> Result<bool, SymphonyError> {
        delta.clear();
        let mut stopped = false;
        for &token_id in token_ids {
            let output = decoder.process_token(token_id).map_err(|e| {
                SymphonyError::Read(format!(
                    "Stop decoder failed to process token {token_id}: {e}"
                ))
            })?;
            let start = delta.text.len();
            match output {
                SequenceDecoderOutput::Text(released) => delta.text.push_str(&released),
                SequenceDecoderOutput::StoppedWithText(released) => {
                    delta.text.push_str(&released);
                    stopped = true;
                }
                SequenceDecoderOutput::Stopped => stopped = true,
                SequenceDecoderOutput::Held => {}
            }
            delta.spans.push(TokenSpan {
                token_id,
                start,
                end: delta.text.len(),
                continued: false,
            });
            self.ends.push(self.read + delta.text.len());
            delta.tokens.push(token_id);
            self.last_token = Some(token_id);
            if stopped {
                break;
            }
        }
        self.read += delta.text.len();
        self.feed(delta.input(), out)?;
        Ok(stopped)
    }

    /// Where each token's text ends in the output read so far, for the logprobs filter the older
    /// path applies: a token belongs to the content when its text does.
    pub(crate) fn token_ends(&self) -> &[usize] {
        &self.ends
    }

    /// Reads `text` the decoder released after its last token, which held it back as a possible
    /// stop: bytes of that token. The text moves into `delta`, whose text the events borrow.
    pub(crate) fn tail<'d>(
        &mut self,
        text: String,
        delta: &'d mut Delta,
        out: &mut Events<'d>,
    ) -> Result<(), SymphonyError> {
        if text.is_empty() {
            return Ok(());
        }
        delta.clear();
        delta
            .spans
            .extend(self.last_token.map(|token_id| TokenSpan {
                token_id,
                start: 0,
                end: text.len(),
                continued: true,
            }));
        delta.text = text;
        self.read += delta.text.len();
        if let Some(last) = self.ends.last_mut() {
            *last = self.read;
        }
        self.feed(delta.input(), out)
    }

    /// Ends the output with the engine's finish, once: the events, the last of them the
    /// `Finish`, go into `out`; nothing when the output already ended. The output is recorded
    /// here, under `who`: its counters, and the line when it was not clean ([`Tally::report`]).
    pub(crate) fn end(
        &mut self,
        finish: EngineFinish,
        who: Who<'_>,
        out: &mut Events<'_>,
    ) -> Result<(), SymphonyError> {
        if self.ended {
            return Ok(());
        }
        self.ended = true;
        self.feed(Input::End { finish }, out)?;
        self.tally.report(self.table, who);
        Ok(())
    }

    /// Whether the output ended.
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    /// Feeds `input` to the engine, its events into `out` after the ones already there: the
    /// reader's one way in. The events this step added are tallied, and limited to the first
    /// call where the request asks. An output the engine refuses is recorded as failed for the
    /// table, which the error names.
    fn feed<'d>(&mut self, input: Input<'d>, out: &mut Events<'d>) -> Result<(), SymphonyError> {
        let before = out.len();
        let last = out.as_slice().last().map_or(0, telemetry::bytes_of);
        self.engine.feed(input, out).map_err(|e| {
            telemetry::record_failed(self.table);
            SymphonyError::Read(format!(
                "Symphony failed to read the output with the {} table: {e}",
                self.table
            ))
        })?;
        // Text of one kind pushed right after text of that kind joins it (`Events::push`): bytes
        // that grew an event already tallied are counted without an event of their own.
        if let Some(grown) = before.checked_sub(1).map(|at| &out.as_slice()[at]) {
            self.tally.grew(telemetry::bytes_of(grown) - last);
        }
        let fresh = &mut out.as_mut_slice()[before..];
        for event in fresh.iter() {
            self.tally.see(event, self.first_call_only);
        }
        if self.first_call_only {
            symphony::event::first_call_only(fresh);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::grpc::regular::symphony::testing::{assign, constraint, *};

    #[test]
    fn a_request_without_tools_is_read_by_every_table_and_its_calls_are_content() {
        // `json` and `xlam` read a bare JSON answer as a call: without tools the request asked
        // for none, so the table goes without its calls and the answer is the text it was, as
        // the legacy parsers leave it when they run no tool parser; a table with call markers
        // goes the same way.
        let json = names(Some("json"), None);
        let without = request(json!({}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &without, None, PROMPT).expect("assigned");
        assert_eq!(assigned.format.name(), "json");
        assert_eq!(assigned.format.call_syntax(), None);
        let declared = serde_json::to_value(tools()).expect("tools");
        let with = request(json!({"tools": declared}));
        let assigned =
            assign(ParserChoice::Symphony, &json, &with, None, PROMPT).expect("assigned");
        assert!(assigned.format.call_syntax().is_some());
        let xlam = request(json!({"model": "Salesforce/Llama-xLAM-2-8b-fc-r"}));
        let by_registry = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &xlam,
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(by_registry.format.name(), "xlam");
        assert_eq!(by_registry.format.call_syntax(), None);
        for named in ["inkling", "llama_nemotron"] {
            let assigned = assign(
                ParserChoice::Symphony,
                &names(Some(named), None),
                &without,
                None,
                PROMPT,
            )
            .expect("assigned");
            assert_eq!(assigned.format.name(), named);
            assert_eq!(assigned.format.call_syntax(), None);
        }
    }

    #[test]
    fn explicit_names_win_over_the_registry() {
        // A table named for a model the registry does not know reads it; a legacy name reads
        // with its alias, or with no alias leaves a model the registry knows to the legacy
        // parsers; a table named for a model the registry reads with another wins, and replays
        // from its own opener.
        let unknown = request(json!({"model": "unknown"}));
        let ling = names(Some("ling"), Some("ling"));
        let assigned = assign(ParserChoice::Symphony, &ling, &unknown, None, PROMPT);
        assert_eq!(assigned.expect("assigned").format.name(), "ling");
        let aliased = names(Some("qwen"), Some("qwen3"));
        let assigned = assign(
            ParserChoice::Symphony,
            &aliased,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let legacy = names(Some("pythonic"), Some("deepseek_r1"));
        assert!(assign(
            ParserChoice::Symphony,
            &legacy,
            &request(json!({})),
            None,
            PROMPT
        )
        .is_none());
        // A reasoning name alone that selects no table leaves the choice to the registry: a
        // known model takes its table and the table's tag, an unknown one the legacy parsers.
        let required = request(json!({"tool_choice": "required"}));
        let reasoning_only = names(None, Some("deepseek_r1"));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &required,
            &tools(),
            false
        )
        .is_some());
        let assigned = assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &request(json!({})),
            None,
            PROMPT,
        );
        assert_eq!(assigned.expect("assigned").format.name(), "qwen3");
        let unknown_required = request(json!({"model": "unknown", "tool_choice": "required"}));
        assert!(constraint(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown_required,
            &tools(),
            false
        )
        .is_none());
        assert!(assign(
            ParserChoice::Symphony,
            &reasoning_only,
            &unknown,
            None,
            PROMPT
        )
        .is_none());
        // `passthrough` alone is the choice to leave the thought as written: the legacy
        // parsers keep a known model too.
        let passthrough_only = names(None, Some("passthrough"));
        assert!(assign(
            ParserChoice::Symphony,
            &passthrough_only,
            &request(json!({})),
            None,
            PROMPT
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
            &ModelNames::default(),
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
    fn the_requests_asks_take_a_variant_of_the_table_and_do_not_switch_the_reader() {
        // A request that keeps the thought in the content, or asks for no calls, is read by the
        // same table without its thought or its calls; the constraint is the full table's.
        let table = |request: &ChatCompletionRequest| {
            reads(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &request.model,
                &templates(),
            )
            .expect("read")
            .expect("a table")
        };
        let assigned = |request: &ChatCompletionRequest| {
            assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                request,
                None,
                PROMPT,
            )
            .expect("assigned")
            .format
        };
        let declared = serde_json::to_value(tools()).expect("tools");
        let kept = request(
            json!({"tools": declared, "tool_choice": "required", "separate_reasoning": false}),
        );
        assert_eq!(
            *assigned(&kept),
            Arc::unwrap_or_clone(table(&kept)).without_reasoning()
        );
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &kept,
            &tools(),
            false
        )
        .is_some());
        let no_calls = request(json!({"tool_choice": "none"}));
        assert_eq!(
            *assigned(&no_calls),
            Arc::unwrap_or_clone(table(&no_calls)).without_calls()
        );
        assert!(constraint(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &no_calls,
            &tools(),
            false
        )
        .is_none());
        let neither = request(json!({"separate_reasoning": false}));
        assert_eq!(
            *assigned(&neither),
            Arc::unwrap_or_clone(table(&neither))
                .without_calls()
                .without_reasoning()
        );
        let both = request(json!({"tools": declared}));
        assert_eq!(assigned(&both), table(&both));
    }

    #[test]
    fn the_calls_stay_wherever_the_legacy_path_runs_a_tool_parser() {
        // The legacy path scans a response for calls when the request declares tools, an empty
        // list included, or when the model's provider reads calls without any; the table keeps
        // its calls exactly then, so a call the conversation established comes back as a call
        // under either parser, and a `tool_choice` of `none` drops them under either.
        let table = |request: &ChatCompletionRequest| {
            reads(
                ParserChoice::Symphony,
                &ModelNames::default(),
                &request.model,
                &templates(),
            )
            .expect("read")
            .expect("a table")
        };
        let assigned = |request: &ChatCompletionRequest| {
            assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                request,
                None,
                PROMPT,
            )
            .expect("assigned")
            .format
        };
        let retry = request(json!({"model": "MiniMax-M3", "messages": retry_messages()}));
        assert!(scans_for_tool_calls(&retry));
        assert_eq!(
            assigned(&retry),
            table(&retry),
            "the provider reads calls without tools"
        );
        let refused = request(json!({
            "model": "MiniMax-M3", "messages": retry_messages(), "tool_choice": "none"
        }));
        assert_eq!(
            *assigned(&refused),
            Arc::unwrap_or_clone(table(&refused)).without_calls()
        );
        let declared_none = request(json!({"tools": []}));
        assert!(scans_for_tool_calls(&declared_none));
        assert_eq!(
            assigned(&declared_none),
            table(&declared_none),
            "an empty list is a declaration"
        );
        let undeclared = request(json!({"messages": retry_messages()}));
        assert!(!scans_for_tool_calls(&undeclared));
        assert_eq!(
            *assigned(&undeclared),
            Arc::unwrap_or_clone(table(&undeclared)).without_calls(),
            "a provider that reads calls only with tools"
        );
        // A Messages request: declared tools, an empty list included, keep the calls; none
        // declared drops them, as the legacy path runs its tool parser on `tools` alone.
        let messages = |extra: Value| {
            super::assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Messages {
                    request: &message_request(extra),
                    continues: false,
                },
                None,
                PROMPT,
                &templates(),
            )
            .expect("read")
            .expect("assigned")
            .format
        };
        assert!(messages(json!({"tools": []})).call_syntax().is_some());
        assert_eq!(messages(json!({})).call_syntax(), None);
    }

    #[test]
    fn a_known_model_is_assigned_its_table_and_the_prompts_replay() {
        let assignment = assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            None,
            PROMPT,
        )
        .expect("assigned");
        assert_eq!(assignment.format.name(), "qwen3");
        assert_eq!(assignment.prompt, "<|im_start|>assistant\n<think>\n");
    }

    #[test]
    fn an_unknown_model_and_a_json_schema_constraint_are_not_assigned() {
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({"model": "unknown"})),
            None,
            PROMPT
        )
        .is_none());
        let schema = ("json_schema".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            Some(&schema),
            PROMPT
        )
        .is_none());
        let tag = ("structural_tag".to_string(), "{}".to_string());
        assert!(assign(
            ParserChoice::Symphony,
            &ModelNames::default(),
            &request(json!({})),
            Some(&tag),
            PROMPT
        )
        .is_some());
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
    fn a_reader_gives_a_json_answer_bare_when_the_model_said_its_kind_again() {
        // The table whose messages say their kind after their opener, as a request without tools
        // reads it, over the two outputs an engine that starts a JSON answer's grammar after its
        // own reasoning parser lets the model write: the text marker twice, or a second thought
        // opened inside the answer's message. The client sees the bare JSON as content, the
        // thoughts as reasoning, and the reader reads no malformed run.
        let assignment = Assignment {
            format: Arc::new(symphony::formats::inkling().without_calls()),
            prompt: "<|message_model|>".to_string(),
        };
        let json = "{\"city\": \"Oslo\", \"country\": \"Norway\"}";
        let thought = "<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>";
        for (output, reasoning) in [
            (
                format!("{thought}<|content_text|>{json}<|end_message|>"),
                "plan",
            ),
            (
                format!(
                    "{thought}<|content_thinking|>more<|end_message|><|message_model|>\
                     <|content_text|>{json}<|end_message|>"
                ),
                "planmore",
            ),
        ] {
            let mut reader = Reader::new(&assignment, Arc::default()).expect("a reader");
            let mut events = reader.tail(&output).expect("read");
            events.extend(reader.end(EngineFinish::Stop, who()).expect("ended"));
            assert_eq!(content_of(&events), json, "{events:?}");
            let thoughts: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Reasoning(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(thoughts, reasoning, "{events:?}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{events:?}"
            );
            let Some(Event::Finish { tool_calls, .. }) = events.last() else {
                panic!("a finish");
            };
            assert_eq!(*tool_calls, 0);
        }
    }

    #[test]
    fn a_reader_ends_once() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
        };
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
        assert!(!reader.ended());
        reader
            .end(EngineFinish::Stop, who(), &mut events)
            .expect("ended");
        assert!(finished(events.as_slice()).is_some());
        assert!(reader.ended());
        let mut again = Events::new();
        reader
            .end(EngineFinish::Stop, who(), &mut again)
            .expect("ended");
        assert!(again.is_empty());
    }

    #[tokio::test]
    async fn hf_reads_with_the_response_templates_table_and_refuses_a_model_without_one() {
        let templates = template::fixtures::templates_with(
            "Qwen/Qwen3-8B",
            template::fixtures::qwen3_5_tokenizer(),
        )
        .await;
        let hf = names(Some("hf"), Some(" HF "));
        let read = |names: &ModelNames<'_>, request: &ChatCompletionRequest| {
            reads(ParserChoice::Symphony, names, &request.model, &templates)
        };
        // The template's table reads the model the registry knows with another table, named
        // on both sides or on either alone.
        for names in [&hf, &names(Some("hf"), None), &names(None, Some("hf"))] {
            let format = read(names, &request(json!({})))
                .expect("read")
                .expect("a table");
            assert_eq!(format.name(), "response_template");
        }
        assert!(
            reads(ParserChoice::Legacy, &hf, "Qwen/Qwen3-8B", &templates)
                .expect("read")
                .is_none()
        );
        // What the request asks of the channels takes a variant of the templates' table.
        let kept = request(json!({"separate_reasoning": false}));
        let assigned = super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&kept),
            None,
            PROMPT,
            &templates,
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(
            *assigned.format,
            Arc::unwrap_or_clone(read(&hf, &kept).expect("read").expect("a table"))
                .without_calls()
                .without_reasoning()
        );
        // A model with no table of its own is refused with the reason, in both decisions.
        let unknown = request(json!({"model": "unknown"}));
        assert_eq!(
            read(&hf, &unknown).expect_err("refused"),
            SymphonyError::Template("no tokenizer is registered for model 'unknown'".to_string())
        );
        assert!(super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&unknown),
            None,
            PROMPT,
            &templates
        )
        .is_err());
        let assigned = super::assign(
            ParserChoice::Symphony,
            &hf,
            Asks::Chat(&request(json!({}))),
            None,
            PROMPT,
            &templates,
        )
        .expect("read")
        .expect("assigned");
        assert_eq!(assigned.format.name(), "response_template");
        assert_eq!(assigned.prompt, "<|im_start|>assistant\n<think>\n");
    }

    /// Limited to one call, the reader keeps the first call's events and
    /// counts one call; unlimited, both calls come through.
    #[test]
    fn a_reader_limited_to_one_call_drops_the_calls_after_the_first() {
        const TWO_CALLS: &str = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>\
                                 <tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2}}\n</tool_call>";
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
        };
        let calls = |first_only: bool| {
            let mut events = Events::new();
            let mut reader = Reader::new(&assignment, tools().into(), &mut events)
                .expect("a reader")
                .first_call_only(first_only);
            let events = read_tail(&mut reader, TWO_CALLS);
            let starts = events
                .iter()
                .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                .count();
            let Some(Event::Finish { tool_calls, .. }) = events.last() else {
                panic!("no finish event");
            };
            (starts, *tool_calls)
        };
        assert_eq!(calls(false), (2, 2));
        assert_eq!(calls(true), (1, 1));
    }

    /// The unary reads feed the output and then the decoder's last text into one buffer, where
    /// content at the output's end and the flushed text join one event: the tally counts the
    /// joined bytes, so a malformed run the end reports after them stands at the byte it stands
    /// at when the output comes in one step.
    #[test]
    fn the_tally_counts_the_bytes_a_later_step_merged_into_an_earlier_event() {
        let assignment = Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: String::new(),
        };
        let first_run_at = |steps: &[&str]| -> (Option<usize>, usize) {
            let mut events = Events::new();
            let mut reader =
                Reader::new(&assignment, tools().into(), &mut events).expect("a reader");
            let mut deltas: Vec<Delta> = steps.iter().map(|_| Delta::default()).collect();
            for (text, delta) in steps.iter().zip(deltas.iter_mut()) {
                reader
                    .tail((*text).to_string(), delta, &mut events)
                    .expect("read");
            }
            reader
                .end(EngineFinish::Stop, who(), &mut events)
                .expect("ended");
            // Where the first malformed event stands, by the events' own bytes.
            let before = events
                .as_slice()
                .iter()
                .take_while(|event| !matches!(event, Event::Malformed { .. }))
                .map(telemetry::bytes_of)
                .sum();
            (reader.tally.first_run().map(|first| first.at), before)
        };
        let (content, rest) = (
            "Hello",
            " there<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": ",
        );
        let whole = format!("{content}{rest}");
        let (in_one_step, by_the_events) = first_run_at(&[&whole]);
        assert_eq!(in_one_step, Some(by_the_events), "the one-step reading");
        assert!(
            by_the_events > content.len(),
            "the cut call stands after the content"
        );
        assert_eq!(
            first_run_at(&[content, rest]),
            (in_one_step, by_the_events),
            "the two-step reading, its second step's content joined to the first's"
        );
    }

    #[test]
    fn a_bare_answer_the_json_table_reads_as_a_call_comes_back_as_its_text() {
        let assignment = Assignment {
            format: Arc::new(symphony::formats::json().without_calls()),
            prompt: String::new(),
        };
        let mut events = Events::new();
        let mut reader = Reader::new(&assignment, Arc::default(), &mut events).expect("a reader");
        let answer = "{\"answer\": 42}";
        let events = read_tail(&mut reader, answer);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(text, answer);
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
    }

    #[test]
    fn a_reader_over_a_tables_variant_gives_the_calls_or_the_thought_as_content() {
        // The table itself: the thought apart, the call a call.
        let events = read_with(qwen3(CallSyntax::Json));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert_eq!(content_of(&events).trim(), "Hello");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 1);
        // Without its calls: the call and its markers are content, the finish counts no call.
        let events = read_with(qwen3(CallSyntax::Json).without_calls());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::Reasoning(_))));
        let content = content_of(&events);
        assert!(
            content
                .trim_start()
                .starts_with("Hello <tool_call>\n{\"name\": \"lookup\""),
            "{content:?}"
        );
        assert!(content.ends_with("</tool_call>"), "{content:?}");
        assert!(!content.contains("<think>"), "{content:?}");
        let Some(Event::Finish { tool_calls, .. }) = events.last() else {
            panic!("a finish");
        };
        assert_eq!(*tool_calls, 0);
        // Without its thought: the thought and its markers are content, the call stays a call.
        let events = read_with(qwen3(CallSyntax::Json).without_reasoning());
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Reasoning(_) | Event::ReasoningStart)));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
        let content = content_of(&events);
        assert!(content.starts_with("<think>plan</think>"), "{content:?}");
        assert!(content.contains("Hello"), "{content:?}");
        assert!(!content.contains("<tool_call>"), "{content:?}");
        let Some(Event::Finish {
            reasoning_tokens, ..
        }) = events.last()
        else {
            panic!("a finish");
        };
        assert_eq!(*reasoning_tokens, 0);
    }
}

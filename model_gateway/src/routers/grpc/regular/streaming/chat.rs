//! The `/v1/chat/completions` stream state machine: one engine stream's
//! choices read to their stops, the parsers' turns, every chunk written
//! through the shared frame, the usage chunk and the settlement handed back
//! to the processor. Moved out of `streaming.rs` as is; the processor's
//! chat entry points stay there and drive it.

use std::{future::Future, sync::Arc, task::Poll, time::Instant};

use bytes::BytesMut;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopRules, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::ChatCompletionStreamResponse,
    common::{
        ChatLogProbs, FunctionCallDelta, StringOrArray, ToolCallDelta, ToolChoice, ToolChoiceValue,
        Usage,
    },
    profile::ProviderProfile,
};
use reasoning_parser::ReasoningParser;
use serde_json::Value;
use symphony::{EngineFinish, Events};
use tool_parser::ToolParser;
use tracing::debug;

use super::{
    super::symphony::{
        engine_finish, finished, reasoning_text_tokens, telemetry::Who, Delta, Reader,
    },
    assistant_choice, decoding_failed, forces_tool_call, ChatChunkFrame, ChatChunkWithUsage,
    ChatStreamUsage, StreamingProcessor,
};
use crate::{
    observability::metrics::{metrics_labels, LocalStopAbort, Metrics, StreamingMetricsParams},
    rate_limit::SharedReservationHandle,
    routers::{
        common::sse::{SseEncoder, SseSender},
        grpc::{
            common::response_formatting::{chat_usage, CompletionTokenTracker, TokenCounts},
            context,
            proto_wrapper::{
                ProtoGenerateComplete, ProtoGenerateStreamChunk, ProtoResponseVariant, ProtoStream,
            },
            spec::ChatResponseSpec,
            utils,
        },
    },
};

/// Text a choice's decoder released for one engine message, on its way to
/// the parsers and the client: the choice index, the text, the logprobs that
/// ride with it, and the tokens it stands for (held ones included).
type PendingText = (u32, String, Option<ChatLogProbs>, u32);

/// What one pass of the loop over the engine's end yields.
enum StreamEnd {
    /// Every choice's decoder and parsers have had their turn; the loop ends.
    Finished,
    /// This pass released nothing to emit.
    Nothing,
    /// Text to emit, and whether it is the parsers' EOF turn for its choice.
    Text(PendingText, bool),
}

/// What a chat stream keeps per choice (`n > 1` streams several): its
/// counters, flags, decoder, parsers and reader, in one entry of the
/// stream's vector indexed by the choice index, which the engines send
/// dense from zero. Nothing is hashed per engine message, and a one-choice
/// stream allocates one entry instead of a map per field.
#[derive(Default)]
struct ChoiceState {
    /// The role chunk went out.
    opened: bool,
    /// The finish reason once known: "stop" when the local stop decoder
    /// fired (pinned first, later engine output for the choice is ignored),
    /// else the engine's.
    finish_reason: Option<String>,
    /// The matched stop recorded with the finish reason.
    matched_stop: Option<Value>,
    /// The local stop decoder fired.
    stopped: bool,
    /// The engine's `Complete` came. A choice is closed by its Complete or
    /// by its local stop; once every expected choice is closed while a
    /// stopped one streams on without its Complete, the engine has outlived
    /// the client's stream (see [`ChatStream::engine_outlived_now`]).
    completed: bool,
    /// The engine streamed a chunk for the choice after its stop: it is
    /// generating past the stop, which only a stop it cannot see does (an
    /// engine that saw the stop ends the choice on that step and sends its
    /// Complete next).
    tailed: bool,
    /// The prompt count the choice's close reported: its `Complete`'s, or
    /// the chunk's the gateway's own stop matched in. `Some` once closed.
    prompt_tokens: Option<u32>,
    cached_tokens: u32,
    reasoning_tokens: u32,
    /// Tokens the gateway's reasoning parser routed to reasoning, and tokens
    /// of chunks that decoded to no text yet (counted with the chunk that
    /// releases their text).
    parsed_reasoning_tokens: u32,
    held_tokens: u32,
    spec_accepted: u32,
    spec_drafted: u32,
    /// The parsers, made on the choice's first text (not pooled: streaming
    /// parsers keep state); this stream's own, nothing else reaches them.
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    tool_parser: Option<Box<dyn ToolParser>>,
    has_tool_calls: bool,
    /// The choice's stop decoder, made on its first chunk.
    stop_decoder: Option<StopSequenceDecoder>,
    /// Symphony's reader when request building assigned the response a
    /// table (see `regular::symphony`): one per choice in place of both
    /// parsers, seeded with the prompt it replays. The role chunk, the stop
    /// decoder, the finish chunks and the usage are the same; only the
    /// reading of the tokens differs.
    reader: Option<Reader>,
}

/// One chat completion's SSE stream being read off the engine: what the
/// spec and the dispatch decide for the whole stream, the per-choice
/// trackers, decoders, parsers and readers the loop keeps, and the frame,
/// encoder and buffer the chunks leave through. Each phase of the stream is
/// a method; `StreamingProcessor::process_streaming_chunks_inner` runs them
/// in order.
pub(super) struct ChatStream<'a> {
    processor: &'a StreamingProcessor,
    spec: &'a ChatResponseSpec,
    /// Dispatch metadata, for consistent response fields.
    dispatch: &'a context::DispatchMetadata,
    tx: &'a SseSender,
    tokenizer: Arc<dyn Tokenizer>,
    /// The request's stop rules, built once; each choice's decoder copies them.
    stop_rules: StopRules,
    skip_special_tokens: bool,
    pd_timing: Option<context::PdTiming>,

    // Metrics timing
    start_time: Instant,
    first_token_time: Option<Instant>,

    /// DeepSeek always reports aggregate usage on the final finish chunk;
    /// the public dialect ends the stream with a usage-only chunk when
    /// `include_usage` asks for one. Under `include_usage` every chunk that
    /// carries no usage writes the `usage: null` placeholder, as the public
    /// API does (`emit_usage_null`).
    deepseek_usage: bool,
    include_usage: bool,
    emit_usage_null: bool,
    continuous_usage: Option<ChatStreamUsage>,

    /// Per-request effective parser names (model-card override -> configured),
    /// and whether each parser is there (checked once upfront, so the warning
    /// is logged once per request).
    reasoning_parser_name: Option<String>,
    tool_parser_name: Option<String>,
    reasoning_parser_available: bool,
    tool_parser_available: bool,
    /// Request building decided whether the rendered prompt ends inside the
    /// model's thinking block; the parsers then start in reasoning mode.
    thinking_override: bool,
    think_in_prefill: bool,
    /// Whether a JSON schema constraint was used (specific function or
    /// required mode), and whether it is the specific-function case (the
    /// model generates the parameters only, no name field; native formats
    /// include framing tokens the parser must handle).
    used_json_schema: bool,
    is_specific_function: bool,
    /// The first delta's shape: a forced tool call opens as the public API
    /// opens a tool-call stream; anything else as a text stream.
    opens_tool_call: bool,

    /// The per-choice state, one entry per choice the request expects
    /// (`spec.expected_choices`), indexed by the choice index the engine
    /// sends; an index past the end is a decoding failure.
    choices: Vec<ChoiceState>,
    /// The engine was streaming past a stop when every choice had closed:
    /// the loop stopped polling it, and the driver aborts it instead of
    /// marking it completed (see [`Self::release_engine`]).
    engine_outlived: bool,
    /// Tokens the engine streamed for choices after their local stop, read
    /// before the abort: the tail generated for nobody.
    tail_tokens: u64,
    completion_tokens: CompletionTokenTracker,
    /// The step's delta and the stream's event buffer, when Symphony reads
    /// the response: a step's events fill the buffer and borrow the delta's
    /// text, and the buffer is recycled once the step's chunks are out (the
    /// crate's `docs/engine.md`, the shape of a step's events). Taken out
    /// for the step and put back after it; a buffer lost to an error is a
    /// stream that ended.
    delta: Delta,
    spare: Events<'static>,
    /// After the engine's end: the (index, parser-EOF) turns still to take,
    /// one per pass of the loop.
    final_indices: Option<Vec<(u32, bool)>>,

    /// Reusable SSE encoder for the post-loop flush / tool / finish / usage
    /// chunks.
    sse_encoder: SseEncoder,
    /// The per-chunk SSE frames: the invariant head of every chunk of this
    /// stream is serialized once; per chunk only its choice (and usage) is
    /// serialized into `sse_buffer`, whose written bytes are split off for
    /// the channel instead of being copied.
    chunk_frame: ChatChunkFrame,
    sse_buffer: BytesMut,
    /// The frame carrying the stream's first token has left. Until then the
    /// buffer leaves before the engine is polled again, so the first token
    /// waits for nothing; after it, the engine messages ready at one poll
    /// share a frame (see [`Self::read_engine`]).
    first_token_out: bool,
}

impl<'a> ChatStream<'a> {
    /// Phase 1: what the spec, the dispatch and the tokenizer decide for the
    /// whole stream, and the empty per-choice state.
    pub(super) fn open(
        processor: &'a StreamingProcessor,
        dispatch: &'a context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        spec: &'a ChatResponseSpec,
        tx: &'a SseSender,
        pd_timing: Option<context::PdTiming>,
    ) -> Self {
        let start_time = Instant::now();
        let model = &dispatch.model;
        let (stop, stop_token_ids, skip_special_tokens, no_stop_trim, ignore_eos) = stop_params;
        let stop_rules = utils::stop_rules(
            &tokenizer,
            stop.as_ref(),
            stop_token_ids.as_ref(),
            no_stop_trim,
            ignore_eos,
        );

        let deepseek_usage = spec.provider == ProviderProfile::DeepSeek;
        let include_usage = spec
            .stream_options
            .as_ref()
            .is_some_and(|opts| opts.include_usage.unwrap_or(false));
        let emit_usage_null = include_usage;
        let mut continuous_usage = spec
            .stream_options
            .as_ref()
            .filter(|opts| {
                !deepseek_usage
                    && opts.include_usage.unwrap_or(false)
                    && opts.continuous_usage_stats.unwrap_or(false)
            })
            .map(|_| ChatStreamUsage::default());

        let reasoning_parser_name = processor
            .parser_resolver
            .reasoning_parser_for(spec.parser_names.as_ref(), model);
        let tool_parser_name = processor
            .parser_resolver
            .tool_parser_for(spec.parser_names.as_ref(), model);

        // Where a Symphony table reads the response its reader takes the thought, and no
        // legacy parser is built to be asked whether one is available.
        let reasoning_parser_available = spec.separate_reasoning
            && spec.symphony.is_none()
            && utils::check_reasoning_parser_availability(
                &processor.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );

        if let Some(usage) = &mut continuous_usage {
            usage.reasoning_parsed = reasoning_parser_available || spec.symphony.is_some();
        }

        let native_tool_format = processor
            .tool_parser_factory
            .registry()
            .uses_native_tool_format(tool_parser_name.as_deref());
        let used_json_schema = if native_tool_format {
            false
        } else {
            spec.tool_choice
                .as_ref()
                .is_some_and(ToolChoice::forces_call)
        };
        let is_specific_function =
            used_json_schema && matches!(spec.tool_choice, Some(ToolChoice::Function { .. }));

        let tool_parser_available = spec.tools.is_some()
            && spec.symphony.is_none()
            && utils::check_tool_parser_availability(
                &processor.tool_parser_factory,
                tool_parser_name.as_deref(),
                model,
            );

        if spec.separate_reasoning && spec.symphony.is_none() && !reasoning_parser_available {
            debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                model
            );
        }

        if spec.tools.is_some() && spec.symphony.is_none() && !tool_parser_available {
            debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                model
            );
        }

        let chunk_frame = ChatChunkFrame::new(
            &dispatch.request_id,
            model,
            dispatch.created,
            dispatch.weight_version.as_deref(),
        );

        Self {
            processor,
            spec,
            dispatch,
            tx,
            think_in_prefill: tokenizer.think_in_prefill(),
            tokenizer,
            stop_rules,
            skip_special_tokens,
            pd_timing,
            start_time,
            first_token_time: None,
            deepseek_usage,
            include_usage,
            emit_usage_null,
            continuous_usage,
            reasoning_parser_name,
            tool_parser_name,
            reasoning_parser_available,
            tool_parser_available,
            thinking_override: spec.reasoning_in_prefill,
            used_json_schema,
            is_specific_function,
            opens_tool_call: forces_tool_call(spec),
            choices: std::iter::repeat_with(ChoiceState::default)
                .take(spec.expected_choices as usize)
                .collect(),
            engine_outlived: false,
            tail_tokens: 0,
            completion_tokens: CompletionTokenTracker::new(),
            delta: Delta::default(),
            spare: Events::new(),
            final_indices: None,
            sse_encoder: SseEncoder::new(),
            chunk_frame,
            sse_buffer: BytesMut::with_capacity(4096),
            first_token_out: false,
        }
    }

    /// The usage a chunk carries when the client asked for continuous usage.
    pub(super) fn usage_snapshot(&self) -> Option<Usage> {
        self.continuous_usage
            .as_ref()
            .map(|tracker| tracker.snapshot(self.spec.unbilled_prompt_tokens))
    }

    /// The stream's first engine output, for the TTFT.
    fn note_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
            if let Some(timing) = &self.pd_timing {
                timing.record_first_decode_token(self.processor.backend_type, &self.dispatch.model);
            }
        }
    }

    /// A choice's first output opens it with the role chunk, as the engine's
    /// own server does, whether or not that output decodes to visible text (a
    /// special token the decoder skips, a partial UTF-8 sequence): a client
    /// timing its first chunk otherwise waits for the next step's text.
    async fn open_choice(&mut self, index: u32) -> Result<(), String> {
        if !self.choices[index as usize].opened {
            let usage = self.usage_snapshot();
            StreamingProcessor::send_role_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                self.opens_tool_call,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.choices[index as usize].opened = true;
        }
        Ok(())
    }

    /// The logprobs of an engine chunk whose text made no delta (a marker a
    /// parser consumed, text a tool parser holds, tokens that decoded to
    /// nothing) go out on a chunk with an empty delta, as the engine's own
    /// server sends a step whose text made none: every generated token's
    /// entry reaches the client once, in order. Nothing without entries.
    async fn push_bare_logprobs(
        &mut self,
        index: u32,
        logprobs: Option<ChatLogProbs>,
        usage: Option<&Usage>,
    ) -> Result<(), String> {
        if !utils::has_entries(logprobs.as_ref()) {
            return Ok(());
        }
        let mut choice = assistant_choice(index, None, None, None);
        choice.logprobs = logprobs;
        StreamingProcessor::push_chunk(
            self.tx,
            &mut self.sse_buffer,
            &self.chunk_frame,
            &choice,
            usage,
            self.emit_usage_null,
        )
        .await
        .map_err(|()| "Failed to send logprobs chunk".to_string())
    }

    /// The logprobs of an engine chunk, in the public API's shape.
    fn logprobs_of(&self, chunk: &ProtoGenerateStreamChunk) -> Option<ChatLogProbs> {
        chunk.output_logprobs().map(|proto_logprobs| {
            utils::convert_proto_to_openai_logprobs(
                proto_logprobs,
                &self.tokenizer,
                self.spec.top_logprobs,
            )
        })
    }

    /// Phase 2: the main loop. The engine stream is polled once in place
    /// before the chunks written so far leave as a frame, so the messages
    /// ready at one poll leave as one frame, after the first token's, which
    /// leaves alone; the text a message releases goes through the one
    /// emission below, so nothing reaches the client without being parsed.
    /// After the engine's end the loop keeps turning until every choice's
    /// decoder and parsers have flushed.
    pub(super) async fn read_engine(
        &mut self,
        grpc_stream: &mut ProtoStream,
    ) -> Result<(), utils::StreamFailure> {
        loop {
            // An engine that outlived the client's stream (every choice
            // closed, a locally stopped one streaming on past its stop) is
            // not polled again: the loop ends as it does at the engine's
            // end, and the driver aborts the engine stream.
            if self.final_indices.is_none() && !self.engine_outlived && self.engine_outlived_now() {
                self.engine_outlived = true;
            }
            let response = if self.final_indices.is_none() && !self.engine_outlived {
                // The packed read, kept inline here: the Messages and
                // completion loops read through
                // `StreamingProcessor::next_packed`, the same read as a
                // shared future, which on this stream cost about 85
                // instructions per delta (+1.1 percent of a 128-delta
                // request on the pair of record); this loop is the
                // most-travelled one, so its twin stays by measurement. It
                // yields once at the first token, as `next_packed` does for
                // the completion stream: this stream's first frame carries
                // the first token, and `DrivenBody` yields after its first
                // frame on its own (`next_packed` yields twice for the
                // Messages stream, whose first frame is `message_start`).
                // The engine is polled once in place before the chunks
                // written so far leave: a message it has ready joins the
                // frame being written, so the messages ready at one poll
                // leave as one frame; when it has none ready, the frame
                // leaves before the reader parks, so no text waits for the
                // engine. The frame carrying the first token leaves before
                // the engine is polled again, whatever is ready behind it:
                // the client's first token waits for nothing. The one poll
                // registers the reader's wake where the message will come
                // from; `next` resumes it after the flush.
                let item = {
                    let mut next = std::pin::pin!(grpc_stream.next());
                    let ready = if self.first_token_out {
                        std::future::poll_fn(|cx| Poll::Ready(next.as_mut().poll(cx))).await
                    } else {
                        Poll::Pending
                    };
                    match ready {
                        Poll::Ready(item) => item,
                        Poll::Pending => {
                            let frame_out = !self.sse_buffer.is_empty();
                            StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
                            if frame_out && !self.first_token_out && self.first_token_time.is_some()
                            {
                                self.first_token_out = true;
                                // The first token's frame reaches the
                                // connection before the engine is read
                                // again: the body driving this reading
                                // hands it out on this yield and writes it
                                // (see `DrivenBody`).
                                tokio::task::yield_now().await;
                            }
                            next.await
                        }
                    }
                };
                if item.is_none() && grpc_stream.client_left() {
                    // The client's departure ends the read here: text the
                    // decoder or a parser still holds can reach nobody, and
                    // the EOF flush would fail on the closed sender before
                    // the caller settles what was read.
                    return Ok(());
                }
                match item.transpose() {
                    Ok(response) => response,
                    Err(e) => {
                        // What was produced before the failure still goes out.
                        StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
                        return Err(utils::StreamFailure::upstream(
                            e,
                            self.completion_tokens.delivered(),
                        ));
                    }
                }
            } else {
                None
            };
            let mut final_chunk = response.is_none();

            // Text the stop decoder produced for this response, if any. Per-chunk
            // text and the end-of-stream flush both funnel into the shared emission
            // below, so neither can reach the client without being parsed.
            let pending = match response.map(|response| response.into_response()) {
                Some(ProtoResponseVariant::Chunk(chunk)) => self.read_chunk(chunk).await?,
                Some(ProtoResponseVariant::Complete(complete)) => {
                    self.read_complete(complete).await?
                }
                Some(ProtoResponseVariant::None) => continue,
                None => match self.read_end().await? {
                    StreamEnd::Finished => break,
                    StreamEnd::Nothing => continue,
                    StreamEnd::Text(pending, parser_eof) => {
                        final_chunk = parser_eof;
                        Some(pending)
                    }
                },
            };
            let Some(pending) = pending else {
                continue;
            };
            self.emit(pending, final_chunk).await?;
        }

        StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
        Ok(())
    }

    /// An engine chunk: the choice's tokens through its stop decoder (or
    /// Symphony's reader, which emits its events here) into the text the
    /// emission parses, with the tokens of chunks that decoded to nothing
    /// yet counted with it.
    async fn read_chunk(
        &mut self,
        chunk: ProtoGenerateStreamChunk,
    ) -> Result<Option<PendingText>, utils::StreamFailure> {
        // Track TTFT immediately on first chunk received from backend
        self.note_first_token();

        let index = chunk.index();

        // Once the local stop decoder has fired for an index, any further
        // engine chunk for it is the tail the engine generated for nobody
        // (an engine that saw the stop sends its Complete instead): counted,
        // not read; the request is aborted once every choice has closed,
        // until then another choice may still be streaming.
        let choice = utils::choice_mut(&mut self.choices, index)?;
        if choice.stopped {
            self.tail_tokens += chunk.token_ids().len() as u64;
            choice.tailed = true;
            return Ok(None);
        }

        self.completion_tokens.record_chunk(&chunk);
        if let Some(usage) = &mut self.continuous_usage {
            usage.record_chunk(&chunk);
        }

        self.open_choice(index).await?;

        // The choice's stop decoder, made on its first chunk as a copy of the
        // request's rules with a decode state of its own, and the fields the
        // step records into.
        let ChoiceState {
            stop_decoder,
            reader,
            finish_reason,
            matched_stop,
            stopped: choice_stopped,
            prompt_tokens,
            cached_tokens,
            reasoning_tokens,
            parsed_reasoning_tokens,
            ..
        } = &mut self.choices[index as usize];
        let stop_decoder = stop_decoder.get_or_insert_with(|| {
            StopSequenceDecoder::with_rules(
                self.tokenizer.clone(),
                self.stop_rules.clone(),
                self.skip_special_tokens,
            )
        });

        // Process tokens through stop decoder: Symphony's reader takes
        // them with their spans and gives events, the older path the text.
        // The step's events borrow the delta, so both leave the stream's
        // state for the step and come back once the chunks are out.
        let mut delta = std::mem::take(&mut self.delta);
        let symphony_events = match self.spec.symphony.as_ref() {
            Some(assignment) => {
                let mut events: Events<'_> = std::mem::take(&mut self.spare);
                let reader = match reader {
                    Some(reader) => reader,
                    None => reader.insert(
                        Reader::new(
                            assignment,
                            self.spec.tools.clone().unwrap_or_default(),
                            &mut events,
                        )?
                        .first_call_only(self.spec.first_tool_call_only),
                    ),
                };
                let stopped =
                    reader.read(stop_decoder, chunk.token_ids(), &mut delta, &mut events)?;
                Some((events, stopped))
            }
            None => None,
        };
        let (chunk_text, should_stop) = match &symphony_events {
            Some((_, stopped)) => (String::new(), *stopped),
            None => StreamingProcessor::process_chunk_tokens(stop_decoder, chunk.token_ids())?,
        };

        if should_stop {
            // Stop-decoder match takes precedence: pin "stop" even if
            // the backend's eventual Complete carries "length" (the
            // local stop sequence fired first). Any pre-stop text in
            // `chunk_text` is still emitted by `emit` before the finish
            // reason is flushed in Phase 4.
            if finish_reason.is_none() {
                *finish_reason = Some("stop".to_string());
                *matched_stop = stop_decoder
                    .matched_stop()
                    .map(|s| Value::String(s.to_string()));
            }
            *choice_stopped = true;
            // The choice closes here, as its Complete would close it: the
            // counts of the chunk the stop matched in are its counts (the
            // engine's own server counts a stopped request through that
            // step as well), and the tracker settles what the stream had
            // reached. What the engine generates past this step is the tail
            // the abort cuts, never the client's usage.
            *prompt_tokens = Some(chunk.prompt_tokens());
            *cached_tokens = chunk.cached_tokens();
            *reasoning_tokens = chunk.reasoning_tokens();
            self.completion_tokens.record_local_stop(index);
        }

        if let Some((mut events, _)) = symphony_events {
            // Recorded before the usage is snapshotted, so a continuous
            // usage counts the reasoning text it rides with.
            let chunk_reasoning_tokens = reasoning_text_tokens(events.as_slice());
            if chunk_reasoning_tokens > 0 {
                *parsed_reasoning_tokens += chunk_reasoning_tokens;
                if let Some(tracker) = &mut self.continuous_usage {
                    tracker.record_reasoning(index, chunk_reasoning_tokens);
                }
            }
            let usage = self.usage_snapshot();
            let choice_logprobs = self.logprobs_of(&chunk);
            StreamingProcessor::push_events(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                &mut events,
                choice_logprobs,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.spare = events.recycle();
            self.delta = delta;
            // The stop ended the choice's output: the reader ends it now,
            // with the finish "stop", as its Complete would have (the
            // decoder holds no tail past a stop).
            if should_stop {
                self.end_reader(index, None).await?;
            }
            return Ok(None);
        }
        self.delta = delta;

        if chunk_text.is_empty() {
            self.choices[index as usize].held_tokens += chunk.token_ids().len() as u32;
            // The chunk's logprobs still reach the client, on a chunk with
            // an empty delta, as the engine's own server sends a step whose
            // text made no delta: every generated token's entry goes out once.
            let logprobs = self.logprobs_of(&chunk);
            if utils::has_entries(logprobs.as_ref()) {
                let usage = self.usage_snapshot();
                self.push_bare_logprobs(index, logprobs, usage.as_ref())
                    .await?;
            }
            return Ok(None);
        }

        // Process logprobs if present
        let choice_logprobs = self.logprobs_of(&chunk);

        let chunk_tokens = chunk.token_ids().len() as u32
            + std::mem::take(&mut self.choices[index as usize].held_tokens);
        Ok(Some((index, chunk_text, choice_logprobs, chunk_tokens)))
    }

    /// A choice's Complete: its decoder finished, its counts and finish
    /// reason recorded (a local stop match keeps precedence), and, for a
    /// choice Symphony reads, its output ended here; otherwise the text the
    /// decoder still held goes to the emission. The Complete of a choice the
    /// engine streamed past its local stop adds only its speculative counts.
    async fn read_complete(
        &mut self,
        complete: ProtoGenerateComplete,
    ) -> Result<Option<PendingText>, utils::StreamFailure> {
        let index = complete.index();
        let request_id = &self.dispatch.request_id;

        // The Complete of a choice the engine generated past its local stop
        // (another choice kept the stream open) is the tail's end: the
        // choice closed at its stop with that step's counts, so only the
        // counts the stop could not know are taken from it.
        let choice = utils::choice_mut(&mut self.choices, index)?;
        if choice.tailed {
            choice.completed = true;
            choice.spec_accepted = complete.spec_accepted_tokens();
            choice.spec_drafted = complete.spec_draft_tokens();
            return Ok(None);
        }

        // Complete byte decoding before routing held text through the
        // same reasoning/tool parsers as ordinary chunks.
        let ChoiceState {
            stop_decoder,
            reader,
            stopped,
            finish_reason,
            matched_stop,
            completed,
            prompt_tokens,
            ..
        } = choice;
        let flushed = match stop_decoder {
            Some(decoder) => {
                let tail = decoder
                    .finish()
                    .map_err(|e| decoding_failed(reader.as_mut(), e))?;
                utils::warn_detokenize_recoveries(request_id, index, decoder);
                if decoder.is_stopped() {
                    *stopped = true;
                    *finish_reason = Some("stop".to_string());
                    *matched_stop = decoder.matched_stop().map(|s| Value::String(s.to_string()));
                }
                match tail {
                    SequenceDecoderOutput::Text(text)
                    | SequenceDecoderOutput::StoppedWithText(text)
                        if !text.is_empty() =>
                    {
                        Some(text)
                    }
                    _ => None,
                }
            }
            None => None,
        };

        // Store metadata
        *completed = true;
        *prompt_tokens = Some(complete.prompt_tokens());

        self.completion_tokens.record_complete(&complete);
        if let Some(usage) = &mut self.continuous_usage {
            usage.record_complete(&complete);
        }

        // A choice's first engine output may be its Complete (a
        // completion with no tokens: a finish that carries none
        // comes without a Chunk). It opens the choice as a first
        // Chunk does, the role chunk before the finish chunk, and
        // records the stream's first-token time; its usage, as a
        // first Chunk's, counts the output it rides on.
        self.note_first_token();
        self.open_choice(index).await?;

        let choice = &mut self.choices[index as usize];
        choice.cached_tokens = complete.cached_tokens();
        choice.reasoning_tokens = complete.reasoning_tokens();
        choice.spec_accepted = complete.spec_accepted_tokens();
        choice.spec_drafted = complete.spec_draft_tokens();

        // A local stop-decoder match already pinned "stop" for this
        // index; don't let the engine's finish reason overwrite it.
        if !choice.stopped {
            choice.finish_reason = Some(complete.finish_reason().to_string());
            choice.matched_stop = complete.matched_stop_json();
        }

        // Symphony ends the choice's output here: the decoder's last
        // text, then the engine's finish (a choice the gateway's own stop
        // ended was ended at its stop). The finish chunk is Phase 4's, with
        // the matched stop; the `Finish` event gives it the calls made, and
        // the usage the reasoning tokens. A choice whose first engine output
        // is its Complete has no reader yet: it gets one for its end, so an
        // output of no tokens is recorded as every output is.
        if self.spec.symphony.is_some() {
            if !choice.reader.as_ref().is_some_and(|reader| reader.ended()) {
                self.end_reader(index, flushed).await?;
            }
            return Ok(None);
        }

        // Don't break - continue reading all Complete messages for n>1
        // Text the decoder held for a partial stop match: the
        // tokens that produced it were counted as held.
        Ok(flushed.map(|text| {
            (
                index,
                text,
                None,
                std::mem::take(&mut self.choices[index as usize].held_tokens),
            )
        }))
    }

    /// A choice Symphony reads ends: the decoder's last text (`flushed`) is
    /// the reader's, then its output ends with the choice's finish reason
    /// (the engine's, or the "stop" a local stop pinned). The `Finish` event
    /// says what the Complete frame would have: the calls made, for the
    /// finish reason, and the reasoning tokens, for the usage. A choice
    /// without a reader yet (its first engine output is its Complete) gets
    /// one for its end, so an output of no tokens is recorded as every
    /// output is.
    async fn end_reader(
        &mut self,
        index: u32,
        flushed: Option<String>,
    ) -> Result<(), utils::StreamFailure> {
        let Some(assignment) = self.spec.symphony.as_ref() else {
            return Ok(());
        };
        let request_id = &self.dispatch.request_id;
        let mut delta = std::mem::take(&mut self.delta);
        let mut events: Events<'_> = std::mem::take(&mut self.spare);
        let ChoiceState {
            finish_reason,
            reader,
            has_tool_calls,
            parsed_reasoning_tokens,
            ..
        } = &mut self.choices[index as usize];
        let reason = finish_reason.as_deref().unwrap_or("stop");
        let reader = match reader {
            Some(reader) => reader,
            None => reader.insert(
                Reader::new(
                    assignment,
                    self.spec.tools.clone().unwrap_or_default(),
                    &mut events,
                )?
                .first_call_only(self.spec.first_tool_call_only),
            ),
        };
        if let Some(text) = flushed {
            reader.tail(text, &mut delta, &mut events)?;
        }
        let who = Who {
            request_id,
            model: &self.dispatch.model,
            choice: index,
        };
        reader.end(engine_finish(reason), who, &mut events)?;
        if let Some(finished) = finished(events.as_slice()) {
            *has_tool_calls = finished.tool_calls > 0;
            *parsed_reasoning_tokens = finished.reasoning_tokens;
        }
        let usage = self.usage_snapshot();
        StreamingProcessor::push_events(
            self.tx,
            &mut self.sse_buffer,
            &self.chunk_frame,
            index,
            &mut events,
            None,
            usage.as_ref(),
            self.emit_usage_null,
        )
        .await?;
        self.spare = events.recycle();
        self.delta = delta;
        Ok(())
    }

    /// Whether the engine has outlived the client's stream: every expected
    /// choice is closed (its Complete came or its local stop fired) while a
    /// stopped choice without its Complete has streamed on past its stop,
    /// so the engine is generating a tail nobody receives. A stopped choice
    /// whose next message has not come yet is waited for: an engine that
    /// saw the stop (an EOS id, a stop the request also gave it) ends the
    /// choice on that step and its Complete is the next message, so nothing
    /// is aborted and the Complete is read as always; a chunk instead is
    /// the tail. Nothing to look at while no stop has fired.
    fn engine_outlived_now(&self) -> bool {
        let mut open_stops = 0u32;
        let mut completed = 0u32;
        let mut tailed = false;
        for choice in &self.choices {
            if choice.completed {
                completed += 1;
            } else if choice.stopped {
                open_stops += 1;
                tailed |= choice.tailed;
            }
        }
        tailed && completed + open_stops >= self.spec.expected_choices
    }

    /// The engine stream, released before the client's finish is written.
    /// An engine the gateway's own stop outlived is aborted now: dropping
    /// the stream un-marked is the abort path a client disconnect takes (the
    /// abort RPC or frame, and the worker's end of the stream drained off
    /// this task), so the slot goes back to the engine's batch within a step
    /// instead of running to `max_tokens` for nobody. An engine that ended
    /// on its own is marked completed, so that its drop sends no abort.
    pub(super) fn release_engine(&self, mut grpc_stream: ProtoStream, endpoint: &'static str) {
        if !self.engine_outlived {
            grpc_stream.mark_completed();
            return;
        }
        Metrics::record_local_stop_abort(LocalStopAbort {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: &self.dispatch.model,
            endpoint,
            tail_tokens: self.tail_tokens,
        });
        drop(grpc_stream);
    }

    /// The engine's end, one pass per call: first the choices Symphony reads
    /// whose Complete never came are ended as aborts, then every remaining
    /// choice's decoder and parsers get their turn, one (index, EOF) turn
    /// per pass, until none is left.
    async fn read_end(&mut self) -> Result<StreamEnd, utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;

        // A choice Symphony reads whose Complete never came: the decoder's
        // last text is the reader's, then the output ends as an abort so
        // what the engine holds goes out, as the older parsers' flush
        // below sends theirs. Those indices are not the loop's below.
        let unended: Vec<u32> = (0..self.choices.len() as u32)
            .filter(|index| {
                self.choices[*index as usize]
                    .reader
                    .as_ref()
                    .is_some_and(|reader| !reader.ended())
            })
            .collect();
        for index in unended {
            let ChoiceState {
                reader,
                stop_decoder,
                stopped,
                finish_reason,
                matched_stop,
                has_tool_calls,
                parsed_reasoning_tokens,
                ..
            } = &mut self.choices[index as usize];
            let Some(reader) = reader else {
                continue;
            };
            let mut delta = std::mem::take(&mut self.delta);
            let mut events: Events<'_> = std::mem::take(&mut self.spare);
            if let Some(decoder) = stop_decoder {
                let tail = decoder.finish().map_err(|e| reader.failed(e.to_string()))?;
                utils::warn_detokenize_recoveries(request_id, index, decoder);
                if decoder.is_stopped() {
                    *stopped = true;
                    *finish_reason = Some("stop".to_string());
                    *matched_stop = decoder.matched_stop().map(|s| Value::String(s.to_string()));
                }
                if let SequenceDecoderOutput::Text(text)
                | SequenceDecoderOutput::StoppedWithText(text) = tail
                {
                    reader.tail(text, &mut delta, &mut events)?;
                }
            }
            let who = Who {
                request_id,
                model: &self.dispatch.model,
                choice: index,
            };
            reader.end(EngineFinish::Abort, who, &mut events)?;
            // The `Finish` event says what the Complete frame would have:
            // the calls made, for the finish reason, and the reasoning
            // tokens, for the usage.
            if let Some(finished) = finished(events.as_slice()) {
                *has_tool_calls = finished.tool_calls > 0;
                *parsed_reasoning_tokens = finished.reasoning_tokens;
            }
            let usage = self.usage_snapshot();
            StreamingProcessor::push_events(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                &mut events,
                None,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.spare = events.recycle();
            self.delta = delta;
        }

        // Finish decoder text before sending EOF to the parsers. A
        // clean transport EOF does not supply Complete metadata.
        let choices = &self.choices;
        let indices = self.final_indices.get_or_insert_with(|| {
            (0..choices.len() as u32)
                .filter(|index| {
                    let choice = &choices[*index as usize];
                    (choice.stop_decoder.is_some() || choice.reasoning_parser.is_some())
                        && choice.reader.is_none()
                })
                .flat_map(|index| [(index, true), (index, false)])
                .collect()
        });
        let Some((index, parser_eof)) = indices.pop() else {
            return Ok(StreamEnd::Finished);
        };
        let text = if parser_eof {
            String::new()
        } else {
            let ChoiceState {
                stop_decoder,
                stopped,
                finish_reason,
                matched_stop,
                ..
            } = &mut self.choices[index as usize];
            match stop_decoder {
                Some(decoder) => {
                    let tail = decoder.finish().map_err(|e| e.to_string())?;
                    utils::warn_detokenize_recoveries(request_id, index, decoder);
                    if decoder.is_stopped() {
                        *stopped = true;
                        *finish_reason = Some("stop".to_string());
                        *matched_stop =
                            decoder.matched_stop().map(|s| Value::String(s.to_string()));
                    }
                    match tail {
                        SequenceDecoderOutput::Text(text)
                        | SequenceDecoderOutput::StoppedWithText(text) => text,
                        _ => String::new(),
                    }
                }
                None => String::new(),
            }
        };
        if !parser_eof && text.is_empty() {
            return Ok(StreamEnd::Nothing);
        }
        Ok(StreamEnd::Text(
            (
                index,
                text,
                None,
                std::mem::take(&mut self.choices[index as usize].held_tokens),
            ),
            parser_eof,
        ))
    }

    /// The one emission: text a message released goes through the reasoning
    /// parser, then the tool parsers, then out as content, each as the chunks
    /// the public API streams; `final_chunk` is the parsers' EOF turn.
    async fn emit(
        &mut self,
        pending: PendingText,
        final_chunk: bool,
    ) -> Result<(), utils::StreamFailure> {
        let (index, text, mut choice_logprobs, chunk_tokens) = pending;
        let model = &self.dispatch.model;
        let mut usage = self.usage_snapshot();

        // Text that reaches here without an engine chunk before it (a
        // decoder flush on the terminal frame) still opens its choice.
        self.open_choice(index).await?;

        // Calculate delta
        let mut delta = text;

        // Reasoning content handling
        let in_reasoning = if self.spec.separate_reasoning && self.reasoning_parser_available {
            let (normal_text, reasoning_choice, in_reasoning, chunk_reasoning_tokens) =
                self.processor.process_reasoning_stream(
                    (!final_chunk).then_some(delta.as_str()),
                    chunk_tokens,
                    index,
                    &mut self.choices[index as usize].reasoning_parser,
                    self.thinking_override,
                    self.think_in_prefill,
                    self.reasoning_parser_name.as_deref(),
                    model,
                );
            // Recorded before this frame's usage is snapshotted again, so
            // a continuous usage counts the reasoning text it rides with,
            // and the content or tool chunks that follow in the frame
            // agree with it.
            if chunk_reasoning_tokens > 0 {
                self.choices[index as usize].parsed_reasoning_tokens += chunk_reasoning_tokens;
                if let Some(tracker) = &mut self.continuous_usage {
                    tracker.record_reasoning(index, chunk_reasoning_tokens);
                    usage = Some(tracker.snapshot(self.spec.unbilled_prompt_tokens));
                }
            }
            if let Some(mut choice) = reasoning_choice {
                // The chunk's token logprobs ride on its reasoning delta when
                // none of its text is content, as on a content delta.
                if normal_text.is_empty() {
                    choice.logprobs = choice_logprobs.take();
                }
                StreamingProcessor::push_chunk(
                    self.tx,
                    &mut self.sse_buffer,
                    &self.chunk_frame,
                    &choice,
                    usage.as_ref(),
                    self.emit_usage_null,
                )
                .await
                .map_err(|()| "Failed to send reasoning chunk".to_string())?;
            }
            delta = normal_text;
            in_reasoning
        } else {
            false
        };

        if final_chunk && delta.is_empty() {
            return Ok(());
        }

        // Tool call handling
        let tool_choice_enabled = !matches!(
            self.spec.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if let Some(tools_ref) = self.spec.tools.as_ref() {
            if !in_reasoning
                && tool_choice_enabled
                && (self.tool_parser_available || self.used_json_schema)
            {
                let ChoiceState {
                    tool_parser,
                    has_tool_calls,
                    ..
                } = &mut self.choices[index as usize];
                let mut tool_choices = if self.is_specific_function {
                    // Handle specific function case - emit tool call deltas with arguments
                    StreamingProcessor::process_specific_function_stream(
                        &delta,
                        index,
                        has_tool_calls,
                        self.spec.tool_choice.as_ref(),
                        model,
                        self.spec.history_tool_calls_count,
                    )
                } else {
                    // Use incremental parser for regular/required modes
                    self.processor
                        .process_tool_calls_stream(
                            &delta,
                            index,
                            tool_parser,
                            has_tool_calls,
                            tools_ref,
                            self.tool_parser_name.as_deref(),
                            model,
                            self.spec.history_tool_calls_count,
                            self.used_json_schema,
                            self.spec.first_tool_call_only,
                        )
                        .await
                };

                // The chunk's logprobs ride on the first of its tool-call
                // deltas, as on a content delta; a chunk the parser holds
                // whole goes out as an empty delta with them.
                if let Some(first) = tool_choices.first_mut() {
                    first.logprobs = choice_logprobs.take();
                }
                for choice in &tool_choices {
                    StreamingProcessor::push_chunk(
                        self.tx,
                        &mut self.sse_buffer,
                        &self.chunk_frame,
                        choice,
                        usage.as_ref(),
                        self.emit_usage_null,
                    )
                    .await
                    .map_err(|()| "Failed to send tool call chunk".to_string())?;
                }
                self.push_bare_logprobs(index, choice_logprobs, usage.as_ref())
                    .await?;

                // Always skip regular content when tool parsing is active
                // Parser either emitted chunks or buffered content
                return Ok(());
            }
        }

        // Regular content emission
        if !delta.is_empty() {
            let mut choice = assistant_choice(index, Some(delta), None, None);
            choice.logprobs = choice_logprobs.take();
            StreamingProcessor::push_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                &choice,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await
            .map_err(|()| "Failed to send content chunk".to_string())?;
        }
        // A chunk whose text made no delta (a marker the reasoning parser
        // consumed) still carries its logprobs, on a chunk with an empty delta.
        self.push_bare_logprobs(index, choice_logprobs, usage.as_ref())
            .await?;
        Ok(())
    }

    /// Phase 3: the end-of-stream parser flush: first any text still buffered
    /// as a prospective tool call that never materialized (dropping it
    /// produced fully-empty streams), then any parsed-but-unstreamed tool
    /// arguments.
    pub(super) async fn flush_parsers(
        &mut self,
        usage: Option<&Usage>,
    ) -> Result<(), utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;
        let model = &self.dispatch.model;
        let created = self.dispatch.created;
        let system_fingerprint = self.dispatch.weight_version.as_deref();

        for (index, choice) in self.choices.iter_mut().enumerate() {
            let Some(parser_guard) = &mut choice.tool_parser else {
                continue;
            };
            let index = index as u32;
            let leftover_text = parser_guard.take_unstreamed_normal_text();
            if !leftover_text.is_empty() {
                let content_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_content_with_logprobs(index, leftover_text, None)
                    .maybe_system_fingerprint(system_fingerprint)
                    .maybe_usage(usage.cloned())
                    .build();

                let sse_chunk = self
                    .sse_encoder
                    .encode_data(&ChatChunkWithUsage::new(
                        &content_chunk,
                        self.emit_usage_null,
                    ))
                    .map_err(|e| format!("Failed to serialize content chunk: {e}"))?;
                self.tx
                    .send(Ok(sse_chunk))
                    .await
                    .map_err(|_| "Failed to send flushed content chunk".to_string())?;
            }

            if let Some(mut unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                if self.spec.first_tool_call_only {
                    utils::keep_first_tool_call_items(&mut unstreamed_items);
                }
                for tool_call_item in unstreamed_items {
                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: if tool_call_item.parameters.is_empty() {
                                None
                            } else {
                                Some(tool_call_item.parameters)
                            },
                        }),
                    };

                    let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_call_delta(index, tool_call_delta)
                        .maybe_system_fingerprint(system_fingerprint)
                        .maybe_usage(usage.cloned())
                        .build();

                    let sse_chunk = self
                        .sse_encoder
                        .encode_data(&ChatChunkWithUsage::new(&tool_chunk, self.emit_usage_null))
                        .map_err(|e| format!("Failed to serialize tool chunk: {e}"))?;
                    self.tx
                        .send(Ok(sse_chunk))
                        .await
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }
        }
        Ok(())
    }

    /// How many choices have closed: each has the prompt count its close
    /// reported, from its `Complete` or from the chunk the gateway's own stop
    /// matched in.
    fn closed_choices(&self) -> u32 {
        self.choices
            .iter()
            .filter(|choice| choice.prompt_tokens.is_some())
            .count() as u32
    }

    /// The counters once, for the usage chunk, the settlement and the
    /// metrics: every choice shares one prompt, so prompt/cache counts take
    /// max; completion and reasoning counts sum across choices.
    fn counts(&self) -> TokenCounts {
        let choices = &self.choices;
        TokenCounts {
            prompt: choices
                .iter()
                .filter_map(|choice| choice.prompt_tokens)
                .max()
                .unwrap_or(0),
            completion: self.completion_tokens.total(),
            cached: choices
                .iter()
                .map(|choice| choice.cached_tokens)
                .max()
                .unwrap_or(0),
            reasoning: choices.iter().map(|choice| choice.reasoning_tokens).sum(),
            spec_accepted: choices.iter().map(|choice| choice.spec_accepted).sum(),
            spec_drafted: choices.iter().map(|choice| choice.spec_drafted).sum(),
        }
    }

    /// The stream's usage when the client asked for one, from
    /// [`Self::counts`]. Built by the non-streamed answer's builder: the
    /// usage chunk carries the usage object that answer carries.
    pub(super) fn final_usage(&self) -> Option<Usage> {
        (self.deepseek_usage || self.include_usage).then(|| {
            chat_usage(
                self.counts().usage(),
                (self.reasoning_parser_available || self.spec.symphony.is_some()).then(|| {
                    self.choices
                        .iter()
                        .map(|choice| choice.parsed_reasoning_tokens)
                        .sum()
                }),
                self.spec.unbilled_prompt_tokens,
            )
        })
    }

    /// Phase 4: the finish reason chunks. Partial counters are not advertised
    /// as a final aggregate if the backend omitted a choice's Complete frame.
    pub(super) async fn send_finish_chunks(
        &mut self,
        final_usage: Option<&Usage>,
        usage: Option<&Usage>,
    ) -> Result<(), utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;
        let model = &self.dispatch.model;
        let created = self.dispatch.created;
        let system_fingerprint = self.dispatch.weight_version.as_deref();

        let complete_usage = self.closed_choices() >= self.spec.expected_choices;
        // One finish chunk per finished choice, in index order.
        let finished = self
            .choices
            .iter()
            .filter(|choice| choice.finish_reason.is_some())
            .count();
        let mut position = 0;
        for (index, choice) in self.choices.iter().enumerate() {
            let Some(finish_reason) = choice.finish_reason.as_deref() else {
                continue;
            };
            let index = index as u32;
            position += 1;
            let final_finish_reason = if choice.has_tool_calls && finish_reason == "stop" {
                "tool_calls".to_string()
            } else {
                finish_reason.to_string()
            };

            let matched_stop_value = choice.matched_stop.clone();

            let finish_usage = if self.deepseek_usage && complete_usage && position == finished {
                final_usage.cloned()
            } else {
                usage.cloned()
            };
            let finish_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_finish_reason(index, final_finish_reason, matched_stop_value)
                .maybe_system_fingerprint(system_fingerprint)
                .maybe_usage(finish_usage)
                .build();

            let sse_chunk = self
                .sse_encoder
                .encode_data(&ChatChunkWithUsage::new(
                    &finish_chunk,
                    self.emit_usage_null,
                ))
                .map_err(|e| format!("Failed to serialize finish chunk: {e}"))?;
            self.tx
                .send(Ok(sse_chunk))
                .await
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }
        Ok(())
    }

    /// Phase 5: the OpenAI dialect keeps its opt-in, usage-only chunk.
    pub(super) async fn send_usage_chunk(
        &mut self,
        final_usage: Option<Usage>,
    ) -> Result<(), utils::StreamFailure> {
        if !self.deepseek_usage {
            if let Some(final_usage) = final_usage {
                let usage_chunk = ChatCompletionStreamResponse::builder(
                    &self.dispatch.request_id,
                    &self.dispatch.model,
                )
                .created(self.dispatch.created)
                .usage(final_usage)
                .maybe_system_fingerprint(self.dispatch.weight_version.as_deref())
                .build();
                let sse_chunk = self
                    .sse_encoder
                    .encode_data(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {e}"))?;
                self.tx
                    .send(Ok(sse_chunk))
                    .await
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }
        Ok(())
    }

    /// The end: the reservation settled with what the stream used
    /// ([`Self::counts`], the engine's counts as they are), and the
    /// streaming metrics recorded.
    /// `client_left` says the read ended with the client's leaving, not with the engine's end:
    /// the reservation is still settled, but no terminal router metric is recorded for it, as
    /// the HTTP router has none for a disconnect (neither a completed request nor a failure).
    pub(super) async fn settle_and_record(
        self,
        reservation: Option<Arc<SharedReservationHandle>>,
        client_left: bool,
    ) {
        let model = &self.dispatch.model;
        let counts = self.counts();
        let (total_prompt, total_completion) = (counts.prompt, counts.completion);
        if let Some(handle) = reservation {
            // `prompt_tokens` has one entry per closed index: from its
            // `Complete`, or from the chunk the gateway's own stop matched
            // in. A clean EOF that didn't close every expected `n>1` choice
            // has only partial usage -- settling with that would understate
            // the real cost. Keep the reserved amount as final instead.
            let expected_choices = self.spec.expected_choices;
            if self.closed_choices() < expected_choices {
                handle.close_reserved_only().await;
            } else {
                handle.settle_success(counts.settlement()).await;
            }
        }
        if client_left {
            return;
        }
        // In PD mode this task only starts once the decode stream exists --
        // for sequential PD that is after prefill completes, with the first
        // decode chunk already buffered -- so `start_time` would report a
        // near-zero TTFT. Anchor at the prefill dispatch instant instead;
        // generation duration must share the base or TPOT underflows.
        let stream_base = self
            .pd_timing
            .as_ref()
            .map(|timing| timing.prefill_start)
            .unwrap_or(self.start_time);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: self.first_token_time.map(|t| t.duration_since(stream_base)),
            generation_duration: stream_base.elapsed(),
            input_tokens: Some(total_prompt as u64),
            cached_tokens: Some(u64::from(counts.cached)),
            output_tokens: total_completion as u64,
        });
    }
}

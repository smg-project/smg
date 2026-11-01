//! The `/v1/completions` stream state machine: one engine stream's choices
//! read to their stops, every chunk written through the shared frame, the
//! totals handed back to the processor for the usage chunk and the metrics
//! record. Moved out of `streaming.rs` as is; the processor's completion
//! entry points stay there and drive it.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use bytes::BytesMut;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{common::StringOrArray, completion::CompletionStreamChoice};

use super::{CompletionChunkFrame, StreamingProcessor};
use crate::{
    observability::metrics::{metrics_labels, LocalStopAbort, Metrics},
    routers::{
        common::sse::SseSender,
        grpc::{
            common::response_formatting::CompletionTokenTracker,
            context,
            proto_wrapper::{
                ProtoGenerateComplete, ProtoGenerateStreamChunk, ProtoResponseVariant, ProtoStream,
            },
            spec::CompletionResponseSpec,
            utils,
        },
    },
    worker::PrefillLoadGuard,
};

/// One backend stream of a `/v1/completions` request. Batched requests fan
/// out into several, each remapped by a prompt-major choice-index offset.
pub(super) enum CompletionStreamUnit {
    Single(ProtoStream),
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        /// PD timing context, for honest PD TTFT (prefill start to first
        /// decode token).
        pd_timing: context::PdTiming,
    },
}

impl CompletionStreamUnit {
    /// The prefill dispatch instant of a disaggregated unit: the base its
    /// TTFT and generation duration are measured from.
    pub(super) fn prefill_start(&self) -> Option<Instant> {
        match self {
            Self::Single(_) => None,
            Self::PrefillDecode { pd_timing, .. } => Some(pd_timing.prefill_start),
        }
    }
}

/// Per-stream token/timing totals returned by the completion chunk loop; the
/// coordinator aggregates them into the single usage chunk and metrics record.
pub(super) struct CompletionStreamOutcome {
    pub(super) prompt_tokens: u32,
    pub(super) cached_tokens: u32,
    pub(super) reasoning_tokens: u32,
    pub(super) spec_accepted_tokens: u32,
    pub(super) spec_draft_tokens: u32,
    pub(super) completion_tokens: u32,
    pub(super) first_token_time: Option<Instant>,
    /// Whether *every* expected `n>1` choice in this unit received a
    /// `Complete` message (the only source of authoritative usage) -- a
    /// clean EOF partway through leaves this `false` even if some choices
    /// did complete, so a partial result is never mistaken for full usage.
    pub(super) saw_complete: bool,
}

/// One completion stream unit being read off the engine (one prompt of a
/// possibly batched request, its choices at `index_offset` and up): what the
/// spec and the dispatch decide for the whole unit, the per-choice decoders
/// and flags, the frame and buffer the chunks leave through, and the totals
/// the coordinator's usage chunk and metrics record read. Each phase is a
/// method; `StreamingProcessor::process_completion_streaming_chunks` runs
/// them in order.
pub(super) struct CompletionStream<'a> {
    /// Dispatch metadata, for consistent response fields.
    dispatch: &'a context::DispatchMetadata,
    tokenizer: Arc<dyn Tokenizer>,
    stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
    spec: &'a CompletionResponseSpec,
    prompt_text: &'a str,
    index_offset: u32,
    tx: &'a SseSender,
    pd_timing: Option<context::PdTiming>,
    /// The processor's backend label, for the PD first-token pair and the
    /// local-stop abort record.
    backend_type: &'static str,

    first_token_time: Option<Instant>,

    // Per-choice state (per index, for n>1 support)
    stop_decoders: HashMap<u32, StopSequenceDecoder>,
    is_firsts: HashMap<u32, bool>,
    /// Indices whose local stop decoder fired: "stop" is emitted even if the
    /// backend's eventual Complete carries "length" (the local stop sequence
    /// fired first), and later engine output for the index is ignored.
    stopped_indices: HashSet<u32>,
    /// Indices that received a `Complete` message -- tracked separately
    /// from `reasoning_tokens` (which exists for a different purpose and
    /// only incidentally gets exactly one insert per completed index
    /// today) so the completeness gate doesn't silently break if that
    /// insert behavior ever changes. A choice is closed by its Complete or
    /// by its local stop (whose usage is as settled, see `read_chunk`).
    completed_indices: HashSet<u32>,
    /// Stopped indices the engine streamed a chunk for after their stop: it
    /// is generating past the stop, which only a stop it cannot see does (an
    /// engine that saw the stop ends the choice on that step and sends its
    /// Complete next).
    tailed_indices: HashSet<u32>,
    /// The engine was streaming past a stop when every choice had closed:
    /// the loop stopped polling it, and the driver aborts it instead of
    /// marking it completed (see [`Self::release_engine`]).
    engine_outlived: bool,
    /// Tokens the engine streamed for choices after their local stop, read
    /// before the abort: the tail generated for nobody.
    tail_tokens: u64,

    /// The per-chunk SSE frames: the bytes around a chunk's choice are the
    /// same for every chunk of this stream and are serialized once; per
    /// chunk only its choice is serialized into `sse_buffer`, whose written
    /// bytes are split off for the channel instead of being copied.
    frame: CompletionChunkFrame,
    sse_buffer: BytesMut,

    /// For n>1, each index shares the same prompt -- the max across Complete
    /// messages rather than the sum (same prompt tokenized once, not per-choice).
    total_prompt: u32,
    total_cached: u32,
    reasoning_tokens: HashMap<u32, u32>,
    spec_accepted: HashMap<u32, u32>,
    spec_drafted: HashMap<u32, u32>,
    total_completion: CompletionTokenTracker,
}

/// A completion chunk's choice: `text` at `index`, with its finish reason.
fn completion_choice(
    index: u32,
    text: String,
    finish_reason: Option<String>,
) -> CompletionStreamChoice {
    CompletionStreamChoice {
        text,
        index,
        logprobs: None,
        finish_reason,
    }
}

impl<'a> CompletionStream<'a> {
    /// What the spec and the dispatch decide for the whole unit, the frame,
    /// and the empty per-choice state and totals. `pd_timing` is `Some` only
    /// in PD mode.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn open(
        dispatch: &'a context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        spec: &'a CompletionResponseSpec,
        prompt_text: &'a str,
        index_offset: u32,
        tx: &'a SseSender,
        pd_timing: Option<context::PdTiming>,
        backend_type: &'static str,
    ) -> Self {
        let frame = CompletionChunkFrame::new(
            &dispatch.request_id,
            &dispatch.model,
            dispatch.created,
            dispatch.weight_version.as_deref(),
        );
        Self {
            dispatch,
            tokenizer,
            stop_params,
            spec,
            prompt_text,
            index_offset,
            tx,
            pd_timing,
            backend_type,
            first_token_time: None,
            stop_decoders: HashMap::new(),
            is_firsts: HashMap::new(),
            stopped_indices: HashSet::new(),
            completed_indices: HashSet::new(),
            tailed_indices: HashSet::new(),
            engine_outlived: false,
            tail_tokens: 0,
            frame,
            sse_buffer: BytesMut::with_capacity(4096),
            total_prompt: 0,
            total_cached: 0,
            reasoning_tokens: HashMap::new(),
            spec_accepted: HashMap::new(),
            spec_drafted: HashMap::new(),
            total_completion: CompletionTokenTracker::new(),
        }
    }

    /// The loop over the engine's messages, each reaching the client as its
    /// own chunks as soon as it is processed.
    pub(super) async fn read_engine(
        &mut self,
        grpc_stream: &mut ProtoStream,
    ) -> Result<(), utils::StreamFailure> {
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| {
                utils::StreamFailure::upstream(e, self.total_completion.delivered())
            })?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => self.read_chunk(chunk).await?,
                ProtoResponseVariant::Complete(complete) => self.read_complete(complete).await?,
                ProtoResponseVariant::None => continue,
            }
            // An engine that outlived the unit (every choice closed, a
            // locally stopped one streaming on past its stop) is not polled
            // again: the driver aborts the engine stream.
            if self.engine_outlived_now() {
                self.engine_outlived = true;
                break;
            }
        }
        Ok(())
    }

    /// An engine chunk: the choice's tokens through its stop decoder (made on
    /// its first chunk), the prompt echoed in front of the first text, the
    /// text out, and on a local stop the suffix and the "stop" finish.
    async fn read_chunk(
        &mut self,
        chunk: ProtoGenerateStreamChunk,
    ) -> Result<(), utils::StreamFailure> {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
            if let Some(timing) = &self.pd_timing {
                timing.record_first_decode_token(self.backend_type, &self.dispatch.model);
            }
        }

        let index = self.index_offset + chunk.index();

        // A choice's engine chunk after its local stop is the tail the
        // engine generated for nobody (an engine that saw the stop sends
        // its Complete instead): counted, not read; the unit is aborted
        // once every choice has closed, until then another choice may
        // still be streaming.
        if self.stopped_indices.contains(&index) {
            self.tail_tokens += chunk.token_ids().len() as u64;
            self.tailed_indices.insert(index);
            return Ok(());
        }

        let is_first = self.is_firsts.entry(index).or_insert(true);
        self.total_completion.record_chunk(&chunk);

        let stop_decoder = self.stop_decoders.entry(index).or_insert_with(|| {
            let (ref stop, ref stop_token_ids, skip_special_tokens, no_stop_trim, ignore_eos) =
                self.stop_params;
            utils::create_stop_decoder(
                &self.tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                skip_special_tokens,
                no_stop_trim,
                ignore_eos,
            )
        });

        let (mut text, stopped) =
            StreamingProcessor::process_chunk_tokens(stop_decoder, chunk.token_ids())?;

        if *is_first {
            if self.spec.echo {
                text.insert_str(0, self.prompt_text);
            }
            *is_first = false;
        }

        if !text.is_empty() {
            StreamingProcessor::send_completion_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.frame,
                &completion_choice(index, text, None),
            )
            .await?;
        }

        if stopped {
            // Stop-decoder match takes precedence: emit "stop" even if
            // the backend's eventual Complete carries "length". This is
            // intentional — the local stop sequence fired first.
            self.stopped_indices.insert(index);
            // The choice closes here, as its Complete would close it: the
            // counts of the chunk the stop matched in are its counts (the
            // engine's own server counts a stopped request through that
            // step as well), and the tracker settles what the stream had
            // reached. What the engine generates past this step is the tail
            // the abort cuts, never the client's usage.
            self.total_prompt = self.total_prompt.max(chunk.prompt_tokens());
            self.total_cached = self.total_cached.max(chunk.cached_tokens());
            self.reasoning_tokens
                .insert(index, chunk.reasoning_tokens());
            self.total_completion.record_local_stop(index);

            if let Some(sfx) = self.spec.suffix.as_deref() {
                StreamingProcessor::send_completion_chunk(
                    self.tx,
                    &mut self.sse_buffer,
                    &self.frame,
                    &completion_choice(index, sfx.to_string(), None),
                )
                .await?;
            }

            StreamingProcessor::send_completion_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.frame,
                &completion_choice(index, String::new(), Some("stop".to_string())),
            )
            .await?;
        }
        Ok(())
    }

    /// The engine's `Complete` for a choice: its counts into the totals
    /// (only its speculative counts for a choice the engine streamed past
    /// its local stop); unless a local stop closed the choice already, the
    /// echo if no chunk carried it, the decoder's tail, the suffix and the
    /// finish chunk.
    async fn read_complete(
        &mut self,
        complete: ProtoGenerateComplete,
    ) -> Result<(), utils::StreamFailure> {
        let index = self.index_offset + complete.index();
        self.completed_indices.insert(index);
        self.spec_accepted
            .insert(index, complete.spec_accepted_tokens());
        self.spec_drafted
            .insert(index, complete.spec_draft_tokens());
        // The Complete of a choice the engine generated past its local stop
        // (another choice of the unit kept the stream open) is the tail's
        // end: the choice closed at its stop with that step's counts, and
        // only the counts the stop could not know are taken from it.
        if self.tailed_indices.contains(&index) {
            return Ok(());
        }
        self.total_prompt = self.total_prompt.max(complete.prompt_tokens());
        self.total_cached = self.total_cached.max(complete.cached_tokens());
        self.reasoning_tokens
            .insert(index, complete.reasoning_tokens());
        self.total_completion.record_complete(&complete);

        if self.stopped_indices.contains(&index) {
            return Ok(());
        }

        // Handle echo when Complete arrives without any preceding Chunks
        // (e.g., max_tokens=0). The Chunk arm normally prepends prompt_text
        // on the first event, but if no Chunks arrive we must emit it here.
        let is_first = self.is_firsts.entry(index).or_insert(true);
        if *is_first && self.spec.echo && !self.prompt_text.is_empty() {
            StreamingProcessor::send_completion_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.frame,
                &completion_choice(index, self.prompt_text.to_string(), None),
            )
            .await?;
            *is_first = false;
        }

        if let Some(decoder) = self.stop_decoders.get_mut(&index) {
            let tail = decoder.finish().map_err(|e| e.to_string())?;
            utils::warn_detokenize_recoveries(&self.dispatch.request_id, index, decoder);
            if decoder.is_stopped() {
                self.stopped_indices.insert(index);
            }
            if let SequenceDecoderOutput::Text(text)
            | SequenceDecoderOutput::StoppedWithText(text) = tail
            {
                if !text.is_empty() {
                    StreamingProcessor::send_completion_chunk(
                        self.tx,
                        &mut self.sse_buffer,
                        &self.frame,
                        &completion_choice(index, text, None),
                    )
                    .await?;
                }
            }
        }

        if let Some(sfx) = self.spec.suffix.as_deref() {
            StreamingProcessor::send_completion_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.frame,
                &completion_choice(index, sfx.to_string(), None),
            )
            .await?;
        }

        let finish_reason = utils::completion_finish_reason(
            self.stopped_indices.contains(&index),
            complete.finish_reason(),
        )
        .to_string();

        StreamingProcessor::send_completion_chunk(
            self.tx,
            &mut self.sse_buffer,
            &self.frame,
            &completion_choice(index, String::new(), Some(finish_reason)),
        )
        .await?;
        Ok(())
    }

    /// After the engine's end: pending text and locally matched stops of
    /// the choices no `Complete` closed go out, without inventing engine
    /// usage. A local stop still honors the requested suffix.
    pub(super) async fn finish_open_choices(&mut self) -> Result<(), utils::StreamFailure> {
        for (index, decoder) in &mut self.stop_decoders {
            if self.completed_indices.contains(index) || self.stopped_indices.contains(index) {
                continue;
            }
            let tail = decoder.finish().map_err(|e| e.to_string())?;
            utils::warn_detokenize_recoveries(&self.dispatch.request_id, *index, decoder);
            let stopped = decoder.is_stopped();
            let mut text = match tail {
                SequenceDecoderOutput::Text(text)
                | SequenceDecoderOutput::StoppedWithText(text) => text,
                _ => String::new(),
            };
            if stopped {
                if let Some(sfx) = self.spec.suffix.as_deref() {
                    text.push_str(sfx);
                }
            }
            if text.is_empty() && !stopped {
                continue;
            }
            StreamingProcessor::send_completion_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.frame,
                &completion_choice(*index, text, stopped.then(|| "stop".to_string())),
            )
            .await?;
        }
        Ok(())
    }

    /// Choices closed by their Complete or by their local stop.
    fn closed_choices(&self) -> u32 {
        (self.completed_indices.len()
            + self
                .stopped_indices
                .difference(&self.completed_indices)
                .count()) as u32
    }

    /// Whether the engine has outlived the unit: every expected choice is
    /// closed while a stopped choice without its Complete has streamed on
    /// past its stop, so the engine is generating a tail nobody receives. A
    /// stopped choice whose next message has not come yet is waited for: an
    /// engine that saw the stop ends the choice on that step and its
    /// Complete is the next message, so nothing is aborted; a chunk instead
    /// is the tail. Nothing to look at while no stop has fired.
    fn engine_outlived_now(&self) -> bool {
        if self.stopped_indices.is_empty() {
            return false;
        }
        let mut open_stops = 0u32;
        let mut tailed = false;
        for index in self.stopped_indices.difference(&self.completed_indices) {
            open_stops += 1;
            tailed |= self.tailed_indices.contains(index);
        }
        tailed && self.completed_indices.len() as u32 + open_stops >= self.spec.choices_per_prompt
    }

    /// The engine stream, released before the unit's open choices are
    /// finished. An engine the gateway's own stop outlived is aborted now:
    /// dropping the stream un-marked is the abort path a client disconnect
    /// takes, so the slot goes back to the engine's batch within a step
    /// instead of running to `max_tokens` for nobody. An engine that ended
    /// on its own is marked completed, so that its drop sends no abort.
    pub(super) fn release_engine(&self, mut grpc_stream: ProtoStream) {
        if !self.engine_outlived {
            grpc_stream.mark_completed();
            return;
        }
        Metrics::record_local_stop_abort(LocalStopAbort {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: &self.dispatch.model,
            endpoint: metrics_labels::ENDPOINT_COMPLETIONS,
            tail_tokens: self.tail_tokens,
        });
        drop(grpc_stream);
    }

    /// The unit's totals for the coordinator. `completed_indices` counts
    /// distinct indices that finished cleanly via a `Complete` message. A
    /// clean EOF partway through this unit's `n>1` choices (some completed,
    /// others didn't) must not be treated as full usage.
    pub(super) fn outcome(self) -> CompletionStreamOutcome {
        let expected_choices = self.spec.choices_per_prompt;
        let saw_complete = self.closed_choices() >= expected_choices;
        CompletionStreamOutcome {
            prompt_tokens: self.total_prompt,
            cached_tokens: self.total_cached,
            reasoning_tokens: self.reasoning_tokens.values().sum(),
            spec_accepted_tokens: self.spec_accepted.values().sum(),
            spec_draft_tokens: self.spec_drafted.values().sum(),
            completion_tokens: self.total_completion.total(),
            first_token_time: self.first_token_time,
            saw_complete,
        }
    }
}

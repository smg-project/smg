//! The `Generate` RPC: the one request path. Resolves the frontend's stop and
//! EOS duties, submits per-choice engine streams, matches string stops on the
//! decoded output, and merges the choices into one response stream that the
//! `Abort` RPC can end.

use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{atomic::Ordering, Arc},
    task::{Context, Poll},
};

use engine_zmq_adapter::{
    fold_tokenizer_eos_backstop, kv_transfer_rejection_params, stops::take_vllm_string_stops,
    VllmGenerateStream, ZmqEngineClient,
};
use futures::{stream::SelectAll, Stream};
use llm_tokenizer::stop::{StopSequenceDecoder, StopSequenceDecoderBuilder};
use smg_grpc_client::vllm_proto as vllm;
use tokio::sync::oneshot;
use tonic::Status;

use super::{
    requests::{register, Registration},
    State,
};
use crate::BoxStream;

/// The string-stop matcher for a request, or `None` when it carries no
/// string stops. Visible (included-in-output) stops only change the text
/// the Router trims; the servicer only needs the match itself.
pub(super) fn stop_decoder(
    state: &State,
    stops: &[String],
    skip_special_tokens: bool,
) -> Result<Option<StopSequenceDecoder>, Status> {
    if stops.is_empty() {
        return Ok(None);
    }
    let Some(tokenizer) = state.tokenizer() else {
        return Err(Status::failed_precondition(
            "string `stop` sequences need the model tokenizer, which this servicer could not \
             load; start it with a local tokenizer directory",
        ));
    };
    let mut builder = StopSequenceDecoderBuilder::new(Arc::clone(tokenizer))
        .skip_special_tokens(skip_special_tokens);
    for stop in stops {
        builder = builder.stop_sequence(stop.clone());
    }
    Ok(Some(builder.build()))
}

/// Handle one `Generate` request against the connected engine.
pub(super) async fn generate(
    state: &Arc<State>,
    req: vllm::GenerateRequest,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    // A PD decode leg refused before admission must still release the blocks
    // its prefill side pinned, as vLLM's own frontend does.
    let rejection = kv_transfer_rejection_params(&req).map(|params| {
        (
            req.request_id.clone(),
            params,
            req.data_parallel_rank
                .and_then(|rank| u32::try_from(rank).ok()),
        )
    });
    match submit(state, client, req).await {
        Ok(stream) => Ok(stream),
        Err(status) => {
            if let Some((request_id, params, rank)) = rejection {
                client
                    .notify_kv_transfer_rejected(&request_id, params, rank)
                    .await;
            }
            Err(status)
        }
    }
}

/// Resolve the frontend duties and submit the request's choices.
async fn submit(
    state: &Arc<State>,
    client: &ZmqEngineClient,
    mut req: vllm::GenerateRequest,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    // Refused here rather than dropped on translation: a request the engine
    // would run without its media must fail, not answer text-only.
    if req
        .media_refs
        .as_ref()
        .is_some_and(|refs| !refs.items.is_empty())
    {
        return Err(Status::unimplemented(
            "media_refs: worker-side media processing is not available on the Rust servicer; \
             have the Router preprocess media (`--mm-processing router`) or use the Python \
             servicer",
        ));
    }
    if !req.extra_mm_inputs.is_empty() {
        return Err(Status::unimplemented(
            "extra_mm_inputs (a second modality batch) is not supported on the Rust servicer yet",
        ));
    }
    let request_id = req.request_id.clone();
    let streaming = req.stream;
    let skip_special_tokens = req
        .sampling_params
        .as_ref()
        .is_some_and(|sp| sp.skip_special_tokens);
    let min_tokens = req.sampling_params.as_ref().map_or(0, |sp| sp.min_tokens);
    // The same stop/EOS finalization the Router applies on its direct-ZMQ
    // lane: string stops leave the engine request (single-token ones become
    // stop ids) and come back as this servicer's own obligation; the
    // tokenizer's EOS set backstops a config without ids.
    let tokenizer = state.tokenizer();
    let stops = req
        .sampling_params
        .as_mut()
        .map(|params| take_vllm_string_stops(params, tokenizer))
        .unwrap_or_default();
    client.adopt_tokenizer_eos(tokenizer);
    fold_tokenizer_eos_backstop(&mut req, tokenizer);
    let decoder = stop_decoder(state, &stops, skip_special_tokens)?;

    // Register before submitting: an `Abort` landing in the gap must find
    // the entry, or the engine keeps generating behind an accepted abort.
    let (registration, cancel) = register(state, &request_id)?;
    let subs = client.generate_vllm_streams(req).await?;
    let mut choices = SelectAll::new();
    for sub in subs {
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(state, &stops, skip_special_tokens)?,
            None => None,
        };
        choices.push(ChoiceStream::new(
            Arc::clone(state),
            sub,
            decoder,
            streaming,
            min_tokens,
        ));
    }
    Ok(Box::pin(GenerateStream {
        choices,
        cancel,
        aborted: None,
        _registration: registration,
    }))
}

/// One choice of a generate request: the ZMQ-mapped stream plus the string
/// stop matcher the engine cannot run itself.
struct ChoiceStream {
    state: Arc<State>,
    /// `None` once this choice ended (terminal response yielded or error).
    /// Dropping it before the engine's own terminal output aborts the
    /// engine-side request.
    inner: Option<VllmGenerateStream>,
    /// Decodes the output incrementally and matches the request's string
    /// stops; `None` when the request carries none.
    decoder: Option<StopSequenceDecoder>,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`, as from the Python
    /// servicer (`output_kind=FINAL_ONLY`).
    streaming: bool,
    /// A frontend-synthesized terminal `Complete` to yield next.
    pending: Option<vllm::GenerateResponse>,
    /// `min_tokens` of the request and the tokens generated so far: string
    /// stops are only honoured past the minimum, as vLLM does.
    min_tokens: u32,
    generated: u32,
    /// Whether this request's prompt tokens went into the stats counters.
    prompt_counted: bool,
}

impl ChoiceStream {
    fn new(
        state: Arc<State>,
        inner: VllmGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
        min_tokens: u32,
    ) -> Self {
        Self {
            state,
            inner: Some(inner),
            decoder,
            streaming,
            pending: None,
            min_tokens,
            generated: 0,
            prompt_counted: false,
        }
    }

    /// Feed this tick's tokens to the stop matcher, honouring `min_tokens` as
    /// vLLM does: string stops are only checked once the output exceeds it,
    /// and the text up to that point is excluded from the search, so a match
    /// inside it is dropped and the matcher restarts there.
    fn match_stops(&mut self, token_ids: &[u32]) -> Result<Option<String>, Status> {
        let Some(decoder) = self.decoder.as_mut() else {
            self.generated = self
                .generated
                .saturating_add(u32::try_from(token_ids.len()).unwrap_or(u32::MAX));
            return Ok(None);
        };
        for &token in token_ids {
            self.generated = self.generated.saturating_add(1);
            decoder.process_token(token).map_err(|error| {
                Status::internal(format!("incremental detokenization failed: {error}"))
            })?;
            if !decoder.is_stopped() {
                continue;
            }
            if self.generated <= self.min_tokens {
                decoder.reset();
                continue;
            }
            // Only string sequences are registered on the decoder, so a stop
            // always names its matched string.
            return Ok(Some(decoder.matched_stop().unwrap_or_default().to_string()));
        }
        Ok(None)
    }

    /// Count the prompt once per request (every choice of an `n > 1` fan-out
    /// reports the same prompt) for the stats line.
    fn count_prompt(&mut self, prompt_tokens: u32) {
        if self.prompt_counted || prompt_tokens == 0 {
            return;
        }
        if self.inner.as_ref().is_some_and(|inner| inner.index() == 0) {
            self.state
                .stats
                .prompt_tokens
                .fetch_add(u64::from(prompt_tokens), Ordering::Relaxed);
        }
        self.prompt_counted = true;
    }

    /// End this choice on an abort: the `Complete` it was about to yield, the
    /// engine's parked one, or a synthesized `abort` one; `None` once it has
    /// already ended. Dropping the engine stream aborts the engine side.
    fn abort(&mut self) -> Option<vllm::GenerateResponse> {
        if let Some(complete) = self.pending.take() {
            self.inner = None;
            return Some(complete);
        }
        let mut inner = self.inner.take()?;
        Some(inner.complete_aborted())
    }
}

type ChoiceItem = Result<vllm::GenerateResponse, Status>;

impl Stream for ChoiceStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(complete) = this.pending.take() {
                // The frontend ended this choice: drop the engine stream so the
                // engine-side request is aborted, then deliver the Complete.
                this.inner = None;
                return Poll::Ready(Some(Ok(complete)));
            }
            let Some(inner) = this.inner.as_mut() else {
                return Poll::Ready(None);
            };
            let item = match Pin::new(inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.inner = None;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(status))) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(status)));
                }
                Poll::Ready(Some(Ok(item))) => item,
            };
            let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref()
            else {
                // The engine's own terminal `Complete` (or an empty response)
                // passes through as-is; the engine stream ends after it.
                if let Some(vllm::generate_response::Response::Complete(complete)) =
                    item.response.as_ref()
                {
                    this.count_prompt(complete.prompt_tokens);
                }
                return Poll::Ready(Some(Ok(item)));
            };
            this.count_prompt(chunk.prompt_tokens);
            this.state
                .stats
                .generation_tokens
                .fetch_add(chunk.token_ids.len() as u64, Ordering::Relaxed);
            let matched = match this.match_stops(&chunk.token_ids) {
                Ok(matched) => matched,
                Err(status) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(status)));
                }
            };
            {
                if let Some(matched) = matched {
                    if let Some(inner) = this.inner.as_mut() {
                        // A stop string that is also a single token reaches the
                        // engine as a stop id, so the engine may finish on the
                        // very tick the string matched. Its parked `Complete`
                        // already holds the accumulated ids and logprobs; let
                        // it through instead of synthesizing an empty one.
                        if !inner.has_parked_complete() {
                            this.pending = Some(inner.complete_with_matched_stop(matched));
                        }
                    }
                    if !this.streaming {
                        continue;
                    }
                    // The triggering delta goes out first, as vLLM's own
                    // frontend delivers the step that matched.
                    return Poll::Ready(Some(Ok(item)));
                }
            }
            if !this.streaming {
                continue;
            }
            return Poll::Ready(Some(Ok(item)));
        }
    }
}

/// The merged response stream of one generate request: its choices polled
/// together (interleaved, each tagged with its `index`), ended early by the
/// `Abort` RPC.
struct GenerateStream {
    choices: SelectAll<ChoiceStream>,
    cancel: oneshot::Receiver<()>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<vllm::GenerateResponse>>,
    _registration: Registration,
}

impl Stream for GenerateStream {
    type Item = ChoiceItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(aborted) = this.aborted.as_mut() {
                return Poll::Ready(aborted.pop_front().map(Ok));
            }
            // Either an `Abort` fired the sender or shutdown dropped the
            // registry: both end the request as the Python servicer does, with
            // each open choice's `Complete(finish_reason = "abort")`. Dropping
            // the choices aborts the engine side.
            if Pin::new(&mut this.cancel).poll(cx).is_ready() {
                let aborted = this
                    .choices
                    .iter_mut()
                    .filter_map(ChoiceStream::abort)
                    .collect();
                this.choices.clear();
                this.aborted = Some(aborted);
                continue;
            }
            return Pin::new(&mut this.choices).poll_next(cx);
        }
    }
}

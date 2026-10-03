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
use llm_tokenizer::stop::StopSequenceDecoder;
use smg_grpc_client::vllm_proto as vllm;
use tokio::sync::oneshot;
use tonic::Status;

use super::{media::process_media_refs, State};
use crate::{
    requests::Registration,
    stop_match::{stop_decoder, StopMatcher},
    BoxStream,
};

/// Handle one `Generate` request against the connected engine.
pub(super) async fn generate(
    state: &Arc<State>,
    req: vllm::GenerateRequest,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    submit(state, client, req).await
}

/// The rejection notice a refused PD decode leg owes its prefill side, so the
/// connector releases the blocks that side pinned: what vLLM's own frontend
/// sends, an immediately aborted one-token request under the refused id.
struct RejectionNotice {
    client: ZmqEngineClient,
    request_id: String,
    params: serde_json::Value,
    rank: Option<u32>,
}

impl RejectionNotice {
    fn owed_by(client: &ZmqEngineClient, req: &vllm::GenerateRequest) -> Option<Self> {
        kv_transfer_rejection_params(req).map(|params| Self {
            client: client.clone(),
            request_id: req.request_id.clone(),
            params,
            rank: req
                .data_parallel_rank
                .and_then(|rank| u32::try_from(rank).ok()),
        })
    }

    /// Send the notice, detached: the refusal does not wait on the engine,
    /// and a caller that already gave up has no future left to carry it.
    fn send(self) {
        let Self {
            client,
            request_id,
            params,
            rank,
        } = self;
        #[expect(
            clippy::disallowed_methods,
            reason = "the notice outlives the refused request's future; it ends once the add is out"
        )]
        tokio::spawn(async move {
            client
                .notify_kv_transfer_rejected(&request_id, params, rank)
                .await;
        });
    }
}

/// Sends the notice it holds when dropped. Armed through admission, where a
/// caller that gives up (a media fetch can take seconds) drops the request's
/// future: the notice still goes out, as the Python servicer shields it
/// through cancellation. `disarm` hands the notice back unsent.
struct ArmedNotice(Option<RejectionNotice>);

impl ArmedNotice {
    fn disarm(&mut self) -> Option<RejectionNotice> {
        self.0.take()
    }
}

impl Drop for ArmedNotice {
    fn drop(&mut self) {
        if let Some(notice) = self.0.take() {
            notice.send();
        }
    }
}

/// Resolve the frontend duties and submit the request's choices.
async fn submit(
    state: &Arc<State>,
    client: &ZmqEngineClient,
    mut req: vllm::GenerateRequest,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    // A PD decode leg refused before admission still owes its prefill side
    // the notice: armed from the first refusal below until the engine can
    // hold the request.
    let mut notice = ArmedNotice(RejectionNotice::owed_by(client, &req));
    // A pooling runner serves no generation task; vLLM's frontend refuses
    // this before the engine sees it, in these words.
    if !state.model.is_generation {
        return Err(Status::invalid_argument(
            "This model does not support generation",
        ));
    }
    let request_id = req.request_id.clone();
    // Register before anything slow: an `Abort` landing while the media is
    // processed must find the entry, and a duplicate id is refused up front.
    let (registration, cancel) = state.registry.register(&request_id).inspect_err(|_| {
        // The id names a request that is still live; a notice under it would
        // reuse it (and, past an `n > 1` fan-out, get in and free the live
        // leg's blocks).
        notice.disarm();
    })?;
    // Worker-side media: the request's references become engine features
    // and its prompt the expanded one, before the frontend duties below.
    let (processed_media, media_identity) = match process_media_refs(state, &mut req).await? {
        Some((media, identity)) => (Some(media), identity.map(Arc::new)),
        None => (None, None),
    };
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
    let decoder = stop_decoder(state.tokenizer(), &stops, skip_special_tokens)?;

    // Admission ends here: the add is on the wire before `submit_with_aux`
    // returns, and an `n > 1` fan-out has live subs while later ones are
    // added. A caller that gives up from now on may leave the engine holding
    // the request, which a notice under its id would reuse, so the guard
    // comes off; a refusal below leaves the engine without the request (a
    // fan-out's earlier subs are aborted with the error) and still owes one.
    let owed = notice.disarm();
    let subs = match client
        .generate_vllm_streams_with_media(req, processed_media)
        .await
    {
        Ok(subs) => subs,
        Err(status) => {
            if let Some(notice) = owed {
                notice.send();
            }
            return Err(status);
        }
    };
    let mut choices = SelectAll::new();
    for sub in subs {
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(state.tokenizer(), &stops, skip_special_tokens)?,
            None => None,
        };
        choices.push(ChoiceStream::new(
            Arc::clone(state),
            sub,
            decoder,
            streaming,
            min_tokens,
            media_identity.clone(),
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
    /// stops, honouring `min_tokens`.
    stops: StopMatcher,
    /// Whether the client asked for incremental chunks. A non-streaming
    /// request receives only the terminal `Complete`, as from the Python
    /// servicer (`output_kind=FINAL_ONLY`).
    streaming: bool,
    /// A frontend-synthesized terminal `Complete` to yield next.
    pending: Option<vllm::GenerateResponse>,
    /// Whether this request's prompt tokens went into the stats counters.
    prompt_counted: bool,
    /// A PD prefill leg's account of the media it processed, stamped on
    /// every `Complete` of the request for the decode leg.
    media_identity: Option<Arc<vllm::MediaIdentity>>,
}

impl ChoiceStream {
    fn new(
        state: Arc<State>,
        inner: VllmGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
        min_tokens: u32,
        media_identity: Option<Arc<vllm::MediaIdentity>>,
    ) -> Self {
        Self {
            media_identity,
            state,
            inner: Some(inner),
            stops: StopMatcher::new(decoder, min_tokens),
            streaming,
            pending: None,
            prompt_counted: false,
        }
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
            return Some(self.stamped(complete));
        }
        let mut inner = self.inner.take()?;
        let complete = inner.complete_aborted();
        Some(self.stamped(complete))
    }

    /// A terminal response with the request's media identity on it, when
    /// there is one; anything else passes through.
    fn stamped(&self, mut response: vllm::GenerateResponse) -> vllm::GenerateResponse {
        if let (Some(identity), Some(vllm::generate_response::Response::Complete(complete))) =
            (self.media_identity.as_deref(), response.response.as_mut())
        {
            complete.media_identity = Some(identity.clone());
        }
        response
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
                return Poll::Ready(Some(Ok(this.stamped(complete))));
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
                return Poll::Ready(Some(Ok(this.stamped(item))));
            };
            this.count_prompt(chunk.prompt_tokens);
            this.state
                .stats
                .generation_tokens
                .fetch_add(chunk.token_ids.len() as u64, Ordering::Relaxed);
            let matched = match this.stops.feed(&chunk.token_ids) {
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

//! The `Generate` RPC: the one request path. Resolves the frontend's stop and
//! EOS duties, submits per-choice engine streams, matches string stops on the
//! decoded output, and merges the choices into one response stream that the
//! `Abort` RPC can end.

use std::{
    collections::BTreeMap,
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use engine_zmq_adapter::{
    fold_tokenizer_eos_backstop, kv_transfer_rejection_params, stops::take_vllm_string_stops,
    VllmGenerateStream, ZmqEngineClient,
};
use futures::Stream;
use llm_tokenizer::stop::StopSequenceDecoder;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use super::{media::process_media_refs, State};
use crate::{
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, PrefillProbe},
    metrics::{complete_finish_reason, ChoiceMetrics},
    requests::HeldTerminal,
    stop_match::{stop_decoder, StopMatcher},
    stream::{poll_engine_item, Choice, Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected engine.
/// `trace_headers` is the caller's W3C trace context (see
/// [`crate::trace_context::trace_headers`]), handed to the engine request in
/// the shape vLLM's own frontend hands its tracer.
pub(super) async fn generate(
    state: &Arc<State>,
    req: vllm::GenerateRequest,
    trace_headers: Option<BTreeMap<String, String>>,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    log_request_line(&req);
    submit(state, client, req, trace_headers, arrival).await
}

/// The refusal of a request that carries encoder-cache transfer parameters
/// (`ec_transfer_params_json`): this servicer runs no encoder-cache
/// connector, which the Router reads off the `supports_ec_transfer` label
/// this servicer leaves unset; the shape is the Python servicer's refusal of
/// a media reference with processing off (`media::PROCESSOR_OFF`).
const NO_ENCODER_CACHE: &str = "ec_transfer_params_json sent but this worker runs no \
                                encoder-cache connector; check this worker's \
                                supports_ec_transfer label";

/// The tracing target of the per-request line: a reader that wants it raises
/// this target alone (`RUST_LOG=info,engine_servicer::request=debug`); the
/// production filter leaves it off, and an event below the filter costs the
/// enabled check and nothing else.
pub(super) const REQUEST_LINE_TARGET: &str = "engine_servicer::request";

/// Log the request as received, at DEBUG under [`REQUEST_LINE_TARGET`], in the
/// words the Python servicer logs for every `Generate` at INFO: what a reader
/// of a worker's log greps for (a PD side's media placement, the rank the
/// gateway pinned). The line is formatted only when that target is enabled.
pub(super) fn log_request_line(req: &vllm::GenerateRequest) {
    if tracing::enabled!(target: REQUEST_LINE_TARGET, tracing::Level::DEBUG) {
        tracing::debug!(target: REQUEST_LINE_TARGET, "{}", request_line(req));
    }
}

/// One line describing a `Generate` request as received, in the Python
/// servicer's words: the input kind by its field name, whether it streams,
/// whether a multimodal batch carries tensors the preprocessed path can use
/// (pixels or model-specific tensors; a PD decode side built from the
/// prefill's identity carries its grid tensors and so reads
/// `preprocessed_mm=True, media_refs=0`, while a bare payload of hashes alone
/// does not count), the count of worker-side media references, and the
/// data-parallel rank the
/// gateway pinned (`None` when it pinned none). Booleans and the absent rank
/// spell as Python does, so the same log readers serve both servicers.
pub(super) fn request_line(req: &vllm::GenerateRequest) -> String {
    let input_type = match req.input {
        Some(vllm::generate_request::Input::Tokenized(_)) => "tokenized",
        Some(vllm::generate_request::Input::Text(_)) => "text",
        None => "None",
    };
    let preprocessed_mm = req
        .mm_inputs
        .iter()
        .chain(req.extra_mm_inputs.iter())
        .any(|batch| batch.pixel_values.is_some() || !batch.model_specific_tensors.is_empty());
    let media_refs = req.media_refs.as_ref().map_or(0, |refs| refs.items.len());
    let dp_rank = req
        .data_parallel_rank
        .map_or_else(|| "None".to_string(), |rank| rank.to_string());
    format!(
        "Generate request {}: input_type={input_type}, stream={}, preprocessed_mm={}, \
         media_refs={media_refs}, dp_rank={dp_rank}",
        req.request_id,
        python_bool(req.stream),
        python_bool(preprocessed_mm),
    )
}

/// A boolean as Python's logger spells it.
fn python_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

/// The rejection notice a refused PD decode side owes its prefill side, so the
/// connector releases the blocks that side pinned: what vLLM's own frontend
/// sends, an immediately aborted one-token request under the refused id.
struct RejectionNotice {
    client: ZmqEngineClient,
    request_id: String,
    params: serde_json::Value,
    rank: Option<u32>,
    /// Whether a Mooncake connector runs in the engine, which gets no
    /// notice (see `send`): the launcher reads it off the raw
    /// `kv_transfer_config`, the top-level connector or any child a wrapper
    /// lists, not off the connector name the projection `GetServerInfo`
    /// reports settles on (a wrapper hands the notice's allocation to every
    /// child and merges every child's finished transfers, so a Mooncake
    /// child dies on it under any projection).
    withheld: bool,
}

impl RejectionNotice {
    /// The notice `req` owes when it is a PD decode side, `runs_mooncake`
    /// being whether a Mooncake connector runs in the engine, as the
    /// launcher read it off the raw config.
    fn owed_by(
        client: &ZmqEngineClient,
        runs_mooncake: bool,
        req: &vllm::GenerateRequest,
    ) -> Option<Self> {
        kv_transfer_rejection_params(req).map(|params| Self {
            client: client.clone(),
            request_id: req.request_id.clone(),
            params,
            rank: req
                .data_parallel_rank
                .and_then(|rank| u32::try_from(rank).ok()),
            withheld: runs_mooncake,
        })
    }

    /// Send the notice, detached: the refusal does not wait on the engine,
    /// and a caller that already gave up has no future left to carry it.
    ///
    /// On the Mooncake connector the notice is withheld, a legitimate
    /// pre-admission one included, with one warning line: that connector
    /// reports the notice's empty pull as a finished transfer and the decode
    /// engine dies on it; the prefill side's blocks are released by the
    /// connector's own abort timeout instead.
    fn send(self) {
        let Self {
            client,
            request_id,
            params,
            rank,
            withheld,
        } = self;
        if withheld {
            // Remove once upstream fixes the connector: in vLLM 0.31's
            // MooncakeConnector the notice's request is admitted and freed in
            // one scheduler step, its empty pull (the prefill-block release)
            // completes afterwards, and `process_pulling_result` reports it
            // as `finished_recving` like a pull that carried KV; the
            // scheduler's `_update_from_kv_xfer_finished` then asserts that
            // the freed id is still tracked and the EngineCore exits. The
            // connector's failure path (`_handle_failed_recv`) guards exactly
            // this case and names this hook; the success path does not. NIXL
            // releases with a notification and no pull, so it keeps the
            // notice.
            tracing::warn!(
                request_id,
                "KV-transfer rejection notice withheld on the Mooncake connector: vLLM 0.31's \
                 MooncakeConnector.process_pulling_result reports the notice's empty pull as \
                 finished_recving for an id Scheduler._update_from_kv_xfer_finished no longer \
                 tracks, which kills the decode engine; the prefill side releases the blocks \
                 at the connector's own abort timeout instead"
            );
            return;
        }
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
    trace_headers: Option<BTreeMap<String, String>>,
    arrival: Instant,
) -> Result<BoxStream<vllm::GenerateResponse>, Status> {
    // A PD decode side refused before admission still owes its prefill side
    // the notice: armed from the first refusal below until the engine can
    // hold the request.
    let mut notice = ArmedNotice(RejectionNotice::owed_by(
        client,
        state.model.runs_mooncake,
        &req,
    ));
    // A pooling runner serves no generation task; vLLM's frontend refuses
    // this before the engine sees it, in these words.
    if !state.model.is_generation {
        return Err(Status::invalid_argument(
            "This model does not support generation",
        ));
    }
    // An encode-prefill-decode leg (the encoder's request with the consumer's
    // destination, or the prefill side's with the encoder's handles) names a
    // handoff this servicer does not run: no encoder-cache connector is
    // configured through it and `GetServerInfo` advertises none, so a Router
    // selects it for neither side. One that arrives all the same is refused
    // here, before the media stage or the engine sees it (the engine would
    // run it as a plain request and the caller would read a completion
    // without the handles), in the shape a media reference is refused on a
    // worker whose processor is off. An empty field is no parameters, as the
    // Python servicer reads it.
    if req
        .ec_transfer_params_json
        .as_deref()
        .is_some_and(|params| !params.is_empty())
    {
        return Err(Status::invalid_argument(NO_ENCODER_CACHE));
    }
    let request_id = req.request_id.clone();
    // Register before anything slow: an `Abort` landing while the media is
    // processed must find the entry, and a duplicate id is refused up front.
    let (registration, cancel) =
        state
            .shared
            .registry
            .register(&request_id)
            .inspect_err(|error| {
                // The id names a request that is still live; a notice under it would
                // reuse it (and, past an `n > 1` fan-out, get in and free the live
                // side's blocks). Only a drain refusal proves admission is closed;
                // a poisoned registry cannot establish whether the id is live.
                if error.code() != tonic::Code::Unavailable {
                    notice.disarm();
                }
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
    let prompt_tokens = match req.input.as_ref() {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => {
            u32::try_from(tokenized.input_ids.len()).unwrap_or(u32::MAX)
        }
        _ => 0,
    };
    let (max_tokens, n) = req
        .sampling_params
        .as_ref()
        .map_or((None, 1), |params| (params.max_tokens, params.n));
    let subs = match client
        .generate_vllm_streams_with_media(req, processed_media, trace_headers)
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
    // Queued token-work for `GetLoads`: in flight from here until the
    // engine's first output (or the stream's end), on the rank the first
    // choice went to (the choice whose first output reports the prompt).
    let rank = subs
        .iter()
        .find(|sub| sub.index() == 0)
        .or_else(|| subs.first())
        .map_or(0, |sub| rank_of(sub.engine_index()));
    state
        .shared
        .loads
        .submitted(rank, &request_id, prompt_tokens);
    let loads = PendingGuard::new(Arc::clone(state), &request_id);
    let metrics = state.shared.metrics.request(
        arrival,
        u32::try_from(rank).ok(),
        prompt_tokens,
        max_tokens,
        n,
    );
    let mut choices = Choices::new();
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
            request_id.clone(),
            metrics.choice(),
        ));
    }
    Ok(Box::pin(
        RequestStream::new(choices, cancel, registration, metrics).holding(loads),
    ))
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
    /// This choice's observation of the request's metrics.
    metrics: ChoiceMetrics,
    /// A PD prefill side's account of the media it processed, stamped on
    /// every `Complete` of the request for the decode side.
    media_identity: Option<Arc<vllm::MediaIdentity>>,
    /// The request this choice belongs to, for the load tracker.
    request_id: String,
    /// This choice's share of the load tracker's bookkeeping; its prefill
    /// sample comes from the wire's engine timing, so its probe stays unarmed.
    load: ChoiceLoad,
}

impl ChoiceStream {
    #[expect(clippy::too_many_arguments)]
    fn new(
        state: Arc<State>,
        inner: VllmGenerateStream,
        decoder: Option<StopSequenceDecoder>,
        streaming: bool,
        min_tokens: u32,
        media_identity: Option<Arc<vllm::MediaIdentity>>,
        request_id: String,
        metrics: ChoiceMetrics,
    ) -> Self {
        Self {
            media_identity,
            request_id,
            load: ChoiceLoad::new(
                rank_of(inner.engine_index()),
                inner.index(),
                PrefillProbe::default(),
            ),
            state,
            inner: Some(inner),
            stops: StopMatcher::new(decoder, min_tokens),
            streaming,
            pending: None,
            metrics,
        }
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

impl Choice for ChoiceStream {
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
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        Some(self.stamped(complete))
    }
}

impl HeldTerminal for ChoiceStream {
    type Response = vllm::GenerateResponse;

    /// The terminal this choice already holds, when the engine (or a stop
    /// string) finished it before the poll that would have yielded it: the
    /// `Complete` it was about to yield, or the engine's parked one. `None`
    /// for a choice still open, which is left as it is.
    fn finished(&mut self) -> Option<vllm::GenerateResponse> {
        if let Some(complete) = self.pending.take() {
            self.inner = None;
            return Some(self.stamped(complete));
        }
        let inner = self.inner.as_mut()?;
        if !inner.has_parked_complete() {
            return None;
        }
        // `complete_aborted` yields the parked `Complete` when there is one.
        let complete = inner.complete_aborted();
        if let Some(reason) = complete_finish_reason(&complete) {
            self.metrics.ended(reason);
        }
        self.inner = None;
        Some(self.stamped(complete))
    }

    /// Whether this choice is still open: neither ended nor holding its
    /// terminal.
    fn is_open(&self) -> bool {
        self.pending.is_none() && self.inner.is_some()
    }
}

type ChoiceItem = Result<vllm::GenerateResponse, Status>;

impl Stream for ChoiceStream {
    type Item = ChoiceItem;

    /// Forward engine output and account for input capacity before frontend stop handling.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(complete) = this.pending.take() {
                // The frontend ended this choice: drop the engine stream so the
                // engine-side request is aborted, then deliver the Complete.
                this.inner = None;
                return Poll::Ready(Some(Ok(this.stamped(complete))));
            }
            let item = match poll_engine_item(&mut this.inner, &mut this.metrics, cx) {
                ControlFlow::Continue(item) => item,
                ControlFlow::Break(poll) => return poll,
            };
            if let Some((tokens, start, end)) = this
                .inner
                .as_mut()
                .and_then(|inner| inner.take_prefill_sample())
            {
                this.state
                    .shared
                    .loads
                    .prefill_computed(this.load.rank(), tokens, start, end);
            }
            let Some(vllm::generate_response::Response::Chunk(chunk)) = item.response.as_ref()
            else {
                // The engine's own terminal `Complete` (or an empty response)
                // passes through as-is; the engine stream ends after it, and
                // the choice is no longer open once the `Complete` is out (a
                // shutdown landing before the next poll must not cut it; the
                // engine stream has already seen the finish, so dropping it
                // sends no abort).
                let mut ended = false;
                if let Some(vllm::generate_response::Response::Complete(complete)) =
                    item.response.as_ref()
                {
                    this.load.count_prompt(
                        &this.state.shared.loads,
                        &this.request_id,
                        complete.prompt_tokens,
                        complete.cached_tokens,
                        Instant::now(),
                    );
                    this.metrics.complete(
                        &complete.finish_reason,
                        complete.prompt_tokens,
                        complete.cached_tokens,
                    );
                    ended = true;
                }
                let response = this.stamped(item);
                if ended {
                    this.inner = None;
                }
                return Poll::Ready(Some(Ok(response)));
            };
            let tokens = u32::try_from(chunk.token_ids.len()).unwrap_or(u32::MAX);
            // The metrics' clock read serves the load tracker's window too.
            let at = this
                .metrics
                .chunk(tokens, chunk.prompt_tokens, chunk.cached_tokens)
                .unwrap_or_else(Instant::now);
            let loads = &this.state.shared.loads;
            this.load.count_prompt(
                loads,
                &this.request_id,
                chunk.prompt_tokens,
                chunk.cached_tokens,
                at,
            );
            this.load.generated(loads, tokens, at);
            let matched = match this.stops.feed(&chunk.token_ids) {
                Ok(matched) => matched,
                Err(status) => {
                    this.inner = None;
                    this.metrics.failed(status.code());
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
                            this.metrics.string_stop();
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

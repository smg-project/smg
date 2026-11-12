//! The `Generate` RPC: the one request path. Resolves the frontend's stop and
//! EOS duties, submits per-choice engine streams, matches string stops on the
//! decoded output, and merges the choices into one response stream that the
//! `Abort` RPC can end.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use engine_zmq_adapter::{
    fold_tokenizer_eos_backstop, kv_transfer_rejection_params, stops::take_vllm_string_stops,
    VllmGenerateStream, ZmqEngineClient,
};
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use super::{encoder_cache, media::process_media_refs, State};
use crate::{
    choice::{Plain, Shape, StopMatched},
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, PrefillProbe},
    stop_match::stop_decoder,
    stream::{Choices, RequestStream},
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
/// (`ec_transfer_params_json`) on an engine without an encoder-cache
/// connector (the launcher reports none off `ec_transfer_config`, and the
/// Router reads it off the `supports_ec_transfer` label left unset); the
/// shape is the Python servicer's refusal of a media reference with
/// processing off (`media::PROCESSOR_OFF`).
const NO_ENCODER_CACHE: &str = "ec_transfer_params_json sent but this worker runs no \
                                encoder-cache connector; check this worker's \
                                supports_ec_transfer label";

/// Whether the request carries encoder-cache transfer parameters: a non-empty
/// `ec_transfer_params_json` (the encoder's request names the consumer's
/// destination, or an empty object for a pull connector; the prefill side's
/// names the encoder's handles). An empty field is no parameters, as the
/// Python servicer reads it.
pub(super) fn ec_transfer_params_sent(req: &vllm::GenerateRequest) -> bool {
    req.ec_transfer_params_json
        .as_deref()
        .is_some_and(|params| !params.is_empty())
}

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
    // An encode-prefill-decode request (the encoder's with the consumer's
    // destination, or the prefill side's with the encoder's handles) rides
    // the engine's encoder-cache connector, which the launcher reports off
    // `ec_transfer_config` and `GetServerInfo` advertises, so a Router selects
    // an engine without one for neither side. A request that arrives there all
    // the same is refused here, before the media stage or the engine sees it
    // (the engine would run it as a plain request and the caller would read
    // a completion without the handles), in the shape a media reference is
    // refused on a worker whose processor is off.
    if !state.model.supports_ec_transfer && ec_transfer_params_sent(&req) {
        return Err(Status::invalid_argument(NO_ENCODER_CACHE));
    }
    // On an engine with a connector, the request's parameters (a JSON object; a
    // malformed field is refused by the adapter's parse) are checked as the
    // Python servicer checks them: a consumer's pixel-less batches need a
    // usable handle per media item before the engine sees the request.
    let ec_params = (state.model.supports_ec_transfer && ec_transfer_params_sent(&req))
        .then(|| {
            req.ec_transfer_params_json
                .as_deref()
                .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
        })
        .flatten();
    if let Some(params) = ec_params.as_ref() {
        encoder_cache::check_consumer_handoff(&req, &state.model, params)?;
    }
    let ec_producer_side = ec_params.is_some() && encoder_cache::is_producer(&state.model);
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
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
    // An encoder's request with router-preprocessed batches gets its identity
    // from the batches themselves (the media stage saw no reference).
    let media_identity = media_identity.or_else(|| {
        ec_producer_side
            .then(|| encoder_cache::producer_identity(&req, &state.model))
            .flatten()
            .map(Arc::new)
    });
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
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there. Its
        // prefill sample comes from the engine's timing, so its probe stays
        // unarmed.
        let load = ChoiceLoad::new(
            rank_of(sub.engine_index()),
            sub.index(),
            PrefillProbe::default(),
        );
        choices.push(StopMatched::new(
            Plain::new(
                Arc::clone(state),
                Arc::clone(&request_id),
                sub,
                streaming,
                metrics.choice(),
                load,
                VllmShape {
                    media_identity: media_identity.clone(),
                    ec_producer_side,
                },
            ),
            decoder,
            min_tokens,
        ));
    }
    Ok(Box::pin(
        RequestStream::new(choices, cancel, registration, metrics).holding(loads),
    ))
}

/// What this wire stamps on a choice's responses and checks on the engine's
/// terminal: a PD prefill side's account of the media it processed, on every
/// `Complete` of the request for the decode side; on the encoder side of an
/// encoder-cache handoff, the connector's handle for every media item of that
/// identity (`encoder_cache::check_producer_handles`). The vLLM proto is the
/// intermediate itself, so nothing else converts.
struct VllmShape {
    media_identity: Option<Arc<vllm::MediaIdentity>>,
    /// Whether this is the encoder side of an encoder-cache handoff.
    ec_producer_side: bool,
}

impl Shape for VllmShape {
    type Engine = VllmGenerateStream;
    type State = State;
    type Response = vllm::GenerateResponse;

    // The pass-through on the poll path: inlined always, so the response is
    // not moved through the call.
    #[inline(always)]
    fn convert(
        &self,
        _request_id: &str,
        _engine: Option<&VllmGenerateStream>,
        mut response: vllm::GenerateResponse,
    ) -> vllm::GenerateResponse {
        if let (Some(identity), Some(vllm::generate_response::Response::Complete(complete))) =
            (self.media_identity.as_deref(), response.response.as_mut())
        {
            complete.media_identity = Some(identity.clone());
        }
        response
    }

    fn completed(&self, state: &State, complete: &vllm::GenerateComplete) -> Result<(), Status> {
        match self.media_identity.as_deref() {
            Some(identity) if self.ec_producer_side => {
                encoder_cache::check_producer_handles(&state.model, identity, complete)
            }
            _ => Ok(()),
        }
    }
}

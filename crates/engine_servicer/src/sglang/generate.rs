//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The headless scheduler keeps its tokenizer
//! on this wire, so string stops ride the request and the scheduler matches
//! them itself; the servicer has no stop matcher of its own.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use engine_zmq_adapter::{to_sglang_response, SglangGenerateStream};
use futures::StreamExt;
use smg_grpc_client::{
    media_identity::{self as media, sglang as contract},
    sglang_proto as sg, vllm_proto as vllm,
};
use tonic::Status;

use super::{hidden_states_cap, HiddenStatesCap, SglangModelInfo, State};
use crate::{
    choice::{Plain, Shape},
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, PrefillProbe, Submission},
    servicer::RequestOrigin,
    stream::{Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected scheduler.
/// `trace_headers` is the caller's W3C trace context (see
/// [`crate::trace_context::trace_headers`]), carried to the scheduler beside
/// each choice's request so the engine's spans join the Router's trace;
/// `origin` says whose request it is, which decides what is counted
/// ([`RequestOrigin`]).
pub(super) async fn generate(
    state: &Arc<State>,
    mut req: sg::GenerateRequest,
    trace_headers: Option<BTreeMap<String, String>>,
    origin: RequestOrigin,
) -> Result<BoxStream<sg::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    if req.return_hidden_states {
        refuse_hidden_states_above_cap(&state.model)?;
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    // The servicer's own probe is registered for cancellation and settling
    // like a client's request, outside the active count the scrape reports.
    let (registration, mut cancel) = match origin {
        RequestOrigin::Client => state.shared.registry.register(&request_id)?,
        RequestOrigin::HealthProbe => state.shared.registry.register_uncounted(&request_id)?,
    };
    let original = req.return_media_identity.then(|| req.clone());
    let reused = super::media_identity::apply(state, &mut req);
    // Hash fields are only accepted on an internally validated pixel-less hint
    // or on the freshly processed Prefill result below.
    if let Some(mm) = &mut req.mm_inputs {
        if !reused {
            mm.feature_hash = None;
            for batch in &mut req.extra_mm_inputs {
                batch.feature_hash = None;
            }
        }
    }
    if req.mm_inputs.as_ref().is_some_and(|mm| {
        mm.pixel_values.is_none() && (!mm.image_urls.is_empty() || !mm.video_urls.is_empty())
    }) {
        if let Some(processor) = &state.media {
            state.shared.metrics.media_identity("processed");
            req = tokio::select! {
                biased;
                _ = &mut cancel => return Err(Status::cancelled("media request aborted")),
                result = processor.process(req) => result?,
            };
        }
    }
    let mut early_identity = None;
    if let Some(original) = original {
        let mut identity = sg::MediaIdentity::default();
        if super::media_identity::mode(state, "prefill") {
            if let (Some(key), Some(hasher), Some(mm)) = (
                super::media_identity::key(state),
                state.mm_hasher.as_ref(),
                req.mm_inputs.as_mut(),
            ) {
                if contract::reusable_request(&original) {
                    tokio::select! {
                        biased;
                        _ = &mut cancel => return Err(Status::cancelled("media identity request aborted")),
                        result = hasher.precompute_proto(mm, &req.request_id) => result?,
                    };
                    let batch: &sg::MultimodalInputs = mm;
                    identity = sg::MediaIdentity {
                        version: contract::VERSION,
                        processor_key: key.to_string(),
                        request_digest: contract::request_digest(&original),
                        input_ids: req
                            .tokenized
                            .as_ref()
                            .map_or_else(Vec::new, |t| t.input_ids.clone()),
                        mm_inputs: media::identity(batch),
                    };
                    if contract::validate(&identity, &original, key) {
                        state.shared.metrics.media_identity("prepared");
                    } else {
                        identity = sg::MediaIdentity::default();
                    }
                }
            }
        }
        early_identity = Some(sg::GenerateResponse {
            request_id: req.request_id.clone(),
            media_identity: Some(identity),
            response: None,
        });
    }
    let streaming = req.stream;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_ids.len()).unwrap_or(u32::MAX)
    });
    let (max_tokens, n) = req
        .sampling_params
        .as_ref()
        .map_or((None, 1), |params| (params.max_new_tokens, params.n));
    // A language-side worker hands a request's media references to the
    // encoders before the submit (the dispatcher the lifecycle owner
    // supplied); every other worker refuses such a request there.
    let subs = client
        .generate_sglang_streams_dispatching_with_mm_hasher(
            req,
            trace_headers,
            state.encode.as_ref(),
            state.mm_hasher.as_ref(),
        )
        .await?;
    // A client's request is in the load tracker's view (in flight on the
    // rank its first choice went to, the choice that reports the prompt,
    // from here until the engine's first output or the stream's end, each
    // choice's prefill sampled on the servicer's clock from here to its
    // first output: this wire carries no engine timing) and in the request
    // metrics; the servicer's own health probe is in neither.
    let (probe, loads, metrics) = match origin {
        RequestOrigin::Client => {
            let at = Instant::now();
            let rank = rank_of(
                subs.iter()
                    .find(|sub| sub.index() == 0)
                    .or_else(|| subs.first())
                    .and_then(SglangGenerateStream::engine_index),
            );
            (
                state.shared.loads.submit(Submission {
                    request_id: &request_id,
                    rank,
                    prompt_tokens,
                    at,
                }),
                Some(PendingGuard::new(Arc::clone(state), &request_id)),
                state.shared.metrics.request(
                    arrival,
                    subs.first().and_then(SglangGenerateStream::engine_index),
                    prompt_tokens,
                    max_tokens,
                    n,
                ),
            )
        }
        RequestOrigin::HealthProbe => (
            PrefillProbe::default(),
            None,
            state.shared.metrics.uncounted(),
        ),
    };
    let mut choices = Choices::new();
    for sub in subs {
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there.
        let load = match origin {
            RequestOrigin::Client => {
                ChoiceLoad::new(rank_of(sub.engine_index()), sub.index(), probe)
            }
            RequestOrigin::HealthProbe => ChoiceLoad::uncounted(),
        };
        choices.push(Plain::new(
            Arc::clone(state),
            Arc::clone(&request_id),
            sub,
            streaming,
            metrics.choice(),
            load,
            SglangShape,
        ));
    }
    let stream = RequestStream::new(choices, cancel, registration, metrics).holding(loads);
    Ok(Box::pin(
        futures::stream::iter(early_identity.map(Ok)).chain(stream),
    ))
}

/// `return_hidden_states` on a scheduler launched below `full`
/// (`--return-hidden-states-mode`): refused ahead of the scheduler with the
/// engine frontend's own words, as the Python servicer refuses it. A launcher
/// whose server args name no cap leaves the decision to the scheduler.
fn refuse_hidden_states_above_cap(model: &SglangModelInfo) -> Result<(), Status> {
    const LAUNCH: &str =
        "Please launch with `--return-hidden-states-mode full` to allow return_hidden_states=True.";
    match hidden_states_cap(model) {
        None | Some(HiddenStatesCap::Full) => Ok(()),
        Some(HiddenStatesCap::Last) => Err(Status::invalid_argument(format!(
            "The requested return_hidden_states mode exceeds the server maximum `last`. {LAUNCH}"
        ))),
        Some(HiddenStatesCap::Null) => Err(Status::invalid_argument(format!(
            "The server is not configured to return hidden states. {LAUNCH}"
        ))),
    }
}

/// What this wire makes of a choice's items: the SGLang-proto response under
/// the request's id, with the choice's reasoning-token count and selected
/// scores read off the open engine stream, which the vLLM intermediate
/// cannot carry.
struct SglangShape;

impl Shape for SglangShape {
    type Engine = SglangGenerateStream;
    type State = State;
    type Response = sg::GenerateResponse;

    fn convert(
        &self,
        request_id: &str,
        engine: Option<&SglangGenerateStream>,
        response: vllm::GenerateResponse,
    ) -> sg::GenerateResponse {
        match engine {
            Some(inner) => inner.to_sglang_response(request_id, response),
            None => to_sglang_response(request_id, response, 0),
        }
    }
}

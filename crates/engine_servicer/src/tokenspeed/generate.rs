//! The `Generate` RPC: the one request path. Takes the string stops the
//! token-only scheduler cannot match, submits per-choice engine streams,
//! matches the stops on the decoded output, and merges the choices into one
//! response stream that the `Abort` RPC can end.

use std::{sync::Arc, time::Instant};

use engine_zmq_adapter::{
    stops::take_tokenspeed_string_stops, to_tokenspeed_response, TokenSpeedGenerateStream,
};
use smg_grpc_client::{tokenspeed_proto as ts, vllm_proto as vllm};
use tonic::Status;

use super::State;
use crate::{
    choice::{Plain, Shape, StopMatched},
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, PrefillProbe, Submission},
    servicer::RequestOrigin,
    stop_match::stop_decoder,
    stream::{Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected scheduler. `origin`
/// says whose request it is, which decides what is counted
/// ([`RequestOrigin`]).
pub(super) async fn generate(
    state: &Arc<State>,
    mut req: ts::GenerateRequest,
    origin: RequestOrigin,
) -> Result<BoxStream<ts::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    // The servicer's own probe is registered for cancellation and settling
    // like a client's request, outside the active count the scrape reports.
    let (registration, cancel) = match origin {
        RequestOrigin::Client => state.shared.registry.register(&request_id)?,
        RequestOrigin::HealthProbe => state.shared.registry.register_uncounted(&request_id)?,
    };
    let streaming = req.stream;
    let (skip_special_tokens, min_tokens) = req
        .sampling_params
        .as_ref()
        .map_or((false, 0), |sp| (sp.skip_special_tokens, sp.min_new_tokens));
    // String stops leave the engine request (single-token ones become stop
    // ids) and come back as this servicer's own obligation; the scheduler
    // stops at EOS itself.
    let tokenizer = state.tokenizer();
    let stops = req
        .sampling_params
        .as_mut()
        .map(|params| take_tokenspeed_string_stops(params, tokenizer))
        .unwrap_or_default();
    let decoder = stop_decoder(tokenizer, &stops, skip_special_tokens)?;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_ids.len()).unwrap_or(u32::MAX)
    });
    let (max_tokens, n) = req
        .sampling_params
        .as_ref()
        .map_or((None, 1), |params| (params.max_new_tokens, params.n));
    let subs = client
        .generate_tokenspeed_streams_for_model(req, state.model.grid_tensors())
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
                    .and_then(TokenSpeedGenerateStream::engine_index),
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
                    subs.first()
                        .and_then(TokenSpeedGenerateStream::engine_index),
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
        // Each choice decodes its own text: the matcher is per sequence.
        let decoder = match &decoder {
            Some(_) => stop_decoder(tokenizer, &stops, skip_special_tokens)?,
            None => None,
        };
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there.
        let load = match origin {
            RequestOrigin::Client => {
                ChoiceLoad::new(rank_of(sub.engine_index()), sub.index(), probe)
            }
            RequestOrigin::HealthProbe => ChoiceLoad::uncounted(),
        };
        choices.push(StopMatched::new(
            Plain::new(
                Arc::clone(state),
                Arc::clone(&request_id),
                sub,
                streaming,
                metrics.choice(),
                load,
                TokenSpeedShape,
            ),
            decoder,
            min_tokens,
        ));
    }
    Ok(Box::pin(
        RequestStream::new(choices, cancel, registration, metrics).holding(loads),
    ))
}

/// What this wire makes of a choice's items: the TokenSpeed-proto response
/// under the request's id.
struct TokenSpeedShape;

impl Shape for TokenSpeedShape {
    type Engine = TokenSpeedGenerateStream;
    type State = State;
    type Response = ts::GenerateResponse;

    fn convert(
        &self,
        request_id: &str,
        _engine: Option<&TokenSpeedGenerateStream>,
        response: vllm::GenerateResponse,
    ) -> ts::GenerateResponse {
        to_tokenspeed_response(request_id, response)
    }
}

//! The `Generate` RPC: the one request path. Validates, registers the request
//! (so `Abort` can end it), submits the per-choice engine streams and merges
//! them into one response stream. The engine keeps its tokenizer on this
//! wire, so string stops ride the request and the engine matches them
//! itself; the servicer has no stop matcher of its own.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use engine_zmq_adapter::{to_trtllm_response, TrtllmGenerateStream};
use smg_grpc_client::{trtllm_proto as trt, vllm_proto as vllm};
use tonic::Status;

use super::State;
use crate::{
    choice::{Plain, Shape},
    load_tracker::{rank_of, ChoiceLoad, PendingGuard, Submission},
    stream::{Choices, RequestStream},
    BoxStream,
};

/// Handle one `Generate` request against the connected engine.
/// `trace_headers` is the caller's W3C trace context (see
/// [`crate::trace_context::trace_headers`]), carried to the engine beside
/// each choice's request so the engine's spans join the Router's trace.
pub(super) async fn generate(
    state: &Arc<State>,
    req: trt::GenerateRequest,
    trace_headers: Option<BTreeMap<String, String>>,
) -> Result<BoxStream<trt::GenerateResponse>, Status> {
    // The request's arrival, the instant its latencies are measured from.
    let arrival = Instant::now();
    let client = state.shared.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    let request_id: Arc<str> = Arc::from(req.request_id.as_str());
    // Register before the submit: an `Abort` landing in the gap must find the
    // entry, and a duplicate id is refused up front.
    let (registration, cancel) = state.shared.registry.register(&request_id)?;
    let streaming = req.streaming;
    let prompt_tokens = req.tokenized.as_ref().map_or(0, |tokenized| {
        u32::try_from(tokenized.input_token_ids.len()).unwrap_or(u32::MAX)
    });
    let max_tokens = (req.max_tokens > 0).then_some(req.max_tokens);
    let n = req
        .sampling_config
        .as_ref()
        .map_or(1, |config| config.num_return_sequences.max(1));
    let subs = client.generate_trtllm_streams(req, trace_headers).await?;
    // The load tracker's view of the request: in flight on the rank its
    // first choice went to (the choice that reports the prompt) from here
    // until the shim's first output (or the stream's end), each choice's
    // prefill sampled on the servicer's clock from here to its first output
    // (this wire carries no engine timing).
    let at = Instant::now();
    let rank = rank_of(
        subs.iter()
            .find(|sub| sub.index() == 0)
            .or_else(|| subs.first())
            .and_then(TrtllmGenerateStream::engine_index),
    );
    let probe = state.shared.loads.submit(Submission {
        request_id: &request_id,
        rank,
        prompt_tokens,
        at,
    });
    let loads = PendingGuard::new(Arc::clone(state), &request_id);
    let metrics = state.shared.metrics.request(
        arrival,
        subs.first().and_then(TrtllmGenerateStream::engine_index),
        prompt_tokens,
        max_tokens,
        n,
    );
    let mut choices = Choices::new();
    for sub in subs {
        // Each choice's engine request went to its own rank (an unpinned
        // fan-out is placed per choice): its tokens are credited there.
        let load = ChoiceLoad::new(rank_of(sub.engine_index()), sub.index(), probe);
        choices.push(Plain::new(
            Arc::clone(state),
            Arc::clone(&request_id),
            sub,
            streaming,
            metrics.choice(),
            load,
            TrtllmShape,
        ));
    }
    let stream = RequestStream::new(choices, cancel, registration, metrics).holding(loads);
    Ok(Box::pin(stream))
}

/// What this wire makes of a choice's items: the TensorRT-LLM-proto response
/// under the request's id.
struct TrtllmShape;

impl Shape for TrtllmShape {
    type Engine = TrtllmGenerateStream;
    type State = State;
    type Response = trt::GenerateResponse;

    fn convert(
        &self,
        request_id: &str,
        _engine: Option<&TrtllmGenerateStream>,
        response: vllm::GenerateResponse,
    ) -> trt::GenerateResponse {
        to_trtllm_response(request_id, response)
    }
}

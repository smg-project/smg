//! Collecting a request's `Complete` frames from its worker stream(s), for
//! the answers that are not streamed to the client. The regular and the
//! Harmony processors both collect here.

use std::sync::Arc;

use axum::response::Response;
use smg_grpc_client::is_request_timeout;
use smg_worker::identity::basic::Worker;
use tracing::error as trace_error;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::{
            attempt_ledger::mark_pending_charge,
            pd_failure::{PdSide, SidePhase},
            prefill_admission::PrefillLoadGuard,
            retry::mark_non_retryable,
        },
        error,
        grpc::{
            common::stages::request_execution::tell_prefill_failure,
            context::ExecutionResult,
            proto_wrapper::{
                FanoutChild, PrefillCachedTokens, ProtoGenerateComplete, ProtoResponseVariant,
                ProtoStream,
            },
            utils::{
                account_pd_side_failure, dead_rendezvous_of, mark_error_reason,
                stream_failure::UpstreamClass,
                tonic_ext::{is_receive_bound_refusal, is_the_engines_own_report, TonicStatusExt},
                DeadRendezvous,
            },
        },
    },
};

/// Which reader drains a pair's prefill stream: what the circuit breaker
/// samples where the stream fails under it, and so what Grand Central is
/// told there. A collected answer does not charge the engine's own report
/// of a failed request. A streamed answer charges every sample. A reader
/// is a type of no size: the drain holds nothing for it.
pub(crate) trait PrefillReader: Copy + Send {
    /// Whether the circuit breaker samples the failure `e` under this
    /// reader.
    fn samples(self, e: &tonic::Status) -> bool;
}

/// The reader of a collected answer.
#[derive(Clone, Copy)]
pub(crate) struct CollectedAnswer;

/// The reader of a streamed answer.
#[derive(Clone, Copy)]
pub(crate) struct StreamedAnswer;

impl PrefillReader for CollectedAnswer {
    #[inline]
    fn samples(self, e: &tonic::Status) -> bool {
        !is_the_engines_own_report(e)
    }
}

impl PrefillReader for StreamedAnswer {
    #[inline]
    fn samples(self, _e: &tonic::Status) -> bool {
        true
    }
}

/// Drain each Prefill sample through its terminal response, releasing only that
/// sample's guard. A fan-out's first `Complete` does not finish its siblings.
/// Keep the streams' abort-on-drop armed until Decode succeeds, as before.
/// Collectors continue to EOF to preserve their metadata collection behavior;
/// streaming handlers can stop once all dispatched Prefill samples complete.
/// Returns the prefix-cache hit each completed sample reported: the client's
/// prompt-side usage, which the decode side's responses are restamped with
/// (`ProtoStream::decode_side`); empty for a stream the execution stage
/// drained itself.
///
/// Grand Central hears a failure here and nothing else: each load still
/// held is told the failure as `reader` samples it. A sample's `Complete`
/// tells nothing: the dispatch told the prefill's good answer at its first
/// response.
pub(crate) async fn drain_prefill<S: FanoutChild>(
    stream: &mut S,
    mut guards: Vec<Option<PrefillLoadGuard>>,
    finish_at_complete: bool,
    reader: impl PrefillReader,
    mut on_complete: impl FnMut(ProtoGenerateComplete) + Send,
) -> Result<PrefillCachedTokens, tonic::Status> {
    let mut remaining = guards.len();
    let single_dispatch = remaining == 1;
    let mut cached_tokens = PrefillCachedTokens::default();
    while let Some(response) = stream.next_item().await {
        // A prefill side the engine retires with its rendezvous verdict is
        // counted under the rendezvous family before the failure propagates.
        let response = response.inspect_err(|status| {
            account_pd_side_failure(metrics_labels::WORKER_PREFILL, status);
            if reader.samples(status) {
                for guard in guards.iter().flatten() {
                    tell_prefill_failure(guard, status);
                }
            }
        })?;
        if let ProtoResponseVariant::Complete(complete) = response.into_response() {
            // Fan-out responses are restamped with their dispatch positions.
            // For a single dispatch, preserve the collector's existing guard
            // lifetime: EOF-based collectors release it when the stream ends.
            if !single_dispatch || finish_at_complete {
                let index = if single_dispatch {
                    0
                } else {
                    complete.index() as usize
                };
                if guards
                    .get_mut(index)
                    .is_some_and(|guard| guard.take().is_some())
                {
                    remaining -= 1;
                }
            }
            cached_tokens.note(&complete);
            on_complete(complete);
            if finish_at_complete && remaining == 0 {
                break;
            }
        }
    }
    Ok(cached_tokens)
}

/// The `Complete` frames of an execution result, one per choice (`n`), as
/// the non-streamed answer is built from. A single worker's stream is
/// collected to EOF; a disaggregated pair's prefill side is drained first and
/// its `input_logprobs` copied onto the decode responses when
/// `merge_logprobs` asks for them (chat with `logprobs: true`). The streams
/// are marked completed only once every side succeeded, so a failure still
/// aborts them on drop.
pub(crate) async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let all_responses = match execution_result {
        ExecutionResult::Single { mut stream, .. } => {
            let responses = collect_stream_responses(&mut stream, "Single", None, None).await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::PrefillDecode {
            prefill,
            decode,
            prefill_guards,
            ..
        } => collect_prefill_decode(prefill, *decode, prefill_guards, merge_logprobs).await?,
        // Embeddings have no `Complete` frame to collect.
        ExecutionResult::Embedding { .. } => {
            return Err(error::internal_error(
                "invalid_execution_mode",
                "Embedding result encountered in response collection",
            ));
        }
        // Batches are split into per-prompt results by the completion processor
        // before collection.
        ExecutionResult::Batch { .. } => {
            return Err(error::internal_error(
                "invalid_execution_mode",
                "Batch result encountered in response collection",
            ));
        }
    };

    if all_responses.is_empty() {
        return Err(error::internal_error(
            "no_responses_from_server",
            "No responses from server",
        ));
    }

    Ok(all_responses)
}

/// [`collect_responses`] for a disaggregated pair: the prefill side drained
/// through its `Complete`, then the decode side collected with the prefill's
/// prompt-side usage.
async fn collect_prefill_decode(
    mut prefill: ProtoStream,
    decode: ProtoStream,
    prefill_guards: Vec<Option<PrefillLoadGuard>>,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    // The side's worker, when its stream is tracked (the parallel dispatch
    // books the side on it and leaves the side's breaker sample to its reader):
    // a failure met here is the worker's, counted once; its breaker sample is
    // the dispatch loop's to take through the request's ledger, so a replayed
    // request charges the worker once. The engine's own report is not
    // charged, as on the decode side.
    let prefill_worker = prefill.worker().cloned();
    let mut prefill_responses = Vec::new();
    let prefill_cached_tokens = drain_prefill(
        &mut prefill,
        prefill_guards,
        false,
        CollectedAnswer,
        |complete| {
            prefill_responses.push(complete);
        },
    )
    .await
    .map_err(|status| {
        let mut failure = stream_failed(Some(PdSide::Prefill), "Prefill", &status, None);
        if let Some(worker) = &prefill_worker {
            if let Some(sample) = status
                .breaker_status()
                .filter(|_| !is_the_engines_own_report(&status))
            {
                mark_pending_charge(&mut failure, worker, sample);
            }
            UpstreamClass::account_collected_failure(worker.as_ref(), &status);
        }
        failure
    })?;

    // The client's prompt-side usage is the prefill side's.
    let mut decode_stream = decode.decode_side(prefill_cached_tokens);
    let mut decode_responses = collect_stream_responses(
        &mut decode_stream,
        "Decode",
        Some(PdSide::Decode),
        prefill_worker.as_ref(),
    )
    .await?;

    // Both sides succeeded: neither stream aborts on drop any more.
    prefill.mark_completed();
    decode_stream.mark_completed();

    if merge_logprobs {
        merge_prefill_logprobs(&prefill_responses, &mut decode_responses);
    }
    Ok(decode_responses)
}

/// Copy the prompt logprobs of the first prefill response onto every decode
/// response. Only the `Complete` variant that carries `input_logprobs` (the
/// one a disaggregated engine family sends) has them to give.
fn merge_prefill_logprobs(
    prefill_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) {
    if let Some(ProtoGenerateComplete::Sglang(prefill_first)) = prefill_responses.first() {
        if let Some(prefill_input_logprobs) = &prefill_first.input_logprobs {
            for response in decode_responses.iter_mut() {
                if let ProtoGenerateComplete::Sglang(decode_resp) = response {
                    decode_resp.input_logprobs = Some(prefill_input_logprobs.clone());
                }
            }
        }
    }
}

/// The client's answer when `worker_name`'s stream ended with `status`
/// before its `Complete`: the status's own HTTP class, under the code and
/// with the words that name the dead rendezvous port when the gateway found
/// one behind the failure; else, for a disaggregated `side`, the side's
/// stream-phase code of the shared vocabulary (`pd_failure`), the one the
/// HTTP prefill-decode router answers for the same failure; for a single
/// worker the code clients know. A worker's report that the engine could not
/// continue the request under its constraint is answered for the constraint.
/// The gateway's own request timeout, on any stream, is the timeout's one
/// answer (`to_http_error`), final for the retry layer.
fn stream_failed(
    side: Option<PdSide>,
    worker_name: &str,
    status: &tonic::Status,
    rendezvous: Option<&DeadRendezvous>,
) -> Response {
    // The engine could not continue the request under its constraint: the
    // request's answer, under its own code, final for the retry layer (see
    // `TonicStatusExt::constraint_failure_response`).
    if let Some(response) = status.constraint_failure_response() {
        return response;
    }
    match (rendezvous, side) {
        (Some(rendezvous), _) => status.to_http_error(
            "pd_rendezvous_failed",
            format!(
                "{worker_name} stream failed: {}; {rendezvous}",
                status.message()
            ),
        ),
        (None, Some(side)) if !is_request_timeout(status) => {
            let mut response =
                side.answer(SidePhase::Stream, status.http_status(), status.message());
            // An answer above the receive bound is as large from any
            // worker: final, as `to_http_error` makes it for a single one.
            if is_receive_bound_refusal(status) {
                mark_non_retryable(&mut response);
            }
            mark_error_reason(&mut response, status.error_reason());
            response
        }
        // A single worker's stream, or a side's the gateway's own request
        // timeout cut: `to_http_error` gives that timeout its one answer.
        (None, _) => status.to_http_error(
            "worker_stream_failed",
            format!("{worker_name} stream failed: {}", status.message()),
        ),
    }
}

/// Collect all `Complete` frames of a gRPC stream; chunks carry nothing a
/// collected answer needs. `side` names the disaggregated side the stream
/// belongs to, whose rendezvous failures are counted before they become the
/// client's answer: the engine's words by phase, and, for a decode side the
/// engine failed without a reason before anything was read from its stream,
/// the `prefill` side's bootstrap port probed now (see [`dead_rendezvous_of`]);
/// a single worker's stream has none. A
/// failed stream is not marked completed, so its drop sends the abort.
async fn collect_stream_responses<S: FanoutChild>(
    stream: &mut S,
    worker_name: &str,
    side: Option<PdSide>,
    prefill: Option<&Arc<dyn Worker>>,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let mut all_responses = Vec::new();
    // Whether the stream handed anything, a chunk included: `all_responses`
    // keeps the `Complete` frames alone.
    let mut read_any = false;

    while let Some(response) = stream.next_item().await {
        let gen_response = match response {
            Ok(response) => {
                read_any = true;
                response
            }
            Err(status) => {
                let rendezvous = match side {
                    Some(side) => {
                        account_pd_side_failure(side.name(), &status);
                        // Before anything was read from the decode stream, as the
                        // streamed path reads it before its first token: once a
                        // response came, a chunk included, the sides met.
                        if side == PdSide::Decode && !read_any {
                            dead_rendezvous_of(prefill, &status).await
                        } else {
                            None
                        }
                    }
                    None => None,
                };
                match &rendezvous {
                    Some(DeadRendezvous(port)) => trace_error!(
                        function = "collect_stream_responses",
                        worker = %worker_name,
                        grpc_code = ?status.code(),
                        error = ?status,
                        prefill = prefill.map_or("none", |prefill| prefill.redacted_url()),
                        bootstrap_host = port.host.as_str(),
                        bootstrap_port = port.port,
                        bootstrap_port_source = port.source.label(),
                        probe_error = %port.error,
                        "Worker stream error, and the prefill side's bootstrap port did not accept \
                         a connection: the sides could not rendezvous"
                    ),
                    None => trace_error!(
                        function = "collect_stream_responses",
                        worker = %worker_name,
                        grpc_code = ?status.code(),
                        error = ?status,
                        "Worker stream error"
                    ),
                }
                let mut failure = stream_failed(side, worker_name, &status, rendezvous.as_ref());
                // The worker's failure counts against it as a streamed
                // answer's does; the client's status and the router-error
                // family are the pipeline's. The breaker's sample is taken by
                // the dispatch loop through the request's ledger, so a replayed
                // request charges the worker once. The engine's own report is
                // replayed on the worker and not charged.
                if let Some(worker) = stream.worker() {
                    UpstreamClass::account_collected_failure(worker.as_ref(), &status);
                    if let Some(sample) = status
                        .breaker_status()
                        .filter(|_| !is_the_engines_own_report(&status))
                    {
                        mark_pending_charge(&mut failure, worker, sample);
                    }
                }
                return Err(failure);
            }
        };
        if let ProtoResponseVariant::Complete(complete) = gen_response.into_response() {
            all_responses.push(complete);
        }
    }

    Ok(all_responses)
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/common/response_collection/tests.rs"]
mod tests;

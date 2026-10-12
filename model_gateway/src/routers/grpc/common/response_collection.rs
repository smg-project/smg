//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;
use tracing::error as trace_error;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        error,
        grpc::{
            context::ExecutionResult,
            proto_wrapper::{
                FanoutChild, PrefillCachedTokens, ProtoGenerateComplete, ProtoResponseVariant,
            },
            utils::{
                account_pd_leg_failure, stream_failure::UpstreamClass, tonic_ext::TonicStatusExt,
            },
        },
    },
    worker::PrefillLoadGuard,
};

/// Drain each Prefill sample through its terminal response, releasing only that
/// sample's guard. A fan-out's first `Complete` does not finish its siblings.
/// Keep the streams' abort-on-drop armed until Decode succeeds, as before.
/// Collectors continue to EOF to preserve their metadata collection behavior;
/// streaming handlers can stop once all dispatched Prefill samples complete.
/// Returns the prefix-cache hit each completed sample reported: the client's
/// prompt-side usage, which the decode leg's responses are restamped with
/// (`ProtoStream::decode_leg`); empty for a stream the execution stage
/// drained itself.
pub(crate) async fn drain_prefill<S: FanoutChild>(
    stream: &mut S,
    mut guards: Vec<Option<PrefillLoadGuard>>,
    finish_at_complete: bool,
    mut on_complete: impl FnMut(ProtoGenerateComplete) + Send,
) -> Result<PrefillCachedTokens, tonic::Status> {
    let mut remaining = guards.len();
    let single_dispatch = remaining == 1;
    let mut cached_tokens = PrefillCachedTokens::default();
    while let Some(response) = stream.next_item().await {
        // A prefill leg the engine retires with its rendezvous verdict is
        // counted under the rendezvous family before the failure propagates.
        let response = response
            .inspect_err(|status| account_pd_leg_failure(metrics_labels::WORKER_PREFILL, status))?;
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
                if let Some(guard) = guards.get_mut(index) {
                    if guard.take().is_some() {
                        remaining -= 1;
                    }
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

/// Collect and merge responses from execution result
///
/// Handles both Single and PrefillDecode execution modes.
/// For PrefillDecode mode, merges prefill input_logprobs into decode responses if requested.
///
/// # Arguments
/// * `execution_result` - The execution result containing stream(s)
/// * `merge_logprobs` - Whether to merge prefill input_logprobs (for chat with logprobs=true)
///
/// # Returns
/// Vector of GenerateComplete responses, one per index (n parameter)
pub(crate) async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let all_responses = match execution_result {
        ExecutionResult::Single { mut stream } => {
            let responses = collect_stream_responses(&mut stream, "Single", None).await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::PrefillDecode {
            mut prefill,
            decode,
            prefill_guards,
            ..
        } => {
            // Collect prefill for input_logprobs (don't mark completed yet)
            let mut prefill_responses = Vec::new();
            // The leg's worker, when its stream is tracked (the parallel
            // dispatch books the leg on it and leaves the leg's breaker
            // sample to its reader): a failure met here is the worker's,
            // sampled on its breaker and counted once, as a streamed
            // answer's reader does it.
            let prefill_worker = prefill.worker().cloned();
            let prefill_cached_tokens =
                drain_prefill(&mut prefill, prefill_guards, false, |complete| {
                    prefill_responses.push(complete);
                })
                .await
                .map_err(|e| {
                    if let Some(worker) = &prefill_worker {
                        if let Some(sample) = UpstreamClass::of(&e).breaker_sample {
                            worker.record_outcome(sample);
                        }
                        UpstreamClass::account_collected_failure(worker.as_ref(), &e);
                    }
                    e.to_http_error(
                        "worker_stream_failed",
                        format!("Prefill stream failed: {}", e.message()),
                    )
                })?;

            // Collect decode for actual output (don't mark completed yet); the
            // client's prompt-side usage is the prefill leg's.
            let mut decode_stream = (*decode).decode_leg(prefill_cached_tokens);
            let mut decode_responses = collect_stream_responses(
                &mut decode_stream,
                "Decode",
                Some(metrics_labels::WORKER_DECODE),
            )
            .await?;

            // Mark both streams as completed now that both succeeded
            prefill.mark_completed();
            decode_stream.mark_completed();

            // Merge prefill input_logprobs if requested
            if merge_logprobs {
                merge_prefill_logprobs(&prefill_responses, &mut decode_responses);
            }

            decode_responses
        }
        ExecutionResult::Embedding { .. } => {
            // Embeddings do not support this path (no generate complete response)
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

/// Merge prefill input_logprobs into decode responses
///
/// Takes input_logprobs from the first prefill response and copies them
/// into all decode responses. This is used in PD mode when logprobs are requested.
/// Only works with SGLang (vLLM doesn't support PD mode).
fn merge_prefill_logprobs(
    prefill_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) {
    // Only SGLang supports PD mode and has input_logprobs
    if let Some(ProtoGenerateComplete::Sglang(prefill_first)) = prefill_responses.first() {
        // Use ref to borrow input_logprobs instead of cloning upfront
        // This avoids one allocation when the Option is Some
        if let Some(ref prefill_input_logprobs) = prefill_first.input_logprobs {
            for response in decode_responses.iter_mut() {
                if let ProtoGenerateComplete::Sglang(decode_resp) = response {
                    decode_resp.input_logprobs = Some(prefill_input_logprobs.clone());
                }
            }
        }
    }
}

/// Collect all complete responses from a gRPC stream, discarding chunks.
/// `leg` names the disaggregated leg the stream belongs to, whose rendezvous
/// failures are counted before they become the client's answer; a single
/// worker's stream has none.
async fn collect_stream_responses<S: FanoutChild>(
    stream: &mut S,
    worker_name: &str,
    leg: Option<&'static str>,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let mut all_responses = Vec::new();

    while let Some(response) = stream.next_item().await {
        match response {
            Ok(gen_response) => {
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        all_responses.push(complete);
                    }
                    ProtoResponseVariant::Chunk(_chunk) => {
                        // Streaming chunk - no action needed
                    }
                    ProtoResponseVariant::None => {
                        // Empty response - no action needed
                    }
                }
            }
            Err(e) => {
                if let Some(leg) = leg {
                    account_pd_leg_failure(leg, &e);
                }
                trace_error!(function = "collect_stream_responses", worker = %worker_name, grpc_code = ?e.code(), error = ?e, "Worker stream error");
                // The worker's failure counts against it as a streamed
                // answer's does; the client's status and the router-error
                // family are the pipeline's.
                if let Some(worker) = stream.worker() {
                    UpstreamClass::account_collected_failure(worker.as_ref(), &e);
                }
                // Don't mark as completed - let Drop send abort for error cases
                return Err(e.to_http_error(
                    "worker_stream_failed",
                    format!("{worker_name} stream failed: {}", e.message()),
                ));
            }
        }
    }

    Ok(all_responses)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;
    use crate::{
        routers::grpc::proto_wrapper::{ProtoGenerateResponse, ENGINE_ERROR_MESSAGE},
        worker::{BasicWorkerBuilder, ConnectionMode, Worker, WorkerType},
    };

    /// A stream that is already at EOF, as sequential PD hands to the
    /// streaming layer after draining prefill in the execution stage.
    struct ExhaustedStream;

    impl FanoutChild for ExhaustedStream {
        async fn next_item(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
            None
        }

        fn mark_completed(&mut self) {}

        fn defer_abort_until_first_item(self) -> Self {
            self
        }
    }

    /// A stream the engine ends with `status` before any response.
    struct FailingStream(Option<tonic::Status>);

    impl FanoutChild for FailingStream {
        async fn next_item(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
            self.0.take().map(Err)
        }

        fn mark_completed(&mut self) {}

        fn defer_abort_until_first_item(self) -> Self {
            self
        }
    }

    #[tokio::test]
    async fn a_prefill_leg_retired_at_its_rendezvous_is_counted_while_draining() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let status = tonic::Status::internal(
            "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41 \
             with exception KVTransferError(bootstrap_room=41): Request 41 timed out after \
             120.0s in KVPoll.Bootstrapping",
        );
        let result = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            drain_prefill(
                &mut FailingStream(Some(status)),
                vec![None],
                true,
                |_complete| {},
            )
            .await
        };
        assert_eq!(
            result.expect_err("the failure propagates").code(),
            tonic::Code::Internal
        );
        let rendered = handle.render();
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_pd_rendezvous_failures_total{")
                    && line.contains(r#"leg="prefill""#)
                    && line.contains(r#"phase="bootstrap""#)
                    && line.ends_with(" 1")
            }),
            "{rendered}"
        );
    }

    /// The decode leg of a non-streamed request (every unary chat,
    /// completion, generate and Harmony request collects it here) that the
    /// engine retired at its handshake is counted, as the streaming path's
    /// decode leg is; a single worker's stream ending the same way is not a
    /// rendezvous failure.
    #[tokio::test]
    async fn a_decode_leg_retired_at_its_handshake_is_counted_while_collecting() {
        let status = tonic::Status::internal(
            "Decode handshake failed for request rank=0 req.rid='r2' req.bootstrap_room=42 \
             with exception KVTransferError(bootstrap_room=42): Request 42 timed out after \
             120.0s in KVPoll.Bootstrapping",
        );
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (decode, single) = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            (
                collect_stream_responses(
                    &mut FailingStream(Some(status.clone())),
                    "Decode",
                    Some(metrics_labels::WORKER_DECODE),
                )
                .await,
                collect_stream_responses(&mut FailingStream(Some(status)), "Single", None).await,
            )
        };
        assert!(decode.is_err(), "the failure becomes the answer");
        assert!(single.is_err(), "the failure becomes the answer");
        let rendered = handle.render();
        let counted: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            .collect();
        assert_eq!(counted.len(), 1, "{rendered}");
        assert!(
            counted[0].contains(r#"leg="decode""#)
                && counted[0].contains(r#"phase="bootstrap""#)
                && counted[0].ends_with(" 1"),
            "{rendered}"
        );
    }

    /// A leg's stream failure read into a non-streamed answer is counted
    /// under the leg's own labels, and only when it is the worker's failure.
    #[test]
    fn a_collected_leg_failure_counts_against_the_legs_worker_when_it_is_the_workers_failure() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode-under-test:30000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        {
            let _guard = metrics::set_default_local_recorder(&recorder);
            UpstreamClass::account_collected_failure(
                decode.as_ref(),
                &tonic::Status::internal(ENGINE_ERROR_MESSAGE),
            );
            UpstreamClass::account_collected_failure(
                decode.as_ref(),
                &tonic::Status::unavailable("gone"),
            );
            // Capacity pushback and the request's own fault are not the worker's.
            UpstreamClass::account_collected_failure(
                decode.as_ref(),
                &tonic::Status::resource_exhausted("full"),
            );
            UpstreamClass::account_collected_failure(
                decode.as_ref(),
                &tonic::Status::invalid_argument("bad"),
            );
        }
        let rendered = handle.render();
        let decode_lines: Vec<&str> = rendered
            .lines()
            .filter(|line| {
                line.starts_with("smg_worker_errors_total{")
                    && line.contains(r#"worker_type="decode""#)
                    && line.contains(r#"connection_mode="grpc""#)
            })
            .collect();
        assert_eq!(decode_lines.len(), 2, "{rendered}");
        for (error_type, count) in [("engine_error", " 1"), ("backend_error", " 1")] {
            let line = decode_lines
                .iter()
                .find(|line| line.contains(&format!(r#"error_type="{error_type}""#)))
                .unwrap_or_else(|| panic!("{error_type} is counted:\n{rendered}"));
            assert!(line.ends_with(count), "{line}");
        }
    }

    #[tokio::test]
    async fn drain_prefill_accepts_exhausted_stream_with_released_guard() {
        // Sequential PD releases the prefill guard in the execution stage, so
        // the streaming layer receives an exhausted stream with `vec![None]`.
        let mut stream = ExhaustedStream;
        let completions = AtomicUsize::new(0);
        let result = drain_prefill(&mut stream, vec![None], true, |_complete| {
            completions.fetch_add(1, Ordering::SeqCst);
        })
        .await;
        assert!(result.is_ok());
        assert_eq!(completions.load(Ordering::SeqCst), 0);
    }
}

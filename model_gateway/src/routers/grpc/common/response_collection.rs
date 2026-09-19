//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;
use tracing::error as trace_error;

use crate::{
    routers::{
        error,
        grpc::{
            context::ExecutionResult,
            proto_wrapper::{
                FanoutChild, ProtoGenerateComplete, ProtoResponseVariant, ProtoStream,
            },
            utils::tonic_ext::TonicStatusExt,
        },
    },
    worker::PrefillLoadGuard,
};

/// Drain each Prefill sample through its terminal response, releasing only that
/// sample's guard. A fan-out's first `Complete` does not finish its siblings.
/// Keep the streams' abort-on-drop armed until Decode succeeds, as before.
/// Collectors continue to EOF to preserve their metadata collection behavior;
/// streaming handlers can stop once all dispatched Prefill samples complete.
pub(crate) async fn drain_prefill<S: FanoutChild>(
    stream: &mut S,
    mut guards: Vec<Option<PrefillLoadGuard>>,
    finish_at_complete: bool,
    mut on_complete: impl FnMut(ProtoGenerateComplete) + Send,
) -> Result<(), tonic::Status> {
    let mut remaining = guards.len();
    let single_dispatch = remaining == 1;
    while let Some(response) = stream.next_item().await {
        if let ProtoResponseVariant::Complete(complete) = response?.into_response() {
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
            on_complete(complete);
            if finish_at_complete && remaining == 0 {
                break;
            }
        }
    }
    Ok(())
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
            let responses = collect_stream_responses(&mut stream, "Single").await?;
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
            drain_prefill(&mut prefill, prefill_guards, false, |complete| {
                prefill_responses.push(complete);
            })
            .await
            .map_err(|e| {
                e.to_http_error(
                    "worker_stream_failed",
                    format!("Prefill stream failed: {}", e.message()),
                )
            })?;

            // Collect decode for actual output (don't mark completed yet)
            let mut decode_stream = *decode;
            let mut decode_responses =
                collect_stream_responses(&mut decode_stream, "Decode").await?;

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
async fn collect_stream_responses(
    stream: &mut ProtoStream,
    worker_name: &str,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let mut all_responses = Vec::new();

    while let Some(response) = stream.next().await {
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
                trace_error!(function = "collect_stream_responses", worker = %worker_name, grpc_code = ?e.code(), error = ?e, "Worker stream error");
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

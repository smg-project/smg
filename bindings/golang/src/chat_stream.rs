//! The chat request path the single client and the policy client share: the
//! caller's request read, rendered and tokenized, then sent to one engine
//! with the converter that turns its stream into the client's chunks. The
//! entry points keep what differs around it: which engine takes the request,
//! and the load a request holds on the worker a policy selected.

use std::{future::Future, os::raw::c_char, sync::Arc, time::Duration};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::chat::ChatCompletionRequest;
use smg::routers::grpc::utils::{chat_reasoning_starts_in_rendered_prefill, process_chat_messages};
use smg_grpc_client::{
    engine_labels::ServerInfo,
    sglang_proto::GetServerInfoResponse,
    sglang_scheduler::{SglangGenerateRequestOptions, SglangSchedulerClient},
    FormatSetAside,
};
use smg_worker::labels::SUPPORTS_THINKING_TOKEN_BUDGET_LABEL;
use uuid::Uuid;

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::GrpcResponseConverterHandle,
    policy::HeldLoad,
    runtime::RUNTIME,
    stream::SglangStreamHandle,
    tool_constraint::{tool_constraint_for, tools_keep_special_tokens},
};

/// Bound metadata reads for the client's fingerprint and the worker's capabilities.
const ENGINE_INFO_BOUND: Duration = Duration::from_secs(2);

/// A legacy worker ignores the new field. Check the selected endpoint only for finite budgets.
fn verify_thinking_budget<E, F: Future<Output = Result<GetServerInfoResponse, E>>>(
    budget: Option<u32>,
    lookup: impl FnOnce() -> F,
) -> Result<(), (SglErrorCode, &'static str)> {
    if budget.is_none() {
        return Ok(());
    }
    let info = RUNTIME
        .block_on(async { tokio::time::timeout(ENGINE_INFO_BOUND, lookup()).await })
        .ok()
        .and_then(Result::ok)
        .ok_or((
            SglErrorCode::UnknownError,
            "Could not verify worker support for thinking_token_budget",
        ))?;
    if ServerInfo::Sglang(Box::new(info))
        .to_labels()
        .get(SUPPORTS_THINKING_TOKEN_BUDGET_LABEL)
        .is_some_and(|value| value == "true")
    {
        Ok(())
    } else {
        Err((
            SglErrorCode::InvalidArgument,
            "Selected worker does not advertise thinking_token_budget support",
        ))
    }
}

/// The weight version the engine reports in its model information, the
/// value the gateway puts in `system_fingerprint`: the gateway's fallback,
/// `default`, when the engine reports a blank one, answers with an error, or
/// does not answer within the bound (both logged, so an absent version is
/// diagnosable).
pub(crate) async fn weight_version_of(client: &SglangSchedulerClient) -> String {
    match tokio::time::timeout(ENGINE_INFO_BOUND, client.get_model_info()).await {
        Ok(Ok(info)) => fingerprint_of(Some(&info.weight_version)),
        Ok(Err(status)) => {
            tracing::warn!(%status, "model info lookup failed; system_fingerprint is the default");
            fingerprint_of(None)
        }
        Err(_) => {
            tracing::warn!(
                ?ENGINE_INFO_BOUND,
                "model info lookup timed out; system_fingerprint is the default"
            );
            fingerprint_of(None)
        }
    }
}

/// The `system_fingerprint` for a weight version the engine reported: the
/// version itself when it is not blank, else the gateway's fallback,
/// `default` (what the gateway sends for a worker without a version).
pub(crate) fn fingerprint_of(weight_version: Option<&str>) -> String {
    match weight_version.map(str::trim) {
        Some(version) if !version.is_empty() => version.to_string(),
        _ => "default".to_string(),
    }
}

/// A chat request read from the caller's JSON, its prompt rendered by the
/// tokenizer's chat template and tokenized.
pub(crate) struct PreparedChat {
    pub(crate) request: ChatCompletionRequest,
    /// The rendered prompt.
    pub(crate) text: String,
    pub(crate) token_ids: Vec<u32>,
    /// What the rendered prompt did, which the reasoning decision reads
    /// (`ProcessedMessages`): whether it carries the generation header and
    /// whether it continues the final message.
    pub(crate) generation_prompt: bool,
    pub(crate) continued_final_message: bool,
}

/// Read, render and tokenize the caller's request; a failure is the entry
/// point's error code and message.
pub(crate) fn prepare_chat(
    request_json: &str,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedChat, (SglErrorCode, String)> {
    let request: ChatCompletionRequest = serde_json::from_str(request_json).map_err(|e| {
        (
            SglErrorCode::ParsingError,
            format!("Failed to parse request JSON: {e}"),
        )
    })?;
    if request
        .resolved_thinking_token_budget()
        .map_err(|err| (SglErrorCode::InvalidArgument, err.to_string()))?
        .is_some()
    {
        return Err((
            SglErrorCode::InvalidArgument,
            "This SGLang connection cannot enforce a finite thinking_token_budget".to_string(),
        ));
    }
    let processed = process_chat_messages(&request, tokenizer, None).map_err(|e| {
        (
            SglErrorCode::TokenizationError,
            format!("Failed to process messages: {e}"),
        )
    })?;
    let token_ids = tokenizer
        .encode(&processed.text, false)
        .map_err(|e| {
            (
                SglErrorCode::TokenizationError,
                format!("Failed to tokenize: {e}"),
            )
        })?
        .token_ids()
        .to_vec();
    Ok(PreparedChat {
        request,
        text: processed.text,
        token_ids,
        generation_prompt: processed.generation_prompt,
        continued_final_message: processed.continued_final_message,
    })
}

/// Send a prepared request to `client` and return the handle its stream is
/// read through: the tool constraint and the special-token decision, the
/// generate request, the stream opened on the runtime, and the converter
/// over the request's tools, tool choice and stops, seeded with the prompt's
/// token count. `load` is the load the request holds on the worker a policy
/// selected (the policy client; the single client holds none): the stream
/// carries it until the caller frees the handle, and a failure here drops
/// it, which gives it back. `weight_version` is the engine's, read when the
/// client was built (the gateway's `default` when it reported none); the
/// chunks carry it as `system_fingerprint`, as the gateway's do. A failure
/// leaves the entry point's message in `error_out` and returns its code.
///
/// # Safety
/// - `error_out` may be null; if non-null, must point to writable memory
pub(crate) unsafe fn start_chat_stream(
    client: &Arc<SglangSchedulerClient>,
    tokenizer: &Arc<dyn Tokenizer>,
    prepared: PreparedChat,
    load: Option<HeldLoad>,
    weight_version: &str,
    error_out: *mut *mut c_char,
) -> Result<SglangStreamHandle, SglErrorCode> {
    let PreparedChat {
        mut request,
        text,
        token_ids,
        generation_prompt,
        continued_final_message,
    } = prepared;
    let prompt_tokens = token_ids.len() as u32;

    let tool_constraint = match tool_constraint_for(&request) {
        Ok(constraint) => constraint,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to generate tool constraints: {e}"),
            );
            return Err(SglErrorCode::ParsingError);
        }
    };
    // The engine keeps its special tokens when a parser will read them:
    // tools at either level, a choice that allows calls, no JSON schema.
    if tools_keep_special_tokens(&request, tool_constraint.as_ref()) {
        request.skip_special_tokens = false;
    }

    let request_id = format!("chatcmpl-{}", Uuid::now_v7());
    // Decided from the rendered prompt, as the gateway decides it: a renderer
    // with an encoder of its own writes the generation header whatever the
    // request asked, and its prompt starts the reasoning as before.
    let require_reasoning = chat_reasoning_starts_in_rendered_prefill(
        &request,
        generation_prompt,
        continued_final_message,
        tokenizer.as_ref(),
    );
    // The readers replay the rendered prompt, as the gateway's readers do:
    // the converter keeps the part they replay, taken before the request
    // takes the text.
    let mut converter = GrpcResponseConverterHandle::for_chat(
        Arc::clone(tokenizer),
        &request,
        request_id.clone(),
        &text,
    );
    let proto_request = match client.build_generate_request_from_chat(
        request_id,
        &request,
        text,
        token_ids,
        SglangGenerateRequestOptions {
            multimodal_inputs: None, // multimodal not supported in golang bindings
            tool_call_constraint: tool_constraint,
            // No table answers the body's format here: the engine reads it as sent.
            format_set_aside: FormatSetAside::default(),
            require_reasoning,
        },
    ) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to build generate request: {e}"));
            return Err(SglErrorCode::ParsingError);
        }
    };
    let budget = proto_request
        .sampling_params
        .as_ref()
        .and_then(|params| params.thinking_token_budget);
    if let Err((code, message)) = verify_thinking_budget(budget, || client.get_server_info()) {
        set_error_message(error_out, message);
        return Err(code);
    }
    let stream = match RUNTIME.block_on(client.generate(proto_request)) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to send request: {e}"));
            return Err(SglErrorCode::UnknownError);
        }
    };

    converter.initial_prompt_tokens = Some(prompt_tokens);
    converter.system_fingerprint = Some(weight_version.to_owned());
    Ok(SglangStreamHandle {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        converter: Arc::new(tokio::sync::Mutex::new(converter)),
        load,
    })
}

#[cfg(test)]
#[path = "../tests/unit/chat_stream/tests.rs"]
mod tests;

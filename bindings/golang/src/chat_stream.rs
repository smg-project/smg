//! The chat request path the single client and the policy client share: the
//! caller's request read, rendered and tokenized, then sent to one engine
//! with the converter that turns its stream into the client's chunks. The
//! entry points keep what differs around it: which engine takes the request,
//! and the load a stream holds on the worker a policy selected.

use std::{os::raw::c_char, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::chat::ChatCompletionRequest;
use smg::routers::grpc::utils::process_chat_messages;
use smg_grpc_client::sglang_scheduler::{SglangGenerateRequestOptions, SglangSchedulerClient};
use uuid::Uuid;

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::GrpcResponseConverterHandle,
    policy::GrpcWorker,
    runtime::RUNTIME,
    stream::SglangStreamHandle,
    tool_constraint::{tool_constraint_for, tools_keep_special_tokens},
    utils::chat_requires_reasoning,
};

/// A chat request read from the caller's JSON, its prompt rendered by the
/// tokenizer's chat template and tokenized.
pub(crate) struct PreparedChat {
    pub(crate) request: ChatCompletionRequest,
    /// The rendered prompt.
    pub(crate) text: String,
    pub(crate) token_ids: Vec<u32>,
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
    })
}

/// Send a prepared request to `client` and return the handle its stream is
/// read through: the tool constraint and the special-token decision, the
/// generate request, the stream opened on the runtime, and the converter
/// over the request's tools, tool choice and stops, seeded with the prompt's
/// token count. `worker` is the selected worker whose load the stream holds
/// (the policy client); the single client passes none. A failure leaves the
/// entry point's message in `error_out` and returns its code.
///
/// # Safety
/// - `error_out` may be null; if non-null, must point to writable memory
pub(crate) unsafe fn start_chat_stream(
    client: &Arc<SglangSchedulerClient>,
    tokenizer: &Arc<dyn Tokenizer>,
    prepared: PreparedChat,
    worker: Option<Arc<GrpcWorker>>,
    error_out: *mut *mut c_char,
) -> Result<SglangStreamHandle, SglErrorCode> {
    let PreparedChat {
        mut request,
        text,
        token_ids,
    } = prepared;
    let prompt_tokens = token_ids.len() as u32;

    let tool_constraint = match tool_constraint_for(&request) {
        Ok(constraint) => constraint.map(|c| c.to_tuple()),
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
    let require_reasoning = chat_requires_reasoning(&request, tokenizer.as_ref());
    let proto_request = match client.build_generate_request_from_chat(
        request_id.clone(),
        &request,
        text,
        token_ids,
        SglangGenerateRequestOptions {
            multimodal_inputs: None, // multimodal not supported in golang bindings
            tool_call_constraint: tool_constraint,
            require_reasoning,
        },
    ) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to build generate request: {e}"));
            return Err(SglErrorCode::ParsingError);
        }
    };
    let stream = match RUNTIME.block_on(client.generate(proto_request)) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to send request: {e}"));
            return Err(SglErrorCode::UnknownError);
        }
    };

    let mut converter =
        GrpcResponseConverterHandle::for_chat(Arc::clone(tokenizer), &request, request_id);
    converter.initial_prompt_tokens = Some(prompt_tokens);
    Ok(SglangStreamHandle {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        converter: Arc::new(tokio::sync::Mutex::new(converter)),
        client: Arc::clone(client),
        prompt_tokens,
        worker,
    })
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::create_tokenizer_from_file;
    use serde_json::json;

    use super::*;

    /// The test tokenizer: the renderer that writes one `role: content` line
    /// per message.
    fn tokenizer() -> Arc<dyn Tokenizer> {
        create_tokenizer_from_file("mock").unwrap()
    }

    #[test]
    fn a_request_is_rendered_and_tokenized() {
        let request = json!({
            "model": "unmapped-model",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let tokenizer = tokenizer();
        let prepared = prepare_chat(&request.to_string(), tokenizer.as_ref()).unwrap();
        assert_eq!(prepared.request.model, "unmapped-model");
        assert!(prepared.text.contains("hi"), "{}", prepared.text);
        assert_eq!(
            prepared.token_ids,
            tokenizer.encode(&prepared.text, false).unwrap().token_ids()
        );
    }

    /// A request that is not JSON, or not a chat request, is the parsing
    /// error of the entry points, worded as before.
    #[test]
    fn a_request_that_does_not_parse_is_a_parsing_error() {
        for body in ["not json", r#"{"model": "m"}"#] {
            let (code, message) = prepare_chat(body, tokenizer().as_ref())
                .err()
                .expect("a parse error");
            assert_eq!(code, SglErrorCode::ParsingError, "{body}");
            assert!(
                message.starts_with("Failed to parse request JSON: "),
                "{message}"
            );
        }
    }
}

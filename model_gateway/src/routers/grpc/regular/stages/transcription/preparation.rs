//! Transcription preparation: resolve the family, validate capabilities,
//! synthesize the chat-shaped backend request, and run the shared chat
//! preparation (template render + multimodal audio expansion + tokenize).

use async_trait::async_trait;
use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use llm_multimodal::registry::qwen3_asr::transcription::TranscriptionFamily;
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{ContentPart, InputAudio},
    transcription::{AudioFile, TranscriptionRequest},
};
use tracing::error;

use super::{audio_format, parse_response_format, resolve_family, supported_families};
use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{PreparationOutput, RequestContext},
        regular::stages::chat::prepare_chat_like,
    },
};

/// Transcription preparation stage.
pub(crate) struct TranscriptionPreparationStage;

#[async_trait]
impl PipelineStage for TranscriptionPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let (request, audio) = ctx.transcription_input_arc();

        // Resolve the family serving this model (fail-closed).
        let family = resolve_family(&ctx.components.worker_registry, &ctx.input.model_id)
            .ok_or_else(|| {
                error::bad_request(
                    "audio_transcription_model_not_supported",
                    format!(
                        "The gRPC transcription endpoint currently supports {} only",
                        supported_families()
                    ),
                )
            })?;

        // Capability gate + response format resolution.
        if request.stream.unwrap_or(false) && !family.supports_streaming() {
            return Err(error::bad_request(
                "streaming_transcription_not_supported",
                format!("{} supports whole-file transcription only", family.name()),
            ));
        }
        if request
            .timestamp_granularities
            .as_ref()
            .is_some_and(|values| !values.is_empty())
            && !family.supports_timestamps()
        {
            return Err(error::bad_request(
                "transcription_timestamps_not_supported",
                format!("{} does not produce timestamps", family.name()),
            ));
        }
        let format = parse_response_format(request.response_format.as_deref()).map_err(|e| *e)?;

        // Synthesize the chat-shaped backend request from the family's prompt
        // convention. This is the audio->chat construction that used to live
        // in the router; it now runs inside the pipeline.
        let chat_request = build_chat_request(&request, &audio, family)?;

        // Run the shared chat preparation (template + multimodal audio
        // expansion + tokenize + stop decoder), then store the transcription
        // variant carrying the synthesized request and response contract.
        let (token_ids, processed_messages, _tool_constraints) =
            prepare_chat_like(ctx, &chat_request).await?;

        ctx.state.preparation = Some(PreparationOutput::Transcription {
            token_ids,
            processed_messages,
            chat_request: std::sync::Arc::new(chat_request),
            format,
            family,
        });
        Ok(())
    }

    fn name(&self) -> &'static str {
        "TranscriptionPreparation"
    }
}

/// Build the chat-shaped request for one audio file: an optional sanitized
/// system prompt, the audio as a user turn, and — when a language is given —
/// an assistant continuation that forces the transcript (via the generic
/// `continue_final_message` prefill). Greedy, single-choice, whole-file.
fn build_chat_request(
    body: &TranscriptionRequest,
    audio: &AudioFile,
    family: &dyn TranscriptionFamily,
) -> Result<ChatCompletionRequest, Response> {
    let mut messages = Vec::with_capacity(3);

    if let Some(prompt) = body.prompt.as_deref() {
        let prompt = family
            .sanitize_prompt(prompt.to_string())
            .map_err(|too_long| {
                error::bad_request(
                    "asr_prompt_too_long",
                    format!("prompt must not exceed {} bytes", too_long.max_bytes),
                )
            })?;
        let prompt = prompt.trim();
        if !prompt.is_empty() {
            messages.push(ChatMessage::System {
                content: MessageContent::Text(prompt.to_string()),
                name: None,
            });
        }
    }

    messages.push(ChatMessage::User {
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: BASE64_STANDARD.encode(&audio.bytes),
                format: audio_format(audio),
            },
        }]),
        name: None,
    });

    let prefill = family
        .assistant_prefill(body.language.as_deref())
        .map_err(|unsupported| {
            error!(function = "TranscriptionPreparation", language = %unsupported.0, "unsupported transcription language");
            error::bad_request(
                "unsupported_transcription_language",
                format!("{} does not support language '{}'", family.name(), unsupported.0),
            )
        })?;
    let continue_final_message = prefill.is_some();
    if let Some(prefill) = prefill {
        messages.push(ChatMessage::Assistant {
            content: Some(MessageContent::Text(prefill)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        });
    }

    Ok(ChatCompletionRequest {
        messages,
        model: body.model.clone(),
        n: Some(1),
        stream: false,
        temperature: Some(body.temperature.unwrap_or(0.0)),
        continue_final_message,
        skip_special_tokens: true,
        separate_reasoning: false,
        stream_reasoning: false,
        ..Default::default()
    })
}

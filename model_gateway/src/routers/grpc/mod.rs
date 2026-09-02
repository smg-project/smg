//! gRPC router implementations

use axum::response::Response;
use openai_protocol::{chat::ChatCompletionRequest, common::StringOrArray};

use crate::routers::error;

pub mod backend_client; // Polymorphism over gRPC vs ZMQ transports
pub mod client; // Used by core/
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod harmony;
pub(crate) mod mode;
pub(crate) mod multimodal;
pub(crate) mod pipeline;
pub(crate) mod proto_wrapper;
pub(crate) mod regular;
pub(crate) mod router; // Used by routers/factory
pub(crate) mod spec;
pub(crate) mod transcription;
pub mod utils; // Used by routers/http and bindings/golang
pub mod zmq_client; // ZMQ backend adapter behind the vLLM client surface
pub(crate) mod zmq_multimodal; // Proto mm inputs → EngineCore mm_features

// Re-export for convenience
pub use proto_wrapper::{MultimodalData, TensorBytes};

fn validate_text_only_output(request: &ChatCompletionRequest) -> Result<(), Box<Response>> {
    if request.return_audio == Some(true) {
        return Err(Box::new(error::bad_request(
            "audio_output_not_supported",
            "'return_audio' must be false because the gRPC backend only supports text output",
        )));
    }

    if let Some(modality) = request.modalities.as_ref().and_then(|modalities| {
        modalities
            .iter()
            .find(|modality| modality.as_str() != "text")
    }) {
        return Err(Box::new(error::bad_request(
            "unsupported_output_modality",
            format!(
                "unsupported output modality '{modality}'; the gRPC backend only supports text output"
            ),
        )));
    }

    Ok(())
}

/// Processed chat messages ready for gRPC generation.
///
/// The multimodal intermediate lives on `ProcessingState`, not here.
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub stop_sequences: Option<StringOrArray>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::error::extract_error_code_from_response;

    #[test]
    fn validates_grpc_text_only_output() {
        for modalities in [
            None,
            Some(vec![]),
            Some(vec!["text".to_string()]),
            Some(vec!["text".to_string(), "text".to_string()]),
        ] {
            let request = ChatCompletionRequest {
                modalities,
                ..Default::default()
            };
            assert!(validate_text_only_output(&request).is_ok());
        }

        let audio_request = ChatCompletionRequest {
            return_audio: Some(true),
            ..Default::default()
        };
        let response = validate_text_only_output(&audio_request).unwrap_err();
        assert_eq!(
            extract_error_code_from_response(&response),
            "audio_output_not_supported"
        );

        let modality_request = ChatCompletionRequest {
            modalities: Some(vec!["text".to_string(), "audio".to_string()]),
            ..Default::default()
        };
        let response = validate_text_only_output(&modality_request).unwrap_err();
        assert_eq!(
            extract_error_code_from_response(&response),
            "unsupported_output_modality"
        );
    }
}

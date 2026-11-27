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

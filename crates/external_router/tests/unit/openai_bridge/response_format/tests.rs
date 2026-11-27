use super::*;

#[test]
fn test_response_format_serde() {
    let formats = vec![
        (ResponseFormat::Passthrough, "\"passthrough\""),
        (ResponseFormat::WebSearchCall, "\"web_search_call\""),
        (
            ResponseFormat::CodeInterpreterCall,
            "\"code_interpreter_call\"",
        ),
        (ResponseFormat::FileSearchCall, "\"file_search_call\""),
        (
            ResponseFormat::ImageGenerationCall,
            "\"image_generation_call\"",
        ),
    ];
    for (format, expected) in formats {
        let serialized = serde_json::to_string(&format).unwrap();
        assert_eq!(serialized, expected);
        let deserialized: ResponseFormat = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, format);
    }
}

#[test]
fn test_response_format_default() {
    assert_eq!(ResponseFormat::default(), ResponseFormat::Passthrough);
}

#[test]
fn test_to_builtin_tool_type_round_trip() {
    let kinds = [
        BuiltinToolType::WebSearchPreview,
        BuiltinToolType::CodeInterpreter,
        BuiltinToolType::FileSearch,
        BuiltinToolType::ImageGeneration,
    ];
    for kind in kinds {
        let fmt: ResponseFormat = kind.response_format().into();
        assert_eq!(fmt.to_builtin_tool_type(), Some(kind));
    }
    assert_eq!(ResponseFormat::Passthrough.to_builtin_tool_type(), None);
}

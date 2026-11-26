use openai_protocol::responses::{
    CodeInterpreterTool, FileSearchTool, ImageGenerationTool, ResponseTool, WebSearchPreviewTool,
};
use serde_json::json;

use super::*;

fn image_gen_tool(size: Option<&str>, quality: Option<&str>, model: Option<&str>) -> ResponseTool {
    ResponseTool::ImageGeneration(ImageGenerationTool {
        action: None,
        background: None,
        input_fidelity: None,
        input_image_mask: None,
        model: model.map(ToString::to_string),
        moderation: None,
        output_compression: None,
        output_format: None,
        partial_images: None,
        quality: quality.map(ToString::to_string),
        size: size.map(ToString::to_string),
    })
}

#[test]
fn extract_image_generation_overrides_drops_nulls() {
    let tools = vec![image_gen_tool(Some("512x512"), None, Some("gpt-image-1.5"))];

    let overrides = extract_hosted_tool_overrides(&tools, BuiltinToolType::ImageGeneration)
        .expect("overrides present when any field is set");

    let obj = overrides.as_object().expect("object overrides");
    assert_eq!(obj.get("size"), Some(&json!("512x512")));
    assert_eq!(obj.get("model"), Some(&json!("gpt-image-1.5")));
    assert!(
        !obj.contains_key("quality"),
        "null fields must not be forwarded as overrides",
    );
    assert!(
        !obj.contains_key("type"),
        "the type discriminator is redundant at dispatch time",
    );
}

#[test]
fn extract_image_generation_overrides_empty_returns_none() {
    let tools = vec![image_gen_tool(None, None, None)];
    assert!(extract_hosted_tool_overrides(&tools, BuiltinToolType::ImageGeneration).is_none());
}

#[test]
fn extract_image_generation_overrides_missing_declaration_returns_none() {
    let tools: Vec<ResponseTool> = vec![];
    assert!(extract_hosted_tool_overrides(&tools, BuiltinToolType::ImageGeneration).is_none());
}

#[test]
fn extract_image_generation_ignores_non_matching_declarations_first() {
    // A non-matching declaration precedes the image_generation one; matching logic
    // should skip past it rather than short-circuiting on the first tool.
    let tools = vec![
        ResponseTool::Computer,
        image_gen_tool(Some("1024x1024"), None, None),
    ];

    let overrides = extract_hosted_tool_overrides(&tools, BuiltinToolType::ImageGeneration)
        .expect("matches the image_generation declaration, not the computer tool");
    assert_eq!(
        overrides.as_object().and_then(|o| o.get("size")),
        Some(&json!("1024x1024")),
    );
}

#[test]
fn extract_other_hosted_kinds_are_empty_by_default_today() {
    // These structs currently have no fields that are caller-configurable-at-dispatch
    // (all are Option<None> when default-constructed). The arms exist for consistency
    // and will yield real overrides automatically once the tool structs grow fields.
    let web_tools = vec![ResponseTool::WebSearchPreview(
        WebSearchPreviewTool::default(),
    )];
    assert!(extract_hosted_tool_overrides(&web_tools, BuiltinToolType::WebSearchPreview).is_none(),);

    let ci_tools = vec![ResponseTool::CodeInterpreter(CodeInterpreterTool::default())];
    assert!(extract_hosted_tool_overrides(&ci_tools, BuiltinToolType::CodeInterpreter).is_none(),);

    let fs_tools = vec![ResponseTool::FileSearch(FileSearchTool {
        vector_store_ids: vec![],
        filters: None,
        max_num_results: None,
        ranking_options: None,
    })];
    // file_search has a required `vector_store_ids` field which serializes even when empty.
    let overrides = extract_hosted_tool_overrides(&fs_tools, BuiltinToolType::FileSearch);
    if let Some(value) = overrides {
        assert!(value
            .as_object()
            .is_some_and(|o| o.contains_key("vector_store_ids")));
    }
}

#[test]
fn apply_overrides_caller_wins_over_model() {
    let mut args = json!({"prompt": "cat", "size": "auto"});
    let overrides = json!({"size": "512x512"});

    apply_hosted_tool_overrides(&mut args, &overrides);

    assert_eq!(args.get("size"), Some(&json!("512x512")));
    assert_eq!(args.get("prompt"), Some(&json!("cat")));
}

#[test]
fn apply_overrides_extends_args_with_new_keys() {
    let mut args = json!({"prompt": "cat"});
    let overrides = json!({"size": "512x512"});

    apply_hosted_tool_overrides(&mut args, &overrides);

    assert_eq!(args.get("prompt"), Some(&json!("cat")));
    assert_eq!(args.get("size"), Some(&json!("512x512")));
}

#[test]
fn apply_overrides_non_object_noop() {
    let mut args = Value::String("bad".to_string());
    let overrides = json!({"size": "512x512"});

    apply_hosted_tool_overrides(&mut args, &overrides);

    assert_eq!(args, Value::String("bad".to_string()));
}

#[test]
fn apply_overrides_empty_overrides_noop() {
    let mut args = json!({"prompt": "cat"});
    let overrides = json!({});

    apply_hosted_tool_overrides(&mut args, &overrides);

    assert_eq!(args, json!({"prompt": "cat"}));
}

#[test]
fn apply_overrides_non_object_overrides_noop() {
    let mut args = json!({"prompt": "cat"});
    let overrides = Value::String("not-an-object".to_string());

    apply_hosted_tool_overrides(&mut args, &overrides);

    assert_eq!(args, json!({"prompt": "cat"}));
}

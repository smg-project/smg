use llm_tokenizer::MockTokenizer;
use serde_json::json;

use super::{fixtures::*, *};

#[tokio::test]
async fn a_model_without_a_tokenizer_or_a_template_has_no_table() {
    let none = Templates::new(Arc::new(TokenizerRegistry::new()));
    assert_eq!(
        none.format("m").expect_err("no tokenizer"),
        "no tokenizer is registered for model 'm'"
    );
    let plain = templates_with("m", Arc::new(MockTokenizer::new())).await;
    assert_eq!(
        plain.format("m").expect_err("no template"),
        "the tokenizer config has no response_template"
    );
    let float = json!({"defaults": {"temperature": 0.5}, "fields": {"content": {}}});
    let float = templates_with(
        "m",
        Arc::new(MockTokenizer::new().with_response_template(float)),
    )
    .await;
    assert!(float.format("m").expect_err("a float").contains("float"));
    let unread = json!({"fields": {"content": {"open": "(?P<x>a"}}});
    let unread = templates_with(
        "m",
        Arc::new(MockTokenizer::new().with_response_template(unread)),
    )
    .await;
    assert!(unread.format("m").is_err());
}

#[tokio::test]
async fn the_template_compiles_into_a_table_with_the_layout_its_chat_template_gives() {
    let templates = templates_with("m", qwen3_5_tokenizer()).await;
    let format = templates.format("m").expect("a table");
    assert_eq!(format.name(), "response_template");
    assert!(format.layout().is_some(), "the renderings give a layout");
    let unrendered = templates_with("m", unrendered_tokenizer()).await;
    let format = unrendered.format("m").expect("a table");
    assert!(format.layout().is_none(), "no rendering, no layout");
}

#[tokio::test]
async fn a_table_is_compiled_once_per_tokenizer() {
    let registry = Arc::new(TokenizerRegistry::new());
    register(&registry, "m", qwen3_5_tokenizer()).await;
    let templates = Templates::new(registry.clone());
    assert!(templates.format("m").is_ok());
    assert!(templates.format("m").is_ok());
    // Another tokenizer under the model compiles again: the cache follows the tokenizer.
    registry.remove("m");
    register(&registry, "m", Arc::new(MockTokenizer::new())).await;
    assert_eq!(
        templates.format("m").expect_err("the new tokenizer"),
        "the tokenizer config has no response_template"
    );
}

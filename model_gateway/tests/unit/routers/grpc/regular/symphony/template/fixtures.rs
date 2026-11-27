use std::sync::Arc;

use llm_tokenizer::{registry::TokenizerRegistry, traits::Tokenizer, MockTokenizer};
use serde_json::Value;

use super::Templates;

const COMPILED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../crates/response_template/tests/fixtures/compiled/templates.jsonl"
));
const CHAT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../crates/response_template/tests/fixtures/chat_templates/templates.jsonl"
));

fn row(rows: &str, id: &str) -> Value {
    rows.lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a fixture row"))
        .find(|row| row["id"] == id)
        .expect("the fixture row")
}

/// transformers' `serve` response template for Qwen 3.5 checkpoints.
pub(crate) fn qwen3_5_response_template() -> Value {
    row(COMPILED, "serve/qwen3_5")["template"].clone()
}

/// The Qwen3.5-9B chat template.
pub(crate) fn qwen3_5_chat_template() -> String {
    row(CHAT, "Qwen/Qwen3.5-9B")["template"]
        .as_str()
        .expect("a Jinja template")
        .to_string()
}

/// A tokenizer whose config ships the Qwen 3.5 response template and whose chat template
/// renders the probe: its table has a layout.
pub(crate) fn qwen3_5_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(
        MockTokenizer::new()
            .with_eos_token("<|im_end|>")
            .with_response_template(qwen3_5_response_template())
            .with_chat_template(&qwen3_5_chat_template())
            .expect("a chat template"),
    )
}

/// A tokenizer whose config ships the template but whose chat template is the mock's own,
/// which renders no probe: its table has no layout.
pub(crate) fn unrendered_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(
        MockTokenizer::new()
            .with_eos_token("<|im_end|>")
            .with_response_template(qwen3_5_response_template()),
    )
}

/// Registers `tokenizer` under `model`.
pub(crate) async fn register(
    registry: &TokenizerRegistry,
    model: &str,
    tokenizer: Arc<dyn Tokenizer>,
) {
    registry
        .load(model, model, "test", || async move { Ok(tokenizer) })
        .await
        .expect("registered");
}

/// Templates over a registry holding `tokenizer` under `model`.
pub(crate) async fn templates_with(model: &str, tokenizer: Arc<dyn Tokenizer>) -> Templates {
    let registry = Arc::new(TokenizerRegistry::new());
    register(&registry, model, tokenizer).await;
    Templates::new(registry)
}

//! The table a model's response template compiles into, which the parser name `hf` selects
//! ([`super::TEMPLATE`]).
//!
//! A tokenizer's `tokenizer_config.json` may carry a `response_template`, transformers' description
//! of how the model's output splits into a thought, content and tool calls. `smg-response-template`
//! compiles one into a Symphony table ([`to_symphony::compile`]); the stop decoder hides the
//! model's end-of-sequence tokens, so their texts are the compile's stops. The table's structural
//! tags need a layout, how the model's chat template writes a turn: it is learned by rendering the
//! compiler's probe conversation through the tokenizer's chat template ([`layout::learn`]); a
//! template whose renderings give no layout still parses, and a forced call or a JSON answer then
//! has no tag ([`super::constraint`]). The vocabulary's added tokens, as the tokenizer lists them,
//! are the markers the tags write as tokens ([`Format::marker_tokens`]).
//!
//! A table is compiled once per model and tokenizer: a tokenizer registered anew under the model
//! compiles again. What stops a compile is kept with the model and logged once, and is the reason
//! a request naming `hf` for the model is refused.

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use chrono::{DateTime, Utc};
use llm_tokenizer::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    registry::TokenizerRegistry,
    traits::Tokenizer,
};
use parking_lot::RwLock;
use serde_json::{json, Value};
use smg_response_template::{
    load_response_template,
    to_symphony::{
        self,
        layout::{self, Case},
    },
};
use symphony::Format;
use tracing::info;

/// The tables the registered tokenizers' response templates compile into, by model.
pub(crate) struct Templates {
    tokenizers: Arc<TokenizerRegistry>,
    compiled: RwLock<HashMap<String, Entry>>,
}

/// What a model's tokenizer compiled into, bound to that tokenizer.
struct Entry {
    tokenizer: Weak<dyn Tokenizer>,
    table: Result<Format, String>,
}

impl Templates {
    pub(crate) fn new(tokenizers: Arc<TokenizerRegistry>) -> Self {
        Self {
            tokenizers,
            compiled: RwLock::default(),
        }
    }

    /// The table `model`'s response template compiles into, or why it has none: no tokenizer, no
    /// template, a template the crate does not read, or one the compile refuses.
    pub(crate) fn format(&self, model: &str) -> Result<Format, String> {
        let Some(tokenizer) = self.tokenizers.get(model) else {
            return Err(format!("no tokenizer is registered for model '{model}'"));
        };
        let current = Arc::downgrade(&tokenizer);
        if let Some(entry) = self.compiled.read().get(model) {
            if entry.tokenizer.ptr_eq(&current) {
                return entry.table.clone();
            }
        }
        let table = compile(model, tokenizer.as_ref());
        match &table {
            Ok(format) => info!(
                model,
                table = format.name(),
                layout = format.layout().is_some(),
                "the response_template compiles into a table"
            ),
            Err(why) => info!(model, why, "the response_template compiles into no table"),
        }
        self.compiled.write().insert(
            model.to_string(),
            Entry {
                tokenizer: current,
                table: table.clone(),
            },
        );
        table
    }
}

/// The table `tokenizer`'s response template compiles into, with the layout its chat template
/// gives and the vocabulary's marker tokens.
fn compile(model: &str, tokenizer: &dyn Tokenizer) -> Result<Format, String> {
    let raw = tokenizer
        .response_template()
        .ok_or_else(|| "the tokenizer config has no response_template".to_string())?;
    if has_float(raw) {
        return Err(
            "the response_template holds a float, which serde_json can read differently from \
             Python"
                .to_string(),
        );
    }
    let template = load_response_template(raw).map_err(|e| e.to_string())?;
    let stops = stop_texts(tokenizer);
    let stops: Vec<&str> = stops.iter().map(String::as_str).collect();
    let compiled = to_symphony::compile(&template, &stops).map_err(|e| e.to_string())?;
    let format = match layout::learn(&compiled.format, &stops, |case| render(tokenizer, case)) {
        Ok(learned) => learned.apply(compiled.format),
        Err(why) => {
            info!(
                model,
                %why,
                "the chat template gives the table no layout: a forced call and a JSON answer \
                 have no structural tag"
            );
            compiled.format
        }
    };
    Ok(format.marker_tokens(token_texts(tokenizer)))
}

/// The probe's conversation `case` as the tokenizer's chat template renders it, with the probe's
/// tools; `None` where the template raises. The messages' text takes the content format the
/// template reads, as a request's does; the clock is pinned, so a template that writes the date
/// writes the same one in every rendering.
fn render(tokenizer: &dyn Tokenizer, case: Case) -> Option<String> {
    let mut messages = case.messages();
    if tokenizer.chat_template_content_format() == ChatTemplateContentFormat::OpenAI {
        for message in messages.as_array_mut()? {
            if let Some(text) = message.get("content").and_then(Value::as_str) {
                let part = json!([{"type": "text", "text": text}]);
                message["content"] = part;
            }
        }
    }
    let tools = layout::tools();
    let params = ChatTemplateParams {
        add_generation_prompt: case.generation_prompt(),
        tools: tools.as_array().map(Vec::as_slice),
        special_tokens: Some(tokenizer.get_special_tokens()),
        now: Some(DateTime::<Utc>::UNIX_EPOCH.fixed_offset()),
        ..ChatTemplateParams::default()
    };
    tokenizer
        .apply_chat_template(messages.as_array()?, params)
        .ok()
}

/// Whether a JSON value holds a float anywhere: serde_json and Python can read one differently,
/// and a template's defaults are compared as values.
fn has_float(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.is_f64(),
        Value::Array(items) => items.iter().any(has_float),
        Value::Object(members) => members.values().any(has_float),
        _ => false,
    }
}

/// The texts of the tokens the model stops on: its end-of-sequence ids, and the named one.
fn stop_texts(tokenizer: &dyn Tokenizer) -> Vec<String> {
    let mut texts: Vec<String> = Vec::new();
    let ids = tokenizer.eos_token_ids().iter();
    let named = tokenizer.get_special_tokens().eos_token.clone();
    for text in ids.filter_map(|&id| tokenizer.id_to_token(id)).chain(named) {
        if !texts.contains(&text) {
            texts.push(text);
        }
    }
    texts
}

/// The texts of the vocabulary's added tokens: the ones the tokenizer lists for its prefix
/// cache, taken on the raw text and after normalization, and the special tokens the config
/// names. A template's markers are often added tokens that are not special.
fn token_texts(tokenizer: &dyn Tokenizer) -> Vec<String> {
    let special = tokenizer.get_special_tokens();
    let named = [
        &special.bos_token,
        &special.eos_token,
        &special.unk_token,
        &special.sep_token,
        &special.pad_token,
        &special.cls_token,
        &special.mask_token,
    ];
    let mut texts: Vec<String> = Vec::new();
    for text in tokenizer
        .added_token_strings()
        .into_iter()
        .chain(tokenizer.added_tokens_after_normalization())
        .chain(named.into_iter().flatten().cloned())
        .chain(special.additional_special_tokens.iter().cloned())
    {
        if !texts.contains(&text) {
            texts.push(text);
        }
    }
    texts
}

/// The recorded templates the compiler's own tests pair: transformers' `serve` response
/// template for Qwen 3.5 checkpoints, and the Qwen3.5-9B chat template, whose renderings give
/// the table a layout.
#[cfg(test)]
pub(crate) mod fixtures {
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
}

#[cfg(test)]
mod tests {
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
}

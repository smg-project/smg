//! The table a model's response template compiles into, which the parser name `hf` selects
//! ([`TEMPLATE_PARSER`](crate::config::TEMPLATE_PARSER)).
//!
//! A tokenizer's `tokenizer_config.json` may carry a `response_template`, transformers' description
//! of how the model's output splits into a thought, content and tool calls. `smg-response-template`
//! compiles one into a Symphony table ([`to_symphony::compile`]); the stop decoder hides the
//! model's end-of-sequence tokens, so their texts are the compile's stops. The table's structural
//! tags need a layout, how the model's chat template writes a turn: it is learned by rendering the
//! compiler's probe conversation through the tokenizer's chat template ([`layout::learn`]); a
//! template whose renderings give no layout still parses, and a forced call or a JSON answer then
//! has no tag ([`constraint_for`](super::constraint::constraint_for)). The vocabulary's added
//! tokens, as the tokenizer lists them, are the markers the tags write as tokens
//! ([`Format::marker_tokens`]).
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

/// The tables the registered tokenizers' response templates compile into, by model; and, by
/// model, whether a table's markers are hidden by the model's decoder, decided once.
pub(crate) struct Templates {
    tokenizers: Arc<TokenizerRegistry>,
    compiled: RwLock<HashMap<String, Entry>>,
    hidden: RwLock<HashMap<String, Hidden>>,
}

/// What a model's tokenizer compiled into, bound to that tokenizer: the table, shared with every
/// request and engine that reads with it.
struct Entry {
    tokenizer: Weak<dyn Tokenizer>,
    table: Result<Arc<Format>, String>,
}

/// Whether a table's markers are hidden by a model's decoder, bound to the tokenizer that decided
/// it and to the markers it decided for.
struct Hidden {
    tokenizer: Weak<dyn Tokenizer>,
    markers: Vec<String>,
    hidden: bool,
}

impl Templates {
    pub(crate) fn new(tokenizers: Arc<TokenizerRegistry>) -> Self {
        Self {
            tokenizers,
            compiled: RwLock::default(),
            hidden: RwLock::default(),
        }
    }

    /// Whether `format`'s markers lose text when `tokenizer`, `model`'s, decodes them with the
    /// special tokens dropped, as `decide` works it out: decided once per model, for the
    /// tokenizer and the markers it was decided for, since deciding encodes and decodes every
    /// marker and the answer depends on nothing else. A model whose tokenizer or whose table's
    /// markers changed is decided again.
    pub(crate) fn markers_hidden(
        &self,
        model: &str,
        format: &Format,
        tokenizer: &Arc<dyn Tokenizer>,
        decide: impl FnOnce() -> bool,
    ) -> bool {
        let current = Arc::downgrade(tokenizer);
        if let Some(entry) = self.hidden.read().get(model) {
            if entry.tokenizer.ptr_eq(&current)
                && entry
                    .markers
                    .iter()
                    .map(String::as_str)
                    .eq(format.terminal_texts())
            {
                return entry.hidden;
            }
        }
        let hidden = decide();
        self.hidden.write().insert(
            model.to_string(),
            Hidden {
                tokenizer: current,
                markers: format.terminal_texts().map(str::to_string).collect(),
                hidden,
            },
        );
        hidden
    }

    /// The table `model`'s response template compiles into, or why it has none: no tokenizer, no
    /// template, a template the crate does not read, or one the compile refuses.
    pub(crate) fn format(&self, model: &str) -> Result<Arc<Format>, String> {
        let Some(tokenizer) = self.tokenizers.get(model) else {
            return Err(format!("no tokenizer is registered for model '{model}'"));
        };
        let current = Arc::downgrade(&tokenizer);
        if let Some(entry) = self.compiled.read().get(model) {
            if entry.tokenizer.ptr_eq(&current) {
                return entry.table.clone();
            }
        }
        let table = compile(model, tokenizer.as_ref()).map(Arc::new);
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
    if tokenizer.chat_template_content_format_for_request(true) == ChatTemplateContentFormat::OpenAI
    {
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
#[path = "../../../../../tests/unit/routers/grpc/regular/symphony/template/fixtures.rs"]
pub(crate) mod fixtures;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/symphony/template/tests.rs"]
mod tests;

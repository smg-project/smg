//! The template value of a request's tools, prepared once per distinct tool
//! set and shared by every request that carries the same tools.
//!
//! A request's typed tools reached the template as JSON values converted for
//! every request, and the render converted those into its own values again:
//! with four schemas, about two fifths of the allocation calls of a
//! tool-calling request, spent on trees that are the same from one request of
//! an application to the next (the tool set is the application's, the
//! messages are the turn's). The table here keeps the prepared value by the
//! tools' serialized bytes; a hit compares the bytes, so a request reads with
//! a value made of tools equal to its own.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use llm_tokenizer::chat_template::PreparedTools;
use openai_protocol::common::Tool;
use parking_lot::RwLock;
use serde_json::Value;

/// How many distinct tool sets are kept before the table starts over: an
/// application's tool sets come to a handful, a gateway serving many
/// applications at once keeps their last distinct sets up to this, and a
/// stream of requests with ever-different tools cannot grow the table without
/// bound.
const PREPARED_TOOLS_KEPT: usize = 64;

/// The largest tool set looked up, by its serialized bytes: a set above it is
/// prepared for its request alone, before any key is hashed or kept. What an
/// entry keeps is larger than its key (the JSON values, the render's values
/// of the same content and the `tojson` texts the tools come to keep), so the
/// kept side is bounded by weight below, not by this.
const PREPARED_TOOLS_MAX_KEY_BYTES: usize = 64 * 1024;

/// The most a kept set may weigh ([`set_weight`]): a heavier one is prepared
/// for its request alone. The weight charges the plain form; a checkpoint
/// whose renderer reads the tools key-sorted keeps the sorted form beside it
/// from its first render (`PreparedTools::key_sorted`), uncharged, so such a
/// set weighs up to about twice this.
const PREPARED_TOOLS_MAX_SET_WEIGHT: usize = 1024 * 1024;

/// The weight the kept sets may sum to before the table starts over, so a
/// run of distinct sets cannot keep more than about this alive after their
/// requests have finished, whatever their shape (about twice this where the
/// sets are of checkpoints that read their tools key-sorted, which keep the
/// sorted form beside the charged one).
const PREPARED_TOOLS_KEPT_WEIGHT: usize = 16 * 1024 * 1024;

/// The kept values by the tools' serialized bytes, and the weight they sum to.
static PREPARED_TOOLS: LazyLock<RwLock<Kept>> = LazyLock::new(|| RwLock::new(Kept::default()));

#[derive(Default)]
struct Kept {
    sets: HashMap<Vec<u8>, Arc<PreparedTools>>,
    weight: usize,
}

/// What an entry costs kept: its key, the values ([`kept_weight`]) and the
/// `tojson` texts its tools keep as the renders go, up to
/// [`PreparedTools::KEPT_TOJSON_TEXTS_PER_TOOL`] per tool, charged at that cap
/// as the tool's share of the key bytes each (a compact form's size; an
/// `indent` form runs about twice that, and the values' term dominates, so
/// the table's bound stays close whatever the template asks of the filter).
fn set_weight(key_len: usize, values: &[Value]) -> usize {
    key_len * (1 + PreparedTools::KEPT_TOJSON_TEXTS_PER_TOOL) + kept_weight(values)
}

/// What a set's values cost kept, estimated from its JSON values: the bytes
/// of its strings and keys plus a fixed cost per value and per map entry,
/// counted twice for the render's values made of the same content; the
/// entry's key bytes are added by the caller. The estimate
/// follows the content's shape (a long enum of small numbers weighs its
/// elements, not its bytes), which the serialized length does not.
fn kept_weight(tools: &[Value]) -> usize {
    const VALUE: usize = 96;
    const ENTRY: usize = 48;
    fn walk(value: &Value) -> usize {
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => VALUE,
            Value::String(text) => VALUE + text.len(),
            Value::Array(items) => VALUE + items.iter().map(walk).sum::<usize>(),
            Value::Object(map) => {
                VALUE
                    + map
                        .iter()
                        .map(|(key, item)| ENTRY + key.len() + walk(item))
                        .sum::<usize>()
            }
        }
    }
    2 * tools.iter().map(walk).sum::<usize>()
}

thread_local! {
    /// The serialization of a request's tools, the table's key: one buffer
    /// per thread, reused, so a hit allocates nothing.
    static KEY: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The tools of a request prepared for the template: the value kept when a
/// request with the same tools came before, else made now and kept.
pub(crate) fn prepare_tools(tools: &[Tool]) -> Result<Arc<PreparedTools>, String> {
    KEY.with(|key| {
        let mut key = key.borrow_mut();
        key.clear();
        serde_json::to_writer(&mut *key, tools)
            .map_err(|e| format!("Failed to serialize tools: {e}"))?;
        if key.len() > PREPARED_TOOLS_MAX_KEY_BYTES {
            // Not kept; the buffer goes back to empty so a worker thread
            // holds at most the cap of it.
            *key = Vec::new();
            return Ok(Arc::new(PreparedTools::new(tool_values(tools)?)));
        }
        if let Some(kept) = PREPARED_TOOLS.read().sets.get(key.as_slice()) {
            return Ok(Arc::clone(kept));
        }
        let values = tool_values(tools)?;
        let weight = set_weight(key.len(), &values);
        let prepared = Arc::new(PreparedTools::new(values));
        if weight > PREPARED_TOOLS_MAX_SET_WEIGHT {
            return Ok(prepared);
        }
        let owned_key = key.clone();
        let dropped = {
            let mut kept = PREPARED_TOOLS.write();
            // Another request may have kept the same set between the read
            // and the write: theirs is the one every request shares from
            // here on.
            if let Some(kept) = kept.sets.get(key.as_slice()) {
                return Ok(Arc::clone(kept));
            }
            let full = kept.sets.len() >= PREPARED_TOOLS_KEPT
                || kept.weight + weight > PREPARED_TOOLS_KEPT_WEIGHT;
            let dropped = full.then(|| std::mem::take(&mut *kept));
            kept.weight += weight;
            kept.sets.insert(owned_key, Arc::clone(&prepared));
            dropped
        };
        // A restart's trees are freed here, after the lock, not under it.
        drop(dropped);
        Ok(prepared)
    })
}

/// The tools as the JSON values the template reads, one per tool.
fn tool_values(tools: &[Tool]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(tool_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to serialize tools: {e}"))
}

/// One tool as the template reads it: its serialized form, with the schema
/// the client did not give handed on as `null`. A serialized function leaves
/// an absent schema out, as any optional field the client left out; the
/// engine's own template reads the declaration with `"parameters": null`, in
/// the place of the schema (after the description, before `strict` and any
/// further field), and renders the tool so.
fn tool_value(tool: &Tool) -> Result<Value, serde_json::Error> {
    let mut value = serde_json::to_value(tool)?;
    if tool.function.parameters.is_none() {
        if let Some(function) = value.get_mut("function").and_then(Value::as_object_mut) {
            let fields = std::mem::take(function);
            let has_description = fields.contains_key("description");
            for (key, field) in fields {
                let schema_follows = key == "description" || (key == "name" && !has_description);
                function.insert(key, field);
                if schema_follows {
                    function.insert("parameters".to_string(), Value::Null);
                }
            }
        }
    }
    Ok(value)
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/prepared_tools/tests.rs"]
mod tests;

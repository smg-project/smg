//! The template value of a request's tools, prepared once per distinct tool
//! set and shared by every request that carries the same tools.
//!
//! A request's typed tools reached the template as JSON values converted for
//! every request, and the render converted those into its own values again:
//! with four schemas, about two fifths of the allocation calls of a
//! tool-calling request, spent on trees that are the same from one request of
//! an application to the next (the tool set is the application's, the
//! messages are the turn's). The table here keeps the prepared value by a key
//! walked from the typed tools ([`write_key`]); a hit compares the key's
//! bytes, so a request reads with a value made of tools equal to its own.

use std::{
    cell::RefCell,
    collections::HashMap,
    io,
    sync::{Arc, LazyLock},
};

use llm_tokenizer::chat_template::PreparedTools;
use openai_protocol::common::{Function, Tool};
use parking_lot::RwLock;
use serde_json::{Map, Value};

/// How many distinct tool sets are kept before the table starts over: an
/// application's tool sets come to a handful, a gateway serving many
/// applications at once keeps their last distinct sets up to this, and a
/// stream of requests with ever-different tools cannot grow the table without
/// bound.
const PREPARED_TOOLS_KEPT: usize = 64;

/// The largest tool set looked up, by its key's bytes (about its serialized
/// size, [`write_key`]): a set above it is prepared for its request alone,
/// before any key is hashed or kept. What an entry keeps is larger than its
/// key (the JSON values, the render's values of the same content and the
/// `tojson` texts the tools come to keep), so the kept side is bounded by
/// weight below, not by this.
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

/// The kept values by the tools' key bytes, and the weight they sum to.
static PREPARED_TOOLS: LazyLock<RwLock<Kept>> = LazyLock::new(|| RwLock::new(Kept::default()));

#[derive(Default)]
struct Kept {
    sets: HashMap<Vec<u8>, Arc<PreparedTools>>,
    weight: usize,
}

/// What an entry costs kept: its key, the values ([`kept_weight`]) and the
/// `tojson` texts its tools keep as the renders go, up to
/// [`PreparedTools::KEPT_TOJSON_TEXTS_PER_TOOL`] per tool, charged at that cap
/// as the tool's share of the set's compact text each ([`serialized_len`]:
/// the key is shorter than that text; an `indent` form runs about twice it,
/// and the values' term dominates, so the table's bound stays close whatever
/// the template asks of the filter).
fn set_weight(key_len: usize, values: &[Value]) -> usize {
    let texts = serialized_len(values) * PreparedTools::KEPT_TOJSON_TEXTS_PER_TOOL;
    key_len + texts + kept_weight(values)
}

/// The length of the values' compact JSON text, counted and not written. It
/// runs for a set that was not kept yet, never for one that is found.
fn serialized_len(values: &[Value]) -> usize {
    struct Count(usize);
    impl io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    // JSON values always serialize and the counter takes every write, so
    // the count is whole either way.
    let _ = serde_json::to_writer(&mut count, values);
    count.0
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
    /// The key of a request's tools ([`write_key`]): one buffer per thread,
    /// reused, so a hit allocates nothing.
    static KEY: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The bytes that say what comes next in a key. A number stands in a key as
/// its JSON text, which begins with a digit or a minus sign, as none of these
/// does, and holds no `END`.
const ABSENT: u8 = 0;
const END: u8 = 0;
const PRESENT: u8 = 1;
const NULL: u8 = 2;
const FALSE: u8 = 3;
const TRUE: u8 = 4;
const STRING: u8 = 5;
const ARRAY: u8 = 6;
const OBJECT: u8 = 7;

/// The table's key of a request's tools: the typed tools walked into bytes,
/// in the request's order, a text behind its length, a value behind its tag,
/// an array up to its end mark. The bytes read back one way only, so two tool
/// sets share a key exactly when they are equal in every field and in order:
/// when they serialize to the same bytes and prepare to the same value. A
/// request with tools pays the walk on every lookup, at a fraction of the
/// serialization it stands for (a string is copied whole, where a JSON writer
/// reads its every byte for an escape). The key of a set within the size
/// bound is no longer than the set serialized, so what the key bound admitted
/// by the serialized bytes it still admits. The weight bounds charge a tool
/// with no schema a little more than the serialized bytes did: its kept
/// texts hold the `null` schema ([`tool_value`]), 18 bytes a text that the
/// tool serialized does not, and [`set_weight`] counts them.
fn write_key(key: &mut Vec<u8>, tools: &[Tool]) -> Result<(), serde_json::Error> {
    for tool in tools {
        // Taken apart whole: a field added to either record does not build
        // here until the key reads it.
        let Tool {
            tool_type,
            function:
                Function {
                    name,
                    description,
                    parameters,
                    strict,
                    extra,
                },
        } = tool;
        key_text(key, tool_type);
        key_text(key, name);
        match description {
            Some(description) => {
                key.push(PRESENT);
                key_text(key, description);
            }
            None => key.push(ABSENT),
        }
        match parameters {
            Some(schema) => key_value(key, schema)?,
            None => key.push(ABSENT),
        }
        key.push(match strict {
            Some(true) => TRUE,
            Some(false) => FALSE,
            None => ABSENT,
        });
        key_map(key, extra)?;
    }
    Ok(())
}

fn key_value(key: &mut Vec<u8>, value: &Value) -> Result<(), serde_json::Error> {
    match value {
        Value::Null => key.push(NULL),
        Value::Bool(flag) => key.push(if *flag { TRUE } else { FALSE }),
        Value::Number(number) => {
            // As it serializes, so two numbers share a key exactly when
            // they share a text.
            serde_json::to_writer(&mut *key, number)?;
            key.push(END);
        }
        Value::String(text) => {
            key.push(STRING);
            key_text(key, text);
        }
        Value::Array(items) => {
            key.push(ARRAY);
            for item in items {
                key_value(key, item)?;
            }
            key.push(END);
        }
        Value::Object(fields) => {
            key.push(OBJECT);
            key_map(key, fields)?;
        }
    }
    Ok(())
}

fn key_map(key: &mut Vec<u8>, fields: &Map<String, Value>) -> Result<(), serde_json::Error> {
    key_len(key, fields.len());
    for (name, value) in fields {
        key_text(key, name);
        key_value(key, value)?;
    }
    Ok(())
}

fn key_text(key: &mut Vec<u8>, text: &str) {
    key_len(key, text.len());
    key.extend_from_slice(text.as_bytes());
}

/// A length seven bits to a byte, the low ones first, the last byte alone
/// under 128: one byte for most names and two for most descriptions, where
/// the serialized form spends two quotes and a separator.
fn key_len(key: &mut Vec<u8>, len: usize) {
    let mut rest = len;
    loop {
        let [low, ..] = rest.to_le_bytes();
        if rest < 0x80 {
            key.push(low);
            return;
        }
        key.push(low | 0x80);
        rest >>= 7;
    }
}

/// The tools of a request prepared for the template: the value kept when a
/// request with the same tools came before, else made now and kept.
pub(crate) fn prepare_tools(tools: &[Tool]) -> Result<Arc<PreparedTools>, String> {
    KEY.with(|key| {
        let mut key = key.borrow_mut();
        key.clear();
        write_key(&mut key, tools).map_err(|e| format!("Failed to serialize tools: {e}"))?;
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

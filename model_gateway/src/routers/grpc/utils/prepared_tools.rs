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
/// for its request alone.
const PREPARED_TOOLS_MAX_SET_WEIGHT: usize = 1024 * 1024;

/// The weight the kept sets may sum to before the table starts over, so a
/// run of distinct sets cannot keep more than about this alive after their
/// requests have finished, whatever their shape.
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
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to serialize tools: {e}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    /// The table is one per process: the tests that read its state run one
    /// at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn tool(name: &str, description: &str) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }
        }))
        .unwrap()
    }

    #[test]
    fn a_tool_set_is_prepared_once_and_shared_by_the_next_request_with_the_same_tools() {
        let _serial = SERIAL.lock().unwrap();
        let tools = [
            tool("prepared_once_a", "the first"),
            tool("prepared_once_b", "the second"),
        ];
        let first = prepare_tools(&tools).unwrap();
        assert_eq!(
            first.tools(),
            tools
                .iter()
                .map(|tool| serde_json::to_value(tool).unwrap())
                .collect::<Vec<_>>(),
            "the prepared values are the request's tools as JSON values"
        );
        let equal = [tools[0].clone(), tools[1].clone()];
        assert!(
            Arc::ptr_eq(&first, &prepare_tools(&equal).unwrap()),
            "equal tools share the one kept value"
        );
        let other = [tool("prepared_once_a", "another description")];
        let third = prepare_tools(&other).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &third),
            "other tools are another value"
        );
        assert_eq!(third.tools(), [serde_json::to_value(&other[0]).unwrap()]);
        let reordered = [tools[1].clone(), tools[0].clone()];
        assert!(
            !Arc::ptr_eq(&first, &prepare_tools(&reordered).unwrap()),
            "the same tools in another order render differently and are another value"
        );
    }

    #[test]
    fn the_table_starts_over_at_its_bound_and_keeps_no_oversize_set() {
        let _serial = SERIAL.lock().unwrap();
        let first_tools = [tool("bound_first", "kept until the bound")];
        let first = prepare_tools(&first_tools).unwrap();
        for i in 0..PREPARED_TOOLS_KEPT {
            prepare_tools(&[tool(&format!("bound_{i}"), "one of many")]).unwrap();
        }
        assert!(
            PREPARED_TOOLS.read().sets.len() <= PREPARED_TOOLS_KEPT,
            "the table never holds more than its bound"
        );
        assert!(
            !Arc::ptr_eq(&first, &prepare_tools(&first_tools).unwrap()),
            "a set the restart dropped is prepared again"
        );
        let oversize = [tool("oversize", &"x".repeat(PREPARED_TOOLS_MAX_KEY_BYTES))];
        let once = prepare_tools(&oversize).unwrap();
        let twice = prepare_tools(&oversize).unwrap();
        assert!(
            !Arc::ptr_eq(&once, &twice),
            "a set above the size bound is prepared for its request alone"
        );
        assert_eq!(once.tools(), twice.tools());
        KEY.with(|key| {
            assert!(
                key.borrow().capacity() <= PREPARED_TOOLS_MAX_KEY_BYTES,
                "the buffer is dropped back after an oversize set"
            );
        });
    }

    #[test]
    fn the_table_starts_over_when_the_kept_sets_would_exceed_their_weight() {
        let _serial = SERIAL.lock().unwrap();
        let first_tools = [tool("weight_first", "kept until the budget")];
        let first = prepare_tools(&first_tools).unwrap();
        // A set of many small values: light on the wire, heavy kept. Under the
        // per-set cap, so a run of them fills the table's weight long before
        // its count.
        let enum_tool = |name: &str| -> Tool {
            serde_json::from_value(json!({"type": "function", "function": {"name": name,
                "parameters": {"type": "object", "properties": {"code": {"type": "integer", "enum": (0..4000).collect::<Vec<_>>()}}}}}))
            .unwrap()
        };
        let one = [enum_tool("weight_probe")];
        let weight = set_weight(
            serde_json::to_vec(&one).unwrap().len(),
            &one.iter()
                .map(|t| serde_json::to_value(t).unwrap())
                .collect::<Vec<_>>(),
        );
        assert!(weight < PREPARED_TOOLS_MAX_SET_WEIGHT, "{weight}");
        let sets = PREPARED_TOOLS_KEPT_WEIGHT / weight + 2;
        assert!(
            sets < PREPARED_TOOLS_KEPT,
            "{sets} sets of {weight} fill the weight before the count"
        );
        for i in 0..sets {
            prepare_tools(&[enum_tool(&format!("weight_{i}"))]).unwrap();
        }
        let kept = PREPARED_TOOLS.read();
        assert!(
            kept.weight <= PREPARED_TOOLS_KEPT_WEIGHT,
            "the kept sets stay within their weight ({})",
            kept.weight
        );
        drop(kept);
        assert!(
            !Arc::ptr_eq(&first, &prepare_tools(&first_tools).unwrap()),
            "a set the weight's restart dropped is prepared again"
        );
        // Heavy kept, light on the wire: 12,000 one-digit numbers serialize to
        // about 24 KiB, under the key cap, and weigh past the per-set cap.
        let heavy = [serde_json::from_value::<Tool>(json!({"type": "function", "function": {"name": "heavy",
            "parameters": {"type": "object", "properties": {"code": {"type": "integer", "enum": (0..12000).map(|i| i % 10).collect::<Vec<_>>()}}}}})).unwrap()];
        let heavy_key = serde_json::to_vec(&heavy).unwrap().len();
        let heavy_weight = set_weight(
            heavy_key,
            &heavy
                .iter()
                .map(|t| serde_json::to_value(t).unwrap())
                .collect::<Vec<_>>(),
        );
        assert!(
            heavy_key <= PREPARED_TOOLS_MAX_KEY_BYTES
                && heavy_weight > PREPARED_TOOLS_MAX_SET_WEIGHT,
            "{heavy_key} bytes, {heavy_weight} weight"
        );
        assert!(
            !Arc::ptr_eq(
                &prepare_tools(&heavy).unwrap(),
                &prepare_tools(&heavy).unwrap()
            ),
            "a set above the weight cap is prepared for its request alone"
        );
    }
}

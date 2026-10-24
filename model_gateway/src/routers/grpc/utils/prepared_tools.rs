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

/// The largest tool set kept, by its serialized bytes: a set above it is
/// prepared for its request alone, which bounds what the table holds to
/// `PREPARED_TOOLS_KEPT` sets of at most this size and their value trees.
const PREPARED_TOOLS_MAX_KEY_BYTES: usize = 256 * 1024;

/// The kept values by the tools' serialized bytes.
static PREPARED_TOOLS: LazyLock<RwLock<HashMap<Vec<u8>, Arc<PreparedTools>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

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
            let prepared = Arc::new(PreparedTools::new(tool_values(tools)?));
            *key = Vec::new();
            return Ok(prepared);
        }
        if let Some(kept) = PREPARED_TOOLS.read().get(key.as_slice()) {
            return Ok(Arc::clone(kept));
        }
        let prepared = Arc::new(PreparedTools::new(tool_values(tools)?));
        let mut kept = PREPARED_TOOLS.write();
        // Another request may have kept the same set between the read and
        // the write: theirs is the one every request shares from here on.
        if let Some(kept) = kept.get(key.as_slice()) {
            return Ok(Arc::clone(kept));
        }
        if kept.len() >= PREPARED_TOOLS_KEPT {
            kept.clear();
        }
        kept.insert(key.clone(), Arc::clone(&prepared));
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
            PREPARED_TOOLS.read().len() <= PREPARED_TOOLS_KEPT,
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
    }
}

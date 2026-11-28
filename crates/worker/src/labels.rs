//! Labels of a worker's spec: the names the worker's own code and its hosts
//! read, and the flat map an engine's answer becomes.

use std::collections::HashMap;

/// Which discovery provider owns the worker: `kubernetes`, `file`, `slurm` or
/// `consul`. A provider's registration carries it, and the health checker
/// reads it to tell a worker that discovery registers again from one that
/// nothing re-adds.
pub const DISCOVERY_PROVIDER_LABEL: &str = "smg.ai/discovery-provider";

/// The label that carries an engine's vision-input capability, as its model
/// info reports it. `"false"` on a multimodal architecture means the engine
/// runs without a vision encoder, so no pixel payload (image, video) may
/// reach it. A multimodal model reports `"true"`; a text-only model also
/// reports `"false"`. The label speaks for the vision encoder only: an engine
/// that still takes audio or pre-computed embeddings reports `"false"` too.
///
/// The label is tri-state. A `false` becomes a label only where the engine's
/// servicer computes the verdict itself; elsewhere a `false` cannot be told
/// from a default and is dropped. A servicer that carries no media to its
/// engine says so in its server info instead, and that deliberate `false` is
/// kept. An absent label reads as unknown, never as a refusal.
pub const SUPPORTS_VISION_LABEL: &str = "supports_vision";

/// Whether text contains a Unicode control character.
pub fn has_control_character(text: &str) -> bool {
    text.chars().any(char::is_control)
}

/// Serialize to flat label map, skipping nulls/zeros/empty.
///
/// Booleans are emitted as `"true"` / `"false"` so downstream consumers
/// (e.g. `is_generation == "false"` for embedding detection) work correctly.
pub fn flat_labels<T: serde::Serialize>(value: &T) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(value) {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) if !s.is_empty() && s != "null" => {
                    labels.insert(key, s);
                }
                serde_json::Value::Number(n) if n.as_f64().is_some_and(|v| v != 0.0) => {
                    // Format integers without decimal point
                    let formatted = n
                        .as_i64()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| n.to_string());
                    labels.insert(key, formatted);
                }
                serde_json::Value::Bool(b) => {
                    labels.insert(key, b.to_string());
                }
                serde_json::Value::Array(arr) if !arr.is_empty() => {
                    if let Ok(s) = serde_json::to_string(&arr) {
                        labels.insert(key, s);
                    }
                }
                _ => {}
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

    use super::flat_labels;

    #[derive(Serialize)]
    struct Nested {
        inner: u32,
    }

    #[derive(Serialize)]
    struct Answer {
        name: &'static str,
        empty: &'static str,
        null_word: &'static str,
        count: u32,
        ratio: f64,
        zero: u32,
        on: bool,
        off: bool,
        list: Vec<&'static str>,
        no_list: Vec<&'static str>,
        absent: Option<u32>,
        nested: Nested,
    }

    /// A string, a whole number, a fraction, either boolean and a list each
    /// become a label. An empty string, the word `null`, a zero, an empty
    /// list, a null and a nested map become none.
    #[test]
    fn a_flat_map_keeps_each_field_that_says_something() {
        let mut labels: Vec<(String, String)> = flat_labels(&Answer {
            name: "model",
            empty: "",
            null_word: "null",
            count: 3,
            ratio: 0.5,
            zero: 0,
            on: true,
            off: false,
            list: vec!["a", "b"],
            no_list: Vec::new(),
            absent: None,
            nested: Nested { inner: 1 },
        })
        .into_iter()
        .collect();
        labels.sort();
        let kept = [
            ("count", "3"),
            ("list", r#"["a","b"]"#),
            ("name", "model"),
            ("off", "false"),
            ("on", "true"),
            ("ratio", "0.5"),
        ];
        assert_eq!(
            labels,
            kept.map(|(key, value)| (key.to_string(), value.to_string()))
        );
    }
}

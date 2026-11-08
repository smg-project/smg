//! What the local workflow steps' tests build a worker's labels from.

use std::collections::HashMap;

/// The label map `pairs` spell, each name and value owned.
pub(crate) fn labels_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

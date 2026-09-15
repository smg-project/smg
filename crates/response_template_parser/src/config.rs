use serde::{Deserialize, Serialize};

/// The established SMG parser bound: four mebibytes.
pub const DEFAULT_BYTE_LIMIT: usize = 4 * 1024 * 1024;

/// Independent UTF-8 byte limits for one parser request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserConfig {
    pub max_pending_bytes: usize,
    pub max_structured_field_bytes: usize,
    pub max_body_bytes: usize,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            max_pending_bytes: DEFAULT_BYTE_LIMIT,
            max_structured_field_bytes: DEFAULT_BYTE_LIMIT,
            max_body_bytes: DEFAULT_BYTE_LIMIT,
        }
    }
}

//! Common test utilities for tool-parser tests
#![allow(
    clippy::allow_attributes,
    reason = "dead_code must use #[allow] not #[expect] in shared test modules because the lint only fires in some compilation units"
)]

use openai_protocol::common::{Function, Tool};
use serde_json::json;

pub mod streaming_helpers {
    //! Streaming Test Helpers
    //!
    //! Utilities for creating realistic streaming chunks that simulate
    //! how LLM tokens actually arrive (1-5 characters at a time).

    /// Split input into realistic char-level chunks (2-3 chars each for determinism)
    #[allow(
        dead_code,
        reason = "shared test utility not used by all test binaries"
    )]
    pub fn create_realistic_chunks(input: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Take 2-3 characters at a time (deterministic for testing)
            let chunk_size = if i + 3 <= chars.len() && chars[i].is_ascii_alphanumeric() {
                3 // Longer chunks for alphanumeric sequences
            } else {
                2 // Shorter chunks for special characters
            };

            let end = (i + chunk_size).min(chars.len());
            let chunk: String = chars[i..end].iter().collect();
            chunks.push(chunk);
            i = end;
        }

        chunks
    }

    /// Split input at strategic positions to test edge cases
    /// This creates chunks that break at critical positions like after quotes, colons, etc.
    #[allow(
        dead_code,
        reason = "shared test utility not used by all test binaries"
    )]
    pub fn create_strategic_chunks(input: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = input.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            current.push(ch);

            // Break after strategic characters
            let should_break = matches!(ch, '"' | ':' | ',' | '{' | '}' | '[' | ']')
                || (i > 0 && chars[i - 1] == '"' && ch == ' ') // Space after quote
                || current.len() >= 5; // Max 5 chars per chunk

            if should_break && !current.is_empty() {
                chunks.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            chunks.push(current);
        }

        chunks
    }

    /// Create the bug scenario chunks: `{"name": "` arrives in parts
    #[allow(dead_code, reason = "shared test utility not used by all test files")]
    pub fn create_bug_scenario_chunks() -> Vec<&'static str> {
        vec![
            r"{",
            r#"""#,
            r"name",
            r#"""#,
            r":",
            r" ",
            r#"""#,    // Bug occurs here: parser has {"name": "
            r"search", // Use valid tool name
            r#"""#,
            r",",
            r" ",
            r#"""#,
            r"arguments",
            r#"""#,
            r":",
            r" ",
            r"{",
            r#"""#,
            r"query",
            r#"""#,
            r":",
            r" ",
            r#"""#,
            r"test query",
            r#"""#,
            r"}",
            r"}",
        ]
    }
}

/// Create a comprehensive set of test tools covering all parser test scenarios
#[allow(
    dead_code,
    reason = "shared test utility not used by all test binaries"
)]
pub fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "search".to_string(),
                description: Some("Search for information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "location": {"type": "string"},
                        "date": {"type": "string"},
                        "units": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calculate".to_string(),
                description: Some("Perform calculations".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "translate".to_string(),
                description: Some("Translate text".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "to": {"type": "string"},
                        "target_lang": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_current_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "format": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "update_settings".to_string(),
                description: Some("Update settings".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "preferences": {"type": "object"},
                        "notifications": {"type": "boolean"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "ping".to_string(),
                description: Some("Ping service".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "process".to_string(),
                description: Some("Process data".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "count": {"type": "number"},
                        "rate": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "data": {"type": "object"},
                        "text": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "web_search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "number"},
                        "search_type": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_tourist_attractions".to_string(),
                description: Some("Get tourist attractions".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "config".to_string(),
                description: Some("Configuration function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "debug": {"type": "boolean"},
                        "verbose": {"type": "boolean"},
                        "optional": {"type": "null"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "test_func".to_string(),
                description: Some("Test function".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "bool_true": {"type": "boolean"},
                        "bool_false": {"type": "boolean"},
                        "none_val": {"type": "null"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "create".to_string(),
                description: Some("Create resource".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "add".to_string(),
                description: Some("Add operation".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "calc".to_string(),
                description: Some("Calculate".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func1".to_string(),
                description: Some("Function 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "func2".to_string(),
                description: Some("Function 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool1".to_string(),
                description: Some("Tool 1".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "tool2".to_string(),
                description: Some("Tool 2".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "y": {"type": "number"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        },
    ]
}

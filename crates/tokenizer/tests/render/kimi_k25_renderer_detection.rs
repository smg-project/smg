//! Essential coverage for the Kimi-K2.5 tool renderer:
//!   - encoder shapes a nested-object schema with `?` for optional fields
//!   - encoder emits a TS union for `enum` schemas
//!   - encoder preserves `$defs` insertion order (byte-equal with upstream)
//!   - encoder abandons the TS namespace (returns `None`) on unsupported
//!     schemas, so the chat template falls back to JSON tool declarations
//!   - end-to-end: a tokenizer loaded with `KimiK25ForConditionalGeneration`
//!     dispatches tools through the TS-namespace encoder (not the JSON fallback)

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;

use llm_tokenizer::{
    chat_template::ChatTemplateParams, encoders::kimi_k25_tools::encode_tools_to_typescript,
    traits::Tokenizer as TokenizerTrait, TiktokenTokenizer,
};
use serde_json::{json, Value};
use tempfile::TempDir;

const MIN_TIKTOKEN_MODEL: &str = "aGVsbG8= 0\n";

#[test]
fn encoder_renders_nested_object_with_required() {
    let tools: Vec<Value> = serde_json::from_str(
        r#"[{
            "type": "function",
            "function": {
                "name": "create_user",
                "description": "Create a new user record.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name"]
                        }
                    },
                    "required": ["user"]
                }
            }
        }]"#,
    )
    .unwrap();

    let expected = "# Tools\n\n## functions\nnamespace functions {\n// Create a new user record.\ntype create_user = (_: {\n  user: {\n    name: string,\n    age?: number\n  }\n}) => any;\n}\n";
    assert_eq!(
        encode_tools_to_typescript(&tools).as_deref(),
        Some(expected)
    );
}

#[test]
fn encoder_renders_enum_as_union() {
    let tools: Vec<Value> = serde_json::from_str(
        r#"[{
            "type": "function",
            "function": {
                "name": "set_status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"enum": ["active", "paused", "done"]}
                    },
                    "required": ["status"]
                }
            }
        }]"#,
    )
    .unwrap();

    let expected = "# Tools\n\n## functions\nnamespace functions {\ntype set_status = (_: {\n  status: \"active\" | \"paused\" | \"done\"\n}) => any;\n}\n";
    assert_eq!(
        encode_tools_to_typescript(&tools).as_deref(),
        Some(expected)
    );
}

#[test]
fn encoder_preserves_defs_insertion_order() {
    // `$defs` are declared Zebra-then-Apple (non-alphabetical). Upstream emits
    // interfaces in insertion order; a naive sort would flip them to Apple,
    // Zebra and break byte-equivalence.
    let tools: Vec<Value> = serde_json::from_str(
        r##"[{
            "type": "function",
            "function": {
                "name": "zoo",
                "parameters": {
                    "type": "object",
                    "$defs": {
                        "Zebra": {"type": "object", "properties": {"stripes": {"type": "integer"}}, "required": ["stripes"]},
                        "Apple": {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]}
                    },
                    "properties": {
                        "z": {"$ref": "#/$defs/Zebra"},
                        "a": {"$ref": "#/$defs/Apple"}
                    },
                    "required": ["z", "a"]
                }
            }
        }]"##,
    )
    .unwrap();

    let out = encode_tools_to_typescript(&tools).expect("schema is fully supported");
    let zebra = out
        .find("interface Zebra")
        .expect("Zebra interface present");
    let apple = out
        .find("interface Apple")
        .expect("Apple interface present");
    assert!(
        zebra < apple,
        "expected $defs insertion order (Zebra before Apple), got:\n{out}"
    );
}

#[test]
fn encoder_returns_none_on_unresolvable_ref() {
    // `$ref` points at a definition that does not exist. Upstream raises and
    // the caller falls back to JSON; we signal that by returning `None`.
    let tools: Vec<Value> = serde_json::from_str(
        r##"[{
            "type": "function",
            "function": {
                "name": "broken",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/Missing"}},
                    "required": ["x"]
                }
            }
        }]"##,
    )
    .unwrap();

    assert_eq!(encode_tools_to_typescript(&tools), None);
}

#[test]
fn encoder_returns_none_on_unrecognized_schema() {
    // A non-empty property schema with no type/anyOf/enum/$ref is invalid;
    // upstream raises `ValueError`, so we abandon the TS namespace.
    let tools: Vec<Value> = serde_json::from_str(
        r#"[{
            "type": "function",
            "function": {
                "name": "weird",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"title": "no recognized keyword"}},
                    "required": ["x"]
                }
            }
        }]"#,
    )
    .unwrap();

    assert_eq!(encode_tools_to_typescript(&tools), None);
}

#[test]
fn chat_template_renders_typescript_namespace() {
    let dir = TempDir::new().unwrap();
    let template = fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kimi_k25/chat_template.jinja"),
    )
    .expect("vendored chat_template.jinja must exist");
    fs::write(dir.path().join("tiktoken.model"), MIN_TIKTOKEN_MODEL).unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#,
    )
    .unwrap();
    fs::write(dir.path().join("chat_template.jinja"), template).unwrap();
    fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();

    let tok = TiktokenTokenizer::from_dir(dir.path()).expect("tokenizer should load");
    let messages = vec![json!({"role": "user", "content": "what's 2+2?"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Compute an expression.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
        }
    })];

    let rendered = tok
        .apply_chat_template(
            &messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                tools: Some(&tools),
                ..Default::default()
            },
        )
        .expect("render should succeed");

    assert!(
        rendered.contains("namespace functions") && rendered.contains("type calc = (_:"),
        "expected TS namespace block, got:\n{rendered}"
    );
    assert!(
        !rendered.contains(r#"[{"function":"#),
        "rendered prompt fell into JSON fallback path:\n{rendered}"
    );
}

use serde_json::Value;

use super::*;
use crate::factory::ParserFactory;

fn _arg(key: &str, typ: &str, value: &str) -> String {
    format!(r#"{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}"#)
}

fn _call(tool: &str, index: i32, args: &[String]) -> String {
    let body: String = args.concat();
    format!(r#"{OPEN}call tool="{tool}" index="{index}"{SEP}{body}{CLOSE}call{SEP}"#)
}

fn _response(content: &str) -> String {
    format!("{OPEN}response{SEP}{content}{CLOSE}response{SEP}")
}

fn _tools(calls: &[String]) -> String {
    let body: String = calls.concat();
    format!("{OPEN}tools{SEP}{body}{CLOSE}tools{SEP}")
}

#[tokio::test]
async fn test_parse_complete_response_and_typed_arguments() {
    let parser = KimiK3Parser::new();
    let input = _response("answer")
        + &_tools(&[_call(
            "calc",
            1,
            &[
                _arg("x", "number", "1"),
                _arg("flag", "boolean", "true"),
                _arg("text", "string", "raw"),
            ],
        )]);

    let (content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(content, "answer");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "calc");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"x": 1, "flag": true, "text": "raw"})
    );
}

#[tokio::test]
async fn test_parse_complete_unescapes_attributes() {
    let parser = KimiK3Parser::new();
    let input = _tools(&[_call(
        "a&amp;b&quot;c",
        1,
        &[_arg("k&amp;q", "string", "v")],
    )]);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "a&b\"c");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"k&q": "v"}));
}

#[tokio::test]
async fn test_parse_complete_allows_less_than_in_attributes() {
    let parser = KimiK3Parser::new();
    let input = _tools(&[_call("calc<beta", 1, &[_arg("foo<bar", "string", "raw")])]);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "calc<beta");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"foo<bar": "raw"}));
}

#[tokio::test]
async fn test_parse_complete_no_tools_channel() {
    let parser = KimiK3Parser::new();
    let input = _response("hi");

    let (content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(content, "hi");
    assert_eq!(calls.len(), 0);
}

#[tokio::test]
async fn test_parse_complete_whitespace_degraded_markers() {
    let parser = KimiK3Parser::new();
    let input = format!("{OPEN} response {SEP}answer{CLOSE} response {SEP}");

    let (content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(content, "answer");
    assert_eq!(calls.len(), 0);
}

#[tokio::test]
async fn test_parse_complete_malformed_json_argument_falls_back_to_raw() {
    let parser = KimiK3Parser::new();
    let input = _tools(&[_call("calc", 1, &[_arg("x", "number", "not-json")])]);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"x": "not-json"}));
}

#[tokio::test]
async fn test_parse_complete_multiple_tool_calls() {
    let parser = KimiK3Parser::new();
    let input = _tools(&[
        _call("first", 1, &[_arg("a", "number", "1")]),
        _call("second", 2, &[_arg("b", "number", "2")]),
    ]);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "first");
    assert_eq!(calls[1].function.name, "second");
}

#[tokio::test]
async fn test_decode_call_without_index_attribute() {
    let parser = KimiK3Parser::new();
    let input =
        format!(r#"{OPEN}tools{SEP}{OPEN}call tool="calc"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}"#);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "calc");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn test_has_tool_markers() {
    let parser = KimiK3Parser::new();
    assert!(parser.has_tool_markers(&_tools(&[])));
    assert!(!parser.has_tool_markers(&_response("hi")));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_streaming_split_markers_do_not_leak() {
    let mut parser = KimiK3Parser::new();
    let hi_chunk = format!("{SEP}Hi");
    let call_open_chunk = format!(r#"{OPEN}call tool="calc" index="1"{SEP}"#);
    let arg_chunk = _arg("x", "number", "1");
    let close_call_chunk = format!("{CLOSE}call");

    let chunks: [&str; 10] = [
        OPEN,
        "response",
        hi_chunk.as_str(),
        OPEN,
        "tools",
        SEP,
        call_open_chunk.as_str(),
        arg_chunk.as_str(),
        close_call_chunk.as_str(),
        SEP,
    ];

    let mut content = String::new();
    let mut calls = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &[]).await.unwrap();
        content.push_str(&result.normal_text);
        calls.extend(result.calls);
    }

    assert_eq!(content, "Hi");
    assert!(!content.contains(OPEN));
    assert!(!content.contains(SEP));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("calc"));
    let args: Value = serde_json::from_str(&calls[0].parameters).unwrap();
    assert_eq!(args, serde_json::json!({"x": 1}));
}

#[tokio::test]
async fn test_streaming_consumed_response_prefix_no_call_keeps_content() {
    let mut parser = KimiK3Parser::new();
    let close_response_chunk = format!("response{SEP}");
    let chunks: [&str; 4] = ["O", "K", CLOSE, close_response_chunk.as_str()];

    let mut content = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &[]).await.unwrap();
        assert!(!result.normal_text.contains(CLOSE));
        content.push_str(&result.normal_text);
    }
    assert_eq!(content, "OK");
}

#[tokio::test]
async fn test_streaming_multiple_tool_calls_increment_tool_index() {
    let mut parser = KimiK3Parser::new();
    let input = _tools(&[
        _call("first", 1, &[_arg("a", "number", "1")]),
        _call("second", 2, &[_arg("b", "number", "2")]),
    ]);

    let result = parser.parse_incremental(&input, &[]).await.unwrap();
    assert_eq!(result.calls.len(), 2);
    assert_eq!(result.calls[0].tool_index, 0);
    assert_eq!(result.calls[1].tool_index, 1);
}

#[tokio::test]
async fn test_streaming_duplicate_xtml_index_uses_ordinal() {
    // A model that emits colliding, sparse, or out-of-order `index`
    // attributes must still yield unique, monotonic streamed indices: the
    // parser assigns them by emission order and ignores the attribute.
    let mut parser = KimiK3Parser::new();
    let input = _tools(&[
        _call("first", 7, &[_arg("a", "number", "1")]),
        _call("second", 7, &[_arg("b", "number", "2")]),
        _call("third", 3, &[_arg("c", "number", "3")]),
    ]);

    let result = parser.parse_incremental(&input, &[]).await.unwrap();
    assert_eq!(result.calls.len(), 3);
    assert_eq!(result.calls[0].tool_index, 0);
    assert_eq!(result.calls[1].tool_index, 1);
    assert_eq!(result.calls[2].tool_index, 2);
}

#[tokio::test]
async fn test_reset_clears_streaming_state() {
    let mut parser = KimiK3Parser::new();

    let primed = format!("{OPEN}response{SEP}Hello{CLOSE}response{SEP}")
        + &_tools(&[_call("calc", 1, &[_arg("x", "number", "1")])]);
    let primed_result = parser.parse_incremental(&primed, &[]).await.unwrap();
    assert_eq!(primed_result.calls.len(), 1);

    parser.reset();

    let fresh_content = format!("{OPEN}response{SEP}Fresh{CLOSE}response{SEP}");
    let content_result = parser.parse_incremental(&fresh_content, &[]).await.unwrap();
    assert_eq!(content_result.normal_text, "Fresh");

    let fresh_call = _tools(&[_call("calc", 1, &[_arg("x", "number", "2")])]);
    let call_result = parser.parse_incremental(&fresh_call, &[]).await.unwrap();
    assert_eq!(call_result.calls.len(), 1);
    assert_eq!(call_result.calls[0].name.as_deref(), Some("calc"));
}

#[test]
fn test_factory_resolves_moonshot_kimi_k3() {
    let factory = ParserFactory::new();
    assert_eq!(
        factory
            .registry()
            .resolve_model_to_parser("moonshotai/Kimi-K3"),
        Some("kimi_k3".to_string())
    );
}

#[test]
fn test_factory_resolves_underscore_kimi_k3() {
    let factory = ParserFactory::new();
    for id in ["kimi_k3", "Kimi_K3", "moonshotai/Kimi_K3"] {
        assert_eq!(
            factory.registry().resolve_model_to_parser(id),
            Some("kimi_k3".to_string()),
            "expected `{id}` to resolve to the kimi_k3 parser"
        );
    }
}

#[test]
fn test_factory_registers_kimi_k3_structural_tag() {
    let factory = ParserFactory::new();
    assert!(factory.registry().has_structural_tag("kimi_k3"));
}

fn sample_tools() -> Vec<Tool> {
    serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "units": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city", "days"]
            }
        }
    }]))
    .unwrap()
}

/// Build a structural tag from one tool whose parameters are `params`.
fn tag_for_params(params: serde_json::Value) -> Value {
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": { "name": "f", "parameters": params }
    }]))
    .unwrap();
    KimiK3Parser::build_structural_tag(&tools, true)
}

/// Collect every `json_schema` payload embedded in the structural tag.
fn embedded_schemas(node: &Value, out: &mut Vec<Value>) {
    match node {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("json_schema") {
                if let Some(schema) = map.get("json_schema") {
                    out.push(schema.clone());
                }
            }
            for value in map.values() {
                embedded_schemas(value, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                embedded_schemas(item, out);
            }
        }
        _ => {}
    }
}

fn schemas_in(tag: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    embedded_schemas(tag, &mut out);
    out
}

// ---- $defs / $ref propagation -------------------------------------
//
// Every embedded `json_schema` is compiled as its own document, so a
// `#/$defs/...` pointer inside one only resolves if the definition block
// travels with it. These shapes are taken from the walle validator cases
// that the Kimi vendor verifier runs (TestRefInProperties, TestReferences,
// TestNestedDefsDepth), each of which failed the whole tool call with
// "Cannot find field $defs" before the block was carried across.

#[test]
fn nested_ref_inside_array_items_carries_defs() {
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Tag": {"type": "string"} },
        "properties": {
            "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}}
        },
        "required": ["tags"]
    }));

    let schemas = schemas_in(&tag);
    assert!(!schemas.is_empty(), "array argument should embed a schema");
    for schema in &schemas {
        assert!(
            schema.get("$defs").is_some(),
            "embedded schema must carry $defs: {schema}"
        );
    }
}

#[test]
fn bare_ref_property_resolves_to_its_definition() {
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }
        },
        "properties": { "address": {"$ref": "#/$defs/Address"} },
        "required": ["address"]
    }));

    // Resolving the ref lets the argument be typed rather than falling
    // through to the permissive shape.
    let rendered = serde_json::to_string(&tag).unwrap();
    assert!(
        rendered.contains(r#"type=\"object\""#),
        "bare $ref should classify as object: {rendered}"
    );
    assert!(!schemas_in(&tag).is_empty());
}

#[test]
fn recursive_definition_does_not_loop() {
    // `Node.next` points back at `Node`; resolution follows one hop only
    // and the block travels with the schema for the compiler to expand.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "next": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/Node"}]}
                }
            }
        },
        "properties": { "node": {"$ref": "#/$defs/Node"} },
        "required": ["node"]
    }));

    for schema in schemas_in(&tag) {
        assert!(
            schema.get("$defs").is_some(),
            "recursive schema must keep $defs: {schema}"
        );
    }
}

#[test]
fn pointer_into_a_definition_subpath_resolves() {
    // `#/$defs/group/items` addresses a node *inside* a definition.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": {
            "group": {
                "type": "array",
                "items": {"type": "object", "properties": {"data": {"type": "string"}}}
            }
        },
        "properties": { "root": {"$ref": "#/$defs/group/items"} },
        "required": ["root"]
    }));

    let rendered = serde_json::to_string(&tag).unwrap();
    assert!(
        rendered.contains(r#"type=\"object\""#),
        "sub-path pointer should resolve to the object it names: {rendered}"
    );
}

#[test]
fn draft07_definitions_block_is_also_carried() {
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "definitions": { "Item": {"type": "string"} },
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/definitions/Item"}}
        },
        "required": ["items"]
    }));

    for schema in schemas_in(&tag) {
        assert!(
            schema.get("definitions").is_some(),
            "embedded schema must carry definitions: {schema}"
        );
    }
}

#[test]
fn schema_without_refs_is_left_untouched() {
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Unused": {"type": "string"} },
        "properties": { "payload": {"type": "object", "properties": {"a": {"type": "string"}}} },
        "required": ["payload"]
    }));

    for schema in schemas_in(&tag) {
        assert!(
            schema.get("$defs").is_none(),
            "a schema with no $ref should not be padded with $defs: {schema}"
        );
    }
}

#[test]
fn dynamic_ref_spelling_also_carries_defs() {
    // `$dynamicRef` orphans just as `$ref` does when the property is
    // sliced out of the parameters document.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Item": {"type": "string"} },
        "properties": {
            "items": {"type": "array", "items": {"$dynamicRef": "#/$defs/Item"}}
        },
        "required": ["items"]
    }));

    for schema in schemas_in(&tag) {
        assert!(
            schema.get("$defs").is_some(),
            "$dynamicRef must also pull the block across: {schema}"
        );
    }
}

#[test]
fn child_defs_block_keeps_root_definitions_too() {
    // A property that declares a partial block of its own must not lose
    // access to the rest of the tool's definitions.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Root": {"type": "string"} },
        "properties": {
            "p": {
                "type": "object",
                "$defs": { "Local": {"type": "number"} },
                "properties": { "x": {"$ref": "#/$defs/Root"} }
            }
        },
        "required": ["p"]
    }));

    let schemas = schemas_in(&tag);
    assert!(!schemas.is_empty());
    for schema in &schemas {
        let defs = schema["$defs"].as_object().expect("block present");
        assert!(defs.contains_key("Local"), "own entry kept: {schema}");
        assert!(defs.contains_key("Root"), "root entry merged in: {schema}");
    }
}

#[test]
fn root_pointer_resolves_to_the_parameter_schema() {
    // `"$ref": "#"` names the root schema — a root-recursive property.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "child": {"$ref": "#"}
        },
        "required": ["child"]
    }));

    let rendered = serde_json::to_string(&tag).unwrap();
    assert!(
        rendered.contains(r#"key=\"child\" type=\"object\""#),
        "root pointer should classify as object: {rendered}"
    );
}

#[test]
fn sibling_type_wins_over_the_referenced_definition() {
    // In 2020-12 `$ref` applies alongside its siblings; a `type` on the
    // property itself must not be discarded with the pointer.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Payload": {"properties": {"a": {"type": "string"}}} },
        "properties": {
            "p": {"$ref": "#/$defs/Payload", "type": "object"}
        },
        "required": ["p"]
    }));

    let rendered = serde_json::to_string(&tag).unwrap();
    assert!(
        rendered.contains(r#"key=\"p\" type=\"object\""#),
        "sibling type should be honoured: {rendered}"
    );
    // The pointer itself still reaches the compiler.
    for schema in schemas_in(&tag) {
        assert_eq!(schema["$ref"], "#/$defs/Payload");
        assert!(schema.get("$defs").is_some());
    }
}

#[test]
fn nested_dangling_pointer_falls_back_to_permissive() {
    // The pointer sits inside `items`, not at the top of the property, and
    // names something the block does not contain.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "$defs": { "Tag": {"type": "string"} },
        "properties": {
            "tags": {"type": "array", "items": {"$ref": "#/$defs/Missing"}}
        },
        "required": ["tags"]
    }));

    assert!(
        schemas_in(&tag).is_empty(),
        "an unresolvable nested pointer must not be embedded"
    );
}

#[test]
fn dangling_ref_falls_back_to_permissive() {
    // Nothing to resolve against: the argument stays unconstrained rather
    // than emitting a grammar the compiler will reject.
    let tag = tag_for_params(serde_json::json!({
        "type": "object",
        "properties": { "x": {"$ref": "#/$defs/Missing"} },
        "required": ["x"]
    }));
    assert!(schemas_in(&tag).is_empty());
}

#[test]
fn test_build_structural_tag_frames_tools_section() {
    let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
    let format = &tag["format"];
    assert_eq!(format["type"], "triggered_tags");
    assert_eq!(format["triggers"][0], TOOLS_OPEN);
    assert_eq!(format["at_least_one"], true);
    // `stop_after_first` is intentionally unset (mirrors K2) so trailing
    // `<|close|>message<|sep|><|end_of_msg|>` tokens stay valid.
    assert!(format.get("stop_after_first").is_none());

    let section = &format["tags"][0];
    assert_eq!(section["begin"], TOOLS_OPEN);
    assert_eq!(section["end"], TOOLS_CLOSE);
    // A tools section is one or more call blocks.
    assert_eq!(section["content"]["type"], "plus");
}

#[test]
fn test_build_structural_tag_allows_closing_prefilled_channel() {
    // Regression: the prompt leaves `think` or `response` open, and
    // `at_least_one` admits only trigger-prefixed tokens. Each closing
    // marker therefore needs both a trigger and a matching tag, or the
    // model can never close the channel it was left in.
    let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
    let format = &tag["format"];

    let triggers: Vec<&str> = format["triggers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert!(triggers.contains(&THINK_CLOSE), "got: {triggers:?}");
    assert!(triggers.contains(&RESPONSE_CLOSE), "got: {triggers:?}");
    assert!(triggers.contains(&TOOLS_OPEN), "got: {triggers:?}");

    let begins: Vec<&str> = format["tags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["begin"].as_str().unwrap())
        .collect();
    assert_eq!(
        begins,
        vec![
            TOOLS_OPEN,
            &format!("{THINK_CLOSE}{TOOLS_OPEN}")[..],
            &format!("{RESPONSE_CLOSE}{TOOLS_OPEN}")[..],
        ]
    );

    // Every variant frames an identical tools section.
    for section in format["tags"].as_array().unwrap() {
        assert_eq!(section["end"], TOOLS_CLOSE);
        assert_eq!(section["content"]["type"], "plus");
    }
}

#[tokio::test]
async fn test_parse_complete_handles_think_close_prefixed_tools_section() {
    // What the grammar now produces under thinking + forced tools: close
    // the prefilled think channel, then open tools.
    let parser = KimiK3Parser::new();
    let input = THINK_CLOSE.to_string()
        + &_tools(&[_call("get_weather", 1, &[_arg("city", "string", "SF")])]);

    let (_content, calls) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"city": "SF"}));
}

#[test]
fn test_build_structural_tag_pins_call_framing_and_tool_name() {
    let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
    let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
    assert_eq!(call["type"], "sequence");
    let elems = call["elements"].as_array().unwrap();
    assert_eq!(
        elems[0]["value"],
        "<|open|>call tool=\"get_weather\" index=\""
    );
    assert_eq!(elems[1]["type"], "regex");
    assert_eq!(elems[2]["value"], "\"<|sep|>");
    assert_eq!(elems[4]["value"], "<|close|>call<|sep|>");
}

#[test]
fn test_build_structural_tag_strict_arguments_by_type() {
    let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
    let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
    let args = &call["elements"][3];
    assert_eq!(args["type"], "sequence");
    let arg_elems = args["elements"].as_array().unwrap();
    // city (required, string), days (required, number), units (optional, enum).
    assert_eq!(arg_elems.len(), 3);

    // Required string: bare sequence, verbatim body via any_text.
    assert_eq!(arg_elems[0]["type"], "sequence");
    assert_eq!(
        arg_elems[0]["elements"][0]["value"],
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
    );
    assert_eq!(arg_elems[0]["elements"][1]["type"], "any_text");

    // Required integer: rendered as type="number", JSON-schema-constrained value.
    assert_eq!(
        arg_elems[1]["elements"][0]["value"],
        "<|open|>argument key=\"days\" type=\"number\"<|sep|>"
    );
    assert_eq!(arg_elems[1]["elements"][1]["type"], "json_schema");

    // Optional string enum: wrapped in `optional`, value is an `or` of literals.
    assert_eq!(arg_elems[2]["type"], "optional");
    let units = &arg_elems[2]["content"];
    assert_eq!(
        units["elements"][0]["value"],
        "<|open|>argument key=\"units\" type=\"string\"<|sep|>"
    );
    assert_eq!(units["elements"][1]["type"], "or");
    assert_eq!(units["elements"][1]["elements"][0]["value"], "c");
    assert_eq!(units["elements"][1]["elements"][1]["value"], "f");
}

#[test]
fn test_build_structural_tag_escapes_attribute_values() {
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {
            "name": "a&b\"c",
            "parameters": {
                "type": "object",
                "properties": {"k\"q": {"type": "string"}},
                "required": ["k\"q"]
            }
        }
    }]))
    .unwrap();
    let tag = KimiK3Parser::build_structural_tag(&tools, true);
    let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
    assert_eq!(
        call["elements"][0]["value"],
        "<|open|>call tool=\"a&amp;b&quot;c\" index=\""
    );
    let arg = &call["elements"][3]["elements"][0];
    assert_eq!(
        arg["elements"][0]["value"],
        "<|open|>argument key=\"k&quot;q\" type=\"string\"<|sep|>"
    );
}

#[test]
fn test_build_structural_tag_falls_back_when_no_properties() {
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {"name": "ping", "parameters": {"type": "object"}}
    }]))
    .unwrap();
    let tag = KimiK3Parser::build_structural_tag(&tools, false);
    assert_eq!(tag["format"]["at_least_one"], false);
    let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
    // Argument region degrades to a permissive `star` of argument blocks.
    assert_eq!(call["elements"][3]["type"], "star");
}

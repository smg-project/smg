use std::sync::Arc;

use serde_json::{json, Value};

use super::*;

/// The test tokenizer: the renderer that writes one `role: content` line
/// per message, with no checkpoint behind it.
fn tokenizer() -> Arc<dyn Tokenizer> {
    create_tokenizer_from_file("mock").unwrap()
}

fn weather_tool() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters":
        {"type": "object", "properties": {"city": {"type": "string"}}}}}])
}

/// A request for an unmapped model; `tools`, `tool_choice` and
/// `skip_special_tokens` are left out when null.
fn request_json(tools: Value, tool_choice: Value, skip_special_tokens: Value) -> Value {
    let mut request = json!({
        "model": "unmapped-model",
        "messages": [{"role": "user", "content": "hi"}],
    });
    for (key, value) in [
        ("tools", tools),
        ("tool_choice", tool_choice),
        ("skip_special_tokens", skip_special_tokens),
    ] {
        if !value.is_null() {
            request[key] = value;
        }
    }
    request
}

fn preprocess(request: Value) -> PreprocessResult {
    let request: ChatCompletionRequest = serde_json::from_value(request).unwrap();
    preprocess_impl(&request, tokenizer().as_ref()).unwrap()
}

/// Tools under a choice that allows calls keep the engine's special tokens
/// for the response parser; a choice that forces none, a choice that
/// constrains the output to a JSON schema, or no tools at all leave the
/// request's setting, which skips them.
#[test]
fn special_tokens_are_kept_when_a_parser_reads_the_tools() {
    for (tool_choice, skip) in [
        (Value::Null, false),
        (json!("auto"), false),
        (json!("none"), true),
        (json!("required"), true),
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            true,
        ),
    ] {
        let result = preprocess(request_json(
            weather_tool(),
            tool_choice.clone(),
            Value::Null,
        ));
        assert_eq!(
            result.skip_special_tokens, skip,
            "tool_choice {tool_choice}"
        );
        assert!(result.tools_json.is_some());
    }
    for tools in [Value::Null, json!([])] {
        let result = preprocess(request_json(tools.clone(), Value::Null, Value::Null));
        assert!(result.skip_special_tokens, "tools {tools}");
        assert!(result.tools_json.is_none());
    }
}

/// The constraint of a forced call is written as one object keyed by its
/// kind, the shape the Go client reads it in: a named function's is that
/// function's parameters schema; a choice that forces no call writes none.
#[test]
fn a_forced_call_constraint_is_one_object_keyed_by_its_kind() {
    let named = preprocess(request_json(
        weather_tool(),
        json!({"type": "function", "function": {"name": "get_weather"}}),
        Value::Null,
    ));
    let written = named
        .tool_constraints
        .expect("a named function is constrained");
    let constraint: Value = serde_json::from_str(written.to_str().unwrap()).unwrap();
    let object = constraint
        .as_object()
        .unwrap_or_else(|| panic!("one object keyed by the kind, not {constraint}"));
    assert_eq!(object.len(), 1, "{constraint}");
    let schema: Value = serde_json::from_str(
        object
            .get("json_schema")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("a json_schema string, not {constraint}")),
    )
    .unwrap();
    assert_eq!(schema, weather_tool()[0]["function"]["parameters"]);

    let required = preprocess(request_json(weather_tool(), json!("required"), Value::Null));
    let written = required
        .tool_constraints
        .expect("a required call is constrained");
    let constraint: Value = serde_json::from_str(written.to_str().unwrap()).unwrap();
    assert!(constraint["json_schema"].is_string(), "{constraint}");

    let auto = preprocess(request_json(weather_tool(), json!("auto"), Value::Null));
    assert!(auto.tool_constraints.is_none());
}

/// A request that asks for its special tokens keeps them whatever its
/// tools say.
#[test]
fn a_request_that_keeps_its_special_tokens_is_honoured() {
    let plain = preprocess(request_json(Value::Null, Value::Null, json!(false)));
    assert!(!plain.skip_special_tokens);
    let none_choice = preprocess(request_json(weather_tool(), json!("none"), json!(false)));
    assert!(!none_choice.skip_special_tokens);
}

/// A tool declared on a message (a provider profile that defines them)
/// counts as the request-level ones do.
#[test]
fn a_message_declared_tool_keeps_the_special_tokens() {
    let result = preprocess(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": weather_tool()},
            {"role": "user", "content": "hi"}
        ]
    }));
    assert!(!result.skip_special_tokens);
    assert!(result.tools_json.is_some());
}

/// The FFI entry point hands the flag out through its out-parameter, next
/// to the parser's tool list, and tolerates a caller that passes none.
#[test]
fn the_ffi_entry_point_returns_the_flag() {
    let request =
        CString::new(request_json(weather_tool(), Value::Null, Value::Null).to_string()).unwrap();
    let tokenizer_path = CString::new("mock").unwrap();
    for read_flag in [true, false] {
        let mut prompt_text: *mut c_char = ptr::null_mut();
        let mut token_ids: *mut c_uint = ptr::null_mut();
        let mut token_ids_len: usize = 0;
        let mut tool_constraints: *mut c_char = ptr::null_mut();
        let mut tools_json: *mut c_char = ptr::null_mut();
        let mut skip_special_tokens: c_int = -1;
        let mut prompt_tokens: c_int = -1;
        let mut error: *mut c_char = ptr::null_mut();
        // SAFETY: every pointer is a live local or null where the
        // function allows null; the strings are null-terminated.
        let code = unsafe {
            sgl_preprocess_chat_request(
                request.as_ptr(),
                tokenizer_path.as_ptr(),
                &mut prompt_text,
                &mut token_ids,
                &mut token_ids_len,
                &mut tool_constraints,
                &mut tools_json,
                if read_flag {
                    &mut skip_special_tokens
                } else {
                    ptr::null_mut()
                },
                &mut prompt_tokens,
                &mut error,
            )
        };
        assert_eq!(code, SglErrorCode::Success);
        assert_eq!(skip_special_tokens, if read_flag { 0 } else { -1 });
        assert!(prompt_tokens >= 0);
        assert!(tool_constraints.is_null());
        assert!(!tools_json.is_null());
        // SAFETY: the pointers were written by the call above and are
        // freed once.
        unsafe {
            sgl_preprocessed_request_free(
                prompt_text,
                token_ids,
                token_ids_len,
                tool_constraints,
                tools_json,
            );
        }
    }
}

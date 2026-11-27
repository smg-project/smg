use serde_json::json;

use super::*;

/// The generation-header switch reaches an engine only when the client
/// turned it off (`true` is what an engine applies without it) and never
/// reaches a third-party API.
#[test]
fn add_generation_prompt_is_forwarded_only_when_false_and_never_to_providers() {
    let mut on = json!({"model": "m", "add_generation_prompt": true});
    strip_default_sglang_fields(&mut on);
    assert_eq!(on.get("add_generation_prompt"), None);
    let mut off = json!({"model": "m", "add_generation_prompt": false});
    strip_default_sglang_fields(&mut off);
    assert_eq!(off.get("add_generation_prompt"), Some(&json!(false)));
    let mut provider = json!({"model": "m", "add_generation_prompt": false});
    strip_sglang_fields(&mut provider);
    assert_eq!(provider.get("add_generation_prompt"), None);
}

#[test]
fn strip_default_sglang_fields_removes_null_and_engine_default_values() {
    let mut payload = json!({
        "continue_final_message": false,
        "messages": [],
        "model": "test-model",
        "no_stop_trim": true,
        "return_hidden_states": null,
        "separate_reasoning": true,
        "skip_special_tokens": true,
        "stream_reasoning": true,
        "top_k": null
    });

    strip_default_sglang_fields(&mut payload);

    assert_eq!(payload.get("continue_final_message"), None);
    assert_eq!(payload.get("return_hidden_states"), None);
    assert_eq!(payload.get("separate_reasoning"), None);
    assert_eq!(payload.get("skip_special_tokens"), None);
    assert_eq!(payload.get("stream_reasoning"), None);
    assert_eq!(payload.get("top_k"), None);
    assert_eq!(payload.get("no_stop_trim"), Some(&json!(true)));
    assert_eq!(payload.get("model"), Some(&json!("test-model")));
}

/// SGLang defaults these three to `true`, so `false` is the value a
/// client sends on purpose; stripping it made the worker apply the
/// opposite of what was asked (inline reasoning came back separated).
#[test]
fn strip_default_sglang_fields_keeps_explicit_false_on_true_default_fields() {
    let mut payload = json!({
        "model": "test-model",
        "separate_reasoning": false,
        "skip_special_tokens": false,
        "stream_reasoning": false
    });

    strip_default_sglang_fields(&mut payload);

    assert_eq!(payload.get("separate_reasoning"), Some(&json!(false)));
    assert_eq!(payload.get("skip_special_tokens"), Some(&json!(false)));
    assert_eq!(payload.get("stream_reasoning"), Some(&json!(false)));
}

#[test]
fn strip_default_sglang_fields_keeps_explicit_true_on_false_default_fields() {
    let mut payload = json!({
        "continue_final_message": true,
        "ignore_eos": true,
        "no_stop_trim": true,
        "return_hidden_states": true
    });

    strip_default_sglang_fields(&mut payload);

    for field in [
        "continue_final_message",
        "ignore_eos",
        "no_stop_trim",
        "return_hidden_states",
    ] {
        assert_eq!(payload.get(field), Some(&json!(true)), "{field}");
    }
}

/// Fields without a boolean default are stripped only when `null`; a
/// concrete value, even one equal to the engine default, is forwarded.
#[test]
fn strip_default_sglang_fields_keeps_concrete_non_boolean_values() {
    let mut payload = json!({
        "lora_path": null,
        "min_p": 0.0,
        "priority": 0,
        "top_k": -1
    });

    strip_default_sglang_fields(&mut payload);

    assert_eq!(payload.get("lora_path"), None);
    assert_eq!(payload.get("min_p"), Some(&json!(0.0)));
    assert_eq!(payload.get("priority"), Some(&json!(0)));
    assert_eq!(payload.get("top_k"), Some(&json!(-1)));
}

#[test]
fn raw_predicate_agrees_with_value_strip_for_every_field() {
    for field in SGLANG_FIELDS {
        for raw in ["null", "false", "true", "0", "1.5", "\"false\"", "[false]"] {
            let value: Value = serde_json::from_str(raw).expect("literal parses");
            let mut fields = serde_json::Map::new();
            fields.insert((*field).to_string(), value);
            let mut payload = Value::Object(fields);
            strip_default_sglang_fields(&mut payload);

            let value_stripped = payload.get(*field).is_none();
            assert_eq!(
                is_stripped_sglang_default(field, raw),
                value_stripped,
                "field={field} raw={raw}"
            );
            assert_eq!(
                may_carry_stripped_sglang_default(format!("{{\"{field}\":{raw}}}").as_bytes()),
                value_stripped,
                "scan field={field} raw={raw}"
            );
        }
    }
}

/// A client's `rid` never reaches a third-party API and always reaches
/// an engine: the provider strip drops it, the proxy strip keeps a string
/// and drops only a null.
#[test]
fn rid_is_dropped_for_providers_and_kept_for_an_engine() {
    let mut provider = json!({"model": "m", "rid": "conv_1", "messages": []});
    strip_sglang_fields(&mut provider);
    assert!(provider.get("rid").is_none(), "{provider}");

    let mut engine = json!({"model": "m", "rid": "conv_1", "messages": []});
    strip_default_sglang_fields(&mut engine);
    assert_eq!(engine["rid"], json!("conv_1"));

    let mut unset = json!({"model": "m", "rid": null});
    strip_default_sglang_fields(&mut unset);
    assert!(unset.get("rid").is_none(), "{unset}");
    assert!(!is_stripped_sglang_default("rid", "\"conv_1\""));
}

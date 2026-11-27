use serde_json::json;

use super::*;

#[test]
fn set_request_model_replaces_an_existing_model_field() {
    let mut body = json!({"model": "GLM-5.2-Coding", "stream": false});
    set_request_model(&mut body, "GLM-5.2");
    assert_eq!(body, json!({"model": "GLM-5.2", "stream": false}));
}

#[test]
fn set_request_model_leaves_a_body_without_the_field_untouched() {
    let mut body = json!({"text": "Hello"});
    set_request_model(&mut body, "GLM-5.2");
    assert_eq!(body, json!({"text": "Hello"}));
}

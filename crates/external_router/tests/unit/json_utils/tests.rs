use serde_json::json;

use super::*;

#[test]
fn a_field_is_missing_when_absent_null_or_empty() {
    assert!(is_missing_or_empty(None));
    assert!(is_missing_or_empty(Some(&Value::Null)));
    assert!(is_missing_or_empty(Some(&json!(""))));
    assert!(!is_missing_or_empty(Some(&json!("resp_1"))));
    assert!(!is_missing_or_empty(Some(&json!(0))));
    assert!(!is_missing_or_empty(Some(&json!({}))));
}

//! JSON tests the providers' response patching shares.

use serde_json::Value;

/// Whether a response field is missing, null or an empty string: the test a
/// response patch reads before it restores the request's value of the field.
pub(crate) fn is_missing_or_empty(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_null() || v.as_str().is_some_and(|s| s.is_empty()),
    }
}

#[cfg(test)]
mod tests {
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
}

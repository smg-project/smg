use serde::Deserialize;
use serde_json::json;

use super::serde_error_kind;

#[derive(Debug, Deserialize)]
struct Item {
    count: u32,
}

/// The drop warning names the error's kind and never its text: a type
/// mismatch's text quotes the value, a missing field's names the field.
#[test]
fn a_serde_error_is_named_by_its_kind_only() {
    let read: Item = serde_json::from_value(json!({"count": 3})).expect("a count");
    assert_eq!(read.count, 3);

    let mismatch =
        serde_json::from_value::<Item>(json!({"count": "the model's words"})).unwrap_err();
    assert!(
        mismatch.to_string().contains("the model's words"),
        "{mismatch}"
    );
    assert_eq!(serde_error_kind(&mismatch), "type mismatch");

    let missing = serde_json::from_value::<Item>(json!({})).unwrap_err();
    assert_eq!(serde_error_kind(&missing), "missing field");

    let truncated = serde_json::from_str::<Item>("{\"count\": ").unwrap_err();
    assert_eq!(serde_error_kind(&truncated), "other");
}

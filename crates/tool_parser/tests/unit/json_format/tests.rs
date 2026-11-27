use serde_json::json;

use super::*;

#[test]
fn matches_python_json_dumps() {
    // json.dumps({...}, ensure_ascii=False): spaced separators, raw unicode.
    let v = json!({"city": "杭州", "count": 42, "flags": [1, true, null], "x": 1.5});
    assert_eq!(
        to_string(&v),
        r#"{"city": "杭州", "count": 42, "flags": [1, true, null], "x": 1.5}"#
    );
}

#[test]
fn empty_containers_and_nested_json_strings() {
    assert_eq!(to_string(&json!({})), "{}");
    assert_eq!(to_string(&json!([])), "[]");
    // A value that is itself a JSON-looking string stays escaped and unspaced.
    assert_eq!(
        to_string(&json!({"query": "{\"a\": 1}", "limit": 2})),
        r#"{"query": "{\"a\": 1}", "limit": 2}"#
    );
}

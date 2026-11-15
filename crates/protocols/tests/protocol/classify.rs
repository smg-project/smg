//! Classify request bodies, as the HTTP router writes them back for the
//! worker.

use openai_protocol::{classify::ClassifyRequest, common::GenerationRequest};
use serde_json::{json, Value};
use validator::Validate;

/// SGLang's `/v1/classify` takes one `rid` per input and a 64-bit
/// `priority`: a batch with a list of rids and a priority past `i32` reads
/// and is written back as sent, and routes by its first rid. One rid and a
/// negative priority read as before.
#[test]
fn a_classify_request_takes_a_list_of_rids_and_a_64_bit_priority() {
    let batch = json!({
        "model": "m",
        "input": ["a", "b"],
        "rid": ["r1", "r2"],
        "priority": 2_147_483_648_i64
    });
    let request: ClassifyRequest =
        serde_json::from_value(batch.clone()).expect("SGLang's classify body");
    assert_eq!(request.rid(), Some("r1"));
    assert_eq!(serde_json::to_value(&request).expect("serialize"), batch);

    let one = json!({"model": "m", "input": "a", "rid": "r1", "priority": -3});
    let request: ClassifyRequest = serde_json::from_value(one.clone()).expect("one rid");
    assert_eq!(request.rid(), Some("r1"));
    assert_eq!(serde_json::to_value(&request).expect("serialize"), one);
}

/// A list of rids names each input's request, as SGLang runs the i-th input
/// under the i-th id: it takes one id per input (a string or an array of token
/// ids is one input, an array of strings or of token id arrays one per member)
/// and none twice. One id takes any batch, as before.
#[test]
fn a_list_of_rids_takes_one_id_per_input_none_twice() {
    let passes = |input: Value, rid: Value| {
        serde_json::from_value::<ClassifyRequest>(json!({"model": "m", "input": input, "rid": rid}))
            .expect("a classify body")
            .validate()
            .is_ok()
    };
    assert!(passes(json!(["a", "b"]), json!(["r1", "r2"])));
    assert!(passes(json!([[1, 2], [3]]), json!(["r1", "r2"])));
    assert!(passes(json!([1, 2, 3]), json!(["r1"])));
    assert!(passes(json!("a"), json!(["r1"])));
    assert!(passes(json!(["a", "b"]), json!("r")));

    assert!(!passes(json!(["a", "b"]), json!(["r1"])));
    assert!(!passes(json!([1, 2, 3]), json!(["r1", "r2", "r3"])));
    assert!(!passes(json!(["a", "b"]), json!(["r1", "r1"])));
    assert!(!passes(json!(["a", "b"]), json!([])));
}

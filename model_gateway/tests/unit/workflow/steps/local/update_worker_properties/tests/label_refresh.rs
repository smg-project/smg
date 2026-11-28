//! The labels a refresh re-reads: engine-sourced labels follow the engine, an operator's override
//! keeps its value and its window, and a key the engine no longer reports is dropped unless the
//! operator set it.

use std::collections::HashMap;

use super::refreshed_labels;
use crate::workflow::steps::local::test_support::labels_of;

#[test]
fn engine_sourced_labels_follow_the_engine() {
    let discovered = labels_of(&[("max_num_seqs", "256"), ("page_size", "128")]);
    let current = labels_of(&[
        ("max_num_seqs", "256"),
        ("page_size", "128"),
        ("tier", "gold"),
    ]);
    let fresh = labels_of(&[
        ("max_num_seqs", "8"),
        ("page_size", "1152"),
        ("supports_vision", "true"),
    ]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[
            ("max_num_seqs", "8"),
            ("page_size", "1152"),
            ("supports_vision", "true"),
            ("tier", "gold"),
        ]),
        "engine keys follow the engine, a new engine key is added, the operator's label stays"
    );
}

#[test]
fn an_operator_override_keeps_its_value() {
    let discovered = labels_of(&[("served_model_name", "/models/x"), ("max_num_seqs", "256")]);
    let current = labels_of(&[("served_model_name", "m1"), ("max_num_seqs", "256")]);
    let fresh = labels_of(&[("served_model_name", "/models/y"), ("max_num_seqs", "8")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("served_model_name", "m1"), ("max_num_seqs", "8")]),
        "a value the operator set over discovery's is kept; the untouched one follows"
    );
}

#[test]
fn a_label_the_engine_no_longer_reports_is_dropped_unless_the_operator_set_it() {
    let discovered = labels_of(&[("mm_processor", "inprocess"), ("supports_vision", "true")]);
    let current = labels_of(&[
        ("mm_processor", "inprocess"),
        ("supports_vision", "true"),
        ("extra", "1"),
    ]);
    let fresh = labels_of(&[("supports_vision", "false")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("supports_vision", "false"), ("extra", "1")]),
        "a vanished engine key goes, a flipped one follows, an operator key stays"
    );
}

#[test]
fn a_new_engine_key_does_not_override_an_operator_label() {
    let discovered = HashMap::new();
    let current = labels_of(&[("weight_version", "pinned")]);
    let fresh = labels_of(&[("weight_version", "v7"), ("vocab_size", "32000")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("weight_version", "pinned"), ("vocab_size", "32000")]),
    );
}

#[test]
fn a_window_the_operator_set_stays_the_only_window_when_the_engine_names_the_other() {
    // Registration read nothing (or answered during the load without the
    // window) and the operator set the window under the name the reader
    // prefers less; the engine's first report names it under the other.
    let discovered = HashMap::new();
    let current = labels_of(&[("max_running_requests", "64")]);
    let fresh = labels_of(&[("max_num_seqs", "256"), ("page_size", "128")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("max_running_requests", "64"), ("page_size", "128")]),
        "the engine's window under the other name is not added beside the operator's"
    );
    // The same window name the operator overrode: the operator's value, as
    // registration and a label update keep it.
    let discovered = labels_of(&[("max_num_seqs", "256")]);
    let current = labels_of(&[("max_num_seqs", "64")]);
    let fresh = labels_of(&[("max_num_seqs", "512"), ("max_running_requests", "512")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("max_num_seqs", "64")]),
        "a hand-set window drops the engine's figure under either name"
    );
    // A window discovery recorded and the operator left alone follows the
    // engine under the same name, as before.
    let discovered = labels_of(&[("max_num_seqs", "256")]);
    let current = labels_of(&[("max_num_seqs", "256")]);
    let fresh = labels_of(&[("max_num_seqs", "8")]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        labels_of(&[("max_num_seqs", "8")]),
    );
}

/// The context window follows the same rule as the running window: a limit
/// the operator set under one of its names stays the worker's only context
/// window when the engine's report (a ZMQ handshake's, a model-info
/// endpoint's) names a wider one under the name the reader prefers.
#[test]
fn a_context_window_the_operator_set_stays_the_only_window_when_the_engine_names_the_other() {
    let discovered = labels_of(&[("constrained_decoding_start", "first_token")]);
    let current = labels_of(&[
        ("constrained_decoding_start", "first_token"),
        ("max_model_len", "2048"),
    ]);
    let fresh = labels_of(&[
        ("constrained_decoding_start", "first_token"),
        ("max_context_length", "4096"),
    ]);
    assert_eq!(
        refreshed_labels(&current, &discovered, &fresh),
        current,
        "the engine's window under the preferred name is not added beside the operator's limit"
    );
}

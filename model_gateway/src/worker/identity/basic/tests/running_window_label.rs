//! The running-window label: a width beyond the count's type saturates, a zero or a word reads as
//! none, and max_num_seqs still wins.

use std::collections::HashMap;

use crate::worker::{BasicWorkerBuilder, Worker};

fn with_labels(pairs: &[(&str, &str)]) -> impl Worker {
    BasicWorkerBuilder::new("http://w:1")
        .labels(
            pairs
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect::<HashMap<_, _>>(),
        )
        .build()
}

/// A window label wider than the count's type (an explicit flag above
/// 65535 on a large engine) saturates, as the widest window the gate and
/// the capacity sum can use, instead of reading as no window at all; a
/// zero or a word still reads as none, and `max_num_seqs` still wins.
#[test]
fn a_window_label_wider_than_the_count_saturates_instead_of_reading_as_none() {
    for (value, expected) in [
        ("8", Some(8)),
        ("65535", Some(65535)),
        ("70000", Some(65535)),
        ("0", None),
        ("none", None),
    ] {
        let worker = with_labels(&[("max_running_requests", value)]);
        assert_eq!(worker.max_running_requests(), expected, "{value}");
    }
    let both = with_labels(&[("max_num_seqs", "16"), ("max_running_requests", "70000")]);
    assert_eq!(both.max_running_requests(), Some(16));
}

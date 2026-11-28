//! The running-window label: a width beyond the count's type saturates, a zero or a word reads as
//! none, and max_num_seqs still wins.

use std::{collections::HashMap, sync::Arc};

use crate::{identity::basic::Worker, test_support::BasicWorkerBuilder};

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

/// The window is read where a worker is built: its spec is set once there
/// and not written afterwards. Labels written in place on a built worker
/// leave the window it was built with, and a worker rebuilt from them, as a
/// label update rebuilds one, reads the new window.
#[test]
fn a_rebuilt_worker_reads_a_new_window_and_a_label_written_in_place_moves_none() {
    let mut worker = BasicWorkerBuilder::new("http://w:1")
        .label("max_num_seqs", "8")
        .build();
    Arc::make_mut(&mut worker.metadata.spec)
        .labels
        .insert("max_num_seqs".to_string(), "32".to_string());
    assert_eq!(worker.max_running_requests(), Some(8));

    let rebuilt = BasicWorkerBuilder::carrying(&(Arc::new(worker) as Arc<dyn Worker>)).build();
    assert_eq!(rebuilt.max_running_requests(), Some(32));
}

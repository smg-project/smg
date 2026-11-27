//! `keep_histograms` answers for the installed store, and this binary
//! installs none: a host that asks before it installed a recorder is told
//! that the histogram families stay with the exporter.

#![cfg(feature = "runtime")]

use smg_worker::metrics::series;

#[test]
fn with_no_store_installed_the_answer_is_no() {
    assert!(!series::keep_histograms());
}

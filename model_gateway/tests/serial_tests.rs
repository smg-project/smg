//! Integration tests marked `#[serial]`: each needs the process to itself while it runs, so they share a binary with nothing else

pub use smg_test_harness as common;

mod serial;

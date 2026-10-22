//! Integration tests marked `#[serial]`: each needs the process to itself while it runs, so they share a binary with nothing else

#[path = "common/mod.rs"]
pub mod common;

mod serial;

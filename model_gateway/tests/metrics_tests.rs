//! Metrics integration tests that drive the recorder `start_prometheus` installs: one binary, the
//! recorder installed once per process (a second install panics), every test taking the handle from
//! `recorder()` and reading series of its own.

#[path = "common/mod.rs"]
pub mod common;

mod metrics;

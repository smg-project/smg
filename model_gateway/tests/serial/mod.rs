//! Integration tests marked `#[serial]`: each needs the process to itself while it runs, so they share a binary with nothing else

mod scheduler_admission_test;
mod zmq_backend_test;

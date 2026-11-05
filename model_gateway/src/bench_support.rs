//! What the crate's benchmarks (`benches/`) reach into the crate for: the
//! seams the per-token paths are measured through. Compiled for the crate's
//! own test targets only (the `test-util` feature the self dev-dependency
//! turns on for them, the benchmarks among them), never into a production
//! build.

pub use crate::routers::grpc::regular::streaming::bench::{EngineScript, StreamBench};

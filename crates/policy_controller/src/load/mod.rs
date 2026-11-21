//! Worker load: the report a worker sends, how a stage reads it, and the named
//! values it carries.

#[cfg(any(feature = "ingest", feature = "worker-host"))]
pub mod load_report;
pub mod reports;
pub mod signals;

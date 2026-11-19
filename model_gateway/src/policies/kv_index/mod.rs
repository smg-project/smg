//! The gateway's code around the KV index: the monitor, whose index side
//! is the host of the worker crate's KV event collector and applies the
//! workers' cache events, and the index backend the cache-aware policy
//! reads. The index's data structure is the `kv_index` crate, whose name
//! this module shares.

pub mod event_monitor;
pub mod index_backend;

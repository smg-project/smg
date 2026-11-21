//! The worker's wire types: what a worker is on the gateway's admin API and
//! in its registry (its spec, its info, its status, its load report), the
//! model card a worker serves, and the entry a card becomes in a model
//! listing.
//!
//! `openai-protocol` re-exports these, and [`UNKNOWN_MODEL_ID`], at the paths
//! they had there.
//!
//! [`settings`] holds the vocabulary the worker's own code shares with a
//! host's configuration; the gateway's configuration module re-exports each
//! item at the path it had there. [`labels`] holds the label names the
//! worker's own code and its hosts read, and the flat map an engine's answer
//! becomes.
//!
//! [`metrics_labels`] holds the metric label values the worker's writers
//! share with a host's; [`worker::WorkerType::as_str`] and
//! [`worker::ConnectionMode::as_str`] spell the two that a type carries.
//!
//! [`identity`] holds a worker's address, parsed once and keyed, and the
//! sampling defaults a backend reports on a worker's labels; [`health`]
//! holds the thresholds past which a load report marks a worker overloaded.
//! The gateway re-exports each module at the path it had there.
//!
//! [`kv_events`] holds the port between the collector of the KV cache events
//! an engine pushes and a host's sink: the sink's calls and the facts that
//! cross them. The collector itself comes with the `runtime` feature: it
//! keeps one subscription task a worker, over the engine handle's two stream
//! calls, for a host that says what an events subscription feeds.
//! The `runtime` feature also adds the rules for applying events to an index.
//!
//! The `metrics` feature adds `metrics`, the mechanism those writers share
//! with a host's: the bounded label set, the model labels' one set and the
//! handles a record keeps.
//!
//! The `runtime` feature adds the worker's runtime, for the process that
//! hosts the workers: the worker itself with its builder and its metadata
//! (`identity`), its runtime state (`runtime`), its liveness, its circuit
//! breaker, its resilience, the fleet's lifecycle and the probe of a
//! bootstrap port (`health`), its requests in flight, the load monitor, the
//! load snapshots it publishes and the fleet's capacity (`load`),
//! its pairing descriptor (`pd`), the registry with its events (`registry`),
//! the metric writers with their series store (`metrics`), and `connection`:
//! the engine handle a host implements, the HTTP client factory and the cache
//! of worker-directed clients. The `jobs` module holds a queue that runs the
//! jobs its host supplies and takes requests through the worker jobs port.
//! Beside them stand the errors a worker operation reports
//! (`identity::error`), `periodic_task`, a task on a thread
//! of its own, and the merge of the metrics scraped from the workers. The
//! gateway re-exports each module at the path it had there. `test-util` adds
//! `test_support`, what a test builds its workers with, and opens the test
//! readers and hooks the registry, the series store, the label interner and
//! the KV event collector and job queue keep for their own cases.
//!
//! The `workflow` feature adds `workflow`, the worker side of the workflows
//! a host runs over its workers on the workflow engine: the steps of a
//! worker's removal and of a registration's detection and creation. It holds
//! their order, the data they pass along and the host trait they read.
//!
//! The `proto` feature adds `common_proto`, the messages every engine's gRPC
//! service shares, and the conversion of the engine load record into the
//! per-rank snapshot of a load report.

/// Default model identifier used when no model is specified.
///
/// This constant should be used instead of hardcoded "unknown" strings
/// throughout the codebase for consistency.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

#[cfg(feature = "runtime")]
pub mod connection;
pub mod health;
pub mod identity;
#[cfg(feature = "runtime")]
pub mod jobs;
pub mod kv_events;
pub mod labels;
#[cfg(feature = "runtime")]
pub mod load;
#[cfg(feature = "proto")]
mod load_record;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod metrics_labels;
pub mod model_card;
pub mod model_type;
pub mod models;
#[cfg(feature = "runtime")]
pub mod pd;
#[cfg(feature = "runtime")]
pub mod periodic_task;
#[cfg(feature = "runtime")]
pub mod registry;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod settings;
#[cfg(all(feature = "runtime", any(test, feature = "test-util")))]
pub mod test_support;
mod value_bounds;
pub mod worker;
#[cfg(feature = "workflow")]
pub mod workflow;

/// The messages every engine's gRPC service shares (`proto/common.proto`,
/// package `smg.grpc.common`): the tokenizer's chunks, the KV cache events,
/// the engine load record, the cache flush, the profiling calls and the
/// tensor handles. `smg-grpc-client` re-exports the module at the path it had
/// there.
#[cfg(feature = "proto")]
pub mod common_proto {
    #![allow(
        clippy::all,
        clippy::absolute_paths,
        clippy::trivially_copy_pass_by_ref,
        unused_qualifications
    )]
    include!(concat!(env!("OUT_DIR"), "/smg.grpc.common.rs"));
}

// The README's Rust blocks, run as doc tests. Its host blocks name the
// runtime, so the page is read where the runtime is built.
#[cfg(all(doctest, feature = "runtime"))]
#[doc = include_str!("../README.md")]
mod readme {}

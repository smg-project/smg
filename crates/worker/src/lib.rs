//! The worker's wire types: what a worker is on the gateway's admin API and
//! in its registry (its spec, its info, its status, its load report), the
//! model card a worker serves, and the entry a card becomes in a model
//! listing.
//!
//! `openai-protocol` re-exports these, and [`UNKNOWN_MODEL_ID`], at the paths
//! they had there.
//!
//! [`settings`] and [`labels`] hold the vocabulary the worker's own code
//! shares with a host's configuration. The gateway re-exports each item at
//! the path it had there.
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
//! of worker-directed clients. Beside them stand the errors a worker
//! operation reports (`identity::error`), `periodic_task`, a task on a thread
//! of its own, and the merge of the metrics scraped from the workers. The
//! gateway re-exports each module at the path it had there. `test-util` adds
//! `test_support`, what a test builds its workers with, and opens the test
//! readers and hooks the registry, the series store and the label interner
//! keep for their own cases.

/// Default model identifier used when no model is specified.
///
/// This constant should be used instead of hardcoded "unknown" strings
/// throughout the codebase for consistency.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

#[cfg(feature = "runtime")]
pub mod connection;
pub mod health;
pub mod identity;
pub mod labels;
#[cfg(feature = "runtime")]
pub mod load;
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
pub mod worker;

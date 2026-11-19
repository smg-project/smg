//! The KV cache events a worker's engine pushes, on their way to a host's
//! sink.
//!
//! [`port`] states what crosses between the two sides: the collector says
//! what a worker's stream brought, and the host's sink
//! ([`port::KvEventSink`]) keeps each worker's stream state and applies what
//! its cursor admits. The port names no generated message and needs no
//! feature.
//!
//! The `runtime` feature adds the collector's side. `KvEventCollector` runs
//! one worker's subscription over the engine handle's two stream calls: it
//! connects, reads the stream, reconnects with backoff, hands the pushed
//! load records to the load monitor (`PushedLoadSink`), says to the sink
//! what the stream brought, and says to it that the worker left. A host
//! starts one loop a worker (`KvEventCollector::subscription_loop`), of the
//! `SubscriptionKind` it wants, and ends it at the worker's removal.
//! `batch_facts` and `batch_rank` read a batch's header as the collector
//! says it to a sink.

pub mod port;
#[cfg(feature = "runtime")]
mod subscription;

#[cfg(feature = "runtime")]
pub use subscription::{
    batch_facts, batch_rank, KvEventCollector, PushedLoadSink, SubscriptionKind,
};
/// The bound on the removals that run at once and its permits, for a host's
/// cases of what a removal holds up.
#[cfg(all(feature = "runtime", any(test, feature = "test-util")))]
pub use subscription::{INDEX_REMOVAL_PERMITS, MAX_CONCURRENT_INDEX_REMOVALS};

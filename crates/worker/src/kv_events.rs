//! The KV cache events a worker's engine pushes, on their way to a host's
//! sink.
//!
//! [`port`] states what crosses between the two sides: the collector says
//! what a worker's stream brought, and the host's sink
//! ([`port::KvEventSink`]) keeps each worker's stream state and applies what
//! its cursor admits. The port names no generated message and needs no
//! feature.
//!
//! The `runtime` feature adds the collector's side. `KvEventCollector` holds
//! the workers' subscriptions: a host adds a worker with the
//! `SubscriptionKind` it wants and removes it, and the collector runs one
//! task a worker. The task connects, reads the stream, reconnects with
//! backoff, hands the pushed load records to the load monitor, says to the
//! sink what the stream brought, and says to it that the worker left. What
//! an events subscription feeds is the host's (`KvEventHost`).
//! With `runtime`, `apply` also holds the rules that put events into an index.

#[cfg(feature = "runtime")]
pub mod apply;
#[cfg(feature = "runtime")]
mod collector;
pub mod port;
#[cfg(feature = "runtime")]
mod subscription;

#[cfg(feature = "runtime")]
pub use collector::{KvEventCollector, KvEventHost};
/// A batch's header as the collector says it to a sink, for a host's cases
/// of its sink.
#[cfg(all(feature = "runtime", any(test, feature = "test-util")))]
pub use subscription::{batch_facts, batch_rank};
#[cfg(feature = "runtime")]
pub use subscription::{KvEventFeed, SubscriptionKind};
/// The bound on the removals that run at once and its permits, for a host's
/// cases of what a removal holds up.
#[cfg(all(feature = "runtime", any(test, feature = "test-util")))]
pub use subscription::{MAX_CONCURRENT_SINK_REMOVALS, SINK_REMOVAL_PERMITS};

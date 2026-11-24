//! The replica's worker objects under the controller: their holder over
//! the controller's store, the lead of a replica that elects, and the
//! health probes and the load monitor it runs while it leads.

mod holder;
mod lead;
mod load_monitor;
mod prober;

#[cfg(all(test, feature = "discovery"))]
pub(crate) use holder::tests::engine;
pub(crate) use holder::WorkerObjects;
pub(crate) use lead::ElectedLead;
pub(crate) use load_monitor::LoadMonitor;
pub(crate) use prober::Prober;

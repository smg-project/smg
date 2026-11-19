//! The hash rings the gateway keeps between membership generations, and the
//! task that fills a generation's ring off the request path.
//!
//! A generation's ring is the ring over its members' URLs. It lives in the
//! gateway's value of the generation ([`ModelWorkerSnapshot::hash_ring`]):
//! filled once, read by every request of the generation, dropped with it.
//! What stands here is the last ring the warm task filled for each model, so
//! the next generation's ring is reconciled from it ([`HashRing::updated`]
//! hashes only the URLs the kept ring does not hold) instead of built whole.
//! A generation published again over the same members, as a refresh of
//! model cards does without an event, takes the kept ring itself.
//!
//! A kept ring is a matter of cost, never of content: whatever ring a fill
//! starts from, the cell holds the ring of its own generation's members. So
//! no lock orders a fill against a membership change, and a kept ring older
//! than the generation before costs a longer reconcile, never a placement.
//!
//! A request reads these rings and writes none. The task
//! ([`KeptRings::keep_warm`]) wakes on the registry's events, fills the ring
//! of each generation a membership change published, and keeps it: a wave of
//! registrations is reconciled off the request path, and a request fills a
//! ring only where it reaches a generation before the task does.

use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};

use super::hash_ring::HashRing;
use crate::worker::{
    registry::{event::WorkerEvent, worker_model_ids},
    ModelWorkerSnapshot, WorkerRegistry, WorkerRegistryExt, UNKNOWN_MODEL_ID,
};

/// The last ring the warm task filled for each model the registry holds,
/// the model-less ring under [`UNKNOWN_MODEL_ID`].
#[derive(Debug, Default)]
pub(crate) struct KeptRings {
    kept: DashMap<Box<str>, Arc<HashRing>, FxBuildHasher>,
}

impl KeptRings {
    /// The ring of `generation`, the registry's generation for `model_id`:
    /// `None` for a generation without a member, as a model nobody serves
    /// has no ring. The first read of a generation the task has not reached
    /// fills its ring from the ring kept under `model_id`.
    pub(crate) fn of(
        &self,
        model_id: &str,
        generation: &ModelWorkerSnapshot,
    ) -> Option<Arc<HashRing>> {
        if generation.is_empty() {
            return None;
        }
        let ring = generation.hash_ring(|| self.kept.get(model_id).map(|kept| Arc::clone(&kept)));
        Some(Arc::clone(ring))
    }

    /// Fills the ring of the generation that stands for `model_id` and keeps
    /// it for the next one; a model without a member keeps none.
    fn warm(&self, registry: &WorkerRegistry, model_id: &str) {
        match self.of(model_id, &registry.get_routing_snapshot(model_id)) {
            Some(ring) => {
                self.kept.insert(model_id.into(), ring);
            }
            None => {
                self.kept.remove(model_id);
            }
        }
    }

    /// Starts the task that keeps the rings warm on `runtime`. It wakes on
    /// the registry's events; for each model a registration, a removal or a
    /// replacement names, and for the model-less ring, it fills the ring of
    /// the generation that stands and keeps it. Events that wait together
    /// are one fill, so a wave is reconciled a batch at a time. At its start
    /// and after a lag it fills every model the registry holds and drops the
    /// kept rings of the models that left.
    ///
    /// The task holds the registry and these rings weakly, and ends with
    /// either.
    pub(crate) fn keep_warm(
        self: &Arc<Self>,
        registry: &Arc<WorkerRegistry>,
        runtime: &tokio::runtime::Handle,
    ) {
        let mut events = registry.subscribe_events();
        let rings = Arc::downgrade(self);
        let registry = Arc::downgrade(registry);
        runtime.spawn(async move {
            // What the registry held before the task listened is one fill,
            // as what a lag missed.
            let mut changed = Changed {
                everything: true,
                models: Vec::new(),
            };
            loop {
                loop {
                    match events.try_recv() {
                        Ok(event) => changed.note(&event),
                        Err(TryRecvError::Lagged(_)) => changed.everything = true,
                        Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                    }
                }
                {
                    let (Some(rings), Some(registry)) = (rings.upgrade(), registry.upgrade())
                    else {
                        return;
                    };
                    if changed.everything {
                        changed.models = registry.get_models();
                        rings.kept.retain(|model, _| {
                            **model == *UNKNOWN_MODEL_ID || registry.contains_model(model)
                        });
                    }
                    changed.models.sort_unstable();
                    changed.models.dedup();
                    for model in &changed.models {
                        rings.warm(&registry, model);
                        tokio::task::yield_now().await;
                    }
                    // A status change alone ends no generation.
                    if changed.everything || !changed.models.is_empty() {
                        rings.warm(&registry, UNKNOWN_MODEL_ID);
                    }
                }
                changed = Changed::default();
                match events.recv().await {
                    Ok(event) => changed.note(&event),
                    Err(RecvError::Lagged(_)) => changed.everything = true,
                    Err(RecvError::Closed) => return,
                }
            }
        });
    }
}

/// The models whose generation the events of one wake ended.
#[derive(Default)]
struct Changed {
    /// Events were missed: every model the registry holds.
    everything: bool,
    models: Vec<String>,
}

impl Changed {
    fn note(&mut self, event: &WorkerEvent) {
        match event {
            WorkerEvent::Registered { worker, .. } | WorkerEvent::Removed { worker, .. } => {
                self.models.extend(worker_model_ids(worker));
            }
            WorkerEvent::Replaced { old, new, .. } => {
                self.models.extend(worker_model_ids(old));
                self.models.extend(worker_model_ids(new));
            }
            // A status change publishes no generation.
            WorkerEvent::StatusChanged { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests;

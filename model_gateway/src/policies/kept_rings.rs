//! The hash rings the gateway keeps between membership generations, and the
//! task that fills a generation's ring off the request path.
//!
//! A generation's ring is the ring over its members' URLs. It lives in the
//! gateway's value of the generation ([`ModelWorkerSnapshot::hash_ring`]):
//! filled once, read by every request of the generation, dropped with it.
//! What stands here is the last ring the warm task filled for each model,
//! and it serves the next generation twice. That generation's ring is
//! reconciled from the kept one ([`HashRing::updated`] hashes only the URLs
//! the kept ring does not hold) instead of built whole. And until that ring
//! is filled, its requests are placed on the kept one: a request reads a
//! ring and builds none.
//!
//! So a request can be placed on the ring of another membership than its
//! own. The policies that walk a ring take a URL only where the request's
//! workers hold an eligible worker at it, and a URL's places on a ring
//! depend on the URL alone. A worker that left is therefore passed over at
//! once, and a worker that joined takes no keyed request until its ring is
//! filled. A key lands where the generation's own ring puts it, unless
//! that ring gives it to a worker that joined. Such a key stays, until the
//! fill, with the worker that held it, or with the next one on the kept
//! ring where that worker left. Where no worker of the kept ring is
//! eligible, a key is placed as with no ring, and the worker that joined
//! can take it. The
//! registry publishes a generation before it sends the event the task
//! wakes on, so the task fills that generation or a newer one. The task
//! alone writes a kept ring, from the generation that stands, so a kept
//! ring never steps back to an earlier membership, and trails the one that
//! stands by one fill. A request writes none. A model's first requests,
//! before a ring is kept for it, are placed with no ring.
//!
//! A generation can be published again over the same list with no event, as
//! a refresh of model cards does, and the task does not come to it. The
//! kept ring is its ring, and the first read puts it into the generation's
//! cell ([`ModelWorkerSnapshot::put_hash_ring`]). A ring keeps its URLs in
//! an order of its own, which a replacement moves apart from its list's, so
//! a kept ring stands with the list it was filled over and is known by it.
//!
//! The task ([`KeptRings::keep_warm`]) wakes on the registry's events, and
//! fills and keeps the ring of each generation a membership change
//! published. Where no task lives, the first request of a generation fills
//! its ring itself.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use dashmap::DashMap;
use openai_protocol::UNKNOWN_MODEL_ID;
use rustc_hash::FxBuildHasher;
use smg_worker::registry::{event::WorkerEvent, worker_model_ids};
use tokio::sync::broadcast::{
    self,
    error::{RecvError, TryRecvError},
};
use tracing::{debug, error};

use super::hash_ring::HashRing;
use crate::routers::common::snapshot::{ModelWorkerSnapshot, WorkerRegistry, WorkerRegistryExt};

/// The last ring the warm task filled for each model the registry holds,
/// the model-less ring under [`UNKNOWN_MODEL_ID`].
#[derive(Debug, Default)]
pub(crate) struct KeptRings {
    kept: DashMap<Box<str>, Kept, FxBuildHasher>,
    /// How many warm tasks live ([`Warming`]). A request reads it to learn
    /// who fills a generation's ring. A count read a moment late places no
    /// request wrongly: a ring a request fills is its generation's own.
    warming: Arc<AtomicUsize>,
}

impl KeptRings {
    /// The ring `model_id`'s requests are placed on, for `generation`, the
    /// registry's generation of the model: `None` for a generation without
    /// a member, as a model nobody serves has no ring. It is the
    /// generation's own ring once that is filled. Until then it is the ring
    /// kept for the model, of another membership, and `None` where none
    /// is kept yet (see the module).
    pub(crate) fn of(
        &self,
        model_id: &str,
        generation: &ModelWorkerSnapshot,
    ) -> Option<Arc<HashRing>> {
        if generation.is_empty() {
            return None;
        }
        match generation.hash_ring() {
            Some(ring) => Some(Arc::clone(ring)),
            None => self.standing(model_id, generation),
        }
    }

    /// [`Self::of`] for a generation whose ring is not filled yet. Out of
    /// line, so the read of a filled ring pays nothing for it.
    #[cold]
    fn standing(&self, model_id: &str, generation: &ModelWorkerSnapshot) -> Option<Arc<HashRing>> {
        if self.warming.load(Ordering::Relaxed) == 0 {
            // No task will fill this generation's ring: this request does.
            let kept = self.kept_for(model_id);
            return Some(Arc::clone(generation.fill_hash_ring(kept)));
        }
        let kept = self.kept.get(model_id).map(|kept| Kept::clone(&kept))?;
        // A generation published again over the same list sends no event,
        // so the task never comes to it. The kept ring is its ring, and is
        // put there for the next read.
        generation.put_hash_ring(&kept.ring, &kept.listed);
        Some(kept.ring)
    }

    /// Whether a request placed now walks the ring of its own membership:
    /// no task lives, so the request fills that ring itself, or every
    /// generation that stands holds its ring. It reads as a request reads,
    /// so a generation published again with no event takes its kept ring
    /// here. For a caller that changed the membership and waits for the
    /// placements to follow, as a test bed does.
    #[cfg(any(test, feature = "test-util"))]
    pub(crate) fn are_current(&self, registry: &WorkerRegistry) -> bool {
        if self.warming.load(Ordering::Relaxed) == 0 {
            return true;
        }
        registry
            .get_models()
            .iter()
            .map(String::as_str)
            .chain([UNKNOWN_MODEL_ID])
            .all(|model_id| {
                let generation = registry.get_routing_snapshot(model_id);
                generation.is_empty()
                    || (self.of(model_id, &generation).is_some()
                        && generation.hash_ring().is_some())
            })
    }

    /// The ring kept for `model_id`.
    fn kept_for(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.kept.get(model_id).map(|kept| Arc::clone(&kept.ring))
    }

    /// Fills the ring of the generation that stands for `model_id` and keeps
    /// it, for the next generation's requests and for its fill to start
    /// from; a model without a member keeps none.
    fn warm(&self, registry: &WorkerRegistry, model_id: &str) {
        let generation = registry.get_routing_snapshot(model_id);
        if generation.is_empty() {
            self.kept.remove(model_id);
            return;
        }
        let started = Instant::now();
        let start = self.start(registry, model_id, &generation);
        let ring = Arc::clone(generation.fill_hash_ring(start));
        debug!(
            model_id,
            workers = ring.worker_count(),
            elapsed = ?started.elapsed(),
            "Filled the hash ring"
        );
        let listed = generation.listed_urls(&ring);
        self.kept.insert(model_id.into(), Kept { ring, listed });
    }

    /// The ring a fill of `generation`, the one that stands for `model_id`,
    /// starts from: the ring kept for the model. The model-less ring is
    /// over every worker, so where one model holds them all its fill starts
    /// from that model's ring and hashes no URL a second time.
    fn start(
        &self,
        registry: &WorkerRegistry,
        model_id: &str,
        generation: &ModelWorkerSnapshot,
    ) -> Option<Arc<HashRing>> {
        if model_id == UNKNOWN_MODEL_ID {
            if let [only] = registry.get_models().as_slice() {
                let ring = self.kept_for(only);
                if ring
                    .as_ref()
                    .is_some_and(|ring| ring.worker_count() == generation.len())
                {
                    return ring;
                }
            }
        }
        self.kept_for(model_id)
    }

    /// The task's step for what one wake heard: it fills and keeps the ring
    /// of each model named, then the model-less ring. After events were
    /// missed it fills every model the registry holds and drops the kept
    /// rings of the models that left.
    fn fill(&self, registry: &WorkerRegistry, mut changed: Changed) {
        if changed.everything {
            changed.models = registry.get_models();
            self.kept
                .retain(|model, _| **model == *UNKNOWN_MODEL_ID || registry.contains_model(model));
        }
        changed.models.sort_unstable();
        changed.models.dedup();
        for model in &changed.models {
            self.warm(registry, model);
        }
        // A status change alone ends no generation.
        if changed.everything || !changed.models.is_empty() {
            self.warm(registry, UNKNOWN_MODEL_ID);
        }
    }

    /// Starts the task that keeps the rings warm on `runtime`. It wakes on
    /// the registry's events; for each model a registration, a removal or a
    /// replacement names, and for the model-less ring, it fills the ring of
    /// the generation that stands and keeps it ([`Self::fill`]). Events that
    /// wait together are one fill, so a wave is reconciled a batch at a
    /// time. At its start and after a lag it fills every model the registry
    /// holds. A fill hashes, tens of milliseconds for a large fleet built
    /// whole, so it runs off the runtime's own threads.
    ///
    /// The task holds the registry and these rings weakly, and ends with
    /// either, or with its runtime. While it lives the requests leave every
    /// fill to it ([`Warming`]).
    pub(crate) fn keep_warm(
        self: &Arc<Self>,
        registry: &Arc<WorkerRegistry>,
        runtime: &tokio::runtime::Handle,
    ) {
        let mut events = registry.subscribe_events();
        let rings = Arc::downgrade(self);
        let registry = Arc::downgrade(registry);
        let warming = Warming::start(&self.warming);
        runtime.spawn(async move {
            // Held to the task's end, however it ends.
            let _warming = warming;
            // What the registry held before the task listened is one fill,
            // as what a lag missed.
            let mut changed = Changed {
                everything: true,
                models: Vec::new(),
            };
            loop {
                changed.drain(&mut events);
                let (Some(rings), Some(registry)) = (rings.upgrade(), registry.upgrade()) else {
                    return;
                };
                let heard = std::mem::take(&mut changed);
                let filled = tokio::task::spawn_blocking(move || rings.fill(&registry, heard));
                // A fill that panicked is this task's last: the requests
                // fill from here on.
                if let Err(ended) = filled.await {
                    if ended.is_panic() {
                        error!("A hash ring fill panicked: requests fill the rings from here on");
                    }
                    return;
                }
                match events.recv().await {
                    Ok(event) => changed.note(&event),
                    Err(RecvError::Lagged(_)) => changed.everything = true,
                    Err(RecvError::Closed) => return,
                }
            }
        });
    }
}

/// A ring the warm task filled, kept with the list it filled it over.
#[derive(Debug, Clone)]
struct Kept {
    ring: Arc<HashRing>,
    /// The URLs of the generation the ring was filled for, in that
    /// generation's order ([`ModelWorkerSnapshot::listed_urls`]).
    listed: Arc<[Arc<str>]>,
}

/// A warm task's life, counted in [`KeptRings::warming`] from the task's
/// start to its end: its return, or the end of its runtime, which also
/// drops a task that never ran.
struct Warming(Arc<AtomicUsize>);

impl Warming {
    fn start(warming: &Arc<AtomicUsize>) -> Self {
        warming.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(warming))
    }
}

impl Drop for Warming {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
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
    /// Notes every event that waits, and waits for none.
    fn drain(&mut self, events: &mut broadcast::Receiver<WorkerEvent>) {
        loop {
            match events.try_recv() {
                Ok(event) => self.note(&event),
                Err(TryRecvError::Lagged(_)) => self.everything = true,
                Err(TryRecvError::Empty | TryRecvError::Closed) => break,
            }
        }
    }

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

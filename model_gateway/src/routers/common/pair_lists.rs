//! A prefill-decode pair's two lists as the policy engine decides over
//! them: the gateway's own data, read where it lives.
//!
//! Each pool's list is the snapshot's, with the engine's value of the list
//! kept beside it. Which prefill pairs with which decode is the answer of
//! the snapshot's pair index, numbered once a membership
//! ([`PdPairIndex::classes`]). Each side's admission is read on the worker
//! at the decision: the engine is handed the reads, never their answers.

use std::sync::Arc;

use smg_policy_controller::{
    document::Pairing,
    engine::{PairAdmission, PairLists, PolicyEngine, PoolMembers, Target, WireModes},
};
use smg_worker::{identity::basic::Worker, pd::pairing::PdPairingMode};

use super::{
    pair_index::{PairSides, PairingRefusal, PdPairIndex, PdWire},
    pd_admission,
    prefill_admission::PrefillAdmission,
    snapshot::{ModelWorkerSnapshot, WorkerRegistry, WorkerRegistryExt},
};
use crate::policy_controller::engine::members_of;

/// A pair route's own narrowing of the two lists a request may use, asked
/// at each decision over the snapshot that decision reads and its pair
/// index: the prefills and the decodes the request may take, or `None`
/// where it may take every worker of the two pools.
pub(crate) type PairNarrowing<'a> =
    &'a (dyn Fn(&ModelWorkerSnapshot, &PdPairIndex) -> Option<PairSides> + Sync);

/// The mode the snapshot's pair index is built under for a document's
/// `pairing`: the two name one rule.
pub(crate) const fn pairing_mode(pairing: Pairing) -> PdPairingMode {
    match pairing {
        Pairing::Off => PdPairingMode::Off,
        Pairing::Lenient => PdPairingMode::Lenient,
        Pairing::Strict => PdPairingMode::Strict,
    }
}

/// The prefill and decode workers of one model as one decision of a pair
/// reads them in the registry: the snapshot of the model's workers as it
/// stood then, and the snapshot's pair index for the route's wire under
/// the document's pairing mode.
pub(crate) struct PdPairSnapshot {
    snapshot: Arc<ModelWorkerSnapshot>,
    index: Arc<PdPairIndex>,
    /// The two lists narrowed for one request, paired again, where the
    /// request may not take every worker of the two pools: the workers its
    /// route's rule leaves, and of those the ones dialed on the route's
    /// wires. Nothing is kept of them.
    narrowed: Option<(PdPairIndex, [PoolMembers; 2])>,
}

impl PdPairSnapshot {
    /// Reads the registry once for a pair of `model` whose route dials
    /// `wires`: the HTTP pools for a route that dials HTTP, the gRPC
    /// pipeline's for any other. The pairing mode is the one the document
    /// `engine` decides under states. `narrowing` is the route's own
    /// narrowing of the two pools by the request, asked here of this
    /// snapshot.
    pub(crate) fn read(
        registry: &WorkerRegistry,
        engine: &PolicyEngine,
        model: &str,
        wires: WireModes,
        narrowing: Option<PairNarrowing<'_>>,
    ) -> Self {
        let wire = if wires.intersects(WireModes::HTTP) {
            PdWire::Http
        } else {
            PdWire::Grpc
        };
        let mode = pairing_mode(engine.spec().pairing());
        let snapshot = registry.get_routing_snapshot(model);
        let index = snapshot.pd_pairs(wire, mode);
        let (prefill, decode) = wire.pools();
        let pooled = [prefill, decode].map(|pool| members_of(engine, &snapshot, pool));
        let [prefill, decode] = pooled;
        let served = prefill.wires().union(decode.wires());
        let off_wires = !wires.is_empty() && !wires.contains(served);
        let stated = narrowing.and_then(|narrow| narrow(&snapshot, &index));
        let narrowed = (stated.is_some() || off_wires).then(|| {
            let lists = stated.unwrap_or_else(|| pooled.map(|pool| Arc::clone(pool.workers())));
            let [prefill, decode] = lists.map(|list| -> Arc<[Arc<dyn Worker>]> {
                if !off_wires {
                    return list;
                }
                let on_wires = |worker: &&Arc<dyn Worker>| {
                    WireModes::from(*worker.connection_mode()).intersects(wires)
                };
                list.iter().filter(on_wires).cloned().collect()
            });
            // Each is a part of its pool, and this one decision's.
            let parts = [PoolMembers::part(&prefill), PoolMembers::part(&decode)];
            (PdPairIndex::build(prefill, decode, wire, mode), parts)
        });
        Self {
            snapshot,
            index,
            narrowed,
        }
    }

    /// The two lists for one decision, under the prefill `limit` the router
    /// holds.
    pub(crate) fn lists<'a>(
        &'a self,
        engine: &PolicyEngine,
        limit: Option<&'a PrefillAdmission>,
    ) -> PdPairLists<'a> {
        let (index, [prefill, decode]) = match &self.narrowed {
            Some((index, [prefill, decode])) => (index, [prefill, decode]),
            None => {
                let (prefill, decode) = self.index.wire.pools();
                let pooled = [prefill, decode].map(|pool| members_of(engine, &self.snapshot, pool));
                (&*self.index, pooled)
            }
        };
        PdPairLists {
            index,
            prefill,
            decode,
            limit,
        }
    }
}

/// A pair's two lists, gathered for one decision.
#[derive(Debug)]
pub(crate) struct PdPairLists<'a> {
    index: &'a PdPairIndex,
    prefill: &'a PoolMembers,
    decode: &'a PoolMembers,
    /// The router's prefill limit, where it holds one.
    limit: Option<&'a PrefillAdmission>,
}

impl<'a> PdPairLists<'a> {
    /// What the engine is handed for one decision over the pair.
    pub(crate) fn lists(&self) -> PairLists<'_> {
        PairLists {
            prefill: self.prefill,
            decode: self.decode,
            classes: self.index.classes(),
            admission: self,
        }
    }

    /// The prefill and the decode `target` names, by their places in the
    /// two lists. `None` for a target that names no pair of these lists.
    pub(crate) fn workers(&self, target: &Target) -> Option<[&'a Arc<dyn Worker>; 2]> {
        let decode = target.pair?.decode_position as usize;
        let prefill = self.prefill.workers().get(target.position as usize)?;
        Some([prefill, self.decode.workers().get(decode)?])
    }

    /// Whether each list holds a worker `sendable` passes.
    pub(crate) fn each_side(&self, sendable: impl Fn(&Arc<dyn Worker>) -> bool) -> bool {
        let any = |side: &PoolMembers| side.workers().iter().any(&sendable);
        any(self.prefill) && any(self.decode)
    }

    /// Why no prefill of the lists pairs with a decode, where both lists
    /// hold workers and none does.
    pub(crate) fn refusal(&self) -> Option<&'a PairingRefusal> {
        self.index.refusal.as_ref()
    }
}

impl PairAdmission for PdPairLists<'_> {
    fn has_capacity(&self, prefill: usize) -> bool {
        let worker = self.prefill.workers().get(prefill);
        !(self.limit.zip(worker)).is_some_and(|(limit, worker)| limit.reads_full(worker))
    }

    fn has_room(&self, decode: usize) -> bool {
        let worker = self.decode.workers().get(decode);
        worker.is_none_or(|worker| pd_admission::has_free_room(worker.as_ref()))
    }
}

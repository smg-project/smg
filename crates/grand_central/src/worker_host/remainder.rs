//! What of a row its object does not hold as stated: the holder keeps it
//! beside the object, and the two together state the row.

use std::{sync::Arc, time::Instant};

use smg_worker::identity::basic::Worker;

use super::{found::facts, membership_writer::RowRevision, object::row_status};
use crate::{
    engine::{demands::DemandFacts, view::WireModes},
    store::{
        encoding::EncodeOffer, pairing::PairingDescriptor, HealthPolicy, Overload, Reachability,
        Registrant, Signals, WorkerFacts, WorkerRow,
    },
};

/// What a worker's row states and its object does not hold as stated. The
/// holder keeps one beside each object, and the two together state the row
/// ([`Objects::row_of`](super::Objects::row_of)).
///
/// Of the 24 facts a row states it holds 17 as stated and one bit of an
/// eighteenth, the engine, and nothing the object holds whole: the key,
/// the role, the models, the labels, the cache block size and the
/// bootstrap port are the object's, and so is the status.
///
/// Eight facts stand twice, in a narrower form on the object and stated
/// here; of the engine one bit stands here. The rank and the width share
/// a line:
///
/// | The fact | The object's form | What the object cannot say |
/// |---|---|---|
/// | `engine` | the address it dials, where that is not its key | whether the row named its key as its engine or named none: one bit here |
/// | `dp_rank`, `dp_size` | a width above one, and a rank unless the row is its engine whole | a width of one or of none, and a rank stated beside either |
/// | `wire_modes` | the one wire it speaks | the other wires of the set |
/// | `engine_kind` | the runtime the worker crate reads the name as | a name that crate does not know, and a known name as it was written |
/// | `health` | its checks in whole seconds, off where its host cannot dial the worker | a part of a second, and the policy of a row built with its checks off |
/// | `overload` | the thresholds its host's monitor judges by, the row's own over the host's defaults, its own alone for an external row or while the host's document switches the defaults off, or none | which of them the row stated |
/// | `drain_settle_secs` | the window in force | a window stated equal to the object's default, which reads back as none with no word of the row |
///
/// Ten are not carried to the object: the incarnation, and the facts a
/// decision or a collector reads of the row alone. Three of them have a
/// place of the object's own that no row fills: its priority is unsigned,
/// it derives its pairing facts at its build, and it reads its admission
/// window off a label. The row's id and revision, who made it and when
/// stand here too.
///
/// It stays inside the crate: its readers are the crate's own, and a fact
/// leaves it when the object gets a field that holds the fact as stated.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RowRemainder {
    /// The row's id, and the revision this was taken at.
    pub standing: RowRevision,
    /// Who made the row.
    pub maker: Registrant,
    /// When the row was created on the host's clock.
    pub admitted_at: Instant,
    /// The store's number for this life of the key.
    pub incarnation: u64,
    /// Whether the row names an engine. One that names none dials its key,
    /// as one that names its key does.
    pub names_engine: bool,
    /// The rank as stated.
    pub dp_rank: u16,
    /// The width as stated.
    pub dp_size: u16,
    /// The whole set of wires.
    pub wire_modes: WireModes,
    /// The engine family as the handshake named it.
    pub engine_kind: String,
    /// The policy in its own milliseconds.
    pub health: HealthPolicy,
    /// The thresholds the registration names for itself, and no default.
    pub overload: Overload,
    /// The drain window, where the registration states one.
    pub drain_settle_secs: Option<u32>,
    /// The zone.
    pub zone: Option<String>,
    /// What the worker offers a prefill-decode partner.
    pub pairing: Option<Arc<PairingDescriptor>>,
    /// The worker's end of the encode handoff.
    pub encode: EncodeOffer,
    /// Whether the worker takes images.
    pub vision: bool,
    /// The engine's admission window.
    pub max_running: Option<u32>,
    /// The priority, signed as the row states it.
    pub priority: i32,
    /// From where the worker is reachable.
    pub reachable_from: Reachability,
    /// How the worker's signals reach the store.
    pub signals: Signals,
    /// What the worker states for a request's hard demands.
    pub demand_facts: DemandFacts,
}

impl RowRemainder {
    /// What `row` states and its object does not hold as stated. The row is
    /// taken apart with no rest pattern, so a fact added to the row does
    /// not build until this function says where it stands.
    pub(super) fn of(row: &WorkerRow) -> Self {
        let WorkerRow {
            id,
            facts,
            status: _,
            revision,
            admitted_at,
            maker,
        } = row;
        let WorkerFacts {
            key: _,
            engine,
            incarnation,
            role: _,
            wire_modes,
            engine_kind,
            models: _,
            dp_rank,
            dp_size,
            labels: _,
            zone,
            pairing,
            encode,
            vision,
            max_running,
            kv_block_size: _,
            bootstrap_port: _,
            priority,
            reachable_from,
            signals,
            health,
            overload,
            drain_settle_secs,
            demand_facts,
        } = facts;
        Self {
            standing: RowRevision {
                row: *id,
                revision: *revision,
            },
            maker: maker.clone(),
            admitted_at: *admitted_at,
            incarnation: *incarnation,
            names_engine: !engine.is_empty(),
            dp_rank: *dp_rank,
            dp_size: *dp_size,
            wire_modes: *wire_modes,
            engine_kind: engine_kind.clone(),
            health: *health,
            overload: *overload,
            drain_settle_secs: *drain_settle_secs,
            zone: zone.clone(),
            pairing: pairing.clone(),
            encode: encode.clone(),
            vision: *vision,
            max_running: *max_running,
            priority: *priority,
            reachable_from: *reachable_from,
            signals: *signals,
            demand_facts: *demand_facts,
        }
    }

    /// The row `worker` and this remainder state together: the six facts
    /// the object holds whole as [`facts`] reads them of it now, its
    /// status, the address it dials as the engine of a row that names one,
    /// and every other fact as it was stated.
    ///
    /// It is the row the object was laid from, where nothing was set on
    /// the object since: models set on the object read back as set, under
    /// the revision this was taken at.
    pub(super) fn row(&self, worker: &dyn Worker) -> WorkerRow {
        let Self {
            standing,
            maker,
            admitted_at,
            incarnation,
            names_engine,
            dp_rank,
            dp_size,
            wire_modes,
            engine_kind,
            health,
            overload,
            drain_settle_secs,
            zone,
            pairing,
            encode,
            vision,
            max_running,
            priority,
            reachable_from,
            signals,
            demand_facts,
        } = self;
        let held = facts(worker, None);
        WorkerRow {
            id: standing.row,
            facts: WorkerFacts {
                key: held.key,
                // The object's address is the row's engine, where it named
                // one.
                engine: if *names_engine {
                    held.engine
                } else {
                    String::new()
                },
                incarnation: *incarnation,
                role: held.role,
                wire_modes: *wire_modes,
                engine_kind: engine_kind.clone(),
                models: held.models,
                dp_rank: *dp_rank,
                dp_size: *dp_size,
                labels: held.labels,
                zone: zone.clone(),
                pairing: pairing.clone(),
                encode: encode.clone(),
                vision: *vision,
                max_running: *max_running,
                kv_block_size: held.kv_block_size,
                bootstrap_port: held.bootstrap_port,
                priority: *priority,
                reachable_from: *reachable_from,
                signals: *signals,
                health: *health,
                overload: *overload,
                drain_settle_secs: *drain_settle_secs,
                demand_facts: *demand_facts,
            },
            status: row_status(worker.status()),
            revision: standing.revision,
            admitted_at: *admitted_at,
            maker: maker.clone(),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/remainder/tests.rs"]
mod tests;

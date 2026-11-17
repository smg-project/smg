//! The one-process store: every ingress trait over plain tables behind
//! mutexes. The controller's own store once its drivers move in (soft state
//! its sources republish, no log), and the double the contract tests and the
//! controller's tests run over. Nothing here is on a decision path: a view is
//! built from the world the store publishes ([`MemoryStore::world`]), not
//! through these locks.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    spec::AppliedSpec,
    store::{
        health::HealthSink,
        index::{indexed_under, KvEvent, KvTables, LaneCounts},
        journal::{Journal, Kinds, Position},
        kv_events::{BatchFacts, BatchOutcome, KvCursor, KvEventSink, ResyncReason},
        leases::{Dispatch, LeaseRecord, LeaseRefused, LeaseState, LeasesStore},
        loads::{LoadSample, LoadSink, LoadSource, PollMode},
        reservations::{ReservationKind, ReservationStore},
        sessions::{Pin, SessionsStore},
        workers::{
            identity, MembershipChange, MembershipTail, WorkerFacts, WorkerPatch, WorkerRow,
            WorkerStatus, WorkersStore,
        },
        world::{
            micros, same_offer, Ask, Built, Classes, PairBuffers, StaleGeneration, ViewBuffers,
            World, Writer, PUBLISH_WINDOW,
        },
        GatewayId,
    },
    view::{LoadRow, Role, WorkerId},
};

/// Membership changes kept for watchers; a watcher further behind snapshots
/// again (the registry's broadcast is sized the same today).
const CHANGE_LOG: usize = 1_024;

/// The store. `Default` is empty with term `0`.
pub struct MemoryStore {
    workers: Mutex<Workers>,
    health: Mutex<HealthTables>,
    /// The load rows, one entry per member: an entry is created with its
    /// worker's row and deleted with it, both under the workers lock (taken
    /// first, as the health writers take it). A load write goes into its
    /// worker's entry or is dropped, under this lock alone: the entry is the
    /// membership test, so a read that lands after a removal re-creates
    /// nothing and a replaced incarnation starts empty. The world's write
    /// side sits behind the same lock: a record is on file and folded into
    /// the working rows in one step, and every world is made and shown
    /// under it, so the deciders see the worlds in the order they were
    /// made. It is the last lock a writer takes: the workers lock, then the
    /// health lock, then this one.
    loads: Mutex<Loads>,
    /// The KV lanes and the indexes they write, behind locks of their own
    /// ([`KvTables`]): a membership write takes the table's after the
    /// workers lock, a batch takes it alone.
    kv: KvTables,
    reservations: Mutex<Reservations>,
    sessions: Mutex<FxHashMap<u64, Pin>>,
    leases: Mutex<Leases>,
    /// What a decision reads, through one load: swapped whole at each
    /// publication.
    world: ArcSwap<World>,
    /// Where the rows' clock starts.
    origin: Instant,
    /// Where the term is told; `None` for a store nobody follows. The
    /// membership writes tell through their table's own handle.
    journal: Option<Arc<dyn Journal>>,
    /// The same journal where it carries load rows: where the load writes
    /// are told.
    load_journal: Option<Arc<dyn Journal>>,
}

/// The store's replicated tables at one place of its history
/// ([`MemoryStore::cut`]).
#[derive(Clone, Debug)]
pub struct Cut {
    /// The last journaled write of the kinds asked that the tables hold,
    /// with its term; the default for a store with no journal.
    pub at: Position,
    /// The membership epoch.
    pub epoch: u64,
    /// Every row, by registration key.
    pub rows: Vec<WorkerRow>,
    /// Every load row on file, by worker and rank; none in a cut of the
    /// watched kinds.
    pub loads: Vec<(WorkerId, u16, LoadSample)>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::with_publish_window(PUBLISH_WINDOW)
    }
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // One guard per table, each dropped before the next is taken: the
        // mutex is not reentrant and a chained statement keeps every guard
        // to its end.
        let (workers, epoch) = {
            let workers = self.workers.lock();
            (workers.rows.len(), workers.epoch)
        };
        let leases = self.leases.lock().records.len();
        f.debug_struct("MemoryStore")
            .field("workers", &workers)
            .field("epoch", &epoch)
            .field("leases", &leases)
            .finish_non_exhaustive()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// A store whose folded load records wait at most `window` for their
    /// publication, where [`MemoryStore::new`] takes [`PUBLISH_WINDOW`]:
    /// for a test of the window itself. A window the clock cannot add to
    /// an instant never ends: a waiting record is then shown by the next
    /// status or membership write.
    pub fn with_publish_window(window: Duration) -> Self {
        let now = Instant::now();
        // An hour before the store was made, where the host's clock reaches
        // that far: a sample older than the store keeps an instant of its
        // own.
        let origin = now.checked_sub(Duration::from_secs(3_600)).unwrap_or(now);
        let writer = Writer::new(window, origin);
        let world = ArcSwap::new(writer.world());
        Self {
            workers: Mutex::default(),
            health: Mutex::default(),
            loads: Mutex::new(Loads {
                ranks: FxHashMap::default(),
                writer,
            }),
            kv: KvTables::default(),
            reservations: Mutex::default(),
            sessions: Mutex::default(),
            leases: Mutex::default(),
            world,
            origin,
            journal: None,
            load_journal: None,
        }
    }

    /// A store that tells `journal` its replicated writes (membership and
    /// status, the term, and the load rows where the journal carries
    /// them), each inside the write's own critical section: a store a
    /// standby or a watcher of its rows can follow.
    pub fn journaled(journal: Arc<dyn Journal>) -> Self {
        let mut store = Self::default();
        store.workers.get_mut().journal = Some(Arc::clone(&journal));
        store.load_journal = journal.carries_loads().then(|| Arc::clone(&journal));
        store.journal = Some(journal);
        store
    }

    /// The term the journal followed where the store's last take started
    /// no term in it: the journal had seen that term or a later one, so
    /// the store's writes stand under no term and no standby follows them
    /// until a later take. `None` while the store leads the journal's
    /// term, and for a store with no journal.
    pub fn followed(&self) -> Option<u64> {
        self.journal.as_ref().and_then(|journal| journal.followed())
    }

    /// The tables of `kinds` as one journaled write of those kinds left
    /// them, with that write's position: the rows, and with [`Kinds::All`]
    /// the load rows. What a watcher of the rows starts from, and what a
    /// standby is restored from. Taken under the locks of the kinds asked,
    /// in the store's order (the membership lock; the load lock for the
    /// load rows alone): every journaled write holds its table's lock from
    /// its apply to its telling, so none stands half told here. A writer
    /// of those tables waits for the copy; a decision reads the published
    /// world and does not, and a cut of the rows leaves the load writers
    /// be. Where the journal carries no load row a cut of every kind has
    /// no tail: that journal is lost to a reader of every kind.
    pub fn cut(&self, kinds: Kinds) -> Cut {
        // Copied under the locks, sorted after them: a writer waits for the
        // copy alone.
        let (at, epoch, mut rows, mut samples) = {
            let workers = self.workers.lock();
            let loads = (kinds == Kinds::All).then(|| self.loads.lock());
            let at = self
                .journal
                .as_ref()
                .map_or_else(Position::default, |journal| journal.head(kinds));
            let rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
            let samples: Vec<(WorkerId, u16, LoadSample)> = loads
                .iter()
                .flat_map(|loads| loads.ranks.iter())
                .flat_map(|(id, ranks)| {
                    ranks
                        .samples
                        .iter()
                        .map(|(rank, sample)| (*id, *rank, *sample))
                })
                .collect();
            (at, workers.epoch, rows, samples)
        };
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        samples.sort_by_key(|(id, rank, _)| (*id, *rank));
        Cut {
            at,
            epoch,
            rows,
            loads: samples,
        }
    }

    /// Apply a generation: every world from here on is published under
    /// `spec`, the first at once, with the standing membership. The store's
    /// one entry for a generation; a store starts under
    /// [`AppliedSpec::unbound`].
    ///
    /// The number only rises. A generation whose number is not above the
    /// standing one is refused by naming the standing one, and nothing is
    /// published: a reader that saw a generation never sees a lower one,
    /// and a document that names no generation (number 0) is never applied.
    ///
    /// The generation applied before the standing one is let go here, by
    /// the applying thread and outside the store's lock, and so is a
    /// generation the store refuses.
    ///
    /// The generation is the one writer of the mode the pair classes are
    /// derived under ([`AppliedSpec::pairing`]). A generation of another
    /// mode derives the standing membership's pools again and publishes
    /// them once, under the new generation: no member moved, so the
    /// store's epoch stays and no watcher is told. An apply takes the
    /// workers lock and then the load lock, as a membership write does: a
    /// writer's wait, never a decision's.
    pub fn apply(&self, spec: AppliedSpec) -> Result<(), StaleGeneration> {
        // The writer takes a second reference: when it refuses, the last
        // one is this one, let go after the lock.
        let spec = Arc::new(spec);
        let before = {
            let workers = self.workers.lock();
            let mut loads = self.loads.lock();
            let Loads { ranks, writer } = &mut *loads;
            let ranks = &*ranks;
            let (world, before) = writer.apply(
                Arc::clone(&spec),
                workers.rows.values().map(|row| &row.row),
                |id, rank| ranks.get(&id)?.samples.get(&rank).copied(),
            )?;
            self.world.store(world);
            before
        };
        drop(before);
        Ok(())
    }

    /// The world as the last publication left it, taken through no lock.
    /// Its identity moves with every publication: [`World::epoch`] says
    /// whether the membership moved.
    ///
    /// This door hands an owned pointer, for a reader that keeps a world
    /// across calls. Taking it and dropping it each write the count every
    /// holder of that world shares, so deciding threads that took it would
    /// write one cache line twice a decision. A decision builds through
    /// [`build_view`](Self::build_view), a pair decision through
    /// [`build_pair`](Self::build_pair), and a decision that runs the
    /// generation's engine over its rows through
    /// [`deciding`](Self::deciding): all three borrow the world and write
    /// no shared count.
    ///
    /// A world kept across two publications of a pool keeps the rows it
    /// lists as they are: the second publication writes over other rows,
    /// and allocates that pool's rows when the writer holds none alone
    /// ([`publication_allocations`](Self::publication_allocations)).
    pub fn world(&self) -> Arc<World> {
        self.world.load_full()
    }

    /// Build one decision's view of the pool serving `model` in `role`
    /// ([`World::build`]) from the world of this instant.
    pub fn build_view(
        &self,
        model: &str,
        role: Role,
        ask: &Ask,
        out: &mut ViewBuffers,
    ) -> Option<Built> {
        self.world.load().build(model, role, ask, out)
    }

    /// Build both sides of one pair decision from the world of this instant
    /// under one borrow ([`World::build_pair`]): the model's prefill pool,
    /// its decode pool and the table of their classes, all of one
    /// publication. `None`, with every buffer cleared, when either pool has
    /// no member: one side alone is no pair.
    ///
    /// This is a pair decision's door. It borrows the world, so the two
    /// builds and the table read one publication and no deciding thread
    /// writes the world's shared count.
    pub fn build_pair(
        &self,
        model: &str,
        ask: &Ask,
        out: &mut PairBuffers,
    ) -> Option<(Built, Built)> {
        self.world.load().build_pair(model, ask, out)
    }

    /// One decision's door: `decide` is handed the world of this instant
    /// under one borrow, for the build of its view ([`World::build`]) and
    /// for the engine of the generation those rows were published under
    /// ([`World::spec`]). The rows, the engine and the generation's number
    /// are then of one load, and the borrow writes no shared count, so no
    /// two deciding threads write one line through it. A publication that
    /// replaces the world meanwhile is the next decision's: the world
    /// handed here stays whole until `decide` returns.
    pub fn deciding<R>(&self, decide: impl FnOnce(&World) -> R) -> R {
        decide(&self.world.load())
    }

    /// `at` in the rows' clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
    }

    /// Derive the pools of the membership as this write left it and publish
    /// them in one world. Called under the workers lock by every membership
    /// write; it takes the load lock, as those writes do, so each row
    /// carries the load record on file and no record is lost across the
    /// swap. `classes` names the worker whose write moved a member or a
    /// member's offer: the pair classes are derived for it alone, or carried
    /// where the write moved none.
    fn rebuild_world(&self, workers: &Workers, classes: Classes) {
        let mut loads = self.loads.lock();
        let Loads { ranks, writer } = &mut *loads;
        let ranks = &*ranks;
        let world = writer.rebuild(
            workers.epoch,
            workers.rows.values().map(|row| &row.row),
            |id, rank| ranks.get(&id)?.samples.get(&rank).copied(),
            classes,
        );
        self.world.store(world);
    }

    /// Show the deciders the world a write made, when it made one. Called
    /// with the load lock held: the worlds reach the cell in the order the
    /// writer made them.
    fn show(&self, world: Option<Arc<World>>) {
        if let Some(world) = world {
            self.world.store(world);
        }
    }

    /// Fold a status change into the world and publish it at once. Called
    /// under the workers lock, as the status was written.
    fn show_status(&self, id: WorkerId, status: WorkerStatus) {
        let mut loads = self.loads.lock();
        let world = loads.writer.fold_status(id, status);
        self.show(world);
    }

    /// Publish `gateway`'s veto on `id`, or its clearing, at once. Called
    /// under the health lock, which serializes the vetoes' writers, so the
    /// published vetoes follow the health table's own order.
    fn show_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        let mut loads = self.loads.lock();
        let world = loads.writer.veto(gateway, id, until);
        self.show(world);
    }

    /// How many entries the health tables hold, over every gateway. A
    /// gateway's table that holds none counts as one: a table left behind
    /// shows.
    #[cfg(test)]
    pub(crate) fn health_entries(&self) -> usize {
        let health = self.health.lock();
        let completions = health.ok_at.values().map(|workers| workers.len().max(1));
        health.vetoes.len() + completions.sum::<usize>()
    }

    /// How many writes to the load table moved an overload verdict: a
    /// record, a report, the word that no report came, an eviction. A
    /// membership write and an applied generation judge again and count
    /// nothing.
    pub fn overload_flips(&self) -> u64 {
        self.loads.lock().writer.flips()
    }

    /// How many workers the load table holds an entry for.
    #[cfg(test)]
    pub(super) fn load_entries(&self) -> usize {
        self.loads.lock().ranks.len()
    }

    /// How many rows that left their pool the writer still keeps.
    #[cfg(test)]
    pub(super) fn held_rows(&self) -> usize {
        self.loads.lock().writer.held_rows()
    }

    /// How many whole derivations made what the writer keeps of `model`'s
    /// pair classes; `None` where it keeps nothing of them.
    #[cfg(test)]
    pub(super) fn whole_derivations(&self, model: &str) -> Option<usize> {
        self.loads.lock().writer.whole_derivations(model)
    }

    /// The KV tables, for the index's cases.
    #[cfg(test)]
    pub(super) fn kv(&self) -> &KvTables {
        &self.kv
    }

    /// Sets a row's probe counters: no case reaches a counter's end by
    /// probes.
    #[cfg(test)]
    pub(super) fn probed(&self, id: WorkerId, failures: u32, successes: u32, pending_total: u32) {
        if let Some(row) = self.workers.lock().rows.get_mut(&id) {
            row.probes = Probes {
                failures,
                successes,
                pending_total,
            };
        }
    }

    /// A worker's row is gone (removed, or replaced by a new incarnation):
    /// the store's tables that live exactly as long as the row leave with
    /// it, here and nowhere else: the health tables, the load rows and the
    /// worker's KV lane with its blocks in the index. The vetoes a world
    /// carries are not among them: they leave with the row in the
    /// publication of the membership write that calls this. Called under
    /// the workers lock:
    /// the health writers take it first and a load write goes into its
    /// worker's entry or nowhere, so a write racing the removal finds the
    /// row gone or lands before the tables are cleared.
    fn forget_worker(&self, id: WorkerId) {
        self.kv.forget(id);
        let mut health = self.health.lock();
        health.forget(id);
        let mut loads = self.loads.lock();
        loads.ranks.remove(&id);
    }

    /// A gateway is gone past its grace period: what the health tables
    /// hold under its key leaves with it, here and nowhere else. Its
    /// completions today; its vetoes join with the arm that writes them.
    /// Called from [`LeasesStore::drop_gateway`]: one door tells the store
    /// a gateway is gone.
    fn forget_gateway(&self, gateway: GatewayId) {
        self.health.lock().ok_at.remove(&gateway);
    }

    /// How many times a publication allocated a pool's rows since the store
    /// was made. A publication writes a pool's rows over rows no world
    /// lists any more: the ones the pool published before its last, else
    /// any of its length that left a pool. It allocates when it finds
    /// none: at a pool's first publication after a membership write,
    /// unless rows that write retired fit it, and past that only while a
    /// reader holds a world from before the pool's last publication. A
    /// membership write's own rows are not counted.
    pub fn publication_allocations(&self) -> u64 {
        self.loads.lock().writer.allocations()
    }

    /// Give the row of `id` its KV lane as its facts ask
    /// ([`KvTables::member`]). Called under the workers lock by every write
    /// that makes a row or changes its facts.
    fn index_member(&self, workers: &Workers, id: WorkerId) {
        if let Some(row) = workers.rows.get(&id) {
            self.kv.member(&row.row);
        }
    }

    /// How many resyncs were recorded for `reason`.
    pub fn resyncs(&self, reason: ResyncReason) -> u64 {
        self.kv.resyncs(reason)
    }

    /// Blocks the index holds for the row of `id` at `rank`; `0` for a row
    /// the store does not index.
    pub fn indexed_blocks(&self, id: WorkerId, rank: u16) -> usize {
        self.kv.indexed_blocks(id, rank)
    }

    /// What the lane of `id` at `rank` counted beside its blocks; `None`
    /// for a row the store does not index.
    pub fn lane_counts(&self, id: WorkerId, rank: u16) -> Option<LaneCounts> {
        self.kv.counts(id, rank)
    }

    /// Rows that ask for events and hold no lane. Each was written while
    /// its model's index stood at its ceiling
    /// ([`INDEX_WORKERS`](crate::store::INDEX_WORKERS)): it holds nothing
    /// in the index until one of its own later writes finds a place.
    pub fn unindexed_rows(&self) -> usize {
        let workers = self.workers.lock();
        let asking = workers
            .rows
            .values()
            .filter(|row| indexed_under(&row.row).is_some())
            .count();
        asking.saturating_sub(self.kv.lanes())
    }
}

// ---------------------------------------------------------------- workers

#[derive(Default)]
struct Workers {
    rows: FxHashMap<WorkerId, Row>,
    by_key: HashMap<String, WorkerId>,
    epoch: u64,
    log: VecDeque<(u64, MembershipChange)>,
    /// The store's journal: every membership and status write passes
    /// [`Workers::record`], which tells it.
    journal: Option<Arc<dyn Journal>>,
}

struct Row {
    row: WorkerRow,
    probes: Probes,
}

/// The prober's counters for one row.
#[derive(Default)]
struct Probes {
    failures: u32,
    successes: u32,
    pending_total: u32,
}

impl Workers {
    fn record(&mut self, change: MembershipChange) -> MembershipChange {
        self.record_over(change, None)
    }

    /// [`Workers::record`] for a write that took a row out of the table (a
    /// removal, a replacement): `left` is that row, for the journal.
    fn record_over(
        &mut self,
        change: MembershipChange,
        left: Option<&WorkerRow>,
    ) -> MembershipChange {
        self.epoch += 1;
        if self.log.len() == CHANGE_LOG {
            self.log.pop_front();
        }
        self.log.push_back((self.epoch, change));
        if let Some(journal) = &self.journal {
            let row = self.rows.get(&change.id()).map(|row| &row.row);
            journal.member(self.epoch, change, row, left);
        }
        change
    }

    /// The row under `id` when `revision` is still its revision.
    fn guarded(&mut self, id: WorkerId, revision: u64) -> Option<&mut Row> {
        self.rows
            .get_mut(&id)
            .filter(|row| row.row.revision == revision)
    }

    fn set_status(&mut self, id: WorkerId, status: WorkerStatus) -> Option<MembershipChange> {
        let row = self.rows.get_mut(&id)?;
        let from = row.row.status;
        if from == status {
            return None;
        }
        // The revision counts the row's stated facts: a status change
        // leaves it, so the holder of a row's revision stays current while
        // the worker's health moves.
        row.row.status = status;
        Some(self.record(MembershipChange::StatusChanged {
            id,
            from,
            to: status,
        }))
    }
}

impl WorkersStore for MemoryStore {
    fn upsert(&self, mut facts: WorkerFacts) -> MembershipChange {
        // A blank value of a descriptor is no report: the row of record holds
        // what the pairing rule reads.
        if let Some(descriptor) = facts.pairing.as_deref_mut() {
            descriptor.normalize();
        }
        let mut workers = self.workers.lock();
        let now = Instant::now();
        if let Some(old_id) = workers.by_key.get(&facts.key).copied() {
            let same_incarnation = workers
                .rows
                .get(&old_id)
                .is_some_and(|row| row.row.facts.incarnation == facts.incarnation);
            if same_incarnation {
                let Some(row) = workers.rows.get_mut(&old_id) else {
                    return MembershipChange::Removed { id: old_id };
                };
                let resized = row.row.facts.dp_size != facts.dp_size;
                // A registration refreshed with the offer it had moves no
                // pair class.
                let classes = if same_offer(&row.row.facts, &facts) {
                    Classes::Carry
                } else {
                    Classes::Written {
                        left: Some(old_id),
                        joined: Some(old_id),
                    }
                };
                row.row.facts = facts;
                row.row.revision += 1;
                let change = workers.record(if resized {
                    MembershipChange::Resized { id: old_id }
                } else {
                    MembershipChange::Relabelled { id: old_id }
                });
                self.index_member(&workers, old_id);
                self.rebuild_world(&workers, classes);
                return change;
            }
            // A new incarnation under the key: the old id leaves with its
            // counters; its status carries over when the registration
            // matches (role, models and wires), as the registry inherits a
            // replaced worker's runtime state today. A drain is of the
            // registration it named and ends with it: a draining
            // incarnation hands no status on.
            let old = workers.rows.remove(&old_id);
            let new_id = identity(&facts.key, facts.incarnation);
            let status = old
                .as_ref()
                .filter(|old| {
                    old.row.status != WorkerStatus::Draining
                        && old.row.facts.role == facts.role
                        && old.row.facts.same_models(&facts)
                        && old.row.facts.wire_modes == facts.wire_modes
                })
                .map_or(WorkerStatus::Pending, |old| old.row.status);
            workers.by_key.insert(facts.key.clone(), new_id);
            workers.rows.insert(
                new_id,
                Row {
                    row: WorkerRow {
                        id: new_id,
                        facts,
                        status,
                        revision: 1,
                        admitted_at: now,
                    },
                    probes: Probes::default(),
                },
            );
            // The old incarnation's tables leave with its row, as on a
            // removal, under the same lock and before the replacement is
            // told: a veto, a completed dispatch or a load row keyed by the
            // old id would outlive it for good, and no load of the old
            // incarnation stands after the replacement in the journal.
            self.forget_worker(old_id);
            let change = workers.record_over(
                MembershipChange::Replaced {
                    old: old_id,
                    new: new_id,
                },
                old.as_ref().map(|old| &old.row),
            );
            // The new incarnation starts with an empty load entry, made
            // after the telling: nothing the old one reported is the new
            // one's, and no load of it stands before the replacement.
            self.loads.lock().ranks.insert(new_id, Ranks::default());
            self.index_member(&workers, new_id);
            let classes = Classes::Written {
                left: Some(old_id),
                joined: Some(new_id),
            };
            self.rebuild_world(&workers, classes);
            return change;
        }
        let id = identity(&facts.key, facts.incarnation);
        workers.by_key.insert(facts.key.clone(), id);
        workers.rows.insert(
            id,
            Row {
                row: WorkerRow {
                    id,
                    facts,
                    status: WorkerStatus::Pending,
                    revision: 1,
                    admitted_at: now,
                },
                probes: Probes::default(),
            },
        );
        // Recorded before the row's load entry exists: no load of a worker
        // stands before its registration in the journal.
        let change = workers.record(MembershipChange::Registered { id });
        // The row's load entry, created with it: what a member's load
        // writes go into.
        self.loads.lock().ranks.insert(id, Ranks::default());
        self.index_member(&workers, id);
        let classes = Classes::Written {
            left: None,
            joined: Some(id),
        };
        self.rebuild_world(&workers, classes);
        change
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        // Every field is named: one more stops the build here until the
        // carry below is asked of it.
        let WorkerPatch {
            labels,
            priority,
            overload,
        } = patch;
        if let Some(labels) = labels {
            row.row.facts.labels = labels;
        }
        if let Some(priority) = priority {
            row.row.facts.priority = priority;
        }
        if let Some(overload) = overload {
            row.row.facts.overload = overload;
        }
        row.row.revision += 1;
        let change = workers.record(MembershipChange::Relabelled { id });
        // Labels, a priority and an overload are no part of an offer.
        self.rebuild_world(&workers, Classes::Carry);
        Some(change)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        workers.guarded(id, revision)?;
        let row = workers.rows.remove(&id)?;
        workers.by_key.remove(&row.row.facts.key);
        // The worker's tables leave with its row, under the workers lock
        // and before the removal is told: a veto, a completed dispatch or a
        // load row keyed by an id that is gone would stay for good, the next
        // incarnation having a new id, and no load of the worker stands
        // after its removal in the journal. A poll or a record that lands
        // later finds no load entry and writes nothing.
        self.forget_worker(id);
        let change = workers.record_over(MembershipChange::Removed { id }, Some(&row.row));
        let classes = Classes::Written {
            left: Some(id),
            joined: None,
        };
        self.rebuild_world(&workers, classes);
        Some(change)
    }

    fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        workers.guarded(id, revision)?;
        let change = workers.set_status(id, status)?;
        self.show_status(id, status);
        Some(change)
    }

    fn get(&self, id: WorkerId) -> Option<WorkerRow> {
        self.workers.lock().rows.get(&id).map(|row| row.row.clone())
    }

    fn by_key(&self, key: &str) -> Option<WorkerRow> {
        let workers = self.workers.lock();
        let id = workers.by_key.get(key)?;
        workers.rows.get(id).map(|row| row.row.clone())
    }

    fn epoch(&self) -> u64 {
        self.workers.lock().epoch
    }

    fn snapshot(&self) -> (u64, Vec<WorkerRow>) {
        let workers = self.workers.lock();
        let mut rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (workers.epoch, rows)
    }

    fn changes_since(&self, epoch: u64) -> MembershipTail {
        let workers = self.workers.lock();
        if epoch > workers.epoch {
            return MembershipTail::SnapshotRequired;
        }
        if epoch == workers.epoch {
            return MembershipTail::Changes(Vec::new());
        }
        match workers.log.front() {
            // The change after `epoch` is still in the log.
            Some(&(first, _)) if first <= epoch + 1 => MembershipTail::Changes(
                workers
                    .log
                    .iter()
                    .filter(|(at, _)| *at > epoch)
                    .copied()
                    .collect(),
            ),
            _ => MembershipTail::SnapshotRequired,
        }
    }
}

// ----------------------------------------------------------------- health

#[derive(Default)]
struct HealthTables {
    /// A gateway's veto on a worker, until the instant.
    vetoes: FxHashMap<(GatewayId, WorkerId), Instant>,
    /// The latest completed dispatch of class `Ok` on a worker by each
    /// gateway: the precedence rule is per gateway. One table a gateway:
    /// a gateway's end lets its own go whole and walks no other's.
    ok_at: FxHashMap<GatewayId, FxHashMap<WorkerId, Instant>>,
}

impl HealthTables {
    /// Drop every gateway's entries on the worker; a gateway's table that
    /// holds none after it goes too.
    fn forget(&mut self, id: WorkerId) {
        self.vetoes.retain(|(_, worker), _| *worker != id);
        self.ok_at.retain(|_, workers| {
            workers.remove(&id);
            !workers.is_empty()
        });
    }
}

/// The status machine of one probe over the row's counters: the registry's
/// `compute_next_status`, with the counters here instead of on the handle.
fn next_status(row: &mut Row, ok: bool) -> Option<WorkerStatus> {
    let policy = row.row.facts.health;
    let status = row.row.status;
    let failure = policy.failure_threshold.max(1);
    let success = policy.success_threshold.max(1);
    // A threshold is any number its writer took: the products and the
    // counters stop at their range's end, so a threshold whose product
    // passes the range is never met before the counter's end.
    let liveness = failure.saturating_mul(3);
    let pending_cap = failure.saturating_mul(10);
    let probes = &mut row.probes;
    if ok {
        probes.failures = 0;
        probes.successes = probes.successes.saturating_add(1);
        probes.pending_total = probes.pending_total.saturating_add(1);
        if matches!(
            status,
            WorkerStatus::Pending | WorkerStatus::NotReady | WorkerStatus::Failed
        ) && probes.successes >= success
        {
            probes.successes = 0;
            probes.pending_total = 0;
            return Some(WorkerStatus::Ready);
        }
        if status == WorkerStatus::Pending && probes.pending_total >= pending_cap {
            probes.successes = 0;
            return Some(WorkerStatus::Failed);
        }
        return None;
    }
    probes.successes = 0;
    probes.failures = probes.failures.saturating_add(1);
    probes.pending_total = probes.pending_total.saturating_add(1);
    match status {
        WorkerStatus::Ready if probes.failures >= failure => {
            probes.failures = 0;
            Some(WorkerStatus::NotReady)
        }
        WorkerStatus::NotReady if probes.failures >= liveness => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        WorkerStatus::Pending if probes.pending_total >= pending_cap => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        _ => None,
    }
}

impl HealthSink for MemoryStore {
    fn probe_outcome(
        &self,
        id: WorkerId,
        ok: bool,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        let from = row.row.status;
        let to = next_status(row, ok)?;
        workers.set_status(id, to)?;
        self.show_status(id, to);
        Some((from, to))
    }

    fn connected(&self, id: WorkerId, revision: u64) -> Option<(WorkerStatus, WorkerStatus)> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        let from = row.row.status;
        if matches!(from, WorkerStatus::Draining | WorkerStatus::Ready) {
            return None;
        }
        row.probes = Probes::default();
        workers.set_status(id, WorkerStatus::Ready)?;
        self.show_status(id, WorkerStatus::Ready);
        Some((from, WorkerStatus::Ready))
    }

    fn gateway_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        match until {
            Some(until) => {
                // Under the workers lock, taken before the health lock as
                // every writer takes it: a veto landing after the worker's
                // removal (a breaker opening on a torn-down worker's
                // failures) would re-create its entry for good.
                let workers = self.workers.lock();
                if !workers.rows.contains_key(&id) {
                    return;
                }
                let mut health = self.health.lock();
                health.vetoes.insert((gateway, id), until);
                self.show_veto(gateway, id, Some(until));
            }
            None => {
                let mut health = self.health.lock();
                health.vetoes.remove(&(gateway, id));
                self.show_veto(gateway, id, None);
            }
        }
    }

    fn vetoed(&self, gateway: GatewayId, id: WorkerId, now: Instant) -> bool {
        self.health
            .lock()
            .vetoes
            .get(&(gateway, id))
            .is_some_and(|until| *until > now)
    }

    fn completed(&self, gateway: GatewayId, id: WorkerId, class: TransportClass, at: Instant) {
        if class != TransportClass::Ok {
            return;
        }
        // Under the workers lock, taken before the health lock: a dispatch
        // completing after its worker's removal (the usual order, a gateway's
        // in-flight work ends after the row is gone) would re-create the
        // entry for good.
        let workers = self.workers.lock();
        if !workers.rows.contains_key(&id) {
            return;
        }
        let mut health = self.health.lock();
        let latest = health
            .ok_at
            .entry(gateway)
            .or_default()
            .entry(id)
            .or_insert(at);
        if at > *latest {
            *latest = at;
        }
    }

    fn completed_ok_since(&self, gateway: GatewayId, id: WorkerId, since: Instant) -> bool {
        self.health
            .lock()
            .ok_at
            .get(&gateway)
            .and_then(|workers| workers.get(&id))
            .is_some_and(|at| *at >= since)
    }
}

// ------------------------------------------------------------------ loads

impl LoadSource {
    /// A source whose records arrive on their own (a stream or its relay):
    /// what the poll decision counts as "the stream working".
    const fn pushes(self) -> bool {
        matches!(self, Self::Stream | Self::Relay(_))
    }
}

/// One worker's ranks: the latest sample per rank and, apart from it, the
/// instant each rank's own record last arrived (a poll never moves it).
#[derive(Default)]
struct Ranks {
    samples: BTreeMap<u16, LoadSample>,
    pushed_at: BTreeMap<u16, Instant>,
}

impl Ranks {
    /// Whether the rank's own record arrived at or after `since`.
    fn pushed_since(&self, rank: u16, since: Instant) -> bool {
        self.pushed_at.get(&rank).is_some_and(|&at| at >= since)
    }
}

/// The load table and the world's write side, behind one lock.
struct Loads {
    ranks: FxHashMap<WorkerId, Ranks>,
    writer: Writer,
}

impl LoadSink for MemoryStore {
    fn record(
        &self,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        let age = Duration::from_millis(u64::from(row.report_age_ms));
        let sample = LoadSample {
            row,
            received_at,
            sampled_at: received_at.checked_sub(age).unwrap_or(received_at),
            source,
        };
        let mut loads = self.loads.lock();
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        // No entry, no member: the worker left (or never registered) and
        // the record is dropped, never an entry of its own.
        let Some(ranks) = table.get_mut(&id) else {
            return;
        };
        if source.pushes() {
            ranks.pushed_at.insert(rank, received_at);
        }
        ranks.samples.insert(rank, sample);
        if let Some(journal) = &self.load_journal {
            journal.load(id, rank, &sample);
        }
        if let Some(place) = writer.fold_load(id, rank, Some(&sample)) {
            writer.recorded(place, received_at);
            let world = writer.folded_at(received_at);
            self.show(world);
        }
    }

    fn report(
        &self,
        answer: &[(WorkerId, u16, Option<LoadRow>)],
        polled_at: Instant,
        source: LoadSource,
    ) {
        let mut loads = self.loads.lock();
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        // The answer is one engine's: its rows are written whole before
        // the one judgment and the one publication, so its ranks are read
        // as one answer left them. A caller's mix of engines judges each,
        // and each engine is heard by its own ranks' rows.
        let mut first: Option<(u32, bool)> = None;
        let mut others: Vec<(u32, bool)> = Vec::new();
        for &(id, rank, row) in answer {
            // A report that lands after its worker left (the poll was out
            // when the row went) finds no entry and is dropped.
            let Some(ranks) = table.get_mut(&id) else {
                continue;
            };
            // A rank that pushed since the poll began keeps its record,
            // and its stream is heard: no word of the poll stands over it.
            if ranks.pushed_since(rank, polled_at) {
                continue;
            }
            let (engine, heard) = match row {
                Some(row) => {
                    let age = Duration::from_millis(u64::from(row.report_age_ms));
                    let sample = LoadSample {
                        row,
                        received_at: polled_at,
                        sampled_at: polled_at.checked_sub(age).unwrap_or(polled_at),
                        source,
                    };
                    ranks.samples.insert(rank, sample);
                    if let Some(journal) = &self.load_journal {
                        journal.load(id, rank, &sample);
                    }
                    let place = writer.fold_load(id, rank, Some(&sample));
                    let engine = place.and_then(|place| {
                        let engine = writer.engine_at(place)?;
                        writer.hear(place, polled_at);
                        Some(engine)
                    });
                    (engine, true)
                }
                // A rank the answer omits keeps what is on file.
                None => (writer.engine_of(id), false),
            };
            let Some(engine) = engine else {
                continue;
            };
            match &mut first {
                None => first = Some((engine, heard)),
                Some(seen) if seen.0 == engine => seen.1 |= heard,
                Some(_) => match others.iter_mut().find(|seen| seen.0 == engine) {
                    Some(seen) => seen.1 |= heard,
                    None => others.push((engine, heard)),
                },
            }
        }
        let Some(first) = first else {
            return;
        };
        let mut changed = false;
        for (engine, heard) in std::iter::once(first).chain(others) {
            changed |= if heard {
                // A row was written: the engine is judged once, and the
                // write is shown whether or not a verdict moved.
                writer.judged(engine, true);
                true
            } else {
                // No row came for the engine: the word that none came,
                // unless a rank of it pushed since the poll began, whether
                // the answer names that rank or not. The word is shown
                // where it moves a verdict.
                let pushed = writer.ranks_of(engine).any(|(id, rank)| {
                    table
                        .get(&id)
                        .is_some_and(|ranks| ranks.pushed_since(rank, polled_at))
                });
                !pushed && writer.silence(engine)
            };
        }
        if changed {
            let world = writer.folded_at(polled_at);
            self.show(world);
        }
    }

    fn heard_every(&self, period: Duration) -> bool {
        let mut loads = self.loads.lock();
        let (stands, moved) = loads.writer.heard_every(period);
        if moved {
            // Every engine was judged again under the period: shown as
            // a load write with no instant of its own is.
            let world = loads.writer.folded();
            self.show(world);
        }
        stands
    }

    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode {
        let loads = self.loads.lock();
        let Some(ranks) = loads.ranks.get(&id) else {
            return PollMode::Poll;
        };
        // The receipts, not the samples: a poll's row replaces a silent
        // rank's sample but never its receipt, so one pushing rank cannot
        // hide another that fell silent.
        let mut pushing = ranks.pushed_at.values();
        let Some(first) = pushing.next() else {
            return PollMode::Poll;
        };
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < interval;
        if fresh(first) && pushing.all(fresh) {
            PollMode::Suppressed
        } else {
            PollMode::Fallback
        }
    }

    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
        self.loads
            .lock()
            .ranks
            .get(&id)?
            .samples
            .get(&rank)
            .copied()
    }

    fn ranks(&self, id: WorkerId) -> usize {
        self.loads
            .lock()
            .ranks
            .get(&id)
            .map_or(0, |ranks| ranks.samples.len())
    }

    /// A member's rows and receipts go and its entry stays, empty: the entry
    /// is what its next report goes into, and it leaves with the worker's
    /// row alone (`remove`, a replacing `upsert`). A worker that left has no
    /// entry and nothing on file already.
    fn evict(&self, id: WorkerId) {
        let mut loads = self.loads.lock();
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        if let Some(ranks) = table.get_mut(&id) {
            let mut folded = None;
            for &rank in ranks.samples.keys() {
                folded = writer.fold_load(id, rank, None).or(folded);
            }
            *ranks = Ranks::default();
            if let Some(place) = folded {
                if let Some(engine) = writer.engine_at(place) {
                    writer.judged(engine, false);
                }
                let world = writer.folded();
                self.show(world);
            }
        }
    }

    fn publish_due(&self) -> Option<Instant> {
        self.loads.lock().writer.due()
    }

    fn publish(&self, now: Instant) {
        let mut loads = self.loads.lock();
        let world = loads.writer.publish_if_due(now);
        self.show(world);
    }
}

// -------------------------------------------------------------- KV events

impl KvEventSink for MemoryStore {
    /// The batch's events as its collector decoded them: the store applies
    /// them to the index of the worker's model
    /// ([`index`](crate::store::index)).
    type Batch = [KvEvent];

    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &Self::Batch,
    ) -> BatchOutcome {
        self.kv.admit(id, rank, facts, batch)
    }

    fn reconnected(&self, id: WorkerId) {
        self.kv.reconnected(id);
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        self.kv.resync(id, rank, reason);
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        self.kv.cursor(id, rank)
    }

    fn removed(&self, id: WorkerId) {
        self.kv.removed(id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.kv.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.kv.set_block_size(model, size);
    }
}

// ----------------------------------------------------------- reservations

/// A gateway's lease: the key a claim is held under.
type LeaseKey = (GatewayId, LeaseId);

#[derive(Default)]
struct Reservations {
    held: FxHashMap<(WorkerId, ReservationKind), u32>,
    by_lease: FxHashMap<LeaseKey, Vec<(WorkerId, ReservationKind, u32)>>,
}

impl ReservationStore for MemoryStore {
    fn window(&self, id: WorkerId, _kind: ReservationKind) -> Option<u32> {
        // Both kinds read the engine's admission window today; a configured
        // per-side cap is the host's until the window moves into the store.
        self.workers.lock().rows.get(&id)?.row.facts.max_running
    }

    fn try_reserve(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        kind: ReservationKind,
        count: u32,
        lease: LeaseId,
    ) -> bool {
        // A worker the store does not know (never registered, or removed) is
        // refused before its window is read: a missing row is no live worker
        // without a bound, and a commit racing a removal gets nothing on it.
        let window = {
            let workers = self.workers.lock();
            let Some(row) = workers.rows.get(&id) else {
                return false;
            };
            // Both kinds read the engine's admission window today, as
            // `window` does.
            row.row.facts.max_running
        };
        let mut reservations = self.reservations.lock();
        let holds_kind = reservations
            .by_lease
            .get(&(gateway, lease))
            .is_some_and(|held| held.iter().any(|(_, k, _)| *k == kind));
        if holds_kind {
            return false;
        }
        let held = reservations.held.entry((id, kind)).or_insert(0);
        let total = held.saturating_add(count);
        if window.is_some_and(|window| total > window) {
            return false;
        }
        *held = total;
        reservations
            .by_lease
            .entry((gateway, lease))
            .or_default()
            .push((id, kind, count));
        true
    }

    fn release(&self, gateway: GatewayId, lease: LeaseId) {
        let mut reservations = self.reservations.lock();
        let Some(held) = reservations.by_lease.remove(&(gateway, lease)) else {
            return;
        };
        for (id, kind, count) in held {
            if let Some(total) = reservations.held.get_mut(&(id, kind)) {
                *total = total.saturating_sub(count);
            }
        }
    }

    fn held(&self, id: WorkerId, kind: ReservationKind) -> u32 {
        self.reservations
            .lock()
            .held
            .get(&(id, kind))
            .copied()
            .unwrap_or(0)
    }
}

// --------------------------------------------------------------- sessions

impl SessionsStore for MemoryStore {
    fn pin(&self, key: u64, worker: WorkerId, at: Instant) {
        if key == 0 {
            return;
        }
        self.sessions.lock().insert(
            key,
            Pin {
                worker,
                last_used: at,
            },
        );
    }

    fn lookup(&self, key: u64) -> Option<Pin> {
        self.sessions.lock().get(&key).copied()
    }

    fn unpin(&self, key: u64) {
        self.sessions.lock().remove(&key);
    }

    fn drop_worker(&self, worker: WorkerId) -> usize {
        let mut sessions = self.sessions.lock();
        let before = sessions.len();
        sessions.retain(|_, pin| pin.worker != worker);
        before - sessions.len()
    }

    fn expire(&self, before: Instant) -> usize {
        let mut sessions = self.sessions.lock();
        let count = sessions.len();
        sessions.retain(|_, pin| pin.last_used >= before);
        count - sessions.len()
    }

    fn merge(&self, key: u64, pin: Pin) {
        if key == 0 {
            return;
        }
        let mut sessions = self.sessions.lock();
        let entry = sessions.entry(key).or_insert(pin);
        if pin.last_used > entry.last_used {
            *entry = pin;
        }
    }

    fn len(&self) -> usize {
        self.sessions.lock().len()
    }
}

// ----------------------------------------------------------------- leases

#[derive(Default)]
struct Leases {
    term: u64,
    records: FxHashMap<(GatewayId, LeaseId), LeaseRecord>,
}

impl LeasesStore for MemoryStore {
    fn term(&self) -> u64 {
        self.leases.lock().term
    }

    fn set_term(&self, term: u64) {
        let mut leases = self.leases.lock();
        if term > leases.term {
            leases.term = term;
            if let Some(journal) = &self.journal {
                journal.term_rose(term);
            }
        }
    }

    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused> {
        let mut leases = self.leases.lock();
        if record.term > leases.term {
            return Err(LeaseRefused::TermAhead {
                lease: record.term,
                store: leases.term,
            });
        }
        let key = (record.gateway, record.lease.id);
        if leases.records.contains_key(&key) {
            return Err(LeaseRefused::Duplicate);
        }
        leases.records.insert(key, record);
        Ok(())
    }

    fn dispatched(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        _dispatch: Dispatch,
        at: Instant,
    ) -> bool {
        let mut leases = self.leases.lock();
        let Some(record) = leases.records.get_mut(&(gateway, lease)) else {
            return false;
        };
        if record.state == LeaseState::Issued {
            record.state = LeaseState::Dispatched { at };
        }
        true
    }

    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool {
        let mut leases = self.leases.lock();
        let Some(record) = leases.records.get(&(gateway, lease)) else {
            return false;
        };
        // A pair's room lives until its decode side ends: the prefill side's
        // completion leaves the record standing.
        if record.decode.is_some() && outcome.side == Side::Prefill {
            return true;
        }
        leases.records.remove(&(gateway, lease));
        true
    }

    fn released(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        _side: Option<Side>,
        _reason: ReleaseReason,
    ) -> bool {
        // No side is read yet: a release of one side of a pair ends the
        // whole lease. The step that issues a pair's lease reads it.
        self.leases
            .lock()
            .records
            .remove(&(gateway, lease))
            .is_some()
    }

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord> {
        self.leases.lock().records.get(&(gateway, lease)).copied()
    }

    fn expire(&self, now: Instant) -> Vec<LeaseRecord> {
        let mut leases = self.leases.lock();
        let gone: Vec<(GatewayId, LeaseId)> = leases
            .records
            .iter()
            .filter(|(_, record)| {
                record.state == LeaseState::Issued && record.dispatch_deadline <= now
            })
            .map(|(key, _)| *key)
            .collect();
        gone.into_iter()
            .filter_map(|key| leases.records.remove(&key))
            .map(|mut record| {
                record.state = LeaseState::Released(ReleaseReason::Abandoned);
                record
            })
            .collect()
    }

    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize {
        let mut leases = self.leases.lock();
        let term = leases.term;
        leases.records.retain(|(owner, _), _| *owner != gateway);
        for record in records {
            if record.gateway != gateway || record.term > term {
                continue;
            }
            leases.records.insert((gateway, record.lease.id), *record);
        }
        leases
            .records
            .keys()
            .filter(|(owner, _)| *owner == gateway)
            .count()
    }

    fn drop_gateway(&self, gateway: GatewayId) -> usize {
        self.forget_gateway(gateway);
        let mut leases = self.leases.lock();
        let before = leases.records.len();
        leases.records.retain(|(owner, _), _| *owner != gateway);
        before - leases.records.len()
    }

    fn live(&self, gateway: GatewayId) -> usize {
        self.leases
            .lock()
            .records
            .iter()
            .filter(|((owner, _), record)| {
                *owner == gateway
                    && matches!(
                        record.state,
                        LeaseState::Issued | LeaseState::Dispatched { .. }
                    )
            })
            .count()
    }
}

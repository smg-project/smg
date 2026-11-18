//! The membership door (design 5.1 and 5.2): the control scope's three
//! registration calls over the store's membership writes. A call writes the
//! row its caller stated or refuses it whole by the field's name, and
//! answers the event that write caused: it drops nothing, discovers nothing
//! and gives no verdict it did not witness.
//!
//! A registered row is pending. One whose checks are off gets the store's
//! connect signal at once, as a worker with its health checks off starts
//! ready; the registrant says "do not check", never "healthy", and says it
//! of its own registration alone: one that states no policy is checked,
//! whatever stood under its URL.
//!
//! An answer carries the log's place after the call's write, with the term
//! that place was written under: the write's own place, since the door is
//! the only writer of the kinds a watcher follows but the start of a term.
//! A caller that watches the rows holds the call's write once its stream
//! has passed that place. A call that wrote nothing answers the place the
//! log stands at.
//!
//! **The door is the process's only membership writer.** A call's read, its
//! write and its read back run under the door's one lock, so a mode's test
//! and the answered row are exact. A second writer (discovery, a standby's
//! apply on a replica that also serves) breaks a mode's test: it and the
//! door then go through a compare-and-write of the store.
//!
//! **A call writes under the term it reads once it holds the door's turn.**
//! The fence is read under the lock, first: a call that waited its turn
//! behind another writes under the term of its own minute, or is refused as
//! a follower's is, and never under a term read before the wait.
//!
//! **Every life of a URL has a number of its own:** the leader's term above the
//! store's epoch when the door numbered it, counted from a base the door
//! draws once a run. The store's id is a function of the URL and that
//! number, so the guard of a patch and of a removal (the id and the revision
//! their caller read) names one registration and never a later one of the
//! URL, at this leader or at another. At a later run under the same term
//! that holds by the draw, not by construction: two runs' counts meet with
//! odds near a run's writes over 2^38.

use std::{collections::hash_map::RandomState, hash::BuildHasher, sync::Arc};

use parking_lot::Mutex;
use tonic::Status;
use tracing::info;

use super::{
    principal::Principal,
    worker_wire::{self, Refused},
};
use crate::{
    proto::{self, register_worker_request::Mode, worker_event::Event},
    replication::Log,
    store::{HealthSink, Kinds, MemoryStore, Position, WorkerRow, WorkerStatus, WorkersStore},
    view::WorkerId,
};

/// A registration's refusal where the store's log does not lead the term the
/// call was admitted under.
const LEADS_NO_TERM: &str =
    "this replica's store does not lead the term the call was admitted under";

/// The door over one store.
pub(crate) struct Door {
    store: Arc<MemoryStore>,
    /// Where this run's count of lives starts: below 2^39, drawn once.
    base: u64,
    /// The log the store journals to: where an answer reads its place.
    log: Arc<Log>,
    /// One call at a time, from its read to its read back.
    turn: Mutex<()>,
}

fn invalid(refused: Refused) -> Status {
    Status::invalid_argument(format!("{}: {}", refused.path, refused.why))
}

/// A `REPLACE` of a URL no row stands under.
fn absent() -> Status {
    Status::not_found("no worker is registered under the URL")
}

/// A row that left or moved between a call's read and its write: no call of
/// the door does that to another, so a second writer did.
fn moved() -> Status {
    Status::aborted("the row moved under the call")
}

/// The contract's names of the door's three calls, as an audit line says
/// them.
const REGISTER: &str = "RegisterWorker";
const PATCH: &str = "PatchWorker";
const DEREGISTER: &str = "DeregisterWorker";

/// The low bits of a life's number hold the store's epoch, counted from the
/// door's base; the leader's term stands above them.
const EPOCH_BITS: u32 = 40;

/// The bound of a base and of an epoch, each: their sum and one stay inside
/// the low bits.
const HALF: u64 = 1 << (EPOCH_BITS - 1);

/// The number of the life a registration makes under `term` in a store that
/// stands at `epoch`, at a door whose count starts at `base`.
///
/// Every life the store held was numbered at an earlier epoch of this term
/// or under an earlier term, and a registration moves the epoch, so no two
/// lives share a number, the later of two lives of one run under one term
/// has the higher one, and the id of a URL's next life is one no life of it
/// had. Another run under the same term counts from another base. A term
/// past 2^24, or a base or an epoch past 2^39, would carry into the other's
/// bits, and a number that wraps repeats: a debug build asserts the three
/// bounds (a term a second for half a year, a thousand writes a second for
/// seventeen years).
fn life(term: u64, base: u64, epoch: u64) -> u64 {
    debug_assert!(
        term >> (u64::BITS - EPOCH_BITS) == 0,
        "a term past its 24 bits: {term}"
    );
    debug_assert!(base < HALF, "a base past its 39 bits: {base}");
    debug_assert!(epoch < HALF, "an epoch past its 39 bits: {epoch}");
    (term << EPOCH_BITS) | (base.wrapping_add(epoch).wrapping_add(1) & ((1 << EPOCH_BITS) - 1))
}

impl Door {
    /// The door over `store` and the log it journals to, its count starting
    /// at a draw of this run: 39 bits of a hash under the keys the standard
    /// library draws for a process's maps, so the crate gains no generator
    /// for one number.
    pub(crate) fn new(store: Arc<MemoryStore>, log: Arc<Log>) -> Self {
        Self::based(store, log, RandomState::new().hash_one(0u8) % HALF)
    }

    /// The door over `store` and its log, its count starting at `base`,
    /// below 2^39: a run draws it, a case fixes it.
    pub(crate) fn based(store: Arc<MemoryStore>, log: Arc<Log>, base: u64) -> Self {
        Self {
            store,
            base,
            log,
            turn: Mutex::new(()),
        }
    }

    /// The log's place: read after a call's write, that write's own.
    fn place(&self) -> Position {
        self.log.head(Kinds::Watched)
    }

    /// The number of the life a registration makes now, under `admitted`,
    /// the term its call was admitted under. **A life is minted only under
    /// a term its log leads:** the log's term is the highest it has seen, so
    /// where the store follows another history, or the term moved after the
    /// call's admission, a number under it could be one another leader gave
    /// another worker. The call is refused then, and no life is made.
    ///
    /// The call was admitted by the same read of the log
    /// ([`Log::leads`]), made again here in the door's turn: the log may
    /// have moved since.
    fn next_life(&self, admitted: u64) -> Result<u64, Status> {
        if self.log.leads(admitted).is_err() {
            return Err(Status::unavailable(LEADS_NO_TERM));
        }
        Ok(life(admitted, self.base, self.store.epoch()))
    }

    /// One call of the door, admitted from `by`: its turn, the term
    /// `leading` (the service's read of the fence) tells once the turn is
    /// held, its write under that term, and the audit line of a refusal.
    /// Every call leaves a line with its caller (design 5.2): a write's is
    /// [`told`]'s, a refusal's says its code and its words, which hold no
    /// URL.
    fn call(
        &self,
        call: &'static str,
        by: &Principal,
        leading: impl FnOnce() -> Result<u64, Status>,
        write: impl FnOnce(u64) -> Result<proto::WorkerEvent, Status>,
    ) -> Result<proto::WorkerEvent, Status> {
        let turn = self.turn.lock();
        let answer = leading().and_then(write);
        drop(turn);
        if let Err(refusal) = &answer {
            info!(
                call,
                by = %by.subject,
                code = ?refusal.code(),
                why = refusal.message(),
                "membership call refused"
            );
        }
        answer
    }

    /// `RegisterWorker`, admitted from `by` under the term `leading` reads.
    ///
    /// `ADD` (and no mode) wants the URL absent; `REPLACE` wants it standing
    /// and makes its next incarnation; `UPSERT` registers an absent URL and
    /// lays the registration over a standing row: equal facts write nothing
    /// and answer the row unmoved, a change of role, wire or models is a new
    /// incarnation, any other change is made in place.
    ///
    /// `wait_ready` is honoured where the checks are off (the answer is the
    /// row after its connect signal) and refused otherwise: no handshake
    /// runs here.
    pub(crate) fn register(
        &self,
        by: &Principal,
        leading: impl FnOnce() -> Result<u64, Status>,
        request: &proto::RegisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        self.call(REGISTER, by, leading, |term| self.upsert(by, term, request))
    }

    /// The write of [`Self::register`], in the door's turn.
    fn upsert(
        &self,
        by: &Principal,
        admitted: u64,
        request: &proto::RegisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let Some(spec) = &request.spec else {
            return Err(invalid(Refused {
                path: "spec",
                why: "a registration carries its spec",
            }));
        };
        let mode = match Mode::try_from(request.mode) {
            Ok(Mode::Unspecified | Mode::Add) => Mode::Add,
            Ok(mode) => mode,
            Err(_) => {
                return Err(invalid(Refused {
                    path: "mode",
                    why: "one of add, replace and upsert",
                }));
            }
        };
        // The conversion reads the standing row once the URL holds its own
        // rules: a URL past its bound is never looked up. What a new row
        // names is no rule of a call that makes none: a REPLACE of a URL
        // that does not stand is not found, whatever it leaves unsaid. The
        // conversion tells a new row's names after every field's own rule,
        // so a field past its bound is refused as itself here too.
        let read = worker_wire::registration(spec, |url| self.store.by_key(url));
        let (mut facts, standing) = match read {
            Err(unnamed) if mode == Mode::Replace && worker_wire::of_a_new_row(&unnamed) => {
                return Err(absent());
            }
            read => read.map_err(invalid)?,
        };
        if request.wait_ready && facts.health.probes {
            return Err(invalid(Refused {
                path: "wait_ready",
                why: "no handshake runs in this build: only a worker whose checks are off is ready at once",
            }));
        }
        let old = match (mode, standing) {
            (Mode::Add, Some(row)) => {
                return Err(Status::already_exists(format!(
                    "the URL is registered as worker {:016x} at revision {}",
                    row.id.0, row.revision
                )));
            }
            (Mode::Replace, None) => return Err(absent()),
            (_, None) => {
                facts.incarnation = self.next_life(admitted)?;
                None
            }
            (mode, Some(row)) => {
                if mode == Mode::Upsert && facts == row.facts {
                    let epoch = self.store.epoch();
                    let at = self.place();
                    let unmoved = worker_wire::relabelled(&row);
                    return Ok(told(REGISTER, by, unmoved, false, epoch, at));
                }
                // The store carries a status across incarnations only where
                // these three stand: a change of one is a new worker under
                // the URL. The models stand as a set: a reorder alone is
                // made in place.
                let another = facts.role != row.facts.role
                    || facts.wire_modes != row.facts.wire_modes
                    || !facts.same_models(&row.facts);
                if mode == Mode::Replace || another {
                    facts.incarnation = self.next_life(admitted)?;
                }
                Some(row)
            }
        };
        let id = self.store.upsert(facts).id();
        // The row, the epoch and the place as the registration's write left
        // them: the answer of a call that does not wait.
        let (mut now, mut epoch, mut at) = (self.read(id)?, self.store.epoch(), self.place());
        // A fresh row is pending: any other status the write left came over
        // from the old incarnation. Read before the connect signal moves it.
        let carried = now.status != WorkerStatus::Pending;
        if !now.facts.health.probes {
            self.store.connected(id, now.revision);
            if request.wait_ready {
                (now, epoch, at) = (self.read(id)?, self.store.epoch(), self.place());
            }
        }
        let event = match old {
            None => worker_wire::registered(&now),
            Some(old) if old.id == id => worker_wire::relabelled(&now),
            Some(old) => worker_wire::replaced(&old, &now, carried),
        };
        Ok(told(REGISTER, by, event, true, epoch, at))
    }

    /// `PatchWorker`, admitted from `by` under the term `leading` reads: the
    /// labels merged, the overload block replaced when present, under the
    /// revision the caller read. A patch that changes nothing writes nothing.
    pub(crate) fn patch(
        &self,
        by: &Principal,
        leading: impl FnOnce() -> Result<u64, Status>,
        request: &proto::PatchWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        self.call(PATCH, by, leading, |_| self.relabel(by, request))
    }

    /// The write of [`Self::patch`], in the door's turn.
    fn relabel(
        &self,
        by: &Principal,
        request: &proto::PatchWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let standing = self.guarded(request.worker_id, request.revision)?;
        let (now, wrote) = match worker_wire::patch(request, &standing.facts).map_err(invalid)? {
            None => (standing, false),
            Some(patch) => {
                self.store
                    .patch(standing.id, patch, standing.revision)
                    .ok_or_else(moved)?;
                (self.read(standing.id)?, true)
            }
        };
        let epoch = self.store.epoch();
        Ok(told(
            PATCH,
            by,
            worker_wire::relabelled(&now),
            wrote,
            epoch,
            self.place(),
        ))
    }

    /// `DeregisterWorker`, admitted from `by` under the term `leading`
    /// reads, under the revision the caller read. A drain moves the row to
    /// draining and answers the status change; no lease is counted here, so
    /// the row leaves by a later call without the flag. A drain of a
    /// draining row writes nothing and answers the row unmoved.
    pub(crate) fn deregister(
        &self,
        by: &Principal,
        leading: impl FnOnce() -> Result<u64, Status>,
        request: &proto::DeregisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        self.call(DEREGISTER, by, leading, |_| self.remove(by, request))
    }

    /// The write of [`Self::deregister`], in the door's turn.
    fn remove(
        &self,
        by: &Principal,
        request: &proto::DeregisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let standing = self.guarded(request.worker_id, request.revision)?;
        let (event, wrote) = if !request.drain {
            self.store
                .remove(standing.id, standing.revision)
                .ok_or_else(moved)?;
            (worker_wire::removed(&standing), true)
        } else if standing.status == WorkerStatus::Draining {
            (worker_wire::relabelled(&standing), false)
        } else {
            self.store
                .set_status(standing.id, WorkerStatus::Draining, standing.revision)
                .ok_or_else(moved)?;
            let event =
                worker_wire::status_changed(standing.id, standing.status, WorkerStatus::Draining);
            (event, true)
        };
        let epoch = self.store.epoch();
        Ok(told(DEREGISTER, by, event, wrote, epoch, self.place()))
    }

    /// The row under `id` when `revision` is the one it stands at.
    fn guarded(&self, id: u64, revision: u64) -> Result<WorkerRow, Status> {
        let row = self
            .store
            .get(WorkerId(id))
            .ok_or_else(|| Status::not_found("no worker has the id"))?;
        if row.revision == revision {
            Ok(row)
        } else {
            Err(Status::failed_precondition(format!(
                "the row is at revision {}",
                row.revision
            )))
        }
    }

    fn read(&self, id: WorkerId) -> Result<WorkerRow, Status> {
        self.store.get(id).ok_or_else(moved)
    }
}

/// The call's answer at `at`, the log's place after the call, and its audit
/// line: the call, who it was from, the worker, what the call made of it
/// (`nothing` where it wrote none) and that place. Never the URL: it may
/// carry a credential.
fn told(
    call: &'static str,
    by: &Principal,
    event: Event,
    wrote: bool,
    membership_epoch: u64,
    at: Position,
) -> proto::WorkerEvent {
    let (made, worker) = match &event {
        Event::Registered(row) => ("registered", row.id),
        Event::Relabelled(row) => ("relabelled", row.id),
        Event::Replaced(replaced) => ("replaced", replaced.new.as_ref().map_or(0, |row| row.id)),
        Event::Removed(removed) => ("removed", removed.worker_id),
        Event::Status(status) => ("status", status.worker_id),
        Event::Snapshot(_) | Event::Needs(_) | Event::SnapshotRequired(_) => ("", 0),
    };
    let made = if wrote { made } else { "nothing" };
    info!(
        call,
        by = %by.subject,
        worker = %format_args!("{worker:016x}"),
        made,
        membership_epoch,
        seq = at.seq,
        term = at.term,
        "membership write"
    );
    worker_wire::event(event, at.seq, membership_epoch, at.term)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        io,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, Barrier,
        },
        thread,
    };

    use parking_lot::Mutex;
    use tonic::{Code, Status};

    use super::{Door, Position};
    use crate::{
        controller::{
            election::LeasePlace,
            principal::{tests::told, Principal},
            replication::{journaled, journaled_from},
        },
        proto::{self, register_worker_request::Mode, worker_event::Event},
        store::{self, identity, MemoryStore, WorkerStatus, WorkersStore},
        view::WorkerId,
    };

    const URL: &str = "grpc://10.0.0.5:8000";
    const TERM: u64 = 7;
    /// The bound of a base and of an epoch by its number, 2^39: a case reads
    /// it here and not through the door's constant, so that constant's value
    /// stands on a case.
    const BOUND: u64 = 1 << 39;
    /// The copies of one call that [`at_once`] lets go together.
    const CALLERS: usize = 4;

    /// The peer the door's calls are admitted from.
    fn operator() -> Principal {
        told("operator-a")
    }

    /// The fence of a replica that leads under `term`, as the service hands
    /// it to a call.
    fn under(term: u64) -> impl Fn() -> Result<u64, Status> {
        move || Ok(term)
    }

    /// A door over a journaled store that stands in [`TERM`], its count
    /// starting at zero so a case's table reads [`life`]: the term's start
    /// is the log's first place, so a case's first write is its second.
    fn door() -> (Arc<MemoryStore>, Door) {
        door_in(TERM)
    }

    /// The same under another term.
    fn door_in(term: u64) -> (Arc<MemoryStore>, Door) {
        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        store::LeasesStore::set_term(store.as_ref(), term);
        (Arc::clone(&store), Door::based(store, feed.log, 0))
    }

    /// A life's number at a door whose count starts at zero.
    fn life(term: u64, epoch: u64) -> u64 {
        super::life(term, 0, epoch)
    }

    /// The id of the life the door numbers under [`TERM`] in a store that
    /// stands at `epoch`.
    fn numbered(url: &str, epoch: u64) -> WorkerId {
        identity(url, life(TERM, epoch))
    }

    fn spec(url: &str) -> proto::WorkerSpec {
        proto::WorkerSpec {
            url: url.to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec!["model-a".to_owned()],
            ..Default::default()
        }
    }

    fn unchecked(url: &str) -> proto::WorkerSpec {
        proto::WorkerSpec {
            health_policy: Some(proto::HealthPolicy {
                disabled: true,
                ..Default::default()
            }),
            ..spec(url)
        }
    }

    fn ask(spec: proto::WorkerSpec, mode: Mode) -> proto::RegisterWorkerRequest {
        proto::RegisterWorkerRequest {
            spec: Some(spec),
            mode: mode as i32,
            wait_ready: false,
        }
    }

    /// The event's row, for the three kinds that carry one.
    fn row_of(event: &proto::WorkerEvent) -> proto::WorkerRow {
        match event.event.clone().expect("an event") {
            Event::Registered(row) | Event::Relabelled(row) => row,
            Event::Replaced(replaced) => replaced.new.expect("the new row"),
            other => panic!("no row in {other:?}"),
        }
    }

    #[test]
    fn a_registration_writes_the_row_its_caller_stated_and_answers_it_pending() {
        let (store, door) = door();
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
        let id = numbered(URL, 0);
        let stored = store.get(id).expect("the row is in the store");
        assert_eq!(
            told,
            proto::WorkerEvent {
                seq: 2,
                membership_epoch: 1,
                event: Some(Event::Registered(super::worker_wire::row(&stored))),
                term: TERM,
            }
        );
        let row = row_of(&told);
        assert_eq!(
            (row.id, row.incarnation, row.revision, row.status),
            (id.0, life(TERM, 0), 1, proto::WorkerStatus::Pending as i32)
        );
        assert_eq!(
            (stored.status, stored.revision, store.epoch()),
            (WorkerStatus::Pending, 1, 1)
        );
        // No mode reads as an add.
        let other = "grpc://10.0.0.6:8000";
        let unset = proto::RegisterWorkerRequest {
            spec: Some(spec(other)),
            ..Default::default()
        };
        let told = door
            .register(&operator(), under(TERM), &unset)
            .expect("registered");
        assert!(matches!(told.event, Some(Event::Registered(_))));
        assert_eq!((told.membership_epoch, told.seq, told.term), (2, 3, TERM));
    }

    #[test]
    fn a_second_add_is_refused_and_writes_nothing() {
        let (store, door) = door();
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
        let refused = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect_err("the URL stands");
        assert_eq!(refused.code(), Code::AlreadyExists);
        assert_eq!(
            refused.message(),
            format!(
                "the URL is registered as worker {:016x} at revision 1",
                numbered(URL, 0).0
            )
        );
        assert!(!refused.message().contains("10.0.0.5"));
        assert_eq!(store.epoch(), 1);
    }

    #[test]
    fn a_replace_wants_a_standing_row_and_makes_its_next_incarnation() {
        let (store, door) = door();
        let absent = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
            .expect_err("nothing to replace");
        assert_eq!(absent.code(), Code::NotFound);
        // What a new row names is no rule of a call that makes none: a
        // REPLACE that leaves the wire and the role unsaid is not found
        // either, and so is one that leaves the role alone unsaid.
        let bare = proto::WorkerSpec {
            url: URL.to_owned(),
            models: vec!["model-a".to_owned()],
            ..Default::default()
        };
        let roleless = proto::WorkerSpec {
            wire: proto::WireMode::Grpc as i32,
            ..bare.clone()
        };
        for unsaid in [bare.clone(), roleless] {
            let absent = door
                .register(&operator(), under(TERM), &ask(unsaid, Mode::Replace))
                .expect_err("nothing to replace");
            assert_eq!(
                (absent.code(), absent.message()),
                (Code::NotFound, "no worker is registered under the URL")
            );
        }
        // A field past its bounds is refused as itself, found or not, and
        // whatever the call leaves unsaid: a field read ahead of the wire
        // (the models) and one read after it (the labels), each with the
        // wire and the role stated and with both left unsaid.
        let wide = |over: proto::WorkerSpec| proto::WorkerSpec {
            models: (0..129).map(|n| format!("model-{n}")).collect(),
            ..over
        };
        let labelled = |over: proto::WorkerSpec| proto::WorkerSpec {
            labels: (0..129)
                .map(|n| (format!("label-{n}"), String::new()))
                .collect(),
            ..over
        };
        let models = "spec.models: a row names at most 128 model ids";
        let labels = "spec.labels: a row holds at most 128 labels";
        for (past, field) in [
            (wide(spec(URL)), models),
            (wide(bare.clone()), models),
            (labelled(spec(URL)), labels),
            (labelled(bare.clone()), labels),
        ] {
            let refused = door
                .register(&operator(), under(TERM), &ask(past, Mode::Replace))
                .expect_err("past a bound");
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::InvalidArgument, field)
            );
        }
        // An UPSERT of the same would make a row, and is held to its names.
        let unnamed = door
            .register(&operator(), under(TERM), &ask(bare, Mode::Upsert))
            .expect_err("a new row names its wire");
        assert_eq!(unnamed.code(), Code::InvalidArgument);
        assert_eq!(store.epoch(), 0);

        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
        let old = store.get(numbered(URL, 0)).expect("the first life");
        store.set_status(old.id, WorkerStatus::Ready, old.revision);
        let old = store.get(old.id).expect("the first life, ready");

        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
            .expect("replaced");
        // Numbered at the epoch the two writes above left.
        let new = store.get(numbered(URL, 2)).expect("the second life");
        assert_eq!(store.get(old.id), None);
        assert_eq!(
            told.event,
            Some(Event::Replaced(proto::Replaced {
                old: Some(super::worker_wire::row(&old)),
                new: Some(super::worker_wire::row(&new)),
                // The same role, wire and models: the status came over.
                state_carried: true,
            }))
        );
        assert_eq!((new.status, new.revision), (WorkerStatus::Ready, 1));
    }

    #[test]
    fn a_replacement_says_carried_only_where_the_status_came_over() {
        let (store, door) = door();
        let waited = |spec, mode| proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(spec, mode)
        };
        door.register(&operator(), under(TERM), &waited(unchecked(URL), Mode::Add))
            .expect("registered");
        // The same role, wire and models: the ready status came over.
        let told = door
            .register(
                &operator(),
                under(TERM),
                &waited(unchecked(URL), Mode::Replace),
            )
            .expect("replaced");
        let Some(Event::Replaced(same)) = told.event else {
            panic!("a replacement: {told:?}");
        };
        assert!(same.state_carried);
        // Another model: a fresh row, ready by its own connect signal, so
        // nothing came over.
        let other = proto::WorkerSpec {
            models: vec!["model-b".to_owned()],
            ..unchecked(URL)
        };
        let told = door
            .register(&operator(), under(TERM), &waited(other, Mode::Replace))
            .expect("replaced");
        let Some(Event::Replaced(fresh)) = told.event else {
            panic!("a replacement: {told:?}");
        };
        assert!(!fresh.state_carried);
        assert_eq!(
            fresh.new.map(|new| new.status),
            Some(proto::WorkerStatus::Ready as i32)
        );
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Ready)
        );
    }

    #[test]
    fn an_upsert_of_equal_facts_writes_nothing_and_answers_the_row_unmoved() {
        let (store, door) = door();
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
            .expect("registered");
        let stood = store.get(numbered(URL, 0)).expect("the row");
        for _ in 0..3 {
            let told = door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
                .expect("answered");
            assert_eq!(
                told,
                proto::WorkerEvent {
                    seq: 2,
                    membership_epoch: 1,
                    event: Some(Event::Relabelled(super::worker_wire::row(&stood))),
                    term: TERM,
                }
            );
        }
        assert_eq!(store.epoch(), 1);
        assert_eq!(store.get(stood.id), Some(stood));
    }

    #[test]
    fn an_upsert_changes_a_standing_row_in_place_or_makes_a_new_incarnation() {
        let (store, door) = door();
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
        let first = numbered(URL, 0);

        // A label: the same worker, its facts moved.
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..spec(URL)
        };
        let told = door
            .register(
                &operator(),
                under(TERM),
                &ask(labelled.clone(), Mode::Upsert),
            )
            .expect("relabelled");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        let row = row_of(&told);
        assert_eq!(
            (row.id, row.incarnation, row.revision),
            (first.0, life(TERM, 0), 2)
        );
        assert_eq!(row.labels.get("tier").map(String::as_str), Some("gold"));
        assert_eq!(told.membership_epoch, 2);

        // The role: another worker under the URL, so a new incarnation.
        let decode = proto::WorkerSpec {
            role: proto::Role::Decode as i32,
            ..labelled
        };
        let told = door
            .register(&operator(), under(TERM), &ask(decode.clone(), Mode::Upsert))
            .expect("replaced");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a change of role is a replacement: {told:?}");
        };
        assert!(!replaced.state_carried);
        let new = replaced.new.expect("the new row");
        assert_eq!(
            (new.id, new.incarnation, new.revision),
            (numbered(URL, 2).0, life(TERM, 2), 1)
        );
        assert_eq!(replaced.old.map(|old| old.id), Some(first.0));
        assert_eq!(store.get(first), None);

        // The models alone, then the wire alone: each is another worker
        // under the URL too.
        let remodelled = proto::WorkerSpec {
            models: vec!["model-b".to_owned()],
            ..decode
        };
        let rewired = proto::WorkerSpec {
            wire: proto::WireMode::Http as i32,
            ..remodelled.clone()
        };
        for (what, changed) in [("models", remodelled), ("wire", rewired)] {
            let stood = store.by_key(URL).expect("the row");
            let told = door
                .register(&operator(), under(TERM), &ask(changed, Mode::Upsert))
                .expect("replaced");
            let Some(Event::Replaced(replaced)) = told.event else {
                panic!("a change of {what} is a replacement: {told:?}");
            };
            assert_eq!(replaced.old.map(|old| old.id), Some(stood.id.0), "{what}");
            assert_eq!(store.get(stood.id), None, "{what}");
        }
    }

    #[test]
    fn an_upsert_of_the_same_models_in_another_order_is_made_in_place() {
        let (store, door) = door();
        let listing = |models: &[&str]| proto::WorkerSpec {
            models: models.iter().map(|&model| model.to_owned()).collect(),
            ..unchecked(URL)
        };
        let two = listing(&["model-a", "model-b"]);
        let told = door
            .register(&operator(), under(TERM), &ask(two.clone(), Mode::Add))
            .expect("registered");
        let first = row_of(&told);
        // Unchecked, the row is ready by its connect signal: a state to
        // carry.
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Ready)
        );
        // The same two in the other order, then one of them named twice:
        // the same worker each time, and the row tells the stated list.
        for stated in [
            &["model-b", "model-a"][..],
            &["model-a", "model-b", "model-a"][..],
        ] {
            let told = door
                .register(
                    &operator(),
                    under(TERM),
                    &ask(listing(stated), Mode::Upsert),
                )
                .expect("reordered");
            let Some(Event::Relabelled(row)) = told.event else {
                panic!("a reorder is made in place: {told:?}");
            };
            assert_eq!((row.id, row.incarnation), (first.id, first.incarnation));
            assert_eq!(row.models, stated);
            assert_eq!(row.status, proto::WorkerStatus::Ready as i32);
        }
        // A replacement that names the same set carries the status too.
        let told = door
            .register(&operator(), under(TERM), &ask(two, Mode::Replace))
            .expect("replaced");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a replacement: {told:?}");
        };
        assert!(replaced.state_carried);
    }

    #[test]
    fn a_registration_with_its_checks_off_is_ready_at_once() {
        let (store, door) = door();
        // Not waited for: the answer is the pending row, and the store has
        // moved it to ready by the next read, at the revision the answer
        // gave: a status change moves none.
        let told = door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered");
        let row = row_of(&told);
        assert_eq!(
            (row.status, row.revision, told.membership_epoch),
            (proto::WorkerStatus::Pending as i32, 1, 1)
        );
        let stored = store.get(numbered(URL, 0)).expect("the row");
        assert_eq!(
            (stored.status, stored.revision, store.epoch()),
            (WorkerStatus::Ready, 1, 2)
        );

        // Waited for: the answer is the row after its connect signal.
        let other = "grpc://10.0.0.6:8000";
        let waited = proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(unchecked(other), Mode::Add)
        };
        let told = door
            .register(&operator(), under(TERM), &waited)
            .expect("registered");
        let row = row_of(&told);
        assert_eq!(
            (row.status, row.revision, told.membership_epoch),
            (proto::WorkerStatus::Ready as i32, 1, 4)
        );
        assert_eq!(row.health, proto::HealthFlag::Healthy as u32);

        // A checked row that turns its checks off is ready from that write.
        let third = "grpc://10.0.0.7:8000";
        door.register(&operator(), under(TERM), &ask(spec(third), Mode::Add))
            .expect("registered");
        door.register(
            &operator(),
            under(TERM),
            &ask(unchecked(third), Mode::Upsert),
        )
        .expect("relabelled");
        let stored = store.by_key(third).expect("the row");
        assert_eq!(stored.status, WorkerStatus::Ready);
    }

    #[test]
    fn a_wait_for_a_checked_worker_and_a_request_out_of_order_are_refused_by_name() {
        let (store, door) = door();
        let waited = proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(spec(URL), Mode::Add)
        };
        let bare = proto::RegisterWorkerRequest::default();
        let unknown = proto::RegisterWorkerRequest {
            mode: 9,
            ..ask(spec(URL), Mode::Add)
        };
        let wide = ask(
            proto::WorkerSpec {
                dp_size: 4,
                ..spec(URL)
            },
            Mode::Add,
        );
        for (request, path) in [
            (waited, "wait_ready: "),
            (bare, "spec: "),
            (unknown, "mode: "),
            (wide, "spec.dp_size: "),
        ] {
            let refused = door
                .register(&operator(), under(TERM), &request)
                .expect_err("refused");
            assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
            assert!(
                refused.message().starts_with(path),
                "{path}: {}",
                refused.message()
            );
        }
        assert_eq!(store.epoch(), 0);
    }

    #[test]
    fn an_answer_carries_the_place_of_its_calls_write_under_that_places_term() {
        let (store, door) = door();
        // Not waited: the registration as its write left it. The connect
        // signal's write is the next place.
        let told = door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered");
        assert_eq!((told.seq, told.term, told.membership_epoch), (2, TERM, 1));
        assert_eq!(row_of(&told).status, proto::WorkerStatus::Pending as i32);
        // Waited: the row after its connect signal, at that write's place.
        let other = "grpc://10.0.0.6:8000";
        let waited = proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(unchecked(other), Mode::Add)
        };
        let told = door
            .register(&operator(), under(TERM), &waited)
            .expect("registered");
        assert_eq!((told.seq, told.term, told.membership_epoch), (5, TERM, 4));
        assert_eq!(row_of(&told).status, proto::WorkerStatus::Ready as i32);
        // A call that writes nothing answers the place the log stands at:
        // here the start of the next term.
        store::LeasesStore::set_term(store.as_ref(), TERM + 1);
        let told = door
            .register(
                &operator(),
                under(TERM + 1),
                &ask(unchecked(other), Mode::Upsert),
            )
            .expect("answered");
        assert_eq!(
            (told.seq, told.term, told.membership_epoch),
            (6, TERM + 1, 4)
        );
        // And the next write is made under that term.
        let drain = proto::DeregisterWorkerRequest {
            worker_id: row_of(&told).id,
            revision: row_of(&told).revision,
            drain: true,
        };
        let told = door
            .deregister(&operator(), under(TERM + 1), &drain)
            .expect("draining");
        assert_eq!(
            (told.seq, told.term, told.membership_epoch),
            (7, TERM + 1, 5)
        );
    }

    #[test]
    fn a_patch_merges_under_the_revision_its_caller_read() {
        let (store, door) = door();
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..spec(URL)
        };
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &ask(labelled, Mode::Add))
                .expect("registered"),
        );
        let patch = proto::PatchWorkerRequest {
            worker_id: row.id,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            remove_labels: vec!["tier".to_owned()],
            overload: None,
            revision: row.revision,
        };
        let told = door
            .patch(&operator(), under(TERM), &patch)
            .expect("patched");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        let now = row_of(&told);
        assert_eq!(now.labels, [("lane".to_owned(), "b".to_owned())].into());
        assert_eq!(
            (now.revision, told.membership_epoch, told.seq, told.term),
            (2, 2, 3, TERM)
        );

        // The same patch under the revision it read: the row moved.
        let stale = door
            .patch(&operator(), under(TERM), &patch)
            .expect_err("a stale revision");
        assert_eq!(stale.code(), Code::FailedPrecondition);
        assert_eq!(stale.message(), "the row is at revision 2");

        // Under the row's revision it changes nothing and writes nothing.
        let again = proto::PatchWorkerRequest {
            revision: 2,
            ..patch.clone()
        };
        let told = door
            .patch(&operator(), under(TERM), &again)
            .expect("answered");
        assert_eq!(
            (row_of(&told).revision, told.membership_epoch, told.seq),
            (2, 2, 3)
        );
        assert_eq!(store.epoch(), 2);

        let nobody = proto::PatchWorkerRequest {
            worker_id: 1,
            ..patch
        };
        let absent = door
            .patch(&operator(), under(TERM), &nobody)
            .expect_err("no such worker");
        assert_eq!(absent.code(), Code::NotFound);
    }

    #[test]
    fn a_row_past_a_bound_is_refused_by_name_and_nothing_is_written() {
        fn labelled(count: usize) -> proto::WorkerSpec {
            let labels = (0..count).map(|n| (n.to_string(), "v".repeat(4_096)));
            proto::WorkerSpec {
                labels: labels.collect(),
                ..spec(URL)
            }
        }
        let (store, door) = door();
        // A registration: by its field, and by its spec for the row whole.
        let long = ask(spec(&"u".repeat(2_049)), Mode::Add);
        let heavy = ask(labelled(4), Mode::Add);
        for (request, path) in [(long, "spec.url: "), (heavy, "spec: ")] {
            let refused = door
                .register(&operator(), under(TERM), &request)
                .expect_err("refused");
            assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
            assert!(
                refused.message().starts_with(path),
                "{path}: {}",
                refused.message()
            );
        }
        assert_eq!(store.epoch(), 0);
        // A patch: by its labels, for the row it would make. The row stands.
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &ask(labelled(3), Mode::Add))
                .expect("registered"),
        );
        let fourth = proto::PatchWorkerRequest {
            worker_id: row.id,
            labels: [("3".to_owned(), "v".repeat(4_096))].into(),
            revision: row.revision,
            ..Default::default()
        };
        let refused = door
            .patch(&operator(), under(TERM), &fourth)
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(
            refused.message().starts_with("labels: "),
            "{}",
            refused.message()
        );
        let stands = store.get(numbered(URL, 0)).expect("the row");
        assert_eq!(
            (stands.revision, stands.facts.labels.len(), store.epoch()),
            (1, 3, 1)
        );
    }

    #[test]
    fn a_drain_answers_the_status_change_and_a_removal_the_row_as_it_last_stood() {
        let (store, door) = door();
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        let stale = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision + 1,
            drain: false,
        };
        let refused = door
            .deregister(&operator(), under(TERM), &stale)
            .expect_err("a stale revision");
        assert_eq!(refused.code(), Code::FailedPrecondition);

        let drain = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: true,
        };
        let told = door
            .deregister(&operator(), under(TERM), &drain)
            .expect("draining");
        assert_eq!(
            told.event,
            Some(Event::Status(proto::StatusChanged {
                worker_id: row.id,
                old: proto::WorkerStatus::Pending as i32,
                new: proto::WorkerStatus::Draining as i32,
                health: proto::HealthFlag::Draining as u32,
            }))
        );
        assert_eq!((told.membership_epoch, told.seq), (2, 3));
        // A second drain writes nothing and answers the row as it stands,
        // under the revision its registration gave: no status change moved
        // it.
        let told = door
            .deregister(&operator(), under(TERM), &drain)
            .expect("draining still");
        let stood = store.get(WorkerId(row.id)).expect("the row stands");
        assert_eq!(
            told.event,
            Some(Event::Relabelled(super::worker_wire::row(&stood)))
        );
        assert_eq!(
            (
                stood.status,
                stood.revision,
                told.membership_epoch,
                told.seq
            ),
            (WorkerStatus::Draining, row.revision, 2, 3)
        );

        let removal = proto::DeregisterWorkerRequest {
            drain: false,
            ..drain
        };
        let told = door
            .deregister(&operator(), under(TERM), &removal)
            .expect("removed");
        assert_eq!(
            told.event,
            Some(Event::Removed(proto::Removed {
                worker_id: row.id,
                registration_key: URL.to_owned(),
                revision: 1,
                draining: true,
            }))
        );
        assert_eq!((told.membership_epoch, told.seq, store.epoch()), (3, 4, 3));
        assert_eq!(store.by_key(URL), None);
        let gone = door
            .deregister(&operator(), under(TERM), &removal)
            .expect_err("gone");
        assert_eq!(gone.code(), Code::NotFound);
    }

    #[test]
    fn a_call_reads_the_fence_once_it_holds_the_doors_turn() {
        let (_store, door) = door();
        // The fence's reader finds the turn taken: the call that reads it
        // holds it.
        let held = || {
            assert!(door.turn.try_lock().is_none(), "read outside the turn");
            Ok(TERM)
        };
        let row = row_of(
            &door
                .register(&operator(), held, &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        let patch = proto::PatchWorkerRequest {
            worker_id: row.id,
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            revision: row.revision,
            ..Default::default()
        };
        let row = row_of(&door.patch(&operator(), held, &patch).expect("patched"));
        let removal = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: false,
        };
        door.deregister(&operator(), held, &removal)
            .expect("removed");
    }

    #[test]
    fn a_call_that_lost_the_lead_while_it_waited_its_turn_is_refused_and_writes_nothing() {
        let (store, door) = door();
        let leads = AtomicBool::new(true);
        let fence = || {
            if leads.load(Ordering::SeqCst) {
                Ok(TERM)
            } else {
                Err(Status::failed_precondition("this replica does not lead"))
            }
        };
        let request = ask(spec(URL), Mode::Add);
        // Another call holds the turn. This one waits behind it, and the
        // lead is lost before its turn comes.
        let turn = door.turn.lock();
        let answer = thread::scope(|scope| {
            let waiting = scope.spawn(|| door.register(&operator(), fence, &request));
            leads.store(false, Ordering::SeqCst);
            drop(turn);
            waiting.join().expect("the call returns")
        });
        let refused = answer.expect_err("the lead was lost before the call's turn");
        assert_eq!(refused.code(), Code::FailedPrecondition);
        assert_eq!((store.epoch(), store.by_key(URL)), (0, None));
    }

    #[test]
    fn a_registration_that_states_no_policy_is_checked_whatever_stood_under_its_url() {
        let (store, door) = door();
        let status = |url: &str| store.by_key(url).map(|row| row.status);
        door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered");
        assert_eq!(status(URL), Some(WorkerStatus::Ready));
        // The URL's next worker (another model: no status comes over)
        // states no policy. Nobody said "do not check" of it: a wait for
        // its handshake is refused as any checked worker's is, and it is
        // pending until a verdict.
        let other = proto::WorkerSpec {
            models: vec!["model-b".to_owned()],
            ..spec(URL)
        };
        let waited = proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(other.clone(), Mode::Replace)
        };
        let refused = door
            .register(&operator(), under(TERM), &waited)
            .expect_err("no handshake runs");
        assert_eq!(refused.code(), Code::InvalidArgument);
        let told = door
            .register(&operator(), under(TERM), &ask(other, Mode::Replace))
            .expect("replaced");
        let row = row_of(&told);
        assert_eq!(
            (row.status, row.health_policy.map(|policy| policy.disabled)),
            (proto::WorkerStatus::Pending as i32, Some(false))
        );
        assert_eq!(status(URL), Some(WorkerStatus::Pending));

        // In place too: an upsert that states no policy over an unchecked
        // row turns its checks on and leaves its status to the prober.
        let third = "grpc://10.0.0.7:8000";
        door.register(&operator(), under(TERM), &ask(unchecked(third), Mode::Add))
            .expect("registered");
        let told = door
            .register(&operator(), under(TERM), &ask(spec(third), Mode::Upsert))
            .expect("in place");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        let row = row_of(&told);
        assert_eq!(
            (row.status, row.health_policy.map(|policy| policy.disabled)),
            (proto::WorkerStatus::Ready as i32, Some(false))
        );
    }

    #[test]
    fn a_drain_ends_with_the_registration_it_named() {
        let (store, door) = door();
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
                .expect("registered"),
        );
        let drain = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: true,
        };
        door.deregister(&operator(), under(TERM), &drain)
            .expect("draining");
        // An upsert in place is the same registration: draining still.
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..unchecked(URL)
        };
        let told = door
            .register(&operator(), under(TERM), &ask(labelled, Mode::Upsert))
            .expect("in place");
        assert_eq!(row_of(&told).status, proto::WorkerStatus::Draining as i32);
        // The URL's next incarnation is another registration: the drain is
        // not handed on, and an unchecked worker is ready from its own.
        let told = door
            .register(
                &operator(),
                under(TERM),
                &ask(unchecked(URL), Mode::Replace),
            )
            .expect("replaced");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a replacement: {told:?}");
        };
        assert!(!replaced.state_carried);
        assert_eq!(
            store.by_key(URL).map(|row| row.status),
            Some(WorkerStatus::Ready)
        );
    }

    #[test]
    fn a_rows_revision_counts_its_stated_facts_and_no_status_change() {
        let (store, door) = door();
        // An unchecked worker turns ready behind its registration's answer
        // and is drained: neither moves the revision that answer gave.
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
                .expect("registered"),
        );
        let drain = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: true,
        };
        door.deregister(&operator(), under(TERM), &drain)
            .expect("draining");
        assert_eq!(
            store.by_key(URL).map(|row| (row.status, row.revision)),
            Some((WorkerStatus::Draining, row.revision))
        );
        // A patch states a fact: the revision moves, the one before it is
        // refused and the new one removes the row.
        let patch = proto::PatchWorkerRequest {
            worker_id: row.id,
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            revision: row.revision,
            ..Default::default()
        };
        let patched = row_of(
            &door
                .patch(&operator(), under(TERM), &patch)
                .expect("patched"),
        );
        assert_eq!(patched.revision, row.revision + 1);
        let stale = proto::DeregisterWorkerRequest {
            drain: false,
            ..drain
        };
        let refused = door
            .deregister(&operator(), under(TERM), &stale)
            .expect_err("the facts moved");
        assert_eq!(refused.code(), Code::FailedPrecondition);
        let removal = proto::DeregisterWorkerRequest {
            revision: patched.revision,
            ..stale
        };
        door.deregister(&operator(), under(TERM), &removal)
            .expect("removed");
    }

    /// The lines a case's own subscriber wrote.
    #[derive(Clone, Default)]
    pub(crate) struct Logged(pub(crate) Arc<Mutex<Vec<u8>>>);

    impl io::Write for Logged {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn every_call_leaves_its_audit_line_written_unmoved_or_refused() {
        // A subscriber of the case's own keeps this thread's lines. A
        // second one stays registered beside it: with one alone a line is
        // on or off by the thread that reaches it first, and a case beside
        // this one reaches it under none.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let (_store, door) = door();
        let add = ask(spec(URL), Mode::Add);
        let row = row_of(
            &door
                .register(&operator(), under(TERM), &add)
                .expect("registered"),
        );
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
            .expect("unmoved");
        door.register(&operator(), under(TERM), &add)
            .expect_err("the URL stands");
        let nothing = proto::PatchWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            ..Default::default()
        };
        door.patch(&operator(), under(TERM), &nothing)
            .expect("unmoved");
        let drain = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: true,
        };
        for answer in ["draining", "draining still"] {
            door.deregister(&operator(), under(TERM), &drain)
                .expect(answer);
        }

        // A line a call, in the calls' order: what a write made, `nothing`
        // where a call wrote none, a refusal by its code.
        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let lines: Vec<&str> = logged.lines().collect();
        let want = [
            ("membership write", "RegisterWorker", "made=\"registered\""),
            ("membership write", "RegisterWorker", "made=\"nothing\""),
            (
                "membership call refused",
                "RegisterWorker",
                "code=AlreadyExists",
            ),
            ("membership write", "PatchWorker", "made=\"nothing\""),
            ("membership write", "DeregisterWorker", "made=\"status\""),
            ("membership write", "DeregisterWorker", "made=\"nothing\""),
        ];
        assert_eq!(lines.len(), want.len(), "{logged}");
        for (line, (kind, call, made)) in lines.iter().zip(want) {
            let call = format!("call=\"{call}\" by=operator-a");
            assert!(
                line.contains(kind) && line.contains(&call) && line.contains(made),
                "{line}"
            );
        }
        // No line holds the URL: it may carry a credential.
        assert!(!logged.contains(URL), "{logged}");
    }

    #[test]
    fn a_registration_states_its_bootstrap_port_and_a_change_of_it_is_made_in_place() {
        let (store, door) = door();
        let with = |bootstrap_port| proto::WorkerSpec {
            pairing: Some(proto::PairingDescriptor {
                bootstrap_port,
                ..Default::default()
            }),
            ..spec(URL)
        };
        let port =
            |told: &proto::WorkerEvent| row_of(told).pairing.map(|pairing| pairing.bootstrap_port);
        let told = door
            .register(&operator(), under(TERM), &ask(with(8_998), Mode::Add))
            .expect("registered");
        let first = row_of(&told);
        assert_eq!(port(&told), Some(8_998));
        assert_eq!(
            store.by_key(URL).and_then(|row| row.facts.bootstrap_port),
            Some(8_998)
        );

        // Another port: the same worker, its facts moved.
        let told = door
            .register(&operator(), under(TERM), &ask(with(9_001), Mode::Upsert))
            .expect("relabelled");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        let moved = row_of(&told);
        assert_eq!((moved.id, moved.revision), (first.id, first.revision + 1));
        assert_eq!(port(&told), Some(9_001));

        // A registration that states no descriptor leaves the port: equal
        // facts, nothing written.
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
            .expect("answered");
        assert_eq!(port(&told), Some(9_001));
        assert_eq!((told.membership_epoch, store.epoch()), (2, 2));

        // A REPLACE that states no descriptor is the worker's next
        // incarnation and keeps the port that stands.
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
            .expect("replaced");
        let next = row_of(&told);
        assert_ne!(next.id, first.id, "a life of its own");
        assert_eq!(port(&told), Some(9_001));

        // Removed and registered again, the worker tells no port.
        let removal = proto::DeregisterWorkerRequest {
            worker_id: next.id,
            revision: next.revision,
            drain: false,
        };
        door.deregister(&operator(), under(TERM), &removal)
            .expect("removed");
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered again");
        assert_eq!(row_of(&told).pairing, None);

        // A worker registered with no port tells no descriptor.
        let other = "grpc://10.0.0.6:8000";
        let told = door
            .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
            .expect("registered");
        assert_eq!(row_of(&told).pairing, None);
    }

    #[test]
    fn a_life_is_numbered_by_the_term_above_the_stores_epoch() {
        assert_eq!(life(7, 0), (7 << 40) | 1);
        assert_eq!(life(7, 12), (7 << 40) | 13);
        assert_eq!(super::life(7, 5, 12), (7 << 40) | 18, "from its base");
        // The highest base with the highest epoch fills the low bits and
        // carries nothing into the term's.
        let highest = super::life(7, BOUND - 1, BOUND - 1);
        assert_eq!(highest, (7 << 40) | ((1 << 40) - 1));
        assert!(
            highest < life(8, 0),
            "a later term stands above every count of an earlier one"
        );

        // Two leaders that take the same URL first number two lives: the
        // pair read at one names no row at the other.
        let (_, earlier) = door_in(7);
        let (store, later) = door_in(8);
        let first = row_of(
            &earlier
                .register(&operator(), under(7), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        let second = row_of(
            &later
                .register(&operator(), under(8), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        assert_eq!(
            (first.incarnation, second.incarnation),
            (life(7, 0), life(8, 0))
        );
        assert_ne!(first.id, second.id);
        assert_eq!(first.revision, second.revision);
        let removal = proto::DeregisterWorkerRequest {
            worker_id: first.id,
            revision: first.revision,
            drain: false,
        };
        let refused = later
            .deregister(&operator(), under(8), &removal)
            .expect_err("another leader's registration");
        assert_eq!(refused.code(), Code::NotFound);
        assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(second.id));
    }

    #[test]
    fn a_life_is_minted_only_under_a_term_its_log_leads() {
        let (store, door) = door();
        let refused = |told: Result<proto::WorkerEvent, Status>| {
            let status = told.expect_err("no life under a term the log does not lead");
            assert_eq!(
                (status.code(), status.message()),
                (Code::Unavailable, super::LEADS_NO_TERM)
            );
        };
        // The call was admitted under a term the log has left, or one it
        // has not entered: the two terms are compared at the mint.
        for admitted in [TERM - 1, TERM + 1] {
            let told = door.register(&operator(), under(admitted), &ask(spec(URL), Mode::Add));
            refused(told);
        }
        // The store follows another leader's history at the admitted term:
        // a number under it could be one that leader gave another worker.
        door.log
            .restart(Position {
                seq: 40,
                term: TERM + 1,
            })
            .expect("a place the log takes");
        let told = door.register(&operator(), under(TERM + 1), &ask(spec(URL), Mode::Add));
        refused(told);
        assert_eq!(store.epoch(), 0, "a refused registration writes nothing");
        assert!(store.by_key(URL).is_none());
        // A take above every term the log has seen starts a lead again.
        store::LeasesStore::set_term(store.as_ref(), TERM + 2);
        let row = row_of(
            &door
                .register(&operator(), under(TERM + 2), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        assert_eq!(row.incarnation, life(TERM + 2, 0));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a term past its 24 bits")]
    fn a_term_past_its_bits_is_asserted_in_a_debug_build() {
        let _ = life(1 << 24, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "an epoch past its 39 bits")]
    fn an_epoch_past_its_bits_is_asserted_in_a_debug_build() {
        let _ = life(7, BOUND);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a base past its 39 bits")]
    fn a_base_past_its_bits_is_asserted_in_a_debug_build() {
        let _ = super::life(7, BOUND, 0);
    }

    #[test]
    fn a_run_draws_its_base_below_the_bound_and_apart_from_another_runs() {
        let drawn: Vec<u64> = (0..64)
            .map(|_| {
                let (store, feed) = journaled(&LeasePlace::Process);
                Door::new(store, feed.log).base
            })
            .collect();
        assert!(drawn.iter().all(|&base| base < BOUND), "{drawn:?}");
        let mut apart = drawn.clone();
        apart.sort_unstable();
        apart.dedup();
        assert_eq!(apart.len(), drawn.len(), "{drawn:?}");
    }

    #[test]
    fn two_runs_under_one_term_number_a_urls_lives_apart() {
        // A controller with no shared lease leads under term one at every
        // start: each run's door draws where its count starts.
        let run = || {
            let (store, feed) = journaled(&LeasePlace::Process);
            store::LeasesStore::set_term(store.as_ref(), 1);
            let door = Door::new(Arc::clone(&store), feed.log);
            let first = row_of(
                &door
                    .register(&operator(), under(1), &ask(spec(URL), Mode::Add))
                    .expect("registered"),
            );
            (store, door, first)
        };
        let (_, earlier, removed) = run();
        let (store, later, standing) = run();
        assert_ne!(removed.incarnation, standing.incarnation);
        assert_ne!(removed.id, standing.id);
        // Inside one run the count keeps its order: a later life has the
        // higher number.
        let next = row_of(
            &earlier
                .register(&operator(), under(1), &ask(spec(URL), Mode::Replace))
                .expect("replaced"),
        );
        assert!(next.incarnation > removed.incarnation);
        // A removal read at the earlier run names no row of the later one.
        let removal = proto::DeregisterWorkerRequest {
            worker_id: removed.id,
            revision: removed.revision,
            drain: false,
        };
        let refused = later
            .deregister(&operator(), under(1), &removal)
            .expect_err("the earlier run's registration");
        assert_eq!(refused.code(), Code::NotFound);
        assert_eq!(store.by_key(URL).map(|row| row.id.0), Some(standing.id));
    }

    #[test]
    fn a_removal_that_arrives_twice_never_takes_the_urls_next_registration() {
        let (store, door) = door();
        let first = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        let removal = proto::DeregisterWorkerRequest {
            worker_id: first.id,
            revision: first.revision,
            drain: false,
        };
        door.deregister(&operator(), under(TERM), &removal)
            .expect("removed");
        // The worker registers again before the removal's retry lands (its
        // first answer was lost): a new row, at the revision the first had.
        let second = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered again"),
        );
        assert_eq!(second.revision, first.revision);
        assert_ne!(second.id, first.id, "a life of its own");
        assert!(second.incarnation > first.incarnation);

        let retried = door
            .deregister(&operator(), under(TERM), &removal)
            .expect_err("the retry names a registration that left");
        assert_eq!(retried.code(), Code::NotFound);
        let stands = store.by_key(URL).expect("the second registration");
        assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
    }

    #[test]
    fn a_patch_under_a_registration_that_left_never_writes_the_urls_next_one() {
        let (store, door) = door();
        let first = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        let removal = proto::DeregisterWorkerRequest {
            worker_id: first.id,
            revision: first.revision,
            drain: false,
        };
        door.deregister(&operator(), under(TERM), &removal)
            .expect("removed");
        let second = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered again"),
        );
        // A patch its caller made of the first registration, arriving late.
        let late = proto::PatchWorkerRequest {
            worker_id: first.id,
            labels: [("lane".to_owned(), "late".to_owned())].into(),
            revision: first.revision,
            ..Default::default()
        };
        let refused = door
            .patch(&operator(), under(TERM), &late)
            .expect_err("the patch names a registration that left");
        assert_eq!(refused.code(), Code::NotFound);
        let stands = store.by_key(URL).expect("the second registration");
        assert_eq!((stands.id.0, stands.revision), (second.id, second.revision));
        assert!(stands.facts.labels.is_empty());
    }

    /// The answers of [`CALLERS`] copies of one call let go at one instant.
    fn at_once<T: Send>(call: impl Fn() -> Result<T, Status> + Sync) -> Vec<Result<T, Status>> {
        let (ready, go) = (Barrier::new(CALLERS + 1), AtomicBool::new(false));
        thread::scope(|scope| {
            let calls: Vec<_> = (0..CALLERS)
                .map(|_| {
                    scope.spawn(|| {
                        ready.wait();
                        while !go.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                        call()
                    })
                })
                .collect();
            ready.wait();
            go.store(true, Ordering::Release);
            calls
                .into_iter()
                .map(|call| call.join().expect("a caller"))
                .collect()
        })
    }

    /// What the callers that were refused were told.
    fn refusals<T>(answers: &[Result<T, Status>]) -> Vec<Code> {
        answers
            .iter()
            .filter_map(|answer| answer.as_ref().err().map(Status::code))
            .collect()
    }

    #[test]
    fn calls_at_once_take_their_turn_at_the_door() {
        // A call's read, its write and its read back are one step under the
        // door's lock. Of the copies of a call let go together one lands and
        // each other is told what it would be told after it, round after
        // round: an ADD that the URL stands, a patch that the row moved, a
        // removal that the row left.
        for round in 0..300 {
            let (store, door) = door();
            let added =
                at_once(|| door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add)));
            assert_eq!(
                refusals(&added),
                [Code::AlreadyExists; CALLERS - 1],
                "round {round}: {added:?}"
            );
            let row = store.by_key(URL).expect("the row");
            assert_eq!((store.epoch(), row.revision), (1, 1), "round {round}");

            let patch = proto::PatchWorkerRequest {
                worker_id: row.id.0,
                labels: [("lane".to_owned(), "b".to_owned())].into(),
                revision: row.revision,
                ..Default::default()
            };
            let patched = at_once(|| door.patch(&operator(), under(TERM), &patch));
            assert_eq!(
                refusals(&patched),
                [Code::FailedPrecondition; CALLERS - 1],
                "round {round}: {patched:?}"
            );

            let removal = proto::DeregisterWorkerRequest {
                worker_id: row.id.0,
                revision: 2,
                drain: false,
            };
            let removed = at_once(|| door.deregister(&operator(), under(TERM), &removal));
            assert_eq!(
                refusals(&removed),
                [Code::NotFound; CALLERS - 1],
                "round {round}: {removed:?}"
            );
            assert_eq!(
                (store.epoch(), store.by_key(URL)),
                (3, None),
                "round {round}"
            );
        }
    }
}

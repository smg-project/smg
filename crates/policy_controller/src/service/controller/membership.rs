//! The membership door (design 5.1 and 5.2) for control calls and provider
//! writes. Control calls answer with a wire event at the log's place or
//! with a refusal. A provider states all ranks in one turn and gets the
//! outcome of each row in the store's terms.
//!
//! A registered row is pending. One whose checks are off gets the store's
//! connect signal at once, as a worker with its health checks off starts
//! ready; the registrant says "do not check", never "healthy", and says it
//! of its own registration alone: one that states no policy is checked,
//! whatever stood under its URL.
//!
//! An answer carries the log's place read after the call's write, with the
//! term that place was written under: the write's own place or a later one.
//! The door is not the only writer of the kinds a watcher follows: a
//! probe's verdict moves a row's status beside it, and so does the start
//! of a term. A caller that watches the rows holds the call's write once
//! its stream has passed that place. A call that wrote nothing answers the
//! place the log stands at.
//!
//! **The door is the process's only membership writer.** A call's read, its
//! write and its read back run under the door's one lock, so a mode's test
//! and the answered row are exact. A second writer (discovery, a standby's
//! apply on a replica that also serves) breaks a mode's test: it and the
//! door then go through a compare-and-write of the store.
//!
//! **A call reads the record that stands off the worker objects.** The row
//! under a URL, the row under an id and an engine's rows are read back from
//! the objects in their holder's turn, behind its read of the store to its
//! end ([`Objects::record`]). A status is the store's word still, so a
//! record read carries the status the table holds.
//!
//! **A call writes under the term it reads once it holds the door's turn.**
//! The fence is read under the lock, first: a call that waited its turn
//! behind another writes under the term of its own minute, or is refused as
//! a follower's is, and never under a term read before the wait.
//!
//! **A call that wrote leaves the replica's worker objects holding its
//! write.** The door is made with the holder of those objects, which the
//! service's decisions read. A call's write of a row (a registration, a
//! patch, a removal) is decided over them and the store follows it
//! ([`Objects::upsert`]): the object stands when the write returns. A status
//! is the store's word still: a drain and a connect signal write the
//! table. So behind a write that answered, the door follows the store into
//! the objects before the call answers: a decision made behind an
//! acknowledged registration, drain or removal reads the fleet the call
//! left. A standby's applier writes its leader's rows through the holder
//! too ([`Objects::follow_at`]).
//!
//! **A registration that states several ranks is one write.** Each rank's
//! facts are read by the call's own rules, a rank at a time. The host
//! door's routine then checks the group, takes the engine's rows the
//! registration does not state, and writes. Every rule holds before the
//! first write, so a refusal of a rank or of the group writes nothing. A
//! lead lost between two ranks refuses there and leaves the ranks written
//! before it: an upsert lays the rest, and so does the same add called
//! again, where each rank that stands is its maker's with equal facts. The
//! answer tells what the call made of each rank, then each row that left,
//! at the place of the call's last write. A registration of one rank for
//! an engine that stands as ranks is refused: a plain row beside its ranks
//! would be the engine twice.
//!
//! A row has one maker ([`Registrant`]), the control call or provider that
//! registered it. A registration keeps the maker while the URL stands.
//! Wire upserts, replacements, patches and removals check the whole maker
//! before checking the revision. An add is refused whenever the row stands.
//! A refusal names the maker without exposing the URL or provider key.
//!
//! A host's drain or removal names the URL, row id and revision. The door
//! checks the guard first, then the asker. A caller acting as a maker must
//! match it whole. The host's discovery source can act for any key of its
//! provider. Health checks can act for any maker in their removal scope.
//!
//! **Every life of a URL has a number of its own:** the leader's term above the
//! store's epoch when the door numbered it, counted from a base the door
//! draws once a run. The store's id is a function of the URL and that
//! number, so the guard of a patch and of a removal (the id and the revision
//! their caller read) names one registration and never a later one of the
//! URL, at this leader or at another. At a later run under the same term
//! that holds by the draw, not by construction: two runs' counts meet with
//! odds near a run's writes over 2^38.

use std::{cell::RefCell, sync::Arc};

use parking_lot::Mutex;
use smg_worker::identity::{basic::Worker, builder::rank_url};
use tonic::{Code, Status};
use tracing::info;

use super::{
    principal::Principal,
    worker_wire::{self, Refused},
};
use crate::{
    engine::view::WorkerId,
    service::{
        proto::{self, register_worker_request::Mode, worker_event::Event},
        replication::Log,
    },
    store::{
        membership::{self, life, FactPastBound, WorkerFactsChange},
        HealthSink, MembershipChange, MemoryStore, Position, Registrant, WorkerFacts, WorkerPatch,
        WorkerRow, WorkerStatus, WorkersStore,
    },
    worker_host::{
        drain, lay, lay_one, lay_ready, refresh, stated, stating, take, Members,
        MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Objects, Over,
        Precondition, Ready, Remover, RowRevision, Stated, Turn,
    },
};

/// A registration's refusal where the store's log does not lead the term the
/// call was admitted under.
const LEADS_NO_TERM: &str =
    "this replica's store does not lead the term the call was admitted under";

/// The control calls that write worker membership, over one store.
pub(crate) struct MembershipCalls {
    store: Arc<MemoryStore>,
    /// Where this run's count of lives starts: below 2^39, drawn once.
    base: u64,
    /// The log the store journals to: where an answer reads its place.
    log: Arc<Log>,
    /// One call at a time, from its read to its read back.
    turn: Mutex<()>,
    /// The replica's worker objects: a call's write of a row is decided
    /// over them, and a call that wrote follows the store into them before
    /// it answers ([`Self::followed`]).
    objects: Arc<Objects<Members>>,
}

fn invalid(refused: Refused) -> Status {
    Status::invalid_argument(format!("{}: {}", refused.path, refused.why))
}

const ABSENT: &str = "no worker is registered under the URL";
const MOVED: &str = "the row moved under the call";
const DRAINING: &str = "the worker under the URL is draining";

/// A `REPLACE` of a URL no row stands under.
fn absent() -> Status {
    Status::not_found(ABSENT)
}

/// A row that left or moved between a call's read and its write: no call of
/// the door does that to another, so a second writer did.
fn moved() -> Status {
    Status::aborted(MOVED)
}

/// A write of a row by a caller that did not make it. The words name the
/// row's maker by its provider, never by its key, and never the URL. A
/// caller of that provider under another key is told so: where a new worker
/// took an address, its provider takes the old one away by the old key.
fn made_by(row: &WorkerRow, caller: &Registrant) -> Result<(), Status> {
    row.made_by(caller)
        .map_err(|not| Status::permission_denied(not.to_string()))
}

/// The contract's names of the door's three calls, as an audit line says
/// them.
const REGISTER: &str = "RegisterWorker";
const PATCH: &str = "PatchWorker";
const DEREGISTER: &str = "DeregisterWorker";

impl MembershipCalls {
    /// The door over `store`, the log it journals to and `objects`, the
    /// holder of the replica's worker objects, which its decisions read.
    /// The count of lives starts at a draw for this run.
    pub(crate) fn new(
        store: Arc<MemoryStore>,
        log: Arc<Log>,
        objects: Arc<Objects<Members>>,
    ) -> Self {
        Self::based(store, log, membership::base(), objects)
    }

    /// The door over `store`, its log and `objects`, its count starting at
    /// `base`, below 2^39: a run draws it, a case fixes it.
    pub(crate) fn based(
        store: Arc<MemoryStore>,
        log: Arc<Log>,
        base: u64,
        objects: Arc<Objects<Members>>,
    ) -> Self {
        Self {
            store,
            base,
            log,
            turn: Mutex::new(()),
            objects,
        }
    }

    /// Reads the store to its end into the worker objects. Its caller
    /// holds neither the door's turn nor a lock of the store: the holder's
    /// follows run one at a time by its own lock, and never inside a
    /// notification of the store ([`Objects::follow`]).
    fn followed(&self) {
        self.objects.follow(&*self.store);
    }

    /// The door's write of a registration's facts, in its turn: decided
    /// over the worker objects, which the store follows
    /// ([`Objects::upsert`]). `None` where another writer moved the row
    /// under the turn.
    fn lay_row(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
        self.objects.upsert(&self.store, facts, maker)
    }

    /// The door's write of a patch, as [`Self::lay_row`] writes a
    /// registration.
    fn patch_row(
        &self,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        self.objects.patch(&self.store, id, patch, revision)
    }

    /// The door's removal of a row, as [`Self::lay_row`] writes a
    /// registration.
    fn take_row(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.objects.remove(&self.store, id, revision)
    }

    /// The door's read of the row under `id`, in its turn: the record read
    /// back from the worker object that stands, with the status the table
    /// holds ([`Objects::record`]).
    fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.objects.record(&self.store, id)
    }

    /// The door's read of the row under `url`, as [`Self::row`] reads one.
    fn row_under(&self, url: &str) -> Option<WorkerRow> {
        self.objects.record_under(&self.store, url)
    }

    /// The door's read of the rows that name `engine`, as [`Self::row`]
    /// reads one.
    fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
        self.objects.records_of(&self.store, engine)
    }

    /// The log's place: read after a call's write, that write's own or one
    /// past it.
    fn place(&self) -> Position {
        self.log.head()
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
        self.check_term(admitted)?;
        Ok(self.life(admitted))
    }

    fn check_term(&self, admitted: u64) -> Result<(), Status> {
        if self.log.leads(admitted).is_err() {
            return Err(Status::unavailable(LEADS_NO_TERM));
        }
        Ok(())
    }

    fn life(&self, term: u64) -> u64 {
        life(term, self.base, self.store.epoch())
    }

    /// One call of the door, admitted from `by`: its turn, the term
    /// `leading` (the service's read of the fence) tells once the turn is
    /// held, its write under that term, and the audit line of a refusal.
    /// Every call leaves a line with its caller (design 5.2): a write's is
    /// [`told`]'s, a refusal's says its code and its words, which hold no
    /// URL.
    ///
    /// The replica's worker objects hold the write when the call answers.
    /// A row's write was decided over them in the turn. Behind a write
    /// that answered, with the turn let go and no lock of the store held,
    /// the door follows the store into them for what the table still
    /// leads, a status ([`Self::followed`]). A decision reads those
    /// objects, so none made behind an acknowledged registration, drain or
    /// removal reads the fleet as it stood before the call. A refused call
    /// follows nothing.
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
        if answer.is_ok() {
            self.followed();
        }
        if let Err(refusal) = &answer {
            audit_refusal(call, by, refusal.code(), refusal.message(), None);
        }
        answer
    }

    /// `RegisterWorker`, admitted from `by` under the term `leading` reads.
    ///
    /// `ADD` (and no mode) wants the URL absent; `REPLACE` wants it standing
    /// and makes its next incarnation; `UPSERT` registers an absent URL and
    /// lays the registration over a standing row: equal facts write nothing
    /// and answer the row unmoved, a change of role, wire or models is a new
    /// incarnation, any other change is made in place. Over a draining row
    /// an `UPSERT` is the URL's next incarnation, whatever its facts.
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
        self.call(REGISTER, by, leading, |term| {
            self.upsert(by, &Registrant::Control, term, request)
        })
    }

    /// The write of [`Self::register`], in the door's turn, from `maker`.
    fn upsert(
        &self,
        by: &Principal,
        maker: &Registrant,
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
        if spec.dp_size > 1 {
            return self.upsert_ranks(by, maker, admitted, request, spec, mode);
        }
        // The conversion reads the standing row once the URL holds its own
        // rules: a URL past its bound is never looked up. What a new row
        // names is no rule of a call that makes none: a REPLACE of a URL
        // that does not stand is not found, whatever it leaves unsaid. The
        // conversion tells a new row's names after every field's own rule,
        // so a field past its bound is refused as itself here too.
        let read = worker_wire::registration(spec, |url| self.row_under(url));
        let (mut facts, standing) = match read {
            Err(unnamed) if mode == Mode::Replace && worker_wire::of_a_new_row(&unnamed) => {
                return Err(absent());
            }
            read => read.map_err(invalid)?,
        };
        // An engine that stands as ranks takes a registration that states
        // its width: a plain row beside its ranks is the engine twice, and
        // one rank laid alone would offer what the others do not. The URL
        // may be the engine's or a rank's own. A caller that did not make
        // the ranks is told so first, as the wide call tells it: an add
        // asks no maker.
        {
            let engine = standing
                .as_ref()
                .map_or(spec.url.as_str(), |row| row.facts.engine.as_str());
            let mut ranks = self.rows_of(engine);
            ranks.retain(|row| row.facts.key != engine);
            if !ranks.is_empty() {
                if mode != Mode::Add {
                    for row in &ranks {
                        made_by(row, maker)?;
                    }
                }
                return Err(Status::failed_precondition(AS_RANKS));
            }
        }
        if request.wait_ready && facts.health.probes {
            return Err(invalid(NO_HANDSHAKE));
        }
        let old = match (mode, standing) {
            // An add changes no standing row, so it asks no maker: its
            // refusal is the same whoever made the row.
            (Mode::Add, Some(row)) => return Err(stands(&row)),
            (Mode::Replace, None) => return Err(absent()),
            (_, None) => {
                facts.incarnation = self.next_life(admitted)?;
                None
            }
            (mode, Some(row)) => {
                // Ahead of the test of equal facts: a caller that did not
                // make the row is told so, and never that it stands unmoved.
                made_by(&row, maker)?;
                let laid = if mode == Mode::Replace {
                    WorkerFactsChange::Another
                } else {
                    row.compare_new_facts(&facts)
                };
                match laid {
                    WorkerFactsChange::Unmoved => {
                        let epoch = self.store.epoch();
                        let at = self.place();
                        let unmoved = worker_wire::relabelled(&row);
                        return Ok(told(REGISTER, by, unmoved, false, epoch, at));
                    }
                    WorkerFactsChange::InPlace => {}
                    WorkerFactsChange::Another => facts.incarnation = self.next_life(admitted)?,
                }
                Some(row)
            }
        };
        // The row's bound holds with its maker: the facts' own was told
        // where they were read.
        worker_wire::room(&facts, &facts.labels, maker, "spec").map_err(invalid)?;
        let wrote = self.lay_row(facts, maker.clone());
        let id = wrote.ok_or_else(moved)?.id();
        // The row, the epoch and the place as the registration's write left
        // them: the answer of a call that does not wait.
        let (mut now, mut epoch, mut at) = (self.read(id)?, self.store.epoch(), self.place());
        // A fresh row is pending: any other status the write left came over
        // from the old incarnation. Read before the connect signal moves it.
        let carried = now.status != WorkerStatus::Pending;
        if !now.facts.health.probes {
            self.store.connected(id, now.revision, &|| true);
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
        self.call(PATCH, by, leading, |_| {
            self.relabel(by, &Registrant::Control, request)
        })
    }

    /// The write of [`Self::patch`], in the door's turn, from `maker`.
    fn relabel(
        &self,
        by: &Principal,
        maker: &Registrant,
        request: &proto::PatchWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let standing = self.guarded(maker, request.worker_id, request.revision)?;
        let (now, wrote) = match worker_wire::patch(request, &standing.facts).map_err(invalid)? {
            None => (standing, false),
            Some(patch) => {
                if let Some(labels) = &patch.labels {
                    worker_wire::room(&standing.facts, labels, &standing.maker, "labels")
                        .map_err(invalid)?;
                }
                self.patch_row(standing.id, patch, standing.revision)
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
        self.call(DEREGISTER, by, leading, |_| {
            self.remove(by, &Registrant::Control, request)
        })
    }

    /// The write of [`Self::deregister`], in the door's turn, from `maker`.
    /// A drain is no exception: it is a write of the row.
    fn remove(
        &self,
        by: &Principal,
        maker: &Registrant,
        request: &proto::DeregisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let standing = self.guarded(maker, request.worker_id, request.revision)?;
        let (event, wrote) = if !request.drain {
            self.take_row(standing.id, standing.revision)
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

    /// The row under `id` when `maker` made it and `revision` is the one it
    /// stands at. The maker is tested first: a caller that did not make the
    /// row is told so whatever revision it read.
    fn guarded(&self, maker: &Registrant, id: u64, revision: u64) -> Result<WorkerRow, Status> {
        let row = self
            .row(WorkerId(id))
            .ok_or_else(|| Status::not_found("no worker has the id"))?;
        made_by(&row, maker)?;
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
        self.row(id).ok_or_else(moved)
    }
}

/// A `wait_ready` where no handshake ends the wait.
const NO_HANDSHAKE: Refused = Refused {
    path: "wait_ready",
    why: "no handshake runs in this build: only a worker whose checks are off is ready at once",
};

/// An add of a URL a row stands under, whoever made the row.
fn stands(row: &WorkerRow) -> Status {
    Status::already_exists(format!(
        "the URL is registered as worker {:016x} at revision {}",
        row.id.0, row.revision
    ))
}

/// A registration of one row for an engine that stands as ranks.
const AS_RANKS: &str =
    "the worker stands as ranks: a registration states its width, or its ranks are removed first";

/// The host door's refusal as a call's answer and an audit line's words.
fn refused(refusal: &MembershipWriteRefused) -> Status {
    match refusal {
        MembershipWriteRefused::NotRegistrant(why) => Status::permission_denied(why.as_str()),
        MembershipWriteRefused::Past(past) => {
            Status::invalid_argument(format!("{}: {}", past.fact, past.why))
        }
        MembershipWriteRefused::Absent => absent(),
        MembershipWriteRefused::NotLeading(why) => Status::unavailable(why.as_str()),
        MembershipWriteRefused::Moved => moved(),
        MembershipWriteRefused::Stands => {
            Status::already_exists("a worker is registered under the URL")
        }
        MembershipWriteRefused::Draining => Status::failed_precondition(DRAINING),
        MembershipWriteRefused::OutOfScope => {
            Status::permission_denied(MembershipWriteRefused::OUT_OF_SCOPE)
        }
        MembershipWriteRefused::NotLabelsAlone => {
            Status::permission_denied(MembershipWriteRefused::NOT_LABELS_ALONE)
        }
    }
}

/// What a write made of its row, as its audit line names it, and the row.
fn made(wrote: MembershipWriteOutcome) -> (&'static str, WorkerId) {
    match wrote {
        MembershipWriteOutcome::Registered(id) => ("registered", id),
        MembershipWriteOutcome::InPlace(id) => ("relabelled", id),
        MembershipWriteOutcome::Another { new, .. } => ("replaced", new),
        MembershipWriteOutcome::Unmoved(id) => ("nothing", id),
        MembershipWriteOutcome::Draining(id) => ("status", id),
        MembershipWriteOutcome::Removed(id) => ("removed", id),
    }
}

/// The turn of a control call that states several ranks: the door's term
/// and its lives, each row as its write left it, for the call's answer,
/// and each write's audit line as it lands. A call refused behind its
/// first write leaves the lines of the writes that stand.
struct CallTurn<'a> {
    door: &'a MembershipCalls,
    by: &'a Principal,
    term: u64,
    laid: RefCell<Vec<WorkerRow>>,
}

impl Turn for CallTurn<'_> {
    fn store(&self) -> &MemoryStore {
        &self.door.store
    }

    fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.door.row(id)
    }

    fn row_under(&self, url: &str) -> Option<WorkerRow> {
        self.door.row_under(url)
    }

    fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
        self.door.rows_of(engine)
    }

    fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
        self.door.lay_row(facts, maker)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.door.patch_row(id, patch, revision)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.door.take_row(id, revision)
    }

    fn check_term(&self) -> Result<(), MembershipWriteRefused> {
        self.door
            .check_term(self.term)
            .map_err(|status| MembershipWriteRefused::NotLeading(status.message().to_owned()))
    }

    fn life(&self) -> Result<u64, MembershipWriteRefused> {
        self.door
            .next_life(self.term)
            .map_err(|status| MembershipWriteRefused::NotLeading(status.message().to_owned()))
    }

    /// The call's own rules spoke where each rank's facts were read.
    fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
        Ok(())
    }

    fn laid(&self, row: &WorkerRow) {
        self.laid.borrow_mut().push(row.clone());
    }

    fn told(&self, wrote: MembershipWriteOutcome) {
        let (made, worker) = made(wrote);
        let (epoch, at) = (self.door.store.epoch(), self.door.place());
        audit_write(REGISTER, self.by, worker.0, made, epoch, at, None);
    }
}

impl MembershipCalls {
    /// The write of a registration that states more than one rank, in the
    /// door's turn, from `maker`. Each rank's row is stated by the call's
    /// own rules, a rank at a time. The group's rule, the rows that leave
    /// and the writes are the host door's one routine ([`lay_ready`]).
    ///
    /// The answer tells what the call made of each rank, in rank order, then
    /// each row of the engine that left: the event a watcher is told of that
    /// row, under the one place read after the call's last write. A call
    /// that waits answers each row after its connect signal.
    fn upsert_ranks(
        &self,
        by: &Principal,
        maker: &Registrant,
        admitted: u64,
        request: &proto::RegisterWorkerRequest,
        spec: &proto::WorkerSpec,
        mode: Mode,
    ) -> Result<proto::WorkerEvent, Status> {
        let width = worker_wire::width(spec).map_err(invalid)?;
        let mut ready: Vec<Ready> = Vec::with_capacity(usize::from(width));
        let mut stood = Vec::with_capacity(usize::from(width));
        for rank in 0..width {
            let url = rank_url(&spec.url, usize::from(rank));
            let read = worker_wire::rank_registration(spec, (&url, rank, width), |url| {
                self.row_under(url)
            });
            let (facts, standing) = match read {
                Err(unnamed) if mode == Mode::Replace && worker_wire::of_a_new_row(&unnamed) => {
                    return Err(absent());
                }
                read => read.map_err(invalid)?,
            };
            if request.wait_ready && facts.health.probes {
                return Err(invalid(NO_HANDSHAKE));
            }
            let laid = match (mode, &standing) {
                (Mode::Replace, None) => return Err(absent()),
                (_, None) => None,
                // An add asks its maker of no row: the group's rule says
                // which standing ranks are the same add's.
                (Mode::Add, Some(row)) => Some((row.id, row.compare_new_facts(&facts))),
                (mode, Some(row)) => {
                    made_by(row, maker)?;
                    let laid = if mode == Mode::Replace {
                        WorkerFactsChange::Another
                    } else {
                        row.compare_new_facts(&facts)
                    };
                    Some((row.id, laid))
                }
            };
            worker_wire::room(&facts, &facts.labels, maker, "spec").map_err(invalid)?;
            ready.push((facts, laid));
            stood.push(standing);
        }
        // The engine's rows as they stand, read before any leaves: the
        // answer tells the last of each row that left.
        let before = self.rows_of(&spec.url);
        let turn = CallTurn {
            door: self,
            by,
            term: admitted,
            laid: RefCell::new(Vec::new()),
        };
        let wrote = lay_ready(&turn, maker, mode == Mode::Add, ready).map_err(|refusal| {
            // An add refused where a rank it states stands names that row,
            // as a call of one row does.
            match (&refusal, stood.iter().flatten().next()) {
                (MembershipWriteRefused::Stands, Some(row)) => stands(row),
                _ => refused(&refusal),
            }
        })?;
        let (epoch, at) = (self.store.epoch(), self.place());
        let mut laid = turn.laid.into_inner().into_iter();
        let mut rows = Vec::with_capacity(wrote.len());
        for (rank, wrote) in wrote.into_iter().enumerate() {
            let old = stood.get(rank).and_then(Option::as_ref);
            let event = match wrote {
                MembershipWriteOutcome::Registered(_) => {
                    let row = self.answered(laid.next(), request)?;
                    worker_wire::registered(&row)
                }
                MembershipWriteOutcome::InPlace(_) => {
                    let row = self.answered(laid.next(), request)?;
                    worker_wire::relabelled(&row)
                }
                MembershipWriteOutcome::Another { .. } => {
                    // A fresh row is pending: any other status came over
                    // from the old life, read before a connect signal.
                    let written = laid.next().ok_or_else(moved)?;
                    let carried = written.status != WorkerStatus::Pending;
                    let row = self.answered(Some(written), request)?;
                    worker_wire::replaced(old.ok_or_else(moved)?, &row, carried)
                }
                MembershipWriteOutcome::Unmoved(_) => {
                    worker_wire::relabelled(old.ok_or_else(moved)?)
                }
                MembershipWriteOutcome::Removed(id) => {
                    let last = before.iter().find(|row| row.id == id);
                    worker_wire::removed(last.ok_or_else(moved)?)
                }
                MembershipWriteOutcome::Draining(_) => return Err(moved()),
            };
            rows.extend(worker_wire::rank_row(event));
        }
        Ok(worker_wire::event(
            Event::Ranks(proto::RankRows { rows }),
            at.seq,
            epoch,
            at.term,
        ))
    }

    /// A row as its write left it, or the row after its connect signal
    /// where the call waits.
    fn answered(
        &self,
        written: Option<WorkerRow>,
        request: &proto::RegisterWorkerRequest,
    ) -> Result<WorkerRow, Status> {
        let written = written.ok_or_else(moved)?;
        if request.wait_ready {
            self.read(written.id)
        } else {
            Ok(written)
        }
    }
}

/// A controller replica's own membership writer: the same calls for the
/// program's own writers, sharing the turn of its control calls.
#[cfg_attr(not(test), expect(dead_code, reason = "the host owns the door"))]
pub(crate) struct ReplicaMembershipWriter {
    door: Arc<MembershipCalls>,
    by: Principal,
    leading: Box<dyn Fn() -> Result<u64, Status> + Send + Sync>,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the host supplies the principal and fence")
)]
impl ReplicaMembershipWriter {
    const REGISTRATION: &str = "worker registration";

    /// Open the host's door with the principal its calls name and the
    /// replica's read of the leading term.
    pub(crate) fn new(
        door: Arc<MembershipCalls>,
        by: Principal,
        leading: impl Fn() -> Result<u64, Status> + Send + Sync + 'static,
    ) -> Self {
        Self {
            door,
            by,
            leading: Box::new(leading),
        }
    }

    fn call<T>(
        &self,
        call: &'static str,
        asker: Option<&str>,
        write: impl FnOnce(u64) -> Result<T, MembershipWriteRefused>,
    ) -> Result<T, MembershipWriteRefused> {
        let turn = self.door.turn.lock();
        let term = match (self.leading)() {
            Ok(term) => term,
            Err(status) => {
                drop(turn);
                audit_refusal(call, &self.by, status.code(), status.message(), asker);
                return Err(MembershipWriteRefused::NotLeading(
                    status.message().to_owned(),
                ));
            }
        };
        match write(term) {
            Ok(answer) => {
                // The worker objects hold the write when the call answers,
                // as behind a control call's ([`MembershipCalls::call`]).
                drop(turn);
                self.door.followed();
                Ok(answer)
            }
            Err(refusal) => {
                drop(turn);
                let status = refused(&refusal);
                audit_refusal(call, &self.by, status.code(), status.message(), asker);
                Err(refusal)
            }
        }
    }

    /// The registration's write over rows of stated facts, in the door's
    /// turn and under the term it reads there.
    fn rows(
        &self,
        maker: &Registrant,
        expect: Precondition,
        rows: &[Stated<'_>],
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        self.call(Self::REGISTRATION, None, |term| {
            lay(&ReplicaTurn { host: self, term }, maker, expect, rows)
        })
    }

    fn audit(&self, call: &'static str, wrote: MembershipWriteOutcome, asker: Option<&str>) {
        let (made, worker) = made(wrote);
        audit_write(
            call,
            &self.by,
            worker.0,
            made,
            self.door.store.epoch(),
            self.door.place(),
            asker,
        );
    }

    fn removal(
        &self,
        call: &'static str,
        by: Remover<'_>,
        write: impl FnOnce(u64) -> Result<MembershipWriteOutcome, MembershipWriteRefused>,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        let asker = Some(match by {
            Remover::Registrant(Registrant::Control) => "control call",
            Remover::Registrant(Registrant::Provider(provider)) => provider.provider(),
            Remover::Provider(name) => name,
            Remover::Health(_) => "health",
        });
        self.call(call, asker, |term| {
            write(term).inspect(|wrote| self.audit(call, *wrote, asker))
        })
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the host reads the term for each registration")
)]
struct ReplicaTurn<'a> {
    host: &'a ReplicaMembershipWriter,
    term: u64,
}

impl Turn for ReplicaTurn<'_> {
    fn store(&self) -> &MemoryStore {
        &self.host.door.store
    }

    fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.host.door.row(id)
    }

    fn row_under(&self, url: &str) -> Option<WorkerRow> {
        self.host.door.row_under(url)
    }

    fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
        self.host.door.rows_of(engine)
    }

    fn upsert(&self, facts: WorkerFacts, maker: Registrant) -> Option<MembershipChange> {
        self.host.door.lay_row(facts, maker)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.host.door.patch_row(id, patch, revision)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.host.door.take_row(id, revision)
    }

    fn check_term(&self) -> Result<(), MembershipWriteRefused> {
        self.host
            .door
            .check_term(self.term)
            .map_err(|status| MembershipWriteRefused::NotLeading(status.message().to_owned()))
    }

    fn life(&self) -> Result<u64, MembershipWriteRefused> {
        self.host
            .door
            .next_life(self.term)
            .map_err(|status| MembershipWriteRefused::NotLeading(status.message().to_owned()))
    }

    fn room(&self, facts: &WorkerFacts, maker: &Registrant) -> Result<(), MembershipWriteRefused> {
        worker_wire::wire_reach(facts.wire_modes, facts.reachable_from, "row.wire")
            .and_then(|()| worker_wire::carried(facts))
            .and_then(|()| worker_wire::room(facts, &facts.labels, maker, "row"))
            .map_err(|refusal| {
                MembershipWriteRefused::Past(FactPastBound {
                    fact: refusal.path,
                    why: refusal.why,
                })
            })
    }

    fn told(&self, wrote: MembershipWriteOutcome) {
        self.host
            .audit(ReplicaMembershipWriter::REGISTRATION, wrote, None);
    }
}

impl MembershipWriter for ReplicaMembershipWriter {
    fn found(
        &self,
        maker: &Registrant,
        expect: Precondition,
        url: &str,
        state: &dyn Fn(Option<&WorkerFacts>) -> WorkerFacts,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        self.call(Self::REGISTRATION, None, |term| {
            lay_one(&ReplicaTurn { host: self, term }, maker, expect, url, state)
        })
    }

    // The controller builds each object from its row: the facts alone.
    fn found_rows(
        &self,
        maker: &Registrant,
        expect: Precondition,
        built: &[Arc<dyn Worker>],
        over: Option<Over<'_>>,
    ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
        stating(built, over, stated, |rows| self.rows(maker, expect, rows))
    }

    fn drained(
        &self,
        by: Remover<'_>,
        url: &str,
        seen: RowRevision,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        self.removal("worker drain", by, |term| {
            drain(&ReplicaTurn { host: self, term }, by, url, seen)
        })
    }

    fn gone(
        &self,
        by: Remover<'_>,
        url: &str,
        seen: RowRevision,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        self.removal("worker removal", by, |term| {
            take(&ReplicaTurn { host: self, term }, by, url, seen)
        })
    }

    // The controller builds each object from its row: the refresh writes
    // the row alone. Its log carries no descriptor, so the room test
    // refuses a refresh that states one.
    fn refreshed(
        &self,
        seen: RowRevision,
        built: &Arc<dyn Worker>,
        _: &Arc<dyn Worker>,
    ) -> Result<MembershipWriteOutcome, MembershipWriteRefused> {
        const CALL: &str = "worker label refresh";
        self.call(CALL, None, |term| {
            refresh(
                &ReplicaTurn { host: self, term },
                seen,
                built.as_ref(),
                stated,
            )
            .inspect(|wrote| self.audit(CALL, *wrote, None))
        })
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
        Event::Snapshot(_) | Event::Needs(_) | Event::SnapshotRequired(_) | Event::Ranks(_) => {
            ("", 0)
        }
    };
    let made = if wrote { made } else { "nothing" };
    audit_write(call, by, worker, made, membership_epoch, at, None);
    worker_wire::event(event, at.seq, membership_epoch, at.term)
}

fn audit_write(
    call: &'static str,
    by: &Principal,
    worker: u64,
    made: &str,
    membership_epoch: u64,
    at: Position,
    asker: Option<&str>,
) {
    info!(
        call,
        by = %by.subject,
        worker = %format_args!("{worker:016x}"),
        made,
        membership_epoch,
        seq = at.seq,
        term = at.term,
        asker,
        "membership write"
    );
}

fn audit_refusal(call: &'static str, by: &Principal, code: Code, why: &str, asker: Option<&str>) {
    info!(call, by = %by.subject, ?code, why, asker, "membership call refused");
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Barrier,
        },
        thread,
    };

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        settings::UnhealthyRemovalScope,
    };
    use tonic::{Code, Status};

    use super::{MembershipCalls, Position, ReplicaMembershipWriter};
    use crate::{
        engine::view::{Role, WireModes, WorkerId},
        service::{
            controller::{
                decision_slots::tests::holder,
                election::LeasePlace,
                principal::{tests::told, Principal},
                replication::{journaled, journaled_from},
            },
            proto::{self, register_worker_request::Mode, worker_event::Event},
            replication,
        },
        store::{
            self, identity, GatewayId, MemoryStore, Reachability, Registrant, WorkerFacts,
            WorkerRow, WorkerStatus, WorkersStore,
        },
        worker_host::{
            check_event_signal, check_expectations, check_guard, check_health_scope, check_maker,
            check_one_row, check_provider, check_rank_rows, check_refresh, FindOne,
            MembershipWriteOutcome, MembershipWriteRefused, MembershipWriter, Precondition,
            Remover, RowRevision, Rows, Stated, StoreDoor,
        },
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
    fn door() -> (Arc<MemoryStore>, MembershipCalls) {
        door_in(TERM)
    }

    /// The same under another term. The door decides over a holder of its
    /// own, built as the controller builds one.
    fn door_in(term: u64) -> (Arc<MemoryStore>, MembershipCalls) {
        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        store::LeasesStore::set_term(store.as_ref(), term);
        let objects = holder(&store);
        (
            Arc::clone(&store),
            MembershipCalls::based(store, feed.log, 0, objects),
        )
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

    /// The replica's worker objects hold a call's write when the call
    /// answers. No task follows the store here, and a decision follows
    /// nothing: each is the in-process front's over the list the holder
    /// lends as the door's last call left it. A worker registered through
    /// the door is named by a decision behind the call. One whose
    /// deregistration was acknowledged is named by none behind it, and
    /// neither is one whose drain was acknowledged. The replica's own
    /// writer shares the door's turn and its follow: a worker it drains is
    /// named by none behind its answer.
    #[tokio::test]
    async fn a_decision_behind_an_acknowledged_call_reads_the_fleet_the_call_left() {
        use std::collections::BTreeSet;

        use crate::service::controller::decision_slots::tests::{frame, lent_pick, preset};

        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let objects = holder(&store);
        let door = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
        let door = Arc::new(door);
        store
            .apply(preset("round_robin", 1))
            .expect("above the unbound start");
        // The workers twelve decisions name, with no follow of the case's.
        let named = || -> BTreeSet<WorkerId> {
            (1..=12)
                .filter_map(|seed| lent_pick(&store, &objects, frame(seed, u64::from(seed))))
                .collect()
        };
        let ids: Vec<WorkerId> = ["w0", "w1", "w2"]
            .iter()
            .map(|name| {
                let asked = ask(unchecked(&format!("grpc://{name}:8000")), Mode::Add);
                let told = door
                    .register(&operator(), under(TERM), &asked)
                    .expect("the door takes the worker");
                WorkerId(row_of(&told).id)
            })
            .collect();
        assert_eq!(named(), ids.iter().copied().collect());
        let leave = |id: WorkerId, drain: bool| {
            let row = WorkersStore::get(&*store, id).expect("the row stands");
            let request = proto::DeregisterWorkerRequest {
                worker_id: id.0,
                revision: row.revision,
                drain,
            };
            door.deregister(&operator(), under(TERM), &request)
                .expect("the door acknowledges");
        };
        // An acknowledged deregistration: no decision behind it names the
        // worker.
        leave(ids[0], false);
        assert_eq!(named(), BTreeSet::from([ids[1], ids[2]]));
        // And an acknowledged drain.
        leave(ids[1], true);
        assert_eq!(named(), BTreeSet::from([ids[2]]));
        // The last worker is drained through the replica's own writer,
        // over the same door: no decision behind its answer names a worker.
        // A drain is the table's write still, so the writer's follow is
        // what takes it to the object.
        let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
        let last = WorkersStore::get(&*store, ids[2]).expect("the row stands");
        let by = Remover::Registrant(&last.maker);
        assert_eq!(
            hosted.drained(by, "grpc://w2:8000", standing(&last)),
            Ok(MembershipWriteOutcome::Draining(ids[2]))
        );
        assert_eq!(named(), BTreeSet::new());
    }

    /// A call's write of a row builds the row's worker object where it is
    /// written, and what the table still leads reaches the object behind
    /// the write: the registry tells a registration, then the status the
    /// connect signal moved, as the log tells the two. So it is for a call
    /// of one row, for each rank of a wide call, and for a row the
    /// replica's own writer lays. A door that wrote the table alone would
    /// build each object behind its connect signal, ready from the start,
    /// and the registry would tell no status.
    #[test]
    fn a_call_builds_a_rows_object_at_its_write_and_its_status_follows() {
        use smg_worker::registry::event::WorkerEvent;

        use crate::store::HealthPolicy;

        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let objects = holder(&store);
        let mut events = objects.registry().subscribe_events();
        let door = Arc::new(MembershipCalls::based(store, feed.log, 0, objects));
        // What the registry told since the last read, in its order.
        let mut told = move || -> Vec<&'static str> {
            let mut told = Vec::new();
            while let Ok(event) = events.try_recv() {
                told.push(match event {
                    WorkerEvent::Registered { .. } => "registered",
                    WorkerEvent::Replaced { .. } => "replaced",
                    WorkerEvent::Removed { .. } => "removed",
                    WorkerEvent::StatusChanged { .. } => "status",
                });
            }
            told
        };
        door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("one row");
        assert_eq!(told(), ["registered", "status"]);
        let ranks = proto::WorkerSpec {
            dp_size: 2,
            ..unchecked("grpc://wide:8000")
        };
        door.register(&operator(), under(TERM), &ask(ranks, Mode::Add))
            .expect("two ranks");
        // Each write reads the store to its end first: the first rank's
        // status reaches its object ahead of the second rank's write.
        assert_eq!(told(), ["registered", "status", "registered", "status"]);
        let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
        let own = "grpc://own:8000";
        let wrote = hosted.found_one(&found("pods/a"), own, &|_| WorkerFacts {
            key: own.to_owned(),
            engine: own.to_owned(),
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..found_facts()
        });
        assert!(
            matches!(wrote, Ok(MembershipWriteOutcome::Registered(_))),
            "{wrote:?}"
        );
        assert_eq!(told(), ["registered", "status"]);
    }

    /// A door's patch and removal are decided over the worker objects, as
    /// its registration is: each runs the store's one body, which the
    /// store's own patch and removal never run, and a removal's object has
    /// left the registry when the store hears of it. So it is for a control
    /// call and for a removal by the replica's own writer.
    #[test]
    fn a_doors_patch_and_removal_are_decided_over_the_worker_objects() {
        use std::{cell::Cell, rc::Rc};

        use crate::store::memory::seam;

        let (store, feed) = journaled_from(&LeasePlace::Process, 0);
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let objects = holder(&store);
        let calls = MembershipCalls::based(Arc::clone(&store), feed.log, 0, Arc::clone(&objects));
        let door = Arc::new(calls);
        // Whether `write` ran the one body, with the object of the row
        // `gone` out of the registry by then.
        let decided = |gone: Option<WorkerId>, write: &dyn Fn()| {
            let ran = Rc::new(Cell::new(false));
            let (told, objects) = (Rc::clone(&ran), Arc::clone(&objects));
            seam::on_release(move || {
                let stands = gone.and_then(|gone| objects.object(gone));
                assert_eq!(stands, None, "the object leaves first");
                told.set(true);
            });
            write();
            ran.get()
        };
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered");
        let row = store.by_key(URL).expect("the row");
        let patch = proto::PatchWorkerRequest {
            worker_id: row.id.0,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            remove_labels: Vec::new(),
            overload: None,
            revision: row.revision,
        };
        let patched = decided(None, &|| {
            door.patch(&operator(), under(TERM), &patch)
                .expect("patched");
        });
        assert!(patched, "a patch call");
        let row = store.by_key(URL).expect("the row");
        let leave = proto::DeregisterWorkerRequest {
            worker_id: row.id.0,
            revision: row.revision,
            drain: false,
        };
        let removed = decided(Some(row.id), &|| {
            door.deregister(&operator(), under(TERM), &leave)
                .expect("removed");
        });
        assert!(removed, "a removal call");
        let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
        let maker = found("pods/a");
        hosted
            .found_one(&maker, URL, &|_| found_facts())
            .expect("found");
        let made = store.by_key(URL).expect("the found row");
        let gone = decided(Some(made.id), &|| {
            hosted
                .gone(Remover::Registrant(&maker), URL, standing(&made))
                .expect("gone");
        });
        assert!(gone, "a removal by the replica's own writer");
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

    /// A call reads the row that stands off the worker objects, behind
    /// their holder's read of the store to its end. An add over a row that
    /// a writer outside the holder left in the table is refused as an add
    /// over any row that stands, and writes nothing.
    #[test]
    fn an_add_over_a_row_written_around_the_holder_is_refused() {
        let (store, door) = door();
        let facts = WorkerFacts {
            key: URL.to_owned(),
            incarnation: 1,
            wire_modes: WireModes::GRPC,
            models: vec!["model-a".to_owned()],
            ..WorkerFacts::default()
        };
        store.upsert_by(facts, Registrant::Control);
        let stood = (store.epoch(), store.by_key(URL));
        let refused = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect_err("the URL stands");
        assert_eq!(refused.code(), Code::AlreadyExists);
        assert_eq!((store.epoch(), store.by_key(URL)), stood);
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
            models: vec!["m".repeat(257)],
            ..over
        };
        let labelled = |over: proto::WorkerSpec| proto::WorkerSpec {
            labels: (0..129)
                .map(|n| (format!("label-{n}"), String::new()))
                .collect(),
            ..over
        };
        let models = "spec.models: a model id is at most 256 bytes";
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
    fn many_short_model_ids_register_and_a_list_past_the_row_size_is_refused() {
        let (store, door) = door();
        let heavy = proto::WorkerSpec {
            models: vec!["m".repeat(256); 64],
            ..spec(URL)
        };
        let status = door
            .register(&operator(), under(TERM), &ask(heavy, Mode::Add))
            .expect_err("past the row's size");
        assert_eq!(
            (status.code(), status.message()),
            (
                Code::InvalidArgument,
                "spec: a row is at most 16384 bytes on the wire"
            )
        );
        assert_eq!(store.epoch(), 0);

        let short = proto::WorkerSpec {
            models: (0..129).map(|n| n.to_string()).collect(),
            ..spec(URL)
        };
        let wrote = door
            .register(&operator(), under(TERM), &ask(short.clone(), Mode::Add))
            .expect("the ids fit");
        assert_eq!(row_of(&wrote).models, short.models);
        assert_eq!(
            store.by_key(URL).expect("the row").facts.models,
            short.models
        );
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
                dp_size: 65,
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

    /// A spec that states `width` ranks.
    fn wide(url: &str, width: u32) -> proto::WorkerSpec {
        proto::WorkerSpec {
            dp_size: width,
            ..spec(url)
        }
    }

    /// What an answer of several ranks made of each row.
    fn made_of(event: &proto::WorkerEvent) -> Vec<proto::rank_row::Made> {
        match event.event.clone().expect("an event") {
            Event::Ranks(ranks) => ranks
                .rows
                .into_iter()
                .map(|row| row.made.expect("a case"))
                .collect(),
            other => panic!("no ranks in {other:?}"),
        }
    }

    /// A registration that states several ranks writes one row a rank under
    /// the rank's URL and answers each as its write left it, at the one
    /// place of the call's last write. Stated again it writes nothing and
    /// answers each rank unmoved.
    #[test]
    fn a_registration_of_several_ranks_writes_a_row_a_rank_and_answers_each() {
        use proto::rank_row::Made;
        let (store, door) = door();
        let told = door
            .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
            .expect("registered");
        assert_eq!((told.membership_epoch, told.seq, told.term), (3, 4, TERM));
        let made = made_of(&told);
        assert_eq!(made.len(), 3);
        for (rank, made) in (0..3_u16).zip(&made) {
            let stored = store
                .by_key(&format!("{URL}@{rank}"))
                .expect("the rank's row");
            assert_eq!(*made, Made::Registered(super::worker_wire::row(&stored)));
            let facts = &stored.facts;
            assert_eq!(
                (facts.engine.as_str(), facts.dp_rank, facts.dp_size),
                (URL, rank, 3)
            );
            assert_eq!((stored.status, stored.revision), (WorkerStatus::Pending, 1));
        }

        let again = door
            .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Upsert))
            .expect("unmoved");
        assert_eq!(
            (again.membership_epoch, again.seq, store.epoch()),
            (3, 4, 3)
        );
        for (rank, made) in made_of(&again).iter().enumerate() {
            let stored = store
                .by_key(&format!("{URL}@{rank}"))
                .expect("the rank's row");
            assert_eq!(*made, Made::Relabelled(super::worker_wire::row(&stored)));
        }
    }

    /// The modes hold for the engine. An add is refused where its ranks
    /// stand or the engine stands in another form. A replacement wants every
    /// stated rank and makes each its next life. An upsert lays each rank,
    /// and the engine's rows it does not state leave with it.
    #[test]
    fn the_modes_of_a_registration_of_several_ranks_hold_for_the_engine() {
        use proto::rank_row::Made;
        let (store, door) = door();
        let register = |spec: proto::WorkerSpec, mode| {
            door.register(&operator(), under(TERM), &ask(spec, mode))
        };
        let rank = |rank: u16| store.by_key(&format!("{URL}@{rank}"));

        let none = register(wide(URL, 2), Mode::Replace).expect_err("no rank stands");
        assert_eq!(none.code(), Code::NotFound);
        register(spec(URL), Mode::Add).expect("the plain row");
        let other_form = register(wide(URL, 2), Mode::Add).expect_err("the plain row stands");
        assert_eq!(other_form.code(), Code::AlreadyExists);
        assert_eq!(store.epoch(), 1);

        // A worker that comes back ranked loses its plain row.
        let plain = store.by_key(URL).expect("the plain row");
        let told = register(wide(URL, 2), Mode::Upsert).expect("ranked");
        let made = made_of(&told);
        assert!(
            matches!(
                made.as_slice(),
                [Made::Registered(_), Made::Registered(_), Made::Removed(left)]
                    if left.worker_id == plain.id.0
            ),
            "{made:?}"
        );
        assert!(store.by_key(URL).is_none());
        assert_eq!((store.epoch(), told.membership_epoch), (4, 4));

        let stands = register(wide(URL, 2), Mode::Add).expect_err("a rank stands");
        assert_eq!(stands.code(), Code::AlreadyExists);
        let short = register(wide(URL, 3), Mode::Replace).expect_err("rank two is absent");
        assert_eq!(short.code(), Code::NotFound);
        assert_eq!(store.epoch(), 4);

        // A replacement is the next life of each rank.
        let old = [0, 1].map(|at| rank(at).expect("a rank"));
        let told = register(wide(URL, 2), Mode::Replace).expect("replaced");
        for (old, made) in old.iter().zip(made_of(&told)) {
            let Made::Replaced(replaced) = made else {
                panic!("{made:?}");
            };
            assert_eq!(replaced.old.map(|row| row.id), Some(old.id.0));
            assert_ne!(replaced.new.map(|row| row.id), Some(old.id.0));
            assert!(!replaced.state_carried);
        }

        // Wider, then narrower: the rank it no longer states leaves.
        let told = register(wide(URL, 3), Mode::Upsert).expect("wider");
        let made = made_of(&told);
        assert!(
            matches!(
                made.as_slice(),
                [
                    Made::Relabelled(_),
                    Made::Relabelled(_),
                    Made::Registered(_)
                ]
            ),
            "{made:?}"
        );
        let third = rank(2).expect("rank two");
        let told = register(wide(URL, 2), Mode::Upsert).expect("narrower");
        let made = made_of(&told);
        assert!(
            matches!(
                made.as_slice(),
                [Made::Relabelled(_), Made::Relabelled(_), Made::Removed(left)]
                    if left.worker_id == third.id.0
            ),
            "{made:?}"
        );
        assert!(rank(2).is_none());
        assert_eq!(rank(0).map(|row| row.facts.dp_size), Some(2));
    }

    /// A registration of one rank for an engine that stands as ranks is
    /// refused by name in every mode and writes nothing: a plain row beside
    /// its ranks would be the engine twice. The way off is a rank at a time.
    #[test]
    fn a_registration_of_one_rank_over_an_engine_that_stands_as_ranks_is_refused() {
        let (store, door) = door();
        door.register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
            .expect("ranks");
        // Under the engine's URL and under a rank's own: one rank laid
        // alone would offer what the engine's other rank does not.
        let rank = format!("{URL}@0");
        for mode in [Mode::Add, Mode::Replace, Mode::Upsert] {
            for url in [URL, rank.as_str()] {
                let refused = door
                    .register(&operator(), under(TERM), &ask(spec(url), mode))
                    .expect_err("the engine stands as ranks");
                assert_eq!(
                    (refused.code(), refused.message()),
                    (Code::FailedPrecondition, super::AS_RANKS),
                    "{mode:?} {url}"
                );
            }
        }
        assert_eq!(store.epoch(), 2);

        for at in 0..2 {
            let row = store.by_key(&format!("{URL}@{at}")).expect("a rank");
            let leave = proto::DeregisterWorkerRequest {
                worker_id: row.id.0,
                revision: row.revision,
                drain: false,
            };
            door.deregister(&operator(), under(TERM), &leave)
                .expect("the rank leaves");
        }
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("the plain row");
        assert!(store.by_key(URL).is_some());
    }

    /// A one-rank replacement or upsert over ranks another maker laid is
    /// told who made them, ahead of the engine's form, as the wide call
    /// tells it. An add asks no maker.
    #[test]
    fn a_registration_of_one_rank_over_another_makers_ranks_is_told_the_maker_first() {
        let (store, door) = door();
        door.upsert(
            &operator(),
            &found(URL),
            TERM,
            &ask(wide(URL, 2), Mode::Add),
        )
        .expect("a provider's ranks");
        let rank = format!("{URL}@0");
        let register = |spec: proto::WorkerSpec, mode| {
            door.register(&operator(), under(TERM), &ask(spec, mode))
                .expect_err("ranks of another maker")
        };
        for mode in [Mode::Replace, Mode::Upsert] {
            for asked in [spec(URL), spec(&rank), wide(URL, 2)] {
                let url = asked.url.clone();
                let refused = register(asked, mode);
                assert_eq!(refused.code(), Code::PermissionDenied, "{mode:?} {url}");
            }
        }
        for url in [URL, rank.as_str()] {
            let refused = register(spec(url), Mode::Add);
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::FailedPrecondition, super::AS_RANKS),
                "{url}"
            );
        }
        assert_eq!(store.epoch(), 2);
    }

    /// The door's log as its store's journal, which leaves its term as a
    /// membership write lands: a lead lost inside one call.
    struct LosesTheLead {
        log: Arc<replication::Log>,
        /// The membership writes the lead still holds for. The write that
        /// finds none left loses it, once.
        holds: AtomicUsize,
    }

    impl store::Journal for LosesTheLead {
        fn member(
            &self,
            at: Option<Position>,
            epoch: u64,
            change: store::MembershipChange,
            row: Option<&WorkerRow>,
            left: Option<&WorkerRow>,
        ) {
            self.log.member(at, epoch, change, row, left);
            match self.holds.load(Ordering::Relaxed) {
                0 => {
                    let next = Position {
                        seq: 40,
                        term: TERM + 1,
                    };
                    assert!(self.log.restart(next).is_ok(), "a place the log takes");
                    self.holds.store(usize::MAX, Ordering::Relaxed);
                }
                more => self.holds.store(more - 1, Ordering::Relaxed),
            }
        }

        fn generation(&self, at: Option<Position>, generation: u64) {
            self.log.generation(at, generation);
        }

        fn term_rose(&self, term: u64) {
            self.log.term_rose(term);
        }

        fn term_followed(&self, at: Position) {
            self.log.term_followed(at);
        }

        fn head(&self) -> Position {
            self.log.head()
        }

        fn followed(&self) -> Option<u64> {
            self.log.followed()
        }

        fn released(&self) {
            self.log.released();
        }

        fn restart(&self, at: Position) -> bool {
            self.log.restart(at).is_ok()
        }
    }

    /// A wide call that loses its lead between two ranks is refused. The
    /// rank written before the refusal stands, and its write leaves its
    /// audit line ahead of the refusal's.
    #[test]
    fn a_term_lost_between_two_ranks_leaves_the_first_rank_and_its_audit_line() {
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let (_, feed) = journaled_from(&LeasePlace::Process, 0);
        let journal = Arc::new(LosesTheLead {
            log: Arc::clone(&feed.log),
            holds: AtomicUsize::new(0),
        });
        let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let log = Arc::clone(&feed.log);
        let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));

        let refused = door
            .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
            .expect_err("the lead was lost at the second rank");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::LEADS_NO_TERM)
        );
        assert!(store.by_key(&format!("{URL}@0")).is_some());
        assert!(store.by_key(&format!("{URL}@1")).is_none());

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let lines: Vec<&str> = logged.lines().collect();
        let [written, refusal] = lines.as_slice() else {
            panic!("one write's line and the refusal's: {logged}");
        };
        assert!(
            written.contains("membership write") && written.contains("made=\"registered\""),
            "{written}"
        );
        assert!(refusal.contains("membership call refused"), "{refusal}");
        assert!(!logged.contains(URL), "{logged}");
    }

    /// A door whose journal loses the lead when told, with `width` ranks of
    /// [`URL`] registered under a lead that holds.
    fn losing(width: u32) -> (Arc<MemoryStore>, MembershipCalls, Arc<LosesTheLead>) {
        let (_, feed) = journaled_from(&LeasePlace::Process, 0);
        let journal = Arc::new(LosesTheLead {
            log: Arc::clone(&feed.log),
            holds: AtomicUsize::new(usize::MAX),
        });
        let handed = Arc::clone(&journal);
        let store = Arc::new(MemoryStore::journaled(handed).with_decision_slots(1));
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let log = Arc::clone(&feed.log);
        let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
        door.register(&operator(), under(TERM), &ask(wide(URL, width), Mode::Add))
            .expect("the ranks registered under a held lead");
        (store, door, journal)
    }

    /// A lead lost behind a rank's write in place refuses the next rank's.
    /// The term is asked ahead of every write, whatever it makes of its
    /// row.
    #[test]
    fn a_term_lost_behind_a_write_in_place_stops_the_next_ranks_write() {
        let (store, door, journal) = losing(2);
        journal.holds.store(0, Ordering::Relaxed);
        let relabelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..wide(URL, 2)
        };
        let refused = door
            .register(&operator(), under(TERM), &ask(relabelled, Mode::Upsert))
            .expect_err("the lead was lost at the second rank");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::LEADS_NO_TERM)
        );
        let labels = |rank: u32| {
            let row = store.by_key(&format!("{URL}@{rank}")).expect("the rank");
            row.facts.labels.len()
        };
        assert_eq!((labels(0), labels(1)), (1, 0));
    }

    /// A lead lost behind the last rank that stays refuses the removal of
    /// the rank a narrower registration no longer states.
    #[test]
    fn a_term_lost_ahead_of_a_removal_leaves_the_rank_that_would_leave() {
        let (store, door, journal) = losing(3);
        journal.holds.store(1, Ordering::Relaxed);
        let refused = door
            .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Upsert))
            .expect_err("the lead was lost ahead of the removal");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::LEADS_NO_TERM)
        );
        let width = |rank: u32| {
            store
                .by_key(&format!("{URL}@{rank}"))
                .map(|row| row.facts.dp_size)
        };
        assert_eq!((width(0), width(1), width(2)), (Some(2), Some(2), Some(3)));
    }

    /// A door whose lead was lost behind the first of an add's three ranks
    /// and came back under a later term: the next leader's door, over the
    /// row the first attempt left.
    fn led_again_behind_a_lost_add() -> (Arc<MemoryStore>, MembershipCalls) {
        let (_, feed) = journaled_from(&LeasePlace::Process, 0);
        let journal = Arc::new(LosesTheLead {
            log: Arc::clone(&feed.log),
            holds: AtomicUsize::new(0),
        });
        let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
        store::LeasesStore::set_term(store.as_ref(), TERM);
        let log = Arc::clone(&feed.log);
        let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
        let lost = door
            .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
            .expect_err("the lead was lost at the second rank");
        assert_eq!(
            (lost.code(), lost.message()),
            (Code::Unavailable, super::LEADS_NO_TERM)
        );
        assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
        store::LeasesStore::set_term(store.as_ref(), TERM + 2);
        (store, door)
    }

    /// An add that lost its lead behind its first rank is the same add at
    /// the next leader's door. The rank it wrote answers unmoved, the
    /// others are registered, and each rank has its one row. Called once
    /// more, the engine stands whole: the add is refused by the first
    /// rank's row.
    #[test]
    fn an_add_called_again_behind_a_lost_lead_lays_the_ranks_it_left() {
        use proto::rank_row::Made;
        let (store, door) = led_again_behind_a_lost_add();
        let first = store.by_key(&format!("{URL}@0")).expect("the first rank");
        let add = || door.register(&operator(), under(TERM + 2), &ask(wide(URL, 3), Mode::Add));

        let told = add().expect("the same add");
        let made = made_of(&told);
        assert!(
            matches!(
                made.as_slice(),
                [Made::Relabelled(stood), Made::Registered(_), Made::Registered(_)]
                    if stood.id == first.id.0
            ),
            "{made:?}"
        );
        assert_eq!(store.by_key(&format!("{URL}@0")).as_ref(), Some(&first));
        assert_eq!((store.epoch(), store.all_rows().1.len()), (3, 3));

        let whole = add().expect_err("the engine stands whole");
        assert_eq!(whole.code(), Code::AlreadyExists);
        assert_eq!(
            whole.message(),
            format!(
                "the URL is registered as worker {:016x} at revision {}",
                first.id.0, first.revision
            )
        );
        assert_eq!(store.epoch(), 3);
    }

    /// The same add with another label than the standing rank's is refused
    /// by that rank's row and writes nothing: an add lays the rest by what
    /// stands alone.
    #[test]
    fn an_add_called_again_with_another_label_is_refused_and_writes_nothing() {
        let (store, door) = led_again_behind_a_lost_add();
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..wide(URL, 3)
        };
        let refused = door
            .register(&operator(), under(TERM + 2), &ask(labelled, Mode::Add))
            .expect_err("the first rank stands with other labels");
        assert_eq!(refused.code(), Code::AlreadyExists);
        assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
    }

    /// A registration states at most 64 ranks. At the bound, with every rank
    /// replaced by rows near their size bound, the answer stays inside a
    /// 4 MiB frame: a replacement tells two rows.
    #[test]
    fn a_registration_states_at_most_64_ranks_and_its_widest_answer_fits_a_frame() {
        use prost::Message as _;
        let (store, door) = door();
        let past = door
            .register(&operator(), under(TERM), &ask(wide(URL, 65), Mode::Add))
            .expect_err("past the bound");
        assert_eq!(
            (past.code(), past.message()),
            (
                Code::InvalidArgument,
                "spec.dp_size: a registration states at most 64 ranks"
            )
        );
        assert_eq!(store.epoch(), 0);

        let heavy = proto::WorkerSpec {
            models: vec!["m".repeat(256); 60],
            ..wide(URL, 64)
        };
        door.register(&operator(), under(TERM), &ask(heavy.clone(), Mode::Add))
            .expect("64 ranks");
        let told = door
            .register(&operator(), under(TERM), &ask(heavy, Mode::Replace))
            .expect("64 ranks replaced");
        let made = made_of(&told);
        assert_eq!(made.len(), 64);
        assert!(made
            .iter()
            .all(|made| matches!(made, proto::rank_row::Made::Replaced(_))));
        let bytes = told.encoded_len();
        assert!(bytes > 1 << 20, "{bytes}");
        assert!(bytes < 4 << 20, "{bytes}");
    }

    /// Each entry of the answer is the event a watcher is told of that row:
    /// the log's entries of one call, as a watch tells them, read beside
    /// the answer. A rank whose checks are off is answered as its write left
    /// it, and its connect signal is the watcher's next event. The answer's
    /// place is the call's last write.
    #[test]
    fn each_entry_of_an_answer_of_several_ranks_is_the_event_a_watcher_is_told() {
        use proto::rank_row::Made;
        let (_store, door) = door();
        let watched = |after: Position| {
            let mut entries = Vec::new();
            door.log
                .read(after, 64, &mut entries)
                .expect("the log holds the call's writes");
            entries
                .iter()
                .filter_map(super::worker_wire::member)
                .collect::<Vec<_>>()
        };
        let entries = |told: &proto::WorkerEvent| match told.event.clone().expect("an event") {
            Event::Ranks(ranks) => ranks.rows,
            other => panic!("no ranks in {other:?}"),
        };
        let ranked = |mode| {
            let spec = proto::WorkerSpec {
                dp_size: 2,
                ..unchecked(URL)
            };
            let before = door.place();
            let told = door
                .register(&operator(), under(TERM), &ask(spec, mode))
                .expect("the ranks are laid");
            let events = watched(before);
            let last = events.last().expect("the call's last write");
            assert_eq!(
                (told.seq, told.membership_epoch, told.term),
                (last.seq, last.membership_epoch, last.term)
            );
            // The connect signals are the watcher's own events.
            let rows: Vec<_> = events
                .iter()
                .filter_map(|event| event.event.clone())
                .filter(|event| !matches!(event, Event::Status(_)))
                .collect();
            let answered = entries(&told);
            assert_eq!(rows.len(), answered.len());
            for (event, entry) in rows.into_iter().zip(&answered) {
                assert_eq!(super::worker_wire::rank_row(event).as_ref(), Some(entry));
            }
            answered
        };

        // A worker that comes back ranked: two rows registered, the plain
        // row removed.
        door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("the plain row");
        let made = ranked(Mode::Upsert);
        assert!(
            matches!(
                made.iter()
                    .map(|entry| entry.made.as_ref())
                    .collect::<Vec<_>>()[..],
                [
                    Some(Made::Registered(_)),
                    Some(Made::Registered(_)),
                    Some(Made::Removed(_))
                ]
            ),
            "{made:?}"
        );
        // A replacement of ranks that are ready hands their status on.
        let made = ranked(Mode::Replace);
        assert_eq!(made.len(), 2);
        for entry in &made {
            let Some(Made::Replaced(replaced)) = &entry.made else {
                panic!("{entry:?}");
            };
            assert!(replaced.state_carried, "{replaced:?}");
        }
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

    /// A provider's maker for one worker.
    fn found(key: &str) -> Registrant {
        Registrant::provider("cluster", key).expect("a whole maker")
    }

    #[test]
    fn a_row_keeps_the_maker_its_registration_gave_it() {
        let (store, door) = door();
        // A control call's row says no maker on the wire.
        let by_hand = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect("registered"),
        );
        assert_eq!(by_hand.maker, None);
        let held = store.get(WorkerId(by_hand.id)).expect("the row");
        assert_eq!(held.maker, Registrant::Control);
        // A provider's registration makes the row its own, and the answer
        // shows whose it is.
        let maker = found("pods/a");
        let other = "grpc://10.0.0.6:8000";
        let made = row_of(
            &door
                .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert))
                .expect("found"),
        );
        let shown = Some(proto::Maker {
            provider: "cluster".to_owned(),
            key: "pods/a".to_owned(),
        });
        assert_eq!(made.maker, shown);
        // Its refresh in place and its new incarnation keep the maker. The
        // refresh comes from a maker made anew: two makers are one where
        // they agree whole, whichever copy a caller holds.
        let relabelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..spec(other)
        };
        let anew = found("pods/a");
        let in_place = row_of(
            &door
                .upsert(&operator(), &anew, TERM, &ask(relabelled, Mode::Upsert))
                .expect("refreshed"),
        );
        assert_eq!((in_place.id, &in_place.maker), (made.id, &shown));
        let again = row_of(
            &door
                .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Replace))
                .expect("replaced"),
        );
        assert_ne!(again.id, made.id);
        assert_eq!(again.maker, shown);
        let held = store.get(WorkerId(again.id)).expect("the row");
        assert_eq!(held.maker, maker);
    }

    #[test]
    fn a_caller_that_did_not_make_a_row_changes_nothing_of_it() {
        let (store, door) = door();
        let maker = found("pods/a");
        let made = row_of(
            &door
                .upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
                .expect("found"),
        );
        let epoch = store.epoch();
        let refused = |answer: Result<proto::WorkerEvent, Status>, whose: &str| {
            let status = answer.expect_err("refused");
            assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
            assert!(status.message().contains(whose), "{status:?}");
            assert!(!status.message().contains("10.0.0"), "no URL: {status:?}");
        };
        // A control call on the provider's row: a registration over it, with
        // other facts and with equal ones, a patch, a drain and a removal.
        let relabelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..spec(URL)
        };
        for request in [
            ask(spec(URL), Mode::Replace),
            ask(relabelled, Mode::Upsert),
            ask(spec(URL), Mode::Upsert),
        ] {
            let answer = door.register(&operator(), under(TERM), &request);
            refused(answer, "provider cluster");
        }
        let patch = proto::PatchWorkerRequest {
            worker_id: made.id,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            remove_labels: Vec::new(),
            overload: None,
            revision: made.revision,
        };
        refused(
            door.patch(&operator(), under(TERM), &patch),
            "provider cluster",
        );
        for drain in [true, false] {
            let leave = proto::DeregisterWorkerRequest {
                worker_id: made.id,
                revision: made.revision,
                drain,
            };
            refused(
                door.deregister(&operator(), under(TERM), &leave),
                "provider cluster",
            );
        }
        // The maker is tested ahead of the revision: a caller that did not
        // make the row is told so whatever revision it read.
        let stale = proto::PatchWorkerRequest {
            revision: made.revision + 7,
            ..patch.clone()
        };
        refused(
            door.patch(&operator(), under(TERM), &stale),
            "provider cluster",
        );
        // The same provider under another key is another worker's maker.
        let another = found("pods/b");
        refused(
            door.upsert(&operator(), &another, TERM, &ask(spec(URL), Mode::Upsert)),
            "provider cluster",
        );
        // Nothing was written, and the maker's own call goes through.
        assert_eq!(store.epoch(), epoch);
        let held = store.get(WorkerId(made.id)).expect("the row");
        assert_eq!(held.revision, made.revision);
        door.relabel(&operator(), &maker, &patch)
            .expect("the maker patches its row");

        // The other way round: a provider changes nothing of a control
        // call's row.
        let other = "grpc://10.0.0.6:8000";
        let by_hand = row_of(
            &door
                .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
                .expect("registered"),
        );
        refused(
            door.upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert)),
            "a control call",
        );
        let leave = proto::DeregisterWorkerRequest {
            worker_id: by_hand.id,
            revision: by_hand.revision,
            drain: false,
        };
        refused(door.remove(&operator(), &maker, &leave), "a control call");
        door.deregister(&operator(), under(TERM), &leave)
            .expect("the control call removes its row");
    }

    #[test]
    fn a_row_holds_its_maker_inside_the_wires_bound() {
        use super::worker_wire;

        let (store, door) = door();
        let maker = found("pods/a");
        // Three labels at a value's bound and one that fills. The widest
        // fill a row of this maker has room for is registered; one byte more
        // is refused by the row's name, and so is a patch of the row at its
        // brim.
        let stated = |fill: usize| proto::WorkerSpec {
            labels: ["a", "b", "c"]
                .map(|key| (key.to_owned(), "v".repeat(4_096)))
                .into_iter()
                .chain([("fill".to_owned(), "v".repeat(fill))])
                .collect(),
            ..spec(URL)
        };
        let fits = |fill: usize| {
            worker_wire::registration(&stated(fill), |_| None).is_ok_and(|(facts, _)| {
                worker_wire::room(&facts, &facts.labels, &maker, "spec").is_ok()
            })
        };
        let widest = (0..=4_096usize)
            .rev()
            .find(|&fill| fits(fill))
            .expect("a fill the row has room for");
        let epoch = store.epoch();
        let past = door.upsert(
            &operator(),
            &maker,
            TERM,
            &ask(stated(widest + 1), Mode::Upsert),
        );
        let status = past.expect_err("past the bound");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().starts_with("spec:"), "{status:?}");
        assert_eq!(store.epoch(), epoch);
        let brim = row_of(
            &door
                .upsert(
                    &operator(),
                    &maker,
                    TERM,
                    &ask(stated(widest), Mode::Upsert),
                )
                .expect("at the bound"),
        );
        let more = proto::PatchWorkerRequest {
            worker_id: brim.id,
            labels: [("lane".to_owned(), "b".to_owned())].into(),
            remove_labels: Vec::new(),
            overload: None,
            revision: brim.revision,
        };
        let status = door
            .relabel(&operator(), &maker, &more)
            .expect_err("past the bound");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().starts_with("labels:"), "{status:?}");
    }

    fn found_facts() -> WorkerFacts {
        WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            wire_modes: WireModes::GRPC,
            models: vec!["model-a".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    /// A standby builds its rows from the log. A fact the log's row does not
    /// carry would leave it another row than its leader holds.
    #[test]
    fn a_fact_the_log_does_not_carry_is_refused_by_its_name_and_nothing_is_written() {
        use super::super::worker_wire;
        use crate::store::{pairing::PairingDescriptor, EncodeHandoff, EncodeOffer};
        type State = fn(&mut WorkerFacts);

        let (store, door) = door();
        let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
        let maker = found("pods/a");
        let epoch = store.epoch();
        let lost: [(&str, State); 9] = [
            ("row.pairing", |facts| {
                facts.pairing = Some(Arc::new(PairingDescriptor {
                    protocol: Some("pd".to_owned()),
                    ..PairingDescriptor::default()
                }));
            }),
            ("row.encode", |facts| {
                facts.encode = EncodeOffer {
                    handoff: EncodeHandoff::Push,
                    producer: true,
                    ..EncodeOffer::default()
                };
            }),
            ("row.vision", |facts| facts.vision = true),
            ("row.priority", |facts| facts.priority = 5),
            ("row.zone", |facts| facts.zone = Some(String::new())),
            ("row.max_running", |facts| facts.max_running = Some(0)),
            ("row.kv_block_size", |facts| facts.kv_block_size = Some(0)),
            ("row.bootstrap_port", |facts| facts.bootstrap_port = Some(0)),
            ("row.drain_settle_secs", |facts| {
                facts.drain_settle_secs = Some(30);
            }),
        ];
        for (fact, state) in lost {
            let answer = hosted.found_one(&maker, URL, &|_| {
                let mut facts = found_facts();
                state(&mut facts);
                facts
            });
            let Err(MembershipWriteRefused::Past(past)) = answer else {
                panic!("{fact}: wrong answer: {answer:?}")
            };
            assert_eq!(past.fact, fact);
        }
        assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

        // What the log carries is taken, and a standby reads it back whole.
        hosted
            .found_one(&maker, URL, &|_| found_facts())
            .expect("found");
        let row = store.by_key(URL).expect("the found row");
        let back = worker_wire::told_row(&worker_wire::row(&row), std::time::Instant::now());
        assert_eq!(back.map(|back| back.facts), Ok(row.facts));
    }

    /// The store reads a row as host-local by its reach alone: a row of the
    /// host-local wire that names no gateway would stand in every gateway's
    /// pool. A store held in process serves one gateway.
    #[test]
    fn a_host_local_wire_names_its_gateway_at_the_controllers_door() {
        let stated = |wire: WireModes, from: Reachability| {
            move |_: Option<&WorkerFacts>| WorkerFacts {
                wire_modes: wire,
                reachable_from: from,
                ..found_facts()
            }
        };
        let (store, door) = door();
        let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
        let maker = found("pods/a");
        let epoch = store.epoch();
        for wire in [WireModes::ZMQ, WireModes::GRPC.union(WireModes::ZMQ)] {
            let answer = hosted.found_one(&maker, URL, &stated(wire, Reachability::Any));
            let Err(MembershipWriteRefused::Past(past)) = answer else {
                panic!("{wire:?}: wrong answer: {answer:?}")
            };
            assert_eq!(past.fact, "row.wire");
        }
        assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

        // A row that names its gateway is in that gateway's pool alone.
        let one = Reachability::Gateway(GatewayId(7));
        hosted
            .found_one(&maker, URL, &stated(WireModes::ZMQ, one))
            .expect("found");
        let row = store.by_key(URL).expect("the found row");
        assert_eq!(row.facts.reachable_from, one);
    }

    fn standing(row: &WorkerRow) -> RowRevision {
        RowRevision {
            row: row.id,
            revision: row.revision,
        }
    }

    impl Rows for ReplicaMembershipWriter {
        fn rows(
            &self,
            maker: &Registrant,
            expect: Precondition,
            rows: &[Stated<'_>],
        ) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
            ReplicaMembershipWriter::rows(self, maker, expect, rows)
        }
    }

    fn hosted_door() -> (Arc<MemoryStore>, Box<dyn Rows>) {
        let (store, door) = door();
        (
            store,
            Box::new(ReplicaMembershipWriter::new(
                Arc::new(door),
                operator(),
                under(TERM),
            )),
        )
    }

    #[test]
    fn a_registration_is_laid_over_what_it_expects() {
        check_expectations(hosted_door);
    }

    #[test]
    fn a_found_row_states_its_event_signal() {
        check_event_signal(hosted_door);
    }

    #[test]
    fn a_hosted_door_checks_and_writes_the_whole_registration() {
        check_rank_rows(hosted_door).expect("rank registrations at the hosted door");
    }

    #[test]
    fn a_hosted_door_keeps_the_write_of_one_row() {
        check_one_row(hosted_door);
    }

    #[test]
    fn a_removal_takes_only_the_row_its_asker_saw() {
        check_guard(hosted_door).expect("both registrations stand in turn");
    }

    #[test]
    fn a_refresh_writes_its_workers_labels_and_no_other_fact() {
        check_refresh(hosted_door).expect("the makers and their rows");
    }

    #[test]
    fn the_hosts_own_source_removes_its_rows_under_any_key() {
        check_provider(hosted_door).expect("the makers and their rows");
    }

    #[test]
    fn health_removes_inside_its_scope_alone() {
        check_health_scope(hosted_door).expect("the makers and their rows");
    }

    #[test]
    fn a_removal_leaves_the_rows_maker_as_it_was() {
        check_maker(hosted_door).expect("the maker and its row");
    }

    #[test]
    fn a_provider_acting_as_maker_takes_only_its_own_guarded_row() {
        let (store, door) = door();
        let door = Arc::new(door);
        let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
        let maker = found("pods/a");
        hosted
            .found_one(&maker, URL, &|_| found_facts())
            .expect("found");
        let made = store.by_key(URL).expect("the found row");
        let other = "grpc://10.0.0.6:8000";
        door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
            .expect("registered");
        let epoch = store.epoch();
        let denied = |answer: Result<MembershipWriteOutcome, MembershipWriteRefused>,
                      whose: &str| {
            let Err(MembershipWriteRefused::NotRegistrant(why)) = answer else {
                panic!("wrong refusal: {answer:?}")
            };
            assert!(why.contains(whose), "{why}");
            assert!(!why.contains("10.0.0"), "{why}");
            assert!(!why.contains("pods/"), "{why}");
        };
        // Another maker's row stays: a control call's, and this provider's
        // own under another key, which is told so.
        let by = Remover::Registrant(&maker);
        let seen = standing(&made);
        let control = store.by_key(other).expect("the control row");
        denied(hosted.gone(by, other, standing(&control)), "a control call");
        denied(
            hosted.gone(Remover::Registrant(&found("pods/b")), URL, seen),
            "provider cluster's under another key",
        );
        // A caller of another provider is told whose the row is, and
        // nothing of a key.
        let stranger = Registrant::provider("nodes", "pods/a").expect("a whole maker");
        denied(
            hosted.gone(Remover::Registrant(&stranger), URL, seen),
            "provider cluster's: its worker",
        );
        // No row under the URL is not found, and a replica that does not
        // lead refuses by name. None of the five wrote.
        assert_eq!(
            hosted.gone(by, "grpc://10.0.0.9:8000", seen),
            Err(MembershipWriteRefused::Absent)
        );
        let follower = || -> Result<u64, Status> {
            Err(Status::failed_precondition("this replica does not lead"))
        };
        let follower = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), follower);
        assert_eq!(
            follower.gone(by, URL, seen),
            Err(MembershipWriteRefused::NotLeading(
                "this replica does not lead".to_owned()
            ))
        );
        assert_eq!(store.epoch(), epoch);
        // The maker's own call takes the row away and answers it removed.
        assert_eq!(
            hosted.gone(by, URL, seen),
            Ok(MembershipWriteOutcome::Removed(made.id))
        );
        assert!(store.get(made.id).is_none());
        assert_eq!(store.epoch(), epoch + 1);
        // The old guard still names the previous row at this URL.
        hosted
            .found_one(&found("pods/c"), URL, &|_| found_facts())
            .expect("found again");
        let again = store.by_key(URL).expect("the new row");
        assert_ne!(again.id, made.id);
        assert_eq!(
            hosted.gone(by, URL, seen),
            Err(MembershipWriteRefused::Moved)
        );
        assert_eq!(store.by_key(URL), Some(again));
    }

    #[test]
    fn a_providers_write_on_a_replica_that_does_not_lead_is_refused_and_writes_nothing() {
        use std::sync::atomic::AtomicUsize;

        let (store, door) = door();
        let door = Arc::new(door);
        let maker = found("pods/a");
        ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM))
            .found_one(&maker, URL, &|_| found_facts())
            .expect("registered");
        let row = store.by_key(URL);
        let seen = standing(row.as_ref().expect("the registered row"));
        let by = Remover::Registrant(&maker);
        let epoch = store.epoch();
        let at = door.place();
        let reads = Arc::new(AtomicUsize::new(0));
        let fence_reads = Arc::clone(&reads);
        let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), move || {
            assert!(door.turn.is_locked(), "the fence is read inside the turn");
            fence_reads.fetch_add(1, Ordering::Relaxed);
            Err(Status::failed_precondition("this replica does not lead"))
        });
        let refused = Err(MembershipWriteRefused::NotLeading(
            "this replica does not lead".to_owned(),
        ));
        assert_eq!(
            hosted.found_one(&maker, URL, &|_| panic!(
                "a follower does not ask for facts"
            )),
            refused
        );
        assert_eq!(hosted.drained(by, URL, seen), refused);
        assert_eq!(hosted.gone(by, URL, seen), refused);
        assert_eq!(reads.load(Ordering::Relaxed), 3);
        assert_eq!(store.by_key(URL), row);
        assert_eq!(store.epoch(), epoch);
        assert_eq!(hosted.door.place(), at);
    }

    #[test]
    fn a_providers_registration_numbers_its_life_under_the_term_the_log_leads() {
        use super::{ReplicaTurn, Turn};
        use crate::store::membership::FactPastBound;

        let (store, door) = door_in(TERM + 1);
        let door = Arc::new(door);
        let maker = found("pods/a");
        let stale = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
        let refusal = Err(MembershipWriteRefused::NotLeading(
            super::LEADS_NO_TERM.to_owned(),
        ));
        assert_eq!(stale.found_one(&maker, URL, &|_| found_facts()), refusal);
        assert_eq!(store.epoch(), 0);

        let leading = ReplicaMembershipWriter::new(door, operator(), under(TERM + 1));
        let number = life(TERM + 1, 0);
        assert_eq!(
            leading.found_one(&maker, URL, &|_| found_facts()),
            Ok(MembershipWriteOutcome::Registered(identity(URL, number)))
        );
        let row = store.by_key(URL).expect("the registered row");
        assert_eq!(row.facts.incarnation, number);
        let replaced = || WorkerFacts {
            role: Role::Decode,
            ..found_facts()
        };
        assert_eq!(stale.found_one(&maker, URL, &|_| replaced()), refusal);
        assert_eq!(store.by_key(URL), Some(row.clone()));
        let new = life(TERM + 1, 1);
        assert_eq!(
            leading.found_one(&maker, URL, &|_| replaced()),
            Ok(MembershipWriteOutcome::Another {
                old: row.id,
                new: identity(URL, new)
            })
        );
        assert_eq!(
            store
                .by_key(URL)
                .expect("the replaced row")
                .facts
                .incarnation,
            new
        );

        let turn = ReplicaTurn {
            host: &leading,
            term: TERM + 1,
        };
        let heavy = WorkerFacts {
            models: vec!["m".repeat(256); 64],
            ..found_facts()
        };
        assert_eq!(
            turn.room(&heavy, &maker),
            Err(MembershipWriteRefused::Past(FactPastBound {
                fact: "row",
                why: "a row is at most 16384 bytes on the wire",
            }))
        );

        let (store, door) = door_in(TERM);
        let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
        let state = |_: Option<&WorkerFacts>| {
            store::LeasesStore::set_term(store.as_ref(), TERM + 1);
            found_facts()
        };
        assert_eq!(hosted.found_one(&maker, URL, &state), refusal);
        assert_eq!(store.all_rows(), (0, Vec::new()));
        // A life is asked a row at a time: once the term has moved the turn
        // makes none, at whichever row of a registration asks it.
        let turn = ReplicaTurn {
            host: &hosted,
            term: TERM,
        };
        assert_eq!(
            turn.life(),
            Err(MembershipWriteRefused::NotLeading(
                super::LEADS_NO_TERM.to_owned()
            ))
        );
    }

    #[test]
    fn the_two_doors_answer_one_row_alike() {
        let (replica, door) = door();
        let local = Arc::new(MemoryStore::default());
        let maker = found("pods/a");
        let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
        let lone = StoreDoor::new(Arc::clone(&local));
        let write = |door: &dyn Rows, store: &MemoryStore| {
            let by = Remover::Registrant(&maker);
            assert_eq!(
                door.gone(
                    by,
                    URL,
                    RowRevision {
                        row: WorkerId(0),
                        revision: 0
                    }
                ),
                Err(MembershipWriteRefused::Absent)
            );
            let registered = door
                .found_one(&maker, URL, &|_| found_facts())
                .expect("registered");
            let first = store.by_key(URL).expect("the first row");
            assert_eq!(registered, MembershipWriteOutcome::Registered(first.id));
            assert_eq!(
                door.found_one(&maker, URL, &|_| found_facts()),
                Ok(MembershipWriteOutcome::Unmoved(first.id))
            );
            let refresh = || WorkerFacts {
                max_running: Some(8),
                ..found_facts()
            };
            assert_eq!(
                door.found_one(&maker, URL, &|_| refresh()),
                Ok(MembershipWriteOutcome::InPlace(first.id))
            );
            let refreshed = store.by_key(URL).expect("the refreshed row");
            let wrote = door.found_one(&maker, URL, &|_| WorkerFacts {
                role: Role::Decode,
                ..refresh()
            });
            let fresh = store.by_key(URL).expect("the new row");
            assert_eq!(
                wrote,
                Ok(MembershipWriteOutcome::Another {
                    old: first.id,
                    new: fresh.id
                })
            );
            let seen = standing(&fresh);
            assert_eq!(
                door.drained(by, URL, seen),
                Ok(MembershipWriteOutcome::Draining(fresh.id))
            );
            let draining = store.by_key(URL).expect("the draining row");
            assert_eq!(
                door.drained(by, URL, seen),
                Ok(MembershipWriteOutcome::Unmoved(fresh.id))
            );
            assert_eq!(
                door.gone(by, URL, seen),
                Ok(MembershipWriteOutcome::Removed(fresh.id))
            );
            assert_eq!(store.by_key(URL), None);
            [first, refreshed, fresh, draining].map(|row| {
                assert_eq!(row.id, identity(URL, row.facts.incarnation));
                (
                    WorkerFacts {
                        incarnation: 0,
                        ..row.facts
                    },
                    row.revision,
                    row.status,
                    row.maker,
                )
            })
        };
        assert_eq!(write(&lone, &local), write(&hosted, &replica));
        assert_eq!(local.epoch(), replica.epoch());
    }

    #[test]
    fn an_add_over_a_standing_row_is_already_there_whoever_made_it() {
        let (store, door) = door();
        let maker = found("pods/a");
        door.upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
            .expect("found");
        let other = "grpc://10.0.0.6:8000";
        door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
            .expect("registered");
        let epoch = store.epoch();
        let there = |answer: Result<proto::WorkerEvent, Status>| {
            let status = answer.expect_err("refused");
            assert_eq!(status.code(), Code::AlreadyExists, "{status:?}");
        };
        // An add asks no maker: a control call's and a provider's read the
        // same over a provider's row and over a control call's.
        for url in [URL, other] {
            there(door.register(&operator(), under(TERM), &ask(spec(url), Mode::Add)));
            there(door.upsert(&operator(), &maker, TERM, &ask(spec(url), Mode::Add)));
        }
        assert_eq!(store.epoch(), epoch);
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
        // A patch is the same registration: draining still.
        let patch = proto::PatchWorkerRequest {
            worker_id: row.id,
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            remove_labels: Vec::new(),
            overload: None,
            revision: row.revision,
        };
        let told = door
            .relabel(&operator(), &Registrant::Control, &patch)
            .expect("patched");
        assert_eq!(row_of(&told).status, proto::WorkerStatus::Draining as i32);
        // A registration over a draining row is the URL's next incarnation,
        // by an upsert as by a replacement: the drain is not handed on, and
        // an unchecked worker is ready from its own.
        let told = door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Upsert))
            .expect("registered anew");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a replacement: {told:?}");
        };
        assert!(!replaced.state_carried);
        let anew = store.by_key(URL).expect("the new row");
        assert_eq!(anew.status, WorkerStatus::Ready);
        let drain = proto::DeregisterWorkerRequest {
            worker_id: anew.id.0,
            revision: anew.revision,
            drain: true,
        };
        door.deregister(&operator(), under(TERM), &drain)
            .expect("draining again");
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

    pub(crate) use crate::document::operator::test_support::Logged;

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

    /// A registration that states several ranks leaves a line a row: what
    /// the call made of each rank, `nothing` where it moved none, and each
    /// row that left.
    #[test]
    fn a_registration_of_several_ranks_leaves_an_audit_line_a_row() {
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let (_store, door) = door();
        for (width, mode) in [(3, Mode::Add), (3, Mode::Upsert), (2, Mode::Upsert)] {
            door.register(&operator(), under(TERM), &ask(wide(URL, width), mode))
                .expect("laid");
        }
        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let made: Vec<&str> = logged
            .lines()
            .map(|line| {
                let call = "call=\"RegisterWorker\" by=operator-a";
                assert!(
                    line.contains("membership write") && line.contains(call),
                    "{line}"
                );
                ["registered", "relabelled", "nothing", "removed"]
                    .into_iter()
                    .find(|made| line.contains(&format!("made=\"{made}\"")))
                    .expect("what the call made of the row")
            })
            .collect();
        assert_eq!(
            made,
            [
                "registered",
                "registered",
                "registered",
                "nothing",
                "nothing",
                "nothing",
                "relabelled",
                "relabelled",
                "removed"
            ]
        );
        // No line holds the URL: it may carry a credential.
        assert!(!logged.contains(URL), "{logged}");
    }

    #[test]
    fn a_removal_health_asked_leaves_an_audit_line_that_names_health() {
        use tracing::{
            subscriber::{self, NoSubscriber},
            Dispatch,
        };

        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = Dispatch::new(NoSubscriber::default());
        let _kept = subscriber::set_default(subscriber);

        let (store, door) = hosted_door();
        let key = "private/health-key";
        let maker = found(key);
        door.found_one(&maker, URL, &|_| found_facts())
            .expect("registered");
        let row = store.by_key(URL).expect("the row health saw");
        let seen = RowRevision {
            row: row.id,
            revision: row.revision,
        };
        assert_eq!(
            door.drained(Remover::Health(UnhealthyRemovalScope::Disabled), URL, seen),
            Err(MembershipWriteRefused::OutOfScope)
        );
        let by = Remover::Health(UnhealthyRemovalScope::DiscoveryOwned);
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Draining(row.id))
        );
        assert_eq!(
            door.gone(by, URL, seen),
            Ok(MembershipWriteOutcome::Removed(row.id))
        );

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let calls: Vec<_> = logged.lines().skip(1).collect();
        assert_eq!(calls.len(), 3, "{logged}");
        assert!(
            calls.iter().all(|line| line.contains("asker=\"health\"")),
            "{logged}"
        );
        assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
        assert!(calls[1].contains("made=\"status\""), "{logged}");
        assert!(calls[2].contains("made=\"removed\""), "{logged}");
        assert!(!logged.contains(URL) && !logged.contains(key), "{logged}");
    }

    /// A refresh leaves one audit line a call, under its own name. One that
    /// states another fact than the labels is refused in a refresh's
    /// words, never in a removal's, and no line holds the URL.
    #[test]
    fn a_refresh_leaves_its_audit_line_in_its_own_words() {
        const HTTP: &str = "http://10.0.0.5:8000";
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let (store, door) = hosted_door();
        let built = |model: &str, tier: &str| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(HTTP).model(ModelCard::new(model));
            Arc::new(worker.label("tier", tier).build())
        };
        let first = built("m", "gold");
        door.found_built(&Registrant::Control, Precondition::Absent, &first, None)
            .expect("registered");
        let row = store.by_key(HTTP).expect("the registered row");
        let seen = standing(&row);
        let stray = door.refreshed(seen, &built("other", "gold"), &first);
        assert_eq!(stray, Err(MembershipWriteRefused::NotLabelsAlone));
        let wrote = door.refreshed(seen, &built("m", "silver"), &first);
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let calls: Vec<_> = logged.lines().skip(1).collect();
        assert_eq!(calls.len(), 2, "{logged}");
        assert!(
            calls
                .iter()
                .all(|line| line.contains("worker label refresh")),
            "{logged}"
        );
        assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
        assert!(
            calls[0].contains(MembershipWriteRefused::NOT_LABELS_ALONE),
            "{logged}"
        );
        assert!(calls[1].contains("made=\"relabelled\""), "{logged}");
        assert!(!logged.contains(HTTP), "{logged}");
    }

    /// The controller's door reads a built worker's facts alone. Its log
    /// carries no pairing descriptor and its room test refuses a row that
    /// states one. So a worker that reports a transport and a version is
    /// registered and refreshed here, and its row states no descriptor.
    #[test]
    fn a_worker_that_reports_pairing_facts_is_laid_with_no_descriptor() {
        const HTTP: &str = "http://10.0.0.6:8000";
        let (store, door) = hosted_door();
        let built = |version: &str| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(HTTP)
                .model(ModelCard::new("m"))
                .runtime_type(smg_worker::worker::RuntimeType::Sglang)
                .label("disaggregation_transfer_backend", "nixl");
            Arc::new(worker.label("version", version).build())
        };
        let first = built("1");
        let offered = first.pd_pairing();
        assert_eq!(
            (offered.transport(), offered.version()),
            (Some("nixl"), Some("1"))
        );
        let wrote = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
        assert!(wrote.is_ok(), "{wrote:?}");
        let row = store.by_key(HTTP).expect("the registered row");
        assert_eq!(row.facts.pairing, None);

        let wrote = door.refreshed(standing(&row), &built("2"), &first);
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        let refreshed = store.by_key(HTTP).expect("the refreshed row");
        assert_eq!(refreshed.facts.pairing, None);
        let version = refreshed.facts.labels.get("version");
        assert_eq!(version.map(String::as_str), Some("2"));
    }

    #[test]
    fn every_write_of_a_provider_leaves_its_audit_line() {
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);

        let (store, door) = door();
        let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
        let key = "private/provider-key";
        let maker = found(key);
        let rank_urls = ["grpc://10.0.0.5:8000@0", "grpc://10.0.0.5:8000@1"];
        let states = [0, 1].map(|rank: u16| {
            let key = rank_urls[usize::from(rank)];
            move |_: Option<&WorkerFacts>| WorkerFacts {
                key: key.to_owned(),
                dp_rank: rank,
                dp_size: 2,
                ..found_facts()
            }
        });
        let ranks = [0, 1].map(|rank| Stated {
            url: rank_urls[rank],
            state: &states[rank],
        });
        hosted
            .rows(&maker, Precondition::Any, &ranks)
            .expect("registered ranks");
        let [first, second] = rank_urls.map(|url| store.by_key(url).expect("a rank").id.0);
        hosted
            .rows(&maker, Precondition::Any, &ranks)
            .expect("unmoved ranks");

        hosted
            .rows(
                &maker,
                Precondition::Any,
                &[Stated {
                    url: URL,
                    state: &|_| found_facts(),
                }],
            )
            .expect("registered plain and retired ranks");
        let id = store.by_key(URL).expect("the registered row").id.0;
        let refreshed = || WorkerFacts {
            max_running: Some(8),
            ..found_facts()
        };
        hosted
            .found_one(&maker, URL, &|_| refreshed())
            .expect("relabelled");
        hosted
            .found_one(&maker, URL, &|_| WorkerFacts {
                role: Role::Decode,
                ..refreshed()
            })
            .expect("replaced");
        let row = store.by_key(URL).expect("the replaced row");
        let new = row.id.0;
        let seen = standing(&row);
        let by = Remover::Registrant(&maker);
        hosted.drained(by, URL, seen).expect("drained");
        hosted.drained(by, URL, seen).expect("unmoved");
        hosted
            .gone(Remover::Registrant(&found("another/key")), URL, seen)
            .expect_err("another maker");
        hosted
            .gone(Remover::Provider("cluster"), URL, seen)
            .expect("gone");
        hosted.gone(by, URL, seen).expect_err("already gone");

        let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let lines: Vec<&str> = logged.lines().collect();
        let registration = "worker registration";
        let want = [
            (registration, "registered", Some((first, 1, 2))),
            (registration, "registered", Some((second, 2, 3))),
            (registration, "nothing", Some((first, 2, 3))),
            (registration, "nothing", Some((second, 2, 3))),
            (registration, "registered", Some((id, 3, 4))),
            (registration, "removed", Some((first, 4, 5))),
            (registration, "removed", Some((second, 5, 6))),
            (registration, "relabelled", Some((id, 6, 7))),
            (registration, "replaced", Some((new, 7, 8))),
            ("worker drain", "status", Some((new, 8, 9))),
            ("worker drain", "nothing", Some((new, 8, 9))),
            (
                "worker removal",
                "code=PermissionDenied why=\"the row is provider cluster's under another key",
                None,
            ),
            ("worker removal", "removed", Some((new, 9, 10))),
            (
                "worker removal",
                "code=NotFound why=\"no worker is registered under the URL\"",
                None,
            ),
        ];
        assert_eq!(lines.len(), want.len(), "{logged}");
        for (line, (call, detail, place)) in lines.iter().zip(want) {
            assert!(
                line.contains(&format!("call=\"{call}\" by=operator-a")),
                "{line}"
            );
            assert_eq!(
                line.contains("asker=\"cluster\""),
                call != registration,
                "{line}"
            );
            if let Some((worker, epoch, seq)) = place {
                let fields = format!("worker={worker:016x} made=\"{detail}\" membership_epoch={epoch} seq={seq} term={TERM}");
                assert!(
                    line.contains("membership write") && line.contains(&fields),
                    "{line}"
                );
            } else {
                assert!(
                    line.contains("membership call refused") && line.contains(detail),
                    "{line}"
                );
            }
        }
        for private in [URL, key, "another/key"] {
            assert!(!logged.contains(private), "{logged}");
        }
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
                let (store, feed) = journaled(&LeasePlace::Process, 1);
                let objects = holder(&store);
                MembershipCalls::new(store, feed.log, objects).base
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
            let (store, feed) = journaled(&LeasePlace::Process, 1);
            store::LeasesStore::set_term(store.as_ref(), 1);
            let door = MembershipCalls::new(Arc::clone(&store), feed.log, holder(&store));
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

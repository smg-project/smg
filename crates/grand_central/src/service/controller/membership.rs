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
//! the objects in their holder's turn
//! ([`Objects::record`]). A worker object's status leads its row's, so
//! a record read carries the status its object states.
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
//! ([`Objects::upsert`]): the object stands when the write returns. A drain
//! and a connect signal go through the holder too: the store is told the
//! status first, and the object moves where the store took it
//! ([`Objects::drain_row_under`], [`Objects::connect_row_under`]). So a
//! decision made behind an acknowledged registration, drain or removal
//! reads the fleet the call left, and the door follows nothing behind its
//! write. A standby's applier writes its leader's rows through the holder
//! too ([`Objects::follow_at`]).
//!
//! **A status is written under its call's term, or not at all.** A connect
//! signal and a drain hand the holder the door's own word: the log leads
//! the term the call was admitted under still ([`Log::leads`]). The holder
//! asks it under the object's lock, and the store asks it again under its
//! membership lock. A call that waited at either through the end of its
//! term writes no status and tells the journal nothing. The term goes on
//! with the word. The store tells its journal first, and the log compares
//! the term in the take of its lock that gives the event its place. Where
//! the log has left the term, it takes no event and the store moves
//! nothing: no term ends between that ask and the stamp.
//! A drain that wrote nothing asks the term once more. Where it ended, the
//! call answers that its store does not lead the term, as a registration
//! that can mint no life does ([`MembershipCalls::next_life`]), and
//! otherwise the row moved. A registration that waits answers the row as it
//! reads behind its connect signal, written or not.
//!
//! **A row is written under its call's term, or not at all.** A
//! registration's write, a patch and a removal go to the holder with the
//! term the call was admitted under ([`Objects::upsert_under`]). The store
//! tells its journal first, with that term, and moves its tables behind
//! the journal's yes. Where the log has left the term, it takes no event
//! and the store moves nothing. The door then asks the term once more and
//! answers that its store does not lead it, as a drain does
//! ([`MembershipCalls::unwritten`]). The status the holder tells behind a
//! change in place goes under that term too. A removal's object left the
//! registry ahead of the store's word: it is laid again from the row
//! read off it, in the status the table holds.
//!
//! **A signal that was not written is sent again by an upsert of the same
//! registration.** Its row waits for it: its checks are off, and it is
//! neither ready nor draining ([`awaits_connect`]). No prober moves such a
//! row. An upsert of equal facts writes no row, moves no revision and makes
//! no life: it sends the signal with the word of its own term, and where it
//! waits it answers the row behind it. A rank that stands unmoved in a
//! wider registration is sent it the same way, by the host door's routine:
//! under an upsert, and under an add that lays the ranks it left. An add
//! over a row that stands, or over an engine that stands whole, is refused
//! ahead of every write and sends none. A term's start sends none.
//!
//! **The lock order of that word.** It takes the log's lock alone, inside
//! the object's lock and the store's membership lock: the order a
//! journaled write takes the three, which tells the log under the
//! membership lock. The log holds neither a store nor an object, and what
//! it wakes it wakes under no lock, so no holder of the log's lock takes
//! one of the other two.
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
        LeadWord, MembershipChange, MemoryStore, Position, Registrant, WorkerFacts, WorkerPatch,
        WorkerRow, WorkerStatus, WorkersStore,
    },
    worker_host::{
        asks_events_under, awaits_connect, dial_address, drain, lay, lay_one, lay_ready, refresh,
        stated, stating, take, Members, MembershipWriteOutcome, MembershipWriteRefused,
        MembershipWriter, Objects, Over, Precondition, Ready, Remover, RowRevision, Stated, Turn,
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
    /// over them, and its write of a status goes through them
    /// ([`Self::connect_row`], [`Self::drain_row`]).
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

    /// The door's write of a registration's facts, in its turn: decided
    /// over the worker objects, which the store follows
    /// ([`Objects::upsert_under`]). It goes with `term`, the term the call
    /// was admitted under: the store's journal takes the row's event only
    /// while the log leads it, asked where the event takes its place.
    /// `None` where another writer moved the row under the turn, and where
    /// the log has left `term`: nothing is written then
    /// ([`Self::unwritten`]).
    fn lay_row(
        &self,
        facts: WorkerFacts,
        maker: Registrant,
        term: u64,
    ) -> Option<MembershipChange> {
        self.objects
            .upsert_under(&self.store, facts, maker, Some(term))
    }

    /// The door's write of a patch, as [`Self::lay_row`] writes a
    /// registration.
    fn patch_row(
        &self,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
        term: u64,
    ) -> Option<MembershipChange> {
        self.objects
            .patch_under(&self.store, id, patch, revision, Some(term))
    }

    /// The door's removal of a row, as [`Self::lay_row`] writes a
    /// registration.
    fn take_row(&self, id: WorkerId, revision: u64, term: u64) -> Option<MembershipChange> {
        self.objects
            .remove_under(&self.store, id, revision, Some(term))
    }

    /// The door's connect signal for a row whose checks are off, in its
    /// turn: written on the row's worker object, and the store is told
    /// ([`Objects::connect_row_under`]). It goes with the word of `term`,
    /// the term the call was admitted under, and with the term itself. A
    /// call whose term ended while the write waited for a lock writes
    /// nothing. The store's journal takes the status only while the log
    /// leads `term`, asked where the event takes its place.
    fn connect_row(&self, id: WorkerId, revision: u64, term: u64) {
        let leads = || self.log.leads(term).is_ok();
        let word = Self::word(&leads, term);
        self.objects
            .connect_row_under(&self.store, id, revision, word);
    }

    /// The door's drain of a row, as [`Self::connect_row`] writes a
    /// status, with the word of `term` and the term. Whether the row
    /// moved: a drain whose term ended by its event's place moves nothing,
    /// as one whose row left the revision.
    fn drain_row(&self, id: WorkerId, revision: u64, term: u64) -> bool {
        let leads = || self.log.leads(term).is_ok();
        let word = Self::word(&leads, term);
        self.objects
            .drain_row_under(&self.store, id, revision, word)
    }

    /// The door's word for a status it writes under `term`: `leads`, its
    /// read of the log, and the term for the store's journal.
    fn word(leads: &dyn Fn() -> bool, term: u64) -> LeadWord<'_> {
        LeadWord {
            leads,
            term: Some(term),
        }
    }

    /// The door's read of the row under `id`, in its turn: the record read
    /// back from the worker object that stands, with the status that
    /// object states ([`Objects::record`]).
    fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.objects.record(id)
    }

    /// The door's read of the row under `url`, as [`Self::row`] reads one.
    fn row_under(&self, url: &str) -> Option<WorkerRow> {
        self.objects.record_under(url)
    }

    /// The door's read of the rows that name `engine`, as [`Self::row`]
    /// reads one.
    fn rows_of(&self, engine: &str) -> Vec<WorkerRow> {
        self.objects.records_of(engine)
    }

    /// Whether a found row of `facts` asks for its engine's cache events,
    /// under the generation in force at the door's store
    /// ([`asks_events_under`]).
    #[cfg_attr(
        not(any(test, feature = "discovery")),
        expect(dead_code, reason = "a row is found at the replica's own writer")
    )]
    fn asks_events(&self, facts: &WorkerFacts) -> bool {
        asks_events_under(facts, &self.store)
    }

    /// The log's place: read after a call's write, that write's own or one
    /// past it.
    fn place(&self) -> Position {
        self.log.head()
    }

    /// The audit line of what `call`, from `by`, made of a row: `wrote`
    /// names the row and what was made of it. The line carries the store's
    /// epoch and the log's place ([`Self::place`]), both read here.
    fn audit(
        &self,
        call: &'static str,
        by: &Principal,
        wrote: MembershipWriteOutcome,
        asker: Option<&str>,
    ) {
        let (made, worker) = made(wrote);
        let (epoch, at) = (self.store.epoch(), self.place());
        audit_write(call, by, worker.0, made, epoch, at, asker);
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

    /// The refusal of a call whose write the store did not take, asked
    /// behind that write. Where the term the call was admitted under has
    /// ended by this ask, the store does not lead it, and the call says
    /// so whatever refused the write: the row need not have moved.
    /// Otherwise the call is refused as one whose row another writer
    /// moved: the table did not hold what the holder read.
    fn unwritten(&self, admitted: u64) -> Status {
        self.check_term(admitted).err().unwrap_or_else(moved)
    }

    fn life(&self, term: u64) -> u64 {
        life(term, self.base, self.store.epoch())
    }

    /// Run `call` in the door's turn: one call at a time, from its read to
    /// its read back. The turn is let go when `call` returns.
    fn in_turn<T>(&self, call: impl FnOnce() -> T) -> T {
        let _turn = self.turn.lock();
        call()
    }

    /// One call of the door, admitted from `by`: its turn, the term
    /// `leading` (the service's read of the fence) tells once the turn is
    /// held, its write under that term, and the audit line of a refusal.
    /// Every call leaves a line with its caller (design 5.2): a write's is
    /// [`told`]'s, a refusal's says its code and its words, which hold no
    /// URL.
    ///
    /// The replica's worker objects hold the write when the call answers.
    /// A row's write was decided over them in the turn, and a status is
    /// written on the object first. A decision reads those objects, so
    /// none made behind an acknowledged registration, drain or removal
    /// reads the fleet as it stood before the call.
    fn call(
        &self,
        call: &'static str,
        by: &Principal,
        leading: impl FnOnce() -> Result<u64, Status>,
        write: impl FnOnce(u64) -> Result<proto::WorkerEvent, Status>,
    ) -> Result<proto::WorkerEvent, Status> {
        let answer = self.in_turn(|| leading().and_then(write));
        if let Err(refusal) = &answer {
            audit_refusal(call, by, refusal.code(), refusal.message(), None);
        }
        answer
    }

    /// One call of the replica's own writer ([`ReplicaMembershipWriter`]),
    /// from `by`: its turn, the term `leading` (the replica's read of the
    /// fence) tells once the turn is held, and its write under that term.
    /// A refusal, the fence's or the write's, leaves its audit line once
    /// the turn is let go, with `asker` where the write was asked for.
    #[cfg_attr(
        not(any(test, feature = "discovery")),
        expect(dead_code, reason = "the replica's own writer makes the call")
    )]
    fn writer_call<T>(
        &self,
        call: &'static str,
        by: &Principal,
        asker: Option<&str>,
        leading: impl FnOnce() -> Result<u64, Status>,
        write: impl FnOnce(u64) -> Result<T, MembershipWriteRefused>,
    ) -> Result<T, MembershipWriteRefused> {
        match self.in_turn(|| leading().map(write)) {
            // The worker objects hold the write when the call answers, as
            // behind a control call's ([`Self::call`]).
            Ok(Ok(answer)) => Ok(answer),
            Ok(Err(refusal)) => {
                let status = refused(&refusal);
                audit_refusal(call, by, status.code(), status.message(), asker);
                Err(refusal)
            }
            Err(status) => {
                audit_refusal(call, by, status.code(), status.message(), asker);
                Err(MembershipWriteRefused::NotLeading(
                    status.message().to_owned(),
                ))
            }
        }
    }

    /// `RegisterWorker`, admitted from `by` under the term `leading` reads.
    ///
    /// `ADD` (and no mode) wants the URL absent; `REPLACE` wants it standing
    /// and makes its next incarnation; `UPSERT` registers an absent URL and
    /// lays the registration over a standing row: equal facts write no row
    /// and answer the row unmoved, a change of role, wire, models, demand
    /// facts or whether the worker takes images is a new incarnation, any
    /// other change is made in place. Over a draining row an `UPSERT` is the
    /// URL's next incarnation, whatever its facts.
    ///
    /// A registration whose checks are off ends with its row's connect
    /// signal, and so does an `UPSERT` of equal facts where the row waits
    /// for that signal still ([`awaits_connect`]). `wait_ready` is honoured
    /// where the checks are off (the answer is the row after its connect
    /// signal) and refused otherwise: no handshake runs here.
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
        // may be the engine's or a rank's own: the ranks are asked under
        // the address the standing row dials, its key where it names no
        // engine. A caller that did not make the ranks is told so first,
        // as the wide call tells it: an add asks no maker.
        {
            let engine = standing
                .as_ref()
                .map_or(spec.url.as_str(), |row| dial_address(&row.facts));
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
                        // Equal facts write no row. A row that waits for
                        // its connect signal still is sent it again, with
                        // the word of this call's term: a call that waits
                        // answers the row behind it, as a registration that
                        // wrote does.
                        let (mut row, mut epoch, mut at) = (row, self.store.epoch(), self.place());
                        if awaits_connect(&row) {
                            self.connect_row(row.id, row.revision, admitted);
                            if request.wait_ready {
                                (row, epoch, at) =
                                    (self.read(row.id)?, self.store.epoch(), self.place());
                            }
                        }
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
        let wrote = self.lay_row(facts, maker.clone(), admitted);
        let id = wrote.ok_or_else(|| self.unwritten(admitted))?.id();
        // The row, the epoch and the place as the registration's write left
        // them: the answer of a call that does not wait.
        let (mut now, mut epoch, mut at) = (self.read(id)?, self.store.epoch(), self.place());
        // A fresh row is pending: any other status the write left came over
        // from the old incarnation. Read before the connect signal moves it.
        let carried = now.status != WorkerStatus::Pending;
        if !now.facts.health.probes {
            self.connect_row(id, now.revision, admitted);
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
        self.call(PATCH, by, leading, |term| {
            self.relabel(by, &Registrant::Control, term, request)
        })
    }

    /// The write of [`Self::patch`], in the door's turn, from `maker`,
    /// under `admitted`, the term the call was admitted under.
    fn relabel(
        &self,
        by: &Principal,
        maker: &Registrant,
        admitted: u64,
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
                self.patch_row(standing.id, patch, standing.revision, admitted)
                    .ok_or_else(|| self.unwritten(admitted))?;
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
        self.call(DEREGISTER, by, leading, |term| {
            self.remove(by, &Registrant::Control, term, request)
        })
    }

    /// The write of [`Self::deregister`], in the door's turn, from `maker`.
    /// A drain is no exception: it is a write of the row, and it goes
    /// with the word of `admitted`, the term the call was admitted under.
    fn remove(
        &self,
        by: &Principal,
        maker: &Registrant,
        admitted: u64,
        request: &proto::DeregisterWorkerRequest,
    ) -> Result<proto::WorkerEvent, Status> {
        let standing = self.guarded(maker, request.worker_id, request.revision)?;
        let (event, wrote) = if !request.drain {
            self.take_row(standing.id, standing.revision, admitted)
                .ok_or_else(|| self.unwritten(admitted))?;
            (worker_wire::removed(&standing), true)
        } else if standing.status == WorkerStatus::Draining {
            (worker_wire::relabelled(&standing), false)
        } else {
            if !self.drain_row(standing.id, standing.revision, admitted) {
                return Err(self.unwritten(admitted));
            }
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
    fn asks_events(&self, facts: &WorkerFacts) -> bool {
        self.door.asks_events(facts)
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
        self.door.lay_row(facts, maker, self.term)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.door.patch_row(id, patch, revision, self.term)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.door.take_row(id, revision, self.term)
    }

    fn connect(&self, id: WorkerId, revision: u64) {
        self.door.connect_row(id, revision, self.term);
    }

    fn drain(&self, id: WorkerId, revision: u64) -> bool {
        self.door.drain_row(id, revision, self.term)
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
        self.door.audit(REGISTER, self.by, wrote, None);
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
                    // A rank that waited for its connect signal was sent
                    // it again: a call that waits answers the row behind it.
                    let old = old.ok_or_else(moved)?;
                    if request.wait_ready && awaits_connect(old) {
                        worker_wire::relabelled(&self.read(old.id)?)
                    } else {
                        worker_wire::relabelled(old)
                    }
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
#[cfg_attr(
    not(any(test, feature = "discovery")),
    expect(dead_code, reason = "the replica's discovery owns the door")
)]
pub(crate) struct ReplicaMembershipWriter {
    door: Arc<MembershipCalls>,
    by: Principal,
    leading: Box<dyn Fn() -> Result<u64, Status> + Send + Sync>,
}

#[cfg_attr(
    not(any(test, feature = "discovery")),
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
        // The door reads the term and makes the write in its own turn, and
        // writes a refusal's audit line ([`MembershipCalls::writer_call`]).
        self.door
            .writer_call(call, &self.by, asker, || (self.leading)(), write)
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
        self.door.audit(call, &self.by, wrote, asker);
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
    not(any(test, feature = "discovery")),
    expect(dead_code, reason = "the host reads the term for each registration")
)]
struct ReplicaTurn<'a> {
    host: &'a ReplicaMembershipWriter,
    term: u64,
}

impl Turn for ReplicaTurn<'_> {
    fn asks_events(&self, facts: &WorkerFacts) -> bool {
        self.host.door.asks_events(facts)
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
        self.host.door.lay_row(facts, maker, self.term)
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        self.host.door.patch_row(id, patch, revision, self.term)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        self.host.door.take_row(id, revision, self.term)
    }

    fn connect(&self, id: WorkerId, revision: u64) {
        self.host.door.connect_row(id, revision, self.term);
    }

    fn drain(&self, id: WorkerId, revision: u64) -> bool {
        self.host.door.drain_row(id, revision, self.term)
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
/// (`nothing` where it wrote no row: a connect signal has no line of its
/// own) and that place. Never the URL: it may carry a credential.
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
#[path = "../../../tests/unit/service/controller/membership/tests.rs"]
pub(crate) mod tests;

//! The controller's policy source and replica over the operators' door
//! ([`DocumentManager`]).
//!
//! A replica that follows a leader has one writer at this door, the log
//! ([`Generations::following`]): it runs each generation its leader applied,
//! under the leader's number, by this build's own pass over the leader's
//! bytes. Its own file is looked at all the while and shown to this door
//! ([`Generations::shown`]), and applied only while it leads. A leader's
//! stream is told a generation by this door ([`Generations::sent`]): the
//! log holds the number, and the document crosses where it still stands.
//!
//! The door holds who writes, under its one lock. Under a shared lease the
//! source's entry reads the replica's fence there, as a decision reads it
//! ([`Generations::under`]): the source writes while its replica leads.
//! A watch that read under a lead that has since ended cannot publish
//! those bytes ([`ApplyOutcome::Waits`]). A replica that has yet to lead
//! applies nothing of its own.
//!
//! A replica under a shared lease applies no document at its start
//! ([`Generations::defer`]): the start passes its file's document, and it
//! is applied under the replica's first lead where the replica runs
//! nothing by then, at the take or behind the lead's opening
//! ([`Generations::deferred_now`]). So a replica that starts runs no number
//! of its own, and follows its leader's generation whatever its number.
//! A leader runs what its file says: the door counts the takes, and behind
//! one the file's watch reads its file as a start does, though it never
//! moved ([`Generations::takes`]), while the file's document has not stood
//! on this replica since the file took it. The watch of a replica that
//! follows shows the door what its file holds, passed and by its digest
//! and its own line, and applies nothing ([`Generations::shown`]): so the
//! door knows a document that
//! was applied here, or told by a leader, since the file took it. Such a
//! file is a mount that lags: it is left for its next move, and brings
//! nothing older back ([`Generations::source_stood`]).
//!
//! Limits. Under a file, a replica that starts remembers nothing. A number
//! only rises in a process, so a replica that led, and comes back behind a
//! leader that never got its generations, may run a number its leader's is
//! not above: it keeps its own and follows no further until its leader's
//! number passes its own, or until it starts again. Under a spec store the
//! numbers are the directory's,
//! and no standby follows beside one in this build: a store is
//! bound only beside a lease in the process
//! ([`Controller::spec_store`](super::Controller::spec_store)). One process
//! holds a store's directory, under the directory's own lock, as far as the
//! file system's locks reach. Past them a second one's write of a number
//! on file is refused by name, and it takes no write until it starts again
//! ([`SpecStore`]). A file in the store that is no document of this build
//! is listed nowhere: the read call does not answer it. A digest is the
//! applier's ([`Digest`]): it travels with its bytes, and a reader of
//! another build does not recompute it to compare.

use std::{
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use parking_lot::{Mutex, MutexGuard};
use tokio::task::JoinError;
use tracing::{debug, warn};

use super::{
    spec_store::{SpecStore, StoreRefused},
    wire,
    worker_wire::{self, Refused as Unfollowed},
};
#[cfg(test)]
pub(crate) use crate::document::operator::Validate;
pub(crate) use crate::document::operator::{
    ApplyOutcome, Format, SpecRejected, DRY_RUNS_WAITING, MAX_DOCUMENT_BYTES, WRITES_WAITING,
};
use crate::{
    document::{
        operator::{
            above, over_bound, Checked, DocumentManager, DryRun, EnforceOutcome, EnforceRefused,
            Held, Line,
        },
        spec::{AppliedSpec, Digest},
    },
    service::{elector::Fence, proto},
    stages::registry::StageRegistry,
    store::{MemoryStore, Position},
};

/// The lead a source read its bytes under ([`Generations::lead`]): its
/// entry is admitted under that lead, and under no other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Leading {
    /// A replica alone over its lease: its source writes always.
    Alone,
    /// A replica under a shared lease, whose fence showed this term.
    Term(u64),
}

/// What became of the document a door's source handed last: its file at
/// the start, at a leader's look, or at the look of a replica that follows
/// ([`Generations::shown`]).
#[derive(Clone, Copy, Eq, PartialEq)]
enum Handed {
    /// It stood on this replica since it was handed: applied here, or told
    /// by a leader while the replica followed. So too where the source
    /// handed none.
    Stood,
    /// It has not stood here since. The validator admitted it.
    Unstood {
        /// This build's digest of it.
        digest: Digest,
        /// Its own `generation` line, 0 where it states none. A digest
        /// reads no line: a document that states a number stood only
        /// where a generation of that number carried its digest.
        line: u64,
    },
    /// The file held no document that could stand: one a look could not
    /// read or parse, or one the validator's pass refused.
    Undigested,
}

/// The length of a digest's text: `sha256:` and 64 hex digits. No digest is
/// written longer, so a longer guard names none.
const DIGEST_TEXT_LEN: usize = "sha256:".len() + 64;

/// Why a spec call gave no answer of its own: a dry run or a write no
/// verdict, a read no document.
#[derive(Debug)]
pub(crate) enum Unanswered {
    /// The document manager refused the call.
    Manager(EnforceRefused),
    /// As many dry runs wait their turn as may ([`DRY_RUNS_WAITING`]).
    Full,
    /// The pass's thread did not end.
    Unended(JoinError),
    /// As many write calls wait for the door as may ([`WRITES_WAITING`]).
    Waiting,
    /// A write call's thread did not end.
    Unwritten(JoinError),
    /// A generation on file was not read from its file.
    Unread {
        /// The generation.
        generation: u64,
        /// What the read failed by.
        why: String,
    },
}

impl From<EnforceRefused> for Unanswered {
    fn from(refused: EnforceRefused) -> Self {
        Self::Manager(refused)
    }
}

/// The controller's spec source, with its replica's admissions.
pub(crate) struct Generations {
    door: DocumentManager,
    /// The start's document of a replica under a shared lease, in its
    /// source's form, from the start that passed it until the replica's
    /// first lead applies it or refuses it, or a followed generation
    /// stands ([`Generations::defer`], [`Generations::deferred_now`]).
    deferred: Mutex<Option<(Vec<u8>, Format)>>,
    /// A file is this door's source: a start passed its document
    /// ([`Generations::defer`]). Never unset.
    file_sourced: AtomicBool,
    /// The fence of a replica under a shared lease
    /// ([`Generations::under`]): the source's entry is admitted while it
    /// shows the term the entry's bytes were read under. None for a
    /// replica alone over its lease, whose source writes always.
    fence: OnceLock<Arc<Fence>>,
    /// The takes this replica made ([`Generations::at_take`]): behind one
    /// the file's watch reads its file as a start does, while the file's
    /// document has not stood here since the file took it.
    takes: AtomicU64,
    /// What became of the document the source handed last
    /// ([`Generations::source_stood`]). A replica that follows writes it
    /// under the lock of `held`, which a generation enters under: the
    /// file's hand and an entry stand in one order.
    handed: Mutex<Handed>,
    /// The leader's generation this door refused last. The same event sent
    /// again at a retry is refused from here: no second pass and no second
    /// warning.
    unfollowed: Mutex<Option<Unfollowable>>,
}

impl Deref for Generations {
    type Target = DocumentManager;

    fn deref(&self) -> &DocumentManager {
        &self.door
    }
}

impl Generations {
    pub(crate) fn new(store: Arc<MemoryStore>, catalog: Arc<StageRegistry>) -> Self {
        Self::over(DocumentManager::new(store, catalog, Vec::new()))
    }

    /// [`Generations::new`] with the validator as its argument: a case
    /// hands in one that faults, or one that stalls.
    #[cfg(test)]
    pub(crate) fn validating(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
        validate: Validate,
    ) -> Self {
        Self::over(DocumentManager::validating(
            store,
            catalog,
            Vec::new(),
            validate,
        ))
    }

    fn over(door: DocumentManager) -> Self {
        Self {
            door,
            deferred: Mutex::new(None),
            file_sourced: AtomicBool::new(false),
            fence: OnceLock::new(),
            takes: AtomicU64::new(0),
            handed: Mutex::new(Handed::Stood),
            unfollowed: Mutex::new(None),
        }
    }

    /// Applies `document`, as its source wrote it in `format`, read under
    /// the lead that shows now ([`Generations::apply_under`]). A refusal
    /// applies nothing and the running generation stays; what a refusal
    /// means is its source's to say: a start refuses to start, and a source
    /// that keeps running counts it ([`DocumentManager::reject`]).
    pub(crate) fn apply(
        &self,
        document: Vec<u8>,
        format: Format,
    ) -> Result<ApplyOutcome, SpecRejected> {
        self.apply_under(self.lead(), document, format)
    }

    /// [`Generations::apply`] for bytes their source read under `lead`,
    /// taken ahead of the read ([`Generations::lead`]). Under a shared
    /// lease the source writes while its replica leads under that term,
    /// and under no other: bytes read under one lead are not applied under
    /// the next, and their source reads again.
    pub(crate) fn apply_under(
        &self,
        lead: Option<Leading>,
        document: Vec<u8>,
        format: Format,
    ) -> Result<ApplyOutcome, SpecRejected> {
        self.apply_where(lead, document, format, true, |_| None)
    }

    /// [`Generations::apply_under`] where `unread` may say first, by the
    /// standing number, that the document is not applied, and that is the
    /// outcome. The lead is asked three times, under the door's lock: ahead
    /// of the document's pass, which it saves; behind it, since a pass
    /// takes its time; and last where the store publishes, behind the locks
    /// its entry waits for. Where the replica no longer leads as it did at
    /// the read, nothing is published.
    ///
    /// `news` says the bytes are the file as a read just found it: they
    /// are the source's hand ([`Generations::source_stood`]). The start's
    /// deferred bytes were handed at the start, and are no news of the
    /// file: their entry leaves what a look showed since.
    fn apply_where(
        &self,
        lead: Option<Leading>,
        document: Vec<u8>,
        format: Format,
        news: bool,
        unread: impl Fn(u64) -> Option<ApplyOutcome>,
    ) -> Result<ApplyOutcome, SpecRejected> {
        self.door.apply_notifying(
            document,
            format,
            |standing| {
                unread(standing)
                    .or_else(|| (!self.leads(lead)).then_some(ApplyOutcome::Waits(standing)))
            },
            |store, spec| store.apply_admitted(spec, || self.leads(lead)),
            |passed| {
                if news {
                    #[cfg(test)]
                    tests::count_replica_notice(self, false);
                    match passed {
                        Some((digest, line)) => self.sourced(digest, line),
                        None => *self.handed.lock() = Handed::Undigested,
                    }
                }
            },
            |generation, digest| self.stands(generation, digest),
        )
    }

    /// Whether this replica leads now as it did when a source took `lead`
    /// ahead of its read: always for a replica alone over its lease, and
    /// while the fence shows that same term for one under a shared lease.
    fn leads(&self, lead: Option<Leading>) -> bool {
        lead.is_some() && self.lead() == lead
    }

    /// Binds the controller's spec store, unless its file source is
    /// already bound or waiting for the first lead.
    pub(crate) fn bind_store(&self, dir: SpecStore) -> Result<(), StoreRefused> {
        self.door.bind_store(dir, || self.deferred.lock().is_some())
    }

    /// [`DocumentManager::dry_run`] off the runtime's workers, one at a time. A
    /// document at the bound validates for tens of milliseconds and holds
    /// tens of mebibytes, and the size bound budgets one pass beside an
    /// apply's: a second call waits its turn, in the order the calls came,
    /// and a call past the [`DRY_RUNS_WAITING`] that wait is refused. The
    /// pass holds the turn, so a caller that leaves frees it no earlier
    /// than its pass ends. The guard is read behind the wait, with the
    /// pass: a call that waited through an apply is judged against what
    /// stands after it. A document over the bound takes no turn: it is
    /// refused by its size, unread, and waits on no pass. A guard longer
    /// than a digest's text takes none either: it names no digest, the
    /// guard refuses it at once, and no turn's wait holds it. Each of the
    /// two still reads on one side of a store's bind, as every dry run
    /// does: a bind in flight at a start holds it until the store is bound.
    pub(crate) async fn dry_run_in_turn(
        self: &Arc<Self>,
        expected: String,
        document: Vec<u8>,
        format: Format,
    ) -> Result<DryRun, Unanswered> {
        if over_bound(&document).is_some() || expected.len() > DIGEST_TEXT_LEN {
            return self
                .dry_run(&expected, &document, format)
                .map_err(Into::into);
        }
        let Some(turn) = self.dry_turn.take().await else {
            return Err(Unanswered::Full);
        };
        let door = Arc::clone(self);
        let pass = tokio::task::spawn_blocking(move || {
            let _held = turn;
            door.dry_run(&expected, &document, format)
                .map_err(Into::into)
        });
        pass.await
            .unwrap_or_else(|join| Err(Unanswered::Unended(join)))
    }

    /// [`DocumentManager::enforce`] off the runtime's workers, behind the write
    /// calls that came first: an apply validates for tens of milliseconds
    /// and syncs its file. No more calls wait than [`WRITES_WAITING`]: one
    /// past them is refused. A document over the bound and a guard longer
    /// than a digest's text wait on nothing: each is refused at once, the
    /// document by its size, unread.
    pub(crate) async fn enforce_in_turn(
        self: &Arc<Self>,
        expected: String,
        document: Vec<u8>,
        format: Format,
    ) -> Result<EnforceOutcome, Unanswered> {
        if expected.len() > DIGEST_TEXT_LEN {
            // It names no digest: the guard refuses it.
            self.guarded(&expected)?;
        }
        if let Some(len) = over_bound(&document) {
            let standing = self.guarded(&expected)?;
            let ran = Err(self.reject(SpecRejected::TooLarge(len)));
            return Ok(EnforceOutcome {
                standing,
                ran,
                needs: None,
            });
        }
        self.in_turn(move |door| {
            door.enforce(&expected, document, format)
                .map_err(Into::into)
        })
        .await
    }

    /// [`DocumentManager::roll_back`] off the runtime's workers, in the write
    /// calls' turn ([`Generations::in_turn`]).
    pub(crate) async fn roll_back_in_turn(
        self: &Arc<Self>,
        wanted: u64,
    ) -> Result<EnforceOutcome, Unanswered> {
        self.in_turn(move |door| door.roll_back(wanted).map_err(Into::into))
            .await
    }

    /// Runs `write` off the runtime's workers, behind the write calls that
    /// came first: a call awaits its turn, in the order the calls came,
    /// and its thread starts once the turn is its own, so no thread waits
    /// for the door. No more calls wait than [`WRITES_WAITING`]: one past
    /// them is refused. A caller that leaves while its call waits writes
    /// nothing and frees its place. Once its thread started, the write
    /// runs to its end whoever left, and holds the turn and the place
    /// until then.
    async fn in_turn(
        self: &Arc<Self>,
        write: impl FnOnce(&Self) -> Result<EnforceOutcome, Unanswered> + Send + 'static,
    ) -> Result<EnforceOutcome, Unanswered> {
        let Some(turn) = self.write_turn.take().await else {
            return Err(Unanswered::Waiting);
        };
        let door = Arc::clone(self);
        let written = tokio::task::spawn_blocking(move || {
            let _held = turn;
            write(&door)
        });
        written
            .await
            .unwrap_or_else(|join| Err(Unanswered::Unwritten(join)))
    }

    /// Whether the controller has a spec source: a file's document was
    /// applied, a start under a shared lease passed its file's, or a spec
    /// store is bound. The spec calls are served and the spec's series
    /// rendered from then on.
    pub(crate) fn bound(&self) -> bool {
        self.writable() || self.file_sourced.load(Ordering::Relaxed) || self.held.lock().is_some()
    }

    /// `GetPolicySpec`: the standing generation's document from memory
    /// ([`DocumentManager::document`]), or, under a spec store, a generation on
    /// file, read from its file off the runtime's workers and answered with
    /// the digest and the time it is listed under. It takes no lock an
    /// apply holds: the read waits on no validation and on no file's sync.
    /// A file an apply removed since the list was read is not held. No turn
    /// bounds the reads: one holds a file's bytes, the size bound at most,
    /// and the runtime's blocking threads are their bound.
    pub(crate) async fn read(&self, wanted: u64) -> Result<proto::PolicySpecDocument, Unanswered> {
        let not_held = match self.document(wanted) {
            Ok(document) => return Ok(wire::spec_document(document)),
            Err(not_held) => not_held,
        };
        let generation = not_held.asked;
        let Some(read) = self.retained(generation) else {
            return Err(EnforceRefused::NotHeld(not_held).into());
        };
        let read = tokio::task::spawn_blocking(read);
        let unread = |why: String| Unanswered::Unread { generation, why };
        match read.await {
            Ok(Ok(document)) => Ok(wire::spec_document(document)),
            Ok(Err(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(EnforceRefused::NotHeld(self.not_held(generation)).into())
            }
            Ok(Err(error)) => Err(unread(error.to_string())),
            Err(join) => Err(unread(join.to_string())),
        }
    }
}

/// Why a leader's generation is not followed, in the stream's words: its
/// number is not above the one this replica runs.
const NOT_ABOVE: &str = "a generation that is not above the one this replica runs";

/// The path of a generation event's number.
const NUMBER: &str = "generation.status.generation";

/// The replication half: what a leader's stream is told of a generation,
/// and how a standby applies one.
impl Generations {
    /// The event a standby is sent of `generation`, a number the log holds:
    /// with the document where this door still holds that generation, and
    /// the number alone where a later one replaced it. An apply swaps the
    /// held document under the lock this takes, with its call of the
    /// store's entry: an entry of the log is never read here ahead of its
    /// document.
    pub(crate) fn sent(&self, generation: u64) -> proto::GenerationEvent {
        let held = self.held.lock();
        match held.as_ref().filter(|held| held.generation == generation) {
            Some(held) => wire::generation_event(held.document()),
            None => worker_wire::generation_told(generation),
        }
    }

    /// The standing generation's event, for the snapshot a standby is
    /// restored from; none where no document is held for it.
    pub(crate) fn standing_sent(&self) -> Option<proto::GenerationEvent> {
        let held = self.held.lock();
        let standing = self.generation();
        held.as_ref()
            .filter(|held| held.generation == standing)
            .map(|held| wire::generation_event(held.document()))
    }

    /// The start's document of a replica under a shared lease: passed here
    /// and not applied. A replica that applied it would run a number of
    /// its own, and a number only rises: it could follow no leader whose
    /// generation is not above that number. So it starts under no
    /// generation and takes its leader's, whatever the number.
    ///
    /// The pass is the start's whole word on the document: one that does
    /// not run refuses the start, as an applied one does. The bytes are
    /// kept for the replica's first lead ([`Generations::deferred_now`]),
    /// and the door has its source from here on ([`Generations::bound`]).
    pub(crate) fn defer(&self, document: Vec<u8>, format: Format) -> Result<(), SpecRejected> {
        let applying = self.applying.lock();
        if applying.is_some() {
            return Err(SpecRejected::SecondSource);
        }
        let standing = self.standing();
        let Checked { digest, line, next } = self.pass(standing, &document, format, Line::Claim)?;
        if let Some(next) = next {
            above(standing.0, next.generation())?;
        }
        self.sourced(digest, line);
        *self.deferred.lock() = Some((document, format));
        self.file_sourced.store(true, Ordering::Relaxed);
        // Until the document lands a request is no decision: the unbound
        // start says so, where the start of a store with no source sheds
        // by the binding.
        if standing.0 == 0 {
            self.store.snapshot().spec().mark_document_awaited();
        }
        drop(applying);
        Ok(())
    }

    /// Puts the source's entry under `fence`, the fence of a replica under
    /// a shared lease: from here the source writes while the fence shows
    /// the term its bytes were read under, and at no other time
    /// ([`Generations::apply_under`]). A replica that has not led yet, one
    /// that follows a leader, and one whose lead ended apply nothing of
    /// their own, whatever their file's watch read ahead of a look. The
    /// first fence handed stands.
    pub(crate) fn under(&self, fence: Arc<Fence>) {
        let _ = self.fence.set(fence);
    }

    /// Whether the source writes now: always for a replica alone over its
    /// lease, and while the fence shows a term for one under a shared lease
    /// ([`Generations::under`]). A case's question: the source takes its
    /// lead ahead of a read ([`Generations::lead`]).
    #[cfg(test)]
    pub(crate) fn source_writes(&self) -> bool {
        self.lead().is_some()
    }

    /// The lead this replica writes under now: a source takes it ahead of
    /// its read and hands it in with the bytes
    /// ([`Generations::apply_under`]). None where the replica is under a
    /// shared lease and its fence shows no term.
    pub(crate) fn lead(&self) -> Option<Leading> {
        match self.fence.get() {
            None => Some(Leading::Alone),
            Some(fence) => fence.leader_term(Instant::now()).map(Leading::Term),
        }
    }

    /// This replica took the lease: the elector calls this at each take,
    /// behind the lease's write and behind the fence's stamp
    /// ([`Elector::at_take`](crate::service::elector::Elector::at_take)).
    ///
    /// The take is counted, and the file's watch reads the file behind it
    /// as a start does, though it never moved, while the file's document
    /// has not stood on this replica since the file took it
    /// ([`SpecFile`](super::spec_file::SpecFile),
    /// [`Generations::source_stood`]): a leader runs what its file says,
    /// and a mount that lags brings nothing older back. The standing
    /// document again is no change, another one is the next generation.
    /// The start's deferred document goes through the source's entry here
    /// ([`Generations::deferred_now`]).
    pub(crate) fn at_take(&self) {
        self.takes.fetch_add(1, Ordering::Relaxed);
        self.deferred_now();
    }

    /// The start's deferred document through the source's entry, where one
    /// still waits: at each take, and by the file's watch ahead of a look.
    ///
    /// Where the replica runs no generation and its fence shows a term,
    /// the document is applied as the replica's first, under that term. A
    /// take whose window is past, or whose lead still waits to open,
    /// publishes nothing: the document waits on, and the watch brings it
    /// once the fence is open ([`Generations::nap`]). A replica that leads
    /// decides under no unbound generation all the same: until its first
    /// document lands a request is no decision, answered `Busy` with no
    /// reason ([`AppliedSpec::mark_document_awaited`]).
    ///
    /// Behind a generation the replica followed, nothing is applied: the
    /// start's bytes are as old as the start, and the watch reads the file
    /// as it stands.
    ///
    /// The bytes are let go once a generation stands, or once the door
    /// refuses them. A refusal is counted, though the same bytes passed at
    /// the start. A refused document is not handed again at the next
    /// opening: the watch's next look reads the file as it stands.
    pub(crate) fn deferred_now(&self) {
        let Some((document, format)) = self.deferred.lock().clone() else {
            return;
        };
        let followed = |standing| (standing != 0).then_some(ApplyOutcome::Unchanged(standing));
        match self.apply_where(self.lead(), document, format, false, followed) {
            // No term shows, or the lead ended under the pass: it waits on.
            Ok(ApplyOutcome::Waits(_)) => return,
            Ok(ApplyOutcome::Generation(_) | ApplyOutcome::Unchanged(_)) => {}
            Err(rejected) => {
                self.reject(rejected);
            }
        }
        *self.deferred.lock() = None;
    }

    /// How long the file's watch sleeps ahead of its next step, `most` at
    /// the longest. Where the start's document waits it is no time while
    /// the fence shows a term, and no longer than to the opening of a lead
    /// that is still ahead: the document is applied as the lead opens, so
    /// the moment a new leader decides nothing is the document's pass.
    pub(crate) fn nap(&self, most: Duration) -> Duration {
        let Some(fence) = self.fence.get() else {
            return most;
        };
        if self.deferred.lock().is_none() {
            return most;
        }
        let now = Instant::now();
        if fence.leader_term(now).is_some() {
            return Duration::ZERO;
        }
        fence
            .turns_at(now)
            .map_or(most, |opens| opens.saturating_duration_since(now).min(most))
    }

    /// The takes this replica made so far. The file's watch reads its file
    /// behind one, as a start does, though the file never moved, while its
    /// document has not stood on this replica since the file took it
    /// ([`Generations::source_stood`]).
    pub(crate) fn takes(&self) -> u64 {
        self.takes.load(Ordering::Relaxed)
    }

    /// Whether the document the source handed last stood on this replica
    /// since it was handed: applied here, or told by a leader while the
    /// replica followed, at its join or later. A door whose source handed
    /// none has none that waits.
    ///
    /// The source hands a document at its start, at a leader's look, and
    /// at the look of a replica that follows, which applies nothing
    /// ([`Generations::shown`]). So the door holds the order of the file's
    /// moves and of the generations. A document that has not stood since
    /// its file took it is the newer word, a rollout's new file or a
    /// rollback: the watch reads the file behind a take though it never
    /// moved. One that stood since is the standing document or an older
    /// one, a mount that lags: the file is left for its next move, and a
    /// take brings nothing older back.
    pub(crate) fn source_stood(&self) -> bool {
        *self.handed.lock() == Handed::Stood
    }

    /// The file as a replica that follows found it at a look: the door is
    /// told what the file holds and applies nothing. The document goes
    /// through the validator's pass, as at a leader's read, and the door
    /// keeps its digest and its own `generation` line. None of the bytes
    /// are kept: a take reads the file again, under its own term, through
    /// the source's entry. `None` is a file the look could not read. It, a
    /// document that does not parse and one the pass refuses count as a
    /// document that never stood, and the take tells them: equal digests
    /// say nothing of a document that was not validated. A replica that
    /// follows says nothing of its file. A leader's look that could read
    /// no file says so here as well, behind its own telling of it.
    ///
    /// The document is compared with the standing generation under the
    /// lock of the bytes held, which a generation enters under: the hand
    /// and an entry stand in one order, whichever thread runs first. The
    /// pass runs ahead of that lock: it takes its time. A document the
    /// pass found standing stood, whichever generation stands under the
    /// lock by then: one that entered under the pass replaced it.
    ///
    /// A file that does not run leaves one debug line here, at each state
    /// of the file: no count and no warning, which are the take's.
    pub(crate) fn shown(&self, document: Option<&[u8]>, format: Format) {
        let passed = document.and_then(|document| {
            self.pass(self.standing(), document, format, Line::Claim)
                .inspect_err(|rejected| {
                    debug!(%rejected, "policy spec file does not run on a standby: its take tells it");
                })
                .ok()
        });
        let _entry = self.held.lock();
        let (number, standing) = self.standing();
        *self.handed.lock() = match passed {
            Some(Checked { next: None, .. }) => Handed::Stood,
            Some(Checked { digest, line, .. })
                if standing == Some(digest) && (line == 0 || line == number) =>
            {
                Handed::Stood
            }
            Some(Checked { digest, line, .. }) => Handed::Unstood { digest, line },
            None => Handed::Undigested,
        };
    }

    /// The source handed a document that passed, by its digest and its
    /// own line: it has not stood here since.
    fn sourced(&self, digest: Digest, line: u64) {
        *self.handed.lock() = Handed::Unstood { digest, line };
    }

    /// A document stands on this replica under `generation`, by its
    /// digest: the source's last one stood, where it is that one. A
    /// document that states its own number is that one under that number
    /// alone. The caller holds the lock of the bytes held.
    fn stands(&self, generation: u64, digest: Digest) {
        #[cfg(test)]
        tests::count_replica_notice(self, true);
        let mut handed = self.handed.lock();
        let stood = matches!(
            *handed,
            Handed::Unstood { digest: held, line }
                if held == digest && (line == 0 || line == generation)
        );
        if stood {
            *handed = Handed::Stood;
        }
    }

    /// A leader's generation as the replication stream tells it, passed
    /// and not yet applied: the log is the one writer of a standby's door.
    /// The document's pass runs here, under the door's lock, and the answer
    /// holds that lock until it lands ([`Following::land`]).
    ///
    /// The number says what the event does. Above the standing one, the
    /// document runs under it ([`Line::Record`]) whatever its digest: a
    /// replica that read the same file under a lower number takes its
    /// leader's. At the standing number with the standing bytes, and with
    /// no document (the generation stood no longer when its leader sent
    /// the event), nothing changes. No other event is followed: the
    /// store's number only rises, so a replica that runs a generation its
    /// leader's is not above keeps its own, says both and leaves the synced
    /// state, as it does for a document its build does not run. The bytes
    /// are compared, never the digests: a digest is its build's.
    ///
    /// An event refused here is kept: the same one sent again at a retry
    /// of the stream is refused from memory, with no second pass and no
    /// second warning.
    pub(crate) fn following(
        &self,
        told: &proto::GenerationEvent,
    ) -> Result<Following<'_>, Unfollowed> {
        let number = told.status.as_ref().map_or(0, |status| status.generation);
        if number == 0 {
            return Err(Unfollowed {
                path: NUMBER,
                why: "a generation's event names its number",
            });
        }
        let applying = self.applying.lock();
        let following = |applying, change| Following {
            door: self,
            _applying: applying,
            number,
            change,
        };
        let Some(push) = &told.document else {
            return Ok(following(applying, None));
        };
        let Some(format) = Format::named(&push.format) else {
            return Err(Unfollowed {
                path: "generation.document.format",
                why: "a form this build does not know",
            });
        };
        if applying.is_some() {
            return Err(Unfollowed {
                path: "generation.document",
                why: "a spec store is bound here, and no standby follows beside one",
            });
        }
        let (standing, digest) = self.standing();
        // The same event again, at a retry of the stream that it halted.
        let kept = self.unfollowed.lock();
        if let Some(again) = kept.as_ref().filter(|kept| {
            (kept.ran, kept.told, kept.format) == (standing, number, format)
                && kept.document == push.document
        }) {
            debug!(
                generation = number,
                standing, "a leader's generation this replica refused is sent again"
            );
            return Err(again.refusal);
        }
        drop(kept);
        let refuse = |refusal: Unfollowed| {
            *self.unfollowed.lock() = Some(Unfollowable {
                ran: standing,
                told: number,
                format,
                document: push.document.clone(),
                refusal,
            });
            refusal
        };
        if number <= standing {
            // The held bytes of the standing generation, compared as bytes.
            let same = number == standing
                && self.held.lock().as_ref().is_some_and(|held| {
                    held.generation == standing
                        && held.format == format
                        && held.document == push.document
                });
            if same {
                return Ok(following(applying, None));
            }
            warn!(
                generation = number,
                digest = %push.digest,
                standing,
                standing_digest = ?digest,
                "a leader's generation is not above the one this replica runs: the replica keeps its own"
            );
            return Err(refuse(Unfollowed {
                path: NUMBER,
                why: NOT_ABOVE,
            }));
        }
        // The standing digest is no part of this pass: the leader's number
        // is taken whatever the document under it.
        let rule = Line::Record(number);
        let (digest, next) = match self.pass((standing, None), &push.document, format, rule) {
            Ok(Checked {
                digest,
                next: Some(next),
                ..
            }) => (digest, next),
            Ok(Checked { next: None, .. }) => return Ok(following(applying, None)),
            Err(rejected) => {
                warn!(
                    generation = number,
                    error = %rejected,
                    "a leader's generation does not run on this replica: the running generation stays"
                );
                return Err(refuse(Unfollowed {
                    path: "generation.document",
                    why: "a document this replica's build does not run",
                }));
            }
        };
        // The leader's own time where it states one: the read call answers
        // one time for a generation on every replica.
        let applied = told
            .status
            .as_ref()
            .and_then(|status| u64::try_from(status.applied_unix_micros).ok())
            .filter(|micros| *micros != 0)
            .map_or_else(SystemTime::now, |micros| {
                UNIX_EPOCH + Duration::from_micros(micros)
            });
        let kept = Held {
            generation: number,
            digest,
            document: push.document.clone(),
            format,
            applied,
        };
        Ok(following(applying, Some((next, kept))))
    }
}

/// A leader's generation this door refused, as the event stood: the number
/// this replica ran, the leader's number and the document in its form, with
/// the refusal it was given. The bytes are the key, never a digest: a digest
/// is its build's.
struct Unfollowable {
    ran: u64,
    told: u64,
    format: Format,
    document: Vec<u8>,
    refusal: Unfollowed,
}

/// A leader's generation that passed here and waits to land
/// ([`Generations::following`]). It holds the door's lock: no apply comes
/// between its pass and its entry.
pub(crate) struct Following<'a> {
    door: &'a Generations,
    _applying: MutexGuard<'a, Option<SpecStore>>,
    number: u64,
    /// The generation to enter, with the bytes it runs from; none where the
    /// event changes nothing here.
    change: Option<(AppliedSpec, Held)>,
}

impl Following<'_> {
    /// Applies it. `at` is the place of the leader's event, which the
    /// store's entry journals; none beside a restore, whose snapshot stands
    /// at one place for all it holds. An event that changes nothing
    /// journals its place alone.
    pub(crate) fn land(self, at: Option<Position>) -> Result<(), Unfollowed> {
        let Self {
            door,
            _applying,
            number,
            change,
        } = self;
        match (change, at) {
            (Some((next, kept)), at) => door
                .enter(
                    next,
                    kept,
                    |store, spec| {
                        match at {
                            Some(at) => store.follow_generation(at, spec),
                            None => store.apply(spec),
                        }
                        .map(|()| true)
                    },
                    |generation, digest| door.stands(generation, digest),
                )
                .map(drop)
                .map_err(|stale| {
                    warn!(
                        generation = number,
                        standing = stale.standing,
                        "a leader's generation is not above the one this replica runs: the replica keeps its own"
                    );
                    Unfollowed {
                        path: NUMBER,
                        why: NOT_ABOVE,
                    }
                }),
            (None, Some(at)) => {
                door.store.follow_generation_place(at, number);
                Ok(())
            }
            (None, None) => Ok(()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        cell::{Cell, RefCell},
        collections::BTreeSet,
        io::Write as _,
        path::Path,
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use tokio::task::JoinSet;

    use super::*;
    pub(crate) use crate::document::operator::test_support::{
        bound_to, faulting, long_text_under_aliases, padded, with_line, LEAST_LOAD, RANDOM,
    };
    use crate::{
        document::{
            operator::{
                needs_rows,
                test_support::{document_of, standing, stock},
                ApplyCounts, NotHeld, OnFile,
            },
            spec::{PolicySpec, SpecError},
        },
        service::{
            controller::{
                spec_store::tests::{name_of, names},
                wire::unix_micros,
            },
            elector::{ElectionSettings, Elector, MemoryLease},
        },
    };

    thread_local! {
        static REPLICA_NOTICES: Cell<Option<u32>> = const { Cell::new(None) };
    }

    pub(super) fn count_replica_notice(door: &Generations, holds_document: bool) {
        REPLICA_NOTICES.with(|notices| {
            let Some(count) = notices.get() else {
                return;
            };
            assert!(door.applying.try_lock().is_none());
            assert_eq!(door.held.try_lock().is_none(), holds_document);
            assert_eq!(door.generation(), u64::from(holds_document));
            notices.set(Some(count + 1));
        });
    }

    fn generations() -> (Arc<MemoryStore>, Generations) {
        let store = Arc::new(MemoryStore::new());
        (Arc::clone(&store), Generations::new(store, stock()))
    }

    /// `document` applied as YAML.
    fn yaml(
        generations: &Generations,
        document: impl Into<Vec<u8>>,
    ) -> Result<ApplyOutcome, SpecRejected> {
        generations.apply(document.into(), Format::Yaml)
    }

    /// An election no case outlasts: a replica's candidate alone over its
    /// lease, with an hour's renew deadline. Its fence shows a term from
    /// its first step to its release.
    pub(crate) fn election() -> Elector {
        election_under(Duration::from_secs(3_600))
    }

    /// A replica's candidate alone over its lease, with `deadline` as its
    /// renew deadline. Under a short one a case closes the fence by the
    /// clock, between two renewals of one term.
    pub(crate) fn election_under(deadline: Duration) -> Elector {
        let settings = ElectionSettings {
            duration: 2 * deadline,
            renew_period: Duration::from_secs(1).min(deadline / 4),
            renew_deadline: deadline,
        };
        Elector::alone("controller-0@127.0.0.1:1", settings).expect("the settings are in order")
    }

    /// A candidate over a lease that is handed in, with `deadline` as its
    /// renew deadline: its take has no holder to wait for, so its lead
    /// opens one renew deadline behind the take.
    pub(crate) fn candidate_under(deadline: Duration) -> Elector {
        let settings = ElectionSettings {
            duration: 2 * deadline,
            renew_period: Duration::from_secs(1).min(deadline / 4),
            renew_deadline: deadline,
        };
        Elector::new(
            Box::new(MemoryLease::new()),
            "controller-0@127.0.0.1:1",
            settings,
        )
        .expect("the settings are in order")
    }

    /// A validator's gate: once armed, the next pass to enter stands at it
    /// until the case releases it, and the gate counts the passes that
    /// entered since.
    pub(crate) struct Gate {
        entered: AtomicU64,
        armed: AtomicBool,
        released: AtomicBool,
    }

    impl Gate {
        pub(crate) const fn new() -> Self {
            Self {
                entered: AtomicU64::new(0),
                armed: AtomicBool::new(false),
                released: AtomicBool::new(false),
            }
        }

        pub(crate) fn arm(&self) {
            self.entered.store(0, Ordering::SeqCst);
            self.armed.store(true, Ordering::SeqCst);
        }

        pub(crate) fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
        }

        pub(crate) fn entered(&self) -> u64 {
            self.entered.load(Ordering::SeqCst)
        }

        /// The passes that entered a pace after the first did: the first
        /// stands at the gate, and one beside it had the pace to enter.
        pub(crate) async fn entered_a_pace_on(&self) -> u64 {
            let started = Instant::now();
            while self.entered() == 0 {
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "no pass reached the validator"
                );
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
            self.entered()
        }
    }

    /// A case's own gate. A validator is a plain function, so a case names
    /// its gate by a type of its own ([`gate`]).
    pub(crate) trait Gated {
        fn gate() -> &'static Gate;
    }

    /// Declares `$name`, a type whose gate is the case's own.
    macro_rules! gate {
        ($name:ident) => {
            struct $name;
            impl Gated for $name {
                fn gate() -> &'static Gate {
                    static GATE: Gate = Gate::new();
                    &GATE
                }
            }
        };
    }
    pub(crate) use gate;

    /// The stock validator behind `G`'s gate: a pass that finds the gate
    /// armed stands there until it is released, ten seconds at the most.
    fn stalling<G: Gated>(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let gate = G::gate();
        gate.entered.fetch_add(1, Ordering::SeqCst);
        if gate.armed.swap(false, Ordering::SeqCst) {
            for _ in 0..10_000 {
                if gate.released.load(Ordering::SeqCst) {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }

    /// A door over a store of its own whose validator stands at `G`'s gate.
    pub(crate) fn gated<G: Gated>() -> Arc<Generations> {
        let store = Arc::new(MemoryStore::new());
        Arc::new(Generations::validating(store, stock(), stalling::<G>))
    }

    /// One dry run at a time: while a pass stands in the validator a second
    /// call's pass does not enter it, and both answer once the first ends.
    #[tokio::test]
    async fn a_second_dry_run_waits_its_turn() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        let run = |document: &str| {
            generations.dry_run_in_turn(String::new(), document.into(), Format::Yaml)
        };
        gate.arm();
        // The first pass stands in the validator; a pace later the passes
        // that entered are counted. The validator is released before the
        // count is judged, so a count that fails ends the case.
        let beside = async {
            let entered = gate.entered_a_pace_on().await;
            gate.release();
            entered
        };
        let (first, second, entered) = tokio::join!(run(LEAST_LOAD), run(RANDOM), beside);
        assert_eq!(entered, 1, "a second dry run ran beside the first");
        for answer in [first, second] {
            let answer = answer.expect("a verdict");
            assert_eq!(answer.would_run.map(|(number, _)| number), Ok(1));
        }
        assert_eq!(gate.entered(), 2);
    }

    /// The pass holds the turn, not its caller: a caller that leaves while
    /// its pass stands in the validator frees no turn, and the next call's
    /// pass enters once that one ends.
    #[tokio::test]
    async fn a_caller_that_leaves_frees_no_turn_before_its_pass_ends() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        let run = |document: &str| {
            generations.dry_run_in_turn(String::new(), document.into(), Format::Yaml)
        };
        gate.arm();
        // The first caller leaves a pace after its pass entered.
        tokio::select! {
            answer = run(LEAST_LOAD) => panic!("the first pass ended: {answer:?}"),
            _ = gate.entered_a_pace_on() => {}
        }
        let beside = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let entered = gate.entered();
            gate.release();
            entered
        };
        let (second, entered) = tokio::join!(run(RANDOM), beside);
        assert_eq!(entered, 1, "the turn was freed under a pass that still ran");
        let second = second.expect("a verdict");
        assert_eq!(second.would_run.map(|(number, _)| number), Ok(1));
        assert_eq!(gate.entered(), 2);
    }

    /// The guard is read behind the wait: a guarded call that waits its turn
    /// through an apply is refused by the generation that stands after it,
    /// as the same call asked then is.
    #[tokio::test]
    async fn a_guard_is_read_once_the_turn_comes() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Ok(ApplyOutcome::Generation(1))
        );
        let (_, first) = generations.standing();
        let first = first.expect("a digest stands").to_string();
        let run = |expected: &str| {
            generations.dry_run_in_turn(expected.to_owned(), RANDOM.into(), Format::Yaml)
        };
        gate.arm();
        // While the first pass stands in the validator the source applies
        // the next generation, and the guarded call waits through it.
        let beside = async {
            gate.entered_a_pace_on().await;
            let applied = yaml(&generations, RANDOM);
            gate.release();
            applied
        };
        let (unguarded, guarded, applied) = tokio::join!(run(""), run(&first), beside);
        assert_eq!(applied, Ok(ApplyOutcome::Generation(2)));
        // The first call read generation 1 before its pass stood.
        let unguarded = unguarded.expect("a verdict");
        assert_eq!(unguarded.would_run.map(|(number, _)| number), Ok(2));
        assert!(
            matches!(
                guarded,
                Err(Unanswered::Manager(EnforceRefused::Guarded(2, Some(_))))
            ),
            "{guarded:?}"
        );
    }

    /// A document over the bound takes no turn: it is refused by its size
    /// while another's pass stands in the validator.
    #[tokio::test]
    async fn a_document_over_the_bound_waits_on_no_pass() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        let over = MAX_DOCUMENT_BYTES + 1;
        let document = vec![b'#'; usize::try_from(over).expect("the bound fits")];
        gate.arm();
        let beside = async {
            gate.entered_a_pace_on().await;
            let call = generations.dry_run_in_turn(String::new(), document, Format::Yaml);
            let refused = tokio::time::timeout(Duration::from_secs(5), call).await;
            gate.release();
            refused
        };
        let first = generations.dry_run_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml);
        let (first, refused) = tokio::join!(first, beside);
        first.expect("a verdict");
        let refused = refused
            .expect("the refusal waits on no pass")
            .expect("a verdict");
        assert_eq!(refused.would_run, Err(SpecRejected::TooLarge(over)));
        // The guard is read ahead of the size: where nothing stands a guard
        // names no standing digest, under the bound and over it.
        let guard = format!("sha256:{}", "0".repeat(64));
        let small = generations.dry_run(&guard, LEAST_LOAD.as_bytes(), Format::Yaml);
        assert!(
            matches!(small, Err(EnforceRefused::Guarded(0, None))),
            "{small:?}"
        );
        let document = vec![b'#'; usize::try_from(over).expect("the bound fits")];
        let large = generations
            .dry_run_in_turn(guard, document, Format::Yaml)
            .await;
        assert!(
            matches!(
                large,
                Err(Unanswered::Manager(EnforceRefused::Guarded(0, None)))
            ),
            "{large:?}"
        );
    }

    /// A guard longer than a digest's text takes no turn: it names no
    /// digest, and the guard refuses it while another's pass stands in the
    /// validator.
    #[tokio::test]
    async fn a_guard_longer_than_a_digest_waits_on_no_pass() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Ok(ApplyOutcome::Generation(1))
        );
        let (_, standing) = generations.standing();
        let standing = standing.expect("a digest stands").to_string();
        assert_eq!(standing.len(), DIGEST_TEXT_LEN);
        let longer = format!("{standing}0");
        gate.arm();
        let beside = async {
            gate.entered_a_pace_on().await;
            let call = generations.dry_run_in_turn(longer, RANDOM.into(), Format::Yaml);
            let refused = tokio::time::timeout(Duration::from_secs(5), call).await;
            gate.release();
            refused
        };
        let first = generations.dry_run_in_turn(String::new(), RANDOM.into(), Format::Yaml);
        let (first, refused) = tokio::join!(first, beside);
        first.expect("a verdict");
        let refused = refused.expect("the refusal waits on no pass");
        assert!(
            matches!(
                refused,
                Err(Unanswered::Manager(EnforceRefused::Guarded(1, Some(_))))
            ),
            "{refused:?}"
        );
    }

    /// No more dry runs wait than [`DRY_RUNS_WAITING`]: the call past them
    /// is refused at once, a waiting call that ends frees its place, and
    /// the ones let in all answer.
    #[tokio::test]
    async fn a_dry_run_past_the_waiting_ones_is_refused() {
        gate!(Here);
        let (gate, generations) = (Here::gate(), gated::<Here>());
        let run = |door: Arc<Generations>| async move {
            door.dry_run_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml)
                .await
        };
        // The places come to `free` once the calls let in were polled.
        let places = |door: Arc<Generations>, free: usize| async move {
            let started = Instant::now();
            while door.dry_turn.available() != free {
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "the places did not come to {free} free"
                );
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        };
        gate.arm();
        let mut let_in = JoinSet::new();
        for _ in 0..DRY_RUNS_WAITING {
            let_in.spawn(run(Arc::clone(&generations)));
        }
        let last = let_in.spawn(run(Arc::clone(&generations)));
        places(Arc::clone(&generations), 0).await;
        let past = run(Arc::clone(&generations)).await;
        assert!(matches!(past, Err(Unanswered::Full)), "{past:?}");
        // A waiting call that ends leaves the wait: its place is one again.
        last.abort();
        places(Arc::clone(&generations), 1).await;
        let_in.spawn(run(Arc::clone(&generations)));
        places(Arc::clone(&generations), 0).await;
        gate.release();
        let (mut answered, mut left) = (0, 0);
        while let Some(answer) = let_in.join_next().await {
            match answer {
                Ok(answer) => {
                    answer.expect("a verdict");
                    answered += 1;
                }
                Err(gone) => {
                    assert!(gone.is_cancelled(), "{gone}");
                    left += 1;
                }
            }
        }
        assert_eq!((answered, left), (DRY_RUNS_WAITING + 1, 1));
        // A call that ran gave its place back, as one that left did.
        assert_eq!(generations.dry_turn.available(), DRY_RUNS_WAITING + 1);
    }

    /// A dry run answers what an apply would do at that instant and applies,
    /// holds and counts nothing. The number is the one an apply would mint
    /// now and it is not reserved: two dry runs answer one number, and an
    /// apply that lands moves the next answer.
    #[test]
    fn a_dry_run_answers_what_an_apply_would_and_applies_nothing() {
        let (store, generations) = generations();
        let dry = |document: &[u8]| {
            generations
                .dry_run("", document, Format::Yaml)
                .expect("no guard")
        };
        let number = |run: DryRun| run.would_run.map(|(number, _)| number);
        let counts = |applied| ApplyCounts {
            applied,
            rejected: 0,
        };

        let first = dry(LEAST_LOAD.as_bytes());
        assert_eq!(first.standing, (0, None));
        // The feed before and after: nothing stands, and the document's one
        // profile reads the loads in every role.
        let feed = first.needs.map(wire::needs_diff).expect("the needs diff");
        assert_eq!(feed.before, Some(proto::NeedsSummary::default()));
        let after = feed.after.expect("the feed after");
        assert_eq!(after.models.len(), 4, "{after:?}");
        assert!(
            after.models.iter().all(|row| row.model == "*" && row.loads),
            "{after:?}"
        );
        let (would, digest) = first.would_run.expect("it would run");
        assert_eq!(would, 1);
        assert_eq!(number(dry(LEAST_LOAD.as_bytes())), Ok(1), "not reserved");
        assert_eq!(standing(&store), (0, None));
        assert!(!generations.bound());
        assert_eq!(generations.apply_counts(), counts(0));

        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Ok(ApplyOutcome::Generation(1))
        );
        let again = dry(LEAST_LOAD.as_bytes());
        assert_eq!(again.standing, (1, Some(digest)));
        assert_eq!(again.would_run, Ok((1, digest)), "the standing document");
        let feed = again.needs.map(wire::needs_diff).expect("the needs diff");
        assert_eq!(feed.before, feed.after, "the feed stands");
        assert_eq!(number(dry(RANDOM.as_bytes())), Ok(2));
        assert_eq!(number(dry(&with_line(RANDOM, 9))), Ok(9));
        assert_eq!(number(dry(&with_line(LEAST_LOAD, 9))), Ok(9));

        assert_eq!(yaml(&generations, RANDOM), Ok(ApplyOutcome::Generation(2)));
        assert_eq!(number(dry(LEAST_LOAD.as_bytes())), Ok(3), "an apply landed");
        assert_eq!(standing(&store).0, 2);
        assert_eq!(generations.apply_counts(), counts(2));
    }

    /// The read call answers the source's bytes, the comment included, with
    /// the number and the digest beside them; a generation that does not
    /// stand has no document, and neither has a generation the store was
    /// handed behind the source.
    #[test]
    fn the_read_answer_is_the_sources_bytes_beside_the_number_and_the_digest() {
        let (store, generations) = generations();
        let nothing = NotHeld {
            asked: 0,
            held: None,
            standing: 0,
            on_file: None,
        };
        assert_eq!(generations.document(0), Err(nothing));

        let written = format!("# routed by load\n{LEAST_LOAD}").into_bytes();
        assert_eq!(
            yaml(&generations, written.clone()),
            Ok(ApplyOutcome::Generation(1))
        );
        let (_, Some(digest)) = standing(&store) else {
            panic!("an applied generation has a digest");
        };
        for wanted in [0, 1] {
            let answer =
                wire::spec_document(generations.document(wanted).expect("the standing document"));
            assert_eq!(answer.generation, 1);
            assert_eq!(answer.digest, digest.to_string());
            assert_eq!(
                answer.document, written,
                "the bytes as the source wrote them"
            );
            assert_eq!(answer.format, "yaml");
            assert_eq!(answer.mode, i32::from(proto::ApplyMode::Enforce));
            assert!(answer.applied_unix_micros > 0);
            assert!(answer.acks.is_empty());
        }

        let older = generations.document(2).expect_err("no generation 2");
        assert_eq!(
            older.to_string(),
            "no document is held for generation 2: the spec source's document is generation 1, generation 1 stands, and the standing generation's document alone is retained"
        );

        let mut preset = PolicySpec::preset("random").expect("a preset");
        preset.generation = 7;
        store
            .apply(AppliedSpec::validate(&preset).expect("a preset validates"))
            .expect("above the standing one");
        let behind = NotHeld {
            asked: 7,
            held: Some(1),
            standing: 7,
            on_file: None,
        };
        assert_eq!(generations.document(0), Err(behind));
    }

    /// The Numbers table's dry-run row: one dry run that would run (parse,
    /// digest, validate, the feed before and after) by the document's size,
    /// with the bytes of the needs diff it answers.
    #[test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    fn a_dry_runs_time_by_the_documents_size() {
        use prost::Message as _;

        const ROUNDS: usize = 15;
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let mut out = std::io::stdout().lock();
        for size in [2 << 10, 64 << 10, bound - 128] {
            let (document, bindings) = document_of(size);
            let (_store, generations) = generations();
            assert_eq!(
                yaml(&generations, LEAST_LOAD),
                Ok(ApplyOutcome::Generation(1))
            );
            let mut micros = Vec::with_capacity(ROUNDS);
            let mut diff = 0;
            for _ in 0..ROUNDS {
                let began = Instant::now();
                let dry = generations
                    .dry_run("", &document, Format::Yaml)
                    .expect("no guard");
                micros.push(began.elapsed().as_micros());
                assert_eq!(dry.would_run.map(|(number, _)| number), Ok(2));
                diff = dry
                    .needs
                    .map_or(0, |needs| wire::needs_diff(needs).encoded_len());
            }
            micros.sort_unstable();
            writeln!(
                out,
                "DRY bytes={} bindings={bindings} rounds={ROUNDS} median_us={} worst_us={} diff_bytes={diff}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
    }

    /// Binds the directory at `dir` as the door's source, as a start does.
    fn bind(generations: &Generations, dir: &Path) {
        let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
        generations.bind_store(opened).expect("the store binds");
    }

    /// A door with a spec store bound over a new, empty directory.
    fn stored() -> (tempfile::TempDir, Arc<MemoryStore>, Arc<Generations>) {
        let dir = tempfile::tempdir().expect("a temp dir");
        let (store, generations) = generations();
        bind(&generations, dir.path());
        (dir, store, Arc::new(generations))
    }

    /// What an enforce of `document` with no guard, as YAML, ran.
    fn enforce(
        generations: &Generations,
        document: impl Into<Vec<u8>>,
    ) -> Result<(u64, Digest), SpecRejected> {
        generations
            .enforce("", document.into(), Format::Yaml)
            .expect("no guard")
            .ran
    }

    fn applied(applied: u64, rejected: u64) -> ApplyCounts {
        ApplyCounts { applied, rejected }
    }

    /// An enforce under a spec store: the document is on file under the
    /// number it runs under, as its caller sent it, and the read call
    /// answers those bytes.
    #[test]
    fn an_enforce_is_persisted_and_runs_as_the_next_generation() {
        let (dir, store, generations) = stored();
        assert!(generations.writable() && generations.bound());
        assert_eq!(standing(&store), (0, None));

        let first = generations
            .enforce("", LEAST_LOAD.into(), Format::Yaml)
            .expect("no guard");
        let (number, digest) = first.ran.expect("it runs");
        assert_eq!((number, first.standing), (1, (0, None)));
        assert_eq!(standing(&store), (1, Some(digest)));
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
        let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
        assert_eq!(on_file, LEAST_LOAD.as_bytes());
        let read = generations.document(0).expect("the standing document");
        assert_eq!(
            (read.generation, read.document.as_slice()),
            (1, LEAST_LOAD.as_bytes())
        );
        // The feed before and after: an enforce answers the diff a dry run
        // does.
        let needs = first.needs.map(wire::needs_diff).expect("the diff");
        assert_eq!(needs.before, Some(proto::NeedsSummary::default()));
        let runs = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));
        assert_eq!(needs.after, Some(runs));

        // A JSON document is on file under its own form.
        let json = r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
        let second = generations
            .enforce("", json.into(), Format::Json)
            .expect("no guard");
        assert_eq!(second.ran.map(|(number, _)| number), Ok(2));
        let both = BTreeSet::from([name_of(1, "yaml"), name_of(2, "json")]);
        assert_eq!(names(dir.path()), both);
        assert_eq!(generations.apply_counts(), applied(2, 0));
    }

    #[test]
    fn a_door_takes_one_source() {
        let (_dir, _store, generations) = stored();
        // A second store, on a directory of its own: the door holds its
        // first one's.
        let other = tempfile::tempdir().expect("a temp dir");
        let open = || SpecStore::open(other.path().to_owned()).expect("the directory opens");
        assert_eq!(
            generations.bind_store(open()),
            Err(StoreRefused::SecondSource)
        );

        // A file's document applied first is a source too.
        let (_store, filed) = self::generations();
        assert_eq!(yaml(&filed, LEAST_LOAD), Ok(ApplyOutcome::Generation(1)));
        assert_eq!(filed.bind_store(open()), Err(StoreRefused::SecondSource));
        assert!(!filed.writable());

        // So is a file's document that waits for its replica's first take.
        let (_store, deferred) = self::generations();
        deferred
            .defer(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        assert!(deferred.bound(), "a source, with nothing applied yet");
        assert_eq!(deferred.bind_store(open()), Err(StoreRefused::SecondSource));

        // And a file's document is not applied beside a store.
        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Err(SpecRejected::SecondSource)
        );
        assert_eq!(generations.apply_counts(), applied(0, 0));
    }

    /// The bound of what waits: no more write calls are let in than the
    /// one that applies and the ones behind it.
    #[tokio::test]
    async fn a_write_call_past_the_waiting_ones_is_refused() {
        let (_dir, store, generations) = stored();
        let taken = generations.write_turn.fill();
        let places = taken.len();
        assert_eq!(places, WRITES_WAITING + 1, "every place is free");
        let enforce =
            || generations.enforce_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml);

        let refused = enforce().await;
        assert!(matches!(refused, Err(Unanswered::Waiting)), "{refused:?}");
        // A rollback waits in the same turn.
        let refused = generations.roll_back_in_turn(1).await;
        assert!(matches!(refused, Err(Unanswered::Waiting)), "{refused:?}");
        assert_eq!(standing(&store), (0, None));
        drop(taken);
        // A call gives its place back at its end: as many calls again as
        // there are places each find one.
        for _ in 0..=places {
            let ran = enforce().await.expect("a place is free").ran;
            assert_eq!(ran.map(|(number, _)| number), Ok(1));
        }
    }

    /// A document over the bound and a guard longer than a digest's text
    /// take no place: each is refused though every place is taken.
    #[tokio::test]
    async fn a_document_over_the_bound_and_a_long_guard_wait_on_no_apply() {
        let (dir, _store, generations) = stored();
        let taken = generations.write_turn.fill();
        assert_eq!(taken.len(), WRITES_WAITING + 1, "every place is free");
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");

        let over = generations
            .enforce_in_turn(String::new(), padded(LEAST_LOAD, bound + 1), Format::Yaml)
            .await
            .expect("refused by its size");
        assert_eq!(
            over.ran,
            Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 1))
        );
        let long = "x".repeat(DIGEST_TEXT_LEN + 1);
        let guarded = generations
            .enforce_in_turn(long, LEAST_LOAD.into(), Format::Yaml)
            .await;
        assert!(
            matches!(
                guarded,
                Err(Unanswered::Manager(EnforceRefused::Guarded(0, None)))
            ),
            "{guarded:?}"
        );
        assert!(names(dir.path()).is_empty());
        assert_eq!(generations.apply_counts(), applied(0, 1));
    }

    /// The status a snapshot tells of the spec. No source: none. A file:
    /// the standing document's number, digest and time, and nothing
    /// retained. A store: the generations on file that a rollback can name
    /// beside the standing one, and a start on the directory lists the same.
    #[test]
    fn the_status_names_the_standing_generation_and_what_a_rollback_can_name() {
        let (_store, sourceless) = generations();
        assert_eq!(sourceless.status(), None);

        let (_store, file) = generations();
        yaml(&file, LEAST_LOAD).expect("it runs");
        let status = file.status().map(wire::spec_status).expect("a source");
        assert!(status.applied_unix_micros > 0, "{status:?}");
        assert_eq!(
            status,
            proto::SpecStatus {
                generation: 1,
                digest: file.standing().1.expect("a digest").to_string(),
                applied_unix_micros: status.applied_unix_micros,
                mode: proto::ApplyMode::Enforce.into(),
                retained: Vec::new(),
                source: "file".to_owned(),
            }
        );

        // A store with nothing on file: the store's own number, and no
        // document's digest, time or mode.
        let (dir, _store, stored) = stored();
        let empty = proto::SpecStatus {
            source: "api".to_owned(),
            ..proto::SpecStatus::default()
        };
        assert_eq!(stored.status().map(wire::spec_status), Some(empty));
        for (number, document) in [
            (1, LEAST_LOAD.to_owned()),
            (2, RANDOM.to_owned()),
            (3, bound_to(3)),
        ] {
            let ran = enforce(&stored, document).map(|(number, _)| number);
            assert_eq!(ran, Ok(number));
        }
        let status = stored.status().map(wire::spec_status).expect("a source");
        assert_eq!(
            (
                status.generation,
                status.retained.as_slice(),
                status.source.as_str()
            ),
            (3, &[1, 2][..], "api")
        );
        assert_eq!(
            status.digest,
            stored.standing().1.expect("a digest").to_string()
        );
        assert_eq!(status.mode, i32::from(proto::ApplyMode::Enforce));
        assert!(status.applied_unix_micros > 0, "{status:?}");

        // A rollback runs generation 1 again as generation 4: the three
        // before it are what the next rollback can name.
        assert_eq!(roll_back(&stored, 1).map(|(number, _)| number), Ok(4));
        let rolled = stored.status().map(wire::spec_status).expect("a source");
        assert_eq!(
            (rolled.generation, rolled.retained.as_slice()),
            (4, &[1, 2, 3][..])
        );

        // A start on the same directory lists the same.
        drop(stored);
        let (_store, again) = generations();
        bind(&again, dir.path());
        let listed = again.status().map(wire::spec_status).expect("a source");
        assert_eq!(
            (
                listed.generation,
                listed.retained,
                listed.digest,
                listed.source
            ),
            (4, vec![1, 2, 3], rolled.digest, "api".to_owned())
        );
    }

    /// A door tells the time and the mode of a document it holds, and of no
    /// other: where the store runs a generation another door applied, the
    /// status names that generation with its digest and no time.
    #[test]
    fn the_status_tells_no_time_of_a_document_its_door_does_not_hold() {
        let (store, first) = generations();
        yaml(&first, LEAST_LOAD).expect("it runs");
        let second = Generations::new(Arc::clone(&store), stock());
        yaml(&second, RANDOM).expect("it runs");
        let (generation, digest) = first.standing();
        assert_eq!(generation, 2);
        assert_eq!(
            first.status().map(wire::spec_status),
            Some(proto::SpecStatus {
                generation,
                digest: digest.expect("a digest").to_string(),
                source: "file".to_owned(),
                ..proto::SpecStatus::default()
            })
        );
        // The door that applied it tells its time.
        let told = second.status().map(wire::spec_status).expect("a source");
        assert_eq!(told.mode, i32::from(proto::ApplyMode::Enforce));
        assert!(told.applied_unix_micros > 0, "{told:?}");
    }

    /// Write calls take the door one at a time, in the order they came: a
    /// call awaits its turn before its thread starts, so while the turn is
    /// held none runs, and then each runs as the next number in its order.
    #[tokio::test]
    async fn write_calls_take_their_turn_in_the_order_they_came() {
        let (_dir, store, generations) = stored();
        let held = generations.write_turn.hold().await;
        let mut calls = JoinSet::new();
        for n in 1..=4_u64 {
            let door = Arc::clone(&generations);
            calls.spawn(async move {
                let document = bound_to(n).into_bytes();
                let call = door.enforce_in_turn(String::new(), document, Format::Yaml);
                (n, call.await)
            });
            // The call reaches its wait before the next one is made.
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(standing(&store).0, 0, "a write ran beside the turn held");
        drop(held);
        while let Some(call) = calls.join_next().await {
            let (n, answer) = call.expect("the call ends");
            let ran = answer.expect("a place").ran;
            assert_eq!(ran.map(|(number, _)| number), Ok(n));
        }
    }

    /// Waits until `free` of the write calls' places are free.
    async fn places_free(door: &Generations, free: usize) {
        let started = Instant::now();
        while door.write_turn.available() != free {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the write calls' places did not come to {free} free"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// A call holds its place until its write ends, and the one whose
    /// thread started holds the turn as long. With the door's lock held, as
    /// a write in flight holds it, the calls that fill the places stay in
    /// them, the turn stays taken and one more call is refused; once the
    /// door is free each of them runs and gives its place back.
    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "the case holds the door's lock, as a write in flight does, while the calls take their places"
    )]
    async fn a_write_call_holds_its_place_until_its_write_ends() {
        let (_dir, store, generations) = stored();
        let enforce = |door: Arc<Generations>, n: usize| async move {
            let n = u64::try_from(n).expect("a small count");
            let document = bound_to(n).into_bytes();
            door.enforce_in_turn(String::new(), document, Format::Yaml)
                .await
        };
        let applying = generations.applying.lock();
        let mut calls = JoinSet::new();
        for n in 0..=WRITES_WAITING {
            calls.spawn(enforce(Arc::clone(&generations), n));
        }
        places_free(&generations, 0).await;
        // The call whose thread waits on the door holds the turn: no other
        // call's thread starts behind it, however long the door is held.
        for _ in 0..50 {
            assert!(
                generations.write_turn.is_held(),
                "the turn was given back before the write ended"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let past = enforce(Arc::clone(&generations), 0).await;
        assert!(matches!(past, Err(Unanswered::Waiting)), "{past:?}");
        assert_eq!(standing(&store).0, 0);
        drop(applying);
        while let Some(call) = calls.join_next().await {
            let answer = call.expect("the call ends").expect("a place");
            answer.ran.expect("it runs");
        }
        let all = u64::try_from(WRITES_WAITING).expect("a small count") + 1;
        assert_eq!(standing(&store).0, all);
        assert_eq!(generations.write_turn.available(), WRITES_WAITING + 1);
    }

    /// A generation whose file is gone under the list, as an apply's prune
    /// takes one between a read's look and its open, is not held: the read
    /// answers so, and no internal fault.
    #[tokio::test]
    async fn a_generation_whose_file_is_gone_under_the_list_is_not_held() {
        let (dir, _store, generations) = stored();
        enforce(&generations, LEAST_LOAD).expect("it runs");
        enforce(&generations, RANDOM).expect("it runs");
        std::fs::remove_file(dir.path().join(name_of(1, "yaml"))).expect("the file goes");

        let gone = generations.read(1).await;
        let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = gone else {
            panic!("generation 1 has no file: {gone:?}");
        };
        assert_eq!((not_held.asked, not_held.standing), (1, 2));
    }

    /// A caller that leaves while its call waits for the turn writes
    /// nothing and frees its place: the next call runs as the first
    /// generation, and its document alone is on file.
    #[tokio::test]
    async fn a_write_call_whose_caller_left_while_it_waited_writes_nothing() {
        let (dir, store, generations) = stored();
        let held = generations.write_turn.hold().await;
        let mut calls = JoinSet::new();
        let door = Arc::clone(&generations);
        let left = calls.spawn(async move {
            door.enforce_in_turn(String::new(), RANDOM.into(), Format::Yaml)
                .await
        });
        places_free(&generations, WRITES_WAITING).await;
        left.abort();
        let gone = calls.join_next().await.expect("the one call");
        let gone = gone.expect_err("its caller left");
        assert!(gone.is_cancelled(), "{gone}");
        drop(held);
        places_free(&generations, WRITES_WAITING + 1).await;
        assert_eq!((standing(&store).0, names(dir.path()).len()), (0, 0));

        let ran = generations
            .enforce_in_turn(String::new(), LEAST_LOAD.into(), Format::Yaml)
            .await
            .expect("a place")
            .ran;
        assert_eq!(ran.map(|(number, _)| number), Ok(1));
        let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
        assert_eq!(on_file, LEAST_LOAD.as_bytes());
    }

    /// What a rollback to generation `wanted`, one on file, ran.
    fn roll_back(generations: &Generations, wanted: u64) -> Result<(u64, Digest), SpecRejected> {
        generations.roll_back(wanted).expect("on file").ran
    }

    /// A rollback runs a generation on file again: its bytes are on file
    /// under the next number and run under it, with their digest and the
    /// feed an enforce answers. The standing generation, and a generation
    /// whose document is the standing one's, are no change.
    #[test]
    fn a_rollback_runs_a_generation_on_file_again_under_the_next_number() {
        let (dir, store, generations) = stored();
        let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
        let (_, second) = enforce(&generations, RANDOM).expect("it runs");
        let ran = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));

        let back = generations.roll_back(1).expect("on file");
        assert_eq!(back.standing, (2, Some(second)));
        assert_eq!(back.ran, Ok((3, first)));
        assert_eq!(standing(&store), (3, Some(first)));
        let on_file = std::fs::read(dir.path().join(name_of(3, "yaml"))).expect("the file");
        assert_eq!(on_file, LEAST_LOAD.as_bytes());
        let needs = back.needs.map(wire::needs_diff).expect("the diff");
        assert_eq!(needs.before, Some(ran));
        let runs = wire::needs_summary(needs_rows(store.snapshot().spec().engine()));
        assert_eq!(needs.after, Some(runs));

        for same in [3, 1] {
            assert_eq!(roll_back(&generations, same), Ok((3, first)));
        }
        assert_eq!(names(dir.path()).len(), 3);
        assert_eq!(generations.apply_counts(), applied(3, 0));
    }

    /// The read call answers a generation on file by its number, with the
    /// digest it ran under and its file's time, while the door's lock is
    /// held: it waits on no apply. A generation that is not on file is not
    /// held. A start lists the generations on file again, under the same
    /// digests and times.
    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "the case holds the door's lock across the read: the read takes none"
    )]
    async fn the_read_call_answers_a_generation_on_file() {
        let (dir, _store, generations) = stored();
        let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
        let (_, second) = enforce(&generations, RANDOM).expect("it runs");

        let applying = generations.applying.lock();
        let old = generations.read(1).await.expect("on file");
        drop(applying);
        assert_eq!(
            (old.generation, old.document.as_slice()),
            (1, LEAST_LOAD.as_bytes())
        );
        assert_eq!(
            (old.digest.clone(), old.format.as_str()),
            (first.to_string(), "yaml")
        );
        let written = std::fs::metadata(dir.path().join(name_of(1, "yaml")))
            .and_then(|file| file.modified())
            .expect("the file's time");
        assert_eq!(old.applied_unix_micros, unix_micros(written));
        let stands = generations.read(0).await.expect("the standing one");
        assert_eq!(
            (stands.generation, stands.digest.clone()),
            (2, second.to_string())
        );

        let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = generations.read(7).await
        else {
            panic!("generation 7 is not on file");
        };
        let two = OnFile {
            count: 2,
            span: (1, 2),
        };
        assert_eq!(not_held.on_file, Some(two));

        // The door is gone, as its process is at the next start.
        drop(generations);
        let (_store, again) = self::generations();
        bind(&again, dir.path());
        let again = Arc::new(again);
        assert_eq!(again.read(1).await.expect("on file"), old);
        assert_eq!(again.read(2).await.expect("on file"), stands);
    }

    /// The read call reads a generation on file through the directory the
    /// store opened. Where the path leads to another directory before the
    /// door takes the store, the answer is still the held file's bytes,
    /// not the other's under that name.
    #[tokio::test]
    async fn the_read_call_reads_the_directory_held_where_the_path_leads_elsewhere() {
        let base = tempfile::tempdir().expect("a temp dir");
        let (dir, moved) = (base.path().join("store"), base.path().join("moved"));
        std::fs::create_dir(&dir).expect("the directory");
        std::fs::write(dir.join(name_of(1, "yaml")), LEAST_LOAD).expect("the first");
        std::fs::write(dir.join(name_of(2, "yaml")), RANDOM).expect("the second");
        let opened = SpecStore::open(dir.clone()).expect("the directory opens");
        std::fs::rename(&dir, &moved).expect("the directory moves");
        std::fs::create_dir(&dir).expect("another stands at its path");
        std::fs::write(dir.join(name_of(1, "yaml")), "other").expect("the other's own");
        let (_store, generations) = generations();
        generations.bind_store(opened).expect("the store binds");
        let old = generations.read(1).await.expect("on file");
        assert_eq!(
            (old.generation, old.document.as_slice()),
            (1, LEAST_LOAD.as_bytes())
        );
    }

    /// An older file that is no document of this build stays on file and is
    /// listed nowhere: the start runs the newest, the read call does not
    /// answer the older one, and a rollback to it is refused by name.
    #[tokio::test]
    async fn an_older_file_that_is_no_document_is_left_on_file_and_not_read() {
        let dir = tempfile::tempdir().expect("a temp dir");
        std::fs::write(dir.path().join(name_of(1, "yaml")), "profiles: [").expect("the file");
        std::fs::write(dir.path().join(name_of(2, "yaml")), LEAST_LOAD).expect("the file");
        let (store, generations) = generations();
        bind(&generations, dir.path());
        let generations = Arc::new(generations);
        assert_eq!(standing(&store).0, 2);

        let Err(Unanswered::Manager(EnforceRefused::NotHeld(not_held))) = generations.read(1).await
        else {
            panic!("generation 1 is no document");
        };
        let one = OnFile {
            count: 1,
            span: (2, 2),
        };
        assert_eq!(not_held.on_file, Some(one));
        let refused = roll_back(&generations, 1).expect_err("no document");
        assert!(
            matches!(refused, SpecRejected::Unparsed { .. }),
            "{refused:?}"
        );
        assert_eq!(names(dir.path()).len(), 2);
    }

    /// What a leader's door sends of its standing generation.
    fn sent(leader: &Generations) -> proto::GenerationEvent {
        leader.standing_sent().expect("a held document")
    }

    /// `told` followed at no place, as a restore lands one.
    fn followed(standby: &Generations, told: &proto::GenerationEvent) -> Result<(), Unfollowed> {
        standby.following(told)?.land(None)
    }

    #[test]
    fn a_standby_runs_its_leaders_generation_under_the_leaders_number() {
        let (led, leader) = generations();
        assert_eq!(leader.standing_sent(), None, "no document is held yet");
        yaml(&leader, RANDOM).expect("a valid document");
        yaml(&leader, LEAST_LOAD).expect("a changed document");
        let told = sent(&leader);
        assert_eq!(told, leader.sent(2), "the standing one, by its number");
        // A generation that stands no longer is sent by its number alone.
        assert_eq!(leader.sent(1), worker_wire::generation_told(1));

        let (store, standby) = generations();
        followed(&standby, &told).expect("above the unbound one");
        assert_eq!(standing(&store), standing(&led));
        // The read call answers the leader's bytes, form and time.
        assert_eq!(
            wire::spec_document(standby.document(0).expect("held")),
            wire::spec_document(leader.document(0).expect("held"))
        );
        assert_eq!(standby.apply_counts().applied, 1);
        // Told again, it changes nothing.
        followed(&standby, &told).expect("the standing one again");
        // Nor does an event with no document, whatever its number.
        followed(&standby, &worker_wire::generation_told(9)).expect("no document");
        assert_eq!(standing(&store), standing(&led));
        assert_eq!(standby.apply_counts().applied, 1);
    }

    #[test]
    fn a_replica_that_read_the_same_file_under_a_lower_number_takes_its_leaders() {
        let (_, leader) = generations();
        yaml(&leader, with_line(LEAST_LOAD, 5)).expect("a claim above none");
        let (store, standby) = generations();
        yaml(&standby, LEAST_LOAD).expect("a valid document");
        let (own, digest) = standing(&store);
        assert_eq!(own, 1);
        followed(&standby, &sent(&leader)).expect("the leader's number is above its own");
        assert_eq!(
            standing(&store),
            (5, digest),
            "one document, the leader's number"
        );
        assert_eq!(standby.document(0).expect("held").generation, 5);
    }

    #[test]
    fn a_generation_that_is_not_above_the_standing_one_is_not_followed() {
        let (_, leader) = generations();
        yaml(&leader, RANDOM).expect("a valid document");
        let (store, standby) = generations();
        yaml(&standby, LEAST_LOAD).expect("a valid document");
        // Its own number over another document.
        let own = standing(&store);
        let refused = followed(&standby, &sent(&leader)).expect_err("another document");
        assert_eq!((refused.path, refused.why), (NUMBER, NOT_ABOVE));
        assert_eq!(standing(&store), own);
        // A number under its own, whatever the document.
        yaml(&standby, RANDOM).expect("a changed document");
        let own = standing(&store);
        assert_eq!(own.0, 2);
        let refused = followed(&standby, &sent(&leader)).expect_err("under its own");
        assert_eq!((refused.path, refused.why), (NUMBER, NOT_ABOVE));
        assert_eq!(standing(&store), own);
        assert_eq!(standby.document(0).expect("held").generation, 2);
    }

    #[test]
    fn an_event_this_replica_cannot_run_is_refused_by_its_path() {
        let (store, standby) = generations();
        yaml(&standby, RANDOM).expect("a valid document");
        let own = standing(&store);
        let push = |document: &str, format: &str| {
            Some(proto::SpecPush {
                generation: 2,
                document: document.as_bytes().to_vec(),
                format: format.to_owned(),
                ..proto::SpecPush::default()
            })
        };
        let mut told = worker_wire::generation_told(2);
        // A stage this build does not hold.
        told.document = push(&LEAST_LOAD.replace("healthy", "no-such-stage"), "yaml");
        let refused = followed(&standby, &told).expect_err("a stage this build lacks");
        assert_eq!(refused.path, "generation.document");
        // A form it does not know, and an event that names no number.
        told.document = push(LEAST_LOAD, "toml");
        let refused = followed(&standby, &told).expect_err("no such form");
        assert_eq!(refused.path, "generation.document.format");
        let refused = followed(&standby, &proto::GenerationEvent::default());
        assert_eq!(refused.expect_err("no number").path, NUMBER);
        assert_eq!(standing(&store), own);
        assert_eq!(standby.apply_counts().applied, 1);
        // The same event with a document that runs is followed.
        told.document = push(LEAST_LOAD, "yaml");
        followed(&standby, &told).expect("above its own");
        assert_eq!(standing(&store).0, 2);
    }

    #[test]
    fn a_deferred_document_runs_at_the_first_take_and_at_no_later_one() {
        let (store, door) = generations();
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        assert_eq!(standing(&store), (0, None), "passed, not applied");
        assert_eq!(door.apply_counts().applied, 0);
        door.at_take();
        assert_eq!(standing(&store).0, 1);
        assert_eq!(door.document(0).expect("held").document, RANDOM.as_bytes());
        assert!(
            door.deferred.lock().is_none(),
            "the start's bytes are let go"
        );
        // A later take applies nothing: what was applied since stands.
        yaml(&door, LEAST_LOAD).expect("a changed document");
        door.at_take();
        assert_eq!(standing(&store).0, 2);
        assert_eq!(door.apply_counts().applied, 2);
    }

    #[test]
    fn a_deferred_document_that_does_not_run_refuses_the_start() {
        let (store, door) = generations();
        let unrun = LEAST_LOAD.replace("healthy", "no-such-stage");
        door.defer(unrun.into_bytes(), Format::Yaml)
            .expect_err("a stage this build lacks");
        // Nothing is kept for a take, and the door has no source.
        assert!(!door.bound());
        door.at_take();
        assert_eq!(standing(&store), (0, None));
        let counts = door.apply_counts();
        assert_eq!((counts.applied, counts.rejected), (0, 0));
    }

    #[test]
    fn a_replica_that_deferred_its_document_follows_a_leader_under_any_number() {
        // The leader runs its first generation over another document: a
        // replica that had applied its own at its start could not follow.
        let (_, leader) = generations();
        yaml(&leader, RANDOM).expect("a valid document");
        let (store, standby) = generations();
        standby
            .defer(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        followed(&standby, &sent(&leader)).expect("above no generation");
        assert_eq!(standing(&store).0, 1);
        assert_eq!(
            standby.document(0).expect("held").document,
            RANDOM.as_bytes()
        );
        // Promoted behind a followed generation, the start's bytes are not
        // applied: they are as old as the start, and the file's watch reads
        // the file as it stands then.
        standby.at_take();
        assert_eq!(standing(&store).0, 1);
        assert_eq!(standby.apply_counts().applied, 1);
        assert_eq!(
            standby.document(0).expect("held").document,
            RANDOM.as_bytes()
        );
    }

    /// A host can publish and serve its start's document without using
    /// the replica's source and standing notifications.
    #[test]
    fn a_local_start_serves_its_document_without_notifying_a_replica() {
        REPLICA_NOTICES.with(|notices| notices.set(Some(0)));
        let (_, replica) = generations();
        let admissions = Cell::new(0);
        let applied = replica.apply_where(
            replica.lead(),
            LEAST_LOAD.into(),
            Format::Yaml,
            true,
            |_| {
                assert_eq!(REPLICA_NOTICES.with(Cell::get), Some(admissions.get()));
                admissions.set(admissions.get() + 1);
                None
            },
        );
        assert_eq!(applied, Ok(ApplyOutcome::Generation(1)));
        assert_eq!(
            REPLICA_NOTICES.with(|notices| notices.replace(Some(0))),
            Some(2)
        );

        let door = DocumentManager::new(Arc::new(MemoryStore::new()), stock(), Vec::new());
        assert_eq!(
            door.apply(RANDOM.into(), Format::Yaml),
            Ok(ApplyOutcome::Generation(1))
        );
        let held = door.document(0).expect("the start's document");
        let status = door.status().expect("the start's status");
        assert_eq!(
            (held.generation, held.document.as_slice()),
            (1, RANDOM.as_bytes())
        );
        assert_eq!((status.generation, status.digest), (1, Some(held.digest)));
        assert_eq!(REPLICA_NOTICES.with(Cell::get), Some(0));
    }

    /// Held bytes of another generation are not the standing one's. Where
    /// another writer's generation runs, an event at the standing number
    /// over the bytes this door holds is no word of what stands: it is
    /// refused as a number that is not above.
    #[test]
    fn an_event_over_the_held_bytes_of_another_generation_is_not_the_standing_one() {
        let (store, door) = generations();
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
        let mut next: PolicySpec = serde_yaml::from_str(LEAST_LOAD).expect("a document");
        next.generation = 2;
        store
            .apply(AppliedSpec::validate(&next).expect("a valid document"))
            .expect("above the standing one");
        // A leader whose generation 2 runs the bytes this door holds under 1.
        let (_, leader) = generations();
        yaml(&leader, LEAST_LOAD).expect("a valid document");
        yaml(&leader, RANDOM).expect("a changed document");
        followed(&door, &sent(&leader)).expect_err("not above the standing one");
        assert_eq!(standing(&store).0, 2);
    }

    /// The document the source handed last stood on its replica once it was
    /// applied here or told by a leader, and at no other time: not while
    /// it only passed, and not while a leader told another one.
    #[test]
    fn the_sources_document_stood_once_it_was_applied_or_told() {
        let (_, alone) = generations();
        assert!(alone.source_stood(), "no source handed one");
        yaml(&alone, RANDOM).expect("a valid document");
        assert!(alone.source_stood(), "applied here");

        let (_, door) = generations();
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        assert!(!door.source_stood(), "passed, never applied");
        // A leader tells another document, then the source's own.
        let (_, leader) = generations();
        yaml(&leader, LEAST_LOAD).expect("a valid document");
        followed(&door, &sent(&leader)).expect("above no generation");
        assert!(!door.source_stood(), "another document than the source's");
        yaml(&leader, RANDOM).expect("a changed document");
        followed(&door, &sent(&leader)).expect("above its own");
        assert!(door.source_stood(), "told while it followed");

        let (_, taken) = generations();
        taken
            .defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        taken.at_take();
        assert!(taken.source_stood(), "applied at its first take");

        // The source's later word replaces its start's: a file that moves
        // to the document its replica follows holds what stands.
        let (_, moved) = generations();
        moved
            .defer(LEAST_LOAD.into(), Format::Yaml)
            .expect("a valid document");
        followed(&moved, &sent(&leader)).expect("above no generation");
        assert!(!moved.source_stood(), "another document than the source's");
        assert!(matches!(
            yaml(&moved, RANDOM),
            Ok(ApplyOutcome::Unchanged(2))
        ));
        assert!(moved.source_stood(), "the source handed what stands");
    }

    /// What the door knows of a file that a replica that follows shows it,
    /// by the order of the hand and of the generations. Shown ahead of the
    /// generation that brings its document, the document stands once that
    /// generation enters, and stays stood when its leader replaces it: a
    /// mount that lags. Shown behind the generation that replaced it, it
    /// has not stood since: a rollback. A file with no document never
    /// stood. The door applies nothing of a hand and keeps no byte of it.
    #[test]
    fn a_shown_document_stood_by_the_order_of_its_hand_and_the_generations() {
        let (store, door) = generations();
        let (_, leader) = generations();
        door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
        assert!(!door.source_stood(), "ahead of its generation");
        yaml(&leader, RANDOM).expect("a valid document");
        followed(&door, &sent(&leader)).expect("above no generation");
        assert!(door.source_stood(), "its generation entered");
        yaml(&leader, LEAST_LOAD).expect("a changed document");
        followed(&door, &sent(&leader)).expect("above its own");
        assert!(
            door.source_stood(),
            "replaced by its leader: a mount that lags"
        );
        door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
        assert!(
            !door.source_stood(),
            "behind the generation that replaced it"
        );
        door.shown(Some(LEAST_LOAD.as_bytes()), Format::Yaml);
        assert!(door.source_stood(), "the document that stands");
        door.shown(None, Format::Yaml);
        assert!(!door.source_stood(), "a file no look could read");
        door.shown(Some(LEAST_LOAD.as_bytes()), Format::Yaml);
        assert!(door.source_stood());
        door.shown(Some(b"{ no policy"), Format::Yaml);
        assert!(!door.source_stood(), "a file that does not parse");
        assert_eq!(standing(&store).0, 2, "a hand applies nothing");
        assert_eq!(door.apply_counts(), applied(2, 0));
        assert_eq!(
            door.document(0).expect("held").document,
            LEAST_LOAD.as_bytes()
        );
    }

    /// The hand of a replica that follows waits for the lock a generation
    /// enters under, the one of the bytes held: a hand and an entry stand
    /// in one order, whichever thread runs first.
    #[test]
    fn a_hand_waits_for_the_lock_a_generation_enters_under() {
        let (_, door) = generations();
        let entry = door.held.lock();
        thread::scope(|scope| {
            let (done, lands) = mpsc::channel();
            let door = &door;
            scope.spawn(move || {
                door.shown(Some(RANDOM.as_bytes()), Format::Yaml);
                done.send(()).expect("the case waits");
            });
            let early = lands.recv_timeout(Duration::from_millis(200));
            assert!(early.is_err(), "the hand waits for the lock");
            assert!(door.source_stood(), "nothing is written ahead of it");
            drop(entry);
            lands
                .recv_timeout(Duration::from_secs(30))
                .expect("the hand lands behind the entry");
        });
        assert!(!door.source_stood());
    }

    /// A door with a spec store bound follows no document: no standby
    /// follows beside a store, and the refusal names the document.
    #[test]
    fn a_door_beside_a_spec_store_follows_no_document() {
        let (_, leader) = generations();
        yaml(&leader, RANDOM).expect("a valid document");
        let (_dir, store, standby) = stored();
        let own = standing(&store);
        let refused = followed(&standby, &sent(&leader)).expect_err("a store is bound here");
        assert_eq!(refused.path, "generation.document");
        assert_eq!(
            refused.why,
            "a spec store is bound here, and no standby follows beside one"
        );
        assert_eq!(standing(&store), own);
    }

    /// Under a shared lease the door holds who writes: its source, while
    /// the replica's fence shows a term. A document the source hands while
    /// the replica does not lead is not read, whether it runs or not and
    /// whatever its watch saw ahead of a look, and nothing is counted. The
    /// log writes there all the same.
    #[test]
    fn under_a_shared_lease_the_source_writes_while_its_replica_leads() {
        let (store, door) = generations();
        let mut elector = election();
        door.under(elector.fence());
        assert!(!door.source_writes(), "it has not led yet");
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
        // Its lead ends: a look that was in flight applies nothing.
        assert!(elector.release().expect("a release"), "it held the lease");
        assert_eq!(yaml(&door, LEAST_LOAD), Ok(ApplyOutcome::Waits(1)));
        assert_eq!(standing(&store).0, 1);
        assert_eq!(door.apply_counts(), applied(1, 0));
        // One that does not run waits all the same: it is not read, so it
        // is not refused.
        let unrun = LEAST_LOAD.replace("healthy", "no-such-stage");
        assert_eq!(yaml(&door, unrun), Ok(ApplyOutcome::Waits(1)));
        assert_eq!(door.apply_counts(), applied(1, 0));
        let (_, leader) = generations();
        yaml(&leader, LEAST_LOAD).expect("a valid document");
        yaml(&leader, bound_to(1)).expect("a changed document");
        followed(&door, &sent(&leader)).expect("above its own");
        assert_eq!(standing(&store).0, 2);
        // A replica alone over its lease has no fence here: its source
        // writes always.
        let (_, alone) = generations();
        assert!(alone.source_writes());
    }

    /// A lead that ends while a document is passed applies nothing: the
    /// source's entry asks who writes again behind the pass, as a decision
    /// reads its fence again where its answer is handed over.
    #[test]
    fn a_lead_that_ends_under_a_documents_pass_applies_nothing() {
        thread_local! {
            static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
        }
        fn ending(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
            assert!(
                matches!(left, Some(Ok(true))),
                "it held the lease: {left:?}"
            );
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), ending);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.apply_counts(), applied(0, 0));
    }

    /// A leader's generation this replica cannot follow is passed once: the
    /// same event sent again at a retry is refused from the door's memory,
    /// by the same words. Another generation is passed anew.
    #[test]
    fn a_refused_generation_sent_again_is_refused_from_memory() {
        thread_local! {
            static PASSES: Cell<u32> = const { Cell::new(0) };
        }
        fn counting(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            PASSES.with(|passes| passes.set(passes.get() + 1));
            faulting(document, digest, catalog, denied_headers)
        }
        let (_, leader) = generations();
        yaml(&leader, RANDOM).expect("a valid document");
        let store = Arc::new(MemoryStore::new());
        let standby = Generations::validating(Arc::clone(&store), stock(), counting);
        let told = sent(&leader);
        let first = followed(&standby, &told).expect_err("a fault of this build's validator");
        assert_eq!(first.path, "generation.document");
        assert_eq!(PASSES.with(Cell::get), 1);
        assert_eq!(followed(&standby, &told), Err(first));
        assert_eq!(PASSES.with(Cell::get), 1, "no second pass");
        yaml(&leader, LEAST_LOAD).expect("a changed document");
        followed(&standby, &sent(&leader)).expect("a document that runs");
        assert_eq!(PASSES.with(Cell::get), 2);
        assert_eq!(standing(&store).0, 2);
    }

    /// The unbound start waits for a document a start deferred, and for no
    /// other: a controller with no source and a replica alone that applied
    /// its file carry no mark. The mark ends at the first publication, by a
    /// take's road or by a followed generation's.
    #[test]
    fn the_unbound_start_awaits_a_deferred_document_alone() {
        let awaits = |store: &MemoryStore| store.snapshot().spec().awaits_document();
        let (store, alone) = generations();
        assert!(!awaits(&store), "no source");
        yaml(&alone, RANDOM).expect("a valid document");
        assert!(!awaits(&store), "a replica alone applied its file");

        let (store, door) = generations();
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        assert!(awaits(&store), "the start's document waits");
        door.at_take();
        assert_eq!(standing(&store).0, 1);
        assert!(!awaits(&store), "it landed at the take");

        let (_, leader) = generations();
        yaml(&leader, LEAST_LOAD).expect("a valid document");
        let (store, standby) = generations();
        standby
            .defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        assert!(awaits(&store));
        followed(&standby, &sent(&leader)).expect("above no generation");
        assert!(!awaits(&store), "a followed generation stands");
    }

    /// The watch's sleep where the start's document waits: no longer than
    /// to the opening of a lead that is still ahead, no time once the fence
    /// shows a term, and the whole of it where nothing waits.
    #[test]
    fn the_watch_sleeps_no_longer_than_to_the_opening_a_starts_document_waits_for() {
        let most = Duration::from_secs(3_600);
        let deadline = Duration::from_secs(600);
        // A replica alone has no fence, and no opening to wait for.
        let (_, alone) = generations();
        assert_eq!(alone.nap(most), most);

        // A take with no holder to wait for opens one renew deadline
        // later, and its first renewal says when.
        let (store, door) = generations();
        let mut elector = candidate_under(deadline);
        door.under(elector.fence());
        assert_eq!(door.nap(most), most, "no document waits");
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        assert_eq!(door.nap(most), most, "no lead is in sight");
        elector.tick(Instant::now()).expect("a take");
        elector.tick(Instant::now()).expect("a renewal");
        let nap = door.nap(most);
        assert!(
            nap <= deadline && deadline - nap < Duration::from_secs(60),
            "to the opening: {nap:?}"
        );
        assert_eq!(standing(&store), (0, None), "the lead is not open");

        // Under an open fence it is no time, and the whole once the
        // document stands.
        let (store, door) = generations();
        let mut elector = election();
        door.under(elector.fence());
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(door.nap(most), Duration::ZERO);
        door.deferred_now();
        assert_eq!(standing(&store).0, 1);
        assert_eq!(door.nap(most), most);
    }

    /// A start's document that its take's pass refuses is counted and let
    /// go, and the door keeps its source: the spec calls are served on.
    #[test]
    fn a_starts_document_refused_at_its_take_leaves_the_door_its_source() {
        thread_local! {
            static PASSES: Cell<u32> = const { Cell::new(0) };
        }
        // The start's pass runs, and the take's faults.
        fn faulting_again(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            let passed = PASSES.with(|passes| passes.replace(passes.get() + 1));
            assert_eq!(passed, 0, "a fault of the validator");
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), faulting_again);
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        door.at_take();
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.apply_counts(), applied(0, 1));
        assert!(door.deferred.lock().is_none(), "the bytes are let go");
        assert!(door.bound(), "a source all the same");
    }

    /// The lock a case holds while an entry of the door's own source waits
    /// behind it.
    enum Locked {
        /// The door's, of the held bytes: the entry has passed both of the
        /// door's reads of the fence and waits in the door.
        Held,
        /// The store's last, of the loads, as a load's write holds it: the
        /// entry waits inside the store.
        Loads,
    }

    /// An entry under a lead whose fence closes by the clock while the
    /// entry waits behind `locked`: nothing is published, the standing
    /// generation stays and no bytes are kept.
    fn a_fence_closes_behind(locked: Locked) {
        let deadline = Duration::from_millis(800);
        let store = Arc::new(MemoryStore::new());
        let door = Generations::new(Arc::clone(&store), stock());
        let mut elector = election_under(deadline);
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        let (held, loads) = match locked {
            Locked::Held => (Some(door.held.lock()), None),
            Locked::Loads => (None, Some(store.holding_loads())),
        };
        let outcome = thread::scope(|scope| {
            let entering = scope.spawn(|| yaml(&door, RANDOM));
            thread::sleep(deadline + Duration::from_millis(400));
            assert!(!door.source_writes(), "the fence closed by the clock");
            drop((held, loads));
            entering.join().expect("the entry's thread")
        });
        assert_eq!(outcome, Ok(ApplyOutcome::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.apply_counts(), applied(0, 0));
        assert!(door.held.lock().is_none(), "no bytes are kept");
    }

    /// A fence that closes while the entry waits for the door's lock of
    /// the held bytes publishes nothing: the store asks its last word
    /// behind the locks its entry waits for.
    #[test]
    fn a_fence_that_closes_while_the_entry_waits_for_a_lock_publishes_nothing() {
        a_fence_closes_behind(Locked::Held);
    }

    /// The store asks its last word behind every lock its entry waits for.
    /// An entry that waits for the loads' lock, the last one, while the
    /// fence closes publishes nothing.
    #[test]
    fn a_fence_that_closes_while_the_store_waits_for_its_last_lock_publishes_nothing() {
        a_fence_closes_behind(Locked::Loads);
    }

    /// The standing document passed again under a lead that ends changes
    /// nothing and keeps no bytes: the fence is read behind a pass ahead of
    /// both of its ends.
    #[test]
    fn the_standing_document_passed_under_a_lead_that_ends_waits() {
        thread_local! {
            static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
            static PASSES: Cell<u32> = const { Cell::new(0) };
        }
        // Under its second pass the replica gives its lease up.
        fn ending(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 1 {
                let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
                assert!(
                    matches!(left, Some(Ok(true))),
                    "it held the lease: {left:?}"
                );
            }
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), ending);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(1)));
        assert_eq!(door.apply_counts(), applied(1, 0));
    }

    /// Bytes read under one term are not applied under the next: an entry
    /// is admitted under the lead its source took ahead of the read, and
    /// the source reads again.
    #[test]
    fn bytes_read_under_one_term_are_not_applied_under_the_next() {
        thread_local! {
            static LEAD: RefCell<Option<Elector>> = const { RefCell::new(None) };
            static PASSES: Cell<u32> = const { Cell::new(0) };
        }
        // Under its first pass the replica is deposed and takes again.
        fn retaking(
            document: &PolicySpec,
            digest: Digest,
            catalog: &StageRegistry,
            denied_headers: &[&str],
        ) -> Result<AppliedSpec, SpecError> {
            if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 0 {
                LEAD.with_borrow_mut(|lead| {
                    let lead = lead.as_mut().expect("an elector");
                    assert!(matches!(lead.release(), Ok(true)), "it held the lease");
                    lead.tick(Instant::now()).expect("a second take");
                });
            }
            AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), retaking);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(door.lead(), Some(Leading::Term(1)));
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.lead(), Some(Leading::Term(2)), "it leads again");
        // Read again under the term that shows now, the bytes are applied.
        assert_eq!(yaml(&door, RANDOM), Ok(ApplyOutcome::Generation(1)));
        assert_eq!(door.apply_counts(), applied(1, 0));
    }
}

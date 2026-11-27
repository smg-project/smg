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
//! source's entry reads the replica's fence there, as a decision reads it:
//! the door is built under that fence ([`Generations::stating`]), and the
//! source writes while its replica leads.
//! A watch that read under a lead that has since ended cannot publish
//! those bytes ([`ApplyOutcome::Waits`]). A replica that has yet to lead
//! applies nothing of its own.
//!
//! A replica under a shared lease applies no document at its start
//! ([`Generations::start_with`]): the start passes its file's document,
//! and it is applied under the replica's first lead where the replica
//! runs nothing by then, at the take or behind the lead's opening
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
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use tokio::task::JoinError;
use tracing::{debug, warn};

use super::{
    spec_store::{SpecStore, StoreRefused},
    wire,
    worker_wire::{self, Refused as Unfollowed},
};
pub(crate) use crate::document::operator::{
    ApplyOutcome, Format, SpecRejected, MAX_DOCUMENT_BYTES,
};
use crate::{
    document::{
        operator::{
            above, over_bound, Checked, DocumentManager, DryRun, EnforceOutcome, EnforceRefused,
            Entered, FollowerTurn, Held, Line, SerialTurn,
        },
        spec::{AppliedSpec, Digest, OverloadSwitch, PolicySpec, SpecError},
    },
    service::{elector::Fence, proto},
    stages::registry::StageRegistry,
    store::{MemoryStore, Position},
    worker_host::{Members, Objects},
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

/// The lead a replica writes under now, by the fence its door was built
/// under: alone where it was built under none, else the term the fence
/// shows by the clock ([`Generations::lead`]). One read of the fence, and
/// no wait.
fn lead_under(fence: Option<&Fence>) -> Option<Leading> {
    match fence {
        None => Some(Leading::Alone),
        Some(fence) => fence.leader_term(Instant::now()).map(Leading::Term),
    }
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

/// How many dry runs may wait their turn: a call past them is refused.
/// The count bounds what waits. Bytes are bounded before the wait: a
/// document over the size bound takes no turn, and neither does a guard
/// longer than a digest's text.
///
/// The heaviest file the search finds passes in under 40 ms at the median
/// by run, so 32 passes take under a second and a half. 32 documents of
/// the size bound, with their guards, are 4 MiB. A waiting call that ends
/// leaves the wait and frees its place.
pub(crate) const DRY_RUNS_WAITING: usize = 32;

/// How many write calls may wait for the door: a call past them is refused.
/// A waiting call holds its document, so what waits is bounded as the dry
/// runs are: 32 documents of the size bound are 4 MiB.
pub(crate) const WRITES_WAITING: usize = 32;

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
    /// One dry run at a time ([`Generations::dry_run_in_turn`]).
    dry_turn: SerialTurn,
    /// Enforce and rollback share one write turn
    /// ([`Generations::in_turn`]).
    write_turn: SerialTurn,
    /// The start's document of a replica under a shared lease, in its
    /// source's form, from the start that passed it until the replica's
    /// first lead applies it or refuses it, or a followed generation
    /// stands ([`Generations::defer`], [`Generations::deferred_now`]).
    deferred: Mutex<Option<(Vec<u8>, Format)>>,
    /// A file is this door's source: a start passed its document
    /// ([`Generations::defer`]). Never unset.
    file_sourced: AtomicBool,
    /// The fence of a replica under a shared lease, handed where the door
    /// is built ([`Generations::stating`]): the source's entry is admitted
    /// while it shows the term the entry's bytes were read under, and a
    /// file's start passes its document and applies none
    /// ([`Generations::start_with`]). None for a replica alone over its
    /// lease, whose source writes always. The door's hook reads the same
    /// fence.
    fence: Option<Arc<Fence>>,
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

/// Why a controller's document says the overload switch once, behind the
/// word the first binding says: the reason of [`one_switch`]'s refusal.
macro_rules! one_word {
    ($first:literal) => {
        concat!(
            "the ",
            $first,
            " bindings[0] says, by its block or the block's default: a controller builds its \
             workers with one default set of overload thresholds, so the bindings of its \
             document say one word"
        )
    };
}

/// A controller's pass of a document: the stock pass, then one word of the
/// overload switch. A controller builds its workers with one default set
/// of overload thresholds, so the bindings of its document say the switch
/// once. A document whose bindings differ is refused by the path of the
/// first binding that differs from the first, with its value
/// ([`OverloadSwitch::differs`]), as a gateway refuses it at its start.
/// Every pass of the door runs this: a source's hand, a dry run, a
/// rollback, a spec store's bind, and a leader's generation that a standby
/// follows.
fn one_switch(
    document: &PolicySpec,
    digest: Digest,
    catalog: &StageRegistry,
    denied_headers: &[&str],
) -> Result<AppliedSpec, SpecError> {
    let applied = AppliedSpec::validate_digested(document, digest, catalog, denied_headers)?;
    let Some((at, says)) = OverloadSwitch::differs(&document.bindings) else {
        return Ok(applied);
    };
    let reason = if says {
        one_word!("false")
    } else {
        one_word!("true")
    };
    Err(SpecError::BadField {
        path: format!("bindings[{at}].overload.enabled"),
        value: says.to_string(),
        reason,
    })
}

/// The hook of a controller's door ([`DocumentManager::telling`]): it
/// states the overload switch that stands in the store on `holder`
/// ([`Objects::state_overload_switch`]): on where every binding leaves it
/// on, off where every binding switches it off. The door's pass refuses
/// bindings that differ. A document of such bindings that stands past the
/// pass keeps the default on, as the unbound start does.
///
/// An object the word lays again took its standing object's status, and
/// the holder settles that status with the table's behind each lay, under
/// the replica's own word of its lead, read by `fence` as the door reads it
/// ([`Generations::lead`]): a leader's table is told the object's status,
/// and a replica that does not lead writes no row.
fn states_on(holder: Arc<Objects<Members>>, fence: Option<Arc<Fence>>) -> Entered {
    Box::new(move |store: &MemoryStore| {
        let switch = store.with_snapshot(|world| world.spec().overload_switch());
        let leads = || lead_under(fence.as_deref()).is_some();
        holder.state_overload_switch(store, switch != OverloadSwitch::Off, &leads);
    })
}

impl Generations {
    /// The door of a replica that keeps worker objects. Its pass refuses a
    /// document whose bindings differ on the overload switch
    /// ([`one_switch`]). Each generation that enters states its overload
    /// switch on `holder` ([`states_on`]). The door's hook does it, once a
    /// generation, behind the store's entry and outside the document lock
    /// ([`DocumentManager::enter`]). It runs under the door's lock, which
    /// comes ahead of the holder's turn: the holder says the order of the
    /// locks once, at its turn ([`Objects`]). So an apply returns behind
    /// the rows its word changed, and two entries in a row leave the rows
    /// of the last word. A panic inside the hook leaves the generation in
    /// force and the rows it had not laid under the old word. A document
    /// that a host applies through the store's own entry passes no door,
    /// and no word is stated for it.
    ///
    /// `fence` is the fence of a replica under a shared lease, and none
    /// for a replica alone over its lease. Under it the source writes
    /// while the fence shows the term its bytes were read under, and at
    /// no other time ([`Generations::apply_under`]). A replica that has
    /// not led yet, one that follows a leader, and one whose lead ended
    /// apply nothing of their own, whatever their file's watch read ahead
    /// of a look. The door's hook reads the replica's lead by the same
    /// fence, so the door takes it with a spec source or with none. It is
    /// handed here and nowhere later: a door never changes the lease it
    /// stands under, and its file's start asks the door which start it is
    /// ([`Generations::start_with`]).
    pub(crate) fn stating(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
        holder: Arc<Objects<Members>>,
        fence: Option<Arc<Fence>>,
    ) -> Self {
        let door = DocumentManager::validating(store, catalog, Vec::new(), one_switch);
        let entered = states_on(holder, fence.clone());
        Self::over(door.telling(entered), fence)
    }

    fn over(door: DocumentManager, fence: Option<Arc<Fence>>) -> Self {
        Self {
            door,
            dry_turn: SerialTurn::new(DRY_RUNS_WAITING),
            write_turn: SerialTurn::new(WRITES_WAITING),
            deferred: Mutex::new(None),
            file_sourced: AtomicBool::new(false),
            fence,
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
                    // A case counts a replica's notices, and this step itself tells each one.
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
        self.writable() || self.file_sourced.load(Ordering::Relaxed) || self.holds()
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
        match self.held_under(generation) {
            Some(document) => wire::generation_event(document),
            None => worker_wire::generation_told(generation),
        }
    }

    /// The standing generation's event, for the snapshot a standby is
    /// restored from; none where no document is held for it.
    pub(crate) fn standing_sent(&self) -> Option<proto::GenerationEvent> {
        self.document(0).ok().map(wire::generation_event)
    }

    /// The start's document of the door's file, as its source wrote it in
    /// `format`. A replica alone over its lease applies it as the source's
    /// first ([`Generations::apply`]). One under a shared lease passes it
    /// and applies none ([`Generations::defer`]). The door knows which of
    /// the two it is since it was built ([`Generations::stating`]): a
    /// start asks nobody else, so no start is deferred on a door that
    /// stands under no fence, and none applies ahead of a lead under one.
    pub(crate) fn start_with(&self, document: Vec<u8>, format: Format) -> Result<(), SpecRejected> {
        match self.fence {
            None => self.apply(document, format).map(drop),
            Some(_) => self.defer(document, format),
        }
    }

    /// Whether the door's file starts by a pass alone: the door was built
    /// under the fence of a shared lease ([`Generations::start_with`]).
    pub(crate) fn starts_deferred(&self) -> bool {
        self.fence.is_some()
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
    fn defer(&self, document: Vec<u8>, format: Format) -> Result<(), SpecRejected> {
        self.admit_deferred(|standing| {
            let Checked { digest, line, next } =
                self.pass(standing, &document, format, Line::Claim)?;
            if let Some(next) = next {
                above(standing.0, next.generation())?;
            }
            self.sourced(digest, line);
            *self.deferred.lock() = Some((document, format));
            self.file_sourced.store(true, Ordering::Relaxed);
            Ok(())
        })
    }

    /// The lead this replica writes under now: a source takes it ahead of
    /// its read and hands it in with the bytes
    /// ([`Generations::apply_under`]). None where the replica is under a
    /// shared lease and its fence shows no term.
    pub(crate) fn lead(&self) -> Option<Leading> {
        lead_under(self.fence.as_deref())
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
        let Some(fence) = &self.fence else {
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
        self.observing(|number, standing| {
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
        });
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
        let turn = self.follower_turn();
        let following = |turn, change| Following {
            door: self,
            turn,
            number,
            change,
        };
        let Some(push) = &told.document else {
            return Ok(following(turn, None));
        };
        let Some(format) = Format::named(&push.format) else {
            return Err(Unfollowed {
                path: "generation.document.format",
                why: "a form this build does not know",
            });
        };
        if turn.beside_spec_store() {
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
            let same = number == standing && self.holds_bytes(standing, format, &push.document);
            if same {
                return Ok(following(turn, None));
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
            Ok(Checked { next: None, .. }) => return Ok(following(turn, None)),
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
        Ok(following(turn, Some((next, kept))))
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
    turn: FollowerTurn<'a>,
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
            turn,
            number,
            change,
        } = self;
        match (change, at) {
            (Some((next, kept)), at) => turn
                .enter(at, next, kept, |generation, digest| door.stands(generation, digest))
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
                turn.place(at, number);
                Ok(())
            }
            (None, None) => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/generations/tests/mod.rs"]
pub(crate) mod tests;

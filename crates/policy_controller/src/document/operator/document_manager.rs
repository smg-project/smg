//! The policy spec's generations and the standing document as its source
//! wrote it.
//!
//! One invariant: **the writer that applies mints the number.** A source
//! hands over a document; the number it runs under is decided here, one
//! apply at a time, against the generation the store holds. A document that writes
//! no `generation` line takes the standing number plus one. A line is a
//! claim the store's entry checks: it admits a number above the standing
//! one and refuses any other by naming the standing one. A document whose
//! digest is the standing one's, under no line or under the standing line,
//! is no change: nothing is applied and nothing counted. A changed document
//! under the standing line is refused by name, so on one replica a number
//! never names two documents. The validator's word comes before either: a
//! document it refuses is refused whatever its digest, which reads no line
//! the canonical form leaves out.
//!
//! A document's own pass (the bound, the form, the digest, the validator
//! and the line) is one function, [`check`], which takes no lock a reader
//! takes: the read call waits on no validation. A fault of that pass is
//! held here: a document that panics its parse, its digest, its validator or
//! its engine's build is a document that does not run, refused by name with
//! the running generation kept. A fault past the pass, in the store's
//! entry, is not held: the store may be half written.
//!
//! A dry run ([`DocumentManager::dry_run`]) runs a document's pass and applies
//! nothing: it answers what an apply would do against the generation that
//! stands at that instant, and it reserves no number. It reads that
//! generation and its source's rule on one side of a store's bind, never
//! one of each. A host can make one dry run at a time to budget one pass
//! beside the source's apply.
//!
//! A spec store is the source an apply call writes ([`SpecStore`],
//! [`DocumentManager::enforce`]): a directory held by the process. One
//! invariant holds there: **a generation is on file before the store's
//! entry is called.** The document's pass, its file and the entry run
//! under this door's one lock, so a generation that runs was persisted
//! first, and a start applies the newest one on file, under the number its
//! file's name states ([`DocumentManager::bind_store`]): a store that runs that
//! document under a lower number takes the file's, so the next write's
//! number is above every file's. A generation the entry refuses goes off
//! file again under that lock. Where its name stays, the write says so
//! beside the refusal, the store lists it no more, and the next write
//! tries its removal first. A call's document writes the next number or
//! none ([`Line::Next`]): another line is refused by naming the standing
//! one. A host has one source: a store can be bound where no document was
//! applied or deferred ([`StoreRefused::SecondSource`]), and a source's
//! document is refused where a store is bound
//! ([`SpecRejected::SecondSource`]).
//!
//! A rollback ([`DocumentManager::roll_back`]) is such a write of a document
//! the store has on file: it runs again under the next number, whatever
//! line its bytes write, and its file is the next number's. The read call
//! answers the standing generation's document from memory and any other
//! generation on file from its file, under the
//! digest and the time the door lists it by, and takes no lock an apply
//! holds.
//!
//! A refusal an apply or a write returns is one event and one count, and
//! the running generation stays. A source that keeps running hands its
//! own findings back the same way ([`DocumentManager::reject`]).
//!
//! The number and the digest have one holder, the store's
//! ([`StoreSnapshot::spec`](crate::store::StoreSnapshot::spec)). What lives
//! here is the record around the bytes: the document as the source wrote it,
//! its format and when it was applied. The read answers those bytes with the
//! number and the digest beside them, never a re-serialized form. One document
//! is held, the standing generation's. A spec store also keeps the last
//! [`RETAINED`](super::spec_store::RETAINED) on file.

use std::{
    any::Any,
    fs::File,
    panic::{self, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::SystemTime,
};

use parking_lot::Mutex;
use tracing::{info, warn};

#[cfg(any(test, feature = "controller"))]
use super::spec_store::StoreRefused;
#[cfg(feature = "controller")]
use super::SerialTurn;
use super::{
    answers::OnFile,
    reader::Refused,
    spec_store::{Persisted, SpecStore},
    ApplyCounts, ApplyOutcome, Document, DryRun, EnforceOutcome, EnforceRefused, Format, NeedsRow,
    NotHeld, SpecRejected, Status, MAX_DOCUMENT_BYTES,
};
use crate::{
    document::spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    engine::{framework, framework::Engine},
    stages::registry::StageRegistry,
    store::{MemoryStore, StaleGeneration},
};

/// A generation on file under a spec store: what the read call answers
/// beside its bytes, and what a rollback finds it by.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(feature = "controller"),
    expect(
        dead_code,
        reason = "Hosts without a spec store do not read its retained documents."
    )
)]
struct Retained {
    persisted: Persisted,
    digest: Digest,
    /// When its file was written.
    applied: SystemTime,
}

/// The standing generation's document as its source wrote it.
pub(crate) struct Held {
    pub(crate) generation: u64,
    pub(crate) digest: Digest,
    pub(crate) document: Vec<u8>,
    pub(crate) format: Format,
    pub(crate) applied: SystemTime,
}

/// What a host must feed under `engine`, in the engine's order.
pub(crate) fn needs_rows(engine: &Engine) -> Vec<NeedsRow> {
    engine
        .needs_rows()
        .into_iter()
        .map(|(model, phase, needs)| NeedsRow {
            model: model.to_owned(),
            phase,
            needs,
        })
        .collect()
}

/// The number and the digest of the standing generation `spec`, where
/// `expected` guards nothing against them: a guard is the digest its caller
/// read as standing, and an empty one guards nothing.
fn guard(spec: &AppliedSpec, expected: &str) -> Result<(u64, Option<Digest>), EnforceRefused> {
    let (standing, digest) = (spec.generation(), spec.digest());
    if !expected.is_empty() && digest.is_none_or(|digest| digest.to_string() != expected) {
        return Err(EnforceRefused::Guarded(standing, digest));
    }
    Ok((standing, digest))
}

/// A document's size, where it is over the bound.
pub(crate) fn over_bound(document: &[u8]) -> Option<u64> {
    let len = u64::try_from(document.len()).unwrap_or(u64::MAX);
    (len > MAX_DOCUMENT_BYTES).then_some(len)
}

/// What a document's pass found it to be.
pub(crate) struct Checked {
    /// The document's digest.
    pub(crate) digest: Digest,
    /// The document's own generation claim, or 0 if it is absent or
    /// the record supplies the number.
    pub(crate) line: u64,
    /// The generation the document validated into; `None` where it is the
    /// standing generation's document again, or in another form.
    pub(crate) next: Option<AppliedSpec>,
}

/// The validator of a document's pass, over the stages a document may
/// name and the header names its host denies.
pub(crate) type Validate =
    fn(&PolicySpec, Digest, &StageRegistry, &[&str]) -> Result<AppliedSpec, SpecError>;

/// How a document's own `generation` line is read, by the source that hands
/// the document over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Line {
    /// A source's claim of any number above the standing one, which the
    /// store's entry checks.
    Claim,
    /// An apply call's: a claim of the next number, which the call writes,
    /// or of the standing one for the standing document again.
    Next,
    /// None: the document is a rollback's and runs under the next number,
    /// whatever its bytes write. The standing document again is no change.
    Rollback,
    /// None: the document is a record's and runs under the number the
    /// record states, whatever its bytes write: a spec store's file by its
    /// name, a leader's generation by its event. A record above the
    /// standing number runs whatever its digest: a store that runs the
    /// same document under a lower number takes the record's.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(
            dead_code,
            reason = "Only a spec store's start and a standby read a record."
        )
    )]
    Record(u64),
}

/// The first steps of a document's pass, in their order: the bound, the
/// form, the digest.
fn digested(document: &[u8], format: Format) -> Result<(PolicySpec, Digest), SpecRejected> {
    if let Some(len) = over_bound(document) {
        return Err(SpecRejected::TooLarge(len));
    }
    let spec = format
        .parse(document)
        .map_err(|Refused { path, message }| SpecRejected::Unparsed {
            format: format.as_str(),
            path,
            message,
        })?;
    let digest = spec.digest()?;
    Ok((spec, digest))
}

/// A document's own pass, in its one order: the bound, the form, the
/// digest, the validator, then the line against the standing generation,
/// read as its source reads one (`rule`). Nothing is applied, nothing held
/// and no lock taken.
fn check(
    (validate, catalog, denied_headers): (Validate, &StageRegistry, &[&str]),
    (standing, standing_digest): (u64, Option<Digest>),
    document: &[u8],
    format: Format,
    rule: Line,
) -> Result<Checked, SpecRejected> {
    let (mut spec, digest) = digested(document, format)?;
    let minted = standing.saturating_add(1);
    let line = match rule {
        // The record states the number: a line in its bytes is no claim.
        Line::Record(number) => {
            spec.generation = number;
            0
        }
        Line::Rollback => {
            spec.generation = minted;
            0
        }
        Line::Claim | Line::Next => {
            if spec.generation == 0 {
                // The mint. At the last number the entry refuses it by
                // name.
                spec.generation = minted;
                0
            } else {
                spec.generation
            }
        }
    };
    // The validator's word comes first: the digest reads no line the
    // canonical form leaves out, so equal digests admit nothing. The digest
    // is taken once, above, and handed on.
    let next = validate(&spec, digest, catalog, denied_headers)?;
    if rule == Line::Next && line != 0 && line != standing && line != minted {
        return Err(SpecRejected::NotTheNext {
            offered: line,
            standing,
        });
    }
    // A record above the standing number is that number's document, not the
    // standing one's again: it runs, so the next number minted is above it.
    let recorded_above = matches!(rule, Line::Record(number) if number > standing);
    if !recorded_above && (line == 0 || line == standing) {
        if standing_digest == Some(digest) {
            // The standing generation's document again, or in another form.
            return Ok(Checked {
                digest,
                line,
                next: None,
            });
        }
        if let (true, Some(standing_digest)) = (line != 0, standing_digest) {
            return Err(SpecRejected::ChangedUnderItsLine {
                generation: standing,
                offered: digest,
                standing: standing_digest,
            });
        }
    }
    Ok(Checked {
        digest,
        line,
        next: Some(next),
    })
}

/// The rule of the store's entry in its own words: a number at or under
/// the standing one is refused by naming it.
pub(crate) fn above(standing: u64, offered: u64) -> Result<(), StaleGeneration> {
    if offered <= standing {
        return Err(StaleGeneration { standing, offered });
    }
    Ok(())
}

/// A panic's message, where it has one.
fn fault_text(fault: &(dyn Any + Send)) -> String {
    fault
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| fault.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_owned())
}

/// Runs `pass`, a document's own, with its fault held: a panic of it is a
/// refusal by name.
fn holding<T>(pass: impl FnOnce() -> Result<T, SpecRejected>) -> Result<T, SpecRejected> {
    panic::catch_unwind(AssertUnwindSafe(pass)).unwrap_or_else(|fault| {
        // The text first: a payload's own drop can panic, and it is freed
        // under a catch of its own.
        let told = fault_text(fault.as_ref());
        framework::release(fault);
        Err(SpecRejected::Faulted(told))
    })
}

/// A dry run's answer over the generation `standing`, by a document's own
/// `pass` against it: the caller's guard against that generation, the pass,
/// then the rule of the store's entry in its own words (a number at or
/// under the standing one is refused by naming it), and what a host must
/// feed before and after. The pass reads the document by its caller's rule
/// and validator. Nothing is applied, held or counted.
fn dry_run_with(
    standing: &AppliedSpec,
    expected: &str,
    pass: impl FnOnce((u64, Option<Digest>)) -> Result<Checked, SpecRejected>,
) -> Result<DryRun, EnforceRefused> {
    let stands = guard(standing, expected)?;
    let mut needs = None;
    let would_run = pass(stands).and_then(|Checked { digest, next, .. }| {
        let before = needs_rows(standing.engine());
        let Some(next) = next else {
            // The standing document again: the feed stands.
            needs = Some((before.clone(), before));
            return Ok((stands.0, digest));
        };
        let offered = next.generation();
        above(stands.0, offered)?;
        needs = Some((before, needs_rows(next.engine())));
        Ok((offered, digest))
    });
    Ok(DryRun {
        standing: stands,
        would_run,
        needs,
    })
}

/// What an apply of `document` would do against `standing`, with nothing
/// applied: the dry run of a host that runs a document and keeps no store
/// for it. `standing` is the generation that host runs, `stages` the ones
/// it validated that generation over, and `denied_headers` the header names
/// it denies a document's view.
///
/// It answers as [`DocumentManager::dry_run`] does where no spec store is
/// bound. `expected` guards the call with the digest its caller read as
/// standing, and an empty one guards nothing. The document's own pass runs with
/// its fault held, and its `generation` line is a claim of any number above the
/// standing one. The answer tells the number an apply would run the document
/// under and what a host must feed before and after. It reserves no number and
/// counts nothing.
pub fn dry_run_against(
    standing: &AppliedSpec,
    stages: &StageRegistry,
    denied_headers: &[&str],
    expected: &str,
    document: &[u8],
    format: Format,
) -> Result<DryRun, EnforceRefused> {
    let validator: (Validate, &StageRegistry, &[&str]) =
        (AppliedSpec::validate_digested, stages, denied_headers);
    dry_run_with(standing, expected, |stands| {
        holding(|| check(validator, stands, document, format, Line::Claim))
    })
}

/// A source with no further admission uses the store's own publication.
fn publish_local(store: &MemoryStore, spec: AppliedSpec) -> Result<bool, StaleGeneration> {
    store.apply(spec).map(|()| true)
}

/// The refusal of a write where no spec store is bound: the document has
/// nowhere to be persisted.
fn no_spec_store() -> SpecRejected {
    SpecRejected::Unpersisted("no spec store is bound".to_owned())
}

/// What a write answers: the number its document runs under, its digest,
/// and what a host must feed under it, `None` where it is the standing
/// document again.
type Written = (u64, Digest, Option<Vec<NeedsRow>>);

/// How many dry runs may wait their turn: a call past them is refused. What
/// waits is bounded by count here, and by bytes before it waits: a document
/// over the size bound takes no turn, and a guard longer than a digest's
/// text takes none. 32 passes of the heaviest file the
/// search finds, each under 40 ms at the median by run, are under a second
/// and a half, and 32 documents of the size bound with their guards are
/// 4 MiB. A waiting call that ends leaves the wait and frees its place.
#[cfg(feature = "controller")]
pub(crate) const DRY_RUNS_WAITING: usize = 32;

/// How many write calls may wait for the door: a call past them is refused.
/// A waiting call holds its document, so what waits is bounded as the dry
/// runs are: 32 documents of the size bound are 4 MiB.
#[cfg(feature = "controller")]
pub(crate) const WRITES_WAITING: usize = 32;

/// A host's manager of its policy document: the pass, the numbering, the
/// no-change rule, the held document and counts. With a spec store it also
/// owns the directory and the generations a rollback can name.
pub struct DocumentManager {
    pub(crate) store: Arc<MemoryStore>,
    /// The stages a document may name: the ones the bind was handed, the
    /// host's own among them.
    catalog: Arc<StageRegistry>,
    denied_headers: Vec<String>,
    validate: Validate,
    /// One apply at a time: the standing number is read, the document's
    /// pass run and the store's entry called under this lock, and a spec
    /// store's directory is written under it alone. No reader takes it. It
    /// holds the spec store, once one is bound.
    pub(crate) applying: Mutex<Option<SpecStore>>,
    /// One side of a store's bind: the bind holds this from before it
    /// applies the newest generation on file until the directory is set,
    /// and a dry run holds it while it reads the standing generation and
    /// the source's rule. No apply and no reader takes it.
    binding: Mutex<()>,
    /// The spec store's directory itself, open, once one is bound: the
    /// source takes writes, and the read call reads a generation on file
    /// through it, under no lock of the door.
    dir: OnceLock<Arc<File>>,
    /// The generations on file under a spec store, ascending, as the read
    /// call answers them. Written under the door's lock, at the bind and at
    /// each write; a reader copies one entry out.
    on_file: Mutex<Vec<Retained>>,
    /// One dry run at a time for a host that uses this turn.
    #[cfg(feature = "controller")]
    pub(crate) dry_turn: SerialTurn,
    /// Enforce and rollback share one write turn.
    #[cfg(feature = "controller")]
    pub(crate) write_turn: SerialTurn,
    /// The standing generation's document. An apply swaps it under this
    /// lock with its call of the store's entry, so a reader sees the number
    /// and the document move together and waits on no validation. `None`
    /// until a source has applied its first document.
    pub(crate) held: Mutex<Option<Held>>,
    applied: AtomicU64,
    rejected: AtomicU64,
}

impl DocumentManager {
    /// The door over `store`, for the stages of `catalog` and the header
    /// names the host denies.
    pub fn new(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
        denied_headers: Vec<String>,
    ) -> Self {
        Self::validating(
            store,
            catalog,
            denied_headers,
            AppliedSpec::validate_digested,
        )
    }

    /// [`DocumentManager::new`] with a supplied validator.
    pub(crate) fn validating(
        store: Arc<MemoryStore>,
        catalog: Arc<StageRegistry>,
        denied_headers: Vec<String>,
        validate: Validate,
    ) -> Self {
        Self {
            store,
            catalog,
            denied_headers,
            validate,
            applying: Mutex::new(None),
            binding: Mutex::new(()),
            dir: OnceLock::new(),
            on_file: Mutex::new(Vec::new()),
            #[cfg(feature = "controller")]
            dry_turn: SerialTurn::new(DRY_RUNS_WAITING),
            #[cfg(feature = "controller")]
            write_turn: SerialTurn::new(WRITES_WAITING),
            held: Mutex::new(None),
            applied: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Applies a source's document through the store's own entry. A host
    /// with one source can apply its start's document here. For an
    /// operator write, use [`DocumentManager::enforce`]. A refusal is counted,
    /// once, and the running generation stays.
    pub fn apply(&self, document: Vec<u8>, format: Format) -> Result<ApplyOutcome, SpecRejected> {
        self.apply_where(document, format, |_| None, publish_local)
    }

    /// Applies a source's `document`. Before and after the pass, `unread`
    /// can return an outcome by the standing number. The apply lock is
    /// held during both callbacks. For a changed document, `publish` runs
    /// the store's entry under the document lock. It can return `false`
    /// to leave both the store and the door as they were. An unchanged
    /// document does not call `publish`. A refusal is counted as
    /// [`DocumentManager::apply`] counts one.
    ///
    /// Both callbacks run under the door's locks. Neither may call a
    /// method of this door: `status`, `document` and the other reads take
    /// the same locks and would wait on themselves.
    pub fn apply_where(
        &self,
        document: Vec<u8>,
        format: Format,
        unread: impl Fn(u64) -> Option<ApplyOutcome>,
        publish: impl FnOnce(&MemoryStore, AppliedSpec) -> Result<bool, StaleGeneration>,
    ) -> Result<ApplyOutcome, SpecRejected> {
        self.apply_notifying(document, format, unread, publish, |_| {}, |_, _| {})
            .map_err(|rejected| self.reject(rejected))
    }

    /// Reports the pass under the apply lock before the second admission
    /// check. A refused pass is reported as `None`. Reports the standing
    /// number and digest under the document lock after a publication or
    /// when the standing document is applied again.
    pub(crate) fn apply_notifying(
        &self,
        document: Vec<u8>,
        format: Format,
        unread: impl Fn(u64) -> Option<ApplyOutcome>,
        publish: impl FnOnce(&MemoryStore, AppliedSpec) -> Result<bool, StaleGeneration>,
        on_pass: impl FnOnce(Option<(Digest, u64)>),
        on_standing: impl FnOnce(u64, Digest),
    ) -> Result<ApplyOutcome, SpecRejected> {
        let applying = self.applying.lock();
        if applying.is_some() {
            return Err(SpecRejected::SecondSource);
        }
        let standing = self.standing();
        if let Some(outcome) = unread(standing.0) {
            return Ok(outcome);
        }
        let checked = self.pass(standing, &document, format, Line::Claim);
        on_pass(
            checked
                .as_ref()
                .ok()
                .map(|checked| (checked.digest, checked.line)),
        );
        let Checked { digest, next, .. } = checked?;
        let (standing, _) = standing;
        if let Some(outcome) = unread(standing) {
            return Ok(outcome);
        }
        let hold = |generation| Held {
            generation,
            digest,
            document,
            format,
            applied: SystemTime::now(),
        };
        let Some(next) = next else {
            // It runs already. Where no bytes of it are held, these are
            // kept, so the read call has the standing one.
            let mut held = self.held.lock();
            if held.as_ref().is_none_or(|held| held.generation != standing) {
                *held = Some(hold(standing));
            }
            on_standing(standing, digest);
            return Ok(ApplyOutcome::Unchanged(standing));
        };
        let generation = next.generation();
        if !self.enter(next, hold(generation), publish, on_standing)? {
            return Ok(ApplyOutcome::Waits(standing));
        }
        drop(applying);
        Ok(ApplyOutcome::Generation(generation))
    }

    /// Publishes `next` and keeps its bytes under the lock a reader takes,
    /// so the number and the bytes move together. A publication that
    /// lands is reported under that lock, then counted and logged once.
    pub(crate) fn enter(
        &self,
        next: AppliedSpec,
        kept: Held,
        publish: impl FnOnce(&MemoryStore, AppliedSpec) -> Result<bool, StaleGeneration>,
        on_standing: impl FnOnce(u64, Digest),
    ) -> Result<bool, StaleGeneration> {
        let (generation, digest, format) = (kept.generation, kept.digest, kept.format);
        let bytes = kept.document.len();
        {
            let mut held = self.held.lock();
            if !publish(&self.store, next)? {
                return Ok(false);
            }
            *held = Some(kept);
            on_standing(generation, digest);
        }
        self.applied.fetch_add(1, Ordering::Relaxed);
        info!(
            generation,
            %digest,
            format = format.as_str(),
            bytes,
            "policy spec applied"
        );
        Ok(true)
    }

    /// A document's pass against the generation that stands, with its fault
    /// held: a panic of the pass is a refusal by name.
    pub(crate) fn pass(
        &self,
        standing: (u64, Option<Digest>),
        document: &[u8],
        format: Format,
        rule: Line,
    ) -> Result<Checked, SpecRejected> {
        let denied_headers: Vec<_> = self.denied_headers.iter().map(String::as_str).collect();
        let validator = (
            self.validate,
            self.catalog.as_ref(),
            denied_headers.as_slice(),
        );
        holding(|| check(validator, standing, document, format, rule))
    }

    /// The standing number and digest, from one load, where `expected`
    /// guards nothing against them ([`guard`]).
    #[cfg(feature = "controller")]
    pub(crate) fn guarded(&self, expected: &str) -> Result<(u64, Option<Digest>), EnforceRefused> {
        guard(self.store.snapshot().spec(), expected)
    }

    /// What an apply of `document` would do at this instant, with nothing
    /// applied, held or counted: the caller's guard against the generation
    /// that stands now, the document's pass
    /// against that same generation, then the rule of the store's entry in
    /// its own words (a number at or under the standing one is refused by
    /// naming it), and what a host must feed before and after. One load
    /// serves the four, taken with the source's rule on one side of a
    /// store's bind. The number answered is the one an apply would run
    /// the document under now. It reserves nothing: an apply that lands
    /// first takes it.
    pub fn dry_run(
        &self,
        expected: &str,
        document: &[u8],
        format: Format,
    ) -> Result<DryRun, EnforceRefused> {
        let (world, rule) = {
            let _one_side = self.binding.lock();
            (self.store.snapshot(), self.line())
        };
        dry_run_with(world.spec(), expected, |standing| {
            self.pass(standing, document, format, rule)
        })
    }

    /// Binds `dir` as the spec's source, at a start: the newest generation
    /// on file is applied under the number its name states, every
    /// generation on file is listed for the read call, and the write calls
    /// are taken from then on. A newest document that does not run refuses
    /// the bind, and so does a source bound before this one. An older file
    /// that is no document of this build stays on file and is not listed:
    /// the read call does not answer it.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn bind_store(
        &self,
        dir: SpecStore,
        deferred: impl FnOnce() -> bool,
    ) -> Result<(), StoreRefused> {
        let mut applying = self.applying.lock();
        if applying.is_some() || self.held.lock().is_some() || deferred() {
            return Err(StoreRefused::SecondSource);
        }
        // A dry run reads on one side of what follows.
        let _one_side = self.binding.lock();
        let mut on_file = Vec::with_capacity(dir.kept().len());
        if let Some((&newest, older)) = dir.kept().split_last() {
            let restored = self
                .restore(&dir, newest)
                .map_err(|source| StoreRefused::Rejected {
                    generation: newest.generation,
                    source,
                })?;
            for &persisted in older {
                match Self::told(&dir, persisted) {
                    Ok(retained) => on_file.push(retained),
                    Err(rejected) => warn!(
                        generation = persisted.generation,
                        error = %rejected,
                        "a generation on file is no document of this build: the read call does not answer it"
                    ),
                }
            }
            on_file.push(restored);
        }
        *self.on_file.lock() = on_file;
        // Set once: a second bind is refused above, under the door's lock.
        let _ = self.dir.set(dir.root());
        *applying = Some(dir);
        Ok(())
    }

    /// Applies the generation `newest` of `dir` as the store's own record:
    /// under its file's number, whatever line its bytes write, as applied
    /// when its file was written. It answers the generation as the read
    /// call lists it.
    #[cfg(any(test, feature = "controller"))]
    fn restore(&self, dir: &SpecStore, newest: Persisted) -> Result<Retained, SpecRejected> {
        let Persisted { generation, format } = newest;
        let unreadable = |error: std::io::Error| SpecRejected::Unreadable(error.to_string());
        let document = dir.read(newest).map_err(unreadable)?;
        let applied = dir.written(newest).map_err(unreadable)?;
        let rule = Line::Record(generation);
        let Checked { digest, next, .. } = self.pass(self.standing(), &document, format, rule)?;
        // Where the store runs this document already, at the file's number
        // or above it, none is entered.
        if let Some(next) = next {
            let kept = Held {
                generation,
                digest,
                document,
                format,
                applied,
            };
            self.enter(next, kept, publish_local, |_, _| {})?;
        }
        Ok(Retained {
            persisted: newest,
            digest,
            applied,
        })
    }

    /// The generation `persisted` of `dir` as the read call lists it: its
    /// digest by the first steps of a document's pass ([`digested`]), a
    /// fault held as the pass holds one, and when its file was written.
    #[cfg(any(test, feature = "controller"))]
    fn told(dir: &SpecStore, persisted: Persisted) -> Result<Retained, SpecRejected> {
        let unreadable = |error: std::io::Error| SpecRejected::Unreadable(error.to_string());
        let document = dir.read(persisted).map_err(unreadable)?;
        let (_, digest) = holding(|| digested(&document, persisted.format))?;
        let applied = dir.written(persisted).map_err(unreadable)?;
        Ok(Retained {
            persisted,
            digest,
            applied,
        })
    }

    /// Applies an operator's document where a spec store is bound: the
    /// caller's guard against the generation that stands, the document's
    /// pass, then the document persisted in the store's directory and only
    /// then handed to the store's entry, all under the door's one lock. The
    /// call writes the next number. The standing document
    /// again is no change: nothing is persisted and nothing counted. A
    /// document that does not run, or that the directory does not take, is
    /// refused with the running generation kept, and counted as rejected.
    pub fn enforce(
        &self,
        expected: &str,
        document: Vec<u8>,
        format: Format,
    ) -> Result<EnforceOutcome, EnforceRefused> {
        let mut applying = self.applying.lock();
        let world = self.store.snapshot();
        let standing = guard(world.spec(), expected)?;
        let before = needs_rows(world.spec().engine());
        let written = match applying.as_mut() {
            Some(dir) => self.write(dir, standing, document, format, Line::Next),
            None => Err(no_spec_store()),
        };
        drop(applying);
        Ok(self.enforced(standing, before, written))
    }

    /// Applies a retained document again where a spec store is bound: the document of
    /// `wanted`, a generation on file, is read from its file and written as
    /// [`DocumentManager::enforce`] writes one, under the next number whatever
    /// line its bytes write: its pass, its file, then the
    /// store's entry, under the door's one lock. The standing generation's
    /// document is no change. A generation that is not on file is not held.
    /// A document that does not run now, or that the directory does not
    /// give or take, is refused with the running generation kept, and
    /// counted as rejected. So is every roll back where no spec store is
    /// bound, in the words an enforce is refused by there.
    pub fn roll_back(&self, wanted: u64) -> Result<EnforceOutcome, EnforceRefused> {
        let mut applying = self.applying.lock();
        let Some(dir) = applying.as_mut() else {
            let standing = self.standing();
            drop(applying);
            return Ok(self.enforced(standing, Vec::new(), Err(no_spec_store())));
        };
        let on_file = dir.kept().iter().copied();
        let Some(persisted) = on_file.rev().find(|kept| kept.generation == wanted) else {
            return Err(EnforceRefused::NotHeld(self.not_held(wanted)));
        };
        let world = self.store.snapshot();
        let standing = (world.spec().generation(), world.spec().digest());
        let before = needs_rows(world.spec().engine());
        let rule = Line::Rollback;
        let written = dir
            .read(persisted)
            .map_err(|error| SpecRejected::Unreadable(error.to_string()))
            .and_then(|document| self.write(dir, standing, document, persisted.format, rule));
        drop(applying);
        Ok(self.enforced(standing, before, written))
    }

    /// A write's answer: the number its document runs under, with the feed
    /// before and after it, or its refusal, counted.
    fn enforced(
        &self,
        standing: (u64, Option<Digest>),
        before: Vec<NeedsRow>,
        written: Result<Written, SpecRejected>,
    ) -> EnforceOutcome {
        match written {
            Ok((generation, digest, after)) => EnforceOutcome {
                standing,
                ran: Ok((generation, digest)),
                needs: Some(match after {
                    Some(after) => (before, after),
                    None => (before.clone(), before),
                }),
            },
            Err(rejected) => EnforceOutcome {
                standing,
                ran: Err(self.reject(rejected)),
                needs: None,
            },
        }
    }

    /// One write under the door's lock: the pass, the persist, the entry,
    /// then the read call's list. A number that is not above the standing
    /// one is refused in the entry's words before a file is written, as a
    /// dry run refuses it. A generation the entry itself refuses goes off
    /// file again, and where it does not the refusal is a failed write's,
    /// which says both. It answers the number the document runs
    /// under with its digest, and what a host must feed under it: `None`
    /// where it is the standing document again, which is no change.
    fn write(
        &self,
        dir: &mut SpecStore,
        standing: (u64, Option<Digest>),
        document: Vec<u8>,
        format: Format,
        rule: Line,
    ) -> Result<Written, SpecRejected> {
        let Checked { digest, next, .. } = self.pass(standing, &document, format, rule)?;
        let Some(next) = next else {
            return Ok((standing.0, digest, None));
        };
        let generation = next.generation();
        above(standing.0, generation)?;
        let after = needs_rows(next.engine());
        let persisted = Persisted { generation, format };
        dir.persist(persisted, &document)
            .map_err(|error| SpecRejected::Unpersisted(error.to_string()))?;
        // The file's own time: a start lists the generation under it too.
        let applied = dir.written(persisted).unwrap_or_else(|_| SystemTime::now());
        let kept = Held {
            generation,
            digest,
            document,
            format,
            applied,
        };
        if let Err(stale) = self.enter(next, kept, publish_local, |_, _| {}) {
            // What was persisted goes before the door's lock does: a start
            // never reads a generation the entry refused. Where it does not
            // go, the store says what stays, beside the refusal.
            return Err(match dir.forget() {
                Ok(()) => stale.into(),
                Err(left) => SpecRejected::Unpersisted(format!("{stale}; {left}")),
            });
        }
        for (pruned, removed) in dir.prune() {
            if let Err(error) = removed {
                warn!(generation = pruned, %error, "a generation past the retained ones stays on file");
            }
        }
        let mut on_file = self.on_file.lock();
        on_file.push(Retained {
            persisted,
            digest,
            applied,
        });
        on_file.retain(|retained| dir.kept().contains(&retained.persisted));
        drop(on_file);
        Ok((generation, digest, Some(after)))
    }

    /// The number and the digest the store runs, from one load.
    pub fn standing(&self) -> (u64, Option<Digest>) {
        let world = self.store.snapshot();
        (world.spec().generation(), world.spec().digest())
    }

    /// Counts and logs a document a running source did not apply, its own
    /// finding included (a file it could not read): one event and one count
    /// a document, and the running generation stays.
    pub(crate) fn reject(&self, rejected: SpecRejected) -> SpecRejected {
        self.rejected.fetch_add(1, Ordering::Relaxed);
        warn!(
            generation = self.generation(),
            error = %rejected,
            "policy spec rejected: the running generation stays"
        );
        rejected
    }

    /// The apply counter by result, as it stands.
    pub fn apply_counts(&self) -> ApplyCounts {
        ApplyCounts {
            applied: self.applied.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }

    /// Whether the spec's source takes writes: a spec store is bound. A
    /// host asks before it offers an [`enforce`](Self::enforce) or a
    /// [`roll_back`](Self::roll_back).
    pub fn writable(&self) -> bool {
        self.dir.get().is_some()
    }

    /// The spec slice of the operators' snapshot, or `None` where the
    /// host has no spec source: the number and the digest the
    /// store runs, the time its document was applied where this door holds
    /// that document, and the generations on file that a rollback can name
    /// beside the standing one, ascending. A file keeps its one document,
    /// so nothing is retained under it. The holder's lock is held across
    /// the list's read: an apply swaps the holder before it lists its file,
    /// so no number above the standing one is told as retained.
    pub fn status(&self) -> Option<Status> {
        let held = self.held.lock();
        let writable = self.writable();
        if !writable && held.is_none() {
            return None;
        }
        let (generation, digest) = self.standing();
        let applied = held
            .as_ref()
            .filter(|held| held.generation == generation)
            .map(|held| held.applied);
        let retained = self
            .on_file
            .lock()
            .iter()
            .map(|retained| retained.persisted.generation)
            .filter(|number| *number != generation)
            .collect();
        Some(Status {
            generation,
            digest,
            applied,
            retained,
            writable,
        })
    }

    /// The generations on file as a refusal names them, under a spec store.
    fn listed(&self) -> Option<OnFile> {
        self.writable().then(|| {
            let on_file = self.on_file.lock();
            let number = |retained: Option<&Retained>| {
                retained.map_or(0, |retained| retained.persisted.generation)
            };
            OnFile {
                count: on_file.len(),
                span: (number(on_file.first()), number(on_file.last())),
            }
        })
    }

    /// Why no document is held for `asked`, as things stand.
    pub(crate) fn not_held(&self, asked: u64) -> NotHeld {
        let held = self.held.lock().as_ref().map(|held| held.generation);
        NotHeld {
            asked,
            held,
            standing: self.generation(),
            on_file: self.listed(),
        }
    }

    /// How this door reads a document's own line: as an apply call's where
    /// a spec store is bound, else as a file's.
    fn line(&self) -> Line {
        if self.writable() {
            Line::Next
        } else {
            Line::Claim
        }
    }

    /// The generation the store runs.
    pub(crate) fn generation(&self) -> u64 {
        self.store.snapshot().spec().generation()
    }

    /// The standing generation's document for `wanted` (`0`: the standing generation):
    /// the source's bytes with the number, the digest and the format beside
    /// them.
    pub fn document(&self, wanted: u64) -> Result<Document, NotHeld> {
        let held = self.held.lock();
        let standing = self.generation();
        let asked = if wanted == 0 { standing } else { wanted };
        match held.as_ref() {
            Some(held) if held.generation == asked && asked == standing => Ok(held.document()),
            other => Err(NotHeld {
                asked,
                held: other.map(|held| held.generation),
                standing,
                on_file: self.listed(),
            }),
        }
    }

    /// A file read prepared from the generations on file. The caller
    /// can run it off its main threads.
    #[cfg(feature = "controller")]
    pub(crate) fn retained(
        &self,
        generation: u64,
    ) -> Option<impl FnOnce() -> std::io::Result<Document> + use<>> {
        let retained = self
            .on_file
            .lock()
            .iter()
            .copied()
            .find(|retained| retained.persisted.generation == generation)?;
        let dir = Arc::clone(self.dir.get()?);
        Some(move || {
            let document = super::spec_store::read_at(&dir, retained.persisted)?;
            Ok(Document {
                generation,
                digest: retained.digest,
                document,
                format: retained.persisted.format,
                applied: retained.applied,
            })
        })
    }

    #[cfg(test)]
    fn bound(&self) -> bool {
        self.writable() || self.held.lock().is_some()
    }
}

impl Held {
    /// The source's bytes and the record beside them.
    pub(crate) fn document(&self) -> Document {
        Document {
            generation: self.generation,
            digest: self.digest,
            document: self.document.clone(),
            format: self.format,
            applied: self.applied,
        }
    }
}

#[cfg(test)]
mod tests;

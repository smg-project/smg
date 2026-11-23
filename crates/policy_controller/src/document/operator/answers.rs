//! What a host reads of a generation and of a document's pass.

use std::{fmt, time::SystemTime};

use super::{Format, MAX_DOCUMENT_BYTES};
use crate::{
    document::spec::{Digest, Needs, SpecError},
    engine::view::Phase,
    store::StaleGeneration,
};

/// What `DocumentManager::apply` did with a document. The contract's
/// `ApplyResult` is another thing: it answers the operator's apply call,
/// which is `EnforceOutcome` here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    /// It runs, as this generation.
    Generation(u64),
    /// It is the document of this generation, which stands: nothing was
    /// applied.
    Unchanged(u64),
    /// The host deferred the publication. Nothing was applied, and this
    /// generation stands.
    Waits(u64),
}

/// Why a policy spec document was not applied. The running generation
/// stays.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SpecRejected {
    /// Its source could not read it.
    #[error("not read: {0}")]
    Unreadable(String),
    /// It is over the bound a source hands over.
    #[error("the document is {0} bytes; at most {MAX_DOCUMENT_BYTES}")]
    TooLarge(u64),
    /// It is not a document of its format, it writes a field the schema
    /// does not know, or its aliases read as more than its bytes bound.
    #[error("not a policy spec in {format}: {}", located(path, message))]
    Unparsed {
        /// `yaml` or `json`.
        format: &'static str,
        /// The field the reader refused, by its path; empty where the
        /// refusal is about the document as a whole.
        path: String,
        /// The reader's message: the place in the file where it has one,
        /// and an unknown field by name.
        message: String,
    },
    /// The validator refused it.
    #[error(transparent)]
    Invalid(#[from] SpecError),
    /// It differs from the standing generation's document and writes that
    /// generation's line.
    #[error(
        "generation {generation}: the document changed under the standing line (digest {offered}, standing {standing}); write a higher number, or none"
    )]
    ChangedUnderItsLine {
        /// The standing generation, which the document's line names.
        generation: u64,
        /// The document's digest.
        offered: Digest,
        /// The standing generation's.
        standing: Digest,
    },
    /// The store's entry refused its number.
    #[error("generation: {0}")]
    Stale(#[from] StaleGeneration),
    /// It writes a line that is neither the next number, which an apply
    /// call writes, nor the standing one.
    #[error(
        "generation {offered}: an apply writes generation {}, the next after the standing generation {standing}; write that number, or none",
        .standing.saturating_add(1)
    )]
    NotTheNext {
        /// The number the document's line writes.
        offered: u64,
        /// The standing generation.
        standing: u64,
    },
    /// The spec store could not persist it: nothing was applied.
    #[error("not persisted: {0}")]
    Unpersisted(String),
    /// It is a file's, and a spec store is bound: a controller has one
    /// source.
    #[error("the controller has a spec source already")]
    SecondSource,
    /// Its own pass panicked, by the panic's message: a defect of this
    /// build that the document reaches, held at the door.
    #[error("the document's own pass faulted: {0}")]
    Faulted(String),
}

/// Why a generation has no document to read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NotHeld {
    /// The generation asked for: the standing one where the call names none.
    pub asked: u64,
    /// The generation of the document the source applied, when it applied
    /// one.
    pub held: Option<u64>,
    /// The generation the store runs.
    pub standing: u64,
    /// Under a spec store, the generations on file. `None` where none is
    /// bound: the standing generation's document alone is retained.
    pub on_file: Option<OnFile>,
}

/// The generations on file under a spec store, as a refusal names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnFile {
    /// How many are listed.
    pub count: usize,
    /// The oldest one's number and the newest one's: zero where none is.
    pub span: (u64, u64),
}

impl fmt::Display for NotHeld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no document is held for generation {}: ", self.asked)?;
        if let Some(OnFile { count, span }) = self.on_file {
            let (oldest, newest) = span;
            return match count {
                0 => f.write_str("the spec store has no generation on file"),
                _ => write!(
                    f,
                    "the spec store has {count} on file, from generation {oldest} to generation {newest}"
                ),
            };
        }
        match self.held {
            Some(held) => write!(f, "the spec source's document is generation {held}")?,
            None => f.write_str("no spec source has applied one")?,
        }
        write!(
            f,
            ", generation {} stands, and the standing generation's document alone is retained",
            self.standing
        )
    }
}

impl std::error::Error for NotHeld {}

/// The apply counter by result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApplyCounts {
    /// Documents applied as a new generation.
    pub applied: u64,
    /// Documents refused: by an apply or a write, or handed back by a
    /// source that keeps running. A dry run counts nothing.
    pub rejected: u64,
}

/// A reader's message under the path of the field it refused, where it
/// named one.
fn located<'a>(path: &'a str, message: &'a str) -> impl fmt::Display + 'a {
    struct Located<'a>(&'a str, &'a str);
    impl fmt::Display for Located<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.0.is_empty() {
                f.write_str(self.1)
            } else {
                write!(f, "{}: {}", self.0, self.1)
            }
        }
    }
    Located(path, message)
}

impl SpecRejected {
    /// The document's field the refusal is about, as a `SpecIssue` writes
    /// it ([`SpecError::field_path`]); empty where it is about the document
    /// as a whole.
    pub fn field_path(&self) -> String {
        match self {
            Self::Unreadable(_)
            | Self::TooLarge(_)
            | Self::Faulted(_)
            | Self::Unpersisted(_)
            | Self::SecondSource => String::new(),
            Self::Unparsed { path, .. } => path.clone(),
            Self::Invalid(invalid) => invalid.field_path(),
            Self::ChangedUnderItsLine { .. } | Self::Stale(_) | Self::NotTheNext { .. } => {
                "generation".to_owned()
            }
        }
    }
}

/// The standing generation as an operator reads it.
#[derive(Debug, Eq, PartialEq)]
pub struct Status {
    /// The number the store runs.
    pub generation: u64,
    /// The running document's digest, if one stands.
    pub digest: Option<Digest>,
    /// When the running document was applied, if this door holds it.
    pub applied: Option<SystemTime>,
    /// Other generations on file, in ascending order.
    pub retained: Vec<u64>,
    /// Whether a spec store is bound and takes writes.
    pub writable: bool,
}

/// A generation's document as its source wrote it.
#[derive(Debug, Eq, PartialEq)]
pub struct Document {
    /// The number it ran under.
    pub generation: u64,
    /// The digest of the document.
    pub digest: Digest,
    /// The document's original bytes.
    pub document: Vec<u8>,
    /// The form its source wrote.
    pub format: Format,
    /// When it was applied.
    pub applied: SystemTime,
}

/// What a host must feed for a model and phase under one generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeedsRow {
    /// The model the row names.
    pub model: String,
    /// The phase it serves.
    pub phase: Phase,
    /// The facts the host feeds.
    pub needs: Needs,
}

/// What a dry run of a document answers.
#[derive(Debug)]
pub struct DryRun {
    /// The number and the digest that stood when the document was checked.
    pub standing: (u64, Option<Digest>),
    /// The number an apply would have run the document under at that
    /// instant, with the document's digest, or why it would not have run.
    pub would_run: Result<(u64, Digest), SpecRejected>,
    /// What a host must feed under the standing generation and under the
    /// document's: the diff an apply returns. None where the document
    /// would not run.
    pub needs: Option<(Vec<NeedsRow>, Vec<NeedsRow>)>,
}

/// What `DocumentManager::enforce` and `roll_back` answer for a document.
/// The contract's `ApplyResult` is built from it.
#[derive(Debug)]
pub struct EnforceOutcome {
    /// The number and the digest that stood when the document was checked.
    pub standing: (u64, Option<Digest>),
    /// The number the document runs under from this apply on, with its
    /// digest: the standing number where it is the standing document
    /// again. Or why it does not run.
    pub ran: Result<(u64, Digest), SpecRejected>,
    /// What a host must feed under the generation that stood and under the
    /// document's. None where the document does not run.
    pub needs: Option<(Vec<NeedsRow>, Vec<NeedsRow>)>,
}

/// Why a dry run or write gave no verdict.
#[derive(Debug)]
pub enum EnforceRefused {
    /// The caller's guard names another digest than the standing one's.
    /// The refusal names the standing number and digest.
    Guarded(u64, Option<Digest>),
    /// No document is held for the generation the call names.
    NotHeld(NotHeld),
}

impl fmt::Display for EnforceRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guarded(standing, Some(digest)) => write!(
                f,
                "expected_digest: generation {standing} stands with digest {digest}"
            ),
            Self::Guarded(..) => f.write_str("expected_digest: no generation stands"),
            Self::NotHeld(not_held) => not_held.fmt(f),
        }
    }
}

impl std::error::Error for EnforceRefused {}

//! The policy spec's generations as the controller holds them (design 4.6
//! and 5.1's spec row): the one door a spec source applies a document
//! through, and the document the standing generation was applied from, as
//! its source wrote it.
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
//! A dry run ([`Generations::dry_run`]) runs a document's pass and applies
//! nothing: it answers what an apply would do against the generation that
//! stands at that instant, and it reserves no number. One dry run runs at
//! a time ([`Generations::dry_run_in_turn`]): the size bound budgets one
//! pass beside an apply's, and a second call waits its turn. Its guard is
//! read behind that wait, from the one load its pass reads, and no more
//! calls wait than [`DRY_RUNS_WAITING`].
//!
//! A spec store is the source an apply call writes ([`SpecStore`],
//! [`Generations::enforce`]): a directory of this replica's own. One
//! invariant holds there: **a generation is on file before the store's
//! entry is called.** The document's pass, its file and the entry run
//! under this door's one lock, so a generation that runs was persisted
//! first, and a start applies the newest one on file, under the number its
//! file's name states ([`Generations::bind_store`]). A call's document
//! writes the next number or none ([`Line::Next`]): another line is
//! refused by naming the standing one. The write calls take the door one
//! at a time, in the order they came, off the runtime's workers
//! ([`Generations::enforce_in_turn`]), and no more wait than
//! [`WRITES_WAITING`]. A controller has one source:
//! a store is bound where no file's document was applied or deferred
//! ([`StoreRefused::SecondSource`]), and a file's document is refused
//! where a store is bound ([`SpecRejected::SecondSource`]).
//!
//! A rollback ([`Generations::roll_back`]) is such a write of a document
//! the store has on file: it runs again under the next number, whatever
//! line its bytes write, and its file is the next number's. The read call
//! answers the standing generation's document from memory and any other
//! generation on file from its file ([`Generations::read`]), under the
//! digest and the time the door lists it by, and takes no lock an apply
//! holds.
//!
//! A source that keeps running hands a refusal back here
//! ([`Generations::reject`]): it is one event and one count, and the
//! running generation stays.
//!
//! The number and the digest have one holder, the store's
//! ([`World::spec`](crate::store::World::spec)). What lives here is the
//! record around the bytes: the document as the source wrote it, its format
//! and when it was applied. `GetPolicySpec` answers those bytes with the
//! number and the digest beside them, never a re-serialized form. One
//! document is held, the standing generation's. A file source retains no
//! other; a spec store keeps the last
//! [`RETAINED`](super::spec_store::RETAINED) on file.
//!
//! A replica that follows a leader has one writer at this door, the log
//! ([`Generations::following`]): it runs each generation its leader applied,
//! under the leader's number, by this build's own pass over the leader's
//! bytes, and its own file is looked at while it leads alone. A leader's
//! stream is told a generation by this door ([`Generations::sent`]): the
//! log holds the number, and the document crosses where it still stands.
//!
//! The door holds who writes, under its one lock. Under a shared lease the
//! source's entry reads the replica's fence there, as a decision reads it
//! ([`Generations::under`]): the source writes while its replica leads.
//! Else its document is not read ([`Applied::Waits`]): a look of the file's
//! watch that was in flight when the replica stopped leading applies
//! nothing beside the log, and a replica that has not led yet applies
//! nothing of its own.
//!
//! A replica under a shared lease applies no document at its start
//! ([`Generations::defer`]): the start passes its file's document, and it
//! is applied under the replica's first lead where the replica runs
//! nothing by then, at the take or behind the lead's opening
//! ([`Generations::deferred_now`]). So a replica that starts runs no number
//! of its own, and follows its leader's generation whatever its number.
//! A leader runs what its file says: the door counts the takes, and behind
//! each the file's watch reads its file as a start does, though it never
//! moved ([`Generations::takes`]).
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
//! holds a store's directory, under the store's own lock, as far as the
//! file system's locks reach. Past them a second one's write of a number
//! on file is refused by name, and it takes no write until it starts again
//! ([`SpecStore`]). A file in the store that is no document of this build
//! is listed nowhere: the read call does not answer it. A digest is the
//! applier's ([`Digest`]): it travels with its bytes, and a reader of
//! another build does not recompute it to compare.

use std::{
    any::Any,
    cell::Cell,
    fmt,
    fs::File,
    panic::{self, AssertUnwindSafe},
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use parking_lot::{Mutex, MutexGuard};
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};
use tokio::{sync::Semaphore, task::JoinError};
use tracing::{debug, info, warn};

use super::{
    spec_store::{self, Persisted, SpecStore, StoreRefused},
    worker_wire::{self, Refused as Unfollowed},
};
use crate::{
    catalog::Catalog,
    elector::Fence,
    framework::Engine,
    proto,
    spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    store::{MemoryStore, Position, StaleGeneration},
    view::Phase,
};

/// The largest document a source hands over, in bytes: 128 KiB. What sets
/// the bound is what an apply may hold beside the document, 64 MiB: the
/// budget of one apply in a controller whose container is a few hundred
/// mebibytes. By run a YAML document of this size holds under 56 MiB
/// ([`read_yaml`]). An apply runs one at a time, and a dry run one at a time
/// beside it ([`Generations::dry_run_in_turn`]). A fleet's document is far under the bound: 143 names
/// with a binding each write 11,523 bytes.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 128 << 10;

/// How many bytes a YAML document may build for each byte of the size bound
/// ([`MAX_DOCUMENT_BYTES`]), as the values it is read into. A value weighs
/// what it holds: its own size and its text, and a key its own size, its
/// text and its place in the map. A document of the size bound that writes
/// its values out builds under the bound: the densest form, a flow map of
/// one-letter keys with no values (`{a,b}`), builds 65 bytes a byte. What an
/// alias multiplies has no bound in the bytes.
///
/// The bound is the largest document's whatever a document's own size: a
/// small file that shares one profile among its names by alias reads as it
/// does written out, and no document reads as more than one of the size
/// bound may.
const BUILT_A_BYTE: usize = 80;

/// How many bytes of text a YAML document may read as for each byte of the
/// size bound, its keys among them. A document that writes its text out
/// reads as at most one and a half for each of its own (an escape of two
/// bytes writes a character of three). Past the read a document is its text
/// and its values, so under this bound an alias makes no document read as
/// more text than twice the size bound.
const TEXT_A_BYTE: usize = 2;

/// Why a document passes a bound its bytes are under: said in each refusal.
const MULTIPLIED: &str = "an alias reads as its anchor's value at each use";

/// What a value holds wherever it stands: its own size.
const NODE: usize = size_of::<Value>();

/// What a key holds beside its text: its own size, its hash and its place
/// in the map's index.
const KEY: usize = size_of::<String>() + 4 * size_of::<usize>();

/// A document's form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Format {
    Yaml,
    Json,
}

impl Format {
    /// A file's form by its name: `.json` is JSON, any other name YAML.
    pub(crate) fn of(path: &Path) -> Self {
        let json = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
        if json {
            Self::Json
        } else {
            Self::Yaml
        }
    }

    /// As `PolicySpecDocument.format` writes it.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Yaml => "yaml",
            Self::Json => "json",
        }
    }

    /// The form a contract's `format` field names, as [`Format::as_str`]
    /// writes one; none for any other name.
    pub(crate) fn named(name: &str) -> Option<Self> {
        [Self::Yaml, Self::Json]
            .into_iter()
            .find(|format| format.as_str() == name)
    }

    /// The document these bytes read as. A YAML document is read once, into
    /// the values it reads as, held under a bound ([`read_yaml`]), and the
    /// document is built from those values alone. So a YAML document reads
    /// as JSON does: text that looks like a number or a boolean is quoted
    /// where a field takes text. A field the reader refuses is named by its
    /// path.
    fn parse(self, document: &[u8]) -> Result<PolicySpec, Refused> {
        match self {
            Self::Yaml => placed(read_yaml(document).map_err(Refused::whole)?),
            // A JSON document that reads is read once, by the typed read
            // alone. A refused one is read again as values and placed as a
            // YAML document's are, for the field of its refusal.
            Self::Json => serde_json::from_slice(document).map_err(|refused| {
                match serde_json::from_slice(document).map(placed) {
                    Ok(Err(at_field)) => at_field,
                    // No values, or values that read (a key written twice
                    // keeps its last): the refusal is the typed read's own.
                    Ok(Ok(_)) | Err(_) => Refused::whole(refused),
                }
            }),
        }
    }
}

/// The typed read of a document's values: a refusal carries the path of the
/// field the read refused.
fn placed(values: Value) -> Result<PolicySpec, Refused> {
    serde_path_to_error::deserialize(values).map_err(|refused| {
        let path = refused.path().to_string();
        let message = refused.into_inner().to_string();
        // The document itself has no name.
        let path = if path == "." { String::new() } else { path };
        Refused {
            path: wanted(path, &message),
            message,
        }
    })
}

/// `path` ended by the key a typed read found missing. The map that wants
/// the key raises that refusal with no key in hand, so the read places it at
/// the map, the document itself for a key of the root, and the reader's
/// message names the key. A key a map does not know is placed at itself.
fn wanted(mut path: String, message: &str) -> String {
    let missing = message
        .strip_prefix("missing field `")
        .and_then(|rest| rest.split_once('`'));
    if let Some((key, _)) = missing {
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(key);
    }
    path
}

/// Why a reader refused a document: the path of the field, where the
/// refusal is about one, and the reader's message.
struct Refused {
    path: String,
    message: String,
}

impl Refused {
    /// A refusal of the document as a whole.
    fn whole(message: impl fmt::Display) -> Self {
        Self {
            path: String::new(),
            message: message.to_string(),
        }
    }
}

/// Reads a YAML document once, into the values it reads as, held under two
/// bounds.
///
/// A document's bytes do not bound what it reads as: an alias reads as its
/// anchor's value at each use, so one list of a thousand values under a
/// thousand aliases is a million values, and one long text under a thousand
/// aliases is a thousand copies of it. Each value is weighed as it is held,
/// by the bytes it holds, and the read ends where the weight passes
/// [`BUILT_A_BYTE`] for each byte of the size bound, or its text alone
/// [`TEXT_A_BYTE`]. What is held is what was weighed (a list and a map keep
/// no room they do not fill), and a document is built from it alone: nothing
/// a document builds is past the bounds either.
///
/// The values are JSON's: a tagged value, a number JSON does not write and
/// a key written twice are each refused by name.
///
/// What an apply holds beside the file is found by run and bounded by no
/// form: a search of the heaviest shapes finds at most 367 to 406 bytes for
/// each byte of the size bound, by the run and the build, and holds every
/// file under 448, 56 MiB (the measurement
/// `what_a_file_at_the_bound_holds`, run by name; on Linux every run of the
/// tests holds its heaviest file to the budget). Two things no bound of the
/// read sees are the larger part. The parser lists a file's nodes whole
/// before the first value is read, three nodes to two bytes at the most and
/// about a hundred bytes a node. And a document's types outweigh the values
/// they are read from: a binding of nothing is read as one value and held as
/// a type six values wide.
/// The heaviest file found writes bindings of nothing to the size bound
/// beside one profile of a thousand scorers that its aliases share among a
/// hundred names.
///
/// The parser has a limit of its own, a hundred aliases followed for each
/// node a file writes: a small nest of aliases meets it before either bound
/// and is refused in the parser's words.
fn read_yaml(document: &[u8]) -> Result<Value, String> {
    let left = Left::whole();
    Weighed(&left)
        .deserialize(serde_yaml::Deserializer::from_slice(document))
        .map_err(|error| error.to_string())
}

/// What a read has left of a document's two bounds.
struct Left {
    /// Of the bytes it may build.
    built: Cell<usize>,
    /// Of the text it may read as.
    text: Cell<usize>,
}

impl Left {
    /// The two bounds, whole: those of a document of the size bound.
    fn whole() -> Self {
        let bytes = usize::try_from(MAX_DOCUMENT_BYTES).unwrap_or(usize::MAX);
        Self {
            // The document itself is a value, an empty one too.
            built: Cell::new(bytes.saturating_mul(BUILT_A_BYTE).saturating_add(NODE)),
            text: Cell::new(bytes.saturating_mul(TEXT_A_BYTE)),
        }
    }

    /// `bytes` more built.
    fn hold<E: de::Error>(&self, bytes: usize) -> Result<(), E> {
        let Some(left) = self.built.get().checked_sub(bytes) else {
            return Err(E::custom(format_args!(
                "the document builds more than {BUILT_A_BYTE} bytes for each byte of the size bound ({MAX_DOCUMENT_BYTES}): {MULTIPLIED}"
            )));
        };
        self.built.set(left);
        Ok(())
    }

    /// `bytes` more of text, read and built.
    fn read<E: de::Error>(&self, bytes: usize) -> Result<(), E> {
        let Some(left) = self.text.get().checked_sub(bytes) else {
            return Err(E::custom(format_args!(
                "the document reads as more than {TEXT_A_BYTE} bytes of text for each byte of the size bound ({MAX_DOCUMENT_BYTES}): {MULTIPLIED}"
            )));
        };
        self.text.set(left);
        self.hold(bytes)
    }
}

/// One node of a document, read into its value against what the bounds have
/// left.
struct Weighed<'a>(&'a Left);

impl Weighed<'_> {
    /// One value more, of `text` bytes of text.
    fn weigh<E: de::Error>(&self, text: usize) -> Result<(), E> {
        self.0.hold(NODE)?;
        self.0.read(text)
    }
}

impl<'de> DeserializeSeed<'de> for Weighed<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Weighed<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a value JSON writes")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::from(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::from(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        self.weigh(0)?;
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom(format_args!("the number {value} is none JSON writes")))
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Value, E> {
        self.weigh(text.len())?;
        Ok(Value::String(text.to_owned()))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        self.weigh(0)?;
        Ok(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        self.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        self.weigh(0)?;
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Weighed(self.0))? {
            items.push(item);
        }
        items.shrink_to_fit();
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        self.weigh(0)?;
        let mut read = Vec::new();
        while let Some(key) = map.next_key_seed(Key(self.0))? {
            read.push((key, map.next_value_seed(Weighed(self.0))?));
        }
        let mut fields = Map::with_capacity(read.len());
        for (key, value) in read {
            match fields.entry(key) {
                Entry::Vacant(place) => {
                    place.insert(value);
                }
                Entry::Occupied(first) => {
                    return Err(de::Error::custom(format_args!(
                        "the key `{}` is written twice",
                        first.key()
                    )));
                }
            }
        }
        Ok(Value::Object(fields))
    }
}

/// A map's key, read as its text: a key that looks like a number is its
/// digits.
struct Key<'a>(&'a Left);

impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value = String;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for Key<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a key of text")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<String, E> {
        self.0.hold(KEY)?;
        self.0.read(text.len())?;
        Ok(text.to_owned())
    }
}

/// What an apply did with a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Applied {
    /// It runs, as this generation.
    Generation(u64),
    /// It is the document of this generation, which stands: nothing was
    /// applied.
    Unchanged(u64),
    /// The source does not write now: its replica is under a shared lease
    /// and does not lead, or no longer leads under the term the bytes were
    /// read under. Nothing was applied, and this generation stands.
    Waits(u64),
}

/// The lead a source read its bytes under ([`Generations::lead`]): its
/// entry is admitted under that lead, and under no other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Leading {
    /// A replica alone over its lease: its source writes always.
    Alone,
    /// A replica under a shared lease, whose fence showed this term.
    Term(u64),
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

/// Why `GetPolicySpec` has no document for a generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NotHeld {
    /// The generation asked for: the standing one where the call names none.
    pub(crate) asked: u64,
    /// The generation of the document the source applied, when it applied
    /// one.
    pub(crate) held: Option<u64>,
    /// The generation the store runs.
    pub(crate) standing: u64,
    /// Under a spec store, the generations on file. `None` under a file,
    /// which retains the standing generation's document alone.
    pub(crate) on_file: Option<OnFile>,
}

/// The generations on file under a spec store, as a refusal names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnFile {
    /// How many are listed.
    pub(crate) count: usize,
    /// The oldest one's number and the newest one's: zero where none is.
    pub(crate) span: (u64, u64),
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

/// The apply counter by result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Counts {
    /// Documents applied as a new generation.
    pub(crate) applied: u64,
    /// Documents a running source handed back as rejected.
    pub(crate) rejected: u64,
}

/// A generation on file under a spec store: what the read call answers
/// beside its bytes, and what a rollback finds it by.
#[derive(Clone, Copy, Debug)]
struct Retained {
    persisted: Persisted,
    digest: Digest,
    /// When its file was written.
    applied: SystemTime,
}

/// The standing generation's document as its source wrote it.
struct Held {
    generation: u64,
    digest: Digest,
    document: Vec<u8>,
    format: Format,
    applied: SystemTime,
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
    pub(crate) fn field_path(&self) -> String {
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

/// What a host must feed under `engine`, as the contract writes it: a row
/// for each of the engine's rows that has a flag to set
/// ([`Engine::needs_rows`]), in their order. A need the contract's row has
/// no flag for is not told here: a worker's requests in flight, the
/// sessions in flight, a key's in-flight count and the affinity column are
/// the controller's own to hold.
pub(crate) fn needs_summary(engine: &Engine) -> proto::NeedsSummary {
    let models = engine
        .needs_rows()
        .into_iter()
        .filter(|(_, _, needs)| {
            needs.text
                || needs.token_hashes
                || needs.text_hashes
                || needs.loads
                || needs.events
                || needs.ring
                || needs.headers
                || needs.media
        })
        .map(|(model, phase, needs)| proto::ModelNeeds {
            model: model.to_owned(),
            role: role(phase).into(),
            text: needs.text,
            token_hashes: needs.token_hashes,
            text_hashes: needs.text_hashes,
            loads: needs.loads,
            events: needs.events,
            ring: needs.ring,
            headers: needs.headers,
            media: needs.media,
        })
        .collect();
    proto::NeedsSummary { models }
}

/// The role a phase's decision places on, as the contract writes it.
const fn role(phase: Phase) -> proto::Role {
    match phase {
        Phase::Single => proto::Role::Single,
        Phase::Prefill => proto::Role::Prefill,
        Phase::Decode => proto::Role::Decode,
        Phase::Encode => proto::Role::Encode,
    }
}

/// The number and the digest of the standing generation `spec`, where
/// `expected` guards nothing against them: a guard is the digest its caller
/// read as standing, and an empty one guards nothing.
fn guard(spec: &AppliedSpec, expected: &str) -> Result<(u64, Option<Digest>), Unanswered> {
    let (standing, digest) = (spec.generation(), spec.digest());
    if !expected.is_empty() && digest.is_none_or(|digest| digest.to_string() != expected) {
        return Err(Unanswered::Guarded(standing, digest));
    }
    Ok((standing, digest))
}

/// What a dry run of a document answers.
#[derive(Debug)]
pub(crate) struct DryRun {
    /// The number and the digest that stood when the document was checked.
    pub(crate) standing: (u64, Option<Digest>),
    /// The number an apply would have run the document under at that
    /// instant, with the document's digest, or why it would not have run.
    pub(crate) would_run: Result<(u64, Digest), SpecRejected>,
    /// What a host must feed under the standing generation and under the
    /// document's: the diff an apply returns. None where the document
    /// would not run.
    pub(crate) needs: Option<proto::NeedsDiff>,
}

/// How many dry runs may wait their turn: a call past them is refused. What
/// waits is bounded by count here, and by bytes before it waits: a document
/// over the size bound takes no turn, and a guard longer than a digest's
/// text ([`DIGEST_TEXT_LEN`]) takes none. 32 passes of the heaviest file the
/// search finds, each under 40 ms at the median by run, are under a second
/// and a half, and 32 documents of the size bound with their guards are
/// 4 MiB. A waiting call that ends leaves the wait and frees its place.
pub(crate) const DRY_RUNS_WAITING: usize = 32;

/// What an enforce of a document answers.
#[derive(Debug)]
pub(crate) struct Enforced {
    /// The number and the digest that stood when the document was checked.
    pub(crate) standing: (u64, Option<Digest>),
    /// The number the document runs under from this apply on, with its
    /// digest: the standing number where it is the standing document
    /// again. Or why it does not run.
    pub(crate) ran: Result<(u64, Digest), SpecRejected>,
    /// What a host must feed under the generation that stood and under the
    /// document's. None where the document does not run.
    pub(crate) needs: Option<proto::NeedsDiff>,
}

/// How many write calls may wait for the door: a call past them is refused.
/// A waiting call holds its document, so what waits is bounded as the dry
/// runs are: 32 documents of the size bound are 4 MiB.
pub(crate) const WRITES_WAITING: usize = 32;

/// The length of a digest's text: `sha256:` and 64 hex digits. No digest is
/// written longer, so a longer guard names none.
const DIGEST_TEXT_LEN: usize = "sha256:".len() + 64;

/// Why a spec call gave no answer of its own: a dry run or a write no
/// verdict, a read no document.
#[derive(Debug)]
pub(crate) enum Unanswered {
    /// The caller's guard names another digest than the standing one's: the
    /// standing number and digest.
    Guarded(u64, Option<Digest>),
    /// As many dry runs wait their turn as may ([`DRY_RUNS_WAITING`]).
    Full,
    /// The pass's thread did not end.
    Unended(JoinError),
    /// As many write calls wait for the door as may ([`WRITES_WAITING`]).
    Waiting,
    /// A write call's thread did not end.
    Unwritten(JoinError),
    /// No document is held for the generation a read or a rollback names.
    NotHeld(NotHeld),
    /// A generation on file was not read from its file.
    Unread {
        /// The generation.
        generation: u64,
        /// What the read failed by.
        why: String,
    },
}

/// A document's size, where it is over the bound.
fn over_bound(document: &[u8]) -> Option<u64> {
    let len = u64::try_from(document.len()).unwrap_or(u64::MAX);
    (len > MAX_DOCUMENT_BYTES).then_some(len)
}

/// What a document's pass found it to be.
struct Checked {
    /// The document's digest.
    digest: Digest,
    /// The generation the document validated into; `None` where it is the
    /// standing generation's document again, or in another form.
    next: Option<AppliedSpec>,
}

/// The validator of a document's pass, over the stages a document may
/// name.
pub(crate) type Validate = fn(&PolicySpec, Digest, &Catalog) -> Result<AppliedSpec, SpecError>;

/// How a document's own `generation` line is read, by the source that hands
/// the document over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Line {
    /// A file's: a claim of any number above the standing one, which the
    /// store's entry checks.
    Claim,
    /// An apply call's: a claim of the next number, which the call writes,
    /// or of the standing one for the standing document again.
    Next,
    /// None: the document is a spec store's own and runs under the number
    /// its file's name states, whatever its bytes write.
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
    (validate, catalog): (Validate, &Catalog),
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
    let next = validate(&spec, digest, catalog)?;
    if rule == Line::Next && line != 0 && line != standing && line != minted {
        return Err(SpecRejected::NotTheNext {
            offered: line,
            standing,
        });
    }
    if line == 0 || line == standing {
        if standing_digest == Some(digest) {
            // The standing generation's document again, or in another form.
            return Ok(Checked { digest, next: None });
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
        next: Some(next),
    })
}

/// The rule of the store's entry in its own words: a number at or under
/// the standing one is refused by naming it.
fn above(standing: u64, offered: u64) -> Result<(), StaleGeneration> {
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
        crate::framework::release(fault);
        Err(SpecRejected::Faulted(told))
    })
}

/// What a write answers: the number its document runs under, its digest,
/// and what a host must feed under it, `None` where it is the standing
/// document again.
type Written = (u64, Digest, Option<proto::NeedsSummary>);

/// The controller's generations: the door a source applies through and the
/// standing document.
pub(crate) struct Generations {
    store: Arc<MemoryStore>,
    /// The stages a document may name: the ones the bind was handed, the
    /// host's own among them.
    catalog: Arc<Catalog>,
    validate: Validate,
    /// One apply at a time: the standing number is read, the document's
    /// pass run and the store's entry called under this lock, and a spec
    /// store's directory is written under it alone. No reader takes it. It
    /// holds the spec store, once one is bound.
    applying: Mutex<Option<SpecStore>>,
    /// The spec store's directory itself, open, once one is bound: the
    /// source takes writes, and the read call reads a generation on file
    /// through it, under no lock of the door.
    dir: OnceLock<Arc<File>>,
    /// The generations on file under a spec store, ascending, as the read
    /// call answers them. Written under the door's lock, at the bind and at
    /// each write; a reader copies one entry out.
    on_file: Mutex<Vec<Retained>>,
    /// The write calls let in: the one that applies and the ones that wait
    /// their turn.
    writing: Arc<Semaphore>,
    /// One write call at a time, in the order the calls came: a call awaits
    /// this turn before its thread starts, and its thread holds it. The
    /// door's lock is then taken by one write, with no other waiting on it.
    write_turn: Arc<tokio::sync::Mutex<()>>,
    /// One dry run at a time. A dry run takes neither lock of the door, and
    /// its pass holds what an apply's does: the pass holds this turn.
    dry_turn: Arc<tokio::sync::Mutex<()>>,
    /// The dry runs let in: the one whose pass runs and the ones that wait
    /// their turn.
    dry_runs: Arc<Semaphore>,
    /// The standing generation's document. An apply swaps it under this
    /// lock with its call of the store's entry, so a reader sees the number
    /// and the document move together and waits on no validation. `None`
    /// until a source has applied its first document.
    held: Mutex<Option<Held>>,
    /// The start's document of a replica under a shared lease, in its
    /// source's form, from the start that passed it until the replica's
    /// first lead applies it or a followed generation stands
    /// ([`Generations::defer`], [`Generations::deferred_now`]).
    deferred: Mutex<Option<(Vec<u8>, Format)>>,
    /// A file is this door's source: a start passed its document
    /// ([`Generations::defer`]). Never unset.
    filed: AtomicBool,
    /// The fence of a replica under a shared lease
    /// ([`Generations::under`]): the source's entry is admitted while it
    /// shows the term the entry's bytes were read under. None for a
    /// replica alone over its lease, whose source writes always.
    fence: OnceLock<Arc<Fence>>,
    /// The takes this replica made ([`Generations::at_take`]): behind each
    /// the file's watch reads its file as a start does.
    takes: AtomicU64,
    /// The leader's generation this door refused last. The same event sent
    /// again at a retry is refused from here: no second pass and no second
    /// warning.
    unfollowed: Mutex<Option<Unfollowable>>,
    applied: AtomicU64,
    rejected: AtomicU64,
}

impl Generations {
    pub(crate) fn new(store: Arc<MemoryStore>, catalog: Arc<Catalog>) -> Self {
        Self::validating(store, catalog, AppliedSpec::validate_digested)
    }

    /// [`Generations::new`] with the validator as its argument: a case
    /// hands in one that faults, or one that stalls.
    pub(crate) fn validating(
        store: Arc<MemoryStore>,
        catalog: Arc<Catalog>,
        validate: Validate,
    ) -> Self {
        Self {
            store,
            catalog,
            validate,
            applying: Mutex::new(None),
            dir: OnceLock::new(),
            on_file: Mutex::new(Vec::new()),
            writing: Arc::new(Semaphore::new(WRITES_WAITING + 1)),
            write_turn: Arc::default(),
            dry_turn: Arc::default(),
            dry_runs: Arc::new(Semaphore::new(DRY_RUNS_WAITING + 1)),
            held: Mutex::new(None),
            deferred: Mutex::new(None),
            filed: AtomicBool::new(false),
            fence: OnceLock::new(),
            takes: AtomicU64::new(0),
            unfollowed: Mutex::new(None),
            applied: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Applies `document`, as its source wrote it in `format`, read under
    /// the lead that shows now ([`Generations::apply_under`]). A refusal
    /// applies nothing and the running generation stays; what a refusal
    /// means is its source's to say: a start refuses to start, and a source
    /// that keeps running counts it ([`Generations::reject`]).
    pub(crate) fn apply(&self, document: Vec<u8>, format: Format) -> Result<Applied, SpecRejected> {
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
    ) -> Result<Applied, SpecRejected> {
        self.apply_where(lead, document, format, |_| None)
    }

    /// [`Generations::apply_under`] where `unread` may say first, by the
    /// standing number, that the document is not applied, and that is the
    /// outcome. The lead is asked three times, under the door's lock: ahead
    /// of the document's pass, which it saves; behind it, since a pass
    /// takes its time; and last where the store publishes, under the lock
    /// it publishes under, since the entry waits for that lock. Where the
    /// replica no longer leads as it did at the read, nothing is published.
    fn apply_where(
        &self,
        lead: Option<Leading>,
        document: Vec<u8>,
        format: Format,
        unread: impl Fn(u64) -> Option<Applied>,
    ) -> Result<Applied, SpecRejected> {
        let applying = self.applying.lock();
        if applying.is_some() {
            return Err(SpecRejected::SecondSource);
        }
        let standing = self.standing();
        let unread = |standing| {
            unread(standing).or_else(|| (!self.leads(lead)).then_some(Applied::Waits(standing)))
        };
        if let Some(outcome) = unread(standing.0) {
            return Ok(outcome);
        }
        let Checked { digest, next } = self.pass(standing, &document, format, Line::Claim)?;
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
            return Ok(Applied::Unchanged(standing));
        };
        let generation = next.generation();
        if !self.enter_where(next, hold(generation), None, || self.leads(lead))? {
            return Ok(Applied::Waits(standing));
        }
        drop(applying);
        Ok(Applied::Generation(generation))
    }

    /// Whether this replica leads now as it did when a source took `lead`
    /// ahead of its read: always for a replica alone over its lease, and
    /// while the fence shows that same term for one under a shared lease.
    fn leads(&self, lead: Option<Leading>) -> bool {
        lead.is_some() && self.lead() == lead
    }

    /// Hands `next` to the store's entry and keeps `kept`, the bytes it
    /// runs from, under the lock the read call takes: a reader sees the
    /// number and its bytes change together. Every generation this door
    /// applies goes through here, whatever its source: one count and one
    /// event each. `at` is the place of the leader's event the generation
    /// applies, on a replica that follows ([`Generations::following`]).
    fn enter(
        &self,
        next: AppliedSpec,
        kept: Held,
        at: Option<Position>,
    ) -> Result<(), StaleGeneration> {
        self.enter_where(next, kept, at, || true).map(drop)
    }

    /// [`Generations::enter`] where `admit` has the last word on an entry
    /// of this door's own source: the store asks it under the locks it
    /// publishes under, and where it refuses nothing is published, kept or
    /// counted, and the answer is `false`. A followed generation is the
    /// log's and asks nobody.
    fn enter_where(
        &self,
        next: AppliedSpec,
        kept: Held,
        at: Option<Position>,
        admit: impl FnOnce() -> bool,
    ) -> Result<bool, StaleGeneration> {
        let (generation, digest, format) = (kept.generation, kept.digest, kept.format);
        let bytes = kept.document.len();
        {
            let mut held = self.held.lock();
            let published = match at {
                Some(at) => self.store.follow_generation(at, next).map(|()| true)?,
                None => self.store.apply_admitted(next, admit)?,
            };
            if !published {
                return Ok(false);
            }
            *held = Some(kept);
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
    fn pass(
        &self,
        standing: (u64, Option<Digest>),
        document: &[u8],
        format: Format,
        rule: Line,
    ) -> Result<Checked, SpecRejected> {
        let validator = (self.validate, self.catalog.as_ref());
        holding(|| check(validator, standing, document, format, rule))
    }

    /// The standing number and digest, from one load, where `expected`
    /// guards nothing against them ([`guard`]).
    pub(crate) fn guarded(&self, expected: &str) -> Result<(u64, Option<Digest>), Unanswered> {
        guard(self.store.world().spec(), expected)
    }

    /// What an apply of `document` would do at this instant, with nothing
    /// applied, held or counted: the caller's guard against the generation
    /// that stands now ([`Generations::guarded`]), the document's pass
    /// against that same generation, then the rule of the store's entry in
    /// its own words (a number at or under the standing one is refused by
    /// naming it), and what a host must feed before and after. One load
    /// serves the four. The number answered is the one an apply would run
    /// the document under now. It reserves nothing: an apply that lands
    /// first takes it.
    pub(crate) fn dry_run(
        &self,
        expected: &str,
        document: &[u8],
        format: Format,
    ) -> Result<DryRun, Unanswered> {
        let world = self.store.world();
        let standing = guard(world.spec(), expected)?;
        let mut needs = None;
        let would_run = self.pass(standing, document, format, self.line()).and_then(
            |Checked { digest, next }| {
                let before = needs_summary(world.spec().engine());
                let Some(next) = next else {
                    // The standing document again: the feed stands.
                    needs = Some(proto::NeedsDiff {
                        before: Some(before.clone()),
                        after: Some(before),
                    });
                    return Ok((standing.0, digest));
                };
                let offered = next.generation();
                above(standing.0, offered)?;
                needs = Some(proto::NeedsDiff {
                    before: Some(before),
                    after: Some(needs_summary(next.engine())),
                });
                Ok((offered, digest))
            },
        );
        Ok(DryRun {
            standing,
            would_run,
            needs,
        })
    }

    /// [`Generations::dry_run`] off the runtime's workers, one at a time. A
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
    /// guard refuses it at once, and no wait holds it.
    pub(crate) async fn dry_run_in_turn(
        self: &Arc<Self>,
        expected: String,
        document: Vec<u8>,
        format: Format,
    ) -> Result<DryRun, Unanswered> {
        if over_bound(&document).is_some() || expected.len() > DIGEST_TEXT_LEN {
            return self.dry_run(&expected, &document, format);
        }
        let Ok(place) = Arc::clone(&self.dry_runs).try_acquire_owned() else {
            return Err(Unanswered::Full);
        };
        let turn = Arc::clone(&self.dry_turn).lock_owned().await;
        let door = Arc::clone(self);
        let pass = tokio::task::spawn_blocking(move || {
            let _held = (turn, place);
            door.dry_run(&expected, &document, format)
        });
        pass.await
            .unwrap_or_else(|join| Err(Unanswered::Unended(join)))
    }

    /// Binds `dir` as the spec's source, at a start: the newest generation
    /// on file is applied under the number its name states, every
    /// generation on file is listed for the read call, and the write calls
    /// are taken from then on. A newest document that does not run refuses
    /// the bind, and so does a source bound before this one. An older file
    /// that is no document of this build stays on file and is not listed:
    /// the read call does not answer it.
    pub(crate) fn bind_store(&self, dir: SpecStore) -> Result<(), StoreRefused> {
        let mut applying = self.applying.lock();
        if applying.is_some() || self.held.lock().is_some() || self.deferred.lock().is_some() {
            return Err(StoreRefused::SecondSource);
        }
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
    fn restore(&self, dir: &SpecStore, newest: Persisted) -> Result<Retained, SpecRejected> {
        let Persisted { generation, format } = newest;
        let unreadable = |error: std::io::Error| SpecRejected::Unreadable(error.to_string());
        let document = dir.read(newest).map_err(unreadable)?;
        let applied = dir.written(newest).map_err(unreadable)?;
        let rule = Line::Record(generation);
        let Checked { digest, next } = self.pass(self.standing(), &document, format, rule)?;
        // Where the store runs this document already, none is entered.
        if let Some(next) = next {
            let kept = Held {
                generation,
                digest,
                document,
                format,
                applied,
            };
            self.enter(next, kept, None)?;
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

    /// `ApplyPolicySpec` in enforce mode where a spec store is bound: the
    /// caller's guard against the generation that stands, the document's
    /// pass, then the document persisted in the store's directory and only
    /// then handed to the store's entry, all under the door's one lock. The
    /// call writes the next number ([`Line::Next`]). The standing document
    /// again is no change: nothing is persisted and nothing counted. A
    /// document that does not run, or that the directory does not take, is
    /// refused with the running generation kept, and counted as rejected.
    pub(crate) fn enforce(
        &self,
        expected: &str,
        document: Vec<u8>,
        format: Format,
    ) -> Result<Enforced, Unanswered> {
        let mut applying = self.applying.lock();
        let world = self.store.world();
        let standing = guard(world.spec(), expected)?;
        let before = needs_summary(world.spec().engine());
        let written = match applying.as_mut() {
            Some(dir) => self.write(dir, standing, document, format, Line::Next),
            None => Err(SpecRejected::Unpersisted(
                "no spec store is bound".to_owned(),
            )),
        };
        drop(applying);
        Ok(self.enforced(standing, before, written))
    }

    /// `RollbackPolicySpec` where a spec store is bound: the document of
    /// `wanted`, a generation on file, is read from its file and written as
    /// [`Generations::enforce`] writes one, under the next number whatever
    /// line its bytes write ([`Line::Record`]): its pass, its file, then the
    /// store's entry, under the door's one lock. The standing generation's
    /// document is no change. A generation that is not on file is not held.
    /// A document that does not run now, or that the directory does not
    /// give or take, is refused with the running generation kept, and
    /// counted as rejected.
    pub(crate) fn roll_back(&self, wanted: u64) -> Result<Enforced, Unanswered> {
        let mut applying = self.applying.lock();
        let found = applying.as_mut().and_then(|dir| {
            let on_file = dir.kept().iter().copied();
            let persisted = on_file.rev().find(|kept| kept.generation == wanted)?;
            Some((dir, persisted))
        });
        let Some((dir, persisted)) = found else {
            return Err(Unanswered::NotHeld(self.not_held(wanted)));
        };
        let world = self.store.world();
        let standing = (world.spec().generation(), world.spec().digest());
        let before = needs_summary(world.spec().engine());
        let rule = Line::Record(standing.0.saturating_add(1));
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
        before: proto::NeedsSummary,
        written: Result<Written, SpecRejected>,
    ) -> Enforced {
        match written {
            Ok((generation, digest, after)) => Enforced {
                standing,
                ran: Ok((generation, digest)),
                needs: Some(proto::NeedsDiff {
                    after: Some(after.unwrap_or_else(|| before.clone())),
                    before: Some(before),
                }),
            },
            Err(rejected) => Enforced {
                standing,
                ran: Err(self.reject(rejected)),
                needs: None,
            },
        }
    }

    /// One write under the door's lock: the pass, the persist, the entry,
    /// then the read call's list. A number that is not above the standing
    /// one is refused in the entry's words before a file is written, as a
    /// dry run refuses it. It answers the number the document runs
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
        let Checked { digest, next } = self.pass(standing, &document, format, rule)?;
        let Some(next) = next else {
            return Ok((standing.0, digest, None));
        };
        let generation = next.generation();
        above(standing.0, generation)?;
        let after = needs_summary(next.engine());
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
        if let Err(stale) = self.enter(next, kept, None) {
            // What was persisted goes before the door's lock does: a start
            // never reads a generation the entry refused.
            if let Err(error) = dir.forget() {
                warn!(generation, %error, "a refused generation stays on file");
            }
            return Err(stale.into());
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

    /// [`Generations::enforce`] off the runtime's workers, behind the write
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
    ) -> Result<Enforced, Unanswered> {
        if expected.len() > DIGEST_TEXT_LEN {
            // It names no digest: the guard refuses it.
            self.guarded(&expected)?;
        }
        if let Some(len) = over_bound(&document) {
            let standing = self.guarded(&expected)?;
            let ran = Err(self.reject(SpecRejected::TooLarge(len)));
            return Ok(Enforced {
                standing,
                ran,
                needs: None,
            });
        }
        self.in_turn(move |door| door.enforce(&expected, document, format))
            .await
    }

    /// [`Generations::roll_back`] off the runtime's workers, in the write
    /// calls' turn ([`Generations::in_turn`]).
    pub(crate) async fn roll_back_in_turn(
        self: &Arc<Self>,
        wanted: u64,
    ) -> Result<Enforced, Unanswered> {
        self.in_turn(move |door| door.roll_back(wanted)).await
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
        write: impl FnOnce(&Self) -> Result<Enforced, Unanswered> + Send + 'static,
    ) -> Result<Enforced, Unanswered> {
        let Ok(place) = Arc::clone(&self.writing).try_acquire_owned() else {
            return Err(Unanswered::Waiting);
        };
        let turn = Arc::clone(&self.write_turn).lock_owned().await;
        let door = Arc::clone(self);
        let written = tokio::task::spawn_blocking(move || {
            let _held = (turn, place);
            write(&door)
        });
        written
            .await
            .unwrap_or_else(|join| Err(Unanswered::Unwritten(join)))
    }

    /// The number and the digest the store runs, from one load.
    pub(crate) fn standing(&self) -> (u64, Option<Digest>) {
        let world = self.store.world();
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
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            applied: self.applied.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }

    /// Whether the controller has a spec source: a file's document was
    /// applied, a start under a shared lease passed its file's, or a spec
    /// store is bound. The spec calls are served and the spec's series
    /// rendered from then on.
    pub(crate) fn bound(&self) -> bool {
        self.writable() || self.filed.load(Ordering::Relaxed) || self.held.lock().is_some()
    }

    /// Whether the spec's source takes writes: a spec store is bound.
    pub(crate) fn writable(&self) -> bool {
        self.dir.get().is_some()
    }

    /// The spec slice of the operators' snapshot, or `None` where the
    /// controller was given no spec source: the number and the digest the
    /// store runs, the time its document was applied where this door holds
    /// that document, and the generations on file that a rollback can name
    /// beside the standing one, ascending. A file keeps its one document,
    /// so nothing is retained under it. The holder's lock is held across
    /// the list's read: an apply swaps the holder before it lists its file,
    /// so no number above the standing one is told as retained.
    pub(crate) fn status(&self) -> Option<proto::SpecStatus> {
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
        let mode = if applied.is_some() {
            proto::ApplyMode::Enforce
        } else {
            proto::ApplyMode::Unspecified
        };
        let retained = self
            .on_file
            .lock()
            .iter()
            .map(|retained| retained.persisted.generation)
            .filter(|number| *number != generation)
            .collect();
        Some(proto::SpecStatus {
            generation,
            digest: digest.map_or_else(String::new, |digest| digest.to_string()),
            applied_unix_micros: applied.map_or(0, unix_micros),
            mode: mode.into(),
            retained,
            source: if writable { "api" } else { "file" }.to_owned(),
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
    fn not_held(&self, asked: u64) -> NotHeld {
        let held = self.held.lock().as_ref().map(|held| held.generation);
        NotHeld {
            asked,
            held,
            standing: self.generation(),
            on_file: self.listed(),
        }
    }

    /// `GetPolicySpec`: the standing generation's document from memory
    /// ([`Generations::document`]), or, under a spec store, a generation on
    /// file, read from its file off the runtime's workers and answered with
    /// the digest and the time it is listed under. It takes no lock an
    /// apply holds: the read waits on no validation and on no file's sync.
    /// A file an apply removed since the list was read is not held. No turn
    /// bounds the reads: one holds a file's bytes, the size bound at most,
    /// and the runtime's blocking threads are their bound.
    pub(crate) async fn read(&self, wanted: u64) -> Result<proto::PolicySpecDocument, Unanswered> {
        let not_held = match self.document(wanted) {
            Ok(document) => return Ok(document),
            Err(not_held) => not_held,
        };
        let generation = not_held.asked;
        let listed = self
            .on_file
            .lock()
            .iter()
            .copied()
            .find(|retained| retained.persisted.generation == generation);
        let (Some(dir), Some(retained)) = (self.dir.get().cloned(), listed) else {
            return Err(Unanswered::NotHeld(not_held));
        };
        let read =
            tokio::task::spawn_blocking(move || spec_store::read_at(&dir, retained.persisted));
        let unread = |why: String| Unanswered::Unread { generation, why };
        match read.await {
            Ok(Ok(document)) => Ok(proto::PolicySpecDocument {
                generation,
                digest: retained.digest.to_string(),
                document,
                format: retained.persisted.format.as_str().to_owned(),
                applied_unix_micros: unix_micros(retained.applied),
                mode: proto::ApplyMode::Enforce.into(),
                acks: Vec::new(),
            }),
            Ok(Err(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(Unanswered::NotHeld(self.not_held(generation)))
            }
            Ok(Err(error)) => Err(unread(error.to_string())),
            Err(join) => Err(unread(join.to_string())),
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
        self.store.world().spec().generation()
    }

    /// `GetPolicySpec`'s answer for `wanted` (`0`: the standing generation):
    /// the source's bytes with the number, the digest and the format beside
    /// them. The acknowledgements arrive with the stream that carries them.
    pub(crate) fn document(&self, wanted: u64) -> Result<proto::PolicySpecDocument, NotHeld> {
        let held = self.held.lock();
        let standing = self.generation();
        let asked = if wanted == 0 { standing } else { wanted };
        match held.as_ref() {
            Some(held) if held.generation == asked && asked == standing => {
                Ok(proto::PolicySpecDocument {
                    generation: held.generation,
                    digest: held.digest.to_string(),
                    document: held.document.clone(),
                    format: held.format.as_str().to_owned(),
                    applied_unix_micros: unix_micros(held.applied),
                    mode: proto::ApplyMode::Enforce.into(),
                    acks: Vec::new(),
                })
            }
            other => Err(NotHeld {
                asked,
                held: other.map(|held| held.generation),
                standing,
                on_file: self.listed(),
            }),
        }
    }
}

/// Why a leader's generation is not followed, in the stream's words: its
/// number is not above the one this replica runs.
const NOT_ABOVE: &str = "a generation that is not above the one this replica runs";

/// The path of a generation event's number.
const NUMBER: &str = "generation.status.generation";

impl Held {
    /// The generation as a standby is sent it: its status, and the document
    /// as the read call answers it.
    fn sent(&self) -> proto::GenerationEvent {
        let digest = self.digest.to_string();
        proto::GenerationEvent {
            status: Some(proto::SpecStatus {
                generation: self.generation,
                digest: digest.clone(),
                applied_unix_micros: unix_micros(self.applied),
                mode: proto::ApplyMode::Enforce.into(),
                ..proto::SpecStatus::default()
            }),
            document: Some(proto::SpecPush {
                generation: self.generation,
                digest,
                document: self.document.clone(),
                format: self.format.as_str().to_owned(),
                needs: None,
            }),
        }
    }
}

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
            Some(held) => held.sent(),
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
            .map(Held::sent)
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
        if let Some(next) = self.pass(standing, &document, format, Line::Claim)?.next {
            above(standing.0, next.generation())?;
        }
        *self.deferred.lock() = Some((document, format));
        self.filed.store(true, Ordering::Relaxed);
        // Until the document lands a request is no decision: the unbound
        // start says so, where the start of a store with no source sheds
        // by the binding.
        if standing.0 == 0 {
            self.store.world().spec().await_document();
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
    /// ([`Generations::under`]). The file's watch asks ahead of a look, to
    /// save the look, and the source's entry asks again under the door's
    /// lock ([`Applied::Waits`]).
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
    /// ([`Elector::at_take`](crate::elector::Elector::at_take)).
    ///
    /// The take is counted, and the file's watch reads the file behind it
    /// as a start does, though it never moved
    /// ([`SpecFile`](super::spec_file::SpecFile)): a leader runs what its
    /// file says. The standing document again is no change, another one is
    /// the next generation. The start's deferred document goes through the
    /// source's entry here ([`Generations::deferred_now`]).
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
    /// reason ([`AppliedSpec::await_document`]). A refusal is counted: the
    /// same bytes passed at the start.
    ///
    /// Behind a generation the replica followed, nothing is applied: the
    /// start's bytes are as old as the start, and the watch reads the file
    /// as it stands. The bytes are let go once a generation stands.
    pub(crate) fn deferred_now(&self) {
        let Some((document, format)) = self.deferred.lock().clone() else {
            return;
        };
        let followed = |standing| (standing != 0).then_some(Applied::Unchanged(standing));
        match self.apply_where(self.lead(), document, format, followed) {
            // No term shows, or the lead ended under the pass: it waits on.
            Ok(Applied::Waits(_)) => return,
            Ok(Applied::Generation(_) | Applied::Unchanged(_)) => {}
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
    /// behind each, as a start does, though the file never moved: a leader
    /// runs what its file says.
    pub(crate) fn takes(&self) -> u64 {
        self.takes.load(Ordering::Relaxed)
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
            (Some((next, kept)), at) => door.enter(next, kept, at).map_err(|stale| {
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

pub(super) fn unix_micros(at: SystemTime) -> i64 {
    at.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|since| i64::try_from(since.as_micros()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        cell::RefCell,
        collections::BTreeSet,
        fmt::Write as _,
        io::Write as _,
        sync::{atomic::AtomicBool, mpsc, OnceLock, Weak},
        thread,
        time::{Duration, Instant},
    };

    use tokio::task::JoinSet;

    use super::*;
    #[cfg(unix)]
    use crate::controller::spec_file::tests::pipe_at;
    use crate::{
        controller::spec_store::{
            tests::{name_of, names},
            RETAINED,
        },
        elector::{ElectionSettings, Elector, MemoryLease},
        Candidate, Constraints, Filter, RequestFacts,
    };

    /// A document in the form of the design's example: no `generation` line.
    pub(crate) const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";
    /// Another one.
    pub(crate) const RANDOM: &str =
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

    fn generations() -> (Arc<MemoryStore>, Generations) {
        let store = Arc::new(MemoryStore::new());
        (Arc::clone(&store), Generations::new(store, stock()))
    }

    /// `document` applied as YAML.
    fn yaml(
        generations: &Generations,
        document: impl Into<Vec<u8>>,
    ) -> Result<Applied, SpecRejected> {
        generations.apply(document.into(), Format::Yaml)
    }

    /// `document` under a written line.
    pub(crate) fn with_line(document: &str, line: u64) -> Vec<u8> {
        format!("generation: {line}\n{document}").into_bytes()
    }

    /// `document` grown to exactly `len` bytes by one comment line at its
    /// end.
    pub(crate) fn padded(document: &str, len: usize) -> Vec<u8> {
        let mut bytes = document.as_bytes().to_vec();
        assert!(len > bytes.len() + 1, "no room for a comment line");
        bytes.push(b'#');
        bytes.resize(len - 1, b'x');
        bytes.push(b'\n');
        bytes
    }

    /// A document whose view's first header is one text of `text` bytes
    /// under an anchor, with `aliases` aliases of it after it: four bytes
    /// an alias in the file, the text's bytes an alias in what it reads as.
    pub(crate) fn long_text_under_aliases(text: usize, aliases: usize) -> String {
        let mut document = String::with_capacity(text + 4 * aliases + 256);
        document.push_str(LEAST_LOAD);
        document.push_str("view:\n  headers: [&h ");
        document.extend(std::iter::repeat_n('x', text));
        for _ in 0..aliases {
            document.push_str(",*h ");
        }
        document.push_str("]\n");
        document
    }

    /// What `document` reads as, with the bytes it built and the text among
    /// them.
    fn weighed(document: &str) -> (Value, usize, usize) {
        let left = Left {
            built: Cell::new(usize::MAX),
            text: Cell::new(usize::MAX),
        };
        let read = Weighed(&left)
            .deserialize(serde_yaml::Deserializer::from_str(document))
            .expect("it reads");
        (
            read,
            usize::MAX - left.built.get(),
            usize::MAX - left.text.get(),
        )
    }

    /// The refusals of the read's two bounds.
    const PAST_BUILT: &str = "builds more than 80 bytes for each byte of the size bound (131072)";
    const PAST_TEXT: &str =
        "reads as more than 2 bytes of text for each byte of the size bound (131072)";

    /// The stock build's stages.
    fn stock() -> Arc<Catalog> {
        Arc::new(Catalog::builtin())
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

    /// A validator that panics on a document that picks at random, and
    /// validates any other.
    pub(crate) fn faulting(
        document: &PolicySpec,
        digest: Digest,
        catalog: &Catalog,
    ) -> Result<AppliedSpec, SpecError> {
        let random = document
            .profiles
            .values()
            .any(|profile| profile.pick == "random".into());
        assert!(!random, "a fault of the validator");
        AppliedSpec::validate_digested(document, digest, catalog)
    }

    /// The number and the digest the store runs.
    fn standing(store: &MemoryStore) -> (u64, Option<Digest>) {
        let world = store.world();
        (world.spec().generation(), world.spec().digest())
    }

    /// The door builds a generation with its document's lists: a document
    /// of engine families alone, applied through it, lists the member that
    /// joins after.
    #[test]
    fn a_document_applied_through_the_door_lists_the_member_that_joins() {
        use crate::store::{OutsideLists, WorkerFacts, WorkersStore};

        let (store, generations) = generations();
        let document = format!("{RANDOM}view:\n  engines: [alpha]\n");
        yaml(&generations, document).expect("a valid document");
        let join = |key: &str, engine: &str| {
            store.upsert(WorkerFacts {
                key: key.to_owned(),
                engine: key.to_owned(),
                incarnation: 1,
                engine_kind: engine.to_owned(),
                models: vec!["m".to_owned()],
                dp_size: 1,
                ..WorkerFacts::default()
            });
            // The members are read under the lists at the build the join
            // owes.
            store.publish_now();
        };
        join("http://a:8000", "alpha");
        assert_eq!(store.outside_lists(), OutsideLists::default());
        join("http://b:8000", "beta");
        let outside = OutsideLists {
            engines: 1,
            zones: 0,
        };
        assert_eq!(store.outside_lists(), outside);
    }

    #[test]
    fn a_files_form_is_read_off_its_name() {
        for (name, format) in [
            ("policy.yaml", Format::Yaml),
            ("policy.yml", Format::Yaml),
            ("policy", Format::Yaml),
            ("policy.json", Format::Json),
            ("POLICY.JSON", Format::Json),
            ("..data/policy.json", Format::Json),
        ] {
            assert_eq!(Format::of(Path::new(name)), format, "{name}");
        }
        assert_eq!(
            (Format::Yaml.as_str(), Format::Json.as_str()),
            ("yaml", "json")
        );
    }

    /// The writer that applies mints the number: a document with no line
    /// starts as generation 1, and a changed one is the next.
    #[test]
    fn a_document_with_no_line_starts_as_generation_one_and_a_changed_one_is_the_next() {
        let (store, generations) = generations();
        assert!(!generations.bound());
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        assert!(generations.bound());
        assert_eq!(standing(&store).0, 1);
        assert_eq!(yaml(&generations, RANDOM), Ok(Applied::Generation(2)));
        assert_eq!((standing(&store).0, generations.generation()), (2, 2));
    }

    /// A written line is a claim the store's entry checks: a number above
    /// the standing one is admitted as written, the standing document
    /// included; a lower one is refused by naming the standing one, and the
    /// running generation stays.
    #[test]
    fn a_written_line_is_a_claim_the_entry_checks() {
        let (store, generations) = generations();
        let first = yaml(&generations, with_line(LEAST_LOAD, 5));
        assert_eq!(first, Ok(Applied::Generation(5)));
        let second = yaml(&generations, with_line(RANDOM, 9));
        assert_eq!(second, Ok(Applied::Generation(9)));
        let random = standing(&store);

        let lower = yaml(&generations, with_line(LEAST_LOAD, 4)).expect_err("a number under 9");
        let stale = StaleGeneration {
            standing: 9,
            offered: 4,
        };
        assert_eq!(lower, SpecRejected::Stale(stale));
        assert_eq!(
            lower.to_string(),
            "generation: generation 4 is not above the standing generation 9"
        );
        assert_eq!(standing(&store), random, "the running generation stays");

        let again = yaml(&generations, with_line(RANDOM, 12));
        assert_eq!(again, Ok(Applied::Generation(12)));
        assert_eq!(
            standing(&store),
            (12, random.1),
            "the standing document under a higher line runs under that line"
        );
    }

    /// One number never names two documents: a changed document under the
    /// standing line is refused by name, and the running generation with
    /// its document stays.
    #[test]
    fn a_changed_document_under_the_standing_line_is_refused_by_name_and_the_running_one_stays() {
        let (store, generations) = generations();
        let first = with_line(LEAST_LOAD, 3);
        assert_eq!(
            yaml(&generations, first.clone()),
            Ok(Applied::Generation(3))
        );
        let running = standing(&store);
        let (_, Some(running_digest)) = running else {
            panic!("an applied generation has a digest");
        };

        let refused = yaml(&generations, with_line(RANDOM, 3)).expect_err("another document");
        let SpecRejected::ChangedUnderItsLine {
            generation: 3,
            offered,
            standing: named,
        } = refused
        else {
            panic!("refused as {refused:?}");
        };
        assert_eq!(named, running_digest);
        assert_ne!(offered, running_digest);
        assert!(
            refused
                .to_string()
                .starts_with("generation 3: the document changed under the standing line"),
            "{refused}"
        );

        assert_eq!(standing(&store), running, "the running generation stays");
        let answer = generations.document(0).expect("the standing document");
        assert_eq!((answer.generation, answer.document), (3, first));
    }

    /// A file that changed its form alone (JSON for YAML, the order of its
    /// keys, a comment, the standing line written out) is no change: nothing
    /// is applied, and the bytes held stay the applied ones.
    #[test]
    fn a_document_that_only_changed_its_form_is_no_change() {
        let (store, generations) = generations();
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let running = standing(&store);

        let json = br#"{"profiles":{"default":{"pick":"best","score":["least-load"],"filter":["healthy"]}},"apiVersion":"smg.policy/v1"}"#;
        let commented = format!("# the same document\n{LEAST_LOAD}");
        for (document, format) in [
            (json.to_vec(), Format::Json),
            (commented.into_bytes(), Format::Yaml),
            (with_line(LEAST_LOAD, 1), Format::Yaml),
        ] {
            let outcome = generations.apply(document, format);
            assert_eq!(outcome, Ok(Applied::Unchanged(1)));
        }
        assert_eq!(standing(&store), running);
        let answer = generations.document(0).expect("the standing document");
        assert_eq!(answer.document, LEAST_LOAD.as_bytes());
    }

    /// The apply counter by result: a new generation counts as applied, a
    /// refusal a running source hands back counts as rejected, once, and the
    /// standing document again counts as neither.
    #[test]
    fn the_counts_follow_the_outcomes() {
        let (_store, generations) = generations();
        let counts = |applied, rejected| Counts { applied, rejected };
        assert_eq!(generations.counts(), counts(0, 0));
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Unchanged(1)));
        assert_eq!(generations.counts(), counts(1, 0));

        let refused = yaml(&generations, with_line(RANDOM, 1))
            .map_err(|rejected| generations.reject(rejected));
        assert!(matches!(
            refused,
            Err(SpecRejected::ChangedUnderItsLine { .. })
        ));
        assert_eq!(generations.counts(), counts(1, 1));
        assert_eq!(yaml(&generations, RANDOM), Ok(Applied::Generation(2)));
        assert_eq!(generations.counts(), counts(2, 1));
    }

    /// A source that finds its document running already (a controller built
    /// in code handed it to the store) applies nothing and holds the
    /// document's bytes, so the read call answers the standing generation.
    #[test]
    fn a_source_that_finds_its_document_running_holds_its_bytes() {
        let (store, generations) = generations();
        let mut running: PolicySpec = serde_yaml::from_str(LEAST_LOAD).expect("a document");
        running.generation = 5;
        store
            .apply(AppliedSpec::validate(&running).expect("a valid document"))
            .expect("above the unbound start");
        assert!(!generations.bound());

        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Unchanged(5)));
        assert!(generations.bound());
        let answer = generations.document(0).expect("the standing document");
        assert_eq!((answer.generation, answer.document), (5, LEAST_LOAD.into()));

        // The same over older held bytes: a second writer of the store runs
        // another document, and the source that then hands that document
        // over holds its bytes in place of the older generation's.
        let mut next: PolicySpec = serde_yaml::from_str(RANDOM).expect("a document");
        next.generation = 6;
        store
            .apply(AppliedSpec::validate(&next).expect("a valid document"))
            .expect("above the standing one");
        assert_eq!(yaml(&generations, RANDOM), Ok(Applied::Unchanged(6)));
        let answer = generations.document(0).expect("the standing document");
        assert_eq!((answer.generation, answer.document), (6, RANDOM.into()));
    }

    /// At the last number there is no next one: the mint stays there and
    /// the entry refuses it by name, as it refuses a line below the
    /// standing one. A refusal of the entry leaves the held bytes.
    #[test]
    fn a_mint_at_the_last_number_is_refused_by_name_and_the_held_bytes_stay() {
        let (store, generations) = generations();
        let last = with_line(LEAST_LOAD, u64::MAX);
        assert_eq!(
            yaml(&generations, last.clone()),
            Ok(Applied::Generation(u64::MAX))
        );
        for (document, offered) in [
            (RANDOM.as_bytes().to_vec(), u64::MAX),
            (with_line(RANDOM, 3), 3),
        ] {
            let stale = StaleGeneration {
                standing: u64::MAX,
                offered,
            };
            assert_eq!(
                yaml(&generations, document),
                Err(SpecRejected::Stale(stale))
            );
        }
        assert_eq!(standing(&store).0, u64::MAX);
        let answer = generations.document(0).expect("the standing document");
        assert_eq!(answer.document, last);
    }

    /// The held document is answered only while its generation stands:
    /// once another writer's generation runs it is not answered under its
    /// own number either, and the refusal names the three numbers.
    #[test]
    fn a_held_document_is_answered_only_while_its_generation_stands() {
        let (store, generations) = generations();
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let mut next: PolicySpec = serde_yaml::from_str(RANDOM).expect("a document");
        next.generation = 2;
        store
            .apply(AppliedSpec::validate(&next).expect("a valid document"))
            .expect("above the standing one");
        for (wanted, asked) in [(0, 2), (1, 1), (2, 2)] {
            let not_held = generations.document(wanted).expect_err("no document");
            let expected = NotHeld {
                asked,
                held: Some(1),
                standing: 2,
                on_file: None,
            };
            assert_eq!(not_held, expected, "asked for {wanted}");
        }
    }

    /// A document that faults its own pass is a document that does not run:
    /// the door holds the panic and refuses the document by the panic's
    /// message, the running generation and its bytes stay, and the door
    /// stands open for the next document.
    #[test]
    fn a_document_that_faults_its_pass_is_refused_by_name() {
        let store = Arc::new(MemoryStore::new());
        let generations = Generations::validating(Arc::clone(&store), stock(), faulting);
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let running = standing(&store);

        assert_eq!(
            yaml(&generations, RANDOM),
            Err(SpecRejected::Faulted("a fault of the validator".to_owned()))
        );
        assert_eq!(standing(&store), running);
        let answer = generations.document(0).expect("the standing document");
        assert_eq!(answer.document, LEAST_LOAD.as_bytes());
        assert_eq!(
            yaml(&generations, with_line(LEAST_LOAD, 4)),
            Ok(Applied::Generation(4))
        );
    }

    /// A panic's payload is the document's too: one whose own drop panics
    /// is freed under a catch of its own, once its text is read, and the
    /// door refuses the document as it does any fault.
    #[test]
    fn a_fault_whose_payload_faults_at_its_drop_is_held() {
        /// A payload that panics when it is dropped.
        struct Bomb;
        impl Drop for Bomb {
            fn drop(&mut self) {
                panic!("the payload's own drop");
            }
        }
        fn dropping(
            document: &PolicySpec,
            digest: Digest,
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            let random = document
                .profiles
                .values()
                .any(|profile| profile.pick == "random".into());
            if random {
                panic::panic_any(Bomb);
            }
            AppliedSpec::validate_digested(document, digest, catalog)
        }
        let store = Arc::new(MemoryStore::new());
        let generations = Generations::validating(Arc::clone(&store), stock(), dropping);
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));

        assert_eq!(
            yaml(&generations, RANDOM),
            Err(SpecRejected::Faulted("a panic with no message".to_owned()))
        );
        assert_eq!(standing(&store).0, 1);
        assert_eq!(
            yaml(&generations, with_line(LEAST_LOAD, 4)),
            Ok(Applied::Generation(4))
        );
    }

    /// A fault is told by its panic's message, a literal or a formatted
    /// one; a panic that carries no text is told as one.
    #[test]
    fn a_fault_is_told_by_its_message() {
        let told = |fault: Box<dyn Any + Send>| fault_text(fault.as_ref());
        assert_eq!(told(Box::new("a literal")), "a literal");
        assert_eq!(
            told(Box::new(String::from("a formatted one"))),
            "a formatted one"
        );
        assert_eq!(told(Box::new(7_u8)), "a panic with no message");
    }

    /// A reader waits on no pass: while a document is validated the read
    /// call answers the standing document.
    #[test]
    fn a_read_during_an_apply_waits_on_no_validation() {
        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASED: AtomicBool = AtomicBool::new(false);
        /// Until `flag` turns, ten seconds at the most.
        fn until(flag: &AtomicBool, never: &str) {
            for _ in 0..10_000 {
                if flag.load(Ordering::SeqCst) {
                    return;
                }
                thread::sleep(Duration::from_millis(1));
            }
            panic!("{never}");
        }
        fn stalling(
            document: &PolicySpec,
            digest: Digest,
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            if document.generation == 2 {
                ENTERED.store(true, Ordering::SeqCst);
                until(&RELEASED, "the validator was never released");
            }
            AppliedSpec::validate_digested(document, digest, catalog)
        }

        let store = Arc::new(MemoryStore::new());
        let generations = Arc::new(Generations::validating(store, stock(), stalling));
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let applying = {
            let generations = Arc::clone(&generations);
            thread::spawn(move || yaml(&generations, RANDOM))
        };
        until(&ENTERED, "the second document never reached the validator");
        let answer = generations.document(0).expect("the standing document");
        assert_eq!((answer.generation, answer.document), (1, LEAST_LOAD.into()));
        assert!(generations.bound());
        RELEASED.store(true, Ordering::SeqCst);
        assert_eq!(
            applying.join().expect("the apply's thread"),
            Ok(Applied::Generation(2))
        );
    }

    /// The number moves under the document's lock: an apply calls the
    /// store's entry with the lock held, so no reader finds the new number
    /// beside the old document. The store lets an old generation go inside
    /// its entry, and the one let go here holds a filter of the case's own
    /// that writes down, at its end, whether the lock was held.
    #[test]
    fn the_stores_entry_is_called_under_the_documents_lock() {
        /// Keeps every worker, and tells at its end how it found the lock.
        #[derive(Debug)]
        struct LetGo {
            door: Arc<OnceLock<Weak<Generations>>>,
            found: Arc<Mutex<Vec<bool>>>,
        }

        impl Filter for LetGo {
            fn name(&self) -> &'static str {
                "let-go"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }
        }

        impl Drop for LetGo {
            fn drop(&mut self) {
                if let Some(door) = self.door.get().and_then(Weak::upgrade) {
                    self.found.lock().push(door.held.try_lock().is_none());
                }
            }
        }

        let door = Arc::new(OnceLock::new());
        let found = Arc::new(Mutex::new(Vec::new()));
        let mut stages = Catalog::builtin();
        {
            let (door, found) = (Arc::clone(&door), Arc::clone(&found));
            let build = move || -> Box<dyn Filter> {
                Box::new(LetGo {
                    door: Arc::clone(&door),
                    found: Arc::clone(&found),
                })
            };
            stages.filter("let-go", build).expect("a free name");
        }
        let store = Arc::new(MemoryStore::new());
        let generations = Arc::new(Generations::new(store, Arc::new(stages)));
        door.set(Arc::downgrade(&generations)).expect("set once");

        // The store keeps the generation before the standing one: the third
        // apply lets the first go.
        for (number, pick) in (1..).zip(["random", "round-robin", "best"]) {
            let document = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [let-go]\n    pick: {pick}\n"
            );
            assert_eq!(
                yaml(&generations, document),
                Ok(Applied::Generation(number))
            );
        }
        assert_eq!(*found.lock(), [true], "the first generation's end");
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
        catalog: &Catalog,
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
        AppliedSpec::validate_digested(document, digest, catalog)
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
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
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
        assert_eq!(applied, Ok(Applied::Generation(2)));
        // The first call read generation 1 before its pass stood.
        let unguarded = unguarded.expect("a verdict");
        assert_eq!(unguarded.would_run.map(|(number, _)| number), Ok(2));
        assert!(
            matches!(guarded, Err(Unanswered::Guarded(2, Some(_)))),
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
            matches!(small, Err(Unanswered::Guarded(0, None))),
            "{small:?}"
        );
        let document = vec![b'#'; usize::try_from(over).expect("the bound fits")];
        let large = generations
            .dry_run_in_turn(guard, document, Format::Yaml)
            .await;
        assert!(
            matches!(large, Err(Unanswered::Guarded(0, None))),
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
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
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
            matches!(refused, Err(Unanswered::Guarded(1, Some(_)))),
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
            while door.dry_runs.available_permits() != free {
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
        assert_eq!(
            generations.dry_runs.available_permits(),
            DRY_RUNS_WAITING + 1
        );
    }

    /// The validator's word comes before the digest's. A `kind` line is in
    /// no digest and the validator refuses it: the standing document with
    /// one added is refused by name, under no line and under the standing
    /// one, and where no bytes are held yet these are not kept.
    #[test]
    fn a_document_the_validator_refuses_is_refused_though_its_digest_stands() {
        let kind = || SpecRejected::Invalid(SpecError::Kind("PolicySpec".to_owned()));
        let with_kind = format!("kind: PolicySpec\n{LEAST_LOAD}");

        {
            let (store, generations) = generations();
            assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
            let running = standing(&store);
            assert_eq!(yaml(&generations, with_kind.as_str()), Err(kind()));
            assert_eq!(yaml(&generations, with_line(&with_kind, 1)), Err(kind()));
            assert_eq!(standing(&store), running);
            let answer = generations.document(0).expect("the standing document");
            assert_eq!(answer.document, LEAST_LOAD.as_bytes());
        }

        // A store that runs the document already, no bytes of it held: the
        // refused ones are not the read call's.
        let (store, generations) = generations();
        let mut running: PolicySpec = serde_yaml::from_str(LEAST_LOAD).expect("a document");
        running.generation = 5;
        store
            .apply(AppliedSpec::validate(&running).expect("a valid document"))
            .expect("above the unbound start");
        assert_eq!(yaml(&generations, with_kind.as_str()), Err(kind()));
        assert!(!generations.bound());
    }

    /// A later document the validator or the parser refuses is rejected
    /// whole, and the running generation stays.
    #[test]
    fn an_invalid_document_is_rejected_whole_and_the_running_generation_stays() {
        let (store, generations) = generations();
        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let running = standing(&store);

        let unknown_filter = SpecError::UnknownFilter {
            profile: "default".to_owned(),
            name: "nowhere".to_owned(),
        };
        assert_eq!(
            yaml(&generations, LEAST_LOAD.replace("[healthy]", "[nowhere]")),
            Err(SpecRejected::Invalid(unknown_filter))
        );
        match yaml(&generations, format!("{LEAST_LOAD}colour: blue\n")) {
            Err(SpecRejected::Unparsed {
                format: "yaml",
                message,
                ..
            }) => assert!(message.contains("colour"), "{message}"),
            other => panic!("applied as {other:?}"),
        }
        match generations.apply(b"{ profiles".to_vec(), Format::Json) {
            Err(SpecRejected::Unparsed { format: "json", .. }) => {}
            other => panic!("applied as {other:?}"),
        }
        assert_eq!(standing(&store), running);
    }

    /// The bytes do not bound what a YAML document reads as: an alias reads
    /// as its anchor's value at each use. One list of 390 numbers under 390
    /// aliases is 153 thousand values in some 5,000 bytes; one text of 8,192
    /// bytes under 2,048 aliases is sixteen mebibytes of text in 16,515; a
    /// nest of a hundred one-item lists under 20,967 aliases is two million
    /// lists in a file of the size bound. Each is refused by what a document
    /// of the size bound may build or read as, before a document is built
    /// from it. An alias under the bounds reads as its anchor's value.
    #[test]
    fn a_yaml_document_builds_no_more_than_the_size_bound_may() {
        let refused = |document: String, bound: &str| {
            let (store, generations) = generations();
            let bytes = document.len();
            match yaml(&generations, document) {
                Err(SpecRejected::Unparsed {
                    format: "yaml",
                    message,
                    ..
                }) => assert!(message.contains(bound), "{message}"),
                other => panic!("{bytes} bytes applied as {other:?}"),
            }
            assert_eq!(standing(&store).0, 0);
        };
        let numbers = vec!["1"; 390].join(", ");
        let aliases = vec!["*numbers"; 390].join(", ");
        refused(
            format!("{LEAST_LOAD}x: &numbers [{numbers}]\ny: [{aliases}]\n"),
            PAST_BUILT,
        );
        refused(long_text_under_aliases(8_192, 2_048), PAST_TEXT);
        let nest = format!("{}x{}", "[".repeat(100), "]".repeat(100));
        let aliases = vec!["*nest"; 20_967].join(",");
        let nested = format!("{LEAST_LOAD}zz:\n  a: &nest {nest}\n  b: [{aliases}]\n");
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let at_the_bound = String::from_utf8(padded(&nested, bound)).expect("text");
        refused(at_the_bound, PAST_BUILT);

        // Under the bound an alias reads as its anchor's value: the three
        // profiles reach the validator, which refuses them for its own
        // reason.
        let shared = "apiVersion: smg.policy/v1\nprofiles:\n  default: {filter: &names [healthy, not-tried], pick: random}\n  p1: {filter: *names, pick: random}\n  p2: {filter: *names, pick: random}\n";
        let (_store, generations) = generations();
        assert_eq!(
            yaml(&generations, shared),
            Err(SpecRejected::Invalid(SpecError::NoBindings { profiles: 3 }))
        );
    }

    /// A YAML document reads as the values JSON writes, a key as its text.
    /// Each value weighs what it holds: its own size and its text, a key
    /// its own size, its text and its place in the map, and a list keeps no
    /// room it does not fill. What JSON does not write is refused by name: a
    /// tagged value, a number past its range, a key written twice.
    #[test]
    fn a_yaml_document_reads_as_the_values_json_writes() {
        let read = read_yaml(
            b"a: [1, -1, 1.5, true, ~, text, \"7\"]\n1: {true: c}\ne: &anchor [x, y]\nf: *anchor\n",
        );
        let expected = serde_json::json!({
            "a": [1, -1, 1.5, true, null, "text", "7"],
            "1": {"true": "c"},
            "e": ["x", "y"],
            "f": ["x", "y"],
        });
        assert_eq!(read, Ok(expected));
        assert_eq!(read_yaml(b""), Ok(Value::Null));

        let (read, built, text) = weighed("{a,b}");
        assert_eq!(read, serde_json::json!({"a": null, "b": null}));
        assert_eq!(
            (built, text),
            (3 * NODE + 2 * (KEY + 1), 2),
            "a map and two nulls, under two keys of a byte"
        );
        let (read, built, text) = weighed("[a,[b]]");
        assert_eq!(read, serde_json::json!(["a", ["b"]]));
        assert_eq!(
            (built, text),
            (4 * NODE + 2, 2),
            "two lists and two texts of a byte"
        );
        let Value::Array(items) = read else {
            panic!("a list reads as a list")
        };
        assert_eq!(items.capacity(), 2, "a list keeps no room it does not fill");

        for (document, named) in [
            ("a: !tagged 1\n", "enum"),
            ("a: 340282366920938463463374607431768211455\n", "u128"),
            ("a: .inf\n", "is none JSON writes"),
            ("a: 1\na: 2\n", "the key `a` is written twice"),
        ] {
            match read_yaml(document.as_bytes()) {
                Err(message) => assert!(message.contains(named), "{document}: {message}"),
                Ok(read) => panic!("{document} read as {read}"),
            }
        }
    }

    /// A document reads as no more text than twice the size bound, its keys
    /// among it: the text an alias multiplies is refused by name where the
    /// bytes it builds are under their own bound, and text written out reads
    /// as it is, an escape of two bytes for a character of three too.
    #[test]
    fn a_yaml_document_reads_as_no_more_text_than_the_size_bound_may() {
        let refused = |document: String| match read_yaml(document.as_bytes()) {
            Err(message) => assert!(message.contains(PAST_TEXT), "{message}"),
            Ok(_) => panic!("{} bytes read", document.len()),
        };
        // A text of 5,000 bytes under 60 aliases: 305,000 bytes of text
        // where 262,144 may be read, and as much built where 10 MiB may be.
        let text = "x".repeat(5_000);
        let aliases = |name: &str, count: usize| vec![name; count].join(",");
        refused(format!("[&t {text},{}]", aliases("*t", 60)));
        // A map of a thousand keys of a hundred bytes under two aliases:
        // 300,000 bytes of keys, and 684,000 built.
        let mut keys = String::new();
        for key in 0..1000 {
            let _ = write!(keys, "{key:0>100}: 1,");
        }
        refused(format!("[&m {{{keys}}},{}]", aliases("*m", 2)));

        let escapes = "\\L".repeat(1000);
        let read = read_yaml(format!("\"{escapes}\"").as_bytes()).expect("it reads");
        assert_eq!(read.as_str().map(str::len), Some(3000));
    }

    /// The bounds are the size bound's, whatever a file's own size, so a
    /// small file that shares one profile among its names by alias runs as
    /// it does written out: four names on a profile of three filters, three
    /// scorers and a picker (277 bytes), and 143 names with a binding each
    /// on a profile of five filters (11,523 bytes).
    #[test]
    fn a_small_file_that_shares_a_profile_by_alias_runs_as_written_out() {
        const FILTERS: [&str; 5] = [
            "healthy",
            "not-overloaded",
            "has-capacity",
            "has-room",
            "not-tried",
        ];
        let file = |filters: usize, names: usize, bound: bool, aliased: bool| {
            let filters: Vec<&str> = FILTERS.iter().take(filters).copied().collect();
            let body = format!(
                "\n    filter: [{}]\n    score: [least-load, least-requests, fewest-sessions]\n    pick: best\n",
                filters.join(", ")
            );
            let mut text = format!("apiVersion: smg.policy/v1\nprofiles:\n  default: &p{body}");
            for name in 0..names {
                if aliased {
                    let _ = writeln!(text, "  tenant-{name:04}: *p");
                } else {
                    let _ = write!(text, "  tenant-{name:04}:{body}");
                }
            }
            text.push_str("bindings:\n  - profile: default\n");
            if bound {
                for name in 0..names {
                    let _ = writeln!(
                        text,
                        "  - match: {{model: \"tenant-{name:04}/*\"}}\n    profile: tenant-{name:04}"
                    );
                }
            }
            text
        };
        for (filters, names, bound, bytes) in [(3, 4, false, 277), (5, 143, true, 11_523)] {
            let shared = file(filters, names, bound, true);
            assert_eq!(shared.len(), bytes);
            for (document, written) in [
                (shared, "by alias"),
                (file(filters, names, bound, false), "written out"),
            ] {
                let (_store, generations) = generations();
                assert_eq!(
                    yaml(&generations, document),
                    Ok(Applied::Generation(1)),
                    "{names} names {written}"
                );
            }
        }
    }

    /// A field the reader refuses is named by its path, and text that looks
    /// like a number is no text until it is quoted, as in JSON.
    #[test]
    fn a_field_the_reader_refuses_is_named_by_its_path() {
        let (_store, generations) = generations();
        let unparsed = |document: String| match yaml(&generations, document) {
            Err(SpecRejected::Unparsed {
                format: "yaml",
                path,
                message,
            }) => (path, message),
            other => panic!("applied as {other:?}"),
        };
        let (path, message) =
            unparsed(LEAST_LOAD.replace("    pick: best\n", "    colour: blue\n"));
        assert_eq!(path, "profiles.default.colour");
        assert!(message.contains("colour"), "{message}");
        let (path, _) = unparsed(LEAST_LOAD.replace("pick: best", "pick: 1"));
        assert_eq!(path, "profiles.default.pick");
        assert_eq!(
            yaml(
                &generations,
                LEAST_LOAD.replace("[healthy]", "[\"healthy\"]")
            ),
            Ok(Applied::Generation(1))
        );
    }

    /// A reader's refusal names the field the typed read refused, in a YAML
    /// document as in a JSON one, and no field where the refusal is about
    /// the document as a whole.
    #[test]
    fn a_readers_refusal_names_the_field_it_refused() {
        let (_store, generations) = generations();
        let path = |document: &str, format| {
            generations
                .dry_run("", document.as_bytes(), format)
                .expect("no guard")
                .would_run
                .expect_err("refused")
                .field_path()
        };
        let yaml = LEAST_LOAD.replace("pick: best", "pick: [best]");
        assert_eq!(path(&yaml, Format::Yaml), "profiles.default.pick");
        let json = r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":["best"]}}}"#;
        assert_eq!(path(json, Format::Json), "profiles.default.pick");
        assert_eq!(path("profiles: [", Format::Yaml), "");
        assert_eq!(path("7", Format::Json), "");
        // No values, and values that read (a key written twice keeps its
        // last): the refusal stays the typed read's, about the document.
        assert_eq!(path("{", Format::Json), "");
        let twice = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{"pick":"random"},"p":{"pick":"random"}}}"#;
        assert_eq!(path(twice, Format::Json), "");
        // A key the root does not know, and one it wants, by their own
        // names: the document that wants a key has no name of its own.
        let unknown = format!("{LEAST_LOAD}colour: blue\n");
        assert_eq!(path(&unknown, Format::Yaml), "colour");
        assert_eq!(path("profiles: {}\n", Format::Yaml), "apiVersion");
        let unknown = r#"{"apiVersion":"smg.policy/v1","profiles":{},"colour":1}"#;
        assert_eq!(path(unknown, Format::Json), "colour");
        let missing = r#"{"apiVersion":"smg.policy/v1"}"#;
        assert_eq!(path(missing, Format::Json), "profiles");
        // A key a nested map wants is joined to the map's own path.
        let nested = r#"{"apiVersion":"smg.policy/v1","profiles":{"p":{}}}"#;
        assert_eq!(path(nested, Format::Json), "profiles.p.pick");
    }

    /// A need the contract's row has no flag for writes no row: the engine
    /// reads the affinity column of a host's own stage in every phase, and
    /// the column is the controller's to hold.
    #[test]
    fn a_need_with_no_flag_in_the_contract_writes_no_row() {
        #[derive(Debug)]
        struct Held;

        impl Filter for Held {
            fn name(&self) -> &'static str {
                "held"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> crate::Needs {
                crate::Needs {
                    affinity: true,
                    ..crate::Needs::default()
                }
            }
        }

        let mut stages = Catalog::builtin();
        stages
            .filter("held", || Box::new(Held))
            .expect("a free name");
        let document = LEAST_LOAD.replace("[healthy]", "[held]");
        let spec: PolicySpec = serde_yaml::from_str(&document).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        let rows = needs_summary(&engine).models;
        // The profile's scorer reads the loads: its rows stand, and say
        // nothing of the column.
        assert_eq!(rows.len(), 4, "{rows:?}");
        let quiet = document
            .replace("    score: [least-load]\n", "")
            .replace("best", "random");
        let spec: PolicySpec = serde_yaml::from_str(&quiet).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        assert_eq!(engine.needs_rows().len(), 4, "the column in every phase");
        assert_eq!(needs_summary(&engine), proto::NeedsSummary::default());
    }

    /// The summary tells each row under its own role and each need under
    /// its own flag: a pair's prefill row and its decode row part, a row of
    /// media alone is told, and a sessions rule's key is a row of its
    /// binding in every phase it covers.
    #[test]
    fn the_summary_tells_each_role_and_each_flag_apart() {
        /// Keeps every worker and reads what it is built to read.
        #[derive(Debug)]
        struct Reads(&'static str, crate::Needs);

        impl Filter for Reads {
            fn name(&self) -> &'static str {
                self.0
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn needs(&self) -> crate::Needs {
                self.1
            }
        }

        let tokens = crate::Needs {
            token_hashes: true,
            ..crate::Needs::default()
        };
        let media = crate::Needs {
            media: true,
            ..crate::Needs::default()
        };
        let mut stages = Catalog::builtin();
        stages
            .filter("tokens", move || Box::new(Reads("tokens", tokens)))
            .expect("a free name")
            .filter("media", move || Box::new(Reads("media", media)))
            .expect("a free name");
        let document = r#"{
          "apiVersion": "smg.policy/v1",
          "view": { "headers": ["x-session-id"] },
          "profiles": {
            "prefill": { "filter": ["tokens"], "pick": "random" },
            "decode": { "filter": ["media"], "pick": "random" },
            "quiet": { "pick": "random" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "chat/*" }, "profile": "quiet", "sessions": { "sticky_header": "x-session-id", "ttl_s": 600 } },
            { "match": { "model": "*" }, "profile": "quiet" }
          ]
        }"#;
        let spec: PolicySpec = serde_json::from_str(document).expect("a document");
        let engine = spec.validate_in(&stages, &[]).expect("validates");
        let rows = needs_summary(&engine).models;
        let row = |model: &str, role: proto::Role| {
            rows.iter()
                .find(|row| row.model == model && row.role == i32::from(role))
                .cloned()
        };
        let told = |model: &str, role: proto::Role| proto::ModelNeeds {
            model: model.to_owned(),
            role: role.into(),
            ..proto::ModelNeeds::default()
        };
        assert_eq!(
            row("pd/*", proto::Role::Prefill),
            Some(proto::ModelNeeds {
                token_hashes: true,
                ..told("pd/*", proto::Role::Prefill)
            })
        );
        assert_eq!(
            row("pd/*", proto::Role::Decode),
            Some(proto::ModelNeeds {
                media: true,
                ..told("pd/*", proto::Role::Decode)
            })
        );
        // The sessions rule reads its key under the view's headers, in
        // every phase its binding covers.
        for role in [
            proto::Role::Single,
            proto::Role::Prefill,
            proto::Role::Decode,
            proto::Role::Encode,
        ] {
            assert_eq!(
                row("chat/*", role),
                Some(proto::ModelNeeds {
                    headers: true,
                    ..told("chat/*", role)
                })
            );
        }
        assert_eq!(rows.len(), 6, "{rows:?}");
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
        let counts = |applied| Counts {
            applied,
            rejected: 0,
        };

        let first = dry(LEAST_LOAD.as_bytes());
        assert_eq!(first.standing, (0, None));
        // The feed before and after: nothing stands, and the document's one
        // profile reads the loads in every role.
        let feed = first.needs.expect("the needs diff");
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
        assert_eq!(generations.counts(), counts(0));

        assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
        let again = dry(LEAST_LOAD.as_bytes());
        assert_eq!(again.standing, (1, Some(digest)));
        assert_eq!(again.would_run, Ok((1, digest)), "the standing document");
        let feed = again.needs.expect("the needs diff");
        assert_eq!(feed.before, feed.after, "the feed stands");
        assert_eq!(number(dry(RANDOM.as_bytes())), Ok(2));
        assert_eq!(number(dry(&with_line(RANDOM, 9))), Ok(9));
        assert_eq!(number(dry(&with_line(LEAST_LOAD, 9))), Ok(9));

        assert_eq!(yaml(&generations, RANDOM), Ok(Applied::Generation(2)));
        assert_eq!(number(dry(LEAST_LOAD.as_bytes())), Ok(3), "an apply landed");
        assert_eq!(standing(&store).0, 2);
        assert_eq!(generations.counts(), counts(2));
    }

    /// Scores every worker alike, by state the lifecycle's calls keep.
    #[derive(Debug)]
    struct Kept;

    impl crate::Lifecycle for Kept {
        fn needs_lifecycle(&self) -> bool {
            true
        }
    }

    impl crate::Scorer for Kept {
        fn name(&self) -> &'static str {
            "kept"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> crate::TieBand {
            out.fill(1.0);
            crate::TieBand::NORMALIZED
        }
    }

    /// A dry run answers a stage in need of the lifecycle's calls as an
    /// apply does: refused by the stage's field, as not in this build.
    #[test]
    fn a_dry_run_refuses_a_stage_in_need_of_the_lifecycle_as_an_apply_does() {
        let mut stages = Catalog::builtin();
        stages
            .scorer("kept", || Box::new(Kept))
            .expect("a free name");
        let store = Arc::new(MemoryStore::new());
        let generations = Generations::new(store, Arc::new(stages.without_lifecycle()));
        let document =
            b"apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score: [kept]\n    pick: best\n"
                .to_vec();
        let refused = generations
            .dry_run("", &document, Format::Yaml)
            .expect("no guard")
            .would_run
            .expect_err("refused");
        assert_eq!(
            refused.field_path(),
            "profiles.default.score[0].kept",
            "{refused}"
        );
        assert!(
            refused.to_string().contains("is not in this build"),
            "{refused}"
        );
        assert_eq!(yaml(&generations, document), Err(refused));
    }

    /// A dry run refuses what an apply would, by the same name: a changed
    /// document under the standing line, a line under the standing one (the
    /// refusal of the store's entry, in its words), a document the validator
    /// refuses, one that is no document and one over the bound. Each names
    /// its field, and the last two none: they are about the document.
    #[test]
    fn a_dry_run_refuses_what_an_apply_would_by_the_same_name() {
        let (_store, generations) = generations();
        assert_eq!(
            yaml(&generations, with_line(LEAST_LOAD, 5)),
            Ok(Applied::Generation(5))
        );
        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 1;
        let rows = [
            (with_line(RANDOM, 5), "generation"),
            (with_line(RANDOM, 3), "generation"),
            (
                LEAST_LOAD.replace("[healthy]", "[nowhere]").into_bytes(),
                "profiles.default.filter",
            ),
            (b"profiles: [".to_vec(), ""),
            (vec![b'#'; over], ""),
        ];
        for (document, path) in rows {
            let refused = generations
                .dry_run("", &document, Format::Yaml)
                .expect("no guard")
                .would_run
                .expect_err("refused");
            assert_eq!(refused.field_path(), path, "{refused}");
            assert_eq!(yaml(&generations, document), Err(refused));
        }
        let lower = generations
            .dry_run("", &with_line(RANDOM, 3), Format::Yaml)
            .expect("no guard");
        let stale = StaleGeneration {
            standing: 5,
            offered: 3,
        };
        assert_eq!(lower.would_run, Err(SpecRejected::Stale(stale)));
        assert!(lower.needs.is_none(), "a refused document carries no diff");
        assert_eq!(
            generations.counts(),
            Counts {
                applied: 1,
                rejected: 0
            }
        );
    }

    /// The bound fails fast: a document one byte over it is refused by its
    /// size, before any of it is parsed. The bound is the largest document:
    /// one of exactly that size runs.
    #[test]
    fn a_document_over_the_bound_is_refused_by_its_size() {
        let (store, generations) = generations();
        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 1;
        let refused = generations.apply(vec![b'{'; over], Format::Json);
        assert_eq!(refused, Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 1)));
        assert_eq!(
            refused.expect_err("refused").to_string(),
            "the document is 131073 bytes; at most 131072",
            "128 KiB and a byte"
        );
        assert_eq!(standing(&store), (0, None));
        assert!(!generations.bound());

        let bound = over - 1;
        assert_eq!(
            generations.apply(padded(LEAST_LOAD, bound), Format::Yaml),
            Ok(Applied::Generation(1))
        );
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
            Ok(Applied::Generation(1))
        );
        let (_, Some(digest)) = standing(&store) else {
            panic!("an applied generation has a digest");
        };
        for wanted in [0, 1] {
            let answer = generations.document(wanted).expect("the standing document");
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

    /// The mint and the entry are one critical section: documents applied
    /// from several threads at once each take a number of their own, in a
    /// row, and none is refused for a number another one took.
    #[test]
    fn applies_from_several_threads_each_mint_a_number_of_their_own() {
        const WRITERS: u64 = 8;
        let (store, generations) = generations();
        let mut minted: Vec<u64> = thread::scope(|scope| {
            let writers: Vec<_> = (0..WRITERS)
                .map(|writer| {
                    let generations = &generations;
                    scope.spawn(move || {
                        let document = LEAST_LOAD.replace("default", &format!("profile-{writer}"));
                        match yaml(generations, document) {
                            Ok(Applied::Generation(generation)) => generation,
                            other => panic!("writer {writer}: {other:?}"),
                        }
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| writer.join().expect("a writer"))
                .collect()
        });
        minted.sort_unstable();
        assert_eq!(minted, (1..=WRITERS).collect::<Vec<_>>());
        assert_eq!(standing(&store).0, WRITERS);
    }

    /// A document of about `bytes` bytes: one profile, and bindings of one
    /// exact model id each until the size is reached.
    fn document_of(bytes: usize) -> (Vec<u8>, u32) {
        let mut text = format!("{LEAST_LOAD}bindings:\n");
        let mut bindings = 0;
        while text.len() < bytes {
            let _ = write!(
                text,
                "  - match: {{ model: \"tenant-{bindings:06}/model\" }}\n    profile: default\n"
            );
            bindings += 1;
        }
        (text.into_bytes(), bindings)
    }

    /// A JSON document of `bytes` at the most whose last binding writes its
    /// profile as a list: the typed read refuses it there, at its end.
    fn refused_json_of(bytes: usize) -> Vec<u8> {
        let mut text = String::from(
            r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}},"bindings":["#,
        );
        let last = r#"{"match":{"model":"last"},"profile":["default"]}]}"#;
        let mut bindings = 0;
        while text.len() + 64 + last.len() < bytes {
            let _ = write!(
                text,
                r#"{{"match":{{"model":"tenant-{bindings:06}/model"}},"profile":"default"}},"#
            );
            bindings += 1;
        }
        text.push_str(last);
        text.into_bytes()
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
            assert_eq!(yaml(&generations, LEAST_LOAD), Ok(Applied::Generation(1)));
            let mut micros = Vec::with_capacity(ROUNDS);
            let mut diff = 0;
            for _ in 0..ROUNDS {
                let began = Instant::now();
                let dry = generations
                    .dry_run("", &document, Format::Yaml)
                    .expect("no guard");
                micros.push(began.elapsed().as_micros());
                assert_eq!(dry.would_run.map(|(number, _)| number), Ok(2));
                diff = dry.needs.map_or(0, |needs| needs.encoded_len());
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

    /// The Numbers table's apply rows: one apply (parse, digest, validate,
    /// the store's entry) by the document's size, up to the bound.
    #[test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    fn an_applys_time_by_the_documents_size() {
        const ROUNDS: usize = 15;
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let mut out = std::io::stdout().lock();
        for size in [2 << 10, 64 << 10, bound - 128] {
            let (document, bindings) = document_of(size);
            let mut micros = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let (_store, generations) = generations();
                let document = document.clone();
                let began = Instant::now();
                let applied = yaml(&generations, document);
                micros.push(began.elapsed().as_micros());
                assert_eq!(applied, Ok(Applied::Generation(1)));
            }
            micros.sort_unstable();
            writeln!(
                out,
                "APPLY bytes={} bindings={bindings} rounds={ROUNDS} median_us={} worst_us={}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
        // A file its aliases multiply past the bound: the time to the
        // refusal, by the file's size.
        for (text, aliases) in [(2_048, 512), (8_192, 2_048), (65_536, 16_000)] {
            let document = long_text_under_aliases(text, aliases);
            let mut micros = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let (_store, generations) = generations();
                let document = document.clone();
                let began = Instant::now();
                let refused = yaml(&generations, document);
                micros.push(began.elapsed().as_micros());
                assert!(
                    matches!(refused, Err(SpecRejected::Unparsed { .. })),
                    "{refused:?}"
                );
            }
            micros.sort_unstable();
            writeln!(
                out,
                "REFUSED bytes={} text={text} aliases={aliases} rounds={ROUNDS} median_us={} worst_us={}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
        // A JSON document the typed read refuses at its last binding is
        // read three times: by the typed read, again as values, and by the
        // typed read of those. The time to that refusal, by its size.
        for size in [2 << 10, 64 << 10, bound - 128] {
            let document = refused_json_of(size);
            let mut micros = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let (_store, generations) = generations();
                let document = document.clone();
                let began = Instant::now();
                let refused = generations.apply(document, Format::Json);
                micros.push(began.elapsed().as_micros());
                assert!(
                    matches!(refused, Err(SpecRejected::Unparsed { format: "json", .. })),
                    "{refused:?}"
                );
            }
            micros.sort_unstable();
            writeln!(
                out,
                "REFUSED_JSON bytes={} rounds={ROUNDS} median_us={} worst_us={}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
        // The heaviest file the search finds: the time of its pass, which
        // ends at the validator's refusal with the document built.
        #[cfg(target_os = "linux")]
        {
            let (document, names) = search::heaviest_file();
            let mut micros = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let (_store, generations) = generations();
                let document = document.clone();
                let began = Instant::now();
                let refused = yaml(&generations, document);
                micros.push(began.elapsed().as_micros());
                assert!(
                    matches!(refused, Err(SpecRejected::Invalid(_))),
                    "{refused:?}"
                );
            }
            micros.sort_unstable();
            writeln!(
                out,
                "HEAVIEST bytes={} names={names} rounds={ROUNDS} median_us={} worst_us={}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
    }

    /// The search behind the read's bounds, by run where a process is told
    /// its resident peak.
    #[cfg(target_os = "linux")]
    mod search {
        use std::{env, fs, io, process::Command};

        use super::*;

        /// What an apply of a YAML file may hold beside the file, for each
        /// byte of the size bound: the most the search finds, with the room
        /// its spread wants. One file's resident peak moves by a twentieth
        /// from a run and a build to the next (367 to 406 by run). Every
        /// file of the search is held to it.
        const HELD_A_BYTE: usize = 448;

        /// The budget of one apply, which sets the size bound: what a
        /// document of the size bound may hold beside its bytes.
        const BUDGET: usize = 64 << 20;

        /// The measurement by its name, and the variable that hands one
        /// file of it to a process of its own.
        const NAME: &str = "controller::generations::tests::search::what_a_file_at_the_bound_holds";
        const ONE: &str = "SMG_POLICY_SPEC_HELD_FILE";

        /// `path`'s file applied in a process of its own, through the
        /// measurement's one-file entry: whether that process held it to
        /// [`HELD_A_BYTE`], and what it wrote.
        fn alone(path: &Path) -> (bool, String) {
            let this = env::current_exe().expect("this executable");
            let child = Command::new(this)
                .args([
                    "--exact",
                    NAME,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(ONE, path)
                .output()
                .expect("the file's own process");
            let mut said = String::from_utf8_lossy(&child.stdout).into_owned();
            said.push_str(&String::from_utf8_lossy(&child.stderr));
            (child.status.success(), said)
        }

        /// A profile of a thousand scorers of a letter, under an anchor.
        fn scorers() -> String {
            let letters = "a,".repeat(1000);
            format!("&p {{filter: [healthy], score: [{letters}], pick: best}}")
        }

        /// A document's head: one profile, and `count` names more that each
        /// write `alias`.
        fn named(anchor: &str, alias: &str, count: usize) -> String {
            let mut head = format!("apiVersion: smg.policy/v1\nprofiles:\n  default: {anchor}\n");
            for name in 0..count {
                let _ = writeln!(head, "  t{name:04}: {alias}");
            }
            head
        }

        /// `head`, then bindings of nothing to `len` bytes.
        fn beside_bindings(head: String, len: usize) -> Vec<u8> {
            filled(&format!("{head}bindings: ["), "{},", "]\n", len)
        }

        /// One file of the search.
        struct Shape {
            name: String,
            /// Whether an alias is in it: a file that writes its values out
            /// is refused by no bound of the read.
            aliases: bool,
            file: Vec<u8>,
        }

        /// `text` with blanks to `len` bytes, a line's end the last.
        fn blanked(text: String, len: usize) -> Vec<u8> {
            let mut file = text.into_bytes();
            assert!(file.len() < len, "the text is past the size");
            file.resize(len - 1, b' ');
            file.push(b'\n');
            file
        }

        /// `open`, then `item` until the next would pass `len` bytes, then
        /// `close`, then blanks to `len` bytes.
        fn filled(open: &str, item: &str, close: &str, len: usize) -> Vec<u8> {
            let mut text = String::with_capacity(len);
            text.push_str(open);
            while text.len() + item.len() + close.len() < len {
                text.push_str(item);
            }
            text.push_str(close);
            blanked(text, len)
        }

        /// A flow map of `count` keys with no values, each key once.
        fn keys(count: usize) -> String {
            let mut text = String::from("{");
            for key in 0..count {
                let _ = write!(text, "{key:x},");
            }
            text.push('}');
            text
        }

        /// Numbers from a seed, the same from the same seed.
        struct Draw(u64);

        impl Draw {
            fn below(&mut self, bound: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % bound
            }
        }

        /// A value of the draw's choosing in flow style, at most `depth`
        /// lists and maps deep.
        fn drawn(draw: &mut Draw, depth: u32, out: &mut String) {
            match draw.below(if depth == 0 { 5 } else { 8 }) {
                0 => out.push('~'),
                1 => out.push_str("true"),
                2 => {
                    let _ = write!(out, "{}", draw.below(100_000));
                }
                3 => out.push_str("[]"),
                4 => {
                    out.push('"');
                    for _ in 0..draw.below(32) {
                        out.push('x');
                    }
                    out.push('"');
                }
                5 | 6 => {
                    out.push('[');
                    for _ in 0..draw.below(6) {
                        drawn(draw, depth - 1, out);
                        out.push(',');
                    }
                    out.push(']');
                }
                _ => {
                    out.push('{');
                    for key in 0..draw.below(6) {
                        let _ = write!(out, "k{key}: ");
                        drawn(draw, depth - 1, out);
                        out.push(',');
                    }
                    out.push('}');
                }
            }
        }

        /// Values of a draw's choosing: eight under anchors, `aliases`
        /// aliases of them, then values written out to `len` bytes.
        fn mixed(seed: u64, aliases: usize, len: usize) -> Vec<u8> {
            let mut draw = Draw(seed);
            let mut text = String::with_capacity(len);
            text.push('[');
            for anchor in 0..8 {
                let _ = write!(text, "&a{anchor} ");
                drawn(&mut draw, 3, &mut text);
                text.push(',');
            }
            for _ in 0..aliases {
                let _ = write!(text, "*a{},", draw.below(8));
            }
            let mut next = String::new();
            loop {
                next.clear();
                drawn(&mut draw, 3, &mut next);
                if text.len() + next.len() + 2 >= len {
                    break;
                }
                text.push_str(&next);
                text.push(',');
            }
            text.push(']');
            blanked(text, len)
        }

        /// The files that write their values out.
        fn written_out(len: usize) -> Vec<Shape> {
            let nest = |depth: usize| format!("{}{},", "[".repeat(depth), "]".repeat(depth));
            let line = |opens: &str| format!("{}a\n", opens.repeat(100));
            let maps = format!("{}1{},", "{a: ".repeat(100), "}".repeat(100));
            let runs = document_of(len - 256).0;
            let runs = std::str::from_utf8(&runs).expect("text");
            let files = [
                ("texts of a byte", filled("[", "a,", "]", len)),
                ("numbers", filled("[", "1,", "]", len)),
                ("nulls", filled("[", "~,", "]", len)),
                ("nulls, a line each", filled("", "-\n", "", len)),
                ("empty lists", filled("[", "[],", "]", len)),
                ("empty maps", filled("[", "{},", "]", len)),
                ("maps of one key", filled("[", "{a},", "]", len)),
                (
                    "nests of a hundred lists",
                    filled("[", &nest(100), "]", len),
                ),
                (
                    "nests past the reader's depth",
                    filled("[", &nest(200), "]", len),
                ),
                ("a hundred lists a line", filled("", &line("- "), "", len)),
                (
                    "a hundred keys of a key a line",
                    filled("", &line("? "), "", len),
                ),
                ("keys of nothing", filled("[", "? ,", "]", len)),
                (
                    "a hundred keys of nothing a line",
                    filled("", &line(": "), "", len),
                ),
                (
                    "lists and keys of a key by turns",
                    filled("", &line("- ? "), "", len),
                ),
                (
                    "a key written twice, to the size",
                    filled("{", "a,", "}", len),
                ),
                ("maps a hundred deep", filled("[", &maps, "]", len)),
                ("keys, each once", blanked(keys(len / 7), len)),
                ("one text", filled("\"", "x", "\"", len)),
                (
                    "texts of a thousand bytes",
                    filled("[", &format!("{},", "x".repeat(1000)), "]", len),
                ),
                (
                    "escapes of two bytes for three",
                    filled("\"", "\\L", "\"", len),
                ),
                ("values by draw 1", mixed(1, 0, len)),
                ("values by draw 2", mixed(2, 0, len)),
                ("a document that runs", padded(runs, len)),
                (
                    "bindings of nothing",
                    filled(&format!("{LEAST_LOAD}bindings: ["), "{},", "]\n", len),
                ),
            ];
            files
                .into_iter()
                .map(|(name, file)| Shape {
                    name: name.to_owned(),
                    aliases: false,
                    file,
                })
                .collect()
        }

        /// A count of `0..=most` whose file the read's bounds admit and
        /// whose next they refuse, found by halving; `most` where they admit
        /// that one.
        fn heaviest(most: usize, file: &dyn Fn(usize) -> Vec<u8>) -> usize {
            let admitted = |count: usize| read_yaml(&file(count)).is_ok();
            assert!(admitted(0), "the bounds admit a family's least file");
            if admitted(most) {
                return most;
            }
            let (mut holds, mut over) = (0, most);
            while over - holds > 1 {
                let middle = holds + (over - holds) / 2;
                if admitted(middle) {
                    holds = middle;
                } else {
                    over = middle;
                }
            }
            holds
        }

        /// An anchor's value of a count of a thing.
        type Anchor = fn(usize) -> String;

        /// What an alias is tried on, each by its name.
        const ANCHORS: [(&str, Anchor); 7] = [
            ("a text of N bytes", |count| {
                format!("\"{}\"", "x".repeat(count))
            }),
            ("N numbers", |count| format!("[{}]", "1,".repeat(count))),
            ("N texts of a byte", |count| {
                format!("[{}]", "a,".repeat(count))
            }),
            ("N empty lists", |count| {
                format!("[{}]", "[],".repeat(count))
            }),
            ("N maps of one key", |count| {
                format!("[{}]", "{a},".repeat(count))
            }),
            ("a nest of N lists and one", |count| {
                format!("{}{}", "[".repeat(count + 1), "]".repeat(count + 1))
            }),
            ("a map of N keys", keys),
        ];

        /// The files an alias multiplies: each family at the most the
        /// bounds admit, and one past.
        fn aliased(len: usize) -> Vec<Shape> {
            let mut shapes = Vec::new();
            let mut both = |name: &str, most: usize, file: &dyn Fn(usize) -> Vec<u8>| {
                let count = heaviest(most, file);
                for (count, said) in [
                    (count, "the most the bounds admit"),
                    (count + 1, "one more"),
                ] {
                    shapes.push(Shape {
                        name: format!("{name}, N = {count}: {said}"),
                        aliases: true,
                        file: file(count),
                    });
                }
            };
            for (name, anchor) in ANCHORS {
                both(&format!("aliases of {name}"), 4096, &|count| {
                    filled(&format!("[&a {},", anchor(count)), "*a,", "]", len)
                });
            }
            let numbers = "1,".repeat(1000);
            both(
                "N aliases of a thousand numbers, the rest a comment",
                (len - 4096) / 3,
                &|count| padded(&format!("[&a [{numbers}],{}]\n", "*a,".repeat(count)), len),
            );
            both(
                "N aliases of a thousand numbers in a file of no more bytes",
                100_000,
                &|count| format!("[&a [{numbers}],{}]\n", "*a,".repeat(count)).into_bytes(),
            );
            let nests = format!("{}{},", "[".repeat(100), "]".repeat(100));
            for (name, item) in [("texts of a byte", "a,"), ("maps of one key", "{a},")] {
                let head = format!("[&a [{}],", item.repeat(100));
                for (rest, tail) in [("nests", nests.as_str()), ("maps of one key", "{a},")] {
                    both(
                        &format!("N aliases of a hundred {name}, the rest {rest}"),
                        len / 4,
                        &|count| filled(&format!("{head}{}", "*a,".repeat(count)), tail, "]", len),
                    );
                }
            }
            both("a header of N bytes under aliases", 4096, &|count| {
                let head = format!(
                    "{LEAST_LOAD}view:\n  headers: [&h \"{}\"",
                    "x".repeat(count)
                );
                filled(&head, ",*h", "]\n", len)
            });
            // The heaviest typed list written out, bindings of nothing,
            // beside what an alias multiplies: the bindings take the file's
            // bytes and a third of what the read may build, and the aliases
            // take the rest in a few thousand bytes.
            let letters = "a,".repeat(1000);
            let beside_bindings = |head: String| beside_bindings(head, len);
            for (name, anchor, alias) in [
                (
                    "names on one profile of a thousand scorers of a letter",
                    scorers(),
                    "*p",
                ),
                (
                    "names on one profile of a thousand filters of a letter",
                    format!("&p {{filter: [{letters}], pick: best}}"),
                    "*p",
                ),
                (
                    "profiles on one list of a thousand scorers of a letter",
                    format!("{{score: &p [{letters}], pick: best}}"),
                    "{score: *p, pick: best}",
                ),
            ] {
                let named = |count: usize| named(&anchor, alias, count);
                both(
                    &format!("N {name}, the rest bindings of nothing"),
                    (len - 8192) / 40,
                    &|count| beside_bindings(named(count)),
                );
                // What a small file's aliases build alone.
                both(
                    &format!("N {name}, in a file of no more bytes"),
                    (len - 8192) / 40,
                    &|count| named(count).into_bytes(),
                );
            }
            both(
                "N aliases of a header of a thousand bytes, the rest bindings of nothing",
                4096,
                &|count| {
                    beside_bindings(format!(
                        "{LEAST_LOAD}view:\n  headers: [&h \"{}\"{}]\n",
                        "x".repeat(1000),
                        ",*h".repeat(count)
                    ))
                },
            );
            // A document that runs: one binding that names a profile, under
            // aliases to the size, each a binding the generation keeps.
            both(
                "N aliases of a binding that names a profile",
                (len - 4096) / 3,
                &|count| {
                    let bindings = ",*b".repeat(count);
                    padded(
                        &format!("{LEAST_LOAD}bindings: [&b {{profile: default}}{bindings}]\n"),
                        len,
                    )
                },
            );
            for seed in 3..11 {
                both(
                    &format!("values by draw {seed} with N aliases"),
                    (len - 8192) / 4,
                    &|count| mixed(seed, count, len),
                );
            }
            // The values an alias builds, then nodes the read never reaches:
            // the parser lists a whole file before the first value is read.
            let texts = format!("- &a [{}]\n", "a,".repeat(100));
            let count = heaviest((len - 4096) / 5, &|count| {
                padded(&format!("{texts}{}", "- *a\n".repeat(count)), len)
            });
            let keys = format!("- {}a\n", "? ".repeat(100));
            for (name, between) in [
                ("keys of a key", ""),
                ("a second document of keys of a key", "---\n"),
            ] {
                let head = format!("{texts}{}{between}", "- *a\n".repeat(count));
                shapes.push(Shape {
                    name: format!("{count} aliases of a hundred texts of a byte, then {name}"),
                    aliases: true,
                    file: filled(&head, &keys, "", len),
                });
            }
            let mut doubling = String::from("[&a0 [x,x],");
            for level in 1..40 {
                let _ = write!(doubling, "&a{level} [*a{0},*a{0}],", level - 1);
            }
            doubling.push_str("]\n");
            shapes.push(Shape {
                name: "aliases of aliases, forty deep".to_owned(),
                aliases: true,
                file: padded(&doubling, len),
            });
            shapes
        }

        /// The process's resident set, or its peak: the line of that name
        /// in the process's status, in bytes.
        fn resident(line: &str) -> usize {
            let status = fs::read_to_string("/proc/self/status").expect("the process's status");
            let kib = status
                .lines()
                .find_map(|found| found.strip_prefix(line))
                .and_then(|rest| rest.trim().strip_suffix("kB"))
                .and_then(|number| number.trim().parse::<usize>().ok())
                .expect("the line's number");
            kib * 1024
        }

        /// One file applied in this process: what the apply held at its
        /// peak, beside what the process held before it.
        fn hold(path: &Path) {
            let file = fs::read(path).expect("the file");
            let bytes = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
            let (_store, generations) = generations();
            // The peak counts from here where the kernel takes the word;
            // where it does not, the peak read is no lower.
            let _ = fs::write("/proc/self/clear_refs", "5");
            let before = resident("VmRSS:");
            let began = Instant::now();
            let applied = yaml(&generations, file);
            let micros = began.elapsed().as_micros();
            let held = resident("VmHWM:").saturating_sub(before);
            let outcome = match applied {
                Ok(applied) => format!("{applied:?}"),
                Err(SpecRejected::Unparsed { message, .. }) => message,
                Err(other) => format!("{other:?}"),
            };
            let outcome: String = outcome
                .chars()
                .take(96)
                .map(|letter| if letter == '\n' { ' ' } else { letter })
                .collect();
            writeln!(
                io::stdout(),
                "HOLDS a_byte={} held={held} micros={micros} outcome={outcome}",
                held / bytes
            )
            .expect("stdout");
            assert!(
                held <= HELD_A_BYTE * bytes,
                "{held} bytes held where the size bound is {bytes}"
            );
        }

        /// The heaviest file the search finds, with its count of names:
        /// bindings of nothing to the size bound, beside one profile of a
        /// thousand scorers under the most names the bounds admit.
        pub(super) fn heaviest_file() -> (Vec<u8>, usize) {
            let len = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
            let profile = scorers();
            let file = |count: usize| beside_bindings(named(&profile, "*p", count), len);
            let count = heaviest((len - 8192) / 40, &file);
            (file(count), count)
        }

        /// The budget holds in every run: the heaviest file the search finds
        /// (bindings of nothing to the size bound, beside one profile of a
        /// thousand scorers under the most names the bounds admit), applied
        /// in a process of its own, holds no more than one apply's budget. A
        /// larger size bound, or a type that grows past the spread between
        /// runs, is told here. A copy of the document that comes back reads
        /// at this case's edge, and the search tells it.
        #[test]
        fn the_heaviest_file_found_holds_no_more_than_an_applys_budget() {
            let (file, count) = heaviest_file();
            let dir = tempfile::tempdir().expect("a directory");
            let path = dir.path().join("spec.yaml");
            fs::write(&path, file).expect("the file");
            let (_, said) = alone(&path);
            // The validator's refusal: the document was built and digested.
            assert!(said.contains("UnknownScorer"), "{count} names: {said}");
            let held = said
                .split_once(" held=")
                .and_then(|(_, rest)| rest.split(' ').next())
                .and_then(|bytes| bytes.parse::<usize>().ok())
                .expect("the bytes the apply held");
            assert!(
                held <= BUDGET,
                "{count} names: {held} bytes held, past the budget of {BUDGET}"
            );
        }

        /// The size bound is the budget's: at the most the search finds for
        /// each byte, a document of the size bound holds no more than one
        /// apply's budget.
        #[test]
        fn a_document_of_the_size_bound_holds_no_more_than_an_applys_budget() {
            let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
            assert!(
                HELD_A_BYTE * bound <= BUDGET,
                "{HELD_A_BYTE} bytes for each of {bound} is past {BUDGET}"
            );
        }

        /// The search: files in the shapes that build the most (values
        /// written out, nests, values of no text, long texts, a typed
        /// document of nothing, aliases at the most the bounds admit and one
        /// past, that typed document beside what an alias shares, a document
        /// that runs under aliases, values by draw), of the size bound's
        /// bytes but for the small files their aliases take to the bounds,
        /// each applied in a process of its own, which reads its resident
        /// peak over the apply.
        /// Every file is held to [`HELD_A_BYTE`], and a file that writes its
        /// values out is refused by no bound of the read.
        #[test]
        #[ignore = "a measurement for the Numbers table, run by name"]
        fn what_a_file_at_the_bound_holds() {
            if let Some(path) = env::var_os(ONE) {
                hold(Path::new(&path));
                return;
            }
            let len = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
            let dir = tempfile::tempdir().expect("a directory");
            let path = dir.path().join("spec.yaml");
            let mut out = io::stdout().lock();
            writeln!(out).expect("stdout");
            let mut most = 0;
            for shape in written_out(len).into_iter().chain(aliased(len)) {
                fs::write(&path, &shape.file).expect("the file");
                let (held, said) = alone(&path);
                // The harness writes a test's name before the test's first line.
                let row = said
                    .lines()
                    .find_map(|line| line.split_once("HOLDS ").map(|(_, row)| row))
                    .unwrap_or("nothing");
                writeln!(
                    out,
                    "HOLDS {row} bytes={} shape={}",
                    shape.file.len(),
                    shape.name
                )
                .expect("stdout");
                assert!(held, "{}: {said}", shape.name);
                assert!(
                    shape.aliases || !row.contains("of the size bound"),
                    "{}: no alias is in it and a bound of the read refuses it",
                    shape.name
                );
                let a_byte = row
                    .split(['=', ' '])
                    .nth(1)
                    .and_then(|number| number.parse::<usize>().ok())
                    .expect("the row's number");
                most = most.max(a_byte);
            }
            writeln!(out, "HOLDS_MOST a_byte={most} of {HELD_A_BYTE}").expect("stdout");
        }
    }

    /// A document under a filter no catalog holds: the validator refuses
    /// it.
    const UNKNOWN_FILTER: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [nowhere]\n    pick: random\n";

    /// Binds the directory at `dir` as the door's source, as a start does.
    fn bind(generations: &Generations, dir: &Path) {
        let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
        generations.bind_store(opened).expect("the store binds");
    }

    /// [`bind`] for a second door on a directory another holds, as a file
    /// system whose locks do not hold between two hosts lets one in.
    fn bind_beside(generations: &Generations, dir: &Path) {
        let opened = SpecStore::open_beside(dir.to_owned()).expect("the directory opens");
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

    /// `LEAST_LOAD` with one binding of its own, by `n`: a document apart
    /// from every other `n`'s.
    pub(crate) fn bound_to(n: u64) -> String {
        format!(
            "{LEAST_LOAD}bindings:\n  - match: {{ model: \"model-{n}\" }}\n    profile: default\n"
        )
    }

    fn applied(applied: u64, rejected: u64) -> Counts {
        Counts { applied, rejected }
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
        let needs = first.needs.expect("the diff");
        assert_eq!(needs.before, Some(proto::NeedsSummary::default()));
        let runs = needs_summary(store.world().spec().engine());
        assert_eq!(needs.after, Some(runs));

        // A JSON document is on file under its own form.
        let json = r#"{"apiVersion":"smg.policy/v1","profiles":{"default":{"pick":"random"}}}"#;
        let second = generations
            .enforce("", json.into(), Format::Json)
            .expect("no guard");
        assert_eq!(second.ran.map(|(number, _)| number), Ok(2));
        let both = BTreeSet::from([name_of(1, "yaml"), name_of(2, "json")]);
        assert_eq!(names(dir.path()), both);
        assert_eq!(generations.counts(), applied(2, 0));
    }

    #[test]
    fn the_standing_document_again_is_no_change_under_a_store() {
        let (dir, store, generations) = stored();
        let first = enforce(&generations, LEAST_LOAD).expect("it runs");
        // Again as written, again under a comment, again under the
        // standing line: one document, and nothing more on file.
        let commented = format!("# the same\n{LEAST_LOAD}").into_bytes();
        for again in [LEAST_LOAD.into(), commented, with_line(LEAST_LOAD, 1)] {
            assert_eq!(enforce(&generations, again), Ok(first));
        }
        assert_eq!(standing(&store).0, 1);
        assert_eq!(names(dir.path()).len(), 1);
        assert_eq!(generations.counts(), applied(1, 0));
    }

    /// An enforce on a door no store was bound to persists nowhere: it is
    /// refused by name, counted, and nothing is applied.
    #[test]
    fn an_enforce_with_no_store_bound_is_refused_and_nothing_applied() {
        let (store, generations) = generations();
        assert!(!generations.writable() && !generations.bound());
        let refused = enforce(&generations, LEAST_LOAD).expect_err("no store");
        assert_eq!(
            refused,
            SpecRejected::Unpersisted("no spec store is bound".to_owned())
        );
        assert_eq!(standing(&store), (0, None));
        assert_eq!(generations.counts(), applied(0, 1));
    }

    /// The guard: an enforce names the digest its caller read as standing,
    /// and is refused where another apply landed first.
    #[test]
    fn an_enforce_is_refused_where_another_apply_landed_first() {
        let (dir, store, generations) = stored();
        let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
        let read = first.to_string();
        let second = generations
            .enforce(&read, RANDOM.into(), Format::Yaml)
            .expect("the digest stands");
        let (number, digest) = second.ran.expect("it runs");
        assert_eq!(number, 2);

        // The same guard again: the digest it names stands no longer.
        let late = generations.enforce(&read, bound_to(1).into(), Format::Yaml);
        assert!(
            matches!(late, Err(Unanswered::Guarded(2, Some(stands))) if stands == digest),
            "{late:?}"
        );
        assert_eq!(standing(&store), (2, Some(digest)));
        assert_eq!(names(dir.path()).len(), 2);

        // Before any generation no digest stands, so a guard names none.
        let (_dir, _store, empty) = stored();
        let guarded = empty.enforce(&read, LEAST_LOAD.into(), Format::Yaml);
        assert!(
            matches!(guarded, Err(Unanswered::Guarded(0, None))),
            "{guarded:?}"
        );
    }

    /// An apply call writes the next number: a document's own line is a
    /// claim of that number, and any other is refused by naming the
    /// standing one. A dry run answers the same.
    #[test]
    fn a_line_is_a_claim_of_the_next_number_under_a_store() {
        let (dir, store, generations) = stored();
        let ran = |document| enforce(&generations, document).map(|(number, _)| number);
        assert_eq!(ran(with_line(LEAST_LOAD, 1)), Ok(1));
        assert_eq!(ran(with_line(RANDOM, 2)), Ok(2));
        let stands = standing(&store).1.expect("a digest");

        for line in [1, 4, 9] {
            let refused = enforce(&generations, with_line(LEAST_LOAD, line)).expect_err("a line");
            let another = SpecRejected::NotTheNext {
                offered: line,
                standing: 2,
            };
            assert_eq!(refused, another);
            assert_eq!(refused.field_path(), "generation");
            assert_eq!(
                refused.to_string(),
                format!("generation {line}: an apply writes generation 3, the next after the standing generation 2; write that number, or none")
            );
        }
        // A changed document under the standing line, as under a file.
        let changed = enforce(&generations, with_line(LEAST_LOAD, 2)).expect_err("its line");
        assert!(
            matches!(changed, SpecRejected::ChangedUnderItsLine { generation: 2, standing, .. } if standing == stands),
            "{changed:?}"
        );
        // Nothing of the four is on file, and the running generation stays.
        assert_eq!(names(dir.path()).len(), 2);
        assert_eq!(standing(&store), (2, Some(stands)));
        assert_eq!(generations.counts(), applied(2, 4));

        let dry = |line| {
            generations
                .dry_run("", &with_line(LEAST_LOAD, line), Format::Yaml)
                .expect("no guard")
                .would_run
                .map(|(number, _)| number)
        };
        let another = SpecRejected::NotTheNext {
            offered: 9,
            standing: 2,
        };
        assert_eq!((dry(9), dry(3)), (Err(another), Ok(3)));
    }

    #[test]
    fn a_document_that_does_not_run_is_refused_with_nothing_on_file() {
        let (dir, store, generations) = stored();
        let (_, stands) = enforce(&generations, LEAST_LOAD).expect("it runs");

        let refused = enforce(&generations, UNKNOWN_FILTER).expect_err("no such filter");
        assert!(
            matches!(
                refused,
                SpecRejected::Invalid(SpecError::UnknownFilter { .. })
            ),
            "{refused:?}"
        );
        let unread = enforce(&generations, "apiVersion: [").expect_err("no document");
        assert!(
            matches!(unread, SpecRejected::Unparsed { .. }),
            "{unread:?}"
        );

        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
        assert_eq!(standing(&store), (1, Some(stands)));
        assert_eq!(generations.counts(), applied(1, 2));
    }

    /// A start: the newest generation on file runs under the number its
    /// name states, not under a first number, and the next enforce writes
    /// the one after it.
    #[test]
    fn a_start_runs_the_newest_generation_on_file_under_its_number() {
        let (dir, _store, before) = stored();
        for document in [LEAST_LOAD, RANDOM, LEAST_LOAD] {
            enforce(&before, document).expect("it runs");
        }
        let (_, stood) = before.standing();
        drop(before);

        let (store, after) = generations();
        bind(&after, dir.path());
        assert_eq!(standing(&store), (3, stood));
        let read = after.document(0).expect("the standing document");
        assert_eq!(
            (read.generation, read.document.as_slice()),
            (3, LEAST_LOAD.as_bytes())
        );
        assert_eq!(after.counts(), applied(1, 0));
        assert_eq!(enforce(&after, RANDOM).map(|(number, _)| number), Ok(4));
    }

    /// A document on file runs under its file's number whatever line its
    /// bytes write.
    #[test]
    fn a_generation_on_file_runs_under_its_names_number_whatever_its_line() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lined = with_line(LEAST_LOAD, 2);
        std::fs::write(dir.path().join(name_of(7, "yaml")), lined).expect("the file");
        let (store, generations) = generations();
        bind(&generations, dir.path());
        assert_eq!(standing(&store).0, 7);
    }

    /// A process that ended between the rename and the store's entry: the
    /// generation is on file, and the next start runs it.
    #[test]
    fn a_generation_persisted_and_never_entered_runs_at_the_next_start() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut on_file = SpecStore::open(dir.path().to_owned()).expect("an empty directory");
        let first = Persisted {
            generation: 1,
            format: Format::Yaml,
        };
        on_file
            .persist(first, LEAST_LOAD.as_bytes())
            .expect("it persists");
        drop(on_file);

        let (store, generations) = generations();
        bind(&generations, dir.path());
        assert_eq!(standing(&store).0, 1);
    }

    #[test]
    fn a_newest_generation_that_does_not_run_refuses_the_bind() {
        let dir = tempfile::tempdir().expect("a temp dir");
        std::fs::write(dir.path().join(name_of(4, "yaml")), LEAST_LOAD).expect("the file");
        std::fs::write(dir.path().join(name_of(5, "yaml")), UNKNOWN_FILTER).expect("the file");
        let (store, generations) = generations();
        let opened = SpecStore::open(dir.path().to_owned()).expect("the directory opens");

        let refused = generations.bind_store(opened).expect_err("it does not run");
        assert!(
            matches!(
                &refused,
                StoreRefused::Rejected {
                    generation: 5,
                    source: SpecRejected::Invalid(_)
                }
            ),
            "{refused:?}"
        );
        assert!(
            refused.to_string().starts_with("generation 5: "),
            "{refused}"
        );
        // Nothing runs and no source is bound: the generation under the
        // newest is not taken in its place.
        assert_eq!(standing(&store), (0, None));
        assert!(!generations.writable() && !generations.bound());
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
        assert_eq!(yaml(&filed, LEAST_LOAD), Ok(Applied::Generation(1)));
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
        assert_eq!(generations.counts(), applied(0, 0));
    }

    /// A number on file that this door did not write, as another process on
    /// the directory leaves one: the apply is refused by name, the file
    /// stays as it was, and the running generation stays.
    #[test]
    fn a_generation_on_file_already_is_not_written_over() {
        let (dir, store, generations) = stored();
        enforce(&generations, LEAST_LOAD).expect("it runs");
        let theirs = dir.path().join(name_of(2, "json"));
        std::fs::write(&theirs, b"theirs").expect("the file");

        let refused = enforce(&generations, RANDOM).expect_err("the number is taken");
        let taken = "generation 2 is on file already".to_owned();
        assert_eq!(refused, SpecRejected::Unpersisted(taken));
        assert_eq!(standing(&store).0, 1);
        assert_eq!(std::fs::read(&theirs).expect("the file"), b"theirs");
        assert_eq!(generations.counts(), applied(1, 1));
    }

    /// A directory that does not take the write: nothing is applied, the
    /// running generation stays, and the refusal says what the directory
    /// said.
    #[test]
    fn a_document_the_directory_does_not_take_is_not_applied() {
        let outer = tempfile::tempdir().expect("a temp dir");
        let dir = outer.path().join("store");
        std::fs::create_dir(&dir).expect("the directory");
        let (store, generations) = generations();
        bind(&generations, &dir);
        let (_, stands) = enforce(&generations, LEAST_LOAD).expect("it runs");
        std::fs::remove_dir_all(&dir).expect("the directory goes");

        let refused = enforce(&generations, RANDOM).expect_err("nowhere to persist");
        assert!(
            matches!(refused, SpecRejected::Unpersisted(_)),
            "{refused:?}"
        );
        assert!(
            refused.to_string().starts_with("not persisted: "),
            "{refused}"
        );
        assert_eq!(refused.field_path(), "");
        assert_eq!(standing(&store), (1, Some(stands)));
        assert_eq!(generations.counts(), applied(1, 1));
    }

    /// The entry's refusal takes the persisted generation off file before
    /// the door's lock goes: a start never reads a generation that did not
    /// run. The case's validator enters the number itself, behind the
    /// door's back, so the door's own call of the entry is refused.
    #[test]
    fn a_generation_the_entry_refuses_goes_off_file_again() {
        static BEHIND: OnceLock<Arc<MemoryStore>> = OnceLock::new();
        fn racing(
            document: &PolicySpec,
            digest: Digest,
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            if let Some(store) = BEHIND.get().filter(|_| document.generation == 2) {
                let taken = AppliedSpec::validate_digested(document, digest, catalog)?;
                store.apply(taken).expect("the number is free");
            }
            AppliedSpec::validate_digested(document, digest, catalog)
        }

        let dir = tempfile::tempdir().expect("a temp dir");
        let store = Arc::new(MemoryStore::new());
        let generations = Generations::validating(Arc::clone(&store), stock(), racing);
        bind(&generations, dir.path());
        enforce(&generations, LEAST_LOAD).expect("it runs");
        assert!(BEHIND.set(Arc::clone(&store)).is_ok(), "set once");

        let refused = enforce(&generations, RANDOM).expect_err("an equal number");
        let stale = StaleGeneration {
            standing: 2,
            offered: 2,
        };
        assert_eq!(refused, SpecRejected::Stale(stale));
        assert_eq!(names(dir.path()), BTreeSet::from([name_of(1, "yaml")]));
        assert_eq!(generations.counts(), applied(1, 1));
    }

    /// Two enforces that read one digest as standing: one lands, and the
    /// others are told that another apply landed first.
    #[test]
    fn enforces_under_one_guard_land_once() {
        const WRITERS: u64 = 8;
        let (dir, store, generations) = stored();
        let (_, first) = enforce(&generations, LEAST_LOAD).expect("it runs");
        let read = first.to_string();
        let landed: Vec<bool> = thread::scope(|scope| {
            let writers: Vec<_> = (0..WRITERS)
                .map(|n| {
                    let (generations, read) = (&generations, &read);
                    scope.spawn(move || {
                        let document = bound_to(n).into_bytes();
                        match generations.enforce(read, document, Format::Yaml) {
                            Ok(enforced) => enforced.ran.is_ok(),
                            Err(Unanswered::Guarded(..)) => false,
                            Err(other) => panic!("{other:?}"),
                        }
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| writer.join().expect("no writer panics"))
                .collect()
        });
        assert_eq!(
            landed.iter().filter(|landed| **landed).count(),
            1,
            "{landed:?}"
        );
        assert_eq!(standing(&store).0, 2);
        assert_eq!(names(dir.path()).len(), 2);
    }

    /// The bound of what waits: no more write calls are let in than the
    /// one that applies and the ones behind it.
    #[tokio::test]
    async fn a_write_call_past_the_waiting_ones_is_refused() {
        let (_dir, store, generations) = stored();
        let places = u32::try_from(WRITES_WAITING + 1).expect("a small count");
        let taken = Arc::clone(&generations.writing)
            .try_acquire_many_owned(places)
            .expect("every place is free");
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
        let places = u32::try_from(WRITES_WAITING + 1).expect("a small count");
        let _taken = Arc::clone(&generations.writing)
            .try_acquire_many_owned(places)
            .expect("every place is free");
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
            matches!(guarded, Err(Unanswered::Guarded(0, None))),
            "{guarded:?}"
        );
        assert!(names(dir.path()).is_empty());
        assert_eq!(generations.counts(), applied(0, 1));
    }

    /// The bound of what a store keeps on file: the last eight
    /// generations, at every apply.
    #[test]
    fn a_store_keeps_its_last_generations_on_file() {
        let (dir, _store, generations) = stored();
        let last = u64::try_from(RETAINED).expect("a small count") + 2;
        for number in 1..=last {
            let ran = enforce(&generations, bound_to(number)).map(|(number, _)| number);
            assert_eq!(ran, Ok(number));
            assert!(names(dir.path()).len() <= RETAINED);
        }
        let on_file = (3..=last).map(|number| name_of(number, "yaml")).collect();
        assert_eq!(names(dir.path()), on_file);
        // The number the flag's help states.
        assert_eq!(names(dir.path()).len(), 8);
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
        let status = file.status().expect("a source");
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
        assert_eq!(stored.status(), Some(empty));
        for (number, document) in [
            (1, LEAST_LOAD.to_owned()),
            (2, RANDOM.to_owned()),
            (3, bound_to(3)),
        ] {
            let ran = enforce(&stored, document).map(|(number, _)| number);
            assert_eq!(ran, Ok(number));
        }
        let status = stored.status().expect("a source");
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
        let rolled = stored.status().expect("a source");
        assert_eq!(
            (rolled.generation, rolled.retained.as_slice()),
            (4, &[1, 2, 3][..])
        );

        // A start on the same directory lists the same.
        drop(stored);
        let (_store, again) = generations();
        bind(&again, dir.path());
        let listed = again.status().expect("a source");
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
            first.status(),
            Some(proto::SpecStatus {
                generation,
                digest: digest.expect("a digest").to_string(),
                source: "file".to_owned(),
                ..proto::SpecStatus::default()
            })
        );
        // The door that applied it tells its time.
        let told = second.status().expect("a source");
        assert_eq!(told.mode, i32::from(proto::ApplyMode::Enforce));
        assert!(told.applied_unix_micros > 0, "{told:?}");
    }

    /// Two doors on one directory, as two processes hold one, each enforce
    /// a document of their own at one moment: one runs it as generation 1,
    /// and the other is refused by name with nothing applied. The directory
    /// holds the document that runs, and a start on it runs that one.
    #[test]
    fn two_doors_on_one_directory_do_not_both_run_one_number() {
        for round in 0..20 {
            let dir = tempfile::tempdir().expect("a temp dir");
            // The second door is a writer the directory's lock did not keep
            // out.
            let binds = [bind as fn(&Generations, &Path), bind_beside];
            let doors = [(LEAST_LOAD, binds[0]), (RANDOM, binds[1])].map(|(document, bind)| {
                let (store, generations) = generations();
                bind(&generations, dir.path());
                (store, generations, document)
            });
            let barrier = std::sync::Barrier::new(2);
            let [first, second] = thread::scope(|scope| {
                let calls = doors.each_ref().map(|(_, generations, document)| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        enforce(generations, *document)
                    })
                });
                calls.map(|call| call.join().expect("no panic"))
            });
            let taken = SpecRejected::Unpersisted("generation 1 is on file already".to_owned());
            let ((won, ran), (lost, refused)) = match (first, second) {
                (Ok(ran), Err(refused)) => ((&doors[0], ran), (&doors[1], refused)),
                (Err(refused), Ok(ran)) => ((&doors[1], ran), (&doors[0], refused)),
                both => panic!("round {round}: one door runs generation 1: {both:?}"),
            };
            assert_eq!(refused, taken);
            let (number, digest) = ran;
            assert_eq!(number, 1);
            assert_eq!(standing(&won.0), (1, Some(digest)));
            assert_eq!(standing(&lost.0), (0, None));
            let on_file = std::fs::read(dir.path().join(name_of(1, "yaml"))).expect("the file");
            assert_eq!(on_file, won.2.as_bytes());

            // The two are gone, as their processes are at the next start.
            drop(doors);
            let (store, again) = generations();
            bind(&again, dir.path());
            assert_eq!(standing(&store), (1, Some(digest)));
        }
    }

    /// Write calls take the door one at a time, in the order they came: a
    /// call awaits its turn before its thread starts, so while the turn is
    /// held none runs, and then each runs as the next number in its order.
    #[tokio::test]
    async fn write_calls_take_their_turn_in_the_order_they_came() {
        let (_dir, store, generations) = stored();
        let held = Arc::clone(&generations.write_turn).lock_owned().await;
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
        while door.writing.available_permits() != free {
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
                generations.write_turn.try_lock().is_err(),
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
        assert_eq!(generations.writing.available_permits(), WRITES_WAITING + 1);
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
        let Err(Unanswered::NotHeld(not_held)) = gone else {
            panic!("generation 1 has no file: {gone:?}");
        };
        assert_eq!((not_held.asked, not_held.standing), (1, 2));
    }

    /// A pipe under a generation's name, with no writer, holds no start:
    /// under the newest name it refuses the bind by that generation, and
    /// under an older one it is not listed, as a file that is no document.
    /// A bind runs on a thread of its own, waited five seconds, so an open
    /// that waits fails the case.
    #[cfg(unix)]
    #[test]
    fn a_pipe_under_a_generations_name_holds_no_start() {
        let bound_in_time = |dir: &Path| {
            let (store, generations) = generations();
            let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
            let (answer, answered) = mpsc::channel();
            thread::spawn(move || {
                let _ = answer.send(generations.bind_store(opened));
            });
            let bound = answered
                .recv_timeout(Duration::from_secs(5))
                .expect("the bind answers: its open does not wait");
            (bound, standing(&store).0)
        };
        let dir = tempfile::tempdir().expect("a temp dir");
        pipe_at(&dir.path().join(name_of(1, "yaml")));

        let (bound, runs) = bound_in_time(dir.path());
        let refused = bound.expect_err("no regular file");
        assert!(
            matches!(
                &refused,
                StoreRefused::Rejected {
                    generation: 1,
                    source: SpecRejected::Unreadable(told)
                } if told == "the path leads to no regular file"
            ),
            "{refused:?}"
        );
        assert_eq!(runs, 0);

        std::fs::write(dir.path().join(name_of(2, "yaml")), LEAST_LOAD).expect("the file");
        let (bound, runs) = bound_in_time(dir.path());
        bound.expect("the newest runs");
        assert_eq!(runs, 2);
    }

    /// A caller that leaves while its call waits for the turn writes
    /// nothing and frees its place: the next call runs as the first
    /// generation, and its document alone is on file.
    #[tokio::test]
    async fn a_write_call_whose_caller_left_while_it_waited_writes_nothing() {
        let (dir, store, generations) = stored();
        let held = Arc::clone(&generations.write_turn).lock_owned().await;
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

    /// At the last number none is left: an enforce and a rollback are
    /// refused in the entry's words, as their dry run is, before a file is
    /// written.
    #[test]
    fn a_write_at_the_last_number_is_refused_as_its_dry_run_is() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let last = u64::MAX;
        std::fs::write(dir.path().join(name_of(last - 1, "yaml")), RANDOM).expect("the file");
        std::fs::write(dir.path().join(name_of(last, "yaml")), LEAST_LOAD).expect("the file");
        let (store, generations) = generations();
        bind(&generations, dir.path());
        assert_eq!(standing(&store).0, last);
        let on_file = names(dir.path());
        let stale = || {
            SpecRejected::Stale(StaleGeneration {
                standing: last,
                offered: last,
            })
        };

        let dry = generations
            .dry_run("", RANDOM.as_bytes(), Format::Yaml)
            .expect("no guard");
        assert_eq!(dry.would_run, Err(stale()));
        assert_eq!(enforce(&generations, RANDOM), Err(stale()));
        assert_eq!(roll_back(&generations, last - 1), Err(stale()));
        assert_eq!(names(dir.path()), on_file);
        assert_eq!(generations.counts(), applied(1, 2));
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
        let ran = needs_summary(store.world().spec().engine());

        let back = generations.roll_back(1).expect("on file");
        assert_eq!(back.standing, (2, Some(second)));
        assert_eq!(back.ran, Ok((3, first)));
        assert_eq!(standing(&store), (3, Some(first)));
        let on_file = std::fs::read(dir.path().join(name_of(3, "yaml"))).expect("the file");
        assert_eq!(on_file, LEAST_LOAD.as_bytes());
        let needs = back.needs.expect("the diff");
        assert_eq!(needs.before, Some(ran));
        let runs = needs_summary(store.world().spec().engine());
        assert_eq!(needs.after, Some(runs));

        for same in [3, 1] {
            assert_eq!(roll_back(&generations, same), Ok((3, first)));
        }
        assert_eq!(names(dir.path()).len(), 3);
        assert_eq!(generations.counts(), applied(3, 0));
    }

    /// A generation that is not on file is not held: one never applied, and
    /// one past the generations the store keeps. Nothing is applied and
    /// nothing counted.
    #[test]
    fn a_rollback_to_a_generation_not_on_file_is_not_held() {
        let (_dir, store, generations) = stored();
        let last = u64::try_from(RETAINED).expect("a small count") + 1;
        for number in 1..=last {
            enforce(&generations, bound_to(number)).expect("it runs");
        }
        for gone in [1, last + 1] {
            let Err(Unanswered::NotHeld(not_held)) = generations.roll_back(gone) else {
                panic!("generation {gone} is not on file");
            };
            let expected = NotHeld {
                asked: gone,
                held: Some(last),
                standing: last,
                on_file: Some(OnFile {
                    count: RETAINED,
                    span: (2, last),
                }),
            };
            assert_eq!(not_held, expected);
            assert_eq!(
                not_held.to_string(),
                format!("no document is held for generation {gone}: the spec store has 8 on file, from generation 2 to generation 9")
            );
        }
        assert_eq!(standing(&store).0, last);
        assert_eq!(generations.counts(), applied(last, 0));
    }

    /// A rollback mints its number whatever line the generation's bytes
    /// write. A file that is no document any more, as a build that dropped
    /// a stage finds one, is refused by name with the running generation
    /// kept, counted, and nothing more on file.
    #[test]
    fn a_rollback_mints_its_number_and_refuses_a_document_that_does_not_run() {
        let (dir, store, generations) = stored();
        enforce(&generations, with_line(LEAST_LOAD, 1)).expect("it runs");
        enforce(&generations, RANDOM).expect("it runs");
        assert_eq!(roll_back(&generations, 1).map(|(number, _)| number), Ok(3));
        let on_file = std::fs::read(dir.path().join(name_of(3, "yaml"))).expect("the file");
        assert_eq!(on_file, with_line(LEAST_LOAD, 1));

        std::fs::write(dir.path().join(name_of(2, "yaml")), UNKNOWN_FILTER).expect("the file");
        let refused = roll_back(&generations, 2).expect_err("no such filter");
        assert!(matches!(refused, SpecRejected::Invalid(_)), "{refused:?}");
        assert_eq!(standing(&store).0, 3);
        assert_eq!(names(dir.path()).len(), 3);
        assert_eq!(generations.counts(), applied(3, 1));
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

        let Err(Unanswered::NotHeld(not_held)) = generations.read(7).await else {
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

        let Err(Unanswered::NotHeld(not_held)) = generations.read(1).await else {
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

    /// The Numbers table's enforce rows: one enforce under a spec store
    /// (parse, digest, validate, the file written and synced, the store's
    /// entry) by the document's size, up to the bound. The directory is the
    /// process's temporary one: point it at a disk for a row that syncs.
    #[test]
    #[ignore = "a measurement for the Numbers table, run by name"]
    fn an_enforces_time_by_the_documents_size() {
        const ROUNDS: usize = 15;
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let mut out = std::io::stdout().lock();
        for size in [2 << 10, 64 << 10, bound - 128] {
            let (document, bindings) = document_of(size);
            let mut micros = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let (_dir, _store, generations) = stored();
                let document = document.clone();
                let began = Instant::now();
                let ran = enforce(&generations, document);
                micros.push(began.elapsed().as_micros());
                assert_eq!(ran.map(|(number, _)| number), Ok(1));
            }
            micros.sort_unstable();
            writeln!(
                out,
                "ENFORCE bytes={} bindings={bindings} rounds={ROUNDS} median_us={} worst_us={}",
                document.len(),
                micros[ROUNDS / 2],
                micros[ROUNDS - 1]
            )
            .expect("stdout");
        }
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
    fn a_form_is_named_as_it_is_written() {
        for format in [Format::Yaml, Format::Json] {
            assert_eq!(Format::named(format.as_str()), Some(format));
        }
        assert_eq!(Format::named("YAML"), None);
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
            standby.document(0).expect("held"),
            leader.document(0).expect("held")
        );
        assert_eq!(standby.counts().applied, 1);
        // Told again, it changes nothing.
        followed(&standby, &told).expect("the standing one again");
        // Nor does an event with no document, whatever its number.
        followed(&standby, &worker_wire::generation_told(9)).expect("no document");
        assert_eq!(standing(&store), standing(&led));
        assert_eq!(standby.counts().applied, 1);
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
        assert_eq!(standby.counts().applied, 1);
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
        assert_eq!(door.counts().applied, 0);
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
        assert_eq!(door.counts().applied, 2);
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
        let counts = door.counts();
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
        assert_eq!(standby.counts().applied, 1);
        assert_eq!(
            standby.document(0).expect("held").document,
            RANDOM.as_bytes()
        );
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
    /// the replica does not lead is not read, whatever its watch saw ahead
    /// of a look, and nothing is counted. The log writes there all the
    /// same.
    #[test]
    fn under_a_shared_lease_the_source_writes_while_its_replica_leads() {
        let (store, door) = generations();
        let mut elector = election();
        door.under(elector.fence());
        assert!(!door.source_writes(), "it has not led yet");
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Generation(1)));
        // Its lead ends: a look that was in flight applies nothing.
        assert!(elector.release().expect("a release"), "it held the lease");
        assert_eq!(yaml(&door, LEAST_LOAD), Ok(Applied::Waits(1)));
        assert_eq!(standing(&store).0, 1);
        assert_eq!(door.counts(), applied(1, 0));
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
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
            assert!(
                matches!(left, Some(Ok(true))),
                "it held the lease: {left:?}"
            );
            AppliedSpec::validate_digested(document, digest, catalog)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), ending);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.counts(), applied(0, 0));
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
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            PASSES.with(|passes| passes.set(passes.get() + 1));
            faulting(document, digest, catalog)
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
        let awaits = |store: &MemoryStore| store.world().spec().awaits_document();
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
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            let passed = PASSES.with(|passes| passes.replace(passes.get() + 1));
            assert_eq!(passed, 0, "a fault of the validator");
            AppliedSpec::validate_digested(document, digest, catalog)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), faulting_again);
        door.defer(RANDOM.into(), Format::Yaml)
            .expect("a valid document");
        door.at_take();
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.counts(), applied(0, 1));
        assert!(door.deferred.lock().is_none(), "the bytes are let go");
        assert!(door.bound(), "a source all the same");
    }

    /// A fence that closes while the entry waits for a lock publishes
    /// nothing: the store asks the last word under the lock it publishes
    /// under, and the standing generation stays.
    #[test]
    fn a_fence_that_closes_while_the_entry_waits_for_a_lock_publishes_nothing() {
        let deadline = Duration::from_millis(800);
        let store = Arc::new(MemoryStore::new());
        let door = Generations::new(Arc::clone(&store), stock());
        let mut elector = election_under(deadline);
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        // The case holds the lock of the held bytes: the entry passes both
        // reads of the fence under the door's lock and waits behind it.
        let gate = door.held.lock();
        let outcome = thread::scope(|scope| {
            let entering = scope.spawn(|| yaml(&door, RANDOM));
            thread::sleep(deadline + Duration::from_millis(400));
            assert!(!door.source_writes(), "the fence closed by the clock");
            drop(gate);
            entering.join().expect("the entry's thread")
        });
        assert_eq!(outcome, Ok(Applied::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.counts(), applied(0, 0));
        assert!(door.held.lock().is_none(), "no bytes are kept");
    }

    /// The store asks its last word behind every lock its entry waits for.
    /// An entry that waits for the loads' lock, the last one, while the
    /// fence closes publishes nothing, and the standing generation stays.
    #[test]
    fn a_fence_that_closes_while_the_store_waits_for_its_last_lock_publishes_nothing() {
        let deadline = Duration::from_millis(800);
        let store = Arc::new(MemoryStore::new());
        let door = Generations::new(Arc::clone(&store), stock());
        let mut elector = election_under(deadline);
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert!(door.source_writes());
        // The case holds the loads' lock, as a load's write does: the entry
        // passes the door's reads of the fence and waits inside the store.
        let gate = store.holding_loads();
        let outcome = thread::scope(|scope| {
            let entering = scope.spawn(|| yaml(&door, RANDOM));
            thread::sleep(deadline + Duration::from_millis(400));
            assert!(!door.source_writes(), "the fence closed by the clock");
            drop(gate);
            entering.join().expect("the entry's thread")
        });
        assert_eq!(outcome, Ok(Applied::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.counts(), applied(0, 0));
        assert!(door.held.lock().is_none(), "no bytes are kept");
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
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 1 {
                let left = LEAD.with_borrow_mut(|lead| lead.as_mut().map(Elector::release));
                assert!(
                    matches!(left, Some(Ok(true))),
                    "it held the lease: {left:?}"
                );
            }
            AppliedSpec::validate_digested(document, digest, catalog)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), ending);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Generation(1)));
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Waits(1)));
        assert_eq!(door.counts(), applied(1, 0));
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
            catalog: &Catalog,
        ) -> Result<AppliedSpec, SpecError> {
            if PASSES.with(|passes| passes.replace(passes.get() + 1)) == 0 {
                LEAD.with_borrow_mut(|lead| {
                    let lead = lead.as_mut().expect("an elector");
                    assert!(matches!(lead.release(), Ok(true)), "it held the lease");
                    lead.tick(Instant::now()).expect("a second take");
                });
            }
            AppliedSpec::validate_digested(document, digest, catalog)
        }
        let store = Arc::new(MemoryStore::new());
        let door = Generations::validating(Arc::clone(&store), stock(), retaking);
        let mut elector = election();
        door.under(elector.fence());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(door.lead(), Some(Leading::Term(1)));
        LEAD.set(Some(elector));
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Waits(0)));
        assert_eq!(standing(&store), (0, None));
        assert_eq!(door.lead(), Some(Leading::Term(2)), "it leads again");
        // Read again under the term that shows now, the bytes are applied.
        assert_eq!(yaml(&door, RANDOM), Ok(Applied::Generation(1)));
        assert_eq!(door.counts(), applied(1, 0));
    }
}

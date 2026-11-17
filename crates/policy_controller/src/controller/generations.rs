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
//! A source that keeps running hands a refusal back here
//! ([`Generations::reject`]): it is one event and one count, and the
//! running generation stays.
//!
//! The number and the digest have one holder, the store's
//! ([`World::spec`](crate::store::World::spec)). What lives here is the
//! record around the bytes: the document as the source wrote it, its format
//! and when it was applied. `GetPolicySpec` answers those bytes with the
//! number and the digest beside them, never a re-serialized form. One
//! document is kept, the standing generation's: a file source retains no
//! other.
//!
//! Limits. A minted number is this replica's own: a replica that starts
//! remembers nothing and admits its source's document as it stands, so two
//! replicas that read one file agree on a number only where the file writes
//! one. A digest is the applier's ([`Digest`]): it travels with its bytes,
//! and a reader of another build does not recompute it to compare.

use std::{
    any::Any,
    cell::Cell,
    fmt,
    panic::{self, AssertUnwindSafe},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{map::Entry, Map, Number, Value};
use tracing::{info, warn};

use crate::{
    catalog::Catalog,
    proto,
    spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    store::{MemoryStore, StaleGeneration},
};

/// The largest document a source hands over, in bytes: 128 KiB. What sets
/// the bound is what an apply may hold beside the document, 64 MiB: the
/// budget of one apply in a controller whose container is a few hundred
/// mebibytes. By run a YAML document of this size holds under 56 MiB
/// ([`read_yaml`]). A fleet's document is far under the bound: 143 names
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

    /// The document these bytes read as. A YAML document is read once, into
    /// the values it reads as, held under a bound ([`read_yaml`]), and the
    /// document is built from those values alone. So a YAML document reads
    /// as JSON does: text that looks like a number or a boolean is quoted
    /// where a field takes text. A field the reader refuses is named by its
    /// path.
    fn parse(self, document: &[u8]) -> Result<PolicySpec, String> {
        match self {
            Self::Yaml => {
                let values = read_yaml(document)?;
                serde_path_to_error::deserialize(values).map_err(|refused| {
                    let path = refused.path().to_string();
                    // The document itself has no name.
                    if path == "." {
                        refused.into_inner().to_string()
                    } else {
                        format!("{path}: {}", refused.inner())
                    }
                })
            }
            Self::Json => serde_json::from_slice(document).map_err(|error| error.to_string()),
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
    #[error("not a policy spec in {format}: {message}")]
    Unparsed {
        /// `yaml` or `json`.
        format: &'static str,
        /// The parser's message: the place, and an unknown field by name.
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
}

impl fmt::Display for NotHeld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no document is held for generation {}: ", self.asked)?;
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

/// The standing generation's document as its source wrote it.
struct Held {
    generation: u64,
    digest: Digest,
    document: Vec<u8>,
    format: Format,
    applied: SystemTime,
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

/// A document's own pass, in its one order: the bound, the form, the
/// digest, the validator, then the line against the standing generation.
/// Nothing is applied, nothing held and no lock taken.
fn check(
    (validate, catalog): (Validate, &Catalog),
    (standing, standing_digest): (u64, Option<Digest>),
    document: &[u8],
    format: Format,
) -> Result<Checked, SpecRejected> {
    let len = u64::try_from(document.len()).unwrap_or(u64::MAX);
    if len > MAX_DOCUMENT_BYTES {
        return Err(SpecRejected::TooLarge(len));
    }
    let mut spec = format
        .parse(document)
        .map_err(|message| SpecRejected::Unparsed {
            format: format.as_str(),
            message,
        })?;
    let digest = spec.digest()?;
    let line = spec.generation;
    if line == 0 {
        // The mint. At the last number the entry refuses it by name.
        spec.generation = standing.saturating_add(1);
    }
    // The validator's word comes first: the digest reads no line the
    // canonical form leaves out, so equal digests admit nothing. The digest
    // is taken once, above, and handed on.
    let next = validate(&spec, digest, catalog)?;
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

/// A panic's message, where it has one.
fn fault_text(fault: &(dyn Any + Send)) -> String {
    fault
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| fault.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_owned())
}

/// The controller's generations: the door a source applies through and the
/// standing document.
pub(crate) struct Generations {
    store: Arc<MemoryStore>,
    /// The stages a document may name: the ones the bind was handed, the
    /// host's own among them.
    catalog: Arc<Catalog>,
    validate: Validate,
    /// One apply at a time: the standing number is read, the document's
    /// pass run and the store's entry called under this lock. No reader
    /// takes it.
    applying: Mutex<()>,
    /// The standing generation's document. An apply swaps it under this
    /// lock with its call of the store's entry, so a reader sees the number
    /// and the document move together and waits on no validation. `None`
    /// until a source has applied its first document.
    held: Mutex<Option<Held>>,
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
            applying: Mutex::new(()),
            held: Mutex::new(None),
            applied: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Applies `document`, as its source wrote it in `format`. A refusal
    /// applies nothing and the running generation stays; what a refusal
    /// means is its source's to say: a start refuses to start, and a source
    /// that keeps running counts it ([`Generations::reject`]).
    pub(crate) fn apply(&self, document: Vec<u8>, format: Format) -> Result<Applied, SpecRejected> {
        let applying = self.applying.lock();
        let standing = self.standing();
        let Checked { digest, next } = self.pass(standing, &document, format)?;
        let (standing, _) = standing;
        let bytes = document.len();
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
        {
            let mut held = self.held.lock();
            self.store.apply(next)?;
            *held = Some(hold(generation));
        }
        drop(applying);

        self.applied.fetch_add(1, Ordering::Relaxed);
        info!(
            generation,
            %digest,
            format = format.as_str(),
            bytes,
            "policy spec applied"
        );
        Ok(Applied::Generation(generation))
    }

    /// A document's pass against the generation that stands, with its fault
    /// held: a panic of the pass is a refusal by name.
    fn pass(
        &self,
        standing: (u64, Option<Digest>),
        document: &[u8],
        format: Format,
    ) -> Result<Checked, SpecRejected> {
        let validator = (self.validate, self.catalog.as_ref());
        let passed = panic::catch_unwind(AssertUnwindSafe(|| {
            check(validator, standing, document, format)
        }));
        passed.unwrap_or_else(|fault| {
            // The text first: a payload's own drop can panic, and it is
            // freed under a catch of its own.
            let told = fault_text(fault.as_ref());
            crate::framework::release(fault);
            Err(SpecRejected::Faulted(told))
        })
    }

    /// The number and the digest the store runs.
    fn standing(&self) -> (u64, Option<Digest>) {
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

    /// Whether a source has applied a document: the spec calls are served
    /// and the spec's series rendered from then on.
    pub(crate) fn bound(&self) -> bool {
        self.held.lock().is_some()
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
            }),
        }
    }
}

fn unix_micros(at: SystemTime) -> i64 {
    at.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|since| i64::try_from(since.as_micros()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        fmt::Write as _,
        io::Write as _,
        sync::{atomic::AtomicBool, OnceLock, Weak},
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::{Candidate, Constraints, Filter, RequestFacts};

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
                message,
            }) => message,
            other => panic!("applied as {other:?}"),
        };
        let unknown = unparsed(LEAST_LOAD.replace("    pick: best\n", "    colour: blue\n"));
        assert!(
            unknown.starts_with("profiles.default") && unknown.contains("colour"),
            "{unknown}"
        );
        let number = unparsed(LEAST_LOAD.replace("pick: best", "pick: 1"));
        assert!(number.starts_with("profiles.default.pick: "), "{number}");
        assert_eq!(
            yaml(
                &generations,
                LEAST_LOAD.replace("[healthy]", "[\"healthy\"]")
            ),
            Ok(Applied::Generation(1))
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

        /// The budget holds in every run: the heaviest file the search finds
        /// (bindings of nothing to the size bound, beside one profile of a
        /// thousand scorers under the most names the bounds admit), applied
        /// in a process of its own, holds no more than one apply's budget. A
        /// larger size bound, or a type that grows past the spread between
        /// runs, is told here. A copy of the document that comes back reads
        /// at this case's edge, and the search tells it.
        #[test]
        fn the_heaviest_file_found_holds_no_more_than_an_applys_budget() {
            let len = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
            let profile = scorers();
            let file = |count: usize| beside_bindings(named(&profile, "*p", count), len);
            let count = heaviest((len - 8192) / 40, &file);
            let dir = tempfile::tempdir().expect("a directory");
            let path = dir.path().join("spec.yaml");
            fs::write(&path, file(count)).expect("the file");
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
}

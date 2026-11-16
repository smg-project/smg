//! The policy spec's generations as the controller holds them (design 4.6
//! and 5.1's spec row): the one door a spec source applies a document
//! through, and the document the standing generation was applied from, as
//! its source wrote it.
//!
//! One invariant: **the writer that applies mints the number.** A source
//! hands over a document; the number it runs under is decided here, under
//! one lock, against the generation the store holds. A document that writes
//! no `generation` line takes the standing number plus one. A line is a
//! claim the store's entry checks: it admits a number above the standing
//! one and refuses any other by naming the standing one. A document whose
//! digest is the standing one's, under no line or under the standing line,
//! is no change: nothing is applied. A changed document under the standing
//! line is refused by name, so on one replica a number never names two
//! documents.
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
    fmt,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use tracing::info;

use crate::{
    proto,
    spec::{AppliedSpec, Digest, PolicySpec, SpecError},
    store::{MemoryStore, StaleGeneration},
};

/// The largest document a source hands over, in bytes: a ConfigMap holds no
/// more, and `GetPolicySpec`'s answer stays inside one frame.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 1 << 20;

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

    fn parse(self, document: &[u8]) -> Result<PolicySpec, String> {
        match self {
            Self::Yaml => serde_yaml::from_slice(document).map_err(|error| error.to_string()),
            Self::Json => serde_json::from_slice(document).map_err(|error| error.to_string()),
        }
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
    /// It is not a document of its format, or it writes a field the schema
    /// does not know.
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

/// The standing generation's document as its source wrote it.
struct Held {
    generation: u64,
    digest: Digest,
    document: Vec<u8>,
    format: Format,
    applied: SystemTime,
}

/// The controller's generations: the door a source applies through and the
/// standing document.
pub(crate) struct Generations {
    store: Arc<MemoryStore>,
    /// One apply at a time: the standing number is read, the next one minted
    /// and the store's entry called under this lock. `None` until a source
    /// has applied its first document.
    held: Mutex<Option<Held>>,
}

impl Generations {
    pub(crate) fn new(store: Arc<MemoryStore>) -> Self {
        Self {
            store,
            held: Mutex::new(None),
        }
    }

    /// Applies `document`, as its source wrote it in `format`. A refusal
    /// applies nothing and the running generation stays; what a refusal
    /// means is its source's to say (a start refuses to start).
    pub(crate) fn apply(&self, document: Vec<u8>, format: Format) -> Result<Applied, SpecRejected> {
        let len = u64::try_from(document.len()).unwrap_or(u64::MAX);
        if len > MAX_DOCUMENT_BYTES {
            return Err(SpecRejected::TooLarge(len));
        }
        let mut spec = format
            .parse(&document)
            .map_err(|message| SpecRejected::Unparsed {
                format: format.as_str(),
                message,
            })?;
        let digest = spec.digest()?;

        let mut held = self.held.lock();
        let (standing, standing_digest) = {
            let world = self.store.world();
            (world.spec().generation(), world.spec().digest())
        };
        let hold = |generation| Held {
            generation,
            digest,
            document,
            format,
            applied: SystemTime::now(),
        };
        let line = spec.generation;
        if line == 0 || line == standing {
            if standing_digest == Some(digest) {
                // The standing generation's document again, or in another
                // form: it runs already. Where no bytes of it are held,
                // these are kept, so the read call has the standing one.
                if held.as_ref().is_none_or(|held| held.generation != standing) {
                    *held = Some(hold(standing));
                }
                return Ok(Applied::Unchanged(standing));
            }
            if let (true, Some(standing_digest)) = (line != 0, standing_digest) {
                return Err(SpecRejected::ChangedUnderItsLine {
                    generation: standing,
                    offered: digest,
                    standing: standing_digest,
                });
            }
        }
        if line == 0 {
            // The mint. At the last number the entry refuses it by name.
            spec.generation = standing.saturating_add(1);
        }
        let applied = AppliedSpec::validate(&spec)?;
        let generation = applied.generation();
        self.store.apply(applied)?;
        *held = Some(hold(generation));
        drop(held);

        info!(
            generation,
            %digest,
            format = format.as_str(),
            bytes = len,
            "policy spec applied"
        );
        Ok(Applied::Generation(generation))
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
    use std::{fmt::Write as _, io::Write as _, thread, time::Instant};

    use super::*;

    /// A document in the form of the design's example: no `generation` line.
    pub(crate) const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";
    /// Another one.
    pub(crate) const RANDOM: &str =
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

    fn generations() -> (Arc<MemoryStore>, Generations) {
        let store = Arc::new(MemoryStore::new());
        (Arc::clone(&store), Generations::new(store))
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

    /// The bound fails fast: a document one byte over it is refused by its
    /// size, before any of it is parsed.
    #[test]
    fn a_document_over_the_bound_is_refused_by_its_size() {
        let (store, generations) = generations();
        let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 1;
        assert_eq!(
            generations.apply(vec![b'{'; over], Format::Json),
            Err(SpecRejected::TooLarge(MAX_DOCUMENT_BYTES + 1))
        );
        assert_eq!(standing(&store), (0, None));
        assert!(!generations.bound());
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
    }
}

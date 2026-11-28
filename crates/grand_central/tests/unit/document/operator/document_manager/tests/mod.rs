use std::{
    fmt::Write as _,
    io::Write as _,
    path::Path,
    sync::{atomic::AtomicBool, Weak},
    thread,
    time::{Duration, Instant},
};

use super::*;
use crate::{
    document::operator::{test_support::*, MAX_DOCUMENT_BYTES},
    engine::{Candidate, Constraints, Filter, RequestFacts},
};

mod on_file;
#[cfg(target_os = "linux")]
mod search;

impl DocumentManager {
    fn bound(&self) -> bool {
        self.writable() || self.held.lock().is_some()
    }
}

/// The door's two locks, for a case in another module.
#[cfg(feature = "controller")]
impl DocumentManager {
    pub(crate) fn applying_lock(&self) -> &Mutex<Option<SpecStore>> {
        &self.applying
    }

    pub(crate) fn held_lock(&self) -> &Mutex<Option<Held>> {
        &self.held
    }
}

const PAST_BUILT: &str = "builds more than 80 bytes for each byte of the size bound (131072)";
const PAST_TEXT: &str =
    "reads as more than 2 bytes of text for each byte of the size bound (131072)";

fn applied(applied: u64, rejected: u64) -> ApplyCounts {
    ApplyCounts { applied, rejected }
}

/// What an enforce of `document` with no guard, as YAML, ran.
fn enforce(
    generations: &DocumentManager,
    document: impl Into<Vec<u8>>,
) -> Result<(u64, Digest), SpecRejected> {
    generations
        .enforce("", document.into(), Format::Yaml)
        .expect("no guard")
        .ran
}

/// A door with a spec store bound over a new, empty directory.
fn stored() -> (tempfile::TempDir, Arc<MemoryStore>, Arc<DocumentManager>) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let (store, generations) = generations();
    bind(&generations, dir.path());
    (dir, store, Arc::new(generations))
}

/// Binds the directory at `dir` as the door's source, as a start does.
fn bind(generations: &DocumentManager, dir: &Path) {
    let opened = SpecStore::open(dir.to_owned()).expect("the directory opens");
    generations
        .bind_store(opened, || false)
        .expect("the store binds");
}

/// `document` applied as YAML.
fn yaml(
    generations: &DocumentManager,
    document: impl Into<Vec<u8>>,
) -> Result<ApplyOutcome, SpecRejected> {
    generations.apply(document.into(), Format::Yaml)
}

fn generations() -> (Arc<MemoryStore>, DocumentManager) {
    let store = Arc::new(MemoryStore::new());
    (
        Arc::clone(&store),
        DocumentManager::new(store, stock(), Vec::new()),
    )
}

/// The door builds a generation with its document's lists: a document
/// of engine families alone, applied through it, lists the member that
/// joins after.
#[test]
fn a_document_applied_through_the_door_lists_the_member_that_joins() {
    use crate::store::{OutsideLists, WorkerFacts};

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

/// A generation whose lists differ from the standing one's has the standing
/// members read again under them at its apply. The count of members outside
/// the lists is the new generation's at once: no membership write and no
/// build come between the apply and the read.
#[test]
fn an_apply_of_other_lists_counts_the_standing_members_under_them() {
    use crate::store::{OutsideLists, WorkerFacts};

    let (store, generations) = generations();
    let listing = |engines: &str| {
        let document = format!("{RANDOM}view:\n  engines: [{engines}]\n");
        yaml(&generations, document).expect("a valid document");
        store.outside_lists().engines
    };
    assert_eq!(listing("alpha"), 0, "no member stands yet");
    for (key, engine) in [("http://a:8000", "alpha"), ("http://b:8000", "beta")] {
        store.upsert(WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation: 1,
            engine_kind: engine.to_owned(),
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
    }
    store.publish_now();
    let one = OutsideLists {
        engines: 1,
        zones: 0,
    };
    assert_eq!(store.outside_lists(), one);
    // The next list takes the second member in, and the one after leaves
    // both out.
    assert_eq!(listing("alpha, beta"), 0);
    assert_eq!(listing("gamma"), 2);
}

/// A member's reading under the lists is carried while its record stands
/// at the revision it was read at, and made again once a write moved it. A
/// member that registers again under a family and a zone the lists do not
/// name is counted outside both at the next build, and inside again under
/// listed ones.
#[test]
fn a_member_written_again_is_read_again_under_the_lists() {
    use crate::store::{OutsideLists, WorkerFacts};

    let (store, generations) = generations();
    let document = format!("{RANDOM}view:\n  engines: [alpha]\n  zones: [east]\n");
    yaml(&generations, document).expect("a valid document");
    let written_under = |family: &str, zone: &str| {
        store.upsert(WorkerFacts {
            key: "http://a:8000".to_owned(),
            engine: "http://a:8000".to_owned(),
            incarnation: 1,
            engine_kind: family.to_owned(),
            zone: Some(zone.to_owned()),
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
        store.publish_now();
        store.outside_lists()
    };
    let (inside, outside) = (
        OutsideLists::default(),
        OutsideLists {
            engines: 1,
            zones: 1,
        },
    );
    assert_eq!(written_under("alpha", "east"), inside);
    assert_eq!(written_under("beta", "west"), outside);
    assert_eq!(written_under("alpha", "east"), inside);
}

/// Lists of label selectors alone leave no member outside them, and the
/// store reads no member under them. The lists that follow count every
/// standing member all the same, at their apply and at the build behind a
/// join: a list of families counts by family, a list of zones by zone.
#[test]
fn lists_of_labels_alone_leave_no_member_outside_and_the_next_lists_count_every_member() {
    use crate::store::{OutsideLists, WorkerFacts};

    let (store, generations) = generations();
    let view = |lists: &str| {
        let document = format!("{RANDOM}view:\n  {lists}\n");
        yaml(&generations, document).expect("a valid document");
        store.outside_lists()
    };
    let join = |key: &str| {
        store.upsert(WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation: 1,
            engine_kind: "beta".to_owned(),
            zone: Some("west".to_owned()),
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
        store.publish_now();
        store.outside_lists()
    };
    let outside = |engines: u32, zones: u32| OutsideLists { engines, zones };
    assert_eq!(view("labels: [tier]"), outside(0, 0));
    assert_eq!(join("http://a:8000"), outside(0, 0));
    assert_eq!(join("http://b:8000"), outside(0, 0));
    // The store keeps no member's read under the selectors alone: the
    // lists pass ran at neither build.
    assert_eq!(store.listed_members(), 0);
    // A list of families counts both members at its apply, and the member
    // that joins behind it at its build.
    assert_eq!(view("engines: [alpha]"), outside(2, 0));
    // Under such a list the store keeps a read for each member: the
    // apply's two, then the build's third.
    assert_eq!(store.listed_members(), 2);
    assert_eq!(join("http://c:8000"), outside(3, 0));
    assert_eq!(store.listed_members(), 3);
    // The selectors alone again: none is outside, at the apply and at a
    // build.
    assert_eq!(view("labels: [tier]"), outside(0, 0));
    // The apply let go of the three reads and read no member.
    assert_eq!(store.listed_members(), 0);
    assert_eq!(join("http://d:8000"), outside(0, 0));
    // The build behind the join kept none either.
    assert_eq!(store.listed_members(), 0);
    // A list of zones counts the four at its apply and the fifth at its
    // build.
    assert_eq!(view("zones: [east]"), outside(0, 4));
    assert_eq!(join("http://e:8000"), outside(0, 5));
}

/// The writer that applies mints the number: a document with no line
/// starts as generation 1, and a changed one is the next.
#[test]
fn a_document_with_no_line_starts_as_generation_one_and_a_changed_one_is_the_next() {
    let (store, generations) = generations();
    assert!(!generations.bound());
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    assert!(generations.bound());
    assert_eq!(standing(&store).0, 1);
    assert_eq!(yaml(&generations, RANDOM), Ok(ApplyOutcome::Generation(2)));
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
    assert_eq!(first, Ok(ApplyOutcome::Generation(5)));
    let second = yaml(&generations, with_line(RANDOM, 9));
    assert_eq!(second, Ok(ApplyOutcome::Generation(9)));
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
    assert_eq!(again, Ok(ApplyOutcome::Generation(12)));
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
        Ok(ApplyOutcome::Generation(3))
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
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    let running = standing(&store);

    let json = br#"{"profiles":{"default":{"pick":"best","score":["least-load"],"filter":["healthy"]}},"apiVersion":"smg.policy/v1"}"#;
    let commented = format!("# the same document\n{LEAST_LOAD}");
    for (document, format) in [
        (json.to_vec(), Format::Json),
        (commented.into_bytes(), Format::Yaml),
        (with_line(LEAST_LOAD, 1), Format::Yaml),
    ] {
        let outcome = generations.apply(document, format);
        assert_eq!(outcome, Ok(ApplyOutcome::Unchanged(1)));
    }
    assert_eq!(standing(&store), running);
    let answer = generations.document(0).expect("the standing document");
    assert_eq!(answer.document, LEAST_LOAD.as_bytes());
}

/// The apply counter by result: a new generation counts as applied, a
/// refusal counts as rejected, once, and the standing document again
/// counts as neither.
#[test]
fn the_counts_follow_the_outcomes() {
    let (_store, generations) = generations();
    let counts = |applied, rejected| ApplyCounts { applied, rejected };
    assert_eq!(generations.apply_counts(), counts(0, 0));
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Unchanged(1))
    );
    assert_eq!(generations.apply_counts(), counts(1, 0));

    let refused = yaml(&generations, with_line(RANDOM, 1));
    assert!(matches!(
        refused,
        Err(SpecRejected::ChangedUnderItsLine { .. })
    ));
    assert_eq!(generations.apply_counts(), counts(1, 1));
    assert_eq!(yaml(&generations, RANDOM), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(generations.apply_counts(), counts(2, 1));
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

    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Unchanged(5))
    );
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
    assert_eq!(yaml(&generations, RANDOM), Ok(ApplyOutcome::Unchanged(6)));
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
        Ok(ApplyOutcome::Generation(u64::MAX))
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
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
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
    let generations =
        DocumentManager::validating(Arc::clone(&store), stock(), Vec::new(), faulting);
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
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
        Ok(ApplyOutcome::Generation(4))
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
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let random = document
            .profiles
            .values()
            .any(|profile| profile.pick == "random".into());
        if random {
            panic::panic_any(Bomb);
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let store = Arc::new(MemoryStore::new());
    let generations =
        DocumentManager::validating(Arc::clone(&store), stock(), Vec::new(), dropping);
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );

    assert_eq!(
        yaml(&generations, RANDOM),
        Err(SpecRejected::Faulted("a panic with no message".to_owned()))
    );
    assert_eq!(standing(&store).0, 1);
    assert_eq!(
        yaml(&generations, with_line(LEAST_LOAD, 4)),
        Ok(ApplyOutcome::Generation(4))
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
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        if document.generation == 2 {
            ENTERED.store(true, Ordering::SeqCst);
            until(&RELEASED, "the validator was never released");
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }

    let store = Arc::new(MemoryStore::new());
    let generations = Arc::new(DocumentManager::validating(
        store,
        stock(),
        Vec::new(),
        stalling,
    ));
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
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
        Ok(ApplyOutcome::Generation(2))
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
        door: Arc<OnceLock<Weak<DocumentManager>>>,
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
    let mut stages = StageRegistry::builtin();
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
    let generations = Arc::new(DocumentManager::new(store, Arc::new(stages), Vec::new()));
    door.set(Arc::downgrade(&generations)).expect("set once");

    // The store keeps the generation before the standing one: the third
    // apply lets the first go.
    for (number, pick) in (1..).zip(["random", "round-robin", "best"]) {
        let document = format!(
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [let-go]\n    pick: {pick}\n"
        );
        assert_eq!(
            yaml(&generations, document),
            Ok(ApplyOutcome::Generation(number))
        );
    }
    assert_eq!(*found.lock(), [true], "the first generation's end");
}

/// The validator's word comes before the digest's. A `kind` line is in
/// no digest and one with a value is refused: the standing document with
/// `kind: PolicySpec` added is refused by name, under no line and under
/// the standing one, and where no bytes are held yet these are not kept.
#[test]
fn a_document_the_validator_refuses_is_refused_though_its_digest_stands() {
    let kind = || SpecRejected::Invalid(SpecError::Kind("PolicySpec".to_owned()));
    let with_kind = format!("kind: PolicySpec\n{LEAST_LOAD}");

    {
        let (store, generations) = generations();
        assert_eq!(
            yaml(&generations, LEAST_LOAD),
            Ok(ApplyOutcome::Generation(1))
        );
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
    assert_eq!(
        yaml(&generations, LEAST_LOAD),
        Ok(ApplyOutcome::Generation(1))
    );
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
                Ok(ApplyOutcome::Generation(1)),
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
    let (path, message) = unparsed(LEAST_LOAD.replace("    pick: best\n", "    colour: blue\n"));
    assert_eq!(path, "profiles.default.colour");
    assert!(message.contains("colour"), "{message}");
    let (path, _) = unparsed(LEAST_LOAD.replace("pick: best", "pick: 1"));
    assert_eq!(path, "profiles.default.pick");
    assert_eq!(
        yaml(
            &generations,
            LEAST_LOAD.replace("[healthy]", "[\"healthy\"]")
        ),
        Ok(ApplyOutcome::Generation(1))
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
    // No values, and a key written twice: the refusal is the read of
    // the values' own, about the document.
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

/// Scores every worker alike, by state the lifecycle's calls keep.
#[derive(Debug)]
struct Kept;

impl crate::stages::Lifecycle for Kept {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl crate::engine::Scorer for Kept {
    fn name(&self) -> &'static str {
        "kept"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> crate::engine::TieBand {
        out.fill(1.0);
        crate::engine::TieBand::NORMALIZED
    }
}

/// A dry run answers a stage in need of the lifecycle's calls as an
/// apply does: refused by the stage's field, as not in this build.
#[test]
fn a_dry_run_refuses_a_stage_in_need_of_the_lifecycle_as_an_apply_does() {
    let mut stages = StageRegistry::builtin();
    stages
        .scorer("kept", || Box::new(Kept))
        .expect("a free name");
    let store = Arc::new(MemoryStore::new());
    let generations = DocumentManager::new(store, Arc::new(stages.without_lifecycle()), Vec::new());
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
        Ok(ApplyOutcome::Generation(5))
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
        let before = generations.apply_counts();
        let refused = generations
            .dry_run("", &document, Format::Yaml)
            .expect("no guard")
            .would_run
            .expect_err("refused");
        assert_eq!(refused.field_path(), path, "{refused}");
        assert_eq!(
            generations.apply_counts(),
            before,
            "a dry run counts nothing"
        );
        assert_eq!(yaml(&generations, document), Err(refused));
        assert_eq!(generations.apply_counts(), applied(1, before.rejected + 1));
    }
    let before = generations.apply_counts();
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
        generations.apply_counts(),
        before,
        "a dry run counts nothing"
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
        Ok(ApplyOutcome::Generation(1))
    );
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
                        Ok(ApplyOutcome::Generation(generation)) => generation,
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
    assert_eq!(generations.apply_counts(), applied(0, 1));
}

#[test]
fn publication_runs_once_for_a_change_and_skips_the_standing_document() {
    use std::cell::Cell;

    let (store, door) = generations();
    let calls = Cell::new(0);
    let apply = || {
        door.apply_where(
            LEAST_LOAD.into(),
            Format::Yaml,
            |_| None,
            |published, spec| {
                calls.set(calls.get() + 1);
                assert!(std::ptr::eq(published, store.as_ref()));
                assert_eq!(spec.generation(), 1);
                published.apply(spec).map(|()| true)
            },
        )
    };
    assert_eq!(apply(), Ok(ApplyOutcome::Generation(1)));
    assert_eq!(apply(), Ok(ApplyOutcome::Unchanged(1)));
    assert_eq!(calls.get(), 1);
}

#[test]
fn a_publication_that_waits_keeps_the_document_counts_and_log_as_they_were() {
    let (store, door) = generations();
    yaml(&door, LEAST_LOAD).expect("the first document runs");
    let stood = door.document(0).expect("the standing document");
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let waited = door.apply_where(
        RANDOM.into(),
        Format::Yaml,
        |_| None,
        |_, spec| {
            assert_eq!(spec.generation(), 2);
            Ok(false)
        },
    );
    assert_eq!(waited, Ok(ApplyOutcome::Waits(1)));
    assert_eq!(standing(&store), (1, Some(stood.digest)));
    assert_eq!(door.document(0), Ok(stood));
    assert_eq!(door.apply_counts(), applied(1, 0));
    assert!(lines.0.lock().is_empty());
}

/// What the door promises of an apply's two notices. The pass is told
/// behind the first admission check and ahead of the second, before the
/// apply publishes: the apply holds the apply lock then, and not the
/// document lock. The standing number is told behind the second check
/// and behind the publication, while the apply holds both locks: the
/// notice is handed the published number and the document's digest. The
/// standing document applied again publishes nothing, and is told the
/// same way with the number the apply read at its start. A document that
/// does not pass is told as no pass and has no standing number. A
/// publication that does not land has none either. An admission check
/// that answers ends the apply where it stands: the first ahead of both
/// notices, the second behind the pass's alone.
#[test]
fn an_applys_two_notices_run_under_the_doors_locks_in_their_order() {
    use std::cell::{Cell, RefCell};

    let (store, door) = generations();
    let admissions = Cell::new(0);
    let lands = Cell::new(true);
    let told = RefCell::new(Vec::new());
    // What a notice finds where it runs: whether the apply lock and the
    // document lock are held, the admission checks asked so far, and the
    // store's number. A standing notice adds the number it was handed,
    // and whether the digest it was handed is the store's.
    let found = |notice, handed: Option<(u64, Digest)>| {
        let locks = (
            door.applying.try_lock().is_none(),
            door.held.try_lock().is_none(),
        );
        let (number, stands) = standing(&store);
        let handed = handed.map(|(handed, digest)| (handed, Some(digest) == stands));
        told.borrow_mut()
            .push((notice, locks, admissions.get(), number, handed));
    };
    // An apply whose admission check answers at its `waits`-th asking,
    // and admits at every asking where that is 0. Its publication lands
    // where `lands` says so.
    let apply = |document: &str, waits: u32| {
        admissions.set(0);
        door.apply_notifying(
            document.into(),
            Format::Yaml,
            |number| {
                admissions.set(admissions.get() + 1);
                (admissions.get() == waits).then_some(ApplyOutcome::Waits(number))
            },
            |published, spec| {
                if lands.get() {
                    publish_local(published, spec)
                } else {
                    Ok(false)
                }
            },
            |passed| found(if passed.is_some() { "pass" } else { "no pass" }, None),
            |number, digest| found("standing", Some((number, digest))),
        )
    };

    assert_eq!(apply(LEAST_LOAD, 0), Ok(ApplyOutcome::Generation(1)));
    let first = [
        ("pass", (true, false), 1, 0, None),
        ("standing", (true, true), 2, 1, Some((1, true))),
    ];
    assert_eq!(told.take(), first);
    assert_eq!(apply(LEAST_LOAD, 0), Ok(ApplyOutcome::Unchanged(1)));
    let again = [
        ("pass", (true, false), 1, 1, None),
        ("standing", (true, true), 2, 1, Some((1, true))),
    ];
    assert_eq!(told.take(), again);

    assert!(apply("{ no policy", 0).is_err());
    assert_eq!(told.take(), [("no pass", (true, false), 1, 1, None)]);
    assert_eq!(apply(RANDOM, 1), Ok(ApplyOutcome::Waits(1)));
    assert!(told.take().is_empty(), "the first check answered");
    assert_eq!(apply(RANDOM, 2), Ok(ApplyOutcome::Waits(1)));
    assert_eq!(told.take(), [("pass", (true, false), 1, 1, None)]);
    lands.set(false);
    assert_eq!(apply(RANDOM, 0), Ok(ApplyOutcome::Waits(1)));
    assert_eq!(told.take(), [("pass", (true, false), 1, 1, None)]);
    assert_eq!(
        standing(&store).0,
        1,
        "none of the last three applies entered"
    );
    assert!(
        door.applying.try_lock().is_some(),
        "the apply lock is let go"
    );
}

/// A followed generation enters through the turn that holds the apply
/// lock from ahead of its pass. Its entry publishes the generation and
/// then tells its number once, while the turn holds the apply lock and
/// the entry the document lock. The turn's end lets the apply lock go.
#[cfg(feature = "controller")]
#[test]
fn a_followed_generations_entry_tells_its_number_under_both_locks() {
    use std::cell::RefCell;

    let (store, door) = generations();
    let turn = door.follower_turn();
    let checked = door.pass((0, None), RANDOM.as_bytes(), Format::Yaml, Line::Record(3));
    let Checked { digest, next, .. } = checked.expect("the document passes");
    let kept = Held {
        generation: 3,
        digest,
        document: RANDOM.into(),
        format: Format::Yaml,
        applied: SystemTime::now(),
    };
    let told = RefCell::new(Vec::new());
    let entered = turn.enter(None, next.expect("above none"), kept, |number, stands| {
        let locks = (
            door.applying.try_lock().is_none(),
            door.held.try_lock().is_none(),
        );
        let seen = (number, stands == digest, locks, standing(&store).0);
        told.borrow_mut().push(seen);
    });
    entered.expect("a number above the standing one enters");
    assert_eq!(told.take(), [(3, true, (true, true), 3)]);
    assert!(door.applying.try_lock().is_some(), "the turn is over");
    assert_eq!(door.document(0).expect("held").document, RANDOM.as_bytes());
}

#[test]
fn the_same_document_is_refused_only_by_a_host_that_denies_its_header() {
    let document = format!("{LEAST_LOAD}view:\n  headers: [x-private]\n");
    let make = |denied_headers| {
        DocumentManager::new(Arc::new(MemoryStore::new()), stock(), denied_headers)
    };
    let denied = make(vec!["x-private".to_owned()]);
    let refused = denied
        .dry_run("", document.as_bytes(), Format::Yaml)
        .expect("no guard")
        .would_run
        .expect_err("the host denies the header");
    assert!(matches!(refused, SpecRejected::Invalid(_)), "{refused}");
    assert!(refused.to_string().contains("x-private"), "{refused}");
    assert_eq!(yaml(&denied, document.as_str()), Err(refused));
    assert_eq!(denied.standing(), (0, None));
    assert_eq!(
        yaml(&make(Vec::new()), document),
        Ok(ApplyOutcome::Generation(1))
    );
}

/// A dry run against a standing generation that is handed in answers as
/// the door's own does where no spec store is bound. The documents: the
/// standing one again with and without its line, a changed one under no
/// line, under the standing line, under a lower one and under a higher
/// one, one the validator refuses, one that reads a header its host
/// denies, one that is no document and one over the bound. Each is asked
/// with no guard, with the standing digest and with another one.
#[test]
fn a_dry_run_against_a_handed_generation_answers_as_a_door_with_no_spec_store() {
    let store = Arc::new(MemoryStore::new());
    let stages = stock();
    let door = DocumentManager::new(
        Arc::clone(&store),
        Arc::clone(&stages),
        vec!["x-private".to_owned()],
    );
    assert_eq!(
        yaml(&door, with_line(LEAST_LOAD, 5)),
        Ok(ApplyOutcome::Generation(5))
    );
    let world = store.snapshot();
    let standing = world.spec();
    let digest = standing.digest().expect("a document stands").to_string();
    let another = format!("sha256:{}", "0".repeat(64));
    let handed = |expected: &str, document: &[u8]| {
        let denied = ["x-private"];
        dry_run_against(standing, &stages, &denied, expected, document, Format::Yaml)
    };
    let over = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits") + 1;
    let documents = [
        LEAST_LOAD.as_bytes().to_vec(),
        with_line(LEAST_LOAD, 5),
        RANDOM.as_bytes().to_vec(),
        with_line(RANDOM, 5),
        with_line(RANDOM, 3),
        with_line(RANDOM, 9),
        LEAST_LOAD.replace("[healthy]", "[nowhere]").into_bytes(),
        format!("{RANDOM}view:\n  headers: [x-private]\n").into_bytes(),
        b"profiles: [".to_vec(),
        vec![b'#'; over],
    ];
    for document in &documents {
        for expected in ["", digest.as_str(), another.as_str()] {
            let by_door = door.dry_run(expected, document, Format::Yaml);
            assert_eq!(
                format!("{:?}", handed(expected, document)),
                format!("{by_door:?}"),
                "guard {expected:?}"
            );
        }
    }

    // The answers are a dry run's own, not one refusal told ten times.
    let again = handed(&digest, LEAST_LOAD.as_bytes()).expect("the standing digest");
    assert_eq!(again.would_run.map(|(number, _)| number), Ok(5));
    let next = handed("", RANDOM.as_bytes()).expect("no guard");
    assert_eq!(next.would_run.map(|(number, _)| number), Ok(6));
    assert!(
        next.needs.is_some(),
        "a document that would run tells its needs"
    );
    let claimed = handed("", &with_line(RANDOM, 9)).expect("no guard");
    assert_eq!(claimed.would_run.map(|(number, _)| number), Ok(9));
    let guarded = handed(&another, RANDOM.as_bytes());
    assert!(matches!(guarded, Err(EnforceRefused::Guarded(5, Some(_)))));
    assert_eq!(
        door.apply_counts(),
        applied(1, 0),
        "a dry run counts nothing"
    );
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
            assert_eq!(applied, Ok(ApplyOutcome::Generation(1)));
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
    // read as one that runs is: into values, and by the typed read of
    // those. The time to that refusal, by its size.
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

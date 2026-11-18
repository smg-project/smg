//! The store lends a publication's rows. A decision over lent rows, with the
//! mask and the pin beside them, answers as the same decision over the rows
//! written for it: every preset of the build over five fleets and eight
//! sizes, on both sides of the size a draw starts at, and each built-in of
//! the catalog alone in a profile. And a decision that
//! straddles a write frees nothing on its thread. A separate executable, so
//! it can declare the allocator that counts the frees.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision freed"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    time::{Duration, Instant},
};

use smg_policy_controller::{
    catalog::Catalog,
    spec::{BindingSpec, OverloadSpec},
    store::{
        Ask, GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts,
        WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    AppliedSpec, Constraints, Engine, LoadRow, PolicySpec, RequestFacts, Role, Scratch,
    SignalSource, SpecError, WorkerId, API_VERSION, PRESETS,
};

thread_local! {
    /// Frees this thread made. A constant without a destructor, so the
    /// allocator reaches it without one.
    static FREES: Cell<usize> = const { Cell::new(0) };
}

/// Frees the calling thread has made so far.
fn frees() -> usize {
    FREES.with(Cell::get)
}

struct CountingFrees;

unsafe impl GlobalAlloc for CountingFrees {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the same layout contract as the caller's, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.with(|frees| frees.set(frees.get() + 1));
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingFrees = CountingFrees;

const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);

fn member_facts(key: String) -> WorkerFacts {
    WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// A store of `n` members serving the model, each with a record on file,
/// published: every fifth not ready where `refused`; three vetoes of the
/// asking gateway and one of another where `vetoed`; the reports sampled
/// up to a minute apart where `aged`, so that a report's age differs row by
/// row at any decision. The members in the order they registered and the
/// instant the last window opened.
fn store_of(
    n: usize,
    refused: bool,
    vetoed: bool,
    aged: bool,
) -> (MemoryStore, Vec<WorkerId>, Instant) {
    let store = MemoryStore::new();
    let opened = Instant::now();
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let id = store
                .upsert(member_facts(format!("http://w{index:04}:8000")))
                .id();
            if !(refused && index % 5 == 2) {
                let revision = WorkersStore::get(&store, id).expect("the row").revision;
                store
                    .set_status(id, WorkerStatus::Ready, revision)
                    .expect("ready");
            }
            let x = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let row = LoadRow {
                running: (x >> 16) as u32 % 12,
                queued: (x >> 24) as u32 % 4,
                live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
                report_age_ms: 5,
                source: SignalSource::Events,
                ..LoadRow::default()
            };
            let sampled = if aged { index as u64 % 7 * 9 } else { 0 };
            let at = opened + Duration::from_secs(sampled);
            store.record(id, 0, row, at, LoadSource::Stream);
            id
        })
        .collect();
    if vetoed {
        let until = Some(opened + Duration::from_secs(3_600));
        for row in [1usize, 3, 6] {
            if let Some(&id) = ids.get(row) {
                store.gateway_veto(ASKER, id, until);
            }
        }
        // Another gateway's veto masks nothing for the asking one.
        store.gateway_veto(GatewayId(2), ids[0], until);
    }
    let opened = opened + Duration::from_secs(if aged { 64 } else { 0 }) + PUBLISH_WINDOW;
    store.publish(opened);
    (store, ids, opened)
}

fn engine_of(preset: &str) -> Engine {
    PolicySpec::preset(preset)
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates")
}

/// An engine of one profile that holds the built-in `name` and nothing a
/// document can leave out: the stage by its name, or by its name and the one
/// parameter the spec has no default for. A built-in that gains such a
/// parameter is refused here by its path, and takes its line.
fn alone(kind: &str, name: &str) -> Result<Engine, SpecError> {
    let stage = match name {
        "resident-work" => r#"{"resident-work":{"request_tokens":2048}}"#.to_string(),
        "best-with-temperature" => r#"{"best-with-temperature":{"temperature":0.5}}"#.to_string(),
        name => format!(r#""{name}""#),
    };
    let validated = |profile: String| {
        let text = format!(r#"{{"apiVersion":"{API_VERSION}","profiles":{{"p":{profile}}}}}"#);
        let document: PolicySpec = serde_json::from_str(&text).expect("a well-formed document");
        document.validate()
    };
    match kind {
        "filter" => validated(format!(r#"{{"filter":[{stage}],"pick":"best"}}"#)),
        "score" => validated(format!(r#"{{"score":[{stage}],"pick":"best"}}"#)),
        // A picker that compares scores is refused alone: it takes the
        // plainest scorer.
        _ => match validated(format!(r#"{{"pick":{stage}}}"#)) {
            Err(SpecError::PickerNeedsScorer { .. }) => {
                validated(format!(r#"{{"score":["least-requests"],"pick":{stage}}}"#))
            }
            built => built,
        },
    }
}

/// What the decisions of a case read.
#[derive(Default)]
struct Read {
    decisions: u64,
    picks: u64,
    masked: u64,
    pinned: u64,
}

impl Read {
    /// The case saw picks, masked rows and pins: it did not pass on sheds.
    fn check(&self) {
        let Self {
            decisions,
            picks,
            masked,
            pinned,
        } = self;
        assert!(
            *picks > 0 && *masked > 0 && *pinned > 0,
            "{decisions} decisions, {picks} picks, {masked} masked rows read, {pinned} pins read"
        );
    }
}

/// Forty decisions over one store of `n` members, each made twice: over the
/// rows written for it, and over the publication's rows, lent, with the
/// mask and the pin beside them. Two engines of one build, each keeping its
/// own state (a rotation, a tally), are fed the same decisions in the same
/// order and answer the same. The fleet is [`store_of`]'s three words and
/// whether a member is pinned.
fn lent_agrees(
    make: &dyn Fn() -> Engine,
    (refused, vetoed, aged, pin): (bool, bool, bool, bool),
    n: usize,
    label: &str,
    read: &mut Read,
) {
    let (store, ids, opened) = store_of(n, refused, vetoed, aged);
    let (written, lent) = (make(), make());
    let ask = Ask {
        gateway: ASKER,
        now: opened + Duration::from_secs(1),
        pinned: pin.then(|| ids[n / 2]),
        affinity: written.needs().affinity,
    };
    let (mut copy, mut out) = (ViewBuffers::default(), ViewBuffers::default());
    let (mut one, mut two) = (Scratch::new(), Scratch::new());
    for seed in 1..=40u64 {
        let built = store
            .build_view(MODEL, Role::Single, &ask, &mut copy)
            .expect("the pool");
        let facts = RequestFacts {
            model: MODEL,
            prompt_tokens: 512,
            seed,
            now_micros: built.now_micros,
            affinity: &copy.affinity,
            ..RequestFacts::default()
        };
        let over_written = written.decide(
            &facts,
            &built.pool_facts(),
            &copy.candidates,
            &Constraints::default(),
            &mut one,
        );
        let over_lent = store.with_world(|world| {
            let view = world
                .view(MODEL, Role::Single, &ask, false, &mut out)
                .expect("the pool");
            read.masked += view.stalled.len() as u64;
            read.pinned += u64::from(view.pinned_at != 0);
            let facts = RequestFacts {
                pinned_at: view.pinned_at,
                now_micros: view.built.now_micros,
                affinity: view.affinity,
                ..facts
            };
            let cx = Constraints {
                stalled: view.stalled,
                ..Constraints::default()
            };
            let pool = view.built.pool_facts();
            lent.decide(&facts, &pool, view.candidates, &cx, &mut two)
        });
        assert_eq!(
            format!("{over_written:?}"),
            format!("{over_lent:?}"),
            "{label} refused={refused} vetoed={vetoed} aged={aged} pin={pin} n={n} seed={seed}"
        );
        read.decisions += 1;
        read.picks += u64::from(over_written.pick().is_some());
    }
}

#[test]
fn a_decision_over_lent_rows_answers_as_the_one_over_written_rows() {
    let mut read = Read::default();
    for preset in PRESETS {
        for fleet in [
            (false, false, false, false),
            (true, false, false, false),
            (false, true, false, false),
            (true, true, true, true),
            (false, false, false, true),
        ] {
            for n in [1usize, 2, 8, 13, 16, 33, 64, 200] {
                lent_agrees(&|| engine_of(preset), fleet, n, preset, &mut read);
            }
        }
    }
    read.check();
}

#[test]
fn each_built_in_alone_answers_over_lent_rows_as_over_written_rows() {
    // The catalog's own lists: a built-in added to the build is held here
    // with no line of this case moved. Each stands alone in a profile,
    // under a standing veto, a pin and reports of different ages. A
    // built-in that read the stalled bit, the session bit or the report's
    // age off the row, and not what stands beside a lent row, would answer
    // two ways.
    let catalog = Catalog::builtin();
    let filters = catalog.filters().map(|name| ("filter", name));
    let scorers = catalog.scorers().map(|name| ("score", name));
    let pickers = catalog.pickers().map(|name| ("pick", name));
    let (mut read, mut walked, mut absent) = (Read::default(), Vec::new(), Vec::new());
    for (kind, name) in filters.chain(scorers).chain(pickers) {
        let label = format!("{kind} {name} alone");
        let built = alone(kind, name);
        // A name the spec knows and this build does not hold has no stage
        // to run: the day the build holds it, it is walked.
        if matches!(built, Err(SpecError::NotInBuild { .. })) {
            absent.push(name);
            continue;
        }
        assert!(built.is_ok(), "{label}: {:?}", built.err());
        for n in [8usize, 16, 64] {
            let make = || alone(kind, name).expect("it validated");
            lent_agrees(&make, (true, true, true, true), n, &label, &mut read);
        }
        walked.push(name);
    }
    assert!(
        walked.len() >= 14 && walked.len() > absent.len(),
        "walked {walked:?}, not in this build {absent:?}"
    );
    read.check();
}

/// A document that validates, under the generation `number`.
fn generation(number: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    AppliedSpec::validate(&document).expect("a preset validates")
}

#[test]
fn a_decision_that_straddles_a_write_frees_nothing() {
    let (store, ids, opened) = store_of(64, false, true, false);
    let ask = Ask {
        gateway: ASKER,
        now: opened + Duration::from_secs(1),
        pinned: None,
        affinity: false,
    };
    let until = Some(opened + Duration::from_secs(7_200));
    let write = |kind: usize, turn: u32| match kind {
        // A record past the open window: it publishes at its own instant,
        // and the end of the window it opens has nothing to publish.
        0 => {
            let at = opened + PUBLISH_WINDOW * (2 * turn + 2);
            let row = LoadRow {
                running: turn,
                ..LoadRow::default()
            };
            store.record(ids[0], 0, row, at, LoadSource::Stream);
            store.publish(at + PUBLISH_WINDOW);
        }
        // A veto of the asking gateway.
        1 => store.gateway_veto(ASKER, ids[10 + turn as usize], until),
        // A membership write, and the build it owes.
        2 => {
            store.upsert(member_facts(format!("http://new{turn}:8000")));
            store.publish_now();
        }
        // A generation applied.
        3 => store
            .apply(generation(u64::from(turn) + 1))
            .expect("above the standing one"),
        // A record inside the open window: it waits, and the window's end
        // publishes it and opens the next. The last record of kind 0 opened
        // the standing window, six windows past `opened`.
        4 => {
            let end = opened + PUBLISH_WINDOW * (turn - 5);
            let row = LoadRow {
                running: turn,
                ..LoadRow::default()
            };
            store.record(ids[0], 0, row, end - PUBLISH_WINDOW / 2, LoadSource::Stream);
            assert_eq!(store.publish_due(), Some(end), "the record waits");
            store.publish(end);
        }
        // A leave: its rows are folded as gone at its answer.
        5 => {
            let id = ids[40 + turn as usize];
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store.remove(id, revision).expect("removed");
        }
        // A new incarnation: the old one's rows are folded as gone at its
        // answer.
        _ => {
            let key = format!("http://w{:04}:8000", 20 + turn);
            store.upsert(WorkerFacts {
                incarnation: 2,
                ..member_facts(key)
            });
        }
    };
    let mut out = ViewBuffers::default();
    for kind in 0..7usize {
        for turn in 0..3u32 {
            // The decision holds its load across the write: it is the last
            // reader of the world the write replaces.
            let freed = store.with_world(|world| {
                let view = world
                    .view(MODEL, Role::Single, &ask, false, &mut out)
                    .expect("the pool");
                let rows = view.candidates.len();
                write(kind, kind as u32 * 3 + turn);
                let standing = store.world();
                assert!(!std::ptr::eq(world, &*standing), "write {kind} published");
                drop(standing);
                assert_eq!(view.candidates.len(), rows, "the rows it holds stand");
                frees()
            });
            assert_eq!(
                frees() - freed,
                0,
                "a decision that straddled write {kind} freed on its thread"
            );
        }
    }
}

/// A generation whose one binding judges every pool by its token usage.
fn generation_over(number: u64, token_usage: f32) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile")
        .clone();
    document.bindings = vec![BindingSpec {
        profile: Some(profile),
        overload: Some(OverloadSpec {
            token_usage: Some(token_usage),
            ..OverloadSpec::default()
        }),
        ..BindingSpec::default()
    }];
    AppliedSpec::validate(&document).expect("the document validates")
}

#[test]
fn a_period_told_before_any_window_shows_its_judgment_and_frees_nothing() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    store.apply(generation_over(1, 0.6)).expect("applied");
    // Two ranks of one engine register before they name a model: their
    // records go on file, no row takes them and no window opens.
    let rank = |rank: u16, models: Vec<String>| {
        store
            .upsert(WorkerFacts {
                key: format!("grpc://e:9000#{rank}"),
                engine: "grpc://e:9000".to_string(),
                incarnation: 1,
                models,
                dp_rank: rank,
                dp_size: 2,
                ..WorkerFacts::default()
            })
            .id()
    };
    let (cool, hot) = (rank(0, Vec::new()), rank(1, Vec::new()));
    let usage = |active_kv_permille: u16| LoadRow {
        active_kv_permille,
        ..LoadRow::default()
    };
    store.record(cool, 0, usage(100), t0, LoadSource::Stream);
    let late = t0 + Duration::from_secs(1_000);
    store.record(hot, 1, usage(900), late, LoadSource::Stream);
    // The model is named: the rows take what is on file.
    let named = (
        rank(0, vec![MODEL.to_string()]),
        rank(1, vec![MODEL.to_string()]),
    );
    assert_eq!(named, (cool, hot));
    // The build those registrations left owed, asked: no window opens.
    store.publish_now();
    let ask = Ask {
        gateway: ASKER,
        now: late,
        pinned: None,
        affinity: false,
    };
    let (mut out, mut after) = (ViewBuffers::default(), ViewBuffers::default());
    let freed = store.with_world(|world| {
        // Under no period the cool rank's old sample counts: the mean of
        // the two is under the threshold.
        let view = world
            .view(MODEL, Role::Single, &ask, false, &mut out)
            .expect("the pool");
        assert_eq!(view.candidates.len(), 2);
        assert!(view.candidates.iter().all(|row| !row.health.overloaded));
        // The telling ends that sample and the hot rank is judged alone.
        // No window ever opened, so the judgment is shown at once.
        assert!(store.heard_every(Duration::from_secs(10)));
        let standing = store.world();
        assert!(!std::ptr::eq(world, &*standing), "the telling published");
        let judged = standing
            .view(MODEL, Role::Single, &ask, false, &mut after)
            .expect("the pool");
        assert!(judged.candidates.iter().all(|row| row.health.overloaded));
        drop(standing);
        frees()
    });
    assert_eq!(
        frees() - freed,
        0,
        "a decision that straddled the telling freed on its thread"
    );
}

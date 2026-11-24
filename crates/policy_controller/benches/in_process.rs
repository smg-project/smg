//! The in-process engine over a host's worker objects, cell by cell.
//!
//! The engine of a gateway's own process ([`PolicyEngine`]) decides over
//! the gateway's lists of worker objects and its load monitor's snapshot.
//! A cell is one such list under one document, with the request its callers
//! make: one entry of [`CELLS`]. A caller asks the one entry a host asks
//! for a request ([`PolicyEngine::decide_at`]) and says what the host then
//! owes the engine of it. A place answered alone owes nothing, and neither
//! does a decision that is not leased. A leased one is held by a guard,
//! which is told the dispatch and the end ([`DecisionGuard`]).
//!
//! The request is made and the clock read where the engine asks for them.
//! The load snapshot is the engine's own: the host hands its load monitor's
//! channel once and spawns the engine's task ([`PolicyEngine::run`]), which
//! keeps the newest snapshot at hand under a document that reads load
//! reports. No other work of a host is done: no route, no metric, no
//! in-flight count.
//!
//! ```text
//! cargo bench -p smg-policy-controller --features in-process --bench in_process -- \
//!     --cell power_of_two --workers 64 --callers 1 --decisions 1000000
//! ```
//!
//! `--workers` is the list's size: every worker is ready and has a load
//! report. `--publish-ms N` publishes the reports again, as a new snapshot,
//! every `N` milliseconds while the timed decisions run; `0`, the default,
//! publishes once. `--overloaded 1` puts every worker over its
//! overload threshold, so the list admits no worker, and gives each worker
//! requests in flight: its place plus one, the middle worker none, so the
//! rule for such a list has loads to steer by. `--refused limit` puts every
//! prefill of a pair's cell at its host's admission limit, and `--refused
//! partners` takes its every decode out of service: each decision is then
//! refused, and a turn refused that way counts. `--shadow twin` names a
//! copy of the cell's profile as its binding's shadow, so each decision is
//! followed by its shadow profile's. `--shadow other` names that shadow at
//! a binding of another model: the cell's decisions then have no shadow,
//! under a document that names one. `--callers` threads each
//! make `--decisions` timed decisions behind untimed first ones. The result
//! is one line.
//!
//! A time needs a quiet machine. A count of instructions does not: run a
//! cell twice under callgrind with one caller and two values of
//! `--decisions`, and divide the difference of the counts by the difference
//! of the decisions. The setup and the untimed decisions are in both runs.
//! Count with `--upkeep-thread 0`: the engine's task then takes its first
//! step and no thread of its own turns it. With that thread, the counts of
//! one binary differ from run to run by the allocator's locks.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "a bench binary: it stops on a broken setup and prints its result"
)]

#[path = "in_process/conversations.rs"]
mod conversations;
#[path = "../tests/common/counting_alloc.rs"]
mod counting_alloc;
#[path = "in_process/encode.rs"]
mod encode;
#[path = "in_process/pair.rs"]
mod pair;
#[path = "in_process/place.rs"]
mod place;
#[path = "in_process/ruled.rs"]
mod ruled;
#[path = "in_process/signals.rs"]
mod signals;
#[path = "in_process/worded.rs"]
mod worded;

#[cfg(any(target_env = "msvc", target_env = "musl"))]
use std::alloc::System as Allocator;
use std::{
    cell::OnceCell,
    collections::HashMap,
    env,
    future::pending,
    hint::black_box,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use counting_alloc::{calls, Counting};
use parking_lot::{Condvar, Mutex};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec, RulePlaced, ShadowSpec},
    engine::{
        Constraints, Decided, Identity, LentList, ListCell, Phase, Plain, PolicyEngine, Request,
        RequestFacts, RequestMaker, TokenEntry, TokenKey, WorkerId, WorkerLists,
    },
    policies::consistent_hashing::{Head, LentRing},
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, TestFeed},
    test_support::ready_worker,
    worker::WorkerLoadResponse,
};
// The allocator a gateway's binary registers, on the targets it registers
// it, with its calls counted: what a decision costs depends on it.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
use tikv_jemallocator::Jemalloc as Allocator;
use tokio::{
    runtime::{Builder, Runtime},
    time::sleep,
};

#[global_allocator]
static GLOBAL: Counting<Allocator> = Counting(Allocator);

/// A gateway's settings of that allocator, in the symbol it reads as it
/// starts: a copy of the controller binary's line, held to it by a case
/// (`tests/binary.rs`). An Apple target keeps the allocator's own settings.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[expect(unsafe_code, reason = "the allocator reads this symbol")]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

const MODEL: &str = "bench-model";

/// Decisions a caller makes untimed first: the buffers a thread keeps.
const CALLER_WARM: u64 = 256;

/// One caller's turns as a cell makes them whose callers make their own:
/// the host, the start every caller waits at, the caller's timed decisions
/// and its number in, and what [`stretch`] says of the turns out.
type Turns = fn(&Host, &Barrier, u64, u64) -> (u64, u64, u64);

/// A cell, by its parts in the order of their names. An entry of [`CELLS`]
/// names its own parts over [`Cell::PLAIN`], whose parts do nothing: a new
/// part is one field here and one line there, and no other cell's entry
/// changes.
#[derive(Clone, Copy)]
struct Cell {
    /// What the cell makes once behind its engine.
    behind: fn(&PolicyEngine),
    /// One caller's turns, where the cell's callers make their own in the
    /// place of the run's `decide`.
    caller: Option<Turns>,
    /// The cell's document.
    document: fn() -> PolicySpec,
    /// What the cell asks of a run ahead of its callers: it stops there a
    /// run its callers could not make.
    fits: fn(&Run),
    /// The cell's name, as `--cell` states it.
    name: &'static str,
    /// The request of one turn of the cell's callers by the turn's seed. It
    /// is plain for every seed or for none: the bench asks it once a cell.
    request: fn(u64) -> Request<'static>,
    /// The words the cell adds at the end of its result line.
    words: fn() -> String,
}

impl Cell {
    /// A cell of plain requests that makes nothing behind its engine, whose
    /// callers ask the run's `decide`, which takes every run and whose line
    /// gains no word. An entry names itself and its document.
    const PLAIN: Self = Self {
        behind: |_| (),
        caller: None,
        document: || preset("round_robin"),
        fits: |_| (),
        name: "",
        request: plain,
        words: String::new,
    };

    /// The cell `name` names.
    fn named(name: &str) -> Self {
        let named = CELLS.iter().find(|cell| cell.name == name);
        *named.unwrap_or_else(|| panic!("no cell {name}"))
    }
}

/// The cells, in the order of their names. A change that moves a policy's
/// cost adds its entry here.
const CELLS: &[Cell] = &[
    Cell {
        name: "bucket",
        document: || preset("bucket"),
        ..Cell::PLAIN
    },
    Cell {
        name: "cache_aware",
        document: || preset("cache_aware"),
        behind: conversations::lends,
        caller: Some(conversations::caller),
        fits: conversations::fits,
        words: conversations::words,
        ..Cell::PLAIN
    },
    Cell {
        name: "cache_aware_sessions",
        document: || sessions_over("cache_aware", RulePlaced::Untold),
        behind: sessions,
        request: session,
        ..Cell::PLAIN
    },
    Cell {
        name: "cache_aware_sessions_told",
        document: || sessions_over("cache_aware", RulePlaced::Told),
        behind: sessions,
        request: session,
        ..Cell::PLAIN
    },
    Cell {
        name: "consistent_hashing",
        document: || preset("consistent_hashing"),
        request: keyed,
        ..Cell::PLAIN
    },
    // The hash ring's document with a request that carries no key: the
    // one cell of that document the one call for a request can be asked
    // for (`--entry place`), where it reads the request's headers for the
    // worker a request may name.
    Cell {
        name: "consistent_hashing_no_key",
        document: || preset("consistent_hashing"),
        ..Cell::PLAIN
    },
    // An encode request held to places of each of its three lists: every
    // worker of the list, as a gateway states the workers on its wire.
    Cell {
        name: "encode_held_least_load",
        document: || encode::of("least_load"),
        request: encode::request,
        words: encode::words,
        ..Cell::PLAIN
    },
    Cell {
        name: "encode_least_load",
        document: || encode::of("least_load"),
        request: encode::request,
        words: encode::words,
        ..Cell::PLAIN
    },
    Cell {
        name: "encode_round_robin",
        document: || encode::of("round_robin"),
        request: encode::request,
        words: encode::words,
        ..Cell::PLAIN
    },
    Cell {
        name: "encode_signal_cost",
        document: signals::encode_document,
        request: encode::request,
        words: encode::words,
        ..Cell::PLAIN
    },
    Cell {
        name: "least_load",
        document: || preset("least_load"),
        ..Cell::PLAIN
    },
    Cell {
        name: "pair_least_load",
        document: || pair::of("least_load"),
        request: pair::request,
        ..Cell::PLAIN
    },
    Cell {
        name: "pair_round_robin",
        document: || pair::of("round_robin"),
        request: pair::request,
        ..Cell::PLAIN
    },
    Cell {
        name: "pair_round_robin_later",
        document: || pair::of("round_robin"),
        request: pair::later,
        ..Cell::PLAIN
    },
    Cell {
        name: "pair_signal_cost",
        document: signals::pair_document,
        request: pair::request,
        ..Cell::PLAIN
    },
    Cell {
        name: "passthrough",
        document: || preset("passthrough"),
        ..Cell::PLAIN
    },
    Cell {
        name: "power_of_two",
        document: || preset("power_of_two"),
        ..Cell::PLAIN
    },
    Cell {
        name: "prefix_hash",
        document: || preset("prefix_hash"),
        request: keyed,
        ..Cell::PLAIN
    },
    Cell {
        name: "prefix_hash_prompt",
        document: || preset("prefix_hash"),
        ..Cell::PLAIN
    },
    Cell {
        name: "random",
        document: || preset("random"),
        ..Cell::PLAIN
    },
    Cell {
        name: "round_robin",
        document: || preset("round_robin"),
        ..Cell::PLAIN
    },
    Cell {
        name: "sent_cost",
        document: sent_cost,
        ..Cell::PLAIN
    },
    Cell {
        name: "signal_cost",
        document: signals::document,
        ..Cell::PLAIN
    },
];

/// A built-in preset as a cell's document.
fn preset(name: &str) -> PolicySpec {
    PolicySpec::preset(name).expect("a preset's name")
}

/// The `sent_cost` cell's document: the `least_load` preset costed by its
/// profile's count of sends. Each decision reads every row's count, the
/// requests the row was sent and their prompt tokens, and each leased
/// decision adds its send to its row's.
fn sent_cost() -> PolicySpec {
    let mut document = preset("least_load");
    let cost = r#"{"least-cost":{"terms":{"load.waiting_uncached_tokens":1,
        "sent.requests":1000,"sent.tokens":1},"unreported":"missing-as-zero"}}"#;
    let profile = document.profiles.get_mut("least_load");
    let profile = profile.expect("the preset's profile");
    profile.score = vec![serde_json::from_str(cost).expect("a scorer")];
    document
}

/// The profile of the preset `name` under the stock sessions rule, a new
/// session placed by the profile's own stages, and `kept`, the rule's word
/// of a request it places alone. A gateway's routing settings write
/// `untold` for the sticky override's rule: a pinned request is kept from
/// the profile, and its prompt is handed to no record. Under `told`, the
/// rule's default, a pinned request is the profile's own: it is named its
/// prompt's holders, and its prompt is handed to the record for its worker.
fn sessions_over(name: &str, kept: RulePlaced) -> PolicySpec {
    let mut document = preset("manual");
    document.profiles = preset(name).profiles;
    let binding = &mut document.bindings[0];
    binding.profile = Some(name.into());
    let rule = binding.sessions.as_mut().expect("the stock rule");
    rule.assignment = smg_policy_controller::document::Assignment::Delegate;
    rule.rule_placed = kept;
    document
}

/// The sessions of a cell under a sessions rule: each one's routing key and
/// the hashes of its prompt.
static SESSIONS: OnceLock<Vec<(String, smg_policy_controller::engine::PromptHashes)>> =
    OnceLock::new();

/// Makes the cell's 4,096 sessions, each with a prompt of 1,024 tokens of
/// its own, hashed as `engine`'s document reads a prompt.
fn sessions(engine: &PolicyEngine) {
    use smg_policy_controller::engine::{DecisionEngine, HashSeeds};
    let one = |at: u32| {
        let tokens: Vec<u32> = (at * 100_000..at * 100_000 + 1_024).collect();
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        (format!("session-{at:04}"), hashes)
    };
    SESSIONS.get_or_init(|| (0..4_096).map(one).collect());
}

/// A request of one of the cell's sessions: its routing key in the clear,
/// which a host counts on each worker, and its prompt's hashes.
fn session(seed: u64) -> Request<'static> {
    let sessions = SESSIONS.get().expect("the cell's sessions");
    let (key, prompt) = &sessions[(seed % sessions.len() as u64) as usize];
    let mut request = plain(seed);
    request.facts.session_key = Identity::plain(key);
    request.facts.session_counted = true;
    prompt.write_to(&mut request.facts);
    request
}

/// A plain single request of 1,024 prompt tokens: it asks only a worker.
fn plain(seed: u64) -> Request<'static> {
    let facts = RequestFacts {
        model: MODEL,
        phase: Phase::Single,
        prompt_tokens: 1_024,
        seed,
        ..RequestFacts::default()
    };
    Request {
        id: b"bench",
        facts,
        constraints: Constraints::default(),
    }
}

/// The routing keys a keyed cell's callers send, one a turn by its seed.
const KEYS: u64 = 1_024;

/// The ring a host lends with a keyed request: the list's addresses in the
/// list's order, walked round once from the place a key states. A stand-in
/// for the ring a host keeps: it hashes nothing and holds no points, so a
/// keyed cell counts what the engine does with a lent ring and no lookup of
/// a host's ring.
#[derive(Debug)]
struct ListOrder(Vec<String>);

impl LentRing for ListOrder {
    fn position(&self, _: &str) -> u64 {
        0
    }

    // A head stands at its own size: nothing is hashed for it either.
    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        Some(match head {
            Head::Tokens(ids) => ids.len() as u64,
            Head::Text(text) => text.len() as u64,
        })
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let (before, behind) = self.0.split_at(position as usize % self.0.len());
        let _ = behind.iter().chain(before).any(|address| take(address));
    }
}

/// The run's ring, made of its list before the first request ([`lend`]).
static RING: OnceLock<ListOrder> = OnceLock::new();

/// Makes the run's ring of `workers`, in the list's order.
fn lend(workers: &[Arc<dyn Worker>]) {
    let addresses = workers.iter().map(|worker| worker.url().to_owned());
    let lent = RING.set(ListOrder(addresses.collect()));
    lent.expect("one list a run");
}

/// Where each routing key stands on the ring: its own number, kept as a
/// host keeps what its ring answered for a key.
static PLACES: [u64; KEYS as usize] = {
    let mut places = [0; KEYS as usize];
    let mut key = 0;
    while key < places.len() {
        places[key] = key as u64;
        key += 1;
    }
    places
};

/// [`plain`]'s request with a routing key, one of [`KEYS`]: a ring places
/// it by the key. Its host lends its ring with it ([`RING`]) and states
/// where the key stands there ([`PLACES`]).
fn keyed(seed: u64) -> Request<'static> {
    let mut request = plain(seed);
    let key = seed % KEYS;
    request.facts.session_key = Identity::hashed(mix(key) | 1);
    if let Some(ring) = RING.get() {
        request.facts.ring = Some(ring);
    }
    request.facts.ring_positions = std::slice::from_ref(&PLACES[key as usize]);
    request
}

/// A mix of `value`'s bits: a seed a request, a report's numbers a worker.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Which binding of a run's document names a copy of the cell's profile as
/// its shadow.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shadowed {
    /// None: the document is the cell's preset as it stands.
    No,
    /// The cell's own binding: each decision has a shadow.
    Twin,
    /// A binding of another model: the cell's decisions have none.
    Other,
}

/// One run, as the switches state it.
struct Run {
    callers: usize,
    cell: Cell,
    decisions: u64,
    /// The entry asked for a request: the bench's `place` file reads it.
    entry: String,
    overloaded: bool,
    publish_ms: u64,
    /// What refuses every pair of a pair's cell: the bench's `pair` file
    /// reads it.
    refused: pair::Refused,
    shadow: Shadowed,
    /// Whether a thread of its own turns the engine's task behind its first
    /// step.
    upkeep_thread: bool,
    workers: usize,
}

fn run() -> Run {
    let mut run = Run {
        callers: 1,
        cell: Cell::named("round_robin"),
        decisions: 1_000_000,
        entry: "decide_at".to_owned(),
        overloaded: false,
        publish_ms: 0,
        refused: pair::Refused::No,
        shadow: Shadowed::No,
        upkeep_thread: true,
        workers: 64,
    };
    let mut switches = env::args().skip(1);
    while let Some(switch) = switches.next() {
        // Cargo's own switch for a bench binary.
        if switch == "--bench" {
            continue;
        }
        let value = switches.next().expect("a value behind each switch");
        let number = || value.parse::<u64>().expect("a number");
        match switch.as_str() {
            "--callers" => run.callers = number() as usize,
            "--cell" => run.cell = Cell::named(&value),
            "--decisions" => run.decisions = number(),
            "--entry" => run.entry = place::named(&value),
            "--overloaded" => run.overloaded = number() != 0,
            "--publish-ms" => run.publish_ms = number(),
            "--refused" => run.refused = pair::Refused::named(&value),
            "--shadow" => {
                run.shadow = match value.as_str() {
                    "twin" => Shadowed::Twin,
                    "other" => Shadowed::Other,
                    other => panic!("no shadow {other}"),
                };
            }
            "--upkeep-thread" => run.upkeep_thread = number() != 0,
            "--workers" => run.workers = number() as usize,
            other => panic!("no switch {other}"),
        }
    }
    run
}

/// One report a worker: queued token-work, running requests, KV pressure
/// and a live rate, as the gateway's decision bench gives its fleet.
fn reports(workers: &[Arc<dyn Worker>]) -> HashMap<String, WorkerLoadResponse> {
    let report = |(at, worker): (usize, &Arc<dyn Worker>)| {
        let draw = |salt: u64| mix(at as u64 * 16 + salt);
        let rate = if draw(3) % 4 == 0 {
            0.0
        } else {
            500.0 + (draw(3) % 3_500) as f64
        };
        let queued = (draw(1) % 8_192) as i32;
        let mut report = one_rank_load(queued, (draw(2) % 95) as f64 / 100.0, rate);
        report.loads[0].num_running_reqs = (draw(4) % 12) as i32;
        (worker.url().to_owned(), report)
    };
    workers.iter().enumerate().map(report).collect()
}

/// One turn's request as its host states it: made where the engine makes
/// one. It gives no load snapshot: the engine reads its own.
struct Stated<'a> {
    host: &'a Host,
    seed: u64,
    /// The turn's prompt, where the cell's callers hold conversations.
    prompt: Option<&'a conversations::Turn<'a>>,
}

impl RequestMaker for Stated<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        let Some(prompt) = self.prompt else {
            return (self.host.request)(self.seed);
        };
        // The conversation so far, as its host hashed it: the first blocks
        // of its full prompt.
        let mut request: Request<'w> = (self.host.request)(self.seed);
        let facts = &mut request.facts;
        prompt.hashes.write_to(facts);
        let blocks = prompt.tokens.len() / facts.block_size as usize;
        facts.block_hashes = &facts.block_hashes[..blocks];
        facts.prompt_tokens = prompt.tokens.len() as u32;
        request
    }

    fn plain(&self) -> Option<Plain> {
        self.host.plain
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.host.list.lent())
    }

    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        self.prompt?.token_key(page)
    }
}

/// A list of worker objects as a host publishes it: the list, and the cell
/// the host keeps beside it.
struct Published {
    workers: Arc<[Arc<dyn Worker>]>,
    cell: ListCell,
}

impl Published {
    /// `workers` published: the cell beside them is empty.
    fn of(workers: &Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            workers: Arc::clone(workers),
            cell: ListCell::default(),
        }
    }

    /// The list, lent with its cell: the host states nothing of its
    /// workers.
    fn lent(&self) -> LentList<'_> {
        LentList {
            workers: &self.workers,
            cell: &self.cell,
            facts: None,
        }
    }
}

/// What a host keeps for one list: the engine, which stands as long as the
/// process, the list as published, and the bench cell's request.
struct Host {
    engine: &'static PolicyEngine,
    list: Published,
    request: fn(u64) -> Request<'static>,
    plain: Option<Plain>,
}

impl Host {
    /// How many times the list's reports were converted so far.
    fn conversions(&self) -> u64 {
        self.engine.conversions(self.list.lent())
    }

    /// One request by the engine's one entry: the place in the list it was
    /// sent to, with the dispatch and the end told where a decision is leased.
    fn place(&self, seed: u64, prompt: Option<&conversations::Turn<'_>>) -> Option<u32> {
        let stated = Stated {
            host: self,
            seed,
            prompt,
        };
        // The instant the engine dated the decision at, where it dated one.
        let dated = OnceCell::new();
        let clock = || *dated.get_or_init(Instant::now);
        match self.engine.decide_at(&stated, clock).ok()? {
            Decided::Alone(pick) => Some(pick.place),
            Decided::Whole(decided, later) => {
                let target = *decided.decision.pick()?.targets.first()?;
                if decided.lease.is_some() {
                    let now = *dated.get()?;
                    let mut held =
                        DecisionGuard::holding(self.engine, GatewayId::LOCAL, decided, later, now);
                    Self::told(&mut held, target.id, now);
                }
                Some(target.position)
            }
        }
    }

    /// The dispatch and the end a host tells of a leased decision, in a
    /// function of its own: a profile then names the decision at the
    /// engine's entry apart from them, as [`Host::place`] less this.
    #[inline(never)]
    fn told(held: &mut DecisionGuard<&PolicyEngine>, served: WorkerId, now: Instant) {
        held.dispatched(now);
        let mut end = Outcome::reported(200, TransportClass::Ok).with_served(Side::Single, served);
        end.tokens_in = 1_024;
        held.completed(&end, now);
    }
}

/// One caller's turns by `turn`: its first untimed, then, behind the start
/// every caller waits at, its timed ones as one stretch. Says how many of
/// those were placed, the stretch in nanoseconds and its allocations.
fn stretch<T>(
    start: &Barrier,
    decisions: u64,
    mut turn: impl FnMut(u64) -> Option<T>,
) -> (u64, u64, u64) {
    for at in 0..CALLER_WARM {
        black_box(turn(at));
    }
    start.wait();
    let (allocated, started) = (calls(), Instant::now());
    let mut took = 0;
    for at in CALLER_WARM..CALLER_WARM + decisions {
        took += u64::from(black_box(turn(at)).is_some());
    }
    let nanos = started.elapsed().as_nanos() as u64;
    (took, nanos, (calls() - allocated) as u64)
}

fn main() {
    let run = run();
    // A pair's cell hands two lists and an encode request's three: such a
    // run is its file's.
    if pair::runs(&run) || encode::runs(&run) {
        return;
    }
    let open = run.refused == pair::Refused::No;
    assert!(open, "a pair's cell alone runs refused");
    let Cell {
        behind,
        document,
        fits,
        request,
        ..
    } = run.cell;
    fits(&run);
    let workers: Arc<[Arc<dyn Worker>]> = (0..run.workers)
        .map(|at| ready_worker(&format!("http://worker-{at:04}:8000"), MODEL))
        .collect();
    lend(&workers);
    if run.overloaded {
        for (at, worker) in workers.iter().enumerate() {
            worker.set_overloaded(true);
            if at != workers.len() / 2 {
                for _ in 0..=at {
                    worker.increment_load();
                }
            }
        }
    }
    let mut document = document();
    if run.shadow != Shadowed::No {
        // A copy of the profile the cell's binding names.
        let live = document.bindings[0].profile.clone();
        let twin = document.profiles[&live.expect("a binding of one profile")].clone();
        document.profiles.insert("twin".to_owned(), twin);
        let named = Some(ShadowSpec::Profile("twin".to_owned()));
        if run.shadow == Shadowed::Other {
            let mut other = document.bindings[0].clone();
            "another-model".clone_into(&mut other.matcher.model);
            other.shadow = named;
            document.bindings.insert(0, other);
        } else {
            document.bindings[0].shadow = named;
        }
    }
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
    behind(engine);
    let mut receiver = None;
    let feed = TestFeed::attached_to(|loads| receiver = loads);
    let loads = reports(&workers);
    let loads = signals::stated(&document, &workers, loads);
    feed.publish(&loads);
    // What a host does once: it hands its load monitor's channel and spawns
    // the engine's task, here on a runtime of its own.
    engine.set_loads(receiver.expect("the feed's channel"));
    let upkeep = Builder::new_current_thread().enable_time().build();
    let upkeep = upkeep.expect("a runtime");
    upkeep.spawn(engine.run());
    let host = Host {
        engine,
        list: Published::of(&workers),
        request,
        plain: request(0).plain(),
    };
    let (publish, converted) = (|| feed.publish(&loads), || host.conversions());
    // The one call a host makes for a request, where the run asks it.
    if place::asked(&run, &host) {
        let (rule, ring) = (ruled::stated(&run), place::lent(&host));
        if worded::stated(&run) {
            measure(&run, &host, upkeep, publish, converted, |_| {
                worded::one(&host, ring)
            });
            return;
        }
        measure(&run, &host, upkeep, publish, converted, |_| {
            place::one(&host, rule, ring)
        });
        return;
    }
    measure(&run, &host, upkeep, publish, converted, |seed| {
        host.place(seed, None)
    });
}

/// What turns the engine's task behind its first step: a thread of its own,
/// as a host's runtime does, or nothing (`--upkeep-thread 0`). A runtime
/// that no thread takes is given back, and its caller keeps it to the run's
/// end. Nothing reads a publication then, so such a run publishes once.
fn turned(run: &Run, upkeep: Runtime) -> Option<Runtime> {
    if run.upkeep_thread {
        thread::spawn(move || upkeep.block_on(pending::<()>()));
        return None;
    }
    assert_eq!(run.publish_ms, 0, "the task's thread reads a publication");
    Some(upkeep)
}

/// The run's decisions, each made by `decide`, and the run's one line.
/// `host` is the cell's host: `upkeep` is the runtime its engine's task was
/// spawned on, and a cell whose callers make turns of their own asks it.
/// `publish` publishes the reports again, as a new snapshot, and
/// `converted` says how many times the cell's lists had their reports
/// converted so far.
fn measure(
    run: &Run,
    host: &Host,
    upkeep: Runtime,
    publish: impl Fn() + Sync,
    converted: impl Fn() -> u64,
    decide: impl Fn(u64) -> Option<u32> + Sync,
) {
    // The task's first step is taken ahead of the list's first requests:
    // the runtime is turned for it, then requests are made until one reads
    // the engine's snapshot, each try a request of its own. A list converts
    // its reports once its requests have looked up as many as it holds, so
    // the tries follow the list's length. A thread of its own turns the
    // runtime from then on, where the run has one (`turned`).
    let read = || !host.engine.spec().needs().loads || converted() > 0;
    for probe in 0..1_000 + run.workers as u64 {
        if read() {
            break;
        }
        if probe < 16 {
            upkeep.block_on(async { sleep(Duration::from_millis(1)).await });
        }
        black_box(decide(mix(probe)));
    }
    assert!(read(), "the engine's task keeps the reports at hand");
    let _upkeep = turned(run, upkeep);
    // The list's first requests, untimed: what they build once.
    for turn in 0..(run.workers as u64 * 8).max(4_096) {
        assert!(decide(mix(turn)).is_some(), "a worker takes it");
    }

    // Each caller makes its first decisions untimed; all start together.
    let start = Barrier::new(run.callers + 1);
    let (busy_nanos, placed, allocations) =
        (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let (publications, left, end) = (AtomicU64::new(0), Mutex::new(run.callers), Condvar::new());
    let converted_before = converted();
    let wall = thread::scope(|scope| {
        let one_caller = |caller: u64| {
            let seed = |turn: u64| mix(caller << 40 | turn);
            // A cell whose callers make turns of their own times them
            // itself; any other caller asks the run's `decide`.
            let (took, nanos, allocated) = match run.cell.caller {
                Some(turns) => turns(host, &start, run.decisions, caller),
                None => stretch(&start, run.decisions, |turn| decide(black_box(seed(turn)))),
            };
            busy_nanos.fetch_add(nanos, Ordering::Relaxed);
            allocations.fetch_add(allocated, Ordering::Relaxed);
            placed.fetch_add(took, Ordering::Relaxed);
            *left.lock() -= 1;
            end.notify_one();
        };
        let callers: Vec<_> = (0..run.callers as u64)
            .map(|caller| scope.spawn(move || one_caller(caller)))
            .collect();
        start.wait();
        let started = Instant::now();
        // The monitor's side, under the lock: none after the last caller's end.
        if run.publish_ms > 0 {
            scope.spawn(|| {
                let (gap, mut left) = (Duration::from_millis(run.publish_ms), left.lock());
                while end.wait_while_for(&mut left, |n| *n > 0, gap).timed_out() && *left > 0 {
                    publish();
                    publications.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        for caller in callers {
            caller.join().expect("a caller that ends");
        }
        started.elapsed()
    });
    let total = run.decisions * run.callers as u64;
    assert_eq!(placed.into_inner(), total, "a worker takes it");
    print!(
        "cell={} workers={} callers={} overloaded={} publish_ms={} publications={} conversions={} allocations_per_decision={:.4} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
        run.cell.name,
        run.workers,
        run.callers,
        run.overloaded,
        run.publish_ms,
        publications.into_inner(),
        converted() - converted_before,
        allocations.into_inner() as f64 / total as f64,
        total,
        wall.as_secs_f64(),
        total as f64 / wall.as_secs_f64(),
        busy_nanos.into_inner() as f64 / total as f64,
    );
    // The words a file of its own adds, then the cell's, and the line's end.
    print!("{}", place::word());
    println!("{}", (run.cell.words)());
}

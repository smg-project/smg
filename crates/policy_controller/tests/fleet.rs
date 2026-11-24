//! The end-to-end case, the harness first: a controller on ephemeral ports,
//! a fleet of mock engines and the generated client, in one process. The
//! fleet registers, an engine of one rank through the control scope's call;
//! the controller's own load monitor polls it; each engine is asked its
//! health and what it answered is written, by the controller's own prober
//! where its build has one and by a probe turn of the case elsewhere; and
//! the pool is read back as the store holds it: every rank a row, healthy
//! by its probe, its load from a poll, an engine that holds requests
//! reading them.
//! That engine's KV events follow: a subscriber of the case hands each batch
//! of its stream to the sink, and its row's index holds every block the
//! events left the engine holding, where the cache-aware preset then asks a
//! prompt that states them. Then one engine is killed: the first probe
//! that hears it dead writes it, and the pool reads its rows unhealthy and
//! every other row healthy. Next, an engine stops answering with its sockets
//! open: its probe waits out its bound and writes it, and a dial turn of
//! the case waits out its own and goes on to every other engine. Last,
//! the gateway sends requests of its own and reports each on its session:
//! the store holds the completion of the engine that answered, none of class
//! ok for the dead and the hung one, and nothing of it for another gateway;
//! a frame that reads wrong ends the session of the gateway that sent it.
//!
//! The gateway asks its first decisions end to end: with a preset of the
//! build applied in the case's process, since no call applies a spec yet,
//! the reference client asks on its decide stream and each pick names a
//! rank of the fleet. The call's bounds are the client's own door: an id at
//! the range's end is answered, a frame past the size leaves cut at its
//! tail, and a frame the call refuses ends the stream, the next ask opening
//! a new one. Then the decisions follow the fleet, each read from the
//! client's answers alone: a load-led pick leaves the engine that holds
//! requests, a rotation names each row in turn, a routing key returns to its
//! worker and a retry leaves it, and no decision names the dead engine.
//! The gateway's own constraints narrow a decision first: a row it
//! excludes is no pick, and a hint of no candidates or a wire no row
//! speaks is shed under the constraint's name.
//! A decision by count follows the leases the decisions before it left on
//! the store's file: with every lease kept, a round of asks by the fewest
//! requests names each row once, and once its own leases are given up the
//! engine that holds requests takes the next pick by the fewest requests,
//! of two draws compared by count, and of a keyless ask under the prefix
//! ring.
//! A session stays where its key was first placed, and where the gateway
//! sent it: a dispatch on a pick's second target moves the key and the
//! lease's count there, and a key served on its failover keeps its worker.
//! A new document's engine starts with no session pin yet: a key that was
//! dispatched is placed anew behind an apply.
//! A prompt returns to its worker as a key does: under the prefix preset a
//! head of 256 tokens names one row whatever follows it, a routing key
//! leads it, a retry walks to one next worker, a worker past its bounded
//! load is passed over until its leases are given up, and a boundary of
//! the document's view keys a prompt at its stem, where a prompt past its
//! loaded worker lands.
//! Each model decides under the profile its own binding names: a second
//! pool, on the wire whose load report counts resident tokens, is asked
//! under the resident form of the two-draw preset while the first turns a
//! rotation, and every pick of it names the engine that holds no tokens.
//! A prefill request goes by its size: the three engines of a prefill pool
//! split the sizes among them, requests of like size meet on one engine,
//! and under the bucket preset's own balance rule an engine a decision
//! booked gives way to one that booked nothing.
//! A host's own stages decide as the stock ones do: a controller bound over
//! a catalog that holds a scorer in code, a WebAssembly guest's and a
//! served plugin's beside the stock stages, on a spec file that names the
//! three, answers each cache namespace of a request on a row of its own,
//! and under either plugin passes over the row that holds a lease.
//!
//! The overload veto is read under an operator's patch on the control
//! listener: the engine that holds requests, patched past its threshold,
//! is no pick of a rotation until the gateway lifts the veto, and a session
//! key keeps its ring worker over it.
//!
//! A gateway's link answers concurrent asks. Their leases leave the store
//! on the caller's reports, or on the link's own release if an answer
//! arrives after the caller's budget. A watch on the link takes the
//! snapshot and the tail beside an ask, and can follow after the link ends.
//!
//! No stub server: the engines are the mock worker's, the controller is the
//! one the binary runs and the client is the generated one. Every wait is
//! bounded and named, and the case stands under one outer bound.

#[expect(
    dead_code,
    reason = "this binary's engines run on runtimes of their own; the engine served on the test's runtime is the load path's"
)]
#[path = "common/engines.rs"]
mod engines;
#[path = "common/fleet.rs"]
mod fleet;
#[cfg(feature = "wasm")]
#[expect(
    dead_code,
    reason = "this binary serves the plugin in its honest behaviour and reads nothing it was sent"
)]
#[expect(
    clippy::expect_used,
    reason = "the served plugin runs inside the case; a failure there is the case's failure"
)]
#[path = "support/grpc_plugin.rs"]
mod grpc_plugin;
#[expect(dead_code, reason = "this binary reads no admin page")]
#[path = "common/net.rs"]
mod net;
#[path = "common/pair.rs"]
mod pair;
#[cfg(feature = "wasm")]
#[expect(
    dead_code,
    reason = "this binary loads the record guest in its honest mode and nothing else of the module"
)]
#[expect(
    clippy::expect_used,
    reason = "the guest's assembly runs inside the case; a failure there is the case's failure"
)]
#[path = "support/wasm_guest.rs"]
mod wasm_guest;

use std::{
    collections::{HashMap, HashSet},
    io::Write,
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

use engines::cpu_seconds;
use fleet::{
    within, DialEnd, Harness, Replica, Shape, DIAL_TIMEOUT, GATEWAY, POLL_INTERVAL, PROBE_INTERVAL,
    PRODUCT_PROBES, ROOM,
};
use net::vm_rss_kib;
use openai_protocol::worker::RuntimeType;
use prost::Message as _;
use smg_policy_controller::{
    document::{BindingSpec, Match, PolicySpec, SpecError},
    engine::{Role, WireModes, WorkerId},
    policies::{bucket, power_of_two::PowerOfTwoMetric},
    service::{
        client::{self, blocking, Answer, Client, DecisionUnavailable, Link},
        controller::{State, PROTOCOL_VERSION},
        proto::{
            self, busy, completed::Class, decision::Outcome, gateway_frame, identity::Form,
            principal, released, worker_event, Completed, DecideRequest, Decision, Dispatched,
            Hello, Identity, Lease, OverloadThresholds, Released, RequestFacts, ShedReason, Side,
            WatchWorkersRequest,
        },
        watcher::{WatchStream, WatchStreamEnd, Watcher},
    },
    store::{LaneCounts, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{Members, Objects},
};
use smg_worker::{registry::WorkerRegistry, worker::WorkerStatus as ObjectStatus};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{oneshot, Barrier, Notify},
    task::JoinSet,
};
use tonic::{transport::Endpoint, Code};

/// A document that holds a stage that keeps load reports is given none in
/// a controller's decision yet: the two-draw preset's resident form holds
/// one.
const READS_KEPT_LOADS: bool = false;

/// A new document's engine starts with no session pin yet: a pinned key is
/// placed anew at its first request behind an apply.
const KEEPS_PINS_ACROSS_AN_APPLY: bool = false;

/// A controller's decisions read no cache event yet: a prompt is not sent
/// to the worker whose events say it holds the blocks.
const READS_CACHE_EVENTS: bool = false;

const MODEL: &str = "fleet-model";

/// The model of the engines whose load reports count resident tokens.
const COUNTED: &str = "fleet-counted-model";

/// The model of the engines that serve the prefill side of a request.
const SIZED: &str = "fleet-sized-model";

/// The fleet: four engines of one rank and one of two serve one model on
/// the first wire, two engines of one rank serve another on the wire whose
/// load report counts resident tokens, and three engines of one rank serve
/// the prefill side of a third.
const FLEET: [Shape; 10] = [
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 2,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: COUNTED,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::TokenSpeed,
    },
    Shape {
        model: COUNTED,
        role: Role::Single,
        ranks: 1,
        runtime: RuntimeType::TokenSpeed,
    },
    Shape {
        model: SIZED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::TokenSpeed,
    },
    Shape {
        model: SIZED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::TokenSpeed,
    },
    Shape {
        model: SIZED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::TokenSpeed,
    },
];

/// The requests the case holds on one engine.
const HELD: u32 = 3;

/// The largest `DecideRequest` as encoded, as the contract states it.
const FRAME: usize = 64 * 1024;

/// An ask for the fleet's model named `id`. It brings the budget of a
/// client's own patience: an ask that states none has none left.
fn ask(id: &[u8]) -> DecideRequest {
    DecideRequest {
        request_id: id.to_vec(),
        facts: Some(RequestFacts {
            model: MODEL.to_owned(),
            budget_micros: 5_000_000,
            ..RequestFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    }
}

/// The routing keys among which one is the busy engine's: a ring of even
/// shares over six rows leaves 256 keys off one row less than once in
/// 10^20 runs.
const KEYS: usize = 256;

/// The asks an assertion makes where one draw could hide what it reads.
const ASKS: u64 = 64;

/// Tokens a block: the view's, where a document names no size.
const BLOCK: u32 = 16;

/// The blocks of a prompt's head, what the prefix preset keys a request
/// within: 256 tokens.
const HEAD: u32 = 256 / BLOCK;

/// The blocks of a prompt's stem, where the case's own view sets its
/// boundary: 64 tokens.
const STEM: u32 = 64 / BLOCK;

/// The kept leases that carry a head's worker past the prefix preset's
/// bounded load, in a pool of six rows that books nothing else: with
/// thirteen kept the next request counts in a mean of 14 over 6, and 13
/// stands past that mean by more than the margin of ten.
const LOADED: u64 = 13;

/// A pick's first target: the row of the fleet it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Picked {
    engine: usize,
    rank: u16,
    /// The row's id, as a retry names it.
    id: u64,
}

/// The `n`th ask of the run named `run`, with an id and a seed of its own.
fn nth(run: &str, n: u64) -> DecideRequest {
    let mut ask = ask(format!("{run}-{n}").as_bytes());
    if let Some(facts) = &mut ask.facts {
        facts.seed = n;
    }
    ask
}

/// Waits until a rotation leaves `engine` out for a whole turn: as many
/// asks of the run named `run` as the fleet's pool has `rows`, one after
/// another, none answered with a row of `engine`. An overload verdict
/// stands nowhere a case reads, so a veto is read in what the service
/// answers.
async fn left_out(
    harness: &mut Harness,
    run: &str,
    rows: usize,
    engine: usize,
) -> Result<(), String> {
    let bound = POLL_INTERVAL + ROOM;
    let asked = Instant::now();
    let mut spared = 0;
    for n in 0u64.. {
        if spared == rows {
            return Ok(());
        }
        if asked.elapsed() > bound {
            break;
        }
        let pick = picked(harness, nth(run, n)).await?;
        spared = if pick.engine == engine { 0 } else { spared + 1 };
    }
    Err(format!(
        "a rotation leaves a vetoed engine out for a turn: not within {bound:?}"
    ))
}

/// Waits until an ask of the run named `run` is answered with a row of
/// `engine`: a rotation names the engine again once no veto stands on it.
async fn named_again(harness: &mut Harness, run: &str, engine: usize) -> Result<(), String> {
    let bound = POLL_INTERVAL + ROOM;
    let asked = Instant::now();
    for n in 0u64.. {
        if picked(harness, nth(run, n)).await?.engine == engine {
            return Ok(());
        }
        if asked.elapsed() > bound {
            break;
        }
    }
    Err(format!(
        "a rotation names an engine whose veto ended: not within {bound:?}"
    ))
}

/// Waits until the controller's load monitor has polled every row of each
/// of `engines` since `since`: a decision is given the monitor's snapshot,
/// and so is the pool's reader.
async fn monitored(
    harness: &Harness,
    engines: &[usize],
    since: Instant,
) -> Result<Duration, String> {
    within(
        "the load monitor polls each engine again",
        POLL_INTERVAL + ROOM,
        || harness.monitored(engines, since),
    )
    .await
}

/// `ask` under the routing key `key`.
fn keyed(mut ask: DecideRequest, key: u64) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        facts.session_key = Some(Identity {
            form: Some(Form::Hash(key)),
        });
    }
    ask
}

/// A key's fingerprint by the contract: FNV-1a over its bytes, a result of
/// zero carried as one.
fn print_of(key: &str) -> u64 {
    let print = key.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |print, byte| {
        (print ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    print.max(1)
}

/// The session a key names within a model under the single phase, by the
/// contract's formula over the key's and the model's fingerprints.
fn session_of(key: &str, model: &str) -> u64 {
    const MIX: u64 = 0x9e37_79b9_7f4a_7c15;
    let scope = print_of(model) ^ MIX;
    let mut value = print_of(key) ^ scope.wrapping_mul(MIX);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    value.max(1)
}

/// Asks under a session key, and reads the first target's id with the
/// session the answer's one lease names; the lease is given up.
async fn sessioned(harness: &mut Harness, ask: DecideRequest) -> Result<(u64, u64), String> {
    let before = harness.leases(GATEWAY);
    let answer = harness.client().decide(ask).await;
    let Ok(Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    })) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let ([lease], Some(target)) = (leases.as_slice(), pick.targets.first()) else {
        return Err(format!(
            "a keyed pick carries one lease and a target: {answer:?}"
        ));
    };
    let read = (target.id, lease.session_key);
    given_up(harness, &[lease.id], before).await?;
    Ok(read)
}

/// `ask` with the prefix block hashes of a prompt of `runs`, each so many
/// blocks of one text. A block's hash names the prompt through its block:
/// two prompts share the hashes of the blocks they share from the start and
/// none past them.
fn prompted(mut ask: DecideRequest, runs: &[(u32, u64)]) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        let mut through = 0u64;
        for &(blocks, text) in runs {
            for _ in 0..blocks {
                through = (through ^ text)
                    .wrapping_mul(0x0100_0000_01b3)
                    .rotate_left(29)
                    | 1;
                facts.block_hashes.push(through);
            }
        }
    }
    ask
}

/// `ask` for the model named `model`.
fn for_model(mut ask: DecideRequest, model: &str) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        model.clone_into(&mut facts.model);
    }
    ask
}

/// `ask` as the prefill side of a request of `tokens` for the sized pool.
fn sized(mut ask: DecideRequest, tokens: u32) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        SIZED.clone_into(&mut facts.model);
        facts.phase = proto::Phase::Prefill.into();
        facts.prompt_tokens = tokens;
    }
    ask
}

/// The bucket preset under a balance rule that no booking of the case
/// reaches and a window that outlasts it: sizes alone place a request.
fn by_size() -> Result<PolicySpec, SpecError> {
    let mut document = PolicySpec::preset("bucket")?;
    for profile in document.profiles.values_mut() {
        let sizes = smg_policy_controller::document::BucketParams {
            abs_threshold: Some(u32::MAX),
            adjust_interval_s: Some(3_600),
            ..Default::default()
        };
        profile.score = vec![smg_policy_controller::document::ScoreStage::Bucket(sizes).into()];
    }
    Ok(document)
}

/// One document of two bindings: the counted model under the two-draw
/// preset's resident form, a request priced at the held prompt's tokens,
/// and every other model under a rotation.
fn by_model() -> Result<PolicySpec, SpecError> {
    let mut document = PolicySpec::preset("round_robin")?;
    let metric = PowerOfTwoMetric::ResidentTokens { request_tokens: 64 };
    let resident = PolicySpec::power_of_two(metric)?;
    for (name, profile) in resident.profiles {
        // The first binding that matches decides: the model's own stands
        // ahead of the one that matches every model.
        document.bindings.insert(
            0,
            BindingSpec {
                matcher: Match {
                    model: COUNTED.to_owned(),
                    ..Match::default()
                },
                profile: Some(name.clone()),
                ..BindingSpec::default()
            },
        );
        document.profiles.insert(name, profile);
    }
    Ok(document)
}

/// The prefix preset under a view whose boundaries stand at `tokens`,
/// ascending.
fn bounded_at(tokens: &[u32]) -> Result<PolicySpec, SpecError> {
    let mut document = PolicySpec::preset("prefix_hash")?;
    document.view.boundaries = tokens.to_vec();
    Ok(document)
}

/// Asks, and reads the pick's first target as the row of the fleet it
/// names, with the leases the answer carries: the caller's to keep or to
/// give up. A lease names its request's session: an answer whose lease
/// names one where the ask states no routing key, or none where it states
/// one, is refused.
async fn leased(harness: &mut Harness, ask: DecideRequest) -> Result<(Picked, Vec<u64>), String> {
    let keyed = ask
        .facts
        .as_ref()
        .is_some_and(|facts| facts.session_key.is_some());
    let answer = answered(harness, ask).await;
    let Ok(Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    })) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    if let Some(lease) = leases
        .iter()
        .find(|lease| (lease.session_key != 0) != keyed)
    {
        return Err(format!(
            "a lease names a session where its ask states a key, and there alone: {lease:?}"
        ));
    }
    let leases = leases.iter().map(|lease| lease.id).collect();
    let target = pick.targets.first().ok_or("a pick with no target")?;
    Ok((row_of(harness, target.id, target.dp_rank)?, leases))
}

/// The pause before an ask is asked again.
const AGAIN: Duration = Duration::from_millis(25);

/// Whether `error` is a session its controller ended because it stopped
/// leading, or leads under a later term: a gateway opens its session again.
fn ended(error: &client::Error) -> bool {
    matches!(error, client::Error::Status(status) if status.code() == Code::Aborted)
}

/// One ask's answer. A controller that elects over a lease file is asked
/// again while it answers that it does not lead. A leader whose renewal
/// lands past its renew deadline leads nobody until one lands in time, and
/// the lease's store on a busy host can hold a renewal that long. A gateway
/// asks again, and so does the case, inside the room. A controller that
/// leads itself is asked once: a case counts what its client sent.
async fn answered(harness: &mut Harness, ask: DecideRequest) -> Result<Answer, client::Error> {
    let began = Instant::now();
    loop {
        match harness.client().decide(ask.clone()).await {
            Ok(Answer::Busy(busy))
                if harness.elects()
                    && busy.reason() == busy::Reason::NotLeader
                    && began.elapsed() < ROOM =>
            {
                tokio::time::sleep(AGAIN).await;
            }
            answer => return answer,
        }
    }
}

/// Gives `leases` up on the session, as a gateway whose client left, and
/// waits until the store's file holds `left` leases of the gateway.
async fn given_up(harness: &mut Harness, leases: &[u64], left: usize) -> Result<(), String> {
    for &lease in leases {
        let released = Released {
            what: Some(released::What::LeaseId(lease)),
            reason: released::Reason::ClientGone.into(),
            ..Released::default()
        };
        let reported = match harness.client().released(released).await {
            // A leader that took its renew deadline whole ended its
            // sessions, and a gateway hears it at its next report: it opens
            // its session again and says the report there. Boxed: the
            // dial's future is large, and every pick of a case would carry
            // it.
            Err(error) if harness.elects() && ended(&error) => {
                Box::pin(harness.redial()).await?;
                harness.client().released(released).await
            }
            reported => reported,
        };
        reported.map_err(|status| format!("a lease is not given up: {status:?}"))?;
    }
    // Every pick of the case passes here, and a report is one frame the
    // session reads in its turn: the wait steps by a millisecond, where
    // the harness's own step would cost each pick twenty-five.
    let gone = async {
        while harness.leases(GATEWAY) != left {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    };
    tokio::time::timeout(ROOM, gone)
        .await
        .map_err(|_| format!("the leases given up leave the store's file: not within {ROOM:?}"))
}

/// Asks, and reads the pick's first target as the row of the fleet it
/// names. The pick's lease is given up on the session and has left the
/// store's file before the next ask reads the rows: the file holds what it
/// held before the ask.
async fn picked(harness: &mut Harness, ask: DecideRequest) -> Result<Picked, String> {
    // The asserts by count lean on a decision that books nothing past its
    // own ask: a lease on file is a request in flight on its worker.
    let before = harness.leases(GATEWAY);
    let (row, leases) = leased(harness, ask).await?;
    given_up(harness, &leases, before).await?;
    Ok(row)
}

/// Asks as [`picked`] asks and keeps the pick's lease: its dispatch is
/// named, so it books its row until the case gives it up.
async fn kept(harness: &mut Harness, ask: DecideRequest) -> Result<(Picked, u64), String> {
    let (row, leases) = leased(harness, ask).await?;
    let &[lease] = leases.as_slice() else {
        return Err(format!("a pick to keep carries one lease: {leases:?}"));
    };
    harness.dispatch(lease).await?;
    Ok((row, lease))
}

/// The row of the fleet a target names by its id and its rank.
fn row_of(harness: &Harness, id: u64, dp_rank: u32) -> Result<Picked, String> {
    match harness.named(id) {
        Some((engine, rank)) if dp_rank == u32::from(rank) => Ok(Picked { engine, rank, id }),
        _ => Err(format!(
            "a target that is no row of the fleet: {id} at rank {dp_rank}"
        )),
    }
}

/// Asks, and reads every target of the pick as a row of the fleet, best
/// first, with the id of the one lease the answer carries. The lease is the
/// caller's, to dispatch on a target and to give up.
async fn offered(harness: &mut Harness, ask: DecideRequest) -> Result<(Vec<Picked>, u64), String> {
    let answer = harness.client().decide(ask).await;
    let Ok(Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    })) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let [lease] = leases.as_slice() else {
        return Err(format!("a pick to dispatch carries one lease: {leases:?}"));
    };
    let mut rows = Vec::with_capacity(pick.targets.len());
    for target in &pick.targets {
        rows.push(row_of(harness, target.id, target.dp_rank)?);
    }
    Ok((rows, lease.id))
}

/// The whole case's bound: its waits sum to seconds on an idle host.
const CASE: Duration = Duration::from_secs(180);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_fleet_registers_its_loads_follow_and_a_dead_engine_leaves_the_healthy_set() {
    let case = async {
        let started = Instant::now();
        let cpu = cpu_seconds();
        let mut harness = Harness::start().await.expect("the harness starts");

        // The session. The leader welcomed the client's hello under the name
        // every call of the client states.
        let welcome = harness.welcome();
        assert_eq!(welcome.protocol, PROTOCOL_VERSION, "{welcome:?}");
        assert_eq!(welcome.controller_id, "fleet-controller");
        let told = welcome
            .principal
            .as_ref()
            .expect("a welcome names its session's principal");
        assert_eq!(
            (told.subject.as_str(), told.scope(), told.method()),
            (
                "fleet-gateway",
                principal::Scope::Gateway,
                principal::Method::AsTold
            ),
            "{told:?}"
        );
        assert!(
            welcome.grace_period_ms > 0 && welcome.leader_poll_ms > 0,
            "a welcome names its periods: {welcome:?}"
        );
        assert!(
            harness.term_floor() >= 1,
            "the client's floor stands at an answered term before it asks again"
        );

        // Ask: who leads, end to end.
        let leader = harness.leader().await.expect("the leader call");
        assert!(leader.is_leader, "a lone controller leads: {leader:?}");
        assert_eq!(leader.controller_id, "fleet-controller");
        assert!(
            leader.term >= 1,
            "a leader answers under a term: {leader:?}"
        );
        assert_eq!(
            harness.term_floor(),
            leader.term,
            "the client's floor is the leader's term"
        );

        // Register. Every rank is a row at once: none healthy before a
        // probe heard its engine, none with a load before a poll read one.
        let engines = harness.register(&FLEET).await.expect("the fleet registers");
        let (busy, dead) = (engines[1], engines[4]);
        let (laden, spare) = (engines[5], engines[6]);
        let ranks: usize = FLEET
            .iter()
            .filter(|shape| shape.model == MODEL)
            .map(|shape| usize::from(shape.ranks))
            .sum();
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        assert_eq!(pool.len(), ranks, "every rank is a row: {pool:?}");
        // A controller that probes by itself asks an engine as its row
        // arrives: nothing is asked yet only where the stand-in probes.
        assert!(
            PRODUCT_PROBES || pool.iter().all(|row| !row.healthy && row.load.is_none()),
            "nothing is known of an engine nobody asked: {pool:?}"
        );

        // Feed. The probe's verdict makes a rank ready and the controller's
        // own load monitor then polls its engine.
        harness.feed();
        let ready = within(
            "every rank is healthy and holds a load report",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness
                    .pool(MODEL, Role::Single)
                    .is_ok_and(|pool| pool.iter().all(|row| row.healthy && row.load.is_some()))
            },
        )
        .await
        .expect("the fleet turns ready");
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        assert_eq!(pool.len(), ranks, "{pool:?}");
        for row in &pool {
            let load = row.load.expect("a polled row");
            assert!(
                load.max_running > 0,
                "an engine reports its admission window: {row:?}"
            );
            assert_eq!(
                (load.running, load.queued),
                (0, 0),
                "an idle fleet: {row:?}"
            );
        }
        let mut two: Vec<u16> = pool
            .iter()
            .filter(|row| row.engine == dead)
            .map(|row| row.rank)
            .collect();
        two.sort_unstable();
        assert_eq!(two, [0, 1], "each rank of one engine is a row of its own");

        // Decide. A preset of the build is applied in the case's process,
        // since no call applies a spec yet. The gateway asks on its decide
        // stream, and the pick names a rank of the fleet.
        harness
            .apply("random")
            .expect("a preset of this build applies");
        picked(&mut harness, ask(b"first"))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert!(
            harness.term_floor() >= leader.term,
            "an answer's term never lowers the floor"
        );
        // The call's bounds are the client's own door. An id at the end of
        // its range is answered.
        picked(&mut harness, ask(&[7; 256]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        // A frame past the size leaves cut at its tail, and the cut is
        // counted: eight bytes a hash.
        let mut long = ask(b"long");
        let facts = long.facts.as_mut().expect("the ask states facts");
        facts.block_hashes = (0u64..).take(8_300).collect();
        let over = long.encoded_len() - FRAME;
        picked(&mut harness, long)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            harness.client().hashes_cut(),
            over.div_ceil(8),
            "{over} bytes over"
        );
        // A frame the call refuses ends the stream by the field's name, and
        // the next ask opens a new one.
        let mut wrong = ask(b"wrong");
        wrong.facts.as_mut().expect("the ask states facts").phase = 99;
        let refused = harness.client().decide(wrong).await;
        assert!(
            matches!(&refused, Err(client::Error::Status(status))
                if status.code() == Code::InvalidArgument
                    && status.message().contains("facts.phase")),
            "{refused:?}"
        );
        picked(&mut harness, ask(b"after"))
            .await
            .expect("a pick answers an ask over a ready fleet");

        // The loads follow the fleet: requests held on one engine reach its
        // row from that engine's own report, and no other row.
        harness
            .hold(busy, HELD as usize)
            .await
            .expect("the held requests");
        harness
            .hold(laden, HELD as usize)
            .await
            .expect("the held requests");
        let followed = within(
            "the busy engine's row reads its held requests",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter().any(|row| {
                        row.engine == busy && row.load.is_some_and(|load| load.running == HELD)
                    })
                })
            },
        )
        .await
        .expect("the loads follow");
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        for row in pool.iter().filter(|row| row.engine != busy) {
            assert_eq!(
                row.load.map(|load| load.running),
                Some(0),
                "only the busy engine runs requests: {row:?}"
            );
        }

        // A second pool, on the wire whose load report counts resident
        // tokens: the requests held on one of its two engines reach that
        // engine's row as the tokens it holds, and the other's row counts
        // none. The resident metric's assert below needs both rows counted:
        // a row that counts nothing is priced from its peer.
        within(
            "the counted pool's rows read their engines' resident tokens",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(COUNTED, Role::Single).is_ok_and(|pool| {
                    pool.len() == 2
                        && pool.iter().all(|row| {
                            row.healthy
                                && row.load.is_some_and(|load| {
                                    load.resident_counted
                                        && (load.resident_tokens > 0) == (row.engine == laden)
                                })
                        })
                })
            },
        )
        .await
        .expect("the resident tokens follow");

        // A decision is given the load monitor's snapshot, which the pool
        // above read. The monitor polls each engine on its own clock: it
        // has polled both pools once more since the pool read so.
        let told = Instant::now();
        monitored(&harness, &engines, told)
            .await
            .expect("the monitor's reports follow");

        // A load-led decision follows the loads the engines told: under the
        // least-load preset no pick names the engine that holds requests.
        harness
            .apply("least_load")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("load", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "a load-led pick leaves the engine that holds requests: {pick:?}"
            );
        }

        // Two draws compared by the loads the engines told: under the
        // power-of-two preset no pick names the engine that holds requests.
        // Unread, the loads leave the busy row one pick in six, so 64 asks
        // miss it less than once in 100,000 runs.
        harness
            .apply("power_of_two")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("two", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "of two draws the better load is taken: {pick:?}"
            );
        }

        // The first row the filters keep, in the host's order: the host's
        // order is the order its workers registered in. Under the
        // passthrough preset every pick names the model's first registered
        // row, and an ask that names it tried is answered with the second
        // registered.
        harness
            .apply("passthrough")
            .expect("a preset of this build applies");
        let registered: [(usize, u16); 2] = [(engines[0], 0), (engines[1], 0)];
        let first = picked(&mut harness, nth("through", 0))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            (first.engine, first.rank),
            registered[0],
            "a passthrough pick is the first row the host registered: {first:?}"
        );
        for n in 1..8 {
            let again = picked(&mut harness, nth("through", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, first,
                "a passthrough pick is one row whatever its seed"
            );
        }
        let mut retry = nth("through", 8);
        let facts = retry.facts.as_mut().expect("the ask states facts");
        facts.tried = vec![first.id];
        facts.attempt = 1;
        let next = picked(&mut harness, retry)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            (next.engine, next.rank),
            registered[1],
            "past its tried first row a passthrough pick is the second registered: {next:?}"
        );

        // The overload veto, and the gateway's lift of it. An operator
        // patches the engine that holds requests to the least KV usage a
        // threshold states, a permille, and a sample that reads its held
        // prompts there reads it overloaded: a rotation names every row
        // but its own, and an ask that lifts the veto is turned over every
        // row again. The patch taken back, the engine is a pick as before.
        // The verdict is read in the picks: the case waits for a turn that
        // leaves the engine out, and for a pick that names it again.
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        let least = OverloadThresholds {
            kv_usage_permille: Some(1),
            ..OverloadThresholds::default()
        };
        harness
            .patch_overload(busy, least)
            .await
            .expect("an operator's patch is taken");
        left_out(&mut harness, "spared", ranks, busy)
            .await
            .expect("an engine at its patched limit leaves the rotation");
        let mut vetoed = HashSet::new();
        for n in (0u64..).take(2 * ranks) {
            let pick = picked(&mut harness, nth("veto", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(pick.engine, busy, "an overloaded engine is no pick");
            vetoed.insert(pick);
        }
        assert_eq!(
            vetoed.len(),
            ranks - 1,
            "a rotation under the veto names every other row: {vetoed:?}"
        );
        let mut lifted = HashSet::new();
        for n in (0u64..).take(2 * ranks) {
            let mut ask = nth("lift", n);
            ask.constraints.get_or_insert_default().overload_lifted = true;
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            lifted.insert(pick);
        }
        assert_eq!(
            lifted.len(),
            ranks,
            "the gateway's lift of the veto keeps the overloaded row: {lifted:?}"
        );
        for n in (0u64..).take(ranks) {
            let pick = picked(&mut harness, nth("stood", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "the veto stood through both rotations: {pick:?}"
            );
        }
        harness
            .patch_overload(busy, OverloadThresholds::default())
            .await
            .expect("an operator's patch is taken");
        named_again(&mut harness, "regained", busy)
            .await
            .expect("the patch taken back, the engine is a pick again");

        // A rotation names every row before it names one again: under the
        // round-robin preset two turns of asks name each row twice.
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        let mut turns: HashMap<Picked, u32> = HashMap::new();
        for n in (0u64..).take(2 * ranks) {
            let pick = picked(&mut harness, nth("turn", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            *turns.entry(pick).or_default() += 1;
        }
        assert!(
            turns.len() == ranks && turns.values().all(|asks| *asks == 2),
            "two turns of a rotation name each row twice: {turns:?}"
        );

        // The gateway's own word narrows a decision before a stage runs. A
        // row it just failed to connect to is excluded: four turns of a
        // rotation name every other row and never that one.
        let shunned = first;
        let mut others = HashSet::new();
        for n in (0u64..).take(4 * ranks) {
            let mut ask = nth("shun", n);
            ask.constraints.get_or_insert_default().excluded = vec![shunned.id];
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(pick, shunned, "a row the gateway excludes is no pick");
            others.insert(pick);
        }
        assert_eq!(
            others.len(),
            ranks - 1,
            "a rotation under an exclusion names every other row: {others:?}"
        );
        // A hint of no candidates and a wire no row speaks each leave a
        // pool that has rows with none: the answer is a shed for want of
        // endpoints, under the constraint's own name and no stage's.
        let mut hinted = nth("subset-hint", 0);
        hinted.constraints.get_or_insert_default().subset_hint = Some(0);
        let mut narrowed = nth("wires", 0);
        narrowed.constraints.get_or_insert_default().wire_modes = u32::from(WireModes::HTTP.bits());
        for (name, ask) in [("subset-hint", hinted), ("wires", narrowed)] {
            let answer = harness.client().decide(ask).await;
            assert!(
                matches!(&answer, Ok(Answer::Decision(Decision { outcome: Some(Outcome::Shed(shed)), .. }))
                    if shed.reason() == ShedReason::NoEndpoints && shed.emptied_by == name),
                "a constraint that leaves no row sheds under its own name, {name}: {answer:?}"
            );
        }
        // The rows' own wire stated, the rotation goes on as with none.
        let mut wired = nth("wired", 0);
        wired.constraints.get_or_insert_default().wire_modes = u32::from(WireModes::GRPC.bits());
        picked(&mut harness, wired)
            .await
            .expect("a pick answers a gateway that states the rows' wire");

        // A decision by count reads what the decisions before it left on
        // file. Under the two-draw preset's form by the fewest requests
        // each pick keeps its lease, and every round of asks names each
        // row once: the engine that holds requests too, which no pick by
        // the expected wait names.
        let fewest = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        harness
            .apply_document(fewest)
            .expect("a form of this build's preset applies");
        let mut booked = Vec::new();
        for n in (0u64..).take(2 * ranks) {
            let pick = kept(&mut harness, nth("fewest", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            booked.push(pick);
        }
        for round in booked.chunks(ranks) {
            let named: HashSet<Picked> = round.iter().map(|(row, _)| *row).collect();
            assert_eq!(
                named.len(),
                ranks,
                "a round by the fewest requests names each row once: {round:?}"
            );
        }
        assert_eq!(
            harness.leases(GATEWAY),
            booked.len(),
            "a kept lease stands on the store's file"
        );
        // A lease given up frees its row. Once the leases on the engine
        // that holds requests are gone, no lease books that engine and two
        // book every other row: by count the next picks are its own,
        // whatever its wait.
        let (freed, booked): (Vec<_>, Vec<_>) =
            booked.into_iter().partition(|(row, _)| row.engine == busy);
        let freed: Vec<u64> = freed.into_iter().map(|(_, lease)| lease).collect();
        given_up(&mut harness, &freed, booked.len())
            .await
            .expect("the leases given up leave the file");
        for n in 0..2 {
            let pick = picked(&mut harness, nth("freed", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                pick.engine, busy,
                "the row with the fewest requests in flight is taken: {pick:?}"
            );
        }

        // Two draws compared by the requests in flight: an ask that names
        // every row tried but that engine's and one idle engine's draws
        // those two, and under the form by requests each pick is the
        // engine that holds requests.
        let (other, _) = booked.first().copied().expect("a kept lease");
        let mut tried: Vec<u64> = booked
            .iter()
            .map(|(row, _)| row.id)
            .filter(|id| *id != other.id)
            .collect();
        tried.sort_unstable();
        tried.dedup();
        let requests = PolicySpec::power_of_two(PowerOfTwoMetric::Requests)
            .expect("a form of this build's preset");
        harness
            .apply_document(requests)
            .expect("a form of this build's preset applies");
        for n in 0..8 {
            let mut retry = nth("requests", n);
            let facts = retry.facts.as_mut().expect("the ask states facts");
            facts.tried.clone_from(&tried);
            facts.attempt = 1;
            let pick = picked(&mut harness, retry)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                pick.engine, busy,
                "of two draws the one with fewer requests in flight is taken: {pick:?}"
            );
        }

        // The fewest requests in flight take a request that brings no key:
        // under the prefix-hash preset each keyless pick is that engine
        // too, one pick in six where the count is not read.
        harness
            .apply("prefix_hash")
            .expect("a preset of this build applies");
        for n in 0..8 {
            let pick = picked(&mut harness, nth("keyless", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                pick.engine, busy,
                "a request with no key goes where the fewest requests are in flight: {pick:?}"
            );
        }
        let rest: Vec<u64> = booked.into_iter().map(|(_, lease)| lease).collect();
        given_up(&mut harness, &rest, 0)
            .await
            .expect("the leases given up leave the file");

        // A session stays where its key was first placed. Under the manual
        // preset a key's first decision places it at random and its later
        // asks name that row whatever their seeds; 63 more keys do not
        // share one row.
        harness
            .apply("manual")
            .expect("a preset of this build applies");
        let stays = picked(&mut harness, keyed(nth("stay", 0), 1))
            .await
            .expect("a pick answers an ask over a ready fleet");
        for n in 1..8 {
            let again = picked(&mut harness, keyed(nth("stay", n), 1))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, stays,
                "a session key stays on the worker of its first decision"
            );
        }
        let mut homes = HashSet::new();
        for key in 2..=ASKS {
            let pick = picked(&mut harness, keyed(nth("homes", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            homes.insert(pick);
        }
        assert!(
            homes.len() > 1,
            "{} session keys do not share one worker: {homes:?}",
            ASKS - 1
        );

        // The session is where the gateway sent its request. A key's first
        // decision claims the first target of its pick; the gateway
        // dispatches on the second, and the key's later asks name that
        // row. The store tells the engine of a dispatch under the turn of
        // the lease's file, and its read of the lease takes that turn: once
        // the store reads the lease dispatched, the next decision reads the
        // move.
        let sticky = ASKS + 1;
        let (offer, sent_lease) = offered(&mut harness, keyed(nth("sent", 0), sticky))
            .await
            .expect("a pick answers an ask over a ready fleet");
        let sent = offer
            .get(1)
            .copied()
            .expect("a key's first placement offers a second target");
        harness
            .dispatch_at(sent_lease, 1)
            .await
            .expect("the store reads the lease dispatched");
        for n in 1..8 {
            let again = picked(&mut harness, keyed(nth("sent", n), sticky))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, sent,
                "a key first sent to its second target stays where it was sent"
            );
        }

        // A key served on its failover keeps its worker. With its row
        // excluded by the gateway the key is placed on another row and
        // dispatched there; the exclusion over, it is back on its own row,
        // and excluded again it returns to that failover.
        let away = |run: &str, n: u64| {
            let mut ask = keyed(nth(run, n), sticky);
            ask.constraints = Some(proto::Constraints {
                excluded: vec![sent.id],
                ..Default::default()
            });
            ask
        };
        let (offer, away_lease) = offered(&mut harness, away("away", 0))
            .await
            .expect("a pick answers an ask over a ready fleet");
        let failover = offer.first().copied().expect("a pick names a target");
        assert_ne!(failover, sent, "a row the gateway excludes is no pick");
        harness
            .dispatch_at(away_lease, 0)
            .await
            .expect("the store reads the lease dispatched");
        let back = picked(&mut harness, keyed(nth("back", 0), sticky))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            back, sent,
            "its own row routable again, a key is back on it"
        );
        for n in 1..8 {
            let again = picked(&mut harness, away("away", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, failover,
                "a key away from its row returns to its failover"
            );
        }
        given_up(&mut harness, &[sent_lease, away_lease], 0)
            .await
            .expect("the leases given up leave the file");

        // A new generation places a pinned key anew. An engine's table of
        // sessions starts empty with its generation, and nothing carries a
        // pin across an apply yet. Eight keys are each dispatched where
        // they were placed; under the preset applied anew each of them,
        // and the key sent to its second target above, is asked again.
        // Placed afresh, a key names its old row once in as many asks as
        // the fleet has rows.
        let mut pinned = vec![(sticky, sent)];
        let mut held = Vec::new();
        for key in (2 * ASKS..).take(8) {
            let (offer, lease) = offered(&mut harness, keyed(nth("pinned", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            let row = offer.first().copied().expect("a pick names a target");
            harness
                .dispatch_at(lease, 0)
                .await
                .expect("the store reads the lease dispatched");
            pinned.push((key, row));
            held.push(lease);
        }
        given_up(&mut harness, &held, 0)
            .await
            .expect("the leases given up leave the file");
        harness
            .apply("manual")
            .expect("a preset of this build applies");
        for (key, row) in pinned {
            let again = picked(&mut harness, keyed(nth("anew", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            if KEEPS_PINS_ACROSS_AN_APPLY {
                assert_eq!(again, row, "a new generation asks a pinned key on its row");
            }
        }

        // A lease's count follows its dispatch. By the fewest requests a
        // kept lease books the first target of its pick; dispatched on the
        // second it books that row instead, and the asks after it, each
        // kept, name every other row once: the first target among them.
        let fewest = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        harness
            .apply_document(fewest)
            .expect("a form of this build's preset applies");
        let (offer, moved) = offered(&mut harness, keyed(nth("moved", 0), sticky + 1))
            .await
            .expect("a pick answers an ask over a ready fleet");
        let claimed = offer.first().copied().expect("a pick names a target");
        let served = offer
            .get(1)
            .copied()
            .expect("a pick by the fewest requests offers a second target");
        harness
            .dispatch_at(moved, 1)
            .await
            .expect("the store reads the lease dispatched");
        let mut moved = vec![moved];
        let mut round = HashSet::new();
        for n in (1u64..).take(ranks - 1) {
            let (row, lease) = kept(&mut harness, nth("moved", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            moved.push(lease);
            round.insert(row);
        }
        assert!(
            round.len() == ranks - 1 && round.contains(&claimed) && !round.contains(&served),
            "a lease dispatched on its second target books that row and frees the first: \
             {claimed:?} then {served:?}, the round {round:?}"
        );
        given_up(&mut harness, &moved, 0)
            .await
            .expect("the leases given up leave the file");

        // The KV events follow the fleet: the busy engine's held requests
        // stored blocks, its stream's batches reached its rank's cursor, and
        // its row's index holds every block the engine holds.
        // The engine stores on while its requests are held, so the row moves
        // for the whole case. The wait keeps the row it judged at rest and
        // the asserts below read that row: a second read can meet the next
        // batch between the two streams.
        let mut at_rest = None;
        let stored = within(
            "the busy engine's blocks reach its row's index",
            PROBE_INTERVAL + ROOM,
            || {
                // The listener's stream and the store's own are two: they
                // are read at rest, the cursor at the last sequence the
                // listener heard and the index holding what it holds.
                at_rest = harness.heard(busy).first().copied().filter(|row| {
                    row.indexed > 0
                        && row.indexed == row.held
                        && row
                            .cursor
                            .is_some_and(|cursor| cursor.applied >= 1 && cursor.applied == row.last)
                });
                at_rest.is_some()
            },
        )
        .await
        .map_err(|bound| {
            format!(
                "{bound}; what refuses its stream: {:?}",
                harness.refused(busy)
            )
        })
        .expect("the KV events follow");
        let fed = at_rest.expect("the wait keeps the row it judged at rest");
        assert!(
            fed.last >= 1
                && fed.cursor.is_some_and(|cursor| {
                    cursor.applied >= fed.last && !cursor.replay_pending && !cursor.degraded
                }),
            "the cursor stands at what the stream delivered: {fed:?}"
        );
        assert!(
            fed.indexed >= 1 && fed.indexed == fed.held,
            "the index holds each block the engine's events left it holding: {fed:?}"
        );
        assert_eq!(
            fed.counts,
            Some(LaneCounts::default()),
            "no store came without its parent and no block was refused: {fed:?}"
        );
        // A prompt goes where its blocks are held. Under the cache-aware
        // preset an ask that states the head the busy engine's events left
        // it holding names that engine, which its load turns every load-led
        // pick away from; the ask after it states no block and is placed by
        // the load, on another engine: no ask reads the holders of the one
        // before it.
        let head = harness.head(busy);
        assert!(!head.is_empty(), "the busy engine holds a prompt's head");
        harness
            .apply("cache_aware")
            .expect("a preset of this build applies");
        for n in 0..8 {
            let mut ask = nth("holder", n);
            let facts = ask.facts.as_mut().expect("the ask states facts");
            facts.block_hashes.clone_from(&head);
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            if READS_CACHE_EVENTS {
                assert_eq!(
                    pick.engine, busy,
                    "a prompt goes to the engine that holds its blocks: {pick:?}"
                );
            }
            let pick = picked(&mut harness, nth("unheld", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "a prompt that states no block is placed by the load: {pick:?}"
            );
        }
        for engine in engines.iter().filter(|engine| **engine != busy) {
            for idle in harness.heard(*engine) {
                assert!(
                    idle.last == 0 && idle.indexed == 0 && idle.held == 0,
                    "an engine that stored no block published none: engine {engine} {idle:?}"
                );
            }
        }
        assert_eq!(harness.gaps(), 0, "no stream named a gap");
        assert_eq!(
            harness.snapshot_chunks(),
            0,
            "an engine's own stream carries no snapshot chunk"
        );

        // A session key keeps its ring worker over the overload veto. A
        // key of the engine that holds requests is found among 256, an
        // operator patches that engine as before, and with the engine
        // vetoed the key's asks still name its row. A key keeps its worker
        // over a veto, so the ring gives no witness that one stands: the
        // rotation preset is applied on either side of the key's asks to
        // show it, and the ring preset between them for the asks. The
        // patch is the row's own and stands under every preset.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let mut held = None;
        for key in (ASKS + 1..).take(KEYS) {
            let pick = picked(&mut harness, keyed(nth("held", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            if pick.engine == busy {
                held = Some((key, pick));
                break;
            }
        }
        let (key, row) = held.expect("of 256 routing keys one is the busy engine's");
        harness
            .patch_overload(busy, least)
            .await
            .expect("an operator's patch is taken");
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        left_out(&mut harness, "ring-spared", ranks, busy)
            .await
            .expect("an engine at its patched limit leaves the rotation");
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        for n in 0..8 {
            let again = picked(&mut harness, keyed(nth("kept", n), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, row,
                "a session key keeps its ring worker over the veto"
            );
        }
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        for n in (0u64..).take(ranks) {
            let pick = picked(&mut harness, nth("ring-stood", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "the veto stood through the key's asks: {pick:?}"
            );
        }
        harness
            .patch_overload(busy, OverloadThresholds::default())
            .await
            .expect("an operator's patch is taken");
        named_again(&mut harness, "ring-regained", busy)
            .await
            .expect("the patch taken back, the engine is a pick again");

        // A keyed ask returns to its worker: under the ring preset one
        // routing key names one row at each ask, and 64 keys do not all
        // share it. On a ring of even shares a correct ring puts 64 keys
        // on one row less than once in 10^40 runs.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let owner = picked(&mut harness, keyed(nth("key", 0), 1))
            .await
            .expect("a pick answers an ask over a ready fleet");
        for n in 1..8 {
            let again = picked(&mut harness, keyed(nth("key", n), 1))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(again, owner, "one routing key returns to one worker");
        }
        let mut owners = HashSet::from([owner]);
        for n in 1..ASKS {
            let other = picked(&mut harness, keyed(nth("keys", n), n + 1))
                .await
                .expect("a pick answers an ask over a ready fleet");
            owners.insert(other);
        }
        assert!(
            owners.len() > 1,
            "{ASKS} routing keys do not share one worker: {owners:?}"
        );
        // A retry that names the worker it tried is answered with another.
        let mut retry = keyed(nth("retry", 0), 1);
        let facts = retry.facts.as_mut().expect("the ask states facts");
        facts.tried = vec![owner.id];
        facts.attempt = 1;
        let next = picked(&mut harness, retry)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_ne!(next, owner, "a retry leaves the worker it names as tried");

        // A key in either form names one session. A gateway states a
        // routing key plain or by its session's value, the key within the
        // model and the phase by the contract's formula. The lease of a
        // plain key carries that value, and an ask that states the value
        // carries it too and returns to the plain key's worker.
        assert_eq!(
            (
                print_of("conversation-7"),
                session_of("conversation-7", "model-a")
            ),
            (0xd449_9ecc_c740_2a12, 0x6e88_ff71_cc98_2a42),
            "this case's formulas give the contract's own example"
        );
        let value = session_of("conversation-7", MODEL);
        let mut plain = nth("plain", 0);
        let facts = plain.facts.as_mut().expect("the ask states facts");
        facts.session_key = Some(Identity {
            form: Some(Form::Plain("conversation-7".to_owned())),
        });
        let (by_text, text_session) = sessioned(&mut harness, plain)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            text_session, value,
            "a plain key's lease carries the session the contract's formula gives"
        );
        let (by_value, value_session) = sessioned(&mut harness, keyed(nth("value", 0), value))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            value_session, value,
            "a key stated by its session's value carries that value on its lease"
        );
        assert_eq!(
            by_value, by_text,
            "a key returns to one worker in either form"
        );

        // A key names a rank of a data-parallel group. The engine of two
        // ranks is two rows of the ring, each at the positions of its own
        // registration key: with every other engine excluded by the
        // gateway, 64 routing keys are placed on that engine and not all
        // on one rank of it, and each key asked again names its rank.
        let apart: Vec<u64> = engines
            .iter()
            .filter(|engine| **engine != dead)
            .map(|engine| harness.worker(*engine).map(|id| id.0))
            .collect::<Result<_, _>>()
            .expect("every engine registered its first rank");
        let grouped = |run: &str, n: u64| {
            let mut ask = keyed(nth(run, n), n + 1);
            ask.constraints.get_or_insert_default().excluded = apart.clone();
            ask
        };
        let mut placed = Vec::new();
        for n in 0..ASKS {
            let row = picked(&mut harness, grouped("group", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                row.engine, dead,
                "with every other engine excluded a key names the group: {row:?}"
            );
            placed.push(row);
        }
        let shares: HashSet<u16> = placed.iter().map(|row| row.rank).collect();
        assert_eq!(
            shares.len(),
            2,
            "{ASKS} routing keys do not share one rank of a data-parallel group: {shares:?}"
        );
        for (n, row) in (0..ASKS).zip(&placed) {
            let again = picked(&mut harness, grouped("regroup", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, *row,
                "a routing key returns to its rank of a data-parallel group"
            );
        }

        // A prompt returns to its worker. Under the prefix preset an ask
        // with no routing key is keyed by its prompt's head, the last whole
        // block within 256 tokens: prompts that share a head name one row
        // whatever follows it, and 64 heads do not all share that row.
        harness
            .apply("prefix_hash")
            .expect("a preset of this build applies");
        let home = picked(&mut harness, prompted(nth("head", 0), &[(HEAD, 1)]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        for n in 1..8 {
            let longer = prompted(nth("head", n), &[(HEAD, 1), (HEAD, ASKS + n)]);
            let again = picked(&mut harness, longer)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(again, home, "one head returns to one worker");
        }
        let mut homes = HashSet::from([home]);
        for n in 1..ASKS {
            let other = picked(&mut harness, prompted(nth("heads", n), &[(HEAD, n + 1)]))
                .await
                .expect("a pick answers an ask over a ready fleet");
            homes.insert(other);
        }
        assert!(
            homes.len() > 1,
            "{ASKS} heads do not share one worker: {homes:?}"
        );
        // A routing key leads the prefix: whatever its head, a keyed ask
        // names the worker the plain ring named for its key.
        for n in 0..8 {
            let led = keyed(prompted(nth("led", n), &[(HEAD, n + 1)]), 1);
            let led = picked(&mut harness, led)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(led, owner, "a routing key leads the prefix");
        }
        // A retry that names its head's worker as tried walks to the ring's
        // next one, the same worker at each retry.
        let mut walked = HashSet::new();
        for n in 0..8 {
            let mut retry = prompted(nth("walk", n), &[(HEAD, 1), (HEAD, ASKS + n)]);
            let facts = retry.facts.as_mut().expect("the ask states facts");
            facts.tried = vec![home.id];
            facts.attempt = 1;
            let next = picked(&mut harness, retry)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(next, home, "a head's retry leaves the worker it tried");
            walked.insert(next);
        }
        assert_eq!(
            walked.len(),
            1,
            "past the worker it tried a head walks to one worker: {walked:?}"
        );
        // A worker past its bounded load is passed over. A kept lease
        // books the head's worker, and the ring keeps the worker while its
        // count stands inside both margins over the pool's mean: of six
        // rows with ten for the absolute margin, through thirteen leases.
        // The next ask of that head has no owner left and goes where the
        // fewest requests are in flight; its leases given up, the head
        // returns.
        let mut loaded = Vec::new();
        for n in 0..LOADED {
            let (row, lease) = kept(&mut harness, prompted(nth("load", n), &[(HEAD, 1)]))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                row, home,
                "inside its bounded load a head's worker keeps it"
            );
            loaded.push(lease);
        }
        let over = picked(&mut harness, prompted(nth("over", 0), &[(HEAD, 1)]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_ne!(
            over, home,
            "past its bounded load a head's worker is passed over"
        );
        given_up(&mut harness, &loaded, 0)
            .await
            .expect("the kept leases are given up");
        let back = picked(&mut harness, prompted(nth("back", 0), &[(HEAD, 1)]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            back, home,
            "its load given up, a head returns to its worker"
        );
        // A boundary of the document's view moves the key. Prompts that
        // share a stem of 64 tokens and nothing past it are 64 heads to the
        // preset; under a view whose boundary stands at the stem's end
        // they are keyed there, and name one row.
        let mut apart = HashSet::new();
        for n in 0..ASKS {
            let ask = prompted(nth("stems", n), &[(STEM, 1), (HEAD - STEM, n + 2)]);
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            apart.insert(pick);
        }
        assert!(
            apart.len() > 1,
            "past one stem {ASKS} heads do not share one worker: {apart:?}"
        );
        let at_stem = bounded_at(&[STEM * BLOCK]).expect("a document of this build's preset");
        harness
            .apply_document(at_stem)
            .expect("a document whose view names a boundary applies");
        let mut together = HashSet::new();
        for n in 0..ASKS {
            let ask = prompted(nth("bounded", n), &[(STEM, 1), (HEAD - STEM, n + 2)]);
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            together.insert(pick);
        }
        assert_eq!(
            together.len(),
            1,
            "a boundary of the view keys a prompt at its stem: {together:?}"
        );
        // A prompt that grew past a loaded worker lands where its stem is
        // held. Under a view of two boundaries a prompt is keyed at the
        // deepest it reaches. With its worker there past its bounded load
        // the ring tries the boundary below, and the stem's worker takes
        // the request.
        let two =
            bounded_at(&[STEM * BLOCK, HEAD * BLOCK]).expect("a document of this build's preset");
        harness
            .apply_document(two)
            .expect("a document whose view names two boundaries applies");
        let stem = picked(&mut harness, prompted(nth("stem", 0), &[(STEM, 1)]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        let mut found = None;
        for n in 0..ASKS {
            let ask = prompted(nth("grown", n), &[(STEM, 1), (HEAD - STEM, n + 2)]);
            let pick = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            if pick != stem {
                found = Some((n, pick));
                break;
            }
        }
        let (grown, deep) =
            found.expect("of 64 prompts on one stem, another worker than the stem's holds one");
        let mut loaded = Vec::new();
        for n in 0..LOADED {
            let ask = prompted(nth("deep", n), &[(STEM, 1), (HEAD - STEM, grown + 2)]);
            let (row, lease) = kept(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                row, deep,
                "inside its bounded load a prompt's worker keeps it"
            );
            loaded.push(lease);
        }
        for n in 0..8 {
            let ask = prompted(nth("below", n), &[(STEM, 1), (HEAD - STEM, grown + 2)]);
            let below = picked(&mut harness, ask)
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                below, stem,
                "past its loaded worker a prompt lands where its stem is held"
            );
        }
        given_up(&mut harness, &loaded, 0)
            .await
            .expect("the kept leases are given up");

        // Each model decides under the profile its binding names. One
        // document binds the counted model to the two-draw preset's
        // resident form and every other model to a rotation: two turns of
        // asks for the first model name each of its rows twice, and every
        // ask for the counted model names the engine that holds no tokens,
        // the lesser work of its pool's two rows.
        let bound = by_model().expect("a document of this build's presets");
        harness
            .apply_document(bound)
            .expect("a document of two bindings applies");
        let mut turns: HashMap<Picked, u32> = HashMap::new();
        for n in (0u64..).take(2 * ranks) {
            let pick = picked(&mut harness, nth("bound", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            *turns.entry(pick).or_default() += 1;
            let counted = picked(&mut harness, for_model(nth("counted", n), COUNTED))
                .await
                .expect("a pick answers an ask over a ready fleet");
            if READS_KEPT_LOADS {
                assert_eq!(
                    counted.engine, spare,
                    "the lesser resident work is taken: {counted:?}"
                );
            }
        }
        assert!(
            turns.len() == ranks && turns.values().all(|asks| *asks == 2),
            "under its own binding a rotation names each row twice: {turns:?}"
        );

        // A prefill request goes to the engine that owns its size. Under
        // the bucket preset the prefill pool's three engines split the
        // span of sizes evenly at their first decision. With a balance
        // rule no booking reaches, eight asks of one size name one row
        // whatever their seeds, and a size from each third of the span
        // names a row of its own.
        within("the prefill pool's rows read healthy", ROOM, || {
            harness
                .pool(SIZED, Role::Prefill)
                .is_ok_and(|pool| pool.len() == 3 && pool.iter().all(|row| row.healthy))
        })
        .await
        .expect("the prefill pool turns ready");
        harness
            .apply_document(by_size().expect("a form of this build's preset"))
            .expect("a form of this build's preset applies");
        let third = bucket::SPAN / 3;
        let mut owners = Vec::new();
        for part in 0u32..3 {
            let tokens = part * third + third / 2;
            let first = u64::from(part) * 8;
            let owner = picked(&mut harness, sized(nth("sized", first), tokens))
                .await
                .expect("a pick answers an ask over a ready fleet");
            for n in 1..8 {
                let again = picked(&mut harness, sized(nth("sized", first + n), tokens))
                    .await
                    .expect("a pick answers an ask over a ready fleet");
                assert_eq!(again, owner, "requests of like size meet on one engine");
            }
            owners.push(owner);
        }
        let apart: HashSet<Picked> = owners.iter().copied().collect();
        assert!(
            apart.len() == 3 && owners.iter().all(|row| engines[7..].contains(&row.engine)),
            "a size from each third of the span has an engine of the pool to itself: {owners:?}"
        );

        // The balance rule spills on what the decisions booked. Under the
        // preset as the catalog spells it an idle pool places a size on
        // its owner, and the tokens of that decision are booked on the
        // engine: the next two asks of the same size go to the two engines
        // that booked nothing.
        harness
            .apply("bucket")
            .expect("a preset of this build applies");
        let mut round = Vec::new();
        for n in 0..3 {
            let pick = picked(&mut harness, sized(nth("spill", n), third / 2))
                .await
                .expect("a pick answers an ask over a ready fleet");
            round.push(pick);
        }
        assert_eq!(
            round.first(),
            owners.first(),
            "an idle pool places a size on its owner"
        );
        let spilled: HashSet<Picked> = round.iter().copied().collect();
        assert_eq!(
            spilled.len(),
            3,
            "an engine a decision booked gives way to one that booked nothing: {round:?}"
        );

        // The death. The kill waits for a pool that reads its engine
        // healthy, so the death alone moves its rows. Its promise is held
        // by order below, whatever the host's clock does: the time from the
        // kill to the pool that reads the death is a number of the case.
        within("the engine to kill reads healthy", ROOM, || {
            harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                pool.iter()
                    .filter(|row| row.engine == dead)
                    .all(|row| row.healthy)
            })
        })
        .await
        .expect("a live engine reads healthy");
        let killed = harness.mark();
        let death = harness.kill(dead).await.expect("the kill");
        // The death's verdicts are the statuses the store moved from the
        // kill on, one a rank, whoever probes. The liveness rule moves a
        // not-ready row on to failed three failures later: that step
        // follows the death, it is not its verdict.
        let verdicts = |harness: &Harness| {
            let mut moved = harness.moved(killed).expect("the store's tail");
            moved.retain(|moved| moved.engine == dead && moved.to != WorkerStatus::Failed);
            moved
        };
        within(
            "the death is written, one verdict a rank",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || verdicts(&harness).len() >= 2,
        )
        .await
        .expect("the death is written");
        let written = death.at.elapsed();
        assert_eq!(
            verdicts(&harness)
                .iter()
                .map(|moved| (moved.from, moved.to))
                .collect::<Vec<_>>(),
            [(WorkerStatus::Ready, WorkerStatus::NotReady); 2],
            "one verdict a rank from the kill on"
        );
        within(
            "a dead engine's rows read unhealthy and every other row healthy",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness
                    .pool(MODEL, Role::Single)
                    .is_ok_and(|pool| pool.iter().all(|row| row.healthy == (row.engine != dead)))
            },
        )
        .await
        .expect("the death is seen");
        let seen = death.at.elapsed();
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        assert_eq!(
            pool.len(),
            ranks,
            "a dead engine's rows stay rows: {pool:?}"
        );
        // Where the stand-in probes, its turns tell which dial wrote: the
        // first that heard the engine dead, a turn in flight at the kill
        // or the first one begun after the death.
        if !PRODUCT_PROBES {
            let first = harness
                .first_unhealthy(dead, death.asked)
                .expect("a dial heard the engine dead");
            assert!(
                first <= death.turn + 1,
                "the first turn after the death dialed it: turn {first} after {death:?}"
            );
            let wrote: Vec<_> = harness
                .verdicts()
                .into_iter()
                .filter(|verdict| {
                    verdict.engine == dead
                        && verdict.at >= death.at
                        && verdict.to != WorkerStatus::Failed
                })
                .collect();
            assert_eq!(wrote.len(), 2, "one verdict a rank: {wrote:?}");
            for verdict in &wrote {
                assert_eq!(
                    (verdict.from, verdict.to),
                    (WorkerStatus::Ready, WorkerStatus::NotReady),
                    "{verdict:?}"
                );
                assert_eq!(
                    verdict.turn, first,
                    "the first dial that heard the engine dead wrote it: {verdict:?} after {death:?}"
                );
            }
        }
        // No decision names the dead engine: from the probe that heard it
        // dead its rows are no candidates.
        harness
            .apply("random")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("dead", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, dead,
                "a dead engine is named by no decision: {pick:?}"
            );
        }

        // The dead engine's event stream broke with it, and its subscriber
        // keeps what refuses it since: the dial of an engine that is gone.
        within(
            "a dead engine's stream is refused at its dial",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness
                    .refused(dead)
                    .is_some_and(|refusal| refusal.starts_with("the dial"))
            },
        )
        .await
        .expect("the subscriber keeps what refuses it");

        // The hang. An engine that stops answering with its sockets open
        // costs a probe turn its dial bound and no more: that turn writes
        // the engine's row not ready, and no live row moves.
        let hung = engines[2];
        let hanged = harness.mark();
        let hang = Instant::now();
        harness.hang(hung).expect("the hang");
        within(
            "a hung engine's rows read unhealthy and every live row healthy",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter()
                        .all(|row| row.healthy == (row.engine != dead && row.engine != hung))
                })
            },
        )
        .await
        .expect("the hang is seen");
        let hung_seen = hang.elapsed();
        let moved = harness.moved(hanged).expect("the store's tail");
        assert_eq!(
            moved
                .iter()
                .filter(|moved| moved.engine == hung && moved.to != WorkerStatus::Failed)
                .map(|moved| (moved.from, moved.to))
                .collect::<Vec<_>>(),
            [(WorkerStatus::Ready, WorkerStatus::NotReady)],
            "one verdict for the hung engine's one rank: {moved:?}"
        );
        // The dialer's turn that waited out the bound went on to every
        // engine after the hung one, and from the hang on no live row
        // moved.
        within(
            "a dial waits out the hung engine",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || harness.first_unhealthy(hung, 0).is_some(),
        )
        .await
        .expect("a dial of the hung engine ends");
        let turn = harness
            .first_unhealthy(hung, 0)
            .expect("the turn that waited out the bound");
        within("the turn that waited out the bound ends", ROOM, || {
            harness.dialed(turn).len() == FLEET.len()
        })
        .await
        .expect("the turn goes on");
        assert_eq!(
            harness.dialed(turn),
            [
                (engines[0], DialEnd::Answered(true)),
                (busy, DialEnd::Answered(true)),
                (hung, DialEnd::TimedOut),
                (engines[3], DialEnd::Answered(true)),
                (dead, DialEnd::Failed),
                (laden, DialEnd::Answered(true)),
                (spare, DialEnd::Answered(true)),
                (engines[7], DialEnd::Answered(true)),
                (engines[8], DialEnd::Answered(true)),
                (engines[9], DialEnd::Answered(true)),
            ],
            "the turn that waited out the bound went on to every engine"
        );
        let mut others = harness.moved(hanged).expect("the store's tail");
        others.retain(|moved| moved.engine != hung && moved.engine != dead);
        assert!(
            others.is_empty(),
            "no live row moves at the hang: {others:?}"
        );
        // A hung engine is not a dead one: its listener still takes a
        // connection.
        harness
            .connect(hung)
            .await
            .expect("a hung engine takes a connection");
        // So the bound is spent on a fresh dial too: the next turn dials the
        // hung engine anew and waits it out again.
        within(
            "a hung engine is dialed again",
            2 * (PROBE_INTERVAL + DIAL_TIMEOUT) + ROOM,
            || harness.ends(hung, turn).len() >= 2,
        )
        .await
        .expect("the next turn dials the hung engine");
        assert_eq!(
            harness.ends(hung, turn)[..2],
            [DialEnd::TimedOut, DialEnd::TimedOut],
            "each dial of a hung engine waits out its bound"
        );

        // The reports. A gateway that decides by itself tells the controller
        // each request's life on its session, and the store holds what a
        // decision's precedence rule reads of it: one gateway's own
        // completions of class ok, a worker at a time.
        let live = engines[0];
        let asked = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers to the end");
        within(
            "the live engine's row holds this gateway's completion",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, asked),
        )
        .await
        .expect("the completion is read");
        let class = harness.send(dead).await.expect("a dead engine is sent");
        assert_eq!(class, Class::ConnectFailed, "a dead engine connects none");
        let class = tokio::time::timeout(ROOM, harness.send(hung))
            .await
            .expect("a hung engine's send ends at the dial bound")
            .expect("a hung engine is sent");
        assert_eq!(class, Class::Timeout, "a hung engine answers nothing");
        // A lease no leader issued is given up: not on file, no error, and
        // the session outlives it: the next request's reports are applied.
        harness
            .client()
            .released(Released {
                what: Some(released::What::LeaseId(u64::MAX)),
                reason: released::Reason::ClientGone.into(),
                ..Released::default()
            })
            .await
            .expect("a release is reported");
        let again = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers a second time");
        within(
            "the session outlives a release of a lease not on file",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, again),
        )
        .await
        .expect("the second completion is read");
        // One session's frames are applied in their order, so the dead and
        // the hung engine's reports were read by now: neither wrote a
        // completion of class ok for this gateway. A completion of another
        // class writes nothing, so these hold the class each frame carries
        // and nothing more.
        assert!(
            !harness.completed_ok(GATEWAY, dead, asked),
            "a dead engine's row holds no completion of class ok"
        );
        assert!(
            !harness.completed_ok(GATEWAY, hung, asked),
            "a hung engine's row holds no completion of class ok"
        );
        // One gateway's completions are its own.
        assert!(
            !harness.completed_ok("another-gateway", live, asked),
            "another gateway reads none of this gateway's completions"
        );
        // A frame that reads wrong ends the session of the gateway that sent
        // it, by its field's name: a second gateway gives up neither a
        // lease nor a ticket.
        let mut other =
            Client::connect(&format!("http://{}", harness.gateway()), "another-gateway")
                .await
                .expect("a second gateway connects");
        other
            .released(Released::default())
            .await
            .expect("the frame is queued");
        let ended = other
            .session_end()
            .await
            .expect("the controller ends the session inside the client's bound");
        assert_eq!(ended.code(), Code::InvalidArgument, "{ended:?}");
        assert!(
            ended.message().contains("Released.what"),
            "the refusal names the field: {ended:?}"
        );
        // The session tells the status it ended with at each ask, and to
        // each report from the first read on.
        let again = other.session_end().await.expect("the end is told again");
        assert_eq!(
            again.code(),
            Code::InvalidArgument,
            "a second ask reads the status the session ended with: {again:?}"
        );
        let told = other
            .released(Released::default())
            .await
            .expect_err("a report into the ended session is refused");
        assert!(
            matches!(&told, client::Error::Status(status) if status.code() == Code::InvalidArgument),
            "a report into the ended session tells the status it ended with: {told:?}"
        );
        // The first gateway's session stands: its next reports are applied.
        let last = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers a third time");
        within(
            "the first gateway's session stands",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, last),
        )
        .await
        .expect("the third completion is read");

        // The blocking form's report leaves with no call after it: its
        // runtime drives the connection between calls. The client is held,
        // and asked nothing more, until the store has read its completion:
        // its drop would not wait for a report still queued.
        let served = harness.worker(live).expect("the live engine's row").0;
        let address = format!("http://{}", harness.gateway());
        let reported = Instant::now();
        let (hold, held) = mpsc::channel::<()>();
        let (told, reporting) = oneshot::channel();
        std::thread::spawn(move || {
            let report =
                blocking::Client::connect(&address, "blocking-gateway").and_then(|mut client| {
                    client.completed(Completed {
                        lease_id: 1,
                        side: Side::Single.into(),
                        served,
                        class: Class::Ok.into(),
                        ..Completed::default()
                    })?;
                    Ok(client)
                });
            match report {
                Ok(client) => {
                    let _ = told.send(Ok(()));
                    let _ = held.recv();
                    drop(client);
                }
                Err(error) => {
                    let _ = told.send(Err(error.to_string()));
                }
            }
        });
        tokio::time::timeout(ROOM, reporting)
            .await
            .expect("the blocking client reports inside its bound")
            .expect("the client's thread sends its answer")
            .expect("the blocking client's report");
        within(
            "the blocking form's report leaves with no call after it",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok("blocking-gateway", live, reported),
        )
        .await
        .expect("the blocking gateway's completion is read");
        drop(hold);

        // The Numbers: one process holds the controller, the engines and
        // the feeds, so its CPU and resident memory are a ceiling over the
        // service's, never the service's own.
        let wall = started.elapsed();
        let mut out = std::io::stdout();
        let rows: usize = FLEET.iter().map(|shape| usize::from(shape.ranks)).sum();
        writeln!(out, "FLEET_ENGINES {} FLEET_RANKS {rows}", FLEET.len()).expect("stdout");
        writeln!(
            out,
            "FLEET_PROBE_INTERVAL_MS {}",
            PROBE_INTERVAL.as_millis()
        )
        .expect("stdout");
        writeln!(out, "FLEET_DIAL_TIMEOUT_MS {}", DIAL_TIMEOUT.as_millis()).expect("stdout");
        writeln!(out, "FLEET_READY_MS {}", ready.as_millis()).expect("stdout");
        writeln!(out, "FLEET_LOADS_FOLLOW_MS {}", followed.as_millis()).expect("stdout");
        writeln!(out, "FLEET_KV_FOLLOW_MS {}", stored.as_millis()).expect("stdout");
        writeln!(out, "FLEET_KV_BATCHES {}", fed.last).expect("stdout");
        writeln!(out, "FLEET_KV_BLOCKS {}", fed.indexed).expect("stdout");
        writeln!(out, "FLEET_DEATH_WRITTEN_MS {}", written.as_millis()).expect("stdout");
        writeln!(out, "FLEET_DEATH_SEEN_MS {}", seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_HANG_SEEN_MS {}", hung_seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_WALL_MS {}", wall.as_millis()).expect("stdout");
        writeln!(out, "FLEET_CPU_MS {:.0}", (cpu_seconds() - cpu) * 1_000.0).expect("stdout");
        writeln!(out, "FLEET_RSS_KIB {}", vm_rss_kib()).expect("stdout");

        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// The model the pair's engines serve.
const PAIRED: &str = "fleet-paired-model";

/// The pair's fleet: two engines prefill and two decode, each of one rank
/// on the first wire.
const PAIR_FLEET: [Shape; 4] = [
    Shape {
        model: PAIRED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Decode,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Decode,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
];

/// The pair plan's document, as an operator writes it: a request at either
/// phase of a pair is decided under the plan, each side by the least load.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": {
    "side": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
  },
  "bindings": [
    {
      "match": { "phase": ["prefill", "decode"] },
      "plan": "pd-pair",
      "profiles": { "prefill": "side", "decode": "side" }
    }
  ]
}"#;

/// The `n`th ask of the pair's run: for the pair's model, at the plan's
/// prefill.
fn at_prefill(n: u64) -> DecideRequest {
    let mut ask = for_model(nth("pair", n), PAIRED);
    if let Some(facts) = &mut ask.facts {
        facts.phase = proto::Phase::Prefill.into();
    }
    ask
}

/// Asks at a pair's prefill, and reads the pick's first target as the two
/// rows of the fleet it names: the prefill side's, then the decode side's.
/// The answer carries a lease a side under one id, the prefill side's
/// first. Both are given up by that id, and have left the store's file
/// before the next ask reads the rows.
async fn paired(harness: &mut Harness, ask: DecideRequest) -> Result<(Picked, Picked), String> {
    let before = harness.leases(GATEWAY);
    let answer = harness.client().decide(ask).await;
    let Ok(Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    })) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let sides: Vec<(u64, Side)> = leases
        .iter()
        .map(|lease| (lease.id, lease.side()))
        .collect();
    let &[(id, Side::Prefill), (decode, Side::Decode)] = sides.as_slice() else {
        return Err(format!(
            "a pair's pick carries a lease a side, the prefill side's first: {leases:?}"
        ));
    };
    if id != decode {
        return Err(format!("a pair's leases are one id's: {leases:?}"));
    }
    let target = pick.targets.first().ok_or("a pick with no target")?;
    let pair = target
        .pair
        .as_ref()
        .ok_or("a pair's target names no decode side")?;
    let rows = (
        row_of(harness, target.id, target.dp_rank)?,
        row_of(harness, pair.decode_id, pair.decode_dp_rank)?,
    );
    given_up(harness, &[id], before).await?;
    Ok(rows)
}

/// A pair plan answers a request that enters at its prefill once, for both
/// sides, over a fleet of its own: two engines that prefill and two that
/// decode. Each side's worker is the one whose engine reports no requests,
/// and the answer's two leases are one id's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pair_plan_answers_its_prefill_for_both_sides_by_the_loads_their_engines_told() {
    let case = async {
        let mut harness = Harness::start().await.expect("the harness starts");
        let engines = harness
            .register(&PAIR_FLEET)
            .await
            .expect("the pair's fleet registers");
        let (held_prefill, idle_prefill) = (engines[0], engines[1]);
        let (idle_decode, held_decode) = (engines[2], engines[3]);
        let sides = [(Role::Prefill, held_prefill), (Role::Decode, held_decode)];
        harness.feed();
        within(
            "each side's rows are healthy and hold a load report",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                sides.iter().all(|&(side, _)| {
                    harness.pool(PAIRED, side).is_ok_and(|pool| {
                        pool.len() == 2 && pool.iter().all(|row| row.healthy && row.load.is_some())
                    })
                })
            },
        )
        .await
        .expect("the pair's fleet turns ready");

        // One engine of each side holds requests: its own report says so,
        // and no other row of its side reads them.
        for (_, held) in sides {
            harness
                .hold(held, HELD as usize)
                .await
                .expect("the held requests");
        }
        within(
            "each side's held engine reads its requests, and no other",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                sides.iter().all(|&(side, held)| {
                    harness.pool(PAIRED, side).is_ok_and(|pool| {
                        pool.iter().all(|row| {
                            row.load
                                .is_some_and(|load| (load.running == HELD) == (row.engine == held))
                        })
                    })
                })
            },
        )
        .await
        .expect("the loads follow on both sides");

        // A decision is given the load monitor's snapshot, where the pools
        // above are the store's table. The monitor polls each engine on its
        // own clock: it has polled all four since the table read so.
        let told = Instant::now();
        monitored(&harness, &engines, told)
            .await
            .expect("the monitor's reports follow");

        // The pair plan's document is applied in the case's process. A
        // request enters the plan at its prefill and is answered once, for
        // both sides, and neither names the engine that holds requests.
        // Unread, a side's loads leave its pick to a draw of two: 64 asks
        // miss it less than once in 10^19 runs.
        let document = serde_json::from_str(PAIR_PLAN).expect("a well-formed document");
        harness
            .apply_document(document)
            .expect("the pair plan's document applies");
        for n in 0..ASKS {
            let (prefill, decode) = paired(&mut harness, at_prefill(n))
                .await
                .expect("a pair's pick answers a request at its prefill");
            assert_eq!(
                prefill.engine, idle_prefill,
                "a pair's prefill side leaves the engine that holds requests: {prefill:?}"
            );
            assert_eq!(
                decode.engine, idle_decode,
                "a pair's decode side leaves the engine that holds requests: {decode:?}"
            );
        }
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// The blocking form of the reference client asks the same started
/// controller from a thread that runs no runtime, and holds the same floor.
/// Its watch tells the controller's refusal of a request and, of one the
/// controller takes, the snapshot first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_blocking_client_asks_a_started_controller_from_a_thread_of_its_own() {
    let case = async {
        let harness = Harness::start().await.expect("the harness starts");
        let address = format!("http://{}", harness.gateway());
        let (answered, answer) = oneshot::channel();
        std::thread::spawn(move || {
            let asked =
                blocking::Client::connect(&address, "blocking-gateway").and_then(|mut client| {
                    let leader = client.leader()?;
                    let subject = client
                        .welcome()
                        .principal
                        .as_ref()
                        .map(|told| told.subject.clone());
                    let decided = client.decide(ask(b"blocking"))?;
                    // A resume under a term with no place is refused by its field.
                    let unplaced = WatchWorkersRequest {
                        resume_term: 1,
                        ..WatchWorkersRequest::default()
                    };
                    let refused = match client.watch_workers(unplaced) {
                        Err(client::Error::Status(status)) => {
                            Some((status.code(), status.message().to_owned()))
                        }
                        _ => None,
                    };
                    let mut watch = client.watch_workers(WatchWorkersRequest::default())?;
                    let told = client.next_of(&mut watch)?;
                    Ok((leader, client.term_floor(), subject, decided, refused, told))
                });
            let _ = answered.send(asked.map_err(|error| error.to_string()));
        });
        let (leader, floor, subject, decided, refused, told) = tokio::time::timeout(ROOM, answer)
            .await
            .expect("the blocking client answers inside its bound")
            .expect("the client's thread sends its answer")
            .expect("the blocking client's calls");
        assert!(leader.is_leader, "a lone controller leads: {leader:?}");
        assert_eq!(leader.controller_id, "fleet-controller");
        assert_eq!(floor, leader.term, "the floor is the leader's term");
        assert_eq!(
            subject.as_deref(),
            Some("blocking-gateway"),
            "the welcome names the gateway the client states"
        );
        // No worker serves the model: the controller decides over no list,
        // and the answer is a shed for want of endpoints under the stage
        // `none`, whatever document is applied.
        assert!(
            matches!(&decided, Answer::Decision(Decision { outcome: Some(Outcome::Shed(shed)), .. })
                if shed.reason() == ShedReason::NoEndpoints && shed.emptied_by == "none"),
            "{decided:?}"
        );
        // The watch: an open the controller refuses tells its status, and
        // the first event of one it takes is the snapshot of no worker.
        assert!(
            matches!(&refused, Some((Code::InvalidArgument, said)) if said.contains("resume_term")),
            "{refused:?}"
        );
        let told = told.expect("the watch tells a snapshot first");
        assert!(
            matches!(&told.event, Some(worker_event::Event::Snapshot(page))
                if page.last && page.workers.is_empty()),
            "{told:?}"
        );
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// Ready workers under a controller that leads itself.
async fn linked_fleet(gateway: &str, workers: usize) -> Result<(Harness, Link), String> {
    let mut harness = Harness::start().await?;
    harness.apply("round_robin")?;
    harness.register(&FLEET[..workers]).await?;
    harness.feed();
    within("the workers are ready", ROOM, || {
        harness
            .pool(MODEL, Role::Single)
            .is_ok_and(|pool| pool.len() == workers && pool.iter().all(|row| row.healthy))
    })
    .await?;
    let endpoint = Endpoint::from_shared(format!("http://{}", harness.gateway()))
        .map_err(|error| error.to_string())?;
    let link = Box::pin(Link::connect(endpoint, gateway))
        .await
        .map_err(|error| error.to_string())?;
    Ok((harness, link))
}

async fn linked_pick(link: &Link, request: DecideRequest) -> Result<Lease, String> {
    let answer = link.ask(request, ROOM).await;
    let Ok(Decision {
        leases,
        outcome: Some(Outcome::Pick(_)),
        ..
    }) = answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let [lease] = leases.as_slice() else {
        return Err(format!("a pick with the wrong leases: {leases:?}"));
    };
    Ok(*lease)
}

async fn leases_before_expiry(
    harness: &Harness,
    link: &Link,
    gateway: &str,
    left: usize,
    before: Instant,
) -> Result<(), String> {
    let span = Duration::from_millis(u64::from(link.welcome().dispatch_ttl_ms));
    within(
        "the store reaches the reported lease count before expiry",
        span.saturating_sub(before.elapsed()),
        || before.elapsed() < span && harness.leases(gateway) == left,
    )
    .await
    .map(|_| ())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_answers_sixty_four_asks_and_reports_the_end_of_each_lease() {
    const GATEWAY: &str = "fleet-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let mut asks = JoinSet::new();
        let count = usize::try_from(ASKS).expect("the number of asks fits");
        let ready = Arc::new(Barrier::new(count + 1));
        let before = Instant::now();
        for n in 0..ASKS {
            let link = link.clone();
            let ready = Arc::clone(&ready);
            asks.spawn(async move {
                let session = session_of(&n.to_string(), MODEL);
                ready.wait().await;
                let lease = linked_pick(&link, keyed(nth("link", n), session))
                    .await
                    .expect("the controller answers with a pick");
                assert_eq!(lease.session_key, session, "each ask gets its own decision");
                (lease.id, lease.side)
            });
        }
        ready.wait().await;
        let mut leases = HashMap::new();
        while let Some(answer) = asks.join_next().await {
            let (id, side) = answer.expect("the caller gets its answer");
            assert!(
                leases.insert(id, side).is_none(),
                "a distinct lease for every ask"
            );
        }
        assert_eq!(harness.leases(GATEWAY), count);
        for (lease_id, side) in leases {
            link.queue_frame(gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease_id)),
                reason: released::Reason::ClientGone.into(),
                side,
            }));
        }
        leases_before_expiry(&harness, &link, GATEWAY, 0, before)
            .await
            .expect("the reports end the leases");
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// A request can outlast the session its decision came over. The gateway
/// dials its link again, and what the request reports then reaches the
/// controller on the link's next session: a dispatched lease, which no
/// deadline ends, ends at that report.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dispatched_lease_is_ended_by_a_report_on_the_links_next_session() {
    const GATEWAY: &str = "fleet-redial";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let endpoint =
            Endpoint::from_shared(format!("http://{}", harness.gateway())).expect("a URI");
        // A request in flight holds a clone of the link.
        let held = link.clone();
        let sent = |lease: Lease| {
            gateway_frame::Frame::Dispatched(Dispatched {
                lease_id: lease.id,
                target_index: 0,
                side: lease.side,
                since_decision_micros: 0,
            })
        };
        let ended = |lease: Lease| {
            gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease.id)),
                reason: released::Reason::ClientGone.into(),
                side: lease.side,
            })
        };

        // The controller ends the session under a dispatched request: a
        // second hello on one stream is refused.
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        held.queue_frame(sent(first));
        within("the store reads the first lease dispatched", ROOM, || {
            harness.dispatched(GATEWAY, first.id)
        })
        .await
        .expect("the session said the dispatch");
        link.queue_frame(gateway_frame::Frame::Hello(Hello::default()));
        tokio::time::timeout(ROOM, link.closed())
            .await
            .expect("the session ends");
        assert!(!held.stands());

        // The request ends with no session standing. Its report waits, and
        // the link's next session says it.
        held.queue_frame(ended(first));
        assert_eq!(harness.leases(GATEWAY), 1, "no session said the end");
        Box::pin(link.redial(endpoint.clone(), GATEWAY))
            .await
            .expect("the link's next session");
        within("the report that waited ends the lease", ROOM, || {
            harness.leases(GATEWAY) == 0
        })
        .await
        .expect("the next session said the report");

        // A redial over a session that stands: the next session takes its
        // place, and the gateway's lease with it.
        let second = linked_pick(&held, ask(b"second"))
            .await
            .expect("a pick on the next session");
        held.queue_frame(sent(second));
        within("the store reads the second lease dispatched", ROOM, || {
            harness.dispatched(GATEWAY, second.id)
        })
        .await
        .expect("the session said the dispatch");
        Box::pin(link.redial(endpoint, GATEWAY))
            .await
            .expect("the link's third session");
        assert_eq!(
            harness.leases(GATEWAY),
            1,
            "the lease stands through the redial"
        );
        held.queue_frame(ended(second));
        within(
            "the report on the third session ends the lease",
            ROOM,
            || harness.leases(GATEWAY) == 0,
        )
        .await
        .expect("the third session said the report");
        assert!(held.stands());
        assert_eq!(held.frames_dropped(), 0);
        drop((link, held));
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_releases_the_lease_of_a_zero_budget_ask() {
    const GATEWAY: &str = "fleet-late-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let before = Instant::now();
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        assert!(matches!(
            link.ask(ask(b"late"), Duration::ZERO).await,
            Err(DecisionUnavailable::Late)
        ));
        let last = linked_pick(&link, ask(b"last")).await.expect("a last pick");
        // The three asks run on the same decider. The gap proves that the
        // middle ask booked a lease even if its release has arrived already.
        assert_eq!(last.id, first.id + 2);
        leases_before_expiry(&harness, &link, GATEWAY, 2, before)
            .await
            .expect("only the first and last leases remain");
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// A frame the call refuses ends the stream it is sent on, and a link's
/// stream is every caller's. The link refuses such a frame at its own door:
/// nothing is sent, the link stands, and the next ask is answered on the
/// stream that stood.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_refuses_a_frame_out_of_the_calls_ranges_and_stands() {
    const GATEWAY: &str = "fleet-ranged-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        let mut wrong = ask(b"wrong");
        wrong.facts.as_mut().expect("the ask states facts").phase = 99;
        let refused = link.ask(wrong, ROOM).await;
        assert!(
            matches!(&refused, Err(DecisionUnavailable::Unsent(error))
                if error.to_string().contains("facts.phase")),
            "{refused:?}"
        );
        assert!(link.stands(), "the link stands");
        let last = linked_pick(&link, ask(b"last"))
            .await
            .expect("a pick after the refusal");
        // Both picks ran on the same decider: no lease was booked between
        // them, so the refused frame reached no decider.
        assert_eq!(last.id, first.id + 1);
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handle_over_a_link_ends_each_lease_on_the_worker_that_served() {
    use smg_policy_controller::{
        engine::{PolicyEngine, Request, RequestFacts},
        tracking::{DecisionGuard, Outcome, Side, TransportClass},
    };

    const GATEWAY: &str = "fleet-handle";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 2))
            .await
            .expect("a ready fleet");
        for id in ["held-0", "held-1", "held-2"] {
            let asked = Request {
                id: id.as_bytes(),
                facts: RequestFacts {
                    model: MODEL,
                    budget_micros: 5_000_000,
                    ..RequestFacts::default()
                },
                constraints: Default::default(),
            };
            let started = Instant::now();
            let held = Box::pin(DecisionGuard::<&PolicyEngine>::asked(
                &link, &asked, started, ROOM,
            ))
            .await;
            let mut handle = held.expect("the controller's decision");
            let taken = handle.take(0).expect("a picked worker");
            let alternate = handle.decision().pick().expect("a pick").targets.as_slice()[1];
            let (picked, _) = harness
                .named(taken.id.0)
                .expect("the picked worker is in the fleet");
            let (served, _) = harness
                .named(alternate.id.0)
                .expect("the alternate is in the fleet");
            assert_ne!(served, picked);
            assert_eq!(harness.leases(GATEWAY), 1, "the controller filed the lease");
            let sent = Instant::now();
            handle.dispatched(sent);
            let (class, frames) = harness
                .serve_request(served, id.to_owned())
                .await
                .expect("the worker answers");
            assert_eq!(class, Class::Ok);
            assert!(frames > 0);
            let ended = Instant::now();
            let mut outcome =
                Outcome::reported(0, TransportClass::Ok).with_served(Side::Single, alternate.id);
            outcome.tokens_in = 8;
            outcome.tokens_out = frames;
            outcome.duration_micros =
                u32::try_from(ended.duration_since(sent).as_micros()).unwrap_or(u32::MAX);
            handle.completed(&outcome, ended);
            leases_before_expiry(&harness, &link, GATEWAY, 0, started)
                .await
                .expect("the completion ends the lease");
            within(
                "the completion is written on the served worker",
                ROOM,
                || harness.completed_ok(GATEWAY, served, started),
            )
            .await
            .expect("the controller records the named worker's completion");
            assert!(!harness.completed_ok(GATEWAY, picked, started));
        }
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A replica that does not lead refuses the client's session and names the
/// leader: two controllers elect over one lease file, and the client dials
/// each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_at_a_replica_that_does_not_lead_is_told_the_leader() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let mut leader = Replica::start("replica-a", Some(&lease))
            .await
            .expect("the first replica starts");
        leader
            .reaches(State::Leader)
            .await
            .expect("the first replica leads");
        let mut follower = Replica::start("replica-b", Some(&lease))
            .await
            .expect("the second replica starts");
        follower
            .reaches(State::Follower)
            .await
            .expect("the second replica follows");

        let at_follower = format!("http://{}", follower.gateway());
        let refused = tokio::time::timeout(ROOM, Client::connect(&at_follower, "pair-gateway"))
            .await
            .expect("the follower answers inside its bound");
        let address = match refused {
            Ok(welcomed) => Err(format!("it welcomed the gateway: {welcomed:?}")),
            Err(client::Error::NotLeader(address)) => Ok(address),
            Err(other) => Err(other.to_string()),
        }
        .expect("a replica that does not lead names the leader");
        assert_eq!(
            address,
            leader.gateway().to_string(),
            "the address is the leader's gateway listener"
        );

        let welcomed = tokio::time::timeout(
            ROOM,
            Client::connect(&format!("http://{address}"), "pair-gateway"),
        )
        .await
        .expect("the leader answers inside its bound")
        .expect("the leader welcomes the gateway its follower sent on");
        assert_eq!(welcomed.welcome().controller_id, "replica-a");
        drop(welcomed);

        follower.stop().await.expect("the follower stops clean");
        leader.stop().await.expect("the leader stops clean");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// A watcher of `store` over worker objects of its own, which follow their
/// rows' status as a gateway's do.
fn watcher_of(store: Arc<MemoryStore>) -> Watcher<Members> {
    let objects = Objects::new(Arc::new(WorkerRegistry::new())).with_status_followed();
    Watcher::new(store, Arc::new(objects))
}

/// What a watcher holds of its controller's workers, by id: each row's
/// facts, status and revision. Each row has its worker object, built at the
/// row's revision, and no other object stands.
fn watched(watcher: &Watcher<Members>) -> Vec<(WorkerId, WorkerFacts, WorkerStatus, u64)> {
    let (_, mut rows) = WorkersStore::all_rows(&**watcher.store());
    rows.sort_by_key(|row| row.id.0);
    let objects = watcher.objects();
    assert_eq!(objects.len(), rows.len(), "one worker object a row");
    let built = |id| Some(objects.standing(&objects.object(id)?)?.revision);
    rows.into_iter()
        .map(|row| {
            assert_eq!(built(row.id), Some(row.revision), "the row's object");
            (row.id, row.facts, row.status, row.revision)
        })
        .collect()
}

/// The status of the worker object a watcher holds for a row, if it holds
/// the row: what the watch told last, with nothing followed behind it.
fn status_of(watcher: &Watcher<Members>, id: WorkerId) -> Option<ObjectStatus> {
    let objects = watcher.objects();
    let object = objects.registry().get(&objects.object(id)?)?;
    Some(object.status())
}

/// Follows the stream until `stands` holds of its watcher, each event
/// inside the room of a wait.
async fn follows(
    following: &mut WatchStream<'_, Members>,
    mut stands: impl FnMut(&Watcher<Members>) -> bool,
) -> Result<(), String> {
    while !stands(following.watcher()) {
        let taken = tokio::time::timeout(ROOM, following.next())
            .await
            .map_err(|_| "the watch told nothing inside its room".to_owned())?
            .map_err(|ended| format!("the stream is followed no further: {ended:?}"))?;
        if !taken {
            return Err("the watch ended with no status".to_owned());
        }
    }
    Ok(())
}

/// A gateway that watches holds the controller's rows. A snapshot tells the
/// fleet as it registered. The tail tells each status the prober writes, an
/// engine that registers behind the snapshot and a dead engine's status. A
/// stream opened after the last place taken is told the tail alone. At one
/// place, the watcher holds what one that started from a later snapshot
/// holds. The later snapshot keeps a lease already on file in its store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_watcher_holds_the_fleets_rows_as_the_controller_tells_them() {
    use smg_policy_controller::{
        store::{GatewayId, LeaseRecord, LeaseState, LeasesStore, PickRecord},
        tracking::{Lease, LeaseId},
    };

    let case = async {
        let mut harness = Harness::start().await.expect("the harness starts");
        let engines = harness
            .register(&FLEET[..3])
            .await
            .expect("the fleet registers");
        let ids: Vec<WorkerId> = engines
            .iter()
            .map(|engine| harness.worker(*engine).expect("the engine's row"))
            .collect();
        let dead = |watcher: &Watcher<Members>, id: WorkerId| {
            status_of(watcher, id).is_some_and(|status| status != ObjectStatus::Ready)
        };

        // The snapshot tells the fleet as it registered. Where the stand-in
        // probes, no prober has started yet and every row is pending. A
        // controller that probes by itself may have readied a row already.
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let opened = watcher.open(harness.client()).await;
        let mut stream = opened.expect("the watch opens");
        let told = follows(&mut stream, |held| held.at().is_some()).await;
        told.expect("a snapshot is told");
        for id in &ids {
            let status = status_of(stream.watcher(), *id);
            assert!(
                status == Some(ObjectStatus::Pending)
                    || (PRODUCT_PROBES && status == Some(ObjectStatus::Ready)),
                "a row of the snapshot reads {status:?}"
            );
        }

        // The prober starts: the tail tells each engine it heard.
        harness.feed();
        let ready = |watcher: &Watcher<Members>| {
            ids.iter()
                .all(|id| status_of(watcher, *id) == Some(ObjectStatus::Ready))
        };
        let told = follows(&mut stream, ready).await;
        told.expect("the fleet is told ready");

        // An engine that registers behind the snapshot.
        let late = harness.register(&FLEET[3..4]).await;
        let late = late.expect("one more engine registers")[0];
        let late = harness.worker(late).expect("the late engine's row");
        let ready =
            |watcher: &Watcher<Members>| status_of(watcher, late) == Some(ObjectStatus::Ready);
        let told = follows(&mut stream, ready).await;
        told.expect("the late engine is told ready");

        // A dead engine: the prober's word reaches the watcher.
        harness.kill(engines[0]).await.expect("an engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[0])).await;
        told.expect("the death is told");

        // A stream opened after the last place taken: the tail alone.
        drop(stream);
        let opened = watcher.open(harness.client()).await;
        let mut stream = opened.expect("the watch opens again");
        harness
            .kill(engines[1])
            .await
            .expect("a second engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[1])).await;
        told.expect("the second death is told");
        assert_eq!(
            stream.watcher().snapshots(),
            1,
            "no snapshot behind a place"
        );

        // A watcher that starts now takes a snapshot. Both are then told
        // one more death, and at its place the first holds what the second
        // holds.
        let store = Arc::new(MemoryStore::new());
        let leases: &dyn LeasesStore = &*store;
        let gateway = GatewayId(1);
        let now = Instant::now();
        leases
            .issue(LeaseRecord {
                lease: Lease {
                    id: LeaseId(1),
                    ..Lease::default()
                },
                gateway,
                request: 1,
                target: late,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 0,
                issued_at: now,
                dispatch_deadline: now + ROOM,
                state: LeaseState::ISSUED,
            })
            .expect("filed");
        assert_eq!(leases.live(gateway), 1);
        let mut fresh = watcher_of(Arc::clone(&store));
        let opened = fresh.open(harness.client()).await;
        let mut snapshot = opened.expect("a second watch opens");
        let told = follows(&mut snapshot, |held| held.at().is_some()).await;
        told.expect("a snapshot is told");
        assert_eq!(leases.live(gateway), 1);
        harness.kill(engines[2]).await.expect("a third engine dies");
        let told = follows(&mut stream, |held| dead(held, ids[2])).await;
        told.expect("the third death is told on the tail");
        let told = follows(&mut snapshot, |held| dead(held, ids[2])).await;
        told.expect("the third death is told behind the snapshot");
        let (first, second) = (stream.watcher(), snapshot.watcher());
        assert_eq!(first.at(), second.at(), "both stand at the death's place");
        assert_eq!(watched(first), watched(second));
        assert_eq!(watched(first).len(), 4, "four engines of one rank");
        assert_eq!((first.snapshots(), second.snapshots()), (1, 1));

        drop((stream, snapshot));
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_answers_beside_its_watch_and_the_watch_outlives_it() {
    const GATEWAY: &str = "fleet-link-watch";
    let case = async {
        let (mut harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let first = harness.worker(0).expect("the first worker's row");
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let mut stream = watcher
            .open_on(&link)
            .await
            .expect("the link opens a watch");
        follows(&mut stream, |held| held.at().is_some())
            .await
            .expect("the watch takes a snapshot");
        assert_eq!(watched(stream.watcher()).len(), 1);
        assert_eq!(
            status_of(stream.watcher(), first),
            Some(ObjectStatus::Ready)
        );

        let tail = follows(&mut stream, |held| watched(held).len() == 2);
        let ask_and_register = async {
            linked_pick(&link, ask(b"watching"))
                .await
                .expect("the link answers while the watch waits");
            harness
                .register(&FLEET[1..2])
                .await
                .expect("a worker joins the fleet");
        };
        let (told, ()) = tokio::join!(biased; tail, ask_and_register);
        told.expect("the watch takes the new row on its tail");
        drop(stream);

        let mut stream = watcher
            .open_on(&link)
            .await
            .expect("the link resumes the watch");
        link.queue_frame(gateway_frame::Frame::Hello(Hello::default()));
        tokio::time::timeout(ROOM, link.closed())
            .await
            .expect("the second hello ends the link");
        drop(link);

        harness.kill(0).await.expect("the first worker dies");
        follows(&mut stream, |held| {
            status_of(held, first).is_some_and(|status| status != ObjectStatus::Ready)
        })
        .await
        .expect("the watch takes the status after the link ends");
        assert_eq!(stream.watcher().snapshots(), 1);
        drop(stream);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A watch whose controller goes silent, with its connection still open,
/// ends: the client keeps its channel alive, and a peer that answers no
/// ping ends the streams on it. A relay stands between the two. At a word
/// it stops relaying and closes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_watch_ends_when_its_controller_goes_silent() {
    let case = async {
        let harness = Harness::start().await.expect("the harness starts");
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let relay = listener.local_addr().expect("the relay's address");
        let controller = harness.gateway();
        let silent = Arc::new(Notify::new());
        let word = Arc::clone(&silent);
        let mut relayed = JoinSet::new();
        relayed.spawn(async move {
            let (mut near, _) = listener.accept().await.expect("the client dials");
            let far = TcpStream::connect(controller).await;
            let mut far = far.expect("the controller listens");
            tokio::select! {
                _ = tokio::io::copy_bidirectional(&mut near, &mut far) => {}
                () = word.notified() => std::future::pending::<()>().await,
            }
        });

        let address = format!("http://{relay}");
        let connected = Client::connect(&address, "silent-gateway").await;
        let mut client = connected.expect("the client connects through the relay");
        let mut watcher = watcher_of(Arc::new(MemoryStore::new()));
        let opened = watcher.open(&mut client).await;
        let mut stream = opened.expect("the watch opens");
        let first = tokio::time::timeout(ROOM, stream.next()).await;
        assert!(matches!(first, Ok(Ok(true))), "the snapshot: {first:?}");

        silent.notify_one();
        // A ping goes out one keepalive after the last frame read, and the
        // channel ends one answer's wait after a ping with no answer.
        let bound = client::KEEPALIVE + client::ANSWER_WAIT + Duration::from_secs(5);
        let ended = tokio::time::timeout(bound, stream.next())
            .await
            .expect("a silent controller ends the stream inside its bound");
        assert!(matches!(ended, Err(WatchStreamEnd::Client(_))), "{ended:?}");

        relayed.abort_all();
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A controller of a host's own build over the fleet: its catalog holds a
/// scorer in code, a WebAssembly guest's and a served plugin's beside the
/// stock stages, and the document of its spec file names the three.
#[cfg(feature = "wasm")]
mod hosted {
    use std::{collections::HashSet, sync::Arc};

    use smg_policy_controller::{
        engine::{Candidate, Scorer, TieBand},
        plugins::{
            grpc::{self, GrpcPluginSpec, GrpcScorer, Transport},
            shared::OpenDecisions,
            wasm::{self, PluginSpec, WasmScorer},
        },
        stages::{Lifecycle, StageRegistry},
    };

    use super::{
        for_model, given_up, grpc_plugin::TestPlugin, kept, nth, picked, wasm_guest, within,
        Harness, Role, Shape, ASKS, CASE, COUNTED, FLEET, MODEL, POLL_INTERVAL, PROBE_INTERVAL,
        ROOM,
    };

    /// The model the served plugin scores: a pool of its own.
    const SERVED: &str = "fleet-served-model";

    /// The guest's stage, by the name the host's document writes.
    const GUEST: &str = "fewest-in-flight";

    /// The served plugin's stage, by its name.
    const REMOTE: &str = "fewest-in-flight-served";

    /// The scorer in code, by its name.
    const NAMESPACED: &str = "of-namespace";

    /// How many times a load, or a round of asks, is made. The host ends a
    /// guest's `init` at a frame of the wall clock and a plugin's stage
    /// that misses its frame steps aside: a runner that holds a thread for
    /// that long ends a call that did nothing wrong, and the load or the
    /// round is made again.
    const TRIES: u64 = 10;

    /// A host's scorer in code: of the rows it is handed, in their order,
    /// the one the request's cache namespace counts to scores one and every
    /// other row nothing. No stock stage reads the namespace: it reaches a
    /// stock decision as the block hashes a gateway folds it into.
    #[derive(Debug)]
    struct OfNamespace;

    impl Lifecycle for OfNamespace {}

    impl Scorer for OfNamespace {
        fn name(&self) -> &'static str {
            NAMESPACED
        }

        fn score(
            &self,
            facts: &smg_policy_controller::engine::RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let rows = out.len().max(1);
            let turn = usize::try_from(facts.namespace_seed).map_or(0, |seed| seed % rows);
            for (place, score) in out.iter_mut().enumerate() {
                *score = if place == turn { 1.0 } else { 0.0 };
            }
            TieBand::NORMALIZED
        }
    }

    /// The host's document: the counted model's requests under the guest's
    /// scorer, the served model's under the served plugin's, every other
    /// model's under the scorer in code.
    fn document() -> String {
        format!(
            r#"{{ "apiVersion": "smg.policy/v1",
  "profiles": {{
    "namespaced": {{ "filter": ["healthy"], "score": ["{NAMESPACED}"], "pick": "best" }},
    "guest": {{ "filter": ["healthy"], "score": ["{GUEST}"], "pick": "best" }},
    "served": {{ "filter": ["healthy"], "score": ["{REMOTE}"], "pick": "best" }}
  }},
  "bindings": [
    {{ "match": {{ "model": "{COUNTED}" }}, "profile": "guest" }},
    {{ "match": {{ "model": "{SERVED}" }}, "profile": "served" }},
    {{ "profile": "namespaced" }}
  ] }}"#
        )
    }

    /// The rows of `model`'s pool among `shapes`: every rank of its engines.
    fn ranks(shapes: &[Shape], model: &str) -> usize {
        shapes
            .iter()
            .filter(|shape| shape.model == model)
            .map(|shape| usize::from(shape.ranks))
            .sum()
    }

    /// One row of `model`'s pool keeps a lease, and [`ASKS`] asks follow:
    /// under a stage that scores by the requests in flight, none of them
    /// names that row. `answered` counts the calls the plugin itself
    /// answered, as many a decision as its kind takes: the round's first
    /// ask says how many. A round whose asks did not each take that many
    /// was reached by a hold of the runner, or met a plugin whose thread
    /// was cold: its stage stepped aside and scored every row alike. Such a
    /// round claims nothing and is made again, [`TRIES`] rounds at most;
    /// the last one is read whatever it says.
    async fn passes_over_a_held_row(
        harness: &mut Harness,
        model: &str,
        answered: impl Fn() -> u64,
    ) -> Result<(), String> {
        let (held, lease) = kept(harness, for_model(nth(model, 0), model)).await?;
        let mut round = 0;
        let (each, calls, named) = loop {
            round += 1;
            let before = answered();
            let mut each = 0;
            let mut named = false;
            for n in 0..ASKS {
                let ask = for_model(nth(model, round * ASKS + n), model);
                named |= picked(harness, ask).await? == held;
                if n == 0 {
                    each = answered() - before;
                }
            }
            let calls = answered() - before;
            if (each != 0 && calls == each * ASKS) || round == TRIES {
                break (each, calls, named);
            }
        };
        given_up(harness, &[lease], 0).await?;
        if each == 0 || calls != each * ASKS {
            return Err(format!(
                "the plugin answered {calls} calls over the {ASKS} asks of its last round of {TRIES}, {each} at the first"
            ));
        }
        if named {
            return Err("an ask named the row that holds a lease".to_owned());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_hosts_own_stages_decide_over_the_fleet() {
        let case = async {
            // The host loads its two plugins and hands the controller its
            // catalog whole, ahead of the bind. One count of open decisions
            // stands behind both runtimes, as in a host's own build.
            let open = OpenDecisions::new();
            let runtime = Arc::new(wasm::Runtime::new(&open).expect("a runtime"));
            let mut spec = PluginSpec::new(GUEST, wasm_guest::record_guest());
            spec.config = vec![wasm_guest::mode::LEAST_IN_FLIGHT];
            // A budget of its own as wide as an ask's ([`ask`]): a plugin
            // that states none is cut at the default frame, and a hold of
            // the runner past that frame would be the plugin's own failure.
            spec.budget_micros = 5_000_000;
            // The guest adds to its counters by id: each id has a name.
            spec.metrics = (0..=wasm_guest::counter::PICK_RANDOM)
                .map(|id| format!("counter-{id}"))
                .collect();
            let mut load = wasm::Plugin::load(&runtime, spec.clone());
            for _ in 1..TRIES {
                if load.is_ok() {
                    break;
                }
                load = wasm::Plugin::load(&runtime, spec.clone());
            }
            let guest = load.expect("the guest loads");

            let plugin = TestPlugin::serve();
            let runtime = grpc::Runtime::new(&open).expect("a runtime");
            let socket = Transport::Unix {
                path: plugin.socket.clone(),
                peer_uid: None,
            };
            let mut spec = GrpcPluginSpec::new(REMOTE, socket);
            // The load's connect and hello, far past a loaded host's round
            // trip, and a budget of its own as wide as an ask's, as the
            // guest states one.
            spec.connect_micros = 5_000_000;
            spec.budget_micros = 5_000_000;
            let served = grpc::Plugin::load(&runtime, spec).expect("the served plugin loads");

            let mut catalog = StageRegistry::builtin();
            catalog
                .scorer(NAMESPACED, || Box::new(OfNamespace))
                .expect("a free name");
            let bound = Arc::clone(&guest);
            catalog
                .scorer(guest.name(), move || {
                    Box::new(WasmScorer::new(bound.bind(), None))
                })
                .expect("a free name");
            let bound = Arc::clone(&served);
            catalog
                .scorer(served.name(), move || {
                    Box::new(GrpcScorer::new(bound.bind(), None))
                })
                .expect("a free name");
            let dir = tempfile::tempdir().expect("a temp dir");
            let file = dir.path().join("policy.json");
            std::fs::write(&file, document()).expect("the document's file");
            let mut harness = Harness::start_hosting(catalog, &file)
                .await
                .expect("a controller of a host's own build starts on its document");

            // The fleet registers with a third pool, the served plugin's, and
            // turns ready as under a stock build.
            let mut shapes = FLEET.to_vec();
            shapes.extend(
                [Shape {
                    model: SERVED,
                    ..FLEET[0]
                }; 2],
            );
            harness
                .register(&shapes)
                .await
                .expect("the fleet registers");
            harness.feed();
            within(
                "every rank of the three pools is healthy",
                PROBE_INTERVAL + POLL_INTERVAL + ROOM,
                || {
                    [MODEL, COUNTED, SERVED].into_iter().all(|model| {
                        harness.pool(model, Role::Single).is_ok_and(|pool| {
                            pool.len() == ranks(&shapes, model)
                                && pool.iter().all(|row| row.healthy)
                        })
                    })
                },
            )
            .await
            .expect("the fleet turns ready");

            // A request's cache namespace reaches the stage that decides it:
            // requests of one namespace meet on one row whatever their
            // draws, and as many namespaces as the pool has rows name each
            // row.
            let rows = ranks(&shapes, MODEL);
            let mut homes = HashSet::new();
            for namespace in (0u64..).take(rows) {
                let mut met = HashSet::new();
                for draw in 0..4 {
                    let mut ask = nth("namespace", 4 * namespace + draw);
                    if let Some(facts) = &mut ask.facts {
                        facts.namespace_seed = namespace;
                    }
                    let row = picked(&mut harness, ask)
                        .await
                        .expect("a pick answers an ask over a ready fleet");
                    met.insert(row);
                }
                assert_eq!(
                    met.len(),
                    1,
                    "requests of one cache namespace meet on one row: {met:?}"
                );
                homes.extend(met);
            }
            assert_eq!(
                homes.len(),
                rows,
                "each cache namespace has a row of its own under the host's scorer: {homes:?}"
            );

            // A plugin's stage decides its model, in both forms: the guest
            // in the controller's process and the plugin served outside it
            // score a row by the requests in flight on it, which are the
            // leases the decisions before left there.
            passes_over_a_held_row(&mut harness, COUNTED, || guest.stats().answered)
                .await
                .expect("the guest passes over the row that holds a lease");
            passes_over_a_held_row(&mut harness, SERVED, || served.stats().answered)
                .await
                .expect("the served plugin passes over the row that holds a lease");

            harness.stop().await.expect("a clean stop");
        };
        tokio::time::timeout(CASE, Box::pin(case))
            .await
            .expect("the case ends inside its outer bound");
    }
}

/// A pair's case in this harness: a standby takes the fleet when its leader
/// stops. Two controllers stand over one lease. The fleet registers at the
/// leader, and the standby holds every rank ready and no load report, under
/// the id the leader gave it, by the leader's log alone. The leader stops
/// and the standby leads under a higher term. A gateway that turns to it
/// is answered over the same rows: each routing key is back on the worker
/// it had, a request in flight at the stop is given up at no cost and
/// booked nowhere twice, and the promoted controller's own load monitor
/// reads the engines.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the standby binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_takes_the_fleet_when_its_leader_stops() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let (mut harness, standby) = Harness::start_pair(&dir.path().join("lease"))
            .await
            .expect("a leader and its standby start");
        let led = harness.term_floor();

        // The fleet registers at the leader and turns ready there.
        let shapes = &FLEET[..3];
        let engines = harness.register(shapes).await.expect("the fleet registers");
        let busy = engines[1];
        let ranks = shapes.len();
        harness.feed();
        within(
            "every rank is healthy and holds a load report",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.len() == ranks && pool.iter().all(|row| row.healthy && row.load.is_some())
                })
            },
        )
        .await
        .expect("the fleet turns ready");

        // The standby follows: the leader's log alone brings it every rank,
        // ready, under the id the leader gave it, and no load report.
        within("the standby holds every rank of the fleet", ROOM, || {
            harness.followed(&standby) == Ok(ranks)
        })
        .await
        .expect("the standby follows its leader");

        // A routing key has its worker under the ring preset, and one
        // request is in flight: its lease is kept and dispatched.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let mut homes = Vec::new();
        for key in 1..=ASKS {
            let home = picked(&mut harness, keyed(nth("home", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            homes.push(home);
        }
        let (_, in_flight) = kept(&mut harness, nth("flight", 0))
            .await
            .expect("a pick answers an ask over a ready fleet");

        // The leader stops, the planned stop. The standby takes the lease
        // and leads under a term of its own, and the gateway that turns to
        // it is welcomed there.
        harness
            .hand_over(standby)
            .await
            .expect("the standby takes the lead");
        assert_eq!(harness.welcome().controller_id, "fleet-standby");
        assert!(
            harness.term_floor() > led,
            "a promoted standby answers under a term above its leader's"
        );
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        assert!(
            pool.len() == ranks && pool.iter().all(|row| row.healthy),
            "a promoted standby holds its leader's rows, each ready: {pool:?}"
        );

        // No document crossed with the log: the case applies the one its
        // leader held, as it did there. Each routing key is back on the
        // worker it had, named by the id the leader gave the row.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        for (key, home) in (1..=ASKS).zip(&homes) {
            let again = picked(&mut harness, keyed(nth("again", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, *home,
                "a routing key keeps its worker when a standby takes the lead"
            );
        }

        // The request in flight at the stop is booked nowhere twice. Its
        // lease was the stopped leader's: the promoted controller holds no
        // lease of the gateway, takes the release with no error, and by
        // the fewest requests a round of asks, each kept, names every row
        // once.
        assert_eq!(
            harness.leases(GATEWAY),
            0,
            "a promoted standby holds no lease its leader issued"
        );
        given_up(&mut harness, &[in_flight], 0)
            .await
            .expect("a lease of the leader that stopped is given up at no cost");
        let fewest = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        harness
            .apply_document(fewest)
            .expect("a form of this build's preset applies");
        let mut round = HashSet::new();
        let mut leases = Vec::new();
        for n in (0u64..).take(ranks) {
            let (row, lease) = kept(&mut harness, nth("round", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            round.insert(row);
            leases.push(lease);
        }
        assert_eq!(
            round.len(),
            ranks,
            "a lease of the leader that stopped books no row of the promoted one: {round:?}"
        );
        given_up(&mut harness, &leases, 0)
            .await
            .expect("the leases given up leave the file");

        // The promoted controller polls the engines itself from here on:
        // requests held on one engine reach its row, and nobody else writes
        // a load now.
        harness
            .hold(busy, HELD as usize)
            .await
            .expect("the held requests");
        within(
            "the busy engine's row reads its held requests",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter().any(|row| {
                        row.engine == busy && row.load.is_some_and(|load| load.running == HELD)
                    })
                })
            },
        )
        .await
        .expect("a promoted standby's own load monitor reads the engines");

        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

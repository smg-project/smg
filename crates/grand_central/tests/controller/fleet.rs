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
//! A request that names its worker is answered that worker over the link,
//! by the number its host states at the door's ask.
//!
//! No stub server: the engines are the mock worker's, the controller is the
//! one the binary runs and the client is the generated one. Every wait is
//! bounded and named, and the case stands under one outer bound.
//!
//! One module a subject holds the cases, each a file under `fleet_cases/`:
//! `routing.rs`, `pair_plan.rs`, `clients.rs`, `watch.rs` and `takeover.rs`.
//! This file holds what they share: the fleet's shapes, the asks, what reads
//! a pick, and the case over a host's own stages (`hosted`).

#[expect(
    dead_code,
    reason = "this binary's engines run on runtimes of their own; the engine served on the test's runtime is the load path's"
)]
#[path = "../common/engines.rs"]
mod engines;
#[path = "../common/fleet.rs"]
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
#[path = "../support/grpc_plugin.rs"]
mod grpc_plugin;
#[expect(dead_code, reason = "this binary reads no admin page")]
#[path = "../common/net.rs"]
mod net;
#[path = "../common/pair.rs"]
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
#[path = "../support/wasm_guest.rs"]
mod wasm_guest;

use std::time::{Duration, Instant};

#[cfg(feature = "wasm")]
use fleet::PROBE_INTERVAL;
use fleet::{within, Harness, Shape, GATEWAY, POLL_INTERVAL, ROOM};
use openai_protocol::worker::RuntimeType;
use smg_grand_central::{
    engine::Role,
    service::{
        client::{self, Answer, Link},
        proto::{
            busy, decision::Outcome, identity::Form, released, DecideRequest, Decision, Identity,
            Lease, Released, RequestFacts,
        },
    },
};
use tonic::{transport::Endpoint, Code};

#[path = "fleet_cases/clients.rs"]
mod clients;
#[path = "fleet_cases/pair_plan.rs"]
mod pair_plan;
#[path = "fleet_cases/routing.rs"]
mod routing;
#[path = "fleet_cases/takeover.rs"]
mod takeover;
#[path = "fleet_cases/watch.rs"]
mod watch;

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

/// The asks an assertion makes where one draw could hide what it reads.
const ASKS: u64 = 64;

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

/// `ask` for the model named `model`.
fn for_model(mut ask: DecideRequest, model: &str) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        model.clone_into(&mut facts.model);
    }
    ask
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
    let Ok(Answer::Decision(
        Decision {
            leases,
            outcome: Some(Outcome::Pick(pick)),
            ..
        },
        _,
    )) = &answer
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

/// The whole case's bound: its waits sum to seconds on an idle host.
const CASE: Duration = Duration::from_secs(180);

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
    let Ok((
        Decision {
            leases,
            outcome: Some(Outcome::Pick(_)),
            ..
        },
        _,
    )) = answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let [lease] = leases.as_slice() else {
        return Err(format!("a pick with the wrong leases: {leases:?}"));
    };
    Ok(*lease)
}

/// A controller of a host's own build over the fleet: its catalog holds a
/// scorer in code, a WebAssembly guest's and a served plugin's beside the
/// stock stages, and the document of its spec file names the three.
#[cfg(feature = "wasm")]
mod hosted {
    use std::{collections::HashSet, sync::Arc};

    use smg_grand_central::{
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
            facts: &smg_grand_central::engine::RequestFacts<'_>,
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

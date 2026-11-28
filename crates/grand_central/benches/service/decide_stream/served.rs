//! A controller served over a row's fleet: its configuration, the fleet's
//! fill with its bands, and the serving thread. A child of the bench's
//! root: it reads the root's private `Shape`, `Fleet`, `model` and the
//! common `filled`.

use std::{
    env,
    net::{Ipv4Addr, SocketAddr},
    sync::{mpsc as std_mpsc, Arc},
    thread,
    time::Duration,
};

use smg_grand_central::{
    document::{AppliedSpec, PolicySpec, ShadowSpec},
    engine::Role,
    service::controller::{Config, Controller, Listener, Scope, State},
    store::{MemoryStore, Registrant, WorkerFacts, WorkerRow},
    worker_host::{Members, Objects},
};
use tokio::{runtime, sync::oneshot};

use super::{filled, model, threads::RUNTIME, Fleet, Shape};

const PRESET: &str = "least_load";

/// The pair plan of the pair count cell: the plan of the engine's own
/// pair cells (`pair_classes`), least load on each side.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": {
    "side": { "filter": ["healthy", "not-overloaded", "not-tried"], "score": ["least-load"], "pick": "best" }
  },
  "bindings": [
    { "match": { "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "side", "decode": "side" } }
  ]
}"#;

/// The requests every member reports beneath its band: more than a wire row
/// has decisions ([`wire`] checks each row's count), so no count of leases
/// on file reaches it and a lease on file moves no row's cost.
pub(super) const HELD: u32 = 1 << 21;
/// The requests between two bands of a row that tells its dispatches,
/// where a row that tells none has one: twice the floor beneath the bands,
/// so more than twice a wire row's decisions. A dispatch told is credited
/// to its member at one request's work (the load stage's mean prefill, a
/// request naming no size) until that member's next report, and every
/// dispatch of a row falls on its least member: the credits stay beneath
/// the next band, and a dispatch told moves no row's place.
const APART: u32 = 2 * HELD;

/// The seed of the order the members are dealt to their bands in.
const DEAL: u64 = 0x0B_A4D5;

/// A controller served on its own thread and runtime.
pub(super) struct Served {
    pub(super) addr: SocketAddr,
    pub(super) store: Arc<MemoryStore>,
    stop: oneshot::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl Served {
    pub(super) fn stop(self) {
        // The serve hears the stop unless it already ended, which its
        // thread's end tells.
        drop(self.stop);
        self.thread
            .join()
            .expect("the controller's thread ends cleanly");
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener::of(name.to_owned(), loopback(), scopes.to_vec(), None)
}

/// The binary's default shape on ephemeral loopback ports.
fn config(deciders: usize) -> Config {
    let mut config = Config::of(
        "bench-controller".to_owned(),
        vec![
            listener("gateway", &[Scope::Gateway]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        loopback(),
    );
    config.shutdown_settle = Duration::from_millis(10);
    config.shutdown_grace = Duration::from_secs(5);
    config.decide_shards = deciders;
    config
}

/// The band of each of `members` rows, a band a request above the one
/// before it. The rows are dealt in an order shuffled by a fixed seed to
/// bands of one row and of two by turns; over 512 members to bands of
/// eight and of sixteen, a sixty-fourth of the rows a unit.
fn bands(members: u32) -> Vec<u32> {
    let rows = usize::try_from(members).expect("a count that fits");
    let mut order: Vec<usize> = (0..rows).collect();
    let mut state = DEAL;
    for last in (1..rows).rev() {
        // One value of SplitMix64 a swap: the same order on every build.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut drawn = state;
        drawn = (drawn ^ (drawn >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        drawn = (drawn ^ (drawn >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        drawn ^= drawn >> 31;
        let below = u64::try_from(last).expect("an index that fits") + 1;
        order.swap(last, usize::try_from(drawn % below).expect("an index"));
    }
    let unit = (rows / 64).max(1);
    let mut band_of = vec![0; rows];
    let (mut band, mut left) = (0, unit);
    for row in order {
        if left == 0 {
            band += 1;
            left = if band % 2 == 0 { unit } else { 2 * unit };
        }
        if let Some(its) = band_of.get_mut(row) {
            *its = band;
        }
        left -= 1;
    }
    band_of
}

/// The shape's fleet: ready members under the row's preset as generation
/// 1, written through `objects`, the controller's holder of its worker
/// objects. No report is written for them ([`READS_LOADS`]):
/// the bands ([`bands`]) set each member's ceiling alone, at twice the
/// highest band's report, where a row that tells its dispatches stands its
/// bands apart ([`APART`]).
/// A shared fleet is one set of members that serve every model; fleets of
/// their own are the same count a model, over members that serve it
/// alone. A pair plan's row holds that many members a side, under the pair
/// cell's plan.
fn fill(store: &MemoryStore, objects: &Objects<Members>, shape: Shape) {
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let fleets: Vec<(String, Vec<String>)> = match shape.fleet {
        Fleet::Shared => vec![(String::new(), models)],
        Fleet::Own => models
            .into_iter()
            .enumerate()
            .map(|(fleet, model)| (format!("{fleet}-"), vec![model]))
            .collect(),
    };
    let sides: &[(Role, &str)] = if shape.pair {
        &[(Role::Prefill, "prefill"), (Role::Decode, "decode")]
    } else {
        &[(Role::Single, "worker")]
    };
    let ceiling = if shape.frames {
        let highest = bands(shape.members).into_iter().max().unwrap_or(0);
        2 * (HELD + APART * highest)
    } else {
        2 * HELD
    };
    for (fleet, models) in &fleets {
        let rows = sides
            .iter()
            .flat_map(|&(role, name)| (0..shape.members).map(move |index| (role, name, index)));
        for (role, name, index) in rows {
            let key = format!("http://{name}-{fleet}{index}.fleet:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role,
                models: models.clone(),
                dp_size: 1,
                max_running: Some(ceiling),
                ..WorkerFacts::default()
            };
            let written = objects.upsert(store, facts, Registrant::Control);
            let id = written.expect("the store takes the write").id();
            let revision = objects.read_row(id).expect("the row").revision;
            objects.connect_row(store, id, revision);
        }
    }
    let mut document = if shape.pair {
        serde_json::from_str::<PolicySpec>(PAIR_PLAN).expect("a well-formed document")
    } else {
        PolicySpec::preset(PRESET).expect("a preset of this build")
    };
    document.generation = 1;
    if let Ok(sample) = env::var("DECIDE_STREAM_EXPLAIN_SAMPLE") {
        document.explain.sample = sample.parse().expect("an unsigned sampling rate");
    }
    if let (false, Ok(shadow)) = (shape.pair, env::var("DECIDE_STREAM_SHADOW")) {
        let twin = document.profiles[PRESET].clone();
        document.profiles.insert("twin".to_owned(), twin);
        let named = Some(ShadowSpec::Profile("twin".to_owned()));
        if shadow == "other" {
            let mut other = document.bindings[0].clone();
            "another-model".clone_into(&mut other.matcher.model);
            other.shadow = named;
            document.bindings.insert(0, other);
        } else {
            document.bindings[0].shadow = named;
        }
    }
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("above the unbound start");
    store.publish_now();
    // The first model's members of each side, read off the holder's cut.
    let (first, on_file) = (model(0), objects.checkpoint(store).rows);
    for &(role, _) in sides {
        let serves = |row: &&WorkerRow| row.facts.role == role && row.facts.models.contains(&first);
        let ids: Vec<_> = on_file.iter().filter(serves).map(|row| row.id).collect();
        assert_eq!(ids.len(), shape.members as usize, "the {role:?} members");
        filled::filled(store, &ids, &first);
    }
}

/// Serves a controller over the shape's fleet; returns once it leads.
pub(super) fn serve(shape: Shape) -> Served {
    let (ready, bound) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name("controller".to_owned())
        .spawn(move || {
            let runtime = runtime::Builder::new_multi_thread()
                .worker_threads(shape.threads)
                .thread_name(RUNTIME)
                .enable_all()
                .build()
                .expect("the controller's runtime");
            runtime.block_on(async move {
                let mut controller = Controller::bind(config(shape.deciders))
                    .await
                    .expect("the listeners bind");
                controller
                    .identity_as_told("gateway")
                    .expect("a listener of the configuration");
                let store = controller.store();
                fill(&store, &controller.workers(), shape);
                let addr = controller
                    .addresses()
                    .of(Scope::Gateway)
                    .expect("the gateway scope is mounted");
                let mut readiness = controller.readiness();
                let (stop, stopped) = oneshot::channel::<()>();
                let serving = controller.serve(async {
                    // The bench's drop of its end is the stop.
                    let _ = stopped.await;
                });
                let told = async {
                    assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
                    ready
                        .send((addr, store, stop))
                        .expect("the bench waits for the address");
                };
                let (served, ()) = tokio::join!(serving, told);
                served.expect("a clean stop");
            });
        })
        .expect("the controller's thread");
    let (addr, store, stop) = bound.recv().expect("the controller serves");
    Served {
        addr,
        store,
        stop,
        thread,
    }
}

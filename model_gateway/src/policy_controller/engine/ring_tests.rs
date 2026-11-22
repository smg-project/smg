//! The engine of this process beside the classic hash ring policy: one
//! ring, and one worker for a key by either.
//!
//! The engine builds no ring. A request's maker hands it the ring the
//! gateway keeps for the request's membership, the one the classic
//! `consistent_hashing` policy walks, and says where the request's key
//! stands on it. Each case here places the same requests twice over one
//! registry of workers: by the classic placement, the setting off, and by
//! the gateway's engine object, the setting on. Both read one set of kept
//! rings, as each reads its own in a gateway, and every request must go to
//! the same worker.

use std::{collections::HashSet, sync::Arc, time::Instant};

use axum::http::{HeaderMap, HeaderValue};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::engine::WorkerId;
use smg_worker::{identity::basic::Worker, test_support::no_health_check};

use super::{mark, Ask, PolicyEngine, Prompt};
use crate::{
    config::{PolicyConfig, RouterConfig},
    policies::{KeptRings, PolicyRegistry},
    policy_controller::start_in_process,
    routers::{
        common::{
            placement::{select_single, PlacementInputs},
            snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt},
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

const MODEL: &str = "model-a";
const POOL: RoutingPool = RoutingPool::HttpRegular;

/// One registry of workers under both deciders.
struct Bed {
    registry: Arc<WorkerRegistry>,
    /// The classic policies of the configuration, the setting off: their
    /// placement is the reference.
    classic: PolicyRegistry,
    /// The gateway's object over the engine a start builds for the same
    /// configuration with the setting on. It reads the hash rings `classic`
    /// keeps.
    engine: PolicyEngine,
    workers: Vec<Arc<dyn Worker>>,
}

/// A ready HTTP worker of [`MODEL`], the `at`-th, registered in `registry`.
fn joined(registry: &WorkerRegistry, at: usize) -> Arc<dyn Worker> {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("http://worker-{at}.pool:8000"))
            .model(ModelCard::new(MODEL))
            .health_config(no_health_check())
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    assert!(registry.register(Arc::clone(&worker)).is_some());
    worker
}

/// The routing settings of a gateway under `policy`.
fn under(policy: PolicyConfig) -> RouterConfig {
    RouterConfig {
        policy,
        ..RouterConfig::default()
    }
}

impl Bed {
    /// `workers` ready workers under `config`, placed by its classic
    /// policies and by the engine its start builds with the setting on.
    fn of(config: &RouterConfig, workers: usize) -> Self {
        let registry = Arc::new(WorkerRegistry::new());
        registry.set_overload_shed(config.worker_overload_shed);
        let workers = (0..workers).map(|at| joined(&registry, at)).collect();
        let classic = PolicyRegistry::with_override(
            config.policy.clone(),
            config.routing_key_override.clone(),
        );
        let picked = RouterConfig {
            policy_controller: true,
            ..config.clone()
        };
        let started = start_in_process(&picked, None)
            .expect("a document the engine runs")
            .expect("a gateway that places by a routing policy");
        assert!(started.decides_whole(), "the routers ask the engine");
        let engine = PolicyEngine::own(started.engine, Arc::clone(&registry));
        classic.share_rings_with(&engine);
        Self {
            registry,
            classic,
            engine,
            workers,
        }
    }

    /// The hash rings both deciders read.
    fn rings(&self) -> &KeptRings {
        self.classic.kept_rings()
    }

    /// The address of the worker the classic placement gives `sent`.
    fn by_classic(&self, sent: &Sent<'_>) -> Option<String> {
        let tried: Vec<String> = sent.tried.iter().map(|url| (*url).to_owned()).collect();
        let inputs = PlacementInputs {
            headers: Some(&sent.headers),
            rid_key: self.classic.derive_rid_key(sent.rid),
            tokens: sent.tokens,
            tried: &tried,
            ..PlacementInputs::default()
        };
        let worker = select_single(&self.registry, &self.classic, MODEL, POOL, None, inputs);
        worker.map(|worker| worker.url().to_owned())
    }

    /// The address of the worker the engine object gives `sent`.
    async fn by_engine(&self, sent: &Sent<'_>) -> Option<String> {
        let tried: Vec<WorkerId> = sent
            .tried
            .iter()
            .map(|url| WorkerId::from_address(url))
            .collect();
        let hashes = sent
            .tokens
            .map(|tokens| self.engine.hashes(MODEL, &Prompt::of(Some(tokens), None)));
        let id = mark();
        let ask = Ask {
            headers: Some(&sent.headers),
            rid: sent.rid,
            hashes: hashes.as_ref(),
            pool: Some(POOL),
            tried: &tried,
            ..Ask::of(&id, MODEL)
        };
        let now = Instant::now();
        let mut placed = self.engine.place(&ask, now).await.ok()?;
        let on = placed.worker().url().to_owned();
        placed.dispatched(now);
        Some(on)
    }

    /// The one worker both deciders give `sent`, or `None` where neither
    /// gives it any. They agree, or the case ends here.
    async fn agreed(&self, sent: &Sent<'_>, what: &str) -> Option<String> {
        let classic = self.by_classic(sent);
        let engine = self.by_engine(sent).await;
        assert_eq!(engine, classic, "{what}: the engine, then classic");
        classic
    }

    /// [`Self::agreed`] for each of `keys` requests of `form`, numbered
    /// from zero: the worker of each, every one placed.
    async fn agreed_on_each(&self, form: Form, keys: usize, what: &str) -> Vec<String> {
        let mut owners = Vec::with_capacity(keys);
        for key in 0..keys {
            let sent = form.of(key);
            let what = format!("{what}, {form:?} {key}");
            let owner = self.agreed(&sent, &what).await;
            owners.push(owner.unwrap_or_else(|| panic!("{what}: no worker")));
        }
        owners
    }

    /// Whether the ring of the membership in force is filled.
    fn ring_is_filled(&self) -> bool {
        self.registry
            .get_routing_snapshot(MODEL)
            .hash_ring_is_filled()
    }
}

/// One request, as each decider reads it.
#[derive(Default)]
struct Sent<'a> {
    headers: HeaderMap,
    /// The request's own id.
    rid: Option<&'a str>,
    /// The prompt's tokens, where a policy places by them.
    tokens: Option<&'a [u32]>,
    /// The addresses of the workers an earlier attempt was sent to.
    tried: &'a [&'a str],
}

/// A request that carries `value` under the header `name`.
fn carrying(name: &'static str, value: &str) -> Sent<'static> {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(value).expect("a header's value");
    headers.insert(name, value);
    Sent {
        headers,
        ..Sent::default()
    }
}

/// The form of a request's key.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// A session's key, under the routing key header.
    RoutingKey,
    /// A client's key: the address it was forwarded for.
    ForwardedFor,
    /// A client's key: its cookie.
    Cookie,
}

const FORMS: [Form; 3] = [Form::RoutingKey, Form::ForwardedFor, Form::Cookie];

impl Form {
    /// The `key`-th request of this form.
    fn of(self, key: usize) -> Sent<'static> {
        match self {
            Self::RoutingKey => carrying("x-smg-routing-key", &format!("conversation-{key}")),
            Self::ForwardedFor => {
                let address = format!("10.{}.{}.7", key / 250, key % 250);
                carrying("x-forwarded-for", &address)
            }
            Self::Cookie => carrying("cookie", &format!("session={key}; theme=dark")),
        }
    }
}

/// How many workers took a key.
fn distinct(owners: &[String]) -> usize {
    owners.iter().collect::<HashSet<_>>().len()
}

/// How many keys have another worker in `now` than in `before`.
fn moved(before: &[String], now: &[String]) -> usize {
    before.iter().zip(now).filter(|(was, is)| was != is).count()
}

/// Every key of a large set has one worker by the classic policy and by
/// the engine, over fleets of every size a gateway holds: a session's key
/// under the routing key header, and a client's key from the address it was
/// forwarded for and from its cookie. Three thousand keys a fleet, each
/// compared.
#[tokio::test]
async fn every_key_has_the_worker_the_classic_ring_gives_it() {
    for workers in [1, 2, 3, 8, 64, 512] {
        let bed = Bed::of(&under(PolicyConfig::ConsistentHashing), workers);
        for form in FORMS {
            let what = format!("{workers} workers");
            let owners = bed.agreed_on_each(form, 1_000, &what).await;
            // The keys spread: the comparison is of a ring's placement.
            let spread = distinct(&owners);
            assert!(spread >= workers.min(6), "{what}, {form:?}: {spread}");
        }
        assert!(bed.ring_is_filled(), "the ring both walked");
        assert_eq!(bed.engine.faults(), 0);
    }
}

/// A worker that is not ready is passed over by both, to the same next
/// worker on the ring, and a key whose worker is ready keeps it. A worker
/// over its overload thresholds keeps its sessions' keys by both, and is
/// passed over for a client's key by both.
#[tokio::test]
async fn a_worker_that_is_down_or_over_its_thresholds_is_passed_as_the_classic_policy_passes_it() {
    let bed = Bed::of(&under(PolicyConfig::ConsistentHashing), 64);
    let mut ready = Vec::new();
    for form in FORMS {
        ready.push(bed.agreed_on_each(form, 1_000, "all ready").await);
    }

    // One worker in five is not ready.
    for worker in bed.workers.iter().step_by(5) {
        worker.set_status(WorkerStatus::NotReady);
    }
    for (form, ready) in FORMS.into_iter().zip(&ready) {
        let owners = bed.agreed_on_each(form, 1_000, "one in five down").await;
        let moved = moved(ready, &owners);
        assert!((120..=300).contains(&moved), "{form:?}: {moved} keys moved");
    }
    for worker in &bed.workers {
        worker.set_status(WorkerStatus::Ready);
    }

    // One worker in four is over its thresholds: a session's key stays, a
    // client's key moves on.
    for worker in bed.workers.iter().step_by(4) {
        worker.set_overloaded(true);
    }
    for (form, ready) in FORMS.into_iter().zip(&ready) {
        let owners = bed.agreed_on_each(form, 1_000, "one in four over").await;
        let moved = moved(ready, &owners);
        match form {
            Form::RoutingKey => assert_eq!(moved, 0, "a session left its worker"),
            Form::ForwardedFor | Form::Cookie => {
                assert!((150..=350).contains(&moved), "{form:?}: {moved} keys moved");
            }
        }
    }

    // Every worker is not ready: neither has a worker for any key.
    for worker in &bed.workers {
        worker.set_status(WorkerStatus::NotReady);
    }
    for form in FORMS {
        let none = bed.agreed(&form.of(7), "nobody ready").await;
        assert_eq!(none, None, "{form:?}");
    }
    assert_eq!(bed.engine.faults(), 0);
}

/// Where the gateway sheds over the thresholds, a worker over them keeps no
/// key: a session's key is passed on by both, to the same worker.
#[tokio::test]
async fn under_the_shed_setting_a_worker_over_its_thresholds_keeps_no_key_by_either() {
    let config = RouterConfig {
        worker_overload_shed: true,
        ..under(PolicyConfig::ConsistentHashing)
    };
    let bed = Bed::of(&config, 64);
    let ready = bed
        .agreed_on_each(Form::RoutingKey, 1_000, "all ready")
        .await;
    for worker in bed.workers.iter().step_by(4) {
        worker.set_overloaded(true);
    }
    let owners = bed.agreed_on_each(Form::RoutingKey, 1_000, "shed").await;
    let moved = moved(&ready, &owners);
    assert!((150..=350).contains(&moved), "{moved} keys moved");
    assert_eq!(bed.engine.faults(), 0);
}

/// A retry passes the workers its request was sent to, by both: the key
/// goes to the next worker on the ring that was not tried, and back to its
/// owner where every worker was.
#[tokio::test]
async fn a_retry_goes_to_the_next_worker_on_the_ring_as_the_classic_policy_sends_it() {
    let bed = Bed::of(&under(PolicyConfig::ConsistentHashing), 8);
    let every: Vec<&str> = bed.workers.iter().map(|worker| worker.url()).collect();
    for key in 0..500 {
        let first = Form::RoutingKey.of(key);
        let owner = bed.agreed(&first, "the first attempt").await;
        let owner = owner.expect("a worker");
        let second = Sent {
            tried: &[owner.as_str()],
            ..Form::RoutingKey.of(key)
        };
        let next = bed.agreed(&second, "the second attempt").await;
        let next = next.expect("a worker");
        assert_ne!(next, owner, "key {key}: the retry went back");
        let third = Sent {
            tried: &[owner.as_str(), next.as_str()],
            ..Form::RoutingKey.of(key)
        };
        let last = bed.agreed(&third, "the third attempt").await;
        let last = last.expect("a worker");
        assert!(last != owner && last != next, "key {key}");
        let spent = Sent {
            tried: &every,
            ..Form::RoutingKey.of(key)
        };
        let back = bed.agreed(&spent, "every worker tried").await;
        assert_eq!(back, Some(owner), "key {key}");
    }
    assert_eq!(bed.engine.faults(), 0);
}

/// Where a task fills the rings, a request ahead of its model's first ring
/// is placed by its key's position among the workers it may take, and by
/// both on the same worker: a key has one worker before the ring too. No
/// request of either builds the ring.
#[tokio::test]
async fn ahead_of_the_first_ring_every_key_has_the_worker_the_classic_policy_gives_it() {
    let bed = Bed::of(&under(PolicyConfig::ConsistentHashing), 64);
    let _task = bed.rings().pretend_warm_task();
    for worker in bed.workers.iter().step_by(5) {
        worker.set_status(WorkerStatus::NotReady);
    }
    for form in FORMS {
        let owners = bed.agreed_on_each(form, 1_000, "no ring yet").await;
        assert!(distinct(&owners) > 32, "{form:?}");
    }
    assert!(!bed.ring_is_filled(), "a request built the ring");
    assert_eq!(bed.engine.faults(), 0);
}

/// A worker joins, another leaves and a third is replaced at its address:
/// each change publishes another list, whose ring the gateway fills off the
/// path of a request. Until it does, here for as long as the case holds
/// the fill back, both walk the ring kept from the membership before, and
/// every key has one worker by both. A worker that joined takes no key
/// until the fill, a key whose worker left goes on, and a worker replaced
/// at its address keeps its keys. Where no worker of the kept ring is left
/// to take a key, both place it by its position over the workers that
/// joined. Behind the fill every key has one worker by both again.
#[tokio::test]
async fn across_a_join_a_leave_and_a_replacement_every_key_keeps_the_classic_worker() {
    let mut bed = Bed::of(&under(PolicyConfig::ConsistentHashing), 16);
    let _task = bed.rings().pretend_warm_task();
    bed.rings().fill_every_model(&bed.registry);
    let keys = 1_000;
    let mut before = Vec::new();
    for form in FORMS {
        before.push(bed.agreed_on_each(form, keys, "the first ring").await);
    }

    // Two workers join.
    for at in [16, 17] {
        let worker = joined(&bed.registry, at);
        bed.workers.push(worker);
    }
    assert!(!bed.ring_is_filled());
    for (form, before) in FORMS.into_iter().zip(&before) {
        let owners = bed.agreed_on_each(form, keys, "behind a join").await;
        assert_eq!(moved(before, &owners), 0, "{form:?}: a key moved");
    }

    // A worker leaves: its keys go on, and no other moves.
    let left = bed.workers.remove(3);
    assert!(bed.registry.remove_by_url(left.url()).is_some());
    for (form, before) in FORMS.into_iter().zip(&before) {
        let owners = bed.agreed_on_each(form, keys, "behind a leave").await;
        let held = before.iter().filter(|owner| *owner == left.url()).count();
        assert!(held > 0, "{form:?}: the worker that left held no key");
        assert_eq!(moved(before, &owners), held, "{form:?}");
        assert!(owners.iter().all(|owner| owner != left.url()));
    }

    // A worker is replaced at its address: its keys stay there.
    let replaced = bed.workers[5].url().to_owned();
    let id = bed.registry.get_id_by_url(&replaced);
    let replacing: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(replaced.clone())
            .model(ModelCard::new(MODEL))
            .health_config(no_health_check())
            .build(),
    );
    replacing.set_status(WorkerStatus::Ready);
    let id = id.expect("the worker to replace");
    assert!(bed.registry.replace(&id, Arc::clone(&replacing)));
    bed.workers[5] = replacing;
    for (form, before) in FORMS.into_iter().zip(&before) {
        let owners = bed.agreed_on_each(form, keys, "behind a replacement").await;
        let kept = |(was, is): (&String, &String)| *was != replaced || is == was;
        assert!(before.iter().zip(&owners).all(kept), "{form:?}");
    }

    // No worker of the kept ring is ready: the two that joined take the
    // keys, each key on one of them by both.
    let (earlier, newcomers) = bed.workers.split_at(bed.workers.len() - 2);
    let came = |owner: &String| newcomers.iter().any(|worker| worker.url() == owner);
    for worker in earlier {
        worker.set_status(WorkerStatus::NotReady);
    }
    for form in FORMS {
        let owners = bed
            .agreed_on_each(form, keys, "the kept ring's all down")
            .await;
        assert_eq!(distinct(&owners), 2, "{form:?}");
        assert!(owners.iter().all(came), "{form:?}");
    }
    for worker in earlier {
        worker.set_status(WorkerStatus::Ready);
    }
    assert!(!bed.ring_is_filled(), "a request built the ring");

    // The fill: the ring of the list in force, and the workers that joined
    // take their keys by both.
    bed.rings().fill_every_model(&bed.registry);
    assert!(bed.ring_is_filled());
    for form in FORMS {
        let owners = bed.agreed_on_each(form, keys, "behind the fill").await;
        let taken = owners.iter().filter(|owner| came(owner)).count();
        assert!(taken > 0, "{form:?}: the workers that joined took no key");
    }
    assert_eq!(bed.engine.faults(), 0);
}

/// Under the routing key override the key of a request is its routing key
/// header's or its own id's, without the marks of a turn and of a retry:
/// every turn of a conversation has one worker, the one the classic policy
/// gives it.
#[tokio::test]
async fn under_the_override_a_conversations_turns_have_the_worker_the_classic_policy_gives_them() {
    let mut config = under(PolicyConfig::ConsistentHashing);
    config.routing_key_override.enabled = true;
    let bed = Bed::of(&config, 64);
    let mut owners = Vec::new();
    for conversation in 0..500 {
        let mut turns = Vec::new();
        for (turn, retry) in [(1, 0), (2, 0), (2, 1), (7, 3)] {
            let rid = format!("chat-{conversation}_t{turn}_r{retry}");
            let by_id = Sent {
                rid: Some(&rid),
                ..Sent::default()
            };
            turns.push(bed.agreed(&by_id, &rid).await.expect("a worker"));
            let key = format!("chat-{conversation}_t{turn}");
            let by_header = carrying("x-smg-routing-key", &key);
            turns.push(bed.agreed(&by_header, &key).await.expect("a worker"));
        }
        assert_eq!(distinct(&turns), 1, "conversation {conversation}");
        owners.push(turns.swap_remove(0));
    }
    assert!(distinct(&owners) > 32);
    assert_eq!(bed.engine.faults(), 0);
}

/// What the engine keeps of a ring is one table a list: where each address
/// of the ring stands in the list. It is made by the list's first decision
/// behind the fill of its membership's ring, and by no decision after: a
/// request in steady state finds it. A request that walks a ring kept from
/// another membership makes none. With the setting off nothing of it is
/// made: the classic placement leaves the list's cell empty.
#[tokio::test]
async fn the_lists_table_is_made_once_behind_the_fill_and_never_with_the_setting_off() {
    let bed = Bed::of(&under(PolicyConfig::ConsistentHashing), 16);
    let _task = bed.rings().pretend_warm_task();
    bed.rings().fill_every_model(&bed.registry);
    // Where the table the engine keeps of the list in force stands.
    let table = || {
        let generation = bed.registry.get_routing_snapshot(MODEL);
        let members = generation.pool_members_cell(POOL).get()?;
        let places = members.ring_places().get()?;
        Some(std::ptr::from_ref(places).addr())
    };

    // The setting off: a thousand classic placements make nothing.
    for key in 0..1_000 {
        assert!(bed.by_classic(&Form::RoutingKey.of(key)).is_some());
    }
    let generation = bed.registry.get_routing_snapshot(MODEL);
    assert!(generation.pool_members_cell(POOL).get().is_none());
    drop(generation);

    // The setting on: the first decision makes the table, and the next
    // thousand find it.
    assert!(bed.by_engine(&Form::RoutingKey.of(0)).await.is_some());
    let made = table().expect("the first decision's table");
    for key in 0..1_000 {
        assert!(bed.by_engine(&Form::RoutingKey.of(key)).await.is_some());
        assert_eq!(table(), Some(made), "a decision made the table again");
    }

    // A worker joins: until the fill a request walks the kept ring, and no
    // table is made for the list in force.
    let _worker = joined(&bed.registry, 16);
    for key in 0..1_000 {
        assert!(bed.by_engine(&Form::RoutingKey.of(key)).await.is_some());
    }
    assert_eq!(table(), None, "a request made a table of a kept ring");
    bed.rings().fill_every_model(&bed.registry);
    assert!(bed.by_engine(&Form::RoutingKey.of(0)).await.is_some());
    let made = table().expect("the first decision behind the fill");
    for key in 0..1_000 {
        assert!(bed.by_engine(&Form::RoutingKey.of(key)).await.is_some());
    }
    assert_eq!(table(), Some(made));

    // Under a document that reads no ring the engine keeps no rings, and a
    // list has no table.
    let turns = Bed::of(&under(PolicyConfig::RoundRobin), 4);
    assert!(turns.engine.rings().is_none());
    assert!(turns.by_engine(&Form::RoutingKey.of(0)).await.is_some());
    let generation = turns.registry.get_routing_snapshot(MODEL);
    let members = generation.pool_members_cell(POOL).get();
    let members = members.expect("what the engine keeps of the list");
    assert!(members.ring_places().get().is_none());
    assert!(!generation.hash_ring_is_filled());
}

/// The prefix form walks the same ring: a prompt's prefix has one worker,
/// keeps it across a join until the gateway fills the next ring, and the
/// prefixes spread over the fleet. The engine reads the gateway's ring for
/// it and builds none. The position a prefix stands at is the engine's
/// own, and not yet the classic `prefix_hash` policy's.
#[tokio::test]
async fn a_prefix_keeps_one_worker_on_the_gateways_ring() {
    let prefix_hash = PolicyConfig::PrefixHash {
        prefix_token_count: 256,
        load_factor: 1.25,
        balance_abs_threshold: 10,
        cache_boundaries: Vec::new(),
    };
    let bed = Bed::of(&under(prefix_hash), 16);
    assert!(bed.engine.rings().is_some(), "the engine reads the rings");
    let _task = bed.rings().pretend_warm_task();
    bed.rings().fill_every_model(&bed.registry);
    let prompts: Vec<Vec<u32>> = (0..64_u32)
        .map(|prompt| (0..320).map(|at| prompt * 1_000 + at).collect())
        .collect();
    let mut owners = Vec::new();
    for tokens in &prompts {
        let sent = Sent {
            tokens: Some(tokens),
            ..Sent::default()
        };
        let mut on = Vec::new();
        for _ in 0..4 {
            on.push(bed.by_engine(&sent).await.expect("a worker"));
        }
        assert_eq!(distinct(&on), 1, "one prompt, more workers than one");
        owners.push(on.swap_remove(0));
    }
    assert!(distinct(&owners) > 8, "the prefixes did not spread");

    let generation = bed.registry.get_routing_snapshot(MODEL);
    let members = generation.pool_members_cell(POOL).get();
    let members = members.expect("what the engine keeps of the list");
    assert!(members.ring_places().get().is_some(), "the gateway's ring");
    drop(generation);

    // A worker joins: each prefix keeps its worker until the fill.
    let _worker = joined(&bed.registry, 16);
    for (tokens, owner) in prompts.iter().zip(&owners) {
        let sent = Sent {
            tokens: Some(tokens),
            ..Sent::default()
        };
        let on = bed.by_engine(&sent).await;
        assert_eq!(on.as_ref(), Some(owner), "a prefix moved at a join");
    }
    assert_eq!(bed.engine.faults(), 0);
}

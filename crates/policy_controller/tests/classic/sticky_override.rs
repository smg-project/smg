//! The classic routing-key override rows: a pool of workers, what a router
//! holds on them (the requests, the routing keys, the requests of one key),
//! and a list of requests, keyed or not, under a policy and one of the
//! sticky map's four assignments. The gateway places a keyed request of a
//! policy that reads no key itself by its sticky map, and puts a key with
//! two requests on its worker to the policy or the map's mode again.

use std::collections::BTreeMap;

use serde::Deserialize;
use smg_policy_controller::{
    document::{
        Assignment, IndexKind, LoadMetricSetting, PolicySettings, PolicySpec,
        RoutingKeyOverrideSettings, RoutingSettings,
    },
    engine::{
        Candidate, Constraints, Engine, Health, Identity, PoolFacts, RequestFacts, Scratch,
        WorkerId,
    },
    stages::Lifecycle,
    tracking::{Lease, Side},
};

use crate::{Outcome, Recorded, Replayed};

/// The rule these rows record, as a sheet's head names it.
pub const POLICY: &str = "sticky_override";

/// The header a document names for the key. A row's request states its key
/// as a fact, so the replay reads no header.
const HEADER: &str = "x-smg-routing-key";

/// One worker of a row: the requests the host holds on it and the routing
/// keys that have a request in flight on it.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    keys: u16,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and its model.
    Pick { key: Option<String>, model: String },
    /// A worker stops being routable.
    Down(usize),
    /// The worker is routable again.
    Up(usize),
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// The requests of one routing key in flight on each worker, from here
    /// on: the count a rule's cap reads.
    Held { key: String, on: Vec<u16> },
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    /// The policy under the override: the preset of the same name here.
    policy: String,
    /// The assignment, in the spelling the gateway's configuration and this
    /// crate's document share.
    assignment: Assignment,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

impl Row {
    /// The document a gateway's routing settings give for the row
    /// ([`RoutingSettings::policy_document`]): the row's policy with every
    /// setting at its flag's default, under the sticky override with the
    /// row's assignment. Or the settings' refusal, in its own words.
    fn spec(&self) -> Result<PolicySpec, String> {
        self.spec_at(14_400, 60)
    }

    /// [`Self::spec`] with the override's idle time and the seconds
    /// between two of its sweeps.
    fn spec_at(
        &self,
        max_idle_secs: u64,
        eviction_interval_secs: u64,
    ) -> Result<PolicySpec, String> {
        self.document(true, max_idle_secs, eviction_interval_secs)
    }

    /// The document the same settings give with the override off: the
    /// row's policy alone.
    fn alone(&self) -> Result<PolicySpec, String> {
        self.document(false, 14_400, 60)
    }

    /// The settings' document for the row's policy, with every setting of
    /// the policy at its flag's default.
    fn document(
        &self,
        overridden: bool,
        max_idle_secs: u64,
        eviction_interval_secs: u64,
    ) -> Result<PolicySpec, String> {
        let policy = match self.policy.as_str() {
            "random" => PolicySettings::Random,
            "round_robin" => PolicySettings::RoundRobin,
            "power_of_two" => PolicySettings::PowerOfTwo {
                load_metric: LoadMetricSetting::ExpectedWait,
            },
            "least_load" => PolicySettings::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
            "passthrough" => PolicySettings::Passthrough,
            "bucket" => PolicySettings::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            },
            "cache_aware" => PolicySettings::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                balance_prefix_margin_tokens: 0,
                balance_uncached_credit: false,
                eviction_interval_secs: 120,
                max_tree_size: 67_108_864,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: IndexKind::Tree,
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            },
            other => return Err(format!("no settings are written here for `{other}`")),
        };
        let settings = RoutingSettings {
            policy,
            routing_key_override: RoutingKeyOverrideSettings {
                enabled: overridden,
                headers: vec![HEADER.into()],
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode: self.assignment,
            },
            ..RoutingSettings::default()
        };
        settings.policy_document().map_err(|err| err.to_string())
    }
}

/// The engine of the row over the row's pool in its starting state, with
/// the host's side of a dispatch: the lease carries the session and what
/// the decision did with its key.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    /// The routing keys each worker held at the row's start.
    keys: Vec<u16>,
    /// The requests of each key the row's steps hold on each worker.
    held: BTreeMap<String, Vec<u16>>,
    scratch: Scratch,
    /// The host's clock, microseconds: each request has its own instant.
    clock: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form.
    fn new(row: &Row) -> Result<Self, String> {
        Self::under(row, &row.spec()?)
    }

    /// The bed of `row` under `spec`.
    fn under(row: &Row, spec: &PolicySpec) -> Result<Self, String> {
        let engine = spec.validate().map_err(|err| err.to_string())?;
        let pool = row
            .workers
            .iter()
            .enumerate()
            .map(|(i, worker)| Candidate {
                position: i as u32,
                id: WorkerId(1_000 + i as u64),
                key: 1_000 + i as u32,
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight: worker.in_flight,
                ..Candidate::default()
            })
            .collect();
        Ok(Self {
            engine,
            pool,
            keys: row.workers.iter().map(|worker| worker.keys).collect(),
            held: BTreeMap::new(),
            scratch: Scratch::new(),
            clock: 1_000,
        })
    }

    /// One request decided and dispatched, as a host commits it: the pins
    /// move by the lease's session and verdict. Each row of the pool states
    /// what the host counts for this request: the requests of its own key on
    /// the worker, and the keys the worker holds. The counts move by the
    /// row's own steps, as they did under the recorder.
    fn pick(&mut self, key: Option<&str>, model: &str, seed: u64) -> usize {
        self.clock += 1_000;
        let own = key.and_then(|key| self.held.get(key));
        for (i, candidate) in self.pool.iter_mut().enumerate() {
            candidate.session_in_flight = own.map_or(0, |on| on[i]);
            let held = self.held.values().filter(|on| on[i] > 0).count();
            candidate.active_sessions = self.keys[i] + held as u16;
        }
        let facts = RequestFacts {
            model,
            session_key: key.map_or(Identity::NONE, Identity::plain),
            session_counted: true,
            seed,
            now_micros: self.clock,
            ..RequestFacts::default()
        };
        let pool_facts = PoolFacts {
            pool_size: self.pool.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool_facts,
            &self.pool,
            &Constraints::default(),
            &mut self.scratch,
        );
        let pick = decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: self.clock,
            session: facts.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        target.position as usize
    }

    /// One step of the row; a request answers the worker it was placed on.
    fn step(&mut self, step: &Step, seed: u64) -> Option<usize> {
        match step {
            Step::Pick { key, model } => return Some(self.pick(key.as_deref(), model, seed)),
            Step::Down(i) => self.pool[*i].health.healthy = false,
            Step::Up(i) => self.pool[*i].health.healthy = true,
            Step::Loads(loads) => {
                assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
                for (candidate, &load) in self.pool.iter_mut().zip(loads) {
                    candidate.in_flight = load;
                }
            }
            Step::Held { key, on } => {
                assert_eq!(on.len(), self.pool.len(), "a count of every worker");
                self.held.insert(key.clone(), on.clone());
            }
        }
        None
    }
}

/// The row run once from a fresh engine: the worker each request went to,
/// or the validator's words. `draw` separates the seeds of a drawn row's
/// runs; a sequence row is run `0`.
fn run(row: &Row, draw: u64) -> Result<Vec<usize>, String> {
    let mut bed = Bed::new(row)?;
    Ok(row
        .steps
        .iter()
        .enumerate()
        .filter_map(|(n, step)| bed.step(step, draw * 1_000 + n as u64 + 1))
        .collect())
}

/// This crate's outcome for the row, in the shape of the recorded one, or
/// its refusal of the row's form.
fn replay(row: &Row) -> Replayed {
    match &row.outcome {
        Outcome::Sequence(_) => match run(row, 0) {
            Ok(picked) => Replayed::Ran(Outcome::Sequence(picked)),
            Err(why) => Replayed::Refused(why),
        },
        Outcome::Counts { draws, .. } => {
            // Each draw runs the row from a fresh engine, as the recorder
            // ran it from a fresh registry: a key pinned in one draw would
            // decide the next. The row's last request is the one counted.
            let mut counts = vec![0u32; row.workers.len()];
            for draw in 1..=u64::from(*draws) {
                match run(row, draw) {
                    Ok(picked) => {
                        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
                    }
                    Err(why) => return Replayed::Refused(why),
                }
            }
            Replayed::Ran(Outcome::Counts {
                draws: *draws,
                counts,
            })
        }
    }
}

impl Recorded for Row {
    const POLICY: &'static str = POLICY;

    fn name(&self) -> &str {
        &self.name
    }

    fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    fn replay(&self) -> Replayed {
        replay(self)
    }
}

/// A key used again after its idle time and before the next sweep keeps
/// its worker, and past that sweep it is placed anew. The gateway's sticky
/// map forgets an idle key at its periodic sweep alone, and a use of the
/// key ahead of that sweep renews it
/// (`model_gateway/src/policies/manual.rs:395` and `:234`).
/// No sheet records a time, so this case states the rule.
#[test]
fn a_key_used_after_its_idle_time_and_before_the_sweep_keeps_its_worker() {
    let row = Row {
        name: "an idle key ahead of its sweep".into(),
        policy: "random".into(),
        assignment: Assignment::MinLoad,
        workers: vec![
            WorkerState {
                in_flight: 1,
                keys: 0,
            },
            WorkerState {
                in_flight: 5,
                keys: 0,
            },
        ],
        steps: Vec::new(),
        outcome: Outcome::Sequence(Vec::new()),
    };
    // A key is idle after a minute, and the sweep runs every hour.
    let spec = row.spec_at(60, 3_600).expect("the settings' document");
    let mut bed = Bed::under(&row, &spec).expect("an engine");
    let first = bed.pick(Some("k1"), "m", 1);
    // The loads turn: a key placed anew would take the other worker.
    bed.pool[0].in_flight = 9;
    bed.pool[1].in_flight = 0;
    bed.clock += 61_000_000;
    let again = bed.pick(Some("k1"), "m", 2);
    assert_eq!((first, again), (0, 0), "the key kept its worker");
    // Past the next sweep the key is forgotten and placed anew.
    bed.clock += 3_600_000_000;
    assert_eq!(bed.pick(Some("k1"), "m", 3), 1, "the sweep forgot the key");
}

/// `passthrough` under the override, which no sheet of the gateway records.
/// It is held to the gateway's rule
/// (`model_gateway/src/policies/registry.rs`, `select_sticky`) by a twin bed
/// that runs the policy alone, asked a request with no key wherever the
/// gateway asks its policy:
///
/// - A key's first request goes where the policy sends it, and is pinned
///   there (`:769-780`).
/// - The key's next request follows the pin, and the policy is not asked
///   (`:733-739`, reached from `:661-662` ahead of the policy's call at
///   `:664`). So a request with no key takes the policy's next answer, the
///   same on both beds, and that answer is another worker by then.
/// - With two requests of the key on its worker the policy is asked again
///   (`:740-748`), and the key's next request goes to its pick, a worker
///   under that count (`:758-760`).
///
/// The case runs over a list of three workers and over one of sixteen. From
/// sixteen workers on the engine draws first for a request with no key and
/// runs the whole pass for a key with no pin, and the gateway asks one
/// policy for both: the two must send a request to one worker.
#[test]
fn a_key_goes_where_its_policy_sends_it_and_then_follows_its_pin() {
    for workers in [3, 16] {
        let row = Row {
            name: "passthrough".into(),
            policy: "passthrough".into(),
            assignment: Assignment::Delegate,
            workers: vec![
                WorkerState {
                    in_flight: 0,
                    keys: 0,
                };
                workers
            ],
            steps: Vec::new(),
            outcome: Outcome::Sequence(Vec::new()),
        };
        let mut keyed = Bed::new(&row).expect("the settings' document under the override");
        let alone = row.alone().expect("the policy's own document");
        let mut alone = Bed::under(&row, &alone).expect("an engine");
        // `passthrough` sends every request to the first worker that takes
        // one: with that worker down, the key lands on the next.
        for bed in [&mut keyed, &mut alone] {
            bed.pool[0].health.healthy = false;
        }
        let pin = keyed.pick(Some("k1"), "m", 1);
        assert_eq!(pin, alone.pick(None, "m", 1), "{workers}: a new key");
        // Worker 0 is back: the policy alone goes there, and the pin stays.
        for bed in [&mut keyed, &mut alone] {
            bed.pool[0].health.healthy = true;
        }
        assert_eq!(keyed.pick(Some("k1"), "m", 2), pin, "{workers}: a pin");
        let next = keyed.pick(None, "m", 3);
        assert_eq!(next, alone.pick(None, "m", 3), "{workers}: no key");
        assert_ne!(next, pin, "{workers}: the policy alone goes elsewhere");
        // Two requests of the key are on its worker.
        let mut on = vec![0; workers];
        on[pin] = 2;
        keyed.held.insert("k1".into(), on);
        let moved = keyed.pick(Some("k1"), "m", 4);
        assert_eq!(moved, alone.pick(None, "m", 4), "{workers}: at the cap");
        assert_ne!(moved, pin, "{workers}: the policy picks another worker");
        keyed.held.clear();
        assert_eq!(
            keyed.pick(Some("k1"), "m", 5),
            moved,
            "{workers}: the key's next request"
        );
    }
}

/// `bucket` under the override, where the rule plans single requests alone.
/// The gateway's policy holds a bucket of sizes for the prefill workers of
/// a pair alone (`model_gateway/src/policies/registry.rs:1564-1575`), and
/// draws a healthy worker among single ones
/// (`model_gateway/src/policies/bucket.rs:186-191`). This crate's stage
/// places a prefill decision alone and draws for every other. So a single
/// request's worker is a draw on both sides, and a twin bed that runs the
/// policy alone draws the same worker for the same seed:
///
/// - A key's first request goes where the draw sends it.
/// - The key's next requests follow that pin, under seeds whose draw is
///   another worker.
/// - With two requests of the key on its worker the draw is asked again,
///   and the pin moves where it lands on another worker.
///
/// No sheet records `bucket` under the override, so this case states the
/// rule.
#[test]
fn bucket_draws_for_a_key_and_then_follows_its_pin() {
    let row = Row {
        name: "bucket".into(),
        policy: "bucket".into(),
        assignment: Assignment::Delegate,
        workers: vec![
            WorkerState {
                in_flight: 0,
                keys: 0,
            };
            3
        ],
        steps: Vec::new(),
        outcome: Outcome::Sequence(Vec::new()),
    };
    let alone = row.alone().expect("the policy's own document");
    let mut pins = Vec::new();
    for seed in 1..=8_u64 {
        let mut keyed = Bed::new(&row).expect("the settings' document under the override");
        let mut alone = Bed::under(&row, &alone).expect("an engine");
        let pin = keyed.pick(Some("k1"), "m", seed);
        assert_eq!(pin, alone.pick(None, "m", seed), "seed {seed}: a new key");
        pins.push(pin);
        // The pin is followed whatever the seed draws.
        let mut elsewhere = false;
        for next in 100..116_u64 {
            assert_eq!(keyed.pick(Some("k1"), "m", next), pin, "seed {seed}: a pin");
            elsewhere |= alone.pick(None, "m", next) != pin;
        }
        assert!(elsewhere, "seed {seed}: the draw alone goes elsewhere");
        // Two requests of the key are on its worker: the draw is asked.
        let mut on = vec![0; 3];
        on[pin] = 2;
        keyed.held.insert("k1".into(), on);
        let moved = (200..216_u64)
            .map(|next| {
                let drawn = keyed.pick(Some("k1"), "m", next);
                assert_eq!(
                    drawn,
                    alone.pick(None, "m", next),
                    "seed {seed}: at the cap"
                );
                drawn
            })
            .find(|&drawn| drawn != pin)
            .expect("a draw on another worker");
        keyed.held.clear();
        assert_eq!(
            keyed.pick(Some("k1"), "m", 300),
            moved,
            "seed {seed}: the pin moved"
        );
    }
    pins.sort_unstable();
    pins.dedup();
    assert!(pins.len() > 1, "the draw spreads new keys: {pins:?}");
}

/// Rows copied from the recorded sheet: a pin under the cap and under two
/// models, a key seen first under three assignments, each pin order after a
/// failover, and at the cap a pin that follows, a pick on the pinned worker
/// again, a pick on a worker at the cap, every worker at the cap and the
/// policy's draw.
pub const INLINE: &str = r#"{
  "recorded_from": "26f90083091e4dd791420febd88beda3bef3faac",
  "policy": "sticky_override",
  "rows": [
    {
      "name": "sticky-min-load-a-key-keeps-its-worker-under-the-cap-past-the-load",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 1, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 0, 1]}
    },
    {
      "name": "sticky-one-key-pins-apart-under-two-models",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m1"}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m2"}},
        {"loads": [9, 9, 0]},
        {"pick": {"key": "k1", "model": "m1"}},
        {"pick": {"key": "k1", "model": "m2"}}
      ],
      "outcome": {"sequence": [1, 0, 1, 0]}
    },
    {
      "name": "sticky-min-group-a-new-key-takes-the-fewest-keys-whatever-the-load",
      "policy": "random",
      "assignment": "min_group",
      "workers": [{"in_flight": 0, "keys": 2}, {"in_flight": 5, "keys": 0}, {"in_flight": 0, "keys": 1}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 1, 0]}},
        {"held": {"key": "x1", "on": [0, 1, 0]}},
        {"held": {"key": "x2", "on": [0, 1, 0]}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 2, 1]}
    },
    {
      "name": "sticky-random-a-new-key-is-drawn-over-every-worker",
      "policy": "random",
      "assignment": "random",
      "workers": [{"in_flight": 9, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [2020, 1989, 1991]}}
    },
    {
      "name": "sticky-min-load-a-key-moves-off-a-worker-that-is-down-and-the-first-owner-leads-again",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 0},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0]}
    },
    {
      "name": "sticky-delegate-a-new-key-is-the-policy-s-pick-and-stays-past-the-load",
      "policy": "power_of_two",
      "assignment": "delegate",
      "workers": [{"in_flight": 5, "keys": 0}, {"in_flight": 0, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"loads": [0, 5]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": null, "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 0, 0, 1]}
    },
    {
      "name": "sticky-delegate-the-latest-worker-leads-after-a-failover",
      "policy": "power_of_two",
      "assignment": "delegate",
      "workers": [{"in_flight": 0, "keys": 0}, {"in_flight": 5, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 1},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 1},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0, 1]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-request-goes-to-the-least-loaded-and-the-pin-follows",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 2, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [1, 2, 0]}},
        {"loads": [5, 0, 2]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 0]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-least-loaded-is-the-pinned-worker-again",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 2, 0]}},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 3, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 1]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-pin-does-not-move-to-a-worker-at-the-cap",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 2, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 1, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 1]}
    },
    {
      "name": "sticky-min-load-every-worker-at-the-cap",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 2, 2]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 1, 2]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 1]}
    },
    {
      "name": "sticky-delegate-at-the-cap-the-policy-draws-over-every-worker",
      "policy": "random",
      "assignment": "delegate",
      "workers": [{"in_flight": 0, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 0, "keys": 0}],
      "steps": [
        {"down": 1},
        {"down": 2},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 1},
        {"up": 2},
        {"held": {"key": "k1", "on": [2, 0, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [1959, 2013, 2028]}}
    }
  ]
}"#;

/// The policies that count what they sent a worker, under the override,
/// through the engine over a gateway's worker objects. There a request's
/// guard says its dispatch, and no stage hears the dispatch of a request
/// the override's rule placed: the rule keeps it from the profile. The
/// beds above run the engine's rows, where a host says every dispatch
/// whole.
mod sends {
    use std::{
        collections::{HashMap, VecDeque},
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_policy_controller::{
        document::{AppliedSpec, Assignment},
        engine::{
            Constraints, DecisionEngine, HashSeeds, Identity, LentList, ListCell, PlacedBy,
            PolicyEngine, PromptHashes, Request, RequestFacts, RequestMaker, SessionVerdict,
            WorkerLists,
        },
        store::GatewayId,
        tracking::{self, DecisionGuard, TransportClass},
    };
    use smg_worker::{
        identity::basic::Worker,
        load::state::{one_rank_load, LoadSnapshot},
        test_support::ready_worker,
        worker::WorkerStatus,
    };

    use super::{Outcome, Row};

    const MODEL: &str = "m";

    /// The engine of the settings' document for `policy` at its flags'
    /// defaults, with the sticky override on or off.
    fn engine(policy: &str, overridden: bool) -> PolicyEngine {
        let row = Row {
            name: policy.into(),
            policy: policy.into(),
            assignment: Assignment::Delegate,
            workers: Vec::new(),
            steps: Vec::new(),
            outcome: Outcome::Sequence(Vec::new()),
        };
        let document = if overridden { row.spec() } else { row.alone() };
        let document = document.expect("the settings' document");
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the document validates"))
    }

    /// A request stated ahead, with the list its host lends and the reports
    /// it hands.
    struct Asks<'a> {
        request: Request<'a>,
        members: LentList<'a>,
        loads: &'a Arc<LoadSnapshot>,
    }

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            MODEL
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.request
        }

        fn workers(&self) -> WorkerLists<'_> {
            WorkerLists::Single(self.members)
        }

        fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
            Some(self.loads)
        }
    }

    /// A decision its host holds: its guard, the place of its first target
    /// and who placed it.
    type Decided<'e> = (DecisionGuard<&'e PolicyEngine>, usize, PlacedBy);

    /// A request in flight: the guards of the engines that decided it, the
    /// place it went to and its key.
    type Sent<'e> = (Vec<DecisionGuard<&'e PolicyEngine>>, usize, &'static str);

    /// `list` lent with `cell`, the cell its host keeps beside it: the host
    /// states nothing of its workers.
    fn lent<'a>(list: &'a Arc<[Arc<dyn Worker>]>, cell: &'a ListCell) -> LentList<'a> {
        LentList {
            workers: list,
            cell,
            facts: None,
        }
    }

    /// A gateway's side of its engines: one list of worker objects, the
    /// reports a decision reads of them, and the requests in flight.
    struct Host<'e> {
        list: Arc<[Arc<dyn Worker>]>,
        loads: Arc<LoadSnapshot>,
        /// The instant every report was sampled at. The request `nth` is
        /// decided and sent `nth + 1` milliseconds behind it, so no report
        /// reflects a send.
        start: Instant,
        held: VecDeque<Sent<'e>>,
    }

    impl<'e> Host<'e> {
        /// A list whose worker `at` reports `queued[at]` tokens waiting and
        /// one request running, which the gateway holds on it.
        fn new(queued: &[i32]) -> Self {
            let list: Arc<[Arc<dyn Worker>]> = (0..queued.len())
                .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
                .collect();
            for worker in list.iter() {
                worker.increment_load();
            }
            let start = Instant::now();
            Self {
                list,
                loads: Self::reports(start, queued),
                start,
                held: VecDeque::new(),
            }
        }

        /// The reports of [`Self::new`], sampled at `start`.
        fn reports(start: Instant, queued: &[i32]) -> Arc<LoadSnapshot> {
            let reports = queued.iter().enumerate().map(|(at, &queued)| {
                let mut report = one_rank_load(queued, 0.1, 1_000.0);
                report.sampled_at = Some(start);
                (format!("http://worker-{at}:8000"), report)
            });
            LoadSnapshot::from_loads_for_test(reports.collect())
        }

        fn at(&self, nth: u32) -> Instant {
            self.start + Duration::from_millis(1) * (nth + 1)
        }

        /// The request `nth` of `tokens` prompt tokens under `key`, with no
        /// key where it is empty, decided by `engine` and not sent yet. It
        /// states no prompt.
        fn decide(
            &self,
            engine: &'e PolicyEngine,
            members: LentList<'_>,
            asked: (&'static str, u32, u32),
        ) -> Decided<'e> {
            self.decide_prompt(engine, members, asked, None)
        }

        /// [`Self::decide`] for a request that states its prompt: `prompt`
        /// is the hashes of the prompt's blocks, as `engine`'s document
        /// reads a prompt.
        fn decide_prompt(
            &self,
            engine: &'e PolicyEngine,
            members: LentList<'_>,
            (key, tokens, nth): (&'static str, u32, u32),
            prompt: Option<&PromptHashes>,
        ) -> Decided<'e> {
            let session_key = match key {
                "" => Identity::NONE,
                key => Identity::plain(key),
            };
            let mut facts = RequestFacts {
                model: MODEL,
                seed: u64::from(nth),
                prompt_tokens: tokens,
                session_key,
                session_counted: true,
                ..RequestFacts::default()
            };
            if let Some(prompt) = prompt {
                prompt.write_to(&mut facts);
            }
            let asks = Asks {
                request: Request {
                    id: b"request",
                    facts,
                    constraints: Constraints::default(),
                },
                members,
                loads: &self.loads,
            };
            let decided =
                DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, self.at(nth));
            let guard = decided.expect("a decision");
            // The host takes no other target: its dispatch names the first.
            // A decision no stage and no rule hears of comes with no lease.
            let pick = guard.decision().pick().expect("a pick");
            let first = pick.targets.first().expect("the first target");
            let (place, by) = (first.position as usize, pick.placed_by());
            (guard, place, by)
        }

        /// The request `nth` goes to `place` under `key`: its dispatch is
        /// said to each engine that decided it, by `guards`, and the
        /// gateway counts it once: the worker object's requests, and its
        /// requests of the key. The request stays in flight.
        fn sent(
            &mut self,
            mut guards: Vec<DecisionGuard<&'e PolicyEngine>>,
            (key, nth): (&'static str, u32),
            place: usize,
        ) {
            for guard in &mut guards {
                guard.dispatched(self.at(nth));
            }
            self.list[place].increment_load();
            if !key.is_empty() {
                self.list[place].increment_routing_key_load(key);
            }
            self.held.push_back((guards, place, key));
        }

        /// One request decided by `engine` and sent: the place it went to
        /// and who placed it.
        fn send(
            &mut self,
            engine: &'e PolicyEngine,
            members: LentList<'_>,
            asked: (&'static str, u32, u32),
        ) -> (usize, PlacedBy) {
            let (guard, place, by) = self.decide(engine, members, asked);
            self.sent(vec![guard], (asked.0, asked.2), place);
            (place, by)
        }

        /// A request in flight ends at the instant of the request `nth`:
        /// each engine that decided it hears its completion, and the
        /// gateway takes it off the worker object's counts.
        fn ended(&mut self, (mut guards, place, key): Sent<'e>, nth: u32) {
            let worker = &self.list[place];
            for guard in &mut guards {
                let pick = guard.decision().pick().expect("a pick");
                let served = pick.targets.first().expect("a target").id;
                let outcome = tracking::Outcome::reported(200, TransportClass::Ok)
                    .with_served(tracking::Side::Single, served);
                guard.completed(&outcome, self.at(nth));
            }
            worker.decrement_load();
            if !key.is_empty() {
                worker.decrement_routing_key_load(key);
            }
        }
    }

    /// A pinned request is none of the requests its policy sent. Two
    /// workers each report one request running, and the second reports
    /// 5,000 tokens waiting. A key's first request of 1,000 tokens goes to
    /// the first worker, placed by the policy, and its next request of
    /// 10,000 tokens follows the pin. The policy then prices the first
    /// worker at the 1,000 tokens it placed there and at one request more
    /// that the gateway holds there, at the mean prompt of 1,024 tokens:
    /// 3,048 with the running request, under the second worker's 6,024. So
    /// a request with no key goes to the first worker. Counted as a send
    /// of the policy's, the pinned request would price that worker at
    /// 12,024.
    ///
    /// The pinned request still weighs what a request in flight weighs:
    /// with 1,500 tokens waiting on the second worker, 2,524 in all, the
    /// request with no key goes there. Once the pinned request ends it
    /// weighs nothing: the first worker stands at 2,024, and the request
    /// with no key goes back to it.
    ///
    /// The gateway's policy hears of no request its sticky map placed
    /// (`model_gateway/src/policies/registry.rs:736-739`), and prices a
    /// request the gateway holds that it did not place at the mean prompt
    /// (`model_gateway/src/policies/least_load.rs:802`). Its `power_of_two`
    /// scores with the same credit
    /// (`model_gateway/src/policies/power_of_two.rs:28`). No sheet records
    /// either under the override with a pinned request of a stated size,
    /// so this case states the rule.
    #[test]
    fn a_pinned_request_is_none_of_the_requests_its_policy_sent() {
        for policy in ["least_load", "power_of_two"] {
            let engine = engine(policy, true);
            let mut host = Host::new(&[0, 5_000]);
            let (list, cell) = (Arc::clone(&host.list), ListCell::default());
            let members = lent(&list, &cell);
            let first = host.send(&engine, members, ("k1", 1_000, 0));
            assert_eq!(first, (0, PlacedBy::Stages), "{policy}: a new key");
            let pinned = host.send(&engine, members, ("k1", 10_000, 1));
            assert_eq!(pinned, (0, PlacedBy::Rule), "{policy}: its pin");
            let (_, place, _) = host.decide(&engine, members, ("", 1_000, 2));
            assert_eq!(
                place, 0,
                "{policy}: the pinned request is no send of the policy's"
            );
            host.loads = Host::reports(host.start, &[0, 1_500]);
            let (_, place, _) = host.decide(&engine, members, ("", 1_000, 3));
            assert_eq!(
                place, 1,
                "{policy}: the pinned request weighs as one in flight"
            );
            let pinned = host.held.pop_back().expect("the pinned request");
            host.ended(pinned, 4);
            let (_, place, _) = host.decide(&engine, members, ("", 1_000, 4));
            assert_eq!(place, 0, "{policy}: an ended pinned request weighs nothing");
        }
    }

    /// Each key's leading pin, kept by the host's own book as the gateway's
    /// rule moves it under `delegate`
    /// (`model_gateway/src/policies/registry.rs`, `select_sticky`). The
    /// book reads the gateway's counts and the place a request went to,
    /// and no word of the engine it is held against. No worker a key is
    /// pinned on stops taking requests in these cases, so a key's leading
    /// pin is the one pin its rule reads.
    #[derive(Default)]
    struct Pins(HashMap<&'static str, usize>);

    impl Pins {
        /// Whether the worker at `at` carries two of `key`'s requests, the
        /// cap of the override's rule.
        fn at_cap(host: &Host<'_>, key: &str, at: usize) -> bool {
            host.list[at].routing_key_inflight(key) >= 2
        }

        /// The place of `key`'s leading pin, and whether that worker stands
        /// at the cap.
        fn of(&self, host: &Host<'_>, key: &str) -> (Option<usize>, bool) {
            let pin = self.0.get(key).copied();
            (pin, pin.is_some_and(|at| Self::at_cap(host, key, at)))
        }

        /// The gateway's rule for a request of `key`, held against the
        /// place it went to and who placed it. A request the rule placed
        /// followed the key's leading pin, under its cap
        /// (`model_gateway/src/policies/registry.rs:736-739`). A keyed
        /// request the policy placed had no pin, or its pin at the cap
        /// (`:740-748`, `:769-780`). Whether the pin stood at its cap.
        fn rule(&self, host: &Host<'_>, key: &str, (place, by): (usize, PlacedBy)) -> bool {
            let (pin, at_cap) = self.of(host, key);
            if by == PlacedBy::Rule {
                assert_eq!(
                    Some(place),
                    pin,
                    "{key:?}: the rule follows the leading pin"
                );
                assert!(!at_cap, "{key:?}: a pin at its cap is no pin to follow");
            } else {
                assert!(
                    key.is_empty() || pin.is_none() || at_cap,
                    "{key:?}: the policy is asked of a pin under its cap"
                );
            }
            at_cap
        }

        /// The request of `key` goes to `place`: the gateway's write of the
        /// key's pin, read ahead of the request's own count. A key with no
        /// pin pins the worker its policy picked (`:774-779`). A pin moves
        /// to another worker that stands under the cap, and to none at it
        /// (`:758-760`). The decision's `verdict` is held to that write: it
        /// states a pin where the key's leading pin is `place` behind the
        /// write, and none where the pin stayed on another worker.
        fn sent(
            &mut self,
            host: &Host<'_>,
            key: &'static str,
            place: usize,
            verdict: SessionVerdict,
        ) {
            if key.is_empty() {
                return;
            }
            let pin = self.0.get(key).copied();
            if pin.is_none_or(|pin| pin != place && !Self::at_cap(host, key, place)) {
                self.0.insert(key, place);
            }
            assert_eq!(
                verdict != SessionVerdict::None,
                self.0.get(key) == Some(&place),
                "{key:?}: {verdict:?} at {place} with the pin {pin:?}"
            );
        }
    }

    /// The verdict a held decision states of its key.
    fn verdict(guard: &DecisionGuard<&PolicyEngine>) -> SessionVerdict {
        guard.decision().pick().expect("a pick").session_verdict
    }

    /// A prompt as a request states it: the number of the key's first
    /// prompt where it grows that one, and its own number.
    type Stated = (Option<u32>, u32);

    /// `least_load` under the override, which no sheet of the gateway
    /// records, held to the gateway's rule
    /// (`model_gateway/src/policies/registry.rs`, `select_sticky`) by a twin
    /// engine that runs the policy alone over the same worker objects. The
    /// twin is asked, and told a dispatch, wherever the gateway asks its
    /// policy: at a key's first request, at a key's request with two of its
    /// requests on its pin's worker, and at a request with no key. It is
    /// not asked at a pinned request and hears of none. So the two engines
    /// hold one count of sends, and each request the policy places goes to
    /// one worker under both, behind pinned requests of 500 to 20,000
    /// tokens. From the ninth request on the oldest request in flight ends
    /// ahead of every second decision, a pinned one or one the policy
    /// placed, so the two engines agree behind ends too, and a key still
    /// meets its cap. Each request is held to the gateway's rule by the
    /// key's leading pin ([`Pins::rule`]), and each decision's verdict to
    /// the gateway's write of that pin ([`Pins::sent`]). Over three workers
    /// and over sixteen.
    #[test]
    fn least_load_places_under_the_override_as_the_policy_alone_places() {
        for workers in [3, 16] {
            let (keyed, alone) = (engine("least_load", true), engine("least_load", false));
            let queued: Vec<i32> = (0..workers).map(|at| at * 700).collect();
            let mut host = Host::new(&queued);
            // One list with its cell, lent to both engines: each reads its
            // own value of it.
            let (list, cell) = (Arc::clone(&host.list), ListCell::default());
            let members = lent(&list, &cell);
            let mut pins = Pins::default();
            let (mut pinned, mut behind, mut ended, mut capped) = (0, 0, 0, 0);
            for nth in 0..48_u32 {
                if nth >= 8 && nth % 2 == 0 {
                    let oldest = host.held.pop_front().expect("a request in flight");
                    ended += u32::from(oldest.0.len() == 1);
                    host.ended(oldest, nth);
                }
                let key = ["k0", "k1", "k2", "k3", "k4", ""][nth as usize % 6];
                let tokens = [1_000, 12_000, 3_000, 20_000, 500][nth as usize % 5];
                let (guard, place, by) = host.decide(&keyed, members, (key, tokens, nth));
                let at_cap = pins.rule(&host, key, (place, by));
                pins.sent(&host, key, place, verdict(&guard));
                let mut guards = vec![guard];
                if by == PlacedBy::Rule {
                    pinned += 1;
                } else {
                    let (asked, twin_place, _) = host.decide(&alone, members, ("", tokens, nth));
                    assert_eq!(
                        place, twin_place,
                        "{workers} workers: request {nth}, {key:?}"
                    );
                    guards.push(asked);
                    behind += u32::from(pinned > 0);
                    capped += u32::from(at_cap);
                }
                host.sent(guards, (key, nth), place);
            }
            assert!(
                pinned >= 8 && behind >= 8 && ended >= 8 && capped >= 8,
                "{workers} workers: {pinned} pinned requests, {behind} placed behind one, \
                 {ended} pinned requests ended, {capped} keyed requests at their pin's cap"
            );
        }
    }

    /// The gateway's rotation under `round_robin`, kept by the host's own
    /// book as its policy turns it
    /// (`model_gateway/src/policies/round_robin.rs:142-168`). A request
    /// takes the list's next row. Where that worker takes no request, it
    /// takes the next of the rotation of the workers that do, and the
    /// list's rotation has moved on all the same. A rotation starts where
    /// its policy object first sights it, which no engine repeats, so the
    /// book keeps every pair of starts the placements so far allow: the
    /// list's rotation's, and the other's.
    struct Turns(Vec<(usize, usize)>);

    impl Turns {
        /// Every pair of starts over a list of `workers` rows.
        fn over(workers: usize) -> Self {
            let starts = (0..workers).flat_map(|row| (0..workers).map(move |turn| (row, turn)));
            Self(starts.collect())
        }

        /// A request the policy placed went to `place`, in a list whose
        /// workers `ready` take requests: the starts that put it there
        /// stay, each a turn on. Whether any stays.
        fn placed(&mut self, ready: &[bool], place: usize) -> bool {
            let kept: Vec<usize> = (0..ready.len()).filter(|&at| ready[at]).collect();
            self.0.retain_mut(|(row, turn)| {
                let at = *row % ready.len();
                *row += 1;
                if ready[at] {
                    return at == place;
                }
                *turn += 1;
                kept[(*turn - 1) % kept.len()] == place
            });
            !self.0.is_empty()
        }
    }

    /// `round_robin` under the override, which no sheet of the gateway
    /// records, held to the gateway's rule
    /// (`model_gateway/src/policies/registry.rs`, `select_sticky`) by a twin
    /// engine that runs the policy alone over the same worker objects, as
    /// `least_load` is above. The gateway asks one policy object for a
    /// request with no key, for a key's first request and for a key at its
    /// pin's cap (`:664`, `:774`, `:745`), and not for a request that
    /// follows its pin (`:736-739`). So the three kinds take the turns of
    /// one rotation, and the twin, asked wherever the gateway asks its
    /// policy, turns its own in step: each request the policy places goes
    /// to one worker under both.
    ///
    /// Each of those requests is held to the gateway's rotation too
    /// ([`Turns`]). Behind the first request, the worker two rows on takes
    /// no request for a third of the run: the list's rotation lands on its
    /// row at once, and no key is pinned on it. Over three workers, under
    /// the size a rotation drew its turn from before, over sixteen, at it,
    /// and over sixty-four. Each request is held to the gateway's rule by
    /// the key's leading pin ([`Pins::rule`]), and each decision's verdict
    /// to the gateway's write of that pin ([`Pins::sent`]).
    #[test]
    fn round_robin_places_under_the_override_as_the_policy_alone_places() {
        for workers in [3, 16, 64] {
            let (keyed, alone) = (engine("round_robin", true), engine("round_robin", false));
            let mut host = Host::new(&vec![0; workers]);
            let (list, cell) = (Arc::clone(&host.list), ListCell::default());
            let members = lent(&list, &cell);
            let (mut pins, mut turns) = (Pins::default(), Turns::over(workers));
            // The workers that take requests, and the one that takes none
            // for a while.
            let (mut ready, mut refused) = (vec![true; workers], None);
            let (mut pinned, mut capped, mut ended) = (0, 0, 0);
            // The requests the policy placed while a worker took none.
            let mut passed = 0;
            for nth in 0..96_u32 {
                if nth == 32 {
                    let at: usize = refused.take().expect("a refused worker");
                    host.list[at].set_status(WorkerStatus::Ready);
                    ready[at] = true;
                }
                if nth >= 8 && nth % 2 == 0 {
                    let oldest = host.held.pop_front().expect("a request in flight");
                    ended += u32::from(oldest.0.len() == 1);
                    host.ended(oldest, nth);
                }
                let key = ["k0", "k1", "k2", "k3", "k4", ""][nth as usize % 6];
                let (guard, place, by) = host.decide(&keyed, members, (key, 1_000, nth));
                let at_cap = pins.rule(&host, key, (place, by));
                pins.sent(&host, key, place, verdict(&guard));
                let mut guards = vec![guard];
                if by == PlacedBy::Rule {
                    pinned += 1;
                } else {
                    let (twin, twin_place, _) = host.decide(&alone, members, ("", 1_000, nth));
                    assert_eq!(
                        place, twin_place,
                        "{workers} workers: request {nth}, {key:?}"
                    );
                    guards.push(twin);
                    assert!(
                        turns.placed(&ready, place),
                        "{workers} workers: request {nth}, {key:?}, on {place}: no start of the \
                         gateway's two rotations puts it there"
                    );
                    capped += u32::from(at_cap);
                    passed += u32::from(refused.is_some());
                }
                host.sent(guards, (key, nth), place);
                if nth == 0 {
                    let at = (place + 2) % workers;
                    host.list[at].set_status(WorkerStatus::NotReady);
                    (ready[at], refused) = (false, Some(at));
                }
            }
            assert!(
                pinned >= 8 && capped >= 8 && ended >= 8 && passed >= 8,
                "{workers} workers: {pinned} pinned requests, {capped} keyed requests at their \
                 pin's cap, {ended} pinned requests ended, {passed} requests placed while a \
                 worker took none"
            );
        }
    }

    /// Two requests of a key that meet its pin at the cap, the second
    /// decided ahead of the first's dispatch, go to one worker. The
    /// gateway's sticky map moves a pin where it selects the worker, ahead
    /// of the request's dispatch (`select_sticky` in
    /// `model_gateway/src/policies/registry.rs`: `pin_front` behind the
    /// policy's pick for a pin over its cap). So the key's next request
    /// follows the moved pin under its cap, and the policy is not asked
    /// again. Under `round_robin` a second ask takes the rotation's next
    /// turn: the key's requests would spread over a third worker, and the
    /// key would lead with that one from then on.
    #[test]
    fn a_capped_keys_requests_follow_the_move_decided_ahead_of_its_dispatch() {
        let engine = engine("round_robin", true);
        let mut host = Host::new(&[0; 4]);
        let (list, cell) = (Arc::clone(&host.list), ListCell::default());
        let members = lent(&list, &cell);
        let mut pins = Pins::default();
        // Two requests of the key fill its pin's worker to the cap.
        for nth in 0..2 {
            let (guard, place, by) = host.decide(&engine, members, ("k1", 1_000, nth));
            pins.rule(&host, "k1", (place, by));
            pins.sent(&host, "k1", place, verdict(&guard));
            host.sent(vec![guard], ("k1", nth), place);
        }
        let (first, _) = pins.of(&host, "k1");
        // The third meets the cap: the policy places it on another worker,
        // and the gateway's map moves the pin there as it selects.
        let (third, moved, by) = host.decide(&engine, members, ("k1", 1_000, 2));
        assert!(
            pins.rule(&host, "k1", (moved, by)),
            "the pin stands at its cap"
        );
        assert_ne!(Some(moved), first, "the rotation's next turn");
        pins.sent(&host, "k1", moved, verdict(&third));
        // The fourth is decided ahead of the third's dispatch.
        let (fourth, place, by) = host.decide(&engine, members, ("k1", 1_000, 3));
        assert_eq!(
            (place, by),
            (moved, PlacedBy::Rule),
            "the fourth follows the pin the third's decision moved"
        );
        pins.rule(&host, "k1", (place, by));
        pins.sent(&host, "k1", place, verdict(&fourth));
        host.sent(vec![third], ("k1", 2), moved);
        host.sent(vec![fourth], ("k1", 3), place);
        // Behind both dispatches, and one end on that worker, the key leads
        // with the worker it moved to.
        let last = host.held.pop_back().expect("the fourth request");
        host.ended(last, 4);
        let (fifth, place, by) = host.decide(&engine, members, ("k1", 1_000, 4));
        assert_eq!(
            (place, by),
            (moved, PlacedBy::Rule),
            "the key leads with the worker it moved to"
        );
        pins.rule(&host, "k1", (place, by));
        pins.sent(&host, "k1", place, verdict(&fifth));
    }

    /// `cache_aware` under the override, which no sheet of the gateway
    /// records, held to the gateway's rule
    /// (`model_gateway/src/policies/registry.rs`, `select_sticky`) by a twin
    /// engine that runs the policy alone over the same worker objects, as
    /// `least_load` is above. The gateway's policy writes a prompt for the
    /// worker it placed the request on, and it is not asked at a pinned
    /// request (`:736-739`). So the twin is asked, and writes a prompt,
    /// wherever the engine under the override does: no pinned request's
    /// prompt is in either record, and each request the policy places goes
    /// to one worker under both.
    ///
    /// Under its cap, and at it on even rounds, a key's request states a
    /// prompt of its own. At the cap on odd rounds it grows the key's first
    /// prompt, as a session grows one prefix: that prompt whole, then a
    /// tail of its own. A request with no key repeats an earlier prompt, a
    /// pinned request's on even rounds and one the policy placed on odd
    /// rounds:
    ///
    /// - A pinned request's prompt is in no record, so the policy places
    ///   its repeat as a prompt nobody was sent, and on another worker than
    ///   the pin's more often than not. An engine that recorded the pinned
    ///   request would send the repeat to the pin's worker, and its twin
    ///   would not.
    /// - A prompt the policy placed goes back to its worker.
    /// - A prompt that grows the key's first one goes, at the cap, where
    ///   the policy finds that prefix: on the pinned worker again more
    ///   often than a prompt of its own does. There the request goes over
    ///   the cap, and the pin stays (`:740-761`).
    ///
    /// Each request is held to the gateway's rule by the key's leading pin
    /// ([`Pins::rule`]), and each decision's verdict to the gateway's write
    /// of that pin ([`Pins::sent`]). Over three workers and over sixteen.
    #[test]
    fn cache_aware_places_under_the_override_as_the_policy_alone_places() {
        for workers in [3, 16] {
            let (keyed, alone) = (engine("cache_aware", true), engine("cache_aware", false));
            let queued: Vec<i32> = (0..workers).map(|at| at * 700).collect();
            let mut host = Host::new(&queued);
            let (list, cell) = (Arc::clone(&host.list), ListCell::default());
            let members = lent(&list, &cell);
            // A prompt of its own: 1,024 tokens no other prompt holds. A
            // prompt that grows a key's first one: that prompt whole, then
            // 256 tokens of its own. Hashed as each engine's document reads
            // a prompt.
            let prompt = |engine: &PolicyEngine, (first, of): Stated| {
                let own = if first.is_some() { 256 } else { 1_024 };
                let head = first.map_or(0..0, |first| first * 100_000..first * 100_000 + 1_024);
                let tokens: Vec<u32> = head.chain(of * 100_000..of * 100_000 + own).collect();
                engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN)
            };
            let mut pins = Pins::default();
            // Each key's first request, by its number.
            let mut firsts: HashMap<&str, u32> = HashMap::new();
            // The prompts of the pinned requests and of the requests the
            // policy placed, each with the place it went to.
            let (mut kept, mut told) = (VecDeque::new(), VecDeque::new());
            let (mut pinned, mut capped, mut ended) = (0, 0, 0);
            // Each reading beside the count of requests it is read of: the
            // repeats of a pinned request's prompt placed off the pin's
            // worker, the repeats of a placed prompt sent back to its
            // worker, and the prompts placed at the cap on the pinned
            // worker again, the grown ones and those of their own.
            let (mut elsewhere, mut back) = ((0, 0), (0, 0));
            let (mut grown, mut own) = ((0, 0), (0, 0));
            for nth in 0..96_u32 {
                if nth >= 8 && nth % 2 == 0 {
                    let oldest = host.held.pop_front().expect("a request in flight");
                    ended += u32::from(oldest.0.len() == 1);
                    host.ended(oldest, nth);
                }
                let key = ["k0", "k1", "k2", "k3", "k4", ""][nth as usize % 6];
                let odd = nth / 6 % 2 == 1;
                // A request with no key repeats a prompt: a pinned
                // request's on even rounds, one the policy placed on odd
                // rounds.
                let repeated = match (key, odd) {
                    ("", false) => kept.pop_front(),
                    ("", true) => told.pop_front(),
                    _ => None,
                };
                // A key's request at its pin's cap grows the key's first
                // prompt on odd rounds. Every other states one of its own.
                let (pin, at_cap) = pins.of(&host, key);
                let first = *firsts.entry(key).or_insert(nth);
                let grows = at_cap && odd;
                let stated: Stated = match repeated {
                    Some((stated, _)) => stated,
                    None => (grows.then_some(first), nth),
                };
                let tokens = if stated.0.is_some() { 1_280 } else { 1_024 };
                let hashes = prompt(&keyed, stated);
                let asked = (key, tokens, nth);
                let (guard, place, by) = host.decide_prompt(&keyed, members, asked, Some(&hashes));
                pins.rule(&host, key, (place, by));
                pins.sent(&host, key, place, verdict(&guard));
                let mut guards = vec![guard];
                if by == PlacedBy::Rule {
                    pinned += 1;
                    kept.push_back((stated, place));
                } else {
                    let hashes = prompt(&alone, stated);
                    let asked = ("", tokens, nth);
                    let (twin, twin_place, _) =
                        host.decide_prompt(&alone, members, asked, Some(&hashes));
                    assert_eq!(
                        place, twin_place,
                        "{workers} workers: request {nth}, {key:?}"
                    );
                    guards.push(twin);
                    capped += u32::from(at_cap);
                    let read = |(hits, of): (u32, u32), hit: bool| (hits + u32::from(hit), of + 1);
                    if grows {
                        grown = read(grown, Some(place) == pin);
                    } else if at_cap {
                        own = read(own, Some(place) == pin);
                    }
                    match repeated {
                        Some((_, at)) if odd => back = read(back, place == at),
                        Some((_, at)) => elsewhere = read(elsewhere, place != at),
                        None => told.push_back((stated, place)),
                    }
                }
                host.sent(guards, (key, nth), place);
            }
            let floors = pinned >= 8 && capped >= 8 && ended >= 8;
            let readings = elsewhere.0 >= 4 && back.0 >= 4 && grown.0 >= 4;
            // A grown prompt takes the pinned worker again more often than
            // a prompt of its own: the prefix places it, not the load.
            let by_prefix = grown.0 * own.1 > own.0 * grown.1;
            assert!(
                floors && readings && by_prefix,
                "{workers} workers: {pinned} pinned requests, {capped} keyed requests at their \
                 pin's cap, {ended} pinned requests ended, {elsewhere:?} repeats of a pinned \
                 request's prompt placed off its pin's worker, {back:?} repeats of a placed \
                 prompt sent back to its worker, {grown:?} grown prompts and {own:?} prompts of \
                 their own placed at the cap on the pinned worker again"
            );
        }
    }
}

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
        Assignment, LoadMetricSetting, ModeSettings, Pairing, PolicySettings, PolicySpec,
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
            "power_of_two" => PolicySettings::PowerOfTwo {
                load_metric: LoadMetricSetting::ExpectedWait,
            },
            "passthrough" => PolicySettings::Passthrough,
            "bucket" => PolicySettings::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            },
            other => return Err(format!("no settings are written here for `{other}`")),
        };
        let settings = RoutingSettings {
            mode: ModeSettings::Regular,
            policy,
            enable_igw: false,
            policy_spec: None,
            grouped_dp_sticky: None,
            dp_minimum_tokens_scheduler: false,
            pd_pairing_mode: Pairing::Lenient,
            worker_warmup_secs: 60,
            worker_warmup_share: 0.25,
            worker_warmup_blocks: 1024,
            worker_warmup_thin_ratio: 0.5,
            worker_warmup_divert_every: 8,
            routing_key_override: RoutingKeyOverrideSettings {
                enabled: overridden,
                headers: vec![HEADER.into()],
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode: self.assignment,
            },
            worker_overload_protection: true,
            worker_overload_waiting_requests: Some(8),
            worker_overload_token_usage: Some(0.8),
            worker_overload_shed: false,
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
///   there (`:576-587`).
/// - The key's next request follows the pin, and the policy is not asked
///   (`:540-546`, reached from `:468-469` ahead of the policy's call at
///   `:471`). So a request with no key takes the policy's next answer, the
///   same on both beds, and that answer is another worker by then.
/// - With two requests of the key on its worker the policy is asked again
///   (`:547-555`), and the key's next request goes to its pick, a worker
///   under that count (`:565-567`).
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
        // The pool turns against the pinned worker.
        for bed in [&mut keyed, &mut alone] {
            bed.pool[0].health.healthy = true;
            bed.pool[pin].in_flight = 200;
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
/// a pair alone (`model_gateway/src/policies/registry.rs:1215-1226`), and
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

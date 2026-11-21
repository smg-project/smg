//! Cases of a document's declared signals under the gateway's feed: the
//! values a worker's load reports state by name reach the decisions the
//! policy engine asks. The store starts as a gateway's does, with a document
//! given and one stage of the host's own. The reports go through the feed's
//! load sink, polled and pushed. The policy engine places by them. The server's
//! cases of a host's own connection borrow the stage and the document.
//!
//! A placement alone can pass by a tie's draw: where no row has a value
//! every row scores the same, and the policy engine draws a fresh seed for each
//! placement. So the stage keeps what each decision read of each row, and
//! each step says it beside the placement.

use std::{
    cmp::Ordering,
    collections::BTreeMap,
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde::Deserialize;
use smg_policy_controller::{
    document::{Needs, PolicySpec},
    engine::{Candidate, Phase, RequestFacts, Role, Scorer, TieBand, WireModes, WorkerId},
    load::SignalColumn,
    stages::{Lifecycle, StageRegistry, StageRegistryError},
    store::{HealthPolicy, LoadSink as StoreLoadSink, MemoryStore, Registrant, WorkerFacts},
    worker_host::{MembershipWriteOutcome, MembershipWriter, Precondition},
};
use smg_worker::{
    identity::basic::Worker,
    load::port::{LoadSink, PolledMember},
    worker::{NamedValue, SchedulerLoadSnapshot, WorkerLoadResponse},
};

use super::{feed, Feed};
use crate::{
    config::RouterConfig,
    policy_controller::{
        engine::{Ask, PolicyEngine},
        start_with,
    },
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BackendConnector},
};

type Case = Result<(), Box<dyn Error>>;

const MODEL: &str = "model-a";
const PERIOD: Duration = Duration::from_secs(1);
/// The host's own stage: a scorer that prefers the row with the most of
/// one declared signal.
const MOST: &str = "most";
/// The signal the documents declare, in `slots`, read for five seconds.
pub(crate) const SLOTS: &str = "free_slots";

/// One profile for every model, scored by the declared signal.
pub(crate) const SINGLE: &str = "\
apiVersion: smg.policy/v1
generation: 3
signals:
  free_slots: { unit: slots, max_age_ms: 5000 }
profiles:
  work:
    score: [{ most: { signal: free_slots } }]
    pick: best
";

/// A pair plan whose two sides each score by the declared signal.
const PAIR: &str = "\
apiVersion: smg.policy/v1
generation: 3
signals:
  free_slots: { unit: slots, max_age_ms: 5000 }
profiles:
  prefill:
    score: [{ most: { signal: free_slots } }]
    pick: best
  decode:
    score: [{ most: { signal: free_slots } }]
    pick: best
bindings:
  - plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
";

/// What the stage's decisions read: a row and its value, at each scoring.
pub(crate) type Read = Arc<Mutex<Vec<(WorkerId, Option<f64>)>>>;

/// Rows and the value a decision read of each, in the rows' order.
type Rows = Vec<(WorkerId, Option<f64>)>;

#[derive(Debug)]
struct Most {
    column: SignalColumn,
    read: Read,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Named {
    signal: String,
}

impl Lifecycle for Most {}

impl Scorer for Most {
    fn name(&self) -> &'static str {
        MOST
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..Needs::default()
        }
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let mut read = self.read.lock();
        for (&row, score) in kept.iter().zip(out) {
            let row = &candidates[row as usize];
            let value = self.column.read(facts, row);
            read.push((row.id, value));
            // A row with no value scores under every row that has one.
            *score = value.map_or(0.0, |value| ((value + 1.0) / 1_000.0) as f32);
        }
        TieBand::NORMALIZED
    }
}

/// The stock stages and the host's own one, [`MOST`], with what its
/// decisions read.
pub(crate) fn stages() -> Result<(StageRegistry, Read), StageRegistryError> {
    let read = Read::default();
    let kept = Arc::clone(&read);
    let mut stages = StageRegistry::builtin();
    stages.scorer_with(MOST, move |build| {
        let named: Named = build.params()?;
        Ok(Box::new(Most {
            column: build.signal_column(&named.signal)?,
            read: Arc::clone(&kept),
        }))
    })?;
    Ok((stages, read))
}

/// The value each row was last read at since the last call, in the rows'
/// order: what the decisions between two calls were handed.
pub(crate) fn last_read(read: &Read) -> Rows {
    let rows: BTreeMap<_, _> = read.lock().drain(..).collect();
    rows.into_iter().collect()
}

/// [`last_read`] as its values alone, the least first. A row read with no
/// value stands ahead of every value.
pub(crate) fn values_read(read: &Read) -> Vec<Option<f64>> {
    let mut values: Vec<_> = last_read(read)
        .into_iter()
        .map(|(_, value)| value)
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values
}

/// A gateway's store started with a document given and the host's stage
/// registered, the feed over it with its load sink, and the policy engine.
struct Bed {
    store: Arc<MemoryStore>,
    registry: Arc<WorkerRegistry>,
    feed: Feed,
    policy_engine: PolicyEngine,
    sink: Arc<dyn LoadSink>,
    read: Read,
}

impl Bed {
    fn started(document: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let policy_spec = document
            .map(serde_yaml::from_str::<PolicySpec>)
            .transpose()?;
        let config = RouterConfig {
            policy_controller: true,
            policy_spec,
            ..RouterConfig::default()
        };
        let (stages, read) = stages()?;
        let started = start_with(&config, Some(stages))?
            .ok_or("the controller was picked and no store started")?;
        let store = started.store;
        let registry = Arc::new(WorkerRegistry::new());
        let feed = feed(&store, &registry, BackendConnector::shared());
        let policy_engine = PolicyEngine::new(Arc::clone(&store), Arc::clone(&feed.objects));
        let sink = feed.load_sink(PERIOD);
        Ok(Self {
            store,
            registry,
            feed,
            policy_engine,
            sink,
            read,
        })
    }

    /// A ready row of `role` at `url`, written at the feed's door, and the
    /// worker object the feed built for it.
    fn worker(&self, url: &str, role: Role) -> Result<(WorkerId, Arc<dyn Worker>), Box<dyn Error>> {
        self.rank(url, url, role, 0, 1)
    }

    /// As [`Self::worker`]: the row of rank `rank` of the `size` ranks of
    /// the engine at `url`, under `key`.
    fn rank(
        &self,
        key: &str,
        url: &str,
        role: Role,
        rank: u16,
        size: u16,
    ) -> Result<(WorkerId, Arc<dyn Worker>), Box<dyn Error>> {
        let facts = WorkerFacts {
            key: key.to_owned(),
            engine: url.to_owned(),
            role,
            wire_modes: WireModes::HTTP,
            models: vec![MODEL.to_owned()],
            dp_rank: rank,
            dp_size: size,
            max_running: Some(64),
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        };
        let door: &dyn MembershipWriter = &*self.feed.door;
        let id = match door.found(&Registrant::Control, Precondition::Absent, key, &|_| {
            facts.clone()
        }) {
            Ok(MembershipWriteOutcome::Registered(id)) => id,
            other => return Err(format!("expected a registration, got {other:?}").into()),
        };
        let object = self
            .feed
            .objects
            .object(id)
            .ok_or("the row has no object")?;
        let worker = self
            .registry
            .get(&object)
            .ok_or("the registry holds no worker of the row")?;
        Ok((id, worker))
    }

    /// One engine's poll that began at `at`: an answer for each of its
    /// members, as the collector hands them.
    fn polled(&self, engine: &[(&Arc<dyn Worker>, WorkerLoadResponse)], at: Instant) {
        let members: Vec<_> = engine
            .iter()
            .map(|(worker, answer)| PolledMember {
                worker,
                answer: Some(answer),
                on_file: None,
            })
            .collect();
        self.sink.report(&members, at);
        self.shown();
    }

    /// A record `worker`'s stream pushed, sampled and received at `at`.
    fn pushed(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        at: Instant,
    ) {
        self.sink.record(worker, rank, report, at, at);
        self.shown();
    }

    /// Show what the store holds waiting, as the feed's loop does when the
    /// store's instant comes: the cases run no loop.
    fn shown(&self) {
        if let Some(due) = StoreLoadSink::publish_due(&*self.store) {
            self.feed.settle_at(due);
        }
    }

    /// The store's count of the values it took, and of each way it
    /// dropped one: written, undeclared, wrong unit, out of range, not a
    /// member, older.
    fn taken(&self) -> [u64; 6] {
        self.store.stats().signal_writes
    }

    /// The row a regular request is placed on at `now`, and what its
    /// decision read of each row.
    async fn placed(&self, id: &[u8], now: Instant) -> Result<(WorkerId, Rows), Box<dyn Error>> {
        let placed = self.policy_engine.place(&Ask::of(id, MODEL), now).await;
        let placed = placed.map_err(|unplaced| format!("not placed: {unplaced:?}"))?;
        Ok((placed.id(), last_read(&self.read)))
    }
}

/// `rows` in the rows' order, as a decision's read is listed.
fn read<const N: usize>(mut rows: [(WorkerId, Option<f64>); N]) -> Rows {
    rows.sort_by_key(|&(id, _)| id);
    rows.to_vec()
}

/// One rank's report that states `value` of the declared name under `unit`,
/// `age_ms` old by its own word.
fn stating(rank: i32, unit: &str, value: f64, age_ms: Option<u32>) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: rank,
            values: vec![NamedValue {
                name: SLOTS.into(),
                unit: unit.into(),
                value,
                age_ms,
            }],
            ..SchedulerLoadSnapshot::default()
        }],
        ..WorkerLoadResponse::default()
    }
}

fn slots(value: f64) -> WorkerLoadResponse {
    stating(0, "slots", value, None)
}

const fn ms(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

#[tokio::test]
async fn a_single_placement_follows_each_workers_latest_declared_value_polled_or_pushed() -> Case {
    let bed = Bed::started(Some(SINGLE))?;
    let (a, first) = bed.worker("http://a:8000", Role::Single)?;
    let (b, second) = bed.worker("http://b:8000", Role::Single)?;
    let (c, third) = bed.worker("http://c:8000", Role::Single)?;
    let start = Instant::now();
    // One round of polls, an engine each: the second worker states the
    // most.
    bed.polled(&[(&first, slots(10.0))], start);
    bed.polled(&[(&second, slots(50.0))], start);
    bed.polled(&[(&third, slots(30.0))], start);
    assert_eq!(bed.taken(), [3, 0, 0, 0, 0, 0]);
    let handed = read([(a, Some(10.0)), (b, Some(50.0)), (c, Some(30.0))]);
    assert_eq!(bed.placed(b"polled", start + ms(10)).await?, (b, handed));

    // A record the third worker pushes later states more: each worker's
    // latest report stands, whichever way it came.
    let later = start + ms(1_000);
    bed.pushed(&third, 0, &slots(90.0), later);
    let handed = read([(a, Some(10.0)), (b, Some(50.0)), (c, Some(90.0))]);
    let placed = bed.placed(b"pushed", later + ms(10)).await?;
    assert_eq!(placed, (c, handed.clone()));

    // A value under another unit than the declared one is dropped and
    // counted: the first worker keeps the value it had.
    let wrong = later + ms(100);
    bed.pushed(&first, 0, &stating(0, "tokens", 500.0, None), wrong);
    assert_eq!(bed.placed(b"unit", wrong + ms(10)).await?, (c, handed));
    assert_eq!(bed.taken(), [4, 0, 1, 0, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn a_declared_values_age_is_its_own_and_its_reports_and_the_maximum_age_ends_it() -> Case {
    let bed = Bed::started(Some(SINGLE))?;
    let (a, first) = bed.worker("http://a:8000", Role::Single)?;
    let (b, second) = bed.worker("http://b:8000", Role::Single)?;
    let start = Instant::now();
    bed.polled(&[(&first, slots(90.0))], start);
    bed.polled(&[(&second, slots(10.0))], start);
    assert_eq!(bed.taken(), [2, 0, 0, 0, 0, 0]);
    let handed = read([(a, Some(90.0)), (b, Some(10.0))]);
    assert_eq!(bed.placed(b"fresh", start + ms(10)).await?, (a, handed));

    // Six seconds on, the first worker's value is past the signal's five
    // seconds and reads as none. The second states a value four seconds
    // old by its own word, which is still read.
    let aged = start + ms(6_000);
    bed.pushed(&second, 0, &stating(0, "slots", 20.0, Some(4_000)), aged);
    let handed = read([(a, None), (b, Some(20.0))]);
    assert_eq!(bed.placed(b"aged", aged + ms(10)).await?, (b, handed));

    // Two seconds later that value is six seconds old, its own age and
    // the time since its report together, and reads as none. A smaller
    // fresh value of the first worker wins.
    let ended = aged + ms(2_000);
    bed.pushed(&first, 0, &slots(5.0), ended);
    let handed = read([(a, Some(5.0)), (b, None)]);
    assert_eq!(bed.placed(b"ended", ended + ms(10)).await?, (a, handed));
    Ok(())
}

#[tokio::test]
async fn each_side_of_a_placed_pair_follows_its_own_pools_declared_values() -> Case {
    let bed = Bed::started(Some(PAIR))?;
    let (prefill_other, prefill_low) = bed.worker("http://p0:8000", Role::Prefill)?;
    let (prefill, prefill_high) = bed.worker("http://p1:8000", Role::Prefill)?;
    let (decode, decode_high) = bed.worker("http://d0:8000", Role::Decode)?;
    let (decode_other, decode_low) = bed.worker("http://d1:8000", Role::Decode)?;
    let now = Instant::now();
    // The prefill pool by its engines' polls, the decode pool by pushed
    // records.
    bed.polled(&[(&prefill_low, slots(10.0))], now);
    bed.polled(&[(&prefill_high, slots(70.0))], now);
    bed.pushed(&decode_high, 0, &slots(60.0), now);
    bed.pushed(&decode_low, 0, &slots(20.0), now);
    assert_eq!(bed.taken(), [4, 0, 0, 0, 0, 0]);
    let ask = Ask {
        phase: Phase::Prefill,
        prompt_tokens: 37,
        ..Ask::of(b"pair", MODEL)
    };
    let placed = bed.policy_engine.place_pair(&ask, None, None).await;
    let (placed, _guard) = placed.map_err(|unplaced| format!("not placed: {unplaced:?}"))?;
    // Each side read its own pool's rows, each row its own value.
    let handed = read([
        (prefill_other, Some(10.0)),
        (prefill, Some(70.0)),
        (decode, Some(60.0)),
        (decode_other, Some(20.0)),
    ]);
    assert_eq!(last_read(&bed.read), handed);
    assert_eq!((placed.id(), placed.decode_id()), (prefill, Some(decode)));
    Ok(())
}

#[tokio::test]
async fn each_rank_of_an_engine_reads_its_own_declared_value() -> Case {
    let bed = Bed::started(Some(SINGLE))?;
    let url = "http://ranked:8000";
    let (rank_zero, zero) = bed.rank("http://ranked:8000@0", url, Role::Single, 0, 2)?;
    let (rank_one, ranked) = bed.rank("http://ranked:8000@1", url, Role::Single, 1, 2)?;
    let (bystander, other) = bed.worker("http://other:8000", Role::Single)?;
    let start = Instant::now();
    // The ranked engine's one poll answers for both ranks, each rank's
    // answer its own, and rank 1 states the most.
    let ranks = [
        (&zero, stating(0, "slots", 10.0, None)),
        (&ranked, stating(1, "slots", 70.0, None)),
    ];
    bed.polled(&ranks, start);
    bed.polled(&[(&other, slots(8.0))], start);
    assert_eq!(bed.taken(), [3, 0, 0, 0, 0, 0]);
    let handed = read([
        (rank_zero, Some(10.0)),
        (rank_one, Some(70.0)),
        (bystander, Some(8.0)),
    ]);
    assert_eq!(
        bed.placed(b"ranks", start + ms(10)).await?,
        (rank_one, handed)
    );

    // Rank 1 pushes a smaller value: its own row moves, and no other.
    // Rank 0 now states the most, above the other engine. Had the record
    // moved rank 0 as well, the other engine would.
    let later = start + ms(1_000);
    bed.pushed(&ranked, 1, &stating(1, "slots", 5.0, None), later);
    let handed = read([
        (rank_zero, Some(10.0)),
        (rank_one, Some(5.0)),
        (bystander, Some(8.0)),
    ]);
    assert_eq!(
        bed.placed(b"moved", later + ms(10)).await?,
        (rank_zero, handed)
    );
    Ok(())
}

#[tokio::test]
async fn a_document_that_declares_no_signal_takes_no_value_and_places_as_before() -> Case {
    let bed = Bed::started(None)?;
    let (only, worker) = bed.worker("http://a:8000", Role::Single)?;
    let now = Instant::now();
    bed.polled(&[(&worker, slots(10.0))], now);
    bed.pushed(&worker, 0, &slots(20.0), now + ms(100));
    assert_eq!(bed.taken(), [0; 6]);
    // No stage of that document reads a signal, so its decision read none.
    assert_eq!(
        bed.placed(b"plain", now + ms(200)).await?,
        (only, Rows::new())
    );
    Ok(())
}

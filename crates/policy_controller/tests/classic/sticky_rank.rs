//! The classic sticky rank rows: one grouped worker, the engines behind it,
//! and the rank the gateway pins each request to by its sticky key. A row
//! of steps holds the rank each request carried, `null` where the gateway
//! set none; a counted row holds how many keys kept one rank, or what each
//! rank took. The replay decides each request over the rank rows of the
//! worker under the `consistent_hashing` preset, keyed by the request's
//! routing key.
//!
//! Two hashes place no key alike. A row of steps replays when a request
//! with no rank there has none here and the requests of one key share one
//! rank here, as they did there; a pairs row replays count for count and a
//! spread row share for share. [`APART`] names the rows this crate decides apart.

use std::time::Instant;

use serde::Deserialize;
use smg_policy_controller::{
    store::{Ask, GatewayId, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore},
    AppliedSpec, Constraints, Engine, Identity, PolicySpec, PoolFacts, RequestFacts, Role, Scratch,
};

use crate::{same_shares, shares};

/// What these rows record, as a sheet's head names it: the rank the
/// gateway's hash placement gives a request on one grouped worker.
pub const POLICY: &str = "sticky_rank";

const MODEL: &str = "m";

/// The worker's engine: what the rows of its ranks share.
const ENGINE: &str = "ipc:///tmp/w0.ipc";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A request with this routing key, or with none.
    Pick { key: Option<String> },
}

/// A counted row's keys, `key-0` to `key-<count - 1>`: asked twice each in a
/// pairs row, once each in a spread row.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Keys {
    count: u32,
    pairs: bool,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The rank each request carried, in order: `None` where the gateway
    /// set none.
    Sequence(Vec<Option<u16>>),
    /// A pairs row: the keys whose two requests took one rank, then the
    /// other keys. A spread row: what each rank took, the largest count
    /// first.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    /// The engines the worker names, where it names a count.
    ranks: Option<u16>,
    /// The ranks the worker's connection places a pin on, where it says.
    pinnable: Option<u16>,
    #[serde(default)]
    steps: Vec<Step>,
    keys: Option<Keys>,
    outcome: Outcome,
}

impl Row {
    /// The rank rows a host lists for the row's worker: the ranks a request
    /// can be pinned to. Those are the ranks the connection places a pin
    /// on, where it says, and the engines the worker names otherwise; a
    /// worker of which neither is known is one row.
    fn listed(&self) -> u16 {
        match (self.ranks, self.pinnable) {
            (Some(ranks), Some(pinnable)) => ranks.min(pinnable),
            (Some(ranks), None) => ranks,
            (None, Some(pinnable)) => pinnable,
            (None, None) => 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// The store with the rank rows of the row's worker in it and the engine
/// that decides over them.
struct Bed {
    store: MemoryStore,
    engine: Engine,
    out: ViewBuffers,
    scratch: Scratch,
    seed: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// preset.
    fn new(ranks: u16) -> Result<Self, String> {
        let mut document =
            PolicySpec::preset("consistent_hashing").map_err(|err| err.to_string())?;
        document.generation = 1;
        let engine = document.validate().map_err(|err| err.to_string())?;
        let applied = AppliedSpec::validate(&document).map_err(|err| err.to_string())?;
        let store = MemoryStore::new();
        for rank in 0..ranks {
            let key = if ranks == 1 {
                ENGINE.to_string()
            } else {
                format!("{ENGINE}@{rank}")
            };
            let id = store
                .upsert(WorkerFacts {
                    key,
                    engine: ENGINE.to_string(),
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_rank: rank,
                    dp_size: ranks,
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
        }
        store.apply(applied).expect("the first generation applies");
        Ok(Self {
            store,
            engine,
            out: ViewBuffers::default(),
            scratch: Scratch::new(),
            seed: 0,
        })
    }

    /// One request decided over the rank rows of the worker: the rank of
    /// its target, or `None` where the decision names no row.
    fn pick(&mut self, key: Option<&str>) -> Option<u16> {
        self.seed += 1;
        let now = Instant::now();
        let ask = Ask {
            gateway: GatewayId(1),
            now,
            pinned: None,
            affinity: false,
        };
        let built = self
            .store
            .build_view(MODEL, Role::Single, &ask, &mut self.out)?;
        let facts = RequestFacts {
            model: MODEL,
            session_key: key.map_or(Identity::NONE, Identity::plain),
            seed: self.seed,
            now_micros: self.store.micros(now),
            ..RequestFacts::default()
        };
        let pool = PoolFacts {
            pool_size: built.pool_size,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool,
            &self.out.candidates,
            &Constraints::default(),
            &mut self.scratch,
        );
        Some(decision.pick()?.targets.first()?.dp_rank)
    }
}

/// A recorded row this crate decides apart from the gateway, by a rule of
/// its own that `rule` states. Its replay must give every request one of
/// the worker's listed ranks and the requests of one key one rank. The
/// entry fails the day its row replays as recorded, so it cannot outlive
/// its rule.
struct Apart {
    row: &'static str,
    rule: &'static str,
}

/// The rows decided apart. The gateway sets no rank on a request with no
/// sticky key, on a worker of one engine or of no count it knows, and on a
/// key whose rank its connection places no pin on: the connection then
/// places the request on any engine. Here every request takes a row of the
/// worker: a key's owner on the ring of the listed ranks, and the picker's
/// draw for a request with no key.
const APART: &[Apart] = &[
    Apart {
        row: "a-request-with-no-key-takes-no-rank",
        rule: "a request with no key is the picker's draw among the worker's ranks",
    },
    Apart {
        row: "a-worker-of-one-engine-takes-no-rank",
        rule: "a worker of one row answers with its row",
    },
    Apart {
        row: "a-worker-that-names-no-count-takes-no-rank-where-its-connection-says-none",
        rule: "a worker of one row answers with its row",
    },
    Apart {
        row: "a-key-whose-rank-the-connection-places-no-pin-on-takes-none",
        rule: "a key's owner is one of the ranks the host lists, so every key is placed",
    },
];

/// What a row gives here, in the shape of its recorded outcome, or the
/// validator's words for a form this crate does not build.
fn replay(row: &Row) -> Result<Outcome, String> {
    let mut bed = Bed::new(row.listed())?;
    let Some(keys) = row.keys else {
        return Ok(Outcome::Sequence(
            row.steps
                .iter()
                .map(|Step::Pick { key }| bed.pick(key.as_deref()))
                .collect(),
        ));
    };
    let mut rank = |i: u32| bed.pick(Some(&format!("key-{i}")));
    let counts = if keys.pairs {
        let together = (0..keys.count)
            .filter(|&i| rank(i).is_some_and(|first| rank(i) == Some(first)))
            .count() as u32;
        vec![together, keys.count - together]
    } else {
        let mut counts = vec![0u32; usize::from(row.listed())];
        for i in 0..keys.count {
            let rank = rank(i).ok_or("a key took no rank")?;
            counts[usize::from(rank)] += 1;
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts
    };
    Ok(Outcome::Counts {
        draws: keys.count,
        counts,
    })
}

/// Whether a row of steps places its requests here as recorded: a request
/// with no rank there has none here, and the requests of one key share one
/// rank here. Two hashes place no key alike, so which rank a key takes, and
/// which keys meet on one, is each side's own.
fn same_ranks(row: &Row, recorded: &[Option<u16>], here: &[Option<u16>]) -> bool {
    recorded.len() == here.len()
        && recorded
            .iter()
            .zip(here)
            .all(|(there, here)| there.is_some() == here.is_some())
        && one_rank_a_key(row, here)
}

/// Whether the requests of one key took one rank.
fn one_rank_a_key(row: &Row, picks: &[Option<u16>]) -> bool {
    let asked = || row.steps.iter().zip(picks);
    asked().enumerate().all(|(at, (Step::Pick { key }, pick))| {
        key.is_none()
            || asked()
                .take(at)
                .all(|(Step::Pick { key: earlier }, took)| earlier != key || took == pick)
    })
}

/// Whether a replayed outcome is the recorded one: a rank for each request
/// that took one and one rank a key, count for count in a pairs row, share for share in a
/// spread row.
fn same_outcome(row: &Row, here: &Outcome) -> bool {
    match (&row.outcome, here) {
        (Outcome::Sequence(recorded), Outcome::Sequence(here)) => same_ranks(row, recorded, here),
        (
            Outcome::Counts {
                draws,
                counts: recorded,
            },
            Outcome::Counts { counts, .. },
        ) => {
            if row.keys.is_some_and(|keys| keys.pairs) {
                recorded == counts
            } else {
                same_shares(&shares(counts, *draws), &shares(recorded, *draws), *draws)
            }
        }
        _ => false,
    }
}

/// `None` when a row replays as recorded, or as its entry holds it when
/// [`APART`] names the row; the reason otherwise.
fn held(row: &Row, here: &Outcome) -> Option<String> {
    let same = same_outcome(row, here);
    let Some(apart) = APART.iter().find(|apart| apart.row == row.name) else {
        return (!same).then(|| format!("recorded {:?}, replayed {here:?}", row.outcome));
    };
    if same {
        return Some(format!(
            "it replays as recorded now ({}): take the row out of APART",
            apart.rule
        ));
    }
    let Outcome::Sequence(picks) = here else {
        return Some("a row decided apart is a row of steps".to_string());
    };
    let ranked = picks
        .iter()
        .all(|pick| pick.is_some_and(|rank| rank < row.listed()));
    let keyed = one_rank_a_key(row, picks);
    (!(ranked && keyed)).then(|| {
        format!(
            "recorded {:?}, apart by its entry ({}), replayed {here:?}",
            row.outcome, apart.rule
        )
    })
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    let sheet: Sheet = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!("{source} is not a sheet of {POLICY} rows: {err}")],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .filter_map(|row| {
            let why = match replay(row) {
                Ok(here) => held(row, &here),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// Every row decided apart is an inline row, so every run of the suite
/// checks that its rule still decides it.
#[test]
fn every_sticky_rank_row_decided_apart_is_an_inline_row() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    for apart in APART {
        assert!(
            sheet.rows.iter().any(|row| row.name == apart.row),
            "{} is decided apart and is no inline row",
            apart.row
        );
    }
}

/// The recorded sheet, whole: its rows are few.
pub const INLINE: &str = r#"{
  "recorded_from": "615542c4c69bb28fb3cb3e9e4ee29a8e2be0a7bc",
  "policy": "sticky_rank",
  "rows": [
    {
      "name": "a-key-takes-one-rank-on-every-request",
      "ranks": 8,
      "pinnable": null,
      "steps": [
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-1"}},
        {"pick": {"key": "conv-2"}},
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-1"}},
        {"pick": {"key": "conv-2"}}
      ],
      "outcome": {"sequence": [4, 5, 7, 4, 5, 7]}
    },
    {
      "name": "a-request-with-no-key-takes-no-rank",
      "ranks": 8,
      "pinnable": null,
      "steps": [
        {"pick": {"key": null}},
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": null}}
      ],
      "outcome": {"sequence": [null, 4, null]}
    },
    {
      "name": "a-worker-of-one-engine-takes-no-rank",
      "ranks": 1,
      "pinnable": null,
      "steps": [
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-1"}}
      ],
      "outcome": {"sequence": [null, null]}
    },
    {
      "name": "a-worker-that-names-no-count-takes-no-rank-where-its-connection-says-none",
      "ranks": null,
      "pinnable": null,
      "steps": [
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-1"}}
      ],
      "outcome": {"sequence": [null, null]}
    },
    {
      "name": "the-connection-s-ranks-stand-where-the-worker-names-no-count",
      "ranks": null,
      "pinnable": 4,
      "steps": [
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-2"}},
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-2"}}
      ],
      "outcome": {"sequence": [0, 3, 0, 3]}
    },
    {
      "name": "a-key-whose-rank-the-connection-places-no-pin-on-takes-none",
      "ranks": 8,
      "pinnable": 4,
      "steps": [
        {"pick": {"key": "conv-0"}},
        {"pick": {"key": "conv-1"}},
        {"pick": {"key": "conv-2"}},
        {"pick": {"key": "conv-3"}},
        {"pick": {"key": "conv-4"}},
        {"pick": {"key": "conv-5"}},
        {"pick": {"key": "conv-6"}},
        {"pick": {"key": "conv-7"}}
      ],
      "outcome": {"sequence": [null, null, null, null, null, 0, 3, 3]}
    },
    {
      "name": "every-key-keeps-one-rank",
      "ranks": 8,
      "pinnable": null,
      "keys": {"count": 512, "pairs": true},
      "outcome": {"counts": {"draws": 512, "counts": [512, 0]}}
    },
    {
      "name": "the-keys-spread-over-the-ranks",
      "ranks": 8,
      "pinnable": null,
      "keys": {"count": 4096, "pairs": false},
      "outcome": {"counts": {"draws": 4096, "counts": [545, 537, 536, 532, 513, 497, 487, 449]}}
    }
  ]
}"#;

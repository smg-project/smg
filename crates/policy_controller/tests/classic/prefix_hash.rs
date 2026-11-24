//! The classic prefix hash rows: a pool of workers, the requests a router
//! holds on them, and requests placed by a prompt's prefix, by a routing key
//! or by nothing to hash, under the policy's configuration.
//!
//! A prompt is runs of consecutive token ids, `[first, count]` each. The
//! classic policy hashes the tokens themselves. A decision here reads block
//! hashes a host made, so this side plays that host: whole blocks of the
//! view's block size, each block's hash made from the block's own ids and
//! nothing ahead of them, as the gateway's host hashes a prompt. A prompt
//! as text is each id spelled as four hex digits, and its blocks are windows
//! of as many characters as a token block has tokens.
//!
//! A ring row lists its workers in the ring order of the request it names.
//! This side finds that order on its own ring as the recorder found it on
//! the gateway's: the worker the request lands on with every worker ready
//! and idle, then with the members found so far down. A request that has no
//! key on this side has no ring order, and its row is read in its own order.
//!
//! A row that says where each of its requests stands on a ring is replayed
//! a second time on a ring its caller lends, built from the row
//! ([`replay_on_a_lent_ring`]): the ring reaches the row's workers in the
//! row's order from where the ring request stands, and in a level's order
//! from where the request's head at that level does. Its requests state
//! those places and no block hash.
//!
//! A pairs row places pairs of prompts on an idle pool and counts the pairs
//! whose two prompts took one worker, then the pairs that took two. Prompt
//! `i` of such a row is `tokens` ids from `1_000_000 + 1_000 * i` on, and
//! its twin differs at the one place `differ_at`, where it holds
//! `2_000_000_000 + i`. Where the row's prompts are `text`, each is carried
//! as text: prompt `i` is ids from `1_000 * i` on, and its twin holds
//! `65_000 + i`. A spread row places `count` such prompts once each
//! and counts what each worker took, the largest count first.

use serde::Deserialize;
use smg_policy_controller::{
    document::PolicySpec,
    engine::{
        fingerprint, Candidate, Constraints, Engine, HashKind, Health, Identity, PoolFacts,
        RequestFacts, Scratch, WorkerId,
    },
    policies::consistent_hashing::LentRing,
};

use crate::{same_outcome, Outcome};

/// The classic policy these rows record, as a sheet's head names it, and
/// the preset that replays it.
pub const POLICY: &str = "prefix_hash";

/// The tokens of a block and the characters of a text window on this side:
/// the view's default block size.
const BLOCK: usize = 16;

/// How many times a sequence row is decided, each from a fresh engine under
/// its own seeds: a row no draw decides gives one sequence every time.
const RUNS: u32 = 32;

/// The policy's configuration, in the gateway's own spelling.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Config {
    prefix_token_count: u32,
    load_factor: f64,
    balance_abs_threshold: u32,
    /// The boundaries a prompt is keyed at, ascending; none for a key at
    /// the one depth.
    #[serde(default)]
    cache_boundaries: Vec<u32>,
}

/// The gateway's default configuration, which the preset carries.
const DEFAULT: Config = Config {
    prefix_token_count: 256,
    load_factor: 1.25,
    balance_abs_threshold: 10,
    cache_boundaries: Vec::new(),
};

/// What a request carries to be placed by.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    tokens: Option<Vec<[u32; 2]>>,
    #[serde(default)]
    text: Option<Vec<[u32; 2]>>,
    /// The workers an earlier attempt of the request already tried.
    #[serde(default)]
    tried: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request.
    Pick(Request),
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Whether each worker is ready, from here on.
    Ready(Vec<bool>),
}

/// The prompts of a pairs row or a spread row.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prompts {
    count: u32,
    tokens: u32,
    #[serde(default)]
    differ_at: Option<u32>,
    /// Whether each prompt is carried as text.
    #[serde(default)]
    text: bool,
}

/// A boundary a level row's request reaches under its deepest one, with the
/// row's workers in the ring order of the request's head at that boundary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Level {
    boundary: u32,
    order: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    name: String,
    config: Config,
    /// A ring row's request: the row's workers are listed in its ring order.
    #[serde(default)]
    ring: Option<Request>,
    /// A level row: the shallower boundaries its request reaches.
    #[serde(default)]
    levels: Vec<Level>,
    /// The requests the host holds on each worker at the start.
    workers: Vec<u32>,
    #[serde(default)]
    steps: Vec<Step>,
    #[serde(default)]
    prompts: Option<Prompts>,
    outcome: Outcome,
}

impl Config {
    /// The preset as named for a row recorded under the gateway's default
    /// configuration, so the preset's own numbers are held to those rows;
    /// for any other row, the preset with the row's numbers in its ring
    /// stage and the row's boundaries in its view.
    fn spec(&self) -> PolicySpec {
        let preset = PolicySpec::preset(POLICY).expect("the preset's form");
        if *self == DEFAULT {
            return preset;
        }
        let mut document = serde_json::to_value(preset).expect("the preset serializes");
        document["view"]["boundaries"] = self.cache_boundaries.clone().into();
        let stage = &mut document["profiles"][POLICY]["score"][0]["consistent-hash"];
        stage["prefix_tokens"] = self.prefix_token_count.into();
        stage["bounded_load"] = serde_json::json!({
            "factor": self.load_factor,
            "abs_margin": self.balance_abs_threshold,
        });
        serde_json::from_value(document).expect("the preset with the row's numbers")
    }
}

fn ids(runs: &[[u32; 2]]) -> Vec<u32> {
    runs.iter()
        .flat_map(|&[first, count]| first..first + count)
        .collect()
}

/// The hashes of `bytes` in whole blocks of `block` bytes, as a host
/// states them: each is of its own block's content.
fn blocks(bytes: &[u8], block: usize) -> Vec<u64> {
    bytes.chunks_exact(block).map(fingerprint).collect()
}

/// A request as a host hands it to a decision: its block hashes, what they
/// were made over, and its size in tokens.
#[derive(Clone)]
struct Hashed {
    key: Option<String>,
    hashes: Vec<u64>,
    kind: HashKind,
    tokens: u32,
    tried: Vec<usize>,
    /// Where the request stands on the ring its caller lends, for a row
    /// replayed on one.
    places: Vec<u64>,
}

impl Hashed {
    /// Token ids hashed in blocks of [`BLOCK`] tokens.
    fn of_tokens(ids: &[u32]) -> Self {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        Self {
            key: None,
            hashes: blocks(&bytes, 4 * BLOCK),
            kind: HashKind::Token,
            tokens: ids.len() as u32,
            tried: Vec::new(),
            places: Vec::new(),
        }
    }

    /// Token ids spelled as text, four hex digits each, hashed in windows
    /// of [`BLOCK`] characters.
    fn of_text(ids: &[u32]) -> Self {
        let text: String = ids.iter().map(|id| format!("{id:04x}")).collect();
        Self {
            key: None,
            hashes: blocks(text.as_bytes(), BLOCK),
            kind: HashKind::Text,
            tokens: ids.len() as u32,
            tried: Vec::new(),
            places: Vec::new(),
        }
    }

    fn of(request: &Request) -> Self {
        let hashed = match (&request.tokens, &request.text) {
            (Some(runs), _) => Self::of_tokens(&ids(runs)),
            (None, Some(runs)) => Self::of_text(&ids(runs)),
            (None, None) => Self::of_tokens(&[]),
        };
        Self {
            key: request.key.clone(),
            tried: request.tried.clone(),
            ..hashed
        }
    }

    /// Whether the request has a key on this side: a routing key, or one
    /// whole block of its prompt.
    fn keyed(&self) -> bool {
        self.key.is_some() || !self.hashes.is_empty()
    }
}

/// The address of the worker a row lists at `i`, under a lent ring.
fn address(i: usize) -> String {
    format!("worker-{i}")
}

/// A caller's hash ring over a row's workers, built from the row: the walk
/// from the place `n` reaches them in the `n`th of these orders, as the
/// recorder found them on the gateway's ring.
#[derive(Debug)]
struct ByOrders(Vec<Vec<usize>>);

impl LentRing for ByOrders {
    fn position(&self, _: &str) -> u64 {
        0
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let order = self.0.get(position as usize).map_or(&[][..], Vec::as_slice);
        let _ = order.iter().any(|&i| take(&address(i)));
    }
}

/// The engine of the row's configuration over the row's pool. `places[i]`
/// is where the row's worker `i` stands in the pool: the row's own order,
/// or the ring order of a ring row's request.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    places: Vec<usize>,
    /// The ring the row's requests are lent, for a row replayed on one.
    lent: Option<ByOrders>,
    scratch: Scratch,
    /// The seed of the next decision: each has its own.
    seed: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form. `run` separates the seeds of a row's runs.
    fn new(config: &Config, loads: &[u32], places: &[usize], run: u32) -> Result<Self, String> {
        assert_eq!(places.len(), loads.len(), "a place of every worker");
        let engine = config.spec().validate().map_err(|err| err.to_string())?;
        let pool = (0..loads.len())
            .map(|i| Candidate {
                position: i as u32,
                id: WorkerId(1_000 + i as u64),
                key: 1_000 + i as u32,
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                ..Candidate::default()
            })
            .collect();
        let mut bed = Self {
            engine,
            pool,
            places: places.to_vec(),
            lent: None,
            scratch: Scratch::new(),
            seed: u64::from(run) * 1_000_000,
        };
        bed.loads(loads);
        Ok(bed)
    }

    /// The bed with `ring` lent to its requests: the pool's rows then
    /// carry their addresses' ids.
    fn lend(mut self, ring: ByOrders) -> Self {
        for row in &mut self.pool {
            row.id = WorkerId::from_address(&address(row.position as usize));
        }
        self.lent = Some(ring);
        self
    }

    fn loads(&mut self, loads: &[u32]) {
        assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
        for (&place, &load) in self.places.iter().zip(loads) {
            self.pool[place].in_flight = load;
        }
    }

    fn ready(&mut self, ready: &[bool]) {
        assert_eq!(ready.len(), self.pool.len(), "a word on every worker");
        for (&place, &ready) in self.places.iter().zip(ready) {
            self.pool[place].health.healthy = ready;
        }
    }

    /// One request decided: the place of its first target among the row's
    /// workers.
    fn place(&mut self, request: &Hashed) -> usize {
        self.seed += 1;
        let tried: Vec<WorkerId> = request
            .tried
            .iter()
            .map(|&i| self.pool[self.places[i]].id)
            .collect();
        let ring = self.lent.as_ref().map(|ring| ring as &dyn LentRing);
        let facts = RequestFacts {
            model: "m",
            session_key: request
                .key
                .as_deref()
                .map_or(Identity::NONE, Identity::plain),
            prompt_tokens: request.tokens,
            // A caller that lends its ring states where the request stands
            // on it: the stage is handed no block hash.
            block_hashes: if ring.is_some() { &[] } else { &request.hashes },
            hash_kind: request.kind,
            tried: &tried,
            seed: self.seed,
            ring,
            ring_positions: &request.places,
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
        let target = *decision
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target");
        self.places
            .iter()
            .position(|&place| place == target.position as usize)
            .expect("a worker of the row")
    }
}

/// The row's own order: worker `i` of the row is worker `i` of the pool.
fn in_order(workers: usize) -> Vec<usize> {
    (0..workers).collect()
}

/// The ring order of `request` over a pool of `workers` on this crate's
/// ring: the worker it lands on with every worker ready and idle, then with
/// the members found so far down, until every worker is placed.
fn ring_order(config: &Config, workers: usize, request: &Hashed) -> Result<Vec<usize>, String> {
    let mut order: Vec<usize> = Vec::with_capacity(workers);
    while order.len() < workers {
        let mut bed = Bed::new(config, &vec![0; workers], &in_order(workers), 0)?;
        let ready: Vec<bool> = (0..workers).map(|place| !order.contains(&place)).collect();
        bed.ready(&ready);
        let next = bed.place(request);
        assert!(
            !order.contains(&next),
            "the request's next member is a worker not yet placed, got {next}"
        );
        order.push(next);
    }
    Ok(order)
}

impl Prompts {
    fn prompt(self, i: u32) -> Vec<u32> {
        let first = if self.text { 0 } else { 1_000_000 };
        (0..self.tokens)
            .map(|place| first + 1_000 * i + place)
            .collect()
    }

    fn twin(self, i: u32, differ_at: u32) -> Vec<u32> {
        let mut twin = self.prompt(i);
        twin[differ_at as usize] = if self.text { 65_000 } else { 2_000_000_000 } + i;
        twin
    }
}

impl Row {
    /// Where each of the row's workers stands in the pool, and for a level
    /// row the prompt that stands in for its request on this crate's ring.
    fn places(&self) -> Result<(Vec<usize>, Option<Hashed>), String> {
        let workers = self.workers.len();
        match self.ring.as_ref().map(Hashed::of) {
            Some(_) if !self.levels.is_empty() => {
                let (own, places) = self.level_request()?;
                Ok((places, Some(own)))
            }
            Some(request) if request.keyed() => {
                Ok((ring_order(&self.config, workers, &request)?, None))
            }
            _ => Ok((in_order(workers), None)),
        }
    }

    /// A level row's request on this crate's ring. Two rings agree on no
    /// pair of keys, so the recorded prompt's two keys stand otherwise
    /// here: the row is replayed with the first prompt of the level rows'
    /// kind, as long as the recorded one, whose head at each shallower
    /// boundary puts the row's workers in the recorded order.
    fn level_request(&self) -> Result<(Hashed, Vec<usize>), String> {
        let workers = self.workers.len();
        let tokens = self
            .ring
            .as_ref()
            .and_then(|ring| ring.tokens.as_deref())
            .map_or(0, |runs| ids(runs).len() as u32);
        for i in 0..1_000u32 {
            let first = 3_000_000 + 1_000 * i;
            let prompt: Vec<u32> = (first..first + tokens).collect();
            let request = Hashed::of_tokens(&prompt);
            let places = ring_order(&self.config, workers, &request)?;
            let mut stands = true;
            for level in &self.levels {
                let head = prompt
                    .get(..level.boundary as usize)
                    .ok_or("a level past the row's request")?;
                let order: Vec<usize> =
                    ring_order(&self.config, workers, &Hashed::of_tokens(head))?
                        .iter()
                        .map(|worker| {
                            places
                                .iter()
                                .position(|place| place == worker)
                                .expect("a worker of the pool")
                        })
                        .collect();
                stands &= order == level.order;
            }
            if stands {
                return Ok((request, places));
            }
        }
        Err("no prompt among a thousand stands as the row's levels do".to_string())
    }

    /// Where `request` stands on the ring this row's caller lends: at the
    /// place `0` with the row's ring request where it shares that request's
    /// key or its head, the first tokens the configuration keys a prompt
    /// by, and behind it at the place of each of the row's levels. A
    /// request with nothing to hash stands nowhere. `None` where the row
    /// does not say: the request's key or head is another one.
    fn stands(&self, request: &Request) -> Option<Vec<u64>> {
        let head = |request: &Request| {
            let (runs, text) = match (&request.tokens, &request.text) {
                (Some(runs), _) => (runs, false),
                (None, Some(runs)) => (runs, true),
                (None, None) => return None,
            };
            let mut ids = ids(runs);
            ids.truncate(self.config.prefix_token_count as usize);
            // A prompt of no token has nothing to hash, as in classic.
            (!ids.is_empty()).then_some((ids, text))
        };
        if request.key.is_none() && head(request).is_none() {
            return Some(Vec::new());
        }
        let ring = self.ring.as_ref()?;
        if request.key.is_some() {
            return (request.key == ring.key).then(|| vec![0]);
        }
        (ring.key.is_none() && head(request) == head(ring))
            .then(|| (0..=self.levels.len() as u64).collect())
    }

    /// The row's steps decided once from a fresh engine: the worker each
    /// request went to. `own` stands in for a level row's request. Under
    /// `lent` the requests are lent a ring built from the row and state
    /// where they stand on it.
    fn run(
        &self,
        places: &[usize],
        own: Option<&Hashed>,
        run: u32,
        lent: bool,
    ) -> Result<Vec<usize>, String> {
        let mut bed = Bed::new(&self.config, &self.workers, places, run)?;
        if lent {
            let levels = self.levels.iter().map(|level| level.order.clone());
            let orders = std::iter::once(places.to_vec()).chain(levels);
            bed = bed.lend(ByOrders(orders.collect()));
        }
        let mut answers = Vec::new();
        for step in &self.steps {
            match step {
                Step::Pick(request) => {
                    let mut hashed = match own {
                        Some(own) if Some(request) == self.ring.as_ref() => own.clone(),
                        _ => Hashed::of(request),
                    };
                    if lent {
                        hashed.places = self.stands(request).unwrap_or_default();
                    }
                    answers.push(bed.place(&hashed));
                }
                Step::Loads(loads) => bed.loads(loads),
                Step::Ready(ready) => bed.ready(ready),
            }
        }
        Ok(answers)
    }

    /// A row of steps: the one sequence every run gives. Where a draw
    /// decides, the runs differ, and the outcome is how often each worker
    /// took the row's last request, a shape no recorded sequence has.
    fn sequence(&self, lent: bool) -> Result<Outcome, String> {
        let (places, own) = if lent {
            (in_order(self.workers.len()), None)
        } else {
            self.places()?
        };
        let first = self.run(&places, own.as_ref(), 0, lent)?;
        let mut counts = vec![0u32; self.workers.len()];
        let mut one = true;
        for run in 0..RUNS {
            let answers = self.run(&places, own.as_ref(), run, lent)?;
            one &= answers == first;
            if let Some(&last) = answers.last() {
                counts[last] += 1;
            }
        }
        Ok(if one {
            Outcome::Sequence(first)
        } else {
            Outcome::Counts {
                draws: RUNS,
                counts,
            }
        })
    }

    /// A pairs row or a spread row over the row's idle pool.
    fn counted(&self, prompts: Prompts) -> Result<Outcome, String> {
        let workers = self.workers.len();
        let mut bed = Bed::new(&self.config, &self.workers, &in_order(workers), 0)?;
        let mut place = |ids: &[u32]| {
            bed.place(&if prompts.text {
                Hashed::of_text(ids)
            } else {
                Hashed::of_tokens(ids)
            })
        };
        let counts = match prompts.differ_at {
            Some(differ_at) => {
                let together = (0..prompts.count)
                    .filter(|&i| place(&prompts.prompt(i)) == place(&prompts.twin(i, differ_at)))
                    .count() as u32;
                vec![together, prompts.count - together]
            }
            None => {
                let mut counts = vec![0u32; workers];
                for i in 0..prompts.count {
                    counts[place(&prompts.prompt(i))] += 1;
                }
                counts.sort_unstable_by(|a, b| b.cmp(a));
                counts
            }
        };
        Ok(Outcome::Counts {
            draws: prompts.count,
            counts,
        })
    }
}

impl Row {
    /// This crate's outcome for the row in the shape of the recorded one,
    /// or the validator's words when it refuses the row's form.
    fn replay(&self) -> Result<Outcome, String> {
        match self.prompts {
            Some(prompts) => self.counted(prompts),
            None => self.sequence(false),
        }
    }

    /// The row's outcome on a ring its caller lends, for a row that says
    /// where each of its requests stands on one ([`Self::stands`]). `None`
    /// for any other: a pairs row, a spread row.
    fn on_a_lent_ring(&self) -> Option<Result<Outcome, String>> {
        let stated = self.steps.iter().all(|step| match step {
            Step::Pick(request) => self.stands(request).is_some(),
            Step::Loads(_) | Step::Ready(_) => true,
        });
        (self.prompts.is_none() && stated).then(|| self.sequence(true))
    }
}

/// The outcome an entry of [`APART`] holds its row to: the picks of a row
/// of steps, or the counts of a pairs row.
#[derive(Debug)]
enum Here {
    Sequence(&'static [usize]),
    Counts(&'static [u32]),
}

impl Here {
    fn is(&self, outcome: &Outcome) -> bool {
        match (self, outcome) {
            (Self::Sequence(here), Outcome::Sequence(picks)) => picks == here,
            (Self::Counts(here), Outcome::Counts { counts, .. }) => counts == here,
            _ => false,
        }
    }
}

/// A recorded row this crate decides apart from the gateway, by a rule of
/// its own that `rule` states: its replay must give `here`. The row fails
/// the day it replays as recorded, so an entry cannot outlive its rule, and
/// on any third outcome.
struct Apart {
    row: &'static str,
    here: Here,
    rule: &'static str,
}

/// The rows decided apart. The gateway hashes a prompt's own tokens, so
/// every token of a prompt under the depth is in its key. A key here is the
/// fold of a request's whole blocks: the tokens past its last whole block
/// are outside the key, and a request under one block has no key and is
/// placed as one with nothing to hash, on the fewest requests.
const APART: &[Apart] = &[
    Apart {
        row: "a-ten-token-prompt-keeps-its-key-s-worker-over-the-fewest-requests",
        here: Here::Sequence(&[1, 1]),
        rule: "a request under one block carries no hash, so it has no key",
    },
    Apart {
        row: "the-last-token-of-a-prompt-under-256-tokens-is-in-the-key",
        here: Here::Counts(&[64, 0]),
        rule: "a key folds whole blocks, so the tokens past the last whole block are outside it",
    },
];

/// `None` when a row's outcome here is the recorded one, or its entry's
/// when [`APART`] names the row; the reason otherwise. A row decided apart
/// that replays as recorded is a defect of its entry, whose rule decides
/// nothing of it.
fn held(name: &str, recorded: &Outcome, here: &Outcome) -> Option<String> {
    let same = same_outcome(recorded, here);
    match APART.iter().find(|apart| apart.row == name) {
        None => (!same).then(|| format!("recorded {recorded:?}, replayed {here:?}")),
        Some(apart) if same => Some(format!(
            "it replays as recorded now ({}): take the row out of APART",
            apart.rule
        )),
        Some(apart) => (!apart.here.is(here)).then(|| {
            format!(
                "recorded {recorded:?}, apart by its entry {:?}, replayed {here:?}",
                apart.here
            )
        }),
    }
}

#[derive(Deserialize)]
struct Sheet {
    rows: Vec<Row>,
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
            let why = match row.replay() {
                Ok(here) => held(&row.name, &row.outcome, &here),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// The rows of a sheet that say where their requests stand on a ring,
/// replayed on a ring their caller lends: how many they are, and each one
/// whose answer is not the recorded one. No row is decided apart there: a
/// request's place on the ring is its caller's to state, whatever its
/// length.
pub fn replay_on_a_lent_ring(text: &str) -> (usize, Vec<String>) {
    let sheet: Sheet = serde_json::from_str(text).expect("a sheet of these rows");
    let replayed: Vec<(&Row, Result<Outcome, String>)> = sheet
        .rows
        .iter()
        .filter_map(|row| Some((row, row.on_a_lent_ring()?)))
        .collect();
    let defects = replayed.iter().filter_map(|(row, here)| match here {
        Ok(here) if same_outcome(&row.outcome, here) => None,
        Ok(here) => Some(format!(
            "{}: recorded {:?}, replayed {here:?}",
            row.name, row.outcome
        )),
        Err(why) => Some(format!("{}: refused here: {why}", row.name)),
    });
    (replayed.len(), defects.collect())
}

/// The rows that say where their requests stand on a ring, played through
/// the one call a gateway asks, over worker objects
/// ([`PolicyEngine::place`](smg_policy_controller::engine::PolicyEngine::place)).
/// The caller lends the row's ring and states no place: the engine cuts
/// each request into its heads and asks the lender where each stands.
#[cfg(feature = "in-process")]
pub mod through_the_one_call {
    use std::sync::Arc;

    use openai_protocol::worker::WorkerStatus;
    use smg_policy_controller::{
        document::AppliedSpec,
        engine::{
            HeaderRead, LentList, ListCell, PlaceRequest, PolicyEngine, Prompt, WorkerId,
            WorkerLists,
        },
        policies::consistent_hashing::{Head, LentRing},
    };
    use smg_worker::{identity::basic::Worker, test_support::ready_worker};

    use super::{ids, in_order, same_outcome, Outcome, Request, Row, Sheet, Step};

    const MODEL: &str = "m";

    /// The address of the worker a row lists at `i`.
    fn url(i: usize) -> String {
        format!("http://worker-{i}:8000")
    }

    /// A prompt carried as text: each id spelled as four hex digits.
    fn spelled(ids: &[u32]) -> String {
        ids.iter().map(|id| format!("{id:04x}")).collect()
    }

    /// A head as a row's lender keeps it.
    #[derive(Debug)]
    enum Kept {
        Tokens(Vec<u32>),
        Text(String),
    }

    /// A row's ring as its caller lends it. The head `heads[n]` stands at
    /// the place `n`, and the walk from there reaches the row's workers in
    /// the order `orders[n]`, as the recorder found them on the gateway's
    /// ring. A head the row says nothing of is no head of its requests.
    #[derive(Debug)]
    struct Lender {
        heads: Vec<Kept>,
        orders: Vec<Vec<usize>>,
    }

    impl LentRing for Lender {
        // The prefix form reads no place of a key's own text: a request
        // stated this one stands where the row's walk reaches no worker.
        fn position(&self, _: &str) -> u64 {
            u64::MAX
        }

        fn head_position(&self, head: Head<'_>) -> Option<u64> {
            let stands = |kept: &Kept| match (kept, head) {
                (Kept::Tokens(kept), Head::Tokens(ids)) => kept == ids,
                (Kept::Text(kept), Head::Text(text)) => kept == text,
                _ => false,
            };
            let at = self.heads.iter().position(stands);
            Some(at.expect("a head the row places") as u64)
        }

        fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
            let order = self.orders.get(position as usize);
            let _ = order.is_some_and(|order| order.iter().any(|&i| take(&url(i))));
        }
    }

    impl Row {
        /// The first `tokens` tokens of `request` as its lender keeps a
        /// head: its routing key whole, else its ids, else its text at
        /// four characters a token.
        fn kept(request: &Request, tokens: usize) -> Option<Kept> {
            if let Some(key) = &request.key {
                return Some(Kept::Text(key.clone()));
            }
            let cut = |runs: &[[u32; 2]]| {
                let mut ids = ids(runs);
                ids.truncate(tokens);
                ids
            };
            // Token ids lead a text, and an empty list of them leaves it.
            let ids_stated = |runs: &&Vec<[u32; 2]>| !runs.is_empty();
            match (request.tokens.as_ref().filter(ids_stated), &request.text) {
                (Some(runs), _) => Some(Kept::Tokens(cut(runs))),
                (None, Some(runs)) => Some(Kept::Text(spelled(&cut(runs)))),
                (None, None) => None,
            }
        }

        /// The ring the row's caller lends: the ring request's deepest
        /// head at the place `0`, where the walk reaches the workers in
        /// the row's own order, and its head at each level's boundary
        /// behind it. The deepest head is cut at the deepest boundary the
        /// request reaches, and at the configuration's depth where it
        /// reaches none.
        fn lender(&self) -> Lender {
            let mut heads = Vec::new();
            if let Some(ring) = &self.ring {
                // Token ids lead a text, and an empty list of them leaves
                // it, as in `kept`.
                let stated = ring.tokens.as_ref().filter(|runs| !runs.is_empty());
                let runs = stated.or(ring.text.as_ref());
                let length = runs.map_or(0, |runs| ids(runs).len());
                let reached = |&&at: &&u32| at as usize <= length;
                let boundaries = self.config.cache_boundaries.iter();
                let deepest = boundaries.filter(reached).max();
                let depth = deepest.unwrap_or(&self.config.prefix_token_count);
                heads.extend(Self::kept(ring, *depth as usize));
                for level in &self.levels {
                    heads.extend(Self::kept(ring, level.boundary as usize));
                }
            }
            let levels = self.levels.iter().map(|level| level.order.clone());
            let orders = std::iter::once(in_order(self.workers.len())).chain(levels);
            Lender {
                heads,
                orders: orders.collect(),
            }
        }

        /// The row's steps through the one call, once, from a fresh
        /// engine: the worker each request went to.
        fn through(&self) -> Result<Vec<usize>, String> {
            let spec = AppliedSpec::validate(&self.config.spec()).map_err(|err| err.to_string())?;
            // An engine as its host holds it: for the life of the process.
            let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
            let list: Arc<[Arc<dyn Worker>]> = (0..self.workers.len())
                .map(|i| ready_worker(&url(i), MODEL))
                .collect();
            // The list as its host lends it: with the cell it keeps beside it.
            let cell = ListCell::default();
            let members = LentList {
                workers: &list,
                cell: &cell,
                facts: None,
            };
            let hold = |loads: &[u32]| {
                for (worker, &load) in list.iter().zip(loads) {
                    while worker.load() < load as usize {
                        worker.increment_load();
                    }
                    while worker.load() > load as usize {
                        worker.decrement_load();
                    }
                }
            };
            hold(&self.workers);
            let lender = self.lender();
            let mut answers = Vec::new();
            for step in &self.steps {
                let request = match step {
                    Step::Pick(request) => request,
                    Step::Loads(loads) => {
                        hold(loads);
                        continue;
                    }
                    Step::Ready(ready) => {
                        for (worker, &ready) in list.iter().zip(ready) {
                            worker.set_status(if ready {
                                WorkerStatus::Ready
                            } else {
                                WorkerStatus::NotReady
                            });
                        }
                        continue;
                    }
                };
                let tokens = request.tokens.as_deref().map(ids);
                let text = request.text.as_deref().map(|runs| spelled(&ids(runs)));
                let keyed = request
                    .key
                    .as_deref()
                    .map(|key| [("x-smg-routing-key", key)]);
                let headers: &dyn HeaderRead = keyed.as_ref().map_or(&(), |keyed| keyed);
                let tried = request
                    .tried
                    .iter()
                    .map(|&i| WorkerId::from_address(&url(i)));
                let tried: Vec<WorkerId> = tried.collect();
                let asked = PlaceRequest {
                    prompt: Prompt {
                        tokens: tokens.as_deref(),
                        text: text.as_deref(),
                        ..Prompt::NONE
                    },
                    headers,
                    tried: &tried,
                    ring: Some(&lender),
                    ..PlaceRequest::of(MODEL, WorkerLists::Single(members))
                };
                let placement = engine.place(&asked).map_err(|why| format!("{why:?}"))?;
                let went = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
                answers.push(list.iter().position(went).expect("a worker of the row"));
            }
            Ok(answers)
        }
    }

    /// The rows of a sheet that say where their requests stand on a ring
    /// ([`Row::stands`]), each played four times through the one call: how
    /// many they are, and each one whose answer is not the recorded one
    /// every time.
    pub fn replay(text: &str) -> (usize, Vec<String>) {
        let sheet: Sheet = serde_json::from_str(text).expect("a sheet of these rows");
        let stated = |row: &&Row| row.on_a_lent_ring().is_some();
        let rows: Vec<&Row> = sheet.rows.iter().filter(stated).collect();
        let defect = |row: &&Row| {
            (0..4).find_map(|_| match row.through() {
                Ok(picks) if same_outcome(&row.outcome, &Outcome::Sequence(picks.clone())) => None,
                Ok(picks) => Some(format!(
                    "{}: recorded {:?}, through the one call {picks:?}",
                    row.name, row.outcome
                )),
                Err(why) => Some(format!("{}: refused here: {why}", row.name)),
            })
        };
        (rows.len(), rows.iter().filter_map(defect).collect())
    }
}

/// Rows copied from the recorded sheet: one or more of each kind, and every
/// row of the sheet on the replay's list.
pub const INLINE: &str = r#"{
  "recorded_from": "d92647556e91a3601ef647f12ae5784df2080dc3",
  "policy": "prefix_hash",
  "rows": [
    {
      "name": "a-prompt-lands-on-its-key-s-worker-every-time",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [0, 0, 0, 0],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [0, 0, 0]}
    },
    {
      "name": "prompts-that-share-their-first-256-tokens-land-together",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [0, 0, 0, 0],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"pick": {"tokens": [[100, 256], [9500, 300]]}}
      ],
      "outcome": {"sequence": [0, 0]}
    },
    {
      "name": "texts-that-share-their-first-1024-characters-land-together",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"text": [[100, 256], [9000, 44]]},
      "workers": [0, 0, 0, 0],
      "steps": [
        {"pick": {"text": [[100, 256], [9000, 44]]}},
        {"pick": {"text": [[100, 256], [9500, 300]]}}
      ],
      "outcome": {"sequence": [0, 0]}
    },
    {
      "name": "a-routing-key-leads-the-prompt",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"key": "k1"},
      "workers": [0, 0, 0, 0],
      "steps": [
        {"pick": {"key": "k1", "tokens": [[100, 256], [9000, 44]]}},
        {"pick": {"key": "k1", "tokens": [[20000, 300]]}}
      ],
      "outcome": {"sequence": [0, 0]}
    },
    {
      "name": "a-key-whose-worker-is-down-passes-to-the-ring-s-next-member-and-returns",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [0, 0, 0, 0],
      "steps": [
        {"ready": [false, true, true, true]},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"ready": [true, true, true, true]},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [1, 0]}
    },
    {
      "name": "a-ten-token-prompt-keeps-its-key-s-worker-over-the-fewest-requests",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[40000, 10]]},
      "workers": [3, 0, 1, 2],
      "steps": [
        {"pick": {"tokens": [[40000, 10]]}},
        {"pick": {"tokens": [[40000, 10]]}}
      ],
      "outcome": {"sequence": [0, 0]}
    },
    {
      "name": "a-passed-key-takes-the-fewest-requests-among-forty",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [50, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [37]}
    },
    {
      "name": "an-owner-past-both-margins-passes-its-key-to-the-fewest-requests",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [140, 100, 110, 90],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "a-passed-key-takes-the-fewest-requests-and-not-the-ring-s-next-member",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [30, 5, 2, 3],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [2]}
    },
    {
      "name": "an-owner-at-the-bound-keeps-its-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [14, 1, 0, 0],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [0]}
    },
    {
      "name": "the-request-being-placed-counts-in-the-mean",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [21, 0],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"loads": [22, 0]},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [0, 1]}
    },
    {
      "name": "the-two-margins-are-the-configuration-s",
      "config": {"prefix_token_count": 256, "load_factor": 2.0, "balance_abs_threshold": 4},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [14, 3, 4, 5],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "an-owner-at-1-3-times-the-mean-keeps-its-key-under-a-factor-of-1-3",
      "config": {"prefix_token_count": 256, "load_factor": 1.3, "balance_abs_threshold": 10},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [52, 27],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}},
        {"loads": [53, 27]},
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [0, 1]}
    },
    {
      "name": "a-product-that-rounds-under-the-owner-s-count-passes-its-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.2, "balance_abs_threshold": 2},
      "ring": {"tokens": [[100, 256], [9000, 44]]},
      "workers": [14, 9, 11],
      "steps": [
        {"pick": {"tokens": [[100, 256], [9000, 44]]}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "a-request-with-nothing-to-hash-takes-the-fewest-requests",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [3, 1, 2, 5],
      "steps": [
        {"pick": {}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "the-256th-token-is-in-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 300, "differ_at": 255},
      "outcome": {"counts": {"draws": 64, "counts": [15, 49]}}
    },
    {
      "name": "the-257th-token-is-outside-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 300, "differ_at": 256},
      "outcome": {"counts": {"draws": 64, "counts": [64, 0]}}
    },
    {
      "name": "the-256th-token-of-a-text-is-in-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 300, "differ_at": 255, "text": true},
      "outcome": {"counts": {"draws": 64, "counts": [13, 51]}}
    },
    {
      "name": "the-257th-token-of-a-text-is-outside-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 300, "differ_at": 256, "text": true},
      "outcome": {"counts": {"draws": 64, "counts": [64, 0]}}
    },
    {
      "name": "the-last-token-of-a-prompt-under-256-tokens-is-in-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 100, "differ_at": 99},
      "outcome": {"counts": {"draws": 64, "counts": [6, 58]}}
    },
    {
      "name": "two-thousand-prompts-spread-over-eight-workers",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 2000, "tokens": 300},
      "outcome": {"counts": {"draws": 2000, "counts": [311, 276, 272, 243, 239, 229, 224, 206]}}
    },
    {
      "name": "under-boundaries-a-worker-under-the-bound-keeps-the-deepest-boundary-s-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "ring": {"tokens": [[3059000, 300]]},
      "levels": [{"boundary": 64, "order": [2, 0, 1, 3]}],
      "workers": [12, 0, 5, 1],
      "steps": [
        {"pick": {"tokens": [[3059000, 300]]}}
      ],
      "outcome": {"sequence": [0]}
    },
    {
      "name": "a-worker-past-the-bound-passes-the-key-to-the-next-boundary-s-worker",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "ring": {"tokens": [[3059000, 300]]},
      "levels": [{"boundary": 64, "order": [2, 0, 1, 3]}],
      "workers": [30, 0, 5, 1],
      "steps": [
        {"pick": {"tokens": [[3059000, 300]]}}
      ],
      "outcome": {"sequence": [2]}
    },
    {
      "name": "workers-past-the-bound-at-every-boundary-pass-the-key-to-the-fewest-requests",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "ring": {"tokens": [[3059000, 300]]},
      "levels": [{"boundary": 64, "order": [2, 0, 1, 3]}],
      "workers": [30, 3, 28, 0],
      "steps": [
        {"pick": {"tokens": [[3059000, 300]]}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "a-down-worker-at-the-next-boundary-hands-the-key-to-that-ring-s-next-member",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "ring": {"tokens": [[3017000, 300]]},
      "levels": [{"boundary": 64, "order": [2, 3, 0, 1]}],
      "workers": [30, 0, 5, 1],
      "steps": [
        {"ready": [true, true, false, true]},
        {"pick": {"tokens": [[3017000, 300]]}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "under-boundaries-the-64th-token-of-a-prompt-that-reaches-64-alone-is-in-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 100, "differ_at": 63},
      "outcome": {"counts": {"draws": 64, "counts": [7, 57]}}
    },
    {
      "name": "under-boundaries-the-65th-token-of-a-prompt-that-reaches-64-alone-is-outside-the-key",
      "config": {"prefix_token_count": 256, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 100, "differ_at": 64},
      "outcome": {"counts": {"draws": 64, "counts": [64, 0]}}
    },
    {
      "name": "a-prompt-that-reaches-a-boundary-is-not-keyed-at-the-one-depth",
      "config": {"prefix_token_count": 128, "load_factor": 1.25, "balance_abs_threshold": 10, "cache_boundaries": [64, 256]},
      "workers": [0, 0, 0, 0, 0, 0, 0, 0],
      "prompts": {"count": 64, "tokens": 300, "differ_at": 200},
      "outcome": {"counts": {"draws": 64, "counts": [9, 55]}}
    }
  ]
}"#;

mod tests {
    use serde_json::{json, Value};

    use super::{replay_rows, Here, APART, INLINE, POLICY};

    /// What does not replay of `row` as the one row of a sheet.
    fn alone(row: &Value) -> Vec<String> {
        let sheet = json!({ "recorded_from": "a commit", "policy": POLICY, "rows": [row] });
        replay_rows(&sheet.to_string(), "a sheet").1
    }

    /// Every row decided apart is an inline row, so a run that reads no
    /// sheet still holds it. As recorded it is no defect; recorded as this
    /// crate decides it, it replays, and its entry answers for that.
    #[test]
    fn a_row_decided_apart_is_held_to_its_entry_until_it_replays() {
        let sheet: Value = serde_json::from_str(INLINE).expect("the inline sheet");
        let rows = sheet["rows"].as_array().expect("its rows");
        for apart in APART {
            let row = rows
                .iter()
                .find(|row| row["name"] == apart.row)
                .unwrap_or_else(|| panic!("{} is no inline row", apart.row));
            assert_eq!(alone(row), Vec::<String>::new(), "{}", apart.row);
            let mut as_here = row.clone();
            as_here["outcome"] = match apart.here {
                Here::Sequence(picks) => json!({ "sequence": picks }),
                Here::Counts(counts) => {
                    json!({ "counts": { "draws": counts.iter().sum::<u32>(), "counts": counts } })
                }
            };
            let defects = alone(&as_here);
            assert!(
                matches!(defects.as_slice(), [one]
                    if one.contains("take the row out of APART") && one.contains(apart.rule)),
                "{}: {defects:#?}",
                apart.row
            );
        }
    }

    /// A row no entry names is held to what was recorded: one pick moved in
    /// its sheet is one defect.
    #[test]
    fn a_row_no_entry_names_is_held_to_what_was_recorded() {
        let sheet: Value = serde_json::from_str(INLINE).expect("the inline sheet");
        let mut row = sheet["rows"][0].clone();
        assert_eq!(alone(&row), Vec::<String>::new());
        row["outcome"] = json!({ "sequence": [1, 0, 0] });
        let defects = alone(&row);
        assert!(
            matches!(defects.as_slice(), [one] if one.contains("recorded Sequence([1, 0, 0])")),
            "{defects:#?}"
        );
    }
}

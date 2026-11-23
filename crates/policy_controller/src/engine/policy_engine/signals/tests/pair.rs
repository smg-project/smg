//! The cases of a pair's two sides: each side reads the named values of its
//! own list. Each decides through the engine's own entries, over two lists
//! of worker objects and a load snapshot, and hands no cell.

use super::*;
use crate::engine::{
    pair_classes::PartnerClasses,
    policy_engine::pair::{PairAdmission, PairLists},
    view::Phase,
};

/// What a side's stage read of the signal `held`: the side, each row it was
/// asked about by the row's place in that side's list, and the value read.
type SideRead = Arc<Mutex<Vec<(Phase, u32, Option<f64>)>>>;

/// A scorer of a host's own build that reads the signal `held` of every row
/// it is asked about, and keeps what it read with the side it was asked on.
#[derive(Debug)]
struct ReadsSide {
    held: SignalColumn,
    read: SideRead,
}

impl Lifecycle for ReadsSide {}

impl Scorer for ReadsSide {
    fn name(&self) -> &'static str {
        "reads-side"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let rows = kept.iter().map(|&row| &candidates[row as usize]);
        let read = rows.map(|row| (facts.phase, row.position, self.held.read(facts, row)));
        self.read.lock().extend(read);
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..Needs::default()
        }
    }
}

/// A scorer of a host's own build that reads no signal, and keeps the side
/// of each decision it scored with whether its facts carried signal cells.
#[derive(Debug)]
struct SeesSide(Arc<Mutex<Vec<(Phase, bool)>>>);

impl Lifecycle for SeesSide {}

impl Scorer for SeesSide {
    fn name(&self) -> &'static str {
        "sees-side"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let seen = (facts.phase, facts.signal_cells.is_some());
        self.0.lock().push(seen);
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// An engine of a pair document with the two scorers above among its
/// stages, and what each kept.
struct Sides {
    engine: PolicyEngine,
    read: SideRead,
    seen: Arc<Mutex<Vec<(Phase, bool)>>>,
}

/// The engine of `document`, a pair document as JSON.
fn sides(document: &str) -> Sides {
    let document: PolicySpec = serde_json::from_str(document).expect("a well-formed document");
    let (read, seen) = (SideRead::default(), Arc::new(Mutex::new(Vec::new())));
    let (kept, saw) = (Arc::clone(&read), Arc::clone(&seen));
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer_with("reads-side", move |build| {
        Ok(Box::new(ReadsSide {
            held: build.signal_column("held")?,
            read: Arc::clone(&kept),
        }))
    });
    added.expect("a free name");
    let added = stages.scorer("sees-side", move || Box::new(SeesSide(Arc::clone(&saw))));
    added.expect("a free name");
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    Sides {
        engine: PolicyEngine::new(spec.expect("the stages hold the two scorers")),
        read,
        seen,
    }
}

/// A pair document under the signals `signals` declares: the profile of
/// each side as JSON, bound to every model.
fn pair(signals: &str, prefill: &str, decode: &str) -> String {
    format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{{signals}}},
            "profiles":{{"prefill":{prefill},"decode":{decode}}},
            "bindings":[{{"match":{{"model":"*"}},"plan":"pd-pair",
                          "profiles":{{"prefill":"prefill","decode":"decode"}}}}]}}"#
    )
}

/// The signal `held`, in tokens.
const HELD: &str = r#""held":{"unit":"tokens"}"#;
/// The two token counts a report offers by name.
const COUNTS: &str = r#""kv_tokens_capacity":{"unit":"tokens"},"kv_tokens_used":{"unit":"tokens"}"#;
/// A side's profile that reads `held` of every worker it scores.
const READS: &str = r#"{"score":["reads-side"],"pick":"best"}"#;
/// A side's profile that reads no signal.
const SEES: &str = r#"{"score":["sees-side"],"pick":"best"}"#;

/// A ready worker of `worker_type` at `url`: one rank of a data-parallel
/// worker of two where `rank` names it, and a worker of one rank otherwise.
fn typed(url: &str, worker_type: WorkerType, rank: Option<usize>) -> Arc<dyn Worker> {
    let builder = <BasicWorkerBuilder>::new(url)
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(MODEL))
        .health_config(no_health_check());
    let worker = match rank {
        Some(rank) => builder.dp_config(rank, 2).build(),
        None => builder.build(),
    };
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A host with no limit of its own: every prefill has capacity and every
/// decode a room.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A pair's host: its two lists of ready workers as it publishes them, the
/// prefills' first, and every prefill paired with every decode.
struct Lists {
    members: [PublishedList; 2],
    classes: PartnerClasses,
}

/// A pair's request stated ahead over its host's two lists, with the
/// snapshot its host gives it, where it gives one.
struct AsksPair<'a> {
    request: Request<'a>,
    lists: PairLists<'a>,
    loads: Option<Arc<LoadSnapshot>>,
}

impl RequestMaker for AsksPair<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Pair(self.lists)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads.as_ref()
    }
}

impl Lists {
    /// `prefills` prefills and `decodes` decodes, each a worker of one rank.
    fn of(prefills: usize, decodes: usize) -> Self {
        let side = |name: &str, worker_type, count: usize| -> Arc<[Arc<dyn Worker>]> {
            let url = |at: usize| format!("http://{name}-{at}:8000");
            (0..count)
                .map(|at| typed(&url(at), worker_type, None))
                .collect()
        };
        let prefills = side("prefill", WorkerType::Prefill, prefills);
        Self::over(prefills, side("decode", WorkerType::Decode, decodes))
    }

    /// The two lists as their host built them.
    fn over(prefills: Arc<[Arc<dyn Worker>]>, decodes: Arc<[Arc<dyn Worker>]>) -> Self {
        let all: Vec<u32> = (0..decodes.len() as u32).collect();
        let every = (0..prefills.len()).map(|_| all.as_slice());
        Self {
            classes: PartnerClasses::number(decodes.len(), every),
            members: [prefills, decodes].map(PublishedList::new),
        }
    }

    /// A snapshot of the report `of` makes for the worker at each place of
    /// each side: the prefills are side `0` and the decodes side `1`.
    fn reports(&self, of: impl Fn(usize, usize) -> WorkerLoadResponse) -> Arc<LoadSnapshot> {
        let sides = self.members.iter().enumerate();
        let reports = sides.flat_map(|(side, list)| {
            let places = list.workers().iter().enumerate();
            let of = &of;
            places.map(move |(at, worker)| (worker.url().to_owned(), of(side, at)))
        });
        LoadSnapshot::from_loads_for_test(reports.collect())
    }

    /// A snapshot in which each worker states `held` at its side's base
    /// more than its place in its list: the prefills' base first.
    fn stepped(&self, bases: [f64; 2]) -> Arc<LoadSnapshot> {
        self.reports(|side, at| report(held(bases[side] + at as f64)))
    }

    /// A snapshot in which the decode at `fits` alone has room for a
    /// request, by the two token counts its report offers.
    fn room_at(&self, fits: usize) -> Arc<LoadSnapshot> {
        self.reports(|side, at| {
            let used = if side == 1 && at == fits { 100 } else { 2_000 };
            holds(used, 1_000)
        })
    }

    /// The pair's request of `seed` over the two lists, every pair open,
    /// with the snapshot its host gives.
    fn asks(&self, seed: u64, loads: Option<&Arc<LoadSnapshot>>) -> AsksPair<'_> {
        let [prefill, decode] = &self.members;
        let lists = PairLists {
            prefill: prefill.lent(),
            decode: decode.lent(),
            classes: &self.classes,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let facts = RequestFacts {
            model: MODEL,
            phase: Phase::Prefill,
            seed,
            ..RequestFacts::default()
        };
        let request = Request {
            id: b"request",
            facts,
            constraints: Constraints::default(),
        };
        AsksPair {
            request,
            lists,
            loads: loads.cloned(),
        }
    }
}

/// The whole first decision of a pair's request at `now`, by the entry that
/// issues its lease.
fn pair_decided(engine: &PolicyEngine, asks: &AsksPair<'_>, now: Instant) -> Decision {
    let decided = engine.decide_under(GatewayId::LOCAL, asks, now);
    let whole = decided.expect("an answer").whole();
    whole.expect("a whole decision").0.decision
}

/// The pair a decision names first: its prefill's place in the prefill
/// list and its decode's place in the decode list.
fn pair_of(decision: &Decision) -> (usize, usize) {
    let pick = decision.pick().expect("a pick");
    let first = pick.targets.first().expect("a target");
    let pair = first.pair.expect("a pair");
    (first.position as usize, pair.decode_position as usize)
}

/// What the sides' stages read since the last call: each row a side was
/// asked about once, the prefills' first, in each list's order.
fn read_of(sides: &Sides) -> Vec<(Phase, u32, Option<f64>)> {
    let mut read = mem::take(&mut *sides.read.lock());
    read.sort_by_key(|&(side, place, _)| (side == Phase::Decode, place));
    read.dedup();
    read
}

/// What a side reads under [`Lists::stepped`] at its `base`, over `workers`
/// workers.
fn stepped_side(side: Phase, workers: u32, base: f64) -> Vec<(Phase, u32, Option<f64>)> {
    let value = |at| Some(base + f64::from(at));
    (0..workers).map(|at| (side, at, value(at))).collect()
}

/// Each side of a pair reads its own list's values, each at the worker's
/// place in that side's list. The two lists here are of different lengths
/// and state different values at the same places. The host hands the
/// snapshot and no cell.
#[test]
fn each_side_of_a_pair_reads_its_own_lists_values() {
    let sides = sides(&pair(HELD, READS, READS));
    let lists = Lists::of(3, 4);
    let loads = lists.stepped([10.0, 100.0]);
    for seed in 0..8 {
        let asks = lists.asks(seed, Some(&loads));
        let decision = pair_decided(&sides.engine, &asks, Instant::now());
        assert!(decision.pick().is_some(), "seed {seed}");
        let mut both = stepped_side(Phase::Prefill, 3, 10.0);
        both.extend(stepped_side(Phase::Decode, 4, 100.0));
        assert_eq!(read_of(&sides), both, "seed {seed}");
    }
    let converted = |list: &PublishedList| sides.engine.conversions(list.lent());
    let converted = lists.members.each_ref().map(converted);
    assert_eq!(converted, [1, 1], "one conversion a list for the snapshot");
}

/// A rule bounded by a named value keeps the decodes of a pair that the
/// request fits: the pair's decode is the one decode whose report says so,
/// at the entry that issues a lease and at the entry that issues none. The
/// decode decided again behind that entry's first decision reads the
/// snapshot of its own call, where another decode alone has room.
#[test]
fn a_rule_bounded_by_a_named_value_keeps_the_decodes_the_request_fits() {
    let plain = r#"{"score":["least-requests"],"pick":"best"}"#;
    let ruled =
        format!(r#"{{"filter":["healthy",{ROOM}],"score":["least-requests"],"pick":"best"}}"#);
    let sides = sides(&pair(COUNTS, plain, &ruled));
    let (engine, lists) = (&sides.engine, Lists::of(2, 4));
    let now = Instant::now();
    for seed in 0..16 {
        let fits = (seed % 4) as usize;
        let mut asks = lists.asks(seed, Some(&lists.room_at(fits)));
        asks.request.facts.prompt_tokens = 200;
        let leased = pair_decided(engine, &asks, now);
        assert_eq!(pair_of(&leased).1, fits, "seed {seed}");
        let bare = engine.decide_without_lease(&asks, now);
        let (bare, _) = bare.expect("a decision");
        assert_eq!(pair_of(&bare).1, fits, "seed {seed}, no lease");

        let pick = bare.pick().expect("a pick");
        let held = *pick.targets.first().expect("a target");
        let moved = (fits + 1) % 4;
        asks.loads = Some(lists.room_at(moved));
        let again = engine.decide_decode_without_lease(&asks, &held, now);
        let again = pair_of(&again.expect("a decision"));
        assert_eq!(again, (held.position as usize, moved), "seed {seed}");
    }
}

/// The stock scorer of costs reads a named signal on a pair's prefill side
/// (`signal.<name>`): the prefill whose report states the least is the
/// pair's, wherever it stands in its list.
#[test]
fn a_cost_term_of_a_named_signal_names_the_prefill_that_states_the_least() {
    let costed = r#"{"score":[{"least-cost":{"terms":{"signal.held":1},
                                               "unreported":"missing-as-zero"}}],
                     "pick":"best"}"#;
    let sides = sides(&pair(HELD, costed, SEES));
    let lists = Lists::of(4, 2);
    for least in 0..4 {
        let value = |side, at| {
            if side == 0 && at == least {
                10.0
            } else {
                500.0
            }
        };
        let loads = lists.reports(|side, at| report(held(value(side, at))));
        for seed in 0..8 {
            let asks = lists.asks(seed, Some(&loads));
            let decision = pair_decided(&sides.engine, &asks, Instant::now());
            assert_eq!(pair_of(&decision).0, least, "seed {seed}");
        }
    }
}

/// Each rank of a data-parallel decode that is a row of its own in the
/// decode list reads its own rank's value. The two decodes here are the two
/// ranks of one worker, and each pushes its records through the worker
/// crate's own monitor, whose report of a row holds that rank alone.
#[tokio::test]
async fn each_rank_that_is_a_row_of_the_decode_list_reads_its_own_value() {
    let sides = sides(&pair(HELD, SEES, READS));
    let prefills = [typed("http://prefill-0:8000", WorkerType::Prefill, None)];
    let ranked = |rank| typed("http://ranked:8000", WorkerType::Decode, Some(rank));
    let decodes: Arc<[Arc<dyn Worker>]> = [0, 1].map(ranked).into();
    let lists = Lists::over(prefills.into(), Arc::clone(&decodes));
    let (monitor, channel) = monitor_over(&decodes);
    let start = Instant::now();
    monitor.apply_pushed_load(&decodes[0], 0, &pushes(5.0), start);
    monitor.apply_pushed_load(&decodes[1], 1, &pushes(7.0), start);
    let loads = published(&monitor, &channel);
    let decision = pair_decided(&sides.engine, &lists.asks(7, Some(&loads)), start);
    assert!(decision.pick().is_some());
    assert_eq!(
        read_of(&sides),
        [(Phase::Decode, 0, Some(5.0)), (Phase::Decode, 1, Some(7.0))]
    );
}

/// What a polled engine of two ranks answers on its load route: both ranks
/// in one answer, each stating `held` in tokens. The second rank's value
/// was 200 ms old when the engine made the answer.
const TWO_RANKS: &str = r#"{"loads":[
    {"dp_rank":0,"values":[{"name":"held","unit":"tokens","value":5.0}]},
    {"dp_rank":1,"values":[{"name":"held","unit":"tokens","value":7.0,"age_ms":200}]}]}"#;

/// Each rank of a polled data-parallel engine that is a row of its own
/// reads its own rank's values, on a single request's list and on a pair's
/// decode side. No rank pushes here: the worker crate's own monitor polls
/// the engine's load route once for its two rank workers and publishes
/// each the report of its one rank, which counts one rank and is read. The
/// second rank's value states an age of its own and ends by it on both
/// paths, counted from the report's sample instant, the monitor's read of
/// the answer.
#[tokio::test]
async fn each_rank_row_of_a_polled_engine_reads_its_own_values() {
    let aged = r#""held":{"unit":"tokens","max_age_ms":500}"#;
    let single = built(&reads(r#","max_age_ms":500"#, "best"));
    let sides = sides(&pair(aged, SEES, READS));
    let (engine, _served) = polled_engine(TWO_RANKS).await;
    let ranked = |rank| typed(&engine, WorkerType::Decode, Some(rank));
    let decodes: Arc<[Arc<dyn Worker>]> = [0, 1].map(ranked).into();
    let (monitor, channel) = monitor_over(&decodes);
    let group = WorkerGroupKey {
        model_id: MODEL.into(),
        worker_type: WorkerType::Decode,
        connection_mode: ConnectionMode::Http,
    };
    poll_group_once_for_test(&monitor, &group, Duration::from_secs(10), Instant::now()).await;
    let loads = Arc::clone(&channel.borrow());
    let (report, sampled) = loads.report(decodes[1].url()).expect("a report");
    assert_eq!((report.loads.len(), report.dp_rank_count), (1, 1));

    let members = PublishedList::new(Arc::clone(&decodes));
    let prefills = [typed("http://prefill-0:8000", WorkerType::Prefill, None)];
    let lists = Lists::over(prefills.into(), Arc::clone(&decodes));
    for (millis, second) in [(300, Some(7.0)), (301, None)] {
        let now = sampled + Duration::from_millis(millis);
        let read = read_at(&single, &asks(7, &members, Some(&loads)), now);
        assert_eq!(read, [(0, Some(5.0)), (1, second)], "{millis} ms");
        let decision = pair_decided(&sides.engine, &lists.asks(7, Some(&loads)), now);
        assert!(decision.pick().is_some());
        assert_eq!(
            read_of(&sides),
            [(Phase::Decode, 0, Some(5.0)), (Phase::Decode, 1, second)],
            "{millis} ms"
        );
    }
}

/// The decode decided again reads the values of the snapshot of its own
/// call, and none its pair's first decision read: here every worker's value
/// moved between the two calls. That call asks no prefill stage, so
/// nothing is read on the prefill side and the prefill list is asked
/// nothing: its host hands new values of the two lists for the call, and
/// the prefills' keeps no word behind it.
#[test]
fn the_decode_decided_again_reads_the_snapshot_of_its_own_call() {
    let sides = sides(&pair(HELD, READS, READS));
    let lists = Lists::of(2, 3);
    let now = Instant::now();
    let mut sent = lists.asks(7, Some(&lists.stepped([10.0, 100.0])));
    sent.request.constraints.decode_later = true;
    let guard = DecisionGuard::decide_under(&sides.engine, GatewayId::LOCAL, &sent, now);
    let mut guard = guard.expect("a decision");
    let mut first = stepped_side(Phase::Prefill, 2, 10.0);
    first.extend(stepped_side(Phase::Decode, 3, 100.0));
    assert_eq!(read_of(&sides), first);
    guard.dispatched(now);

    let [prefills, decodes] = lists
        .members
        .each_ref()
        .map(|list| Arc::clone(list.workers()));
    let anew = Lists::over(prefills, decodes);
    let mut call = anew.asks(7, Some(&anew.stepped([50.0, 500.0])));
    call.request.constraints.decode_later = true;
    let again = guard.decide_decode_under(&call, now).expect("a decision");
    assert!(again.pick().is_some());
    assert_eq!(read_of(&sides), stepped_side(Phase::Decode, 3, 500.0));
    let asked = |list: &PublishedList| sides.engine.members_of(list.lent()).signal.get().is_some();
    let asked = anew.members.each_ref().map(asked);
    assert_eq!(
        asked,
        [false, true],
        "the lists asked their word at the call"
    );
}

/// A side whose profile reads no signal is lent no cells, while the other
/// side of the same pair reads its list's values: the prefill side alone
/// under the first document here, and the decode side alone under the
/// second.
#[test]
fn a_side_that_reads_no_signal_is_lent_no_cells() {
    for (prefill, decode, reads, base) in [
        (READS, SEES, Phase::Prefill, 10.0),
        (SEES, READS, Phase::Decode, 100.0),
    ] {
        let sides = sides(&pair(HELD, prefill, decode));
        let lists = Lists::of(2, 2);
        let loads = lists.stepped([10.0, 100.0]);
        let asks = lists.asks(7, Some(&loads));
        let decision = pair_decided(&sides.engine, &asks, Instant::now());
        assert!(decision.pick().is_some(), "{reads:?}");
        assert_eq!(read_of(&sides), stepped_side(reads, 2, base));
        let seen = mem::take(&mut *sides.seen.lock());
        let other = |&(side, cells): &(Phase, bool)| side != reads && !cells;
        assert!(!seen.is_empty() && seen.iter().all(other), "{seen:?}");
    }
}

/// Under no snapshot every worker of both sides reads no value, and the
/// pair is still decided. Cells a caller put on the request are read by
/// neither side: they are by the caller's own positions, here a value at
/// every place, and each side reads what the snapshot states.
#[test]
fn under_no_snapshot_both_sides_read_no_value_and_the_pair_is_decided() {
    let sides = sides(&pair(HELD, READS, READS));
    let lists = Lists::of(2, 2);
    let callers = SignalCell {
        value: 7.0,
        sampled_at_micros: 0,
    };
    let cells = SignalCells::new(vec![callers; 2], 1);
    let mut asks = lists.asks(7, None);
    asks.request.facts.signal_cells = Some(&cells);
    let decision = pair_decided(&sides.engine, &asks, Instant::now());
    assert!(decision.pick().is_some());
    let none = |side| [(side, 0, None), (side, 1, None)];
    let both = [none(Phase::Prefill), none(Phase::Decode)].concat();
    assert_eq!(read_of(&sides), both);

    asks.loads = Some(lists.stepped([10.0, 100.0]));
    let _ = pair_decided(&sides.engine, &asks, Instant::now());
    let mut both = stepped_side(Phase::Prefill, 2, 10.0);
    both.extend(stepped_side(Phase::Decode, 2, 100.0));
    assert_eq!(read_of(&sides), both);
}

/// A list's word is of the model and the phase it was first asked for, and
/// a decision of another phase asks the document. Under a document whose
/// decode side alone reads a signal, one list asked for both phases answers
/// each by that phase's profile, whichever phase is asked first.
#[test]
fn a_lists_word_is_of_the_phase_it_was_first_asked_for() {
    let sides = sides(&pair(HELD, SEES, READS));
    let engine = &sides.engine;
    let list = pool(2);
    for order in [
        [Phase::Prefill, Phase::Decode],
        [Phase::Decode, Phase::Prefill],
    ] {
        let published = PublishedList::new(Arc::clone(&list));
        let members = engine.members_of(published.lent());
        for phase in order.into_iter().chain(order) {
            let reads = members.reads_signal(engine.spec(), MODEL, phase);
            assert_eq!(reads, phase == Phase::Decode, "{phase:?} of {order:?}");
        }
    }
}

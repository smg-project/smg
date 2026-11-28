//! A place one pick decides but for its row-wise scorers: the pick first,
//! then each scorer asked of the picked row alone.

use super::*;
use crate::{document::spec::PRESETS, engine::framework::ScoreSpread};

/// What a scorer was handed at each call: how many rows, and the places
/// kept.
type Handed = Arc<Mutex<Vec<(usize, Vec<u32>)>>>;

/// What a scorer read at one call: the request's facts but for the two a
/// decision draws for itself, those two, and each kept row's record.
#[derive(Debug)]
struct Read {
    /// The facts, their seed and their instant left out.
    facts: String,
    /// The request's seed.
    seed: u64,
    /// The decision's instant.
    now_micros: u64,
    /// The record of each kept row, by its place.
    rows: Vec<(u32, String)>,
}

/// What a scorer read at each call.
type Reads = Arc<Mutex<Vec<Read>>>;

/// What a host's scorer says of itself.
#[derive(Clone, Copy, Debug)]
struct Says {
    /// The word `row_wise`.
    row_wise: bool,
    /// The word `refuses_by_reports`: a stage that says nothing keeps
    /// `true`.
    by_reports: bool,
    /// Whether the word `spread` says that the scorer orders nothing.
    alike: bool,
    /// What the scorer declares it reads.
    needs: Needs,
}

impl Says {
    /// The words of a scorer the engine may ask behind a pick.
    const BEHIND: Self = Self {
        row_wise: true,
        by_reports: false,
        alike: false,
        needs: Needs::NONE,
    };
}

/// A scorer of a host that keeps what it was handed. It scores a row at an
/// odd place over a row at an even one, so a picker that read a score would
/// not pick as one that reads none.
#[derive(Debug)]
struct SeesRows {
    handed: Handed,
    reads: Reads,
    says: Says,
}

impl Lifecycle for SeesRows {}

impl Scorer for SeesRows {
    fn name(&self) -> &'static str {
        "sees-rows"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        // A kept row's record stands at its own place.
        for &place in kept {
            assert_eq!(rows[place as usize].position as usize, place as usize);
        }
        self.handed.lock().push((rows.len(), kept.to_vec()));
        let shown = RequestFacts {
            seed: 0,
            now_micros: 0,
            ..*facts
        };
        let row = |&place: &u32| (place, format!("{:?}", rows[place as usize]));
        self.reads.lock().push(Read {
            facts: format!("{shown:?}"),
            seed: facts.seed,
            now_micros: facts.now_micros,
            rows: kept.iter().map(row).collect(),
        });
        for (score, &place) in out.iter_mut().zip(kept) {
            *score = match (self.says.alike, place % 2) {
                (true, _) => 0.5,
                (false, 0) => 0.25,
                (false, _) => 0.75,
            };
        }
        // A stage that orders nothing returns no band.
        if self.says.alike {
            TieBand(0.0)
        } else {
            TieBand::NORMALIZED
        }
    }

    fn row_wise(&self) -> bool {
        self.says.row_wise
    }

    fn refuses_by_reports(&self) -> bool {
        self.says.by_reports
    }

    fn needs(&self) -> Needs {
        self.says.needs
    }

    fn spread(&self) -> ScoreSpread {
        if self.says.alike {
            ScoreSpread::Alike
        } else {
            ScoreSpread::Any
        }
    }
}

/// An engine under the preset `name` as it ships but for the one scorer,
/// added behind the scorers of every profile, and for an `explain.sample`
/// that is not zero. With it, what that scorer was handed and what it
/// read. `None` where the document is refused.
fn reading_as(name: &str, says: Says, sample: u32) -> Option<(&'static InProcess, Handed, Reads)> {
    let handed = Handed::default();
    let reads = Reads::default();
    let (seen, read) = (Arc::clone(&handed), Arc::clone(&reads));
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("sees-rows", move || {
        Box::new(SeesRows {
            handed: Arc::clone(&seen),
            reads: Arc::clone(&read),
            says,
        })
    });
    added.expect("a free name");
    let engine = hosted_with(name, &stages, "sees-rows", sample)?;
    Some((engine, handed, reads))
}

/// An engine under the preset `name` as it ships but for the scorer the
/// `stages` hold as `scorer`, added behind the scorers of every profile, and
/// for an `explain.sample` that is not zero. `None` where the document is
/// refused.
fn hosted_with(
    name: &str,
    stages: &StageRegistry,
    scorer: &str,
    sample: u32,
) -> Option<&'static InProcess> {
    let document = PolicySpec::preset(name).expect("a built-in preset");
    let mut document = serde_json::to_value(document).expect("the document as JSON");
    let profiles = document["profiles"].as_object_mut();
    for profile in profiles.expect("the profiles").values_mut() {
        let score = &mut profile["score"];
        if let Some(score) = score.as_array_mut() {
            score.push(serde_json::json!(scorer));
        } else {
            *score = serde_json::json!([scorer]);
        }
    }
    if sample != 0 {
        document["explain"]["sample"] = serde_json::json!(sample);
    }
    let mut document: PolicySpec = serde_json::from_value(document).ok()?;
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, stages, &[]).ok()?;
    Some(hosted(InProcess::new(spec)))
}

/// [`reading_as`] for a case that asks what the scorer was handed alone.
fn scored_as(name: &str, says: Says, sample: u32) -> Option<(&'static InProcess, Handed)> {
    reading_as(name, says, sample).map(|(engine, handed, _)| (engine, handed))
}

/// [`scored_as`] under a document that keeps no sample, as a preset ships.
fn scored(name: &str, says: Says) -> (&'static InProcess, Handed) {
    scored_as(name, says, 0).expect("the stages hold the scorer")
}

/// The place in `list` of the worker a plain request is placed on.
fn placed_at(
    engine: &'static InProcess,
    list: &Arc<[Arc<dyn Worker>]>,
    members: &PublishedList,
) -> usize {
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let placement = engine.place(&request).expect("a placement");
    list.iter()
        .position(|worker| Arc::ptr_eq(worker, &placement.worker))
        .expect("a worker of the list")
}

/// What the scorer of `engine` was handed at each of three plain requests
/// over a list of `workers` workers.
fn handed_over(
    engine: &'static InProcess,
    handed: &Handed,
    workers: usize,
) -> [Vec<(usize, Vec<u32>)>; 3] {
    let list = pool(workers);
    let members = PublishedList::new(Arc::clone(&list));
    [(); 3].map(|()| {
        placed_at(engine, &list, &members);
        std::mem::take(&mut *handed.lock())
    })
}

/// Under a picker whose one pick decides the place, a row-wise scorer of a
/// host is asked once a decision, behind the pick, of the picked row alone:
/// it is handed the list's rows with that one place kept.
#[test]
fn a_row_wise_scorer_is_handed_the_picked_row_alone() {
    for name in ["passthrough", "round_robin"] {
        for workers in [2, 8, 128] {
            let (engine, handed) = scored(name, Says::BEHIND);
            let list = pool(workers);
            let members = PublishedList::new(Arc::clone(&list));
            for turn in 0..24 {
                let at = placed_at(engine, &list, &members);
                let calls = std::mem::take(&mut *handed.lock());
                let picked = u32::try_from(at).expect("a place");
                assert_eq!(
                    calls,
                    [(workers, vec![picked])],
                    "{name} over {workers} workers, turn {turn}"
                );
            }
        }
    }
}

/// `random` draws its place, so no case can foretell it: whatever row a
/// request is drawn to, the scorer is asked once for that request, of that
/// row alone.
#[test]
fn a_row_wise_scorer_under_random_is_handed_the_drawn_row_alone() {
    for workers in [2, 8, 128] {
        let (engine, handed) = scored("random", Says::BEHIND);
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        for turn in 0..24 {
            let at = placed_at(engine, &list, &members);
            let calls = std::mem::take(&mut *handed.lock());
            let drawn = u32::try_from(at).expect("a place");
            assert_eq!(
                calls,
                [(workers, vec![drawn])],
                "random over {workers} workers, turn {turn}"
            );
        }
    }
}

/// The presets, as they ship, under which a row-wise scorer a host adds is
/// asked behind the pick: the three whose picker picks alone. Under every
/// other the whole pass asks it.
#[test]
fn the_presets_one_pick_decides_ask_a_row_wise_scorer_behind_it() {
    let mut behind = Vec::new();
    for name in PRESETS {
        let Some((engine, handed)) = scored_as(name, Says::BEHIND, 0) else {
            continue;
        };
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let alone = (0..6).all(|_| {
            let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
            let Ok(placement) = engine.place(&request) else {
                return false;
            };
            let at = list
                .iter()
                .position(|worker| Arc::ptr_eq(worker, &placement.worker))
                .and_then(|at| u32::try_from(at).ok());
            let calls = std::mem::take(&mut *handed.lock());
            at.is_some_and(|at| calls == [(8, vec![at])])
        });
        if alone {
            behind.push(name);
        }
    }
    assert_eq!(behind, ["random", "round_robin", "passthrough"]);
}

/// Over a list of one worker the scorer is asked of it, once a decision,
/// as the whole pass asks.
#[test]
fn a_row_wise_scorer_is_asked_of_a_list_of_one_worker() {
    for name in ["passthrough", "round_robin"] {
        let (engine, handed) = scored(name, Says::BEHIND);
        for calls in handed_over(engine, &handed, 1) {
            assert_eq!(calls, [(1, vec![0])], "{name}");
        }
    }
}

/// A scorer that does not say the word keeps the whole pass: it is handed
/// every row of the list, once a decision.
#[test]
fn a_scorer_that_is_not_row_wise_keeps_the_whole_pass() {
    let says = Says {
        row_wise: false,
        ..Says::BEHIND
    };
    for name in ["passthrough", "round_robin"] {
        let (engine, handed) = scored(name, says);
        let every: Vec<u32> = (0..8).collect();
        for calls in handed_over(engine, &handed, 8) {
            assert_eq!(calls, [(8, every.clone())], "{name}");
        }
    }
}

/// A row-wise scorer that says nothing of its gateways' reports may refuse
/// by them, and keeps the whole pass: it is handed both rows of two.
#[test]
fn a_scorer_that_may_refuse_by_reports_keeps_the_whole_pass() {
    let says = Says {
        by_reports: true,
        ..Says::BEHIND
    };
    let (engine, handed) = scored("passthrough", says);
    for calls in handed_over(engine, &handed, 2) {
        assert_eq!(calls, [(2, vec![0, 1])]);
    }
}

/// A scorer that says how its values lie is a rank of the whole pass,
/// which asks a rank its own way, and keeps the whole pass: one that orders
/// nothing is handed both rows of two.
#[test]
fn a_scorer_that_says_how_its_values_lie_keeps_the_whole_pass() {
    let says = Says {
        alike: true,
        ..Says::BEHIND
    };
    let (engine, handed) = scored("passthrough", says);
    for calls in handed_over(engine, &handed, 2) {
        assert_eq!(calls, [(2, vec![0, 1])]);
    }
}

/// Under a document that keeps a sample of its decisions the whole pass
/// decides: the scorer is handed both rows of two.
#[test]
fn a_document_that_keeps_a_sample_keeps_the_whole_pass() {
    let (engine, handed) = scored_as("passthrough", Says::BEHIND, 1).expect("a document");
    for calls in handed_over(engine, &handed, 2) {
        assert_eq!(calls, [(2, vec![0, 1])]);
    }
}

/// Under a document that keeps a sample the whole pass alone turns the
/// list's rotation: no pick is made ahead of it and thrown away. So
/// `round_robin` places as many requests as a list of two or of eight
/// workers holds on every worker of it, each where the same document
/// places it under a scorer that is not row-wise.
#[test]
fn a_document_that_keeps_a_sample_takes_one_turn_a_request() {
    let whole_pass = Says {
        row_wise: false,
        ..Says::BEHIND
    };
    for workers in [2, 8] {
        let places = |says: Says| -> Vec<usize> {
            let (engine, _) = scored_as("round_robin", says, 1).expect("a document");
            let list = pool(workers);
            let members = PublishedList::new(Arc::clone(&list));
            (0..workers)
                .map(|_| placed_at(engine, &list, &members))
                .collect()
        };
        let behind = places(Says::BEHIND);
        let placed: BTreeSet<_> = behind.iter().collect();
        assert_eq!(placed.len(), workers, "a worker a turn: {behind:?}");
        assert_eq!(behind, places(whole_pass), "the whole pass's places");
    }
}

/// A scorer that declares a need beyond the request's size and a worker's
/// load report keeps the whole pass, which feeds the need: a signal's cells
/// and the affinity column are lent a request inside that pass. It is
/// handed both rows of two. A scorer that declares those two alone is asked
/// behind the pick, of the picked row.
#[test]
fn a_scorer_that_declares_a_need_the_whole_pass_feeds_keeps_the_whole_pass() {
    let fed_by_the_whole_pass = [
        Needs::NONE.with_text(),
        Needs::NONE.with_token_hashes(),
        Needs::NONE.with_text_hashes(),
        Needs::NONE.with_events(),
        Needs::NONE.with_sent_prefixes(),
        Needs::NONE.with_ring(),
        Needs::NONE.with_headers(),
        Needs::NONE.with_media(),
        Needs::NONE.with_active_sessions(),
        Needs::NONE.with_session_in_flight(),
        Needs::NONE.with_affinity(),
        Needs::NONE.with_signal_cells(),
    ];
    // The needs a document of this preset is refused for, so that no
    // request is decided under one: a stage that declares a signal's cells
    // resolves the signal's column, and the signals' own cases hold it.
    let mut refused = Vec::new();
    for needs in fed_by_the_whole_pass {
        let says = Says {
            needs,
            ..Says::BEHIND
        };
        let Some((engine, handed)) = scored_as("passthrough", says, 0) else {
            refused.push(needs);
            continue;
        };
        for calls in handed_over(engine, &handed, 2) {
            assert_eq!(calls, [(2, vec![0, 1])], "{needs:?}");
        }
    }
    assert_eq!(
        refused,
        [Needs::NONE.with_signal_cells()],
        "the one need refused"
    );
    let says = Says {
        needs: Needs::NONE.with_loads().with_prompt_size(),
        ..Says::BEHIND
    };
    let (engine, handed) = scored("passthrough", says);
    for calls in handed_over(engine, &handed, 2) {
        assert_eq!(calls, [(2, vec![0])], "the picked row alone");
    }
}

/// The words of its build a scorer was asked, each by its name.
type Asked = Arc<Mutex<Vec<&'static str>>>;

/// A scorer of a host the engine may ask behind a pick, which keeps each
/// word of its build it was asked: what it reads, how its values lie,
/// whether it places a request by itself and whether it refuses by a
/// report.
#[derive(Debug)]
struct BuildWords {
    asked: Asked,
}

impl Lifecycle for BuildWords {}

impl Scorer for BuildWords {
    fn name(&self) -> &'static str {
        "build-words"
    }

    fn score(
        &self,
        _facts: &RequestFacts<'_>,
        _rows: &[Candidate],
        _kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        out.fill(0.5);
        TieBand::NORMALIZED
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn refuses_by_reports(&self) -> bool {
        self.asked.lock().push("refuses_by_reports");
        false
    }

    fn needs(&self) -> Needs {
        self.asked.lock().push("needs");
        Needs::NONE
    }

    fn spread(&self) -> ScoreSpread {
        self.asked.lock().push("spread");
        ScoreSpread::Any
    }

    fn owns(&self) -> bool {
        self.asked.lock().push("owns");
        false
    }
}

/// The words a scorer says for its build ([`Scorer::needs`],
/// [`Scorer::spread`], [`Scorer::owns`], [`Scorer::refuses_by_reports`])
/// are asked while the engine is built and by no decision of the short
/// road: a host's scorer may answer them from what it keeps for its build
/// alone. So at a list's first request, where the engine finds how the
/// place is decided, at a later one, and at a request of another model
/// over the list, which the document answers at its call. (The whole pass
/// of a debug build asks [`Scorer::spread`] again in its assertions; the
/// requests here are decided by one pick.)
#[test]
fn no_decision_asks_a_scorer_a_word_of_its_build() {
    for name in ["passthrough", "round_robin"] {
        let asked = Asked::default();
        let words = Arc::clone(&asked);
        let mut stages = StageRegistry::builtin();
        let added = stages.scorer("build-words", move || {
            Box::new(BuildWords {
                asked: Arc::clone(&words),
            })
        });
        added.expect("a free name");
        let engine = hosted_with(name, &stages, "build-words", 0).expect("a document");
        let build = std::mem::take(&mut *asked.lock());
        assert!(
            build.contains(&"needs"),
            "{name}: the build asked {build:?}"
        );
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        for model in [MODEL, MODEL, "model-b", MODEL] {
            let request = PlaceRequest::of(model, WorkerLists::Single(members.lent()));
            drop(engine.place(&request).expect("a placement"));
            let late = std::mem::take(&mut *asked.lock());
            assert!(
                late.is_empty(),
                "{name}: a request of {model} asked {late:?}"
            );
        }
    }
}

/// A scorer asked behind the pick that declares a worker's load report
/// reads the picked row's record as the whole pass writes it, with the
/// report of the snapshot the engine's task keeps: the record differs from
/// the one read where no report was fed.
#[tokio::test(start_paused = true)]
async fn a_scorer_asked_behind_the_pick_reads_the_picked_rows_load_report() {
    let behind = Says {
        needs: Needs::NONE.with_loads().with_prompt_size(),
        ..Says::BEHIND
    };
    let whole_pass = Says {
        row_wise: false,
        ..behind
    };
    let reports: HashMap<_, _> = (0..4)
        .map(|at| {
            let url = format!("http://worker-{at}:8000");
            (url, one_rank_load(1_000 * (at + 1), 0.1, 1_000.0))
        })
        .collect();
    let mut read = Vec::new();
    for (says, fed) in [(behind, true), (whole_pass, true), (behind, false)] {
        let (engine, _, reads) = reading_as("passthrough", says, 0).expect("a document");
        let mut channel = None;
        let feed = TestFeed::attached_to(|loads| channel = loads);
        if fed {
            feed.publish(&reports);
        }
        engine.set_loads(channel.expect("the feed's channel"));
        #[expect(
            clippy::disallowed_methods,
            reason = "the case holds the task's handle and ends the task"
        )]
        let task = tokio::spawn(engine.run());
        // The task's first step takes the channel and keeps its snapshot.
        sleep(Duration::from_millis(1)).await;
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        placed_at(engine, &list, &members);
        task.abort();
        read.push(reads.lock().pop().expect("the scorer was asked"));
    }
    let [behind, whole, unfed]: [Read; 3] = read.try_into().expect("three reads");
    // A record with its report's instant left out: each engine's task took
    // its snapshot at an instant of its own.
    let undated = |row: &str| {
        let (ahead, behind) = row
            .split_once("load_sampled_at_micros: ")
            .expect("a record states its report's instant");
        let (_, behind) = behind.split_once(',').expect("a field behind it");
        format!("{ahead}{behind}")
    };
    assert_eq!(behind.rows.len(), 1, "the picked row alone");
    assert_eq!(whole.rows.len(), 4, "every row");
    let (picked, row) = &behind.rows[0];
    let at_pick = whole.rows.iter().find(|(place, _)| place == picked);
    let at_pick = at_pick.map(|(_, row)| undated(row));
    assert_eq!(at_pick, Some(undated(row)), "the row");
    assert_eq!(behind.facts, whole.facts, "the facts");
    let (at, bare) = &unfed.rows[0];
    assert_eq!(at, picked, "one place under both");
    assert_ne!(
        undated(bare),
        undated(row),
        "the report stands in the record"
    );
}

/// The pick is the picker's own: over a list of any size, an engine whose
/// row-wise scorer is asked behind the pick places every request where the
/// preset alone, which holds no scorer, places it, a worker that leaves on
/// the way included. The scorer scores the rows apart, and no pick reads it.
#[test]
fn a_scorer_asked_behind_the_pick_changes_no_pick() {
    for name in ["passthrough", "round_robin"] {
        for workers in [2, 4, 8, 16, 32, 64, 128] {
            let places = |engine: &'static InProcess| {
                let list = pool(workers);
                let members = PublishedList::new(Arc::clone(&list));
                let mut places = Vec::new();
                for turn in 0..3 * workers + 5 {
                    if turn == workers + 1 {
                        list[workers / 2].set_status(WorkerStatus::NotReady);
                    }
                    places.push(placed_at(engine, &list, &members));
                }
                places
            };
            let (asked, _) = scored(name, Says::BEHIND);
            let alone = hosted(preset(name));
            assert_eq!(
                places(asked),
                places(alone),
                "{name} over {workers} workers"
            );
        }
    }
}

/// What a scorer asked behind the pick reads is what the whole pass hands
/// one: every fact of the request as its host stated it, an instant read
/// inside the decision, a seed of the decision's own, and the picked row's
/// record. The whole pass is the same preset under a scorer that is not
/// row-wise, which is handed every row.
#[test]
fn a_scorer_asked_behind_the_pick_reads_what_the_whole_pass_hands_it() {
    const TOKENS: &[u32] = &[11, 12, 13, 14, 15];
    let whole_pass = Says {
        row_wise: false,
        ..Says::BEHIND
    };
    // What the scorer of `says` read at each of two decisions of one
    // request over eight workers, each with a count in flight of its own.
    let read_under = |name: &str, says: Says| -> [Read; 2] {
        let (engine, _, reads) = reading_as(name, says, 0).expect("a document");
        let list = pool(8);
        for (at, worker) in list.iter().enumerate() {
            (0..=at).for_each(|_| worker.increment_load());
        }
        let members = PublishedList::new(Arc::clone(&list));
        let mut request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
        request.tenant = Some("tenant-a");
        request.priority = 3;
        request.streaming = true;
        request.max_output_tokens = 256;
        request.budget_micros = 750_000;
        request.prompt.tokens = Some(TOKENS);
        [(); 2].map(|()| {
            let before = engine.clock().micros(Instant::now());
            drop(engine.place(&request).expect("a placement"));
            let after = engine.clock().micros(Instant::now());
            let mut calls = std::mem::take(&mut *reads.lock());
            let read = calls.pop().expect("the scorer was asked");
            assert!(calls.is_empty(), "{name}: once a decision");
            assert!(
                (before..=after).contains(&read.now_micros),
                "{name}: dated inside the decision"
            );
            read
        })
    };
    for name in ["random", "round_robin", "passthrough"] {
        let [behind, again] = read_under(name, Says::BEHIND);
        let [whole, _] = read_under(name, whole_pass);
        // The picked row alone behind the pick, every row in the whole pass.
        assert_eq!(behind.rows.len(), 1, "{name}: the picked row alone");
        assert_eq!(whole.rows.len(), 8, "{name}: every row");
        let (picked, row) = &behind.rows[0];
        let at_pick = whole.rows.iter().find(|(place, _)| place == picked);
        assert_eq!(at_pick.map(|(_, row)| row), Some(row), "{name}: the row");
        assert_eq!(behind.facts, whole.facts, "{name}: the facts");
        assert_eq!(again.facts, whole.facts, "{name}: the facts again");
        assert_ne!(behind.seed, again.seed, "{name}: a seed a decision");
        // What the case states of its request stands in the facts read.
        for stated in ["priority: 3", "streaming: true", "max_output_tokens: 256"] {
            assert!(behind.facts.contains(stated), "{name}: {stated}");
        }
    }
}

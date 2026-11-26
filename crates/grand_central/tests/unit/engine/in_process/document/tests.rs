//! The cases of a new document put in force on a running engine
//! ([`InProcess::apply`]): each asks the engine's own entries, over a
//! list of worker objects and the worker crate's registry, and none polls
//! the engine's task.

use std::{
    cell::Cell,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    thread,
    time::Instant,
};

use parking_lot::Mutex;

use super::{InForce, Loaded, SpecHandle};
use crate::{
    document::spec::{AppliedSpec, BindingSpec, Match, PolicySpec},
    engine::{
        decision::{Decision, Target, Targets},
        decision_engine::{DecisionEngine, DecisionRefused, Request, RequestMaker, WorkerLists},
        framework::{EncodeItems, EncodePlaced, Picker, TieBand},
        in_process::{
            membership::tests::{handed, id, registry, remove},
            single::tests::{
                decided, hearing_spec_at, hearing_stages, pool, request, Asks, Heard, MODEL,
            },
            task::tests::stock,
            InProcess,
        },
        list_cell::PublishedList,
        prompt_hashes::HashSeeds,
        shadow::{ShadowCounts, Shadowed},
        view::{Candidate, RequestFacts, WorkerId},
    },
    policies::random::Random,
    stages::{registry::StageRegistry, Lifecycle},
    store::{GatewayId, StaleGeneration},
    tracking::{lease::Membership, DecisionGuard, Placement, PlacementSide},
};

impl InProcess {
    /// The engine with the document in force, for a case that asks a
    /// function of a decision by itself. The document it shows is never
    /// freed.
    pub(crate) fn loaded(&self) -> Loaded<'_, 'static> {
        self.under(Box::leak(Box::new(self.document.load_full())))
    }

    /// The document in force, for a case that reads it past one statement.
    /// It is never freed.
    pub(crate) fn standing_spec(&self) -> &'static AppliedSpec {
        &self.loaded().document.spec
    }

    /// [`Loaded::decide_without_lease`] under the document in force, for a
    /// case of an engine by itself: no world hands it a document.
    pub(crate) fn decide_without_lease(
        &self,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<(Decision, Option<Shadowed>), DecisionRefused> {
        let document = self.in_force();
        self.under(&document).decide_without_lease(maker, now)
    }

    /// [`Loaded::decide_decode_without_lease`] under the document in
    /// force, for a case of an engine by itself. The case states no date
    /// of the pair's lease.
    pub(crate) fn decide_decode_without_lease(
        &self,
        maker: &impl RequestMaker,
        held: &Target,
        now: Instant,
    ) -> Result<Decision, DecisionRefused> {
        let document = self.in_force();
        self.under(&document)
            .decide_decode_without_lease(maker, held, 0, now)
    }

    /// [`Loaded::decide_encode_without_lease`] under the document in
    /// force, for a case of an engine by itself.
    pub(crate) fn decide_encode_without_lease(
        &self,
        maker: &impl RequestMaker,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<Decision, DecisionRefused> {
        let document = self.in_force();
        self.under(&document)
            .decide_encode_without_lease(maker, items, placed, now)
    }

    /// [`Loaded::count_shadow`] under the document in force, for a case of
    /// an engine by itself.
    pub(crate) fn count_shadow(&self, said: Option<Shadowed>) {
        let document = self.in_force();
        self.under(&document).count_shadow(said);
    }
}

impl InForce {
    /// The tables of the shadow decisions counted so far, for a case that
    /// reads whether they were allocated or writes a count of its own.
    pub(in crate::engine::in_process) fn shadow_tables(&self) -> &OnceLock<Box<ShadowCounts>> {
        &self.shadow_tables
    }
}

/// A picker of a host's own build that picks among the workers it heard
/// join, and names nobody while it heard none.
#[derive(Debug, Default)]
struct AmongJoined(Mutex<Vec<WorkerId>>);

impl Lifecycle for AmongJoined {
    fn on_membership(&self, change: &Membership<'_>) {
        if let Membership::Added(id) = *change {
            self.0.lock().push(id);
        }
    }
}

impl Picker for AmongJoined {
    fn name(&self) -> &'static str {
        "among-joined"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        let joined = self.0.lock();
        let heard = |at: &u32| joined.contains(&candidates[*at as usize].id);
        let among: Vec<u32> = kept.iter().copied().filter(heard).collect();
        Random.pick(facts, candidates, &among, scores, band, out);
    }
}

/// The `random` preset under the generation `number`, with [`AmongJoined`]
/// in its pick's place.
fn among_joined(number: u64) -> AppliedSpec {
    let mut stages = StageRegistry::builtin();
    let make = || -> Box<dyn Picker> { Box::new(AmongJoined::default()) };
    stages.picker("among-joined", make).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = number;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "among-joined".into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    spec.expect("the stages hold the picker")
}

/// A new document's stages hear the workers that stand before its first
/// decision: the call that puts it in force tells them on its caller's
/// thread, with no step of the engine's task between. The first document's
/// stage heard nobody, since no task ran, and names no worker. The
/// decision made right behind the call names one.
#[test]
fn a_new_documents_stages_hear_the_standing_workers_before_its_first_decision() {
    let engine = InProcess::new(among_joined(1));
    engine.set_registry(handed(&registry(3)));
    let members = PublishedList::new(pool(3));
    let asks = Asks::of(request(), Some(&members));
    let untold = decided(&engine, &asks).decision;
    assert!(untold.pick().is_none(), "no task ran: nobody was heard");

    let applied = engine.apply(among_joined(2));
    applied.expect("a number above the standing one");
    let told = decided(&engine, &asks).decision;
    let pick = told.pick().expect("a worker the stage heard join");
    assert_eq!(pick.trace.generation, 2, "under the new document");
}

/// The call tells the new document's stages what the task's whole read of
/// the registry tells: each worker that stands as joined, then the ids
/// that stand. The document it replaces hears nothing of it.
#[test]
fn a_new_document_is_told_each_join_and_then_the_ids_that_stand() {
    let (first, before) = hearing_spec_at(1);
    let engine = InProcess::new(first);
    engine.set_registry(handed(&registry(2)));
    let (next, heard) = hearing_spec_at(2);
    let applied = engine.apply(next);
    applied.expect("a number above the standing one");

    let heard = heard.lock();
    let words: Vec<&str> = heard.iter().map(|word| word.0).collect();
    assert_eq!(words, ["joined", "joined", "standing", "standing"]);
    for told in [&heard[..2], &heard[2..]] {
        let mut ids: Vec<WorkerId> = told.iter().map(|word| word.1).collect();
        ids.sort_by_key(|id| id.0);
        let mut standing = vec![id(0), id(1)];
        standing.sort_by_key(|id| id.0);
        assert_eq!(ids, standing);
    }
    assert!(before.lock().is_empty(), "the document replaced hears none");
}

/// The sends the document in force counts for the worker `to` names.
fn sends_to(engine: &InProcess, to: &Target) -> u32 {
    let row = Candidate {
        id: to.id,
        dp_rank: to.dp_rank,
        ..Candidate::default()
    };
    let credit = &engine.standing_spec().engine().sent_credits[0];
    credit.read(&RequestFacts::default(), &row).requests
}

/// The stock document `name` under the generation `number`, its profile's
/// filters in the other order: they keep the same rows, and the profile has
/// another text, so a running engine builds its stages new.
fn stock_anew(name: &str, number: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = number;
    let profile = document.profiles.get_mut(name);
    profile.expect("the preset's profile").filter.reverse();
    AppliedSpec::validate(&document).expect("the document validates")
}

/// A profile's count of sends taken over at the call stands behind the
/// joins the call tells: a join told of a worker that stands drops nothing
/// the standing document counted for it. The later document's profile has
/// another text, so its count is built new and takes the standing one over.
/// It stands behind the whole read of the registry the call owes the
/// engine's task too: that read tells no worker the task holds as joined
/// again. The case makes the task's two steps itself, the first ahead of
/// the request as a program's task runs.
#[test]
fn a_count_of_sends_taken_over_stands_behind_the_joins_told_at_an_apply() {
    let engine = InProcess::new(stock("least_load", 1));
    engine.set_registry(handed(&registry(2)));
    let mut follow = None;
    engine.follow_registry(&mut follow);
    let members = PublishedList::new(pool(2));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let mut first =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    let sent_to = first.take(0).expect("the first target");
    first.dispatched(now);
    assert!([id(0), id(1)].contains(&sent_to.id), "a worker that stands");
    assert_eq!(sends_to(&engine, &sent_to), 1, "the send is counted");

    let applied = engine.apply(stock_anew("least_load", 2));
    applied.expect("a number above the standing one");
    assert_eq!(engine.spec().generation(), 2);
    assert_eq!(sends_to(&engine, &sent_to), 1, "behind the told joins");
    engine.follow_registry(&mut follow);
    assert_eq!(sends_to(&engine, &sent_to), 1, "behind the read owed");
}

/// A count of sends takes over ahead of the list the call tells: the list
/// drops from it what the standing count holds under an id that stands no
/// more. A worker is removed from the registry between its decision and
/// its dispatch's word, and the dispatch enters its rank in the standing
/// count, as a dispatch told behind a leave does. A count built new does
/// not hold that rank once the call returns. A count kept hears no list at
/// the call: it lets the rank go at the task's next upkeep, where the whole
/// read the call owes tells every stage the ids that stand.
#[test]
fn a_send_to_a_worker_that_stands_no_more_is_not_taken_over_at_an_apply() {
    for (later, at_the_call) in [
        (stock_anew("least_load", 2), 0),
        (stock("least_load", 2), 1),
    ] {
        let engine = InProcess::new(stock("least_load", 1));
        let live = registry(2);
        engine.set_registry(handed(&live));
        let members = PublishedList::new(pool(2));
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let decided = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now);
        let mut first = decided.expect("a decision");
        let sent_to = first.take(0).expect("the first target");
        remove(&live, usize::from(sent_to.id == id(1)));
        first.dispatched(now);
        assert_eq!(sends_to(&engine, &sent_to), 1, "the rank is entered");

        engine
            .apply(later)
            .expect("a number above the standing one");
        assert_eq!(sends_to(&engine, &sent_to), at_the_call, "at the call");
        let mut follow = None;
        engine.follow_registry(&mut follow);
        assert_eq!(sends_to(&engine, &sent_to), 0, "behind the read owed");
    }
}

/// The stock document `least_load` under the generation `number`, with a
/// second profile `late` of the same stages bound to a model no case asks
/// for. `limits` are the milliseconds a send counts for under the stock
/// profile and under `late`: two limits, so each profile reads a count of
/// its own.
fn two_limits(number: u64, limits: [u32; 2]) -> AppliedSpec {
    let mut document = PolicySpec::preset("least_load").expect("a built-in preset");
    document.generation = number;
    let mut late = document.profiles["least_load"].clone();
    late.sent_ttl_ms = Some(limits[1]);
    document.profiles.insert("late".into(), late);
    let stock = document.profiles.get_mut("least_load");
    stock.expect("the preset's profile").sent_ttl_ms = Some(limits[0]);
    let elsewhere = BindingSpec {
        matcher: Match {
            model: "elsewhere".into(),
            ..Match::default()
        },
        profile: Some("late".into()),
        ..BindingSpec::default()
    };
    document.bindings.insert(0, elsewhere);
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The sends the document in force counts for the worker `to` names under
/// the time limit `limit_ms`, as a decision at `now_micros` reads them.
fn sends_under(engine: &InProcess, limit_ms: u32, to: &Target, now_micros: u64) -> u32 {
    let row = Candidate {
        id: to.id,
        dp_rank: to.dp_rank,
        ..Candidate::default()
    };
    let facts = RequestFacts {
        now_micros,
        ..RequestFacts::default()
    };
    let mut counts = engine.standing_spec().engine().sent_credits.iter();
    let count = counts.find(|count| count.settings.ttl_ms == Some(limit_ms));
    let count = count.expect("a count under the limit");
    count.read(&facts, &row).requests
}

/// A count of sends built new at the call holds the sends its own time
/// limit still counts, whichever standing count let them go. The first
/// document's two profiles count a send for one millisecond and for nine.
/// Two milliseconds behind a send, the count under one has let it go and
/// the count under nine holds it. The second document writes nine for the
/// one and a longer limit for the nine: the send is inside both, and both
/// of its counts hold it.
#[test]
fn a_count_under_another_limit_keeps_the_sends_that_limit_counts_at_an_apply() {
    let engine = InProcess::new(two_limits(1, [1, 9]));
    engine.set_registry(handed(&registry(2)));
    let members = PublishedList::new(pool(2));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let decided = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now);
    let mut first = decided.expect("a decision");
    let sent_to = first.take(0).expect("the first target");
    first.dispatched(now);
    let later = engine.clock().micros(now) + 2_000;
    assert_eq!(sends_under(&engine, 1, &sent_to, later), 0, "past one");
    assert_eq!(sends_under(&engine, 9, &sent_to, later), 1, "inside nine");

    let applied = engine.apply(two_limits(2, [9, 43]));
    applied.expect("a number above the standing one");
    let held = [9, 43].map(|limit| sends_under(&engine, limit, &sent_to, later));
    assert_eq!(held, [1, 1], "under the limit that stood and a longer one");
}

/// A document whose number is not above the standing one is refused, and
/// the standing document decides on.
#[test]
fn a_document_whose_number_is_not_above_the_standing_one_is_refused() {
    let engine = InProcess::new(stock("random", 2));
    for offered in [1, 2] {
        let refused = StaleGeneration {
            standing: 2,
            offered,
        };
        assert_eq!(engine.apply(stock("least_load", offered)), Err(refused));
    }
    let standing = engine.spec();
    assert_eq!(standing.generation(), 2);
    assert!(!standing.needs().loads, "the document that stood");
}

/// A handle keeps showing the document it was asked under, and a copy of
/// it does: the engine decides under the next one meanwhile.
#[test]
fn a_handle_shows_the_document_it_was_asked_under() {
    let engine = InProcess::new(stock("random", 1));
    let first = engine.spec();
    let copy = first.clone();
    let applied = engine.apply(stock("least_load", 2));
    applied.expect("a number above the standing one");
    assert_eq!((first.generation(), copy.generation()), (1, 1));
    assert_eq!(engine.spec().generation(), 2);
}

/// A word of a lease issued under a document that was replaced since is
/// told to nobody: the stages that issued it decide no more, and the
/// stages in force never heard of the lease.
#[test]
fn a_word_of_a_lease_of_a_replaced_document_is_told_to_nobody() {
    let (first, heard) = hearing_spec_at(1);
    let engine = InProcess::new(first);
    let members = PublishedList::new(pool(4));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let mut held =
        DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
    assert!(held.take(0).is_some(), "the first target");
    let issued: Vec<&str> = heard.lock().iter().map(|word| word.0).collect();
    assert_eq!(issued, ["issued"]);

    let (next, later) = hearing_spec_at(2);
    let applied = engine.apply(next);
    applied.expect("a number above the standing one");
    held.dispatched(now);
    drop(held);
    assert_eq!(heard.lock().len(), 1, "its own stages hear no more");
    assert!(later.lock().is_empty(), "the stages in force hear none");
}

/// A maker that puts a new document in force on its engine while it makes
/// its request, as a program's apply that lands inside a decision does.
struct AppliesInside<'a> {
    asks: Asks<'a>,
    engine: &'a InProcess,
    next: Cell<Option<AppliedSpec>>,
}

impl RequestMaker for AppliesInside<'_> {
    fn model(&self) -> &str {
        self.asks.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        if let Some(next) = self.next.take() {
            let applied = self.engine.apply(next);
            applied.expect("a number above the standing one");
        }
        self.asks.make(spec)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.asks.workers()
    }
}

/// A decision ends under the document it began under. A document put in
/// force while the decision makes its request changes nothing of it: the
/// trace and the lease name the first document, and that document's stage
/// hears the issue. The next decision runs under the new one.
#[test]
fn a_document_put_in_force_inside_a_decision_decides_the_next_one() {
    let (first, heard) = hearing_spec_at(1);
    let (next, later) = hearing_spec_at(2);
    let engine = InProcess::new(first);
    let members = PublishedList::new(pool(4));
    let maker = AppliesInside {
        asks: Asks::of(request(), Some(&members)),
        engine: &engine,
        next: Cell::new(Some(next)),
    };
    let answer = engine.decide_under(GatewayId::LOCAL, &maker, Instant::now());
    let (began, _) = answer.expect("an answer").whole().expect("no place");
    let lease = began.lease.expect("a lease").lease;
    let stated = (began.decision.trace().generation, lease.generation);
    assert_eq!(stated, (1, 1), "the document the decision began under");
    assert_eq!(heard.lock().len(), 1, "its stage heard the issue");
    assert!(later.lock().is_empty(), "the new document's heard none");

    assert_eq!(engine.spec().generation(), 2);
    let after = decided(&engine, &maker.asks).decision;
    assert_eq!(after.trace().generation, 2, "the next decision");
}

/// The record of sent prompts is the engine's and stands across a new
/// document: a prompt sent under one document goes back to its worker
/// under the next, at every seed. The next document's stages are built
/// new: what they read of a prompt's worker is the engine's record.
#[test]
fn a_prompt_goes_back_to_its_worker_under_the_next_document() {
    let engine = InProcess::new(stock("cache_aware", 1));
    let members = PublishedList::new(pool(8));
    let sent = |tokens: &[u32], seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let decision = decided(&engine, &Asks::of(asked, Some(&members))).decision;
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    };
    let prompt = |start: u32| (start..start + 1_024).collect::<Vec<u32>>();
    let prompts: Vec<Vec<u32>> = (1..=16).map(|at| prompt(at * 100_000)).collect();
    let seeded = (0_u64..).zip(&prompts);
    let picks: Vec<u32> = seeded.map(|(seed, tokens)| sent(tokens, seed)).collect();
    assert!(picks.iter().any(|&position| position != picks[0]));

    let applied = engine.apply(stock_anew("cache_aware", 2));
    applied.expect("a number above the standing one");
    for seed in 100..104 {
        for (tokens, &position) in prompts.iter().zip(&picks) {
            assert_eq!(sent(tokens, seed), position, "seed {seed}");
        }
    }
}

/// The `random` preset under the generation `number`, picked by the `hears`
/// of `stages` ([`hearing_stages`]), after `edit`.
fn hearing_over(
    stages: &StageRegistry,
    number: u64,
    edit: impl FnOnce(&mut PolicySpec),
) -> AppliedSpec {
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = number;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "hears".into();
    edit(&mut document);
    let spec = AppliedSpec::validate_in(&document, stages, &[]);
    spec.expect("the stages hold the picker")
}

/// The words the picker a table built at the place `at` heard, in order.
fn words(built: &Mutex<Vec<Heard>>, at: usize) -> Vec<&'static str> {
    let heard = Arc::clone(&built.lock()[at]);
    let words = heard.lock().iter().map(|word| word.0).collect();
    words
}

/// A later document keeps the stages of a profile whose name and text did
/// not change: the picker that decided under the first document decides
/// and hears under the next. The call tells it no worker again, and the
/// whole read of the registry the call owes names it the ids that stand
/// and no join: it heard each worker as it came. The picker the later
/// document's validation built is let go untold.
#[test]
fn a_later_document_keeps_the_stages_of_a_profile_whose_text_did_not_change() {
    let (stages, built) = hearing_stages();
    let engine = InProcess::new(hearing_over(&stages, 1, |_| {}));
    engine.set_registry(handed(&registry(2)));
    let mut follow = None;
    engine.follow_registry(&mut follow);
    let joins = ["joined", "joined", "standing", "standing"];
    assert_eq!(words(&built, 0), joins, "the first read of the registry");

    let applied = engine.apply(hearing_over(&stages, 2, |_| {}));
    applied.expect("a number above the standing one");
    assert_eq!(words(&built, 0), joins, "the call tells a kept stage none");
    engine.follow_registry(&mut follow);
    let told = words(&built, 0);
    assert_eq!(told[4..], ["standing", "standing"], "and no join again");
    assert!(words(&built, 1).is_empty(), "the one built for the call");

    let members = PublishedList::new(pool(2));
    let after = decided(&engine, &Asks::of(request(), Some(&members))).decision;
    assert_eq!(after.trace().generation, 2, "under the later document");
    assert_eq!(
        words(&built, 0).last(),
        Some(&"issued"),
        "the kept one hears"
    );
    assert!(words(&built, 1).is_empty());
}

/// A stage a later document builds new is told the workers that stand by
/// the call, beside a stage the document kept, which is told none. The
/// document changes the text of one of its two profiles.
#[test]
fn a_stage_built_new_is_told_the_workers_that_stand_beside_a_stage_kept() {
    let (stages, built) = hearing_stages();
    let engine = InProcess::new(hearing_over(&stages, 1, second_profile));
    engine.set_registry(handed(&registry(2)));
    assert!(words(&built, 0).is_empty() && words(&built, 1).is_empty());

    let applied = engine.apply(hearing_over(&stages, 2, second_anew));
    applied.expect("a number above the standing one");
    assert!(words(&built, 0).is_empty(), "the profile kept: told none");
    let joins = ["joined", "joined", "standing", "standing"];
    assert_eq!(words(&built, 3), joins, "the profile built new");
    assert!(
        words(&built, 1).is_empty(),
        "the one it replaces hears none"
    );
    assert!(words(&built, 2).is_empty(), "the one let go for the kept");
}

/// [`hearing_over`]'s edit that adds the profile `second`, picked by
/// `hears` too and bound to a model no case asks for.
fn second_profile(document: &mut PolicySpec) {
    let profile = document.profiles["random"].clone();
    document.profiles.insert("second".into(), profile);
    let elsewhere = BindingSpec {
        matcher: Match {
            model: "elsewhere".into(),
            ..Match::default()
        },
        profile: Some("second".into()),
        ..BindingSpec::default()
    };
    document.bindings.insert(0, elsewhere);
}

/// [`second_profile`] with the filters of `second` in the other order: the
/// profile has another text, and `random` the same.
fn second_anew(document: &mut PolicySpec) {
    second_profile(document);
    let second = document.profiles.get_mut("second");
    second.expect("the profile added").filter.reverse();
}

/// A word of a request whose lease an earlier document issued reaches the
/// stages that stood under that document where the document in force kept
/// them, and no stage built since: such a stage never heard the lease's
/// issue. A request of the first document is told to the profile kept
/// since then. A request of the second is told to the profile built under
/// the second too. The third document keeps both.
#[test]
fn an_older_requests_words_reach_the_stages_that_stood_under_its_document() {
    let (stages, built) = hearing_stages();
    let engine = InProcess::new(hearing_over(&stages, 1, second_profile));
    let members = PublishedList::new(pool(4));
    let asks = Asks::of(request(), Some(&members));
    let now = Instant::now();
    let held = || {
        let decided = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now);
        let mut held = decided.expect("a decision");
        assert!(held.take(0).is_some(), "the first target");
        held
    };
    let mut first = held();
    let applied = engine.apply(hearing_over(&stages, 2, second_anew));
    applied.expect("a number above the standing one");
    let mut second = held();
    let applied = engine.apply(hearing_over(&stages, 3, second_anew));
    applied.expect("a number above the standing one");

    first.dispatched(now);
    second.dispatched(now);
    // The pickers by the place the table built them at: `random` of the
    // first document, kept by both later ones, its `second`, replaced by
    // the second document's, and that one, kept by the third.
    let kept_from_the_first = ["issued", "issued", "dispatched", "dispatched"];
    assert_eq!(words(&built, 0), kept_from_the_first);
    assert_eq!(words(&built, 1), ["issued"], "replaced: it hears no more");
    assert_eq!(
        words(&built, 3),
        ["issued", "dispatched"],
        "its own request"
    );
    for let_go in [2, 4, 5] {
        assert!(words(&built, let_go).is_empty(), "the picker at {let_go}");
    }
}

/// What a store reads for a lease on its file is the stages that stood
/// under the lease's generation ([`Loaded::hearing`]), and none built
/// since, though a later document kept that one too. The second document
/// builds the profile `second` anew and the third keeps both profiles. A
/// dispatch told for a lease of the first document reaches the stage kept
/// since then alone, and one for a lease of the second reaches both.
#[cfg(feature = "controller")]
#[test]
fn a_stage_built_since_a_leases_generation_and_kept_later_did_not_stand_at_it() {
    let (stages, built) = hearing_stages();
    let engine = InProcess::new(hearing_over(&stages, 1, second_profile));
    for number in [2, 3] {
        let applied = engine.apply(hearing_over(&stages, number, second_anew));
        applied.expect("a number above the standing one");
    }
    for generation in [1, 2] {
        let lease = crate::tracking::Lease {
            generation,
            ..Default::default()
        };
        let held = engine.spec();
        let stood = held.under(&engine).hearing(generation);
        let stood = stood.expect("a generation under the standing one");
        let side = crate::tracking::Side::Single;
        stood.on_dispatched(&lease, &Target::default(), side, 0);
    }
    // The pickers by the place the table built them at: `random` of the
    // first document, kept by both later ones, and the second document's
    // `second`, kept by the third.
    assert_eq!(words(&built, 0), ["dispatched", "dispatched"]);
    assert_eq!(words(&built, 3), ["dispatched"], "born under the second");
}

/// A send said behind a later document, for a decision the document before
/// made, is counted once by the count of sends in force, built new or kept:
/// a send is a fact of its worker, whatever document decided it. A count
/// kept is the standing object, and the stages kept hear the word beside
/// it: the count takes the send once.
#[test]
fn an_older_requests_send_is_counted_once_by_the_count_in_force() {
    for later in [stock_anew("least_load", 2), stock("least_load", 2)] {
        let engine = InProcess::new(stock("least_load", 1));
        let members = PublishedList::new(pool(2));
        let asks = Asks::of(request(), Some(&members));
        let now = Instant::now();
        let decided = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now);
        let mut first = decided.expect("a decision");
        let sent_to = first.take(0).expect("the first target");
        engine
            .apply(later)
            .expect("a number above the standing one");
        assert_eq!(sends_to(&engine, &sent_to), 0, "not said yet");
        first.dispatched(now);
        assert_eq!(sends_to(&engine, &sent_to), 1, "once");
    }
}

/// A later document that changes its `index` block alone keeps every
/// profile's stages: no stage reads that block at its build. The engine's
/// record of sent prompts takes the block's new bound where the first
/// value of a list is made under the later document.
#[test]
fn a_later_document_that_changes_its_index_alone_keeps_its_stages() {
    let engine = InProcess::new(stock("cache_aware", 1));
    let members = PublishedList::new(pool(4));
    let tokens: Vec<u32> = (0..1_024).collect();
    let sent = |seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens[..]), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        hashes.write_to(&mut asked.facts);
        let decision = decided(&engine, &Asks::of(asked, Some(&members))).decision;
        assert!(decision.pick().is_some(), "a pick");
    };
    sent(1);
    let stood = engine.standing_spec();
    let whole = engine.affinity().bound_of(MODEL);
    assert!(whole.is_some_and(|bound| bound > 1), "the default bound");

    let mut next = PolicySpec::preset("cache_aware").expect("a built-in preset");
    next.generation = 2;
    // One block's worth: the record is held to one placement.
    next.index.entry(MODEL.into()).or_default().max_size = u64::from(stood.block_size());
    let next = AppliedSpec::validate(&next).expect("the document validates");
    engine.apply(next).expect("a number above the standing one");
    let (new, old) = (engine.standing_spec().engine(), stood.engine());
    assert!(Arc::ptr_eq(&new.profiles()[0], &old.profiles()[0]), "kept");
    assert_eq!(engine.affinity().bound_of(MODEL), whole, "no list read yet");
    sent(2);
    assert_eq!(engine.affinity().bound_of(MODEL), Some(1), "the new bound");
}

thread_local! {
    /// What a thread holds from ahead of its first read of an engine, and
    /// what it holds from behind it ([`AsksAtItsEnd`]): whichever way round
    /// the thread's own storage is torn down, one of the two is dropped
    /// behind the document the thread kept.
    static AHEAD: Cell<Option<AsksAtItsEnd>> = const { Cell::new(None) };
    static BEHIND: Cell<Option<AsksAtItsEnd>> = const { Cell::new(None) };
}

/// Asks its engine for the generation in force where it is dropped, as a
/// host's value that says a word of a lease from a destructor does, and
/// writes down what the entry answered: `None` where it panicked.
struct AsksAtItsEnd(&'static InProcess, Arc<Mutex<Vec<Option<u64>>>>);

impl Drop for AsksAtItsEnd {
    fn drop(&mut self) {
        let asked = || self.0.with_spec(AppliedSpec::generation);
        let answered = catch_unwind(AssertUnwindSafe(asked));
        self.1.lock().push(answered.ok());
    }
}

/// A thread keeps one counted reference of the document it read last, and
/// an entry compares that document's number with the engine's. An entry
/// that finds it takes no reference of the document. The first entry
/// behind a new document reads the engine's cell, and the thread lets the
/// replaced document go there: until then the thread holds it.
#[test]
fn an_entry_behind_a_new_document_reads_it_and_lets_the_replaced_one_go() {
    let engine = InProcess::new(stock("random", 1));
    let first = engine.spec();
    let held = || Arc::strong_count(&first.0);
    assert_eq!(held(), 2, "the handle and the engine's cell");
    assert_eq!(engine.in_force().spec.generation(), 1);
    assert_eq!(held(), 3, "and this thread, from its first entry on");
    let again = engine.in_force();
    assert_eq!(held(), 3, "a later entry takes none");
    drop(again);

    let applied = engine.apply(stock("least_load", 2));
    applied.expect("a number above the standing one");
    assert_eq!(held(), 2, "the handle, and the thread that read it last");
    assert_eq!(engine.in_force().spec.generation(), 2, "the next entry");
    assert_eq!(held(), 1, "the handle alone");
}

/// An entry made inside another on one thread counts on the reference the
/// thread keeps: it reads no cell and takes no reference of the document.
#[test]
fn an_entry_inside_another_counts_on_the_threads_reference() {
    let engine = InProcess::new(stock("random", 1));
    let handle = engine.spec();
    let outer = engine.in_force();
    let counted = Arc::strong_count(&handle.0);
    let inner = engine.in_force();
    assert!(Rc::ptr_eq(&outer.0, &inner.0), "the thread's one reference");
    assert_eq!(engine.with_spec(AppliedSpec::generation), 1);
    assert_eq!(
        Arc::strong_count(&handle.0),
        counted,
        "none of the document"
    );
}

/// A thread keeps one document, of the engine it asked last. A thread that
/// asks two engines by turns reads each one's cell at each turn, and each
/// engine is answered its own document: the numbers are compared, and no
/// two documents of the process share one.
#[test]
fn a_thread_that_asks_two_engines_by_turns_reads_each_ones_own_document() {
    let one = InProcess::new(stock("random", 7));
    let two = InProcess::new(stock("least_load", 9));
    for _ in 0..3 {
        let first = one.in_force();
        assert_eq!(first.spec.generation(), 7);
        assert_eq!(two.in_force().spec.generation(), 9);
        let back = one.in_force();
        assert!(!Rc::ptr_eq(&first.0, &back.0), "the cell is read again");
        assert_eq!(back.applied, first.applied, "the one document");
    }
}

/// A panic that unwinds through an entry lets the entry's count go and
/// leaves the thread its reference: the next entry counts on it.
#[test]
fn a_panic_through_an_entry_leaves_the_thread_its_reference() {
    let engine = InProcess::new(stock("random", 1));
    let before = engine.in_force();
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _held = engine.in_force();
        panic!("a decision's panic");
    }));
    assert!(unwound.is_err());
    let after = engine.in_force();
    assert!(Rc::ptr_eq(&before.0, &after.0));
    let counts = Rc::strong_count(&after.0);
    assert_eq!(counts, 3, "the thread's and the two held here");
}

/// A picker of a host's own build whose drop panics while its flag is set.
#[derive(Debug)]
struct PanicsAtItsDrop(Arc<AtomicBool>);

impl Lifecycle for PanicsAtItsDrop {}

impl Picker for PanicsAtItsDrop {
    fn name(&self) -> &'static str {
        "panics-at-its-drop"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        Random.pick(facts, candidates, kept, scores, band, out);
    }
}

impl Drop for PanicsAtItsDrop {
    fn drop(&mut self) {
        assert!(!self.0.load(Ordering::Relaxed), "a stage's drop");
    }
}

/// The stages of a replaced document that the new one did not keep are
/// dropped where its last reference goes: as a rule at the next entry of
/// a thread that read it last. A stage's drop that panics there unwinds
/// out of that entry. The thread is left no document, and its next entry
/// reads the engine's cell.
#[test]
fn a_stages_drop_that_panics_at_an_entry_unwinds_out_of_it() {
    let armed = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&armed);
    let make = move || -> Box<dyn Picker> { Box::new(PanicsAtItsDrop(Arc::clone(&flag))) };
    let mut stages = StageRegistry::builtin();
    stages
        .picker("panics-at-its-drop", make)
        .expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "panics-at-its-drop".into();
    let first = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = InProcess::new(first.expect("the stages hold the picker"));
    assert_eq!(
        engine.in_force().spec.generation(),
        1,
        "the thread keeps it"
    );

    // The next document is the stock table's: it keeps no stage of the
    // first, and this thread holds the first's last reference.
    let applied = engine.apply(stock("random", 2));
    applied.expect("a number above the standing one");
    armed.store(true, Ordering::Relaxed);
    let unwound = catch_unwind(AssertUnwindSafe(|| engine.in_force().spec.generation()));
    armed.store(false, Ordering::Relaxed);
    assert!(unwound.is_err(), "the drop's panic leaves the entry");
    assert_eq!(
        engine.in_force().spec.generation(),
        2,
        "the cell, read again"
    );
}

/// A thread lets the document it read last go at its end.
#[test]
fn a_thread_lets_the_document_it_read_last_go_at_its_end() {
    let engine = InProcess::new(stock("random", 1));
    let handle = engine.spec();
    let held = || Arc::strong_count(&handle.0);
    assert_eq!(held(), 2, "the handle and the engine's cell");
    let while_it_ran = thread::scope(|scope| {
        let read = scope.spawn(|| {
            assert_eq!(engine.in_force().spec.generation(), 1);
            held()
        });
        // The join returns behind the end of the thread's own storage.
        read.join().expect("the thread's read")
    });
    assert_eq!(while_it_ran, 3, "the thread kept one");
    assert_eq!(held(), 2, "and let it go at its end");
}

/// The call that puts a document in force stores the document and then
/// its number. An entry made between the two reads a document the engine
/// held: a thread that kept the standing one decides under it once more,
/// as a decision that began ahead of the call does, and a thread that
/// kept none reads the new one of the cell. Once the number stands, every
/// entry reads the new document.
#[test]
fn an_entry_between_an_applys_two_stores_reads_a_document_the_engine_held() {
    let engine = InProcess::new(stock("random", 1));
    let standing = engine.in_force().applied;
    let applied = engine.apply(stock("least_load", 2));
    applied.expect("a number above the standing one");
    // The engine as it stands between the two stores: the new document in
    // its cell, under the standing number.
    let next = engine.applied.swap(standing, Ordering::Release);
    assert_eq!(engine.in_force().spec.generation(), 1, "the one it kept");
    let fresh = thread::scope(|scope| {
        let read = scope.spawn(|| engine.in_force().spec.generation());
        read.join().expect("the thread's read")
    });
    assert_eq!(fresh, 2, "a thread that kept none reads the cell");

    engine.applied.store(next, Ordering::Release);
    assert_eq!(engine.in_force().spec.generation(), 2, "the number stands");
}

/// An entry made while a thread's own storage is torn down answers, and
/// does not panic: where the thread's kept document is gone already, the
/// entry reads the engine's cell. A value the thread holds from ahead of
/// its first read and one it holds from behind it each ask at their drop.
#[test]
fn an_entry_made_while_a_threads_storage_is_torn_down_reads_the_cell() {
    let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(stock("random", 4))));
    let answers = Arc::new(Mutex::new(Vec::new()));
    let told = Arc::clone(&answers);
    let asking = thread::spawn(move || {
        AHEAD.set(Some(AsksAtItsEnd(engine, Arc::clone(&told))));
        assert_eq!(engine.in_force().spec.generation(), 4);
        BEHIND.set(Some(AsksAtItsEnd(engine, told)));
    });
    asking.join().expect("the thread ends");
    assert_eq!(*answers.lock(), [Some(4), Some(4)]);
}

/// The read stays on its thread, and no public value holds one: each
/// public value that holds or hands a reference of a document is `Send`
/// and `Sync`, as it was before a thread kept its document.
#[test]
fn a_public_value_that_holds_a_document_is_send_and_sync() {
    fn sent<T: Send + Sync>() {}
    sent::<InProcess>();
    sent::<SpecHandle>();
    sent::<Placement<'static>>();
    sent::<PlacementSide>();
    sent::<DecisionGuard<&'static InProcess>>();
}

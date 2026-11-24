//! The cases of a new document put in force on a running engine
//! ([`PolicyEngine::apply`]): each asks the engine's own entries, over a
//! list of worker objects and the worker crate's registry, and none polls
//! the engine's task.

use std::{cell::Cell, sync::Arc, time::Instant};

use parking_lot::Mutex;

use super::Loaded;
use crate::{
    document::spec::{AppliedSpec, BindingSpec, Match, PolicySpec},
    engine::{
        decision::{Target, Targets},
        decision_engine::{DecisionEngine, Request, RequestMaker, WorkerLists},
        framework::{Picker, TieBand},
        list_cell::PublishedList,
        policy_engine::{
            membership::tests::{handed, id, registry, remove},
            single::tests::{
                decided, hearing_spec_at, hearing_stages, pool, request, Asks, Heard, MODEL,
            },
            task::tests::stock,
            PolicyEngine,
        },
        prompt_hashes::HashSeeds,
        view::{Candidate, RequestFacts, WorkerId},
    },
    policies::random::Random,
    stages::{registry::StageRegistry, Lifecycle},
    store::{GatewayId, StaleGeneration},
    tracking::{lease::Membership, DecisionGuard},
};

impl PolicyEngine {
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
    let engine = PolicyEngine::new(among_joined(1));
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
    let engine = PolicyEngine::new(first);
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
fn sends_to(engine: &PolicyEngine, to: &Target) -> u32 {
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
    let engine = PolicyEngine::new(stock("least_load", 1));
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
        let engine = PolicyEngine::new(stock("least_load", 1));
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

/// A document whose number is not above the standing one is refused, and
/// the standing document decides on.
#[test]
fn a_document_whose_number_is_not_above_the_standing_one_is_refused() {
    let engine = PolicyEngine::new(stock("random", 2));
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
    let engine = PolicyEngine::new(stock("random", 1));
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
    let engine = PolicyEngine::new(first);
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
    engine: &'a PolicyEngine,
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
    let engine = PolicyEngine::new(first);
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
    let engine = PolicyEngine::new(stock("cache_aware", 1));
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
    let engine = PolicyEngine::new(hearing_over(&stages, 1, |_| {}));
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
    let engine = PolicyEngine::new(hearing_over(&stages, 1, second_profile));
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
    let engine = PolicyEngine::new(hearing_over(&stages, 1, second_profile));
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

/// A send said behind a later document, for a decision the document before
/// made, is counted once by the count of sends in force, built new or kept:
/// a send is a fact of its worker, whatever document decided it. A count
/// kept is the standing object, and the stages kept hear the word beside
/// it: the count takes the send once.
#[test]
fn an_older_requests_send_is_counted_once_by_the_count_in_force() {
    for later in [stock_anew("least_load", 2), stock("least_load", 2)] {
        let engine = PolicyEngine::new(stock("least_load", 1));
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
    let engine = PolicyEngine::new(stock("cache_aware", 1));
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

//! The cases of a pair's shadows by side, each pair decided through the
//! engine's own entry over the host's two lists of worker objects.
//!
//! The classic router asks each side's one policy for a pair's pick
//! (`model_gateway/src/routers/common/placement.rs:3039` for the prefill,
//! `:3101` for the decode, on main `bd55beae5`) and runs no second one
//! beside it. So a side's count is the engine's own answer in every case
//! here, and the pair a request is given is held to the one the same
//! document gives with no shadow named.

use std::{collections::BTreeMap, sync::atomic::AtomicU64};

use super::*;
use crate::{
    document::spec::{Assignment, ShadowSpec},
    engine::{
        decision::LiftedVeto,
        policy_engine::with_shadow::tests::{Heard, Rule, Takes},
        shadow::{ShadowSide, SHADOW_FLOOR},
    },
};

/// The names of a case's pickers on a pair's two sides: the live one's,
/// then its shadow's.
const SIDE_PICKERS: [(&str, &str); 2] = [
    ("takes-prefill", "takes-prefill-next"),
    ("takes-decode", "takes-decode-next"),
];

/// What a case reads of its sides' shadows: the decisions that asked each
/// side's shadow picker, the prefill's then the decode's, and the lease
/// issues every picker heard.
#[derive(Default)]
struct Seen {
    asked: [Arc<AtomicU64>; 2],
    heard: Heard,
}

impl Seen {
    /// How many decisions asked each side's shadow.
    fn asks(&self) -> [u64; 2] {
        self.asked
            .each_ref()
            .map(|asked| asked.load(Ordering::Relaxed))
    }
}

/// A host of `prefills` prefills and `decodes` decodes that pair each with
/// each, under the stock pair document with a case's picker on each side
/// and no filter: `live[0]` picks the prefill and `live[1]` the decode.
/// Each side `next` names is shadowed by a profile of the same stages
/// under that side's own picker. `edit` has the document before it is
/// applied.
fn shadowing(
    (prefills, decodes): (usize, usize),
    live: [Rule; 2],
    next: [Option<Rule>; 2],
    edit: impl FnOnce(&mut PolicySpec),
) -> (Host, Seen) {
    let seen = Seen::default();
    let picker = |name: &'static str, rule: Rule, asked: &Arc<AtomicU64>| {
        let (asked, heard) = (Arc::clone(asked), Arc::clone(&seen.heard));
        move || -> Box<dyn Picker> {
            Box::new(Takes {
                name,
                rule,
                asked: Arc::clone(&asked),
                named: Arc::default(),
                heard: Arc::clone(&heard),
            })
        }
    };
    let mut catalog = StageRegistry::builtin();
    let document = PolicySpec::pair_presets("least_load", "least_load");
    let mut document = document.expect("the pair presets");
    document.generation = 3;
    let mut sides = BTreeMap::new();
    for (at, side) in ["prefill", "decode"].into_iter().enumerate() {
        let (own, shadow) = SIDE_PICKERS[at];
        let added = catalog.picker(own, picker(own, live[at], &Arc::default()));
        added.expect("a free name");
        let profile = document.profiles.get_mut(side);
        let profile = profile.expect("the side's profile");
        profile.filter.clear();
        profile.pick = own.into();
        let Some(rule) = next[at] else { continue };
        let added = catalog.picker(shadow, picker(shadow, rule, &seen.asked[at]));
        added.expect("a free name");
        let mut twin = profile.clone();
        twin.pick = shadow.into();
        let name = format!("{side}-next");
        document.profiles.insert(name.clone(), twin);
        sides.insert(side.to_owned(), name);
    }
    if !sides.is_empty() {
        document.bindings[0].shadow = Some(ShadowSpec::Sides(sides));
    }
    edit(&mut document);
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let spec = spec.expect("a valid document");
    (Host::under((spec, Arc::default()), prefills, decodes), seen)
}

/// [`shadowing`] over two prefills and two decodes.
fn bed(live: [Rule; 2], next: [Option<Rule>; 2]) -> (Host, Seen) {
    shadowing((2, 2), live, next, |_| {})
}

/// A host of `prefills` and `decodes` workers under the stock pair of
/// `live`, the prefills' policy then the decodes', each side `next` names
/// shadowed by that stock policy's own profile.
fn stock(live: [&str; 2], next: [Option<&str>; 2], (prefills, decodes): (usize, usize)) -> Host {
    let document = PolicySpec::pair_presets(live[0], live[1]);
    let mut document = document.expect("the pair presets");
    let mut sides = BTreeMap::new();
    for (side, next) in ["prefill", "decode"].into_iter().zip(next) {
        let Some(next) = next else { continue };
        let preset = PolicySpec::preset(next).expect("a built-in preset");
        let profile = preset.profiles.into_values().next();
        let name = format!("{side}-next");
        let profile = profile.expect("the preset's profile");
        document.profiles.insert(name.clone(), profile);
        sides.insert(side.to_owned(), name);
    }
    document.bindings[0].shadow = Some(ShadowSpec::Sides(sides));
    host_of(document, prefills, decodes)
}

/// The engine's counts of the shadows of each side of the first binding's
/// pairs, the prefill's then the decode's: agreed, another target, another
/// outcome, skipped. They are read as a host is told them
/// ([`PolicyEngine::shadow_counts`]), where a side that names no shadow has
/// no count. No other table counts in a case here.
fn side_kinds(host: &Host) -> [[u64; 4]; 2] {
    side_kinds_of(host, 0)
}

/// [`side_kinds`] of the pairs of the binding at `binding`, the one
/// binding of the document that names a shadow: a host is told no count
/// of another.
fn side_kinds_of(host: &Host, binding: u32) -> [[u64; 4]; 2] {
    let mut kinds = [[0; 4]; 2];
    let _ = host.engine.shadow_counts(|count| {
        let side = match count.side {
            ShadowSide::Prefill => PairSide::Prefill,
            ShadowSide::Decode => PairSide::Decode,
            other => return assert_eq!(count.count, 0, "{other:?} counts none"),
        };
        assert_eq!(count.binding, binding, "the one binding's");
        kinds[side.index()][count.kind.index()] = count.count;
    });
    kinds
}

/// One pair asked of the host's engine through the entry that issues its
/// lease, for a request that states `budget_micros`.
fn asked(host: &Host, budget_micros: u32) -> DecideOutcome {
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let mut asks = host.asks(&classes, false);
    asks.0.facts.budget_micros = budget_micros;
    leased(host, &asks, Instant::now())
}

/// [`asked`] for a request that states no budget.
fn paired(host: &Host) -> DecideOutcome {
    asked(host, 0)
}

/// The floor of a shadow's budget, in the microseconds a request states.
fn floor() -> u32 {
    u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds")
}

/// One pair decided over the host's lists for a request that states
/// `facts`, by the lists' decision lent room for its sides' shadows: the
/// decision, and what each side's shadow said beside it.
fn said_beside(host: &Host, facts: RequestFacts<'_>) -> (Decision, SidesSaid) {
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let Asks(mut request, lists, _) = host.asks(&classes, false);
    request.facts = facts;
    let own = host.engine.loaded();
    let sides = (own.members_of(lists.prefill), own.members_of(lists.decode));
    let (mut said, now): (SidesSaid, _) = ([None; 2], Instant::now());
    let (counted, room) = (SessionCountedUnder::default, Some(&mut said));
    let decision = lists.decide_sides(
        own,
        sides,
        None,
        now,
        &mut request,
        None,
        |_| true,
        counted,
        room,
    );
    (decision, said)
}

/// A pair's binding names a shadow a side. With the decode side alone
/// named, the decode is compared and the prefill is not: one count moves,
/// the decode side's. The pair is the live profiles' own, and a pair
/// counts nothing with the singles' shadows.
#[test]
fn a_pairs_decode_shadow_alone_is_compared_and_counted_by_its_side() {
    let (plain, _) = bed([Rule::Place(0); 2], [None; 2]);
    let live = plain.picked(&paired(&plain).decision);
    assert_eq!(side_kinds(&plain), [[0; 4]; 2]);
    let (host, seen) = bed([Rule::Place(0); 2], [None, Some(Rule::Place(1))]);
    let decided = paired(&host);
    assert!(decided.lease.is_some());
    assert_eq!(host.picked(&decided.decision), live, "the live pair");
    assert_eq!(side_kinds(&host), [[0; 4], [0, 1, 0, 0]]);
    assert_eq!(seen.asks(), [0, 1]);
    // A side that names no shadow has none to skip under a short budget.
    assert!(asked(&host, floor() - 1).lease.is_some());
    assert_eq!(side_kinds(&host), [[0; 4], [0, 1, 0, 1]]);
    // A host is told no count of the prefill side or of single requests,
    // and neither table holds one.
    let document = host.engine.in_force();
    let tables = document.shadow_counts.get();
    let held = |side, kind| tables.expect("the tables").of(0, side, kind);
    for side in [ShadowSide::Single, ShadowSide::Prefill] {
        let counts = ShadowKind::ALL.map(|kind| held(side, kind).load(Ordering::Relaxed));
        assert_eq!(counts, [0; 4], "{side:?}");
    }
}

/// The decode asked again behind a pair's first decision asks no side's
/// shadow and counts none: the first decision counted its sides, and the
/// counts and the shadow pickers' asks stand as it left them. The call
/// issues the decode, as it does under a document that names no shadow.
#[test]
fn a_decode_decided_again_asks_no_sides_shadow() {
    let (host, seen) = bed([Rule::Place(0); 2], [Some(Rule::Place(0)); 2]);
    let classes = every(2, 2);
    let asks = host.asks(&classes, true);
    let now = Instant::now();
    let first = leased(&host, &asks, now);
    let counted = side_kinds(&host);
    assert_eq!((counted, seen.asks()), ([[1, 0, 0, 0]; 2], [1, 1]));

    let target = target_at(&first.decision, 0);
    let lease = first.lease.expect("the pair's lease").lease;
    let held = HeldDecision {
        decision: &first.decision,
        lease: &lease,
        index: 0,
        target: &target,
        decode: HeldDecode {
            session: first.decode_session,
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
            chosen: true,
        },
    };
    let call = now + Duration::from_millis(1);
    let engine = &host.engine;
    let again = engine.decide_decode_under(GatewayId::LOCAL, &asks, &held, call);
    let again = again.expect("a decision");
    assert!(again.lease.is_some(), "the call issues the decode");
    assert_eq!((side_kinds(&host), seen.asks()), (counted, [1, 1]));
}

/// With both sides named each is compared with its own: two counts, one a
/// side. A shadow under its live side's own rule agrees with it, and a
/// side's shadow that declines is another outcome. A side is compared by
/// its own worker of the live pair: where the live pair is the first
/// prefill with the second decode, a shadow that takes the second row
/// agrees on the decode side and takes another target on the prefill side.
#[test]
fn a_pair_with_both_sides_named_counts_one_a_side() {
    let next = [Some(Rule::Place(0)), Some(Rule::Place(1))];
    let (host, seen) = bed([Rule::Place(0); 2], next);
    paired(&host);
    assert_eq!(side_kinds(&host), [[1, 0, 0, 0], [0, 1, 0, 0]]);
    assert_eq!(seen.asks(), [1, 1]);
    let next = [Some(Rule::Nobody), Some(Rule::Place(0))];
    let (host, _) = bed([Rule::Place(0); 2], next);
    paired(&host);
    assert_eq!(side_kinds(&host), [[0, 0, 1, 0], [1, 0, 0, 0]]);
    let live = [Rule::Place(0), Rule::Place(1)];
    let (host, _) = bed(live, [Some(Rule::Place(1)); 2]);
    assert_eq!(host.picked(&paired(&host).decision), (0, 1));
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0], [1, 0, 0, 0]]);
}

/// What a side's shadow said names its binding, its kind and the worker it
/// took: the room a decision is lent holds one word a side.
#[test]
fn what_a_sides_shadow_said_names_its_binding_its_kind_and_its_worker() {
    let (host, seen) = bed([Rule::Place(0); 2], [Some(Rule::Place(1)); 2]);
    let (decision, said) = said_beside(&host, keyed());
    assert_eq!(host.picked(&decision), (0, 0));
    let took = |worker: &Arc<dyn Worker>| {
        Some(Shadowed {
            binding: 0,
            kind: ShadowKind::OtherTarget,
            first: Some(worker_id(worker.as_ref())),
        })
    };
    assert_eq!(said, [took(&host.prefills[1]), took(&host.decodes[1])]);
    assert_eq!(seen.asks(), [1, 1]);
    // The decision counts nothing: its entry does.
    assert_eq!(side_kinds(&host), [[0; 4]; 2]);
}

/// A pair's binding that stands second in its document counts under its
/// own position. What each side's shadow said names that binding, and a
/// host is told no count of another binding.
#[test]
fn a_pairs_binding_behind_another_counts_under_its_own_position() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = shadowing((2, 2), [Rule::Place(0); 2], next, |document| {
        // Another model's binding ahead of the pair's: no request here
        // states that model.
        let mut ahead = document.bindings[0].clone();
        ahead.matcher.model = "another-model".to_owned();
        ahead.shadow = None;
        document.bindings.insert(0, ahead);
    });
    let (decision, said) = said_beside(&host, keyed());
    assert_eq!(host.picked(&decision), (0, 0));
    let named = said.map(|said| said.map(|said| said.binding));
    assert_eq!(named, [Some(1); 2]);
    paired(&host);
    assert_eq!(side_kinds_of(&host, 1), [[0, 1, 0, 0]; 2]);
    assert_eq!(seen.asks(), [2, 2]);
}

/// A shadowed pair issues what the live pair alone does: one lease a side,
/// each on the live pair's worker. The stages of every profile hear those
/// two issues and none on a worker a shadow took. A pair's side claims no
/// session pin, live or as a shadow: the table holds none behind a keyed
/// pair.
#[test]
fn a_shadowed_pair_issues_the_live_pairs_leases_alone_and_claims_no_pin() {
    let (host, seen) = bed([Rule::Place(0); 2], [Some(Rule::Place(1)); 2]);
    let decided = paired(&host);
    let (prefill, decode) = host.picked(&decided.decision);
    assert_eq!((prefill, decode), (0, 0));
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0]; 2]);
    let live = [
        worker_id(host.prefills[prefill].as_ref()),
        worker_id(host.decodes[decode].as_ref()),
    ];
    let heard = std::mem::take(&mut *seen.heard.lock());
    let booked = |name| heard.iter().filter(move |word| word.0 == name);
    for (own, shadow) in SIDE_PICKERS {
        for name in [own, shadow] {
            let booked: Vec<_> = booked(name).map(|word| word.1).collect();
            assert_eq!(booked, live, "what {name} heard issued");
        }
    }
    assert_eq!(host.engine.spec().engine().sessions().len(), 0);
}

/// A pair with less than the floor of its budget keeps it for its answer:
/// no side's shadow is asked, and each named side is counted as skipped.
/// At the floor both are asked.
#[test]
fn a_pairs_budget_under_the_floor_skips_each_named_side() {
    let (host, seen) = bed([Rule::Place(0); 2], [Some(Rule::Place(1)); 2]);
    let short = asked(&host, floor() - 1);
    assert!(short.lease.is_some());
    assert_eq!(side_kinds(&host), [[0, 0, 0, 1]; 2]);
    assert_eq!(seen.asks(), [0, 0]);
    let held = asked(&host, floor());
    assert!(held.lease.is_some());
    assert_eq!(side_kinds(&host), [[0, 1, 0, 1]; 2]);
    assert_eq!(seen.asks(), [1, 1]);
}

/// A side's shadow that panics costs its pair nothing: the pair and its
/// lease are the live profiles', the fault is counted with the engine's
/// and that side counts nothing. The other side is asked all the same, and
/// the next pair decides.
#[test]
fn a_sides_shadow_that_panics_costs_its_pair_nothing_and_counts_a_fault() {
    let next = [Some(Rule::Panics), Some(Rule::Place(1))];
    let (host, seen) = bed([Rule::Place(0); 2], next);
    let decided = paired(&host);
    assert!(decided.lease.is_some());
    let live = host.picked(&decided.decision);
    assert_eq!(live, (0, 0));
    assert_eq!((host.engine.faults(), host.engine.unheard()), (1, 0));
    assert_eq!(side_kinds(&host), [[0; 4], [0, 1, 0, 0]]);
    assert_eq!(seen.asks(), [1, 1], "the side behind a fault is asked");
    assert_eq!(host.picked(&paired(&host).decision), live);
    assert_eq!(host.engine.faults(), 2);
    assert_eq!(side_kinds(&host), [[0; 4], [0, 2, 0, 0]]);
}

/// What a side's shadow said is counted whatever the pair's issue does. A
/// stage that panics at that issue is held and counted with the words a
/// stage did not hear: the pair keeps its lease, no fault is counted, and
/// its side is counted as beside any pair.
#[test]
fn a_pair_whose_issue_panics_counts_its_sides() {
    let live = [Rule::IssuePanics, Rule::Place(0)];
    let (host, seen) = shadowing((2, 2), live, [None, Some(Rule::Place(1))], |_| {});
    let decided = paired(&host);
    assert!(decided.lease.is_some(), "the pair stands");
    assert_eq!((host.engine.faults(), host.engine.unheard()), (0, 1));
    assert_eq!(seen.asks(), [0, 1]);
    assert_eq!(side_kinds(&host), [[0; 4], [0, 1, 0, 0]]);
}

/// A pair its profiles shed has no live side to set a shadow beside: no
/// side is asked, and each named side is counted as skipped, so a side's
/// counts add up to the decisions its pair's profiles made. A request
/// refused ahead of those profiles has no shadow, as a single's has none:
/// no profile decided it.
#[test]
fn a_pair_its_profiles_shed_counts_each_named_side_as_skipped() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = bed([Rule::Nobody, Rule::Place(0)], next);
    let decided = paired(&host);
    assert!(matches!(decided.decision, Decision::Shed(_)));
    assert!(decided.lease.is_none());
    assert_eq!(seen.asks(), [0, 0]);
    assert_eq!(side_kinds(&host), [[0, 0, 0, 1]; 2]);

    let classes = every(2, 2);
    let mut hinted = host.asks(&classes, false);
    hinted.0.constraints.subset_hint = Some(1);
    let refused = leased(&host, &hinted, Instant::now());
    let hint = (ShedReason::PolicyDeclined, SUBSET_HINT);
    assert_eq!(shed_for(&refused.decision), hint);
    assert_eq!(side_kinds(&host), [[0, 0, 0, 1]; 2], "none counted");

    // A list with no worker is the profiles' to shed too.
    let (alone, seen) = shadowing((2, 0), [Rule::Place(0); 2], next, |_| {});
    assert!(matches!(paired(&alone).decision, Decision::Shed(_)));
    assert_eq!(seen.asks(), [0, 0]);
    assert_eq!(side_kinds(&alone), [[0, 0, 0, 1]; 2]);
}

/// An entry that lends its decision no room asks no side's shadow: the
/// entry that issues no lease, the lists' own decision, and the decode
/// decided again behind a first decision that asked both sides.
#[test]
fn an_entry_that_lends_no_room_asks_no_sides_shadow() {
    let (host, seen) = bed([Rule::Place(0); 2], [Some(Rule::Place(1)); 2]);
    let classes = every(2, 2);
    let (asks, now) = (host.asks(&classes, true), Instant::now());
    let unleased = host.engine.decide_without_lease(&asks, now);
    let (decision, said) = unleased.expect("a decision");
    assert!(decision.pick().is_some());
    assert_eq!(said, None);
    assert!(host.decide(&OPEN, 11, None).pick().is_some());
    assert_eq!(seen.asks(), [0, 0]);

    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    assert_eq!(seen.asks(), [1, 1]);
    guard.dispatched(now);
    let again = *guard.decide_decode_under(&asks, now).expect("a decision");
    assert!(again.pick().is_some());
    assert_eq!(seen.asks(), [1, 1]);
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0]; 2]);
}

/// A side's shadow profile that would be handed another column than its
/// live side's pool carries is not asked: a shadow that reads cached
/// blocks, beside a side that reads none, is counted as skipped. The other
/// side's shadow is asked.
#[test]
fn a_sides_shadow_of_another_column_is_skipped_and_counted() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = shadowing((2, 2), [Rule::Place(0); 2], next, |document| {
        let aware = PolicySpec::preset("cache_aware").expect("a built-in preset");
        let aware = aware.profiles.into_values().next();
        let shadow = document.profiles.get_mut("decode-next");
        shadow.expect("the decode's shadow").score = aware.expect("the preset's profile").score;
    });
    paired(&host);
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0], [0, 0, 0, 1]]);
    assert_eq!(seen.asks(), [1, 0]);
}

/// Beside a pair the binding's all-vetoed rule placed, the decode's shadow
/// is not asked: its run is put to no rule, and would shed the partners
/// its live side was steered among. It is counted as skipped, and the
/// prefill's shadow is asked and compared.
#[test]
fn a_pair_a_lifted_veto_placed_counts_its_decode_shadow_as_skipped() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = shadowing((2, 2), [Rule::Place(0); 2], next, |document| {
        let stock = PolicySpec::pair_presets("least_load", "least_load");
        let stock = stock.expect("the pair presets").profiles.remove("decode");
        let filter = stock.expect("the decode's profile").filter;
        for name in ["decode", "decode-next"] {
            let profile = document.profiles.get_mut(name);
            profile
                .expect("a decode profile")
                .filter
                .clone_from(&filter);
        }
    });
    for decode in host.decodes.iter() {
        decode.set_overloaded(true);
    }
    let decided = paired(&host);
    let live = decided.decision.pick().expect("a pair");
    assert_eq!(live.trace.lifted, LiftedVeto::Overload);
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0], [0, 0, 0, 1]]);
    assert_eq!(seen.asks(), [1, 0]);
}

/// A sessions rule on the pair's binding, with no cap, that leaves a new
/// session to each side's profile.
fn sticky(document: &mut PolicySpec) {
    let manual = PolicySpec::preset("manual").expect("a built-in preset");
    document.view.headers.clone_from(&manual.view.headers);
    let mut rule = manual.bindings[0].sessions.clone().expect("its rule");
    (rule.assignment, rule.inflight_cap) = (Assignment::Delegate, u16::MAX);
    document.bindings[0].sessions = Some(rule);
}

/// Beside a side its binding's sessions rule placed, the side's shadow is
/// not asked: a shadow's run is put to no rule, and its bare profile
/// beside a key's pin would say nothing true of the profile. So a keyed
/// pair under a rule counts each named side as skipped: the key's first
/// pair, and the next behind it. The host's request states a key.
#[test]
fn a_keyed_pair_under_a_sessions_rule_counts_each_side_as_skipped() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = shadowing((2, 2), [Rule::Place(0); 2], next, sticky);
    for pairs in 1..=2 {
        assert_eq!(host.picked(&paired(&host).decision), (0, 0));
        assert_eq!(side_kinds(&host), [[0, 0, 0, pairs]; 2]);
    }
    assert_eq!(seen.asks(), [0, 0]);
}

/// A request with no key under a sessions rule is placed by its sides'
/// profiles, as under no rule: both sides' shadows are asked and compared.
#[test]
fn a_pair_with_no_key_under_a_sessions_rule_asks_both_shadows() {
    let next = [Some(Rule::Place(1)); 2];
    let (host, seen) = shadowing((2, 2), [Rule::Place(0); 2], next, sticky);
    let classes = every(2, 2);
    let mut asks = host.asks(&classes, false);
    asks.0.facts.session_key = Identity::NONE;
    leased(&host, &asks, Instant::now());
    assert_eq!(seen.asks(), [1, 1]);
    assert_eq!(side_kinds(&host), [[0, 1, 0, 0]; 2]);
}

/// A side's shadow is held to the places its host states for that side, as
/// its live side is. Each shadow here takes the first row it is shown: the
/// one place its side may take, where its live side stands. Shown its
/// whole list it would take the first place, another worker.
#[test]
fn a_sides_shadow_is_held_to_the_places_of_its_side() {
    let next = [Some(Rule::Place(0)); 2];
    let (host, seen) = shadowing((3, 3), [Rule::Place(0); 2], next, |_| {});
    let classes = every(3, 3);
    let mut asks = host.asks(&classes, false);
    asks.1.prefill_places = Some(&[1]);
    asks.1.decode_places = Some(&[2]);
    let decided = leased(&host, &asks, Instant::now());
    assert_eq!(host.picked(&decided.decision), (1, 2));
    assert_eq!(seen.asks(), [1, 1]);
    assert_eq!(side_kinds(&host), [[1, 0, 0, 0]; 2]);
}

/// A side's shadow writes nothing to the records of sent prefixes. Under
/// `cache_aware` on both sides a prompt goes back to the pair it was sent
/// to at every draw, never to the workers its shadows took: each side's
/// shadow takes its list's workers in turn, and is counted beside every
/// pair.
#[test]
fn a_sides_shadow_writes_nothing_to_the_records_of_sent_prefixes() {
    let host = stock(["cache_aware"; 2], [Some("round_robin"); 2], (4, 4));
    let classes = every(4, 4);
    let sent = prompt(&host, 1_000);
    let placed = |seed: u64| {
        let mut asks = host.asks(&classes, false);
        asks.0.facts = RequestFacts { seed, ..facts() };
        sent.write_to(&mut asks.0.facts);
        host.picked(&leased(&host, &asks, Instant::now()).decision)
    };
    let first = placed(1);
    for seed in 2..34 {
        assert_eq!(placed(seed), first, "seed {seed}");
    }
    let kinds = side_kinds(&host);
    for side in kinds {
        assert_eq!((side.iter().sum::<u64>(), side[3]), (33, 0), "{kinds:?}");
        assert!(side[1] > 16, "a rotation takes other workers: {kinds:?}");
    }
}

/// A side's shadow turns no rotation of its live side. Under round robin
/// on both sides, each shadowed by a round robin of its own, the pairs are
/// those the same document gives with no shadow named, turn by turn.
#[test]
fn a_sides_shadow_turns_no_rotation_of_its_live_side() {
    let plain = host_under("round_robin", "round_robin");
    let host = stock(["round_robin"; 2], [Some("round_robin"); 2], (4, 4));
    let turns = |host: &Host| -> Vec<(usize, usize)> {
        (0..12)
            .map(|_| host.picked(&paired(host).decision))
            .collect()
    };
    let (shadowed, unshadowed) = (turns(&host), turns(&plain));
    assert_eq!(shadowed, unshadowed);
    let prefills: BTreeSet<usize> = shadowed.iter().map(|pair| pair.0).collect();
    assert_eq!(prefills.len(), 4, "every prefill takes its turn");
    let kinds = side_kinds(&host);
    for side in kinds {
        assert_eq!((side.iter().sum::<u64>(), side[3]), (12, 0), "{kinds:?}");
    }
}

/// The ring a pair's caller lends, of both lists' workers, and the facts
/// of a keyed request that stands at `position` of it.
fn on_ring<'a>(ring: &'a InOrder, position: &'a [u64; 1]) -> RequestFacts<'a> {
    RequestFacts {
        ring: Some(ring),
        ring_positions: position,
        ..keyed()
    }
}

/// The model's one ring over three prefills and three decodes.
fn model_ring() -> InOrder {
    let address = |at: &str| format!("grpc://{at}:9000");
    let model = "decode-1 prefill-2 decode-0 prefill-0 prefill-1 decode-2";
    InOrder(model.split(' ').map(address).collect())
}

/// What the shadow of `side` took at each of six places a request states
/// on the lent ring, beside the row its live side took there.
fn taken_on_ring(host: &Host, side: PairSide) -> Vec<(usize, Option<WorkerId>)> {
    let ring = model_ring();
    let taken = |position: u64| {
        let position = [position];
        let (decision, said) = said_beside(host, on_ring(&ring, &position));
        let live = host.picked(&decision);
        let shadow = said[side.index()].expect("the side's shadow");
        assert_ne!(shadow.kind, ShadowKind::Skipped, "the shadow ran");
        let live = match side {
            PairSide::Prefill => live.0,
            PairSide::Decode => live.1,
        };
        (live, shadow.first)
    };
    (0..6).map(taken).collect()
}

/// A side's shadow that reads a ring by the place both live sides state is
/// lent the ring its request lends: under the plain ring on both sides, a
/// decode shadow of the plain ring takes the decode its live side takes,
/// at every place the request states. The live decode moves with the place
/// stated, so a shadow on a ring of its own would not follow it.
#[test]
fn a_sides_shadow_of_the_live_sides_ring_form_places_on_the_lent_ring() {
    let plain = "consistent_hashing";
    let host = stock([plain; 2], [None, Some(plain)], (3, 3));
    let taken = taken_on_ring(&host, PairSide::Decode);
    let lives: BTreeSet<usize> = taken.iter().map(|taken| taken.0).collect();
    assert!(
        lives.len() > 1,
        "the live decode follows the place: {taken:?}"
    );
    for (live, shadow) in taken {
        let live = worker_id(host.decodes[live].as_ref());
        assert_eq!(shadow, Some(live));
    }
}

/// A side's shadow that reads a ring by another place than its live side
/// states is lent none, and places on its own ring: a decode shadow of the
/// prefix form, beside the plain ring on both sides, takes one decode
/// whatever place the request states. Lent the ring, it would read the
/// routing value's place as a head's and follow it.
#[test]
fn a_sides_shadow_of_another_ring_form_places_on_its_own_ring() {
    let plain = "consistent_hashing";
    let host = stock([plain; 2], [None, Some("prefix_hash")], (3, 3));
    let taken = taken_on_ring(&host, PairSide::Decode);
    let lives: BTreeSet<usize> = taken.iter().map(|taken| taken.0).collect();
    assert!(
        lives.len() > 1,
        "the live decode follows the place: {taken:?}"
    );
    let shadows: BTreeSet<Option<WorkerId>> = taken.iter().map(|taken| taken.1).collect();
    assert_eq!(shadows.len(), 1, "the shadow reads no place: {taken:?}");
}

/// A pair is lent one ring and one list of places for its two sides, so a
/// side's shadow is held to the rule against both live sides. In each bed
/// one live side states the routing value's place on the plain ring and
/// the other reads no ring. A shadow of the prefix form is apart from that
/// place, whichever live side states it: its own, or the other one where
/// its own states none. It is lent no ring, and takes one worker whatever
/// place the request states. The four beds are the two sides' shadows
/// against each of the two live sides.
#[test]
fn a_sides_shadow_apart_from_either_live_side_is_lent_no_ring() {
    let (plain, form) = ("consistent_hashing", Some("prefix_hash"));
    let beds = [
        (PairSide::Prefill, ["least_load", plain], [form, None]),
        (PairSide::Decode, [plain, "least_load"], [None, form]),
        (PairSide::Prefill, [plain, "least_load"], [form, None]),
        (PairSide::Decode, ["least_load", plain], [None, form]),
    ];
    for (side, live, next) in beds {
        let host = stock(live, next, (3, 3));
        let taken = taken_on_ring(&host, side);
        let shadows: BTreeSet<Option<WorkerId>> = taken.iter().map(|taken| taken.1).collect();
        assert_eq!(shadows.len(), 1, "{side:?} reads no place: {taken:?}");
    }
}

//! The admission limit of a pair's prefills, wherever a prefill profile
//! lists it among its hard filters. Its name is the word a host waits on
//! for a prefill's room (`Unplaced::PrefillsFull`). A picker's decline
//! aside, where the prefill side's own run sheds, the decision names
//! that limit for a prefill with an open partner that stands but for its
//! limit, and where the pair's default takes a prefill at its limit.
//! Behind that default and its question of saturation, where no prefill
//! at the request's places stands but for its limit, the limit names no
//! shed, and the prefill side's stage is read with the limit set aside.
//! The wait can answer a call ahead of the route's rule.
//!
//! The pair presets list the limit behind the health stage and the
//! overload veto. Each state here is asked with the limit behind those
//! two and ahead of both, and is answered alike. The table's
//! seventeenth and eighteenth states list the veto ahead of the health
//! stage, and its every other state lists the two as the presets do.
//! The gateway's own pair selection asks its overload shed ahead of its
//! capacity test (`pair_pass` in
//! `model_gateway/src/routers/common/placement.rs`).
//!
//! Two states more stand outside the table. Each states a hard demand
//! and a route's rule that takes no prefill, and their case holds
//! whether the call asks that rule. One case more lists a host's own
//! filter behind the limit, one that reads the request's ring: the pass
//! that sets the limit aside is shown that ring and the request's place
//! on it.
//!
//! A worker is written as in the pair's rows
//! ([`rows`](super::super::pair::rows)): `o` under its thresholds, `v`
//! over them and `u` not ready, `+` where the rule takes it and `-` where
//! it refuses it, and `x` where it pairs with nobody. One encoder stands
//! open and serves every prefill, and every decode takes images.

use super::{
    super::pair::rows::{partners, stated},
    *,
};

/// One item, under a key its host holds as a fingerprint.
const KEYS: [Identity<'static>; 1] = [Identity::hashed(1)];

/// The prefill profile's three hard filters with the admission limit
/// behind the other two: in the presets' order, and with the overload
/// veto ahead of the health stage.
const LIMIT_LAST: [[&str; 3]; 2] = [
    ["healthy", "not-overloaded", "has-capacity"],
    ["not-overloaded", "healthy", "has-capacity"],
];

/// The same two orders with the limit ahead of the other two.
const LIMIT_FIRST: [[&str; 3]; 2] = [
    ["has-capacity", "healthy", "not-overloaded"],
    ["has-capacity", "not-overloaded", "healthy"],
];

/// An engine under the encode document of the `power_of_two` preset, with
/// bindings that shed a list wholly over its thresholds (`sheds`) or
/// steer. The prefill profile lists the health stage, the overload veto
/// and its admission limit in `order`, at the three places the preset
/// lists them.
fn under(order: [&'static str; 3], sheds: bool) -> &'static InProcess {
    let mut document = fewest(DecodePick::default());
    for binding in &mut document.bindings {
        binding.overload = Some(OverloadSpec {
            shed_when_all_vetoed: sheds,
            ..OverloadSpec::default()
        });
    }
    let prefill = document.profiles.get_mut("prefill");
    let filters = &mut prefill.expect("the prefill profile").filter;
    let at = |name: &str| {
        let named = FilterSpec::Name(name.to_owned());
        let found = filters.iter().position(|filter| *filter == named);
        found.expect("a filter of the preset")
    };
    // The preset lists the health stage, the veto and then the limit.
    let places = [at("healthy"), at("not-overloaded"), at("has-capacity")];
    let rising = places.windows(2).all(|two| two[0] < two[1]);
    assert!(rising, "the preset's order");
    // An order names each of the three once.
    let mut names = order;
    names.sort_unstable();
    assert_eq!(names, ["has-capacity", "healthy", "not-overloaded"]);
    for (place, name) in places.into_iter().zip(order) {
        filters[place] = FilterSpec::Name(name.to_owned());
    }
    hearing(document).0
}

/// The request a state asks over its lists.
#[derive(Clone, Copy, Debug)]
enum Asks {
    /// An encode request with one item: the encode pass is laid for it.
    Item,
    /// An encode request with no item, whose decode must take images.
    Image,
    /// An encode request with no item that states references beside an
    /// image: the two together lay the pass.
    Pixels,
    /// A pair's request, over the pair's two lists.
    Pair,
    /// A pair's request that states a runtime, which every worker of its
    /// two lists states too. The demand leaves both lists whole, and the
    /// pair's decision is still handed a part.
    Demand,
}

/// What a call answers a request it places on no pair: the five answers
/// these states hear. Any other stops a case at its state ([`told`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Told {
    /// Its prefills are full: the word its host waits on for room.
    Waits,
    /// A shed as saturated under the overload veto, for the prefill side.
    Saturated,
    /// A shed with no endpoint under the health stage, for the prefill
    /// side.
    NotReady,
    /// A shed with no endpoint under `partnered`, for the prefill side.
    Unpaired,
    /// A shed with no endpoint under the route's rule's name, for no
    /// side: the rule was asked and takes no prefill.
    Ruled,
}

/// One state: its two lists, the place of the one prefill at its
/// admission limit (the first or the second: no state here names a
/// third), the request, whether it states its lists' rule, the binding's
/// word, whether its profile lists the overload veto ahead of the health
/// stage, and what the call answers.
struct State {
    name: &'static str,
    prefills: &'static str,
    decodes: &'static str,
    full: usize,
    asks: Asks,
    ruled: bool,
    sheds: bool,
    veto_first: bool,
    told: Told,
}

const fn state(
    name: &'static str,
    lists: (&'static str, &'static str),
    full: usize,
    asks: Asks,
    ruled: bool,
    told: Told,
) -> State {
    State {
        name,
        prefills: lists.0,
        decodes: lists.1,
        full,
        asks,
        ruled,
        sheds: true,
        veto_first: false,
        told,
    }
}

/// The states. In the first eleven no prefill stands but for its limit:
/// each is over its thresholds or not ready, at its limit or not. In the
/// tenth and the eleventh the limit, listed first, drops the prefill over
/// its thresholds ahead of the stage that refuses the one that is not
/// ready. In the next four a prefill at its limit is waited for: it
/// stands but for its limit beside an open partner, or the binding
/// steers and the pair's default takes the prefill over its thresholds.
/// In the sixteenth the binding steers and the one decode is not ready:
/// no class is open, and the prefill side's own run ends ahead of its
/// filters. A pass that reads that shed's stage with the limit in it,
/// listed first, ends at the health stage: the limit drops the prefill
/// over its thresholds.
///
/// The seventeenth and the eighteenth list the veto ahead of the health
/// stage, and the prefill at its limit is the one that is not ready. Such
/// a pass ends at the veto there: the limit drops the prefill that is not
/// ready. One state stands behind a hard demand's part, and the other
/// over whole lists, under a binding that steers and beside a decode
/// that is not ready.
///
/// In the last a route's rule takes the one prefill at its limit, which
/// is not ready, and refuses the other, which is over its thresholds.
/// The decision among the rule's workers reads that prefill alone.
const STATES: [State; 19] = [
    state(
        "an item; the rule's one prefill is full and pairs with nobody",
        ("v- v+x", "o+"),
        1,
        Asks::Item,
        true,
        Told::Saturated,
    ),
    state(
        "an image need; the same lists",
        ("v- v+x", "o+"),
        1,
        Asks::Image,
        true,
        Told::Saturated,
    ),
    state(
        "references beside an image; the same lists",
        ("v- v+x", "o+"),
        1,
        Asks::Pixels,
        true,
        Told::Saturated,
    ),
    state(
        "an item; the rule's one prefill is full and pairs",
        ("v- v+", "o+"),
        1,
        Asks::Item,
        true,
        Told::Saturated,
    ),
    state(
        "an item; the one prefill is full",
        ("v+", "o+"),
        0,
        Asks::Item,
        false,
        Told::Saturated,
    ),
    state(
        "a hard demand; the one prefill is full",
        ("v+", "o+"),
        0,
        Asks::Demand,
        false,
        Told::Saturated,
    ),
    state(
        "a pair; the one prefill is full",
        ("v+", "o+"),
        0,
        Asks::Pair,
        false,
        Told::Saturated,
    ),
    state(
        "a pair; the one prefill is full and not ready",
        ("u+", "o+"),
        0,
        Asks::Pair,
        false,
        Told::NotReady,
    ),
    state(
        "an item; the one prefill is full and not ready",
        ("u+", "o+"),
        0,
        Asks::Item,
        false,
        Told::NotReady,
    ),
    state(
        "a hard demand; a prefill not ready beside one full and over",
        ("u+ v+", "o+"),
        1,
        Asks::Demand,
        false,
        Told::Saturated,
    ),
    state(
        "an item; a prefill not ready beside one full and over",
        ("u+ v+", "o+"),
        1,
        Asks::Item,
        false,
        Told::Saturated,
    ),
    state(
        "a hard demand; the one prefill is full and under its thresholds",
        ("o+", "o+"),
        0,
        Asks::Demand,
        false,
        Told::Waits,
    ),
    state(
        "an item; the one prefill is full and under its thresholds",
        ("o+", "o+"),
        0,
        Asks::Item,
        false,
        Told::Waits,
    ),
    state(
        "a pair; the first prefill is full and under its thresholds",
        ("o+ v+", "o+"),
        0,
        Asks::Pair,
        false,
        Told::Waits,
    ),
    State {
        sheds: false,
        ..state(
            "a pair under a binding that steers; the one prefill is full",
            ("v+", "o+"),
            0,
            Asks::Pair,
            false,
            Told::Waits,
        )
    },
    State {
        sheds: false,
        ..state(
            "a pair under a binding that steers; no decode is ready, and one prefill is full",
            ("v+ u+", "u+"),
            0,
            Asks::Pair,
            false,
            Told::Unpaired,
        )
    },
    State {
        veto_first: true,
        ..state(
            "a hard demand, the veto first; a prefill full and not ready beside one over",
            ("u+ v+", "o+"),
            0,
            Asks::Demand,
            false,
            Told::NotReady,
        )
    },
    State {
        sheds: false,
        veto_first: true,
        ..state(
            "a pair that steers, the veto first; no decode is ready, nor the full prefill",
            ("v+ u+", "u+"),
            1,
            Asks::Pair,
            false,
            Told::NotReady,
        )
    },
    state(
        "an item; the rule's one prefill is full and not ready, beside one over",
        ("u+ v-", "o+"),
        0,
        Asks::Item,
        true,
        Told::NotReady,
    ),
];

/// Two states outside the table, each with how often its call asks the
/// rule of a worker. Each states a hard demand and a route's rule that
/// takes no prefill, over a prefill that is not ready and one over its
/// thresholds. In the first the profile lists the health stage ahead of
/// the veto, and the prefill over its thresholds is at its limit. In the
/// second it lists the veto first, and the prefill that is not ready is
/// at its limit.
const RULED: [(State, usize); 2] = [
    (
        state(
            "a hard demand, a rule that takes no prefill; one not ready beside one full and over",
            ("u- v-", "o+"),
            1,
            Asks::Demand,
            true,
            Told::Ruled,
        ),
        2,
    ),
    (
        State {
            veto_first: true,
            ..state(
                "the same demand and rule, the veto first; one full and not ready beside one over",
                ("u- v-", "o+"),
                0,
                Asks::Demand,
                true,
                Told::NotReady,
            )
        },
        0,
    ),
];

/// The engines of one place of the limit. The first two list the health
/// stage ahead of the overload veto, and the last two the veto first:
/// each two under bindings that steer, and under bindings that shed.
type Engines = [[&'static InProcess; 2]; 2];

/// The engines of the two orders of `orders` ([`LIMIT_LAST`],
/// [`LIMIT_FIRST`]).
fn engines(orders: [[&'static str; 3]; 2]) -> Engines {
    orders.map(|order| [false, true].map(|sheds| under(order, sheds)))
}

/// The answer of [`heard`].
fn told(engines: Engines, state: &State) -> Told {
    heard(engines, state).0
}

/// What the engine of `engines` for the state's order of the two vetoes
/// and for its binding answers `state`, and how often the call asked
/// the state's rule of a worker. An answer of no shape of [`Told`] stops
/// the case there, with the state's name, and so does a fault of the
/// engine.
fn heard(engines: Engines, state: &State) -> (Told, usize) {
    let engine = engines[usize::from(state.veto_first)][usize::from(state.sheds)];
    let (prefills, decodes) = (stated(state.prefills), stated(state.decodes));
    let mut three = Three::of([prefills.len(), decodes.len(), 1], |list, _| list == 1);
    three.partners = partners(&prefills, &decodes);
    let demanded = matches!(state.asks, Asks::Demand);
    if demanded {
        for list in [0, 1] {
            let runtime = |_: &Arc<dyn Worker>| {
                let mut facts = FixedFacts {
                    vision: list == 1,
                    ..FixedFacts::default()
                };
                facts.demands.runtime = 2;
                facts
            };
            let workers = Arc::clone(&three.workers[list]);
            three.published[list] = PublishedList::with_facts(workers, runtime);
        }
    }
    let mut taken = Vec::new();
    for (list, workers) in [&prefills, &decodes].into_iter().enumerate() {
        for (at, worker) in workers.iter().enumerate() {
            worker.set(three.workers[list][at].as_ref());
            if worker.taken {
                taken.push(three.id(list, at));
            }
        }
    }
    let asks = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        asks.set(asks.get() + 1);
        taken.contains(&worker_id(worker))
    };
    // The place at its limit is read as the first prefill or the second,
    // and names a prefill of the state: a state that names another stops
    // here.
    let listed = || state.prefills.split_whitespace();
    let full = listed().nth(state.full).filter(|_| state.full < 2);
    let Some(full) = full else {
        panic!("{}: no such prefill", state.name);
    };
    // Over two prefills, one over its thresholds and one not ready, a
    // pass of the three hard filters with the limit listed first ends
    // where one with the limit aside ends, unless the limit is on the
    // prefill the next filter keeps. That is the one over its thresholds
    // where the health stage is next, and the one not ready where the
    // veto is. No state here states places, and a demand here leaves
    // both lists whole: so under no rule the pass that sets the limit
    // aside reads both prefills. Such a state puts the limit there, and
    // one that does not stops here.
    let any = |kind: char| listed().any(|word| word.starts_with(kind));
    let two = !state.ruled && listed().count() == 2 && any('v') && any('u');
    let kept = if state.veto_first { 'u' } else { 'v' };
    let told_apart = !two || full.starts_with(kept);
    assert!(told_apart, "{}: the limit is on {full}", state.name);
    let room: fn(usize) -> bool = match state.full {
        0 => |place| place != 0,
        _ => |place| place != 1,
    };
    let admission = Capacity(room);
    let classes = vec![1; prefills.len()];
    let mut lists = three.lists((&[PLAIN], &classes));
    lists.pair.admission = &admission;
    let items = match state.asks {
        Asks::Item => Some(&KEYS[..]),
        Asks::Image | Asks::Pixels => Some(&[][..]),
        Asks::Pair | Asks::Demand => None,
    };
    let pixels = matches!(state.asks, Asks::Pixels);
    let asked = match items {
        Some(keys) => asking(
            lists,
            EncodeItems {
                keys,
                decode_images: matches!(state.asks, Asks::Image),
                references: pixels,
                ..EncodeItems::default()
            },
        ),
        None => PlaceRequest::of(MODEL, WorkerLists::Pair(lists.pair)),
    };
    let request = PlaceRequest {
        admits: if state.ruled { Some(&admits) } else { None },
        media: MediaCounts {
            images: u16::from(pixels),
            ..MediaCounts::NONE
        },
        demands: Demands {
            runtime: if demanded { 2 } else { 0 },
            ..Demands::default()
        },
        ..asked
    };
    let answered = match engine.place(&request).map(drop) {
        Err(Unplaced::PrefillsFull) => Told::Waits,
        Err(Unplaced::Shed {
            reason: ShedReason::Saturated,
            emptied_by: "not-overloaded",
            side: Some(PairSide::Prefill),
            ..
        }) => Told::Saturated,
        Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "healthy",
            side: Some(PairSide::Prefill),
            ..
        }) => Told::NotReady,
        Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "partnered",
            side: Some(PairSide::Prefill),
            ..
        }) => Told::Unpaired,
        Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "admits",
            side: None,
            ..
        }) => Told::Ruled,
        other => panic!("{}: answered {other:?}", state.name),
    };
    let faults = (engine.faults(), engine.unheard());
    assert_eq!(faults, (0, 0), "{}", state.name);
    (answered, asks.get())
}

/// What each state is answered under a profile that lists the admission
/// limit ahead of its other filters. The health stage stands ahead of
/// the overload veto in every state but the seventeenth and the
/// eighteenth. Each state has one decode, which stands in every state
/// but the sixteenth and the eighteenth. In these states the limit names
/// a wait for a prefill with an open partner that stands but for its
/// limit, and for the one the pair's default takes.
///
/// In these states, under a binding that sheds, where each prefill at
/// the request's places is over its thresholds, none stands but for its
/// limit: the request is shed as saturated for the prefill side, under
/// the overload veto's name, and its host is told no wait. So it is for
/// a request the encode pass is laid for, by an item, by an image need
/// or by references beside an image, and for a pair that states a hard
/// demand: behind either part no question says the shed's reason, and
/// the shed takes the name of the stage that leaves the pass nobody with
/// the limit set aside. Over whole lists the count of a pair's prefills
/// sheds it so, ahead of any name. Where the one prefill is not ready
/// and under its thresholds it is not waited for either, over whole
/// lists and behind a part: the request has no endpoint, under the
/// health stage's name.
///
/// Beside a prefill that is not ready, one over its thresholds and at
/// its limit keeps the shed saturated behind a part. The limit, listed
/// first, drops that prefill ahead of the health stage, which then leaves
/// the profile's own pass nobody. The pass that reads the stage sets the
/// limit aside, so the overload veto's name stands as under the presets'
/// order.
///
/// In these states a prefill at its limit that is ready and under its
/// thresholds is waited for, behind a part too: its one partner stands.
/// Under the binding that steers, the pair's default lifts the overload
/// veto and takes the prefill over its thresholds: that prefill is
/// waited for as well.
///
/// In the sixteenth state the binding steers, the one decode is not ready,
/// and a prefill over its thresholds at its limit stands beside one that
/// is not ready. The request has no endpoint, over whole lists. No class
/// has an open partner, so the prefill side's own run ends ahead of its
/// filters. The pass that reads the shed's stage sets the limit aside
/// and ends at the overload veto. With the limit in it, listed first,
/// that pass ends at the health stage: the limit drops the prefill over
/// its thresholds. So the shed takes the name it has under the presets'
/// order: `partnered`, for a prefill that stands but for that veto and
/// its limit, and pairs with nobody that stands.
///
/// The seventeenth and the eighteenth states list the veto ahead of the
/// health stage, and the prefill at its limit is the one that is not
/// ready. The pass that reads the shed's stage sets the limit aside and
/// ends at the health stage: the request has no endpoint, behind a hard
/// demand's part and over whole lists alike. With the limit in it,
/// listed first, that pass ends at the veto: the limit drops the prefill
/// that is not ready. Behind the part that shed is saturated. Over whole
/// lists, under the binding that steers, it has no endpoint under
/// `partnered`.
///
/// Behind the part the answer is not the gateway's own selection's,
/// under either place of the limit: the decision asks no question there.
/// That selection sheds the state for overload: of its prefills the one
/// that is ready is over its thresholds (`all_overloaded` in
/// `model_gateway/src/routers/common/overload.rs`).
///
/// In the last state a route's rule takes the prefill at its limit,
/// which is not ready, and refuses the one over its thresholds. The
/// first decision reads both prefills and is shed as saturated, so the
/// rule is asked. The decision among the rule's workers reads the one
/// prefill, and the pass that reads its shed's stage sets the limit
/// aside: the request has no endpoint, under the health stage's name.
/// With the limit in it, listed first, that pass ends at the limit,
/// whose name says a wait.
///
/// A failure names each state that parts, with what it was answered. An
/// answer of no shape of [`Told`], or a fault, stops the case at its
/// state.
#[test]
fn a_refused_prefill_sides_stage_is_read_with_its_limit_aside() {
    let limit_first = engines(LIMIT_FIRST);
    let answers = STATES.iter().map(|state| (state, told(limit_first, state)));
    let parted = answers.filter(|(state, answer)| *answer != state.told);
    let parted: Vec<_> = parted
        .map(|(state, answer)| format!("{}: {answer:?}", state.name))
        .collect();
    assert!(parted.is_empty(), "{parted:#?}");
}

/// Where a profile lists the admission limit among its hard filters
/// changes no answer of these states: each is answered alike with the
/// limit behind the health stage and the overload veto, where the pair
/// presets list it, and ahead of both. The filters are the stock ones,
/// whose word on a row does not turn on how often they are asked.
///
/// A failure names each state that parts, with its answer under each
/// order. An answer of no shape of [`Told`], or a fault, stops the case
/// at its state.
#[test]
fn the_place_of_the_limit_among_a_profiles_filters_changes_no_answer() {
    let (limit_last, limit_first) = (engines(LIMIT_LAST), engines(LIMIT_FIRST));
    let answers = STATES.iter().map(|state| {
        let (behind, ahead) = (told(limit_last, state), told(limit_first, state));
        (state.name, behind, ahead)
    });
    let parted = answers.filter(|(_, behind, ahead)| behind != ahead);
    let parted: Vec<_> = parted
        .map(|(name, behind, ahead)| format!("{name}: {behind:?} behind, {ahead:?} ahead"))
        .collect();
    assert!(parted.is_empty(), "{parted:#?}");
}

/// Behind a part a call asks its route's rule by its shed's reason, and
/// the place of the limit changes neither the reason nor the ask. A
/// request that states a hard demand goes to its rule behind a shed as
/// saturated alone. Each state of [`RULED`] states such a demand and a
/// rule that takes no prefill, and is answered alike with the limit
/// behind the profile's other two hard filters and ahead of both. The
/// rule's asks of a worker are counted.
///
/// In the first state the pass that reads the shed's stage sets the
/// limit aside and ends at the overload veto: the shed is saturated, and
/// the rule is asked twice. It takes no prefill, and the request is shed
/// under the rule's name, for no side. With the limit in it, listed
/// first, that pass ends at the health stage, and the call asks no rule
/// behind that shed.
///
/// In the second state the profile lists the veto first, and that pass
/// ends at the health stage: the request has no endpoint for the prefill
/// side, and the rule is asked of no worker. With the limit in it, listed
/// first, that pass ends at the veto, and the call asks the rule behind
/// that shed.
///
/// Neither answer is the gateway's own selection's. That selection
/// judges both prefills there: a narrowing that admits no prefill is
/// left out, the wire's and its filter's alike (`admitted_pool` in
/// `model_gateway/src/routers/common/placement.rs`). It sheds both
/// states for overload: the one prefill that is ready is over its
/// thresholds (`all_overloaded` in
/// `model_gateway/src/routers/common/overload.rs`).
///
/// An answer of no shape of [`Told`], or a fault, stops the case at its
/// state.
#[test]
fn the_place_of_the_limit_changes_no_ask_of_a_routes_rule_behind_a_part() {
    let (limit_last, limit_first) = (engines(LIMIT_LAST), engines(LIMIT_FIRST));
    for (state, asks) in &RULED {
        let both = (heard(limit_last, state), heard(limit_first, state));
        let held = (state.told, *asks);
        assert_eq!(both, (held, held), "{}", state.name);
    }
}

/// A host's own filter that reads the request's ring. Shown a ring that
/// holds a worker and a place on it, it keeps the worker the walk from
/// the first such place reaches first, the key's owner. Shown no ring,
/// no place, or a ring that holds no worker, it keeps every row. At each
/// ask that shows it a ring and a place it notes the place it reads,
/// the first.
#[derive(Debug)]
struct KeysOwner(Arc<Mutex<Vec<u64>>>);

impl Filter for KeysOwner {
    fn name(&self) -> &'static str {
        "keys-owner"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        let (Some(ring), Some(&place)) = (facts.ring, facts.ring_positions.first()) else {
            return true;
        };
        self.0.lock().push(place);
        owner_at(ring, place).is_none_or(|owner| row.id == owner)
    }

    fn needs(&self) -> Needs {
        Needs {
            ring: true,
            ..Needs::default()
        }
    }
}

/// The worker a walk of `ring` from `place` reaches first, by the id its
/// address states. `None` on a ring that holds no worker.
fn owner_at(ring: &dyn LentRing, place: u64) -> Option<WorkerId> {
    let mut owner = None;
    ring.walk(place, &mut |address| {
        owner = Some(WorkerId::from_address(address));
        true
    });
    owner
}

/// The pass that reads a refused prefill side's stage is shown the
/// request's ring and its place on it. A host's own filter keeps the
/// key's owner alone where it is shown a ring that holds a worker and a
/// place on it, and every row else ([`KeysOwner`]). The live prefill
/// profile lists it behind the admission limit. The one prefill is at
/// its limit and is not the key's owner, and the one decode stands open.
///
/// The prefill does not stand but for its limit: the host's filter
/// refuses it too. So the request has no endpoint for the prefill side,
/// under that filter's name, and its host is told no wait. The profile's
/// own pass ends at the limit, whose name says a wait. A pass with the
/// limit aside that is shown no ring, or no place, would end at no
/// stage. The filter notes a place, and each place it notes is the
/// request's own.
#[test]
fn a_hosts_filter_that_reads_the_ring_names_the_shed_behind_the_limit() {
    let three = Three::of([1, 1, 1], |list, _| list == 1);
    let name = "keys-owner";
    let mut stages = StageRegistry::builtin();
    let shown = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&shown);
    let added = stages.filter(name, move || Box::new(KeysOwner(Arc::clone(&noted))));
    added.expect("a free name");
    let plain = "consistent_hashing";
    let mut document = PolicySpec::pair_presets(plain, plain).expect("the pair presets");
    let prefill = document.profiles.get_mut("prefill");
    let filters = &mut prefill.expect("the prefill profile").filter;
    // The host's filter goes right behind the limit.
    let limit = FilterSpec::Name("has-capacity".to_owned());
    let listed = filters.iter().position(|filter| *filter == limit);
    let behind = listed.expect("the preset lists the limit") + 1;
    filters.insert(behind, FilterSpec::Name(name.to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    // The ring holds the pair's two workers, and the walk from the key's
    // place reaches the decode first: the prefill is not the key's owner.
    let addresses = [0, 1].map(|list| three.workers[list][0].url().to_owned());
    let ring = Lent(Vec::from(addresses));
    let place = ring.position("conv-1");
    let owner = owner_at(&ring, place);
    assert_eq!(owner, Some(three.id(1, 0)), "the key's owner");
    let admission = Capacity(|_| false);
    let mut lists = three.lists((&[PLAIN], &[1]));
    lists.pair.admission = &admission;
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.pair))
    };
    let named = match engine.place(&request).map(drop) {
        Err(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            side: Some(PairSide::Prefill),
            ..
        }) => emptied_by,
        other => panic!("answered {other:?}"),
    };
    assert_eq!(named, name, "the stage that names the shed");
    // The filter noted a place, and each it noted is the request's own.
    let shown = shown.lock();
    let own = !shown.is_empty() && shown.iter().all(|&at| at == place);
    assert!(own, "the places the filter was shown: {shown:?}");
    assert_eq!(engine.faults(), 0, "no fault");
}

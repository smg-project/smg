//! The cases of a shadow profile on the ring a request's caller lends: which
//! shadow is lent the ring, where it walks it beside a live profile that
//! reads none, what a filter of it that declares the ring need is shown,
//! the ring a prefill's shadow judges its admission limit on, and what the
//! lender is asked for a pair's two sides' shadows.

use super::*;

/// A shadow profile is lent the request's ring only where it reads a ring
/// by the live profile's kind of place. Beside a live `consistent_hashing`
/// profile the place stated is the routing value's own text, and the
/// client's for a request its client alone keys: no place of `prefix_hash`,
/// which has nothing to hash for such a request and takes the worker with
/// the fewest requests (`PrefixHashPolicy::select_worker_impl` in
/// `model_gateway/src/policies/prefix_hash.rs`). So the shadow answers the
/// idle worker, and is counted as another target than the ring's owner,
/// which holds two requests.
#[test]
fn a_shadow_of_another_ring_form_is_lent_no_ring() {
    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    let prefix = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let twin = prefix.profiles.into_values().next();
    document
        .profiles
        .insert("twin".into(), twin.expect("the preset's profile"));
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let ring = Lent(vec![list[0].url().to_owned()]);
    list[0].increment_load();
    list[0].increment_load();
    let headers = [("x-forwarded-for", "10.0.0.1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the ring's owner");
    // Agreed, another target, another outcome, skipped.
    let counted = engine.shadow_kinds(0);
    assert_eq!(counted, [0, 1, 0, 0]);
}

/// [`Lent`], with a count of its walks.
#[derive(Debug)]
struct Walked(Lent, AtomicU32);

impl LentRing for Walked {
    fn position(&self, key: &str) -> u64 {
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.0.head_position(head)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.walk(position, take);
    }
}

/// A shadow profile that places as its live profile does is lent the
/// request's ring and walks it as the live profile does: a shadow of the
/// plain ring beside the plain ring, and a shadow of the prefix form that
/// cuts a request by the live profile's rule. The two then agree on the
/// ring's owner. A shadow of the prefix form that cuts by another depth is
/// lent none: the places stated are the live profile's heads', so the
/// lent ring is walked for the live profile alone, as beside a shadow of
/// another ring form.
#[test]
fn a_shadow_is_lent_the_ring_where_it_places_as_its_live_profile_does() {
    // The lent ring's walks under one request, and the shadow's counts.
    let walks = |live: &str, twin: &str, depth: Option<u32>| {
        let mut document = PolicySpec::preset(live).expect("a built-in preset");
        let twin = PolicySpec::preset(twin).expect("a built-in preset");
        let mut twin = twin.profiles.into_values().next().expect("its profile");
        if let Some(depth) = depth {
            twin.score[0] = ScoreStage::ConsistentHash(ConsistentHashParams {
                prefix_tokens: Some(depth),
                ..ConsistentHashParams::default()
            })
            .into();
        }
        document.profiles.insert("twin".into(), twin);
        document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
        document.generation = 1;
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let ring = Walked(Lent(vec![list[0].url().to_owned()]), AtomicU32::new(0));
        let ids: Vec<u32> = (0..64).collect();
        let headers = [("x-smg-routing-key", "conv-1")];
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers: &headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the ring's owner");
        // Agreed, another target, another outcome, skipped.
        let counted = engine.shadow_kinds(0);
        (ring.1.load(atomic::Ordering::Relaxed), counted)
    };
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    for (live, other) in [(plain, prefix), (prefix, plain)] {
        let (alone, _) = walks(live, other, None);
        assert!(alone > 0, "{live}: the live profile walks the lent ring");
        let (both, counted) = walks(live, live, None);
        assert_eq!(both, 2 * alone, "{live}: its twin walks it too");
        assert_eq!(counted, [1, 0, 0, 0], "{live}: the two agree");
    }
    let (alone, _) = walks(prefix, plain, None);
    let (another_depth, _) = walks(prefix, prefix, Some(128));
    assert_eq!(another_depth, alone, "a shadow of another depth");
}

/// A shadow profile that places by a ring is lent the request's ring
/// behind a live profile that reads none, and walks it as it would live:
/// the live profile takes the idle worker and the shadow the ring's
/// owner. The lender is asked there for the shadow alone, so by the
/// shadow's rules: a request that keeps its budget for its answer asks it
/// nothing, and a lender that panics costs its request nothing.
#[test]
fn a_shadow_alone_walks_the_lent_ring() {
    // The ring preset's binding keeps its routing key: the preset's
    // profile is its shadow, and the profile of `least_load` decides.
    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    let twin = document.profiles.keys().next().cloned();
    let live = PolicySpec::preset("least_load").expect("a built-in preset");
    let live = live.profiles.into_values().next();
    document
        .profiles
        .insert("live".into(), live.expect("the preset's profile"));
    document.bindings[0].profile = Some("live".into());
    document.bindings[0].shadow = twin.map(ShadowSpec::Profile);
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    // The key stands at the second address: the ring's owner is the
    // busy worker, and the live profile takes the idle one.
    assert_eq!(ring.position("conv-1") % 2, 1, "the key's owner");
    for _ in 0..8 {
        list[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    // Agreed, another target, another outcome, skipped.
    let counted = || engine.shadow_kinds(0);
    let placed = |ring: &dyn LentRing, budget_micros| {
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(ring),
            budget_micros,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
        (counted(), engine.faults())
    };
    let said = placed(&ring, 0);
    let walks = ring.1.load(atomic::Ordering::Relaxed);
    assert_eq!((walks, said), (1, ([0, 1, 0, 0], 0)), "the lent ring");
    // Under the shadow's floor the lender is asked nothing.
    assert_eq!(placed(&Unasked, 500), ([0, 1, 0, 1], 0), "a short budget");
    // A lender that panics is its shadow's fault: counted, and nothing
    // is compared.
    assert_eq!(placed(&Unasked, 0), ([0, 1, 0, 1], 1), "a lender's panic");
    // A request no worker serves is refused ahead of its shadow.
    let unserved = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let refused = |emptied_by| {
        Some(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            retry_after_secs: 1,
            side: None,
        })
    };
    assert_eq!(engine.place(&unserved).err(), refused("vision"));
    assert_eq!((counted(), engine.faults()), ([0, 1, 0, 1], 1), "a refusal");
    // Over a list of nobody no row stands to be placed: the lender is
    // asked nothing, and the two refusals are counted as agreed.
    let nobody = PublishedList::new(pool(0));
    let empty = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(nobody.lent()))
    };
    assert_eq!(engine.place(&empty).err(), refused("none"));
    assert_eq!((counted(), engine.faults()), ([1, 1, 0, 1], 1), "nobody");
}

/// The `least_load` pair with the ring pair of `prefill` and `decode` as
/// its sides' shadows: the ring pair's binding keeps its routing key, its
/// two profiles are the sides' shadows, and the profiles of the
/// `least_load` pair decide. `after` picks the decode once the prefill is
/// sent.
fn ring_shadowed(prefill: &str, decode: &str, after: bool) -> PolicySpec {
    let document = PolicySpec::pair_presets(prefill, decode);
    let mut document = document.expect("the pair presets");
    let live = PolicySpec::pair_presets("least_load", "least_load").expect("the pair presets");
    let profiles = &mut document.profiles;
    let sides = live.profiles.into_iter().map(|(side, live)| {
        let twin = profiles.insert(side.clone(), live);
        let name = format!("{side}-next");
        profiles.insert(name.clone(), twin.expect("the side's ring profile"));
        (side, name)
    });
    let sides = ShadowSpec::Sides(sides.collect());
    document.bindings[0].shadow = Some(sides);
    if after {
        document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    }
    document.generation = 1;
    document
}

/// What each side's shadow said under `engine`: agreed, another target,
/// another outcome, skipped; the prefill's shadow, the decode's, and the
/// decode's beside a decode decided again.
fn sides_said(engine: &InProcess) -> [[u64; 4]; 3] {
    use crate::engine::explain::shadow::ShadowSide;

    let mut kinds = [[0; 4]; 3];
    let _ = engine.shadow_counts(|count| {
        let side = match count.side {
            ShadowSide::Prefill => 0,
            ShadowSide::Decode => 1,
            ShadowSide::DecodeAgain => 2,
            _ => return,
        };
        kinds[side][count.kind.index()] = count.count;
    });
    kinds
}

/// A pair side's shadow profile that places by a ring is lent the
/// request's ring behind two live sides that read none, and walks it as it
/// would live: each live side takes the idle worker and each side's shadow
/// the ring's owner. A side's shadow of the plain ring beside the other
/// side's of the prefix form is stated its own place too: the key's text
/// for the one, the key as the request's one head for the other.
#[test]
fn a_sides_shadow_alone_walks_the_lent_ring() {
    // The lent ring's walks under one request, what each side's shadow
    // said and the engine's faults, under the pair of `ring_shadowed`.
    let walks = |prefill: &str, decode: &str, lifted: bool| {
        let document = ring_shadowed(prefill, decode, false);
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let addresses = list.iter().map(|worker| worker.url().to_owned());
        let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
        // The key stands at the second address: the ring's owner is the
        // busy worker, and each live side takes the idle one.
        assert_eq!(ring.position("conv-1") % 2, 1, "the key's owner");
        for _ in 0..8 {
            list[1].increment_load();
        }
        // Every worker overloaded: no prefill stands but for the veto, and
        // the pair is made past it, which the prefill's buffer marks. The
        // ring lent then panics where it is asked.
        if lifted {
            for worker in list.iter() {
                worker.set_overloaded(true);
            }
        }
        let headers = [("x-smg-routing-key", "conv-1")];
        let lender: &dyn LentRing = if lifted { &Unasked } else { &ring };
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(lender),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let placement = engine.place(&request).expect("a pair");
        assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
        (
            ring.1.load(atomic::Ordering::Relaxed),
            sides_said(engine),
            engine.faults(),
        )
    };
    let (plain, prefix) = ("consistent_hashing", "prefix_hash");
    let (agreed, another) = ([1, 0, 0, 0], [0, 1, 0, 0]);
    assert_eq!(
        walks(plain, plain, false),
        (2, [another, another, [0; 4]], 0),
        "the ring's owner"
    );
    // The key as the prefix form's one head stands at the idle worker on
    // the lent ring: that side's shadow agrees with its live side, and
    // the plain ring's beside it names the key's owner.
    assert_eq!(
        walks(plain, prefix, false),
        (2, [another, agreed, [0; 4]], 0),
        "the decode's prefix form"
    );
    assert_eq!(
        walks(prefix, plain, false),
        (2, [agreed, another, [0; 4]], 0),
        "the prefill's prefix form"
    );
    // Beside a pair made past a veto that left the prefills nobody, each
    // side's shadow follows the pair's default and agrees. Neither is
    // stated a place: the filters empty the prefill shadow's own set, and
    // the default asks for none. So the lender is not asked, and a ring
    // that panics where it is asked counts no fault.
    assert_eq!(
        walks(plain, plain, true),
        (0, [agreed, agreed, [0; 4]], 0),
        "the veto's mark"
    );
}

/// [`ring_shadowed`] with the prefix form as both sides' shadows, the
/// decode's `depth` tokens deep where one is named: the prefill's cuts a
/// request by the stock depth, 256 tokens. The stock weight keeps the
/// ring's owner above the least loaded worker.
fn prefix_shadowed(depth: Option<u32>, after: bool) -> PolicySpec {
    let mut document = ring_shadowed("prefix_hash", "prefix_hash", after);
    if let Some(depth) = depth {
        let decode = document.profiles.get_mut("decode-next");
        decode.expect("the decode's shadow").score[0] =
            ScoreStage::ConsistentHash(ConsistentHashParams {
                weight: 8.0,
                prefix_tokens: Some(depth),
                ..ConsistentHashParams::default()
            })
            .into();
    }
    document
}

/// A pair's two sides' shadows of the prefix form walk the lent ring once
/// each, of one depth and of two: a routing value that leads the prompt
/// is the request's one head under both rules.
#[test]
fn both_sides_shadows_of_the_prefix_form_walk_the_lent_ring() {
    let walks = |depth: Option<u32>| {
        let document = prefix_shadowed(depth, false);
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let addresses = list.iter().map(|worker| worker.url().to_owned());
        let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
        let headers = [("x-smg-routing-key", "conv-1")];
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        engine.place(&request).expect("a pair");
        (ring.1.load(atomic::Ordering::Relaxed), engine.faults())
    };
    assert_eq!(walks(None), (2, 0), "one depth");
    assert_eq!(walks(Some(128)), (2, 0), "another depth");
}

/// A decode's shadow that cuts a request by another depth than the
/// prefill's shadow is stated the place of its own head on the lent ring,
/// at the pair's decision and at the decode decided again. The prefill's
/// shadow walks the ring from the place of the prompt's 256 tokens, where
/// the walk reaches the idle worker both live sides take: it agrees. The
/// decode's walks it from the place of the first sixteen, where the walk
/// reaches the busy worker: it names another target, both times.
#[test]
fn a_decodes_shadow_of_another_depth_walks_the_lent_ring_from_its_own_head() {
    let engine = hosted(engine_of(&prefix_shadowed(Some(16), true)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    for _ in 0..8 {
        list[1].increment_load();
    }
    let ring = ByDepth {
        shallowest: list[1].url().to_owned(),
        deeper: list[0].url().to_owned(),
    };
    let ids: Vec<u32> = (0..256).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let (agreed, another) = ([1, 0, 0, 0], [0, 1, 0, 0]);
    let mut placement = engine.place(&request).expect("a pair");
    assert!(Arc::ptr_eq(&placement.worker, &list[0]), "the idle worker");
    let first = sides_said(engine);
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(
        (first, sides_said(engine)),
        ([agreed, another, [0; 4]], [agreed, another, another]),
        "the pair's decision, then the later call"
    );
    assert_eq!(engine.faults(), 0, "no fault");
}

/// A maker that hands on the engine's own and keeps the shadow profiles
/// it was asked for.
struct Recorded<'r, 'a>(Made<'r, 'a>, RefCell<Vec<u16>>);

impl RequestMaker for Recorded<'_, '_> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.0.workers()
    }

    fn shadow_ring<'w>(
        &'w self,
        spec: &'w AppliedSpec,
        shadow: u16,
    ) -> Option<(&'w dyn LentRing, &'w [u64])> {
        self.1.borrow_mut().push(shadow);
        self.0.shadow_ring(spec, shadow)
    }
}

/// A decode's shadow of another depth moves no live side, and a maker is
/// asked once for each side's shadow. Under the document of the test
/// above, a maker that hands on the engine's own is asked for two shadow
/// profiles in the pair's decision, and the request made again on it
/// states no ring and no place: what a shadow's ask kept reaches no live
/// statement. The live pair is the one the same document takes with no
/// shadow, at the pair's decision and at the decode decided again.
#[test]
fn a_decodes_shadow_of_another_depth_asks_its_own_profile_and_moves_no_live_side() {
    let live = |shadowed: bool| {
        let mut document = prefix_shadowed(Some(16), true);
        if !shadowed {
            document.bindings[0].shadow = None;
        }
        let engine = hosted(engine_of(&document));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        for _ in 0..8 {
            list[1].increment_load();
        }
        let ring = ByDepth {
            shallowest: list[1].url().to_owned(),
            deeper: list[0].url().to_owned(),
        };
        let ids: Vec<u32> = (0..256).collect();
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(&ring),
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let at = |placed: &_| list.iter().position(|member| Arc::ptr_eq(member, placed));
        let mut placement = engine.place(&request).expect("a pair");
        let first = (
            at(&placement.worker),
            placement.decode.as_ref().and_then(at),
        );
        placement.sent(Side::Prefill);
        let decided = placement.decode_again(&request);
        assert!(decided.is_ok(), "the decode decided again");
        let again = placement.decode.as_ref().and_then(at);
        let sides = (first, again, engine.faults());
        // The pair's decision once more, through a maker that keeps what
        // it was asked: then the request made again on that maker.
        let maker = Recorded(Made::new(engine, &request, &[]), RefCell::default());
        let decided = engine.decide_at(&maker, Instant::now);
        assert!(decided.is_ok(), "the pair's decision");
        let facts = maker.0.make(engine.standing_spec()).facts;
        let unstated = facts.ring.is_none()
            && facts.ring_positions.is_empty()
            && facts.decode_ring_positions.is_none();
        (sides, unstated, maker.1.take())
    };
    let (shadowed, plain) = (live(true), live(false));
    assert_eq!(shadowed.0, plain.0, "the live pair, shadowed or not");
    let pair = plain.0;
    assert!(
        matches!(pair, ((Some(0), _), Some(_), 0)),
        "the idle prefill, a decode, no fault: {pair:?}"
    );
    assert!(
        shadowed.1 && plain.1,
        "the request made again states no place"
    );
    let (asked, none) = (shadowed.2, plain.2);
    assert_eq!(none, [0_u16; 0], "no shadow, no ask");
    assert_eq!(asked.len(), 2, "one ask a side's shadow: {asked:?}");
    assert_ne!(asked[0], asked[1], "each side's own profile");
}

/// A lender that places no head is asked about one head for a pair's two
/// sides' shadows of two depths: it declined the prefill's shadow's, and
/// is asked about none for the decode's. Neither shadow walks the ring.
#[test]
fn a_lender_that_declined_one_sides_shadow_is_asked_nothing_for_the_other() {
    let engine = hosted(engine_of(&prefix_shadowed(Some(16), false)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let ring = Scripted(&[None], AtomicU32::new(0));
    let ids: Vec<u32> = (0..256).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    engine.place(&request).expect("a pair");
    let asked = ring.1.load(atomic::Ordering::Relaxed);
    assert_eq!((asked, engine.faults()), (1, 0), "one head, no walk");
}

/// A pair whose request states less than the floor of a shadow's budget
/// asks the lender nothing: neither side's shadow runs, so no head's place
/// is wanted. At the floor the same pair asks about one head, as it does
/// where no budget is stated.
#[test]
fn a_pair_under_the_shadows_floor_asks_the_lender_nothing() {
    use crate::engine::explain::shadow::SHADOW_FLOOR;

    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let ids: Vec<u32> = (0..256).collect();
    let floor = u32::try_from(SHADOW_FLOOR.as_micros()).expect("a floor in microseconds");
    let asked = |budget_micros: u32| {
        let engine = hosted(engine_of(&prefix_shadowed(Some(16), false)));
        let ring = Scripted(&[None], AtomicU32::new(0));
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(&ring),
            budget_micros,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        engine.place(&request).expect("a pair");
        (ring.1.load(atomic::Ordering::Relaxed), engine.faults())
    };
    assert_eq!(asked(floor - 1), (0, 0), "under the floor no shadow runs");
    assert_eq!(asked(floor), (1, 0), "one head, no walk");
}

/// A lender that panics when it is asked a head's place faults each
/// side's shadow that asks it, of one depth and of two: two faults a
/// pair's decision, and neither shadow says a word. The pair is placed
/// all the same.
#[test]
fn a_lender_that_panics_at_a_head_faults_both_sides_shadows() {
    let reading = |depth: Option<u32>| {
        let engine = hosted(engine_of(&prefix_shadowed(depth, false)));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let ids: Vec<u32> = (0..256).collect();
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(&Unasked),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let placed = engine.place(&request).is_ok();
        (placed, engine.faults(), sides_said(engine))
    };
    let both = (true, 2, [[0; 4]; 3]);
    assert_eq!(reading(None), both, "one depth");
    assert_eq!(reading(Some(16)), both, "two depths");
}

/// A pair's decode decided again runs the decode's shadow once more, and
/// that shadow is stated its own place on the lent ring once more: it
/// walks the lent ring a further time, and names the ring's owner beside
/// the idle worker its call takes.
#[test]
fn a_decode_decided_again_walks_the_lent_ring_once_more() {
    let plain = "consistent_hashing";
    let engine = hosted(engine_of(&ring_shadowed(plain, plain, true)));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    // The key's owner is the busy worker, as in the test above.
    for _ in 0..8 {
        list[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let seen = || (ring.1.load(atomic::Ordering::Relaxed), sides_said(engine));
    let another = [0, 1, 0, 0];
    let mut placement = engine.place(&request).expect("a pair");
    assert_eq!(
        seen(),
        (2, [another, another, [0; 4]]),
        "the first decision"
    );
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(seen(), (3, [another; 3]), "one more walk");
    assert_eq!(engine.faults(), 0, "no fault");
}

/// A decode's shadow the all-vetoed rule steers, none of whose filters
/// declares the ring need, is stated no place: every partner of the
/// prefill stands over its thresholds, the rule picks among them for the
/// shadow as it does for the live decode, and no stage reads a place. The
/// lender is not asked, at the pair's decision or at the decode decided
/// again: a ring that panics where it is asked counts no fault, and the
/// shadow agrees both times.
#[test]
fn a_steered_decodes_shadow_asks_the_lender_nothing() {
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, true);
    // The decode's shadow alone: the prefill's would ask for its place.
    if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
        sides.retain(|side, _| side.contains("decode"));
    }
    let engine = hosted(engine_of(&document));
    let members = PublishedList::new(pool(1));
    let decodes: Arc<[Arc<dyn Worker>]> = (0..2)
        .map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL))
        .collect();
    let decode_members = PublishedList::new(Arc::clone(&decodes));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..]]);
    for worker in decodes.iter() {
        worker.set_overloaded(true);
    }
    // One decode the busier: the rule's pick is the other, for both.
    for _ in 0..8 {
        decodes[1].increment_load();
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let pair = PairLists {
        decode: decode_members.lent(),
        ..pair_of(&members, &classes)
    };
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let seen = || (sides_said(engine), engine.faults());
    let agreed = [1, 0, 0, 0];
    let mut placement = engine.place(&request).expect("a pair");
    assert_eq!(seen(), ([[0; 4], agreed, [0; 4]], 0), "the pair's decision");
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(seen(), ([[0; 4], agreed, agreed], 0), "the later call");
}

/// The prefix form of the ring as a shadow behind a live profile that
/// reads none: the shadow is stated the places of its request's heads, by
/// its own rule, and walks the lent ring. A lender that places no head
/// lends the shadow no ring: it is asked nothing more, and the shadow
/// places by its own.
#[test]
fn a_prefix_shadow_alone_walks_the_lent_ring() {
    let mut document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let twin = document.profiles.keys().next().cloned();
    let live = PolicySpec::preset("least_load").expect("a built-in preset");
    let live = live.profiles.into_values().next();
    document
        .profiles
        .insert("live".into(), live.expect("the preset's profile"));
    document.bindings[0].profile = Some("live".into());
    document.bindings[0].shadow = twin.map(ShadowSpec::Profile);
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Walked(Lent(addresses.collect()), AtomicU32::new(0));
    let ids: Vec<u32> = (0..64).collect();
    // The shadow's skips and the faults, behind one more request.
    let placed = |ring: &dyn LentRing| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring: Some(ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).expect("a placement");
        let [.., skipped] = engine.shadow_kinds(0);
        (skipped, engine.faults())
    };
    assert_eq!(placed(&ring), (0, 0), "the shadow ran");
    assert_eq!(ring.1.load(atomic::Ordering::Relaxed), 1, "the lent ring");
    // A lender that places no head panics where it is asked for a key's
    // place or walked: the shadow ran, and no fault is counted.
    assert_eq!(placed(&Headless), (0, 0), "a lender of no head");
}

/// What a [`SeesPlace`] was shown: at each ask whether a ring, and the
/// places.
type Shown = BTreeSet<(bool, Vec<u64>)>;

/// A filter of the host's own that records what each ask shows it. It
/// declares the ring need or not.
#[derive(Debug)]
struct SeesPlace {
    name: &'static str,
    declares: bool,
    shown: Arc<Mutex<Shown>>,
}

impl Filter for SeesPlace {
    fn name(&self) -> &'static str {
        self.name
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        let shown = (facts.ring.is_some(), facts.ring_positions.to_vec());
        self.shown.lock().insert(shown);
        true
    }

    fn needs(&self) -> Needs {
        Needs {
            ring: self.declares,
            ..Needs::default()
        }
    }
}

/// A filter in a pair side's shadow profile that is lent the request's
/// ring. One that declares the ring need is shown the ring and the place
/// of the request's key on it at every ask, as a live profile shows it.
/// So is a filter that declares nothing and is listed ahead of it in the
/// same profile: no filter of that profile is asked ahead of the lender.
/// The other side's shadow holds a filter that declares nothing, which is
/// shown neither. Each way round: under each side's shadow at the pair's
/// decision, and under the decode's at the decode decided again. Each
/// side's shadow names the ring's owner still.
#[test]
fn a_filter_that_declares_the_ring_need_is_shown_the_ring_a_sides_shadow_is_lent() {
    let names = ["declares-ring", "declares-nothing", "listed-first"];
    // What the three filters were shown at the pair's decision and then at
    // the decode decided again, the shadow of the side `declaring` holding
    // the third ahead of the first and the other side's the second: beside
    // the place of the key on the lent ring, and what each side's shadow
    // said by the end with the engine's faults.
    let shown = |declaring: &str| {
        let cells = [(); 3].map(|()| Arc::new(Mutex::new(Shown::new())));
        let mut stages = StageRegistry::builtin();
        for (at, name) in names.into_iter().enumerate() {
            let cell = Arc::clone(&cells[at]);
            let added = stages.filter(name, move || {
                Box::new(SeesPlace {
                    name,
                    declares: at == 0,
                    shown: Arc::clone(&cell),
                })
            });
            added.expect("a free name");
        }
        let plain = "consistent_hashing";
        let mut document = ring_shadowed(plain, plain, true);
        for shadow in ["prefill-next", "decode-next"] {
            let declares = shadow == declaring;
            let filter = names[usize::from(!declares)];
            let shadow = document.profiles.get_mut(shadow);
            let shadow = shadow.expect("the side's shadow profile");
            shadow.filter.push(FilterSpec::Name(filter.to_owned()));
            if declares {
                let first = FilterSpec::Name(names[2].to_owned());
                shadow.filter.insert(0, first);
            }
        }
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let engine = hosted(InProcess::new(spec.expect("the stages hold the filters")));
        let list = pool(2);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let addresses = list.iter().map(|worker| worker.url().to_owned());
        let ring = Lent(addresses.collect());
        // The key's owner is the busy worker: each live side takes the
        // idle one, and each side's shadow names the owner.
        for _ in 0..8 {
            list[1].increment_load();
        }
        let headers = [("x-smg-routing-key", "conv-1")];
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let read = || [0, 1, 2].map(|at| std::mem::take(&mut *cells[at].lock()));
        let mut placement = engine.place(&request).expect("a pair");
        let first = read();
        placement.sent(Side::Prefill);
        let decided = placement.decode_again(&request);
        assert!(decided.is_ok(), "the decode decided again");
        let said = (sides_said(engine), engine.faults());
        (first, read(), ring.position("conv-1"), said)
    };
    // Each side's shadow names the ring's owner, as under no such filter.
    let another = [0, 1, 0, 0];
    // The decode's shadow alone is asked again: `later` are the filters it
    // holds.
    for (declaring, later) in [("prefill-next", &[1][..]), ("decode-next", &[0, 2][..])] {
        let (first, again, place, said) = shown(declaring);
        let ring = Shown::from([(true, vec![place])]);
        let none = Shown::from([(false, Vec::new())]);
        // The filter listed first is shown what the declaring one is.
        let all = [ring.clone(), none, ring];
        let mut alone = [Shown::new(), Shown::new(), Shown::new()];
        for &at in later {
            alone[at].clone_from(&all[at]);
        }
        assert_eq!((first, again), (all, alone), "{declaring} declares");
        assert_eq!(said, ([another; 3], 0), "the sides' words, and no fault");
    }
}

/// A side's shadow follows the steps of the pair's default: the
/// prefill's behind its own run's refusal, the decode's beside a pair the
/// default made, where it makes no run of its own. A filter of its
/// profile that declares the ring need is shown the lent ring and the
/// place of the request's key on it in those steps too, as it is in the
/// shadow's own run: the prefill's shadow in the default's first step,
/// the decode's in its second. Every worker is overloaded, so no prefill
/// stands but for the veto and the pair is made past it. Each side's
/// shadow agrees with its live side, and neither faults. The lender is
/// asked once for each side's shadow: for the prefill's in its run, for
/// the decode's in the default's step.
#[test]
fn a_filter_that_declares_the_ring_need_is_shown_the_lent_ring_in_the_defaults_steps() {
    let names = ["prefill-declares", "decode-declares"];
    let cells = [(); 2].map(|()| Arc::new(Mutex::new(Shown::new())));
    let mut stages = StageRegistry::builtin();
    for (at, name) in names.into_iter().enumerate() {
        let cell = Arc::clone(&cells[at]);
        let added = stages.filter(name, move || {
            Box::new(SeesPlace {
                name,
                declares: true,
                shown: Arc::clone(&cell),
            })
        });
        added.expect("a free name");
    }
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, false);
    for (shadow, name) in ["prefill-next", "decode-next"].into_iter().zip(names) {
        let shadow = document.profiles.get_mut(shadow);
        let shadow = shadow.expect("the side's shadow profile");
        shadow.filter.push(FilterSpec::Name(name.to_owned()));
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filters")));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Lent(addresses.collect());
    for worker in list.iter() {
        worker.set_overloaded(true);
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let maker = Recorded(Made::new(engine, &request, &[]), RefCell::default());
    assert!(engine.decide_at(&maker, Instant::now).is_ok(), "a pair");
    let agreed = [1, 0, 0, 0];
    let said = (sides_said(engine), engine.faults());
    assert_eq!(said, ([agreed, agreed, [0; 4]], 0), "the sides' words");
    let shown = [0, 1].map(|at| std::mem::take(&mut *cells[at].lock()));
    let placed = Shown::from([(true, vec![ring.position("conv-1")])]);
    assert_eq!(shown, [placed.clone(), placed], "each side's filter");
    let asked = maker.1.take();
    assert_eq!(asked.len(), 2, "one ask a side's shadow: {asked:?}");
    assert_ne!(asked[0], asked[1], "each side's own profile");
}

/// A filter that declares the ring need and answers by the ring it is
/// shown: with a ring it keeps the one worker it holds, with none every
/// row.
#[derive(Debug)]
struct OwnerAlone(WorkerId);

impl Filter for OwnerAlone {
    fn name(&self) -> &'static str {
        "owner-alone"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        facts.ring.is_none() || row.id == self.0
    }

    fn needs(&self) -> Needs {
        Needs {
            ring: true,
            ..Needs::default()
        }
    }
}

/// A prefill's shadow one of whose filters declares the ring need judges
/// its admission limit on the lent ring too. The pair is made among first
/// partners that do not list the key's owner, and the filter keeps the
/// owner alone where it is shown a ring. The shadow's run among them is
/// refused and none of them stands but for the limit: the shadow runs
/// again over every open class and names the owner, as a live profile with
/// that filter does. It does not say the limit's shed. Its two runs and
/// the test between them read one answer: the lender is asked once.
#[test]
fn a_prefills_shadow_judges_its_admission_limit_on_the_lent_ring() {
    let list = pool(2);
    let owner = worker_id(list[1].as_ref());
    let name = "owner-alone";
    let mut stages = StageRegistry::builtin();
    let added = stages.filter(name, move || Box::new(OwnerAlone(owner)));
    added.expect("a free name");
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, false);
    // The prefill's shadow alone: every ask of the lender is its own.
    if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
        sides.retain(|side, _| side.contains("prefill"));
    }
    let shadow = document.profiles.get_mut("prefill-next");
    let shadow = shadow.expect("the prefill's shadow profile");
    shadow.filter.push(FilterSpec::Name(name.to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    let members = PublishedList::new(Arc::clone(&list));
    // The first prefill goes first with the first decode: the pair is made
    // among those two, and the key's owner, the second worker, is not
    // listed.
    let both = &[0_u32, 1][..];
    let classes = PartnerClasses::number_first(2, [(both, &[0][..]), (both, &[][..])]);
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Lent(addresses.collect());
    assert_eq!(ring.position("conv-1") % 2, 1, "the key's owner");
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let maker = Counted(Made::new(engine, &request, &[]), AtomicU32::new(0));
    let decided = engine.decide_at(&maker, Instant::now);
    assert!(decided.is_ok(), "the pair's decision");
    // Another target, the owner: no other outcome, no fault, and one ask
    // of the lender across the shadow's two runs.
    let asks = maker.1.load(atomic::Ordering::Relaxed);
    let said = (sides_said(engine)[0], engine.faults(), asks);
    assert_eq!(said, ([0, 1, 0, 0], 0, 1), "the prefill's shadow");
}

/// A prefill's shadow whose own runs shed follows the default's first
/// step beside a pair the live profiles made too, and a filter of its
/// profile that declares the ring need is shown the lent ring there, as
/// in its runs. A second filter keeps a worker outside the pool alone
/// where it is shown a ring: both runs of the shadow are refused, and
/// the default's step takes no prefill. The first filter is shown the
/// ring and the request's place at every ask. The shadow keeps its
/// shed: another outcome, no fault, and one ask of the lender across
/// its runs and that step.
#[test]
fn a_prefills_shadow_that_sheds_is_shown_the_lent_ring_in_the_defaults_step() {
    let list = pool(2);
    let stranger = worker_id(pool(3)[2].as_ref());
    let cell = Arc::new(Mutex::new(Shown::new()));
    let seen = Arc::clone(&cell);
    // The filter that records is asked ahead of the one that refuses.
    let [records, refuses] = ["prefill-declares", "owner-alone"];
    let mut stages = StageRegistry::builtin();
    let added = stages.filter(records, move || {
        Box::new(SeesPlace {
            name: records,
            declares: true,
            shown: Arc::clone(&seen),
        })
    });
    added.expect("a free name");
    let added = stages.filter(refuses, move || Box::new(OwnerAlone(stranger)));
    added.expect("a free name");
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, false);
    if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
        sides.retain(|side, _| side.contains("prefill"));
    }
    let shadow = document.profiles.get_mut("prefill-next");
    let shadow = shadow.expect("the prefill's shadow profile");
    for name in [records, refuses] {
        shadow.filter.push(FilterSpec::Name(name.to_owned()));
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filters")));
    let members = PublishedList::new(Arc::clone(&list));
    let both = &[0_u32, 1][..];
    let classes = PartnerClasses::number_first(2, [(both, &[0][..]), (both, &[][..])]);
    let addresses = list.iter().map(|worker| worker.url().to_owned());
    let ring = Lent(addresses.collect());
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
    };
    let maker = Counted(Made::new(engine, &request, &[]), AtomicU32::new(0));
    let decided = engine.decide_at(&maker, Instant::now);
    assert!(decided.is_ok(), "the pair's decision");
    let ids = [0, 1].map(|at| worker_id(list[at].as_ref()));
    assert!(!ids.contains(&stranger), "a worker outside the pool");
    let asks = maker.1.load(atomic::Ordering::Relaxed);
    let said = (sides_said(engine)[0], engine.faults(), asks);
    assert_eq!(said, ([0, 0, 1, 0], 0, 1), "the prefill's shadow");
    let shown = std::mem::take(&mut *cell.lock());
    let placed = Shown::from([(true, vec![ring.position("conv-1")])]);
    assert_eq!(shown, placed, "the filter at every ask");
}

/// A decode's shadow the all-vetoed rule steers, one of whose filters
/// declares the ring need, asks the lender ahead of its filters: a ring
/// that panics where it is asked counts one fault of the shadow at the
/// pair's decision and one at the decode decided again, the shadow says
/// nothing, and both live decisions stand.
#[test]
fn a_steered_decodes_shadow_whose_filter_declares_the_ring_need_asks_the_lender() {
    let mut stages = StageRegistry::builtin();
    let added = stages.filter("declares-ring", || {
        Box::new(SeesPlace {
            name: "declares-ring",
            declares: true,
            shown: Arc::default(),
        })
    });
    added.expect("a free name");
    let plain = "consistent_hashing";
    let mut document = ring_shadowed(plain, plain, true);
    // The decode's shadow alone, as under the steered shadow that asks
    // nothing.
    if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
        sides.retain(|side, _| side.contains("decode"));
    }
    let shadow = document.profiles.get_mut("decode-next");
    let shadow = shadow.expect("the decode's shadow profile");
    shadow
        .filter
        .push(FilterSpec::Name("declares-ring".to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    let members = PublishedList::new(pool(1));
    let decodes: Arc<[Arc<dyn Worker>]> = (0..2)
        .map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL))
        .collect();
    let decode_members = PublishedList::new(Arc::clone(&decodes));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..]]);
    for worker in decodes.iter() {
        worker.set_overloaded(true);
    }
    let headers = [("x-smg-routing-key", "conv-1")];
    let pair = PairLists {
        decode: decode_members.lent(),
        ..pair_of(&members, &classes)
    };
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&Unasked),
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let seen = || (sides_said(engine), engine.faults());
    let mut placement = engine.place(&request).expect("a pair");
    assert_eq!(seen(), ([[0; 4]; 3], 1), "the pair's decision");
    placement.sent(Side::Prefill);
    let decided = placement.decode_again(&request);
    assert!(decided.is_ok(), "the decode decided again");
    assert_eq!(seen(), ([[0; 4]; 3], 2), "the later call");
}

/// A maker that hands on the engine's own and counts the asks for a
/// shadow's ring.
struct Counted<'r, 'a>(Made<'r, 'a>, AtomicU32);

impl RequestMaker for Counted<'_, '_> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.0.make(spec)
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.0.workers()
    }

    fn shadow_ring<'w>(
        &'w self,
        spec: &'w AppliedSpec,
        shadow: u16,
    ) -> Option<(&'w dyn LentRing, &'w [u64])> {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.shadow_ring(spec, shadow)
    }
}

/// How often a pair's decision asks the lender for the ring of a side's
/// shadow: once where the shadow's filters keep a worker, whether or not
/// one of them declares the ring need; once too where one declares it and
/// they empty the set, ahead of them; and never where none declares it
/// and they empty the set.
#[test]
fn a_sides_shadow_asks_the_lender_once_a_decision_or_never() {
    // The asks of one pair's decision and the engine's faults, under the
    // decode's shadow alone: its profile holds a filter that declares the
    // ring need or not, and every decode stands over its thresholds or
    // none does.
    let asks = |declares: bool, emptied: bool| {
        let mut stages = StageRegistry::builtin();
        let added = stages.filter("sees-place", move || {
            Box::new(SeesPlace {
                name: "sees-place",
                declares,
                shown: Arc::default(),
            })
        });
        added.expect("a free name");
        let plain = "consistent_hashing";
        let mut document = ring_shadowed(plain, plain, true);
        if let Some(ShadowSpec::Sides(sides)) = document.bindings[0].shadow.as_mut() {
            sides.retain(|side, _| side.contains("decode"));
        }
        let shadow = document.profiles.get_mut("decode-next");
        let shadow = shadow.expect("the decode's shadow profile");
        let filter = FilterSpec::Name("sees-place".to_owned());
        shadow.filter.push(filter);
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
        let members = PublishedList::new(pool(1));
        let decodes: Arc<[Arc<dyn Worker>]> = (0..2)
            .map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL))
            .collect();
        let decode_members = PublishedList::new(Arc::clone(&decodes));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..]]);
        for worker in decodes.iter() {
            worker.set_overloaded(emptied);
        }
        let addresses = decodes.iter().map(|worker| worker.url().to_owned());
        let ring = Lent(addresses.collect());
        let headers = [("x-smg-routing-key", "conv-1")];
        let pair = PairLists {
            decode: decode_members.lent(),
            ..pair_of(&members, &classes)
        };
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        let maker = Counted(Made::new(engine, &request, &[]), AtomicU32::new(0));
        let decided = engine.decide_at(&maker, Instant::now);
        assert!(decided.is_ok(), "the pair's decision");
        (maker.1.load(atomic::Ordering::Relaxed), engine.faults())
    };
    assert_eq!(asks(true, false), (1, 0), "declared, a worker kept");
    assert_eq!(asks(true, true), (1, 0), "declared, the set emptied");
    assert_eq!(asks(false, false), (1, 0), "not declared, a worker kept");
    assert_eq!(asks(false, true), (0, 0), "not declared, the set emptied");
}

/// A decode's shadow that cuts a request by another depth than the
/// prefill's shadow, and one of whose filters declares the ring need, is
/// asked for its own profile ahead of its filters: also where every
/// decode stands over its thresholds and they drop every worker. A maker
/// is asked once for each side's shadow profile in that pair's decision.
#[test]
fn a_declaring_decodes_shadow_of_another_depth_is_asked_ahead_of_its_filters() {
    let mut stages = StageRegistry::builtin();
    let added = stages.filter("sees-place", || {
        Box::new(SeesPlace {
            name: "sees-place",
            declares: true,
            shown: Arc::default(),
        })
    });
    added.expect("a free name");
    let mut document = prefix_shadowed(Some(16), true);
    let shadow = document.profiles.get_mut("decode-next");
    let shadow = shadow.expect("the decode's shadow profile");
    let filter = FilterSpec::Name("sees-place".to_owned());
    shadow.filter.push(filter);
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    let members = PublishedList::new(pool(1));
    let decodes: Arc<[Arc<dyn Worker>]> = (0..2)
        .map(|at| ready_worker(&format!("http://decode-{at}:8000"), MODEL))
        .collect();
    let decode_members = PublishedList::new(Arc::clone(&decodes));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..]]);
    for worker in decodes.iter() {
        worker.set_overloaded(true);
    }
    let ring = ByDepth {
        shallowest: decodes[0].url().to_owned(),
        deeper: decodes[1].url().to_owned(),
    };
    let ids: Vec<u32> = (0..256).collect();
    let pair = PairLists {
        decode: decode_members.lent(),
        ..pair_of(&members, &classes)
    };
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let maker = Recorded(Made::new(engine, &request, &[]), RefCell::default());
    let decided = engine.decide_at(&maker, Instant::now);
    assert!(decided.is_ok(), "the pair's decision");
    let asked = maker.1.take();
    let faults = engine.faults();
    assert_eq!(asked.len(), 2, "one ask a side's shadow: {asked:?}");
    assert_ne!(asked[0], asked[1], "each side's own profile");
    assert_eq!(faults, 0, "no fault");
}

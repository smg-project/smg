//! The cases of the hash ring a request's caller lends: where the request's
//! walk reaches, what its lender is asked about a routing value and about a
//! prompt's heads, and what a pair's two sides are lent.

use super::*;

/// A request whose caller lends its ring goes to the worker that ring's
/// walk reaches from the place of the value its document's sources derive:
/// the routing key, and with none the client's address. The gateway's own
/// policy takes the two in that order and places each by its own text
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 189 to 241 at
/// 882be6f4f). The caller states the ring and nothing else: the one call
/// reads the value's place with the request's keys. A request with neither
/// value states no place, and one that lends no ring is placed by the
/// stage's own ring. Under a bounded-load rule the ring stage reads the
/// lent ring alike: every worker here is idle, so the place's owner is
/// within the bound. A decision held to the workers a route's rule takes
/// walks the ring from the same place: the walk passes every other worker
/// over, and the first of the rule's workers it reaches is the place's
/// owner, whichever of them stands first in the list.
#[test]
fn a_request_that_lends_its_ring_goes_where_that_rings_walk_reaches() {
    lent_ring_places(hosted(preset("consistent_hashing")));
    let mut document = PolicySpec::preset("consistent_hashing").expect("a preset's name");
    for profile in document.profiles.values_mut() {
        profile.score[0] = ScoreStage::ConsistentHash(ConsistentHashParams {
            bounded_load: Some(BoundedLoadParams::default()),
            ..ConsistentHashParams::default()
        })
        .into();
    }
    lent_ring_places(hosted(engine_of(&document)));
}

/// The requests of
/// [`a_request_that_lends_its_ring_goes_where_that_rings_walk_reaches`]
/// under `engine`.
fn lent_ring_places(engine: &'static InProcess) {
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let ring = Lent(order.iter().map(|&at| list[at].url().to_owned()).collect());
    // The worker the ring's walk reaches first from where `text` stands.
    let reached = |text: &str| order[(ring.position(text) % 8) as usize];
    let placed = |headers: &dyn HeaderRead, ring: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        list.iter().position(worker).expect("a worker of the list")
    };
    // What the request's facts state: whether a ring, and the places.
    let stated = |headers: &dyn HeaderRead, ring: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        (facts.ring.is_some(), facts.ring_positions.to_vec())
    };
    let keyed = [
        ("x-forwarded-for", "10.0.0.1"),
        ("x-smg-routing-key", "conv-1"),
    ];
    let client = [("x-forwarded-for", "10.0.0.1")];
    assert_eq!(placed(&keyed, Some(&ring)), reached("conv-1"));
    assert_eq!(placed(&client, Some(&ring)), reached("10.0.0.1"));
    assert_eq!(
        stated(&keyed, Some(&ring)),
        (true, vec![ring.position("conv-1")])
    );
    assert_eq!(stated(&(), Some(&ring)), (true, Vec::new()));
    let _ = placed(&(), Some(&ring));
    // No ring lent: the stage's own ring keeps the value on one worker.
    assert_eq!(stated(&keyed, None), (false, Vec::new()));
    assert_eq!(placed(&keyed, None), placed(&keyed, None));
    // A route's rule that takes two workers alone: the sixth and the eighth
    // the walk reaches from the key's place. It refuses the first decision's
    // four targets, so the next one is held to the rule's two places.
    let mut eighth_first = 0;
    for key in ["conv-1", "conv-2", "conv-3", "conv-4", "conv-5", "conv-6"] {
        let from = ring.position(key) % 8;
        let along = |steps: u64| order[((from + steps) % 8) as usize];
        let taken = [along(5), along(7)].map(|at| worker_id(list[at].as_ref()));
        let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
        let headers = [("x-smg-routing-key", key)];
        let ruled = PlaceRequest {
            headers: &headers,
            ring: Some(&ring),
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&ruled).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[along(5)]), "{key}");
        eighth_first += usize::from(along(7) < along(5));
    }
    assert!(eighth_first > 0, "no key has the eighth ahead in the list");
}

/// [`Lent`], with a count of the keys and the heads it was asked the place
/// of.
#[derive(Debug)]
struct Asked(Lent, AtomicU32);

impl LentRing for Asked {
    fn position(&self, key: &str) -> u64 {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.position(key)
    }

    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        self.1.fetch_add(1, atomic::Ordering::Relaxed);
        self.0.head_position(head)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        self.0.walk(position, take);
    }
}

/// A request's key is placed on the lent ring once: a second making of the
/// request states the place the first took, and the ring is asked nothing.
#[test]
fn a_requests_key_is_asked_of_the_lent_ring_once() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let order = vec![list[1].url().to_owned(), list[0].url().to_owned()];
    let ring = Asked(Lent(order), AtomicU32::new(0));
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let first = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    let again = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    assert_eq!(first, [ring.0.position("conv-1")]);
    assert_eq!(again, first);
    assert_eq!(ring.1.load(atomic::Ordering::Relaxed), 1);
}

/// Under `prefix_hash` a lender that places no head lends no ring: the
/// request states no place, and the stage's own ring places it on the
/// worker it has with no ring lent. The lender is not asked where a key's
/// own text stands, which is no place of this form: a routing value and a
/// prompt are answered alike. A document with no ring stage asks a ring
/// nothing at all.
#[test]
fn a_lender_that_places_no_head_lends_no_ring_under_the_prefix_form() {
    let tokens: Vec<u32> = (0..64).collect();
    let keyed = [("x-smg-routing-key", "conv-1")];
    let rings: [(&str, &'static dyn LentRing); 2] =
        [("prefix_hash", &Headless), ("random", &Unasked)];
    for (name, ring) in rings {
        let engine = hosted(preset(name));
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        for headers in [&keyed as &dyn HeaderRead, &()] {
            let request = |ring: Option<&'static dyn LentRing>| PlaceRequest {
                prompt: Prompt {
                    tokens: Some(&tokens),
                    ..Prompt::NONE
                },
                headers,
                ring,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
            };
            let placed = |request: &PlaceRequest<'_>| {
                let placement = engine.place(request).expect("a placement");
                let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
                list.iter().position(worker).expect("a worker of the list")
            };
            let lent = request(Some(ring));
            let made = Made::new(engine, &lent, &[]);
            let facts = made.make(engine.standing_spec()).facts;
            assert!(facts.ring.is_none(), "{name}");
            assert!(facts.ring_positions.is_empty(), "{name}");
            let with = placed(&lent);
            if name == "prefix_hash" {
                assert_eq!(with, placed(&request(None)), "the stage's own ring");
            }
        }
    }
}

/// Under the prefix form a request through the one call states where its
/// lender puts each of its heads, the deepest first: a head at each of the
/// view's boundaries the prompt reaches, and for a prompt that reaches
/// none the one head of the form's depth, which a shorter prompt is whole.
/// The gateway's own policy cuts a prompt so
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 361 to 386 at
/// 882be6f4f). A routing value leads the prompt: its text is the one head,
/// placed as a head is and never as a key's own text. A client's value
/// places nothing. The request goes where the ring's walk reaches from its
/// first place. The lender is asked once for each head: a second making
/// states the places the first took.
#[test]
fn a_request_under_the_prefix_form_states_where_its_heads_stand() {
    let engine = hosted(prefixed(&[64, 256]));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let urls = order.iter().map(|&at| list[at].url().to_owned());
    let ring = Asked(Lent(urls.collect()), AtomicU32::new(0));
    let asks = || ring.1.load(atomic::Ordering::Relaxed);
    let at = |head: Head<'_>| ring.0.head_position(head).expect("a lender of heads");
    // The request's places, and the worker of the list it went to.
    let stated = |tokens: Option<&[u32]>, text: Option<&str>, headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens,
                text,
                ..Prompt::NONE
            },
            headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let before = asks();
        let facts = made.make(engine.standing_spec()).facts;
        let places = facts.ring_positions.to_vec();
        assert_eq!(
            facts.ring.is_some(),
            !places.is_empty(),
            "lent with a place"
        );
        assert_eq!(asks() - before, places.len() as u32, "one ask a head");
        assert_eq!(
            made.make(engine.standing_spec()).facts.ring_positions,
            places
        );
        assert_eq!(asks() - before, places.len() as u32, "a second making");
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        let went = list.iter().position(worker).expect("a worker of the list");
        if let Some(&first) = places.first() {
            assert_eq!(went, order[(first % 8) as usize], "the walk's first");
        }
        places
    };
    let ids: Vec<u32> = (500..800).collect();
    let both = [at(Head::Tokens(&ids[..256])), at(Head::Tokens(&ids[..64]))];
    assert_eq!(stated(Some(&ids), None, &()), both);
    assert_eq!(
        stated(Some(&ids[..100]), None, &()),
        [at(Head::Tokens(&ids[..64]))]
    );
    assert_eq!(
        stated(Some(&ids[..40]), None, &()),
        [at(Head::Tokens(&ids[..40]))]
    );
    // A text is cut at four characters a token.
    let text: String = (0..300).map(|at| format!("{at:04}")).collect();
    let cut = [at(Head::Text(&text[..1024])), at(Head::Text(&text[..256]))];
    assert_eq!(stated(None, Some(&text), &()), cut);
    assert_eq!(stated(Some(&[]), Some(&text), &()), cut, "no token id");
    let keyed = [("x-smg-routing-key", "conv-1")];
    let led = stated(Some(&ids), None, &keyed);
    assert_eq!(led, [at(Head::Text("conv-1"))]);
    assert_ne!(led, [ring.0.position("conv-1")], "the key's own text");
    let client = [("x-client", "10.0.0.1")];
    assert_eq!(stated(Some(&ids), None, &client), both);
    assert_eq!(stated(None, None, &client), [0_u64; 0]);
}

/// The `prefix_hash` preset over a view of `boundaries`, whose binding
/// reads a routing value from the header `x-smg-routing-key` and then from
/// the request's own id, the turn and retry suffixes off: the sources a
/// gateway's sticky override names. The view lists the header, so its
/// value leaves in the clear as the id's does. `rule` is the binding's
/// sessions rule.
fn prefixed_by_id(boundaries: &[u32], rule: Option<SessionsSpec>) -> InProcess {
    let mut document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    document.view.boundaries = boundaries.to_vec();
    document.view.headers.push("x-smg-routing-key".to_owned());
    let header = KeySourceSpec {
        header: Some("x-smg-routing-key".to_owned()),
        ..KeySourceSpec::default()
    };
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    let binding = &mut document.bindings[0];
    binding.routing_key = Some(RoutingKeySpec {
        sources: vec![header, rid],
        strip_lineage: true,
    });
    binding.sessions = rule;
    engine_of(&document)
}

/// Under the prefix form a routing value leads the prompt only where a
/// header handed it. The gateway's own policy hashes a header's value
/// (`model_gateway/src/policies/prefix_hash.rs:358`) and places every other
/// request by its prompt (`:360-372`), one keyed by its own id among them.
/// So a request whose value its own id handed states its prompt's heads,
/// and lands where the same prompt lands with no value, on a lent ring and
/// on the stage's own. The value is still the request's session key, and
/// the one call says where it came from.
#[test]
fn a_value_the_requests_own_id_handed_leads_no_prompt_under_the_prefix_form() {
    let engine = hosted(prefixed_by_id(&[64, 256], None));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let urls = order.iter().map(|&at| list[at].url().to_owned());
    let ring = Lent(urls.collect());
    let at = |head: Head<'_>| ring.head_position(head).expect("a lender of heads");
    let ids: Vec<u32> = (500..800).collect();
    // The request's places, whether its own id handed its value, the value
    // in the clear, and the worker of the list it went to.
    let stated = |headers: &dyn HeaderRead, id: Option<&str>, lent: bool| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers,
            request_id: id,
            ring: lent.then_some(&ring as &dyn LentRing),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        let (places, by_id) = (facts.ring_positions.to_vec(), facts.session_key_from_rid);
        let key = facts.session_key.plain.map(str::to_owned);
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        let went = list.iter().position(worker).expect("a worker of the list");
        (places, by_id, key, went)
    };
    let heads = vec![at(Head::Tokens(&ids[..256])), at(Head::Tokens(&ids[..64]))];
    let keyed = [("x-smg-routing-key", "conv-1")];
    for lent in [true, false] {
        let places = |heads: &[u64]| if lent { heads.to_vec() } else { Vec::new() };
        let unkeyed = stated(&(), None, lent);
        assert_eq!(unkeyed, (places(&heads), false, None, unkeyed.3));
        // The id's value, its turn and retry suffixes off: the prompt's
        // heads and the prompt's worker, under two ids.
        for id in ["conv-1_t2_r1", "conv-2"] {
            let key = id.split('_').next().map(str::to_owned);
            let by_id = stated(&(), Some(id), lent);
            assert_eq!(by_id, (places(&heads), true, key, unkeyed.3), "{id}");
        }
        // A header's value leads: its text is the one head, whatever the
        // request's id.
        let led = stated(&keyed, Some("conv-9"), lent);
        let head = [at(Head::Text("conv-1"))];
        assert_eq!(led.0, places(&head));
        assert_eq!((led.1, led.2.as_deref()), (false, Some("conv-1")));
    }
    assert_ne!(heads[0], at(Head::Text("conv-1")), "two places");
}

/// A sessions rule pins the value the request's own id handed, as it pins
/// a header's: the key's next request follows the pin, whatever its
/// prompt. The ring placed the key's first request by its prompt, where a
/// request of that prompt with no value lands, and another id's request of
/// the second prompt lands on that prompt's worker.
#[test]
fn a_sessions_rule_pins_the_value_the_requests_own_id_handed() {
    let rule = SessionsSpec {
        sticky_header: String::new(),
        assignment: Assignment::Delegate,
        recovery: Recovery::Latest,
        ..PolicySpec::preset("manual")
            .expect("a built-in preset")
            .bindings
            .remove(0)
            .sessions
            .expect("the manual preset's rule")
    };
    let engine = hosted(prefixed_by_id(&[], Some(rule)));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let went = |tokens: &[u32], id: Option<&str>| {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(tokens),
                ..Prompt::NONE
            },
            request_id: id,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        let worker = |held: &Arc<dyn Worker>| Arc::ptr_eq(held, &placement.worker);
        list.iter().position(worker).expect("a worker of the list")
    };
    let prompt = |from: u32| (from..from + 256).collect::<Vec<u32>>();
    let first = prompt(500);
    let home = went(&first, None);
    // A second prompt whose worker is another.
    let (second, other) = (1..)
        .map(|n| prompt(n * 1000))
        .map(|tokens| {
            let worker = went(&tokens, None);
            (tokens, worker)
        })
        .find(|&(_, worker)| worker != home)
        .expect("a prompt with another worker");
    assert_eq!(went(&first, Some("conv-1_t1")), home, "by its prompt");
    assert_eq!(went(&second, Some("conv-1_t2")), home, "by its pin");
    assert_eq!(went(&second, Some("conv-2_t1")), other, "another id");
}

/// A prompt that reaches more boundaries than its list of places holds in
/// itself states every one of them, the deepest first, and is tried at
/// each in turn: the gateway's own policy lists every boundary a prompt
/// reaches and tries each until one's owner is under the bound
/// (`model_gateway/src/policies/prefix_hash.rs`, lines 172 to 180 and 291
/// to 319 at 882be6f4f). Here nine. The owner of the eight deepest is past
/// the bound, and the ninth's takes the request, though two workers hold
/// fewer requests. A prompt that reaches a tenth boundary states ten
/// places: none is lost and none stated twice.
#[test]
fn a_prompt_that_reaches_nine_boundaries_is_tried_at_the_ninth() {
    let boundaries: Vec<u32> = (1..=10).map(|at| 16 * at).collect();
    let engine = hosted(prefixed(&boundaries));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    (0..30).for_each(|_| list[0].increment_load());
    (0..5).for_each(|_| list[1].increment_load());
    let ring = ByDepth {
        shallowest: list[1].url().to_owned(),
        deeper: list[0].url().to_owned(),
    };
    let ids: Vec<u32> = (0..150).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let places = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    assert_eq!(places, [144, 128, 112, 96, 80, 64, 48, 32, 16]);
    let placement = engine.place(&request).expect("a placement");
    assert!(
        Arc::ptr_eq(&placement.worker, &list[1]),
        "the ninth's owner"
    );
    let ids: Vec<u32> = (0..160).collect();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&ids),
            ..Prompt::NONE
        },
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let made = Made::new(engine, &request, &[]);
    let places = made
        .make(engine.standing_spec())
        .facts
        .ring_positions
        .to_vec();
    assert_eq!(places, [160, 144, 128, 112, 96, 80, 64, 48, 32, 16]);
}

/// A scorer that scores every worker alike, and counts how often it is
/// asked what it needs and how it places by heads.
#[derive(Debug)]
struct CountsAsks(Arc<AtomicU32>);

impl Lifecycle for CountsAsks {}

impl Scorer for CountsAsks {
    fn name(&self) -> &'static str {
        "counts-asks"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        self.0.fetch_add(1, atomic::Ordering::Relaxed);
        Needs::default()
    }

    fn places_by_heads(&self) -> Option<HeadRule<'_>> {
        self.0.fetch_add(1, atomic::Ordering::Relaxed);
        None
    }
}

/// A stage says how it places by heads, and what it needs, where its
/// generation is built. No making of a request and no decision asks it
/// again, under a lent ring or under none: the engine answers from what
/// the build kept.
#[test]
fn a_stage_is_asked_how_it_places_where_its_generation_is_built() {
    let asks = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&asks);
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("counts-asks", move || {
        Box::new(CountsAsks(Arc::clone(&counted)))
    });
    added.expect("a free name");
    // The prefix form's document, the counting stage beside its ring stage.
    let document = PolicySpec::preset("prefix_hash").expect("a built-in preset");
    let mut document = serde_json::to_value(document).expect("the document as JSON");
    let profiles = document["profiles"].as_object_mut();
    for profile in profiles.expect("the profiles").values_mut() {
        let score = profile["score"].as_array_mut();
        score
            .expect("the profile's scorers")
            .push("counts-asks".into());
    }
    let mut document: PolicySpec = serde_json::from_value(document).expect("a document");
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the scorer")));
    let built = asks.load(atomic::Ordering::Relaxed);
    assert!(built > 0, "the build asks the stage");

    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let lent = Lent(vec![list[0].url().to_owned()]);
    let ids: Vec<u32> = (0..64).collect();
    for ring in [Some(&lent as &dyn LentRing), None] {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            ring,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        assert_eq!(facts.ring.is_some(), ring.is_some());
        engine.place(&request).expect("a placement");
    }
    let asked = asks.load(atomic::Ordering::Relaxed);
    assert_eq!(asked, built, "no decision asks the stage");
}

/// A lender that answers nothing for a head lends no ring, whichever head
/// it declines: the request states no place, a place answered ahead of the
/// `None` among them, and the lender is asked about no head behind that
/// one. A routing value is the request's one head: where the lender
/// declines it, the prompt behind it is not asked about. A second making
/// asks nothing again.
#[test]
fn a_lender_that_declines_a_head_is_asked_nothing_more() {
    let engine = hosted(prefixed(&[16, 32, 48]));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let ids: Vec<u32> = (0..48).collect();
    let keyed = [("x-smg-routing-key", "conv-1")];
    // The lender's script, the request's headers, the asks the lender
    // counts.
    let rows: [(&'static [Option<u64>], &dyn HeaderRead, u32); 4] = [
        (&[None, Some(1), Some(2)], &(), 1),
        (&[Some(1), None, Some(2)], &(), 2),
        (&[Some(1), Some(2), None], &(), 3),
        (&[None, Some(1), Some(2)], &keyed, 1),
    ];
    for (script, headers, asks) in rows {
        let ring = Scripted(script, AtomicU32::new(0));
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            headers,
            ring: Some(&ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let made = Made::new(engine, &request, &[]);
        for making in 0..2 {
            let facts = made.make(engine.standing_spec()).facts;
            assert!(facts.ring.is_none(), "{script:?}, making {making}");
            assert!(facts.ring_positions.is_empty(), "{script:?}");
            let asked = ring.1.load(atomic::Ordering::Relaxed);
            assert_eq!(asked, asks, "{script:?}, making {making}");
        }
    }
}

/// A pair's request is lent the ring where a side's ring stage places a
/// routing value by its own text, the prefill's or the decode's: both
/// sides are handed the one ring, and each takes it by its own stage
/// ([`RequestFacts::ring`]). The prefix form is lent it by the same rule,
/// and states the lender's place of the routing value as a head. A side
/// that reads no ring holds nothing back. A pair with no ring stage is
/// lent none: the caller's ring is asked nothing. A pair of two ring forms
/// or of two depths has no one list of places, and the engine's two
/// questions answer nothing for it: each of its sides is stated its own
/// (`pair::ring_words`).
#[test]
fn a_pairs_request_is_lent_the_ring_a_side_of_it_places_by() {
    let list = pool(2);
    let headers = [("x-smg-routing-key", "conv-1")];
    let ring = Lent(vec![list[1].url().to_owned(), list[0].url().to_owned()]);
    let stated = |document: PolicySpec, ring: &dyn LentRing| {
        let engine = engine_of(&document);
        let members = PublishedList::new(Arc::clone(&list));
        let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
        let pair = pair_of(&members, &classes);
        let request = PlaceRequest {
            headers: &headers,
            ring: Some(ring),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        let made = Made::new(&engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        (facts.ring.is_some(), facts.ring_positions.to_vec())
    };
    let pair = |prefill: &str, decode: &str| {
        PolicySpec::pair_presets(prefill, decode).expect("the pair presets")
    };
    let (ringed, prefixed, plain) = ("consistent_hashing", "prefix_hash", "round_robin");
    let lent = (true, vec![ring.position("conv-1")]);
    assert_eq!(stated(pair(ringed, ringed), &ring), lent);
    assert_eq!(stated(pair(plain, ringed), &ring), lent);
    assert_eq!(stated(pair(ringed, plain), &ring), lent);
    let head = ring.head_position(Head::Text("conv-1"));
    let lent = (true, head.into_iter().collect::<Vec<_>>());
    assert_eq!(lent.1.len(), 1);
    assert_eq!(stated(pair(prefixed, prefixed), &ring), lent);
    assert_eq!(stated(pair(plain, prefixed), &ring), lent);
    assert_eq!(stated(pair(prefixed, plain), &ring), lent);
    let none = (false, Vec::new());
    assert_eq!(stated(pair(plain, plain), &Unasked), none);
    // The engine's own word, phase by phase: a pair asked at its prefill
    // is answered for both its profiles, and a decode asked alone for its
    // own.
    let depths = |document: PolicySpec| {
        let engine = engine_of(&document);
        let spec = engine.standing_spec();
        let rule = |phase| spec.engine().states_head_places(MODEL, phase);
        [Phase::Prefill, Phase::Decode].map(|phase| rule(phase).map(|rule| rule.depth))
    };
    assert_eq!(depths(pair(prefixed, plain)), [Some(256), None]);
    assert_eq!(depths(pair(plain, prefixed)), [Some(256), Some(256)]);
    assert_eq!(depths(pair(prefixed, ringed)), [None, None]);
    assert_eq!(depths(pair(ringed, prefixed)), [None, Some(256)]);
}

/// An `epd` binding pairs its prefill and its decode as a `pd-pair`
/// binding does: asked at its prefill, the engine answers the lent ring's
/// two questions for the decode profile too. Here the decode profile alone
/// reads the ring, by a routing value's own text and then by heads.
#[test]
fn an_epd_models_prefill_is_answered_for_its_decode_profile_on_a_lent_ring() {
    let epd = |prefill: &str, decode: &str| {
        PolicySpec::epd_presets("round_robin", prefill, decode).expect("the epd presets")
    };
    let (ringed, prefixed, plain) = ("consistent_hashing", "prefix_hash", "round_robin");
    let phases = [Phase::Prefill, Phase::Decode];
    let by_text = |document: PolicySpec| {
        let engine = engine_of(&document);
        phases.map(|phase| engine.spec().engine().states_key_place(MODEL, phase))
    };
    assert_eq!(by_text(epd(plain, ringed)), [true, true]);
    assert_eq!(by_text(epd(ringed, plain)), [true, false]);
    assert_eq!(by_text(epd(plain, plain)), [false, false]);
    let depths = |document: PolicySpec| {
        let engine = engine_of(&document);
        let spec = engine.standing_spec();
        let rule = |phase| spec.engine().states_head_places(MODEL, phase);
        phases.map(|phase| rule(phase).map(|rule| rule.depth))
    };
    assert_eq!(depths(epd(plain, prefixed)), [Some(256), Some(256)]);
    assert_eq!(depths(epd(prefixed, plain)), [Some(256), None]);
    assert_eq!(depths(epd(plain, plain)), [None, None]);
}

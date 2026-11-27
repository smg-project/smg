//! The cases of what a caller states of a request and of the request the
//! engine makes of it: its id and its seed, its routing value, the headers
//! a document's view lists, its prompt's blocks, and the budget and the
//! output limit it states.

use super::*;

/// The request's routing value is the first source of its binding that
/// holds a value: the header ahead of the request's own id. A header the
/// document shows no stage leaves as its fingerprint alone, and the
/// request's id in the clear. The client's value is read only where no
/// source holds a routing value. Under a document that names no source a
/// request carries neither, whatever its headers.
#[test]
fn the_routing_value_and_the_clients_value_come_from_the_documents_sources() {
    let engine = keyed();
    let keys = |headers: &dyn HeaderRead, request_id| {
        let request = PlaceRequest {
            headers,
            request_id,
            ..PlaceRequest::of(MODEL, WorkerLists::None)
        };
        let made = Made::new(&engine, &request, &[]);
        let facts = made.make(engine.standing_spec()).facts;
        let clear = facts.session_key.plain.is_some();
        assert_eq!(
            facts.session_counted, clear,
            "a key in the clear is counted"
        );
        (facts.session_key.hash, clear, facts.client_key)
    };
    let both = [("x-conversation", "talk-7"), ("x-client", "10.0.0.7")];
    assert_eq!(
        keys(&both, Some("request-1")),
        (fingerprint(b"talk-7"), false, 0),
        "the header is the first source"
    );
    let client = [("x-client", "10.0.0.7")];
    assert_eq!(
        keys(&client, Some("request-1")),
        (fingerprint(b"request-1"), true, 0),
        "then the request's own id"
    );
    assert_eq!(
        keys(&client, None),
        (0, false, fingerprint(b"10.0.0.7")),
        "the client's value where no source holds a routing value"
    );
    assert_eq!(keys(&(), None), (0, false, 0));

    let unkeyed = preset("round_robin");
    let request = PlaceRequest {
        headers: &both,
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&unkeyed, &request, &[]);
    let facts = made.make(unkeyed.standing_spec()).facts;
    assert_eq!((facts.session_key, facts.client_key), (Identity::NONE, 0));
}

/// What a caller states is the request's fact as stated: the attempt's
/// number beside the workers tried, which need not count alike, its own
/// id, and each plain field. The targets this call's earlier decisions
/// missed are left out of the next one. A request with a pair's two lists
/// is made at its prefill. A prompt's `Debug` form shows sizes alone.
#[test]
fn a_request_is_made_of_what_its_caller_states() {
    let engine = preset("round_robin");
    let (tried, missed) = ([WorkerId(5), WorkerId(9)], [WorkerId(3)]);
    let media = MediaCounts {
        images: 2,
        ..MediaCounts::NONE
    };
    let request = PlaceRequest {
        request_id: Some("request-1"),
        tenant: Some("tenant-a"),
        priority: 3,
        streaming: true,
        max_output_tokens: 256,
        media,
        attempt: 1,
        tried: &tried,
        budget_micros: 5_000,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &missed);
    let Request {
        id,
        facts,
        constraints,
    } = made.make(engine.standing_spec());
    assert_eq!(id, b"request-1");
    assert_eq!((facts.model, facts.phase), (MODEL, Phase::Single));
    assert_eq!((facts.attempt, facts.tried), (1, &tried[..]));
    assert_eq!(facts.tenant, Identity::plain("tenant-a"));
    assert_eq!((facts.priority, facts.streaming), (3, true));
    assert_eq!((facts.max_output_tokens, facts.budget_micros), (256, 5_000));
    assert_eq!(facts.media, media);
    assert_eq!(constraints.excluded, &missed[..]);
    assert!(constraints.demands.is_none() && !constraints.decode_later);
    // A request with no id bears an empty one.
    let bare = PlaceRequest::of(MODEL, WorkerLists::None);
    assert!(Made::new(&engine, &bare, &[])
        .make(engine.standing_spec())
        .id
        .is_empty());
    // A request with a pair's two lists is made, and says it is plain, at
    // its prefill.
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let paired = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let made = Made::new(&engine, &paired, &[]);
    assert_eq!(
        made.make(engine.standing_spec()).facts.phase,
        Phase::Prefill
    );
    assert_eq!(made.plain().map(|plain| plain.phase), Some(Phase::Prefill));
    // And so is a request with an encode request's three.
    let three = PlaceRequest::of(MODEL, three_of(pair));
    let made = Made::new(&engine, &three, &[]);
    assert_eq!(
        made.make(engine.standing_spec()).facts.phase,
        Phase::Prefill
    );
    // A client's tokens and text reach no log line.
    let prompt = Prompt {
        tokens: Some(&[7, 8, 9]),
        text: Some("secret words"),
        ..Prompt::NONE
    };
    let shown = format!("{prompt:?}");
    assert!(shown.contains("tokens: Some(3)"), "{shown}");
    assert!(shown.contains("text: Some(12)"), "{shown}");
    assert!(!shown.contains("secret"), "{shown}");
}

/// A request states its id with no making. An engine that keeps a sample of
/// its decisions asks it of a plain request, which is never made. A request
/// with no id states an empty one.
#[test]
fn a_request_states_its_id_with_no_making() {
    let engine = preset("round_robin");
    let request = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    assert_eq!(made.request_id(engine.standing_spec()), b"request-1");
    assert!(made.headers.get().is_none(), "the request was not made");
    let _ = made.make(engine.standing_spec());
    assert!(made.headers.get().is_some(), "a making fills its headers");
    let bare = PlaceRequest::of(MODEL, WorkerLists::None);
    let made = Made::new(&engine, &bare, &[]);
    assert!(made.request_id(engine.standing_spec()).is_empty());
}

/// The id a host states for a kept decision alone is no id of the request
/// made. Under a document that keys by the request's id, a request that
/// states one is made with that id and its key. One that states an explain
/// id alone is made as a request that states neither: an empty id and no
/// key. The maker answers the explain id beside the request's own.
#[test]
fn an_explain_id_is_no_id_of_the_request_made() {
    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.generation = 1;
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        }],
        strip_lineage: false,
    });
    let engine = engine_of(&document);
    let spec = engine.standing_spec();
    let stated = |request_id, explain_id| PlaceRequest {
        request_id,
        explain_id,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let keys = |request: &Request<'_>| {
        let facts = &request.facts;
        let keyed = facts.session_key != Identity::NONE;
        (keyed, facts.session_key_from_rid, facts.client_key)
    };

    let both = stated(Some("request-1"), Some("explain-1"));
    let made = Made::new(&engine, &both, &[]);
    let request = made.make(spec);
    assert_eq!(request.id, b"request-1");
    assert!(keys(&request).0, "keyed by its own id");
    let ids = (made.request_id(spec), made.explain_id());
    assert_eq!(ids, (&b"request-1"[..], &b"explain-1"[..]));

    let named = stated(None, Some("explain-1"));
    let bare = stated(None, None);
    let named = Made::new(&engine, &named, &[]);
    let bare = Made::new(&engine, &bare, &[]);
    let (request, unnamed) = (named.make(spec), bare.make(spec));
    assert!(request.id.is_empty(), "the explain id is not the request's");
    assert!(!keys(&request).0, "and it is no key");
    assert_eq!(keys(&request), keys(&unnamed));
    assert_eq!(named.explain_id(), b"explain-1");
    assert!(bare.explain_id().is_empty());
}

/// Each making draws a seed no other making of the thread drew, and a
/// second thread draws others.
#[test]
fn each_making_draws_a_seed_of_its_own() {
    let draw = || {
        let engine = preset("random");
        let request = PlaceRequest::of(MODEL, WorkerLists::None);
        let made = Made::new(&engine, &request, &[]);
        (0..64)
            .map(|_| made.make(engine.standing_spec()).facts.seed)
            .collect::<BTreeSet<u64>>()
    };
    let here = draw();
    assert_eq!(here.len(), 64, "no seed twice");
    let there = thread::spawn(draw).join().expect("the thread ends");
    assert_eq!(there.len(), 64);
    assert!(here.is_disjoint(&there), "two threads draw apart");
}

/// The prompt's blocks are the ones the engine hashes of it under the
/// seeds its caller states, token ids ahead of a text, and they are hashed
/// once: a second making borrows the first one's blocks. Token ids alone
/// and a text alone are hashed too. A request with no prompt states no
/// block and no size, and the seed of its partition like any other.
///
/// A text alone states no count of tokens, so a stage that credits a send
/// of no stated size credits it at its own mean. The classic router's
/// `least_load` prices a request with no token ids at its mean prefill
/// (`model_gateway/src/policies/least_load.rs:812-816`). The text states
/// its two lengths. Its bytes are at hand, and a reader that sizes a
/// request of text reads them over four. Its characters are counted
/// because a stage of this document reads a request's size. Token ids
/// win: beside them no length is stated.
#[test]
fn the_prompts_blocks_are_hashed_under_its_partition_once() {
    let engine = preset("cache_aware");
    let (tokens, text) = ((0..600).collect::<Vec<u32>>(), "t".repeat(4_000));
    let seeds = HashSeeds {
        event: 11,
        partition: 13,
    };
    let stated = |tokens, text| PlaceRequest {
        prompt: Prompt {
            tokens,
            text,
            seeds,
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };

    let request = stated(Some(&tokens[..]), Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let (first, again) = (
        made.make(engine.standing_spec()).facts,
        made.make(engine.standing_spec()).facts,
    );
    let mut expected = RequestFacts::default();
    let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, seeds);
    hashes.write_to(&mut expected);
    assert!(!first.block_hashes.is_empty());
    assert_eq!(first.block_hashes, expected.block_hashes);
    assert_eq!(
        (first.hash_kind, first.block_size),
        (HashKind::Token, expected.block_size)
    );
    assert_eq!((first.namespace_seed, first.prompt_tokens), (13, 600));
    assert_eq!(first.prompt_chars, 0, "the text is not passed over");
    assert!(
        ptr::eq(first.block_hashes, again.block_hashes),
        "hashed once"
    );

    // Token ids alone, as a tokenized request states its prompt.
    let request = stated(Some(&tokens[..]), None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert_eq!(facts.block_hashes, expected.block_hashes);
    assert_eq!(
        (facts.hash_kind, facts.prompt_tokens),
        (HashKind::Token, 600)
    );

    let request = stated(None, Some(&text[..]));
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert_eq!(facts.hash_kind, HashKind::Text);
    // The bytes are stated under any document, and the size is theirs.
    assert_eq!(
        (facts.prompt_tokens, facts.prompt_bytes, facts.prompt_size()),
        (0, 4_000, 1_000)
    );
    assert!(!facts.block_hashes.is_empty());

    let request = stated(None, None);
    let made = Made::new(&engine, &request, &[]);
    let facts = made.make(engine.standing_spec()).facts;
    assert!(facts.block_hashes.is_empty());
    assert_eq!(facts.prompt_tokens, 0);
    assert_eq!(facts.namespace_seed, 13, "its partition all the same");
}

/// A request that is lent a worker is hashed by the block size the host
/// knows under the id the first worker of its list registered under, the
/// id the list's table finds its index by, whatever size the host knows
/// under the id the request states. A pair is hashed by its prefill list.
#[test]
fn a_prompt_is_hashed_by_the_block_size_of_its_workers_id() {
    struct Learned;
    impl CacheIndexes for Learned {
        fn index(&self, _: &str) -> Option<Arc<dyn CacheIndex>> {
            None
        }

        fn block_size(&self, model: &str) -> Option<u32> {
            Some(if model == MODEL { 4 } else { 8 })
        }
    }
    let engine = preset("cache_aware");
    engine.affinity.set_indexes(Arc::new(Learned));
    let members = PublishedList::new(pool(2));
    // The pair's decode registered under the id the request states.
    let decodes = PublishedList::new([ready_worker("http://decode-0:8000", "another-id")].into());
    let classes = PartnerClasses::number(1, [&[0_u32][..], &[0][..]]);
    let pair = PairLists {
        decode: decodes.lent(),
        ..pair_of(&members, &classes)
    };
    let tokens: Vec<u32> = (0..600).collect();
    for lists in [
        WorkerLists::Single(members.lent()),
        WorkerLists::Pair(pair),
        three_of(pair),
    ] {
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&tokens),
                ..Prompt::NONE
            },
            ..PlaceRequest::of("another-id", lists)
        };
        let made = Made::new(&engine, &request, &[]);
        assert_eq!(made.make(engine.standing_spec()).facts.block_size, 4);
    }
}

/// A prompt is hashed under the document its decision hands the maker,
/// whatever document the engine holds in force by then: a decision that
/// began under one document reads no word of the next. The first document
/// reads a prompt's blocks, and the one put in force behind it reads none.
#[test]
fn a_prompt_is_hashed_under_the_document_its_decision_runs_under() {
    let engine = preset("cache_aware");
    let began_under = engine.spec();
    let mut next = PolicySpec::preset("random").expect("a built-in preset");
    next.generation = 2;
    let next = AppliedSpec::validate(&next).expect("the document validates");
    engine.apply(next).expect("a number above the standing one");

    let tokens = (0..600).collect::<Vec<u32>>();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    let hashed = made.make(&began_under).facts.block_hashes.len();
    assert!(hashed > 0, "the decision's document reads the blocks");
    let made = Made::new(&engine, &request, &[]);
    let in_force = engine.spec();
    assert!(made.make(&in_force).facts.block_hashes.is_empty());
}

/// A plain pick's lease sizes the prompt under the document its decision
/// runs under, whatever document the engine holds in force by then. The
/// first document's picker places by itself, reads a request's size and
/// takes a lease's issue. Its pick puts the `random` preset in force,
/// none of whose stages reads a size, ahead of the lease.
#[test]
fn a_plain_picks_lease_sizes_its_prompt_under_its_decisions_document() {
    const NAME: &str = "sizes-alone";
    type Host = Arc<OnceLock<&'static InProcess>>;

    /// Keeps the characters each lease it hears issued states.
    #[derive(Debug)]
    struct SizesAlone(Arc<Mutex<Vec<u32>>>, Host);

    impl Lifecycle for SizesAlone {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.lock().push(lease.chars);
        }
    }

    impl Picker for SizesAlone {
        fn name(&self) -> &'static str {
            NAME
        }

        fn needs(&self) -> Needs {
            Needs {
                prompt_size: true,
                ..Needs::default()
            }
        }

        fn refuses_by_reports(&self) -> bool {
            false
        }

        fn picks_alone(&self) -> bool {
            true
        }

        // The pool's first worker, behind the next document's apply.
        fn pick_alone(
            &self,
            _: &PoolWord<'_>,
            keep: &mut dyn FnMut(usize) -> bool,
        ) -> Option<usize> {
            let engine = self.1.get().expect("the engine is hosted");
            let mut next = PolicySpec::preset("random").expect("a built-in preset");
            next.generation = 2;
            let next = AppliedSpec::validate(&next).expect("the document validates");
            engine.apply(next).expect("a number above the standing one");
            keep(0).then_some(0)
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            push_drawn(candidates, kept, out);
        }
    }

    let chars = Arc::new(Mutex::new(Vec::new()));
    let host = Host::default();
    let (kept, hosted_at) = (Arc::clone(&chars), Arc::clone(&host));
    let mut stages = StageRegistry::builtin();
    stages
        .picker(NAME, move || {
            Box::new(SizesAlone(Arc::clone(&kept), Arc::clone(&hosted_at)))
        })
        .expect("a free picker name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    let profile = document.profiles.get_mut("random");
    profile.expect("the random profile").pick = NAME.into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("a valid document");
    let engine = hosted(InProcess::new(spec));
    let began_under = engine.spec();
    assert!(began_under.needs().prompt_size, "a stage reads the size");
    let place = began_under.place_for(MODEL, Phase::Single);
    assert!(matches!(place, Place::Alone(_)), "{place:?}");
    host.set(engine).expect("hosted once");

    // A text of 200 two-byte characters, held as text alone.
    let text = "\u{e9}".repeat(200);
    let members = PublishedList::new(pool(4));
    let request = PlaceRequest {
        prompt: Prompt {
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(placement.owes_events(), "the pick has a lease");
    let in_force = engine.spec();
    assert_eq!(in_force.generation(), 2, "the pick put the next in force");
    assert!(!in_force.needs().prompt_size, "no stage of it reads a size");
    assert_eq!(*chars.lock(), [200], "sized under the decision's document");
}

/// A first attempt that names nothing tried or missed, demands nothing,
/// carries no encode item and asks no decode again says it is plain, with
/// the size its request would state. The engine then answers it a place where one pick decides its
/// place: no clock is read for it. Each fact that asks more than a worker
/// of the list takes the request off that path. A decode asked again
/// does not: over one list the word asks nothing. Neither does a list
/// whose host states facts of its workers: a request that demands nothing
/// reads none of them.
#[test]
fn a_plain_request_says_so_and_is_answered_a_place() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens = [7_u32; 40];
    let plain = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let said = Made::new(&engine, &plain, &[]).plain();
    assert_eq!(
        said,
        Some(Plain {
            phase: Phase::Single,
            wires: WireModes::NONE,
            prompt_tokens: 40,
        })
    );
    // A text alone, as a request that is not tokenized states its prompt,
    // says no count of tokens. Asked its text's lengths, it states the
    // bytes under any document, and counts the characters under a
    // document a stage of which reads a request's size, as `bucket` does,
    // and none under this one. The text is of 200 two-byte characters.
    let text = "\u{e9}".repeat(200);
    let worded = PlaceRequest {
        prompt: Prompt {
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, plain.lists)
    };
    let sizes = |engine: &InProcess, request: &PlaceRequest<'_>| {
        let (document, call) = (engine.in_force(), Call::default());
        let made = Made::asked(engine.under(&document), request, &call);
        let said = made.plain().expect("a plain request");
        (said.prompt_tokens, made.prompt_chars(), made.prompt_bytes())
    };
    assert_eq!(sizes(&engine, &worded), (0, 0, 400));
    assert_eq!(sizes(&preset("bucket"), &worded), (0, 200, 400));
    // Token ids win: a request that holds its ids and a text states the
    // count of the ids, no length of the text and no pass over it.
    let both = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            text: Some(&text[..]),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, plain.lists)
    };
    assert_eq!(sizes(&preset("bucket"), &both), (40, 0, 0));
    let read = Cell::new(0);
    let clock = || {
        read.set(read.get() + 1);
        Instant::now()
    };
    let decided = engine.decide_at(&Made::new(&engine, &plain, &[]), clock);
    assert!(matches!(decided, Ok(Decided::Alone(pick)) if pick.place < 4));
    assert_eq!(read.get(), 0, "a place dates nothing");
    // A list whose host states a fact of each worker, as a host that dials
    // several runtimes does: the fill of the list's cell takes the facts,
    // and the plain request is answered a place as over the silent list.
    let stating = PublishedList::with_facts(Arc::clone(&list), |_| {
        let mut facts = FixedFacts::default();
        facts.demands.runtime = 7;
        facts
    });
    let stated = PlaceRequest {
        prompt: plain.prompt,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(stating.lent()))
    };
    assert_eq!(Made::new(&engine, &stated, &[]).plain(), said);
    let decided = engine.decide_at(&Made::new(&engine, &stated, &[]), clock);
    assert!(matches!(decided, Ok(Decided::Alone(pick)) if pick.place < 4));
    assert_eq!(read.get(), 0, "a place dates nothing");

    let tried = [WorkerId(1)];
    let demands = Demands {
        vision: true,
        ..Demands::default()
    };
    let more = [
        PlaceRequest {
            attempt: 1,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            tried: &tried,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            demands,
            ..PlaceRequest::of(MODEL, plain.lists)
        },
        PlaceRequest {
            encode: Some(EncodeItems::default()),
            ..PlaceRequest::of(MODEL, plain.lists)
        },
    ];
    for request in &more {
        assert_eq!(Made::new(&engine, request, &[]).plain(), None);
    }
    assert_eq!(
        Made::new(&engine, &plain, &tried).plain(),
        None,
        "a missed target"
    );
    // A decode asked again is a pair's word: over one list it asks nothing.
    let again = PlaceRequest {
        decode_again: true,
        ..plain
    };
    assert_eq!(Made::new(&engine, &again, &[]).plain(), said);
}

/// A request states the later decode where its caller asks the decode
/// again and the document picks the decode after the prefill. Under a
/// document that picks both sides at once it states none, and so does a
/// caller that asks no decode again.
///
/// The word is a pair's, over its two lists or an encode request's three.
/// Over one list, and over none, a request that carries it states no
/// later decode: the request made is as plain as its plain test says.
#[test]
fn the_later_decode_is_stated_where_the_document_picks_it_so() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    let at_once = engine_of(&document);
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let after = engine_of(&document);
    let members = PublishedList::new(pool(2));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let pair = pair_of(&members, &classes);
    let later = |engine: &InProcess, lists: WorkerLists<'_>, decode_again| {
        let request = PlaceRequest {
            decode_again,
            ..PlaceRequest::of(MODEL, lists)
        };
        let made = Made::new(engine, &request, &[]);
        made.make(engine.standing_spec()).constraints.decode_later
    };
    for lists in [WorkerLists::Pair(pair), three_of(pair)] {
        assert!(later(&after, lists, true));
        assert!(!later(&after, lists, false));
        assert!(!later(&at_once, lists, true));
    }
    for lists in [WorkerLists::Single(members.lent()), WorkerLists::None] {
        let request = PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, lists)
        };
        let made = Made::new(&after, &request, &[]);
        let stated = made.make(after.standing_spec());
        assert!(!stated.constraints.decode_later);
        assert!(made.plain().is_some());
        assert_eq!(stated.plain(), made.plain());
    }
}

/// An HTTP server's header map is read as it stands. The map keeps its
/// names lowercase, as a document writes them, whatever case the client
/// sent. A value that is not text is handed as its bytes, and the source
/// that reads it passes it over: the next source holds the routing value.
#[test]
fn a_header_map_hands_the_documents_sources_their_values() {
    let engine = keyed();
    let keys = |headers: &HeaderMap| {
        let request = PlaceRequest {
            headers,
            request_id: Some("request-1"),
            ..PlaceRequest::of(MODEL, WorkerLists::None)
        };
        let made = Made::new(&engine, &request, &[]);
        let key = made.make(engine.standing_spec()).facts.session_key;
        (key.hash, key.plain.is_some())
    };
    let name = HeaderName::from_bytes(b"X-Conversation").expect("a header's name");
    let mut headers = HeaderMap::new();
    headers.insert(name.clone(), HeaderValue::from_static("talk-7"));
    assert_eq!(headers.header("x-conversation"), Some(&b"talk-7"[..]));
    assert_eq!(headers.header("x-client"), None);
    assert_eq!(keys(&headers), (fingerprint(b"talk-7"), false));

    let bytes = HeaderValue::from_bytes(&[0xFF, 0xFE]).expect("a header's value");
    headers.insert(name, bytes);
    assert_eq!(headers.header("x-conversation"), Some(&[0xFF, 0xFE][..]));
    assert_eq!(keys(&headers), (fingerprint(b"request-1"), true));
}

/// A room rule that names the request's output limit reserves room for the
/// answer its caller states through the one call: a worker whose requests
/// in flight leave no room for the answer is passed over while another
/// has room. No classic policy reads an output limit: the rule is a
/// document's own.
#[test]
fn the_output_limit_a_caller_states_reserves_room_under_the_documents_rule() {
    let document = r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{
        "filter":["healthy",{"has-room":{
            "used":{"candidate.in_flight":100,"request.max_output_tokens":1},
            "capacity":1000}}],
        "pick":"passthrough"}}}"#;
    let document: PolicySpec = serde_json::from_str(document).expect("a well-formed document");
    let engine = hosted(engine_of(&document));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    // Five requests in flight on the list's first worker: half its capacity.
    for _ in 0..5 {
        list[0].increment_load();
    }
    let placed_at = |max_output_tokens| {
        let request = PlaceRequest {
            max_output_tokens,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placement = engine.place(&request).expect("a placement");
        list.iter()
            .position(|worker| Arc::ptr_eq(worker, &placement.worker))
    };
    // An answer of 500 tokens fits beside the five requests, so the first
    // worker of the list takes the request. A token more does not fit.
    assert_eq!(placed_at(500), Some(0));
    assert_eq!(placed_at(501), Some(1));
}

/// A filter of the host's own. It keeps a worker for a request that is
/// shown `blue` in the second header of the document's view and nothing
/// else, and for no other request.
#[derive(Debug)]
struct BlueRoute;

impl Filter for BlueRoute {
    fn name(&self) -> &'static str {
        "blue-route"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        let blue = Header {
            name: 1,
            value: "blue",
        };
        facts.headers == [blue]
    }
}

/// A stage is shown the headers of the document's view that the request
/// carries, each named by its place in that list: a filter that asks for
/// one of them keeps its workers for a request that carries it. A request
/// that carries another value, the value under another name, no header or
/// a value that is not text is left no worker by that filter.
#[test]
fn a_stage_is_shown_the_headers_the_documents_view_lists() {
    let mut stages = StageRegistry::builtin();
    let added = stages.filter("blue-route", || Box::new(BlueRoute));
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.generation = 1;
    document.view.headers = vec!["x-tier".to_owned(), "x-route".to_owned()];
    let profile = document.profiles.get_mut("random");
    let profile = profile.expect("the preset's profile");
    profile
        .filter
        .push(FilterSpec::Name("blue-route".to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the filter")));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    // The stage that left the request no worker; none where it is placed.
    let emptied_by = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        match engine.place(&request) {
            Ok(_) => None,
            Err(Unplaced::Shed { emptied_by, .. }) => Some(emptied_by),
            Err(other) => panic!("{other:?}"),
        }
    };
    assert_eq!(emptied_by(&[("x-route", "blue")]), None);
    assert_eq!(emptied_by(&[("x-route", "green")]), Some("blue-route"));
    assert_eq!(emptied_by(&[("x-tier", "blue")]), Some("blue-route"));
    assert_eq!(emptied_by(&()), Some("blue-route"));
    let mut headers = HeaderMap::new();
    let bytes = HeaderValue::from_bytes(b"blue\xFF").expect("a header's value");
    headers.insert(HeaderName::from_static("x-route"), bytes);
    assert_eq!(emptied_by(&headers), Some("blue-route"));
}

/// A view lists 256 headers at most, and the last one's place is 255: a
/// request that carries it is shown it under that place.
#[test]
fn the_last_of_256_view_headers_is_named_by_its_place() {
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    document.view.headers = (0..256).map(|at| format!("x-header-{at}")).collect();
    let engine = engine_of(&document);
    let headers = [("x-header-255", "last")];
    let request = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let made = Made::new(&engine, &request, &[]);
    let last = Header {
        name: 255,
        value: "last",
    };
    assert_eq!(made.make(engine.standing_spec()).facts.headers, [last]);
}

/// A later document whose view differs in its header list alone keeps the
/// profile's stages, however the list changed. A kept stage that reads a
/// header by its place reads that place of the later list. A stage built
/// anew from the later document reads the same.
#[test]
fn a_stage_kept_across_a_new_header_list_reads_its_place_of_that_list() {
    /// Draws among the workers the filters keep, and notes the value the
    /// request is shown at the view's first place: the empty text for none.
    #[derive(Debug)]
    struct ReadsFirst(Arc<Mutex<Vec<String>>>);

    impl Lifecycle for ReadsFirst {}

    impl Picker for ReadsFirst {
        fn name(&self) -> &'static str {
            "reads-first"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            let first = facts.headers.iter().find(|header| header.name == 0);
            let value = first.map_or("", |header| header.value);
            self.0.lock().push(value.to_owned());
            push_drawn(candidates, kept, out);
        }
    }

    let read = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&read);
    let mut stages = StageRegistry::builtin();
    let make = move || -> Box<dyn Picker> { Box::new(ReadsFirst(Arc::clone(&noted))) };
    stages.picker("reads-first", make).expect("a free name");
    let listing = |generation: u64, headers: &[&str]| {
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "reads-first".into();
        document.view.headers = headers.iter().map(|name| (*name).to_owned()).collect();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        spec.expect("the stages hold the picker")
    };
    let members = PublishedList::new(pool(2));
    // What the picker in force notes of a request that carries two headers.
    let asked = |engine: &'static InProcess| {
        let headers = [("x-a", "one"), ("x-b", "two")];
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        assert!(engine.place(&request).is_ok(), "a placement");
        std::mem::take(&mut *read.lock())
    };
    let engine = hosted(InProcess::new(listing(1, &["x-a"])));
    assert_eq!(asked(engine), ["one"]);
    // The later list, and the value at its first place. The list grew at
    // its end, shrank from its end, took another header first, and took a
    // first header the request does not carry.
    let later: [(&[&str], &str); 4] = [
        (&["x-a", "x-b"], "one"),
        (&["x-a"], "one"),
        (&["x-b", "x-a"], "two"),
        (&["x-c", "x-a"], ""),
    ];
    for (generation, (headers, value)) in (2..).zip(later) {
        let stood = engine.standing_spec().engine();
        let applied = engine.apply(listing(generation, headers));
        applied.expect("a number above the standing one");
        let stands = engine.standing_spec().engine();
        let kept = Arc::ptr_eq(&stands.profiles()[0], &stood.profiles()[0]);
        assert!(kept, "{headers:?}: the profile stands");
        assert_eq!(asked(engine), [value], "{headers:?}: the stage kept");
        let anew = hosted(InProcess::new(listing(generation, headers)));
        assert_eq!(asked(anew), [value], "{headers:?}: a stage built anew");
    }
}

/// A later document that lists another header ahead of the sessions rule's
/// keeps the profile's stages, and the rule reads its key from its own
/// header at its new place. A pinned key's request follows its pin, though
/// the header now first carries a text no key holds. A request that
/// carries a pinned key's text under that first header, and a new key
/// under the rule's own, is a new key's.
#[test]
fn a_sessions_rule_reads_its_header_behind_a_new_header_list() {
    let listing = |generation: u64, headers: &[&str]| {
        let mut document = PolicySpec::preset("manual").expect("a built-in preset");
        document.generation = generation;
        document.view.headers = headers.iter().map(|name| (*name).to_owned()).collect();
        AppliedSpec::validate(&document).expect("the document validates")
    };
    let engine = hosted(InProcess::new(listing(1, &[MANUAL_SESSION_HEADER])));
    let members = PublishedList::new(pool(16));
    let went = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let mut placement = engine.place(&request).expect("a placement");
        placement.sent(Side::Single);
        placement.id()
    };
    let pins = || engine.standing_spec().engine().sessions().len();
    let keys = ["talk-1", "talk-2", "talk-3", "talk-4"];
    let homes = keys.map(|key| went(&[(MANUAL_SESSION_HEADER, key)]));
    assert_eq!(pins(), keys.len());

    let stood = engine.standing_spec().engine();
    let applied = engine.apply(listing(2, &["x-first", MANUAL_SESSION_HEADER]));
    applied.expect("a number above the standing one");
    let stands = engine.standing_spec().engine();
    let kept = Arc::ptr_eq(&stands.profiles()[0], &stood.profiles()[0]);
    assert!(kept, "the profile stands");
    for (key, home) in keys.into_iter().zip(homes) {
        let headers = [("x-first", "talk-none"), (MANUAL_SESSION_HEADER, key)];
        assert_eq!(went(&headers), home, "{key} follows its pin");
    }
    assert_eq!(pins(), keys.len(), "no new key");
    let headers = [("x-first", "talk-1"), (MANUAL_SESSION_HEADER, "talk-new")];
    went(&headers);
    assert_eq!(pins(), keys.len() + 1, "a new key");
}

/// The profile's picker receives the request's stated budget in microseconds.
/// The request gets a leased pick.
#[test]
fn the_picker_is_told_the_budget_a_request_states() {
    const NAME: &str = "record-budget";

    #[derive(Debug)]
    struct BudgetPicker(Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for BudgetPicker {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }
    }

    impl Picker for BudgetPicker {
        fn name(&self) -> &'static str {
            NAME
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            self.0.lock().push(facts.budget_micros);
            push_drawn(candidates, kept, out);
        }
    }

    let budgets = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&budgets);
    let mut stages = StageRegistry::builtin();
    stages
        .picker(NAME, move || Box::new(BudgetPicker(Arc::clone(&observed))))
        .expect("a free picker name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    let profile = document
        .profiles
        .get_mut("random")
        .expect("the random profile");
    profile.pick = NAME.into();
    let spec = AppliedSpec::validate_in(&document, &stages, &[]).expect("a valid document");
    let engine = hosted(InProcess::new(spec));

    let members = PublishedList::new(pool(3));
    let request = PlaceRequest {
        budget_micros: 5_000,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(placement.owes_events(), "the pick has a lease");
    assert_eq!(budgets.lock().as_slice(), [5_000]);
}

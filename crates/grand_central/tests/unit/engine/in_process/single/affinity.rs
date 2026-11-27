//! The cases of a key's pin and of the tree a host lends: the count a capped
//! sessions rule reads, what a lent tree holds of a followed pin's prompt, a
//! request with no key, a pinned key held to places, the entry a decision
//! asks for, and a worker's sessions when it is said gone and back.

use super::*;

/// The count a capped sessions rule reads stands on a worker's object
/// under the name the request's maker states. A maker that says
/// nothing is of a host whose own guard counts, and the count is read
/// under the key's text. A maker of a host that counts by the
/// decisions it holds on file says the session's number, and the count
/// is read under that number's text, though the request carries its
/// key in the clear. A count under the other name is no count of the
/// key: its next request stays on its worker.
#[test]
fn a_keys_count_is_read_under_the_name_its_requests_maker_states() {
    /// The maker of a host that counts by the decisions it holds.
    struct ByNumber<'a>(Asks<'a>);

    impl RequestMaker for ByNumber<'_> {
        fn model(&self) -> &str {
            self.0.model()
        }

        fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
            self.0.make(spec)
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.0.workers()
        }

        fn session_counted_under(&self) -> SessionCountedUnder {
            SessionCountedUnder::SessionNumber
        }
    }

    /// The key's first request pins it on a worker, two requests of
    /// the key are counted there under `name`, and the key's next
    /// request is decided: what its rule said of the pin, and whether
    /// the request left the pin's worker.
    fn next(
        engine: &InProcess,
        list: &[Arc<dyn Worker>],
        maker: &impl RequestMaker,
        name: &str,
    ) -> (SessionVerdict, bool) {
        let now = Instant::now();
        let mut guard =
            DecisionGuard::decide_under(engine, GatewayId::LOCAL, maker, now).expect("a decision");
        let owner = guard.take(0).expect("the first target");
        guard.dispatched(now);
        for _ in 0..2 {
            list[owner.position as usize].increment_load();
            list[owner.position as usize].increment_routing_key_load(name);
        }
        let decided = engine.decide_under(GatewayId::LOCAL, maker, now);
        let (decided, _) = decided.expect("an answer").whole().expect("a whole one");
        let pick = *decided.decision.pick().expect("a pick");
        let went = pick.targets.first().expect("a target");
        (pick.session_verdict, went.id != owner.id)
    }

    let keyed = || {
        let mut keyed = request();
        keyed.facts.session_key = Identity::plain("talk-1");
        keyed
    };
    let mut digits = [0; Session::TEXT_LEN];
    let number = keyed().facts.session().text(&mut digits).to_owned();
    let decided = |by_number: bool, name: &str| {
        // The stock rule with an in-flight cap of two, which places a
        // key on the worker with the fewest requests.
        let mut document = PolicySpec::preset("manual").expect("a built-in preset");
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the stock rule");
        (rule.inflight_cap, rule.assignment) = (2, Assignment::MinLoad);
        document.generation = 1;
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let (engine, list) = (InProcess::new(spec), pool(4));
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(keyed(), Some(&members));
        if by_number {
            next(&engine, &list, &ByNumber(asks), name)
        } else {
            next(&engine, &list, &asks, name)
        }
    };
    let (handed_on, kept) = (
        (SessionVerdict::Respill, true),
        (SessionVerdict::Pinned, false),
    );
    assert_eq!(decided(false, "talk-1"), handed_on, "a guard's count");
    assert_eq!(decided(false, &number), kept, "no count of a guard's");
    assert_eq!(
        decided(true, &number),
        handed_on,
        "the held decisions' count"
    );
    assert_eq!(decided(true, "talk-1"), kept, "no count of a held decision");
}

/// [`Asks`] of a host that names the request's entry in its token tree
/// and says whether it sends to a target of the pick: `to` for each.
/// `page` is the tree's page it was last asked the entry by, zero while
/// it was not asked.
struct Sends<'a> {
    asks: Asks<'a>,
    to: bool,
    page: Cell<usize>,
}

impl RequestMaker for Sends<'_> {
    fn model(&self) -> &str {
        self.asks.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.asks.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        self.asks.plain()
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.asks.workers()
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.asks.loads()
    }

    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        self.page.set(page);
        self.asks.token_key(page)
    }

    fn text_key(&self) -> Option<TextKey<'_>> {
        self.asks.text_key()
    }

    fn sends_to(&self, _: &Target) -> bool {
        self.to
    }
}

/// A pin followed under a host that lends its token tree and names each
/// request's entry there. Where the rule keeps the request from the
/// profile its pin returns ahead of the lookup: the host is not asked
/// for the entry and the tree holds no token of the prompt. Where the
/// request is the profile's own the host is asked once, by the tree's
/// own page, and the whole decision does not run. The tree then holds
/// the whole prompt for the pin's worker alone, and none of a prompt
/// whose host says it does not send to that worker: the hand-over asks
/// the request's own maker which target its caller takes.
#[test]
fn a_followed_pin_is_stored_in_a_lent_tree_by_the_rules_word() {
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let (engine, tree) = lending(default_under_sessions(kept));
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
        // One request of `tokens`, which its host names as its entry,
        // sent to its first target and served there. Its host sends to
        // a target of its pick where `to`. The place it went to, its
        // pick, how often its host was asked for the entry, and by
        // which page.
        let sent = |tokens: &[u32], key: u64, seed: u64, to: bool| {
            let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
            let mut asked = request();
            asked.facts.seed = seed;
            asked.facts.session_key = Identity::hashed(key);
            hashes.write_to(&mut asked.facts);
            let asks = Asks {
                key: Some(tokens),
                ..Asks::of(asked, Some(&members))
            };
            let host = Sends {
                asks,
                to,
                page: Cell::new(0),
            };
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &host, now)
                .expect("a decision");
            let pick = *guard.decision().pick().expect("a pick");
            let target = guard.take(0).expect("the first target");
            guard.dispatched(now);
            guard.completed(
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            (
                target.position,
                pick,
                (host.asks.keyed.get(), host.page.get()),
            )
        };
        // The tokens of a prompt the tree holds, and the workers that
        // hold them.
        let stored = |tokens: &[u32]| {
            let found = tree.match_prefix_with_counts(tokens);
            (found.matched_token_count, found.matched_tenants)
        };
        let own = kept == RulePlaced::Told;
        for key in 1..=6_u32 {
            let seed = u64::from(key);
            let (pin, _, entry) = sent(&prompt(key * 100_000), seed, seed, true);
            assert_eq!(entry, (1, 16), "{kept:?}: key {key}: a new key's entry");
            let at = vec![TenantId::from(list[pin as usize].url())];
            // The key's pin, of a prompt nobody was sent.
            let fresh = prompt(key * 100_000 + 50_000);
            let (went, pick, entry) = sent(&fresh, seed, 1, true);
            assert_eq!(
                (went, pick.placed_by()),
                (pin, PlacedBy::Rule),
                "{kept:?}: key {key}"
            );
            let whole = !pick.trace.drew_first();
            assert!(!whole, "{kept:?}: key {key}: the whole decision ran");
            if own {
                assert_eq!(
                    entry,
                    (1, 16),
                    "key {key}: the pin's entry, by the tree's page"
                );
                assert_eq!(stored(&fresh), (1_024, at), "key {key}: the pin's prompt");
            } else {
                assert_eq!(entry, (0, 0), "key {key}: a kept pin's entry");
                assert_eq!(stored(&fresh), (0, Vec::new()), "key {key}: a kept prompt");
            }
            // The key's pin again, where the host answers that its caller
            // sends to no target of the pick: the tree takes none of the
            // prompt. The helper still dispatches on the pin's worker, so
            // this holds what that answer does to the tree and no more.
            let refused = prompt(key * 100_000 + 70_000);
            let (went, _, entry) = sent(&refused, seed, 2, false);
            assert_eq!(went, pin, "{kept:?}: key {key}");
            assert_eq!(entry.0, u32::from(own), "{kept:?}: key {key}: the entry");
            assert_eq!(stored(&refused), (0, Vec::new()), "{kept:?}: key {key}");
        }
    }
}

/// A request with no key is never its sessions rule's, also where the
/// profile's picker leaves the placement to the binding: the binding's
/// placing assignment places such a request, under a rule and under
/// none, and its pick keeps the stages' word. The rule's word is for
/// its key alone, which the same assignment places. So is the lease:
/// the preset's stages take no word of one, the session table follows
/// a key's dispatch alone, and a request with no key is issued none.
#[test]
fn a_request_with_no_key_is_never_its_sessions_rules() {
    for ruled in [true, false] {
        let engine = {
            let mut document = PolicySpec::preset("manual").expect("a built-in preset");
            if !ruled {
                document.bindings[0].sessions = None;
            }
            document.generation = 1;
            let applied = AppliedSpec::validate(&document).expect("the document validates");
            InProcess::new(applied)
        };
        let list = pool(8);
        let members = PublishedList::new(Arc::clone(&list));
        let placed_by = |key: u64| {
            let mut asked = request();
            asked.facts.seed = 7;
            asked.facts.session_key = Identity::hashed(key);
            let asks = Asks::of(asked, Some(&members));
            let now = Instant::now();
            let mut guard = DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now)
                .expect("a decision");
            // A guard that holds no lease hands no target by `take`:
            // its host reads the decision, as under a document that
            // takes no word.
            let leased = ruled && key != 0;
            assert_eq!(guard.lease().is_some(), leased, "key {key}");
            assert_eq!(guard.take(0).is_some(), leased, "key {key}");
            guard.decision().pick().expect("a pick").placed_by()
        };
        assert_eq!(placed_by(0), PlacedBy::Stages, "no key, ruled: {ruled}");
        // A key is the rule's where a rule stands, and no one's else.
        let keyed = if ruled {
            PlacedBy::Rule
        } else {
            PlacedBy::Stages
        };
        assert_eq!(placed_by(9), keyed, "a key, ruled: {ruled}");
    }
}

/// A pin followed under a host that lends its string tree. The
/// hand-over of the pin's made answer asks the request's maker for its
/// text entry, once, and the tree then holds the whole text of a prompt
/// nobody was sent, for the pin's worker alone: as the token tree does
/// ([`a_followed_pin_is_stored_in_a_lent_tree_by_the_rules_word`]).
#[test]
fn a_followed_pin_of_text_is_stored_in_a_lent_string_tree() {
    let (engine, tree) = lending_text(default_under_sessions(RulePlaced::Told));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let prompt = |first: u32| format!("{first:08}").repeat(512);
    // One request of `text` under the key `key`, sent to its first
    // target and served there. The place it went to, its pick, and how
    // often its host was asked for the text entry.
    let sent = |text: &str, key: u64, seed: u64| {
        let hashes = engine.prompt_hashes(MODEL, None, Some(text), HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = seed;
        asked.facts.session_key = Identity::hashed(key);
        hashes.write_to(&mut asked.facts);
        let asks = Asks {
            text: Some(text),
            ..Asks::of(asked, Some(&members))
        };
        let now = Instant::now();
        let mut guard =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let pick = *guard.decision().pick().expect("a pick");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        guard.completed(
            &Outcome {
                side: Side::Single,
                served: target.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        (target.position, pick, asks.worded.get())
    };
    for key in 1..=6_u32 {
        let seed = u64::from(key);
        let (pin, _, entry) = sent(&prompt(key * 100_000), seed, seed);
        assert_eq!(entry, 1, "key {key}: a new key's entry");
        // The key's pin, of a prompt nobody was sent.
        let fresh = prompt(key * 100_000 + 50_000);
        let (went, pick, entry) = sent(&fresh, seed, 1);
        assert_eq!((went, pick.placed_by()), (pin, PlacedBy::Rule), "key {key}");
        assert_eq!(entry, 1, "key {key}: the pin's entry");
        let found = tree.match_prefix_with_counts(&fresh);
        let at = vec![TenantId::from(list[pin as usize].url())];
        let stored = (found.matched_char_count, found.matched_tenants);
        assert_eq!(stored, (fresh.len(), at), "key {key}: the pin's prompt");
    }
}

/// A request its host holds to places is never the keyed entry's: the
/// entry answers nothing for it, so a key's pin outside the places is
/// not followed ahead of the lookup, and the hand-over of a made answer
/// never sees a request held to places, which is why it hands the
/// naming none (`overlap::decide_behind`). The whole decision places
/// the request among its places, and the lent tree holds its prompt
/// for the worker picked there.
#[test]
fn a_pinned_key_held_to_places_is_decided_among_them() {
    let (engine, tree) = lending(default_under_sessions(RulePlaced::Told));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let prompt = |first: u32| (first..first + 1_024).collect::<Vec<u32>>();
    // One request of `tokens` under the key `key`, held to `places`
    // where its host states any, sent to its first target and served
    // there. The place it went to.
    let sent = |tokens: &[u32], key: u64, places: Option<&[u32]>| {
        let hashes = engine.prompt_hashes(MODEL, Some(tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        asked.facts.seed = key;
        asked.facts.session_key = Identity::hashed(key);
        asked.constraints.places = places;
        hashes.write_to(&mut asked.facts);
        let asks = Asks {
            key: Some(tokens),
            ..Asks::of(asked, Some(&members))
        };
        let now = Instant::now();
        let mut guard =
            DecisionGuard::decide_under(&engine, GatewayId::LOCAL, &asks, now).expect("a decision");
        let target = guard.take(0).expect("the first target");
        guard.dispatched(now);
        guard.completed(
            &Outcome {
                side: Side::Single,
                served: target.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        target.position
    };
    for key in 1..=6_u32 {
        let pin = sent(&prompt(key * 100_000), u64::from(key), None);
        // The key again, held to two places that leave its pin out.
        let places: Vec<u32> = (0..8).filter(|&at| at != pin).take(2).collect();
        let fresh = prompt(key * 100_000 + 50_000);
        let went = sent(&fresh, u64::from(key), Some(&places));
        assert!(places.contains(&went), "key {key}: {went} of {places:?}");
        let found = tree.match_prefix_with_counts(&fresh);
        let at = vec![TenantId::from(list[went as usize].url())];
        let stored = (found.matched_token_count, found.matched_tenants);
        assert_eq!(stored, (1_024, at), "key {key}: the prompt's worker");
    }
}

/// The entry is asked for where the host lends a tree of sent prefixes
/// and the profile reads one, and nowhere else. A place one pick decides
/// is answered ahead of the whole pass and never asks. A whole pass
/// whose profile reads no record of sent prefixes never asks either, and
/// neither does a decision whose host lends no tree.
#[test]
fn a_decision_that_is_lent_no_tree_asks_for_no_entry() {
    let tokens: Vec<u32> = (0..256).collect();
    let asked_of = |name: &str, lends: bool| {
        let engine = if lends {
            lent(name, Also::Nothing).0
        } else {
            preset(name)
        };
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks {
            key: Some(&tokens),
            ..Asks::of(request(), Some(&members))
        };
        let alone = matches!(answered(&engine, &asks), Decided::Alone(_));
        (alone, asks.keyed.get())
    };
    assert_eq!(asked_of("round_robin", true), (true, 0));
    assert_eq!(asked_of("least_load", true), (false, 0));
    assert_eq!(asked_of("cache_aware", true), (false, 1));
    assert_eq!(asked_of("cache_aware", false), (false, 0));
}

/// A worker said gone and back at its address serves its sessions
/// again, with no join said: the session table is not told a leave, as
/// the classic policies' map is not. A key is pinned on a worker that
/// leaves, and is served by the worker that stayed for twice the time
/// the table holds an absent worker's pins, a quarter of that time
/// apart, by the engine's own number. Back in the list, its first
/// worker serves it again.
#[test]
fn a_worker_said_gone_and_back_at_its_address_serves_its_sessions_again() {
    let (engine, both) = (preset("manual"), pool(2));
    let start = Instant::now();
    let absence = Duration::from_micros(engine.spec().engine().pins_absence_micros());
    // The key's request over `list` in quarter `turn`: its worker.
    let sent = |list: &Arc<[Arc<dyn Worker>]>, turn: u32| {
        let members = PublishedList::new(Arc::clone(list));
        let mut keyed = request();
        keyed.facts.session_key = Identity::hashed(1);
        let asks = Asks::of(keyed, Some(&members));
        served(&engine, &asks, start + absence / 4 * turn).id
    };
    let first = sent(&both, 0);
    let at = usize::from(worker_id(both[1].as_ref()) == first);
    let (gone, other) = (&both[at], Arc::from([Arc::clone(&both[1 - at])]));
    engine.forget(gone.url());
    for turn in 1..=8 {
        assert_ne!(sent(&other, turn), first);
    }
    assert_eq!(sent(&both, 9), first, "it has its key again");
}

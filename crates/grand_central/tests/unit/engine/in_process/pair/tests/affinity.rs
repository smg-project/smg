//! Where a prompt and a key go again: what each side reads and writes of
//! who was sent a prompt and who holds it, and each side's key under a
//! sessions rule: its pin, its claim, its count, the date of its life
//! and the word its side is told by.

use super::*;

/// A pair's first pick states what its prefill holds of the request
/// ([`Pick::overlap`](crate::engine::Pick::overlap)), where the
/// request's own profile reads who holds the prompt at the prefill
/// phase: with no prompt sent before, a miss of the request's blocks.
/// The decode asked again states nothing: its prefill ran, and no
/// column of the prefills is read for it. Neither does a pair whose
/// prefill side reads no holder, whatever its decode side reads.
#[test]
fn a_pairs_first_pick_states_what_its_prefill_holds() {
    use crate::engine::decision::Overlap;

    let under = |prefill: &str, decode: &str| {
        let document = PolicySpec::pair_presets(prefill, decode).expect("the pair presets");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        Host::under((spec, Arc::default()), 2, 2)
    };
    let stated = |decision: &Decision| decision.pick().expect("a pick").overlap();

    let host = under("cache_aware", "cache_aware");
    let first = host.decide(&OPEN, 3, None);
    let miss = stated(&first);
    assert_ne!(miss.of, 0, "a miss is stated of the request's blocks");
    let nothing_held = Overlap {
        of: miss.of,
        ..Overlap::NONE
    };
    assert_eq!(miss, nothing_held);
    let held = first_target(&first);
    let again = host.decide(&OPEN, 4, Some(held.id));
    assert_eq!(stated(&again), Overlap::NONE, "the decode asked again");

    let decode_alone = under("round_robin", "cache_aware");
    let first = decode_alone.decide(&OPEN, 3, None);
    assert_eq!(
        stated(&first),
        Overlap::NONE,
        "a prefill side that reads none"
    );
}

/// Under `cache_aware` on both sides, a prompt sent to a pair goes to
/// that pair again under any draw: each side reads its own list's
/// column of who was sent the prompt, and its pick is written behind
/// it. Prompts nobody was sent are placed by their draws, on more than
/// one pair.
#[test]
fn a_prompt_sent_to_a_pair_goes_to_that_pair_again() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    for seed in 2..34 {
        let again = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(again, first, "seed {seed}");
    }
    let fresh: BTreeSet<(usize, usize)> = (0..32_u32)
        .map(|at| {
            let unsent = prompt(&host, 100_000 + 1_000 * at);
            host.picked(&decide_prompt(&host, &unsent, u64::from(at), None))
        })
        .collect();
    assert!(fresh.len() > 1, "{fresh:?}");
}

/// The decode asked again behind a held prefill reads the decodes'
/// column: it goes to the decode its prompt was sent to, under any
/// draw.
#[test]
fn the_decode_asked_again_goes_to_the_decode_its_prompt_was_sent_to() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let (prefill, decode) = host.picked(&decide_prompt(&host, &sent, 1, None));
    let held = prefill_alone(worker_id(host.prefills[prefill].as_ref()));
    for seed in 2..34 {
        let again = decide_prompt(&host, &sent, seed, Some(&held));
        assert_eq!(host.picked(&again), (prefill, decode), "seed {seed}");
    }
}

/// The decode asked again writes no prefill's record. Behind a held
/// prefill that is not the one its prompt was sent to, it goes to the
/// prompt's decode, and the prompt's next first decisions still go to
/// the pair it was sent to, under any draw.
#[test]
fn the_decode_asked_again_writes_no_prefill() {
    let host = host_under("cache_aware", "cache_aware");
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    let other = (first.0 + 1) % host.prefills.len();
    let held = prefill_alone(worker_id(host.prefills[other].as_ref()));
    for seed in 2..10 {
        let again = decide_prompt(&host, &sent, seed, Some(&held));
        assert_eq!(host.picked(&again), (other, first.1), "seed {seed}");
    }
    for seed in 10..42 {
        let next = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(next, first, "seed {seed}");
    }
}

/// A binding of the decode phase alone listed ahead of the pair's takes
/// a decode request asked alone and leaves the pair its decode side.
/// The decodes' list keeps the record that side reads where the first
/// binding reads none, so a prompt's decode stays there too.
#[test]
fn a_decode_binding_listed_ahead_leaves_the_pair_its_record() {
    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("round_robin").expect("a preset");
    let mut ahead = stock.bindings[0].clone();
    ahead.matcher.phase = vec![Phase::Decode];
    ahead.profile = Some("ahead".into());
    document.bindings.insert(0, ahead);
    let turns = stock.profiles.into_values().next();
    let turns = turns.expect("the preset's profile");
    document.profiles.insert("ahead".into(), turns);
    the_decode_stays_and_the_prefills_turn(&host_of(document, 4, 4));
}

/// A host's indexes of cache events: one chain index for every model,
/// of blocks of the sixteen tokens [`prompt`] hashes by.
struct Indexes(Arc<ChainIndex>);

impl overlap::CacheIndexes for Indexes {
    fn index(&self, _: &str) -> Option<Arc<dyn overlap::CacheIndex>> {
        Some(Arc::clone(&self.0) as Arc<dyn overlap::CacheIndex>)
    }

    fn block_size(&self, _: &str) -> Option<u32> {
        Some(16)
    }
}

/// The document of [`decodes_with_blocks`]: prefills that take turns
/// and decodes under `cache_aware`, behind a binding of the decode
/// phase alone listed ahead of the pair's, `cache_aware` too. Where
/// `pair_alone`, the pair's decode profile reads sent prefixes alone
/// (`source: predictor`) and the binding ahead reads cache events too.
/// Where not, the two stand the other way round.
fn decode_binding_ahead(pair_alone: bool) -> PolicySpec {
    use crate::document::spec::{AffinitySource, ScoreSpec, ScoreStage};

    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("cache_aware").expect("a preset");
    let mut ahead = stock.bindings[0].clone();
    ahead.matcher.phase = vec![Phase::Decode];
    ahead.profile = Some("ahead".into());
    document.bindings.insert(0, ahead);
    let aware = stock.profiles.into_values().next();
    let aware = aware.expect("the preset's profile");
    document.profiles.insert("ahead".into(), aware);
    let alone = if pair_alone { "decode" } else { "ahead" };
    let profile = document.profiles.get_mut(alone).expect("the profile");
    for stage in &mut profile.score {
        if let ScoreSpec::Params(ScoreStage::PrefixAffinity(prefix)) = stage {
            prefix.source = Some(AffinitySource::Predictor);
        }
    }
    document
}

/// A host of four prefills and four decodes under
/// [`decode_binding_ahead`]. The host's index of cache events holds a
/// block of another prompt for every decode. The decodes' pool keeps
/// the word of the binding ahead, which is never the pair's decode
/// profile's here.
fn decodes_with_blocks(pair_alone: bool) -> Host {
    let host = host_of(decode_binding_ahead(pair_alone), 4, 4);
    let spec = host.engine.spec();
    let engine = spec.engine();
    assert_eq!(
        engine.side_reads_sent_alone(MODEL, Phase::Decode),
        pair_alone
    );
    let pools_word = engine.pool_record_for(MODEL, Phase::Decode);
    assert_eq!(pools_word, Some(!pair_alone), "the binding ahead's");
    let index = Arc::new(ChainIndex::new());
    for (at, decode) in host.decodes.iter().enumerate() {
        let number = index.intern_worker(decode.url()).expect("a number");
        let content_hash = ContentHash(at as u64 + 1);
        let blocks: Vec<StoredBlock> = request_prefix_hashes(&[content_hash])
            .into_iter()
            .map(|seq_hash| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect();
        let stored = index.apply_stored(number, &blocks, None, &mut ChainBlockMap::new());
        stored.expect("the block is stored");
    }
    host.engine.affinity().set_indexes(Arc::new(Indexes(index)));
    host
}

/// A pair's decode side reads the record of sent prefixes by its own
/// profile's word, whichever profile its pool keeps the record for.
/// The index of cache events holds a block for every decode, and a
/// binding of the decode phase alone is listed ahead of the pair's.
/// Where the pair's decode profile reads sent prefixes alone, a
/// prompt's decode stays, at the pair's first decision and at the
/// decode asked again: that side is handed what it sent to a worker
/// with a block, and its pick is written there. Where the binding
/// ahead reads alone and the pair's decode profile reads cache events
/// too, that side reads the index for those workers, as before: a
/// prompt the index holds for none of them takes its draws.
#[test]
fn a_pairs_decode_side_reads_the_record_by_its_own_profiles_word() {
    the_decode_stays_and_the_prefills_turn(&decodes_with_blocks(true));

    let host = decodes_with_blocks(false);
    let sent = prompt(&host, 1_000);
    let decodes: BTreeSet<usize> = (1..33)
        .map(|seed| host.picked(&decide_prompt(&host, &sent, seed, None)).1)
        .collect();
    assert!(decodes.len() > 1, "{decodes:?}");
}

/// A document that turns a side's word of the record is read by the
/// pair's next decision, inside the second a list's table stands. The
/// word is read once a table, and the table is of the list's value,
/// which is made under one applied document and read under it alone.
/// Every decision here is made at one instant, so a table that stood
/// across the apply would be read by each of them. Under the first
/// document the pair's decode profile reads cache events too: the index
/// holds a block for every decode, the side reads no record, and a
/// prompt the index holds for no decode takes its draws. The next
/// document's decode profile reads sent prefixes alone: its first
/// decision writes the prompt's decode, and the prompt stays there.
#[test]
fn a_document_that_turns_a_sides_word_is_read_by_the_next_decision() {
    let host = decodes_with_blocks(false);
    let sent = prompt(&host, 1_000);
    let now = Instant::now();
    let decode = |seed| {
        host.picked(&decide_prompt_at(&host, &sent, seed, None, now))
            .1
    };
    let mut before = BTreeSet::new();
    for seed in 1..33 {
        before.insert(decode(seed));
    }
    assert!(before.len() > 1, "{before:?}");

    let mut next = decode_binding_ahead(true);
    next.generation = 4;
    let next = AppliedSpec::validate(&next).expect("a valid document");
    let applied = host.engine.apply(next);
    applied.expect("a number above the standing one");
    let first = decode(33);
    for seed in 34..66 {
        assert_eq!(decode(seed), first, "seed {seed}");
    }
}

/// Each side's column is its own list's, whatever the lists' sizes:
/// over two prefills and four decodes a prompt sent to a pair goes to
/// that pair again under any draw.
#[test]
fn a_prompt_stays_with_its_pair_over_lists_of_two_sizes() {
    let pair = PolicySpec::pair_presets("cache_aware", "cache_aware");
    let host = host_of(pair.expect("the pair presets"), 2, 4);
    let sent = prompt(&host, 1_000);
    let first = host.picked(&decide_prompt(&host, &sent, 1, None));
    for seed in 2..34 {
        let again = host.picked(&decide_prompt(&host, &sent, seed, None));
        assert_eq!(again, first, "seed {seed}");
    }
}

/// The maker of a host that counts a key's requests by the decisions it
/// holds: the count stands under the session's number.
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

/// The key of the capped cases' request, in the clear.
const TALK: &str = "talk-1";

/// A host of `prefills` and `decodes` workers under the pair of the
/// stock sessions presets and a rule with the in-flight cap `cap`,
/// which places a key on the worker with the fewest requests.
fn ruled_host(cap: u16, prefills: usize, decodes: usize) -> Host {
    use crate::document::spec::Assignment;

    let pair = PolicySpec::pair_presets("manual", "manual");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let ruled = rule.as_mut().expect("the stock rule");
    (ruled.inflight_cap, ruled.assignment) = (cap, Assignment::MinLoad);
    document.bindings[0].sessions = rule;
    host_of(document, prefills, decodes)
}

/// [`ruled_host`] under an in-flight cap of two.
fn capped_host(prefills: usize, decodes: usize) -> Host {
    ruled_host(2, prefills, decodes)
}

/// The name each side of [`TALK`]'s request is counted under where the
/// session's number names the count: the prefill side's, then the
/// decode side's.
fn numbers() -> [String; 2] {
    let facts = RequestFacts {
        session_key: Identity::plain(TALK),
        ..facts()
    };
    let mut digits = [0; Session::TEXT_LEN];
    [facts.session(), decode_session(&facts)].map(|side| side.text(&mut digits).to_owned())
}

/// `count` requests in flight on `worker`, each counted under `name`.
fn counted_on(worker: &Arc<dyn Worker>, name: &str, count: usize) {
    for _ in 0..count {
        worker.increment_load();
        worker.increment_routing_key_load(name);
    }
}

/// Under a capped sessions rule each side of a pair reads the count of
/// its key's requests on a worker's object, under the name the
/// request's maker states for that side's session. A maker that says
/// nothing is of a host whose own guard counts: the key's text names
/// both sides. A maker of a host that counts by the decisions it holds
/// says the session's number, and each side keeps a session of its
/// own: the prefills' count stands under the prefill side's number and
/// the decodes' under the decode side's. A side whose pin carries the
/// cap under its own name hands the key on, and a count under any
/// other name is no count of that side's key.
#[test]
fn a_pairs_sides_read_their_keys_count_under_the_name_the_maker_states() {
    /// The key's first pair is placed and both its sides dispatched,
    /// two requests of the key are counted on each side's worker
    /// under that side's name of `names`, the prefills' first, and
    /// the key's next pair is decided: each side's verdict.
    fn next(
        host: &Host,
        maker: &impl RequestMaker,
        names: [&str; 2],
    ) -> (SessionVerdict, DecodeVerdict) {
        let now = Instant::now();
        let mut guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, maker, now)
            .expect("a decision");
        let (prefill, decode) = host.picked(guard.decision());
        guard.dispatched(now);
        guard.dispatched_decode(now);
        counted_on(&host.prefills[prefill], names[0], 2);
        counted_on(&host.decodes[decode], names[1], 2);
        let decided = host.engine.decide_under(GatewayId::LOCAL, maker, now);
        let (decided, _) = decided.expect("an answer").whole().expect("a whole one");
        let pick = *decided.decision.pick().expect("a pick");
        let pair = pick.targets.first().and_then(|target| target.pair);
        (pick.session_verdict, pair.expect("a pair").decode_verdict)
    }

    let numbers = numbers();
    let numbers = [numbers[0].as_str(), numbers[1].as_str()];
    assert_ne!(numbers[0], numbers[1], "each side keeps its own session");
    let classes = every(2, 2);
    let decided = |by_number: bool, names: [&str; 2]| {
        let host = capped_host(2, 2);
        let mut asks = host.asks(&classes, false);
        asks.0.facts.session_key = Identity::plain(TALK);
        if by_number {
            next(&host, &ByNumber(asks), names)
        } else {
            next(&host, &asks, names)
        }
    };
    let (handed_on, kept) = (
        (SessionVerdict::Respill, DecodeVerdict::RESPILL),
        (SessionVerdict::Pinned, DecodeVerdict::PINNED),
    );
    assert_eq!(decided(false, [TALK; 2]), handed_on, "a guard's count");
    assert_eq!(decided(false, numbers), kept, "no count of a guard's");
    assert_eq!(decided(true, numbers), handed_on, "the held decisions'");
    assert_eq!(decided(true, [TALK; 2]), kept, "no held decision's count");
    // Each side under its own session's number: the other side's is no
    // count of its key.
    let crossed = [numbers[1], numbers[0]];
    assert_eq!(decided(true, crossed), kept, "the other side's number");
}

/// A decode asked again under a capped rule reads its key's count as
/// its first decision does, under the name the request's maker states.
/// A host that counts the request on its first decode across the call
/// says so, and its guard counts under the request's key: that
/// decode's record is read with one of the key's requests less. One
/// other request of the key there leaves the request on its decode
/// under a cap of two, and two hand it on. A host that says nothing is
/// read as it counts: the request is one of the key's there. A host
/// that counts by the decisions it holds is read under the decode
/// side's number.
#[test]
fn a_decode_asked_again_reads_its_keys_other_requests_on_its_first_decode() {
    /// A pair's request is decided with its decode sent later, and
    /// its prefill dispatched. `others` requests of the key stand on
    /// the decode it chose, under `name`, and the host's guard counts
    /// the request itself there under the key's text. The decode is
    /// asked again: whether it stands, and the side's verdict.
    fn again(by_number: bool, counted: bool, name: &str, others: usize) -> (bool, DecodeVerdict) {
        let (host, classes) = (capped_host(1, 3), every(1, 3));
        let mut asks = host.asks(&classes, true);
        asks.0.facts.session_key = Identity::plain(TALK);
        asks.0.constraints.counted_on_decode = counted;
        let maker = ByNumber(asks);
        let now = Instant::now();
        let decided = if by_number {
            DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &maker, now)
        } else {
            DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &maker.0, now)
        };
        let mut guard = decided.expect("a decision");
        let chosen = host.picked(guard.decision()).1;
        guard.dispatched(now);
        counted_on(&host.decodes[chosen], TALK, 1);
        counted_on(&host.decodes[chosen], name, others);
        let call = now + Duration::from_millis(1);
        let asked = if by_number {
            guard.decide_decode_under(&maker, call)
        } else {
            guard.decide_decode_under(&maker.0, call)
        };
        let asked = *asked.expect("a decision");
        let pair = target_at(&asked, 0).pair.expect("a pair");
        (host.picked(&asked).1 == chosen, pair.decode_verdict)
    }

    let (stays, handed_on) = (
        (true, DecodeVerdict::PINNED),
        (false, DecodeVerdict::RESPILL),
    );
    assert_eq!(again(false, true, TALK, 1), stays, "one other request");
    assert_eq!(again(false, true, TALK, 2), handed_on, "two others");
    assert_eq!(again(false, false, TALK, 1), handed_on, "read as it counts");
    let [_, decode_number] = numbers();
    assert_eq!(
        again(true, false, &decode_number, 2),
        handed_on,
        "by number"
    );
    assert_eq!(
        again(true, false, TALK, 2),
        stays,
        "no held decision's count"
    );
}

/// A decode asked again takes the key's pin with it from the first claim
/// of its own pair alone. A key that idled out behind a pair's first
/// decision starts over, and a second request of the key that claims the
/// same decode holds a claim of its own. The first pair's decode, asked
/// again, then claims its decode behind that claim, and its dispatch
/// leaves both: the key's next request follows the second request's
/// decode, though it is the busier of the two by then.
#[test]
fn a_decode_asked_again_leaves_the_claim_of_a_later_request_of_its_key() {
    let pair = PolicySpec::pair_presets("manual", "manual");
    let mut document = pair.expect("the pair presets");
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone();
    let ruled = rule.as_mut().expect("the stock rule");
    // A key idles out inside the life of a pair's lease.
    (ruled.ttl_s, ruled.sweep_s) = (1, 1);
    ruled.assignment = Assignment::MinLoad;
    document.bindings[0].sessions = rule;
    let (host, classes) = (host_of(document, 1, 2), every(1, 2));
    let asks = host.asks(&classes, true);
    let decided = |at: Instant| {
        DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, at).expect("a decision")
    };
    let verdict = |decision: &Decision| {
        let pair = target_at(decision, 0).pair.expect("a pair");
        (host.picked(decision), pair.decode_verdict)
    };
    let claimed = |decode: usize| ((0, decode), DecodeVerdict::CLAIMED);
    let now = Instant::now();
    let later = now + Duration::from_secs(3);
    // The first pair claims the one decode that is ready.
    host.decodes[1].set_status(WorkerStatus::NotReady);
    let mut first = decided(now);
    assert_eq!(verdict(first.decision()), claimed(0), "the first claim");
    first.dispatched(now);
    // Past the key's idle time a second request claims that decode anew.
    let second = decided(later);
    assert_eq!(verdict(second.decision()), claimed(0), "the second claim");
    // The first pair's decode, asked again, lands on the other decode and
    // claims it.
    host.decodes[0].set_status(WorkerStatus::NotReady);
    host.decodes[1].set_status(WorkerStatus::Ready);
    let again = *first.decide_decode_under(&asks, later).expect("a decision");
    assert_eq!(verdict(&again), claimed(1), "the call's claim");
    first.dispatched_decode(later);
    // The key's next request, with the first decode ready again and the
    // busier of the two: the load alone would take the call's decode.
    host.decodes[0].set_status(WorkerStatus::Ready);
    for _ in 0..8 {
        host.decodes[0].increment_load();
    }
    let next = decided(later);
    assert_eq!(
        host.picked(next.decision()),
        (0, 0),
        "the second request's decode"
    );
}

/// A pair's decision asks its request's maker where the key's requests
/// are counted under a document that reads the count, once a decision,
/// and under no other: a rule with no cap asks nothing, and neither
/// does a binding with no rule.
#[test]
fn a_pairs_decision_asks_where_a_key_is_counted_under_a_capped_rule_alone() {
    use std::cell::Cell;

    /// A maker that counts how often it is asked.
    struct Asked<'a>(Asks<'a>, Cell<u32>);

    impl RequestMaker for Asked<'_> {
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
            self.1.set(self.1.get() + 1);
            SessionCountedUnder::KeyText
        }
    }

    let classes = every(1, 1);
    let asked = |host: &Host| {
        let mut asks = host.asks(&classes, false);
        asks.0.facts.session_key = Identity::plain(TALK);
        let maker = Asked(asks, Cell::new(0));
        let decided = host
            .engine
            .decide_under(GatewayId::LOCAL, &maker, Instant::now());
        assert!(decided.is_ok(), "a decision");
        maker.1.get()
    };
    assert_eq!(asked(&capped_host(1, 1)), 1, "a capped rule");
    assert_eq!(asked(&ruled_host(u16::MAX, 1, 1)), 0, "a rule with no cap");
    assert_eq!(asked(&Host::of(1, 1)), 0, "no rule");
}

/// A pick of a decode that holds the whole prompt, among however many
/// holders, moves no count of writes left out through the pair's own
/// decision: the decode side's write is handed the decode side's
/// column. Ten decodes are each sent the prompt once. Then, while a
/// reader holds the record, a thread of its own picks the ninth and the
/// tenth and ends: a write it had kept would be left out there, as the
/// write of a prompt no decode was sent is.
#[test]
fn a_pick_of_a_ninth_decode_holder_leaves_nothing_out() {
    let pair = PolicySpec::pair_presets("round_robin", "cache_aware");
    let host = host_of(pair.expect("the pair presets"), 2, 10);
    // One instant for every decision: the list's table is read once.
    let now = Instant::now();
    let decide_among = |sent: &PromptHashes, decode: u32| {
        let (prefills, class) = (host.prefills.len(), [decode]);
        let classes = PartnerClasses::number(host.decodes.len(), (0..prefills).map(|_| &class[..]));
        let Asks(mut request, lists, _) = host.asks(&classes, false);
        request.facts = facts();
        sent.write_to(&mut request.facts);
        let decision = lists.decide(&host.engine, None, now, &mut request, None);
        assert_eq!(host.picked(&decision).1, decode as usize);
    };
    let sent = prompt(&host, 1_000);
    (0..10).for_each(|decode| decide_among(&sent, decode));
    let affinity = host.engine.affinity();
    let left_out_behind = |sent: &PromptHashes, decodes: &[u32]| {
        affinity.while_held(|| {
            thread::scope(|scope| {
                let picks = scope.spawn(|| {
                    for &decode in decodes {
                        decide_among(sent, decode);
                    }
                });
                picks.join().expect("the picks");
            });
        });
        affinity.left_out()
    };
    assert_eq!(left_out_behind(&sent, &[8, 9]), 0);
    // The count is live in this setup: every block of a prompt no
    // decode was sent is still to add, and its write meets the reader.
    assert_eq!(left_out_behind(&prompt(&host, 2_000), &[9]), 1);
}

/// The prefill side's twin of the case above: the prefill side's write
/// is handed the prefill side's column. Ten prefills are each sent the
/// prompt once, each decision held to its one place of the prefill
/// list. Then, while a reader holds the record, a thread of its own
/// picks the ninth and the tenth and ends, and no write is left out.
#[test]
fn a_pick_of_a_ninth_prefill_holder_leaves_nothing_out() {
    let pair = PolicySpec::pair_presets("cache_aware", "round_robin");
    let host = host_of(pair.expect("the pair presets"), 10, 2);
    // One instant for every decision: the list's table is read once.
    let now = Instant::now();
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let decide_at = |sent: &PromptHashes, prefill: u32| {
        let place = [prefill];
        let Asks(mut request, mut lists, _) = host.asks(&classes, false);
        lists.prefill_places = Some(&place);
        request.facts = facts();
        sent.write_to(&mut request.facts);
        let decision = lists.decide(&host.engine, None, now, &mut request, None);
        assert_eq!(host.picked(&decision).0, prefill as usize);
    };
    let sent = prompt(&host, 1_000);
    (0..10).for_each(|prefill| decide_at(&sent, prefill));
    let affinity = host.engine.affinity();
    let left_out_behind = |sent: &PromptHashes, prefills: &[u32]| {
        affinity.while_held(|| {
            thread::scope(|scope| {
                let picks = scope.spawn(|| {
                    for &prefill in prefills {
                        decide_at(sent, prefill);
                    }
                });
                picks.join().expect("the picks");
            });
        });
        affinity.left_out()
    };
    assert_eq!(left_out_behind(&sent, &[8, 9]), 0);
    // The count is live here too: a prompt no prefill was sent has
    // every block still to add, and its write meets the reader.
    assert_eq!(left_out_behind(&prompt(&host, 2_000), &[9]), 1);
}

/// Who placed a side, and whether its rule keeps it from the profile.
type SideWord = (PlacedBy, bool);

/// Each side's word of `decision`: the prefill side's, then the decode
/// side's.
fn side_words(decision: &Decision) -> [SideWord; 2] {
    let pick = decision.pick().expect("a pick");
    let decode = first_target(decision).pair.expect("a pair").decode_placing;
    assert_eq!(pick.decode_untold(), decode.untold());
    [
        (pick.placed_by(), pick.untold()),
        (decode.placed_by(), decode.untold()),
    ]
}

/// How a pair's host sends its decode.
#[derive(Clone, Copy, Debug)]
enum Sent {
    /// With its prefill.
    Together,
    /// Behind its prefill, decided again there.
    Again,
    /// Behind its prefill, with no call.
    Later,
}

/// The words a stage hears of a pair's side: each word and its side.
type SideWords = Vec<(&'static str, Side)>;

/// The request of `key` served through its guard as its host serves
/// it: decided, its prefill dispatched, its decode sent as `sent` says
/// and both sides ended. The two places, each side's word as the
/// guard's pick states it behind every call, and what the stages heard.
fn served(host: &Host, key: u64, sent: Sent) -> ((usize, usize), [SideWord; 2], SideWords) {
    let classes = every(host.prefills.len(), host.decodes.len() as u32);
    let mut asks = host.asks(&classes, !matches!(sent, Sent::Together));
    asks.0.facts.session_key = Identity::hashed(key);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    guard.dispatched(now);
    if matches!(sent, Sent::Again) {
        guard.decide_decode_under(&asks, now).expect("a decision");
    }
    guard.dispatched_decode(now);
    let (places, words) = (host.picked(guard.decision()), side_words(guard.decision()));
    let first = first_target(guard.decision());
    let decode = first.pair.expect("a pair").decode_id;
    guard.completed(&outcome(Side::Prefill, first.id), now);
    guard.completed(&outcome(Side::Decode, decode), now);
    let heard = host.words().into_iter().map(|word| (word.0, word.1));
    (places, words, heard.collect())
}

/// Each side of a pair is told to the stages by its own word, through
/// the engine's entries and a guard's words. A key's first pair is the
/// stages' on both sides. Its next follows both pins and is the
/// rule's on both. With the worker of its prefill pin down, the stages
/// place the prefill beside a decode the rule placed, and the other
/// way round with the worker of its decode pin down. Under the rule's
/// word `untold` a stage hears the issue and the dispatch of a side
/// the stages placed and of no side the rule placed alone, and the end
/// of both. Under its default word a stage hears every word. The case's
/// stage names all four words: the engine tells a stage a word only
/// where the stage names it (`Lifecycle::hears`).
///
/// So it is for a decode sent with its prefill, for one decided again
/// behind its prefill, and for one sent there with no call. A decode
/// the stages placed at the pair's first decision stays the stages'
/// where it is decided again: the call follows the claim that decision
/// made for the key, and places nothing. The pick the guard keeps
/// behind the call states the first decision's word of the prefill
/// side.
///
/// A gateway's sticky map tags a key by its role
/// (`model_gateway/src/policies/registry.rs:762-769`) and selects a
/// pair's two roles apart, each once a request
/// (`model_gateway/src/routers/common/placement.rs:3026`, `:3088`): a
/// role whose pin stands under its cap asks its policy nothing
/// (`registry.rs:794-797`), beside a role its policy places.
#[test]
fn a_guard_tells_each_side_of_a_pair_by_the_sides_own_word() {
    let (prefill, decode) = (Side::Prefill, Side::Decode);
    let stages: SideWord = (PlacedBy::Stages, false);
    // What a stage hears of a request whose sides state `words`.
    let heard = |sent: Sent, words: [SideWord; 2]| -> SideWords {
        let told = match sent {
            Sent::Together => [
                ("issued", prefill),
                ("issued", decode),
                ("dispatched", prefill),
                ("dispatched", decode),
            ],
            Sent::Again | Sent::Later => [
                ("issued", prefill),
                ("dispatched", prefill),
                ("issued", decode),
                ("dispatched", decode),
            ],
        };
        let hears = |side: Side| !words[usize::from(side == decode)].1;
        let mut heard: SideWords = told.into_iter().filter(|word| hears(word.1)).collect();
        heard.extend([("completed", prefill), ("completed", decode)]);
        heard
    };
    for kept in [RulePlaced::Untold, RulePlaced::Told] {
        let host = kept_host(Assignment::Delegate, kept);
        let ruled: SideWord = (PlacedBy::Rule, kept == RulePlaced::Untold);
        let downs = [None, Some(prefill), Some(decode)];
        let ways = [Sent::Together, Sent::Again, Sent::Later];
        let cases = downs
            .into_iter()
            .flat_map(|down| ways.map(|sent| (down, sent)));
        for (nth, (down, sent)) in cases.enumerate() {
            let case = format!("{kept:?}, {down:?} down, {sent:?}");
            // A key of the case's own: the stages place its first
            // pair, which pins it on both sides.
            let key = 100 + nth as u64;
            let (pinned, words, told) = served(&host, key, sent);
            assert_eq!(words, [stages; 2], "{case}: a new key");
            assert_eq!(told, heard(sent, words), "{case}: a new key");
            let gone = match down {
                Some(Side::Prefill) => Some(&host.prefills[pinned.0]),
                Some(Side::Decode) => Some(&host.decodes[pinned.1]),
                _ => None,
            };
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::NotReady);
            }
            let (places, words, told) = served(&host, key, sent);
            if let Some(worker) = gone {
                worker.set_status(WorkerStatus::Ready);
            }
            // The rule follows a pin that stands, and the stages place
            // the side whose pin's worker is down.
            let by = |side: Side| if down == Some(side) { stages } else { ruled };
            assert_eq!(words, [by(prefill), by(decode)], "{case}");
            assert_eq!(told, heard(sent, words), "{case}");
            let stayed = (places.0 == pinned.0, places.1 == pinned.1);
            assert_eq!(
                stayed,
                (down != Some(prefill), down != Some(decode)),
                "{case}"
            );
        }
    }
}

/// The session table hears the dispatch of a side its rule keeps from
/// the profile. Under a rule whose own assignment places a new key,
/// both sides of the key's first pair are the rule's, and no stage
/// hears their issue or their dispatch. The host takes the pick's
/// second target, the other decode: the decode's dispatch pins the key
/// there, and the key's next pair follows it.
///
/// A gateway's sticky map keeps the worker a key's request went to,
/// whoever placed it (`model_gateway/src/policies/registry.rs:832-841`).
#[test]
fn the_session_table_hears_the_dispatch_of_a_side_kept_from_the_profile() {
    let host = kept_host(Assignment::MinLoad, RulePlaced::Untold);
    let classes = every(2, 2);
    let mut asks = host.asks(&classes, false);
    asks.0.facts.session_key = Identity::hashed(7);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    let ruled: SideWord = (PlacedBy::Rule, true);
    assert_eq!(side_words(guard.decision()), [ruled; 2]);
    let first = host.picked(guard.decision());
    let taken = guard.take(1).expect("a second target");
    let pair = taken.pair.expect("a pair");
    assert_eq!(taken.position as usize, first.0, "the one prefill");
    assert_ne!(pair.decode_position as usize, first.1, "another decode");
    guard.dispatched(now);
    guard.dispatched_decode(now);
    guard.completed(&outcome(Side::Prefill, taken.id), now);
    guard.completed(&outcome(Side::Decode, pair.decode_id), now);
    let ends = [("completed", Side::Prefill), ("completed", Side::Decode)];
    let heard: SideWords = host.words().into_iter().map(|w| (w.0, w.1)).collect();
    assert_eq!(heard, ends);
    let (places, words, heard) = served(&host, 7, Sent::Together);
    assert_eq!(places, (first.0, pair.decode_position as usize));
    assert_eq!((words, heard), ([ruled; 2], ends.to_vec()));
}

/// A guard's decode call dates the claim it makes by its pair's lease.
/// The pair's first decision claims a decode for the key, and that
/// decode is down at the call, a second later: the call claims the
/// other decode. The guard's release for a refused commit is told under
/// the pair's lease and takes the call's claim back. The key's next
/// pair follows the decode the first decision claimed.
#[test]
fn a_guards_decode_call_claims_under_its_pairs_lease() {
    let mut document = kept_document("round_robin", Assignment::Delegate, RulePlaced::Told);
    let rule = document.bindings[0].sessions.as_mut();
    // The claim made last leads the key's pins while it stands.
    rule.expect("the pair's rule").recovery = Recovery::Latest;
    let host = host_of(document, 2, 2);
    let classes = every(2, 2);
    let mut asks = host.asks(&classes, true);
    asks.0.facts.session_key = Identity::hashed(7);
    let now = Instant::now();
    let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
    let mut guard = guard.expect("a decision");
    let first = host.picked(guard.decision());
    guard.dispatched(now);
    let claimed = &host.decodes[first.1];
    claimed.set_status(WorkerStatus::NotReady);
    let call = now + Duration::from_secs(1);
    let again = *guard.decide_decode_under(&asks, call).expect("a decision");
    claimed.set_status(WorkerStatus::Ready);
    let moved = host.picked(&again);
    assert_eq!(moved.0, first.0, "the held prefill");
    assert_ne!(moved.1, first.1, "the call claims the other decode");
    guard.released(ReleaseReason::CommitFailed, call);
    drop(guard);
    let (next, _, _) = served(&host, 7, Sent::Together);
    assert_eq!(next.1, first.1, "the call's claim left with its release");
}

/// A decode call that follows a pin hands the session table its pair's
/// lease date through each of its two entries. One is a guard's call,
/// and the other issues no lease. The pair's first decision is older than
/// the key's idle time by the call. Another request began the key's new
/// life and was served, and the call follows that pin.
///
/// A word under the pair's lease is then of that life, and a word one
/// microsecond older is not.
///
/// A gateway's sticky map pins a decode where that role's selection
/// picks it (`model_gateway/src/policies/manual.rs:322-368`).
#[test]
fn a_decode_call_that_follows_a_pin_dates_its_keys_life_through_each_entry() {
    for leased in [true, false] {
        let mut document = kept_document("round_robin", Assignment::Delegate, RulePlaced::Told);
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the pair's rule");
        (rule.ttl_s, rule.sweep_s, rule.recovery) = (1, 1, Recovery::Latest);
        let host = host_of(document, 2, 2);
        let classes = every(2, 2);
        let mut asks = host.asks(&classes, true);
        asks.0.facts.session_key = Identity::hashed(7);
        // The pair's first decision, and its prefill's dispatch.
        let now = Instant::now() + Duration::from_secs(1);
        let guard = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut guard = guard.expect("a decision");
        guard.dispatched(now);
        let issued = guard.lease().expect("the pair's lease").lease.issued_micros;
        assert_ne!(issued, 0, "the lease states its instant");
        // The key idles out. Another request of the key begins its new
        // life and is served.
        let later = now + Duration::from_secs(10);
        let other = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, later);
        let mut other = other.expect("a decision");
        other.dispatched(later);
        other.dispatched_decode(later);
        let served = first_target(other.decision())
            .pair
            .expect("a pair")
            .decode_id;
        // The first request's decode call follows that pin.
        let call = later + Duration::from_secs(1);
        let again = if leased {
            *guard.decide_decode_under(&asks, call).expect("a decision")
        } else {
            let held = first_target(guard.decision());
            let own = host.engine.loaded();
            let none = (Session::NONE, WorkerId(0));
            let again = own.decide_decode_without_lease(&asks, &held, none, issued, call);
            again.expect("a decision")
        };
        let pair = first_target(&again).pair.expect("a pair");
        assert_eq!(pair.decode_id, served, "the new life's pin");
        let verdict = pair.decode_verdict.on(served);
        assert_eq!(verdict, SessionVerdict::Pinned, "{leased}");
        let own = host.engine.loaded();
        let table = own.document.spec.engine().sessions();
        let (key, elsewhere) = (decode_session(&asks.0.facts), WorkerId(9_000));
        table.dispatched(key, SessionVerdict::Pinned, elsewhere, issued - 1);
        let front = || table.pins(key).map(|pins| pins[0]);
        assert_eq!(front(), Some(served), "{leased}: one before");
        table.dispatched(key, SessionVerdict::Pinned, elsewhere, issued);
        assert_eq!(front(), Some(elsewhere), "{leased}: the lease's date");
    }
}

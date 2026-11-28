//! The cases of a retry: the pass and the rotation it is put to, the ranks
//! of an engine it tried, the list of tried workers it reads, its lease, and
//! the lent tree it is decided inside.

use super::*;

/// What the one turn does not decide goes to the whole pass over
/// records: a retry never meets the worker it tried, and a pool whose
/// every worker is overloaded is answered by the binding's own rule.
#[test]
fn a_retry_and_a_pool_no_turn_admits_are_put_to_the_whole_pass() {
    let engine = preset("round_robin");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let tried = [worker_id(list[0].as_ref())];
    let mut retry = request();
    retry.facts.tried = &tried;
    retry.facts.attempt = 1;
    let asks = Asks::of(retry, Some(&members));
    for _ in 0..8 {
        let decided = decided(&engine, &asks);
        let pick = decided.decision.pick().expect("a pick");
        assert!(pick.targets.iter().all(|target| target.id != tried[0]));
    }
    for worker in list.iter() {
        worker.set_overloaded(true);
    }
    let asks = Asks::of(request(), Some(&members));
    match decided(&engine, &asks).decision {
        Decision::Pick(pick) => assert_eq!(pick.trace.lifted, LiftedVeto::Overload),
        Decision::Shed(shed) => assert_eq!(shed.emptied_by, "not-overloaded"),
        Decision::Queue(_) => panic!("no queue is planned"),
    }
}

/// A retry takes no turn of the list: it turns the rotation of the
/// workers it has not tried, at every size. So the first attempts
/// between the retries take the list's workers one after the other,
/// and each worker not tried takes an exact share of the retries. The
/// gateway's classic placement hands its policy a retry's untried
/// workers as a list of their own, which nobody names
/// (`model_gateway/src/routers/common/placement.rs:1256-1269`,
/// `:1277`), and its policy takes the list's turn for a list named
/// whole alone (`model_gateway/src/policies/round_robin.rs:145`), a set's
/// rotation for any other (`:158-167`).
///
/// A request that names a worker it tried is a retry whatever number it
/// states: a host that states none is answered as one that does.
#[test]
fn a_retry_turns_the_untried_workers_rotation_and_no_turn_of_the_list() {
    let sizes = [4, DRAW_FROM, 64];
    for (workers, attempt) in sizes.into_iter().flat_map(|n| [(n, 1), (n, 0)]) {
        let engine = preset("round_robin");
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        let tried = [worker_id(list[0].as_ref())];
        let mut retry = request();
        retry.facts.tried = &tried;
        retry.facts.attempt = attempt;
        let (first, retry) = (
            Asks::of(request(), Some(&members)),
            Asks::of(retry, Some(&members)),
        );
        // As many first attempts as turn the list once for each worker
        // not tried, and a retry behind each: the retries go once round
        // the untried workers for each worker of the list.
        let requests = workers * (workers - 1);
        let (mut firsts, mut retries) = (Vec::new(), vec![0_usize; workers]);
        for _ in 0..requests {
            firsts.push(placed(&engine, &first).place as usize);
            let decided = decided(&engine, &retry);
            let pick = decided.decision.pick().expect("a pick");
            retries[pick.targets.first().expect("the target").position as usize] += 1;
        }
        let in_turn: Vec<usize> = (0..requests)
            .map(|turn| (firsts[0] + turn) % workers)
            .collect();
        let said = format!("{workers} workers, a retry that states attempt {attempt}");
        assert_eq!(firsts, in_turn, "{said}: the first attempts");
        let mut shares = vec![workers; workers];
        shares[0] = 0;
        assert_eq!(retries, shares, "{said}: the retries");
    }
}

/// A retry leaves every rank of the engine it tried while its list
/// holds another worker: no target it is answered, first or
/// alternate, is a rank of that engine, under each built-in policy,
/// over a short list and over one long enough for the passes that
/// draw. Named the tried rank alone, the stages kept the engine's
/// other ranks. That is the gateway's classic answer: its tried list
/// names an engine by the base address its ranks share, and a tried
/// engine steps aside for the retry while another candidate is
/// available (`model_gateway/src/routers/common/placement.rs:556-576`
/// and `:1174-1179` at main `ef096118a`).
#[test]
fn a_retry_leaves_every_rank_of_the_engine_it_tried() {
    // Four ranks, a worker alone, two ranks and a worker alone; then
    // three engines of eight ranks.
    for shape in [&[4, 1, 2, 1][..], &[8, 8, 8]] {
        let (list, engine_of) = engines(shape);
        // The first engine's last rank was tried.
        let tried = [worker_id(list[shape[0] - 1].as_ref())];
        for name in [
            "round_robin",
            "random",
            "passthrough",
            "power_of_two",
            "least_load",
            "bucket",
            "cache_aware",
            "consistent_hashing",
            "prefix_hash",
            "manual",
        ] {
            let engine = preset(name);
            let members = PublishedList::new(Arc::clone(&list));
            let mut landed = BTreeSet::new();
            for seed in 0..64 {
                let mut retry = request();
                retry.facts.seed = seed;
                retry.facts.attempt = 1;
                retry.facts.tried = &tried;
                // A ring and a sessions rule place a request by its key.
                retry.facts.session_key = Identity::hashed(seed + 1);
                let asks = Asks::of(retry, Some(&members));
                let outcome = decided(&engine, &asks);
                let pick = outcome.decision.pick().expect("a pick");
                landed.extend(pick.targets.iter().map(|target| target.position));
            }
            let tried_engine: Vec<_> = landed
                .iter()
                .filter(|&&place| engine_of[place as usize] == 0)
                .collect();
            assert!(
                tried_engine.is_empty(),
                "{name} over {shape:?}: places {tried_engine:?} are ranks of the tried engine"
            );
            assert!(!landed.is_empty(), "{name} over {shape:?}");
        }
    }
}

/// Over a list that is one engine, a retry is handed the engine again:
/// no other worker is left, so the stage that passes tried workers over
/// steps aside for the whole list, the tried rank with the others. That
/// is the gateway's classic answer: where no untried candidate is
/// available, its pool is used whole
/// (`model_gateway/src/routers/common/placement.rs:571-573` at main
/// `ef096118a`). Named the tried rank alone, the stage kept the other
/// ranks and never handed that one.
#[test]
fn a_retry_over_one_engine_is_handed_its_ranks_again() {
    let engine = preset("random");
    let (list, _) = engines(&[4]);
    let members = PublishedList::new(Arc::clone(&list));
    let tried = [worker_id(list[1].as_ref())];
    let mut firsts = BTreeSet::new();
    for seed in 0..64 {
        let mut retry = request();
        retry.facts.seed = seed;
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let outcome = decided(&engine, &asks);
        let pick = outcome.decision.pick().expect("a pick");
        firsts.insert(pick.targets.first().expect("a target").position);
    }
    assert_eq!(firsts, BTreeSet::from([0, 1, 2, 3]), "the engine whole");
}

/// A key pinned on one rank of an engine leaves that engine on a retry
/// that tried another of its ranks: the key's rows are read under the
/// retry's whole list of tried workers, so the pin is not followed, and
/// the whole pass places the request on another engine. The gateway's
/// classic sticky policy is handed its candidates with the tried
/// engines out, and picks another worker where none of its key's is
/// among them (`model_gateway/src/routers/common/placement.rs:556-576`
/// and `model_gateway/src/policies/manual.rs:303-308` at main
/// `ef096118a`).
#[test]
fn a_keyed_retry_leaves_the_engine_its_key_is_pinned_on() {
    // The stock sessions rule: its binding pins a key.
    let engine = preset("manual");
    let (list, engine_of) = engines(&[4, 4]);
    let members = PublishedList::new(Arc::clone(&list));
    // One request of `key` that tried `tried`, served by its first
    // target: that target's place.
    let sent = |key: u64, tried: &[WorkerId]| {
        let mut keyed = request();
        keyed.facts.seed = key;
        keyed.facts.session_key = Identity::hashed(key);
        keyed.facts.tried = tried;
        let asks = Asks::of(keyed, Some(&members));
        served(&engine, &asks, Instant::now()).position as usize
    };
    for key in 1..=16 {
        let pin = sent(key, &[]);
        assert_eq!(sent(key, &[]), pin, "key {key}: its pin");
        // Another rank of the pin's engine was tried.
        let other = (pin & !3) | ((pin + 1) & 3);
        let tried = [worker_id(list[other].as_ref())];
        let went = sent(key, &tried);
        assert_ne!(
            engine_of[went], engine_of[pin],
            "key {key}: place {went} is a rank of the tried engine"
        );
    }
}

/// The workers a retry is decided to have tried stand in the calling
/// thread's own list, which only a retry over ranks takes: a first
/// attempt leaves it as it was, and so does a retry over workers that
/// each stand alone. It grows at a thread's first such retry and is
/// kept: the retries behind it allocate nothing for it.
#[test]
fn only_a_retry_over_ranks_takes_the_threads_list_and_it_grows_once() {
    let engine = preset("power_of_two");
    let (ranks, _) = engines(&[4, 4]);
    let ranked = PublishedList::new(Arc::clone(&ranks));
    let alone = pool(8);
    let unranked = PublishedList::new(Arc::clone(&alone));
    // Where the thread's list stands and what it holds room for.
    let kept = || {
        let list = TRIED.take();
        let stands = (list.as_ptr(), list.capacity());
        TRIED.set(list);
        stands
    };
    let asked = |members: &PublishedList, tried: &[WorkerId]| {
        let mut asked = request();
        asked.facts.tried = tried;
        let asks = Asks::of(asked, Some(members));
        assert!(decided(&engine, &asks).decision.pick().is_some());
    };
    asked(&ranked, &[]);
    assert_eq!(kept().1, 0, "a first attempt takes no list");
    asked(&unranked, &[worker_id(alone[0].as_ref())]);
    assert_eq!(kept().1, 0, "nor a retry over workers that stand alone");
    let tried = [worker_id(ranks[1].as_ref())];
    asked(&ranked, &tried);
    let grown = kept();
    assert!(grown.1 >= 4, "the engine's four ranks stand in it");
    asked(&ranked, &tried);
    assert_eq!(kept(), grown, "the list is kept as it grew");
}

/// Over a list of workers that each stand alone the records hand no list
/// of tried workers back ([`PoolRecords::tried_engines`]). A retry decided
/// there by the function a retry over ranks runs ([`retried`]) reads the
/// request's own list: the worker the retry names is passed over. The
/// decision's own entry asks that function of no such list.
#[test]
fn a_retry_decided_over_workers_that_stand_alone_reads_the_requests_own_list() {
    let engine = preset("round_robin");
    let list = pool(3);
    let members = engine.pool_members(&list);
    let own = engine.loaded();
    let spec = &own.document.spec;
    let records = members.records(spec, &own.clock);
    assert!(!records.has_ranks(), "each worker stands alone");
    let tried = [worker_id(list[1].as_ref())];
    for seed in 0..16 {
        let mut retry = request();
        retry.facts.seed = seed;
        retry.facts.attempt = 1;
        retry.facts.tried = &tried;
        // One instant: the records' and the decision's.
        let now = Instant::now();
        let live = records::Live::at(spec.engine(), &own.clock, now, None, None)
            .turning(&members.rotation.0);
        let mut facts = retry.facts;
        let cx = Constraints::default();
        let decision = retried(
            own,
            &members,
            records,
            &live,
            &mut facts,
            &cx,
            &retry,
            None,
            now,
            WithoutShadow,
        );
        let pick = decision.pick().expect("a pick");
        let places: Vec<u32> = pick.targets.iter().map(|target| target.position).collect();
        assert!(!places.is_empty(), "seed {seed}");
        assert!(!places.contains(&1), "seed {seed}: {places:?}");
    }
}

/// A retry decided under its whole list of tried workers is leased as
/// a first attempt is: its lease is dated at the decision's instant on
/// the engine's clock, which the decision writes into the request.
#[test]
fn a_retry_over_ranks_is_leased_at_its_decisions_instant() {
    // The stock sessions rule takes a word on a lease.
    let engine = preset("manual");
    let (list, _) = engines(&[4, 4]);
    let members = PublishedList::new(Arc::clone(&list));
    let now = Instant::now() + Duration::from_secs(2);
    let dated = |tried: &[WorkerId]| {
        let mut keyed = request();
        keyed.facts.session_key = Identity::hashed(9);
        keyed.facts.tried = tried;
        let asks = Asks::of(keyed, Some(&members));
        let decided = engine.decide_under(GatewayId::LOCAL, &asks, now);
        let whole = decided.expect("an answer").whole().expect("whole").0;
        whole.lease.expect("a lease").lease.issued_micros
    };
    let first = dated(&[]);
    assert!(first >= 2_000_000, "the engine's clock at `now`: {first}");
    let tried = [worker_id(list[1].as_ref())];
    assert_eq!(dated(&tried), first, "the retry's lease is dated alike");
}

/// A tried worker its list no longer holds names itself alone: the
/// list says nothing of its engine, so the engine's other ranks stay
/// candidates. A list holds some ranks of an engine and not the others
/// while its registry removes them, one after another. The gateway's
/// classic passes them over there: its tried list names the engine's
/// base address, whichever of its ranks the list still holds
/// (`model_gateway/src/routers/common/placement.rs:1174-1179` at main
/// `ef096118a`).
#[test]
fn a_tried_worker_its_list_no_longer_holds_names_itself_alone() {
    let engine = preset("random");
    let (whole, engine_of) = engines(&[4, 4]);
    // The first engine's first rank was tried, and has left the list.
    let tried = [worker_id(whole[0].as_ref())];
    let list: Arc<[Arc<dyn Worker>]> = whole[1..].iter().cloned().collect();
    let members = PublishedList::new(Arc::clone(&list));
    let mut firsts = BTreeSet::new();
    for seed in 0..64 {
        let mut retry = request();
        retry.facts.seed = seed;
        retry.facts.tried = &tried;
        let asks = Asks::of(retry, Some(&members));
        let outcome = decided(&engine, &asks);
        let pick = outcome.decision.pick().expect("a pick");
        firsts.insert(pick.targets.first().expect("a target").position);
    }
    // A place of the shorter list is one less than the worker's place
    // in the whole one.
    let stayed = firsts
        .iter()
        .any(|&place| engine_of[place as usize + 1] == 0);
    assert!(
        stayed,
        "{firsts:?}: the tried engine's ranks stay candidates"
    );
}

/// A retry over a list where an engine holds several workers is decided
/// inside the lent tree as a first attempt is. Its facts are a copy
/// that names every worker of the engines it tried, and the copy goes
/// to the same lookup with the request's maker: the host is asked for
/// the entry once. The tree names the worker the first attempt was
/// sent to, and the retry leaves that worker's engine.
#[test]
fn a_retry_over_ranks_is_decided_inside_the_lent_tree() {
    let (engine, _tree) = lent("cache_aware", Also::Nothing);
    let (list, engine_of) = engines(&[2, 2]);
    let members = PublishedList::new(Arc::clone(&list));
    let tokens: Vec<u32> = (0..256).collect();
    let sent = |tried: &[WorkerId]| {
        let hashes = engine.prompt_hashes(MODEL, Some(&tokens), None, HashSeeds::PLAIN);
        let mut asked = request();
        hashes.write_to(&mut asked.facts);
        asked.facts.tried = tried;
        let asks = Asks {
            key: Some(&tokens),
            ..Asks::of(asked, Some(&members))
        };
        let decided = decided(&engine, &asks);
        assert_eq!(asks.keyed.get(), 1, "the entry is asked for once");
        let pick = decided.decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position as usize
    };
    let first = sent(&[]);
    let again = sent(&[worker_id(list[first].as_ref())]);
    assert_ne!(engine_of[again], engine_of[first]);
}

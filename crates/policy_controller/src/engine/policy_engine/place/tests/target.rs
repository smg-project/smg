//! The cases of a single request that names its worker.

use super::*;
use crate::policies::consistent_hashing::target_source;

/// The header the stock hash ring document names as its target source.
const TARGET: &str = "x-smg-target-worker";

/// The shed of a request whose named worker cannot take it.
const MISSED: Unplaced = Unplaced::Shed {
    reason: ShedReason::NoEndpoints,
    emptied_by: "target-worker",
    retry_after_secs: 1,
};

/// Adds the hash ring's target source to the one binding of `document`,
/// as its block's last source. A binding that named no block is given the
/// routing key header as its one source of a key first: a target source
/// is no block alone.
fn names_target(document: &mut PolicySpec) {
    let keyed = || RoutingKeySpec {
        sources: vec![KeySourceSpec {
            header: Some(MANUAL_SESSION_HEADER.to_owned()),
            ..KeySourceSpec::default()
        }],
        strip_lineage: false,
    };
    let block = document.bindings[0].routing_key.get_or_insert_with(keyed);
    block.sources.push(target_source());
}

/// The preset `name` with the target source in its binding.
fn naming(name: &str) -> &'static PolicyEngine {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    names_target(&mut document);
    hosted(engine_of(&document))
}

/// A request's headers that count how often the target header is read.
struct Counted<'a> {
    carried: &'a [(&'a str, &'a str)],
    reads: Cell<u32>,
}

impl<'a> Counted<'a> {
    fn of(carried: &'a [(&'a str, &'a str)]) -> Self {
        let reads = Cell::new(0);
        Self { carried, reads }
    }
}

impl HeaderRead for Counted<'_> {
    fn header(&self, name: &str) -> Option<&[u8]> {
        if name == TARGET {
            self.reads.set(self.reads.get() + 1);
        }
        let (_, value) = self.carried.iter().find(|(held, _)| *held == name)?;
        Some(value.as_bytes())
    }
}

/// A request that names its worker goes to that worker, ahead of its
/// routing value: under the stock hash ring document, over eight workers
/// that are all available, each number takes its request to the worker at
/// that place of the list, whichever worker owns the request's routing
/// value, and with no routing value too. A ring the request's caller lends
/// changes nothing of that: its walk places the request that names no
/// worker, and decides none that is held to a place.
///
/// The classic policy reads the header ahead of every other rule
/// (`model_gateway/src/policies/consistent_hashing.rs:187-193`).
#[test]
fn a_request_that_names_its_worker_goes_to_that_worker() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let order = [6, 1, 4, 0, 7, 2, 5, 3];
    let ring = Lent(order.iter().map(|&at| list[at].url().to_owned()).collect());
    let placed = |headers: &dyn HeaderRead, lent: Option<&dyn LentRing>| {
        let request = PlaceRequest {
            headers,
            ring: lent,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        Arc::clone(&engine.place(&request).expect("a placement").worker)
    };
    let mut owners = BTreeSet::new();
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let keyed = [(MANUAL_SESSION_HEADER, key)];
        owners.insert(worker_id(&*placed(&keyed, None)));
        // The lent ring places the request that names no worker: it goes
        // where that ring's walk reaches first from the key's place.
        let reached = order[(ring.position(key) % 8) as usize];
        let walked = placed(&keyed, Some(&ring));
        assert!(Arc::ptr_eq(&walked, &list[reached]), "{key}: the ring's");
        for place in 0..8 {
            let number = place.to_string();
            let named = [(MANUAL_SESSION_HEADER, key), (TARGET, number.as_str())];
            for lent in [None, Some(&ring as &dyn LentRing)] {
                let worker = placed(&named, lent);
                assert!(Arc::ptr_eq(&worker, &list[place]), "{key}, place {place}");
            }
        }
    }
    assert!(owners.len() > 1, "the keys have more than one owner");
    for place in 0..8 {
        let number = format!("+{place}");
        let worker = placed(&[(TARGET, number.as_str())], None);
        assert!(Arc::ptr_eq(&worker, &list[place]), "no key, place {place}");
    }
}

/// A header that is absent, empty or not text names no worker: the request
/// goes to its routing value's owner, as it does with no such header.
///
/// The classic policy takes the header's value where it is text and not
/// empty (`crates/external_router/src/header_utils.rs:94-103`), and reads
/// the routing value where it names no worker
/// (`model_gateway/src/policies/consistent_hashing.rs:188`, `:194`).
#[test]
fn a_header_that_is_empty_or_not_text_names_no_worker() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let placed = |headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).expect("a placement").id()
    };
    for key in ["talk-1", "talk-2", "talk-3", "talk-4"] {
        let owner = placed(&[(MANUAL_SESSION_HEADER, key)]);
        let empty = [(MANUAL_SESSION_HEADER, key), (TARGET, "")];
        assert_eq!(placed(&empty), owner, "{key}: an empty value");
        let mut opaque = HeaderMap::new();
        let value = |bytes: &[u8]| HeaderValue::from_bytes(bytes).expect("a header's value");
        opaque.insert(MANUAL_SESSION_HEADER, value(key.as_bytes()));
        opaque.insert(TARGET, value(&[0xc3, 0xa9]));
        assert_eq!(placed(&opaque), owner, "{key}: a value that is not text");
    }
}

/// The number counts the workers that are available when the request
/// arrives, in the list's order. Over three workers, with the one behind
/// number 1 not ready, number 1 lands on the worker that was behind
/// number 2, and number 2 is shed with no worker asked. Ready again, the
/// numbers read as before. A worker whose circuit breaker is open and one
/// over its overload thresholds are passed over the same way.
///
/// The classic router narrows a hash policy's workers to the available
/// ones ahead of the policy, and the policy counts in what is left
/// (`model_gateway/src/routers/common/placement.rs:1225-1234`,
/// `model_gateway/src/policies/consistent_hashing.rs:154-160`). A request
/// that names its worker lifts no overload veto there (`:262-264`). The
/// gateway holds the rule in a case of its own
/// (`model_gateway/tests/routing/header_routing_hints_test.rs:262-268`).
#[test]
fn the_number_counts_the_workers_that_are_available() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(3);
    let members = PublishedList::new(Arc::clone(&list));
    let placed = |number: &str| {
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).map(|placement| placement.id())
    };
    let id = |at: usize| worker_id(list[at].as_ref());
    list[1].set_status(WorkerStatus::NotReady);
    assert_eq!(placed("1"), Ok(id(2)), "the worker behind it");
    assert_eq!(placed("2"), Err(MISSED), "two workers are available");
    assert_eq!(placed("0"), Ok(id(0)));
    list[1].set_status(WorkerStatus::Ready);
    assert_eq!(placed("1"), Ok(id(1)), "ready again");

    list[0].set_overloaded(true);
    assert_eq!(placed("0"), Ok(id(1)), "over its thresholds");
    list[0].set_overloaded(false);
    trip(&list[0]);
    assert_eq!(placed("0"), Ok(id(1)), "an open breaker");
    assert_eq!(placed("1"), Ok(id(2)));
}

/// The number counts among the workers the route's rule takes: the rule
/// is asked of each available worker up to the one the number reaches,
/// and once more of that worker as the pick. A number past the workers
/// the rule takes is shed.
///
/// The classic router narrows a request's workers by its rule first
/// (`model_gateway/src/routers/common/placement.rs:1187-1188`).
#[test]
fn the_number_counts_among_the_workers_the_routes_rule_takes() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) != id(1) && worker_id(worker) != id(2)
    };
    let placed = |number: &str| {
        asked.borrow_mut().clear();
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            admits: Some(&admits),
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).map(|placement| placement.id())
    };
    assert_eq!(placed("1"), Ok(id(3)), "the second worker the rule takes");
    let calls: Vec<_> = [0, 1, 2, 3, 3].map(id).to_vec();
    assert_eq!(*asked.borrow(), calls, "the count, then the pick");
    assert_eq!(placed("3"), Ok(id(5)));
    assert_eq!(placed("4"), Err(MISSED), "the rule takes four workers");
    assert_eq!(asked.borrow().len(), 6, "each worker once, and no pick");
}

/// A request whose header's text is no number, or whose number is past
/// the workers it can be sent to, is shed with no endpoint under
/// `target-worker`, and no decision is made for it: the picker hears
/// nothing.
///
/// The classic policy misses a text it cannot read as a number and a
/// number outside its workers
/// (`model_gateway/src/policies/consistent_hashing.rs:155-160`), and names
/// no other worker (`:191`).
#[test]
fn a_number_that_reaches_no_worker_is_shed_with_no_decision() {
    let (engine, heard, _) = in_order_edited(false, names_target);
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let shed = |named: &str| {
        let headers = [(TARGET, named)];
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).err()
    };
    for named in ["6", "4294967295", "4294967296", "-1", "1.5", "one", " 1"] {
        assert_eq!(shed(named), Some(MISSED), "{named:?}");
    }
    assert_eq!(heard.lock().as_slice(), [], "no decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Where no worker of the list can be sent to, there is no worker to
/// count: the request leaves its worker's hold and is placed by the
/// binding's rule for workers that are all vetoed, whatever its header's
/// text. Over four workers that are all over their thresholds it goes to
/// the one with the fewest requests in flight. Its routing value keeps no
/// worker there: the same value with no named worker stays on its owner
/// over the veto, and with one it goes to the least loaded. Under a
/// binding that sheds such a list the request is shed as any other, under
/// the filter's name.
///
/// The classic router finds no available worker ahead of its policy, and
/// steers the request to its least-loaded worker
/// (`model_gateway/src/routers/common/placement.rs:1256-1258`). A request
/// that names its worker lifts no veto by its routing value there
/// (`model_gateway/src/policies/consistent_hashing.rs:262-264`).
#[test]
fn where_no_worker_can_be_sent_to_the_request_leaves_its_workers_hold() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(4);
    for (at, worker) in list.iter().enumerate() {
        worker.set_overloaded(true);
        for _ in 0..[3, 2, 0, 1][at] {
            worker.increment_load();
        }
    }
    let members = PublishedList::new(Arc::clone(&list));
    let placed = |engine: &'static PolicyEngine, headers: &dyn HeaderRead| {
        let request = PlaceRequest {
            headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        engine.place(&request).map(|placement| placement.id())
    };
    let id = |at: usize| worker_id(list[at].as_ref());
    let least = Ok(id(2));
    for named in ["0", "3", "9", "one"] {
        assert_eq!(placed(engine, &[(TARGET, named)]), least, "{named:?}");
    }
    let mut elsewhere = 0;
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let owner = placed(engine, &[(MANUAL_SESSION_HEADER, key)]);
        elsewhere += usize::from(owner != least);
        let named = [(MANUAL_SESSION_HEADER, key), (TARGET, "0")];
        assert_eq!(placed(engine, &named), least, "{key}");
    }
    assert!(elsewhere > 0, "a key's owner is another worker");

    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let Err(Unplaced::Shed {
        reason, emptied_by, ..
    }) = placed(shedding, &[(TARGET, "0")])
    else {
        panic!("a shed");
    };
    assert_eq!(reason, ShedReason::Saturated);
    assert_ne!(emptied_by, "target-worker");
}

/// A request whose named worker cannot take it after all is shed with no
/// endpoint under `target-worker`, behind its one decision, and moves to
/// no other worker. A rule that takes the worker when the number counts it
/// and refuses it as the pick leaves the request no worker: the decision
/// is given up by name and no second one is made. A worker that goes over
/// its overload thresholds behind the count is answered by the decision
/// held to it: under a binding that sheds workers that are all over them,
/// the request is shed under `target-worker` too. So is one whose worker
/// stops being ready behind the count: the held decision's own filter
/// leaves it no worker, and the shed takes the target's name, not the
/// filter's.
///
/// The classic policy reads the named worker's health once more behind
/// the count, and misses it there
/// (`model_gateway/src/policies/consistent_hashing.rs:156-160`).
#[test]
fn a_named_worker_that_cannot_take_the_request_after_all_is_shed() {
    let (engine, heard, _) = in_order_edited(false, names_target);
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let third = worker_id(list[2].as_ref());
    let headers = [(TARGET, "2")];
    let asked = Cell::new(0);
    let admits = |_: &dyn Worker| {
        asked.set(asked.get() + 1);
        asked.get() <= 3
    };
    let ruled = PlaceRequest {
        headers: &headers,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert_eq!(engine.place(&ruled).err(), Some(MISSED));
    assert_eq!(asked.get(), 4, "three of the count, and the pick");
    assert_eq!(heard.lock().as_slice(), given_up(third), "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    let mut document = PolicySpec::preset("consistent_hashing").expect("a built-in preset");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let overloads = |worker: &dyn Worker| {
        worker.set_overloaded(true);
        true
    };
    let racing = PlaceRequest {
        headers: &headers,
        admits: Some(&overloads),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert_eq!(shedding.place(&racing).err(), Some(MISSED));

    let filtering = hosted(preset("consistent_hashing"));
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let leaves = |worker: &dyn Worker| {
        worker.set_status(WorkerStatus::NotReady);
        true
    };
    let leaving = PlaceRequest {
        headers: &headers,
        admits: Some(&leaves),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert_eq!(filtering.place(&leaving).err(), Some(MISSED));
}

/// A retry that names its worker is handed that worker again, though the
/// request's failed attempt was sent to it.
///
/// The classic router leaves the workers a named request tried in its
/// candidates (`model_gateway/src/routers/common/placement.rs:1235-1243`).
#[test]
fn a_retry_that_names_its_worker_is_handed_it_again() {
    let engine = hosted(preset("consistent_hashing"));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "5")];
    let tried = [worker_id(list[5].as_ref())];
    let retry = PlaceRequest {
        headers: &headers,
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[5]));
}

/// A request that names its worker is never plain: under a document whose
/// one pick decides a plain request with no request made, it is made and
/// held to its place. Sixteen requests that name one place all go there,
/// and sixteen that name none rotate over the list.
#[test]
fn a_request_that_names_its_worker_is_not_plain() {
    let engine = naming("round_robin");
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let headers = [(TARGET, "5")];
    let named = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let plain = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    for turn in 0..16 {
        let placement = engine.place(&named).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[5]), "turn {turn}");
    }
    let rotated = (0..16).map(|_| engine.place(&plain).expect("a placement").id());
    let rotated: BTreeSet<WorkerId> = rotated.collect();
    assert_eq!(rotated.len(), 8, "a request that names none rotates");

    let held = Made {
        places: Some(&[5]),
        ..Made::new(engine, &named, &[])
    };
    assert!(held.plain().is_none(), "held to a place");
    assert!(Made::new(engine, &named, &[]).plain().is_some());
}

/// The target header is read once a call, ahead of the call's decision,
/// under a document whose binding names it: for a request that names a
/// worker, for one that names none, and for one that is shed. Under a
/// document that names no target source it is never read, and a request
/// that carries it is placed as any other.
#[test]
fn the_target_header_is_read_once_a_call_and_only_where_the_document_names_it() {
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let reads = |engine: &'static PolicyEngine, carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let placed = engine.place(&request).map(|placement| placement.id());
        (headers.reads.get(), placed)
    };
    let id = |at: usize| worker_id(list[at].as_ref());
    let naming = hosted(preset("consistent_hashing"));
    assert!(naming.spec().names_target());
    assert_eq!(reads(naming, &[(TARGET, "6")]), (1, Ok(id(6))));
    assert_eq!(reads(naming, &[(TARGET, "8")]), (1, Err(MISSED)));
    let (read, placed) = reads(naming, &[(MANUAL_SESSION_HEADER, "talk-1")]);
    assert_eq!((read, placed.is_ok()), (1, true));

    for name in ["round_robin", "manual", "prefix_hash", "cache_aware"] {
        let unnamed = hosted(preset(name));
        assert!(!unnamed.spec().names_target(), "{name}");
        let (read, placed) = reads(unnamed, &[(TARGET, "8")]);
        assert_eq!((read, placed.is_ok()), (0, true), "{name}");
    }
}

/// A pair's request is not read for a target: under the stock pair of two
/// hash rings, whose binding names the target source, a pair that carries
/// the header is placed on the workers its routing value owns, as it is
/// without the header, and the header is never read.
///
/// The classic router reads the header once a side, each side's policy
/// over its own workers
/// (`model_gateway/src/routers/common/placement.rs:2519`, `:2540`,
/// `:2616`): the engine does not yet.
#[test]
fn a_pairs_request_is_not_read_for_a_target() {
    let document = PolicySpec::pair_presets("consistent_hashing", "consistent_hashing");
    let engine = hosted(engine_of(&document.expect("the pair presets")));
    assert!(engine.spec().names_target());
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let classes = PartnerClasses::number(2, [&[0_u32, 1][..], &[0, 1][..]]);
    let placed = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair_of(&members, &classes)))
        };
        let placement = engine.place(&request).expect("a placement");
        let decode = placement.decode.as_deref().map(worker_id);
        (headers.reads.get(), placement.id(), decode)
    };
    for key in ["talk-1", "talk-2", "talk-3", "talk-4"] {
        let (_, prefill, decode) = placed(&[(MANUAL_SESSION_HEADER, key)]);
        for named in ["0", "1", "7"] {
            let with = placed(&[(MANUAL_SESSION_HEADER, key), (TARGET, named)]);
            assert_eq!(with, (0, prefill, decode), "{key}, names {named}");
        }
    }
}

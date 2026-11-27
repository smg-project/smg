//! The cases of the worker a call may be sent to: one that is ready and
//! that the route's own rule admits. They hold where a retry is placed,
//! how a call decides again past the targets it could not send to, three
//! decisions at most, and which workers its later decisions are held to.

use super::*;

/// A worker can be sent to while it is ready and its circuit breaker lets
/// requests through: either one alone holds a send back. The overload veto
/// does not: the decision that picked the worker read it.
#[test]
fn a_worker_is_ready_by_its_status_and_its_breaker() {
    let list = pool(2);
    let (worker, tripped) = (&list[0], &list[1]);
    assert!(ready(worker.as_ref()));
    worker.set_overloaded(true);
    assert!(
        ready(worker.as_ref()),
        "the overload veto is the decision's"
    );
    worker.set_status(WorkerStatus::NotReady);
    assert!(!ready(worker.as_ref()));

    for _ in 0..64 {
        if !tripped.circuit_breaker_can_execute() {
            break;
        }
        tripped.record_circuit_breaker_outcome(false);
    }
    assert!(tripped.routing_state().healthy, "still ready by its status");
    assert!(
        !ready(tripped.as_ref()),
        "an open breaker holds a send back"
    );
}

/// A picker that places the worker at place 0 of its list by itself, takes
/// no word of a lease, and marks that worker not ready behind the read that
/// found it eligible.
#[derive(Debug)]
struct Turns(Arc<dyn Worker>);

impl Lifecycle for Turns {}

impl Picker for Turns {
    fn name(&self) -> &'static str {
        "turns"
    }

    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn picks_alone(&self) -> bool {
        true
    }

    fn pick_alone(&self, _: &PoolWord<'_>, keep: &mut dyn FnMut(usize) -> bool) -> Option<usize> {
        let kept = keep(0);
        if kept {
            self.0.set_status(WorkerStatus::NotReady);
        }
        kept.then_some(0)
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

/// Under a document whose session table alone takes a word on a lease, the
/// worker at a plain request's place is read a second time, as a noted
/// pick's target is at its walk: where its picker turned it, the pick is
/// noted, and the call's next decision places the request on a worker that
/// is ready.
#[test]
fn a_place_under_a_table_that_alone_takes_a_word_is_read_ready_a_second_time() {
    let list = pool(2);
    let first = Arc::clone(&list[0]);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("turns", move || Box::new(Turns(Arc::clone(&first))));
    added.expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a built-in preset");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "turns".into();
    let spec = AppliedSpec::validate_in(&ruled_ahead(document), &stages, &[]);
    let engine = hosted(InProcess::new(spec.expect("the stages hold the picker")));
    let hears = engine.in_force().hears.intersects(EventSet::LEASE);
    let beside = engine.in_force().hears_beside_table;
    assert!(
        hears && !beside.intersects(EventSet::LEASE),
        "the session table alone takes a word"
    );
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let placement = engine.place(&request).expect("a placement");
    assert!(!ready(list[0].as_ref()), "the pick turned its worker");
    assert!(
        Arc::ptr_eq(&placement.worker, &list[1]),
        "the worker that is ready"
    );
}

/// A retry states the worker it tried: it is decided whole and placed on
/// another worker of its list.
///
/// The engine leaves out the one worker a retry names, by its id. The
/// classic router leaves out every rank at that worker's base address,
/// which the ranks of a data-parallel worker share
/// (`model_gateway/src/routers/common/placement.rs:1174-1179`, used at
/// `:568`). Over workers of one rank each, as here, the two agree.
#[test]
fn a_retry_is_placed_past_the_worker_it_tried() {
    let engine = hosted(preset("round_robin"));
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let tried = [worker_id(list[0].as_ref())];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    for _ in 0..4 {
        let placement = engine.place(&retry).expect("a placement");
        assert!(Arc::ptr_eq(&placement.worker, &list[1]));
        assert_eq!(placement.id(), worker_id(list[1].as_ref()));
    }
}

/// A retry whose tried worker has left its list names no worker of it and
/// states its number alone. It takes no turn of the list: the first
/// attempts between such retries take the list's workers one after the
/// other, and the retries go round every worker of the list, at every
/// size.
///
/// The classic router gives a request that tried a worker an untried list
/// of its own, whether or not that worker is still listed
/// (`model_gateway/src/routers/common/placement.rs:1256-1269`), and its
/// policy takes a list's turn for a list named whole alone (`:1277`,
/// `model_gateway/src/policies/round_robin.rs:145`). The gateway states the
/// number as the count of the workers a request tried and names the ones
/// still listed (`model_gateway/src/routers/common/place_request.rs:33-44`,
/// `:94`).
#[test]
fn a_retry_whose_tried_worker_left_the_list_takes_no_turn_of_it() {
    for workers in [4, DRAW_FROM, 64] {
        let engine = hosted(preset("round_robin"));
        let list = pool(workers);
        let members = PublishedList::new(Arc::clone(&list));
        let first = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
        let retry = PlaceRequest {
            attempt: 1,
            ..PlaceRequest::of(MODEL, first.lists)
        };
        let at = |request: &PlaceRequest<'_>| {
            let placement = engine.place(request).expect("a placement");
            let taken = |worker: &Arc<dyn Worker>| Arc::ptr_eq(worker, &placement.worker);
            list.iter().position(taken).expect("a worker of the list")
        };
        // As many first attempts as turn the list once for each worker but
        // one, and a retry behind each: the retries go a whole number of
        // times round the list.
        let requests = workers * (workers - 1);
        let (mut firsts, mut retries) = (Vec::new(), vec![0_usize; workers]);
        for _ in 0..requests {
            firsts.push(at(&first));
            retries[at(&retry)] += 1;
        }
        let in_turn: Vec<usize> = (0..requests)
            .map(|turn| (firsts[0] + turn) % workers)
            .collect();
        assert_eq!(firsts, in_turn, "{workers} workers: the first attempts");
        let shares = vec![workers - 1; workers];
        assert_eq!(retries, shares, "{workers} workers: the retries");
    }
}

/// A whole decision's targets are asked best first, and the first that can
/// be sent to is taken: ready, and let through by its circuit breaker. The
/// lease issued on the pick's first target says its dispatch on the target
/// taken, under that target's index.
///
/// The classic router sends to no other worker either. Its selection keeps
/// a worker that is available
/// (`model_gateway/src/routers/common/placement.rs:536-539`, over
/// `crates/worker/src/identity/basic.rs:438-443`), and where every worker
/// of a pool is vetoed it still takes one that is ready with its circuit
/// closed (`model_gateway/src/routers/common/overload.rs:59-61`, `:145`).
#[test]
fn the_first_target_that_can_be_sent_to_is_taken() {
    let (engine, heard) = in_order(false);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    for worker in &list[..2] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let mut placement = engine.place(&request).expect("a placement");
    let (first, taken) = (worker_id(list[0].as_ref()), worker_id(list[2].as_ref()));
    assert!(Arc::ptr_eq(&placement.worker, &list[2]));
    assert_eq!(placement.id(), taken);
    assert!(placement.owes_events());
    placement.sent(Side::Single);
    assert_eq!(
        heard.lock().as_slice(),
        [
            ("issued at 0".to_owned(), first),
            ("dispatched at 2".to_owned(), taken)
        ]
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Where none of a decision's targets can be sent to, the lease its pick
/// was issued is given up by name at once, and the call decides again.
/// Over a list that holds no other worker the next decision names the same
/// workers, and after three decisions, each given up by name, the request
/// has no worker. Under a document none of whose stages takes a word, no
/// lease was issued and nothing is said.
///
/// The classic router answers no worker there too: its selection misses
/// where no candidate is available
/// (`model_gateway/src/routers/common/placement.rs:471-472`).
#[test]
fn a_decision_no_target_of_which_can_be_sent_to_is_given_up_by_name() {
    let (engine, heard) = in_order(false);
    let list = pool(3);
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    let first = worker_id(list[0].as_ref());
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(first), given_up(first), given_up(first)].concat()
    );

    let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
    let profile = document.profiles.get_mut("round_robin");
    let filters = &mut profile.expect("the preset's profile").filter;
    filters.retain(|filter| *filter != FilterSpec::Name("healthy".to_owned()));
    let quiet = hosted(engine_of(&document));
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    assert_eq!(quiet.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!((quiet.faults(), quiet.unheard()), (0, 0));
}

/// Where no target of a decision can be sent to, the call decides again and
/// leaves those targets out: the second decision names the workers behind
/// the first one's, the third the workers behind the second's, and the
/// first target that can be sent to is taken. A decision that takes no
/// target is given up by name before the next is made. A worker the
/// request's failed attempt was sent to is left out beside them.
///
/// The classic router has no pick to take back: it narrows a request's
/// workers to the available ones ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:536-539`, `:549-553`),
/// and runs its selection once more where a worker came free (`:457-482`).
#[test]
fn a_call_decides_again_past_the_targets_it_could_not_send_to() {
    let (engine, heard) = in_order(false);
    let list = pool(14);
    for worker in &list[..8] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let mut placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[8]));
    placement.sent(Side::Single);
    let mut said = [given_up(id(0)), given_up(id(4))].concat();
    said.push(("issued at 0".to_owned(), id(8)));
    said.push(("dispatched at 0".to_owned(), id(8)));
    assert_eq!(heard.lock().as_slice(), said);
    drop(placement);
    heard.lock().clear();

    // A retry that tried the ninth worker is placed on the tenth.
    let tried = [id(8)];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[9]));
    assert_eq!(
        heard.lock()[..4],
        [given_up(id(0)), given_up(id(4))].concat()
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A call makes three decisions at most, and those of a request that
/// states no rule are held to no worker. Over a list none of whose first
/// twelve workers can be sent to, the three decisions name them four by
/// four, each is given up by name, and the request has no worker. No fourth
/// decision names the workers behind them.
///
/// The classic router bounds its selection too: one pass, and one more
/// where a worker came free
/// (`model_gateway/src/routers/common/placement.rs:457-482`).
#[test]
fn a_call_makes_three_decisions_at_most() {
    let (engine, heard) = in_order(false);
    let list = pool(16);
    for worker in &list[..12] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let request = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(id(0)), given_up(id(4)), given_up(id(8))].concat()
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The route's own rule is asked of the worker picked: a request whose
/// first pick it admits pays one call of it, on the worker the request is
/// placed on, and the list is not asked.
///
/// The classic router asks a request's rule of the candidates ahead of its
/// policy (`model_gateway/src/routers/common/placement.rs:504-509`, over
/// `:361-370`). Both send to no worker the rule refuses.
#[test]
fn the_routes_rule_is_asked_of_the_pick_alone_while_it_admits_it() {
    let engine = hosted(preset("round_robin"));
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        true
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    for turn in 1..=8 {
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(asked.borrow().len(), turn, "one call a request");
        assert_eq!(asked.borrow().last(), Some(&placement.id()));
    }
}

/// No request is placed on a worker the route's rule refuses. Behind the
/// first pick it refuses, the rule is asked of every worker of the list
/// once, in the list's order, and the call's next decision is held to the
/// workers it takes. That holds for a place's worker, and for a whole
/// decision's targets, whose walk asks the rule of a worker that is ready
/// and of no other. The worker the request is placed on is asked once
/// more, as the pick it is.
///
/// The classic router sends to no worker its request's rule refuses either:
/// it narrows the candidates by the rule ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:504-509`).
#[test]
fn a_worker_the_routes_rule_refuses_is_passed_over() {
    // A place: the list's first worker is the pick of every plain request,
    // and the rule takes the sixth alone. The rule is asked of the place,
    // of the six workers of the list, and of the sixth as the pick of the
    // call's second decision, which is held to it.
    let engine = hosted(preset("passthrough"));
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[5]));
    let calls: Vec<_> = [0].into_iter().chain(0..6).chain([5]).map(id).collect();
    assert_eq!(*asked.borrow(), calls);

    // A whole decision: the first worker is not ready, and the rule takes
    // the sixth alone.
    let (engine, heard) = in_order(false);
    let list = pool(8);
    list[0].set_status(WorkerStatus::NotReady);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let sixth = id(5);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) == sixth
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let mut placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[5]));
    let calls: Vec<_> = (1..4).chain(0..8).chain([5]).map(id).collect();
    assert_eq!(*asked.borrow(), calls);
    placement.sent(Side::Single);
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), id(5)));
    said.push(("dispatched at 0".to_owned(), id(5)));
    assert_eq!(heard.lock().as_slice(), said);
}

/// The one worker of a long list the route's rule takes is found by the
/// call's second decision: the first names four others, the rule refuses
/// them, the rule is asked of the sixteen workers of the list, and the
/// second decision is held to the worker it takes.
///
/// The classic router places such a request too: it narrows the list by
/// the rule ahead of its policy
/// (`model_gateway/src/routers/common/placement.rs:504-509`).
#[test]
fn the_one_worker_a_rule_takes_in_a_long_list_is_found_by_the_second_decision() {
    let (engine, heard) = in_order(false);
    let list = pool(16);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(14);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[14]));
    assert_eq!(calls.get(), 4 + 16 + 1, "four picks, the list, the pick");
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), far));
    assert_eq!(heard.lock().as_slice(), said);
}

/// The call's later decisions are made among the workers the rule takes,
/// so the rule's one worker is placed where a decision over the whole list
/// never names it. A retry that tried it is handed it again: the filter of
/// tried workers is soft. Over its overload thresholds, it is answered by
/// the binding's rule for a list wholly over them: taken where the binding
/// falls back, shed as saturated where the binding sheds.
///
/// The classic router answers the same three ways: it narrows by the rule
/// first (`model_gateway/src/routers/common/placement.rs:504-509`), steps
/// the tried workers aside while another is left (`:556-576`), and falls
/// back over a pool that is all overloaded (`:577-579`).
#[test]
fn the_rules_only_worker_is_placed_though_tried_or_overloaded() {
    let engine = hosted(preset("passthrough"));
    let list = pool(5);
    let members = PublishedList::new(Arc::clone(&list));
    let only = worker_id(list[4].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == only;
    let tried = [only];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&retry).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[4]), "tried, and alone");
    drop(placement);

    list[4].set_overloaded(true);
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&first).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[4]), "the fallback");
    drop(placement);

    let mut document = PolicySpec::preset("passthrough").expect("a built-in preset");
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let members = PublishedList::new(Arc::clone(&list));
    let first = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let Some(Unplaced::Shed { reason, .. }) = shedding.place(&first).err() else {
        panic!("a shed");
    };
    assert_eq!(reason, ShedReason::Saturated);
}

/// A request whose route's rule takes no worker of its list is shed with no
/// endpoint under the rule's name, behind one decision: the rule is asked
/// of that decision's picks, then once of every worker of the list, ready
/// or not, and no second decision is made.
///
/// The classic gRPC route answers a refusal of its own where no worker can
/// take a request's images, by what the workers can do and not by their
/// health
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:1465-1472`).
#[test]
fn a_rule_that_takes_no_worker_is_shed_under_its_name() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    list[5].set_status(WorkerStatus::NotReady);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let calls: Vec<_> = (0..4).chain(0..6).map(id).collect();
    assert_eq!(*asked.borrow(), calls);
    assert_eq!(heard.lock().as_slice(), given_up(id(0)), "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A rule that takes no worker sheds the request so where no pick of the
/// first decision is ready too. No pick is asked of the rule: it is asked
/// once of every worker of the list, and no second decision is made.
#[test]
fn a_rule_that_takes_no_worker_is_shed_where_no_pick_is_ready() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    for worker in list.iter() {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let calls: Vec<_> = (0..6).map(id).collect();
    assert_eq!(*asked.borrow(), calls, "no pick, and the list once");
    assert_eq!(heard.lock().as_slice(), given_up(id(0)), "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A rule that takes workers none of which is ready is the decision's to
/// answer: held to them, it sheds by its own filter and names it. The rule
/// is asked of a worker that is not ready too, which tells this request
/// from one whose rule takes no worker.
#[test]
fn a_rule_whose_workers_are_not_ready_is_answered_by_the_decision() {
    let engine = hosted(preset("round_robin"));
    let list = pool(4);
    list[3].set_status(WorkerStatus::NotReady);
    let members = PublishedList::new(Arc::clone(&list));
    let last = worker_id(list[3].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == last;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "healthy"));
}

/// The places a call keeps are the list's places whose worker the rule
/// takes, ascending, each once, ready or not. Under a profile that draws,
/// sixty-four requests all land on workers the rule takes, and on more
/// than one of them.
#[test]
fn the_places_a_rule_takes_are_ascending_and_hold_every_later_pick() {
    let engine = hosted(preset("random"));
    let list = pool(64);
    list[3].set_status(WorkerStatus::NotReady);
    let members = PublishedList::new(Arc::clone(&list));
    let thirds = || (0..64).step_by(3);
    let taken: BTreeSet<WorkerId> = thirds().map(|at| worker_id(list[at].as_ref())).collect();
    let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let mut call = Call::default();
    call.missed.push(worker_id(list[1].as_ref()));
    // A request that states no rule keeps no places.
    let unruled = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    hold_to_rule(&unruled, Some(members.lent()), None, &mut call.admitted).expect("no rule");
    assert_eq!(call.admitted, None);
    hold_to_rule(&request, Some(members.lent()), None, &mut call.admitted)
        .expect("the rule takes workers");
    let places: Vec<u32> = thirds().map(|at| at as u32).collect();
    assert_eq!(call.admitted.as_deref(), Some(&places[..]));

    let mut placed = BTreeSet::new();
    for turn in 0..64 {
        let placement = engine.place(&request).expect("a placement");
        assert!(taken.contains(&placement.id()), "turn {turn}");
        placed.insert(placement.id());
    }
    assert!(placed.len() > 1, "more than one of the rule's workers");
}

/// A keyed request whose owner the rule refuses is placed among the
/// workers the rule takes, and the key's next request under the same rule
/// lands on the same worker. A key that held one worker keeps it: its
/// request with no rule lands there as before. Under the sessions rule the
/// first decision names the key's pin alone, so every such request asks
/// the rule of the list: one call for the pin, sixteen for the list, one
/// for the pick. Under the hash ring the first decision names the owner
/// and three workers behind it, and the list is asked where the rule
/// refuses all four.
#[test]
fn a_keyed_request_whose_owner_the_rule_refuses_stays_on_one_worker() {
    for policy in ["manual", "consistent_hashing"] {
        let engine = hosted(preset(policy));
        let list = pool(16);
        let members = PublishedList::new(Arc::clone(&list));
        let mut by_list = 0;
        for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
            let headers = [(MANUAL_SESSION_HEADER, key)];
            let keyed = PlaceRequest {
                headers: &headers,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
            };
            let owner = engine.place(&keyed).expect("a placement").id();
            let calls = Cell::new(0);
            let admits = |worker: &dyn Worker| {
                calls.set(calls.get() + 1);
                let id = worker_id(worker);
                id != owner && id.0 % 2 == owner.0 % 2
            };
            let ruled = PlaceRequest {
                admits: Some(&admits),
                ..keyed
            };
            let first = engine.place(&ruled).expect("a placement").id();
            assert!(first != owner && first.0 % 2 == owner.0 % 2, "{policy}");
            by_list += usize::from(calls.get() > 16);
            if policy == "manual" {
                assert_eq!(calls.get(), 1 + 16 + 1, "{key}");
            }
            let again = engine.place(&ruled).expect("a placement").id();
            assert_eq!(again, first, "{policy}, {key}");
            let unruled = engine.place(&keyed).expect("a placement").id();
            assert_eq!(unruled, owner, "{policy}, {key}: the key keeps its owner");
        }
        assert!(by_list > 0, "{policy}: no key asked the rule of the list");
    }
}

/// A fresh key whose first decision names workers the rule refuses keeps no
/// pin of them: the commit refused every target of that decision, and the
/// claim it made goes with it. The request is sent to the rule's one
/// worker, and that worker is the key's pin: the key's request with no rule
/// lands there.
///
/// The classic router pins no worker its rule refuses: it narrows the list
/// by the rule ahead of its policy (`Candidates::of`,
/// `model_gateway/src/routers/common/placement.rs`).
#[test]
fn a_fresh_key_keeps_no_pin_of_a_worker_its_rule_refused() {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    rule.expect("the stock rule").assignment = Assignment::MinLoad;
    let engine = hosted(engine_of(&document));
    // Four idle workers and a busier one: the first decision names the four.
    let list = pool(5);
    list[4].increment_load();
    let members = PublishedList::new(Arc::clone(&list));
    let busy = worker_id(list[4].as_ref());
    let admits = |worker: &dyn Worker| worker_id(worker) == busy;
    let headers = [(MANUAL_SESSION_HEADER, "talk-fresh")];
    let keyed = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let ruled = PlaceRequest {
        admits: Some(&admits),
        ..keyed
    };
    let mut placement = engine.place(&ruled).expect("a placement");
    placement.sent(Side::Single);
    assert_eq!(placement.id(), busy);
    let next = engine.place(&keyed).expect("a placement");
    assert_eq!(next.id(), busy, "the pin is the worker that served the key");
}

/// A call's first decision is given the request's whole budget, and a later
/// one what the call has left of it. A request none of whose budget is left
/// is shed under `budget`, with no further decision. A request that states
/// no budget is given none, and no instant is kept for it.
#[test]
fn a_later_decision_is_given_what_the_call_has_left_of_the_budget() {
    let slow = |list: &Arc<[Arc<dyn Worker>]>| {
        let far = worker_id(list[6].as_ref());
        move |worker: &dyn Worker| {
            thread::sleep(Duration::from_millis(2));
            worker_id(worker) == far
        }
    };
    let (engine, heard, budgets) = in_order_budgets(false);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let admits = slow(&list);
    let request = PlaceRequest {
        budget_micros: 10_000_000,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[6]));
    let given = budgets.lock().clone();
    assert_eq!((given.len(), given[0]), (2, 10_000_000));
    // Less by the rule's calls at least, and not a tenth of the budget
    // gone: what is left of it, and no other number.
    let left = 9_000_000..=10_000_000 - 2_000;
    assert!(left.contains(&given[1]), "{given:?}");
    drop(placement);
    heard.lock().clear();

    let short = PlaceRequest {
        budget_micros: 1_000,
        ..request
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::Deadline,
        emptied_by: "budget",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(engine.place(&short).err(), Some(shed));
    let first = worker_id(list[0].as_ref());
    assert_eq!(heard.lock().as_slice(), given_up(first), "one decision");

    let unbounded = PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()));
    let mut call = Call::default();
    assert_eq!(call.left(&unbounded), Ok(0));
    assert!(
        call.began.is_none(),
        "no clock for a request with no budget"
    );
}

/// A rule that takes a worker when it is asked of the list and refuses it
/// as a pick places no request on it: the pick's answer stands. The call
/// asks the list no second time, names the worker again while no other is
/// left, and ends with no worker.
#[test]
fn a_rule_that_changes_its_answer_places_no_request_on_a_refused_pick() {
    let (engine, heard) = in_order(false);
    let list = pool(8);
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let calls = Cell::new(0);
    // Calls 5 to 12 are the list's walk: the sixth worker is taken there,
    // and nowhere else.
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        (5..=12).contains(&calls.get()) && worker_id(worker) == id(5)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 4 + 8 + 2, "the list is asked once");
    assert_eq!(
        heard.lock().as_slice(),
        [given_up(id(0)), given_up(id(5)), given_up(id(5))].concat()
    );
}

/// A call whose first decision names workers that are not ready has its
/// route's rule asked of the list behind that decision, though the rule
/// refused nobody: a pick that is not ready is not asked of it, so such a
/// decision learns nothing of the rule. Over a list whose first twelve
/// workers are not ready, three decisions over the whole list name them
/// four by four, and the request had no worker though the thirteenth is
/// ready and the rule takes it. Held to that worker, the second decision
/// names it.
///
/// The classic router places it: it narrows the list by the rule
/// (`accepted_by`, `model_gateway/src/routers/common/placement.rs:474-483`
/// on main `77d6a42d3`) and picks among the workers that are available
/// (`:1338-1341` there).
#[test]
fn a_call_whose_first_picks_are_not_ready_is_held_to_the_rules_workers() {
    let (engine, heard) = in_order(false);
    let list = pool(13);
    for worker in &list[..12] {
        worker.set_status(WorkerStatus::NotReady);
    }
    let members = PublishedList::new(Arc::clone(&list));
    let id = |at: usize| worker_id(list[at].as_ref());
    let far = id(12);
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(Arc::ptr_eq(&placement.worker, &list[12]));
    assert_eq!(calls.get(), 13 + 1, "the list, then the pick");
    let mut said = given_up(id(0)).to_vec();
    said.push(("issued at 0".to_owned(), far));
    assert_eq!(heard.lock().as_slice(), said);
    drop(placement);
    heard.lock().clear();

    // A rule that takes that worker of the list and refuses it as a pick:
    // the list is asked once, and the third decision is the call's last.
    calls.set(0);
    let fickle = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        calls.get() <= 13 && worker_id(worker) == far
    };
    let request = PlaceRequest {
        admits: Some(&fickle),
        ..request
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(calls.get(), 13 + 2, "the list is asked once");
    let three = [0, 12, 12].map(|first| given_up(id(first)));
    assert_eq!(heard.lock().as_slice(), three.concat());
}

/// The rule is asked of the list while the call holds no decision: where it
/// panics there, the decision it refused was given up by name already, and
/// no lease is left behind.
#[test]
fn a_rule_that_panics_at_the_list_leaves_no_lease_behind() {
    let (engine, heard) = in_order(false);
    let list = pool(6);
    let members = PublishedList::new(Arc::clone(&list));
    let first = worker_id(list[0].as_ref());
    let calls = Cell::new(0);
    // The four picks are refused, and the list's first worker panics.
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        assert!(calls.get() <= 4, "this rule panics");
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the rule's panic reaches the caller");
    assert_eq!(heard.lock().as_slice(), given_up(first));
    assert_eq!(engine.unheard(), 0);
}

/// An ask is made of what its call's earlier asks handed on: the targets
/// they could not send to are left out of its decision, so it is not plain,
/// and the prompt's blocks are the ones the call's first making hashed.
#[test]
fn an_ask_is_made_of_what_its_call_handed_on() {
    let engine = preset("cache_aware");
    let tokens = (0..600).collect::<Vec<u32>>();
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        ..PlaceRequest::of(MODEL, WorkerLists::None)
    };
    let mut call = Call::default();
    let document = engine.in_force();
    let blocks = {
        let first = Made::asked(engine.under(&document), &request, &call);
        assert!(first.plain().is_some());
        let facts = first.make(engine.standing_spec()).facts;
        assert!(!facts.block_hashes.is_empty());
        facts.block_hashes.as_ptr()
    };
    assert!(call.hashes.get().is_some(), "hashed in the call's cell");
    call.missed.push(WorkerId(3));
    let again = Made::asked(engine.under(&document), &request, &call);
    assert_eq!(again.plain(), None, "a missed target");
    let made = again.make(engine.standing_spec());
    assert_eq!(made.constraints.excluded, [WorkerId(3)]);
    assert!(
        ptr::eq(made.facts.block_hashes.as_ptr(), blocks),
        "hashed once a call"
    );
}

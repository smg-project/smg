//! What a host asks of the holder for the objects it builds: a host-local
//! row's checks off, each row's overload thresholds, and its document's
//! overload switch.

use super::*;

/// A host that is no gateway asks for a host-local row's object with
/// its checks off, whatever the row's policy says. The object of a row
/// any gateway dials keeps its checks, and so does the row's own once
/// its reach turns. A holder that was not asked builds each as its
/// policy says.
#[test]
fn a_host_local_rows_object_is_unchecked_where_the_host_asked() {
    let checked = |replica: &Replica, url| {
        !replica
            .at(url)
            .metadata()
            .health_config
            .disable_health_check
    };
    let local = WorkerFacts {
        reachable_from: Reachability::Gateway(GatewayId(1)),
        ..facts(A, 1)
    };
    let mut outside = Replica::new();
    let registry = Arc::clone(outside.objects.registry());
    outside.objects = Arc::new(Objects::new(registry).with_host_local_unchecked());
    for (replica, asked) in [(Replica::new(), false), (outside, true)] {
        replica.write(local.clone());
        replica.write(facts(B, 1));
        assert_eq!(checked(&replica, A), !asked, "the host-local row");
        assert!(checked(&replica, B), "a row any gateway dials");

        replica.write(facts(A, 1));
        assert!(checked(&replica, A), "the row once any gateway dials it");
    }
}

/// A host whose own load monitor judges overload asks for each object
/// built with its row's thresholds, each over the host's default for
/// its signal. A row whose thresholds move is laid again, and its
/// object states the new ones. A holder that was not asked builds an
/// object that states none, so its host's monitor writes no verdict.
#[test]
fn an_object_states_its_rows_thresholds_where_the_host_judges_overload() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let stated = |replica: &Replica, url| replica.at(url).metadata().overload;
    let mut judging = Replica::new();
    let registry = Arc::clone(judging.objects.registry());
    judging.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    for (replica, asked) in [(Replica::new(), false), (judging, true)] {
        replica.write(facts(A, 1));
        replica.write(WorkerFacts {
            overload: Overload {
                kv_usage_permille: Some(650),
                waiting_requests: None,
            },
            ..facts(B, 1)
        });
        let own = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.65),
        };
        assert_eq!(stated(&replica, A), if asked { defaults } else { none });
        assert_eq!(stated(&replica, B), if asked { own } else { none });

        // The row's thresholds move: its object is laid again.
        let moved = Overload {
            kv_usage_permille: None,
            waiting_requests: Some(3),
        };
        replica.patch(
            B,
            WorkerPatch {
                overload: Some(moved),
                ..WorkerPatch::default()
            },
        );
        let own = OverloadThresholds {
            waiting_requests: Some(3),
            token_usage: Some(0.8),
        };
        assert_eq!(stated(&replica, B), if asked { own } else { none });
    }
}

/// The overload switch stated on `replica`'s holder, as a replica that
/// leads states it: how many objects were laid again.
fn switch(replica: &Replica, on: bool) -> usize {
    let store = replica.store.as_ref();
    replica.objects.state_overload_switch(store, on, &|| true)
}

/// A host states its document's overload switch on the holder. Off, the
/// object of an engine's row states its row's own thresholds alone, and
/// on again the host's default under them. A moved word lays again the
/// objects it changes and no other: a row that names both of its own
/// and an external row keep their objects. The same word again lays
/// nothing. A row laid again keeps its id,
/// its status and its count in flight, and loses its overload verdict
/// alone. A row that comes under a word is built by it, once.
#[test]
fn the_overload_switch_lays_again_the_rows_whose_default_it_changes() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let set = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    let own = |kv_usage_permille, waiting_requests| Overload {
        kv_usage_permille,
        waiting_requests,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    // A names none, B a usage alone, C both, and D is external.
    replica.write(facts(A, 1));
    replica.write(WorkerFacts {
        overload: own(Some(650), None),
        ..facts(B, 1)
    });
    replica.write(WorkerFacts {
        overload: own(Some(650), Some(3)),
        ..facts(C, 1)
    });
    replica.write(WorkerFacts {
        engine_kind: "external".to_string(),
        ..facts(D, 1)
    });
    let urls = [A, B, C, D];
    let stated = |replica: &Replica, url| replica.at(url).metadata().overload;
    // Every standing object runs: a request in flight, ready, vetoed.
    let before = urls.map(|url| replica.at(url));
    let ids = urls.map(|url| replica.id(url));
    for worker in &before {
        worker.increment_load();
        worker.set_status(ObjectStatus::Ready);
        assert!(replica
            .objects
            .registry()
            .set_worker_overloaded(worker, true));
    }
    replica.told();

    // On is the word a holder starts under: stated again, it lays none.
    assert_eq!(switch(&replica, true), 0);
    assert!(replica.told().is_empty());
    // Off: the two rows that took a default are laid again.
    assert_eq!(switch(&replica, false), 2);
    let mut replaced = replica.told();
    replaced.retain(|told| told.0 == "replaced");
    replaced.sort_by(|one, other| one.1.as_str().cmp(other.1.as_str()));
    let mut laid = vec![("replaced", ids[0].clone()), ("replaced", ids[1].clone())];
    laid.sort_by(|one, other| one.1.as_str().cmp(other.1.as_str()));
    assert_eq!(replaced, laid);
    assert_eq!(stated(&replica, A), set(None, None));
    assert_eq!(stated(&replica, B), set(None, Some(0.65)));
    assert_eq!(stated(&replica, C), set(Some(3), Some(0.65)));
    assert_eq!(stated(&replica, D), set(None, None));
    for (at, url) in urls.into_iter().enumerate() {
        let (after, again) = (replica.at(url), at < 2);
        assert_eq!(!Arc::ptr_eq(&before[at], &after), again, "{url}");
        assert_eq!(replica.id(url), ids[at], "{url}: the same id");
        let running = (after.status(), after.load());
        assert_eq!(
            running,
            (ObjectStatus::Ready, 1),
            "{url}: its running state"
        );
        assert_eq!(after.is_overloaded(), !again, "{url}: the verdict alone");
    }

    // The same word again lays nothing.
    assert_eq!(switch(&replica, false), 0);
    assert!(replica.told().is_empty());
    // A row that comes under the word is built by it, once.
    replica.write(facts(E, 1));
    assert_eq!(replica.told(), [("registered", replica.id(E))]);
    assert_eq!(stated(&replica, E), set(None, None));

    // On again: each row that takes a default is laid again.
    assert_eq!(switch(&replica, true), 3);
    let replaced = replica
        .told()
        .into_iter()
        .filter(|told| told.0 == "replaced");
    assert_eq!(replaced.count(), 3);
    assert_eq!(stated(&replica, A), defaults);
    assert_eq!(stated(&replica, B), set(Some(8), Some(0.65)));
    assert_eq!(stated(&replica, C), set(Some(3), Some(0.65)));
    assert_eq!(stated(&replica, D), set(None, None));
    assert_eq!(stated(&replica, E), defaults);
}

/// A host that states the word ahead of its first write builds each
/// object once, by that word: nothing stands to lay again. A holder
/// that was handed no thresholds keeps the word and lays nothing: its
/// objects state none under either word.
#[test]
fn the_overload_switch_stated_ahead_of_the_first_write_builds_each_object_once() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let mut judging = Replica::new();
    let registry = Arc::clone(judging.objects.registry());
    judging.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    for (mut replica, asked) in [(Replica::new(), false), (judging, true)] {
        assert_eq!(switch(&replica, false), 0);
        assert!(replica.told().is_empty(), "nothing is built by the word");
        replica.write(facts(A, 1));
        replica.write(facts(B, 1));
        let built: Vec<_> = replica.told().into_iter().map(|told| told.0).collect();
        assert_eq!(built, ["registered", "registered"], "each object once");
        for url in [A, B] {
            assert_eq!(replica.at(url).metadata().overload, none, "{url}");
        }
        // The word back on changes a holder with thresholds alone.
        let laid = switch(&replica, true);
        assert_eq!(laid, if asked { 2 } else { 0 });
        let stands = if asked { defaults } else { none };
        for url in [A, B] {
            assert_eq!(replica.at(url).metadata().overload, stands, "{url}");
        }
    }
}

/// The switch lays again a row as the holder holds it, and reads no row
/// of the store. A row that left and came back under its id, neither
/// read by the holder yet, is another life. A moved word lays the first
/// life's object again under the word, and nothing of the second life
/// takes its place: the second takes no status and no count in flight
/// of the first.
#[test]
fn the_overload_switch_lays_no_row_the_holder_has_not_read() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    // The first life stands relabelled "iron": the second life's row
    // states "gold".
    let first = replica.write(facts(A, 1));
    replica.patch(A, labels("iron"));
    let (old_id, before) = (replica.id(A), replica.at(A));
    before.increment_load();
    before.set_status(ObjectStatus::Ready);
    replica.told();

    // The second life stands in the store alone.
    replica.remove_around(A);
    assert_eq!(replica.store.upsert(labelled(A, 1, "gold")).id(), first);
    assert_eq!(switch(&replica, false), 1);
    let replaced = replica
        .told()
        .into_iter()
        .filter(|told| told.0 == "replaced");
    assert_eq!(replaced.collect::<Vec<_>>(), [("replaced", old_id)]);
    // The first life's object under the word: its status and its count,
    // and no label of the second life.
    let laid = replica.at(A);
    assert_eq!(
        replica.tier(A).as_deref(),
        Some("iron"),
        "the first life's row"
    );
    assert_eq!(laid.metadata().overload, none);
    assert_eq!((laid.status(), laid.load()), (ObjectStatus::Ready, 1));
}

/// A report that lands inside the registry's replacement leaves no
/// verdict on an object the switch laid with no threshold. A writer
/// judges as a load monitor does: it writes its verdict through the
/// object the registry holds, where that object states a threshold. It
/// runs against the switch's moves, off and on. The registry releases
/// a verdict ahead of its swap, so the writer's verdict lands between
/// the two in some round, and the object laid with no threshold would
/// stand vetoed with no report left to judge it.
#[test]
fn the_overload_switch_leaves_no_verdict_a_report_set_inside_a_replacement() {
    /// Stops the writer where the case ends, a failed case too.
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Relaxed);
        }
    }
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects =
        Arc::new(Objects::new(Arc::clone(&registry)).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let stopped = AtomicBool::new(false);
    thread::scope(|scope| {
        let _stop = Stop(&stopped);
        scope.spawn(|| {
            while !stopped.load(Relaxed) {
                let judged = registry
                    .get_by_url(A)
                    .filter(|object| object.metadata().overload.is_enabled());
                if let Some(object) = judged {
                    registry.set_worker_overloaded(&object, true);
                }
            }
        });
        for round in 0..400 {
            assert_eq!(switch(&replica, false), 1);
            let laid = replica.at(A);
            assert!(!laid.metadata().overload.is_enabled());
            assert!(!laid.is_overloaded(), "round {round}");
            assert_eq!(switch(&replica, true), 1);
        }
    });
}

/// The overload switch settles a status with the table behind each
/// lay. A health driver moved the standing object, and its telling
/// finds that object replaced and tells nothing: the object laid again
/// took the moved status, and the table holds its own. On a replica
/// that leads the table is told the object's status by the time the
/// call returns. One that does not lead writes no row, and its object
/// takes the status its row holds.
#[test]
fn the_overload_switch_settles_a_status_that_moved_ahead_of_its_lay() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    for leads in [true, false] {
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
        let row = replica.write(facts(A, 1));
        let before = replica.store.get(row).expect("the row");
        assert_eq!(before.status, WorkerStatus::Pending);
        // The driver's move, with no word to the table.
        replica.at(A).set_status(ObjectStatus::Ready);
        let epoch = replica.store.epoch();
        let store = replica.store.as_ref();
        let laid = replica
            .objects
            .state_overload_switch(store, false, &|| leads);
        assert_eq!(laid, 1);
        let after = replica.store.get(row).expect("the row");
        let object = replica.at(A).status();
        if leads {
            assert_eq!(after.status, WorkerStatus::Ready, "the table is told");
            assert_eq!(object, ObjectStatus::Ready);
        } else {
            assert_eq!(replica.store.epoch(), epoch, "no row is written");
            assert_eq!(after.status, WorkerStatus::Pending);
            assert_eq!(object, ObjectStatus::Pending, "its row's status");
        }
    }
}

/// The overload switch holds the holder's turn over its lays, so every
/// build reads one word. Its caller's word of the lead is asked behind
/// a lay, and finds the turn held. Behind the call the turn is free.
#[test]
fn the_overload_switch_holds_the_holders_turn_over_its_lays() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let (objects, store) = (&replica.objects, replica.store.as_ref());
    let turns = RefCell::new(Vec::new());
    let leads = || {
        turns.borrow_mut().push(objects.turn.is_locked());
        true
    };
    assert_eq!(objects.state_overload_switch(store, false, &leads), 1);
    let turns = turns.into_inner();
    assert!(!turns.is_empty(), "the lay asks for the lead");
    assert!(turns.iter().all(|&held| held), "inside the turn");
    assert!(!objects.turn.is_locked(), "the turn is let go");
}

/// The overload switch takes the holder's turn ahead of its word. While
/// another holds the turn the call waits there and the word stands, as
/// a read under that turn builds every object by one word.
#[test]
fn the_overload_switch_takes_the_holders_turn_ahead_of_its_word() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: None,
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    replica.write(facts(A, 1));
    let (objects, store) = (&replica.objects, replica.store.as_ref());
    let (stated, state) = mpsc::channel();
    let turn = objects.turn.lock();
    thread::scope(|scope| {
        let call = scope.spawn(move || {
            let laid = objects.state_overload_switch(store, false, &|| true);
            let _ = stated.send(());
            laid
        });
        assert_eq!(
            state.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        assert!(objects.defaults_on.load(Relaxed), "the word waits");
        drop(turn);
        assert_eq!(call.join().expect("the call ends"), 1);
    });
    assert!(!objects.defaults_on.load(Relaxed), "the word moved");
}

/// A default that states one signal alone changes the rows that take
/// that signal, and no other: a row that names it keeps its object
/// under either word, whatever it leaves out of the other signal.
#[test]
fn the_overload_switch_leaves_a_row_that_takes_no_signal_of_the_default() {
    let count = Overload {
        kv_usage_permille: None,
        waiting_requests: Some(3),
    };
    let usage = Overload {
        kv_usage_permille: Some(650),
        waiting_requests: None,
    };
    let stating = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    // Each default with the row that takes it and the row that names it.
    for (defaults, takes, names) in [
        (stating(Some(8), None), usage, count),
        (stating(None, Some(0.8)), count, usage),
    ] {
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
        replica.write(WorkerFacts {
            overload: takes,
            ..facts(A, 1)
        });
        replica.write(WorkerFacts {
            overload: names,
            ..facts(B, 1)
        });
        let (took, named) = (replica.at(A), replica.at(B));
        let stated = named.metadata().overload;
        for on in [false, true] {
            let laid = switch(&replica, on);
            assert_eq!(laid, 1, "{defaults:?} switched to {on}");
            assert!(!Arc::ptr_eq(&took, &replica.at(A)), "{defaults:?}");
            assert!(Arc::ptr_eq(&named, &replica.at(B)), "{defaults:?}");
            assert_eq!(replica.at(B).metadata().overload, stated);
        }
    }
}

/// A row that states the external runtime takes no default: its object
/// states the thresholds its row names and no other, a queue alone or a
/// usage alone, and in any spelling of the runtime's name. An engine's
/// row takes the default, one that names its runtime too. A gateway
/// registers an engine's worker with its default thresholds (its
/// registration workflow's `overload_defaults`) and an external worker
/// with none (`build_external_worker`), so until that worker's first
/// update its monitor judges it by nothing.
#[test]
fn an_external_rows_object_states_no_default_threshold() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    replica.objects = Arc::new(Objects::new(registry).with_overload_thresholds(defaults));
    let of_kind = |url, kind: &str| WorkerFacts {
        engine_kind: kind.to_string(),
        ..facts(url, 1)
    };
    replica.write(of_kind(A, "external"));
    replica.write(WorkerFacts {
        overload: Overload {
            kv_usage_permille: None,
            waiting_requests: Some(3),
        },
        ..of_kind(B, "external")
    });
    replica.write(facts(C, 1));
    replica.write(WorkerFacts {
        overload: Overload {
            kv_usage_permille: Some(500),
            waiting_requests: None,
        },
        ..of_kind(D, "external")
    });
    replica.write(of_kind(E, "vllm"));
    replica.write(of_kind(F, "EXTERNAL"));

    let stated = |url| replica.at(url).metadata().overload;
    let queue = OverloadThresholds {
        waiting_requests: Some(3),
        token_usage: None,
    };
    let usage = OverloadThresholds {
        waiting_requests: None,
        token_usage: Some(0.5),
    };
    let none = OverloadThresholds::default();
    assert_eq!(stated(A), none);
    assert_eq!(stated(B), queue, "an external row states its own alone");
    assert_eq!(stated(D), usage, "and no default queue beside its usage");
    assert_eq!(stated(F), none, "in any spelling of the name");
    assert_eq!(stated(C), defaults, "an engine's row takes the default");
    assert_eq!(stated(E), defaults, "one that names its runtime too");
}

//! The standby's applier: what it applies of its leader's log, what the
//! replica then holds and decides over, the place it asks from, how a
//! snapshot's parts are staged, and what a restore from them leaves. One
//! ignored case measures an apply and a restore.

use super::*;

/// A standby's applier over a replica's store and log, with the holder
/// of its worker objects as a controller keeps one.
fn applier((store, log): &Replica) -> Follower {
    let objects = WorkerObjects::new(Arc::clone(store)).holder();
    Follower::new(Arc::clone(store), Arc::clone(log), objects)
}

/// A leader's store and log under term 1, and a standby's applier over
/// its own.
fn pair() -> (Replica, Follower, Replica) {
    let leader = replica();
    leader.0.set_term(1);
    let standby = replica();
    let follower = applier(&standby);
    (leader, follower, standby)
}

/// A log's entries by place, each as its kind and what it names.
fn told(log: &Log) -> Vec<(Position, String)> {
    entries(log, start(log))
        .into_iter()
        .map(|entry| {
            let what = match entry.event {
                Event::Member { epoch, change, .. } => format!("{epoch} {change:?}"),
                Event::Status {
                    epoch,
                    id,
                    from,
                    to,
                    models,
                } => format!("{epoch} {id:?} {from:?} to {to:?} serving {models:?}"),
                Event::Term => "term".to_owned(),
                Event::Generation { generation } => format!("generation {generation}"),
            };
            (entry.at, what)
        })
        .collect()
}

/// Writes under the first term: registrations, a status change, a
/// relabel and a resize.
fn first(leader: &MemoryStore) {
    let revision = |id| WorkersStore::get(leader, id).expect("the row").revision;
    let a = leader.upsert(facts("http://a:8000", 1)).id();
    leader.upsert(facts("http://b:8000", 1));
    leader.set_status(a, WorkerStatus::Ready, revision(a));
    let mut labelled = facts("http://a:8000", 1);
    labelled.labels.insert("tier".to_owned(), "gold".to_owned());
    leader.upsert(labelled);
    let wide = WorkerFacts {
        dp_size: 2,
        ..facts("http://b:8000", 1)
    };
    leader.upsert(wide);
}

/// What follows: a new life of a ready worker (its status comes over),
/// a removal, the next term's start and a registration under it.
fn rest(leader: &MemoryStore) {
    leader.upsert(facts("http://a:8000", 2));
    let b = leader.by_key("http://b:8000").expect("the row");
    leader.remove(b.id, b.revision);
    leader.set_term(2);
    leader.upsert(facts("http://c:8000", 1));
}

#[test]
fn a_standby_fed_the_leaders_log_holds_its_tables_and_its_log() {
    let ((leader, led), follower, (standby, followed)) = pair();
    // It applied nothing: it names no place of any run, and its log
    // stands where its own run drew.
    assert_eq!(follower.place(), Position::default());
    assert_eq!(start(&followed), followed.head());
    let at = Instant::now();
    first(&leader);
    follow(&follower, &led, at);
    // The join told the store its leader's term: under the first term
    // the standby holds what its leader holds, the term with it.
    assert_eq!(
        format!("{:?}", held(&standby)),
        format!("{:?}", held(&leader))
    );
    rest(&leader);
    follow(&follower, &led, at);
    // Its log starts where its leader's does: it serves a standby that
    // applied nothing from there when it leads.
    assert_eq!(start(&followed), start(&led));
    assert_eq!(
        format!("{:?}", held(&standby)),
        format!("{:?}", held(&leader))
    );
    assert_eq!(told(&followed), told(&led));
    assert_eq!(follower.place(), led.head());
    assert!(told(&led).len() > 8, "{:?}", told(&led));
}

/// The holder of `follower`'s worker objects has the object of every
/// row `store` holds and of no other, each stating its row.
fn each_row_has_its_object(follower: &Follower, store: &MemoryStore) {
    let (_, rows) = store.all_rows();
    assert!(!rows.is_empty());
    for row in &rows {
        let object = follower.objects.object(row.id);
        let object = object.expect("the row's object");
        assert_eq!(follower.objects.row_of(&object).as_ref(), Some(row));
    }
    assert_eq!(follower.objects.len(), rows.len(), "an object a row");
}

/// The applier lays what it applies. Behind each batch the holder of
/// the standby's worker objects has the object of every row its table
/// holds and of no other, each stating its row, with no follow.
#[test]
fn a_standby_fed_the_leaders_log_holds_each_rows_object() {
    let ((leader, led), follower, (standby, _)) = pair();
    let at = Instant::now();
    first(&leader);
    follow(&follower, &led, at);
    each_row_has_its_object(&follower, &standby);
    rest(&leader);
    follow(&follower, &led, at);
    each_row_has_its_object(&follower, &standby);
}

/// A restore lays what it restores: the cut's rows have their objects
/// when it returns, with no follow.
#[test]
fn a_standby_restored_from_the_snapshot_holds_each_rows_object() {
    let ((leader, _), follower, (standby, _)) = pair();
    first(&leader);
    assert_eq!(follower.restore(leader.checkpoint(), None), Ok(true));
    each_row_has_its_object(&follower, &standby);
}

/// The pair classes `objects` derives of the model's two lists under
/// `mode`, each list's by place.
fn pair_classes(objects: &Objects<Members>, mode: Pairing) -> [Vec<u32>; 2] {
    let generation = objects.registry().model_generation("m");
    let generation = generation.expect("the model's generation");
    let classes = objects.pair_classes(&generation, mode);
    [classes.prefills().to_vec(), classes.decodes().to_vec()]
}

/// A standby pairs as its leader does. A row that states no pairing
/// descriptor pairs by what its worker object offers, and each replica
/// builds a row's object of the row alone. So behind the leader's log
/// the standby's holder derives the classes a holder over the leader's
/// own table derives, under each mode, and neither pairs two workers
/// whose labels name two pairing protocols.
#[test]
fn a_standby_derives_its_leaders_pair_classes_of_the_rows_it_followed() {
    let ((leader, led), follower, _) = pair();
    let side = |key: &str, role: Role, protocol: &str| WorkerFacts {
        role,
        labels: [("pairing_protocol".to_owned(), protocol.to_owned())].into(),
        ..facts(key, 1)
    };
    leader.upsert(side("http://pa:8000", Role::Prefill, "blue"));
    leader.upsert(side("http://pb:8000", Role::Prefill, "green"));
    leader.upsert(side("http://da:8000", Role::Decode, "green"));
    follow(&follower, &led, Instant::now());
    let leading = WorkerObjects::new(Arc::clone(&leader)).holder();
    lay_standing(&leading, &leader);
    for mode in [Pairing::Lenient, Pairing::Strict] {
        let behind = pair_classes(&follower.objects, mode);
        assert_eq!(behind, pair_classes(&leading, mode), "{mode:?}");
    }
    let generation = follower.objects.registry().model_generation("m");
    let generation = generation.expect("the model's generation");
    let lenient = follower.objects.pair_classes(&generation, Pairing::Lenient);
    assert_eq!([lenient.pairs(0, 0), lenient.pairs(1, 0)], [false, true]);
}

/// What the applier's way through the holder is for: a replica that
/// takes the lead right behind an applied event decides over that
/// event's worker. A decision over the holder's lists names the worker
/// its leader registered and made ready, with no follow run.
#[test]
fn a_replica_that_takes_the_lead_decides_over_its_last_applied_event() {
    let ((leader, led), follower, (standby, _)) = pair();
    let id = leader.upsert(facts("http://a:8000", 1)).id();
    leader.set_status(id, WorkerStatus::Ready, 1);
    follow(&follower, &led, Instant::now());
    // The standby takes the lead, and applies a document of its own.
    standby.set_term(2);
    let applied = standby.apply(preset("round_robin", 1));
    applied.expect("above the unbound start");
    let named = lent_pick(&standby, &follower.objects, frame_for("m", 1, 1));
    assert_eq!(named, Some(id));
}

/// The cut a standby stages from the `parts` of a snapshot, read at
/// `at`.
fn staged(parts: &[proto::ReplicationBatch], at: Instant) -> StoreCheckpoint {
    let mut staged = Staged::default();
    let mut cut = None;
    for part in parts {
        assert!(cut.is_none(), "a part after the last");
        cut = staged.take(part, at).expect("the snapshot's next part");
    }
    cut.expect("the last part").0
}

/// A load row as an older leader's stream told it: a leader of this
/// build sends none.
fn told_load(id: WorkerId) -> proto::ReplicationEvent {
    proto::ReplicationEvent {
        event: Some(Sent::Load(proto::LoadSample {
            worker_id: id.0,
            load: Some(proto::LoadRow {
                running: 3,
                ..proto::LoadRow::default()
            }),
            ..proto::LoadSample::default()
        })),
    }
}

#[test]
fn a_load_row_an_older_leader_tells_takes_its_place_and_writes_nothing() {
    let ((leader, led), follower, (standby, followed)) = pair();
    let at = Instant::now();
    let a = leader.upsert(facts("http://a:8000", 1)).id();
    leader.upsert(facts("http://b:8000", 1));
    let sent = entries(&led, start(&led));
    let [term, first, second] = &sent[..] else {
        panic!("the term's start and two registrations, not {sent:?}");
    };
    // An older leader's history: a load row of the first worker stands
    // between the two registrations, at a place of its own.
    let event = |entry: &Entry| worker_wire::replication(entry);
    let older = proto::ReplicationBatch {
        first_seq: term.at.seq,
        term: term.at.term,
        events: vec![event(term), event(first), told_load(a), event(second)],
        resync_required: false,
        snapshot: None,
        head_seq: term.at.seq + 3,
    };
    assert_eq!(follower.apply(&older, at), Ok(()));
    // The standby went on with the next watched event: it holds both
    // rows.
    assert_eq!(standby.all_rows().1.len(), 2);
    // The load row's place is journaled alone, so the next event
    // followed it and the standby stands at the batch's last place.
    let last = Position {
        seq: term.at.seq + 3,
        term: term.at.term,
    };
    assert_eq!(follower.place(), last);
    let kinds: Vec<String> = told(&followed).into_iter().map(|(_, what)| what).collect();
    assert_eq!(kinds[2], "term", "{kinds:?}");
    assert_eq!(kinds.len(), 4, "{kinds:?}");
}

#[test]
fn a_restores_load_part_is_read_past() {
    let (leader, _follower, (standby, _followed)) = pair();
    let at = Instant::now();
    let a = leader.0.upsert(facts("http://a:8000", 1)).id();
    // An older leader's snapshot: its last part carries a load row, and
    // one this build could not read were it read.
    let mut parts = snapshot(&leader, at);
    let last = parts.last_mut().and_then(|batch| batch.snapshot.as_mut());
    let last = last.expect("the snapshot's one part");
    let mut unread = told_load(a);
    if let Some(Sent::Load(load)) = &mut unread.event {
        load.dp_rank = 65_536;
        last.loads.push(*load);
    }
    assert_eq!(last.loads.len(), 1);
    let cut = staged(&parts, at);
    assert_eq!(cut.rows.len(), 1);
    assert!(standby.restore_from(cut));
    assert_eq!(standby.all_rows().1.len(), 1);
}

#[test]
fn a_standby_restored_from_the_snapshot_continues_from_its_place() {
    let (leader, follower, (standby, _followed)) = pair();
    let at = Instant::now();
    first(&leader.0);
    // The standby held a row of its own, under no term.
    standby.upsert(facts("http://stale:8000", 1));
    let first = snapshot(&leader, at);
    assert_eq!(follower.restore(staged(&first, at), None), Ok(true));
    assert_eq!(follower.place(), leader.1.head());
    assert_eq!(held(&standby), held(&leader.0));
    rest(&leader.0);
    follow(&follower, &leader.1, at);
    assert_eq!(held(&standby), held(&leader.0));
    // A snapshot of the term it followed before is a history it left:
    // refused, the store as it was.
    let place = follower.place();
    assert_eq!(follower.restore(staged(&first, at), None), Ok(false));
    assert_eq!(follower.place(), place);
    assert_eq!(held(&standby), held(&leader.0));
}

#[test]
fn a_batch_that_is_not_its_snapshots_next_part_is_refused_by_its_path() {
    let (leader, ..) = pair();
    let at = Instant::now();
    first(&leader.0);
    a_page_and_one(&leader.0);
    let parts = snapshot(&leader, at);
    assert_eq!(parts.len(), 2, "two parts of rows");
    let events = batch(&entries(&leader.1, start(&leader.1)).remove(0));
    let next = parts[1].clone();
    let part = next.snapshot.clone().expect("a part");
    // Each after the snapshot's first part, which is staged.
    let refused = |batch: proto::ReplicationBatch| {
        let mut staged = Staged::default();
        assert!(matches!(staged.take(&parts[0], at), Ok(None)));
        let refused = staged.take(&batch, at).expect_err("not the next part");
        (refused.path, refused.why)
    };
    let none = (
        "batch.snapshot",
        "no part of the snapshot the leader announced",
    );
    assert_eq!(refused(events.clone()), none);
    assert_eq!(refused(proto::ReplicationBatch::default()), none);
    let carried = proto::ReplicationBatch {
        events: events.events,
        ..next.clone()
    };
    let why = "events in a part of a snapshot";
    assert_eq!(refused(carried), ("batch.events", why));
    let why = "a part out of its snapshot's order";
    assert_eq!(refused(parts[0].clone()), ("batch.snapshot.part", why));
    let elsewhere = [
        proto::ReplicationBatch {
            first_seq: next.first_seq + 1,
            ..next.clone()
        },
        proto::ReplicationBatch {
            term: next.term + 1,
            ..next.clone()
        },
        proto::ReplicationBatch {
            snapshot: Some(proto::ReplicationSnapshot {
                membership_epoch: part.membership_epoch + 1,
                ..part
            }),
            ..next.clone()
        },
    ];
    let why = "a part at another place than its snapshot's";
    for batch in elsewhere {
        assert_eq!(refused(batch), ("batch.first_seq", why));
    }
    // A first part that says position zero stands at no place.
    let nowhere = proto::ReplicationBatch {
        first_seq: 0,
        ..parts[0].clone()
    };
    let refused = Staged::default().take(&nowhere, at).expect_err("no place");
    assert_eq!(
        (refused.path, refused.why),
        ("batch.first_seq", "a snapshot at no place")
    );
}

/// A part the leader did not announce is no heartbeat and no batch of
/// events: the applier refuses it by its path with nothing applied,
/// the place and the tables as they were.
#[test]
fn a_part_the_leader_did_not_announce_is_refused_by_its_path() {
    let (leader, follower, (standby, _followed)) = pair();
    let at = Instant::now();
    first(&leader.0);
    let parts = snapshot(&leader, at);
    let (place, before) = (follower.place(), held(&standby));
    let why = "a part of a snapshot the leader did not announce";
    for part in &parts {
        let Err(Halt::Unread(refused)) = follower.apply(part, at) else {
            panic!("a part with no word before it was applied");
        };
        assert_eq!((refused.path, refused.why), ("batch.snapshot", why));
    }
    assert_eq!((follower.place(), held(&standby)), (place, before));
}

#[test]
fn a_standby_that_asks_names_its_heads_number_past_zero_under_no_term() {
    // A log whose places start at the start: its head is zero under no
    // term, the place of a standby that applied nothing and no ask.
    let (store, feed) = journaled_from(&LeasePlace::File("shared".into()), 0);
    let follower = applier(&(store, Arc::clone(&feed.log)));
    assert_eq!(follower.place(), Position::default());
    assert_eq!(follower.asking(), Position { seq: 1, term: 0 });
    // Past it, the head's own number.
    let leader = led();
    follow(&follower, &leader.1, Instant::now());
    let head = follower.place();
    assert!(head.seq > 1 && head.term == 1, "{head:?}");
    assert_eq!(follower.asking(), Position { term: 0, ..head });
}

#[test]
fn an_event_that_does_not_land_halts_the_batch_where_it_stands() {
    let ((leader, led), follower, (standby, _followed)) = pair();
    let at = Instant::now();
    let a = leader.upsert(facts("http://a:8000", 1)).id();
    let revision = WorkersStore::get(&*leader, a).expect("the row").revision;
    leader.set_status(a, WorkerStatus::Ready, revision);
    let sent = entries(&led, start(&led));
    let [term, registered, readied] = &sent[..] else {
        panic!("the term's start, a registration and a status change, not {sent:?}");
    };
    // A history starts with its term's start: the registration before
    // it is no place a standby that applied nothing joins at.
    let lost = Err(Halt::Lost(registered.at));
    assert_eq!(follower.apply(&batch(registered), at), lost);
    assert_eq!(follower.place(), Position::default());
    assert_eq!(follower.apply(&batch(term), at), Ok(()));
    // The term's start again: a replay, and no second join.
    let replay = Err(Halt::Lost(term.at));
    assert_eq!(follower.apply(&batch(term), at), replay);
    // A status change of a worker the standby does not hold.
    let stray = proto::ReplicationBatch {
        first_seq: registered.at.seq,
        ..batch(readied)
    };
    assert_eq!(follower.apply(&stray, at), lost);
    // An event of no kind, and one of a kind this build does not apply.
    for event in [None, Some(Sent::Health(proto::HealthRow::default()))] {
        let unread = proto::ReplicationBatch {
            events: vec![proto::ReplicationEvent { event }],
            ..batch(registered)
        };
        assert!(matches!(
            follower.apply(&unread, at),
            Err(Halt::Unread(refused)) if refused.path == "batch.events"
        ));
    }
    // A row this build cannot read is refused by its path.
    let mut unknown = batch(registered);
    if let Some(Sent::Membership(member)) = &mut unknown.events[0].event {
        if let Some(proto::worker_event::Event::Registered(row)) = &mut member.event {
            row.status = 99;
        }
    }
    assert!(matches!(
        follower.apply(&unknown, at),
        Err(Halt::Unread(refused)) if refused.path == "row.status"
    ));
    // None of them was applied: the standby stands at the term's start.
    assert_eq!((follower.place(), standby.epoch()), (term.at, 0));
    // The first event of a batch lands and the second does not fit:
    // the first stands at its place.
    let twice = proto::ReplicationBatch {
        events: vec![worker_wire::replication(registered); 2],
        ..batch(registered)
    };
    assert_eq!(follower.apply(&twice, at), Err(Halt::Lost(readied.at)));
    assert_eq!((follower.place(), standby.epoch()), (registered.at, 1));
}

/// A measurement, run by name with `--ignored --nocapture` in the
/// release profile: what the applier costs a status change in batches
/// of 256 as the leader sends them, and what a restore from the
/// snapshot's parts costs at 512 and 4,096 workers. The wire's own
/// decode is the transport's and is not in it.
#[test]
#[ignore = "a measurement: run by name"]
fn the_standbys_numbers() {
    use io::Write as _;
    let mut out = io::stdout();
    for workers in [512u32, 4_096] {
        let (leader, follower, _standby) = pair();
        let at = Instant::now();
        let ids: Vec<_> = (0..workers)
            .map(|n| leader.0.upsert(facts(&format!("http://w{n}:8000"), 1)).id())
            .collect();
        let parts = snapshot(&leader, at);
        let mut micros: Vec<u128> = (0..21)
            .map(|_| {
                let began = Instant::now();
                assert_eq!(
                    follower.restore(staged(&parts, at), None),
                    Ok(true),
                    "a restore"
                );
                began.elapsed().as_micros()
            })
            .collect();
        micros.sort_unstable();
        let bytes: usize = parts.iter().map(prost::Message::encoded_len).sum();
        writeln!(
            out,
            "RESTORE workers={workers} parts={} bytes={bytes} median_us={}",
            parts.len(),
            micros[10]
        )
        .expect("stdout");
        if workers != 512 {
            continue;
        }
        // The tail after the snapshot: 3,000 status changes, in
        // batches of 256.
        let place = follower.place();
        let id = ids[0];
        for turn in 1..=3_000 {
            let status = if turn % 2 == 1 {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            };
            let row = WorkersStore::get(&*leader.0, id).expect("the row");
            leader.0.set_status(id, status, row.revision);
        }
        let sent = entries(&leader.1, place);
        assert_eq!(sent.len(), 3_000, "each write is one event");
        let batches: Vec<_> = sent
            .chunks(256)
            .map(|chunk| proto::ReplicationBatch {
                first_seq: chunk[0].at.seq,
                term: chunk[0].at.term,
                events: chunk.iter().map(worker_wire::replication).collect(),
                resync_required: false,
                snapshot: None,
                head_seq: 0,
            })
            .collect();
        let began = Instant::now();
        for batch in &batches {
            follower.apply(batch, at).expect("a batch that lands");
        }
        let nanos = began.elapsed().as_nanos();
        writeln!(
            out,
            "APPLY kind=status events={} ns_an_event={}",
            sent.len(),
            nanos / sent.len() as u128
        )
        .expect("stdout");
    }
}

/// A standby's applier over its own store and log, through `door`.
fn through((store, log): &Replica, door: &Arc<Generations>) -> Follower {
    applier(&(Arc::clone(store), Arc::clone(log))).through(Arc::clone(door))
}

/// One entry as a leader's stream sends it: a generation with the
/// document `documents` holds for it at the send.
fn sent(documents: &Generations, entry: &Entry) -> proto::ReplicationBatch {
    let mut batch = batch(entry);
    if let Event::Generation { generation } = entry.event {
        batch.events[0].event = Some(Sent::Generation(documents.sent(generation)));
    }
    batch
}

#[test]
fn a_standby_applies_its_leaders_generations_through_its_own_door() {
    use crate::service::controller::{
        generations::{
            tests::{LEAST_LOAD, RANDOM},
            Format,
        },
        wire::spec_document,
    };

    let (leader, _, standby) = pair();
    let (leading, following) = (door(&leader), door(&standby));
    let follower = through(&standby, &following);
    let at = Instant::now();
    let apply = |document: &str| leading.apply(document.into(), Format::Yaml);
    apply(RANDOM).expect("a valid document");
    leader.0.upsert(facts("http://a:8000", 1));
    apply(LEAST_LOAD).expect("a changed document");
    // An event a batch, each generation with the document the leader's
    // door holds at the send: the first stands no longer.
    let led = entries(&leader.1, start(&leader.1));
    for entry in &led {
        let batch = sent(&leading, entry);
        assert_eq!(follower.apply(&batch, at), Ok(()), "{entry:?}");
    }
    assert_eq!(told(&standby.1), told(&leader.1), "one history");
    assert_eq!(held(&standby.0), held(&leader.0));
    assert_eq!(
        spec_document(following.document(0).expect("held")),
        spec_document(leading.document(0).expect("held"))
    );
    assert_eq!(following.apply_counts().applied, 1, "the one that stood");

    // An applier with no door does not apply a generation: its standby
    // leaves the synced state at that place, the term's start applied.
    let other = replica();
    let bare = applier(&other);
    assert_eq!(bare.apply(&batch(&led[0]), at), Ok(()));
    assert_eq!(
        bare.apply(&sent(&leading, &led[1]), at),
        Err(Halt::Unread(UNAPPLIED))
    );
}

#[test]
fn a_snapshot_restores_its_leaders_standing_generation_or_nothing() {
    use crate::service::controller::{
        generations::{tests::LEAST_LOAD, Format},
        wire::spec_document,
    };

    let (leader, _, standby) = pair();
    let (leading, following) = (door(&leader), door(&standby));
    let follower = through(&standby, &following);
    let at = Instant::now();
    first(&leader.0);
    a_page_and_one(&leader.0);
    leading
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let told = leading.standing_sent();
    // The first of two parts states the generation, and the last hands it
    // on with the cut.
    let mut parts = snapshot(&leader, at);
    assert_eq!(parts.len(), 2, "two parts of rows");
    parts[0].snapshot.as_mut().expect("a part").generation = told.clone().map(Box::new);
    let stage = || {
        let mut staging = Staged::default();
        let mut whole = None;
        for part in &parts {
            whole = staging.take(part, at).expect("the snapshot's next part");
        }
        whole.expect("the last part")
    };
    let (cut, stated) = stage();
    assert_eq!(stated, told);
    // A document this replica does not read: nothing is restored.
    let mut unread = told.clone().expect("a held document");
    unread.document.as_mut().expect("a document").format = "toml".to_owned();
    let before = follower.place();
    let refused = follower.restore(cut, Some(&unread));
    assert_eq!(
        refused.expect_err("no such form").path,
        "generation.document.format"
    );
    assert_eq!(follower.place(), before);
    assert_eq!(following.generation(), 0);
    // The generation lands behind the restore, at no place of its own.
    assert_eq!(follower.restore(stage().0, stated.as_ref()), Ok(true));
    assert_eq!(follower.place(), leader.1.head());
    assert_eq!(held(&standby.0), held(&leader.0));
    assert_eq!(
        spec_document(following.document(0).expect("held")),
        spec_document(leading.document(0).expect("held"))
    );
}

/// A snapshot of one part is whole at that part: the part that states
/// its leader's standing generation is the first and the last, and hands
/// it on with the cut.
#[test]
fn a_snapshot_of_one_part_hands_on_the_generation_its_part_states() {
    use crate::service::controller::generations::{tests::LEAST_LOAD, Format};

    let (leader, _, _) = pair();
    let leading = door(&leader);
    let at = Instant::now();
    first(&leader.0);
    leading
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let told = leading.standing_sent().expect("a held document");
    let mut parts = snapshot(&leader, at);
    assert_eq!(parts.len(), 1, "one part of rows");
    parts[0].snapshot.as_mut().expect("a part").generation = Some(Box::new(told.clone()));
    let (_, stated) = Staged::default()
        .take(&parts[0], at)
        .expect("the snapshot's one part")
        .expect("the last part");
    assert_eq!(stated, Some(told));
}

/// A snapshot whose parts state no generation, as a leader that holds no
/// document sends it, hands on none with the cut.
#[test]
fn a_snapshot_that_states_no_generation_hands_on_none() {
    let (leader, _, _) = pair();
    let at = Instant::now();
    first(&leader.0);
    let parts = snapshot(&leader, at);
    assert_eq!(parts.len(), 1, "one part of rows");
    let (_, stated) = Staged::default()
        .take(&parts[0], at)
        .expect("the snapshot's one part")
        .expect("the last part");
    assert_eq!(stated, None);
}

/// Two parts of one snapshot that each state a generation: the last part
/// hands on the one stated first. A leader of this build states its
/// generation on the first part alone.
#[test]
fn a_later_parts_generation_does_not_replace_the_one_stated_first() {
    use crate::service::controller::generations::{tests::LEAST_LOAD, Format};

    let (leader, _, _) = pair();
    let leading = door(&leader);
    let at = Instant::now();
    first(&leader.0);
    a_page_and_one(&leader.0);
    leading
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    let told = leading.standing_sent().expect("a held document");
    let mut later = told.clone();
    later.status.as_mut().expect("a status").generation += 1;
    let mut parts = snapshot(&leader, at);
    assert_eq!(parts.len(), 2, "two parts of rows");
    parts[0].snapshot.as_mut().expect("a part").generation = Some(Box::new(told.clone()));
    parts[1].snapshot.as_mut().expect("a part").generation = Some(Box::new(later));
    let mut staging = Staged::default();
    assert!(matches!(staging.take(&parts[0], at), Ok(None)));
    let (_, stated) = staging
        .take(&parts[1], at)
        .expect("the snapshot's next part")
        .expect("the last part");
    assert_eq!(stated, Some(told));
}

/// A restore lands the snapshot's generation behind the cut's rows: the
/// rows are built under the word that stood, and the replica's door
/// then states the generation's overload switch on the holder. Under a
/// document that switches the default off, the restore returns with
/// every row that takes a default under the document's word: the row
/// that stood, laid again with its id, its status and its count in
/// flight, and the row the cut brought.
#[test]
fn a_restore_leaves_every_row_under_its_generations_overload_switch() {
    use smg_worker::{
        health::overload::OverloadThresholds, registry::WorkerRegistry,
        worker::WorkerStatus as ObjectStatus,
    };

    use crate::service::controller::generations::{
        tests::{election, switched},
        Format,
    };

    let (a, b) = ("http://a:8000", "http://b:8000");
    let start = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let none = OverloadThresholds::default();
    let (leader, _, standby) = pair();
    let leading = door(&leader);
    // The standby's holder states a start's thresholds, and its door
    // states each generation's word on it.
    let objects = Objects::new(Arc::new(WorkerRegistry::new())).with_overload_thresholds(start);
    let objects = Arc::new(objects);
    let catalog = Arc::new(registry::StageRegistry::builtin());
    let (store, log) = (Arc::clone(&standby.0), Arc::clone(&standby.1));
    // Under a shared lease, and it has not led: its door writes no row.
    let elector = election();
    let fence = Some(elector.fence());
    let following = Generations::stating(Arc::clone(&store), catalog, Arc::clone(&objects), fence);
    let follower = Follower::new(store, log, Arc::clone(&objects)).through(Arc::new(following));
    let object = |url: &str| objects.registry().get_by_url(url).expect("an object");
    let at = Instant::now();

    // A first restore under no generation: the row takes the start's.
    let row = leader.0.upsert(facts(a, 1)).id();
    let revision = WorkersStore::get(&*leader.0, row)
        .expect("the row")
        .revision;
    leader.0.set_status(row, WorkerStatus::Ready, revision);
    let cut = staged(&snapshot(&leader, at), at);
    assert_eq!(follower.restore(cut, None), Ok(true));
    assert_eq!(object(a).metadata().overload, start);
    let id = objects.registry().get_id_by_url(a);
    object(a).increment_load();

    // The leader runs a document that switches the default off, and
    // takes a second row: the next restore brings both.
    leading
        .apply(switched(Some(false), Some(false)).into(), Format::Yaml)
        .expect("every binding off");
    leader.0.upsert(facts(b, 1));
    let told = leading.standing_sent();
    let cut = staged(&snapshot(&leader, at), at);
    assert_eq!(follower.restore(cut, told.as_ref()), Ok(true));
    assert_eq!(object(a).metadata().overload, none);
    assert_eq!(object(b).metadata().overload, none);
    assert_eq!(objects.registry().get_id_by_url(a), id, "the same id");
    let running = (object(a).status(), object(a).load());
    assert_eq!(running, (ObjectStatus::Ready, 1), "its running state");
}

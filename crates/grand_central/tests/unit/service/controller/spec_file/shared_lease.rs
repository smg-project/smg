//! Under a shared lease: the start's document and the lead that applies
//! it, and the look of a replica that follows. Also the looks behind a
//! take, a lost lead or a renewal, and a take past its window.

use super::*;

/// [`bed`] whose generations are built under `fence`, as a replica's
/// under a shared lease: its file's start passes the document and
/// applies none.
fn bed_under(document: &[u8], fence: Arc<Fence>) -> Bed {
    bed_of(document, AppliedSpec::validate_digested, Some(fence))
}

/// The watch of a replica under a shared lease looks whether its
/// replica leads or follows. While the replica follows, a look shows
/// the door what the file became and applies nothing: the log writes at
/// its door. Behind the replica's take the file is read again, though
/// it never moved since: its document never stood there, and it is
/// applied as a leader's is. Back under a leader, the watch looks on
/// and applies nothing.
#[tokio::test(start_paused = true)]
async fn the_watch_of_a_replica_that_follows_looks_and_applies_nothing() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    let file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    follow(&bed, &[RANDOM, &bound_to(1)]);
    let looks = Arc::new(AtomicUsize::new(0));
    let counted = {
        let looks = Arc::clone(&looks);
        move |path: &Path| {
            let n = looks.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(n <= 20, "the loop looks without end");
            look(path)
        }
    };
    let (stop, stopped) = watch::channel(false);
    let mut watching = JoinSet::new();
    watching.spawn(file.run_looking(stopped, counted));

    // Between two polls, so the take and a look never share an instant.
    replace(&bed.path, RANDOM.as_bytes());
    sleep(9 * POLL / 2).await;
    // Four polls: the look that sees the file moved, the read between
    // two looks, and two looks that find the file as it was shown.
    assert_eq!(looks.load(Ordering::SeqCst), 5, "it looks while it follows");
    assert_eq!(bed.generation(), 2, "and applies nothing");
    assert_eq!(bed.rejected(), 0);

    // The take: the file stands as it was shown, and its document never
    // stood here. One look that settles it, then the read between two
    // looks.
    elector.tick(Instant::now()).expect("a take");
    sleep(2 * POLL).await;
    assert_eq!(looks.load(Ordering::SeqCst), 8);
    assert_eq!(bed.generation(), 3, "applied as a leader's file is");
    assert_eq!(bed.answered().2, RANDOM.as_bytes());

    assert!(elector.release().expect("a release"), "it held the lease");
    follow(&bed, &[RANDOM, &bound_to(1), &bound_to(2), &bound_to(3)]);
    replace(&bed.path, LEAST_LOAD.as_bytes());
    sleep(5 * POLL).await;
    assert_eq!(
        looks.load(Ordering::SeqCst),
        14,
        "it looks on once it follows"
    );
    assert_eq!(bed.generation(), 4, "and applies nothing");

    stop.send_replace(true);
    ended(&mut watching).await.expect("the loop's last join");
}

/// `bed` follows the standing generation of a leader that applied
/// `documents` in their order, as a stream that begins now brings it.
fn follow(bed: &Bed, documents: &[&str]) {
    follow_at(&bed.generations, documents);
}

/// [`follow`] on a door.
fn follow_at(door: &Generations, documents: &[&str]) {
    let leader = bed(b"");
    for document in documents {
        let applied = leader
            .generations
            .apply(document.as_bytes().to_vec(), Format::Yaml);
        applied.expect("a document that runs");
    }
    let told = leader.generations.standing_sent().expect("it stands");
    let following = door.following(&told);
    following
        .and_then(|following| following.land(None))
        .expect("above no generation");
}

/// The start of a replica under a shared lease passes its file and
/// applies nothing: the document is applied under the replica's first
/// lead, here at its take, where the replica followed nothing. Behind
/// a take the watch reads the file as a start does, though it never
/// moved, while its document has not stood on the replica since the
/// file took it: a replica that followed another document applies its
/// own file's as the next generation. A file whose document stood
/// there since, applied or told, is a mount that lags: it is left for
/// its next move. A file that does not run refuses this start as it
/// refuses the other.
#[test]
fn a_start_under_a_shared_lease_passes_its_file_and_the_first_take_applies_it() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    assert_eq!(bed.generation(), 0, "passed, not applied");
    assert_eq!(file.poll(), Look::Same, "no take, and the file never moved");
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 1);
    assert_eq!(file.poll(), Look::Same, "the file's document stands");

    // The rollout: a replica that starts over a new file follows its
    // old leader's document. The take applies none of the start's
    // bytes, and behind it the file is read though it never moved: the
    // new document stands as the next generation.
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[RANDOM]);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 1, "the leader's, and none of the start's");
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    // Behind its next take the file's document stands: no read.
    bed.generations.at_take();
    assert_eq!(file.poll(), Look::Same);
    // A file that moved is read at any look.
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));

    // A mount that lags: a standby's file holds the document it was
    // told at its join, and its leader applied another since. The take
    // brings nothing older back: the file is left for its next move.
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD]);
    follow(&bed, &[LEAST_LOAD, RANDOM]);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(file.poll(), Look::Same, "an older document of its leader's");
    assert_eq!((bed.generation(), bed.answered().2), (2, RANDOM.into()));
    replace(&bed.path, RANDOM.as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Unchanged(2)));

    let mut elector = election();
    let unrun = LEAST_LOAD.replace("healthy", "no-such-stage");
    let unrun = bed_under(unrun.as_bytes(), elector.fence());
    let refused = SpecFile::start(unrun.path.clone(), Arc::clone(&unrun.generations));
    assert!(refused.is_err(), "a stage this build lacks");
    elector.tick(Instant::now()).expect("a take");
    unrun.generations.at_take();
    assert_eq!(unrun.generation(), 0);
}

/// A file that went bad between the start and the first lead leaves the
/// start's document. The watch hands that document first, and the bad
/// file is one rejected state behind it.
#[test]
fn a_file_gone_bad_since_the_start_leaves_the_starts_document() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    replace(&bed.path, b"{ profiles");
    // A take with no step: the start's document waits under an open
    // fence, and the watch's look hands it.
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 0);
    let read = settle(&mut file);
    assert!(read.is_err(), "the bad file is refused");
    assert_eq!(bed.generation(), 1, "the start's document stands");
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    assert_eq!(bed.rejected(), 1, "the bad file is one rejected state");
    assert_eq!(file.poll(), Look::Same, "told once");
}

/// A start's document that its take's pass refuses is counted and let
/// go, and the mark of a document awaited stays: no decision is made.
/// The watch reads the file behind the take and lands its document, and
/// the mark goes.
#[test]
fn a_starts_document_refused_at_its_take_is_landed_by_the_next_look() {
    thread_local! {
        static PASSES: Cell<u32> = const { Cell::new(0) };
    }
    // The start's pass runs, the take's faults, and every later one
    // runs.
    fn faulting_once(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        let passed = PASSES.with(|passes| passes.replace(passes.get() + 1));
        assert_ne!(passed, 1, "a fault of the validator");
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let dir = tempfile::tempdir().expect("a directory");
    let path = dir.path().join("policy.yaml");
    fs::write(&path, LEAST_LOAD).expect("the file");
    let store = Arc::new(MemoryStore::new());
    let catalog = Arc::new(StageRegistry::builtin());
    let mut elector = election();
    let fence = Some(elector.fence());
    let door = Generations::validating(Arc::clone(&store), catalog, faulting_once, fence);
    let door = Arc::new(door);
    let awaits = || store.snapshot().spec().awaits_document();
    let mut file = SpecFile::start(path, Arc::clone(&door)).expect("a file that runs");
    assert!(awaits(), "the start's document waits");
    let stepped = Arc::clone(&door);
    elector.at_take(move || stepped.at_take());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(door.generation(), 0, "refused at the take");
    assert_eq!(door.apply_counts().rejected, 1);
    assert!(awaits(), "the mark stays");
    assert!(door.bound(), "the door keeps its source");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(1)));
    assert!(!awaits(), "the mark went with the unbound start");
}

/// A look that resumes behind a lost lead applies nothing: it is the
/// look of a replica that follows, and shows the door the file. The
/// looks of the replica's next lead read the state again, behind a
/// take or behind none: its document has not stood there since.
#[test]
fn a_look_that_resumes_behind_a_lost_lead_shows_the_file_and_applies_nothing() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 1, "the start's document, at the take");
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    // The replica is deposed here and follows its next leader, whose
    // generation is above its own.
    assert!(elector.release().expect("a release"), "it held the lease");
    follow(&bed, &[RANDOM, &bound_to(2)]);
    assert_eq!(bed.generation(), 2);
    // The look resumes: the door is shown the file and applies none of
    // it.
    assert_eq!(file.poll(), Look::Shown);
    assert_eq!(file.poll(), Look::Same, "shown once");
    assert_eq!(bed.generation(), 2);
    assert_eq!(bed.answered().2, bound_to(2).as_bytes());
    assert_eq!(bed.rejected(), 0);
    // Behind its next take the state is read again: its document has
    // not stood here.
    elector.tick(Instant::now()).expect("a take");
    assert!(bed.generations.source_writes(), "it leads again");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
    assert_eq!(bed.answered().2, bound_to(1).as_bytes());

    // A lead comes back with no take as well: a fence closes by itself
    // at the renew deadline, and the holder's next renewal opens it
    // again under the same term. A take whose write returned past its
    // window shows no term: it publishes nothing, and the start's
    // document waits on. A look in between is a follower's, and shows
    // the file. Behind the renewal that opens the fence the watch
    // brings both: the start's document, then the file as it was shown.
    let deadline = Duration::from_secs(1);
    let mut elector = election_under(deadline);
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    // The take's write lands two deadlines late: its fence is closed.
    let late = Instant::now()
        .checked_sub(2 * deadline)
        .expect("a host that is up for two seconds");
    elector.tick(late).expect("a take");
    assert!(!bed.generations.source_writes(), "past its deadline");
    assert_eq!(bed.generation(), 0, "no term showed at the take's step");
    assert!(bed.generations.bound(), "the door keeps its source");
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    assert_eq!(file.poll(), Look::Shown);
    assert_eq!(bed.generation(), 0, "nothing is applied");
    // A round whose window closed again ahead of its read waits, or
    // finds the file as it was shown.
    let read = (0..8).find_map(|_| {
        let renewed = elector.tick(Instant::now());
        assert!(
            matches!(renewed, Ok(TickOutcome::Renewed { .. })),
            "a renewal"
        );
        assert!(matches!(file.poll(), Look::Settling | Look::Same));
        Some(file.poll()).filter(|look| matches!(look, Look::Read(_)))
    });
    assert_eq!(read, Some(Look::Read(Ok(ApplyOutcome::Generation(2)))));
    assert_eq!(bed.answered().2, bound_to(1).as_bytes());
    assert_eq!(bed.generations.takes(), 1, "no take since the first");
}

/// A file shown while its leader's fence stood closed is read behind
/// the renewal that opens the fence. No take stands between the two
/// leads, and the leader's own look had read behind its take: the look
/// reads the state again because a follower's look showed it last.
#[test]
fn a_file_shown_under_a_closed_fence_is_read_behind_the_renewal_that_opens_it() {
    let deadline = Duration::from_secs(30);
    let mut elector = election_under(deadline);
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 1, "the start's document, at the take");
    // A leader's look reads behind the take: no later look reads for
    // the take's sake.
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    // A renewal whose write returns two deadlines late stamps a window
    // that has ended: the replica holds its lease and shows no term.
    // The file moves, and the look is a follower's.
    let late = Instant::now()
        .checked_sub(2 * deadline)
        .expect("a host that is up for a minute");
    let renewed = elector.tick(late);
    assert!(
        matches!(renewed, Ok(TickOutcome::Renewed { .. })),
        "a renewal"
    );
    assert!(!bed.generations.source_writes(), "past its deadline");
    shown(&bed, &mut file, bound_to(2).as_bytes());
    // The next renewal opens the fence under the same term, and the
    // look reads the state it was shown: its document has not stood
    // here.
    let renewed = elector.tick(Instant::now());
    assert!(
        matches!(renewed, Ok(TickOutcome::Renewed { .. })),
        "a renewal"
    );
    assert!(bed.generations.source_writes(), "it leads again");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
    assert_eq!(bed.answered().2, bound_to(2).as_bytes());
    assert_eq!(bed.generations.takes(), 1, "no take since the first");
    assert_eq!(file.poll(), Look::Same, "read once");
}

/// The loop sleeps no poll ahead of the start's document where its lead
/// is open: the document is applied at once, by the loop's own clock.
#[tokio::test(start_paused = true)]
async fn the_loop_brings_the_starts_document_with_no_polls_sleep_ahead_of_it() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    // A take with no step of its own: the start's document waits under
    // an open fence.
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 0);
    let (stop, stopped) = watch::channel(false);
    let mut watching = JoinSet::new();
    let began = tokio::time::Instant::now();
    watching.spawn(file.run(stopped));
    while bed.generation() == 0 {
        assert!(began.elapsed() < 20 * POLL, "the loop brings the document");
        sleep(POLL / 100).await;
    }
    assert!(
        began.elapsed() < POLL,
        "ahead of a poll's sleep: {:?}",
        began.elapsed()
    );
    stop.send_replace(true);
    ended(&mut watching).await.expect("the loop's last join");
}

/// Woken short of a poll, at the opening of a lead the start's document
/// waited for, the watch hands the door that document and looks at
/// nothing. Behind a whole poll it looks.
#[test]
fn woken_at_a_leads_opening_the_watch_hands_over_the_starts_document_and_looks_at_nothing() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    // A take with no step of its own: the start's document waits under
    // an open fence.
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 0);
    assert_eq!(bed.generations.nap(POLL), Duration::ZERO);
    let looks = AtomicUsize::new(0);
    let counted = |path: &Path| {
        looks.fetch_add(1, Ordering::SeqCst);
        look(path)
    };
    file.woken(Duration::ZERO, &counted);
    assert_eq!(bed.generation(), 1, "the start's document");
    assert_eq!(looks.load(Ordering::SeqCst), 0, "no look");
    file.woken(POLL, &counted);
    assert_eq!(looks.load(Ordering::SeqCst), 1, "a whole poll looks");
    assert_eq!(bed.generation(), 1);
}

/// A look whose replica was deposed and took again under its read
/// applies nothing: its bytes were read under the term before. The
/// state stays unread, and the next looks read the file as it stands.
#[test]
fn a_look_whose_lead_changed_under_its_read_applies_nothing_and_reads_again() {
    let mut elector = election();
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    let mut file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(bed.generation(), 1, "the start's document, at the take");
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    // The second look of a poll stands behind its read: there the
    // replica is deposed and takes again.
    let mut looks = 0;
    let retaken = file.poll_looking(|path| {
        looks += 1;
        if looks == 2 {
            assert!(elector.release().expect("a release"), "it held the lease");
            elector.tick(Instant::now()).expect("a second take");
        }
        look(path)
    });
    assert_eq!(retaken, Look::Waits(1));
    assert_eq!(bed.generation(), 1);
    assert!(bed.generations.source_writes(), "it leads again");
    // Read again under the term that shows now, the file is applied.
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.answered().2, bound_to(1).as_bytes());
    assert_eq!(bed.rejected(), 0);
}

/// A replica under a shared lease that follows, over a file that holds
/// `document`: its bed, built under the fence of its election, its
/// watch and that election, the take's step wired.
fn standby(document: &[u8]) -> (Bed, SpecFile, Elector) {
    standby_validating(document, AppliedSpec::validate_digested)
}

/// [`standby`] with the validator of its generations.
fn standby_validating(document: &[u8], validate: Validate) -> (Bed, SpecFile, Elector) {
    let mut elector = election();
    let bed = bed_of(document, validate, Some(elector.fence()));
    let file =
        SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    (bed, file, elector)
}

/// The file moves to `document`, and the watch of a replica that
/// follows shows it to the door: two looks, nothing applied and nothing
/// said.
fn shown(bed: &Bed, file: &mut SpecFile, document: &[u8]) {
    let before = (bed.generation(), bed.rejected());
    replace(&bed.path, document);
    assert_eq!(file.poll(), Look::Settling);
    assert_eq!(file.poll(), Look::Shown);
    assert_eq!(file.poll(), Look::Same, "shown once");
    assert_eq!((bed.generation(), bed.rejected()), before);
}

/// A file that states its own generation stood only under that number:
/// a digest reads no line. The standby's file states 10, and its
/// leader runs the same document as generation 1 and then replaces it.
/// The take reads the file, and its higher claim runs.
#[test]
fn a_numbered_file_above_the_generations_it_followed_runs_at_the_take() {
    let numbered = with_line(LEAST_LOAD, 10);
    let (bed, mut file, mut elector) = standby(&numbered);
    follow(&bed, &[LEAST_LOAD]);
    follow(&bed, &[LEAST_LOAD, RANDOM]);
    assert_eq!(bed.generation(), 2);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(10)));
    assert_eq!(bed.answered().2, numbered);
    assert_eq!(bed.rejected(), 0);
}

/// A file the validator refuses never stood, whatever its digest. The
/// standby's file gains a `kind` line, which no digest reads, under the
/// document its replica runs. The take reads the file and tells the
/// refusal once, and the running generation stays.
#[test]
fn a_file_the_validator_refuses_is_told_at_the_take_though_its_digest_stands() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD]);
    shown(
        &bed,
        &mut file,
        format!("kind: PolicySpec\n{LEAST_LOAD}").as_bytes(),
    );
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(
        settle(&mut file),
        Err(SpecRejected::Invalid(SpecError::Kind(
            "PolicySpec".to_owned()
        )))
    );
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));
    assert_eq!(file.poll(), Look::Same, "told once");
}

/// A generation that enters while a standby passes its file leaves the
/// file's document one that stood: the pass found it standing. The
/// file is a mount one step behind, and the take leaves it.
#[test]
fn a_generation_that_enters_under_a_standbys_pass_leaves_the_file_stood() {
    thread_local! {
        static DOOR: RefCell<Option<Arc<Generations>>> = const { RefCell::new(None) };
    }
    // The pass of the file's document brings the leader's next
    // generation first, once: it enters under the look's pass.
    fn entered_under(
        document: &PolicySpec,
        digest: Digest,
        catalog: &StageRegistry,
        denied_headers: &[&str],
    ) -> Result<AppliedSpec, SpecError> {
        if let Some(door) = DOOR.with(|door| door.borrow_mut().take()) {
            follow_at(&door, &[LEAST_LOAD, RANDOM]);
        }
        AppliedSpec::validate_digested(document, digest, catalog, denied_headers)
    }
    let (bed, mut file, mut elector) = standby_validating(LEAST_LOAD.as_bytes(), entered_under);
    follow(&bed, &[LEAST_LOAD]);
    DOOR.with(|door| *door.borrow_mut() = Some(Arc::clone(&bed.generations)));
    // The mount turns to the same document, and the look reads it.
    replace(&bed.path, LEAST_LOAD.as_bytes());
    assert_eq!(file.poll(), Look::Settling);
    assert_eq!(file.poll(), Look::Shown);
    assert_eq!(bed.generation(), 2, "it entered under the pass");
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(file.poll(), Look::Same, "a mount one step behind");
    assert_eq!(bed.generation(), 2);
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
}

/// A file that gains a number while its replica follows has not stood
/// under it. The standby's file takes the running document under the
/// line 10: the take reads it and runs it under that number.
#[test]
fn a_file_that_gains_a_number_while_its_replica_follows_runs_at_the_take() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD]);
    let numbered = with_line(LEAST_LOAD, 10);
    shown(&bed, &mut file, &numbered);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(10)));
    assert_eq!(bed.answered().2, numbered);
}

/// A mount one step behind. The standby's file takes the first of two
/// changes, and its leader applies both. That document stood on the
/// replica since the file took it, so the take leaves the file: nothing
/// older comes back. The file's next move is read as a leader's is.
#[test]
fn a_mount_one_step_behind_brings_nothing_older_back_at_a_take() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD]);
    // The first change reaches this mount ahead of its leader's word of
    // it. The second reaches the leader's mount alone.
    shown(&bed, &mut file, RANDOM.as_bytes());
    follow(&bed, &[LEAST_LOAD, RANDOM]);
    follow(&bed, &[LEAST_LOAD, RANDOM, &bound_to(1)]);
    assert_eq!(bed.generation(), 3);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(file.poll(), Look::Same, "its document stood here since");
    assert_eq!(file.poll(), Look::Same);
    assert_eq!(
        (bed.generation(), bed.answered().2),
        (3, bound_to(1).into())
    );
    assert_eq!(bed.rejected(), 0);
    // The mount's sync is a move: it is read, the standing document
    // again.
    replace(&bed.path, bound_to(1).as_bytes());
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Unchanged(3)));

    // The same where the leader's word came first: the file takes the
    // document that stands, and its leader replaces it.
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD, RANDOM]);
    shown(&bed, &mut file, RANDOM.as_bytes());
    follow(&bed, &[LEAST_LOAD, RANDOM, &bound_to(1)]);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(file.poll(), Look::Same);
    assert_eq!(
        (bed.generation(), bed.answered().2),
        (3, bound_to(1).into())
    );
}

/// A rollback that reaches the standby's mount first. Its leader ran a
/// document and replaced it, and then the standby's file takes the
/// first one again. That document has not stood on the replica since
/// the file took it: it is the newer word, and the take runs it as the
/// next generation. The bytes are those of a mount that lags: the
/// order of the hand and the generations tells the two apart.
#[test]
fn a_rollback_that_reaches_the_standby_first_runs_at_the_take() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[LEAST_LOAD]);
    shown(&bed, &mut file, RANDOM.as_bytes());
    follow(&bed, &[LEAST_LOAD, RANDOM]);
    // The rollback: the file holds the first document again, behind the
    // generation that replaced it.
    shown(&bed, &mut file, LEAST_LOAD.as_bytes());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());
    assert_eq!(file.poll(), Look::Same);
}

/// A replica that follows says nothing of a file it cannot read or
/// parse: its look shows the door a file with no document, and no
/// rejected state is told. The take reads the file again and tells it
/// once, as a leader does, and the replica keeps what it followed.
#[test]
fn a_file_a_standby_cannot_read_or_parse_is_told_once_at_the_take() {
    for gone in [false, true] {
        let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
        follow(&bed, &[RANDOM]);
        if gone {
            fs::remove_file(&bed.path).expect("the file goes");
            assert_eq!(file.poll(), Look::Settling);
            assert_eq!(file.poll(), Look::Shown);
        } else {
            shown(&bed, &mut file, b"{ no policy");
        }
        assert_eq!(file.poll(), Look::Same);
        assert_eq!(bed.rejected(), 0, "a standby's silence is no verdict");
        elector.tick(Instant::now()).expect("a take");
        let told = settle(&mut file);
        if gone {
            assert!(matches!(told, Err(SpecRejected::Unreadable(_))), "{told:?}");
        } else {
            assert!(
                matches!(told, Err(SpecRejected::Unparsed { .. })),
                "{told:?}"
            );
        }
        assert_eq!(bed.rejected(), 1, "told once, at the take");
        assert_eq!(file.poll(), Look::Same, "one event under one lead");
        assert_eq!((bed.generation(), bed.answered().2), (1, RANDOM.into()));
    }
}

/// The look of a replica that follows publishes nothing, and none of
/// the bytes it read are published later: the door takes their digest
/// and keeps no byte. The take reads the file as it stands then, under
/// its own term.
#[test]
fn a_standbys_look_publishes_nothing_and_its_bytes_are_never_published() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[RANDOM]);
    shown(&bed, &mut file, bound_to(1).as_bytes());
    assert_eq!((bed.generation(), bed.answered().2), (1, RANDOM.into()));
    // The file moves again, and no look sees it ahead of the take.
    replace(&bed.path, bound_to(2).as_bytes());
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(2)));
    assert_eq!(bed.answered().2, bound_to(2).as_bytes());
}

/// What a document's own number does to a mount that lags. A file whose
/// document never stood on the replica counts as the newer word: a
/// replica that started over an older file runs it at its take. Where
/// that document states its generation, a number at or under the
/// standing one is refused by name and the standing generation stays:
/// an author who numbers the documents gets no older one back.
#[test]
fn a_numbered_file_under_the_standing_generation_is_refused_at_the_take() {
    // No number: the older file comes back, as the next generation.
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[RANDOM, &bound_to(1)]);
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(settle(&mut file), Ok(ApplyOutcome::Generation(3)));
    assert_eq!(bed.answered().2, LEAST_LOAD.as_bytes());

    // Numbered 1 under its leader's generation 2: refused, told once.
    let older = with_line(LEAST_LOAD, 1);
    let (bed, mut file, mut elector) = standby(&older);
    follow(&bed, &[RANDOM, &bound_to(1)]);
    elector.tick(Instant::now()).expect("a take");
    let refused = settle(&mut file);
    assert!(
        matches!(refused, Err(SpecRejected::Stale(_))),
        "{refused:?}"
    );
    assert_eq!((bed.generation(), bed.rejected()), (2, 1));
    assert_eq!(bed.answered().2, bound_to(1).as_bytes());
    assert_eq!(file.poll(), Look::Same);
}

/// A file that never rests under a replica that follows is told by no
/// look of it: the count waits one short of the event. The first look
/// of the replica's lead that finds the file still moving tells it,
/// once.
#[test]
fn a_restless_file_is_told_by_no_look_of_a_replica_that_follows() {
    let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
    follow(&bed, &[RANDOM]);
    // A writer that grows the file before every look.
    let mut writes = 0;
    let mut restless = |file: &mut SpecFile| {
        file.poll_looking(|path| {
            writes += 1;
            let grown = format!("{RANDOM}#{}\n", "x".repeat(writes));
            fs::write(path, grown).expect("the file");
            look(path)
        })
    };
    for _ in 0..3 * RESTLESS_LOOKS {
        assert_eq!(restless(&mut file), Look::Settling);
    }
    assert_eq!(bed.rejected(), 0, "a standby says nothing of its file");
    elector.tick(Instant::now()).expect("a take");
    assert_eq!(restless(&mut file), Look::Restless, "its lead's first look");
    assert_eq!(restless(&mut file), Look::Settling);
    assert_eq!((bed.rejected(), bed.generation()), (1, 1));
}

/// A file a leader's look cannot run is told once under that lead. The
/// door knows the file holds no document that stood, so behind the
/// replica's next take the file is read and told again, though it never
/// moved.
#[test]
fn a_file_a_leader_cannot_run_is_told_once_under_each_lead() {
    for gone in [false, true] {
        let (bed, mut file, mut elector) = standby(LEAST_LOAD.as_bytes());
        elector.tick(Instant::now()).expect("a take");
        assert_eq!(bed.generation(), 1, "the start's document, at the take");
        assert_eq!(file.poll(), Look::Same);
        if gone {
            fs::remove_file(&bed.path).expect("the file goes");
        } else {
            replace(&bed.path, b"{ no policy");
        }
        for told in 1..=2 {
            let refused = settle(&mut file);
            assert!(refused.is_err(), "{refused:?}");
            assert_eq!(bed.rejected(), told);
            assert_eq!(file.poll(), Look::Same, "one event under one lead");
            // Deposed, and back by a take: the file never moved.
            assert!(elector.release().expect("a release"), "it held the lease");
            assert_eq!(file.poll(), Look::Same, "nothing new to show");
            elector.tick(Instant::now()).expect("a take again");
        }
        assert_eq!(bed.generation(), 1);
    }
}

/// A take that never led publishes nothing, so its replica follows the
/// next leader under any number, the first included: no number of its
/// own stands in the way.
#[test]
fn a_take_past_its_window_publishes_nothing_and_follows_any_number() {
    let deadline = Duration::from_secs(1);
    let mut elector = election_under(deadline);
    let bed = bed_under(LEAST_LOAD.as_bytes(), elector.fence());
    SpecFile::start(bed.path.clone(), Arc::clone(&bed.generations)).expect("a file that runs");
    let door = Arc::clone(&bed.generations);
    elector.at_take(move || door.at_take());
    let late = Instant::now()
        .checked_sub(2 * deadline)
        .expect("a host that is up for two seconds");
    elector.tick(late).expect("a take");
    assert_eq!(bed.generation(), 0, "no term showed at the take's step");
    // Another replica leads meanwhile, under generation 1 of another
    // document.
    follow(&bed, &[RANDOM]);
    assert_eq!(bed.generation(), 1);
    assert_eq!(bed.answered().2, RANDOM.as_bytes());
    assert_eq!(bed.rejected(), 0);
}

//! The standby's loop: the lag it writes, its waits between two streams,
//! what it says of a streak of streams that applied nothing, when the next
//! snapshot may be asked, and its streams with a scripted leader.

use super::*;

#[test]
fn the_lag_is_the_age_of_the_last_applied_frame() {
    let lag = Lag::new(Duration::from_secs(1));
    let at = Instant::now();
    let millis = |now| lag.age(now).map(|age| age.as_millis());
    assert_eq!(millis(at), None);
    lag.applied(at);
    assert_eq!(millis(at), Some(0));
    assert_eq!(millis(at + Duration::from_millis(250)), Some(250));
    lag.applied(at + Duration::from_millis(200));
    assert_eq!(millis(at + Duration::from_millis(250)), Some(50));
}

#[test]
fn the_sync_holds_under_the_bound_and_turns_by_the_clock_of_the_write() {
    let bound = Duration::from_secs(1);
    let lag = Lag::new(bound);
    assert_eq!(lag.within(), bound);
    let turns = lag.turns();
    let turned = || *turns.borrow();
    // A frame dated `at`, written when the clock reads `wrote`.
    let write = |at, wrote| lag.turn(lag.store(at), wrote);
    let start = Instant::now();
    // Before any frame: not synced, nothing to lapse, no turn told.
    assert!(!lag.synced(start));
    assert_eq!((lag.read(start), turned()), (None, 0));

    // The first frame turns the sync.
    write(start, start);
    // The lag holds the frame's instant at a whole microsecond.
    let frame = start - lag.age(start).expect("a frame applied");
    assert_eq!(turned(), 1);
    let under = bound - Duration::from_micros(1);
    assert!(lag.synced(frame + under));
    assert_eq!(lag.read(frame + under), Some((under, Some(frame + bound))));
    // At the bound it is gone, and there is nothing left to lapse.
    assert!(!lag.synced(frame + bound));
    assert_eq!(lag.read(frame + bound), Some((bound, None)));

    // A frame written inside the bound keeps the sync and tells no
    // turn.
    let last = frame + under;
    write(last, last);
    assert_eq!(turned(), 1);
    // An apply that straddles the lapse: its frame is dated inside the
    // bound and written at it, after a reader read the sync gone.
    write(last + under, last + bound);
    assert_eq!(turned(), 2);
    assert!(lag.synced(last + bound));

    // A replica follows from the serve's word alone.
    assert!(!lag.follows());
    lag.follow();
    assert!(lag.follows());
}

#[test]
fn an_applied_frame_dated_before_the_lapse_and_written_after_it_tells_a_turn() {
    // The loop reads its instant, applies the batch, then writes the
    // lag: the clock of that write says whether a reader saw the sync
    // gone, not the instant the frame carries.
    let bound = Duration::from_millis(50);
    let lag = Lag::new(bound);
    let turns = lag.turns();
    let first = Instant::now();
    lag.applied(first);
    assert_eq!(*turns.borrow(), 1);
    // The bound passes with no frame.
    std::thread::sleep(bound + Duration::from_millis(10));
    // A frame dated inside the bound, written now.
    lag.applied(first + bound - Duration::from_millis(1));
    assert_eq!(*turns.borrow(), 2);
}

#[test]
fn a_forgotten_frame_reads_as_none_and_tells_a_turn() {
    let lag = Lag::new(Duration::from_secs(1));
    let turns = lag.turns();
    let now = Instant::now();
    // Nothing to forget: no turn told.
    lag.forget();
    assert_eq!(*turns.borrow(), 0);
    lag.applied(now);
    assert!(lag.synced(now));
    // The frame is gone, and a reader that read it looks again.
    lag.forget();
    assert_eq!((lag.read(now), *turns.borrow()), (None, 2));
    lag.forget();
    assert_eq!(*turns.borrow(), 2);
    // The next frame is a first frame.
    lag.applied(now);
    assert_eq!(*turns.borrow(), 3);
}

#[test]
fn a_streak_says_a_reason_once_and_starts_again_after_a_stream_that_applied() {
    let mut streak = Streak::default();
    let waits = |streak: &mut Streak| [streak.wait("a"), streak.wait("a"), streak.wait("a")];
    let first = waits(&mut streak);
    assert_eq!(first, [backoff(0, "a"), backoff(1, "a"), backoff(2, "a")]);
    // A reason is said once while it stays, and another one is said.
    let refused = || Some("refused".to_owned());
    assert_eq!(streak.ended(false, refused()), Some("refused"));
    assert_eq!(streak.ended(false, refused()), None);
    assert_eq!(streak.ended(false, None), None);
    assert_eq!(
        streak.ended(false, Some("silent".to_owned())),
        Some("silent")
    );
    assert_eq!(streak.wait("a"), backoff(3, "a"));
    // A stream that applied ends the streak: its reason is said anew
    // and the waits start again.
    assert_eq!(
        streak.ended(true, Some("silent".to_owned())),
        Some("silent")
    );
    assert_eq!(waits(&mut streak), first);
}

#[tokio::test(start_paused = true)]
async fn a_streaks_wait_ends_at_the_step_that_names_another_holder() {
    let stands = |holder: &str, term| ElectionObservation {
        holder: holder.to_owned(),
        term,
        looked: true,
    };
    let (standing, mut steps) = watch::channel(stands("a@h:1", 1));
    // A step that names the same holder: the wait runs out.
    let began = tokio::time::Instant::now();
    let same = async { standing.send_replace(stands("a@h:1", 1)) };
    tokio::join!(pause(&mut steps, false, MOST), same);
    assert_eq!(began.elapsed(), MOST);
    // A step that names another holder ends it, and any step ends a
    // wait for the lease.
    let began = tokio::time::Instant::now();
    let other = async { standing.send_replace(stands("b@h:1", 2)) };
    tokio::join!(pause(&mut steps, false, MOST), other);
    let again = async { standing.send_replace(stands("b@h:1", 2)) };
    tokio::join!(pause(&mut steps, true, MOST), again);
    assert_eq!(began.elapsed(), Duration::ZERO);
}

#[test]
fn a_streaks_wait_doubles_to_its_bound_and_half_of_it_is_the_replicas_own() {
    let replicas = ["controller-0@10.0.0.7:30100", "controller-1@10.0.0.8:30100"];
    for tries in 0..40 {
        let whole = FIRST.saturating_mul(1 << tries.min(8)).min(MOST);
        for identity in replicas {
            let wait = backoff(tries, identity);
            assert!(whole / 2 <= wait && wait <= whole, "{tries}: {wait:?}");
            assert_eq!(wait, backoff(tries, identity));
        }
    }
    assert_eq!(
        backoff(0, replicas[0]).as_millis() / 50,
        1,
        "the first wait"
    );
    assert!(backoff(39, replicas[0]) >= MOST / 2);
    // Two standbys of one leader do not wait alike along a streak.
    assert!((0..8).any(|tries| backoff(tries, replicas[0]) != backoff(tries, replicas[1])));
}

#[test]
fn a_refusals_code_says_which_wait_follows() {
    for (status, next) in [
        (
            Status::unavailable("not yet written under term 4"),
            Next::Wait,
        ),
        (Status::resource_exhausted("over its budget"), Next::Wait),
        (Status::unauthenticated("no principal"), Next::Wait),
        (Status::failed_precondition("not the leader"), Next::Lease),
        (Status::aborted("the fence closed"), Next::Lease),
    ] {
        let ended = Ended::refused(&status, true);
        assert_eq!((ended.next, ended.applied), (next, true), "{status}");
        let why = ended.why.expect("a refusal is said");
        assert!(why.contains(status.message()), "{why}");
    }
}

#[test]
fn a_standby_is_synced_under_one_load_poll_interval_and_never_under_two_beats() {
    assert_eq!(
        synced_within(Duration::from_secs(10)),
        Duration::from_secs(10)
    );
    assert_eq!(synced_within(Duration::from_secs(1)), 2 * BEAT);
    assert_eq!(synced_within(Duration::ZERO), 2 * BEAT);
}

#[tokio::test(start_paused = true)]
async fn a_snapshot_is_asked_a_beat_after_the_last_at_the_soonest() {
    let pace = Pace::new(RESYNC);
    let began = tokio::time::Instant::now();
    pace.wait().await;
    assert_eq!(began.elapsed(), Duration::ZERO, "the first at once");
    // A second and a third in a row: a beat each.
    pace.wait().await;
    assert_eq!(began.elapsed(), BEAT);
    pace.wait().await;
    assert_eq!(began.elapsed(), 2 * RESYNC);
    // One asked later than a beat after the last waits for nothing.
    tokio::time::sleep(3 * RESYNC).await;
    let late = tokio::time::Instant::now();
    pace.wait().await;
    assert_eq!(late.elapsed(), Duration::ZERO);
    // The one right after it waits a beat from then: the late ask is
    // the last asked, not the instant it was due.
    pace.wait().await;
    assert_eq!(late.elapsed(), RESYNC);
}

const FIVE: Duration = Duration::from_secs(5);

/// One `Replicate` stream as the scripted leader holds it: the
/// standby's frames, and where the case puts the batches it answers.
struct Opened {
    frames: Streaming<proto::ReplicaFrame>,
    batches: mpsc::Sender<Result<proto::ReplicationBatch, Status>>,
}

impl Opened {
    /// The standby's next frame; `None` once it left the stream.
    async fn frame(&mut self) -> Option<Frame> {
        let next = timeout(FIVE, self.frames.message()).await;
        let next = next.expect("a frame, or the stream's end, in five seconds");
        next.ok().flatten().and_then(|frame| frame.frame)
    }

    async fn says(&self, batch: proto::ReplicationBatch) {
        let sent = self.batches.send(Ok(batch)).await;
        assert!(sent.is_ok(), "the stream stands");
    }

    async fn refuses(&self, status: Status) {
        let sent = self.batches.send(Err(status)).await;
        assert!(sent.is_ok(), "the stream stands");
    }
}

/// The stream of a call the scripted leader does not serve.
type Unscripted<T> = ReceiverStream<Result<T, Status>>;

/// A leader that says what the case tells it: each stream a standby
/// opens is handed to the case, with the name the call stated. While
/// `mute`, it opens no stream.
#[derive(Clone)]
struct Scripted {
    streams: mpsc::UnboundedSender<Opened>,
    names: Arc<Mutex<Vec<String>>>,
    mute: Arc<AtomicBool>,
}

impl Scripted {
    fn named<T>(&self, request: &Request<T>) {
        let name = request.metadata().get(ENTRY);
        let name = name.and_then(|name| name.to_str().ok()).unwrap_or_default();
        self.names.lock().push(name.to_owned());
    }
}

macro_rules! scripted {
        ($($call:ident($asked:ty) -> $answer:ty;)*) => {
            #[tonic::async_trait]
            impl GrandCentral for Scripted {
                type SessionStream = Unscripted<proto::ControllerFrame>;
                type DecideStream = Unscripted<proto::DecideAnswer>;
                type WatchWorkersStream = Unscripted<proto::WorkerEvent>;
                type WatchRegimeStream = Unscripted<proto::Regime>;
                type ReplicateStream = Unscripted<proto::ReplicationBatch>;

                $(async fn $call(
                    &self,
                    _: Request<$asked>,
                ) -> Result<Response<$answer>, Status> {
                    Err(Status::unimplemented("not scripted"))
                })*

                async fn replicate(
                    &self,
                    request: Request<Streaming<proto::ReplicaFrame>>,
                ) -> Result<Response<Self::ReplicateStream>, Status> {
                    self.named(&request);
                    if self.mute.load(Ordering::Relaxed) {
                        pending::<()>().await;
                    }
                    let (batches, sent) = mpsc::channel(16);
                    let frames = request.into_inner();
                    let _ = self.streams.send(Opened { frames, batches });
                    Ok(Response::new(ReceiverStream::new(sent)))
                }
            }
        };
    }

scripted! {
    session(Streaming<proto::GatewayFrame>) -> Unscripted<proto::ControllerFrame>;
    decide(Streaming<proto::DecideRequest>) -> Unscripted<proto::DecideAnswer>;
    watch_workers(proto::WatchWorkersRequest) -> Unscripted<proto::WorkerEvent>;
    register_worker(proto::RegisterWorkerRequest) -> proto::WorkerEvent;
    patch_worker(proto::PatchWorkerRequest) -> proto::WorkerEvent;
    deregister_worker(proto::DeregisterWorkerRequest) -> proto::WorkerEvent;
    get_snapshot(proto::SnapshotRequest) -> proto::Snapshot;
    explain(proto::ExplainRequest) -> proto::DecisionRecord;
    apply_policy_spec(proto::ApplyRequest) -> proto::ApplyResult;
    get_policy_spec(proto::GetPolicySpecRequest) -> proto::PolicySpecDocument;
    rollback_policy_spec(proto::RollbackRequest) -> proto::ApplyResult;
    leader(proto::LeaderRequest) -> proto::LeaderInfo;
    step_down(proto::StepDownRequest) -> proto::Ack;
    watch_regime(proto::RegimeRequest) -> Unscripted<proto::Regime>;
}

/// The scripted leader on a loopback port, and what it hands the case.
struct Bed {
    streams: mpsc::UnboundedReceiver<Opened>,
    names: Arc<Mutex<Vec<String>>>,
    mute: Arc<AtomicBool>,
    addr: SocketAddr,
    _serving: JoinSet<Result<(), Transport>>,
}

impl Bed {
    async fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await;
        let listener = listener.expect("a loopback port");
        let addr = listener.local_addr().expect("the bound address");
        let (streams, opened) = mpsc::unbounded_channel();
        let names = Arc::new(Mutex::new(Vec::new()));
        let mute = Arc::new(AtomicBool::new(false));
        let leader = Scripted {
            streams,
            names: Arc::clone(&names),
            mute: Arc::clone(&mute),
        };
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .add_service(GrandCentralServer::new(leader))
                .serve_with_incoming(TcpIncoming::from(listener)),
        );
        Self {
            streams: opened,
            names,
            mute,
            addr,
            _serving: serving,
        }
    }

    /// The next stream the standby opens.
    async fn opened(&mut self) -> Opened {
        let opened = timeout(FIVE, self.streams.recv()).await;
        opened
            .expect("a stream in five seconds")
            .expect("the leader stands")
    }

    /// What the election saw: this leader holds the lease as `id`
    /// under `term`.
    fn holds(&self, id: &str, term: u64) -> ElectionObservation {
        ElectionObservation {
            holder: format!("{id}@{}", self.addr),
            term,
            looked: true,
        }
    }
}

const OWN: &str = "standby-0@127.0.0.1:1";

/// A replica that follows: its candidate, its election's writer and
/// its loop, which dials the port of `bed` and holds a stream quiet
/// for `quiet` at most.
fn standing_by(
    bed: &Bed,
    quiet: Duration,
) -> (Elector, watch::Sender<ElectionObservation>, Standby) {
    let elector = Elector::alone(OWN, ElectionSettings::default())
        .expect("the default settings are in order");
    let (standing, stood) = watch::channel(ElectionObservation::default());
    // One place for the log and for the document door: a standby's
    // lease is one another replica can take.
    let place = LeasePlace::File("shared".into());
    let journal = journaled(&place, 1);
    let (id, address) = ("standby-0".to_owned(), "127.0.0.1:1".to_owned());
    let catalog = Arc::new(crate::stages::StageRegistry::builtin());
    let journal = followed(journal);
    let source = (catalog, &place);
    let shared = Shared::new(id, address, &elector, unserved(stood), journal, source, 1)
        .expect("the deciders start");
    let mut standby = Standby::new(Arc::new(shared), bed.addr.port());
    // The bound a replica runs with: three of its leader's heartbeats,
    // so one late beat is not a silent leader.
    assert_eq!(standby.quiet, 3 * BEAT);
    standby.quiet = quiet;
    (elector, standing, standby)
}

fn resync() -> proto::ReplicationBatch {
    proto::ReplicationBatch {
        resync_required: true,
        ..proto::ReplicationBatch::default()
    }
}

#[tokio::test]
async fn the_hello_names_the_standbys_place_and_each_batch_of_events_is_acknowledged_there() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    // The standby stands past a leader's first writes, under term 1,
    // and the lease names the leader under term 7.
    let leader = led();
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    assert!(place.seq > 0 && place.term == 1, "{place:?}");
    standing.send_replace(bed.holds("leader-1", 7));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        let hello = proto::ReplicaHello {
            replica_id: "standby-0".to_owned(),
            last_applied_seq: place.seq,
            term_seen: 7,
            max_backlog: 0,
            last_applied_term: 1,
        };
        assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
        // The leader's beat draws no acknowledgement; a batch of
        // events draws one, at the place the standby reached.
        opened.says(proto::ReplicationBatch::default()).await;
        leader.0.upsert(facts("http://b:8000", 1));
        leader.0.upsert(facts("http://c:8000", 1));
        for entry in entries(&leader.1, place) {
            opened.says(batch(&entry)).await;
            let seq = entry.at.seq;
            let ack = Frame::Ack(proto::ReplicationAck { seq });
            assert_eq!(opened.frame().await, Some(ack));
        }
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    assert_eq!(ended.why.as_deref(), Some("Aborted: deposed"));
    assert_eq!(standby.follower.place(), leader.1.head());
    assert_eq!(*bed.names.lock(), ["standby-0"]);
}

#[tokio::test]
async fn a_standby_reads_synced_at_the_frame_that_leaves_it_at_its_leaders_head() {
    // A leader that states its head, and one of a build before the
    // field, which states none.
    for stated in [true, false] {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        // The standby stands past a leader's first writes, and the
        // leader wrote three rows since: a replay of two frames.
        let leader = led();
        let now = Instant::now();
        follow(&standby.follower, &leader.1, now);
        let place = standby.follower.place();
        for key in ["http://b:8000", "http://c:8000", "http://d:8000"] {
            leader.0.upsert(facts(key, 1));
        }
        let tail = entries(&leader.1, place);
        let head = if stated { leader.1.head().seq } else { 0 };
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let synced = || standby.shared.lag.age(Instant::now()).is_some();
        let script = async {
            let mut opened = bed.opened().await;
            assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
            assert!(!synced(), "before any frame");
            let ack = |entry: &Entry| {
                let seq = entry.at.seq;
                Some(Frame::Ack(proto::ReplicationAck { seq }))
            };
            // The replay's first frame ends short of the head it
            // states: the standby applied it and stands behind.
            opened.says(batch_of(&tail[..1], head)).await;
            assert_eq!(opened.frame().await, ack(&tail[0]));
            assert_eq!(synced(), !stated, "a frame short of the head");
            // The frame that ends there leaves it at the head.
            opened.says(batch_of(&tail[1..], head)).await;
            assert_eq!(opened.frame().await, ack(&tail[2]));
            assert!(synced(), "the frame that reached the head");
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    }
}

#[tokio::test]
async fn a_head_stated_at_the_number_of_the_standbys_own_row_moves_no_sync() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    // A log's places start where its run drew, and a row of the
    // store's own stands past every place its log drew: it takes the
    // next number where the leader's places lie above the standby's
    // own. This leader's start there.
    let drawn = standby.follower.log.head();
    let above = journaled_from(&LeasePlace::File("shared".into()), drawn.seq + 1);
    let leader: Replica = (above.0, above.1.log);
    leader.0.set_term(1);
    leader.0.upsert(facts("http://a:8000", 1));
    let now = Instant::now();
    leader.0.upsert(facts("http://b:8000", 1));
    follow(&standby.follower, &leader.1, now);
    let followed = standby.follower.place();
    // A write of the standby's own lands at the next number, under no
    // term.
    standby.follower.store.upsert(facts("http://own:8000", 1));
    let own = standby.follower.place();
    assert_eq!((own.seq, own.term), (followed.seq + 1, 0));
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        // The leader wrote one event since, and its heartbeat states
        // that head: the number of the standby's own row.
        let beat = proto::ReplicationBatch {
            first_seq: followed.seq + 1,
            term: followed.term,
            head_seq: own.seq,
            ..proto::ReplicationBatch::default()
        };
        opened.says(beat).await;
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    let age = standby.shared.lag.age(Instant::now());
    assert!(age.is_none(), "synced by a head at an own row's number");
}

#[tokio::test]
async fn a_restored_standby_reads_synced_once_it_applied_what_was_written_during_the_send() {
    for tail in [false, true] {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        let parts = snapshot(&leader, now);
        let place = leader.1.head();
        // A row written while the snapshot was sent: the last part
        // states a head one past the snapshot's place.
        leader.0.upsert(facts("http://b:8000", 1));
        let written = entries(&leader.1, place);
        let head = leader.1.head().seq;
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
            opened.says(resync()).await;
            for part in &parts {
                let mut part = part.clone();
                part.head_seq = head;
                opened.says(part).await;
            }
            if tail {
                opened.says(batch_of(&written, head)).await;
                let ack = Frame::Ack(proto::ReplicationAck { seq: head });
                assert_eq!(opened.frame().await, Some(ack));
            }
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        // Restored and behind what was written since, it is not synced:
        // the frame that brings that tail leaves it at the head.
        let synced = standby.shared.lag.age(Instant::now()).is_some();
        assert_eq!(synced, tail, "the tail applied: {tail}");
    }
}

#[tokio::test]
async fn a_standby_that_applied_nothing_says_so_and_joins_at_its_leaders_first_write() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        // No place of any run: zero for both, wherever its log stands.
        let hello = proto::ReplicaHello {
            replica_id: "standby-0".to_owned(),
            last_applied_seq: 0,
            term_seen: 1,
            max_backlog: 0,
            last_applied_term: 0,
        };
        assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
        // The leader's log from its first write, a term's start at the
        // position after the place the leader's run drew.
        for entry in entries(&leader.1, start(&leader.1)) {
            opened.says(batch(&entry)).await;
            let seq = entry.at.seq;
            let ack = Frame::Ack(proto::ReplicationAck { seq });
            assert_eq!(opened.frame().await, Some(ack));
        }
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    assert_eq!(standby.follower.place(), leader.1.head());
    assert_eq!(start(&standby.shared.feed.log), start(&leader.1));
}

#[tokio::test]
async fn a_resync_is_the_leaders_snapshot_in_parts_on_the_stream_it_holds() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let now = Instant::now();
    a_page_and_one(&leader.0);
    let parts = snapshot(&leader, now);
    assert_eq!(parts.len(), 2, "two parts of rows");
    let place = leader.1.head();
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        // The leader's word, then its snapshot, a part a batch.
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        // The log from the snapshot's place, on the same stream: the
        // standby applies it there and acknowledges it.
        leader.0.upsert(facts("http://c:8000", 1));
        for entry in entries(&leader.1, place) {
            opened.says(batch(&entry)).await;
            let seq = entry.at.seq;
            let ack = Frame::Ack(proto::ReplicationAck { seq });
            assert_eq!(opened.frame().await, Some(ack));
        }
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    assert_eq!(standby.follower.place(), leader.1.head());
    assert_eq!(held(&standby.shared.store), held(&leader.0));
    let lag = standby.shared.lag.age(Instant::now());
    assert!(lag.is_some(), "a restore is a frame applied");
    // One stream and no other call: the snapshot rode it. The leader's
    // word is no ask of the standby's.
    assert_eq!(*bed.names.lock(), ["standby-0"]);
    assert_eq!(standby.shared.lag.asks(), 0);
}

/// The generation a snapshot's first part states rides the stream to the
/// restore: once the last part is in, the standby's door holds its
/// leader's document beside its leader's rows.
#[tokio::test]
async fn a_resync_lands_the_generation_its_snapshots_first_part_states() {
    use crate::service::controller::{
        generations::{tests::LEAST_LOAD, Format},
        wire::spec_document,
    };

    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let leading = door(&leader);
    leading
        .apply(LEAST_LOAD.into(), Format::Yaml)
        .expect("a valid document");
    a_page_and_one(&leader.0);
    let mut parts = snapshot(&leader, Instant::now());
    assert_eq!(parts.len(), 2, "two parts of rows");
    let told = leading.standing_sent().map(Box::new);
    parts[0].snapshot.as_mut().expect("a part").generation = told;
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    assert_eq!(held(&standby.shared.store), held(&leader.0));
    let following = &standby.shared.generations;
    assert_eq!(
        spec_document(following.document(0).expect("held")),
        spec_document(leading.document(0).expect("held"))
    );
}

#[tokio::test]
async fn a_stream_that_ends_inside_a_snapshot_leaves_the_store_as_it_was() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    // The standby follows a leader's first writes.
    let leader = led();
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    let before = held(&standby.shared.store);
    // The leader moves on, and its snapshot is two parts.
    a_page_and_one(&leader.0);
    let parts = snapshot(&leader, now);
    assert_eq!(parts.len(), 2);
    let elsewhere = proto::ReplicationBatch {
        first_seq: parts[1].first_seq + 1,
        ..parts[1].clone()
    };
    let event = entries(&leader.1, place).remove(0);
    // After the first part: the stream's end, the same part again, a
    // part at another place, a batch of events.
    let cut_short = [
        ("the stream ends", Err(Status::unavailable("gone"))),
        ("a part out of its order", Ok(parts[0].clone())),
        ("a part at another place", Ok(elsewhere)),
        ("a batch of events", Ok(batch(&event))),
    ];
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    for (what, next) in cut_short {
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(resync()).await;
            opened.says(parts[0].clone()).await;
            let sent = opened.batches.send(next).await;
            assert!(sent.is_ok(), "the stream stands");
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Wait, false), "{what}");
        assert_eq!(standby.follower.place(), place, "{what}");
        assert_eq!(held(&standby.shared.store), before, "{what}");
    }
    // The whole snapshot on the next stream restores it. The word said
    // again starts the staging over: the parts after it are a
    // snapshot's from its first.
    let applied = AtomicBool::new(false);
    let script = async {
        let opened = bed.opened().await;
        opened.says(resync()).await;
        opened.says(parts[0].clone()).await;
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, true));
    assert_eq!(held(&standby.shared.store), held(&leader.0));
}

#[tokio::test]
async fn a_standby_whose_event_did_not_land_reads_not_synced() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    // The leader's event after the next: past a gap, it does not land.
    leader.0.upsert(facts("http://b:8000", 1));
    leader.0.upsert(facts("http://c:8000", 1));
    let past = entries(&leader.1, place).pop().expect("the leader wrote");
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let mut turns = standby.shared.lag.turns();
    let synced = || standby.shared.lag.age(Instant::now()).is_some();
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        // A heartbeat that states no head: the standby reads synced.
        let _ = turns.borrow_and_update();
        opened.says(proto::ReplicationBatch::default()).await;
        let turned = timeout(FIVE, turns.changed()).await;
        assert!(matches!(turned, Ok(Ok(()))), "the heartbeat's turn");
        assert!(synced(), "at the heartbeat");
        opened.says(batch(&past)).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!(ended.next, Next::Join);
    assert!(!synced(), "the sync outlived an event that did not land");
}

#[tokio::test]
async fn an_event_that_does_not_land_asks_a_restore_and_two_in_a_row_wait_a_streak_apart() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    // The leader's event after the next: past a gap, it does not land.
    let gap = |leader: &Replica, after: Position| {
        leader
            .0
            .upsert(facts(&format!("http://g{}:8000", after.seq), 1));
        leader
            .0
            .upsert(facts(&format!("http://h{}:8000", after.seq), 1));
        let past = entries(&leader.1, after).pop().expect("the leader wrote");
        batch(&past)
    };
    standing.send_replace(bed.holds("leader-1", 7));
    let named = standby.shared.leader(None);
    let hello = |seq, term| {
        Some(Frame::Hello(proto::ReplicaHello {
            replica_id: "standby-0".to_owned(),
            last_applied_seq: seq,
            term_seen: 7,
            max_backlog: 0,
            last_applied_term: term,
        }))
    };

    // The first: the stream is left, the ask is counted, and the next
    // stream opens at once.
    let lost = gap(&leader, place);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert_eq!(opened.frame().await, hello(place.seq, place.term));
        opened.says(lost).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Join, false));
    assert!(ended.why.is_none());
    assert_eq!(standby.shared.lag.asks(), 1);

    // The next hello asks: its head's number under no term, the term
    // it saw as it is. The leader answers its snapshot, and the event
    // after it again does not land: a streak's wait, not a stream.
    let parts = snapshot(&leader, now);
    let restored = leader.1.head();
    let lost = gap(&leader, restored);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert_eq!(opened.frame().await, hello(place.seq, 0));
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.says(lost).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Wait, false));
    let why = "restored twice with no event landed between";
    assert_eq!(ended.why.as_deref(), Some(why));
    assert_eq!(standby.follower.place(), restored);
    assert_eq!(standby.shared.lag.asks(), 2);
    let first = standby.pace.due.lock().expect("a restore was asked");

    // It asks again, a beat after its last ask at the soonest.
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert_eq!(opened.frame().await, hello(restored.seq, 0));
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, false));
    let second = standby.pace.due.lock().expect("a restore was asked");
    assert!(second >= first + RESYNC, "{first:?} then {second:?}");
    // A restore, an event that lands, then one that does not: the pair
    // is over, and the next stream opens at once.
    let parts = snapshot(&leader, now);
    let restored = leader.1.head();
    leader.0.upsert(facts("http://landed:8000", 1));
    let lands = entries(&leader.1, restored).remove(0);
    let lost = gap(&leader, lands.at);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.says(batch(&lands)).await;
        opened.says(lost).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Join, true));
    assert_eq!(standby.follower.place(), lands.at);
    assert_eq!(standby.shared.lag.asks(), 3);
    assert_eq!(*bed.names.lock(), ["standby-0"; 4]);
    // A restore, then one batch whose first event lands and whose
    // second does not: an event landed since the restore all the same.
    let parts = snapshot(&leader, now);
    let restored = leader.1.head();
    leader.0.upsert(facts("http://halved:8000", 1));
    let lands = entries(&leader.1, restored).remove(0);
    let halved = proto::ReplicationBatch {
        events: vec![worker_wire::replication(&lands); 2],
        ..batch(&lands)
    };
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.says(halved).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Join, true));
    assert!(ended.why.is_none(), "{:?}", ended.why);
    assert_eq!(standby.follower.place(), lands.at);
    assert_eq!(standby.shared.lag.asks(), 4);
    assert_eq!(*bed.names.lock(), ["standby-0"; 5]);
}

/// A stream's first batch that lands in part: the event that landed is
/// the stream's first, and the stream ends as one that applied, so the
/// next opens at once and a streak of empty streams ends there.
#[tokio::test]
async fn a_first_batch_that_lands_in_part_ends_its_stream_as_one_that_applied() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    standing.send_replace(bed.holds("leader-1", 7));
    let named = standby.shared.leader(None);
    // One batch of the leader's next event twice: the first lands and
    // the second, at a number the log has passed, does not.
    leader.0.upsert(facts("http://halved:8000", 1));
    let lands = entries(&leader.1, place).remove(0);
    let halved = proto::ReplicationBatch {
        events: vec![worker_wire::replication(&lands); 2],
        ..batch(&lands)
    };
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        opened.says(halved).await;
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Join, true));
    assert!(ended.why.is_none(), "{:?}", ended.why);
    assert!(!applied.load(Ordering::Relaxed), "no whole frame landed");
    assert_eq!(standby.follower.place(), lands.at);
    assert_eq!(standby.shared.lag.asks(), 1);
}

#[tokio::test]
async fn an_ask_states_the_floor_its_place_no_longer_names() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    // The standby followed a leader into term 9, and the lease it reads
    // names a leader of term 7.
    let leader = led();
    leader.0.set_term(9);
    follow(&standby.follower, &leader.1, Instant::now());
    let place = standby.follower.place();
    assert_eq!(place.term, 9);
    standing.send_replace(bed.holds("leader-1", 7));
    let named = standby.shared.leader(None);
    standby.asks.store(true, Ordering::Relaxed);
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        let hello = proto::ReplicaHello {
            replica_id: "standby-0".to_owned(),
            last_applied_seq: place.seq,
            term_seen: 9,
            max_backlog: 0,
            last_applied_term: 0,
        };
        assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
        opened.refuses(Status::failed_precondition("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, false));
}

#[tokio::test]
async fn a_restore_ends_the_ask_and_the_next_hello_names_the_place_restored() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let leader = led();
    let now = Instant::now();
    let parts = snapshot(&leader, now);
    let restored = leader.1.head();
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    let hello = |place: Position| {
        Some(Frame::Hello(proto::ReplicaHello {
            replica_id: "standby-0".to_owned(),
            last_applied_seq: place.seq,
            term_seen: 1,
            max_backlog: 0,
            last_applied_term: place.term,
        }))
    };
    // An event did not land: the hello asks, and the snapshot answers.
    standby.asks.store(true, Ordering::Relaxed);
    let asking = standby.follower.asking();
    assert!(asking.seq > 0 && asking.term == 0, "{asking:?}");
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert_eq!(opened.frame().await, hello(asking));
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened.refuses(Status::unavailable("gone")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Wait, true));
    // Restored, it names its place under its term again.
    let applied = AtomicBool::new(false);
    let script = async {
        let mut opened = bed.opened().await;
        assert_eq!(opened.frame().await, hello(restored));
        opened.refuses(Status::aborted("deposed")).await;
    };
    let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, false));
}

#[tokio::test]
async fn a_snapshot_the_log_does_not_follow_is_left_and_the_lease_read_again() {
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    // The standby followed a leader into term 9.
    let leader = led();
    leader.0.set_term(9);
    let now = Instant::now();
    follow(&standby.follower, &leader.1, now);
    let place = standby.follower.place();
    let before = held(&standby.shared.store);
    // A replica that still believes it leads term 1 restores it.
    let parts = snapshot(&led(), now);
    standing.send_replace(bed.holds("leader-1", 9));
    let named = standby.shared.leader(None);
    let applied = AtomicBool::new(false);
    let script = async {
        let opened = bed.opened().await;
        opened.says(resync()).await;
        for part in &parts {
            opened.says(part.clone()).await;
        }
        opened
    };
    let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
    assert_eq!((ended.next, ended.applied), (Next::Lease, false));
    let why = ended.why.expect("a reason");
    assert!(
        why.ends_with("which this replica's log does not follow"),
        "{why}"
    );
    assert_eq!(standby.follower.place(), place);
    assert_eq!(held(&standby.shared.store), before);
}

#[tokio::test]
async fn a_replica_that_leads_by_now_asks_no_restore() {
    let mut bed = Bed::new().await;
    let (mut elector, standing, standby) = standing_by(&bed, FIVE);
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    // An event did not land, and the replica took its own lease since.
    standby.asks.store(true, Ordering::Relaxed);
    let took = elector.tick(Instant::now());
    assert!(matches!(took, Ok(TickOutcome::Acquired { .. })), "{took:?}");
    let applied = AtomicBool::new(false);
    let ended = standby.stream(&named, &applied).await;
    assert_eq!((ended.next, ended.applied), (Next::Lease, false));
    assert!(ended.why.is_none());
    assert!(bed.streams.try_recv().is_err(), "no stream was opened");
}

#[tokio::test]
async fn the_loop_leaves_a_stream_once_the_lease_names_another_leader_or_this_replica() {
    let mut bed = Bed::new().await;
    let (mut elector, standing, standby) = standing_by(&bed, FIVE);
    let (stop, stopped) = watch::channel(false);
    standing.send_replace(bed.holds("leader-1", 1));
    let mut running = JoinSet::new();
    running.spawn(standby.run(stopped));
    let seen = |frame: Option<Frame>| match frame {
        Some(Frame::Hello(hello)) => Some(hello.term_seen),
        _ => None,
    };
    let mut first = bed.opened().await;
    assert_eq!(seen(first.frame().await), Some(1));
    // The lease names another leader. The first one's stream stands
    // and says nothing: it is left, and the leader named is joined.
    standing.send_replace(bed.holds("leader-2", 2));
    let mut second = bed.opened().await;
    assert_eq!(seen(second.frame().await), Some(2));
    assert_eq!(first.frame().await, None, "the first stream is left");
    // The same leader under its next term, the lease lost and taken
    // again: the stream of its last term is left too.
    standing.send_replace(bed.holds("leader-2", 3));
    let mut third = bed.opened().await;
    assert_eq!(seen(third.frame().await), Some(3));
    assert_eq!(second.frame().await, None, "the last term's stream");
    // This replica takes the lease: it leaves the stream, and at the
    // election's next step it dials nobody.
    let took = elector.tick(Instant::now());
    assert!(matches!(took, Ok(TickOutcome::Acquired { .. })), "{took:?}");
    let own = ElectionObservation {
        holder: OWN.to_owned(),
        term: 4,
        looked: true,
    };
    standing.send_replace(own.clone());
    assert_eq!(third.frame().await, None, "a leader follows nobody");
    standing.send_replace(own);
    let dialed = timeout(Duration::from_millis(300), bed.streams.recv()).await;
    assert!(dialed.is_err(), "a leader dials nobody");
    stop.send_replace(true);
    let ended = timeout(FIVE, running.join_next()).await;
    assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
}

/// The sync is with one leader's term: once the lease names another
/// leader, or the same one's next term, the frame applied before is
/// forgotten, and the replica is synced again at that leader's first.
#[tokio::test]
async fn a_standby_forgets_its_sync_once_its_lease_names_another_leaders_term() {
    let far = Duration::from_secs(60);
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let shared = Arc::clone(&standby.shared);
    let mut turns = shared.lag.turns();
    let holds = || shared.lag.age(Instant::now()).is_some();
    let (stop, stopped) = watch::channel(false);
    let mut running = JoinSet::new();
    running.spawn(standby.run(stopped));
    for (leader, term) in [("leader-1", 1), ("leader-2", 2), ("leader-2", 3)] {
        standing.send_replace(bed.holds(leader, term));
        let mut opened = bed.opened().await;
        assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
        // The stream stands and no frame of it came: what the last
        // leader's left is forgotten before the dial.
        assert!(!holds(), "{leader} under term {term}, before its frame");
        // A heartbeat that states no head: the frame is at the head.
        let _ = turns.borrow_and_update();
        opened.says(proto::ReplicationBatch::default()).await;
        let turned = timeout(far, turns.changed()).await;
        assert!(matches!(turned, Ok(Ok(()))), "{leader} under term {term}");
        assert!(holds(), "{leader} under term {term}, at its frame");
    }
    stop.send_replace(true);
    let ended = timeout(FIVE, running.join_next()).await;
    assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
}

/// The sync is with one leader's term: a stream that ends and is dialed
/// again under that term keeps the instant its last frame left.
#[tokio::test]
async fn a_standby_keeps_its_sync_across_a_dial_under_one_term() {
    let far = Duration::from_secs(60);
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let shared = Arc::clone(&standby.shared);
    let mut turns = shared.lag.turns();
    let (stop, stopped) = watch::channel(false);
    let mut running = JoinSet::new();
    running.spawn(standby.run(stopped));
    standing.send_replace(bed.holds("leader-1", 1));
    let mut first = bed.opened().await;
    assert!(matches!(first.frame().await, Some(Frame::Hello(_))));
    // A heartbeat that states no head: the frame is at the head.
    let _ = turns.borrow_and_update();
    first.says(proto::ReplicationBatch::default()).await;
    let turned = timeout(far, turns.changed()).await;
    assert!(matches!(turned, Ok(Ok(()))), "the first stream's frame");
    let held = shared.lag.age(Instant::now());
    assert!(held.is_some(), "synced at the first stream's frame");
    // The leader ends the stream, and the lease names it still: the
    // standby dials it again under the same term.
    first.refuses(Status::unavailable("not yet")).await;
    let mut second = bed.opened().await;
    assert!(matches!(second.frame().await, Some(Frame::Hello(_))));
    // No frame of the second stream came, and the first's still stands.
    let kept = shared.lag.age(Instant::now());
    assert!(
        kept >= held,
        "the dial forgot the sync: {kept:?} after {held:?}"
    );
    stop.send_replace(true);
    let ended = timeout(FIVE, running.join_next()).await;
    assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
}

#[tokio::test]
async fn a_replica_that_led_is_not_synced_before_a_frame_of_its_present_leader() {
    let mut bed = Bed::new().await;
    let (mut elector, standing, standby) = standing_by(&bed, FIVE);
    let shared = Arc::clone(&standby.shared);
    let held = || shared.lag.age(Instant::now()).is_some();
    let hello = |frame: Option<Frame>| matches!(frame, Some(Frame::Hello(_)));
    let (stop, stopped) = watch::channel(false);
    let leader = led();
    standing.send_replace(bed.holds("leader-1", 1));
    let mut running = JoinSet::new();
    running.spawn(standby.run(stopped));
    // The leader's events, acknowledged: the lag holds a frame.
    let mut first = bed.opened().await;
    assert!(hello(first.frame().await));
    let mut place = start(&leader.1);
    for entry in entries(&leader.1, place) {
        first.says(batch(&entry)).await;
        place = entry.at;
        let ack = Frame::Ack(proto::ReplicationAck { seq: place.seq });
        assert_eq!(first.frame().await, Some(ack));
    }
    assert!(held());

    // The lease names another leader: the last one's frame says
    // nothing of it, though this replica held no lease between the
    // two. It is synced again at a frame of the leader named.
    standing.send_replace(bed.holds("leader-2", 2));
    let mut second = bed.opened().await;
    assert!(hello(second.frame().await));
    assert_eq!(first.frame().await, None, "the first stream is left");
    assert!(!held(), "no frame of its present leader yet");
    leader.0.upsert(facts("http://b:8000", 1));
    for entry in entries(&leader.1, place) {
        second.says(batch(&entry)).await;
        place = entry.at;
        let ack = Frame::Ack(proto::ReplicationAck { seq: place.seq });
        assert_eq!(second.frame().await, Some(ack));
    }
    assert!(held());

    // This replica takes the lease: the frame it applied before its
    // lead is forgotten.
    let took = elector.tick(Instant::now());
    assert!(matches!(took, Ok(TickOutcome::Acquired { .. })), "{took:?}");
    standing.send_replace(ElectionObservation {
        holder: OWN.to_owned(),
        term: 3,
        looked: true,
    });
    assert_eq!(second.frame().await, None, "a leader follows nobody");
    assert!(!held(), "a replica that led holds no frame");

    // It leaves the lease and follows the next holder: not synced
    // before a frame of that leader, synced at it.
    assert!(elector.release().expect("the release"));
    standing.send_replace(bed.holds("leader-3", 4));
    let mut third = bed.opened().await;
    assert!(hello(third.frame().await));
    assert!(!held(), "no frame of its present leader yet");
    leader.0.upsert(facts("http://c:8000", 1));
    for entry in entries(&leader.1, place) {
        third.says(batch(&entry)).await;
        let ack = Frame::Ack(proto::ReplicationAck { seq: entry.at.seq });
        assert_eq!(third.frame().await, Some(ack));
    }
    assert!(held());
    stop.send_replace(true);
    let ended = timeout(FIVE, running.join_next()).await;
    assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
}

/// One stream of a standby driven to its end, while the scripted
/// leader says `first` on every stream it sees open. A stream the host
/// held past the standby's bound reaches the leader late, or never: a
/// late one is said the same frame, which nobody reads, and none is
/// waited for. The streams are returned held, so none ends from the
/// leader's side first.
async fn leaving<T>(
    bed: &mut Bed,
    stream: impl std::future::Future<Output = T>,
    first: &proto::ReplicationBatch,
) -> (T, Vec<Opened>) {
    tokio::pin!(stream);
    let mut opened = Vec::new();
    loop {
        tokio::select! {
            ended = &mut stream => return (ended, opened),
            Some(next) = bed.streams.recv() => {
                // The standby may have left this stream already.
                let _ = next.batches.send(Ok(first.clone())).await;
                opened.push(next);
            }
        }
    }
}

/// The bound past which a step of the quiet case is not taken again:
/// far past any stall of a loaded host.
const WIDEST: Duration = Duration::from_secs(10);

#[tokio::test]
async fn a_leader_that_says_nothing_is_left_at_the_quiet_bound() {
    let mut bed = Bed::new().await;
    let (_elector, standing, mut standby) = standing_by(&bed, Duration::from_millis(100));
    standing.send_replace(bed.holds("leader-1", 1));
    let named = standby.shared.leader(None);
    // A stream that stands and brings no frame after its first. The
    // bound runs on the host's clock, from the dial to the stream's
    // opening and from there to the first frame, and a held host runs
    // either past it: the stream is then left with nothing applied.
    // The step is taken again under twice the bound until a first
    // frame was applied inside it.
    let first = proto::ReplicationBatch::default();
    loop {
        let quiet = standby.quiet;
        let applied = AtomicBool::new(false);
        let left = leaving(&mut bed, standby.stream(&named, &applied), &first);
        let left = timeout(FIVE + 3 * quiet, left).await;
        let (ended, opened) = left.expect("the stream is left");
        drop(opened);
        assert_eq!(ended.next, Next::Wait);
        if ended.applied {
            let why = format!("the leader sent no frame for {quiet:?}");
            assert_eq!(ended.why.as_deref(), Some(why.as_str()));
            break;
        }
        assert!(quiet < WIDEST, "no first frame inside {quiet:?}");
        standby.quiet = 2 * quiet;
    }
    // A snapshot whose next part does not arrive: the same bound. A
    // stream the host held unopened past the bound is taken again the
    // same way.
    let word = resync();
    loop {
        let quiet = standby.quiet;
        let applied = AtomicBool::new(false);
        let left = leaving(&mut bed, standby.stream(&named, &applied), &word);
        let left = timeout(FIVE + 3 * quiet, left).await;
        let (ended, held) = left.expect("the stream is left");
        drop(held);
        assert_eq!((ended.next, ended.applied), (Next::Wait, false));
        let unopened = format!("the leader opened no stream for {quiet:?}");
        if ended.why.as_deref() != Some(unopened.as_str()) {
            let why = format!("the leader sent no frame for {quiet:?}");
            assert_eq!(ended.why.as_deref(), Some(why.as_str()));
            break;
        }
        assert!(quiet < WIDEST, "no stream opened inside {quiet:?}");
        standby.quiet = 2 * quiet;
    }
    // A leader that opens no stream.
    bed.mute.store(true, Ordering::Relaxed);
    let quiet = standby.quiet;
    let applied = AtomicBool::new(false);
    let left = standby.stream(&named, &applied);
    let ended = timeout(FIVE + quiet, left).await;
    let ended = ended.expect("the stream is left");
    assert_eq!((ended.next, ended.applied), (Next::Wait, false));
    let why = format!("the leader opened no stream for {quiet:?}");
    assert_eq!(ended.why.as_deref(), Some(why.as_str()));
}

/// The lines a case's subscriber kept.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<u8>>>);

impl io::Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn a_streak_of_streams_that_applied_nothing_is_said_once_and_again_after_one_that_did() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // A second subscriber stays registered beside it: with one alone a
    // line is on or off by the thread that reaches it first.
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);
    let said = || {
        let kept = lines.0.lock();
        let text = String::from_utf8_lossy(&kept);
        text.matches("this standby does not follow its leader")
            .count()
    };
    let mut bed = Bed::new().await;
    let (_elector, standing, standby) = standing_by(&bed, FIVE);
    let (stop, stopped) = watch::channel(false);
    standing.send_replace(bed.holds("leader-1", 1));
    let mut running = JoinSet::new();
    running.spawn(standby.run(stopped));
    // Three streams the leader refuses: one line.
    for _ in 0..3 {
        let opened = bed.opened().await;
        opened.refuses(Status::unavailable("not yet")).await;
    }
    // A stream that applies a frame ends the streak: the reason of the
    // next one is said anew.
    let opened = bed.opened().await;
    assert_eq!(said(), 1);
    opened.says(proto::ReplicationBatch::default()).await;
    opened.refuses(Status::unavailable("not yet")).await;
    let _next = bed.opened().await;
    assert_eq!(said(), 2);
    stop.send_replace(true);
    let ended = timeout(FIVE, running.join_next()).await;
    assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
}

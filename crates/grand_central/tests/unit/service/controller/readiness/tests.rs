use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use tokio::sync::watch;

use super::{channel, Phase, Readiness, Sample, State};
use crate::{
    engine::view::WorkerId,
    service::{
        controller::{election::ElectionObservation, standby::Lag},
        elector::{ElectionSettings, Elector, Fence, MemoryLease, TickOutcome},
        replication::{Event, Log, LogPositionLost},
    },
    store::{Journal, MembershipChange},
};

/// The run loop's writer and a reader over `fence`, as [`channel`]
/// makes them, for a case that reads no log: the reader's log is its
/// own and empty.
pub(crate) fn detached(
    fence: Arc<Fence>,
    steps: watch::Receiver<ElectionObservation>,
    lag: Arc<Lag>,
) -> (watch::Sender<Phase>, Readiness) {
    channel(fence, steps, lag, Arc::new(Log::new(1)))
}

/// A candidate alone over a lease of its own and a readiness that reads
/// its fence: the run loop's writer, the election's writer, the reader.
pub(crate) fn fenced() -> (
    Elector,
    watch::Sender<Phase>,
    watch::Sender<ElectionObservation>,
    Readiness,
) {
    over(
        Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
            .expect("the default settings are in order"),
    )
}

/// The same over the candidate that is handed in.
fn over(
    elector: Elector,
) -> (
    Elector,
    watch::Sender<Phase>,
    watch::Sender<ElectionObservation>,
    Readiness,
) {
    let (steps, stepped) = watch::channel(ElectionObservation::default());
    let (phase, readiness) = detached(elector.fence(), stepped, Arc::new(Lag::new(BOUND)));
    (elector, phase, steps, readiness)
}

/// The bound the cases' standby reads synced under: no case waits it
/// out.
const BOUND: Duration = Duration::from_secs(60);

#[test]
fn a_leader_and_a_follower_are_ready_and_every_state_has_a_name() {
    assert!(State::Leader.is_ready());
    assert!(State::Follower.is_ready());
    assert!(!State::Starting.is_ready());
    assert!(!State::Draining.is_ready());
    assert!(!State::Syncing.is_ready());
    assert_eq!(
        State::ALL.map(State::as_str),
        ["starting", "leader", "syncing", "follower", "draining"]
    );
}

#[test]
fn the_role_is_the_fences_at_the_instant_of_the_read() {
    // A lease that is handed in may have a holder this candidate never
    // saw, so its take waits to lead.
    let settings = ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(100),
        renew_deadline: Duration::from_millis(400),
    };
    let (mut elector, phase, _steps, readiness) = over(
        Elector::new(
            Box::new(MemoryLease::new()),
            "controller-0@127.0.0.1:1",
            settings,
        )
        .expect("the settings are in order"),
    );
    assert_eq!(readiness.state(), State::Starting);

    // Serving with no lease held: a follower.
    phase.send_replace(Phase::Serving);
    assert_eq!(readiness.state(), State::Follower);

    // The take of a lease that held no record opens no fence: still a
    // follower, through the renewals inside the wait. The lead opens one
    // renew deadline after the take's write, between two steps: the
    // leader from that instant, with no step taken and no message sent.
    // The renewals are a chain ahead of the clock, each just short of
    // the deadline the one before it stamped: the fence stays open
    // seconds past the opening, so the read needs no narrow window.
    let taken = Instant::now();
    assert_eq!(
        elector.tick(taken).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(readiness.state(), State::Follower);
    let mut renewed = taken;
    for _ in 0..10 {
        renewed += settings.renew_deadline - Duration::from_millis(10);
        assert_eq!(
            elector.tick(renewed).expect("the renewal"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(readiness.state(), State::Follower);
    }
    thread::sleep(settings.renew_deadline);
    assert_eq!(readiness.state(), State::Leader);

    // Draining says so whatever the fence says.
    phase.send_replace(Phase::Draining);
    assert_eq!(readiness.state(), State::Draining);
}

#[test]
fn a_fence_past_its_deadline_reads_as_a_follower_though_no_step_said_so() {
    let (mut elector, phase, _steps, readiness) = fenced();
    phase.send_replace(Phase::Serving);

    // A last renewal dated a second past the renew deadline: the holder
    // still holds the record, and its fence is closed at the read.
    let stale = Instant::now()
        .checked_sub(ElectionSettings::default().renew_deadline + Duration::from_secs(1))
        .expect("an instant three seconds back");
    elector.lead_at(stale);
    assert!(elector.is_holder());
    assert_eq!(readiness.state(), State::Follower);

    // The next renewal reopens it.
    assert_eq!(
        elector.tick(Instant::now()).expect("the renewal"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(readiness.state(), State::Leader);
}

/// A wait that fails where a state that never comes would hang it.
async fn within(reader: &mut Readiness, wanted: State) -> Option<State> {
    tokio::time::timeout(Duration::from_secs(10), reader.wait_for(wanted))
        .await
        .expect("the wait ends within ten seconds")
}

#[tokio::test]
async fn a_reader_sees_the_transitions_and_the_writer_leaving() {
    let (mut elector, phase, steps, mut reader) = fenced();
    assert_eq!(
        within(&mut reader, State::Starting).await,
        Some(State::Starting)
    );

    phase.send_replace(Phase::Serving);
    assert_eq!(
        within(&mut reader, State::Follower).await,
        Some(State::Follower)
    );

    // The waiter looks again after an election step.
    let mut waiter = reader.clone();
    let (waited, ()) = tokio::join!(within(&mut waiter, State::Leader), async {
        tokio::task::yield_now().await;
        elector.lead_at(Instant::now());
        steps.send_replace(ElectionObservation::default());
    });
    assert_eq!(waited, Some(State::Leader));
    assert_eq!(reader.clone().state(), State::Leader);

    // The election gone, the run loop's moves still wake the waiter.
    drop(steps);
    let mut waiter = reader.clone();
    let (waited, ()) = tokio::join!(within(&mut waiter, State::Draining), async {
        tokio::task::yield_now().await;
        phase.send_replace(Phase::Draining);
    });
    assert_eq!(waited, Some(State::Draining));

    drop(phase);
    assert_eq!(within(&mut reader, State::Leader).await, None);
}

/// The clock alone turns a leader's state where its fence's window
/// ends: a waiter is woken there, though the election took no step, no
/// frame came and the run loop did not move.
#[tokio::test]
async fn a_waiter_is_woken_where_the_fences_window_ends_with_no_step() {
    let (mut elector, phase, _steps, mut reader) = fenced();
    phase.send_replace(Phase::Serving);
    reader.lag.follow();
    // A take dated so that its window ends a tenth of a second from
    // now: a renewal that stalls from here closes the fence then.
    let now = Instant::now();
    let ends = now + Duration::from_millis(100);
    let taken = ends
        .checked_sub(ElectionSettings::default().renew_deadline)
        .expect("an instant a renew deadline back");
    elector.lead_at(taken);
    let (sample, turn) = reader.sample_at(now);
    assert_eq!((sample.state, turn), (State::Leader, Some(ends)));
    assert_eq!(
        within(&mut reader, State::Syncing).await,
        Some(State::Syncing)
    );
}

#[test]
fn a_follower_that_runs_a_standby_is_ready_only_while_it_is_synced() {
    let (mut elector, phase, _steps, readiness) = fenced();
    phase.send_replace(Phase::Serving);
    let lag = &readiness.lag;
    let read = |at| readiness.sample_at(at).0;
    let sample = |state, lag, synced| Sample {
        state,
        lag,
        synced,
        restores: 0,
    };

    // No standby runs: a follower is ready, as before one existed.
    let start = Instant::now();
    assert_eq!(read(start), sample(State::Follower, None, false));

    // The serve runs one: not ready before its first frame.
    lag.follow();
    assert_eq!(read(start), sample(State::Syncing, None, false));

    // A frame: ready while it is younger than the bound, not at it.
    lag.applied(start);
    // The lag holds the frame's instant at a whole microsecond.
    let frame = start - lag.age(start).expect("a frame applied");
    let under = BOUND - Duration::from_micros(1);
    assert_eq!(
        read(frame + under),
        sample(State::Follower, Some(under), true)
    );
    assert_eq!(
        read(frame + BOUND),
        sample(State::Syncing, Some(BOUND), false)
    );

    // The lapse comes with the state, by the same read of the clock.
    assert_eq!(readiness.sample_at(frame + under).1, Some(frame + BOUND));
    assert_eq!(readiness.sample_at(frame + BOUND).1, None);

    // The next frame ends the silence.
    lag.applied(frame + BOUND);
    assert_eq!(
        read(frame + BOUND),
        sample(State::Follower, Some(Duration::ZERO), true)
    );

    // The fence says leader whatever the lag says, and a leader
    // follows nobody: it has no lag and is not synced.
    let taken = Instant::now();
    assert_eq!(
        elector.tick(taken).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    assert_eq!(read(taken), sample(State::Leader, None, false));

    // A leader has no sync to lapse: the clock turns its state where
    // its fence's window ends, one renew deadline past the take.
    let ends = taken + ElectionSettings::default().renew_deadline;
    assert_eq!(readiness.sample_at(taken).1, Some(ends));
    assert_eq!(read(ends), sample(State::Leader, None, false));
    // Past it the fence is closed, and the sync's lapse alone is ahead:
    // the last frame is dated after this instant.
    let past = ends + Duration::from_nanos(1);
    assert_eq!(
        read(past),
        sample(State::Follower, Some(Duration::ZERO), true)
    );
    assert_eq!(readiness.sample_at(past).1, Some(frame + BOUND + BOUND));

    // Draining says so whatever the lag says.
    phase.send_replace(Phase::Draining);
    assert_eq!(readiness.state(), State::Draining);
}

#[tokio::test]
async fn a_waiter_looks_again_when_a_frame_ends_a_silence_and_when_the_sync_lapses() {
    // A frame ends the silence: the lag tells the waiter, with no move
    // of the run loop and no step of the election.
    let (elector, phase, _steps, mut reader) = fenced();
    phase.send_replace(Phase::Serving);
    let lag = Arc::clone(&reader.lag);
    lag.follow();
    assert_eq!(
        within(&mut reader, State::Syncing).await,
        Some(State::Syncing)
    );
    let mut waiter = reader.clone();
    let (waited, ()) = tokio::join!(within(&mut waiter, State::Follower), async {
        tokio::task::yield_now().await;
        lag.applied(Instant::now());
    });
    assert_eq!(waited, Some(State::Follower));

    // The sync lapses by the clock alone, under a bound short enough
    // to wait out: the waiter is at the frame well inside it.
    let short = Arc::new(Lag::new(Duration::from_millis(300)));
    short.follow();
    let (_standing, stepped) = watch::channel(ElectionObservation::default());
    let (phase, mut reader) = detached(elector.fence(), stepped, Arc::clone(&short));
    phase.send_replace(Phase::Serving);
    short.applied(Instant::now());
    assert_eq!(
        within(&mut reader, State::Syncing).await,
        Some(State::Syncing)
    );
}

#[tokio::test(start_paused = true)]
async fn a_waiter_wakes_at_the_lapse_it_read_and_no_later() {
    const HOUR: Duration = Duration::from_secs(3600);
    // A timer is a millisecond coarse.
    const COARSE: Duration = Duration::from_millis(2);
    let (_elector, _phase, _steps, mut reader) = fenced();
    // The paused clock steps to the next timer: its reading at the
    // wake is the instant the waiter slept to.
    let start = tokio::time::Instant::now();
    let held = Duration::from_millis(300);
    let lapse = start.into_std() + held;
    let woke = tokio::time::timeout(HOUR, reader.moved_by(Some(lapse))).await;
    assert!(matches!(woke, Ok(Some(()))), "{woke:?}");
    let late = start.elapsed().checked_sub(held);
    assert!(late.is_some_and(|late| late < COARSE), "{late:?}");

    // A lapse already past wakes at once: a sync that lapsed since
    // its read is read again.
    let past = tokio::time::Instant::now();
    let woke = tokio::time::timeout(HOUR, reader.moved_by(Some(lapse))).await;
    assert!(matches!(woke, Ok(Some(()))), "{woke:?}");
    assert!(past.elapsed() < COARSE, "{:?}", past.elapsed());

    // With no lapse the waiter sleeps on.
    let woke = tokio::time::timeout(HOUR, reader.moved_by(None)).await;
    assert!(woke.is_err(), "{woke:?}");
}

#[test]
fn a_sample_tells_the_restores_the_standby_asked_whatever_the_state() {
    let (mut elector, phase, _steps, readiness) = fenced();
    phase.send_replace(Phase::Serving);
    let lag = &readiness.lag;
    lag.follow();
    let asked = |at| readiness.sample_at(at).0.restores;
    assert_eq!(asked(Instant::now()), 0);
    lag.asked();
    lag.asked();
    assert_eq!(asked(Instant::now()), 2);
    // The count is the replica's since its start: a take keeps it.
    assert_eq!(
        elector.tick(Instant::now()).expect("the take"),
        TickOutcome::Acquired { term: 1 }
    );
    let sample = readiness.sample_at(Instant::now()).0;
    assert_eq!((sample.state, sample.restores), (State::Leader, 2));
}

/// A host reads the replica's membership log through the handle: where
/// it stands, and the writes behind a mark in the order written. A place
/// the log no longer holds is told as the log tells it.
#[test]
fn a_handle_reads_the_logs_head_and_the_writes_behind_a_mark() {
    let log = Arc::new(Log::new(2));
    let (_elector, _phase, _steps, handle) = fenced();
    let readiness = Readiness {
        log: Arc::clone(&log),
        ..handle
    };
    let write = |epoch| {
        let change = MembershipChange::Relabelled {
            id: WorkerId(epoch),
        };
        assert!(log.member(None, None, epoch, change, None, None));
    };
    write(1);
    let mark = readiness.log_head();
    assert_eq!(mark, log.head());
    write(2);
    write(3);
    let mut out = Vec::new();
    assert_eq!(readiness.log_after(mark, Log::BATCH, &mut out), Ok(()));
    let epochs = out.iter().filter_map(|entry| match entry.event {
        Event::Member { epoch, .. } => Some(epoch),
        Event::Term | Event::Generation { .. } => None,
    });
    assert_eq!(epochs.collect::<Vec<_>>(), [2, 3]);
    assert_eq!(out.last().map(|entry| entry.at), Some(readiness.log_head()));
    // A call reads no more than its caller asks.
    assert_eq!(readiness.log_after(mark, 1, &mut out), Ok(()));
    assert_eq!(out.len(), 1);
    // A ring of two lets the mark go at the next write.
    write(4);
    assert_eq!(readiness.log_after(mark, 1, &mut out), Err(LogPositionLost));
}

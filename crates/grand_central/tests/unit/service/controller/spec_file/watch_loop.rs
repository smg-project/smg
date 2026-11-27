//! The serve's loop on a clock the case holds: the wait between two
//! looks, and the stop with a look in flight or between two looks.

use super::*;

/// Waits on the real clock, five seconds at most, until look `n` is in
/// flight.
async fn in_flight(looks: &AtomicUsize, n: usize) {
    for _ in 0..5_000 {
        if looks.load(Ordering::SeqCst) >= n {
            return;
        }
        tokio::task::yield_now().await;
        thread::sleep(Duration::from_millis(1));
    }
    panic!("look {n} did not start");
}

/// The serve's loop on a clock the case holds. It takes one look a poll
/// after the last one ended: a look that stalls for three polls is
/// followed a poll after it ends, never at once. The stop ends the loop
/// with a look in flight. The look is handed in and writes down the
/// clock's instant: the verdict is those instants, read off the held
/// clock. A loop that looks without end fails at the look past the
/// case's bound, and the loop's end is waited on the real clock, so the
/// case hangs on neither.
#[tokio::test(start_paused = true)]
async fn the_watch_looks_a_poll_after_the_last_look_and_ends_at_the_stop() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let file = bed.start();
    let began = tokio::time::Instant::now();
    let looks = Arc::new(AtomicUsize::new(0));
    let stalls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let (release, released) = mpsc::channel::<()>();
    let counted = {
        let (looks, stalls, seen) = (Arc::clone(&looks), Arc::clone(&stalls), Arc::clone(&seen));
        move |path: &Path| {
            seen.lock().push(began.elapsed());
            let n = looks.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(n <= 20, "the loop looks without end");
            if n == stalls.load(Ordering::SeqCst) {
                released.recv().expect("the case lets the look go");
            }
            look(path)
        }
    };
    let (stop, stopped) = watch::channel(false);
    let mut watching = JoinSet::new();
    watching.spawn(file.run_looking(stopped, counted));

    // No look before a poll has passed, then a step a poll: the first
    // sees the file moved, the second reads it between two looks.
    replace(&bed.path, RANDOM.as_bytes());
    sleep(POLL / 2).await;
    assert!(seen.lock().is_empty(), "no look before a poll has passed");
    sleep(2 * POLL).await;
    assert_eq!(*seen.lock(), [POLL, 2 * POLL, 2 * POLL]);
    assert_eq!(bed.generation(), 2);

    // The third step's look stalls while three polls pass: the next
    // comes a poll after it ended, not at once.
    stalls.store(4, Ordering::SeqCst);
    tokio::time::advance(POLL / 2).await;
    in_flight(&looks, 4).await;
    tokio::time::advance(3 * POLL).await;
    let let_go = began.elapsed();
    release.send(()).expect("the look waits");
    sleep(POLL * 3 / 2).await;
    let next = seen.lock()[4..].to_vec();
    assert_eq!(next, [let_go + POLL], "a poll after the stalled look");

    // The stop ends the loop with a look in flight.
    stalls.store(6, Ordering::SeqCst);
    tokio::time::advance(POLL / 2).await;
    in_flight(&looks, 6).await;
    stop.send_replace(true);
    ended(&mut watching).await.expect("the loop's last join");
    release.send(()).expect("the look waits");
}

/// The stop ends the loop between two looks as it ends it with a look
/// in flight: at once. The held clock has not moved when the loop has
/// ended, and no look followed the stop.
#[tokio::test(start_paused = true)]
async fn the_stop_ends_the_watch_between_two_looks() {
    let bed = bed(LEAST_LOAD.as_bytes());
    let file = bed.start();
    let began = tokio::time::Instant::now();
    let looks = Arc::new(AtomicUsize::new(0));
    let counted = {
        let looks = Arc::clone(&looks);
        move |path: &Path| {
            looks.fetch_add(1, Ordering::SeqCst);
            look(path)
        }
    };
    let (stop, stopped) = watch::channel(false);
    let mut watching = JoinSet::new();
    watching.spawn(file.run_looking(stopped, counted));

    // One look, and half a poll into the wait for the next.
    sleep(POLL * 3 / 2).await;
    assert_eq!(looks.load(Ordering::SeqCst), 1);
    let stopped_at = began.elapsed();
    stop.send_replace(true);
    ended(&mut watching).await.expect("the loop's last join");
    assert_eq!(began.elapsed(), stopped_at, "the loop waited no poll out");
    assert_eq!(looks.load(Ordering::SeqCst), 1, "no look after the stop");
}

use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use tracing_appender::non_blocking::DEFAULT_BUFFERED_LINES_LIMIT;

use super::*;

/// A sink whose first `write` blocks until released: a file write that the
/// kernel is throttling, as seen from the thread that issued it.
struct StalledSink {
    gate: Option<mpsc::Receiver<()>>,
}

impl Write for StalledSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(gate) = self.gate.take() {
            let _ = gate.recv();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A sink that forwards every write on a channel, so a test can see
/// whether a line reached the writer thread.
struct ChannelSink(mpsc::Sender<Vec<u8>>);

impl Write for ChannelSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _ = self.0.send(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A sink that keeps every write as it was given: a test reads the write
/// boundaries as well as the bytes.
#[derive(Clone, Default)]
struct RecordingSink(Arc<Mutex<Vec<Vec<u8>>>>);

impl RecordingSink {
    fn writes(&self) -> Vec<Vec<u8>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// The writes once they hold `len` bytes, or after ten seconds.
    fn writes_of(&self, len: usize) -> Vec<Vec<u8>> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let writes = self.writes();
            if writes.iter().map(Vec::len).sum::<usize>() >= len || Instant::now() > deadline {
                return writes;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Write for RecordingSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn numbered_lines(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("{i:06} finished processing request\n").into_bytes())
        .collect()
}

/// Every write ends at a line boundary, and the writes in their order are
/// exactly `lines`: no line split, lost, repeated or reordered.
fn assert_whole_lines_in_order(writes: &[Vec<u8>], lines: &[Vec<u8>]) {
    assert!(
        writes.iter().all(|write| write.ends_with(b"\n")),
        "every write ends at a line boundary"
    );
    let (written, given) = (writes.concat(), lines.concat());
    assert_eq!(written.len(), given.len(), "every byte given is written");
    assert!(written == given, "the lines are written as given, in order");
}

/// The writer threads belong to the process, not to the value
/// `init_logging` returns: a router started after the first in one
/// process logs through the same subscriber, whose writer must still be
/// running once the first `startup` has returned and dropped its guard.
#[test]
fn the_writer_outlives_the_handle_init_returns() {
    let (tx, rx) = mpsc::channel();
    let mut writer = install(ChannelSink(tx), "test-log-outlives");
    {
        let _handle = LogGuard(());
    }
    writer
        .write_all(b"after the handle dropped\n")
        .expect("a lossy writer reports every line as written");
    let line = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the writer thread still runs after the handle dropped");
    assert_eq!(line, b"after the handle dropped\n");
}

/// Lines a full queue drops are counted per sink and reported as the
/// total and the delta since the last report.
#[test]
fn dropped_lines_are_counted_and_reported_per_sink() {
    let (release, gate) = mpsc::channel();
    let mut writer = install(StalledSink { gate: Some(gate) }, "test-log-dropped");
    // More lines than the queue and the writer's batch hold together: the
    // batch takes `BATCH_BYTES / LINE.len() + 1` lines at most before its
    // write stalls on the sink.
    const LINE: &[u8] = b"finished processing request\n";
    for _ in 0..DEFAULT_BUFFERED_LINES_LIMIT + BATCH_BYTES / LINE.len() + 1_000 {
        writer
            .write_all(LINE)
            .expect("a lossy writer reports every line as written");
    }
    let counted = dropped_lines()
        .into_iter()
        .find(|(name, _)| *name == "test-log-dropped")
        .map(|(_, dropped)| dropped)
        .expect("the sink is counted");
    assert!(counted > 0, "overflow is counted");
    let (_, total, since_last) = report_dropped_lines()
        .into_iter()
        .find(|(name, _, _)| *name == "test-log-dropped")
        .expect("the sink is reported");
    assert_eq!(total, counted);
    assert_eq!(since_last, counted, "the first report carries everything");
    let (_, _, since_last) = report_dropped_lines()
        .into_iter()
        .find(|(name, _, _)| *name == "test-log-dropped")
        .expect("the sink is reported");
    assert_eq!(since_last, 0, "a second report carries only new drops");
    let _ = release.send(());
}

#[test]
fn a_stalled_sink_never_blocks_the_emitting_thread() {
    let (release, gate) = mpsc::channel();
    let (mut writer, guard) = non_blocking(StalledSink { gate: Some(gate) }, "test-log-stalled");
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        // More lines than the channel holds, while the sink accepts none.
        for _ in 0..DEFAULT_BUFFERED_LINES_LIMIT + 1_000 {
            writer
                .write_all(b"finished processing request\n")
                .expect("a lossy writer reports every line as written");
        }
        let _ = done.send(writer.error_counter().dropped_lines());
    });
    let dropped = finished.recv_timeout(Duration::from_secs(10));
    // Unblock the writer thread whatever happened, so a failure does not leak it.
    let _ = release.send(());
    let dropped = dropped.expect("writes against a stalled sink must return, not wait for it");
    assert!(
        dropped > 0,
        "the queue is bounded: overflow is dropped, never waited for"
    );
    drop(guard);
}

/// Lines that arrive within the window leave in one write, whole and in
/// order: the write and the wake-up per line are gone.
#[test]
fn lines_within_the_window_leave_in_one_write() {
    let sink = RecordingSink::default();
    let (mut writer, guard) = non_blocking(
        Batched::with(sink.clone(), Duration::from_millis(50), BATCH_BYTES),
        "test-log-batched-window",
    );
    let lines = numbered_lines(1_000);
    for line in &lines {
        writer
            .write_all(line)
            .expect("a lossy writer reports every line as written");
    }
    assert_eq!(writer.error_counter().dropped_lines(), 0);
    drop(guard);
    let writes = sink.writes_of(lines.concat().len());
    assert_whole_lines_in_order(&writes, &lines);
    assert!(
        writes.len() <= lines.len() / 100,
        "{} writes for {} lines",
        writes.len(),
        lines.len()
    );
}

/// A line with no company still leaves, once the window has passed.
#[test]
fn a_lone_line_leaves_after_the_window() {
    let sink = RecordingSink::default();
    let (mut writer, guard) = non_blocking(
        Batched::with(sink.clone(), Duration::from_millis(5), BATCH_BYTES),
        "test-log-batched-lone",
    );
    let line = b"a line on its own\n".to_vec();
    writer
        .write_all(&line)
        .expect("a lossy writer reports every line as written");
    assert_eq!(sink.writes_of(line.len()), vec![line]);
    drop(guard);
}

/// A batch that reaches the byte cap is written inside the window, and the
/// rest of the backlog follows without another wait.
#[test]
fn the_byte_cap_writes_a_batch_inside_the_window() {
    let cap = 1_024;
    let sink = RecordingSink::default();
    let (mut writer, guard) = non_blocking(
        Batched::with(sink.clone(), Duration::from_millis(200), cap),
        "test-log-batched-cap",
    );
    // 36 KB in all: some 35 batches of the cap.
    let lines = numbered_lines(1_000);
    let started = Instant::now();
    for line in &lines {
        writer
            .write_all(line)
            .expect("a lossy writer reports every line as written");
    }
    drop(guard);
    let writes = sink.writes_of(lines.concat().len());
    let drained_in = started.elapsed();
    assert_whole_lines_in_order(&writes, &lines);
    let longest = writes.iter().map(Vec::len).max().unwrap_or(0);
    assert!(
        longest < cap + lines[0].len(),
        "a batch is written once it reaches the cap, not later: {longest} bytes"
    );
    assert!(writes.len() >= 30, "{} writes", writes.len());
    assert!(
        drained_in < Duration::from_secs(4),
        "the backlog drained without a window per batch: {drained_in:?}"
    );
}

/// The flush at shutdown writes the lines still in the window: a clean
/// exit loses nothing.
#[test]
fn shutdown_writes_the_lines_still_in_the_window() {
    let sink = RecordingSink::default();
    let (mut writer, guard) = non_blocking(
        Batched::with(sink.clone(), Duration::from_millis(20), BATCH_BYTES),
        "test-log-batched-shutdown",
    );
    let lines = numbered_lines(10);
    for line in &lines {
        writer
            .write_all(line)
            .expect("a lossy writer reports every line as written");
    }
    // The writer is inside its window; the guard's drop waits for the flush.
    drop(guard);
    assert_whole_lines_in_order(&sink.writes(), &lines);
}

/// A sink that fails its writes (a closed stdout pipe, a full disk) does
/// not break the shutdown handshake: the writer thread takes the shutdown
/// message, meets the guard and ends, instead of blocking on its queue for
/// good while the guard waits out its rendezvous timeout; the batch is
/// gone, as a line whose own write failed was before.
#[test]
fn a_failing_sink_does_not_break_the_shutdown_handshake() {
    /// Refuses every write; sets `ended` when it is dropped, which is when
    /// the writer thread that owns it has ended.
    struct FailingSink {
        ended: Arc<AtomicBool>,
    }

    impl Write for FailingSink {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for FailingSink {
        fn drop(&mut self) {
            self.ended.store(true, Ordering::SeqCst);
        }
    }

    let ended = Arc::new(AtomicBool::new(false));
    let sink = FailingSink {
        ended: ended.clone(),
    };
    // A window: the writer thread sleeps inside its first `write` while
    // the guard's drop queues the shutdown message, so the flush that
    // fails is the one that ends the shutdown drain, every time (with no
    // window the worker may drain the line alone, fail its flush before
    // the shutdown message is queued, and pass against the old flush).
    let (mut writer, guard) = non_blocking(
        Batched::with(sink, Duration::from_millis(50), BATCH_BYTES),
        "test-log-batched-failing",
    );
    writer
        .write_all(b"a line the sink refuses\n")
        .expect("a lossy writer reports every line as written");
    let started = Instant::now();
    drop(guard);
    let dropped_in = started.elapsed();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ended.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        ended.load(Ordering::SeqCst),
        "the writer thread ended after the guard dropped (the drop took {dropped_in:?})"
    );
}

/// A sink whose first write takes `delay`: a file write the kernel holds
/// for longer than the guard's moment.
struct SlowFirstWrite {
    delay: Option<Duration>,
    sink: RecordingSink,
}

impl Write for SlowFirstWrite {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(delay) = self.delay.take() {
            thread::sleep(delay);
        }
        self.sink.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The close waits for a writer whose last write outlasts the guard's
/// moment (about a second): the line logged before the close is in the
/// sink when the close returns, not written after it.
#[test]
fn the_close_waits_for_a_writer_held_past_the_guards_moment() {
    let recording = RecordingSink::default();
    let (mut writer, sink) = start_sink(
        Batched::with(
            SlowFirstWrite {
                delay: Some(Duration::from_millis(1_500)),
                sink: recording.clone(),
            },
            Duration::ZERO,
            BATCH_BYTES,
        ),
        "test-log-close-waits",
    );
    let line = b"a line logged before the close\n".to_vec();
    writer
        .write_all(&line)
        .expect("a lossy writer reports every line as written");
    // The writer thread is inside its slow write when the close begins.
    thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    let all_ended = close_sinks(vec![sink]);
    let took = started.elapsed();
    assert!(
        all_ended,
        "the writer thread ended within the close's wait (the close took {took:?})"
    );
    assert_eq!(
        recording.writes(),
        vec![line],
        "the line is in the sink when the close returns"
    );
}

/// A writer with nothing left to write ends inside the guard's handshake:
/// the close costs no wait after it. The bound is half of [`CLOSE_WAIT`]:
/// a close that waited the whole of it for a writer already ended would
/// fail, while a guard handshake that runs out its own second on a busy
/// runner (the idle writer thread scheduled late) would not. The time is
/// the output, for the record (`--nocapture`).
#[test]
#[expect(clippy::print_stdout, reason = "the measurement is the output")]
fn the_close_returns_as_soon_as_an_idle_writer_has_ended() {
    let recording = RecordingSink::default();
    let (mut writer, sink) = start_sink(
        Batched::with(recording.clone(), Duration::ZERO, BATCH_BYTES),
        "test-log-close-idle",
    );
    let line = b"a line written long before the close\n".to_vec();
    writer
        .write_all(&line)
        .expect("a lossy writer reports every line as written");
    assert_eq!(recording.writes_of(line.len()), vec![line]);
    let started = Instant::now();
    assert!(close_sinks(vec![sink]), "an idle writer ends in time");
    let took = started.elapsed();
    println!("idle close took {took:?}");
    assert!(
        took < CLOSE_WAIT / 2,
        "an idle writer ends inside the guard's handshake, with no wait after it: {took:?}"
    );
}

/// Writes per line on a steady stream, with the batching and without: a
/// measurement (`--ignored --nocapture`), not a check.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the measurement is the output")]
fn writes_per_line_on_a_steady_stream() {
    const LINES: usize = 20_000;
    const INTERVAL: Duration = Duration::from_micros(250);
    fn stream<W: Write + Send + 'static>(
        sink: W,
        recording: &RecordingSink,
        name: &str,
    ) -> (usize, Duration) {
        let (mut writer, guard) = non_blocking(sink, name);
        let lines = numbered_lines(LINES);
        let started = Instant::now();
        let mut due = started;
        for line in &lines {
            due += INTERVAL;
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }
            writer
                .write_all(line)
                .expect("a lossy writer reports every line as written");
        }
        let elapsed = started.elapsed();
        drop(guard);
        let writes = recording.writes_of(lines.concat().len());
        assert_whole_lines_in_order(&writes, &lines);
        (writes.len(), elapsed)
    }
    let plain = RecordingSink::default();
    let (writes, elapsed) = stream(plain.clone(), &plain, "test-log-stream-plain");
    println!(
        "plain: {LINES} lines in {elapsed:?} ({:.0} lines/s), {writes} writes, \
             {:.3} writes per line",
        LINES as f64 / elapsed.as_secs_f64(),
        writes as f64 / LINES as f64
    );
    let batched = RecordingSink::default();
    let (writes, elapsed) = stream(
        Batched::new(batched.clone()),
        &batched,
        "test-log-stream-batched",
    );
    println!(
        "batched: {LINES} lines in {elapsed:?} ({:.0} lines/s), {writes} writes, \
             {:.3} writes per line",
        LINES as f64 / elapsed.as_secs_f64(),
        writes as f64 / LINES as f64
    );
}

//! A lost stream: how it is counted, and how the host opens it again.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use smg_grand_central::plugins::{
    grpc::{Plugin, Runtime},
    shared::{BreakerSpec, OpenDecisions},
};

use crate::{
    decide, decide_pair, decide_short, facts, lead, load, made_again, pool, profile, runtime,
    single_engine, spec,
    support::{
        grpc_plugin::{eventually, Behaviour, TestPlugin, ALL},
        said::{lines, said},
    },
};

#[test]
fn a_stream_that_ends_is_counted_once_and_reopens_behind_a_decision_a_pause_later() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "leaving");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(103),
        "the decision in flight falls back when its stream goes"
    );
    assert!(!plugin.is_warm());
    // The lane's state moved before the plugin was told: the count follows.
    assert!(eventually(|| plugin.stats().failed_streams == 1));
    let stats = plugin.stats();
    assert_eq!(
        (stats.failed_streams, stats.missed, stats.cold),
        (1, 0, 0),
        "the stream's end, not a missed deadline"
    );

    // The next decision does not wait for a connect: it falls back at once.
    // The stream is asked for behind a decision a pause after the failure.
    served.behave(Behaviour::Honest);
    assert_eq!(lead(&decide(&engine, &candidates, 2)), Some(103));
    assert_eq!(plugin.stats().cold, 1);
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 2);
            plugin.is_warm()
        }),
        "reopened behind a decision"
    );
    let before = plugin.stats();
    assert_eq!(lead(&decide(&engine, &candidates, 3)), Some(103));
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.answered - before.answered,
            stats.connects,
            stats.failed_streams
        ),
        (1, 2, 1)
    );
    assert_eq!(served.seen().hellos.len(), 2);
}

/// A stream's failure is told to the plugin through the host's clock, on
/// the I/O thread, and the lane's state moves first: here that read of the
/// clock panics, and the thread's next decisions still find the stream gone
/// and reopen it. Told first, the lane would read as open over a dead
/// stream, and no decision would ever reopen it.
#[test]
fn a_stream_that_ends_under_a_clock_that_panics_still_reopens() {
    let served = TestPlugin::serve();
    let armed = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&armed);
    let deciding = std::thread::current().id();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        // Off the deciding thread the clock is read where a stream's
        // failure is told: that read panics, once.
        assert!(
            std::thread::current().id() == deciding || !told.swap(false, Ordering::Relaxed),
            "the runtime's clock panics on the I/O thread"
        );
        1_000_000
    })
    .expect("a runtime");
    let plugin = load(&runtime, &served, "ends-under-a-panic");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
    assert_eq!(plugin.stats().answered, 1);
    served.behave(Behaviour::EndStream);
    armed.store(true, Ordering::Relaxed);
    // The decision in flight falls back when its stream goes.
    assert_eq!(lead(&decide_short(&engine, &candidates, 2)), Some(103));
    // The plugin reads its behaviour as it takes a frame, and the short
    // decision may give up before it took this one: it is told to be
    // honest again once it took the frame it ends the stream at.
    assert!(
        eventually(|| served.frames() == 2),
        "the plugin took the frame it ends its stream at"
    );
    served.behave(Behaviour::Honest);
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 3);
            plugin.stats().connects == 2
        }),
        "the lane was left open over a dead stream: {:?}",
        plugin.stats()
    );
    let before = plugin.stats().answered;
    assert_eq!(lead(&decide(&engine, &candidates, 4)), Some(103));
    assert_eq!(
        plugin.stats().answered - before,
        1,
        "the reopened stream answers"
    );
}

/// A lane that failed asks for its next stream no sooner than a pause
/// later, whatever the breaker is set to. Against a socket that refuses at
/// once, under a cool-down of zero, the decisions of a third of a second
/// ask for a handful of streams, not one each.
#[test]
fn a_refused_socket_under_a_cool_down_of_zero_is_asked_once_a_pause() {
    let mut served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "refused");
    spec.breaker = BreakerSpec {
        cooldown_micros: 0,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
    served.stop();
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the stream went with the server"
    );
    let started = Instant::now();
    let mut decisions = 0_u64;
    while started.elapsed() < Duration::from_millis(300) {
        assert_eq!(lead(&decide(&engine, &candidates, 2)), Some(103));
        decisions += 1;
    }
    let asked = plugin.stats().failed_streams - 1;
    assert!(
        decisions > 100,
        "a failed lane's decisions fall back at once"
    );
    assert!(
        asked <= 6,
        "{asked} streams asked for by {decisions} decisions in 300 ms"
    );
}

/// A stream that ends is one failure on the breaker, with no decision to
/// mark: under a threshold of one it opens, and the next decision is
/// bypassed.
#[test]
fn a_stream_that_ends_counts_on_the_breaker() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let mut spec = spec(&served, "leaving-once");
    spec.breaker = BreakerSpec {
        failures: 1,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
    assert!(
        eventually(|| plugin.stats().breaker_opens == 1),
        "the stream's end reached the breaker: {:?}",
        plugin.stats()
    );
    assert_eq!(lead(&decide(&engine, &candidates, 2)), Some(103));
    assert_eq!(plugin.stats().bypassed, 1);
}

/// A plugin that goes away ends every deciding thread's stream at once.
/// That is one outage and one failure on the breaker, not one a thread:
/// here two threads' streams end together under a threshold of two and the
/// breaker stays closed. The plugin stays away, a decision asks for its
/// stream again a pause later, and that failed open is the second failure:
/// the breaker opens, as it does for one thread.
#[test]
fn streams_that_end_together_are_one_failure_on_the_breaker() {
    let mut served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "gone-once");
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // A second deciding thread, with a stream of its own.
    std::thread::scope(|scope| {
        let warmed = scope.spawn(|| plugin.warm()).join();
        assert!(matches!(warmed, Ok(Ok(()))), "the second thread's stream");
    });
    // The count moves on the I/O thread, just after the lane reads ready.
    assert!(
        eventually(|| plugin.stats().connects == 2),
        "{:?}",
        plugin.stats()
    );
    served.stop();
    assert!(
        eventually(|| plugin.stats().failed_streams == 2),
        "both streams ended: {:?}",
        plugin.stats()
    );
    assert_eq!(
        plugin.stats().breaker_opens,
        0,
        "one outage is one failure: {:?}",
        plugin.stats()
    );
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 1);
            plugin.stats().breaker_opens == 1
        }),
        "a failed open a pause later is the second failure: {:?}",
        plugin.stats()
    );
}

/// The apply's ask does not wait the pause: a thread whose stream just
/// failed warms at once when the plugin answers again. Here a second server
/// takes the first's socket path before the first goes.
#[test]
fn the_applys_ask_does_not_wait_the_pause() {
    let mut first = TestPlugin::serve();
    let second = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &first, "warmed-again");
    let path = first.socket.clone();
    first.stop();
    std::fs::rename(&second.socket, &path).expect("the second server takes the first's path");
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "{:?}",
        plugin.stats()
    );
    let warmed = plugin.warm();
    assert!(
        warmed.is_ok(),
        "the ask inside the pause was refused: {warmed:?}"
    );
    // The count moves on the I/O thread, just after the lane reads ready.
    assert!(
        eventually(|| plugin.stats().connects == 2),
        "{:?}",
        plugin.stats()
    );
}

/// A broken stream's status is the plugin's words: the host's line carries
/// their start and no more than the bound. The words are three-byte
/// characters, so the cut falls inside one and backs up to its start.
#[test]
fn a_streams_status_is_cut_in_the_hosts_line() {
    said();
    let served = TestPlugin::serve();
    served.script(|s| {
        s.behaviour = Behaviour::EndStream;
        s.end_status = "\u{20ac}".repeat(1_000);
    });
    let runtime = runtime();
    let plugin = load(&runtime, &served, "wordy-status");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert!(
        eventually(|| !lines("wordy-status", "a plugin stream failed").is_empty()),
        "no line of the stream: {:?}",
        plugin.stats()
    );
    let line = lines("wordy-status", "a plugin stream failed").remove(0);
    let words = line.matches('\u{20ac}').count();
    assert!(
        (20..=85).contains(&words),
        "the line carries {words} of the plugin's characters"
    );
}

/// A stream that fails to open is told to the plugin like one that ends:
/// the lane's state moves first. Here the telling's read of the clock
/// panics after a failed open: the host says its task panicked, the lane
/// reads as failed all the same, and a decision a pause later asks for the
/// stream again. Told first, the lane would stay connecting for good.
#[test]
fn an_open_that_fails_under_a_clock_that_panics_is_asked_for_again() {
    let kept = said();
    let mut served = TestPlugin::serve();
    let armed = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&armed);
    let deciding = std::thread::current().id();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        assert!(
            std::thread::current().id() == deciding || !told.swap(false, Ordering::Relaxed),
            "the clock panics where a failed open is told"
        );
        1_000_000
    })
    .expect("a runtime");
    let plugin = load(&runtime, &served, "opens-under-a-panic");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
    served.stop();
    assert!(eventually(|| plugin.stats().failed_streams == 1));
    armed.store(true, Ordering::Relaxed);
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 2);
            !armed.load(Ordering::Relaxed)
        }),
        "no failed open was told: {:?}",
        plugin.stats()
    );
    assert!(
        eventually(|| {
            kept.lock().expect("the kept lines").iter().any(|line| {
                line.contains("task panicked") && line.contains("a failed open is told")
            })
        }),
        "the host said nothing of the task that panicked"
    );
    let before = plugin.stats().failed_streams;
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 3);
            plugin.stats().failed_streams > before
        }),
        "the lane was left connecting over a failed open: {:?}",
        plugin.stats()
    );
}

/// The hello's ack is awaited under the connect frame on the I/O thread
/// too. A plugin that takes a cold thread's stream and never acks leaves
/// it failed at the frame's end, and the thread's later decisions ask for
/// it again.
#[test]
fn a_hello_never_acked_fails_a_cold_threads_stream_at_the_connect_frame() {
    let runtime = runtime();
    // The frame is short for the cold stream's sake, and the load's own
    // hello is acked under it too: a load the runner held past it is made
    // again, on a plugin that has seen no hello.
    let until = Instant::now() + Duration::from_secs(60);
    let (served, plugin) = loop {
        let served = TestPlugin::serve();
        let mut spec = spec(&served, "mute-hello");
        spec.connect_micros = 100_000;
        match Plugin::load(&runtime, spec) {
            Ok(plugin) => break (served, plugin),
            Err(why) => {
                assert!(Instant::now() < until, "the plugin loads: {why:?}");
                made_again(
                    "a mute hello's load",
                    "the load's own hello was acked past the connect frame",
                );
            }
        }
    };
    served.script(|s| s.hello_delay = Duration::from_secs(30));
    let engine = single_engine(profile(&plugin, ALL));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // This thread's stream was never opened: it opens behind the
            // thread's first decision, which falls back.
            assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
            assert!(
                eventually(|| plugin.stats().failed_streams == 1),
                "the wait for the ack ended: {:?}",
                plugin.stats()
            );
            assert!(
                eventually(|| {
                    decide(&engine, &pool(4), 2);
                    served.seen().hellos.len() >= 3
                }),
                "the failed stream is asked for again"
            );
        });
    });
}

/// A stream that goes takes its frames in flight with it and no more: the
/// places of its window that were free serve the stream that reopens. One
/// frame is in flight when the stream goes. A pair then has two in flight
/// at once on the reopened stream, and the second takes a place the lost
/// stream left free.
#[test]
fn a_place_free_when_its_stream_went_serves_the_reopened_stream() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "leaving-a-free-place");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the stream went"
    );
    // The stream opens behind a decision, a pause after the failure.
    served.behave(Behaviour::Honest);
    assert!(
        eventually(|| {
            decide(&engine, &candidates, 2);
            plugin.is_warm()
        }),
        "reopened behind a decision"
    );
    drop(engine);
    let before = plugin.stats();
    assert!(decide_pair(&plugin, &facts(9)).pick().is_some());
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered - before.answered, stats.failed_streams),
        (2, 1),
        "both frames of the pair were answered on the reopened stream: {stats:?}"
    );
}

#[test]
fn a_thread_whose_stream_is_cold_takes_its_fallbacks_at_once() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "per-thread");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                assert!(!plugin.is_warm(), "this thread has no stream yet");
                assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(103));
                assert_eq!(
                    plugin.stats().cold,
                    1,
                    "no frame and no wait: the stream opens behind"
                );
                assert_eq!(served.frames(), 0);
                assert!(eventually(|| plugin.is_warm()));
                assert_eq!(lead(&decide(&engine, &candidates, 2)), Some(103));
                assert_eq!(plugin.stats().answered, 1);
            })
            .join()
            .expect("the deciding thread");
    });
    assert_eq!(plugin.stats().connects, 2, "one stream per deciding thread");
}

#[test]
fn a_dropped_plugin_ends_its_streams() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "leaving");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert!(
        eventually(|| served.seen().observe_hellos.len() == 1),
        "the observe stream opened behind the first decide stream"
    );
    drop(engine);
    drop(plugin);
    assert!(
        eventually(|| served.seen().ended_streams == 1),
        "the decide stream ended with the plugin"
    );
    assert!(
        eventually(|| served.seen().acked_observes == 1),
        "the observe stream ended and took its ack"
    );
}

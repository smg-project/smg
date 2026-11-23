//! The gRPC user plugin against a `PolicyPlugin` served in the test's own
//! process (`support::grpc_plugin`): what a load needs, what each binding
//! asks, the answer's validity rules, the budget and the breaker, a lost
//! stream, the lifecycle and both transports.
//!
//! The answered cases wait under a budget no loaded host misses, and the
//! cases that miss do so against a plugin that never answers (or answers
//! many frames later), so no case reads a race between an answer and its
//! deadline.
#![expect(
    clippy::expect_used,
    reason = "the served plugin, the certificates and the loads run inside the tests; a failure there is the test's failure"
)]

mod support {
    pub mod grpc_plugin;
    pub mod said;
    #[cfg(feature = "wasm")]
    #[expect(
        dead_code,
        reason = "the one case over both kinds loads one guest; the others are the WebAssembly binary's"
    )]
    pub mod wasm_guest;
}

use std::{
    os::unix::fs::MetadataExt,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
#[cfg(feature = "wasm")]
use smg_policy_controller::plugins::wasm;
use smg_policy_controller::{
    document::Needs,
    engine::{
        Binding, Candidate, Constraints, Decision, Engine, Filter, Header, Health, Identity,
        PairPools, Picker, Plan, PoolFacts, Profile, RequestFacts, Scorer, Scratch, ShedReason,
        Target, Targets, TieBand, WeightedScorer, WorkerId,
    },
    plugins::{
        grpc::{
            GrpcFilter, GrpcPicker, GrpcPluginSpec, GrpcScorer, LoadError, Plugin, PluginBinding,
            Runtime, Tls, Transport,
        },
        shared::{BreakerSpec, Exposure, Failure, FrameBudget, HeaderSet, OpenDecisions, Reveal},
    },
    service::proto,
    stages::{Best, Healthy, LeastRequests, Lifecycle},
    tracking::{EventKind, EventSet, Lease, Membership, ReleaseReason, Side, Signal},
};
use support::{
    grpc_plugin::{eventually, Behaviour, Gate, ServerTls, TestPlugin, ALL, FILTER, PICK, SCORE},
    said::{lines, openings, panic_at, said, trap_spent},
};

/// The decision budget of the answered cases, and the connect frame of
/// every load: far past what a loaded host takes for one round trip.
const BUDGET_MICROS: u32 = 5_000_000;
/// The budget of the cases that miss on purpose.
const SHORT_MICROS: u32 = 20_000;

fn runtime() -> Arc<Runtime> {
    Runtime::new(&OpenDecisions::new()).expect("a runtime")
}

fn unix(served: &TestPlugin) -> Transport {
    Transport::Unix {
        path: served.socket.clone(),
        peer_uid: None,
    }
}

fn spec(served: &TestPlugin, name: &str) -> GrpcPluginSpec {
    let mut spec = GrpcPluginSpec::new(name, unix(served));
    spec.connect_micros = BUDGET_MICROS;
    // The plugin states its frame, as wide as an answered case's budget. A
    // plugin that states none takes the default frame, which has a case of
    // its own.
    spec.budget_micros = BUDGET_MICROS;
    // Every kind of event: the lifecycle's cases tell them, and a decision
    // reads none of it.
    spec.observes = EventSet::ALL;
    spec
}

fn load(runtime: &Arc<Runtime>, served: &TestPlugin, name: &str) -> Arc<Plugin> {
    Plugin::load(runtime, spec(served, name)).expect("the plugin loads")
}

/// A try or a round a hold of the test's runner can have ended says so: the
/// one line of a try that asserts nothing and is made again (read under
/// `--nocapture`), so a row of runs can say in how many a hold reached a
/// case.
#[expect(
    clippy::print_stdout,
    reason = "a try that is made again reports it on stdout under --nocapture"
)]
fn made_again(case: &str, why: &str) {
    println!("{case}: made again in this run: {why}");
}

/// The loads of the one WebAssembly guest a case makes at most.
#[cfg(feature = "wasm")]
const GUEST_LOADS: u32 = 10;

/// Loads the WebAssembly guest of the case over both kinds. Nothing watches
/// an `init`: the host ends it at its frame on the wall clock, so a runner
/// that holds the loading thread ends an `init` that did nothing. The load
/// claims nothing of that time, so it is made again whenever the host ended
/// its `init` at the frame, ten loads at most: the last load asserts what it
/// says. The WebAssembly plugin binary's loads that must succeed do the
/// same.
#[cfg(feature = "wasm")]
fn guest_loaded(runtime: &Arc<wasm::Runtime>, spec: &wasm::PluginSpec) -> Arc<wasm::Plugin> {
    use smg_policy_controller::plugins::wasm;

    let mut loads = 0;
    loop {
        let load = wasm::Plugin::load(runtime, spec.clone());
        loads += 1;
        let ended = matches!(
            &load,
            Err(wasm::LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        );
        if !ended || loads == GUEST_LOADS {
            return load.expect("the guest loads");
        }
        made_again(
            "a guest's load",
            "the host ended its init at the frame's deadline",
        );
    }
}

/// A profile that binds the plugin as `stages` and runs the built-in stage
/// elsewhere; with `fallbacks`, each bound stage falls back to the built-in
/// one. The one profile of its build that binds the plugin.
fn profile_of(plugin: &Arc<Plugin>, stages: u32, fallbacks: bool) -> Profile {
    bound_profile(plugin.bind(), stages, fallbacks)
}

/// `profile_of` over a bound set the caller took.
fn bound_profile(bound: Arc<PluginBinding>, stages: u32, fallbacks: bool) -> Profile {
    let filter: Box<dyn Filter> = if stages & FILTER != 0 {
        let fallback: Option<Box<dyn Filter>> = fallbacks.then(|| Box::new(Healthy) as _);
        Box::new(GrpcFilter::new(Arc::clone(&bound), fallback))
    } else {
        Box::new(Healthy)
    };
    let scorer: Box<dyn Scorer> = if stages & SCORE != 0 {
        let fallback: Option<Box<dyn Scorer>> = fallbacks.then(|| Box::new(LeastRequests) as _);
        Box::new(GrpcScorer::new(Arc::clone(&bound), fallback))
    } else {
        Box::new(LeastRequests)
    };
    let picker: Box<dyn Picker> = if stages & PICK != 0 {
        let fallback: Option<Box<dyn Picker>> = fallbacks.then(|| Box::new(Best) as _);
        Box::new(GrpcPicker::new(bound, fallback))
    } else {
        Box::new(Best)
    };
    Profile {
        name: "plugin".into(),
        filters: vec![filter],
        scorers: vec![WeightedScorer {
            scorer,
            weight: 1.0,
        }],
        picker,
    }
}

fn profile(plugin: &Arc<Plugin>, stages: u32) -> Profile {
    profile_of(plugin, stages, true)
}

/// A profile whose filter is one plugin and whose picker another, each
/// with its built-in fallback. The picker's frame leaves at the pick and
/// names what the decision has left by then.
fn filter_then_picker(filter: &Arc<Plugin>, picker: &Arc<Plugin>) -> Profile {
    let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
    let best: Option<Box<dyn Picker>> = Some(Box::new(Best));
    Profile {
        name: "two-plugins".into(),
        filters: vec![Box::new(GrpcFilter::new(filter.bind(), healthy))],
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(GrpcPicker::new(picker.bind(), best)),
    }
}

fn binding(plan: Plan, decode_profile: Option<u16>) -> Binding {
    Binding {
        model: "*".into(),
        phases: vec![],
        plan,
        profile: 0,
        decode_profile,
        shadow: None,
        decode_shadow: None,
        sessions: None,
        overload: None,
    }
}

fn single_engine(profile: Profile) -> Engine {
    Engine::new(
        vec![profile],
        vec![binding(Plan::Single, None)],
        1,
        Needs::default(),
    )
}

/// `n` healthy candidates (ids from 100) whose in-flight counts descend:
/// row 0 is the most loaded, the last row the least.
fn pool(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(u64::from(i) + 100),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight: n - i,
            ..Candidate::default()
        })
        .collect()
}

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        budget_micros: BUDGET_MICROS,
        seed,
        now_micros: 777_000,
        ..RequestFacts::default()
    }
}

fn pool_facts(candidates: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: candidates.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    }
}

fn decide_with(engine: &Engine, facts: &RequestFacts<'_>, candidates: &[Candidate]) -> Decision {
    engine.decide(
        facts,
        &pool_facts(candidates),
        candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

fn decide(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    decide_with(engine, &facts(seed), candidates)
}

/// A decision under the short budget: the frame's deadline is 20 ms.
fn decide_short(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(seed)
    };
    decide_with(engine, &short, candidates)
}

/// Decides under `facts` until `judged` of the plugin's frames were answered
/// or missed with the I/O thread seen running, whatever the host's load. A
/// frame the stream's window refuses, a cold stream, or a frame the I/O
/// thread stalled under (held off its core) is no sample, so a decision
/// that judged nothing gives the I/O thread a moment and is tried again.
fn decide_until_judged(
    engine: &Engine,
    plugin: &Plugin,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    judged: u64,
) {
    let count = || {
        let stats = plugin.stats();
        stats.answered + stats.missed - stats.stalls
    };
    let mut tries = 0_u32;
    while count() < judged {
        tries += 1;
        assert!(
            tries <= 10_000,
            "the plugin's frames do not leave: {:?}",
            plugin.stats()
        );
        let before = count();
        assert!(matches!(
            decide_with(engine, facts, candidates),
            Decision::Pick(_)
        ));
        if count() == before {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The lead target's id; a shed or a queue has none.
fn lead(decision: &Decision) -> Option<u64> {
    decision
        .pick()
        .and_then(|pick| pick.targets.first().map(|t| t.id.0))
}

// --- the hello and the load -------------------------------------------------

#[test]
fn the_hello_names_the_plugin_and_the_window_and_the_load_settles_the_stream() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "greeter");
    spec.config = b"mode=1".to_vec();
    spec.metrics = vec!["hits".to_owned(), "skips".to_owned()];
    spec.view = Exposure {
        tenant: Reveal::Hash,
        session_key: Reveal::None,
        headers: HeaderSet::of([2, 5]),
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    assert!(plugin.is_warm(), "the load opened this thread's stream");
    assert_eq!(plugin.name(), "greeter");
    // The I/O thread counts a stream after it opens it, and the load
    // returns at the open.
    assert!(
        eventually(|| plugin.stats().connects == 1),
        "the settled stream is counted: {:?}",
        plugin.stats()
    );
    let seen = served.seen();
    assert_eq!(seen.hellos.len(), 1);
    let hello = &seen.hellos[0];
    assert_eq!(
        (hello.protocol_min, hello.protocol_max),
        (1, 2),
        "no stream stands yet: the host's range"
    );
    assert_eq!(hello.plugin, "greeter");
    assert_eq!(hello.config, b"mode=1");
    assert_eq!(hello.counters, ["hits", "skips"]);
    assert_eq!(hello.stages, ALL, "any of the three may be asked");
    assert_eq!(hello.max_in_flight, 8, "the stream's window");
    let exposure = hello.exposure.as_ref().expect("the exposure");
    assert_eq!(
        (exposure.tenant, exposure.session_key),
        (
            proto::exposure::Mode::Hash as i32,
            proto::exposure::Mode::None as i32
        )
    );
    assert_eq!(exposure.headers, [2, 5]);
}

#[test]
fn a_plugin_that_cannot_be_reached_or_whose_init_fails_does_not_load() {
    let runtime = runtime();
    // No server behind the path.
    let dir = tempfile::tempdir().expect("a directory");
    let mut nowhere = GrpcPluginSpec::new(
        "nowhere",
        Transport::Unix {
            path: dir.path().join("none.sock"),
            peer_uid: None,
        },
    );
    nowhere.connect_micros = 200_000;
    assert!(matches!(
        Plugin::load(&runtime, nowhere),
        Err(LoadError::Connect(_))
    ));

    // The ack says `init` failed: its reason fails the apply.
    let served = TestPlugin::serve();
    served.script(|s| {
        s.ack_ok = false;
        s.ack_error = "no table".to_owned();
    });
    match Plugin::load(&runtime, spec(&served, "failing")) {
        Err(LoadError::Init(why)) => assert_eq!(why, "no table"),
        other => panic!("an init failure, not {other:?}"),
    }

    // The ack names a version outside the offered range.
    let served = TestPlugin::serve();
    served.script(|s| s.ack_protocol = 3);
    match Plugin::load(&runtime, spec(&served, "newer")) {
        Err(LoadError::Connect(why)) => assert!(
            why.contains("protocol 3") && why.contains("offered 1 to 2"),
            "{why}"
        ),
        other => panic!("a refused version, not {other:?}"),
    }

    // The ack comes after the connect frame.
    let served = TestPlugin::serve();
    served.script(|s| s.hello_delay = Duration::from_millis(1_500));
    let mut late = spec(&served, "late-hello");
    late.connect_micros = 100_000;
    match Plugin::load(&runtime, late) {
        Err(LoadError::Connect(why)) => assert!(why.contains("connect frame"), "{why}"),
        other => panic!("a missed connect frame, not {other:?}"),
    }
}

/// The ack's reason is the plugin's own words, of any length: the error
/// that fails the apply carries their start.
#[test]
fn an_init_failure_carries_the_start_of_the_plugins_words() {
    let served = TestPlugin::serve();
    served.script(|s| {
        s.ack_ok = false;
        s.ack_error = "\u{e9}".repeat(20_000);
    });
    let runtime = runtime();
    match Plugin::load(&runtime, spec(&served, "wordy")) {
        Err(LoadError::Init(why)) => {
            assert!(why.starts_with("\u{e9}\u{e9}"), "{why}");
            assert!(why.len() <= 300, "{} bytes of the plugin's", why.len());
        }
        other => panic!("an init failure, not {other:?}"),
    }
}

/// The cut of the plugin's words falls at a character's start: three-byte
/// characters put the bound's last byte inside one, and the words end at
/// the character before it.
#[test]
fn an_init_failures_words_are_cut_at_a_characters_start() {
    let served = TestPlugin::serve();
    served.script(|s| {
        s.ack_ok = false;
        s.ack_error = "\u{20ac}".repeat(4_000);
    });
    let runtime = runtime();
    match Plugin::load(&runtime, spec(&served, "wordy-euros")) {
        Err(LoadError::Init(why)) => {
            assert!(why.ends_with("..."), "the cut is said: {why}");
            let words = why.trim_end_matches("...");
            assert!(
                words.chars().all(|c| c == '\u{20ac}'),
                "the words end inside a character: {why}"
            );
            let euros = why.matches('\u{20ac}').count();
            assert_eq!(euros, 85, "256 bytes hold 85 whole characters of three");
        }
        other => panic!("an init failure, not {other:?}"),
    }
}

/// One plugin speaks one version: the first stream to settle fixes it, a
/// stream opened while it stands is offered that version alone, and one
/// whose ack names another is cancelled with an error that names both,
/// however many deciding threads open theirs at once. The stream that
/// settled keeps deciding.
#[test]
fn an_ack_under_another_version_is_cancelled_on_every_racing_thread_naming_both() {
    const THREADS: usize = 4;
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "one-version");
    // The refused streams count on the breaker: far from its threshold.
    spec.breaker = BreakerSpec {
        failures: 100,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // Every later ack names another version, late enough that the threads'
    // hellos are all out before the first of them is answered.
    served.script(|s| {
        s.ack_protocol = 2;
        s.hello_delay = Duration::from_millis(50);
    });
    let refused: Vec<String> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|_| scope.spawn(|| plugin.warm()))
            .collect();
        threads
            .into_iter()
            .map(|thread| match thread.join() {
                Ok(Err(LoadError::Connect(why))) => why,
                other => panic!("a cancelled stream, not {other:?}"),
            })
            .collect()
    });
    for why in &refused {
        assert!(
            why.contains("protocol 2") && why.contains("offered 1"),
            "both versions are named: {why}"
        );
    }
    {
        let seen = served.seen();
        assert_eq!(seen.hellos.len(), 1 + THREADS, "one hello a stream");
        let offered = |hello: &proto::PluginHello| (hello.protocol_min, hello.protocol_max);
        assert_eq!(offered(&seen.hellos[0]), (1, 2), "the load's hello");
        assert!(
            seen.hellos[1..]
                .iter()
                .all(|hello| offered(hello) == (1, 1)),
            "every later hello names the fixed version alone"
        );
    }
    assert!(
        eventually(|| served.seen().ended_streams as usize >= THREADS),
        "the host ended each refused stream"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.connects, stats.failed_streams),
        (1, THREADS as u64),
        "{stats:?}"
    );
    // The stream that settled first still decides: row 0 is the plugin's.
    served.behave(Behaviour::KeepFirst);
    let engine = single_engine(profile(&plugin, ALL));
    assert!(plugin.is_warm());
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(100));
}

/// Two streams offered the host's range settle one after the other: the
/// first ack fixes the plugin's version, and the stream whose ack then names
/// another is cancelled with an error that names both. Here the load's
/// stream has ended, so no stream stands as two threads' hellos go out, and
/// the plugin answers the second hello first. The events call is opened
/// again under the version that settle fixed.
#[test]
fn an_ack_that_settles_second_under_another_version_is_cancelled_naming_both() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let mut spec = spec(&served, "second-to-settle");
    // The ended and the refused stream count on the breaker: far from its
    // threshold.
    spec.breaker = BreakerSpec {
        failures: 100,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    // The plugin ends the load's stream at its first frame.
    decide(&engine, &pool(4), 1);
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the load's stream ended: no stream stands"
    );
    // The events call that names the load's version is open before the
    // version moves: its hello is the first of the two read below.
    assert!(
        eventually(|| served.seen().observe_hellos.len() == 1),
        "the events call opened under the load's version"
    );
    // The first hello's ack names 1 and waits at a gate. The second's names
    // 2 at once, and the gate opens once that stream has settled.
    let gate = Gate::default();
    served.script(|s| {
        s.behaviour = Behaviour::Honest;
        s.hello_gate = Some(gate.clone());
    });
    std::thread::scope(|scope| {
        let first = scope.spawn(|| plugin.warm());
        assert!(
            eventually(|| served.seen().hellos.len() == 2),
            "the first thread's hello is out"
        );
        served.script(|s| {
            s.ack_protocol = 2;
            s.hello_gate = None;
        });
        let second = scope.spawn(|| plugin.warm());
        assert!(
            matches!(second.join(), Ok(Ok(()))),
            "the stream answered first settles"
        );
        gate.open();
        match first.join() {
            Ok(Err(LoadError::Connect(why))) => assert!(
                why.contains("protocol 1 on this stream and 2 on the one that settled first"),
                "both versions are named: {why}"
            ),
            other => panic!("a cancelled stream, not {other:?}"),
        }
    });
    assert!(
        eventually(|| served.seen().observe_hellos.len() == 2),
        "the events call is opened again"
    );
    let seen = served.seen();
    let offered = |hello: &proto::PluginHello| (hello.protocol_min, hello.protocol_max);
    let offers: Vec<(u32, u32)> = seen.hellos.iter().map(offered).collect();
    assert_eq!(offers, [(1, 2); 3], "each went out while no stream stood");
    let named: Vec<(u32, u32)> = seen.observe_hellos.iter().map(offered).collect();
    assert_eq!(named, [(1, 1), (2, 2)], "one events call a version");
}

/// The first stream to settle fixes the version a plugin speaks: the load's
/// hello offers the host's range, and once its ack named a version every
/// later hello names that one alone, a second thread's `Decide` and the
/// `Observe` call's.
#[test]
fn the_first_settled_stream_fixes_the_version_every_later_hello_names() {
    for version in [1, 2] {
        let served = TestPlugin::serve();
        served.script(|s| s.ack_protocol = version);
        let runtime = runtime();
        let plugin = load(&runtime, &served, "fixed");
        std::thread::scope(|scope| {
            let second = scope.spawn(|| plugin.warm());
            assert!(
                matches!(second.join(), Ok(Ok(()))),
                "a second thread's stream"
            );
        });
        assert!(
            eventually(|| served.seen().observe_hellos.len() == 1),
            "the events call opens at the first settle"
        );
        let seen = served.seen();
        let offered: Vec<(u32, u32)> = seen
            .hellos
            .iter()
            .chain(&seen.observe_hellos)
            .map(|hello| (hello.protocol_min, hello.protocol_max))
            .collect();
        assert_eq!(
            offered,
            [(1, 2), (version, version), (version, version)],
            "protocol {version}"
        );
    }
}

#[test]
fn a_row_the_host_cannot_connect_to_is_refused_before_any_connect() {
    let runtime = runtime();
    let at = |path: &str| Transport::Unix {
        path: path.into(),
        peer_uid: None,
    };
    let mut no_candidates = GrpcPluginSpec::new("p", at("/nonexistent/p.sock"));
    no_candidates.max_candidates = 0;
    let mut no_connect = GrpcPluginSpec::new("p", at("/nonexistent/p.sock"));
    no_connect.connect_micros = 0;
    let no_pem = GrpcPluginSpec::new(
        "p",
        Transport::Tls {
            address: "127.0.0.1:1".to_owned(),
            tls: Tls {
                server_name: "plugin.test".to_owned(),
                ca_pem: Vec::new(),
                cert_pem: b"x".to_vec(),
                key_pem: b"x".to_vec(),
            },
        },
    );
    for (row, why) in [
        (GrpcPluginSpec::new("", at("/nonexistent/p.sock")), "name"),
        (GrpcPluginSpec::new("p", at("")), "path"),
        (no_candidates, "max_candidates"),
        (no_connect, "connect"),
        (no_pem, "PEM"),
    ] {
        match Plugin::load(&runtime, row) {
            Err(LoadError::Spec(text)) => assert!(text.contains(why), "{text}"),
            other => panic!("a refused row ({why}), not {other:?}"),
        }
    }
}

#[test]
fn the_peer_of_a_unix_socket_must_run_as_the_expected_user() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    // The server is this process: the socket's owner is its user.
    let own = std::fs::metadata(&served.socket).expect("the socket").uid();
    let mut other = spec(&served, "other-user");
    other.transport = Transport::Unix {
        path: served.socket.clone(),
        peer_uid: Some(own.wrapping_add(1)),
    };
    match Plugin::load(&runtime, other) {
        Err(LoadError::Connect(why)) => assert!(why.contains("uid"), "{why}"),
        other => panic!("a refused peer, not {other:?}"),
    }
    let mut named = spec(&served, "named-user");
    named.transport = Transport::Unix {
        path: served.socket.clone(),
        peer_uid: Some(own),
    };
    Plugin::load(&runtime, named).expect("the named user is the peer");
}

// --- what each binding asks ---------------------------------------------------

#[test]
fn one_frame_answers_the_three_stages() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "whole");
    let engine = single_engine(profile(&plugin, ALL));
    let mut candidates = pool(4);
    candidates[3].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(102),
        "the least loaded of the rows the plugin's filter kept"
    );
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 1, "one frame for the decision");
    let frame = &seen.frames[0];
    assert_eq!(frame.stages, ALL);
    assert_eq!(frame.candidates.len(), 4, "the whole view");
    assert!(
        frame.scores_in.is_empty(),
        "asked at the open: no scores yet"
    );
    let request = frame.request.as_ref().expect("the request");
    assert_eq!(
        (frame.budget_micros, request.budget_micros),
        (BUDGET_MICROS, BUDGET_MICROS),
        "the plugin's deadline is what the decision has left, never more"
    );
    assert_eq!((request.model.as_str(), request.seed), ("m", 1));
    assert_eq!(
        frame.pool.as_ref().map(|p| (p.pool_size, p.generation)),
        Some((4, 1))
    );
    drop(seen);
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.declined, stats.invalid, stats.missed),
        (1, 0, 0, 0)
    );
}

#[test]
fn a_filter_alone_narrows_the_pool() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "filter");
    let engine = single_engine(profile(&plugin, FILTER));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(100),
        "the one row the plugin kept, the most loaded: no built-in stage leaves it"
    );
    assert_eq!(served.seen().frames[0].stages, FILTER);
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_scorer_alone_is_asked_for_scores_alone() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::ScoreMostLoaded);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(100),
        "the built-in picker follows the plugin's scores to the most loaded row"
    );
    assert_eq!(served.seen().frames[0].stages, SCORE);
    assert_eq!(plugin.stats().answered, 1);
}

// The best scores of the four rows are 1/2 and 1/3, followed by 1/4.
const SCORES_WITH_BAND: Behaviour = Behaviour::ScoreBand(210);
const BAND_ROUNDS: u64 = 64;

#[test]
fn a_plugins_band_widens_the_tie_the_built_in_picker_draws_from() {
    let served = TestPlugin::serve();
    served.behave(SCORES_WITH_BAND);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    let mut led = [false; 2];
    for seed in 0..BAND_ROUNDS {
        let picked = lead(&decide(&engine, &candidates, seed));
        assert!(matches!(picked, Some(102 | 103)), "seed {seed}: {picked:?}");
        led[0] |= picked == Some(102);
        led[1] |= picked == Some(103);
    }
    assert_eq!(led, [true; 2], "each row in the band led");
    assert_eq!(plugin.stats().answered, BAND_ROUNDS);
}

#[test]
fn a_plugin_that_sends_no_band_decides_as_before() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    for seed in 0..BAND_ROUNDS {
        assert_eq!(
            lead(&decide(&engine, &candidates, seed)),
            Some(103),
            "seed {seed}"
        );
    }
    assert_eq!(plugin.stats().answered, BAND_ROUNDS);
}

#[test]
fn a_band_is_not_carried_to_the_next_decision() {
    let served = TestPlugin::serve();
    served.behave(SCORES_WITH_BAND);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    let seed = 13;
    assert_eq!(lead(&decide(&engine, &candidates, seed)), Some(102));
    served.behave(Behaviour::Honest);
    assert_eq!(lead(&decide(&engine, &candidates, seed)), Some(103));
    assert_eq!(plugin.stats().answered, 2);
}

#[test]
fn a_picker_alone_is_asked_at_the_pick_with_the_combined_scores() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "picker");
    let engine = single_engine(profile(&plugin, PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    // The scripted picker takes the surviving row with the LOWEST combined
    // score: the most loaded healthy row, which the built-in picker (the
    // highest score) would not choose.
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(101));
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 1);
    let frame = &seen.frames[0];
    assert_eq!(frame.stages, PICK);
    assert_eq!(frame.scores_in.len(), 4, "one combined score per candidate");
    assert!(
        frame.scores_in[0] < 0.0,
        "the row the filter dropped scores below zero"
    );
    assert!(
        frame.scores_in[1..].iter().all(|s| *s >= 0.0),
        "a surviving row never does: {:?}",
        frame.scores_in
    );
    drop(seen);
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_pick_of_a_row_the_frame_showed_as_dropped_is_the_plugins_fault() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickDroppedRow);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "blind-picker");
    let engine = single_engine(profile(&plugin, PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(103),
        "the fallback picker decides"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.answered),
        (1, 0),
        "the frame named its survivors: the answer is refused whole"
    );
}

#[test]
fn a_pick_the_native_filters_dropped_since_the_open_is_no_fault() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "early-picker");
    // Asked at the open over the whole view (the scorer is bound): the
    // plugin cannot know what the built-in filter drops meanwhile.
    let engine = single_engine(profile(&plugin, SCORE | PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(103),
        "the fallback picker decides over the plugin's scores"
    );
    assert_eq!(served.seen().frames[0].stages, SCORE | PICK);
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.answered, stats.breaker_opens),
        (0, 1, 0),
        "a pick within the view it was shown is within the contract"
    );
}

#[test]
fn a_plugin_is_asked_for_the_stages_it_implements_alone() {
    let served = TestPlugin::serve();
    served.script(|s| s.implements = SCORE);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scores-only");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert_eq!(
        served.seen().frames[0].stages,
        SCORE,
        "the profile binds three stages, the ack implements one"
    );
    assert_eq!(plugin.stats().answered, 1);

    let idle = TestPlugin::serve();
    idle.script(|s| s.implements = 0);
    let plugin = load(&runtime, &idle, "implements-nothing");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert_eq!(idle.frames(), 0, "nothing to ask: no frame");
}

/// A stage the profile binds and the plugin's ack leaves out takes its
/// fallback on every decision, and no counter moves for it: the host says
/// so once a bound set, which is what names the mismatch.
#[test]
fn a_stage_bound_and_not_implemented_says_so_once() {
    said();
    let served = TestPlugin::serve();
    served.script(|s| s.implements = SCORE);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "partly-implemented");
    let engine = single_engine(profile(&plugin, ALL));
    for seed in 1..=3 {
        assert_eq!(lead(&decide(&engine, &pool(4), seed)), Some(103));
    }
    let said_once = lines("partly-implemented", "does not implement");
    assert_eq!(said_once.len(), 1, "one line a bound set: {said_once:?}");
    assert!(
        said_once[0].contains(&format!("bound_stages={ALL}"))
            && said_once[0].contains(&format!("implemented_stages={SCORE}")),
        "{}",
        said_once[0]
    );

    // An ack that implements no stage loads (the plugin may observe alone)
    // and is named the same way.
    let idle = TestPlugin::serve();
    idle.script(|s| s.implements = 0);
    let plugin = load(&runtime, &idle, "observes-alone");
    let engine = single_engine(profile(&plugin, ALL));
    decide(&engine, &pool(4), 1);
    decide(&engine, &pool(4), 2);
    assert_eq!(lines("observes-alone", "does not implement").len(), 1);
}

#[test]
fn a_pair_asks_both_sides_at_once_on_one_stream() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "pair");
    let engine = Engine::new(
        vec![profile(&plugin, ALL)],
        vec![binding(Plan::PdPair, Some(0))],
        1,
        Needs::default(),
    );
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let decision = engine.decide_pair(
        &facts(9),
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("a pick");
    let lead = pick.targets.first().expect("a target");
    assert_eq!(lead.id, WorkerId(102), "the least loaded prefill");
    assert_eq!(
        lead.pair.map(|p| p.decode_id),
        Some(WorkerId(203)),
        "the least loaded decode of its class"
    );
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 2, "one frame per side");
    let phases: Vec<i32> = seen
        .frames
        .iter()
        .map(|f| f.request.as_ref().map_or(0, |r| r.phase))
        .collect();
    assert_eq!(
        phases,
        [proto::Phase::Prefill as i32, proto::Phase::Decode as i32]
    );
    assert_ne!(
        seen.frames[0].decision_id, seen.frames[1].decision_id,
        "each frame its own id on the stream"
    );
    assert_eq!(
        (
            seen.frames[0].candidates.len(),
            seen.frames[1].candidates.len()
        ),
        (3, 4)
    );
    drop(seen);
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.late_answers, stats.missed),
        (2, 0, 0),
        "both frames were in flight together and each answer found its own"
    );
}

/// One pair decision with `plugin` bound on both sides: three prefills,
/// four decodes of one pair class.
fn decide_pair(plugin: &Arc<Plugin>, facts: &RequestFacts<'_>) -> Decision {
    let engine = Engine::new(
        vec![profile(plugin, ALL)],
        vec![binding(Plan::PdPair, Some(0))],
        1,
        Needs::default(),
    );
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    engine.decide_pair(
        facts,
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// The decision is the breaker's unit in this kind too. A plugin that
/// answers outside the contract on both sides of a pair fails the decision
/// once: both frames left at the opens, the second answer is not read.
#[test]
fn an_invalid_answer_fails_its_decision_once_on_a_pair() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "pair-sloppy");
    spec.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    served.behave(Behaviour::ScoreNan);
    let decision = decide_pair(&plugin, &facts(9));
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    assert!(
        eventually(|| served.frames() == 2),
        "both frames left at the opens"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.breaker_opens),
        (1, 0),
        "one failure a decision: the other side's answer was not read"
    );
    // The next such decision is the threshold's second failure.
    decide_pair(&plugin, &facts(10));
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.breaker_opens), (2, 1));
}

/// A plugin asked for the pick alone on both sides of a pair, whose first
/// pick is a row its frame showed as dropped: the decision is failed once
/// and the other side's pick sends no frame, as an open after the failure
/// sends none.
#[test]
fn a_pick_only_frame_is_not_sent_once_the_plugin_failed_its_decision() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickDroppedRow);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "pair-blind-picker");
    let engine = Engine::new(
        vec![profile(&plugin, PICK)],
        vec![binding(Plan::PdPair, Some(0))],
        1,
        Needs::default(),
    );
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    // A row the built-in filter drops on each side: each frame names it.
    prefills[0].health.healthy = false;
    decodes[0].health.healthy = false;
    let decision = engine.decide_pair(
        &facts(9),
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (1, 0), "{stats:?}");
    // A second frame, were it sent, is answered into nothing.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        (served.frames(), plugin.stats().late_answers),
        (1, 0),
        "the failed decision's other side asked nothing"
    );
}

/// One decision over both kinds: a WebAssembly plugin filters and a remote
/// plugin scores on both sides of a pair, their runtimes built over the same
/// open decisions. The decision is seeded once and has one remainder: each
/// remote frame carries what the WebAssembly plugin's constructors left of
/// the budget by its open, less than the budget and less again on the second
/// side. And the decision is one unit to each plugin: the WebAssembly filter
/// traps and the remote scores are not numbers, one failure each, however
/// many sides opened them.
#[cfg(feature = "wasm")]
#[test]
fn a_pair_decision_over_both_kinds_is_one_decision_to_every_plugin() {
    use smg_policy_controller::plugins::wasm;
    use support::wasm_guest::{mode, record_guest};

    let open = OpenDecisions::new();
    let guests = Arc::new(wasm::Runtime::new(&open).expect("a runtime"));
    let runtime = Runtime::new(&open).expect("a runtime");
    let mut row = wasm::PluginSpec::new("trapping", record_guest());
    row.config = vec![mode::TRAP_IN_FILTER];
    row.budget_micros = 10_000;
    let guest = guest_loaded(&guests, &row);
    let served = TestPlugin::serve();
    let remote = load(&runtime, &served, "sloppy");
    served.behave(Behaviour::ScoreNan);
    let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
    let least: Option<Box<dyn Scorer>> = Some(Box::new(LeastRequests));
    let engine = Engine::new(
        vec![Profile {
            name: "both-kinds".into(),
            filters: vec![Box::new(wasm::WasmFilter::new(guest.bind(), healthy))],
            scorers: vec![WeightedScorer {
                scorer: Box::new(GrpcScorer::new(remote.bind(), least)),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        }],
        vec![binding(Plan::PdPair, Some(0))],
        1,
        Needs::default(),
    );
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let request = facts(9);
    let decision = engine.decide_pair(
        &request,
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    assert!(
        eventually(|| served.frames() == 2),
        "both remote frames left at the opens"
    );
    let (g, r) = (guest.stats(), remote.stats());
    assert_eq!(
        (g.traps, r.invalid),
        (1, 1),
        "one failure a decision for each plugin: {g:?} {r:?}"
    );
    let left: Vec<u32> = served
        .seen()
        .frames
        .iter()
        .map(|frame| frame.request.as_ref().map_or(0, |r| r.budget_micros))
        .collect();
    assert!(
        left[1] < left[0] && left[0] < request.budget_micros,
        "one remainder: each frame carries what the WebAssembly constructors left of {}: {left:?}",
        request.budget_micros
    );
}

/// A frame budget gone on arrival is one budget hit a decision, on a pair
/// as on a single placement: the second side falls back without a count.
#[test]
fn a_gone_budget_fails_its_decision_once_on_a_pair() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "pair-budgetless");
    spec.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let gone = RequestFacts {
        budget_micros: 0,
        ..facts(9)
    };
    let decision = decide_pair(&plugin, &gone);
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    let stats = plugin.stats();
    assert_eq!(
        (stats.budget_gone, stats.breaker_opens),
        (1, 0),
        "one budget hit a decision"
    );
    decide_pair(&plugin, &gone);
    let stats = plugin.stats();
    assert_eq!((stats.budget_gone, stats.breaker_opens), (2, 1));
    assert_eq!(served.frames(), 0, "no frame left the host");
}

/// An opening is otherwise silent (the stages take their fallback): in
/// this kind too it writes one line, with the plugin's own name, the rule
/// and the cool-down.
#[test]
fn an_opening_writes_one_line_under_the_plugins_name() {
    said();
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "grpc-opener");
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let gone = RequestFacts {
        budget_micros: 0,
        ..facts(1)
    };
    let candidates = pool(4);
    decide_with(&engine, &gone, &candidates);
    assert!(
        openings("grpc-opener").is_empty(),
        "one failure under a threshold of two opens nothing"
    );
    decide_with(&engine, &gone, &candidates);
    decide_with(&engine, &gone, &candidates);
    assert_eq!(plugin.stats().bypassed, 1);
    let lines = openings("grpc-opener");
    assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
    assert!(
        lines[0].contains("rule=\"failures\"") && lines[0].contains("cooldown_micros=60000000"),
        "{}",
        lines[0]
    );
}

#[test]
fn every_decision_of_a_thread_rides_its_one_stream() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "steady");
    let engine = single_engine(profile(&plugin, ALL));
    for seed in 1..=5 {
        assert_eq!(lead(&decide(&engine, &pool(4), seed)), Some(103));
    }
    let stats = plugin.stats();
    assert_eq!((stats.connects, stats.answered), (1, 5));
    let seen = served.seen();
    assert_eq!(seen.hellos.len(), 1);
    let ids: Vec<u64> = seen.frames.iter().map(|f| f.decision_id).collect();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]) && ids[0] != 0,
        "the thread numbers its frames, never zero: {ids:?}"
    );
}

// --- declines and invalid answers ------------------------------------------

#[test]
fn a_declined_stage_takes_its_fallback_and_the_others_answer() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Decline {
        stages: FILTER,
        reason: proto::decline::Reason::NotApplicable as i32,
    });
    let runtime = runtime();
    let plugin = load(&runtime, &served, "no-filter");
    let engine = single_engine(profile(&plugin, ALL));
    let mut candidates = pool(4);
    candidates[3].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(102),
        "the fallback filter drops the unhealthy row, the plugin's pick stands"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.declined, stats.busy, stats.answered, stats.invalid),
        (1, 0, 1, 0)
    );
}

#[test]
fn a_budget_decline_is_busy_and_a_frame_that_answers_nothing_declines_every_stage() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Decline {
        stages: ALL,
        reason: proto::decline::Reason::Budget as i32,
    });
    let runtime = runtime();
    let plugin = load(&runtime, &served, "busy");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    let stats = plugin.stats();
    assert_eq!(
        (stats.busy, stats.declined, stats.answered),
        (1, 3, 0),
        "the plugin's own budget: every stage falls back at once"
    );

    served.behave(Behaviour::Unanswered);
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    let stats = plugin.stats();
    assert_eq!(
        (stats.busy, stats.declined, stats.answered),
        (1, 6, 0),
        "an asked stage left unanswered is declined"
    );
    assert_eq!(
        (stats.invalid, stats.breaker_opens),
        (0, 0),
        "a decline is no fault"
    );
}

#[test]
fn each_invalid_answer_is_one_fault_and_the_frame_falls_back_whole() {
    let behaviours = [
        Behaviour::SurvivorOutOfRange,
        Behaviour::SurvivorTwice,
        Behaviour::ScoreCount,
        Behaviour::ScoreNan,
        Behaviour::PickOutsideOwnSurvivors,
        Behaviour::DeclinedAndAnswered,
        Behaviour::BandNan,
        Behaviour::BandAboveOne,
        Behaviour::BandBelowZero,
        Behaviour::BandWithoutScores,
        Behaviour::BandWithScoreDecline,
    ];
    let failures = behaviours.len();
    let served = TestPlugin::serve();
    served.script(|s| s.counters = vec![(0, 1)]);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), || 1_234).expect("a runtime");
    let mut spec = spec(&served, "sloppy");
    spec.breaker = BreakerSpec {
        failures: failures as u32,
        ..BreakerSpec::default()
    };
    spec.metrics = vec!["answers".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for (i, behaviour) in behaviours.into_iter().enumerate() {
        served.behave(behaviour);
        assert_eq!(
            lead(&decide(&engine, &pool(4), i as u64 + 1)),
            Some(103),
            "{behaviour:?}: the fallbacks decide"
        );
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.invalid,
                stats.answered,
                stats.declined,
                stats.breaker_opens
            ),
            (i as u64 + 1, 0, 0, u64::from(i + 1 == failures)),
            "{behaviour:?}"
        );
    }
    assert!(plugin.counters().read().all(|(_, value)| value == 0));
}

#[test]
fn the_counters_an_answer_carries_are_added_by_declared_id() {
    let served = TestPlugin::serve();
    // Id 7 is not declared: the host ignores it. An answer holds one
    // increment a declared counter at most, so three are declared.
    served.script(|s| s.counters = vec![(0, 2), (1, 5), (7, 1)]);
    let runtime = runtime();
    let mut spec = spec(&served, "counting");
    spec.metrics = vec!["hits".to_owned(), "skips".to_owned(), "spare".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    decide(&engine, &pool(4), 1);
    decide(&engine, &pool(4), 2);
    let read: Vec<(String, u64)> = plugin
        .counters()
        .read()
        .map(|(name, value)| (name.to_owned(), value))
        .collect();
    assert_eq!(
        read,
        [
            ("hits".to_owned(), 4),
            ("skips".to_owned(), 10),
            ("spare".to_owned(), 0)
        ],
        "two answers"
    );
}

/// An answer holds no more increments than the plugin declared counters:
/// a longer list is outside the contract, one fault, and none of it is
/// added.
#[test]
fn more_increments_than_declared_counters_is_one_fault() {
    let served = TestPlugin::serve();
    served.script(|s| s.counters = vec![(0, 2), (1, 5), (0, 1)]);
    let runtime = runtime();
    let mut spec = spec(&served, "over-counting");
    spec.metrics = vec!["first".to_owned(), "second".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (1, 0), "{stats:?}");
    assert!(
        plugin.counters().read().all(|(_, value)| value == 0),
        "nothing of the list is added"
    );
}

/// An answer is bounded in bytes by what its row lets it hold: a message
/// past that ends the stream before it is decoded.
#[test]
fn an_answer_past_the_message_bound_ends_its_stream() {
    let served = TestPlugin::serve();
    // Hundreds of kilobytes of increments under a row of 128 candidates.
    served.script(|s| s.counters = vec![(1, 1); 50_000]);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "flooding");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the stream ends at the message: {:?}",
        plugin.stats()
    );
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (0, 0), "{stats:?}");
}

// --- the budget and the breaker ----------------------------------------------

/// A deadline missed under the plugin's own bound, on an I/O thread seen
/// running inside a flight longer than its promise, is the plugin's
/// failure: one of its decision, counted on the breaker like a fault.
#[test]
fn a_missed_deadline_under_the_plugins_own_bound_is_one_failure_of_its_decision() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut spec = spec(&served, "silent");
    // One failure opens the breaker: the miss is one.
    spec.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let started = Instant::now();
    assert_eq!(
        lead(&decide_short(&engine, &pool(4), 1)),
        Some(103),
        "the fallbacks decide"
    );
    assert!(
        started.elapsed() >= Duration::from_micros(u64::from(SHORT_MICROS)),
        "the first stage waited out the frame's deadline"
    );
    assert!(eventually(|| served.frames() == 1));
    assert_eq!(
        served.seen().frames[0].budget_micros,
        SHORT_MICROS,
        "the deadline the frame named"
    );
    // A loaded host can hold the I/O thread under any one frame, and a held
    // miss is the host's: the next frame that leaves is judged.
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    decide_until_judged(&engine, &plugin, &short, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.answered
        ),
        (1, 1, 0, 0),
        "{stats:?}"
    );
    assert_eq!(
        stats.budget_spent, stats.stalls,
        "a miss of its own is not the decision's budget spent: {stats:?}"
    );
    assert_eq!(
        (stats.invalid, stats.failed_streams, stats.breaker_opens),
        (0, 0, 1),
        "the miss is the plugin's failure: {stats:?}"
    );
}

/// A plugin that states no budget takes the default frame, under a request
/// that states none too: a stage is shown the widest budget there, the
/// frame names the default as the plugin's deadline, and a plugin that
/// answers nothing is waited for that long and no longer. The fallbacks
/// decide, and the missed frame is the plugin's own failure.
#[test]
fn a_plugin_with_no_budget_is_waited_for_the_default_frame_under_a_request_with_none() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut unset = spec(&served, "unbudgeted");
    unset.budget_micros = 0;
    let plugin = Plugin::load(&runtime, unset).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    // What a stage is shown for a request that states no budget.
    let widest = RequestFacts {
        budget_micros: u32::MAX,
        ..facts(1)
    };
    let default = FrameBudget::DEFAULT_MICROS;
    let started = Instant::now();
    assert_eq!(
        lead(&decide_with(&engine, &widest, &pool(4))),
        Some(103),
        "the fallbacks decide"
    );
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_micros(u64::from(default)),
        "the first stage waited out the default frame: {waited:?}"
    );
    assert!(
        waited < Duration::from_micros(u64::from(BUDGET_MICROS)),
        "and nothing like a request's budget: {waited:?}"
    );
    assert!(eventually(|| served.frames() == 1));
    let seen = served.seen();
    let frame = &seen.frames[0];
    assert_eq!(frame.budget_micros, default, "the deadline the frame named");
    assert_eq!(
        frame.request.as_ref().map(|request| request.budget_micros),
        Some(u32::MAX),
        "the request's own budget, as a stage is shown it"
    );
    // A loaded host can hold the I/O thread under any one frame, and a held
    // miss is the host's: the next frame that leaves is judged.
    decide_until_judged(&engine, &plugin, &widest, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.answered
        ),
        (1, 1, 0, 0),
        "the missed default frame is the plugin's own: {stats:?}"
    );
    assert_eq!(
        stats.budget_spent, stats.stalls,
        "no budget of the request's was spent: {stats:?}"
    );
}

#[test]
fn a_plugin_that_misses_every_deadline_opens_its_breaker_through_the_latency_sample() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "always-late");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let quick = RequestFacts {
        budget_micros: 2_000,
        ..facts(1)
    };
    // One sample per frame that left, each the whole budget: the window of
    // 64 reads slow at its first evaluation.
    decide_until_judged(&engine, &plugin, &quick, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "{stats:?}"
    );
    // A budget no longer than the I/O thread's promise: no miss of it is
    // the plugin's failure, the ring alone judged these.
    assert_eq!(
        (stats.own_misses, stats.unread),
        (0, stats.missed - stats.stalls),
        "{stats:?}"
    );
    assert!(matches!(
        decide_with(&engine, &quick, &candidates),
        Decision::Pick(_)
    ));
    let after = plugin.stats();
    assert_eq!(
        (after.missed, after.bypassed),
        (stats.missed, 1),
        "inside the cool-down the plugin is not asked"
    );
}

/// The same plugin under a request that states no budget, with a budget of
/// its own that it misses every time: a stage is shown the widest budget
/// there, and the plugin's time is set against its own frames, not against
/// that budget. So the ring still judges it, and opens the breaker at its
/// first window.
#[test]
fn a_plugin_that_misses_every_deadline_of_its_own_is_judged_under_a_request_with_no_budget() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut tight = spec(&served, "always-late-of-its-own");
    // A budget no longer than the I/O thread's promise: no miss of it is
    // the plugin's failure, so the ring alone can turn the plugin off.
    tight.budget_micros = 2_000;
    let plugin = Plugin::load(&runtime, tight).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    // What a stage is shown for a request that states no budget.
    let widest = RequestFacts {
        budget_micros: u32::MAX,
        ..facts(1)
    };
    decide_until_judged(&engine, &plugin, &widest, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "{stats:?}"
    );
    assert_eq!(
        (stats.own_misses, stats.unread),
        (0, stats.missed - stats.stalls),
        "the ring alone judged these: {stats:?}"
    );
    assert!(matches!(
        decide_with(&engine, &widest, &candidates),
        Decision::Pick(_)
    ));
    let after = plugin.stats();
    assert_eq!(
        (after.missed, after.bypassed),
        (stats.missed, 1),
        "inside the cool-down the plugin is not asked"
    );
}

#[test]
fn a_plugin_that_answers_past_half_the_budget_opens_its_breaker_through_the_latency_sample() {
    let served = TestPlugin::serve();
    // Six tenths of the budget before each answer: a loaded host only adds
    // to it, and a frame that misses samples the whole budget.
    served.script(|s| s.delay = Duration::from_millis(6));
    let runtime = runtime();
    let mut spec = spec(&served, "slow-answers");
    // A loaded host turns some of these answers into misses of the plugin's
    // own, and five of those would open the breaker by the failure rule
    // first. That rule is set out of reach: the latency sample alone opens.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let tight = RequestFacts {
        budget_micros: 10_000,
        ..facts(1)
    };
    // One sample per frame that left, each a round trip past half the
    // budget: the window of 64 reads slow at its first evaluation.
    decide_until_judged(&engine, &plugin, &tight, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.answered + stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "an answer in time is still judged by its round trip: {stats:?}"
    );
}

/// The I/O thread is the host's part of a round trip. Held from before a
/// frame leaves until past its deadline, an honest plugin misses every
/// frame and none of the misses is its own: each frame falls back, each is
/// counted as a stall, the ring takes no sample and the breaker stays shut.
/// Without the witness every miss samples its whole budget, and the window
/// reads slow at its first read.
#[test]
fn frames_missed_across_a_hold_of_the_io_thread_are_the_hosts_and_open_nothing() {
    const HOLD: Duration = Duration::from_millis(25);
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "held-io");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let tight = RequestFacts {
        budget_micros: 10_000,
        ..facts(1)
    };
    let mut rounds = 0_u64;
    let more = |rounds: u64| {
        let stats = plugin.stats();
        stats.stalls < 64 && stats.breaker_opens == 0 && rounds < 96
    };
    while more(rounds) {
        rounds += 1;
        runtime.hold_io(HOLD);
        assert!(matches!(
            decide_with(&engine, &tight, &candidates),
            Decision::Pick(_)
        ));
        // Released, the thread hands the frame over and the plugin answers:
        // late, or in time when this thread itself was held off the send.
        // Either way nothing queues behind the next hold.
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.late_answers + stats.answered == rounds
            }),
            "round {rounds}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.stalls, stats.breaker_opens, stats.bypassed),
        (64, 0, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        stats.missed + stats.answered,
        rounds,
        "every frame left and was judged: {stats:?}"
    );
}

/// An answer that arrives in time across a stall of the I/O thread is read
/// and used, and its round trip is no sample: the time is the host's. Each
/// answer here waits out a hold of more than half the budget: without the
/// witness the window reads slow by its median at its first read.
#[test]
fn answers_that_cross_a_hold_of_the_io_thread_are_used_and_are_no_sample() {
    const HOLD: Duration = Duration::from_millis(20);
    let served = TestPlugin::serve();
    // Row 0, which no built-in stage would leave: the answer is seen used.
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "held-answers");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let snug = RequestFacts {
        budget_micros: 36_000,
        ..facts(1)
    };
    let mut used = 0_u64;
    for round in 1..=64_u64 {
        runtime.hold_io(HOLD);
        let decision = decide_with(&engine, &snug, &candidates);
        // A fallback's pick is a frame a loaded host made miss after the
        // hold: a stall all the same.
        used += u64::from(lead(&decision) == Some(100));
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.answered + stats.late_answers == round
            }),
            "round {round}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.breaker_opens, stats.bypassed),
        (used, 0, 0),
        "every answer in time is used and none is a slow sample: {stats:?}"
    );
    assert_eq!(stats.answered + stats.missed, 64, "{stats:?}");
    assert!(
        used >= 48 && stats.stalls >= 60,
        "the held round trips are stalls, but for a frame this thread sent late: {stats:?}"
    );
}

/// A slow answer on an I/O thread that is seen running is the plugin's
/// time, never a stall: the thread ticks while the frame is in flight.
#[test]
fn a_slow_answer_on_a_thread_seen_running_is_the_plugins_time() {
    let served = TestPlugin::serve();
    // Three times the I/O thread's promise before each answer.
    served.script(|s| s.delay = Duration::from_millis(12));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "slow-not-stalled");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    // A loaded host can hold the I/O thread under any one frame: one round
    // trip seen whole is the proof.
    let mut rounds = 0;
    loop {
        rounds += 1;
        assert!(matches!(decide(&engine, &candidates, 1), Decision::Pick(_)));
        let stats = plugin.stats();
        if stats.answered > stats.stalls {
            break;
        }
        assert!(rounds < 50, "every slow answer read as a stall: {stats:?}");
    }
}

/// A missed frame samples its grant, the budget it was given and used up,
/// never the wait the deciding thread measured: the grant is the whole
/// budget and not past it, so two misses in a window of fast answers do not
/// open by the tail. Sampled at the wait, each would read past the budget by
/// the thread's own lateness at its wake, and the second-slowest of the
/// window would be one of them.
#[test]
fn two_missed_frames_in_a_window_of_fast_answers_do_not_open_by_the_tail() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "two-misses");
    // The window's tail alone is read here. An honest answer a loaded host
    // delivers late is one more miss, and five misses in the window open
    // the breaker by the failure rule before the window is read.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    served.behave(Behaviour::Silent);
    decide_until_judged(&engine, &plugin, &short, &candidates, 2);
    served.behave(Behaviour::Honest);
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &plugin, &short, &candidates, 64);
    // Both made misses are the plugin's own and in its window. An answer
    // the I/O thread stalled under is a stall too, so the misses less the
    // stalls say nothing here.
    let stats = plugin.stats();
    assert_eq!(
        (stats.own_misses >= 2, stats.breaker_opens),
        (true, 0),
        "{stats:?}"
    );
}

/// A filter of the host's own that holds its decision's thread once, for
/// the milliseconds it is told: a plugin's frame sent at the open is in
/// flight meanwhile.
#[derive(Debug)]
struct HoldsTheThread(Arc<AtomicU64>);

impl Filter for HoldsTheThread {
    fn name(&self) -> &'static str {
        "holds-the-thread"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        let hold = self.0.swap(0, Ordering::Relaxed);
        if hold != 0 {
            std::thread::sleep(Duration::from_millis(hold));
        }
        true
    }
}

/// An answer that arrives after its frame's deadline, while the decision's
/// thread is elsewhere, is used when the thread comes to read it, and it is
/// sampled at its whole flight. A miss is a frame the thread stopped
/// waiting for; the lateness is the plugin's and the latency rule reads it.
/// So two such answers in a window of fast ones open the breaker by the
/// tail, where two missed frames, sampled at their grant, do not.
#[test]
fn late_answers_the_thread_finds_without_waiting_are_used_and_sampled_at_their_flight() {
    said();
    let served = TestPlugin::serve();
    // Row 0, which no built-in stage would leave: the answer is seen used.
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let mut spec = spec(&served, "late-answers");
    // The latency rule alone opens here.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let hold = Arc::new(AtomicU64::new(0));
    // The host's filter runs ahead of the plugin's: the frame left at the
    // open, and the thread reads its answer after the hold.
    let mut held = profile(&plugin, ALL);
    held.filters
        .insert(0, Box::new(HoldsTheThread(Arc::clone(&hold))));
    let engine = single_engine(held);
    let candidates = pool(4);
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    // The answer leaves three budgets after its frame, and the thread is
    // held ten.
    served.script(|s| s.delay = Duration::from_millis(60));
    let mut late = 0;
    for _ in 0..16 {
        if late == 2 {
            break;
        }
        let before = plugin.stats();
        hold.store(200, Ordering::Relaxed);
        let decision = decide_with(&engine, &short, &candidates);
        let after = plugin.stats();
        if after.stalls != before.stalls || after.missed != before.missed {
            made_again(
                "late answers",
                "the host held the answer past the thread's hold, or held its I/O thread",
            );
            continue;
        }
        assert_eq!(lead(&decision), Some(100), "the late answer is used");
        assert_eq!(after.answered - before.answered, 1, "{after:?}");
        late += 1;
    }
    assert_eq!(late, 2, "two late answers were read: {:?}", plugin.stats());
    served.script(|s| s.delay = Duration::ZERO);
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &plugin, &short, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        stats.breaker_opens, 1,
        "two flights past the budget are the window's tail: {stats:?}"
    );
    let opened = openings("late-answers");
    assert!(
        opened.len() == 1 && opened[0].contains("rule=\"latency\""),
        "{opened:?}"
    );
}

/// A frame the decision's remainder clamped at its send is not under the
/// plugin's own bound: its miss is the decision's budget spent, whoever
/// spent it, and no failure.
#[test]
fn a_miss_under_the_remainders_clamp_is_the_decisions_budget_spent_and_no_failure() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let mute = TestPlugin::serve();
    mute.behave(Behaviour::Silent);
    let runtime = runtime();
    // The silent filter's own budget is a quarter of the decision's, and
    // its own misses never open it here.
    let mut quarter = spec(&silent, "silent-filter");
    quarter.budget_micros = SHORT_MICROS / 4;
    quarter.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let filter = Plugin::load(&runtime, quarter).expect("the plugin loads");
    // One failure would open the picker's breaker: a clamped miss is none.
    let mut clamped = spec(&mute, "clamped-picker");
    clamped.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let picker = Plugin::load(&runtime, clamped).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    // The picker is asked at the pick under the rest, three quarters of the
    // budget: a flight far longer than the I/O thread's promise.
    decide_until_judged(&engine, &picker, &short, &pool(4), 1);
    let stats = picker.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.breaker_opens
        ),
        (1, 0, 0, 0),
        "{stats:?}"
    );
    assert_eq!(stats.budget_spent, stats.missed, "{stats:?}");
}

/// A profile whose filters are these plugins in order, each with its
/// built-in fallback: every frame leaves at the decision's open.
fn filters_of(plugins: &[&Arc<Plugin>]) -> Profile {
    Profile {
        name: "filters".into(),
        filters: plugins
            .iter()
            .map(|plugin| {
                let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
                Box::new(GrpcFilter::new(plugin.bind(), healthy)) as Box<dyn Filter>
            })
            .collect(),
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    }
}

/// A frame sent with its whole budget is still the remainder's where the
/// remainder ended its wait first. Three silent plugins in front draw the
/// whole budget while their frames fly together, so the last plugin's
/// wait ends at about a third of the frame it was sent with: its miss is
/// the decision's budget spent, no failure, and its sample is what it was
/// given, about a third of the budget, which no window reads slow. Judged
/// by the frame as sent it would fail every decision; sampled at the frame
/// as sent its window would read slow at its first read.
#[test]
fn a_wait_the_remainder_ended_first_is_the_decisions_spend_at_what_the_frame_was_given() {
    let runtime = runtime();
    let fronts: Vec<TestPlugin> = (0..3).map(|_| TestPlugin::serve()).collect();
    let front: Vec<Arc<Plugin>> = fronts
        .iter()
        .enumerate()
        .map(|(i, served)| {
            served.behave(Behaviour::Silent);
            // Seven of the decision's twenty milliseconds each, their own
            // misses never opening them here. Each sample of theirs is a
            // whole frame, so their windows read slow: under a cool-down of
            // zero that opening bypasses nothing.
            let mut spec = spec(served, &format!("front-{i}"));
            spec.budget_micros = 7_000;
            spec.breaker = BreakerSpec {
                failures: u32::MAX,
                cooldown_micros: 0,
                ..BreakerSpec::default()
            };
            Plugin::load(&runtime, spec).expect("the plugin loads")
        })
        .collect();
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let last = load(&runtime, &served, "starved");
    let engine = single_engine(filters_of(&[&front[0], &front[1], &front[2], &last]));
    let candidates = pool(4);
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    // The window's first read is at its 64th sample. A round whose frame
    // the stream did not take (the I/O thread held, its window full) is no
    // sample and costs microseconds, so the loop is bounded on the clock:
    // a count of rounds runs out before a held thread is seen again.
    let started = Instant::now();
    let mut rounds = 0_u32;
    loop {
        let stats = last.stats();
        if stats.missed - stats.stalls >= 64 || stats.breaker_opens != 0 {
            break;
        }
        rounds += 1;
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the frames do not leave: {stats:?}"
        );
        assert!(matches!(
            decide_with(&engine, &short, &candidates),
            Decision::Pick(_)
        ));
    }
    let stats = last.stats();
    assert_eq!(
        (stats.unread, stats.breaker_opens, stats.bypassed),
        (0, 0, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        stats.budget_spent + stats.own_misses,
        stats.missed,
        "{stats:?}"
    );
    assert!(
        stats.own_misses <= 2,
        "a wait the remainder ended is not the plugin's own: {stats:?}"
    );
}

/// A frame sent under what a decision had left is read against its plugin's
/// whole frame. A silent filter draws three quarters of the budget, so the
/// picker's frame leaves with the last quarter as its deadline and misses
/// it: the decision's budget spent, whoever spent it, and no failure. Its
/// sample is a quarter of the plugin's own frame, which no window reads
/// slow. Read against the frame as sent, every sample would be a whole
/// frame and the window would open at its first read.
#[test]
fn a_frame_sent_under_the_rest_is_read_against_its_plugins_whole_frame() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let mute = TestPlugin::serve();
    mute.behave(Behaviour::Silent);
    let runtime = runtime();
    // The silent filter's own budget is three quarters of the decision's.
    // Nothing bypasses it here: its misses never count to an opening, and
    // an opening by its own slow samples lasts no time.
    let mut front = spec(&silent, "silent-front");
    front.budget_micros = SHORT_MICROS / 4 * 3;
    front.breaker = BreakerSpec {
        failures: u32::MAX,
        cooldown_micros: 0,
        ..BreakerSpec::default()
    };
    let filter = Plugin::load(&runtime, front).expect("the plugin loads");
    // The picker's own frame is the whole budget. A round whose filter frame
    // did not leave asks it under that, and such a miss is its own: its
    // failures never open it here.
    let mut rest = spec(&mute, "under-the-rest");
    rest.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let picker = Plugin::load(&runtime, rest).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    let short = RequestFacts {
        budget_micros: SHORT_MICROS,
        ..facts(1)
    };
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &picker, &short, &pool(4), 64);
    let stats = picker.stats();
    assert_eq!(
        (stats.unread, stats.breaker_opens, stats.bypassed),
        (0, 0, 0),
        "{stats:?}"
    );
    assert_eq!(
        stats.budget_spent + stats.own_misses,
        stats.missed,
        "{stats:?}"
    );
    assert!(
        stats.own_misses <= 2,
        "a frame the remainder clamped is not the plugin's own: {stats:?}"
    );
}

/// The second arm of the plugin's own bound: a frame the remainder clamped
/// is still the plugin's own where the plugin alone spent the budget. One
/// plugin is the filter and the picker of a decision and answers after six
/// tenths of the budget: the filter's frame is answered, the picker's frame
/// leaves under the four tenths left, a flight longer than the I/O thread's
/// promise, and misses. Nothing but the plugin drew the budget: the miss is
/// its failure.
#[test]
fn a_miss_under_a_remainder_the_plugin_alone_spent_is_its_own() {
    let served = TestPlugin::serve();
    served.script(|s| s.delay = Duration::from_millis(12));
    let runtime = runtime();
    let mut spec = spec(&served, "spends-alone");
    // Its failures never open it here: a loaded host takes more than one
    // decision to show one whole.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&plugin, &plugin));
    let candidates = pool(4);
    // A loaded host makes another round of it, each told by a count of the
    // plugin's own: the first frame's answer past the budget (the plugin
    // answers at six tenths of it), the I/O thread stalled, or a remainder
    // too short for the witness. Such a round shows nothing of the rule: it
    // says so and is not one of the twenty.
    let (mut rounds, mut held) = (0_u32, 0_u32);
    loop {
        rounds += 1;
        let before = plugin.stats();
        assert!(matches!(
            decide_short(&engine, &candidates, 1),
            Decision::Pick(_)
        ));
        let after = plugin.stats();
        let seen = (
            after.answered - before.answered,
            after.own_misses - before.own_misses,
            after.budget_spent - before.budget_spent,
        );
        if seen == (1, 1, 0) {
            break;
        }
        if after.answered == before.answered
            || after.stalls != before.stalls
            || after.unread != before.unread
        {
            held += 1;
            made_again(
                "a miss under a remainder the plugin alone spent",
                &format!("the host held round {rounds}, which read {seen:?}"),
            );
        }
        assert!(
            rounds - held < 20 && rounds < 400,
            "no decision answered its first frame and missed its second as its own in {rounds} rounds, {held} of them held: {after:?}"
        );
        // The answer of the missed frame is late: nothing queues behind.
        std::thread::sleep(Duration::from_millis(15));
    }
}

/// A frame whose flight is no longer than the I/O thread's promise ends
/// before that thread can be seen late: a hold shorter than the promise
/// across its deadline leaves no gap for the witness. An honest plugin
/// under such a budget, the I/O thread held across each deadline for less
/// than the promise, misses every frame, and nothing says whose the miss
/// is: each is the decision's budget spent, counted unread, and no miss is
/// the plugin's failure. Judged as its own, the fifth would open the
/// breaker. The latency rule alone judges such frames: a window of them
/// used up whole reads slow at its first read, whoever was late.
#[test]
fn a_miss_too_short_for_the_witness_is_unread_and_the_decisions_budget_spent() {
    // Longer than the plugin's budget, shorter than the promise.
    const HOLD: Duration = Duration::from_millis(3);
    let served = TestPlugin::serve();
    // The plugin's usual time: inside its budget below, and over half of it.
    // So a round answered in time is a slow sample too, like a miss.
    served.script(|s| s.delay = Duration::from_micros(1_200));
    let runtime = runtime();
    // The plugin's own budget: a fifth of the decision's, half the promise.
    let mut spec = spec(&served, "short-frames");
    spec.budget_micros = 2_000;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let tight = RequestFacts {
        budget_micros: 10_000,
        ..facts(1)
    };
    // Up to the window's first read, at its 64th sample: an unread miss, or
    // a round answered in time. A round the I/O thread was seen late in is
    // charged nothing, answered or missed, so it is no sample.
    let mut rounds = 0_u64;
    let more = |rounds: u64| {
        let stats = plugin.stats();
        stats.answered + stats.missed - stats.stalls < 64
            && stats.breaker_opens == 0
            && rounds < 192
    };
    while more(rounds) {
        rounds += 1;
        runtime.hold_io(HOLD);
        assert!(matches!(
            decide_with(&engine, &tight, &candidates),
            Decision::Pick(_)
        ));
        // Released, the thread hands the frame over and the plugin answers:
        // late, or in time when this thread itself was held off the send.
        // Either way nothing queues behind the next hold.
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.late_answers + stats.answered == rounds
            }),
            "round {rounds}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    // The window's sixty-four samples: every one an unread miss, but a
    // round answered in time, where this thread was held off its send. Each
    // is past half of the plugin's frame, so the window's first read opens.
    assert_eq!(
        (
            stats.answered + stats.missed - stats.stalls,
            stats.own_misses,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 0, 1, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        (stats.missed + stats.answered, stats.budget_spent),
        (rounds, stats.missed),
        "every frame left, and every miss is the decision's budget spent: {stats:?}"
    );
}

#[test]
fn an_answered_wait_draws_the_time_the_thread_blocked() {
    let slow = TestPlugin::serve();
    // Far inside the answered cases' budget: the answer always arrives.
    slow.script(|s| s.delay = Duration::from_millis(200));
    let quick = TestPlugin::serve();
    let runtime = runtime();
    let filter = load(&runtime, &slow, "slow-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    // The scripted picker, asked alone, takes the most loaded row.
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(100));
    assert_eq!((filter.stats().answered, picker.stats().answered), (1, 1));
    // The picker's frame left after the filter's wait and names what the
    // decision had left. Half the 200 ms the thread blocked must be gone
    // from it; the other half is what a stalled thread may lose between
    // the send and the wait.
    let named = quick.seen().frames[0].budget_micros;
    assert!(
        named <= BUDGET_MICROS - 100_000,
        "the wait drew nothing like its 200 ms: {named} of {BUDGET_MICROS} left"
    );
}

/// A stream that goes while its decision waits ends the wait, and the time
/// the thread blocked is drawn like an answered wait's: a plugin asked
/// later in the decision is asked under the rest.
#[test]
fn a_wait_its_stream_broke_draws_the_time_the_thread_blocked() {
    let leaving = TestPlugin::serve();
    // The stream ends 200 ms after the frame, far inside the budget.
    leaving.script(|s| {
        s.behaviour = Behaviour::EndStream;
        s.delay = Duration::from_millis(200);
    });
    let quick = TestPlugin::serve();
    let runtime = runtime();
    let filter = load(&runtime, &leaving, "leaving-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    // The filter falls back; the scripted picker, asked alone, takes the
    // most loaded row.
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(100));
    assert!(
        eventually(|| filter.stats().failed_streams == 1),
        "the stream went under the wait"
    );
    assert_eq!((filter.stats().missed, picker.stats().answered), (0, 1));
    // As for an answered wait: half the 200 ms must be gone from what the
    // picker's frame names.
    let named = quick.seen().frames[0].budget_micros;
    assert!(
        named <= BUDGET_MICROS - 100_000,
        "the broken wait drew nothing like its 200 ms: {named} of {BUDGET_MICROS} left"
    );
}

#[test]
fn a_missed_frame_draws_its_grant_and_the_next_plugin_is_asked_under_the_rest() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let quick = TestPlugin::serve();
    let runtime = runtime();
    // The silent filter's own budget is a quarter of the decision's.
    let mut quarter = spec(&silent, "silent-filter");
    quarter.budget_micros = SHORT_MICROS / 4;
    let filter = Plugin::load(&runtime, quarter).expect("the plugin loads");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    assert!(matches!(
        decide_short(&engine, &pool(4), 1),
        Decision::Pick(_)
    ));
    let stats = filter.stats();
    assert_eq!(
        (stats.missed, stats.own_misses + stats.budget_spent),
        (1, 1),
        "its own miss, or the host's on a stalled thread: {stats:?}"
    );
    // The miss drew the frame's grant, not a clock: the picker's frame
    // names exactly the rest, whatever the host's load.
    assert!(eventually(|| quick.frames() == 1));
    assert_eq!(
        quick.seen().frames[0].budget_micros,
        SHORT_MICROS - SHORT_MICROS / 4
    );
    // The silent filter's frame named two budgets: its own quarter as the
    // frame's deadline, the decision's remainder as the request's. Its
    // server records the frame on a thread of its own: the read waits.
    assert!(eventually(|| silent.frames() == 1));
    let sent = silent.seen();
    assert_eq!(
        (
            sent.frames[0].budget_micros,
            sent.frames[0].request.as_ref().map(|r| r.budget_micros)
        ),
        (SHORT_MICROS / 4, Some(SHORT_MICROS))
    );
}

#[test]
fn a_pick_only_ask_after_the_budget_is_spent_sends_no_frame() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let quick = TestPlugin::serve();
    let runtime = runtime();
    // The silent filter's grant is the whole budget: its miss leaves none.
    let filter = load(&runtime, &silent, "silent-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    assert_eq!(
        lead(&decide_short(&engine, &pool(4), 1)),
        Some(103),
        "the fallbacks decide"
    );
    assert_eq!(filter.stats().missed, 1);
    let stats = picker.stats();
    assert_eq!(
        (stats.budget_spent, stats.missed, stats.answered),
        (1, 0, 0),
        "the picker is not asked under a spent budget"
    );
    assert_eq!(stats.breaker_opens, 0, "and it is not its fault");
    assert_eq!(quick.frames(), 0);
}

#[test]
fn an_answer_after_its_deadline_is_late() {
    let served = TestPlugin::serve();
    // Far past the short budget: the wait cannot end with the answer.
    served.script(|s| s.delay = Duration::from_millis(1_500));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "slow");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide_short(&engine, &pool(4), 1)), Some(103));
    assert_eq!(plugin.stats().missed, 1);
    assert!(
        eventually(|| plugin.stats().late_answers == 1),
        "the answer arrived for a decision nobody waits for"
    );
    assert_eq!(plugin.stats().answered, 0);
}

#[test]
fn a_frame_its_decision_never_read_is_forgotten_at_the_close() {
    let served = TestPlugin::serve();
    // The answer leaves the plugin long after the decision closed.
    served.script(|s| s.delay = Duration::from_millis(1_000));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "unread");
    // The scorer is bound, so the frame leaves at the open; the built-in
    // filter then empties the pool and no stage of the plugin's runs.
    let engine = single_engine(profile(&plugin, SCORE | PICK));
    let mut candidates = pool(3);
    for c in &mut candidates {
        c.health.healthy = false;
    }
    assert!(matches!(decide(&engine, &candidates, 1), Decision::Shed(_)));
    assert!(
        eventually(|| plugin.stats().late_answers == 1),
        "the answer found no frame waiting"
    );
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.missed), (0, 0));
    // The place it held is free: the next decision is answered.
    served.script(|s| s.delay = Duration::ZERO);
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_frame_past_the_window_is_refused_without_a_fault() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "nine");
    // One failure would open the breaker: a refused frame is none.
    spec.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // Nine bound sets of one build score in one profile: each opens its own
    // frame at the decision's open, on the deciding thread's one stream,
    // before any of them is read.
    let binder = plugin.binder();
    let scorers = (0..9)
        .map(|_| {
            let fallback: Option<Box<dyn Scorer>> = Some(Box::new(LeastRequests));
            WeightedScorer {
                scorer: Box::new(GrpcScorer::new(binder.bind(), fallback)) as Box<dyn Scorer>,
                weight: 1.0,
            }
        })
        .collect();
    let engine = single_engine(Profile {
        name: "nine".into(),
        filters: vec![Box::new(Healthy)],
        scorers,
        picker: Box::new(Best),
    });
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(103),
        "eight answers and one fallback agree"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.refused),
        (8, 1),
        "the ninth frame found the window full"
    );
    assert_eq!(
        (stats.breaker_opens, stats.bypassed, stats.missed),
        (0, 0, 0),
        "a full window is not the plugin's fault"
    );
    assert_eq!(served.frames(), 8, "the refused frame never left");
    // Every place is free again at the close: the next decision's first
    // eight frames are answered.
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.refused), (16, 2));
}

#[test]
fn a_gone_budget_sends_no_frame_and_counts_on_the_breaker() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "budgetless");
    spec.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let gone = RequestFacts {
        budget_micros: 0,
        ..facts(1)
    };
    let candidates = pool(4);
    assert!(matches!(
        decide_with(&engine, &gone, &candidates),
        Decision::Pick(_)
    ));
    let stats = plugin.stats();
    assert_eq!((stats.budget_gone, stats.breaker_opens), (1, 0));
    decide_with(&engine, &gone, &candidates);
    let stats = plugin.stats();
    assert_eq!(
        (stats.budget_gone, stats.breaker_opens),
        (2, 1),
        "a budget hit each, to the threshold"
    );
    assert_eq!(lead(&decide(&engine, &candidates, 3)), Some(103));
    assert_eq!(plugin.stats().bypassed, 1);
    assert_eq!(served.frames(), 0, "no frame left the host");
}

#[test]
fn a_pool_past_max_candidates_sends_no_frame() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "small-view");
    spec.max_candidates = 3;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(103),
        "whole view or none: the fallbacks decide"
    );
    assert_eq!(plugin.stats().too_many, 1);
    assert_eq!(served.frames(), 0);
    assert_eq!(lead(&decide(&engine, &pool(3), 2)), Some(102));
    assert_eq!((plugin.stats().too_many, plugin.stats().answered), (1, 1));
}

/// The plugin's own open marks its phase open and counts the decision
/// before it reads the runtime's clock, and the engine does not close a
/// scope whose open did not return. So an open that stops half-way undoes
/// itself: here the runtime's clock panics inside it, and the thread's next
/// decision is a decision of its own that asks the plugin. Left half-open,
/// the phase would read as open to the next decision, and it would not ask
/// the plugin.
#[test]
fn a_plugins_open_that_stops_half_way_undoes_itself() {
    let served = TestPlugin::serve();
    let clock_panics = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&clock_panics);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        assert!(
            !told.swap(false, Ordering::Relaxed),
            "the runtime's clock panics, once"
        );
        1_000_000
    })
    .expect("a runtime");
    let plugin = load(&runtime, &served, "half-open");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(3);
    // A whole decision first: the stream is settled, and what stops the next
    // open is the clock alone.
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(102));
    let before = plugin.stats();
    clock_panics.store(true, Ordering::Relaxed);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 2)
    }));
    assert!(caught.is_err(), "the plugin's open panicked at the clock");
    assert_eq!(
        plugin.stats().answered,
        before.answered,
        "the open stopped before its frame"
    );
    let next = decide(&engine, &candidates, 3);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        1,
        "the next decision asked the plugin: {asked:?}"
    );
    assert_eq!(lead(&next), Some(102), "the plugin's pick leads");
}

/// What the case below shares between its picker and the runtime's clock.
#[derive(Debug, Default)]
struct PastThePick {
    /// The case wants a close's read of the clock to panic.
    armed: AtomicBool,
    /// The decision's pick ran: its thread's next read of the clock is a
    /// close's.
    picked: AtomicBool,
}

/// The built-in best picker, telling the clock when it ran.
#[derive(Debug)]
struct TellsThePick(Arc<PastThePick>);

impl Lifecycle for TellsThePick {}

impl Picker for TellsThePick {
    fn name(&self) -> &'static str {
        "tells-the-pick"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        if self.0.armed.load(Ordering::Relaxed) {
            self.0.picked.store(true, Ordering::Relaxed);
        }
        Best.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The plugin's own close takes its decision's latency sample, and a window
/// that reads slow asks the runtime's clock: the host's code, inside the
/// close. So a close that stops half-way leaves its scope closed: here the
/// clock panics inside it, and the thread's next decision is a decision of
/// its own, asked under its own budget. Left open, the phase would read as
/// open to the next decision, and it would not ask the plugin. With its
/// place in the thread's decision not given back, the next decision would
/// run on what the stopped one had left.
#[test]
fn a_plugins_close_that_stops_half_way_leaves_its_scope_closed() {
    let served = TestPlugin::serve();
    let shared = Arc::new(PastThePick::default());
    let told = Arc::clone(&shared);
    let deciding = std::thread::current().id();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        // Past an armed pick the deciding thread reads the clock in a close
        // alone: that read panics, once.
        assert!(
            std::thread::current().id() != deciding || !told.picked.swap(false, Ordering::Relaxed),
            "the runtime's clock panics inside a close"
        );
        1_000_000
    })
    .expect("a runtime");
    let mut spec = spec(&served, "half-closed");
    // Every answer below is slow or a miss. Five misses of the plugin's own
    // would open the breaker by the failure rule first, so that rule is set
    // out of reach: the window's read is what asks the clock.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // The plugin is bound at one stage, so a decision closes its scope
    // once. Bound at two, the engine closes the same bound set twice, and
    // the second close forgets what the first left open.
    let mut bound = profile(&plugin, FILTER);
    bound.picker = Box::new(TellsThePick(Arc::clone(&shared)));
    let engine = single_engine(bound);
    // A whole decision first: the stream is settled, and its lead is what
    // the plugin makes of this pool.
    let want = lead(&decide(&engine, &pool(3), 1));
    // Six tenths of the budget before each answer, and a miss samples the
    // whole budget: the window reads slow at its first read, which is at
    // the close of the decision that fills it.
    served.script(|s| s.delay = Duration::from_millis(6));
    let tight = RequestFacts {
        budget_micros: 10_000,
        ..facts(2)
    };
    let candidates = pool(4);
    shared.armed.store(true, Ordering::Relaxed);
    let mut stopped = false;
    for _ in 0..1_024 {
        shared.picked.store(false, Ordering::Relaxed);
        let decided = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decide_with(&engine, &tight, &candidates)
        }));
        if decided.is_err() {
            stopped = true;
            break;
        }
    }
    shared.armed.store(false, Ordering::Relaxed);
    shared.picked.store(false, Ordering::Relaxed);
    assert!(
        stopped,
        "a close read the clock and stopped there: {:?}",
        plugin.stats()
    );
    // The next decision's answer takes twice the stopped decision's whole
    // budget: it is read only under a budget of the decision's own.
    served.script(|s| s.delay = Duration::from_millis(20));
    let before = plugin.stats();
    let next = decide(&engine, &pool(3), 1);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        1,
        "the next decision asked the plugin under its own budget: {asked:?}"
    );
    assert_eq!(lead(&next), want, "and picks from its own view");
}

#[test]
fn the_breaker_opens_on_a_run_of_faults_and_closes_after_the_cool_down() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::ScoreNan);
    let clock = Arc::new(AtomicU64::new(1_000));
    let reads = Arc::clone(&clock);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || reads.load(Ordering::Relaxed))
        .expect("a runtime");
    let mut spec = spec(&served, "breaking");
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 1_000_000,
        cooldown_micros: 5_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    decide(&engine, &candidates, 1);
    decide(&engine, &candidates, 2);
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.breaker_opens, stats.bypassed),
        (2, 1, 0)
    );
    assert_eq!(lead(&decide(&engine, &candidates, 3)), Some(103));
    assert_eq!(plugin.stats().bypassed, 1);
    assert!(
        eventually(|| served.frames() == 2),
        "the bypassed decision sent nothing"
    );

    // Past the cool-down on the runtime's clock the plugin is asked again.
    clock.store(6_100_000, Ordering::Relaxed);
    served.behave(Behaviour::Honest);
    assert_eq!(lead(&decide(&engine, &candidates, 4)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.bypassed), (1, 1));
}

#[test]
fn without_a_fallback_the_failure_policy_decides() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    // Open, the default: every stage steps aside.
    let plugin = load(&runtime, &served, "lenient");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    assert!(matches!(
        decide_short(&engine, &pool(4), 1),
        Decision::Pick(_)
    ));

    let mut closed = spec(&served, "strict");
    closed.failure = Failure::Closed;
    let plugin = Plugin::load(&runtime, closed).expect("the plugin loads");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    match decide_short(&engine, &pool(4), 2) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "strict");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a shed naming the plugin, not {other:?}"),
    }
}

// --- the stream ---------------------------------------------------------------

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

// --- the lifecycle and the exposure -------------------------------------------

/// A scorer whose state the lifecycle's calls keep.
#[derive(Debug)]
struct Kept;

impl Lifecycle for Kept {
    fn needs_lifecycle(&self) -> bool {
        true
    }
}

impl Scorer for Kept {
    fn name(&self) -> &'static str {
        "kept"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand::NORMALIZED
    }
}

/// A plugin whose row lists no event declares no need of the lifecycle's
/// calls: an adapter needs them where its fallback does, and an engine built
/// in code says so. A plugin whose row lists an event needs them, whatever
/// its fallback and whichever adapter forwards.
#[test]
fn an_adapter_needs_the_lifecycle_where_its_row_or_its_fallback_does() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut unlisted = spec(&served, "kept");
    unlisted.observes = EventSet::NONE;
    let plugin = Plugin::load(&runtime, unlisted).expect("the plugin loads");
    let binder = plugin.binder();
    let adapter = |fallback: Option<Box<dyn Scorer>>| GrpcScorer::new(binder.bind(), fallback);
    assert!(adapter(Some(Box::new(Kept))).needs_lifecycle());
    assert!(!adapter(Some(Box::new(LeastRequests))).needs_lifecycle());
    assert!(!adapter(None).needs_lifecycle());

    let engine_of = |fallback: Option<Box<dyn Scorer>>| {
        single_engine(Profile {
            name: "adapted".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(adapter(fallback)),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        })
    };
    assert!(engine_of(Some(Box::new(Kept))).needs_lifecycle());
    assert!(!engine_of(None).needs_lifecycle());

    let mut listed = spec(&served, "hearing");
    listed.observes = EventSet::of([EventKind::Completed]);
    let hearing = Plugin::load(&runtime, listed).expect("the plugin loads");
    let binder = hearing.binder();
    // The build's observer and the adapter that is not: each says so.
    assert!(GrpcScorer::new(binder.bind(), None).needs_lifecycle());
    assert!(GrpcPicker::new(binder.bind(), None).needs_lifecycle());
}

/// The models a worker left ride protocol 2. A plugin whose streams settled
/// on 2 hears the word once, with its names, ahead of the relabelling of the
/// same write. One that settled on 1 is not sent it and hears the
/// relabelling alone, numbered as the first event: the word takes no number
/// on a stream that does not carry it.
#[test]
fn the_models_a_worker_left_reach_a_plugin_on_protocol_2_alone() {
    use proto::{lifecycle_event::Event, membership_change::Kind};
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    for version in [2, 1] {
        let served = TestPlugin::serve();
        served.script(|s| s.ack_protocol = version);
        let runtime = runtime();
        let plugin = load(&runtime, &served, "leaver");
        let engine = single_engine(profile(&plugin, ALL));
        engine.on_membership(&Membership::left(WorkerId(7), &models));
        engine.on_membership(&Membership::Relabelled(WorkerId(7)));
        // While a stream stands on 1 the left word is not raised.
        let raised = if version == 2 { 2 } else { 1 };
        assert_eq!(plugin.stats().observed, raised, "protocol {version}");
        // The relabelling is the last event of either stream.
        let relabelled = |events: &[proto::LifecycleEvent]| {
            events.last().is_some_and(|last| {
                matches!(&last.event, Some(Event::Membership(m)) if m.kind == Kind::Relabelled as i32)
            })
        };
        assert!(
            eventually(|| relabelled(&served.seen().events)),
            "protocol {version}"
        );
        let seen = served.seen();
        let told: Vec<(u64, i32, u64, &[String])> = seen
            .events
            .iter()
            .map(|event| match &event.event {
                Some(Event::Membership(m)) => (event.seq, m.kind, m.worker_id, &m.models[..]),
                other => panic!("a membership word, not {other:?}"),
            })
            .collect();
        let left = (1, Kind::Left as i32, 7, &models[..]);
        if version == 2 {
            assert_eq!(told, [left, (2, Kind::Relabelled as i32, 7, &[][..])]);
        } else {
            assert_eq!(told, [(1, Kind::Relabelled as i32, 7, &[][..])]);
        }
    }
}

/// A stream on protocol 1 hears what it heard before the word came, at the
/// queue's bound too. While a stream stands on 1 a left word is not raised:
/// twice the queue's worth of them takes no place and counts no loss, and
/// the relabelling behind them is the stream's first event.
#[test]
fn left_words_past_the_queues_bound_cost_a_stream_on_protocol_1_nothing() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "older");
    let engine = single_engine(profile(&plugin, ALL));
    let models = ["m-chat".to_owned()];
    // The queue holds 1,024 events.
    for _ in 0..2_048 {
        engine.on_membership(&Membership::left(WorkerId(7), &models));
    }
    engine.on_membership(&Membership::Relabelled(WorkerId(7)));
    assert!(
        eventually(|| !served.seen().events.is_empty()),
        "the relabelling reaches the plugin"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.observed, stats.observe_dropped),
        (1, 0),
        "no left word took a place: {stats:?}"
    );
    let seen = served.seen();
    assert_eq!(seen.events.len(), 1, "the relabelling alone");
    assert_eq!(seen.events[0].seq, 1, "no loss is numbered");
}

/// A left word raised while no stream stands waits for the stream that
/// settles next: the events call that still stands yields at it, and the
/// word is kept with the events behind it. Where that stream settles on
/// protocol 1 the word is not written: the plugin hears the relabelling
/// behind it alone, as the first event, and the word takes no number on a
/// stream that does not carry it. Here the load's stream has ended as the
/// two words are raised.
#[test]
fn a_left_word_that_waited_for_a_stream_is_not_written_on_protocol_1() {
    use proto::{lifecycle_event::Event, membership_change::Kind};
    let served = TestPlugin::serve();
    served.behave(Behaviour::EndStream);
    let runtime = runtime();
    let mut spec = spec(&served, "waited");
    // The ended stream counts on the breaker: far from its threshold.
    spec.breaker = BreakerSpec {
        failures: 100,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    // The plugin ends the load's stream at its first frame.
    decide(&engine, &pool(4), 1);
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the load's stream ended: no stream stands"
    );
    let models = ["m-chat".to_owned()];
    engine.on_membership(&Membership::left(WorkerId(7), &models));
    engine.on_membership(&Membership::Relabelled(WorkerId(7)));
    assert_eq!(
        plugin.stats().observed,
        2,
        "no stream stands, so both words are raised"
    );
    // The next stream settles on protocol 1, the fixture's own.
    served.behave(Behaviour::Honest);
    plugin.warm().expect("a stream settles");
    assert!(
        eventually(|| !served.seen().events.is_empty()),
        "the relabelling reaches the plugin"
    );
    let seen = served.seen();
    let told: Vec<(u64, i32, u64)> = seen
        .events
        .iter()
        .map(|event| match &event.event {
            Some(Event::Membership(m)) => (event.seq, m.kind, m.worker_id),
            other => panic!("a membership word, not {other:?}"),
        })
        .collect();
    assert_eq!(
        told,
        [(1, Kind::Relabelled as i32, 7)],
        "the left word was not written, and took no number"
    );
}

#[test]
fn the_lifecycle_reaches_the_plugin_once_in_order() {
    let served = TestPlugin::serve();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), || 400_000).expect("a runtime");
    let plugin = load(&runtime, &served, "observer");
    // Three adapters of one profile: one of them forwards.
    let engine = single_engine(profile(&plugin, ALL));
    let lease = Lease::default();
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    engine.on_released(&lease, ReleaseReason::ClientGone);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(1), WorkerId(2)]));
    engine.on_signal(&Signal::gateway_veto(WorkerId(1), 500_000));
    assert_eq!(plugin.stats().observed, 4);
    assert!(eventually(|| served.seen().events.len() == 4));
    let seen = served.seen();
    assert_eq!(seen.observe_hellos.len(), 1);
    assert_eq!(seen.observe_hellos[0].plugin, "observer");
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, [1, 2, 3, 4], "numbered in the order they were raised");
    use proto::lifecycle_event::Event;
    match &seen.events[0].event {
        Some(Event::Dispatched(d)) => assert_eq!(d.since_decision_micros, 3),
        other => panic!("dispatched, not {other:?}"),
    }
    match &seen.events[1].event {
        Some(Event::Released(r)) => {
            assert_eq!(r.reason, proto::released::Reason::ClientGone as i32);
        }
        other => panic!("released, not {other:?}"),
    }
    match &seen.events[2].event {
        Some(Event::Membership(m)) => assert_eq!(m.ids, [1, 2]),
        other => panic!("membership, not {other:?}"),
    }
    match &seen.events[3].event {
        Some(Event::Signal(s)) => assert_eq!(
            s.kind,
            Some(proto::signal::Kind::GatewayVetoMicros(100_000)),
            "what the veto has left on the runtime's clock"
        ),
        other => panic!("a signal, not {other:?}"),
    }
}

/// A row lists the kinds of event its plugin hears, none by default. A
/// plugin whose row lists none is told nothing, and no `Observe` stream is
/// opened for it. One whose row lists some is sent those kinds alone,
/// numbered with no gap for the kinds it does not hear. A loaded plugin
/// says the kinds its row lists.
#[test]
fn a_plugin_hears_the_kinds_of_event_its_row_lists_and_no_other() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let tell = |engine: &Engine| {
        let lease = Lease::default();
        engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
        engine.on_released(&lease, ReleaseReason::ClientGone);
        engine.on_membership(&Membership::Snapshot(&[WorkerId(1), WorkerId(2)]));
        engine.on_signal(&Signal::gateway_veto(WorkerId(1), 500_000));
    };
    // The default row lists nothing.
    let mut unlisted = GrpcPluginSpec::new("unlisted", unix(&served));
    unlisted.connect_micros = BUDGET_MICROS;
    unlisted.budget_micros = BUDGET_MICROS;
    assert_eq!(unlisted.observes, EventSet::NONE);
    let deaf = Plugin::load(&runtime, unlisted).expect("the plugin loads");
    assert_eq!(deaf.observes(), EventSet::NONE);
    let deaf_engine = single_engine(profile(&deaf, ALL));
    assert!(!deaf_engine.needs_lifecycle());
    assert_eq!(lead(&decide(&deaf_engine, &pool(4), 1)), Some(103));
    tell(&deaf_engine);
    let stats = deaf.stats();
    assert_eq!((stats.observed, stats.observe_dropped), (0, 0), "{stats:?}");

    let mut listed = spec(&served, "listed");
    listed.observes = EventSet::of([EventKind::Released, EventKind::Membership]);
    let hearing = Plugin::load(&runtime, listed).expect("the plugin loads");
    assert_eq!(
        hearing.observes(),
        EventSet::of([EventKind::Released, EventKind::Membership])
    );
    let hearing_engine = single_engine(profile(&hearing, ALL));
    assert!(hearing_engine.needs_lifecycle());
    tell(&hearing_engine);
    assert_eq!(hearing.stats().observed, 2);
    assert!(eventually(|| served.seen().events.len() == 2));
    let seen = served.seen();
    // One stream, the hearing plugin's: the other plugin's `Decide` stream
    // settled first, answered a decision and opened none.
    let hellos: Vec<&str> = seen
        .observe_hellos
        .iter()
        .map(|hello| hello.plugin.as_str())
        .collect();
    assert_eq!(hellos, ["listed"]);
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    assert_eq!(seqs, [1, 2], "no gap for the kinds it does not hear");
    use proto::lifecycle_event::Event;
    assert!(
        matches!(seen.events[0].event, Some(Event::Released(_))),
        "{:?}",
        seen.events[0]
    );
    assert!(
        matches!(seen.events[1].event, Some(Event::Membership(_))),
        "{:?}",
        seen.events[1]
    );
}

/// A dispatch on another target of its pick is told with a lease that names
/// that target's index, and the event's index is the lease's.
#[test]
fn a_dispatch_tells_the_plugin_the_index_its_lease_names() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "second-target");
    let engine = single_engine(profile(&plugin, ALL));
    let lease = Lease {
        target_index: 1,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    assert!(eventually(|| served.seen().events.len() == 1));
    let seen = served.seen();
    match &seen.events[0].event {
        Some(proto::lifecycle_event::Event::Dispatched(d)) => assert_eq!(d.target_index, 1),
        other => panic!("dispatched, not {other:?}"),
    }
}

/// The events' numbers are the stream's: an event is numbered where it
/// enters the stream, on the I/O thread, so threads that raise events at
/// once cannot put a later number ahead of an earlier one.
#[test]
fn events_raised_on_many_threads_are_numbered_in_the_order_they_are_sent() {
    const THREADS: usize = 16;
    const EACH: usize = 60;
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "many-observers");
    let engine = single_engine(profile(&plugin, ALL));
    let start = std::sync::Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                start.wait();
                for _ in 0..EACH {
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                }
            });
        }
    });
    // Fewer than the queue holds: none is dropped.
    assert_eq!(plugin.stats().observed, (THREADS * EACH) as u64);
    assert!(eventually(|| served.seen().events.len() == THREADS * EACH));
    let seen = served.seen();
    let seqs: Vec<u64> = seen.events.iter().map(|e| e.seq).collect();
    let in_order: Vec<u64> = (1..=(THREADS * EACH) as u64).collect();
    assert_eq!(seqs, in_order, "the numbers follow the order on the stream");
}

/// An event the queue cannot take is dropped and counted, never a wait on
/// the thread that raised it. The I/O thread is held off the queue here,
/// so nothing leaves it while 2,000 events are raised.
#[test]
fn events_past_the_queue_are_dropped_and_counted() {
    for _ in 0..FLOODS {
        let served = TestPlugin::serve();
        let runtime = runtime();
        let plugin = load(&runtime, &served, "flooded-observer");
        let engine = single_engine(profile(&plugin, ALL));
        if !flooded_under_a_hold(&runtime, &engine) {
            made_again("a flood", HELD_PAST_THE_HOLD);
            continue;
        }
        let stats = plugin.stats();
        assert_eq!(stats.observed + stats.observe_dropped, 2_000, "{stats:?}");
        assert!(
            stats.observed >= 1_024 && stats.observe_dropped > 0,
            "the queue holds 1,024 and the rest is dropped: {stats:?}"
        );
        return;
    }
    panic!("the runner held every flood");
}

/// The floods a case makes at most: one whose test thread a held runner
/// paused past the hold of the I/O thread is made again.
const FLOODS: u32 = 5;

/// How long a flood holds the I/O thread off the queue.
const FLOOD_HOLD: Duration = Duration::from_millis(300);

/// What a flood that is made again says of itself.
const HELD_PAST_THE_HOLD: &str = "its thread was held past the hold of the I/O thread";

/// Raises 2,000 events while the I/O thread is held off the queue, and says
/// whether the hold outlasted them. A test thread paused past the hold lets
/// the queue drain under the events, and what is raised after it carries the
/// count of refusals: that flood asserts nothing.
fn flooded_under_a_hold(runtime: &Arc<Runtime>, engine: &Engine) -> bool {
    let asked = Instant::now();
    runtime.hold_io(FLOOD_HOLD);
    for _ in 0..2_000 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    asked.elapsed() < FLOOD_HOLD
}

/// A refused event leaves its mark on the numbers, where the loss fell. An
/// event's number counts the events sent before it and the events the queue
/// had refused when it entered the queue. Three events go first and arrive;
/// then the I/O thread is held off the queue while it fills and refuses.
/// The events the queue took were raised before the loss: they arrive
/// gapless. The next event raised skips by what was lost, so a plugin that
/// keeps a ledger of leases reads the gap behind everything raised before
/// it.
#[test]
fn a_refused_event_leaves_a_gap_in_the_numbers() {
    for _ in 0..FLOODS {
        let served = TestPlugin::serve();
        let runtime = runtime();
        let plugin = load(&runtime, &served, "ledger-keeper");
        let engine = single_engine(profile(&plugin, ALL));
        let dispatch =
            || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        for _ in 0..3 {
            dispatch();
        }
        assert!(eventually(|| served.seen().events.len() == 3));
        if !flooded_under_a_hold(&runtime, &engine) {
            made_again("a refused event", HELD_PAST_THE_HOLD);
            continue;
        }
        let stats = plugin.stats();
        let (sent, refused) = (stats.observed, stats.observe_dropped);
        assert!(refused > 0, "the queue refused some: {stats:?}");
        assert!(eventually(|| served.seen().events.len() as u64 == sent));
        let seqs: Vec<u64> = served.seen().events.iter().map(|e| e.seq).collect();
        assert!(
            seqs.windows(2).all(|pair| pair[0] + 1 == pair[1]),
            "an event raised before the loss arrived behind a gap"
        );
        assert_eq!(
            seqs.last().copied(),
            Some(sent),
            "the events raised before the loss read gapless"
        );
        dispatch();
        assert!(eventually(|| served.seen().events.len() as u64 == sent + 1));
        assert_eq!(
            served.seen().events.last().map(|e| e.seq),
            Some(sent + 1 + refused),
            "the next event raised skips by what the queue refused"
        );

        return;
    }
    panic!("the runner held every flood");
}

/// The numbers go on across two `Observe` streams: the first server goes, a
/// second takes its socket's path, and what the second is sent is numbered
/// above everything the first was.
#[test]
fn the_numbers_go_on_across_two_observe_streams() {
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, "two-streams");
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for _ in 0..40 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    assert!(eventually(|| first.seen().events.len() == 40));
    let last = first.seen().events.last().map(|event| event.seq);
    assert_eq!(last, Some(40));
    let path = first.socket.clone();
    first.stop();
    let second = TestPlugin::serve();
    std::fs::rename(&second.socket, &path).expect("the second server takes the first's path");
    let mut raised = 0_u32;
    while second.seen().events.len() < 20 {
        raised += 1;
        assert!(raised <= 15_000, "no event reached the second server");
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    let seen = second.seen();
    let numbers: Vec<u64> = seen.events.iter().map(|event| event.seq).collect();
    assert!(
        numbers.first().copied() > last,
        "the numbers went down across two streams: {numbers:?}"
    );
    assert!(
        numbers.windows(2).all(|pair| pair[1] > pair[0]),
        "{numbers:?}"
    );
}

/// The numbers are kept across a turn of the `Observe` task that panicked.
/// The first server goes, and the host's line of that end panics where it
/// is said, once. The task is started again a pause later, and what the
/// second server is sent is numbered above everything the first was.
#[test]
fn the_numbers_go_on_across_a_turn_that_panicked() {
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, "numbers-over-a-panic");
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for _ in 0..40 {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    assert!(eventually(|| first.seen().events.len() == 40));
    let last = first.seen().events.last().map(|event| event.seq);
    assert_eq!(last, Some(40));
    let path = first.socket.clone();
    panic_at("numbers-over-a-panic", "observe stream failed", 1);
    first.stop();
    // No event is raised while the first server goes: the call's own end is
    // then all the task sees, and its line panics where it is said. With an
    // event queued at that instant the send can see the end first, and that
    // end has no line.
    assert!(
        eventually(|| trap_spent("numbers-over-a-panic")),
        "the line of the first stream's end did not panic"
    );
    let second = TestPlugin::serve();
    std::fs::rename(&second.socket, &path).expect("the second server takes the first's path");
    let mut raised = 0_u32;
    while second.seen().events.len() < 20 {
        raised += 1;
        assert!(raised <= 15_000, "no event reached the second server");
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !lines("numbers-over-a-panic", "observe task panicked").is_empty(),
        "the host said nothing of the task that panicked"
    );
    let seen = second.seen();
    let numbers: Vec<u64> = seen.events.iter().map(|event| event.seq).collect();
    assert!(
        numbers.first().copied() > last,
        "the numbers started again with the turn: {numbers:?}"
    );
    assert!(
        numbers.windows(2).all(|pair| pair[1] > pair[0]),
        "{numbers:?}"
    );
}

/// A telling that panics is told once an outage: the outage's line is held
/// outside the task's turn. Here the host's line of a failed stream is armed
/// to panic three times, and the first server goes with no event raised: a
/// task that has sent nothing says its first panic. A second server takes
/// the path, is sent events and goes, and a third is reached: the task went
/// on, no call stood, and an event sent arms no line. The outage's line and
/// the task's were each said once.
#[test]
fn a_telling_that_panics_is_told_once_an_outage() {
    let name = "panics-in-a-row";
    let mut first = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&first, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert!(eventually(|| first.seen().observe_hellos.len() == 1));
    let path = first.socket.clone();
    panic_at(name, "observe stream failed", 3);
    first.stop();
    assert!(
        eventually(|| lines(name, "observe task panicked").len() == 1),
        "the first panic of a task that sent nothing was not said"
    );
    // A stopped server stays to the case's end: the path is in the first
    // one's directory.
    let reach = |served: &TestPlugin| {
        std::fs::rename(&served.socket, &path).expect("the next server takes the path");
        let mut raised = 0_u32;
        while served.seen().events.is_empty() {
            raised += 1;
            assert!(raised <= 15_000, "no event reached the next server");
            engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let mut second = TestPlugin::serve();
    reach(&second);
    second.stop();
    let third = TestPlugin::serve();
    reach(&third);
    assert!(!trap_spent(name), "the outage was told again");
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
    assert_eq!(
        lines(name, "observe task panicked").len(),
        1,
        "{:?}",
        lines(name, "observe task panicked")
    );
}

/// The ends of calls that did not stand are one outage, whichever side of
/// the host's task saw each: the call's own outcome, or a send that found
/// the stream gone. Here four servers go one after another while events are
/// raised, so every call carried events and a send can meet an end: an
/// event carried arms no line, and the four ends say one.
#[test]
fn ends_of_calls_that_did_not_stand_say_one_line() {
    const ENDS: usize = 4;
    said();
    let name = "ends-under-sends";
    let mut served = TestPlugin::serve();
    let runtime = runtime();
    let mut row = spec(&served, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let path = served.socket.clone();
    let mut stopped = Vec::new();
    // The server after the last end is reached too: the task saw that end.
    for server in 1..=ENDS + 1 {
        let mut raised = 0_u32;
        while served.seen().events.is_empty() {
            raised += 1;
            assert!(raised <= 15_000, "no event reached server {server}");
            engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            std::thread::sleep(Duration::from_millis(1));
        }
        if server <= ENDS {
            served.stop();
            // A stopped server stays to the case's end: the path is in the
            // first one's directory.
            stopped.push(std::mem::replace(&mut served, TestPlugin::serve()));
            std::fs::rename(&served.socket, &path).expect("the next server takes the path");
        }
    }
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
}

/// A plugin that ends every `Observe` call is one outage, with events
/// waiting or with none: a call that ends at its hello did not stand, and an
/// event the host sent into it arms no line. The server's word is set before
/// the load, so the first call ends at its hello and says the line. Events
/// are raised from then on: the next calls end at theirs, a connect frame
/// apart, and say nothing.
#[test]
fn calls_that_end_at_their_hello_say_one_line() {
    said();
    let name = "ends-at-the-hello";
    let served = TestPlugin::serve();
    served.script(|s| s.observe_ack = "enough".to_owned());
    let runtime = runtime();
    let mut row = spec(&served, name);
    row.connect_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, row).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let mut raised = 0_u32;
    while served.seen().observe_hellos.len() < 4 {
        raised += 1;
        assert!(raised <= 15_000, "no fourth call: {:?}", plugin.stats());
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        lines(name, "observe stream failed").len(),
        1,
        "{:?}",
        lines(name, "observe stream failed")
    );
}

/// The `Observe` call's answer is bounded in bytes like a decision's: an Ack
/// past the row's bound ends the call as a status about its length, and the
/// host's line carries no more of the status than its own bound.
#[test]
fn an_observe_answer_past_the_message_bound_is_refused_by_its_length() {
    said();
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "wordy-ack");
    let engine = single_engine(profile(&plugin, ALL));
    served.script(|s| s.observe_ack = "x".repeat(300_000));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert!(
        eventually(|| !lines("wordy-ack", "observe stream failed").is_empty()),
        "no line of the observe stream: {:?}",
        plugin.stats()
    );
    let line = lines("wordy-ack", "observe stream failed").remove(0);
    assert!(
        line.contains("too large") || line.contains("OutOfRange"),
        "an Ack past the bound was read: {line}"
    );
    assert!(line.len() < 600, "the line is {} bytes", line.len());
}

/// A build that binds the plugin to two profiles takes both bound sets from
/// one binder: the engine hands an event to every profile and the plugin
/// hears it once. An engine built later over the same plugin has an observer
/// of its own, beside the first engine and once it is gone.
#[test]
fn the_lifecycle_reaches_the_plugin_once_per_engine_build() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "rebound");
    let dispatch = |engine: &Engine| {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        plugin.stats().observed
    };
    let binder = plugin.binder();
    let first = Engine::new(
        vec![
            bound_profile(binder.bind(), ALL, true),
            bound_profile(binder.bind(), ALL, true),
        ],
        vec![binding(Plan::PdPair, Some(1))],
        1,
        Needs::default(),
    );
    assert_eq!(
        dispatch(&first),
        1,
        "two bound sets of one build, one observer"
    );
    let later = single_engine(profile(&plugin, ALL));
    assert_eq!(
        dispatch(&later),
        2,
        "the later engine forwards beside the first"
    );
    assert_eq!(dispatch(&first), 3, "and the first still does");
    drop(first);
    assert_eq!(dispatch(&later), 4, "and once the first is gone");
}

#[test]
fn the_exposure_withholds_what_the_row_does_not_reveal() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "narrow-view");
    spec.view = Exposure {
        tenant: Reveal::None,
        session_key: Reveal::Plain,
        headers: HeaderSet::of([2]),
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let headers = [
        Header {
            name: 2,
            value: "exposed",
        },
        Header {
            name: 9,
            value: "hidden",
        },
    ];
    let request = RequestFacts {
        tenant: Identity::plain("acme"),
        session_key: Identity::plain("s-1"),
        headers: &headers,
        ..facts(1)
    };
    let candidates = pool(4);
    assert_eq!(
        lead(&decide_with(&engine, &request, &candidates)),
        Some(103)
    );
    let seen = served.seen();
    let sent = seen.frames[0].request.as_ref().expect("the request");
    assert_eq!(sent.tenant, None, "withheld");
    assert_eq!(
        sent.session_key.as_ref().and_then(|id| id.form.clone()),
        Some(proto::identity::Form::Plain("s-1".to_owned())),
        "in the clear, as the row asks and the host holds it"
    );
    assert_eq!(
        sent.headers,
        [proto::Header {
            name: 2,
            value: "exposed".to_owned(),
        }],
        "the allow-listed header alone"
    );
}

/// A frame carries the request's block hashes to a plugin whose row
/// declares them, in either form, and none to one whose row declares
/// neither: a prompt's length sizes no frame of a plugin that reads none.
#[test]
fn a_frame_carries_block_hashes_to_a_row_that_declares_them_alone() {
    let runtime = runtime();
    let hashes: Vec<u64> = (1..=300).collect();
    let request = RequestFacts {
        block_hashes: &hashes,
        ..facts(1)
    };
    let candidates = pool(4);
    let sent = |name: &str, needs: Needs| {
        let served = TestPlugin::serve();
        let mut spec = spec(&served, name);
        spec.needs = needs;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = single_engine(profile(&plugin, ALL));
        assert_eq!(
            lead(&decide_with(&engine, &request, &candidates)),
            Some(103),
            "{name}: the plugin answers"
        );
        let seen = served.seen();
        let sent = seen.frames[0].request.as_ref().expect("the request");
        sent.block_hashes.clone()
    };
    assert!(
        sent("reads-none", Needs::default()).is_empty(),
        "a row that declares no hashes is sent none"
    );
    let tokens = Needs::NONE.with_token_hashes();
    assert_eq!(sent("reads-tokens", tokens), hashes);
    let text = Needs::NONE.with_text_hashes();
    assert_eq!(sent("reads-text", text), hashes);
}

/// A lease on the lifecycle carries the session's value, and a request
/// shows the same value in the hash form, so a plugin matches the two: a
/// row that withholds the session key withholds both.
#[test]
fn a_lease_and_its_request_show_one_session_value_or_none() {
    let keyed = RequestFacts {
        session_key: Identity::plain("s-1"),
        ..facts(1)
    };
    let lease = Lease {
        session: keyed.session(),
        ..Lease::default()
    };
    let value = lease.session.get();
    assert_ne!(value, 0, "a keyed request has a session");
    let runtime = runtime();
    // The session as the request's frame and as the lease's event show it.
    let shown_under = |name: &str, session_key: Reveal| {
        let served = TestPlugin::serve();
        let mut spec = spec(&served, name);
        spec.view = Exposure {
            session_key,
            ..Exposure::default()
        };
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = single_engine(profile(&plugin, ALL));
        let decision = decide_with(&engine, &keyed, &pool(4));
        engine.on_decision_issued(&lease, &decision);
        assert!(eventually(|| served.seen().events.len() == 1));
        let seen = served.seen();
        let request = seen.frames[0].request.as_ref().expect("the request");
        let in_the_frame = request.session_key.as_ref().and_then(|id| id.form.clone());
        let on_the_lease = match &seen.events[0].event {
            Some(proto::lifecycle_event::Event::Issued(issued)) => {
                issued.decision.as_ref().expect("the decision").leases[0].session_key
            }
            other => panic!("issued, not {other:?}"),
        };
        (in_the_frame, on_the_lease)
    };
    assert_eq!(
        shown_under("session-withheld", Reveal::None),
        (None, 0),
        "withheld from both"
    );
    assert_eq!(
        shown_under("session-hashed", Reveal::Hash),
        (Some(proto::identity::Form::Hash(value)), value),
        "one value in both, under a row that reveals the key's hash"
    );
}

/// A row printed in its debug form writes none of its PEMs: one of them
/// is the private key.
#[test]
fn a_printed_row_withholds_its_pems() {
    let pems: [&[u8]; 3] = [b"AUTHORITY", b"CERTIFICATE", b"PRIVATE"];
    let row = GrpcPluginSpec::new(
        "printed",
        Transport::Tls {
            address: "plugin.example:9000".to_owned(),
            tls: Tls {
                server_name: "plugin.example".to_owned(),
                ca_pem: pems[0].to_vec(),
                cert_pem: pems[1].to_vec(),
                key_pem: pems[2].to_vec(),
            },
        },
    );
    let printed = format!("{row:?}");
    assert!(printed.contains("plugin.example"), "{printed}");
    for pem in pems {
        // A byte vector prints as its numbers between brackets.
        let numbers = format!("{pem:?}");
        assert!(
            !printed.contains(&numbers[1..numbers.len() - 1]),
            "a PEM's bytes are printed: {printed}"
        );
    }
}

// --- mutual TLS ---------------------------------------------------------------

/// A certificate authority minted for one test.
struct Authority {
    issuer: Issuer<'static, KeyPair>,
    pem: String,
}

impl Authority {
    fn mint(name: &str) -> Self {
        let key = KeyPair::generate().expect("a key");
        let mut params = CertificateParams::new(Vec::<String>::new()).expect("parameters");
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.distinguished_name.push(DnType::CommonName, name);
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let pem = params.self_signed(&key).expect("the authority").pem();
        Self {
            issuer: Issuer::new(params, key),
            pem,
        }
    }

    /// A certificate and its key, PEM, for `names` with one purpose.
    fn issue(&self, names: &[&str], purpose: ExtendedKeyUsagePurpose) -> (String, String) {
        let key = KeyPair::generate().expect("a key");
        let names: Vec<String> = names.iter().map(|n| (*n).to_owned()).collect();
        let mut params = CertificateParams::new(names).expect("parameters");
        params.distinguished_name.push(DnType::CommonName, "leaf");
        params.extended_key_usages = vec![purpose];
        let cert = params.signed_by(&key, &self.issuer).expect("a certificate");
        (cert.pem(), key.serialize_pem())
    }
}

#[test]
fn mutual_tls_with_the_pinned_server_name() {
    let authority = Authority::mint("the plugin authority");
    let (server_cert, server_key) =
        authority.issue(&["plugin.test"], ExtendedKeyUsagePurpose::ServerAuth);
    let (client_cert, client_key) = authority.issue(&[], ExtendedKeyUsagePurpose::ClientAuth);
    let served = TestPlugin::serve_tls(&ServerTls {
        cert_pem: server_cert,
        key_pem: server_key,
        client_ca_pem: authority.pem.clone(),
    });
    let address = served
        .address
        .expect("the TLS server's address")
        .to_string();
    let tls = |server_name: &str, cert: &str, key: &str| Transport::Tls {
        address: address.clone(),
        tls: Tls {
            server_name: server_name.to_owned(),
            ca_pem: authority.pem.clone().into_bytes(),
            cert_pem: cert.as_bytes().to_vec(),
            key_pem: key.as_bytes().to_vec(),
        },
    };
    let runtime = runtime();
    let mut spec = GrpcPluginSpec::new("remote", tls("plugin.test", &client_cert, &client_key));
    spec.connect_micros = BUDGET_MICROS;
    spec.budget_micros = BUDGET_MICROS;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads over mutual TLS");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert_eq!(plugin.stats().answered, 1);

    // Another name than the certificate carries: refused at the handshake.
    let mut renamed = GrpcPluginSpec::new("renamed", tls("other.test", &client_cert, &client_key));
    renamed.connect_micros = 2_000_000;
    assert!(matches!(
        Plugin::load(&runtime, renamed),
        Err(LoadError::Connect(_))
    ));

    // A client the plugin's authority did not sign: refused by the plugin.
    let stranger = Authority::mint("another authority");
    let (cert, key) = stranger.issue(&[], ExtendedKeyUsagePurpose::ClientAuth);
    let mut unsigned = GrpcPluginSpec::new("unsigned", tls("plugin.test", &cert, &key));
    unsigned.connect_micros = 2_000_000;
    assert!(matches!(
        Plugin::load(&runtime, unsigned),
        Err(LoadError::Connect(_))
    ));
}

// --- the measurement --------------------------------------------------------

/// Reads `VmRSS` of this process, in KiB.
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// The user and system CPU time of one `/proc` stat line, in clock ticks
/// (100 per second).
fn ticks(stat: &str) -> Option<u64> {
    let after = stat.rsplit(") ").next()?;
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// This thread's CPU ticks.
fn own_cpu_ticks() -> u64 {
    std::fs::read_to_string("/proc/thread-self/stat")
        .ok()
        .and_then(|s| ticks(&s))
        .unwrap_or(0)
}

/// The CPU ticks of this process's threads named `comm`, from `/proc`.
fn cpu_ticks_of(comm: &str) -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let name = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (name.trim() == comm).then_some(task.path().join("stat"))
        })
        .filter_map(|stat| ticks(&std::fs::read_to_string(stat).ok()?))
        .sum()
}

/// The time on the CPU of this process's threads named `comm`, in
/// nanoseconds, by the scheduler's own clock.
fn cpu_nanos_of(comm: &str) -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let name = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (name.trim() == comm).then_some(task.path().join("schedstat"))
        })
        .filter_map(|stat| {
            let line = std::fs::read_to_string(stat).ok()?;
            line.split_whitespace().next()?.parse::<u64>().ok()
        })
        .sum()
}

/// A measurement for the Numbers table, not a check: run in release with
/// `--ignored measure --nocapture`. The RSS of 64 open streams beside 64
/// idle threads; the round trip of one decision, each timed alone, as the
/// three stages (one frame at the open) and as the pick alone (one frame at
/// the pick), at 8, 64 and 512 candidates; and the CPU of the deciding
/// thread and of the runtime's I/O thread over 100,000 decisions at 64
/// candidates, by their clock ticks. The plugin is served in this process
/// over a Unix socket, reads each frame and answers at once: a round trip
/// is the host's two ends, the transport and that plugin.
#[test]
#[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
#[expect(
    clippy::print_stdout,
    reason = "the measurement reports its rows on stdout under --nocapture"
)]
fn measure_the_round_trip_the_rss_per_stream_and_the_cpu_per_decision() {
    let runtime = runtime();
    // A plugin that keeps none of the frames it is sent, behind a host
    // that takes 512 candidates.
    let serve = |name: &str| {
        let served = TestPlugin::serve();
        served.script(|s| s.forget = true);
        let mut spec = spec(&served, name);
        spec.max_candidates = 512;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        (served, plugin)
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();

    // 64 threads each open their own stream, after 64 that only wait: the
    // difference is the streams', both ends of each in this process.
    let (_served, plugin) = serve("streams");
    let threads = 64u64;
    let rss_of = |warm: bool| {
        let barrier = Arc::new(std::sync::Barrier::new(threads as usize + 1));
        let before = rss_kib();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let plugin = Arc::clone(&plugin);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    if warm {
                        plugin.warm().expect("warms");
                    }
                    barrier.wait();
                    barrier.wait();
                })
            })
            .collect();
        barrier.wait();
        let grown = rss_kib().saturating_sub(before);
        barrier.wait();
        for h in handles {
            h.join().expect("joins");
        }
        grown
    };
    let (idle, open) = (rss_of(false), rss_of(true));
    println!(
        "RSS per open stream {} KiB, both ends ({threads} idle threads grew {idle} KiB, {threads} threads with a stream each {open} KiB)",
        open.saturating_sub(idle) / threads
    );

    for (binding, stages) in [("three stages", ALL), ("the pick alone", PICK)] {
        let (_served, plugin) = serve("timed");
        let engine = single_engine(profile_of(&plugin, stages, false));
        for n in [8u32, 64, 512] {
            let candidates = pool(n);
            let pool_facts = pool_facts(&candidates);
            for seed in 0..2_000 {
                engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
            }
            let rounds = 20_000u64;
            let mut nanos: Vec<u128> = (0..rounds)
                .map(|seed| {
                    let started = Instant::now();
                    engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
                    started.elapsed().as_nanos()
                })
                .collect();
            nanos.sort_unstable();
            let at = |permille: usize| {
                let rank = (nanos.len() * permille / 1_000).min(nanos.len() - 1);
                nanos[rank] as f64 / 1e3
            };
            println!(
                "{binding} at {n} candidates: round trip p50 {:.1} us, p90 {:.1} us, p99 {:.1} us, p99.9 {:.1} us ({rounds} decisions)",
                at(500),
                at(900),
                at(990),
                at(999)
            );
        }
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.missed,
                stats.invalid,
                stats.declined,
                stats.cold,
                stats.refused
            ),
            (0, 0, 0, 0, 0),
            "{binding}: every frame was answered: {stats:?}"
        );
    }

    let (_served, plugin) = serve("cpu");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    let candidates = pool(64);
    let pool_facts = pool_facts(&candidates);
    let decisions = 100_000u64;
    let io = "smg-policy-grpc";
    let (own0, io0) = (own_cpu_ticks(), cpu_ticks_of(io));
    let started = Instant::now();
    for seed in 0..decisions {
        engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
    }
    let wall = started.elapsed();
    let own_ms = (own_cpu_ticks() - own0) * 10;
    let io_ms = (cpu_ticks_of(io) - io0) * 10;
    // CPU per decision times 10,000 per second, as a share of one core.
    let share = |ms: u64| ms as f64 / decisions as f64 * 10_000.0 / 1_000.0 * 100.0;
    println!(
        "{decisions} decisions at 64 candidates: wall {:.1} us each; the deciding thread's CPU {own_ms} ms = {:.2}% of a core at 10,000 decisions/s; the I/O thread's CPU {io_ms} ms = {:.2}% of a core at 10,000 decisions/s",
        wall.as_secs_f64() * 1e6 / decisions as f64,
        share(own_ms),
        share(io_ms)
    );
    assert_eq!(
        plugin.stats().answered,
        decisions,
        "every frame was answered"
    );
    println!(
        "the witness over those {decisions} decisions: {} answered across a stall of the I/O thread",
        plugin.stats().stalls
    );

    // The witness's price: the I/O thread's time on the CPU over two seconds
    // with no frame in flight, then over one frame a plugin never answers
    // under a two-second budget: the ticker's wakes alone.
    let window = Duration::from_secs(2);
    let idle0 = cpu_nanos_of(io);
    std::thread::sleep(window);
    let idle = cpu_nanos_of(io) - idle0;
    let (silent, plugin) = serve("ticks");
    silent.behave(Behaviour::Silent);
    let engine = single_engine(profile_of(&plugin, ALL, true));
    let held = RequestFacts {
        budget_micros: 2_000_000,
        ..facts(1)
    };
    let flight0 = cpu_nanos_of(io);
    engine.decide(&held, &pool_facts, &candidates, &cx, &mut scratch);
    let flight = cpu_nanos_of(io) - flight0;
    println!(
        "the witness's price: the I/O thread's CPU over 2 s idle {:.1} us, over 2 s with one frame in flight {:.1} us ({:.3}% of a core)",
        idle as f64 / 1e3,
        flight as f64 / 1e3,
        flight as f64 / window.as_nanos() as f64 * 100.0
    );
}

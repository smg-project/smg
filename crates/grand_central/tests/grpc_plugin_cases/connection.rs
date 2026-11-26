//! What a load needs: the hello, the version a stream settles, the rows the
//! host refuses before it connects, and both transports.

use std::{os::unix::fs::MetadataExt, time::Duration};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use smg_grand_central::{
    plugins::{
        grpc::{GrpcPluginSpec, LoadError, Plugin, Tls, Transport},
        shared::{BreakerSpec, Exposure, HeaderSet, Reveal},
    },
    service::proto,
};

use crate::{
    decide, lead, load, pool, profile, runtime, single_engine, spec,
    support::grpc_plugin::{eventually, Behaviour, Gate, ServerTls, TestPlugin, ALL},
    BUDGET_MICROS,
};

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

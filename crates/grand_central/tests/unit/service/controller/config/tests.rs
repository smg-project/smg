use std::{
    net::{Ipv4Addr, Ipv6Addr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

use super::{dialable, Config, ConfigError, Listener, OverloadThresholds, Scope, Tls};
use crate::service::{
    controller::election::{ElectionConfig, LeasePlace},
    elector::ElectionSettings,
};

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, port))
}

fn listener(name: &str, port: u16, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(port),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// The binary's default shape on ephemeral ports.
pub(crate) fn valid() -> Config {
    Config {
        controller_id: "controller-0".to_owned(),
        listeners: vec![
            listener("gateway", 0, &[Scope::Gateway]),
            listener("control", 0, &[Scope::Control]),
            listener("replication", 0, &[Scope::Replication]),
        ],
        admin: loopback(0),
        shutdown_settle: Duration::from_millis(100),
        shutdown_grace: Duration::from_secs(2),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        worker_overload: Config::DEFAULT_WORKER_OVERLOAD,
        decide_shards: 2,
        election: ElectionConfig::default(),
        discovery: None,
    }
}

/// A setting names a listener by its name, so no two share one.
#[test]
fn two_listeners_of_one_name_are_refused() {
    let mut config = valid();
    if let Some(second) = config.listeners.get_mut(1) {
        second.name = "gateway".to_owned();
    }
    assert_eq!(
        config.validate(),
        Err(ConfigError::ListenerNameShared("gateway".to_owned()))
    );
}

/// A default overload threshold is held to its bounds, by name. A
/// worker at or above a threshold is vetoed: a count of zero would veto
/// a worker with nothing queued, and so would a count over the most a
/// load report states, which is read as a negative one. A usage of zero
/// would veto a worker with nothing in use. A usage is a fraction, at
/// most one, and one that is no number compares with none.
/// The ends are taken: a count of 1 and the most a report states, and a
/// usage of 1 and the least above 0. A signal that states none has no
/// default and is no refusal, one signal or both.
#[test]
fn a_default_overload_threshold_is_one_a_worker_can_stay_under() {
    let stating = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    // The most queued requests a load report states, and the least
    // usage above zero.
    let most = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
    let least = f64::from_bits(1);
    let mut config = valid();
    assert_eq!(config.worker_overload, stating(Some(8), Some(0.8)));
    for taken in [
        stating(Some(1), Some(1.0)),
        stating(Some(most), Some(least)),
        stating(None, Some(0.5)),
        stating(Some(3), None),
        stating(None, None),
    ] {
        config.worker_overload = taken;
        assert_eq!(config.validate(), Ok(()), "{taken:?}");
    }
    for count in [Some(0), most.checked_add(1)].into_iter().flatten() {
        for usage in [None, Some(0.5)] {
            config.worker_overload = stating(Some(count), usage);
            assert_eq!(
                config.validate(),
                Err(ConfigError::OverloadWaitingRequests(count)),
                "{count} beside {usage:?}"
            );
        }
    }
    for usage in [0.0, -0.5, 1.0 + f64::EPSILON, 1.5, f64::INFINITY, f64::NAN] {
        for waiting in [None, Some(1)] {
            config.worker_overload = stating(waiting, Some(usage));
            assert_eq!(
                config.validate(),
                Err(ConfigError::OverloadTokenUsageOutOfRange),
                "{usage} beside {waiting:?}"
            );
        }
    }
    assert_eq!(
            ConfigError::OverloadWaitingRequests(0).to_string(),
            "a default overload threshold of 0 queued requests: the count is from 1 to 9223372036854775807, the most a load report states"
        );
    assert_eq!(
            ConfigError::OverloadTokenUsageOutOfRange.to_string(),
            "the default overload threshold of usage is not above 0 and at most 1: a worker at or above it is vetoed"
        );
}

#[test]
fn the_deciding_threads_are_counted_from_one_to_their_ceiling() {
    let mut config = valid();
    for count in [1, Config::MAX_DECIDE_SHARDS] {
        config.decide_shards = count;
        assert_eq!(config.validate(), Ok(()), "{count}");
    }
    for count in [0, Config::MAX_DECIDE_SHARDS + 1] {
        config.decide_shards = count;
        assert_eq!(
            config.validate(),
            Err(ConfigError::DecideShards(count)),
            "{count}"
        );
    }
    assert_eq!(
        ConfigError::DecideShards(0).to_string(),
        "0 deciding threads: the count is from 1 to 256"
    );
    assert_eq!(Config::MAX_DECIDE_SHARDS, 256, "the words' ceiling");
    assert!((1..=8).contains(&Config::default_decide_shards()));
}

#[test]
fn the_default_shape_is_accepted_with_or_without_tls() {
    let mut config = valid();
    assert_eq!(config.validate(), Ok(()));
    config.listeners[0].tls = Some(Tls {
        cert: PathBuf::from("server.pem"),
        key: PathBuf::from("server.key"),
        client_ca: Some(PathBuf::from("gateways-ca.pem")),
    });
    assert_eq!(config.validate(), Ok(()));
}

#[test]
fn the_controller_id_is_one_plain_token() {
    let mut config = valid();
    config.controller_id = String::new();
    assert_eq!(config.validate(), Err(ConfigError::EmptyControllerId));
    for bad in ["a b", "a@b", "a/b", "a:b"] {
        config.controller_id = bad.to_owned();
        assert_eq!(
            config.validate(),
            Err(ConfigError::MalformedControllerId(bad.to_owned())),
            "{bad}"
        );
    }
    for good in ["controller-0", "pod.ns_1", "A-Z.0-9"] {
        config.controller_id = good.to_owned();
        assert_eq!(config.validate(), Ok(()), "{good}");
    }
}

#[test]
fn every_scope_is_mounted_exactly_once() {
    let mut config = valid();
    config.listeners.clear();
    assert_eq!(config.validate(), Err(ConfigError::NoListeners));

    let mut config = valid();
    config.listeners[1].scopes.clear();
    assert_eq!(
        config.validate(),
        Err(ConfigError::ListenerWithoutScope("control".to_owned()))
    );

    let mut config = valid();
    config.listeners[2].scopes.push(Scope::Control);
    assert_eq!(
        config.validate(),
        Err(ConfigError::ScopeMountedTwice {
            scope: Scope::Control,
            first: "control".to_owned(),
            second: "replication".to_owned(),
        })
    );

    let mut config = valid();
    config.listeners.pop();
    assert_eq!(
        config.validate(),
        Err(ConfigError::ScopeUnmounted(Scope::Replication))
    );

    let mut config = valid();
    config.listeners.pop();
    config.listeners[1].scopes.push(Scope::Replication);
    assert_eq!(config.validate(), Ok(()), "two scopes share a listener");
}

#[test]
fn two_listeners_never_share_a_real_address() {
    let mut config = valid();
    config.listeners[0].addr = loopback(30100);
    config.listeners[2].addr = loopback(30100);
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: loopback(30100),
            first: "gateway".to_owned(),
            second: "replication".to_owned(),
        })
    );

    let mut config = valid();
    config.listeners[1].addr = loopback(30103);
    config.admin = loopback(30103);
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: loopback(30103),
            first: "control".to_owned(),
            second: "admin".to_owned(),
        })
    );

    let config = valid();
    assert!(
        config.listeners.iter().all(|l| l.addr.port() == 0),
        "ephemeral ports are not shared addresses"
    );
    assert_eq!(config.validate(), Ok(()));
}

#[test]
fn a_wildcard_and_a_specific_listener_never_share_a_real_port() {
    let wildcard = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 30100));
    let specific = loopback(30100);

    let mut config = valid();
    config.listeners[0].addr = wildcard;
    config.listeners[1].addr = specific;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: specific,
            first: "gateway".to_owned(),
            second: "control".to_owned(),
        })
    );

    let mut config = valid();
    config.listeners[0].addr = specific;
    config.listeners[1].addr = wildcard;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: wildcard,
            first: "gateway".to_owned(),
            second: "control".to_owned(),
        })
    );

    let mut config = valid();
    config.listeners[1].addr = wildcard;
    config.admin = specific;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: specific,
            first: "control".to_owned(),
            second: "admin".to_owned(),
        })
    );

    let mut config = valid();
    config.listeners[1].addr = specific;
    config.admin = wildcard;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: wildcard,
            first: "control".to_owned(),
            second: "admin".to_owned(),
        })
    );

    let v6_wildcard = SocketAddr::from((Ipv6Addr::UNSPECIFIED, 30100));
    let v6_specific = SocketAddr::from((Ipv6Addr::LOCALHOST, 30100));
    let mut config = valid();
    config.listeners[0].addr = v6_wildcard;
    config.listeners[1].addr = v6_specific;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: v6_specific,
            first: "gateway".to_owned(),
            second: "control".to_owned(),
        })
    );
}

#[test]
fn wildcard_listeners_overlap_every_address_they_cover() {
    let specific = loopback(30100);

    let mapped_wildcard = SocketAddr::from((Ipv4Addr::UNSPECIFIED.to_ipv6_mapped(), 30100));
    let mut config = valid();
    config.listeners[0].addr = mapped_wildcard;
    config.listeners[1].addr = specific;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: specific,
            first: "gateway".to_owned(),
            second: "control".to_owned(),
        })
    );

    let mapped_specific = SocketAddr::from((Ipv4Addr::LOCALHOST.to_ipv6_mapped(), 30100));
    let mut config = valid();
    config.listeners[0].addr = mapped_wildcard;
    config.listeners[1].addr = mapped_specific;
    assert_eq!(
        config.validate(),
        Err(ConfigError::AddressShared {
            addr: mapped_specific,
            first: "gateway".to_owned(),
            second: "control".to_owned(),
        })
    );
}

#[test]
fn listener_addresses_that_do_not_overlap_are_accepted() {
    let mut config = valid();
    config.listeners[0].addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
    config.listeners[1].addr = SocketAddr::from((Ipv4Addr::new(10, 0, 0, 1), 30100));
    config.listeners[2].addr = loopback(30100);
    config.admin = SocketAddr::from((Ipv6Addr::LOCALHOST, 30100));
    assert_eq!(
        config.validate(),
        Ok(()),
        "port 0, two specific addresses and two families share no address"
    );

    let mut config = valid();
    config.listeners[1].addr = SocketAddr::from((Ipv6Addr::UNSPECIFIED, 30100));
    config.listeners[2].addr = loopback(30100);
    assert_eq!(
        config.validate(),
        Ok(()),
        "an IPv6 wildcard is not known to cover the IPv4 family"
    );
}

#[test]
fn the_shutdown_bounds_hold_each_other() {
    let mut config = valid();
    config.shutdown_grace = Duration::ZERO;
    assert_eq!(config.validate(), Err(ConfigError::ZeroShutdownGrace));

    let mut config = valid();
    config.shutdown_grace = Duration::from_secs(8);
    config.shutdown_settle = Duration::from_secs(5);
    assert_eq!(
        config.validate(),
        Err(ConfigError::SettleOverHalfTheGrace {
            settle: Duration::from_secs(5),
            grace: Duration::from_secs(8),
        })
    );
    config.shutdown_settle = Duration::from_secs(4);
    assert_eq!(config.validate(), Ok(()), "half the grace is the bound");
}

#[test]
fn the_elections_settings_keep_their_order() {
    let mut config = valid();
    assert_eq!(config.election.settings, ElectionSettings::default());
    config.election.settings.renew_period = Duration::ZERO;
    assert_eq!(
        config.validate(),
        Err(ConfigError::ElectionSettings("the renew period is zero"))
    );
    let mut config = valid();
    config.election.settings.renew_deadline = Duration::from_secs(1);
    assert_eq!(
        config.validate(),
        Err(ConfigError::ElectionSettings(
            "the renew deadline is not longer than the renew period"
        ))
    );
    let mut config = valid();
    config.election.settings.duration = Duration::from_secs(2);
    assert_eq!(
        config.validate(),
        Err(ConfigError::ElectionSettings(
            "the duration is not longer than the renew deadline"
        ))
    );
}

#[test]
fn a_shared_lease_needs_an_address_a_client_can_dial() {
    // A lone controller needs none, on any listener.
    let mut config = valid();
    config.listeners[0].addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 30100));
    assert_eq!(config.validate(), Ok(()));

    // A shared lease behind the wildcard address is refused by name.
    config.election.lease = LeasePlace::File(PathBuf::from("/run/smg/lease"));
    assert_eq!(
        config.validate(),
        Err(ConfigError::AdvertiseAddressRequired)
    );
    // The IPv4 wildcard mapped into IPv6 is a wildcard listener too.
    config.listeners[0].addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED.to_ipv6_mapped(), 30100));
    assert_eq!(
        config.validate(),
        Err(ConfigError::AdvertiseAddressRequired)
    );
    // ... until an address is advertised, or the listener binds one.
    config.election.advertise = Some("10.0.0.7".to_owned());
    assert_eq!(config.validate(), Ok(()));
    config.election.advertise = None;
    config.listeners[0].addr = loopback(30100);
    assert_eq!(config.validate(), Ok(()));

    for bad in [
        "",
        "a b",
        "controller@host",
        "host:0",
        "host:port",
        ":30100",
        "a/b",
    ] {
        config.election.advertise = Some(bad.to_owned());
        assert_eq!(
            config.validate(),
            Err(ConfigError::MalformedAdvertiseAddress(bad.to_owned())),
            "{bad:?}"
        );
    }
    // The wildcard is a listener's address, never a client's, and a
    // host that ends in a number is an address to a resolver.
    for wildcard in ["0.0.0.0", "[::]:30100"] {
        config.election.advertise = Some(wildcard.to_owned());
        assert_eq!(
            config.validate(),
            Err(ConfigError::WildcardAdvertiseAddress(wildcard.to_owned())),
            "{wildcard:?}"
        );
    }
    config.election.advertise = Some("0:30100".to_owned());
    assert_eq!(
        config.validate(),
        Err(ConfigError::NumericAdvertiseHost("0:30100".to_owned()))
    );
    for good in [
        "10.0.0.7",
        "10.0.0.7:30100",
        "fd00::7",
        "[fd00::7]",
        "[fd00::7]:30100",
        "controller-0.policy.svc",
        "controller-0.policy.svc:30100",
    ] {
        config.election.advertise = Some(good.to_owned());
        assert_eq!(config.validate(), Ok(()), "{good}");
    }

    config.election.advertise = None;
    config.election.lease = LeasePlace::File(PathBuf::new());
    assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseName));
}

#[cfg(feature = "kubernetes")]
#[test]
fn a_lease_object_takes_a_name_and_whole_seconds() {
    let object = |namespace: Option<&str>, name: &str| LeasePlace::Kubernetes {
        namespace: namespace.map(str::to_owned),
        name: name.to_owned(),
    };
    let mut config = valid();
    config.election.lease = object(None, "smg-policy");
    assert_eq!(config.validate(), Ok(()));
    config.election.lease = object(Some("policy"), "smg-policy");
    assert_eq!(config.validate(), Ok(()));

    config.election.lease = object(None, "");
    assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseName));
    config.election.lease = object(Some(""), "smg-policy");
    assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseNamespace));

    // The object stores whole seconds: a fraction is refused here, not
    // at the first write.
    config.election.lease = object(None, "smg-policy");
    config.election.settings = ElectionSettings {
        duration: Duration::from_millis(2500),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_millis(1500),
    };
    assert_eq!(
        config.validate(),
        Err(ConfigError::LeaseDurationNotWholeSeconds(
            Duration::from_millis(2500)
        ))
    );
    // A file takes the same settings.
    config.election.lease = LeasePlace::File(PathBuf::from("/run/smg/lease"));
    assert_eq!(config.validate(), Ok(()));
}

#[test]
fn a_scope_names_itself_everywhere() {
    for scope in Scope::ALL {
        assert_eq!(scope.to_string(), scope.as_str());
    }
    assert_eq!(
        ConfigError::ScopeUnmounted(Scope::Replication).to_string(),
        "scope replication is mounted on no listener"
    );
}

#[test]
fn an_advertised_host_takes_the_listeners_port_and_a_host_with_a_port_keeps_its_own() {
    for (advertise, dialed) in [
        ("10.0.0.7", "10.0.0.7:30100"),
        ("10.0.0.7:31000", "10.0.0.7:31000"),
        ("fd00::7", "[fd00::7]:30100"),
        ("[fd00::7]", "[fd00::7]:30100"),
        ("[fd00::7]:31000", "[fd00::7]:31000"),
        ("controller-0.policy.svc", "controller-0.policy.svc:30100"),
        (
            "controller-0.policy.svc:31000",
            "controller-0.policy.svc:31000",
        ),
    ] {
        assert_eq!(
            dialable(advertise, 30100).as_deref(),
            Ok(dialed),
            "{advertise}"
        );
    }
    assert_eq!(
        dialable("10.0.0.7:0", 30100),
        Err(ConfigError::MalformedAdvertiseAddress(
            "10.0.0.7:0".to_owned()
        ))
    );
}

#[test]
fn the_wildcard_is_refused_in_every_spelling() {
    // A client that dials the wildcard reaches its own host: alone or
    // with a port, in either family, bracketed or bare, and as the IPv4
    // one mapped into IPv6.
    for wildcard in [
        "0.0.0.0",
        "0.0.0.0:31000",
        "::",
        "[::]",
        "[::]:31000",
        "::ffff:0.0.0.0",
        "[::ffff:0.0.0.0]",
        "[::ffff:0.0.0.0]:31000",
    ] {
        assert_eq!(
            dialable(wildcard, 30100),
            Err(ConfigError::WildcardAdvertiseAddress(wildcard.to_owned())),
            "{wildcard}"
        );
    }
    // Every other address is taken as written: a loopback one, a mapped
    // one, and a multicast or the broadcast one, the dial's to fail.
    for (advertise, dialed) in [
        ("0.0.0.1", "0.0.0.1:30100"),
        ("127.0.0.1", "127.0.0.1:30100"),
        ("127.0.0.1:31000", "127.0.0.1:31000"),
        ("::1", "[::1]:30100"),
        ("::ffff:10.0.0.7", "[::ffff:10.0.0.7]:30100"),
        ("224.0.0.1", "224.0.0.1:30100"),
        ("224.0.0.1:31000", "224.0.0.1:31000"),
        ("255.255.255.255", "255.255.255.255:30100"),
        ("[ff02::1]:31000", "[ff02::1]:31000"),
    ] {
        assert_eq!(
            dialable(advertise, 30100).as_deref(),
            Ok(dialed),
            "{advertise}"
        );
    }
}

#[test]
fn a_host_that_ends_in_a_number_is_refused_alone_and_with_a_port() {
    // What a resolver reads as an address though the standard parser
    // does not, and not always the one that was meant: `0` and `0x0`
    // are the wildcard, `010.0.0.1` is `8.0.0.1`, `8080` is
    // `0.0.31.144`.
    for numeric in [
        "0",
        "00",
        "0.0",
        "0.0.0",
        "0x0",
        "0X0",
        "0x0.0",
        "00.0.0.0",
        "000.000.000.000",
        "4294967295",
        "0xffffffff",
        "0xe0000001",
        "8080",
        "1.2.3",
        "010.0.0.1",
        "0.",
        "0x",
        "1.",
        "999.1.1.1",
        "example.123",
        "0xcafe",
    ] {
        for advertise in [numeric.to_owned(), format!("{numeric}:31000")] {
            assert_eq!(
                dialable(&advertise, 30100),
                Err(ConfigError::NumericAdvertiseHost(advertise.clone())),
                "{advertise}"
            );
        }
    }
    // A number inside a name, or a last label that only opens like one,
    // is a name's and taken as given; two closing dots leave an empty
    // last label, which is no number.
    for name in [
        "pod-0",
        "a1",
        "10.example",
        "0xcafe.example",
        "0xzz",
        "node7.cluster.local",
        "host.example.",
        "1..",
    ] {
        assert_eq!(dialable(name, 30100), Ok(format!("{name}:30100")), "{name}");
        let with_a_port = format!("{name}:31000");
        assert_eq!(
            dialable(&with_a_port, 30100).as_deref(),
            Ok(with_a_port.as_str()),
            "{with_a_port}"
        );
    }
}

#[test]
fn the_address_defaults_to_a_listener_that_binds_one() {
    let bound = |ip: std::net::IpAddr| SocketAddr::new(ip, 40123);
    let mut election = ElectionConfig::default();
    assert_eq!(
        election.address(bound(Ipv4Addr::LOCALHOST.into())),
        Ok(Some("127.0.0.1:40123".to_owned()))
    );
    assert_eq!(
        election.address(bound(Ipv6Addr::LOCALHOST.into())),
        Ok(Some("[::1]:40123".to_owned()))
    );
    // Behind a wildcard listener no client could be told an address:
    // either family's, and the IPv4 one mapped into IPv6.
    for wildcard in [
        Ipv4Addr::UNSPECIFIED.into(),
        Ipv6Addr::UNSPECIFIED.into(),
        Ipv4Addr::UNSPECIFIED.to_ipv6_mapped().into(),
    ] {
        assert_eq!(election.address(bound(wildcard)), Ok(None), "{wildcard}");
    }
    // A mapped address that is bound is told as the listener reads it.
    assert_eq!(
        election.address(bound(Ipv4Addr::LOCALHOST.to_ipv6_mapped().into())),
        Ok(Some("[::ffff:127.0.0.1]:40123".to_owned()))
    );

    // An advertised host takes the port the listener bound.
    election.advertise = Some("10.0.0.7".to_owned());
    assert_eq!(
        election.address(bound(Ipv4Addr::UNSPECIFIED.into())),
        Ok(Some("10.0.0.7:40123".to_owned()))
    );
}

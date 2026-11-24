//! The controller's bind installs the process crypto provider. One case, in
//! a test target of its own: a process starts with no provider installed,
//! and any other bind, listener or client of the process could install one
//! first. Where a build links one provider, the TLS library takes it by
//! itself at the first TLS configuration it builds: the install matters to
//! a build that links two.

use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use rustls::crypto::CryptoProvider;
use smg_policy_controller::service::controller::{
    Config, Controller, ElectionConfig, Listener, Scope,
};

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// A bind that names no TLS material installs the provider all the same:
/// the install stands ahead of every listener's material.
#[tokio::test]
async fn a_bind_installs_the_process_crypto_provider() {
    assert!(
        CryptoProvider::get_default().is_none(),
        "the process starts with no provider installed"
    );
    let config = Config {
        controller_id: "test-controller".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(100),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        worker_overload: Config::DEFAULT_WORKER_OVERLOAD,
        decide_shards: 2,
        election: ElectionConfig::default(),
        discovery: None,
    };
    let _controller = Controller::bind(config)
        .await
        .expect("the controller binds");
    assert!(
        CryptoProvider::get_default().is_some(),
        "the bind installed no crypto provider"
    );
}

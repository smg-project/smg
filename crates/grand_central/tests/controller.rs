//! The controller's listeners end to end: ephemeral ports, the health
//! service on every gRPC listener, the admin endpoints, readiness through a
//! shutdown, mutual TLS, and the refusals a bind names. Every test binds
//! `127.0.0.1:0`, so they run beside each other and beside another process.
//! The helpers return `Result`s and the tests expect them: a helper is not a
//! test to the lints.
//!
//! One module a subject holds the cases, each a file under
//! `controller_cases/`: `listeners.rs`, `sessions.rs`, `decisions.rs`,
//! `readiness.rs`, `loads.rs` and `measurement.rs`. This file holds what
//! two of them read: a listener and a start's configuration, the health
//! calls, a call that states a name, and a worker made ready.

#[expect(
    dead_code,
    reason = "this binary serves one mock engine and waits by its own bounds; the process's CPU time is another binary's measurement"
)]
#[path = "common/engines.rs"]
mod engines;
#[path = "common/net.rs"]
mod net;

use std::{net::SocketAddr, time::Duration};

use net::loopback;
use smg_grand_central::{
    engine::WorkerId,
    service::controller::{Config, Listener, Scope},
    store::{MemoryStore, Registrant, WorkerFacts},
    worker_host::{Members, Objects},
};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Request};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

#[path = "controller_cases/decisions.rs"]
mod decisions;
#[path = "controller_cases/listeners.rs"]
mod listeners;
#[path = "controller_cases/loads.rs"]
mod loads;
#[path = "controller_cases/measurement.rs"]
mod measurement;
#[path = "controller_cases/readiness.rs"]
mod readiness;
#[path = "controller_cases/sessions.rs"]
mod sessions;

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener::of(name.to_owned(), loopback(), scopes.to_vec(), None)
}

/// The binary's default shape on ephemeral loopback ports, with short
/// shutdown bounds.
fn config() -> Config {
    let mut config = Config::of(
        "test-controller".to_owned(),
        vec![
            listener("gateway", &[Scope::Gateway]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        loopback(),
    );
    config.shutdown_settle = Duration::from_millis(100);
    config.shutdown_grace = Duration::from_secs(5);
    config.decide_shards = 2;
    config
}

/// A plaintext health check of `service` on the listener at `addr`.
async fn health(addr: SocketAddr, service: &str) -> Result<ServingStatus, String> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .map_err(|error| error.to_string())?
        .connect()
        .await
        .map_err(|error| error.to_string())?;
    check(&mut HealthClient::new(channel), service).await
}

/// The status of `service`, or the gRPC code's name.
async fn check(client: &mut HealthClient<Channel>, service: &str) -> Result<ServingStatus, String> {
    client
        .check(HealthCheckRequest {
            service: service.to_owned(),
        })
        .await
        .map(|response| response.into_inner().status())
        .map_err(|status| format!("{:?}", status.code()))
}

/// `message` as a call that states `name`, or states none.
fn stating<T>(name: Option<&'static str>, message: T) -> Request<T> {
    use smg_grand_central::service::controller::PRINCIPAL_ENTRY;

    let mut request = Request::new(message);
    if let Some(name) = name {
        request
            .metadata_mut()
            .insert(PRINCIPAL_ENTRY, AsciiMetadataValue::from_static(name));
    }
    request
}

/// A ready worker of `stated`, written as a registration call writes one:
/// through the served controller's holder ([`Controller::workers`]), with
/// its connect signal behind the write. `None` where `store` took no
/// write.
///
/// [`Controller::workers`]: smg_grand_central::service::controller::Controller::workers
fn ready(holder: &Objects<Members>, store: &MemoryStore, stated: WorkerFacts) -> Option<WorkerId> {
    let id = holder.upsert(store, stated, Registrant::Control)?.id();
    let revision = holder.read_row(id)?.revision;
    holder.connect_row(store, id, revision);
    Some(id)
}

//! Test utilities for mesh module
use std::{io, net::SocketAddr, time::Duration};

use hyper_util::rt::TokioIo;
use tokio::{net::TcpListener, task::JoinHandle};
use tonic::transport::{Channel, Endpoint, Server};
use tower::service_fn;

use crate::{
    gossip_service::GossipService,
    service::gossip::{gossip_client::GossipClient, gossip_server::GossipServer},
};

/// Bind to an ephemeral port and return the listener + address.
/// The caller must keep the listener alive and pass it to `mesh_run!`
/// to avoid a TOCTOU port race.
pub async fn bind_node() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tracing::debug!("Bound node to {}", addr);
    (listener, addr)
}

/// Poll `condition` every 100ms until it returns true or `timeout` expires.
pub async fn wait_for<F>(condition: F, timeout: Duration, msg: &str)
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while !condition() {
        assert!(
            start.elapsed() <= timeout,
            "Timeout after {timeout:?} waiting for: {msg}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Connect the real gossip service over in-memory HTTP/2. Unlike a TCP socket,
/// this transport needs no wall-clock waits when a test pauses Tokio's clock.
pub async fn connect_gossip(service: GossipService) -> (GossipClient<Channel>, JoinHandle<()>) {
    let (client_io, server_io) = tokio::io::duplex(4096);
    #[expect(
        clippy::disallowed_methods,
        reason = "the caller owns the server task and aborts and awaits it"
    )]
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(GossipServer::new(service))
            .serve_with_incoming(tokio_stream::iter([Ok::<_, io::Error>(server_io)]))
            .await
            .unwrap();
    });
    let mut client_io = Some(client_io);
    let channel = Endpoint::from_static("http://mesh.test")
        .connect_with_connector(service_fn(move |_| {
            let io = client_io.take();
            async move {
                io.map(TokioIo::new)
                    .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))
            }
        }))
        .await
        .unwrap();
    (GossipClient::new(channel), server)
}

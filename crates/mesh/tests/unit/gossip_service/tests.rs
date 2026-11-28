//! The acceptor's connection accounting, read the way an operator reads
//! it: the text of a Prometheus recorder installed for the test thread
//! (the gateway installs the same recorder type for the process), with
//! the mesh nodes on a current-thread runtime so every task of theirs
//! records into it.

use std::{future::Future, net::SocketAddr, time::Duration};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::{
    service::gossip::{gossip_client::GossipClient, StreamMessage},
    tests::test_utils::{bind_node, wait_for},
    transport::sync_stream::build_heartbeat,
    MeshServerBuilder, MeshServerHandler,
};

const PEER_GAUGE: &str = "router_mesh_peer_connections";

/// Runs `scenario` on a current-thread runtime under a Prometheus
/// recorder installed for this thread; the rendered text, the way
/// `/metrics` serves it, once the scenario is done.
fn metrics_of<F, Fut>(scenario: F) -> String
where
    F: FnOnce(PrometheusHandle) -> Fut,
    Fut: Future<Output = ()>,
{
    let (rendered, ()) = metrics_and_result_of(scenario);
    rendered
}

/// [`metrics_of`] for a scenario with a result of its own: the rendered
/// text and that result.
fn metrics_and_result_of<F, Fut, T>(scenario: F) -> (String, T)
where
    F: FnOnce(PrometheusHandle) -> Fut,
    Fut: Future<Output = T>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let result = metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(scenario(handle.clone()))
    });
    (handle.render(), result)
}

/// The peer gauge's lines of a rendered scrape.
fn peer_gauge_lines(rendered: &str) -> Vec<&str> {
    rendered
        .lines()
        .filter(|line| line.starts_with(PEER_GAUGE))
        .collect()
}

/// A mesh node on an ephemeral loopback port, joining `init_peers`.
async fn node(name: &str, init_peers: Vec<SocketAddr>) -> (SocketAddr, MeshServerHandler) {
    let (listener, addr) = bind_node().await;
    let (server, handler) =
        MeshServerBuilder::new(name.to_string(), addr, addr, init_peers).build();
    #[expect(
        clippy::disallowed_methods,
        reason = "test node runs in the background for the duration of the assertion"
    )]
    tokio::spawn(async move {
        if let Err(e) = server.start_with_listener(listener).await {
            tracing::error!("Mesh node failed: {e}");
        }
    });
    (addr, handler)
}

/// Two members: A, the lower name, dials the sync stream; B accepts it
/// and records the peer gauge. The gauge appears under A's name and
/// under nothing else: no series with an empty `peer` for the time
/// before A's first frame named it.
#[test]
fn the_peer_gauge_carries_the_dialers_name_and_no_empty_label() {
    let rendered = metrics_of(|metrics| async move {
        let (addr_a, handler_a) = node("A", Vec::new()).await;
        let (_addr_b, handler_b) = node("B", vec![addr_a]).await;
        wait_for(
            || {
                peer_gauge_lines(&metrics.render())
                    .iter()
                    .any(|line| line.contains("peer=\"A\"") && line.ends_with(" 1"))
            },
            Duration::from_secs(20),
            "B accepts A's sync stream",
        )
        .await;
        handler_a.shutdown();
        handler_b.shutdown();
    });

    let lines = peer_gauge_lines(&rendered);
    assert!(
        !lines.iter().any(|line| line.contains("peer=\"\"")),
        "a peer series without a name: {lines:?}"
    );
}

/// A stream that closes before its first frame named the dialer was
/// never a peer connection: neither the gauge nor the reconnect count is
/// recorded for it, so no series with an empty `peer` is left behind.
#[test]
fn a_stream_closed_before_naming_its_peer_leaves_no_series() {
    let rendered = metrics_of(|_| async move {
        let (addr, handler) = node("A", Vec::new()).await;
        wait_for(
            || std::net::TcpStream::connect(addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;
        let mut client = GossipClient::connect(format!("http://{addr}"))
            .await
            .expect("connect to the mesh listener");
        // An outbound stream that ends at once: the acceptor reads no
        // frame. Its response stream ends once its handler has run its
        // close path, which is what the assertion is about.
        let response = client
            .sync_stream(tokio_stream::iter(Vec::<StreamMessage>::new()))
            .await
            .expect("open the sync stream");
        let mut inbound = response.into_inner();
        tokio::time::timeout(Duration::from_secs(10), async {
            while inbound.next().await.is_some() {}
        })
        .await
        .expect("the acceptor closes its side of a stream the dialer ended");
        handler.shutdown();
    });

    assert!(
        !rendered.contains("peer=\"\""),
        "a series without a peer name:\n{rendered}"
    );
}

/// A stream whose dialer named itself and then ended: the acceptor's
/// series for that name reads 1 while the stream is open and 0 after,
/// the closed stream is counted under that name, and no series carries
/// an empty `peer` at any point.
#[test]
fn a_named_stream_that_closed_reads_zero_under_its_name() {
    let rendered = metrics_of(|_| async move {
        let (addr, handler) = node("A", Vec::new()).await;
        wait_for(
            || std::net::TcpStream::connect(addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;
        let mut client = GossipClient::connect(format!("http://{addr}"))
            .await
            .expect("connect to the mesh listener");
        // One heartbeat naming the dialer, then the end of the stream.
        let response = client
            .sync_stream(tokio_stream::iter(vec![build_heartbeat(1, "probe")]))
            .await
            .expect("open the sync stream");
        let mut inbound = response.into_inner();
        tokio::time::timeout(Duration::from_secs(10), async {
            while inbound.next().await.is_some() {}
        })
        .await
        .expect("the acceptor closes its side of a stream the dialer ended");
        handler.shutdown();
    });

    assert!(
        rendered.contains("router_mesh_peer_connections{peer=\"probe\"} 0"),
        "the closed stream's series does not read 0 under its name:\n{rendered}"
    );
    assert!(
        rendered.contains("router_mesh_peer_reconnects_total{peer=\"probe\"} 1"),
        "the closed stream is not counted under its name:\n{rendered}"
    );
    assert!(
        !rendered.contains("peer=\"\""),
        "a series without a peer name:\n{rendered}"
    );
}

/// The peer gauge's line for `peer` in a rendered scrape.
fn peer_gauge_line(rendered: &str, peer: &str) -> Option<String> {
    let prefix = format!("{PEER_GAUGE}{{peer=\"{peer}\"}} ");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix))
        .map(str::to_owned)
}

/// Two accepted streams from one dialer overlap, as they do when the
/// dialer redials before the acceptor's handler of the old stream has
/// timed out: the gauge counts the open streams, so it reads 2 while
/// both are open and 1 once the first closed, not 0 while a stream is
/// live.
#[test]
fn overlapping_streams_of_one_dialer_add_up() {
    let (_, (while_both_open, after_first_closed)) = metrics_and_result_of(|metrics| async move {
        let (addr, handler) = node("A", Vec::new()).await;
        wait_for(
            || std::net::TcpStream::connect(addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;
        let mut client = GossipClient::connect(format!("http://{addr}"))
            .await
            .expect("connect to the mesh listener");
        // Each stream names the dialer with one heartbeat; the acceptor's
        // answer to it is the sign that its handler learned the name.
        let mut open = Vec::new();
        for _ in 0..2 {
            let (tx, rx) = mpsc::channel::<StreamMessage>(4);
            let mut inbound = client
                .sync_stream(ReceiverStream::new(rx))
                .await
                .expect("open the sync stream")
                .into_inner();
            tx.send(build_heartbeat(1, "probe"))
                .await
                .expect("send the heartbeat");
            tokio::time::timeout(Duration::from_secs(10), inbound.next())
                .await
                .expect("the acceptor answers the heartbeat")
                .expect("a frame")
                .expect("a frame, not a status");
            open.push((tx, inbound));
        }
        let while_both_open = metrics.render();

        // The first stream ends; its response stream ends once the
        // acceptor's handler has run its close path.
        let (tx_first, mut inbound_first) = open.remove(0);
        drop(tx_first);
        tokio::time::timeout(Duration::from_secs(10), async {
            while inbound_first.next().await.is_some() {}
        })
        .await
        .expect("the acceptor closes its side of a stream the dialer ended");
        let after_first_closed = metrics.render();

        drop(open);
        handler.shutdown();
        (while_both_open, after_first_closed)
    });

    assert_eq!(
        peer_gauge_line(&after_first_closed, "probe").as_deref(),
        Some("router_mesh_peer_connections{peer=\"probe\"} 1"),
        "one stream is still open after the first closed:\n{after_first_closed}"
    );
    assert_eq!(
        peer_gauge_line(&while_both_open, "probe").as_deref(),
        Some("router_mesh_peer_connections{peer=\"probe\"} 2"),
        "two streams were open:\n{while_both_open}"
    );
}

//! Both production ends of an idle gRPC stream, under Tokio's virtual clock.

use tokio::{sync::mpsc, time};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use super::*;
use crate::{
    gossip_service::GossipService, service::gossip::StreamMessageType,
    tests::test_utils::connect_gossip, transport::limits::STREAM_IDLE_TIMEOUT,
};

/// Real stream endpoints keep heartbeats periodic without echoing replies.
#[tokio::test(start_paused = true)]
async fn heartbeats_do_not_echo_and_keep_an_idle_stream_open() {
    let addr = "127.0.0.1:0".parse().unwrap();
    let acceptor = GossipService::new(Arc::default(), addr, addr, "B");
    let (mut client, server) = connect_gossip(acceptor).await;
    let (tx, rx) = mpsc::channel::<StreamMessage>(128);
    let (observed, mut heartbeats) = mpsc::unbounded_channel();
    let outbound = ReceiverStream::new(rx).map(move |msg| {
        if msg.message_type() == StreamMessageType::Heartbeat {
            observed.send(()).unwrap();
        }
        msg
    });
    let inbound = client.sync_stream(outbound).await.unwrap().into_inner();
    let dialer = GossipController::new(Arc::default(), addr, "A", Vec::new(), Dialer::new(None));
    let handler = dialer.spawn_sync_stream_handler(inbound, tx, "A".into(), "B".into());

    // Observe beyond the idle timeout on the same connection. An echo arrives
    // too early; a missing periodic heartbeat or reply eventually ends it.
    let rounds = STREAM_IDLE_TIMEOUT.as_secs() / STREAM_HEARTBEAT_INTERVAL.as_secs() + 1;
    for round in 0..=rounds {
        let start = time::Instant::now();
        time::timeout(
            STREAM_HEARTBEAT_INTERVAL + Duration::from_secs(2),
            heartbeats.recv(),
        )
        .await
        .expect("heartbeat before the deadline")
        .expect("stream remains open");
        if round > 0 {
            assert!(
                start.elapsed() >= STREAM_HEARTBEAT_INTERVAL,
                "heartbeat echo"
            );
        }
        assert!(!handler.is_finished(), "the same idle stream stays open");
    }
    assert!(
        time::timeout(Duration::from_secs(1), heartbeats.recv())
            .await
            .is_err(),
        "the final heartbeat's reply must not be echoed"
    );
    handler.abort();
    let _ = handler.await;
    server.abort();
    let _ = server.await;
}

//! Regression for zeromq/zmq.rs#249 at the Runtime::block_on entry point.
use std::{
    process::Command,
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use zeromq::{Socket, SocketRecv, SocketSend};

/// Bound the regression externally because a spinning poll prevents Tokio timeouts.
#[test]
fn large_frame_receive_yields() {
    const CHILD: &str = "SMG_ZMQ_FAIR_QUEUE_CHILD";
    if let Some(marker) = std::env::var_os(CHILD) {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(receive_large_frames());
        std::fs::write(marker, b"done").unwrap();
        return;
    }

    let marker_dir = tempfile::tempdir().unwrap();
    let marker = marker_dir.path().join("done");
    // Tokio timeouts cannot stop a FairQueue poll that spins forever.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "large_frame_receive_yields", "--nocapture"])
        .env(CHILD, &marker)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "large-frame receiver failed: {status}");
            assert!(
                marker.exists(),
                "child did not run the large-frame regression"
            );
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("large-frame receiver did not yield within 30 seconds");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// This transport regression deliberately uses Tokio directly, without SMG task wrappers.
#[expect(clippy::disallowed_methods, clippy::unwrap_used)]
async fn receive_large_frames() {
    const SIZE: usize = 64 * 1024 * 1024;
    let mut receiver = zeromq::PullSocket::new();
    let endpoint = receiver
        .bind("tcp://127.0.0.1:0")
        .await
        .unwrap()
        .to_string();
    let mut senders = Vec::new();
    for peer in 0..2u8 {
        let mut sender = zeromq::PushSocket::new();
        sender.connect(&endpoint).await.unwrap();
        senders.push(tokio::spawn(async move {
            for _ in 0..2 {
                sender
                    .send(Bytes::from(vec![peer; SIZE]).into())
                    .await
                    .unwrap();
            }
            sender
        }));
    }
    let mut received = [0; 2];
    for _ in 0..4 {
        let message = tokio::time::timeout(Duration::from_secs(10), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(message.len(), 1);
        let frame = message.get(0).unwrap();
        assert_eq!(frame.len(), SIZE);
        let peer = usize::from(frame[0]);
        assert!(peer < received.len());
        assert!(frame.iter().all(|byte| usize::from(*byte) == peer));
        received[peer] += 1;
    }
    assert_eq!(received, [2, 2]);
    // Keep both peers connected while checking an idle receive still times out.
    let mut connected = Vec::new();
    for sender in senders {
        connected.push(sender.await.unwrap());
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(50), receiver.recv())
            .await
            .is_err()
    );
    drop(connected);
}

/// Cancel an idle receive, then preserve empty/multipart frames and the next message.
#[tokio::test]
async fn cancelled_receive_preserves_multipart_and_following_message() {
    let mut receiver = zeromq::PullSocket::new();
    let endpoint = receiver
        .bind("tcp://127.0.0.1:0")
        .await
        .unwrap()
        .to_string();
    let mut sender = zeromq::PushSocket::new();
    sender.connect(&endpoint).await.unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), receiver.recv())
            .await
            .is_err()
    );
    let frames: Vec<Bytes> = (0..4096)
        .map(|index| {
            if index % 2 == 0 {
                Bytes::new()
            } else {
                Bytes::from(index.to_string())
            }
        })
        .collect();
    let message = zeromq::ZmqMessage::try_from(frames.clone()).unwrap();
    let (sent, received) = tokio::join!(sender.send(message), receiver.recv());
    sent.unwrap();
    let received = received.unwrap();
    assert_eq!(received.len(), frames.len());
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(received.get(index).unwrap(), frame);
    }
    let (sent, received) = tokio::join!(
        sender.send(Bytes::from_static(b"following").into()),
        receiver.recv()
    );
    sent.unwrap();
    assert_eq!(
        received.unwrap().get(0).unwrap(),
        &Bytes::from_static(b"following")
    );
}

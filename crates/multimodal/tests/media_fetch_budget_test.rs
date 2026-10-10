//! The fetch budgets of the media connector: a video fetched by URL has a
//! budget of its own, sized like the serving engine's own loader's (longer
//! than an image's), and a fetch that runs out of its budget fails at the
//! budget naming the URL and the budget, whether the headers or the body
//! were still outstanding.
#![allow(clippy::expect_used, clippy::print_stderr)]

use std::{
    io::{Read, Write},
    net::TcpListener,
    process::Stdio,
    thread,
    time::{Duration, Instant},
};

use llm_multimodal::{
    ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource, VideoFetchConfig,
};
use reqwest::Client;
use rustls::crypto::ring;
use tokio::process::Command;

/// One HTTP/1.1 response for the first connection: the headers after
/// `header_delay`, then `first` bytes of the body, then the rest after
/// `pause`. Returns the URL to fetch.
fn serve_once(
    body: Vec<u8>,
    content_type: &'static str,
    first: usize,
    pause: Duration,
    header_delay: Duration,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut head = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let read = stream.read(&mut buf).unwrap_or(0);
            if read == 0 {
                break;
            }
            head.extend_from_slice(&buf[..read]);
            if head.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        thread::sleep(header_delay);
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n",
            body.len()
        );
        if stream.write_all(headers.as_bytes()).is_err() {
            return;
        }
        let first = first.min(body.len());
        if stream.write_all(&body[..first]).is_err() {
            return;
        }
        let _ = stream.flush();
        thread::sleep(pause);
        let _ = stream.write_all(&body[first..]);
        let _ = stream.flush();
    });
    format!("http://127.0.0.1:{port}/media")
}

fn connector(config: MediaConnectorConfig) -> MediaConnector {
    let _ = ring::default_provider().install_default();
    let client = Client::builder().no_proxy().build().expect("client");
    MediaConnector::new(client, config).expect("media connector")
}

/// A short generated clip, or `None` where ffmpeg is not on PATH.
async fn tiny_clip() -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("clip.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .ok()?;
    if !status.success() {
        return None;
    }
    std::fs::read(&path).ok()
}

/// A clip whose body takes longer than the image budget still arrives: a
/// video is fetched under its own, longer budget.
#[tokio::test]
async fn a_slow_video_completes_within_the_video_budget() {
    let Some(clip) = tiny_clip().await else {
        eprintln!("ffmpeg not available; skipping");
        return;
    };
    let pause = Duration::from_millis(2500);
    let url = serve_once(clip, "video/mp4", 512, pause, Duration::ZERO);
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(1),
        ..MediaConnectorConfig::default()
    });
    let started = Instant::now();
    let clip = connector
        .fetch_video(MediaSource::Url(url), VideoFetchConfig::default())
        .await
        .expect("a video budget longer than the image budget covers a 2.5 s body");
    assert!(started.elapsed() >= pause, "{:?}", started.elapsed());
    assert!(!clip
        .materialized_frames()
        .expect("decoded frames")
        .is_empty());
}

/// A body that stalls past the budget fails at the budget, promptly, with
/// the URL and the budget in the error.
#[tokio::test]
async fn a_stalled_body_fails_at_the_budget_naming_the_url() {
    let url = serve_once(
        vec![0u8; 8192],
        "image/png",
        1024,
        Duration::from_secs(30),
        Duration::ZERO,
    );
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(1),
        ..MediaConnectorConfig::default()
    });
    let started = Instant::now();
    let error = connector
        .fetch_image(MediaSource::Url(url.clone()), ImageFetchConfig::default())
        .await
        .expect_err("the body never completes");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    let message = error.to_string();
    assert!(
        message.contains(&url) && message.contains("timed out after 1s"),
        "{message}"
    );
}

/// Headers that never come fail the same way.
#[tokio::test]
async fn late_headers_fail_at_the_budget_naming_the_url() {
    let url = serve_once(
        vec![0u8; 1024],
        "image/png",
        1024,
        Duration::ZERO,
        Duration::from_secs(30),
    );
    let connector = connector(MediaConnectorConfig {
        fetch_timeout: Duration::from_secs(1),
        ..MediaConnectorConfig::default()
    });
    let started = Instant::now();
    let error = connector
        .fetch_image(MediaSource::Url(url.clone()), ImageFetchConfig::default())
        .await
        .expect_err("the headers never come");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    let message = error.to_string();
    assert!(
        message.contains(&url) && message.contains("timed out after 1s"),
        "{message}"
    );
}

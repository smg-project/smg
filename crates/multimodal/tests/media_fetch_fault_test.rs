//! Whose fault a media fetch failure is, as the connector reads it off a live
//! fetch: the request's own for a host that cannot be resolved or refuses the
//! connection, a 4xx, or a body that is not the media it claims to be; the
//! host's for now for a 5xx, a transfer that breaks before the last byte, or
//! a fetch that runs out of its budget.
#![allow(clippy::expect_used)]

use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
    time::Duration,
};

use llm_multimodal::{
    ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaFault, MediaSource,
};
use reqwest::Client;

/// One HTTP/1.1 answer for the first connection, after `header_delay`:
/// `status` with `body` under `content_type`, declared as `declared` bytes
/// long (more than `body` breaks the transfer), then the connection closes.
/// Returns the URL to fetch.
fn serve_once(
    status: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
    declared: usize,
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
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {declared}\r\n\
             Connection: close\r\n\r\n"
        );
        let _ = stream.write_all(headers.as_bytes());
        let _ = stream.write_all(&body);
        let _ = stream.flush();
    });
    format!("http://127.0.0.1:{port}/picture.png")
}

/// The connector's reading of fetching `url` as an image under
/// `fetch_timeout`, with the error's words.
async fn fault_of(url: String, fetch_timeout: Duration) -> (MediaFault, String) {
    let client = Client::builder().no_proxy().build().expect("client");
    let connector = MediaConnector::new(
        client,
        MediaConnectorConfig {
            fetch_timeout,
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector");
    let error = connector
        .fetch_image(MediaSource::Url(url), ImageFetchConfig::default())
        .await
        .expect_err("the fetch fails");
    (error.fault(), error.to_string())
}

const BUDGET: Duration = Duration::from_secs(5);

#[tokio::test]
async fn a_host_that_answers_404_is_the_requests_fault() {
    let url = serve_once(
        "404 Not Found",
        "text/plain",
        b"missing".to_vec(),
        7,
        Duration::ZERO,
    );
    let (fault, message) = fault_of(url, BUDGET).await;
    assert_eq!(fault, MediaFault::Client, "{message}");
    assert!(message.contains("404"), "{message}");
}

#[tokio::test]
async fn a_host_that_cannot_be_resolved_is_the_requests_fault() {
    let (fault, message) = fault_of("http://media.invalid/picture.png".to_string(), BUDGET).await;
    assert_eq!(fault, MediaFault::Client, "{message}");
}

#[tokio::test]
async fn a_host_that_refuses_the_connection_is_the_requests_fault() {
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.local_addr().expect("local addr").port()
    };
    let (fault, message) = fault_of(format!("http://127.0.0.1:{port}/picture.png"), BUDGET).await;
    assert_eq!(fault, MediaFault::Client, "{message}");
}

#[tokio::test]
async fn a_body_that_is_not_the_media_it_claims_to_be_is_the_requests_fault() {
    let url = serve_once("200 OK", "text/plain", b"hello".to_vec(), 5, Duration::ZERO);
    let (fault, message) = fault_of(url, BUDGET).await;
    assert_eq!(fault, MediaFault::Client, "{message}");
    assert!(message.contains("image decode error"), "{message}");
}

#[tokio::test]
async fn a_host_that_answers_5xx_is_the_hosts_fault_for_now() {
    let url = serve_once(
        "503 Service Unavailable",
        "text/plain",
        b"later".to_vec(),
        5,
        Duration::ZERO,
    );
    let (fault, message) = fault_of(url, BUDGET).await;
    assert_eq!(fault, MediaFault::Transient, "{message}");
    assert!(message.contains("503"), "{message}");
}

#[tokio::test]
async fn a_transfer_that_breaks_is_the_hosts_fault_for_now() {
    let url = serve_once("200 OK", "image/png", vec![0u8; 16], 4096, Duration::ZERO);
    let (fault, message) = fault_of(url, BUDGET).await;
    assert_eq!(fault, MediaFault::Transient, "{message}");
}

#[tokio::test]
async fn a_fetch_that_runs_out_of_its_budget_is_the_hosts_fault_for_now() {
    let url = serve_once(
        "200 OK",
        "image/png",
        vec![0u8; 16],
        16,
        Duration::from_secs(3),
    );
    let (fault, message) = fault_of(url, Duration::from_millis(300)).await;
    assert_eq!(fault, MediaFault::Transient, "{message}");
    assert!(message.contains("timed out"), "{message}");
}

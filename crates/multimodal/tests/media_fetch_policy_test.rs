//! The media connector's host policy against local one-shot origins: an
//! allow-list bounds every host a fetch touches, redirect targets included; a
//! URL that names or resolves to a loopback or otherwise non-public address
//! is refused before anything is connected, unless private media hosts are
//! allowed; a redirect is followed only to an `http`/`https` target the
//! policy passes, at most ten times. Every refusal is the request's fault.
#![allow(clippy::expect_used, clippy::print_stderr)]

use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD, Engine};
use llm_multimodal::{
    ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaConnectorError, MediaFault,
    MediaSource,
};

mod common;

const TINY_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn read_head(stream: &mut std::net::TcpStream) {
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
}

/// An origin on `bind` answering every request with `respond(port, n)` for
/// its `n`th request; the counter is the requests it answered.
fn serve(
    bind: &str,
    respond: impl Fn(u16, usize) -> Vec<u8> + Send + 'static,
) -> Option<(u16, Arc<AtomicUsize>)> {
    let listener = TcpListener::bind(bind).ok()?;
    let port = listener.local_addr().expect("local addr").port();
    let hits = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&hits);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            read_head(&mut stream);
            let n = seen.fetch_add(1, Ordering::SeqCst);
            let _ = stream.write_all(&respond(port, n));
            let _ = stream.flush();
        }
    });
    Some((port, hits))
}

/// The tiny PNG on 127.0.0.1 (or on `bind`).
fn serve_png_on(bind: &str) -> Option<(u16, Arc<AtomicUsize>)> {
    serve(bind, |_, _| {
        let body = STANDARD.decode(TINY_PNG_B64).expect("png");
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(&body);
        response
    })
}

fn serve_png() -> (u16, Arc<AtomicUsize>) {
    serve_png_on("127.0.0.1:0").expect("bind loopback")
}

/// A redirector on 127.0.0.1: every request answers `status` with
/// `Location: <target(port, n)>`.
fn serve_redirect(
    status: u16,
    target: impl Fn(u16, usize) -> String + Send + 'static,
) -> (u16, Arc<AtomicUsize>) {
    serve("127.0.0.1:0", move |port, n| {
        format!(
            "HTTP/1.1 {status} Redirect\r\nLocation: {}\r\nContent-Length: 0\r\n\
             Connection: close\r\n\r\n",
            target(port, n)
        )
        .into_bytes()
    })
    .expect("bind loopback")
}

fn connector(allowed: Option<&[&str]>, allow_private_hosts: bool) -> MediaConnector {
    MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allowed_domains: allowed.map(|hosts| hosts.iter().map(ToString::to_string).collect()),
            allow_private_hosts,
            fetch_timeout: Duration::from_secs(5),
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector")
}

async fn fetch(connector: &MediaConnector, url: String) -> Result<(u32, u32), MediaConnectorError> {
    connector
        .fetch_image(MediaSource::Url(url), ImageFetchConfig::default())
        .await
        .map(|frame| (frame.size().width, frame.size().height))
}

/// The allow-list refuses a URL outside it by name before anything is
/// connected: the origin sees no request.
#[tokio::test]
async fn a_url_outside_the_allow_list_is_refused_before_connecting() {
    let (port, hits) = serve_png();
    let error = fetch(
        &connector(Some(&["example.com"]), true),
        format!("http://127.0.0.1:{port}/img.png"),
    )
    .await
    .expect_err("127.0.0.1 is not example.com");
    assert!(
        matches!(&error, MediaConnectorError::DisallowedDomain(host) if host == "127.0.0.1"),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Client);
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the origin must not be contacted"
    );
}

/// A redirect from an allow-listed host to a host outside the list is
/// refused, naming the target; the target sees no request.
#[tokio::test]
async fn a_redirect_from_an_allowed_host_to_another_host_is_refused() {
    let (png_port, png_hits) = serve_png();
    let (redirect_port, redirect_hits) = serve_redirect(302, move |_, _| {
        format!("http://127.0.0.1:{png_port}/img.png")
    });
    let error = fetch(
        &connector(Some(&["localhost"]), true),
        format!("http://localhost:{redirect_port}/redirect"),
    )
    .await
    .expect_err("the redirect target is not in the allow list");
    match &error {
        MediaConnectorError::DisallowedRedirect {
            url,
            target,
            reason,
        } => {
            assert_eq!(url, &format!("http://localhost:{redirect_port}/redirect"));
            assert_eq!(target, &format!("http://127.0.0.1:{png_port}/img.png"));
            assert!(reason.contains("not in the allow list"), "{reason}");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(error.fault(), MediaFault::Client);
    assert_eq!(redirect_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        png_hits.load(Ordering::SeqCst),
        0,
        "the redirect target must not be contacted"
    );
}

/// A redirect whose target passes the policy is followed, for each of the
/// redirect statuses a client follows, and the media arrives.
#[tokio::test]
async fn a_redirect_within_the_policy_is_followed() {
    for status in [301, 302, 303, 307, 308] {
        let (png_port, png_hits) = serve_png();
        let (redirect_port, redirect_hits) = serve_redirect(status, move |_, _| {
            format!("http://127.0.0.1:{png_port}/img.png")
        });
        let size = fetch(
            &connector(Some(&["127.0.0.1"]), true),
            format!("http://127.0.0.1:{redirect_port}/redirect"),
        )
        .await
        .unwrap_or_else(|error| panic!("{status}: {error}"));
        assert_eq!(size, (1, 1), "{status}");
        assert_eq!(redirect_hits.load(Ordering::SeqCst), 1, "{status}");
        assert_eq!(png_hits.load(Ordering::SeqCst), 1, "{status}");
    }
}

/// Without the opt-in, a URL on a loopback address is refused before
/// anything is connected, whether it names the address or a name that
/// resolves to it.
#[tokio::test]
async fn a_loopback_target_is_refused_unless_private_hosts_are_allowed() {
    let (port, hits) = serve_png();
    let strict = connector(None, false);

    let error = fetch(&strict, format!("http://127.0.0.1:{port}/img.png"))
        .await
        .expect_err("a loopback literal is refused");
    assert!(
        matches!(
            &error,
            MediaConnectorError::DisallowedAddress { host, class: "loopback", .. }
                if host == "127.0.0.1"
        ),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Client);

    let error = fetch(&strict, format!("http://localhost:{port}/img.png"))
        .await
        .expect_err("a name that resolves to loopback is refused");
    assert!(
        matches!(
            &error,
            MediaConnectorError::DisallowedAddress { host, class: "loopback", .. }
                if host == "localhost"
        ),
        "{error}"
    );

    let error = fetch(&strict, format!("http://[::ffff:127.0.0.1]:{port}/img.png"))
        .await
        .expect_err("loopback in IPv6 notation is refused");
    assert!(
        matches!(
            &error,
            MediaConnectorError::DisallowedAddress {
                class: "loopback",
                ..
            }
        ),
        "{error}"
    );
    if let Some((port6, hits6)) = serve_png_on("[::1]:0") {
        let error = fetch(&strict, format!("http://[::1]:{port6}/img.png"))
            .await
            .expect_err("the IPv6 loopback is refused");
        assert!(
            matches!(
                &error,
                MediaConnectorError::DisallowedAddress {
                    class: "loopback",
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(hits6.load(Ordering::SeqCst), 0);
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0, "nothing was connected");

    // The opt-in, for a deployment whose media origin is internal.
    let open = connector(None, true);
    assert_eq!(
        fetch(&open, format!("http://127.0.0.1:{port}/img.png"))
            .await
            .expect("allowed by the opt-in"),
        (1, 1)
    );
    assert_eq!(
        fetch(&open, format!("http://localhost:{port}/img.png"))
            .await
            .expect("allowed by the opt-in, by name"),
        (1, 1)
    );
    assert_eq!(hits.load(Ordering::SeqCst), 2);
}

/// A chain of more than ten redirects is refused where a client's default
/// would stop too, and the chain's URL is named.
#[tokio::test]
async fn a_redirect_chain_longer_than_ten_hops_is_refused() {
    let (port, hits) = serve_redirect(302, |port, n| {
        format!("http://127.0.0.1:{port}/hop{}", n + 1)
    });
    let error = fetch(
        &connector(None, true),
        format!("http://127.0.0.1:{port}/hop0"),
    )
    .await
    .expect_err("the chain never reaches media");
    assert!(
        matches!(
            &error,
            MediaConnectorError::TooManyRedirects { url, limit: 10 }
                if url == &format!("http://127.0.0.1:{port}/hop0")
        ),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Client);
    assert_eq!(
        hits.load(Ordering::SeqCst),
        11,
        "the first request and ten redirects"
    );
}

/// A redirect to a target that is no media URL (a file, for one) is refused.
#[tokio::test]
async fn a_redirect_to_a_non_http_target_is_refused() {
    let (port, hits) = serve_redirect(302, |_, _| "file:///etc/hostname".to_string());
    let error = fetch(
        &connector(None, true),
        format!("http://127.0.0.1:{port}/redirect"),
    )
    .await
    .expect_err("a file is no media URL");
    assert!(
        matches!(
            &error,
            MediaConnectorError::DisallowedRedirect { target, reason, .. }
                if target == "file:///etc/hostname" && reason.contains("unsupported media scheme: file")
        ),
        "{error}"
    );
    assert_eq!(error.fault(), MediaFault::Client);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

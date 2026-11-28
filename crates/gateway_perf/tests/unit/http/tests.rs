use std::{
    collections::BTreeMap,
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
};

use super::*;

/// A one-thread HTTP server for the tests: a map of `METHOD /path` to a raw answer
/// (the status and body; the headers are the server's), serving until dropped.
pub(crate) struct FakeServer {
    pub url: String,
    pub seen: Arc<Mutex<Vec<String>>>,
}

pub(crate) fn serve(answers: BTreeMap<String, (u16, String)>, chunked: bool) -> FakeServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_in = Arc::clone(&seen);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            // Read until the head and the declared body are in.
            while let Ok(n) = stream.read(&mut buf) {
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                if let Some(split) = find(&raw, b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&raw[..split]).to_string();
                    let length: usize = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if raw.len() >= split + 4 + length {
                        break;
                    }
                }
            }
            let head = String::from_utf8_lossy(&raw).to_string();
            let first = head.lines().next().unwrap_or_default().to_string();
            let mut parts = first.split_whitespace();
            let key = format!(
                "{} {}",
                parts.next().unwrap_or_default(),
                parts.next().unwrap_or_default()
            );
            seen_in.lock().unwrap().push(key.clone());
            let (status, body) = answers
                .get(&key)
                .cloned()
                .unwrap_or_else(|| (404, "unknown".to_string()));
            let reply = if chunked {
                format!(
                    "HTTP/1.1 {status} X\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                    body.len(),
                    body
                )
            } else {
                format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
            };
            let _ = stream.write_all(reply.as_bytes());
        }
    });
    FakeServer {
        url: format!("http://127.0.0.1:{port}"),
        seen,
    }
}

#[test]
fn get_post_and_delete_round_trip_with_content_length_and_chunked_bodies() {
    for chunked in [false, true] {
        let answers = BTreeMap::from([
            ("GET /a".to_string(), (200, r#"{"x":1}"#.to_string())),
            ("POST /b".to_string(), (200, "posted".to_string())),
            ("DELETE /c".to_string(), (204, String::new())),
        ]);
        let server = serve(answers, chunked);
        let t = Duration::from_secs(5);
        assert_eq!(get(&format!("{}/a", server.url), t).unwrap(), r#"{"x":1}"#);
        assert_eq!(
            post(&format!("{}/b", server.url), &serde_json::json!([1, 2]), t).unwrap(),
            "posted"
        );
        assert_eq!(delete(&format!("{}/c", server.url), t).unwrap(), "");
        let err = get(&format!("{}/missing", server.url), t)
            .unwrap_err()
            .to_string();
        assert!(err.contains("HTTP 404"), "{err}");
        assert_eq!(
            server.seen.lock().unwrap().as_slice(),
            ["GET /a", "POST /b", "DELETE /c", "GET /missing"]
        );
    }
}

#[test]
fn a_closed_port_is_an_error_not_a_hang() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let err = get(
        &format!("http://127.0.0.1:{port}/x"),
        Duration::from_millis(500),
    )
    .unwrap_err();
    assert!(err.to_string().contains("connecting"), "{err:#}");
}

#[test]
fn raw_answers_parse() {
    let r = parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi!!").unwrap();
    assert_eq!(
        r,
        Response {
            status: 200,
            body: "hi".to_string()
        }
    );
    let r = parse_response(b"HTTP/1.1 500 Oops\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;ext\r\nde\r\n0\r\n\r\n")
        .unwrap();
    assert_eq!(
        r,
        Response {
            status: 500,
            body: "abcde".to_string()
        }
    );
    assert!(parse_response(b"garbage").is_err());
}

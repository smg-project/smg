//! A small HTTP/1.1 client over `std::net` for the bed's loopback endpoints (the mock
//! workers' admin API, the gateway's metrics). The tool talks to 127.0.0.1 only, so it
//! carries no TLS stack and no async runtime: one connection per request, `Connection:
//! close`, the body read to the end (Content-Length or chunked).

use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

/// A response: its status and its body as text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: String,
}

/// `http://host:port/path` split into the authority and the path.
fn split_url(url: &str) -> Result<(&str, &str)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("{url}: only http:// urls are supported"))?;
    Ok(match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    })
}

/// One request; a 2xx answer's body, an error otherwise (as `urllib` raises on a
/// non-2xx status: the callers treat both the same).
pub fn request(method: &str, url: &str, body: Option<&Value>, timeout: Duration) -> Result<String> {
    let response = exchange(method, url, body, timeout)?;
    if !(200..300).contains(&response.status) {
        bail!(
            "{method} {url}: HTTP {} {}",
            response.status,
            response.body.chars().take(200).collect::<String>()
        );
    }
    Ok(response.body)
}

/// One request, whatever the status.
pub fn exchange(
    method: &str,
    url: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<Response> {
    let (authority, path) = split_url(url)?;
    let addr = authority
        .to_socket_addrs()
        .with_context(|| format!("{authority}: not an address"))?
        .next()
        .ok_or_else(|| anyhow!("{authority}: no address"))?;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("connecting to {authority}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let payload = body.map(serde_json::to_vec).transpose()?;
    let mut head =
        format!("{method} {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n");
    if let Some(payload) = &payload {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if let Some(payload) = &payload {
        stream.write_all(payload)?;
    }
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .with_context(|| format!("reading the answer of {method} {url}"))?;
    parse_response(&raw).with_context(|| format!("{method} {url}"))
}

/// The status line, the headers and the body of a raw HTTP/1.1 answer.
pub fn parse_response(raw: &[u8]) -> Result<Response> {
    let split = find(raw, b"\r\n\r\n").ok_or_else(|| anyhow!("no header end in the answer"))?;
    let header = std::str::from_utf8(&raw[..split]).context("the headers are not UTF-8")?;
    let mut lines = header.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("no status in {status_line:?}"))?;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "content-length" {
            content_length = value.parse().ok();
        } else if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    let rest = &raw[split + 4..];
    let body = if chunked {
        dechunk(rest)?
    } else if let Some(n) = content_length {
        rest[..n.min(rest.len())].to_vec()
    } else {
        rest.to_vec()
    };
    Ok(Response {
        status,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// The payload of a chunked body (sizes in hex, the trailer ignored).
fn dechunk(mut rest: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end =
            find(rest, b"\r\n").ok_or_else(|| anyhow!("a chunk size line is missing"))?;
        let size_text = std::str::from_utf8(&rest[..line_end])?.trim();
        let size_text = size_text.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("chunk size {size_text:?}"))?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if rest.len() < size {
            bail!("a chunk of {size} bytes is cut short");
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size..];
        rest = rest.strip_prefix(b"\r\n").unwrap_or(rest);
    }
}

/// `GET url`.
pub fn get(url: &str, timeout: Duration) -> Result<String> {
    request("GET", url, None, timeout)
}

/// `POST url` with a JSON body.
pub fn post(url: &str, body: &Value, timeout: Duration) -> Result<String> {
    request("POST", url, Some(body), timeout)
}

/// `DELETE url`.
pub fn delete(url: &str, timeout: Duration) -> Result<String> {
    request("DELETE", url, None, timeout)
}

#[cfg(test)]
#[path = "../tests/unit/http/tests.rs"]
pub(crate) mod tests;

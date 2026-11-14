//! What the controller's and the load driver's test binaries share, included
//! by path so each binary compiles only what it uses: a loopback address, one
//! HTTP/1.1 GET over a plain socket and the process's resident size.

use std::net::{Ipv4Addr, SocketAddr};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

/// Port `0` on the loopback interface: the bind picks the port.
pub fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

/// One HTTP/1.1 GET over a plain socket: the status code and the body.
pub async fn http_get(addr: SocketAddr, path: &str) -> Result<(u16, String), String> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|error| format!("connect {addr}: {error}"))?;
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .map_err(|error| error.to_string())?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|error| error.to_string())?;
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| format!("no status line in {text:?}"))?;
    let body = text.split("\r\n\r\n").nth(1).unwrap_or_default().to_owned();
    Ok((status, body))
}

/// The process's resident size in KiB, from `/proc/self/status`.
pub fn vm_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmRSS:")
                    .and_then(|rest| rest.trim().split(' ').next())
                    .and_then(|kib| kib.parse().ok())
            })
        })
        .unwrap_or(0)
}

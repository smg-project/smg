//! A TCP relay in front of a store, for the opt-in tests that need the store
//! to misbehave on the wire: it can hold every byte (a store that accepts the
//! connection and does not answer) or delay each chunk (a store far away).

#![allow(dead_code)]

use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

pub struct Relay {
    local: String,
    hold: Arc<AtomicBool>,
    delay_ms: Arc<AtomicU64>,
}

impl Relay {
    /// Starts a relay on a free local port in front of the server `url`
    /// names (`scheme://[user:password@]host[:port][/...]`, `default_port`
    /// when the url has none) and returns it with the url that goes through
    /// it.
    pub fn in_front_of(url: &str, default_port: u16) -> Result<(Relay, String), String> {
        let (before, host_port, after) = split_authority(url)?;
        let upstream = if host_port.contains(':') {
            host_port.to_string()
        } else {
            format!("{host_port}:{default_port}")
        };
        let upstream_addr = upstream
            .to_socket_addrs()
            .map_err(|e| format!("resolve {upstream}: {e}"))?
            .next()
            .ok_or_else(|| format!("{upstream} resolves to nothing"))?;
        let listener =
            TcpListener::bind("127.0.0.1:0").map_err(|e| format!("bind the relay: {e}"))?;
        let local = listener
            .local_addr()
            .map_err(|e| format!("the relay's address: {e}"))?
            .to_string();
        let hold = Arc::new(AtomicBool::new(false));
        let delay_ms = Arc::new(AtomicU64::new(0));
        let (hold_flag, delay) = (hold.clone(), delay_ms.clone());
        thread::spawn(move || {
            for client in listener.incoming() {
                let Ok(client) = client else { break };
                let Ok(server) = TcpStream::connect(upstream_addr) else {
                    break;
                };
                let (Ok(client_rx), Ok(server_rx)) = (client.try_clone(), server.try_clone())
                else {
                    break;
                };
                let (h, d) = (hold_flag.clone(), delay.clone());
                thread::spawn(move || pump(client_rx, server, h, d));
                let (h, d) = (hold_flag.clone(), delay.clone());
                thread::spawn(move || pump(server_rx, client, h, d));
            }
        });
        let relayed_url = format!("{before}{local}{after}");
        Ok((
            Relay {
                local,
                hold,
                delay_ms,
            },
            relayed_url,
        ))
    }

    /// The relay's `host:port`.
    pub fn addr(&self) -> &str {
        &self.local
    }

    /// Holds every byte in both directions, on the open connections and on
    /// new ones, until [`Relay::release`]: the store accepts and does not
    /// answer.
    pub fn hold(&self) {
        self.hold.store(true, Ordering::SeqCst);
    }

    /// Lets the held bytes and every later one through again.
    pub fn release(&self) {
        self.hold.store(false, Ordering::SeqCst);
    }

    /// Delays every chunk of bytes in each direction by `delay`.
    pub fn delay(&self, delay: Duration) {
        self.delay_ms.store(
            u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
            Ordering::SeqCst,
        );
    }
}

/// Splits `url` into what precedes its `host[:port]`, that authority, and
/// what follows it.
fn split_authority(url: &str) -> Result<(&str, &str, &str), String> {
    let scheme_end = url
        .find("://")
        .ok_or_else(|| format!("no scheme in the url {url}"))?
        + 3;
    let rest = &url[scheme_end..];
    let authority_end = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let host_start = authority.rfind('@').map_or(0, |i| i + 1);
    Ok((
        &url[..scheme_end + host_start],
        &authority[host_start..],
        &rest[authority_end..],
    ))
}

/// Copies one direction of a relayed connection, holding or delaying each
/// chunk as the relay is set.
fn pump(mut from: TcpStream, mut to: TcpStream, hold: Arc<AtomicBool>, delay_ms: Arc<AtomicU64>) {
    let mut chunk = [0u8; 65536];
    loop {
        let n = match from.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        while hold.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(20));
        }
        let delay = delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            thread::sleep(Duration::from_millis(delay));
        }
        if to.write_all(&chunk[..n]).is_err() {
            break;
        }
    }
    let _ = to.shutdown(Shutdown::Write);
}

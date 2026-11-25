//! What the test binaries that start a leader and its standby share,
//! included by path: the numbers a pair's replication port is drawn from,
//! and whether one is free on the standby's address.

use std::{
    io::{self, ErrorKind},
    net::{IpAddr, Ipv4Addr, TcpListener},
    sync::atomic::{AtomicUsize, Ordering},
};

/// The numbers a pair's replication port is drawn from. The leader of a
/// pair binds one, and its standby binds the same number on its own
/// address. A bind of port `0` by any other case or process could have
/// been handed that number there, so these lie below the range the host
/// hands such a bind from. Each draw of this process takes the next
/// number, a leader's and another address's from ranges apart: a replica
/// served alone on the standby's address takes no number a leader holds.
/// A number is its pair's while the pair's leader lives. A standby that
/// outlives its leader keeps it on its own address, so a pair's leader
/// takes a number only where [`free`] finds no holder there.
pub fn ports(ip: IpAddr) -> impl Iterator<Item = u16> {
    const SPAN: u16 = 4096;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    // Linux says where the range starts. A host that does not is taken to
    // start it at 32,768, Linux's own default, and so is one that starts
    // it at or under 12,288: its draws then lie inside its own range,
    // where a bind of port `0` can be handed one.
    let low = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
        .ok()
        .and_then(|range| range.split_whitespace().next()?.parse::<u16>().ok())
        .filter(|low| *low > 3 * SPAN)
        .unwrap_or(32_768);
    let leads = ip == IpAddr::V4(Ipv4Addr::LOCALHOST);
    let base = low - if leads { 2 * SPAN } else { SPAN };
    // Processes started together have ids side by side. Each starts at a
    // place its id spreads over the span, so their runs seldom meet.
    let spread = std::process::id().wrapping_mul(0x9E37_79B1) >> 20;
    let start = usize::try_from(spread).unwrap_or(0);
    (0..SPAN).map(move |_| {
        let drawn = start.wrapping_add(NEXT.fetch_add(1, Ordering::Relaxed)) % usize::from(SPAN);
        // A count under the span.
        base + u16::try_from(drawn).unwrap_or(0)
    })
}

/// Whether nothing holds `port` on `ip`. A pair's leader asks it of the
/// standby's address once it has bound the number itself. No other pair's
/// standby binds a number this leader holds, so the look disturbs no pair
/// and finds one holder alone: a standby that outlived its leader. A bind
/// refused for another reason is the caller's to fail on, as the draw's
/// own is.
pub fn free(ip: IpAddr, port: u16) -> io::Result<bool> {
    match TcpListener::bind((ip, port)) {
        Ok(_) => Ok(true),
        // In use on that address: a holder.
        Err(error) if error.kind() == ErrorKind::AddrInUse => Ok(false),
        Err(error) => Err(error),
    }
}

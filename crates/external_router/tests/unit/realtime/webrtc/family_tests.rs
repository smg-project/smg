use super::*;

fn v4(last: u8) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(198, 51, 100, last)), 3478)
}

fn v6(last: u16) -> SocketAddr {
    SocketAddr::new(
        IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, last)),
        3478,
    )
}

#[test]
fn an_ipv4_bind_takes_the_first_ipv4_answer() {
    assert_eq!(
        pick_stun_address([v6(1), v4(1), v4(2)], IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
        Some(v4(1))
    );
    assert_eq!(
        pick_stun_address([v6(1)], IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
        None
    );
}

#[test]
fn an_ipv6_bind_takes_the_first_ipv6_answer() {
    let bind = IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 7));
    assert_eq!(pick_stun_address([v4(1), v6(1), v6(2)], bind), Some(v6(1)));
    assert_eq!(
        pick_stun_address([v4(1)], bind),
        None,
        "a socket bound to one IPv6 address cannot reach an IPv4 server"
    );
}

#[test]
fn a_dual_stack_bind_falls_back_to_a_mapped_ipv4_answer() {
    let bind = IpAddr::V6(Ipv6Addr::UNSPECIFIED);
    assert_eq!(pick_stun_address([v4(1), v6(1)], bind), Some(v6(1)));
    assert_eq!(
        pick_stun_address([v4(1)], bind),
        Some(SocketAddr::new(
            IpAddr::V6(Ipv4Addr::new(198, 51, 100, 1).to_ipv6_mapped()),
            3478
        ))
    );
}

#[test]
fn the_default_bind_follows_the_serving_hosts_family() {
    for host in ["0.0.0.0", "127.0.0.1", "10.0.0.7", "localhost", ""] {
        assert_eq!(
            default_bind_addr(host),
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            "{host}"
        );
    }
    for host in ["::", "[::]", "::1", "[fd00::7]", " :: "] {
        assert_eq!(
            default_bind_addr(host),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            "{host}"
        );
    }
}

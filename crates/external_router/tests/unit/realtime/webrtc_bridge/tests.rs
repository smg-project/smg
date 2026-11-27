use std::net::{Ipv4Addr, Ipv6Addr};

use super::*;
use crate::test_support::{http_client, RefusedAddress};

const TXN: [u8; 12] = [
    0xb7, 0xe7, 0xa7, 0x01, 0xbc, 0x34, 0xd6, 0x86, 0xfa, 0x87, 0xdf, 0xae,
];

/// A STUN Binding Success Response carrying `addr` in XOR-MAPPED-ADDRESS,
/// encoded as RFC 5389 section 15.2 says.
fn binding_success(addr: SocketAddr) -> Vec<u8> {
    let cookie = 0x2112_A442u32.to_be_bytes();
    let mut attr = vec![0u8, if addr.is_ipv4() { 0x01 } else { 0x02 }];
    attr.extend_from_slice(&(addr.port() ^ 0x2112).to_be_bytes());
    match addr.ip() {
        IpAddr::V4(ip) => {
            attr.extend(ip.octets().iter().zip(cookie).map(|(o, k)| o ^ k));
        }
        IpAddr::V6(ip) => {
            let key: Vec<u8> = cookie.iter().chain(TXN.iter()).copied().collect();
            attr.extend(ip.octets().iter().zip(key).map(|(o, k)| o ^ k));
        }
    }
    let mut resp = vec![0x01, 0x01];
    resp.extend_from_slice(&((attr.len() + 4) as u16).to_be_bytes());
    resp.extend_from_slice(&cookie);
    resp.extend_from_slice(&TXN);
    resp.extend_from_slice(&0x0020u16.to_be_bytes());
    resp.extend_from_slice(&(attr.len() as u16).to_be_bytes());
    resp.extend(attr);
    resp
}

#[test]
fn xor_mapped_address_decodes_an_ipv4_reflexive_address() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 50_000);
    assert_eq!(
        parse_stun_xor_mapped_address(&binding_success(addr), &TXN),
        Some(addr)
    );
}

#[test]
fn xor_mapped_address_decodes_an_ipv6_reflexive_address() {
    let addr = SocketAddr::new(
        IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0xdead, 0xbeef)),
        50_001,
    );
    assert_eq!(
        parse_stun_xor_mapped_address(&binding_success(addr), &TXN),
        Some(addr)
    );
}

#[tokio::test]
async fn an_unspecified_ipv4_bind_takes_the_route_to_the_probe() {
    let probe = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 3478);
    let ip = resolve_candidate_ip(IpAddr::V4(Ipv4Addr::UNSPECIFIED), Some(probe))
        .await
        .unwrap();
    assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
}

#[tokio::test]
async fn an_unspecified_ipv6_bind_takes_the_route_to_the_probe() {
    // Nothing to assert on a host without an IPv6 loopback.
    if std::net::UdpSocket::bind("[::1]:0").is_err() {
        return;
    }
    let probe = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 3478);
    let ip = resolve_candidate_ip(IpAddr::V6(Ipv6Addr::UNSPECIFIED), Some(probe))
        .await
        .unwrap();
    assert_eq!(ip, IpAddr::V6(Ipv6Addr::LOCALHOST));
}

#[tokio::test]
async fn a_specific_bind_address_is_the_candidate() {
    let ip = IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 7));
    assert_eq!(resolve_candidate_ip(ip, None).await.unwrap(), ip);
}

#[test]
fn a_response_to_another_transaction_is_ignored() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)), 50_000);
    let other = [0u8; 12];
    assert_eq!(
        parse_stun_xor_mapped_address(&binding_success(addr), &other),
        None
    );
}

#[test]
fn idle_beyond_max_triggers_timeout() {
    assert!(bridge_should_time_out(
        Duration::from_secs(1),
        BRIDGE_MAX_IDLE + Duration::from_secs(1)
    ));
}

#[test]
fn session_beyond_max_triggers_timeout() {
    assert!(bridge_should_time_out(
        BRIDGE_MAX_SESSION + Duration::from_secs(1),
        Duration::from_secs(0)
    ));
}

#[test]
fn active_call_within_limits_does_not_time_out() {
    assert!(!bridge_should_time_out(
        Duration::from_secs(600),
        Duration::from_secs(1)
    ));
}

#[test]
fn outcome_classification_matches_who_left() {
    // cancelled → success regardless of anything else.
    assert!(bridge_ended_successfully(true, true, true, true));
    // client hung up (client not alive) → success.
    assert!(bridge_ended_successfully(false, false, false, true));
    // upstream died while client connected → worker failure.
    assert!(!bridge_ended_successfully(false, false, true, false));
    // upstream timed out silently while client connected → worker failure.
    assert!(!bridge_ended_successfully(false, true, true, true));
    // client-side idle timeout (upstream not flagged), both still alive → success.
    assert!(bridge_ended_successfully(false, false, true, true));
    // client hung up first, even if the upstream had also gone idle → success
    // (a client close must never be recorded as an upstream 502).
    assert!(bridge_ended_successfully(false, true, false, true));
}

/// A data-channel packet that arrives before the far side's channel is
/// open waits in the pending queue with its bytes and its binary flag,
/// in both directions; the queue holds a thousand packets and drops the
/// next.
#[tokio::test]
async fn a_packet_buffered_before_the_far_channel_opens_keeps_its_bytes() {
    use str0m::channel::ChannelConfig;
    let now = Instant::now();
    let mut client_rtc = RtcConfig::new().build(now);
    let mut upstream_rtc = RtcConfig::new().build(now);
    let client_id = client_rtc
        .direct_api()
        .create_data_channel(ChannelConfig::default());
    let upstream_id = upstream_rtc
        .direct_api()
        .create_data_channel(ChannelConfig::default());
    let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let bind = || UdpSocket::bind(SocketAddr::new(loopback, 0));
    let mut bridge = WebRtcBridge {
        client_rtc,
        client_socket: bind().await.expect("a loopback socket"),
        client_candidate_addr: SocketAddr::new(loopback, 1),
        upstream_rtc,
        upstream_socket: bind().await.expect("a loopback socket"),
        upstream_candidate_addr: SocketAddr::new(loopback, 2),
        client_channel: None,
        upstream_channel: Some(upstream_id),
        upstream_channel_ready: false,
        upstream_audio_mid: None,
        client_audio_mid: None,
        pending_to_client: Vec::new(),
        pending_to_upstream: Vec::new(),
        call_id: "call".to_string(),
        cancel_token: CancellationToken::new(),
    };
    let packet = |id, n: usize| ChannelData {
        id,
        binary: n.is_multiple_of(2),
        data: vec![n as u8; 4096],
    };
    for n in 0..1001 {
        bridge.handle_client_event(Event::ChannelData(packet(client_id, n)));
        bridge.handle_upstream_event(Event::ChannelData(packet(upstream_id, n)));
    }
    for pending in [&bridge.pending_to_upstream, &bridge.pending_to_client] {
        assert_eq!(pending.len(), 1000, "a thousand wait, the next is dropped");
        for (n, (binary, data)) in pending.iter().enumerate() {
            assert_eq!(*binary, n.is_multiple_of(2));
            assert_eq!(data.len(), 4096);
            assert!(data.iter().all(|byte| *byte == n as u8), "packet {n}");
        }
    }
}

/// An offer nothing answers fails without the request URL in its text,
/// which the setup's failure line writes. The HTTP client's own error
/// text ends with the URL, its query kept.
#[tokio::test]
async fn an_offer_nothing_answers_goes_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("http://{}/v1/realtime/calls?key=s3cret", refused.addr());

    let Err(BridgeSetupError::Other(error)) =
        send_sdp_to_upstream(&http_client(), &url, "Bearer k", "v=0", None, "m").await
    else {
        panic!("a failed send");
    };

    let text = error.to_string();
    assert!(!text.contains("s3cret"), "{text}");
    assert!(!text.contains(&refused.addr().to_string()), "{text}");
}

//! WebRTC-to-WebRTC relay bridge.
//!
//! Analogous to [`super::proxy`] for WebSocket, this module implements a
//! bidirectional bridge between a client-facing and an upstream-facing WebRTC
//! peer connection.  SMG terminates both connections and relays data-channel
//! messages plus audio RTP packets, giving it full visibility into the traffic.

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};

use str0m::{
    change::{SdpAnswer, SdpOffer},
    channel::{ChannelData, ChannelId},
    media::{Direction, MediaKind, Mid},
    net::{Protocol, Receive},
    rtp::{RtpPacket, RtpWrite},
    Candidate, Event, Input, Output, Rtc, RtcConfig,
};
use tokio::net::UdpSocket;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, trace, warn};

use super::registry::{ConnectionState, RealtimeRegistry};

// ---------------------------------------------------------------------------
// Termination policy
// ---------------------------------------------------------------------------

/// Tear down a bridge that has received no peer traffic for this long. A live
/// call carries constant RTP plus ICE/DTLS keepalives, so total silence this
/// long means the peer vanished without a graceful close (network drop, crash).
/// This is the backstop for the case `Event::Closed` never arrives — without it
/// the relay task, its two UDP sockets, both `Rtc` states, the registry entry,
/// and the worker-load guard would leak until process restart.
const BRIDGE_MAX_IDLE: Duration = Duration::from_secs(60);

/// Absolute cap on a single bridged call — a final backstop against any leak,
/// generous enough not to cut off a legitimately long realtime session.
const BRIDGE_MAX_SESSION: Duration = Duration::from_secs(4 * 60 * 60);

/// Whether the relay should terminate on its idle / max-duration backstop.
/// Pure and side-effect-free for testability.
fn bridge_should_time_out(session_elapsed: Duration, idle_elapsed: Duration) -> bool {
    session_elapsed > BRIDGE_MAX_SESSION || idle_elapsed > BRIDGE_MAX_IDLE
}

/// Classify a bridge's terminal outcome. `true` = success (client hangup,
/// cancellation, or max-session with both peers healthy); `false` = worker
/// failure (upstream died, or timed out silently, while the client was still
/// connected — recorded as a 502). Pure and side-effect-free for testability.
fn bridge_ended_successfully(
    cancelled: bool,
    upstream_timed_out: bool,
    client_alive: bool,
    upstream_alive: bool,
) -> bool {
    // Failure only when the upstream failed (died or timed out silently) *while
    // the client was still connected*. A client that hung up first is always a
    // success, even if the upstream had also gone idle by then.
    cancelled || !client_alive || (!upstream_timed_out && upstream_alive)
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error returned by [`WebRtcBridge::setup`].
#[derive(Debug)]
pub enum BridgeSetupError {
    /// Upstream returned a non-success HTTP status (e.g. 401, 400).
    /// Contains the status code and response body so the caller can forward
    /// the appropriate error to the client instead of always returning 502.
    UpstreamHttp {
        status: reqwest::StatusCode,
        body: String,
        content_type: Option<String>,
    },
    /// Any other setup failure (network, SDP parsing, ICE, etc.).
    Other(anyhow::Error),
}

impl std::fmt::Display for BridgeSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpstreamHttp { status, body, .. } => {
                write!(
                    f,
                    "Upstream SDP exchange failed: status={status}, body={body}"
                )
            }
            Self::Other(e) => write!(f, "{e}"),
        }
    }
}

impl From<anyhow::Error> for BridgeSetupError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err)
    }
}

impl From<std::io::Error> for BridgeSetupError {
    fn from(err: std::io::Error) -> Self {
        Self::Other(err.into())
    }
}

impl From<reqwest::Error> for BridgeSetupError {
    fn from(err: reqwest::Error) -> Self {
        Self::Other(err.into())
    }
}

impl From<str0m::RtcError> for BridgeSetupError {
    fn from(err: str0m::RtcError) -> Self {
        Self::Other(err.into())
    }
}

impl From<str0m::error::IceError> for BridgeSetupError {
    fn from(err: str0m::error::IceError) -> Self {
        Self::Other(err.into())
    }
}

impl From<str0m::error::SdpError> for BridgeSetupError {
    fn from(err: str0m::error::SdpError) -> Self {
        Self::Other(err.into())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`WebRtcBridge::setup`] that the caller can
/// `tokio::spawn` to run the relay loop.
pub struct WebRtcBridge {
    client_rtc: Rtc,
    client_socket: UdpSocket,
    /// The candidate address advertised in ICE (resolved IP + port).
    /// Used as `destination` in `Receive::new` so str0m matches packets.
    client_candidate_addr: SocketAddr,

    upstream_rtc: Rtc,
    upstream_socket: UdpSocket,
    /// The candidate address advertised in ICE (resolved IP + port).
    upstream_candidate_addr: SocketAddr,

    /// Data channel id on the *client* peer (set when ChannelOpen fires).
    client_channel: Option<ChannelId>,
    /// Data channel id on the *upstream* peer.
    /// Negotiated channel id, set at construction from the SDP exchange.
    /// Present early but NOT writable until `Event::ChannelOpen` fires.
    upstream_channel: Option<ChannelId>,
    /// `true` only after `Event::ChannelOpen` for `upstream_channel`.
    /// Needed because `upstream_channel` is `Some` from construction,
    /// before DTLS + SCTP complete and the channel becomes writable.
    upstream_channel_ready: bool,

    /// Audio mid on the *upstream* peer — used to look up a TX stream for
    /// forwarding client audio.
    upstream_audio_mid: Option<Mid>,
    /// Audio mid on the *client* peer — used to look up a TX stream for
    /// forwarding upstream audio.
    client_audio_mid: Option<Mid>,

    /// Upstream data-channel messages received before the client channel opens.
    /// Flushed once `client_channel` becomes `Some`.
    pending_to_client: Vec<(bool, Vec<u8>)>,

    /// Client data-channel messages received before the upstream channel opens.
    /// Flushed once `upstream_channel_ready` becomes `true` on `ChannelOpen`.
    pending_to_upstream: Vec<(bool, Vec<u8>)>,

    call_id: String,
    cancel_token: CancellationToken,
}

impl WebRtcBridge {
    /// Create both peer connections, perform SDP exchange with upstream, and
    /// return `(bridge, client_sdp_answer)`.
    ///
    /// The caller should then register the call, spawn `bridge.run()`, and
    /// return the SDP answer to the client.
    #[expect(
        clippy::too_many_arguments,
        reason = "setup requires all connection parameters"
    )]
    pub async fn setup(
        client_sdp_offer_str: &str,
        upstream_url: &str,
        auth_header: &str,
        session_config: Option<serde_json::Value>,
        call_id: String,
        http_client: &reqwest::Client,
        bind_addr: IpAddr,
        stun_server: Option<SocketAddr>,
    ) -> Result<(Self, String), BridgeSetupError> {
        // -- 1. Bind two UDP sockets (ephemeral ports) -----------------------
        // In production, `bind_addr` is typically `0.0.0.0` and both sockets
        // share the same bind/candidate IP (the server's routable address).
        //
        // For local development the user may set `--webrtc-bind-addr 127.0.0.1`
        // so the browser (same machine) can reach the client-facing peer.
        // A loopback address can't reach external servers, so the upstream
        // socket falls back to `0.0.0.0` in that case.
        let upstream_bind = if bind_addr.is_loopback() {
            IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)
        } else {
            bind_addr
        };

        let client_candidate_ip = resolve_candidate_ip(bind_addr).await?;
        let upstream_candidate_ip = resolve_candidate_ip(upstream_bind).await?;

        let client_socket = UdpSocket::bind(SocketAddr::new(bind_addr, 0)).await?;
        let upstream_socket = UdpSocket::bind(SocketAddr::new(upstream_bind, 0)).await?;

        let client_candidate =
            SocketAddr::new(client_candidate_ip, client_socket.local_addr()?.port());
        let upstream_candidate =
            SocketAddr::new(upstream_candidate_ip, upstream_socket.local_addr()?.port());

        // -- 1b. Gather server-reflexive candidate for *upstream* via STUN ---
        // Client-facing peer uses ICE-lite (host candidates only), so no STUN
        // needed there.
        let upstream_srflx = match stun_server {
            Some(stun) => {
                let srflx = stun_gather_srflx(&upstream_socket, stun).await;
                if srflx.is_some() {
                    info!(upstream_srflx = ?srflx, "STUN gathering complete");
                } else {
                    warn!(%stun, "STUN gathering failed for upstream socket");
                }
                srflx
            }
            None => None,
        };

        debug!(
            client_candidate = %client_candidate,
            upstream_candidate = %upstream_candidate,
            "Bound UDP sockets for WebRTC bridge"
        );

        // -- 2. Create upstream Rtc (offerer) --------------------------------
        let now = Instant::now();
        let mut upstream_rtc = RtcConfig::new().set_rtp_mode(true).build(now);
        upstream_rtc.add_local_candidate(Candidate::host(upstream_candidate, Protocol::Udp)?);
        if let Some(srflx) = upstream_srflx {
            upstream_rtc.add_local_candidate(Candidate::server_reflexive(
                srflx,
                upstream_candidate,
                Protocol::Udp,
            )?);
        }

        // Add audio transceiver + data channel
        let mut sdp_api = upstream_rtc.sdp_api();
        let upstream_audio_mid =
            sdp_api.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None);
        let upstream_channel_id = sdp_api.add_channel("oai-events".to_string());
        let (upstream_offer, pending) = sdp_api
            .apply()
            .ok_or_else(|| anyhow::anyhow!("SDP apply produced no offer"))?;

        // -- 3. Send offer to OpenAI, get answer ----------------------------
        let upstream_answer = send_sdp_to_upstream(
            http_client,
            upstream_url,
            auth_header,
            &upstream_offer.to_sdp_string(),
            session_config,
        )
        .await?;

        upstream_rtc
            .sdp_api()
            .accept_answer(pending, upstream_answer)?;

        // -- 4. Create client Rtc (answerer, ICE-lite) ----------------------
        // ICE-lite: SMG only responds to the browser's connectivity checks.
        // This avoids needing to resolve mDNS candidates the browser may
        // advertise, and is the standard mode for SFU/relay servers.
        let mut client_rtc = RtcConfig::new()
            .set_rtp_mode(true)
            .set_ice_lite(true)
            .build(Instant::now());
        client_rtc.add_local_candidate(Candidate::host(client_candidate, Protocol::Udp)?);

        let client_offer = SdpOffer::from_sdp_string(client_sdp_offer_str)?;
        let client_answer = client_rtc.sdp_api().accept_offer(client_offer)?;
        let client_audio_mid = find_audio_mid(&client_rtc);

        let answer_sdp = client_answer.to_sdp_string();

        // Log sanitized SDP metadata (never the raw SDP which contains
        // ICE credentials, candidate IPs, and DTLS fingerprints).
        let ice_candidates = answer_sdp
            .lines()
            .filter(|l| l.starts_with("a=candidate:"))
            .count();
        let has_ice_credentials = answer_sdp.contains("a=ice-ufrag:");
        let dtls_fingerprint_algo = answer_sdp
            .lines()
            .find(|l| l.starts_with("a=fingerprint:"))
            .and_then(|l| l.strip_prefix("a=fingerprint:"))
            .and_then(|l| l.split_whitespace().next())
            .unwrap_or("none");
        let codec_names: Vec<&str> = answer_sdp
            .lines()
            .filter_map(|l| l.strip_prefix("a=rtpmap:"))
            .filter_map(|l| l.split_whitespace().nth(1))
            .filter_map(|l| l.split('/').next())
            .collect();
        info!(
            call_id,
            ?upstream_audio_mid,
            ?client_audio_mid,
            ice_candidates,
            has_ice_credentials,
            dtls_fingerprint_algo,
            ?codec_names,
            "WebRTC bridge SDP exchange complete"
        );

        Ok((
            Self {
                client_rtc,
                client_socket,
                client_candidate_addr: client_candidate,
                upstream_rtc,
                upstream_socket,
                upstream_candidate_addr: upstream_candidate,
                client_channel: None,
                pending_to_client: Vec::new(),
                upstream_channel: Some(upstream_channel_id),
                upstream_channel_ready: false,
                pending_to_upstream: Vec::new(),
                upstream_audio_mid: Some(upstream_audio_mid),
                client_audio_mid,
                call_id,
                cancel_token: CancellationToken::new(),
            },
            answer_sdp,
        ))
    }

    /// Replace the bridge's cancel token with the registry's token so that
    /// hangup cancellation is observed by the relay loop.
    pub fn set_cancel_token(&mut self, token: CancellationToken) {
        self.cancel_token = token;
    }

    /// Run the bidirectional relay until cancelled or disconnected.
    ///
    /// Returns `true` if the session ended normally (cancellation or client
    /// disconnect), `false` if the upstream worker died while the client was
    /// still connected (indicates a worker-side failure).
    pub async fn run(mut self, registry: Arc<RealtimeRegistry>) -> bool {
        registry.set_call_state(&self.call_id, ConnectionState::Connected);

        // 4096 bytes is sufficient for RTP packets (~1200-1500 bytes) and
        // DTLS handshake fragments (up to MTU). Larger DTLS messages are
        // fragmented at the DTLS layer before hitting UDP.
        let mut buf_client = vec![0u8; 4096];
        let mut buf_upstream = vec![0u8; 4096];

        let mut cancelled = false;
        // Set when the idle backstop fires with the *upstream* side silent, so the
        // outcome is classified as a worker failure (502) rather than success.
        let mut upstream_timed_out = false;

        // Liveness backstop: a peer that vanishes without a graceful
        // `Event::Closed` (network drop, crash, sustained ICE disconnect) simply
        // stops sending. Track per-peer when we last accepted a *valid* packet so
        // the loop tears down even when only one side goes silent while the other
        // keeps sending — otherwise the live side would refresh a shared timer.
        let session_start = Instant::now();
        let mut client_last_activity = Instant::now();
        let mut upstream_last_activity = Instant::now();

        // Seed initial timeout by draining any outputs produced during setup.
        let t_c = self.process_outputs(Side::Client).await;
        let t_u = self.process_outputs(Side::Upstream).await;
        let mut next_timeout = earliest_timeout(t_c, t_u);

        loop {
            // Dynamic timeout driven by str0m's internal state machines
            // (DTLS retransmits, ICE keepalives, RTCP timers). Falls back to
            // 50ms if neither peer has a scheduled timeout.
            let sleep_dur = next_timeout.saturating_duration_since(Instant::now());

            tokio::select! {
                result = self.client_socket.recv_from(&mut buf_client) => {
                    if self.handle_udp_recv(result, &buf_client, Side::Client) {
                        client_last_activity = Instant::now();
                    }
                }

                result = self.upstream_socket.recv_from(&mut buf_upstream) => {
                    if self.handle_udp_recv(result, &buf_upstream, Side::Upstream) {
                        upstream_last_activity = Instant::now();
                    }
                }

                () = tokio::time::sleep(sleep_dur) => {
                    let now = Instant::now();
                    let _ = self.client_rtc.handle_input(Input::Timeout(now));
                    let _ = self.upstream_rtc.handle_input(Input::Timeout(now));
                }

                () = self.cancel_token.cancelled() => {
                    debug!(call_id = self.call_id, "WebRTC bridge cancelled");
                    cancelled = true;
                    break;
                }
            }

            // Drain outputs from both peers and compute next timeout.
            let t_c = self.process_outputs(Side::Client).await;
            let t_u = self.process_outputs(Side::Upstream).await;
            next_timeout = earliest_timeout(t_c, t_u);

            // Backstop for a peer that vanished without a graceful close: if
            // *either* side is silent past the idle limit (or the session hits its
            // max), tear down. Record an upstream-side idle timeout so the outcome
            // is classified as a worker failure below.
            let client_idle = client_last_activity.elapsed();
            let upstream_idle = upstream_last_activity.elapsed();
            if bridge_should_time_out(session_start.elapsed(), client_idle.max(upstream_idle)) {
                upstream_timed_out = upstream_idle > BRIDGE_MAX_IDLE;
                warn!(
                    call_id = self.call_id,
                    session_secs = session_start.elapsed().as_secs(),
                    client_idle_secs = client_idle.as_secs(),
                    upstream_idle_secs = upstream_idle.as_secs(),
                    "WebRTC bridge timed out (idle or max duration); tearing down"
                );
                break;
            }

            // Exit if either peer is dead
            if !self.client_rtc.is_alive() || !self.upstream_rtc.is_alive() {
                info!(
                    call_id = self.call_id,
                    client_alive = self.client_rtc.is_alive(),
                    upstream_alive = self.upstream_rtc.is_alive(),
                    "WebRTC bridge peer disconnected"
                );
                break;
            }
        }

        // Upstream dying — or timing out silently — while the client is still
        // connected is a worker failure; cancellation and client hangup succeed.
        let success = bridge_ended_successfully(
            cancelled,
            upstream_timed_out,
            self.client_rtc.is_alive(),
            self.upstream_rtc.is_alive(),
        );

        self.client_rtc.disconnect();
        self.upstream_rtc.disconnect();
        registry.set_call_state(&self.call_id, ConnectionState::Disconnected);
        debug!(call_id = self.call_id, success, "WebRTC bridge ended");

        success
    }
}

// ---------------------------------------------------------------------------
// Internal: side abstraction for deduplication
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Side {
    Client,
    Upstream,
}

impl WebRtcBridge {
    fn rtc(&mut self, side: Side) -> &mut Rtc {
        match side {
            Side::Client => &mut self.client_rtc,
            Side::Upstream => &mut self.upstream_rtc,
        }
    }

    fn candidate_addr(&self, side: Side) -> SocketAddr {
        match side {
            Side::Client => self.client_candidate_addr,
            Side::Upstream => self.upstream_candidate_addr,
        }
    }

    fn socket(&self, side: Side) -> &UdpSocket {
        match side {
            Side::Client => &self.client_socket,
            Side::Upstream => &self.upstream_socket,
        }
    }

    fn side_label(side: Side) -> &'static str {
        match side {
            Side::Client => "client",
            Side::Upstream => "upstream",
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: UDP recv + output processing
// ---------------------------------------------------------------------------

impl WebRtcBridge {
    /// Feed a received datagram into the peer's `Rtc`. Returns `true` only when a
    /// valid datagram was accepted by str0m, so the caller counts it as liveness
    /// activity — a recv error or a rejected/garbage datagram must not refresh
    /// the idle timeout (otherwise a peer could be kept "alive" by junk traffic).
    fn handle_udp_recv(
        &mut self,
        result: std::io::Result<(usize, SocketAddr)>,
        buf: &[u8],
        side: Side,
    ) -> bool {
        let label = Self::side_label(side);
        let (n, source) = match result {
            Ok(pair) => pair,
            Err(e) => {
                warn!(call_id = self.call_id, error = %e, "{label} UDP recv error");
                return false;
            }
        };

        trace!(call_id = self.call_id, %source, n, "{label} UDP packet");
        let dest = self.candidate_addr(side);
        match Receive::new(Protocol::Udp, source, dest, &buf[..n]) {
            Ok(recv) => match self
                .rtc(side)
                .handle_input(Input::Receive(Instant::now(), recv))
            {
                Ok(()) => true,
                Err(e) => {
                    warn!(call_id = self.call_id, error = %e, %source, "{label}_rtc rejected input");
                    false
                }
            },
            Err(e) => {
                debug!(call_id = self.call_id, error = %e, %source, n, "{label} Receive::new failed");
                false
            }
        }
    }

    /// Drain all pending outputs from one Rtc peer.
    ///
    /// Returns the `Instant` at which str0m next needs a timeout input,
    /// used by the caller to compute a precise sleep duration.
    async fn process_outputs(&mut self, side: Side) -> Option<Instant> {
        loop {
            let output = match side {
                Side::Client => self.client_rtc.poll_output(),
                Side::Upstream => self.upstream_rtc.poll_output(),
            };
            match output {
                Ok(Output::Transmit(t)) => {
                    let label = Self::side_label(side);
                    let dest = t.destination;
                    let len = t.contents.len();
                    trace!(call_id = self.call_id, %dest, len, "{label} Rtc → transmit");
                    if let Err(e) = self.socket(side).send_to(&t.contents, dest).await {
                        warn!(
                            call_id = self.call_id,
                            error = %e,
                            %dest,
                            len,
                            "{label} UDP send failed, tearing down bridge"
                        );
                        self.cancel_token.cancel();
                        return None;
                    }
                }
                Ok(Output::Event(event)) => match side {
                    Side::Client => self.handle_client_event(event),
                    Side::Upstream => self.handle_upstream_event(event),
                },
                Ok(Output::Timeout(t)) => return Some(t),
                Err(e) => {
                    debug!(call_id = self.call_id, error = %e, "poll_output error");
                    return None;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: event handling
// ---------------------------------------------------------------------------

impl WebRtcBridge {
    fn handle_client_event(&mut self, event: Event) {
        match event {
            Event::Connected => {
                info!(call_id = self.call_id, "Client peer connected");
            }
            Event::IceConnectionStateChange(state) => {
                info!(call_id = self.call_id, ?state, "Client ICE state");
            }
            Event::ChannelOpen(id, label) => {
                debug!(
                    call_id = self.call_id,
                    ?id,
                    label,
                    "Client data channel opened"
                );
                self.client_channel = Some(id);
                self.flush_pending_to_client();
            }
            Event::ChannelData(data) => {
                log_data_channel(&self.call_id, &data, "Client→Upstream");
                if self.upstream_channel_ready {
                    if let Some(ch_id) = self.upstream_channel {
                        if let Some(mut ch) = self.upstream_rtc.channel(ch_id) {
                            let _ = ch.write(data.binary, &data.data);
                        }
                    }
                } else if self.pending_to_upstream.len() >= 1000 {
                    warn!(
                        call_id = self.call_id,
                        "Pending-to-upstream buffer full, dropping message"
                    );
                } else {
                    trace!(
                        call_id = self.call_id,
                        "Buffering client event (upstream channel not open)"
                    );
                    self.pending_to_upstream
                        .push((data.binary, data.data.to_vec()));
                }
            }
            Event::ChannelClose(id) => {
                debug!(call_id = self.call_id, ?id, "Client data channel closed");
                if self.client_channel == Some(id) {
                    self.client_channel = None;
                    self.pending_to_client.clear();
                }
            }
            Event::RtpPacket(pkt) => {
                trace!(call_id = self.call_id, "Client→Upstream RTP");
                self.forward_rtp(&pkt, Side::Upstream);
            }
            Event::MediaAdded(added) => {
                debug!(call_id = self.call_id, mid = ?added.mid, kind = ?added.kind, "Client media added");
            }
            // Graceful peer close (DTLS close_notify / SCTP association lost,
            // e.g. a browser `pc.close()`). str0m does not flip `is_alive()`
            // itself, so disconnect the reporting side to make the relay loop's
            // `is_alive()` exit fire — otherwise the call leaks until restart.
            Event::Closed => {
                info!(
                    call_id = self.call_id,
                    "Client peer closed; tearing down bridge"
                );
                self.client_rtc.disconnect();
            }
            _ => {}
        }
    }

    fn handle_upstream_event(&mut self, event: Event) {
        match event {
            Event::Connected => {
                info!(call_id = self.call_id, "Upstream peer connected");
            }
            Event::IceConnectionStateChange(state) => {
                info!(call_id = self.call_id, ?state, "Upstream ICE state");
            }
            Event::ChannelOpen(id, label) => {
                debug!(
                    call_id = self.call_id,
                    ?id,
                    label,
                    "Upstream data channel opened"
                );
                self.upstream_channel = Some(id);
                self.upstream_channel_ready = true;
                self.flush_pending_to_upstream();
            }
            Event::ChannelData(data) => {
                log_data_channel(&self.call_id, &data, "Upstream→Client");
                if let Some(ch_id) = self.client_channel {
                    if let Some(mut ch) = self.client_rtc.channel(ch_id) {
                        let _ = ch.write(data.binary, &data.data);
                    }
                } else if self.pending_to_client.len() >= 1000 {
                    warn!(
                        call_id = self.call_id,
                        "Pending-to-client buffer full, dropping message"
                    );
                } else {
                    trace!(
                        call_id = self.call_id,
                        "Buffering upstream event (client channel not open)"
                    );
                    self.pending_to_client
                        .push((data.binary, data.data.to_vec()));
                }
            }
            Event::ChannelClose(id) => {
                debug!(call_id = self.call_id, ?id, "Upstream data channel closed");
                if self.upstream_channel == Some(id) {
                    self.upstream_channel = None;
                    self.upstream_channel_ready = false;
                    self.pending_to_upstream.clear();
                }
            }
            Event::RtpPacket(pkt) => {
                trace!(call_id = self.call_id, "Upstream→Client RTP");
                self.forward_rtp(&pkt, Side::Client);
            }
            Event::MediaAdded(added) => {
                debug!(call_id = self.call_id, mid = ?added.mid, kind = ?added.kind, "Upstream media added");
            }
            // Graceful peer close — disconnect the reporting side so the relay
            // loop's `is_alive()` exit fires (str0m does not flip it itself).
            Event::Closed => {
                info!(
                    call_id = self.call_id,
                    "Upstream peer closed; tearing down bridge"
                );
                self.upstream_rtc.disconnect();
            }
            _ => {}
        }
    }

    /// Send all buffered upstream events to the now-open client data channel.
    fn flush_pending_to_client(&mut self) {
        let Some(ch_id) = self.client_channel else {
            return;
        };
        let pending = std::mem::take(&mut self.pending_to_client);
        if pending.is_empty() {
            return;
        }
        info!(
            call_id = self.call_id,
            count = pending.len(),
            "Flushing buffered upstream events to client"
        );
        if let Some(mut ch) = self.client_rtc.channel(ch_id) {
            for (binary, data) in pending {
                let _ = ch.write(binary, &data);
            }
        }
    }

    fn flush_pending_to_upstream(&mut self) {
        let Some(ch_id) = self.upstream_channel else {
            return;
        };
        let pending = std::mem::take(&mut self.pending_to_upstream);
        if pending.is_empty() {
            return;
        }
        info!(
            call_id = self.call_id,
            count = pending.len(),
            "Flushing buffered client events to upstream"
        );
        if let Some(mut ch) = self.upstream_rtc.channel(ch_id) {
            for (binary, data) in pending {
                let _ = ch.write(binary, &data);
            }
        }
    }

    /// Forward an RTP packet to the target side's audio track.
    fn forward_rtp(&mut self, pkt: &RtpPacket, target: Side) {
        let (mid, rtc) = match target {
            Side::Upstream => (self.upstream_audio_mid, &mut self.upstream_rtc),
            Side::Client => (self.client_audio_mid, &mut self.client_rtc),
        };
        let Some(mid) = mid else { return };
        if let Some(tx) = rtc.direct_api().stream_tx_by_mid(mid, None) {
            // str0m 0.20 replaced the positional write_rtp args with an
            // RtpWrite builder; optional fields default off, so marker/
            // ext_vals/nackable must be set explicitly to preserve behavior.
            tx.write_rtp(
                RtpWrite::new(
                    pkt.header.payload_type,
                    pkt.seq_no,
                    pkt.header.timestamp,
                    pkt.timestamp,
                    pkt.payload.clone(),
                )
                .marker(pkt.header.marker)
                .ext_vals(pkt.header.ext_vals.clone())
                .nackable(true),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Logging helpers — mirror proxy.rs patterns
// ---------------------------------------------------------------------------

/// Log a data channel message with appropriate verbosity based on event type.
fn log_data_channel(call_id: &str, data: &ChannelData, direction: &str) {
    if data.binary {
        trace!(call_id, bytes = data.data.len(), "{direction} binary");
        return;
    }
    let Ok(text) = std::str::from_utf8(&data.data) else {
        return;
    };

    // Extract the raw "type" field from JSON so we always log the actual
    // event type, even for events our protocol crate doesn't recognise yet.
    let et = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|v| v.get("type")?.as_str().map(String::from));

    let Some(et) = et else {
        debug!(call_id, "{direction} (non-JSON or missing type)");
        return;
    };

    match et.as_str() {
        // High-frequency streaming deltas → trace
        "input_audio_buffer.append"
        | "response.output_audio.delta"
        | "response.output_text.delta"
        | "response.output_audio_transcript.delta"
        | "response.function_call_arguments.delta" => {
            trace!(call_id, event_type = et, "{direction}");
        }
        // Key lifecycle events → info
        "session.created"
        | "session.updated"
        | "response.created"
        | "response.done"
        | "response.function_call_arguments.done"
        | "error" => {
            info!(call_id, event_type = et, "{direction}");
        }
        // Everything else → debug
        _ => {
            debug!(call_id, event_type = et, "{direction}");
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream SDP exchange
// ---------------------------------------------------------------------------

/// Send SMG's SDP offer to upstream (OpenAI) and parse the returned SDP
/// answer.  Supports multipart (with session config) and direct SDP.
async fn send_sdp_to_upstream(
    client: &reqwest::Client,
    upstream_url: &str,
    auth_header: &str,
    sdp_offer: &str,
    session_config: Option<serde_json::Value>,
) -> Result<SdpAnswer, BridgeSetupError> {
    let req = client
        .post(upstream_url)
        .header("Authorization", auth_header);

    let resp = if let Some(session) = session_config {
        let form = reqwest::multipart::Form::new()
            .part(
                "sdp",
                reqwest::multipart::Part::text(sdp_offer.to_string())
                    .mime_str("application/sdp")?,
            )
            .part(
                "session",
                reqwest::multipart::Part::text(session.to_string()).mime_str("application/json")?,
            );
        req.multipart(form).send().await?
    } else {
        req.header("Content-Type", "application/sdp")
            .body(sdp_offer.to_string())
            .send()
            .await?
    };

    let status = resp.status();
    if !status.is_success() {
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        let body = resp.text().await.unwrap_or_default();
        return Err(BridgeSetupError::UpstreamHttp {
            status,
            body,
            content_type,
        });
    }

    let answer_text = resp.text().await?;
    Ok(SdpAnswer::from_sdp_string(&answer_text)?)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pick the earliest `Instant` from two optional str0m timeouts.
/// Falls back to `now + 50ms` if neither peer has a scheduled timeout.
fn earliest_timeout(a: Option<Instant>, b: Option<Instant>) -> Instant {
    match (a, b) {
        (Some(a), Some(b)) => a.min(b),
        (Some(t), None) | (None, Some(t)) => t,
        (None, None) => Instant::now() + Duration::from_millis(50),
    }
}

/// Resolve the effective IP for ICE candidates.
///
/// When `addr` is unspecified (`0.0.0.0` / `::`), performs a non-sending UDP
/// "connect" to a public address to let the OS routing table pick the default
/// outbound interface.  No traffic is sent.
async fn resolve_candidate_ip(addr: IpAddr) -> anyhow::Result<IpAddr> {
    if !addr.is_unspecified() {
        return Ok(addr);
    }
    let sock = UdpSocket::bind("0.0.0.0:0").await?;
    sock.connect("8.8.8.8:80").await?;
    Ok(sock.local_addr()?.ip())
}

/// Find the first audio Mid in an Rtc instance (after SDP negotiation).
///
/// str0m's `Rtc` only exposes `media(mid)` for individual lookup — there is no
/// public iterator over all media sections.  SDP mid values are string
/// representations of small integers ("0", "1", …).  OpenAI realtime sessions
/// typically have ≤3 media lines (audio + data channel), so probing 0..16
/// provides generous headroom while keeping the search bounded.
fn find_audio_mid(rtc: &Rtc) -> Option<Mid> {
    (0..16u32).find_map(|i| {
        let mid = Mid::from(i.to_string().as_str());
        rtc.media(mid)
            .filter(|m| m.kind() == MediaKind::Audio)
            .map(|_| mid)
    })
}

// ---------------------------------------------------------------------------
// STUN binding — minimal client for server-reflexive candidate gathering
// ---------------------------------------------------------------------------

/// Perform a STUN Binding Request (RFC 5389) on `socket` to discover its
/// server-reflexive (public) address.  Returns `None` on timeout or parse
/// failure — callers should proceed with host-only candidates in that case.
async fn stun_gather_srflx(socket: &UdpSocket, stun_server: SocketAddr) -> Option<SocketAddr> {
    // 20-byte STUN Binding Request
    let mut req = [0u8; 20];
    req[0..2].copy_from_slice(&0x0001u16.to_be_bytes()); // Binding Request
                                                         // Length = 0 (no attributes)
    req[4..8].copy_from_slice(&0x2112_A442u32.to_be_bytes()); // Magic Cookie
                                                              // STUN transaction ID is 12 bytes (RFC 5389); truncate the 16-byte UUID.
    let txn = uuid::Uuid::now_v7();
    req[8..20].copy_from_slice(&txn.as_bytes()[..12]);

    if let Err(e) = socket.send_to(&req, stun_server).await {
        warn!(%stun_server, error = %e, "STUN send failed");
        return None;
    }
    debug!(%stun_server, "STUN Binding Request sent");

    let mut buf = [0u8; 512];
    let (n, from) =
        match tokio::time::timeout(Duration::from_secs(3), socket.recv_from(&mut buf)).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => {
                warn!(error = %e, "STUN recv error");
                return None;
            }
            Err(_) => {
                warn!(%stun_server, "STUN response timed out (3s)");
                return None;
            }
        };

    let addr = parse_stun_xor_mapped_address(&buf[..n], &req[8..20]);
    if addr.is_some() {
        info!(%from, srflx = ?addr, "STUN srflx discovered");
    } else {
        warn!(%from, n, "STUN response unparsable");
    }
    addr
}

/// Parse the XOR-MAPPED-ADDRESS attribute from a STUN Binding Success
/// Response.  Returns the decoded `SocketAddr` or `None`.
fn parse_stun_xor_mapped_address(resp: &[u8], txn_id: &[u8]) -> Option<SocketAddr> {
    if resp.len() < 20 || resp[0] != 0x01 || resp[1] != 0x01 {
        return None;
    }
    if &resp[8..20] != txn_id {
        return None;
    }

    let msg_len = u16::from_be_bytes([resp[2], resp[3]]) as usize;
    let end = (20 + msg_len).min(resp.len());
    let mut off = 20;

    while off + 4 <= end {
        let attr_type = u16::from_be_bytes([resp[off], resp[off + 1]]);
        let attr_len = u16::from_be_bytes([resp[off + 2], resp[off + 3]]) as usize;
        off += 4;
        if off + attr_len > end {
            break;
        }

        // XOR-MAPPED-ADDRESS = 0x0020 (RFC 5389 §15.2)
        // Port is XORed with top 16 bits of magic cookie (0x2112).
        // IPv4 address is XORed with the full magic cookie (0x2112_A442).
        if attr_type == 0x0020 && attr_len >= 8 && resp[off + 1] == 0x01 {
            let port = u16::from_be_bytes([resp[off + 2], resp[off + 3]]) ^ 0x2112;
            let ip = std::net::Ipv4Addr::new(
                resp[off + 4] ^ 0x21,
                resp[off + 5] ^ 0x12,
                resp[off + 6] ^ 0xA4,
                resp[off + 7] ^ 0x42,
            );
            return Some(SocketAddr::new(IpAddr::V4(ip), port));
        }

        // Attributes are padded to 4-byte boundaries
        off += (attr_len + 3) & !3;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

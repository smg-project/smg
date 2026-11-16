//! The dials of the membership protocol: one gRPC channel per peer address,
//! shared by the probes, the relayed probes and the status broadcasts.
//!
//! Before, every one of them built an endpoint, connected and let the
//! connection go: a TCP handshake per probe, and under mTLS a TLS handshake
//! and four reads of the key material (the CA twice, the certificate, the
//! key) besides a client configuration built and dropped on the spot; a
//! member paid that once a second for the life of the process, more under
//! suspicion and at every status change. A [`Channel`] connects on its first
//! request and reconnects by itself after a failure, so one per peer address
//! carries every dial to that peer, and a peer that restarts on its address
//! answers again through the same channel. The sync stream to a peer runs
//! on the same channel: a stream is one long request, and the request
//! deadline bounds the wait for its response head only, not the frames
//! after it, so a pair of members holds one connection per direction. The
//! stream and the probes then share the connection's flow-control window
//! on the accepting side, which is why the listener gives a connection more
//! credit than one stream may hold (see the service's server builder).

use std::{collections::HashMap, net::SocketAddr, str::FromStr, sync::Arc, time::Duration};

use parking_lot::Mutex;
use tonic::{
    transport::{Channel, Endpoint},
    Request,
};

use crate::{
    mtls::{tls_server_name, MTLSManager},
    service::gossip::{
        gossip_client::GossipClient, gossip_message, GossipMessage, NodeState, NodeUpdate,
    },
    transport::limits::MAX_MESSAGE_SIZE,
};

/// How long a dial waits for its connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a dial waits for the peer's answer once its connection is up
/// (the connect has [`CONNECT_TIMEOUT`] of its own; the wait in the
/// channel's queue behind other dials to the same address is unbounded).
const RPC_TIMEOUT: Duration = Duration::from_secs(10);
/// How often a kept channel pings its peer over HTTP/2, idle or not, and
/// how long it waits for the pong: a peer gone without closing its socket
/// (a dead host, a route that stopped) is found within two of these, and
/// the channel redials on the next request instead of waiting out
/// [`RPC_TIMEOUT`] on a half-open connection until the kernel gives up.
const KEEP_ALIVE: Duration = Duration::from_secs(5);

/// The node's dials: its TLS identity when the mesh has one, and the channel
/// it keeps per peer address.
pub(crate) struct Dialer {
    mtls_manager: Option<Arc<MTLSManager>>,
    /// A channel reconnects by itself, so an entry lives as long as its
    /// address is a member's; [`Self::retain`] drops the rest.
    channels: Mutex<HashMap<SocketAddr, Channel>>,
}

impl std::fmt::Debug for Dialer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dialer")
            .field("mtls", &self.mtls_manager.is_some())
            .field("channels", &self.channels.lock().len())
            .finish()
    }
}

impl Dialer {
    /// A dialer presenting the identity `mtls_manager` holds, or dialing in
    /// the clear without one.
    pub(crate) fn new(mtls_manager: Option<Arc<MTLSManager>>) -> Arc<Self> {
        Arc::new(Self {
            mtls_manager,
            channels: Mutex::new(HashMap::new()),
        })
    }

    /// The node's TLS identity, for the listener's server configuration.
    pub(crate) fn mtls_manager(&self) -> Option<&Arc<MTLSManager>> {
        self.mtls_manager.as_ref()
    }

    /// A `PingServer` call to `peer_node` carrying `payload` (a probe, a
    /// relayed probe or a status broadcast), and the peer's answer.
    pub(crate) async fn ping(
        &self,
        peer_node: &NodeState,
        payload: Option<gossip_message::Payload>,
    ) -> Result<NodeUpdate, tonic::Status> {
        let peer_addr = SocketAddr::from_str(&peer_node.address).map_err(|e| {
            tonic::Status::invalid_argument(format!(
                "Invalid address for node {}: {}, {}",
                peer_node.name, peer_node.address, e
            ))
        })?;
        let channel = self.channel(&peer_node.name, peer_addr).await?;
        let mut client = GossipClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);
        let response = client
            .ping_server(Request::new(GossipMessage { payload }))
            .await?;
        Ok(response.into_inner())
    }

    /// The channel to `addr`: the one kept, or a new one that connects on
    /// its first request. Two dials racing for a new address keep the first
    /// channel made, so an address never has two. The dials in this module
    /// and the sync stream to the peer at `addr` share it.
    pub(crate) async fn channel(
        &self,
        peer_name: &str,
        addr: SocketAddr,
    ) -> Result<Channel, tonic::Status> {
        if let Some(channel) = self.channels.lock().get(&addr) {
            return Ok(channel.clone());
        }
        let channel = self.endpoint(peer_name, addr).await?.connect_lazy();
        Ok(self.channels.lock().entry(addr).or_insert(channel).clone())
    }

    /// The endpoint for `addr`: the connect and request deadlines, and under
    /// mTLS the node's identity with the peer's address as the server name
    /// (`peer_name` when the URL has no host, which a socket address always
    /// has).
    async fn endpoint(&self, peer_name: &str, addr: SocketAddr) -> Result<Endpoint, tonic::Status> {
        let scheme = if self.mtls_manager.is_some() {
            "https"
        } else {
            "http"
        };
        let connect_url = format!("{scheme}://{addr}");
        let mut endpoint = Endpoint::from_shared(connect_url.clone())
            .map_err(|e| {
                tonic::Status::invalid_argument(format!(
                    "Invalid endpoint for node {peer_name}: {connect_url}, {e}"
                ))
            })?
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(RPC_TIMEOUT)
            .http2_keep_alive_interval(KEEP_ALIVE)
            .keep_alive_timeout(KEEP_ALIVE)
            .keep_alive_while_idle(true);
        if let Some(mtls_manager) = &self.mtls_manager {
            let server_name = endpoint.uri().host().map_or_else(
                || peer_name.to_owned(),
                |host| tls_server_name(host).to_owned(),
            );
            let tls = mtls_manager
                .client_tls_config(&server_name)
                .await
                .map_err(|e| {
                    tonic::Status::unavailable(format!(
                        "Failed to load the mTLS identity for {peer_name}: {e}"
                    ))
                })?;
            endpoint = endpoint.tls_config(tls).map_err(|e| {
                tonic::Status::unavailable(format!(
                    "Failed to configure TLS endpoint for {peer_name}: {e}"
                ))
            })?;
        }
        Ok(endpoint)
    }

    /// Drops the channels of every address `keep` refuses: the members that
    /// left the table.
    pub(crate) fn retain(&self, keep: impl Fn(&SocketAddr) -> bool) {
        self.channels.lock().retain(|addr, _| keep(addr));
    }

    /// How many peer addresses have a channel.
    #[cfg(test)]
    pub(crate) fn channel_count(&self) -> usize {
        self.channels.lock().len()
    }
}

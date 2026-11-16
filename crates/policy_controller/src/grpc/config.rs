//! What a gRPC plugin row resolves to before the host connects to it. The
//! spec's `plugins` section (a row naming an address beside a module's)
//! deserializes into these; until that row lands a host builds them
//! directly. The parts a row has whatever its kind are the shared ones of
//! [`crate::plugins::config`], re-exported here.

use std::path::PathBuf;

pub use crate::plugins::config::{BreakerSpec, Exposure, Failure, FrameBudget, HeaderSet, Reveal};
use crate::spec::Needs;

/// The client side of mutual TLS: the authority that signed the plugin's
/// certificate, the identity the plugin verifies, and the name pinned on the
/// plugin's certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tls {
    /// The server name the plugin's certificate must carry.
    pub server_name: String,
    /// The authority's certificate, PEM.
    pub ca_pem: Vec<u8>,
    /// This host's certificate chain and private key, PEM.
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
}

/// Where the plugin answers. The design names two transports and no plain
/// TCP: a socket on the host, whose peer is checked by its credentials, or
/// TCP under mutual TLS with the server name pinned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transport {
    /// A Unix socket. At connect the peer's uid must be this process's own,
    /// or `peer_uid` when the plugin runs as another user by design.
    Unix {
        path: PathBuf,
        peer_uid: Option<u32>,
    },
    /// `host:port` under mutual TLS.
    Tls { address: String, tls: Tls },
}

/// One gRPC plugin as the host connects to it.
#[derive(Clone, Debug)]
pub struct GrpcPluginSpec {
    /// The name the trace and the shed carry as the stage.
    pub name: String,
    pub transport: Transport,
    /// Passed in the hello, opaque to the host.
    pub config: Vec<u8>,
    /// The plugin's own budget per decision, in microseconds; `0` takes the
    /// decision's remaining budget. The host clamps it to at most the
    /// decision's; the frame names it as the plugin's deadline. A missed
    /// deadline of a budget no longer than the I/O thread's promise (four
    /// milliseconds) is never counted as the plugin's failure: the latency
    /// rule alone judges it.
    pub budget_micros: u32,
    /// The connect and the hello run under this: at the apply, and on the
    /// I/O thread behind a cold thread's first decision, which takes its
    /// fallbacks and never waits for them.
    pub connect_micros: u32,
    pub view: Exposure,
    /// What the host must feed for the plugin's stages to mean anything;
    /// the plugin declares it, the host cannot read it off a server.
    pub needs: Needs,
    /// The counters the plugin may add to, by id; a name per id.
    pub metrics: Vec<String>,
    pub failure: Failure,
    pub breaker: BreakerSpec,
    /// The most candidates one frame carries. A larger pool is never
    /// truncated: the plugin is not asked, the stages fall back and the
    /// decision counts `too_many`.
    pub max_candidates: u32,
}

impl GrpcPluginSpec {
    /// The default candidate bound: the record view is 40 KB at 512.
    pub const DEFAULT_MAX_CANDIDATES: u32 = 128;

    /// The default connect frame: the WebAssembly host's `init` frame.
    pub const DEFAULT_CONNECT_MICROS: u32 = 100_000;

    /// A plugin of `name` behind `transport` with every default.
    pub fn new(name: impl Into<String>, transport: Transport) -> Self {
        Self {
            name: name.into(),
            transport,
            config: Vec::new(),
            budget_micros: 0,
            connect_micros: Self::DEFAULT_CONNECT_MICROS,
            view: Exposure::default(),
            needs: Needs::default(),
            metrics: Vec::new(),
            failure: Failure::default(),
            breaker: BreakerSpec::default(),
            max_candidates: Self::DEFAULT_MAX_CANDIDATES,
        }
    }

    /// The plugin's frame budget: the owner of the clamp to the decision's
    /// remainder. A gRPC plugin burns no fuel; the calibration is one, so
    /// the frame reads in microseconds.
    pub fn frame(&self) -> FrameBudget {
        FrameBudget {
            micros: self.budget_micros,
            fuel_per_micro: 1,
        }
    }
}

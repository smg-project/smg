//! One parser and one canonical key for every worker address in the gateway.
//!
//! Worker addresses arrive in several spellings for the same backend: bare
//! `host:port` from discovery, `http://`/`grpc://` forms from configuration
//! and the admin API, `ipc://` paths for ZMQ, and a `@rank` suffix that the
//! registration workflow appends when it expands a data-parallel worker. Before
//! this module each consumer stripped what it cared about with its own ad-hoc
//! string handling, and those implementations disagreed — notably on `@` inside
//! an IPC path or an address such as `http://user@worker:3000`, and on
//! uppercase schemes.
//!
//! [`Endpoint`] parses a spelling once; [`EndpointKey`] answers "is this the
//! same backend?" for snapshot uniqueness, registry lookup, ownership grouping,
//! and removal. The key is scheme- and rank-stripped, so `http://h:p`,
//! `grpc://h:p`, `h:p`, and `h:p@3` all share one key — which is what lets the
//! reconciler notice a stale-scheme sibling occupying an address it wants.
//!
//! What the key deliberately does *not* normalize: a missing port is not filled
//! in from the scheme's default, so `http://h` and `h:80` stay distinct. The
//! gateway never dials a worker on an implied port, and guessing one would make
//! two genuinely different registrations collide.

use std::{
    fmt,
    net::{IpAddr, Ipv6Addr},
};

/// Transports a worker endpoint can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scheme {
    Http,
    Https,
    Grpc,
    Grpcs,
    Ipc,
}

impl Scheme {
    /// The prefix as written, including `://`.
    pub fn as_prefix(self) -> &'static str {
        match self {
            Scheme::Http => "http://",
            Scheme::Https => "https://",
            Scheme::Grpc => "grpc://",
            Scheme::Grpcs => "grpcs://",
            Scheme::Ipc => "ipc://",
        }
    }

    /// Match a scheme prefix case-insensitively, returning it and the rest.
    fn split_from(input: &str) -> Option<(Scheme, &str)> {
        const SCHEMES: [Scheme; 5] = [
            Scheme::Http,
            Scheme::Https,
            Scheme::Grpc,
            Scheme::Grpcs,
            Scheme::Ipc,
        ];
        SCHEMES.into_iter().find_map(|scheme| {
            let prefix = scheme.as_prefix();
            let head = input.get(..prefix.len())?;
            head.eq_ignore_ascii_case(prefix)
                .then(|| (scheme, &input[prefix.len()..]))
        })
    }

    /// Whether this scheme carries a socket path rather than a host and port.
    fn is_ipc(self) -> bool {
        matches!(self, Scheme::Ipc)
    }
}

/// The address half of an endpoint.
///
/// `ipc://` lives here rather than in a separate `ipc_path` field so that an
/// IPC endpoint cannot also carry a port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Host {
    /// An IPv4 or IPv6 literal. IPv6 is stored parsed, so `[::1]` and
    /// `[0:0:0:0:0:0:0:1]` compare and render identically.
    Ip(IpAddr),
    /// A DNS name, lowercased (DNS is case-insensitive).
    Name(String),
    /// A filesystem socket path from an `ipc://` endpoint.
    IpcPath(String),
}

/// Why a spelling could not be parsed as a worker endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointError {
    /// Nothing, or only a scheme.
    Empty,
    /// A `@rank` suffix where a base endpoint was required.
    UnexpectedRank { rank: usize },
    /// A bracketed host that is not a valid IPv6 literal, or an unclosed one.
    MalformedIpv6 { host: String },
    /// A port that is not a number in `1..=65535`.
    InvalidPort { port: String },
    /// A host containing a character that cannot appear in one.
    InvalidHost { host: String },
}

impl fmt::Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EndpointError::Empty => write!(f, "endpoint is empty"),
            EndpointError::UnexpectedRank { rank } => write!(
                f,
                "endpoint carries a DP rank suffix '@{rank}'; a base endpoint was expected"
            ),
            EndpointError::MalformedIpv6 { host } => {
                write!(f, "malformed IPv6 host '{host}'")
            }
            EndpointError::InvalidPort { port } => write!(f, "invalid port '{port}'"),
            EndpointError::InvalidHost { host } => write!(f, "invalid host '{host}'"),
        }
    }
}

impl std::error::Error for EndpointError {}

/// A worker address, parsed once.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Endpoint {
    scheme: Option<Scheme>,
    /// Anything before the authority's final `@`. `validate_worker_url`
    /// accepts `http://user:pass@worker:3000`, so it has to survive into the
    /// key rather than being rejected or folded into the host, where its
    /// second colon would look like an unbracketed IPv6 literal.
    userinfo: Option<String>,
    host: Host,
    port: Option<u16>,
    /// Path prefix, when the address carries one. `validate_worker_url`
    /// accepts `http://proxy:8080/sglang`, and `endpoint_url` appends the
    /// route to it, so the prefix is part of which backend this names.
    /// Stored with its leading `/` and without a trailing one.
    path: Option<String>,
}

/// Canonical identity of a backend: scheme- and rank-stripped.
///
/// Two endpoints share a key exactly when they name the same backend, however
/// they were spelled.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EndpointKey(String);

impl fmt::Debug for EndpointKey {
    /// A key that carries userinfo would print a password, and the bare
    /// string cannot say which half is which. A key holding an `@` is
    /// therefore masked whole; one without is printed, because that is the
    /// common case and the one worth reading in a failing assertion.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.contains('@') {
            f.write_str("EndpointKey(<redacted>)")
        } else {
            f.debug_tuple("EndpointKey").field(&self.0).finish()
        }
    }
}

impl EndpointKey {
    /// The key as stored.
    ///
    /// This is a lookup and comparison value, and it retains any userinfo the
    /// address carried, so it can name credentials. It is deliberately not
    /// `Display`: use [`Endpoint::redacted`] for anything a person reads.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Endpoint {
    /// Parse a base endpoint, rejecting a `@rank` suffix.
    ///
    /// Provider records and configuration name backends, not the per-rank
    /// workers the registration workflow derives from them, so a rank here is
    /// a mistake worth surfacing rather than silently dropping.
    pub fn parse(input: &str) -> Result<Self, EndpointError> {
        match Self::parse_with_rank(input)? {
            (_, Some(rank)) => Err(EndpointError::UnexpectedRank { rank }),
            (endpoint, None) => Ok(endpoint),
        }
    }

    /// Parse an endpoint that may carry the workflow's `@rank` suffix.
    ///
    /// A suffix counts as a rank only when every character after the final `@`
    /// is a digit, so an IPC path is left intact and an authority's own `@`
    /// stays put — it is split off as userinfo further down, not treated as a
    /// rank marker.
    pub fn parse_with_rank(input: &str) -> Result<(Self, Option<usize>), EndpointError> {
        let trimmed = input.trim();
        let (body, rank) = split_rank(trimmed);
        let (scheme, rest) = match Scheme::split_from(body) {
            Some((scheme, rest)) => (Some(scheme), rest),
            None => (None, body),
        };
        if rest.is_empty() {
            return Err(EndpointError::Empty);
        }

        if scheme.is_some_and(Scheme::is_ipc) {
            return Ok((
                Endpoint {
                    scheme,
                    userinfo: None,
                    host: Host::IpcPath(rest.to_string()),
                    port: None,
                    path: None,
                },
                rank,
            ));
        }

        let (authority, path) = split_path(rest);
        if authority.is_empty() {
            return Err(EndpointError::Empty);
        }
        let (userinfo, hostport) = match authority.rsplit_once('@') {
            Some((userinfo, hostport)) => (Some(userinfo.to_string()), hostport),
            None => (None, authority),
        };
        if hostport.is_empty() {
            return Err(EndpointError::Empty);
        }
        let (host, port) = split_host_port(hostport)?;
        Ok((
            Endpoint {
                scheme,
                userinfo,
                host,
                port,
                path,
            },
            rank,
        ))
    }

    pub fn scheme(&self) -> Option<Scheme> {
        self.scheme
    }

    pub fn host(&self) -> &Host {
        &self.host
    }

    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// Path prefix, if the address carries one.
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Whether this endpoint names an `ipc://` socket.
    pub fn is_ipc(&self) -> bool {
        matches!(self.host, Host::IpcPath(_))
    }

    /// The canonical identity: no scheme, no rank, IPv6 in its shortest form.
    ///
    /// `ipc://` is stripped like every other scheme, so a scheme-less query
    /// selects an IPC registration just as it selects an HTTP one. A socket
    /// path is a filesystem path, so in practice it cannot collide with a
    /// `host:port`; a contrived path shaped like one would.
    pub fn key(&self) -> EndpointKey {
        let path = self.path.as_deref().unwrap_or("");
        let user = self
            .userinfo
            .as_deref()
            .map(|u| format!("{u}@"))
            .unwrap_or_default();
        EndpointKey(match (&self.host, self.port) {
            (Host::IpcPath(socket), _) => socket.clone(),
            (host, Some(port)) => format!("{user}{}:{port}{path}", render_host(host)),
            (host, None) => format!("{user}{}{path}", render_host(host)),
        })
    }

    /// The form to submit when something has to look this endpoint up again
    /// by string, as a removal job does.
    ///
    /// Identical to [`Self::key`] for a network address. An IPC endpoint keeps
    /// its `ipc://` prefix, because its key is a bare socket path and a bare
    /// path is not a parseable address — feeding the key back would select
    /// nothing.
    pub fn lookup_form(&self) -> String {
        match &self.host {
            Host::IpcPath(socket) => format!("ipc://{socket}"),
            _ => self.key().0,
        }
    }

    /// Render the endpoint as written, preserving the scheme and bracketing
    /// IPv6. Round-trips through [`Self::parse`].
    pub fn render(&self) -> String {
        let prefix = self.scheme.map(Scheme::as_prefix).unwrap_or("");
        let path = self.path.as_deref().unwrap_or("");
        let user = self
            .userinfo
            .as_deref()
            .map(|u| format!("{u}@"))
            .unwrap_or_default();
        match (&self.host, self.port) {
            (Host::IpcPath(socket), _) => format!("ipc://{socket}"),
            (host, Some(port)) => format!("{prefix}{user}{}:{port}{path}", render_host(host)),
            (host, None) => format!("{prefix}{user}{}{path}", render_host(host)),
        }
    }

    /// A form safe to log: the secret half of any userinfo is masked.
    ///
    /// The username is kept, because it distinguishes two registrations at one
    /// address and is what makes a log line useful.
    pub fn redacted(&self) -> String {
        let Some(userinfo) = self.userinfo.as_deref() else {
            return self.render();
        };
        let masked = match userinfo.split_once(':') {
            Some((user, _)) => format!("{user}:***"),
            None => userinfo.to_string(),
        };
        let exact = self.render();
        let Some((prefix, rest)) = exact.split_once(&format!("{userinfo}@")) else {
            // Unreachable while `render` emits the userinfo it was built from;
            // masking everything is the safe answer if that ever changes.
            return exact.replace(userinfo, "***");
        };
        format!("{prefix}{masked}@{rest}")
    }

    /// Render this endpoint with the workflow's rank suffix.
    pub fn render_with_rank(&self, rank: usize) -> String {
        format!("{}@{rank}", self.render())
    }
}

impl fmt::Debug for Endpoint {
    /// Masked like [`Self::redacted`]. The derive would have printed
    /// `userinfo: Some("user:pass")`, and `?endpoint` is the usual idiom when
    /// debugging the reconciler.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Endpoint").field(&self.redacted()).finish()
    }
}

impl fmt::Display for Endpoint {
    /// Redacted, so an endpoint cannot leak a password into a log by being
    /// formatted. [`Endpoint::render`] gives the exact address.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.redacted())
    }
}

/// Canonical key for a spelling, without building an [`Endpoint`].
///
/// Convenience for lookup paths that only need identity.
pub fn endpoint_key(input: &str) -> Result<EndpointKey, EndpointError> {
    Endpoint::parse_with_rank(input).map(|(endpoint, _)| endpoint.key())
}

fn render_host(host: &Host) -> String {
    match host {
        Host::Ip(IpAddr::V6(addr)) => format!("[{addr}]"),
        Host::Ip(IpAddr::V4(addr)) => addr.to_string(),
        Host::Name(name) => name.clone(),
        Host::IpcPath(path) => path.clone(),
    }
}

/// Split a trailing `@<digits>` rank suffix, if there is one.
fn split_rank(input: &str) -> (&str, Option<usize>) {
    let Some(at) = input.rfind('@') else {
        return (input, None);
    };
    let suffix = &input[at + 1..];
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return (input, None);
    }
    match suffix.parse::<usize>() {
        Ok(rank) => (&input[..at], Some(rank)),
        // A run of digits too long for usize is not a rank anyone assigned.
        Err(_) => (input, None),
    }
}

/// Split an authority from its path prefix.
///
/// The search starts after a bracketed IPv6 host so the brackets' own colons
/// and the authority are never mistaken for a path.
fn split_path(rest: &str) -> (&str, Option<String>) {
    // Only a bracketed host has brackets to skip. Searching for `]`
    // unconditionally let one inside a path hide the `/` that preceded it.
    let after_brackets = if rest.starts_with('[') {
        rest.find(']').map_or(0, |close| close + 1)
    } else {
        0
    };
    let Some(offset) = rest[after_brackets..].find('/') else {
        return (rest, None);
    };
    let at = after_brackets + offset;
    let path = rest[at..].trim_end_matches('/');
    if path.is_empty() {
        // A bare trailing slash names the same backend; `endpoint_url` appends
        // a route that already starts with one.
        return (&rest[..at], None);
    }
    (&rest[..at], Some(path.to_string()))
}

fn split_host_port(rest: &str) -> Result<(Host, Option<u16>), EndpointError> {
    if let Some(close) = rest.find(']') {
        // Bracketed IPv6, optionally followed by `:port`.
        if !rest.starts_with('[') {
            return Err(EndpointError::MalformedIpv6 {
                host: rest.to_string(),
            });
        }
        let literal = &rest[1..close];
        let addr: Ipv6Addr = literal.parse().map_err(|_| EndpointError::MalformedIpv6 {
            host: literal.to_string(),
        })?;
        let tail = &rest[close + 1..];
        let port = match tail.strip_prefix(':') {
            Some(port) => Some(parse_port(port)?),
            None if tail.is_empty() => None,
            None => {
                return Err(EndpointError::MalformedIpv6 {
                    host: rest.to_string(),
                })
            }
        };
        return Ok((Host::Ip(IpAddr::V6(addr)), port));
    }

    if rest.starts_with('[') {
        return Err(EndpointError::MalformedIpv6 {
            host: rest.to_string(),
        });
    }

    // An unbracketed IPv6 literal has more than one colon; treat it as a host
    // with no port rather than mis-splitting it.
    if rest.matches(':').count() > 1 {
        let addr: Ipv6Addr = rest.parse().map_err(|_| EndpointError::MalformedIpv6 {
            host: rest.to_string(),
        })?;
        return Ok((Host::Ip(IpAddr::V6(addr)), None));
    }

    let (host, port) = match rest.split_once(':') {
        Some((host, port)) => (host, Some(parse_port(port)?)),
        None => (rest, None),
    };
    Ok((parse_host(host)?, port))
}

fn parse_port(port: &str) -> Result<u16, EndpointError> {
    match port.parse::<u16>() {
        Ok(0) | Err(_) => Err(EndpointError::InvalidPort {
            port: port.to_string(),
        }),
        Ok(port) => Ok(port),
    }
}

fn parse_host(host: &str) -> Result<Host, EndpointError> {
    if host.is_empty() {
        return Err(EndpointError::Empty);
    }
    if let Ok(addr) = host.parse::<IpAddr>() {
        return Ok(Host::Ip(addr));
    }
    // Conservative: a DNS name, and nothing that would change how the key is
    // read back (no path, query, or whitespace).
    // Userinfo was split off at the authority's final `@` before this point,
    // so a host reaching here can no longer contain one — `@` is rejected to
    // hold that invariant here rather than leaving it implied by the caller.
    if host.bytes().any(|b| {
        b.is_ascii_whitespace() || matches!(b, b'/' | b'?' | b'#' | b'\\' | b'[' | b']' | b'@')
    }) {
        return Err(EndpointError::InvalidHost {
            host: host.to_string(),
        });
    }
    Ok(Host::Name(host.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_of(input: &str) -> String {
        endpoint_key(input).expect(input).as_str().to_string()
    }

    #[test]
    fn every_scheme_spelling_of_one_backend_shares_a_key() {
        for spelling in [
            "10.0.0.1:8080",
            "http://10.0.0.1:8080",
            "https://10.0.0.1:8080",
            "grpc://10.0.0.1:8080",
            "grpcs://10.0.0.1:8080",
            "HTTP://10.0.0.1:8080",
            "GrPc://10.0.0.1:8080",
            "10.0.0.1:8080@0",
            "http://10.0.0.1:8080@7",
        ] {
            assert_eq!(key_of(spelling), "10.0.0.1:8080", "{spelling}");
        }
    }

    #[test]
    fn ipv6_spellings_normalize_to_one_key() {
        for spelling in [
            "[::1]:8080",
            "[0:0:0:0:0:0:0:1]:8080",
            "http://[::1]:8080",
            "grpc://[0000:0000:0000:0000:0000:0000:0000:0001]:8080@2",
        ] {
            assert_eq!(key_of(spelling), "[::1]:8080", "{spelling}");
        }
    }

    #[test]
    fn dns_names_are_case_folded() {
        assert_eq!(key_of("Worker-1.NS.svc:8080"), "worker-1.ns.svc:8080");
    }

    #[test]
    fn ipc_paths_are_scheme_stripped_and_survive_at_signs() {
        assert_eq!(key_of("ipc:///tmp/w.sock"), "/tmp/w.sock");
        // The old `split('@').next()` truncated this to `/tmp/a`.
        assert_eq!(key_of("ipc:///tmp/a@b.sock"), "/tmp/a@b.sock");
        // A scheme-less query reaches an IPC registration, as it always has.
        assert_eq!(key_of("ipc://worker:3000"), key_of("worker:3000"));
    }

    #[test]
    fn ipc_rank_suffix_is_a_rank_not_part_of_the_path() {
        let (endpoint, rank) = Endpoint::parse_with_rank("ipc:///tmp/w.sock@3").unwrap();
        assert_eq!(rank, Some(3));
        assert_eq!(endpoint.key().as_str(), "/tmp/w.sock");
        assert_eq!(endpoint.render(), "ipc:///tmp/w.sock");
    }

    #[test]
    fn rank_is_split_only_when_the_suffix_is_all_digits() {
        assert_eq!(
            Endpoint::parse_with_rank("10.0.0.1:8080@2").unwrap().1,
            Some(2)
        );
        // Not a rank, so the `@` separates userinfo from a host, which is how
        // `url::Url` reads it too. Either way the key is not truncated.
        let (endpoint, rank) = Endpoint::parse_with_rank("10.0.0.1:8080@abc").unwrap();
        assert_eq!(rank, None);
        assert_eq!(endpoint.key().as_str(), "10.0.0.1:8080@abc");
    }

    /// The registry holds workers registered as `http://user@worker:3000`, so
    /// an `@` in the address belongs to the address. `canonical_host_port`
    /// split on the first one and returned `user`.
    #[test]
    fn an_at_sign_in_the_address_is_kept_not_truncated() {
        assert_eq!(key_of("http://user@worker:3000"), "user@worker:3000");
        // Credentials carry a second colon, which read as an unbracketed IPv6
        // literal before userinfo became its own component. `url::Url` accepts
        // this shape, so `validate_worker_url` lets it register.
        assert_eq!(
            key_of("http://user:pass@worker:3000"),
            "user:pass@worker:3000"
        );
        assert_eq!(
            key_of("grpc://user:pass@worker:3000@1"),
            "user:pass@worker:3000"
        );
        let (endpoint, rank) = Endpoint::parse_with_rank("http://user@worker:3000@0").unwrap();
        assert_eq!(rank, Some(0));
        assert_eq!(endpoint.key().as_str(), "user@worker:3000");
        // A prefix of the address is not the same backend.
        assert_ne!(key_of("http://user"), key_of("http://user@worker:3000"));
    }

    #[test]
    fn parse_rejects_a_rank_but_parse_with_rank_accepts_it() {
        assert_eq!(
            Endpoint::parse("10.0.0.1:8080@2").unwrap_err(),
            EndpointError::UnexpectedRank { rank: 2 }
        );
        assert!(Endpoint::parse_with_rank("10.0.0.1:8080@2").is_ok());
    }

    #[test]
    fn render_round_trips_through_parse() {
        for spelling in [
            "10.0.0.1:8080",
            "http://10.0.0.1:8080",
            "grpcs://worker.ns.svc:9000",
            "[::1]:8080",
            "http://[fd00::1]:8080",
            "ipc:///tmp/w.sock",
            "worker",
        ] {
            let parsed = Endpoint::parse(spelling).expect(spelling);
            let rendered = parsed.render();
            let reparsed = Endpoint::parse(&rendered).expect(&rendered);
            assert_eq!(parsed, reparsed, "{spelling} -> {rendered}");
        }
    }

    #[test]
    fn render_brackets_ipv6_so_the_port_stays_readable() {
        let endpoint = Endpoint::parse("http://[fd00::1]:8080").unwrap();
        assert_eq!(endpoint.render(), "http://[fd00::1]:8080");
        assert_eq!(endpoint.render_with_rank(2), "http://[fd00::1]:8080@2");
    }

    #[test]
    fn a_missing_port_is_not_filled_in_from_the_scheme() {
        assert_ne!(key_of("http://worker"), key_of("worker:80"));
        assert_eq!(key_of("http://worker"), "worker");
    }

    #[test]
    fn malformed_input_is_rejected() {
        assert_eq!(Endpoint::parse("").unwrap_err(), EndpointError::Empty);
        assert_eq!(
            Endpoint::parse("http://").unwrap_err(),
            EndpointError::Empty
        );
        assert!(matches!(
            Endpoint::parse("10.0.0.1:0").unwrap_err(),
            EndpointError::InvalidPort { .. }
        ));
        assert!(matches!(
            Endpoint::parse("10.0.0.1:99999").unwrap_err(),
            EndpointError::InvalidPort { .. }
        ));
        assert!(matches!(
            Endpoint::parse("[::1:8080").unwrap_err(),
            EndpointError::MalformedIpv6 { .. }
        ));
        assert!(matches!(
            Endpoint::parse("[not-v6]:8080").unwrap_err(),
            EndpointError::MalformedIpv6 { .. }
        ));
        assert!(matches!(
            Endpoint::parse("ho st:8080").unwrap_err(),
            EndpointError::InvalidHost { .. }
        ));
        assert_eq!(
            Endpoint::parse("/just/a/path").unwrap_err(),
            EndpointError::Empty
        );
    }

    #[test]
    fn bare_ipv6_without_brackets_is_a_host_not_a_host_port() {
        let endpoint = Endpoint::parse("fd00::1").unwrap();
        assert_eq!(endpoint.port(), None);
        assert_eq!(endpoint.key().as_str(), "[fd00::1]");
    }

    /// `validate_worker_url` accepts a path prefix and `endpoint_url` appends
    /// the route to it, so the prefix names part of the backend and has to
    /// survive into the key. Rejecting these made a worker registered behind a
    /// path unfindable for removal and update.
    #[test]
    fn a_path_prefix_is_part_of_the_backend() {
        assert_eq!(key_of("http://proxy:8080/sglang"), "proxy:8080/sglang");
        assert_eq!(key_of("https://api.example.com/v1"), "api.example.com/v1");
        assert_eq!(key_of("http://[fd00::1]:8080/v1"), "[fd00::1]:8080/v1");

        // Same backend, different spellings.
        assert_eq!(key_of("grpc://proxy:8080/sglang@2"), "proxy:8080/sglang");
        // Different path, different backend.
        assert_ne!(key_of("http://proxy:8080/a"), key_of("http://proxy:8080/b"));
        // A path is not optional decoration.
        assert_ne!(key_of("http://proxy:8080/a"), key_of("http://proxy:8080"));
    }

    /// The key retains userinfo because two registrations at one address with
    /// different credentials are different backends. That makes the key unsafe
    /// to log, so `Display` is redacted and `EndpointKey` has none at all —
    /// a caller wanting the raw value has to ask for `as_str`.
    #[test]
    fn a_password_is_masked_for_display_but_kept_for_lookup() {
        let endpoint = Endpoint::parse("http://user:pass@worker:3000").unwrap();

        assert_eq!(endpoint.redacted(), "http://user:***@worker:3000");
        assert_eq!(format!("{endpoint}"), "http://user:***@worker:3000");
        assert!(!endpoint.redacted().contains("pass"));

        // Identity and lookup keep the exact value.
        assert_eq!(endpoint.key().as_str(), "user:pass@worker:3000");
        assert_eq!(endpoint.lookup_form(), "user:pass@worker:3000");
        assert_eq!(endpoint.render(), "http://user:pass@worker:3000");

        // Two credentials at one address stay distinct.
        assert_ne!(
            key_of("http://user:a@worker:3000"),
            key_of("http://user:b@worker:3000")
        );
    }

    /// `{:?}` is the usual idiom when debugging the reconciler, and the
    /// structs that hold an endpoint derive `Debug`. A derived impl would
    /// have printed `userinfo: Some("user:pass")` verbatim.
    #[test]
    fn debug_masks_credentials_too() {
        let endpoint = Endpoint::parse("http://user:pass@worker:3000").unwrap();
        let shown = format!("{endpoint:?}");
        assert!(!shown.contains("pass"), "{shown}");
        assert!(shown.contains("user:***"), "{shown}");

        // A key cannot tell userinfo from the rest of its own string, so one
        // carrying an `@` is masked whole.
        assert!(!format!("{:?}", endpoint.key()).contains("pass"));

        // The common case stays readable, so a failing assertion still says
        // which key it got.
        let plain = Endpoint::parse("http://worker:3000").unwrap();
        assert_eq!(
            format!("{:?}", plain.key()),
            r#"EndpointKey("worker:3000")"#
        );
        assert!(format!("{plain:?}").contains("http://worker:3000"));
    }

    #[test]
    fn redaction_leaves_addresses_without_secrets_alone() {
        for spelling in [
            "http://worker:3000",
            "grpc://[fd00::1]:8080/v1",
            "ipc:///tmp/w.sock",
            // A bare username is not a secret and identifies the registration.
            "http://user@worker:3000",
        ] {
            let endpoint = Endpoint::parse(spelling).expect(spelling);
            assert_eq!(endpoint.redacted(), endpoint.render(), "{spelling}");
        }
    }

    /// A `]` in a path used to abort the path scan, leaving the whole string
    /// as an authority that then failed as a malformed IPv6 host. `url::Url`
    /// leaves brackets unencoded in a path, so this registers.
    #[test]
    fn brackets_inside_a_path_do_not_hide_it() {
        assert_eq!(key_of("http://proxy:8080/v1/[x]"), "proxy:8080/v1/[x]");
        // A genuinely bracketed host still splits at the right place.
        assert_eq!(
            key_of("http://[fd00::1]:8080/v1/[x]"),
            "[fd00::1]:8080/v1/[x]"
        );
    }

    #[test]
    fn a_trailing_slash_names_the_same_backend() {
        assert_eq!(key_of("http://proxy:8080/"), key_of("http://proxy:8080"));
        assert_eq!(
            key_of("http://proxy:8080/v1/"),
            key_of("http://proxy:8080/v1")
        );
        assert_eq!(
            Endpoint::parse("http://proxy:8080/v1/").unwrap().render(),
            "http://proxy:8080/v1"
        );
    }

    /// An IPC key is a bare socket path, which is not a parseable address, so
    /// a removal that re-submits the key by string must use the lookup form.
    #[test]
    fn lookup_form_round_trips_where_the_key_cannot() {
        let ipc = Endpoint::parse("ipc:///tmp/w.sock").unwrap();
        assert_eq!(ipc.key().as_str(), "/tmp/w.sock");
        assert!(Endpoint::parse(ipc.key().as_str()).is_err());

        assert_eq!(ipc.lookup_form(), "ipc:///tmp/w.sock");
        assert_eq!(
            Endpoint::parse(&ipc.lookup_form()).unwrap().key(),
            ipc.key()
        );

        // For a network address the two coincide, and stay scheme-less so a
        // removal still reaches every spelling of the backend.
        let net = Endpoint::parse("http://proxy:8080/sglang").unwrap();
        assert_eq!(net.lookup_form(), net.key().as_str());
        assert_eq!(
            Endpoint::parse(&net.lookup_form()).unwrap().key(),
            net.key()
        );
    }

    #[test]
    fn distinct_backends_keep_distinct_keys() {
        let keys = [
            key_of("10.0.0.1:8080"),
            key_of("10.0.0.1:8081"),
            key_of("10.0.0.2:8080"),
            key_of("[::1]:8080"),
            key_of("ipc:///tmp/w.sock"),
            key_of("ipc:///tmp/other.sock"),
            key_of("worker.ns.svc:8080"),
        ];
        let mut unique = keys.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "keys collided: {keys:?}");
    }
}

//! The worker URL door: which addresses a host takes as a worker's. One door
//! stands behind every way an address arrives (a start-up list, the worker
//! API, a discovery manifest), so each refuses the same addresses in the same
//! words. A manifest's address is asked as its registration keeps it: its
//! scheme in lowercase however the manifest spells it, and a scheme before
//! an address that has none. The other ways ask of the address as written,
//! so `HTTP://` passes from a manifest and is refused from the worker API.

use std::net::IpAddr;

use super::{
    endpoint::{Endpoint, EndpointError, Host as EndpointHost, Scheme},
    metadata::join_base,
};

/// Why the door refuses an address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlRefusal {
    /// The address as a refusal may show it: masked, its scheme alone, or
    /// nothing of it.
    pub named: String,
    /// The reason. It names no part of the address behind its scheme.
    pub reason: String,
}

impl UrlRefusal {
    fn new(named: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            named: named.into(),
            reason: reason.into(),
        }
    }
}

/// Why the door refuses `url`, `None` when it takes it. The door takes an
/// address that is not empty, starts with an allowed scheme, holds a host the
/// URL library reads, and is one address to both of its readers
/// (`read_two_ways`: the URL library a client dials by and this crate's own
/// type every line names by), as written and in the form routes are joined
/// to.
pub fn worker_url_refusal(url: &str) -> Option<UrlRefusal> {
    if url.is_empty() {
        return Some(UrlRefusal::new(url, "URL cannot be empty"));
    }

    // Exact (lowercase) scheme allow-list. Case-insensitive matching is
    // tempting but wrong here: the AddWorker workflow's normalize_url
    // matches schemes case-sensitively, so a mixed-case scheme would be
    // rewritten downstream and diverge from the reservation key — the
    // same orphan failure as a schemeless URL.
    const ALLOWED_SCHEMES: &[&str] = &["http", "https", "grpc", "ipc", "tcp"];
    let scheme = url.split_once("://").map_or("", |(s, _)| s);
    // A gRPC worker over TLS is refused by name, with the reason: the gRPC
    // client's channel speaks plain HTTP/2 over TCP, and no build of it
    // holds a root store to validate an engine's certificate against, so no
    // `grpcs://` worker ever completed a handshake a host could trust; the
    // refusal belongs here, at start-up, not at the first call.
    if scheme == "grpcs" {
        return Some(UrlRefusal::new(
            named_in_a_refusal(url),
            "grpcs:// workers are not supported: the gateway's gRPC channel speaks plain \
             HTTP/2 and holds no certificate store to validate an engine's; use grpc://"
                .to_string(),
        ));
    }
    // A refusal names the address through the address type's masking: it
    // is written to stderr at start-up and answered to the API caller, and
    // the URL may carry credentials. What the address parser accepts (a
    // schemeless address, a known scheme in upper case) is written with
    // its userinfo masked and its scheme lowercase; what it refuses (an
    // unknown scheme, no host, a parse failure) reads `<unparsable
    // address>`; what it reads with a rank shows its scheme alone
    // (`named_in_a_refusal`). The scheme is what this refusal is about and
    // the masked address no longer shows it as written, so the reason names
    // it when it is one (RFC 3986 scheme characters): the prefix before a
    // `://` further into a URL could be a userinfo, and is left out.
    if !ALLOWED_SCHEMES.contains(&scheme) {
        let rule =
            "URL must start with a lowercase http://, https://, grpc://, ipc://, or tcp:// scheme";
        let is_a_scheme = !scheme.is_empty()
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
        let reason = if is_a_scheme {
            format!("scheme '{scheme}://' is not allowed; {rule}")
        } else {
            rule.to_string()
        };
        return Some(UrlRefusal::new(named_in_a_refusal(url), reason));
    }

    // ipc:// worker URLs are same-host ZMQ unix-socket paths (no host); validate
    // the path is present rather than requiring a host below.
    if scheme == "ipc" {
        let path = url.strip_prefix("ipc://").unwrap_or("");
        if path.is_empty() {
            return Some(UrlRefusal::new(
                named_in_a_refusal(url),
                "ipc:// worker URL must include a socket path",
            ));
        }
        return None;
    }

    let dialled = match ::url::Url::parse(url) {
        Ok(parsed) if parsed.host_str().is_some() => parsed,
        Ok(_) => {
            return Some(UrlRefusal::new(
                named_in_a_refusal(url),
                "URL must have a valid host",
            ));
        }
        Err(e) => {
            return Some(UrlRefusal::new(
                named_in_a_refusal(url),
                format!("Invalid URL format: {e}"),
            ));
        }
    };
    // A tcp:// worker URL is what the ZMQ sockets bind, where a userinfo
    // (`user:secret@`) has no meaning at all: refused, by the adapter's own
    // reading (an `@` anywhere after the scheme), so that nothing the
    // validator passes is refused at registration instead, whatever the
    // `Endpoint` grammar makes of the rest of the string. (A gRPC worker
    // URL's userinfo is accepted: a host names it in a warning of its own,
    // at start-up and at registration, where a log line is seen.)
    if url
        .strip_prefix("tcp://")
        .is_some_and(|authority| authority.contains('@'))
    {
        return Some(UrlRefusal::new(
            named_in_a_refusal(url),
            "a tcp:// worker URL takes no userinfo: the gateway binds its ZMQ sockets at the \
             host and port",
        ));
    }
    // Routes join to the address less its trailing slashes (`join_base`),
    // and a line may name the worker by that form, so both readers find one
    // host and one port there too. A client reads one authority in both
    // forms. The address type reads a rank in the second that the first
    // does not show: `http://token@8080/` is a userinfo and a host as
    // written, and its base is a host and its rank.
    let base = join_base(url);
    let read_apart = read_two_ways(url, &dialled).or_else(|| {
        (base.len() != url.len())
            .then(|| read_two_ways(base, &dialled))
            .flatten()
    });
    if let Some(reason) = read_apart {
        // No part of the address is shown. What one reading takes for a
        // host the other may take for a credential, and the address type's
        // rendering trusts its own reading: `http://token@8080` is a host
        // and its rank to it, written whole.
        return Some(UrlRefusal::new(format!("{scheme}://***"), reason));
    }
    None
}

/// Why the worker URL door refuses `url`, `None` when it takes it: the
/// reason of [`worker_url_refusal`] alone, without the door's naming of the
/// address. A reason names no part of the address behind its scheme: the
/// door refuses what its two readers read apart, and there one reader's
/// host or port is the other's credential. A scheme the door does not know
/// is named as written, and no reader takes a scheme for a credential.
fn refusal_reason(url: &str) -> Option<String> {
    worker_url_refusal(url).map(|refusal| refusal.reason)
}

/// Why the worker URL door refuses an address a discovery manifest lists,
/// `None` when the door takes it: the door's reason alone, so a manifest's
/// refusal names no part of the address behind its scheme.
///
/// A scheme is a known one at the start of the address, as the address type
/// reads it ([`Scheme::split_from`]) and the registration keeps it, and it
/// is stored lowercase however the manifest spells it. An address with none
/// may hold a `://` further in, in a path or a query, and the registration
/// writes `http://` or `grpc://` before it, by what the engine answers.
/// Such an address is asked under `http://`: what the door refuses under
/// `grpc://` it refuses under `http://`, where a client reads a host in
/// more forms and ends an authority at one character more. One the URL
/// library cannot read under `http://` is asked under `grpc://`: no HTTP
/// client builds it, so the registration stores it there or not at all.
/// What stands behind the scheme is asked as given: the manifest's reader
/// asks of the address as written, and again as it is kept.
pub fn listed_worker_url_refusal(url: &str) -> Option<String> {
    match Scheme::split_from(url) {
        Some((scheme, rest)) => refusal_reason(&format!("{}{rest}", scheme.as_prefix())),
        None => {
            let http = format!("http://{url}");
            if ::url::Url::parse(&http).is_ok() {
                refusal_reason(&http)
            } else {
                refusal_reason(&format!("grpc://{url}"))
            }
        }
    }
}

/// How a refusal of the worker URL door names the address: by the address
/// type's masked rendering, and by its scheme alone when that rendering reads a
/// rank. No caller's address carries a rank, and what stands before the `@`
/// may be a credential the rendering takes for a host: `tcp://token@8080`
/// is a host and its rank to it, written whole.
fn named_in_a_refusal(url: &str) -> String {
    match Endpoint::parse_with_rank(url) {
        Ok((ranked, Some(_))) => {
            format!("{}***", ranked.scheme().map_or("", Scheme::as_prefix))
        }
        _ => Endpoint::redacted_str(url),
    }
}

/// Why a worker URL the URL library took is refused all the same, `None`
/// when this crate's address type reads it as the library does.
///
/// The address is read twice after this door. The health probe and the gRPC
/// channel dial what a URL client reads, `dialled` here. Every log line,
/// masked name and identity key follows [`Endpoint`]. Where the two part, a
/// line names one host and the client dials another:
/// `http://10.0.0.5:8000@2130706433` is rank 2130706433 of `10.0.0.5:8000`
/// to the address type, and the host 127.0.0.1 behind a userinfo to a
/// client. So the door takes an address only when both read one host and
/// one port in it, and the userinfo the address type reads holds no
/// character at which a client ends the authority. A rank suffix is the
/// registration's own, appended to a data-parallel worker's address after
/// this door, never a caller's.
///
/// Neither the refusal nor its reason names a part of the address: what
/// one reading takes for a host the other may take for a credential.
fn read_two_ways(url: &str, dialled: &::url::Url) -> Option<&'static str> {
    let named = match Endpoint::parse(url) {
        Ok(named) => named,
        Err(EndpointError::UnexpectedRank { .. }) => {
            return Some(
                "a worker URL cannot end in '@' and digits, trailing slashes aside: the gateway \
                 reads them as a rank suffix, which only its own registration appends",
            );
        }
        Err(_) => {
            return Some(
                "the gateway cannot read this address as a host and a port, though a URL \
                 client can: write it as scheme://host:port, with a port from 1 to 65535",
            );
        }
    };
    let same_host = match (dialled.host(), named.host()) {
        (Some(::url::Host::Ipv4(dialled)), EndpointHost::Ip(IpAddr::V4(named))) => {
            dialled == *named
        }
        (Some(::url::Host::Ipv6(dialled)), EndpointHost::Ip(IpAddr::V6(named))) => {
            dialled == *named
        }
        // A client reads an IPv4 address in more notations than the
        // address type does (`2130706433`, `0x7f.1`): one host, when the
        // name is that address to the library.
        (Some(::url::Host::Ipv4(dialled)), EndpointHost::Name(named)) => {
            matches!(::url::Host::parse(named), Ok(::url::Host::Ipv4(same)) if same == dialled)
        }
        // The library keeps the host of a `grpc://` or `tcp://` URL as
        // text, an address literal included.
        (Some(::url::Host::Domain(dialled)), EndpointHost::Ip(named)) => {
            dialled.parse::<IpAddr>().is_ok_and(|ip| ip == *named)
        }
        // One name, whatever its case; or the form the library gives the
        // name: an internationalized or a percent-encoded one under `http`
        // and `https`, one kept as text with its bytes outside ASCII
        // percent-encoded under `grpc` and `tcp`. The library keeps the
        // ASCII case of that text and the address type lowers it, so the
        // third compare takes no account of it either.
        (Some(::url::Host::Domain(dialled)), EndpointHost::Name(named)) => {
            dialled.eq_ignore_ascii_case(named)
                || matches!(
                    ::url::Host::parse(named),
                    Ok(::url::Host::Domain(same)) if same == dialled
                )
                || matches!(
                    ::url::Host::parse_opaque(named),
                    Ok(::url::Host::Domain(same)) if same.eq_ignore_ascii_case(dialled)
                )
        }
        _ => false,
    };
    // A client fills a missing port from the scheme and drops one that
    // equals the scheme's own.
    let same_port = match (dialled.port(), named.port()) {
        (dialled, named) if dialled == named => true,
        (None, Some(named)) => dialled.port_or_known_default() == Some(named),
        _ => false,
    };
    if !(same_host && same_port) {
        return Some(
            "the address reads two ways: a client would dial one host and port and the gateway \
             would name another; percent-encode an '@' or a '\\' inside a userinfo",
        );
    }
    // The two also end the authority apart. A client ends it at the first
    // `/`, `?` or `#`, and under `http` and `https` at a backslash too. The
    // address type ends it at the first `/`, `?` or `#` behind a
    // bracketed host: it reads on past a backslash, and past all four
    // before the first `]` of an authority that opens with a bracket. With
    // one host and port on both sides (`http://w:1\@w:1`,
    // `http://[u@w:1/]@w:1`) the two still part: a client joins every route
    // under the rest as a path, and the address type reads a userinfo
    // that is not there. So the userinfo it reads holds none of them.
    let ends_a_clients_authority = |c: char| {
        matches!(c, '/' | '?' | '#') || (c == '\\' && matches!(dialled.scheme(), "http" | "https"))
    };
    named.userinfo_holds(ends_a_clients_authority).then_some(
        "the address reads two ways: a client ends the authority inside what the gateway \
         reads as a userinfo, and joins its routes under the rest as a path; percent-encode \
         a '/', a '?', a '#' or a '\\' inside a userinfo",
    )
}

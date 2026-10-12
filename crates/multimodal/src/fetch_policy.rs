//! Which hosts a client-supplied media URL may be fetched from.
//!
//! Two rules, checked on every hop of a fetch (the URL the request named and
//! each redirect target): an optional allow-list of host names, and an
//! address policy that refuses a host that names or resolves to an address
//! that is not public (loopback, link-local, private, shared address space,
//! unspecified, multicast, broadcast) unless the deployment allows internal
//! media origins. The address policy is judged on the addresses the host
//! resolves to, and the connection goes to those same addresses: the
//! connector resolves the host, checks every address, and pins the answer for
//! the HTTP client's resolver for as long as the hop is in flight
//! ([`MediaHostResolver`]), so a DNS answer cannot change between the check
//! and the connect. Behind an egress proxy the proxy resolves the name; the
//! check still happens here, on this side's answer.

use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    io,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{Arc, Mutex, PoisonError},
};

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use url::{Host, Url};

use crate::error::MediaConnectorError;

/// The allow-list and the address policy of one connector.
#[derive(Debug)]
pub(crate) struct MediaHostPolicy {
    /// Host names a URL may name, lowercase; `None` is any host.
    allowed_domains: Option<HashSet<String>>,
    /// Whether a host may name or resolve to a non-public address.
    allow_private_hosts: bool,
}

impl MediaHostPolicy {
    pub(crate) fn new(allowed_domains: Option<Vec<String>>, allow_private_hosts: bool) -> Self {
        Self {
            allowed_domains: allowed_domains.map(|domains| {
                domains
                    .into_iter()
                    .map(|domain| domain.to_ascii_lowercase())
                    .collect()
            }),
            allow_private_hosts,
        }
    }

    /// The URL's host against the allow-list, by name as the URL spells it
    /// (an IP literal counts as its own name); the name, lowercase.
    fn check_name(&self, url: &Url) -> Result<String, MediaConnectorError> {
        let host = url
            .host_str()
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| MediaConnectorError::InvalidUrl(url.to_string()))?;
        if let Some(allowed) = &self.allowed_domains {
            if !allowed.contains(&host) {
                return Err(MediaConnectorError::DisallowedDomain(host));
            }
        }
        Ok(host)
    }

    /// One address of `host` against the address policy.
    fn check_address(&self, host: &str, address: IpAddr) -> Result<(), MediaConnectorError> {
        if self.allow_private_hosts {
            return Ok(());
        }
        match non_public_address_class(address) {
            Some(class) => Err(MediaConnectorError::DisallowedAddress {
                host: host.to_string(),
                address,
                class,
            }),
            None => Ok(()),
        }
    }
}

/// The class of an address media is not fetched from, as the refusal names
/// it; `None` for a public address. An IPv4 address carried in IPv6 notation
/// (`::ffff:a.b.c.d`) is judged as the IPv4 address it is.
pub(crate) fn non_public_address_class(address: IpAddr) -> Option<&'static str> {
    match address {
        IpAddr::V4(address) => ipv4_class(address),
        IpAddr::V6(address) => address
            .to_ipv4_mapped()
            .map_or_else(|| ipv6_class(address), ipv4_class),
    }
}

fn ipv4_class(address: Ipv4Addr) -> Option<&'static str> {
    let [first, second, ..] = address.octets();
    if first == 0 {
        // 0.0.0.0/8: "this network", the unspecified address included.
        Some("unspecified")
    } else if address.is_loopback() {
        Some("loopback")
    } else if address.is_private() {
        Some("private")
    } else if address.is_link_local() {
        Some("link-local")
    } else if first == 100 && (64..128).contains(&second) {
        // 100.64.0.0/10, the carrier-grade NAT range.
        Some("shared address space")
    } else if address.is_multicast() {
        Some("multicast")
    } else if address.is_broadcast() {
        Some("broadcast")
    } else {
        None
    }
}

fn ipv6_class(address: Ipv6Addr) -> Option<&'static str> {
    if address.is_unspecified() {
        Some("unspecified")
    } else if address.is_loopback() {
        Some("loopback")
    } else if address.is_unique_local() {
        Some("unique local")
    } else if address.is_unicast_link_local() {
        Some("link-local")
    } else if address.is_multicast() {
        Some("multicast")
    } else {
        None
    }
}

/// The connector's HTTP client resolver. A host the connector has admitted
/// resolves to the addresses it checked, for as long as a fetch of it is in
/// flight; any other name (an egress proxy's, for one) resolves as the system
/// resolver answers.
#[derive(Debug, Default)]
pub(crate) struct MediaHostResolver {
    pins: Mutex<HashMap<String, Pin>>,
}

#[derive(Debug)]
struct Pin {
    addresses: Vec<IpAddr>,
    /// Fetches of the host in flight; the pin goes with the last of them.
    fetches: usize,
}

impl MediaHostResolver {
    /// Admit one hop of a fetch under `policy`: the host by name against the
    /// allow-list, then its address (an IP literal) or every address it
    /// resolves to against the address policy. The guard keeps the checked
    /// addresses pinned for this resolver until the hop is over, so the
    /// client connects to what was checked.
    pub(crate) async fn admit(
        self: &Arc<Self>,
        policy: &MediaHostPolicy,
        url: &Url,
    ) -> Result<HostPin, MediaConnectorError> {
        let name = policy.check_name(url)?;
        match url.host() {
            Some(Host::Domain(domain)) => {
                let addresses = resolve(domain).await?;
                for &address in &addresses {
                    policy.check_address(domain, address)?;
                }
                Ok(self.pin(domain, addresses))
            }
            Some(Host::Ipv4(address)) => {
                policy.check_address(&name, IpAddr::V4(address))?;
                Ok(HostPin { pinned: None })
            }
            Some(Host::Ipv6(address)) => {
                policy.check_address(&name, IpAddr::V6(address))?;
                Ok(HostPin { pinned: None })
            }
            None => Err(MediaConnectorError::InvalidUrl(url.to_string())),
        }
    }

    fn pin(self: &Arc<Self>, host: &str, addresses: Vec<IpAddr>) -> HostPin {
        let mut pins = self.pins.lock().unwrap_or_else(PoisonError::into_inner);
        match pins.entry(host.to_string()) {
            Entry::Occupied(mut entry) => {
                let pin = entry.get_mut();
                pin.addresses = addresses;
                pin.fetches += 1;
            }
            Entry::Vacant(entry) => {
                entry.insert(Pin {
                    addresses,
                    fetches: 1,
                });
            }
        }
        HostPin {
            pinned: Some((Arc::clone(self), host.to_string())),
        }
    }

    fn unpin(&self, host: &str) {
        let mut pins = self.pins.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(pin) = pins.get_mut(host) {
            pin.fetches = pin.fetches.saturating_sub(1);
            if pin.fetches == 0 {
                pins.remove(host);
            }
        }
    }

    fn pinned(&self, host: &str) -> Option<Vec<IpAddr>> {
        self.pins
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(host)
            .map(|pin| pin.addresses.clone())
    }
}

impl Resolve for MediaHostResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let pinned = self.pinned(name.as_str());
        Box::pin(async move {
            let addresses: Vec<SocketAddr> = match pinned {
                Some(addresses) => addresses
                    .into_iter()
                    .map(|address| SocketAddr::new(address, 0))
                    .collect(),
                None => tokio::net::lookup_host((name.as_str(), 0)).await?.collect(),
            };
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

/// A hop's hold on its host's pinned addresses; dropping it releases them.
#[derive(Debug)]
pub(crate) struct HostPin {
    pinned: Option<(Arc<MediaHostResolver>, String)>,
}

impl Drop for HostPin {
    fn drop(&mut self) {
        if let Some((resolver, host)) = self.pinned.take() {
            resolver.unpin(&host);
        }
    }
}

/// The addresses `host` resolves to, through the system resolver, each once.
async fn resolve(host: &str) -> Result<Vec<IpAddr>, MediaConnectorError> {
    let mut addresses: Vec<IpAddr> = Vec::new();
    for address in tokio::net::lookup_host((host, 0))
        .await
        .map_err(|source| unresolvable(host, source))?
    {
        if !addresses.contains(&address.ip()) {
            addresses.push(address.ip());
        }
    }
    if addresses.is_empty() {
        return Err(unresolvable(
            host,
            io::Error::new(io::ErrorKind::NotFound, "the name has no address"),
        ));
    }
    Ok(addresses)
}

fn unresolvable(host: &str, source: io::Error) -> MediaConnectorError {
    MediaConnectorError::UnresolvableHost {
        host: host.to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn ip(text: &str) -> IpAddr {
        IpAddr::from_str(text).expect("an address")
    }

    /// The ranges a client-supplied URL must not lead to, and the ones it may.
    #[test]
    fn non_public_addresses_are_classified_and_public_ones_pass() {
        for (address, class) in [
            ("127.0.0.1", "loopback"),
            ("127.255.255.254", "loopback"),
            ("::1", "loopback"),
            ("10.0.0.5", "private"),
            ("172.16.0.1", "private"),
            ("172.31.255.255", "private"),
            ("192.168.1.1", "private"),
            ("fd12:3456::1", "unique local"),
            ("fc00::1", "unique local"),
            ("169.254.169.254", "link-local"),
            ("fe80::1", "link-local"),
            ("100.64.0.1", "shared address space"),
            ("100.127.255.255", "shared address space"),
            ("0.0.0.0", "unspecified"),
            ("0.1.2.3", "unspecified"),
            ("::", "unspecified"),
            ("224.0.0.1", "multicast"),
            ("ff02::1", "multicast"),
            ("255.255.255.255", "broadcast"),
            // IPv4 in IPv6 notation is the IPv4 address it carries.
            ("::ffff:127.0.0.1", "loopback"),
            ("::ffff:10.1.2.3", "private"),
        ] {
            assert_eq!(
                non_public_address_class(ip(address)),
                Some(class),
                "{address}"
            );
        }
        for address in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "100.63.255.255",
            "100.128.0.0",
            "172.15.255.255",
            "172.32.0.0",
            "2606:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert_eq!(non_public_address_class(ip(address)), None, "{address}");
        }
    }

    /// The allow-list is a name check, case-insensitive, and an IP literal is
    /// its own name; the address policy judges a literal's address.
    #[tokio::test]
    async fn the_allow_list_and_the_address_policy_each_refuse_their_own() {
        let resolver = Arc::new(MediaHostResolver::default());
        let listed = MediaHostPolicy::new(Some(vec!["Media.Example".to_string()]), true);
        let url = Url::parse("http://media.example/a.png").expect("url");
        // A listed name whose address is public cannot be resolved here, so
        // the name check is observed through an unlisted one.
        let other = Url::parse("http://other.example/a.png").expect("url");
        assert!(matches!(
            resolver.admit(&listed, &other).await,
            Err(MediaConnectorError::DisallowedDomain(host)) if host == "other.example"
        ));
        let literal = Url::parse("http://127.0.0.1:8080/a.png").expect("url");
        assert!(matches!(
            resolver.admit(&listed, &literal).await,
            Err(MediaConnectorError::DisallowedDomain(host)) if host == "127.0.0.1"
        ));
        drop(url);

        let any = MediaHostPolicy::new(None, false);
        let error = resolver
            .admit(&any, &literal)
            .await
            .expect_err("loopback is refused");
        assert!(
            matches!(
                &error,
                MediaConnectorError::DisallowedAddress { host, address, class }
                    if host == "127.0.0.1" && *address == ip("127.0.0.1") && *class == "loopback"
            ),
            "{error}"
        );
        let mapped = Url::parse("http://[::ffff:10.0.0.9]/a.png").expect("url");
        assert!(matches!(
            resolver.admit(&any, &mapped).await,
            Err(MediaConnectorError::DisallowedAddress {
                class: "private",
                ..
            })
        ));
        let allowed = MediaHostPolicy::new(None, true);
        let pin = resolver
            .admit(&allowed, &literal)
            .await
            .expect("private hosts allowed");
        assert!(pin.pinned.is_none(), "a literal needs no pin");
    }

    /// A name is judged on the addresses it resolves to, and the client's
    /// resolver is handed those addresses while the hop is in flight; after
    /// the hop it resolves as the system does.
    #[tokio::test]
    async fn a_resolved_name_is_judged_and_pinned_for_the_client() {
        let resolver = Arc::new(MediaHostResolver::default());
        let any = MediaHostPolicy::new(None, false);
        let url = Url::parse("http://localhost:8080/a.png").expect("url");
        let error = resolver
            .admit(&any, &url)
            .await
            .expect_err("localhost resolves to loopback");
        assert!(
            matches!(
                &error,
                MediaConnectorError::DisallowedAddress { host, class: "loopback", .. }
                    if host == "localhost"
            ),
            "{error}"
        );

        let allowed = MediaHostPolicy::new(None, true);
        let pin = resolver
            .admit(&allowed, &url)
            .await
            .expect("private hosts allowed");
        let name = Name::from_str("localhost").expect("a name");
        let pinned: Vec<SocketAddr> = resolver.resolve(name).await.expect("pinned").collect();
        assert!(!pinned.is_empty());
        assert!(
            pinned
                .iter()
                .all(|address| address.ip().is_loopback() && address.port() == 0),
            "{pinned:?}"
        );
        let checked = resolver
            .pinned("localhost")
            .expect("pinned while the hop is in flight");
        assert_eq!(
            pinned.iter().map(SocketAddr::ip).collect::<Vec<_>>(),
            checked
        );
        drop(pin);
        assert!(
            resolver.pinned("localhost").is_none(),
            "released with the hop"
        );

        // Two hops of the same host in flight: the pin outlives the first.
        let first = resolver.admit(&allowed, &url).await.expect("first");
        let second = resolver.admit(&allowed, &url).await.expect("second");
        drop(first);
        assert!(resolver.pinned("localhost").is_some());
        drop(second);
        assert!(resolver.pinned("localhost").is_none());
    }

    /// A name nothing resolves is the request's error, named. A label over
    /// 63 characters is refused by the resolver itself wherever the test
    /// runs (a sinkhole DNS may answer any well-formed name).
    #[tokio::test]
    async fn an_unresolvable_name_is_reported_as_such() {
        let host = format!("{}.invalid", "a".repeat(64));
        let error = resolve(&host).await.expect_err("unresolvable");
        assert!(
            matches!(&error, MediaConnectorError::UnresolvableHost { host: named, .. } if named == &host),
            "{error}"
        );
    }
}

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
//! check still happens here, on this side's answer. With private media hosts
//! allowed there is no address to refuse, so a host is admitted by name
//! alone: no lookup here and no pin, and the client (or the proxy) resolves
//! it as it did before there was a policy.

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

/// An allow-list as the policy reads it: each entry trimmed and lowercased,
/// empty entries dropped (a list spelled with a space after each comma names
/// the same hosts), and a list left with no names is no allow-list (`None`):
/// any host the address policy passes, the meaning the gateway's settings and
/// its Python parser already give an empty list. Sorted, without repeats, so
/// the start-up line that prints it reads the same way every start. The one
/// reading for every way a list arrives (the CLI, a configuration file, a
/// worker's flag) and for what is logged about it.
pub fn effective_allowed_domains(allowed_domains: Option<&[String]>) -> Option<Vec<String>> {
    let mut domains: Vec<String> = allowed_domains?
        .iter()
        .map(|domain| domain.trim().to_ascii_lowercase())
        .filter(|domain| !domain.is_empty())
        .collect();
    domains.sort();
    domains.dedup();
    (!domains.is_empty()).then_some(domains)
}

impl MediaHostPolicy {
    /// `allowed_domains` by name, as [`effective_allowed_domains`] reads it:
    /// decided here, once, for every way a list arrives.
    pub(crate) fn new(allowed_domains: Option<Vec<String>>, allow_private_hosts: bool) -> Self {
        Self {
            allowed_domains: effective_allowed_domains(allowed_domains.as_deref())
                .map(|domains| domains.into_iter().collect::<HashSet<String>>()),
            allow_private_hosts,
        }
    }

    /// Whether hosts are judged by address at all. With private hosts
    /// allowed there is no address to refuse, so a host is admitted by name
    /// alone and left to the client to resolve, as it was before there was
    /// a policy (through an egress proxy, where there is one).
    fn checks_addresses(&self) -> bool {
        !self.allow_private_hosts
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
}

/// One address of `host` against the address policy, asked only when the
/// policy checks addresses: a non-public address is refused, named with its
/// class.
fn check_address(host: &str, address: IpAddr) -> Result<(), MediaConnectorError> {
    match non_public_address_class(address) {
        Some(class) => Err(MediaConnectorError::DisallowedAddress {
            host: host.to_string(),
            address,
            class,
        }),
        None => Ok(()),
    }
}

/// The class of an address media is not fetched from, as the refusal names
/// it; `None` for a public address. An IPv6 address that carries an IPv4
/// address a translator or a relay connects to ([`embedded_ipv4`]) is judged
/// as that IPv4 address.
pub(crate) fn non_public_address_class(address: IpAddr) -> Option<&'static str> {
    match address {
        IpAddr::V4(address) => ipv4_class(address),
        IpAddr::V6(address) => match embedded_ipv4(address) {
            Some(carried) => ipv4_class(carried),
            None => ipv6_class(address),
        },
    }
}

/// The IPv4 address an IPv6 address carries and leads to, in the forms a
/// translator or a relay turns into an IPv4 connection: IPv4-mapped
/// (`::ffff:a.b.c.d`), IPv4-compatible (`::a.b.c.d`; `::` and `::1` are
/// addresses of their own), NAT64 on the well-known prefix `64:ff9b::/96`
/// and on `64:ff9b:1::/96` of the local-use `64:ff9b:1::/48`, and 6to4
/// (`2002::/16`, the IPv4 address in the next 32 bits). The rest of the
/// local-use prefix carries no readable address here: RFC 6052's /48, /56
/// and /64 layouts put the IPv4 address higher, a translator ignores the
/// suffix bits, and the address alone does not say which layout is in use,
/// so the low 32 bits of such an address can be made to spell any public
/// address; [`non_public_address_class`] refuses it outright. (A /96 carved
/// elsewhere in the local-use prefix is refused with it, under the strict
/// policy.)
fn embedded_ipv4(address: Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return Some(mapped);
    }
    let low_32 = |high: u16, low: u16| Ipv4Addr::from((u32::from(high) << 16) | u32::from(low));
    match address.segments() {
        [0, 0, 0, 0, 0, 0, 0, 0 | 1] => None,
        [0, 0, 0, 0, 0, 0, high, low] => Some(low_32(high, low)),
        [0x64, 0xff9b, 0, 0, 0, 0, high, low] => Some(low_32(high, low)),
        [0x64, 0xff9b, 1, 0, 0, 0, high, low] => Some(low_32(high, low)),
        [0x2002, high, low, ..] => Some(low_32(high, low)),
        _ => None,
    }
}

/// Whether the address is in the local-use NAT64 prefix `64:ff9b:1::/48`
/// carved in a layout [`embedded_ipv4`] cannot read.
fn is_nat64_local_use_with_an_unreadable_layout(address: Ipv6Addr) -> bool {
    matches!(address.segments(), [0x64, 0xff9b, 1, ..]) && embedded_ipv4(address).is_none()
}

fn ipv4_class(address: Ipv4Addr) -> Option<&'static str> {
    let [first, second, third, _] = address.octets();
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
    } else if first == 192 && second == 0 && third == 0 {
        // 192.0.0.0/24, the IETF protocol assignments.
        Some("reserved")
    } else if first == 198 && (second == 18 || second == 19) {
        // 198.18.0.0/15, the benchmarking range.
        Some("benchmarking")
    } else if address.is_multicast() {
        Some("multicast")
    } else if address.is_broadcast() {
        Some("broadcast")
    } else if first >= 240 {
        // 240.0.0.0/4, reserved.
        Some("reserved")
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
    } else if is_nat64_local_use_with_an_unreadable_layout(address) {
        // The translator connects to an IPv4 address this side cannot read:
        // refused, rather than taken for a public one.
        Some("NAT64 local-use with an unreadable layout")
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
    /// The answers a test gives for names nothing resolves where it runs
    /// ([`Self::answer`]); empty otherwise, and the system resolver answers.
    answers: Mutex<HashMap<String, Vec<IpAddr>>>,
}

#[derive(Debug)]
struct Pin {
    addresses: Vec<IpAddr>,
    /// Fetches of the host in flight; the pin goes with the last of them.
    fetches: usize,
}

impl MediaHostResolver {
    /// Admit one hop of a fetch under `policy`: the host by name against the
    /// allow-list, then, when the policy checks addresses, its address (an
    /// IP literal) or every address it resolves to against the address
    /// policy. The guard keeps the checked addresses pinned for this resolver
    /// until the hop is over, so the client connects to what was checked. A
    /// policy that checks no address (private hosts allowed) admits the host
    /// by name alone, with no lookup and nothing to pin.
    pub(crate) async fn admit(
        self: &Arc<Self>,
        policy: &MediaHostPolicy,
        url: &Url,
    ) -> Result<HostPin, MediaConnectorError> {
        let name = policy.check_name(url)?;
        if !policy.checks_addresses() {
            return Ok(HostPin { pinned: None });
        }
        match url.host() {
            Some(Host::Domain(domain)) => {
                let addresses = self.lookup(domain).await?;
                for &address in &addresses {
                    check_address(domain, address)?;
                }
                Ok(self.pin(domain, addresses))
            }
            Some(Host::Ipv4(address)) => {
                check_address(&name, IpAddr::V4(address))?;
                Ok(HostPin { pinned: None })
            }
            Some(Host::Ipv6(address)) => {
                check_address(&name, IpAddr::V6(address))?;
                Ok(HostPin { pinned: None })
            }
            None => Err(MediaConnectorError::InvalidUrl(url.to_string())),
        }
    }

    /// The addresses `host` resolves to: a test's answer for it when there
    /// is one, else the system resolver's, each once.
    async fn lookup(&self, host: &str) -> Result<Vec<IpAddr>, MediaConnectorError> {
        let answered = self
            .answers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(host)
            .cloned();
        match answered {
            Some(addresses) => Ok(addresses),
            None => resolve(host).await,
        }
    }

    /// What `host` resolves to from now on, for a test: no public name
    /// resolves where the tests run, and the pin of an admitted hop is
    /// observed through one.
    #[cfg(test)]
    fn answer(&self, host: &str, addresses: Vec<IpAddr>) {
        self.answers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(host.to_string(), addresses);
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
            ("192.0.0.1", "reserved"),
            ("192.0.0.255", "reserved"),
            ("198.18.0.1", "benchmarking"),
            ("198.19.255.255", "benchmarking"),
            ("240.0.0.1", "reserved"),
            ("255.255.255.254", "reserved"),
            // IPv4 in IPv6 notation is the IPv4 address it carries: mapped,
            // compatible, NAT64 (both prefixes) and 6to4.
            ("::ffff:127.0.0.1", "loopback"),
            ("::ffff:10.1.2.3", "private"),
            ("::10.1.2.3", "private"),
            ("64:ff9b::a9fe:a9fe", "link-local"),
            ("64:ff9b::7f00:1", "loopback"),
            ("64:ff9b:1::a00:5", "private"),
            // The local-use prefix anywhere but 64:ff9b:1::/96: the IPv4
            // address sits where the address alone does not say, so it is
            // refused whatever the low bits spell (10.0.0.1 and 127.0.0.1 in
            // the /64 layout read 1.0.0.0 from the low bits; 10.1.0.0 in the
            // /48 layout with a public address in the suffix; a public
            // address in the /64 layout; a /96 carved at a non-zero /64).
            (
                "64:ff9b:1:1:a:0:100:0",
                "NAT64 local-use with an unreadable layout",
            ),
            (
                "64:ff9b:1:1:7f:0:100:0",
                "NAT64 local-use with an unreadable layout",
            ),
            (
                "64:ff9b:1:a01::5db8:d822",
                "NAT64 local-use with an unreadable layout",
            ),
            (
                "64:ff9b:1:1:5d:b8d8:2200:0",
                "NAT64 local-use with an unreadable layout",
            ),
            (
                "64:ff9b:1:abcd::c0a8:101",
                "NAT64 local-use with an unreadable layout",
            ),
            ("2002:a00:5::", "private"),
            ("2002:7f00:1::1", "loopback"),
            ("2002:a9fe:a9fe::", "link-local"),
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
            "::5db8:d822",
            "64:ff9b::5db8:d822",
            "64:ff9b:1::5db8:d822",
            "2002:5db8:d822::1",
            "192.0.1.0",
            "198.17.255.255",
            "198.20.0.0",
            "223.255.255.255",
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

    /// A name is judged on the addresses it resolves to; the addresses a hop
    /// was admitted with are handed to the client's resolver for as long as
    /// the hop is in flight, and after it the name resolves as the system
    /// does.
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

        // The pin of an admitted hop, with the addresses a public name would
        // have resolved to (none resolves here): the client is handed the
        // checked addresses, nothing else, until the hop is over.
        let checked = vec![
            ip("93.184.216.34"),
            ip("2606:2800:21f:cb07:6820:80da:af6b:8b2c"),
        ];
        let pin = resolver.pin("media.example", checked.clone());
        let name = Name::from_str("media.example").expect("a name");
        let pinned: Vec<SocketAddr> = resolver.resolve(name).await.expect("pinned").collect();
        assert_eq!(
            pinned.iter().map(SocketAddr::ip).collect::<Vec<_>>(),
            checked
        );
        assert!(pinned.iter().all(|address| address.port() == 0));
        assert_eq!(resolver.pinned("media.example"), Some(checked.clone()));
        drop(pin);
        assert!(
            resolver.pinned("media.example").is_none(),
            "released with the hop"
        );

        // Two hops of the same host in flight: the pin outlives the first.
        let first = resolver.pin("media.example", checked.clone());
        let second = resolver.pin("media.example", checked);
        drop(first);
        assert!(resolver.pinned("media.example").is_some());
        drop(second);
        assert!(resolver.pinned("media.example").is_none());
    }

    /// With private hosts allowed there is no address to judge, so a host is
    /// admitted by name alone: no lookup, no pin. A deployment that resolves
    /// external names only at its egress proxy keeps fetching that way.
    #[tokio::test]
    async fn a_name_is_admitted_without_a_lookup_when_private_hosts_are_allowed() {
        let resolver = Arc::new(MediaHostResolver::default());
        // A label over 63 characters: no resolver answers it, wherever the
        // test runs, so a lookup here would fail.
        let host = format!("{}.invalid", "a".repeat(64));
        let url = Url::parse(&format!("http://{host}/a.png")).expect("url");

        let strict = MediaHostPolicy::new(None, false);
        let error = resolver
            .admit(&strict, &url)
            .await
            .expect_err("judged by address: the lookup fails");
        assert!(
            matches!(&error, MediaConnectorError::UnresolvableHost { host: named, .. } if named == &host),
            "{error}"
        );

        let open = MediaHostPolicy::new(None, true);
        let pin = resolver
            .admit(&open, &url)
            .await
            .expect("admitted by name alone");
        assert!(pin.pinned.is_none(), "nothing was resolved, nothing to pin");
        assert!(resolver.pinned(&host).is_none());

        // The allow-list still holds, by name.
        let listed = MediaHostPolicy::new(Some(vec!["media.example".to_string()]), true);
        assert!(matches!(
            resolver.admit(&listed, &url).await,
            Err(MediaConnectorError::DisallowedDomain(named)) if named == host
        ));
    }

    /// Under the strict policy a public name is admitted on the addresses it
    /// resolves to, every one of them judged, and the client is handed those
    /// same addresses until the hop is over: check, then connect to what was
    /// checked. The answers are the test's: no public name resolves here.
    #[tokio::test]
    async fn a_public_name_is_admitted_and_its_checked_addresses_pinned() {
        let resolver = Arc::new(MediaHostResolver::default());
        let strict = MediaHostPolicy::new(None, false);
        let checked = vec![
            ip("93.184.216.34"),
            ip("2606:2800:21f:cb07:6820:80da:af6b:8b2c"),
        ];
        resolver.answer("media.example", checked.clone());
        let url = Url::parse("http://media.example/a.png").expect("url");
        let pin = resolver
            .admit(&strict, &url)
            .await
            .expect("a public name is admitted");
        assert_eq!(resolver.pinned("media.example"), Some(checked.clone()));
        let name = Name::from_str("media.example").expect("a name");
        let pinned: Vec<SocketAddr> = resolver.resolve(name).await.expect("pinned").collect();
        assert_eq!(
            pinned.iter().map(SocketAddr::ip).collect::<Vec<_>>(),
            checked
        );
        drop(pin);
        assert!(
            resolver.pinned("media.example").is_none(),
            "released with the hop"
        );

        // One non-public address among the answers refuses the name, and
        // nothing is pinned for a refused hop.
        resolver.answer("mixed.example", vec![ip("93.184.216.34"), ip("10.0.0.5")]);
        let url = Url::parse("http://mixed.example/a.png").expect("url");
        let error = resolver
            .admit(&strict, &url)
            .await
            .expect_err("judged on every address");
        assert!(
            matches!(
                &error,
                MediaConnectorError::DisallowedAddress { host, class: "private", .. }
                    if host == "mixed.example"
            ),
            "{error}"
        );
        assert!(resolver.pinned("mixed.example").is_none());
    }

    /// The allow-list a start-up line prints is the one the policy enforces:
    /// trimmed, lowercased, without empties or repeats, sorted, and `None`
    /// for a list left with no names.
    #[test]
    fn the_effective_allow_list_is_the_one_logged_and_enforced() {
        let spelled: Vec<String> = [" Media.Example", "", "cdn.example", "cdn.example ", "  "]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(
            effective_allowed_domains(Some(&spelled)),
            Some(vec!["cdn.example".to_string(), "media.example".to_string()])
        );
        let blank: Vec<String> = vec![" ".to_string(), String::new()];
        assert_eq!(effective_allowed_domains(Some(&blank)), None);
        assert_eq!(effective_allowed_domains(Some(&[])), None);
        assert_eq!(effective_allowed_domains(None), None);
        let policy = MediaHostPolicy::new(Some(spelled), false);
        assert_eq!(
            policy.allowed_domains,
            Some(HashSet::from([
                "cdn.example".to_string(),
                "media.example".to_string()
            ]))
        );
    }

    /// An allow-list entry is matched by its name alone: surrounding blanks
    /// (a space after the comma of a list) do not make a host unmatchable,
    /// an empty entry lists nothing, and a list with no names left is no
    /// allow-list at all (any host the address policy passes), whichever
    /// way it arrived.
    #[tokio::test]
    async fn allow_list_entries_are_trimmed_and_empty_ones_dropped() {
        let resolver = Arc::new(MediaHostResolver::default());
        let listed = MediaHostPolicy::new(
            Some(vec![
                "media.example".to_string(),
                " cdn.example".to_string(),
                String::new(),
                "  ".to_string(),
            ]),
            true,
        );
        for host in ["media.example", "cdn.example", "CDN.example"] {
            let url = Url::parse(&format!("http://{host}/a.png")).expect("url");
            resolver
                .admit(&listed, &url)
                .await
                .unwrap_or_else(|error| panic!("{host}: {error}"));
        }
        let url = Url::parse("http://other.example/a.png").expect("url");
        assert!(matches!(
            resolver.admit(&listed, &url).await,
            Err(MediaConnectorError::DisallowedDomain(named)) if named == "other.example"
        ));

        for blank in [
            vec![],
            vec![String::new()],
            vec![" ".to_string(), String::new()],
        ] {
            let unlisted = MediaHostPolicy::new(Some(blank.clone()), true);
            assert!(
                unlisted.allowed_domains.is_none(),
                "{blank:?} names no host: no allow-list"
            );
            resolver
                .admit(&unlisted, &url)
                .await
                .unwrap_or_else(|error| panic!("{blank:?}: {error}"));
        }
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

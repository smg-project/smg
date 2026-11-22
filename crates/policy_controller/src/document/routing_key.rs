//! A request's routing key from the sources its binding names (design 4.6).
//!
//! A binding lists where the key comes from, in order: request headers and
//! the request's own id. The host reads them and the engine reads one key
//! and no raw header ([`RequestFacts::session_key`]). The derivation is
//! one function, [`KeySources::derive`], for a host in process and for a
//! gateway that asks a controller under an address.
//!
//! A source can also tell one client from another: a header a proxy or a
//! browser sets. Its key is a second fact, read by
//! [`KeySources::derive_client`]: a ring places by it where the request
//! carries no session key, and nothing else reads it
//! ([`RequestFacts::client_key`]).
//!
//! A host that lends its own hash ring to a decision says where the
//! request's key stands on it ([`RequestFacts::ring_position`]). The key's
//! text is read here and leaves in no other form, so the position is taken
//! here, with the two keys, by one read of the sources
//! ([`KeySources::derive_keys`]).
//!
//! [`RequestFacts::client_key`]: crate::engine::RequestFacts::client_key
//! [`RequestFacts::ring_position`]: crate::engine::RequestFacts::ring_position
//! [`RequestFacts::session_key`]: crate::engine::RequestFacts::session_key

use crate::{
    document::spec::{BindingSpec, RoutingKeySpec, SessionsSpec, ViewSpec, KEY_BYTES},
    engine::{
        framework::{slot, PHASES},
        view::{fingerprint, Identity, Phase},
    },
    policies::consistent_hashing::HashRing,
};

/// The sources a binding's requests take their routing key from, as a host
/// reads them ([`AppliedSpec::key_sources_for`]): in order, and the first
/// that holds a value is the key.
///
/// [`AppliedSpec::key_sources_for`]: crate::document::AppliedSpec::key_sources_for
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeySources {
    sources: Box<[Source]>,
    strip_lineage: bool,
    /// The phases whose requests the binding's profile places by a hash
    /// ring, each at its slot: the keys of such a request are read with
    /// where its key stands on its host's ring
    /// ([`DerivedKeys::ring_position`]).
    ring_phases: [bool; PHASES.len()],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Source {
    /// The header's name; `None` for the request's own id.
    header: Option<Box<str>>,
    /// The most bytes of a value this source hands.
    max_bytes: usize,
    /// Whether the key leaves in the clear: the document lets a stage read
    /// the header, or the source is the request's own id.
    plain: bool,
    /// Whether the source tells one client from another: it hands the
    /// client's key and never the session's.
    client: bool,
}

/// Where a derived key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeySource<'a> {
    /// The request header of this name.
    Header(&'a str),
    /// The request's own id.
    Rid,
}

/// A request's routing key and the source it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivedKey<'a> {
    /// The key as a decision takes it. It carries its plain form only where
    /// the document lets a stage read its source: a header `view.headers`
    /// lists, or the request's own id. A key from any other header is its
    /// fingerprint alone, so its text reaches no view, no plugin and no
    /// record.
    pub key: Identity<'a>,
    pub source: KeySource<'a>,
}

/// A request's keys by the sources its binding names
/// ([`KeySources::derive_keys`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivedKeys<'a> {
    /// Its routing key ([`KeySources::derive`]), `None` where it carries
    /// none.
    pub session: Option<DerivedKey<'a>>,
    /// Its client's key ([`KeySources::derive_client`]), read where it
    /// carries no routing key: `0` under a routing key, and where it
    /// carries none.
    pub client_key: u64,
    /// Where the key a ring places the request by stands on its host's hash
    /// ring: the ring's own hash ([`HashRing::hash_position`]) of the
    /// routing key's text as its source handed it, and of the client's
    /// value where the request carries no routing key. What the host states
    /// as the request's position
    /// ([`RequestFacts::ring_position`](crate::engine::RequestFacts::ring_position)).
    ///
    /// `0` where the request carries neither key, under a binding whose
    /// profile reads no ring, and from a host that lends no ring: nothing
    /// is hashed for such a request.
    pub ring_position: u64,
}

impl DerivedKeys<'_> {
    /// The keys of a request that carries none.
    pub const NONE: Self = Self {
        session: None,
        client_key: 0,
        ring_position: 0,
    };
}

impl KeySources {
    /// The sources of `binding`: its `routing_key` block, or the headers of
    /// its sessions rule where the rule names them. `None` where the
    /// binding names neither: its requests carry no key.
    pub(crate) fn of_binding(binding: &BindingSpec, view: &ViewSpec) -> Option<Self> {
        match (&binding.routing_key, &binding.sessions) {
            (Some(block), _) => Some(Self::of_block(block, view)),
            (None, Some(rule)) if !rule.sticky_header.is_empty() => Some(Self::of_rule(rule)),
            _ => None,
        }
    }

    fn of_block(block: &RoutingKeySpec, view: &ViewSpec) -> Self {
        let sources = block
            .sources
            .iter()
            .map(|source| Source {
                plain: source
                    .header
                    .as_ref()
                    .is_none_or(|name| view.headers.contains(name)),
                header: source.header.as_deref().map(Box::from),
                max_bytes: bytes(source.max_bytes.unwrap_or(KEY_BYTES)),
                client: source.client,
            })
            .collect();
        Self {
            sources,
            strip_lineage: block.strip_lineage,
            ring_phases: [false; PHASES.len()],
        }
    }

    /// A sessions rule's own headers: each is listed in `view.headers`, so
    /// its key leaves in the clear.
    fn of_rule(rule: &SessionsSpec) -> Self {
        let named = if rule.headers.is_empty() {
            std::slice::from_ref(&rule.sticky_header)
        } else {
            rule.headers.as_slice()
        };
        let sources = named
            .iter()
            .map(|name| Source {
                header: Some(name.as_str().into()),
                max_bytes: bytes(KEY_BYTES),
                plain: true,
                client: false,
            })
            .collect();
        Self {
            sources,
            strip_lineage: false,
            ring_phases: [false; PHASES.len()],
        }
    }

    /// These sources, for a binding whose profile places a request of a
    /// phase by a hash ring where `reads_ring` says so of the phase: the keys
    /// of such a request are read with its key's position on its host's
    /// ring ([`Self::derive_keys`]).
    #[must_use]
    pub(crate) fn with_ring_phases(mut self, reads_ring: impl Fn(Phase) -> bool) -> Self {
        self.ring_phases = PHASES.map(reads_ring);
        self
    }

    /// The headers these sources read, in order: what a host keeps of a
    /// request for [`Self::derive`].
    pub fn headers(&self) -> impl Iterator<Item = &str> {
        self.sources
            .iter()
            .filter_map(|source| source.header.as_deref())
    }

    /// Whether a source is the request's own id: a host that reads it keeps
    /// the request's body readable.
    pub fn reads_rid(&self) -> bool {
        self.sources.iter().any(|source| source.header.is_none())
    }

    /// The request's key: the first source that holds a value, with the
    /// source it came from. `header` hands a request header's value by its
    /// lowercase name, and `rid` is the request's own id.
    ///
    /// A source holds a value where it is present, not empty, UTF-8 and
    /// within the source's byte cap. Any other is passed over, never an
    /// error, and the next source is read. Under `strip_lineage` the turn
    /// and retry suffixes leave the key ([`strip_lineage`]): a header is
    /// measured as sent, and the request's id without them. `None` where no
    /// source holds a value: the request carries no key.
    pub fn derive<'a>(
        &'a self,
        header: impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
    ) -> Option<DerivedKey<'a>> {
        self.session(&header, rid).map(|(key, _)| key)
    }

    /// [`Self::derive`], with the key's text as its source handed it: the
    /// text a host's hash ring places the request by.
    fn session<'a>(
        &'a self,
        header: &impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
    ) -> Option<(DerivedKey<'a>, &'a str)> {
        let lineage = |key| {
            if self.strip_lineage {
                strip_lineage(key)
            } else {
                key
            }
        };
        let mut session = self.sources.iter().filter(|source| !source.client);
        session.find_map(|source| {
            let (key, from) = match &source.header {
                Some(name) => {
                    let value = header(name)?;
                    if value.is_empty() || value.len() > source.max_bytes {
                        return None;
                    }
                    let value = std::str::from_utf8(value).ok()?;
                    (lineage(value), KeySource::Header(name))
                }
                // The suffixes are the gateway's own marks on an id: the id
                // is measured as the key it hands.
                None => {
                    let key = lineage(rid?);
                    if key.is_empty() || key.len() > source.max_bytes {
                        return None;
                    }
                    (key, KeySource::Rid)
                }
            };
            let derived = DerivedKey {
                key: if source.plain {
                    Identity::plain(key)
                } else {
                    Identity::hashed(fingerprint(key.as_bytes()))
                },
                source: from,
            };
            Some((derived, key))
        })
    }

    /// The request's client key: the fingerprint of the value under the
    /// first client source's header the request carries, `0` where it
    /// carries none of them or that value cannot be handed. A value is
    /// handed where it is not empty and is visible ASCII, as a header's
    /// text is. It is read whole, with any suffix, and no source behind it
    /// is read. Its text leaves in no form.
    pub fn derive_client<'a>(&self, header: impl Fn(&str) -> Option<&'a [u8]>) -> u64 {
        self.client(&header)
            .map_or(0, |value| fingerprint(value.as_bytes()))
    }

    /// The value [`Self::derive_client`] takes the client's key of, where
    /// it can be handed: visible ASCII, and so text.
    fn client<'a>(&self, header: &impl Fn(&str) -> Option<&'a [u8]>) -> Option<&'a str> {
        self.sources
            .iter()
            .filter(|source| source.client)
            .find_map(|source| header(source.header.as_deref()?))
            .filter(|value| {
                !value.is_empty()
                    && value
                        .iter()
                        .all(|&byte| matches!(byte, b'\t' | b' '..=b'~'))
            })
            .and_then(|value| std::str::from_utf8(value).ok())
    }

    /// The request's keys, by one read of the sources: its routing key
    /// ([`Self::derive`]), its client's key where it carries no routing
    /// key ([`Self::derive_client`]), and where the key a ring places it by
    /// stands on its host's hash ring ([`DerivedKeys::ring_position`]).
    ///
    /// The position is the ring's own hash of the key's text, the one its
    /// host's classic policy takes of the same text, so the key has one
    /// owner by either. It is taken where the host lends its own hash ring
    /// to the request's decision, and `ring_phase` is then the request's phase,
    /// under a binding whose profile places a request of that phase by a
    /// ring. It is taken for no other request: a host that keeps no ring
    /// hashes nothing here.
    pub fn derive_keys<'a>(
        &'a self,
        header: impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
        ring_phase: Option<Phase>,
    ) -> DerivedKeys<'a> {
        let on_ring = ring_phase.is_some_and(|phase| self.ring_phases[slot(phase)]);
        // A position of zero says none is stated: the one key in 2^64 that
        // hashes there stands at the next position, which the ring's walk
        // starts from all the same.
        let position = |text: &str| {
            if on_ring {
                HashRing::hash_position(text).max(1)
            } else {
                0
            }
        };
        match self.session(&header, rid) {
            Some((key, text)) => DerivedKeys {
                session: Some(key),
                client_key: 0,
                ring_position: position(text),
            },
            None => {
                let client = self.client(&header);
                DerivedKeys {
                    client_key: client.map_or(0, |value| fingerprint(value.as_bytes())),
                    ring_position: client.map_or(0, position),
                    ..DerivedKeys::NONE
                }
            }
        }
    }
}

/// A document's byte count as a length.
fn bytes(count: u32) -> usize {
    usize::try_from(count).unwrap_or(usize::MAX)
}

/// A key without its lineage: one trailing `_r<n>` retry suffix leaves, then
/// one trailing `_t<n>` turn suffix, so every turn and retry of a
/// conversation shares one key (`conv_t2_r1` keys as `conv`). A key that is
/// nothing but suffix keys as itself. A suffix's counter has any number of
/// digits, as the gateway's own strip takes it.
pub fn strip_lineage(key: &str) -> &str {
    let stripped = strip_suffix(strip_suffix(key, b'r'), b't');
    if stripped.is_empty() {
        key
    } else {
        stripped
    }
}

/// `key` without one trailing `_<tag><digits>`.
fn strip_suffix(key: &str, tag: u8) -> &str {
    let Some((head, tail)) = key.rsplit_once('_') else {
        return key;
    };
    match tail.as_bytes() {
        [first, digits @ ..]
            if *first == tag && !digits.is_empty() && digits.iter().all(u8::is_ascii_digit) =>
        {
            head
        }
        _ => key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::spec::PolicySpec;

    /// The sources of the one binding of a document with this `routing_key`
    /// block, under a view that lists `x-smg-routing-key` alone.
    fn sources(block: &str) -> KeySources {
        let document: PolicySpec = serde_json::from_str(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{{"headers":["x-smg-routing-key"]}},"profiles":{{"p":{{"pick":"best"}}}},"bindings":[{{"profile":"p","routing_key":{block}}}]}}"#
        ))
        .expect("a well-formed document");
        document.validate().expect("the document validates");
        let binding = document.bindings.first().expect("the binding");
        KeySources::of_binding(binding, &document.view).expect("the binding's sources")
    }

    /// A request's headers as a host hands them.
    fn request<'a>(headers: &'a [(&'a str, &'a [u8])]) -> impl Fn(&str) -> Option<&'a [u8]> {
        move |name| {
            headers
                .iter()
                .find(|(listed, _)| *listed == name)
                .map(|(_, value)| *value)
        }
    }

    const ORDERED: &str = r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"x-forwarded-for"},{"header":"cookie","max_bytes":8192}]}"#;

    #[test]
    fn the_first_source_that_holds_a_value_is_the_key() {
        let sources = sources(ORDERED);
        let all: [(&str, &[u8]); 3] = [
            ("cookie", b"session=abc"),
            ("x-forwarded-for", b"10.0.0.1"),
            ("x-smg-routing-key", b"conv-1"),
        ];
        let derived = sources.derive(request(&all), Some("rid-9")).expect("a key");
        assert_eq!(
            (derived.key, derived.source),
            (
                Identity::plain("conv-1"),
                KeySource::Header("x-smg-routing-key")
            )
        );
        // No routing key header: the request's own id.
        let derived = sources
            .derive(request(&all[..2]), Some("rid-9"))
            .expect("a key");
        assert_eq!(
            (derived.key, derived.source),
            (Identity::plain("rid-9"), KeySource::Rid)
        );
        // Neither: the next header, which the view does not list.
        let derived = sources.derive(request(&all[..2]), None).expect("a key");
        assert_eq!(derived.source, KeySource::Header("x-forwarded-for"));
        assert_eq!(derived.key, Identity::hashed(fingerprint(b"10.0.0.1")));
        // No source holds a value: the request carries no key.
        assert_eq!(sources.derive(request(&[]), None), None);
        assert_eq!(
            sources.headers().collect::<Vec<_>>(),
            ["x-smg-routing-key", "x-forwarded-for", "cookie"]
        );
        assert!(sources.reads_rid());
    }

    /// An empty value, one past its source's byte cap and one that is not
    /// UTF-8 are passed over, and the next source is read.
    #[test]
    fn a_value_a_source_cannot_hand_is_passed_over() {
        let sources = sources(ORDERED);
        let long = [b'k'; 129];
        for unusable in [&b""[..], &long[..], &[0xff, 0xfe][..]] {
            let headers: [(&str, &[u8]); 1] = [("x-smg-routing-key", unusable)];
            let derived = sources
                .derive(request(&headers), Some("rid-9"))
                .expect("a key");
            assert_eq!(derived.source, KeySource::Rid, "{unusable:?}");
        }
        // At the cap the value is the key, and an entry's cap is its own.
        let headers: [(&str, &[u8]); 1] = [("x-smg-routing-key", &long[..128])];
        assert_eq!(
            sources.derive(request(&headers), None).map(|d| d.source),
            Some(KeySource::Header("x-smg-routing-key"))
        );
        let cookie = [b'c'; 8192];
        let headers: [(&str, &[u8]); 1] = [("cookie", &cookie[..])];
        assert_eq!(
            sources.derive(request(&headers), None).map(|d| d.source),
            Some(KeySource::Header("cookie"))
        );
        // The request's own id is held to its cap too.
        let rid = "r".repeat(129);
        assert_eq!(sources.derive(request(&[]), Some(rid.as_str())), None);
        assert_eq!(sources.derive(request(&[]), Some("")), None);
    }

    /// Under `strip_lineage` the turn and retry suffixes leave a key from
    /// any source, and a key that is nothing but suffix keys as itself.
    #[test]
    fn the_lineage_leaves_a_key_where_the_block_says_so() {
        for (key, stripped) in [
            ("conv_t2", "conv"),
            ("conv_t2_r1", "conv"),
            ("conv_r3", "conv"),
            ("conv_t2_t3", "conv_t2"),
            ("_t1", "_t1"),
            ("conv_t", "conv_t"),
            ("conv_tx", "conv_tx"),
            ("conv", "conv"),
        ] {
            assert_eq!(strip_lineage(key), stripped, "{key}");
        }
        // A suffix's counter has any number of digits, as the gateway's own
        // strip takes it: padded turns of a conversation share its key.
        let padded = "9".repeat(21);
        assert_eq!(strip_lineage(&format!("conv_t{padded}")), "conv");
        assert_eq!(strip_lineage(&format!("conv_t2_r{padded}")), "conv");
        let strips = sources(
            r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true}],"strip_lineage":true}"#,
        );
        let keeps = sources(r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true}]}"#);
        let headers: [(&str, &[u8]); 1] = [("x-smg-routing-key", b"conv_t2_r1")];
        assert_eq!(
            strips.derive(request(&headers), None).map(|d| d.key),
            Some(Identity::plain("conv"))
        );
        assert_eq!(
            keeps.derive(request(&headers), None).map(|d| d.key),
            Some(Identity::plain("conv_t2_r1"))
        );
        assert_eq!(
            strips.derive(request(&[]), Some("rid_t4")).map(|d| d.key),
            Some(Identity::plain("rid"))
        );
        // A header is measured as sent and the request's id as the key it
        // hands: at the cap with a suffix behind it, the header is passed
        // over and the id keys.
        let base = "k".repeat(128);
        let suffixed = format!("{base}_t2");
        let headers: [(&str, &[u8]); 1] = [("x-smg-routing-key", suffixed.as_bytes())];
        assert_eq!(strips.derive(request(&headers), None), None);
        assert_eq!(
            strips.derive(request(&[]), Some(&suffixed)).map(|d| d.key),
            Some(Identity::plain(&base))
        );
        assert_eq!(keeps.derive(request(&[]), Some(&suffixed)), None);
        let over = format!("{base}k_t2");
        assert_eq!(strips.derive(request(&[]), Some(&over)), None);
    }

    /// A client source hands the client's key and never the session's. The
    /// first of them the request carries is the one read: a value it cannot
    /// hand ends the read, and the value is read whole.
    #[test]
    fn a_client_source_hands_the_clients_key_alone() {
        let sources = sources(
            r#"{"sources":[{"header":"x-smg-routing-key"},{"header":"x-forwarded-for","client":true},{"header":"cookie","client":true}],"strip_lineage":true}"#,
        );
        let cookie = "c".repeat(10_000);
        let both: [(&str, &[u8]); 2] = [
            ("cookie", cookie.as_bytes()),
            ("x-forwarded-for", b"10.0.0.1_t2"),
        ];
        assert_eq!(sources.derive(request(&both), None), None, "no session key");
        assert_eq!(
            sources.derive_client(request(&both)),
            fingerprint(b"10.0.0.1_t2"),
            "the first client header, whole"
        );
        assert_eq!(
            sources.derive_client(request(&both[..1])),
            fingerprint(cookie.as_bytes()),
            "the next one where the first is absent, at any length"
        );
        for unusable in [&b""[..], &[0xc3, 0xa9][..]] {
            let present: [(&str, &[u8]); 2] =
                [("cookie", b"session=abc"), ("x-forwarded-for", unusable)];
            assert_eq!(sources.derive_client(request(&present)), 0, "{unusable:?}");
        }
        assert_eq!(sources.derive_client(request(&[])), 0);
        assert_eq!(
            sources.headers().collect::<Vec<_>>(),
            ["x-smg-routing-key", "x-forwarded-for", "cookie"]
        );
    }

    /// The keys of a request are its two keys, as each is derived alone,
    /// and where the key a ring places it by stands on its host's ring: the
    /// ring's own hash of the routing key's text as its source handed it,
    /// with no lineage where the block strips it, and of the client's value
    /// where the request carries no routing key. The position is taken on a
    /// ring the host lends, under a binding on a ring, and for no other
    /// request.
    #[test]
    fn the_keys_of_a_request_state_where_its_key_stands_on_its_hosts_ring() {
        let block = r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"x-forwarded-for","client":true},{"header":"cookie","client":true}],"strip_lineage":true}"#;
        let on_ring = sources(block).with_ring_phases(|phase| phase == Phase::Single);
        let single = Some(Phase::Single);
        let stands = |text: &str| HashRing::hash_position(text);
        let keyed: [(&str, &[u8]); 2] = [
            ("x-forwarded-for", b"10.0.0.1"),
            ("x-smg-routing-key", b"conv-1_t2_r1"),
        ];
        let keys = on_ring.derive_keys(request(&keyed), Some("rid-9"), single);
        assert_eq!(keys.session, on_ring.derive(request(&keyed), Some("rid-9")));
        assert_eq!((keys.client_key, keys.ring_position), (0, stands("conv-1")));
        // No routing key header: the request's own id, without its lineage.
        let keys = on_ring.derive_keys(request(&keyed[..1]), Some("rid-9_r3"), single);
        assert_eq!(keys.session.map(|key| key.source), Some(KeySource::Rid));
        assert_eq!(keys.ring_position, stands("rid-9"));
        // No session key: the client's value, whole.
        let keys = on_ring.derive_keys(request(&keyed[..1]), None, single);
        assert_eq!(keys.session, None);
        assert_eq!(keys.client_key, fingerprint(b"10.0.0.1"));
        assert_eq!(keys.ring_position, stands("10.0.0.1"));
        // No key at all: no position.
        assert_eq!(
            on_ring.derive_keys(request(&[]), None, single),
            DerivedKeys::NONE
        );
        // A host that lends no ring, a phase the binding places by none,
        // and a binding on no ring: the two keys, and nothing hashed for a
        // position.
        let off_ring = sources(block);
        for (sources, lent) in [
            (&on_ring, None),
            (&on_ring, Some(Phase::Encode)),
            (&off_ring, single),
        ] {
            let keys = sources.derive_keys(request(&keyed), None, lent);
            assert_eq!(keys.session, on_ring.derive(request(&keyed), None));
            assert_eq!(keys.ring_position, 0);
            let keys = sources.derive_keys(request(&keyed[..1]), None, lent);
            assert_eq!(keys.client_key, fingerprint(b"10.0.0.1"));
            assert_eq!(keys.ring_position, 0);
        }
    }

    /// A key from a header the view does not list leaves as its fingerprint
    /// alone. `cookie` is on no allow-list, so its text reaches no view; and
    /// with no plain form, a plugin's frame, a record and a log line carry
    /// the fingerprint. A header the document lets a stage read, and the
    /// request's own id, leave in the clear.
    #[test]
    fn a_key_from_an_unlisted_header_leaves_as_its_fingerprint_alone() {
        let sources = sources(ORDERED);
        let secret = "session=abc123; theme=dark";
        let headers: [(&str, &[u8]); 1] = [("cookie", secret.as_bytes())];
        let derived = sources.derive(request(&headers), None).expect("a key");
        assert_eq!(derived.source, KeySource::Header("cookie"));
        assert_eq!(derived.key.plain, None, "no plain form");
        assert_eq!(derived.key.hash, fingerprint(secret.as_bytes()));
        let shown = format!("{derived:?}");
        assert!(!shown.contains("abc123"), "{shown}");
        let listed: [(&str, &[u8]); 1] = [("x-smg-routing-key", b"conv-1")];
        assert_eq!(
            sources.derive(request(&listed), None).map(|d| d.key.plain),
            Some(Some("conv-1"))
        );
        assert_eq!(
            sources
                .derive(request(&[]), Some("rid-9"))
                .map(|d| d.key.plain),
            Some(Some("rid-9"))
        );
    }
}

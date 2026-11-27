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
//! A caller that lends its hash ring to a decision says where the request
//! stands on it ([`RequestFacts::ring_positions`]). The key's text is read
//! here and leaves in no other form, so its position is taken here, with
//! the two keys, by one read of the sources ([`KeySources::derive_keys`]).
//!
//! A source can name the one worker a request goes to, by a number that
//! counts the workers of the request's list. It hands no key: its number
//! is read apart ([`KeySources::target`]), by the one call that places a
//! request in process.
//!
//! [`RequestFacts::client_key`]: crate::engine::RequestFacts::client_key
//! [`RequestFacts::ring_positions`]: crate::engine::RequestFacts::ring_positions
//! [`RequestFacts::session_key`]: crate::engine::RequestFacts::session_key

use crate::{
    document::spec::{BindingSpec, RoutingKeySpec, SessionsSpec, ViewSpec, KEY_BYTES},
    engine::view::{fingerprint, Identity},
    names::intern,
    policies::consistent_hashing::LentRing,
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
    /// The header of the block's target source, where it names one
    /// ([`Self::target`]): no source of a key.
    target: Option<Box<str>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Source {
    /// The header's name, kept for the life of the process: a key names
    /// its source past the document it was derived under. `None` for the
    /// request's own id.
    header: Option<&'static str>,
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
#[non_exhaustive]
pub enum KeySource<'a> {
    /// The request header of this name.
    Header(&'a str),
    /// The request's own id.
    Rid,
}

/// A request's routing key and the source it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
/// ([`KeySources::derive_keys`]). The default holds none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DerivedKeys<'a> {
    /// Its routing key ([`KeySources::derive`]), `None` where it carries
    /// none.
    pub session: Option<DerivedKey<'a>>,
    /// Its client's key ([`KeySources::derive_client`]), read where it
    /// carries no routing key: `0` under a routing key, which a ring places
    /// by ahead of it, and where it carries none.
    pub client_key: u64,
    /// Where the key a ring places the request by stands on the ring its
    /// caller lends: that ring's own hash ([`LentRing::position`]) of the
    /// routing key's text as its source handed it, and of the client's
    /// value where the request carries no routing key. Under the plain
    /// ring it is the one entry the caller states as the request's place
    /// on the ring
    /// ([`RequestFacts::ring_positions`](crate::engine::RequestFacts::ring_positions)).
    /// A caller whose own policy asks its ring for another text than the
    /// key's states that text's position itself: under the gateway's
    /// `prefix_hash` the text is sixteen hex digits of a hash of the key's
    /// bytes, which the one call leaves to the lender
    /// ([`LentRing::head_position`]).
    ///
    /// `None` where the request carries neither key, and where no ring is
    /// lent: nothing is hashed for such a request.
    pub ring_position: Option<u64>,
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
        let mut target = block.sources.iter().filter(|source| source.target);
        let target = target.find_map(|source| source.header.as_deref());
        let sources = block
            .sources
            .iter()
            .filter(|source| !source.target)
            .map(|source| Source {
                plain: source
                    .header
                    .as_ref()
                    .is_none_or(|name| view.headers.contains(name)),
                header: source.header.as_deref().map(intern),
                max_bytes: bytes(source.max_bytes.unwrap_or(KEY_BYTES)),
                client: source.client,
            })
            .collect();
        Self {
            sources,
            strip_lineage: block.strip_lineage,
            target: target.map(Box::from),
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
                header: Some(intern(name)),
                max_bytes: bytes(KEY_BYTES),
                plain: true,
                client: false,
            })
            .collect();
        Self {
            sources,
            strip_lineage: false,
            target: None,
        }
    }

    /// The headers these sources read, in order: what a host keeps of a
    /// request for [`Self::derive`].
    pub fn headers(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().filter_map(|source| source.header)
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
        &self,
        header: impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
    ) -> Option<DerivedKey<'a>> {
        self.session(&header, rid).map(|(key, _)| key)
    }

    /// [`Self::derive`], with the key's text as its source handed it: the
    /// text a lent ring places the request by.
    fn session<'a>(
        &self,
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
            let (key, from) = match source.header {
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
        let mut clients = self.sources.iter().filter(|source| source.client);
        text(clients.find_map(|source| header(source.header?))?)
    }

    /// Whether the block names a target source: a host reads
    /// [`Self::target`] for its requests where it does.
    pub fn names_target(&self) -> bool {
        self.target.is_some()
    }

    /// The number a request names its worker by, under the block's target
    /// source: `Some(None)` where the request names a worker and its text
    /// is no number, so it reaches no worker of any list. `None` where the
    /// request names no worker: the block has no target source, or the
    /// request carries no text under its header, a value that is empty or
    /// not visible ASCII among them. A number is decimal digits, with one
    /// `+` ahead of them or none. Which workers it counts, and so which
    /// worker it reaches, is its reader's to say.
    pub fn target<'a>(&self, header: impl Fn(&str) -> Option<&'a [u8]>) -> Option<Option<u32>> {
        let named = text(header(self.target.as_deref()?)?)?;
        Some(named.parse().ok())
    }

    /// The request's keys, by one read of the sources: its routing key
    /// ([`Self::derive`]), its client's key where it carries no routing
    /// key ([`Self::derive_client`]), and where the key a ring places it by
    /// stands on `ring` ([`DerivedKeys::ring_position`]).
    ///
    /// `ring` is the ring the request's caller lends to its decision
    /// ([`RequestFacts::ring`](crate::engine::RequestFacts::ring)): the
    /// position is that ring's own hash of the key's text, so the key has
    /// the owner the caller's own walk finds where that walk starts from
    /// the key's text too. A caller hands it for a request a ring places,
    /// where
    /// [`Engine::states_key_place`](crate::engine::Engine::states_key_place)
    /// answers `true` for the request's model and phase, and `None` for any
    /// other: the ring is asked nothing then.
    pub fn derive_keys<'a>(
        &self,
        header: impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
        ring: Option<&dyn LentRing>,
    ) -> DerivedKeys<'a> {
        self.derive_keys_by(header, rid, |text, _| ring.map(|ring| ring.position(text)))
    }

    /// [`Self::derive_keys`] with the place asked of `place`: it is handed
    /// the text of the value a ring places the request by, once, with the
    /// source of a routing value and `None` for the client's value, and is
    /// not asked where the request carries neither value. The prefix form
    /// of a ring is asked where a routing value's text stands as a head
    /// does where a header handed it, and nothing of a client's value or
    /// of one the request's own id handed.
    pub(crate) fn derive_keys_by<'a>(
        &self,
        header: impl Fn(&str) -> Option<&'a [u8]>,
        rid: Option<&'a str>,
        place: impl FnOnce(&str, Option<KeySource<'a>>) -> Option<u64>,
    ) -> DerivedKeys<'a> {
        match self.session(&header, rid) {
            Some((key, text)) => DerivedKeys {
                session: Some(key),
                client_key: 0,
                ring_position: place(text, Some(key.source)),
            },
            None => {
                let client = self.client(&header);
                DerivedKeys {
                    session: None,
                    client_key: client.map_or(0, |value| fingerprint(value.as_bytes())),
                    ring_position: client.and_then(|value| place(value, None)),
                }
            }
        }
    }
}

/// A header's value as its text, where it can be handed: not empty, and
/// visible ASCII.
fn text(value: &[u8]) -> Option<&str> {
    let visible = |&byte: &u8| matches!(byte, b'\t' | b' '..=b'~');
    let value = Some(value).filter(|value| !value.is_empty() && value.iter().all(visible))?;
    std::str::from_utf8(value).ok()
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
#[path = "../../tests/unit/document/routing_key/tests.rs"]
mod tests;

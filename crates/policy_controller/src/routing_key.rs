//! A request's routing key from the sources its binding names (design 4.6).
//!
//! A binding lists where the key comes from, in order: request headers and
//! the request's own id. The host reads them and the engine reads one key
//! and no raw header ([`RequestFacts::session_key`]). The derivation is
//! one function, [`KeySources::derive`], for a host in process and for a
//! gateway that asks a controller under an address.
//!
//! [`RequestFacts::session_key`]: crate::RequestFacts::session_key

use crate::{
    spec::{BindingSpec, RoutingKeySpec, SessionsSpec, ViewSpec, KEY_BYTES},
    view::{fingerprint, Identity},
};

/// The sources a binding's requests take their routing key from, as a host
/// reads them ([`AppliedSpec::key_sources_for`]): in order, and the first
/// that holds a value is the key.
///
/// [`AppliedSpec::key_sources_for`]: crate::AppliedSpec::key_sources_for
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeySources {
    sources: Box<[Source]>,
    strip_lineage: bool,
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
            })
            .collect();
        Self {
            sources,
            strip_lineage: block.strip_lineage,
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
            })
            .collect();
        Self {
            sources,
            strip_lineage: false,
        }
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
        let lineage = |key| {
            if self.strip_lineage {
                strip_lineage(key)
            } else {
                key
            }
        };
        self.sources.iter().find_map(|source| {
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
            Some(DerivedKey {
                key: if source.plain {
                    Identity::plain(key)
                } else {
                    Identity::hashed(fingerprint(key.as_bytes()))
                },
                source: from,
            })
        })
    }
}

/// A document's byte count as a length.
fn bytes(count: u32) -> usize {
    usize::try_from(count).unwrap_or(usize::MAX)
}

/// A key without its lineage: one trailing `_r<n>` retry suffix leaves, then
/// one trailing `_t<n>` turn suffix, so every turn and retry of a
/// conversation shares one key (`conv_t2_r1` keys as `conv`). A key that is
/// nothing but suffix keys as itself.
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
    use crate::spec::PolicySpec;

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

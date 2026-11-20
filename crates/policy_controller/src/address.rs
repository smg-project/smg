//! Split an address around its userinfo without guessing where it ends.

/// What stands for an address whose userinfo cannot be told from what
/// follows it: the marker the gateway writes for an address it cannot
/// parse.
pub(crate) const UNPARSABLE: &str = "<unparsable address>";

/// An address in its parts around the userinfo ([`split`]). The userinfo
/// itself is in none of them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Address<'a> {
    /// The scheme through its `://`; empty where the address has none.
    pub(crate) scheme: &'a str,
    /// Whether a userinfo stood before the host.
    pub(crate) userinfo: bool,
    /// The host and the port.
    pub(crate) host: &'a str,
    /// What follows the authority: a path, a query, a fragment.
    pub(crate) tail: &'a str,
}

/// `address` split around its userinfo: the one place of the crate that
/// tells a credential from an address. A scheme is one by its form
/// ([`is_scheme`]). Where the text before the first `://` is none, the
/// address has no scheme and its authority starts at its first byte: a
/// `://` in a path, a query or a userinfo opens nothing. The authority ends
/// at the first `/`, `?` or `#`, and its last `@` ends the userinfo. `None`
/// for an address with an `@` past the authority's end: a userinfo that
/// holds one of the three raw reads like a path or a query that holds an
/// `@`, and a guess could tell part of the userinfo.
pub(crate) fn split(address: &str) -> Option<Address<'_>> {
    let start = address
        .split_once("://")
        .filter(|(scheme, _)| is_scheme(scheme))
        .map_or(0, |(scheme, _)| scheme.len() + "://".len());
    let (scheme, rest) = address.split_at(start);
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(end);
    if tail.contains('@') {
        return None;
    }
    let (userinfo, host) = authority
        .rsplit_once('@')
        .map_or((false, authority), |(_userinfo, host)| (true, host));
    Some(Address {
        scheme,
        userinfo,
        host,
        tail,
    })
}

/// Whether `text` is a URL's scheme by its form (RFC 3986): a letter, then
/// letters, digits, `+`, `-` or `.`.
fn is_scheme(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
}

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

/// A target source names the request's worker by a number and hands no
/// key: neither read of a key takes its header, and a host is told no
/// header of it to keep for a key. A number is decimal digits, with one
/// `+` ahead of them or none, as large as a place can be. A text that
/// is no such number names a worker that is no place of any list. A
/// header that is absent, empty or not visible ASCII names no worker,
/// and neither does any header under a block with no target source.
///
/// The gateway's own policy takes the header's value where it is text
/// and not empty (`crates/external_router/src/header_utils.rs`, lines
/// 94 to 103) and reads it as a number
/// (`model_gateway/src/policies/consistent_hashing.rs`, lines 154 and
/// 155).
#[test]
fn a_target_source_names_a_place_and_hands_no_key() {
    let block = sources(
        r#"{"sources":[{"header":"x-smg-routing-key"},{"header":"x-smg-target-worker","target":true},{"header":"cookie","client":true}]}"#,
    );
    assert!(block.names_target());
    let named = |value: &[u8]| block.target(request(&[("x-smg-target-worker", value)]));
    for (text, place) in [("0", 0), ("7", 7), ("+7", 7), ("007", 7)] {
        assert_eq!(named(text.as_bytes()), Some(Some(place)), "{text}");
    }
    assert_eq!(named(b"4294967295"), Some(Some(u32::MAX)));
    for text in ["4294967296", "-1", "1.5", " 1", "1 ", "one", "+", "0x1"] {
        assert_eq!(named(text.as_bytes()), Some(None), "{text:?}");
    }
    for unusable in [&b""[..], &[0xc3, 0xa9][..], &b"1\x00"[..]] {
        assert_eq!(named(unusable), None, "{unusable:?}");
    }
    assert_eq!(block.target(request(&[])), None, "absent");

    let only: [(&str, &[u8]); 1] = [("x-smg-target-worker", b"7")];
    assert_eq!(block.derive(request(&only), None), None);
    assert_eq!(block.derive_client(request(&only)), 0);
    assert_eq!(
        block.headers().collect::<Vec<_>>(),
        ["x-smg-routing-key", "cookie"]
    );

    let unnamed = sources(ORDERED);
    assert!(!unnamed.names_target());
    assert_eq!(unnamed.target(request(&only)), None);
}

/// A ring for these cases: a key stands at its text's fingerprint.
#[derive(Debug)]
struct Stands;

impl LentRing for Stands {
    fn position(&self, key: &str) -> u64 {
        fingerprint(key.as_bytes())
    }

    fn walk(&self, _: u64, _: &mut dyn FnMut(&str) -> bool) {}
}

/// The keys of a request are its two keys, as each is derived alone,
/// and where the key a ring places it by stands on the ring its caller
/// lends: that ring's own hash of the routing key's text as its source
/// handed it, with no lineage where the block strips it, and of the
/// client's value where the request carries no routing key. With no
/// ring lent the two keys are read and no position is taken.
///
/// The gateway's own policy asks its ring for the same text: the
/// routing key's header value, or the request id behind it, each
/// without its lineage under the override
/// (`model_gateway/src/policies/registry.rs`, lines 534 to 541 and 497
/// to 503 at 882be6f4f; `consistent_hashing.rs`, lines 170 to 174), and
/// with neither the client's `x-forwarded-for` or `cookie`, whole
/// (lines 206 to 222). Its ring hashes the text it is asked for
/// (`hash_ring.rs`, line 268).
#[test]
fn the_keys_of_a_request_state_where_its_key_stands_on_the_lent_ring() {
    let sources = sources(
        r#"{"sources":[{"header":"x-smg-routing-key"},{"rid":true},{"header":"x-forwarded-for","client":true},{"header":"cookie","client":true}],"strip_lineage":true}"#,
    );
    let lent: Option<&dyn LentRing> = Some(&Stands);
    let stands = |text: &str| Some(fingerprint(text.as_bytes()));
    let keyed: [(&str, &[u8]); 2] = [
        ("x-forwarded-for", b"10.0.0.1"),
        ("x-smg-routing-key", b"conv-1_t2_r1"),
    ];
    let keys = sources.derive_keys(request(&keyed), Some("rid-9"), lent);
    assert_eq!(keys.session, sources.derive(request(&keyed), Some("rid-9")));
    assert_eq!((keys.client_key, keys.ring_position), (0, stands("conv-1")));
    // No routing key header: the request's own id, without its lineage.
    let keys = sources.derive_keys(request(&keyed[..1]), Some("rid-9_r3"), lent);
    assert_eq!(keys.session.map(|key| key.source), Some(KeySource::Rid));
    assert_eq!(keys.ring_position, stands("rid-9"));
    // No session key: the client's value, whole.
    let keys = sources.derive_keys(request(&keyed[..1]), None, lent);
    let client = sources.derive_client(request(&keyed[..1]));
    assert_eq!((keys.session, keys.client_key), (None, client));
    assert_eq!(keys.ring_position, stands("10.0.0.1"));
    // No key at all: no position.
    let none = sources.derive_keys(request(&[]), None, lent);
    assert_eq!(
        (none.session, none.client_key, none.ring_position),
        (None, 0, None)
    );
    // No ring lent: the two keys, and nothing hashed for a position.
    let keys = sources.derive_keys(request(&keyed), None, None);
    assert_eq!(keys.session, sources.derive(request(&keyed), None));
    assert_eq!(keys.ring_position, None);
    let keys = sources.derive_keys(request(&keyed[..1]), None, None);
    assert_eq!((keys.client_key, keys.ring_position), (client, None));
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

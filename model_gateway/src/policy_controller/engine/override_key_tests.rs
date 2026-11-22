//! The sticky override's key, read twice of one request.
//!
//! The policy registry reads it for the load guard of each dispatch, which
//! counts the key's requests on the worker it is sent to
//! ([`PolicyRegistry::sticky_key`]). The engine's own ask reads it for the
//! decision ([`Ask::keys`]): the sessions rule pins it, and the rule's cap
//! reads the guard's count under it. The cap compares the right count only
//! where the two read one text, so the table below holds them to each
//! other source by source.

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use smg_policy_controller::document::{AppliedSpec, KeySource};

use super::Ask;
use crate::{
    config::{PolicyConfig, RouterConfig, RoutingKeyOverrideConfig},
    policies::PolicyRegistry,
};

/// The two headers the table's gateway reads a key from, in their order.
const FIRST: &str = "x-session-id";
const SECOND: &str = "x-smg-routing-key";
/// The source that is the request's own id.
const RID: &str = "rid";

/// What one request carries: its headers and the `rid` of its body, with
/// the key both readers hand and the source it came from, a header's name
/// or [`RID`].
struct Carried {
    said: &'static str,
    headers: Vec<(&'static str, Vec<u8>)>,
    rid: Option<String>,
    key: Option<(String, &'static str)>,
}

fn carried(
    said: &'static str,
    headers: &[(&'static str, &str)],
    rid: Option<&str>,
    key: Option<(&str, &'static str)>,
) -> Carried {
    Carried {
        said,
        headers: headers
            .iter()
            .map(|(name, value)| (*name, value.as_bytes().to_vec()))
            .collect(),
        rid: rid.map(str::to_owned),
        key: key.map(|(key, source)| (key.to_owned(), source)),
    }
}

/// Every source of the override's key, one request a line: each configured
/// header in its order, a value a source cannot hand, the turn and retry
/// suffixes, and the request's own id behind the headers.
fn table() -> Vec<Carried> {
    let at_cap = "k".repeat(128);
    let past_cap = "k".repeat(129);
    // 132 bytes as sent, 126 without its suffixes.
    let long_with_suffixes = format!("{}_t2_r1", "k".repeat(126));
    // 131 bytes as sent, 128 without its suffix.
    let id_at_cap = format!("{at_cap}_t9");
    let id_past_cap = format!("{past_cap}_t9");
    vec![
        carried(
            "the first header",
            &[(FIRST, "conv")],
            None,
            Some(("conv", FIRST)),
        ),
        carried(
            "the second header where the first is absent",
            &[(SECOND, "hint")],
            None,
            Some(("hint", SECOND)),
        ),
        carried(
            "the first header ahead of the second and of the id",
            &[(SECOND, "hint"), (FIRST, "conv")],
            Some("rid-1"),
            Some(("conv", FIRST)),
        ),
        carried(
            "a header that no setting names",
            &[("x-conversation", "other")],
            None,
            None,
        ),
        carried(
            "a turn and a retry suffix",
            &[(FIRST, "conv_t2_r1")],
            None,
            Some(("conv", FIRST)),
        ),
        carried(
            "a retry suffix alone",
            &[(FIRST, "conv_r3")],
            None,
            Some(("conv", FIRST)),
        ),
        carried(
            "a turn suffix alone",
            &[(SECOND, "conv_t12")],
            None,
            Some(("conv", SECOND)),
        ),
        carried(
            "one turn suffix leaves, and one alone",
            &[(FIRST, "conv_t1_t2")],
            None,
            Some(("conv_t1", FIRST)),
        ),
        carried(
            "a retry suffix ahead of a turn suffix is no lineage",
            &[(FIRST, "conv_r1_t2")],
            None,
            Some(("conv_r1", FIRST)),
        ),
        carried(
            "a suffix with no counter stays",
            &[(FIRST, "conv_t")],
            None,
            Some(("conv_t", FIRST)),
        ),
        carried(
            "a key that is nothing but suffix keys as itself",
            &[(FIRST, "_t2_r1")],
            None,
            Some(("_t2_r1", FIRST)),
        ),
        carried(
            "an empty header is passed over",
            &[(FIRST, ""), (SECOND, "hint")],
            None,
            Some(("hint", SECOND)),
        ),
        carried(
            "a header at the byte cap",
            &[(FIRST, at_cap.as_str())],
            None,
            Some((at_cap.as_str(), FIRST)),
        ),
        carried(
            "a header past the byte cap is passed over",
            &[(FIRST, past_cap.as_str())],
            Some("rid-1"),
            Some(("rid-1", RID)),
        ),
        carried(
            "a header is measured as sent, with its suffixes",
            &[(FIRST, long_with_suffixes.as_str())],
            None,
            None,
        ),
        Carried {
            said: "a header that is not UTF-8 is passed over",
            headers: vec![(FIRST, vec![0xff, 0xfe]), (SECOND, b"hint_t1".to_vec())],
            rid: None,
            key: Some(("hint".to_owned(), SECOND)),
        },
        carried(
            "the id where no header holds a value",
            &[],
            Some("talk"),
            Some(("talk", RID)),
        ),
        carried(
            "the id's turn and retry suffixes",
            &[],
            Some("talk_t2_r1"),
            Some(("talk", RID)),
        ),
        carried(
            "an id that is nothing but suffix keys as itself",
            &[],
            Some("_r1"),
            Some(("_r1", RID)),
        ),
        carried(
            "an id is measured without its suffixes",
            &[],
            Some(id_at_cap.as_str()),
            Some((at_cap.as_str(), RID)),
        ),
        carried(
            "an id past the byte cap",
            &[],
            Some(id_past_cap.as_str()),
            None,
        ),
        carried("an empty id", &[], Some(""), None),
        carried("no source", &[], None, None),
    ]
}

/// The registry's key of a request and the engine's are one text from one
/// source, for every line of the table: the guard counts a key's requests
/// under the text the sessions rule's cap reads them by. The engine's key
/// carries its text whatever its source, a header's as the view lists the
/// header, so the cap has a count to read.
#[test]
fn the_registry_and_the_engine_read_one_key_of_a_request_source_by_source() {
    let overridden = RoutingKeyOverrideConfig {
        enabled: true,
        headers: vec![FIRST.to_owned(), SECOND.to_owned()],
        ..RoutingKeyOverrideConfig::default()
    };
    let config = RouterConfig {
        policy_controller: true,
        policy: PolicyConfig::RoundRobin,
        routing_key_override: overridden.clone(),
        ..RouterConfig::default()
    };
    let document = config.policy_document().expect("the settings' document");
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let registry = PolicyRegistry::with_override(PolicyConfig::RoundRobin, overridden);

    for line in table() {
        let mut headers = HeaderMap::new();
        for (name, value) in &line.headers {
            headers.insert(
                HeaderName::from_static(name),
                HeaderValue::from_bytes(value).expect("a header's value"),
            );
        }
        let rid = line.rid.as_deref();
        let expected = line
            .key
            .as_ref()
            .map(|(key, source)| (key.as_str(), *source));

        // The registry's: a header's key, else the key of the id. It names
        // no header, so its source reads as a header or the id.
        let by_header = registry.sticky_header_key(Some(&headers));
        let by_registry = registry
            .sticky_key(Some(&headers), registry.derive_rid_key(rid))
            .map(|key| (key, by_header.is_none()));
        assert_eq!(
            by_registry,
            expected.map(|(key, source)| (key, source == RID)),
            "the registry: {}",
            line.said
        );

        // The engine's: the first of the document's sources that holds a
        // value, with its text.
        let ask = Ask {
            headers: Some(&headers),
            rid,
            ..Ask::of(b"request", "a-model")
        };
        let (derived, client) = ask.keys(&spec);
        assert_eq!(client, 0, "{}", line.said);
        let by_engine = derived.map(|derived| {
            let source = match derived.source {
                KeySource::Header(name) => name,
                KeySource::Rid => RID,
            };
            (derived.key.plain, source)
        });
        assert_eq!(
            by_engine,
            expected.map(|(key, source)| (Some(key), source)),
            "the engine: {}",
            line.said
        );
    }
}

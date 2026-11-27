//! Cases of the routing key's settings: where the configured headers of a
//! key land, and what the sticky override writes.

use super::*;

#[test]
fn the_headers_of_a_routing_key_land_where_the_policy_reads_its_key() {
    let keyed = |policy: &str, headers: &[&str]| {
        let mut keyed = config(at_flag_defaults(policy));
        keyed.routing_key_override.headers =
            headers.iter().map(|name| (*name).to_string()).collect();
        keyed
    };

    // The manual policy reads the configured headers, then the routing
    // key header. Its rule names them: the first is the sticky one, and
    // the rule lists them where there is more than one, each once and
    // lowercase.
    let both = |document: &mut Value| {
        document["view"]["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
        let sessions = &mut document["bindings"][0]["sessions"];
        sessions["sticky_header"] = json!("x-session-id");
        sessions["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
    };
    let listed = ["X-Session-Id", "x-smg-routing-key", "x-session-id"];
    assert_eq!(
        keyed("manual", &listed).policy_document(),
        Ok(edited(preset("manual"), both))
    );
    assert_eq!(
        keyed("manual", &["x-session-id"]).policy_document(),
        Ok(edited(preset("manual"), both)),
        "the routing key header is read behind a configured one that is not it"
    );
    assert_eq!(
        keyed("manual", &[]).policy_document(),
        Ok(preset("manual")),
        "with no header configured the manual policy still reads the routing key header"
    );

    // The prefix ring reads the configured headers alone, and none is no
    // block. The hash ring reads the routing key header behind them.
    let sources = |name: &str, sources: Value| {
        edited(preset(name), |document| {
            document["bindings"][0]["routing_key"] = json!({ "sources": sources });
        })
    };
    assert_eq!(
        keyed("prefix_hash", &["x-session-id", "X-Other-Key"]).policy_document(),
        Ok(sources(
            "prefix_hash",
            json!([{ "header": "x-session-id" }, { "header": "x-other-key" }])
        ))
    );
    assert_eq!(
        keyed("prefix_hash", &[]).policy_document(),
        Ok(edited(preset("prefix_hash"), |document| {
            document["bindings"][0]["routing_key"] = Value::Null;
        }))
    );
    assert_eq!(
        keyed("consistent_hashing", &["x-session-id"]).policy_document(),
        Ok(sources(
            "consistent_hashing",
            json!([
                { "header": "x-session-id" },
                { "header": "x-smg-routing-key" },
                { "header": "x-forwarded-for", "client": true },
                { "header": "cookie", "client": true },
                { "header": "x-smg-target-worker", "target": true }
            ])
        ))
    );

    // Eight sources bound a block. The manual policy's rule holds the
    // configured headers, bounded by the view: 256 with the routing key
    // header.
    let eight: Vec<String> = (0..8).map(|n| format!("x-key-{n}")).collect();
    let eight: Vec<&str> = eight.iter().map(String::as_str).collect();
    assert_eq!(
        refusal(&keyed("consistent_hashing", &eight)).0,
        "routing_key_override.headers"
    );
    assert!(keyed("manual", &eight).policy_document().is_ok());
    // The hash ring's two client sources are counted apart: seven
    // configured headers and the routing key header are its eight.
    let seven = keyed("consistent_hashing", &eight[..7]);
    let written = seven.policy_document().expect("eight sources");
    assert!(validated(&written).is_ok());

    // The hash ring reads a configured `x-forwarded-for` both ways,
    // within the cap as a session's key and whole as a client's. The
    // document names it twice, and the controller's check takes it.
    let both_ways = keyed("consistent_hashing", &["x-forwarded-for"]);
    let document = sources(
        "consistent_hashing",
        json!([
            { "header": "x-forwarded-for" },
            { "header": "x-smg-routing-key" },
            { "header": "x-forwarded-for", "client": true },
            { "header": "cookie", "client": true },
            { "header": "x-smg-target-worker", "target": true }
        ]),
    );
    assert_eq!(both_ways.policy_document().as_ref(), Ok(&document));
    assert!(validated(&document).is_ok());

    // No other policy reads a key, so its document names no source.
    let keyed_policies = ["manual", "prefix_hash", "consistent_hashing"];
    for name in PRESETS
        .iter()
        .filter(|name| !keyed_policies.contains(*name))
    {
        assert_eq!(
            keyed(name, &["x-session-id"]).policy_document(),
            Ok(preset(name)),
            "{name}"
        );
    }
}

#[test]
fn the_override_adds_the_requests_id_where_a_policy_keeps_its_own_key() {
    let overridden = |policy: &str| {
        let mut overridden = config(at_flag_defaults(policy));
        overridden.routing_key_override.enabled = true;
        overridden
    };

    // The hash ring keeps its own placement under the override. Its
    // sources gain the request's id behind the headers, and every source
    // loses its turn and retry suffixes.
    assert_eq!(
        overridden("consistent_hashing").policy_document(),
        Ok(edited(preset("consistent_hashing"), |document| {
            document["bindings"][0]["routing_key"] = json!({
                "sources": [
                    { "header": "x-smg-routing-key" },
                    { "rid": true },
                    { "header": "x-forwarded-for", "client": true },
                    { "header": "cookie", "client": true },
                    { "header": "x-smg-target-worker", "target": true }
                ],
                "strip_lineage": true
            });
        }))
    );

    // The manual policy's rule says headers alone. The id and the strip
    // take the binding's block, and the rule then names no header.
    assert_eq!(
        overridden("manual").policy_document(),
        Ok(edited(preset("manual"), |document| {
            let binding = &mut document["bindings"][0];
            binding["routing_key"] = json!({
                "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                "strip_lineage": true
            });
            binding["sessions"]["sticky_header"] = json!("");
        }))
    );

    // Behind other configured headers the gateway reads the routing key
    // header as sent, and a block strips every source or none.
    let mut behind = overridden("consistent_hashing");
    behind.routing_key_override.headers = vec!["x-session-id".to_string()];
    assert_eq!(
        refusal(&behind),
        (
            "routing_key_override.headers",
            r#"["x-session-id"]"#.to_string()
        )
    );

    // The encode side reads no key and the override never places it.
    let mut plan = RoutingSettings {
        mode: ModeSettings::EncodePrefillDecode {
            encode_policy: Some(PolicySettings::RoundRobin),
            prefill_policy: Some(PolicySettings::ConsistentHashing),
            decode_policy: Some(PolicySettings::ConsistentHashing),
        },
        ..config(PolicySettings::RoundRobin)
    };
    plan.routing_key_override.enabled = true;
    let sources = plan.policy_document().map(|document| {
        document
            .bindings
            .first()
            .and_then(|binding| binding.routing_key.as_ref())
            .map(|key| key.sources.len())
    });
    assert_eq!(sources, Ok(Some(5)));
}

/// Beside a policy that reads no key the override lands as a session
/// rule on the policy's own binding: the gateway's sticky map in the
/// document's words. The key is the configured headers, then the
/// request's id, each without its turn and retry suffixes, and the view
/// lists the headers, so the rule's cap reads the key's text. The
/// controller's check takes the document, and its engine asks its host
/// for the count of a key's requests on a worker.
#[test]
fn the_override_lands_as_a_session_rule_beside_a_policy_that_reads_no_key() {
    for name in PLACED {
        let expected = edited(preset(name), |document| {
            document["view"]["headers"] = json!(["x-smg-routing-key"]);
            let binding = &mut document["bindings"][0];
            binding["routing_key"] = json!({
                "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                "strip_lineage": true
            });
            binding["sessions"] = json!({
                "ttl_s": 14_400,
                "sweep_s": 60,
                "assignment": "delegate",
                "inflight_cap": 2,
                "recovery": "latest",
                "rule_placed": "untold"
            });
        });
        assert_eq!(sticky(name).policy_document().as_ref(), Ok(&expected));
        let applied = validated(&expected).expect("the controller takes the document");
        assert!(applied.engine().needs().session_in_flight, "{name}");
    }
}

/// Beside the prefix ring the override lands as the same session rule
/// over the same sources: the configured headers, then the request's
/// own id, each without its turn and retry suffixes. The ring reads a
/// key too. A header's key leads the prompt, as under the policy alone:
/// the gateway hands its policy that key without its suffixes
/// (`model_gateway/src/policies/registry.rs:646`). The key the
/// request's own id hands names the session alone: the gateway's policy
/// never hashes it (`model_gateway/src/policies/prefix_hash.rs:384`).
/// With no header configured the id is the one source.
#[test]
fn the_override_lands_as_a_session_rule_beside_the_prefix_ring() {
    let expected = edited(preset("prefix_hash"), |document| {
        document["view"]["headers"] = json!(["x-smg-routing-key"]);
        let binding = &mut document["bindings"][0];
        binding["routing_key"] = json!({
            "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
            "strip_lineage": true
        });
        binding["sessions"] = json!({
            "ttl_s": 14_400,
            "sweep_s": 60,
            "assignment": "delegate",
            "inflight_cap": 2,
            "recovery": "latest",
            "rule_placed": "untold"
        });
    });
    let settings = sticky("prefix_hash");
    assert_eq!(settings.policy_document().as_ref(), Ok(&expected));
    let applied = validated(&expected).expect("the controller takes the document");
    assert!(applied.engine().needs().session_in_flight);

    let mut bare = sticky("prefix_hash");
    bare.routing_key_override.headers.clear();
    let document = bare.policy_document().expect("a document");
    assert!(validated(&document).is_ok(), "{document:?}");
    let binding = document.bindings.into_iter().next().expect("the binding");
    let key = binding.routing_key.expect("the key's sources");
    assert_eq!(key.sources.len(), 1);
    assert!(key.sources[0].rid && key.strip_lineage);
    assert!(document.view.headers.is_empty());
}

/// The override's rule stands in the way of one pick. `passthrough`
/// picks alone, so the document's answer for its place is one pick
/// under the settings as they default. Under the override the answer
/// names the rule: no caller is told to pick by itself past a pinned
/// key.
#[test]
fn the_overrides_rule_keeps_a_place_from_one_pick() {
    let place_of = |settings: RoutingSettings| {
        let document = settings.policy_document().expect("a document");
        let applied = validated(&document).expect("the controller takes the document");
        let place = applied.engine().place_for("", Phase::Single);
        let ruled = Place::NotAlone {
            profile: "passthrough",
            in_the_way: InTheWay::Sessions,
        };
        (matches!(place, Place::Alone(_)), place == ruled)
    };
    let plain = config(at_flag_defaults("passthrough"));
    assert_eq!(place_of(plain), (true, false));
    assert_eq!(place_of(sticky("passthrough")), (false, true));
}

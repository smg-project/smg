use super::*;
use crate::{
    document::spec::tests::{doc, spec_with_binding},
    engine::view::Phase,
};

/// A sessions rule runs under the single plan, and on the two sides of
/// a pair, under a cap or under none. A cap under the pair makes the
/// document's engine read a worker's count of a key's requests, the
/// schema's default cap as one that is written. A rule that keeps what
/// it places from the profile runs under a pair in the build of a host
/// that makes the lifecycle's calls, where each side of a pick states
/// its own word. In the build of a host that makes some only it is
/// refused by that field's name, and the refusal says why: that host
/// tells a lease's words off a file, which keeps no word of a pair's
/// side.
///
/// Under `epd` the rule runs on the pair's two sides, under either word,
/// in the build of a host that makes the lifecycle's calls. A cap makes
/// the document's engine read a key's count as under a pair. The build of
/// a host that makes some only takes the rule that tells its stages, with
/// a cap or with none. It refuses the one that keeps what it places from
/// the profile, by that field's name, and its reason names the `epd` plan.
///
/// A gateway's sticky map places each role of a pair under that role's
/// own key, and hands its policy no request it placed
/// (`model_gateway/src/policies/registry.rs:762-769`, `:794-797`).
#[test]
fn a_sessions_rule_runs_under_a_pair_and_under_epd_by_its_hosts_calls() {
    let built = |plan: &str, profiles: &str, sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"{plan}","profiles":{profiles},"sessions":{sessions}}}"#
        ))
        .validate()
    };
    let under = |plan: &str, profiles: &str, sessions: &str| built(plan, profiles, sessions).err();
    let pair = r#"{"prefill":"q","decode":"q"}"#;
    let uncapped = r#"{"sticky_header":"x-session-id","inflight_cap":65535}"#;
    // Whether the pair's engine reads the count of a key's requests.
    let counted = |sessions: &str| {
        built("pd-pair", pair, sessions).map(|built| built.needs().session_in_flight)
    };
    assert_eq!(counted(uncapped), Ok(false));
    for capped in [
        r#"{"sticky_header":"x-session-id"}"#,
        r#"{"sticky_header":"x-session-id","inflight_cap":4}"#,
    ] {
        assert_eq!(counted(capped), Ok(true), "{capped}");
    }
    let untold = r#"{"sticky_header":"x-session-id","inflight_cap":65535,"rule_placed":"untold"}"#;
    assert_eq!(under("pd-pair", pair, untold), None);
    let some = StageRegistry::builtin().without_lifecycle();
    let on_file = |sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"pd-pair","profiles":{pair},"sessions":{sessions}}}"#
        ))
        .validate_in(&some, &[])
        .err()
    };
    assert_eq!(on_file(uncapped), None, "a rule that tells its stages");
    let refused = on_file(untold);
    assert_eq!(
        refused,
        Some(SpecError::BadSessions {
            index: 0,
            name: "rule_placed",
            reason: UNTOLD_ON_FILE,
        })
    );
    assert_eq!(
        refused.map(|err| err.to_string()),
        Some(format!(
            "binding 0: sessions rule_placed is not {UNTOLD_ON_FILE}"
        ))
    );
    assert!(UNTOLD_ON_FILE.ends_with("a file of leases keeps no side's word"));
    // The single plan takes each of the three, under either host.
    for rule in [uncapped, untold, r#"{"sticky_header":"x-session-id"}"#] {
        let single = spec_with_binding(&format!(r#"{{"profile":"q","sessions":{rule}}}"#));
        assert_eq!(single.validate().err(), None, "{rule}");
        assert_eq!(single.validate_in(&some, &[]).err(), None, "{rule}");
    }
    let epd = r#"{"encode":"q","prefill":"q","decode":"q"}"#;
    let capped = r#"{"sticky_header":"x-session-id"}"#;
    for rule in [uncapped, untold, capped] {
        assert_eq!(under("epd", epd, rule), None, "{rule}");
    }
    // The build of a host that tells a lease's words off a file: whether
    // its engine reads the count of a key's requests under `epd`.
    let epd_on_file = |sessions: &str| {
        spec_with_binding(&format!(
            r#"{{"plan":"epd","profiles":{epd},"sessions":{sessions}}}"#
        ))
        .validate_in(&some, &[])
        .map(|built| built.needs().session_in_flight)
    };
    assert_eq!(
        epd_on_file(uncapped),
        Ok(false),
        "a rule that tells its stages"
    );
    assert_eq!(epd_on_file(capped), Ok(true), "a cap reads the count");
    let refused = epd_on_file(untold).err();
    assert_eq!(
        refused,
        Some(SpecError::BadSessions {
            index: 0,
            name: "rule_placed",
            reason: UNTOLD_ON_FILE_EPD,
        })
    );
    let path = refused.as_ref().map(SpecError::field_path);
    assert_eq!(path.as_deref(), Some("bindings[0].sessions.rule_placed"));
    // Each plan's refusal as an operator reads it: the pair's words stand
    // as they stood, and the other names its own plan.
    assert_eq!(
        refused.map(|err| err.to_string()),
        Some(
            "binding 0: sessions rule_placed is not `told` under the epd plan in this build: \
             a file of leases keeps no side's word"
                .to_string()
        )
    );
    assert_eq!(
        UNTOLD_ON_FILE,
        "`told` under the pd-pair plan in this build: a file of leases keeps no side's word"
    );
    let counts =
        |sessions: &str| built("epd", epd, sessions).map(|built| built.needs().session_in_flight);
    assert_eq!((counts(uncapped), counts(capped)), (Ok(false), Ok(true)));
    // The rule's needs are the pair's two sides': the encode phase
    // takes none of them, by any of the engine's three answers.
    let ruled = built("epd", epd, capped).expect("a rule under epd");
    let bare = spec_with_binding(&format!(r#"{{"plan":"epd","profiles":{epd}}}"#))
        .validate()
        .expect("the plan alone");
    for side in [Phase::Prefill, Phase::Decode] {
        assert!(ruled.needs_for_phase(side).session_in_flight, "{side:?}");
        assert!(!bare.needs_for_phase(side).session_in_flight, "{side:?}");
    }
    let of_encode = |engine: &Engine| {
        let rows: Vec<Needs> = engine
            .needs_rows()
            .into_iter()
            .filter(|row| row.1 == Phase::Encode)
            .map(|row| row.2)
            .collect();
        (
            engine.needs_for_phase(Phase::Encode),
            engine.reads_for("m", Phase::Encode),
            rows,
        )
    };
    assert_eq!(of_encode(&ruled), of_encode(&bare));
}

/// A sessions rule under a pair names the sides it covers (`sides`):
/// `prefill`, `decode` or both, each once and in either order. Left
/// out, the rule covers both sides, and a document that names none
/// writes no `sides` key.
///
/// The check refuses, each by the field's path: a list under the single
/// plan or under `epd`, an empty list and a side named twice. It refuses
/// the encode side and every other name too. A list that leaves out a
/// side whose picker leaves the placement to the rule is not in this
/// build, by the path of that side's profile.
///
/// A gateway's sticky map places a role at a time. It asks the role's
/// policy, and leaves the role to one that keeps a routing key's
/// handling to itself: `manual` and `consistent_hashing` (`sticky_for`
/// in `model_gateway/src/policies/registry.rs`).
#[test]
fn a_sessions_rule_names_the_sides_of_a_pair_it_covers() {
    // One binding under a rule, beside the profile `q` and the profile
    // `sticky`, whose picker leaves the placement to the rule.
    let bound = |binding: &str, sides: &str| {
        doc(&format!(
            r#"{{"apiVersion":"smg.policy/v1","view":{{"headers":["x-session-id"]}},"profiles":{{"q":{{"pick":"random"}},"sticky":{{"pick":"manual"}}}},"bindings":[{{{binding},"sessions":{{"sticky_header":"x-session-id"{sides}}}}}]}}"#
        ))
    };
    let pair = |prefill: &str, decode: &str| {
        format!(r#""plan":"pd-pair","profiles":{{"prefill":"{prefill}","decode":"{decode}"}}"#)
    };
    let named = |sides: &str| format!(r#","sides":{sides}"#);
    let plain = pair("q", "q");
    let covered = |sides: &str| {
        let built = bound(&plain, sides).validate();
        let built = built.expect("the document validates");
        let binding = built.binding("m", Phase::Prefill).expect("the binding");
        binding.sessions.as_ref().expect("the rule").sides
    };
    assert_eq!(covered(""), SessionSides::Any);
    // The type's own default is that rule: one built in code that names
    // no side covers both.
    assert_eq!(SessionSides::default(), SessionSides::Any);
    for (sides, rule) in [
        (r#"["prefill"]"#, SessionSides::Prefill),
        (r#"["decode"]"#, SessionSides::Decode),
        (r#"["prefill","decode"]"#, SessionSides::Any),
        (r#"["decode","prefill"]"#, SessionSides::Any),
    ] {
        assert_eq!(covered(&named(sides)), rule, "{sides}");
    }
    // The rule as a document writes it.
    let written = |sides: &str| {
        let document = serde_json::to_value(bound(&plain, sides)).expect("a document");
        document["bindings"][0]["sessions"].clone()
    };
    assert_eq!(written("").get("sides"), None);
    let decode = written(&named(r#"["decode"]"#));
    assert_eq!(decode["sides"], serde_json::json!(["decode"]));
    // A `null` is the field left out, under the single plan too.
    let single = bound(r#""profile":"q""#, &named("null")).validate();
    assert!(single.is_ok(), "{:?}", single.err());

    // Under `epd` a rule covers the pair's two sides, and takes no list.
    let epd = r#""plan":"epd","profiles":{"encode":"q","prefill":"q","decode":"q"}"#;
    assert_eq!(bound(epd, "").validate().err(), None);
    let (by_prefill, by_decode) = (pair("sticky", "q"), pair("q", "sticky"));
    let other = "`prefill`, `decode` or both: a rule covers no other side";
    for (binding, sides, reason) in [
        (
            r#""profile":"q""#,
            r#"["prefill"]"#,
            "left out under the single plan: a single request has one side",
        ),
        (
            epd,
            r#"["decode"]"#,
            "left out under the epd plan in this build: a rule covers the pair's two sides there",
        ),
        (
            plain.as_str(),
            "[]",
            "a list of one side at least: leave it out for both",
        ),
        (
            plain.as_str(),
            r#"["decode","decode"]"#,
            "a list that names each side once",
        ),
        (
            plain.as_str(),
            r#"["prefill","prefill"]"#,
            "a list that names each side once",
        ),
        // A third item is read as the first two are.
        (
            plain.as_str(),
            r#"["prefill","decode","decode"]"#,
            "a list that names each side once",
        ),
        (plain.as_str(), r#"["encode"]"#, other),
        (plain.as_str(), r#"["prefill","both"]"#, other),
        // A name is read as written, with a space before it, behind it
        // or both.
        (plain.as_str(), r#"[" decode"]"#, other),
        (plain.as_str(), r#"["decode "]"#, other),
        (plain.as_str(), r#"[" decode "]"#, other),
    ] {
        let refused = bound(binding, &named(sides)).validate().err();
        let by_name = SpecError::BadSessions {
            index: 0,
            name: "sides",
            reason,
        };
        assert_eq!(refused.as_ref(), Some(&by_name), "{sides}");
        assert_eq!(by_name.field_path(), "bindings[0].sessions.sides");
        assert_eq!(
            by_name.to_string(),
            format!("binding 0: sessions sides is not {reason}")
        );
    }
    // A refusal names its own binding: the second of two here.
    let second = doc(&format!(
        r#"{{"apiVersion":"smg.policy/v1","view":{{"headers":["x-session-id"]}},"profiles":{{"q":{{"pick":"random"}}}},"bindings":[{{"match":{{"model":"other"}},"profile":"q"}},{{{plain},"sessions":{{"sticky_header":"x-session-id","sides":[]}}}}]}}"#
    ));
    let by_name = SpecError::BadSessions {
        index: 1,
        name: "sides",
        reason: "a list of one side at least: leave it out for both",
    };
    assert_eq!(second.validate().err().as_ref(), Some(&by_name));
    assert_eq!(by_name.field_path(), "bindings[1].sessions.sides");
    // A side whose picker leaves the placement to the rule is not left
    // out: the binding's one rule would not place it, and it holds no
    // second.
    for (binding, sides, side) in [
        (&by_prefill, r#"["decode"]"#, "prefill"),
        (&by_decode, r#"["prefill"]"#, "decode"),
    ] {
        let lacking = SpecError::NotInBuild {
            path: format!("bindings[0].profiles.{side}"),
            what: BY_RULE_ON_A_SIDE_LEFT_OUT,
        };
        let refused = bound(binding, &named(sides)).validate().err();
        assert_eq!(refused, Some(lacking), "{sides}");
    }
    assert_eq!(
        BY_RULE_ON_A_SIDE_LEFT_OUT,
        "a picker that places by the sessions rule on a side the rule's `sessions.sides` leaves out",
    );
    // Such a side stands under a rule that names it, alone or beside
    // the other side, and under one that names none.
    for (binding, sides) in [
        (&by_prefill, named(r#"["prefill"]"#)),
        (&by_decode, named(r#"["decode"]"#)),
        (&by_decode, named(r#"["decode","prefill"]"#)),
        (&by_decode, String::new()),
    ] {
        assert_eq!(bound(binding, &sides).validate().err(), None, "{sides}");
    }
}

/// An index key, with a count of the times it is read.
struct Read<'a>(&'a str, &'a std::cell::Cell<usize>);

impl AsRef<str> for Read<'_> {
    fn as_ref(&self) -> &str {
        self.1.set(self.1.get() + 1);
        self.0
    }
}

/// The overlap pass reads each index key once, whatever their number: a
/// thousand keys are a thousand reads and eight thousand are eight
/// thousand. A pass over every pair reads them by the square.
#[test]
fn the_overlap_pass_reads_each_index_key_once() {
    for entries in [1_000, 8_000] {
        // A prefix and a suffix by turns, of two literal lengths: no two
        // overlap, so no pass ends early.
        let keys: Vec<String> = (0..entries)
            .map(|n| {
                if n % 2 == 0 {
                    format!("p{n:05}*")
                } else {
                    format!("*{n:05}ss")
                }
            })
            .collect();
        let reads = std::cell::Cell::new(0);
        let found = overlap(keys.iter().map(|key| Read(key, &reads)));
        assert!(found.is_none(), "{entries} keys of no overlap");
        assert_eq!(reads.get(), entries, "{entries} keys");
    }
}

/// The pass over every pair, as the check ran before: the measure of
/// the pair the overlap pass names.
fn pairwise<'a>(keys: &[&'a str]) -> Option<(&'a str, &'a str)> {
    let ambiguous = |a: &str, b: &str| matches!((side(a), side(b)), (Some((sa, la)), Some((sb, lb))) if sa != sb && la == lb);
    keys.iter().enumerate().find_map(|(place, first)| {
        let second = keys[place + 1..]
            .iter()
            .find(|second| ambiguous(first, second))?;
        Some((*first, *second))
    })
}

/// The next of a draw's numbers, under `bound`.
fn below(state: &mut u64, bound: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    usize::try_from((*state >> 33) % bound).expect("under the bound")
}

/// The pair refused is the pair a pass over every pair names: the first
/// key with a later partner, and its first partner. Four thousand key
/// sets by draw, of ids, prefixes, suffixes and `*`, in the map's order.
#[test]
fn the_overlap_named_is_the_pair_a_pass_over_every_pair_names() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let (mut overlapping, mut distinct) = (0, 0);
    for _ in 0..4_000 {
        let mut keys = BTreeSet::new();
        for _ in 0..below(&mut state, 12) {
            let literal: String = (0..below(&mut state, 4))
                .map(|_| ["a", "b"][below(&mut state, 2)])
                .collect();
            keys.insert(match below(&mut state, 4) {
                0 => format!("{literal}*"),
                1 => format!("*{literal}"),
                2 => "*".to_owned(),
                _ => format!("{literal}x"),
            });
        }
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let named = overlap(keys.iter().copied());
        assert_eq!(named, pairwise(&keys), "{keys:?}");
        if named.is_some() {
            overlapping += 1;
        } else {
            distinct += 1;
        }
    }
    assert!(
        overlapping > 500 && distinct > 500,
        "{overlapping} sets overlap, {distinct} do not"
    );
}

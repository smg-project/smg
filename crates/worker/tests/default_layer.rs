//! The default layer of this crate stays below the protocol crate.
//!
//! `openai-protocol` re-exports this crate's modules at the paths they had
//! there, so it depends on this crate, and cargo refuses a dependency cycle
//! even when an optional dependency closes it. A consumer that needs the
//! worker's types alone must also not be handed the protocol crate through
//! them. So nothing this crate links by default may link `openai-protocol`,
//! and nothing a feature of it links may either. The default layer itself is
//! held too: no feature on, and the six crates the types need.
//! The case reads the workspace's manifests through `cargo metadata` and
//! walks what the default features link.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{cargo_metadata, this_package, Outcome};
use serde_json::{json, Value};

const THIS_CRATE: &str = "smg-worker";
const PROTOCOL_CRATE: &str = "openai-protocol";

/// What the walk needs of one workspace member.
struct Member {
    /// Every dependency a build of it can link: normal and build edges,
    /// optional or not.
    linked: Vec<String>,
    /// The dependencies its default features link: the ones that are not
    /// optional, and the optional ones `default` switches on.
    default_layer: Vec<String>,
}

/// The optional dependencies the `default` feature switches on, directly or
/// through the features it names.
fn default_optionals(features: &Value) -> BTreeSet<String> {
    let mut optionals = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut queue = vec!["default".to_owned()];
    while let Some(feature) = queue.pop() {
        if !seen.insert(feature.clone()) {
            continue;
        }
        let Some(entries) = features[feature.as_str()].as_array() else {
            continue;
        };
        for entry in entries.iter().filter_map(Value::as_str) {
            if let Some(dependency) = entry.strip_prefix("dep:") {
                optionals.insert(dependency.to_owned());
            } else if let Some((dependency, _)) = entry.split_once('/') {
                // `name/feature` switches the dependency on, `name?/feature` does not.
                if !dependency.ends_with('?') {
                    optionals.insert(dependency.to_owned());
                }
            } else {
                // Another feature of the crate, or the implicit feature of an optional dependency.
                optionals.insert(entry.to_owned());
                queue.push(entry.to_owned());
            }
        }
    }
    optionals
}

/// The workspace's members by name, from `cargo metadata --no-deps`.
fn members(metadata: &Value) -> Outcome<BTreeMap<String, Member>> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or("the metadata lists no packages")?;
    let mut out = BTreeMap::new();
    for package in packages {
        let name = package["name"].as_str().ok_or("a package has no name")?;
        let dependencies = package["dependencies"]
            .as_array()
            .ok_or("a package lists no dependencies")?;
        let switched_on = default_optionals(&package["features"]);
        let mut member = Member {
            linked: Vec::new(),
            default_layer: Vec::new(),
        };
        for dependency in dependencies {
            // A dev-dependency is linked into tests alone, and cargo allows a cycle through one.
            if dependency["kind"].as_str() == Some("dev") {
                continue;
            }
            let dependency_name = dependency["name"]
                .as_str()
                .ok_or("a dependency has no name")?;
            // A feature names an optional dependency by its key in the manifest.
            let key = dependency["rename"].as_str().unwrap_or(dependency_name);
            member.linked.push(dependency_name.to_owned());
            if dependency["optional"].as_bool() != Some(true) || switched_on.contains(key) {
                member.default_layer.push(dependency_name.to_owned());
            }
        }
        out.insert(name.to_owned(), member);
    }
    Ok(out)
}

/// The chain of crates from this crate's default layer to the protocol
/// crate, when one exists. The walk stops at a crate from the registry: its
/// manifest is not among the workspace's, and it cannot name a workspace
/// crate by path.
fn chain_to_protocol(members: &BTreeMap<String, Member>) -> Outcome<Option<Vec<String>>> {
    chain_from(members, |this| &this.default_layer)
}

/// [`chain_to_protocol`] from everything a build of this crate can link,
/// whatever features it switches on: the edges of `runtime` too.
fn chain_to_protocol_under_any_feature(
    members: &BTreeMap<String, Member>,
) -> Outcome<Option<Vec<String>>> {
    chain_from(members, |this| &this.linked)
}

/// The walk from the first hops of this crate that `first_hops` picks.
fn chain_from(
    members: &BTreeMap<String, Member>,
    first_hops: fn(&Member) -> &Vec<String>,
) -> Outcome<Option<Vec<String>>> {
    let this = members
        .get(THIS_CRATE)
        .ok_or("the workspace has no smg-worker")?;
    if !members.contains_key(PROTOCOL_CRATE) {
        return Err("the workspace has no openai-protocol".into());
    }
    let mut seen = BTreeSet::new();
    let mut stack: Vec<Vec<String>> = first_hops(this)
        .iter()
        .map(|first| vec![THIS_CRATE.to_owned(), first.clone()])
        .collect();
    while let Some(chain) = stack.pop() {
        let Some(last) = chain.last() else {
            continue;
        };
        if last == PROTOCOL_CRATE {
            return Ok(Some(chain));
        }
        if !seen.insert(last.clone()) {
            continue;
        }
        let Some(member) = members.get(last) else {
            continue;
        };
        for next in &member.linked {
            let mut longer = chain.clone();
            longer.push(next.clone());
            stack.push(longer);
        }
    }
    Ok(None)
}

#[test]
fn the_default_layer_links_nothing_that_links_the_protocol_crate() -> Outcome<()> {
    assert_eq!(chain_to_protocol(&members(&cargo_metadata()?)?)?, None);
    Ok(())
}

/// The runtime lives in this crate behind `runtime`: what a feature links may
/// not link the protocol crate either, or the two crates close a cycle.
#[test]
fn no_feature_links_anything_that_links_the_protocol_crate() -> Outcome<()> {
    assert_eq!(
        chain_to_protocol_under_any_feature(&members(&cargo_metadata()?)?)?,
        None
    );
    Ok(())
}

/// The types alone: the default feature set switches nothing on, and the
/// default layer links six crates and no other. A consumer of the types gains
/// a package only when this case changes.
#[test]
fn the_default_feature_set_is_empty_and_its_layer_is_six_crates() -> Outcome<()> {
    let metadata = cargo_metadata()?;
    assert_eq!(this_package(&metadata)?["features"]["default"], json!([]));
    let mut layer = members(&metadata)?
        .remove(THIS_CRATE)
        .ok_or("the workspace has no smg-worker")?
        .default_layer;
    layer.sort();
    assert_eq!(
        layer,
        [
            "bitflags",
            "schemars",
            "serde",
            "serde_json",
            "serde_with",
            "url"
        ]
    );
    Ok(())
}

/// The walk on a workspace made in code: it names the chain when an edge
/// reaches the protocol crate, it reads an optional edge of this crate as the
/// default layer's only when `default` switches it on, and past the first hop
/// it follows every edge, optional or not.
#[test]
fn the_walk_names_the_chain_that_reaches_the_protocol_crate() -> Outcome<()> {
    let edge = |name: &str, optional: bool| json!({ "name": name, "kind": null, "optional": optional, "rename": null });
    let workspace = |worker_edge: Value, worker_features: Value, middle_edge: Value| {
        json!({ "packages": [
            { "name": THIS_CRATE, "dependencies": [worker_edge], "features": worker_features },
            { "name": "middle", "dependencies": [middle_edge], "features": {} },
            { "name": PROTOCOL_CRATE, "dependencies": [edge(THIS_CRATE, false)], "features": {} },
        ] })
    };
    let chain = Some(vec![
        THIS_CRATE.to_owned(),
        "middle".to_owned(),
        PROTOCOL_CRATE.to_owned(),
    ]);

    let always = workspace(
        edge("middle", false),
        json!({}),
        edge(PROTOCOL_CRATE, false),
    );
    assert_eq!(chain_to_protocol(&members(&always)?)?, chain);

    let behind_a_feature = workspace(
        edge("middle", true),
        json!({ "default": [], "runtime": ["dep:middle"] }),
        edge(PROTOCOL_CRATE, false),
    );
    assert_eq!(chain_to_protocol(&members(&behind_a_feature)?)?, None);
    assert_eq!(
        chain_to_protocol_under_any_feature(&members(&behind_a_feature)?)?,
        chain
    );

    let on_by_default = workspace(
        edge("middle", true),
        json!({ "default": ["runtime"], "runtime": ["dep:middle"] }),
        edge(PROTOCOL_CRATE, false),
    );
    assert_eq!(chain_to_protocol(&members(&on_by_default)?)?, chain);

    // Past the first hop every edge counts: a build of the middle crate can
    // switch an optional one on.
    let optional_later = workspace(edge("middle", false), json!({}), edge(PROTOCOL_CRATE, true));
    assert_eq!(chain_to_protocol(&members(&optional_later)?)?, chain);
    Ok(())
}

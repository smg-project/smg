//! Grand Central's two wire contracts, compiled and parsed at test
//! time (`contracts/README.md`): the gRPC protocol in package `smg.policy.v1`
//! and the WebAssembly world `smg:policy@1.2.0`. Nothing at run time reads
//! the files; the world's `types` interface is rendered from the crate's
//! types by the `wit` module and held to the committed file by
//! `tests/plugin/wit.rs`. The tests here hold four things: the
//! files compile; the view, decision and lifecycle messages and records
//! mirror this crate's types field for field, in the crate's order; the
//! versioned names a plugin author builds against (the package, the RPCs and
//! their streaming shapes, the `oneof` cases, the worlds, the interfaces and
//! their functions) cannot move without a change to the lists below; and the
//! packed request block's padding keeps its u64 sections 8-byte aligned, as
//! the world's comment states.
//!
//! The protocol compiles through `protoc` on the path (or named by `PROTOC`),
//! the compiler the repository's generated gRPC clients already need.
//!
//! One module a wire holds that wire's own cases, each a file under
//! `contracts_cases/`: `protocol.rs` and `world.rs`. This file holds the case
//! of the header deny-list and the case of the versioned names with what they
//! alone read, the lists of the crate's records and enums, and what more
//! than one of the three files reads. The shared `World` type stays whole:
//! three of its methods serve `world.rs` alone.

use std::{
    io,
    path::{Path, PathBuf},
};

use prost_types::{
    DescriptorProto, FileDescriptorProto, FileDescriptorSet, ServiceDescriptorProto,
};
use smg_grand_central::{
    cases,
    document::DENIED_HEADERS,
    engine::{Phase, Role, ShedReason},
    mirrored,
    tracking::Side,
};
use wit_parser::{InterfaceId, PackageId, Resolve, TypeDefKind, WorldId, WorldItem, WorldKey};

#[path = "contracts_cases/protocol.rs"]
mod protocol;
#[path = "contracts_cases/world.rs"]
mod world;

const PLUGIN_PROTO: &str = "policy_plugin.proto";
const CONTROLLER_PROTO: &str = "grand_central.proto";
const WORLD_WIT: &str = "smg-policy.wit";
const PACKAGE: &str = "smg.policy.v1";
const WIT_NAMESPACE: &str = "smg";
const WIT_PACKAGE: &str = "policy";
const WIT_VERSION: &str = "1.2.0";

/// The header deny-list of design section 4.6: the names no view may read
/// and no routing-key override may key on, lowercase, in this order, each
/// once. A name added or moved here is a contract change.
const DENIED_HEADER_NAMES: [&str; 7] = [
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
];

#[test]
fn the_header_deny_list_holds_its_seven_names() {
    assert_eq!(DENIED_HEADERS, DENIED_HEADER_NAMES, "the deny-list moved");
    for name in DENIED_HEADERS {
        assert_eq!(name, name.to_ascii_lowercase(), "{name} is not lowercase");
    }
}

// A record's field names and an enum's case names are listed by the
// library's two proof macros (`record!`, `cases!`, its module `mirrored`):
// a pattern that names every field without `..` and a match without a
// wildcard arm, so a type cannot gain, lose or rename one without this file
// failing to compile. No such pattern of a type marked non-exhaustive
// builds outside the library: the lists of those types are stated there, by
// the same macros, and read here (`mirrored::outcome` and its like).

fn candidate_record() -> &'static [&'static str] {
    mirrored::candidate()
}

/// A candidate's affinity facts: a column beside the rows in the crate, five
/// fields of the candidate's own record on both wires, at the numbers they
/// were given as fields of the row.
fn affinity_record() -> &'static [&'static str] {
    mirrored::affinity_facts()
}

fn request_record() -> &'static [&'static str] {
    mirrored::request_facts()
}

fn load_row_record() -> &'static [&'static str] {
    mirrored::load_row()
}

fn constraints_record() -> &'static [&'static str] {
    mirrored::constraints()
}

fn pool_record() -> &'static [&'static str] {
    mirrored::pool_facts()
}

fn zone_record() -> &'static [&'static str] {
    mirrored::zone_capacity()
}

fn media_record() -> &'static [&'static str] {
    mirrored::media_counts()
}

fn health_record() -> &'static [&'static str] {
    mirrored::health()
}

fn target_record() -> &'static [&'static str] {
    mirrored::target()
}

fn pair_record() -> &'static [&'static str] {
    mirrored::pair()
}

fn shed_record() -> &'static [&'static str] {
    mirrored::shed()
}

fn queue_record() -> &'static [&'static str] {
    mirrored::queue()
}

fn trace_record() -> &'static [&'static str] {
    mirrored::trace()
}

/// The lease's session crosses under the name both contracts still spell,
/// `session_key`. A gateway's client reads it under that name. A change that
/// renames the field renames it in both contracts and takes this pair out.
fn lease_record() -> &'static [&'static str] {
    mirrored::lease()
}

fn outcome_record() -> &'static [&'static str] {
    mirrored::outcome()
}

/// The crate's enums the two contracts spell out, with the proto enum (a
/// top-level name, or `Message.Nested`) and its value prefix, and the name
/// of the WIT enum or variant.
fn enum_mirrors() -> Vec<(
    &'static str,
    &'static str,
    &'static str,
    &'static [&'static str],
)> {
    vec![
        (
            "Role",
            "ROLE",
            "role",
            cases!(
                Role,
                Role {
                    Single,
                    Prefill,
                    Decode,
                    Encode
                }
            ),
        ),
        (
            "Phase",
            "PHASE",
            "phase",
            cases!(
                Phase,
                Phase {
                    Single,
                    Prefill,
                    Decode,
                    Encode
                }
            ),
        ),
        (
            "Side",
            "SIDE",
            "side",
            cases!(
                Side,
                Side {
                    Single,
                    Prefill,
                    Decode,
                    Encode
                }
            ),
        ),
        ("HashKind", "HASH_KIND", "hash-kind", mirrored::hash_kind()),
        (
            "SignalSource",
            "SIGNAL_SOURCE",
            "signal-source",
            mirrored::signal_source(),
        ),
        (
            "OverloadRegime",
            "OVERLOAD_REGIME",
            "overload-regime",
            mirrored::overload_regime(),
        ),
        (
            "LeaseKind",
            "LEASE_KIND",
            "lease-kind",
            mirrored::lease_kind(),
        ),
        (
            "Completed.Class",
            "CLASS",
            "transport-class",
            mirrored::transport_class(),
        ),
        (
            "Released.Reason",
            "REASON",
            "release-reason",
            mirrored::release_reason(),
        ),
        (
            "MembershipChange.Kind",
            "KIND",
            "membership",
            mirrored::membership(),
        ),
    ]
}

/// The shed reasons have a proto enum and no WIT counterpart (a guest never
/// sheds; the host does).
fn shed_reasons() -> &'static [&'static str] {
    cases!(
        ShedReason,
        ShedReason {
            Saturated,
            FlowLimit,
            NoEndpoints,
            PolicyDeclined,
            Deadline
        }
    )
}

/// The signal kinds have a WIT variant; the proto spells them as a `oneof`
/// whose third case names its unit.
fn signal_cases() -> &'static [&'static str] {
    mirrored::signal()
}

/// The services, each RPC with its request, its answer and its streaming
/// shape (client, server). A later version adds a row; it never changes or
/// removes one.
type Rpc = (&'static str, &'static str, &'static str, bool, bool);

const PLUGIN_RPCS: &[Rpc] = &[
    ("Decide", "PluginRequest", "PluginResponse", true, true),
    ("Observe", "ObserveRequest", "Ack", true, false),
];

const CONTROLLER_RPCS: &[Rpc] = &[
    ("Session", "GatewayFrame", "ControllerFrame", true, true),
    ("Decide", "DecideRequest", "DecideAnswer", true, true),
    (
        "WatchWorkers",
        "WatchWorkersRequest",
        "WorkerEvent",
        false,
        true,
    ),
    (
        "RegisterWorker",
        "RegisterWorkerRequest",
        "WorkerEvent",
        false,
        false,
    ),
    (
        "PatchWorker",
        "PatchWorkerRequest",
        "WorkerEvent",
        false,
        false,
    ),
    (
        "DeregisterWorker",
        "DeregisterWorkerRequest",
        "WorkerEvent",
        false,
        false,
    ),
    ("GetSnapshot", "SnapshotRequest", "Snapshot", false, false),
    ("Explain", "ExplainRequest", "DecisionRecord", false, false),
    (
        "ApplyPolicySpec",
        "ApplyRequest",
        "ApplyResult",
        false,
        false,
    ),
    (
        "GetPolicySpec",
        "GetPolicySpecRequest",
        "PolicySpecDocument",
        false,
        false,
    ),
    (
        "RollbackPolicySpec",
        "RollbackRequest",
        "ApplyResult",
        false,
        false,
    ),
    ("Leader", "LeaderRequest", "LeaderInfo", false, false),
    ("StepDown", "StepDownRequest", "Ack", false, false),
    ("WatchRegime", "RegimeRequest", "Regime", false, true),
    ("Replicate", "ReplicaFrame", "ReplicationBatch", true, true),
];

/// A `oneof` only grows: the cases of every frame and answer, in order.
const ONEOFS: &[(&str, &str, &str, &[&str])] = &[
    (PLUGIN_PROTO, "Identity", "form", &["hash", "plain"]),
    (
        PLUGIN_PROTO,
        "Decision",
        "outcome",
        &["pick", "shed", "queue"],
    ),
    (PLUGIN_PROTO, "Released", "what", &["lease_id", "ticket"]),
    (
        PLUGIN_PROTO,
        "Signal",
        "kind",
        &["load", "health", "gateway_veto_micros"],
    ),
    (
        PLUGIN_PROTO,
        "LifecycleEvent",
        "event",
        &[
            "issued",
            "dispatched",
            "completed",
            "released",
            "membership",
            "signal",
        ],
    ),
    (
        PLUGIN_PROTO,
        "PluginRequest",
        "request",
        &["hello", "decide"],
    ),
    (
        PLUGIN_PROTO,
        "ObserveRequest",
        "request",
        &["hello", "event"],
    ),
    (
        PLUGIN_PROTO,
        "PluginResponse",
        "response",
        &["hello", "decision"],
    ),
    (
        CONTROLLER_PROTO,
        "GatewayFrame",
        "frame",
        &[
            "hello",
            "hello_page",
            "dispatched",
            "completed",
            "released",
            "veto",
            "spec_ack",
        ],
    ),
    (
        CONTROLLER_PROTO,
        "ControllerFrame",
        "frame",
        &["welcome", "not_leader", "preempt", "spec"],
    ),
    (
        CONTROLLER_PROTO,
        "DecideAnswer",
        "answer",
        &["decision", "busy", "credit"],
    ),
    (
        CONTROLLER_PROTO,
        "WorkerEvent",
        "event",
        &[
            "snapshot",
            "registered",
            "replaced",
            "removed",
            "status",
            "relabelled",
            "needs",
            "snapshot_required",
            "ranks",
        ],
    ),
    (
        CONTROLLER_PROTO,
        "RankRow",
        "made",
        &["registered", "replaced", "removed", "relabelled"],
    ),
    (CONTROLLER_PROTO, "ReplicaFrame", "frame", &["hello", "ack"]),
    (
        CONTROLLER_PROTO,
        "ReplicationEvent",
        "event",
        &[
            "membership",
            "health",
            "load",
            "pin",
            "reservation",
            "generation",
            "ids",
            "lease_end",
            "term_start",
        ],
    ),
    (
        CONTROLLER_PROTO,
        "ExplainRequest",
        "key",
        &["decision_id", "request_id"],
    ),
    (
        CONTROLLER_PROTO,
        "Reachability",
        "scope",
        &["any", "gateway_id"],
    ),
];

/// The worlds with their imports and exports, by interface name. Every
/// import is an interface of this package: a module that needs anything else
/// (a `wasi:` interface) is refused by the engine.
const WORLDS: &[(&str, &[&str], &[&str])] = &[
    ("policy", &["host", "types"], &["plugin"]),
    ("policy-packed", &["host", "types"], &["plugin-packed"]),
];
const INTERFACES: &[&str] = &["types", "host", "plugin", "plugin-packed"];
const HOST_FUNCTIONS: &[&str] = &["log", "now-micros", "counter", "random-u64"];
const PLUGIN_FUNCTIONS: &[&str] = &[
    "init",
    "[constructor]decision",
    "[method]decision.filter",
    "[method]decision.score",
    "[method]decision.pick",
    "observe",
];

fn contracts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts")
}

fn compile_protocol() -> io::Result<FileDescriptorSet> {
    let dir = contracts_dir();
    let mut config = prost_build::Config::new();
    // proto3 `optional` needs the flag below protoc 3.15, the oldest the
    // repository's build scripts run under.
    config.protoc_arg("--experimental_allow_proto3_optional");
    config.load_fds(
        &[dir.join(PLUGIN_PROTO), dir.join(CONTROLLER_PROTO)],
        &[dir],
    )
}

fn file<'a>(set: &'a FileDescriptorSet, name: &str) -> Option<&'a FileDescriptorProto> {
    set.file.iter().find(|file| file.name() == name)
}

fn message<'a>(file: &'a FileDescriptorProto, name: &str) -> Option<&'a DescriptorProto> {
    file.message_type
        .iter()
        .find(|message| message.name() == name)
}

fn service<'a>(file: &'a FileDescriptorProto, name: &str) -> Option<&'a ServiceDescriptorProto> {
    file.service.iter().find(|service| service.name() == name)
}

/// The cases of a named `oneof`, in declaration order.
fn oneof_cases(message: &DescriptorProto, oneof: &str) -> Option<Vec<String>> {
    let index = message
        .oneof_decl
        .iter()
        .position(|decl| decl.name() == oneof)?;
    let index = i32::try_from(index).ok()?;
    Some(
        message
            .field
            .iter()
            .filter(|field| field.oneof_index == Some(index))
            .map(|field| field.name().to_owned())
            .collect(),
    )
}

/// `BreakerOpen` to `breaker-open`.
fn kebab_case(camel: &str) -> String {
    let mut out = String::new();
    for (i, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('-');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

struct World {
    resolve: Resolve,
    package: PackageId,
}

fn parse_world() -> Result<World, String> {
    let mut resolve = Resolve::new();
    let (package, _sources) = resolve
        .push_path(contracts_dir().join(WORLD_WIT))
        .map_err(|err| format!("{err:#}"))?;
    Ok(World { resolve, package })
}

impl World {
    fn interface(&self, name: &str) -> Option<InterfaceId> {
        self.resolve.packages[self.package]
            .interfaces
            .get(name)
            .copied()
    }

    fn world(&self, name: &str) -> Option<WorldId> {
        self.resolve.packages[self.package]
            .worlds
            .get(name)
            .copied()
    }

    /// The interface names behind a world's imports or exports: every item
    /// is an interface of this package (a function or a type under the same
    /// key is not), an interface key names the item it carries, and a named
    /// item is the package's interface of that name.
    fn interface_names<'a>(
        &self,
        items: impl Iterator<Item = (&'a WorldKey, &'a WorldItem)>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for (key, item) in items {
            assert!(
                matches!(item, WorldItem::Interface { .. }),
                "world item {key:?} is not an interface"
            );
            let WorldItem::Interface { id, .. } = item else {
                continue;
            };
            let interface = &self.resolve.interfaces[*id];
            assert_eq!(
                interface.package,
                Some(self.package),
                "a world item outside the package"
            );
            let name = match key {
                WorldKey::Interface(key_id) => {
                    assert_eq!(key_id, id, "an interface key naming another item");
                    interface.name.clone().unwrap_or_default()
                }
                WorldKey::Name(name) => {
                    assert_eq!(
                        self.interface(name),
                        Some(*id),
                        "a named world item is not the package's interface"
                    );
                    name.clone()
                }
            };
            names.push(name);
        }
        names.sort();
        names
    }

    fn type_kind(&self, name: &str) -> Option<&TypeDefKind> {
        let id = self.interface("types")?;
        let ty = *self.resolve.interfaces[id].types.get(name)?;
        Some(&self.resolve.types[ty].kind)
    }

    fn record_fields(&self, name: &str) -> Option<Vec<String>> {
        match self.type_kind(name)? {
            TypeDefKind::Record(record) => Some(
                record
                    .fields
                    .iter()
                    .map(|field| field.name.clone())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The cases of an enum or a variant, or the flags of a flags type.
    fn cases(&self, name: &str) -> Option<Vec<String>> {
        match self.type_kind(name)? {
            TypeDefKind::Enum(e) => Some(e.cases.iter().map(|case| case.name.clone()).collect()),
            TypeDefKind::Variant(v) => Some(v.cases.iter().map(|case| case.name.clone()).collect()),
            TypeDefKind::Flags(f) => Some(f.flags.iter().map(|flag| flag.name.clone()).collect()),
            _ => None,
        }
    }

    fn functions(&self, interface: &str) -> Option<Vec<String>> {
        let id = self.interface(interface)?;
        Some(
            self.resolve.interfaces[id]
                .functions
                .keys()
                .cloned()
                .collect(),
        )
    }
}

#[test]
fn the_versioned_names_a_plugin_author_builds_against_are_pinned() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    let controller =
        file(&set, CONTROLLER_PROTO).expect("the controller file is in the descriptor set");
    for (file, service_name, rpcs) in [
        (plugin, "PolicyPlugin", PLUGIN_RPCS),
        (controller, "GrandCentral", CONTROLLER_RPCS),
    ] {
        let found: Vec<(String, String, String, bool, bool)> = service(file, service_name)
            .unwrap_or_else(|| panic!("service {service_name} is missing"))
            .method
            .iter()
            .map(|method| {
                (
                    method.name().to_owned(),
                    method.input_type().to_owned(),
                    method.output_type().to_owned(),
                    method.client_streaming(),
                    method.server_streaming(),
                )
            })
            .collect();
        let expected: Vec<(String, String, String, bool, bool)> = rpcs
            .iter()
            .map(|(name, input, output, client, server)| {
                (
                    (*name).to_owned(),
                    format!(".{PACKAGE}.{input}"),
                    format!(".{PACKAGE}.{output}"),
                    *client,
                    *server,
                )
            })
            .collect();
        assert_eq!(
            found, expected,
            "service {service_name}: the RPC list moved"
        );
    }
    for (file_name, message_name, oneof, cases) in ONEOFS {
        let file = if *file_name == PLUGIN_PROTO {
            plugin
        } else {
            controller
        };
        let compiled = message(file, message_name)
            .unwrap_or_else(|| panic!("{message_name} is missing from {file_name}"));
        let found = oneof_cases(compiled, oneof)
            .unwrap_or_else(|| panic!("{message_name} has no oneof {oneof}"));
        assert_eq!(
            found,
            strings(cases),
            "{message_name}.{oneof}: the cases moved"
        );
    }

    let world = parse_world().expect("the world resolves");
    let package = &world.resolve.packages[world.package];
    assert_eq!(package.name.namespace, WIT_NAMESPACE);
    assert_eq!(package.name.name, WIT_PACKAGE);
    let version = package.name.version.as_ref().map(ToString::to_string);
    assert_eq!(version.as_deref(), Some(WIT_VERSION));
    let interfaces: Vec<String> = package.interfaces.keys().cloned().collect();
    assert_eq!(interfaces, strings(INTERFACES));
    let worlds: Vec<String> = package.worlds.keys().cloned().collect();
    assert_eq!(
        worlds,
        WORLDS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect::<Vec<_>>()
    );
    for (world_name, imports, exports) in WORLDS {
        let id = world
            .world(world_name)
            .unwrap_or_else(|| panic!("world {world_name} is missing"));
        let item = &world.resolve.worlds[id];
        assert_eq!(
            world.interface_names(item.imports.iter()),
            strings(imports),
            "world {world_name} imports"
        );
        assert_eq!(
            world.interface_names(item.exports.iter()),
            strings(exports),
            "world {world_name} exports"
        );
    }
    assert_eq!(
        world.functions("host").expect("host is an interface"),
        strings(HOST_FUNCTIONS)
    );
    assert_eq!(
        world.functions("plugin").expect("plugin is an interface"),
        strings(PLUGIN_FUNCTIONS)
    );
    assert_eq!(
        world
            .functions("plugin-packed")
            .expect("plugin-packed is an interface"),
        strings(PLUGIN_FUNCTIONS)
    );
}

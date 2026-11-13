//! The policy controller's two wire contracts, compiled and parsed at test
//! time (`contracts/README.md`): the gRPC protocol in package `smg.policy.v1`
//! and the WebAssembly world `smg:policy@1.0.0`. No code is generated and
//! nothing at run time reads the files. The tests hold four things: the
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

use std::{
    io,
    path::{Path, PathBuf},
};

use prost_types::{
    field_descriptor_proto::{Label, Type},
    DescriptorProto, FileDescriptorProto, FileDescriptorSet, ServiceDescriptorProto,
};
use smg_policy_controller::{
    Candidate, Constraints, HashKind, Health, Lease, LeaseKind, LoadRow, MediaCounts, Membership,
    Outcome, OverloadRegime, Pair, Phase, PoolFacts, Queue, ReleaseReason, RequestFacts, Role,
    Shed, ShedReason, Side, Signal, SignalSource, Target, Trace, TransportClass, ZoneCapacity,
};
use wit_parser::{InterfaceId, PackageId, Resolve, TypeDefKind, WorldId, WorldItem, WorldKey};

const PLUGIN_PROTO: &str = "policy_plugin.proto";
const CONTROLLER_PROTO: &str = "policy_controller.proto";
const ENGINE_WIRE_PROTO: &str = "common.proto";
const WORLD_WIT: &str = "smg-policy.wit";
const PACKAGE: &str = "smg.policy.v1";
const WIT_NAMESPACE: &str = "smg";
const WIT_PACKAGE: &str = "policy";
const WIT_VERSION: &str = "1.0.0";

/// A record's field names in declaration order. The pattern names every
/// field without `..`, so the crate's type cannot gain, lose or rename one
/// without this file failing to compile: the wire tables below cannot drift
/// from the record silently.
macro_rules! record {
    ($ty:ty, $name:ident { $($field:ident),+ $(,)? }) => {{
        // Never called: its pattern is the exhaustiveness proof.
        fn exhaustive(value: $ty) {
            let $name { $($field: _),+ } = value;
        }
        let _: fn($ty) = exhaustive;
        &[$(stringify!($field)),+][..]
    }};
}

/// An enum's variant names in declaration order, by the same proof: a match
/// without a wildcard arm.
macro_rules! cases {
    ($ty:ty, $name:ident { $($case:ident),+ $(,)? }) => {{
        fn exhaustive(value: $ty) {
            match value { $($name::$case { .. } => {}),+ }
        }
        let _: fn($ty) = exhaustive;
        &[$(stringify!($case)),+][..]
    }};
}

fn candidate_record() -> &'static [&'static str] {
    record!(
        Candidate,
        Candidate {
            position,
            id,
            role,
            dp_rank,
            dp_size,
            health,
            wire_modes,
            engine_kind,
            connector,
            in_flight,
            load,
            load_sampled_at_micros,
            overlap_blocks,
            predicted_blocks,
            session_hit,
            session_in_flight,
            pair_class,
            has_capacity,
            has_room,
            admitted_age_ms,
            indexed_growth,
            zone,
            labels,
            ec_compatible,
            vision,
            host_local,
            active_sessions,
            key,
        }
    )
}

fn request_record() -> &'static [&'static str] {
    record!(
        RequestFacts<'_>,
        RequestFacts {
            model,
            namespace_seed,
            tenant,
            priority,
            streaming,
            max_output_tokens,
            prompt_tokens,
            block_hashes,
            hash_kind,
            session_key,
            phase,
            attempt,
            tried,
            budget_micros,
            headers,
            media,
            seed,
            now_micros,
        }
    )
}

fn load_row_record() -> &'static [&'static str] {
    record!(
        LoadRow,
        LoadRow {
            running,
            queued,
            waiting_uncached_tokens,
            kv_usage_permille,
            active_kv_permille,
            live_rate,
            resident_tokens,
            report_age_ms,
            source,
            max_running,
        }
    )
}

fn constraints_record() -> &'static [&'static str] {
    record!(
        Constraints<'_>,
        Constraints {
            excluded,
            wire_modes,
            overload_lifted,
            subset_hint
        }
    )
}

fn pool_record() -> &'static [&'static str] {
    record!(
        PoolFacts<'_>,
        PoolFacts {
            pool_size,
            membership_epoch,
            ring_epoch,
            regime,
            generation,
            zones,
        }
    )
}

fn zone_record() -> &'static [&'static str] {
    record!(
        ZoneCapacity,
        ZoneCapacity {
            zone,
            healthy,
            total
        }
    )
}

fn media_record() -> &'static [&'static str] {
    record!(
        MediaCounts,
        MediaCounts {
            images,
            audio,
            video,
            bytes
        }
    )
}

fn health_record() -> &'static [&'static str] {
    record!(
        Health,
        Health {
            healthy,
            breaker_open,
            stalled,
            draining,
            overloaded
        }
    )
}

fn target_record() -> &'static [&'static str] {
    record!(
        Target,
        Target {
            position,
            id,
            dp_rank,
            pair
        }
    )
}

fn pair_record() -> &'static [&'static str] {
    record!(
        Pair,
        Pair {
            decode_position,
            decode_id,
            decode_dp_rank
        }
    )
}

fn shed_record() -> &'static [&'static str] {
    record!(
        Shed,
        Shed {
            reason,
            emptied_by,
            retry_after_secs,
            wait_allowed,
            trace
        }
    )
}

fn queue_record() -> &'static [&'static str] {
    record!(
        Queue,
        Queue {
            ticket,
            max_wait_micros,
            trace
        }
    )
}

fn trace_record() -> &'static [&'static str] {
    record!(
        Trace,
        Trace {
            profile,
            survivors,
            top,
            top_len,
            seed,
            generation
        }
    )
}

fn lease_record() -> &'static [&'static str] {
    record!(
        Lease,
        Lease {
            id,
            kind,
            side,
            target_index,
            generation,
            session_key,
            session,
            issued_micros,
            tokens,
        }
    )
}

fn outcome_record() -> &'static [&'static str] {
    record!(
        Outcome,
        Outcome {
            side,
            served,
            status,
            class,
            ttft_micros,
            tokens_in,
            tokens_out,
            duration_micros,
        }
    )
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
        (
            "HashKind",
            "HASH_KIND",
            "hash-kind",
            cases!(HashKind, HashKind { Token, Text }),
        ),
        (
            "SignalSource",
            "SIGNAL_SOURCE",
            "signal-source",
            cases!(
                SignalSource,
                SignalSource {
                    None,
                    Events,
                    Scrape,
                    Relay,
                    Poll
                }
            ),
        ),
        (
            "OverloadRegime",
            "OVERLOAD_REGIME",
            "overload-regime",
            cases!(
                OverloadRegime,
                OverloadRegime {
                    None,
                    Partial,
                    Saturated
                }
            ),
        ),
        (
            "LeaseKind",
            "LEASE_KIND",
            "lease-kind",
            cases!(
                LeaseKind,
                LeaseKind {
                    Credit,
                    Reservation
                }
            ),
        ),
        (
            "Completed.Class",
            "CLASS",
            "transport-class",
            cases!(
                TransportClass,
                TransportClass {
                    Ok,
                    ClientError,
                    ServerError,
                    ConnectFailed,
                    Timeout,
                    Cancelled,
                }
            ),
        ),
        (
            "Released.Reason",
            "REASON",
            "release-reason",
            cases!(
                ReleaseReason,
                ReleaseReason {
                    Abandoned,
                    LateDecision,
                    ClientGone,
                    CommitFailed,
                    Expired,
                    WorkerRemoved,
                }
            ),
        ),
        (
            "MembershipChange.Kind",
            "KIND",
            "membership",
            cases!(
                Membership<'_>,
                Membership {
                    Added,
                    Removed,
                    Replaced,
                    Relabelled,
                    Resized,
                    Snapshot,
                }
            ),
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
    cases!(
        Signal,
        Signal {
            Load,
            Health,
            GatewayVeto
        }
    )
}

/// How a message of the protocol relates to the record it mirrors.
struct ProtoMirror {
    message: &'static str,
    /// The record's fields, in the crate's order.
    record: &'static [&'static str],
    /// Record fields the wire does not carry: a host instant, a host-local
    /// position, the in-process length of a fixed array, the trace the wire
    /// hoists into `Decision`.
    host_only: &'static [&'static str],
    /// Wire fields outside the record: an event's key before it, controller
    /// facts, a wire-only input and the observability record's fields after
    /// it.
    wire_only: &'static [&'static str],
    /// Every wire field in number order with its scalar kind.
    wire: &'static [(&'static str, Type)],
    /// The `repeated` fields among them.
    repeated: &'static [&'static str],
    /// The proto3 `optional` fields among them: explicit presence, the
    /// crate's `Option`.
    optional: &'static [&'static str],
}

fn proto_mirrors() -> Vec<ProtoMirror> {
    vec![
        ProtoMirror {
            message: "LoadRow",
            record: load_row_record(),
            host_only: &[],
            wire_only: &[],
            wire: &[
                ("running", Type::Uint32),
                ("queued", Type::Uint32),
                ("waiting_uncached_tokens", Type::Uint32),
                ("kv_usage_permille", Type::Uint32),
                ("active_kv_permille", Type::Uint32),
                ("live_rate", Type::Uint32),
                ("resident_tokens", Type::Uint32),
                ("report_age_ms", Type::Uint32),
                ("source", Type::Enum),
                ("max_running", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "MediaCounts",
            record: media_record(),
            host_only: &[],
            wire_only: &[],
            wire: &[
                ("images", Type::Uint32),
                ("audio", Type::Uint32),
                ("video", Type::Uint32),
                ("bytes", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Candidate",
            record: candidate_record(),
            host_only: &["position", "load_sampled_at_micros"],
            wire_only: &[],
            wire: &[
                ("id", Type::Fixed64),
                ("role", Type::Enum),
                ("dp_rank", Type::Uint32),
                ("dp_size", Type::Uint32),
                ("health", Type::Uint32),
                ("wire_modes", Type::Uint32),
                ("engine_kind", Type::Uint32),
                ("connector", Type::Uint32),
                ("in_flight", Type::Uint32),
                ("load", Type::Message),
                ("overlap_blocks", Type::Uint32),
                ("predicted_blocks", Type::Uint32),
                ("session_hit", Type::Bool),
                ("session_in_flight", Type::Uint32),
                ("pair_class", Type::Uint32),
                ("has_capacity", Type::Bool),
                ("has_room", Type::Bool),
                ("admitted_age_ms", Type::Uint32),
                ("indexed_growth", Type::Uint32),
                ("zone", Type::Uint32),
                ("labels", Type::Fixed64),
                ("ec_compatible", Type::Bool),
                ("vision", Type::Bool),
                ("host_local", Type::Bool),
                ("active_sessions", Type::Uint32),
                ("key", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "RequestFacts",
            record: request_record(),
            host_only: &["now_micros"],
            wire_only: &["prompt_text"],
            wire: &[
                ("model", Type::String),
                ("namespace_seed", Type::Fixed64),
                ("tenant", Type::Message),
                ("priority", Type::Sint32),
                ("streaming", Type::Bool),
                ("max_output_tokens", Type::Uint32),
                ("prompt_tokens", Type::Uint32),
                ("block_hashes", Type::Fixed64),
                ("hash_kind", Type::Enum),
                ("session_key", Type::Message),
                ("phase", Type::Enum),
                ("attempt", Type::Uint32),
                ("tried", Type::Fixed64),
                ("budget_micros", Type::Uint32),
                ("headers", Type::Message),
                ("media", Type::Message),
                ("seed", Type::Fixed64),
                ("prompt_text", Type::Bytes),
            ],
            repeated: &["block_hashes", "tried", "headers"],
            optional: &[],
        },
        ProtoMirror {
            message: "Constraints",
            record: constraints_record(),
            host_only: &[],
            wire_only: &[],
            wire: &[
                ("excluded", Type::Fixed64),
                ("wire_modes", Type::Uint32),
                ("overload_lifted", Type::Bool),
                ("subset_hint", Type::Uint32),
            ],
            repeated: &["excluded"],
            optional: &["subset_hint"],
        },
        ProtoMirror {
            message: "PoolFacts",
            record: pool_record(),
            host_only: &[],
            wire_only: &[],
            wire: &[
                ("pool_size", Type::Uint32),
                ("membership_epoch", Type::Uint64),
                ("ring_epoch", Type::Uint64),
                ("regime", Type::Enum),
                ("generation", Type::Uint64),
                ("zones", Type::Message),
            ],
            repeated: &["zones"],
            optional: &[],
        },
        ProtoMirror {
            message: "ZoneCapacity",
            record: zone_record(),
            host_only: &[],
            wire_only: &[],
            wire: &[
                ("zone", Type::Uint32),
                ("healthy", Type::Uint32),
                ("total", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Target",
            record: target_record(),
            host_only: &["position"],
            wire_only: &[],
            wire: &[
                ("id", Type::Fixed64),
                ("dp_rank", Type::Uint32),
                ("pair", Type::Message),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Pair",
            record: pair_record(),
            host_only: &["decode_position"],
            wire_only: &[],
            wire: &[
                ("decode_id", Type::Fixed64),
                ("decode_dp_rank", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Shed",
            record: shed_record(),
            host_only: &["trace"],
            wire_only: &[],
            wire: &[
                ("reason", Type::Enum),
                ("emptied_by", Type::String),
                ("retry_after_secs", Type::Uint32),
                ("wait_allowed", Type::Bool),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Queue",
            record: queue_record(),
            host_only: &["trace"],
            wire_only: &[],
            wire: &[("ticket", Type::Fixed64), ("max_wait_micros", Type::Uint32)],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Trace",
            record: trace_record(),
            host_only: &["top_len"],
            wire_only: &[
                "plan",
                "drops",
                "picker",
                "shadow_pick",
                "fallback",
                "affinity_source",
            ],
            wire: &[
                ("profile", Type::Uint32),
                ("survivors", Type::Uint32),
                ("top", Type::Message),
                ("seed", Type::Fixed64),
                ("generation", Type::Uint64),
                ("plan", Type::Uint32),
                ("drops", Type::Message),
                ("picker", Type::Uint32),
                ("shadow_pick", Type::Fixed64),
                ("fallback", Type::Bool),
                ("affinity_source", Type::Enum),
            ],
            repeated: &["top", "drops"],
            optional: &[],
        },
        ProtoMirror {
            message: "Lease",
            record: lease_record(),
            host_only: &["session", "issued_micros"],
            wire_only: &["term", "dispatch_ttl_ms", "origin"],
            wire: &[
                ("id", Type::Fixed64),
                ("kind", Type::Enum),
                ("side", Type::Enum),
                ("target_index", Type::Uint32),
                ("generation", Type::Uint64),
                ("session_key", Type::Fixed64),
                ("tokens", Type::Uint32),
                ("term", Type::Uint64),
                ("dispatch_ttl_ms", Type::Uint32),
                ("origin", Type::Enum),
            ],
            repeated: &[],
            optional: &[],
        },
        ProtoMirror {
            message: "Completed",
            record: outcome_record(),
            host_only: &[],
            wire_only: &["lease_id"],
            wire: &[
                ("lease_id", Type::Fixed64),
                ("side", Type::Enum),
                ("served", Type::Fixed64),
                ("status", Type::Uint32),
                ("class", Type::Enum),
                ("ttft_micros", Type::Uint32),
                ("tokens_in", Type::Uint32),
                ("tokens_out", Type::Uint32),
                ("duration_micros", Type::Uint32),
            ],
            repeated: &[],
            optional: &[],
        },
    ]
}

/// How a record of the world's `types` interface relates to the crate's.
struct WitMirror {
    record: &'static str,
    crate_record: &'static [&'static str],
    host_only: &'static [&'static str],
    /// Fields the WIT record carries before the crate's (an event's lease).
    before: &'static [&'static str],
}

fn wit_mirrors() -> Vec<WitMirror> {
    vec![
        WitMirror {
            record: "load",
            crate_record: load_row_record(),
            host_only: &[],
            before: &[],
        },
        WitMirror {
            record: "candidate",
            crate_record: candidate_record(),
            host_only: &["position", "load_sampled_at_micros"],
            before: &[],
        },
        WitMirror {
            record: "media",
            crate_record: media_record(),
            host_only: &[],
            before: &[],
        },
        WitMirror {
            record: "request",
            crate_record: request_record(),
            host_only: &["now_micros"],
            before: &[],
        },
        WitMirror {
            record: "zone-capacity",
            crate_record: zone_record(),
            host_only: &[],
            before: &[],
        },
        WitMirror {
            record: "pool",
            crate_record: pool_record(),
            host_only: &[],
            before: &[],
        },
        WitMirror {
            record: "lease",
            crate_record: lease_record(),
            host_only: &["session", "issued_micros"],
            before: &[],
        },
        WitMirror {
            record: "pair",
            crate_record: pair_record(),
            host_only: &["decode_position"],
            before: &[],
        },
        WitMirror {
            record: "target",
            crate_record: target_record(),
            host_only: &["position"],
            before: &[],
        },
        WitMirror {
            record: "completed",
            crate_record: outcome_record(),
            host_only: &[],
            before: &["lease"],
        },
    ]
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
    ("PushLoad", "LoadReport", "Ack", true, false),
    ("PushKvEvents", "RelayKvFrame", "Ack", true, false),
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
        ],
    ),
    (
        CONTROLLER_PROTO,
        "RelayKvFrame",
        "frame",
        &["batch", "subscribed", "resync", "ended"],
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

/// The engine wire's protos (`smg.grpc.common`): the controller protocol
/// imports the load record and the event batch a relay forwards unchanged.
fn engine_wire_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../grpc_client/proto")
}

fn compile_protocol() -> io::Result<FileDescriptorSet> {
    let dir = contracts_dir();
    let mut config = prost_build::Config::new();
    // proto3 `optional` needs the flag below protoc 3.15, the oldest the
    // repository's build scripts run under.
    config.protoc_arg("--experimental_allow_proto3_optional");
    config.load_fds(
        &[dir.join(PLUGIN_PROTO), dir.join(CONTROLLER_PROTO)],
        &[dir, engine_wire_dir()],
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

/// The fields of a message in number order: name, number, kind, repeated,
/// proto3 optional.
fn fields(message: &DescriptorProto) -> Vec<(String, i32, Type, bool, bool)> {
    let mut fields: Vec<_> = message
        .field
        .iter()
        .map(|field| {
            (
                field.name().to_owned(),
                field.number(),
                field.r#type(),
                field.label() == Label::Repeated,
                field.proto3_optional(),
            )
        })
        .collect();
    fields.sort_by_key(|(_, number, _, _, _)| *number);
    fields
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

/// The values of a top-level enum (`Role`) or a nested one (`Completed.Class`).
fn enum_values(file: &FileDescriptorProto, path: &str) -> Option<Vec<(String, i32)>> {
    let found = match path.split_once('.') {
        None => file.enum_type.iter().find(|e| e.name() == path)?,
        Some((owner, nested)) => message(file, owner)?
            .enum_type
            .iter()
            .find(|e| e.name() == nested)?,
    };
    Some(
        found
            .value
            .iter()
            .map(|v| (v.name().to_owned(), v.number()))
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

/// `BreakerOpen` to `BREAKER_OPEN`.
fn screaming_case(camel: &str) -> String {
    kebab_case(camel).replace('-', "_").to_ascii_uppercase()
}

/// `wire_modes` to `wire-modes`.
fn kebab_field(snake: &str) -> String {
    snake.replace('_', "-")
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

/// The proto enum a crate enum mirrors: `PREFIX_UNSPECIFIED = 0`, then the
/// crate's variants in order, numbered from 1.
fn proto_enum(prefix: &str, cases: &[&str]) -> Vec<(String, i32)> {
    let mut values = vec![(format!("{prefix}_UNSPECIFIED"), 0)];
    for (i, case) in cases.iter().enumerate() {
        let number = i32::try_from(i + 1).unwrap_or(i32::MAX);
        values.push((format!("{prefix}_{}", screaming_case(case)), number));
    }
    values
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
fn the_protocol_compiles_and_mirrors_the_engine_record() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    let controller =
        file(&set, CONTROLLER_PROTO).expect("the controller file is in the descriptor set");
    assert_eq!(plugin.package(), PACKAGE);
    assert_eq!(controller.package(), PACKAGE);
    assert_eq!(
        controller.dependency,
        strings(&[PLUGIN_PROTO, ENGINE_WIRE_PROTO]),
        "the controller protocol imports the plugin vocabulary and the engine wire"
    );

    for mirror in proto_mirrors() {
        let message_name = mirror.message;
        // The wire table is the crate's record, less the host's fields.
        let table: Vec<&str> = mirror
            .wire
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !mirror.wire_only.contains(name))
            .collect();
        let record: Vec<&str> = mirror
            .record
            .iter()
            .copied()
            .filter(|name| !mirror.host_only.contains(name))
            .collect();
        assert_eq!(
            table, record,
            "{message_name}: the wire table is not the crate's record"
        );
        // The compiled message is the wire table, numbered 1..=n in order.
        let compiled = message(plugin, message_name)
            .unwrap_or_else(|| panic!("{message_name} is missing from {PLUGIN_PROTO}"));
        let expected: Vec<(String, i32, Type, bool, bool)> = mirror
            .wire
            .iter()
            .enumerate()
            .map(|(i, (name, ty))| {
                let number = i32::try_from(i + 1).expect("a field number fits");
                (
                    (*name).to_owned(),
                    number,
                    *ty,
                    mirror.repeated.contains(name),
                    mirror.optional.contains(name),
                )
            })
            .collect();
        assert_eq!(
            fields(compiled),
            expected,
            "{message_name}: the fields differ from the record"
        );
    }

    for (proto_name, prefix, _, cases) in enum_mirrors() {
        let found = enum_values(plugin, proto_name)
            .unwrap_or_else(|| panic!("enum {proto_name} is missing from {PLUGIN_PROTO}"));
        assert_eq!(found, proto_enum(prefix, cases), "enum {proto_name}");
    }
    let found = enum_values(plugin, "ShedReason").expect("ShedReason is an enum");
    assert_eq!(found, proto_enum("SHED_REASON", shed_reasons()));
    // The health bits: the record's flags in order, one bit each.
    let found = enum_values(plugin, "HealthFlag").expect("HealthFlag is an enum");
    let mut expected = vec![("HEALTH_FLAG_UNSPECIFIED".to_owned(), 0)];
    for (i, flag) in health_record().iter().enumerate() {
        expected.push((format!("HEALTH_FLAG_{}", screaming_case(flag)), 1 << i));
    }
    assert_eq!(
        found, expected,
        "the health flags are the record's, one bit each"
    );
    let found = enum_values(plugin, "WireMode").expect("WireMode is an enum");
    assert_eq!(
        found,
        [
            ("WIRE_MODE_UNSPECIFIED", 0),
            ("WIRE_MODE_HTTP", 1),
            ("WIRE_MODE_GRPC", 2),
            ("WIRE_MODE_ZMQ", 4)
        ]
        .map(|(name, number)| (name.to_owned(), number))
    );
}

#[test]
fn the_wasm_world_resolves_and_mirrors_the_engine_record() {
    let world = parse_world().expect("the world resolves");
    let name = &world.resolve.packages[world.package].name;
    assert_eq!(
        name.to_string(),
        format!("{WIT_NAMESPACE}:{WIT_PACKAGE}@{WIT_VERSION}")
    );

    for mirror in wit_mirrors() {
        let record_name = mirror.record;
        let expected: Vec<String> = mirror
            .before
            .iter()
            .map(|name| (*name).to_owned())
            .chain(
                mirror
                    .crate_record
                    .iter()
                    .filter(|name| !mirror.host_only.contains(name))
                    .map(|name| kebab_field(name)),
            )
            .collect();
        let found = world
            .record_fields(record_name)
            .unwrap_or_else(|| panic!("record {record_name} is missing from types"));
        assert_eq!(found, expected, "record {record_name} is not the crate's");
    }
    assert_eq!(
        world
            .record_fields("dispatched")
            .expect("dispatched is a record"),
        strings(&["lease", "target", "side", "since-decision-micros"]),
        "the dispatch event carries the lifecycle hook's arguments"
    );

    for (_, _, wit_name, cases) in enum_mirrors() {
        let expected: Vec<String> = cases.iter().map(|case| kebab_case(case)).collect();
        let found = world
            .cases(wit_name)
            .unwrap_or_else(|| panic!("{wit_name} is missing from types"));
        assert_eq!(found, expected, "{wit_name} is not the crate's enum");
    }
    let expected: Vec<String> = signal_cases().iter().map(|case| kebab_case(case)).collect();
    assert_eq!(
        world.cases("signal").expect("signal is a variant"),
        expected
    );
    let expected: Vec<String> = health_record()
        .iter()
        .map(|flag| kebab_field(flag))
        .collect();
    assert_eq!(
        world.cases("health").expect("health is a flags type"),
        expected
    );
    assert_eq!(
        world
            .cases("wire-modes")
            .expect("wire-modes is a flags type"),
        strings(&["http", "grpc", "zmq"])
    );
    assert_eq!(
        world.cases("identity").expect("identity is a variant"),
        strings(&["none", "hash", "plain"])
    );
    assert_eq!(
        world.cases("decline").expect("decline is a variant"),
        strings(&["not-applicable", "budget", "internal"])
    );
    assert_eq!(
        world
            .cases("lifecycle-event")
            .expect("lifecycle-event is a variant"),
        strings(&[
            "issued",
            "dispatched",
            "completed",
            "released",
            "membership",
            "signal"
        ])
    );
}

#[test]
fn the_versioned_names_a_plugin_author_builds_against_are_pinned() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    let controller =
        file(&set, CONTROLLER_PROTO).expect("the controller file is in the descriptor set");
    for (file, service_name, rpcs) in [
        (plugin, "PolicyPlugin", PLUGIN_RPCS),
        (controller, "PolicyController", CONTROLLER_RPCS),
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

/// The packed request block's layout as the world's comment states it: a
/// 72-byte header, the model bytes, zero bytes to the next multiple of 8,
/// then the u64 sections. The arithmetic is mirrored here so the alignment
/// rule is pinned beside the sentence that states it.
const PACKED_REQUEST_HEADER: usize = 72;

fn packed_hashes_offset(model_len: usize) -> usize {
    (PACKED_REQUEST_HEADER + model_len).div_ceil(8) * 8
}

#[test]
fn the_packed_request_block_keeps_its_u64_sections_aligned() {
    assert_eq!(packed_hashes_offset(12), 88, "a 12-byte model name");
    for model_len in 0..=255usize {
        let hashes = packed_hashes_offset(model_len);
        assert_eq!(hashes % 8, 0, "block hashes at a {model_len}-byte name");
        let tried = hashes + 8 * 16;
        assert_eq!(
            tried % 8,
            0,
            "tried after 16 hashes at a {model_len}-byte name"
        );
    }
    let wit =
        std::fs::read_to_string(contracts_dir().join(WORLD_WIT)).expect("the world file reads");
    for sentence in [
        "Request block: a 72-byte header",
        "zero bytes to the next multiple of 8",
        "puts the hashes at 88",
        "returns 8-aligned memory for these lists",
    ] {
        assert!(
            wit.contains(sentence),
            "the world states the alignment rule: {sentence:?}"
        );
    }
}

/// The lease-id layout's bounds as `LeaseOrigin` states them: terms below
/// 2^31 fit the 31 id bits, the per-term counter starts at 1 and never wraps
/// into the term bits, so a controller-minted id is never 0 and never repeats
/// across terms. The sentences are pinned beside the layout they bound.
#[test]
fn the_lease_id_layout_states_its_bounds() {
    let proto = std::fs::read_to_string(contracts_dir().join(PLUGIN_PROTO))
        .expect("the plugin protocol reads");
    for sentence in [
        "terms\n// stay below 2^31",
        "the counter starts at 1",
        "bumps its term before it mints again",
        "never 0 and\n// never repeats across terms",
    ] {
        assert!(
            proto.contains(sentence),
            "LeaseOrigin states the id bounds: {sentence:?}"
        );
    }
}

/// The hello version fields: `PluginHello` offers a range on a Decide stream
/// and one version (`protocol_min` = `protocol_max`) on an Observe stream;
/// `PluginHelloAck.protocol` is the version a Decide stream settled, the one
/// every Observe hello of that instance then names.
type HelloField = (&'static str, i32, Type);

const HELLO_FIELDS: &[(&str, &[HelloField])] = &[
    (
        "PluginHello",
        &[
            ("protocol_min", 1, Type::Uint32),
            ("protocol_max", 2, Type::Uint32),
        ],
    ),
    ("PluginHelloAck", &[("protocol", 1, Type::Uint32)]),
];

#[test]
fn the_hello_version_fields_are_pinned() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    for (message_name, expected) in HELLO_FIELDS {
        let compiled = message(plugin, message_name)
            .unwrap_or_else(|| panic!("{message_name} is missing from {PLUGIN_PROTO}"));
        let found: Vec<(String, i32, Type)> = fields(compiled)
            .into_iter()
            .filter(|(name, _, _, _, _)| {
                expected
                    .iter()
                    .any(|(pinned, _, _)| *pinned == name.as_str())
            })
            .map(|(name, number, kind, _, _)| (name, number, kind))
            .collect();
        let expected: Vec<(String, i32, Type)> = expected
            .iter()
            .map(|(name, number, kind)| ((*name).to_owned(), *number, *kind))
            .collect();
        assert_eq!(
            found, expected,
            "{message_name}: a hello version field moved"
        );
    }
}

/// The decline a plugin answers a stage with: `stage` is Stage bits, `reason`
/// the enum, `detail` free text; a frame declining a stage and filling that
/// stage's answer field is invalid.
type DeclineField = (&'static str, i32, Type);

const DECLINE_FIELDS: &[(&str, &[DeclineField])] = &[(
    "Decline",
    &[
        ("stage", 1, Type::Uint32),
        ("reason", 2, Type::Enum),
        ("detail", 3, Type::String),
    ],
)];

#[test]
fn the_decline_fields_are_pinned() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    for (message_name, expected) in DECLINE_FIELDS {
        let compiled = message(plugin, message_name)
            .unwrap_or_else(|| panic!("{message_name} is missing from {PLUGIN_PROTO}"));
        let found: Vec<(String, i32, Type)> = fields(compiled)
            .into_iter()
            .map(|(name, number, kind, _, _)| (name, number, kind))
            .collect();
        let expected: Vec<(String, i32, Type)> = expected
            .iter()
            .map(|(name, number, kind)| ((*name).to_owned(), *number, *kind))
            .collect();
        assert_eq!(found, expected, "{message_name}: a decline field moved");
    }
}

/// Every lease end names exactly one lifecycle event: a completion, or a
/// release with its reason. An end with no event is the leak a stateful plugin
/// cannot see, so a `LeaseEnd` cause without a row here fails the test.
const LEASE_END_EVENTS: &[(&str, &str)] = &[
    ("CAUSE_COMPLETED", "completed"),
    ("CAUSE_RELEASED", "released(reason)"),
    ("CAUSE_EXPIRED", "released(REASON_EXPIRED)"),
    ("CAUSE_WORKER_REMOVED", "released(REASON_WORKER_REMOVED)"),
];

#[test]
fn every_lease_end_cause_names_one_lifecycle_event() {
    let set = compile_protocol().expect("protoc compiles the two protocol files");
    let plugin = file(&set, PLUGIN_PROTO).expect("the plugin file is in the descriptor set");
    let controller =
        file(&set, CONTROLLER_PROTO).expect("the controller file is in the descriptor set");
    let causes: Vec<String> = enum_values(controller, "LeaseEnd.Cause")
        .expect("LeaseEnd.Cause is an enum")
        .into_iter()
        .filter(|(_, number)| *number != 0)
        .map(|(name, _)| name)
        .collect();
    let mapped: Vec<String> = LEASE_END_EVENTS
        .iter()
        .map(|(cause, _)| (*cause).to_owned())
        .collect();
    assert_eq!(causes, mapped, "a LeaseEnd cause has no lifecycle event");
    let reasons: Vec<String> = enum_values(plugin, "Released.Reason")
        .expect("Released.Reason is an enum")
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    for (cause, event) in LEASE_END_EVENTS {
        if let Some(reason) = event
            .strip_prefix("released(")
            .and_then(|rest| rest.strip_suffix(')'))
            .filter(|reason| reason.starts_with("REASON_"))
        {
            assert!(
                reasons.iter().any(|r| r == reason),
                "{cause}: {reason} is not a Released.Reason"
            );
        }
    }
}

//! The `smg:policy` types interface as data: every record, flags, enum and
//! variant in the committed order with its documentation, rendered back into
//! the WIT text by `render_types`. The records are tied to the
//! engine's Rust types by the probes in `probes`: a struct that gains,
//! loses, renames or retypes a field fails to compile or fails the WIT test.

use super::{Field, Item};

/// The interface, item by item.
pub(crate) const TYPES: &[Item] = &[
    Item::Flags {
        name: "health",
        doc: &[" Health as the pool reads it, one bit each."],
        cases: &[
            "healthy",
            "breaker-open",
            "stalled",
            "draining",
            "overloaded",
        ],
        blank_before: false,
    },
    Item::Flags {
        name: "wire-modes",
        doc: &[" The wires a worker serves."],
        cases: &["http", "grpc", "zmq"],
        blank_before: false,
    },
    Item::Enum {
        name: "role",
        doc: &[],
        cases: &["single", "prefill", "decode", "encode"],
        blank_before: false,
    },
    Item::Enum {
        name: "phase",
        doc: &[],
        cases: &["single", "prefill", "decode", "encode"],
        blank_before: false,
    },
    Item::Enum {
        name: "side",
        doc: &[],
        cases: &["single", "prefill", "decode", "encode"],
        blank_before: false,
    },
    Item::Enum {
        name: "hash-kind",
        doc: &[" What the block hashes were computed over."],
        cases: &["token", "text"],
        blank_before: false,
    },
    Item::Enum {
        name: "signal-source",
        doc: &[" Where a load row came from."],
        cases: &["none", "events", "scrape", "relay", "poll"],
        blank_before: false,
    },
    Item::Enum {
        name: "overload-regime",
        doc: &[],
        cases: &["none", "partial", "saturated"],
        blank_before: false,
    },
    Item::Enum {
        name: "lease-kind",
        doc: &[],
        cases: &["credit", "reservation"],
        blank_before: false,
    },
    Item::Enum {
        name: "transport-class",
        doc: &[" The transport outcome of a dispatch, beside its status."],
        cases: &[
            "ok",
            "client-error",
            "server-error",
            "connect-failed",
            "timeout",
            "cancelled",
        ],
        blank_before: false,
    },
    Item::Enum {
        name: "release-reason",
        doc: &[" Why a lease was released before its completion."],
        cases: &[
            "abandoned",
            "late-decision",
            "client-gone",
            "commit-failed",
            "expired",
            "worker-removed",
        ],
        blank_before: false,
    },
    Item::Variant {
        name: "identity",
        doc: &[
            " A tenant or a session key under the plugin's exposure: `hash` unless the",
            " spec says `plain`, `none` when withheld.",
        ],
        cases: &["none", "hash(u64)", "plain(string)"],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "load",
        doc: &[
            " The engine load record's routing core, rounded to narrow fields by the",
            " host; permille fields are the record's ratios times 1000.",
        ],
        fields: &[
            Field {
                name: "running",
                ty: "u32",
            },
            Field {
                name: "queued",
                ty: "u32",
            },
            Field {
                name: "waiting-uncached-tokens",
                ty: "u32",
            },
            Field {
                name: "kv-usage-permille",
                ty: "u16",
            },
            Field {
                name: "active-kv-permille",
                ty: "u16",
            },
            Field {
                name: "live-rate",
                ty: "u32",
            },
            Field {
                name: "resident-tokens",
                ty: "u32",
            },
            Field {
                name: "report-age-ms",
                ty: "u32",
            },
            Field {
                name: "source",
                ty: "signal-source",
            },
            Field {
                name: "max-running",
                ty: "u32",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Record {
        name: "candidate",
        doc: &[
            " One candidate row: a worker, or one DP rank of one. Positions are the",
            " row's index in the list the decision was given; the stage results name",
            " positions. `zone`, `engine-kind` and `connector` are one-based positions",
            " into the spec's declared lists (0 is unzoned, unspecified or none;",
            " `zone-capacity.zone` is never 0); `labels` are bits over the declared",
            " label list (at most 64; validation refuses a longer list). `load` is",
            " absent when the host has no report; `admitted-age-ms` saturates at the",
            " u32 maximum, never wraps.",
        ],
        fields: &[
            Field {
                name: "id",
                ty: "u64",
            },
            Field {
                name: "role",
                ty: "role",
            },
            Field {
                name: "dp-rank",
                ty: "u16",
            },
            Field {
                name: "dp-size",
                ty: "u16",
            },
            Field {
                name: "health",
                ty: "health",
            },
            Field {
                name: "wire-modes",
                ty: "wire-modes",
            },
            Field {
                name: "engine-kind",
                ty: "u16",
            },
            Field {
                name: "connector",
                ty: "u16",
            },
            Field {
                name: "in-flight",
                ty: "u32",
            },
            Field {
                name: "load",
                ty: "option<load>",
            },
            Field {
                name: "overlap-blocks",
                ty: "u32",
            },
            Field {
                name: "predicted-blocks",
                ty: "u32",
            },
            Field {
                name: "session-hit",
                ty: "bool",
            },
            Field {
                name: "session-in-flight",
                ty: "u16",
            },
            Field {
                name: "pair-class",
                ty: "u32",
            },
            Field {
                name: "has-capacity",
                ty: "bool",
            },
            Field {
                name: "has-room",
                ty: "bool",
            },
            Field {
                name: "admitted-age-ms",
                ty: "u32",
            },
            Field {
                name: "indexed-growth",
                ty: "u32",
            },
            Field {
                name: "zone",
                ty: "u16",
            },
            Field {
                name: "labels",
                ty: "u64",
            },
            Field {
                name: "ec-compatible",
                ty: "bool",
            },
            Field {
                name: "vision",
                ty: "bool",
            },
            Field {
                name: "host-local",
                ty: "bool",
            },
            Field {
                name: "active-sessions",
                ty: "u16",
            },
            Field {
                name: "key",
                ty: "u32",
            },
            Field {
                name: "indexed-blocks",
                ty: "u32",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Record {
        name: "media",
        doc: &[" Media on the request: counts and bytes."],
        fields: &[
            Field {
                name: "images",
                ty: "u16",
            },
            Field {
                name: "audio",
                ty: "u16",
            },
            Field {
                name: "video",
                ty: "u16",
            },
            Field {
                name: "bytes",
                ty: "u32",
            },
        ],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "request",
        doc: &[
            " One request as a decision sees it. `headers` pairs a zero-based position",
            " into the spec's `view.headers` list with the value (unlike the one-based",
            " `zone`, `engine-kind` and `connector`); `budget-micros` is relative to",
            " the host's receipt; `seed` is the only randomness (see `host`).",
        ],
        fields: &[
            Field {
                name: "model",
                ty: "string",
            },
            Field {
                name: "namespace-seed",
                ty: "u64",
            },
            Field {
                name: "tenant",
                ty: "identity",
            },
            Field {
                name: "priority",
                ty: "s32",
            },
            Field {
                name: "streaming",
                ty: "bool",
            },
            Field {
                name: "max-output-tokens",
                ty: "u32",
            },
            Field {
                name: "prompt-tokens",
                ty: "u32",
            },
            Field {
                name: "block-hashes",
                ty: "list<u64>",
            },
            Field {
                name: "hash-kind",
                ty: "hash-kind",
            },
            Field {
                name: "session-key",
                ty: "identity",
            },
            Field {
                name: "phase",
                ty: "phase",
            },
            Field {
                name: "attempt",
                ty: "u8",
            },
            Field {
                name: "tried",
                ty: "list<u64>",
            },
            Field {
                name: "budget-micros",
                ty: "u32",
            },
            Field {
                name: "headers",
                ty: "list<tuple<u8, string>>",
            },
            Field {
                name: "media",
                ty: "media",
            },
            Field {
                name: "seed",
                ty: "u64",
            },
            Field {
                name: "block-size",
                ty: "u32",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Record {
        name: "zone-capacity",
        doc: &[" Healthy capacity of one zone, by the zone's position."],
        fields: &[
            Field {
                name: "zone",
                ty: "u16",
            },
            Field {
                name: "healthy",
                ty: "u32",
            },
            Field {
                name: "total",
                ty: "u32",
            },
        ],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "pool",
        doc: &[" Pool-wide facts beside the candidates."],
        fields: &[
            Field {
                name: "pool-size",
                ty: "u32",
            },
            Field {
                name: "membership-epoch",
                ty: "u64",
            },
            Field {
                name: "ring-epoch",
                ty: "u64",
            },
            Field {
                name: "regime",
                ty: "overload-regime",
            },
            Field {
                name: "generation",
                ty: "u64",
            },
            Field {
                name: "zones",
                ty: "list<zone-capacity>",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Variant {
        name: "decline",
        doc: &[" A stage that steps aside without a trap: the configured fallback runs."],
        cases: &["not-applicable", "budget", "internal(string)"],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "lease",
        doc: &[
            " The record of one issued decision, or of one side of a pair decision. The",
            " id is the host's and unique within its lifecycle stream: the controller",
            " mints the ids a plugin sees (unique across gateways and terms: the",
            " minting term in the high bits, under the bounds LeaseOrigin states in the",
            " proto) and re-mints a gateway-minted fallback id before it reaches a",
            " plugin.",
        ],
        fields: &[
            Field {
                name: "id",
                ty: "u64",
            },
            Field {
                name: "kind",
                ty: "lease-kind",
            },
            Field {
                name: "side",
                ty: "side",
            },
            Field {
                name: "target-index",
                ty: "u8",
            },
            Field {
                name: "generation",
                ty: "u64",
            },
            Field {
                name: "session-key",
                ty: "u64",
            },
            Field {
                name: "tokens",
                ty: "u32",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Record {
        name: "pair",
        doc: &[" The decode side of a pair target."],
        fields: &[
            Field {
                name: "decode-id",
                ty: "u64",
            },
            Field {
                name: "decode-dp-rank",
                ty: "u16",
            },
        ],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "target",
        doc: &[
            " A target by id; `pair` names the decode half when the decision is a",
            " pair, and the target itself is then the prefill.",
        ],
        fields: &[
            Field {
                name: "id",
                ty: "u64",
            },
            Field {
                name: "dp-rank",
                ty: "u16",
            },
            Field {
                name: "pair",
                ty: "option<pair>",
            },
        ],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "dispatched",
        doc: &[
            " The host dispatched to `target` on `side`, `since-decision-micros` after",
            " the decision on its clock: a load credit dates from the dispatch.",
        ],
        fields: &[
            Field {
                name: "lease",
                ty: "lease",
            },
            Field {
                name: "target",
                ty: "target",
            },
            Field {
                name: "side",
                ty: "side",
            },
            Field {
                name: "since-decision-micros",
                ty: "u32",
            },
        ],
        inline: true,
        blank_before: true,
    },
    Item::Record {
        name: "completed",
        doc: &[
            " What a completed dispatch reported; `duration-micros` saturates at the",
            " u32 maximum, never wraps.",
        ],
        fields: &[
            Field {
                name: "lease",
                ty: "lease",
            },
            Field {
                name: "side",
                ty: "side",
            },
            Field {
                name: "served",
                ty: "u64",
            },
            Field {
                name: "status",
                ty: "u16",
            },
            Field {
                name: "class",
                ty: "transport-class",
            },
            Field {
                name: "ttft-micros",
                ty: "u32",
            },
            Field {
                name: "tokens-in",
                ty: "u32",
            },
            Field {
                name: "tokens-out",
                ty: "u32",
            },
            Field {
                name: "duration-micros",
                ty: "u32",
            },
        ],
        inline: false,
        blank_before: true,
    },
    Item::Variant {
        name: "membership",
        doc: &[
            " A membership change; `replaced` carries (old, new), `snapshot` every",
            " worker of every pool the engine decides over.",
        ],
        cases: &[
            "added(u64)",
            "removed(u64)",
            "replaced(tuple<u64, u64>)",
            "relabelled(u64)",
            "resized(u64)",
            "snapshot(list<u64>)",
        ],
        inline: false,
        blank_before: true,
    },
    Item::Variant {
        name: "signal",
        doc: &[
            " A signal about one worker: a load row for (worker, rank), a health",
            " transition, or the worker's standing veto for the given microseconds,",
            " aggregated over every gateway (it stands while any gateway holds one, the",
            " microseconds are the longest remaining, 0 only when none remains). The",
            " deciding gateway's own veto is applied before any stage runs: in that",
            " gateway's candidate rows alone the worker reads breaker-open (a breaker",
            " veto) or stalled (a liveness veto).",
        ],
        cases: &[
            "load(tuple<u64, u16, load>)",
            "health(tuple<u64, health>)",
            "gateway-veto(tuple<u64, u32>)",
        ],
        inline: false,
        blank_before: true,
    },
    Item::Variant {
        name: "lifecycle-event",
        doc: &[" The lifecycle as a versioned variant, never bytes."],
        cases: &[
            "issued(tuple<lease, list<target>>)",
            "dispatched(dispatched)",
            "completed(completed)",
            "released(tuple<lease, release-reason>)",
            "membership(membership)",
            "signal(signal)",
        ],
        inline: false,
        blank_before: true,
    },
];

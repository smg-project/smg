//! The names the two wire contracts mirror, each list held by the
//! compiler: a pattern that names every field of a record with no `..`,
//! and a `match` that names every case of an enum with no arm for the
//! rest. A type that gains, loses or renames one stops the build of
//! whatever states its list.
//!
//! The contracts' own check (`tests/contracts.rs`) states the lists of the
//! types it holds to the two wires by the two macros here. No such pattern
//! of a type marked non-exhaustive builds outside this crate, so the lists
//! of the marked types are stated here, beside the macros, and the check
//! reads them: a new case of [`TransportClass`] still stops that check
//! until both wires carry it.
//!
//! The package's own tests build this module (`test-util`); no build of a
//! product has it.

use crate::{
    engine::{
        AffinityFacts, Candidate, Constraints, HashKind, Health, LoadRow, MediaCounts,
        OverloadRegime, PoolFacts, RequestFacts, SignalSource, ZoneCapacity,
    },
    tracking::{Membership, Outcome, ReleaseReason, Signal, TransportClass},
};

/// A record's field names in declaration order. The pattern names every
/// field without `..`, so the type cannot gain, lose or rename one without
/// the list failing to compile: a wire table held to the list cannot drift
/// from the record silently. A field written `name as "spelled"` is listed
/// under the name the contracts spell, the pattern still naming the
/// struct's own: a pair, both sides checked.
#[doc(hidden)]
#[macro_export]
macro_rules! record {
    (@spelled $field:ident) => { stringify!($field) };
    (@spelled $field:ident $spelled:literal) => { $spelled };
    ($ty:ty, $name:ident { $($field:ident $(as $spelled:literal)?),+ $(,)? }) => {{
        // Never called: its pattern is the exhaustiveness proof.
        fn exhaustive(value: $ty) {
            let $name { $($field: _),+ } = value;
        }
        let _: fn($ty) = exhaustive;
        &[$($crate::record!(@spelled $field $($spelled)?)),+][..]
    }};
}

/// An enum's variant names in declaration order, by the same proof: a match
/// without a wildcard arm.
#[doc(hidden)]
#[macro_export]
macro_rules! cases {
    ($ty:ty, $name:ident { $($case:ident),+ $(,)? }) => {{
        fn exhaustive(value: $ty) {
            match value { $($name::$case { .. } => {}),+ }
        }
        let _: fn($ty) = exhaustive;
        &[$(stringify!($case)),+][..]
    }};
}

/// The fields of [`Outcome`].
#[must_use]
pub fn outcome() -> &'static [&'static str] {
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

/// The cases of [`TransportClass`].
#[must_use]
pub fn transport_class() -> &'static [&'static str] {
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
    )
}

/// The cases of [`ReleaseReason`].
#[must_use]
pub fn release_reason() -> &'static [&'static str] {
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
    )
}

/// The cases of [`Membership`].
#[must_use]
pub fn membership() -> &'static [&'static str] {
    cases!(
        Membership<'_>,
        Membership {
            Added,
            Removed,
            Replaced,
            Relabelled,
            Resized,
            Snapshot,
            Left,
        }
    )
}

/// The cases of [`Signal`].
#[must_use]
pub fn signal() -> &'static [&'static str] {
    cases!(
        Signal<'_>,
        Signal {
            Load,
            Health,
            GatewayVeto,
        }
    )
}

/// The fields of [`Candidate`].
#[must_use]
pub fn candidate() -> &'static [&'static str] {
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
            session_hit,
            session_in_flight,
            pair_class,
            has_capacity,
            has_room,
            zone,
            labels,
            ec_compatible,
            vision,
            host_local,
            active_sessions,
            key,
            engine,
        }
    )
}

/// The fields of [`AffinityFacts`].
#[must_use]
pub fn affinity_facts() -> &'static [&'static str] {
    record!(
        AffinityFacts,
        AffinityFacts {
            overlap_blocks,
            predicted_blocks,
            admitted_age_ms,
            indexed_growth,
            indexed_blocks,
        }
    )
}

/// The fields of [`RequestFacts`].
#[must_use]
pub fn request_facts() -> &'static [&'static str] {
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
            block_size,
            prompt_chars,
            prompt_bytes,
            now_micros,
            session_counted,
            affinity,
            named,
            model_fingerprint,
            signal_cells,
            client_key,
            ring,
            ring_positions,
            session_key_from_rid,
        }
    )
}

/// The fields of [`LoadRow`].
#[must_use]
pub fn load_row() -> &'static [&'static str] {
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
            resident_counted,
            report_age_ms,
            source,
            max_running,
        }
    )
}

/// The fields of [`Constraints`].
#[must_use]
pub fn constraints() -> &'static [&'static str] {
    record!(
        Constraints<'_>,
        Constraints {
            excluded,
            wire_modes,
            overload_lifted,
            decode_later,
            counted_on_decode,
            subset_hint,
            demands,
            stalled,
            held,
            places
        }
    )
}

/// The fields of [`PoolFacts`].
#[must_use]
pub fn pool_facts() -> &'static [&'static str] {
    record!(
        PoolFacts<'_>,
        PoolFacts {
            pool_size,
            membership_epoch,
            ring_epoch,
            regime,
            generation,
            zones,
            reach,
            reported,
            engines,
            word,
        }
    )
}

/// The fields of [`ZoneCapacity`].
#[must_use]
pub fn zone_capacity() -> &'static [&'static str] {
    record!(
        ZoneCapacity,
        ZoneCapacity {
            zone,
            healthy,
            total
        }
    )
}

/// The fields of [`MediaCounts`].
#[must_use]
pub fn media_counts() -> &'static [&'static str] {
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

/// The fields of [`Health`].
#[must_use]
pub fn health() -> &'static [&'static str] {
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

/// The cases of [`HashKind`].
#[must_use]
pub fn hash_kind() -> &'static [&'static str] {
    cases!(HashKind, HashKind { Token, Text })
}

/// The cases of [`SignalSource`].
#[must_use]
pub fn signal_source() -> &'static [&'static str] {
    cases!(
        SignalSource,
        SignalSource {
            None,
            Events,
            Scrape,
            Relay,
            Poll
        }
    )
}

/// The cases of [`OverloadRegime`].
#[must_use]
pub fn overload_regime() -> &'static [&'static str] {
    cases!(
        OverloadRegime,
        OverloadRegime {
            None,
            Partial,
            Saturated
        }
    )
}

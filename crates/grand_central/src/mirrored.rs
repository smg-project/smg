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

use crate::tracking::{Membership, Outcome, ReleaseReason, Signal, TransportClass};

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

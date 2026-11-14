//! The probes that tie each WIT record to the engine's Rust struct: an
//! exhaustive destructuring per record (a struct that gains, loses or renames
//! a field fails to compile) and, per field, the WIT spelling of its Rust type
//! against the descriptor in [`super::types`] (a retyped field fails the WIT
//! test; the WIT order is the descriptor's, held by the byte-for-byte test,
//! a struct's declaration order binding nothing). The host-only fields (a load row's
//! resident-count flag, a candidate's position and
//! sample instant, a request's host clock, a lease's issue instant and
//! session verdict, a target's and a pair's positions) are named as such.
//! `dispatched` has no struct of its own: the host composes it from a lease,
//! a target and a side.

use crate::{
    decision::{Pair, Target},
    lease::{Lease, LeaseId, LeaseKind, Outcome, Side, TransportClass},
    sessions::SessionVerdict,
    view::{
        Candidate, HashKind, Header, Health, Identity, LoadRow, MediaCounts, OverloadRegime, Phase,
        PoolFacts, RequestFacts, Role, SignalSource, WireModes, WorkerId, ZoneCapacity,
    },
};

/// The WIT spelling of a Rust type a record carries.
pub(crate) trait Wit {
    const TY: &'static str;
}

macro_rules! wit {
    ($($t:ty => $s:literal),+ $(,)?) => {
        $(impl Wit for $t {
            const TY: &'static str = $s;
        })+
    };
}

wit! {
    u8 => "u8",
    u16 => "u16",
    u32 => "u32",
    u64 => "u64",
    i32 => "s32",
    bool => "bool",
    &str => "string",
    WorkerId => "u64",
    LeaseId => "u64",
    Role => "role",
    Phase => "phase",
    Side => "side",
    HashKind => "hash-kind",
    SignalSource => "signal-source",
    OverloadRegime => "overload-regime",
    LeaseKind => "lease-kind",
    TransportClass => "transport-class",
    Health => "health",
    WireModes => "wire-modes",
    Identity<'_> => "identity",
    LoadRow => "load",
    MediaCounts => "media",
    ZoneCapacity => "zone-capacity",
    Pair => "pair",
    &[u64] => "list<u64>",
    &[WorkerId] => "list<u64>",
    &[Header<'_>] => "list<tuple<u8, string>>",
    &[ZoneCapacity] => "list<zone-capacity>",
    Option<LoadRow> => "option<load>",
    Option<Pair> => "option<pair>",
    // Host-only: a lease's session verdict never reaches a guest.
    SessionVerdict => "host-only",
}

// The spellings above that look inside a type, tied to it: an inner width that
// moves stops the build here, the table being typed by hand.
const _: fn(WorkerId) -> u64 = |id| id.0;
const _: fn(LeaseId) -> u64 = |id| id.0;
const _: fn(Header<'_>) -> (u8, &str) = |Header { name, value }| (name, value);

fn ty<T: Wit>(_: &T) -> &'static str {
    T::TY
}

/// What one probe found: the record's WIT name and the struct's fields in
/// the struct's order as (WIT name, WIT type), the host-only ones left out,
/// and the WIT-only fields the host composes in front (`completed` carries
/// the lease its outcome belongs to).
pub(crate) struct Probe {
    pub(crate) record: &'static str,
    pub(crate) before: &'static [(&'static str, &'static str)],
    pub(crate) fields: Vec<(&'static str, &'static str)>,
}

/// The WIT name of a Rust field: kebab-case.
fn kebab(field: &'static str) -> String {
    field.replace('_', "-")
}

macro_rules! probe {
    ($record:literal, before [$($bn:literal: $bt:literal),*], $ty:ident { $($field:ident),+ $(,)? } host [$($host:ident),*], $value:expr) => {{
        // The pattern names every field without `..`: the exhaustiveness
        // proof, and the bindings each field's type for its WIT spelling.
        let $ty { $($field),+ } = $value;
        $(let _ = $host;)*
        let host: &[&str] = &[$(stringify!($host)),*];
        let fields: Vec<(&'static str, &'static str)> = vec![$((stringify!($field), ty($field))),+];
        let fields = fields.into_iter().filter(|(name, _)| !host.contains(name)).collect();
        Probe { record: $record, before: &[$(($bn, $bt)),*], fields }
    }};
}

/// Every record's probe, over a default value of its struct.
pub(crate) fn all() -> Vec<Probe> {
    let candidate = Candidate::default();
    let load = LoadRow::default();
    let media = MediaCounts::default();
    let request = RequestFacts::default();
    let zone = ZoneCapacity::default();
    let pool = PoolFacts::default();
    let lease = Lease::default();
    let pair = Pair::default();
    let target = Target::default();
    let outcome = Outcome {
        side: Side::Single,
        served: WorkerId(0),
        status: 0,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    vec![
        probe!("load", before [], LoadRow {
            running, queued, waiting_uncached_tokens, kv_usage_permille, active_kv_permille,
            live_rate, resident_tokens, resident_counted, report_age_ms, source, max_running,
        } host [resident_counted], &load),
        probe!("candidate", before [], Candidate {
            position, id, role, dp_rank, dp_size, health, wire_modes, engine_kind, connector,
            in_flight, load, load_sampled_at_micros, overlap_blocks, predicted_blocks,
            session_hit, session_in_flight, pair_class, has_capacity, has_room,
            admitted_age_ms, indexed_growth, zone, labels, ec_compatible, vision, host_local,
            active_sessions, key, indexed_blocks,
        } host [position, load_sampled_at_micros], &candidate),
        probe!("media", before [], MediaCounts { images, audio, video, bytes } host [], &media),
        probe!("request", before [], RequestFacts {
            model, namespace_seed, tenant, priority, streaming, max_output_tokens,
            prompt_tokens, block_hashes, hash_kind, session_key, phase, attempt, tried,
            budget_micros, headers, media, seed, block_size, now_micros,
        } host [now_micros], &request),
        probe!("zone-capacity", before [], ZoneCapacity { zone, healthy, total } host [], &zone),
        probe!("pool", before [], PoolFacts {
            pool_size, membership_epoch, ring_epoch, regime, generation, zones,
        } host [], &pool),
        probe!("lease", before [], Lease {
            id, kind, side, target_index, generation, session_key, session, issued_micros,
            tokens,
        } host [session, issued_micros], &lease),
        probe!("pair", before [], Pair { decode_position, decode_id, decode_dp_rank } host [decode_position], &pair),
        probe!("target", before [], Target { position, id, dp_rank, pair } host [position], &target),
        probe!("completed", before ["lease": "lease"], Outcome {
            side, served, status, class, ttft_micros, tokens_in, tokens_out, duration_micros,
        } host [], &outcome),
    ]
}

/// Each probe against its descriptor: the same fields, in the same order,
/// with the same types, the WIT-only ones in front. Empty when the engine's
/// types and the WIT agree; each line names a record that drifted.
pub fn drift() -> Vec<String> {
    let mut drift = Vec::new();
    for probe in all() {
        let Some(item) = super::record(probe.record) else {
            drift.push(format!("{}: no record in the descriptors", probe.record));
            continue;
        };
        let expected: Vec<(String, &str)> = probe
            .before
            .iter()
            .map(|(n, t)| ((*n).to_string(), *t))
            .chain(probe.fields.iter().map(|(n, t)| (kebab(n), *t)))
            .collect();
        let actual: Vec<(String, &str)> = item
            .fields()
            .iter()
            .map(|f| (f.name.to_string(), f.ty))
            .collect();
        if expected != actual {
            drift.push(format!(
                "{}: the struct reads {expected:?}, the descriptor {actual:?}",
                probe.record
            ));
        }
    }
    drift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_record_probe_matches_its_descriptor() {
        let drift = drift();
        assert!(drift.is_empty(), "{}", drift.join("\n"));
    }

    #[test]
    fn the_probes_cover_every_record_the_host_does_not_compose() {
        let probed: Vec<&str> = all().into_iter().map(|p| p.record).collect();
        let records: Vec<&str> = super::super::types::TYPES
            .iter()
            .filter(|i| matches!(i, super::super::Item::Record { .. }))
            .map(|i| i.name())
            .collect();
        for record in records {
            assert!(
                record == "dispatched" || probed.contains(&record),
                "{record} has no probe"
            );
        }
    }
}

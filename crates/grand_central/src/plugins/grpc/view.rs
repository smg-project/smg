//! The view and the lifecycle as the protocol carries them, and the answer
//! as the host reads it back. The exposure decides what of the identities
//! and headers crosses, as it does for the WebAssembly worlds, and the
//! plugin's declared needs whether the request's block hashes do. A frame is
//! a fresh message (the stream takes it whole); the messages rendered from
//! the engine's types as the WIT is are a later step.

use crate::{
    document::spec::Needs,
    engine::{
        decision,
        decision::{Decision, Target},
        view::{
            AffinityFacts, Candidate, HashKind, Health, Identity, LoadRow, OverloadRegime, Phase,
            PoolFacts, RequestFacts, Role, SignalSource, WireModes, ZoneCapacity,
        },
    },
    plugins::shared::{
        config::{Exposure, Reveal},
        shown_hashes, shown_size,
    },
    service::proto,
    tracking::lease::{wire, Lease, LeaseKind, Membership, Outcome, ReleaseReason, Side, Signal},
};

/// The stage bits of the protocol, as `Stage` numbers them.
pub(crate) const FILTER: u32 = proto::Stage::Filter as u32;
pub(crate) const SCORE: u32 = proto::Stage::Score as u32;
pub(crate) const PICK: u32 = proto::Stage::Pick as u32;

fn health_bits(h: Health) -> u32 {
    // The same numbering as the WIT's `health` flags and the packed row.
    let mut bits = 0;
    for (on, bit) in [
        (h.healthy, proto::HealthFlag::Healthy),
        (h.breaker_open, proto::HealthFlag::BreakerOpen),
        (h.stalled, proto::HealthFlag::Stalled),
        (h.draining, proto::HealthFlag::Draining),
        (h.overloaded, proto::HealthFlag::Overloaded),
    ] {
        if on {
            bits |= bit as u32;
        }
    }
    bits
}

fn wire_bits(w: WireModes) -> u32 {
    let mut bits = 0;
    for (on, bit) in [
        (w.contains(WireModes::HTTP), proto::WireMode::Http),
        (w.contains(WireModes::GRPC), proto::WireMode::Grpc),
        (w.contains(WireModes::ZMQ), proto::WireMode::Zmq),
    ] {
        if on {
            bits |= bit as u32;
        }
    }
    bits
}

fn role(r: Role) -> i32 {
    (match r {
        Role::Single => proto::Role::Single,
        Role::Prefill => proto::Role::Prefill,
        Role::Decode => proto::Role::Decode,
        Role::Encode => proto::Role::Encode,
    }) as i32
}

fn phase(p: Phase) -> i32 {
    (match p {
        Phase::Single => proto::Phase::Single,
        Phase::Prefill => proto::Phase::Prefill,
        Phase::Decode => proto::Phase::Decode,
        Phase::Encode => proto::Phase::Encode,
    }) as i32
}

fn side(s: Side) -> i32 {
    proto::Side::from(s).into()
}

fn hash_kind(k: HashKind) -> i32 {
    (match k {
        HashKind::Token => proto::HashKind::Token,
        HashKind::Text => proto::HashKind::Text,
    }) as i32
}

fn source(s: SignalSource) -> i32 {
    (match s {
        SignalSource::None => proto::SignalSource::None,
        SignalSource::Events => proto::SignalSource::Events,
        SignalSource::Scrape => proto::SignalSource::Scrape,
        SignalSource::Relay => proto::SignalSource::Relay,
        SignalSource::Poll => proto::SignalSource::Poll,
    }) as i32
}

fn regime(r: OverloadRegime) -> i32 {
    (match r {
        OverloadRegime::None => proto::OverloadRegime::None,
        OverloadRegime::Partial => proto::OverloadRegime::Partial,
        OverloadRegime::Saturated => proto::OverloadRegime::Saturated,
    }) as i32
}

fn load(row: &LoadRow) -> proto::LoadRow {
    proto::LoadRow {
        running: row.running,
        queued: row.queued,
        waiting_uncached_tokens: row.waiting_uncached_tokens,
        kv_usage_permille: u32::from(row.kv_usage_permille),
        active_kv_permille: u32::from(row.active_kv_permille),
        live_rate: row.live_rate,
        resident_tokens: row.resident_tokens,
        report_age_ms: row.report_age_ms,
        source: source(row.source),
        max_running: row.max_running,
    }
}

/// One candidate as the frame carries it: the row, and the affinity facts the
/// host holds for it beside the row (`held`), which the record carries as
/// five of its own fields. Its requests in flight are what the request's
/// facts read for it (`in_flight`).
fn candidate(c: &Candidate, held: AffinityFacts, in_flight: u32) -> proto::Candidate {
    proto::Candidate {
        id: c.id.0,
        role: role(c.role),
        dp_rank: u32::from(c.dp_rank),
        dp_size: u32::from(c.dp_size),
        health: health_bits(c.health),
        wire_modes: wire_bits(c.wire_modes),
        engine_kind: u32::from(c.engine_kind),
        connector: u32::from(c.connector),
        in_flight,
        load: c.load.as_ref().map(load),
        overlap_blocks: held.overlap_blocks,
        predicted_blocks: held.predicted_blocks,
        session_hit: c.session_hit,
        session_in_flight: u32::from(c.session_in_flight),
        pair_class: c.pair_class,
        has_capacity: c.has_capacity,
        has_room: c.has_room,
        admitted_age_ms: held.admitted_age_ms,
        indexed_growth: held.indexed_growth,
        zone: u32::from(c.zone),
        labels: c.labels,
        ec_compatible: c.ec_compatible,
        vision: c.vision,
        host_local: c.host_local,
        active_sessions: u32::from(c.active_sessions),
        key: c.key,
        indexed_blocks: held.indexed_blocks,
    }
}

/// An identity under the plugin's exposure: withheld (no message), its
/// fingerprint, or the plain form when the host holds it (its fingerprint
/// otherwise).
fn identity(id: &Identity<'_>, reveal: Reveal) -> Option<proto::Identity> {
    exposed(id, reveal, || id.hash)
}

/// [`identity`] with the hash form's value made on demand: the exposure is
/// asked first, and `hash` runs only where that form crosses.
fn exposed(
    id: &Identity<'_>,
    reveal: Reveal,
    hash: impl FnOnce() -> u64,
) -> Option<proto::Identity> {
    if id.is_none() {
        return None;
    }
    let form = match (reveal, id.plain) {
        (Reveal::None, _) => return None,
        (Reveal::Plain, Some(plain)) => proto::identity::Form::Plain(plain.to_owned()),
        (Reveal::Hash | Reveal::Plain, _) => proto::identity::Form::Hash(hash()),
    };
    Some(proto::Identity { form: Some(form) })
}

/// The request's session key as a plugin sees it under its exposure: the
/// session's value in the hash form, which is what a lease carries, so a
/// plugin matches the two; the key itself in the plain form. A value the
/// plugin is not shown is never made.
fn session_identity(facts: &RequestFacts<'_>, reveal: Reveal) -> Option<proto::Identity> {
    exposed(&facts.session_key, reveal, || facts.session().get())
}

/// The exposure as the hello states it.
pub(crate) fn exposure(e: &Exposure) -> proto::Exposure {
    fn mode(r: Reveal) -> i32 {
        (match r {
            Reveal::None => proto::exposure::Mode::None,
            Reveal::Hash => proto::exposure::Mode::Hash,
            Reveal::Plain => proto::exposure::Mode::Plain,
        }) as i32
    }
    proto::Exposure {
        tenant: mode(e.tenant),
        session_key: mode(e.session_key),
        headers: (0..=u8::MAX)
            .filter(|p| e.headers.contains(*p))
            .map(u32::from)
            .collect(),
    }
}

/// The request under the plugin's exposure and its declared needs. The
/// budget is written when the frame is sent: what the decision has left at
/// that instant.
fn request(facts: &RequestFacts<'_>, exposure: &Exposure, needs: Needs) -> proto::RequestFacts {
    proto::RequestFacts {
        model: facts.model.to_owned(),
        namespace_seed: facts.namespace_seed,
        tenant: identity(&facts.tenant, exposure.tenant),
        priority: facts.priority,
        streaming: facts.streaming,
        max_output_tokens: facts.max_output_tokens,
        prompt_tokens: shown_size(facts, needs),
        block_hashes: shown_hashes(facts, needs).to_vec(),
        hash_kind: hash_kind(facts.hash_kind),
        session_key: session_identity(facts, exposure.session_key),
        phase: phase(facts.phase),
        attempt: u32::from(facts.attempt),
        tried: facts.tried.iter().map(|w| w.0).collect(),
        budget_micros: 0,
        headers: facts
            .headers
            .iter()
            .filter(|h| exposure.headers.contains(h.name))
            .map(|h| proto::Header {
                name: u32::from(h.name),
                value: h.value.to_owned(),
            })
            .collect(),
        media: Some(proto::MediaCounts {
            images: u32::from(facts.media.images),
            audio: u32::from(facts.media.audio),
            video: u32::from(facts.media.video),
            bytes: facts.media.bytes,
        }),
        seed: facts.seed,
        prompt_text: Vec::new(),
        block_size: facts.block_size,
        // The client's key is the host's and the controller's: no exposure
        // shows it.
        client_key: 0,
        // Who handed the session key is the host's and the controller's
        // too: no exposure shows it.
        session_key_from_rid: false,
    }
}

fn zone(z: &ZoneCapacity) -> proto::ZoneCapacity {
    proto::ZoneCapacity {
        zone: u32::from(z.zone),
        healthy: z.healthy,
        total: z.total,
    }
}

fn pool(p: &PoolFacts<'_>) -> proto::PoolFacts {
    proto::PoolFacts {
        pool_size: p.pool_size,
        membership_epoch: p.membership_epoch,
        ring_epoch: p.ring_epoch,
        regime: regime(p.regime),
        generation: p.generation,
        zones: p.zones.iter().map(zone).collect(),
    }
}

/// One decision's frame: the view under the plugin's exposure and its
/// declared needs. The stages asked, the two budgets and a pick-only ask's
/// combined scores are written when the frame is sent ([`stamp`]).
pub(crate) fn decide(
    decision_id: u64,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    pool_facts: &PoolFacts<'_>,
    exposure: &Exposure,
    needs: Needs,
) -> proto::PluginDecide {
    // A candidate's affinity facts ride the request's column, by its
    // index: the same index reads the row and its entry.
    let rows: Vec<proto::Candidate> = (0u32..)
        .zip(candidates)
        .map(|(index, c)| candidate(c, facts.affinity_at(index), facts.in_flight(c)))
        .collect();
    proto::PluginDecide {
        decision_id,
        request: Some(request(facts, exposure, needs)),
        candidates: rows,
        pool: Some(pool(pool_facts)),
        stages: 0,
        scores_in: Vec::new(),
        budget_micros: 0,
    }
}

/// Writes what a frame carries at its send: the stages asked, the plugin's
/// own deadline for it (`frame_micros`) and what the whole decision has left
/// (`remaining_micros`, never less than the frame's).
pub(crate) fn stamp(
    frame: &mut proto::PluginDecide,
    stages: u32,
    frame_micros: u32,
    remaining_micros: u32,
) {
    frame.stages = stages;
    frame.budget_micros = frame_micros;
    if let Some(request) = frame.request.as_mut() {
        request.budget_micros = remaining_micros.max(frame_micros);
    }
}

// --- the lifecycle ---------------------------------------------------------

fn lease(l: &Lease, exposure: &Exposure) -> proto::Lease {
    proto::Lease {
        id: l.id.0,
        kind: (match l.kind {
            LeaseKind::Credit => proto::LeaseKind::Credit,
            LeaseKind::Reservation => proto::LeaseKind::Reservation,
        }) as i32,
        side: side(l.side),
        target_index: u32::from(l.target_index),
        generation: l.generation,
        session_key: match exposure.session_key {
            Reveal::None => 0,
            Reveal::Hash | Reveal::Plain => l.session.get(),
        },
        tokens: l.prompt_size(),
        term: 0,
        dispatch_ttl_ms: 0,
        // Bit 63 set is a gateway-minted lease (the design's two id spaces).
        origin: (if l.id.0 >> 63 == 1 {
            proto::LeaseOrigin::Local
        } else {
            proto::LeaseOrigin::Controller
        }) as i32,
    }
}

fn target(t: &Target) -> proto::Target {
    proto::Target {
        id: t.id.0,
        dp_rank: u32::from(t.dp_rank),
        pair: t.pair.map(|p| proto::Pair {
            decode_id: p.decode_id.0,
            decode_dp_rank: u32::from(p.decode_dp_rank),
        }),
    }
}

fn decision(d: &Decision, l: &Lease, exposure: &Exposure) -> proto::Decision {
    use proto::decision::Outcome;
    let (outcome, t) = match d {
        Decision::Pick(pick) => (
            Outcome::Pick(proto::Pick {
                targets: pick.targets.iter().map(target).collect(),
            }),
            &pick.trace,
        ),
        Decision::Shed(shed) => (
            Outcome::Shed(proto::Shed {
                reason: (match shed.reason {
                    decision::ShedReason::Saturated => proto::ShedReason::Saturated,
                    decision::ShedReason::FlowLimit => proto::ShedReason::FlowLimit,
                    decision::ShedReason::NoEndpoints => proto::ShedReason::NoEndpoints,
                    decision::ShedReason::PolicyDeclined => proto::ShedReason::PolicyDeclined,
                    decision::ShedReason::Deadline => proto::ShedReason::Deadline,
                }) as i32,
                emptied_by: shed.emptied_by.to_owned(),
                retry_after_secs: u32::from(shed.retry_after_secs),
                wait_allowed: shed.wait_allowed,
            }),
            &shed.trace,
        ),
        Decision::Queue(queue) => (
            Outcome::Queue(proto::Queue {
                ticket: queue.ticket,
                max_wait_micros: queue.max_wait_micros,
            }),
            &queue.trace,
        ),
    };
    proto::Decision {
        leases: vec![lease(l, exposure)],
        outcome: Some(outcome),
        trace: Some(t.wire()),
    }
}

pub(crate) fn issued(
    l: &Lease,
    d: &Decision,
    exposure: &Exposure,
) -> proto::lifecycle_event::Event {
    proto::lifecycle_event::Event::Issued(proto::Issued {
        decision: Some(decision(d, l, exposure)),
        decision_id: l.id.0,
    })
}

pub(crate) fn dispatched(
    l: &Lease,
    s: Side,
    since_decision_micros: u32,
) -> proto::lifecycle_event::Event {
    proto::lifecycle_event::Event::Dispatched(proto::Dispatched {
        lease_id: l.id.0,
        target_index: u32::from(l.target_index),
        side: side(s),
        since_decision_micros,
    })
}

pub(crate) fn completed(l: &Lease, o: &Outcome) -> proto::lifecycle_event::Event {
    proto::lifecycle_event::Event::Completed(wire::completed(l.id, o))
}

pub(crate) fn released(l: &Lease, r: ReleaseReason) -> proto::lifecycle_event::Event {
    proto::lifecycle_event::Event::Released(proto::Released {
        what: Some(proto::released::What::LeaseId(l.id.0)),
        reason: proto::released::Reason::from(r).into(),
        side: side(l.side),
    })
}

/// A membership word as the plugin contract carries it. A stream whose
/// protocol version has no kind for a word is not sent it ([`since`]).
pub(crate) fn membership(m: &Membership<'_>) -> proto::lifecycle_event::Event {
    use proto::membership_change::Kind;
    let mut models = Vec::new();
    let (kind, worker_id, old_worker_id, ids) = match m {
        Membership::Left { id, models: left } => {
            models.extend_from_slice(left);
            (Kind::Left, id.0, 0, Vec::new())
        }
        Membership::Added(w) => (Kind::Added, w.0, 0, Vec::new()),
        Membership::Removed(w) => (Kind::Removed, w.0, 0, Vec::new()),
        Membership::Replaced { old, new } => (Kind::Replaced, new.0, old.0, Vec::new()),
        Membership::Relabelled(w) => (Kind::Relabelled, w.0, 0, Vec::new()),
        Membership::Resized(w) => (Kind::Resized, w.0, 0, Vec::new()),
        Membership::Snapshot(workers) => {
            (Kind::Snapshot, 0, 0, workers.iter().map(|w| w.0).collect())
        }
    };
    proto::lifecycle_event::Event::Membership(proto::MembershipChange {
        kind: kind as i32,
        worker_id,
        old_worker_id,
        ids,
        models,
    })
}

/// What [`since`] says of the word [`membership`] makes of `change`, read
/// before that word is made.
#[inline]
pub(crate) fn since_of(change: &Membership<'_>) -> u32 {
    match change {
        Membership::Left { .. } => 2,
        Membership::Added(_)
        | Membership::Removed(_)
        | Membership::Replaced { .. }
        | Membership::Relabelled(_)
        | Membership::Resized(_)
        | Membership::Snapshot(_) => 1,
    }
}

/// The first protocol version whose streams carry `event`. A stream on an
/// older one is not sent it, and hears what it heard before the word came.
#[inline]
pub(crate) fn since(event: &proto::lifecycle_event::Event) -> u32 {
    use proto::{lifecycle_event::Event, membership_change::Kind};
    match event {
        Event::Membership(change) if change.kind == Kind::Left as i32 => 2,
        _ => 1,
    }
}

/// A signal; a veto's instant on the host's clock becomes the microseconds
/// it has left at `now_micros` on that clock (`0` when none).
pub(crate) fn signal(s: &Signal<'_>, now_micros: u64) -> proto::lifecycle_event::Event {
    use proto::signal::Kind;
    let (worker_id, dp_rank, kind) = match s {
        Signal::Load {
            worker,
            dp_rank,
            row,
            ..
        } => (worker.0, u32::from(*dp_rank), Kind::Load(load(row))),
        Signal::Health { worker, health: h } => (worker.0, 0, Kind::Health(health_bits(*h))),
        Signal::GatewayVeto {
            worker,
            until_micros,
        } => (
            worker.0,
            0,
            Kind::GatewayVetoMicros(
                u32::try_from(until_micros.saturating_sub(now_micros)).unwrap_or(u32::MAX),
            ),
        ),
    };
    proto::lifecycle_event::Event::Signal(proto::Signal {
        worker_id,
        dp_rank,
        kind: Some(kind),
    })
}

// --- the answer ------------------------------------------------------------

/// Why an answer broke the contract; the log line names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Invalid {
    /// A declined stage that carries an answer.
    DeclinedAndAnswered,
    /// A survivor position out of range or named twice.
    Survivor,
    /// Not one finite score per candidate.
    Scores,
    /// A tie band outside its range or without scores.
    Band,
    /// A pick outside the surviving set.
    Pick,
    /// More counter increments than the plugin declared counters.
    Counters,
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::DeclinedAndAnswered => "a declined stage carries an answer",
            Self::Survivor => "a surviving position is out of range or repeated",
            Self::Scores => "the scores are not one finite value per candidate",
            Self::Band => "the tie band is not a finite value from 0 to 1 beside its scores",
            Self::Pick => "the pick is outside the surviving set",
            Self::Counters => "more counter increments than declared counters",
        })
    }
}

/// One frame's answer as the stages read it, validated once at the read.
#[derive(Debug, Default)]
pub(crate) struct Answer {
    /// The stages the plugin declined, as bits; an asked stage left
    /// unanswered is declined too.
    pub(crate) declined: u32,
    /// A budget decline: the plugin's Busy.
    pub(crate) busy: bool,
    /// The survivors as bits over the candidates, when the filter answered.
    pub(crate) mask: Vec<u64>,
    pub(crate) mask_set: bool,
    /// One score per candidate, clamped by the host, when the scorer
    /// answered.
    pub(crate) scores: Vec<f32>,
    /// The band the scorer gave with its scores.
    pub(crate) band: Option<f32>,
    /// The pick, when the picker answered.
    pub(crate) pick: Option<u32>,
    /// The counter increments by declared id.
    pub(crate) counters: Vec<proto::CounterDelta>,
}

pub(crate) fn bit(mask: &[u64], position: u32) -> bool {
    mask.get((position / 64) as usize)
        .is_some_and(|word| word & (1 << (position % 64)) != 0)
}

/// Reads a frame's answer under the protocol's validity rules. A present
/// `survivors` field with no positions sheds. A pick after a PICK decline
/// is unset, and an asked stage left unanswered is declined. A declined
/// stage that carries an answer is invalid. The scorer's band must be
/// finite, from 0 to 1, and beside scores. The pick is judged here against the
/// plugin's own survivors when its filter answered in the frame, and against
/// the candidate count otherwise; the picker judges it against the rows the
/// profile's filters left. The counter list is judged first: an answer
/// holds no more increments than the plugin declared counters (`declared`),
/// so what the deciding thread walks after its wait is the row's size and
/// never the plugin's.
pub(crate) fn read(
    d: proto::PluginDecision,
    asked: u32,
    count: u32,
    declared: usize,
) -> Result<Answer, Invalid> {
    if d.counters.len() > declared {
        return Err(Invalid::Counters);
    }
    let mut answer = Answer {
        counters: d.counters,
        ..Answer::default()
    };
    if let Some(decline) = &d.decline {
        answer.declined = decline.stage & asked;
        answer.busy = decline.reason == proto::decline::Reason::Budget as i32;
    }
    // The stages the decline names; a stage asked and left unanswered joins
    // them below.
    let named = answer.declined;
    // The filter.
    if named & FILTER != 0 {
        if d.survivors.is_some() {
            return Err(Invalid::DeclinedAndAnswered);
        }
    } else if asked & FILTER != 0 {
        match &d.survivors {
            Some(survivors) => {
                answer.mask.resize((count as usize).div_ceil(64), 0);
                for &p in &survivors.positions {
                    if p >= count || bit(&answer.mask, p) {
                        return Err(Invalid::Survivor);
                    }
                    answer.mask[(p / 64) as usize] |= 1 << (p % 64);
                }
                answer.mask_set = true;
            }
            None => answer.declined |= FILTER,
        }
    }
    // The scorer.
    if named & SCORE != 0 {
        if !d.scores.is_empty() || d.tie_band.is_some() {
            return Err(Invalid::DeclinedAndAnswered);
        }
    } else if asked & SCORE != 0 {
        if d.scores.is_empty() {
            if d.tie_band.is_some() {
                return Err(Invalid::Band);
            }
            answer.declined |= SCORE;
        } else {
            if d.scores.len() != count as usize || d.scores.iter().any(|s| !s.is_finite()) {
                return Err(Invalid::Scores);
            }
            if let Some(band) = d.tie_band {
                if !(0.0..=1.0).contains(&band) {
                    return Err(Invalid::Band);
                }
                answer.band = Some(band);
            }
            answer.scores = d.scores;
            for s in &mut answer.scores {
                *s = s.clamp(0.0, 1.0);
            }
        }
    }
    // The picker.
    if named & PICK != 0 {
        if d.pick.is_some() {
            return Err(Invalid::DeclinedAndAnswered);
        }
    } else if asked & PICK != 0 {
        match d.pick {
            None => answer.declined |= PICK,
            Some(p) => {
                let within = if answer.mask_set {
                    bit(&answer.mask, p)
                } else {
                    p < count
                };
                if !within {
                    return Err(Invalid::Pick);
                }
                answer.pick = Some(p);
            }
        }
    }
    Ok(answer)
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/grpc/view/tests.rs"]
mod tests;

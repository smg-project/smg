//! The view and the lifecycle as the protocol carries them, and the answer
//! as the host reads it back. The exposure decides what of the identities
//! and headers crosses, as it does for the WebAssembly worlds, and the
//! plugin's declared needs whether the request's block hashes do. A frame is
//! a fresh message (the stream takes it whole); the messages rendered from
//! the engine's types as the WIT is are a later step.

use crate::{
    counts::with_counts,
    decision::{Decision, Target, Trace},
    lease::{Lease, LeaseKind, Membership, Outcome, ReleaseReason, Side, Signal, TransportClass},
    plugins::{
        config::{Exposure, Reveal},
        shown_hashes,
    },
    proto,
    spec::Needs,
    view::{
        AffinityFacts, Candidate, HashKind, Health, Identity, LoadRow, OverloadRegime, Phase,
        PoolFacts, RequestFacts, Role, SignalSource, WireModes, ZoneCapacity,
    },
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
    (match s {
        Side::Single => proto::Side::Single,
        Side::Prefill => proto::Side::Prefill,
        Side::Decode => proto::Side::Decode,
        Side::Encode => proto::Side::Encode,
    }) as i32
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
        prompt_tokens: facts.prompt_tokens,
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
    proto::PluginDecide {
        decision_id,
        request: Some(request(facts, exposure, needs)),
        // A candidate's affinity facts ride the request's column, by its
        // index: the same index reads the row and its entry.
        candidates: with_counts!(facts, |counts| (0u32..)
            .zip(candidates)
            .map(|(index, c)| candidate(c, facts.affinity_at(index), counts(c)))
            .collect()),
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
        tokens: l.tokens,
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

fn trace(t: &Trace) -> proto::Trace {
    proto::Trace {
        profile: u32::from(t.profile),
        survivors: t.survivors.get(),
        top: t.top[..usize::from(t.top_len)]
            .iter()
            .map(|(id, score)| proto::Scored {
                worker_id: id.0,
                score: *score,
                contributions: Vec::new(),
                dp_rank: 0,
            })
            .collect(),
        seed: t.seed,
        generation: t.generation,
        plan: 0,
        drops: Vec::new(),
        picker: 0,
        shadow_pick: 0,
        fallback: false,
        affinity_source: 0,
        drew_first: t.drew_first(),
        lifted: proto::LiftedVeto::from(t.lifted) as i32,
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
                    crate::decision::ShedReason::Saturated => proto::ShedReason::Saturated,
                    crate::decision::ShedReason::FlowLimit => proto::ShedReason::FlowLimit,
                    crate::decision::ShedReason::NoEndpoints => proto::ShedReason::NoEndpoints,
                    crate::decision::ShedReason::PolicyDeclined => {
                        proto::ShedReason::PolicyDeclined
                    }
                    crate::decision::ShedReason::Deadline => proto::ShedReason::Deadline,
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
        trace: Some(trace(t)),
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
    proto::lifecycle_event::Event::Completed(proto::Completed {
        lease_id: l.id.0,
        side: side(o.side),
        served: o.served.0,
        status: u32::from(o.status),
        class: (match o.class {
            TransportClass::Ok => proto::completed::Class::Ok,
            TransportClass::ClientError => proto::completed::Class::ClientError,
            TransportClass::ServerError => proto::completed::Class::ServerError,
            TransportClass::ConnectFailed => proto::completed::Class::ConnectFailed,
            TransportClass::Timeout => proto::completed::Class::Timeout,
            TransportClass::Cancelled => proto::completed::Class::Cancelled,
        }) as i32,
        ttft_micros: o.ttft_micros,
        tokens_in: o.tokens_in,
        tokens_out: o.tokens_out,
        duration_micros: o.duration_micros,
    })
}

pub(crate) fn released(l: &Lease, r: ReleaseReason) -> proto::lifecycle_event::Event {
    proto::lifecycle_event::Event::Released(proto::Released {
        what: Some(proto::released::What::LeaseId(l.id.0)),
        reason: (match r {
            ReleaseReason::Abandoned => proto::released::Reason::Abandoned,
            ReleaseReason::LateDecision => proto::released::Reason::LateDecision,
            ReleaseReason::ClientGone => proto::released::Reason::ClientGone,
            ReleaseReason::CommitFailed => proto::released::Reason::CommitFailed,
            ReleaseReason::Expired => proto::released::Reason::Expired,
            ReleaseReason::WorkerRemoved => proto::released::Reason::WorkerRemoved,
        }) as i32,
        side: side(l.side),
    })
}

pub(crate) fn membership(m: &Membership<'_>) -> proto::lifecycle_event::Event {
    use proto::membership_change::Kind;
    let (kind, worker_id, old_worker_id, ids) = match m {
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
    })
}

/// A signal; a veto's instant on the host's clock becomes the microseconds
/// it has left at `now_micros` on that clock (`0` when none).
pub(crate) fn signal(s: &Signal, now_micros: u64) -> proto::lifecycle_event::Event {
    use proto::signal::Kind;
    let (worker_id, dp_rank, kind) = match s {
        Signal::Load {
            worker,
            dp_rank,
            row,
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
    /// A declined stage whose answer field is filled.
    DeclinedAndAnswered,
    /// A survivor position out of range or named twice.
    Survivor,
    /// Not one finite score per candidate.
    Scores,
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
    /// The pick, when the picker answered.
    pub(crate) pick: Option<u32>,
    /// The counter increments by declared id.
    pub(crate) counters: Vec<proto::CounterDelta>,
}

pub(crate) fn bit(mask: &[u64], position: u32) -> bool {
    mask.get((position / 64) as usize)
        .is_some_and(|word| word & (1 << (position % 64)) != 0)
}

/// Reads a frame's answer under the validity rules (design appendix B and
/// the protocol's own comments): each asked stage answers its own field;
/// `survivors` present and empty sheds; a pick after a PICK decline is
/// unset; an asked stage left unanswered is declined; a declined stage
/// with its field filled is invalid. The pick is judged here against the
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
        if !d.scores.is_empty() {
            return Err(Invalid::DeclinedAndAnswered);
        }
    } else if asked & SCORE != 0 {
        if d.scores.is_empty() {
            answer.declined |= SCORE;
        } else {
            if d.scores.len() != count as usize || d.scores.iter().any(|s| !s.is_finite()) {
                return Err(Invalid::Scores);
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
mod tests {
    use super::{proto, Answer, Invalid, FILTER, PICK, SCORE};

    /// The stages' rules, under a plugin that declared no counter.
    fn read(d: proto::PluginDecision, asked: u32, count: u32) -> Result<Answer, Invalid> {
        super::read(d, asked, count, 0)
    }

    fn decision(
        survivors: Option<Vec<u32>>,
        scores: Vec<f32>,
        pick: Option<u32>,
        decline: Option<(u32, i32)>,
    ) -> proto::PluginDecision {
        proto::PluginDecision {
            decision_id: 7,
            survivors: survivors.map(|positions| proto::Survivors { positions }),
            scores,
            pick,
            decline: decline.map(|(stage, reason)| proto::Decline {
                stage,
                reason,
                detail: String::new(),
            }),
            counters: Vec::new(),
        }
    }

    #[test]
    fn each_asked_stage_answers_its_own_field_and_an_unanswered_one_is_declined() {
        let a = read(
            decision(Some(vec![2, 0]), vec![0.5, 2.0, -1.0], Some(2), None),
            FILTER | SCORE | PICK,
            3,
        )
        .unwrap();
        assert!(
            a.mask_set
                && super::bit(&a.mask, 0)
                && !super::bit(&a.mask, 1)
                && super::bit(&a.mask, 2)
        );
        assert_eq!(a.scores, vec![0.5, 1.0, 0.0], "clamped by the host");
        assert_eq!(a.pick, Some(2));
        assert_eq!(a.declined, 0);

        let a = read(
            decision(None, Vec::new(), None, None),
            FILTER | SCORE | PICK,
            3,
        )
        .unwrap();
        assert_eq!(
            a.declined,
            FILTER | SCORE | PICK,
            "nothing answered: every asked stage is declined"
        );
        assert!(!a.busy);

        let a = read(
            decision(Some(Vec::new()), Vec::new(), None, None),
            FILTER,
            3,
        )
        .unwrap();
        assert!(
            a.mask_set && a.mask.iter().all(|w| *w == 0),
            "present and empty sheds, never a fallback"
        );
    }

    #[test]
    fn a_decline_names_its_stages_and_a_budget_decline_is_busy() {
        let a = read(
            decision(
                None,
                Vec::new(),
                None,
                Some((SCORE | PICK, proto::decline::Reason::Budget as i32)),
            ),
            FILTER | SCORE | PICK,
            2,
        )
        .unwrap();
        assert_eq!(
            a.declined,
            FILTER | SCORE | PICK,
            "the filter asked and unanswered is declined too"
        );
        assert!(a.busy);
        let a = read(
            decision(
                Some(vec![1]),
                Vec::new(),
                None,
                Some((PICK, proto::decline::Reason::NotApplicable as i32)),
            ),
            FILTER | PICK,
            2,
        )
        .unwrap();
        assert_eq!(a.declined, PICK);
        assert!(a.mask_set);
    }

    #[test]
    fn the_validity_rules_refuse_the_answer_whole() {
        assert_eq!(
            read(decision(Some(vec![3]), Vec::new(), None, None), FILTER, 3).unwrap_err(),
            Invalid::Survivor
        );
        assert_eq!(
            read(
                decision(Some(vec![1, 1]), Vec::new(), None, None),
                FILTER,
                3
            )
            .unwrap_err(),
            Invalid::Survivor
        );
        assert_eq!(
            read(decision(None, vec![0.1], None, None), SCORE, 3).unwrap_err(),
            Invalid::Scores
        );
        assert_eq!(
            read(
                decision(None, vec![0.1, f32::NAN, 0.3], None, None),
                SCORE,
                3
            )
            .unwrap_err(),
            Invalid::Scores
        );
        assert_eq!(
            read(
                decision(Some(vec![0]), Vec::new(), Some(1), None),
                FILTER | PICK,
                3
            )
            .unwrap_err(),
            Invalid::Pick
        );
        assert_eq!(
            read(decision(None, Vec::new(), Some(3), None), PICK, 3).unwrap_err(),
            Invalid::Pick
        );
        assert_eq!(
            read(
                decision(Some(vec![0]), Vec::new(), None, Some((FILTER, 1))),
                FILTER,
                3
            )
            .unwrap_err(),
            Invalid::DeclinedAndAnswered
        );
        assert_eq!(
            read(
                decision(None, Vec::new(), Some(0), Some((PICK, 1))),
                PICK,
                3
            )
            .unwrap_err(),
            Invalid::DeclinedAndAnswered
        );
    }

    #[test]
    fn more_increments_than_declared_counters_break_the_contract() {
        let counted = |increments: usize| proto::PluginDecision {
            counters: vec![proto::CounterDelta { id: 0, by: 1 }; increments],
            ..decision(None, Vec::new(), Some(2), None)
        };
        let within = super::read(counted(2), PICK, 3, 2).unwrap();
        assert_eq!(within.counters.len(), 2, "an id named twice is no fault");
        assert_eq!(
            super::read(counted(3), PICK, 3, 2).unwrap_err(),
            Invalid::Counters
        );
        assert_eq!(
            super::read(counted(1), PICK, 3, 0).unwrap_err(),
            Invalid::Counters,
            "a plugin that declared none sends none"
        );
    }

    #[test]
    fn a_pick_without_a_filter_answer_is_judged_against_the_candidate_count() {
        let a = read(decision(None, Vec::new(), Some(2), None), PICK, 3).unwrap();
        assert_eq!(a.pick, Some(2));
    }

    /// A frame carries the request's block hashes for a row that declares
    /// them in either form, the first of a list past the contract's range,
    /// and none for a row that declares neither, whatever else it declares.
    #[test]
    fn a_frame_carries_the_block_hashes_its_row_declares() {
        use crate::{
            plugins::config::Exposure,
            spec::{Needs, MAX_BLOCK_HASHES},
            view::{PoolFacts, RequestFacts},
        };

        let most = usize::try_from(MAX_BLOCK_HASHES).expect("the range fits a length");
        let held: Vec<u64> = (1..).take(most + 2).collect();
        let sent = |list: &[u64], needs: Needs| {
            let facts = RequestFacts {
                model: "m",
                block_hashes: list,
                ..RequestFacts::default()
            };
            super::decide(
                1,
                &facts,
                &[],
                &PoolFacts::default(),
                &Exposure::default(),
                needs,
            )
            .request
            .expect("a frame carries its request")
            .block_hashes
        };
        let tokens = Needs {
            token_hashes: true,
            ..Needs::default()
        };
        let text = Needs {
            text_hashes: true,
            ..Needs::default()
        };
        // Every other need, and neither form of the hashes.
        let neither = Needs {
            text: true,
            token_hashes: false,
            text_hashes: false,
            loads: true,
            events: true,
            ring: true,
            headers: true,
            media: true,
            active_sessions: true,
            session_in_flight: true,
            affinity: true,
            sample: 8,
        };
        assert_eq!(sent(&held[..3], tokens), [1, 2, 3]);
        assert_eq!(sent(&held[..3], text), [1, 2, 3]);
        assert!(sent(&held[..3], Needs::default()).is_empty());
        assert!(sent(&held, neither).is_empty());
        assert_eq!(
            sent(&held[..most], tokens),
            held[..most],
            "a list at the range, whole"
        );
        assert_eq!(
            sent(&held, tokens),
            held[..most],
            "the first of a longer list"
        );
    }

    /// The frame carries each candidate's requests in flight as the
    /// request's facts read them: the lent count where the host lends a
    /// table, the rows themselves reading zero.
    #[test]
    fn the_frame_gives_a_row_the_count_its_request_reads() {
        use crate::{
            counts::tests::Lent,
            plugins::config::Exposure,
            spec::Needs,
            view::{Candidate, PoolFacts, RequestFacts},
        };

        let mut rows: Vec<Candidate> = (0..)
            .zip([4, 0, 9])
            .map(|(position, in_flight)| Candidate {
                position,
                in_flight,
                ..Candidate::default()
            })
            .collect();
        let lent = Lent::take(&mut rows);
        let facts = lent.beside(&RequestFacts::default());
        let frame = super::decide(
            1,
            &facts,
            &rows,
            &PoolFacts::default(),
            &Exposure::default(),
            Needs::default(),
        );
        let read: Vec<u32> = frame.candidates.iter().map(|c| c.in_flight).collect();
        assert_eq!(read, [4, 0, 9]);
    }

    /// The view's trace is the contract's pair. A counted trace leaves with
    /// its count and no word of a draw; a drawn one leaves with the count
    /// unset, which is no zero, and `drew_first` set.
    #[test]
    fn the_views_trace_says_its_count_or_that_the_decision_drew() {
        use crate::decision::{Survivors, Trace};
        let counted = Trace {
            profile: 3,
            survivors: Survivors::counted(5),
            seed: 41,
            generation: 6,
            ..Trace::default()
        };
        let wire = super::trace(&counted);
        assert_eq!((wire.survivors, wire.drew_first), (Some(5), false));
        assert_eq!((wire.profile, wire.seed, wire.generation), (3, 41, 6));
        let drawn = Trace {
            survivors: Survivors::NOT_COUNTED,
            ..counted
        };
        let wire = super::trace(&drawn);
        assert_eq!((wire.survivors, wire.drew_first), (None, true));
    }

    /// The trace a plugin is shown carries the veto the all-vetoed rule
    /// lifted, and none for a pick no rule made.
    #[test]
    fn the_views_trace_carries_the_lifted_veto() {
        use crate::decision::{Lifted, Trace};
        let steered = Trace {
            lifted: Lifted::Overload,
            ..Trace::default()
        };
        assert_eq!(
            super::trace(&steered).lifted,
            proto::LiftedVeto::Overload as i32
        );
        assert_eq!(
            super::trace(&Trace::default()).lifted,
            proto::LiftedVeto::None as i32
        );
    }

    /// A measurement for the Numbers table, not a check: run in release
    /// with `--ignored measure --nocapture`. What one frame costs before
    /// the stream takes it, with a load row on every candidate: the message
    /// built from the view (the deciding thread's share), its encoding (the
    /// I/O thread's) and its size on the wire. The row declares the
    /// request's hashes, and the last two frames carry a prompt's: the
    /// view's default of 256 and the contract's most.
    #[test]
    #[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
    #[expect(
        clippy::print_stdout,
        reason = "the measurement reports its rows on stdout under --nocapture"
    )]
    fn measure_the_frame_built_and_encoded() {
        use std::{hint::black_box, time::Instant};

        use prost::Message;

        use crate::{
            plugins::config::Exposure,
            spec::{Needs, MAX_BLOCK_HASHES},
            view::{Candidate, LoadRow, PoolFacts, RequestFacts, WorkerId},
        };

        let facts = RequestFacts {
            model: "m",
            prompt_tokens: 512,
            budget_micros: 25_000,
            seed: 9,
            ..RequestFacts::default()
        };
        let exposure = Exposure::default();
        let needs = Needs {
            token_hashes: true,
            ..Needs::default()
        };
        let prompt: Vec<u64> = (1..=u64::from(MAX_BLOCK_HASHES)).collect();
        let frames = [
            (8u32, 0),
            (64, 0),
            (128, 0),
            (512, 0),
            (64, 256),
            (64, prompt.len()),
        ];
        for (n, hashes) in frames {
            let facts = RequestFacts {
                block_hashes: &prompt[..hashes],
                ..facts
            };
            let candidates: Vec<Candidate> = (0..n)
                .map(|i| Candidate {
                    position: i,
                    id: WorkerId(u64::from(i) + 100),
                    in_flight: n - i,
                    load: Some(LoadRow {
                        running: i,
                        queued: 1,
                        ..LoadRow::default()
                    }),
                    ..Candidate::default()
                })
                .collect();
            let pool = PoolFacts {
                pool_size: n,
                generation: 1,
                ..PoolFacts::default()
            };
            let rounds = 20_000u32;
            let started = Instant::now();
            for id in 0..rounds {
                black_box(super::decide(
                    u64::from(id),
                    &facts,
                    black_box(&candidates),
                    &pool,
                    &exposure,
                    needs,
                ));
            }
            let build = started.elapsed().as_nanos() / u128::from(rounds);
            let request = proto::PluginRequest {
                request: Some(proto::plugin_request::Request::Decide(super::decide(
                    1,
                    &facts,
                    &candidates,
                    &pool,
                    &exposure,
                    needs,
                ))),
            };
            let mut wire = Vec::with_capacity(request.encoded_len());
            let started = Instant::now();
            for _ in 0..rounds {
                wire.clear();
                black_box(&request)
                    .encode(&mut wire)
                    .expect("the buffer grows");
            }
            let encode = started.elapsed().as_nanos() / u128::from(rounds);
            println!(
                "a frame of {n} candidates and {hashes} block hashes: built in {build} ns, encoded in {encode} ns, {} bytes",
                wire.len()
            );
        }
    }
}

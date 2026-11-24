//! The `manual` policy: how it is composed, its session rule, and the picker
//! that leaves the placement to that rule. The session table is the engine's
//! ([`SessionTable`](crate::engine::SessionTable)).

use crate::{
    document::spec::{
        Assignment, ProfileSpec, Recovery, RulePlaced, SessionsSpec, MANUAL_SESSION_HEADER,
    },
    engine::{
        decision::Targets,
        framework::{Picker, TieBand},
        view::{Candidate, RequestFacts},
    },
    policies::standard_filters,
    stages::lifecycle::Lifecycle,
};

/// How the `manual` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    ProfileSpec {
        filter: standard_filters(),
        score: vec![],
        pick: "manual".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The session rule of the `manual` policy.
pub(crate) fn sessions() -> SessionsSpec {
    // Sticky sessions over a random assignment: a key stays with the
    // first worker it was placed on, no in-flight cap, the first
    // worker back in the lead when it recovers; the idle TTL of four
    // hours swept each minute, as the keyed policy of today. The
    // picker leaves the placement to the rule, so another assignment
    // is the rule's one field and no other profile. A request the rule
    // places is the profile's own: no stage counts under that picker.
    SessionsSpec {
        sticky_header: MANUAL_SESSION_HEADER.into(),
        headers: vec![],
        ttl_s: 14_400,
        sweep_s: 60,
        assignment: Assignment::Random,
        inflight_cap: u16::MAX,
        issue_session_id: false,
        recovery: Recovery::Original,
        rule_placed: RulePlaced::Told,
    }
}

/// The headers the `manual` policy adds to the view: the one its session rule
/// reads.
pub(crate) fn view_headers() -> Vec<String> {
    vec![MANUAL_SESSION_HEADER.into()]
}

/// The picker of a profile that leaves the placement to its binding's
/// sessions rule. The engine reads [`Picker::by_assignment`] once per profile
/// and places every request of such a profile by the rule's assignment, a
/// keyed request with no routable pin and a keyless one alike: a random
/// worker, the fewest requests in flight or the fewest session keys. With
/// no stages of its own to delegate to, `delegate` and a binding without a
/// rule read the fewest requests in flight. A side of a pair is placed the
/// same way. The picker itself ranks nothing: asked to pick, as a side's
/// shadow asks its profile, it declines.
#[derive(Debug, Default)]
pub struct Manual;

impl Lifecycle for Manual {}

impl Picker for Manual {
    fn name(&self) -> &'static str {
        "manual"
    }

    // Declines by the placement its request's rule names, never by a count.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[f32],
        _: TieBand,
        _: &mut Targets,
    ) {
    }

    fn by_assignment(&self) -> bool {
        true
    }
}

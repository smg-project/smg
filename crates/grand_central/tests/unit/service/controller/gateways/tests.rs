use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{sync::watch, task::JoinSet, time::timeout};

use super::{
    over_kinds, DropPause, Gateways, Refused, SessionPlace, Vetoed, KINDS_MAX, VETOES_MAX,
};
use crate::{
    document::spec::{AppliedSpec, Assignment, PolicySpec, Recovery, RulePlaced},
    engine::{
        sticky_sessions::{SessionRule, SessionVerdict},
        view::{Role, Session, WorkerId},
    },
    service::{
        controller::lifecycle::tests::worker,
        elector::{ElectionSettings, Elector, TickOutcome},
    },
    store::{
        GatewayId, HealthSink, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord,
        WorkersStore,
    },
    tracking::lease::{Lease, LeaseId},
};

impl SessionPlace {
    /// The session ended at `now`, where a case names the instant.
    pub(in crate::service::controller) fn end(self, now: Instant) {
        self.gateways
            .gone(self.gateway, self.session, &self.superseded, now);
    }
}

impl Gateways {
    /// The ceiling of rows.
    pub(in crate::service::controller) fn ceiling(&self) -> usize {
        self.ceiling
    }

    /// How many replaced sessions the row of `gateway` notes.
    fn noted(&self, gateway: GatewayId) -> usize {
        self.table
            .lock()
            .rows
            .get(&gateway)
            .map_or(0, |row| row.replaced.len())
    }

    /// How many rows the sweep has visited.
    fn take_sweep_visits(&self) -> usize {
        std::mem::take(&mut self.table.lock().sweep_visits)
    }

    /// How many ends the table keeps for the gateway: the row's count,
    /// which every write keeps in step with the note.
    fn kept(&self, gateway: GatewayId) -> usize {
        let table = self.table.lock();
        table.rows.get(&gateway).map_or(0, |row| {
            let noted: usize = row.vetoes.values().map(Vec::len).sum();
            assert_eq!(row.held, noted, "the count follows the note");
            let earliest = row.vetoes.values().flatten().map(|&(_, end)| end).min();
            assert!(
                earliest.is_none_or(|end| row.next_end.is_some_and(|next| next <= end)),
                "no end is earlier than the note says"
            );
            row.held
        })
    }
}

pub(crate) const GRACE: Duration = Duration::from_secs(10);
const A: GatewayId = GatewayId(11);
const B: GatewayId = GatewayId(12);
const C: GatewayId = GatewayId(13);
const D: GatewayId = GatewayId(14);
const MS: Duration = Duration::from_millis(1);
/// The term of the first take.
const TERM: u64 = 1;

/// Under these a fence taken once stays open for half an hour of a
/// case's clock: the case's instants decide what is over, not the
/// election's.
pub(crate) const PATIENT: ElectionSettings = ElectionSettings {
    duration: Duration::from_secs(3600),
    renew_period: Duration::from_secs(1),
    renew_deadline: Duration::from_secs(1800),
};

/// A candidate that leads under its own lease from the returned instant,
/// a store, and the table over that candidate's fence and that store.
pub(crate) fn leading(grace: Duration) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
    ceiled(grace, 1_024)
}

/// The same, with a table that holds `ceiling` rows.
fn ceiled(grace: Duration, ceiling: usize) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
    let mut elector =
        Elector::alone("controller-0@10.0.0.7:30100", PATIENT).expect("the settings are in order");
    let taken = Instant::now();
    elector.lead_at(taken);
    assert_eq!(elector.fence().leader_term(taken), Some(1));
    let store = Arc::new(MemoryStore::new());
    let gateways = Arc::new(Gateways::new(
        grace,
        ceiling,
        elector.fence(),
        Arc::clone(&store),
    ));
    (elector, store, gateways, taken)
}

/// A `Hello` the table takes.
fn welcomed(
    gateways: &Arc<Gateways>,
    gateway: GatewayId,
    subject: &str,
    term: u64,
    now: Instant,
) -> SessionPlace {
    gateways
        .welcome(gateway, subject, term, now)
        .expect("the table takes the Hello")
}

/// One lease of `gateway` as a decision records it.
pub(crate) fn lease(gateway: GatewayId, id: u64, now: Instant) -> LeaseRecord {
    LeaseRecord {
        lease: Lease {
            id: LeaseId(id),
            ..Lease::default()
        },
        gateway,
        request: id,
        target: WorkerId(7),
        picked: PickRecord::default(),
        decode: None,
        untold: false,
        term: 0,
        issued_at: now,
        dispatch_deadline: now + Duration::from_secs(5),
        state: LeaseState::ISSUED,
    }
}

#[test]
fn a_gateway_is_present_from_its_welcome_until_its_grace_period_ends() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    assert!(gateways.presence(A, t0).is_none(), "never welcomed");

    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert!(gateways.presence(A, t0).is_some());
    // A session that stands is present however long it stands.
    assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
    assert!(gateways.presence(B, t0).is_none(), "another gateway");

    // Its end starts the period; the clock alone ends the presence, at
    // the period's last instant, with no sweep passed.
    let ended = t0 + Duration::from_secs(60);
    seat.end(ended);
    assert!(gateways.presence(A, ended + GRACE - MS).is_some());
    assert!(gateways.presence(A, ended + GRACE).is_none());
}

#[test]
fn a_hello_inside_the_grace_period_changes_nothing() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

    // The session comes back a millisecond before the period's end.
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE - MS);
    assert_eq!(gateways.sweep(t0 + GRACE), 0);
    assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
    assert_eq!(store.live(A), 1, "the gap's lease stands");
    assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
    drop(seat);
}

#[test]
fn the_sweep_visits_only_rows_with_a_due_deadline() {
    const ROWS: u64 = 4_096;

    let (_elector, store, gateways, t0) = ceiled(GRACE, ROWS as usize);
    let gone = welcomed(&gateways, GatewayId(1), "gateway-1", TERM, t0);
    let standing: Vec<SessionPlace> = (2..=ROWS)
        .map(|id| welcomed(&gateways, GatewayId(id), &format!("gateway-{id}"), TERM, t0))
        .collect();

    assert_eq!(gateways.sweep(t0 + Duration::from_secs(1)), 0);
    assert_eq!(gateways.take_sweep_visits(), 0, "no row has work due");

    let noted = worker(&store, "http://noted:8000");
    assert_eq!(
        standing[0].veto(noted, 1, Some(t0 + 2 * GRACE), t0),
        Vetoed::Shown
    );
    gone.end(t0);
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
    assert_eq!(
        gateways.take_sweep_visits(),
        1,
        "only the expired row is visited"
    );

    assert_eq!(gateways.sweep(t0 + 2 * GRACE), 0);
    assert_eq!(gateways.take_sweep_visits(), 1, "the due note is visited");
    assert_eq!(gateways.kept(GatewayId(2)), 0);
}

#[tokio::test]
async fn past_the_grace_period_the_sweep_lets_the_gateway_go_and_no_other() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    store.issue(lease(A, 2, t0)).expect("a lease");
    store.issue(lease(B, 1, t0)).expect("a lease");
    let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let _stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
    let mut of_a = gateways.presence(A, t0).expect("present");
    let mut of_b = gateways.presence(B, t0).expect("present");
    gone.end(t0);

    assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
    assert_eq!(store.live(A), 2, "inside the period nothing is let go");
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
    assert_eq!(store.live(A), 0, "the gateway's leases are released");
    assert_eq!(store.live(B), 1, "another gateway's stand");
    assert_eq!(gateways.sweep(t0 + GRACE), 0, "let go once");

    // The handle a decide stream holds resolves at the end, and only
    // the ended gateway's.
    timeout(Duration::from_secs(20), of_a.ended())
        .await
        .expect("the ended presence resolves its handle");
    assert!(
        timeout(Duration::from_millis(50), of_b.ended())
            .await
            .is_err(),
        "a presence that stands resolves no handle"
    );
}

/// A frame of a replaced session that is past the seat's question may
/// still be written. The gateway is not let go while that session
/// runs.
#[test]
fn a_gateway_is_not_let_go_while_a_session_it_replaced_still_runs() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let second = welcomed(&gateways, A, "gateway-a", TERM, t0);
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    assert!(first.replaced() && second.replaced());
    assert_eq!(gateways.sweep(t0 + GRACE), 0, "two replaced sessions run");
    assert!(gateways.presence(A, t0 + GRACE).is_some());

    // The later of the two ends first: the earlier one holds the row.
    drop(second);
    assert_eq!(gateways.sweep(t0 + GRACE), 0, "a replaced session runs");
    drop(first);
    assert_eq!(gateways.sweep(t0 + GRACE), 1);

    // The other order, on a second gateway: the earlier of the two ends
    // first, and the later one holds the row alone.
    let earlier = welcomed(&gateways, B, "gateway-b", TERM, t0);
    let later = welcomed(&gateways, B, "gateway-b", TERM, t0);
    welcomed(&gateways, B, "gateway-b", TERM, t0).end(t0);
    drop(earlier);
    assert_eq!(
        gateways.sweep(t0 + GRACE),
        0,
        "the later replaced session runs"
    );
    drop(later);
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
}

#[test]
fn a_replaced_sessions_end_and_due_sweep_cleanup_atomically() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    let mut replaced = welcomed(&gateways, A, "gateway-a", TERM, t0);
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    assert!(replaced.replaced());

    let pause = Arc::new(DropPause {
        gone: std::sync::Barrier::new(2),
        swept: std::sync::Barrier::new(2),
    });
    replaced.drop_pause = Some(Arc::clone(&pause));
    let ending = std::thread::spawn(move || drop(replaced));
    pause.gone.wait();
    let let_go = gateways.sweep(t0 + GRACE);
    pause.swept.wait();
    ending.join().expect("the session ends");

    assert_eq!(let_go, 1, "the next due sweep lets the gateway go");
    assert_eq!(store.live(A), 0, "the gateway's lease is released");
}

/// A replaced session that never ends holds its gateway's row for one
/// period more and no longer: the gateway is let go two periods after
/// its last session ended, whatever still runs.
#[test]
fn a_replaced_session_that_never_ends_holds_its_gateway_one_period_more() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    let never_ends = welcomed(&gateways, A, "gateway-a", TERM, t0);
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    assert_eq!(gateways.sweep(t0 + 2 * GRACE - MS), 0, "one period more");
    assert!(gateways.presence(A, t0 + 2 * GRACE - MS).is_some());
    assert_eq!(gateways.sweep(t0 + 2 * GRACE), 1, "and no longer");
    assert!(gateways.presence(A, t0 + 2 * GRACE).is_none());
    drop(never_ends);
}

/// A row notes only the replaced sessions that still run: a gateway
/// that says Hello again and again grows no row.
#[test]
fn a_row_keeps_no_note_of_a_replaced_session_that_ended() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    let mut stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
    for _ in 0..8 {
        let newer = welcomed(&gateways, A, "gateway-a", TERM, t0);
        // The replaced session ends on the word.
        drop(stands);
        stands = newer;
    }
    assert_eq!(gateways.noted(A), 0, "no ended session remains");
    drop(stands);
}

/// A welcome and a letting-go of one gateway at one instant are
/// ordered: welcomed with what it held, or let go and welcomed with
/// nothing. Never welcomed and then let go.
#[tokio::test]
async fn a_hello_at_the_periods_end_meets_the_letting_go_in_one_order() {
    // The Hello first: the period was over, so the gateway is let go
    // before it is welcomed, and a sweep of that instant lets nothing
    // of the new presence go.
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    let mut earlier = gateways.presence(A, t0).expect("present");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
    assert_eq!(store.live(A), 0, "the earlier presence's lease is released");
    timeout(Duration::from_secs(20), earlier.ended())
        .await
        .expect("the earlier presence ended");
    store.issue(lease(A, 2, t0 + GRACE)).expect("a lease");
    assert_eq!(gateways.sweep(t0 + GRACE), 0);
    assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
    assert_eq!(store.live(A), 1, "the new presence keeps what it holds");
    drop(seat);

    // The sweep first: the gateway is let go, then welcomed with nothing.
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
    assert_eq!(store.live(A), 0);
    assert_eq!(gateways.sweep(t0 + GRACE), 0);
    assert!(gateways.presence(A, t0 + GRACE).is_some());
    drop(seat);
}

#[test]
fn a_closed_fence_lets_nothing_go_and_a_take_starts_every_period_again() {
    let (mut elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

    // The lease is given up: nobody leads here, and the period's end
    // lets nothing go.
    assert!(elector.release().expect("the release"));
    assert_eq!(gateways.sweep(t0 + GRACE), 0);
    assert_eq!(gateways.sweep(t0 + 3 * GRACE), 0);
    assert_eq!(store.live(A), 1, "a fenced replica releases nothing");

    // The take, three periods later: the gateway could say no Hello
    // here meanwhile, so its period starts at the take.
    let retaken = t0 + 3 * GRACE;
    assert_eq!(
        elector.tick(retaken).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(gateways.sweep(retaken), 0);
    assert!(gateways.presence(A, retaken + GRACE - MS).is_some());
    assert_eq!(gateways.sweep(retaken + GRACE - MS), 0);
    assert_eq!(store.live(A), 1);
    assert_eq!(gateways.sweep(retaken + GRACE), 1);
    assert_eq!(store.live(A), 0);
}

/// A renewal that lands late opens the fence again under the same term.
/// No `Hello` could be welcomed here while it was closed, so the first
/// read after it starts every gone gateway's period again, as a take
/// does.
#[test]
fn a_late_renewal_under_the_same_term_starts_every_gone_gateways_period_again() {
    let (mut elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

    // Past the renew deadline the fence is closed and the lease is
    // still this candidate's: a read finds nobody leading here.
    let closed = t0 + PATIENT.renew_deadline + MS;
    assert_eq!(gateways.sweep(closed), 0);
    assert_eq!(store.live(A), 1, "a fenced replica releases nothing");

    // The renewal lands late, under the same term: the gateway's
    // period starts at the first read after it.
    let landed = closed + GRACE;
    assert_eq!(
        elector.tick(landed).expect("the renewal"),
        TickOutcome::Renewed { term: TERM }
    );
    assert_eq!(gateways.sweep(landed), 0);
    assert!(gateways.presence(A, landed + GRACE - MS).is_some());
    assert_eq!(gateways.sweep(landed + GRACE - MS), 0);
    assert_eq!(store.live(A), 1);
    assert_eq!(gateways.sweep(landed + GRACE), 1);
    assert_eq!(store.live(A), 0);
}

/// The take's rule holds at a Hello too: a Hello that is the first read
/// after a take finds the period started again, and lets nothing go.
#[test]
fn a_hello_right_after_a_take_keeps_what_the_gateway_held() {
    let (mut elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
    assert!(elector.release().expect("the release"));
    assert_eq!(gateways.sweep(t0 + GRACE), 0);

    let retaken = t0 + 3 * GRACE;
    assert_eq!(
        elector.tick(retaken).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    let seat = welcomed(&gateways, A, "gateway-a", 2, retaken);
    assert_eq!(store.live(A), 1, "its lease stands through the gap");
    drop(seat);
}

#[tokio::test]
async fn a_second_session_supersedes_the_first_whose_end_starts_no_period() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert!(
        timeout(Duration::from_millis(50), first.superseded())
            .await
            .is_err(),
        "a lone session is not superseded"
    );
    let second = welcomed(&gateways, A, "gateway-a", TERM, t0 + MS);
    timeout(Duration::from_secs(20), first.superseded())
        .await
        .expect("the first session is told");
    assert!(
        timeout(Duration::from_millis(50), second.superseded())
            .await
            .is_err(),
        "the newer session stands"
    );

    // The superseded stream's end is not the gateway's.
    first.end(t0 + 2 * MS);
    assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
    assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
    // The standing one's is.
    second.end(t0 + 3 * MS);
    assert!(gateways.presence(A, t0 + 3 * MS + GRACE).is_none());
}

#[test]
fn a_seats_drop_is_its_sessions_end() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    drop(welcomed(&gateways, A, "gateway-a", TERM, t0));
    let dropped = Instant::now();
    assert!(gateways.presence(A, dropped).is_some());
    assert!(gateways.presence(A, dropped + GRACE).is_none());
}

/// A seat tells the grace period of the table it stands in, under the
/// stock period and over it.
#[test]
fn a_seat_tells_the_grace_period_of_its_own_table() {
    for grace in [Duration::from_millis(60), Duration::from_secs(3_600)] {
        let (_elector, _store, gateways, t0) = leading(grace);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.grace(), grace);
    }
}

/// A session's number is its alone for the table's life: the late end
/// of a session that an earlier presence replaced is no end of the
/// session that stands now. A replaced session reaches a later
/// presence only by outliving two periods.
#[test]
fn a_replaced_sessions_late_end_is_no_end_of_a_later_presences_session() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let second = welcomed(&gateways, A, "gateway-a", TERM, t0 + MS);
    second.end(t0 + 2 * MS);
    assert_eq!(gateways.sweep(t0 + 2 * MS + 2 * GRACE), 1);

    // A new presence, and only then the end of the replaced session.
    let third = welcomed(&gateways, A, "gateway-a", TERM, t0 + 2 * GRACE + 3 * MS);
    first.end(t0 + 2 * GRACE + 4 * MS);
    let later = t0 + 5 * GRACE;
    assert!(gateways.presence(A, later).is_some(), "its session stands");
    assert_eq!(gateways.sweep(later), 0);
    drop(third);
}

#[test]
fn the_session_counter_never_wraps_or_reuses_zero() {
    let (_elector, _store, gateways, t0) = leading(GRACE);
    gateways.table.lock().sessions = u64::MAX - 1;
    let stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(stands.session, u64::MAX);

    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gateways.welcome(A, "gateway-a", TERM, t0)
    }));
    assert!(attempted.is_ok(), "counter exhaustion must not panic");
    assert!(
        matches!(attempted.expect("checked above"), Err(Refused::Exhausted)),
        "an exhausted counter refuses the Hello"
    );
    assert!(!stands.replaced(), "the standing session is not superseded");
    let table = gateways.table.lock();
    let row = table.rows.get(&A).expect("the standing row");
    assert_eq!(table.sessions, u64::MAX);
    assert_eq!(row.session, u64::MAX);
    assert!(row.gone.is_none(), "the row's session still stands");
}

/// The seat is the table's own read of the fence: a replica that
/// stopped leading after the session's read seats nobody and releases
/// nothing.
#[test]
fn a_hello_whose_fence_closed_since_the_sessions_read_takes_no_seat_and_lets_nothing_go() {
    let (mut elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

    // The period is over with no sweep passed, and the lease is given
    // up between a session's read and the table's lock.
    assert!(elector.release().expect("the release"));
    assert_eq!(
        gateways.welcome(A, "gateway-a", TERM, t0 + GRACE).err(),
        Some(Refused::Stepped {
            read: TERM,
            shown: None
        })
    );
    assert_eq!(store.live(A), 1, "a fenced replica releases nothing");
}

/// A `Hello` read under a term the election has left takes no seat:
/// the session that stands under the later term keeps its own.
#[tokio::test]
async fn a_hello_read_under_an_earlier_term_supersedes_no_session_of_the_later_one() {
    let (mut elector, _store, gateways, t0) = leading(GRACE);
    // The replica gives its lease up and takes it again: term 2.
    assert!(elector.release().expect("the release"));
    let again = t0 + MS;
    assert_eq!(
        elector.tick(again).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    let stands = welcomed(&gateways, A, "gateway-a", 2, again);

    // A Hello its session read under term 1 reaches the table now.
    assert_eq!(
        gateways.welcome(A, "gateway-a", TERM, again).err(),
        Some(Refused::Stepped {
            read: TERM,
            shown: Some(2)
        })
    );
    assert!(
        timeout(Duration::from_millis(50), stands.superseded())
            .await
            .is_err(),
        "the standing session keeps its seat"
    );
    assert!(gateways.presence(A, again).is_some());
    drop(stands);
}

/// The table's bound that is not time: a `Hello` that would make a row
/// past the ceiling is refused at the door, and a present gateway's is
/// always taken.
#[test]
fn a_hello_past_the_ceiling_is_refused_and_a_present_gateways_is_taken() {
    let (_elector, _store, gateways, t0) = ceiled(GRACE, 3);
    let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let second = welcomed(&gateways, B, "gateway-b", TERM, t0);
    let third = welcomed(&gateways, C, "gateway-c", TERM, t0);
    // The fourth would be a new row: refused, and no row is made.
    assert_eq!(
        gateways.welcome(D, "gateway-d", TERM, t0).err(),
        Some(Refused::Full { ceiling: 3 })
    );
    assert!(gateways.presence(D, t0).is_none());
    // A present gateway says Hello again: no new row, so it is taken.
    let again = welcomed(&gateways, A, "gateway-a", TERM, t0);

    // Inside its grace period a gateway's row still counts.
    third.end(t0);
    assert_eq!(
        gateways
            .welcome(D, "gateway-d", TERM, t0 + GRACE - MS)
            .err(),
        Some(Refused::Full { ceiling: 3 })
    );
    // Once the sweep let one go, the fourth is welcomed.
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
    let fourth = welcomed(&gateways, D, "gateway-d", TERM, t0 + GRACE);
    drop((first, second, again, fourth));
}

/// One key, one subject: a `Hello` of another subject under a present
/// gateway's key is refused, and nothing of that gateway is touched.
/// The table is handed the key and the subject apart, so the case names
/// one key under two subjects.
#[tokio::test]
async fn a_hello_of_another_subject_under_a_present_gateways_key_is_refused() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    store.issue(lease(A, 1, t0)).expect("a lease");
    let stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(
        gateways.welcome(A, "gateway-b", TERM, t0).err(),
        Some(Refused::AnotherSubject)
    );
    // The standing session is not replaced, and what it held stands.
    assert!(
        timeout(Duration::from_millis(50), stands.superseded())
            .await
            .is_err(),
        "the present gateway's session stands"
    );
    assert_eq!(store.live(A), 1);

    // Inside the grace period the key is still the first subject's.
    stands.end(t0);
    assert_eq!(
        gateways
            .welcome(A, "gateway-b", TERM, t0 + GRACE - MS)
            .err(),
        Some(Refused::AnotherSubject)
    );
    // Once the first is let go the key is free, and the other subject
    // starts with nothing of it.
    let other = welcomed(&gateways, A, "gateway-b", TERM, t0 + GRACE);
    assert_eq!(store.live(A), 0, "the first gateway's lease went with it");
    drop(other);
}

/// The task over the real clock: a gone gateway is let go a grace
/// period after its session's end, and the task ends at its signal.
#[tokio::test]
async fn the_sweeps_task_lets_a_gone_gateway_go_and_ends_at_its_stop() {
    let grace = Duration::from_millis(60);
    let (_elector, store, gateways, t0) = leading(grace);
    store.issue(lease(A, 1, t0)).expect("a lease");
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    let sweeper = Arc::clone(&gateways);
    task.spawn(async move { sweeper.run(stopped).await });

    let ended = Instant::now();
    drop(welcomed(&gateways, A, "gateway-a", TERM, ended));
    let started = Instant::now();
    while store.live(A) > 0 {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the sweep let nothing go in 20 s"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(
        ended.elapsed() >= grace,
        "let go {:?} after the end, inside the period of {grace:?}",
        ended.elapsed()
    );

    stop.send_replace(true);
    timeout(Duration::from_secs(20), task.join_next())
        .await
        .expect("the task ends within 20 s of its stop")
        .expect("one task")
        .expect("it ended by itself");
}

/// The leases' due ends ride the sweeps' task: a lease past its
/// dispatch deadline leaves by the task's own tick where no decider
/// issues, its gateway's session standing, and a lease inside its
/// deadline stays.
#[tokio::test]
async fn the_sweeps_task_ends_a_lease_past_its_dispatch_deadline() {
    let (_elector, store, gateways, t0) = leading(Duration::from_millis(60));
    let due = LeaseRecord {
        dispatch_deadline: t0,
        ..lease(A, 1, t0)
    };
    store.issue(due).expect("a lease");
    store.issue(lease(A, 2, t0)).expect("a lease");
    let _stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    let sweeper = Arc::clone(&gateways);
    task.spawn(async move { sweeper.run(stopped).await });

    let started = Instant::now();
    while store.live(A) > 1 {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the sweep ended no due lease in 20 s"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(LeasesStore::get(&*store, A, LeaseId(1)).is_none());
    assert!(LeasesStore::get(&*store, A, LeaseId(2)).is_some());

    stop.send_replace(true);
    timeout(Duration::from_secs(20), task.join_next())
        .await
        .expect("the task ends within 20 s of its stop")
        .expect("one task")
        .expect("it ended by itself");
}

/// The names that left ride the sweeps' task: one that stays away past
/// the absence's time takes its session pins with it by the task's own
/// tick, and no sooner, which no other caller of the store's tick
/// stands in for. The engine's upkeep rides the same tick and takes
/// this key out by nothing of its own. The key is read behind the
/// leave, so the engine's table filed no absence, and under a rule that
/// never ends it.
#[tokio::test]
async fn the_sweeps_task_ends_the_absence_of_a_key_that_stays_away() {
    let (_elector, store, gateways, _t0) = leading(Duration::from_millis(60));
    // The absence's time is the longest TTL of the document's session
    // rules: one second.
    let mut document = PolicySpec::preset("manual").expect("a preset of this build");
    document.generation = 1;
    let rule = document.bindings[0].sessions.as_mut();
    let rule = rule.expect("the preset's rule");
    (rule.ttl_s, rule.sweep_s) = (1, 1);
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(spec).expect("a first generation");
    let w = worker(&store, "http://a:8000");
    let key = Session::from_raw(7);
    let held = || {
        let world = store.snapshot();
        let table = world.spec().engine().sessions();
        table.pins(key).map(|pins| pins[0])
    };
    let left = Instant::now();
    let revision = WorkersStore::get(&*store, w).expect("the row").revision;
    store.remove(w, revision).expect("removed");
    let never = SessionRule {
        assignment: Assignment::default(),
        inflight_cap: 0,
        ttl_micros: None,
        sweep_micros: 0,
        recovery: Recovery::default(),
        rule_placed: RulePlaced::default(),
    };
    let world = store.snapshot();
    let table = world.spec().engine().sessions();
    table.lookup(key, store.micros(left), &never);
    table.dispatched(key, SessionVerdict::Pinned, w, 0);
    assert_eq!(held(), Some(w), "inside its time");
    let (stop, stopped) = watch::channel(false);
    let mut task = JoinSet::new();
    let sweeper = Arc::clone(&gateways);
    task.spawn(async move { sweeper.run(stopped).await });
    while held().is_some() {
        assert!(
            left.elapsed() < Duration::from_secs(20),
            "the task ended no absence in 20 s"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(
        left.elapsed() >= Duration::from_secs(1),
        "the pin left inside the absence's time"
    );
    stop.send_replace(true);
    timeout(Duration::from_secs(20), task.join_next())
        .await
        .expect("the task ends within 20 s of its stop")
        .expect("one task")
        .expect("it ended by itself");
}

const BREAKER: i32 = 1;
const LIVENESS: i32 = 2;
const S: Duration = Duration::from_secs(1);

#[test]
fn the_store_shows_the_latest_end_of_a_workers_kinds_and_a_clear_lifts_its_kind_alone() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
    // The later end is shown, to its gateway alone.
    assert!(store.vetoed(A, w, t0 + 9 * S - MS));
    assert!(!store.vetoed(A, w, t0 + 9 * S));
    assert!(!store.vetoed(B, w, t0));

    // The later kind cleared: the earlier end is shown again.
    assert_eq!(seat.veto(w, LIVENESS, None, t0 + S), Vetoed::Shown);
    assert!(store.vetoed(A, w, t0 + 5 * S - MS));
    assert!(!store.vetoed(A, w, t0 + 5 * S));
    // A kind set again moves its own end.
    assert_eq!(
        seat.veto(w, BREAKER, Some(t0 + 3 * S), t0 + S),
        Vetoed::Shown
    );
    assert!(!store.vetoed(A, w, t0 + 3 * S));
    // The last kind cleared lifts the veto, and nothing is kept.
    assert_eq!(seat.veto(w, BREAKER, None, t0 + 2 * S), Vetoed::Shown);
    assert!(!store.vetoed(A, w, t0 + 2 * S));
    assert_eq!(gateways.kept(A), 0);
}

/// An end at the frame's own instant is no veto: the frame lifts its
/// kind as a clear does, and the worker is not read as one with no row.
#[test]
fn an_end_at_the_frames_instant_lifts_its_kind_as_a_clear_does() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + S), t0 + S), Vetoed::Shown);
    assert!(!store.vetoed(A, w, t0 + S));
    assert_eq!(gateways.kept(A), 0);
}

/// A kind is kept by its number: one this build has no name for stands
/// and clears like the named ones.
#[test]
fn a_kind_this_build_has_no_name_for_stands_and_clears_like_the_others() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(seat.veto(w, 7, Some(t0 + 9 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, BREAKER, None, t0), Vetoed::Shown);
    assert!(
        store.vetoed(A, w, t0 + 9 * S - MS),
        "the unnamed kind stands"
    );
    assert_eq!(seat.veto(w, 7, None, t0), Vetoed::Shown);
    assert!(!store.vetoed(A, w, t0));
}

/// What the published world's veto reader hands `gateway` of the worker
/// at `now`: whether a hold stands on it, and whether a veto does.
fn bits(store: &MemoryStore, gateway: GatewayId, worker: WorkerId, now: Instant) -> (bool, bool) {
    // A registration is in force at its window's publication: the case
    // asks for one.
    store.publish_now();
    let world = store.snapshot();
    assert!(world.booked_of(worker).is_some(), "the worker is published");
    let vetoes = world.gateway_vetoes("m", Role::Single, gateway);
    let veto = vetoes.standing_at(now).find(|veto| veto.id == worker);
    veto.map_or((false, false), |veto| (veto.held, true))
}

/// The store reads no kind: the note says which ends hold. Every kind
/// but the doubt of liveness holds, a number this build has no name for
/// among them, and the hold shown is the latest of their ends. A worker
/// reads to its gateway as masked to its veto's end, and as behind an
/// open breaker too while a hold stands.
#[test]
fn the_store_is_shown_the_hold_among_a_workers_kinds() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let (held, doubted, clear) = ((true, true), (false, true), (false, false));
    // A doubt of liveness alone holds nothing.
    assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
    assert_eq!(bits(&store, A, w, t0), doubted);
    // A breaker's window beside it holds to its own end, and the doubt
    // stands from there.
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
    assert_eq!(bits(&store, A, w, t0 + 5 * S - MS), held);
    assert_eq!(bits(&store, A, w, t0 + 5 * S), doubted);
    assert_eq!(bits(&store, A, w, t0 + 9 * S), clear);
    assert_eq!(bits(&store, B, w, t0), clear);
    // The doubt cleared: the breaker's window holds alone.
    assert_eq!(seat.veto(w, LIVENESS, None, t0), Vetoed::Shown);
    assert_eq!(bits(&store, A, w, t0), held);
    assert_eq!(bits(&store, A, w, t0 + 5 * S), clear);
    // The breaker closed beside a new doubt: nothing holds.
    assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, BREAKER, None, t0), Vetoed::Shown);
    assert_eq!(bits(&store, A, w, t0), doubted);
    // A kind with no name holds: the rule lifts what it can name.
    assert_eq!(seat.veto(w, 7, Some(t0 + 3 * S), t0), Vetoed::Shown);
    assert_eq!(bits(&store, A, w, t0), held);
    assert_eq!(bits(&store, A, w, t0 + 3 * S), doubted);
}

/// A `Hello`'s list is shown the same way: each listed worker's latest
/// end, with the hold among its kinds.
#[test]
fn a_hellos_list_is_shown_with_the_hold_among_each_workers_kinds() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let (held, doubted) = (
        worker(&store, "http://a:8000"),
        worker(&store, "http://b:8000"),
    );
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let list = [
        (held, BREAKER, t0 + 5 * S),
        (held, LIVENESS, t0 + 9 * S),
        (doubted, LIVENESS, t0 + 9 * S),
    ];
    assert_eq!(seat.restore(&list, t0), 3);
    assert_eq!(bits(&store, A, held, t0), (true, true));
    assert_eq!(bits(&store, A, held, t0 + 5 * S), (false, true));
    assert_eq!(bits(&store, A, doubted, t0), (false, true));
    assert_eq!(bits(&store, A, doubted, t0 + 9 * S), (false, false));
}

#[test]
fn a_veto_on_a_worker_with_no_row_is_dropped_and_nothing_of_it_is_kept() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let unheld = WorkerId(77);
    assert_eq!(
        seat.veto(unheld, BREAKER, Some(t0 + 5 * S), t0),
        Vetoed::NoWorker
    );
    assert!(!store.vetoed(A, unheld, t0));
    assert_eq!(gateways.kept(A), 0);
    // Its clear finds nothing to lift.
    assert_eq!(seat.veto(unheld, BREAKER, None, t0), Vetoed::Shown);
}

/// A veto is about a worker's row and leaves with it: a worker removed
/// and registered again starts clean, whatever the table noted.
#[test]
fn a_worker_registered_again_starts_clean_and_an_end_past_its_instant_is_not_kept() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
    let revision = WorkersStore::get(&*store, w).expect("the row").revision;
    store.remove(w, revision).expect("removed");
    assert_eq!(worker(&store, "http://a:8000"), w, "the same id");
    assert!(!store.vetoed(A, w, t0), "the veto left with the row");

    // The other kind's frame shows its own end, not the removed row's.
    assert_eq!(
        seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0 + S),
        Vetoed::Shown
    );
    assert!(store.vetoed(A, w, t0 + 5 * S - MS));
    assert!(!store.vetoed(A, w, t0 + 5 * S));
    assert_eq!(gateways.kept(A), 1);

    // Past their instants the ends are no vetoes: after a pass the
    // table keeps none, however many workers came and went.
    for n in 0..40 {
        let w = worker(&store, &format!("http://w{n}:8000"));
        assert_eq!(
            seat.veto(w, BREAKER, Some(t0 + 5 * S), t0 + S),
            Vetoed::Shown
        );
    }
    assert_eq!(gateways.kept(A), 41);
    assert_eq!(gateways.sweep(t0 + 5 * S), 0);
    assert_eq!(gateways.kept(A), 0);
}

#[test]
fn the_letting_go_lifts_every_veto_of_the_gateway_and_no_other_gateways() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
    let until = t0 + 100 * GRACE;
    assert_eq!(gone.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
    assert_eq!(stands.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
    gone.end(t0);

    // Inside the period the veto stands, as the leases do.
    assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
    assert!(store.vetoed(A, w, t0 + GRACE - MS));
    assert_eq!(gateways.sweep(t0 + GRACE), 1);
    assert!(!store.vetoed(A, w, t0 + GRACE));
    assert!(store.vetoed(B, w, t0 + GRACE), "another gateway's stands");
    drop(stands);
}

/// One pass lets every gateway whose period is over go in one write of
/// the store: their vetoes leave in one publication, and a present
/// gateway's stands. A pass that lets none go publishes nothing.
#[test]
fn a_pass_lets_its_gateways_go_in_one_publication_of_the_store() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let until = t0 + 100 * GRACE;
    let stands = welcomed(&gateways, C, "gateway-c", TERM, t0);
    assert_eq!(stands.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
    for (gateway, subject) in [(A, "gateway-a"), (B, "gateway-b")] {
        let gone = welcomed(&gateways, gateway, subject, TERM, t0);
        assert_eq!(gone.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
        gone.end(t0);
    }

    let published = store.publications();
    assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
    assert_eq!(store.publications(), published);
    assert_eq!(gateways.sweep(t0 + GRACE), 2);
    assert_eq!(
        store.publications(),
        published + 1,
        "one world for the pass"
    );
    assert!(!store.vetoed(A, w, t0 + GRACE) && !store.vetoed(B, w, t0 + GRACE));
    assert!(store.vetoed(C, w, t0 + GRACE), "a present gateway's stands");
    drop(stands);
}

#[test]
fn a_replaced_sessions_veto_writes_nothing() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let second = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(
        first.veto(w, BREAKER, Some(t0 + 5 * S), t0),
        Vetoed::Replaced
    );
    assert!(!store.vetoed(A, w, t0));
    assert_eq!(second.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
    // It lifts nothing either, and its list restores nothing.
    assert_eq!(first.veto(w, BREAKER, None, t0), Vetoed::Replaced);
    assert_eq!(first.restore(&[], t0), 0);
    assert!(store.vetoed(A, w, t0));
    drop((first, second));
}

/// A `Hello`'s list is the gateway's vetoes, whole: what it omits is
/// lifted, and of two entries of one worker and kind the later end
/// stands, in either order.
#[test]
fn a_hellos_list_is_the_gateways_vetoes_whole_and_the_later_end_of_a_pair_stands() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let listed = worker(&store, "http://a:8000");
    let omitted = worker(&store, "http://b:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(
        seat.veto(listed, BREAKER, Some(t0 + 60 * S), t0),
        Vetoed::Shown
    );
    assert_eq!(
        seat.veto(omitted, BREAKER, Some(t0 + 60 * S), t0),
        Vetoed::Shown
    );
    seat.end(t0);

    // Back inside the period. One veto is listed with a nearer end, a
    // second kind twice, one worker is not held, one end is past.
    let now = t0 + S;
    let seat = welcomed(&gateways, A, "gateway-a", TERM, now);
    for pair in [[6, 7], [7, 6]] {
        let list = [
            (listed, BREAKER, now + 3 * S),
            (listed, LIVENESS, now + pair[0] * S),
            (listed, LIVENESS, now + pair[1] * S),
            (WorkerId(77), BREAKER, now + 9 * S),
            (omitted, LIVENESS, now),
        ];
        assert_eq!(seat.restore(&list, now), 2, "two ends of one worker");
        assert!(store.vetoed(A, listed, now + 7 * S - MS));
        assert!(!store.vetoed(A, listed, now + 7 * S), "the later end");
        assert!(!store.vetoed(A, omitted, now), "what the list omits");
        assert_eq!(gateways.kept(A), 2);
    }
    // An empty list lifts everything.
    assert_eq!(seat.restore(&[], now), 0);
    assert!(!store.vetoed(A, listed, now));
    assert_eq!(gateways.kept(A), 0);
    drop(seat);
}

/// A worker holds [`KINDS_MAX`] kinds for a gateway and no more: the
/// veto that would set one more writes nothing, and a kind that stands
/// or a clear is always taken.
#[test]
fn a_worker_holds_its_ceiling_of_kinds_and_one_more_writes_nothing() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    for kind in 1..=kinds {
        assert_eq!(seat.veto(w, kind, Some(t0 + 5 * S), t0), Vetoed::Shown);
    }
    assert_eq!(gateways.kept(A), KINDS_MAX);

    // One more kind: nothing is written, and what stands stays.
    assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 9 * S), t0), Vetoed::Full);
    assert_eq!(gateways.kept(A), KINDS_MAX);
    assert!(store.vetoed(A, w, t0 + 5 * S - MS));
    assert!(
        !store.vetoed(A, w, t0 + 5 * S),
        "the refused end is not shown"
    );

    // A kind that stands moves its own end; a clear of a kind that
    // does not stand is taken; a clear makes room for the new kind.
    assert_eq!(seat.veto(w, 1, Some(t0 + 6 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, kinds + 1, None, t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, 2, None, t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 9 * S), t0), Vetoed::Shown);
    assert!(store.vetoed(A, w, t0 + 9 * S - MS));
    assert_eq!(gateways.kept(A), KINDS_MAX);
    drop(seat);
}

/// A ceiling stands in the way only while the note's ends stand: a
/// worker removed and registered again holds none, and the kind that
/// was one too many is taken.
#[test]
fn a_workers_ceiling_of_kinds_leaves_with_its_row() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    for kind in 1..=kinds {
        assert_eq!(seat.veto(w, kind, Some(t0 + 60 * S), t0), Vetoed::Shown);
    }
    assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0), Vetoed::Full);
    let revision = WorkersStore::get(&*store, w).expect("the row").revision;
    store.remove(w, revision).expect("removed");
    assert_eq!(worker(&store, "http://a:8000"), w, "the same id");
    assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 5 * S), t0), Vetoed::Shown);
    assert!(
        !store.vetoed(A, w, t0 + 5 * S),
        "no end of the row before it"
    );
    assert_eq!(gateways.kept(A), 1);
    drop(seat);
}

/// A gateway holds [`VETOES_MAX`] vetoes and no more, what its `Hello`
/// lists whole: the frame that would set one more writes nothing. A
/// kind that stands and a clear are always taken, and an end past its
/// instant leaves the count at the frame that needs its place, with no
/// pass of the sweep between.
#[test]
fn a_gateway_holds_its_ceiling_of_vetoes_and_one_more_writes_nothing() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    let workers: Vec<WorkerId> = (0..VETOES_MAX / KINDS_MAX)
        .map(|n| worker(&store, &format!("http://w{n}:8000")))
        .collect();
    // The first worker's ends pass at five seconds, the others' at sixty.
    let list: Vec<(WorkerId, i32, Instant)> = workers
        .iter()
        .enumerate()
        .flat_map(|(n, &w)| {
            let end = t0 + if n == 0 { 5 * S } else { 60 * S };
            (1..=kinds).map(move |kind| (w, kind, end))
        })
        .collect();
    assert_eq!(seat.restore(&list, t0), VETOES_MAX);
    assert_eq!(gateways.kept(A), VETOES_MAX);

    // One veto more, on a worker that holds none: nothing is written.
    let one_more = worker(&store, "http://one-more:8000");
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
        Vetoed::Over
    );
    assert!(!store.vetoed(A, one_more, t0 + S));
    assert_eq!(gateways.kept(A), VETOES_MAX);
    // A kind that stands moves its own end, and a clear is taken.
    assert_eq!(
        seat.veto(workers[1], 1, Some(t0 + 61 * S), t0 + S),
        Vetoed::Shown
    );
    assert_eq!(seat.veto(one_more, BREAKER, None, t0 + S), Vetoed::Shown);
    assert_eq!(gateways.kept(A), VETOES_MAX);

    // Past the first worker's instant its eight ends are no vetoes:
    // the frame takes a place they held.
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + 5 * S),
        Vetoed::Shown
    );
    assert!(store.vetoed(A, one_more, t0 + 5 * S));
    assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
    drop(seat);
}

/// A list that fills a note to its ceiling: [`KINDS_MAX`] kinds on each
/// of `workers`, every end at sixty seconds.
fn full_list(workers: &[WorkerId], t0: Instant) -> Vec<(WorkerId, i32, Instant)> {
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    workers
        .iter()
        .flat_map(|&w| (1..=kinds).map(move |kind| (w, kind, t0 + 60 * S)))
        .collect()
}

/// Fills the seat's note to its ceiling of [`VETOES_MAX`] ends on newly
/// registered workers ([`full_list`]). Returns the workers.
fn filled(store: &MemoryStore, seat: &SessionPlace, t0: Instant) -> Vec<WorkerId> {
    let workers: Vec<WorkerId> = (0..VETOES_MAX / KINDS_MAX)
        .map(|n| worker(store, &format!("http://w{n}:8000")))
        .collect();
    assert_eq!(seat.restore(&full_list(&workers, t0), t0), VETOES_MAX);
    workers
}

/// A gateway's ceiling counts the vetoes the store shows: the ends of
/// workers whose rows left go before the refusal, and the frame that
/// would set one more is taken.
#[test]
fn a_gateways_ceiling_counts_no_end_of_a_row_that_left() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let workers = filled(&store, &seat, t0);
    // Every row but the first leaves, and its vetoes with it: the note
    // still holds their ends.
    for &w in &workers[1..] {
        let revision = WorkersStore::get(&*store, w).expect("the row").revision;
        store.remove(w, revision).expect("removed");
    }
    assert_eq!(gateways.kept(A), VETOES_MAX);

    // One veto more, on a worker that holds none: the ends of the rows
    // that left go, and the veto is shown.
    let one_more = worker(&store, "http://one-more:8000");
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
        Vetoed::Shown
    );
    assert!(store.vetoed(A, one_more, t0 + S));
    assert_eq!(gateways.kept(A), KINDS_MAX + 1);
    drop(seat);
}

/// A refusal at a true ceiling asks the store of the note's rows once
/// a membership epoch. The table alone writes a gateway's vetoes and
/// a row leaves by a membership write alone: a veto the store lost by
/// another hand is not found under the same epoch, and the next
/// membership write has the store asked again.
#[test]
fn a_refusal_at_a_true_ceiling_asks_the_store_once_a_membership_epoch() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let workers = filled(&store, &seat, t0);
    let one_more = worker(&store, "http://one-more:8000");
    // Every row stands: the frame is refused, and the store was asked.
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
        Vetoed::Over
    );
    // The store loses one worker's veto behind the note, by no
    // membership write: under the same epoch the store is not asked
    // again, and the frame is refused as before.
    HealthSink::gateway_veto(&*store, A, workers[1], None);
    assert!(!store.vetoed(A, workers[1], t0 + S));
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
        Vetoed::Over,
        "the store was asked twice under one membership epoch"
    );
    // A membership write moves the epoch: the next frame asks the
    // store, that worker's ends go and the veto is shown.
    let _late = worker(&store, "http://late:8000");
    assert_eq!(
        seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
        Vetoed::Shown
    );
    assert!(store.vetoed(A, one_more, t0 + S));
    assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
    drop(seat);
}

/// A noted worker removed after a refusal kept the epoch: the store's
/// removal is a membership write, so the next frame at the ceiling
/// asks the store again, the removed worker's ends go and the veto is
/// shown.
#[test]
fn a_noted_worker_removed_under_a_kept_epoch_has_the_store_asked_again() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let workers = filled(&store, &seat, t0);
    let one_more = worker(&store, "http://one-more:8000");
    let frame = || seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S);
    // Refused twice, the second time under the epoch the first kept.
    assert_eq!((frame(), frame()), (Vetoed::Over, Vetoed::Over));
    let revision = WorkersStore::get(&*store, workers[1])
        .expect("the row")
        .revision;
    store.remove(workers[1], revision).expect("removed");
    assert_eq!(frame(), Vetoed::Shown);
    assert!(store.vetoed(A, one_more, t0 + S));
    assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
    drop(seat);
}

/// A `Hello`'s list replaces the note, and the epoch kept for the old
/// note's walk goes with it: the first frame at the ceiling after the
/// list asks the store, under the same membership epoch.
#[test]
fn a_list_that_replaces_the_note_drops_the_epoch_its_walk_kept() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let workers = filled(&store, &seat, t0);
    let one_more = worker(&store, "http://one-more:8000");
    let frame = || seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S);
    // The refusal walks the note and keeps the epoch.
    assert_eq!(frame(), Vetoed::Over);
    // The gateway's list is laid again, whole, with no membership
    // write between.
    assert_eq!(seat.restore(&full_list(&workers, t0), t0), VETOES_MAX);
    // The store loses one worker's veto behind the new note, by no
    // membership write: the frame asks the store all the same, that
    // worker's ends go and the veto is shown.
    HealthSink::gateway_veto(&*store, A, workers[1], None);
    assert_eq!(frame(), Vetoed::Shown);
    assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
    drop(seat);
}

/// A frame's end nearer than every end the note holds is the note's
/// earliest from then on, and a walk keeps the earliest end left: each
/// such end frees its place at the ceiling once past, with no pass of
/// the sweep between, and the ceiling stands until then.
#[test]
fn an_end_a_frame_sets_nearest_frees_its_place_at_the_ceiling_once_past() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let workers = filled(&store, &seat, t0);
    // Two frames move one kind's end each, to five and to seven
    // seconds: the list's ends stand at sixty.
    assert_eq!(
        seat.veto(workers[0], 1, Some(t0 + 5 * S), t0),
        Vetoed::Shown
    );
    assert_eq!(
        seat.veto(workers[1], 1, Some(t0 + 7 * S), t0),
        Vetoed::Shown
    );
    assert_eq!(gateways.kept(A), VETOES_MAX);
    let first = worker(&store, "http://one-more:8000");
    let second = worker(&store, "http://two-more:8000");
    let frame = |w, at| seat.veto(w, BREAKER, Some(t0 + 60 * S), t0 + at);
    // Before the nearest end the ceiling stands; at it one place is
    // free.
    assert_eq!(frame(first, S), Vetoed::Over);
    assert_eq!(frame(first, 5 * S), Vetoed::Shown);
    // The walk kept the earliest end left: the ceiling stands again
    // until that one passes.
    assert_eq!(frame(second, 6 * S), Vetoed::Over);
    assert_eq!(frame(second, 7 * S), Vetoed::Shown);
    assert_eq!(gateways.kept(A), VETOES_MAX);
    drop(seat);
}

/// The store is shown the latest end that stands: a kind set with an
/// earlier end under a later end of another kind leaves the later one
/// shown.
#[test]
fn a_kinds_earlier_end_under_a_later_one_leaves_the_later_end_shown() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0), Vetoed::Shown);
    // Past the earlier end the later one still stands in the store.
    assert!(store.vetoed(A, w, t0 + 30 * S));
    assert!(!store.vetoed(A, w, t0 + 60 * S));
    drop(seat);
}

/// A third kind's frame leaves the latest of the other kinds' ends
/// shown, not the earliest of them.
#[test]
fn a_third_kinds_frame_leaves_the_latest_of_the_other_ends_shown() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let third = BREAKER.max(LIVENESS) + 1;
    assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
    assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0), Vetoed::Shown);
    // The third kind's own end is the earliest of the three.
    assert_eq!(seat.veto(w, third, Some(t0 + S), t0), Vetoed::Shown);
    assert!(store.vetoed(A, w, t0 + 30 * S));
    assert!(!store.vetoed(A, w, t0 + 60 * S));
    drop(seat);
}

/// At a frame, with no pass of the sweep between, an end past its
/// instant frees its place among a worker's kinds: one kind more is
/// taken where the ceiling stood.
#[test]
fn an_end_past_its_instant_frees_its_place_at_a_frame() {
    let (_elector, store, gateways, t0) = leading(GRACE);
    let w = worker(&store, "http://a:8000");
    let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    // The first kind ends a second on, the others a minute on.
    assert_eq!(seat.veto(w, 1, Some(t0 + S), t0), Vetoed::Shown);
    for kind in 2..=kinds {
        assert_eq!(seat.veto(w, kind, Some(t0 + 60 * S), t0), Vetoed::Shown);
    }
    assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0), Vetoed::Full);
    // At the first kind's instant its place is free.
    assert_eq!(
        seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0 + S),
        Vetoed::Shown,
        "an end past its instant held its place"
    );
    assert_eq!(gateways.kept(A), KINDS_MAX);
    drop(seat);
}

#[test]
fn a_list_holds_a_workers_ceiling_of_kinds_and_no_more() {
    let kinds = i32::try_from(KINDS_MAX).expect("a small number");
    let w = WorkerId(1);
    let list = |last: i32| (1..=last).map(move |kind| (w, kind));
    assert!(!over_kinds(list(kinds)));
    assert!(over_kinds(list(kinds + 1)));
    // A kind listed twice is one kind, and each worker has its count.
    assert!(!over_kinds(list(kinds).chain(list(kinds))));
    let other = (1..=kinds).map(|kind| (WorkerId(2), kind));
    assert!(!over_kinds(list(kinds).chain(other)));
}

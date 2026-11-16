//! Who is present (design 4.3, the stream rule): a gateway is present from
//! its `Welcome` until its grace period ends with no new `Hello`, and what
//! the store holds under its key lives exactly that long: its lease
//! records and its completions.
//!
//! Presence is read from the clock at the instant of the question, as the
//! fence is: a gateway is present while a session of it stands and for the
//! grace period after the last one ended. No task tells a reader who is
//! present. The sweep only lets go what a gateway that ended held, so a
//! late sweep changes no answer.
//!
//! Three rules come from the fence (design 5.3). A replica whose fence is
//! closed lets nothing go: a fenced leader releases nothing. At a take
//! every gone gateway's period starts again: no `Hello` could be welcomed
//! here while this replica did not lead, so that time is not the gateway's
//! silence. And a `Hello` is seated under the term the table's own read of
//! the fence shows, under its lock, or not at all: one read under a term
//! the election has left supersedes no session of the term that stands.
//!
//! A `Welcome` and the letting-go of one gateway are ordered by the table's
//! lock, held across the store's release: a gateway is welcomed with what
//! it held, or let go and welcomed with nothing, never welcomed and then
//! let go. The table's lock is taken before any lock of the store.
//!
//! A session that was replaced applies no frame from the instant it is
//! told: the session asks its seat before each frame, under no lock. A
//! frame already past that question may still be written, so a gateway is
//! not let go while a session it replaced still runs: what that frame
//! writes lands under the row it was read for and leaves with it. A
//! replaced session ends at its next turn; one that never ends holds the
//! row for one period more and no longer. What it writes after that
//! lands under a key no row holds, and stands until the gateway is
//! welcomed and let go again.
//!
//! The table holds two bounds of its own. A ceiling of rows: a `Hello` that
//! would make a new row past it is refused, so the names a listener lets in
//! never grow the table without end; a present gateway's `Hello` makes no
//! row and is always taken. And one subject a key: the store keys a gateway
//! by a fingerprint of its subject, so a `Hello` of another subject under a
//! present gateway's key is refused, never merged into that gateway's row.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{watch, Notify},
    time::MissedTickBehavior,
};
use tracing::{info, warn};

use crate::{
    elector::Fence,
    store::{GatewayId, LeasesStore, MemoryStore},
};

/// Why a `Hello` takes no seat. A refused `Hello` changes nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Refused {
    /// A gateway of another subject is present under the same key.
    AnotherSubject,
    /// The table holds its ceiling of rows and this `Hello` would make a
    /// new one.
    Full { ceiling: usize },
    /// The table's read of the fence does not show the term the session
    /// read for this `Hello`: the fence is closed (`None`), or open under
    /// a later term.
    Stepped { read: u64, shown: Option<u64> },
}

/// The presence table.
pub(crate) struct Gateways {
    /// How long a gateway stays present after its last session ended.
    grace: Duration,
    /// How many rows the table holds at most: the gateways with a standing
    /// session or inside their grace period.
    ceiling: usize,
    fence: Arc<Fence>,
    store: Arc<MemoryStore>,
    table: Mutex<Table>,
}

#[derive(Default)]
struct Table {
    rows: FxHashMap<GatewayId, Row>,
    /// The highest term the table was read under since this replica last
    /// led; `None` when the last read found the fence closed, and before
    /// the first.
    led: Option<u64>,
    /// Numbers every session the table welcomed and never restarts: a
    /// session's number is its alone, whatever became of its gateway's row.
    sessions: u64,
}

struct Row {
    /// The verified subject, for the log.
    subject: String,
    /// The number of the gateway's latest session, `0` before the first: a
    /// stream's end is the row's only while the number is that stream's own.
    session: u64,
    /// When the last session ended; `None` while one stands.
    gone: Option<Instant>,
    /// Wakes the standing session when a newer one takes its place.
    superseded: Arc<Replaced>,
    /// The sessions a newer one replaced that have not ended yet. The
    /// gateway is not let go under one: a frame past its question of the
    /// seat may still be written.
    replaced: Vec<Weak<Replaced>>,
    /// Held for as long as this presence lasts: a [`Presence`] waits for
    /// its drop.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a decide stream subscribes to it; the Decide call is the step's next PR"
        )
    )]
    ended: watch::Sender<()>,
}

/// What a session is told when a newer one of its gateway takes its place.
/// The flag is asked before every frame, under no lock; the wake ends a
/// stream that waits.
#[derive(Default)]
struct Replaced {
    told: AtomicBool,
    wake: Notify,
}

/// A standing session's place in the table. Its drop is the session's end.
pub(crate) struct Seat {
    gateways: Arc<Gateways>,
    gateway: GatewayId,
    session: u64,
    superseded: Arc<Replaced>,
}

impl Seat {
    /// Resolves when a newer session of the gateway took this one's place.
    pub(crate) async fn superseded(&self) {
        self.superseded.wake.notified().await;
    }

    /// Whether a newer session of the gateway took this one's place: asked
    /// before a frame is applied, under no lock.
    pub(crate) fn replaced(&self) -> bool {
        self.superseded.told.load(Ordering::Acquire)
    }

    /// The session ended at `now`, where a case names the instant.
    #[cfg(test)]
    pub(crate) fn end(self, now: Instant) {
        self.gateways.gone(self.gateway, self.session, now);
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        // A no-op after `end`: the row's period started there.
        self.gateways
            .gone(self.gateway, self.session, Instant::now());
    }
}

/// A gateway's presence as a decide stream holds it.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Decide call holds one a decide stream; that call is the step's next PR"
    )
)]
#[derive(Clone)]
pub(crate) struct Presence(watch::Receiver<()>);

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Decide call ends a decide stream on it; that call is the step's next PR"
    )
)]
impl Presence {
    /// Resolves when the presence ended.
    pub(crate) async fn ended(&mut self) {
        // No value is ever sent: the row holds the sender and goes with
        // the presence.
        while self.0.changed().await.is_ok() {}
    }
}

impl Gateways {
    pub(crate) fn new(
        grace: Duration,
        ceiling: usize,
        fence: Arc<Fence>,
        store: Arc<MemoryStore>,
    ) -> Self {
        Self {
            grace,
            ceiling,
            fence,
            store,
            table: Mutex::new(Table::default()),
        }
    }

    /// The ceiling of rows.
    #[cfg(test)]
    pub(crate) fn ceiling(&self) -> usize {
        self.ceiling
    }

    /// How many replaced sessions the row of `gateway` notes.
    #[cfg(test)]
    pub(crate) fn noted(&self, gateway: GatewayId) -> usize {
        self.table
            .lock()
            .rows
            .get(&gateway)
            .map_or(0, |row| row.replaced.len())
    }

    /// The grace period, as a `Welcome` tells it.
    pub(crate) fn grace(&self) -> Duration {
        self.grace
    }

    /// A `Welcome` at `now` under `term`, the session's read of the fence,
    /// which the table's own read under its lock must show too: the
    /// gateway is present, and the session takes its seat. A session of
    /// the gateway that still stands is superseded. A gateway whose period
    /// had ended before this `Hello`, the sweep not yet passed, is let go
    /// first: the new presence starts with nothing of the earlier one.
    ///
    /// # Errors
    ///
    /// The fence no longer shows `term`: no seat is taken, no session is
    /// superseded and nothing is let go. Or a gateway of another subject
    /// is present under the key, or the row would be a new one past the
    /// table's ceiling. A row whose period is over counts until the
    /// sweep's next pass, a tenth of the period at most.
    pub(crate) fn welcome(
        self: &Arc<Self>,
        gateway: GatewayId,
        subject: &str,
        term: u64,
        now: Instant,
    ) -> Result<Seat, Refused> {
        let mut table = self.table.lock();
        let shown = self.fence.leader_term(now);
        Self::observe(&mut table, shown, now);
        // The seat is taken under the term this read shows or not at all:
        // a Hello read under a term the election has left would supersede
        // a session of the term that stands.
        if shown != Some(term) {
            return Err(Refused::Stepped { read: term, shown });
        }
        if table
            .rows
            .get(&gateway)
            .is_some_and(|row| self.over(row, now))
        {
            if let Some(row) = table.rows.remove(&gateway) {
                self.end(gateway, row);
            }
        }
        match table.rows.get(&gateway) {
            // One key, one subject: two names of one fingerprint are never
            // one gateway. Nothing of the present gateway is touched.
            Some(row) if row.subject != subject => {
                warn!(
                    present = %row.subject,
                    refused = %subject,
                    "a Hello's subject shares its key with a present gateway of another subject: refused"
                );
                return Err(Refused::AnotherSubject);
            }
            // A new row past the ceiling: the table's bound that is not time.
            // The table has no gauge yet: this line and the refusal are its
            // signal.
            None if table.rows.len() >= self.ceiling => {
                warn!(
                    refused = %subject,
                    ceiling = self.ceiling,
                    "a Hello would make a new row past the table's ceiling of present gateways: refused"
                );
                return Err(Refused::Full {
                    ceiling: self.ceiling,
                });
            }
            _ => {}
        }
        table.sessions += 1;
        let session = table.sessions;
        let row = table.rows.entry(gateway).or_insert_with(|| Row {
            subject: subject.to_owned(),
            session: 0,
            gone: None,
            superseded: Arc::new(Replaced::default()),
            replaced: Vec::new(),
            ended: watch::channel(()).0,
        });
        if row.session > 0 && row.gone.is_none() {
            row.superseded.told.store(true, Ordering::Release);
            row.superseded.wake.notify_one();
            // The replaced session runs until it reads the word: the row
            // waits for its end.
            row.replaced.retain(|session| session.strong_count() > 0);
            row.replaced.push(Arc::downgrade(&row.superseded));
        }
        row.session = session;
        row.gone = None;
        row.superseded = Arc::new(Replaced::default());
        Ok(Seat {
            gateways: Arc::clone(self),
            gateway,
            session,
            superseded: Arc::clone(&row.superseded),
        })
    }

    /// The gateway's presence at `now` for a decide stream's open: `None`
    /// when it is not present. No decision asks: a stream's open and end do.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Decide call asks it at a decide stream's open; that call is the step's next PR"
        )
    )]
    pub(crate) fn presence(&self, gateway: GatewayId, now: Instant) -> Option<Presence> {
        let mut table = self.table.lock();
        Self::observe(&mut table, self.fence.leader_term(now), now);
        let row = table.rows.get(&gateway)?;
        (!self.over(row, now)).then(|| Presence(row.ended.subscribe()))
    }

    /// One pass of the sweep at `now`: every gateway whose period is over
    /// is let go. Returns how many were let go. A replica that does not
    /// lead at `now` lets nothing go.
    pub(crate) fn sweep(&self, now: Instant) -> usize {
        let mut table = self.table.lock();
        if !Self::observe(&mut table, self.fence.leader_term(now), now) {
            return 0;
        }
        let over: Vec<GatewayId> = table
            .rows
            .iter()
            .filter(|(_, row)| self.over(row, now))
            .map(|(gateway, _)| *gateway)
            .collect();
        for gateway in &over {
            if let Some(row) = table.rows.remove(gateway) {
                self.end(*gateway, row);
            }
        }
        over.len()
    }

    /// The sweep's task: a pass every tenth of the grace period until
    /// `stop`, on no stream's path.
    pub(crate) async fn run(&self, mut stop: watch::Receiver<bool>) {
        let mut tick = tokio::time::interval((self.grace / 10).max(Duration::from_millis(1)));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    self.sweep(Instant::now());
                }
                _ = stop.wait_for(|stop| *stop) => break,
            }
        }
    }

    /// Whether `term`, a read of the fence at `now`, shows a leader, with
    /// the take's rule applied to the table: at the first read under a term
    /// above the one the table was last read under, or after a read that
    /// found the fence closed, every gone gateway's period starts again at
    /// `now`.
    fn observe(table: &mut Table, term: Option<u64>, now: Instant) -> bool {
        match (term, table.led) {
            // The term the table was last read under, or a read that lost
            // its race to a later one: nothing moved.
            (Some(term), Some(led)) if term <= led => {}
            (Some(_), _) => {
                for row in table.rows.values_mut() {
                    if row.gone.is_some() {
                        row.gone = Some(now);
                    }
                }
                table.led = term;
            }
            (None, _) => table.led = None,
        }
        term.is_some()
    }

    /// Whether the row is over at `now`: its grace period past with no
    /// session a newer one replaced still running, or twice the period
    /// past whatever runs. A replaced session is told at its replacement
    /// and ends at its next turn; one that never ends holds its gateway's
    /// row for one period more and no longer.
    fn over(&self, row: &Row, now: Instant) -> bool {
        row.gone.is_some_and(|gone| {
            let silent = now.saturating_duration_since(gone);
            let runs = || {
                row.replaced
                    .iter()
                    .any(|session| session.strong_count() > 0)
            };
            silent >= self.grace.saturating_mul(2) || (silent >= self.grace && !runs())
        })
    }

    /// The end of a presence, under the table's lock: the store lets go
    /// what it holds under the gateway's key through its one door for a
    /// gone gateway (its lease records and its completions), then the row
    /// goes and its decide streams are told.
    fn end(&self, gateway: GatewayId, row: Row) {
        let leases = self.store.drop_gateway(gateway);
        info!(
            gateway = %row.subject,
            leases,
            "a gateway's grace period ended with no new Hello: its leases are released"
        );
        drop(row);
    }

    /// The session `session` of `gateway` ended at `now`: its grace period
    /// starts, unless a newer session stands or the period already runs.
    fn gone(&self, gateway: GatewayId, session: u64, now: Instant) {
        let mut table = self.table.lock();
        if let Some(row) = table.rows.get_mut(&gateway) {
            if row.session == session && row.gone.is_none() {
                row.gone = Some(now);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use tokio::{sync::watch, task::JoinSet, time::timeout};

    use super::{Gateways, Refused, Seat};
    use crate::{
        controller::lifecycle::{
            self,
            tests::{a_completed, worker},
        },
        elector::{ElectionSettings, Elector, MemoryLease, Tick},
        lease::{Lease, LeaseId},
        store::{GatewayId, HealthSink, LeaseRecord, LeaseState, LeasesStore, MemoryStore},
        view::WorkerId,
    };

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

    /// A candidate that took its own lease at the returned instant, a store,
    /// and the table over that candidate's fence and that store.
    pub(crate) fn leading(grace: Duration) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
        ceiled(grace, 1_024)
    }

    /// The same, with a table that holds `ceiling` rows.
    fn ceiled(
        grace: Duration,
        ceiling: usize,
    ) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
        let mut elector = Elector::new(
            Box::new(MemoryLease::new()),
            "controller-0@10.0.0.7:30100",
            PATIENT,
        )
        .expect("the settings are in order");
        let taken = Instant::now();
        assert_eq!(
            elector.tick(taken).expect("the take"),
            Tick::Acquired { term: 1 }
        );
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
    ) -> Seat {
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
            decode: None,
            term: 0,
            issued_at: now,
            dispatch_deadline: now + Duration::from_secs(5),
            state: LeaseState::Issued,
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

    /// What the store holds under a gateway's key ends with its presence:
    /// its completions with its lease records, and no other gateway's.
    #[test]
    fn a_gateway_let_go_leaves_no_completion_in_the_store() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let _stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
        for gateway in [A, B] {
            lifecycle::completed(&store, gateway, &a_completed(1, served), t0)
                .expect("the frame reads");
        }
        gone.end(t0);

        assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
        assert!(
            store.completed_ok_since(A, served, t0),
            "inside the period nothing is let go"
        );
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the gateway's completion went with it"
        );
        assert!(
            store.completed_ok_since(B, served, t0),
            "another gateway's stands"
        );
    }

    /// A `Hello` past the grace period lets the gateway go before it is
    /// welcomed anew, with no pass of the sweep between: the new presence
    /// starts with no completion of the one that ended.
    #[test]
    fn a_gateway_welcomed_anew_past_its_grace_period_starts_with_no_completion() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        lifecycle::completed(&store, A, &a_completed(1, served), t0).expect("the frame reads");
        gone.end(t0);

        let _anew = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the completion ended with the presence that wrote it"
        );
        assert_eq!(store.health_entries(), 0);
    }

    /// A frame of a replaced session that is past the seat's question may
    /// still be written. The gateway is not let go while that session
    /// runs: what the frame writes lands under the row it was read for and
    /// leaves with it.
    #[test]
    fn a_gateway_is_not_let_go_while_a_session_it_replaced_still_runs() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let second = welcomed(&gateways, A, "gateway-a", TERM, t0);
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert!(first.replaced() && second.replaced());
        assert_eq!(gateways.sweep(t0 + GRACE), 0, "two replaced sessions run");
        assert!(gateways.presence(A, t0 + GRACE).is_some());

        // The later of the two ends first: the earlier one holds the row.
        drop(second);
        assert_eq!(gateways.sweep(t0 + GRACE), 0, "a replaced session runs");
        // Its late frame, then its end.
        lifecycle::completed(&store, A, &a_completed(1, served), t0 + GRACE)
            .expect("the frame reads");
        drop(first);
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the late frame's completion left with the row"
        );
        assert_eq!(store.health_entries(), 0);
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
        assert_eq!(gateways.noted(A), 1, "the session the last Hello replaced");
        drop(stands);
    }

    /// The completions on file are bounded by who is present, never by who
    /// was ever welcomed: waves of gateways under the table's ceiling, each
    /// reporting a completion and let go, leave nothing behind.
    #[test]
    fn the_completions_on_file_do_not_grow_with_the_gateways_ever_welcomed() {
        let (_elector, store, gateways, t0) = ceiled(GRACE, 8);
        let served = worker(&store, "http://a:8000");
        let mut now = t0;
        for wave in 0..5_u64 {
            for n in 0..8_u64 {
                let gateway = GatewayId(100 + wave * 8 + n);
                let subject = format!("gateway-{wave}-{n}");
                let seat = welcomed(&gateways, gateway, &subject, TERM, now);
                lifecycle::completed(&store, gateway, &a_completed(1, served), now)
                    .expect("the frame reads");
                seat.end(now);
            }
            assert_eq!(store.health_entries(), 8, "a completion a gateway");
            now += GRACE;
            assert_eq!(gateways.sweep(now), 8);
            assert_eq!(store.health_entries(), 0, "wave {wave}");
        }
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
            Tick::Acquired { term: 2 }
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
            Tick::Renewed { term: TERM }
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
            Tick::Acquired { term: 2 }
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
            Tick::Acquired { term: 2 }
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
}

//! Cases of a pair's second ask of its decode: the document says when the
//! decode is picked, and a route that asks again reads the rows of the
//! instant its prefill answered.

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use smg_policy_controller::{
    spec::DecodePick,
    store::{LeasesStore, LoadSink, LoadSource, MemoryStore, WorkersStore},
    view::{LoadRow, Phase, WorkerId},
    TransportClass,
};
use smg_worker::identity::basic::{Worker, WorkerType};

use super::{
    testing::{pair_bed_picking, pair_in_flight, MODEL},
    Ask, End, Host, KeptAsk, Placed, GATEWAY,
};

const PREFILL: &str = "http://prefill-0:8000";
const DECODES: [&str; 2] = ["http://decode-0:8000", "http://decode-1:8000"];

/// A served side's end.
const SERVED: End = End {
    status: 200,
    class: TransportClass::Ok,
    ttft_micros: 0,
    tokens_in: 37,
    tokens_out: 4,
    duration_micros: 1_000,
};

/// One prefill and two decodes under the stock pair presets, with the
/// document's word for when a pair's decode is picked. The two decodes'
/// rows come with it.
fn bed(decode: DecodePick) -> (Host, Arc<MemoryStore>, [WorkerId; 2]) {
    let (host, store, _) = pair_bed_picking(decode, &[PREFILL], &DECODES);
    let rows = DECODES.map(|url| store.by_key(url).expect("the decode's row").id);
    (host, store, rows)
}

/// The engines report `busy` deep in its queue and `idle` empty, and the
/// store publishes the two loads. The publication's instant comes back: a
/// decision at it reads them.
fn report(store: &MemoryStore, busy: WorkerId, idle: WorkerId) -> Instant {
    let at = Instant::now();
    for (worker, queued) in [(busy, 80), (idle, 0)] {
        let load = LoadRow {
            queued,
            live_rate: 2_000,
            ..LoadRow::default()
        };
        store.record(worker, 0, load, at, LoadSource::Stream);
    }
    // A publication shows what waits once it is due.
    let published = store.publish_due().map_or(at, |due| due.max(at));
    store.publish(published);
    published
}

/// A pair's ask whose route asks its decode again, or does not.
fn ask(id: &'static [u8], decode_again: bool) -> Ask<'static> {
    Ask {
        phase: Phase::Prefill,
        prompt_tokens: 37,
        decode_again,
        ..Ask::of(id, MODEL)
    }
}

async fn place(host: &Host, ask: &Ask<'_>) -> Placed {
    let (placed, _room) = host
        .place_pair(ask, None, None)
        .await
        .expect("the pair is placed");
    placed
}

fn decode_of(placed: &Placed) -> WorkerId {
    placed.decode_id().expect("a pair's decode")
}

/// The leases the store holds live, read until `expected` stand or a bound
/// passes: an end is applied behind the call that says it.
fn live(store: &MemoryStore, expected: usize) -> usize {
    let patience = Instant::now() + Duration::from_secs(5);
    loop {
        let live = LeasesStore::live(store, GATEWAY);
        if live == expected || Instant::now() > patience {
            return live;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

/// Says a placed pair's sends and ends in a route's order behind its
/// second ask: the decode's dispatch, the prefill's end, the decode's end.
fn end(mut placed: Placed, now: Instant) {
    placed.decode_dispatched(now);
    placed.completed(SERVED, now);
    placed.decode_completed(SERVED, now);
}

#[tokio::test]
async fn a_decode_picked_after_its_prefill_is_chosen_on_the_loads_of_that_instant() {
    let (host, store, [d0, d1]) = bed(DecodePick::AfterPrefill);
    // The second decode reports a deep queue, so the first decision names
    // the first one.
    report(&store, d1, d0);
    let asked = ask(b"pair", true);
    let mut placed = place(&host, &asked).await;
    let prefill = placed.id();
    assert_eq!(decode_of(&placed), d0);
    // That decision booked the prefill alone.
    assert!(placed.decode_waits());
    assert_eq!(pair_in_flight(&store, prefill), 1);
    assert_eq!(pair_in_flight(&store, d0), 0);

    placed.dispatched(Instant::now());
    // While the prefill runs the loads turn over.
    let answered = report(&store, d0, d1);
    let moved = placed
        .decode_again(&host, &asked, |_| true, answered)
        .map(|decode| decode.url().to_owned());
    assert_eq!(decode_of(&placed), d1, "the second ask read the loads");
    assert_eq!(moved.as_deref(), Some(DECODES[1]));
    let sent_to = placed.decode().expect("the decode's worker");
    assert_eq!(*sent_to.worker_type(), WorkerType::Decode);
    assert_eq!(sent_to.url(), DECODES[1], "the worker is the row's");
    // The second ask booked its decode, and the first-named one holds
    // nothing of this request.
    assert!(!placed.decode_waits());
    assert_eq!(pair_in_flight(&store, d1), 1);
    assert_eq!(pair_in_flight(&store, d0), 0);
    // A decode asked once is not asked twice.
    let again = placed.decode_again(&host, &asked, |_| true, answered);
    assert!(again.is_none());
    assert_eq!(decode_of(&placed), d1);

    end(placed, answered);
    assert_eq!(live(&store, 0), 0);
    assert_eq!(pair_in_flight(&store, d1), 0);
}

#[tokio::test]
async fn a_decode_picked_with_its_prefill_is_decided_once() {
    // The rule's default, by name and by a document that names none of
    // the plan's rules, and an ask whose route never asks again under the
    // other rule: each is one decision that books both sides.
    for (rule, decode_again) in [
        (DecodePick::WithPrefill, true),
        (DecodePick::default(), true),
        (DecodePick::AfterPrefill, false),
    ] {
        let (host, store, [d0, d1]) = bed(rule);
        report(&store, d1, d0);
        let asked = ask(b"pair", decode_again);
        let mut placed = place(&host, &asked).await;
        assert_eq!(decode_of(&placed), d0, "{rule:?} {decode_again}");
        assert!(!placed.decode_waits(), "{rule:?} {decode_again}");
        assert_eq!(pair_in_flight(&store, placed.id()), 1);
        assert_eq!(pair_in_flight(&store, d0), 1, "{rule:?} {decode_again}");

        placed.dispatched(Instant::now());
        // The loads turn over, and nothing asks the decode again.
        let answered = report(&store, d0, d1);
        let again = placed.decode_again(&host, &asked, |_| true, answered);
        assert!(again.is_none(), "{rule:?} {decode_again}");
        assert_eq!(decode_of(&placed), d0, "the first pick's decode stands");
        assert_eq!(pair_in_flight(&store, d1), 0);

        end(placed, answered);
        assert_eq!(live(&store, 0), 0, "{rule:?} {decode_again}");
    }
}

#[tokio::test]
async fn a_second_ask_no_decode_of_which_the_route_admits_sends_the_first_decode() {
    let (host, store, decodes) = bed(DecodePick::AfterPrefill);
    let asked = ask(b"pair", true);
    let mut placed = place(&host, &asked).await;
    let first = decode_of(&placed);
    let now = Instant::now();
    placed.dispatched(now);

    // The route's rule refuses every decode the second ask names.
    let no_decode = |worker: &Arc<dyn Worker>| *worker.worker_type() != WorkerType::Decode;
    let again = placed.decode_again(&host, &asked, no_decode, Instant::now());
    assert!(again.is_none());
    assert_eq!(decode_of(&placed), first, "the first pick's decode is sent");
    // The decode side was given up: nothing stays booked for it.
    assert!(!placed.decode_waits());
    for decode in decodes {
        assert_eq!(pair_in_flight(&store, decode), 0, "{decode:?}");
    }

    end(placed, Instant::now());
    assert_eq!(live(&store, 0), 0, "the request ended well");
}

/// A route whose second ask is stages behind its first keeps the ask in
/// the placement. The second ask then goes under the rule the route kept
/// and under the rule of the call's own instant, both.
#[tokio::test]
async fn a_kept_ask_is_asked_again_under_the_routes_rule_and_the_calls() {
    let other = |worker: &dyn Worker| worker.url() != DECODES[1];
    for (kept_rule, call_rule) in [(true, false), (false, true)] {
        let (host, store, [d0, d1]) = bed(DecodePick::AfterPrefill);
        let host = Arc::new(host);
        report(&store, d1, d0);
        let asked = ask(b"pair", true);
        let mut placed = place(&host, &asked).await;
        assert_eq!(decode_of(&placed), d0);
        // One of the two rules refuses the second decode, the other nobody.
        let kept = move |worker: &dyn Worker| !kept_rule || other(worker);
        placed.keep_ask(KeptAsk::of(&host, &asked, kept));
        placed.dispatched(Instant::now());
        let answered = report(&store, d0, d1);
        let call = |worker: &Arc<dyn Worker>| !call_rule || other(worker.as_ref());
        // The loads favour the second decode, and a rule refuses it: the
        // request stays on the first.
        let moved = placed
            .decode_again_kept(call, answered)
            .map(|decode| decode.url().to_owned());
        assert_ne!(
            moved.as_deref(),
            Some(DECODES[1]),
            "{kept_rule} {call_rule}"
        );
        assert_eq!(decode_of(&placed), d0, "{kept_rule} {call_rule}");
        assert_eq!(pair_in_flight(&store, d1), 0);
        // The kept ask is spent.
        assert!(placed.decode_again_kept(|_| true, answered).is_none());
        end(placed, answered);
        assert_eq!(live(&store, 0), 0);
    }
}

/// A placement keeps an ask for a decode that waits, and for no other. A
/// route that knows the decode cannot move lets the kept ask go, and the
/// first decision's decode is booked at that word.
#[tokio::test]
async fn a_placement_keeps_an_ask_for_a_waiting_decode_alone() {
    // A decode decided with its prefill keeps nothing.
    let (host, store, [d0, d1]) = bed(DecodePick::WithPrefill);
    let host = Arc::new(host);
    report(&store, d1, d0);
    let asked = ask(b"pair", true);
    let mut placed = place(&host, &asked).await;
    placed.keep_ask(KeptAsk::of(&host, &asked, |_| true));
    placed.dispatched(Instant::now());
    let answered = report(&store, d0, d1);
    assert!(placed.decode_again_kept(|_| true, answered).is_none());
    assert_eq!(decode_of(&placed), d0);
    end(placed, answered);
    assert_eq!(live(&store, 0), 0);

    // A waiting decode whose route lets the kept ask go keeps its first
    // decode, booked there and then: nothing asks it again.
    let (host, store, [d0, d1]) = bed(DecodePick::AfterPrefill);
    let host = Arc::new(host);
    report(&store, d1, d0);
    let mut placed = place(&host, &asked).await;
    placed.keep_ask(KeptAsk::of(&host, &asked, |_| true));
    let sent = Instant::now();
    placed.dispatched(sent);
    assert_eq!(pair_in_flight(&store, d0), 0, "a chosen decode is booked");
    placed.keep_first_decode(sent);
    assert!(!placed.decode_waits());
    let booked = [d0, d1].map(|decode| pair_in_flight(&store, decode));
    assert_eq!(booked, [1, 0]);
    let answered = report(&store, d0, d1);
    assert!(placed.decode_again_kept(|_| true, answered).is_none());
    assert_eq!(decode_of(&placed), d0);
    placed.completed(SERVED, answered);
    placed.decode_completed(SERVED, answered);
    assert_eq!(live(&store, 0), 0);
}

#[tokio::test]
async fn a_first_decode_sent_with_no_second_ask_is_booked_at_its_send() {
    let (host, store, _) = bed(DecodePick::AfterPrefill);
    let mut placed = place(&host, &ask(b"pair", true)).await;
    let first = decode_of(&placed);
    assert_eq!(pair_in_flight(&store, first), 0);

    // A route that sends both sides at once says both dispatches together.
    let now = Instant::now();
    placed.dispatched(now);
    placed.decode_dispatched(now);
    assert!(!placed.decode_waits());
    assert_eq!(pair_in_flight(&store, first), 1);

    placed.completed(SERVED, now);
    placed.decode_completed(SERVED, now);
    assert_eq!(live(&store, 0), 0);
}

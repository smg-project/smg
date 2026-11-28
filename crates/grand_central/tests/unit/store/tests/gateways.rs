//! A gateway's leases and vetoes: what its hello pages restore, a veto
//! written while its note stands, and what leaves with a dropped gateway.

use super::*;

#[test]
fn hello_pages_restore_exactly_what_they_list_and_a_gone_gateway_loses_its_leases() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    leases.set_term(5);
    for id in 1..=3 {
        assert_eq!(leases.issue(lease(g1, id, 5, now)), Ok(()));
    }
    assert_eq!(leases.issue(lease(g2, 1, 5, now)), Ok(()));
    let pages = [
        lease(g1, 2, 4, now),
        lease(g1, 9, 4, now),
        lease(g1, 10, 6, now),
        lease(g2, 5, 4, now),
    ];
    assert_eq!(
        leases.restore(g1, &pages),
        2,
        "what the pages list, under the term, for this gateway"
    );
    assert!(leases.get(g1, LeaseId(1)).is_none(), "omitted: released");
    assert!(leases.get(g1, LeaseId(2)).is_some());
    assert!(
        leases.get(g1, LeaseId(9)).is_some(),
        "an older term is kept"
    );
    assert!(
        leases.get(g1, LeaseId(10)).is_none(),
        "a term ahead is skipped"
    );
    assert!(
        leases.get(g2, LeaseId(5)).is_none(),
        "another gateway's page"
    );
    assert_eq!(leases.live(g2), 1, "the other gateway is untouched");
    assert_eq!(leases.drop_gateways(&[g1]), [2]);
    assert_eq!(leases.live(g1), 0);
    assert_eq!(leases.live(g2), 1);
}

#[test]
fn a_noted_veto_is_written_only_while_its_note_stands() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let (first, second) = (now + Duration::from_secs(30), now + Duration::from_secs(60));
    let g = GatewayId(1);
    let w = ready(&store, "http://a:8000");

    // No note behind the write: the veto is written as it is said, here
    // with no hold.
    let written = store.gateway_veto_noted(g, w, Some(first), None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    // The note's end stands in the store: the later end is written.
    let written = store.gateway_veto_noted(g, w, Some(second), None, true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, first), "the later end is shown");

    // The row leaves and comes back under the same id, and its vetoes left
    // with it. A write that says its note stands writes nothing: the row
    // registered again takes no end of the row before it.
    let revision = store.get(w).expect("the row").revision;
    store.remove(w, revision).expect("removed");
    assert_eq!(ready(&store, "http://a:8000"), w, "the same id");
    for until in [Some(second), None] {
        let written = store.gateway_veto_noted(g, w, until, None, true, now);
        assert_eq!(written, VetoWrite::Left);
    }
    assert!(!store.vetoed(g, w, now));
    assert_eq!(store.health_entries(), 0);
    // With no note behind it the same end is the gateway's word of now,
    // and a clear behind a note that stands lifts it.
    let written = store.gateway_veto_noted(g, w, Some(second), None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(store.vetoed(g, w, now));
    let written = store.gateway_veto_noted(g, w, None, None, true, now);
    assert_eq!(written, VetoWrite::Written);
    assert!(!store.vetoed(g, w, now));

    // A worker without a row: a veto writes nothing, a clear is taken.
    let none = WorkerId(7);
    let written = store.gateway_veto_noted(g, none, Some(first), None, false, now);
    assert_eq!(written, VetoWrite::NoRow);
    let written = store.gateway_veto_noted(g, none, None, None, false, now);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(store.health_entries(), 0);
}

#[test]
fn a_dropped_gateway_takes_its_vetoes_and_no_other_gateways() {
    let store = MemoryStore::new();
    let now = Instant::now();
    let until = now + Duration::from_secs(60);
    let (g1, g2) = (GatewayId(1), GatewayId(2));
    let a = ready(&store, "http://a:8000");
    let b = ready(&store, "http://b:8000");
    for gateway in [g1, g2] {
        store.gateway_veto(gateway, a, Some(until));
    }
    store.gateway_veto(g1, b, Some(until));
    assert_eq!(store.health_entries(), 3);

    assert_eq!(store.drop_gateways(&[g1]), [0], "it held no lease");
    assert!(!store.vetoed(g1, a, now) && !store.vetoed(g1, b, now));
    // The other gateway's stand.
    assert!(store.vetoed(g2, a, now));
    assert_eq!(store.health_entries(), 1);
    // A gateway that holds nothing is dropped as cleanly.
    assert_eq!(store.drop_gateways(&[g1]), [0]);
    assert_eq!(store.health_entries(), 1);
}

#[test]
fn gateways_dropped_together_leave_whole_and_say_their_leases_in_the_order_asked() {
    let store = MemoryStore::new();
    let leases: &dyn LeasesStore = &store;
    let now = Instant::now();
    let until = now + Duration::from_secs(60);
    let (g1, g2, g3) = (GatewayId(1), GatewayId(2), GatewayId(3));
    let a = ready(&store, "http://a:8000");
    leases.set_term(5);
    let two = [lease(g1, 1, 5, now), lease(g1, 2, 5, now)];
    assert_eq!(leases.restore(g1, &two), 2);
    assert_eq!(leases.restore(g2, &[lease(g2, 3, 5, now)]), 1);
    assert_eq!(leases.restore(g3, &[lease(g3, 4, 5, now)]), 1);
    for gateway in [g1, g2, g3] {
        store.gateway_veto(gateway, a, Some(until));
    }
    assert_eq!(store.health_entries(), 3);

    // Asked out of order, and one of them twice: each count is its own
    // gateway's, where the caller named it.
    assert_eq!(leases.drop_gateways(&[g3, g1, g3]), [1, 2, 1]);
    for gateway in [g1, g3] {
        assert_eq!(leases.live(gateway), 0);
        assert!(!store.vetoed(gateway, a, now));
    }
    // The gateway the list did not name keeps what it held.
    assert_eq!(leases.live(g2), 1);
    assert!(store.vetoed(g2, a, now));
    assert_eq!(store.health_entries(), 1);
    // Asked in falling order the list is found whole all the same: the
    // store sorts what it searches.
    let falling = [GatewayId(6), GatewayId(5), GatewayId(4)];
    for gateway in falling {
        store.gateway_veto(gateway, a, Some(until));
    }
    assert_eq!(leases.drop_gateways(&falling), [0, 0, 0]);
    for gateway in falling {
        assert!(!store.vetoed(gateway, a, now), "{gateway:?} keeps its veto");
    }
    assert!(leases.drop_gateways(&[]).is_empty());
    assert_eq!(store.health_entries(), 1);
}

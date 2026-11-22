//! The cases of a publication's reader of one gateway's standing vetoes by
//! worker id ([`StoreSnapshot::gateway_vetoes`]): what a veto written
//! through the store's door reads as, and that the reader and the view of
//! the same ask name the same workers.
//!
//! Every case reads the reader's answer as a worker's id with whether the
//! veto's hold stands, never as a row of a pool.

use std::time::{Duration, Instant};

use super::{
    super::StandingVeto, admit, ask, facts, load, member, StoreSnapshot, ViewBuffers, ViewQuery,
    ASKER, MODEL,
};
use crate::{
    engine::view::{Role, WorkerId},
    store::{
        GatewayId, HealthSink, LeasesStore, LoadSink, LoadSource, MemoryStore, Reachability,
        VetoWrite, WorkerFacts, WorkersStore,
    },
};

/// A gateway that asks beside [`ASKER`].
const OTHER: GatewayId = GatewayId(2);

const MS: Duration = Duration::from_millis(1);
const S: Duration = Duration::from_secs(1);

/// A standing veto whose hold stands.
fn hold(id: WorkerId) -> StandingVeto {
    StandingVeto { id, held: true }
}

/// A standing veto with no hold, or past its hold's end.
fn doubt(id: WorkerId) -> StandingVeto {
    StandingVeto { id, held: false }
}

/// What `world`'s reader hands `gateway` of `model`'s single pool at `now`.
fn read_of(
    world: &StoreSnapshot,
    model: &str,
    gateway: GatewayId,
    now: Instant,
) -> Vec<StandingVeto> {
    world
        .gateway_vetoes(model, Role::Single, gateway)
        .standing_at(now)
        .collect()
}

/// The reader's answer in the store's standing publication.
fn read(store: &MemoryStore, model: &str, gateway: GatewayId, now: Instant) -> Vec<StandingVeto> {
    store.with_snapshot(|world| read_of(world, model, gateway, now))
}

/// A veto with no hold, written through the door a controller writes by.
fn doubted(store: &MemoryStore, gateway: GatewayId, id: WorkerId, until: Instant, now: Instant) {
    let written = store.gateway_veto_noted(gateway, id, Some(until), None, false, now);
    assert_eq!(written, VetoWrite::Written);
}

/// A veto written through today's door is read under its worker's id by
/// the gateway that wrote it, and by no other: the reader hands the run
/// the view masks its rows by (`StoreSnapshot::view`, and its case
/// `a_noted_veto_is_shown_and_lifted_in_the_published_world`). The ids
/// ascend whatever the order of the writes.
#[test]
fn a_veto_written_through_the_door_is_read_by_its_workers_id_for_its_gateway_alone() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids =
        ["http://a:8000", "http://b:8000", "http://c:8000"].map(|key| member(&store, key));
    ids.sort_unstable();
    // No veto stands in the store: nothing to read.
    assert_eq!(read(&store, MODEL, ASKER, t0), []);

    let until = t0 + 10 * S;
    doubted(&store, ASKER, ids[2], until, t0);
    doubted(&store, ASKER, ids[0], until, t0);
    let own = [doubt(ids[0]), doubt(ids[2])];
    assert_eq!(read(&store, MODEL, ASKER, t0), own);
    assert_eq!(read(&store, MODEL, OTHER, t0), []);

    doubted(&store, OTHER, ids[1], until, t0);
    assert_eq!(read(&store, MODEL, OTHER, t0), [doubt(ids[1])]);
    assert_eq!(read(&store, MODEL, ASKER, t0), own);
}

/// A veto is read until its end and as held until its hold's end, by the
/// view's own two tests: at its end's instant it stands no more, though
/// the publication still holds its entry. A hold noted past the veto's end
/// is cut to it, and a veto told with no kind holds whole (the view's
/// cases `a_veto_reads_as_an_open_breaker_while_its_hold_stands_and_as_a_doubt_after`
/// and `a_hold_noted_past_its_vetos_end_is_cut_to_that_end`).
#[test]
fn a_veto_is_read_until_its_end_and_held_until_its_holds_end() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    let at = |now: Instant| read(&store, MODEL, ASKER, now);
    let (hold_end, until) = (t0 + 2 * S, t0 + 5 * S);
    let written = store.gateway_veto_noted(ASKER, id, Some(until), Some(hold_end), false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(t0), [hold(id)]);
    assert_eq!(at(hold_end - MS), [hold(id)]);
    // From the hold's end to the veto's own the worker is doubted alone.
    assert_eq!(at(hold_end), [doubt(id)]);
    assert_eq!(at(until - MS), [doubt(id)]);
    assert_eq!(at(until), []);

    // A hold past the veto's end holds to that end and no longer.
    let later = until + 5 * S;
    let written = store.gateway_veto_noted(ASKER, id, Some(later), Some(later + 60 * S), false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(later - MS), [hold(id)]);
    assert_eq!(at(later), []);

    // The door that tells no kind: the whole veto holds.
    store.gateway_veto(ASKER, id, Some(until));
    assert_eq!(at(until - MS), [hold(id)]);
    assert_eq!(at(until), []);

    let written = store.gateway_veto_noted(ASKER, id, None, None, false, t0);
    assert_eq!(written, VetoWrite::Written);
    assert_eq!(at(t0), []);
}

/// A worker of two pools is read in each pool's ask, as its veto masks its
/// row in each (`a_veto_masks_its_workers_row_in_every_pool_it_serves`). A
/// model the store does not know and a role with no pool read nothing.
#[test]
fn a_worker_of_two_pools_is_read_in_each_pools_ask() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + S;
    let both = admit(
        &store,
        facts("http://a:8000", Role::Single, &["one", "two"]),
    );
    let one = admit(&store, facts("http://b:8000", Role::Single, &["one"]));
    store.gateway_veto(ASKER, both, Some(until));
    store.gateway_veto(ASKER, one, Some(until));
    let read_in = |model: &str| read(&store, model, ASKER, t0);
    assert!(read_in("one").contains(&hold(both)) && read_in("one").contains(&hold(one)));
    assert!(read_in("two").contains(&hold(both)));
    assert_eq!(read_in("three"), []);
    let prefills = store.with_snapshot(|world| {
        let vetoes = world.gateway_vetoes("one", Role::Prefill, ASKER);
        vetoes.standing_at(t0).count()
    });
    assert_eq!(prefills, 0);
}

/// A veto that left is read no more: a departed worker's leaves with its
/// row, a dropped gateway's with the gateway, and one past its end at the
/// next window's publication, as each leaves the view
/// (`a_departing_workers_vetoes_leave_alone_and_every_other_veto_stands`,
/// `a_dropped_gateways_vetoes_leave_in_one_publication_and_every_other_stands`,
/// `a_veto_past_its_end_leaves_at_the_next_windows_publication`).
#[test]
fn a_veto_that_left_is_read_no_more() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let mut ids =
        ["http://a:8000", "http://b:8000", "http://c:8000"].map(|key| member(&store, key));
    ids.sort_unstable();
    let [leaving, ending, staying] = ids;
    let (soon, until) = (t0 + 10 * MS, t0 + 60 * S);
    doubted(&store, ASKER, leaving, until, t0);
    doubted(&store, ASKER, ending, soon, t0);
    doubted(&store, ASKER, staying, until, t0);
    doubted(&store, OTHER, staying, until, t0);
    let all = [doubt(leaving), doubt(ending), doubt(staying)];
    assert_eq!(read(&store, MODEL, ASKER, t0), all);

    let revision = WorkersStore::get(&store, leaving)
        .expect("the row")
        .revision;
    store.remove(leaving, revision).expect("removed");
    store.publish_now();
    assert_eq!(
        read(&store, MODEL, ASKER, t0),
        [doubt(ending), doubt(staying)]
    );

    // A window's publication at the first veto's end takes its entry out:
    // an ask dated before that end reads it no more.
    store.record(staying, 0, load(1, 0), soon, LoadSource::Stream);
    assert_eq!(read(&store, MODEL, ASKER, t0), [doubt(staying)]);

    store.drop_gateways(&[ASKER]);
    assert_eq!(read(&store, MODEL, ASKER, t0), []);
    assert_eq!(read(&store, MODEL, OTHER, t0), [doubt(staying)]);
}

/// A pool names a worker another gateway alone reaches, and a gateway can
/// veto it. The view leaves that worker out of the asking gateway's list
/// (`a_veto_beside_host_local_rows_masks_its_own_row_in_each_gateways_slice`).
/// The reader hands the veto: its caller passes over a worker its own
/// list does not hold.
#[test]
fn a_veto_on_a_worker_its_gateway_does_not_reach_is_read_and_is_no_row_of_the_view() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let until = t0 + S;
    let register = |key: &str, reachable_from: Reachability| {
        admit(
            &store,
            WorkerFacts {
                reachable_from,
                ..facts(key, Role::Single, &[MODEL])
            },
        )
    };
    let open = register("http://a:8000", Reachability::Any);
    let theirs = register("http://b:8000", Reachability::Gateway(OTHER));
    store.gateway_veto(ASKER, open, Some(until));
    store.gateway_veto(ASKER, theirs, Some(until));
    let mut vetoed = [hold(open), hold(theirs)];
    vetoed.sort_unstable_by_key(|veto| veto.id);
    assert_eq!(read(&store, MODEL, ASKER, t0), vetoed);
    let (reached, named) = store.with_snapshot(|world| viewed(world, MODEL, &ask(t0)));
    assert_eq!(reached, [open]);
    assert_eq!(named, [hold(open)]);
}

/// What the view of `asked` names in `model`'s single pool: the workers of
/// the asking gateway's list, by id, and each one a veto masks with
/// whether a hold stands on it. A lent view names its masked rows beside
/// the rows, by position, and the held ones apart
/// ([`StoreSnapshot::held`]). A copied view carries the two bits on its
/// rows and names none beside them.
fn viewed(
    world: &StoreSnapshot,
    model: &str,
    asked: &ViewQuery,
) -> (Vec<WorkerId>, Vec<StandingVeto>) {
    let mut out = ViewBuffers::default();
    let Some(view) = world.view(model, Role::Single, asked, false, &mut out) else {
        return (Vec::new(), Vec::new());
    };
    let mut rows = Vec::new();
    let held = world.held(&view, asked, &mut rows);
    let id_at = |at: u32| view.candidates[at as usize].id;
    let lent = view.stalled.iter().map(|&at| StandingVeto {
        id: id_at(at),
        held: held.binary_search(&at).is_ok(),
    });
    let copied = view
        .candidates
        .iter()
        .filter(|row| row.health.stalled)
        .map(|row| StandingVeto {
            id: row.id,
            held: row.health.breaker_open,
        });
    let named = lent.chain(copied).collect();
    (view.candidates.iter().map(|row| row.id).collect(), named)
}

/// The run's draws: the same writes on every run.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Over a run of writes, clears, expiries, joins, leaves and steps of the
/// clock, the reader and the view answer every ask alike: for each gateway
/// and pool, the workers the reader hands that the gateway's list holds
/// are the workers the view names masked, and held where it names them
/// held. The pool of "one" is lent to every gateway. The pool of "two"
/// holds host-local rows, so its view is each gateway's own copy.
#[test]
fn the_reader_and_the_view_answer_every_ask_alike_over_a_run_of_writes() {
    const WORKERS: usize = 9;
    const GATEWAYS: [GatewayId; 3] = [ASKER, OTHER, GatewayId(3)];
    let store = MemoryStore::new();
    let t0 = Instant::now();
    // A worker serves "one", "two" or both. Two workers of "two" alone are
    // host-local, one to each of two gateways.
    let facts_of = |n: usize| {
        let models: &[&str] = match n % 3 {
            0 => &["one", "two"],
            1 => &["one"],
            _ => &["two"],
        };
        WorkerFacts {
            reachable_from: match n {
                2 => Reachability::Gateway(OTHER),
                5 => Reachability::Gateway(ASKER),
                _ => Reachability::Any,
            },
            ..facts(&format!("http://w{n}:8000"), Role::Single, models)
        }
    };
    let mut ids: Vec<WorkerId> = (0..WORKERS).map(|n| admit(&store, facts_of(n))).collect();
    let mut present = [true; WORKERS];
    let mut draws = Draws(0x9E37_79B9_7F4A_7C15);
    let mut now = t0;
    // What the run met: standing vetoes, held ones, vetoes of a copied
    // view, and vetoes the reader handed on a worker outside the list.
    let (mut standing, mut held, mut copied, mut outside) = (0_usize, 0_usize, 0_usize, 0_usize);
    for step in 0..600 {
        let gateway = GATEWAYS[draws.below(3) as usize];
        let n = draws.below(WORKERS as u64) as usize;
        let until = now + MS * (1 + draws.below(40) as u32);
        match draws.below(14) {
            // A veto through the door a controller writes by: with no
            // hold, with one inside the veto, or with one past its end.
            0..=5 => {
                let hold = match draws.below(3) {
                    0 => None,
                    _ => Some(now + MS * (1 + draws.below(60) as u32)),
                };
                let written =
                    store.gateway_veto_noted(gateway, ids[n], Some(until), hold, false, now);
                let expected = if present[n] {
                    VetoWrite::Written
                } else {
                    VetoWrite::NoRow
                };
                assert_eq!(written, expected, "step {step}");
            }
            // The door that tells no kind: the whole veto holds.
            6 => {
                store.gateway_veto(gateway, ids[n], Some(until));
            }
            7 => {
                let written = store.gateway_veto_noted(gateway, ids[n], None, None, false, now);
                assert_eq!(written, VetoWrite::Written, "step {step}");
            }
            8 if present[n] => {
                let revision = WorkersStore::get(&store, ids[n]).expect("the row").revision;
                store.remove(ids[n], revision).expect("removed");
                store.publish_now();
                present[n] = false;
            }
            9 if !present[n] => {
                ids[n] = admit(&store, facts_of(n));
                present[n] = true;
            }
            10 => {
                store.drop_gateways(&[gateway]);
            }
            // A load report: a window's publication drops what has ended.
            11 if present[n] => {
                store.record(ids[n], 0, load(1, 0), now, LoadSource::Stream);
            }
            _ => now += MS * (1 + draws.below(15) as u32),
        }
        store.with_snapshot(|world| {
            for (model, lent) in [("one", true), ("two", false)] {
                for gateway in GATEWAYS {
                    let asked = ViewQuery {
                        gateway,
                        ..ask(now)
                    };
                    let (reached, named) = viewed(world, model, &asked);
                    let read = read_of(world, model, gateway, now);
                    let rising = read.windows(2).all(|pair| pair[0].id < pair[1].id);
                    assert!(rising, "step {step}: ascending by id, each once: {read:?}");
                    // The reader's vetoes on the workers of the gateway's
                    // own list: what a caller keeps of them.
                    let own: Vec<StandingVeto> = read
                        .iter()
                        .copied()
                        .filter(|veto| reached.binary_search(&veto.id).is_ok())
                        .collect();
                    assert_eq!(own, named, "step {step}: {model} for {gateway:?}");
                    standing += own.len();
                    held += own.iter().filter(|veto| veto.held).count();
                    outside += read.len() - own.len();
                    if !lent {
                        copied += own.len();
                    }
                }
            }
        });
    }
    // The run met every form: vetoes with and without a hold, in a lent
    // view and in a copied one, and on a worker the gateway's list leaves
    // out. Its draws are the same on every run, and they read 2,340
    // standing, 1,417 of them held, 1,055 in a copied view and 185 outside
    // the list: the bounds are half of each.
    assert!(
        standing > 1_000 && held > 500 && standing > held + 400 && copied > 500 && outside > 90,
        "{standing} standing, {held} held, {copied} in a copied view, {outside} outside the list"
    );
}

//! The cases of an encode request's named signals: each of its three sides
//! reads the named values of its own list. Each decides through the
//! engine's own entries, over the host's three lists and a load snapshot,
//! and hands no cell.

use std::mem;

use smg_worker::worker::{NamedValue, SchedulerLoadSnapshot};

use super::*;
use crate::{document::spec::Needs, engine::framework::Scorer, load::SignalColumn};

/// The three sides of an encode request, in the order of a host's lists:
/// the prefills' (list `0`), the decodes' (`1`), the encoders' (`2`).
const SIDES: [Phase; 3] = [Phase::Prefill, Phase::Decode, Phase::Encode];

/// What a side's stage read of the signal `held`: the side, each worker it
/// was asked about, and the value read. A worker is kept by its id and not
/// by its row's position: the engine numbers anew the rows it admits for a
/// request's items and the encoders it offers a key.
type SideRead = Arc<Mutex<Vec<(Phase, WorkerId, Option<f64>)>>>;

/// The side of each decision a stage scored, with whether its facts carried
/// signal cells.
type SideSeen = Arc<Mutex<Vec<(Phase, bool)>>>;

/// A scorer of a host's own build, under two names. As `reads-side` it
/// reads the signal `held` of every row it is asked about and keeps what it
/// read. As `sees-side` it reads no signal. Under both it keeps the side it
/// was asked on with whether the facts carried signal cells.
#[derive(Debug)]
struct OnSide {
    held: Option<SignalColumn>,
    read: SideRead,
    seen: SideSeen,
}

impl Lifecycle for OnSide {}

impl Scorer for OnSide {
    fn name(&self) -> &'static str {
        match self.held {
            Some(_) => "reads-side",
            None => "sees-side",
        }
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let seen = (facts.phase, facts.signal_cells.is_some());
        self.seen.lock().push(seen);
        if let Some(held) = &self.held {
            let rows = kept.iter().map(|&row| &candidates[row as usize]);
            let read = rows.map(|row| (facts.phase, row.id, held.read(facts, row)));
            self.read.lock().extend(read);
        }
        out.fill(0.0);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: self.held.is_some(),
            ..Needs::default()
        }
    }
}

/// A host of three lists whose engine holds the scorer above among its
/// stages under its two names, and what the scorer kept.
struct Reading {
    host: Host,
    read: SideRead,
    seen: SideSeen,
}

/// The host of `document`, an encode document as JSON, over as many
/// prefills, decodes and encoders as `workers` says.
fn reading(document: &str, [prefills, decodes, encoders]: [usize; 3]) -> Reading {
    let document: PolicySpec = serde_json::from_str(document).expect("a well-formed document");
    let (read, seen) = (SideRead::default(), SideSeen::default());
    let mut stages = StageRegistry::builtin();
    let (kept, saw) = (Arc::clone(&read), Arc::clone(&seen));
    let added = stages.scorer_with("reads-side", move |build| {
        Ok(Box::new(OnSide {
            held: Some(build.signal_column("held")?),
            read: Arc::clone(&kept),
            seen: Arc::clone(&saw),
        }))
    });
    added.expect("a free name");
    let (kept, saw) = (Arc::clone(&read), Arc::clone(&seen));
    let added = stages.scorer("sees-side", move || {
        Box::new(OnSide {
            held: None,
            read: Arc::clone(&kept),
            seen: Arc::clone(&saw),
        })
    });
    added.expect("a free name");
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let spec = spec.expect("the stages hold the scorer's two names");
    Reading {
        host: Host::under(spec, prefills, decodes, encoders),
        read,
        seen,
    }
}

/// An encode document under the signals `signals` declares: the profile of
/// each side as JSON, the prefill side's first and the encode side's last,
/// bound to every model.
fn epd(signals: &str, [prefill, decode, encode]: [&str; 3]) -> String {
    format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{{signals}}},
            "profiles":{{"prefill":{prefill},"decode":{decode},"encode":{encode}}},
            "bindings":[{{"match":{{"model":"*"}},"plan":"epd",
                          "profiles":{{"prefill":"prefill","decode":"decode",
                                       "encode":"encode"}}}}]}}"#
    )
}

/// The signal `held`, in tokens.
const HELD: &str = r#""held":{"unit":"tokens"}"#;
/// The two token counts a report offers by name.
const COUNTS: &str = r#""kv_tokens_capacity":{"unit":"tokens"},"kv_tokens_used":{"unit":"tokens"}"#;
/// A side's profile that reads `held` of every worker it scores.
const READS: &str = r#"{"score":["reads-side"],"pick":"best"}"#;
/// A side's profile that reads no signal.
const SEES: &str = r#"{"score":["sees-side"],"pick":"best"}"#;
/// A side's profile under the rule of the two token counts: a worker has
/// room where what it uses, with the request's own tokens, is within its
/// capacity.
const ROOM: &str = r#"{"filter":[{"has-room":{
    "used":{"signal.kv_tokens_used":1,"request.prompt_tokens":1},
    "capacity":"signal.kv_tokens_capacity"}}],"score":["least-requests"],"pick":"best"}"#;
/// A side's profile that reads no report.
const PLAIN: &str = r#"{"score":["least-requests"],"pick":"best"}"#;

/// A report of one rank that states `values` by name and uses `used` of
/// `capacity` tokens, the two counts a report offers by name.
fn rank(values: Vec<NamedValue>, used: i32, capacity: i32) -> WorkerLoadResponse {
    let rank = SchedulerLoadSnapshot {
        values,
        num_used_tokens: used,
        max_total_num_tokens: capacity,
        ..SchedulerLoadSnapshot::default()
    };
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![rank],
        ..WorkerLoadResponse::default()
    }
}

/// A snapshot of the report `of` makes for the worker at each place of each
/// of the host's three lists.
fn stated(host: &Host, of: impl Fn(usize, usize) -> WorkerLoadResponse) -> Arc<LoadSnapshot> {
    let lists = host.lists.iter().enumerate();
    let reports = lists.flat_map(|(list, members)| {
        let of = &of;
        (0..members.workers().len()).map(move |place| (url(list, place), of(list, place)))
    });
    LoadSnapshot::from_loads_for_test(reports.collect())
}

/// A snapshot in which each worker states `held` at its list's base more
/// than its place in its list: the prefills' base first.
fn stepped(host: &Host, bases: [f64; 3]) -> Arc<LoadSnapshot> {
    stated(host, |list, place| {
        let held = NamedValue {
            name: "held".into(),
            unit: "tokens".into(),
            value: bases[list] + place as f64,
            age_ms: None,
        };
        rank(vec![held], 0, 0)
    })
}

/// A snapshot in which one worker of each of the lists `ruled` alone has
/// room for a request, by the two token counts: the one at `fits`.
fn room_at(host: &Host, ruled: &[usize], fits: usize) -> Arc<LoadSnapshot> {
    stated(host, |list, place| {
        let full = ruled.contains(&list) && place != fits;
        rank(vec![], if full { 2_000 } else { 100 }, 1_000)
    })
}

/// What the sides read under [`stepped`] at the places `places` names of
/// each list, the prefills' first.
fn stepped_sides(bases: [f64; 3], places: [&[usize]; 3]) -> Vec<(Phase, usize, Option<f64>)> {
    let sides = SIDES.into_iter().zip(bases).zip(places);
    let read = sides.flat_map(|((side, base), places)| {
        let value = move |place: usize| Some(base + place as f64);
        places.iter().map(move |&place| (side, place, value(place)))
    });
    read.collect()
}

/// What the sides' stages read since the last call: each worker a side was
/// asked about once, by its place in that side's list, the prefills' first
/// and the encoders' last. A worker a side read is one of that side's list.
fn read_of(reading: &Reading) -> Vec<(Phase, usize, Option<f64>)> {
    let read = mem::take(&mut *reading.read.lock());
    let placed = |(side, id, value): (Phase, WorkerId, Option<f64>)| {
        let list = SIDES.iter().position(|&of| of == side);
        let list = list.expect("a side of the three");
        let mut workers = reading.host.lists[list].workers().iter();
        let place = workers.position(|worker| worker_id(worker.as_ref()) == id);
        (list, place.expect("a worker of its side's list"), value)
    };
    let mut read: Vec<_> = read.into_iter().map(placed).collect();
    read.sort_by_key(|&(list, place, _)| (list, place));
    read.dedup();
    let sided = |(list, place, value): (usize, usize, Option<f64>)| (SIDES[list], place, value);
    read.into_iter().map(sided).collect()
}

/// Which sides' facts carried signal cells since the last call: each side
/// that was asked once, the prefills' first.
fn seen_of(reading: &Reading) -> Vec<(Phase, bool)> {
    let mut seen = mem::take(&mut *reading.seen.lock());
    seen.sort_by_key(|&(side, cells)| (SIDES.iter().position(|&of| of == side), cells));
    seen.dedup();
    seen
}

/// A request of one item a key of `keys`.
fn items<'a>(keys: &'a [Identity<'a>]) -> EncodeItems<'a> {
    EncodeItems {
        keys,
        ..EncodeItems::default()
    }
}

/// Each side of an encode request reads its own list's values, each at the
/// worker's place in that side's list: at the entry that issues the pair's
/// lease and at the entry that issues none. The three lists here are of
/// different lengths and state different values at the same places. The
/// host hands the snapshot and no cell.
#[test]
fn each_side_of_an_encode_request_reads_its_own_lists_values() {
    let mut reading = reading(&epd(HELD, [READS; 3]), [2, 3, 4]);
    let bases = [10.0, 100.0, 1_000.0];
    reading.host.loads = Some(stepped(&reading.host, bases));
    let host = &reading.host;
    let said = (&[of(1), of(1), of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let asks = host.asks(said);
    let all = stepped_sides(bases, [&[0, 1], &[0, 1, 2], &[0, 1, 2, 3]]);

    let leased = (host.engine).decide_encode_at(&asks, items(&keys), &mut placed, || now);
    let (leased, _) = leased.expect("a decision");
    assert!(leased.lease.is_some() && placed.encoders().len() == 2);
    assert_eq!(read_of(&reading), all);
    assert_eq!(seen_of(&reading), SIDES.map(|side| (side, true)));

    let bare = (host.engine).decide_encode_without_lease(&asks, items(&keys), &mut placed, now);
    let bare = bare.expect("a decision");
    assert!(bare.pick().is_some() && placed.encoders().len() == 2);
    assert_eq!(read_of(&reading), all, "no lease");
}

/// A rule bounded by a named value keeps the workers an encode request
/// fits, on its decode side and on its encode side: the pair's decode is
/// the one decode whose report says it has room, and each key's encoder the
/// one encoder whose report says so, at both entries. The rule on the
/// encode side is read twice: where the engine asks which encoders are
/// open, by the list's places, and where a key is placed.
#[test]
fn a_rule_bounded_by_a_named_value_keeps_the_workers_an_encode_request_fits() {
    let mut reading = reading(&epd(COUNTS, [PLAIN, ROOM, ROOM]), [2, 4, 4]);
    let said = (&[of(1), of(1), of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item-a"), Identity::plain("item-b")];
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    for fits in 0..4 {
        reading.host.loads = Some(room_at(&reading.host, &[1, 2], fits));
        let host = &reading.host;
        let asks = host.asks(said);
        let both = |placed: &EncodePlaced| -> Vec<u32> {
            let encoders = placed.encoders().iter();
            encoders.map(|encoder| encoder.position).collect()
        };

        let leased = (host.engine).decide_encode_at(&asks, items(&keys), &mut placed, || now);
        let (leased, _) = leased.expect("a decision");
        assert_eq!(decode_of(&leased.decision), Some(fits));
        assert_eq!(both(&placed), [fits as u32; 2]);

        let bare = (host.engine).decide_encode_without_lease(&asks, items(&keys), &mut placed, now);
        assert_eq!(
            decode_of(&bare.expect("a decision")),
            Some(fits),
            "no lease"
        );
        assert_eq!(both(&placed), [fits as u32; 2], "no lease");
    }
}

/// A side whose profile reads no signal is lent no cells, while the one
/// side that reads one reads its list's values: each of the three sides in
/// its turn. Under the document whose encode side alone reads, the
/// encoders' values are read for a key and the pair's two sides are lent
/// none.
#[test]
fn a_side_of_an_encode_request_that_reads_no_signal_is_lent_no_cells() {
    let bases = [10.0, 100.0, 1_000.0];
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    for (at, reads) in SIDES.into_iter().enumerate() {
        let mut profiles = [SEES; 3];
        profiles[at] = READS;
        let mut reading = reading(&epd(HELD, profiles), [2, 2, 2]);
        reading.host.loads = Some(stepped(&reading.host, bases));
        let host = &reading.host;
        let (mut placed, now) = (EncodePlaced::new(), Instant::now());
        let asks = host.asks(said);
        let decided = (host.engine).decide_encode_at(&asks, items(&keys), &mut placed, || now);
        let (decided, _) = decided.expect("a decision");
        assert!(decided.decision.pick().is_some(), "{reads:?}");
        let mut places: [&[usize]; 3] = [&[]; 3];
        places[at] = &[0, 1];
        assert_eq!(read_of(&reading), stepped_sides(bases, places), "{reads:?}");
        let lent = SIDES.map(|side| (side, side == reads));
        assert_eq!(seen_of(&reading), lent, "{reads:?}");
    }
}

/// The encode side of an encode request runs under its pair's binding,
/// whatever binding covers the encode phase first, and it is lent its
/// encoders' cells by that binding's encode profile. Here a binding of one
/// profile that reads no signal takes the encode phase ahead of the pair's
/// binding, whose encode profile reads one: that profile's stage is asked,
/// and it reads the encoders' values. The pair's two sides read no signal
/// and are lent none.
#[test]
fn the_encode_side_reads_its_values_behind_a_binding_that_takes_the_encode_phase() {
    let document = format!(
        r#"{{"apiVersion":"smg.policy/v1","generation":1,
            "signals":{{{HELD}}},
            "profiles":{{"prefill":{SEES},"decode":{SEES},"encode":{READS},"alone":{PLAIN}}},
            "bindings":[{{"match":{{"model":"*","phase":["encode"]}},"profile":"alone"}},
                        {{"match":{{"model":"*"}},"plan":"epd",
                          "profiles":{{"prefill":"prefill","decode":"decode",
                                       "encode":"encode"}}}}]}}"#
    );
    let mut reading = reading(&document, [2, 2, 2]);
    let bases = [10.0, 100.0, 1_000.0];
    reading.host.loads = Some(stepped(&reading.host, bases));
    let host = &reading.host;
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let asks = host.asks(said);
    let decided = (host.engine).decide_encode_at(&asks, items(&keys), &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert!(decided.decision.pick().is_some() && placed.encoders().len() == 1);
    assert_eq!(read_of(&reading), stepped_sides(bases, [&[], &[], &[0, 1]]));
    let lent = SIDES.map(|side| (side, side == Phase::Encode));
    assert_eq!(seen_of(&reading), lent);
}

/// A request whose items narrow its rows reads each admitted worker's own
/// value, on all three sides: the engine numbers the rows it admits anew,
/// and their cells with them. Here the first prefill names a class of no
/// encoder, the last decode alone takes the images the request asks of its
/// decode, and the class of the two prefills left holds the second and the
/// fourth encoder.
#[test]
fn the_rows_admitted_for_an_encode_requests_items_read_their_own_values() {
    let mut reading = reading(&epd(HELD, [READS; 3]), [3, 3, 4]);
    reading.host.says_the_decode_takes_images(2);
    let bases = [10.0, 100.0, 1_000.0];
    reading.host.loads = Some(stepped(&reading.host, bases));
    let host = &reading.host;
    let said = (&[of(9), of(2), of(9), of(2)][..], &[1, 2, 2][..]);
    let keys = [Identity::plain("item")];
    let asked = EncodeItems {
        decode_images: true,
        ..items(&keys)
    };
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let asks = host.asks(said);
    let decided = (host.engine).decide_encode_at(&asks, asked, &mut placed, || now);
    let (decided, _) = decided.expect("a decision");
    assert_eq!(decode_of(&decided.decision), Some(2));
    assert_eq!(placed.encoders().len(), 1);
    let admitted = stepped_sides(bases, [&[1, 2], &[2], &[1, 3]]);
    assert_eq!(read_of(&reading), admitted);
}

/// Under no snapshot every worker of the three sides reads no value, and
/// the request is still decided: its pair is picked and its key takes an
/// encoder. Behind a snapshot the same host's sides read what it states.
#[test]
fn under_no_snapshot_the_three_sides_read_no_value_and_the_request_is_decided() {
    let mut reading = reading(&epd(HELD, [READS; 3]), [2, 2, 2]);
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let decide = |host: &Host, placed: &mut EncodePlaced| {
        let asks = host.asks(said);
        let decided = (host.engine).decide_encode_at(&asks, items(&keys), placed, || now);
        let (decided, _) = decided.expect("a decision");
        decided.decision.pick().is_some() && placed.encoders().len() == 1
    };
    assert!(decide(&reading.host, &mut placed));
    let none = |side| [(side, 0, None), (side, 1, None)];
    assert_eq!(read_of(&reading), SIDES.map(none).concat());
    assert_eq!(seen_of(&reading), SIDES.map(|side| (side, false)));

    let bases = [10.0, 100.0, 1_000.0];
    reading.host.loads = Some(stepped(&reading.host, bases));
    assert!(decide(&reading.host, &mut placed));
    assert_eq!(read_of(&reading), stepped_sides(bases, [&[0, 1]; 3]));
}

/// The decode asked again of an encode request reads its decode list's
/// values, of the snapshot of its own call: here every worker's value moved
/// between the two calls. That call asks no stage of the prefill side and
/// none of the encode side.
#[test]
fn the_decode_asked_again_of_an_encode_request_reads_its_lists_values() {
    let mut reading = reading(&epd(HELD, [READS; 3]), [2, 3, 2]);
    let bases = [10.0, 100.0, 1_000.0];
    reading.host.loads = Some(stepped(&reading.host, bases));
    let host = &reading.host;
    let said = (&[of(1), of(1)][..], &[1, 1][..]);
    let keys = [Identity::plain("item")];
    let (mut placed, now) = (EncodePlaced::new(), Instant::now());
    let asks = Asks {
        later: true,
        ..host.asks(said)
    };
    let first = (host.engine).decide_encode_at(&asks, items(&keys), &mut placed, || now);
    let (first, later) = first.expect("a decision");
    assert!(later && first.lease.is_some(), "the prefill is issued");
    let all = stepped_sides(bases, [&[0, 1], &[0, 1, 2], &[0, 1]]);
    assert_eq!(read_of(&reading), all);
    assert_eq!(seen_of(&reading), SIDES.map(|side| (side, true)));

    let moved = [50.0, 500.0, 5_000.0];
    let loads = stepped(host, moved);
    let call = Asks {
        loads: Some(&loads),
        ..asks
    };
    let again = decode_again(host, &call, first, now);
    assert!(again.decision.pick().is_some());
    let decodes = stepped_sides(moved, [&[], &[0, 1, 2], &[]]);
    assert_eq!(read_of(&reading), decodes);
    assert_eq!(seen_of(&reading), [(Phase::Decode, true)]);
}

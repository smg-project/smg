//! A registration of an engine's ranks: the rows one write makes, the
//! ranks that leave with it, and what refuses it whole. An add tried
//! again lays the ranks its first attempt left.

use super::*;

#[test]
fn a_width_of_four_makes_four_rows_in_one_write() {
    check_four_rows(local_door).expect("four ranks registered");
}

fn check_four_rows(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let mut facts = ranked(4);
    facts.rotate_left(2);
    for row in &mut facts[..2] {
        row.health.probes = false;
    }
    let answer = find_rows(&*door, &Registrant::Control, &facts).map_err(|_| "registered")?;
    assert_eq!(answer.len(), 4);
    let mut lives = Vec::new();
    for (index, stated) in facts.iter().enumerate() {
        let row = store.by_key(&stated.key).ok_or("the registered rank")?;
        assert_eq!(answer[index], MembershipWriteOutcome::Registered(row.id));
        assert_eq!(row.facts.key, stated.key);
        assert_eq!(row.facts.engine, URL);
        assert_eq!((row.facts.dp_rank, row.facts.dp_size), (stated.dp_rank, 4));
        let status = if index < 2 {
            WorkerStatus::Ready
        } else {
            WorkerStatus::Pending
        };
        assert_eq!(row.status, status);
        lives.push(row.facts.incarnation);
    }
    let steps: Vec<_> = lives
        .windows(2)
        .map(|pair| pair[1].checked_sub(pair[0]))
        .collect();
    assert_eq!(steps, [Some(2), Some(2), Some(1)], "{lives:?}");
    Ok(())
}

#[test]
fn a_registration_refused_at_one_rank_writes_no_rank() {
    check_bad_label(local_door);
}

fn check_bad_label(open: Open) {
    let (store, door) = open();
    let mut ranks = ranked(4);
    ranks[2].labels.insert("tier".to_owned(), "x".repeat(4097));
    let epoch = store.epoch();
    assert_eq!(
        find_rows(&*door, &Registrant::Control, &ranks),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "labels",
            why: "a label's value is at most 4096 bytes",
        }))
    );
    assert_eq!(store.epoch(), epoch);
    assert!(store.all_rows().1.is_empty());
}

#[test]
fn a_worker_that_comes_back_narrower_loses_the_ranks_it_no_longer_states() {
    check_narrower(local_door).expect("four ranks, then two");
    // Once more through a door whose store is journaled: the log tells
    // each of the four changes by its kind.
    let log = Arc::new(Log::new(64));
    let store = Arc::new(MemoryStore::journaled(Arc::clone(&log) as _));
    let door = StoreDoor::new(Arc::clone(&store));
    let (old, epoch) = narrower(&store, &door).expect("four ranks, then two");
    let mut told = Vec::new();
    log.read(Position::default(), Log::BATCH, &mut told)
        .expect("the log holds its start");
    let changes = told.iter().filter_map(|entry| match entry.event {
        Event::Member {
            epoch: at, change, ..
        } if at > epoch => Some(change),
        _ => None,
    });
    assert_eq!(
        changes.collect::<Vec<_>>(),
        [
            MembershipChange::Resized { id: old[0].id },
            MembershipChange::Resized { id: old[1].id },
            MembershipChange::Removed { id: old[2].id },
            MembershipChange::Removed { id: old[3].id },
        ]
    );
}

fn check_narrower(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    narrower(&store, &*door).map(drop)
}

/// A worker of four ranks comes back with two, through `door` over
/// `store`. The four rows it had, by rank, and the epoch its second
/// registration started behind.
fn narrower(store: &MemoryStore, door: &dyn Rows) -> Result<(Vec<WorkerRow>, u64), &'static str> {
    let maker = Registrant::Control;
    let four = ranked(4);
    find_rows(door, &maker, &four).map_err(|_| "registered four ranks")?;
    let old: Vec<_> = four
        .iter()
        .map(|facts| store.by_key(&facts.key).ok_or("the old rank"))
        .collect::<Result<_, _>>()?;
    let alias = WorkerFacts {
        key: format!("{URL}@03"),
        engine: format!("{URL}@03"),
        ..stated()
    };
    door.found_one(&maker, &alias.key, &|_| alias.clone())
        .map_err(|_| "the alias registered")?;
    // The door numbers the alias's life: what it wrote must stand.
    let alias = store.by_key(&alias.key).ok_or("the alias row")?.facts;
    let epoch = store.epoch();
    let two = ranked(2);
    assert_eq!(
        find_rows(door, &maker, &two),
        Ok(vec![
            MembershipWriteOutcome::InPlace(old[0].id),
            MembershipWriteOutcome::InPlace(old[1].id),
            MembershipWriteOutcome::Removed(old[2].id),
            MembershipWriteOutcome::Removed(old[3].id),
        ])
    );
    // Four writes and no more: each moves the epoch by one.
    assert_eq!(store.epoch(), epoch + 4, "four changes after this epoch");
    for row in &old[..2] {
        let resized = store.get(row.id).ok_or("the resized rank")?;
        assert_eq!(resized.facts.dp_size, 2);
        assert_eq!(resized.facts.incarnation, row.facts.incarnation);
    }
    assert!(four[2..]
        .iter()
        .all(|facts| store.by_key(&facts.key).is_none()));
    assert_eq!(store.by_key(&alias.key).map(|row| row.facts), Some(alias));
    Ok((old, epoch))
}

#[test]
fn ranks_left_above_removed_ranks_are_the_engines_still() {
    check_partial_removal(local_door).expect("two ranks of four removed");
}

/// Ranks zero and one of four are removed one row at a time. The two
/// left refuse an add, and leave with the next registration.
fn check_partial_removal(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let maker = Registrant::Control;
    let four = ranked(4);
    find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
    let old: Vec<_> = four
        .iter()
        .map(|facts| store.by_key(&facts.key).ok_or("the old rank"))
        .collect::<Result<_, _>>()?;
    for row in &old[..2] {
        door.gone(Remover::Registrant(&maker), &row.facts.key, standing(row))
            .map_err(|_| "a low rank removed")?;
    }
    let epoch = store.epoch();
    let two = ranked(2);
    assert_eq!(
        find_rows_over(&*door, &maker, Precondition::Absent, &two),
        Err(MembershipWriteRefused::Stands)
    );
    assert_eq!(store.epoch(), epoch);

    let answer = find_rows(&*door, &maker, &two).map_err(|_| "two ranks registered")?;
    assert!(matches!(
        answer[..2],
        [
            MembershipWriteOutcome::Registered(_),
            MembershipWriteOutcome::Registered(_)
        ]
    ));
    assert_eq!(
        answer[2..],
        [
            MembershipWriteOutcome::Removed(old[2].id),
            MembershipWriteOutcome::Removed(old[3].id)
        ]
    );
    assert!(four[2..]
        .iter()
        .all(|facts| store.by_key(&facts.key).is_none()));
    Ok(())
}

#[test]
fn a_plain_worker_that_comes_back_ranked_loses_its_plain_row() {
    check_ranked_then_plain(local_door).expect("plain, ranked and plain again");
}

fn check_ranked_then_plain(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let maker = Registrant::Control;
    find_rows(&*door, &maker, &[stated()]).map_err(|_| "registered the plain row")?;
    let plain = store.by_key(URL).ok_or("the plain row")?;
    let four = ranked(4);
    let answer = find_rows(&*door, &maker, &four).map_err(|_| "registered four ranks")?;
    assert_eq!(answer.len(), 5);
    assert_eq!(answer[4], MembershipWriteOutcome::Removed(plain.id));
    assert_eq!(store.by_key(URL), None);
    let ids: Vec<_> = four
        .iter()
        .map(|facts| store.by_key(&facts.key).map(|row| row.id).ok_or("the rank"))
        .collect::<Result<_, _>>()?;
    let answer = find_rows(&*door, &maker, &[stated()]).map_err(|_| "plain again")?;
    let fresh = store.by_key(URL).ok_or("the new plain row")?;
    let expected: Vec<_> = std::iter::once(MembershipWriteOutcome::Registered(fresh.id))
        .chain(ids.into_iter().map(MembershipWriteOutcome::Removed))
        .collect();
    assert_eq!(answer, expected);
    assert!(four.iter().all(|facts| store.by_key(&facts.key).is_none()));
    assert_ne!(plain.id, fresh.id);
    Ok(())
}

#[test]
fn one_row_under_the_engines_own_url_is_the_engine_at_any_width() {
    check_whole_row(local_door).expect("the engine stands as one row");
}

/// A worker that serves several engines over one connection states
/// their count and no rank: one row under the engine's own URL, which
/// is no rank of that width.
fn check_whole_row(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let maker = Registrant::Control;
    let whole = WorkerFacts {
        dp_size: 4,
        ..stated()
    };
    let answer =
        find_rows(&*door, &maker, std::slice::from_ref(&whole)).map_err(|_| "registered")?;
    let row = store.by_key(URL).ok_or("the engine's one row")?;
    assert_eq!(answer, [MembershipWriteOutcome::Registered(row.id)]);
    assert_eq!((row.facts.dp_rank, row.facts.dp_size), (0, 4));
    // The same facts again write nothing, and another width is laid in
    // place.
    assert_eq!(
        find_rows(&*door, &maker, std::slice::from_ref(&whole)),
        Ok(vec![MembershipWriteOutcome::Unmoved(row.id)])
    );
    let wider = WorkerFacts {
        dp_size: 8,
        ..whole
    };
    assert_eq!(
        find_rows(&*door, &maker, &[wider]),
        Ok(vec![MembershipWriteOutcome::InPlace(row.id)])
    );
    assert_eq!(store.by_key(URL).map(|row| row.facts.dp_size), Some(8));
    // Four ranks retire it, as they retire a row of one.
    let answer = find_rows(&*door, &maker, &ranked(4)).map_err(|_| "four ranks")?;
    assert_eq!(answer.len(), 5);
    assert_eq!(answer[4], MembershipWriteOutcome::Removed(row.id));
    assert_eq!(store.by_key(URL), None);
    Ok(())
}

#[test]
fn another_makers_row_of_the_engine_refuses_the_registration_whole() {
    check_other_maker(local_door).expect("the control row stands alone");
}

fn check_other_maker(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let provider = Registrant::provider("pool", "private/key").map_err(|_| "the provider")?;
    find_rows(&*door, &Registrant::Control, &[stated()]).map_err(|_| "the control registration")?;
    let before = store.all_rows();
    assert_eq!(
        find_rows(&*door, &provider, &ranked(4)),
        Err(MembershipWriteRefused::NotRegistrant(
            "the row was made by a control call: its maker alone changes it".to_owned()
        ))
    );
    assert_eq!(store.all_rows(), before);
    Ok(())
}

#[test]
fn ranks_that_are_not_zero_to_the_width_each_once_are_refused() {
    check_invalid_ranks(local_door);
}

fn check_invalid_ranks(open: Open) {
    let (store, door) = open();
    let four = ranked(4);
    let mut duplicate = four.clone();
    duplicate[1] = four[0].clone();
    let mut past_width = four.clone();
    past_width[3].dp_rank = 4;
    past_width[3].key = format!("{URL}@4");
    let mut another_engine = four.clone();
    another_engine[1].engine = "http://other:8000".to_owned();
    // The door reads the engine from the first row: a first row that
    // states another, or every row stating another, is held to its key.
    let mut another_first = four.clone();
    another_first[0].engine = "http://other:8000".to_owned();
    let mut another_all = four.clone();
    for facts in &mut another_all {
        facts.engine = "http://other:8000".to_owned();
    }
    let mut wrong_key = four.clone();
    wrong_key[1].key = format!("{URL}@01");
    let mut wrong_width = four.clone();
    wrong_width[1].dp_size = 2;
    let mut wrong_plain = stated();
    wrong_plain.dp_rank = 1;
    let missing_rank = four[..3].to_vec();
    for ranks in [
        duplicate,
        past_width,
        another_engine,
        another_first,
        another_all,
        wrong_key,
        wrong_width,
        vec![wrong_plain],
        missing_rank,
    ] {
        assert!(
            matches!(
                find_rows(&*door, &Registrant::Control, &ranks),
                Err(MembershipWriteRefused::Past(FactPastBound {
                    fact: "dp_rank",
                    ..
                }))
            ),
            "{ranks:?}"
        );
        assert_eq!(store.all_rows(), (0, Vec::new()));
    }
    // Facts under another key than their row's URL are refused by the
    // key's name, ahead of the group's shape.
    let row = Stated {
        url: "http://other:8000",
        state: &|_| stated(),
    };
    assert!(matches!(
        door.rows(&Registrant::Control, Precondition::Any, &[row]),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "key",
            ..
        }))
    ));
    assert_eq!(store.all_rows(), (0, Vec::new()));
}

#[test]
fn the_same_ranks_stated_again_write_nothing() {
    let (store, door) = door();
    check_unchanged(&store, &door).expect("the same four ranks stand");
}

fn check_unchanged(store: &MemoryStore, door: &dyn Rows) -> Result<(), &'static str> {
    let maker = Registrant::Control;
    let four = ranked(4);
    let first = find_rows(door, &maker, &four).map_err(|_| "registered four ranks")?;
    let epoch = store.epoch();
    let expected = first
        .into_iter()
        .map(|wrote| match wrote {
            MembershipWriteOutcome::Registered(id) => Ok(MembershipWriteOutcome::Unmoved(id)),
            _ => Err("a new rank must be registered"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(find_rows(door, &maker, &four), Ok(expected));
    assert_eq!(store.epoch(), epoch);
    Ok(())
}

#[test]
fn the_write_of_one_row_holds_no_group_rule_and_takes_no_row() {
    check_one_row(local_door);
}

/// The first form of the write, as its callers hand it: a rank laid
/// alone, and one rank's row changed beside the other.
pub(crate) fn check_one_row(open: Open) {
    let (store, door) = open();
    let ranks = ranked(2);
    let one = |facts: &WorkerFacts, expect| {
        door.found(&Registrant::Control, expect, &facts.key, &|_| facts.clone())
    };
    let low = one(&ranks[0], Precondition::Absent);
    assert!(
        matches!(low, Ok(MembershipWriteOutcome::Registered(_))),
        "{low:?}"
    );
    // An add of the row again is refused before a fact is stated,
    // whoever made the row: one row has no rest to lay.
    let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };
    let pool = Registrant::provider("pool", "private/key").expect("a maker");
    for maker in [&Registrant::Control, &pool] {
        assert_eq!(
            door.found(maker, Precondition::Absent, &ranks[0].key, &unstated),
            Err(MembershipWriteRefused::Stands)
        );
    }
    // The rows write holds the engine to its width where one rank stands.
    let epoch = store.epoch();
    assert!(matches!(
        find_rows(&*door, &Registrant::Control, &ranks[1..]),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "dp_rank",
            ..
        }))
    ));
    assert_eq!(store.epoch(), epoch);
    let high = one(&ranks[1], Precondition::Absent);
    assert!(
        matches!(high, Ok(MembershipWriteOutcome::Registered(_))),
        "{high:?}"
    );

    // One rank's row is changed where it stands, and the other stays.
    let other = store.by_key(&ranks[0].key).expect("the low rank");
    let seen = standing(&store.by_key(&ranks[1].key).expect("the high rank"));
    let resized = WorkerFacts {
        max_running: Some(8),
        ..ranks[1].clone()
    };
    let wrote = one(&resized, Precondition::AtRevision(seen));
    assert!(
        matches!(
            wrote,
            Ok(MembershipWriteOutcome::InPlace(_) | MembershipWriteOutcome::Another { .. })
        ),
        "{wrote:?}"
    );
    assert_eq!(store.by_key(&ranks[0].key), Some(other));
    assert_eq!(
        store.by_key(&ranks[1].key).map(|row| row.facts.max_running),
        Some(Some(8))
    );
}

#[test]
fn an_add_tried_again_lays_the_ranks_its_first_attempt_left() {
    check_same_add(local_door).expect("the same add");
}

/// The rows a lost lead leaves, then the same add: the rank that stands
/// answers unmoved under its id and the others are registered. With
/// every rank standing the engine is whole, and the add is refused.
fn check_same_add(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let three = ranked(3);
    let first = door
        .found(
            &Registrant::Control,
            Precondition::Absent,
            &three[0].key,
            &|_| three[0].clone(),
        )
        .map_err(|_| "the first rank alone")?;
    let add = || find_rows_over(&*door, &Registrant::Control, Precondition::Absent, &three);
    let wrote = add().map_err(|_| "the same add")?;
    let [MembershipWriteOutcome::Unmoved(stood), MembershipWriteOutcome::Registered(_), MembershipWriteOutcome::Registered(_)] =
        wrote.as_slice()
    else {
        return Err("one rank unmoved and two registered");
    };
    assert_eq!(first, MembershipWriteOutcome::Registered(*stood));
    assert_eq!(store.all_rows().1.len(), 3);
    let epoch = store.epoch();
    assert_eq!(add(), Err(MembershipWriteRefused::Stands));
    assert_eq!(store.epoch(), epoch);
    Ok(())
}

#[test]
fn an_add_over_a_standing_rank_that_is_not_its_own_writes_nothing() {
    check_add_not_its_own(local_door).expect("each add refused");
}

/// An add lays the rest by what stands alone. A standing rank whose
/// facts differ, a rank another maker wrote and a rank past the stated
/// width each refuse it whole as a row that stands, and nothing is
/// written.
fn check_add_not_its_own(open: Open) -> Result<(), &'static str> {
    let (store, door) = open();
    let three = ranked(3);
    door.found(
        &Registrant::Control,
        Precondition::Absent,
        &three[2].key,
        &|_| three[2].clone(),
    )
    .map_err(|_| "the last rank alone")?;
    let before = store.all_rows();
    let refused = |maker: &Registrant, facts: &[WorkerFacts]| {
        let add = find_rows_over(&*door, maker, Precondition::Absent, facts);
        assert_eq!(add, Err(MembershipWriteRefused::Stands));
        assert_eq!(store.all_rows(), before);
    };

    // The add states a label the standing rank does not.
    let mut labelled = three.clone();
    for facts in &mut labelled {
        facts.labels.insert("tier".to_owned(), "gold".to_owned());
    }
    refused(&Registrant::Control, &labelled);
    // Another maker states the same rows.
    let provider = Registrant::provider("pool", "private/key").map_err(|_| "the provider")?;
    refused(&provider, &three);
    // The standing rank is past the width the add states.
    refused(&Registrant::Control, &ranked(2));
    Ok(())
}

#[test]
fn no_row_stated_is_refused() {
    check_empty(local_door);
}

fn check_empty(open: Open) {
    let (store, door) = open();
    assert!(matches!(
        door.rows(&Registrant::Control, Precondition::Any, &[]),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "rows",
            ..
        }))
    ));
    assert_eq!(store.all_rows(), (0, Vec::new()));
}

#[cfg(feature = "controller")]
pub(crate) fn check_rank_rows(open: Open) -> Result<(), &'static str> {
    check_four_rows(open)?;
    check_bad_label(open);
    check_narrower(open)?;
    check_partial_removal(open)?;
    check_ranked_then_plain(open)?;
    check_whole_row(open)?;
    check_other_maker(open)?;
    check_invalid_ranks(open);
    check_same_add(open)?;
    check_add_not_its_own(open)?;
    let (store, door) = open();
    check_unchanged(&store, &*door)?;
    check_empty(open);
    Ok(())
}

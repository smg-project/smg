//! A partition that lists nobody leaves the table. Its cases cover what
//! the feed, a later entry and a decider that held it find there.

use super::*;

#[test]
fn a_model_whose_last_worker_left_takes_its_partition_with_it() {
    let (scorer, pool) = three_classes();
    let held = of(&scorer, "m");
    // A snapshot that names one of the three keeps the partition.
    scorer.on_membership(&Membership::Snapshot(&[pool[2].id]));
    assert_eq!(loads(&scorer), (vec![3_000], 6_000));
    assert!(!held.lock().left);
    // The last worker takes it along, window and all: the table holds
    // no model that lists nobody, and no worker's index names it.
    scorer.on_membership(&Membership::Removed(pool[2].id));
    assert!(scorer.partitions.load().is_empty());
    assert!(scorer.listed.load().is_empty());
    assert!(held.lock().left, "a decision that holds it takes the next");
    // The three are fenced: a view that still shows them lists none,
    // and the partition its decision made leaves the table again.
    assert_eq!(led(&scorer, &pool, 3_000, 16 * SECOND), None);
    assert!(scorer.partitions.load().is_empty());
    // The feed names them again, and the model's next decision lists
    // them from an even split.
    for worker in &pool {
        scorer.on_membership(&Membership::Added(worker.id));
    }
    assert_eq!(led(&scorer, &pool, 3_000, 17 * SECOND), Some(2));
    assert_eq!(loads(&scorer), (vec![0, 0, 0], 0));
}

/// The feed's walk takes a partition out where its word emptied the
/// listing, and nowhere else. An entry a decision made and has listed
/// nobody in yet stays, whatever the word, and that decision lists in
/// it.
#[test]
fn a_word_of_the_feed_leaves_an_entry_that_lists_nobody_yet() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    let made = scorer.partition("m");
    let (gone, new) = (WorkerId(7), WorkerId(8));
    for word in [
        Membership::Removed(gone),
        Membership::Replaced { old: gone, new },
        Membership::Left {
            id: gone,
            models: &["m".into()],
        },
        Membership::Snapshot(&[pool[0].id, pool[1].id]),
    ] {
        scorer.on_membership(&word);
    }
    assert!(!made.lock().left, "the entry's own decision lists in it");
    assert_eq!(led(&scorer, &pool, 0, SECOND), Some(0));
    assert!(Arc::ptr_eq(&of(&scorer, "m"), &made));
    assert_eq!(loads(&scorer), (vec![0, 0], 0));
}

#[test]
fn a_view_of_fenced_rows_alone_leaves_no_partition_behind() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    // The model's two workers leave, and its partition with the last.
    scorer.on_membership(&Membership::Removed(pool[0].id));
    scorer.on_membership(&Membership::Removed(pool[1].id));
    assert!(scorer.partitions.load().is_empty());
    // A decision whose view was captured before the leaves shows the
    // two alone. Nobody is listed, and the partition the decision made
    // leaves the table again: no later word of the feed would take it.
    assert_eq!(led(&scorer, &pool, 300, 2 * SECOND), None);
    assert!(scorer.partitions.load().is_empty(), "an empty entry stands");
    scorer.on_membership(&Membership::Snapshot(&[]));
    assert!(scorer.partitions.load().is_empty());
    assert!(scorer.listed.load().is_empty());
    // The feed names one of the two again, and the model's next view
    // lists it in a partition of its own.
    scorer.on_membership(&Membership::Added(pool[0].id));
    assert_eq!(led(&scorer, &pool, 300, 3 * SECOND), Some(0));
    assert_eq!(ranges(&scorer), [(0, 0, u32::MAX)]);

    // A decider that took the entry before it left finds it marked,
    // and takes the model's next one.
    let scorer = self::scorer(NEVER, 1.5, 5);
    scorer.on_membership(&Membership::Removed(pool[0].id));
    scorer.on_membership(&Membership::Removed(pool[1].id));
    let held = scorer.partition("m");
    assert_eq!(led(&scorer, &pool, 300, SECOND), None);
    assert!(held.lock().left, "a decision that holds it takes the next");
    assert!(scorer.partitions.load().is_empty());
}

/// The removal takes a partition out only where it still stands under
/// its model: an entry made since under the same name stays.
#[test]
fn forgetting_a_partition_leaves_an_entry_made_since_under_its_model() {
    let scorer = scorer(NEVER, 1.5, 5);
    let left = scorer.partition("m");
    scorer.forget("m", &left);
    assert!(scorer.partitions.load().is_empty());
    let made = scorer.partition("m");
    scorer.forget("m", &left);
    assert!(Arc::ptr_eq(&of(&scorer, "m"), &made));
    scorer.forget("m", &made);
    assert!(scorer.partitions.load().is_empty());
}

/// A decider that took a partition from the table before it left finds
/// it marked behind its guard and takes the model's next one: nobody is
/// listed in a partition the table no longer holds.
#[test]
fn a_decider_that_took_a_partition_before_it_left_takes_the_next() {
    let scorer = Arc::new(scorer(NEVER, 1.5, 5));
    let pool = pool(2);
    // An entry that lists nobody, under the guard of the one that
    // retires it.
    let left = scorer.partition("m");
    let mut held = left.lock();
    let hands = Arc::strong_count(&left);
    let (tell, decided) = mpsc::channel();
    {
        let (scorer, pool) = (Arc::clone(&scorer), pool.clone());
        std::thread::spawn(move || tell.send(led(&scorer, &pool, 0, SECOND)));
    }
    // The decider has the entry once one more hand holds it: it waits
    // for the guard, or is about to ask for it.
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while Arc::strong_count(&left) == hands && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        Arc::strong_count(&left) > hands,
        "the decider took no entry"
    );
    scorer.retire_if_empty("m", &left, &mut held);
    drop(held);
    // A decider that met a marked entry in the table again would ask
    // for ever: its answer is waited for under a deadline.
    let answer = decided.recv_timeout(std::time::Duration::from_secs(5));
    assert_eq!(answer, Ok(Some(0)), "the decider's answer");
    assert!(left.lock().by_id.is_empty(), "listed in an entry that left");
    assert!(!Arc::ptr_eq(&of(&scorer, "m"), &left));
    assert_eq!(loads(&scorer), (vec![0, 0], 0));
}

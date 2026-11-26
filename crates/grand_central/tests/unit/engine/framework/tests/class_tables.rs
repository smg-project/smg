use super::*;

/// A pair decision under the table of `lists` ([`table_of`]).
fn decide_tabled(
    engine: &Engine,
    prefills: &[Candidate],
    decodes: &[Candidate],
    lists: &[&[u32]],
) -> Decision {
    let (ends, classes) = table_of(lists);
    let facts = RequestFacts {
        model: "m",
        ..RequestFacts::default()
    };
    engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills,
            decodes,
            table: PairTable::new(&ends, &classes),
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

#[test]
fn a_class_lists_what_the_table_holds_for_it_and_its_own_number_under_none() {
    let none = PairTable::default();
    assert!(none.is_empty());
    assert_eq!(none.of(&7), [7], "by number");
    assert!(none.of(&0).is_empty(), "class 0 pairs with nothing");

    let (ends, classes) = table_of(&[&[1, 3], &[], &[2]]);
    let table = PairTable::new(&ends, &classes);
    assert!(!table.is_empty());
    assert_eq!(table.of(&1), [1, 3]);
    assert!(table.of(&2).is_empty(), "an empty list");
    assert_eq!(table.of(&3), [2]);
    assert!(table.of(&0).is_empty(), "class 0 under a table");
    assert!(table.of(&4).is_empty(), "a class past the table");
    assert_eq!(table.width(), 4, "one past the largest class named");

    // Ends the lists do not hold cost a class its list.
    let past = PairTable::new(&[3], &[1, 2]);
    assert!(past.of(&1).is_empty());
    let backwards = PairTable::new(&[2, 1], &[1, 2, 3]);
    assert_eq!(backwards.of(&1), [1, 2]);
    assert!(backwards.of(&2).is_empty());
}

/// The width's walk reads each list's last entry, whichever list holds
/// the largest, and the first last entry above the count hands the
/// table to the list by list read. Both name one width: one past the
/// largest class a list's own read holds.
#[test]
fn a_tables_width_is_one_past_the_largest_class_its_lists_hold() {
    let agree = |table: &PairTable<'_>, width: usize, what: &str| {
        assert_eq!(table.width(), width, "{what}");
        assert_eq!(table.width_by_lists(), width, "{what}, list by list");
    };
    let tables: [(&[&[u32]], usize); 10] = [
        // The largest class in the first list, a middle one, the last.
        (&[&[1, 4], &[], &[2, 3]], 5),
        (&[&[1], &[2, 4], &[3]], 5),
        (&[&[1, 2], &[3, 4]], 5),
        // In a list of one entry behind an empty list.
        (&[&[1], &[], &[2]], 3),
        // An empty list names no class: two entries, the width of one.
        (&[&[1], &[1], &[]], 2),
        // A last entry above the count, in the first list and in the
        // last: its list ends at the classes it holds.
        (&[&[1, 9], &[2, 3]], 4),
        (&[&[2, 3], &[1, 9]], 4),
        // One above the count is above it.
        (&[&[1, 3]], 2),
        // No list holds a class, and no list holds an entry.
        (&[&[8], &[9]], 1),
        (&[&[], &[]], 1),
    ];
    for (lists, width) in tables {
        let (ends, classes) = table_of(lists);
        agree(
            &PairTable::new(&ends, &classes),
            width,
            &format!("{lists:?}"),
        );
    }
    // Ends the entries do not hold: an end past them costs its list,
    // and each list is cut by its own two ends, whatever the last did.
    agree(&PairTable::new(&[3], &[1, 2]), 1, "an end past the entries");
    agree(&PairTable::new(&[3, 0, 1], &[2, 1]), 3, "ends out of order");
}

/// The question a host asks once where it builds a table: each clause
/// is refused by its name, the first a table fails.
#[test]
fn a_hosts_validation_names_the_clause_a_table_fails() {
    let validate = |ends: &[u32], classes: &[u32]| PairTable::new(ends, classes).validate();
    assert_eq!(PairTable::default().validate(), Ok(()), "no table");
    assert_eq!(
        validate(&[2, 2, 3], &[1, 3, 2]),
        Ok(()),
        "an empty list too"
    );
    assert_eq!(
        validate(&[2, 1], &[1, 2, 3]),
        Err(PairTableError::EndsDescend)
    );
    assert_eq!(validate(&[3], &[1, 2]), Err(PairTableError::EndPastLists));
    // Ends written as each list's start: every list reads the one before.
    for (ends, classes) in [
        (&[0, 2][..], &[1, 2, 3, 4][..]),
        (&[1], &[1, 2]),
        (&[], &[1]),
    ] {
        assert_eq!(
            validate(ends, classes),
            Err(PairTableError::EntriesPastEnds),
            "{ends:?}"
        );
    }
    for misnamed in [[2, 1], [1, 1]] {
        assert_eq!(
            validate(&[2], &misnamed),
            Err(PairTableError::ListOutOfOrder),
            "{misnamed:?}"
        );
    }
    for zero in [&[0][..], &[0, 2]] {
        let ends = [zero.len() as u32];
        assert_eq!(validate(&ends, zero), Err(PairTableError::ClassZero));
    }
    assert_eq!(
        validate(&[2], &[1, 3]),
        Err(PairTableError::NumberAboveCount)
    );
    // The first clause a table fails is the one named.
    assert_eq!(validate(&[3, 1], &[2, 1]), Err(PairTableError::EndsDescend));
    assert_eq!(validate(&[2], &[0, 0]), Err(PairTableError::ListOutOfOrder));
    assert_eq!(validate(&[2], &[0, 3]), Err(PairTableError::ClassZero));
    // Each clause reads as what it found.
    assert_eq!(
        PairTableError::NumberAboveCount.to_string(),
        "a list names a number above the count of the lists' entries"
    );
}

/// The question a host asks once where it builds first partners
/// beside its table: each clause is refused by its name, the first
/// they fail.
#[test]
fn a_hosts_question_names_the_clause_first_partners_fail() {
    // Two prefill classes that pair with decode classes 1 and 2, and a
    // third with class 3 alone.
    let table = PairTable::new(&[2, 4, 5], &[1, 2, 1, 2, 3]);
    let within = |ends: &[u32], classes: &[u32]| FirstPartners::new(ends, classes).within(&table);
    let lists = |clause| Err(FirstPartnersError::Lists(clause));
    assert_eq!(FirstPartners::default().within(&table), Ok(()), "none");
    assert_eq!(within(&[1, 1], &[1]), Ok(()), "one class's alone");
    assert_eq!(within(&[2, 3, 4], &[1, 2, 2, 3]), Ok(()), "every partner");
    // A list names classes of the table, so a number above the lists'
    // own count stands.
    assert_eq!(within(&[0, 0, 1], &[3]), Ok(()), "above the count");
    assert_eq!(within(&[1, 0], &[1]), lists(PairTableError::EndsDescend));
    assert_eq!(within(&[2], &[1]), lists(PairTableError::EndPastLists));
    assert_eq!(
        within(&[1], &[1, 2]),
        lists(PairTableError::EntriesPastEnds)
    );
    assert_eq!(within(&[2], &[2, 1]), lists(PairTableError::ListOutOfOrder));
    assert_eq!(within(&[1], &[0]), lists(PairTableError::ClassZero));
    // A class of the table that is no partner of the list's prefill
    // class, and a list for a prefill class the table does not hold.
    let no_partner = Err(FirstPartnersError::NoPartner);
    assert_eq!(within(&[1], &[3]), no_partner);
    assert_eq!(within(&[0, 0, 0, 1], &[1]), no_partner);
    // Under no table a class pairs with its own number.
    let none = PairTable::default();
    assert_eq!(FirstPartners::new(&[1], &[1]).within(&none), Ok(()));
    assert_eq!(FirstPartners::new(&[1], &[2]).within(&none), no_partner);
    assert_eq!(
        FirstPartnersError::NoPartner.to_string(),
        "a first partner is no partner of its prefill class in the pair table"
    );
}

/// One invariant: a host's table costs a decision at most the table's
/// own size.
#[test]
fn a_number_above_the_count_of_a_tables_entries_is_no_class_of_the_table() {
    // Two entries: the count itself is a class, as the largest number
    // of a dense table is.
    let dense = PairTable::new(&[2], &[1, 2]);
    assert_eq!(dense.of(&1), [1, 2]);
    assert_eq!(dense.validate(), Ok(()));
    // One above the count is no class.
    let above = PairTable::new(&[2], &[1, 3]);
    assert_eq!(above.of(&1), [1]);
    assert_eq!(above.validate(), Err(PairTableError::NumberAboveCount));

    // Three entries, two of them near the top of the number's range.
    let far = u32::MAX - 1;
    let entries = [1, far, u32::MAX];
    let sparse = PairTable::new(&[3], &entries);
    assert_eq!(sparse.of(&1), [1], "every read asks here");
    assert_eq!(sparse.validate(), Err(PairTableError::NumberAboveCount));
    assert_eq!(sparse.width(), 2);
    // The partner question: a decode of such a number opens nothing,
    // and the set of bits stays the table's size.
    let mut scratch = Scratch::new();
    let (classes, marks) = (&mut scratch.classes, &mut scratch.marks);
    sparse.partnered([far].into_iter(), classes, marks);
    assert!(classes.is_empty(), "{classes:?}");
    sparse.partnered([far, 1].into_iter(), classes, marks);
    assert_eq!(*classes, [1]);
    // The shed's list: the routable class lists the one class it holds.
    sparse.listed(classes, marks);
    assert_eq!(*classes, [1]);
    assert_eq!(marks.words.len(), 1, "one word of bits");

    // A list of such numbers alone is an empty list.
    let lone = PairTable::new(&[1], &[u32::MAX]);
    assert!(lone.of(&1).is_empty());
    assert_eq!(lone.width(), 1);

    // The cap itself, asked apart: no table's read hands it a number
    // above the count.
    assert_eq!(PairTable::capped(3, 3), 4, "the count is a class");
    assert_eq!(PairTable::capped(far, 3), 4);
    assert_eq!(PairTable::capped(u32::MAX, 0), 1);

    // A list out of order is cut wherever the search lands: the set
    // stays the table's size all the same.
    for unordered in [[u32::MAX, 1, far], [far, u32::MAX, 1], [2, u32::MAX, far]] {
        let table = PairTable::new(&[3], &unordered);
        assert!(table.width() <= 4, "{unordered:?}: {}", table.width());
        assert_eq!(table.validate(), Err(PairTableError::ListOutOfOrder));
    }
}

/// Class 0 is no class of the table, whatever a list names: the row
/// tests pass it over, and the validation names the clause.
#[test]
fn a_listed_class_zero_is_no_class_of_the_table() {
    // Alone it is a list of no class: no pass compares a row with it.
    let alone = PairTable::new(&[1], &[0]);
    assert_eq!(alone.of(&1), [0], "a read answers the list as it stands");
    assert_eq!(only(alone.of(&1)), None);
    assert!(!classed(alone.of(&1), 0));
    assert_eq!(alone.validate(), Err(PairTableError::ClassZero));
    // Beside a class it adds no row to that class's.
    let beside = PairTable::new(&[2], &[0, 2]);
    assert_eq!(beside.of(&1), [0, 2]);
    assert_eq!(only(beside.of(&1)), None);
    assert!(classed(beside.of(&1), 2));
    assert!(!classed(beside.of(&1), 0));
    assert_eq!(beside.validate(), Err(PairTableError::ClassZero));
    // The partner question: an open decode class finds the prefill
    // class through the class beside, never through the 0.
    let mut scratch = Scratch::new();
    let (classes, marks) = (&mut scratch.classes, &mut scratch.marks);
    alone.partnered([2].into_iter(), classes, marks);
    assert!(classes.is_empty(), "{classes:?}");
    beside.partnered([2].into_iter(), classes, marks);
    assert_eq!(*classes, [1]);
    // The shed's list holds the 0 a list names, and a decode of class 0
    // is none of the decodes that name the shed.
    beside.listed(classes, marks);
    assert_eq!(*classes, [0, 2]);
    assert!(classed(classes, 2));
    assert!(!classed(classes, 0));
}

#[test]
fn a_set_of_classes_holds_the_numbers_it_was_sized_for() {
    let mut set = ClassBits::default();
    set.reset(130);
    for class in [0, 1, 63, 64, 65, 129] {
        assert!(set.insert(class), "{class}");
    }
    assert!(!set.insert(192), "a number past the set's last word");
    assert!([0, 1, 63, 64, 65, 129].iter().all(|&class| set.has(class)));
    assert!(![2, 62, 66, 128, 192, u32::MAX]
        .iter()
        .any(|&class| set.has(class)));
    assert_eq!(set.classes().collect::<Vec<_>>(), [0, 1, 63, 64, 65, 129]);
    // A reset empties it, whatever its new width.
    set.reset(1);
    assert_eq!(set.classes().count(), 0);
    assert!(!set.has(1));
}

#[test]
fn a_prefill_pairs_with_the_decodes_of_every_class_its_class_lists() {
    let engine = Engine::new(
        vec![least_requests_profile()],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    // Decode class 3 pairs with both prefill classes and stands once;
    // the last decode is outside any pairing.
    let mut prefills = classed_rows(&[1, 2]);
    let decodes = classed_rows(&[1, 2, 3, 0]);
    let lists: [&[u32]; 2] = [&[1, 3], &[2, 3]];
    prefills[1].in_flight = 5;
    let first = decide_tabled(&engine, &prefills, &decodes, &lists);
    assert_eq!(lead(&first), Some(0));
    assert_eq!(decodes_of(&first), [0, 2], "class 1's decodes: {first:?}");
    prefills[0].in_flight = 9;
    let second = decide_tabled(&engine, &prefills, &decodes, &lists);
    assert_eq!(lead(&second), Some(1));
    assert_eq!(decodes_of(&second), [1, 2], "class 2's decodes: {second:?}");
}

#[test]
fn the_partner_pass_ends_at_the_decode_that_opens_the_tables_last_class() {
    use std::sync::atomic::{AtomicU32, Ordering};

    // The pass alone: the third decode opens the last class, and the
    // two behind it are not read.
    let lists: [&[u32]; 2] = [&[1], &[2]];
    let (ends, listed) = table_of(&lists);
    let mut scratch = Scratch::new();
    let mut open = [1, 1, 2, 2, 1].into_iter();
    PairTable::new(&ends, &listed).partnered(
        open.by_ref(),
        &mut scratch.classes,
        &mut scratch.marks,
    );
    assert_eq!(scratch.classes, [1, 2]);
    assert_eq!(open.as_slice(), [2, 1]);

    // A filter the engine does not hold itself is asked about every
    // decode all the same: five asks in the pass, then three in the
    // decode run of the prefill's class.
    static ASKS: AtomicU32 = AtomicU32::new(0);
    #[derive(Debug)]
    struct Asked;
    impl Filter for Asked {
        fn name(&self) -> &'static str {
            "asked"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            ASKS.fetch_add(1, Ordering::Relaxed);
            true
        }
    }
    let engine = Engine::new(
        vec![
            least_requests_profile(),
            Profile {
                filters: vec![Box::new(Asked)],
                ..least_requests_profile()
            },
        ],
        vec![Binding {
            decode_profile: Some(1),
            ..pair_binding()
        }],
        1,
        Needs::default(),
    );
    let mut prefills = classed_rows(&[1, 2]);
    prefills[1].in_flight = 5;
    let decodes = classed_rows(&[1, 1, 2, 2, 1]);
    let decision = decide_tabled(&engine, &prefills, &decodes, &lists);
    assert_eq!(decodes_of(&decision), [0, 1, 4]);
    assert_eq!(ASKS.load(Ordering::Relaxed), 5 + 3);
}

#[test]
fn a_table_says_which_classes_pair_where_none_leaves_it_to_the_numbers() {
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1]);
    // The second decode's class stands at the edge of a word of bits.
    let decodes = classed_rows(&[1, 64]);
    let by_number = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(decodes_of(&by_number), [0], "{by_number:?}");
    // A second prefill class names the 63 classes under it: the table
    // numbers densely.
    let under: Vec<u32> = (1..=63).collect();
    let by_table = decide_tabled(&engine, &prefills, &decodes, &[&[64], &under]);
    assert_eq!(decodes_of(&by_table), [1], "{by_table:?}");
    // A class the table lists nothing for has no partner, whatever its
    // number; nor has a class past the table.
    let unpartnered = Some((ShedReason::NoEndpoints, "partnered"));
    assert_eq!(
        shed_of(&decide_tabled(&engine, &prefills, &decodes, &[&[]])),
        unpartnered
    );
    let past = classed_rows(&[2]);
    assert_eq!(
        shed_of(&decide_tabled(&engine, &past, &decodes, &[&[1]])),
        unpartnered
    );
}

#[test]
fn a_prefill_class_has_a_partner_when_any_class_it_lists_holds_an_open_decode() {
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1, 2]);
    let mut decodes = classed_rows(&[1, 2, 3]);
    let lists: [&[u32]; 2] = [&[1, 2], &[3]];
    // Class 1's first listed class is down and its second is up; class
    // 2's one class is down.
    decodes[0].health.healthy = false;
    decodes[2].health.healthy = false;
    let decision = decide_tabled(&engine, &prefills, &decodes, &lists);
    assert_eq!(lead(&decision), Some(0), "{decision:?}");
    assert_eq!(decodes_of(&decision), [1]);
    // The other way round: class 2 alone has a partner.
    decodes[1].health.healthy = false;
    decodes[2].health.healthy = true;
    let decision = decide_tabled(&engine, &prefills, &decodes, &lists);
    assert_eq!(lead(&decision), Some(1), "{decision:?}");
    assert_eq!(decodes_of(&decision), [2]);
}

#[test]
fn under_a_table_a_shed_is_named_by_the_decodes_a_routable_prefill_lists() {
    let engine = strict_pair_engine();
    // A routable prefill of class 1 lists class 2, whose decode is
    // unhealthy; an unhealthy prefill of class 2 lists that class and
    // class 3, whose decode is overloaded; a healthy decode of class 1,
    // the routable prefill's own number, is listed by neither.
    let mut prefills = classed_rows(&[1, 2]);
    prefills[1].health.healthy = false;
    let mut decodes = classed_rows(&[2, 3, 1]);
    decodes[0].health.healthy = false;
    decodes[1].health.overloaded = true;
    let lists: [&[u32]; 2] = [&[2], &[2, 3]];
    assert_eq!(
        shed_of(&decide_tabled(&engine, &prefills, &decodes, &lists)),
        Some((ShedReason::NoEndpoints, "healthy")),
        "the routable prefill's partner names the shed"
    );
    // Both prefills routable: the stage that drops the last of their
    // partners names it, and that one is saturation.
    prefills[1].health.healthy = true;
    assert_eq!(
        shed_of(&decide_tabled(&engine, &prefills, &decodes, &lists)),
        Some((ShedReason::Saturated, "not-overloaded"))
    );
    // By number the routable prefill's partner is the decode of its own
    // class, whatever the number: an overloaded one is saturation under
    // a binding that sheds.
    let mut engine = engine;
    engine.shed_when_all_vetoed(0);
    let prefills = classed_rows(&[100]);
    let mut decodes = classed_rows(&[100]);
    decodes[0].health.overloaded = true;
    assert_eq!(
        shed_of(&decide_pair(&engine, &prefills, &decodes)),
        Some((ShedReason::Saturated, "not-overloaded"))
    );
}

#[test]
fn by_number_a_class_is_found_wherever_its_decode_stands_in_the_pool() {
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1]);
    // The classes of the open decodes arrive out of order.
    let decodes = classed_rows(&[2, 3, 1]);
    let decision = decide_pair(&engine, &prefills, &decodes);
    assert_eq!(decodes_of(&decision), [2], "{decision:?}");
}

#[test]
fn a_scratch_carries_no_class_from_one_pair_decision_into_the_next() {
    let engine = Engine::new(
        vec![Profile {
            name: "first".into(),
            filters: vec![Box::new(Healthy), Box::new(NotOverloaded)],
            scorers: vec![],
            picker: Box::new(FirstKept),
        }],
        vec![pair_binding()],
        1,
        Needs::default(),
    );
    let prefills = classed_rows(&[1, 2]);
    let mut decodes = classed_rows(&[1, 2]);
    let (ends, classes) = table_of(&[&[1], &[2]]);
    for table in [PairTable::default(), PairTable::new(&ends, &classes)] {
        let mut scratch = Scratch::new();
        let mut decide = |decodes: &[Candidate]| {
            engine.decide_pair(
                &RequestFacts {
                    model: "m",
                    ..RequestFacts::default()
                },
                &PoolFacts::default(),
                PairPools {
                    prefills: &prefills,
                    decodes,
                    table,
                    ..PairPools::default()
                },
                &Constraints::default(),
                &Constraints::default(),
                &mut scratch,
            )
        };
        decodes[0].health.healthy = true;
        assert_eq!(lead(&decide(&decodes)), Some(0), "each class has a partner");
        // The first class's decode goes down: the second prefill alone
        // has a partner, and the picker hands no alternate.
        decodes[0].health.healthy = false;
        let next = decide(&decodes);
        assert_eq!(lead(&next), Some(1), "{next:?}");
    }
}

/// Class 0 pairs with nothing at the decode run too: a picker that hands
/// a class-0 prefill the filters dropped gets no decode for it, under no
/// table as under one.
#[test]
fn a_class_zero_prefill_pairs_with_no_decode_whoever_hands_it() {
    /// Hands the pool's first row, whichever rows the filters kept.
    #[derive(Debug)]
    struct FirstRow;
    impl Lifecycle for FirstRow {}
    impl Picker for FirstRow {
        fn name(&self) -> &'static str {
            "first-row"
        }
        fn pick(
            &self,
            _: &RequestFacts<'_>,
            candidates: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            out: &mut Targets,
        ) {
            if let Some(first) = candidates.first() {
                out.push(target_of(first));
            }
        }
    }
    let engine = Engine::new(
        vec![
            Profile {
                name: "prefill".into(),
                filters: vec![Box::new(Healthy)],
                scorers: vec![],
                picker: Box::new(FirstRow),
            },
            strict_profile(),
        ],
        vec![Binding {
            decode_profile: Some(1),
            ..pair_binding()
        }],
        1,
        Needs::default(),
    );
    // The second prefill pairs with the second decode, so the prefill
    // pool keeps a row; the picker hands the first, of class 0, beside a
    // decode of class 0.
    let prefills = classed_rows(&[0, 1]);
    let decodes = classed_rows(&[0, 1]);
    let lists: [&[u32]; 1] = [&[1]];
    for (decision, how) in [
        (decide_pair(&engine, &prefills, &decodes), "by number"),
        (
            decide_tabled(&engine, &prefills, &decodes, &lists),
            "by a table",
        ),
    ] {
        assert_eq!(
            shed_of(&decision),
            Some((ShedReason::NoEndpoints, "partners")),
            "{how}: {decision:?}"
        );
    }
}

/// A list that names class 0 is not valid: a debug build stops at the
/// clause, and a release build pairs no decode of class 0 through it.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "the pair table is not valid: a list names class 0")
)]
fn a_listed_class_zero_stops_a_debug_build_and_pairs_nothing_in_a_release_build() {
    // Alone it is no partner, and beside a class it adds no decode to
    // that class's.
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1]);
    let decodes = classed_rows(&[0, 2]);
    assert_eq!(
        shed_of(&decide_tabled(&engine, &prefills, &decodes, &[&[0]])),
        Some((ShedReason::NoEndpoints, "partnered"))
    );
    let beside = decide_tabled(&engine, &prefills, &decodes, &[&[0, 2]]);
    assert_eq!(decodes_of(&beside), [1], "{beside:?}");
    // Nor is a decode of class 0 a partner the decode profile lost: an
    // unhealthy one beside that list names no stage of the shed.
    let mut down = classed_rows(&[0, 2]);
    down[0].health.healthy = false;
    assert_eq!(
        shed_of(&decide_tabled(&engine, &prefills, &down, &[&[0]])),
        Some((ShedReason::NoEndpoints, "partnered"))
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(
        expected = "the pair table is not valid: a list names its classes out of ascending order"
    )
)]
fn a_misnamed_table_stops_a_debug_build_and_costs_a_release_build_its_pairs() {
    let engine = strict_pair_engine();
    let prefills = classed_rows(&[1]);
    // A list out of order: the process stands, whatever the list pairs.
    let decodes = classed_rows(&[1, 2]);
    let decision = decide_tabled(&engine, &prefills, &decodes, &[&[2, 1]]);
    assert!(decision.pick().is_some() || shed_of(&decision).is_some());
}

/// One invariant, over drawn fleets: a table over one row a decode
/// decides as the same fleet spelled without one, a decode row for each
/// prefill class it pairs with: the same shed by reason and stage, the
/// same prefills and the same decodes in the same order.
#[test]
fn a_table_decides_as_the_same_fleet_spelled_a_decode_row_a_prefill_class() {
    fn engine_of(picker: Box<dyn Picker>) -> Engine {
        let profile = Profile {
            name: "drawn".into(),
            filters: vec![
                Box::new(Healthy),
                Box::new(NotOverloaded),
                Box::new(NotTried),
            ],
            scorers: vec![WeightedScorer {
                scorer: Box::new(LeastRequests),
                weight: 1.0,
            }],
            picker,
        };
        Engine::new(vec![profile], vec![pair_binding()], 1, Needs::default())
    }
    /// A row whose class runs from 0 (outside any pairing) to one past
    /// its side's last class (a class the table does not hold).
    fn row(draw: &mut Draw, position: u32, classes: u32) -> Candidate {
        Candidate {
            pair_class: draw.below(classes as usize + 2) as u32,
            health: Health {
                healthy: draw.below(5) != 0,
                overloaded: draw.below(4) == 0,
                ..Health::default()
            },
            ..candidate(position, draw.below(6) as u32)
        }
    }
    /// What a decision says, by worker: a shed's reason and stage, or
    /// each target's prefill and decode; and the survivors it counted.
    type Said = (
        Option<(ShedReason, &'static str)>,
        Vec<(WorkerId, Option<WorkerId>)>,
        Survivors,
    );
    fn said(decision: &Decision) -> Said {
        match decision {
            Decision::Pick(pick) => (
                None,
                pick.targets
                    .iter()
                    .map(|target| (target.id, target.pair.map(|pair| pair.decode_id)))
                    .collect(),
                pick.trace.survivors,
            ),
            Decision::Shed(shed) => (
                Some((shed.reason, shed.emptied_by)),
                Vec::new(),
                shed.trace.survivors,
            ),
            Decision::Queue(_) => panic!("no profile here queues"),
        }
    }
    let engines = [
        engine_of(Box::new(Best)),
        engine_of(Box::new(PowerOfTwo)),
        engine_of(Box::new(CountFirst)),
    ];
    let mut draw = Draw::new(0x7ab1e);
    let (mut picks, mut sheds) = (0u32, 0u32);
    for fleet in 0..600u64 {
        let prefill_classes = 1 + draw.below(4) as u32;
        let decode_classes = 1 + draw.below(5) as u32;
        let mut lists: Vec<Vec<u32>> = (0..prefill_classes)
            .map(|_| {
                (1..=decode_classes)
                    .filter(|_| draw.below(2) == 0)
                    .collect()
            })
            .collect();
        // A table numbers densely: a class no list drew joins one.
        for class in 1..=decode_classes {
            if !lists.iter().any(|list| list.contains(&class)) {
                let list = &mut lists[draw.below(prefill_classes as usize)];
                list.push(class);
                list.sort_unstable();
            }
        }
        let prefills: Vec<Candidate> = (0..draw.below(6) as u32)
            .map(|position| row(&mut draw, position, prefill_classes))
            .collect();
        let decodes: Vec<Candidate> = (0..draw.below(9) as u32)
            .map(|position| row(&mut draw, position, decode_classes))
            .collect();
        // The same fleet without a table: a decode row for each prefill
        // class that lists its class, of that class's number; a decode
        // no class lists keeps one row, outside any pairing.
        let mut rows: Vec<Candidate> = Vec::new();
        for decode in &decodes {
            let before = rows.len();
            rows.extend(
                lists
                    .iter()
                    .zip(1u32..)
                    .filter(|(list, _)| decode.pair_class != 0 && list.contains(&decode.pair_class))
                    .map(|(_, pair_class)| Candidate {
                        pair_class,
                        ..*decode
                    }),
            );
            if rows.len() == before {
                rows.push(Candidate {
                    pair_class: 0,
                    ..*decode
                });
            }
        }
        for (row, position) in rows.iter_mut().zip(0u32..) {
            row.position = position;
        }
        let lists: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
        let (ends, classes) = table_of(&lists);
        // Some workers were just tried: the soft filter steps in.
        let excluded: Vec<WorkerId> = (0..draw.below(3) as u64)
            .map(|_| WorkerId(100 + draw.below(9) as u64))
            .collect();
        let cx = Constraints {
            excluded: &excluded,
            ..Constraints::default()
        };
        for (engine, seed) in engines.iter().zip(fleet..) {
            let facts = RequestFacts {
                model: "m",
                seed,
                ..RequestFacts::default()
            };
            let decide = |pools: PairPools<'_>| {
                engine.decide_pair(
                    &facts,
                    &PoolFacts::default(),
                    pools,
                    &cx,
                    &cx,
                    &mut Scratch::new(),
                )
            };
            let tabled = decide(PairPools {
                prefills: &prefills,
                decodes: &decodes,
                table: PairTable::new(&ends, &classes),
                ..PairPools::default()
            });
            let spelled = decide(PairPools {
                prefills: &prefills,
                decodes: &rows,
                ..PairPools::default()
            });
            assert_eq!(
                said(&tabled),
                said(&spelled),
                "fleet {fleet}, seed {seed}: lists {lists:?} over {prefills:?} and {decodes:?}"
            );
            match tabled {
                Decision::Pick(_) => picks += 1,
                _ => sheds += 1,
            }
        }
    }
    assert!(picks >= 300 && sheds >= 300, "{picks} picks, {sheds} sheds");
}

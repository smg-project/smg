use super::Counters;

#[test]
fn a_declared_id_adds_and_an_undeclared_one_is_counted_apart() {
    let counters = Counters::new(vec!["hits".into(), "misses".into()]);
    counters.add(0, 3);
    counters.add(1, 1);
    counters.add(1, 1);
    counters.add(7, 5);
    counters.add(u32::MAX, 5);
    let read: Vec<(&str, u64)> = counters.read().collect();
    assert_eq!(read, vec![("hits", 3), ("misses", 2)]);
    assert_eq!(counters.undeclared(), 2);
}

#[test]
fn a_declared_counter_saturates_at_the_maximum() {
    let counters = Counters::new(vec!["hits".into()]);
    counters.add(0, u64::MAX);
    counters.add(0, 1);

    assert_eq!(
        counters.read().collect::<Vec<_>>(),
        vec![("hits", u64::MAX)]
    );
}

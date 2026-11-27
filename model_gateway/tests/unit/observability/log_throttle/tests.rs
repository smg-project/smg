use super::*;

const WINDOW: Duration = Duration::from_secs(1);

/// A second of refusals writes the configured lines and no more; the
/// first line of the next window carries the count of what was hidden,
/// once.
#[test]
fn a_burst_in_one_window_emits_the_limit_and_the_next_line_carries_the_suppressed_count() {
    let throttle = LogThrottle::new(5, WINDOW);
    let t0 = Instant::now();

    let admitted = (0..1000u64)
        .filter(|i| {
            throttle
                .admit_at(t0 + Duration::from_micros(i * 500))
                .is_some()
        })
        .count();
    assert_eq!(admitted, 5, "the window admits exactly its limit");

    let first_of_next = throttle
        .admit_at(t0 + WINDOW)
        .expect("the next window admits again");
    assert_eq!(
        first_of_next.suppressed, 995,
        "the summary names every line the burst hid"
    );
    let second_of_next = throttle
        .admit_at(t0 + WINDOW + Duration::from_millis(1))
        .expect("still inside the new window's allowance");
    assert_eq!(second_of_next.suppressed, 0, "the summary is reported once");
}

/// Under the limit nothing is held back and no summary appears.
#[test]
fn lines_under_the_limit_are_all_admitted_with_nothing_suppressed() {
    let throttle = LogThrottle::new(5, WINDOW);
    let t0 = Instant::now();
    for window in 0..3u32 {
        for line in 0..3u32 {
            let at = t0 + WINDOW * window + Duration::from_millis(u64::from(line) * 100);
            assert_eq!(throttle.admit_at(at), Some(Admitted { suppressed: 0 }));
        }
    }
}

/// A window that fills, then silence: the next call, however much later,
/// opens a fresh window with the whole allowance and reports the backlog.
#[test]
fn a_quiet_gap_starts_a_fresh_window_with_the_full_allowance() {
    let throttle = LogThrottle::new(2, WINDOW);
    let t0 = Instant::now();
    assert!(throttle.admit_at(t0).is_some());
    assert!(throttle.admit_at(t0).is_some());
    assert!(throttle.admit_at(t0).is_none());
    assert!(throttle.admit_at(t0 + Duration::from_millis(999)).is_none());

    let later = t0 + Duration::from_secs(10);
    assert_eq!(throttle.admit_at(later), Some(Admitted { suppressed: 2 }));
    assert_eq!(throttle.admit_at(later), Some(Admitted { suppressed: 0 }));
    assert!(throttle.admit_at(later).is_none());
}

/// A clock that reads before the window's start is treated as inside it.
#[test]
fn an_earlier_clock_reading_stays_in_the_current_window() {
    let throttle = LogThrottle::new(1, WINDOW);
    let t0 = Instant::now() + Duration::from_secs(5);
    assert!(throttle.admit_at(t0).is_some());
    assert!(throttle.admit_at(t0 - Duration::from_secs(1)).is_none());
}

/// Two models refused in the same second each get the whole allowance,
/// and each one's summary counts only its own hidden lines.
#[test]
fn each_key_has_its_own_allowance_and_its_own_suppressed_count() {
    let throttle = KeyedLogThrottle::new(2, WINDOW, 64);
    let t0 = Instant::now();
    for _ in 0..10 {
        throttle.admit_at("a", t0);
    }
    assert_eq!(
        throttle.admit_at("b", t0),
        Some(Admitted { suppressed: 0 }),
        "a's storm leaves b's first line untouched"
    );
    assert!(throttle.admit_at("b", t0).is_some());
    assert!(throttle.admit_at("b", t0).is_none());

    let next = t0 + WINDOW;
    assert_eq!(
        throttle.admit_at("a", next),
        Some(Admitted { suppressed: 8 })
    );
    assert_eq!(
        throttle.admit_at("b", next),
        Some(Admitted { suppressed: 1 })
    );
}

/// Past the key bound the key seen least recently goes, and comes back
/// with a fresh window when it returns.
#[test]
fn the_key_set_is_bounded_and_evicts_the_least_recently_seen_key() {
    let throttle = KeyedLogThrottle::new(1, WINDOW, 2);
    let t0 = Instant::now();
    assert!(throttle.admit_at("a", t0).is_some());
    assert!(throttle.admit_at("b", t0).is_some());
    assert!(throttle.admit_at("a", t0).is_none(), "a's window is full");
    // a was used after b, so c evicts b.
    assert!(throttle.admit_at("c", t0).is_some());
    assert!(
        throttle.admit_at("b", t0).is_some(),
        "b returns as a new key with a fresh allowance"
    );
    assert!(throttle.admit_at("c", t0).is_none(), "c's window is full");
    assert!(
        throttle.admit_at("a", t0 + WINDOW).is_some(),
        "a was evicted by b's return and starts afresh"
    );
    assert_eq!(
        throttle.admit_at("a", t0 + WINDOW),
        None,
        "a fresh key still has one line per window"
    );
}

use super::*;

/// Concurrent intervals count once and capacity expires independently per rank.
#[test]
fn prefill_capacity_unions_concurrency_preserves_idle_and_expires_per_rank() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    tracker.prefill_computed_at(0, 1_000, 10.0, 12.0, now);
    assert_eq!(tracker.estimate_at(0, 0, now).prefill_throughput, None);
    tracker.prefill_computed_at(0, 2_000, 11.0, 13.0, now);
    // 3000 computed tokens / 3 active seconds, not 4 summed request seconds.
    assert_eq!(
        tracker.estimate_at(0, 0, now).prefill_throughput,
        Some(1_000.0)
    );
    tracker.generated_at(0, 1, now);
    assert_eq!(tracker.estimate_at(0, 0, now).gen_throughput, 0.5);
    assert_eq!(
        tracker
            .estimate_at(0, 0, now + Duration::from_secs(30))
            .prefill_throughput,
        Some(1_000.0)
    );
    assert_eq!(tracker.estimate_at(1, 0, now).prefill_throughput, None);
    assert_eq!(
        tracker
            .estimate_at(0, 0, now + Duration::from_secs(61))
            .prefill_throughput,
        None
    );
}

/// Invalid timings and cache-only work cannot train the capacity estimate.
#[test]
fn prefill_capacity_rejects_cache_hits_and_invalid_samples() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    for (tokens, start, end) in [
        (0, 1.0, 2.0),
        (100, 2.0, 1.0),
        (100, 1.0, 1.0),
        (100, f64::NAN, 2.0),
        (100, 1.0, f64::INFINITY),
    ] {
        tracker.prefill_computed_at(0, tokens, start, end, now);
    }
    assert_eq!(tracker.estimate_at(0, 0, now).prefill_throughput, None);
    tracker.prefill_computed_at(0, 1_000, 1.0, 2.0, now);
    tracker.prefill_computed_at(0, 1_000, 100.0, 101.0, now);
    // Idle engine time between requests is excluded.
    assert_eq!(
        tracker.estimate_at(0, 0, now).prefill_throughput,
        Some(1_000.0)
    );
}

#[test]
fn queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    assert_eq!(tracker.estimate_at(0, 5, now), LoadEstimate::default());

    // Three in flight, none started: the engine says two are waiting, so
    // the oldest is in prefill and the two youngest are queued work.
    tracker.submitted(0, &Arc::from("a"), 1_000);
    tracker.submitted(0, &Arc::from("b"), 300);
    tracker.submitted(0, &Arc::from("c"), 200);
    assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 500);
    assert_eq!(tracker.estimate_at(0, 0, now).queued_token_work, 0);
    assert_eq!(
        tracker.estimate_at(0, 10, now).queued_token_work,
        1_500,
        "never more than we hold"
    );

    // The first output of `a`: half its prompt was cached. Queued prompts
    // are discounted by that rate.
    tracker.first_output(0, "a", 1_000, 500);
    assert_eq!(tracker.pending(), 2);
    let estimate = tracker.estimate_at(0, 2, now);
    assert_eq!(estimate.cache_hit_rate, 0.5);
    assert_eq!(estimate.queued_token_work, 250);

    // A request that ends before starting leaves the queue.
    tracker.finished("b");
    assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 100);
    tracker.first_output(0, "c", 200, 200);
    assert_eq!(tracker.pending(), 0);
    assert_eq!(tracker.estimate_at(0, 3, now).queued_token_work, 0);
    // 700 cached of 1_200 prompt tokens.
    assert!((tracker.estimate_at(0, 0, now).cache_hit_rate - 700.0 / 1_200.0).abs() < 1e-9);
}

#[test]
fn throughput_counts_tokens_inside_the_window_only() {
    let tracker = LoadTracker::default();
    let start = Instant::now();
    tracker.generated_at(0, 1_000, start);
    tracker.generated_at(0, 3_000, start + Duration::from_secs(1));
    let per_second = |estimate: LoadEstimate| estimate.gen_throughput;
    assert_eq!(
        per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(1))),
        2_000.0
    );
    // Two seconds after the first sample it falls out of the window.
    assert_eq!(
        per_second(tracker.estimate_at(0, 0, start + Duration::from_millis(2_500))),
        1_500.0
    );
    assert_eq!(
        per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
        0.0
    );
    tracker.generated_at(0, 0, start + Duration::from_secs(4));
    assert_eq!(
        per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
        0.0
    );
}

#[test]
fn the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    tracker.submitted(0, &Arc::from("x"), 0);
    tracker.first_output(0, "x", 0, 0);
    assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
    for index in 0..HIT_RATE_SAMPLES {
        tracker.first_output(0, &format!("old-{index}"), 100, 0);
    }
    assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
    for index in 0..HIT_RATE_SAMPLES {
        tracker.first_output(0, &format!("new-{index}"), 100, 100);
    }
    assert_eq!(
        tracker.estimate_at(0, 0, now).cache_hit_rate,
        1.0,
        "the old samples aged out"
    );
    // Cached can never exceed the prompt.
    tracker.first_output(0, "odd", 10, 50);
    assert!(tracker.estimate_at(0, 0, now).cache_hit_rate <= 1.0);
}

/// One recorded sequence of prefills reads the same capacity whether its
/// ends come from an engine's clock or from the servicer's instants (the
/// wires without engine timing): the two paths share the estimator.
#[test]
fn servicer_clock_samples_read_the_same_capacity_as_engine_clock_ones() {
    let recorded = [
        (1_000_u32, 10.0_f64, 12.0_f64),
        (2_000, 11.0, 13.0),
        (600, 20.0, 20.5),
    ];
    let engine_timed = LoadTracker::default();
    let servicer_timed = LoadTracker::default();
    let now = Instant::now();
    let origin = servicer_timed.epoch;
    for (tokens, start, end) in recorded {
        engine_timed.prefill_computed_at(0, tokens, start, end, now);
        servicer_timed.prefill_observed(
            0,
            tokens,
            origin + Duration::from_secs_f64(start),
            origin + Duration::from_secs_f64(end),
        );
    }
    // 3600 computed tokens over 3 + 0.5 active seconds, the overlap of
    // the first two prefills counted once.
    let expected = 3_600.0 / 3.5;
    assert_eq!(
        engine_timed.estimate_at(0, 0, now).prefill_throughput,
        Some(expected)
    );
    assert_eq!(
        servicer_timed
            .estimate_at(0, 0, origin + Duration::from_secs(21))
            .prefill_throughput,
        Some(expected)
    );
    assert_eq!(servicer_timed.prefill_samples(0), 3);
}

/// A servicer-clock sample is taken once per request, from an armed
/// probe only, and its tokens are the prompt less what was cached.
#[test]
fn a_probe_samples_the_computed_tokens_once_when_armed() {
    let tracker = LoadTracker::default();
    // On the tracker's epoch with whole-second offsets, so the seconds
    // the probe reads are exact and the quotients below compare exactly.
    let start = tracker.epoch;
    let second = Duration::from_secs(1);
    let mut unarmed = PrefillProbe::default();
    unarmed.first_output(&tracker, 0, 1_000, 0, start + second);
    assert_eq!(tracker.prefill_samples(0), 0);
    let mut armed = PrefillProbe::armed(start);
    armed.first_output(&tracker, 0, 1_000, 400, start + second);
    // A later output of the same request adds nothing.
    armed.first_output(&tracker, 0, 1_000, 400, start + 2 * second);
    assert_eq!(tracker.prefill_samples(0), 1);
    let mut later = PrefillProbe::armed(start + 10 * second);
    later.first_output(&tracker, 0, 300, 0, start + 11 * second);
    // 600 + 300 computed tokens over two one-second prefills.
    assert_eq!(
        tracker
            .estimate_at(0, 0, start + 11 * second)
            .prefill_throughput,
        Some(450.0)
    );
    // A prompt served whole from the cache computes nothing.
    let mut hit = PrefillProbe::armed(start + 20 * second);
    hit.first_output(&tracker, 0, 500, 500, start + 21 * second);
    assert_eq!(tracker.prefill_samples(0), 2);
}

/// Every submitted request carries an armed probe, queue or no queue:
/// eight requests handed over together and answered together after one batch
/// read the batch's tokens over its seconds once, and a request answered
/// behind another adds its tokens while its wait, spent on the one ahead
/// of it, is counted once.
#[test]
fn overlapping_prefills_count_their_seconds_once() {
    let loads = LoadTracker::default();
    // On the tracker's epoch with whole-second offsets: exact seconds,
    // exact quotients (an instant read from the clock lands microseconds
    // past the epoch and a measured quotient is off by an ulp).
    let now = loads.epoch;
    let second = Duration::from_secs(1);
    let submit = |request_id: &Arc<str>, at: Instant| {
        loads.submit(Submission {
            request_id,
            rank: 0,
            prompt_tokens: 1_000,
            at,
        })
    };
    // A batch of eight, 1_000 computed tokens each, answered together
    // two seconds after the submit: 8_000 tokens over two seconds.
    let ids: Vec<Arc<str>> = (0..8).map(|index| Arc::from(format!("b{index}"))).collect();
    let mut batch: Vec<ChoiceLoad> = ids
        .iter()
        .map(|id| ChoiceLoad::new(0, 0, submit(id, now)))
        .collect();
    assert_eq!(loads.pending(), 8);
    for (id, choice) in ids.iter().zip(batch.iter_mut()) {
        choice.count_prompt(&loads, id, 1_000, 0, now + 2 * second);
    }
    assert_eq!((loads.pending(), loads.prefill_samples(0)), (0, 8));
    assert_eq!(
        loads.estimate_at(0, 0, now + 2 * second).prefill_throughput,
        Some(4_000.0)
    );
    // Two handed over together and answered one after the other, a
    // second each: the second one's wait was the first one's prefill.
    let later = now + 10 * second;
    let mut ahead = ChoiceLoad::new(0, 0, submit(&Arc::from("ahead"), later));
    let mut behind = ChoiceLoad::new(0, 0, submit(&Arc::from("behind"), later));
    ahead.count_prompt(&loads, "ahead", 1_000, 0, later + second);
    behind.count_prompt(&loads, "behind", 1_000, 0, later + 2 * second);
    assert_eq!(loads.prefill_samples(0), 10);
    // 10_000 tokens over 2 + 2 active seconds.
    assert_eq!(
        loads
            .estimate_at(0, 0, later + 2 * second)
            .prefill_throughput,
        Some(2_500.0)
    );
    assert_eq!(
        (rank_of(None), rank_of(Some(3)), rank_of(Some(u32::MAX))),
        (0, 3, 0)
    );
}

/// Samples whose intervals overlap merge into one entry as they arrive,
/// so a burst of any size is one busy period with every token counted
/// over its seconds once (a ring of per-request samples kept the newest
/// 64 over the seconds of all); a gap starts the next entry; the ring
/// bounds busy periods; a sample arriving after an entry's span opens
/// the next entry with its own interval, and each entry is forgotten a
/// window after its own newest sample, its seconds with it.
#[test]
fn a_busy_period_is_one_entry_however_many_prefills_it_held() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    for _ in 0..256 {
        tracker.prefill_computed_at(0, 1_000, 10.0, 12.0, now);
    }
    assert_eq!(
        (tracker.prefill_entries(0), tracker.prefill_samples(0)),
        (1, 256)
    );
    // 256_000 computed tokens over the two seconds they shared.
    assert_eq!(
        tracker.estimate_at(0, 0, now).prefill_throughput,
        Some(128_000.0)
    );
    // A chain of overlapping prefills extends the entry; a gap ends it.
    tracker.prefill_computed_at(0, 1_000, 11.5, 13.0, now);
    tracker.prefill_computed_at(0, 500, 13.0, 13.5, now);
    tracker.prefill_computed_at(0, 500, 14.0, 15.0, now);
    assert_eq!(
        (tracker.prefill_entries(0), tracker.prefill_samples(0)),
        (2, 259)
    );
    // 258_000 tokens over 3.5 + 1 active seconds.
    assert_eq!(
        tracker.estimate_at(0, 0, now).prefill_throughput,
        Some(258_000.0 / 4.5)
    );

    // More disjoint periods than the ring holds: the oldest leave with
    // their tokens and their seconds together.
    let bursts = LoadTracker::default();
    for index in 0..(PREFILL_ENTRIES + 6) {
        let start = index as f64 * 10.0;
        bursts.prefill_computed_at(1, 100, start, start + 1.0, now);
    }
    assert_eq!(
        (bursts.prefill_entries(1), bursts.prefill_samples(1)),
        (PREFILL_ENTRIES, PREFILL_ENTRIES)
    );
    assert_eq!(
        bursts.estimate_at(1, 0, now).prefill_throughput,
        Some(100.0)
    );

    // The seconds leave with the tokens and never before them: a sample
    // arriving after the entry's span opens the next entry with its own
    // interval (the union counts the overlap once while both are kept),
    // and each entry is forgotten a window after its own newest sample.
    let late = LoadTracker::default();
    let origin = late.epoch;
    let later = origin + Duration::from_secs(50);
    late.prefill_computed_at(0, 1_000, 0.0, 1.0, origin);
    late.prefill_computed_at(0, 1_000, 0.5, 1.5, later);
    late.prefill_computed_at(0, 1_000, 1.0, 1.5, later);
    assert_eq!(late.prefill_entries(0), 2);
    assert_eq!(
        late.estimate_at(0, 0, origin + Duration::from_secs(30))
            .prefill_throughput,
        Some(2_000.0)
    );
    assert_eq!(
        late.estimate_at(0, 0, origin + Duration::from_secs(100))
            .prefill_throughput,
        Some(2_000.0)
    );
    assert_eq!(
        late.estimate_at(0, 0, origin + Duration::from_secs(111))
            .prefill_throughput,
        None
    );
}

/// A rank that never idles reads its recent rate, not the average since
/// it last idled: an entry takes samples for a second after its first
/// and is forgotten a window after its newest, so the seconds leave
/// with the tokens. Three windows without a gap, the mix
/// shifting at two: within the window after the shift the reading is the
/// new rate (an unbounded merge read the whole period's average, 1.9x
/// the new rate here).
#[test]
fn a_rank_that_never_idles_reads_its_recent_rate() {
    let tracker = LoadTracker::default();
    let origin = tracker.epoch;
    let window = PREFILL_SAMPLE_TTL.as_secs_f64();
    let at = |seconds: f64| origin + Duration::from_secs_f64(seconds);
    // A prefill every half second, back to back: 2_400 tokens per second
    // for two windows, then 1_000 for one more.
    let prefill = |index: u64| {
        let start = index as f64 * 0.5;
        let tokens = if start < 2.0 * window { 1_200 } else { 500 };
        tracker.prefill_computed_at(0, tokens, start, start + 0.5, at(start + 0.5));
    };
    let shift = 4 * PREFILL_SAMPLE_TTL.as_secs();
    (0..shift).for_each(&prefill);
    assert_eq!(
        tracker
            .estimate_at(0, 0, at(2.0 * window))
            .prefill_throughput,
        Some(2_400.0)
    );
    (shift..shift + 2 * PREFILL_SAMPLE_TTL.as_secs()).for_each(&prefill);
    // A window after the shift one straddling entry is left (a second of
    // the old rate in sixty-one); half a second later it is gone.
    let reading = tracker
        .estimate_at(0, 0, at(3.0 * window))
        .prefill_throughput
        .expect("a busy rank reads");
    assert!(
        (reading - 1_000.0).abs() < 30.0,
        "{reading} tokens per second a window after the shift"
    );
    assert_eq!(
        tracker
            .estimate_at(0, 0, at(3.0 * window + 0.5))
            .prefill_throughput,
        Some(1_000.0)
    );
    assert!(tracker.prefill_entries(0) <= PREFILL_ENTRIES);
}

/// A long prefill beside sparse short ones reads its tokens over its
/// own seconds: the short entries own nothing between them, so a sample
/// that spanned those idle seconds keeps them (a start clipped to the
/// newest entry's end dropped the seconds and kept the tokens: 19,091
/// against 8,400 here, 10,462 against 8,160 on the sealed path, and a
/// sample delivered after a later one counted its tokens over no
/// seconds at all).
#[test]
fn a_prefill_that_spanned_idle_seconds_keeps_them() {
    let tracker = LoadTracker::default();
    let origin = tracker.epoch;
    let at = |seconds: f64| origin + Duration::from_secs_f64(seconds);
    // Twenty short prefills of 100 tokens, 0.1 s each at a 0.25 s pitch,
    // beside a long one of 40,000 tokens over the whole 5 s, delivered
    // last: 42,000 tokens over 5 s.
    for index in 0..20 {
        let start = 0.1 + 0.25 * index as f64;
        tracker.prefill_computed_at(0, 100, start, start + 0.1, at(start + 0.1));
    }
    tracker.prefill_computed_at(0, 40_000, 0.0, 5.0, at(5.0));
    assert_eq!(
        tracker.estimate_at(0, 0, at(5.0)).prefill_throughput,
        Some(8_400.0)
    );
    // The same beside a sealed entry: four shorts in the first second,
    // the long one after: 20,400 tokens over 2.5 s.
    let sealed = LoadTracker::default();
    let origin = sealed.epoch;
    let at = |seconds: f64| origin + Duration::from_secs_f64(seconds);
    for index in 0..4 {
        let start = 0.1 + 0.25 * index as f64;
        sealed.prefill_computed_at(0, 100, start, start + 0.1, at(start + 0.1));
    }
    sealed.prefill_computed_at(0, 20_000, 0.0, 2.5, at(2.5));
    assert_eq!(
        sealed.estimate_at(0, 0, at(2.5)).prefill_throughput,
        Some(8_160.0)
    );
    // A sample delivered after a later one (an engine stream polled
    // late) brings its own seconds: 2,000 tokens over 2 s.
    let late = LoadTracker::default();
    let origin = late.epoch;
    let at = |seconds: f64| origin + Duration::from_secs_f64(seconds);
    late.prefill_computed_at(0, 1_000, 2.0, 3.0, at(3.0));
    late.prefill_computed_at(0, 1_000, 0.0, 1.0, at(3.0));
    assert_eq!(
        late.estimate_at(0, 0, at(3.0)).prefill_throughput,
        Some(1_000.0)
    );
}

/// Each choice of a request credits its own rank: a fan-out placed per
/// choice never lands choice `i`'s tokens on choice 0's rank, the first
/// choice alone counts the prompt (once) and drops the pending entry,
/// and every choice samples its own prefill.
#[test]
fn choice_loads_credit_their_own_rank() {
    let loads = LoadTracker::default();
    let now = Instant::now();
    let probe = loads.submit(Submission {
        request_id: &Arc::from("r"),
        rank: 0,
        prompt_tokens: 100,
        at: now,
    });
    assert_eq!(loads.pending(), 1);
    let mut first = ChoiceLoad::new(0, 0, probe);
    let mut second = ChoiceLoad::new(1, 1, probe);
    assert_eq!((first.rank(), second.rank()), (0, 1));
    let later = now + Duration::from_millis(50);
    // Not the first choice: its own prefill sampled on its rank, no
    // prompt counted, the request stays pending.
    second.count_prompt(&loads, "r", 100, 0, later);
    assert_eq!((loads.pending(), loads.prefill_samples(1)), (1, 1));
    second.generated(&loads, 7, later);
    first.count_prompt(&loads, "r", 100, 40, later);
    // Once per request: a later output of the first choice adds nothing.
    first.count_prompt(&loads, "r", 100, 40, later);
    first.generated(&loads, 3, later);
    assert_eq!((loads.pending(), loads.prefill_samples(0)), (0, 1));
    let rank0 = loads.estimate_at(0, 0, later);
    let rank1 = loads.estimate_at(1, 0, later);
    // 3 tokens on rank 0 and 7 on rank 1 over the window, 40 of 100
    // prompt tokens cached on rank 0 and nothing counted on rank 1.
    assert_eq!((rank0.gen_throughput, rank1.gen_throughput), (1.5, 3.5));
    assert_eq!((rank0.cache_hit_rate, rank1.cache_hit_rate), (0.4, 0.0));
    let armed = loads.submit(Submission {
        request_id: &Arc::from("s"),
        rank: 1,
        prompt_tokens: 100,
        at: later,
    });
    let mut only = ChoiceLoad::new(1, 0, armed);
    only.count_prompt(&loads, "s", 100, 0, later + Duration::from_millis(10));
    assert_eq!((loads.pending(), loads.prefill_samples(1)), (0, 2));
}

/// The uncounted share (the servicer's own health probe) records
/// nothing: no prompt, no prefill sample, no generation; the rank it
/// met reads idle after it, as before.
#[test]
fn an_uncounted_choice_records_nothing() {
    let loads = LoadTracker::default();
    let now = Instant::now();
    let mut probe = ChoiceLoad::uncounted();
    probe.count_prompt(
        &loads,
        "HEALTH_CHECK_1",
        1,
        0,
        now + Duration::from_millis(10),
    );
    probe.generated(&loads, 1, now + Duration::from_millis(10));
    assert_eq!((loads.pending(), loads.prefill_samples(0)), (0, 0));
    let rank = loads.estimate_at(0, 0, now + Duration::from_millis(10));
    assert_eq!(
        (
            rank.cache_hit_rate,
            rank.gen_throughput,
            rank.prefill_throughput
        ),
        (0.0, 0.0, None)
    );
}

/// Two ranks serve: each rank's hit rate, throughput and queued work are
/// its own, and a rank that served nothing reads as idle, not as its
/// peer.
#[test]
fn each_rank_keeps_its_own_hit_rate_throughput_and_queue() {
    let tracker = LoadTracker::default();
    let now = Instant::now();
    tracker.submitted(0, &Arc::from("a"), 1_000);
    tracker.submitted(1, &Arc::from("b"), 400);
    tracker.submitted(1, &Arc::from("c"), 600);
    tracker.first_output(0, "a", 1_000, 0);
    tracker.first_output(1, "b", 400, 400);
    tracker.generated_at(0, 100, now);
    tracker.generated_at(1, 900, now);

    let rank0 = tracker.estimate_at(0, 1, now);
    let rank1 = tracker.estimate_at(1, 1, now);
    assert_eq!((rank0.cache_hit_rate, rank1.cache_hit_rate), (0.0, 1.0));
    assert_eq!((rank0.gen_throughput, rank1.gen_throughput), (50.0, 450.0));
    // Rank 0 has nothing waiting; rank 1's `c` waits behind `b`, with
    // rank 1's own hit rate discounting it to nothing.
    assert_eq!((rank0.queued_token_work, rank1.queued_token_work), (0, 0));
    tracker.first_output(1, "b", 400, 0);
    assert_eq!(tracker.estimate_at(1, 1, now).queued_token_work, 300);
    assert_eq!(tracker.estimate_at(2, 1, now), LoadEstimate::default());
}

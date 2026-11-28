use super::*;

thread_local! {
    /// What a case runs on its own thread between a retirement's two
    /// writes, behind the estimates' and the count's and ahead of the
    /// held word's: the row as a read finds it there.
    pub(crate) static BETWEEN_A_RETIREMENTS_WRITES: Cell<Option<fn(&RankCredit)>> =
        const { Cell::new(None) };

    /// What a case runs on its own thread at each dispatch that a release
    /// lets go, ahead of the held word's write and, where the dispatch
    /// stated no size, behind the estimates' and the count's: the row as a
    /// read finds it between the two. A release of a row that holds no
    /// such dispatch runs it at none: that release has one write a
    /// dispatch. The release holds every lane's lock there, so what a
    /// case runs is to release nothing: a second release would wait for
    /// those locks without end.
    pub(crate) static BETWEEN_A_RELEASES_WRITES: Cell<Option<fn(&RankCredit)>> =
        const { Cell::new(None) };

    /// What a case runs on its own thread inside a read, behind the held
    /// word's load and ahead of the count's or the estimates': a release
    /// run there is one the read goes across.
    pub(crate) static BETWEEN_A_READS_LOADS: Cell<Option<fn(&RankCredit)>> =
        const { Cell::new(None) };
}

/// Pins this thread's lane, so a test owns one lane per thread. A lane
/// that a row does not have is refused here, not at the next dispatch.
pub(crate) fn take_lane(lane: usize) {
    assert!(lane < DISPATCH_LANES, "a row has {DISPATCH_LANES} lanes");
    LANE.with(|drawn| drawn.set(Some(lane)));
}

// What a test reads of a row. The row's words stay `rank_credit.rs`'s own.
impl RankCredit {
    /// The held token-work and requests, and the sends of no stated size.
    pub(crate) fn held_for_test(&self) -> (HeldCredit, u16) {
        (
            self.held_credit(),
            self.unsized_held.load(Ordering::Relaxed),
        )
    }

    /// The row's three words as they stand: the stated token-work, the
    /// requests, the sends of no stated size and their estimates.
    fn words_for_test(&self) -> (u64, u64, u16, u64) {
        let (HeldCredit { tokens, requests }, unsized_held) = self.held_for_test();
        (tokens, requests, unsized_held, self.estimated())
    }

    /// One retirement, as a dispatch that found the row at its bound runs
    /// it.
    pub(crate) fn retire_for_test(&self) {
        self.retire_oldest();
    }

    /// Whether every lane's lock is held now: what a dispatch's record
    /// and a release wait for.
    pub(crate) fn lanes_held_for_test(&self) -> bool {
        self.lanes.iter().all(|lane| lane.dispatches.is_locked())
    }
}

/// A lane's dispatches are held by instant and let go through one: one
/// dated earlier lands in front, one made at a held instant behind it,
/// and the oldest goes where it was made at or before the instant.
#[test]
fn a_lanes_dispatches_are_held_by_instant_and_let_go_through_one() {
    let mut dispatches = Dispatches::default();
    assert_eq!(dispatches.oldest(), None);
    // The first held is the oldest. A later one is not.
    assert!(dispatches.hold(20, SendSize::stated(2)));
    assert!(!dispatches.hold(30, SendSize::stated(3)));
    // One dated earlier is the oldest now.
    assert!(dispatches.hold(10, SendSize::stated(1)));
    // One dated between lands between, one at a held instant behind it.
    assert!(!dispatches.hold(25, SendSize::stated(4)));
    assert!(!dispatches.hold(20, SendSize::stated(5)));
    assert_eq!((dispatches.oldest(), dispatches.len()), (Some(10), 5));
    // The oldest stays while the instant is before it and goes at its
    // own instant. The next stays one microsecond short of its own.
    assert_eq!(dispatches.let_go(9), None);
    assert_eq!(dispatches.let_go(10), Some(SendSize::stated(1)));
    assert_eq!(dispatches.let_go(19), None);
    assert_eq!(dispatches.oldest(), Some(20));
    // Oldest first, and of one instant the earlier held first.
    let gone: Vec<u32> = std::iter::from_fn(|| dispatches.let_go(u64::MAX))
        .map(SendSize::tokens)
        .collect();
    assert_eq!(gone, [2, 5, 4, 3]);
    assert_eq!((dispatches.oldest(), dispatches.len()), (None, 0));
}

/// Three ranks whose walks start at one slot are each found, and the two
/// behind are still found once the first is dropped: a walk stops at a
/// free slot, so the table is laid again and no slot is freed in place.
#[test]
fn a_rank_is_found_behind_one_the_table_dropped() {
    let at = |id: u64| RankTable::start(&(WorkerId(id), 0)) & 7;
    let alike: Vec<RankKey> = (1..)
        .filter(|&id| at(id) == at(1))
        .take(3)
        .map(|id| (WorkerId(id), 0))
        .collect();
    let mut table = RankTable::default();
    for &rank in &alike {
        table.insert(rank, Arc::new(RankCredit::default()));
    }
    assert!(alike.iter().all(|rank| table.contains_key(rank)));
    let table = table.retained(|rank, _| *rank != alike[0]);
    assert!(!table.contains_key(&alike[0]));
    assert!(table.contains_key(&alike[1]) && table.contains_key(&alike[2]));
    assert_eq!(table.iter().count(), 2);
}

/// A read loads the held word ahead of the count of the dispatches of
/// no tokens. The row holds such dispatches alone, and a release that
/// lets every one go runs whole between those two loads. The read finds
/// the dispatches the held word counted and none of no tokens: no more
/// of them than dispatches.
#[test]
fn a_read_across_a_release_counts_no_more_unsized_dispatches_than_dispatches() {
    const HELD: u32 = 64;
    fn release(row: &RankCredit) {
        // The hook takes itself off: it runs inside one read alone.
        BETWEEN_A_READS_LOADS.set(None);
        row.release_through(u64::from(HELD));
    }
    let row = RankCredit::default();
    for at in 1..=u64::from(HELD) {
        row.record(at, SendSize::stated(0));
    }
    BETWEEN_A_READS_LOADS.set(Some(release));
    // An instant ahead of every dispatch: the read itself releases none.
    let read = row.sent_through(0);
    assert!(
        BETWEEN_A_READS_LOADS.get().is_none(),
        "the release ran inside the read"
    );
    assert_eq!((read.requests, read.unsized_requests), (HELD, 0));
    assert_eq!(row.sent_through(0), SentCredit::default());
}

/// A read of the documents' two numbers loads the held word ahead of
/// the estimates' word. Every dispatch here is an estimate of three
/// tokens, and a release that lets every one go runs whole between
/// those two loads. The read finds the dispatches the held word counted
/// and no token: it adds no estimate of a dispatch the held word has
/// let go.
#[test]
fn a_read_across_a_release_adds_no_estimate_of_a_dispatch_let_go() {
    const HELD: u32 = 64;
    const EACH: u32 = 3;
    fn release(row: &RankCredit) {
        // The hook takes itself off: it runs inside one read alone.
        BETWEEN_A_READS_LOADS.set(None);
        row.release_through(u64::from(HELD));
    }
    let row = RankCredit::default();
    for at in 1..=u64::from(HELD) {
        row.record(at, SendSize::estimated(EACH));
    }
    BETWEEN_A_READS_LOADS.set(Some(release));
    // An instant ahead of every dispatch: the read itself releases none.
    let read = row.counts_through(0);
    assert!(
        BETWEEN_A_READS_LOADS.get().is_none(),
        "the release ran inside the read"
    );
    let counts = DispatchCounts {
        requests: HELD,
        tokens: 0,
    };
    assert_eq!(read, counts);
    assert_eq!(row.counts_through(0), DispatchCounts::default());
}

/// The count of the dispatches of no tokens goes down ahead of the held
/// word at each such dispatch a release lets go. The row holds such
/// dispatches alone, and a read of the row's two words between the two
/// writes of each one finds the held word still counting it and the
/// count not: one dispatch more than dispatches of no tokens, at every
/// dispatch the release lets go.
#[test]
fn a_read_between_a_releases_writes_counts_no_more_unsized_dispatches_than_dispatches() {
    const HELD: u32 = 64;
    thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
    fn read(row: &RankCredit) {
        // An instant ahead of every dispatch releases none.
        let read = row.sent_through(0);
        // The dispatches the release let go ahead of this one.
        let gone = READS.get();
        assert_eq!(
            (read.requests, read.unsized_requests),
            (HELD - gone, HELD - gone - 1),
            "the dispatches beside those of no tokens, {gone} let go"
        );
        READS.set(gone + 1);
    }
    let row = RankCredit::default();
    for at in 1..=u64::from(HELD) {
        row.record(at, SendSize::stated(0));
    }
    BETWEEN_A_RELEASES_WRITES.set(Some(read));
    let left = row.sent_through(u64::from(HELD));
    BETWEEN_A_RELEASES_WRITES.set(None);
    assert_eq!(READS.get(), HELD, "one read at each dispatch let go");
    assert_eq!(left, SentCredit::default());
}

/// The estimates' word goes down ahead of the held word at each
/// dispatch of an estimate that a release lets go. Every dispatch here
/// is an estimate of three tokens, and a read of the documents' two
/// numbers between the two writes of each one finds the held word still
/// counting it and its estimate gone: it adds no estimate of a dispatch
/// the held word has let go, and reads that one dispatch at no tokens.
#[test]
fn a_read_between_a_releases_writes_adds_no_estimate_of_a_dispatch_let_go() {
    const HELD: u32 = 64;
    const EACH: u32 = 3;
    thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
    fn read(row: &RankCredit) {
        // An instant ahead of every dispatch releases none.
        let read = row.counts_through(0);
        // The dispatches the release let go ahead of this one.
        let gone = READS.get();
        let counts = DispatchCounts {
            requests: HELD - gone,
            tokens: u64::from(EACH) * u64::from(HELD - gone - 1),
        };
        assert_eq!(read, counts, "the documents' two numbers, {gone} let go");
        READS.set(gone + 1);
    }
    let row = RankCredit::default();
    for at in 1..=u64::from(HELD) {
        row.record(at, SendSize::estimated(EACH));
    }
    BETWEEN_A_RELEASES_WRITES.set(Some(read));
    let left = row.counts_through(u64::from(HELD));
    BETWEEN_A_RELEASES_WRITES.set(None);
    assert_eq!(READS.get(), HELD, "one read at each dispatch let go");
    assert_eq!(left, DispatchCounts::default());
}

/// The estimates' word goes down ahead of the held word at a
/// retirement too. Every dispatch here is an estimate of three tokens,
/// and the dispatch past the row's bound lets the row's oldest go. A
/// read of the documents' two numbers between the retirement's two
/// writes finds the held word still counting that one and no more than
/// three tokens a dispatch: it adds no estimate of a dispatch the held
/// word has let go.
#[test]
fn a_read_between_a_retirements_writes_adds_no_estimate_of_a_dispatch_let_go() {
    const EACH: u32 = 3;
    thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
    fn read(row: &RankCredit) {
        // An instant ahead of every dispatch releases none.
        let read = row.counts_through(0);
        assert!(
            read.tokens <= u64::from(EACH) * u64::from(read.requests),
            "{} estimated tokens beside {} dispatches",
            read.tokens,
            read.requests
        );
        READS.set(READS.get() + 1);
    }
    let row = RankCredit::default();
    for at in 1..=RANK_CREDIT_KEPT {
        row.record(at, SendSize::estimated(EACH));
    }
    BETWEEN_A_RETIREMENTS_WRITES.set(Some(read));
    row.record(RANK_CREDIT_KEPT + 1, SendSize::estimated(EACH));
    BETWEEN_A_RETIREMENTS_WRITES.set(None);
    assert_eq!(READS.get(), 1, "one retirement, read between its writes");
    let kept = RANK_CREDIT_KEPT;
    let words = (0, kept, kept as u16, u64::from(EACH) * kept);
    assert_eq!(row.words_for_test(), words);
}

/// A copy takes the longest of its reads of a rank's standing rows,
/// and of two as long the earlier. Of an empty read and a read of one
/// dispatch the second is handed, and of two reads of one dispatch the
/// first: their sizes tell them apart. So a copy of two tallies whose
/// first let its one dispatch go holds the dispatch the second still
/// holds.
#[test]
fn a_copy_holds_the_longest_of_its_reads_and_of_two_as_long_the_earlier() {
    let read = |tokens| vec![(91_300, SendSize::stated(tokens))];
    let longest = Credits::longest_read([Vec::new(), read(317)].into_iter());
    assert_eq!(longest, read(317), "the read that holds a dispatch");
    let longest = Credits::longest_read([read(317), read(211)].into_iter());
    assert_eq!(longest, read(317), "the earlier of two as long");
    let worker = WorkerId(7);
    let standing = [Credits::default(), Credits::default()];
    for tally in &standing {
        let row = tally.entry((worker, 0)).expect("the rank's entry");
        row.record(91_300, SendSize::stated(317));
    }
    let first = standing[0].entry((worker, 0)).expect("the rank's entry");
    first.release_through(u64::MAX);
    let copy = Credits::default();
    copy.copy_fullest(standing.iter());
    assert_eq!(copy.held(worker), 1, "the dispatch the second holds");
    let held: Vec<_> = standing.iter().map(|tally| tally.held(worker)).collect();
    assert_eq!(held, [0, 1], "the first let its own go");
}

/// The longest read is handed oldest first, whatever order the row's
/// lanes hand it in. The standing row holds its newer dispatches in an
/// earlier lane than its older ones, so its own read hands the newer
/// first. The longest of the reads holds the same five by their
/// instants, and the copy holds every one.
#[test]
fn the_longest_read_of_a_copy_is_handed_oldest_first() {
    let worker = WorkerId(7);
    let standing = Credits::default();
    let row = standing.entry((worker, 0)).expect("the rank's entry");
    // The first lane takes the three newer dispatches, the second the
    // two older ones.
    take_lane(0);
    for at in [91_310, 91_320, 91_330] {
        row.record(at, SendSize::stated(317));
    }
    take_lane(1);
    for at in [91_300, 91_305] {
        row.record(at, SendSize::stated(317));
    }
    let instants = |read: &[(u64, SendSize)]| read.iter().map(|&(at, _)| at).collect::<Vec<_>>();
    let read = row.held_sends();
    assert_eq!(instants(&read), [91_310, 91_320, 91_330, 91_300, 91_305]);
    let longest = Credits::longest_read(std::iter::once(read));
    assert_eq!(instants(&longest), [91_300, 91_305, 91_310, 91_320, 91_330]);
    let copy = Credits::default();
    copy.copy_fullest(std::iter::once(&standing));
    assert_eq!(copy.held(worker), 5, "every dispatch");
}

/// A dispatch that states no tokens states no size: it is held as an
/// estimate of none, among the sends of no stated size.
#[test]
fn a_stated_size_of_no_tokens_is_held_as_a_send_of_no_stated_size() {
    assert_eq!(SendSize::stated(0), SendSize::estimated(0));
    let row = RankCredit::default();
    row.record(1, SendSize::stated(0));
    assert_eq!(row.words_for_test(), (0, 1, 1, 0));
}

/// A size is held under the mark of an estimate, stated or estimated:
/// one at the mark's bit and one past it are held as the largest
/// under it. A stated one stays a stated one. It is not read as an
/// estimate, and its release takes it out of the held word and
/// leaves the estimates' word as it stands.
#[test]
fn a_size_at_the_marks_bit_is_held_under_it_and_in_its_own_word() {
    const MOST: u64 = (1 << 31) - 1;
    for size in [1u32 << 31, u32::MAX] {
        let row = RankCredit::default();
        row.record(1, SendSize::stated(size));
        assert_eq!(row.words_for_test(), (MOST, 1, 0, 0), "stated {size}");
        row.record(2, SendSize::estimated(size));
        assert_eq!(row.words_for_test(), (MOST, 2, 1, MOST), "{size}");
        // A report sampled at the first lets the stated one go alone.
        let read = row.counts_through(1);
        assert_eq!((read.requests, read.tokens), (1, MOST), "{size}");
        assert_eq!(row.words_for_test(), (0, 1, 1, MOST), "{size}");
        row.counts_through(2);
        assert_eq!(row.words_for_test(), (0, 0, 0, 0), "{size}");
    }
}

/// A lane that holds a stated size, an estimate and a stated size
/// reads back to nothing in each of the row's words, by a release and
/// by a retirement: an estimate goes with its dispatch. The
/// documents' read adds the estimate to the stated token-work, and
/// the read of a stage that prices such a send itself counts it and
/// adds none.
#[test]
fn a_mixed_lane_reads_back_to_nothing_by_a_release_and_by_a_retirement() {
    let mixed = [
        SendSize::stated(7),
        SendSize::estimated(101),
        SendSize::stated(5),
    ];
    let row = RankCredit::default();
    for (at, size) in (1..).zip(mixed) {
        row.record(at, size);
    }
    assert_eq!(row.words_for_test(), (12, 3, 1, 101));
    // An instant ahead of every dispatch releases none.
    let read = row.counts_through(0);
    assert_eq!((read.requests, read.tokens), (3, 113));
    let sent = row.sent_through(0);
    assert_eq!((sent.requests, sent.tokens), (3, 12));
    assert_eq!(sent.unsized_requests, 1);
    // A release, one dispatch at a time.
    row.counts_through(1);
    assert_eq!(row.words_for_test(), (5, 2, 1, 101));
    row.counts_through(2);
    assert_eq!(row.words_for_test(), (5, 1, 0, 0));
    row.counts_through(3);
    assert_eq!(row.words_for_test(), (0, 0, 0, 0));

    // A retirement: each of two dispatches that land past the row's
    // bound lets the row's oldest go, a stated size and then an
    // estimate.
    let row = RankCredit::default();
    for at in 0..RANK_CREDIT_KEPT {
        row.record(at + 1, mixed[(at % 3) as usize]);
    }
    assert_eq!(
        row.words_for_test(),
        (2_731 * 7 + 2_730 * 5, 8_192, 2_731, 2_731 * 101)
    );
    for at in RANK_CREDIT_KEPT..RANK_CREDIT_KEPT + 2 {
        row.record(at + 1, mixed[(at % 3) as usize]);
    }
    assert_eq!(
        row.words_for_test(),
        (2_731 * 12, 8_192, 2_730, 2_730 * 101),
        "the oldest two went, each with its own word's share"
    );
    row.counts_through(u64::MAX);
    assert_eq!(row.words_for_test(), (0, 0, 0, 0));
}

/// A read of a row's dispatches whose instant is short of the row's
/// oldest writes nothing of the row. A retirement leaves that instant
/// where it was, below the oldest dispatch the row still holds: a read
/// short of it hands every held dispatch and leaves it there. A read
/// that reaches it runs the release, which lets nothing go here and
/// states the oldest held dispatch's instant.
#[test]
fn a_read_of_a_rows_dispatches_short_of_its_oldest_instant_writes_nothing_of_the_row() {
    let row = RankCredit::default();
    // The dispatch that finds the row at its bound lets the oldest go.
    let (first, last) = (10, 10 + RANK_CREDIT_KEPT);
    for at in first..=last {
        row.record(at, SendSize::stated(1));
    }
    let oldest_held = first + 1;
    let held: Vec<u64> = (oldest_held..=last).collect();
    let oldest = || row.oldest_micros.load(Ordering::Acquire);
    let handed = |through_micros| {
        let mut handed = Vec::new();
        row.each_through(through_micros, |at_micros, _| handed.push(at_micros));
        handed
    };
    assert_eq!(oldest(), first, "a retirement leaves it");
    assert_eq!(handed(first - 1), held);
    assert_eq!(oldest(), first, "a read short of it stores none");
    assert_eq!(handed(first), held);
    assert_eq!(oldest(), oldest_held, "a read that reaches it releases");
}

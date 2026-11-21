//! The warm-up ruling: which rows of a pool warm.
//!
//! The `warmup-slice` stage of the `cache_aware` policy applies it to a
//! decision's rows. A host that names a pool's warming rows asks the same
//! ruling ([`WarmUpRule::list_warming_rows`]), so the two never differ.

/// How a host reads the rows a ruling named ([`WarmUpRule::list_warming_rows`])
/// at an instant of the ruling's clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WarmUpState {
    /// The rows named are the thin.
    pub thin: bool,
    /// The instant from which the rows are named: `0`, unless every row
    /// was young. Such a fleet has nobody to catch up to until the first
    /// of its rows is young no more.
    pub from: u64,
}

/// The warm-up stage's ruling of who warms, as its three numbers: what the
/// stage rules a host's column by, and what a host that names a pool's
/// warming rows enters ([`WarmUpRule::list_warming_rows`]). The ruling has this
/// one home: a host hands it the pool's sizes and admissions and never restates
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmUpRule {
    /// How long after its admission a worker is young, in microseconds; `0`
    /// keeps nobody young.
    pub(crate) window_micros: u64,
    /// The level from which a fleet holds a cache worth catching up to, and
    /// the growth a young worker is short of, in blocks.
    pub(crate) blocks: u32,
    /// The share of the fleet's level under which an index is thin; `0`
    /// keeps nobody thin.
    pub(crate) thin_ratio: f32,
}

impl WarmUpRule {
    /// Whether an index of `indexed` blocks is thin against the fleet's
    /// `level`: empty, or below the ratio of the level, once the fleet holds
    /// a cache worth catching up to.
    #[inline]
    pub(crate) fn is_thin(&self, indexed: u32, level: u32) -> bool {
        self.thin_ratio > 0.0
            && level >= self.blocks
            && (indexed == 0 || f64::from(indexed) < f64::from(level) * f64::from(self.thin_ratio))
    }

    /// Whether a worker admitted `age_micros` ago whose index grew by
    /// `growth` blocks since is young: within the window and short of its
    /// first blocks.
    #[inline]
    pub(crate) fn is_young(&self, age_micros: u64, growth: u32) -> bool {
        age_micros < self.window_micros && growth < self.blocks
    }

    /// The fleet's index level: the upper median of the rows' index sizes,
    /// so one emptied worker does not drag it down. Reorders `sizes`.
    #[inline]
    pub(crate) fn level(sizes: &mut [u32]) -> u32 {
        if sizes.is_empty() {
            return 0;
        }
        let mid = sizes.len() / 2;
        let (_, level, _) = sizes.select_nth_unstable(mid);
        *level
    }

    /// The pool's warming rows by this ruling, for a host that names them
    /// ([`AffinityRows::warming`]). `sizes` holds each row's index size
    /// and `admitted` each row's admission, on the clock `now` is read on,
    /// in microseconds; `scratch` is the host's, for the level. The rows go
    /// to `out`, emptied first, ascending, each with the instant its naming
    /// ends: the thin whatever the fleet's age, with no end (`u64::MAX`),
    /// else the young, each until its window is over, as the stage rules a
    /// column of the same sizes and ages. A row is admitted with nothing
    /// indexed, so its growth since admission is its size.
    ///
    /// The rows stand for every later instant of that clock while the sizes
    /// do: at an instant a host names the rows whose end is past it, and
    /// nobody before [`WarmUpState::from`]. So a host that reads the sizes at
    /// its own pace still ends a young row's naming on time.
    ///
    /// [`AffinityRows::warming`]: crate::engine::view::AffinityRows::warming
    pub fn list_warming_rows(
        &self,
        sizes: &[u32],
        admitted: &[u64],
        now: u64,
        scratch: &mut Vec<u32>,
        out: &mut Vec<(u32, u64)>,
    ) -> WarmUpState {
        self.list_warming_rows_grown(sizes, sizes, admitted, now, scratch, out)
    }

    /// [`Self::list_warming_rows`] for a host whose index may hold blocks of
    /// a row from before its admission: `growth` holds what each row's index
    /// gained since the row was admitted, which a young row is short of,
    /// and `sizes` what it holds, which thinness is read from. A worker
    /// that came back under its address keeps the blocks its last run
    /// stored until its first events say otherwise, so its size alone would
    /// end a warm-up that has not begun.
    pub fn list_warming_rows_grown(
        &self,
        sizes: &[u32],
        growth: &[u32],
        admitted: &[u64],
        now: u64,
        scratch: &mut Vec<u32>,
        out: &mut Vec<(u32, u64)>,
    ) -> WarmUpState {
        out.clear();
        let young = |grown: u32, admitted: u64| self.is_young(now.saturating_sub(admitted), grown);
        let any_indexed = sizes.iter().any(|&size| size > 0);
        let youngs = growth
            .iter()
            .zip(admitted)
            .filter(|&(&grown, &admitted)| young(grown, admitted))
            .count();
        if !any_indexed && youngs == 0 {
            return WarmUpState::default();
        }
        let level = if any_indexed {
            scratch.clear();
            scratch.extend_from_slice(sizes);
            Self::level(scratch)
        } else {
            0
        };
        out.extend(
            (0u32..)
                .zip(sizes)
                .filter(|&(_, &size)| self.is_thin(size, level))
                .map(|(row, _)| (row, u64::MAX)),
        );
        if !out.is_empty() {
            return WarmUpState {
                thin: true,
                from: 0,
            };
        }
        let window = self.window_micros;
        out.extend(
            (0u32..)
                .zip(growth.iter().zip(admitted))
                .filter(|&(_, (&grown, &admitted))| young(grown, admitted))
                .map(|(row, (_, &admitted))| (row, admitted.saturating_add(window))),
        );
        // A fleet that is all young has nobody to catch up to, until the
        // first of its rows is young no more.
        let first_end = out.iter().map(|&(_, until)| until).min();
        let from = first_end.filter(|_| youngs == sizes.len()).unwrap_or(0);
        WarmUpState { thin: false, from }
    }
}

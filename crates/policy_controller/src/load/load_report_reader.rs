//! The view's reading of a row's load report, for a stage a host
//! registers. A document's built-in stages that read a report are wrapped
//! where the document builds them, so each reads a report past the view's
//! maximum age as none. A registered stage is not wrapped. It takes
//! [`LoadReportReader`] from its build context
//! ([`StageBuild::load_report_reader`](crate::stages::StageBuild::load_report_reader))
//! and reads each row through it, one pass a decision ([`LoadReportPass`]). The
//! boundary is the view's one cutoff, the one the wrapped stages decide
//! through: this module holds no copy of it.

use crate::engine::{
    view::{Candidate, LoadReportCutoff, LoadRow},
    RequestFacts,
};

/// How the view reads a row's load report, for a stage a host registers.
///
/// A document may set `view.max_report_age_ms`. A report older than that
/// at a decision then reads as no report, and the built-in stages that
/// read a report read every row that way. Nothing wraps a stage of your
/// own: `row.load` hands it the report as the host holds it, at any age.
/// A stage that should see what the built-in stages see takes this value
/// from its build context ([`StageBuild::load_report_reader`]), keeps it, and
/// asks it for each row's report.
///
/// ```
/// use smg_policy_controller::{
///     engine::{Candidate, RequestFacts, Scorer, TieBand},
///     load::LoadReportReader,
///     stages::{StageRegistry, Lifecycle},
/// };
///
/// /// Favors the rows whose report the view still reads.
/// #[derive(Debug)]
/// struct Reported(LoadReportReader);
///
/// impl Lifecycle for Reported {}
///
/// impl Scorer for Reported {
///     fn name(&self) -> &'static str {
///         "reported"
///     }
///
///     fn score(
///         &self,
///         facts: &RequestFacts<'_>,
///         candidates: &[Candidate],
///         kept: &[u32],
///         out: &mut [f32],
///     ) -> TieBand {
///         // One pass a decision: the cutoff is worked out here, once.
///         let reports = self.0.pass(facts);
///         for (&row, score) in kept.iter().zip(out) {
///             let report = reports.of(&candidates[row as usize]);
///             *score = if report.is_some() { 1.0 } else { 0.0 };
///         }
///         TieBand::NORMALIZED
///     }
/// }
///
/// let mut stages = StageRegistry::builtin();
/// stages.scorer_with("reported", |build| Ok(Box::new(Reported(build.load_report_reader()))))?;
/// # Ok::<(), smg_policy_controller::stages::StageRegistryError>(())
/// ```
///
/// A report's age at a decision is the request's `now_micros` less the
/// row's `load_sampled_at_micros`. A report exactly as old as the maximum
/// is still read, and one a microsecond older is not: the boundary the
/// built-in stages hold, decided in the one place they decide it. Under a
/// document that sets no maximum, every report is read as it stands.
///
/// [`StageBuild::load_report_reader`]: crate::stages::StageBuild::load_report_reader
#[derive(Clone, Copy, Debug)]
pub struct LoadReportReader {
    max_age_ms: Option<u64>,
}

impl LoadReportReader {
    /// The reading of a view with this maximum report age, or with none.
    pub(crate) fn new(max_age_ms: Option<u64>) -> Self {
        Self { max_age_ms }
    }

    /// One pass's reads, at the request's clock. A stage that scores many
    /// rows opens one pass a decision and asks it for each row.
    #[inline]
    pub fn pass(&self, facts: &RequestFacts<'_>) -> LoadReportPass {
        LoadReportPass {
            cutoff: self
                .max_age_ms
                .map_or(LoadReportCutoff::ANY_AGE, |max_age_ms| {
                    LoadReportCutoff::new(facts.now_micros, max_age_ms)
                }),
        }
    }

    /// One row's report: [`Self::pass`] and [`LoadReportPass::of`] in one
    /// call, for a stage that is asked one row at a time, as a filter is.
    #[inline]
    pub fn read<'r>(&self, facts: &RequestFacts<'_>, row: &'r Candidate) -> Option<&'r LoadRow> {
        self.pass(facts).of(row)
    }
}

/// A pass over the rows' reports at one request's clock
/// ([`LoadReportReader::pass`]).
#[derive(Clone, Copy, Debug)]
pub struct LoadReportPass {
    cutoff: LoadReportCutoff,
}

impl LoadReportPass {
    /// The row's report as the view reads it: none where the row holds
    /// none, and none where it is older than the view's maximum age.
    #[inline]
    pub fn of(self, row: &Candidate) -> Option<&LoadRow> {
        self.cutoff.load(row)
    }
}

#[cfg(test)]
mod tests;

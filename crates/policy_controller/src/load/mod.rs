//! Worker load: the report a worker sends, how a stage reads it, and the named
//! values it carries.

#[cfg(feature = "ingest")]
pub mod ingest;
#[cfg(any(feature = "ingest", feature = "worker-host"))]
pub(crate) mod load_report;
pub(crate) mod load_report_reader;
pub(crate) mod signals;

#[cfg(any(feature = "ingest", feature = "worker-host"))]
pub use load_report::{load_row, signal_values, RankValues, ReportOf};
pub use load_report_reader::{LoadReportPass, LoadReportReader};
pub use signals::{SignalCell, SignalCells, SignalColumn};

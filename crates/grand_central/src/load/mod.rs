//! Worker load: the report a worker sends, how a stage reads it, and the named
//! values it carries.

#[cfg(feature = "in-process")]
pub(crate) mod load_report;
pub(crate) mod load_report_reader;
pub(crate) mod signals;

#[cfg(feature = "in-process")]
pub use load_report::{load_row, signal_values};
pub use load_report_reader::{LoadReportPass, LoadReportReader};
pub use signals::{ReportSignals, SignalCell, SignalCells, SignalColumn};

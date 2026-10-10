//! A worker's report that the media a request names failed, carried on the
//! status of the `Generate` it refused.

use tonic::{metadata::MetadataValue, Status};

/// Whose fault the request's media failure is, as a worker reports it on the
/// status of the `Generate` it refused, in the trailer [`Self::METADATA_KEY`].
/// A router that reads the trailer answers the client for the media (a 4xx
/// for the request's own fault, a 5xx that is not the worker's for the media
/// host's) and records no circuit-breaker sample against a worker that is
/// healthy; a status without the trailer is read as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMediaFault {
    /// The request's own: a host that cannot be resolved or refuses the
    /// connection, a 4xx answer, a payload that is not the media it claims to
    /// be, a media parameter out of range.
    Client,
    /// The media host's, for now: a fetch that ran out of its budget, a 5xx
    /// answer, a transfer that broke while the body arrived.
    Transient,
}

impl WorkerMediaFault {
    /// The gRPC trailer that carries the report.
    pub const METADATA_KEY: &'static str = "smg-media-fault";

    /// The trailer's value for this fault.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::Transient => "transient",
        }
    }

    /// `status` with this report on it.
    pub fn stamp(self, mut status: Status) -> Status {
        status.metadata_mut().insert(
            Self::METADATA_KEY,
            MetadataValue::from_static(self.as_str()),
        );
        status
    }

    /// The report `status` carries, if any.
    pub fn from_status(status: &Status) -> Option<Self> {
        match status.metadata().get(Self::METADATA_KEY)?.to_str().ok()? {
            "client" => Some(Self::Client),
            "transient" => Some(Self::Transient),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use super::*;

    #[test]
    fn a_report_survives_the_status_it_is_stamped_on() {
        for fault in [WorkerMediaFault::Client, WorkerMediaFault::Transient] {
            let status = fault.stamp(Status::unavailable("media fetch timed out after 10s"));
            assert_eq!(status.code(), Code::Unavailable);
            assert_eq!(status.message(), "media fetch timed out after 10s");
            assert_eq!(WorkerMediaFault::from_status(&status), Some(fault));
        }
    }

    #[test]
    fn a_status_without_the_report_carries_none() {
        assert_eq!(
            WorkerMediaFault::from_status(&Status::unavailable("engine unavailable")),
            None
        );
        let mut odd = Status::unavailable("odd");
        odd.metadata_mut().insert(
            WorkerMediaFault::METADATA_KEY,
            MetadataValue::from_static("elsewhere"),
        );
        assert_eq!(WorkerMediaFault::from_status(&odd), None);
    }
}

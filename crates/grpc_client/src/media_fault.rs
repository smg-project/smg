//! A worker's report that the media a request names failed, carried on the
//! status of the `Generate` it refused.

use tonic::{metadata::MetadataValue, Status};

/// Whose fault the request's media failure is, as a worker reports it on the
/// status of the `Generate` it refused, in the trailer [`Self::METADATA_KEY`].
/// A router that reads the trailer answers the client for the media (a 400
/// for either fault: an origin that does not deliver is the request's input
/// as much as a reference it got wrong, under the code the worker names in
/// [`MEDIA_CODE_METADATA_KEY`]) and records no circuit-breaker sample
/// against a worker that is healthy; a status without the trailer is read
/// as before. The fault tells the logs whether the origin or the reference
/// failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerMediaFault {
    /// The request's own: a host that cannot be resolved or refuses the
    /// connection, a 4xx answer, a payload that is not the media it claims to
    /// be or is over a size cap, a media parameter out of range.
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

/// The trailer that carries the code the public API answers for the
/// request's media failure (`media_fetch_failed`, `media_too_large`), next
/// to the fault's: the worker names the code, the router answers it under
/// the 400. A router that reads none (a worker that predates the trailer, a
/// code it does not know) answers `media_fetch_failed`, the code every
/// failure of the media had before the trailer.
pub const MEDIA_CODE_METADATA_KEY: &str = "smg-media-code";

/// `status` with `code` on it; see [`MEDIA_CODE_METADATA_KEY`].
pub fn stamp_media_code(mut status: Status, code: &'static str) -> Status {
    status
        .metadata_mut()
        .insert(MEDIA_CODE_METADATA_KEY, MetadataValue::from_static(code));
    status
}

/// The code `status` carries, if any; see [`MEDIA_CODE_METADATA_KEY`].
pub fn media_code(status: &Status) -> Option<&str> {
    status
        .metadata()
        .get(MEDIA_CODE_METADATA_KEY)?
        .to_str()
        .ok()
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
    fn a_code_survives_the_status_it_is_stamped_on() {
        let status = stamp_media_code(
            WorkerMediaFault::Client.stamp(Status::invalid_argument("over the cap")),
            "media_too_large",
        );
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(media_code(&status), Some("media_too_large"));
        assert_eq!(
            WorkerMediaFault::from_status(&status),
            Some(WorkerMediaFault::Client)
        );
        assert_eq!(media_code(&Status::invalid_argument("plain")), None);
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

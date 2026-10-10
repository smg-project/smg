use std::time::Duration;

use thiserror::Error;

pub type MultiModalResult<T> = Result<T, MultiModalError>;

/// Errors that can occur while transforming media into encoder inputs.
#[derive(Debug, Error)]
pub enum TransformError {
    #[error("Invalid tensor shape: expected {expected}, got {actual:?}")]
    InvalidShape {
        expected: String,
        actual: Vec<usize>,
    },

    #[error("Empty batch: cannot stack zero tensors")]
    EmptyBatch,

    #[error("Inconsistent tensor shapes in batch")]
    InconsistentShapes,

    #[error("Shape error: {0}")]
    ShapeError(String),
}

#[derive(Debug, Error)]
pub enum MediaConnectorError {
    #[error("max_long_side_pixel must be a positive multiple of {factor}, got {value}")]
    InvalidMaxLongSidePixel { value: u32, factor: u32 },
    #[error("fps must be between {min} and {max}, got {value}")]
    InvalidSampleFps { value: f64, min: f64, max: f64 },
    #[error(
        "video max_long_side_pixel must be a multiple of {factor} within {min}..={max}, got {value}"
    )]
    InvalidVideoLongSideCap {
        value: u32,
        factor: u32,
        min: u32,
        max: u32,
    },
    #[error("unsupported media scheme: {0}")]
    UnsupportedScheme(String),
    #[error("invalid media URL: {0}")]
    InvalidUrl(String),
    #[error("media domain '{0}' is not in the allow list")]
    DisallowedDomain(String),
    #[error("local media path is not allowed: {0}")]
    DisallowedLocalPath(String),
    #[error("HTTP error while fetching media: {0}")]
    Http(#[from] reqwest::Error),
    #[error("I/O error while reading media: {0}")]
    Io(#[from] std::io::Error),
    #[error("base64 decode error: {0}")]
    Base64Decode(#[from] base64::DecodeError),
    #[error("data URL parse error: {0}")]
    DataUrl(String),
    #[error("{media} input payload exceeds the maximum size of {limit} bytes")]
    PayloadTooLarge { media: &'static str, limit: usize },
    #[error("media decode task failed: {0}")]
    Blocking(#[from] tokio::task::JoinError),
    #[error("image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("audio decode error: {0}")]
    AudioDecode(String),
    #[error("video decode error: {0}")]
    VideoDecode(String),
    #[error("media fetch of {url} timed out after {budget:?}")]
    Timeout { url: String, budget: Duration },
}

#[derive(Debug, Error)]
pub enum MultiModalError {
    #[error(transparent)]
    Media(#[from] MediaConnectorError),
    #[error("unsupported content part: {0}")]
    UnsupportedContent(&'static str),
    #[error("tracker task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("tracker validation error: {0}")]
    Validation(String),
}

/// Whose fault a [`MediaConnectorError`] is: it decides how the request is
/// answered and whether the process that fetched is at fault at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaFault {
    /// The request's own: a URL that cannot be parsed or is not allowed, a
    /// host that cannot be resolved or refuses the connection, a host that
    /// answers 4xx, a local file that cannot be read, a payload that is not
    /// the media it claims to be or is too large, a media parameter out of
    /// range.
    Client,
    /// The media host's, for now: a fetch that ran out of its budget, a 5xx
    /// answer, a transfer that broke while the body arrived.
    Transient,
    /// This process's: a decode task that panicked or was cancelled.
    Internal,
}

impl MediaConnectorError {
    /// Whose fault this error is; see [`MediaFault`].
    pub fn fault(&self) -> MediaFault {
        match self {
            Self::Http(http) => http_fault(http),
            Self::Timeout { .. } => MediaFault::Transient,
            Self::Blocking(_) => MediaFault::Internal,
            Self::InvalidMaxLongSidePixel { .. }
            | Self::InvalidSampleFps { .. }
            | Self::InvalidVideoLongSideCap { .. }
            | Self::UnsupportedScheme(_)
            | Self::InvalidUrl(_)
            | Self::DisallowedDomain(_)
            | Self::DisallowedLocalPath(_)
            | Self::Io(_)
            | Self::Base64Decode(_)
            | Self::DataUrl(_)
            | Self::PayloadTooLarge { .. }
            | Self::Image(_)
            | Self::AudioDecode(_)
            | Self::VideoDecode(_) => MediaFault::Client,
        }
    }
}

/// Whose fault an HTTP fetch error is. A host that answered is judged by its
/// status; one that did not answer in time is the host's for now; one that
/// cannot be resolved or refuses the connection, and a redirect that leads
/// nowhere, are the request's; a transfer that broke before the last byte is
/// the host's.
fn http_fault(error: &reqwest::Error) -> MediaFault {
    if let Some(status) = error.status() {
        return if status.is_server_error() {
            MediaFault::Transient
        } else {
            MediaFault::Client
        };
    }
    if error.is_timeout() {
        MediaFault::Transient
    } else if error.is_connect() || error.is_redirect() || error.is_builder() {
        MediaFault::Client
    } else if error.is_body() || error.is_decode() {
        MediaFault::Transient
    } else {
        MediaFault::Client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference the request got wrong, media that is not what it claims
    /// to be, a payload over the cap: the request's own fault.
    #[test]
    fn a_fault_of_the_request_is_the_clients() {
        for error in [
            MediaConnectorError::UnsupportedScheme("ftp".to_string()),
            MediaConnectorError::InvalidUrl("not a url".to_string()),
            MediaConnectorError::DisallowedDomain("media.example".to_string()),
            MediaConnectorError::DataUrl("missing comma in data url".to_string()),
            MediaConnectorError::PayloadTooLarge {
                media: "image",
                limit: 1,
            },
            MediaConnectorError::Image(image::ImageError::Decoding(
                image::error::DecodingError::new(
                    image::error::ImageFormatHint::Unknown,
                    "not an image",
                ),
            )),
            MediaConnectorError::VideoDecode("not a video".to_string()),
            MediaConnectorError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)),
        ] {
            assert_eq!(error.fault(), MediaFault::Client, "{error}");
        }
    }

    /// A decode task that never answers is this process's fault.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "a test task, aborted at once for its join error"
    )]
    async fn a_decode_task_that_dies_is_ours() {
        let task = tokio::spawn(std::future::pending::<()>());
        task.abort();
        let join = task.await.expect_err("aborted");
        assert_eq!(
            MediaConnectorError::Blocking(join).fault(),
            MediaFault::Internal
        );
    }
}

use std::{net::IpAddr, time::Duration};

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
    /// The host names or resolves to an address media is not fetched from
    /// (loopback, link-local, private, shared address space, unspecified,
    /// multicast, broadcast) and private media hosts are not allowed.
    #[error(
        "media host '{host}' resolves to {address}, a {class} address, which media is not \
         fetched from unless private media hosts are allowed"
    )]
    DisallowedAddress {
        host: String,
        address: IpAddr,
        class: &'static str,
    },
    #[error("media host '{host}' could not be resolved: {source}")]
    UnresolvableHost {
        host: String,
        #[source]
        source: std::io::Error,
    },
    /// A redirect to a target the host policy refuses, or that is no media
    /// URL; `reason` is the refusal in the policy's words.
    #[error("the fetch of {url} was redirected to {target}, which is refused: {reason}")]
    DisallowedRedirect {
        url: String,
        target: String,
        reason: String,
    },
    #[error("the fetch of {url} was redirected more than {limit} times")]
    TooManyRedirects { url: String, limit: usize },
    #[error("local media path is not allowed: {0}")]
    DisallowedLocalPath(String),
    #[error("HTTP error while fetching media: {0}")]
    Http(#[from] reqwest::Error),
    /// The local file the request names could not be read: the request's
    /// own (see [`MediaFault::Client`]).
    #[error("I/O error while reading media: {0}")]
    Io(#[source] std::io::Error),
    #[error("base64 decode error: {0}")]
    Base64Decode(#[from] base64::DecodeError),
    #[error("data URL parse error: {0}")]
    DataUrl(String),
    #[error("{media} input payload exceeds the maximum size of {limit} bytes")]
    PayloadTooLarge { media: &'static str, limit: usize },
    #[error("media decode task failed: {0}")]
    Blocking(#[from] tokio::task::JoinError),
    /// This process's own I/O after the bytes arrived: the scratch file a
    /// decoder reads from, the pipe of the decoder process it runs (see
    /// [`MediaFault::Internal`]); never the request's or the media host's.
    #[error("I/O error while decoding media: {0}")]
    DecodeIo(#[source] std::io::Error),
    /// A decoder this process cannot run: not installed, not executable, or
    /// a backend this build does not have. This process's own (see
    /// [`MediaFault::Internal`]), whatever clip the request sent; and a
    /// verdict on nothing: as the fallback of a decoder that did run, it
    /// leaves that decoder's verdict standing (see [`Self::fallback_failed`]).
    #[error("media decoder unavailable: {0}")]
    DecoderUnavailable(#[source] std::io::Error),
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
    /// The request's own: a URL that cannot be parsed or is not allowed (by
    /// name, by the address it leads to, or by where it redirects), a host
    /// that cannot be resolved or refuses the connection, a host that
    /// answers 4xx, a local file the request names that cannot be read, a
    /// payload that is not the media it claims to be or is too large, a
    /// media parameter out of range. The host's reachability is judged from
    /// this side: a resolver, proxy or trust store that is broken here makes
    /// every fetch look like the request's fault, so such a worker stays in
    /// rotation and each refusal is logged with its fault and the connector's
    /// words; charging it to the worker would let any client's URL open the
    /// worker's breaker.
    Client,
    /// The media host's, for now: a fetch that ran out of its budget, a 5xx
    /// answer, a transfer that broke while the body arrived.
    Transient,
    /// This process's: a decode task that panicked or was cancelled, a
    /// scratch file or a decoder process that failed on this side (a
    /// temporary directory that is full or not writable, a pipe that broke),
    /// a decoder this side cannot start (not installed, not executable, a
    /// backend this build does not have), unless it was the fallback of a
    /// decoder that did run: that decoder's verdict on the clip stands.
    Internal,
}

impl MediaConnectorError {
    /// Whose fault this error is; see [`MediaFault`]. An I/O error is judged
    /// by where it happened, never by its kind: on the local file the request
    /// names it is the request's ([`Self::Io`]), on this process's scratch
    /// file or decoder process it is this process's ([`Self::DecodeIo`]), as
    /// is a decoder this process cannot start ([`Self::DecoderUnavailable`]).
    pub fn fault(&self) -> MediaFault {
        match self {
            Self::Http(http) => http_fault(http),
            Self::Timeout { .. } => MediaFault::Transient,
            Self::Blocking(_) | Self::DecodeIo(_) | Self::DecoderUnavailable(_) => {
                MediaFault::Internal
            }
            Self::InvalidMaxLongSidePixel { .. }
            | Self::InvalidSampleFps { .. }
            | Self::InvalidVideoLongSideCap { .. }
            | Self::UnsupportedScheme(_)
            | Self::InvalidUrl(_)
            | Self::DisallowedDomain(_)
            | Self::DisallowedAddress { .. }
            | Self::UnresolvableHost { .. }
            | Self::DisallowedRedirect { .. }
            | Self::TooManyRedirects { .. }
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

    /// The error of a decoder this process cannot run, in the operator's
    /// words: not installed, not executable, or a backend this build does
    /// not have ([`Self::DecoderUnavailable`]).
    pub fn decoder_unavailable(message: String) -> Self {
        Self::DecoderUnavailable(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            message,
        ))
    }

    /// The error of a decode whose fallback failed too, with both decoders'
    /// words in `message`; `media` builds the error for the media's fault
    /// ([`Self::VideoDecode`], [`Self::AudioDecode`]). The decoder that
    /// judged the clip last decides whose fault the failure is: the
    /// fallback's, unless the fallback could not start
    /// ([`Self::DecoderUnavailable`]) and judged nothing, in which case the
    /// first decoder's verdict stands. A verdict of this process's own I/O or
    /// task keeps the error this process's ([`Self::DecodeIo`]); two decoders
    /// that could not start leave it unavailable, so a decode that runs this
    /// one as its fallback reads it the same way.
    pub fn fallback_failed(
        first: Self,
        fallback: Self,
        message: String,
        media: fn(String) -> Self,
    ) -> Self {
        let verdict = match fallback {
            Self::DecoderUnavailable(_) => first,
            fallback => fallback,
        };
        match verdict {
            Self::DecoderUnavailable(io) => {
                Self::DecoderUnavailable(std::io::Error::new(io.kind(), message))
            }
            Self::DecodeIo(io) => Self::DecodeIo(std::io::Error::new(io.kind(), message)),
            verdict if verdict.fault() == MediaFault::Internal => {
                Self::DecodeIo(std::io::Error::other(message))
            }
            _ => media(message),
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
            MediaConnectorError::DisallowedAddress {
                host: "media.example".to_string(),
                address: IpAddr::from([10, 0, 0, 5]),
                class: "private",
            },
            MediaConnectorError::UnresolvableHost {
                host: "media.invalid".to_string(),
                source: std::io::Error::from(std::io::ErrorKind::NotFound),
            },
            MediaConnectorError::DisallowedRedirect {
                url: "http://media.example/a.png".to_string(),
                target: "http://10.0.0.5/a.png".to_string(),
                reason: "private".to_string(),
            },
            MediaConnectorError::TooManyRedirects {
                url: "http://media.example/a.png".to_string(),
                limit: 10,
            },
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

    /// An I/O error is judged by where it happened, not by its kind: on this
    /// process's scratch file or decoder process it is this process's fault,
    /// on the local file the request names it is the request's, whatever the
    /// kind.
    #[test]
    fn io_is_classified_by_where_it_failed_not_by_its_kind() {
        for kind in [
            std::io::ErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::BrokenPipe,
            std::io::ErrorKind::Other,
        ] {
            assert_eq!(
                MediaConnectorError::DecodeIo(std::io::Error::from(kind)).fault(),
                MediaFault::Internal,
                "{kind:?}"
            );
            assert_eq!(
                MediaConnectorError::DecoderUnavailable(std::io::Error::from(kind)).fault(),
                MediaFault::Internal,
                "{kind:?}"
            );
            assert_eq!(
                MediaConnectorError::Io(std::io::Error::from(kind)).fault(),
                MediaFault::Client,
                "{kind:?}"
            );
        }
    }

    /// A decode whose fallback failed too is judged by the decoder that
    /// judged the clip last: the fallback, or the first decoder when the
    /// fallback could not start and judged nothing. A verdict of this
    /// process's own I/O keeps the error this process's; two decoders that
    /// could not start leave it unavailable, which a decode that ran them as
    /// its fallback reads the same way. The same for video and audio.
    #[test]
    fn a_fallback_that_could_not_start_leaves_the_first_decoders_verdict() {
        use std::io::{Error, ErrorKind};
        let message = || "first: bad frame; fallback: ffmpeg executable not found".to_string();
        let unavailable =
            || MediaConnectorError::decoder_unavailable("ffmpeg executable not found".to_string());
        let bad_clip = || MediaConnectorError::VideoDecode("bad frame".to_string());
        let scratch = || MediaConnectorError::DecodeIo(Error::other("no space left on device"));
        let compose = |first, fallback, media| {
            MediaConnectorError::fallback_failed(first, fallback, message(), media)
        };

        for media in [
            MediaConnectorError::VideoDecode as fn(String) -> MediaConnectorError,
            MediaConnectorError::AudioDecode,
        ] {
            // The fallback ran and judged the clip: its verdict, whatever the
            // first decoder's failure was.
            let theirs = compose(scratch(), bad_clip(), media);
            assert_eq!(theirs.fault(), MediaFault::Client, "{theirs}");
            assert!(theirs.to_string().contains(&message()), "{theirs}");
            // The fallback failed on this side: this process's.
            let ours = compose(bad_clip(), scratch(), media);
            assert!(
                matches!(&ours, MediaConnectorError::DecodeIo(io) if io.kind() == ErrorKind::Other),
                "{ours}"
            );
            assert_eq!(ours.fault(), MediaFault::Internal);
            // The fallback could not start: the first decoder's verdict.
            let judged_by_first = compose(bad_clip(), unavailable(), media);
            assert_eq!(
                judged_by_first.fault(),
                MediaFault::Client,
                "{judged_by_first}"
            );
            assert!(
                judged_by_first.to_string().contains(&message()),
                "{judged_by_first}"
            );
            let first_was_ours = compose(scratch(), unavailable(), media);
            assert!(
                matches!(first_was_ours, MediaConnectorError::DecodeIo(_)),
                "{first_was_ours}"
            );
            assert_eq!(first_was_ours.fault(), MediaFault::Internal);
            // Neither could start: unavailable, so a decode that ran them as
            // its fallback is judged by its own first decoder.
            let neither = compose(unavailable(), unavailable(), media);
            assert!(
                matches!(&neither, MediaConnectorError::DecoderUnavailable(io) if io.kind() == ErrorKind::Unsupported),
                "{neither}"
            );
            assert_eq!(neither.fault(), MediaFault::Internal);
            let outer = compose(bad_clip(), neither, media);
            assert_eq!(outer.fault(), MediaFault::Client, "{outer}");
        }
    }
}

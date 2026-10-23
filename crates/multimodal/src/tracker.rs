use std::{collections::HashMap, future::Future, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MediaConnectorError, MultiModalError, MultiModalResult},
    media::{
        FetchSource, FrameSampling, ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig,
    },
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};
use crate::types::TransparencyPolicy;

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

/// One media slot of a request: either its own fetch, or the same media a
/// slot before it already fetches.
enum Slot {
    Fetch(PendingTask),
    SameAs(usize),
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

/// What a fetch is looked up by before its payload is compared: the
/// modality, the fetch settings, and the payload's kind and length. Hashing
/// the payload itself would cost a pass over every data URL of the request;
/// comparing it costs a pass only over actual duplicates (and over the first
/// few distinct payloads of one length, see `COMPARE_CANDIDATES`).
#[derive(Debug, PartialEq, Eq, Hash)]
struct FetchKey {
    modality: Modality,
    settings: String,
    kind: u8,
    len: usize,
}

/// How many distinct payloads of one `FetchKey` a new part is compared
/// against byte for byte. Past that the bucket is matched by digest, which
/// keeps a request full of same-length, long-shared-prefix payloads at one
/// pass over each payload instead of a pass per earlier candidate.
const COMPARE_CANDIDATES: usize = 4;

/// The fetching slots of one `FetchKey`, remembered so later parts can be
/// matched against them.
#[derive(Default)]
struct Bucket {
    /// The first distinct payloads, kept so parts can be compared against
    /// them byte for byte.
    compared: Vec<Compared>,
    /// Later distinct payloads, kept as their digest only: nothing compares
    /// their bytes again, so the table does not hold them past their fetch.
    hashed: Vec<Hashed>,
}

struct Compared {
    source: Arc<FetchSource>,
    slot: usize,
}

struct Hashed {
    digest: [u8; 32],
    slot: usize,
}

/// Where a fetched item's bytes come from, for a [`FetchReport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchSourceKind {
    /// A URL fetched over the network.
    Url,
    /// A `data:` URL carried in the request.
    DataUrl,
    /// Bytes carried in the request.
    InlineBytes,
    /// A file of the host.
    File,
}

impl FetchSourceKind {
    fn of(source: &FetchSource) -> Self {
        match source {
            FetchSource::Url(_) => Self::Url,
            FetchSource::DataUrl(_) => Self::DataUrl,
            FetchSource::InlineBytes(_) => Self::InlineBytes,
            FetchSource::File(_) => Self::File,
        }
    }

    /// The kind a URL part of a request is fetched as: a `data:` URL (the
    /// tracker's rule: the scheme, however spelled) or a URL fetched over the
    /// network.
    #[must_use]
    pub fn of_url(url: &str) -> Self {
        if is_data_url(url, "data:") {
            Self::DataUrl
        } else {
            Self::Url
        }
    }

    /// The label value: `url`, `data_url`, `inline` or `file`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::DataUrl => "data_url",
            Self::InlineBytes => "inline",
            Self::File => "file",
        }
    }
}

/// How a fetch ended, for a [`FetchReport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Fetched and decoded.
    Ok,
    /// The payload, or what it decodes to, is over a size limit.
    TooLarge,
    /// The caller's fault: a URL that cannot be fetched, bytes that do not
    /// decode, a host the policy refuses.
    ClientFault,
    /// Not the caller's fault: a fetch that timed out, a host that failed,
    /// a decoder this process lacks.
    HostFault,
}

impl FetchOutcome {
    fn of(result: &MultiModalResult<TrackedMedia>) -> Self {
        match result {
            Ok(_) => Self::Ok,
            Err(MultiModalError::Media(error)) => {
                if error.code() == crate::error::MediaErrorCode::TooLarge {
                    Self::TooLarge
                } else if error.fault() == crate::error::MediaFault::Client {
                    Self::ClientFault
                } else {
                    Self::HostFault
                }
            }
            Err(_) => Self::ClientFault,
        }
    }

    /// The label value: `ok`, `too_large`, `client_fault` or `host_fault`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::TooLarge => "too_large",
            Self::ClientFault => "client_fault",
            Self::HostFault => "host_fault",
        }
    }
}

/// One fetch, as the tracker reports it when the fetch ends: the modality,
/// where the bytes came from, how it ended, how long the fetch and decode
/// took, and the bytes it brought in (none when it failed). A part that
/// repeats another part of the same request is not a fetch and is not
/// reported.
#[derive(Clone, Copy, Debug)]
pub struct FetchReport {
    pub modality: Modality,
    pub source: FetchSourceKind,
    pub outcome: FetchOutcome,
    pub duration: std::time::Duration,
    pub bytes: usize,
}

/// A fetch observer: called once per fetch as it ends, on the fetch's own
/// task, with the label the tracker was given (a request's model, say).
pub type FetchObserver = fn(label: &str, report: &FetchReport);

/// The bytes a fetched item brought in.
fn fetched_bytes(media: &TrackedMedia) -> usize {
    match media {
        TrackedMedia::Image(frame) => frame.raw_bytes.len(),
        TrackedMedia::Video(clip) => clip.raw_bytes.len(),
        TrackedMedia::Audio(clip) => clip.raw_bytes.len(),
        TrackedMedia::Embeddings => 0,
    }
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    /// Where every fetch is reported, under the label, when set.
    fetch_observer: Option<(Arc<str>, FetchObserver)>,
    pending: HashMap<Modality, Vec<Slot>>,
    uuids: MultiModalUUIDs,
    first_slot: HashMap<FetchKey, Bucket>,
    /// Frame rate to sample a video at when the request names none; `None`
    /// keeps the connector default.
    default_video_sample_fps: Option<f64>,
    video_frame_sampling: FrameSampling,
    /// The policy the fetched image frames decode an alpha channel under (see
    /// [`TransparencyPolicy`]); `KeepAlpha` unless the request's processing
    /// stage set the destination engine's.
    image_transparency: TransparencyPolicy,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            fetch_observer: None,
            pending: HashMap::new(),
            uuids: HashMap::new(),
            first_slot: HashMap::new(),
            default_video_sample_fps: None,
            video_frame_sampling: FrameSampling::default(),
            image_transparency: TransparencyPolicy::KeepAlpha,
        }
    }

    /// Decode the request's image frames under `policy` (the destination
    /// engine's own transparency rule); it becomes part of each frame's hash.
    pub fn with_image_transparency(mut self, policy: TransparencyPolicy) -> Self {
        self.image_transparency = policy;
        self
    }

    /// Report every fetch to `observer` under `label` (see [`FetchReport`]).
    #[must_use]
    pub fn with_fetch_observer(mut self, label: Arc<str>, observer: FetchObserver) -> Self {
        self.fetch_observer = Some((label, observer));
        self
    }

    /// Sample videos that name no `fps` at this rate (the model's reference
    /// default) instead of the connector default.
    pub fn with_default_video_sample_fps(mut self, fps: Option<f64>) -> Self {
        self.default_video_sample_fps = fps;
        self
    }

    /// Place sampled video frames the way the model's reference does.
    pub fn with_video_frame_sampling(mut self, sampling: FrameSampling) -> Self {
        self.video_frame_sampling = sampling;
        self
    }

    pub fn push_part(&mut self, part: MediaContentPart) -> MultiModalResult<()> {
        match part {
            MediaContentPart::Text { .. } => {}
            MediaContentPart::ImageUrl {
                url,
                detail,
                uuid,
                max_long_side_pixel,
            } => {
                let source = media_url_source(url, "data:image/");
                self.enqueue_image(
                    source,
                    detail.unwrap_or_default(),
                    uuid,
                    max_long_side_pixel,
                );
            }
            MediaContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                    None,
                );
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
            MediaContentPart::AudioUrl { url, uuid } => {
                let source = media_url_source(url, "data:audio/");
                self.enqueue_audio(source, uuid);
            }
            MediaContentPart::AudioData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_audio(MediaSource::InlineBytes(data), uuid);
            }
            MediaContentPart::VideoUrl {
                url,
                uuid,
                fps,
                max_long_side_pixel,
            } => {
                let source = media_url_source(url, "data:video/");
                self.enqueue_video(source, uuid, fps, max_long_side_pixel)?;
            }
            MediaContentPart::VideoData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_video(MediaSource::InlineBytes(data), uuid, None, None)?;
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, slots) in self.pending.drain() {
            let mut items: Vec<TrackedMedia> = Vec::with_capacity(slots.len());
            for slot in slots {
                let media = match slot {
                    Slot::Fetch(task) => task.await??,
                    Slot::SameAs(first) => items.get(first).cloned().ok_or_else(|| {
                        MultiModalError::Validation(format!(
                            "{modality} slot refers to a slot that was never fetched"
                        ))
                    })?,
                };
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    /// The slot that already fetches this media with these settings, if an
    /// earlier part named it; otherwise the next slot is claimed for it.
    fn same_media_as(
        &mut self,
        modality: Modality,
        settings: String,
        source: &Arc<FetchSource>,
    ) -> Option<usize> {
        let next = self.pending.entry(modality).or_default().len();
        let key = FetchKey {
            modality,
            settings,
            kind: source.kind(),
            len: source.len(),
        };
        let bucket = self.first_slot.entry(key).or_default();
        // The first few distinct payloads are compared byte for byte: a repeat
        // (the common case) matches the first one, a different payload usually
        // differs within its first bytes. A part that differs from all of them
        // cannot match them later either, so they never need a digest.
        if let Some(first) = bucket
            .compared
            .iter()
            .find(|candidate| *candidate.source == **source)
        {
            return Some(first.slot);
        }
        if bucket.compared.len() < COMPARE_CANDIDATES {
            bucket.compared.push(Compared {
                source: Arc::clone(source),
                slot: next,
            });
            return None;
        }
        // Past them the bucket is matched by digest: one pass over this
        // payload, and only its digest is kept.
        let digest: [u8; 32] = blake3::hash(source.payload()).into();
        if let Some(first) = bucket
            .hashed
            .iter()
            .find(|candidate| candidate.digest == digest)
        {
            return Some(first.slot);
        }
        bucket.hashed.push(Hashed { digest, slot: next });
        None
    }

    fn enqueue_image(
        &mut self,
        source: MediaSource,
        detail: ImageDetail,
        uuid: Option<String>,
        max_long_side_pixel: Option<u32>,
    ) {
        let config = ImageFetchConfig {
            detail,
            max_long_side_pixel,
            transparency: self.image_transparency,
        };
        self.enqueue(
            Modality::Image,
            uuid,
            format!("{config:?}"),
            source,
            move |connector, source| async move {
                let frame = connector.fetch_image_from(&source, config).await?;
                Ok(TrackedMedia::Image(frame))
            },
        );
    }

    fn enqueue_video(
        &mut self,
        source: MediaSource,
        uuid: Option<String>,
        fps: Option<f64>,
        max_long_side_pixel: Option<u32>,
    ) -> MultiModalResult<()> {
        let cfg = video_fetch_config(
            fps,
            max_long_side_pixel,
            self.default_video_sample_fps,
            self.video_frame_sampling,
        )?;
        self.enqueue(
            Modality::Video,
            uuid,
            format!("{cfg:?}"),
            source,
            move |connector, source| async move {
                let clip = connector.fetch_video_from(&source, cfg).await?;
                Ok(TrackedMedia::Video(clip))
            },
        );
        Ok(())
    }

    fn enqueue_audio(&mut self, source: MediaSource, uuid: Option<String>) {
        self.enqueue(
            Modality::Audio,
            uuid,
            String::new(),
            source,
            |connector, source| async move {
                let clip = connector.fetch_audio_from(&source).await?;
                Ok(TrackedMedia::Audio(clip))
            },
        );
    }

    /// The slot of one media item: its uuid recorded, then either the fetch an
    /// earlier item of the same modality, settings and bytes already runs, or
    /// its own fetch spawned now and awaited in [`Self::finalize`].
    fn enqueue<F, Fut>(
        &mut self,
        modality: Modality,
        uuid: Option<String>,
        settings: String,
        source: MediaSource,
        fetch: F,
    ) where
        F: FnOnce(Arc<MediaConnector>, Arc<FetchSource>) -> Fut,
        Fut: Future<Output = MultiModalResult<TrackedMedia>> + Send + 'static,
    {
        self.uuids.entry(modality).or_default().push(uuid);

        let source = Arc::new(FetchSource::from(source));
        let slot = match self.same_media_as(modality, settings, &source) {
            Some(first) => Slot::SameAs(first),
            None => {
                let connector = Arc::clone(&self.media_connector);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
                )]
                let handle = match &self.fetch_observer {
                    None => tokio::spawn(fetch(connector, source)),
                    Some((label, observer)) => {
                        let (label, observer) = (Arc::clone(label), *observer);
                        let kind = FetchSourceKind::of(&source);
                        let fetching = fetch(connector, source);
                        tokio::spawn(async move {
                            let started = std::time::Instant::now();
                            let result = fetching.await;
                            observer(
                                &label,
                                &FetchReport {
                                    modality,
                                    source: kind,
                                    outcome: FetchOutcome::of(&result),
                                    duration: started.elapsed(),
                                    bytes: result.as_ref().map_or(0, fetched_bytes),
                                },
                            );
                            result
                        })
                    }
                };
                Slot::Fetch(handle)
            }
        };
        self.pending.entry(modality).or_default().push(slot);
    }
}

// Avoid scanning and allocating the entire payload just to identify its scheme.
// Only canonical opaque data URLs of the part's own media type take this path
// (`data:image/`, `data:audio/`, `data:video/`); noncanonical forms and
// invalid data:// authorities retain URL parsing. Connector validation still
// receives the original input unchanged.
fn media_url_source(url: String, canonical_data_prefix: &str) -> MediaSource {
    if is_data_url(&url, canonical_data_prefix) {
        MediaSource::DataUrl(url)
    } else {
        MediaSource::Url(url)
    }
}

/// Whether `url` is a `data:` URL: the canonical prefix, or the scheme as the
/// URL parser reads it (any spelling of its case).
fn is_data_url(url: &str, canonical_data_prefix: &str) -> bool {
    url.starts_with(canonical_data_prefix)
        || url::Url::parse(url).is_ok_and(|parsed| parsed.scheme() == "data")
}

/// The fetch settings for one video: the request's `fps` when given (and
/// valid), else the model's default, else the connector default; plus the
/// long-side cap when given and the model's frame placement.
fn video_fetch_config(
    fps: Option<f64>,
    max_long_side_pixel: Option<u32>,
    default_sample_fps: Option<f64>,
    sampling: FrameSampling,
) -> MultiModalResult<VideoFetchConfig> {
    let mut cfg = VideoFetchConfig {
        sampling,
        ..VideoFetchConfig::default()
    };
    match fps {
        Some(fps) => cfg.sample_fps = validate_sample_fps(fps)?,
        None => {
            if let Some(default) = default_sample_fps {
                cfg.sample_fps = validate_sample_fps(default)?;
            }
        }
    }
    if let Some(cap) = max_long_side_pixel {
        validate_video_long_side_cap(cap)?;
        cfg.max_long_side_pixel = Some(cap);
    }
    Ok(cfg)
}

/// Lowest sampling rate MiniMax-M3 accepts for a video clip.
pub const MIN_SAMPLE_FPS: f64 = 0.2;
/// Highest sampling rate MiniMax-M3 accepts for a video clip.
pub const MAX_SAMPLE_FPS: f64 = 5.0;
/// Smallest per-frame long-side cap MiniMax-M3 accepts.
pub const MIN_VIDEO_LONG_SIDE: u32 = 150;
/// Largest per-frame long-side cap MiniMax-M3 accepts.
pub const MAX_VIDEO_LONG_SIDE: u32 = 3584;
/// Vision patch factor the per-frame cap must align to.
pub const VIDEO_LONG_SIDE_FACTOR: u32 = 28;

/// Reject a sampling rate outside M3's accepted range.
fn validate_sample_fps(value: f64) -> MultiModalResult<f64> {
    if !value.is_finite() || !(MIN_SAMPLE_FPS..=MAX_SAMPLE_FPS).contains(&value) {
        return Err(MediaConnectorError::InvalidSampleFps {
            value,
            min: MIN_SAMPLE_FPS,
            max: MAX_SAMPLE_FPS,
        }
        .into());
    }
    Ok(value)
}

/// Reject a per-frame long-side cap that is out of range or off the patch grid.
fn validate_video_long_side_cap(value: u32) -> MultiModalResult<()> {
    if !(MIN_VIDEO_LONG_SIDE..=MAX_VIDEO_LONG_SIDE).contains(&value)
        || !value.is_multiple_of(VIDEO_LONG_SIDE_FACTOR)
    {
        return Err(MediaConnectorError::InvalidVideoLongSideCap {
            value,
            factor: VIDEO_LONG_SIDE_FACTOR,
            min: MIN_VIDEO_LONG_SIDE,
            max: MAX_VIDEO_LONG_SIDE,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod video_param_tests {
    use super::*;
    use crate::media::sampled_frame_indices;

    #[test]
    fn a_request_fps_wins_then_the_model_default_then_the_connector_default() {
        let requested =
            video_fetch_config(Some(0.5), None, Some(1.0), FrameSampling::Even).unwrap();
        assert_eq!(requested.sample_fps, 0.5);

        let model_default =
            video_fetch_config(None, Some(1008), Some(1.0), FrameSampling::Interval).unwrap();
        assert_eq!(model_default.sample_fps, 1.0);
        assert_eq!(model_default.max_long_side_pixel, Some(1008));
        assert_eq!(model_default.sampling, FrameSampling::Interval);

        let connector_default = video_fetch_config(None, None, None, FrameSampling::Even).unwrap();
        assert_eq!(
            connector_default.sample_fps,
            VideoFetchConfig::default().sample_fps
        );

        // A model's own default is held to the same range as a requested one,
        // right up to the edge of it: nothing reaches sampling unchecked just
        // because the model named it rather than the caller.
        assert!(video_fetch_config(Some(100.0), None, Some(1.0), FrameSampling::Even).is_err());
        for bad_default in [100.0, 5.1, 0.19, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                video_fetch_config(None, None, Some(bad_default), FrameSampling::Even).is_err(),
                "{bad_default}"
            );
        }
    }

    /// The rate a request names, and the model's own default, reach the picks
    /// as the doubles they are: 0.3 frames a second over 20 frames at 3 fps
    /// picks the reference's `[0, 11]`, not the `[0, 10]` of the rate narrowed
    /// to binary32 on the way.
    #[test]
    fn the_rate_reaches_the_picks_as_the_double_it_is() {
        let sampling = FrameSampling::UpToThenRate {
            max_frames: 32,
            rate_max_frames: 2048,
        };
        let requested = video_fetch_config(Some(0.3), None, Some(2.0), sampling).unwrap();
        assert_eq!(sampled_frame_indices(20, 3.0, requested), [0, 11]);
        let model_default = video_fetch_config(None, None, Some(0.3), sampling).unwrap();
        assert_eq!(sampled_frame_indices(20, 3.0, model_default), [0, 11]);
    }

    #[test]
    fn accepts_the_documented_fps_range() {
        // The contract suite's valid tiers and both boundaries.
        for fps in [0.2, 0.5, 1.0, 2.0, 5.0] {
            assert!(validate_sample_fps(fps).is_ok(), "{fps}");
        }
    }

    #[test]
    fn rejects_fps_outside_the_range() {
        // 100 is the value the contract suite sends as clearly out of range.
        for fps in [100.0, 5.1, 0.19, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(validate_sample_fps(fps).is_err(), "{fps}");
        }
    }

    #[test]
    fn accepts_the_documented_long_side_tiers() {
        // 504 / 1008 / 2016 are the suite's low/default/high tiers.
        for cap in [168, 504, 1008, 2016, 3584] {
            assert!(validate_video_long_side_cap(cap).is_ok(), "{cap}");
        }
    }

    #[test]
    fn rejects_long_side_out_of_range_or_off_grid() {
        // 140 is below the minimum, 3612 above the maximum, 1009 off the grid.
        for cap in [0, 140, 3612, 1009] {
            assert!(validate_video_long_side_cap(cap).is_err(), "{cap}");
        }
    }
}

#[cfg(test)]
mod repeat_tests {

    use super::*;
    use crate::media::MediaConnectorConfig;

    const TINY_PNG_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";

    #[test]
    fn fast_data_scheme_classification_matches_url_parser() {
        for (prefix, mime) in [
            ("data:image/", "image/png"),
            ("data:audio/", "audio/wav"),
            ("data:video/", "video/mp4"),
        ] {
            let mut cases = vec![
                TINY_PNG_URL.to_owned(),
                format!("data:{mime};base64,abc"),
                "data:".to_owned(),
                format!("DATA:{mime};base64,abc"),
                " data:text/plain,hello".to_owned(),
                "https://example.com/image".to_owned(),
                "not-a-url".to_owned(),
                "data://[invalid];base64,abc".to_owned(),
                "data:\n//[invalid];base64,abc".to_owned(),
                "data://user:password@[invalid]/image;base64,abc".to_owned(),
                "data:;base64,abc".to_owned(),
            ];
            for byte in 0..=127u8 {
                cases.push(format!("data:{};base64,abc", char::from(byte)));
                cases.push(format!("data:text/plain,abc{}def", char::from(byte)));
                cases.push(format!("data:{mime};base64,abc{}def", char::from(byte)));
            }
            for input in cases {
                let expected = url::Url::parse(&input).is_ok_and(|url| url.scheme() == "data");
                match media_url_source(input.clone(), prefix) {
                    MediaSource::DataUrl(actual) => {
                        assert!(expected, "{input:?}");
                        assert_eq!(actual, input);
                    }
                    MediaSource::Url(actual) => {
                        assert!(!expected, "{input:?}");
                        assert_eq!(actual, input);
                    }
                    _ => panic!("unexpected media source"),
                }
            }
        }
    }

    fn tracker() -> AsyncMultiModalTracker {
        let connector = MediaConnector::new(
            crate::test_support::http_client_builder(),
            MediaConnectorConfig::default(),
        )
        .expect("default connector");
        AsyncMultiModalTracker::new(Arc::new(connector))
    }

    /// The plan side classifies a URL part as the fetch will: the scheme as
    /// the URL parser reads it, whatever its case, is a data URL.
    #[test]
    fn of_url_reads_the_scheme_as_the_fetch_does() {
        assert_eq!(
            FetchSourceKind::of_url("data:image/png;base64,AAAA"),
            FetchSourceKind::DataUrl
        );
        assert_eq!(
            FetchSourceKind::of_url("DATA:image/png;base64,AAAA"),
            FetchSourceKind::DataUrl
        );
        assert_eq!(
            FetchSourceKind::of_url("https://example.test/a.png"),
            FetchSourceKind::Url
        );
        assert_eq!(FetchSourceKind::of_url("not a url"), FetchSourceKind::Url);
    }

    /// Every fetch is reported once, with its outcome and its bytes, under
    /// the tracker's label; a part that repeats another is not a fetch.
    #[tokio::test]
    async fn the_fetch_observer_sees_each_fetch_once() {
        static REPORTS: std::sync::Mutex<Vec<(String, FetchReport)>> =
            std::sync::Mutex::new(Vec::new());
        fn observe(label: &str, report: &FetchReport) {
            REPORTS
                .lock()
                .expect("reports")
                .push((label.to_string(), *report));
        }
        let mut tracker = tracker().with_fetch_observer(Arc::from("m"), observe);
        tracker.push_part(image_part(None)).expect("a part");
        tracker
            .push_part(image_part(None))
            .expect("the same part again: no fetch of its own");
        tracker
            .push_part(MediaContentPart::ImageUrl {
                url: "data:image/png;base64,AAAA".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            })
            .expect("a part whose bytes do not decode");
        tracker
            .finalize()
            .await
            .expect_err("the undecodable part fails the request");
        let reports = REPORTS.lock().expect("reports").clone();
        assert_eq!(reports.len(), 2, "{reports:?}");
        assert!(reports.iter().all(|(label, _)| label == "m"));
        let fetched = reports
            .iter()
            .find(|(_, report)| report.outcome == FetchOutcome::Ok)
            .expect("the decoded fetch");
        assert_eq!(fetched.1.modality, Modality::Image);
        assert_eq!(fetched.1.source, FetchSourceKind::DataUrl);
        assert!(fetched.1.bytes > 0, "{:?}", fetched.1);
        let failed = reports
            .iter()
            .find(|(_, report)| report.outcome == FetchOutcome::ClientFault)
            .expect("the failed fetch");
        assert_eq!(failed.1.bytes, 0, "{:?}", failed.1);
    }

    fn image_part(max_long_side_pixel: Option<u32>) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: TINY_PNG_URL.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel,
        }
    }

    async fn images(tracker: AsyncMultiModalTracker) -> Vec<TrackedMedia> {
        tracker
            .finalize()
            .await
            .expect("every part resolves")
            .data
            .remove(&Modality::Image)
            .expect("the request carries images")
    }

    #[tokio::test]
    async fn the_tracker_s_transparency_policy_reaches_the_frame_and_its_hash() {
        use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
        let png = BASE64_STANDARD
            .decode(TINY_PNG_URL.split_once(',').expect("data url").1)
            .expect("base64");
        for policy in [
            TransparencyPolicy::KeepAlpha,
            TransparencyPolicy::CompositeOverWhite,
            TransparencyPolicy::DropAlpha,
        ] {
            let mut tracker = tracker().with_image_transparency(policy);
            tracker.push_part(image_part(None)).expect("part");
            let items = images(tracker).await;
            let Some(TrackedMedia::Image(frame)) = items.first() else {
                panic!("the slot must hold an image");
            };
            assert_eq!(frame.transparency(), policy);
            assert_eq!(
                frame.hash,
                crate::hasher::hash_image_with_transparency(&png, None, policy),
                "{policy:?}"
            );
        }
        let mut tracker = tracker();
        tracker.push_part(image_part(None)).expect("part");
        let items = images(tracker).await;
        let Some(TrackedMedia::Image(frame)) = items.first() else {
            panic!("the slot must hold an image");
        };
        assert_eq!(frame.transparency(), TransparencyPolicy::KeepAlpha);
        assert_eq!(frame.hash, crate::hasher::hash_image(&png));
    }

    #[tokio::test]
    async fn an_image_named_twice_is_fetched_once() {
        let mut tracker = tracker();
        tracker.push_part(image_part(None)).expect("first part");
        tracker.push_part(image_part(None)).expect("second part");

        let items = images(tracker).await;
        assert_eq!(items.len(), 2);
        let mut iter = items.iter();
        let (Some(TrackedMedia::Image(first)), Some(TrackedMedia::Image(second))) =
            (iter.next(), iter.next())
        else {
            panic!("both slots must hold an image");
        };
        assert!(Arc::ptr_eq(first, second));
    }

    #[tokio::test]
    async fn an_image_asked_for_at_two_sizes_is_fetched_twice() {
        let mut tracker = tracker();
        tracker.push_part(image_part(None)).expect("first part");
        tracker
            .push_part(image_part(Some(504)))
            .expect("second part");

        let items = images(tracker).await;
        assert_eq!(items.len(), 2);
        let mut iter = items.iter();
        let (Some(TrackedMedia::Image(first)), Some(TrackedMedia::Image(second))) =
            (iter.next(), iter.next())
        else {
            panic!("both slots must hold an image");
        };
        assert!(!Arc::ptr_eq(first, second));
    }

    #[test]
    fn a_fetch_is_shared_only_with_the_same_media_and_settings() {
        let mut tracker = tracker();
        let clip = Arc::new(FetchSource::Url("https://example.test/clip.mp4".into()));
        // Same length and kind, different bytes: never merged.
        let other = Arc::new(FetchSource::Url("https://example.test/clop.mp4".into()));

        assert_eq!(
            tracker.same_media_as(Modality::Video, "one".into(), &clip),
            None
        );
        assert_eq!(
            tracker.same_media_as(Modality::Video, "one".into(), &clip),
            Some(0)
        );
        assert_eq!(
            tracker.same_media_as(Modality::Video, "two".into(), &clip),
            None
        );
        assert_eq!(
            tracker.same_media_as(Modality::Image, "one".into(), &clip),
            None
        );
        assert_eq!(
            tracker.same_media_as(Modality::Video, "one".into(), &other),
            None
        );
        assert_eq!(
            tracker.same_media_as(Modality::Video, "one".into(), &other),
            Some(0)
        );
        assert_eq!(
            tracker.same_media_as(Modality::Video, "one".into(), &clip),
            Some(0)
        );
    }

    /// Distinct data URLs of one length that share everything but their last
    /// character: the worst case for comparison-based matching.
    fn same_length_variants(count: usize) -> Vec<String> {
        let mut base = TINY_PNG_URL.to_string();
        base.pop().expect("non-empty data url");
        (0..count)
            .map(|i| {
                format!(
                    "{base}{}",
                    char::from(b'A' + u8::try_from(i).expect("few variants"))
                )
            })
            .collect()
    }

    #[test]
    fn a_crowded_bucket_matches_by_digest_past_the_first_few_candidates() {
        let mut tracker = tracker();
        let sources: Vec<Arc<FetchSource>> = same_length_variants(10)
            .into_iter()
            .map(|url| Arc::new(FetchSource::DataUrl(url)))
            .collect();
        for source in &sources {
            assert_eq!(
                tracker.same_media_as(Modality::Image, String::new(), source),
                None,
                "every distinct payload fetches"
            );
            // Stand-in for the fetch slot the caller pushes; slots are indices
            // into this list.
            tracker
                .pending
                .entry(Modality::Image)
                .or_default()
                .push(Slot::SameAs(usize::MAX));
        }
        // Every repeat, in either region of the bucket, finds its first slot.
        for (slot, source) in sources.iter().enumerate().rev() {
            assert_eq!(
                tracker.same_media_as(Modality::Image, String::new(), source),
                Some(slot)
            );
        }
        // The first candidates are kept to be compared, so a part is compared
        // against at most COMPARE_CANDIDATES payloads; the rest are kept as a
        // digest and a slot only, so their bytes are not retained by the table.
        let bucket = tracker
            .first_slot
            .values()
            .find(|bucket| !bucket.hashed.is_empty())
            .expect("one bucket: same modality, settings, kind and length");
        assert_eq!(bucket.compared.len(), COMPARE_CANDIDATES);
        assert_eq!(bucket.hashed.len(), sources.len() - COMPARE_CANDIDATES);
        for (i, source) in sources.iter().enumerate() {
            let expected = if i < COMPARE_CANDIDATES { 2 } else { 1 };
            assert_eq!(
                Arc::strong_count(source),
                expected,
                "candidate {i}: only the compared ones are held by the table"
            );
        }
        // A lone payload of another length is neither compared nor hashed.
        let lone = Arc::new(FetchSource::DataUrl(format!("{TINY_PNG_URL}=")));
        assert_eq!(
            tracker.same_media_as(Modality::Image, String::new(), &lone),
            None
        );
        let lone_bucket = tracker
            .first_slot
            .values()
            .find(|bucket| bucket.compared.len() == 1)
            .expect("its own bucket");
        assert!(lone_bucket.hashed.is_empty());
        assert_eq!(Arc::strong_count(&lone), 2);
    }

    #[tokio::test]
    async fn many_same_length_data_urls_are_fetched_once_each() {
        let mut tracker = tracker();
        let variants = same_length_variants(8);
        for _ in 0..2 {
            for url in &variants {
                tracker
                    .push_part(MediaContentPart::ImageUrl {
                        url: url.clone(),
                        detail: None,
                        uuid: None,
                        max_long_side_pixel: None,
                    })
                    .expect("part");
            }
        }
        let slots = tracker.pending.get(&Modality::Image).expect("image slots");
        let fetches = slots
            .iter()
            .filter(|slot| matches!(slot, Slot::Fetch(_)))
            .count();
        let repeats: Vec<usize> = slots
            .iter()
            .filter_map(|slot| match slot {
                Slot::SameAs(first) => Some(*first),
                Slot::Fetch(_) => None,
            })
            .collect();
        assert_eq!(fetches, variants.len());
        assert_eq!(repeats, (0..variants.len()).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn data_urls_differing_in_one_byte_are_fetched_separately() {
        let mut tracker = tracker();
        let mut other = TINY_PNG_URL.to_string();
        // Flip the last base64 character so the payload differs but not its length.
        let last = other.pop().expect("non-empty data url");
        other.push(if last == 'A' { 'B' } else { 'A' });
        tracker.push_part(image_part(None)).expect("first part");
        tracker
            .push_part(MediaContentPart::ImageUrl {
                url: other,
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            })
            .expect("second part");

        let slots = tracker
            .pending
            .get(&Modality::Image)
            .expect("image slots")
            .iter()
            .filter(|slot| matches!(slot, Slot::Fetch(_)))
            .count();
        assert_eq!(slots, 2);
    }
}

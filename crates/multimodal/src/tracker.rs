use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MediaConnectorError, MultiModalError, MultiModalResult},
    media::{FrameSampling, ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

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

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    pending: HashMap<Modality, Vec<Slot>>,
    uuids: MultiModalUUIDs,
    first_slot: HashMap<[u8; 32], usize>,
    /// Frame rate to sample a video at when the request names none; `None`
    /// keeps the connector default.
    default_video_sample_fps: Option<f32>,
    video_frame_sampling: FrameSampling,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            pending: HashMap::new(),
            uuids: HashMap::new(),
            first_slot: HashMap::new(),
            default_video_sample_fps: None,
            video_frame_sampling: FrameSampling::default(),
        }
    }

    /// Sample videos that name no `fps` at this rate (the model's reference
    /// default) instead of the connector default.
    pub fn with_default_video_sample_fps(mut self, fps: Option<f32>) -> Self {
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
                let source = image_url_source(url);
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
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
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
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
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

    /// The slot that already fetches this media, if an earlier part named it;
    /// otherwise the next slot is claimed for it.
    fn same_media_as(&mut self, modality: Modality, key: [u8; 32]) -> Option<usize> {
        let next = self.pending.entry(modality).or_default().len();
        if let Some(&first) = self.first_slot.get(&key) {
            return Some(first);
        }
        self.first_slot.insert(key, next);
        None
    }

    fn enqueue_image(
        &mut self,
        source: MediaSource,
        detail: ImageDetail,
        uuid: Option<String>,
        max_long_side_pixel: Option<u32>,
    ) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let config = ImageFetchConfig {
            detail,
            max_long_side_pixel,
        };
        let key = fetch_key(modality, &format!("{config:?}"), &source);
        if let Some(first) = self.same_media_as(modality, key) {
            self.pending
                .entry(modality)
                .or_default()
                .push(Slot::SameAs(first));
            return;
        }

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector.fetch_image(source, config).await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending
            .entry(modality)
            .or_default()
            .push(Slot::Fetch(handle));
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

        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let key = fetch_key(modality, &format!("{cfg:?}"), &source);
        if let Some(first) = self.same_media_as(modality, key) {
            self.pending
                .entry(modality)
                .or_default()
                .push(Slot::SameAs(first));
            return Ok(());
        }

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_video(source, cfg).await?;
            Ok(TrackedMedia::Video(clip))
        });

        self.pending
            .entry(modality)
            .or_default()
            .push(Slot::Fetch(handle));
        Ok(())
    }

    fn enqueue_audio(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Audio;
        self.uuids.entry(modality).or_default().push(uuid);

        let key = fetch_key(modality, "", &source);
        if let Some(first) = self.same_media_as(modality, key) {
            self.pending
                .entry(modality)
                .or_default()
                .push(Slot::SameAs(first));
            return;
        }

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_audio(source).await?;
            Ok(TrackedMedia::Audio(clip))
        });

        self.pending
            .entry(modality)
            .or_default()
            .push(Slot::Fetch(handle));
    }
}

// Avoid scanning and allocating the entire payload just to identify its scheme.
// Only canonical opaque image URLs take this path; noncanonical forms and
// invalid data:// authorities retain URL parsing. Connector validation still
// receives the original input unchanged.
fn image_url_source(url: String) -> MediaSource {
    if url.starts_with("data:image/") {
        return MediaSource::DataUrl(url);
    }
    match url::Url::parse(&url) {
        Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
        _ => MediaSource::Url(url),
    }
}

/// Identity of one fetch: the media a part names, together with the settings
/// it would be fetched with.
fn fetch_key(modality: Modality, settings: &str, source: &MediaSource) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(modality.to_string().as_bytes());
    hasher.update(b"\0");
    hasher.update(settings.as_bytes());
    hasher.update(b"\0");
    match source {
        MediaSource::Url(url) => {
            hasher.update(b"url\0");
            hasher.update(url.as_bytes());
        }
        MediaSource::DataUrl(url) => {
            hasher.update(b"data\0");
            hasher.update(url.as_bytes());
        }
        MediaSource::InlineBytes(bytes) => {
            hasher.update(b"bytes\0");
            hasher.update(bytes);
        }
        MediaSource::File(path) => {
            hasher.update(b"file\0");
            hasher.update(path.to_string_lossy().as_bytes());
        }
    }
    hasher.finalize().into()
}

/// The fetch settings for one video: the request's `fps` when given (and
/// valid), else the model's default, else the connector default; plus the
/// long-side cap when given and the model's frame placement.
fn video_fetch_config(
    fps: Option<f64>,
    max_long_side_pixel: Option<u32>,
    default_sample_fps: Option<f32>,
    sampling: FrameSampling,
) -> MultiModalResult<VideoFetchConfig> {
    let mut cfg = VideoFetchConfig {
        sampling,
        ..VideoFetchConfig::default()
    };
    match fps {
        Some(fps) => cfg.sample_fps = validate_sample_fps(fps)? as f32,
        None => {
            if let Some(default) = default_sample_fps {
                validate_sample_fps(f64::from(default))?;
                cfg.sample_fps = default;
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
        for bad_default in [100.0, 5.1, 0.19, 0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                video_fetch_config(None, None, Some(bad_default), FrameSampling::Even).is_err(),
                "{bad_default}"
            );
        }
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
    fn fast_image_scheme_classification_matches_url_parser() {
        let mut cases = vec![
            TINY_PNG_URL.to_owned(),
            "data:".to_owned(),
            "DATA:image/png;base64,abc".to_owned(),
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
            cases.push(format!("data:image/png;base64,abc{}def", char::from(byte)));
        }
        for input in cases {
            let expected = url::Url::parse(&input).is_ok_and(|url| url.scheme() == "data");
            match image_url_source(input.clone()) {
                MediaSource::DataUrl(actual) => {
                    assert!(expected, "{input:?}");
                    assert_eq!(actual, input);
                }
                MediaSource::Url(actual) => {
                    assert!(!expected, "{input:?}");
                    assert_eq!(actual, input);
                }
                _ => panic!("unexpected image source"),
            }
        }
    }

    fn tracker() -> AsyncMultiModalTracker {
        let connector =
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .expect("default connector");
        AsyncMultiModalTracker::new(Arc::new(connector))
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
        let clip = MediaSource::Url("https://example.test/clip.mp4".to_string());
        let key = fetch_key(Modality::Video, "one", &clip);

        assert_eq!(key, fetch_key(Modality::Video, "one", &clip));
        assert_ne!(key, fetch_key(Modality::Video, "two", &clip));
        assert_ne!(key, fetch_key(Modality::Image, "one", &clip));
        assert_ne!(
            key,
            fetch_key(
                Modality::Video,
                "one",
                &MediaSource::Url("https://example.test/other.mp4".to_string()),
            )
        );
    }
}

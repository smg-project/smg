use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MediaConnectorError, MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
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
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
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
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
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

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(
                    source,
                    ImageFetchConfig {
                        detail,
                        max_long_side_pixel,
                    },
                )
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_video(
        &mut self,
        source: MediaSource,
        uuid: Option<String>,
        fps: Option<f64>,
        max_long_side_pixel: Option<u32>,
    ) -> MultiModalResult<()> {
        let mut cfg = VideoFetchConfig::default();
        if let Some(fps) = fps {
            cfg.sample_fps = validate_sample_fps(fps)? as f32;
        }
        if let Some(cap) = max_long_side_pixel {
            validate_video_long_side_cap(cap)?;
            cfg.max_long_side_pixel = Some(cap);
        }

        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_video(source, cfg).await?;
            Ok(TrackedMedia::Video(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
        Ok(())
    }

    fn enqueue_audio(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Audio;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_audio(source).await?;
            Ok(TrackedMedia::Audio(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
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

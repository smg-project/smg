//! Per-item media hints: what a media reference may carry besides its URL,
//! the knobs of the media pipeline that processes the item.
//!
//! [`MediaItemHints`] is the one shape every reader and writer of the hints
//! goes through: the router that puts them on a reference, the servicer that
//! reads them off it, and the pipeline item that applies them. A new hint is
//! a field here, on the wire (`vllm.grpc.engine.MediaRef`) and in the
//! pipeline that applies it.

use crate::vllm_proto::MediaRef;

/// The per-item hints of one media item, as the API request carried them:
/// the long-side cap of an image or of each video frame, and a video's
/// sampling rate. Their bounds are the pipeline's that applies them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MediaItemHints {
    pub max_long_side_pixel: Option<u32>,
    pub fps: Option<f64>,
}

impl MediaItemHints {
    /// Whether any hint is set.
    pub fn is_set(self) -> bool {
        self.max_long_side_pixel.is_some() || self.fps.is_some()
    }
}

impl MediaRef {
    /// The per-item hints this reference carries.
    pub fn item_hints(&self) -> MediaItemHints {
        MediaItemHints {
            max_long_side_pixel: self.max_long_side_pixel,
            fps: self.fps,
        }
    }

    /// Put `hints` on this reference, replacing any it carried.
    pub fn set_item_hints(&mut self, hints: MediaItemHints) {
        self.max_long_side_pixel = hints.max_long_side_pixel;
        self.fps = hints.fps;
    }

    /// Whether this reference carries a per-item hint, which only a worker
    /// that applies hints (`GetServerInfoResponse.mm_item_hints`) may take.
    pub fn has_item_hints(&self) -> bool {
        self.item_hints().is_set()
    }
}

#[cfg(test)]
mod tests {
    use super::{MediaItemHints, MediaRef};

    /// A reference's hints read back as they were put on it, and a reference
    /// carries a hint exactly when one of them is set.
    #[test]
    fn a_reference_carries_the_hints_put_on_it() {
        let mut reference = MediaRef::default();
        assert!(!reference.has_item_hints());
        for hints in [
            MediaItemHints {
                max_long_side_pixel: Some(504),
                fps: None,
            },
            MediaItemHints {
                max_long_side_pixel: None,
                fps: Some(2.0),
            },
        ] {
            reference.set_item_hints(hints);
            assert_eq!(reference.item_hints(), hints);
            assert!(reference.has_item_hints());
        }
        reference.set_item_hints(MediaItemHints::default());
        assert!(!reference.has_item_hints());
    }
}

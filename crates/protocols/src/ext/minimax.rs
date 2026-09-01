//! MiniMax protocol extensions (M3 provider manual; MiniMax-Provider-Verifier).

use serde::{Deserialize, Serialize};

/// Per-image sizing parameters accepted on `image_url` content parts.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct MinimaxImageExt {
    /// Downscale target for the image's long side, in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_long_side_pixel: Option<u32>,
}

/// Per-video sizing and sampling parameters accepted on `video_url` content parts.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct MinimaxVideoExt {
    /// Downscale target for each frame's long side, in pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_long_side_pixel: Option<u32>,

    /// Frame sampling rate; the M3 contract accepts [0.2, 5].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

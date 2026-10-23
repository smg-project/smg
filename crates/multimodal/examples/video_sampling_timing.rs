//! Times the media connector's video path on local clips and digests the
//! frames it samples, so two builds can be compared for speed and for
//! bitwise parity of the sampled frames.
//!
//! ```text
//! cargo run --release -p llm-multimodal --example video_sampling_timing -- \
//!     [--sampling up_to_32|even|even_rounded|interval|temporal_groups]... <clip>...
//! ```
//!
//! Without `--sampling` every rule runs. One tab-separated line per clip and
//! rule: the clip, the rule, the frames sampled, the source frame rate, the
//! wall time in milliseconds, this process's peak resident size in MB (the
//! decoder processes' own is not in it), a digest over the width, height and
//! RGB bytes of every sampled frame in order, and the first source indices.
//! A failed fetch prints `ERROR` and the message in place of the frames.

use std::{process::ExitCode, time::Instant};

use llm_multimodal::{
    FrameSampling, MediaConnector, MediaConnectorConfig, MediaSource, VideoClip, VideoFetchConfig,
};
use rustls::crypto::ring;

const SAMPLINGS: [&str; 5] = [
    "up_to_32",
    "even",
    "even_rounded",
    "interval",
    "temporal_groups",
];

/// The connector's configuration for a named sampling rule, in the shape the
/// families use it: the loader's budget of 32, the rate-based rules at the
/// connector's defaults, one frame a second for the interval rule.
fn sampling(name: &str) -> Option<VideoFetchConfig> {
    let base = VideoFetchConfig::default();
    Some(match name {
        "up_to_32" => VideoFetchConfig {
            sampling: FrameSampling::UpTo { max_frames: 32 },
            ..base
        },
        "even" => base,
        "even_rounded" => VideoFetchConfig {
            sampling: FrameSampling::EvenRounded,
            ..base
        },
        "interval" => VideoFetchConfig {
            min_frames: 1,
            max_frames: 128,
            sample_fps: 1.0,
            sampling: FrameSampling::Interval,
            ..base
        },
        "temporal_groups" => VideoFetchConfig {
            sampling: FrameSampling::TemporalGroups {
                frame_factor: 2,
                max_frames: 96,
            },
            ..base
        },
        _ => return None,
    })
}

#[expect(
    clippy::disallowed_methods,
    reason = "an example binary's client factory; the process TLS crypto provider is installed first"
)]
fn http_client_builder() -> reqwest::ClientBuilder {
    let _ = ring::default_provider().install_default();
    reqwest::Client::builder()
}

/// A digest over every sampled frame's size and RGB bytes, in order.
fn frame_digest(clip: &VideoClip) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    if let Some(video) = &clip.rgb_video {
        for frame in video.frame_refs()? {
            hasher.update(&frame.width.to_le_bytes());
            hasher.update(&frame.height.to_le_bytes());
            hasher.update(frame.data);
        }
    } else {
        for frame in &clip.frames {
            let rgb = frame.to_rgb8();
            hasher.update(&rgb.width().to_le_bytes());
            hasher.update(&rgb.height().to_le_bytes());
            hasher.update(rgb.as_raw());
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// This process's peak resident size in MB, from the kernel's accounting.
fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let kb = line.strip_prefix("VmHWM:")?.trim().strip_suffix("kB")?;
                kb.trim().parse::<f64>().ok()
            })
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

#[expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool: the table on stdout, usage and failures on stderr"
)]
#[tokio::main]
async fn main() -> ExitCode {
    let mut samplings = Vec::new();
    let mut clips = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--sampling" {
            match args.next() {
                Some(name) if sampling(&name).is_some() => samplings.push(name),
                other => {
                    eprintln!(
                        "--sampling takes one of {}; got {other:?}",
                        SAMPLINGS.join(", ")
                    );
                    return ExitCode::FAILURE;
                }
            }
        } else {
            clips.push(arg);
        }
    }
    if clips.is_empty() {
        eprintln!("usage: video_sampling_timing [--sampling <rule>]... <clip>...");
        return ExitCode::FAILURE;
    }
    if samplings.is_empty() {
        samplings.extend(SAMPLINGS.iter().map(|name| (*name).to_string()));
    }

    let connector =
        match MediaConnector::new(http_client_builder(), MediaConnectorConfig::default()) {
            Ok(connector) => connector,
            Err(error) => {
                eprintln!("media connector: {error}");
                return ExitCode::FAILURE;
            }
        };

    for clip in &clips {
        let bytes = match std::fs::read(clip) {
            Ok(bytes) => bytes,
            Err(error) => {
                println!("{clip}\t-\tERROR\t{error}");
                continue;
            }
        };
        for name in &samplings {
            let Some(cfg) = sampling(name) else {
                continue;
            };
            let started = Instant::now();
            let fetched = connector
                .fetch_video(MediaSource::InlineBytes(bytes.clone()), cfg)
                .await;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            match fetched {
                Ok(video) => {
                    let digest =
                        frame_digest(&video).unwrap_or_else(|error| format!("ERROR {error}"));
                    let (frames, source_fps, indices) = match &video.sampling {
                        Some(sampling) => (
                            sampling.frame_indices.len(),
                            format!("{:.3}", sampling.source_fps),
                            sampling
                                .frame_indices
                                .iter()
                                .take(8)
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        ),
                        None => (
                            video
                                .rgb_video
                                .as_ref()
                                .map_or(video.frames.len(), |rgb| rgb.frames.len()),
                            "-".to_string(),
                            "-".to_string(),
                        ),
                    };
                    println!(
                        "{clip}\t{name}\t{frames}\t{source_fps}\t{wall_ms:.1}\t{:.1}\t{digest}\t{indices}",
                        peak_rss_mb()
                    );
                }
                Err(error) => {
                    println!(
                        "{clip}\t{name}\tERROR\t{error}\t{wall_ms:.1}\t{:.1}",
                        peak_rss_mb()
                    );
                }
            }
        }
    }
    ExitCode::SUCCESS
}

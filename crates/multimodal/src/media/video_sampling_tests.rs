use super::*;

fn cfg() -> VideoFetchConfig {
    VideoFetchConfig {
        min_frames: 4,
        max_frames: 8,
        sample_fps: 2.0,
        max_long_side_pixel: None,
        sampling: FrameSampling::Even,
    }
}

fn interval_cfg(sample_fps: f64, max_frames: usize) -> VideoFetchConfig {
    VideoFetchConfig {
        min_frames: 1,
        max_frames,
        sample_fps,
        max_long_side_pixel: None,
        sampling: FrameSampling::Interval,
    }
}

#[test]
fn interval_sampling_takes_one_frame_per_interval_and_keeps_the_last() {
    assert_eq!(
        sampled_frame_indices(360, 30.0, interval_cfg(1.0, 768)),
        vec![0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 359]
    );
    assert_eq!(
        sampled_frame_indices(60, 30.0, interval_cfg(1.0, 768)),
        vec![0, 30, 59]
    );
    assert_eq!(
        sampled_frame_indices(100, 25.0, interval_cfg(0.5, 768)),
        vec![0, 50, 99]
    );
    assert_eq!(
        sampled_frame_indices(300, 29.97, interval_cfg(2.0, 768)),
        vec![
            0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270,
            285, 299
        ]
    );
}

fn up_to_then_rate_cfg(sample_fps: f64) -> VideoFetchConfig {
    VideoFetchConfig {
        sample_fps,
        sampling: FrameSampling::UpToThenRate {
            max_frames: 32,
            rate_max_frames: 2048,
        },
        ..VideoFetchConfig::default()
    }
}

/// A rate binary32 cannot hold (0.3, 0.7, 0.9 or 3.3 frames a second, from
/// a request or a checkpoint's video processor config) reaches the pick
/// arithmetic as the double the model's reference processor computes
/// with. The expected picks are the reference formulas evaluated in
/// double precision; every case picks differently once the rate is
/// narrowed to binary32 and widened back (20 frames at 3 fps sampled at
/// 0.3: `[0, 10]` instead of the reference's `[0, 11]`).
#[test]
fn a_rate_binary32_cannot_hold_picks_the_references_frames() {
    let loader_then_rate = |max_frames| FrameSampling::UpToThenRate {
        max_frames,
        rate_max_frames: 2048,
    };
    let temporal_groups = FrameSampling::TemporalGroups {
        frame_factor: 2,
        max_frames: 96,
    };
    let cases: &[(FrameSampling, usize, f64, f64, &[usize])] = &[
        // The GLM-5-Next rule over a clip the loader kept whole.
        (loader_then_rate(32), 20, 3.0, 0.3, &[0, 11]),
        (loader_then_rate(32), 21, 3.0, 0.3, &[0, 11]),
        (loader_then_rate(32), 30, 3.0, 0.3, &[0, 11, 21, 21]),
        (
            loader_then_rate(32),
            30,
            3.0,
            0.7,
            &[0, 5, 9, 13, 18, 22, 26, 26],
        ),
        (loader_then_rate(32), 10, 3.0, 0.9, &[0, 4, 7, 7]),
        (loader_then_rate(32), 21, 3.0, 0.9, &[0, 4, 8, 12, 16, 20]),
        (loader_then_rate(2048), 80, 12.0, 0.3, &[0, 41]),
        (
            loader_then_rate(2048),
            80,
            8.0,
            0.7,
            &[0, 12, 23, 35, 46, 58, 69, 69],
        ),
        (loader_then_rate(2048), 40, 12.0, 0.9, &[0, 14, 27, 27]),
        (
            loader_then_rate(2048),
            40,
            12.0,
            3.3,
            &[0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 33],
        ),
        // The connector's even rule.
        (FrameSampling::Even, 55, 3.0, 0.3, &[0, 13, 27, 40, 54]),
        (FrameSampling::Even, 15, 3.0, 0.9, &[0, 3, 7, 10, 14]),
        (FrameSampling::Even, 5, 3.0, 3.3, &[0, 0, 1, 2, 3, 4]),
        // The rounded even rule.
        (
            FrameSampling::EvenRounded,
            70,
            3.0,
            0.3,
            &[0, 14, 28, 41, 55, 69],
        ),
        (
            FrameSampling::EvenRounded,
            30,
            3.0,
            0.7,
            &[0, 5, 10, 14, 19, 24, 29],
        ),
        (
            FrameSampling::EvenRounded,
            20,
            3.0,
            0.9,
            &[0, 4, 8, 11, 15, 19],
        ),
        // The temporal-group rule.
        (temporal_groups, 20, 3.0, 0.9, &[0, 3, 7, 11, 15, 19]),
        (
            temporal_groups,
            60,
            3.0,
            0.7,
            &[0, 4, 9, 13, 18, 22, 27, 31, 36, 40, 45, 49, 54, 59],
        ),
    ];
    for (sampling, total_frames, fps, sample_fps, expected) in cases {
        let cfg = VideoFetchConfig {
            sample_fps: *sample_fps,
            sampling: *sampling,
            ..VideoFetchConfig::default()
        };
        assert_eq!(
            sampled_frame_indices(*total_frames, *fps, cfg),
            *expected,
            "{total_frames} frames at {fps} fps sampled at {sample_fps} under {sampling:?}"
        );
    }
}

/// A clip the loader keeps whole is resampled by the GLM-5-Next reference
/// processor's `sample_frames`; the expected picks are transformers' own
/// for each clip (`total_frames`, `fps`, sampling rate). A longer clip
/// keeps the loader's picks, and a clip shorter than one sampling
/// interval, which the reference refuses, keeps the loader's frames.
#[test]
fn up_to_then_rate_resamples_a_clip_the_loader_kept_whole() {
    let cases: &[(usize, f64, f64, &[usize])] = &[
        (32, 8.0, 2.0, &[0, 4, 8, 12, 16, 20, 24, 28]),
        (30, 30.0, 2.0, &[0, 15]),
        (17, 8.0, 2.0, &[0, 4, 8, 12]),
        (31, 10.0, 2.0, &[0, 5, 10, 15, 20, 25]),
        (25, 29.97, 2.0, &[0, 0]),
        (12, 2.0, 2.0, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        (7, 1.0, 2.0, &[0, 1, 2, 3, 4, 5, 6, 6]),
        (10, 3.0, 2.0, &[0, 2, 3, 5, 6, 8]),
        (32, 8.0, 1.0, &[0, 8, 16, 24]),
        (
            20,
            4.0,
            5.0,
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
        ),
        (24, 23.976, 2.0, &[0, 12]),
        (30, 29.97, 1.0, &[0, 0]),
        (3, 1.0, 2.0, &[0, 1, 2, 2]),
        (1, 30.0, 2.0, &[0]),
    ];
    for &(total, fps, rate, expected) in cases {
        assert_eq!(
            sampled_frame_indices(total, fps, up_to_then_rate_cfg(rate)),
            expected,
            "{total} frames at {fps} fps, sampled at {rate}"
        );
    }
    assert_eq!(
        glm5_next_frame_indices(90, 30.0, 2.0, 2048),
        Some(vec![0, 15, 30, 45, 60, 75])
    );
    assert_eq!(
        glm5_next_frame_indices(58, 29.97, 2.0, 2048),
        Some(vec![0, 7, 15, 15])
    );
    // The cap on the reference's count: a 32-frame clip at 1 fps sampled
    // at 2 fps asks for 64 frames, capped to 16 of the loop's 32 picks by
    // `linspace(0, 31, 16)`; at 8 the cap equals the count.
    assert_eq!(
        glm5_next_frame_indices(32, 1.0, 2.0, 16),
        Some(vec![
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31
        ])
    );
    assert_eq!(
        glm5_next_frame_indices(32, 8.0, 2.0, 8),
        Some(vec![0, 4, 8, 12, 16, 20, 24, 28])
    );
    assert_eq!(glm5_next_frame_indices(5, 24.0, 2.0, 2048), None);
    assert_eq!(glm5_next_frame_indices(1, 30.0, 2.0, 2048), None);
    assert_eq!(
        sampled_frame_indices(5, 24.0, up_to_then_rate_cfg(2.0)),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        sampled_frame_indices(50, 10.0, up_to_then_rate_cfg(2.0)),
        sampled_frame_indices(50, 10.0, up_to_cfg(32))
    );
    assert_eq!(
        FrameSampling::UpToThenRate {
            max_frames: 32,
            rate_max_frames: 2048
        }
        .with_frame_budget(Some(16)),
        FrameSampling::UpToThenRate {
            max_frames: 16,
            rate_max_frames: 2048
        }
    );
}

fn rate_walk_cfg(sample_fps: f64, max_frames: usize) -> VideoFetchConfig {
    VideoFetchConfig {
        sample_fps,
        sampling: FrameSampling::RateWalk {
            max_frames,
            temporal_patch_size: 2,
        },
        ..VideoFetchConfig::default()
    }
}

/// The GLM-5 family's own loader rule: the expected picks are the engine's
/// own sampler's for each clip (`total_frames`, `fps`, rate, cap), each
/// distinct pick once as its read hands them on. The first three are the
/// clips measured against the engine's own server.
#[test]
fn rate_walk_picks_what_the_familys_loader_picks() {
    let cases: &[(usize, f64, f64, usize, &[usize])] = &[
        (32, 8.0, 2.0, 2048, &[0, 4, 8, 13, 17, 22, 26, 31]),
        (50, 10.0, 2.0, 2048, &[0, 5, 10, 16, 21, 27, 32, 38, 43, 49]),
        (36, 12.0, 2.0, 2048, &[0, 7, 14, 21, 28, 35]),
        (
            278,
            30.0,
            2.0,
            2048,
            &[
                0, 16, 32, 48, 65, 81, 97, 114, 130, 146, 162, 179, 195, 211, 228, 244, 260, 277,
            ],
        ),
        // Fewer frames than the count: `floor(i x n / count)`, every frame
        // once; and a source at the rate itself, where the walk takes each
        // frame.
        (7, 1.0, 2.0, 2048, &[0, 1, 2, 3, 4, 5, 6]),
        (3, 1.0, 2.0, 2048, &[0, 1, 2]),
        (12, 2.0, 2.0, 2048, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        (2, 2.0, 2.0, 2048, &[0, 1]),
        // An odd count stays odd (the reference's repeat of the last pick
        // never reaches the processor, which pads it).
        (28, 8.0, 2.0, 2048, &[0, 4, 9, 13, 18, 22, 27]),
        // Non-integer source rates.
        (58, 29.97, 2.0, 2048, &[0, 28, 57]),
        (24, 23.976, 2.0, 2048, &[0, 23]),
        (30, 30.0, 2.0, 2048, &[0, 29]),
        (300, 29.97, 0.7, 2048, &[0, 49, 99, 149, 199, 249, 299]),
        // The cap on the count: 32 frames at 1 fps ask for 64, 16 of them
        // by `linspace(0, 31, 16)`; 60 at 30 fps ask for 4 under a cap of 8.
        (
            32,
            1.0,
            2.0,
            16,
            &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31],
        ),
        (60, 30.0, 2.0, 8, &[0, 19, 39, 59]),
        // Request rates: a walk at 1 / (2 x 0.3) seconds over 20 frames
        // at 3 fps, re-spread to two; a sub-second clip at 5 a second
        // under-collects and is re-spread onto its one frame.
        (20, 3.0, 0.3, 2048, &[0, 19]),
        (20, 30.0, 5.0, 2048, &[0]),
        // A clip shorter than one sampling interval (1 / rate s) picks nothing.
        (5, 24.0, 2.0, 2048, &[]),
        (1, 30.0, 2.0, 2048, &[]),
        // A container claiming no frames: the reference's one-second
        // fallback picks the first frame; at 1 fps it picks nothing.
        (0, 8.0, 2.0, 2048, &[0]),
        (0, 1.0, 2.0, 2048, &[]),
    ];
    for &(total, fps, rate, cap, expected) in cases {
        assert_eq!(
            sampled_frame_indices(total, fps, rate_walk_cfg(rate, cap)),
            expected,
            "{total} frames at {fps} fps, sampled at {rate}, at most {cap}"
        );
    }
    assert!(rate_walk_frame_indices(32, 0.0, 2.0, 2048, 2).is_empty());
    assert!(rate_walk_frame_indices(32, 8.0, 0.0, 2048, 2).is_empty());
    assert!(rate_walk_frame_indices(32, 8.0, 2.0, 2048, 0).is_empty());
    // A forged header claiming billions of frames costs the picks, not
    // the frames: at 1 fps every frame passes the walk, which ends once
    // it holds more picks than the count; at a billion fps the walk jumps
    // a quarter of a billion frames per pick. Both answer in microseconds.
    let started = Instant::now();
    let forged = rate_walk_frame_indices(u32::MAX as usize, 1.0, 2.0, 2048, 2);
    assert_eq!(
        forged,
        numpy_linspace_indices(0.0, (u32::MAX - 1) as f64, 2048)
    );
    assert_eq!(forged.len(), 2048);
    assert_eq!(
        rate_walk_frame_indices(4_000_000_000, 1.0e9, 2.0, 2048, 2),
        numpy_linspace_indices(0.0, 3_999_999_999.0, 8)
    );
    assert!(started.elapsed() < Duration::from_secs(5));
    // The loader's frame budget is not a factor for this rule.
    let walk = FrameSampling::RateWalk {
        max_frames: 2048,
        temporal_patch_size: 2,
    };
    assert_eq!(walk.with_frame_budget(Some(16)), walk);
    assert_eq!(walk.with_frame_budget(Some(0)), walk);
    assert!(walk.follows_the_loader());
    assert!(!FrameSampling::EvenRounded.follows_the_loader());
}

/// A clip the family's rule picks nothing of is refused before any
/// decode, with the clip's numbers in the reason; a clip it samples, a
/// clip whose rate the probe did not state, and any other rule pass.
#[test]
fn a_clip_the_rate_walk_picks_nothing_of_is_refused_before_the_decode() {
    let probed = |fps, frames| VideoMetadata {
        width: 64,
        height: 48,
        duration_seconds: None,
        source_fps: fps,
        total_frames: frames,
        container_frames: frames,
        from_container: false,
    };
    let reason = refused_by_the_familys_rule(probed(Some(24.0), Some(5)), rate_walk_cfg(2.0, 2048))
        .expect("refused");
    assert!(reason.contains("5 frames at 24 fps"), "{reason}");
    assert!(
        refused_by_the_familys_rule(probed(Some(1.0), Some(0)), rate_walk_cfg(2.0, 2048)).is_some()
    );
    assert_eq!(
        refused_by_the_familys_rule(probed(Some(8.0), Some(32)), rate_walk_cfg(2.0, 2048)),
        None
    );
    assert_eq!(
        refused_by_the_familys_rule(probed(Some(8.0), Some(0)), rate_walk_cfg(2.0, 2048)),
        None
    );
    assert_eq!(
        refused_by_the_familys_rule(probed(None, Some(5)), rate_walk_cfg(2.0, 2048)),
        None
    );
    let even_rounded = VideoFetchConfig {
        sampling: FrameSampling::EvenRounded,
        ..VideoFetchConfig::default()
    };
    assert_eq!(
        refused_by_the_familys_rule(probed(Some(24.0), Some(5)), even_rounded),
        None
    );
}

fn up_to_cfg(max_frames: usize) -> VideoFetchConfig {
    VideoFetchConfig {
        sampling: FrameSampling::UpTo { max_frames },
        ..VideoFetchConfig::default()
    }
}

/// `numpy.linspace(0, n - 1, min(n, 32), dtype=int)`, the engine's default
/// video loader; the min/max frame counts and the rate play no part.
#[test]
fn up_to_sampling_takes_every_frame_or_spreads_the_budget() {
    assert_eq!(
        sampled_frame_indices(16, 8.0, up_to_cfg(32)),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        sampled_frame_indices(32, 30.0, up_to_cfg(32)),
        (0..32).collect::<Vec<_>>()
    );
    assert_eq!(
        sampled_frame_indices(50, 10.0, up_to_cfg(32)),
        vec![
            0, 1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 20, 22, 23, 25, 26, 28, 30, 31, 33, 34,
            36, 37, 39, 41, 42, 44, 45, 47, 49
        ]
    );
    assert_eq!(
        sampled_frame_indices(36, 12.0, up_to_cfg(32)),
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 29, 30, 31, 32, 33, 35
        ]
    );
    let long = sampled_frame_indices(1000, 30.0, up_to_cfg(32));
    assert_eq!(long.len(), 32);
    assert_eq!((long[0], long[1], long[16], long[31]), (0, 32, 515, 999));
    assert_eq!(sampled_frame_indices(1, 30.0, up_to_cfg(32)), vec![0]);
    assert_eq!(sampled_frame_indices(2, 30.0, up_to_cfg(32)), vec![0, 1]);
    assert_eq!(sampled_frame_indices(5, 30.0, up_to_cfg(0)), vec![0]);
    // The loader reads the first frame of a clip whose count it does not know.
    assert_eq!(sampled_frame_indices(0, 30.0, up_to_cfg(32)), vec![0]);
}

#[test]
fn the_loader_sizes_a_clip_by_what_it_can_see() {
    // The claim holds: the usual spread; nothing claimed: the first frame.
    assert_eq!(loader_frame_indices(60, 60, 32), spread_evenly(60, 32));
    assert_eq!(
        loader_frame_indices(16, 16, 32),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(loader_frame_indices(16, 16, 0), vec![0]);
    assert_eq!(loader_frame_indices(0, 0, 32), vec![0]);
    // A claim two or more frames too long gives way to the visible count.
    assert_eq!(loader_frame_total(58, 40), 40);
    assert_eq!(loader_frame_indices(58, 40, 32), spread_evenly(40, 32));
    assert_eq!(loader_frame_indices(77, 60, 32), spread_evenly(60, 32));
    // One frame too long: the claim stands and the pick past the end is left out.
    assert_eq!(loader_frame_total(61, 60), 61);
    let expected: Vec<usize> = spread_evenly(61, 32)
        .into_iter()
        .filter(|index| *index < 60)
        .collect();
    assert_eq!(expected.len(), 31);
    assert_eq!(expected.last(), Some(&58));
    assert_eq!(loader_frame_indices(61, 60, 32), expected);
    // Nothing visible at all: nothing to read.
    assert_eq!(loader_frame_indices(60, 0, 32), Vec::<usize>::new());
}

#[test]
fn interval_sampling_thins_to_the_frame_budget_and_fills_to_the_minimum() {
    assert_eq!(
        sampled_frame_indices(3000, 30.0, interval_cfg(1.0, 10)),
        vec![0, 330, 660, 990, 1320, 1680, 2010, 2340, 2670, 2999]
    );
    let long = sampled_frame_indices(54_000, 30.0, interval_cfg(1.0, 768));
    assert_eq!(long.len(), 768);
    assert_eq!(long.last(), Some(&53_999));

    let mut short = interval_cfg(1.0, 768);
    short.min_frames = 4;
    assert_eq!(sampled_frame_indices(2, 30.0, short), vec![0, 0, 0, 1]);
    assert_eq!(sampled_frame_indices(0, 30.0, short), Vec::<usize>::new());
}

fn metadata(source_fps: Option<f64>, total_frames: Option<usize>) -> VideoMetadata {
    VideoMetadata {
        width: 64,
        height: 48,
        duration_seconds: Some(2.0),
        source_fps,
        total_frames,
        container_frames: total_frames,
        from_container: false,
    }
}

/// Frames and sampling of a decode, whichever runner produced them.
fn frames_and_sampling(decoded: DecodedVideoFrames) -> (Vec<Vec<u8>>, Option<VideoSamplingInfo>) {
    match decoded {
        DecodedVideoFrames::Rgb {
            video, sampling, ..
        } => (
            video
                .frame_refs()
                .expect("frame refs")
                .iter()
                .map(|frame| frame.data.to_vec())
                .collect(),
            sampling,
        ),
        DecodedVideoFrames::Images {
            frames, sampling, ..
        } => (
            frames
                .iter()
                .map(|frame| frame.to_rgb8().into_raw())
                .collect(),
            sampling,
        ),
    }
}

/// `false` when ffmpeg is not on PATH or will not write the clip.
async fn generate_clip(path: &Path, source: &str, output_args: &[&str]) -> bool {
    let generated = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            source,
        ])
        .args(output_args)
        .arg("-y")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    let probed = Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    matches!((generated, probed), (Ok(generated), Ok(probed)) if generated.success() && probed.success())
}

#[test]
fn both_backends_sample_the_same_source_indices() {
    assert_eq!(sampled_frame_indices(60, 30.0, cfg()), vec![0, 19, 39, 59]);
    // Long clips clamp to max_frames; short ones repeat frames up to min_frames.
    assert_eq!(
        sampled_frame_indices(300, 30.0, cfg()),
        vec![0, 42, 85, 128, 170, 213, 256, 299]
    );
    assert_eq!(sampled_frame_indices(2, 30.0, cfg()), vec![0, 0, 0, 1]);
}

#[test]
fn frame_selection_needs_a_frame_rate_and_a_frame_count() {
    assert_eq!(
        FrameSelection::from_metadata(metadata(None, Some(60)), cfg()),
        None
    );
    assert_eq!(
        FrameSelection::from_metadata(metadata(Some(30.0), None), cfg()),
        None
    );
    assert_eq!(
        FrameSelection::from_metadata(metadata(Some(30.0), Some(0)), cfg()),
        None
    );

    let selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(60)), cfg())
        .expect("frame rate and count are known");
    assert_eq!(selection.frame_indices, vec![0, 19, 39, 59]);
    assert_eq!(
        selection.unique_frames,
        vec![(0, 1), (19, 1), (39, 1), (59, 1)]
    );
    assert_eq!(
        selection.sampling_info(),
        VideoSamplingInfo {
            source_fps: 30.0,
            frame_indices: vec![0, 19, 39, 59],
        }
    );
}

#[test]
fn up_to_selection_follows_the_container_claim_and_the_stream_length() {
    let cfg = up_to_cfg(32);
    // No frame count anywhere: the loader reads one frame.
    let selection = FrameSelection::from_metadata(
        VideoMetadata {
            total_frames: None,
            container_frames: None,
            ..metadata(Some(25.0), None)
        },
        cfg,
    )
    .expect("a frame rate is enough");
    assert_eq!(selection.frame_indices, vec![0]);
    assert_eq!(
        selection.sampling_info(),
        VideoSamplingInfo {
            source_fps: 25.0,
            frame_indices: vec![0],
        }
    );
    assert_eq!(
        FrameSelection::from_metadata(metadata(None, None), cfg),
        None
    );
    // The container runs longer than the stream (an audio tail): sized by the stream.
    let selection = FrameSelection::from_metadata(
        VideoMetadata {
            total_frames: Some(60),
            container_frames: Some(77),
            ..metadata(Some(30.0), None)
        },
        cfg,
    )
    .expect("frame rate and counts");
    assert_eq!(selection.total_frames, 60);
    assert_eq!(selection.frame_indices, spread_evenly(60, 32));
    // Longer by one frame: the claim stands, the pick past the end is left out.
    let selection = FrameSelection::from_metadata(
        VideoMetadata {
            total_frames: Some(60),
            container_frames: Some(61),
            ..metadata(Some(30.0), None)
        },
        cfg,
    )
    .expect("frame rate and counts");
    assert_eq!(selection.total_frames, 61);
    assert_eq!(selection.frame_indices.len(), 31);
    assert_eq!(selection.frame_indices.last(), Some(&58));
    // A decode that shows the stream shorter than claimed resizes the loader's way.
    let claimed = FrameSelection::from_metadata(metadata(Some(30.0), Some(58)), cfg)
        .expect("frame rate and count");
    assert_eq!(claimed.frame_indices, spread_evenly(58, 32));
    let Resized::Selection(resized) = claimed.sized_to_visible(40, cfg) else {
        panic!("shorter than claimed: a selection over the visible frames");
    };
    assert_eq!(resized.total_frames, 40);
    assert_eq!(resized.frame_indices, spread_evenly(40, 32));
    assert_eq!(resized.unique_count(), 32);
    assert_eq!(claimed.sized_to_visible(58, cfg), Resized::Unchanged);
    assert_eq!(claimed.sized_to_visible(0, cfg), Resized::Unchanged);
    // The rate walk: a visible clip shorter than one sampling interval is
    // empty, told apart from unchanged, so the decode refuses it the way a
    // probe of the true length would instead of resampling it by rate.
    let walk = VideoFetchConfig {
        sampling: FrameSampling::RateWalk {
            max_frames: 2048,
            temporal_patch_size: 2,
        },
        sample_fps: 2.0,
        ..cfg
    };
    let walked = FrameSelection::from_metadata(metadata(Some(30.0), Some(300)), walk)
        .expect("frame rate and count");
    assert_eq!(
        walked.sized_to_visible(10, walk),
        Resized::Empty { total_frames: 10 },
        "10 frames at 30 fps is 0.33 s, under the 0.5 s interval at 2 fps"
    );
    assert!(matches!(
        walked.sized_to_visible(60, walk),
        Resized::Selection(_)
    ));
    // A probe that decoded no frame says nothing about the length: unchanged, so the
    // decode reports its own errors instead of a length refusal.
    assert_eq!(walked.sized_to_visible(0, walk), Resized::Unchanged);
    assert!(the_familys_rule_picks_nothing(10, 30.0, walk).ends_with("(0.500 s)"));
    // The rate is shown as configured, not widened.
    let tenth = VideoFetchConfig {
        sample_fps: 0.3,
        ..walk
    };
    assert!(
        the_familys_rule_picks_nothing(10, 30.0, tenth)
            .contains("at 0.3 frames a second (3.333 s)"),
        "{}",
        the_familys_rule_picks_nothing(10, 30.0, tenth)
    );
    // The rate-based samplings keep needing a count and are never resized.
    assert_eq!(
        FrameSelection::from_metadata(
            VideoMetadata {
                total_frames: None,
                container_frames: None,
                ..metadata(Some(25.0), None)
            },
            self::cfg(),
        ),
        None
    );
    let even = FrameSelection::from_metadata(metadata(Some(30.0), Some(60)), self::cfg())
        .expect("frame rate and count");
    assert_eq!(even.sized_to_visible(40, self::cfg()), Resized::Unchanged);
}

#[test]
fn select_filter_escapes_commas_and_lists_each_distinct_index_once() {
    let selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(2)), cfg())
        .expect("frame rate and count are known");
    assert_eq!(selection.frame_indices, vec![0, 0, 0, 1]);
    assert_eq!(selection.unique_count(), 2);
    assert_eq!(selection.select_filter(), r"select='eq(n\,0)+eq(n\,1)'");

    let selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(60)), cfg())
        .expect("frame rate and count are known");
    assert_eq!(
        selection.select_filter(),
        r"select='eq(n\,0)+eq(n\,19)+eq(n\,39)+eq(n\,59)'"
    );
}

#[test]
fn a_long_selection_is_described_rather_than_listed() {
    // The nearest-frame spread spells ties-to-even out of floor, gt, eq and
    // mod, as spread_select_filter documents; x is the picker's own position.
    let x = "round(n*767/17999)*(17999/767)";
    for (sampling, expected) in [
        (
            FrameSampling::Even,
            r"select='eq(n\,floor(round(n*767/17999)*17999/767))'".to_string(),
        ),
        (
            FrameSampling::EvenRounded,
            format!(
                r"select='eq(n\,floor({x})+gt({x}-floor({x})\,0.5)+eq({x}-floor({x})\,0.5)*mod(floor({x})\,2))'"
            ),
        ),
    ] {
        let long = VideoFetchConfig {
            max_frames: 768,
            sampling,
            ..cfg()
        };
        let selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(18_000)), long)
            .expect("frame rate and count are known");
        assert!(selection.unique_count() > FFMPEG_MAX_SELECT_TERMS);

        let filter = selection.select_filter();
        assert_eq!(filter, expected, "{sampling:?}");
        assert!(
            filter.matches('+').count() < FFMPEG_MAX_SELECT_TERMS,
            "a selection past the cutoff is described, not listed"
        );
    }
}

/// The selection run through ffmpeg itself, where it is installed: a
/// synthetic clip longer than the list cutoff, the filter as the crate
/// builds it in both of its shapes (the described spread and the explicit
/// list), the frames ffmpeg hands back read as the source indices they
/// carry, so the frames picked are the Rust selection's.
#[tokio::test]
async fn the_selection_picks_the_same_frames_through_ffmpeg() {
    use std::process::{Command as Sync, Stdio};
    let have = |tool: &str| {
        Sync::new(tool)
            .arg("-version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    };
    if !have("ffmpeg") {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let total = 900usize;
    let fps = 30.0;
    let path = dir.path().join("clip.mkv");
    // Frame i carries i in its pixels: sixteen 4x4 blocks, white where
    // the bit is set, black elsewhere, kept exact by a lossless codec
    // every ffmpeg build carries, so a decoded frame names its own index.
    let side = 16usize;
    let mut raw = Vec::with_capacity(total * side * side * 3);
    for index in 0..total {
        for y in 0..side {
            for x in 0..side {
                let bit = (y / 4) * 4 + x / 4;
                let value = if (index >> bit) & 1 == 1 { 255u8 } else { 0u8 };
                raw.extend_from_slice(&[value, value, value]);
            }
        }
    }
    let mut encode = Sync::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "16x16",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "ffv1",
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg runs");
    {
        use std::io::Write;
        let mut stdin = encode.stdin.take().expect("ffmpeg stdin");
        stdin.write_all(&raw).expect("frames written");
    }
    let encoded = encode.wait_with_output().expect("ffmpeg finishes");
    assert!(
        encoded.status.success(),
        "ffmpeg encode: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    // every frame, decoded the way the selected frames are decoded
    let mut all = ffmpeg_decode_command(2, true);
    all.arg(&path);
    all.args([
        "-vf",
        RGB_CONVERSION,
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]);
    let all = run_video_command_output(all, "ffmpeg")
        .await
        .expect("full decode");
    assert!(all.status.success());
    let every_frame = parse_ppm_stream(&all.stdout).expect("ppm frames");
    assert_eq!(every_frame.len(), total, "the clip holds every frame");
    let index_of = |frame: &image::DynamicImage| -> usize {
        let rgb = frame.to_rgb8();
        assert_eq!((rgb.width(), rgb.height()), (16, 16));
        (0..16u32).fold(0usize, |index, bit| {
            let (x, y) = (bit % 4 * 4 + 2, bit / 4 * 4 + 2);
            let bright = rgb.get_pixel(x, y)[0] > 127;
            index | (usize::from(bright) << bit)
        })
    };
    assert_eq!(
        every_frame.iter().map(index_of).collect::<Vec<_>>(),
        (0..total).collect::<Vec<_>>(),
        "the clip's frames decode to their own indices, in order"
    );
    // The loader's spread, an exact rate spread, and a rate whose picks
    // end on the clip's last frame (the spread plus one explicit term):
    // all three are the described form, and all three go through ffmpeg.
    for (label, config, extra_terms) in [
        ("loader spread", up_to_cfg(300), 0),
        (
            "rate spread ending on the last frame",
            interval_cfg(15.0, 768),
            1,
        ),
        (
            "exact rate spread",
            VideoFetchConfig {
                sample_fps: 12.0,
                max_frames: 768,
                ..cfg()
            },
            0,
        ),
    ] {
        let selection = FrameSelection::from_metadata(metadata(Some(fps), Some(total)), config)
            .expect("frame rate and count are known");
        assert!(
            selection.unique_count() > FFMPEG_MAX_SELECT_TERMS,
            "{label}: {} distinct frames are past the list limit",
            selection.unique_count()
        );
        let filter = selection.select_filter();
        assert!(
            filter.starts_with("select='eq(n\\,floor("),
            "{label}: described, not listed: {filter}"
        );
        assert_eq!(
            filter.matches('+').count(),
            extra_terms,
            "{label}: explicit terms beyond the spread: {filter}"
        );
        let mut command = ffmpeg_decode_command(2, true);
        command.arg(&path);
        FfmpegFrameArgs::selected(&selection)
            .await
            .apply(&mut command);
        command.args([
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]);
        let output = run_video_command_output(command, "ffmpeg")
            .await
            .expect("selected decode");
        assert!(
            output.status.success(),
            "{label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let picked = parse_ppm_stream(&output.stdout).expect("ppm frames");
        let indices: Vec<usize> = picked.iter().map(index_of).collect();
        let expected: Vec<usize> = selection
            .unique_frames
            .iter()
            .map(|(index, _)| *index)
            .collect();
        assert_eq!(
            indices, expected,
            "{label}: ffmpeg picked other frames than the selection"
        );
    }
}

/// `(offset, header length, size)` of the first `kind` box in
/// `bytes[start..end]`.
fn mp4_child(bytes: &[u8], start: usize, end: usize, kind: [u8; 4]) -> (usize, usize, usize) {
    let mut offset = start;
    while offset + 8 <= end {
        let stated = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let (size, header) = match stated {
            1 => (
                u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()) as usize,
                16,
            ),
            0 => (end - offset, 8),
            stated => (stated as usize, 8),
        };
        if bytes[offset + 4..offset + 8] == kind {
            return (offset, header, size);
        }
        offset += size;
    }
    panic!("no {} box", String::from_utf8_lossy(&kind));
}

/// The payload bounds of the box at the end of `kinds`, each kind
/// the first of its name inside the previous one.
fn mp4_payload(bytes: &[u8], kinds: &[[u8; 4]]) -> (usize, usize) {
    let (mut start, mut end) = (0, bytes.len());
    for kind in kinds {
        let (offset, header, size) = mp4_child(bytes, start, end, *kind);
        start = offset + header;
        end = offset + size;
    }
    (start, end)
}

/// Writes `width` into the first track's first sample entry, the muxer's
/// stated size the container reader reports: a clip whose stated size is
/// not the decoder's.
fn narrow_first_sample_entry(path: &Path, width: u16) {
    use std::fs as std_fs;
    let mut bytes = std_fs::read(path).expect("clip bytes");
    let (stsd, _) = mp4_payload(
        &bytes,
        &[*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"stsd"],
    );
    // After the sample description's version, flags and entry count, the
    // first entry: its width sits 32 bytes in (the sample entry header,
    // then the visual entry's reserved fields).
    let entry = stsd + 8;
    bytes[entry + 32..entry + 34].copy_from_slice(&width.to_be_bytes());
    std_fs::write(path, bytes).expect("clip rewritten");
}

/// Writes a 90-degree display rotation into the first track's header
/// matrix, the way a phone stores a portrait recording: the rotation lives
/// in the container, the coded size stays what it was. Written by hand
/// because not every ffmpeg build turns a `rotate` tag into the matrix.
fn rotate_first_track(path: &Path) {
    use std::fs as std_fs;
    let mut bytes = std_fs::read(path).expect("clip bytes");
    let (tkhd, _) = mp4_payload(&bytes, &[*b"moov", *b"trak", *b"tkhd"]);
    let payload = tkhd;
    let matrix = payload + if bytes[payload] == 1 { 52 } else { 40 };
    let rotated: [i32; 9] = [0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30];
    let (slots, _) = bytes[matrix..matrix + 36].as_chunks_mut::<4>();
    for (slot, value) in slots.iter_mut().zip(rotated) {
        *slot = value.to_be_bytes();
    }
    std_fs::write(path, bytes).expect("clip rewritten");
}

/// A one-second 64x48 clip at 24 fps written by ffmpeg into `dir`, or
/// `None` where the tools are not installed.
fn small_clip(dir: &Path, name: &str) -> Option<PathBuf> {
    use std::process::Command as Sync;
    let have = |tool: &str| {
        Sync::new(tool)
            .arg("-version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    };
    if !have("ffmpeg") || !have("ffprobe") {
        return None;
    }
    let path = dir.join(name);
    let encoded = Sync::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=24:duration=1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
        ])
        .arg(&path)
        .output()
        .expect("ffmpeg runs");
    assert!(
        encoded.status.success(),
        "ffmpeg wrote the clip: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    Some(path)
}

/// A clip stored with a 90-degree display rotation (the portrait phone
/// shape) through both runners, where the tools are installed: the headed
/// decode autorotates and hands back the transposed size, which the size
/// check accepts on the first decode; the raw runner, without
/// autorotation, frames the coded size.
#[tokio::test]
async fn a_rotated_clip_decodes_through_both_runners() {
    let dir = tempfile::tempdir().expect("temp dir");
    let Some(path) = small_clip(dir.path(), "rotated.mp4") else {
        return;
    };
    rotate_first_track(&path);
    let metadata = probe_video_metadata(&path).await.expect("metadata");
    assert_eq!(
        (metadata.width, metadata.height),
        (64, 48),
        "the coded size"
    );
    let config = up_to_cfg(4);
    let selection =
        FrameSelection::from_metadata(metadata, config).expect("frame rate and count are known");
    let headed = decode_video_with_ffmpeg_ppm(&path, config, metadata, Some(&selection))
        .await
        .expect("the headed decode accepts the transposed frames");
    assert_eq!(headed.frames.len(), selection.frame_indices.len());
    assert_eq!((headed.frames[0].width, headed.frames[0].height), (48, 64));
    let raw = decode_video_with_ffmpeg_raw(&path, config, metadata, Some(&selection))
        .await
        .expect("the raw decode frames the coded size");
    assert_eq!(raw.frames.len(), selection.frame_indices.len());
    assert_eq!((raw.frames[0].width, raw.frames[0].height), (64, 48));
}

/// A clip whose sample entry states half its width: the container path
/// reports that size, the headed decode fails on the frames' real size,
/// the probe is read, and the second headed decode returns the frames at
/// the probe's size.
#[tokio::test]
async fn a_container_size_the_decoder_disagrees_with_is_re_read_from_the_probe() {
    let dir = tempfile::tempdir().expect("temp dir");
    let Some(path) = small_clip(dir.path(), "narrow.mp4") else {
        return;
    };
    narrow_first_sample_entry(&path, 32);
    let metadata = probe_video_metadata(&path).await.expect("metadata");
    assert!(metadata.from_container, "read from the container's tables");
    assert_eq!(
        (metadata.width, metadata.height),
        (32, 48),
        "the stated size"
    );
    let config = up_to_cfg(4);
    let selection =
        FrameSelection::from_metadata(metadata, config).expect("frame rate and count are known");
    let input_bytes = std::fs::metadata(&path).expect("clip size").len() as usize;
    let decoded = decode_video_with_ffmpeg_runners(
        &path,
        input_bytes,
        config,
        Some(metadata),
        Some(&selection),
    )
    .await
    .expect("the retry on the probe's size succeeds");
    let DecodedVideoFrames::Rgb { video, .. } = decoded else {
        panic!("the second headed decode returns raw RGB frames");
    };
    assert_eq!(video.frames.len(), selection.frame_indices.len());
    assert_eq!((video.frames[0].width, video.frames[0].height), (64, 48));
}

/// The same clip with the probe unable to run: the mismatch stands and
/// the request ends in the headed PNG runner, whose frames carry their
/// own size, never framed raw by the container's.
#[tokio::test]
async fn a_container_size_mismatch_with_no_probe_ends_in_the_headed_png_runner() {
    let dir = tempfile::tempdir().expect("temp dir");
    let Some(path) = small_clip(dir.path(), "narrow-no-probe.mp4") else {
        return;
    };
    narrow_first_sample_entry(&path, 32);
    let metadata = probe_video_metadata(&path).await.expect("metadata");
    assert!(metadata.from_container);
    let config = up_to_cfg(4);
    let selection =
        FrameSelection::from_metadata(metadata, config).expect("frame rate and count are known");
    let input_bytes = std::fs::metadata(&path).expect("clip size").len() as usize;
    probe_switch::UNAVAILABLE.with(|off| off.set(true));
    let decoded = decode_video_with_ffmpeg_runners(
        &path,
        input_bytes,
        config,
        Some(metadata),
        Some(&selection),
    )
    .await;
    probe_switch::UNAVAILABLE.with(|off| off.set(false));
    let DecodedVideoFrames::Images { frames, .. } = decoded.expect("the headed PNG runner") else {
        panic!("a size mismatch without a probe ends in the headed PNG runner");
    };
    assert_eq!(frames.len(), selection.frame_indices.len());
    assert_eq!((frames[0].width(), frames[0].height()), (64, 48));
}

#[test]
fn the_described_selection_covers_the_same_frames() {
    // ffmpeg evaluates the expression per frame with its own `round` (half
    // away from zero); the clips include the parity set's 10-minute clip at
    // 24 fps and the shapes the nearest-frame spread is checked against.
    let cases = [
        (30.0, 9_000, 768),
        (30.0, 18_000, 768),
        (30.0, 36_000, 768),
        (30.0, 7_500, 300),
        (24.0, 14_400, 768),
        (30.0, 1_800, 768),
        (24.0, 2_400, 768),
        (24.0, 1_500, 768),
        // 103 picks over 1238 frames: pick 51 falls on 618.5 exactly, a tie
        // whose floor is even (the connector takes 618, ffmpeg's `round`
        // would take 619).
        (24.0, 1_238, 768),
        (25.0, 1_289, 768),
        // 139 picks over 1672 frames: step 69 is a tie in exact arithmetic
        // (835.5) that the picker's `i * step` lands just under (835).
        (24.0, 1_672, 768),
    ];
    for sampling in [FrameSampling::Even, FrameSampling::EvenRounded] {
        for (fps, total, max_frames) in cases {
            let selection = FrameSelection::from_metadata(
                metadata(Some(fps), Some(total)),
                VideoFetchConfig {
                    max_frames,
                    sampling,
                    ..cfg()
                },
            )
            .expect("frame rate and count are known");
            assert!(selection.unique_count() > FFMPEG_MAX_SELECT_TERMS);
            let last = total - 1;
            let denom = selection.frame_indices.len() - 1;
            let frame_at = |step: f64| match sampling {
                // The picker's arithmetic: the stride rounded once, then scaled.
                FrameSampling::EvenRounded => {
                    (step * (last as f64 / denom as f64)).round_ties_even() as usize
                }
                _ => (step * last as f64 / denom as f64).floor() as usize,
            };
            let described: Vec<usize> = (0..total)
                .filter(|index| {
                    let step = ((index * denom) as f64 / last as f64).round();
                    frame_at(step) == *index
                })
                .collect();
            let listed: Vec<usize> = selection
                .unique_frames
                .iter()
                .map(|(index, _)| *index)
                .collect();
            assert_eq!(
                described, listed,
                "{sampling:?} total={total} max_frames={max_frames}"
            );
            assert!(
                selection.spread_select_filter().is_some(),
                "{sampling:?} total={total} max_frames={max_frames}: not described"
            );
        }
    }
}

#[test]
fn a_nearest_frame_tie_is_described_to_even() {
    // 10 frames at 4 fps, 5 picks at 0, 2.25, 4.5, 6.75 and 9: the connector
    // takes 0, 2, 4 (the tie, to even), 7 and 9. ffmpeg's `round` would take
    // the tie to 5 and its `floor` would take 6.75 to 6; the ties-to-even
    // expression picks this set. Five frames are still listed (under the
    // term limit), but the description is there for a longer clip.
    let rounded = VideoFetchConfig {
        sampling: FrameSampling::EvenRounded,
        ..cfg()
    };
    assert_eq!(sampled_frame_indices(10, 4.0, rounded), vec![0, 2, 4, 7, 9]);
    let selection = FrameSelection::from_metadata(metadata(Some(4.0), Some(10)), rounded)
        .expect("frame rate and count are known");
    let described = selection
        .spread_select_filter()
        .expect("a tie to even is described");
    assert!(described.contains("mod(floor("), "{described}");
    assert_eq!(
        selection.select_filter(),
        r"select='eq(n\,0)+eq(n\,2)+eq(n\,4)+eq(n\,7)+eq(n\,9)'"
    );
}

/// Every nearest-frame spread over the term limit is described, ties
/// included: a sweep over the clip lengths from 51 s to 400 s at the
/// common frame rates (every length at 24 fps, every 13th at the others),
/// where the half-away-from-zero form left about one clip in five listed.
#[test]
fn every_nearest_frame_spread_over_the_term_limit_is_described() {
    let rounded = VideoFetchConfig {
        max_frames: 768,
        sampling: FrameSampling::EvenRounded,
        ..cfg()
    };
    let mut checked = 0usize;
    for (fps, stride) in [(24usize, 1usize), (25, 13), (30, 13), (60, 13)] {
        for total in (51 * fps..=400 * fps).step_by(stride) {
            let selection =
                FrameSelection::from_metadata(metadata(Some(fps as f64), Some(total)), rounded)
                    .expect("frame rate and count are known");
            if selection.unique_count() <= FFMPEG_MAX_SELECT_TERMS {
                continue;
            }
            checked += 1;
            assert!(
                selection.spread_select_filter().is_some(),
                "{fps} fps, {total} frames ({} picks): listed, not described",
                selection.unique_count()
            );
        }
    }
    assert!(checked > 8_000, "{checked} selections checked");
}

/// A selection with no even spread in it (gaps alternating 4 and 2) keeps
/// the explicit list; one that is its own even spread plus a trailing
/// frame, which the picker's whole-stream spread misses, is described by
/// that spread, bounded to its last frame, plus the frame.
#[test]
fn a_selection_the_formula_misses_keeps_the_explicit_list() {
    let mut selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(18_000)), cfg())
        .expect("frame rate and count are known");
    selection.unique_frames = (0..FFMPEG_MAX_SELECT_TERMS + 1)
        .map(|index| (index * 3 + index % 2, 1))
        .collect();
    assert!(selection.spread_select_filter().is_none());
    assert!(selection.select_filter().starts_with(r"select='eq(n\,0)+"));

    selection.unique_frames = (0..FFMPEG_MAX_SELECT_TERMS + 1)
        .map(|index| (index * 3, 1))
        .collect();
    let filter = selection.select_filter();
    assert_eq!(
        filter,
        r"select='eq(n\,floor(round(n*99/297)*297/99))*lte(n\,297)+eq(n\,300)'"
    );
}

#[test]
fn a_frame_count_the_container_made_up_still_builds_a_filter() {
    // The count is whatever the file claims. Walking every frame of it to
    // find the ones the formula picks turns a made-up claim into a stall
    // before ffmpeg is even started.
    let selection = FrameSelection::from_metadata(
        metadata(Some(30.0), Some(usize::MAX / 2)),
        VideoFetchConfig {
            max_frames: 768,
            ..cfg()
        },
    )
    .expect("frame rate and count are known");

    let filter = selection.select_filter();
    assert!(
        filter.contains("floor(round("),
        "described, not listed: {filter}"
    );
}

#[test]
fn expanding_selected_frames_repeats_them_to_match_the_indices() {
    let selection = FrameSelection::from_metadata(metadata(Some(30.0), Some(2)), cfg())
        .expect("frame rate and count are known");
    assert_eq!(
        selection
            .expand(vec!['a', 'b'])
            .expect("one frame per distinct index"),
        vec!['a', 'a', 'a', 'b']
    );
    // A short or long decode fails the runner instead of misaligning the indices.
    assert!(selection.expand(vec!['a']).is_err());
    assert!(selection.expand(vec!['a', 'b', 'c']).is_err());
}

#[test]
fn passthrough_flag_follows_the_ffmpeg_version() {
    for version in [
        "ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\nbuilt with gcc",
        "ffmpeg version 5.1.4-0+deb12u1 Copyright (c) 2000-2023 the FFmpeg developers",
        "ffmpeg version n7.0.2-6-gabcdef Copyright (c) 2000-2024 the FFmpeg developers",
        "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) 2000-2024",
    ] {
        assert_eq!(
            passthrough_flag_for_version(version),
            FPS_MODE_PASSTHROUGH,
            "{version}"
        );
    }
    for version in [
        "ffmpeg version 5.0.3 Copyright (c) 2000-2022 the FFmpeg developers",
        "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers",
        "ffmpeg version N-112345-gabcdef Copyright (c) 2000-2023 the FFmpeg developers",
        "",
    ] {
        assert_eq!(
            passthrough_flag_for_version(version),
            VSYNC_PASSTHROUGH,
            "{version}"
        );
    }
}

#[test]
fn ffprobe_prefers_avg_frame_rate_and_nb_frames() {
    let output = br#"{
        "streams": [{
            "width": 640,
            "height": 360,
            "duration": "10.010000",
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30000/1001",
            "nb_frames": "300"
        }]
    }"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    let source_fps = info.source_fps.expect("frame rate");
    assert!((source_fps - 30000.0 / 1001.0).abs() < 1e-9, "{source_fps}");
    assert_eq!(info.total_frames, Some(300));
}

#[test]
fn ffprobe_falls_back_to_r_frame_rate() {
    // An unknown average rate is reported as `0/0`.
    let output = br#"{"streams": [{"width": 640, "height": 360, "r_frame_rate": "25/1", "avg_frame_rate": "0/0"}]}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.source_fps, Some(25.0));
    assert_eq!(info.total_frames, None);

    let output = br#"{"streams": [{"width": 640, "height": 360, "r_frame_rate": "25/1"}]}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.source_fps, Some(25.0));
}

#[test]
fn ffprobe_derives_the_frame_count_from_the_duration() {
    for nb_frames in [r#""nb_frames": "N/A","#, r#""nb_frames": "0","#, ""] {
        let output = format!(
            r#"{{"streams": [{{"width": 640, "height": 360, "duration": "2.500000", "avg_frame_rate": "24/1", {nb_frames} "time_base": "1/24"}}]}}"#
        );
        let info = parse_ffprobe_video_info(output.as_bytes()).expect("valid ffprobe output");
        assert_eq!(info.total_frames, Some(60), "{nb_frames}");
    }
}

#[test]
fn ffprobe_sizes_the_container_claim_by_the_format_duration_first() {
    // The container outlasts its video stream (an audio tail).
    let output = br#"{"streams": [{"width": 320, "height": 240, "duration": "2.000000", "avg_frame_rate": "30/1"}], "format": {"duration": "2.566667"}}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.total_frames, Some(60));
    assert_eq!(info.container_frames, Some(77));
    // nb_frames settles both.
    let output = br#"{"streams": [{"width": 320, "height": 240, "duration": "2.000000", "avg_frame_rate": "30/1", "nb_frames": "40"}], "format": {"duration": "2.566667"}}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(
        (info.total_frames, info.container_frames),
        (Some(40), Some(40))
    );
    // A live-muxed container states no duration at all.
    let output = br#"{"streams": [{"width": 320, "height": 240, "avg_frame_rate": "25/1", "r_frame_rate": "25/1"}], "format": {}}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.source_fps, Some(25.0));
    assert_eq!((info.total_frames, info.container_frames), (None, None));
}

#[test]
fn a_timed_out_command_is_told_apart_from_other_decode_failures() {
    assert!(is_video_command_timeout(&MediaConnectorError::VideoDecode(
        "ffmpeg timed out after 30.000 seconds".to_string()
    )));
    assert!(!is_video_command_timeout(
        &MediaConnectorError::VideoDecode("ffmpeg produced 31 frames, expected 32".to_string())
    ));
    assert!(!is_video_command_timeout(&MediaConnectorError::Timeout {
        url: "https://media.example/clip.mp4".to_string(),
        budget: Duration::from_secs(30),
    }));
}

#[test]
fn ffprobe_without_a_frame_rate_reports_nothing() {
    let output = br#"{"streams": [{"width": 640, "height": 360, "duration": "2.0", "avg_frame_rate": "0/0", "r_frame_rate": "0/0", "nb_frames": "N/A"}]}"#;
    let info = parse_ffprobe_video_info(output).expect("valid ffprobe output");
    assert_eq!(info.source_fps, None);
    assert_eq!(info.total_frames, None);
}

/// A probe that states no size keeps the clip's frames and duration, so
/// the clip gets its selection, and goes straight to the headed PNG
/// runner: the selection's frames come back, each at its own size, and
/// the decode ends at the last pick rather than resampling every source
/// frame. Skipped when ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn a_probe_that_states_no_size_goes_straight_to_the_png_runner() {
    unsized_clip_decodes(24, 4).await;
}

/// The same clip behind a probe that states no size and twice the frames
/// the stream holds (an estimate, as such probes' durations are): the
/// PNG runner comes back short of the selection, and the recovery a
/// sized clip gets runs for this one too: the decoded count is probed,
/// the selection sized to it, and the frames come back. Skipped when
/// ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn a_probe_that_states_no_size_and_too_many_frames_recovers_as_a_sized_clip_does() {
    unsized_clip_decodes(48, 4).await;
}

/// A one-second 24 fps clip (not an MP4, so the probe answers for it)
/// decoded under a stated `0x0` probe with `probed_frames`, four frames
/// asked for: the frames come back from the PNG runner, `expected` of
/// them, at the clip's size.
async fn unsized_clip_decodes(probed_frames: usize, expected: usize) {
    let dir = tempfile::tempdir().expect("temp dir");
    // Not an MP4 whose tables are read: the probe answers for the clip.
    let Some(path) = small_clip(dir.path(), "unsized.avi") else {
        return;
    };
    let input_bytes = std::fs::metadata(&path).expect("clip size").len() as usize;
    probe_switch::ANSWER.with(|answer| {
        answer.set(Some(ProbedVideoInfo {
            width: Some(0),
            height: Some(0),
            duration_seconds: Some(probed_frames as f64 / 24.0),
            source_fps: Some(24.0),
            total_frames: Some(probed_frames),
            container_frames: Some(probed_frames),
        }));
    });
    let decoded = decode_video_with_ffmpeg(&path, input_bytes, up_to_cfg(expected)).await;
    probe_switch::ANSWER.with(|answer| answer.set(None));
    let DecodedVideoFrames::Images {
        frames, sampling, ..
    } = decoded.expect("the headed PNG runner decodes the clip")
    else {
        panic!("a probe that states no size goes to the headed PNG runner");
    };
    let sampling = sampling.expect("the clip kept its selection");
    assert_eq!(sampling.source_fps, 24.0);
    assert_eq!(sampling.frame_indices.len(), expected);
    assert!(
        sampling.frame_indices.iter().all(|&index| index < 24),
        "every pick is a frame the stream holds: {:?}",
        sampling.frame_indices
    );
    assert_eq!(
        frames.len(),
        expected,
        "the selection's frames, not every source frame"
    );
    assert_eq!((frames[0].width(), frames[0].height()), (64, 48));
}

/// Skipped when ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn ffmpeg_path_decodes_exactly_the_sampled_source_frames() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("clip.mp4");
    let generated = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=30:duration=2",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    let probed = Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    if !matches!((generated, probed), (Ok(generated), Ok(probed)) if generated.success() && probed.success())
    {
        return;
    }

    let metadata = probe_video_metadata(&path).await.expect("ffprobe metadata");
    assert_eq!(metadata.total_frames, Some(60));
    assert_eq!(metadata.source_fps, Some(30.0));

    let input_bytes = fs::read(&path).await.expect("generated clip").len();
    let decoded = decode_video_with_ffmpeg(&path, input_bytes, cfg())
        .await
        .expect("ffmpeg decode");
    let (frames, sampling): (Vec<Vec<u8>>, _) = match decoded {
        DecodedVideoFrames::Rgb {
            video, sampling, ..
        } => (
            video
                .frame_refs()
                .expect("frame refs")
                .iter()
                .map(|frame| frame.data.to_vec())
                .collect(),
            sampling,
        ),
        DecodedVideoFrames::Images {
            frames, sampling, ..
        } => (
            frames
                .iter()
                .map(|frame| frame.to_rgb8().into_raw())
                .collect(),
            sampling,
        ),
    };
    let sampling = sampling.expect("ffprobe reports the frame rate and count");
    assert!(
        (sampling.source_fps - 30.0).abs() < 1e-6,
        "{}",
        sampling.source_fps
    );
    assert_eq!(
        sampling.frame_indices,
        sampled_frame_indices(60, 30.0, cfg())
    );
    assert_eq!(sampling.frame_indices, vec![0, 19, 39, 59]);
    assert_eq!(frames.len(), sampling.frame_indices.len());
    // Distinct source frames: padding to a constant rate would repeat frame 0 instead.
    for pair in frames.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }

    // A frame count past the stream's end leaves the selection short and fails the runner.
    let overshoot = VideoMetadata {
        total_frames: Some(61),
        ..metadata
    };
    let selection = FrameSelection::from_metadata(overshoot, cfg()).expect("frame rate and count");
    assert_eq!(selection.frame_indices, vec![0, 20, 40, 60]);
    let error = decode_video_with_ffmpeg_ppm(&path, cfg(), overshoot, Some(&selection))
        .await
        .expect_err("frame 60 does not exist");
    assert!(error.to_string().contains("expected 4"), "{error}");
}

/// Every frame of a generated clip as the engine's own loader hands it
/// over: libswscale's 32-bit RGB conversion, padding byte dropped.
async fn engine_loader_frames(path: &Path, width: usize, height: usize) -> Vec<Vec<u8>> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
        .stdin(Stdio::null())
        .output()
        .await
        .expect("ffmpeg");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let frame_bytes = width * height * 4;
    assert_eq!(output.stdout.len() % frame_bytes, 0);
    output
        .stdout
        .chunks_exact(frame_bytes)
        .map(|frame| {
            frame
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| pixel[..3].iter().copied())
                .collect()
        })
        .collect()
}

/// The RGB bytes of each decoded frame, whichever runner produced them.
fn decoded_rgb_frames(decoded: DecodedVideoFrames) -> (Vec<Vec<u8>>, Option<VideoSamplingInfo>) {
    match decoded {
        DecodedVideoFrames::Rgb {
            video, sampling, ..
        } => (
            video
                .frame_refs()
                .expect("frame refs")
                .iter()
                .map(|frame| frame.data.to_vec())
                .collect(),
            sampling,
        ),
        DecodedVideoFrames::Images {
            frames, sampling, ..
        } => (
            frames
                .iter()
                .map(|frame| frame.to_rgb8().into_raw())
                .collect(),
            sampling,
        ),
    }
}

/// The pipeline's frames are the engine loader's frames, pixel for pixel,
/// on every ffmpeg runner. Skipped when ffmpeg or ffprobe is not on PATH.
#[tokio::test]
async fn ffmpeg_path_converts_frames_like_the_engine_loader() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("clip.mp4");
    // Rows of 16 pixels and an even height: the sizes the SIMD converters take.
    let generated = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=30:duration=1",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
        ])
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    let probed = Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
    if !matches!((generated, probed), (Ok(generated), Ok(probed)) if generated.success() && probed.success())
    {
        return;
    }

    let reference = engine_loader_frames(&path, 64, 48).await;
    assert_eq!(reference.len(), 30);
    let metadata = probe_video_metadata(&path).await.expect("ffprobe metadata");
    assert_eq!((metadata.width, metadata.height), (64, 48));
    let selection = FrameSelection::from_metadata(metadata, cfg()).expect("frame rate and count");
    assert_eq!(selection.frame_indices, vec![0, 9, 19, 29]);

    let input_bytes = fs::read(&path).await.expect("generated clip").len();
    let decoded = decode_video_with_ffmpeg(&path, input_bytes, cfg())
        .await
        .expect("ffmpeg decode");
    let (frames, sampling) = decoded_rgb_frames(decoded);
    let sampling = sampling.expect("ffprobe reports the frame rate and count");
    assert_eq!(sampling.frame_indices, selection.frame_indices);
    let raw = decode_video_with_ffmpeg_raw(&path, cfg(), metadata, Some(&selection))
        .await
        .expect("rawvideo decode");
    let raw: Vec<Vec<u8>> = raw
        .frame_refs()
        .expect("frame refs")
        .iter()
        .map(|frame| frame.data.to_vec())
        .collect();
    let (png, _) = decode_video_with_ffmpeg_png(&path, cfg(), Some(metadata), Some(&selection))
        .await
        .expect("png decode");
    let png: Vec<Vec<u8>> = png.iter().map(|frame| frame.to_rgb8().into_raw()).collect();

    for (runner, frames) in [("ppm", &frames), ("rawvideo", &raw), ("png", &png)] {
        assert_eq!(frames.len(), selection.frame_indices.len(), "{runner}");
        for (frame, &index) in frames.iter().zip(&selection.frame_indices) {
            let expected = &reference[index];
            assert_eq!(frame.len(), expected.len(), "{runner} frame {index}");
            let differing = frame.iter().zip(expected).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing, 0,
                "{runner} frame {index}: {differing} of {} bytes differ from the engine loader's conversion",
                frame.len()
            );
        }
    }
}

/// Skipped when ffmpeg or ffprobe is not on PATH. A container that states
/// no frame count makes the engine's loader read one frame; one that claims
/// more frames than its stream holds makes it sample over what it can see.
#[tokio::test]
async fn up_to_sampling_without_frame_metadata_follows_the_engine_loader() {
    let dir = tempfile::tempdir().expect("temp dir");
    let cfg = up_to_cfg(32);

    // A live-muxed Matroska stream: a frame rate (the track's default
    // duration), no duration, no frame count.
    let live = dir.path().join("live.mkv");
    if !generate_clip(
        &live,
        "testsrc=size=64x48:rate=25:duration=2",
        &[
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-f", "matroska", "-live", "1",
        ],
    )
    .await
    {
        return;
    }
    let metadata = probe_video_metadata(&live).await.expect("ffprobe metadata");
    assert_eq!(metadata.source_fps, Some(25.0));
    assert_eq!(
        (metadata.total_frames, metadata.container_frames),
        (None, None)
    );
    let input_bytes = fs::read(&live).await.expect("generated clip").len();
    let decoded = decode_video_with_ffmpeg(&live, input_bytes, cfg)
        .await
        .expect("ffmpeg decode");
    let (frames, sampling) = frames_and_sampling(decoded);
    assert_eq!(
        sampling,
        Some(VideoSamplingInfo {
            source_fps: 25.0,
            frame_indices: vec![0],
        })
    );
    assert_eq!(frames.len(), 1);

    // A fragmented MP4 (empty moov): no nb_frames, the duration at the
    // frame rate claims the 60 frames the stream holds.
    let fragmented = dir.path().join("fragmented.mp4");
    if !generate_clip(
        &fragmented,
        "testsrc=size=64x48:rate=30:duration=2",
        &[
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "frag_keyframe+empty_moov",
        ],
    )
    .await
    {
        return;
    }
    let metadata = probe_video_metadata(&fragmented)
        .await
        .expect("ffprobe metadata");
    assert_eq!(metadata.source_fps, Some(30.0));
    assert_eq!(metadata.container_frames, Some(60));
    let input_bytes = fs::read(&fragmented).await.expect("generated clip").len();
    let decoded = decode_video_with_ffmpeg(&fragmented, input_bytes, cfg)
        .await
        .expect("ffmpeg decode");
    let (frames, sampling) = frames_and_sampling(decoded);
    let sampling = sampling.expect("frame rate and claim");
    assert_eq!(sampling.frame_indices, spread_evenly(60, 32));
    assert_eq!(frames.len(), 32);

    // A variable-rate Matroska stream: 30 frames in the first second, 10
    // in the next; the segment's duration at the track's nominal rate
    // claims more frames than the 40 it holds.
    let passthrough = ffmpeg_passthrough_flag().await;
    let variable = dir.path().join("variable.mkv");
    if !generate_clip(
        &variable,
        "testsrc=size=64x48:rate=30:duration=1[a];testsrc=size=64x48:rate=10:duration=1[b];[a][b]concat=n=2:v=1:a=0",
        &[passthrough[0], passthrough[1], "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-f", "matroska"],
    )
    .await
    {
        return;
    }
    let metadata = probe_video_metadata(&variable)
        .await
        .expect("ffprobe metadata");
    assert_eq!(metadata.source_fps, Some(30.0));
    let claimed = metadata
        .container_frames
        .expect("duration at the nominal rate");
    assert!(claimed > 41, "{claimed}");
    let input_bytes = fs::read(&variable).await.expect("generated clip").len();
    let decoded = decode_video_with_ffmpeg(&variable, input_bytes, cfg)
        .await
        .expect("ffmpeg decode");
    let (frames, sampling) = frames_and_sampling(decoded);
    let sampling = sampling.expect("frame rate and claim");
    assert_eq!(sampling.frame_indices, spread_evenly(40, 32));
    assert_eq!(frames.len(), 32);
    for pair in frames.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }
}

#[cfg(feature = "opencv-video")]
#[test]
fn opencv_path_reports_the_source_fps_and_one_index_per_frame() {
    use opencv::core::{Scalar, Size, CV_8UC3};

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("clip.avi");
    let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G').expect("fourcc");
    let mut writer = videoio::VideoWriter::new(
        path.to_str().expect("utf-8 path"),
        fourcc,
        30.0,
        Size::new(64, 48),
        true,
    )
    .expect("video writer");
    for shade in 0..60 {
        let frame =
            Mat::new_rows_cols_with_default(48, 64, CV_8UC3, Scalar::all(f64::from(shade) * 4.0))
                .expect("frame");
        writer.write(&frame).expect("write frame");
    }
    writer.release().expect("release writer");

    let decoded = decode_video_with_opencv_file(&path, cfg()).expect("opencv decode");
    let DecodedVideoFrames::Rgb {
        video, sampling, ..
    } = decoded
    else {
        panic!("expected RGB");
    };
    let sampling = sampling.expect("CAP_PROP_FPS is known");
    assert!(
        (sampling.source_fps - 30.0).abs() < 1e-6,
        "{}",
        sampling.source_fps
    );
    assert_eq!(sampling.frame_indices.len(), video.frames.len());
    assert_eq!(sampling.frame_indices, vec![0, 19, 39, 59]);
}

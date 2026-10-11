//! Audio decode helpers shared by model-specific audio preprocessors.
//!
//! The default path mirrors SMG video decode: use an in-process decoder first,
//! then fall back to an external FFmpeg binary for difficult containers/codecs.

use std::{
    io::{Cursor, Write},
    mem::size_of,
    path::Path,
    process::{Output, Stdio},
    sync::OnceLock,
    time::{Duration, Instant},
};

use symphonia::{
    core::{
        codecs::audio::AudioDecoderOptions,
        errors::Error as SymphoniaError,
        formats::{probe::Hint, FormatOptions, TrackType},
        io::MediaSourceStream,
        meta::MetadataOptions,
    },
    default::{get_codecs, get_probe},
};
use tokio::{process::Command, task, time};
use tracing::debug;

use crate::error::{MediaConnectorError, TransformError};

const DEFAULT_AUDIO_PROCESS_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_AUDIO_MAX_DECODED_BYTES: usize = 256 * 1024 * 1024;

static AUDIO_PROCESS_TIMEOUT: OnceLock<Duration> = OnceLock::new();
static AUDIO_MAX_DECODED_BYTES: OnceLock<usize> = OnceLock::new();

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
}

/// The clip as mono `f32` samples, by the in-process decoder first and the
/// external one when it fails. What a decoder says about the clip is the
/// media's fault ([`MediaConnectorError::AudioDecode`]); this process's own
/// scratch file, pipe or task, and a decoder it cannot start, are its own
/// (see [`MediaConnectorError::fault`]).
pub async fn decode_audio_mono_f32(bytes: &[u8]) -> Result<DecodedAudio, MediaConnectorError> {
    match audio_decode_backend_override() {
        Some("symphonia") => decode_audio_with_symphonia_blocking(bytes).await,
        Some("ffmpeg") => decode_audio_with_ffmpeg(bytes).await,
        Some(backend) => Err(MediaConnectorError::decoder_unavailable(format!(
            "unsupported SMG_AUDIO_DECODE_BACKEND={backend}; expected auto, symphonia, or ffmpeg"
        ))),
        None => match decode_audio_with_symphonia_blocking(bytes).await {
            Ok(decoded) => Ok(decoded),
            Err(symphonia_error) => {
                debug!(
                    error = %symphonia_error,
                    "smg_mm_timing audio_decode_auto_symphonia_fallback"
                );
                decode_audio_with_ffmpeg(bytes).await.map_err(|ffmpeg_error| {
                    let message = format!(
                        "Symphonia audio decode failed: {symphonia_error}; ffmpeg fallback failed: {ffmpeg_error}"
                    );
                    MediaConnectorError::fallback_failed(
                        symphonia_error,
                        ffmpeg_error,
                        message,
                        MediaConnectorError::AudioDecode,
                    )
                })
            }
        },
    }
}

async fn decode_audio_with_symphonia_blocking(
    bytes: &[u8],
) -> Result<DecodedAudio, MediaConnectorError> {
    let bytes = bytes.to_vec();
    task::spawn_blocking(move || decode_audio_mono_f32_symphonia(&bytes))
        .await
        .map_err(MediaConnectorError::Blocking)?
        .map_err(audio_decode)
}

/// What a decoder said about the clip: the media's fault.
fn audio_decode(error: TransformError) -> MediaConnectorError {
    MediaConnectorError::AudioDecode(error.to_string())
}

pub(crate) fn decode_audio_mono_f32_symphonia(
    bytes: &[u8],
) -> Result<DecodedAudio, TransformError> {
    decode_audio_mono_f32_symphonia_with_limits(
        bytes,
        audio_max_decoded_bytes(),
        audio_process_timeout(),
    )
}

fn decode_audio_mono_f32_symphonia_with_limits(
    bytes: &[u8],
    max_decoded_bytes: usize,
    timeout: Duration,
) -> Result<DecodedAudio, TransformError> {
    let started = Instant::now();
    let mut hint = Hint::new();
    if let Some(ext) = audio_extension_hint(bytes) {
        hint.with_extension(ext);
    }

    let cursor = Cursor::new(bytes.to_vec());
    let media_source = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut format = get_probe()
        .probe(
            &hint,
            media_source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| TransformError::ShapeError(format!("Symphonia probe failed: {e}")))?;

    let track = format.default_track(TrackType::Audio).ok_or_else(|| {
        TransformError::ShapeError("Symphonia found no supported audio track".to_string())
    })?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| {
            TransformError::ShapeError(
                "Symphonia audio track is missing codec parameters".to_string(),
            )
        })?;
    let mut decoder = get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| TransformError::ShapeError(format!("Symphonia decoder failed: {e}")))?;

    let mut sample_rate = audio_params.sample_rate.map(|rate| rate as usize);
    let mut mono = Vec::new();
    let mut interleaved = Vec::new();
    loop {
        ensure_symphonia_deadline(started, timeout)?;
        let Some(packet) = format.next_packet().map_err(|error| {
            TransformError::ShapeError(format!("Symphonia packet read failed: {error}"))
        })?
        else {
            break;
        };
        if packet.track_id != track_id {
            continue;
        }

        let audio_buf = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => {
                return Err(TransformError::ShapeError(format!(
                    "Symphonia packet decode failed: {error}"
                )));
            }
        };
        ensure_symphonia_deadline(started, timeout)?;

        let spec = audio_buf.spec();
        sample_rate = Some(spec.rate() as usize);
        let channels = spec.channels().count();
        if channels == 0 {
            return Err(TransformError::ShapeError(
                "decoded audio has zero channels".to_string(),
            ));
        }

        let interleaved_samples = audio_buf.samples_interleaved();
        ensure_decoded_sample_limit(0, interleaved_samples, max_decoded_bytes)?;
        let additional_samples = interleaved_samples / channels;
        ensure_decoded_sample_limit(mono.len(), additional_samples, max_decoded_bytes)?;
        mono.try_reserve(additional_samples).map_err(|error| {
            TransformError::ShapeError(format!(
                "failed to reserve {additional_samples} decoded audio samples: {error}"
            ))
        })?;
        interleaved.resize(interleaved_samples, 0.0);
        audio_buf.copy_to_slice_interleaved(&mut interleaved);
        for frame in interleaved.chunks_exact(channels) {
            mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
        }
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        TransformError::ShapeError("decoded audio is missing sample rate".to_string())
    })?;
    finish_decoded_audio(mono, sample_rate)
}

async fn decode_audio_with_ffmpeg(bytes: &[u8]) -> Result<DecodedAudio, MediaConnectorError> {
    let input_file = write_temp_audio_file_async(bytes).await?;
    let sample_rate = probe_audio_sample_rate(input_file.path()).await?;
    // Ask FFmpeg for one sample beyond our limit so a longer stream is
    // distinguishable from a valid stream whose size is exactly the limit.
    let output_limit = audio_max_decoded_bytes()
        .saturating_add(size_of::<f32>())
        .to_string();

    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
        .arg(input_file.path())
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-fs",
            &output_limit,
            "-f",
            "f32le",
            "-sample_fmt",
            "flt",
            "pipe:1",
        ]);
    let output = run_audio_command_output(command, "ffmpeg").await?;
    if !output.status.success() {
        return Err(MediaConnectorError::AudioDecode(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    ensure_decoded_byte_limit(output.stdout.len(), audio_max_decoded_bytes())
        .map_err(audio_decode)?;
    if output.stdout.len() % 4 != 0 {
        return Err(MediaConnectorError::AudioDecode(format!(
            "ffmpeg f32le output has trailing partial sample: {} bytes",
            output.stdout.len() % 4
        )));
    }

    let samples = output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    finish_decoded_audio(samples, sample_rate).map_err(audio_decode)
}

fn ensure_symphonia_deadline(started: Instant, timeout: Duration) -> Result<(), TransformError> {
    if started.elapsed() >= timeout {
        return Err(TransformError::ShapeError(format!(
            "Symphonia timed out after {:.3} seconds",
            timeout.as_secs_f64()
        )));
    }
    Ok(())
}

fn ensure_decoded_sample_limit(
    existing_samples: usize,
    additional_samples: usize,
    max_decoded_bytes: usize,
) -> Result<(), TransformError> {
    let total_samples = existing_samples
        .checked_add(additional_samples)
        .ok_or_else(|| {
            TransformError::ShapeError("decoded audio sample count overflow".to_string())
        })?;
    let decoded_bytes = total_samples.checked_mul(size_of::<f32>()).ok_or_else(|| {
        TransformError::ShapeError("decoded audio byte size overflow".to_string())
    })?;
    ensure_decoded_byte_limit(decoded_bytes, max_decoded_bytes)
}

fn ensure_decoded_byte_limit(
    decoded_bytes: usize,
    max_decoded_bytes: usize,
) -> Result<(), TransformError> {
    if decoded_bytes > max_decoded_bytes {
        return Err(TransformError::ShapeError(format!(
            "decoded audio payload is {decoded_bytes} bytes, exceeding SMG_AUDIO_MAX_DECODED_BYTES={max_decoded_bytes}"
        )));
    }
    Ok(())
}

fn finish_decoded_audio(
    samples: Vec<f32>,
    sample_rate: usize,
) -> Result<DecodedAudio, TransformError> {
    if samples.is_empty() {
        return Err(TransformError::ShapeError(
            "decoded audio produced no samples".to_string(),
        ));
    }
    Ok(DecodedAudio {
        samples,
        sample_rate,
    })
}

async fn probe_audio_sample_rate(input_path: &Path) -> Result<usize, MediaConnectorError> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input_path);
    let output = run_audio_command_output(command, "ffprobe").await?;
    if !output.status.success() {
        return Err(MediaConnectorError::AudioDecode(format!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| line.trim().parse::<usize>().ok())
        .filter(|rate| *rate > 0)
        .ok_or_else(|| {
            MediaConnectorError::AudioDecode(format!(
                "failed to parse ffprobe sample rate: {stdout:?}"
            ))
        })
}

async fn write_temp_audio_file_async(
    bytes: &[u8],
) -> Result<tempfile::NamedTempFile, MediaConnectorError> {
    let bytes = bytes.to_vec();
    task::spawn_blocking(move || write_temp_audio_file(&std::env::temp_dir(), &bytes))
        .await
        .map_err(MediaConnectorError::Blocking)?
}

/// The scratch file the external decoder reads `bytes` from, in `dir`. Its
/// I/O is this process's own ([`MediaConnectorError::DecodeIo`]): the bytes
/// have arrived by now, and a directory that is full or not writable is not
/// the media's fault.
fn write_temp_audio_file(
    dir: &Path,
    bytes: &[u8],
) -> Result<tempfile::NamedTempFile, MediaConnectorError> {
    let mut input_file = tempfile::Builder::new()
        .prefix("smg-audio-")
        .suffix(audio_temp_suffix(bytes))
        .tempfile_in(dir)
        .map_err(MediaConnectorError::DecodeIo)?;
    input_file
        .write_all(bytes)
        .map_err(MediaConnectorError::DecodeIo)?;
    input_file.flush().map_err(MediaConnectorError::DecodeIo)?;
    Ok(input_file)
}

/// Runs a decoder command to completion under the audio process timeout. A
/// decoder this process cannot start (not installed, not executable) is this
/// process's ([`MediaConnectorError::DecoderUnavailable`]), reported in the
/// operator's terms, and so is the command's own I/O (a pipe that breaks:
/// [`MediaConnectorError::DecodeIo`]); only what the decoder says about the
/// clip, and a decode that runs out of its time, is the media's.
async fn run_audio_command_output(
    mut command: Command,
    program: &'static str,
) -> Result<Output, MediaConnectorError> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = command.spawn().map_err(|e| {
        let message = if e.kind() == std::io::ErrorKind::NotFound {
            format!("{program} executable not found; install {program} to decode audio inputs")
        } else {
            format!("{program} could not be started: {e}")
        };
        MediaConnectorError::DecoderUnavailable(std::io::Error::new(e.kind(), message))
    })?;

    let timeout = audio_process_timeout();
    match time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(MediaConnectorError::DecodeIo(error)),
        Err(_) => Err(MediaConnectorError::AudioDecode(format!(
            "{program} timed out after {:.3} seconds",
            timeout.as_secs_f64()
        ))),
    }
}

fn audio_decode_backend_override() -> Option<&'static str> {
    static BACKEND: OnceLock<Option<String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            std::env::var("SMG_AUDIO_DECODE_BACKEND")
                .ok()
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty() && value != "auto")
        })
        .as_deref()
}

fn audio_process_timeout() -> Duration {
    *AUDIO_PROCESS_TIMEOUT.get_or_init(|| {
        std::env::var("SMG_AUDIO_PROCESS_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(DEFAULT_AUDIO_PROCESS_TIMEOUT)
    })
}

fn audio_max_decoded_bytes() -> usize {
    *AUDIO_MAX_DECODED_BYTES.get_or_init(|| {
        std::env::var("SMG_AUDIO_MAX_DECODED_BYTES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|bytes| *bytes > 0)
            .unwrap_or(DEFAULT_AUDIO_MAX_DECODED_BYTES)
    })
}

fn audio_temp_suffix(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        return ".wav";
    }
    if bytes.starts_with(b"fLaC") {
        return ".flac";
    }
    if bytes.starts_with(b"ID3") || is_mp3_frame_sync(bytes) {
        return ".mp3";
    }
    if bytes.starts_with(b"OggS") {
        return ".ogg";
    }
    if bytes.len() >= 12 && bytes.get(4..8) == Some(b"ftyp") {
        return ".m4a";
    }
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return ".webm";
    }
    if bytes.len() >= 12
        && bytes.starts_with(b"FORM")
        && matches!(bytes.get(8..12), Some(b"AIFF" | b"AIFC"))
    {
        return ".aiff";
    }
    if bytes.len() >= 4 && bytes.starts_with(b"caff") {
        return ".caf";
    }
    ".audio"
}

fn audio_extension_hint(bytes: &[u8]) -> Option<&'static str> {
    match audio_temp_suffix(bytes) {
        ".wav" => Some("wav"),
        ".flac" => Some("flac"),
        ".mp3" => Some("mp3"),
        ".ogg" => Some("ogg"),
        ".m4a" => Some("m4a"),
        ".webm" => Some("webm"),
        ".aiff" => Some("aiff"),
        ".caf" => Some("caf"),
        _ => None,
    }
}

fn is_mp3_frame_sync(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xff && (bytes[1] & 0xe0) == 0xe0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MediaFault;

    fn wav_i16_mono(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_bytes = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn symphonia_decodes_wav_to_mono_f32() {
        let wav = wav_i16_mono(16_000, &[0, 16_384, -16_384]);
        let decoded = decode_audio_mono_f32_symphonia(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.samples.len(), 3);
        assert!(decoded.samples[0].abs() < 1e-6);
        assert!((decoded.samples[1] - 0.5).abs() < 1e-4);
        assert!((decoded.samples[2] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn symphonia_enforces_decoded_byte_limit() {
        let wav = wav_i16_mono(16_000, &[0, 1, 2]);
        let error = decode_audio_mono_f32_symphonia_with_limits(
            &wav,
            2 * size_of::<f32>(),
            Duration::from_secs(1),
        )
        .unwrap_err();

        assert!(error.to_string().contains("SMG_AUDIO_MAX_DECODED_BYTES=8"));
    }

    #[test]
    fn symphonia_enforces_decode_deadline() {
        let wav = wav_i16_mono(16_000, &[0]);
        let error = decode_audio_mono_f32_symphonia_with_limits(&wav, usize::MAX, Duration::ZERO)
            .unwrap_err();

        assert!(error.to_string().contains("Symphonia timed out"));
    }

    #[test]
    fn empty_decoded_audio_is_rejected() {
        let error = finish_decoded_audio(Vec::new(), 16_000).unwrap_err();
        assert!(error.to_string().contains("produced no samples"));
    }

    #[test]
    fn audio_suffixes_cover_common_containers() {
        assert_eq!(audio_temp_suffix(b"fLaC..."), ".flac");
        assert_eq!(audio_temp_suffix(b"ID3..."), ".mp3");
        assert_eq!(audio_temp_suffix(b"OggS..."), ".ogg");
        assert_eq!(audio_temp_suffix(b"\x1a\x45\xdf\xa3..."), ".webm");
        assert_eq!(audio_temp_suffix(b"\0\0\0\x18ftypM4A "), ".m4a");
    }

    /// A decoder this process cannot start is this process's fault, not the
    /// media's, whatever clip the request sent: the executable is not there,
    /// or is there but cannot be run. The operator's words stay on the error.
    #[tokio::test]
    async fn a_decoder_that_cannot_start_is_our_fault() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let missing = dir.path().join("ffprobe");
        let error = run_audio_command_output(Command::new(&missing), "ffprobe")
            .await
            .expect_err("a decoder that is not there");
        assert!(
            matches!(&error, MediaConnectorError::DecoderUnavailable(io) if io.kind() == std::io::ErrorKind::NotFound),
            "{error}"
        );
        assert_eq!(error.fault(), MediaFault::Internal);
        assert!(
            error
                .to_string()
                .contains("ffprobe executable not found; install ffprobe"),
            "{error}"
        );

        let not_executable = dir.path().join("ffmpeg");
        std::fs::write(&not_executable, b"not a program").expect("a plain file");
        let error = run_audio_command_output(Command::new(&not_executable), "ffmpeg")
            .await
            .expect_err("a decoder that cannot be run");
        assert!(
            matches!(&error, MediaConnectorError::DecoderUnavailable(io) if io.kind() == std::io::ErrorKind::PermissionDenied),
            "{error}"
        );
        assert_eq!(error.fault(), MediaFault::Internal);
        assert!(
            error.to_string().contains("ffmpeg could not be started"),
            "{error}"
        );
    }

    /// The scratch file the external decoder reads from is this process's
    /// own: in a directory that is not there (or not writable) the error is
    /// this process's fault, whatever bytes the request sent; the same bytes
    /// in a writable directory are written as they are.
    #[test]
    fn a_scratch_file_that_cannot_be_written_is_our_fault() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let bytes = b"not audio, never decoded";
        let written = write_temp_audio_file(dir.path(), bytes).expect("a writable dir");
        assert_eq!(
            std::fs::read(written.path()).expect("the scratch file"),
            bytes
        );
        let missing = dir.path().join("missing");
        let error = write_temp_audio_file(&missing, bytes).expect_err("a dir that is not there");
        assert!(matches!(error, MediaConnectorError::DecodeIo(_)), "{error}");
        assert_eq!(error.fault(), MediaFault::Internal);
    }

    /// Bytes no decoder can read are the request's fault, whether the
    /// external decoder ran and said so or could not start at all: the
    /// in-process decoder's verdict stands when the fallback judged nothing.
    #[tokio::test]
    async fn a_clip_no_decoder_can_read_is_the_requests() {
        let error = decode_audio_mono_f32(b"not audio at all")
            .await
            .expect_err("not audio");
        assert!(
            matches!(error, MediaConnectorError::AudioDecode(_)),
            "{error}"
        );
        assert_eq!(error.fault(), MediaFault::Client);
        if audio_decode_backend_override().is_none() {
            let words = error.to_string();
            assert!(
                words.contains("Symphonia") && words.contains("ffmpeg"),
                "{words}"
            );
        }
    }
}

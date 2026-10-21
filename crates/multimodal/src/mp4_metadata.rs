//! The video stream metadata the probe tool reports for an MP4 / MOV file,
//! read from the container's own tables.
//!
//! The media connector sizes a clip (its frame count, frame rate and
//! duration) before it decodes it, to pin the frames the engine's loader
//! would pick. That took a probe process per clip; for the common file shape
//! the same numbers sit in the `moov` box and are read here without one. The
//! shape is checked, not assumed: the first video track (the probe's `v:0`;
//! later video tracks are ignored, as the probe ignores them), a single
//! constant frame duration, an edit list that is the whole track at rate 1.0
//! exactly or none, a stated size and a positive duration. Any other file
//! returns `None` and keeps the probe process, so the values this module
//! yields are the probe's for every file it accepts (the agreement test runs
//! the tool when it is installed). One caveat: the size is the sample entry's,
//! what the muxer wrote, which the probe replaces with the decoder's own size
//! after it has looked at the stream; the two agree on the files common
//! muxers write, and the connector checks the decoded frames' own size
//! against this one and re-reads a disagreeing file from the probe before any
//! raw framing:
//!
//! - `width` / `height`: the first visual sample entry's;
//! - `duration_ts` / `time_base`: the media header's duration and time scale
//!   (the stream duration the tool prints is their product, with six decimals);
//! - `nb_frames`: the sample count;
//! - `avg_frame_rate`: the time scale times the sample count over the track
//!   duration, in lowest terms (the demuxer's own derivation);
//! - the movie header's duration and time scale for the container's duration.

use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

/// The largest `moov` box read into memory; a sane container's is kilobytes
/// per minute of video.
const MAX_MOOV_BYTES: u64 = 64 * 1024 * 1024;

/// What the container says about its first video track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContainerVideoInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The track's media duration, in `timescale` units per second.
    pub(crate) duration_ts: u64,
    pub(crate) timescale: u32,
    /// The average frame rate as a reduced fraction.
    pub(crate) frame_rate: (u64, u64),
    pub(crate) frame_count: u64,
    /// The movie's duration and time scale (the container's duration).
    pub(crate) movie_duration_ts: u64,
    pub(crate) movie_timescale: u32,
}

impl ContainerVideoInfo {
    /// Seconds as the probe tool prints a time: the ticks times the time
    /// base, written with six decimals and read back.
    fn seconds(ticks: u64, timescale: u32) -> Option<f64> {
        if timescale == 0 {
            return None;
        }
        let value = ticks as f64 * (1.0 / f64::from(timescale));
        let printed = format!("{value:.6}");
        printed
            .parse::<f64>()
            .ok()
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
    }

    /// The stream duration in seconds, as the tool reports it.
    pub(crate) fn stream_duration_seconds(&self) -> Option<f64> {
        Self::seconds(self.duration_ts, self.timescale)
    }

    /// The container duration in seconds, as the tool reports it.
    pub(crate) fn movie_duration_seconds(&self) -> Option<f64> {
        Self::seconds(self.movie_duration_ts, self.movie_timescale)
    }

    /// The frame rate as the tool's `num/den` string reads back.
    pub(crate) fn frames_per_second(&self) -> Option<f64> {
        let (num, den) = self.frame_rate;
        let ratio = num as f64 / den as f64;
        (ratio.is_finite() && ratio > 0.0).then_some(ratio)
    }
}

/// Reads the file's container tables; `Ok(None)` when the file is not an
/// MP4 / MOV of the shape described above.
pub(crate) fn read(path: &Path) -> io::Result<Option<ContainerVideoInfo>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut offset = 0u64;
    let mut first = true;
    while offset + 8 <= len {
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header[..8])?;
        let mut size = u64::from(u32::from_be_bytes([
            header[0], header[1], header[2], header[3],
        ]));
        let kind = [header[4], header[5], header[6], header[7]];
        let mut header_len = 8u64;
        if size == 1 {
            file.read_exact(&mut header[8..16])?;
            size = u64::from_be_bytes([
                header[8], header[9], header[10], header[11], header[12], header[13], header[14],
                header[15],
            ]);
            header_len = 16;
        } else if size == 0 {
            size = len - offset;
        }
        // `checked_add`: a crafted 64-bit size that wraps the sum would otherwise
        // pass the bound and send the walk back to the start, forever.
        if size < header_len || offset.checked_add(size).is_none_or(|end| end > len) {
            return Ok(None);
        }
        if first {
            if &kind != b"ftyp" {
                return Ok(None);
            }
            first = false;
        }
        if &kind == b"moov" {
            let payload = size - header_len;
            if payload > MAX_MOOV_BYTES {
                return Ok(None);
            }
            let mut moov = vec![0u8; payload as usize];
            file.read_exact(&mut moov)?;
            return Ok(parse_moov(&moov));
        }
        offset += size;
    }
    Ok(None)
}

/// The boxes of `data`: `(kind, payload)` in order; `None` on a malformed size.
fn boxes(data: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let mut size = u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        let kind: [u8; 4] = data[offset + 4..offset + 8].try_into().ok()?;
        let mut header = 8usize;
        if size == 1 {
            let large = u64::from_be_bytes(data.get(offset + 8..offset + 16)?.try_into().ok()?);
            size = usize::try_from(large).ok()?;
            header = 16;
        } else if size == 0 {
            size = data.len() - offset;
        }
        if size < header || offset.checked_add(size).is_none_or(|end| end > data.len()) {
            return None;
        }
        out.push((kind, &data[offset + header..offset + size]));
        offset += size;
    }
    Some(out)
}

fn child<'a>(list: &[([u8; 4], &'a [u8])], kind: [u8; 4]) -> Option<&'a [u8]> {
    list.iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, payload)| *payload)
}

fn be_u32(data: &[u8], at: usize) -> Option<u32> {
    data.get(at..at + 4)
        .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn be_u64(data: &[u8], at: usize) -> Option<u64> {
    data.get(at..at + 8).map(|bytes| {
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    })
}

fn be_u16(data: &[u8], at: usize) -> Option<u16> {
    data.get(at..at + 2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
}

/// `(timescale, duration)` of a movie or media header (versions 0 and 1).
fn header_timescale_duration(payload: &[u8]) -> Option<(u32, u64)> {
    match payload.first()? {
        0 => Some((be_u32(payload, 12)?, u64::from(be_u32(payload, 16)?))),
        1 => Some((be_u32(payload, 20)?, be_u64(payload, 24)?)),
        _ => None,
    }
}

fn parse_moov(moov: &[u8]) -> Option<ContainerVideoInfo> {
    let top = boxes(moov)?;
    let (movie_timescale, movie_duration_ts) = header_timescale_duration(child(&top, *b"mvhd")?)?;
    for (kind, trak) in &top {
        if kind != b"trak" {
            continue;
        }
        let trak = boxes(trak)?;
        let mdia = boxes(child(&trak, *b"mdia")?)?;
        let hdlr = child(&mdia, *b"hdlr")?;
        if hdlr.get(8..12)? != b"vide" {
            continue;
        }
        // The first video track decides; a shape this module does not
        // accept keeps the probe process.
        let (timescale, duration_ts) = header_timescale_duration(child(&mdia, *b"mdhd")?)?;
        if timescale == 0 || duration_ts == 0 {
            return None;
        }
        if let Some(edts) = child(&trak, *b"edts") {
            if !edit_list_is_whole_track(
                child(&boxes(edts)?, *b"elst")?,
                duration_ts,
                timescale,
                movie_timescale,
            ) {
                return None;
            }
        }
        let stbl = boxes(child(&boxes(child(&mdia, *b"minf")?)?, *b"stbl")?)?;
        let (width, height) = sample_entry_size(child(&stbl, *b"stsd")?)?;
        let stts = child(&stbl, *b"stts")?;
        if be_u32(stts, 4)? != 1 {
            return None;
        }
        let (count, delta) = (u64::from(be_u32(stts, 8)?), u64::from(be_u32(stts, 12)?));
        if count == 0 || delta == 0 || count.checked_mul(delta)? != duration_ts {
            return None;
        }
        let sample_count = match (child(&stbl, *b"stsz"), child(&stbl, *b"stz2")) {
            (Some(stsz), _) => u64::from(be_u32(stsz, 8)?),
            (None, Some(stz2)) => u64::from(be_u32(stz2, 8)?),
            (None, None) => return None,
        };
        if sample_count != count {
            return None;
        }
        // The demuxer's average frame rate: time scale x frames / duration, reduced.
        let num = u64::from(timescale).checked_mul(count)?;
        let g = gcd(num, duration_ts);
        return Some(ContainerVideoInfo {
            width,
            height,
            duration_ts,
            timescale,
            frame_rate: (num / g, duration_ts / g),
            frame_count: count,
            movie_duration_ts,
            movie_timescale,
        });
    }
    None
}

/// True for an edit list of one entry that maps the whole track from its
/// start at normal speed: the shape under which the demuxer's stream
/// duration is the media header's.
fn edit_list_is_whole_track(
    elst: &[u8],
    duration_ts: u64,
    timescale: u32,
    movie_timescale: u32,
) -> bool {
    let Some(entries) = be_u32(elst, 4) else {
        return false;
    };
    if entries != 1 {
        return false;
    }
    // The rate is 16.16 fixed point: its integer part and its fraction.
    let (segment_duration, media_time, rate, fraction) = match elst.first() {
        Some(0) => match (
            be_u32(elst, 8),
            be_u32(elst, 12),
            be_u16(elst, 16),
            be_u16(elst, 18),
        ) {
            (Some(d), Some(t), Some(r), Some(f)) => (u64::from(d), i64::from(t as i32), r, f),
            _ => return false,
        },
        Some(1) => match (
            be_u64(elst, 8),
            be_u64(elst, 16),
            be_u16(elst, 24),
            be_u16(elst, 26),
        ) {
            (Some(d), Some(t), Some(r), Some(f)) => (d, t as i64, r, f),
            _ => return false,
        },
        _ => return false,
    };
    // media_time 0 at rate 1.0 exactly, and the edit's duration (in the
    // movie's time scale) equal to the track's (in the media's).
    media_time == 0
        && rate == 1
        && fraction == 0
        && u128::from(segment_duration) * u128::from(timescale)
            == u128::from(duration_ts) * u128::from(movie_timescale)
}

/// `(width, height)` of the first visual sample entry of an `stsd` box.
fn sample_entry_size(stsd: &[u8]) -> Option<(u32, u32)> {
    if be_u32(stsd, 4)? == 0 {
        return None;
    }
    // After the full-box header and the entry count: the first sample entry,
    // a box whose visual payload carries the size at offsets 24 and 26.
    let entry = &stsd[8..];
    let size = be_u32(entry, 0)? as usize;
    if size < 8 + 28 || size > entry.len() {
        return None;
    }
    let payload = &entry[8..size];
    let width = u32::from(be_u16(payload, 24)?);
    let height = u32::from(be_u16(payload, 26)?);
    (width > 0 && height > 0).then_some((width, height))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(&kind);
        out.extend_from_slice(payload);
        out
    }

    fn full(version: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![version, 0, 0, 0];
        out.extend_from_slice(body);
        out
    }

    /// A movie of `frames` frames of `delta` ticks at `timescale`, one video
    /// track, `width` x `height`, an optional edit list.
    fn movie(
        timescale: u32,
        delta: u32,
        frames: u32,
        width: u16,
        height: u16,
        edit: Option<Vec<u8>>,
        stts_entries: Vec<(u32, u32)>,
    ) -> Vec<u8> {
        let duration = u64::from(delta) * u64::from(frames);
        let movie_timescale = 1000u32;
        let movie_duration = (duration * u64::from(movie_timescale) / u64::from(timescale)) as u32;
        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&[0u8; 8]);
        mvhd.extend_from_slice(&movie_timescale.to_be_bytes());
        mvhd.extend_from_slice(&movie_duration.to_be_bytes());
        mvhd.extend_from_slice(&[0u8; 80]);
        let mut mdhd = Vec::new();
        mdhd.extend_from_slice(&[0u8; 8]);
        mdhd.extend_from_slice(&timescale.to_be_bytes());
        mdhd.extend_from_slice(&(duration as u32).to_be_bytes());
        mdhd.extend_from_slice(&[0u8; 4]);
        let mut hdlr = vec![0u8; 4];
        hdlr.extend_from_slice(b"vide");
        hdlr.extend_from_slice(&[0u8; 13]);
        let mut avc1 = vec![0u8; 24];
        avc1.extend_from_slice(&width.to_be_bytes());
        avc1.extend_from_slice(&height.to_be_bytes());
        avc1.extend_from_slice(&[0u8; 50]);
        let mut stsd = 1u32.to_be_bytes().to_vec();
        stsd.extend_from_slice(&boxed(*b"avc1", &avc1));
        let mut stts = (stts_entries.len() as u32).to_be_bytes().to_vec();
        for (count, d) in &stts_entries {
            stts.extend_from_slice(&count.to_be_bytes());
            stts.extend_from_slice(&d.to_be_bytes());
        }
        let mut stsz = 0u32.to_be_bytes().to_vec();
        stsz.extend_from_slice(&frames.to_be_bytes());
        let stbl = [
            boxed(*b"stsd", &full(0, &stsd)),
            boxed(*b"stts", &full(0, &stts)),
            boxed(*b"stsz", &full(0, &stsz)),
        ]
        .concat();
        let minf = boxed(*b"stbl", &stbl);
        let mdia = [
            boxed(*b"mdhd", &full(0, &mdhd)),
            boxed(*b"hdlr", &full(0, &hdlr)),
            boxed(*b"minf", &minf),
        ]
        .concat();
        let mut trak = Vec::new();
        if let Some(elst) = edit {
            trak.extend_from_slice(&boxed(*b"edts", &boxed(*b"elst", &elst)));
        }
        trak.extend_from_slice(&boxed(*b"mdia", &mdia));
        let moov = [boxed(*b"mvhd", &full(0, &mvhd)), boxed(*b"trak", &trak)].concat();
        let mut file = boxed(*b"ftyp", b"isom\0\0\0\0isom");
        file.extend_from_slice(&boxed(*b"mdat", &[7u8; 40]));
        file.extend_from_slice(&boxed(*b"moov", &moov));
        file
    }

    fn whole_track_edit(movie_duration_ms: u32) -> Vec<u8> {
        let mut elst = full(0, &1u32.to_be_bytes());
        elst.extend_from_slice(&movie_duration_ms.to_be_bytes());
        elst.extend_from_slice(&0u32.to_be_bytes());
        elst.extend_from_slice(&1u16.to_be_bytes());
        elst.extend_from_slice(&0u16.to_be_bytes());
        elst
    }

    fn written(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(bytes).expect("write");
        file
    }

    /// A 64-bit box size chosen so that `offset + size` wraps to zero: the
    /// walk rejects the file instead of starting over at the top forever.
    #[test]
    fn a_box_size_that_wraps_the_offset_is_rejected_not_walked() {
        let mut bytes = boxed(*b"ftyp", &[0u8; 8]);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.extend_from_slice(&(u64::MAX - 15).to_be_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let file = written(&bytes);
        assert!(read(file.path()).expect("readable").is_none());

        // The same wrap inside a payload: an empty box first, so the 64-bit
        // size starts at offset 8 and `8 + (2^64 - 8)` wraps to zero.
        let mut payload = boxed(*b"free", &[]);
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(b"free");
        payload.extend_from_slice(&(u64::MAX - 7).to_be_bytes());
        assert!(boxes(&payload).is_none());
    }

    /// An edit at rate 1.5 (`0x0001_8000`) is not the whole track at normal
    /// speed, even though its integer part reads 1.
    #[test]
    fn an_edit_at_a_fractional_rate_keeps_the_probe() {
        let mut elst = whole_track_edit(1000);
        let end = elst.len();
        elst[end - 2..].copy_from_slice(&0x8000u16.to_be_bytes());
        let file = written(&movie(
            12288,
            512,
            24,
            320,
            240,
            Some(elst),
            vec![(24, 512)],
        ));
        assert!(read(file.path()).expect("readable").is_none());
    }

    /// A 24 fps track of 24 frames with the whole-track edit list: the
    /// probe's numbers (1.000000 s, 24/1, 24 frames, 320x240).
    #[test]
    fn a_constant_rate_track_is_read_like_the_probe() {
        let file = written(&movie(
            12288,
            512,
            24,
            320,
            240,
            Some(whole_track_edit(1000)),
            vec![(24, 512)],
        ));
        let info = read(file.path()).expect("read").expect("accepted");
        assert_eq!((info.width, info.height), (320, 240));
        assert_eq!((info.duration_ts, info.timescale), (12288, 12288));
        assert_eq!(info.frame_rate, (24, 1));
        assert_eq!(info.frame_count, 24);
        assert_eq!(info.stream_duration_seconds(), Some(1.0));
        assert_eq!(info.movie_duration_seconds(), Some(1.0));
        assert_eq!(info.frames_per_second(), Some(24.0));
    }

    /// NTSC rates reduce the way the demuxer reduces them and print the way
    /// the tool prints them.
    #[test]
    fn fractional_rates_reduce_and_round_like_the_tool() {
        let file = written(&movie(30000, 1001, 300, 1280, 720, None, vec![(300, 1001)]));
        let info = read(file.path()).expect("read").expect("accepted");
        assert_eq!(info.frame_rate, (30000, 1001));
        assert_eq!(info.stream_duration_seconds(), Some(10.01));
        assert_eq!(info.frames_per_second(), Some(30000.0 / 1001.0));
    }

    /// Shapes the module does not vouch for keep the probe process.
    #[test]
    fn other_shapes_are_declined() {
        let vfr = movie(12288, 512, 25, 320, 240, None, vec![(24, 512), (1, 1024)]);
        assert_eq!(read(written(&vfr).path()).expect("read"), None);
        let mut offset_edit = full(0, &1u32.to_be_bytes());
        offset_edit.extend_from_slice(&1000u32.to_be_bytes());
        offset_edit.extend_from_slice(&512u32.to_be_bytes());
        offset_edit.extend_from_slice(&[0, 1, 0, 0]);
        let edited = movie(12288, 512, 24, 320, 240, Some(offset_edit), vec![(24, 512)]);
        assert_eq!(read(written(&edited).path()).expect("read"), None);
        assert_eq!(
            read(written(b"RIFF\0\0\0\0WEBPVP8 ").path()).expect("read"),
            None
        );
        assert_eq!(read(written(&[]).path()).expect("read"), None);
    }
}

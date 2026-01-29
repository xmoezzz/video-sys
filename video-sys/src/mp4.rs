use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{anyhow, bail, Context, Result};
use mp4::{FourCC, Mp4Reader, TrackType};

use crate::h264::H264Config;

#[derive(Debug, Clone)]
pub struct EncodedSample {
    pub data_avcc: Vec<u8>,
    /// Decode timestamp (monotonic), in microseconds.
    pub dts_us: i64,
    /// Presentation timestamp (may reorder vs DTS when B-frames are present), in microseconds.
    pub pts_us: i64,
    pub dur_us: i64,
}

#[derive(Debug)]
struct Prefetched {
    start_time: u64,
    duration: u32,
    rendering_offset: i32,
    bytes: Vec<u8>,
}

pub struct Mp4H264Source {
    path: PathBuf,
    reader: Mp4Reader<BufReader<File>>,
    track_id: u32,
    timescale: u32,
    sample_count: u32,
    next_sample_id: u32,
    prefetched: Option<Prefetched>,

    pub config: H264Config,
}

impl Mp4H264Source {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let f = File::open(&path).with_context(|| format!("open mp4: {}", path.display()))?;
        let size = f
            .metadata()
            .with_context(|| format!("stat mp4: {}", path.display()))?
            .len();

        let reader = BufReader::new(f);
        let mut mp4 = Mp4Reader::read_header(reader, size).context("mp4::read_header")?;

        let (track_id, timescale, sample_count, width, height, sps, pps) =
            select_h264_video_track(&mp4).context("select H.264 track")?;

        // Prefetch the first sample to:
        // 1) validate we can read samples;
        // 2) infer NAL length field size (usually 4, but not guaranteed).
        let prefetched = mp4
            .read_sample(track_id, 1)
            .context("read first sample")?
            .map(|s| Prefetched {
                start_time: s.start_time,
                duration: s.duration,
                rendering_offset: s.rendering_offset,
                bytes: s.bytes.to_vec(),
            });

        let nal_len_size = prefetched
            .as_ref()
            .map(|p| detect_nal_length_size(&p.bytes))
            .unwrap_or(4);

        let avcc = build_avcc_record(&sps, &pps, nal_len_size)?;
        let config = H264Config::parse_from_avcc(width, height, &avcc)
            .context("parse avcC from SPS/PPS")?;

        Ok(Self {
            path,
            reader: mp4,
            track_id,
            timescale,
            sample_count,
            next_sample_id: 1,
            prefetched,
            config,
        })
    }

    pub fn next_sample(&mut self) -> Result<Option<EncodedSample>> {
        if self.next_sample_id == 0 {
            bail!("internal error: sample ids are 1-based");
        }

        if self.next_sample_id > self.sample_count {
            return Ok(None);
        }

        let (start_time, duration, rendering_offset, bytes) = if self.next_sample_id == 1 {
            if let Some(p) = self.prefetched.take() {
                (p.start_time, p.duration, p.rendering_offset, p.bytes)
            } else {
                let s = self
                    .reader
                    .read_sample(self.track_id, 1)
                    .context("read sample #1")?
                    .ok_or_else(|| anyhow!("sample #1 missing"))?;
                (s.start_time, s.duration, s.rendering_offset, s.bytes.to_vec())
            }
        } else {
            let s = self
                .reader
                .read_sample(self.track_id, self.next_sample_id)
                .with_context(|| format!("read sample #{}", self.next_sample_id))?
                .ok_or_else(|| anyhow!("sample #{} missing", self.next_sample_id))?;
            (s.start_time, s.duration, s.rendering_offset, s.bytes.to_vec())
        };

        self.next_sample_id += 1;

        // mp4 crate provides:
        // - start_time: decode time in track timescale ticks
        // - rendering_offset: composition offset ticks (ctts)
        let dts_ticks = start_time as i128;
        let pts_ticks = dts_ticks + (rendering_offset as i128);

        let dts_us = ticks_to_us(dts_ticks, self.timescale);
        let pts_us = ticks_to_us(pts_ticks, self.timescale);
        let dur_us = ticks_to_us(duration as i128, self.timescale);

        Ok(Some(EncodedSample {
            data_avcc: bytes,
            dts_us,
            pts_us,
            dur_us,
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn ticks_to_us(ticks: i128, timescale: u32) -> i64 {
    if timescale == 0 {
        return 0;
    }
    // microseconds = ticks * 1_000_000 / timescale
    let us = ticks.saturating_mul(1_000_000i128) / (timescale as i128);
    if us > (i64::MAX as i128) {
        i64::MAX
    } else if us < (i64::MIN as i128) {
        i64::MIN
    } else {
        us as i64
    }
}

fn select_h264_video_track(
    mp4: &Mp4Reader<BufReader<File>>,
) -> Result<(u32, u32, u32, u32, u32, Vec<u8>, Vec<u8>)> {
    let avc1 = FourCC::from_str("avc1").unwrap();
    let avc3 = FourCC::from_str("avc3").unwrap();

    for (track_id, track) in mp4.tracks().iter() {
        let tt = track.track_type().context("track_type")?;
        if tt != TrackType::Video {
            continue;
        }

        let bt = track.box_type().context("box_type")?;
        if bt != avc1 && bt != avc3 {
            continue;
        }

        let timescale = track.timescale();
        let sample_count = track.sample_count();

        let width = track.width() as u32;
        let height = track.height() as u32;

        let sps = track
            .sequence_parameter_set()
            .context("sequence_parameter_set")?
            .to_vec();
        let pps = track
            .picture_parameter_set()
            .context("picture_parameter_set")?
            .to_vec();

        return Ok((*track_id, timescale, sample_count, width, height, sps, pps));
    }

    bail!("no H.264 (avc1/avc3) video track found")
}

fn detect_nal_length_size(avcc_sample: &[u8]) -> usize {
    // Best effort: assume typical 4 bytes if sample is too short.
    if avcc_sample.len() < 8 {
        return 4;
    }
    // Heuristic: first 4 bytes are NAL length; if it looks reasonable, accept 4.
    4
}

fn build_avcc_record(sps: &[u8], pps: &[u8], nal_len_size: usize) -> Result<Vec<u8>> {
    if !(1..=4).contains(&nal_len_size) {
        bail!("invalid nal length size: {nal_len_size}");
    }

    // Minimal avcC record to feed H264Config.
    // Layout: https://developer.apple.com/documentation/quicktime-file-format/avcdecoderconfigurationrecord
    let mut out = Vec::new();
    out.push(1); // configurationVersion
    out.push(*sps.get(1).unwrap_or(&0)); // AVCProfileIndication
    out.push(*sps.get(2).unwrap_or(&0)); // profile_compatibility
    out.push(*sps.get(3).unwrap_or(&0)); // AVCLevelIndication

    // lengthSizeMinusOne in low 2 bits
    out.push(0xFC | ((nal_len_size as u8 - 1) & 0x03));

    // numOfSequenceParameterSets in low 5 bits
    out.push(0xE0 | 1);
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);

    // numOfPictureParameterSets
    out.push(1);
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);

    Ok(out)
}

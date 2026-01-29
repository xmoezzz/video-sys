use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::backend::{create_default_h264_decoder, H264Decoder};
use crate::mp4::{EncodedSample, Mp4H264Source};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit RGBA (R,G,B,A in memory order).
    Rgba8,
    /// 8-bit BGRA (B,G,R,A in memory order).
    Bgra8,
}

/// Frame payload bytes.
///
/// Some backends (notably macOS VideoToolbox) can allocate multi-megabyte
/// buffers per frame. Repeated allocations can cause RSS growth due to allocator
/// behavior. `FrameData` optionally carries a pool handle so the buffer can be
/// recycled on drop, keeping memory stable.
#[derive(Debug)]
pub struct FrameData {
    buf: Vec<u8>,
    pool: Option<Arc<parking_lot::Mutex<Vec<Vec<u8>>>>>,
    pool_cap: usize,
}

impl FrameData {
    pub fn new(buf: Vec<u8>) -> Self {
        Self {
            buf,
            pool: None,
            pool_cap: 0,
        }
    }

    pub fn with_pool(
        buf: Vec<u8>,
        pool: Arc<parking_lot::Mutex<Vec<Vec<u8>>>>,
        pool_cap: usize,
    ) -> Self {
        Self {
            buf,
            pool: Some(pool),
            pool_cap,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Detach from the pool and return the owned `Vec<u8>`.
    pub fn into_vec(mut self) -> Vec<u8> {
        self.pool = None;
        std::mem::take(&mut self.buf)
    }
}

impl std::ops::Deref for FrameData {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for FrameData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        let Some(pool) = self.pool.take() else { return; };
        if self.pool_cap == 0 {
            return;
        }
        // Return buffer to pool if there's room.
        let mut g = pool.lock();
        if g.len() < self.pool_cap {
            let mut v = Vec::new();
            std::mem::swap(&mut v, &mut self.buf);
            g.push(v);
        }
    }
}

#[derive(Debug)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub pts_us: i64,
    pub format: PixelFormat,
    /// Tight-packed pixel buffer.
    ///
    /// For `Rgba8` or `Bgra8`, length is `width * height * 4`.
    pub data: FrameData,
}

pub struct VideoCore {
    path: PathBuf,
    src: Mp4H264Source,
    dec: Box<dyn H264Decoder>,

    pending: Option<EncodedSample>,
    stash: VecDeque<VideoFrame>,

    eof: bool,
    flushed: bool,
}

impl VideoCore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let src = Mp4H264Source::open(&path).context("open mp4 source")?;
        let dec = create_default_h264_decoder(&src.config).context("create decoder backend")?;

        Ok(Self {
            path,
            src,
            dec,
            pending: None,
            stash: VecDeque::new(),
            eof: false,
            flushed: false,
        })
    }

    pub fn width(&self) -> u32 {
        self.src.config.width
    }

    pub fn height(&self) -> u32 {
        self.src.config.height
    }

    pub fn is_eof(&self) -> bool {
        self.eof
    }

    pub fn reset(&mut self) -> Result<()> {
        let src = Mp4H264Source::open(&self.path)?;
        let dec = create_default_h264_decoder(&src.config)?;
        self.src = src;
        self.dec = dec;

        self.pending = None;
        self.stash.clear();
        self.eof = false;
        self.flushed = false;
        Ok(())
    }

    /// Feed some compressed samples into the decoder and drain all available decoded frames.
    ///
    /// This is a pure "producer" pump: it does NOT follow wall-clock or playhead.
    /// The renderer/player should decide what to present based on PTS.
    pub fn pump(&mut self) -> Result<()> {
        // Feed a bounded number of samples per pump to avoid monopolizing CPU.
        const FEED_BUDGET: usize = 4;

        if !self.eof {
            for _ in 0..FEED_BUDGET {
                match self.next_sample_cached()? {
                    Some(s) => {
                        self.dec.push(s)?;
                    }
                    None => {
                        self.eof = true;
                        if !self.flushed {
                            self.dec.flush()?;
                            self.flushed = true;
                        }
                        break;
                    }
                }
            }
        }

        // Drain decoder outputs into stash (do NOT drop anything here).
        while let Some(f) = self.dec.try_receive()? {
            self.stash.push_back(VideoFrame {
                width: f.width,
                height: f.height,
                pts_us: f.pts_us,
                format: f.format,
                data: f.data,
            });
        }

        Ok(())
    }

    /// Pop the next decoded frame in presentation order (if any).
    pub fn pop_decoded(&mut self) -> Option<VideoFrame> {
        self.stash.pop_front()
    }

    /// Finished means: we reached EOF, flushed, and no pending input/output remains.
    pub fn is_finished(&self) -> bool {
        self.eof && self.flushed && self.pending.is_none() && self.stash.is_empty()
    }

    fn next_sample_cached(&mut self) -> Result<Option<EncodedSample>> {
        if let Some(s) = self.pending.take() {
            return Ok(Some(s));
        }
        self.src.next_sample()
    }
}

#[derive(Debug)]
pub struct VideoStream {
    width: u32,
    height: u32,
    rx: Receiver<VideoFrame>,
    stop: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl VideoStream {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, VideoStreamOptions::default())
    }

    pub fn open_with_options(path: impl AsRef<Path>, opt: VideoStreamOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let src = Mp4H264Source::open(&path).context("open mp4 source for config")?;
        let width = src.config.width;
        let height = src.config.height;
        drop(src);

        let (tx, rx) = crossbeam_channel::bounded::<VideoFrame>(opt.channel_depth.max(1));
        let stop = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));

        let stop_t = stop.clone();
        let finished_t = finished.clone();
        let path_t = path.clone();

        // Helper functions for PTS reordering.
        // NOTE: these are plain functions (not closures) to avoid borrowing `reorder`
        // mutably from multiple closures at the same time.
        fn push_reorder(
            reorder: &mut BTreeMap<i64, VecDeque<VideoFrame>>,
            reorder_len: &mut usize,
            opt: &VideoStreamOptions,
            f: VideoFrame,
        ) {
            reorder.entry(f.pts_us).or_default().push_back(f);
            *reorder_len += 1;

            // Hard cap: if something goes wrong (consumer stalls, timestamps weird),
            // never allow unbounded growth.
            while *reorder_len > opt.reorder_max_frames {
                let k = match reorder.keys().next().copied() {
                    Some(k) => k,
                    None => break,
                };

                let mut remove_key = false;
                let dropped = {
                    let q = match reorder.get_mut(&k) {
                        Some(q) => q,
                        None => break,
                    };
                    let dropped = q.pop_front().is_some();
                    if q.is_empty() {
                        remove_key = true;
                    }
                    dropped
                };

                if remove_key {
                    reorder.remove(&k);
                }

                if dropped {
                    *reorder_len = (*reorder_len).saturating_sub(1);
                } else {
                    // If we couldn't drop anything, stop to avoid spinning.
                    break;
                }
            }
        }

        fn pop_next_ready(
            reorder: &mut BTreeMap<i64, VecDeque<VideoFrame>>,
            reorder_len: &mut usize,
        ) -> Option<VideoFrame> {
            let k = reorder.keys().next().copied()?;

            let mut remove_key = false;
            let f = {
                let q = reorder.get_mut(&k)?;
                let f = q.pop_front();
                if q.is_empty() {
                    remove_key = true;
                }
                f
            };

            if remove_key {
                reorder.remove(&k);
            }
            if f.is_some() {
                *reorder_len = (*reorder_len).saturating_sub(1);
            }
            f
        }

        let join = thread::spawn(move || {
            let mut core = match VideoCore::open(&path_t) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("VideoCore::open failed in decode thread: {e:?}");
                    finished_t.store(true, Ordering::Relaxed);
                    return;
                }
            };

            // PTS reordering buffer: some decoders (depending on backend and flags)
            // can output frames in decode order. For H.264 with B-frames, PTS is
            // not monotonic in decode order. We reorder by PTS with a small window.
            let mut reorder: BTreeMap<i64, VecDeque<VideoFrame>> = BTreeMap::new();
            let mut reorder_len: usize = 0;
            let mut started_at: Option<std::time::Instant> = None;
            let mut base_pts_us: i64 = 0;

            let mut pace_next = |pts_us: i64| {
                if !opt.paced {
                    return;
                }
                let now = std::time::Instant::now();
                if started_at.is_none() {
                    started_at = Some(now);
                    base_pts_us = pts_us;
                    return;
                }
                let delta_us = pts_us.saturating_sub(base_pts_us).max(0) as u64;
                let target = started_at.unwrap() + Duration::from_micros(delta_us);
                if target > now {
                    // Sleep in small slices so we can still observe stop signals.
                    let mut remaining = target.duration_since(now);
                    while remaining > Duration::from_millis(5) {
                        thread::sleep(Duration::from_millis(5));
                        if stop_t.load(Ordering::Relaxed) {
                            return;
                        }
                        remaining = target.saturating_duration_since(std::time::Instant::now());
                    }
                    if remaining > Duration::from_micros(0) {
                        thread::sleep(remaining);
                    }
                }
            };

            // `push_reorder`/`pop_next_ready` are plain fns (defined above).

            loop {
                if stop_t.load(Ordering::Relaxed) {
                    break;
                }

                // 1) Ensure we have a small amount of decoded frames buffered.
                let want = opt.ahead_frames.max(opt.reorder_depth_frames);
                while reorder_len < want {
                    if let Err(e) = core.pump() {
                        log::error!("video decode thread pump error: {e:?}");
                        finished_t.store(true, Ordering::Relaxed);
                        return;
                    }
                    let mut produced_any = false;
                    while let Some(frame) = core.pop_decoded() {
                        produced_any = true;
                        push_reorder(&mut reorder, &mut reorder_len, &opt, frame);
                    }
                    if !produced_any {
                        break;
                    }
                }

                // 2) Ship the next frame (paced) or idle.
                // We wait until we have enough frames to safely reorder, unless at EOF.
                let have_ready = reorder_len >= opt.reorder_depth_frames || core.is_finished();
                if have_ready {
                    if let Some(mut frame) = pop_next_ready(&mut reorder, &mut reorder_len) {
                        pace_next(frame.pts_us);

                        if stop_t.load(Ordering::Relaxed) {
                            break;
                        }

                        // Try to send; if consumer is behind, drop frames (real-time).
                        match tx.try_send(frame) {
                            Ok(()) => {}
                            Err(TrySendError::Full(f)) => {
                                // Drop this frame to avoid unbounded latency.
                                drop(f);
                                thread::yield_now();
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                finished_t.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    } else {
                        thread::sleep(Duration::from_millis(1));
                    }
                } else {
                    // No decoded frames available yet.
                    thread::sleep(Duration::from_millis(1));
                }

                if core.is_finished() && reorder_len == 0 {
                    finished_t.store(true, Ordering::Relaxed);
                    break;
                }
            }
        });

        Ok(Self {
            width,
            height,
            rx,
            stop,
            finished,
            join: Some(join),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }

    pub fn try_recv_one(&self) -> Option<VideoFrame> {
        match self.rx.try_recv() {
            Ok(f) => Some(f),
            Err(_) => None,
        }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoStreamOptions {
    /// If true, the stream will pace frames according to PTS.
    pub paced: bool,
    /// Channel depth between decode thread and consumer.
    pub channel_depth: usize,
    /// How many decoded frames to keep buffered ahead.
    pub ahead_frames: usize,

    /// Reorder window (frames). If output PTS is already monotonic, this is harmless.
    pub reorder_depth_frames: usize,

    /// Hard cap to prevent unbounded memory growth due to timestamp/pathological cases.
    pub reorder_max_frames: usize,
}

impl Default for VideoStreamOptions {
    fn default() -> Self {
        Self {
            paced: true,
            channel_depth: 8,
            ahead_frames: 6,
            reorder_depth_frames: 16,
            reorder_max_frames: 64,
        }
    }
}

impl Drop for VideoStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

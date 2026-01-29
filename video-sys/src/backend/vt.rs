use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;

use crate::backend::{H264Decoder, VideoFrame};
use crate::core::{FrameData, PixelFormat};
use crate::h264::H264Config;
use crate::mp4::EncodedSample;
use crate::pixel::nv12_to_rgba_strided;

type OSStatus = i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

impl CMTime {
    fn from_us(us: i64) -> Self {
        Self {
            value: us,
            timescale: 1_000_000,
            flags: 1, // kCMTimeFlags_Valid
            epoch: 0,
        }
    }

    fn to_us(self) -> i64 {
        if self.timescale == 0 {
            return 0;
        }
        (self.value as i128 * 1_000_000i128 / self.timescale as i128) as i64
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CMSampleTimingInfo {
    duration: CMTime,
    presentationTimeStamp: CMTime,
    decodeTimeStamp: CMTime,
}

type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFNumberRef = *const c_void;
type CFStringRef = *const c_void;

#[repr(C)]
struct CFDictionaryKeyCallBacks {
    _private: [u8; 0],
}

#[repr(C)]
struct CFDictionaryValueCallBacks {
    _private: [u8; 0],
}

type CMVideoFormatDescriptionRef = *mut c_void;
type CMBlockBufferRef = *mut c_void;
type CMSampleBufferRef = *mut c_void;

type VTDecompressionSessionRef = *mut c_void;
type CVImageBufferRef = *mut c_void;
type CVPixelBufferRef = *mut c_void;

type VTDecodeFrameFlags = u32;
type VTDecodeInfoFlags = u32;

const kVTDecodeFrame_EnableAsynchronousDecompression: VTDecodeFrameFlags = 1 << 0;

const kCVPixelBufferLock_ReadOnly: u64 = 0x00000001;

// OSType for BGRA.
const kCVPixelFormatType_32BGRA: u32 = 0x4247_5241; // 'BGRA'

// CFNumberType: kCFNumberSInt32Type = 3
const kCFNumberSInt32Type: i32 = 3;

// Keep the output queue bounded to prevent unbounded memory growth when the consumer stalls.
const MAX_OUT_FRAMES: usize = 16;
// Reuse a small number of big BGRA/RGBA buffers to keep RSS stable.
const POOL_CAP: usize = 32;

#[repr(C)]
struct VTDecompressionOutputCallbackRecord {
    decompressionOutputCallback: Option<
        unsafe extern "C" fn(
            decompressionOutputRefCon: *mut c_void,
            sourceFrameRefCon: *mut c_void,
            status: OSStatus,
            infoFlags: VTDecodeInfoFlags,
            imageBuffer: CVImageBufferRef,
            presentationTimeStamp: CMTime,
            presentationDuration: CMTime,
        ),
    >,
    decompressionOutputRefCon: *mut c_void,
}

#[link(name = "VideoToolbox", kind = "framework")]
#[link(name = "CoreMedia", kind = "framework")]
#[link(name = "CoreVideo", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CFAllocatorRef,
        parameterSetCount: usize,
        parameterSetPointers: *const *const u8,
        parameterSetSizes: *const usize,
        nalUnitHeaderLength: i32,
        formatDescriptionOut: *mut CMVideoFormatDescriptionRef,
    ) -> OSStatus;

    fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        videoFormatDescription: CMVideoFormatDescriptionRef,
        decoderSpecification: CFDictionaryRef,
        imageBufferAttributes: CFDictionaryRef,
        outputCallback: *const VTDecompressionOutputCallbackRecord,
        decompressionSessionOut: *mut VTDecompressionSessionRef,
    ) -> OSStatus;

    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sampleBuffer: CMSampleBufferRef,
        decodeFlags: VTDecodeFrameFlags,
        sourceFrameRefCon: *mut c_void,
        infoFlagsOut: *mut VTDecodeInfoFlags,
    ) -> OSStatus;

    fn VTDecompressionSessionWaitForAsynchronousFrames(session: VTDecompressionSessionRef) -> OSStatus;

    fn CMBlockBufferCreateWithMemoryBlock(
        allocator: CFAllocatorRef,
        memoryBlock: *mut c_void,
        blockLength: usize,
        blockAllocator: CFAllocatorRef,
        customBlockSource: *const c_void,
        offsetToData: usize,
        dataLength: usize,
        flags: u32,
        blockBufferOut: *mut CMBlockBufferRef,
    ) -> OSStatus;

    fn CMBlockBufferReplaceDataBytes(
        sourceBytes: *const c_void,
        destinationBuffer: CMBlockBufferRef,
        offsetIntoDestination: usize,
        dataLength: usize,
    ) -> OSStatus;

    fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        dataBuffer: CMBlockBufferRef,
        formatDescription: CMVideoFormatDescriptionRef,
        numSamples: i64,
        numSampleTimingEntries: usize,
        sampleTimingArray: *const CMSampleTimingInfo,
        numSampleSizeEntries: usize,
        sampleSizeArray: *const usize,
        sampleBufferOut: *mut CMSampleBufferRef,
    ) -> OSStatus;

    fn CFRelease(cf: *const c_void);

    // CoreFoundation helpers for imageBufferAttributes
    static kCFTypeDictionaryKeyCallBacks: CFDictionaryKeyCallBacks;
    static kCFTypeDictionaryValueCallBacks: CFDictionaryValueCallBacks;

    fn CFNumberCreate(allocator: CFAllocatorRef, theType: i32, valuePtr: *const c_void) -> CFNumberRef;

    fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const *const c_void,
        values: *const *const c_void,
        numValues: isize,
        keyCallBacks: *const CFDictionaryKeyCallBacks,
        valueCallBacks: *const CFDictionaryValueCallBacks,
    ) -> CFDictionaryRef;

    // CoreVideo constants
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;

    fn CVPixelBufferLockBaseAddress(pixelBuffer: CVPixelBufferRef, lockFlags: u64) -> OSStatus;
    fn CVPixelBufferUnlockBaseAddress(pixelBuffer: CVPixelBufferRef, lockFlags: u64) -> OSStatus;
    fn CVPixelBufferGetPlaneCount(pixelBuffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(pixelBuffer: CVPixelBufferRef, planeIndex: usize) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer: CVPixelBufferRef, planeIndex: usize) -> usize;
    fn CVPixelBufferGetBaseAddress(pixelBuffer: CVPixelBufferRef) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pixelBuffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidth(pixelBuffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pixelBuffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetPixelFormatType(pixelBuffer: CVPixelBufferRef) -> u32;
}

struct CallbackCtx {
    queue: Mutex<VecDeque<VideoFrame>>,
    pool: Arc<parking_lot::Mutex<Vec<Vec<u8>>>>,
}

fn bgra_to_rgba_inplace(p: &mut [u8]) {
    // BGRA -> RGBA by swapping B and R.
    // Layout per pixel: [B, G, R, A]
    for px in p.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
}

unsafe extern "C" fn vt_output_cb(
    decompressionOutputRefCon: *mut c_void,
    _sourceFrameRefCon: *mut c_void,
    status: OSStatus,
    _infoFlags: VTDecodeInfoFlags,
    imageBuffer: CVImageBufferRef,
    presentationTimeStamp: CMTime,
    _presentationDuration: CMTime,
) {
    if status != 0 || imageBuffer.is_null() {
        return;
    }

    let ctx = unsafe { &*(decompressionOutputRefCon as *const CallbackCtx) };
    let pb = imageBuffer as CVPixelBufferRef;

    unsafe {
        if CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != 0 {
            return;
        }

        let width = CVPixelBufferGetWidth(pb) as u32;
        let height = CVPixelBufferGetHeight(pb) as u32;
        let planes = CVPixelBufferGetPlaneCount(pb);
        let fmt = CVPixelBufferGetPixelFormatType(pb);

        let mut out: Option<(PixelFormat, FrameData)> = None;

        // Preferred: BGRA (single-plane).
        if planes == 0 && fmt == kCVPixelFormatType_32BGRA {
            let base = CVPixelBufferGetBaseAddress(pb) as *const u8;
            let stride = CVPixelBufferGetBytesPerRow(pb);
            if !base.is_null() {
                let need = (width as usize) * (height as usize) * 4;
                // Acquire a buffer from the pool (or allocate).
                let mut buf = {
                    let mut g = ctx.pool.lock();
                    g.pop().unwrap_or_else(|| vec![0u8; need])
                };
                buf.resize(need, 0u8);
                let row_bytes = (width as usize) * 4;

                for y in 0..(height as usize) {
                    let src = base.add(y * stride);
                    let dst = buf.as_mut_ptr().add(y * row_bytes);
                    ptr::copy_nonoverlapping(src, dst, row_bytes);
                }

                // Keep the public contract consistent: output RGBA.
                bgra_to_rgba_inplace(&mut buf);
                out = Some((
                    PixelFormat::Rgba8,
                    FrameData::with_pool(buf, ctx.pool.clone(), POOL_CAP),
                ));
            }
        }

        // Fallback: NV12 (bi-planar).
        if out.is_none() && planes >= 2 {
            let y_ptr = CVPixelBufferGetBaseAddressOfPlane(pb, 0) as *const u8;
            let uv_ptr = CVPixelBufferGetBaseAddressOfPlane(pb, 1) as *const u8;
            let y_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
            let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);

            if !y_ptr.is_null() && !uv_ptr.is_null() {
                let y_len = y_stride * height as usize;
                let uv_len = uv_stride * (height as usize / 2);
                let y = std::slice::from_raw_parts(y_ptr, y_len);
                let uv = std::slice::from_raw_parts(uv_ptr, uv_len);

                out = Some((
                    PixelFormat::Rgba8,
                    FrameData::new(nv12_to_rgba_strided(
                        width,
                        height,
                        y_stride,
                        uv_stride,
                        y,
                        uv,
                    )),
                ));
            }
        }

        if let Some((format, data)) = out {
            let mut q = ctx.queue.lock();
            while q.len() >= MAX_OUT_FRAMES {
                q.pop_front();
            }
            q.push_back(VideoFrame {
                width,
                height,
                pts_us: presentationTimeStamp.to_us(),
                format,
                data,
            });
        }

        let _ = CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    }
}

pub struct VtH264Decoder {
    _cfg: H264Config,
    session: VTDecompressionSessionRef,
    format_desc: CMVideoFormatDescriptionRef,
    ctx: *mut CallbackCtx,
}

unsafe impl Send for VtH264Decoder {}
unsafe impl Sync for VtH264Decoder {}

impl VtH264Decoder {
    pub fn new(cfg: H264Config) -> Result<Self> {
        if cfg.sps.is_empty() || cfg.pps.is_empty() {
            bail!("missing SPS/PPS");
        }

        let sps = &cfg.sps[0];
        let pps = &cfg.pps[0];
        let ps_ptrs: [*const u8; 2] = [sps.as_ptr(), pps.as_ptr()];
        let ps_sizes: [usize; 2] = [sps.len(), pps.len()];

        let mut format_desc: CMVideoFormatDescriptionRef = ptr::null_mut();
        let st = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                ptr::null(),
                2,
                ps_ptrs.as_ptr(),
                ps_sizes.as_ptr(),
                cfg.nal_length_size as i32,
                &mut format_desc as *mut _,
            )
        };
        if st != 0 || format_desc.is_null() {
            bail!(
                "CMVideoFormatDescriptionCreateFromH264ParameterSets failed: status={}",
                st
            );
        }

        let ctx = Box::new(CallbackCtx {
            queue: Mutex::new(VecDeque::new()),
            pool: Arc::new(Mutex::new(Vec::new())),
        });
        let ctx_ptr = Box::into_raw(ctx);

        let cb = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(vt_output_cb),
            decompressionOutputRefCon: ctx_ptr as *mut c_void,
        };

        // Request BGRA output to avoid NV12 conversion overhead on CPU.
        let bgra = kCVPixelFormatType_32BGRA as i32;
        let num = unsafe { CFNumberCreate(ptr::null(), kCFNumberSInt32Type, &bgra as *const _ as *const c_void) };
        if num.is_null() {
            unsafe {
                CFRelease(format_desc as *const _);
                drop(Box::from_raw(ctx_ptr));
            }
            bail!("CFNumberCreate failed");
        }

        let keys: [*const c_void; 1] = [unsafe { kCVPixelBufferPixelFormatTypeKey as *const c_void }];
        let vals: [*const c_void; 1] = [num as *const c_void];
        let dict = unsafe {
            CFDictionaryCreate(
                ptr::null(),
                keys.as_ptr(),
                vals.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks as *const _,
                &kCFTypeDictionaryValueCallBacks as *const _,
            )
        };

        if dict.is_null() {
            unsafe {
                CFRelease(num);
                CFRelease(format_desc as *const _);
                drop(Box::from_raw(ctx_ptr));
            }
            bail!("CFDictionaryCreate failed");
        }

        let mut session: VTDecompressionSessionRef = ptr::null_mut();
        let st = unsafe {
            VTDecompressionSessionCreate(
                ptr::null(),
                format_desc,
                ptr::null(),
                dict,
                &cb as *const _,
                &mut session as *mut _,
            )
        };

        unsafe {
            CFRelease(num);
            CFRelease(dict as *const _);
        }

        if st != 0 || session.is_null() {
            unsafe {
                CFRelease(format_desc as *const _);
                drop(Box::from_raw(ctx_ptr));
            }
            bail!("VTDecompressionSessionCreate failed: status={}", st);
        }

        Ok(Self {
            _cfg: cfg,
            session,
            format_desc,
            ctx: ctx_ptr,
        })
    }
}

impl Drop for VtH264Decoder {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                let _ = VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session as *const _);
                self.session = ptr::null_mut();
            }
            if !self.format_desc.is_null() {
                CFRelease(self.format_desc as *const _);
                self.format_desc = ptr::null_mut();
            }
            if !self.ctx.is_null() {
                drop(Box::from_raw(self.ctx));
                self.ctx = ptr::null_mut();
            }
        }
    }
}

impl H264Decoder for VtH264Decoder {
    fn push(&mut self, sample: EncodedSample) -> Result<()> {
        let mut block: CMBlockBufferRef = ptr::null_mut();
        let st = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                ptr::null(),
                ptr::null_mut(),
                sample.data_avcc.len(),
                ptr::null(),
                ptr::null(),
                0,
                sample.data_avcc.len(),
                0,
                &mut block as *mut _,
            )
        };
        if st != 0 || block.is_null() {
            bail!("CMBlockBufferCreateWithMemoryBlock failed: status={}", st);
        }

        let st = unsafe {
            CMBlockBufferReplaceDataBytes(
                sample.data_avcc.as_ptr() as *const c_void,
                block,
                0,
                sample.data_avcc.len(),
            )
        };
        if st != 0 {
            unsafe { CFRelease(block as *const _) };
            bail!("CMBlockBufferReplaceDataBytes failed: status={}", st);
        }

        // IMPORTANT:
        // - decodeTimeStamp must be DTS (monotonic)
        // - presentationTimeStamp must be PTS
        let timing = CMSampleTimingInfo {
            duration: CMTime::from_us(sample.dur_us.max(0)),
            presentationTimeStamp: CMTime::from_us(sample.pts_us.max(0)),
            decodeTimeStamp: CMTime::from_us(sample.dts_us.max(0)),
        };
        let sample_size = sample.data_avcc.len();

        let mut sbuf: CMSampleBufferRef = ptr::null_mut();
        let st = unsafe {
            CMSampleBufferCreateReady(
                ptr::null(),
                block,
                self.format_desc,
                1,
                1,
                &timing as *const _,
                1,
                &sample_size as *const _,
                &mut sbuf as *mut _,
            )
        };
        unsafe { CFRelease(block as *const _) };
        if st != 0 || sbuf.is_null() {
            bail!("CMSampleBufferCreateReady failed: status={}", st);
        }

        let mut info_flags: VTDecodeInfoFlags = 0;
        let st = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                sbuf,
                kVTDecodeFrame_EnableAsynchronousDecompression,
                ptr::null_mut(),
                &mut info_flags as *mut _,
            )
        };
        unsafe { CFRelease(sbuf as *const _) };
        if st != 0 {
            return Err(anyhow!("VTDecompressionSessionDecodeFrame failed: status={}", st));
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let st = unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        if st != 0 {
            bail!("VTDecompressionSessionWaitForAsynchronousFrames failed: status={}", st);
        }
        Ok(())
    }

    fn try_receive(&mut self) -> Result<Option<VideoFrame>> {
        let ctx = unsafe { &*(self.ctx as *const CallbackCtx) };
        Ok(ctx.queue.lock().pop_front())
    }
}

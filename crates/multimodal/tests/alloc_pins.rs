//! Allocation pins of the media pipeline, per image class and stage: what the decode,
//! the frame copy the gateway makes before the processor, the one-pass preprocess into
//! the wire's dtype and the move of the wire bytes allocate today, counted by a global
//! allocator. A pin holds the count where it stands, so a copy that comes back (or a new
//! one) fails the test; it is lowered on purpose when a copy goes.
//!
//! The counts are exact for the stages the crate owns outright (the copy, the preprocess
//! and the move) and run with every task inline on the test thread, so no pool thread
//! allocates into the window. The decode's own calls depend on which decoder is loaded
//! (TurboJPEG and libwebp are runtime bindings; their C allocations never reach this
//! allocator), so the decode pin counts the large buffers only: the one decoded frame.
//! A separate executable, so it can declare the counting allocator.
//!
//! `MM_PINS_RECORD=1 cargo test -p llm-multimodal --test alloc_pins -- --nocapture`
//! prints the table instead of asserting, for re-pinning after a change that moves a
//! count on purpose. The preprocess calls are pinned per resize row path (scalar, NEON):
//! a re-pin records each count on its own architecture, and a failing run prints its
//! table with the failure.
#![expect(unsafe_code, reason = "a global allocator that counts allocations")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    io::Cursor,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
};

use bytes::Bytes;
use image::{DynamicImage, ImageReader, Rgba, RgbaImage};
use llm_multimodal::{
    configure_parallelism,
    encoder_inputs::EncoderDtype,
    types::{ImageDetail, ImageFrame, ImageSize, ImageSource, TransparencyPolicy},
    vision::{
        preprocessor_config::PreProcessorConfig, processors::Qwen3VLProcessor, VisionPreProcessor,
    },
    Parallelism,
};

/// An allocation at or above this size is a buffer of the image's own scale (a decoded
/// frame, a resized plane, a wire), not bookkeeping.
const LARGE: usize = 256 * 1024;

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LARGE_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        if layout.size() >= LARGE {
            LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        if new_size >= LARGE {
            LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The counters are process-wide and libtest runs tests on parallel threads: every
/// measured window holds this lock.
static WINDOW: Mutex<()> = Mutex::new(());

fn window() -> MutexGuard<'static, ()> {
    WINDOW
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Count {
    calls: usize,
    bytes: usize,
    large: usize,
}

fn counted<T>(work: impl FnOnce() -> T) -> (T, Count) {
    let calls = CALLS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let large = LARGE_CALLS.load(Ordering::Relaxed);
    let value = work();
    let count = Count {
        calls: CALLS.load(Ordering::Relaxed) - calls,
        bytes: BYTES.load(Ordering::Relaxed) - bytes,
        large: LARGE_CALLS.load(Ordering::Relaxed) - large,
    };
    (value, count)
}

/// The Qwen3-VL family's image preprocessing settings (its checkpoint's
/// `preprocessor_config.json`): 16-pixel patches merged two by two, 65,536 to 16,777,216
/// pixels, bicubic.
const CONFIG: &str = r#"{"do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
 "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16,
 "resample": 3, "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 16777216, "shortest_edge": 65536},
 "temporal_patch_size": 2}"#;

/// The encoded bytes of one class.
struct Class {
    name: &'static str,
    bytes: Vec<u8>,
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/images")
            .join(name),
    )
    .expect("fixture")
}

/// A 640x480 RGBA PNG with a soft alpha ramp, made here: the fixtures carry no RGBA image.
fn rgba_png() -> Vec<u8> {
    let image = RgbaImage::from_fn(640, 480, |x, y| {
        Rgba([
            (x % 256) as u8,
            (y % 256) as u8,
            ((x + y) % 256) as u8,
            (x / 3 % 256) as u8,
        ])
    });
    let mut out = Vec::new();
    DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("png");
    out
}

fn classes() -> Vec<Class> {
    vec![
        Class {
            name: "jpeg-large",
            bytes: fixture("large.jpg"),
        },
        Class {
            name: "jpeg-small",
            bytes: fixture("small.jpg"),
        },
        Class {
            name: "jpeg-grayscale",
            bytes: fixture("grayscale.jpg"),
        },
        Class {
            name: "jpeg-cmyk",
            bytes: fixture("cmyk_adobe_16x16.jpg"),
        },
        Class {
            name: "png-rgb",
            bytes: fixture("deepseek_v41_carrots.png"),
        },
        Class {
            name: "png-rgba",
            bytes: rgba_png(),
        },
    ]
}

/// The frame as the gateway builds it from fetched bytes: the header read for the size,
/// the pixels still encoded.
fn frame(bytes: &[u8]) -> Arc<ImageFrame> {
    let (width, height) = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .expect("format")
        .into_dimensions()
        .expect("dimensions");
    Arc::new(
        ImageFrame::encoded(
            Bytes::copy_from_slice(bytes),
            ImageDetail::Auto,
            ImageSource::InlineBytes,
            String::new(),
            ImageSize::new(width, height),
            None,
        )
        .with_transparency(TransparencyPolicy::CompositeOverWhite),
    )
}

/// What one class allocates at every stage.
struct Measured {
    decode: Count,
    copy: Count,
    preprocess: Count,
    serialize: Count,
    decoded_bytes: usize,
    wire_bytes: usize,
}

fn measure(processor: &dyn VisionPreProcessor, class: &Class) -> Measured {
    // Warm: the first pass pays for the decoder binding, the thread-local scratch and the
    // processor's one-time tables; the pins hold the steady state.
    {
        let warm = frame(&class.bytes);
        ImageFrame::decode_all(std::slice::from_ref(&warm)).expect("decode");
        let image = warm.image().expect("image").clone();
        let inputs = processor
            .preprocess_wire(std::slice::from_ref(&image), EncoderDtype::BFloat16)
            .expect("one-pass")
            .expect("preprocess");
        drop(inputs.encoder_input.into_bytes_as(EncoderDtype::BFloat16));
    }
    let frame = frame(&class.bytes);
    let ((), decode) =
        counted(|| ImageFrame::decode_all(std::slice::from_ref(&frame)).expect("decode"));
    let (image, copy) = counted(|| frame.image().expect("image").clone());
    let decoded_bytes = image.as_bytes().len();
    let (inputs, preprocess) = counted(|| {
        processor
            .preprocess_wire(std::slice::from_ref(&image), EncoderDtype::BFloat16)
            .expect("one-pass")
            .expect("preprocess")
    });
    let (wire, serialize) = counted(|| {
        inputs
            .encoder_input
            .into_bytes_as(EncoderDtype::BFloat16)
            .expect("wire")
    });
    Measured {
        decode,
        copy,
        preprocess,
        serialize,
        decoded_bytes,
        wire_bytes: wire.len(),
    }
}

/// The pins: per class, the decode's large buffers, the copy's calls and large buffers,
/// the preprocess's calls and large buffers, the move's calls. Today's values, with the
/// copy that each large buffer is.
///
/// - decode: the large buffers of the decode (the decoded frame and the decoder's own
///   planes; a small input decodes under the threshold). The calls are not pinned: they
///   differ between the TurboJPEG binding and the fallback decoder, the large buffers do
///   not.
/// - copy: one call, one large buffer: the gateway's per-frame clone before the processor
///   (the first copy to go; its pin falls to zero calls when it does).
/// - preprocess: the calls of the resize, the layout and the wire write of one image, one
///   count per resize row path (`Calls`), and the large buffers among them (the same
///   under either decoder and on either path: both hand the processor RGB8 pixels).
/// - serialize: zero: the wire bytes move.
struct Pin {
    class: &'static str,
    decode_large: usize,
    copy: (usize, usize),
    preprocess_calls: Calls,
    preprocess_large: usize,
    serialize_calls: usize,
}

/// A call count per resize row path. The resize's rows run through NEON on aarch64
/// (`vision/torch_resample.rs`): its width pass collects wide taps and a scratch row of
/// its own, and its height pass skips the scalar path's sums buffer, so a class counts
/// one or two calls more or fewer than on the scalar path. Both counts are recorded on
/// their own architecture; CI runs this test on each.
struct Calls {
    scalar: usize,
    neon: usize,
}

impl Calls {
    fn here(&self) -> usize {
        if cfg!(target_arch = "aarch64") {
            self.neon
        } else {
            self.scalar
        }
    }
}

const PINS: &[Pin] = &[
    Pin {
        class: "jpeg-large",
        decode_large: 2,
        copy: (1, 1),
        preprocess_calls: Calls {
            scalar: 6039,
            neon: 6038,
        },
        preprocess_large: 2,
        serialize_calls: 0,
    },
    Pin {
        class: "jpeg-small",
        decode_large: 0,
        copy: (1, 0),
        preprocess_calls: Calls {
            scalar: 1052,
            neon: 1053,
        },
        preprocess_large: 1,
        serialize_calls: 0,
    },
    Pin {
        class: "jpeg-grayscale",
        decode_large: 1,
        copy: (1, 1),
        preprocess_calls: Calls {
            scalar: 1372,
            neon: 1373,
        },
        preprocess_large: 3,
        serialize_calls: 0,
    },
    Pin {
        class: "jpeg-cmyk",
        decode_large: 0,
        copy: (1, 0),
        preprocess_calls: Calls {
            scalar: 1052,
            neon: 1053,
        },
        preprocess_large: 1,
        serialize_calls: 0,
    },
    Pin {
        class: "png-rgb",
        decode_large: 2,
        copy: (1, 1),
        preprocess_calls: Calls {
            scalar: 1431,
            neon: 1430,
        },
        preprocess_large: 2,
        serialize_calls: 0,
    },
    Pin {
        class: "png-rgba",
        decode_large: 3,
        copy: (1, 1),
        preprocess_calls: Calls {
            scalar: 17,
            neon: 17,
        },
        preprocess_large: 1,
        serialize_calls: 0,
    },
];

#[test]
fn every_stage_allocates_what_it_allocates_today() {
    let _window = window();
    // Every task on this thread: no pool thread allocates into a measured window.
    assert_eq!(
        configure_parallelism(Parallelism::Inline),
        Ok(()),
        "the pins count the calling thread alone: this test must own the process's mode"
    );
    let config = PreProcessorConfig::from_json(CONFIG).expect("config");
    let processor = Qwen3VLProcessor::from_preprocessor_config(&config);
    let record = std::env::var_os("MM_PINS_RECORD").is_some();
    let mut failures = Vec::new();
    println!("class | stage | calls | bytes | large");
    for class in classes() {
        let measured = measure(&processor, &class);
        for (stage, count) in [
            ("decode", measured.decode),
            ("copy", measured.copy),
            ("preprocess", measured.preprocess),
            ("serialize", measured.serialize),
        ] {
            println!(
                "{} | {stage} | {} | {} | {}",
                class.name, count.calls, count.bytes, count.large
            );
        }
        println!(
            "{} | decoded {} B, wire {} B",
            class.name, measured.decoded_bytes, measured.wire_bytes
        );
        if record {
            continue;
        }
        let pin = PINS
            .iter()
            .find(|pin| pin.class == class.name)
            .expect("a pin per class");
        let checks = [
            (
                "decode large buffers",
                measured.decode.large,
                pin.decode_large,
            ),
            ("copy calls", measured.copy.calls, pin.copy.0),
            ("copy large buffers", measured.copy.large, pin.copy.1),
            (
                "preprocess calls",
                measured.preprocess.calls,
                pin.preprocess_calls.here(),
            ),
            (
                "preprocess large buffers",
                measured.preprocess.large,
                pin.preprocess_large,
            ),
            (
                "serialize calls",
                measured.serialize.calls,
                pin.serialize_calls,
            ),
        ];
        for (what, got, want) in checks {
            if got != want {
                failures.push(format!("{}: {what} = {got}, pinned at {want}", class.name));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "allocation pins moved (a copy came back, or one went: re-pin on purpose with MM_PINS_RECORD=1):\n{}",
        failures.join("\n")
    );
}

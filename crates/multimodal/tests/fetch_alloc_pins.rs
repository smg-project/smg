//! Allocation pins of the media connector's body collection: what a fetched body allocates on
//! its way from the socket to the frame, by declared length and by chunked transfer, for bodies
//! of a few hundred kilobytes to a few megabytes. A pin holds the count where it stands, so a
//! growing buffer that comes back fails the test. A separate executable, so it can declare the
//! counting allocator; the fetch runs on a current-thread runtime, so every allocation of the
//! transfer is the test thread's own.
//!
//! `cargo test -p llm-multimodal --test fetch_alloc_pins -- --nocapture` prints the table.
#![expect(unsafe_code, reason = "a global allocator that counts allocations")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

mod common;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    io::{Cursor, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use image::{DynamicImage, Rgb, RgbImage};
use llm_multimodal::{ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource};
use rustix::time::{clock_gettime, ClockId};

/// An allocation at or above this size is a body buffer, not bookkeeping: the HTTP client's
/// own read buffers stay under it.
const LARGE: usize = 1024 * 1024;

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);
static LARGE_CALLS: AtomicUsize = AtomicUsize::new(0);
static LARGE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn note(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    if size >= LARGE {
        LARGE_CALLS.fetch_add(1, Ordering::Relaxed);
        LARGE_BYTES.fetch_add(size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: the layout is the caller's, handed to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        // SAFETY: `ptr` came from `alloc` or `realloc` above with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Clone, Copy, Debug, Default)]
struct Count {
    calls: usize,
    large_calls: usize,
    large_bytes: usize,
}

fn counted<T>(work: impl FnOnce() -> T) -> (T, Count) {
    let calls = CALLS.load(Ordering::Relaxed);
    let large_calls = LARGE_CALLS.load(Ordering::Relaxed);
    let large_bytes = LARGE_BYTES.load(Ordering::Relaxed);
    let value = work();
    let count = Count {
        calls: CALLS.load(Ordering::Relaxed) - calls,
        large_calls: LARGE_CALLS.load(Ordering::Relaxed) - large_calls,
        large_bytes: LARGE_BYTES.load(Ordering::Relaxed) - large_bytes,
    };
    (value, count)
}

/// A PNG of about `bytes` bytes: noise compresses to nothing, so the file is about three bytes
/// a pixel.
fn png_of_about(bytes: usize) -> Vec<u8> {
    let side = ((bytes / 3) as f64).sqrt() as u32;
    let mut state: u32 = 0x9E37_79B9;
    let image = RgbImage::from_fn(side, side, |_, _| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let [a, b, c, _] = state.to_le_bytes();
        Rgb([a, b, c])
    });
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(image)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("png");
    out
}

/// How an origin sends the body: with its length declared, or chunked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transfer {
    Declared,
    Chunked,
}

/// An origin on the loopback interface serving `body` as `transfer`, written `chunk` bytes at
/// a time, to every request until the listener is dropped.
fn serve(body: Arc<Vec<u8>>, transfer: Transfer, chunk: usize) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().expect("local addr").port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            common::read_head(&mut stream);
            let head = match transfer {
                Transfer::Declared => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n",
                    body.len()
                ),
                Transfer::Chunked => "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\
                     Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                    .to_string(),
            };
            if stream.write_all(head.as_bytes()).is_err() {
                continue;
            }
            for piece in body.chunks(chunk) {
                let written = match transfer {
                    Transfer::Declared => stream.write_all(piece),
                    Transfer::Chunked => stream
                        .write_all(format!("{:x}\r\n", piece.len()).as_bytes())
                        .and_then(|()| stream.write_all(piece))
                        .and_then(|()| stream.write_all(b"\r\n")),
                };
                if written.is_err() {
                    break;
                }
                let _ = stream.flush();
            }
            if transfer == Transfer::Chunked {
                let _ = stream.write_all(b"0\r\n\r\n");
            }
            let _ = stream.flush();
        }
    });
    port
}

fn connector() -> MediaConnector {
    MediaConnector::new(
        common::http_client_builder().no_proxy(),
        MediaConnectorConfig {
            allow_private_hosts: true,
            fetch_timeout: Duration::from_secs(30),
            ..MediaConnectorConfig::default()
        },
    )
    .expect("media connector")
}

/// The process's CPU time so far.
fn cpu_time() -> Duration {
    let now = clock_gettime(ClockId::ProcessCPUTime);
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// One fetch of `url`: the body collected and the frame built (the header read, the hash; the
/// pixels wait), as the connector does for an image by URL. Returns the fetched bytes, the
/// allocations, the wall time and the process CPU time of the fetch.
fn fetch(
    runtime: &tokio::runtime::Runtime,
    connector: &MediaConnector,
    url: &str,
) -> (bytes::Bytes, Count, Duration, Duration) {
    let started = Instant::now();
    let cpu = cpu_time();
    let (frame, count) = counted(|| {
        runtime
            .block_on(connector.fetch_image(
                MediaSource::Url(url.to_string()),
                ImageFetchConfig::default(),
            ))
            .expect("fetch")
    });
    let cpu = cpu_time() - cpu;
    (frame.raw_bytes.clone(), count, started.elapsed(), cpu)
}

/// The pin per body size and transfer: the large buffers of the collection and their bytes.
struct Pin {
    bytes: usize,
    transfer: Transfer,
    /// Allocations of a megabyte or more: the body's one buffer when the body is that large.
    large_calls: usize,
    /// Their bytes: the body's length, once.
    large_bytes_are_the_body: bool,
}

const CHUNK: usize = 16 * 1024;

const PINS: &[Pin] = &[
    Pin {
        bytes: 256 * 1024,
        transfer: Transfer::Declared,
        large_calls: 0,
        large_bytes_are_the_body: false,
    },
    Pin {
        bytes: 256 * 1024,
        transfer: Transfer::Chunked,
        large_calls: 0,
        large_bytes_are_the_body: false,
    },
    Pin {
        bytes: 1024 * 1024,
        transfer: Transfer::Declared,
        large_calls: 1,
        large_bytes_are_the_body: true,
    },
    Pin {
        bytes: 1024 * 1024,
        transfer: Transfer::Chunked,
        large_calls: 1,
        large_bytes_are_the_body: true,
    },
    Pin {
        bytes: 4 * 1024 * 1024,
        transfer: Transfer::Declared,
        large_calls: 1,
        large_bytes_are_the_body: true,
    },
    Pin {
        bytes: 4 * 1024 * 1024,
        transfer: Transfer::Chunked,
        large_calls: 1,
        large_bytes_are_the_body: true,
    },
];

#[test]
fn a_fetched_body_is_collected_into_one_buffer_whether_or_not_its_length_is_declared() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let connector = connector();
    let mut failures = Vec::new();
    println!(
        "body | transfer | calls (median of 5) | large calls | large bytes | ms wall (median of 5) | ms cpu (median of 5)"
    );
    for pin in PINS {
        let body = Arc::new(png_of_about(pin.bytes));
        let port = serve(Arc::clone(&body), pin.transfer, CHUNK);
        let url = format!("http://127.0.0.1:{port}/image.png");
        // Warm: the connection pool, the client's buffers, the hasher's tables.
        let (fetched, _, _, _) = fetch(&runtime, &connector, &url);
        assert_eq!(
            fetched.as_ref(),
            body.as_slice(),
            "the fetched bytes are the body"
        );
        let mut runs: Vec<(Count, Duration, Duration)> = (0..5)
            .map(|_| {
                let (fetched, count, elapsed, cpu) = fetch(&runtime, &connector, &url);
                assert_eq!(
                    fetched.as_ref(),
                    body.as_slice(),
                    "the fetched bytes are the body"
                );
                (count, elapsed, cpu)
            })
            .collect();
        runs.sort_by_key(|(count, _, _)| count.calls);
        let (count, _, _) = runs[2];
        runs.sort_by_key(|(_, elapsed, _)| *elapsed);
        let elapsed = runs[2].1;
        runs.sort_by_key(|(_, _, cpu)| *cpu);
        let cpu = runs[2].2;
        println!(
            "{} B | {:?} | {} | {} | {} | {:.2} | {:.2}",
            body.len(),
            pin.transfer,
            count.calls,
            count.large_calls,
            count.large_bytes,
            elapsed.as_secs_f64() * 1e3,
            cpu.as_secs_f64() * 1e3
        );
        // The large buffers are the same on every run: the body's one buffer.
        for (run, _, _) in &runs {
            if run.large_calls != pin.large_calls {
                failures.push(format!(
                    "{} B {:?}: {} large allocations, pinned at {}",
                    body.len(),
                    pin.transfer,
                    run.large_calls,
                    pin.large_calls
                ));
                break;
            }
            let large_bytes = if pin.large_bytes_are_the_body {
                body.len()
            } else {
                0
            };
            if run.large_bytes != large_bytes {
                failures.push(format!(
                    "{} B {:?}: {} bytes in large allocations, pinned at {}",
                    body.len(),
                    pin.transfer,
                    run.large_bytes,
                    large_bytes
                ));
                break;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the body collection's allocations moved (a growing buffer came back?):\n{}",
        failures.join("\n")
    );
}

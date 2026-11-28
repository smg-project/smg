//! The kernel's calls a row asks for itself, on a target that has them: a
//! thread's core, and one thread's cycles and instructions in user space.
//! The bench's root keeps the stand-in for a target without them.

use std::ffi::{c_int, c_long, c_ulong, c_void};

extern "C" {
    fn sched_getaffinity(thread: c_int, size: usize, mask: *mut u64) -> c_int;
    fn sched_setaffinity(thread: c_int, size: usize, mask: *const u64) -> c_int;
    fn syscall(number: c_long, ...) -> c_long;
    fn read(file: c_int, buffer: *mut c_void, count: usize) -> isize;
    fn close(file: c_int) -> c_int;
}

/// A core mask as the kernel reads one: 1,024 cores.
type Mask = [u64; 16];

/// The cores the calling thread may run on, in their order.
pub(super) fn given() -> Vec<usize> {
    let mut mask: Mask = [0; 16];
    // SAFETY: the mask is `size_of_val(&mask)` writable bytes, and thread 0 is the caller.
    let refused = unsafe { sched_getaffinity(0, size_of_val(&mask), mask.as_mut_ptr()) };
    if refused != 0 {
        return Vec::new();
    }
    let cores = mask.iter().enumerate().flat_map(|(word, bits)| {
        (0..64).filter_map(move |bit| ((bits >> bit) & 1 == 1).then_some(word * 64 + bit))
    });
    cores.collect()
}

/// Pins `thread` to `core` alone, the caller for thread 0; whether the
/// kernel took it.
pub(super) fn pin(thread: c_int, core: usize) -> bool {
    let mut mask: Mask = [0; 16];
    let Some(word) = mask.get_mut(core / 64) else {
        return false;
    };
    *word = 1 << (core % 64);
    // SAFETY: the mask is `size_of_val(&mask)` readable bytes; the kernel refuses a thread that ended.
    unsafe { sched_setaffinity(thread, size_of_val(&mask), mask.as_ptr()) == 0 }
}

/// The number of `perf_event_open`, which the C library wraps in no call.
#[cfg(target_arch = "x86_64")]
const PERF_EVENT_OPEN: c_long = 298;
#[cfg(target_arch = "aarch64")]
const PERF_EVENT_OPEN: c_long = 241;

/// The first version of the kernel's `perf_event_attr`: 64 bytes.
#[repr(C)]
#[derive(Default)]
struct Attr {
    kind: u32,
    size: u32,
    config: u64,
    sample: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup: u32,
    breakpoint: u32,
    config1: u64,
}

/// One hardware counter over one thread's own work: user space, no
/// kernel and no hypervisor.
pub(super) struct Counter(c_int);

impl Counter {
    /// A counter of `thread`'s cycles (`event` 0) or instructions (1),
    /// counting from now; none where the kernel or the host has none
    /// to give.
    pub(super) fn open(thread: c_int, event: u64) -> Option<Self> {
        let attr = Attr {
            size: 64,
            config: event,
            // exclude_kernel and exclude_hv.
            flags: (1 << 5) | (1 << 6),
            ..Attr::default()
        };
        let (every_core, no_group): (c_int, c_int) = (-1, -1);
        let no_flags: c_ulong = 0;
        // SAFETY: the kernel reads `size` bytes of the attribute, which outlives the call.
        let file = unsafe {
            syscall(
                PERF_EVENT_OPEN,
                &raw const attr,
                thread,
                every_core,
                no_group,
                no_flags,
            )
        };
        c_int::try_from(file)
            .ok()
            .filter(|file| *file >= 0)
            .map(Self)
    }

    /// The count so far.
    pub(super) fn count(&self) -> Option<u64> {
        let mut count = 0_u64;
        // SAFETY: the buffer is the eight bytes of `count`.
        let read = unsafe { read(self.0, (&raw mut count).cast(), size_of::<u64>()) };
        (read == 8).then_some(count)
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        // SAFETY: the file is this counter's own, closed once.
        unsafe { close(self.0) };
    }
}

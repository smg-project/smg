//! The binary's settings of the shipped allocator, and the guard that
//! refuses a run on another allocator or under other settings. A case
//! reads this file by its path (`tests/binary.rs`) and holds the copy to
//! the binary's line.

use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
    hint::black_box,
    io, ptr,
};

/// The settings, in the symbol the allocator reads them from as it
/// starts: a copy of the binary's line, which a bench cannot import. A
/// case reads both sources by path (`tests/binary.rs`): the copy moves
/// with the binary's, or that case fails.
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

extern "C" {
    /// The allocator's control call, under the prefix its build gives
    /// every name of its own: the one [`MALLOC_CONF`] is exported
    /// under.
    #[link_name = "_rjem_mallctl"]
    fn mallctl(
        name: *const c_char,
        old: *mut c_void,
        old_len: *mut usize,
        new: *mut c_void,
        new_len: usize,
    ) -> c_int;
}

/// What the allocator answers for `name`, where the answer is a `T`:
/// `None` for a name it does not know or an answer of another size.
fn asked<T: Copy + Default>(name: &CStr) -> Option<T> {
    let mut answer = T::default();
    let mut len = size_of::<T>();
    // SAFETY: `name` ends in NUL, and the allocator writes `len` bytes
    // at most into `answer`, a plain value of that size. Nothing is
    // written to the allocator.
    let refused = unsafe {
        mallctl(
            name.as_ptr(),
            (&raw mut answer).cast(),
            &raw mut len,
            ptr::null_mut(),
            0,
        )
    };
    (refused == 0 && len == size_of::<T>()).then_some(answer)
}

/// Refuses a run that is not on the binary's allocator under the
/// binary's settings: its rows would pass for the binary's. The
/// allocator answers for itself. Each entry of [`MALLOC_CONF`] must
/// read back as the option it sets, whatever the environment adds to
/// the line. And the global allocator must be this one: asked for as
/// much as its oversize threshold, it stands up the arena it keeps for
/// such requests, the one past the `narenas` it spreads threads over,
/// which a request to another allocator does not. That request is one
/// mapping, made and returned before a row starts.
pub(super) fn stands() -> io::Result<()> {
    let refused = |why: String| {
        io::Error::other(format!(
            "this run is not on the binary's allocator and settings: {why}"
        ))
    };
    let line = MALLOC_CONF
        .strip_suffix(b"\0")
        .and_then(|line| str::from_utf8(line).ok())
        .ok_or_else(|| refused("the settings line is no text that ends in NUL".into()))?;
    for entry in line.split(',') {
        let (key, value) = entry.split_once(':').unwrap_or((entry, ""));
        let read = match key {
            "background_thread" => asked::<bool>(c"opt.background_thread").map(|on| on.to_string()),
            "dirty_decay_ms" => asked::<isize>(c"opt.dirty_decay_ms").map(|ms| ms.to_string()),
            "muzzy_decay_ms" => asked::<isize>(c"opt.muzzy_decay_ms").map(|ms| ms.to_string()),
            "narenas" => asked::<c_uint>(c"opt.narenas").map(|arenas| arenas.to_string()),
            "oversize_threshold" => {
                asked::<usize>(c"opt.oversize_threshold").map(|bytes| bytes.to_string())
            }
            // A setting the guard cannot ask for is one it cannot hold.
            _ => None,
        };
        if read.as_deref() != Some(value) {
            let read = read.as_deref().unwrap_or("nothing");
            return Err(refused(format!(
                "`{key}` reads back as {read}, the line says {value}"
            )));
        }
    }
    let asked_by = asked::<usize>(c"opt.oversize_threshold").zip(asked::<c_uint>(c"opt.narenas"));
    let Some((threshold, arenas)) = asked_by else {
        return Err(refused("it names no oversize threshold".into()));
    };
    drop(black_box(Vec::<u8>::with_capacity(threshold)));
    // The allocator's answers on its arenas are of its last epoch.
    let mut epoch = 1_u64;
    // SAFETY: the name ends in NUL, nothing is read back, and the
    // allocator reads the eight bytes of `epoch`.
    let stale = unsafe {
        mallctl(
            c"epoch".as_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            (&raw mut epoch).cast(),
            size_of::<u64>(),
        )
    };
    let arena = CString::new(format!("arena.{arenas}.initialized"))
        .map_err(|_| refused("an arena's name holds a NUL".into()))?;
    if stale != 0 || asked::<bool>(&arena) != Some(true) {
        return Err(refused(format!(
            "a request of {threshold} bytes did not reach it"
        )));
    }
    Ok(())
}

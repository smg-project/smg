//! The C arguments and results of the entry points: a JSON argument read into
//! its type and a result written back as a C string, each failure the entry
//! point's error code and message.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
};

use serde::de::DeserializeOwned;

use super::error::{clear_error_message, set_error_message, SglErrorCode};

/// Read the JSON a C string argument carries into `T`. `name` is the
/// argument's name as the entry point documents it (`request_json`,
/// `messages_json`): a null pointer or bytes that are not UTF-8 are
/// `InvalidArgument`, JSON that is not a `T` is `ParsingError`, each with the
/// message the entry points gave before.
///
/// # Safety
/// - `ptr` must be null or a valid null-terminated C string
/// - `error_out` may be null; if non-null, must point to writable memory
pub(crate) unsafe fn read_json_arg<T: DeserializeOwned>(
    ptr: *const c_char,
    name: &str,
    error_out: *mut *mut c_char,
) -> Result<T, SglErrorCode> {
    if ptr.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return Err(SglErrorCode::InvalidArgument);
    }
    let Ok(text) = CStr::from_ptr(ptr).to_str() else {
        set_error_message(error_out, &format!("Invalid UTF-8 in {name}"));
        return Err(SglErrorCode::InvalidArgument);
    };
    serde_json::from_str(text).map_err(|e| {
        let noun = name.strip_suffix("_json").unwrap_or(name);
        set_error_message(error_out, &format!("Failed to parse {noun} JSON: {e}"));
        SglErrorCode::ParsingError
    })
}

/// Write `text` into the caller's `out` string, which the caller frees with
/// `sgl_free_string`, and clear `error_out`. A text with an interior NUL is
/// `MemoryError`.
///
/// # Safety
/// - `out` must point to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
pub(crate) unsafe fn write_string_result(
    text: String,
    out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> Result<(), SglErrorCode> {
    let result = CString::new(text).map_err(|e| {
        set_error_message(error_out, &format!("Failed to create result string: {e}"));
        SglErrorCode::MemoryError
    })?;
    *out = result.into_raw();
    clear_error_message(error_out);
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/args/tests.rs"]
mod tests;

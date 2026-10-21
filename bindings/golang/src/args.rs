//! The C arguments and results of the entry points: a JSON argument read into
//! its type and a result written back as a C string, each failure the entry
//! point's error code and message.

use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
};

use serde::{de::DeserializeOwned, Serialize};

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

/// Write `value` as JSON into the caller's `out` string, which the caller
/// frees with `sgl_free_string`, and clear `error_out`. A value that does not
/// serialize is `ParsingError`; one with an interior NUL is `MemoryError`.
///
/// # Safety
/// - `out` must point to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
pub(crate) unsafe fn write_json_result<T: Serialize>(
    value: &T,
    out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> Result<(), SglErrorCode> {
    let text = serde_json::to_string(value).map_err(|e| {
        set_error_message(error_out, &format!("Failed to serialize JSON: {e}"));
        SglErrorCode::ParsingError
    })?;
    write_string_result(text, out, error_out)
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
mod tests {
    use std::ptr;

    use serde_json::{json, Value};

    use super::*;
    use crate::memory::sgl_free_string;

    /// The message an entry point left, taken and freed.
    unsafe fn take_message(error: *mut c_char) -> Option<String> {
        if error.is_null() {
            return None;
        }
        let message = CStr::from_ptr(error).to_string_lossy().into_owned();
        sgl_free_string(error);
        Some(message)
    }

    /// What a read gives back: the value, or the code and the start of the
    /// message.
    type Read = Result<Value, (SglErrorCode, &'static str)>;

    #[test]
    fn a_json_argument_is_read_or_refused_with_the_entry_points_message() {
        let valid = CString::new(r#"{"a": 1}"#).unwrap();
        let invalid_utf8 = CString::new([0xFF_u8, 0xFE].to_vec()).unwrap();
        let not_json = CString::new("nope").unwrap();
        let cases: [(*const c_char, Read); 4] = [
            (valid.as_ptr(), Ok(json!({"a": 1}))),
            (
                ptr::null(),
                Err((
                    SglErrorCode::InvalidArgument,
                    "Invalid arguments: null pointer",
                )),
            ),
            (
                invalid_utf8.as_ptr(),
                Err((
                    SglErrorCode::InvalidArgument,
                    "Invalid UTF-8 in request_json",
                )),
            ),
            (
                not_json.as_ptr(),
                Err((SglErrorCode::ParsingError, "Failed to parse request JSON: ")),
            ),
        ];
        for (ptr, expected) in cases {
            let mut error: *mut c_char = ptr::null_mut();
            // SAFETY: the pointers are live locals or null; the message is
            // taken once.
            let (read, message) = unsafe {
                let read = read_json_arg::<Value>(ptr, "request_json", &mut error);
                (read, take_message(error))
            };
            match expected {
                Ok(value) => {
                    assert_eq!(read.unwrap(), value);
                    assert_eq!(message, None);
                }
                Err((code, prefix)) => {
                    assert_eq!(read.unwrap_err(), code);
                    assert!(
                        message.as_deref().is_some_and(|m| m.starts_with(prefix)),
                        "{message:?}"
                    );
                }
            }
        }
    }

    /// The noun of the parse message is the argument's name without its
    /// `_json` suffix, as the entry points worded it (`messages JSON`,
    /// `tools JSON`).
    #[test]
    fn the_parse_message_names_the_argument() {
        let not_json = CString::new("[").unwrap();
        for (name, prefix) in [
            ("messages_json", "Failed to parse messages JSON: "),
            ("tools_json", "Failed to parse tools JSON: "),
            ("text", "Failed to parse text JSON: "),
        ] {
            let mut error: *mut c_char = ptr::null_mut();
            // SAFETY: as above.
            let message = unsafe {
                let _ = read_json_arg::<Value>(not_json.as_ptr(), name, &mut error);
                take_message(error)
            };
            assert!(
                message.as_deref().is_some_and(|m| m.starts_with(prefix)),
                "{name}: {message:?}"
            );
        }
    }

    #[test]
    fn a_result_is_written_as_json_and_a_nul_is_refused() {
        let mut out: *mut c_char = ptr::null_mut();
        let mut error: *mut c_char = CString::new("stale").unwrap().into_raw();
        // SAFETY: the out pointers are live locals; the strings are taken once.
        unsafe {
            write_json_result(&json!({"normal_text": "hi"}), &mut out, &mut error).unwrap();
            assert_eq!(
                CStr::from_ptr(out).to_str().unwrap(),
                r#"{"normal_text":"hi"}"#
            );
            sgl_free_string(out);
            assert!(error.is_null(), "a result clears the error");

            let mut out: *mut c_char = ptr::null_mut();
            let code = write_string_result("a\0b".to_string(), &mut out, &mut error).unwrap_err();
            assert_eq!(code, SglErrorCode::MemoryError);
            assert!(out.is_null());
            let message = take_message(error).unwrap();
            assert!(
                message.starts_with("Failed to create result string: "),
                "{message}"
            );
        }
    }
}

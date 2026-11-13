//! Read-only kernel synchronization health for absolute managed admission deadlines.

use tonic::Status;

/// Fail closed when the OS cannot bound wall-clock error; this never adjusts the clock.
#[cfg(target_os = "linux")]
#[expect(
    unsafe_code,
    reason = "read-only adjtimex query with modes=0 and owned initialized storage"
)]
pub fn check(max_error_ms: u64) -> Result<(), Status> {
    // SAFETY: all-zero timex has modes=0 (query only). The syscall receives writable
    // storage of the platform libc ABI size; no pointer or storage escapes the call.
    let (state, sample) = unsafe {
        let mut sample: libc::timex = std::mem::zeroed();
        let state = libc::adjtimex(&raw mut sample);
        (state, sample)
    };
    if max_error_ms == 0
        || state < 0
        || state == libc::TIME_ERROR
        || sample.status & libc::STA_UNSYNC != 0
        || sample.maxerror < 0
        || sample.maxerror as u64 > max_error_ms.saturating_mul(1000)
    {
        return Err(Status::unavailable(
            "managed clock synchronization outside configured tolerance",
        ));
    }
    Ok(())
}

/// Managed absolute deadlines require an implemented synchronization health source.
#[cfg(not(target_os = "linux"))]
pub fn check(_max_error_ms: u64) -> Result<(), Status> {
    Err(Status::unimplemented(
        "managed clock health currently requires Linux",
    ))
}

"""Read-only Linux/glibc clock health; unknown synchronization refuses new work."""

from __future__ import annotations

import ctypes
import sys


class Timeval(ctypes.Structure):
    """The platform libc timeval used by ntp_gettime."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class NtpTimeval(ctypes.Structure):
    """glibc ntptimeval ABI, including its reserved storage."""

    _fields_ = [
        ("time", Timeval),
        ("maxerror", ctypes.c_long),
        ("esterror", ctypes.c_long),
        ("tai", ctypes.c_long),
        ("reserved", ctypes.c_long * 4),
    ]


def healthy(max_error_ms):
    """Do not confuse a wall timestamp with evidence that the node clock is synchronized."""
    if sys.platform != "linux" or ctypes.sizeof(ctypes.c_long) != 8 or max_error_ms <= 0:
        return False
    libc = ctypes.CDLL(None)
    query = getattr(libc, "ntp_gettime", None)
    if query is None:
        return False
    query.argtypes = [ctypes.POINTER(NtpTimeval)]
    query.restype = ctypes.c_int
    sample = NtpTimeval()
    state = query(ctypes.byref(sample))
    return 0 <= state < 5 and 0 <= sample.maxerror <= max_error_ms * 1000

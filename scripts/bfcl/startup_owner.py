#!/usr/bin/env python3
"""Own direct BFCL startup children until the launcher commits or aborts.

This private pipe protocol is not shared-host admission or a runtime guardian.
Session-escaping descendants and physical GPU/IPC release require the external
executor. Never resolve ownership from a process name or persisted pidfile.
"""

import contextlib
import os
import signal
import subprocess
import sys
import time


def field(stream, eof=False):
    value = bytearray()
    while len(value) <= 65536:
        byte = stream.read(1)
        if not byte:
            if eof and not value:
                return None
            raise ValueError("incomplete control record")
        if byte == b"\0":
            return os.fsdecode(bytes(value))
        value.extend(byte)
    raise ValueError("oversize control field")


class StartupOwner:
    def __init__(self):
        self.children = {}
        self.next_handle = 1

    def start(self, log, argv):
        if len(self.children) >= 16:
            raise ValueError("too many startup children")
        output = sys.stderr if log == "-" else open(log, "wb")
        try:
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            if log != "-":
                output.close()
        handle = str(self.next_handle)
        self.next_handle += 1
        self.children[handle] = child
        # No poll/wait before completing all signals:
        # even an already-exited child stays unreaped and reserves its PID/PGID.
        return handle, child.pid

    def stop(self, handles):
        children = [self.children[handle] for handle in handles]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for child in children:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, sig)
            if sig == signal.SIGTERM and children:
                time.sleep(0.1)
        # All intended group signaling precedes reaping any original leader.
        deadline = time.monotonic() + 1
        for handle in handles:
            child = self.children[handle]
            child.wait(timeout=max(0, deadline - time.monotonic()))
            del self.children[handle]

    def release(self):
        self.children.clear()


def main():
    # Only this private owner creates children; they must remain waitable.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.set_inheritable(sys.stdin.fileno(), False)
    os.set_inheritable(sys.stdout.fileno(), False)
    owner = StartupOwner()
    committed = False
    try:
        while True:
            command = field(sys.stdin.buffer, eof=True)
            if command == "start":
                log = field(sys.stdin.buffer)
                count = int(field(sys.stdin.buffer))
                if not 0 < count <= 1024:
                    raise ValueError("invalid argument count")
                argv = [field(sys.stdin.buffer) for _ in range(count)]
                handle, pid = owner.start(log, argv)
                print(handle, pid, flush=True)
            elif command == "stop":
                owner.stop([field(sys.stdin.buffer)])
                print("ok", flush=True)
            elif command == "commit":
                print("ok", flush=True)
                committed = True
                return 0
            elif command == "abort":
                owner.stop(list(owner.children))
                print("ok", flush=True)
                return 0
            else:
                raise ValueError("missing or invalid control command")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"[launch_arm] startup owner failed: {type(error).__name__}", file=sys.stderr)
        return 1
    finally:
        try:
            if not committed:
                owner.stop(list(owner.children))
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[launch_arm] startup cleanup failed: {type(error).__name__}", file=sys.stderr)
        finally:
            owner.release()


if __name__ == "__main__":
    raise SystemExit(main())

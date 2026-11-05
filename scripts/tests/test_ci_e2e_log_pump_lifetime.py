"""CPU-only regression checks for the actual GPU workflow's log-pump lifetime.

Ubuntu CI supplies Linux /proc, child subreapers and GNU tail --pid. The mocked
pytest owns every process here; no engine, service, network or GPU is used.
"""

import ctypes
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Closed fixture environments still need the setup interpreter's loader paths.
INTERPRETER_ENV = {
    name: os.environ[name]
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    if name in os.environ
}

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github/workflows/e2e-gpu-job.yml"

COMMAND = r'''
import json, os, subprocess, sys, time
from pathlib import Path
name, args = Path(sys.argv[0]).name, sys.argv[1:]
def reject():
    print("UNKNOWN MOCK CALL: " + repr([name, *args]), file=sys.stderr, flush=True)
    raise SystemExit(97)
if name == "bash":
    if args != ["scripts/ci_killall_sglang.sh", "rocm"]:
        reject()
    raise SystemExit(0)
if name == "mkdir":
    if args != ["-p", "e2e-logs"]:
        reject()
    os.execv("/bin/mkdir", ["/bin/mkdir", *args])
if name == "tee":
    if args not in [["e2e-logs/pytest.stdout"], ["e2e-logs/pytest.stderr"]]:
        reject()
    os.execv("/usr/bin/tee", ["/usr/bin/tee", *args])
if name == "tail":
    if (len(args) != 6 or args[:2] != ["-n", "+1"]
            or not args[2].startswith("--pid=") or not args[2][6:].isdigit()
            or args[3:5] != ["--sleep-interval=0.1", "-f"]
            or args[5] not in ["e2e-logs/pytest.stdout.source", "e2e-logs/pytest.stderr.source"]):
        reject()
    os.execv("/usr/bin/tail", ["/usr/bin/tail", *args])
if name != "pytest":
    reject()
expected = [
    "e2e_test/chat_completions", "--ignore=e2e_test/benchmarks",
    "--ignore=e2e_test/bindings_go", "-k", "original_case",
    "--reruns", "2", "--reruns-delay", "5", "-s", "-vv",
    "--junitxml=e2e-logs/sglang-results.xml",
]
if args != expected:
    reject()
print("EARLY STDOUT", flush=True)
print("EARLY STDERR", file=sys.stderr, flush=True)
child_code = """import sys, time
from pathlib import Path
print("DESCENDANT STDOUT", flush=True)
print("DESCENDANT STDERR", file=sys.stderr, flush=True)
Path("writer-ready").write_text("ready")
until = time.monotonic() + 8
while not Path("release-writer").exists() and time.monotonic() < until:
    time.sleep(.01)
"""
writer = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)
birth = Path(f"/proc/{writer.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
Path("writer.json").write_text(json.dumps({
    "PID": writer.pid, "PGID": os.getpgid(writer.pid), "starttime": birth,
    "pytest_PID": os.getpid(), "owner": "this closed fake pytest",
}))
until = time.monotonic() + 5
while not Path("release-pytest").exists():
    if time.monotonic() >= until:
        raise SystemExit(98)
    time.sleep(.01)
print("FINAL STDOUT", flush=True)
print("FINAL STDERR", file=sys.stderr, flush=True)
Path("pytest-completed").write_text("completed")
raise SystemExit(int(os.environ["MOCK_PYTEST_STATUS"]))
'''


def run_body(text):
    start = text.index("      - name: Run E2E tests\n")
    end = text.find("\n      - name:", start + 1)
    block = text[start : end if end >= 0 else len(text)]
    lines = block.splitlines()
    marker = lines.index("        run: |")
    body = []
    for line in lines[marker + 1 :]:
        if line and not line.startswith("          "):
            break
        body.append(line[10:] if line else "")
    values = {
        "provider_api_keys": "false",
        "gpu_vendor": "amd",
        "test_dirs": "e2e_test/chat_completions",
        "test_filter": "-k original_case",
    }

    def render(match):
        return values[match.group(1)]

    return re.sub(r"\$\{\{\s*inputs\.([a-z_]+)\s*\}\}", render, "\n".join(body) + "\n")


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


@unittest.skipUnless(
    sys.platform == "linux" and Path("/proc/self/stat").is_file(),
    "requires the Linux process-lifetime contract used by Ubuntu CI",
)
class LogPumpLifetimeTests(unittest.TestCase):
    def check_pytest_lifetime(self, status):
        for binary in ["/bin/bash", "/bin/mkdir", "/usr/bin/tail", "/usr/bin/tee"]:
            self.assertTrue(os.access(binary, os.X_OK), binary)
        temp = tempfile.TemporaryDirectory(prefix="smg-ci-log-lifetime-")
        root = Path(temp.name)
        trace = {"pytest_status": status, "fixture": str(root)}
        process = writer = None
        libc = ctypes.CDLL(None, use_errno=True)
        old_subreaper = ctypes.c_int()
        enabled = False
        observed = {}
        try:
            bin_dir = root / "closed-bin"
            bin_dir.mkdir()
            for command in ["bash", "mkdir", "pytest", "tail", "tee"]:
                path = bin_dir / command
                path.write_text("#!" + sys.executable + "\n" + COMMAND)
                path.chmod(0o700)
            script = root / "actual-workflow.bash"
            script.write_text(run_body(WORKFLOW.read_text()))
            self.assertEqual(libc.prctl(37, ctypes.byref(old_subreaper), 0, 0, 0), 0)
            self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
            enabled = True
            env = {
                **INTERPRETER_ENV,
                "PATH": str(bin_dir),
                "FINAL_ENVIRONMENT_CHECK": "1",
                "MOCK_PYTEST_STATUS": str(status),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            process = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            outer_birth = identity(process.pid)[1]
            trace.update({"outer_PID": process.pid, "outer_birth": outer_birth})
            observed = {process.stdout: b"", process.stderr: b""}
            deadline = time.monotonic() + 3
            while True:
                paths = [root / "e2e-logs/pytest.stdout", root / "e2e-logs/pytest.stderr"]
                markers = ["EARLY STDOUT", "EARLY STDERR"]
                retained = all(p.exists() and m in p.read_text() for p, m in zip(paths, markers))
                if (
                    b"EARLY STDOUT" in observed[process.stdout]
                    and b"EARLY STDERR" in observed[process.stderr]
                    and retained
                    and (root / "writer-ready").exists()
                ):
                    break
                self.assertIsNone(process.poll(), "pytest must still be blocked before release")
                self.assertLess(time.monotonic(), deadline, "both streams must be live before exit")
                ready, _, _ = select.select(list(observed), [], [], 0.01)
                for stream in ready:
                    observed[stream] += os.read(stream.fileno(), 4096)
            writer = json.loads((root / "writer.json").read_text())
            self.assertEqual(writer["PID"], writer["PGID"])
            self.assertEqual(identity(writer["PID"])[1], writer["starttime"])
            (root / "release-pytest").write_text("release")
            deadline = time.monotonic() + 1
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            pytest_id = identity(writer["pytest_PID"])
            writer_id = identity(writer["PID"])
            trace.update(
                {
                    "live_before_pytest_exit": True,
                    "pytest_exited": pytest_id is None or pytest_id[0] == "Z",
                    "writer_still_alive": writer_id is not None
                    and writer_id[1] == writer["starttime"]
                    and writer_id[0] != "Z",
                    "shell_finished_before_writer_release": process.poll() is not None,
                    "writer": writer,
                }
            )
            self.assertTrue(trace["pytest_exited"], trace)
            self.assertTrue(trace["writer_still_alive"], trace)
            self.assertIsNotNone(
                process.poll(), "pytest exited; retained descendant FDs still block the shell"
            )
            out, err = process.communicate(timeout=2)
            self.assertNotIn(b"UNKNOWN MOCK CALL", b"".join(observed.values()) + out + err)
            self.assertEqual(process.returncode, status)
            self.assertIn("FINAL STDOUT", (root / "e2e-logs/pytest.stdout").read_text())
            self.assertIn("FINAL STDERR", (root / "e2e-logs/pytest.stderr").read_text())
        finally:
            cleanup_error = None
            reaped = False
            try:
                # No process outside this born-here session/writer can be signalled.
                (root / "release-pytest").write_text("release")
                (root / "release-writer").write_text("release")
                if writer is None and (root / "writer.json").exists():
                    writer = json.loads((root / "writer.json").read_text())
                if process is not None:
                    try:
                        process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        current = identity(process.pid)
                        self.assertIsNotNone(current)
                        self.assertEqual(current[1], trace["outer_birth"])
                        self.assertEqual(os.getpgid(process.pid), process.pid)
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate(timeout=2)
                    trace["outer_reaped"] = process.poll() is not None
                reaped = False
                if writer is not None:
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        try:
                            waited, _ = os.waitpid(writer["PID"], os.WNOHANG)
                        except ChildProcessError:
                            reaped = identity(writer["PID"]) is None
                            break
                        if waited:
                            reaped = True
                            break
                        time.sleep(0.01)
                    if not reaped:
                        current = identity(writer["PID"])
                        self.assertIsNotNone(current)
                        self.assertEqual(current[1], writer["starttime"])
                        self.assertEqual(os.getpgid(writer["PID"]), writer["PGID"])
                        os.killpg(writer["PGID"], signal.SIGKILL)
                        waited, _ = os.waitpid(writer["PID"], 0)
                        reaped = waited == writer["PID"]
                    trace["writer_reaped"] = reaped
                    trace["writer_absent"] = identity(writer["PID"]) is None
            except BaseException as exc:
                cleanup_error = exc
                trace["cleanup_error"] = repr(exc)
            if enabled:
                trace["subreaper_restored"] = libc.prctl(36, old_subreaper.value, 0, 0, 0) == 0
            temp.cleanup()
            trace["fixture_absent"] = not root.exists()
            print("SMG_E2E_LIFETIME " + json.dumps(trace), flush=True)
            if cleanup_error is not None:
                raise cleanup_error
            self.assertTrue(writer is not None and reaped and trace["writer_absent"], trace)
            self.assertTrue(trace.get("outer_reaped") and trace.get("subreaper_restored"), trace)
            self.assertTrue(trace["fixture_absent"], trace)

    def test_success_does_not_wait_for_descendant_stream_handles(self):
        self.check_pytest_lifetime(0)

    def test_failure_does_not_wait_for_descendant_stream_handles(self):
        self.check_pytest_lifetime(7)

"""Sample existing per-rank scheduler load only while original cases execute."""

import hashlib
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request

from reuse import NoRedirect, require

LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:[^"\\]|\\.)*")(?:,|$)')
SAMPLE = re.compile(r"^smg_engine_running_requests\{(.*)\}\s+(\S+)(?:\s+\d+)?$")
LIMIT = 512 * 1024


def running_ranks(text, worker, model):
    positive, lines = set(), []
    for line in text.splitlines():
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        labels, end = {}, 0
        for label in LABEL.finditer(match[1]):
            require(label.start() == end and label[1] not in labels, "invalid metric labels")
            labels[label[1]], end = json.loads(label[2]), label.end()
        require(end == len(match[1]), "incomplete metric labels")
        if labels.get("worker") != worker or labels.get("model") != model:
            continue
        require(labels.get("dp_rank") in {"0", "1"}, "unexpected actual DP rank")
        value = float(match[2])
        require(math.isfinite(value) and value >= 0, "invalid running-request gauge")
        lines.append(line)
        if value > 0:
            positive.add(int(labels["dp_rank"]))
    return positive, lines


class ActivityWindow:
    def __init__(self, url, worker, model, nodeid, output, state):
        parsed = urllib.parse.urlsplit(url)
        require(
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port
            and not parsed.username
            and not parsed.password
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment,
            "metrics must be the owned local listener",
        )
        self.url, self.worker, self.model = url.rstrip("/") + "/metrics", worker, model
        self.nodeid, self.output, self.state = nodeid, output, state
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=1)
        require(not self.thread.is_alive(), "metrics sampler did not stop")

    def run(self):
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
        while not self.stop.is_set() and self.state["ranks"] != {0, 1}:
            try:
                with opener.open(self.url, timeout=0.2) as response:
                    raw = response.read(LIMIT + 1)
                require(len(raw) <= LIMIT, "metrics response cap")
                ranks, lines = running_ranks(raw.decode(), self.worker, self.model)
                new = ranks - self.state["ranks"]
                if new:
                    name = (
                        "rank-load-"
                        + hashlib.sha256((self.nodeid + str(time.time_ns())).encode()).hexdigest()[
                            :20
                        ]
                        + ".prom"
                    )
                    (self.output / name).write_bytes(raw)
                    self.state["records"].append(
                        {
                            "nodeid": self.nodeid,
                            "phase": "call",
                            "observed_at_epoch": time.time(),
                            "new_positive_ranks": sorted(new),
                            "lines": lines,
                            "raw_file": name,
                            "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
                    self.state["ranks"].update(new)
            except Exception as error:
                self.state["error_count"] += 1
                if len(self.state["errors"]) < 8:
                    self.state["errors"].append(str(error))
            self.stop.wait(0.5)

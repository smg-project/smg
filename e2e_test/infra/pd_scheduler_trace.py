"""Primitive trace records for the unconnected, test-only scheduler prototype."""

import json
import os
import uuid
from pathlib import Path


class SchedulerTrace:
    def __init__(self, sink, context):
        self.sink = sink
        self.context = dict(context)
        self.sequence = 0
        self.unknown = None

    @classmethod
    def from_env(cls):
        context = {
            key: os.environ.get(f"SMG_PD_TRACE_{key.upper()}")
            for key in ("run", "attempt", "worker", "role")
        }
        context.update(pid=os.getpid(), instance=uuid.uuid4().hex)
        trace = cls(None, context)
        try:
            if any(
                not isinstance(context[key], str) or not context[key]
                for key in ("run", "attempt", "worker", "role")
            ):
                raise ValueError("missing trace identity")
            if context["role"] not in ("prefill", "decode"):
                raise ValueError("unknown worker role")
            directory = Path(os.environ["SMG_PD_TRACE_DIR"])
            path = directory / f"scheduler-{context['pid']}-{context['instance']}.jsonl"
            path.touch(exist_ok=False)

            def write(line):
                with path.open("a", encoding="utf-8") as stream:
                    if stream.write(line + "\n") != len(line) + 1:
                        raise OSError("short trace write")

            trace.sink = write
        except Exception as exc:
            trace.mark_unknown("setup", exc)
        return trace

    def mark_unknown(self, phase, exc):
        if self.unknown is None:
            self.unknown = f"{phase}:{type(exc).__name__}"

    def emit(self, event, **fields):
        if self.unknown is not None:
            return
        try:
            self.sequence += 1
            record = dict(
                self.context,
                schema=1,
                seq=self.sequence,
                event=event,
                observation="partial",
                **fields,
            )
            line = json.dumps(record, allow_nan=False, separators=(",", ":"))
            if self.sink is None:
                raise RuntimeError("trace sink unavailable")
            self.sink(line)
        except Exception as exc:
            self.mark_unknown(event, exc)


def nonnegative_int(value):
    if type(value) is not int or value < 0:
        raise ValueError("missing or invalid scheduler integer")
    return value

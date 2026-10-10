#!/usr/bin/env python3
"""Generate nightly benchmark summary for GitHub Actions.

Produces a gRPC vs HTTP comparison report with aggregate stats, per-concurrency
breakdown, win/loss scorecard, top wins, and per-model detail tables.

Usage:
    python nightly_summarize.py [base_dir]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# ---------------------------------------------------------------------------
# Metric definitions — single source of truth
#
# Each metric is defined once in _METRIC_REGISTRY. Section-specific lists
# are composed by selecting from the registry, avoiding tuple duplication.
# ---------------------------------------------------------------------------

# (label, field_name, lower_is_better)
_MetricDef = tuple[str, str, bool]

# Canonical definitions — every metric appears exactly once.
_M_TTFT_MEAN: _MetricDef = ("TTFT mean", "ttft_mean", True)
_M_TTFT_P99: _MetricDef = ("TTFT p99", "ttft_p99", True)
_M_TPOT_MEAN: _MetricDef = ("TPOT mean", "tpot_mean", True)
_M_TPOT_P99: _MetricDef = ("TPOT p99", "tpot_p99", True)
_M_E2E_MEAN: _MetricDef = ("E2E mean", "e2e_mean", True)
_M_E2E_P99: _MetricDef = ("E2E p99", "e2e_p99", True)
_M_OUT_TPUT: _MetricDef = ("Output tput", "output_throughput", False)
_M_TOT_TPUT: _MetricDef = ("Total throughput", "total_throughput", False)
_M_RPS: _MetricDef = ("RPS", "rps", False)

# Per-section selections (composed, not copy-pasted).
AGGREGATE_METRICS: list[_MetricDef] = [
    _M_TTFT_MEAN,
    _M_TTFT_P99,
    _M_E2E_MEAN,
    _M_E2E_P99,
    _M_TPOT_MEAN,
    _M_TPOT_P99,
    _M_OUT_TPUT,
    _M_TOT_TPUT,
    _M_RPS,
]
TOP_WINS_METRICS: list[_MetricDef] = [
    _M_TTFT_P99,
    _M_TTFT_MEAN,
    _M_E2E_MEAN,
    _M_E2E_P99,
    _M_TPOT_P99,
    _M_OUT_TPUT,
    _M_RPS,
]
PER_MODEL_METRICS: list[_MetricDef] = [_M_TTFT_P99, _M_TPOT_P99, _M_E2E_P99, _M_OUT_TPUT]
CONCURRENCY_METRICS: list[_MetricDef] = [
    _M_TTFT_MEAN,
    _M_TTFT_P99,
    _M_TPOT_MEAN,
    _M_TPOT_P99,
    _M_E2E_MEAN,
    _M_E2E_P99,
    _M_OUT_TPUT,
]
SCORECARD_METRICS: list[_MetricDef] = [_M_E2E_MEAN, _M_TTFT_MEAN, _M_OUT_TPUT]

# Chart metrics carry a unit for axis labels.
CHART_METRICS: list[tuple[str, str, bool, str]] = [
    (*_M_TTFT_P99, "ms"),
    (*_M_TPOT_P99, "ms"),
    (*_M_E2E_P99, "s"),
    (*_M_OUT_TPUT, "tok/s"),
]

# ---------------------------------------------------------------------------
# Glossary — displayed once in the summary header
# ---------------------------------------------------------------------------

_GLOSSARY_LINES = [
    "#### Metrics",
    "",
    "| Metric | Description |",
    "|--------|-------------|",
    "| **TTFT** | Time To First Token — latency from request sent to first token received |",
    "| **TPOT** | Time Per Output Token — average time between consecutive output tokens |",
    "| **E2E** | End-to-End latency — total time from request sent to last token received |",
    "| **Output tput** | Output throughput — tokens generated per second (tok/s) |",
    "| **Total tput** | Total throughput — input + output tokens processed per second |",
    "| **RPS** | Requests Per Second — completed requests per second |",
    "| **p99** | 99th percentile — the value below which 99% of observations fall (worst 1% of requests) |",
    "| **mean** | Arithmetic average across all requests |",
    "",
    "#### Traffic Scenarios",
    "",
    "| Pattern | Description |",
    "|---------|-------------|",
    "| **D(in, out)** | Deterministic — fixed input/output token lengths, e.g. `D(100,100)` |",
    "| **N(μ,σ)/(μ,σ)** | Normal distribution — input/output lengths drawn from Gaussian |",
    "| **E(size)** | Embedding — input of given token length, used for embedding model benchmarks |",
    "",
    "#### Comparison Columns",
    "",
    "| Value | Meaning |",
    "|-------|---------|",
    "| **gRPC X%** | gRPC is X% better than HTTP for this metric |",
    "| **HTTP X%** | HTTP is X% better than gRPC for this metric |",
    "| **{Runtime} X%** | That runtime is X% better than the other for this metric (e.g. SGLang, vLLM, TRT-LLM) |",
    "| **~** | Difference is within 2% — essentially a tie |",
    "",
    "*Lower is better for latency metrics (TTFT, TPOT, E2E). Higher is better for throughput (tput, RPS).*",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Single benchmark run result."""

    scenario: str
    concurrency: int
    rps: float
    output_throughput: float
    total_throughput: float
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    tpot_p99: float
    e2e_mean: float
    e2e_p99: float
    error_rate: float


@dataclass
class ExperimentInfo:
    """Parsed experiment metadata."""

    model: str  # short name (e.g. Llama-3.1-8B-Instruct)
    protocol: str  # http, grpc
    runtime: str  # sglang, vllm
    worker_type: str  # single, multi
    gpu_type: str
    gpu_count: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def group_key(self) -> str:
        """Key for grouping gRPC vs HTTP pairs."""
        return f"{self.model}|{self.runtime}|{self.worker_type}"

    @property
    def table_key(self) -> str:
        """Key for overview table columns."""
        return f"{self.protocol}_{self.runtime}_{self.worker_type}"


@dataclass
class ComparisonPoint:
    """A matched gRPC vs HTTP data point."""

    model: str
    runtime: str
    worker_type: str
    scenario: str
    concurrency: int
    grpc: RunResult
    http: RunResult

    @property
    def config(self) -> str:
        return f"{self.runtime}/{self.worker_type}"


@dataclass
class RuntimeComparisonPoint:
    """A matched runtime-A vs runtime-B data point (same protocol)."""

    model: str
    protocol: str
    worker_type: str
    scenario: str
    concurrency: int
    runtime_a: str  # alphabetically first, e.g. "sglang"
    runtime_b: str  # e.g. "vllm"
    run_a: RunResult
    run_b: RunResult

    @property
    def config(self) -> str:
        return f"{self.protocol}/{self.worker_type}"


@dataclass
class SummaryResult:
    """Return value of generate_summary."""

    markdown: str
    experiments: list[ExperimentInfo]
    comparisons: list[ComparisonPoint]
    runtime_comparisons: list[RuntimeComparisonPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _get_float(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key)
    return float(val) if val is not None else default


_KNOWN_PROTOCOLS = {"http", "grpc"}
_KNOWN_WORKER_TYPES = {"single", "multi"}
# Runtimes recognized in folder names. Add new runtimes here.
_KNOWN_RUNTIMES = {"sglang", "vllm", "trtllm"}


def parse_folder_name(folder_name: str) -> dict:
    """Parse experiment info from folder name.

    Expected: nightly_{model}_{protocol}_{runtime}_{worker_type}
    """
    info = {
        "model": "unknown",
        "protocol": "unknown",
        "runtime": None,
        "worker_type": "single",
    }
    name = folder_name.replace("nightly_", "")
    parts = name.rsplit("_", 3)

    if len(parts) >= 4 and parts[-1] in _KNOWN_WORKER_TYPES and parts[-2] in _KNOWN_RUNTIMES:
        info["worker_type"] = parts[-1]
        info["runtime"] = parts[-2]
        info["protocol"] = parts[-3]
        info["model"] = "_".join(parts[:-3])
    elif len(parts) >= 3 and parts[-1] in _KNOWN_RUNTIMES:
        info["runtime"] = parts[-1]
        info["protocol"] = parts[-2]
        info["model"] = "_".join(parts[:-2])
    elif len(parts) >= 2 and parts[-1] in _KNOWN_PROTOCOLS:
        info["protocol"] = parts[-1]
        info["model"] = "_".join(parts[:-1])
    else:
        info["model"] = name

    return info


def parse_experiment(folder: Path) -> ExperimentInfo | None:
    """Parse experiment folder into ExperimentInfo."""
    metadata_path = folder / "experiment_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse metadata in {folder}: {e}", file=sys.stderr)
        return None

    folder_info = parse_folder_name(folder.name)

    model_path = meta.get("model", "unknown")
    model = model_path.split("/")[-1] if "/" in model_path else model_path

    runtime = meta.get("server_engine")
    if not runtime or runtime == "unknown":
        runtime = folder_info.get("runtime")
    if not runtime:
        # Fallback: detect runtime from folder name
        folder_lower = folder.name.lower()
        for rt in _KNOWN_RUNTIMES:
            if rt in folder_lower:
                runtime = rt
                break
        else:
            runtime = "unknown"
    runtime = runtime.lower() if runtime else "unknown"

    worker_type = folder_info.get("worker_type", "single")
    gpu_type = meta.get("server_gpu_type") or "unknown"
    try:
        gpu_count = int(meta.get("server_gpu_count") or "1")
    except (ValueError, TypeError):
        gpu_count = 1

    info = ExperimentInfo(
        model=model,
        protocol=folder_info.get("protocol", "unknown"),
        runtime=runtime,
        worker_type=worker_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )

    for json_file in folder.glob("*.json"):
        if "experiment_metadata" in json_file.name or "gpu_utilization" in json_file.name:
            continue

        try:
            with json_file.open() as f:
                data = json.load(f)

            agg = data.get("aggregated_metrics", {})
            stats = agg.get("stats", {})
            ttft = stats.get("ttft", {})
            tpot = stats.get("tpot", {})
            e2e = stats.get("e2e_latency", {})

            run = RunResult(
                scenario=agg.get("scenario", "unknown"),
                concurrency=agg.get("num_concurrency", 0) or 0,
                rps=_get_float(agg, "requests_per_second"),
                output_throughput=_get_float(agg, "mean_output_throughput_tokens_per_s"),
                total_throughput=_get_float(agg, "mean_total_tokens_throughput_tokens_per_s"),
                ttft_mean=_get_float(ttft, "mean"),
                ttft_p99=_get_float(ttft, "p99"),
                tpot_mean=_get_float(tpot, "mean"),
                tpot_p99=_get_float(tpot, "p99"),
                e2e_mean=_get_float(e2e, "mean"),
                e2e_p99=_get_float(e2e, "p99"),
                error_rate=_get_float(agg, "error_rate"),
            )
            info.runs.append(run)
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)

    return info if info.runs else None


def discover_experiments(base_dir: Path) -> list[ExperimentInfo]:
    """Discover and parse all nightly experiment folders."""
    experiments = []
    for folder in base_dir.rglob("nightly_*"):
        if folder.is_dir():
            exp = parse_experiment(folder)
            if exp:
                experiments.append(exp)
    return experiments


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def build_comparisons(experiments: list[ExperimentInfo]) -> list[ComparisonPoint]:
    """Match gRPC and HTTP runs for the same model/runtime/worker/scenario/concurrency."""
    groups: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        groups[exp.group_key][exp.protocol] = exp

    comparisons = []
    for protocols in groups.values():
        if "grpc" not in protocols or "http" not in protocols:
            continue

        grpc_exp = protocols["grpc"]
        http_exp = protocols["http"]

        http_runs = {(r.scenario, r.concurrency): r for r in http_exp.runs}

        for grpc_run in grpc_exp.runs:
            http_run = http_runs.get((grpc_run.scenario, grpc_run.concurrency))
            if http_run and grpc_run.error_rate == 0 and http_run.error_rate == 0:
                comparisons.append(
                    ComparisonPoint(
                        model=grpc_exp.model,
                        runtime=grpc_exp.runtime,
                        worker_type=grpc_exp.worker_type,
                        scenario=grpc_run.scenario,
                        concurrency=grpc_run.concurrency,
                        grpc=grpc_run,
                        http=http_run,
                    )
                )

    return comparisons


def build_runtime_comparisons(experiments: list[ExperimentInfo]) -> list[RuntimeComparisonPoint]:
    """Match runs across runtimes for the same model/protocol/worker/scenario/concurrency."""
    groups: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        key = f"{exp.model}|{exp.protocol}|{exp.worker_type}"
        groups[key][exp.runtime] = exp

    comparisons = []
    for group_key, runtimes in groups.items():
        if len(runtimes) < 2:
            continue

        sorted_rts = sorted(runtimes.keys())
        for i in range(len(sorted_rts)):
            for j in range(i + 1, len(sorted_rts)):
                rt_a, rt_b = sorted_rts[i], sorted_rts[j]
                exp_a, exp_b = runtimes[rt_a], runtimes[rt_b]
                runs_b = {(r.scenario, r.concurrency): r for r in exp_b.runs}

                for run_a in exp_a.runs:
                    run_b = runs_b.get((run_a.scenario, run_a.concurrency))
                    if run_b and run_a.error_rate == 0 and run_b.error_rate == 0:
                        comparisons.append(
                            RuntimeComparisonPoint(
                                model=exp_a.model,
                                protocol=exp_a.protocol,
                                worker_type=exp_a.worker_type,
                                scenario=run_a.scenario,
                                concurrency=run_a.concurrency,
                                runtime_a=rt_a,
                                runtime_b=rt_b,
                                run_a=run_a,
                                run_b=run_b,
                            )
                        )

    return comparisons


# ---------------------------------------------------------------------------
# Formatting helpers
#
# All percentages are shown as "gRPC advantage": positive = gRPC is better.
# For latency metrics (lower=better): advantage = (HTTP - gRPC) / HTTP
# For throughput metrics (higher=better): advantage = (gRPC - HTTP) / HTTP
# ---------------------------------------------------------------------------


def _advantage(grpc_val: float, http_val: float, lower_is_better: bool) -> float | None:
    """gRPC advantage %: positive = gRPC is better, negative = HTTP is better."""
    if http_val == 0:
        return None
    pct = (grpc_val - http_val) / http_val * 100
    return -pct if lower_is_better else pct


def _cp_advantage(cp: ComparisonPoint, fld: str, lower_is_better: bool) -> float | None:
    """Shorthand: compute gRPC advantage for a ComparisonPoint and field name."""
    return _advantage(getattr(cp.grpc, fld), getattr(cp.http, fld), lower_is_better)


def _avg_advantage(cps: list[ComparisonPoint], fld: str, lower_is_better: bool) -> float | None:
    """Average gRPC advantage across a list of comparison points."""
    advs = [a for cp in cps if (a := _cp_advantage(cp, fld, lower_is_better)) is not None]
    return sum(advs) / len(advs) if advs else None


def _fmt_winner(
    pct: float | None,
    threshold: float = 2.0,
    bold: bool = False,
    label_a: str = "gRPC",
    label_b: str = "HTTP",
) -> str:
    """Format as 'A X%' or 'B X%' or '~'. pct > 0 means A is better."""
    if pct is None:
        return "N/A"
    if abs(pct) < threshold:
        return "~"
    name = label_a if pct > 0 else label_b
    if bold:
        name = f"**{name}**"
    return f"{name} {abs(pct):.1f}%"


def _fmt_latency_s(val_s: float) -> str:
    """Format latency value (in seconds) for display."""
    ms = val_s * 1000
    return f"{ms:.0f}ms" if ms < 1000 else f"{val_s:.2f}s"


def _fmt_throughput(val: float) -> str:
    return f"{val / 1000:.1f}K" if val >= 1000 else f"{val:.0f}"


def _fmt_metric_value(val: float, label: str) -> str:
    """Format a metric value based on its label/type."""
    label_lower = label.lower()
    if any(k in label_lower for k in ("ttft", "e2e")):
        return _fmt_latency_s(val)
    if "tpot" in label_lower:
        ms = val * 1000
        return f"{ms:.1f}ms" if ms >= 1 else f"{ms:.2f}ms"
    if "tput" in label_lower or "throughput" in label_lower:
        return f"{_fmt_throughput(val)} tok/s"
    if "rps" in label_lower:
        return f"{val:.1f}"
    return f"{val:.1f}"


def _group_by_concurrency(
    comparisons: list[ComparisonPoint],
) -> tuple[dict[int, list[ComparisonPoint]], list[int]]:
    """Group comparisons by concurrency level, return (mapping, sorted_levels)."""
    by_conc: dict[int, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_conc[cp.concurrency].append(cp)
    return by_conc, sorted(by_conc.keys())


# ---------------------------------------------------------------------------
# Chart generation (optional — requires matplotlib)
# ---------------------------------------------------------------------------


def _plot_comparison_grid(
    title: str,
    by_conc: dict[int, list[ComparisonPoint]],
    conc_levels: list[int],
    output_dir: Path,
    filename: str,
) -> str | None:
    """Plot a 2x2 grid of CHART_METRICS. Returns filename or None on failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, (label, fld, _lower_better, unit) in zip(axes.flat, CHART_METRICS):
        grpc_vals, http_vals = [], []
        for conc in conc_levels:
            cps = by_conc[conc]
            g = sum(getattr(cp.grpc, fld) for cp in cps) / len(cps)
            h = sum(getattr(cp.http, fld) for cp in cps) / len(cps)
            if unit == "ms":
                g *= 1000
                h *= 1000
            grpc_vals.append(g)
            http_vals.append(h)

        x = range(len(conc_levels))
        ax.plot(x, grpc_vals, "o-", label="gRPC", color="#2196F3", linewidth=2)
        ax.plot(x, http_vals, "s--", label="HTTP", color="#FF9800", linewidth=2)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in conc_levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        nz = [v for v in grpc_vals + http_vals if v > 0]
        if nz and max(nz) > 10 * min(nz):
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def generate_charts(
    comparisons: list[ComparisonPoint],
    output_dir: Path,
) -> list[str]:
    """Generate comparison charts as PNGs. Returns list of generated filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []

    by_conc, conc_levels = _group_by_concurrency(comparisons)
    if not conc_levels:
        return []

    # Aggregate comparison
    fname = _plot_comparison_grid(
        "gRPC vs HTTP — Aggregate Comparison",
        by_conc,
        conc_levels,
        output_dir,
        "aggregate_comparison.png",
    )
    if fname:
        charts.append(fname)

    # Per-model/config charts
    by_model_config: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_model_config[f"{cp.model}|{cp.config}"].append(cp)

    for key, cps in sorted(by_model_config.items()):
        model, config = key.split("|", 1)
        mc_conc, mc_levels = _group_by_concurrency(cps)
        if len(mc_levels) < 3:
            continue

        display_name = model.split("/")[-1] if "/" in model else model
        safe = model.replace("/", "__")
        safe_cfg = config.replace("/", "_")
        fname = _plot_comparison_grid(
            f"{display_name} ({config}): gRPC vs HTTP",
            mc_conc,
            mc_levels,
            output_dir,
            f"{safe}_{safe_cfg}_comparison.png",
        )
        if fname:
            charts.append(fname)

    return charts


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

_RUNTIME_DISPLAY = {"sglang": "SGLang", "vllm": "vLLM", "trtllm": "TRT-LLM"}
_PROTOCOL_DISPLAY = {"http": "HTTP", "grpc": "gRPC"}


def _section_key_findings(
    comparisons: list[ComparisonPoint],
    experiments: list[ExperimentInfo],
    runtime_comparisons: list[RuntimeComparisonPoint] | None = None,
) -> list[str]:
    """Auto-generated executive summary of the benchmark results."""
    if not comparisons and not runtime_comparisons:
        return []

    lines = ["### Key Findings", ""]

    # 1. Protocol verdict per key metric
    if comparisons:
        lines.append("**Protocol (gRPC vs HTTP):**")
        lines.append("")
        for label, fld, lower_better, _unit in CHART_METRICS:
            advs = [
                a for cp in comparisons if (a := _cp_advantage(cp, fld, lower_better)) is not None
            ]
            if not advs:
                continue
            avg_adv = sum(advs) / len(advs)
            grpc_wins = sum(1 for a in advs if a > 2)
            http_wins = sum(1 for a in advs if a < -2)
            ties = len(advs) - grpc_wins - http_wins
            if abs(avg_adv) < 1:
                lines.append(
                    f"- **{label}**: No clear winner — essentially tied across all scenarios"
                )
            else:
                winner = "gRPC" if avg_adv > 0 else "HTTP"
                lines.append(
                    f"- **{label}**: {winner} wins {max(grpc_wins, http_wins)}/{len(advs)} "
                    f"comparisons (avg {abs(avg_adv):.1f}% better), "
                    f"{min(grpc_wins, http_wins)} losses, {ties} ties"
                )
        lines.append("")

    # 2. Runtime verdict per key metric (one block per runtime pair)
    if runtime_comparisons:
        by_pair = _group_by_runtime_pair(runtime_comparisons)
        for (rt_a, rt_b), pair_rcps in sorted(by_pair.items()):
            rt_a_display = _RUNTIME_DISPLAY.get(rt_a, rt_a)
            rt_b_display = _RUNTIME_DISPLAY.get(rt_b, rt_b)

            lines.append(f"**Runtime ({rt_a_display} vs {rt_b_display}):**")
            lines.append("")
            for label, fld, lower_better, _unit in CHART_METRICS:
                advs = [
                    a
                    for rcp in pair_rcps
                    if (a := _rt_advantage(rcp, fld, lower_better)) is not None
                ]
                if not advs:
                    continue
                avg_adv = sum(advs) / len(advs)
                a_wins = sum(1 for a in advs if a > 2)
                b_wins = sum(1 for a in advs if a < -2)
                ties = len(advs) - a_wins - b_wins
                if abs(avg_adv) < 1:
                    lines.append(
                        f"- **{label}**: No clear winner — essentially tied across all scenarios"
                    )
                else:
                    winner = rt_a_display if avg_adv > 0 else rt_b_display
                    lines.append(
                        f"- **{label}**: {winner} wins "
                        f"{max(a_wins, b_wins)}/{len(advs)} "
                        f"comparisons (avg {abs(avg_adv):.1f}% better), "
                        f"{min(a_wins, b_wins)} losses, {ties} ties"
                    )
            lines.append("")

    # 3. Error rates
    total_runs = sum(len(e.runs) for e in experiments)
    error_runs = [(e, r) for e in experiments for r in e.runs if r.error_rate > 0]
    if error_runs:
        lines.append(f"- **Errors**: {len(error_runs)}/{total_runs} runs had non-zero error rates")
    else:
        lines.append(f"- **Errors**: All {total_runs} runs completed with 0% error rate")

    # 4. Biggest protocol outliers
    if comparisons:
        biggest_grpc_win = biggest_http_win = None
        biggest_grpc_adv = biggest_http_adv = 0.0
        for cp in comparisons:
            for label, fld, lower_better, _ in CHART_METRICS:
                adv = _cp_advantage(cp, fld, lower_better)
                if adv is not None and adv > biggest_grpc_adv:
                    biggest_grpc_adv = adv
                    biggest_grpc_win = (cp, label)
                if adv is not None and adv < biggest_http_adv:
                    biggest_http_adv = adv
                    biggest_http_win = (cp, label)

        if biggest_grpc_win and biggest_grpc_adv > 10:
            cp, metric = biggest_grpc_win
            lines.append(
                f"- **Largest gRPC win**: {biggest_grpc_adv:.0f}% on {metric} "
                f"— {cp.model} `{cp.scenario}` C={cp.concurrency}"
            )
        if biggest_http_win and abs(biggest_http_adv) > 10:
            cp, metric = biggest_http_win
            lines.append(
                f"- **Largest HTTP win**: {abs(biggest_http_adv):.0f}% on {metric} "
                f"— {cp.model} `{cp.scenario}` C={cp.concurrency}"
            )

    lines.append("")
    return lines


def _section_error_rates(experiments: list[ExperimentInfo]) -> list[str]:
    """Surface any non-zero error rates."""
    errors = [(e, r) for e in experiments for r in e.runs if r.error_rate > 0]
    if not errors:
        return []

    lines = [
        "<details>",
        f"<summary><b>Runs with Errors</b> ({len(errors)} runs)</summary>",
        "",
        "| Model | Protocol | Runtime | Workers | Scenario | C | Error Rate |",
        "|-------|----------|---------|---------|----------|--:|-----------:|",
    ]

    for e, r in sorted(errors, key=lambda x: x[1].error_rate, reverse=True):
        lines.append(
            f"| {e.model} | {e.protocol} | {e.runtime} | {e.worker_type} "
            f"| `{r.scenario}` | {r.concurrency} | {r.error_rate:.1%} |"
        )

    lines.extend(["", "</details>", ""])
    return lines


def _section_overview(experiments: list[ExperimentInfo], models_with_data: set[str]) -> list[str]:
    """Overview table — only models that have comparison data."""
    by_model: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        by_model[exp.model][exp.table_key] = exp

    # Discover columns dynamically
    all_keys: set[str] = set()
    for model in models_with_data:
        if model in by_model:
            all_keys.update(by_model[model].keys())

    _worker_order = {"single": 0, "multi": 1}
    _protocol_order = {"http": 0, "grpc": 1}

    def _col_sort_key(key: str) -> tuple:
        protocol, runtime, worker = key.split("_")
        return (
            _worker_order.get(worker, 9),
            runtime,
            _protocol_order.get(protocol, 9),
        )

    table_order = []
    for key in sorted(all_keys, key=_col_sort_key):
        protocol, runtime, worker = key.split("_")
        p_disp = _PROTOCOL_DISPLAY.get(protocol, protocol.upper())
        r_disp = _RUNTIME_DISPLAY.get(runtime, runtime)
        w_disp = worker.capitalize()
        table_order.append((key, f"{p_disp} {r_disp} {w_disp}"))

    header_cols = ["Model"] + [title for _, title in table_order]
    lines = [
        "### Overview",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["---"] * len(header_cols)) + "|",
    ]

    for model in sorted(models_with_data):
        if model not in by_model:
            continue
        model_exps = by_model[model]
        row = [model]
        for table_key, _ in table_order:
            if table_key not in model_exps:
                row.append("\u2796")
            else:
                exp = model_exps[table_key]
                has_errors = any(r.rps == 0 or r.output_throughput == 0 for r in exp.runs)
                row.append("\u26a0\ufe0f" if has_errors else "\u2705")
        lines.append("| " + " | ".join(row) + " |")

    # Note excluded models
    excluded = set(by_model.keys()) - models_with_data
    if excluded:
        names = ", ".join(sorted(excluded))
        lines.append("")
        lines.append(f"*Excluded from comparison (no matched gRPC/HTTP data): {names}*")

    lines.append("")
    return lines


def _aggregate_table(
    comparisons: list[ComparisonPoint],
    label_a: str = "gRPC",
    label_b: str = "HTTP",
) -> list[str]:
    """Render an aggregate comparison table for a list of ComparisonPoints."""
    lines = [
        "| Metric | Avg | Median | Verdict |",
        "|--------|----:|-------:|---------|",
    ]
    for label, fld, lower_better in AGGREGATE_METRICS:
        advs = [a for cp in comparisons if (a := _cp_advantage(cp, fld, lower_better)) is not None]
        if not advs:
            continue
        avg = sum(advs) / len(advs)
        med = median(advs)
        lines.append(
            f"| {label} "
            f"| {_fmt_winner(avg, label_a=label_a, label_b=label_b)} "
            f"| {_fmt_winner(med, label_a=label_a, label_b=label_b)} "
            f"| {_fmt_winner(avg, bold=True, label_a=label_a, label_b=label_b)} |"
        )
    return lines


def _section_aggregate(comparisons: list[ComparisonPoint]) -> list[str]:
    """Aggregate gRPC vs HTTP comparison table with per-runtime breakdown."""
    if not comparisons:
        return ["*No gRPC vs HTTP comparison data available.*", ""]

    lines = [
        "### Aggregate: gRPC vs HTTP",
        "",
        f"*{len(comparisons)} matched data points. "
        "Shows which protocol is better and by how much.*",
        "",
    ]
    lines.extend(_aggregate_table(comparisons))
    lines.append("")

    # Per-runtime breakdown
    by_runtime: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_runtime[cp.runtime].append(cp)

    if len(by_runtime) > 1:
        for runtime in sorted(by_runtime.keys()):
            rt_cps = by_runtime[runtime]
            rt_display = _RUNTIME_DISPLAY.get(runtime, runtime)
            lines.extend(
                [
                    f"#### {rt_display}: gRPC vs HTTP ({len(rt_cps)} points)",
                    "",
                ]
            )
            lines.extend(_aggregate_table(rt_cps))
            lines.append("")

    return lines


def _section_by_concurrency(comparisons: list[ComparisonPoint]) -> list[str]:
    """Per-concurrency breakdown for TTFT, E2E, and throughput."""
    if not comparisons:
        return []

    by_conc, conc_levels = _group_by_concurrency(comparisons)
    lines = ["### Performance by Concurrency", ""]

    for title, fld, lower_is_better in CONCURRENCY_METRICS:
        tbl = [
            "<details>",
            f"<summary><b>{title}</b></summary>",
            "",
            "| Concurrency | gRPC | HTTP | Faster |",
            "|---:|---:|---:|:---|",
        ]

        for conc in conc_levels:
            cps = by_conc[conc]
            g_avg = sum(getattr(cp.grpc, fld) for cp in cps) / len(cps)
            h_avg = sum(getattr(cp.http, fld) for cp in cps) / len(cps)
            adv = _advantage(g_avg, h_avg, lower_is_better)
            tbl.append(
                f"| {conc} | {_fmt_metric_value(g_avg, title)} "
                f"| {_fmt_metric_value(h_avg, title)} "
                f"| {_fmt_winner(adv, bold=True)} |"
            )
        tbl.extend(["", "</details>", ""])
        lines.extend(tbl)

    return lines


def _section_scorecard(comparisons: list[ComparisonPoint]) -> list[str]:
    """Win/loss scorecard at different thresholds."""
    if not comparisons:
        return []

    lines = [
        "### Win/Loss Scorecard",
        "",
        "*How often is one protocol better by > N%?*",
        "",
    ]

    thresholds = [1, 2, 5, 10]

    for label, fld, lower_better in SCORECARD_METRICS:
        lines.extend(
            [
                f"**{label}:**",
                "",
                "| Threshold | gRPC better | HTTP better | Within |",
                "|---:|---:|---:|---:|",
            ]
        )

        for thresh in thresholds:
            grpc_w = http_w = within = 0
            for cp in comparisons:
                adv = _cp_advantage(cp, fld, lower_better)
                if adv is None:
                    continue
                if adv > thresh:
                    grpc_w += 1
                elif adv < -thresh:
                    http_w += 1
                else:
                    within += 1

            g_str = f"**{grpc_w}**" if grpc_w > http_w else str(grpc_w)
            h_str = f"**{http_w}**" if http_w > grpc_w else str(http_w)
            lines.append(f"| >{thresh}% | {g_str} | {h_str} | {within} |")

        lines.append("")

    return lines


def _section_top_wins(comparisons: list[ComparisonPoint], threshold: float = 30.0) -> list[str]:
    """Table of all gRPC wins exceeding the threshold."""
    if not comparisons:
        return []

    wins: list[tuple[float, str, ComparisonPoint, float, float]] = []
    for cp in comparisons:
        for label, fld, lower_better in TOP_WINS_METRICS:
            g = getattr(cp.grpc, fld)
            h = getattr(cp.http, fld)
            adv = _advantage(g, h, lower_better)
            if adv is not None and adv > threshold:
                wins.append((adv, label, cp, g, h))

    wins.sort(key=lambda x: x[0], reverse=True)

    if not wins:
        return []

    lines = [
        "<details>",
        f"<summary><b>Top gRPC Wins (&gt;{threshold:.0f}%)</b> — {len(wins)} entries</summary>",
        "",
        "| gRPC better by | Model | Config | Scenario | C | Metric | gRPC | HTTP |",
        "|---------------:|-------|--------|----------|--:|--------|-----:|-----:|",
    ]

    for adv, label, cp, g, h in wins:
        g_str = _fmt_metric_value(g, label)
        h_str = _fmt_metric_value(h, label)
        lines.append(
            f"| {adv:.1f}% | {cp.model} | {cp.config} | "
            f"`{cp.scenario}` | {cp.concurrency} | {label} | {g_str} | {h_str} |"
        )

    lines.extend(["", "</details>", ""])
    return lines


def _section_per_model(comparisons: list[ComparisonPoint]) -> list[str]:
    """Per-model summary table plus collapsible detail tables."""
    if not comparisons:
        return []

    by_model: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_model[cp.model].append(cp)

    headers = [m[0] for m in PER_MODEL_METRICS]
    lines = [
        "### Per-Model Summary",
        "",
        "*Each cell shows which protocol is better and by how much.*",
        "",
        "| Model | " + " | ".join(headers) + " | N |",
        "|-------" + "|--------:" * len(headers) + "|---:|",
    ]

    for model in sorted(by_model.keys()):
        cps = by_model[model]
        cells = [_fmt_winner(_avg_advantage(cps, fld, lb)) for _, fld, lb in PER_MODEL_METRICS]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {len(cps)} |")

    lines.append("")

    # Detailed per-model/config tables
    for model in sorted(by_model.keys()):
        cps = by_model[model]
        by_config: dict[str, list[ComparisonPoint]] = defaultdict(list)
        for cp in cps:
            by_config[cp.config].append(cp)

        for config in sorted(by_config.keys()):
            config_cps = sorted(by_config[config], key=lambda x: (x.scenario, x.concurrency))
            lines.extend(
                [
                    "<details>",
                    f"<summary><b>{model} ({config})</b> — {len(config_cps)} points</summary>",
                    "",
                    "| Scenario | C | " + " | ".join(headers) + " |",
                    "|----------|--:" + "|--------:" * len(headers) + "|",
                ]
            )

            for cp in config_cps:
                cells = [
                    _fmt_winner(_cp_advantage(cp, fld, lb)) for _, fld, lb in PER_MODEL_METRICS
                ]
                lines.append(f"| `{cp.scenario}` | {cp.concurrency} | " + " | ".join(cells) + " |")

            lines.extend(["", "</details>", ""])

    return lines


def _rt_advantage(rcp: RuntimeComparisonPoint, fld: str, lower_is_better: bool) -> float | None:
    """Shorthand: compute runtime_a advantage for a RuntimeComparisonPoint."""
    return _advantage(getattr(rcp.run_a, fld), getattr(rcp.run_b, fld), lower_is_better)


def _avg_rt_advantage(
    rcps: list[RuntimeComparisonPoint], fld: str, lower_is_better: bool
) -> float | None:
    """Average runtime_a advantage across a list of runtime comparison points."""
    advs = [a for rcp in rcps if (a := _rt_advantage(rcp, fld, lower_is_better)) is not None]
    return sum(advs) / len(advs) if advs else None


def _group_by_runtime_pair(
    runtime_comparisons: list[RuntimeComparisonPoint],
) -> dict[tuple[str, str], list[RuntimeComparisonPoint]]:
    """Group runtime comparisons by (runtime_a, runtime_b) pair."""
    by_pair: dict[tuple[str, str], list[RuntimeComparisonPoint]] = defaultdict(list)
    for rcp in runtime_comparisons:
        by_pair[(rcp.runtime_a, rcp.runtime_b)].append(rcp)
    return by_pair


def _rt_aggregate_table(
    rcps: list[RuntimeComparisonPoint],
    rt_a_display: str,
    rt_b_display: str,
) -> list[str]:
    """Render an aggregate runtime comparison table."""
    lines = [
        "| Metric | Avg | Median | Verdict |",
        "|--------|----:|-------:|---------|",
    ]
    for label, fld, lower_better in AGGREGATE_METRICS:
        advs = [a for rcp in rcps if (a := _rt_advantage(rcp, fld, lower_better)) is not None]
        if not advs:
            continue
        avg = sum(advs) / len(advs)
        med = median(advs)
        lines.append(
            f"| {label} "
            f"| {_fmt_winner(avg, label_a=rt_a_display, label_b=rt_b_display)} "
            f"| {_fmt_winner(med, label_a=rt_a_display, label_b=rt_b_display)} "
            f"| {_fmt_winner(avg, bold=True, label_a=rt_a_display, label_b=rt_b_display)} |"
        )
    return lines


def _section_runtime_comparison(runtime_comparisons: list[RuntimeComparisonPoint]) -> list[str]:
    """Cross-runtime comparison sections, one per runtime pair."""
    if not runtime_comparisons:
        return []

    by_pair = _group_by_runtime_pair(runtime_comparisons)
    lines: list[str] = []

    for (rt_a, rt_b), pair_rcps in sorted(by_pair.items()):
        rt_a_display = _RUNTIME_DISPLAY.get(rt_a, rt_a)
        rt_b_display = _RUNTIME_DISPLAY.get(rt_b, rt_b)

        lines.extend(
            [
                f"### Runtime Comparison: {rt_a_display} vs {rt_b_display}",
                "",
                f"*{len(pair_rcps)} matched data points. "
                f"Shows which runtime is better and by how much.*",
                "",
            ]
        )
        lines.extend(_rt_aggregate_table(pair_rcps, rt_a_display, rt_b_display))
        lines.append("")

        # Per-protocol sub-tables
        by_protocol: dict[str, list[RuntimeComparisonPoint]] = defaultdict(list)
        for rcp in pair_rcps:
            by_protocol[rcp.protocol].append(rcp)

        if len(by_protocol) > 1:
            for protocol in sorted(by_protocol.keys()):
                p_cps = by_protocol[protocol]
                p_display = _PROTOCOL_DISPLAY.get(protocol, protocol.upper())
                lines.extend(
                    [
                        f"#### {p_display}: {rt_a_display} vs {rt_b_display} ({len(p_cps)} points)",
                        "",
                    ]
                )
                lines.extend(_rt_aggregate_table(p_cps, rt_a_display, rt_b_display))
                lines.append("")

        # Per-model runtime comparison
        by_model: dict[str, list[RuntimeComparisonPoint]] = defaultdict(list)
        for rcp in pair_rcps:
            by_model[rcp.model].append(rcp)

        if by_model:
            headers = [m[0] for m in PER_MODEL_METRICS]
            lines.extend(
                [
                    f"#### Per-Model: {rt_a_display} vs {rt_b_display}",
                    "",
                    "| Model | " + " | ".join(headers) + " | N |",
                    "|-------" + "|--------:" * len(headers) + "|---:|",
                ]
            )
            for model in sorted(by_model.keys()):
                rcps = by_model[model]
                cells = [
                    _fmt_winner(
                        _avg_rt_advantage(rcps, fld, lb),
                        label_a=rt_a_display,
                        label_b=rt_b_display,
                    )
                    for _, fld, lb in PER_MODEL_METRICS
                ]
                lines.append(f"| {model} | " + " | ".join(cells) + f" | {len(rcps)} |")
            lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def generate_summary(base_dir: Path) -> SummaryResult:
    """Generate the full markdown summary."""
    experiments = discover_experiments(base_dir)

    if not experiments:
        return SummaryResult(
            markdown="## Nightly Benchmark Summary\n\nNo benchmark results found.",
            experiments=[],
            comparisons=[],
        )

    comparisons = build_comparisons(experiments)
    runtime_comparisons = build_runtime_comparisons(experiments)

    # Only include models that have comparison data
    models_with_data = {cp.model for cp in comparisons}

    grpc_count = sum(1 for e in experiments if e.protocol == "grpc")
    http_count = sum(1 for e in experiments if e.protocol == "http")
    total_runs = sum(len(e.runs) for e in experiments)

    # Count experiments per runtime
    runtime_counts: dict[str, int] = defaultdict(int)
    for e in experiments:
        runtime_counts[e.runtime] += 1
    runtime_parts = ", ".join(
        f"{count} {_RUNTIME_DISPLAY.get(rt, rt)}" for rt, count in sorted(runtime_counts.items())
    )

    lines = [
        "## Nightly Benchmark Summary",
        "",
        f"> **{len(experiments)} experiments** ({runtime_parts}; "
        f"{grpc_count} gRPC, {http_count} HTTP), "
        f"**{total_runs} benchmark runs**, "
        f"**{len(comparisons)} protocol comparisons**, "
        f"**{len(runtime_comparisons)} runtime comparisons**",
        "",
        "<details>",
        "<summary><b>Glossary</b></summary>",
        "",
        *_GLOSSARY_LINES,
        "",
        "</details>",
        "",
        "---",
        "",
    ]

    lines.extend(_section_key_findings(comparisons, experiments, runtime_comparisons))
    lines.extend(_section_overview(experiments, models_with_data))
    lines.extend(_section_aggregate(comparisons))
    lines.extend(_section_runtime_comparison(runtime_comparisons))
    lines.extend(_section_by_concurrency(comparisons))
    lines.extend(_section_scorecard(comparisons))
    lines.extend(_section_top_wins(comparisons))
    lines.extend(_section_per_model(comparisons))
    lines.extend(_section_error_rates(experiments))

    lines.append("---")
    lines.append(
        f"*Generated from {len(experiments)} experiment(s), "
        f"{len(comparisons)} protocol comparisons, "
        f"{len(runtime_comparisons)} runtime comparisons*"
    )

    return SummaryResult(
        markdown="\n".join(lines),
        experiments=experiments,
        comparisons=comparisons,
        runtime_comparisons=runtime_comparisons,
    )


def main() -> None:
    """Main entry point.

    Usage: nightly_summarize.py [base_dir] [--charts-dir DIR]
    """
    args = sys.argv[1:]
    base_dir = Path.cwd()
    charts_dir: Path | None = None

    i = 0
    while i < len(args):
        if args[i] == "--charts-dir" and i + 1 < len(args):
            charts_dir = Path(args[i + 1])
            i += 2
        elif not args[i].startswith("-"):
            base_dir = Path(args[i])
            i += 1
        else:
            i += 1

    result = generate_summary(base_dir)

    # Generate comparison charts if requested
    if charts_dir and result.comparisons:
        chart_files = generate_charts(result.comparisons, charts_dir)
        if chart_files:
            print(
                f"Generated {len(chart_files)} chart(s) in {charts_dir}",
                file=sys.stderr,
            )

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(result.markdown)
            f.write("\n")
        print(f"Summary written to {summary_file}")
    else:
        print(result.markdown)


if __name__ == "__main__":
    main()

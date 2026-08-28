//! Per-request records and the end-of-run summary: latency percentiles,
//! cache-hit ratios, turn-2 worker affinity, and the turn-1 worker spread.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::args::Args;

/// A request's `cached_tokens / prompt_tokens` at or above this counts as a
/// cache hit (the design's hit definition).
const CACHE_HIT_RATIO: f64 = 0.3;

/// One completed (or failed) `/generate` request.
#[derive(Debug)]
pub struct RequestRecord {
    pub turn: u8,
    pub session: u64,
    pub key: String,
    pub smg: usize,
    pub worker_port: Option<u64>,
    /// Local count of the input ids sent, not the server's echo.
    pub prompt_tokens: usize,
    pub cached_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub max_new: u32,
    /// Elapsed to the first SSE data frame; `None` for non-streaming requests.
    pub ttft_ms: Option<f64>,
    pub e2e_ms: f64,
    /// HTTP status; 0 for transport failures (connect or mid-stream).
    pub status: u16,
    /// Request start, milliseconds since the Unix epoch.
    pub start_ms: u64,
    /// Gateway's remote-index echo (`x-smg-index-source`): what the
    /// prefetch resolved for this decision. `None` when the gateway runs
    /// without a remote index.
    pub index_source: Option<String>,
    /// Gateway's predicted cached tokens for the served worker
    /// (`x-smg-index-predicted-tokens`); comparing against the worker's
    /// actual `cached_tokens` separates index error from policy spill.
    pub index_predicted_tokens: Option<u64>,
}

impl RequestRecord {
    pub fn is_ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    fn finish_ms(&self) -> u64 {
        self.start_ms.saturating_add(self.e2e_ms as u64)
    }

    fn cached_ratio(&self) -> Option<f64> {
        let cached = self.cached_tokens? as f64;
        if self.prompt_tokens == 0 {
            return None;
        }
        Some(cached / self.prompt_tokens as f64)
    }

    /// One JSONL line.
    pub fn to_json(&self) -> Value {
        json!({
            "turn": self.turn,
            "session": self.session,
            "key": self.key,
            "smg": self.smg,
            "worker_port": self.worker_port,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "max_new": self.max_new,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "status": self.status,
            "start_ms": self.start_ms,
            "index_source": self.index_source,
            "index_predicted_tokens": self.index_predicted_tokens,
        })
    }
}

/// Build the summary document. Statistics cover the STEADY-STATE window
/// only: requests completing after `--warmup-secs` and before the arrival
/// window closes (`--duration-secs`). The drain tail — turns that finish
/// after arrivals stop — is reported separately and never mixed into
/// cache/latency/throughput comparisons.
pub fn summarize(
    args: &Args,
    records: &[RequestRecord],
    run_start_ms: u64,
    elapsed_secs: f64,
    sessions: u64,
) -> Value {
    let warmup_end_ms = run_start_ms.saturating_add(args.warmup_secs.saturating_mul(1000));
    let arrival_end_ms = run_start_ms.saturating_add(args.duration_secs.saturating_mul(1000));
    let measured: Vec<&RequestRecord> = records
        .iter()
        .filter(|r| {
            let finish = r.finish_ms();
            finish >= warmup_end_ms && finish < arrival_end_ms
        })
        .collect();
    let drain: Vec<&RequestRecord> = records
        .iter()
        .filter(|r| r.finish_ms() >= arrival_end_ms)
        .collect();
    let measured_secs = (args.duration_secs.saturating_sub(args.warmup_secs)).max(1) as f64;

    let mut errors: BTreeMap<String, u64> = BTreeMap::new();
    for record in records {
        if !record.is_ok() {
            *errors.entry(record.status.to_string()).or_insert(0) += 1;
        }
    }

    let ok: Vec<&RequestRecord> = measured.iter().copied().filter(|r| r.is_ok()).collect();
    let ttfts: Vec<f64> = ok.iter().filter_map(|r| r.ttft_ms).collect();
    let e2es: Vec<f64> = ok.iter().map(|r| r.e2e_ms).collect();

    // Turn-1 worker per session, from the whole run: a warmup turn 1 still
    // anchors its session's turn-2 affinity.
    let mut t1_ports: HashMap<u64, u64> = HashMap::new();
    for record in records {
        if record.turn == 1 && record.is_ok() {
            if let Some(port) = record.worker_port {
                t1_ports.entry(record.session).or_insert(port);
            }
        }
    }
    let t2_matches: Vec<bool> = ok
        .iter()
        .filter(|r| r.turn == 2)
        .filter_map(|r| {
            let port = r.worker_port?;
            Some(port == *t1_ports.get(&r.session)?)
        })
        .collect();
    let same_worker_rate = if t2_matches.is_empty() {
        None
    } else {
        Some(t2_matches.iter().filter(|&&same| same).count() as f64 / t2_matches.len() as f64)
    };

    // Consecutive-turn stickiness across ALL follow-up turns: each turn t is
    // compared with its session's turn t-1 worker (from the whole run).
    let mut turn_ports: HashMap<(u64, u8), u64> = HashMap::new();
    for record in records {
        if record.is_ok() {
            if let Some(port) = record.worker_port {
                turn_ports
                    .entry((record.session, record.turn))
                    .or_insert(port);
            }
        }
    }
    let followup_matches: Vec<bool> = ok
        .iter()
        .filter(|r| r.turn >= 2)
        .filter_map(|r| {
            let port = r.worker_port?;
            Some(port == *turn_ports.get(&(r.session, r.turn - 1))?)
        })
        .collect();
    let followup_same_worker_rate = if followup_matches.is_empty() {
        None
    } else {
        Some(
            followup_matches.iter().filter(|&&same| same).count() as f64
                / followup_matches.len() as f64,
        )
    };

    // Mean turns per session, over sessions with at least one recorded turn.
    let mut max_turn: HashMap<u64, u8> = HashMap::new();
    for record in records {
        let entry = max_turn.entry(record.session).or_insert(0);
        *entry = (*entry).max(record.turn);
    }
    let mean_turns = if max_turn.is_empty() {
        None
    } else {
        Some(max_turn.values().map(|&t| f64::from(t)).sum::<f64>() / max_turn.len() as f64)
    };

    let mut worker_counts: BTreeMap<u64, u64> = BTreeMap::new();
    for record in ok.iter().filter(|r| r.turn == 1) {
        if let Some(port) = record.worker_port {
            *worker_counts.entry(port).or_insert(0) += 1;
        }
    }
    let t1_total: u64 = worker_counts.values().sum();
    let distinct = worker_counts.len();
    let max_share = if t1_total > 0 {
        worker_counts
            .values()
            .max()
            .map(|&m| m as f64 / t1_total as f64)
    } else {
        None
    };
    // Normalized over the observed workers; a single worker is complete
    // concentration, so it reports 0 rather than dividing by ln(1).
    let normalized_entropy = if t1_total == 0 {
        None
    } else if distinct > 1 {
        let h: f64 = worker_counts
            .values()
            .map(|&c| {
                let p = c as f64 / t1_total as f64;
                -p * p.ln()
            })
            .sum();
        Some(h / (distinct as f64).ln())
    } else {
        Some(0.0)
    };

    let mut per_smg: Vec<u64> = vec![0; args.smg_urls.len()];
    for record in records {
        if let Some(slot) = per_smg.get_mut(record.smg) {
            *slot += 1;
        }
    }
    let per_smg_requests: Vec<Value> = args
        .smg_urls
        .iter()
        .zip(&per_smg)
        .map(|(url, &requests)| json!({"url": url, "requests": requests}))
        .collect();

    json!({
        "config": config_json(args),
        "totals": {
            "sessions": sessions,
            "requests": records.len(),
            "errors": errors,
        },
        "elapsed_secs": elapsed_secs,
        // Steady-state window: [warmup, arrival end). Offered rate and the
        // drain tail are reported separately so scenarios stay comparable.
        "window": {
            "start_secs": args.warmup_secs,
            "end_secs": args.duration_secs,
            "measured_requests": measured.len(),
        },
        "offered_session_rps": args.session_rps,
        "drain": {
            "requests": drain.len(),
            "ok": drain.iter().filter(|r| r.is_ok()).count(),
        },
        "achieved_rps": measured.len() as f64 / measured_secs,
        "ttft_ms": stats(&ttfts),
        "e2e_ms": stats(&e2es),
        "turns": {
            "turn1": turn_block(&measured, Some(1)),
            "turn2": turn_block(&measured, Some(2)),
            "followup": turn_block(&measured, None),
        },
        // All turns together; `cached_token_ratio` here is THE number to
        // compare with backend cached-token telemetry.
        "overall": overall_block(&measured),
        "turn2_same_worker_rate": same_worker_rate,
        "followup_same_worker_rate": followup_same_worker_rate,
        "mean_turns_per_session": mean_turns,
        "turn1_workers": {
            "distinct": distinct,
            "max_share": max_share,
            "normalized_entropy": normalized_entropy,
        },
        "per_smg_requests": per_smg_requests,
    })
}

fn config_json(args: &Args) -> Value {
    json!({
        "smg_urls": args.smg_urls,
        "duration_secs": args.duration_secs,
        "session_rps": args.session_rps,
        "t2_ratio": args.t2_ratio,
        "think_secs": args.think_secs,
        "stream": args.stream,
        "http2": args.http2,
        "conns_per_origin": args.conns_per_origin,
        "max_turns": args.max_turns,
        "request_timeout_secs": args.request_timeout_secs,
        "key_per_turn": args.key_per_turn,
        "ingress": args.ingress.as_str(),
        "turn2_ingress": args.turn2_ingress.as_str(),
        "routing_key_reuse": args.routing_key_reuse,
        "system_prefix_tokens": args.system_prefix_tokens,
        "image_count": args.image_count,
        "image_bytes": args.image_bytes,
        "image_placeholder_id": args.image_placeholder_id,
        "image_placeholder_run": args.image_placeholder_run,
        "t2_suffix_tokens": args.t2_suffix_tokens,
        "prompt_cdf": cdf_json(&args.prompt_cdf),
        "prompt_max": args.prompt_max,
        "output_cdf": cdf_json(&args.output_cdf),
        "output_max": args.output_max,
        "tokens_hint": args.tokens_hint,
        "payload": args.payload.as_str(),
        "model": args.model,
        "max_inflight": args.max_inflight,
        "warmup_secs": args.warmup_secs,
        "seed": args.seed,
        "out": args.out,
    })
}

fn cdf_json(anchors: &[(u32, f64)]) -> Vec<Value> {
    anchors
        .iter()
        .map(|&(tokens, cum)| json!([tokens, cum]))
        .collect()
}

/// Stats over every measured request regardless of turn.
fn overall_block(measured: &[&RequestRecord]) -> Value {
    let ok: Vec<&RequestRecord> = measured.iter().copied().filter(|r| r.is_ok()).collect();
    let ratios: Vec<f64> = ok.iter().filter_map(|r| r.cached_ratio()).collect();
    let prompt_sum: u64 = ok
        .iter()
        .filter(|r| r.cached_tokens.is_some())
        .map(|r| r.prompt_tokens as u64)
        .sum();
    let cached_sum: u64 = ok.iter().filter_map(|r| r.cached_tokens).sum();
    json!({
        "ok": ok.len(),
        "prompt_tokens_sum": prompt_sum,
        "cached_tokens_sum": cached_sum,
        "cached_token_ratio": if prompt_sum > 0 {
            Value::from(cached_sum as f64 / prompt_sum as f64)
        } else {
            Value::Null
        },
        "cached_ratio_request_mean": mean(&ratios),
    })
}

/// Stats over one exact turn (`Some(n)`) or every follow-up turn (`None`,
/// i.e. turn >= 2).
fn turn_block(measured: &[&RequestRecord], turn: Option<u8>) -> Value {
    let of_turn: Vec<&RequestRecord> = measured
        .iter()
        .copied()
        .filter(|r| match turn {
            Some(t) => r.turn == t,
            None => r.turn >= 2,
        })
        .collect();
    let ok: Vec<&RequestRecord> = of_turn.iter().copied().filter(|r| r.is_ok()).collect();
    let ratios: Vec<f64> = ok.iter().filter_map(|r| r.cached_ratio()).collect();
    let hit_rate = if ratios.is_empty() {
        None
    } else {
        Some(ratios.iter().filter(|&&r| r >= CACHE_HIT_RATIO).count() as f64 / ratios.len() as f64)
    };
    // Token-weighted ratio (Σcached/Σprompt) is the number comparable with
    // backend cached-token telemetry; the per-request mean is kept under its
    // own name because long prompts weigh the two differently.
    let prompt_sum: u64 = ok
        .iter()
        .filter(|r| r.cached_tokens.is_some())
        .map(|r| r.prompt_tokens as u64)
        .sum();
    let cached_sum: u64 = ok.iter().filter_map(|r| r.cached_tokens).sum();
    let ttfts: Vec<f64> = ok.iter().filter_map(|r| r.ttft_ms).collect();
    let e2es: Vec<f64> = ok.iter().map(|r| r.e2e_ms).collect();
    json!({
        "count": of_turn.len(),
        "ok": ok.len(),
        "prompt_tokens_sum": prompt_sum,
        "cached_tokens_sum": cached_sum,
        "cached_token_ratio": if prompt_sum > 0 {
            Value::from(cached_sum as f64 / prompt_sum as f64)
        } else {
            Value::Null
        },
        "cached_ratio_request_mean": mean(&ratios),
        "hit_rate": hit_rate,
        "ttft_ms": stats(&ttfts),
        "e2e_ms": stats(&e2es),
    })
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// {mean, p50, p90, p99} with nearest-rank percentiles from a sorted copy.
fn stats(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"mean": null, "p50": null, "p90": null, "p99": null});
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let pct = |q: f64| {
        let rank = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[rank]
    };
    json!({
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(turn: u8, session: u64, worker: u64, prompt: usize, cached: u64) -> RequestRecord {
        RequestRecord {
            turn,
            session,
            key: format!("sess-{session}"),
            smg: 0,
            worker_port: Some(worker),
            prompt_tokens: prompt,
            cached_tokens: Some(cached),
            completion_tokens: Some(8),
            max_new: 8,
            ttft_ms: Some(10.0),
            e2e_ms: 100.0,
            status: 200,
            start_ms: 1000,
            index_source: None,
            index_predicted_tokens: None,
        }
    }

    fn args() -> Args {
        let mut args = Args::defaults();
        args.smg_urls = vec!["http://127.0.0.1:30000".to_string()];
        args
    }

    #[test]
    fn token_weighted_ratio_is_not_the_request_mean() {
        // One fully-cached short prompt + one uncached long prompt: the
        // request mean says 0.5, the token-weighted ratio says 0.1. Backend
        // telemetry is token-weighted, so the summary must carry both under
        // distinct names.
        let records = vec![rec(1, 1, 9001, 100, 100), rec(1, 2, 9002, 900, 0)];
        let summary = summarize(&args(), &records, 0, 10.0, 2);
        let t1 = &summary["turns"]["turn1"];
        assert_eq!(t1["prompt_tokens_sum"], 1000);
        assert_eq!(t1["cached_tokens_sum"], 100);
        let token_weighted = t1["cached_token_ratio"].as_f64().unwrap();
        let request_mean = t1["cached_ratio_request_mean"].as_f64().unwrap();
        assert!((token_weighted - 0.1).abs() < 1e-9);
        assert!((request_mean - 0.5).abs() < 1e-9);
        let overall = &summary["overall"];
        assert!((overall["cached_token_ratio"].as_f64().unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn consecutive_turn_stickiness_and_mean_turns() {
        // Session 1: three turns all on 9001 (two sticky follow-ups).
        // Session 2: turn 2 moves workers (one non-sticky follow-up).
        let records = vec![
            rec(1, 1, 9001, 1000, 0),
            rec(2, 1, 9001, 2000, 1900),
            rec(3, 1, 9001, 3000, 2900),
            rec(1, 2, 9001, 1000, 0),
            rec(2, 2, 9002, 2000, 100),
        ];
        let summary = summarize(&args(), &records, 0, 10.0, 2);
        let followup_rate = summary["followup_same_worker_rate"].as_f64().unwrap();
        assert!((followup_rate - 2.0 / 3.0).abs() < 1e-9);
        // turn2_same_worker_rate keeps its original turn-2-vs-turn-1 meaning.
        assert!((summary["turn2_same_worker_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        let mean_turns = summary["mean_turns_per_session"].as_f64().unwrap();
        assert!((mean_turns - 2.5).abs() < 1e-9);
        // Follow-up block spans turns 2 and 3.
        assert_eq!(summary["turns"]["followup"]["ok"], 3);
    }
}

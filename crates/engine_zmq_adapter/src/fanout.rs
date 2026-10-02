//! `n > 1` fan-out shared by both dialects: the wire has no per-sample
//! demux, so each sample is its own engine request.

/// Shared n>1 fan-out scaffolding: clone the request into `n` subs and let
/// `per_sub` apply the engine-specific rid suffix and sampling tweaks. An
/// `n <= 1` request passes through untouched. The last sub reuses `req`
/// itself, so a multimodal payload is copied `n - 1` times, not `n`.
pub(crate) fn fan_out_n<R: Clone>(
    mut req: R,
    n: u32,
    mut per_sub: impl FnMut(&mut R, u32),
) -> Vec<R> {
    if n <= 1 {
        return vec![req];
    }
    let mut subs = Vec::with_capacity(n as usize);
    for i in 0..n - 1 {
        let mut sub = req.clone();
        per_sub(&mut sub, i);
        subs.push(sub);
    }
    per_sub(&mut req, n - 1);
    subs.push(req);
    subs
}

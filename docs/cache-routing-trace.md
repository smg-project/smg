# Cache routing diagnostics

Set `SMG_CACHE_TRACE=1` to capture gRPC pipeline routing evidence. Capture is
disabled by default. Dispatch records preserve the accepted middleware ID, a
unique dispatch ID, retry attempt and actual engine IDs. Client body IDs and
sticky routing semantics are unchanged. The TokenSpeed Python servicer uses
the same flag to log explicit parent/child mappings.

Selection records include sticky branches, policy predictions and count-pressure
gates. Predictions distinguish approximate tree units, hash levels and event
index overlap; they are not engine cache hits. Loads observed after selection
are labeled as such. Cache-aware eligibility vetoes retain their observed
protection state. Failures are logged even when no dispatch occurs.

To send compact `x-smg-cache-trace` diagnostics to a trusted gateway, also set
`SMG_CACHE_TRACE_HEADER=1`. Log capture alone adds no response header. The header
contains join fields and policy predictions, excludes candidate addresses and
load/protection state, and is limited to 2 KiB. Larger or non-ASCII captures are
log-only. Per request, at most 16 selections and 32 candidates or gates per
selection are retained, with explicit truncation. Downstream gateways should
verify the root ID, keep known fields and strip the internal header. Keep trace
details private. Missing predictions, filtered candidate sets and missing engine
lifecycle evidence remain unknown. This does not change the routing algorithm
or prove a globally attainable cache hit rate.

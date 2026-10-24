This source candidate leaves SMG_QWEN_MM_TRACE unset in the original six-case
caller. Activation needs a separately reviewed child with value 1 and
SMG_QWEN_MM_MODEL=Qwen/Qwen3-VL-8B-Instruct. Canonical MROPE, original image
order, all six quality oracles and the 20/16-minute budgets are unchanged.

The extra exact-image overlay is selected only by that opt-in. It observes
decoded CPU pixels by ordinal, CPU position tables before their existing H2D,
and actual full/per-item embedding-cache lookups. Cache events link to the
current thread's prefill batch using item id plus hash/pad/grid/offset values.
Missing, changed or multiply owned items report an error. No item/tensor is
retained, no GPU value is copied/read, and no request or image split is added.

Limits are first nine RIDs per PID, eight records per RID, 64 records per PID,
4096 bytes per JSON body, four items/grid rows per item, 8 MiB per CPU hash and
64 MiB total hashing per PID. Errors, caps and absent required stages are HOLD
for diagnostic acceptance; a green API case alone cannot close this evidence.
GPU cached-embedding values are explicitly NOT_READ. These observations do not
prove that a cached GPU tensor contains the CPU wire pixels or fix model quality.

The original five overlay import gates and scheduler witness stay in force.
Only the two exact new engine hooks defer origins to their actual events; the
new helper is checked during apply/final. Distribution metadata is not rewritten.
No new CPU, image-overlay application, model or GPU execution has occurred here.

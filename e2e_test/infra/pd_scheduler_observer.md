# Partial scheduler trace writer

The writer emits primitive JSONL observations marked partial.
Observation errors latch unknown and stop writing; native code is not imported.
There is no final completeness handshake or acceptance/ACK result.

Admission and finish hooks delegate once and retain native return/exception identity.
Returned request statuses are observed; input abort IDs never become ACKs.
Natural completion and unfinished results stay distinct from actual ABORTED returns.
Only primitive request snapshots are retained, with exact original IDs.
CPU tests use explicit native contract stubs; no backend or fixture is enabled.

First-scheduled markers follow each native registration through full free.
Connector hold retains the marker; streaming updates do not reset it.
Deferred queue and fence snapshots remain distinct from request removal.
Transfer ACK and peer-memory release stay unknown; no fixture is enabled.

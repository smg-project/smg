# Partial scheduler trace writer

The writer emits primitive JSONL observations marked partial.
Observation errors latch unknown and stop writing; native code is not imported.
There is no final completeness handshake or acceptance/ACK result.

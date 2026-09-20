# Experimental controls and metrics

Within a comparison, task instructions, tool definitions, model, sampling,
endpoint, evaluation, and baseline revision are held constant. Factors include
policy, short/medium/long context budget, task category, execution limits, and
seed. Trials rotate or seed-shuffle policy order to reduce temperature, cache,
and serving-load bias.

Every run records model/tokenizer identifiers, endpoint settings, platform,
seed, trial, execution index, baseline SHA, patch, trace, and official result.
Nothing from a prior run—solution, model state, retrieval index, or workspace—is
reused.

Metrics cover correctness; prompt/generated/auxiliary inference; request and
tool latency; first token; tool count; repeated reads/searches; context growth;
tool-output reductions; retrieval; compaction; and failures. Inter-token timing
depends on streamed server events and is unavailable when an endpoint does not
expose timestamps.

Aggregation reports sample size, successful and failed outcomes, medians, and
25th/75th/95th percentiles. Compare total inference tokens, not just mean prompt
length. A short prompt may need more requests or auxiliary processing and thus
cost more end to end.

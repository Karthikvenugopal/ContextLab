# Experiment manifests

Keep publication or study-specific YAML manifests here. A manifest should pin
task files, all four policies, context budgets, trials, seeds, ordering strategy,
model and tokenizer identifiers, endpoint settings, and output directory.

Do not commit generated runs or model caches. The small executable default lives
at `configs/benchmark.yaml`; copy it here before changing factors for a study.

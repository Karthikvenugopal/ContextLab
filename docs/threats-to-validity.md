# Threats to experimental validity

- **Model nondeterminism:** seeds do not guarantee identical GPU kernels or
  serving schedules. Use repeated trials and distributions.
- **Retrieval quality:** BM25 favors lexical overlap and may miss semantic
  relationships; dense models add their own training and version confounds.
- **Compaction loss:** summaries can omit constraints or introduce errors. Count
  revisits and inspect canonical traces.
- **Contamination:** instruction models may have seen public tasks or solutions.
  Prefer controlled private fixtures and disclose provenance.
- **Cache and order effects:** prefix/KV caches, GPU temperature, and concurrent
  load can favor later runs. Rotate order and record server settings.
- **Tokenizer mismatch:** fallback estimates differ from server usage. Preserve
  both, and use the exact served tokenizer for formal results.
- **Hardware variability:** batching, quantization, drivers, accelerators, and
  storage affect latency independently of policy.
- **Benchmark scope:** small fixtures improve reproducibility but have weak
  external validity for large production repositories.
- **Attribution:** repeated exploration after removal is evidence of temporal
  association, not proof that context loss caused the action.

Reports should distinguish observations, hypotheses, and future work, show
sample sizes, retain failed runs, and avoid claims of improvement unless the
measured design supports them.

# Reproducing benchmark results

1. Record the ContextLab commit, operating system, CPU/GPU, driver, Docker, model
   revision, tokenizer revision, and vLLM arguments.
2. Install the locked major dependency ranges with `pip install -e
   ".[dev,analysis]"` and run `contextlab doctor`.
3. Run `pytest`, then execute `contextlab benchmark run --config
   configs/benchmark.yaml` for the CPU mock validation.
4. Generate artifacts with `contextlab benchmark report --results
   results/context-policy-demo`.
5. For real inference, set `mock: false`, fix the endpoint/model, use multiple
   seeds, and archive the raw `config.json`, `runs/`, and `traces/` directories.

The mock configuration validates plumbing only. Do not compare its policy token
numbers as coding intelligence. Do not hand-edit aggregate files: the report
command regenerates them from immutable run and trace artifacts.

Generated charts cover success, prompt growth, total inference tokens, latency,
tool calls, retrieval volume, and compaction input. CSV and JSON are the source
for downstream statistics.

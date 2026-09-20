# Bundled internal benchmarks

`fixtures/` contains agent-visible repositories, `tasks/` contains strict task
manifests, and `evaluation/` contains official tests that are never copied into
an agent workspace. Results from these fixtures are **internal ContextLab
results**, not SWE-bench or another external benchmark.

Run `contextlab tasks list` to inspect the suite and read
`docs/benchmarks.md` before adding a task.

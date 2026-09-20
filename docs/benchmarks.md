# Benchmark tasks and official evaluation

A task YAML includes a stable ID, category, repository source, baseline revision
label, natural-language request, setup argv, visible validation argv,
evaluation-only test directory, success statement, and execution limits.

Create fixtures that are small enough for CPU CI but require authentic
repository work. Keep visible tests inside the fixture and official tests
outside it. Include edge cases only in official tests, but do not rely on
obscurity: workspace path enforcement must make them inaccessible.

The runner makes a deterministic git baseline, clones that exact SHA for every
condition, runs the agent, then invokes evaluation tests externally with the
workspace on `PYTHONPATH`. JUnit output determines passes, failures, errors,
skips, and pass rate. Agent completion and visible-test success are never
official correctness.

External suites need adapters that preserve their official data, images, patch
application rules, and evaluator. ContextLab does not claim SWE-bench Verified
support until an installation passes the official harness end to end.

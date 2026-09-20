# Coding-agent architecture

The runtime is deliberately small: prepare policy context, request one model
decision, validate its JSON, execute at most one tool, append the complete result
to canonical state, notify the policy, and repeat. A `finish` decision ends agent
work but does not determine benchmark success.

Available tools list files, read bounded line ranges, search text, write files,
run allowlisted argv commands, and explicitly retrieve context. Tool arguments
and results are Pydantic models. Paths are relative, symlinks are resolved, and
escapes are rejected. Commands do not invoke a shell and receive a limited
environment, timeout, and workspace cwd.

Limits independently cap agent steps, primary model requests, generated tokens,
tool calls, wall time, and each command. Malformed decisions consume normal
steps. Context overflow, inference errors, command failures, and timeouts remain
visible in traces.

The inference protocol is OpenAI-compatible. Regular and streamed responses
record server usage, total latency, and time to first token when available. Main
agent calls use `request_kind=agent`; compaction and future learned-retrieval
calls use distinct kinds so auxiliary work cannot disappear from totals.

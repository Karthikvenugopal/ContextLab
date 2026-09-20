# Context-management policy design

`ContextPolicy.prepare_context`, `observe`, and `recover` form the common
contract. `AgentState.canonical_messages` is the audit truth; a
`PreparedContext` is a policy-specific view with an audit map of retained,
removed, compressed, and recovered information.

Full history copies every message in chronological order and fails explicitly
when it does not fit. Bounded output reduces only tool messages and records the
call ID, original/retained/discarded tokens, and method. Retrieval retains the
original two instructions plus a recent window and inserts budgeted, attributed
BM25 results. Compaction retains instructions and recent messages around a
structured summary.

Every policy uses the same token budget. The usable prompt is model context
minus reserved generation and safety margin. System instructions, task text,
tool protocol, active history, recovery, and summary all count. A policy that
cannot fit must not silently switch strategy.

Compaction can trigger on tokens, utilization, step interval, or tool-output
volume. Its summary covers requirements, objective, findings, edits, errors,
tests, completed actions, and remaining work. The previous summary is included
in later compactions, while canonical events remain untouched.

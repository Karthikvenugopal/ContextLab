# Repository retrieval and context recovery

The indexer discovers supported text/code files, rejects `.git`, hidden
evaluation tests, and reference-patch directories, limits file size, then emits
overlapping line chunks. Each chunk has a source ID, path, line interval, SHA-256
content hash, and source type.

The default ranker is a dependency-free BM25 implementation with identifier and
CamelCase tokenization, deterministic tie breaking, source filtering, and path
prefix filtering. It works without a GPU. `DenseRetriever` is an optional
adapter contract for sentence-transformers/FAISS deployments.

Historical tool observations live in a separate immutable store and are ranked
independently from repository chunks before merged selection. Recovery obeys a
token cap and reports query, scores, sources, latency, volume, and repeat number.
The agent may also call the explicit `retrieve` tool.

Repeated queries, file reads, and searches are measured. Association with prior
context removal is labelled only when the trace supports it; association is not
causation.

# vLLM setup and GPU requirements

The GPU profile pins a vLLM OpenAI server image and defaults to
`Qwen/Qwen2.5-Coder-7B-Instruct`. Actual VRAM needs depend on weight precision,
context length, concurrency, and KV-cache utilization; validate against your
hardware rather than copying a claimed minimum.

```bash
cp .env.example .env
docker compose --profile gpu up --build vllm
curl http://127.0.0.1:8000/v1/models
contextlab doctor --endpoint http://127.0.0.1:8000/v1
```

Set the model identifier and maximum context in `.env`. Ensure the tokenizer
matches the served model and record both revisions for formal experiments.
ContextLab uses a strict JSON action protocol because native tool-call behavior
varies by model/template. The Compose command enables the Hermes parser for
users who also test native calls.

GPU tests are marked `vllm` and excluded from ordinary CPU assumptions. The
manual GitHub workflow requires a self-hosted `gpu` runner. Never report it as
passing without an actual workflow run and compatible hardware.

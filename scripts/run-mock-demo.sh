#!/usr/bin/env sh
set -eu
contextlab agent run --task benchmarks/tasks/localized_bug.yaml --policy full-history --config configs/agent.yaml --mock --output results

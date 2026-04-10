# Kit de rescate de memoria para agentes de IA: soluciona problemas de memoria de agentes de IA en 15 minutos

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

**The most common AI agent problems are memory problems.** Your agent crashes mid-conversation, forgets what you told it, or costs 10× more than it should — all because of broken memory management.

This repo contains a runnable demo of the diagnostic tools inside the [AI Agent Memory Rescue Kit](https://taiwildlab.com) (full kit at taiwildlab.com).

---

## The problem

Most AI agent failures come down to three root causes:

| Symptom | Root cause |
|---|---|
| Agent crashes after ~15 messages | Context window overflow |
| Agent forgets things you told it | Stale memory accumulation |
| API costs 10× higher than expected | No context compaction |
| Agent slows down over time | Memory store too large, linear search |
| Agent gives inconsistent answers | No memory prioritization |

These are not framework bugs. They are architectural problems that `ConversationBufferMemory` doesn't solve.

---

## Quick start

```bash
git clone https://github.com/taiwildlab/ai-agent-memory-rescue-kit
cd ai-agent-memory-rescue-kit
python3 demo.py
```

No pip install. No API keys. Runs in 3 seconds.

**Expected output:**

```
═══════════════════════════════════════════════════════════════
  AI AGENT MEMORY RESCUE KIT — Demo
  Python 3.8+ | No dependencies | stdlib only
═══════════════════════════════════════════════════════════════

  PATTERN 1 — Context Window Overflow Detector
  Model:        gpt-3.5-turbo
  Usage:        [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.6%
  Status:       OK
  ✓  Safe — ~514 more messages until overflow

  PATTERN 2 — Stale Memory Detector
  Stale (>24h):    6 (51 tokens)
  Recoverable:     [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10.8%

  CONTEXT COMPRESSOR — Sliding Window
  Original:    420 tokens (16 messages)
  Compressed:  275 tokens (7 messages)
  Reduction:   [██████████░░░░░░░░░░░░░░░░░░░░] 34.5%
```

---

## What this demo includes

### `ContextOverflowDetector`

Detects context window overflow **before** it crashes your agent. Supports model-specific limits:

```python
from demo import ContextOverflowDetector

detector = ContextOverflowDetector(model="gpt-4o")
detector.add_message("user", "Help me debug my agent", tokens=12)
detector.add_message("assistant", "What framework are you using?", tokens=9)

status = detector.check()
# {
#   "total_tokens": 21,
#   "usage_pct": 0.016,
#   "severity": "OK",
#   "msgs_until_overflow": 5120,
#   "will_overflow": False
# }

print(detector.get_fix())  # actionable fix if overflowing
```

Supported models: `gpt-4o`, `gpt-3.5-turbo`, `claude-3-5-sonnet`, `claude-haiku`, `llama-3.1-70b`, `qwen2.5-72b`.

### `StaleMemoryDetector`

Identifies messages consuming tokens without adding value — messages older than N hours that the agent ignores but still pays for.

```python
from demo import StaleMemoryDetector

detector = StaleMemoryDetector(stale_threshold_hours=24)
detector.add_messages(conversation_history)

analysis = detector.analyze()
# {
#   "stale_messages": 6,
#   "stale_tokens": 51,
#   "recovery_pct": 10.8,
#   "oldest_stale_hours": 48.0
# }
```

### `ContextCompressor`

Sliding-window compressor: keeps the last N messages in full, summarizes everything older. 34-60% token reduction.

```python
from demo import ContextCompressor

compressor = ContextCompressor(hot_window=6)
result = compressor.compress(messages)
# result["reduction_pct"] → 34.5
# result["compressed"] → list ready to pass to your LLM
```

---

## What's in the full kit

The demo covers 2 of the 5 patterns. The [full AI Agent Memory Rescue Kit](https://taiwildlab.com) ($75) includes:

| Feature | Demo | Full kit |
|---|---|---|
| Context overflow detection | ✓ | ✓ |
| Stale memory detection | ✓ | ✓ |
| Semantic importance scoring | — | ✓ |
| Hybrid vector + KV memory store | — | ✓ |
| Automatic pruning on overflow | — | ✓ |
| LangChain drop-in adapter | — | ✓ |
| CrewAI drop-in adapter | — | ✓ |
| System prompt templates (6 architectures) | — | ✓ |
| Implementation checklist (21 items) | — | ✓ |
| 30-min setup guarantee | — | ✓ |

**Full kit:** [taiwildlab.com](https://taiwildlab.com)

---

## Why memory problems are hard to diagnose

The failure mode is silent. Your agent doesn't throw an exception when it overflows — it either truncates the context (losing information) or hallucinates (making things up). You only notice when users report wrong answers.

The standard fix (`ConversationSummaryBufferMemory`) helps with overflow but doesn't solve:
- Which memories are worth keeping
- How to score relevance across time
- How to integrate with vector stores without external infrastructure
- How to handle multi-agent memory sharing

The full kit solves all of these.

---

## Keywords

`AI agent memory` · `LLM memory management` · `context window overflow` · `agent memory toolkit` · `LangChain memory` · `CrewAI memory` · `context compressor` · `token optimization` · `agent debugging` · `AI agent context`

---

## License

MIT — see [LICENSE](LICENSE).

Built by [TaiwildLab](https://taiwildlab.com) · Articles at [Descubriendo lo Esencial](https://descubriendoloesencial.substack.com)

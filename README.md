# EdgeMoE

Run frontier MoE models (Qwen3-235B, DeepSeek-R1 671B) on any laptop or free Kaggle GPU — for $0.

EdgeMoE streams expert weights on-demand from Google Drive or HuggingFace instead of loading the entire model into memory. Only the attention backbone plus a small hot-expert cache sits in VRAM; cold experts flow in through a proactive ML-predicted prefetcher while the GPU computes.

## Install

```bash
pip install edgemoe
```

## Quick start

```bash
# zero-setup streaming from HuggingFace
edgemoe run Qwen/Qwen3-30B-A3B

# your own 2TB Google Drive
edgemoe run Qwen3-235B --backend gdrive --gdrive-path /models/qwen3-235b/

# OpenAI-compatible API server
edgemoe serve Qwen3-235B --port 8080
```

## What's inside

| Module | What it does |
|---|---|
| `storage/` | mmap local SSD, Google Drive, HuggingFace hf-xet backends |
| `cache.py` | LSTM expert predictor + LRU fallback |
| `prefetch.py` | Background thread streaming layer N+1..N+3 |
| `quantization/turboquant.py` | 3-bit KV cache via rotation + polar + Lloyd-Max |
| `quantization/bitnet.py` | 1.58-bit ternary expert quantization |
| `attention/qjl.py` | QJL O(n) attention for 1M token context |
| `speculative.py` | Llama3-8B draft + big-model verify |
| `speculative_router.py` | Predict expert outputs, skip loading easy tokens |

## Targets

- **Qwen3-235B on Kaggle T4:** 8-15 tok/s
- **Context length on T4:** 1M tokens via QJL
- **Expert storage:** 57 GB (1.58-bit ternary) vs 140 GB Q4
- **User setup time:** 0 minutes

See the [project brief](docs/BRIEF.md) for full design rationale.

"""Benchmark EdgeMoE vs optional llama.cpp baseline."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def run_edgemoe(model: str, num_tokens: int = 100) -> dict:
    from edgemoe import EdgeMoE
    engine = EdgeMoE(model=model, backend="local")
    stats = engine.benchmark(num_tokens=num_tokens)
    engine.close()
    return stats


def run_llama_cpp(model_path: str, num_tokens: int = 100) -> dict:
    """Invoke llama.cpp's `main -p "..." -n N` and parse its timings.

    Requires `llama-cli` or `main` on PATH pointing at a GGUF model.
    Returns best-effort stats — skips gracefully if llama.cpp isn't set up.
    """
    binary = None
    for candidate in ("llama-cli", "main"):
        try:
            subprocess.run([candidate, "--help"], capture_output=True, check=False)
            binary = candidate
            break
        except FileNotFoundError:
            continue
    if binary is None:
        return {"error": "llama.cpp not installed or not on PATH"}

    t0 = time.perf_counter()
    proc = subprocess.run(
        [binary, "-m", model_path, "-p", "The quick brown fox", "-n", str(num_tokens)],
        capture_output=True, text=True, check=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "tokens": num_tokens,
        "seconds": elapsed,
        "tok_s": num_tokens / elapsed,
        "stdout_tail": proc.stdout.splitlines()[-5:] if proc.stdout else [],
    }


def run_benchmark(model: str, compare: str | None = None, num_tokens: int = 100) -> None:
    print(f"[bench] EdgeMoE on {model}")
    edge = run_edgemoe(model, num_tokens=num_tokens)
    print(json.dumps(edge, indent=2))

    if compare == "llama.cpp":
        print("[bench] llama.cpp baseline")
        llama = run_llama_cpp(model, num_tokens=num_tokens)
        print(json.dumps(llama, indent=2))
        if "tok_s" in llama and edge.get("tok_s"):
            ratio = edge["tok_s"] / llama["tok_s"]
            print(f"[bench] Speed ratio EdgeMoE / llama.cpp = {ratio:.2f}x")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--compare", default=None, choices=[None, "llama.cpp"])
    p.add_argument("--num-tokens", type=int, default=100)
    args = p.parse_args()
    run_benchmark(args.model, compare=args.compare, num_tokens=args.num_tokens)

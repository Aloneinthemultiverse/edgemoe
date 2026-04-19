"""EdgeMoE command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from edgemoe import EdgeMoE


@click.group()
@click.version_option()
def main():
    """EdgeMoE — expert-aware MoE inference for consumer hardware."""


@main.command()
@click.argument("model")
@click.option("--backend", default="huggingface",
              type=click.Choice(["local", "gdrive", "huggingface"]))
@click.option("--gdrive-path", default=None, help="Path inside Google Drive.")
@click.option("--model-path", default=None, help="Local SSD path to model dir.")
@click.option("--vram-gb", default=10.0, help="VRAM budget for expert cache.")
@click.option("--ram-gb", default=15.0, help="RAM budget for prefetch buffer.")
@click.option("--prompt", default="Explain quantum computing in one paragraph.")
@click.option("--max-tokens", default=128)
@click.option("--temperature", default=0.7)
def run(model, backend, gdrive_path, model_path, vram_gb, ram_gb,
        prompt, max_tokens, temperature):
    """Run a single prompt against the model."""
    if backend == "local":
        target = model_path or model
    elif backend == "gdrive":
        target = gdrive_path or model
    else:
        target = model

    engine = EdgeMoE(
        model=target,
        backend=backend,
        vram_budget_gb=vram_gb,
        ram_buffer_gb=ram_gb,
    )
    click.echo(f"[edgemoe] Generating {max_tokens} tokens…")
    text = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    click.echo(text)
    click.echo("---")
    click.echo(json.dumps(engine.cache.stats(), indent=2))
    engine.close()


@main.command()
@click.argument("model")
@click.option("--port", default=8080)
@click.option("--backend", default="huggingface")
def serve(model, port, backend):
    """Serve the model over an OpenAI-compatible REST API."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        click.echo("Install fastapi + uvicorn for the server.", err=True)
        sys.exit(1)

    engine = EdgeMoE(model=model, backend=backend)
    app = FastAPI(title="EdgeMoE")

    class ChatRequest(BaseModel):
        model: str
        messages: list
        max_tokens: int = 256
        temperature: float = 0.7

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        stats = engine.benchmark(num_tokens=req.max_tokens)
        return {
            "id": "edgemoe-0",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "(tokenizer pending)"},
                "finish_reason": "length",
            }],
            "usage": stats,
        }

    uvicorn.run(app, host="0.0.0.0", port=port)


@main.command()
@click.argument("hf_model")
@click.option("--output", required=True, type=click.Path())
@click.option("--quantize", default="q4",
              type=click.Choice(["q4", "q3", "q2", "bitnet"]))
def prepare(hf_model, output, quantize):
    """Split a HuggingFace MoE model into per-expert files."""
    from edgemoe.tools.split_experts import split_model
    split_model(hf_model, output_dir=output, quant_mode=quantize)


@main.command()
@click.argument("experts_dir", type=click.Path(exists=True))
@click.option("--bitnet/--no-bitnet", default=True)
def quantize(experts_dir, bitnet):
    """Apply further quantization to a prepared model."""
    from edgemoe.tools.quantize_experts import requantize_experts
    requantize_experts(experts_dir, to_bitnet=bitnet)


@main.command()
@click.argument("model")
@click.option("--compare", default=None)
@click.option("--num-tokens", default=100)
def benchmark(model, compare, num_tokens):
    """Benchmark inference; optionally compare against llama.cpp."""
    from edgemoe.tools.benchmark import run_benchmark
    run_benchmark(model, compare=compare, num_tokens=num_tokens)


if __name__ == "__main__":
    main()

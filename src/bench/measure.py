"""Timing and token counting for one generation run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from bench.config import ModelVariant
from bench.models import GenerationChunk, generate_stream
from bench.prompts import Prompt


@dataclass
class RunResult:
    variant_repo: str
    variant_quant: str
    prompt_id: str
    ttft_ms: float
    tokens_generated: int
    generation_time_s: float
    tokens_per_sec: float  # overall (kept for backward compat)
    prefill_tps: float  # prompt processing throughput
    decode_tps: float  # token generation throughput
    prompt_tokens: int  # number of input tokens
    peak_memory_bytes: int
    output_text: str
    is_warmup: bool = False


def measure_one(
    variant: ModelVariant,
    kind: str,
    model: Any,
    tokenizer_or_processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float = 0.0,
    is_warmup: bool = False,
) -> RunResult:
    """Run one generation and measure performance metrics."""
    # Reset peak memory counter
    mx.reset_peak_memory()

    chunks: list[GenerationChunk] = []
    ttft_ms = 0.0
    start = time.perf_counter()

    for i, chunk in enumerate(
        generate_stream(
            variant, kind, model, tokenizer_or_processor, prompt, max_tokens, temperature
        )
    ):
        if i == 0:
            ttft_ms = (time.perf_counter() - start) * 1000.0
        chunks.append(chunk)

    end = time.perf_counter()
    generation_time = end - start

    output_text = "".join(c.text for c in chunks)
    num_tokens = len(chunks)

    # Get prefill/decode stats from the last chunk (mlx_lm accumulates these)
    last = chunks[-1] if chunks else None
    prefill_tps = last.prompt_tps if last else 0.0
    decode_tps = last.generation_tps if last else 0.0
    prompt_tokens = last.prompt_tokens if last else 0

    # For vision models that return full output as one chunk, estimate token count
    if num_tokens == 1 and len(output_text) > 10:
        if hasattr(tokenizer_or_processor, "encode"):
            num_tokens = len(tokenizer_or_processor.encode(output_text))
        elif hasattr(tokenizer_or_processor, "tokenizer") and hasattr(
            tokenizer_or_processor.tokenizer, "encode"
        ):
            num_tokens = len(tokenizer_or_processor.tokenizer.encode(output_text))

    tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0.0

    # Prefer mlx_lm's decode_tps when available (more accurate);
    # fall back to our own calculation for vision models
    if decode_tps == 0.0:
        decode_tps = tokens_per_sec

    peak_memory = mx.get_peak_memory()

    return RunResult(
        variant_repo=variant.repo,
        variant_quant=variant.quant,
        prompt_id=prompt.id,
        ttft_ms=ttft_ms,
        tokens_generated=num_tokens,
        generation_time_s=generation_time,
        tokens_per_sec=tokens_per_sec,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        prompt_tokens=prompt_tokens,
        peak_memory_bytes=peak_memory,
        output_text=output_text,
        is_warmup=is_warmup,
    )

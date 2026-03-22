# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync                                          # Install dependencies
uv run bench                                     # TUI mode (default)
uv run bench --no-tui                            # CLI mode (stdout)
uv run bench --config configs/qwen35.toml --no-tui -v  # Custom config, verbose
```

Entry point: `bench = "bench.cli:main"` in pyproject.toml → `src/bench/cli.py:main()`.

## Architecture

### Data Flow

```
cli.py → runner.run_benchmark(config, on_progress)
           ├── For each ModelFamily:
           │     ├── Randomize variant order (thermal bias reduction)
           │     └── For each ModelVariant:
           │           ├── models.load_model() → (model, tokenizer)
           │           ├── power.begin_window()
           │           ├── For each Prompt:
           │           │     ├── Warmup runs (discarded)
           │           │     └── Measured runs → RunResult (TTFT, tok/s, memory)
           │           ├── power.end_window() → PowerReading
           │           ├── quality.compute_perplexity()
           │           ├── quality.eval_mmlu()
           │           └── gc.collect() + unload model
           ├── Compute output_similarity (quantized vs reference)
           ├── stats.aggregate() all metrics
           ├── store.save_session() → JSON
           └── report.generate_report() → HTML
```

### Key Abstractions

- **ModelFamily**: Groups variants of the same base model at different quantization levels (bf16/8bit/4bit). The `reference` variant is the quality baseline.
- **Prompt**: Has optional `max_tokens` override and `image` field. Text models skip vision prompts; vision models run all.
- **ProgressEvent**: Callback pattern (`on_progress`) drives both TUI and CLI output. Stages: loading → warmup → measuring → quality → done.
- **AggregatedMetric**: Median, mean, std, 95% CI (hardcoded t-table, no scipy), CV%. Flagged unreliable if CV% > 10%.

### Text vs Vision Model Handling

Text models use `mlx_lm.stream_generate()` which yields one token per step. Vision models use `mlx_vlm.generate()` which returns the full output as one chunk. For vision, token count is estimated by re-tokenizing the output post-hoc (`measure.py` lines 63-71).

## Library Workarounds

### mlx_lm 0.31+ API

`stream_generate()` no longer accepts `temp=`. Must use `sampler=make_sampler(temp=...)` from `mlx_lm.generate`. Both `models.py` and `quality.py` use this pattern.

### zeus-ml 0.15.0 Bugs (power.py)

Two issues, both worked around:

1. **Abstract method naming**: `AppleSiliconMeasurement` has `zero_all_fields()` but `DeprecatedAliasABCMeta` registers `zeroAllFields` as abstract → class is uninstantiable. Fix: remove `zeroAllFields` from `__abstractmethods__` before use.

2. **ZeusMonitor doesn't aggregate SoC energy**: `end_window().total_energy` returns 0. Fix: bypass `ZeusMonitor` entirely, use `AppleSilicon` SoC interface directly and read millijoule fields (cpu_total_mj, gpu_mj, dram_mj, ane_mj).

## Memory Management

- `mx.reset_peak_memory()` before each measured run
- Model + tokenizer deleted after each variant, followed by `gc.collect()` + `mx.reset_peak_memory()`
- OOM on model load is caught gracefully — variant is skipped, benchmark continues

## Adding a New Metric

1. Add field to `RunResult` in `measure.py`
2. Compute it in `measure_one()`
3. Aggregate in `runner.py` → `vr.aggregated["new_metric"] = aggregate([...])`
4. Add column to `report.py` tables (use `_th("Label", "num")` for tooltip)
5. Add tooltip text to `_HELP` dict in `report.py`

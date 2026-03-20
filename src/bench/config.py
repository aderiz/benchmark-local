"""Benchmark configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelVariant:
    repo: str
    quant: str


@dataclass
class ModelFamily:
    name: str
    kind: str  # "text" or "vision"
    size: str
    variants: list[ModelVariant]
    reference: str  # quant label of the reference variant

    def get_reference_variant(self) -> ModelVariant | None:
        for v in self.variants:
            if v.quant == self.reference:
                return v
        return None


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 3
    measured_runs: int = 10
    max_tokens: int = 256
    temperature: float = 0.0
    randomize_order: bool = True
    prompt_suite: str = "prompts/suite.toml"
    output_dir: str = "results"
    model_families: list[ModelFamily] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> BenchmarkConfig:
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        bench = data.get("benchmark", {})
        families = []
        for fam in data.get("model_family", []):
            variants = [
                ModelVariant(repo=v["repo"], quant=v["quant"])
                for v in fam["variants"]
            ]
            families.append(
                ModelFamily(
                    name=fam["name"],
                    kind=fam["kind"],
                    size=fam["size"],
                    variants=variants,
                    reference=fam["reference"],
                )
            )

        return cls(
            warmup_runs=bench.get("warmup_runs", 3),
            measured_runs=bench.get("measured_runs", 10),
            max_tokens=bench.get("max_tokens", 256),
            temperature=bench.get("temperature", 0.0),
            randomize_order=bench.get("randomize_order", True),
            prompt_suite=bench.get("prompt_suite", "prompts/suite.toml"),
            output_dir=bench.get("output_dir", "results"),
            model_families=families,
        )

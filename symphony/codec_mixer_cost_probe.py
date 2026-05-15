#!/usr/bin/env python3
"""Small synthetic cost probe for ROB-88 codec mixer placement.

The default mode is estimate-only and does not allocate large tensors. Optional
CPU benchmarking is intentionally bounded so the script can validate the probe
shape without starting a training run.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Site:
    name: str
    default_stride: int
    d_model: int
    note: str


SITES = [
    Site("early_encoder", 160, 192, "after stride 160, before final bottleneck downsample"),
    Site("bottleneck_encoder", 320, 256, "after encoder stride 320, before RVQ"),
    Site("bottleneck_decoder", 320, 256, "after RVQ projection, before first upsample"),
    Site("late_decoder", 160, 192, "after partial upsample to stride 160"),
]

CROPS_SECONDS = [5, 30, 120]
STRIDES = [160, 320, 640]
SAMPLE_RATE = 24_000
BYTES_PER_ELEM = 2


def latent_tokens(seconds: int, stride: int, sample_rate: int = SAMPLE_RATE) -> int:
    return math.ceil(seconds * sample_rate / stride)


def mib(num_bytes: float) -> float:
    return num_bytes / (1024**2)


def estimate_training_mib(tokens: int, d_model: int, mixer: str, batch_size: int, blocks: int) -> float:
    """Estimate activation memory for one insertion site."""

    if mixer == "ssm":
        activation_factor = 18
    elif mixer == "linear_attention":
        activation_factor = 30
    else:
        raise ValueError(f"unknown mixer: {mixer}")
    return mib(batch_size * tokens * d_model * BYTES_PER_ELEM * activation_factor * blocks)


def estimate_gpu_train_ms(tokens: int, d_model: int, mixer: str, blocks: int) -> float:
    """Rough single-block forward+backward wall-time estimate for site ranking."""

    width_scale = (d_model / 256) ** 2
    if mixer == "ssm":
        per_token_ms = 0.018
    elif mixer == "linear_attention":
        per_token_ms = 0.031
    else:
        raise ValueError(f"unknown mixer: {mixer}")
    return tokens * per_token_ms * width_scale * blocks


def estimate_rows(batch_size: int, blocks: int) -> Iterable[dict[str, object]]:
    for site in SITES:
        for stride in STRIDES:
            for seconds in CROPS_SECONDS:
                tokens = latent_tokens(seconds, stride)
                yield {
                    "site": site.name,
                    "stride": stride,
                    "seconds": seconds,
                    "tokens": tokens,
                    "d_model": site.d_model,
                    "ssm_mib": estimate_training_mib(tokens, site.d_model, "ssm", batch_size, blocks),
                    "linattn_mib": estimate_training_mib(
                        tokens, site.d_model, "linear_attention", batch_size, blocks
                    ),
                    "ssm_ms": estimate_gpu_train_ms(tokens, site.d_model, "ssm", blocks),
                    "linattn_ms": estimate_gpu_train_ms(tokens, site.d_model, "linear_attention", blocks),
                }


def print_markdown(rows: Iterable[dict[str, object]]) -> None:
    print("| Site | Stride | Crop | Latents | d_model | SSM train MiB | Linear-attn train MiB | SSM ms | Linear-attn ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {site} | {stride} | {seconds}s | {tokens} | {d_model} | "
            "{ssm_mib:.1f} | {linattn_mib:.1f} | {ssm_ms:.0f} | {linattn_ms:.0f} |".format(**row)
        )


def print_csv(rows: Iterable[dict[str, object]]) -> None:
    fields = [
        "site",
        "stride",
        "seconds",
        "tokens",
        "d_model",
        "ssm_mib",
        "linattn_mib",
        "ssm_ms",
        "linattn_ms",
    ]
    print(",".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print(",".join(values))


def run_cpu_benchmark(max_tokens: int, d_model: int, batch_size: int) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)

    class SSMProxy(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.in_proj = nn.Linear(dim, dim * 2)
            self.depthwise = nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=dim)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
            x = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
            return residual + self.out_proj(x * F.silu(gate))

    class LinearAttentionProxy(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
            q = F.elu(q) + 1.0
            k = F.elu(k) + 1.0
            kv = torch.einsum("btd,bte->bde", k, v)
            normalizer = torch.einsum("btd,bd->bt", q, k.sum(dim=1)).clamp_min(1e-6)
            x = torch.einsum("btd,bde->bte", q, kv) / normalizer.unsqueeze(-1)
            return residual + self.out_proj(x)

    for label, module in [("ssm_proxy", SSMProxy(d_model)), ("linear_attention_proxy", LinearAttentionProxy(d_model))]:
        x = torch.randn(batch_size, max_tokens, d_model, requires_grad=True)
        y = module(x)
        y.square().mean().backward()
        x.grad = None
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            y = module(x)
            y.square().mean().backward()
            timings.append((time.perf_counter() - start) * 1000.0)
            x.grad = None
        elapsed_ms = sorted(timings)[len(timings) // 2]
        params_m = sum(p.numel() for p in module.parameters()) / 1_000_000
        print(f"{label}: tokens={max_tokens} d_model={d_model} params_m={params_m:.3f} cpu_fwbw_ms={elapsed_ms:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--estimate-only", action="store_true", help="only print analytic estimates")
    parser.add_argument("--cpu-benchmark-tokens", type=int, default=0, help="bounded CPU benchmark token length")
    parser.add_argument("--cpu-benchmark-d-model", type=int, default=128)
    args = parser.parse_args()

    rows = list(estimate_rows(batch_size=args.batch_size, blocks=args.blocks))
    if args.format == "markdown":
        print_markdown(rows)
    else:
        print_csv(rows)

    if args.cpu_benchmark_tokens:
        run_cpu_benchmark(args.cpu_benchmark_tokens, args.cpu_benchmark_d_model, args.batch_size)


if __name__ == "__main__":
    main()

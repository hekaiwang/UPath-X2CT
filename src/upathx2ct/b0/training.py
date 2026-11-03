from __future__ import annotations

import contextlib
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        return DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=torch.device("cuda", local_rank))


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank * 100003
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed + worker_id)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(precision: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=precision == "fp16")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    value = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= context.world_size
    return value


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int, minimum_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class CSVMetricLogger:
    def __init__(self, path: str | Path, fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, extrasaction="ignore")
        if not exists:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step,
        "best_metric": best_metric,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def restore_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    strict: bool = True,
) -> tuple[int, float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_metric", -math.inf)), checkpoint


def prune_checkpoints(directory: str | Path, pattern: str, keep: int = 3) -> None:
    directory = Path(directory)
    paths = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    for path in paths[:-keep]:
        path.unlink()


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)

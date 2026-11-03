from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Warm-started model EMA stored alongside raw optimization weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        warm_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        decay = min(self.decay, warm_decay)
        for name, value in model.state_dict().items():
            if name in self.shadow and torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow")
        if not isinstance(shadow, dict):
            raise ValueError("EMA checkpoint is missing a shadow state")
        loaded = {}
        for name, value in shadow.items():
            if not isinstance(value, torch.Tensor):
                continue
            name = str(name)
            target = self.shadow.get(name)
            loaded[name] = value.detach().to(target).clone() if target is not None else value.detach().clone()
        self.shadow = loaded

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        state = model.state_dict()

        def swap() -> None:
            with torch.no_grad():
                for name, value in state.items():
                    if name not in self.shadow:
                        continue
                    temporary = value.detach().clone()
                    value.copy_(self.shadow[name])
                    self.shadow[name].copy_(temporary)

        try:
            swap()
            yield
        finally:
            swap()


def resolve_checkpoint_model_state(
    checkpoint: dict[str, object],
    prefer_ema: bool = True,
) -> dict[str, torch.Tensor]:
    raw = checkpoint.get("model", checkpoint)
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint does not contain a model state")
    state = {str(name): value for name, value in raw.items() if isinstance(value, torch.Tensor)}
    ema = checkpoint.get("ema")
    if prefer_ema and isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        state.update(
            {
                str(name): value
                for name, value in ema["shadow"].items()
                if isinstance(value, torch.Tensor)
            }
        )
    return state

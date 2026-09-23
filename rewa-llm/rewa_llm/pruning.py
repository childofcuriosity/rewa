"""Global magnitude pruning utilities for the stage-one experiment.

The pruning population is deliberately identical to the ReWA population:
the attention and MLP weight matrices selected by
``rewa_llm.model.is_rewa_parameter``.  A single global ordering is used, so
layers are free to end up at different sparsities.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import nn

from .model import is_rewa_parameter


@dataclass(frozen=True)
class PruningSelection:
    """Description of one exact global pruning selection."""

    target_sparsity: float
    total_eligible: int
    selected_count: int

    @property
    def realized_selection_sparsity(self) -> float:
        if self.total_eligible == 0:
            return 0.0
        return self.selected_count / self.total_eligible


def eligible_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return eligible parameters in the model's stable registration order."""

    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if is_rewa_parameter(name, parameter)
    ]


def _selection_count(total: int, target_sparsity: float) -> int:
    if not math.isfinite(target_sparsity) or not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity must be a finite number in [0, 1]")
    # Round to the nearest representable number of weights.  Avoid Python's
    # bankers rounding so a half weight is handled consistently.
    return min(total, int(math.floor(total * target_sparsity + 0.5)))


class GlobalMagnitudePruner:
    """Snapshot a model and apply exact, reversible global magnitude pruning.

    The absolute values are sorted once on CPU.  ``stable=True`` supplies a
    deterministic tie break based on parameter registration order and flattened
    element index.  Consequently exactly ``round(sparsity * N)`` entries are
    selected even when many weights have identical magnitude.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.named_parameters = eligible_named_parameters(model)
        if not self.named_parameters:
            raise ValueError("model has no eligible attention/MLP weight matrices")

        self.names = [name for name, _ in self.named_parameters]
        self.sizes = [parameter.numel() for _, parameter in self.named_parameters]
        self.total_eligible = sum(self.sizes)
        self._originals = [
            parameter.detach().clone(memory_format=torch.preserve_format)
            for _, parameter in self.named_parameters
        ]

        magnitudes = torch.cat(
            [original.detach().abs().reshape(-1).cpu() for original in self._originals]
        )
        if not bool(torch.isfinite(magnitudes).all()):
            raise ValueError("eligible weights contain NaN or infinity")
        self._global_order = torch.argsort(magnitudes, stable=True)
        self._active_selection: PruningSelection | None = None

    @property
    def active_selection(self) -> PruningSelection | None:
        return self._active_selection

    @torch.no_grad()
    def restore(self) -> None:
        """Restore every eligible matrix to its construction-time value."""

        for (_, parameter), original in zip(self.named_parameters, self._originals):
            parameter.copy_(original)
        self._active_selection = None

    @torch.no_grad()
    def apply(self, target_sparsity: float) -> PruningSelection:
        """Restore the snapshot, then zero an exact global number of entries."""

        selected_count = _selection_count(self.total_eligible, target_sparsity)
        self.restore()

        # A boolean selection mask makes ties harmless: the stable global order
        # chooses an exact subset rather than thresholding all equal values.
        selected = torch.zeros(self.total_eligible, dtype=torch.bool)
        if selected_count:
            selected[self._global_order[:selected_count]] = True

        offset = 0
        for (_, parameter), size in zip(self.named_parameters, self.sizes):
            local_selection = selected[offset : offset + size].view(parameter.shape)
            parameter.masked_fill_(local_selection.to(parameter.device), 0)
            offset += size

        selection = PruningSelection(
            target_sparsity=float(target_sparsity),
            total_eligible=self.total_eligible,
            selected_count=selected_count,
        )
        self._active_selection = selection
        return selection

    @contextmanager
    def pruned(self, target_sparsity: float) -> Iterator[PruningSelection]:
        """Temporarily apply pruning and restore weights on every exit path."""

        selection = self.apply(target_sparsity)
        try:
            yield selection
        finally:
            self.restore()


def _parameter_stats(name: str, parameter: nn.Parameter) -> dict[str, int | float | str]:
    zeros = int(torch.count_nonzero(parameter.detach() == 0).item())
    total = parameter.numel()
    return {
        "name": name,
        "zeros": zeros,
        "numel": total,
        "nonzeros": total - zeros,
        "sparsity": zeros / total if total else 0.0,
    }


@torch.no_grad()
def sparsity_report(model: nn.Module) -> dict[str, object]:
    """Measure exact-zero sparsity for eligible matrices and the whole model."""

    eligible = eligible_named_parameters(model)
    eligible_layers = [_parameter_stats(name, parameter) for name, parameter in eligible]
    eligible_total = sum(int(layer["numel"]) for layer in eligible_layers)
    eligible_zeros = sum(int(layer["zeros"]) for layer in eligible_layers)

    groups: dict[str, dict[str, int | float]] = {}
    for group_name, marker in (("attention", ".attn."), ("mlp", ".mlp.")):
        group_layers = [layer for layer in eligible_layers if marker in str(layer["name"])]
        group_total = sum(int(layer["numel"]) for layer in group_layers)
        group_zeros = sum(int(layer["zeros"]) for layer in group_layers)
        groups[group_name] = {
            "zeros": group_zeros,
            "numel": group_total,
            "nonzeros": group_total - group_zeros,
            "sparsity": group_zeros / group_total if group_total else 0.0,
        }

    # named_parameters removes tied duplicates by default.  Thus a tied token
    # embedding / LM head contributes once to whole-model parameter counts.
    whole_layers = [
        _parameter_stats(name, parameter) for name, parameter in model.named_parameters()
    ]
    whole_total = sum(int(layer["numel"]) for layer in whole_layers)
    whole_zeros = sum(int(layer["zeros"]) for layer in whole_layers)

    return {
        "eligible": {
            "zeros": eligible_zeros,
            "numel": eligible_total,
            "nonzeros": eligible_total - eligible_zeros,
            "sparsity": eligible_zeros / eligible_total if eligible_total else 0.0,
        },
        "whole_model": {
            "zeros": whole_zeros,
            "numel": whole_total,
            "nonzeros": whole_total - whole_zeros,
            "sparsity": whole_zeros / whole_total if whole_total else 0.0,
        },
        "groups": groups,
        "layers": eligible_layers,
    }

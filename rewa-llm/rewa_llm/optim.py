from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


def signed_power(value: Tensor, exponent: float) -> Tensor:
    """Return ``sign(value) * abs(value) ** exponent`` elementwise.

    ReWA only calls this function with a positive exponent.  Keeping the sign
    outside ``pow`` is important: a fractional power of a negative number is
    otherwise NaN, while zero remains exactly zero.
    """
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("signed_power exponent must be finite and positive")
    return value.sign() * value.abs().pow(exponent)


def _working_dtype(parameter: Tensor) -> torch.dtype:
    """Use FP32 for normal model dtypes, without downcasting FP64 tests."""
    return torch.float64 if parameter.dtype == torch.float64 else torch.float32


class ReWAAdamW(Optimizer):
    """AdamW with optional ReWA updates selected per parameter group.

    Groups with ``rewa=True`` are updated in y-space where
    x = sign(y) * |y|**K. Other groups receive an ordinary AdamW update.
    Optimizer moments and the y-space calculation use at least FP32.  This is
    also true when model parameters are FP16/BF16 and after loading a state
    dict.  Callers should normally keep model parameters in FP32 and use
    autocast for the forward pass.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rewa_k: float = 3.0,
        rewa_m: float = 2.0,
        rewa_eps: float = 1e-8,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0 or rewa_eps < 0:
            raise ValueError("optimizer rates and epsilons must be non-negative")
        if rewa_k <= 0:
            raise ValueError("rewa_k must be positive")
        if rewa_m < 0:
            raise ValueError("rewa_m must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rewa=False,
            rewa_k=rewa_k,
            rewa_m=rewa_m,
            rewa_eps=rewa_eps,
        )
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict):
        """Load a checkpoint while retaining full-precision moment buffers.

        ``Optimizer.load_state_dict`` normally casts floating-point state to
        the parameter dtype.  That would silently turn the moments into FP16
        when loading a mixed-precision checkpoint, so cast the two Adam
        buffers back to the optimizer working dtype after the standard load.
        """
        super().load_state_dict(state_dict)
        for parameter, state in self.state.items():
            state_dtype = _working_dtype(parameter)
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].to(
                        device=parameter.device, dtype=state_dtype
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]
            weight_decay = group["weight_decay"]
            use_rewa = group.get("rewa", False)
            k = float(group.get("rewa_k", 3.0))
            m = float(group.get("rewa_m", 2.0))
            rewa_eps = float(group.get("rewa_eps", 1e-8))

            if k <= 0.0 or not math.isfinite(k):
                raise ValueError("rewa_k must be finite and positive")
            if m < 0.0 or not math.isfinite(m):
                raise ValueError("rewa_m must be finite and non-negative")
            if rewa_eps < 0.0 or not math.isfinite(rewa_eps):
                raise ValueError("rewa_eps must be finite and non-negative")

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("ReWAAdamW does not support sparse gradients")
                if not parameter.is_floating_point():
                    raise RuntimeError("ReWAAdamW parameters must be floating point")

                state_dtype = _working_dtype(parameter)
                grad = parameter.grad.detach().to(dtype=state_dtype)
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=state_dtype, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=state_dtype, memory_format=torch.preserve_format
                    )

                # Be defensive about state created by an older implementation
                # or cast by a checkpoint loader.
                for name in ("exp_avg", "exp_avg_sq"):
                    if (
                        state[name].dtype != state_dtype
                        or state[name].device != parameter.device
                    ):
                        state[name] = state[name].to(
                            device=parameter.device, dtype=state_dtype
                        )

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if use_rewa:
                    parameter_fp = parameter.detach().to(dtype=state_dtype)
                    working = signed_power(parameter_fp, 1.0 / k)
                    abs_y = working.abs()
                    if rewa_eps == 0.0:
                        scale = abs_y.pow(m)
                    else:
                        jacobian = k * abs_y.pow(k - 1.0)
                        # Written this way, the limiting cases remain finite:
                        # jacobian=0 gives 0 and jacobian=inf gives 1.
                        attenuation = torch.reciprocal(
                            1.0 + rewa_eps / jacobian
                        )
                        scale = attenuation.mul(abs_y.pow(m))
                    grad = grad.mul(scale)
                else:
                    if parameter.dtype == state_dtype:
                        working = parameter
                    else:
                        working = parameter.detach().to(dtype=state_dtype)

                if weight_decay:
                    working.mul_(1.0 - lr * weight_decay)

                # ``lerp_`` mirrors torch.optim.AdamW's single-tensor update.
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(adam_eps)
                working.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)

                if use_rewa:
                    parameter.copy_(signed_power(working, k).to(parameter.dtype))
                elif working is not parameter:
                    parameter.copy_(working.to(parameter.dtype))

        return loss

"""
JumpReLU activation function as defined in SAE lens
https://github.com/jbloomAus/SAELens/blob/main/sae_lens/saes/jumprelu_sae.py
"""

import torch
from typing import Any

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        grad_per_element = -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        
        dims_to_sum = tuple(range(x.dim() - 1)) # Account for sequence
        threshold_grad = torch.sum(grad_per_element, dim=dims_to_sum)
        
        return None, threshold_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        
        grad_per_element = -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        dims_to_sum = tuple(range(x.dim() - 1))
        threshold_grad = torch.sum(grad_per_element, dim=dims_to_sum)

        return x_grad, threshold_grad, None
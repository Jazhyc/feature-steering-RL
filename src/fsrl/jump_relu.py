# src/fsrl/utils/jump_relu.py

import torch
from torch import nn

class _JumpReLUWithSparsity(torch.autograd.Function):
    """
    A unified function that computes both the JumpReLU activation and a
    differentiable sparsity mask.

    Outputs:
    1. post_activations: The result of the JumpReLU function (same as before).
    2. sparsity_mask: The result of a Heaviside step function, which can be
                       summed to get a differentiable L0 norm.

    The backward pass correctly routes gradients from the reconstruction loss
    (via post_activations) and the L0 sparsity loss (via sparsity_mask)
    to the learnable threshold parameter.
    """
    @staticmethod
    def forward(ctx, pre_activations, threshold, bandwidth):
        """
        Forward pass computes both JumpReLU and the Heaviside mask.
        """
        ctx.save_for_backward(pre_activations, threshold)
        ctx.bandwidth = bandwidth

        # Output 1: Sparsity mask (just the Heaviside result)
        sparsity_mask = (pre_activations > threshold).to(pre_activations.dtype)

        # Output 2: JumpReLU post-activations
        post_activations = pre_activations * sparsity_mask
        
        return post_activations, sparsity_mask

    @staticmethod
    def backward(ctx, grad_post_activations, grad_sparsity_mask):
        """
        Backward pass handles gradients from two different outputs.
        """
        pre_activations, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Get the dtype and device from an existing tensor to ensure consistency.
        dtype = pre_activations.dtype
        device = pre_activations.device

        # --- Gradient for the Pre-activations (input x) ---
        # This only comes from the reconstruction loss path (via post_activations).
        # The L0 loss does not train the pre-activations.
        is_active_mask = (pre_activations > threshold).to(dtype)
        grad_pre_activations = grad_post_activations * is_active_mask
        
        # --- Gradient for the Threshold (learnable parameter θ) ---
        diff = pre_activations - threshold
        half_bandwidth = torch.tensor(bandwidth / 2.0, dtype=dtype, device=device)
        rectangle_window = (torch.abs(diff) < half_bandwidth).to(dtype)

        # Part 1: Gradient from the reconstruction loss (from grad_post_activations)
        # This corresponds to Eq. (11) in the paper.
        grad_from_recon = -(threshold / bandwidth) * rectangle_window
        grad_from_recon *= grad_post_activations

        # Part 2: Gradient from the sparsity loss (from grad_sparsity_mask)
        # This corresponds to Eq. (12) in the paper.
        inv_bandwidth = torch.tensor(1.0 / bandwidth, dtype=dtype, device=device)
        grad_from_sparsity = -inv_bandwidth * rectangle_window
        grad_from_sparsity *= grad_sparsity_mask

        # Total gradient for the threshold is the sum of both paths.
        total_grad_for_threshold = grad_from_recon + grad_from_sparsity

        # The gradient for the threshold parameter must match its shape (num_features,).
        # We sum over dim=0 (the batch dimension) to achieve this.
        total_grad_for_threshold = total_grad_for_threshold.sum(dim=0)
        
        # Return gradients for each input of forward(): pre_activations, threshold, bandwidth.
        return grad_pre_activations, total_grad_for_threshold, None

class JumpReLU(nn.Module):
    """
    JumpReLU activation function that also returns a differentiable sparsity mask
    for calculating the L0 norm penalty, as described in https://arxiv.org/abs/2407.14435.
    """
    def __init__(self, num_features: int, initial_threshold: float = 0.001, bandwidth: float = 0.001):
        super().__init__()
        if initial_threshold <= 0:
            raise ValueError("Initial threshold must be positive.")
        self.log_threshold = nn.Parameter(torch.full((num_features,), torch.log(torch.tensor(initial_threshold))))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The pre-activation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - post_activations (torch.Tensor): The output of the JumpReLU function.
                - sparsity_mask (torch.Tensor): A differentiable mask where 1 indicates an
                  active feature, for use in L0 norm calculation.
        """
        threshold = torch.exp(self.log_threshold)
        return _JumpReLUWithSparsity.apply(x, threshold, self.bandwidth)
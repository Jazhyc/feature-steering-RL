import torch
from torch import nn

class _JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre_activations, threshold, bandwidth):
        output = pre_activations * (pre_activations > threshold).float()
        ctx.save_for_backward(pre_activations, threshold)
        ctx.bandwidth = bandwidth
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pre_activations, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # Compute ReLU
        grad_pre_activations = grad_output * (pre_activations > threshold).float()
        
        # Compute pseudo-derivative for the threshold (Eq. 12, pg. 25)
        rectangle_window = torch.abs(pre_activations - threshold) < (bandwidth / 2.0)
        pseudo_deriv_wrt_theta = -(threshold / bandwidth) * rectangle_window.float()
        grad_threshold = (grad_output * pseudo_deriv_wrt_theta).sum(dim=0)
        
        grad_bandwidth = None
        return grad_pre_activations, grad_threshold, grad_bandwidth

class JumpReLU(nn.Module):
    def __init__(self, num_features: int, initial_threshold: float = 0.001, bandwidth: float = 0.001):
        super().__init__()
        if initial_threshold <= 0:
            raise ValueError("Initial threshold must be positive.")
        self.log_threshold = nn.Parameter(torch.full((num_features,), torch.log(torch.tensor(initial_threshold))))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = torch.exp(self.log_threshold)
        return _JumpReLUFunction.apply(x, threshold, self.bandwidth)
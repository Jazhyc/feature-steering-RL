import torch
from fsrl.hooked_model import HookedModel
from fsrl.sae_adapter import SAEAdapter


class SAEfeatureAnalyzer:
    """
    This class takes in an SAEAdapter and can be used to inspect
    its features and the steering vector of the policy.
    """
    def __init__(self, sae_hooked_model: HookedModel):
        self.hooked_model = sae_hooked_model
        self.input = input
        return self
    
    def set_input(self, input: torch.Tensor):
        self.input = input
        return self

    def get_feature_description(self):
        return self

    def create_viz(self):
        return self


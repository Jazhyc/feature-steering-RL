from sae_lens import SAE, SAEConfig
from transformer_lens.hook_points import HookPoint

class SAEAdapter(SAE):
    
    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        
        # Use the explicit form of super() to avoid the metaclass conflict
        super(SAEAdapter, self).__init__(cfg, use_error_term=use_error_term)
        
        # Freeze all parameters currently in the SAE
        for param in self.parameters():
            param.requires_grad = False
        
        # Set up extra adapter hook
        self.hook_sae_adapter = HookPoint()
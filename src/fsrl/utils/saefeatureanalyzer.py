import torch
from fsrl.hooked_model import HookedModel
from fsrl.sae_adapter import SAEAdapter
from IPython.display import display, IFrame
import requests
import os
from neuronpedia.np_sae_feature import SAEFeature

class SAEfeatureAnalyzer:
    """
    This class takes in an SAEAdapter and can be used to inspect
    its features and the steering vector of the policy.
    Uses this libary to interact with Neuronpedia API.
    https://github.com/hijohnnylin/neuronpedia-python
    """
    def __init__(self, sae_hooked_model: HookedModel, api_key: str):
        self.hooked_model = sae_hooked_model
        self.input = None
        self.sae = self.hooked_model.sae_adapter
        self.model_id = self.sae.cfg.model_name
        
        # The name of the model is a bit different in Neuronpedia
        self.sae_id = self.sae.cfg.neuronpedia_id.split('/')[-1]
        
        self.html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
        
        os.environ["NEURONPEDIA_API_KEY"] = api_key   # Not sure if the request here require the API key

        # An SAE feature id is [MODEL_ID]@[SAE_ID]:[FEATURE_IDX]
        self.feature_info = {}
        self._collect_feature_labels()

    def _collect_feature_labels(self) -> None:
        for idx in range(self.sae.cfg.d_sae):
            self.feature_info[idx] = SAEFeature.get(self.model_id, self.sae_id, idx)
    
    def _get_dashboard_html(self, sae_release: str, sae_id: str, feature_idx: int):
        return self.html_template.format(sae_release, sae_id, feature_idx)
    
    def set_input(self, x: torch.Tensor):
        self.input = x
        return self

    def get_feature_page(self, feature_idx: int) -> IFrame:
        # for now get a random feature idx
        html = self._get_dashboard_html(
            sae_release=self.sae.cfg.model_name,
            sae_id=self.sae.cfg.sae_id,
            feature_idx=feature_idx
        )

        return IFrame(html, width=1200, height=600)

    def create_viz(self):
        return self


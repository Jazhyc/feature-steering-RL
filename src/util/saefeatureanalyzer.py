import torch
from fsrl.hooked_model import HookedModel
from fsrl.sae_adapter import SAEAdapter
from IPython.display import display, IFrame
import requests
import os
from neuronpedia.np_sae_feature import SAEFeature
"""
I think for this week, it would be great if you could

Quickly familiarize yourself with the codebase

Write some code to get the labels of the SAE features from Neuronpedia (The labels for the regular SAE features and the steering vector from our adapter will be identical since they have the same dimensionality)

Experiment with some visualizations from SAE Viz (https://github.com/callummcdougall/sae_vis/tree/main)
"""


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
        self.sae_id = self.sae.cfg.sae_id
        
        os.environ["NEURONPEDIA_API_KEY"] = api_key   # Not sure if the request here require the API key

        # An SAE feature id is [MODEL_ID]@[SAE_ID]:[FEATURE_IDX]
        self.feature_info = {}
        self._collect_feature_labels()
        self.html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
        print(self.feature_info)

    def _collect_feature_labels(self) -> None:
        # For some reason I keep getting error code 500
        # (https://www.reddit.com/r/CharacterAI/comments/15vjbsh/serious_question_what_is_the_500_server_error_and/)
        for idx in range(self.sae.cfg.d_sae):
            self.feature_info[idx] = SAEFeature.get(self.model_id, self.sae_id, str(idx))

    """
    def _compile_feature_labels(self) -> None:
        sae = self.hooked_model.sae_adapter
        for idx in range(sae.cfg.d_sae):
            print(f"https://www.neuronpedia.org/api/feature/{sae.cfg.model_name}/{sae.cfg.sae_id}/{idx}")
            response = requests.get(f"https://www.neuronpedia.org/api/feature/{sae.cfg.model_name}/{sae.cfg.sae_id}/{idx}")
            if response.status_code == 200:
                data = response.json()
                explanation = data.get("explanation") or data.get("label")
                self.feature_labels[idx] = explanation
                print("Explanation/Label:", explanation)
            else:
                print("[SAEfeatureAnalyzer] sae feature request failed with status code:", response.status_code)
    """
    
    def _get_dashboard_html(self, sae_release: str = "gpt2-small", sae_id: str = "7-res-jb", feature_idx: int = 0):
        return self.html_template.format(sae_release, sae_id, feature_idx)
    
    def set_input(self, x: torch.Tensor):
        self.input = x
        return self

    def get_feature_description(self):
        # for now get a random feature idx
        sae = self.hooked_model.sae_adapter
        feature_idx = torch.randint(0, sae.cfg.d_sae, (1,)).item()
        
        html = self._get_dashboard_html(
            sae_release=sae.cfg.model_name,
            sae_id=sae.cfg.sae_id,
            feature_idx=feature_idx
        )

        display(IFrame(html, width=1200, height=600))
        print(html)
        
        return self

    def create_viz(self):
        return self


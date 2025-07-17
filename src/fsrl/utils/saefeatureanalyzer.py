import torch
from fsrl.hooked_model import HookedModel
from fsrl.sae_adapter import SAEAdapter
from IPython.display import display, IFrame
import requests
import os
import pandas as pd
import tqdm

class SAEfeatureAnalyzer:
    """
    This class takes in an SAEAdapter and can be used to inspect
    its features and the steering vector of the policy.
    Uses this libary to interact with Neuronpedia API.
    https://github.com/hijohnnylin/neuronpedia-python
    Also some things from this notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb
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
        
        # This will currently take forever
        self._collect_feature_labels()

    def _collect_feature_labels(self) -> None:
        """
        Fetches all feature explanations in a single bulk request from the
        Neuronpedia API, which is much faster than fetching them one by one.
        """
        base_url = "https://www.neuronpedia.org/api/explanation/export"
        
        params = {
            "modelId": self.model_id,
            "saeId": self.sae_id
        }
        
        print(f"Fetching all explanations for {self.model_id}/{self.sae_id}...")
        
        try:
            response = requests.get(base_url, params=params)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            
            explanations_list = response.json()
            
            # The response is a list of explanation objects.
            # We convert it to a dictionary keyed by the feature index for fast lookup.
            for explanation in explanations_list:
                feature_idx = int(explanation["index"])
                # Store the entire explanation object, which includes the 'description' field.
                self.feature_info[feature_idx] = explanation

            print(f"Successfully loaded {len(self.feature_info)} feature explanations.")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching feature explanations: {e}")
            print("Response body:", e.response.text if e.response else "No response")
    
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

    @torch.no_grad()
    def get_feature_property_df(self, feature_sparsity: torch.Tensor) -> pd.DataFrame:
        """
        feature_property_df = get_feature_property_df(sae, log_feature_density.cpu())
        """
        sae = self.sae

        W_dec_normalized = (
            sae.W_dec.cpu()
        )  # / sparse_autoencoder.W_dec.cpu().norm(dim=-1, keepdim=True)
        W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T

        d_e_projection = (W_dec_normalized * W_enc_normalized).sum(-1)
        b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T

        return pd.DataFrame(
            {
                "log_feature_sparsity": feature_sparsity + 1e-10,
                "d_e_projection": d_e_projection,
                # "d_e_projection_normalized": d_e_projection_normalized,
                "b_enc": sae.b_enc.detach().cpu(),
                "b_dec_projection": b_dec_projection,
                "feature": list(range(sae.cfg.d_sae)),  # type: ignore
                "dead_neuron": (feature_sparsity < -9).cpu(),
            }
        )
    
    @torch.no_grad()
    def get_stats_df(self, projection: torch.Tensor) -> pd.DataFrame:
        """
        Returns a dataframe with the mean, std, skewness and kurtosis of the projection
        """
        mean = projection.mean(dim=1, keepdim=True)
        diffs = projection - mean
        var = (diffs**2).mean(dim=1, keepdim=True)
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0), dim=1)
        kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=1)

        return pd.DataFrame(
            {
                "feature": range(len(skews)),
                "mean": mean.numpy().squeeze(),
                "std": std.numpy().squeeze(),
                "skewness": skews.numpy(),
                "kurtosis": kurtosis.numpy(),
            }
        )

    # Temporarily commented out
    # @torch.no_grad()
    # def get_all_stats_dfs(self,
    #     gpt2_small_sparse_autoencoders: dict[str, SAE],  # [hook_point, sae]
    #     gpt2_small_sae_sparsities: dict[str, torch.Tensor],  # [hook_point, sae]
    #     model: HookedTransformer,
    #     cosine_sim: bool = False,
    # ):
    #     stats_dfs = []
    #     pbar = tqdm(gpt2_small_sparse_autoencoders.keys())
    #     for key in pbar:
    #         layer = int(key.split(".")[1])
    #         sparse_autoencoder = gpt2_small_sparse_autoencoders[key]
    #         pbar.set_description(f"Processing layer {sparse_autoencoder.cfg.hook_name}")
    #         W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(
    #             sparse_autoencoder.W_dec.cpu(), model, cosine_sim
    #         )
    #         log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
    #         W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
    #         W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
    #         stats_dfs.append(W_U_stats_df_dec)

    #     return pd.concat(stats_dfs, axis=0)

    # @torch.no_grad()
    # def get_W_U_W_dec_stats_df(self,
    #     W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False
    # ) -> tuple[pd.DataFrame, torch.Tensor]:
    #     W_U = model.W_U.detach().cpu()
    #     if cosine_sim:
    #         W_U = W_U / W_U.norm(dim=0, keepdim=True)
    #     dec_projection_onto_W_U = W_dec @ W_U
    #     W_U_stats_df = self.get_stats_df(dec_projection_onto_W_U)
    #     return W_U_stats_df, dec_projection_onto_W_U
from fsrl.hooked_model import HookedModel
from fsrl.sae_adapter import SAEAdapter
from IPython.display import IFrame
import requests
import os
import pandas as pd
import tqdm
import torch
import plotly_express as px
import numpy as np
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

class SAEfeatureAnalyzer:
    """
    This class takes in an SAEAdapter and can be used to inspect
    its features and the steering vector of the policy.
    Uses this libary to interact with Neuronpedia API.
    https://github.com/hijohnnylin/neuronpedia-python
    Also some things from this notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb
    """
    def __init__(self,
                 sae_hooked_model: HookedModel,
                 base_url: str = "https://www.neuronpedia.org/api/explanation/export"):
        """
        :param sae_hooked_model: a model with a SAE adapter.
        :param base_url: used for accessing the Neuronpedia API. Particularly
        for getting information on a given SAE features.
        """
        if "NEURONPEDIA_API_KEY" not in os.environ:
            raise EnvironmentError("Neuronpedia API key not found :(")
        # models
        self.hooked_model = sae_hooked_model
        self.sae = self.hooked_model.sae_adapter
        # model ids
        self.model_id = self.sae.cfg.model_name
        # The name of the model is a bit different in Neuronpedia
        self.sae_id = self.sae.cfg.neuronpedia_id.split('/')[-1]

        self.base_url = base_url
        self.html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

        # An SAE feature id is [MODEL_ID]@[SAE_ID]:[FEATURE_IDX]
        self.feature_info = {}
        self._collect_feature_labels()

        self.cached_feature_logit_distr = None

    def _collect_feature_labels(self) -> None:
        """
        Fetches all feature explanations in a single bulk request from the
        Neuronpedia API, which is much faster than fetching them one by one.
        Postcondition: the feature_info dict is set with feature_idx mapping
        to feature information e.g. feature_info[i]['description']
        """
        base_url = self.base_url
        
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

    def get_feature_page(self, feature_idx: int) -> IFrame:
        """
        Retrieves an SAE feature dashboard from Neuronpedia.
        :param feature_idx: index of the feature to retrieve.
        """
        # for now get a random feature idx
        html = self._get_dashboard_html(
            sae_release=self.sae.cfg.model_name,
            sae_id=self.sae_id,
            feature_idx=feature_idx
        )

        return IFrame(html, width=1200, height=600)

    def get_steered_features_info(self, steering_vector: torch.Tensor, threshold: float = 1e-3, top_k: int | None = None) -> pd.DataFrame:
        """
        Given a steering vector (torch.Tensor), return a DataFrame
        containing info about SAE features that are significantly steered.

        :param steering_vector: 1D tensor of length equal to number of SAE features.
        :param threshold: minimum absolute value for a feature to be considered "steered".
        :param top_k: if specified, return only the top_k features by absolute steering value.

        :return: pd.DataFrame with columns: feature_idx, steering_value, description, and other info if available.
        """
        if not isinstance(steering_vector, torch.Tensor):
            steering_vector = torch.tensor(steering_vector)

        # Ensure steering_vector is 1D and matches number of SAE features
        assert steering_vector.ndim == 1, "steering_vector must be 1D"
        assert steering_vector.shape[0] == self.sae.cfg.d_sae, (
            f"steering_vector length {steering_vector.shape[0]} does not match SAE features {self.sae.cfg.d_sae}"
        )

        # Find indices of features with steering magnitude above threshold
        steered_indices = (steering_vector.abs() > threshold).nonzero(as_tuple=True)[0]

        data = []
        for idx in steered_indices.tolist():
            feature_val = steering_vector[idx].item()
            info = self.feature_info.get(idx, {})
            description = info.get("description", "No description available")
            additional_info = {k: v for k, v in info.items() if k != "description"}
            data.append({
                "feature_idx": idx,
                "steering_value": feature_val,
                "description": description,
                **additional_info,
            })

        df = pd.DataFrame(data)
        # Sort by absolute steering value descending
        df = df.sort_values(by="steering_value", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

        # If top_k is specified, slice the top_k rows
        if top_k is not None:
            df = df.head(top_k)

        return df, self._plot_steering_value_distribution(df)
    
    def _plot_steering_value_distribution(self, df: pd.DataFrame):
        """
        Given a DataFrame with at least columns: 'steering_value', 'feature_idx', 'description',
        plot a detailed interactive distribution visualization.

        Args:
            df: DataFrame from get_steered_features_info with steering values and descriptions.
        """

        # Extract values
        steering_values = df["steering_value"].values
        feature_indices = df["feature_idx"].values
        descriptions = df.get("description", ["No description"] * len(df))

        # Colors by sign: positive = blue, negative = orange
        colors = np.where(steering_values >= 0, 'royalblue', 'orange')

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=steering_values,
            nbinsx=40,
            marker_color='lightgrey',
            name="Histogram",
            opacity=0.6,
            histnorm='probability density',  # normalize for overlaying KDE
        ))

        # Rug plot for individual points with hover info
        fig.add_trace(go.Scatter(
            x=steering_values,
            y=np.zeros_like(steering_values),
            mode='markers',
            marker=dict(color=colors, size=10, symbol='line-ns-open'),
            hovertemplate=(
                "Feature: %{customdata[0]}<br>"
                "Value: %{x:.5f}<br>"
                "Desc: %{customdata[1]}"
            ),
            customdata=np.stack((feature_indices, descriptions), axis=-1),
            showlegend=False,
        ))

        # KDE curve (smooth distribution)
        kde = gaussian_kde(steering_values)
        x_range = np.linspace(steering_values.min(), steering_values.max(), 500)
        kde_values = kde(x_range)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_values,
            mode='lines',
            name='KDE',
            line=dict(color='darkblue', width=3),
        ))

        # Layout tweaks
        fig.update_layout(
            title="Steering Vector Distribution (Histogram + KDE + Rug Plot)",
            xaxis_title="Steering Value",
            yaxis_title="Density",
            bargap=0.2,
            height=450,
            width=900,
            hovermode="closest",
        )

        return fig

    def plot_logit_distr_skewness(self, save_path: str | None = None) -> None:
        if self.cached_feature_logit_distr is None:
            self.logit_distr()

        fig = px.histogram(
            self.cached_feature_logit_distr,
            x="skewness",
            width=800,
            height=300,
            nbins=1000,
            title="Skewness of the Logit Weight Distributions",
        )

        if save_path:
            fig.write_image(save_path) if save_path.endswith(".png") else fig.write_html(save_path)
        else:
            fig.show()

    def plot_logit_distr_kurtosis(self, save_path: str | None = None) -> None:
        if self.cached_feature_logit_distr is None:
            self.logit_distr()

        fig = px.histogram(
            self.cached_feature_logit_distr,
            x=np.log10(self.cached_feature_logit_distr["kurtosis"]),
            width=800,
            height=300,
            nbins=1000,
            title="Kurtosis of the Logit Weight Distributions",
        )

        if save_path:
            fig.write_image(save_path) if save_path.endswith(".png") else fig.write_html(save_path)
        else:
            fig.show()

    def plot_logit_distr_skewness_vs_kurtosis(self, save_path: str | None = None) -> None:
        """
        See https://www.alignmentforum.org/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens
        for how to interpret this figure.
        """
        if self.cached_feature_logit_distr is None:
            self.logit_distr()

        fig = px.scatter(
            self.cached_feature_logit_distr,
            x="skewness",
            y="kurtosis",
            color="std",
            color_continuous_scale="Portland",
            hover_name="feature",
            width=800,
            height=500,
            log_y=True,
            labels={"x": "Skewness", "y": "Kurtosis", "color": "Standard Deviation"},
            title=f"Skewness vs Kurtosis of the Logit Weight Distributions",
        )

        fig.update_traces(marker=dict(size=3))

        if save_path:
            fig.write_image(save_path) if save_path.endswith(".png") else fig.write_html(save_path)
        else:
            fig.show()

    # The functions below are for calculation the logit weight distributions
    def logit_distr(self):
        """
        Computes relevant statistics of the logit weight distribution
        e.g., per feature mean, stdev, kurtosis, etc.
        """
        if self.cached_feature_logit_distr:
            return self.cached_feature_logit_distr
        W_dec = self.sae.W_dec.detach().cpu()
        # Calculate the approximate statistics of the logit weight distributions
        W_U_stats_df_dec, _ = self.get_W_U_W_dec_stats_df(
            W_dec, self.hooked_model.model, cosine_sim=False, use_batches=True, batch_size=1000
        )
        self.cached_feature_logit_distr = W_U_stats_df_dec
        return W_U_stats_df_dec

    # Helper functions below
    @torch.no_grad()
    def get_feature_property_df(self, feature_sparsity: torch.Tensor) -> pd.DataFrame:
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

    @torch.no_grad()
    def get_all_stats_dfs(
        self,
        gpt2_small_sparse_autoencoders: dict[str, SAEAdapter],
        gpt2_small_sae_sparsities: dict[str, torch.Tensor],
        model: HookedModel,
        cosine_sim: bool = False,
    ) -> pd.DataFrame:
        stats_dfs = []
        pbar = tqdm.tqdm(gpt2_small_sparse_autoencoders.keys())
        for key in pbar:
            layer = int(key.split(".")[1])
            sparse_autoencoder = gpt2_small_sparse_autoencoders[key]
            pbar.set_description(f"Processing layer {sparse_autoencoder.cfg.hook_name}")
            W_U_stats_df_dec, _ = self.get_W_U_W_dec_stats_df(
                sparse_autoencoder.W_dec.cpu(), model, cosine_sim
            )
            log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
            W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
            W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
            stats_dfs.append(W_U_stats_df_dec)

        return pd.concat(stats_dfs, axis=0)

    @torch.no_grad()
    def get_W_U_W_dec_stats_df(
            self,
            W_dec: torch.Tensor,
            model: HookedModel,
            cosine_sim: bool = False,
            use_batches: bool = True,
            batch_size: int = 1000
    ) -> tuple[pd.DataFrame, torch.Tensor]:
        W_U = model.W_U.detach().cpu()
        if cosine_sim:
            W_U = W_U / W_U.norm(dim=0, keepdim=True)

        if use_batches:
            # Approximate distribution by batching over W_U's vocab dimension
            num_batches = W_U.shape[1] // batch_size
            all_stats = []

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                W_U_batch = W_U[:, start:end]  # shape: [d_model, batch_size]
                projection = W_dec @ W_U_batch  # shape: [d_sae, batch_size]
                stats_df = self.get_stats_df(projection)
                all_stats.append(stats_df)

            # Average stats across batches
            W_U_stats_df = pd.concat(all_stats).groupby("feature").mean().reset_index()
            return W_U_stats_df, None

        else:
            # Full projection (expensive!)
            W_U = W_U.cpu()
            W_dec = W_dec.cpu()
            dec_projection_onto_W_U = W_dec @ W_U  # [d_sae, vocab_size]
            W_U_stats_df = self.get_stats_df(dec_projection_onto_W_U)
            return W_U_stats_df, dec_projection_onto_W_U

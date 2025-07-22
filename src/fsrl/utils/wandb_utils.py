"""
Utilities for downloading and organizing models from Weights & Biases.
"""

import os
import wandb
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_FAMILIES = [
    "Gemma2-2B", # Just one for now
]

class WandBModelDownloader:
    """
    Downloads and organizes trained models from Weights & Biases.
    """
    
    def __init__(self, entity: str, project: str, models_base_dir: Optional[str] = None):
        """
        Initialize the WandB model downloader.
        
        Args:
            entity: WandB entity name
            project: WandB project name
            models_base_dir: Base directory to store downloaded models (defaults to repo_root/models)
        """
        self.entity = entity
        self.project = project
        
        # Find the repository root (where pyproject.toml is located)
        current_dir = Path.cwd()
        repo_root = current_dir
        while repo_root != repo_root.parent:
            if (repo_root / "pyproject.toml").exists():
                break
            repo_root = repo_root.parent
        else:
            # Fallback: assume we're already in the repo root
            repo_root = current_dir
            
        if models_base_dir is None:
            self.models_base_dir = repo_root / "models"
        else:
            # If absolute path, use as-is; if relative, make it relative to repo root
            models_path = Path(models_base_dir)
            if models_path.is_absolute():
                self.models_base_dir = models_path
            else:
                self.models_base_dir = repo_root / models_path
                
        self.api = wandb.Api()
        
    def get_completed_runs(self) -> List[Any]:
        """
        Get all completed runs from the WandB project.
        
        Returns:
            List of completed wandb runs
        """
        runs = self.api.runs(f"{self.entity}/{self.project}")
        completed_runs = [run for run in runs if run.state == "finished"]
        logger.info(f"Found {len(completed_runs)} completed runs in {self.entity}/{self.project}")
        return completed_runs
    
    def get_run_artifacts(self, run: Any, artifact_type: str = "model") -> List[Any]:
        """
        Get artifacts of specified type from a run.
        
        Args:
            run: WandB run object
            artifact_type: Type of artifact to retrieve (default: "model")
            
        Returns:
            List of artifacts
        """
        artifacts = []
        try:
            for artifact in run.logged_artifacts():
                if artifact.type == artifact_type:
                    artifacts.append(artifact)
        except Exception as e:
            logger.warning(f"Could not retrieve artifacts for run {run.name}: {e}")
        
        return artifacts
    
    def is_model_downloaded(self, model_dir: str, run_name: str) -> bool:
        """
        Check if a model is already downloaded.
        
        Args:
            model_dir: Model directory name (e.g., "gemma2_2B-fsrl")
            run_name: WandB run name
            
        Returns:
            True if model is already downloaded, False otherwise
        """
        model_path = self.models_base_dir / model_dir / run_name
        if model_path.exists() and any(model_path.iterdir()):
            logger.info(f"Model already exists at {model_path}")
            return True
        return False
    
    def download_model(self, run: Any, model_dir: str, force_download: bool = False) -> Optional[Path]:
        """
        Download model artifacts from a specific run.
        
        Args:
            run: WandB run object
            model_dir: Model directory name (e.g., "gemma2_2B-fsrl")
            force_download: Whether to force re-download if model exists
            
        Returns:
            Path to downloaded model directory, or None if no artifacts found
        """
        # Check if already downloaded
        if not force_download and self.is_model_downloaded(model_dir, run.name):
            return self.models_base_dir / model_dir / run.name
        
        # Get model artifacts
        artifacts = self.get_run_artifacts(run)
        if not artifacts:
            logger.warning(f"No model artifacts found for run {run.name}")
            return None
        
        # Create directory structure
        download_path = self.models_base_dir / model_dir / run.name
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Download each artifact
        downloaded_files = []
        for artifact in artifacts:
            try:
                logger.info(f"Downloading artifact {artifact.name} from run {run.name}")
                artifact_dir = artifact.download(root=str(download_path))
                downloaded_files.append(artifact_dir)
                logger.info(f"Downloaded to {artifact_dir}")
            except Exception as e:
                logger.error(f"Failed to download artifact {artifact.name}: {e}")
        
        if downloaded_files:
            logger.info(f"Successfully downloaded {len(downloaded_files)} artifacts for run {run.name}")
            return download_path
        else:
            logger.warning(f"No artifacts were successfully downloaded for run {run.name}")
            return None
    
    def download_all_models(self, model_dir: str, force_download: bool = False) -> Dict[str, Optional[Path]]:
        """
        Download all models from completed runs.
        
        Args:
            model_dir: Model directory name (e.g., "gemma2_2B-fsrl")
            force_download: Whether to force re-download existing models
            
        Returns:
            Dictionary mapping run names to download paths
        """
        completed_runs = self.get_completed_runs()
        download_results = {}
        
        for run in completed_runs:
            logger.info(f"Processing run: {run.name} (state: {run.state})")
            download_path = self.download_model(run, model_dir, force_download)
            download_results[run.name] = download_path
        
        # Summary
        successful_downloads = sum(1 for path in download_results.values() if path is not None)
        logger.info(f"Downloaded {successful_downloads}/{len(completed_runs)} models")
        
        return download_results
    
    def list_downloaded_models(self, model_dir: str) -> List[str]:
        """
        List all downloaded models for a specific model directory.
        
        Args:
            model_dir: Model directory name
            
        Returns:
            List of run names that have been downloaded
        """
        model_path = self.models_base_dir / model_dir
        if not model_path.exists():
            return []
        
        downloaded = []
        for item in model_path.iterdir():
            if item.is_dir() and any(item.iterdir()):
                downloaded.append(item.name)
        
        return sorted(downloaded)


def download_model_family(family: str = "Gemma2-2B", force_download: bool = False) -> Dict[str, Optional[Path]]:
    """
    Convenience function to download model families.
    
    Args:
        family: Model family name (e.g., "Gemma2-2B")
        force_download: Whether to force re-download existing models
        
    Returns:
        Dictionary mapping run names to download paths
    """
    downloader = WandBModelDownloader(
        entity="feature-steering-RL",
        project=family
    )
    return downloader.download_all_models(family, force_download)


def download_all_families(force_download: bool = False) -> Dict[str, Dict[str, Optional[Path]]]:
    """
    Download all model families.
    
    Args:
        force_download: Whether to force re-download existing models
        
    Returns:
        Dictionary mapping family names to dictionaries of run names and download paths
    """
    all_results = {}
    for family in ALL_FAMILIES:
        logger.info(f"Downloading models for family: {family}")
        results = download_model_family(family, force_download)
        all_results[family] = results
    
    return all_results


def list_model_family(family: str = "Gemma2-2B") -> List[str]:
    """
    Convenience function to list downloaded model families.
    
    Returns:
        List of downloaded model run names
    """
    downloader = WandBModelDownloader(
        entity="feature-steering-RL", 
        project=family
    )
    return downloader.list_downloaded_models(family)
## Installation Instructions

To set up the `feature-steering-RL` project, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/Jazhyc/feature-steering-RL
cd feature-steering-RL
```

2. Install the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/):
```bash
sudo snap install astral-uv --classic
```

3. Install dependencies:
```bash
uv sync
```
4. To add more libraries, you can use:
```bash
uv add <package_name>
```

## Running the project

To run the project, you can use uv:
```bash
uv run python <script_name>
```

Alternatively, you can activate the virtual environment and run Python scripts directly:
```bash
source .venv/bin/activate
python <script_name>
```

## Training

The project includes a comprehensive training script that supports SimPO training with Sparse Autoencoders (SAEs).

### Configuration

The training script uses Hydra for configuration management with organized configuration files:

- `config/architecture/` - Model, SAE, and dataset configurations (e.g., `gpt2_default.yaml`)
- `config/training/` - Training parameters and hyperparameters (e.g., `simpo_default.yaml`,)
- `config/wandb.yaml` - Weights & Biases logging configuration
- `config/config.yaml` - Main configuration that combines components

### Environment Setup

Create a `.env` file in the project root to store environment variables. You can start by copying the example file:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual values:
```bash
# Example .env file
WANDB_API_KEY=your_wandb_api_key_here
# Add other environment variables as needed
```

### Running Training

The project supports two training modes:

#### SAE Adapter Training (Default)
Train with Sparse Autoencoder adapters:
```bash
uv run python src/fsrl/train.py
# or
uv run python src/fsrl/train.py --config-name=gpt2  # GPT-2 with SAE
deepspeed src/fsrl/train.py --config-name=gemma2_2B  # Gemma2-2B with SAE
deepspeed src/fsrl/train.py --config-name=gemma2_9B  # Gemma2-9B with SAE
```

#### Full Model Training (No SAE)
Train the complete model without SAE adapters:
```bash
uv run python src/fsrl/train.py --config-name=gpt2_full
deepspeed src/fsrl/train.py --config-name=gemma2_2B_full
```

### Loading Trained Models

#### Loading SAE Adapter Models
```python
from fsrl import HookedModel, SAEAdapter
from transformer_lens import HookedTransformer

# Load base model
model = HookedTransformer.from_pretrained("gpt2")

# Load trained SAE adapter
sae = SAEAdapter.from_pretrained("path/to/saved/adapter")

# Create hooked model
hooked_model = HookedModel(model, sae)
```

#### Loading Full Trained Models
```python
from fsrl import BaseHookedModel

# Load full trained model directly
model = BaseHookedModel.from_pretrained(
    "path/to/saved/model",
    device="cuda",
    dtype="bfloat16"
)
```

### Available Configurations

**Architecture configurations:**
- `gpt2_default` - GPT-2 small with SAE settings
- `gpt2_full` - GPT-2 small without SAE (full model training)
- `gemma2_2B` - Gemma2-2B with SAE settings  
- `gemma2_2B_full` - Gemma2-2B without SAE (full model training)

**Training configurations:**
- `gpt2_default` - Standard GPT-2 training settings
- `gpt2_full` - Full GPT-2 model training settings (same hyperparameters as SAE)
- `gemma2_2B` - Standard Gemma2-2B training settings
- `gemma2_2B_full` - Full Gemma2-2B model training settings (same hyperparameters as SAE)

### Training Features

- **Dual Training Modes**: Support for both SAE adapter training and full model training
- **SimPO training**: Optimized training with or without Sparse Autoencoder adapters
- **Model Loading**: Easy loading of trained models with `from_pretrained` methods
- **WandB logging**: Automatic experiment tracking with configurable project names
- **Hydra configuration**: Easy parameter management and experiment reproducibility
- **Environment variable support**: Secure configuration through .env files

### Notes to self
- export PATH="$PWD:$PATH"
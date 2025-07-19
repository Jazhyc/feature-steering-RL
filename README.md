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

To start training with default configuration:
```bash
uv run python src/fsrl/train.py
```

You can override any configuration parameter from the command line:
```bash
# Use a different premade config
python src/fsrl/train.py --config-name=gemma2_2B

# Use different architecture and training configs
uv run python src/fsrl/train.py architecture=gpt2_medium training=simpo_production

# Use test configuration for quick experiments
uv run python src/fsrl/train.py training=simpo_test

# Override specific parameters
uv run python src/fsrl/train.py training.batch_size=4 training.learning_rate=5e-5

# Limit dataset size for testing
uv run python src/fsrl/train.py architecture.dataset.sample_size=100
```

### Available Configurations

**Architecture configurations:**
- `gpt2_default` - GPT-2 small with default SAE settings

**Training configurations:**
- `simpo_default` - Standard training settings (100 epochs, batch size 2)

### Training Features

- **SimPO training**: Optimized training with Sparse Autoencoder adapters using LoRA
- **WandB logging**: Automatic experiment tracking with randomly generated run names
- **Hydra configuration**: Easy parameter management and experiment reproducibility
- **Environment variable support**: Secure configuration through .env files
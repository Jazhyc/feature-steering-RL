## Installation Instructions

To set up the `feature-steering-RL` project, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/Jazhyc/feature-steering-RL
cd feature-steering-RL
```

2. Install the [UV package manager](https://docs.astral.sh/uv/getting-started/installation/):

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
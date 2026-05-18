## LLMs Project Initialization

This repository contains small experiments building GPT-style components in PyTorch: attention, GPT blocks, tokenization, and data loading.

## Contents
- `GPT_architecture.py`: A minimal GPT-2–like architecture with generation.
- `Attention.py`: Scaled masked multi-head self-attention implementations.
- `sampling_data.py`: Dataset + DataLoader using `tiktoken` and sliding window.
- `tokenisasi.py`: Tokenization experiments with `tiktoken`.
- `shakespare.txt`: Sample corpus.
- `checking.ipynb`: Notebook used for quick checks.

## Quickstart

1) Create a virtual environment (Windows PowerShell):
```
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Run the scripts:
```
python GPT_architecture.py
python sampling_data.py
python tokenisasi.py
```

Notes:
- `torch` may require a specific build (CPU/CUDA). If installation fails, follow the official PyTorch install selector for your platform, then re-run `pip install tiktoken torchinfo numpy`.
- `tiktoken` uses the `gpt2` encoding locally (no API needed).

## Suggested Next Steps
- Add a training loop and loss (e.g., cross-entropy) wired to `GPT_Dummy_model`.
- Split modules under a package (e.g., `src/llms/`) for cleaner imports.
- Add simple tests for attention shapes and text generation.
- Parameterize configs and IO via a `config.yaml`.

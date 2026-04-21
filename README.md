# GUT-FORMer

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-BSD_3--Clause-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Ruff](https://img.shields.io/badge/code_style-ruff-purple)

GUT-FORMer is a Transformer-based model that jointly encodes microbiome taxonomy (which organisms are present) and functional pathways (what they are doing) into a shared latent space. It learns to reconstruct both modalities simultaneously, enabling cross-modal prediction and compact microbiome embeddings.

Developed by TomaszLab.

## 🚀 Setup

### 📋 Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) — Python version management
- [Poetry](https://python-poetry.org/) — Dependency management

Both can be installed via `brew install pyenv poetry`.

### ⚙️ Installation

```bash
pyenv install 3.12  # if not already installed
make install
```

## 📂 Data

Place input files in the `data/` directory:

```
data/
├── taxonomy_{dataset}.csv     # samples × species (pipe-separated taxonomy strings)
├── pathways_{dataset}.csv     # samples × pathways
└── metadata_{dataset}.csv     # sample metadata, must include study_condition column
```

All files are indexed by sample ID. The `{dataset}` name is passed via `--dataset` to all scripts.

A sample dataset (`--dataset sample`) is included in the repository to get started quickly.

## 🧠 Model

### Train from scratch

```bash
poetry run python src/gut_former/run_training.py
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `sample` | Dataset name |
| `--epochs` | `55` | Number of training epochs |
| `--embedding_dim` | `128` | Embedding dimension |
| `--latent_dim` | `64` | Latent space dimension |
| `--batch_size` | `16` | Batch size |
| `--learning_rate` | `1.89e-4` | Learning rate |

**Outputs** saved to `output/`:
- `{date}_{dataset}_checkpoint.pt` — model weights
- `{date}_{dataset}_training_stats.csv` — per-epoch metrics

### Fine-tuning

Continue training from an existing checkpoint (replace the checkpoint path with your own):

```bash
nohup poetry run python src/gut_former/run_training.py \
  --dataset my_dataset \
  --epochs 1000 \
  --checkpoint output/{date}_{dataset}_checkpoint.pt \
  > output/training.log 2>&1 &
```

Follow progress:

```bash
tail -f output/training.log
```

### Inference

```bash
poetry run python src/gut_former/run_inference.py \
  --dataset sample \
  --checkpoint output/20260412_sample_checkpoint.pt
```

**Outputs** saved to `output/`:
- `latent_{dataset}.csv` — latent embeddings
- `pred_taxonomy_{dataset}.csv` — predicted taxonomy
- `pred_pathways_{dataset}.csv` — predicted pathways

## 🧬 NMF

NMF decomposes microbiome data into latent signatures. Supports both pathway and taxonomy data, with optional filtering by health status.

```bash
poetry run python src/gut_former/nmf/run_nmf.py --type pathways --dataset sample
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--type` | required | Data type: `pathways` or `taxonomy` |
| `--dataset` | `sample` | Dataset name |
| `--n_signatures` | `8` | Number of NMF signatures |
| `--n_runs` | `100` | Runs per seed — best selected by validation metrics |
| `--filter` | `all` | Sample subset: `all`, `healthy`, `non-healthy` |

**Examples:**

```bash
# Pathway NMF, all samples
poetry run python src/gut_former/nmf/run_nmf.py --type pathways --dataset sample

# Taxonomy NMF, healthy samples only
poetry run python src/gut_former/nmf/run_nmf.py --type taxonomy --dataset sample --filter healthy
```

**Outputs** saved to `output/nmf/`:
- `H_{n}_{dataset}_{type}_{filter}.csv` — signature matrix
- `W_train_{n}_{dataset}_{type}_{filter}.csv` — sample loadings, train set
- `W_val_{n}_{dataset}_{type}_{filter}.csv` — sample loadings, val set
- `nmf_stats_{n}_{dataset}_{type}_{filter}.csv` — per-run metrics

## 🔍 biCV — Rank Selection

Bi-Cross-Validation selects the optimal number of NMF signatures for pathway data by evaluating reconstruction quality on held-out data blocks across a range of ranks. For taxonomy, the number of signatures is established in the literature.

```bash
poetry run python src/gut_former/nmf/run_bicv.py --dataset sample
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `sample` | Dataset name |
| `--n_signatures_min` | `2` | Minimum number of signatures to test |
| `--n_signatures_max` | `15` | Maximum number of signatures to test |
| `--n_runs` | `50` | NMF runs per fold per signature |

**Output** saved to `output/bicv/`:
- `bicv_nmf_stats.csv` — metrics for all folds, signatures, and runs

The optimal signature count is printed at the end of the run.

## 🔧 Troubleshooting

To start fresh (remove virtual environment and reinstall):

```bash
make clean
make install
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

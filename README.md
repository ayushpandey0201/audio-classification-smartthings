# Binary Kitchen Audio Classification using CNN14 and AST

![Samsung PRISM](https://img.shields.io/badge/Research-Samsung%20PRISM-1428A0)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)

## Overview

This repository contains the Member 3 contribution to a Samsung PRISM
industry-academia research project on kitchen audio understanding. It trains
CNN14 and Audio Spectrogram Transformer (AST) branches to classify a clip as
either **cooking** or **non-cooking**. The two models capture complementary
local convolutional and global transformer features, then contribute to an
AST-heavy weighted ensemble. The project targets a test F1 score above 0.95
and is designed to support later compression and lightweight deployment.

## Architecture

CNN14 is trained from scratch on log-mel spectrograms, while AST is initialized
from a Hugging Face checkpoint and fine-tuned in stages. Their logits are
combined before applying the binary decision threshold.

```text
                         +----------------+
                    +--->| CNN14 branch   |---+
                    |    +----------------+   |
Audio Input --------+                         +--> Weighted Ensemble --> Binary Output
                    |    +----------------+   |                         cooking / non-cooking
                    +--->| AST branch     |---+
                         +----------------+
```

## Dataset: kitpri_v2

`kitpri_v2` is a custom synthetic audio dataset containing approximately 3,508
usable mono WAV files sampled at 32 kHz. Samples were created by mixing
foreground and background audio at controlled SNR bands, using source material
from FSD50K, AudioSet, ESC-50, MUSAN, and Freesound.org. The binary label is
determined solely by the foreground class; pitch shifting and time stretching
provide a 2x augmentation expansion.

| Split | Samples |
|---|---:|
| Train | 2,428 |
| Validation | 531 |
| Test | 549 |
| **Total** | **3,508** |

The split is stratified at approximately 70/15/15. Download the dataset from
[Kaggle: ayushalia/kitpri-v2](https://www.kaggle.com/datasets/ayushalia/kitpri-v2).
Metadata CSV files contain paths relative to the dataset root.

## Model Details

### CNN14

- PANNs-style convolutional architecture implemented locally.
- Trained from scratch because `panns-inference` was unavailable in the Kaggle
  training environment.
- Accepts 128-bin log-mel spectrograms and uses six convolutional blocks,
  temporal-frequency pooling, and a binary classification head.

### AST

- Uses Hugging Face `ASTForAudioClassification`.
- Receives the same normalized log-mel representation, adapted to AST input
  dimensions by the local wrapper.
- Fine-tuned with staged unfreezing to retain pretrained audio representations
  while adapting the classifier and final transformer blocks.

## Training Strategy

Training is performed in two stages:

1. **Stage 1:** freeze both backbones and train only classification heads for
   8 epochs at a higher learning rate.
2. **Stage 2:** unfreeze CNN14 and the final four AST transformer blocks, then
   train for up to 40 epochs with a lower learning rate and cosine annealing.

The positive-class weight addresses the observed non-cooking recall bias.
Binary label smoothing is applied before `BCEWithLogitsLoss`.

| Hyperparameter | Value |
|---|---:|
| Sample rate | 32,000 Hz |
| Clip duration | 10 seconds |
| Mel bins | 128 |
| Batch size | 24 |
| Stage 1 learning rate | 3e-4 |
| Stage 2 learning rate | 5e-5 |
| Weight decay | 5e-3 |
| Label smoothing | 0.05 |
| Positive-class weight | 1.2 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Seed | 42 |

## Results

| Model | Val F1 | Notes |
|---|---:|---|
| CNN14 (from scratch) | ~0.80 | Individual |
| AST (staged fine-tune) | ~0.93 | Individual |
| CNN14+AST Ensemble (v2, full fine-tune) | 0.9420 | Overfit (Train F1=1.0) |
| CNN14+AST Ensemble (v4, staged) | In Progress | Target >0.95 |

Reported values summarize research experiments and are not reproduced by this
initial repository scaffold.

## Setup & Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ayushpandey0201/audio-classification-smartthings.git
cd audio-classification-smartthings
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the Kaggle API by placing your token at `~/.kaggle/kaggle.json`, then
restrict its permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
./scripts/prepare_dataset.sh
```

The script downloads and extracts `kitpri_v2` beneath `data/raw/`. Dataset
audio, metadata exports, and checkpoints are intentionally excluded from Git.

## Usage

Train the staged CNN14 + AST model:

```bash
./scripts/run_training.sh
```

Or invoke the module directly:

```bash
python -m src.train \
  --config configs/config.yaml \
  --train-csv data/raw/kitpri-v2/metadata/train.csv \
  --val-csv data/raw/kitpri-v2/metadata/val.csv \
  --data-root data/raw/kitpri-v2
```

Evaluate a saved checkpoint:

```bash
python -m src.evaluate \
  --config configs/config.yaml \
  --csv data/raw/kitpri-v2/metadata/test.csv \
  --data-root data/raw/kitpri-v2 \
  --checkpoint results/checkpoints/best.pt
```

Evaluate a custom CNN14/AST weighting:

```bash
python -m src.ensemble \
  --config configs/config.yaml \
  --csv data/raw/kitpri-v2/metadata/test.csv \
  --data-root data/raw/kitpri-v2 \
  --checkpoint results/checkpoints/best.pt \
  --cnn14-weight 0.35 \
  --ast-weight 0.65
```

Run `python -m src.train --help`, `python -m src.evaluate --help`, or
`python -m src.ensemble --help` for all options.

## Project Structure

```text
audio-classification-smartthings/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── metadata/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn14.py
│   │   └── ast_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   └── utils.py
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_cnn14.ipynb
│   ├── 03_training_ast.ipynb
│   └── 04_ensemble_eval.ipynb
├── scripts/
│   ├── prepare_dataset.sh
│   └── run_training.sh
└── results/
    ├── README.md
    └── .gitkeep
```

## Team & Context

This work is part of a four-member Samsung PRISM research collaboration:

| Member | Contribution |
|---|---|
| Member 1 | YAMNet |
| Member 2 | EfficientAT |
| **Member 3** | **CNN14 + AST comparison and ensemble contribution (this repository)** |
| Member 4 | CLAP + final ensemble |

The broader workflow includes pruning and quantization toward an approximately
60 MB RAM footprint, followed by deployment through a Telegram bot exposed via
an ngrok tunnel.

## License

This project is licensed under the MIT License. Dataset files retain the terms
of their respective upstream sources and are not distributed in this repository.

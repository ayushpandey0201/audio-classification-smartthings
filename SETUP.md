# Environment and Dataset Setup

This guide walks you through setting up the Python environment and downloading the `kitpri_v2` dataset required for training and evaluation.

## 1. Environment Setup

We recommend using a Python virtual environment (Python 3.10+).

```bash
# Clone the repository
git clone https://github.com/ayushalia/audio-classification-smartthings.git
cd audio-classification-smartthings

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and pre-commit hooks
make install
```

## 2. Dataset Download (Kaggle)

The `kitpri_v2` dataset is hosted on Kaggle. You will need a Kaggle account and an API token (`kaggle.json`).

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings) and click **Create New Token**.
2. Save the downloaded `kaggle.json` file to `~/.kaggle/kaggle.json`.
3. Set the correct permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

Download and extract the dataset using the provided Kaggle CLI:

```bash
# Create the target data directory
mkdir -p data/raw

# Download the dataset into the directory
kaggle datasets download -d ayushalia/kitpri-v2 -p data/raw --unzip
```

The `data/raw/kitpri-v2` folder should now contain:
- `audio/` (containing 3,500+ .wav files)
- `metadata/` (containing `train.csv`, `val.csv`, and `test.csv`)

## 3. Verify Setup

Run the test suite to ensure your environment is configured correctly:

```bash
make test
```

If all tests pass, you are ready to start training or evaluating the model!

# Kitchen Audio Classification for Smart Kitchen IoT

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Samsung PRISM](https://img.shields.io/badge/Research-Samsung_PRISM-blueviolet.svg)

> **Elevator Pitch:** A highly optimized, dual-branch ensemble model (DeiT-Small + CNN14) designed to detect cooking sound events in real-time, built specifically for deployment on memory-constrained edge hardware like Samsung SmartThings hubs.

<p align="center">
  <img src="demo/demo.gif" alt="Telegram Bot Demo" width="300"/>
</p>

---

## 🍳 Why This Matters

Smart kitchen appliances and IoT hubs (like Samsung SmartThings) need to understand ambient audio context to trigger proactive automations. For example, if the hub hears sizzling or chopping, it can automatically turn on the exhaust fan, adjust the ambient music volume, or alert the user if cooking is left unattended. 

This project solves the challenge of performing high-accuracy acoustic scene classification on **edge devices** with strict memory footprints (< 60 MB RAM target), avoiding the latency and privacy concerns of cloud-based audio processing.

---

## ⚡ Demo: Telegram Bot

We built a live Telegram bot that accepts audio clips and returns real-time predictions. 

**Try it live right now!**
- 🥇 **Primary Bot:** [@kitpribot](https://t.me/kitpribot)
- 🥈 **Secondary Bot:** [@binary_cooking_bot](https://t.me/binary_cooking_bot)

*(Just click a link above, press start, and send it an audio file or voice message!)*

**Need some audio to test with? Download these samples:**
- 🍳 [Download Cooking Sample (Frying)](demo/cooking_frying.wav)
- 🗣️ [Download Non-Cooking Sample (Speech)](demo/noncooking_speech.wav)

<p align="center">
  <img src="demo/demo.gif" alt="Telegram Bot Demo" width="300"/>
</p>

### How to run the bot:
1. Ensure your virtual environment is active (`source .venv/bin/activate`).
2. Run the bot from your terminal:
   ```bash
   make bot
   ```
3. Open Telegram and search for your bot (or click the `t.me/...` link provided by BotFather).
4. **Send an audio file** or **record a voice message** directly in the chat.
5. The bot will instantly reply with the classification:

```text
User  → [sends .wav / .m4a / voice message]
Bot   → 🍳 Cooking (confidence: 94.2%)
```

See the [`telegram_bot/README.md`](telegram_bot/README.md) for more details on architecture and adding your bot token. The [`demo/`](demo/) folder contains example audio files you can test with.

---

## 📊 Results Summary

The current state-of-the-art model is a weighted ensemble of an Audio Spectrogram Transformer (AST) with a DeiT-Small backbone and a CNN14 local feature extractor.

| Metric | Score | Notes |
|---|---|---|
| **Test F1** | **0.9275** | Approaching the edge-deployment target of >0.95 |
| **Test AUC** | 0.9573 | Excellent separability between classes |
| **Validation F1** | 0.9420 | Best checkpoint evaluated on hold-out validation set |

*Note: These results reflect the `kitpri_v2` dataset (3,500+ synthesised kitchen audio scenes). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for details on the dataset methodology and architecture choices.*

---

## 🛠️ Reproducibility & Setup

We have made it as simple as possible to reproduce our Kaggle training runs locally.

### 1. Environment and Dataset
Follow the step-by-step instructions in [`SETUP.md`](SETUP.md) to initialize your virtual environment and download the dataset from Kaggle.

### 2. Training
Trigger the two-stage training pipeline (warm-up + full fine-tuning) with a single command:
```bash
make train
```

### 3. Evaluation
Evaluate the best checkpoint against the test set:
```bash
make evaluate
```

### 4. Try it Yourself!
Test the model instantly on a single audio file:
```bash
python -m examples.predict demo/cooking_frying.wav
# Expected: 🍳 Cooking (confidence: > 90%)
```

---

## 📂 Project Structure

```text
audio-classification-smartthings/
├── telegram_bot/        # Telegram bot inference and deployment code
├── configs/             # YAML configurations (hyperparameters)
├── data/                # Dataset (downloaded via Kaggle)
├── demo/                # Sample audio files and demonstration assets
├── examples/            # Minimal inference scripts
├── notebooks/           # Research notebooks, EDA, and Kaggle training references
├── results/             # Saved checkpoints and metrics
├── src/                 # Core model architecture, datasets, and training loops
├── ARCHITECTURE.md      # Detailed explanation of design decisions
├── SETUP.md             # Environment and dataset setup instructions
├── CONTRIBUTING.md      # Guidelines for contributing
└── Makefile             # Command orchestration
```

---

## 🚧 Limitations & Future Work

While the current model achieves strong results, we are actively working on the following:

- **Overfitting Gap:** The model achieves 1.000 F1 on the training set but 0.9275 on the test set. We are exploring heavier SpecAugment and Mixup configurations (v6 run) to close this generalization gap.
- **Model Compression:** While DeiT-Small is lightweight (~22M parameters), it still exceeds our < 60 MB RAM target in float32. We plan to implement int8 quantization and magnitude pruning.
- **Knowledge Distillation:** We are exploring using the current ensemble as a "Teacher" to train a much smaller, mobile-friendly "Student" CNN for edge deployment.
- **On-Device Testing:** Actual deployment and latency testing on the Samsung SmartThings hub hardware is pending.

---

## 📜 Citation

```text
Samsung PRISM Research Internship, 2024–2025
Kitchen Audio Classification for Smart Kitchen IoT Devices
```

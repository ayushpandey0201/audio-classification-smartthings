# Project Architecture

This document explains the key design decisions behind the dual-branch ensemble model and the overall pipeline architecture for the kitchen audio classification project.

## 1. Why AST + DeiT-Small?

We chose an **Audio Spectrogram Transformer (AST)** built on top of a **DeiT-Small** (Data-efficient Image Transformers) backbone due to its excellent parameter-to-performance ratio:

- **Constraint:** The model is targeted for edge deployment on Samsung SmartThings hubs, which have strict memory constraints (target < 60 MB RAM footprint).
- **DeiT-Small advantage:** With ~22 million parameters, DeiT-Small fits comfortably within edge constraints after int8 quantization, unlike base/large ViT models. By initializing from ImageNet weights, the model learns strong representational priors even on our relatively small dataset of 3,500 clips.
- **Why Transformers?** Sound events in a kitchen (like sizzling or a microwave beep) often have complex temporal relationships with background noise. Transformers, via self-attention mechanisms, capture these global contexts better than purely convolutional architectures.

## 2. Why CNN14 is included in the Ensemble

While the AST excels at global context, we retain a **CNN14** architecture in a weighted ensemble:

- **Local Feature Extraction:** CNNs are naturally adept at capturing local, shift-invariant features (e.g., the sharp transient of a knife chop).
- **PANNs Inspiration:** Our CNN14 implementation is inspired by PANNs (Pre-trained Audio Neural Networks) but trained from scratch on log-mel spectrograms. It uses an AdaptiveAvgPool2d head to ensure robustness to variable-length inputs.
- **Ensemble Synergy:** The AST and CNN14 make different types of errors. The AST captures the "overall scene" (global attention), while the CNN14 focuses on local acoustic patterns. A weighted combination (CNN14: 0.45, AST: 0.55) yields superior recall and F1 compared to either model alone.

## 3. Discarding `kitpri_v1` for `v2`

The first iteration of the dataset (`kitpri_v1`) was discarded in favor of a complete rebuild (`kitpri_v2`):

- **Signal-to-Noise Ratio (SNR):** `v1` suffered from unrealistic background mixing, where foreground cooking sounds were often overpowered or unnaturally blended with ambient noise.
- **Synthetic Scene Generation:** For `v2`, we developed a robust synthetic generation pipeline (detailed in `notebooks/05_dataset_creation.ipynb`) that layers 1-2 foreground events over authentic background noise (sourced from Freesound, AudioSet, etc.) with strict dB controls, resulting in a much more challenging and realistic dataset for edge device conditions.

## 4. Staged vs. Single-Phase Training

We implement a **Two-Stage Unfreeze** training strategy for the AST:

- **Stage 1 (Warm-up):** We freeze the DeiT backbone and only train the newly initialized classification head. This prevents the large, random gradients of the untrained head from destroying the valuable ImageNet pretrained weights in the backbone.
- **Stage 2 (Fine-tuning):** We unfreeze the last $N$ transformer blocks (e.g., the last 4 blocks) and train with a lower learning rate.
- **Trade-off:** Staged training takes slightly more boilerplate code (implemented in `src/train.py` via `configure_stage`), but significantly reduces overfitting and stabilizes early training dynamics compared to a single-phase full fine-tune.

# Experiment Results

This directory is reserved for generated experiment artifacts:

- `checkpoints/`: PyTorch model checkpoints (`.pt`, `.pth`, `.ckpt`)
- `metrics/`: per-epoch and final metric JSON files
- `plots/`: confusion matrices, learning curves, and comparison figures

Generated outputs are ignored by Git. Keep only lightweight, intentionally
curated summaries under version control when they are needed for a report.

## Recorded Experiment History

| Run | Train F1 | Val F1 | Test F1 | Test AUC | Notes |
|---|---:|---:|---:|---:|---|
| v2 | 1.0000 | 0.9420 | 0.9275 | 0.9573 | Full fine-tune; overfit |
| v3 | ~0.82 | ~0.80 | 0.7937 | Not recorded | Frozen-backbone underfit |
| v4/v5 | ~0.91 | ~0.87 | 0.8426 | 0.9118 | Early staged-unfreeze experiments |
| v6 | In progress | In progress | In progress | In progress | Regularized staged training |

Every new run should preserve its resolved configuration, random seed,
checkpoint, threshold-selection source, and final metric JSON. Thresholds must
be selected on validation data, never on the test set.

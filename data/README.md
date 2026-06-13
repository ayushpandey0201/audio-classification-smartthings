# Dataset Layout

This project uses
[kitpri_v2](https://www.kaggle.com/datasets/ayushalia/kitpri-v2), a synthetic
binary kitchen-audio dataset containing approximately 3,508 usable 32 kHz mono
WAV files.

## Construction

Foreground clips labelled as cooking or non-cooking were mixed with background
audio at controlled signal-to-noise ratios. Source collections include FSD50K,
AudioSet, ESC-50, MUSAN, and Freesound.org. The foreground class alone determines
the final target. Pitch-shift and time-stretch augmentations provide an
approximately 2x expansion.

## Class Distribution

| Class | Clips |
|---|---:|
| Cooking | 1,784 |
| Non-cooking | 1,724 |
| **Total** | **3,508** |

These counts describe the disk-verified usable subset rather than the nominal
size of earlier generation runs.

## Expected Local Structure

Run `../scripts/prepare_dataset.sh` from the repository root. The exact archive
layout may vary, but training commands expect metadata CSV paths and an audio
root resembling:

```text
data/
├── README.md
├── metadata/              # tracked placeholder only
└── raw/                   # ignored by Git
    └── kitpri-v2/
        ├── metadata/
        │   ├── train.csv
        │   ├── val.csv
        │   └── test.csv
        └── audio/
            └── ...
```

## Metadata Contract

| Column | Required | Description |
|---|---|---|
| `file_path` | Yes | Path relative to the configured audio root |
| `label` | Yes | `0` for non-cooking, `1` for cooking |
| `split` | Recommended | `train`, `val`, or `test` |
| `clip_id` | Optional | Stable clip identifier |
| `rms_db` | Optional | Clip RMS level in decibels |
| `is_augmented` | Optional | Whether the row represents an augmented clip |

The loader also accepts common aliases such as `path`, `filepath`, `file`, and
`audio_path`. Absolute paths and paths that escape the configured dataset root
are rejected to keep experiments portable and safe.

## Dataset Lineage

`kitpri_v1` was discarded after label-quality issues produced misleadingly high
scores. `kitpri_v2` was rebuilt and disk-verified; all current experiments must
record the dataset version and metadata split used.

Do not commit raw audio, downloaded metadata, or generated dataset artifacts.

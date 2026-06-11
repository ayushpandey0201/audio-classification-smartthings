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

Metadata should contain a relative audio path column such as `path`,
`filepath`, or `file`, plus a `label` column. Labels may be numeric (`0`/`1`)
or strings (`non-cooking`/`cooking`).

Do not commit raw audio, downloaded metadata, or generated dataset artifacts.

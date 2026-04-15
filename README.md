# VLA Prototype

Simple Vision-Language-Action prototype for robot imitation learning.

This repository started as a small VLA baseline for a local robot dataset and was later extended to fine-tune and evaluate on LIBERO-style data.

## Overview

Current model inputs:

- `image1`
- `image2`
- `language instruction`
- `state`

Current model output:

- action vector
  - local dataset: `6-dim`
  - LIBERO fine-tuning: `7-dim`

Main components:

- `VisionEncoder`: ViT-B/16 backbone
- `LanguageEncoder`: DistilBERT with local cache fallback
- `StateEncoder`: small MLP
- `Fusion`: feature concatenation
- `Action Head`: MLP policy head

## Repository Structure

```text
VLA/
├── configs/
│   └── base.yaml
├── data/
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── fusion.py
│   ├── language_encoder.py
│   ├── policy.py
│   ├── state.py
│   └── vision_encoder.py
├── train/
│   ├── loss.py
│   └── train.py
├── checkpoints/
├── eval_logs/
├── external/
├── fine-tuning.py
├── generic_finetune.py
├── evaluate_libero.py
└── WORKLOG.md
```

## Datasets

### 1. Local robot dataset

- source: `boxbrown/mydataset`
- local path: `external/boxbrown-mydataset`
- format: LeRobot v3-style dataset
- used episodes: `270`
  - episode `0` and `271` excluded

This dataset is loaded through a custom loader in [data/dataset.py](/home/user/background/project/VLA/data/dataset.py) instead of directly depending on `lerobot`.

### 2. LIBERO fine-tuning dataset

- dataset id: `HuggingFaceVLA/libero`
- used in [fine-tuning.py](/home/user/background/project/VLA/fine-tuning.py)

The LIBERO fine-tuning setup uses:

- `observation.images.image`
- `observation.images.image2`
- `observation.state`
- `action`
- `task_index`

`task_index` is mapped to the actual `libero_10` language instruction during fine-tuning.

## Model Variants

### Local VLA baseline

Used for the local robot dataset:

- image + text + state
- action dimension: `6`

Main training entry:

- [train/train.py](/home/user/background/project/VLA/train/train.py)

### LIBERO fine-tuning model

Used for LIBERO-style fine-tuning:

- image1 + image2 + text + state
- action dimension: `7`

Main entry:

- [fine-tuning.py](/home/user/background/project/VLA/fine-tuning.py)

During LIBERO fine-tuning:

- pretrained weights are loaded for:
  - `vision_encoder`
  - `language_encoder`
- newly initialized modules are:
  - `state_encoder`
  - `action_head`

If `--freeze-backbone` is used, both vision and language backbones are frozen.

## Training

### Local dataset training

```bash
conda run -n vla python -m train.train
```

Notes:

- episode-based train/val split is used
- current loss: `MSELoss`
- checkpoints are saved under `checkpoints/`

### LIBERO fine-tuning

```bash
conda run -n vla python fine-tuning.py
```

Example with frozen backbone:

```bash
conda run -n vla python fine-tuning.py --freeze-backbone
```

Current LIBERO fine-tuning setup:

- dataset: `HuggingFaceVLA/libero`
- suite assumption: `libero_10`
- loss: `MSELoss`
- policy output: `7-dim` action

## Evaluation on LIBERO

Evaluation is done with a custom rollout script instead of `lerobot-eval`.

Main entry:

- [evaluate_libero.py](/home/user/background/project/VLA/evaluate_libero.py)

Example:

```bash
conda run -n vla_libero python evaluate_libero.py \
  --checkpoint-path checkpoints/libero_finetune/libero_epoch_5.pt \
  --suite-name libero_10 \
  --n-episodes 10 \
  --save-video
```

This script:

- loads a `.pt` checkpoint directly
- creates LIBERO tasks and environments
- uses `task.language` as text input
- rolls out the policy in the simulator
- saves:
  - task-wise success rates
  - JSON summary
  - optional rollout videos

Output examples:

- JSON: `eval_logs/libero_eval.json`
- videos: `eval_logs/videos/`

## Cached Pretrained Models

The repository uses local cache directories:

- `.cache/torch`
- `.cache/huggingface`

This is used to avoid repeated downloads and to make fallback behavior easier when the network is unstable.

## Experimental Notes

- DistilBERT pretrained loading was verified and is currently used in the main setup.
- LIBERO evaluation currently runs with a custom script and direct checkpoint loading.
- The project is still a prototype and has not yet been turned into a standardized policy package.
- Current fusion is simple concatenation.

## Known Limitations

- The project is still baseline-oriented rather than benchmark-complete.
- LIBERO evaluation depends on local LIBERO installation and environment setup.
- The current policy is simple action regression with `MSELoss`.
- Performance on LIBERO is still under active debugging and iteration.

## Useful Files

- [models/policy.py](/home/user/background/project/VLA/models/policy.py)
- [models/language_encoder.py](/home/user/background/project/VLA/models/language_encoder.py)
- [models/vision_encoder.py](/home/user/background/project/VLA/models/vision_encoder.py)
- [fine-tuning.py](/home/user/background/project/VLA/fine-tuning.py)
- [generic_finetune.py](/home/user/background/project/VLA/generic_finetune.py)
- [evaluate_libero.py](/home/user/background/project/VLA/evaluate_libero.py)
- [WORKLOG.md](/home/user/background/project/VLA/WORKLOG.md)

## Acknowledgement

This repository uses:

- Hugging Face datasets
- DistilBERT
- torchvision ViT
- LIBERO
- PyTorch

# Industrial Tube Inner-Wall Defect Segmentation

Research-stage implementation of pixel-level defect segmentation for industrial endoscopic inspection.

**Researcher:** Naomi Imali Okanda

**Affiliation:** Tsinghua University, Shenzhen International Graduate School

## Research Objective

Investigate the trade-off between pixel-level segmentation accuracy, small-defect sensitivity, and false-positive predictions using ResNet34 U-Net.

## Methods

- Baseline U-Net trained from scratch.
- Targeted fine-tuning using defect-focused and clean-negative training patches.
- Pixel-level and connected-component evaluation.
- Connected-component filtering for false-positive analysis.

## Validation Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Pixel AUROC | 0.97048 | 0.96269 |
| Pixel AP | 0.79944 | 0.77727 |
| Pixel F1 | 0.71074 | 0.62978 |
| Pixel recall | 0.90011 | 0.90015 |
| Pixel FPR | 0.05593 | 0.08472 |
| Small-component detection | 5/38 (13.16%) | 29/38 (76.32%) |

Results are from 129 validation crops, with hole-disjoint development, validation, and test splits. The evaluated crops are defect-positive; pixel FPR measures background pixels within these crops, not the false-alarm rate on entirely normal images.

## Repository Contents

- `src/training/`: baseline training and targeted fine-tuning.
- `src/evaluation/`: pixel metrics and defect-component analysis.
- `src/postprocessing/`: component filtering.
- `docs/`: methodology, results, and limitations.
- `results/validation_summary.json`: aggregate validation results.
- `requirements.txt`: Python dependency specification.

## Reproducibility

The original experiments used Python 3.12, CUDA 12.8, and an NVIDIA-customized PyTorch build. Install a compatible PyTorch and torchvision pair before installing the remaining requirements in a virtual environment.

The scripts retain some experiment-specific manifest, checkpoint, and directory assumptions. End-to-end reproduction on a fresh environment has not been verified.

## Limitations

The baseline achieved stronger pixel-level F1 and lower FPR, while targeted fine-tuning improved small-component detection at the cost of additional false-positive regions. These are validation results; an untouched final test of the fine-tuned model is not reported.

## Data Availability

Industrial inspection images, annotations, and model checkpoints are excluded. Code and aggregate results remain subject to laboratory release authorization.

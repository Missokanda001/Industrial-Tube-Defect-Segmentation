# Reproducibility Guide

This repository contains research-stage EXP009D implementations, not a verified end-to-end reproduction package.

## Environment
Python 3.12, CUDA 12.8, and segmentation-models-pytorch 0.5.0 were used in the original research environment.

## Required Inputs
Authorized image and mask files, development_manifest.csv, validation_manifest.csv, the baseline and fine-tuned checkpoints, missed_defect_patches.csv, and pure_background_hard_negatives.csv.

## Workflow
1. Train the baseline U-Net.
2. Generate targeted mining and clean-negative sampling inputs.
3. Fine-tune the baseline.
4. Evaluate pixel and component metrics.
5. Analyze connected-component filtering.

## Limitations
Intermediate data-generation scripts and confidential industrial data are not included. The fine-tuned model has no reported untouched final test evaluation. Code and results require laboratory release authorization.

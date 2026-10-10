# Results and Discussion

## Validation results

Evaluation: 129 validation images. All results are validation-only.

| Metric | Original U-Net | Fine-tuned U-Net |
|---|---:|---:|
| Pixel AUROC (approx.) | 0.97048 | 0.96269 |
| Pixel AP (approx.) | 0.79944 | 0.77727 |
| Pixel accuracy | 0.94050 | 0.91405 |
| Pixel recall | 0.90011 | 0.90015 |
| Pixel FPR | 0.05593 | 0.08472 |
| Pixel F1 | 0.71074 | 0.62978 |

## Interpretation

The original U-Net provides stronger aggregate pixel-level segmentation performance. Fine-tuning improves small-component sensitivity in the internal component-level audit but increases false-positive predictions. These configurations represent different operating priorities.

## Important limitations

Thresholds were calibrated on validation data. Pixel FPR refers to background pixels within defect-positive crops, not a normal-image false-alarm rate. The original model did not meet the 90% recall requirement on the previously examined test split. No independent final test success is claimed for the fine-tuned model.

Industrial images, masks, annotations, and checkpoints are excluded from public release.

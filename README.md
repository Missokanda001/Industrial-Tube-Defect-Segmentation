# Industrial-Tube-Defect-Segmentation
Research on pixel-level segmentation and small-defect detection in industrial tube inner-wall images using U-Net, targeted fine-tuning, and false-positive suppression.


# EXP009D: Small-Defect-Sensitive Industrial Tube Inspection

Research code and evaluation documentation for a ResNet34 U-Net study of binary pixel-level defect segmentation in industrial tube inner-wall imagery.

> **Status:** Internal research evaluation. Public dataset, figures, and model checkpoints are **not included** pending supervisor/lab approval. The validation results are not deployment guarantees.

## Key finding

Targeted fine-tuning increased annotated small-component detection recall from **13.16% (5/38)** to **76.32% (29/38)** on the 129-image validation split, while increasing false-positive regions. The original U-Net remains stronger on pixel F1 (71.07% vs 62.98%). Both meet five specified numerical criteria on validation, but not all criteria were confirmed on the previously inspected original-model test.

## Evaluation summary (validation)

| Metric | Original U-Net | Clean-negative fine-tuned U-Net |
|---|---:|---:|
| Approx. pixel AUROC | 0.97048 | 0.96269 |
| Approx. pixel AP | 0.79944 | 0.77727 |
| Pixel accuracy | 0.94050 | 0.91405 |
| Pixel recall | 0.90011 | 0.90015 |
| Pixel FPR | 0.05593 | 0.08472 |
| Pixel F1 | 0.71074 | 0.62978 |
| Component recall (≥10% overlap) | 0.6772 | 0.8987 |

## Reproduction

Scripts and configuration files are included in the **internal reproducibility package**. Paths in legacy scripts may reference the original lab server and require configuration before use elsewhere. The dataset is not distributed here; request access through the responsible research group. The training/evaluation pipeline uses Python, PyTorch, OpenCV, and segmentation-models-pytorch.

## Project structure

- `report/`: Technical report and verified metric tables
- `code/`: Curated experiment and evaluation scripts (only after path/security review)
- `results/`: Non-sensitive aggregate CSV/JSON summaries
- `figures/`: Approved figures only
- `environment/`: Dependency and hardware notes
- `data/`: **Not committed**
- `checkpoints/`: **Not committed unless explicitly approved**

## Important limitations

Thresholds were calibrated on validation. False-positive rate is measured over background pixels in defect-positive crops. The original model's earlier test recall was **0.89818**, below the 0.90 criterion; the improved model has no newly independent final test claim. Model selection, data access, and industrial-image release constraints apply.


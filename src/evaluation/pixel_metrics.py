import os

from pathlib import Path

import json



import cv2

import numpy as np

import pandas as pd

import torch

import segmentation_models_pytorch as smp



ROOT = Path(os.environ.get("TUBE_PROJECT_ROOT", ".")).expanduser().resolve()

DATA = Path(os.environ.get("TUBE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

BASE = Path(os.environ.get("TUBE_EXPERIMENT_DIR", str(ROOT / "experiments"))).expanduser().resolve()

OUT = BASE / "EXP009D-9B_final_metrics"

VIS = OUT / "visualizations"

VIS.mkdir(parents=True, exist_ok=True)



CHECKPOINT = BASE / "EXP009D-8B_clean_negative_training/epoch_06.pth"

THRESHOLD = 0.01727294921875

SIZE = 512

BINS = 16384



device = "cuda" if torch.cuda.is_available() else "cpu"



model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1

).to(device)



checkpoint = torch.load(

    CHECKPOINT, map_location="cpu", weights_only=True

)

model.load_state_dict(checkpoint["model"])

model.eval()



df = pd.read_csv(DATA / "validation_manifest.csv")



assert len(df) == 129

assert set(df["speed_mm_s"].astype(int)) == {50}



pos_hist = np.zeros(BINS, dtype=np.int64)

neg_hist = np.zeros(BINS, dtype=np.int64)



rows = []



def color_heatmap(prob):

    values = np.clip(prob * 255, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(values, cv2.COLORMAP_JET)



def save_visualization(name, image, gt, prob, pred):

    h, w = image.shape[:2]



    original = image.copy()



    gt_panel = image.copy()

    gt_panel[gt] = (

        0.35 * gt_panel[gt] +

        0.65 * np.array([0, 255, 0])

    ).astype(np.uint8)



    heat = color_heatmap(prob)



    pred_panel = image.copy()

    pred_panel[pred] = (

        0.35 * pred_panel[pred] +

        0.65 * np.array([0, 0, 255])

    ).astype(np.uint8)



    panels = [original, gt_panel, heat, pred_panel]

    titles = [

        "Original",

        "Ground Truth",

        "U-Net Probability",

        "Prediction"

    ]



    target_w = 480

    target_h = max(1, round(h * target_w / w))



    labeled = []



    for panel, title in zip(panels, titles):

        panel = cv2.resize(

            panel, (target_w, target_h),

            interpolation=cv2.INTER_AREA

        )



        canvas = np.full(

            (target_h + 40, target_w, 3),

            255, dtype=np.uint8

        )



        canvas[40:] = panel



        cv2.putText(

            canvas, title, (12, 27),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7, (0, 0, 0), 2

        )



        labeled.append(canvas)



    combined = np.hstack(labeled)

    cv2.imwrite(str(VIS / f"{Path(name).stem}_comparison.jpg"), combined)





print("=== EXP009D-5D ===", flush=True)

print("Checkpoint epoch:", checkpoint["epoch"], flush=True)

print("Validation images:", len(df), flush=True)

print("Fixed threshold:", THRESHOLD, flush=True)



with torch.inference_mode():



    for index, row in df.iterrows():



        name = Path(str(row["image_name"])).name

        stem = Path(name).stem



        image_path = DATA / "crops_for_labelme" / name

        mask_path = DATA / str(row["binary_mask_path"])



        if not mask_path.exists():

            matches = list(

                (DATA / "binary_masks").glob(stem + "*.png")

            )

            assert len(matches) == 1, (name, matches)

            mask_path = matches[0]



        image = cv2.imread(str(image_path))

        mask = cv2.imread(

            str(mask_path), cv2.IMREAD_UNCHANGED

        )



        assert image is not None, image_path

        assert mask is not None, mask_path



        if mask.ndim == 3:

            mask = mask[:, :, 0]



        h, w = image.shape[:2]

        assert mask.shape == (h, w), name



        gt = mask > 0



        resized = cv2.resize(

            image, (SIZE, SIZE),

            interpolation=cv2.INTER_LINEAR

        )



        resized = cv2.cvtColor(

            resized, cv2.COLOR_BGR2RGB

        )



        tensor = torch.from_numpy(

            resized.transpose(2, 0, 1).copy()

        ).float().unsqueeze(0).to(device) / 255.0



        if device == "cuda":

            with torch.autocast("cuda", dtype=torch.float16):

                logits = model(tensor)

        else:

            logits = model(tensor)



        prob_small = torch.sigmoid(

            logits

        ).float().cpu().numpy()[0, 0]



        prob = cv2.resize(

            prob_small, (w, h),

            interpolation=cv2.INTER_LINEAR

        )



        prob = np.nan_to_num(

            prob, nan=0.0, posinf=1.0, neginf=0.0

        )

        prob = np.clip(prob, 0.0, 1.0)



        pred = prob >= THRESHOLD



        bin_ids = np.minimum(

            (prob * BINS).astype(np.int32),

            BINS - 1

        )



        pos_hist += np.bincount(

            bin_ids[gt].ravel(), minlength=BINS

        )

        neg_hist += np.bincount(

            bin_ids[~gt].ravel(), minlength=BINS

        )



        tp_i = int(np.logical_and(pred, gt).sum())

        fp_i = int(np.logical_and(pred, ~gt).sum())

        fn_i = int(np.logical_and(~pred, gt).sum())

        tn_i = int(np.logical_and(~pred, ~gt).sum())



        rows.append({

            "image_name": name,

            "hole_id": str(row["hole_id"]),

            "gt_positive_pixels": int(gt.sum()),

            "tp": tp_i,

            "fp": fp_i,

            "fn": fn_i,

            "tn": tn_i,

            "recall": tp_i / max(tp_i + fn_i, 1),

            "fpr": fp_i / max(fp_i + tn_i, 1),

            "precision": tp_i / max(tp_i + fp_i, 1)

        })



        # Spread 12 diagnostic images across validation.

        if index in np.linspace(

            0, len(df) - 1, 12, dtype=int

        ):

            save_visualization(

                name, image, gt, prob, pred

            )



        if (index + 1) % 25 == 0:

            print(

                f"Processed {index + 1}/{len(df)}",

                flush=True

            )



P = int(pos_hist.sum())

N = int(neg_hist.sum())



tp = np.cumsum(pos_hist[::-1]).astype(np.float64)

fp = np.cumsum(neg_hist[::-1]).astype(np.float64)



recall = tp / P

fpr = fp / N

precision = tp / np.maximum(tp + fp, 1)



auroc = float(

    np.trapz(

        np.r_[0.0, recall],

        np.r_[0.0, fpr]

    )

)



ap = float(

    np.sum(

        np.diff(np.r_[0.0, recall]) * precision

    )

)



# Exact fixed-threshold confusion matrix,

# aggregated from original-resolution image predictions.

tp_fixed = sum(r["tp"] for r in rows)

fp_fixed = sum(r["fp"] for r in rows)

fn_fixed = sum(r["fn"] for r in rows)

tn_fixed = sum(r["tn"] for r in rows)



fixed_recall = tp_fixed / max(tp_fixed + fn_fixed, 1)

fixed_fpr = fp_fixed / max(fp_fixed + tn_fixed, 1)

fixed_precision = tp_fixed / max(tp_fixed + fp_fixed, 1)



fixed_f1 = (

    2 * fixed_precision * fixed_recall /

    max(fixed_precision + fixed_recall, 1e-12)

)



results = {

    "experiment": "EXP009D-5D",

    "split": "validation",

    "images": len(df),

    "checkpoint_epoch": checkpoint["epoch"],

    "inference_resolution": [512, 512],

    "evaluation_resolution": "original",

    "fixed_threshold": THRESHOLD,

    "pixel_auroc_approx": auroc,

    "pixel_ap_approx": ap,

    "pixel_recall": fixed_recall,

    "pixel_fnr": 1.0 - fixed_recall,

    "pixel_fpr": fixed_fpr,

    "pixel_accuracy": (

        (tp_fixed + tn_fixed) /

        max(tp_fixed + fp_fixed + fn_fixed + tn_fixed, 1)

    ),

    "pixel_precision": fixed_precision,

    "pixel_f1": fixed_f1,

    "confusion_matrix": {

        "TP": tp_fixed,

        "FP": fp_fixed,

        "FN": fn_fixed,

        "TN": tn_fixed

    },

    "target_checks": {

        "auroc_ge_90": auroc >= 0.90,

        "accuracy_ge_90": (

            (tp_fixed + tn_fixed) /

            (tp_fixed + fp_fixed + fn_fixed + tn_fixed)

        ) >= 0.90,

        "recall_ge_90": fixed_recall >= 0.90,

        "fnr_le_10": (1.0 - fixed_recall) <= 0.10,

        "fpr_le_10": fixed_fpr <= 0.10

    }

}



pd.DataFrame(rows).to_csv(

    OUT / "per_image_metrics.csv", index=False

)



(OUT / "validation_metrics.json").write_text(

    json.dumps(results, indent=2)

)



print("\n=== ORIGINAL-RESOLUTION RESULTS ===")

print(json.dumps(results, indent=2))

print("\nVisualizations:", VIS)

print("Results:", OUT)

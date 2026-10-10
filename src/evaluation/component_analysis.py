import os

from pathlib import Path

import cv2

import json

import numpy as np

import pandas as pd

import torch

import segmentation_models_pytorch as smp



ROOT = Path(os.environ.get("TUBE_PROJECT_ROOT", ".")).expanduser().resolve()

DATA = Path(os.environ.get("TUBE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

BASE = Path(os.environ.get("TUBE_EXPERIMENT_DIR", str(ROOT / "experiments"))).expanduser().resolve()

OUT = BASE / "EXP009D-8D_component_audit"

OUT.mkdir(parents=True, exist_ok=True)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = 512

MIN_GT_AREA = 20

MIN_FP_AREA = 20

DETECTION_OVERLAP = 0.10



CONFIGS = [

    {

        "name": "original",

        "checkpoint": BASE / "EXP009D-5B_unet/best_model.pth",

        "threshold": 0.01904296875

    },

    {

        "name": "clean_negative_epoch_06",

        "checkpoint": BASE / "EXP009D-8B_clean_negative_training/epoch_06.pth",

        "threshold": 0.01727294921875

    }

]



manifest = pd.read_csv(DATA / "validation_manifest.csv")

assert len(manifest) == 129



def load_model(path):

    model = smp.Unet(

        encoder_name="resnet34",

        encoder_weights=None,

        in_channels=3,

        classes=1

    ).to(DEVICE)



    ckpt = torch.load(

        path, map_location="cpu", weights_only=True

    )

    model.load_state_dict(ckpt["model"])

    model.eval()

    return model



def component_category(area):

    if area < 500:

        return "tiny"

    if area < 5000:

        return "small"

    if area < 20000:

        return "medium"

    return "large"



def connected_components(mask):

    count, labels, stats, centroids = (

        cv2.connectedComponentsWithStats(

            mask.astype(np.uint8),

            connectivity=8

        )

    )

    return count, labels, stats



component_rows = []

image_rows = []

fp_rows = []



print("=== EXP009D-8D COMPONENT AUDIT ===", flush=True)

print("Device:", DEVICE, flush=True)

print("Validation images:", len(manifest), flush=True)

print("Minimum GT component:", MIN_GT_AREA, "pixels", flush=True)

print("Minimum FP component:", MIN_FP_AREA, "pixels", flush=True)

print("Detection overlap:", DETECTION_OVERLAP, flush=True)



for cfg in CONFIGS:

    model = load_model(cfg["checkpoint"])

    threshold = cfg["threshold"]



    print("\nEvaluating:", cfg["name"], flush=True)



    with torch.inference_mode():

        for index, row in enumerate(manifest.itertuples(), 1):

            name = Path(str(row.image_name)).name

            image = cv2.imread(

                str(DATA / "crops_for_labelme" / name)

            )



            mask_path = DATA / str(row.binary_mask_path)

            if not mask_path.exists():

                matches = list(

                    (DATA / "binary_masks").glob(

                        Path(name).stem + "*.png"

                    )

                )

                assert len(matches) == 1, name

                mask_path = matches[0]



            gt = cv2.imread(

                str(mask_path),

                cv2.IMREAD_UNCHANGED

            )



            assert image is not None and gt is not None, name

            if gt.ndim == 3:

                gt = gt[:, :, 0]



            gt = gt > 0

            h, w = gt.shape



            rgb = cv2.cvtColor(

                cv2.resize(image, (SIZE, SIZE)),

                cv2.COLOR_BGR2RGB

            )



            x = torch.from_numpy(

                rgb.transpose(2, 0, 1).copy()

            ).float().unsqueeze(0).to(DEVICE) / 255.0



            if DEVICE == "cuda":

                with torch.autocast("cuda", dtype=torch.float16):

                    logits = model(x)

            else:

                logits = model(x)



            prob = torch.sigmoid(

                logits

            ).float().cpu().numpy()[0, 0]



            prob = cv2.resize(

                prob, (w, h),

                interpolation=cv2.INTER_LINEAR

            )



            pred = prob >= threshold



            gt_count, gt_labels, gt_stats = (

                connected_components(gt)

            )

            pred_count, pred_labels, pred_stats = (

                connected_components(pred)

            )



            gt_components = 0

            detected = 0

            completely_missed = 0



            for cid in range(1, gt_count):

                area = int(

                    gt_stats[cid, cv2.CC_STAT_AREA]

                )

                if area < MIN_GT_AREA:

                    continue



                x0 = int(gt_stats[cid, cv2.CC_STAT_LEFT])

                y0 = int(gt_stats[cid, cv2.CC_STAT_TOP])

                cw = int(gt_stats[cid, cv2.CC_STAT_WIDTH])

                ch = int(gt_stats[cid, cv2.CC_STAT_HEIGHT])



                local_gt = (

                    gt_labels[y0:y0+ch, x0:x0+cw] == cid

                )

                local_pred = pred[

                    y0:y0+ch, x0:x0+cw

                ]



                overlap = int(

                    (local_gt & local_pred).sum()

                )

                recall = overlap / area

                is_detected = recall >= DETECTION_OVERLAP



                gt_components += 1

                detected += int(is_detected)

                completely_missed += int(overlap == 0)



                component_rows.append({

                    "model": cfg["name"],

                    "image_name": name,

                    "component_id": cid,

                    "area": area,

                    "category": component_category(area),

                    "width": cw,

                    "height": ch,

                    "overlap_pixels": overlap,

                    "component_recall": recall,

                    "detected_10pct": int(is_detected),

                    "any_overlap": int(overlap > 0),

                    "completely_missed": int(overlap == 0)

                })



            false_positive_regions = 0

            false_positive_pixels = 0



            for pid in range(1, pred_count):

                area = int(

                    pred_stats[pid, cv2.CC_STAT_AREA]

                )

                if area < MIN_FP_AREA:

                    continue



                x0 = int(pred_stats[pid, cv2.CC_STAT_LEFT])

                y0 = int(pred_stats[pid, cv2.CC_STAT_TOP])

                pw = int(pred_stats[pid, cv2.CC_STAT_WIDTH])

                ph = int(pred_stats[pid, cv2.CC_STAT_HEIGHT])



                local_component = (

                    pred_labels[y0:y0+ph, x0:x0+pw] == pid

                )

                local_gt = gt[

                    y0:y0+ph, x0:x0+pw

                ]



                overlap = int(

                    (local_component & local_gt).sum()

                )



                # A predicted component with zero GT overlap

                # is classified as a wholly false-positive region.

                if overlap == 0:

                    false_positive_regions += 1

                    false_positive_pixels += area



                    fp_rows.append({

                        "model": cfg["name"],

                        "image_name": name,

                        "pred_component_id": pid,

                        "area": area,

                        "x": x0,

                        "y": y0,

                        "width": pw,

                        "height": ph

                    })



            image_rows.append({

                "model": cfg["name"],

                "image_name": name,

                "gt_components": gt_components,

                "detected_components": detected,

                "completely_missed_components": completely_missed,

                "false_positive_regions": false_positive_regions,

                "false_positive_region_pixels": false_positive_pixels

            })



            if index % 25 == 0:

                print(

                    f"Processed {index}/{len(manifest)}",

                    flush=True

                )



components = pd.DataFrame(component_rows)

images = pd.DataFrame(image_rows)

fps = pd.DataFrame(fp_rows)



components.to_csv(

    OUT / "component_level_results.csv",

    index=False

)

images.to_csv(

    OUT / "image_level_results.csv",

    index=False

)

fps.to_csv(

    OUT / "false_positive_regions.csv",

    index=False

)



print("\n=== COMPONENT DETECTION SUMMARY ===")



summary_rows = []



for model_name in [c["name"] for c in CONFIGS]:

    c = components[

        components.model == model_name

    ]

    i = images[

        images.model == model_name

    ]



    summary_rows.append({

        "model": model_name,

        "gt_components": len(c),

        "detected_10pct": int(c.detected_10pct.sum()),

        "detection_recall": float(c.detected_10pct.mean()),

        "any_overlap_recall": float(c.any_overlap.mean()),

        "completely_missed": int(c.completely_missed.sum()),

        "fp_regions_total": int(i.false_positive_regions.sum()),

        "fp_regions_per_image": float(

            i.false_positive_regions.mean()

        ),

        "images_with_fp_regions": int(

            (i.false_positive_regions > 0).sum()

        )

    })



summary = pd.DataFrame(summary_rows)

print(summary.round(4).to_string(index=False))

summary.to_csv(

    OUT / "component_detection_summary.csv",

    index=False

)



print("\n=== COMPONENT DETECTION BY SIZE ===")



size_summary = (

    components.groupby(

        ["model", "category"]

    )

    .agg(

        components=("detected_10pct", "size"),

        detection_recall=("detected_10pct", "mean"),

        any_overlap_recall=("any_overlap", "mean"),

        completely_missed=("completely_missed", "sum")

    )

    .reset_index()

)



print(

    size_summary.round(4).to_string(index=False)

)

size_summary.to_csv(

    OUT / "component_size_summary.csv",

    index=False

)



print("\n=== FALSE-POSITIVE REGION SUMMARY ===")



if len(fps):

    print(

        fps.groupby("model")

        .agg(

            regions=("area", "size"),

            mean_area=("area", "mean"),

            median_area=("area", "median"),

            max_area=("area", "max")

        )

        .round(2)

        .to_string()

    )

else:

    print("No wholly false-positive regions found.")



print("\nSaved:", OUT)

print("=== EXP009D-8D COMPLETE ===")

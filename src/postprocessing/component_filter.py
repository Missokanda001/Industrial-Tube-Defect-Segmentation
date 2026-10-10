import os

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

import torch

import segmentation_models_pytorch as smp



ROOT = Path(os.environ.get("TUBE_PROJECT_ROOT", ".")).expanduser().resolve()

DATA = Path(os.environ.get("TUBE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

BASE = Path(os.environ.get("TUBE_EXPERIMENT_DIR", str(ROOT / "experiments"))).expanduser().resolve()

OUT = BASE / "EXP009D-8E_postprocessing"

OUT.mkdir(parents=True, exist_ok=True)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.01727294921875

FILTER_SIZES = [0, 20, 50, 100, 200, 500]

MIN_GT_COMPONENT = 20

DETECTION_OVERLAP = 0.10

MIN_FP_COMPONENT = 20



manifest = pd.read_csv(DATA / "validation_manifest.csv")

assert len(manifest) == 129



model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1

).to(DEVICE)



ckpt = torch.load(

    BASE / "EXP009D-8B_clean_negative_training/epoch_06.pth",

    map_location="cpu",

    weights_only=True

)

model.load_state_dict(ckpt["model"])

model.eval()



def filter_components(binary, min_area):

    if min_area == 0:

        return binary.copy()



    n, labels, stats, _ = cv2.connectedComponentsWithStats(

        binary.astype(np.uint8), connectivity=8

    )

    keep = np.zeros(n, dtype=np.uint8)

    keep[1:] = (

        stats[1:, cv2.CC_STAT_AREA] >= min_area

    ).astype(np.uint8)

    return keep[labels].astype(bool)



totals = {}

for size in FILTER_SIZES:

    totals[size] = {

        "TP": 0, "FP": 0, "FN": 0, "TN": 0,

        "gt_components": 0,

        "detected_components": 0,

        "small_components": 0,

        "detected_small": 0,

        "medium_components": 0,

        "detected_medium": 0,

        "large_components": 0,

        "detected_large": 0,

        "completely_missed": 0,

        "fp_regions": 0,

        "images_with_fp": 0

    }



print("=== EXP009D-8E POSTPROCESSING ===", flush=True)

print("Device:", DEVICE, flush=True)

print("Images:", len(manifest), flush=True)

print("Threshold:", THRESHOLD, flush=True)



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



        mask = cv2.imread(

            str(mask_path), cv2.IMREAD_UNCHANGED

        )

        assert image is not None and mask is not None, name

        if mask.ndim == 3:

            mask = mask[:, :, 0]



        gt = mask > 0

        h, w = gt.shape

        assert image.shape[:2] == (h, w), name



        rgb = cv2.cvtColor(

            cv2.resize(image, (512, 512)),

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

            prob, (w, h), interpolation=cv2.INTER_LINEAR

        )

        raw_pred = prob >= THRESHOLD



        ng, gt_labels, gt_stats, _ = (

            cv2.connectedComponentsWithStats(

                gt.astype(np.uint8), connectivity=8

            )

        )



        gt_components = []

        for cid in range(1, ng):

            area = int(gt_stats[cid, cv2.CC_STAT_AREA])

            if area < MIN_GT_COMPONENT:

                continue

            x0 = int(gt_stats[cid, cv2.CC_STAT_LEFT])

            y0 = int(gt_stats[cid, cv2.CC_STAT_TOP])

            cw = int(gt_stats[cid, cv2.CC_STAT_WIDTH])

            ch = int(gt_stats[cid, cv2.CC_STAT_HEIGHT])

            gt_components.append((cid, area, x0, y0, cw, ch))



        for size in FILTER_SIZES:

            pred = filter_components(raw_pred, size)

            t = totals[size]



            t["TP"] += int((pred & gt).sum())

            t["FP"] += int((pred & ~gt).sum())

            t["FN"] += int((~pred & gt).sum())

            t["TN"] += int((~pred & ~gt).sum())



            for cid, area, x0, y0, cw, ch in gt_components:

                local_gt = (

                    gt_labels[y0:y0+ch, x0:x0+cw] == cid

                )

                local_pred = pred[y0:y0+ch, x0:x0+cw]

                overlap = int((local_gt & local_pred).sum())

                detected = overlap / area >= DETECTION_OVERLAP



                t["gt_components"] += 1

                t["detected_components"] += int(detected)

                t["completely_missed"] += int(overlap == 0)



                if area < 5000:

                    category = "small" if area < 500 else "medium"

                else:

                    category = "large"



                t[f"{category}_components"] += 1

                t[f"detected_{category}"] += int(detected)



            npred, plabels, pstats, _ = (

                cv2.connectedComponentsWithStats(

                    pred.astype(np.uint8), connectivity=8

                )

            )



            image_fp_regions = 0

            for pid in range(1, npred):

                area = int(pstats[pid, cv2.CC_STAT_AREA])

                if area < MIN_FP_COMPONENT:

                    continue



                px = int(pstats[pid, cv2.CC_STAT_LEFT])

                py = int(pstats[pid, cv2.CC_STAT_TOP])

                pw = int(pstats[pid, cv2.CC_STAT_WIDTH])

                ph = int(pstats[pid, cv2.CC_STAT_HEIGHT])



                local_pred = (

                    plabels[py:py+ph, px:px+pw] == pid

                )

                local_gt = gt[py:py+ph, px:px+pw]



                if not np.any(local_pred & local_gt):

                    image_fp_regions += 1



            t["fp_regions"] += image_fp_regions

            t["images_with_fp"] += int(image_fp_regions > 0)



        if index % 25 == 0:

            print(

                f"Processed {index}/{len(manifest)}",

                flush=True

            )



rows = []

for size, t in totals.items():

    TP, FP, FN, TN = (

        t["TP"], t["FP"], t["FN"], t["TN"]

    )

    recall = TP / max(TP + FN, 1)

    fpr = FP / max(FP + TN, 1)

    accuracy = (TP + TN) / max(TP + FP + FN + TN, 1)

    precision = TP / max(TP + FP, 1)

    f1 = 2 * TP / max(2 * TP + FP + FN, 1)



    rows.append({

        "min_region_area": size,

        "recall": recall,

        "fnr": 1 - recall,

        "fpr": fpr,

        "accuracy": accuracy,

        "precision": precision,

        "f1": f1,

        "component_recall": (

            t["detected_components"] / max(t["gt_components"], 1)

        ),

        "small_component_recall": (

            t["detected_small"] / max(t["small_components"], 1)

        ),

        "medium_component_recall": (

            t["detected_medium"] / max(t["medium_components"], 1)

        ),

        "large_component_recall": (

            t["detected_large"] / max(t["large_components"], 1)

        ),

        "completely_missed": t["completely_missed"],

        "fp_regions": t["fp_regions"],

        "fp_regions_per_image": t["fp_regions"] / len(manifest),

        "images_with_fp": t["images_with_fp"]

    })



df = pd.DataFrame(rows)

df.to_csv(OUT / "postprocessing_results.csv", index=False)



print("\n=== PIXEL-LEVEL COMPARISON ===")

print(

    df[[

        "min_region_area", "recall", "fnr",

        "fpr", "accuracy", "precision", "f1"

    ]].round(5).to_string(index=False)

)



print("\n=== COMPONENT-LEVEL COMPARISON ===")

print(

    df[[

        "min_region_area", "component_recall",

        "small_component_recall",

        "medium_component_recall",

        "large_component_recall",

        "completely_missed",

        "fp_regions_per_image"

    ]].round(4).to_string(index=False)

)



eligible = df[

    (df.recall >= 0.90)

    & (df.fpr <= 0.10)

    & (df.accuracy >= 0.90)

]



print("\n=== ELIGIBLE CONFIGURATIONS ===")

if len(eligible):

    print(

        eligible[[

            "min_region_area", "recall", "fpr",

            "f1", "small_component_recall",

            "fp_regions_per_image"

        ]].round(4).to_string(index=False)

    )

else:

    print("No configurations meet all constraints.")



print("\nSaved:", OUT)

print("=== EXP009D-8E COMPLETE ===")

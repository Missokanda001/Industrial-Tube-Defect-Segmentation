import os

import json

import random

from pathlib import Path



import cv2

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

import segmentation_models_pytorch as smp



ROOT = Path(os.environ.get("TUBE_PROJECT_ROOT", ".")).expanduser().resolve()

DATA = Path(os.environ.get("TUBE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

OUT = ROOT / "experiments/EXP009D/EXP009D-5B_unet"

OUT.mkdir(parents=True, exist_ok=True)



SIZE = 512

BATCH = 4

EPOCHS = 20

LR = 0.0003

SEED = 42



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)



torch.backends.cudnn.benchmark = True



class DefectDataset(Dataset):

    def __init__(self, split, augment=False):

        self.df = pd.read_csv(DATA / f"{split}_manifest.csv")

        self.augment = augment

        self.split = split



        assert len(self.df) > 0

        assert set(self.df["speed_mm_s"].astype(int)) == {50}



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):

        row = self.df.iloc[index]

        name = Path(str(row["image_name"]).replace("\\", "/")).name

        stem = Path(name).stem



        image_path = DATA / "crops_for_labelme" / name



        mask_path = DATA / str(row["binary_mask_path"])

        if not mask_path.exists():

            mask_path = DATA / "binary_masks" / f"{stem}.png"



        if not mask_path.exists():

            matches = list((DATA / "binary_masks").glob(stem + "*.png"))

            if len(matches) != 1:

                raise FileNotFoundError(f"Mask for {name}: {matches}")

            mask_path = matches[0]



        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)



        if image is None or mask is None:

            raise RuntimeError(f"Cannot read {image_path} or {mask_path}")



        if mask.ndim == 3:

            mask = mask[:, :, 0]



        if image.shape[:2] != mask.shape:

            raise RuntimeError(f"Image/mask mismatch: {name}")



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)



        mask = (mask > 0).astype(np.float32)



        if self.augment:

            if random.random() < 0.5:

                image = np.flip(image, axis=1)

                mask = np.flip(mask, axis=1)



            if random.random() < 0.5:

                image = np.flip(image, axis=0)

                mask = np.flip(mask, axis=0)



            if random.random() < 0.5:

                alpha = random.uniform(0.85, 1.15)

                beta = random.uniform(-15, 15)

                image = np.clip(

                    image.astype(np.float32) * alpha + beta,

                    0, 255

                ).astype(np.uint8)



        image = np.ascontiguousarray(image)

        mask = np.ascontiguousarray(mask)



        image = torch.from_numpy(

            image.transpose(2, 0, 1).copy()

        ).float() / 255.0



        mask = torch.from_numpy(mask.copy()).unsqueeze(0)



        return image, mask





train_ds = DefectDataset("development", augment=True)

val_ds = DefectDataset("validation", augment=False)



train_holes = set(train_ds.df["hole_id"].astype(str))

val_holes = set(val_ds.df["hole_id"].astype(str))

test_df = pd.read_csv(DATA / "test_manifest.csv")

test_holes = set(test_df["hole_id"].astype(str))



assert not train_holes.intersection(val_holes)

assert not train_holes.intersection(test_holes)

assert not val_holes.intersection(test_holes)



print("Hole-disjoint split: VERIFIED", flush=True)

print("Train:", len(train_ds), "Validation:", len(val_ds), flush=True)



train_loader = DataLoader(

    train_ds, batch_size=BATCH, shuffle=True,

    num_workers=4, pin_memory=True

)



val_loader = DataLoader(

    val_ds, batch_size=BATCH, shuffle=False,

    num_workers=4, pin_memory=True

)



device = torch.device("cuda")



model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1

).to(device)



optimizer = torch.optim.AdamW(

    model.parameters(), lr=LR, weight_decay=1e-4

)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer, T_max=EPOCHS

)



dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

bce_loss = torch.nn.BCEWithLogitsLoss()



scaler = torch.amp.GradScaler("cuda")



best_ap = -1.0

history = []



print("Starting EXP009D-5B training...", flush=True)



for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0



    for images, masks in train_loader:

        images = images.to(device, non_blocking=True)

        masks = masks.to(device, non_blocking=True)



        optimizer.zero_grad(set_to_none=True)



        with torch.autocast(device_type="cuda", dtype=torch.float16):

            logits = model(images)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)



        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()



        running_loss += float(loss.item())



    scheduler.step()



    model.eval()

    val_scores = []

    val_labels = []



    # Fixed-seed stratified pixel sampling for fast validation.

    # Full-resolution evaluation will follow model selection.

    rng = np.random.default_rng(SEED)



    with torch.inference_mode():

        for images, masks in val_loader:

            images = images.to(device, non_blocking=True)



            with torch.autocast(device_type="cuda", dtype=torch.float16):

                logits = model(images)



            probabilities = torch.sigmoid(logits).float().cpu().numpy()

            targets = masks.numpy()



            for pred, gt in zip(probabilities, targets):

                pred = pred.ravel()

                gt = gt.ravel().astype(np.uint8)



                pos = np.flatnonzero(gt == 1)

                neg = np.flatnonzero(gt == 0)



                # Equal per-class sample caps for efficient ranking diagnostics.

                # These sampled AP values are NOT full-dataset pixel AP.

                n_pos = min(len(pos), 1000)

                n_neg = min(len(neg), 1000)



                if n_pos:

                    p = rng.choice(pos, n_pos, replace=False)

                    val_scores.append(pred[p])

                    val_labels.append(gt[p])



                if n_neg:

                    n = rng.choice(neg, n_neg, replace=False)

                    val_scores.append(pred[n])

                    val_labels.append(gt[n])



    scores = np.concatenate(val_scores)

    labels = np.concatenate(val_labels)



    auroc = float(roc_auc_score(labels, scores))

    ap = float(average_precision_score(labels, scores))



    avg_loss = running_loss / len(train_loader)



    record = {

        "epoch": epoch,

        "train_loss": avg_loss,

        "sampled_pixel_auroc": auroc,

        "sampled_pixel_ap": ap

    }



    history.append(record)



    print(

        f"Epoch {epoch:02d}/{EPOCHS} | "

        f"Loss {avg_loss:.4f} | "

        f"Sampled AUROC {auroc:.4f} | "

        f"Sampled AP {ap:.4f}",

        flush=True

    )



    # Checkpoint selection uses the same deterministic sampling protocol.

    if ap > best_ap:

        best_ap = ap

        torch.save({

            "model": model.state_dict(),

            "epoch": epoch,

            "sampled_pixel_ap": ap,

            "size": SIZE,

            "encoder": "resnet34",

            "encoder_weights": None

        }, OUT / "best_model.pth")



        print("  Saved new best checkpoint.", flush=True)



    pd.DataFrame(history).to_csv(

        OUT / "training_history.csv", index=False

    )



    (OUT / "training_history.json").write_text(

        json.dumps(history, indent=2)

    )



print("\nTRAINING COMPLETE", flush=True)

print("Best sampled validation AP:", best_ap, flush=True)

print("Checkpoint:", OUT / "best_model.pth", flush=True)

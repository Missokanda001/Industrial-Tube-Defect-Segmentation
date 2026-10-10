import os

from pathlib import Path

import json, random, time

import cv2

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp



ROOT = Path(os.environ.get("TUBE_PROJECT_ROOT", ".")).expanduser().resolve()

DATA = Path(os.environ.get("TUBE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()

BASE = Path(os.environ.get("TUBE_EXPERIMENT_DIR", str(ROOT / "experiments"))).expanduser().resolve()

MINE = BASE / "EXP009D-7C_mining"

OUT = BASE / "EXP009D-8B_clean_negative_training"

OUT.mkdir(parents=True, exist_ok=True)



SEED = 2026

SIZE = 512

EPOCHS = 6

BATCH = 4

LR = 2e-5

STEPS_PER_EPOCH = 130



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)



torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"



manifest = pd.read_csv(DATA / "development_manifest.csv")

missed = pd.read_csv(MINE / "missed_defect_patches.csv")

hard = pd.read_csv(BASE / "EXP009D-8A_clean_sampling" / "pure_background_hard_negatives.csv")



assert len(manifest) == 517

assert len(missed) > 0 and len(hard) > 0



# Restrict mined examples to development images.

allowed = set(manifest["image_name"].map(

    lambda x: Path(str(x)).name

))

assert set(missed.image_name).issubset(allowed)

assert set(hard.image_name).issubset(allowed)



image_rows = {

    Path(str(r.image_name)).name: r

    for r in manifest.itertuples()

}



class TargetedDataset(Dataset):

    def __init__(self):

        self.names = list(image_rows)

        self.missed = missed.to_dict("records")

        self.hard = hard.to_dict("records")

        self.length = STEPS_PER_EPOCH * BATCH



    def __len__(self):

        return self.length



    def __getitem__(self, index):

        # Exactly 50% full, 25% missed, 25% hard-negative

        mode = index % 4



        if mode < 2:

            name = random.choice(self.names)

            patch_info = None

        elif mode == 2:

            patch_info = random.choice(self.missed)

            name = patch_info["image_name"]

        else:

            patch_info = random.choice(self.hard)

            name = patch_info["image_name"]



        row = image_rows[name]

        image_path = DATA / "crops_for_labelme" / name

        mask_path = DATA / str(row.binary_mask_path)



        if not mask_path.exists():

            stem = Path(name).stem

            matches = list(

                (DATA / "binary_masks").glob(stem + "*.png")

            )

            assert len(matches) == 1, name

            mask_path = matches[0]



        image = cv2.imread(str(image_path))

        mask = cv2.imread(

            str(mask_path), cv2.IMREAD_UNCHANGED

        )

        assert image is not None and mask is not None, name



        if mask.ndim == 3:

            mask = mask[:, :, 0]



        assert image.shape[:2] == mask.shape, name

        mask = (mask > 0).astype(np.uint8)



        if patch_info is not None:

            x = int(patch_info["x0"])

            y = int(patch_info["y0"])

            p = int(patch_info["patch_size"])



            image = image[y:y+p, x:x+p]

            mask = mask[y:y+p, x:x+p]



        image = cv2.resize(

            image, (SIZE, SIZE),

            interpolation=cv2.INTER_LINEAR

        )

        mask = cv2.resize(

            mask, (SIZE, SIZE),

            interpolation=cv2.INTER_NEAREST

        )



        if random.random() < 0.5:

            image = cv2.flip(image, 1)

            mask = cv2.flip(mask, 1)



        if random.random() < 0.5:

            image = cv2.flip(image, 0)

            mask = cv2.flip(mask, 0)



        if random.random() < 0.25:

            image = cv2.rotate(

                image, cv2.ROTATE_90_CLOCKWISE

            )

            mask = cv2.rotate(

                mask, cv2.ROTATE_90_CLOCKWISE

            )



        if random.random() < 0.4:

            alpha = random.uniform(0.85, 1.15)

            beta = random.uniform(-12, 12)

            image = np.clip(

                image.astype(np.float32) * alpha + beta,

                0, 255

            ).astype(np.uint8)



        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(

            rgb.transpose(2, 0, 1).copy()

        ).float() / 255.0



        y = torch.from_numpy(

            mask.copy()

        ).float().unsqueeze(0)



        return x, y



dataset = TargetedDataset()

loader = DataLoader(

    dataset,

    batch_size=BATCH,

    shuffle=True,

    num_workers=2,

    pin_memory=(device == "cuda"),

    drop_last=True

)



model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1

).to(device)



source = BASE / "EXP009D-5B_unet/best_model.pth"

ckpt = torch.load(

    source, map_location="cpu", weights_only=True

)

model.load_state_dict(ckpt["model"])



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR,

    weight_decay=1e-4

)



dice_loss = smp.losses.DiceLoss(

    mode="binary",

    from_logits=True

)



history = []



print("=== EXP009D-8B TARGETED FINE-TUNING ===", flush=True)

print("Device:", device, flush=True)

print("Original checkpoint epoch:", ckpt["epoch"], flush=True)

print("Development images:", len(manifest), flush=True)

print("Missed patches:", len(missed), flush=True)

print("Hard-negative patches:", len(hard), flush=True)

print("Steps per epoch:", len(loader), flush=True)



for epoch in range(1, EPOCHS + 1):

    start = time.time()

    model.train()

    total_loss = 0.0

    total_bce = 0.0

    total_dice = 0.0



    for step, (x, y) in enumerate(loader, 1):

        x = x.to(device, non_blocking=True)

        y = y.to(device, non_blocking=True)



        optimizer.zero_grad(set_to_none=True)

        logits = model(x)



        # Mild positive weighting to emphasize missed defects.

        bce = F.binary_cross_entropy_with_logits(

            logits, y,

            pos_weight=torch.tensor(

                [1.0], device=device

            )

        )

        dice = dice_loss(logits, y)

        loss = bce + dice



        if not torch.isfinite(loss):

            raise RuntimeError("Non-finite training loss")



        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(), 1.0

        )

        optimizer.step()



        total_loss += loss.item()

        total_bce += bce.item()

        total_dice += dice.item()



        if step % 50 == 0:

            print(

                f"Epoch {epoch}/{EPOCHS} "

                f"step {step}/{len(loader)} "

                f"loss={loss.item():.4f}",

                flush=True

            )



    record = {

        "epoch": epoch,

        "loss": total_loss / len(loader),

        "bce": total_bce / len(loader),

        "dice": total_dice / len(loader),

        "seconds": round(time.time() - start, 1)

    }

    history.append(record)



    # Save each epoch; no validation-based selection yet.

    torch.save({

        "model": model.state_dict(),

        "epoch": epoch,

        "experiment": "EXP009D-8B",

        "source_checkpoint": str(source),

        "training_config": {

            "lr": LR,

            "batch_size": BATCH,

            "steps_per_epoch": STEPS_PER_EPOCH,

            "positive_weight": 2.0,

            "sampling": {

                "full": 0.50,

                "missed_defect": 0.25,

                "hard_negative": 0.25

            }

        }

    }, OUT / f"epoch_{epoch:02d}.pth")



    print(

        f"Epoch {epoch} completed | "

        f"loss={record['loss']:.5f} | "

        f"time={record['seconds']}s",

        flush=True

    )



pd.DataFrame(history).to_csv(

    OUT / "training_history.csv", index=False

)



(OUT / "training_summary.json").write_text(

    json.dumps({

        "experiment": "EXP009D-8B",

        "split": "development",

        "epochs_completed": EPOCHS,

        "history": history,

        "validation_evaluated": False,

        "test_evaluated": False

    }, indent=2)

)



print("\n=== TRAINING COMPLETE ===")

print(pd.DataFrame(history).to_string(index=False))

print("\nSaved checkpoints:", OUT)

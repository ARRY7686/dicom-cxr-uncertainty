
import os
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.rsna_dataset import RSNADicomDataset
from src.models.densenet import build_model
from src.training.trainer import (
    train_one_epoch,
    validate
)
from src.utils.transforms import (
    get_train_transforms,
    get_val_transforms
)


# =========================
# CONFIG
# =========================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

DATA_DIR = (
    "/kaggle/input/"
    "competitions/"
    "rsna-pneumonia-detection-challenge/"
)

IMAGE_DIR = os.path.join(
    DATA_DIR,
    "stage_2_train_images"
)

LABELS_PATH = os.path.join(
    DATA_DIR,
    "stage_2_train_labels.csv"
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# LOAD LABELS
# =========================

df = pd.read_csv(LABELS_PATH)

classification_df = (
    df.groupby("patientId")["Target"]
    .max()
    .reset_index()
)

train_df, val_df = train_test_split(
    classification_df,
    test_size=0.2,
    stratify=classification_df["Target"],
    random_state=42
)


# =========================
# DATASETS
# =========================

train_dataset = RSNADicomDataset(
    dataframe=train_df,
    image_dir=IMAGE_DIR,
    transform=get_train_transforms()
)

val_dataset = RSNADicomDataset(
    dataframe=val_df,
    image_dir=IMAGE_DIR,
    transform=get_val_transforms()
)


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# =========================
# MODEL
# =========================

model = build_model()

model = model.to(device)

pos_count = train_df["Target"].sum()

neg_count = len(train_df) - pos_count

pos_weight = torch.tensor(
    [neg_count / pos_count],
    dtype=torch.float32
).to(device)

print("Positive Weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING LOOP
# =========================

best_auc = 0.0

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    val_loss, val_auc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUROC: {val_auc:.4f}")

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved")


print(f"\nBest Validation AUROC: {best_auc:.4f}")

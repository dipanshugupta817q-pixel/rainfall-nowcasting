from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)


# ============================================================
# IMPORTS
# ============================================================

from data.dataset import ERA5SequenceDataset
from models.era5_model import ERA5ForecastModel


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

EPOCHS = 10

HIDDEN_CHANNELS = 64

GRADIENT_CLIP = 1.0

PATIENCE = 3

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "model_b_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "model_b_last.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOSS
# ============================================================

class ERA5HuberLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.HuberLoss(
            delta=1.0
        )

    def forward(
        self,
        prediction,
        target
    ):

        return self.loss(
            prediction,
            target
        )


# ============================================================
# DATA LOADERS
# ============================================================

def create_loaders():

    train_dataset = ERA5SequenceDataset(
        split="train",
        transform=True
    )

    validation_dataset = ERA5SequenceDataset(
        split="validation",
        transform=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    total_batches = len(loader)

    for batch_idx, (x, target) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(x)

        loss = criterion(
            prediction,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        running_loss += loss.item()

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/{total_batches} "
                f"| Loss: {loss.item():.6f}"
            )

    return (
        running_loss
        / total_batches
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    total_batches = len(loader)

    for x, target in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        prediction = model(x)

        loss = criterion(
            prediction,
            target
        )

        running_loss += loss.item()

    return (
        running_loss
        / total_batches
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL B — ERA5 DEEP LEARNING TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = create_loaders()

    print()
    print(
        "Training sequences:",
        f"{len(train_dataset):,}"
    )

    print(
        "Validation sequences:",
        f"{len(validation_dataset):,}"
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ERA5ForecastModel(
        input_channels=7,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=7,
        forecast_steps=12,
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = ERA5HuberLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()

        print()
        print("=" * 60)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss = validate(
            model,
            validation_loader,
            criterion
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time()
            - start_time
        )

        print()
        print(
            f"Train loss      : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Learning rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch time      : "
            f"{elapsed / 60:.2f} min"
        )

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "train_loss":
                    train_loss,
                "validation_loss":
                    validation_loss,
            },
            LAST_MODEL_PATH
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "scheduler_state_dict":
                        scheduler.state_dict(),
                    "train_loss":
                        train_loss,
                    "validation_loss":
                        validation_loss,
                },
                BEST_MODEL_PATH
            )

            print(
                "✓ BEST MODEL SAVED"
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement:"
                f" {epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("MODEL B TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )

    print(
        "Best checkpoint:",
        BEST_MODEL_PATH
    )

    print(
        "Last checkpoint:",
        LAST_MODEL_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
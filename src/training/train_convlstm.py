from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from data.dataset import IMERGSequenceDataset
from models.convlstm import RainfallConvLSTM


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

LEARNING_RATE = 1e-3

EPOCHS = 10

HIDDEN_CHANNELS = 64

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "model_a_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "model_a_last.pt"
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

class RainfallHuberLoss(nn.Module):

    def __init__(
        self,
        delta=1.0
    ):

        super().__init__()

        self.loss = nn.HuberLoss(
            delta=delta
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
# DATA
# ============================================================

def create_loaders():

    train_dataset = IMERGSequenceDataset(
        split="train",
        transform=True
    )

    validation_dataset = IMERGSequenceDataset(
        split="validation",
        transform=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    )


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    batches = 0

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(x)

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        batches += 1

    return total_loss / batches


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

    total_loss = 0.0

    batches = 0

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        prediction = model(x)

        loss = criterion(
            prediction,
            y
        )

        total_loss += (
            loss.item()
        )

        batches += 1

    return total_loss / batches


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("MODEL A — CONVLSTM TRAINING")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "WARNING: Running on CPU."
        )

    print()

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
        f"Training sequences   : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences : "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs               : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate        : "
        f"{LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RainfallConvLSTM(
        input_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=1,
        input_frames=12,
        output_frames=12
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Model parameters     : "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = RainfallHuberLoss(
        delta=1.0
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "validation_loss":
                    validation_loss,
                "train_loss":
                    train_loss,
                "learning_rate":
                    current_lr
            },
            LAST_MODEL_PATH
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            validation_loss
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        validation_loss,
                    "train_loss":
                        train_loss,
                    "learning_rate":
                        current_lr
                },
                BEST_MODEL_PATH
            )

            best_marker = "  <-- BEST"

        else:

            best_marker = ""

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

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
            f"Time            : "
            f"{elapsed / 60:.2f} min"
        )

        print(
            f"Best val loss   : "
            f"{best_validation_loss:.6f}"
            f"{best_marker}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation loss : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Best checkpoint      : "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Last checkpoint      : "
        f"{LAST_MODEL_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
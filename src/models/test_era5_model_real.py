import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

from data.era5_dataset import ERA5SequenceDataset
from models.era5_model import ERA5ForecastModel


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL B — REAL ERA5 FORECAST TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ERA5SequenceDataset(
        split="train",
        transform=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Real batch
    # --------------------------------------------------------

    x, target = next(iter(loader))

    print()
    print("REAL ERA5 BATCH")

    print(
        "Input :",
        tuple(x.shape)
    )

    print(
        "Target:",
        tuple(target.shape)
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    x = x.to(
        device,
        non_blocking=True
    )

    target = target.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ERA5ForecastModel(
        input_channels=7,
        hidden_channels=64,
        output_channels=7,
        forecast_steps=12,
    ).to(device)

    model.train()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    prediction = model(x)

    print()
    print("MODEL OUTPUT")

    print(
        "Prediction:",
        tuple(prediction.shape)
    )

    print(
        "Prediction min:",
        prediction.min().item()
    )

    print(
        "Prediction max:",
        prediction.max().item()
    )

    # --------------------------------------------------------
    # Loss
    #
    # Huber is robust to occasional large atmospheric
    # prediction errors.
    # --------------------------------------------------------

    criterion = nn.HuberLoss(
        delta=1.0
    )

    loss = criterion(
        prediction,
        target
    )

    print()
    print(
        "Huber loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # --------------------------------------------------------
    # Gradient validation
    # --------------------------------------------------------

    gradient_tensors = 0
    total_gradient_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "NaN/Inf gradient detected"
                )

            gradient_tensors += 1

            total_gradient_norm += (
                parameter.grad.norm().item()
            )

    print()
    print(
        "Gradient tensors:",
        gradient_tensors
    )

    print(
        "Gradient norm:",
        total_gradient_norm
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert x.shape == (
        BATCH_SIZE,
        12,
        7,
        20,
        30
    )

    assert target.shape == (
        BATCH_SIZE,
        12,
        7,
        20,
        30
    )

    assert prediction.shape == (
        BATCH_SIZE,
        12,
        7,
        20,
        30
    )

    assert torch.isfinite(
        prediction
    ).all()

    assert torch.isfinite(
        loss
    )

    if gradient_tensors == 0:

        raise RuntimeError(
            "No gradients were produced"
        )

    print()
    print("=" * 60)
    print(
        "STATUS: REAL ERA5 MODEL "
        "FORWARD/BACKWARD TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

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
from models.era5_encoder import ERA5AtmosphericEncoder


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL B — REAL ERA5 + ENCODER TEST")
    print("=" * 60)

    print("Device:", DEVICE)

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

    x, y = next(iter(loader))

    print()
    print("REAL ERA5 BATCH")

    print(
        "Input :",
        tuple(x.shape)
    )

    print(
        "Target:",
        tuple(y.shape)
    )

    print(
        "Input min:",
        x.min().item()
    )

    print(
        "Input max:",
        x.max().item()
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    x = x.to(
        DEVICE,
        non_blocking=True
    )

    y = y.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ERA5AtmosphericEncoder(
        input_channels=7,
        hidden_channels=64
    ).to(DEVICE)

    model.train()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    latent = model(x)

    print()
    print("ENCODER OUTPUT")

    print(
        "Latent shape:",
        tuple(latent.shape)
    )

    print(
        "Latent min:",
        latent.min().item()
    )

    print(
        "Latent max:",
        latent.max().item()
    )

    # --------------------------------------------------------
    # Dummy projection
    #
    # We need a scalar objective for backward testing.
    # This does NOT represent the final training loss.
    # --------------------------------------------------------

    loss = latent.mean()

    print()
    print(
        "Test loss:",
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
    # Gradient check
    # --------------------------------------------------------

    total_gradient = 0.0
    gradient_tensors = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "NaN/Inf gradient detected"
                )

            total_gradient += (
                parameter.grad.norm().item()
            )

            gradient_tensors += 1

    print()
    print(
        "Gradient tensors:",
        gradient_tensors
    )

    print(
        "Gradient norm:",
        total_gradient
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

    assert y.shape == (
        BATCH_SIZE,
        12,
        7,
        20,
        30
    )

    assert latent.shape == (
        BATCH_SIZE,
        64,
        20,
        30
    )

    assert torch.isfinite(
        latent
    ).all()

    if gradient_tensors == 0:

        raise RuntimeError(
            "No gradients were produced"
        )

    print()
    print("=" * 60)
    print(
        "STATUS: REAL ERA5 + ENCODER "
        "FORWARD/BACKWARD TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
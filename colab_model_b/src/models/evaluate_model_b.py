
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "colab_model_b" / "src"))

from data.dataset import ERA5SequenceDataset
from models.era5_model import ERA5ForecastModel

DEVICE = torch.device("cpu")
CHECKPOINT = ROOT / "models" / "checkpoints" / "model_b_best.pt"

VARIABLES = ["t2m", "d2m", "u10", "v10", "sp", "msl", "tcc"]

BATCH_SIZE = 2
MAX_BATCHES = 10  # Initial evaluation: up to 20 samples


def calculate_metrics(pred, target):
    error = pred - target
    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()
    return mae, rmse


def main():
    print("=" * 60)
    print("MODEL B — REAL VALIDATION INFERENCE")
    print("=" * 60)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    dataset = ERA5SequenceDataset(
        split="validation",
        transform=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ERA5ForecastModel(
        input_channels=7,
        hidden_channels=64,
        output_channels=7,
        forecast_steps=12
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Recorded validation loss:", checkpoint.get("validation_loss"))
    print("Validation sequences:", len(dataset))

    total_abs = torch.zeros(7)
    total_sq = torch.zeros(7)
    baseline_abs = torch.zeros(7)
    baseline_sq = torch.zeros(7)
    total_values = 0

    with torch.inference_mode():
        for batch_idx, (x, target) in enumerate(loader):
            if batch_idx >= MAX_BATCHES:
                break

            x = x.to(DEVICE)
            target = target.to(DEVICE)

            prediction = model(x)

            # Persistence baseline:
            # Repeat the last observed frame for all 12 steps.
            baseline = x[:, -1:, :, :, :].expand_as(target)

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Shape mismatch: {prediction.shape} vs {target.shape}"
                )

            if not torch.isfinite(prediction).all():
                raise RuntimeError("NaN/Inf in model predictions")

            error = prediction - target
            base_error = baseline - target

            # Sum over batch, time, height and width.
            total_abs += error.abs().sum(dim=(0, 1, 3, 4))
            total_sq += (error ** 2).sum(dim=(0, 1, 3, 4))

            baseline_abs += base_error.abs().sum(dim=(0, 1, 3, 4))
            baseline_sq += (base_error ** 2).sum(dim=(0, 1, 3, 4))

            total_values += (
                target.shape[0]
                * target.shape[1]
                * target.shape[3]
                * target.shape[4]
            )

            print(
                f"Batch {batch_idx + 1}: "
                f"prediction shape {tuple(prediction.shape)}"
            )

    if total_values == 0:
        raise RuntimeError("No validation samples evaluated")

    # Normalized-space metrics.
    mae = total_abs / total_values
    rmse = torch.sqrt(total_sq / total_values)

    base_mae = baseline_abs / total_values
    base_rmse = torch.sqrt(baseline_sq / total_values)

    # Convert errors to physical units using training std.
    stats = np.load(ROOT / "data" / "processed" / "era5_era5_stats.npz")
    std = torch.tensor(stats["std"], dtype=torch.float32)

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)

    print(f"{'Variable':<10} {'Model MAE':>12} {'Model RMSE':>12} "
          f"{'Persistence MAE':>18} {'Persistence RMSE':>20}")

    for i, name in enumerate(VARIABLES):
        print(
            f"{name:<10} "
            f"{mae[i].item():>12.5f} "
            f"{rmse[i].item():>12.5f} "
            f"{base_mae[i].item():>18.5f} "
            f"{base_rmse[i].item():>20.5f}"
        )

    print("\nApproximate physical-unit MAE:")
    for i, name in enumerate(VARIABLES):
        print(f"{name}: {(mae[i] * std[i]).item():.6f}")

    print("\nEvaluated samples:", min(
        len(dataset), MAX_BATCHES * BATCH_SIZE
    ))

    print("=" * 75)
    print("STATUS: INFERENCE EVALUATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()

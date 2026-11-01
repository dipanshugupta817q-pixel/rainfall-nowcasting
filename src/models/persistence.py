from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import IMERGSequenceDataset


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0
FORECAST_STEPS = 12


# ============================================================
# METRICS
# ============================================================

def calculate_mae(prediction, target):
    return float(
        np.mean(
            np.abs(prediction - target)
        )
    )


def calculate_rmse(prediction, target):
    return float(
        np.sqrt(
            np.mean(
                (prediction - target) ** 2
            )
        )
    )


def calculate_bias(prediction, target):
    return float(
        np.mean(
            prediction - target
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_persistence():

    print("=" * 70)
    print("MODEL A — PERSISTENCE BASELINE")
    print("=" * 70)

    print()
    print("Validation period : 2021–2024")
    print("Forecast horizon  : +30 min → +6 hours")
    print()

    dataset = IMERGSequenceDataset(
        split="validation",
        transform=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Validation sequences : {len(dataset):,}"
    )

    print()
    print("Starting evaluation...")
    print()

    all_predictions = [
        []
        for _ in range(FORECAST_STEPS)
    ]

    all_targets = [
        []
        for _ in range(FORECAST_STEPS)
    ]

    # --------------------------------------------------------
    # Process validation dataset
    # --------------------------------------------------------

    for batch_number, (x, y) in enumerate(loader):

        # x:
        # (B, 12, 20, 30, 1)

        # y:
        # (B, 12, 20, 30, 1)

        # Latest observed rainfall frame
        last_observed = x[:, -1:, :, :, :]

        # Repeat latest frame for all future horizons
        prediction = last_observed.repeat(
            1,
            FORECAST_STEPS,
            1,
            1,
            1
        )

        # Convert log1p → mm/hr
        prediction = torch.expm1(prediction)
        target = torch.expm1(y)

        prediction = prediction.numpy()
        target = target.numpy()

        for horizon in range(FORECAST_STEPS):

            all_predictions[horizon].append(
                prediction[:, horizon]
            )

            all_targets[horizon].append(
                target[:, horizon]
            )

        if (batch_number + 1) % 100 == 0:

            print(
                f"Processed batches: "
                f"{batch_number + 1:,}"
            )

    # ========================================================
    # HORIZON-WISE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PERSISTENCE BASELINE RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'Horizon':<12}"
        f"{'Time':<12}"
        f"{'MAE':<15}"
        f"{'RMSE':<15}"
        f"{'Bias':<15}"
    )

    print("-" * 70)

    all_prediction_values = []
    all_target_values = []

    for horizon in range(FORECAST_STEPS):

        prediction = np.concatenate(
            all_predictions[horizon],
            axis=0
        )

        target = np.concatenate(
            all_targets[horizon],
            axis=0
        )

        mae = calculate_mae(
            prediction,
            target
        )

        rmse = calculate_rmse(
            prediction,
            target
        )

        bias = calculate_bias(
            prediction,
            target
        )

        minutes = (
            (horizon + 1) * 30
        )

        if minutes < 60:
            time_label = f"+{minutes}m"
        else:
            time_label = f"+{minutes / 60:g}h"

        print(
            f"{horizon + 1:<12}"
            f"{time_label:<12}"
            f"{mae:<15.6f}"
            f"{rmse:<15.6f}"
            f"{bias:<15.6f}"
        )

        all_prediction_values.append(
            prediction.reshape(-1)
        )

        all_target_values.append(
            target.reshape(-1)
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    all_prediction_values = np.concatenate(
        all_prediction_values
    )

    all_target_values = np.concatenate(
        all_target_values
    )

    overall_mae = calculate_mae(
        all_prediction_values,
        all_target_values
    )

    overall_rmse = calculate_rmse(
        all_prediction_values,
        all_target_values
    )

    overall_bias = calculate_bias(
        all_prediction_values,
        all_target_values
    )

    print()
    print("=" * 70)
    print("OVERALL 6-HOUR PERSISTENCE BASELINE")
    print("=" * 70)

    print(
        f"MAE  : {overall_mae:.6f} mm/hr"
    )

    print(
        f"RMSE : {overall_rmse:.6f} mm/hr"
    )

    print(
        f"Bias : {overall_bias:.6f} mm/hr"
    )

    print("=" * 70)

    print()
    print("STATUS: PERSISTENCE BASELINE COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_persistence()
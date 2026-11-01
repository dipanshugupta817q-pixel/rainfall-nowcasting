import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PATH
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

BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.HuberLoss(
    delta=1.0
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL A — REAL DATA LOSS TEST")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = IMERGSequenceDataset(
        split="train",
        transform=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Get one real batch
    # --------------------------------------------------------

    x, y = next(iter(loader))

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    print()
    print("REAL DATA")
    print(
        "Input shape :",
        tuple(x.shape)
    )

    print(
        "Target shape:",
        tuple(y.shape)
    )

    print(
        "Input min   :",
        x.min().item()
    )

    print(
        "Input max   :",
        x.max().item()
    )

    print(
        "Target min  :",
        y.min().item()
    )

    print(
        "Target max  :",
        y.max().item()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RainfallConvLSTM(
        input_channels=1,
        hidden_channels=64,
        output_channels=1,
        input_frames=12,
        output_frames=12
    ).to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(x)

        loss = criterion(
            prediction,
            y
        )

    print()
    print("MODEL OUTPUT")

    print(
        "Prediction shape:",
        tuple(prediction.shape)
    )

    print(
        "Prediction min  :",
        prediction.min().item()
    )

    print(
        "Prediction max  :",
        prediction.max().item()
    )

    print(
        "Prediction mean :",
        prediction.mean().item()
    )

    print()
    print(
        "Huber loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    assert prediction.shape == y.shape

    assert torch.isfinite(
        prediction
    ).all()

    assert torch.isfinite(
        loss
    )

    assert (
        prediction.min().item()
        >= 0
    )

    print()
    print("=" * 70)
    print("STATUS: REAL DATA + MODEL + LOSS TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
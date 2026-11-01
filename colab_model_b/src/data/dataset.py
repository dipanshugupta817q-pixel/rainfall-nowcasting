from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "era5_aligned"
SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "era5_sequences"
STATS_FILE = PROJECT_ROOT / "data" / "processed" / "era5_era5_stats.npz"

VARIABLES = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "sp",
    "msl",
    "tcc",
]

EXPECTED_FRAME_SHAPE = (7, 20, 30)


# ============================================================
# DATASET
# ============================================================

class ERA5SequenceDataset(Dataset):

    def __init__(
        self,
        split="train",
        transform=True,
    ):

        self.split = split
        self.transform = transform

        sequence_file = (
            SEQUENCE_DIR
            / f"{split}.json"
        )

        if not sequence_file.exists():
            raise FileNotFoundError(
                f"Sequence file not found: {sequence_file}"
            )

        with open(
            sequence_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.sequences = json.load(f)

        if not self.sequences:
            raise RuntimeError(
                f"No sequences found for split: {split}"
            )

        # ----------------------------------------------------
        # Load normalization statistics
        # ----------------------------------------------------

        if not STATS_FILE.exists():
            raise FileNotFoundError(
                f"Statistics file not found: {STATS_FILE}"
            )

        stats = np.load(STATS_FILE)

        self.mean = stats["mean"].astype(
            np.float32
        )

        self.std = stats["std"].astype(
            np.float32
        )

        # Prevent division by zero
        self.std = np.maximum(
            self.std,
            1e-6
        )

        if self.mean.shape != (7,):
            raise RuntimeError(
                f"Mean shape incorrect: {self.mean.shape}"
            )

        if self.std.shape != (7,):
            raise RuntimeError(
                f"Std shape incorrect: {self.std.shape}"
            )

        print(
            f"{split.upper()} dataset loaded: "
            f"{len(self.sequences):,} sequences"
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.sequences)


    # ========================================================
    # LOAD FRAME
    # ========================================================

    def _load_frame(self, file_path, frame_index):

        path = Path(file_path)

        # ----------------------------------------------------
        # Handle relative paths
        # ----------------------------------------------------

        if not path.is_absolute():

            # JSON contains project-relative paths
            # such as:
            #
            # data\processed\era5_aligned\20210601.npz

            path = PROJECT_ROOT / path

        if not path.exists():

            raise FileNotFoundError(
                f"ERA5 file not found: {path}"
            )

        with np.load(path) as data:

            arr = data["data"][frame_index]

        # ----------------------------------------------------
        # Expected:
        #
        # (7,20,30)
        # ----------------------------------------------------

        if arr.shape != EXPECTED_FRAME_SHAPE:

            raise RuntimeError(
                f"Unexpected frame shape "
                f"{arr.shape} in {path.name}"
            )

        arr = arr.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not np.isfinite(arr).all():

            raise RuntimeError(
                f"NaN/Inf detected in {path.name}"
            )

        return arr


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        sequence = self.sequences[index]

        input_frames = sequence["input"]
        target_frames = sequence["target"]

        # ----------------------------------------------------
        # Load input
        # ----------------------------------------------------

        x_frames = []

        for item in input_frames:

            frame = self._load_frame(
                item["file"],
                item["frame"]
            )

            x_frames.append(frame)

        # ----------------------------------------------------
        # Load target
        # ----------------------------------------------------

        y_frames = []

        for item in target_frames:

            frame = self._load_frame(
                item["file"],
                item["frame"]
            )

            y_frames.append(frame)

        # ----------------------------------------------------
        # Stack
        #
        # X/Y:
        # (12,7,20,30)
        # ----------------------------------------------------

        x = np.stack(
            x_frames,
            axis=0
        )

        y = np.stack(
            y_frames,
            axis=0
        )

        # ----------------------------------------------------
        # Normalize using TRAIN statistics
        #
        # mean/std shape:
        # (7,)
        #
        # reshape:
        # (1,7,1,1)
        # ----------------------------------------------------

        if self.transform:

            mean = self.mean[
                None, :, None, None
            ]

            std = self.std[
                None, :, None, None
            ]

            x = (
                x - mean
            ) / std

            y = (
                y - mean
            ) / std

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        x = torch.from_numpy(
            x.copy()
        ).float()

        y = torch.from_numpy(
            y.copy()
        ).float()

        return x, y


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    split="train",
    batch_size=8,
    shuffle=None,
    num_workers=0,
):

    dataset = ERA5SequenceDataset(
        split=split,
        transform=True
    )

    if shuffle is None:

        shuffle = (
            split == "train"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL B — ERA5 DATASET TEST")
    print("=" * 60)

    dataset, loader = create_dataloader(
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    print()
    print("Dataset length:", len(dataset))

    # --------------------------------------------------------
    # Single sample
    # --------------------------------------------------------

    x, y = dataset[0]

    print()
    print("SINGLE SAMPLE")
    print("Input shape :", tuple(x.shape))
    print("Target shape:", tuple(y.shape))

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
    # Batch
    # --------------------------------------------------------

    xb, yb = next(iter(loader))

    print()
    print("DATALOADER BATCH")

    print(
        "Input shape :",
        tuple(xb.shape)
    )

    print(
        "Target shape:",
        tuple(yb.shape)
    )

    print(
        "Input dtype :",
        xb.dtype
    )

    print(
        "Target dtype:",
        yb.dtype
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert x.shape == (12, 7, 20, 30)
    assert y.shape == (12, 7, 20, 30)

    assert xb.shape == (
        2, 12, 7, 20, 30
    )

    assert yb.shape == (
        2, 12, 7, 20, 30
    )

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()

    assert torch.isfinite(xb).all()
    assert torch.isfinite(yb).all()

    print()
    print("=" * 60)
    print("STATUS: ERA5 DATASET TEST PASSED")
    print("=" * 60)
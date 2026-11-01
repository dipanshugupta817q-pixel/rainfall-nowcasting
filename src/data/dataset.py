import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

MEMMAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "memmap"
)

MEMMAP_FILE = (
    MEMMAP_DIR
    / "imerg_rainfall.npy"
)

FRAME_INDEX_FILE = (
    MEMMAP_DIR
    / "frame_index.json"
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FRAMES = 12
OUTPUT_FRAMES = 12

HEIGHT = 20
WIDTH = 30

EXPECTED_FRAME_SHAPE = (
    HEIGHT,
    WIDTH
)


# ============================================================
# FILENAME PATTERNS
# ============================================================

# Canonical:
#
# 20080601_0000.npz

CANONICAL_PATTERN = re.compile(
    r"^(\d{8})_(\d{4})\.npz$"
)


# Original IMERG:
#
# 3B-HHR.MS.MRG.3IMERG.20080601-S000000-E002959.0000.V07B.HDF5.npz
#
# We only need:
#
# YYYYMMDD
# HHMMSS

IMERG_PATTERN = re.compile(
    r"\.3IMERG\.(\d{8})-S(\d{2})(\d{2})(\d{2})-"
)


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def normalize_timestamp(value):
    """
    Convert sequence entry into the exact timestamp key used
    by frame_index.json.

    Supported inputs:

        2008-06-01T00:00:00

        2008-06-01T00:00:00Z

        data\\processed\\imerg\\20080601_0000.npz

        data\\processed\\imerg\\
        3B-HHR.MS.MRG.3IMERG.20080601-S000000-...
    """

    value = str(value)

    # --------------------------------------------------------
    # Case 1:
    # Already ISO timestamp
    # --------------------------------------------------------

    if value in (
        normalize_timestamp._frame_index
        if hasattr(
            normalize_timestamp,
            "_frame_index"
        )
        else {}
    ):

        return value

    # --------------------------------------------------------
    # Remove trailing Z
    # --------------------------------------------------------

    value_no_z = value.rstrip("Z")

    if value_no_z in (
        normalize_timestamp._frame_index
        if hasattr(
            normalize_timestamp,
            "_frame_index"
        )
        else {}
    ):

        return value_no_z

    # --------------------------------------------------------
    # Extract filename
    # --------------------------------------------------------

    filename = Path(value).name

    # --------------------------------------------------------
    # Canonical format
    # --------------------------------------------------------

    match = CANONICAL_PATTERN.match(
        filename
    )

    if match:

        date_part = match.group(1)

        time_part = match.group(2)

        return (
            f"{date_part[:4]}-"
            f"{date_part[4:6]}-"
            f"{date_part[6:8]}T"
            f"{time_part[:2]}:"
            f"{time_part[2:4]}:00"
        )

    # --------------------------------------------------------
    # Original IMERG format
    # --------------------------------------------------------

    match = IMERG_PATTERN.search(
        filename
    )

    if match:

        date_part = match.group(1)

        hour = match.group(2)

        minute = match.group(3)

        second = match.group(4)

        return (
            f"{date_part[:4]}-"
            f"{date_part[4:6]}-"
            f"{date_part[6:8]}T"
            f"{hour}:"
            f"{minute}:"
            f"{second}"
        )

    return None


# ============================================================
# DATASET
# ============================================================

class IMERGSequenceDataset(Dataset):

    def __init__(
        self,
        split="train",
        transform=True
    ):

        self.split = split

        self.transform = transform

        # ----------------------------------------------------
        # Validate split
        # ----------------------------------------------------

        valid_splits = {
            "train",
            "validation",
            "test"
        }

        if split not in valid_splits:

            raise ValueError(
                f"Invalid split '{split}'. "
                f"Choose from {valid_splits}"
            )

        # ----------------------------------------------------
        # Sequence file
        # ----------------------------------------------------

        sequence_file = (
            SEQUENCE_DIR
            / f"{split}.json"
        )

        if not sequence_file.exists():

            raise FileNotFoundError(
                f"Sequence file not found:\n"
                f"{sequence_file}"
            )

        with open(
            sequence_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.sequences = json.load(f)

        # ----------------------------------------------------
        # Memmap existence
        # ----------------------------------------------------

        if not MEMMAP_FILE.exists():

            raise FileNotFoundError(
                f"Memmap not found:\n"
                f"{MEMMAP_FILE}"
            )

        # ----------------------------------------------------
        # Frame index
        # ----------------------------------------------------

        if not FRAME_INDEX_FILE.exists():

            raise FileNotFoundError(
                f"Frame index not found:\n"
                f"{FRAME_INDEX_FILE}"
            )

        with open(
            FRAME_INDEX_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.frame_index = json.load(f)

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Do not open memmap immediately.
        #
        # This makes the Dataset safer when DataLoader uses
        # multiple workers.
        # ----------------------------------------------------

        self.rainfall = None

        # ----------------------------------------------------
        # Precompute sequence indices
        #
        # Instead of parsing filenames during every
        # __getitem__ call, convert everything once.
        # ----------------------------------------------------

        print(
            f"{split.upper()} dataset loaded: "
            f"{len(self.sequences):,} sequences"
        )

        print(
            "Building fast sequence index..."
        )

        self.sequence_indices = (
            self._build_sequence_indices()
        )

        print(
            f"Indexed sequences: "
            f"{len(self.sequence_indices):,}"
        )

    # ========================================================
    # OPEN MEMMAP
    # ========================================================

    def _ensure_memmap(self):

        if self.rainfall is None:

            self.rainfall = np.load(
                MEMMAP_FILE,
                mmap_mode="r"
            )

            if self.rainfall.ndim != 3:

                raise ValueError(
                    "Unexpected memmap dimensions: "
                    f"{self.rainfall.shape}"
                )

            if self.rainfall.shape[1:] != (
                HEIGHT,
                WIDTH
            ):

                raise ValueError(
                    "Unexpected memmap grid shape: "
                    f"{self.rainfall.shape}"
                )

        return self.rainfall

    # ========================================================
    # BUILD FAST SEQUENCE INDEX
    # ========================================================

    def _build_sequence_indices(self):

        indexed_sequences = []

        missing = []

        total = len(
            self.sequences
        )

        for sequence_number, sequence in enumerate(
            self.sequences
        ):

            input_entries = (
                sequence["input"]
            )

            target_entries = (
                sequence["target"]
            )

            # ------------------------------------------------
            # Input
            # ------------------------------------------------

            input_indices = []

            for entry in input_entries:

                timestamp = normalize_timestamp(
                    entry
                )

                if timestamp is None:

                    missing.append(
                        entry
                    )

                    continue

                if timestamp not in self.frame_index:

                    missing.append(
                        entry
                    )

                    continue

                input_indices.append(
                    self.frame_index[
                        timestamp
                    ]
                )

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            target_indices = []

            for entry in target_entries:

                timestamp = normalize_timestamp(
                    entry
                )

                if timestamp is None:

                    missing.append(
                        entry
                    )

                    continue

                if timestamp not in self.frame_index:

                    missing.append(
                        entry
                    )

                    continue

                target_indices.append(
                    self.frame_index[
                        timestamp
                    ]
                )

            # ------------------------------------------------
            # Validate sequence length
            # ------------------------------------------------

            if len(input_indices) != INPUT_FRAMES:

                raise RuntimeError(
                    "Invalid input sequence at "
                    f"sequence {sequence_number}: "
                    f"expected {INPUT_FRAMES}, "
                    f"got {len(input_indices)}"
                )

            if len(target_indices) != OUTPUT_FRAMES:

                raise RuntimeError(
                    "Invalid target sequence at "
                    f"sequence {sequence_number}: "
                    f"expected {OUTPUT_FRAMES}, "
                    f"got {len(target_indices)}"
                )

            indexed_sequences.append(
                (
                    np.asarray(
                        input_indices,
                        dtype=np.int32
                    ),
                    np.asarray(
                        target_indices,
                        dtype=np.int32
                    )
                )
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (sequence_number + 1) % 10000 == 0
                or sequence_number == total - 1
            ):

                print(
                    f"\rIndexing: "
                    f"{sequence_number + 1:,}/"
                    f"{total:,}",
                    end=""
                )

        print()

        # ----------------------------------------------------
        # Missing entries
        # ----------------------------------------------------

        if missing:

            preview = missing[:5]

            raise RuntimeError(
                f"Could not resolve "
                f"{len(missing):,} sequence entries.\n"
                f"Examples:\n"
                + "\n".join(
                    str(x)
                    for x in preview
                )
            )

        return indexed_sequences

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.sequence_indices
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        idx
    ):

        rainfall = (
            self._ensure_memmap()
        )

        input_indices, target_indices = (
            self.sequence_indices[idx]
        )

        # ----------------------------------------------------
        # Fast memmap access
        # ----------------------------------------------------

        input_frames = np.asarray(
            rainfall[input_indices],
            dtype=np.float32
        )

        target_frames = np.asarray(
            rainfall[target_indices],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if input_frames.shape != (
            INPUT_FRAMES,
            HEIGHT,
            WIDTH
        ):

            raise ValueError(
                f"Unexpected input shape: "
                f"{input_frames.shape}"
            )

        if target_frames.shape != (
            OUTPUT_FRAMES,
            HEIGHT,
            WIDTH
        ):

            raise ValueError(
                f"Unexpected target shape: "
                f"{target_frames.shape}"
            )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not np.isfinite(
            input_frames
        ).all():

            raise ValueError(
                "Input contains NaN/Inf"
            )

        if not np.isfinite(
            target_frames
        ).all():

            raise ValueError(
                "Target contains NaN/Inf"
            )

        if (
            input_frames < 0
        ).any():

            raise ValueError(
                "Input contains negative rainfall"
            )

        if (
            target_frames < 0
        ).any():

            raise ValueError(
                "Target contains negative rainfall"
            )

        # ----------------------------------------------------
        # log1p
        # ----------------------------------------------------

        if self.transform:

            input_frames = np.log1p(
                input_frames
            )

            target_frames = np.log1p(
                target_frames
            )

        # ----------------------------------------------------
        # Add channel
        #
        # (T,H,W)
        # ->
        # (T,H,W,C)
        # ----------------------------------------------------

        input_frames = (
            input_frames[..., None]
        )

        target_frames = (
            target_frames[..., None]
        )

        # ----------------------------------------------------
        # NumPy → PyTorch
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            input_frames
        ).float()

        target_tensor = torch.from_numpy(
            target_frames
        ).float()

        return (
            input_tensor,
            target_tensor
        )


# ============================================================
# DATASET TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL A — MEMMAP DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = IMERGSequenceDataset(
        split="train",
        transform=True
    )

    print()
    print(
        "Dataset length:",
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Single sample
    # --------------------------------------------------------

    x, y = dataset[0]

    print()
    print("SINGLE SAMPLE")

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
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    batch_x, batch_y = next(
        iter(loader)
    )

    print()
    print("DATALOADER BATCH")

    print(
        "Input shape :",
        tuple(batch_x.shape)
    )

    print(
        "Target shape:",
        tuple(batch_y.shape)
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert batch_x.shape == (
        2,
        INPUT_FRAMES,
        HEIGHT,
        WIDTH,
        1
    )

    assert batch_y.shape == (
        2,
        OUTPUT_FRAMES,
        HEIGHT,
        WIDTH,
        1
    )

    assert torch.isfinite(
        batch_x
    ).all()

    assert torch.isfinite(
        batch_y
    ).all()

    assert (
        batch_x.min().item()
        >= 0
    )

    assert (
        batch_y.min().item()
        >= 0
    )

    print()
    print("=" * 70)
    print(
        "STATUS: MEMMAP DATASET TEST PASSED"
    )
    print("=" * 70)

    # Explicit cleanup
    del loader
    del dataset
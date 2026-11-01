from pathlib import Path
import shutil


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PACKAGE_DIR = (
    PROJECT_ROOT
    / "colab_model_a"
)


# ============================================================
# SOURCE PATHS
# ============================================================

MEMMAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "memmap"
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)


# ============================================================
# FILE COPY HELPER
# ============================================================

def copy_file(
    source,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"COPY  {source.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL A — COLAB PACKAGE BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Create package
    # --------------------------------------------------------

    PACKAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("Preparing training data...")

    copy_file(
        MEMMAP_DIR / "imerg_rainfall.npy",
        PACKAGE_DIR
        / "data"
        / "imerg_rainfall.npy"
    )

    copy_file(
        MEMMAP_DIR / "frame_index.json",
        PACKAGE_DIR
        / "data"
        / "frame_index.json"
    )

    # --------------------------------------------------------
    # ONLY TRAIN + VALIDATION
    #
    # test.json intentionally excluded
    # --------------------------------------------------------

    copy_file(
        SEQUENCE_DIR / "train.json",
        PACKAGE_DIR
        / "data"
        / "sequences"
        / "train.json"
    )

    copy_file(
        SEQUENCE_DIR / "validation.json",
        PACKAGE_DIR
        / "data"
        / "sequences"
        / "validation.json"
    )

    # --------------------------------------------------------
    # CODE
    # --------------------------------------------------------

    print()
    print("Preparing source code...")

    copy_file(
        SRC_DIR / "data" / "dataset.py",
        PACKAGE_DIR
        / "src"
        / "data"
        / "dataset.py"
    )

    copy_file(
        SRC_DIR / "models" / "convlstm.py",
        PACKAGE_DIR
        / "src"
        / "models"
        / "convlstm.py"
    )

    copy_file(
        SRC_DIR / "training" / "train_convlstm.py",
        PACKAGE_DIR
        / "src"
        / "training"
        / "train_convlstm.py"
    )

    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

    requirements = """numpy
torch
"""

    requirements_file = (
        PACKAGE_DIR
        / "requirements.txt"
    )

    requirements_file.write_text(
        requirements,
        encoding="utf-8"
    )

    print(
        "CREATE requirements.txt"
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme = """# Model A — IMERG ConvLSTM

## Dataset

Historical IMERG rainfall.

Training:
2008–2020

Validation:
2021–2024

Final test:
2025

The 2025 test split is intentionally NOT included
in this Colab training package.

## Input

12 rainfall frames
30-minute resolution

Total history:
6 hours

Grid:
20 × 30

## Output

12 future rainfall frames

Forecast horizon:
6 hours

Resolution:
30 minutes

## Transformation

Training uses:

log1p(rainfall)

Predictions are converted back using:

expm1(prediction)

## Model

Temporal ConvLSTM encoder-decoder.

## Training

Google Colab GPU.
"""

    readme_file = (
        PACKAGE_DIR
        / "README.md"
    )

    readme_file.write_text(
        readme,
        encoding="utf-8"
    )

    print(
        "CREATE README.md"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLAB PACKAGE READY")
    print("=" * 70)

    print()
    print(
        "Package:",
        PACKAGE_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "2025 test data was NOT copied."
    )

    print()
    print(
        "Next step: ZIP this package and upload "
        "it to Google Colab."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
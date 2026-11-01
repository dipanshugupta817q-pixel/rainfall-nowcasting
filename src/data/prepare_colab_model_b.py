from pathlib import Path
import shutil


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "colab_model_b"

ERA5_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "era5_aligned"
)

SEQUENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "era5_sequences"
)

STATS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "era5_era5_stats.npz"
)

SOURCE_FILES = {
    "dataset.py":
        PROJECT_ROOT / "src" / "data" / "era5_dataset.py",

    "era5_encoder.py":
        PROJECT_ROOT / "src" / "models" / "era5_encoder.py",

    "era5_model.py":
        PROJECT_ROOT / "src" / "models" / "era5_model.py",

    "train_era5.py":
        PROJECT_ROOT / "src" / "training" / "train_era5.py",
}


# ============================================================
# RESET OUTPUT
# ============================================================

if OUTPUT_ROOT.exists():

    print("Removing old package...")

    shutil.rmtree(
        OUTPUT_ROOT
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DATA_DIR = OUTPUT_ROOT / "data"

ALIGNED_DIR = (
    DATA_DIR / "era5_aligned"
)

SEQUENCE_DIR = (
    DATA_DIR / "era5_sequences"
)

SRC_DATA_DIR = (
    OUTPUT_ROOT / "src" / "data"
)

SRC_MODELS_DIR = (
    OUTPUT_ROOT / "src" / "models"
)

SRC_TRAINING_DIR = (
    OUTPUT_ROOT / "src" / "training"
)

ALIGNED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEQUENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SRC_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SRC_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SRC_TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY ONLY 2021–2024 ERA5
#
# 2025 = FINAL TEST → DO NOT COPY
# ============================================================

print("=" * 60)
print("MODEL B — COLAB PACKAGE BUILDER")
print("=" * 60)

era5_files = []

for file in sorted(
    ERA5_ROOT.glob("*.npz")
):

    year = int(
        file.stem[:4]
    )

    if 2021 <= year <= 2024:

        era5_files.append(file)


print()
print(
    "ERA5 training/validation files:",
    len(era5_files)
)

if len(era5_files) != 488:

    raise RuntimeError(
        "Expected 488 files for 2021–2024, "
        f"found {len(era5_files)}"
    )


# ============================================================
# COPY ERA5 DATA
# ============================================================

print()
print("COPY ERA5 ALIGNED DATA")

for i, file in enumerate(
    era5_files,
    1
):

    shutil.copy2(
        file,
        ALIGNED_DIR / file.name
    )

    if i % 50 == 0 or i == len(era5_files):

        print(
            f"  {i}/{len(era5_files)}"
        )


# ============================================================
# COPY TRAIN + VALIDATION SEQUENCES
#
# DO NOT COPY test.json
# ============================================================

print()
print("COPY SEQUENCE FILES")

for split in [
    "train",
    "validation"
]:

    source = (
        SEQUENCE_ROOT
        / f"{split}.json"
    )

    destination = (
        SEQUENCE_DIR
        / f"{split}.json"
    )

    if not source.exists():

        raise FileNotFoundError(
            source
        )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"  COPY {split}.json"
    )


# ============================================================
# COPY NORMALIZATION STATISTICS
# ============================================================

print()
print("COPY NORMALIZATION STATISTICS")

if not STATS_FILE.exists():

    raise FileNotFoundError(
        STATS_FILE
    )

shutil.copy2(
    STATS_FILE,
    DATA_DIR / STATS_FILE.name
)

print(
    f"  COPY {STATS_FILE.name}"
)


# ============================================================
# COPY SOURCE CODE
# ============================================================

print()
print("COPY SOURCE CODE")

for name, source in SOURCE_FILES.items():

    if not source.exists():

        raise FileNotFoundError(
            source
        )

    if name == "dataset.py":

        destination = (
            SRC_DATA_DIR / name
        )

    elif name in [
        "era5_encoder.py",
        "era5_model.py"
    ]:

        destination = (
            SRC_MODELS_DIR / name
        )

    else:

        destination = (
            SRC_TRAINING_DIR / name
        )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"  COPY {name}"
    )


# ============================================================
# CREATE INIT FILES
# ============================================================

for directory in [
    OUTPUT_ROOT / "src",
    SRC_DATA_DIR,
    SRC_MODELS_DIR,
    SRC_TRAINING_DIR,
]:

    init_file = (
        directory / "__init__.py"
    )

    init_file.touch()


# ============================================================
# REQUIREMENTS
# ============================================================

requirements = """numpy
torch
"""

(
    OUTPUT_ROOT / "requirements.txt"
).write_text(
    requirements,
    encoding="utf-8"
)


# ============================================================
# README
# ============================================================

readme = """# Model B — ERA5 Atmospheric Forecasting

Deep-learning atmospheric forecasting model for the rainfall nowcasting system.

## Data

Input:
- 12 ERA5 frames
- 30-minute resolution
- 6-hour history
- 7 atmospheric variables
- 20 × 30 spatial grid

Output:
- 12 future frames
- next 6 hours
- 7 atmospheric variables

## Variables

1. t2m
2. d2m
3. u10
4. v10
5. sp
6. msl
7. tcc

## Dataset split

TRAIN:
2021–2023

VALIDATION:
2024

FINAL TEST:
2025

2025 test data is intentionally NOT included in this package.

## Model

ConvLSTM encoder + autoregressive ConvLSTM decoder.

## Training

Run:

python src/training/train_era5.py
"""

(
    OUTPUT_ROOT / "README.md"
).write_text(
    readme,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("COLAB MODEL B PACKAGE READY")
print("=" * 60)

print(
    "Package:",
    OUTPUT_ROOT
)

print()
print("Included:")
print("  ERA5 2021–2024 : 488 files")
print("  train.json")
print("  validation.json")
print("  normalization statistics")
print("  era5_dataset.py")
print("  era5_encoder.py")
print("  era5_model.py")
print("  train_era5.py")

print()
print("EXCLUDED:")
print("  test.json")
print("  ERA5 2025 test data")

print()
print(
    "STATUS: COLAB MODEL B PACKAGE READY"
)
print("=" * 60)
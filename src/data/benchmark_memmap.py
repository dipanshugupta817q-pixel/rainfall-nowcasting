import sys
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader


# ============================================================
# PATH
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

NUM_BATCHES = 100


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMERG MEMMAP PERFORMANCE BENCHMARK")
    print("=" * 70)

    print()
    print("Batch size :", BATCH_SIZE)
    print("Workers    :", NUM_WORKERS)
    print("Batches    :", NUM_BATCHES)

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
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print("Warm-up...")

    iterator = iter(loader)

    for _ in range(3):

        x, y = next(iterator)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("Benchmarking...")

    start = time.perf_counter()

    total_sequences = 0

    total_batches = 0

    iterator = iter(loader)

    while total_batches < NUM_BATCHES:

        try:

            x, y = next(iterator)

        except StopIteration:

            break

        # Force access to tensors
        # so lazy operations cannot
        # remain deferred.

        _ = x.sum().item()
        _ = y.sum().item()

        total_sequences += x.shape[0]

        total_batches += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    batches_per_second = (
        total_batches / elapsed
    )

    sequences_per_second = (
        total_sequences / elapsed
    )

    avg_batch_time = (
        elapsed / total_batches
    )

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Total batches       : "
        f"{total_batches}"
    )

    print(
        f"Total sequences     : "
        f"{total_sequences:,}"
    )

    print(
        f"Total time          : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Average batch time  : "
        f"{avg_batch_time:.4f} sec"
    )

    print(
        f"Batches / second    : "
        f"{batches_per_second:.2f}"
    )

    print(
        f"Sequences / second  : "
        f"{sequences_per_second:.2f}"
    )

    # --------------------------------------------------------
    # Estimated epoch time
    # --------------------------------------------------------

    total_sequences_dataset = len(
        dataset
    )

    estimated_epoch_seconds = (
        total_sequences_dataset
        / sequences_per_second
    )

    print(
        f"Estimated 1 epoch   : "
        f"{estimated_epoch_seconds / 60:.2f} min"
    )

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    print()
    print(
        "Final batch input shape :",
        tuple(x.shape)
    )

    print(
        "Final batch target shape:",
        tuple(y.shape)
    )

    assert x.shape[1:] == (
        12,
        20,
        30,
        1
    )

    assert y.shape[1:] == (
        12,
        20,
        30,
        1
    )

    print()
    print("=" * 70)
    print(
        "STATUS: MEMMAP PERFORMANCE BENCHMARK PASSED"
    )
    print("=" * 70)

    # Explicitly release memmap reference
    del dataset


if __name__ == "__main__":

    main()
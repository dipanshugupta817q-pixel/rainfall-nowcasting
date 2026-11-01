from pathlib import Path
import numpy as np


SEQUENCE_FILE = Path(
    "data/processed/sequences/20240815_0600.npz"
)


def main():

    print("=" * 70)
    print("EVENT SEQUENCE VALIDATION")
    print("=" * 70)

    assert SEQUENCE_FILE.exists(), (
        f"Missing file: {SEQUENCE_FILE}"
    )

    data = np.load(
        SEQUENCE_FILE
    )

    X = data["X"]
    Y = data["Y"]
    timestamps = data["timestamps"]

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert X.shape == (
        12, 20, 30
    )

    assert Y.shape == (
        12, 20, 30
    )

    assert timestamps.shape == (
        24,
    )

    # --------------------------------------------------------
    # Temporal checks
    # --------------------------------------------------------

    time_diffs = np.diff(
        timestamps
    )

    assert np.all(
        time_diffs == 1800
    ), (
        "Frames are not exactly "
        "30 minutes apart."
    )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    assert np.isfinite(X).all()
    assert np.isfinite(Y).all()

    assert np.all(X >= 0)
    assert np.all(Y >= 0)

    # --------------------------------------------------------
    # Input/target continuity
    # --------------------------------------------------------

    assert (
        timestamps[11]
        + 1800
        == timestamps[12]
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"Y shape       : {Y.shape}"
    )

    print(
        f"Timestamp diff: "
        f"{np.unique(time_diffs)} seconds"
    )

    print(
        f"X min/max     : "
        f"{X.min():.4f} / {X.max():.4f}"
    )

    print(
        f"Y min/max     : "
        f"{Y.min():.4f} / {Y.max():.4f}"
    )

    print(
        f"NaN X         : "
        f"{np.isnan(X).sum()}"
    )

    print(
        f"NaN Y         : "
        f"{np.isnan(Y).sum()}"
    )

    print("\nALL EVENT SEQUENCE CHECKS PASSED.")


if __name__ == "__main__":
    main()
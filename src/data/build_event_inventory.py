from pathlib import Path
from datetime import datetime, timedelta
import csv


# ============================================================
# CONFIG
# ============================================================

REGION = {
    "lat_min": 28.0,
    "lat_max": 30.0,
    "lon_min": 76.0,
    "lon_max": 79.0,
}

OUTPUT_FILE = Path("data/processed/event_inventory.csv")


# Candidate seasons
YEARS = [2022, 2023, 2024, 2025]
MONTHS = [6, 7, 8, 9]


# ============================================================
# EVENT WINDOW
# ============================================================

# Each candidate event will eventually represent
# a continuous temporal window.

INPUT_HOURS = 6
FORECAST_HOURS = 6

TOTAL_WINDOW_HOURS = INPUT_HOURS + FORECAST_HOURS


# ============================================================
# PLACEHOLDER EVENT RECORD
# ============================================================

def create_candidate_dates():

    dates = []

    for year in YEARS:

        for month in MONTHS:

            # Generate one candidate per day.
            # Actual rainfall intensity will be evaluated
            # after IMERG data acquisition.

            current = datetime(
                year,
                month,
                1,
                0,
                0
            )

            while current.month == month:

                dates.append(current)

                current += timedelta(days=1)

    return dates


# ============================================================
# SAVE INVENTORY
# ============================================================

def save_inventory(dates):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "candidate_date",
            "year",
            "month",
            "season",
            "status",
            "notes"
        ])

        for dt in dates:

            season = (
                "monsoon"
                if dt.month in [6, 7, 8, 9]
                else "other"
            )

            writer.writerow([
                dt.strftime("%Y-%m-%d"),
                dt.year,
                dt.month,
                season,
                "candidate",
                "rainfall_intensity_pending"
            ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dates = create_candidate_dates()

    save_inventory(dates)

    print("=" * 60)
    print("EVENT INVENTORY CREATED")
    print("=" * 60)

    print(f"Candidate dates : {len(dates)}")
    print(f"Output          : {OUTPUT_FILE}")

    print("\nNext step:")
    print("Evaluate candidate dates using IMERG rainfall data.")
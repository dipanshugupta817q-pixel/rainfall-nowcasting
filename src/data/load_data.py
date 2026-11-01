from pathlib import Path

import h5py
import numpy as np


FILL_VALUE = -9999.9


def load_imerg_file(file_path):
    """
    Load one NASA IMERG HDF5 file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"IMERG file not found: {file_path}"
        )

    with h5py.File(file_path, "r") as f:

        grid = f["Grid"]

        precipitation = grid["precipitation"][:]
        lat = grid["lat"][:]
        lon = grid["lon"][:]
        time = grid["time"][:]
        quality = grid["precipitationQualityIndex"][:]
        random_error = grid["randomError"][:]

    # Convert fill values to NaN
    precipitation = precipitation.astype(np.float32)
    precipitation[precipitation <= FILL_VALUE + 1] = np.nan

    quality = quality.astype(np.float32)
    quality[quality <= FILL_VALUE + 1] = np.nan

    random_error = random_error.astype(np.float32)
    random_error[random_error <= FILL_VALUE + 1] = np.nan

    return {
        "precipitation": precipitation,
        "lat": lat,
        "lon": lon,
        "time": time,
        "quality": quality,
        "random_error": random_error,
    }


def crop_region(
    data,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    """
    Crop IMERG data to a geographic region.

    Input rainfall:
        (time, lon, lat)

    Output rainfall:
        (time, lat, lon)
    """

    precipitation = data["precipitation"]

    latitude = np.asarray(data["lat"])
    longitude = np.asarray(data["lon"])

    latitude = np.squeeze(latitude)
    longitude = np.squeeze(longitude)

    if latitude.ndim != 1:
        raise ValueError(
            f"Latitude must be 1-D, got shape {latitude.shape}"
        )

    if longitude.ndim != 1:
        raise ValueError(
            f"Longitude must be 1-D, got shape {longitude.shape}"
        )

    # Find latitude indices
    lat_idx = np.where(
        (latitude >= lat_min)
        & (latitude <= lat_max)
    )[0]

    # Find longitude indices
    lon_idx = np.where(
        (longitude >= lon_min)
        & (longitude <= lon_max)
    )[0]

    if len(lat_idx) == 0:
        raise ValueError(
            f"No latitude points found between "
            f"{lat_min} and {lat_max}"
        )

    if len(lon_idx) == 0:
        raise ValueError(
            f"No longitude points found between "
            f"{lon_min} and {lon_max}"
        )

    # IMERG:
    # (time, lon, lat)

    cropped = precipitation[
        :,
        lon_idx,
        :
    ]

    cropped = cropped[
        :,
        :,
        lat_idx
    ]

    # Convert:
    # (time, lon, lat)
    # ->
    # (time, lat, lon)

    cropped = np.transpose(
        cropped,
        (0, 2, 1)
    )

    return {
        "precipitation": cropped,
        "lat": latitude[lat_idx],
        "lon": longitude[lon_idx],
        "time": data["time"],
        "quality": data["quality"],
        "random_error": data["random_error"],
    }
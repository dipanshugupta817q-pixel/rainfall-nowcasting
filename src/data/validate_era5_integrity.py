from pathlib import Path
import numpy as np

ROOT = Path("data/processed/era5_aligned")

files = sorted(ROOT.glob("*.npz"))

bad = []

for i, file in enumerate(files, 1):

    try:
        with np.load(file) as z:

            data = z["data"]

            if data.shape != (48, 7, 20, 30):
                bad.append(
                    (file.name, f"wrong shape: {data.shape}")
                )
                continue

            if not np.isfinite(data).all():
                bad.append(
                    (file.name, "NaN/Inf detected")
                )
                continue

    except Exception as e:

        bad.append(
            (file.name, f"read error: {e}")
        )

    if i % 100 == 0:
        print(f"Checked: {i}/{len(files)}")


print()
print("=" * 60)
print("ERA5 INTEGRITY AUDIT")
print("=" * 60)

print("Files checked :", len(files))
print("Bad files     :", len(bad))

if bad:

    print("\nBAD FILES:")

    for item in bad[:20]:
        print(item)

else:

    print("Bad files     : NONE")

print("=" * 60)

if len(files) == 610 and not bad:
    print("STATUS: ERA5 INTEGRITY PASSED")
else:
    print("STATUS: ERA5 INTEGRITY FAILED")
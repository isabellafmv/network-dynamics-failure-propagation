import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# ---------- CONFIG ----------

# Folder with all .dly files (change if needed)
DATA_DIR = Path("/Users/isabellamueller-vogt/Downloads/ghcnd_all/ghcnd_all")

# Output CSV path (will be created or overwritten)
OUTPUT_FILE = Path("ghcn_clean_small.csv")

# Elements to keep
KEEP_ELEMENTS = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD"}

# Number of .dly files (stations) to sample at random
# Set to None to use ALL files
SAMPLE_SIZE = 2_000

# Random seed for reproducibility
RANDOM_SEED = 42

# ----------------------------


def parse_dly(file_path):
    """
    Parse a single .dly file and return a list of [station, date, variable, value].
    """
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            station = line[0:11]
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21]

            # Only keep selected variables
            if element not in KEEP_ELEMENTS:
                continue

            # Each day is an 8-character field: value(5) mflag(1) qflag(1) sflag(1)
            for day in range(1, 32):
                pos = 21 + (day - 1) * 8
                value_str = line[pos:pos+5]

                # If there is no value (end of line / malformed), skip
                if not value_str.strip():
                    continue

                value = int(value_str)
                qflag = line[pos+6]  # 7th char in the 8-char block

                # Missing value
                if value == -9999:
                    continue

                # Skip if quality flag is not blank (failed or suspect QC)
                if qflag.strip():
                    continue

                # Build date string
                date = f"{year}-{month:02d}-{day:02d}"
                rows.append([station, date, element, value])

    return rows


def main():
    # List all .dly files
    dly_files = list(DATA_DIR.glob("*.dly"))
    if not dly_files:
        print(f"No .dly files found in {DATA_DIR}")
        return

    print(f"Found {len(dly_files)} .dly files in {DATA_DIR}")

    # Random sampling of files (stations)
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(dly_files):
        random.seed(RANDOM_SEED)
        dly_files = random.sample(dly_files, SAMPLE_SIZE)
        print(f"Randomly selected {len(dly_files)} files to process")
    else:
        print("Processing all files (no sampling)")

    first_write = True

    for dly_file in tqdm(dly_files):
        data = parse_dly(dly_file)

        if not data:
            continue

        df = pd.DataFrame(data, columns=["station", "date", "variable", "value"])

        # Make "value" float so unit conversions (divide by 10) are safe
        df["value"] = df["value"].astype("float32")

        # Temperatures: tenths of °C -> °C
        temp_mask = df["variable"].isin(["TMAX", "TMIN"])
        df.loc[temp_mask, "value"] = df.loc[temp_mask, "value"] / 10.0

        # Precip & snow: tenths of mm -> mm
        precip_mask = df["variable"].isin(["PRCP", "SNOW", "SNWD"])
        df.loc[precip_mask, "value"] = df.loc[precip_mask, "value"] / 10.0

        # Append to CSV
        df.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )

        first_write = False

    print("✅ Finished: ghcn_clean_small.csv created")


if __name__ == "__main__":
    main()

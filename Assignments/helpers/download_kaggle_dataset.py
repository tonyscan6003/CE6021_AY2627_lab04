#!/usr/bin/env python3
"""download_kaggle_dataset.py — Download and extract a Kaggle dataset.

Usage (from any directory):
    python Assignments/helpers/download_kaggle_dataset.py

Credentials are picked up in this order:
    1. KAGGLE_API_TOKEN environment variable — already set, nothing to do.
    2. Data/access_token   — new-style Kaggle API token (plain text file,
       generated from https://www.kaggle.com/settings/api, "Generate New
       Token"). Copied to ~/.kaggle/access_token.
    3. Data/kaggle.json    — legacy username/key credentials ("Create Legacy
       API Key" on the same settings page). Copied to ~/.kaggle/kaggle.json.

Dataset downloaded: rhammell/ships-in-satellite-imagery
Extracted to:       Data/shipsnet/shipsnet/*.png  (positive: 1_*.png, negative: 0_*.png)
"""
import os
import shutil
import stat
import subprocess
import sys
import zipfile

# ── Locate paths ────────────────────────────────────────────────────────────
_here    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(_here)), "Data")

TOKEN_SRC        = os.path.join(DATA_DIR, "access_token")
KAGGLE_JSON_SRC  = os.path.join(DATA_DIR, "kaggle.json")
KAGGLE_DIR       = os.path.expanduser("~/.kaggle")
TOKEN_DEST       = os.path.join(KAGGLE_DIR, "access_token")
KAGGLE_JSON_DEST = os.path.join(KAGGLE_DIR, "kaggle.json")

DATASET_SLUG = "rhammell/ships-in-satellite-imagery"
ZIP_NAME     = "ships-in-satellite-imagery.zip"
ZIP_PATH     = os.path.join(DATA_DIR, ZIP_NAME)


def _install(src, dest):
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    shutil.copy(src, dest)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)  # chmod 600
    print(f"Kaggle credentials configured from {src}")


def setup_credentials():
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("Using Kaggle API token from KAGGLE_API_TOKEN environment variable.")
        return

    if os.path.exists(TOKEN_SRC):
        _install(TOKEN_SRC, TOKEN_DEST)
        return

    if os.path.exists(KAGGLE_JSON_SRC):
        _install(KAGGLE_JSON_SRC, KAGGLE_JSON_DEST)
        return

    print(f"ERROR: no Kaggle credentials found in {DATA_DIR}")
    print("Get a token from https://www.kaggle.com/settings/api and either:")
    print(f"  - place the API token in {TOKEN_SRC} (new: 'Generate New Token'), or")
    print(f"  - place kaggle.json in {KAGGLE_JSON_SRC} (legacy: 'Create Legacy API Key'), or")
    print("  - export KAGGLE_API_TOKEN=<your-token>")
    sys.exit(1)


def download_dataset():
    # Check if already extracted
    expected = os.path.join(DATA_DIR, "shipsnet", "shipsnet")
    if os.path.isdir(expected) and any(
        f.startswith("1_") for f in os.listdir(expected)
    ):
        print(f"Dataset already extracted at {expected} — skipping download.")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset '{DATASET_SLUG}' to {DATA_DIR} …")
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET_SLUG, "-p", DATA_DIR],
            check=True,
        )
        print("Download complete.")
    else:
        print(f"Zip already present at {ZIP_PATH} — skipping download.")

    print(f"Extracting {ZIP_NAME} to {DATA_DIR} …")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extraction complete. Images are in {expected}")


if __name__ == "__main__":
    setup_credentials()
    download_dataset()

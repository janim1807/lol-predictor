"""
refresh_data.py — Download the latest Oracle's Elixir CSVs and bust the Parquet cache.

Setup (one-time):
  1. Go to https://oracleselixir.com/tools/downloads
  2. Right-click the 2026 download link → Copy link address
  3. Extract the file ID from the URL:
       https://drive.google.com/file/d/<FILE_ID>/view  →  paste FILE_ID below
  4. Run:  uv run python refresh_data.py

Daily use:
  uv run python refresh_data.py          # update CSVs
  streamlit run app.py                   # restart app to pick up new data

Scheduling (Windows Task Scheduler / cron):
  # cron: run at 04:00 every day
  0 4 * * * cd /path/to/LoL && .venv/bin/python refresh_data.py
"""

import os
import sys
from pathlib import Path

# Google Drive file IDs from oracleselixir.com/tools/downloads
# The ID is the long string between /d/ and /view in the Drive URL.
DRIVE_FILE_IDS: dict[int, str] = {
    2026: "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH",
    # 2025: "REPLACE_WITH_2025_FILE_ID",
}

CSV_DIR = Path(__file__).parent / "src" / "data" / "oracleselixir"
CACHE   = CSV_DIR.parent / "game_df_cache.parquet"


def _gdrive_download(file_id: str, dest: Path) -> None:
    """Download a public Google Drive file, handling the large-file confirmation redirect."""
    import requests

    session = requests.Session()
    # First request — may redirect to a virus-scan confirmation page for large files
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    resp = session.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"\r  {total / 1e6:.1f} MB downloaded…", end="", flush=True)
    print()


def download_year(year: int, file_id: str) -> bool:
    """Download one year's CSV from Google Drive. Returns True if updated."""
    if file_id.startswith("REPLACE_"):
        print(f"  [{year}] Skipped — file ID not configured.")
        return False

    dest = CSV_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    tmp  = dest.with_suffix(".csv.tmp")

    print(f"  [{year}] Downloading from Google Drive…")
    try:
        _gdrive_download(file_id, tmp)
    except Exception as e:
        print(f"  [{year}] Download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False

    if tmp.stat().st_size < 1_000:
        print(f"  [{year}] Downloaded file looks empty — check the file ID.")
        tmp.unlink(missing_ok=True)
        return False

    # Only replace if different size (avoid unnecessary cache busting)
    if dest.exists() and dest.stat().st_size == tmp.stat().st_size:
        print(f"  [{year}] No change (same size as existing file).")
        tmp.unlink()
        return False

    tmp.replace(dest)
    print(f"  [{year}] Saved → {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def bust_cache():
    if CACHE.exists():
        CACHE.unlink()
        print(f"  Cache deleted → will rebuild on next app start: {CACHE.name}")
    else:
        print("  No cache to delete (will be built fresh on next app start).")


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    print("Oracle's Elixir CSV refresh")
    print("=" * 40)

    updated = False
    for year, fid in DRIVE_FILE_IDS.items():
        if download_year(year, fid):
            updated = True

    if updated:
        print()
        bust_cache()
        print()
        print("Done. Restart the Streamlit app to load the new data.")
    else:
        print()
        print("No files updated. Data is already current (or file IDs need configuring).")


if __name__ == "__main__":
    main()

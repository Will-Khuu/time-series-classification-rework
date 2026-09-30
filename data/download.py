"""Fetch the AReM dataset from UCI into data/AReM and verify it.

Standard library only, so it runs before the project's dependencies are installed.
Safe to rerun: a verified existing copy is left alone.

Usage: python data/download.py
"""

import hashlib
import io
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/366/activity+recognition+system+based+on+multisensor+data+fusion+arem.zip"
SHA256 = "a328bd9f4a4017546c2359b371c1a820441e9e83346922fde3c3b058c7209000"
DEST = Path(__file__).parent / "AReM"

EXPECTED_FILES = {
    "bending1": 7,
    "bending2": 6,
    "cycling": 15,
    "lying": 15,
    "sitting": 15,
    "standing": 15,
    "walking": 15,
}
ROWS_PER_RECORDING = 480
SHORT_RECORDINGS = {"sitting/dataset8.csv": 479}


def problems(root: Path) -> list[str]:
    found = []
    for folder, count in EXPECTED_FILES.items():
        files = sorted((root / folder).glob("dataset*.csv"))
        if len(files) != count:
            found.append(f"{folder}: expected {count} files, found {len(files)}")
        for f in files:
            name = f"{folder}/{f.name}"
            rows = sum(1 for line in f.read_text().splitlines() if line.strip() and not line.startswith("#"))
            expected = SHORT_RECORDINGS.get(name, ROWS_PER_RECORDING)
            if rows != expected:
                found.append(f"{name}: expected {expected} rows, found {rows}")
    return found


def download() -> bytes:
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        sys.exit(f"checksum mismatch: expected {SHA256}, got {digest}")
    return payload


def main() -> None:
    if DEST.exists() and not problems(DEST):
        print(f"{DEST} already present and verified")
        return
    with tempfile.TemporaryDirectory(dir=DEST.parent) as tmp:
        staged = Path(tmp) / DEST.name
        zipfile.ZipFile(io.BytesIO(download())).extractall(staged)
        if found := problems(staged):
            sys.exit("integrity check failed:\n" + "\n".join(found))
        shutil.rmtree(DEST, ignore_errors=True)
        staged.rename(DEST)
    print(f"downloaded and verified {sum(EXPECTED_FILES.values())} recordings into {DEST}")


if __name__ == "__main__":
    main()

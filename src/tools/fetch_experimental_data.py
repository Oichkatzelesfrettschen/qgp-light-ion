#!/usr/bin/env python3
"""
fetch_experimental_data.py

Fetch and verify experimental data from HEPData with checksums.
Supports --verify-only and --update modes for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib


HEPDATA_BASE = "https://www.hepdata.net/download/table"
MANIFEST_PATH = Path(__file__).parent.parent.parent / "experimental" / "FETCH_MANIFEST.toml"
CHECKSUMS_PATH = Path(__file__).parent.parent.parent / "experimental" / "CHECKSUMS.sha256"
DATA_DIR = Path(__file__).parent.parent.parent / "experimental"


def fetch_hepdata_table(
    hepdata_id: str,
    table: str,
    max_retries: int = 3,
    base_delay_s: float = 1.0,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Fetch a HEPData table with exponential backoff on transient errors.

    Retries on: urllib.error.URLError, connection errors, HTTP 429, HTTP 503.
    Raises immediately on: HTTP 404 (not found), HTTP 400 (bad request).

    Args:
        hepdata_id: HEPData record ID (e.g. "ins2825461")
        table: Table identifier (e.g. "Table 1")
        max_retries: Maximum number of retry attempts
        base_delay_s: Base delay for exponential backoff (seconds)
        rng: Random generator for jitter (if None, creates default)

    Returns:
        List of dictionaries parsed from HEPData JSON response
    """
    _rng = rng or random.Random()

    for attempt in range(max_retries + 1):
        try:
            url = f"{HEPDATA_BASE}/{hepdata_id}/{table}/json"
            with urllib.request.urlopen(url, timeout=30) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                values = data.get("values", [])
                return list(values) if values else []
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                raise  # Non-retriable errors
            if attempt == max_retries:
                raise
            print(f"  WARNING: HTTP {exc.code}, retrying...", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError):
            if attempt == max_retries:
                raise
            print("  WARNING: Connection error, retrying...", file=sys.stderr)

        # Exponential backoff with jitter: sleep in [0, base * 2^attempt]
        delay = _rng.uniform(0, base_delay_s * (2 ** attempt))
        time.sleep(delay)

    raise RuntimeError("unreachable")


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: File path

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_dat_file(
    path: Path,
    source: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Write experimental data to .dat file with provenance header.

    Args:
        path: Output file path
        source: Source dictionary from manifest with metadata
        rows: Data rows from HEPData
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # Write provenance header
        f.write("# " + "=" * 77 + "\n")
        f.write(f"#  {source['description']}\n")
        f.write("# " + "=" * 77 + "\n")
        f.write("#\n")
        f.write(f"# DATA TYPE: {source['provenance']}\n")
        if "doi" in source:
            f.write(f"# DOI: {source['doi']}\n")
        if "hepdata_id" in source:
            f.write(f"# HEPData ID: {source['hepdata_id']}\n")
        f.write("#\n")

        # Write column names
        columns = source.get("columns", [])
        f.write("# Columns: " + ", ".join(columns) + "\n")
        f.write("#\n")

        # Write data rows
        for row in rows:
            values = [str(row.get(col, "")) for col in columns]
            f.write("  ".join(values) + "\n")


def verify_checksums(data_dir: Path, checksums_path: Path) -> bool:
    """Verify all files in data_dir match checksums.

    Args:
        data_dir: Directory containing data files
        checksums_path: Path to CHECKSUMS.sha256 file

    Returns:
        True if all checksums match, False otherwise
    """
    if not checksums_path.exists():
        print("ERROR: CHECKSUMS.sha256 not found", file=sys.stderr)
        return False

    with open(checksums_path) as f:
        checksums = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                hash_val = parts[0]
                filename = parts[-1]
                checksums[filename] = hash_val

    all_match = True
    for filename, expected_hash in checksums.items():
        filepath = data_dir / Path(filename).name
        if not filepath.exists():
            print(f"  ERROR: {filepath} not found", file=sys.stderr)
            all_match = False
            continue

        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            print(
                f"  MISMATCH: {filepath}\n"
                f"    Expected: {expected_hash}\n"
                f"    Got:      {actual_hash}",
                file=sys.stderr,
            )
            all_match = False
        else:
            print(f"  OK: {filepath.name}")

    return all_match


def main(argv: list[str] | None = None) -> None:
    """Main entry point for fetch_experimental_data.

    Args:
        argv: Command-line arguments (if None, uses sys.argv[1:])
    """
    parser = argparse.ArgumentParser(description="Fetch and verify experimental data from HEPData")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not write files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify checksums, do not fetch",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-generate CHECKSUMS.sha256 after fetching",
    )

    args = parser.parse_args(argv)

    if args.verify_only:
        print("Verifying experimental data checksums...")
        success = verify_checksums(DATA_DIR, CHECKSUMS_PATH)
        sys.exit(0 if success else 1)

    # Load manifest
    print("Loading fetch manifest...")
    with open(MANIFEST_PATH, "rb") as f:
        manifest = tomllib.load(f)

    sources = manifest.get("source", [])
    print(f"Found {len(sources)} sources in manifest")

    # Fetch each source
    for source in sources:
        filename = source["filename"]
        hepdata_id = source["hepdata_id"]
        table = source["table"]

        print(f"\nFetching {filename} (HEPData {hepdata_id})...")

        try:
            rows = fetch_hepdata_table(hepdata_id, table)
            print(f"  Fetched {len(rows)} rows")

            if not args.dry_run:
                output_path = DATA_DIR / filename
                write_dat_file(output_path, source, rows)
                print(f"  Wrote {output_path}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if args.update:
        print("\nRegenerating CHECKSUMS.sha256...")
        with open(CHECKSUMS_PATH, "w") as f:
            for dat_file in sorted(DATA_DIR.glob("*.dat")):
                hash_val = compute_sha256(dat_file)
                f.write(f"{hash_val}  {dat_file.relative_to(DATA_DIR.parent)}\n")
        print(f"Wrote {CHECKSUMS_PATH}")

    print("\nDone!")


if __name__ == "__main__":
    main()

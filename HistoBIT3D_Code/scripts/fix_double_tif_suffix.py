#!/usr/bin/env python3
"""
Rename images that accidentally end with ".tif.tif" to a single ".tif".

Example:
  MUSE_..._img=0_P=1.tif.tif  ->  MUSE_..._img=0_P=1.tif

Safe-by-default:
  - Dry-run is the default (prints what it *would* do)
  - Pass --apply to actually rename on disk
  - Skips any rename that would overwrite an existing file
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _iter_candidates(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.tif.tif" if recursive else "*.tif.tif"
    return sorted(root.glob(pattern))


def _target_path(src: Path) -> Path:
    # ".tif.tif" -> ".tif"
    # Path.stem would only remove one suffix, so do it explicitly.
    name = src.name
    if not name.lower().endswith(".tif.tif"):
        raise ValueError(f"Not a .tif.tif file: {src}")
    return src.with_name(name[:-4])  # remove the trailing ".tif"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove the extra '.tif' from '*.tif.tif' filenames by renaming on disk."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing images to rename (e.g. .../BIT/trainA).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top-level directory (default scans recursively).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. If omitted, runs as a dry-run.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    recursive = not args.no_recursive
    candidates = _iter_candidates(root, recursive=recursive)

    if not candidates:
        print(f"No '*.tif.tif' files found under: {root}")
        return 0

    dry_run = not args.apply
    print(f"Found {len(candidates)} '*.tif.tif' files under: {root}")
    print("Mode:", "DRY-RUN (no changes)" if dry_run else "APPLY (renaming files)")

    renamed = 0
    skipped = 0

    for src in candidates:
        dst = _target_path(src)

        if dst.exists():
            print(f"[SKIP] target exists: {src} -> {dst}")
            skipped += 1
            continue

        print(f"[RENAME] {src} -> {dst}")
        if not dry_run:
            src.rename(dst)
        renamed += 1

    print(f"Done. Renamed: {renamed}, skipped: {skipped}.")
    if dry_run:
        print("Re-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


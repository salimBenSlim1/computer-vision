#!/usr/bin/env python3
"""
add_coffee.py

Small CLI to add new coffee class names to data.yaml and optionally create
an example label file for a given image so you can test quickly.

Usage:
  python add_coffee.py add Latte
  python add_coffee.py add "Flat White" --example-image coffee_dataset/images/train/cappucino_000.jpg

This script updates `data.yaml` in-place by appending the new class name with the next index.
It will also create any missing folders under `coffee_dataset/images` and `coffee_dataset/labels`.
"""

import argparse
import re
from pathlib import Path

DATA_YAML = Path("data.yaml")
DATASET_DIR = Path("coffee_dataset")


def _read_data_yaml(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_data_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def add_class_to_data_yaml(class_name: str) -> int:
    """Add class_name to data.yaml `names` block and return the new index.

    If class already exists, returns its existing index.
    """
    if not DATA_YAML.exists():
        raise FileNotFoundError("data.yaml not found in project root")

    text = _read_data_yaml(DATA_YAML)

    # Find names block
    m = re.search(r"^names:\n((?:^\s+\d+:.*\n)+)", text, flags=re.MULTILINE)
    if not m:
        # No names block found — append one
        # We'll attempt to add at the end
        lines = text.strip().splitlines()
        lines.append("names:")
        lines.append(f"  0: {class_name}")
        _write_data_yaml(DATA_YAML, "\n".join(lines) + "\n")
        return 0

    names_block = m.group(1)
    # Parse existing names
    existing = {}
    for line in names_block.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        idx_str, name = line.split(":", 1)
        try:
            idx = int(idx_str.strip())
        except ValueError:
            continue
        existing[name.strip()] = idx

    # If exists, return
    if class_name in existing:
        return existing[class_name]

    # Find next index
    if existing:
        max_idx = max(existing.values())
        new_idx = max_idx + 1
    else:
        new_idx = 0

    # Insert new line after the names block
    insert_after = m.end(1)
    new_line = f"  {new_idx}: {class_name}\n"
    new_text = text[:insert_after] + new_line + text[insert_after:]
    _write_data_yaml(DATA_YAML, new_text)

    return new_idx


def ensure_dataset_folders():
    (DATASET_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)


def write_example_label(image_path: Path, class_idx: int) -> Path:
    """Create a small centered label for image so you have at least one label entry.

    Label format: class x_center y_center width height (normalized)
    We'll create a modest centered box covering 30% width and 30% height.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Determine label path
    label_dir = DATASET_DIR / "labels" / ("train" if "train" in str(image_path.parent) else "val")
    if not label_dir.exists():
        # fallback to train
        label_dir = DATASET_DIR / "labels" / "train"
    label_name = image_path.stem + ".txt"
    label_path = label_dir / label_name

    # default normalized box
    x_center = 0.5
    y_center = 0.5
    w = 0.3
    h = 0.3
    line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
    label_path.write_text(line, encoding="utf-8")
    return label_path


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    add = sub.add_parser("add", help="Add a new coffee class")
    add.add_argument("name", help="Name of the coffee class (e.g. Latte)")
    add.add_argument("--example-image", help="Path to an image to create an example label for", default=None)

    args = p.parse_args()
    if args.cmd == "add":
        class_name = args.name.strip()
        ensure_dataset_folders()
        idx = add_class_to_data_yaml(class_name)
        print(f"Added/confirmed class '{class_name}' at index {idx} in data.yaml")
        if args.example_image:
            img = Path(args.example_image)
            try:
                label_path = write_example_label(img, idx)
                print(f"Wrote example label: {label_path}")
            except Exception as e:
                print(f"Warning: could not write example label: {e}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()

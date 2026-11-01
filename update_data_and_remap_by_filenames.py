#!/usr/bin/env python3
"""Auto-update `data.yaml` class names from image filename prefixes and remap label files.

Usage: python update_data_and_remap_by_filenames.py

This script will:
- Read existing `data.yaml` and preserve its classes.
- Scan `coffee_dataset/images/train` and `coffee_dataset/images/val` for filename prefixes.
- Append any new prefixes as class names to `data.yaml` (with safe normalization).
- Back up `data.yaml` and `coffee_dataset/labels` to timestamped backups.
- Remap the first token (class id) in label files to the new class indices based on filename prefix.
- Run a sanity pass and print a summary.
"""
import os
import shutil
import time
import yaml

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_YAML = os.path.join(ROOT, "data.yaml")
IM_DIRS = [os.path.join(ROOT, "coffee_dataset", "images", d) for d in ("train", "val")]
LABEL_DIR = os.path.join(ROOT, "coffee_dataset", "labels")


def norm_name(s: str) -> str:
    s = s.lower()
    s = s.replace(' ', '_')
    # keep letters, numbers, dash and underscore
    out = []
    for ch in s:
        if ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


def collect_prefixes():
    prefixes = set()
    for d in IM_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                name = f
                # strip extensions repeatedly (e.g. .png.jpg, .npy)
                while True:
                    base, ext = os.path.splitext(name)
                    if not ext:
                        break
                    name = base
                if not name:
                    continue
                prefix = name.split('_', 1)[0]
                if prefix:
                    prefixes.add(norm_name(prefix))
    return sorted(p for p in prefixes if p)


def load_data_yaml(path):
    with open(path, 'r', encoding='utf8') as f:
        return yaml.safe_load(f)


def write_data_yaml(path, data):
    with open(path, 'w', encoding='utf8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def backup(src, prefix):
    ts = time.strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(src):
        return None
    dest = os.path.abspath(f"{src}.{prefix}.{ts}.bak")
    if os.path.isdir(src):
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


def remap_labels(mapping):
    changed = 0
    files_changed = 0
    for subset in ("train", "val"):
        label_dir = os.path.join(LABEL_DIR, subset)
        if not os.path.isdir(label_dir):
            continue
        for root, _, files in os.walk(label_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                # infer prefix from filename (image name)
                base = fname
                while True:
                    base_noext, ext = os.path.splitext(base)
                    if not ext:
                        break
                    base = base_noext
                prefix = base.split('_', 1)[0]
                prefix_n = norm_name(prefix)
                if prefix_n not in mapping:
                    continue
                new_id = mapping[prefix_n]
                # read label file
                with open(fpath, 'r', encoding='utf8') as f:
                    lines = [l.rstrip('\n') for l in f if l.strip()]
                updated = False
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    old = parts[0]
                    if old != str(new_id):
                        parts[0] = str(new_id)
                        updated = True
                        changed += 1
                    new_lines.append(' '.join(parts))
                if updated:
                    files_changed += 1
                    with open(fpath, 'w', encoding='utf8') as f:
                        f.write('\n'.join(new_lines) + ("\n" if new_lines else ""))
    return files_changed, changed


def main():
    print('Collecting filename prefixes...')
    prefixes = collect_prefixes()
    print(f'Found {len(prefixes)} unique prefixes (sample): {prefixes[:20]}')

    if not os.path.exists(DATA_YAML):
        print('ERROR: data.yaml not found at', DATA_YAML)
        return 1

    data = load_data_yaml(DATA_YAML) or {}
    existing = []
    if 'names' in data and isinstance(data['names'], dict):
        # convert dict to list preserving numeric order
        max_k = max(int(k) for k in data['names'].keys()) if data['names'] else -1
        existing = [data['names'].get(str(i)) for i in range(max_k + 1)]
        existing = [n for n in existing if n is not None]
    elif 'names' in data and isinstance(data['names'], list):
        existing = data['names']

    print('Existing classes:', existing)

    new_names = list(existing)
    for p in prefixes:
        if p in new_names:
            continue
        new_names.append(p)

    # prepare new data.yaml structure (use mapping format)
    new_names_dict = {i: name for i, name in enumerate(new_names)}
    new_data = dict(data)
    new_data['names'] = new_names_dict

    # backups
    print('Backing up data.yaml...')
    bak1 = backup(DATA_YAML, 'data_yaml')
    print('Backup of data.yaml:', bak1)
    print('Backing up labels folder...')
    bak2 = backup(LABEL_DIR, 'labels')
    print('Backup of labels:', bak2)

    # write new data.yaml
    print('Writing updated data.yaml with', len(new_names), 'classes...')
    write_data_yaml(DATA_YAML, new_data)

    # build mapping prefix -> id
    mapping = {name: idx for idx, name in enumerate(new_names)}

    print('Remapping label files based on filename prefix...')
    files_changed, total_changes = remap_labels(mapping)
    print('Remap summary: files changed =', files_changed, 'total label tokens changed =', total_changes)

    print('Done. Please inspect backups and run sanitize_labels.py if needed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

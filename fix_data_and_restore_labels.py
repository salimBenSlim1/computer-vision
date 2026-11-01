#!/usr/bin/env python3
"""Restore labels from the previous backup and rebuild `data.yaml` merging original classes with filename prefixes.

This script will:
- Find the latest `data.yaml.*.bak` and `coffee_dataset/labels.*.bak` created by the previous step.
- Restore the labels directory from the labels backup.
- Build a merged class list: original classes (from the data.yaml backup) first, then filename prefixes.
- Remap label files so the class id matches the filename-prefix-derived id.
"""
import os
import glob
import shutil
import yaml
import time

ROOT = os.path.abspath(os.path.dirname(__file__))
LABEL_DIR = os.path.join(ROOT, 'coffee_dataset', 'labels')
IM_DIRS = [os.path.join(ROOT, 'coffee_dataset', 'images', d) for d in ('train', 'val')]
DATA_YAML = os.path.join(ROOT, 'data.yaml')


def find_latest(pattern):
    files = glob.glob(os.path.join(ROOT, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def norm_name(s: str) -> str:
    s = s.lower()
    s = s.replace(' ', '_')
    out = []
    for ch in s:
        if ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_')


def collect_prefixes():
    prefixes = []
    seen = set()
    for d in IM_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                name = f
                while True:
                    base, ext = os.path.splitext(name)
                    if not ext:
                        break
                    name = base
                if not name:
                    continue
                prefix = name.split('_', 1)[0]
                p = norm_name(prefix)
                if p and p not in seen:
                    prefixes.append(p)
                    seen.add(p)
    return prefixes


def load_yaml(path):
    with open(path, 'r', encoding='utf8') as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, 'w', encoding='utf8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def restore_labels(backup_dir):
    if not os.path.isdir(backup_dir):
        print('Labels backup not found:', backup_dir)
        return False
    # remove current labels and restore
    if os.path.isdir(LABEL_DIR):
        shutil.rmtree(LABEL_DIR)
    shutil.copytree(backup_dir, LABEL_DIR)
    return True


def remap_labels(mapping):
    files_changed = 0
    total_changed = 0
    for subset in ('train', 'val'):
        d = os.path.join(LABEL_DIR, subset)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                fpath = os.path.join(root, fname)
                base = fname
                while True:
                    base_noext, ext = os.path.splitext(base)
                    if not ext:
                        break
                    base = base_noext
                prefix = base.split('_', 1)[0]
                p = norm_name(prefix)
                if p not in mapping:
                    continue
                new_id = mapping[p]
                with open(fpath, 'r', encoding='utf8') as f:
                    lines = [l.rstrip('\n') for l in f if l.strip()]
                updated = False
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] != str(new_id):
                        parts[0] = str(new_id)
                        updated = True
                        total_changed += 1
                    new_lines.append(' '.join(parts))
                if updated:
                    files_changed += 1
                    with open(fpath, 'w', encoding='utf8') as f:
                        f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))
    return files_changed, total_changed


def main():
    # find backups
    data_bak = find_latest('data.yaml.data_yaml.*.bak')
    labels_bak = find_latest('coffee_dataset\\labels.labels.*.bak')
    print('Found data backup:', data_bak)
    print('Found labels backup:', labels_bak)

    if not data_bak:
        print('No data.yaml backup found; aborting')
        return 1
    orig = load_yaml(data_bak)
    orig_names = []
    if 'names' in orig and isinstance(orig['names'], dict):
        max_k = max(int(k) for k in orig['names'].keys()) if orig['names'] else -1
        orig_names = [orig['names'].get(str(i)) for i in range(max_k + 1)]
        orig_names = [n for n in orig_names if n is not None]
    elif 'names' in orig and isinstance(orig['names'], list):
        orig_names = orig['names']

    print('Original names from backup:', orig_names)

    prefixes = collect_prefixes()
    print('Collected prefixes:', prefixes)

    # build merged names: original first
    merged = list(orig_names)
    for p in prefixes:
        if p not in merged:
            merged.append(p)

    # restore labels
    if labels_bak:
        print('Restoring labels from backup...')
        restore_labels(labels_bak)
    else:
        print('No labels backup found, will remap in-place')

    # write merged data.yaml
    new_names_dict = {i: name for i, name in enumerate(merged)}
    data = load_yaml(DATA_YAML) or {}
    data['names'] = new_names_dict
    print('Writing merged data.yaml with', len(merged), 'classes')
    write_yaml(DATA_YAML, data)

    # build mapping prefix->id (map existing names too)
    mapping = {name: idx for idx, name in enumerate(merged)}

    print('Remapping label files to new indices...')
    files_changed, total_changed = remap_labels(mapping)
    print('Remap result: files_changed=', files_changed, 'total_changed=', total_changed)

    print('All done. Please inspect `data.yaml` and the labels folder. Backups retained.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

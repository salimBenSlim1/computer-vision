from pathlib import Path
import sys

def find_class_id(name: str, data_yaml='data.yaml'):
    p = Path(data_yaml)
    if not p.exists():
        return None
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            s = ln.strip()
            if not s:
                continue
            if ':' not in s:
                continue
            key, val = s.split(':', 1)
            try:
                idx = int(key.strip())
            except Exception:
                continue
            nm = val.strip().strip('"').strip("'")
            if nm == name:
                return idx
    except Exception:
        return None
    return None

def remap(old_id: int, new_id: int, labels_root='coffee_dataset/labels'):
    root = Path(labels_root)
    if not root.exists():
        print('Labels folder not found:', labels_root)
        return 1

    changed_files = 0
    total_replaced = 0
    for subset in ('train', 'val'):
        folder = root / subset
        if not folder.exists():
            continue
        for f in folder.glob('*.txt'):
            txt = f.read_text(encoding='utf-8')
            lines = txt.splitlines()
            new_lines = []
            replaced = 0
            for ln in lines:
                if not ln.strip():
                    continue
                parts = ln.split()
                try:
                    cls = int(parts[0])
                except Exception:
                    new_lines.append(ln)
                    continue
                if cls == old_id:
                    parts[0] = str(new_id)
                    replaced += 1
                    total_replaced += 1
                new_lines.append(' '.join(parts))
            if replaced > 0:
                changed_files += 1
                f.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
                print(f'Remapped {replaced} lines in {f} (old={old_id} -> new={new_id})')

    print(f'Done. Files changed: {changed_files}. Total replacements: {total_replaced}')
    return 0


if __name__ == '__main__':
    # remap COCO cup id (41) to the dataset id for 'cup'
    target = find_class_id('cup')
    if target is None:
        print('Could not find class id for "cup" in data.yaml')
        sys.exit(1)
    sys.exit(remap(41, target))
